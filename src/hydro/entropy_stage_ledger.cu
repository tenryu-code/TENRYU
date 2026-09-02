#include "hydro/entropy_stage_ledger.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
namespace {

constexpr int kStageCount = 3;
constexpr int kColumnCount = 8;
constexpr int kColumnStride = 192;
constexpr int kShellCellCount = 288 * kColumnStride;
constexpr int kBlockSize = 256;
constexpr std::array<int, kColumnCount> kTrackedColumns = {
    0, 1, 2, 3, 4, 8, 96, 191};

struct EntropyLedgerState {
  std::array<core::DeviceArray<double>, kStageCount> before_k;
  core::DeviceArray<double> step_start_k;
  core::DeviceArray<double> contributions;
  std::array<core::DeviceArray<double>, 3> work_contributions;
  core::DeviceArray<int> skipped;
  std::array<bool, kStageCount> stage_open{};
  std::array<std::array<double, kColumnCount>, kStageCount> stage_sums{};
  std::array<std::array<double, kColumnCount>, 3> work_sums{};
  std::array<double, kColumnCount> whole_step_sums{};
  unsigned long long skipped_count = 0;
  int active_step = -1;
  int steps_since_log = 0;
  std::size_t tracked_cells = 0;
};

int entropy_ledger_cadence() {
  static const int cadence = []() {
    const char* raw = std::getenv("TENRYU_I1B_ENTROPY_LEDGER");
    if (raw == nullptr || raw[0] == '\0') {
      return 0;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || value <= 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
      return 0;
    }
    return static_cast<int>(value);
  }();
  return cadence;
}

EntropyLedgerState& entropy_ledger_state() {
  static EntropyLedgerState state;
  return state;
}

int stage_index(const EntropyLedgerStage stage) {
  const int index = static_cast<int>(stage);
  TENRYU_ASSERT(index >= 0 && index < kStageCount,
                "entropy ledger stage is out of range");
  return index;
}

__device__ int tracked_column_slot(const int column) {
  switch (column) {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 2;
    case 3:
      return 3;
    case 4:
      return 4;
    case 8:
      return 5;
    case 96:
      return 6;
    case 191:
      return 7;
    default:
      return -1;
  }
}

__device__ double entropy_proxy(const double rho,
                                const double ee,
                                const double ei,
                                const double gamma) {
  return (ee + ei) * pow(rho, 1.0 - gamma);
}

__global__ void capture_entropy_proxy_kernel(double* k,
                                             const double* rho,
                                             const double* ee,
                                             const double* ei,
                                             const double gamma,
                                             const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n_cells) {
    k[c] = entropy_proxy(rho[c], ee[c], ei[c], gamma);
  }
}

__global__ void entropy_contribution_kernel(double* contributions,
                                            int* skipped,
                                            const double* before_k,
                                            const double* rho,
                                            const double* mass,
                                            const double* ee,
                                            const double* ei,
                                            const double gamma,
                                            const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  contributions[c] = 0.0;
  skipped[c] = 0;
  if (tracked_column_slot(c % kColumnStride) < 0 || !(rho[c] > 1.0)) {
    return;
  }
  const double after_k = entropy_proxy(rho[c], ee[c], ei[c], gamma);
  if (!(before_k[c] > 0.0) || !(after_k > 0.0) ||
      !isfinite(before_k[c]) || !isfinite(after_k)) {
    skipped[c] = 1;
    return;
  }
  contributions[c] = mass[c] * log(after_k / before_k[c]);
}

__global__ void work_bucket_contribution_kernel(
    double* work_p_contributions,
    double* work_av_contributions,
    double* work_sub_contributions,
    const double* rho,
    const double* work_p,
    const double* work_av,
    const double* work_sub,
    const double dt,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  work_p_contributions[c] = 0.0;
  work_av_contributions[c] = 0.0;
  work_sub_contributions[c] = 0.0;
  if (tracked_column_slot(c % kColumnStride) < 0 || !(rho[c] > 1.0)) {
    return;
  }
  work_p_contributions[c] = work_p[c] * dt;
  work_av_contributions[c] = work_av[c] * dt;
  work_sub_contributions[c] = work_sub[c] * dt;
}

std::size_t tracked_cell_count(const core::State& state) {
  return std::min(state.rho.size(), static_cast<std::size_t>(kShellCellCount));
}

double ledger_gamma(const core::Config& cfg) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "entropy ledger requires at least one material");
  return cfg.materials.materials.front().ideal_gas_gamma;
}

void validate_fields(const core::State& state, const std::size_t n_cells) {
  TENRYU_ASSERT(state.ee.size() >= n_cells && state.ei.size() >= n_cells &&
                    state.mass.size() >= n_cells,
                "entropy ledger requires rho/mass/ee/ei field size agreement");
}

void ensure_scratch(EntropyLedgerState& ledger, const std::size_t n_cells) {
  if (ledger.tracked_cells == n_cells) {
    return;
  }
  ledger.step_start_k.reset(n_cells);
  ledger.contributions.reset(n_cells);
  for (auto& contributions : ledger.work_contributions) {
    contributions.reset(n_cells);
  }
  ledger.skipped.reset(n_cells);
  for (auto& before : ledger.before_k) {
    before.reset(n_cells);
  }
  ledger.tracked_cells = n_cells;
}

void capture_entropy_proxy(core::DeviceArray<double>& destination,
                           const core::State& state,
                           const double gamma,
                           const std::size_t n_cells) {
  if (n_cells == 0) {
    return;
  }
  const int n = static_cast<int>(n_cells);
  const int blocks = (n + kBlockSize - 1) / kBlockSize;
  capture_entropy_proxy_kernel<<<blocks, kBlockSize>>>(
      destination.data(), state.rho.data(), state.ee.data(), state.ei.data(),
      gamma, n);
  CUDA_CHECK(cudaGetLastError());
}

void prepare_step(EntropyLedgerState& ledger,
                  const core::State& state,
                  const core::Config& cfg) {
  if (ledger.active_step == state.step) {
    return;
  }
  const std::size_t n_cells = tracked_cell_count(state);
  validate_fields(state, n_cells);
  ensure_scratch(ledger, n_cells);
  ledger.stage_open.fill(false);
  capture_entropy_proxy(ledger.step_start_k, state, ledger_gamma(cfg), n_cells);
  ledger.active_step = state.step;
}

void accumulate_increment(
    EntropyLedgerState& ledger,
    const core::DeviceArray<double>& before_k,
    const core::State& state,
    const double gamma,
    std::array<double, kColumnCount>& sums,
    const bool count_skips) {
  const std::size_t n_cells = ledger.tracked_cells;
  if (n_cells == 0) {
    return;
  }
  const int n = static_cast<int>(n_cells);
  const int blocks = (n + kBlockSize - 1) / kBlockSize;
  entropy_contribution_kernel<<<blocks, kBlockSize>>>(
      ledger.contributions.data(), ledger.skipped.data(), before_k.data(),
      state.rho.data(), state.mass.data(), state.ee.data(), state.ei.data(),
      gamma, n);
  CUDA_CHECK(cudaGetLastError());

  std::vector<double> host_contributions;
  std::vector<int> host_skipped;
  ledger.contributions.copy_to_host(host_contributions);
  ledger.skipped.copy_to_host(host_skipped);
  for (std::size_t c = 0; c < n_cells; ++c) {
    const int column = static_cast<int>(c % kColumnStride);
    int column_slot = -1;
    for (int j = 0; j < kColumnCount; ++j) {
      if (column == kTrackedColumns[j]) {
        column_slot = j;
        break;
      }
    }
    if (column_slot >= 0) {
      sums[column_slot] += host_contributions[c];
    }
    if (count_skips) {
      ledger.skipped_count +=
          static_cast<unsigned long long>(host_skipped[c] != 0);
    }
  }
}

void accumulate_work_buckets(EntropyLedgerState& ledger,
                             const core::State& state,
                             const double dt) {
  const std::size_t n_cells = ledger.tracked_cells;
  if (n_cells == 0) {
    return;
  }
  TENRYU_ASSERT(state.work_p_per_cell.size() >= n_cells &&
                    state.work_av_per_cell.size() >= n_cells &&
                    state.work_sub_per_cell.size() >= n_cells,
                "entropy ledger requires compatible work buffer size agreement");
  const int n = static_cast<int>(n_cells);
  const int blocks = (n + kBlockSize - 1) / kBlockSize;
  work_bucket_contribution_kernel<<<blocks, kBlockSize>>>(
      ledger.work_contributions[0].data(),
      ledger.work_contributions[1].data(),
      ledger.work_contributions[2].data(), state.rho.data(),
      state.work_p_per_cell.data(), state.work_av_per_cell.data(),
      state.work_sub_per_cell.data(), dt, n);
  CUDA_CHECK(cudaGetLastError());

  std::array<std::vector<double>, 3> host_contributions;
  for (int bucket = 0; bucket < 3; ++bucket) {
    ledger.work_contributions[bucket].copy_to_host(
        host_contributions[bucket]);
  }
  for (std::size_t c = 0; c < n_cells; ++c) {
    const int column = static_cast<int>(c % kColumnStride);
    int column_slot = -1;
    for (int j = 0; j < kColumnCount; ++j) {
      if (column == kTrackedColumns[j]) {
        column_slot = j;
        break;
      }
    }
    if (column_slot >= 0) {
      for (int bucket = 0; bucket < 3; ++bucket) {
        ledger.work_sums[bucket][column_slot] +=
            host_contributions[bucket][c];
      }
    }
  }
}

void reset_window(EntropyLedgerState& ledger) {
  for (auto& stage_sum : ledger.stage_sums) {
    stage_sum.fill(0.0);
  }
  for (auto& work_sum : ledger.work_sums) {
    work_sum.fill(0.0);
  }
  ledger.whole_step_sums.fill(0.0);
  ledger.skipped_count = 0;
  ledger.steps_since_log = 0;
}

void append_columns(std::ostringstream& stream,
                    const std::array<double, kColumnCount>& sums) {
  for (int j = 0; j < kColumnCount; ++j) {
    stream << " j" << kTrackedColumns[j] << "=" << sums[j];
  }
}

}  // namespace

void entropy_ledger_step_begin(core::State& state, const core::Config& cfg) {
  if (entropy_ledger_cadence() <= 0) {
    return;
  }
  prepare_step(entropy_ledger_state(), state, cfg);
}

void entropy_ledger_begin(core::State& state,
                          const core::Config& cfg,
                          const EntropyLedgerStage stage) {
  if (entropy_ledger_cadence() <= 0) {
    return;
  }
  EntropyLedgerState& ledger = entropy_ledger_state();
  const int index = stage_index(stage);
  prepare_step(ledger, state, cfg);
  const double gamma = ledger_gamma(cfg);
  TENRYU_ASSERT(!ledger.stage_open[index],
                "entropy ledger stage begin called while already open");
  capture_entropy_proxy(ledger.before_k[index], state, gamma,
                        ledger.tracked_cells);
  ledger.stage_open[index] = true;
}

void entropy_ledger_end(core::State& state,
                        const core::Config& cfg,
                        const EntropyLedgerStage stage) {
  if (entropy_ledger_cadence() <= 0) {
    return;
  }
  EntropyLedgerState& ledger = entropy_ledger_state();
  const int index = stage_index(stage);
  TENRYU_ASSERT(ledger.stage_open[index],
                "entropy ledger stage end called without begin");
  validate_fields(state, ledger.tracked_cells);
  accumulate_increment(ledger, ledger.before_k[index], state,
                       ledger_gamma(cfg), ledger.stage_sums[index], true);
  ledger.stage_open[index] = false;
  // Rollbacks give ~0; retried s1 pairs accumulate without changing the polar-equator differential.
}

void entropy_ledger_accumulate_work_buckets(core::State& state,
                                            const core::Config& cfg,
                                            const double dt) {
  if (entropy_ledger_cadence() <= 0) {
    return;
  }
  EntropyLedgerState& ledger = entropy_ledger_state();
  prepare_step(ledger, state, cfg);
  accumulate_work_buckets(ledger, state, dt);
}

void entropy_ledger_log(core::State& state, const core::Config& cfg) {
  const int cadence = entropy_ledger_cadence();
  if (cadence <= 0) {
    return;
  }
  EntropyLedgerState& ledger = entropy_ledger_state();
  if (ledger.active_step >= 0) {
    for (const bool open : ledger.stage_open) {
      TENRYU_ASSERT(!open, "entropy ledger log called with an open stage");
    }
    validate_fields(state, ledger.tracked_cells);
    accumulate_increment(ledger, ledger.step_start_k, state,
                         ledger_gamma(cfg), ledger.whole_step_sums, false);
    ledger.active_step = -1;
  }
  ++ledger.steps_since_log;
  if (ledger.steps_since_log < cadence) {
    return;
  }

  std::array<double, kColumnCount> other_sums{};
  for (int j = 0; j < kColumnCount; ++j) {
    other_sums[j] = ledger.whole_step_sums[j];
    for (int stage = 0; stage < kStageCount; ++stage) {
      other_sums[j] -= ledger.stage_sums[stage][j];
    }
  }

  std::ostringstream stream;
  stream << std::scientific << std::setprecision(3)
         << "[entropy-ledger] step=" << state.step << " t=" << state.t
         << " s1(hydro):";
  append_columns(stream, ledger.stage_sums[0]);
  stream << " | s2(rad):";
  append_columns(stream, ledger.stage_sums[1]);
  stream << " | s3(ale):";
  append_columns(stream, ledger.stage_sums[2]);
  stream << " | s4(other):";
  append_columns(stream, other_sums);
  stream << " | w_p:";
  append_columns(stream, ledger.work_sums[0]);
  stream << " | w_av:";
  append_columns(stream, ledger.work_sums[1]);
  stream << " | w_sub:";
  append_columns(stream, ledger.work_sums[2]);
  stream << " | skipped=" << ledger.skipped_count;
  core::log_info(stream.str());
  reset_window(ledger);
}

}  // namespace tenryu::hydro
