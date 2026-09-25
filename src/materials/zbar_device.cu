#include "materials/zbar_device.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/launch_shape.hpp"
#include "core/nvtx_range.hpp"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/zbar_math.hpp"

namespace tenryu::materials {
namespace {

void cuda_check(const cudaError_t error, const char* message) {
  TENRYU_ASSERT(error == cudaSuccess, message);
}

struct ZbarMaterialDevice {
  double Z = 0.0;
  double A = 1.0;
  bool is_void = false;
  ZbarTableView zbar;

  bool operator==(const ZbarMaterialDevice& other) const {
    return Z == other.Z && A == other.A && is_void == other.is_void &&
           zbar.rho_min == other.zbar.rho_min && zbar.rho_max == other.zbar.rho_max &&
           zbar.T_min == other.zbar.T_min && zbar.T_max == other.zbar.T_max &&
           zbar.table.n_rho == other.zbar.table.n_rho && zbar.table.n_T == other.zbar.table.n_T &&
           zbar.table.log_rho_grid == other.zbar.table.log_rho_grid &&
           zbar.table.log_T_grid == other.zbar.table.log_T_grid &&
           zbar.table.e_table == other.zbar.table.e_table;
  }
};

struct ZbarTableStorage {
  std::shared_ptr<const IonmixZbarTable> source;
  core::DeviceBuffer<double> log_rho;
  core::DeviceBuffer<double> log_T;
  core::DeviceBuffer<double> values;

  ZbarTableView view_for(const std::shared_ptr<const IonmixZbarTable>& table) {
    TENRYU_ASSERT(table != nullptr && !table->rho_grid.empty() && !table->T_grid_eV.empty(),
                  "Device Zbar requires non-empty table grids");
    TENRYU_ASSERT(table->n_rho() <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                      table->n_T() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                  "Device Zbar table dimensions exceed kernel limits");
    TENRYU_ASSERT(table->log_rho_grid.size() == table->n_rho() &&
                      table->log_T_grid.size() == table->n_T() &&
                      table->zbar_table.size() == table->n_rho() * table->n_T(),
                  "Device Zbar table storage size mismatch");
    if (source != table) {
      log_rho.reset(table->n_rho());
      log_T.reset(table->n_T());
      values.reset(table->zbar_table.size());
      log_rho.copy_from_host(table->log_rho_grid);
      log_T.copy_from_host(table->log_T_grid);
      values.copy_from_host(table->zbar_table);
      source = table;
    }
    ZbarTableView view;
    view.table.log_rho_grid = log_rho.data();
    view.table.log_T_grid = log_T.data();
    view.table.e_table = values.data();
    view.table.n_rho = static_cast<int>(table->n_rho());
    view.table.n_T = static_cast<int>(table->n_T());
    view.rho_min = table->rho_grid.front();
    view.rho_max = table->rho_grid.back();
    view.T_min = table->T_grid_eV.front();
    view.T_max = table->T_grid_eV.back();
    return view;
  }
};

constexpr int kZbarWarningRecords = 11;
struct ZbarClampWarning {
  double rho_input;
  double T_input;
  double rho_used;
  double T_used;
};
struct ZbarClampSummary {
  int count = 0;
  ZbarClampWarning records[kZbarWarningRecords];
};

template <bool Tabular>
__global__ void update_zbar_fields_kernel(
    const double* rho, const double* Te, const double* volfrac,
    const std::uint8_t* cell_is_void, const ZbarMaterialDevice* materials,
    const int n_cells, const int n_materials, double* zbar,
    std::uint8_t* clamped_inputs) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) return;
  double weighted = 0.0;
  double frac_sum = 0.0;
  for (int m = 0; m < n_materials; ++m) {
    const auto& mat = materials[m];
    if (mat.is_void) continue;
    const std::size_t index = static_cast<std::size_t>(c) * n_materials + m;
    double value;
    if constexpr (Tabular) {
      const ZbarClampedInput input = zbar_clamp_input(mat.zbar, rho[c], Te[c]);
      value = zbar_tabular_value(mat.zbar, input);
      if (clamped_inputs != nullptr) clamped_inputs[index] = input.clamped ? 1U : 0U;
    } else {
      value = zbar_tf_value(rho[c], Te[c], mat.Z, mat.A);
    }
    if (n_materials == 1) {
      weighted = value;
    } else {
      // Match std::max(raw_fraction, 0.0), including its NaN behavior.
      const double f = reclose_max(volfrac[index], 0.0);
      frac_sum += f;
      weighted = zbar_separate_add_product(weighted, f, value);
    }
  }
  if (n_materials > 1 && frac_sum > 1.0e-30) weighted /= frac_sum;
  // The host evaluates even void cells before zeroing the stored field.
  zbar[c] = cell_is_void[c] != 0U ? 0.0 : weighted;
}

__global__ void collect_zbar_clamp_warnings_kernel(
    const double* rho, const double* Te, const std::uint8_t* clamped_inputs,
    const ZbarMaterialDevice* materials, const int n_cells, const int n_materials,
    ZbarClampSummary* summary) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  summary->count = 0;
  for (int c = 0; c < n_cells; ++c) {
    for (int m = 0; m < n_materials; ++m) {
      if (materials[m].is_void ||
          clamped_inputs[static_cast<std::size_t>(c) * n_materials + m] == 0U) continue;
      const auto used = zbar_clamp_input(materials[m].zbar, rho[c], Te[c]);
      summary->records[summary->count++] = {rho[c], Te[c], used.rho, used.T};
      if (summary->count == kZbarWarningRecords) return;
    }
  }
}

}  // namespace

struct ZbarDeviceContext::Impl {
  std::vector<ZbarTableStorage> tables;
  std::vector<ZbarMaterialDevice> cached_materials;
  core::DeviceBuffer<ZbarMaterialDevice> materials;
  std::vector<std::uint8_t> cached_cell_is_void;
  core::DeviceBuffer<std::uint8_t> cell_is_void;
  core::DeviceBuffer<std::uint8_t> clamped_inputs;
  core::DeviceBuffer<ZbarClampSummary> clamp_summary;
  // Clamp-warning summaries come back asynchronously (pinned buffer and an
  // event) and are reported at a later call, so collecting them costs no
  // host synchronization (2026-09-23; the synchronous read ran on every call
  // for as long as fewer than the capped number of warnings had been
  // printed, i.e. for the whole run when nothing clamped).
  ZbarClampSummary* pinned_summary = nullptr;
  cudaEvent_t summary_ready = nullptr;
  bool summary_pending = false;

  void report_pending(const bool wait) {
    if (!summary_pending) return;
    if (wait) {
      if (cudaEventSynchronize(summary_ready) != cudaSuccess) return;
    } else if (cudaEventQuery(summary_ready) != cudaSuccess) {
      return;  // still in flight (or failed; the next call's check reports it)
    }
    summary_pending = false;
    for (int i = 0; i < pinned_summary->count && i < kZbarWarningRecords; ++i) {
      const auto& warning = pinned_summary->records[i];
      report_zbar_clamped_input(warning.rho_input, warning.T_input,
                                warning.rho_used, warning.T_used);
    }
  }

  ~Impl() {
    try {
      report_pending(true);
    } catch (...) {
    }
    if (summary_ready != nullptr) static_cast<void>(cudaEventDestroy(summary_ready));
    if (pinned_summary != nullptr) static_cast<void>(cudaFreeHost(pinned_summary));
  }
};

ZbarDeviceContext::ZbarDeviceContext() = default;
ZbarDeviceContext::~ZbarDeviceContext() = default;

void update_zbar_fields_device(ZbarDeviceContext& context,
                               core::State& state, const core::Config& cfg) {
  const bool tabular = cfg.materials.zbar.model == "tabular";
  const core::NvtxRange nvtx_range(tabular ? "material.zbar_tabular" : "material.zbar_tf");
  if (!tabular && cfg.materials.zbar.model != "thomas_fermi") return;
  const std::size_t n = state.rho.size();
  const std::size_t n_mat = cfg.materials.materials.size();
  if (n == 0 || n_mat == 0) return;
  TENRYU_ASSERT(state.Te.size() == n && state.zbar.size() == n && state.cell_is_void.size() == n,
                "Device Zbar state size mismatch");
  TENRYU_ASSERT(cfg.materials.first_nonvoid_material_index() >= 0,
                "Device Zbar requires a non-void material");
  TENRYU_ASSERT(n_mat == 1 || state.volFrac.size() == n * n_mat,
                "Device Zbar volume-fraction size mismatch");
  TENRYU_ASSERT(!tabular || cfg.materials.zbar_tables.size() == n_mat,
                "Device Zbar table count mismatch");
  TENRYU_ASSERT(n <= static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
                    n_mat <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "Device Zbar state exceeds kernel limits");
  if (!context.impl) context.impl = std::make_unique<ZbarDeviceContext::Impl>();
  auto& cache = *context.impl;
  if (cache.cached_cell_is_void != state.cell_is_void) {
    cache.cell_is_void.reset(n);
    cache.cell_is_void.copy_from_host(state.cell_is_void);
    cache.cached_cell_is_void = state.cell_is_void;
  }
  if (tabular) cache.tables.resize(n_mat);
  std::vector<ZbarMaterialDevice> material_params(n_mat);
  for (std::size_t m = 0; m < n_mat; ++m) {
    auto& params = material_params[m];
    const auto& mat = cfg.materials.materials[m];
    params.Z = mat.Z;
    params.A = mat.A;
    params.is_void = mat.is_void;
    if (tabular && !mat.is_void) params.zbar = cache.tables[m].view_for(cfg.materials.zbar_tables[m]);
  }
  if (cache.cached_materials != material_params) {
    cache.materials.reset(n_mat);
    cache.materials.copy_from_host(material_params);
    cache.cached_materials = material_params;
  }
  cache.report_pending(false);
  // One summary in flight at a time: while the previous one is still being
  // copied back, this call collects none.
  const bool collect_warnings =
      tabular && !cache.summary_pending && !zbar_clamp_warning_limit_reached();
  if (collect_warnings) {
    cache.clamped_inputs.reset(n * n_mat);
    cache.clamp_summary.reset(1);
    if (cache.pinned_summary == nullptr) {
      void* pinned = nullptr;
      cuda_check(cudaHostAlloc(&pinned, sizeof(ZbarClampSummary), cudaHostAllocDefault),
                 "Device Zbar pinned warning summary allocation failed");
      cache.pinned_summary = static_cast<ZbarClampSummary*>(pinned);
      cuda_check(cudaEventCreateWithFlags(&cache.summary_ready, cudaEventDisableTiming),
                 "Device Zbar warning event creation failed");
    }
  }
  const int cells = static_cast<int>(n);
  const int materials_count = static_cast<int>(n_mat);
  const int threads = core::serial_cell_block_size(cells);
  const int blocks = core::serial_cell_blocks(cells);
  if (tabular) {
    update_zbar_fields_kernel<true><<<blocks, threads>>>(
        state.rho.data(), state.Te.data(), state.volFrac.data(), cache.cell_is_void.data(),
        cache.materials.data(), cells, materials_count, state.zbar.data(),
        collect_warnings ? cache.clamped_inputs.data() : nullptr);
  } else {
    update_zbar_fields_kernel<false><<<blocks, threads>>>(
        state.rho.data(), state.Te.data(), state.volFrac.data(), cache.cell_is_void.data(),
        cache.materials.data(), cells, materials_count, state.zbar.data(), nullptr);
  }
  cuda_check(cudaGetLastError(), "update_zbar_fields_kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "update_zbar_fields_kernel failed");
  if (collect_warnings) {
    collect_zbar_clamp_warnings_kernel<<<1, 1>>>(
        state.rho.data(), state.Te.data(), cache.clamped_inputs.data(),
        cache.materials.data(), cells, materials_count, cache.clamp_summary.data());
    cuda_check(cudaGetLastError(), "collect_zbar_clamp_warnings_kernel launch failed");
    cuda_check(cudaMemcpyAsync(cache.pinned_summary, cache.clamp_summary.data(),
                               sizeof(ZbarClampSummary), cudaMemcpyDeviceToHost, nullptr),
               "Device Zbar warning summary D2H failed");
    cuda_check(cudaEventRecord(cache.summary_ready, nullptr),
               "Device Zbar warning event record failed");
    cache.summary_pending = true;
  }
}

}  // namespace tenryu::materials
