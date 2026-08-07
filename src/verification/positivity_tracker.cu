#include "verification/positivity_tracker.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/state.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::verification {
namespace {

constexpr int kBlockSize = 256;
constexpr int kNoCell = std::numeric_limits<int>::max();

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__device__ double atomic_min_double(double* address, const double val) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (val < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull,
                    assumed,
                    static_cast<unsigned long long>(__double_as_longlong(val)));
    if (assumed == old) {
      break;
    }
  }
  return __longlong_as_double(static_cast<long long>(old));
}

__device__ __forceinline__ double positivity_scan_value(const double value) {
  return isfinite(value) ? value : -INFINITY;
}

__device__ __forceinline__ double pressure_value(
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const int c) {
  if (Pe == nullptr && Pi == nullptr) {
    return INFINITY;
  }
  const double pe = Pe != nullptr ? Pe[c] : 0.0;
  const double pi = Pi != nullptr ? Pi[c] : 0.0;
  if (!isfinite(pe) || !isfinite(pi)) {
    return -INFINITY;
  }
  return pe + pi;
}

__device__ __forceinline__ double radiation_energy_value(
    const double* __restrict__ rad_E,
    const int c,
    const int n_groups) {
  if (rad_E == nullptr || n_groups <= 0) {
    return INFINITY;
  }
  double out = INFINITY;
  const int base = c * n_groups;
  for (int g = 0; g < n_groups; ++g) {
    out = fmin(out, positivity_scan_value(rad_E[base + g]));
  }
  return out;
}

__device__ __forceinline__ double kappa_value(
    const double* __restrict__ kappa,
    const int c,
    const int n_groups) {
  if (kappa == nullptr || n_groups <= 0) {
    return INFINITY;
  }
  double out = -INFINITY;
  const int base = c * n_groups;
  for (int g = 0; g < n_groups; ++g) {
    const double value = kappa[base + g];
    if (!isfinite(value)) {
      return -INFINITY;
    }
    out = fmax(out, value);
  }
  return out;
}

__device__ __forceinline__ double field_value(
    const int field,
    const int c,
    const int n_groups,
    const double* __restrict__ rho,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ rad_E,
    const double* __restrict__ kappa) {
  switch (field) {
    case 0:
      return rho != nullptr ? positivity_scan_value(rho[c]) : INFINITY;
    case 1:
      return pressure_value(Pe, Pi, c);
    case 2:
      return Te != nullptr ? positivity_scan_value(Te[c]) : INFINITY;
    case 3:
      return Ti != nullptr ? positivity_scan_value(Ti[c]) : INFINITY;
    case 4:
      return radiation_energy_value(rad_E, c, n_groups);
    case 5:
      return kappa_value(kappa, c, n_groups);
    default:
      break;
  }
  return INFINITY;
}

__global__ void positivity_min_kernel(
    double* __restrict__ minima,
    const double* __restrict__ rho,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ rad_E,
    const double* __restrict__ kappa,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  for (int field = 0; field < static_cast<int>(kPositivityFieldCount);
       ++field) {
    atomic_min_double(minima + field,
                      field_value(field,
                                  c,
                                  n_groups,
                                  rho,
                                  Pe,
                                  Pi,
                                  Te,
                                  Ti,
                                  rad_E,
                                  kappa));
  }
}

__global__ void positivity_min_cell_kernel(
    const double* __restrict__ minima,
    int* __restrict__ min_cells,
    const double* __restrict__ rho,
    const double* __restrict__ Pe,
    const double* __restrict__ Pi,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const double* __restrict__ rad_E,
    const double* __restrict__ kappa,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells,
    const int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  for (int field = 0; field < static_cast<int>(kPositivityFieldCount);
       ++field) {
    const double min_value = minima[field];
    if (!(min_value < 0.0)) {
      continue;
    }
    const double value = field_value(field,
                                     c,
                                     n_groups,
                                     rho,
                                     Pe,
                                     Pi,
                                     Te,
                                     Ti,
                                     rad_E,
                                     kappa);
    if (value == min_value || (!isfinite(value) && !isfinite(min_value))) {
      atomicMin(min_cells + field, c);
    }
  }
}

template <typename Field>
std::vector<double> copy_field(const Field& field) {
  std::vector<double> host(field.size(), 0.0);
  if (!host.empty()) {
    field.copy_to_host(host.data());
  }
  return host;
}

const double* select_kappa_field(const core::State& state,
                                 const std::size_t expected) {
  if (expected == 0U) {
    return nullptr;
  }
  if (state.fld_sigma_R.size() == expected) {
    return state.fld_sigma_R.data();
  }
  if (state.sn_sigma_s.size() == expected) {
    return state.sn_sigma_s.data();
  }
  if (state.fld_sigma_a.size() == expected) {
    return state.fld_sigma_a.data();
  }
  if (state.sn_sigma_a.size() == expected) {
    return state.sn_sigma_a.data();
  }
  if (state.fld_sigma_pe.size() == expected) {
    return state.fld_sigma_pe.data();
  }
  if (state.sn_sigma_pe.size() == expected) {
    return state.sn_sigma_pe.data();
  }
  return nullptr;
}

struct HostFields {
  std::vector<double> rho;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> Te;
  std::vector<double> Ti;
  std::vector<double> rad_E;
  std::vector<double> kappa;
};

double host_radiation_energy_value(const HostFields& fields,
                                   const int c,
                                   const int n_groups) {
  if (fields.rad_E.empty() || n_groups <= 0) {
    return 0.0;
  }
  double out = std::numeric_limits<double>::infinity();
  const std::size_t base =
      static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
  for (int g = 0; g < n_groups; ++g) {
    const double value = fields.rad_E[base + static_cast<std::size_t>(g)];
    out = std::min(out, value);
  }
  return std::isfinite(out) ? out : 0.0;
}

double host_kappa_value(const HostFields& fields,
                        const int c,
                        const int n_groups) {
  if (fields.kappa.empty() || n_groups <= 0) {
    return 0.0;
  }
  double out = -std::numeric_limits<double>::infinity();
  const std::size_t base =
      static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
  for (int g = 0; g < n_groups; ++g) {
    out = std::max(out, fields.kappa[base + static_cast<std::size_t>(g)]);
  }
  return std::isfinite(out) ? out : 0.0;
}

double host_violation_value(const PositivityField field,
                            const HostFields& fields,
                            const int c,
                            const int n_groups) {
  const std::size_t idx = static_cast<std::size_t>(c);
  switch (field) {
    case PositivityField::Rho:
      return idx < fields.rho.size() ? fields.rho[idx] : 0.0;
    case PositivityField::Pressure: {
      const double pe = idx < fields.Pe.size() ? fields.Pe[idx] : 0.0;
      const double pi = idx < fields.Pi.size() ? fields.Pi[idx] : 0.0;
      return pe + pi;
    }
    case PositivityField::ElectronTemperature:
      return idx < fields.Te.size() ? fields.Te[idx] : 0.0;
    case PositivityField::IonTemperature:
      return idx < fields.Ti.size() ? fields.Ti[idx] : 0.0;
    case PositivityField::RadiationEnergy:
      return host_radiation_energy_value(fields, c, n_groups);
    case PositivityField::Kappa:
      return host_kappa_value(fields, c, n_groups);
  }
  return 0.0;
}

PositivityCellState make_cell_state(const HostFields& fields,
                                    const int c,
                                    const int n_cells,
                                    const int nz,
                                    const int n_groups) {
  PositivityCellState out;
  if (c < 0 || c >= n_cells) {
    return out;
  }
  const std::size_t idx = static_cast<std::size_t>(c);
  out.valid = true;
  out.cell_id = c;
  out.cell_i = nz > 0 ? c / nz : c;
  out.cell_j = nz > 0 ? c - out.cell_i * nz : 0;
  out.rho = idx < fields.rho.size() ? fields.rho[idx] : 0.0;
  const double pe = idx < fields.Pe.size() ? fields.Pe[idx] : 0.0;
  const double pi = idx < fields.Pi.size() ? fields.Pi[idx] : 0.0;
  out.p = pe + pi;
  out.T_e = idx < fields.Te.size() ? fields.Te[idx] : 0.0;
  out.T_i = idx < fields.Ti.size() ? fields.Ti[idx] : 0.0;
  out.E_r = host_radiation_energy_value(fields, c, n_groups);
  out.kappa = host_kappa_value(fields, c, n_groups);
  return out;
}

HostFields gather_host_fields(const core::State& state,
                              const int n_cells,
                              const int n_groups,
                              const bool has_rad_E,
                              const bool has_kappa) {
  HostFields fields;
  fields.rho = copy_field(state.rho);
  fields.Pe = copy_field(state.Pe);
  fields.Pi = copy_field(state.Pi);
  fields.Te = copy_field(state.Te);
  fields.Ti = copy_field(state.Ti);
  if (has_rad_E) {
    fields.rad_E = copy_field(state.rad_E);
  }
  if (has_kappa) {
    if (state.fld_sigma_R.size() ==
        static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups)) {
      fields.kappa = copy_field(state.fld_sigma_R);
    } else if (state.sn_sigma_s.size() ==
               static_cast<std::size_t>(n_cells) *
                   static_cast<std::size_t>(n_groups)) {
      fields.kappa = copy_field(state.sn_sigma_s);
    } else if (state.fld_sigma_a.size() ==
               static_cast<std::size_t>(n_cells) *
                   static_cast<std::size_t>(n_groups)) {
      fields.kappa = copy_field(state.fld_sigma_a);
    } else if (state.sn_sigma_a.size() ==
               static_cast<std::size_t>(n_cells) *
                   static_cast<std::size_t>(n_groups)) {
      fields.kappa = copy_field(state.sn_sigma_a);
    } else if (state.fld_sigma_pe.size() ==
               static_cast<std::size_t>(n_cells) *
                   static_cast<std::size_t>(n_groups)) {
      fields.kappa = copy_field(state.fld_sigma_pe);
    } else if (state.sn_sigma_pe.size() ==
               static_cast<std::size_t>(n_cells) *
                   static_cast<std::size_t>(n_groups)) {
      fields.kappa = copy_field(state.sn_sigma_pe);
    }
  }
  return fields;
}

PositivityViolation make_violation(const PositivityField field,
                                   const int cell,
                                   const core::State& state,
                                   const int n_cells,
                                   const int n_groups,
                                   const bool has_rad_E,
                                   const bool has_kappa) {
  const int nz = state.mesh.topo.nz > 0 ? state.mesh.topo.nz : 1;
  const int nr = (nz > 0 && n_cells % nz == 0) ? n_cells / nz : n_cells;
  const HostFields fields =
      gather_host_fields(state, n_cells, n_groups, has_rad_E, has_kappa);
  PositivityViolation out;
  out.valid = true;
  out.field = field;
  out.cell_id = cell;
  out.cell_i = nz > 0 ? cell / nz : cell;
  out.cell_j = nz > 0 ? cell - out.cell_i * nz : 0;
  out.value = host_violation_value(field, fields, cell, n_groups);
  out.cell_state = make_cell_state(fields, cell, n_cells, nz, n_groups);

  const int i = out.cell_i;
  const int j = out.cell_j;
  const std::array<int, 4> neighbor_cells = {
      (i > 0) ? ((i - 1) * nz + j) : -1,
      (i + 1 < nr) ? ((i + 1) * nz + j) : -1,
      (j > 0) ? (i * nz + (j - 1)) : -1,
      (j + 1 < nz) ? (i * nz + (j + 1)) : -1,
  };
  for (std::size_t k = 0; k < neighbor_cells.size(); ++k) {
    out.neighbors[k] =
        make_cell_state(fields, neighbor_cells[k], n_cells, nz, n_groups);
  }
  return out;
}

void append_number(std::ostream& os, const double value) {
  if (std::isfinite(value)) {
    os << std::scientific << std::setprecision(17) << value;
  } else {
    os << "null";
  }
}

void append_cell_state(std::ostream& os, const PositivityCellState& state) {
  if (!state.valid) {
    os << "invalid";
    return;
  }
  os << "cell=" << state.cell_id
     << "(i=" << state.cell_i
     << ",j=" << state.cell_j
     << ",rho=";
  append_number(os, state.rho);
  os << ",p=";
  append_number(os, state.p);
  os << ",T_e=";
  append_number(os, state.T_e);
  os << ",T_i=";
  append_number(os, state.T_i);
  os << ",E_r=";
  append_number(os, state.E_r);
  os << ",kappa=";
  append_number(os, state.kappa);
  os << ")";
}

void log_violation(const PositivityScanResult& result,
                   const PositivityViolation& violation) {
  std::ostringstream oss;
  oss << "[positivity-audit] step=" << result.step
      << " t=";
  append_number(oss, result.time);
  oss << " field=" << positivity_field_name(violation.field)
      << " cell_id=" << violation.cell_id
      << " i=" << violation.cell_i
      << " j=" << violation.cell_j
      << " value=";
  append_number(oss, violation.value);
  oss << " center=";
  append_cell_state(oss, violation.cell_state);
  oss << " neighbors=[";
  for (std::size_t k = 0; k < violation.neighbors.size(); ++k) {
    if (k != 0U) {
      oss << ";";
    }
    append_cell_state(oss, violation.neighbors[k]);
  }
  oss << "]";
  core::log_warning(oss.str());
}

}  // namespace

const char* positivity_field_name(const PositivityField field) {
  const auto idx = static_cast<std::size_t>(field);
  if (idx >= kPositivityFieldNames.size()) {
    return "unknown";
  }
  return kPositivityFieldNames[idx];
}

const char* positivity_min_dataset_name(const PositivityField field) {
  const auto idx = static_cast<std::size_t>(field);
  if (idx >= kPositivityMinDatasetNames.size()) {
    return "unknown_min";
  }
  return kPositivityMinDatasetNames[idx];
}

void PositivityTracker::initialize(const ProductionAuditConfig& cfg,
                                   const bool emit_logs) {
  *this = PositivityTracker{};
  emit_logs_ = emit_logs;
  if (!cfg.enabled || !cfg.positivity.enabled) {
    return;
  }
  enabled_ = true;
  fatal_on_neg_ = cfg.positivity.fatal_on_neg;
  tier_ = cfg.tier;
  audit_status_ = "RUNNING";
}

PositivityScanResult PositivityTracker::scan(
    const core::State& state,
    const int step,
    const double time,
    const int n_groups_input,
    const parallel::Reduction* reduction) {
  PositivityScanResult result;
  result.step = step;
  result.time = time;
  if (!enabled_) {
    return result;
  }
  result.enabled = true;

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_groups = std::max(n_groups_input, 1);
  if (n_cells <= 0) {
    return result;
  }
  TENRYU_ASSERT(state.hydro_active.empty() ||
                    static_cast<int>(state.hydro_active.size()) == n_cells,
                "positivity tracker requires hydro_active size == rho size");
  TENRYU_ASSERT(state.Pe.empty() ||
                    static_cast<int>(state.Pe.size()) == n_cells,
                "positivity tracker requires Pe size == rho size");
  TENRYU_ASSERT(state.Pi.empty() ||
                    static_cast<int>(state.Pi.size()) == n_cells,
                "positivity tracker requires Pi size == rho size");
  TENRYU_ASSERT(state.Te.empty() ||
                    static_cast<int>(state.Te.size()) == n_cells,
                "positivity tracker requires Te size == rho size");
  TENRYU_ASSERT(state.Ti.empty() ||
                    static_cast<int>(state.Ti.size()) == n_cells,
                "positivity tracker requires Ti size == rho size");

  const std::size_t n_cell_groups =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  const bool has_rad_E = state.rad_E.size() == n_cell_groups;
  const double* d_rad_E = has_rad_E ? state.rad_E.data() : nullptr;
  const double* d_kappa = select_kappa_field(state, n_cell_groups);
  const bool has_kappa = d_kappa != nullptr;

  double* d_minimum = nullptr;
  int* d_min_cells = nullptr;
  const std::int8_t* d_active = nullptr;
  constexpr std::size_t kMinBytes = kPositivityFieldCount * sizeof(double);
  constexpr std::size_t kCellBytes = kPositivityFieldCount * sizeof(int);
  auto* pack_base = static_cast<unsigned char*>(core::device_scratch_acquire(
      "positivity_tracker:scan_pack", kMinBytes + kCellBytes));
  d_minimum = reinterpret_cast<double*>(pack_base);
  d_min_cells = reinterpret_cast<int*>(pack_base + kMinBytes);
  ++allocation_count_;
  d_active = state.hydro_active_device_ptr();

  auto* h_pack = static_cast<unsigned char*>(core::host_pinned_scratch_acquire(
      "positivity_tracker:scan_pack_host", kMinBytes + kCellBytes));
  {
    auto* h_min = reinterpret_cast<double*>(h_pack);
    auto* h_cells = reinterpret_cast<int*>(h_pack + kMinBytes);
    const double inf = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < kPositivityFieldCount; ++i) {
      h_min[i] = inf;
      h_cells[i] = kNoCell;
    }
  }
  cuda_check(cudaMemcpy(pack_base, h_pack, kMinBytes + kCellBytes, cudaMemcpyHostToDevice),
             "positivity tracker: init scan pack failed");

  const int blocks = (n_cells + kBlockSize - 1) / kBlockSize;
  positivity_min_kernel<<<blocks, kBlockSize>>>(
      d_minimum,
      state.rho.data(),
      state.Pe.empty() ? nullptr : state.Pe.data(),
      state.Pi.empty() ? nullptr : state.Pi.data(),
      state.Te.empty() ? nullptr : state.Te.data(),
      state.Ti.empty() ? nullptr : state.Ti.data(),
      d_rad_E,
      d_kappa,
      d_active,
      n_cells,
      n_groups);
  ++kernel_launch_count_;
  cuda_check(cudaGetLastError(), "positivity tracker: min kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "positivity tracker: min kernel failed");

  positivity_min_cell_kernel<<<blocks, kBlockSize>>>(
      d_minimum,
      d_min_cells,
      state.rho.data(),
      state.Pe.empty() ? nullptr : state.Pe.data(),
      state.Pi.empty() ? nullptr : state.Pi.data(),
      state.Te.empty() ? nullptr : state.Te.data(),
      state.Ti.empty() ? nullptr : state.Ti.data(),
      d_rad_E,
      d_kappa,
      d_active,
      n_cells,
      n_groups);
  ++kernel_launch_count_;
  cuda_check(cudaGetLastError(), "positivity tracker: min-cell kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "positivity tracker: min-cell kernel failed");

  std::array<int, kPositivityFieldCount> min_cells{};
  cuda_check(cudaMemcpy(h_pack, pack_base, kMinBytes + kCellBytes, cudaMemcpyDeviceToHost),
             "positivity tracker: copy scan pack failed");
  {
    const auto* h_min = reinterpret_cast<const double*>(h_pack);
    const auto* h_cells = reinterpret_cast<const int*>(h_pack + kMinBytes);
    for (std::size_t i = 0; i < kPositivityFieldCount; ++i) {
      result.minimum[i] = h_min[i];
      min_cells[i] = h_cells[i];
    }
  }

  for (std::size_t i = 0; i < kPositivityFieldCount; ++i) {
    if (result.minimum[i] < 0.0) {
      result.has_negative = true;
      result.local_has_negative = true;
      if (min_cells[i] != kNoCell) {
        result.has_violation[i] = true;
        result.violations[i] = make_violation(static_cast<PositivityField>(i),
                                              min_cells[i],
                                              state,
                                              n_cells,
                                              n_groups,
                                              has_rad_E,
                                              has_kappa);
      }
    }
  }

  if (reduction != nullptr) {
    reduction->allreduce_min(result.minimum.data(),
                             static_cast<int>(result.minimum.size()));
  }
  result.has_negative = false;
  for (const double value : result.minimum) {
    if (value < 0.0) {
      result.has_negative = true;
      break;
    }
  }
  return result;
}

PositivityAuditDecision PositivityTracker::record_scan(
    const PositivityScanResult& result) {
  PositivityAuditDecision decision;
  if (!enabled_ || !result.enabled) {
    return decision;
  }
  if (!result.has_negative) {
    audit_status_ = "RUNNING";
    return decision;
  }

  ++negative_event_count_;
  if (emit_logs_) {
    bool logged_local = false;
    for (std::size_t i = 0; i < result.violations.size(); ++i) {
      if (result.has_violation[i]) {
        log_violation(result, result.violations[i]);
        logged_local = true;
      }
    }
    if (!logged_local) {
      core::log_warning("[positivity-audit] negative positivity minimum observed "
                        "on another rank at step=" +
                        std::to_string(result.step));
    }
  }

  if (tier_ == "A" || fatal_on_neg_) {
    decision.fatal = true;
    decision.audit_status = "TIER_A_FAIL_POSITIVITY_VIOLATION";
    decision.reason = "production audit detected a negative positivity field";
    audit_status_ = decision.audit_status;
    return decision;
  }

  audit_status_ = "RUNNING";
  return decision;
}

PositivityAuditDecision PositivityTracker::finalize(const int, const double) {
  PositivityAuditDecision decision;
  if (!enabled_) {
    return decision;
  }
  if (tier_ == "A" && negative_event_count_ > 0) {
    decision.fatal = true;
    decision.audit_status = "TIER_A_FAIL_POSITIVITY_VIOLATION";
    decision.reason = "Tier-A production audit forbids negative positivity fields";
    audit_status_ = decision.audit_status;
    return decision;
  }
  audit_status_ = negative_event_count_ > 0 ? "WARN_POSITIVITY_VIOLATION" : "PASS";
  return decision;
}

static_assert(kPositivityFieldCount == kPositivityFieldNames.size(),
              "positivity field name table size mismatch");
static_assert(kPositivityFieldCount == kPositivityMinDatasetNames.size(),
              "positivity dataset name table size mismatch");

}  // namespace tenryu::verification
