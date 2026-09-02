#include "hydro/hydro_2d.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/config_validate.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "coupling/profile_observability.hpp"
#include "diagnostics/diagnostics.hpp"
#include "diagnostics/mesh_deform_attribution.hpp"
#include "diagnostics/mesh_diag_dump.hpp"
#include "hydro/anti_hourglass.cuh"
#include "hydro/ale_identity_diag.hpp"
#include "hydro/ale_driver.cuh"
#include "hydro/ale_remap_2d_rz.hpp"
#include "hydro/artificial_viscosity.hpp"
#include "hydro/axis_projection.hpp"
#include "hydro/braginskii_viscosity.cuh"
#include "hydro/boundary_pressure_force.cuh"
#include "hydro/boundary_2d.hpp"
#include "hydro/cap_energy_audit.hpp"
#include "hydro/carrier_constraint_adapter.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/compatible_av_csw.cuh"
#include "hydro/compatible_av_tensor.cuh"
#include "hydro/compatible_force_work_2d.cuh"
#include "hydro/compatible_subzonal_pressure.cuh"
#include "hydro/conservation_audit.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "hydro/eos_context.hpp"
#include "hydro/evacuated_cell_shadow.hpp"
#include "hydro/hllc_z_flux_2d_rz.hpp"
#include "hydro/hydro_multiblock_topology.cuh"
#include "hydro/mesh_motion_trace.hpp"
#include "hydro/mesh_regime.cuh"
#include "hydro/per_material_eos_accessors.cuh"
#include "hydro/per_material_eos_project.cuh"
#include "hydro/pentagon_geometry.cuh"
#include "hydro/pole_angular_derefine.cuh"
#include "hydro/pole_angular_coarsen.cuh"
#include "hydro/pole_axis_bbsw.hpp"
#include "hydro/pole_axis_diag.hpp"
#include "hydro/pole_axis_constraints.cuh"
#include "hydro/ring7_seam_quotient_remap.cuh"
#include "hydro/rz_area_weighted.cuh"
#include "hydro/rz_corner_mass.cuh"
#include "hydro/state_supply_bc.hpp"
#include "materials/eos_device.cuh"
#include "materials/eos_device_table.cuh"
#include "materials/helmholtz_jet_device.cuh"
#include "materials/helmholtz_spline_device.cuh"
#include "mesh/boundary_kind.hpp"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/path_admissibility.cuh"
#include "mesh/rz_moments.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::detail {

void launch_compute_node_mass_2d_multiblock(
    double* node_mass,
    const double* corner_mass,
    const double* mass_cell,
    const double* x_r,
    const int* reverse_csr_node_offsets,
    const int* reverse_csr_node_cells,
    const int* reverse_csr_node_corners,
    int n_nodes_total,
    int equal_split_node_mass_mode,
    int corner_stride);

}

namespace tenryu::hydro {

bool polar_tier_fo_diag_enabled() {
  static const bool enabled = []() {
    const char* value = std::getenv("TENRYU_POLAR_TIER_FO");
    return value != nullptr && value[0] != '\0' &&
           std::string(value) != "0";
  }();
  return enabled;
}

static bool aw_axis_slave_enabled() {
  static const bool enabled = [] {
    const char* const raw = std::getenv("TENRYU_AW_AXIS_SLAVE");
    return raw == nullptr || std::string(raw) != "0";
  }();
  return enabled;
}

AwAxisSlaveStructuredLines detect_aw_axis_slave_structured_lines(
    const core::State& state,
    const core::Config& cfg) {
  AwAxisSlaveStructuredLines lines;
  if (!aw_axis_slave_enabled() ||
      !cfg.numerics.hydro.aw_compatible_force_work ||
      state.mesh.topo.multiblock.has_value() || state.mesh.topo.nz < 2 ||
      state.mesh.topo.node_flags.size() != state.x_r.size()) {
    return lines;
  }
  const auto axis_line_is_pole = [&](const int j) {
    for (int i = 1; i <= state.mesh.topo.nr; ++i) {
      const int n = state.mesh.topo.node_index(i, j);
      if ((state.mesh.topo.node_flags[static_cast<std::size_t>(n)] &
           mesh::NODE_POLE_AXIS) == 0U) {
        return false;
      }
    }
    return state.mesh.topo.nr > 0;
  };
  lines.theta0_active = axis_line_is_pole(0);
  lines.theta_pi_active = axis_line_is_pole(state.mesh.topo.nz);
  return lines;
}

namespace {

struct BandWorkAudit {
  bool enabled = false;
  bool parsed = false;
  double s_lo = 0.0, s_hi = 0.0, t_lo = 0.0, t_hi = 0.0;
  std::string path;
  std::vector<int> cells;          // capture set (fixed at first active step)
  std::vector<double> s_capture;   // centroid radius at capture time [cm]
  bool captured = false;
  long call_seq = 0;
  FILE* fp = nullptr;
};

BandWorkAudit& band_work_audit() {
  static BandWorkAudit a;
  if (!a.parsed) {
    a.parsed = true;
    const char* raw = std::getenv("TENRYU_HYDRO_BAND_WORK_AUDIT");
    if (raw != nullptr) {
      double v[4] = {0, 0, 0, 0};
      std::string s(raw);
      std::array<std::string, 5> tok;
      std::size_t pos = 0;
      bool ok = true;
      for (int k = 0; k < 4; ++k) {
        const std::size_t next = s.find(':', pos);
        if (next == std::string::npos) { ok = false; break; }
        tok[k] = s.substr(pos, next - pos);
        pos = next + 1;
      }
      if (ok) {
        tok[4] = s.substr(pos);
        char* end = nullptr;
        for (int k = 0; k < 4; ++k) {
          v[k] = std::strtod(tok[k].c_str(), &end);
          if (end == tok[k].c_str()) { ok = false; break; }
        }
      }
      if (ok && !tok[4].empty() && v[1] > v[0] && v[3] > v[2]) {
        a.s_lo = v[0]; a.s_hi = v[1]; a.t_lo = v[2]; a.t_hi = v[3];
        a.path = tok[4];
        a.enabled = true;
      } else {
        core::log_warning(
            "TENRYU_HYDRO_BAND_WORK_AUDIT malformed; expected "
            "s_lo:s_hi:t_lo:t_hi:path — audit disabled");
      }
    }
  }
  return a;
}

constexpr double kEvToErg = 1.6022e-12;
constexpr double kProtonMass = tenryu::core::constants::proton_mass;
constexpr std::uint8_t kNodeCenterFlag = pole_axis::kNodeCenterFlag;
constexpr std::uint8_t kNodePoleAxisFlag = pole_axis::kNodePoleAxisFlag;
// q-cap scope kernel ABI: 0=GLOBAL, 1=TRI_FAN_RADIAL_INDEX,
// 2=CENTROID_R_LE_R_MATCH.
constexpr int kAvQcapScopeGlobal =
    static_cast<int>(core::AvQcapScope::GLOBAL);
constexpr int kAvQcapScopeTriFanRadialIndex =
    static_cast<int>(core::AvQcapScope::TRI_FAN_RADIAL_INDEX);
constexpr int kAvQcapScopeCentroidRLeRMatch =
    static_cast<int>(core::AvQcapScope::CENTROID_R_LE_R_MATCH);
constexpr double kCompatibleEnergyColdPressureFloor = 1.0e-20;

inline int equal_split_node_mass_mode(const std::string& convention) {
  if (convention == "equal_split_all") {
    return 2;
  }
  return convention == "equal_split" ? 1 : 0;
}

struct Hydro2DMaterialParams {
  double A = 1.0;
  double Zbar = 1.0;
  double gamma = 5.0 / 3.0;
};

std::vector<Hydro2DMaterialParams> make_hydro2d_material_params(
    const core::Config& cfg) {
  std::vector<Hydro2DMaterialParams> params(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    Hydro2DMaterialParams p{};
    p.A = std::max(mat.A, 1.0e-12);
    p.gamma = std::max(mat.ideal_gas_gamma, 1.0 + 1.0e-12);
    if (cfg.materials.zbar.model == "fixed" && cfg.materials.zbar.fixed_value >= 0.0) {
      p.Zbar = cfg.materials.zbar.fixed_value;
    } else {
      p.Zbar = (mat.Z > 0.0) ? mat.Z : 1.0;
    }
    if (mat.is_void) {
      p.Zbar = 0.0;
    }
    params[m] = p;
  }
  return params;
}

__host__ __device__ inline bool energy_needs_repair(const double e) {
  return !isfinite(e) || e < 0.0;
}

__host__ __device__ inline double energy_inverse_input(const double e) {
  return isfinite(e) ? e : 0.0;
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void sync_kernel(const char* message) {
  cuda_check(cudaGetLastError(), message);
  if (core::sync_every_kernel_enabled()) {
    cuda_check(cudaDeviceSynchronize(), message);
  }
}

bool origin_trace_enabled() {
  static const bool enabled = [] {
    const char* const raw = std::getenv("TENRYU_I1B_ORIGIN_TRACE");
    return raw != nullptr && raw[0] != '\0' && std::string(raw) != "0";
  }();
  return enabled;
}

int origin_trace_node(const core::State& state) {
  static const int node = [&state] {
    const auto& node_flags = state.mesh.topo.node_flags;
    const auto it = std::find_if(
        node_flags.begin(), node_flags.end(), [](const std::uint8_t flags) {
          return (flags & kNodeCenterFlag) != 0U;
        });
    const int resolved = it == node_flags.end()
                             ? -1
                             : static_cast<int>(it - node_flags.begin());
    std::ostringstream os;
    os << "[origin-trace] node=" << resolved;
    core::log_info(os.str());
    return resolved;
  }();
  return node;
}

double copy_origin_trace_value(const double* const values, const int node) {
  double value = 0.0;
  cuda_check(cudaDeviceSynchronize(),
             "Hydro2D origin trace synchronize failed");
  cuda_check(cudaMemcpy(&value, values + node, sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D origin trace copy failed");
  return value;
}

void emit_origin_trace(const core::State& state,
                       const char* const stage,
                       const double* const values) {
  if (!origin_trace_enabled()) {
    return;
  }
  const int node = origin_trace_node(state);
  if (node < 0 || node >= static_cast<int>(state.v_z.size())) {
    return;
  }
  const double value = copy_origin_trace_value(values, node);
  std::ostringstream os;
  os << std::setprecision(17)
     << "[origin-trace] step=" << state.step
     << " stage=" << stage
     << " v_z=" << value;
  core::log_info(os.str());
}

void emit_origin_trace_step_begin(const core::State& state,
                                  const double* const uz_old) {
  if (!origin_trace_enabled()) {
    return;
  }
  const int node = origin_trace_node(state);
  if (node < 0 || node >= static_cast<int>(state.v_z.size())) {
    return;
  }
  const double old_value = copy_origin_trace_value(uz_old, node);
  const double state_value = copy_origin_trace_value(state.v_z.data(), node);
  std::ostringstream os;
  os << std::setprecision(17)
     << "[origin-trace] step=" << state.step
     << " stage=step_begin uz_old=" << old_value
     << " v_z=" << state_value;
  core::log_info(os.str());
}

void clear_av_max_diagnostic(core::State& state) {
  state.av_max_cell_id = -1;
  state.av_max_i = -1;
  state.av_max_j = -1;
  state.av_q_visc_max = 0.0;
  state.av_rho_at_max = 0.0;
  state.av_cs_at_max = 0.0;
  state.av_delta_u_at_max = 0.0;
}

struct AvMaxDiagOut {
  int cell;
  double q;
  double rho;
  double cs;
  double div_u;
  double dl;
};

// Deterministic single-block argmax: each thread scans a strided stripe
// in ascending index order keeping (value, index) under the total order
// "larger value wins; equal value -> smaller index wins", then the
// shared-memory tree merges with the same comparator. The result is the
// first (lowest-index) maximum over finite, active entries — exactly
// the retired host scan's semantics — for ANY scheduling.
__global__ void av_max_diag_kernel(const double* __restrict__ q,
                                   const double* __restrict__ rho,
                                   const double* __restrict__ cs,
                                   const double* __restrict__ div_u,
                                   const double* __restrict__ dl,
                                   const std::int8_t* __restrict__ active,
                                   AvMaxDiagOut* __restrict__ out,
                                   int n) {
  constexpr int kAvMaxBlock = 256;
  __shared__ double s_val[kAvMaxBlock];
  __shared__ int s_idx[kAvMaxBlock];
  const int tid = threadIdx.x;
  double best_val = 0.0;
  int best_idx = -1;
  for (int c = tid; c < n; c += blockDim.x) {
    if (active != nullptr && active[c] == 0) {
      continue;
    }
    const double v = q[c];
    if (!isfinite(v)) {
      continue;
    }
    if (best_idx < 0 || v > best_val || (v == best_val && c < best_idx)) {
      best_val = v;
      best_idx = c;
    }
  }
  s_val[tid] = best_val;
  s_idx[tid] = best_idx;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      const double other_val = s_val[tid + stride];
      const int other_idx = s_idx[tid + stride];
      if (other_idx >= 0 &&
          (s_idx[tid] < 0 || other_val > s_val[tid] ||
           (other_val == s_val[tid] && other_idx < s_idx[tid]))) {
        s_val[tid] = other_val;
        s_idx[tid] = other_idx;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    const int c = s_idx[0];
    out->cell = c;
    if (c >= 0) {
      out->q = q[c];
      out->rho = rho[c];
      out->cs = cs[c];
      out->div_u = div_u[c];
      out->dl = dl[c];
    }
  }
}

void update_av_max_diagnostic(core::State& state,
                              const core::CellField1D& cs,
                              const core::CellField1D& div_u,
                              const core::CellField1D& dl,
                              const std::int8_t* d_hydro_active) {
  clear_av_max_diagnostic(state);
  const int n_cells = static_cast<int>(state.Qvisc.size());
  if (state.mesh.dim != 2 || n_cells == 0 ||
      state.rho.size() != static_cast<std::size_t>(n_cells) ||
      cs.size() != static_cast<std::size_t>(n_cells) ||
      div_u.size() != static_cast<std::size_t>(n_cells) ||
      dl.size() != static_cast<std::size_t>(n_cells)) {
    return;
  }

  auto* d_out = static_cast<AvMaxDiagOut*>(core::device_scratch_acquire(
      "h2d:av_max_diag_out", sizeof(AvMaxDiagOut)));
  cuda_check(cudaMemsetAsync(d_out, 0, sizeof(AvMaxDiagOut), 0),
             "av_max diagnostic scratch init failed");
  const std::int8_t* active =
      state.hydro_active.empty() ? nullptr : d_hydro_active;
  av_max_diag_kernel<<<1, 256>>>(state.Qvisc.data(),
                                 state.rho.data(),
                                 cs.data(),
                                 div_u.data(),
                                 dl.data(),
                                 active,
                                 d_out,
                                 n_cells);
  cuda_check(cudaGetLastError(), "av_max diagnostic launch failed");
  AvMaxDiagOut host_out{};
  cuda_check(cudaMemcpy(&host_out,
                        d_out,
                        sizeof(host_out),
                        cudaMemcpyDeviceToHost),
             "av_max diagnostic readback failed");
  const int c_max = host_out.cell;
  if (c_max < 0) {
    return;
  }

  const int nz = state.mesh.topo.nz;
  state.av_max_cell_id = c_max;
  const bool split_polar_shell =
      state.mesh.topo.multiblock.has_value() &&
      mesh::mesh_topo_multiblock_polar_shell_block_count(
          *state.mesh.topo.multiblock) > 1;
  if (split_polar_shell) {
    // The raw cell id remains valid; -1 marks structured i/j as unavailable.
    state.av_max_i = -1;
    state.av_max_j = -1;
  } else {
    state.av_max_i = c_max / nz;
    state.av_max_j = c_max - state.av_max_i * nz;
  }
  state.av_q_visc_max = host_out.q;
  state.av_rho_at_max = host_out.rho;
  state.av_cs_at_max = host_out.cs;
  state.av_delta_u_at_max = std::abs(host_out.div_u) * host_out.dl;
}

struct HydroTableViews {
  tenryu::materials::DeviceEOSTableView tab_ion{};
  tenryu::materials::DeviceEOSTableView tab_ele{};
  tenryu::materials::DeviceEOSTableView tab_total{};
  tenryu::materials::HelmholtzSplineDeviceView spline_total{};
  tenryu::materials::HelmholtzJetDeviceView jet_total{};
  std::uint8_t helmholtz_backend_kind = 0u;
};

inline bool use_helmholtz_spline_backend(const HydroTableViews& views) {
  return views.helmholtz_backend_kind == 1u;
}

inline bool use_helmholtz_jet_backend(const HydroTableViews& views) {
  return views.helmholtz_backend_kind == 2u;
}

inline HydroTableViews select_hydro_table_views(const HydroEOSContext* eos_ctx) {
  HydroTableViews views{};
  if (eos_ctx == nullptr || eos_ctx->n_materials <= 0) {
    return views;
  }
  views.tab_ion = eos_ctx->ion_view(0);
  views.tab_ele = eos_ctx->electron_view(0);
  views.tab_total = eos_ctx->total_view(0);
  views.helmholtz_backend_kind = eos_ctx->material_hydro_backend_kind(0);
  if (views.helmholtz_backend_kind == 1u) {
    views.spline_total = eos_ctx->total_helmholtz_view(0);
  } else if (views.helmholtz_backend_kind == 2u) {
    views.jet_total = eos_ctx->total_helmholtz_jet_view(0);
  }
  return views;
}

__device__ inline double atomic_add_double(double* address, const double val) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, val);
#else
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  unsigned long long assumed = 0;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(
                        val + __longlong_as_double(static_cast<long long>(assumed)))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

__device__ inline double t21_axis_cell_margin(
    const double r_outer_j,
    const double z_outer_j,
    const double r_outer_jp1,
    const double z_outer_jp1,
    const double z_axis_j,
    const double z_axis_jp1) {
  const double s = z_axis_jp1 - z_axis_j;
  if (s <= 0.0 || r_outer_j <= 0.0 || r_outer_jp1 <= 0.0) {
    return -1.0;
  }
  const double Q = r_outer_j * (z_outer_jp1 - z_axis_jp1) -
                   r_outer_jp1 * (z_outer_j - z_axis_j);
  return s * fmin(r_outer_j, r_outer_jp1) + fmin(Q, 0.0);
}

__device__ inline void t21_axis_atomic_min(double* address, const double value) {
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *address_as_ull;
  while (value < __longlong_as_double(static_cast<long long>(old))) {
    const unsigned long long assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    static_cast<unsigned long long>(__double_as_longlong(value)));
    if (assumed == old) {
      break;
    }
  }
}

struct MeshDeviceCache {
  core::CellField1D Svec_r;
  core::CellField1D Svec_z;
  core::CellField1D area;

  MeshDeviceCache() = default;
  MeshDeviceCache(const char* r_tag, const char* z_tag, const char* a_tag)
      : Svec_r(r_tag), Svec_z(z_tag), area(a_tag) {}
};

int button_outer_node_ring_or_disabled(const core::State& state) {
  return (state.mesh.button_center && state.mesh.button_center->enabled)
             ? state.mesh.button_center->outer_node_ring
             : -1;
}

struct FloorClampExclusionMasks {
  const std::uint8_t* central_member = nullptr;
  const std::uint8_t* central_passive = nullptr;
  const std::uint8_t* pole_member = nullptr;
};

FloorClampExclusionMasks floor_clamp_exclusion_masks(core::State& state) {
  FloorClampExclusionMasks masks;
  if (central_pseudo_core::active(state)) {
    masks.central_member = state.central_pseudo_core.d_member_mask.data();
    masks.central_passive = state.central_pseudo_core.d_passive_mask.data();
  }
  if (pole_angular_derefine::active(state)) {
    masks.pole_member = state.pole_angular_derefine.d_member_mask.data();
  }
  return masks;
}

__device__ inline bool floor_clamp_excluded(
    const int c,
    const std::uint8_t* __restrict__ central_member_mask,
    const std::uint8_t* __restrict__ central_passive_mask,
    const std::uint8_t* __restrict__ pole_member_mask) {
  return (central_member_mask != nullptr && central_member_mask[c] != 0U) ||
         (central_passive_mask != nullptr && central_passive_mask[c] != 0U) ||
         (pole_member_mask != nullptr && pole_member_mask[c] != 0U);
}

__global__ void compute_density_kernel(double* __restrict__ rho,
                                       const double* __restrict__ mass,
                                       const double* __restrict__ vol,
                                       const std::uint8_t* __restrict__ cell_is_void,
                                       const std::uint8_t* __restrict__ central_member_mask,
                                       const std::uint8_t* __restrict__ central_passive_mask,
                                       const std::uint8_t* __restrict__ pole_member_mask,
                                       const int c_begin,
                                       const int c_end,
                                       const int own_begin,
                                       const int own_end,
                                       const int n_cells,
                                       const double rho_floor,
                                       int* __restrict__ rho_clamp_count) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (floor_clamp_excluded(c, central_member_mask, central_passive_mask,
                           pole_member_mask)) {
    return;
  }
  const double rho_raw = (vol[c] > 0.0) ? (mass[c] / vol[c]) : 0.0;
  const double floor = fmax(rho_floor, 0.0);
  const double rho_new = fmax(rho_raw, floor);
  // Density floor is applied to rho only. Cell mass remains the Lagrangian state,
  // so floor-limited cells can have rho != mass/vol by construction.
  rho[c] = rho_new;
  const bool is_void = (cell_is_void != nullptr &&
                        cell_is_void[c] != static_cast<std::uint8_t>(0));
  // Clamp diagnostics count owned cells only: the compute window is
  // ghost-inclusive under MPI and ghost clamps would double-count the
  // owning rank's tally.
  if (!is_void && rho_clamp_count != nullptr && rho_raw < floor &&
      c >= own_begin && c < own_end) {
    atomicAdd(rho_clamp_count, 1);
  }
}

__global__ void enforce_1t_closure_kernel(double* __restrict__ ee,
                                          double* __restrict__ ei,
                                          double* __restrict__ Te,
                                          double* __restrict__ Ti,
                                          double* __restrict__ Pe,
                                          double* __restrict__ Pi,
                                          const double* __restrict__ rho,
                                          const double* __restrict__ zbar,
                                          const std::int8_t* __restrict__ hydro_active,
                                          const int c_begin,
                                          const int c_end,
                                          const int n_cells,
                                          const double gamma,
                                          const double A,
                                          const double fallback_z,
                                          const double te_floor,
                                          const tenryu::materials::DeviceEOSTableView tab_total,
                                          const tenryu::materials::HelmholtzSplineDeviceView
                                              spline_total,
                                          const tenryu::materials::HelmholtzJetDeviceView
                                              jet_total,
                                          const bool use_helmholtz,
                                          const bool use_helmholtz_jet,
                                          const bool eos_writeback,
                                          double* __restrict__ cv_e_out,
                                          double* __restrict__ cv_i_out) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const double rho_i = fmax(rho[c], 1.0e-30);
  ee[c] += ei[c];
  const bool allow_table_energy_writeback = eos_writeback;
  if (use_helmholtz && spline_total.n_rho > 0 && rho_i >= exp(spline_total.log_rho_min)) {
    const double e_raw = ee[c];
    const double e = energy_inverse_input(e_raw);
    const bool repair_energy = energy_needs_repair(e_raw);
    const double T =
        fmax(tenryu::materials::device_helmholtz_T_from_e(spline_total, rho_i, e), te_floor);
    const double logT = log(fmax(T, 1.0e-30));
    const auto thermo =
        tenryu::materials::device_helmholtz_eval_thermo(spline_total, rho_i, logT);
    if (allow_table_energy_writeback || repair_energy) {
      ee[c] = thermo.energy;
    }
    ei[c] = 0.0;
    Te[c] = T;
    Ti[c] = T;
    Pe[c] = thermo.pressure;
    Pi[c] = 0.0;
    if (cv_e_out != nullptr) {
      cv_e_out[c] = thermo.cv;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[c] = 0.0;
    }
    return;
  }
  if (use_helmholtz_jet && jet_total.n_rho > 0 && rho_i >= exp(jet_total.log_rho_min)) {
    const double e_raw = ee[c];
    const double e = energy_inverse_input(e_raw);
    const bool repair_energy = energy_needs_repair(e_raw);
    const double T =
        fmax(tenryu::materials::device_helmholtz_jet_T_from_e(jet_total, rho_i, e), te_floor);
    const double logT = log(fmax(T, 1.0e-30));
    const auto thermo =
        tenryu::materials::device_helmholtz_jet_eval_thermo(jet_total, rho_i, logT);
    if (allow_table_energy_writeback || repair_energy) {
      ee[c] = thermo.energy;
    }
    ei[c] = 0.0;
    Te[c] = T;
    Ti[c] = T;
    Pe[c] = thermo.pressure;
    Pi[c] = 0.0;
    if (cv_e_out != nullptr) {
      cv_e_out[c] = thermo.cv;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[c] = 0.0;
    }
    return;
  }
  if (tab_total.n_rho > 0 && rho_i >= exp(tab_total.log_rho_min)) {
    const double e_raw = ee[c];
    const double e = energy_inverse_input(e_raw);
    const bool repair_energy = energy_needs_repair(e_raw);
    ei[c] = 0.0;
    const auto rb = tenryu::materials::find_rho_bracket(tab_total, rho_i);
    const double T =
        fmax(tenryu::materials::device_eos_T_from_e_monotone(tab_total, rb, e), te_floor);
    const double logT = log(fmax(T, 1.0e-30));
    if (allow_table_energy_writeback || repair_energy) {
      ee[c] = tenryu::materials::device_eos_energy(tab_total, rb, logT);
    }
    Te[c] = T;
    Ti[c] = T;
    Pe[c] = tenryu::materials::device_eos_pressure(tab_total, rb, logT);
    Pi[c] = 0.0;
    const double cv = fmax(tenryu::materials::device_eos_cv(tab_total, rb, logT), 0.0);
    if (cv_e_out != nullptr) {
      cv_e_out[c] = cv;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[c] = 0.0;
    }
    return;
  }

  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  const double cv_total =
      (1.0 + z) * kEvToErg / (A * kProtonMass * (gamma - 1.0));

  const double e = fmax(ee[c], 0.0);
  ee[c] = e;
  ei[c] = 0.0;

  double te = te_floor;
  if (e > 0.0 && cv_total > 0.0) {
    te = fmax(e / cv_total, te_floor);
  }

  Te[c] = te;
  Ti[c] = te;
  Pi[c] = 0.0;
  Pe[c] = (gamma - 1.0) * rho[c] * e;
  if (cv_e_out != nullptr) {
    cv_e_out[c] = cv_total;
  }
  if (cv_i_out != nullptr) {
    cv_i_out[c] = 0.0;
  }
}

__global__ void enforce_2t_closure_kernel(double* __restrict__ ee,
                                          double* __restrict__ ei,
                                          double* __restrict__ Te,
                                          double* __restrict__ Ti,
                                          double* __restrict__ Pe,
                                          double* __restrict__ Pi,
                                          const double* __restrict__ rho,
                                          const double* __restrict__ zbar,
                                          const std::int8_t* __restrict__ hydro_active,
                                          const int c_begin,
                                          const int c_end,
                                          const int n_cells,
                                          const double gamma,
                                          const double A,
                                          const double fallback_z,
                                          const double te_floor,
                                          const double ti_floor,
                                          const tenryu::materials::DeviceEOSTableView tab_ion,
                                          const tenryu::materials::DeviceEOSTableView tab_ele,
                                          const tenryu::materials::HelmholtzSplineDeviceView
                                              spline_total,
                                          const tenryu::materials::HelmholtzJetDeviceView
                                              jet_total,
                                          const bool use_helmholtz,
                                          const bool use_helmholtz_jet,
                                          const bool eos_writeback,
                                          double* __restrict__ cv_e_out,
                                          double* __restrict__ cv_i_out) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const double rho_i = fmax(rho[c], 1.0e-30);
  const bool allow_table_energy_writeback = eos_writeback;
  if (tab_ion.n_rho > 0 && rho_i >= exp(tab_ion.log_rho_min)) {
    const double e_i_raw = ei[c];
    const double e_e_raw = ee[c];
    const bool repair_i = energy_needs_repair(e_i_raw);
    const bool repair_e = energy_needs_repair(e_e_raw);
    const double e_i = energy_inverse_input(e_i_raw);
    const double e_e = energy_inverse_input(e_e_raw);
    const auto rb_ion = tenryu::materials::find_rho_bracket(tab_ion, rho_i);
    const double ti = fmax(
        tenryu::materials::device_eos_T_from_e_monotone(tab_ion, rb_ion, e_i), ti_floor);
    const double logTi = log(fmax(ti, 1.0e-30));
    const auto rb_ele = tenryu::materials::find_rho_bracket(tab_ele, rho_i);
    const double te = fmax(
        tenryu::materials::device_eos_T_from_e_monotone(tab_ele, rb_ele, e_e), te_floor);
    const double logTe = log(fmax(te, 1.0e-30));
    const double ei_table = tenryu::materials::device_eos_energy(tab_ion, rb_ion, logTi);
    const double ee_table = tenryu::materials::device_eos_energy(tab_ele, rb_ele, logTe);
    if (allow_table_energy_writeback || repair_i) {
      ei[c] = ei_table;
    }
    if (allow_table_energy_writeback || repair_e) {
      ee[c] = ee_table;
    }
    Ti[c] = ti;
    Te[c] = te;
    const double pi_legacy = tenryu::materials::device_eos_pressure(tab_ion, rb_ion, logTi);
    const double pe_legacy = tenryu::materials::device_eos_pressure(tab_ele, rb_ele, logTe);
    if (use_helmholtz && spline_total.n_rho > 0 &&
        rho_i >= exp(spline_total.log_rho_min)) {
      const double e_total = ei[c] + ee[c];
      const double T_total =
          fmax(tenryu::materials::device_helmholtz_T_from_e(spline_total, rho_i, e_total),
               fmax(te_floor, ti_floor));
      const double p_total =
          tenryu::materials::device_helmholtz_eval_thermo(spline_total,
                                                          rho_i,
                                                          log(fmax(T_total, 1.0e-30)))
              .pressure;
      const double p_legacy = pi_legacy + pe_legacy;
      if (fabs(p_legacy) > 1.0e-30 && isfinite(p_legacy)) {
        const double scale = p_total / p_legacy;
        Pi[c] = scale * pi_legacy;
        Pe[c] = scale * pe_legacy;
      } else {
        Pi[c] = 0.0;
        Pe[c] = p_total;
      }
    } else if (use_helmholtz_jet && jet_total.n_rho > 0 &&
               rho_i >= exp(jet_total.log_rho_min)) {
      const double e_total = ei[c] + ee[c];
      const double T_total =
          fmax(tenryu::materials::device_helmholtz_jet_T_from_e(jet_total, rho_i, e_total),
               fmax(te_floor, ti_floor));
      const double p_total =
          tenryu::materials::device_helmholtz_jet_eval_thermo(
              jet_total, rho_i, log(fmax(T_total, 1.0e-30)))
              .pressure;
      const double p_legacy = pi_legacy + pe_legacy;
      if (fabs(p_legacy) > 1.0e-30 && isfinite(p_legacy)) {
        const double scale = p_total / p_legacy;
        Pi[c] = scale * pi_legacy;
        Pe[c] = scale * pe_legacy;
      } else {
        Pi[c] = 0.0;
        Pe[c] = p_total;
      }
    } else {
      Pi[c] = pi_legacy;
      Pe[c] = pe_legacy;
    }
    if (cv_i_out != nullptr) {
      cv_i_out[c] = fmax(tenryu::materials::device_eos_cv(tab_ion, rb_ion, logTi), 0.0);
    }
    if (cv_e_out != nullptr) {
      cv_e_out[c] = fmax(tenryu::materials::device_eos_cv(tab_ele, rb_ele, logTe), 0.0);
    }
    return;
  }

  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  const double cv_i = kEvToErg / (A * kProtonMass * (gamma - 1.0));
  const double cv_e = z * kEvToErg / (A * kProtonMass * (gamma - 1.0));

  double e_i = fmax(ei[c], 0.0);
  double e_e = fmax(ee[c], 0.0);
  double ti = ti_floor;
  double te = te_floor;
  if (cv_i > 0.0) {
    ti = fmax(tenryu::materials::eos_temperature_ion(e_i, cv_i), ti_floor);
  } else {
    e_i = 0.0;
  }
  if (cv_e > 0.0) {
    te = fmax(tenryu::materials::eos_temperature_electron(e_e, cv_e), te_floor);
  } else {
    e_e = 0.0;
  }

  ei[c] = e_i;
  ee[c] = e_e;
  Ti[c] = ti;
  Te[c] = te;
  Pi[c] = tenryu::materials::eos_pressure_ion(rho[c], e_i, gamma);
  Pe[c] = tenryu::materials::eos_pressure_electron(rho[c], e_e, gamma);
  if (cv_i_out != nullptr) {
    cv_i_out[c] = cv_i;
  }
  if (cv_e_out != nullptr) {
    cv_e_out[c] = cv_e;
  }
}

__global__ void state_supply_energy_from_temperature_kernel(
    const std::int8_t* __restrict__ state_supply_mask,
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ rho,
    const double* __restrict__ zbar,
    const double* __restrict__ Te,
    const double* __restrict__ Ti,
    const int c_begin,
    const int c_end,
    const int nz,
    const int bottom_active,
    const int top_active,
    const double gamma,
    const double A,
    const double fallback_z,
    const double te_floor,
    const double ti_floor,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const tenryu::materials::DeviceEOSTableView tab_total,
    const bool use_two_temp) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end || state_supply_mask == nullptr || state_supply_mask[c] == 0) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  if ((bottom_active == 0 || j != 0) &&
      (top_active == 0 || j != nz - 1)) {
    return;
  }

  const double rho_c = fmax(rho[c], 1.0e-30);
  const double te = fmax(Te[c], te_floor);
  const double ti = fmax(Ti[c], ti_floor);
  if (use_two_temp) {
    if (tab_ion.n_rho > 0 && rho_c >= exp(tab_ion.log_rho_min) &&
        tab_ele.n_rho > 0 && rho_c >= exp(tab_ele.log_rho_min)) {
      const auto rb_ion = tenryu::materials::find_rho_bracket(tab_ion, rho_c);
      const auto rb_ele = tenryu::materials::find_rho_bracket(tab_ele, rho_c);
      ei[c] = tenryu::materials::device_eos_energy(
          tab_ion, rb_ion, log(fmax(ti, 1.0e-30)));
      ee[c] = tenryu::materials::device_eos_energy(
          tab_ele, rb_ele, log(fmax(te, 1.0e-30)));
      return;
    }
    const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
    const double cv_i = kEvToErg / (A * kProtonMass * fmax(gamma - 1.0, 1.0e-12));
    const double cv_e =
        z * kEvToErg / (A * kProtonMass * fmax(gamma - 1.0, 1.0e-12));
    ei[c] = cv_i * ti;
    ee[c] = cv_e * te;
    return;
  }

  if (tab_total.n_rho > 0 && rho_c >= exp(tab_total.log_rho_min)) {
    const auto rb_total = tenryu::materials::find_rho_bracket(tab_total, rho_c);
    ee[c] = tenryu::materials::device_eos_energy(
        tab_total, rb_total, log(fmax(te, 1.0e-30)));
    ei[c] = 0.0;
    return;
  }
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  const double cv_total =
      (1.0 + z) * kEvToErg /
      (A * kProtonMass * fmax(gamma - 1.0, 1.0e-12));
  ee[c] = cv_total * te;
  ei[c] = 0.0;
}

__global__ void compute_sound_speed_1t_kernel(double* __restrict__ cs,
                                              const double* __restrict__ ee,
                                              const double* __restrict__ Te,
                                              const double* __restrict__ rho,
                                              const tenryu::materials::DeviceEOSTableView tab_total,
                                              const tenryu::materials::HelmholtzSplineDeviceView
                                                  spline_total,
                                              const tenryu::materials::HelmholtzJetDeviceView
                                                  jet_total,
                                              const double* __restrict__ cv_arr,
                                              const std::int8_t* __restrict__ hydro_active,
                                              const int c_begin,
                                              const int c_end,
                                              const int n_cells,
                                              const bool use_helmholtz,
                                              const bool use_helmholtz_jet,
                                              const double gamma) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    cs[c] = 0.0;
    return;
  }

  const double rho_i = fmax(rho[c], 1.0e-30);
  if (use_helmholtz && spline_total.n_rho > 0 && rho_i >= exp(spline_total.log_rho_min)) {
    const double T = (Te != nullptr)
                         ? fmax(Te[c], 1.0e-30)
                         : fmax(tenryu::materials::device_helmholtz_T_from_e(
                                    spline_total, rho_i, ee[c]),
                                1.0e-30);
    cs[c] = tenryu::materials::device_helmholtz_eval_thermo(
                spline_total, rho_i, log(T))
                .sound_speed;
    return;
  }
  if (use_helmholtz_jet && jet_total.n_rho > 0 && rho_i >= exp(jet_total.log_rho_min)) {
    const double T = (Te != nullptr)
                         ? fmax(Te[c], 1.0e-30)
                         : fmax(tenryu::materials::device_helmholtz_jet_T_from_e(
                                    jet_total, rho_i, ee[c]),
                                1.0e-30);
    cs[c] = tenryu::materials::device_helmholtz_jet_eval_thermo(
                jet_total, rho_i, log(T))
                .sound_speed;
    return;
  }
  if (tab_total.n_rho > 0 && rho_i >= exp(tab_total.log_rho_min)) {
    const auto rb = tenryu::materials::find_rho_bracket(tab_total, rho_i);
    const double e = (tab_total.n_rho > 0) ? ee[c] : fmax(ee[c], 0.0);
    const double T =
        fmax(tenryu::materials::device_eos_T_from_e_monotone(tab_total, rb, e), 1.0e-30);
    const double logT = log(T);
    const double cv =
        (cv_arr != nullptr)
            ? fmax(cv_arr[c], 0.0)
            : fmax(tenryu::materials::device_eos_cv(tab_total, rb, logT), 0.0);
    cs[c] = tenryu::materials::device_eos_sound_speed(tab_total, rb, logT, rho_i, cv);
    return;
  }

  const double e = (tab_total.n_rho > 0) ? ee[c] : fmax(ee[c], 0.0);
  cs[c] = sqrt(gamma * (gamma - 1.0) * e);
}

__global__ void compute_sound_speed_2t_kernel(double* __restrict__ cs,
                                              const double* __restrict__ rho,
                                              const double* __restrict__ ee,
                                              const double* __restrict__ ei,
                                              const double* __restrict__ Pe,
                                              const double* __restrict__ Pi,
                                              const double* __restrict__ Te,
                                              const double* __restrict__ Ti,
                                              const tenryu::materials::DeviceEOSTableView tab_ion,
                                              const tenryu::materials::DeviceEOSTableView tab_ele,
                                              const tenryu::materials::HelmholtzSplineDeviceView
                                                  spline_total,
                                              const tenryu::materials::HelmholtzJetDeviceView
                                                  jet_total,
                                              const double* __restrict__ cv_i_arr,
                                              const double* __restrict__ cv_e_arr,
                                              const std::int8_t* __restrict__ hydro_active,
                                              const int c_begin,
                                              const int c_end,
                                              const int n_cells,
                                              const bool use_helmholtz,
                                              const bool use_helmholtz_jet,
                                              const double gamma) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    cs[c] = 0.0;
    return;
  }

  const double rho_i = fmax(rho[c], 1.0e-30);
  if (use_helmholtz && spline_total.n_rho > 0 && rho_i >= exp(spline_total.log_rho_min)) {
    const double e_total = ee[c] + ei[c];
    const double T_total = fmax(
        tenryu::materials::device_helmholtz_T_from_e(spline_total, rho_i, e_total), 1.0e-30);
    cs[c] = tenryu::materials::device_helmholtz_eval_thermo(spline_total,
                                                            rho_i,
                                                            log(T_total))
                .sound_speed;
    return;
  }
  if (use_helmholtz_jet && jet_total.n_rho > 0 && rho_i >= exp(jet_total.log_rho_min)) {
    const double e_total = ee[c] + ei[c];
    const double T_total = fmax(
        tenryu::materials::device_helmholtz_jet_T_from_e(jet_total, rho_i, e_total), 1.0e-30);
    cs[c] = tenryu::materials::device_helmholtz_jet_eval_thermo(jet_total,
                                                                rho_i,
                                                                log(T_total))
                .sound_speed;
    return;
  }
  if (tab_ion.n_rho > 0 && rho_i >= exp(tab_ion.log_rho_min)) {
    const auto rb_ion = tenryu::materials::find_rho_bracket(tab_ion, rho_i);
    const auto rb_ele = tenryu::materials::find_rho_bracket(tab_ele, rho_i);
    const double logTi = log(fmax(Ti[c], 1.0e-30));
    const double logTe = log(fmax(Te[c], 1.0e-30));
    const double cv_i =
        (cv_i_arr != nullptr)
            ? fmax(cv_i_arr[c], 0.0)
            : fmax(tenryu::materials::device_eos_cv(tab_ion, rb_ion, logTi), 0.0);
    const double cv_e =
        (cv_e_arr != nullptr)
            ? fmax(cv_e_arr[c], 0.0)
            : fmax(tenryu::materials::device_eos_cv(tab_ele, rb_ele, logTe), 0.0);
    const double cs_ion =
        tenryu::materials::device_eos_sound_speed(tab_ion, rb_ion, logTi, rho_i, cv_i);
    const double cs_ele =
        tenryu::materials::device_eos_sound_speed(tab_ele, rb_ele, logTe, rho_i, cv_e);
    cs[c] = sqrt(cs_ion * cs_ion + cs_ele * cs_ele);
    return;
  }

  cs[c] = tenryu::materials::eos_sound_speed_2t(gamma, Pe[c], gamma, Pi[c], rho[c]);
}

__global__ void compute_node_activity_2d_kernel(std::uint8_t* __restrict__ node_active,
                                                const std::int8_t* __restrict__ hydro_active,
                                                const int c_begin,
                                                const int c_end,
                                                const int nr,
                                                const int nz) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  node_active[n00] = 1u;
  node_active[n10] = 1u;
  node_active[n11] = 1u;
  node_active[n01] = 1u;
}

__global__ void compute_corner_mass_2d_kernel(double* __restrict__ corner_mass,
                                              const double* __restrict__ mass,
                                              const double* __restrict__ x_r,
                                              const double* __restrict__ x_z,
                                              const std::uint8_t* __restrict__ cell_nverts,
                                              rz::CornerMassFallbackRecorder*
                                                  fallback_recorder,
                                              const int corner_mass_convention,
                                              const bool partition_normalized,
                                              const int c_begin,
                                              const int c_end,
                                              const int nr,
                                              const int nz,
                                              const bool aw_compatible_force_work,
                                              const int button_outer_node_ring,
                                              const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }
  if (button_outer_node_ring >= 1 && c == 0) {
    corner_mass[0] = 0.0;
    corner_mass[1] = 0.0;
    corner_mass[2] = 0.0;
    corner_mass[3] = 0.0;
    return;
  }

  const double m_cell = mass[c];
  if (partition_normalized) {
    const int i = c / nz;
    const int j = c - i * nz;
    const int stride = nz + 1;
    const int n00 = i * stride + j;
    const int n10 = (i + 1) * stride + j;
    const int n11 = (i + 1) * stride + (j + 1);
    const int n01 = i * stride + (j + 1);
    double m_corner[4] = {};
    if (mesh::mesh_topo_cell_active_nverts(cell_nverts, c) == 3) {
      rz::compute_triangle_corner_masses_exact(
          m_cell, x_r[n00], x_z[n00], x_r[n10], x_z[n10], x_r[n11],
          x_z[n11], m_corner);
    } else {
      rz::compute_quad_corner_masses_partitioned_subpolygon(
          m_cell, x_r[n00], x_z[n00], x_r[n10], x_z[n10], x_r[n11],
          x_z[n11], x_r[n01], x_z[n01], m_corner);
    }
    corner_mass[c * corner_stride + 0] = m_corner[0];
    corner_mass[c * corner_stride + 1] = m_corner[1];
    corner_mass[c * corner_stride + 2] = m_corner[2];
    corner_mass[c * corner_stride + 3] = m_corner[3];
    return;
  }

  rz::CornerMassFallbackProbe probe{};
  rz::compute_rz_corner_masses_for_cell(c, nz, m_cell, x_r, x_z, cell_nverts,
                                        aw_compatible_force_work, corner_mass,
                                        corner_stride, &probe,
                                        corner_mass_convention);
  if (probe.fired == 1) {
    rz::record_corner_mass_fallback(
        fallback_recorder,
        probe,
        true,
        c,
        rz::kCornerMassFallbackStageHydroStructured,
        -2);
  }
}

__global__ void compute_corner_mass_2d_multiblock_kernel(
    double* __restrict__ corner_mass,
    const double* __restrict__ mass,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    rz::CornerMassFallbackRecorder* fallback_recorder,
    const int c_begin,
    const int c_end,
    const int n_cells_total,
    const bool partition_normalized,
    const bool polar_tier_equal_planar_area,
    const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const double m_cell = mass[c];
  const int off = cell_node_csr_offsets[c];
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double m_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  if (active_nverts == 3) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    if (polar_tier_equal_planar_area) {
      rz::compute_triangle_corner_masses_equal_planar_area(
          m_cell, x_r[n0], x_z[n0], x_r[n1], x_z[n1], x_r[n2], x_z[n2],
          m_corner);
    } else {
      rz::compute_triangle_corner_masses_exact(m_cell,
                                               x_r[n0],
                                               x_z[n0],
                                               x_r[n1],
                                               x_z[n1],
                                               x_r[n2],
                                               x_z[n2],
                                               m_corner);
    }
  } else if (active_nverts == 4) {
    const int n00 = cell_node_csr_indices[off + 0];
    const int n10 = cell_node_csr_indices[off + 1];
    const int n11 = cell_node_csr_indices[off + 2];
    const int n01 = cell_node_csr_indices[off + 3];
    if (partition_normalized) {
      rz::compute_quad_corner_masses_partitioned_subpolygon(m_cell,
                                                            x_r[n00],
                                                            x_z[n00],
                                                            x_r[n10],
                                                            x_z[n10],
                                                            x_r[n11],
                                                            x_z[n11],
                                                            x_r[n01],
                                                            x_z[n01],
                                                            m_corner);
    } else {
      rz::CornerMassFallbackProbe probe{};
      rz::compute_quad_corner_masses_exact_subpolygon(m_cell,
                                                      x_r[n00],
                                                      x_z[n00],
                                                      x_r[n10],
                                                      x_z[n10],
                                                      x_r[n11],
                                                      x_z[n11],
                                                      x_r[n01],
                                                      x_z[n01],
                                                      m_corner,
                                                      &probe);
      if (probe.fired == 1) {
        rz::record_corner_mass_fallback(
            fallback_recorder,
            probe,
            false,
            c,
            rz::kCornerMassFallbackStageHydroMultiblock,
            -2);
      }
    }
  } else if (active_nverts >= 5 &&
             active_nverts <= mesh::kMeshTopoCellStorageSlotsMaxGeneral) {
    double v_corner[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
    if (active_nverts == 5) {
      PentagonPoint x[5];
      for (int v = 0; v < 5; ++v) {
        const int n = cell_node_csr_indices[off + v];
        x[v] = {x_r[n], x_z[n]};
      }
      pentagon_corner_rz_volumes(x, v_corner);
    } else {
      // Shapes above five vertices keep star-P1; no runtime subzonal mismatch was measured.
      double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
      double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
      for (int v = 0; v < active_nverts; ++v) {
        const int n = cell_node_csr_indices[off + v];
        r[v] = x_r[n];
        z[v] = x_z[n];
      }
      mesh::moments::star_p1_vertex_r_moments(
          r, z, active_nverts, v_corner);
    }
    double W = 0.0;
    bool finite_weights = true;
    double weight_scale = 0.0;
    for (int v = 0; v < active_nverts; ++v) {
      W += v_corner[v];
      finite_weights = finite_weights && rz::finite_double(v_corner[v]);
      weight_scale = fmax(weight_scale, fabs(v_corner[v]));
    }
    // A124(b) fail-loud: the pentagon/star-P1 vertex-moment partition
    // preconditions same-sign per-vertex weights (star-visible polygon in
    // either orientation); finite mixed-sign weights beyond roundoff are a
    // contract violation, not a fallback state (NUMERICS §3.2.4; precedent:
    // csr_compute_pentagon_qk_corner_masses). Uniform-sign negative sets
    // keep the pre-existing W<=0 fallback path unchanged.
    if (finite_weights) {
      constexpr double weight_eps_rel = 1.0e-12;
      const double weight_tol = weight_eps_rel * weight_scale;
      bool any_positive = false;
      bool any_negative = false;
      for (int v = 0; v < active_nverts; ++v) {
        any_positive = any_positive || v_corner[v] > weight_tol;
        any_negative = any_negative || v_corner[v] < -weight_tol;
      }
      if (any_positive && any_negative) {
#ifdef __CUDA_ARCH__
        printf("corner mass: mixed-sign pentagon/star-P1 vertex weights\n");
        __trap();
#else
        ::tenryu::core::tenryu_abort(
            "uniform-sign vertex weights",
            "corner mass: mixed-sign pentagon/star-P1 vertex weights (A124b)",
            __FILE__, __LINE__);
#endif
      }
    }
    if (!(W > 0.0) || !finite_weights) {
      rz::CornerMassFallbackProbe probe{};
      probe.fired = 1;
      probe.vals[0] = v_corner[0];
      probe.vals[1] = v_corner[1];
      probe.vals[2] = v_corner[2];
      probe.vals[3] = v_corner[3];
      probe.vals[4] = W;
      const double m_equal = m_cell / static_cast<double>(active_nverts);
      for (int v = 0; v < active_nverts; ++v) {
        m_corner[v] = m_equal;
      }
      if (probe.fired == 1) {
        rz::record_corner_mass_fallback(
            fallback_recorder,
            probe,
            false,
            c,
            rz::kCornerMassFallbackStageHydroMultiblock,
            -3);
      }
    } else {
      for (int v = 0; v < active_nverts; ++v) {
        m_corner[v] = m_cell * (v_corner[v] / W);
      }
    }
  } else {
#ifdef __CUDA_ARCH__
    __trap();
#else
    ::tenryu::core::tenryu_abort(
        "active_nverts <= mesh::kMeshTopoCellStorageSlotsMaxGeneral",
        "corner mass: cell valence exceeds general slot capacity",
        __FILE__, __LINE__);
#endif
  }
  corner_mass[c * corner_stride + 0] = m_corner[0];
  corner_mass[c * corner_stride + 1] = m_corner[1];
  corner_mass[c * corner_stride + 2] = m_corner[2];
  corner_mass[c * corner_stride + 3] = m_corner[3];
  if (active_nverts >= 5) {
    for (int v = 4; v < active_nverts; ++v) {
      corner_mass[c * corner_stride + v] = m_corner[v];
    }
  }
  for (int v = (active_nverts > 4) ? active_nverts : 4; v < corner_stride; ++v) {
    corner_mass[c * corner_stride + v] = 0.0;
  }
}

__global__ void compute_node_mass_2d_kernel(double* __restrict__ node_mass,
                                            const double* __restrict__ corner_mass,
                                            const double* __restrict__ mass_cell,
                                            const double* __restrict__ x_r,
                                            const double* __restrict__ x_z,
                                            const std::uint8_t* __restrict__ cell_nverts,
                                            const std::int8_t* __restrict__ hydro_active,
                                            const int c_begin,
                                            const int c_end,
                                            const int nr,
                                            const int nz,
                                            const int button_outer_node_ring,
                                            const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  if (button_outer_node_ring >= 1 && c == 0) {
    double centroid_r = 0.0;
    double centroid_z = 0.0;
    rz::button_polygon_area_centroid_from_nodes(
        x_r, x_z, button_outer_node_ring, nz, &centroid_r, &centroid_z);
    const double volume = rz::button_polygon_volume_from_nodes(
        x_r, x_z, button_outer_node_ring, nz);
    if (!(volume > 0.0) || !isfinite(volume)) {
      return;
    }
    const double rho_button = mass_cell[0] / volume;
    const int nverts = nz + 1;
    for (int k = 0; k < nverts; ++k) {
      const int n = rz::button_seam_node_index(button_outer_node_ring, k, nz);
      const double m_corner = rz::button_corner_mass_exact_subpolygon(
          rho_button, x_r, x_z, button_outer_node_ring, nz, k, centroid_r,
          centroid_z);
      atomic_add_double(node_mass + n, m_corner);
    }
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  atomic_add_double(node_mass + n00, corner_mass[c * corner_stride + 0]);
  atomic_add_double(node_mass + n10, corner_mass[c * corner_stride + 1]);
  atomic_add_double(node_mass + n11, corner_mass[c * corner_stride + 2]);
  if (mesh::mesh_topo_cell_active_nverts(cell_nverts, c) != 3) {
    atomic_add_double(node_mass + n01, corner_mass[c * corner_stride + 3]);
  }
}

__global__ void compute_node_planar_mass_2d_kernel(
    double* __restrict__ node_planar_mass,
    const double* __restrict__ rho,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  const double r[4] = {x_r[n00], x_r[n10], x_r[n11], x_r[n01]};
  const double z[4] = {x_z[n00], x_z[n10], x_z[n11], x_z[n01]};
  double area[4];
  rz::aw_planar_subzonal_areas(r, z, area);
  const double rho_cell = rho[c];

  atomic_add_double(node_planar_mass + n00, rho_cell * area[0]);
  atomic_add_double(node_planar_mass + n10, rho_cell * area[1]);
  atomic_add_double(node_planar_mass + n11, rho_cell * area[2]);
  atomic_add_double(node_planar_mass + n01, rho_cell * area[3]);
}

template <int SlotCap = mesh::kMeshTopoCellStorageSlotsMax>
__global__ void compute_node_planar_mass_2d_multiblock_kernel(
    double* __restrict__ node_planar_mass,
    const double* __restrict__ rho,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int n_nodes_total) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes_total) {
    return;
  }

  node_planar_mass[n] = rz::aw_csr_node_planar_mass<SlotCap>(
      rho, x_r, x_z, reverse_csr_node_offsets, reverse_csr_node_cells,
      reverse_csr_node_corners, cell_node_csr_offsets,
      cell_node_csr_indices, cell_nverts, hydro_active, n);
}

__global__ void compute_node_activity_2d_multiblock_kernel(
    std::uint8_t* __restrict__ node_active,
    const int* __restrict__ reverse_csr_node_offsets,
    const int n_begin,
    const int n_end,
    const int n_nodes_total) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }

  const int degree =
      reverse_csr_node_degree(reverse_csr_node_offsets, n);
  node_active[n] = (degree > 0) ? 1u : 0u;
}

__global__ void compute_node_mass_2d_multiblock_kernel(
    double* __restrict__ node_mass,
    const double* __restrict__ corner_mass,
    const double* __restrict__ mass_cell,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const std::int8_t* __restrict__ hydro_active,
    const int n_begin,
    const int n_end,
    const int n_nodes_total,
    const bool enable_control_volume_fallback,
    const bool enable_bbs_axis_policy,
    const int equal_split_node_mass_mode,
    const int corner_stride) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }

  const int degree =
      reverse_csr_node_degree(reverse_csr_node_offsets, n);
  double sum = 0.0;
  double fallback_sum = 0.0;
  double bbs_axis_sum = 0.0;
  const bool bbs_axis_node =
      enable_bbs_axis_policy && x_r != nullptr && fabs(x_r[n]) <= 1.0e-300;
  const bool equal_split_axis_node =
      (equal_split_node_mass_mode == 1) && x_r != nullptr &&
      fabs(x_r[n]) <= 1.0e-300;
  const bool equal_split_all_node = (equal_split_node_mass_mode == 2);
  for (int k = 0; k < degree; ++k) {
    const int c =
        reverse_csr_node_cell(reverse_csr_node_offsets,
                              reverse_csr_node_cells,
                              n,
                              k);
    if (hydro_active != nullptr && hydro_active[c] == 0) {
      continue;
    }
    const int corner =
        reverse_csr_node_corner(reverse_csr_node_offsets,
                                reverse_csr_node_corners,
                                n,
                                k);
    sum += corner_mass[c * corner_stride + corner];
    const int active_nverts =
        mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
    if (mass_cell != nullptr) {
      const double m = mass_cell[c];
      if (m > 0.0 && isfinite(m)) {
        fallback_sum +=
            (active_nverts == 3) ? (m / 3.0) : (0.25 * m);
      }
    }
    if (bbs_axis_node && mass_cell != nullptr && x_r != nullptr &&
        x_z != nullptr && cell_node_csr_offsets != nullptr &&
        cell_node_csr_indices != nullptr) {
      const double m = mass_cell[c];
      const int off = cell_node_csr_offsets[c];
      double m_corner[4] = {0.0, 0.0, 0.0, 0.0};
      if (active_nverts == 3) {
        const int n0 = cell_node_csr_indices[off + 0];
        const int n1 = cell_node_csr_indices[off + 1];
        const int n2 = cell_node_csr_indices[off + 2];
        rz::compute_triangle_corner_masses_exact(fmax(m, 0.0),
                                                 x_r[n0],
                                                 x_z[n0],
                                                 x_r[n1],
                                                 x_z[n1],
                                                 x_r[n2],
                                                 x_z[n2],
                                                 m_corner);
        bbs_axis_sum += fmax(m_corner[corner], 0.0);
      } else {
        const int n00 = cell_node_csr_indices[off + 0];
        const int n10 = cell_node_csr_indices[off + 1];
        const int n11 = cell_node_csr_indices[off + 2];
        const int n01 = cell_node_csr_indices[off + 3];
        if (rz::compute_quad_corner_masses_rz_volume_no_planar_fallback(
                fmax(m, 0.0),
                x_r[n00],
                x_z[n00],
                x_r[n10],
                x_z[n10],
                x_r[n11],
                x_z[n11],
                x_r[n01],
                x_z[n01],
                m_corner)) {
          bbs_axis_sum += fmax(m_corner[corner], 0.0);
        }
      }
    }
  }
  // The 5-block central Cartesian core can put an axis node on a remapped
  // corner-mass cancellation point.  A denormal denominator there feeds an
  // overflow compatible acceleration.  The default legacy path uses the
  // incident-cell planar control volume only for roundoff-scale cancellations.
  // The opt-in BBS axis path replaces that denominator at R=0 with exact RZ
  // subzonal volume masses, with no planar fallback.
  // Axis-node mass-convention opt-in: assemble axis-node (r==0) mass from equal-split shares,
  // matching the structured polar path's corner-mass convention. The exact
  // r-weighted subzonal sum halves the axis-node mass and makes the pole
  // accelerate 4/3x the interior under drive (the pole impedance seed).
  if ((equal_split_axis_node || equal_split_all_node) && degree > 0 &&
      isfinite(fallback_sum) && fallback_sum > 0.0) {
    sum = fallback_sum;
  }
  if (degree > 0 && enable_control_volume_fallback && isfinite(sum) &&
      !(sum > 1.0e-300)) {
    if (bbs_axis_node) {
      sum = isfinite(bbs_axis_sum) ? bbs_axis_sum : sum;
    } else if (fallback_sum > 1.0e-300 && isfinite(fallback_sum) &&
               sum >= -1.0e-8 * fallback_sum) {
      sum = fallback_sum;
    }
  }
  // Non-5-block CSR ALE remap can leave roundoff-scale negative nodal sums;
  // preserving the legacy floor keeps those paths byte-identical.
  if (degree > 0 && !enable_control_volume_fallback && isfinite(sum) &&
      !(sum > 1.0e-300)) {
    sum = 2.0e-300;
  }
  node_mass[n] = sum;
}

__global__ void build_cell_pq_kernel(double* __restrict__ pq,
                                     const double* __restrict__ Pe,
                                     const double* __restrict__ Pi,
                                     const double* __restrict__ Qvisc,
                                     const std::int8_t* __restrict__ hydro_active,
                                     const int c_begin,
                                     const int c_end,
                                     const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  pq[c] = active ? (Pe[c] + Pi[c] + Qvisc[c]) : 0.0;
}

__global__ void build_cell_pressure_kernel(double* __restrict__ p,
                                           const double* __restrict__ Pe,
                                           const double* __restrict__ Pi,
                                           const std::int8_t* __restrict__ hydro_active,
                                           const int c_begin,
                                           const int c_end,
                                           const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  p[c] = active ? (Pe[c] + Pi[c]) : 0.0;
}

__global__ void build_cell_pressure_plus_q_kernel(
    double* __restrict__ pq,
    const double* __restrict__ pressure,
    const double* __restrict__ Qvisc,
    const std::int8_t* __restrict__ hydro_active,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  pq[c] = active ? (pressure[c] + Qvisc[c]) : 0.0;
}

__global__ void build_cell_q_kernel(double* __restrict__ q,
                                    const double* __restrict__ Qvisc,
                                    const std::int8_t* __restrict__ hydro_active,
                                    const int c_begin,
                                    const int c_end,
                                    const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  q[c] = active ? Qvisc[c] : 0.0;
}

__global__ void qcap_apply_2d_kernel(double* __restrict__ Qvisc,
                                     double* __restrict__ Qvisc_per_material,
                                     const double* __restrict__ Pe,
                                     const double* __restrict__ Pi,
                                     const std::int8_t* __restrict__ hydro_active,
                                     const int c_begin,
                                     const int c_end,
                                     const int n_cells,
                                     const int nz,
                                     const int n_mat,
                                     const double qcap_over_p,
                                     const int scope,
                                     const int center_band_max_i,
                                     const double* __restrict__ cell_centroid_r,
                                     const double r_match_threshold) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  if (scope == kAvQcapScopeGlobal) {
    // No per-cell predicate.
  } else if (scope == kAvQcapScopeTriFanRadialIndex) {
    const int i = c / nz;
    if (i > center_band_max_i) {
      return;
    }
  } else if (scope == kAvQcapScopeCentroidRLeRMatch) {
    if (cell_centroid_r == nullptr || cell_centroid_r[c] > r_match_threshold) {
      return;
    }
  }

  const double q_raw = Qvisc[c];
  const double p_nonnegative = fmax(Pe[c] + Pi[c], 0.0);
  const double q_limit = qcap_over_p * p_nonnegative;
  const double q_capped = (q_limit <= 0.0) ? 0.0 : fmin(q_raw, q_limit);
  Qvisc[c] = q_capped;

  if (Qvisc_per_material != nullptr && n_mat > 0) {
    constexpr double eps = 1.0e-300;
    const double factor = q_capped / fmax(q_raw, eps);
    const int base = c * n_mat;
    for (int m = 0; m < n_mat; ++m) {
      Qvisc_per_material[base + m] *= factor;
    }
  }
}

__global__ void compute_cell_dVdt_kernel(double* __restrict__ dVdt,
                                         const double* __restrict__ v_r,
                                         const double* __restrict__ v_z,
                                         const double* __restrict__ Svec_r,
                                         const double* __restrict__ Svec_z,
                                         const std::int8_t* __restrict__ hydro_active,
                                         const double* __restrict__ x_r,
                                         const double* __restrict__ x_z,
                                         const int c_begin,
                                         const int c_end,
                                         const int nr,
                                         const int nz,
                                         const int button_outer_node_ring,
                                         const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    dVdt[c] = 0.0;
    return;
  }

  if (button_outer_node_ring >= 1 && c == 0) {
    double centroid_r = 0.0;
    double centroid_z = 0.0;
    rz::button_polygon_area_centroid_from_nodes(
        x_r, x_z, button_outer_node_ring, nz, &centroid_r, &centroid_z);
    const int nverts = nz + 1;
    double sum = 0.0;
    for (int k = 0; k < nverts; ++k) {
      const int n = rz::button_seam_node_index(button_outer_node_ring, k, nz);
      double sr = 0.0;
      double sz = 0.0;
      rz::button_polygon_svec_from_nodes(x_r, x_z, button_outer_node_ring, nz,
                                         k, centroid_r, centroid_z, &sr, &sz);
      sum += v_r[n] * sr + v_z[n] * sz;
    }
    dVdt[c] = sum;
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  const int idx = c * corner_stride;
  double sum = 0.0;
  sum += v_r[n00] * Svec_r[idx + 0] + v_z[n00] * Svec_z[idx + 0];
  sum += v_r[n10] * Svec_r[idx + 1] + v_z[n10] * Svec_z[idx + 1];
  sum += v_r[n11] * Svec_r[idx + 2] + v_z[n11] * Svec_z[idx + 2];
  sum += v_r[n01] * Svec_r[idx + 3] + v_z[n01] * Svec_z[idx + 3];
  dVdt[c] = sum;
}

__global__ void compute_cell_dVdt_2d_multiblock_kernel(
    double* __restrict__ dVdt,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ Svec_r,
    const double* __restrict__ Svec_z,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const std::uint8_t* __restrict__ cell_nverts,
    const int c_begin,
    const int c_end,
    const int n_cells_total,
    const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const int off = cell_node_csr_offsets[c];
  const int idx = c * corner_stride;
  const int active_nverts = mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  double sum = 0.0;
  if (active_nverts == 3) {
    const int n0 = cell_node_csr_indices[off + 0];
    const int n1 = cell_node_csr_indices[off + 1];
    const int n2 = cell_node_csr_indices[off + 2];
    sum += v_r[n0] * Svec_r[idx + 0] + v_z[n0] * Svec_z[idx + 0];
    sum += v_r[n1] * Svec_r[idx + 1] + v_z[n1] * Svec_z[idx + 1];
    sum += v_r[n2] * Svec_r[idx + 2] + v_z[n2] * Svec_z[idx + 2];
  } else if (active_nverts >= 5) {
    for (int k = 0; k < active_nverts; ++k) {
      const int node = cell_node_csr_indices[off + k];
      sum += v_r[node] * Svec_r[idx + k] +
             v_z[node] * Svec_z[idx + k];
    }
  } else {
    const int n00 = cell_node_csr_indices[off + 0];
    const int n10 = cell_node_csr_indices[off + 1];
    const int n11 = cell_node_csr_indices[off + 2];
    const int n01 = cell_node_csr_indices[off + 3];
    sum += v_r[n00] * Svec_r[idx + 0] + v_z[n00] * Svec_z[idx + 0];
    sum += v_r[n10] * Svec_r[idx + 1] + v_z[n10] * Svec_z[idx + 1];
    sum += v_r[n11] * Svec_r[idx + 2] + v_z[n11] * Svec_z[idx + 2];
    sum += v_r[n01] * Svec_r[idx + 3] + v_z[n01] * Svec_z[idx + 3];
  }
  dVdt[c] = sum;
}

__global__ void compute_cell_div_u_kernel(double* __restrict__ div_u,
                                          const double* __restrict__ dVdt,
                                          const double* __restrict__ vol,
                                          const int c_begin,
                                          const int c_end,
                                          const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  div_u[c] = (vol[c] > 0.0) ? (dVdt[c] / vol[c]) : 0.0;
}

__global__ void compute_cell_length_scale_kernel(double* __restrict__ dl,
                                                 const double* __restrict__ area,
                                                 const int c_begin,
                                                 const int c_end,
                                                 const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  dl[c] = (area[c] > 0.0) ? sqrt(area[c]) : 0.0;
}

template <int RZ_SCHEME>
__global__ void compute_force_from_cells_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ cell_pq,
    const double* __restrict__ Svec_r,
    const double* __restrict__ Svec_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int nr,
    const int nz,
    const int button_outer_node_ring,
    const int corner_stride) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= c_end) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    return;
  }

  if (button_outer_node_ring >= 1 && c == 0) {
    double centroid_r = 0.0;
    double centroid_z = 0.0;
    rz::button_polygon_area_centroid_from_nodes(
        x_r, x_z, button_outer_node_ring, nz, &centroid_r, &centroid_z);
    const double pq = cell_pq[c];
    const int nverts = nz + 1;
    for (int k = 0; k < nverts; ++k) {
      const int n = rz::button_seam_node_index(button_outer_node_ring, k, nz);
      double sr = 0.0;
      double sz = 0.0;
      rz::button_polygon_svec_from_nodes(x_r, x_z, button_outer_node_ring, nz,
                                         k, centroid_r, centroid_z, &sr, &sz);
      atomic_add_double(force_r + n, pq * sr);
      atomic_add_double(force_z + n, pq * sz);
    }
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  const int idx = c * corner_stride;
  const double pq = cell_pq[c];
  if constexpr (RZ_SCHEME == 1) {
    const double r[4] = {x_r[n00], x_r[n10], x_r[n11], x_r[n01]};
    const double z[4] = {x_z[n00], x_z[n10], x_z[n11], x_z[n01]};
    double planar_Sr[4];
    double planar_Sz[4];
    rz::aw_planar_corner_vectors(r, z, planar_Sr, planar_Sz);
    atomic_add_double(force_r + n00, +pq * planar_Sr[0]);
    atomic_add_double(force_z + n00, +pq * planar_Sz[0]);
    atomic_add_double(force_r + n10, +pq * planar_Sr[1]);
    atomic_add_double(force_z + n10, +pq * planar_Sz[1]);
    atomic_add_double(force_r + n11, +pq * planar_Sr[2]);
    atomic_add_double(force_z + n11, +pq * planar_Sz[2]);
    atomic_add_double(force_r + n01, +pq * planar_Sr[3]);
    atomic_add_double(force_z + n01, +pq * planar_Sz[3]);
  } else {
    atomic_add_double(force_r + n00, +pq * Svec_r[idx + 0]);
    atomic_add_double(force_z + n00, +pq * Svec_z[idx + 0]);
    atomic_add_double(force_r + n10, +pq * Svec_r[idx + 1]);
    atomic_add_double(force_z + n10, +pq * Svec_z[idx + 1]);
    atomic_add_double(force_r + n11, +pq * Svec_r[idx + 2]);
    atomic_add_double(force_z + n11, +pq * Svec_z[idx + 2]);
    atomic_add_double(force_r + n01, +pq * Svec_r[idx + 3]);
    atomic_add_double(force_z + n01, +pq * Svec_z[idx + 3]);
  }
}

template <int RZ_SCHEME>
__global__ void compute_force_from_cells_2d_multiblock_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    const double* __restrict__ cell_pq,
    const double* __restrict__ Svec_r,
    const double* __restrict__ Svec_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::int8_t* __restrict__ hydro_active,
    const int* __restrict__ reverse_csr_node_offsets,
    const int* __restrict__ reverse_csr_node_cells,
    const int* __restrict__ reverse_csr_node_corners,
    const int* __restrict__ cell_node_csr_offsets,
    const int* __restrict__ cell_node_csr_indices,
    const int n_begin,
    const int n_end,
    const int n_nodes_total,
    const int corner_stride) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }

  const int off = reverse_csr_node_offsets[n];
  const int end = reverse_csr_node_offsets[n + 1];
  double fr = 0.0;
  double fz = 0.0;
  for (int k = off; k < end; ++k) {
    const int c = reverse_csr_node_cells[k];
    if (hydro_active != nullptr && hydro_active[c] == 0) {
      continue;
    }
    const int corner = reverse_csr_node_corners[k];
    const double pq = cell_pq[c];
    if constexpr (RZ_SCHEME == 1) {
      const int cell_off = cell_node_csr_offsets[c];
      const int n00 = cell_node_csr_indices[cell_off + 0];
      const int n10 = cell_node_csr_indices[cell_off + 1];
      const int n11 = cell_node_csr_indices[cell_off + 2];
      const int n01 = cell_node_csr_indices[cell_off + 3];
      const double r[4] = {x_r[n00], x_r[n10], x_r[n11], x_r[n01]};
      const double z[4] = {x_z[n00], x_z[n10], x_z[n11], x_z[n01]};
      double planar_Sr[4];
      double planar_Sz[4];
      rz::aw_planar_corner_vectors(r, z, planar_Sr, planar_Sz);
      fr += pq * planar_Sr[corner];
      fz += pq * planar_Sz[corner];
    } else {
      fr += pq * Svec_r[c * corner_stride + corner];
      fz += pq * Svec_z[c * corner_stride + corner];
    }
  }
  force_r[n] = fr;
  force_z[n] = fz;
}

template <int RZ_SCHEME>
__global__ void compute_accel_from_force_kernel(
    double* __restrict__ accel_r,
    double* __restrict__ accel_z,
    const double* __restrict__ force_r,
    const double* __restrict__ force_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ node_planar_mass,
    const std::uint8_t* __restrict__ node_active,
    const std::uint8_t* __restrict__ node_flags,
    const int n_begin,
    const int n_end,
    const int n_nodes) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }

  if constexpr (RZ_SCHEME == 1) {
    if (node_active[n] == 0u || node_planar_mass[n] <= 0.0) {
      accel_r[n] = 0.0;
      accel_z[n] = 0.0;
      return;
    }
    if (node_flags != nullptr && (node_flags[n] & kNodeCenterFlag) != 0U) {
      accel_r[n] = 0.0;
      accel_z[n] = force_z[n] / node_planar_mass[n];
      return;
    }

    accel_r[n] = force_r[n] / node_planar_mass[n];
    accel_z[n] = force_z[n] / node_planar_mass[n];
  } else {
    // Volumetric nodal mass degenerates at the axis, so the center remains
    // fully constrained on this legacy path.
    if ((node_flags != nullptr && (node_flags[n] & kNodeCenterFlag) != 0U) ||
        node_active[n] == 0u || node_mass[n] <= 0.0) {
      accel_r[n] = 0.0;
      accel_z[n] = 0.0;
      return;
    }

    accel_r[n] = force_r[n] / node_mass[n];
    accel_z[n] = force_z[n] / node_mass[n];
  }
}

// The degenerate polar-center column is one material point: every
// NODE_CENTER node aliases (R,Z)=(0,0), so integrating each node's own
// F_z/m lets O(1) per-node discrete residuals drift the shared material
// velocity even in symmetric flows. Aggregate instead: one axial
// acceleration (sum F_z)/(sum m_planar) over the column, applied to every
// center node. Summation is mirror-paired (first+last inward) over the
// ascending node order, so an exactly mirror-antisymmetric force field
// cancels bitwise and symmetric flows keep A_z == 0.0 exactly; a uniform
// boost has F == 0 and v_z is preserved by integration (Galilean).
// Single-thread on purpose: the pair order is the determinism and the
// exact-cancellation guarantee; the column has O(n_theta) nodes.
__global__ void aggregate_center_axial_accel_kernel(
    double* __restrict__ accel_r,
    double* __restrict__ accel_z,
    const double* __restrict__ force_z,
    const double* __restrict__ node_planar_mass,
    const std::uint8_t* __restrict__ node_flags,
    const std::uint8_t* __restrict__ node_active,
    const int begin,
    const int end) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  // Documented capacity: covers every production ladder (n_theta <= 192).
  // A larger column falls back to the per-node integration unchanged.
  constexpr int kMaxCenterColumn = 1024;
  int centers[kMaxCenterColumn];
  int count = 0;
  for (int n = begin; n < end; ++n) {
    if ((node_flags[n] & kNodeCenterFlag) != 0U && node_active[n] != 0u &&
        node_planar_mass[n] > 0.0) {
      if (count >= kMaxCenterColumn) {
        return;
      }
      centers[count] = n;
      ++count;
    }
  }
  if (count == 0) {
    return;
  }
  double force_sum = 0.0;
  double mass_sum = 0.0;
  int low = 0;
  int high = count - 1;
  for (; low < high; ++low, --high) {
    force_sum += force_z[centers[low]] + force_z[centers[high]];
    mass_sum += node_planar_mass[centers[low]] +
                node_planar_mass[centers[high]];
  }
  if (low == high) {
    force_sum += force_z[centers[low]];
    mass_sum += node_planar_mass[centers[low]];
  }
  const double accel = mass_sum > 0.0 ? force_sum / mass_sum : 0.0;
  for (int index = 0; index < count; ++index) {
    accel_r[centers[index]] = 0.0;
    accel_z[centers[index]] = accel;
  }
}

__global__ void apply_evac_contact_accel_constraints_kernel(
    double* accel_r,
    double* accel_z,
    const double* node_mass_resolved,
    const int* pair_nodes,
    const double* pair_normal,
    double* pair_lambda,
    const int n_pairs) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  for (int pair = 0; pair < n_pairs; ++pair) {
    const int node_a = pair_nodes[2 * pair];
    const int node_b = pair_nodes[2 * pair + 1];
    const double normal_r = pair_normal[2 * pair];
    const double normal_z = pair_normal[2 * pair + 1];
    const double mA = node_mass_resolved[node_a];
    const double mB = node_mass_resolved[node_b];
    if (!(mA > 0.0) || !(mB > 0.0)) {
      pair_lambda[pair] = 0.0;
      continue;
    }
    const double aA_n =
        accel_r[node_a] * normal_r + accel_z[node_a] * normal_z;
    const double aB_n =
        accel_r[node_b] * normal_r + accel_z[node_b] * normal_z;
    // evacuated_cell_shadow.hpp is the source of truth for these two
    // contact multiplier/common-acceleration formulas.
    pair_lambda[pair] = (aA_n - aB_n) * (mA * mB / (mA + mB));
    const double common_accel =
        (mA * aA_n + mB * aB_n) / (mA + mB);
    accel_r[node_a] += (common_accel - aA_n) * normal_r;
    accel_z[node_a] += (common_accel - aA_n) * normal_z;
    accel_r[node_b] += (common_accel - aB_n) * normal_r;
    accel_z[node_b] += (common_accel - aB_n) * normal_z;
  }
}

__device__ inline void accumulate_evac_contact_mortar_row_scale(
    const int neighbor,
    const int n_cells,
    const int shared0,
    const int shared1,
    const int outer0,
    const int outer1,
    const double xi,
    const double normal_r,
    const double normal_z,
    const double* const x_r,
    const double* const x_z,
    const double* const cell_sound_speed,
    const std::uint8_t* const inactive_member_mask,
    const std::int8_t* const hydro_active,
    double* const h_n,
    double* const c_contact) {
  if (neighbor < 0 || neighbor >= n_cells ||
      inactive_member_mask[neighbor] != 0U ||
      (hydro_active != nullptr && hydro_active[neighbor] == 0)) {
    return;
  }
  const double shared_r =
      (1.0 - xi) * x_r[shared0] + xi * x_r[shared1];
  const double shared_z =
      (1.0 - xi) * x_z[shared0] + xi * x_z[shared1];
  const double outer_r =
      (1.0 - xi) * x_r[outer0] + xi * x_r[outer1];
  const double outer_z =
      (1.0 - xi) * x_z[outer0] + xi * x_z[outer1];
  const double altitude =
      fabs((shared_r - outer_r) * normal_r +
           (shared_z - outer_z) * normal_z);
  if (isfinite(altitude) && altitude > 0.0) {
    *h_n = fmin(*h_n, altitude);
  }
  const double sound_speed = cell_sound_speed[neighbor];
  if (isfinite(sound_speed) && sound_speed > 0.0) {
    *c_contact = fmin(*c_contact, sound_speed);
  }
}

__device__ inline void evac_contact_row_hold_sweep(
    double* v_r,
    double* v_z,
    const double* node_mass_resolved,
    const int* row_nodes,
    const double* row_xi,
    const double* row_normal,
    double* row_dk,
    int n_rows) {
  for (int r = 0; r < n_rows; ++r) {
    const int ns = row_nodes[3*r], na = row_nodes[3*r+1], nb = row_nodes[3*r+2];
    const double xi = row_xi[r];
    const double nrm_r = row_normal[2*r], nrm_z = row_normal[2*r+1];
    const double ms = node_mass_resolved[ns], ma = node_mass_resolved[na],
                 mb = node_mass_resolved[nb];
    if (!(ms > 0.0) || !(ma > 0.0) || !(mb > 0.0)) continue;
    // G v = n·v_s - (1-xi) n·v_a - xi n·v_b ; closing when G v < 0.
    const double vs_n = v_r[ns]*nrm_r + v_z[ns]*nrm_z;
    const double va_n = v_r[na]*nrm_r + v_z[na]*nrm_z;
    const double vb_n = v_r[nb]*nrm_r + v_z[nb]*nrm_z;
    const double gv = vs_n - (1.0 - xi)*va_n - xi*vb_n;
    if (!(gv < 0.0)) continue;               // unilateral: only closing rows push
    const double w_row = 1.0/ms + (1.0 - xi)*(1.0 - xi)/ma + xi*xi/mb;
    const double lambda = -gv / w_row;       // impulse enforcing G v = 0
    const double kinetic_before =
        0.5*ms*vs_n*vs_n + 0.5*ma*va_n*va_n + 0.5*mb*vb_n*vb_n;
    v_r[ns] += (lambda/ms)*nrm_r;            v_z[ns] += (lambda/ms)*nrm_z;
    v_r[na] -= ((1.0 - xi)*lambda/ma)*nrm_r; v_z[na] -= ((1.0 - xi)*lambda/ma)*nrm_z;
    v_r[nb] -= (xi*lambda/mb)*nrm_r;         v_z[nb] -= (xi*lambda/mb)*nrm_z;
    const double vs_n2 = v_r[ns]*nrm_r + v_z[ns]*nrm_z;
    const double va_n2 = v_r[na]*nrm_r + v_z[na]*nrm_z;
    const double vb_n2 = v_r[nb]*nrm_r + v_z[nb]*nrm_z;
    const double kinetic_after =
        0.5*ms*vs_n2*vs_n2 + 0.5*ma*va_n2*va_n2 + 0.5*mb*vb_n2*vb_n2;
    if (row_dk != nullptr) row_dk[r] += kinetic_before - kinetic_after;
  }
}

__device__ inline void evac_contact_union_velocity_sweep(
    double* v_r,
    double* v_z,
    const double* node_mass_resolved,
    const int* pair_nodes,
    const double* pair_normal,
    const std::uint8_t* __restrict__ pair_row_covered,
    double* pair_dk_accum,
    const int n_pairs,
    const int* row_nodes,
    const double* row_xi,
    const double* row_normal,
    double* row_dk,
    const int n_rows) {
  if (n_rows > 0) {
    evac_contact_row_hold_sweep(
        v_r, v_z, node_mass_resolved, row_nodes, row_xi, row_normal, row_dk,
        n_rows);
  }
  for (int p = 0; p < n_pairs; ++p) {
    if (n_rows > 0 && pair_row_covered != nullptr &&
        pair_row_covered[p] != 0U) {
      continue;
    }
    const int na = pair_nodes[2*p], nb = pair_nodes[2*p+1];
    const double nr = pair_normal[2*p], nz = pair_normal[2*p+1];
    const double ma = node_mass_resolved[na], mb = node_mass_resolved[nb];
    if (!(ma > 0.0) || !(mb > 0.0)) continue;
    const double van = v_r[na]*nr + v_z[na]*nz;
    const double vbn = v_r[nb]*nr + v_z[nb]*nz;
    const double uc = van - vbn;
    if (!(uc > 0.0)) continue;               // unilateral: only closing
    const double meff = ma*mb/(ma+mb);
    const double common = (ma*van + mb*vbn)/(ma+mb);
    v_r[na] += (common - van)*nr;  v_z[na] += (common - van)*nz;
    v_r[nb] += (common - vbn)*nr;  v_z[nb] += (common - vbn)*nz;
    if (pair_dk_accum != nullptr) {
      pair_dk_accum[p] += 0.5*meff*uc*uc;
    }
  }
}

__global__ void apply_evac_contact_velocity_constraints_kernel(
    double* v_r, double* v_z,
    const double* node_mass_resolved,
    const double* x_r,
    const double* x_z,
    const double* cell_sound_speed,
    const std::uint8_t* inactive_member_mask,
    const std::int8_t* hydro_active,
    const int* pair_nodes,
    const double* pair_normal,
    const std::uint8_t* __restrict__ pair_row_covered,
    double* pair_dk_accum,
    const int n_pairs,
    const double dt,
    const int* active_cells,
    const int* active_axis,
    const int* active_face_nodes,
    const double* active_face_normal_ref,
    const double* active_face_g0,
    const double* active_vol_at_engagement,
    const std::uint8_t* active_devolumized,
    const int* active_pair_dk_owner,
    double* mortar_g_hold,
    std::uint8_t* mortar_g_hold_valid,
    const double mortar_position_drift_beta,
    int* mortar_drift_count,
    double* mortar_drift_max_ucorr,
    int* projection_count,
    const int n_active_cells,
    const int nr_cells,
    const int nz_cells,
    const int* __restrict__ node_axis_alias,
    const int* __restrict__ row_nodes,
    const double* __restrict__ row_xi,
    const double* __restrict__ row_normal,
    double* row_dk,
    const int n_rows) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  evac_contact_union_velocity_sweep(
      v_r, v_z, node_mass_resolved, pair_nodes, pair_normal,
      pair_row_covered, pair_dk_accum, n_pairs, row_nodes, row_xi,
      row_normal, row_dk, n_rows);
  for (int active = 0; active < n_active_cells; ++active) {
    const int pair_begin = active_pair_dk_owner[active];
    const int pair_end = active + 1 < n_active_cells
                             ? active_pair_dk_owner[active + 1]
                             : n_pairs;
    const int active_cell_pair_count = pair_end - pair_begin;
    const int active_cell = active_cells[active];

    const int face_nodes[4] = {
        active_face_nodes[4 * active],
        active_face_nodes[4 * active + 1],
        active_face_nodes[4 * active + 2],
        active_face_nodes[4 * active + 3]};
    double face_r[4] = {};
    double face_z[4] = {};
    double face_mass[4] = {};
    double face_v_r[4] = {};
    double face_v_z[4] = {};
    for (int k = 0; k < 4; ++k) {
      const int node = face_nodes[k];
      face_r[k] = x_r[node];
      face_z[k] = x_z[node];
      face_mass[k] = node_mass_resolved[node];
      face_v_r[k] = v_r[node];
      face_v_z[k] = v_z[node];
    }
    double tangent[2] = {};
    double normal[2] = {};
    evacuated_cell_contact_common_frame(
        face_r[0], face_z[0], face_r[1], face_z[1],
        face_r[2], face_z[2], face_r[3], face_z[3],
        active_face_normal_ref[2 * active],
        active_face_normal_ref[2 * active + 1], tangent, normal);
    const double gap0 = (face_r[2] - face_r[0]) * normal[0] +
                        (face_z[2] - face_z[0]) * normal[1];
    const double gap1 = (face_r[3] - face_r[1]) * normal[0] +
                        (face_z[3] - face_z[1]) * normal[1];
    const double chi = evacuated_cell_contact_patch_fraction(
        gap0, gap1, active_face_g0[active]);
    if (chi > 0.0) {
      double row_r[kEvacContactMortarQuadCount][4];
      double row_z[kEvacContactMortarQuadCount][4];
      double area_q[kEvacContactMortarQuadCount];
      double phi_q[kEvacContactMortarQuadCount];
      const int n_rows = evacuated_cell_contact_mortar_rows(
          face_r, face_z, normal, chi, gap0, gap1, active_face_g0[active],
          row_r, row_z, area_q, phi_q);
      if (n_rows > 0) {
        // Rate form: the held seam's equilibrium gap sits below g0 by the
        // hold-floor rule, so the position-level shifted gap would fight the
        // hold every step; the complementarity acts on approach rates only.
        double phi_rate[kEvacContactMortarQuadCount] = {};
        double correction_velocity[kEvacContactMortarQuadCount] = {};
        bool has_position_correction = false;
        if (mortar_position_drift_beta > 0.0) {
          constexpr double gauss_x[kEvacContactMortarQuadCount] = {
              0.5 * (1.0 - kEvacContactGaussAbscissa),
              0.5,
              0.5 * (1.0 + kEvacContactGaussAbscissa)};
          const double mid0_r = 0.5 * (face_r[0] + face_r[2]);
          const double mid0_z = 0.5 * (face_z[0] + face_z[2]);
          const double mid1_r = 0.5 * (face_r[1] + face_r[3]);
          const double mid1_z = 0.5 * (face_z[1] + face_z[3]);
          const double face_dr = mid1_r - mid0_r;
          const double face_dz = mid1_z - mid0_z;
          const double face_length = sqrt(face_dr * face_dr + face_dz * face_dz);
          const int cell_i = active_cell / nz_cells;
          const int cell_j = active_cell - cell_i * nz_cells;
          const int stride = nz_cells + 1;
          for (int q = 0; q < n_rows; ++q) {
            const int hold_index =
                active * core::kEvacContactMortarRowCapacity + q;
            const double g_q = phi_q[q] + active_face_g0[active];
            if (mortar_g_hold_valid[hold_index] == 0U) {
              mortar_g_hold[hold_index] = g_q;
              mortar_g_hold_valid[hold_index] = 1U;
            }
            const double phi = g_q - mortar_g_hold[hold_index];
            const double xi = chi * gauss_x[q];
            double h_n = INFINITY;
            double c_contact = INFINITY;
            if (active_axis[active] == 1) {
              if (cell_j > 0) {
                accumulate_evac_contact_mortar_row_scale(
                    cell_i * nz_cells + cell_j - 1,
                    nr_cells * nz_cells,
                    face_nodes[0], face_nodes[1],
                    cell_i * stride + cell_j - 1,
                    (cell_i + 1) * stride + cell_j - 1,
                    xi, normal[0], normal[1], x_r, x_z, cell_sound_speed,
                    inactive_member_mask, hydro_active, &h_n, &c_contact);
              }
              if (cell_j + 1 < nz_cells) {
                accumulate_evac_contact_mortar_row_scale(
                    cell_i * nz_cells + cell_j + 1,
                    nr_cells * nz_cells,
                    face_nodes[2], face_nodes[3],
                    cell_i * stride + cell_j + 2,
                    (cell_i + 1) * stride + cell_j + 2,
                    xi, normal[0], normal[1], x_r, x_z, cell_sound_speed,
                    inactive_member_mask, hydro_active, &h_n, &c_contact);
              }
            } else {
              if (cell_i > 0) {
                accumulate_evac_contact_mortar_row_scale(
                    (cell_i - 1) * nz_cells + cell_j,
                    nr_cells * nz_cells,
                    face_nodes[0], face_nodes[1],
                    (cell_i - 1) * stride + cell_j,
                    (cell_i - 1) * stride + cell_j + 1,
                    xi, normal[0], normal[1], x_r, x_z, cell_sound_speed,
                    inactive_member_mask, hydro_active, &h_n, &c_contact);
              }
              if (cell_i + 1 < nr_cells) {
                accumulate_evac_contact_mortar_row_scale(
                    (cell_i + 1) * nz_cells + cell_j,
                    nr_cells * nz_cells,
                    face_nodes[2], face_nodes[3],
                    (cell_i + 2) * stride + cell_j,
                    (cell_i + 2) * stride + cell_j + 1,
                    xi, normal[0], normal[1], x_r, x_z, cell_sound_speed,
                    inactive_member_mask, hydro_active, &h_n, &c_contact);
              }
            }
            if (!isfinite(h_n) || !(h_n > 0.0)) {
              h_n = face_length;
            }
            const double phi_soft = 0.05 * h_n;
            if (phi < -phi_soft) {
              ++mortar_drift_count[1];
              continue;
            }
            if (!isfinite(c_contact) || !(c_contact > 0.0)) {
              continue;
            }
            double u_corr =
                (mortar_position_drift_beta / dt) * fmax(-phi, 0.0);
            u_corr = fmin(u_corr, 0.05 * c_contact);
            u_corr = fmin(u_corr, 0.05 * h_n / dt);
            if (u_corr > 0.0) {
              correction_velocity[q] = u_corr;
              has_position_correction = true;
              ++mortar_drift_count[0];
              mortar_drift_max_ucorr[0] =
                  fmax(mortar_drift_max_ucorr[0], u_corr);
            }
          }
        }
        double j_q[kEvacContactMortarQuadCount];
        double mortar_kinetic_energy_loss = 0.0;
        bool projected = false;
        if (has_position_correction) {
          projected = evacuated_cell_contact_mortar_project(
              face_mass, face_nodes,
              pair_nodes + 2 * pair_begin, pair_normal + 2 * pair_begin,
              active_cell_pair_count, row_r, row_z, area_q, phi_rate,
              n_rows, dt, face_v_r, face_v_z,
              &mortar_kinetic_energy_loss, j_q, correction_velocity);
          if (!projected) {
            ++mortar_drift_count[2];
            mortar_kinetic_energy_loss = 0.0;
            projected = evacuated_cell_contact_mortar_project(
                face_mass, face_nodes,
                pair_nodes + 2 * pair_begin, pair_normal + 2 * pair_begin,
                active_cell_pair_count, row_r, row_z, area_q, phi_rate,
                n_rows, dt, face_v_r, face_v_z,
                &mortar_kinetic_energy_loss, j_q);
          }
        } else {
          projected = evacuated_cell_contact_mortar_project(
              face_mass, face_nodes,
              pair_nodes + 2 * pair_begin, pair_normal + 2 * pair_begin,
              active_cell_pair_count, row_r, row_z, area_q, phi_rate,
              n_rows, dt, face_v_r, face_v_z,
              &mortar_kinetic_energy_loss, j_q);
        }
        if (projected) {
          for (int k = 0; k < 4; ++k) {
            const int node = face_nodes[k];
            v_r[node] = face_v_r[k];
            v_z[node] = face_v_z[k];
          }
          if (pair_dk_accum != nullptr) {
            pair_dk_accum[active_pair_dk_owner[active]] +=
                mortar_kinetic_energy_loss;
            ++projection_count[1];
          }
        }
      }
    }

    int corners[4] = {};
    evacuated_cell_corner_nodes(
        active_cells[active], nr_cells, nz_cells, corners);
    int nodes[4] = {corners[0], corners[1], corners[3], corners[2]};
    for (int k = 0; k < 4; ++k) {
      const int raw = nodes[k];
      nodes[k] = (node_axis_alias != nullptr && node_axis_alias[raw] >= 0)
                     ? node_axis_alias[raw]
                     : raw;
    }
    double r[4] = {};
    double z[4] = {};
    double mass[4] = {};
    double local_v_r[4] = {};
    double local_v_z[4] = {};
    for (int k = 0; k < 4; ++k) {
      const int node = nodes[k];
      r[k] = x_r[node];
      z[k] = x_z[node];
      mass[k] = node_mass_resolved[node];
      local_v_r[k] = v_r[node];
      local_v_z[k] = v_z[node];
    }
    double kinetic_energy_loss = 0.0;
    if (active_devolumized != nullptr &&
        active_devolumized[active] != 0U) {
      continue;
    }
    if (!evacuated_cell_contact_volume_floor_project(
            r, z, mass, nodes, pair_nodes + 2 * pair_begin,
            pair_normal + 2 * pair_begin, active_cell_pair_count,
            active_vol_at_engagement[active],
            local_v_r, local_v_z, &kinetic_energy_loss)) {
      continue;
    }
    for (int k = 0; k < 4; ++k) {
      const int node = nodes[k];
      v_r[node] = local_v_r[k];
      v_z[node] = local_v_z[k];
    }
    if (pair_dk_accum != nullptr) {
      pair_dk_accum[active_pair_dk_owner[active]] += kinetic_energy_loss;
      ++projection_count[0];
    }
  }
}

__global__ void validate_compatible_pre_accel_kernel(
    int* __restrict__ bad_node,
    int* __restrict__ bad_code,
    const double* __restrict__ force_r,
    const double* __restrict__ force_z,
    const double* __restrict__ node_mass,
    const std::uint8_t* __restrict__ node_active,
    const std::uint8_t* __restrict__ node_flags,
    const int n_begin,
    const int n_end,
    const int n_nodes,
    const double mass_floor) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  if ((node_flags != nullptr && (node_flags[n] & kNodeCenterFlag) != 0U) ||
      (node_active != nullptr && node_active[n] == 0u)) {
    return;
  }

  int code = 0;
  if (!isfinite(force_r[n])) {
    code = 1;
  } else if (!isfinite(force_z[n])) {
    code = 2;
  } else if (!isfinite(node_mass[n])) {
    code = 3;
  } else if (!(node_mass[n] > mass_floor)) {
    code = 4;
  }
  if (code != 0 && atomicCAS(bad_node, -1, n) == -1) {
    *bad_code = code;
  }
}

__device__ inline int boundary_type_to_velocity_bc_mode(const int boundary_type) {
  if (boundary_type == static_cast<int>(Boundary2DType::FIXED)) {
    return pole_axis::kVelocityBcFixed;
  }
  if (boundary_type == static_cast<int>(Boundary2DType::REFLECT)) {
    return pole_axis::kVelocityBcReflect;
  }
  if (boundary_type == static_cast<int>(Boundary2DType::STATE_SUPPLY)) {
    return pole_axis::kVelocityBcStateSupply;
  }
  return pole_axis::kVelocityBcFree;
}

__device__ inline void remove_spherical_normal_component(double& vector_r,
                                                        double& vector_z,
                                                        const double R,
                                                        const double Z) {
  const double s = sqrt(R * R + Z * Z);
  if (s > 0.0) {
    const double inv_s = 1.0 / s;
    const double nr = R * inv_s;
    const double nz = Z * inv_s;
    const double vr = vector_r;
    const double vz = vector_z;
    const double vn = vr * nr + vz * nz;
    vector_r = vr - vn * nr;
    vector_z = vz - vn * nz;
  }
}

__device__ inline void apply_r_outer_physical_vector_constraint(
    double& vector_r,
    double& vector_z,
    const double R,
    const double Z,
    const int r_outer_type) {
  switch (static_cast<Boundary2DType>(r_outer_type)) {
    case Boundary2DType::FIXED:
      vector_r = 0.0;
      vector_z = 0.0;
      break;
    case Boundary2DType::REFLECT:
      remove_spherical_normal_component(vector_r, vector_z, R, Z);
      break;
    case Boundary2DType::FREE:
    case Boundary2DType::PRESSURE:
      break;
    case Boundary2DType::STATE_SUPPLY:
    default:
      remove_spherical_normal_component(vector_r, vector_z, R, Z);
      break;
  }
}

__global__ void apply_boundary_accel_constraints_kernel(double* __restrict__ accel_r,
                                                        double* __restrict__ accel_z,
                                                        const std::uint8_t* __restrict__ node_flags,
                                                        const int n_begin,
                                                        const int n_end,
                                                        const int nr,
                                                        const int nz,
                                                        const int r_outer_type,
                                                        const int z_bottom_type,
                                                        const int z_top_type) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (n >= n_end) {
    return;
  }

  const int stride = nz + 1;
  const int i = n / stride;
  const int j = n - i * stride;

  pole_axis::apply_2d_boundary_vector_constraints(
      accel_r[n],
      accel_z[n],
      node_flags,
      n,
      i,
      j,
      nr,
      nz,
      boundary_type_to_velocity_bc_mode(r_outer_type),
      boundary_type_to_velocity_bc_mode(z_bottom_type),
      boundary_type_to_velocity_bc_mode(z_top_type),
      true);
}

__global__ void apply_boundary_accel_constraints_multiblock_kernel(
    double* __restrict__ accel_r,
    double* __restrict__ accel_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const std::uint8_t* __restrict__ node_flags,
    const int n_begin,
    const int n_end,
    const int n_nodes_total,
    const int r_inner_type,
    const int r_outer_type,
    const int z_bottom_type,
    const int z_top_type) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end || node_flags == nullptr) {
    return;
  }

  const std::uint8_t flags = node_flags[n];
  if ((flags & kNodeCenterFlag) != 0U) {
    accel_r[n] = 0.0;
    return;
  }

  static_cast<void>(z_bottom_type);
  static_cast<void>(z_top_type);
  if ((flags & (mesh::NODE_AXIS | kNodePoleAxisFlag)) != 0U) {
    accel_r[n] = 0.0;
  }
  if ((flags & mesh::NODE_INNER_PHYSICAL_BOUNDARY) != 0U) {
    const auto inner_type = static_cast<Boundary2DType>(r_inner_type);
    if (inner_type == Boundary2DType::PINNED) {
      accel_r[n] = 0.0;
      accel_z[n] = 0.0;
      return;
    }
    if (inner_type == Boundary2DType::AXIS) {
      remove_spherical_normal_component(
          accel_r[n], accel_z[n], x_r[n], x_z[n]);
      return;
    }
  }
  if ((flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U) {
    apply_r_outer_physical_vector_constraint(
        accel_r[n], accel_z[n], x_r[n], x_z[n], r_outer_type);
    return;
  }
  // NODE_BOUNDARY without axis/center/inner/outer flags does not occur in any
  // multiblock builder (flags are set only on the cylindrical axis, the
  // annular inner ring, the cap center, and the outer physical shell), so
  // internal block-seam nodes are ordinary interior nodes and receive no
  // vector constraint (2026-07-26 review: the historical seam
  // tangent projection fallthrough was unreachable and has been removed).
}

__global__ void update_node_velocity_2d_kernel(double* __restrict__ v_r,
                                               double* __restrict__ v_z,
                                               const double* __restrict__ old_v_r,
                                               const double* __restrict__ old_v_z,
                                               const double* __restrict__ a_r,
                                               const double* __restrict__ a_z,
                                               const std::uint8_t* __restrict__ node_active,
                                               const std::uint8_t* __restrict__ node_flags,
                                               const int n_begin,
                                               const int n_end,
                                               const int n_nodes,
                                               const double scale_dt,
                                               double* __restrict__ node_accel_r,
                                               double* __restrict__ node_accel_z) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  const std::uint8_t flags =
      node_flags != nullptr ? node_flags[n] : 0U;
  if ((flags & kNodeCenterFlag) != 0U) {
    v_r[n] = 0.0;
    v_z[n] = old_v_z[n] + scale_dt * a_z[n];
  } else if (node_active[n] != 0u) {
    v_r[n] = old_v_r[n] + scale_dt * a_r[n];
    v_z[n] = old_v_z[n] + scale_dt * a_z[n];
  } else {
    v_r[n] = old_v_r[n];
    v_z[n] = old_v_z[n];
  }
  if ((flags & (mesh::NODE_AXIS | kNodePoleAxisFlag)) != 0U) {
    v_r[n] = 0.0;
  }
  if (node_accel_r != nullptr && node_accel_z != nullptr) {
    if (scale_dt > 0.0) {
      node_accel_r[n] = (v_r[n] - old_v_r[n]) / scale_dt;
      node_accel_z[n] = (v_z[n] - old_v_z[n]) / scale_dt;
    } else {
      node_accel_r[n] = 0.0;
      node_accel_z[n] = 0.0;
    }
  }
}

__device__ inline void apply_aw_axis_velocity_slave_node(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    double* __restrict__ kappa_saved,
    const int p,
    const int q,
    const bool capture_kappa) {
  constexpr double kDrTolerance = 1.0e-300;
  if (capture_kappa) {
    const double dr = x_r[q] - x_r[p];
    kappa_saved[p] =
        fabs(dr) <= kDrTolerance
            ? nan("")
            : (x_z[q] - x_z[p]) / dr;
  }
  v_r[p] = 0.0;
  const double kappa = kappa_saved[p];
  if (!isnan(kappa)) {
    v_z[p] = v_z[q] - kappa * v_r[q];
  }
}

__global__ void apply_aw_axis_velocity_slave_2d_kernel(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    double* __restrict__ kappa_saved,
    const int nr,
    const int nz,
    const int first_axis_i,
    const bool theta0_axis_active,
    const bool theta_pi_axis_active,
    const bool capture_kappa) {
  const int i =
      blockIdx.x * blockDim.x + threadIdx.x + first_axis_i;
  if (i > nr) {
    return;
  }
  const int stride = nz + 1;
  if (theta0_axis_active) {
    const int p = i * stride;
    apply_aw_axis_velocity_slave_node(
        v_r, v_z, x_r, x_z, kappa_saved, p, p + 1, capture_kappa);
  }
  if (theta_pi_axis_active) {
    const int p = i * stride + nz;
    apply_aw_axis_velocity_slave_node(
        v_r, v_z, x_r, x_z, kappa_saved, p, p - 1, capture_kappa);
  }
}

__global__ void snap_aw_axis_coordinates_2d_kernel(
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    const double* __restrict__ kappa_saved,
    const int nr,
    const int nz,
    const int first_axis_i,
    const bool theta0_axis_active,
    const bool theta_pi_axis_active) {
  const int i =
      blockIdx.x * blockDim.x + threadIdx.x + first_axis_i;
  if (i > nr) {
    return;
  }
  const int stride = nz + 1;
  if (theta0_axis_active) {
    const int p = i * stride;
    const int q = p + 1;
    x_r[p] = 0.0;
    const double kappa = kappa_saved[p];
    if (!isnan(kappa)) {
      x_z[p] = x_z[q] - kappa * x_r[q];
    }
  }
  if (theta_pi_axis_active) {
    const int p = i * stride + nz;
    const int q = p - 1;
    x_r[p] = 0.0;
    const double kappa = kappa_saved[p];
    if (!isnan(kappa)) {
      x_z[p] = x_z[q] - kappa * x_r[q];
    }
  }
}

__global__ void apply_aw_axis_velocity_slave_2d_multiblock_kernel(
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    double* __restrict__ kappa_saved,
    const int* __restrict__ axis_slave_p,
    const int* __restrict__ axis_slave_q,
    const int pair_count,
    const bool capture_kappa) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pair_count) {
    return;
  }
  constexpr double kDrTolerance = 1.0e-300;
  const int p = axis_slave_p[pair];
  const int q = axis_slave_q[pair];
  if (capture_kappa) {
    const double dr = x_r[q] - x_r[p];
    kappa_saved[pair] =
        fabs(dr) <= kDrTolerance
            ? nan("")
            : (x_z[q] - x_z[p]) / dr;
  }
  v_r[p] = 0.0;
  const double kappa = kappa_saved[pair];
  if (!isnan(kappa)) {
    v_z[p] = v_z[q] - kappa * v_r[q];
  }
}

__global__ void snap_aw_axis_coordinates_2d_multiblock_kernel(
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    const double* __restrict__ kappa_saved,
    const int* __restrict__ axis_slave_p,
    const int* __restrict__ axis_slave_q,
    const int pair_count) {
  const int pair = blockIdx.x * blockDim.x + threadIdx.x;
  if (pair >= pair_count) {
    return;
  }
  const int p = axis_slave_p[pair];
  const int q = axis_slave_q[pair];
  x_r[p] = 0.0;
  const double kappa = kappa_saved[pair];
  if (!isnan(kappa)) {
    x_z[p] = x_z[q] - kappa * x_r[q];
  }
}

__global__ void commit_position_2d_kernel(double* __restrict__ x_r,
                                          double* __restrict__ x_z,
                                          const double* __restrict__ r_old,
                                          const double* __restrict__ z_old,
                                          const double* __restrict__ pos_v_r,
                                          const double* __restrict__ pos_v_z,
                                          const std::uint8_t* __restrict__ node_active,
                                          const std::uint8_t* __restrict__ node_flags,
                                          const int n_begin,
                                          const int n_end,
                                          const int n_nodes,
                                          const double dt) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  const std::uint8_t flags =
      node_flags != nullptr ? node_flags[n] : 0U;
  if ((flags & kNodeCenterFlag) != 0U) {
    // The center pin is mesh-only (w=0) and does not constrain material velocity.
    x_r[n] = 0.0;
    x_z[n] = 0.0;
  } else if (node_active[n] != 0u) {
    x_r[n] = r_old[n] + dt * pos_v_r[n];
    x_z[n] = z_old[n] + dt * pos_v_z[n];
  } else {
    x_r[n] = r_old[n];
    x_z[n] = z_old[n];
  }
  if ((flags & (mesh::NODE_AXIS | kNodePoleAxisFlag)) != 0U) {
    x_r[n] = 0.0;
  }
}

// --- I1B axis contact guard (env TENRYU_I1B_AXIS_CONTACT_GUARD, default OFF) ---
// RZ coordinate validity: no node may cross the symmetry axis (r >= 0 is a
// geometric invariant of the axisymmetric half-plane; a node at r < 0 is
// meaningless, yet r-weighted volume and corner-J checks can stay positive for
// an axis-crossing folded quad, so nothing else enforces this). Non-axis nodes
// carried into the axis by converging flow are treated as a unilateral
// contact: the radial POSITION velocity is limited so the node lands exactly
// on r = 0 instead of folding through (mesh w != u at the contact; the
// material velocity state.v is untouched). Suppressed radial mesh motion is
// tallied (events, 1/2 m dv^2, sum |dv|) and logged — nothing is silently
// dropped.
bool axis_contact_guard_enabled(const core::Config& cfg) {
  static const char* const raw =
      std::getenv("TENRYU_I1B_AXIS_CONTACT_GUARD");
  static const bool env_present = raw != nullptr && raw[0] != '\0';
  static const bool env_enabled = env_present && raw[0] != '0';
  return env_present ? env_enabled : cfg.numerics.ale.axis_contact_guard_enabled;
}

double* axis_contact_guard_stats() {
  static double* d_stats = [] {
    double* p = nullptr;
    cuda_check(cudaMalloc(&p, 3 * sizeof(double)),
               "Hydro2D: axis contact guard stats alloc failed");
    cuda_check(cudaMemset(p, 0, 3 * sizeof(double)),
               "Hydro2D: axis contact guard stats memset failed");
    return p;
  }();
  return d_stats;
}

__global__ void axis_contact_guard_kernel(
    const double* __restrict__ r_base,
    double* __restrict__ pos_v_r,
    const double* __restrict__ node_mass,
    const std::uint8_t* __restrict__ node_active,
    const std::uint8_t* __restrict__ node_flags,
    double* __restrict__ stats,
    const int n_begin,
    const int n_end,
    const int n_nodes,
    const double dt) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end || !(dt > 0.0)) {
    return;
  }
  if (node_active != nullptr && node_active[n] == 0u) {
    return;
  }
  if (node_flags != nullptr && (node_flags[n] & kNodeCenterFlag) != 0U) {
    return;
  }
  const double r0 = r_base[n];
  const double v = pos_v_r[n];
  if (!(r0 + dt * v < 0.0)) {
    return;
  }
  const double v_lim = r0 > 0.0 ? (-r0 / dt) : 0.0;
  const double dv = v - v_lim;
  atomic_add_double(stats + 0, 1.0);
  if (node_mass != nullptr) {
    atomic_add_double(stats + 1, 0.5 * node_mass[n] * dv * dv);
  }
  atomic_add_double(stats + 2, -dv);
  pos_v_r[n] = v_lim;
}

void apply_axis_contact_guard(const char* phase,
                              const core::Config& cfg,
                              const core::NodeField1D& r_base,
                              double* pos_v_r,
                              const core::NodeField1D& node_mass,
                              const std::uint8_t* d_node_active,
                              const std::uint8_t* d_node_flags,
                              const int n_nodes,
                              const core::State::LaunchWindow& nw,
                              const double dt_eff,
                              const int step) {
  if (!axis_contact_guard_enabled(cfg) || pos_v_r == nullptr ||
      !(dt_eff > 0.0)) {
    return;
  }
  axis_contact_guard_kernel<<<nw.blocks(), 256>>>(
      r_base.data(), pos_v_r, node_mass.data(), d_node_active, d_node_flags,
      axis_contact_guard_stats(), nw.begin, nw.end, n_nodes, dt_eff);
  sync_kernel("Hydro2D axis_contact_guard kernel failed");
  double h_stats[3] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(h_stats, axis_contact_guard_stats(),
                        3 * sizeof(double), cudaMemcpyDeviceToHost),
             "Hydro2D: axis contact guard stats readback failed");
  static double last_logged_events = 0.0;
  if (h_stats[0] > last_logged_events) {
    static long log_calls = 0;
    ++log_calls;
    if (log_calls <= 50 || (log_calls % 200) == 0) {
      std::ostringstream oss;
      oss << std::scientific << std::setprecision(6);
      oss << "[axis_contact_guard] step=" << step << " phase=" << phase
          << " events_total=" << h_stats[0]
          << " ke_removed_total_erg=" << h_stats[1]
          << " sum_dv_total_cm_s=" << h_stats[2];
      core::log_warning(oss.str());
    }
    last_logged_events = h_stats[0];
  }
}

__global__ void enforce_node_axis_state_kernel(
    double* __restrict__ x_r,
    double* __restrict__ x_z,
    double* __restrict__ v_r,
    double* __restrict__ v_z,
    const std::uint8_t* __restrict__ node_flags,
    const int n_begin,
    const int n_end,
    const int n_nodes) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end || node_flags == nullptr) {
    return;
  }
  const std::uint8_t flags = node_flags[n];
  if ((flags & kNodeCenterFlag) != 0U) {
    x_r[n] = 0.0;
    x_z[n] = 0.0;
    v_r[n] = 0.0;
  } else if ((flags & (mesh::NODE_AXIS | kNodePoleAxisFlag)) != 0U) {
    x_r[n] = 0.0;
    v_r[n] = 0.0;
  }
}

__global__ void zero_axis_z_position_velocity_kernel(double* __restrict__ pos_v_z,
                                                     const int nz) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  pos_v_z[j] = 0.0;
}

__global__ void zero_state_supply_z_position_velocity_kernel(
    double* __restrict__ pos_v_z,
    const int nr,
    const int nz,
    const int bottom_active,
    const int top_active) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i > nr) {
    return;
  }

  const int stride = nz + 1;
  if (bottom_active != 0) {
    pos_v_z[i * stride] = 0.0;
  }
  if (top_active != 0) {
    pos_v_z[i * stride + nz] = 0.0;
  }
}

__global__ void scale_axis_z_position_velocity_kernel(double* __restrict__ pos_v_z,
                                                      const int nz,
                                                      const double sigma) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  pos_v_z[j] *= sigma;
}

__global__ void build_axis_only_z_displacement_kernel(
    double* __restrict__ delta_r,
    double* __restrict__ delta_z,
    const double* __restrict__ pos_v_z,
    const int n_begin,
    const int n_end,
    const int n_nodes,
    const int nz,
    const double dt) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  delta_r[n] = 0.0;
  delta_z[n] = (n <= nz) ? dt * pos_v_z[n] : 0.0;
}

__global__ void t21_axis_motion_preflight_kernel(
    double* __restrict__ node_scale,
    const double* __restrict__ x_r_old,
    const double* __restrict__ x_z_old,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const int nz,
    const double dt,
    const double margin_floor_fraction) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= nz) {
    return;
  }

  const int stride = nz + 1;
  const int n_axis_j = j;
  const int n_axis_jp1 = j + 1;
  const int n_outer_j = stride + j;
  const int n_outer_jp1 = stride + j + 1;

  const double old_margin =
      t21_axis_cell_margin(x_r_old[n_outer_j], x_z_old[n_outer_j],
                           x_r_old[n_outer_jp1], x_z_old[n_outer_jp1],
                           x_z_old[n_axis_j], x_z_old[n_axis_jp1]);
  if (!(old_margin > 0.0)) {
    return;
  }
  const double margin_floor = margin_floor_fraction * old_margin;
  const double z_outer_j_trial = x_z_old[n_outer_j] + dt * v_z[n_outer_j];
  const double z_outer_jp1_trial = x_z_old[n_outer_jp1] + dt * v_z[n_outer_jp1];
  const double z_axis_j_trial = x_z_old[n_axis_j] + dt * v_z[n_axis_j];
  const double z_axis_jp1_trial = x_z_old[n_axis_jp1] + dt * v_z[n_axis_jp1];
  const double margin_full =
      t21_axis_cell_margin(x_r_old[n_outer_j] + dt * v_r[n_outer_j],
                           z_outer_j_trial,
                           x_r_old[n_outer_jp1] + dt * v_r[n_outer_jp1],
                           z_outer_jp1_trial, z_axis_j_trial, z_axis_jp1_trial);
  if (margin_full >= margin_floor) {
    return;
  }

  double lo = 0.0;
  double hi = 1.0;
  for (int iter = 0; iter < 48; ++iter) {
    const double mid = 0.5 * (lo + hi);
    const double margin_mid =
        t21_axis_cell_margin(x_r_old[n_outer_j] + dt * mid * v_r[n_outer_j],
                             z_outer_j_trial,
                             x_r_old[n_outer_jp1] + dt * mid * v_r[n_outer_jp1],
                             z_outer_jp1_trial, z_axis_j_trial, z_axis_jp1_trial);
    if (margin_mid >= margin_floor) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  t21_axis_atomic_min(node_scale + j, lo);
  t21_axis_atomic_min(node_scale + j + 1, lo);
}

__global__ void t21_apply_axis_motion_scale_kernel(double* __restrict__ v_r,
                                                   double* __restrict__ coupled_v_r,
                                                   const double* __restrict__ node_scale,
                                                   const int nz) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > nz) {
    return;
  }
  const int n = (nz + 1) + j;
  const double scale = node_scale[j];
  v_r[n] *= scale;
  if (coupled_v_r != nullptr) {
    coupled_v_r[n] *= scale;
  }
}

__global__ void copy_array_kernel(double* __restrict__ dst,
                                  const double* __restrict__ src,
                                  const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = src[i];
}

__global__ void average_arrays_kernel(double* __restrict__ out,
                                      const double* __restrict__ a,
                                      const double* __restrict__ b,
                                      const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = 0.5 * (a[i] + b[i]);
}

__device__ inline void compatible_energy_split_fractions(
    const double pe,
    const double pi,
    double* electron_fraction,
    double* ion_fraction,
    bool* cold_fallback) {
  const double p_tot = pe + pi;
  if (isfinite(p_tot) && p_tot > kCompatibleEnergyColdPressureFloor) {
    *electron_fraction = pe / p_tot;
    *ion_fraction = pi / p_tot;
    *cold_fallback = false;
    return;
  }
  *electron_fraction = 0.5;
  *ion_fraction = 0.5;
  *cold_fallback = true;
}

// VISC_SPLIT=0 executes the pre-electron arithmetic source-identically
// (OFF and species="ion" dispatch here; visc_heat_i/e are unused nullptr).
// VISC_SPLIT=1 (species="electron"/"both") splits the viscous work tally
// per channel by the electron fraction f_e = H_e/(H_i+H_e) of the
// corrector-stage heat rates (equal to eta_e/eta_eff exactly in real
// arithmetic; W:W = 0 makes the work itself vanish, so the f_e = 0 guard
// routes nothing): ion share -> ei, electron share -> ee (2T). 1T keeps
// everything in the single matter energy (unchanged expression).
template <int VISC_SPLIT>
__global__ void apply_compatible_energy_work_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ mass,
    const double* __restrict__ Pe_half,
    const double* __restrict__ Pi_half,
    const double* __restrict__ work_p,
    const double* __restrict__ work_sub,
    const double* __restrict__ work_av,
    const double* __restrict__ work_visc,
    const double* __restrict__ visc_heat_i,
    const double* __restrict__ visc_heat_e,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ central_member_mask,
    const std::uint8_t* __restrict__ central_passive_mask,
    const std::uint8_t* __restrict__ pole_member_mask,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const int use_two_temp,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count,
    int* __restrict__ cold_fallback_cell) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (floor_clamp_excluded(c, central_member_mask, central_passive_mask,
                           pole_member_mask)) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }
  const double m = mass[c];
  if (!(m > 0.0)) {
    return;
  }

  const double wp = work_p[c];
  const double ws = work_sub[c];
  const double wav = work_av[c];
  const double dt_over_m = dt / m;
  if (use_two_temp != 0) {
    double fe = 0.5;
    double fi = 0.5;
    bool cold_fallback = false;
    compatible_energy_split_fractions(Pe_half[c], Pi_half[c], &fe, &fi,
                                      &cold_fallback);
    if (cold_fallback && (wp != 0.0 || ws != 0.0) &&
        cold_fallback_cell != nullptr) {
      atomicCAS(cold_fallback_cell, -1, c);
    }
    if constexpr (VISC_SPLIT == 0) {
      const double pressure_like_work = wp + ws;
      const double ee_raw = ee[c] + dt_over_m * fe * pressure_like_work;
      const double ei_term = (work_visc == nullptr)
          ? (fi * pressure_like_work + wav)
          : (fi * pressure_like_work + wav + work_visc[c]);
      const double ei_raw = ei[c] + dt_over_m * ei_term;
      const double ee_new = fmax(ee_raw, 0.0);
      const double ei_new = fmax(ei_raw, 0.0);
      ee[c] = ee_new;
      ei[c] = ei_new;
      if (ee_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
      if (ei_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ei_new - ei_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
      return;
    } else {
      const double pressure_like_work = wp + ws;
      double visc_to_ee = 0.0;
      double visc_to_ei = 0.0;
      if (work_visc != nullptr) {
        const double wv = work_visc[c];
        const double hi = visc_heat_i[c];
        const double he = visc_heat_e[c];
        const double hsum = hi + he;
        const double f_e = (hsum > 0.0) ? (he / hsum) : 0.0;
        visc_to_ee = f_e * wv;
        visc_to_ei = (1.0 - f_e) * wv;
      }
      const double ee_raw =
          ee[c] + dt_over_m * (fe * pressure_like_work + visc_to_ee);
      const double ei_term = fi * pressure_like_work + wav + visc_to_ei;
      const double ei_raw = ei[c] + dt_over_m * ei_term;
      const double ee_new = fmax(ee_raw, 0.0);
      const double ei_new = fmax(ei_raw, 0.0);
      ee[c] = ee_new;
      ei[c] = ei_new;
      if (ee_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
      if (ei_raw < 0.0) {
        if (E_floor_injected != nullptr) {
          atomic_add_double(E_floor_injected, m * (ei_new - ei_raw));
        }
        if (clamp_count != nullptr) {
          atomicAdd(clamp_count, 1);
        }
      }
      return;
    }
  }

  const double ee_term = (work_visc == nullptr)
      ? (wp + ws + wav)
      : (wp + ws + wav + work_visc[c]);
  const double ee_raw = ee[c] + dt_over_m * ee_term;
  const double ee_new = fmax(ee_raw, 0.0);
  ee[c] = ee_new;
  if (ee_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

__global__ void apply_visc_heat_kernel(double* __restrict__ ee,
                                       double* __restrict__ ei,
                                       const double* __restrict__ mass,
                                       const double* __restrict__ heat_rate,
                                       const std::int8_t* __restrict__ hydro_active,
                                       const int c_begin,
                                       const int c_end,
                                       const int n_cells,
                                       const double dt,
                                       const int use_two_temp) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) { return; }
  if (hydro_active != nullptr && hydro_active[c] == 0) { return; }
  const double m = mass[c];
  if (!(m > 0.0)) { return; }
  const double dq = dt * heat_rate[c] / m;
  if (use_two_temp != 0) { ei[c] += dq; } else { ee[c] += dq; }
}

__global__ void energy_update_with_old_volume_kernel(
    double* __restrict__ ee,
    const double* __restrict__ e_old,
    const double* __restrict__ vol,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ P_half,
    const double* __restrict__ Q_half,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ central_member_mask,
    const std::uint8_t* __restrict__ central_passive_mask,
    const std::uint8_t* __restrict__ pole_member_mask,
    const int c_begin,
    const int c_end,
    const int n_cells,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (floor_clamp_excluded(c, central_member_mask, central_passive_mask,
                           pole_member_mask)) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    ee[c] = e_old[c];
    return;
  }

  const double m = mass[c];
  if (m <= 0.0) {
    ee[c] = e_old[c];
    return;
  }

  const double dV = vol[c] - vol_old[c];
  const double e_raw = e_old[c] - (P_half[c] + Q_half[c]) * dV / m;
  const double e_new = fmax(e_raw, 0.0);
  ee[c] = e_new;
  if (e_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (e_new - e_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

__global__ void energy_update_with_old_volume_2t_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ ee_old,
    const double* __restrict__ ei_old,
    const double* __restrict__ rho_half,
    const double* __restrict__ Te_half,
    const double* __restrict__ Ti_half,
    const double* __restrict__ vol,
    const double* __restrict__ vol_old,
    const double* __restrict__ mass,
    const double* __restrict__ Pe_half,
    const double* __restrict__ Pi_half,
    const double* __restrict__ Q_half,
    const double* __restrict__ zbar,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ central_member_mask,
    const std::uint8_t* __restrict__ central_passive_mask,
    const std::uint8_t* __restrict__ pole_member_mask,
    const int c_begin,
    const int c_end,
    const int n_cells,
    const double dt,
    const double gamma,
    const double A,
    const double fallback_z,
    const double qei_multiplier,
    const tenryu::materials::DeviceEOSTableView tab_ion,
    const tenryu::materials::DeviceEOSTableView tab_ele,
    const double* __restrict__ cv_e_arr,
    const double* __restrict__ cv_i_arr,
    const int q_heat_to_electron,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  if (floor_clamp_excluded(c, central_member_mask, central_passive_mask,
                           pole_member_mask)) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    ee[c] = ee_old[c];
    ei[c] = ei_old[c];
    return;
  }

  const double m = mass[c];
  if (m <= 0.0) {
    ee[c] = ee_old[c];
    ei[c] = ei_old[c];
    return;
  }

  const double dV = vol[c] - vol_old[c];
  const double rho_qei = fmax(rho_half[c], 0.0);
  const double z = (zbar != nullptr) ? fmax(zbar[c], 0.0) : fallback_z;
  double qei_term = 0.0;
  if (tab_ion.n_rho > 0 && tab_ele.n_rho > 0 &&
      cv_e_arr != nullptr && cv_i_arr != nullptr) {
    qei_term = tenryu::materials::compute_qei_term_with_cv(
        rho_qei, fmax(Te_half[c], 0.0), fmax(Ti_half[c], 0.0), z, A,
        fmax(cv_e_arr[c], 0.0), fmax(cv_i_arr[c], 0.0), dt, qei_multiplier);
  } else {
    qei_term = tenryu::materials::compute_qei_term_analytical(
        rho_qei, fmax(Te_half[c], 0.0), fmax(Ti_half[c], 0.0), z, A, gamma, dt,
        qei_multiplier);
  }

  const double q_work = -Q_half[c] * dV / m;
  // Q_ei applied as conservative transfer: +Q_ei to ions, -Q_ei to electrons
  // (see NUMERICS §1.1.3).
  double de_i = -Pi_half[c] * dV / m + qei_term;
  double de_e = -Pe_half[c] * dV / m - qei_term;
  if (q_heat_to_electron != 0) {
    de_e += q_work;
  } else {
    de_i += q_work;
  }

  const double ei_raw = ei_old[c] + de_i;
  const double ee_raw = ee_old[c] + de_e;
  const bool use_table = (tab_ion.n_rho > 0);
  const double ei_new = use_table ? ei_raw : fmax(ei_raw, 0.0);
  const double ee_new = use_table ? ee_raw : fmax(ee_raw, 0.0);
  ei[c] = ei_new;
  ee[c] = ee_new;

  if (!use_table && ei_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ei_new - ei_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
  if (!use_table && ee_raw < 0.0) {
    if (E_floor_injected != nullptr) {
      atomic_add_double(E_floor_injected, m * (ee_new - ee_raw));
    }
    if (clamp_count != nullptr) {
      atomicAdd(clamp_count, 1);
    }
  }
}

void apply_compatible_energy_work_2d(core::State& state,
                                     const core::Config& cfg,
                                     const bool force_enabled,
                                     const core::CellField1D& Pe_half,
                                     const core::CellField1D& Pi_half,
                                     const bool use_two_temp,
                                     const double dt,
                                     const std::int8_t* d_hydro_active,
                                     double* E_floor_injected,
                                     int* clamp_count) {
  if (!force_enabled && !compatible::compatible_force_work_enabled(cfg)) {
    return;
  }
  const int n_cells = static_cast<int>(state.mass.size());
  if (n_cells <= 0) {
    return;
  }
  TENRYU_ASSERT(state.work_p_per_cell.size() == state.mass.size() &&
                    state.work_sub_per_cell.size() == state.mass.size() &&
                    state.work_av_per_cell.size() == state.mass.size(),
                "compatible energy work buffers must match cell count");
  TENRYU_ASSERT(Pe_half.size() == state.mass.size() &&
                    Pi_half.size() == state.mass.size(),
                "compatible energy split requires half-step pressures");
  const braginskii::Params visc_energy_params =
      braginskii::params_from_config(cfg);
  const bool visc_on = visc_energy_params.enabled;
  if (visc_on) {
    TENRYU_ASSERT(state.work_visc_per_cell.size() == state.mass.size(),
                  "compatible energy requires the viscous work tally when "
                  "plasma_viscosity is enabled");
  }
  const bool visc_split = visc_on && visc_energy_params.species != 0;
  if (visc_split) {
    TENRYU_ASSERT(state.visc_heat_rate_per_cell.size() == state.mass.size() &&
                      state.visc_heat_rate_e_per_cell.size() ==
                          state.mass.size(),
                  "compatible electron viscous split requires both "
                  "corrector-stage heat-rate buffers");
  }

  int* d_cold_fallback_cell = nullptr;
  const int no_cell = -1;
  d_cold_fallback_cell = static_cast<int*>(
      core::device_scratch_acquire("h2d:compat_cold_fallback", sizeof(int)));
  cuda_check(cudaMemcpy(d_cold_fallback_cell,
                        &no_cell,
                        sizeof(int),
                        cudaMemcpyHostToDevice),
             "Hydro2D compatible energy init cold fallback cell failed");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const FloorClampExclusionMasks exclusion_masks =
      floor_clamp_exclusion_masks(state);
  if (visc_split) {
    apply_compatible_energy_work_kernel<1><<<cw.blocks(), 256>>>(
        state.ee.data(),
        state.ei.data(),
        state.mass.data(),
        Pe_half.data(),
        Pi_half.data(),
        state.work_p_per_cell.data(),
        state.work_sub_per_cell.data(),
        state.work_av_per_cell.data(),
        state.work_visc_per_cell.data(),
        state.visc_heat_rate_per_cell.data(),
        state.visc_heat_rate_e_per_cell.data(),
        d_hydro_active,
        exclusion_masks.central_member,
        exclusion_masks.central_passive,
        exclusion_masks.pole_member,
        cw.begin,
        cw.end,
        n_cells,
        dt,
        use_two_temp ? 1 : 0,
        E_floor_injected,
        clamp_count,
        d_cold_fallback_cell);
  } else {
    apply_compatible_energy_work_kernel<0><<<cw.blocks(), 256>>>(
        state.ee.data(),
        state.ei.data(),
        state.mass.data(),
        Pe_half.data(),
        Pi_half.data(),
        state.work_p_per_cell.data(),
        state.work_sub_per_cell.data(),
        state.work_av_per_cell.data(),
        visc_on ? state.work_visc_per_cell.data() : nullptr,
        nullptr,
        nullptr,
        d_hydro_active,
        exclusion_masks.central_member,
        exclusion_masks.central_passive,
        exclusion_masks.pole_member,
        cw.begin,
        cw.end,
        n_cells,
        dt,
        use_two_temp ? 1 : 0,
        E_floor_injected,
        clamp_count,
        d_cold_fallback_cell);
  }
  sync_kernel("Hydro2D compatible energy work kernel failed");

  int cold_fallback_cell = -1;
  cuda_check(cudaMemcpy(&cold_fallback_cell,
                        d_cold_fallback_cell,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Hydro2D compatible energy copy cold fallback cell failed");
  if (cold_fallback_cell >= 0) {
    core::log_debug("compatible energy: 2T split cold-limit fallback (50/50) at cell=" +
                    std::to_string(cold_fallback_cell) + " step=" +
                    std::to_string(state.step));
  }
  {
    BandWorkAudit& audit = band_work_audit();
    const double t_now = state.t;
    if (audit.enabled && t_now >= audit.t_lo && t_now <= audit.t_hi) {
      const int n_nodes = static_cast<int>(state.x_r.size());
      std::vector<double> h_xr(static_cast<std::size_t>(n_nodes));
      std::vector<double> h_xz(static_cast<std::size_t>(n_nodes));
      if (!audit.captured) {
        cuda_check(cudaMemcpy(h_xr.data(), state.x_r.data(),
                              sizeof(double) * h_xr.size(),
                              cudaMemcpyDeviceToHost),
                   "band work audit: x_r D2H failed");
        cuda_check(cudaMemcpy(h_xz.data(), state.x_z.data(),
                              sizeof(double) * h_xz.size(),
                              cudaMemcpyDeviceToHost),
                   "band work audit: x_z D2H failed");
        TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                      "band work audit requires multiblock topology");
        const auto& mb = *state.mesh.topo.multiblock;
        for (int c = 0; c < n_cells; ++c) {
          const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
          const int end =
              mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1];
          double cr = 0.0;
          double cz = 0.0;
          for (int p = off; p < end; ++p) {
            const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(p)];
            cr += h_xr[static_cast<std::size_t>(n)];
            cz += h_xz[static_cast<std::size_t>(n)];
          }
          const double inv = 1.0 / static_cast<double>(end - off);
          const double s_c = std::hypot(cr * inv, cz * inv);
          if (s_c >= audit.s_lo && s_c <= audit.s_hi) {
            audit.cells.push_back(c);
            audit.s_capture.push_back(s_c);
          }
        }
        audit.captured = true;
        audit.fp = std::fopen(audit.path.c_str(), "w");
        if (audit.fp == nullptr) {
          core::log_warning("band work audit: cannot open output path; "
                            "audit disabled");
          audit.enabled = false;
        } else {
          std::fprintf(audit.fp,
                       "# call_seq step t dt cell s_capture_cm work_p "
                       "work_sub work_av rho\n");
          core::log_info("band work audit: captured " +
                         std::to_string(audit.cells.size()) +
                         " cells in band");
        }
      }
      if (audit.enabled && !audit.cells.empty()) {
        std::vector<double> h_wp(static_cast<std::size_t>(n_cells));
        std::vector<double> h_ws(static_cast<std::size_t>(n_cells));
        std::vector<double> h_wa(static_cast<std::size_t>(n_cells));
        std::vector<double> h_rho(static_cast<std::size_t>(n_cells));
        cuda_check(cudaMemcpy(h_wp.data(), state.work_p_per_cell.data(),
                              sizeof(double) * h_wp.size(),
                              cudaMemcpyDeviceToHost),
                   "band work audit: work_p D2H failed");
        cuda_check(cudaMemcpy(h_ws.data(), state.work_sub_per_cell.data(),
                              sizeof(double) * h_ws.size(),
                              cudaMemcpyDeviceToHost),
                   "band work audit: work_sub D2H failed");
        cuda_check(cudaMemcpy(h_wa.data(), state.work_av_per_cell.data(),
                              sizeof(double) * h_wa.size(),
                              cudaMemcpyDeviceToHost),
                   "band work audit: work_av D2H failed");
        cuda_check(cudaMemcpy(h_rho.data(), state.rho.data(),
                              sizeof(double) * h_rho.size(),
                              cudaMemcpyDeviceToHost),
                   "band work audit: rho D2H failed");
        audit.call_seq += 1;
        for (std::size_t k = 0; k < audit.cells.size(); ++k) {
          const int c = audit.cells[k];
          std::fprintf(audit.fp,
                       "%ld %d %.17e %.9e %d %.9e %.9e %.9e %.9e %.9e\n",
                       audit.call_seq, state.step, t_now, dt, c,
                       audit.s_capture[k],
                       h_wp[static_cast<std::size_t>(c)],
                       h_ws[static_cast<std::size_t>(c)],
                       h_wa[static_cast<std::size_t>(c)],
                       h_rho[static_cast<std::size_t>(c)]);
        }
        std::fflush(audit.fp);
      }
    }
  }
}

__global__ void energy_update_with_old_volume_2t_per_material_kernel(
    double* __restrict__ Ee_per_material,
    double* __restrict__ Ei_per_material,
    const double* __restrict__ mass_per_material,
    const double* __restrict__ volfrac,
    const double* __restrict__ vol_new,
    const double* __restrict__ vol_old,
    const double* __restrict__ vol_half,
    const double* __restrict__ Q_per_material_half,
    double* __restrict__ Te_per_material,
    double* __restrict__ Ti_per_material,
    std::uint8_t* __restrict__ Te_per_material_valid,
    std::uint8_t* __restrict__ Ti_per_material_valid,
    unsigned long long* __restrict__ counts,
    const tenryu::materials::DeviceEOSTableView* __restrict__ electron_views,
    const tenryu::materials::DeviceEOSTableView* __restrict__ ion_views,
    const Hydro2DMaterialParams* __restrict__ params,
    const std::int8_t* __restrict__ hydro_active,
    const std::uint8_t* __restrict__ central_member_mask,
    const std::uint8_t* __restrict__ central_passive_mask,
    const std::uint8_t* __restrict__ pole_member_mask,
    const int n_cells,
    const int n_mat,
    const double te_floor,
    const double ti_floor,
    const double presence_threshold_volfrac,
    const double presence_threshold_mass_density,
    const int q_heat_to_electron,
    const bool lazy_cache_enabled,
    const bool low_density_extrap,
    double* __restrict__ E_floor_injected,
    int* __restrict__ clamp_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_mat;
  if (idx >= total) {
    return;
  }

  const int c = idx / n_mat;
  const int m = idx - c * n_mat;
  if (floor_clamp_excluded(c, central_member_mask, central_passive_mask,
                           pole_member_mask)) {
    return;
  }
  if (hydro_active != nullptr && hydro_active[c] == 0) {
    return;
  }

  const double vf = volfrac[idx];
  const double mass_m = mass_per_material[idx];
  const double V_half = vol_half[c];
  if (!(vf > presence_threshold_volfrac) || !(mass_m > 0.0) ||
      !(V_half > 0.0) || !isfinite(vf) || !isfinite(mass_m) || !isfinite(V_half)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    return;
  }
  const double rho_m = mass_m / (vf * V_half);
  if (!(rho_m > presence_threshold_mass_density) || !isfinite(rho_m)) {
    if (counts != nullptr) {
      atomicAdd(counts + tenryu::hydro::per_material::kPerMaterialCounterPresenceAbsent,
                1ULL);
    }
    return;
  }

  tenryu::hydro::per_material::PerMaterialAccessorView view{};
  view.mass_per_material = mass_per_material;
  view.Ee_per_material = Ee_per_material;
  view.Ei_per_material = Ei_per_material;
  view.volfrac = volfrac;
  view.vol = vol_half;
  view.Te_per_material = lazy_cache_enabled ? Te_per_material : nullptr;
  view.Ti_per_material = lazy_cache_enabled ? Ti_per_material : nullptr;
  view.Te_per_material_valid = lazy_cache_enabled ? Te_per_material_valid : nullptr;
  view.Ti_per_material_valid = lazy_cache_enabled ? Ti_per_material_valid : nullptr;
  view.lazy_cache_te_m_enabled = lazy_cache_enabled;
  view.presence_threshold_volfrac = presence_threshold_volfrac;
  view.presence_threshold_mass_density_g_per_cc = presence_threshold_mass_density;
  view.d_counts = counts;
  view.n_cells = n_cells;
  view.n_mat = n_mat;

  const Hydro2DMaterialParams p = params[m];
  const auto electron_view =
      (electron_views != nullptr) ? electron_views[m] : tenryu::materials::DeviceEOSTableView{};
  const auto ion_view =
      (ion_views != nullptr) ? ion_views[m] : tenryu::materials::DeviceEOSTableView{};
  const auto electron = tenryu::hydro::per_material::get_electron_thermo_per_material(
      view, electron_view, c, m, p.Zbar, p.A, te_floor, low_density_extrap, p.gamma);
  const auto ion = tenryu::hydro::per_material::get_ion_thermo_per_material(
      view, ion_view, c, m, p.A, ti_floor, low_density_extrap, p.gamma);

  const double dV = vol_new[c] - vol_old[c];
  const double dE_Q = -vf * Q_per_material_half[idx] * dV;
  const double dE_e =
      -vf * tenryu::hydro::per_material::get_pe(electron) * dV +
      ((q_heat_to_electron != 0) ? dE_Q : 0.0);
  const double dE_i =
      -vf * tenryu::hydro::per_material::get_pi(ion) * dV +
      ((q_heat_to_electron != 0) ? 0.0 : dE_Q);
  const double Ee_raw = Ee_per_material[idx] + dE_e;
  const double Ei_raw = Ei_per_material[idx] + dE_i;
  const double Ee_new = fmax(Ee_raw, 0.0);
  const double Ei_new = fmax(Ei_raw, 0.0);
  Ee_per_material[idx] = Ee_new;
  Ei_per_material[idx] = Ei_new;

  if (Ee_raw < 0.0 && E_floor_injected != nullptr) {
    atomic_add_double(E_floor_injected, Ee_new - Ee_raw);
  }
  if (Ei_raw < 0.0 && E_floor_injected != nullptr) {
    atomic_add_double(E_floor_injected, Ei_new - Ei_raw);
  }
  if ((Ee_raw < 0.0 || Ei_raw < 0.0) && clamp_count != nullptr) {
    atomicAdd(clamp_count, 1);
  }
  if (Te_per_material_valid != nullptr) {
    Te_per_material_valid[idx] = 0u;
  }
  if (Ti_per_material_valid != nullptr) {
    Ti_per_material_valid[idx] = 0u;
  }
}

__global__ void reduce_active_mass_internal_kernel(
    const double* __restrict__ mass,
    const double* __restrict__ ee,
    const std::int8_t* __restrict__ hydro_active,
    const int c_begin,
    const int c_end,
    const int n_cells,
    double* __restrict__ active_mass,
    double* __restrict__ active_internal) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }

  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    return;
  }

  const double m = mass[c];
  atomic_add_double(active_mass, m);
  atomic_add_double(active_internal, m * fmax(ee[c], 0.0));
}

__global__ void scale_active_energy_kernel(double* __restrict__ ee,
                                           const std::int8_t* __restrict__ hydro_active,
                                           const int c_begin,
                                           const int c_end,
                                           const int n_cells,
                                           const double scale) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    return;
  }
  ee[c] = fmax(ee[c] * scale, 0.0);
}

__global__ void shift_active_energy_kernel(double* __restrict__ ee,
                                           const std::int8_t* __restrict__ hydro_active,
                                           const int c_begin,
                                           const int c_end,
                                           const int n_cells,
                                           const double de) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  const bool active = (hydro_active == nullptr) || (hydro_active[c] != 0);
  if (!active) {
    return;
  }
  ee[c] = fmax(ee[c] + de, 0.0);
}

__global__ void add_first_active_residual_kernel(double* __restrict__ ee,
                                                 const double* __restrict__ mass,
                                                 const int first_active,
                                                 const double residual) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && first_active >= 0) {
    const double m = mass[first_active];
    if (m > 0.0) {
      ee[first_active] += residual / m;
    }
  }
}

constexpr int kCapEnergyAuditReduceBlockSize = 256;

struct CapEnergyAuditCompatibleTerms {
  double W_pq = 0.0;
  double W_uF = 0.0;
  double W_apex_pinned = 0.0;
};

__global__ void cap_energy_audit_cell_work_rate_kernel(
    double* __restrict__ contrib,
    const double* __restrict__ work_p,
    const double* __restrict__ work_sub,
    const double* __restrict__ work_av,
    const double* __restrict__ work_visc,
    const int c_begin,
    const int c_end,
    const int n_cells) {
  const int c = c_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= c_end) {
    return;
  }
  contrib[c] = (work_visc == nullptr)
      ? -(work_p[c] + work_sub[c] + work_av[c])
      : -(work_p[c] + work_sub[c] + work_av[c] + work_visc[c]);
}

__global__ void cap_energy_audit_nodal_work_rate_kernel(
    double* __restrict__ contrib,
    const double* __restrict__ force_r,
    const double* __restrict__ force_z,
    const double* __restrict__ velocity_r,
    const double* __restrict__ velocity_z,
    const int n_begin,
    const int n_end,
    const int n_nodes) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  contrib[n] = force_r[n] * velocity_r[n] + force_z[n] * velocity_z[n];
}

__global__ void cap_energy_audit_apex_pinned_rate_kernel(
    double* __restrict__ contrib,
    const double* __restrict__ force_r,
    const double* __restrict__ force_z,
    const double* __restrict__ old_velocity_r,
    const double* __restrict__ old_velocity_z,
    const double* __restrict__ node_mass,
    const std::uint8_t* __restrict__ node_active,
    const std::uint8_t* __restrict__ node_flags,
    const int n_begin,
    const int n_end,
    const int n_nodes,
    const double dt) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  contrib[n] = 0.0;
  if (node_flags == nullptr || (node_flags[n] & kNodeCenterFlag) == 0U) {
    return;
  }
  if (node_active != nullptr && node_active[n] == 0U) {
    return;
  }
  const double m = node_mass[n];
  if (!(m > 0.0)) {
    return;
  }
  const double fr = force_r[n];
  const double fz = force_z[n];
  const double u_pre_r = old_velocity_r[n] + dt * fr / m;
  const double u_pre_z = old_velocity_z[n] + dt * fz / m;
  contrib[n] = fr * u_pre_r + fz * u_pre_z;
}

__global__ void cap_energy_audit_reduce_sum_kernel(
    const double* __restrict__ in,
    double* __restrict__ block_sums,
    const int n) {
  __shared__ double s[kCapEnergyAuditReduceBlockSize];
  const int tid = threadIdx.x;
  const int base = 2 * blockIdx.x * blockDim.x;
  const int i0 = base + tid;
  const int i1 = i0 + blockDim.x;
  double value = 0.0;
  if (i0 < n) {
    value += in[i0];
  }
  if (i1 < n) {
    value += in[i1];
  }
  s[tid] = value;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s[tid] += s[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    block_sums[blockIdx.x] = s[0];
  }
}

__global__ void r_momentum_source_audit_node_momentum_kernel(
    double* __restrict__ momentum_r,
    const double* __restrict__ node_mass,
    const double* __restrict__ velocity_r,
    const int n_nodes) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_nodes) {
    return;
  }
  momentum_r[n] = node_mass[n] * velocity_r[n];
}

double cap_energy_audit_reduce_sum(const double* values, const int n) {
  if (n <= 0) {
    return 0.0;
  }
  const int blocks =
      (n + 2 * kCapEnergyAuditReduceBlockSize - 1) /
      (2 * kCapEnergyAuditReduceBlockSize);
  double* d_block_sums = nullptr;
  d_block_sums = static_cast<double*>(core::device_scratch_acquire(
      "h2d:cap_audit_block_sums",
      static_cast<std::size_t>(blocks) * sizeof(double)));
  cap_energy_audit_reduce_sum_kernel<<<blocks, kCapEnergyAuditReduceBlockSize>>>(
      values, d_block_sums, n);
  sync_kernel("Hydro2D cap-energy-audit reduce kernel failed");
  std::vector<double> host_block_sums(static_cast<std::size_t>(blocks), 0.0);
  cuda_check(cudaMemcpy(host_block_sums.data(),
                        d_block_sums,
                        static_cast<std::size_t>(blocks) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D cap-energy-audit reduce cudaMemcpy failed");
  long double total = 0.0L;
  for (const double value : host_block_sums) {
    total += static_cast<long double>(value);
  }
  return static_cast<double>(total);
}

CapEnergyAuditCompatibleTerms compute_cap_energy_audit_compatible_terms(
    const core::State& state,
    const double dt,
    const double* work_visc,
    const core::NodeField1D& work_velocity_r,
    const core::NodeField1D& work_velocity_z,
    const core::NodeField1D& force_half_r,
    const core::NodeField1D& force_half_z,
    const core::NodeField1D& old_velocity_r,
    const core::NodeField1D& old_velocity_z,
    const core::NodeField1D& node_mass,
    const std::uint8_t* d_node_active,
    const std::uint8_t* d_node_flags) {
  CapEnergyAuditCompatibleTerms out{};
  if (!(dt > 0.0) || state.rho.empty()) {
    return out;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(state.work_p_per_cell.size() == static_cast<std::size_t>(n_cells) &&
                    state.work_sub_per_cell.size() ==
                        static_cast<std::size_t>(n_cells) &&
                    state.work_av_per_cell.size() ==
                        static_cast<std::size_t>(n_cells),
                "Hydro2D cap-energy-audit compatible work buffers missing");
  TENRYU_ASSERT(force_half_r.size() == static_cast<std::size_t>(n_nodes) &&
                    force_half_z.size() == static_cast<std::size_t>(n_nodes) &&
                    work_velocity_r.size() == static_cast<std::size_t>(n_nodes) &&
                    work_velocity_z.size() == static_cast<std::size_t>(n_nodes) &&
                    old_velocity_r.size() == static_cast<std::size_t>(n_nodes) &&
                    old_velocity_z.size() == static_cast<std::size_t>(n_nodes) &&
                    node_mass.size() == static_cast<std::size_t>(n_nodes),
                "Hydro2D cap-energy-audit node buffers size mismatch");

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  core::CellField1D cell_contrib;
  cell_contrib.reset(static_cast<std::size_t>(n_cells));
  cap_energy_audit_cell_work_rate_kernel<<<cw.blocks(), 256>>>(
      cell_contrib.data(),
      state.work_p_per_cell.data(),
      state.work_sub_per_cell.data(),
      state.work_av_per_cell.data(),
      work_visc,
      cw.begin,
      cw.end,
      n_cells);
  sync_kernel("Hydro2D cap-energy-audit cell work-rate kernel failed");
  out.W_pq =
      dt * cap_energy_audit_reduce_sum(cell_contrib.data() + cw.begin, cw.count());

  core::NodeField1D node_contrib;
  node_contrib.reset(static_cast<std::size_t>(n_nodes));
  cap_energy_audit_nodal_work_rate_kernel<<<nw.blocks(), 256>>>(
      node_contrib.data(),
      force_half_r.data(),
      force_half_z.data(),
      work_velocity_r.data(),
      work_velocity_z.data(),
      nw.begin,
      nw.end,
      n_nodes);
  sync_kernel("Hydro2D cap-energy-audit nodal work-rate kernel failed");
  out.W_uF =
      dt * cap_energy_audit_reduce_sum(node_contrib.data() + nw.begin, nw.count());

  cap_energy_audit_apex_pinned_rate_kernel<<<nw.blocks(), 256>>>(
      node_contrib.data(),
      force_half_r.data(),
      force_half_z.data(),
      old_velocity_r.data(),
      old_velocity_z.data(),
      node_mass.data(),
      d_node_active,
      d_node_flags,
      nw.begin,
      nw.end,
      n_nodes,
      dt);
  sync_kernel("Hydro2D cap-energy-audit apex-pinned work-rate kernel failed");
  out.W_apex_pinned =
      dt * cap_energy_audit_reduce_sum(node_contrib.data() + nw.begin, nw.count());
  return out;
}

std::string format_scientific(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

template <typename Field>
std::vector<double> copy_field_to_vector(const Field& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

struct StructuredAwShadowProjectionStorage {
  core::DeviceArray<double> rho_axis_raw;
  core::DeviceArray<double> rho_partner_raw;
  core::DeviceArray<double> rho_projected;
  core::DeviceArray<double> merit;
  core::DeviceArray<double> x_merit;

  void reset(const std::size_t n_cells) {
    rho_axis_raw.reset(2U * n_cells);
    rho_partner_raw.reset(2U * n_cells);
    rho_projected.reset(2U * n_cells);
    merit.reset(n_cells);
    x_merit.reset(n_cells);
  }

  SubzonalPressureProjectionDebugBuffers device_buffers() {
    return SubzonalPressureProjectionDebugBuffers{
        rho_axis_raw.data(),
        rho_partner_raw.data(),
        rho_projected.data(),
        merit.data(),
        x_merit.data(),
    };
  }
};

bool emit_structured_aw_shadow_replay(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* const eos_ctx,
    const std::int8_t* const d_hydro_active,
    const bool aw_axis_slave_theta0_active,
    const bool aw_axis_slave_theta_pi_active,
    const StructuredAwShadowProjectionStorage& projection_storage) {
  constexpr double pi =
      3.141592653589793238462643383279502884;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_radial = nr * (nz + 1);
  const int n_edges = n_radial + (nr + 1) * nz;
  const std::size_t n_cells =
      static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) *
      static_cast<std::size_t>(nz + 1);
  if (nr <= 0 || nz <= 0 || state.rho.size() != n_cells ||
      state.x_r.size() != n_nodes || state.v_r.size() != n_nodes ||
      state.v_z.size() != n_nodes ||
      state.edge_force_av_r.size() != static_cast<std::size_t>(n_edges) ||
      state.edge_force_av_z.size() != static_cast<std::size_t>(n_edges) ||
      state.mass.size() != n_cells ||
      state.work_p_per_cell.size() != n_cells ||
      state.work_sub_per_cell.size() != n_cells ||
      state.work_av_per_cell.size() != n_cells) {
    core::log_warning(
        "[shadow-replay] skipped=invalid-structured-compatible-layout");
    return false;
  }

  sync_kernel("Hydro2D structured AW shadow replay pre-copy failed");
  const std::vector<double> edge_force_r =
      copy_field_to_vector(state.edge_force_av_r);
  const std::vector<double> edge_force_z =
      copy_field_to_vector(state.edge_force_av_z);
  const std::vector<double> velocity_r =
      copy_field_to_vector(state.v_r);
  const std::vector<double> velocity_z =
      copy_field_to_vector(state.v_z);
  const std::vector<double> x_r = copy_field_to_vector(state.x_r);
  const std::vector<double> cell_mass =
      copy_field_to_vector(state.mass);
  const std::vector<double> work_p =
      copy_field_to_vector(state.work_p_per_cell);
  const std::vector<double> work_sub =
      copy_field_to_vector(state.work_sub_per_cell);
  const std::vector<double> work_av =
      copy_field_to_vector(state.work_av_per_cell);

  struct AvClassAccum {
    std::size_t n_edges = 0U;
    long double sum_force_magnitude = 0.0L;
    long double edot_rz = 0.0L;
  };
  enum AvClass : int {
    Axisline = 0,
    Pq = 1,
    QqRadial = 2,
    Other = 3,
  };
  const std::array<const char*, 4> class_name = {
      "AXISLINE", "PQ", "QQ_RADIAL", "OTHER"};
  std::array<AvClassAccum, 4> av_class{};
  const int node_stride = nz + 1;
  for (int e = 0; e < n_edges; ++e) {
    int n0 = -1;
    int n1 = -1;
    AvClass edge_class = Other;
    if (e < n_radial) {
      const int i = e / (nz + 1);
      const int j = e - i * (nz + 1);
      n0 = i * node_stride + j;
      n1 = (i + 1) * node_stride + j;
      if ((aw_axis_slave_theta0_active && j == 0) ||
          (aw_axis_slave_theta_pi_active && j == nz)) {
        edge_class = Axisline;
      } else if (j == 1 || j == nz - 1) {
        edge_class = QqRadial;
      }
    } else {
      const int angular_edge = e - n_radial;
      const int i = angular_edge / nz;
      const int j = angular_edge - i * nz;
      n0 = i * node_stride + j;
      n1 = i * node_stride + j + 1;
      if (j == 0 || j == nz - 1) {
        edge_class = Pq;
      }
    }

    const double fr = edge_force_r[static_cast<std::size_t>(e)];
    const double fz = edge_force_z[static_cast<std::size_t>(e)];
    const double w0r =
        2.0 * pi * x_r[static_cast<std::size_t>(n0)];
    const double w1r =
        2.0 * pi * x_r[static_cast<std::size_t>(n1)];
    const double dv_r =
        w1r * velocity_r[static_cast<std::size_t>(n1)] -
        w0r * velocity_r[static_cast<std::size_t>(n0)];
    const double dv_z =
        w1r * velocity_z[static_cast<std::size_t>(n1)] -
        w0r * velocity_z[static_cast<std::size_t>(n0)];
    const double edge_edot = -(fr * dv_r + fz * dv_z);
    AvClassAccum& accum = av_class[static_cast<std::size_t>(edge_class)];
    ++accum.n_edges;
    accum.sum_force_magnitude +=
        static_cast<long double>(std::hypot(fr, fz));
    accum.edot_rz += static_cast<long double>(edge_edot);
  }
  for (std::size_t k = 0; k < av_class.size(); ++k) {
    std::ostringstream os;
    os << "[shadow-av] class=" << class_name[k]
       << " n_edges=" << av_class[k].n_edges
       << " sum|f|="
       << format_scientific(
              static_cast<double>(av_class[k].sum_force_magnitude))
       << " Edot_rz="
       << format_scientific(static_cast<double>(av_class[k].edot_rz));
    core::log_info(os.str());
  }
  core::log_info(
      "[shadow-av-identity] Edot_axisline=" +
      format_scientific(
          static_cast<double>(av_class[Axisline].edot_rz)));

  const std::array<int, 6> requested_columns = {
      0, 1, 2, nz - 3, nz - 2, nz - 1};
  std::vector<int> wedge_columns;
  for (const int j : requested_columns) {
    if (j >= 0 && j < nz &&
        std::find(wedge_columns.begin(), wedge_columns.end(), j) ==
            wedge_columns.end()) {
      wedge_columns.push_back(j);
    }
  }
  for (int i = std::max(0, nr - 3); i < nr; ++i) {
    for (const int j : wedge_columns) {
      const std::size_t c =
          static_cast<std::size_t>(i * nz + j);
      const double mass = cell_mass[c];
      const double inv_mass =
          mass != 0.0
              ? 1.0 / mass
              : std::numeric_limits<double>::quiet_NaN();
      std::ostringstream os;
      os << "[shadow-wedge] i=" << i
         << " j=" << j
         << " m_rz=" << format_scientific(mass)
         << " work_av=" << format_scientific(work_av[c])
         << " work_sub=" << format_scientific(work_sub[c])
         << " work_p=" << format_scientific(work_p[c])
         << " edot_av=" << format_scientific(work_av[c] * inv_mass)
         << " edot_sub=" << format_scientific(work_sub[c] * inv_mass)
         << " edot_p=" << format_scientific(work_p[c] * inv_mass);
      core::log_info(os.str());
    }
  }

  const SubzonalPressureShadowReplay subzonal_replay =
      compute_compatible_subzonal_pressure_shadow_replay_2d(
          state,
          cfg,
          eos_ctx,
          d_hydro_active,
          aw_axis_slave_theta0_active,
          aw_axis_slave_theta_pi_active);
  const auto subzonal_edot =
      [&](const int c,
          const std::vector<double>& force_r,
          const std::vector<double>& force_z) {
        const int i = c / nz;
        const int j = c - i * nz;
        const int nodes[4] = {
            i * node_stride + j,
            (i + 1) * node_stride + j,
            (i + 1) * node_stride + j + 1,
            i * node_stride + j + 1,
        };
        long double edot = 0.0L;
        for (int k = 0; k < 4; ++k) {
          const std::size_t corner =
              static_cast<std::size_t>(4 * c + k);
          const std::size_t node =
              static_cast<std::size_t>(nodes[k]);
          edot -=
              static_cast<long double>(2.0 * pi * x_r[node]) *
              static_cast<long double>(
                  force_r[corner] * velocity_r[node] +
                  force_z[corner] * velocity_z[node]);
        }
        return static_cast<double>(edot);
      };
  const std::array<int, 2> shadow_wedge_columns = {0, nz - 1};
  for (int i = 0; i < nr; ++i) {
    for (int pole = 0; pole < 2; ++pole) {
      const int j = shadow_wedge_columns[static_cast<std::size_t>(pole)];
      if (pole == 1 && j == shadow_wedge_columns[0]) {
        continue;
      }
      const int c = i * nz + j;
      const double axis_source_edot =
          subzonal_edot(c,
                        subzonal_replay.axis_source_force_r,
                        subzonal_replay.axis_source_force_z);
      const double offaxis_source_edot =
          subzonal_edot(c,
                        subzonal_replay.offaxis_source_force_r,
                        subzonal_replay.offaxis_source_force_z);
      std::ostringstream os;
      os << "[shadow-sz] i=" << i
         << " j=" << j
         << " Edot_axis_source="
         << format_scientific(axis_source_edot)
         << " Edot_offaxis_source="
         << format_scientific(offaxis_source_edot);
      core::log_info(os.str());
    }
  }

  std::vector<double> rho_axis_raw;
  std::vector<double> rho_partner_raw;
  std::vector<double> rho_projected;
  std::vector<double> merit;
  std::vector<double> x_merit;
  projection_storage.rho_axis_raw.copy_to_host(rho_axis_raw);
  projection_storage.rho_partner_raw.copy_to_host(rho_partner_raw);
  projection_storage.rho_projected.copy_to_host(rho_projected);
  projection_storage.merit.copy_to_host(merit);
  projection_storage.x_merit.copy_to_host(x_merit);
  const std::array<const char*, 2> pair_name = {"inner", "outer"};
  for (int i = 0; i < nr; ++i) {
    for (int pole = 0; pole < 2; ++pole) {
      const int j = shadow_wedge_columns[static_cast<std::size_t>(pole)];
      if (pole == 1 && j == shadow_wedge_columns[0]) {
        continue;
      }
      const std::size_t c =
          static_cast<std::size_t>(i * nz + j);
      for (int pair = 0; pair < 2; ++pair) {
        const std::size_t pair_index = 2U * c +
            static_cast<std::size_t>(pair);
        std::ostringstream os;
        os << "[shadow-rho] i=" << i
           << " j=" << j
           << " rho_axis_raw="
           << format_scientific(rho_axis_raw[pair_index])
           << " rho_partner_raw="
           << format_scientific(rho_partner_raw[pair_index])
           << " rho_projected="
           << format_scientific(rho_projected[pair_index])
           << " merit=" << format_scientific(merit[c])
           << " x_merit=" << format_scientific(x_merit[c])
           << " pair=" << pair_name[static_cast<std::size_t>(pair)];
        core::log_info(os.str());
      }
    }
  }
  return true;
}

bool shell_drive_sym_diag_env_flag_enabled(const char* const name) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "YES" || value == "on" ||
         value == "ON";
}

int shell_drive_sym_diag_env_int(const char* const name,
                                 const int fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw) {
    return fallback;
  }
  return static_cast<int>(parsed);
}

double shell_drive_sym_diag_env_double(const char* const name,
                                       const double fallback) {
  const char* const raw = std::getenv(name);
  if (raw == nullptr) {
    return fallback;
  }
  char* end = nullptr;
  const double parsed = std::strtod(raw, &end);
  if (end == raw || !std::isfinite(parsed)) {
    return fallback;
  }
  return parsed;
}

struct ShellDriveSymDiagConfig {
  bool enabled = false;
  int every = 1;
  int max_step = 100;
  double t_max = -1.0;
};

const ShellDriveSymDiagConfig& shell_drive_sym_diag_config() {
  static const ShellDriveSymDiagConfig cfg = [] {
    ShellDriveSymDiagConfig out{};
    out.enabled =
        shell_drive_sym_diag_env_flag_enabled("TENRYU_I1B_DRIVE_SYM_DIAG");
    out.every = std::max(1, shell_drive_sym_diag_env_int(
                                "TENRYU_I1B_DRIVE_SYM_DIAG_EVERY", 1));
    out.max_step =
        shell_drive_sym_diag_env_int("TENRYU_I1B_DRIVE_SYM_DIAG_MAX_STEP",
                                     100);
    out.t_max =
        shell_drive_sym_diag_env_double("TENRYU_I1B_DRIVE_SYM_DIAG_T_MAX_S",
                                        -1.0);
    return out;
  }();
  return cfg;
}

bool shell_drive_sym_diag_due(const core::State& state,
                              const core::Config& cfg) {
  const ShellDriveSymDiagConfig& diag = shell_drive_sym_diag_config();
  if (!diag.enabled || state.mesh.dim != 2 || cfg.main.dimension != "2D_RZ") {
    return false;
  }
  if (parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer) !=
      Boundary2DType::PRESSURE) {
    return false;
  }
  if (diag.max_step >= 0 && state.step > diag.max_step) {
    return false;
  }
  if (diag.t_max >= 0.0 && state.t > diag.t_max) {
    return false;
  }
  return (state.step % diag.every) == 0;
}

struct MeshForecastDiagConfig {
  bool enabled = false;
  int every = 1;
  double q_warn = 5.0e-2;
};

bool mesh_forecast_diag_env_enabled() {
  const char* const raw = std::getenv("TENRYU_I1B_MESH_FORECAST_DIAG");
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  const std::string value(raw);
  return value != "0" && value != "false" && value != "FALSE" &&
         value != "off" && value != "OFF" && value != "no" &&
         value != "NO";
}

const MeshForecastDiagConfig& mesh_forecast_diag_config() {
  static const MeshForecastDiagConfig cfg = [] {
    MeshForecastDiagConfig out{};
    out.enabled = mesh_forecast_diag_env_enabled();
    out.every = std::max(1, shell_drive_sym_diag_env_int(
                                "TENRYU_I1B_MESH_FORECAST_EVERY_N",
                                shell_drive_sym_diag_env_int(
                                    "TENRYU_I1B_MESH_FORECAST_DIAG", 1)));
    out.q_warn = std::max(
        1.0e-12,
        shell_drive_sym_diag_env_double("TENRYU_I1B_MESH_FORECAST_Q_WARN",
                                        out.q_warn));
    return out;
  }();
  return cfg;
}

bool mesh_forecast_diag_due(const core::State& state) {
  const MeshForecastDiagConfig& diag = mesh_forecast_diag_config();
  return diag.enabled && diag.every > 0 && (state.step % diag.every) == 0;
}

bool shell_rezone_replay_probe_requested() {
  const char* const raw = std::getenv("TENRYU_I1B_SHELL_REZONE_REPLAY");
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  const std::string value(raw);
  return value != "0" && value != "false" && value != "FALSE" &&
         value != "off" && value != "OFF" && value != "no" &&
         value != "NO";
}

bool shell_subcycle_requested() {
  const char* const raw = std::getenv("TENRYU_I1B_SHELL_SUBCYCLE");
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }
  const std::string value(raw);
  return value != "0" && value != "false" && value != "FALSE" &&
         value != "off" && value != "OFF" && value != "no" &&
         value != "NO";
}

bool mesh_forecast_or_shell_replay_due(const core::State& state) {
  return mesh_forecast_diag_due(state) || shell_rezone_replay_probe_requested();
}

bool mesh_forecast_post_commit_due(const core::State& state) {
  return mesh_forecast_or_shell_replay_due(state) || shell_subcycle_requested();
}

std::string mesh_forecast_bits_string(const std::uint32_t bits) {
  if (bits == 0U) {
    return "none";
  }
  std::string out;
  const auto append = [&](const std::uint32_t bit, const char* name) {
    if ((bits & bit) == 0U) {
      return;
    }
    if (!out.empty()) {
      out += "|";
    }
    out += name;
  };
  append(tenryu::mesh::kMeshForecastFailureQJ, "q_J");
  append(tenryu::mesh::kMeshForecastFailureQV, "q_V");
  append(tenryu::mesh::kMeshForecastFailureQEdge, "q_edge");
  append(tenryu::mesh::kMeshForecastFailureQR, "q_R");
  append(tenryu::mesh::kMeshForecastFailureQPhi, "q_phi");
  return out;
}

std::string mesh_forecast_component_record(
    const char* const prefix,
    const tenryu::mesh::MeshForecastComponents& q) {
  return std::string(", ") + prefix + "_q_J=" + format_scientific(q.q_J) +
         ", " + prefix + "_q_V=" + format_scientific(q.q_V) +
         ", " + prefix + "_q_edge=" + format_scientific(q.q_edge) +
         ", " + prefix + "_q_R=" + format_scientific(q.q_R) +
         ", " + prefix + "_q_phi=" + format_scientific(q.q_phi);
}

void emit_mesh_forecast_diag(
    const core::State& state,
    const char* const stage,
    const double path_dt,
    const tenryu::mesh::PathAdmissibilityResult& path_check,
    const tenryu::mesh::MeshForecast& forecast) {
  std::ostringstream oss;
  oss << "[mesh_forecast]"
      << " step=" << state.step
      << " t=" << format_scientific(state.t)
      << " stage=" << stage
      << " path_dt=" << format_scientific(path_dt)
      << " endpoint_valid=" << forecast.endpoint_valid
      << " tau_zero=" << format_scientific(forecast.tau_zero)
      << " tau_warn=" << format_scientific(forecast.tau_warn)
      << " q_warn=" << format_scientific(forecast.q_warn)
      << " q_min_now=" << format_scientific(forecast.q_min_now)
      << " q_min_end=" << format_scientific(forecast.q_min_end)
      << " q_min_path=" << format_scientific(forecast.q_min_path)
      << " first_cell=" << forecast.first_cell
      << " block=" << forecast.block_id
      << " local_i=" << forecast.local_i
      << " local_j=" << forecast.local_j
      << " failure_bits=" << forecast.failure_bits
      << " failure_components="
      << mesh_forecast_bits_string(forecast.failure_bits)
      << " seed_count=" << forecast.seed_count
      << " path_first_cell=" << path_check.first_failing_cell
      << " path_lambda="
      << format_scientific(path_check.first_failing_lambda)
      << " path_source="
      << tenryu::mesh::path_admissibility_source_kind_name(
             path_check.path_source_kind)
      << " path_metric="
      << tenryu::mesh::path_admissibility_metric_kind_name(
             path_check.first_failing_metric_kind)
      << mesh_forecast_component_record("now", forecast.worst_now)
      << mesh_forecast_component_record("warn", forecast.worst_warn)
      << mesh_forecast_component_record("end", forecast.worst_end)
      << " q_trace_tau0=" << format_scientific(forecast.q_trace_tau0)
      << " q_trace_tau25=" << format_scientific(forecast.q_trace_tau25)
      << " q_trace_tau50=" << format_scientific(forecast.q_trace_tau50)
      << " q_trace_tau75=" << format_scientific(forecast.q_trace_tau75)
      << " q_trace_tau100=" << format_scientific(forecast.q_trace_tau100);
  tenryu::core::log_info(oss.str());
}

void populate_mesh_forecast_result(
    tenryu::coupling::HydroStepResult& result,
    const tenryu::mesh::MeshForecast& forecast) {
  result.tau_warn = forecast.tau_warn;
  result.tau_zero = forecast.tau_zero;
  result.q_warn = forecast.q_warn;
  result.forecast_q_min_now = forecast.q_min_now;
  result.forecast_q_min_end = forecast.q_min_end;
  result.forecast_q_min_path = forecast.q_min_path;
  result.forecast_seed_count = forecast.seed_count;
  result.forecast_endpoint_valid = forecast.endpoint_valid != 0;
  result.forecast_first_cell = forecast.first_cell;
  result.forecast_block_id = forecast.block_id;
  result.forecast_local_i = forecast.local_i;
  result.forecast_local_j = forecast.local_j;
  result.forecast_seed_mask = forecast.seed_mask;
}

void observe_path_admissibility_margin(
    tenryu::coupling::HydroStepResult& result,
    const tenryu::mesh::PathAdmissibilityResult& path_check) {
  if (path_check.min_margin < result.min_path_margin) {
    result.min_path_margin = path_check.min_margin;
    result.min_path_margin_cell = path_check.min_margin_cell;
  }
}

struct ShellDriveSymNodeRow {
  bool valid = false;
  bool multiblock = false;
  int node_begin = 0;
  int stride = 0;
  int local_i = -1;
  int n_j = 0;
};

ShellDriveSymNodeRow shell_drive_sym_outer_node_row(
    const core::State& state,
    const core::Config& cfg) {
  ShellDriveSymNodeRow row{};
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    if (!state.mesh.topo.multiblock.has_value()) {
      return row;
    }
    const mesh::BlockInfo& shell =
        mesh::mesh_topo_multiblock_outermost_polar_shell_block(
            *state.mesh.topo.multiblock);
    if (shell.owned_node_begin < 0 || shell.n_i_cells <= 0 ||
        shell.n_j_cells <= 0) {
      return row;
    }
    row.valid = true;
    row.multiblock = true;
    row.node_begin =
        mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
            state.mesh.topo);
    row.stride = shell.n_j_cells + 1;
    row.local_i = shell.n_i_cells;
    row.n_j = shell.n_j_cells;
    return row;
  }

  if (state.mesh.topo.nr <= 0 || state.mesh.topo.nz <= 0) {
    return row;
  }
  row.valid = true;
  row.node_begin = 0;
  row.stride = state.mesh.topo.nz + 1;
  row.local_i = state.mesh.topo.nr;
  row.n_j = state.mesh.topo.nz;
  return row;
}

struct ShellDriveSymDiagSnapshot {
  bool valid = false;
  bool multiblock = false;
  int local_i = -1;
  int n_j = 0;
  int node_count = 0;
  double p_ext = 0.0;
  double eval_time = 0.0;
  double eps_tan = std::numeric_limits<double>::quiet_NaN();
  double eps_rad = std::numeric_limits<double>::quiet_NaN();
  double a_rad_mean = std::numeric_limits<double>::quiet_NaN();
  double a_rad_std = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> force_r;
  std::vector<double> force_z;
  std::vector<double> a_rad_bins;
};

void assemble_shell_drive_force_2d(core::NodeField1D& force_r,
                                   core::NodeField1D& force_z,
                                   const core::State& state,
                                   const core::Config& cfg,
                                   const double eval_time,
                                   double* p_ext_out) {
  force_r.reset(state.x_r.size());
  force_z.reset(state.x_r.size());
  if (p_ext_out != nullptr) {
    *p_ext_out = 0.0;
  }
  if (state.x_r.empty() ||
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer) !=
          Boundary2DType::PRESSURE) {
    return;
  }
  TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                "shell_drive_sym pressure boundary requires pressure table");
  const double p_ext = state.pressure_drive_1d->eval(eval_time);
  if (p_ext_out != nullptr) {
    *p_ext_out = p_ext;
  }
  const auto& pcfg =
      cfg.numerics.hydro.pressure_drive_perturbation;
  PressureDrivePerturbationParams pp{};
  if (pcfg.enabled) {
    pp = core::make_pressure_drive_perturbation_params(pcfg);
  }
  const bool rz_exact_endpoint =
      state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane;
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    mesh::mesh_topo_assert_multiblock_outermost_polar_shell_outer_boundary(
        state.mesh.topo);
    core::NodeField1D diag_node_mass;
    if (state.corner_mass_initialized &&
        !state.mesh.multiblock_reverse_csr_node_offsets.empty()) {
      diag_node_mass.reset(state.x_r.size());
      detail::launch_compute_node_mass_2d_multiblock(
          diag_node_mass.data(),
          state.corner_mass.data(),
          state.mass.data(),
          state.x_r.data(),
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          state.mesh.multiblock_reverse_csr_node_cells.data(),
          state.mesh.multiblock_reverse_csr_node_corners.data(),
          static_cast<int>(state.x_r.size()),
          equal_split_node_mass_mode(
              cfg.numerics.hydro.axis_node_mass_convention),
          state.corner_stride);
    }
    detail::launch_multiblock_polar_shell_pressure_forces(
        force_r.data(),
        force_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
            state.mesh.topo),
        mesh::mesh_topo_multiblock_outermost_polar_shell_nr(state.mesh.topo),
        mesh::mesh_topo_multiblock_outermost_polar_shell_nz(state.mesh.topo),
        p_ext,
        rz_exact_endpoint,
        diag_node_mass.empty() ? nullptr : diag_node_mass.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id,
        pp,
        pcfg.enabled);
  } else {
    detail::launch_r_outer_boundary_pressure_forces(
        force_r.data(),
        force_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        p_ext,
        rz_exact_endpoint,
        cfg.numerics.hydro.rz_momentum_scheme_id,
        pp,
        pcfg.enabled);
  }
  sync_kernel("Hydro2D shell_drive_sym drive-force assembly failed");
}

}  // namespace

__global__ void evac_contact_row_hold_sweep_test_kernel(
    double* v_r,
    double* v_z,
    const double* node_mass_resolved,
    const int* row_nodes,
    const double* row_xi,
    const double* row_normal,
    double* row_dk,
    int n_rows) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    evac_contact_row_hold_sweep(
        v_r, v_z, node_mass_resolved, row_nodes, row_xi, row_normal, row_dk,
        n_rows);
  }
}

void launch_evac_contact_row_hold_sweep_for_test(
    double* const d_v_r,
    double* const d_v_z,
    const double* const d_node_mass,
    const int* const d_row_nodes,
    const double* const d_row_xi,
    const double* const d_row_normal,
    double* const d_row_dk,
    const int n_rows) {
  evac_contact_row_hold_sweep_test_kernel<<<1, 1>>>(
      d_v_r, d_v_z, d_node_mass, d_row_nodes, d_row_xi, d_row_normal,
      d_row_dk, n_rows);
  cuda_check(cudaDeviceSynchronize(),
             "Hydro2D evacuated contact row-hold test kernel failed");
}

__global__ void evac_contact_union_velocity_sweep_test_kernel(
    double* v_r,
    double* v_z,
    const double* node_mass_resolved,
    const int* pair_nodes,
    const double* pair_normal,
    const std::uint8_t* pair_row_covered,
    double* pair_dk,
    int n_pairs,
    const int* row_nodes,
    const double* row_xi,
    const double* row_normal,
    double* row_dk,
    int n_rows) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    evac_contact_union_velocity_sweep(
        v_r, v_z, node_mass_resolved, pair_nodes, pair_normal,
        pair_row_covered, pair_dk, n_pairs, row_nodes, row_xi, row_normal,
        row_dk, n_rows);
  }
}

void launch_evac_contact_union_velocity_sweep_for_test(
    double* const d_v_r,
    double* const d_v_z,
    const double* const d_node_mass,
    const int* const d_pair_nodes,
    const double* const d_pair_normal,
    const std::uint8_t* const d_pair_row_covered,
    double* const d_pair_dk,
    const int n_pairs,
    const int* const d_row_nodes,
    const double* const d_row_xi,
    const double* const d_row_normal,
    double* const d_row_dk,
    const int n_rows) {
  evac_contact_union_velocity_sweep_test_kernel<<<1, 1>>>(
      d_v_r, d_v_z, d_node_mass, d_pair_nodes, d_pair_normal,
      d_pair_row_covered, d_pair_dk, n_pairs, d_row_nodes, d_row_xi,
      d_row_normal, d_row_dk, n_rows);
  cuda_check(cudaDeviceSynchronize(),
             "Hydro2D evacuated contact union test kernel failed");
}

void launch_apply_compatible_energy_work_2d(
    core::State& state,
    const core::Config& cfg,
    const core::CellField1D& Pe_half,
    const core::CellField1D& Pi_half,
    const bool use_two_temp,
    const double dt,
    const std::int8_t* const hydro_active,
    double* const E_floor_injected,
    int* const clamp_count,
    const bool force_enabled) {
  apply_compatible_energy_work_2d(
      state, cfg, force_enabled, Pe_half, Pi_half, use_two_temp, dt, hydro_active,
      E_floor_injected, clamp_count);
}

void launch_update_node_velocity_2d(
    double* const v_r,
    double* const v_z,
    const double* const old_v_r,
    const double* const old_v_z,
    const double* const a_r,
    const double* const a_z,
    const std::uint8_t* const node_active,
    const std::uint8_t* const node_flags,
    const int n_nodes,
    const double scale_dt,
    core::State& state,
    const core::Config& cfg) {
  double* node_accel_r = nullptr;
  double* node_accel_z = nullptr;
  if (cfg.numerics.dt.edge_accel_displacement_cfl_enabled) {
    const std::size_t node_count = state.v_r.size();
    if (state.node_accel_r.size() != node_count ||
        state.node_accel_z.size() != node_count) {
      state.node_accel_r.reset(node_count);
      state.node_accel_z.reset(node_count);
      state.node_accel_r.fill(0.0);
      state.node_accel_z.fill(0.0);
      // The first initialized state is zero, so before an acceleration sample
      // is stored the guard reduces to its velocity-only displacement bound.
    }
    node_accel_r = state.node_accel_r.data();
    node_accel_z = state.node_accel_z.data();
  }
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  update_node_velocity_2d_kernel<<<nw.blocks(), 256>>>(
      v_r, v_z, old_v_r, old_v_z, a_r, a_z, node_active, node_flags, nw.begin,
      nw.end, n_nodes, scale_dt, node_accel_r, node_accel_z);
}

void launch_apply_aw_axis_velocity_slave_2d(
    double* const v_r,
    double* const v_z,
    const double* const x_r,
    const double* const x_z,
    double* const kappa_saved,
    const int nr,
    const int nz,
    const int first_axis_i,
    const bool theta0_axis_active,
    const bool theta_pi_axis_active,
    const bool capture_kappa) {
  if (first_axis_i < 0 || first_axis_i > nr || nz < 2 ||
      (!theta0_axis_active && !theta_pi_axis_active)) {
    return;
  }
  const int axis_node_count = nr - first_axis_i + 1;
  const int blocks = (axis_node_count + 255) / 256;
  apply_aw_axis_velocity_slave_2d_kernel<<<blocks, 256>>>(
      v_r, v_z, x_r, x_z, kappa_saved, nr, nz, first_axis_i,
      theta0_axis_active, theta_pi_axis_active, capture_kappa);
}

void launch_snap_aw_axis_coordinates_2d(
    double* const x_r,
    double* const x_z,
    const double* const kappa_saved,
    const int nr,
    const int nz,
    const int first_axis_i,
    const bool theta0_axis_active,
    const bool theta_pi_axis_active) {
  if (first_axis_i < 0 || first_axis_i > nr || nz < 2 ||
      (!theta0_axis_active && !theta_pi_axis_active)) {
    return;
  }
  const int axis_node_count = nr - first_axis_i + 1;
  const int blocks = (axis_node_count + 255) / 256;
  snap_aw_axis_coordinates_2d_kernel<<<blocks, 256>>>(
      x_r, x_z, kappa_saved, nr, nz, first_axis_i, theta0_axis_active,
      theta_pi_axis_active);
}

AwAxisSlaveMasterList build_aw_axis_slave_master_list_multiblock(
    const core::State& state,
    const core::Config& cfg) {
  AwAxisSlaveMasterList master;
  if (!state.mesh.topo.multiblock.has_value() ||
      !mesh::mesh_topo_polar_tier_family(cfg.mesh)) {
    return master;
  }

  const mesh::MultiBlockTopology& mb = *state.mesh.topo.multiblock;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_aw_axis_prefix_nodes =
      mesh::mesh_topo_has_polar_tier_cart_center(cfg.mesh)
          ? mesh::mesh_topo_n_aw_axis_prefix_nodes(mb)
          : n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  const int corner_stride = state.corner_stride;
  TENRYU_ASSERT(n_nodes >= 0 && n_cells >= 0,
                "AW axis master list requires non-negative topology sizes");
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(n_nodes),
                "AW axis master list requires node flags");
  TENRYU_ASSERT(state.x_r_initial.size() ==
                    static_cast<std::size_t>(n_nodes),
                "AW axis master list requires all initial node coordinates");
  TENRYU_ASSERT(state.corner_mass_initialized &&
                    state.corner_mass.size() ==
                        static_cast<std::size_t>(n_cells) *
                            static_cast<std::size_t>(corner_stride),
                "AW axis master list requires initialized corner masses");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U &&
                    mb.cell_node_csr_indices.size() ==
                        static_cast<std::size_t>(n_cells) *
                            static_cast<std::size_t>(corner_stride),
                "AW axis master list requires stride-sized cell-node CSR");

  std::vector<double> host_initial_r(
      static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> corner_mass(
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(corner_stride),
      0.0);
  state.x_r_initial.copy_to_host(host_initial_r.data());
  state.corner_mass.copy_to_host(corner_mass.data());

  std::vector<double> planar_mass(static_cast<std::size_t>(n_nodes), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int nverts =
        mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner)];
      const double contribution =
          corner_mass[static_cast<std::size_t>(c) *
                          static_cast<std::size_t>(corner_stride) +
                      static_cast<std::size_t>(corner)];
      TENRYU_ASSERT(node >= 0 && node < n_nodes,
                    "AW axis master list CSR node out of range");
      TENRYU_ASSERT(std::isfinite(contribution) && contribution >= 0.0,
                    "AW axis master list corner mass must be finite non-negative");
      planar_mass[static_cast<std::size_t>(node)] += contribution;
    }
  }

  std::vector<std::vector<int>> candidates(
      static_cast<std::size_t>(n_nodes));
  const auto add_candidate = [&](const int p, const int q) {
    TENRYU_ASSERT(p >= 0 && p < n_nodes && q >= 0 && q < n_nodes,
                  "AW axis master candidate node out of range");
    if (p >= n_aw_axis_prefix_nodes) {
      return;
    }
    const std::uint8_t flags =
        state.mesh.topo.node_flags[static_cast<std::size_t>(p)];
    if ((flags & mesh::NODE_POLE_AXIS) == 0U ||
        (flags & mesh::NODE_CENTER) != 0U ||
        host_initial_r[static_cast<std::size_t>(p)] != 0.0) {
      return;
    }
    TENRYU_ASSERT(host_initial_r[static_cast<std::size_t>(q)] > 0.0,
                  "AW axis master candidate must be first off-axis node");
    candidates[static_cast<std::size_t>(p)].push_back(q);
  };

  for (const mesh::BlockInfo& block : mb.blocks) {
    if (block.role != mesh::BlockRole::POLAR_SHELL &&
        block.role != mesh::BlockRole::POLAR_TIER) {
      continue;
    }
    TENRYU_ASSERT(block.n_i_cells > 0 && block.n_j_cells >= 2 &&
                      block.cell_count ==
                          block.n_i_cells * block.n_j_cells,
                  "AW axis master list requires structured polar block dimensions");
    for (int i = 0; i < block.n_i_cells; ++i) {
      const int north_cell =
          block.cell_begin + i * block.n_j_cells;
      const int south_cell =
          north_cell + block.n_j_cells - 1;
      const int north_off =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(north_cell)];
      const int south_off =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(south_cell)];
      const auto north = [&](const int corner) {
        return mb.cell_node_csr_indices[
            static_cast<std::size_t>(north_off + corner)];
      };
      const auto south = [&](const int corner) {
        return mb.cell_node_csr_indices[
            static_cast<std::size_t>(south_off + corner)];
      };
      add_candidate(north(0), north(1));
      add_candidate(north(3), north(2));
      add_candidate(south(1), south(0));
      add_candidate(south(2), south(3));
    }
  }

  for (std::size_t block_id = 0; block_id < mb.blocks.size(); ++block_id) {
    const mesh::BlockInfo& block = mb.blocks[block_id];
    if ((block.role != mesh::BlockRole::TRANSITION_BELT &&
         block.role != mesh::BlockRole::BRIDGE) ||
        (block.role == mesh::BlockRole::TRANSITION_BELT &&
         block.n_i_cells < 2)) {
      continue;
    }
    for (int cell = block.cell_begin;
         cell < block.cell_begin + block.cell_count; ++cell) {
      const int nverts =
          mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
      const int off =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      for (int k = 0; k < nverts; ++k) {
        const int p =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (host_initial_r[static_cast<std::size_t>(p)] != 0.0) {
          continue;
        }
        // For a structured quad with its theta=0 edge on the axis, winding
        // adjacency reproduces exactly (0,1)/(3,2) north and (1,0)/(2,3)
        // south pairs, extending the same geometric rule to belt polygons.
        const int next =
            mb.cell_node_csr_indices[
                static_cast<std::size_t>(off + (k + 1) % nverts)];
        if (host_initial_r[static_cast<std::size_t>(next)] > 0.0) {
          add_candidate(p, next);
        }
        const int prev =
            mb.cell_node_csr_indices[static_cast<std::size_t>(
                off + (k - 1 + nverts) % nverts)];
        if (host_initial_r[static_cast<std::size_t>(prev)] > 0.0) {
          add_candidate(p, prev);
        }
      }
    }
  }

  std::vector<int> fallback_axis_nodes;
  for (int p = 0; p < n_aw_axis_prefix_nodes; ++p) {
    const std::uint8_t flags =
        state.mesh.topo.node_flags[static_cast<std::size_t>(p)];
    if ((flags & mesh::NODE_POLE_AXIS) != 0U &&
        (flags & mesh::NODE_CENTER) == 0U &&
        host_initial_r[static_cast<std::size_t>(p)] == 0.0 &&
        candidates[static_cast<std::size_t>(p)].empty()) {
      fallback_axis_nodes.push_back(p);
    }
  }

  for (std::size_t block_id = 0; block_id < mb.blocks.size(); ++block_id) {
    const mesh::BlockInfo& block = mb.blocks[block_id];
    if (block.role != mesh::BlockRole::TRANSITION_BELT ||
        block.n_i_cells != 1) {
      continue;
    }
    for (int cell = block.cell_begin;
         cell < block.cell_begin + block.cell_count; ++cell) {
      const int nverts =
          mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
      const int off =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      for (int k = 0; k < nverts; ++k) {
        const int p =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (host_initial_r[static_cast<std::size_t>(p)] != 0.0 ||
            !std::binary_search(fallback_axis_nodes.begin(),
                                fallback_axis_nodes.end(), p)) {
          continue;
        }
        const int next =
            mb.cell_node_csr_indices[
                static_cast<std::size_t>(off + (k + 1) % nverts)];
        if (host_initial_r[static_cast<std::size_t>(next)] > 0.0) {
          add_candidate(p, next);
        }
        const int prev =
            mb.cell_node_csr_indices[static_cast<std::size_t>(
                off + (k - 1 + nverts) % nverts)];
        if (host_initial_r[static_cast<std::size_t>(prev)] > 0.0) {
          add_candidate(p, prev);
        }
      }
    }
  }

  for (int p = 0; p < n_aw_axis_prefix_nodes; ++p) {
    const std::uint8_t flags =
        state.mesh.topo.node_flags[static_cast<std::size_t>(p)];
    if ((flags & mesh::NODE_POLE_AXIS) == 0U ||
        (flags & mesh::NODE_CENTER) != 0U ||
        host_initial_r[static_cast<std::size_t>(p)] != 0.0) {
      continue;
    }
    std::vector<int>& peer = candidates[static_cast<std::size_t>(p)];
    std::sort(peer.begin(), peer.end());
    peer.erase(std::unique(peer.begin(), peer.end()), peer.end());
    TENRYU_ASSERT(!peer.empty(),
                  "AW axis master list found axis node without angular peer");
    int best_q = peer.front();
    double best_mass = planar_mass[static_cast<std::size_t>(best_q)];
    for (const int q : peer) {
      const double mass = planar_mass[static_cast<std::size_t>(q)];
      if (mass > best_mass || (mass == best_mass && q < best_q)) {
        best_q = q;
        best_mass = mass;
      }
    }
    master.p.push_back(p);
    master.q.push_back(best_q);
  }
  return master;
}

void launch_apply_aw_axis_velocity_slave_2d_multiblock(
    double* const v_r,
    double* const v_z,
    const double* const x_r,
    const double* const x_z,
    double* const kappa_saved,
    const int* const axis_slave_p,
    const int* const axis_slave_q,
    const int pair_count,
    const bool capture_kappa) {
  if (pair_count <= 0) {
    return;
  }
  const int blocks = (pair_count + 255) / 256;
  apply_aw_axis_velocity_slave_2d_multiblock_kernel<<<blocks, 256>>>(
      v_r, v_z, x_r, x_z, kappa_saved, axis_slave_p, axis_slave_q,
      pair_count, capture_kappa);
}

void launch_snap_aw_axis_coordinates_2d_multiblock(
    double* const x_r,
    double* const x_z,
    const double* const kappa_saved,
    const int* const axis_slave_p,
    const int* const axis_slave_q,
    const int pair_count) {
  if (pair_count <= 0) {
    return;
  }
  const int blocks = (pair_count + 255) / 256;
  snap_aw_axis_coordinates_2d_multiblock_kernel<<<blocks, 256>>>(
      x_r, x_z, kappa_saved, axis_slave_p, axis_slave_q, pair_count);
}

// Verdict-#7 E2: one-step pressure-force / geometry audit. On the CURRENT
// mesh, assemble the drive force at a plateau pressure and test the
// work-conjugacy identities that a spherical surface must satisfy:
//   radial     sum f.(-x_hat)      = P * A         (A = 4 pi S^2)
//   homologous sum f.(-x)          = 3 P V_encl    (V = 4 pi/3 S^3)
//   net z      sum f_z             = 0
// A constant convention error in areas/normals shows directly as a ratio.
void run_drive_geometry_audit(const core::State& state,
                              const core::Config& cfg) {
  static const double audit_t = [] {
    const char* raw = std::getenv("TENRYU_I1B_DRIVE_GEOM_AUDIT_T");
    const double v = raw != nullptr && raw[0] != '\0' ? std::atof(raw) : 0.0;
    return (std::isfinite(v) && v > 0.0) ? v : 5.0e-10;
  }();
  core::NodeField1D force_r;
  core::NodeField1D force_z;
  double p_ext = 0.0;
  assemble_shell_drive_force_2d(force_r, force_z, state, cfg, audit_t,
                                &p_ext);
  if (force_r.empty() || !(p_ext > 0.0)) {
    std::fprintf(stderr,
                 "[drive_geom_audit] SKIP (no force or p_ext=%.3e at "
                 "t=%.3e)\n",
                 p_ext,
                 audit_t);
    return;
  }
  std::vector<double> fr;
  std::vector<double> fz;
  std::vector<double> xr;
  std::vector<double> xz;
  force_r.copy_to_host(fr);
  force_z.copy_to_host(fz);
  state.x_r.copy_to_host(xr);
  state.x_z.copy_to_host(xz);
  std::vector<double> vol_h;
  state.vol.copy_to_host(vol_h);
  long double v_domain = 0.0L;
  for (const double v : vol_h) {
    v_domain += v;
  }
  long double s_max = 0.0L;
  long double sum_f_radial = 0.0L;   // sum f . (-x_hat)
  long double sum_f_homolog = 0.0L;  // sum f . (-x)
  long double sum_fz = 0.0L;
  long double sum_fmag = 0.0L;
  int loaded_nodes = 0;
  const std::size_t n = std::min(fr.size(), xr.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double fmag = std::hypot(fr[i], fz[i]);
    if (fmag <= 0.0) {
      continue;
    }
    ++loaded_nodes;
    const double rr = std::hypot(xr[i], xz[i]);
    if (rr > s_max) {
      s_max = rr;
    }
    if (rr > 0.0) {
      sum_f_radial += -(fr[i] * xr[i] + fz[i] * xz[i]) / rr;
    }
    sum_f_homolog += -(fr[i] * xr[i] + fz[i] * xz[i]);
    sum_fz += fz[i];
    sum_fmag += fmag;
  }
  // Per-node audit (verdict-#7 / Addendum-13): compare each loaded node's
  // assembled inward-normal force with P times the node's GEOMETRIC ring
  // area (the revolved half-edges adjacent to the node along the outer
  // polyline, Pappus: A_edge = pi*(r_a+r_b)*L_edge, half to each node). A
  // convention error at the axis endpoints appears as a per-node ratio
  // away from 1 even when the integral identities are exact.
  {
    std::vector<std::size_t> loaded;
    for (std::size_t i = 0; i < n; ++i) {
      if (std::hypot(fr[i], fz[i]) > 0.0) {
        loaded.push_back(i);
      }
    }
    // Order the outer-surface nodes by polar angle theta = atan2(r, z).
    std::sort(loaded.begin(), loaded.end(),
              [&](const std::size_t a, const std::size_t b) {
                return std::atan2(xr[a], xz[a]) < std::atan2(xr[b], xz[b]);
              });
    // Nodal masses (the same corner-mass basis the momentum update uses):
    // print force/mass per node so an attribution INCONSISTENCY between the
    // FE-consistent force weights and the corner-mass weights shows up as
    // per-node acceleration structure at t=0.
    std::vector<double> nm;
    if (state.mesh.topo.multiblock.has_value() &&
        state.corner_mass_initialized &&
        !state.mesh.multiblock_reverse_csr_node_offsets.empty()) {
      core::NodeField1D node_mass;
      node_mass.reset(state.x_r.size());
      detail::launch_compute_node_mass_2d_multiblock(
          node_mass.data(),
          state.corner_mass.data(),
          state.mass.data(),
          state.x_r.data(),
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          state.mesh.multiblock_reverse_csr_node_cells.data(),
          state.mesh.multiblock_reverse_csr_node_corners.data(),
          static_cast<int>(state.x_r.size()),
          equal_split_node_mass_mode(
              cfg.numerics.hydro.axis_node_mass_convention),
          state.corner_stride);
      node_mass.copy_to_host(nm);
    }
    std::ostringstream per_node;
    per_node << "[drive_geom_audit_node] ratios=[";
    for (std::size_t k = 0; k < loaded.size(); ++k) {
      const std::size_t i = loaded[k];
      double a_geom = 0.0;
      if (k > 0) {
        const std::size_t p1 = loaded[k - 1];
        const double L = std::hypot(xr[i] - xr[p1], xz[i] - xz[p1]);
        a_geom += 0.5 * M_PI * (xr[i] + xr[p1]) * L;
      }
      if (k + 1 < loaded.size()) {
        const std::size_t p2 = loaded[k + 1];
        const double L = std::hypot(xr[p2] - xr[i], xz[p2] - xz[i]);
        a_geom += 0.5 * M_PI * (xr[i] + xr[p2]) * L;
      }
      const double f_n = std::hypot(fr[i], fz[i]);
      const double ratio =
          a_geom > 0.0 ? f_n / (p_ext * a_geom) : -1.0;
      if (k > 0) {
        per_node << ",";
      }
      per_node << k << ":" << ratio;
    }
    per_node << "]";
    std::fprintf(stderr, "%s\n", per_node.str().c_str());
    if (!nm.empty()) {
      std::ostringstream accel;
      accel << "[drive_geom_audit_accel] f_over_m=[";
      double a_sum = 0.0;
      int a_n = 0;
      std::vector<double> a_list(loaded.size(), 0.0);
      for (std::size_t k = 0; k < loaded.size(); ++k) {
        const std::size_t i = loaded[k];
        const double m_n = i < nm.size() ? nm[i] : 0.0;
        const double a =
            m_n > 0.0 ? std::hypot(fr[i], fz[i]) / m_n : -1.0;
        a_list[k] = a;
        if (a > 0.0) {
          a_sum += a;
          ++a_n;
        }
      }
      const double a_mean = a_n > 0 ? a_sum / a_n : 1.0;
      for (std::size_t k = 0; k < loaded.size(); ++k) {
        if (k > 0) {
          accel << ",";
        }
        accel << k << ":" << a_list[k] / a_mean;
      }
      accel << "] a_mean=" << a_mean;
      std::fprintf(stderr, "%s\n", accel.str().c_str());
    }
  }
  const double S = static_cast<double>(s_max);
  const double A_exact = 4.0 * M_PI * S * S;
  const double V_exact = 4.0 * M_PI / 3.0 * S * S * S;
  const double A_eff = static_cast<double>(sum_f_radial) / p_ext;
  const double V3_eff = static_cast<double>(sum_f_homolog) / p_ext;
  std::fprintf(stderr,
               "[drive_geom_audit] t=%.3e p_ext=%.4e loaded_nodes=%d "
               "S_h=%.10e V_domain=%.10e V_exact=%.10e V_ratio=%.6f "
               "A_eff=%.10e A_exact=%.10e A_ratio=%.6f "
               "homolog_eff=%.10e 3PV_exact_norm=%.6f "
               "net_fz_rel=%.3e\n",
               audit_t,
               p_ext,
               loaded_nodes,
               S,
               static_cast<double>(v_domain),
               V_exact,
               static_cast<double>(v_domain) / V_exact,
               A_eff,
               A_exact,
               A_eff / A_exact,
               V3_eff,
               V3_eff / (3.0 * V_exact),
               static_cast<double>(sum_fz) /
                   std::max(static_cast<double>(sum_fmag), 1.0e-300));
}

// Verdict-#7 E6 discriminator (env TENRYU_I1B_UTHETA_SUPPRESS, default
// off; PROBE ONLY): after each committed step, project every node velocity
// onto the spherical radial direction, v <- (v.r_hat) r_hat. If the +42%
// drive-work excess collapses under this projection, the excess is the
// smeared surface's 2D lateral-compliance dynamics (R2, physical); if it
// persists, the radial scheme (AV, R3) is implicated. The removed
// tangential KE is accumulated and logged (an explicit energy sink — this
// mode is NOT a production configuration).
void suppress_tangential_velocity(core::State& state, const double t) {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_UTHETA_SUPPRESS");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  if (!enabled) {
    return;
  }
  std::vector<double> vr;
  std::vector<double> vz;
  std::vector<double> xr;
  std::vector<double> xz;
  state.v_r.copy_to_host(vr);
  state.v_z.copy_to_host(vz);
  state.x_r.copy_to_host(xr);
  state.x_z.copy_to_host(xz);
  static long double ke_removed_cum = 0.0L;
  static long long calls = 0;
  for (std::size_t i = 0; i < vr.size(); ++i) {
    const double rr = std::hypot(xr[i], xz[i]);
    if (!(rr > 0.0)) {
      continue;
    }
    const double er_r = xr[i] / rr;
    const double er_z = xz[i] / rr;
    const double v_rad = vr[i] * er_r + vz[i] * er_z;
    const double v2_before = vr[i] * vr[i] + vz[i] * vz[i];
    vr[i] = v_rad * er_r;
    vz[i] = v_rad * er_z;
    ke_removed_cum += 0.5 * (v2_before - v_rad * v_rad);
  }
  state.v_r.copy_from_host(vr);
  state.v_z.copy_from_host(vz);
  ++calls;
  if (calls % 500 == 0) {
    std::fprintf(stderr,
                 "[utheta_suppress] step_calls=%lld t=%.4e "
                 "specific_ke_removed_cum=%.6e\n",
                 calls,
                 t,
                 static_cast<double>(ke_removed_cum));
  }
}

void accumulate_drive_work_ledger(const core::State& state,
                                  const core::Config& cfg,
                                  const double dt,
                                  const double t_mid) {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_DRIVE_WORK_LEDGER");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  if (!enabled) {
    return;
  }
  // Robust step-size: subcycle-committed steps can reach the epilogue with
  // state.dt unset/zero; integrate on the actual advance of state.t.
  static double prev_t = -1.0;
  double dt_used = dt;
  if (!(dt_used > 0.0)) {
    dt_used = prev_t >= 0.0 ? state.t - prev_t : 0.0;
  }
  prev_t = state.t;
  if (!(dt_used > 0.0)) {
    return;
  }
  static const int log_every = [] {
    const char* raw = std::getenv("TENRYU_I1B_DRIVE_WORK_LEDGER_EVERY");
    const int v = raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
    return v > 0 ? v : 200;
  }();
  static const bool theta_enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_DRIVE_WORK_LEDGER_THETA");
    return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
  }();
  core::NodeField1D force_r;
  core::NodeField1D force_z;
  double p_ext = 0.0;
  assemble_shell_drive_force_2d(force_r, force_z, state, cfg, t_mid, &p_ext);
  if (force_r.empty()) {
    return;
  }
  std::vector<double> fr;
  std::vector<double> fz;
  std::vector<double> vr;
  std::vector<double> vz;
  force_r.copy_to_host(fr);
  force_z.copy_to_host(fz);
  state.v_r.copy_to_host(vr);
  state.v_z.copy_to_host(vz);
  // Midpoint velocity: end-of-step velocities systematically OVERSHOOT the
  // work integral during the strong drive acceleration (measured +48% vs
  // the 1D reference with end-of-step dotting). Cache the previous step's
  // end velocities as this step's start (exact between hydro steps; ALE
  // velocity remap perturbs it only off the pinned outer row).
  static std::vector<double> vr_prev;
  static std::vector<double> vz_prev;
  const bool have_prev =
      vr_prev.size() == vr.size() && vz_prev.size() == vz.size();
  static std::vector<long double> w_theta;
  static long double w_theta_ovf = 0.0L;
  long long theta_base = -1;
  long long theta_cols = 0;
  if (theta_enabled) {
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      const int begin =
          mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
              state.mesh.topo);
      const int nr_shell =
          mesh::mesh_topo_multiblock_outermost_polar_shell_nr(state.mesh.topo);
      const int nz_polar =
          mesh::mesh_topo_multiblock_outermost_polar_shell_nz(state.mesh.topo);
      theta_base = static_cast<long long>(begin) +
                   static_cast<long long>(nr_shell) * (nz_polar + 1);
      theta_cols = nz_polar + 1;
    } else {
      theta_base = static_cast<long long>(state.mesh.topo.nr) *
                   (state.mesh.topo.nz + 1);
      theta_cols = state.mesh.topo.nz + 1;
    }
    if (w_theta.size() != static_cast<std::size_t>(theta_cols)) {
      w_theta.assign(static_cast<std::size_t>(theta_cols), 0.0L);
    }
  }
  long double w_step = 0.0L;
  const std::size_t n = std::min(fr.size(), vr.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double vmr = have_prev ? 0.5 * (vr[i] + vr_prev[i]) : vr[i];
    const double vmz = have_prev ? 0.5 * (vz[i] + vz_prev[i]) : vz[i];
    const long double term = (static_cast<long double>(fr[i]) * vmr +
                              static_cast<long double>(fz[i]) * vmz) *
                             dt_used;
    w_step += term;
    if (theta_enabled && (fr[i] != 0.0 || fz[i] != 0.0)) {
      const long long col = static_cast<long long>(i) - theta_base;
      if (col >= 0 && col < theta_cols) {
        w_theta[static_cast<std::size_t>(col)] += term;
      } else {
        w_theta_ovf += term;
      }
    }
  }
  vr_prev = vr;
  vz_prev = vz;
  static long double w_cum = 0.0L;
  static long long calls = 0;
  w_cum += w_step;
  ++calls;
  if (calls == 1) {
    run_drive_geometry_audit(state, cfg);
  }
  if (calls % log_every == 0) {
    // Work-identity cross-check: for uniform P over the closed outer
    // surface, dW = P * (-dV_domain). V_domain printed so the offline diff
    // can separate an area-vector overcount from a genuinely faster
    // surface sweep.
    std::vector<double> vol_h;
    state.vol.copy_to_host(vol_h);
    long double v_dom = 0.0L;
    for (const double v : vol_h) {
      v_dom += v;
    }
    std::fprintf(stderr,
                 "[drive_work_v] step=%d V_domain=%.10e\n",
                 state.step,
                 static_cast<double>(v_dom));
    std::fprintf(stderr,
                 "[drive_work] step=%d t=%.6e p_ext=%.4e W_step=%.6e "
                 "W_cum=%.10e\n",
                 state.step,
                 state.t,
                 p_ext,
                 static_cast<double>(w_step),
                 static_cast<double>(w_cum));
    if (theta_enabled) {
      long double theta_sum = w_theta_ovf;
      for (const long double w : w_theta) {
        theta_sum += w;
      }
      const long double denom =
          fabsl(static_cast<long double>(w_cum)) > 0.0L
              ? fabsl(static_cast<long double>(w_cum))
              : 1.0e-300L;
      const double sum_dev =
          static_cast<double>(fabsl(theta_sum - w_cum) / denom);
      std::string line;
      line.reserve(64 + 16 * w_theta.size());
      char buf[64];
      std::snprintf(buf, sizeof(buf), "[drive_work_theta] step=%d t=%.6e W_j=",
                    state.step, state.t);
      line += buf;
      for (std::size_t c = 0; c < w_theta.size(); ++c) {
        std::snprintf(buf, sizeof(buf), c == 0 ? "%.6e" : ",%.6e",
                      static_cast<double>(w_theta[c]));
        line += buf;
      }
      if (w_theta_ovf != 0.0L) {
        std::snprintf(buf, sizeof(buf), " ovf=%.6e",
                      static_cast<double>(w_theta_ovf));
        line += buf;
      }
      std::snprintf(buf, sizeof(buf), " sum_dev=%.3e\n", sum_dev);
      line += buf;
      std::fprintf(stderr, "%s", line.c_str());
    }
  }
}

namespace {

ShellDriveSymDiagSnapshot capture_shell_drive_sym_diag_snapshot(
    const core::State& state,
    const core::Config& cfg,
    const double eval_time,
    const core::NodeField1D& node_mass) {
  ShellDriveSymDiagSnapshot snap{};
  const ShellDriveSymNodeRow row = shell_drive_sym_outer_node_row(state, cfg);
  if (!row.valid || node_mass.size() != state.x_r.size()) {
    return snap;
  }

  core::NodeField1D drive_force_r;
  core::NodeField1D drive_force_z;
  assemble_shell_drive_force_2d(
      drive_force_r, drive_force_z, state, cfg, eval_time, &snap.p_ext);

  const std::vector<double> force_r = copy_field_to_vector(drive_force_r);
  const std::vector<double> force_z = copy_field_to_vector(drive_force_z);
  const std::vector<double> x_r = copy_field_to_vector(state.x_r);
  const std::vector<double> x_z = copy_field_to_vector(state.x_z);
  const std::vector<double> mass = copy_field_to_vector(node_mass);
  const std::size_t n_nodes = state.x_r.size();
  if (force_r.size() != n_nodes || force_z.size() != n_nodes ||
      x_r.size() != n_nodes || x_z.size() != n_nodes ||
      mass.size() != n_nodes) {
    return snap;
  }

  snap.valid = true;
  snap.multiblock = row.multiblock;
  snap.local_i = row.local_i;
  snap.n_j = row.n_j;
  snap.eval_time = eval_time;
  snap.force_r = force_r;
  snap.force_z = force_z;
  snap.a_rad_bins.assign(static_cast<std::size_t>(row.n_j + 1),
                         std::numeric_limits<double>::quiet_NaN());

  long double sum_m_rad2 = 0.0L;
  long double sum_m_tan2 = 0.0L;
  long double sum_rad = 0.0L;
  long double sum_rad2 = 0.0L;
  int valid_bins = 0;

  for (int j = 0; j <= row.n_j; ++j) {
    const int node = row.node_begin + row.local_i * row.stride + j;
    if (node < 0 || static_cast<std::size_t>(node) >= n_nodes) {
      continue;
    }
    const double m = mass[static_cast<std::size_t>(node)];
    const double r = x_r[static_cast<std::size_t>(node)];
    const double z = x_z[static_cast<std::size_t>(node)];
    const double radius = std::hypot(r, z);
    if (!(m > 0.0) || !(radius > 0.0) || !std::isfinite(m) ||
        !std::isfinite(radius)) {
      continue;
    }
    const double ar = force_r[static_cast<std::size_t>(node)] / m;
    const double az = force_z[static_cast<std::size_t>(node)] / m;
    if (!std::isfinite(ar) || !std::isfinite(az)) {
      continue;
    }
    const double n_r = r / radius;
    const double n_z = z / radius;
    const double a_rad = ar * n_r + az * n_z;
    const double a_tan = -ar * n_z + az * n_r;
    snap.a_rad_bins[static_cast<std::size_t>(j)] = a_rad;
    sum_m_rad2 += static_cast<long double>(m) *
                  static_cast<long double>(a_rad) *
                  static_cast<long double>(a_rad);
    sum_m_tan2 += static_cast<long double>(m) *
                  static_cast<long double>(a_tan) *
                  static_cast<long double>(a_tan);
    sum_rad += static_cast<long double>(a_rad);
    sum_rad2 += static_cast<long double>(a_rad) *
                static_cast<long double>(a_rad);
    ++valid_bins;
  }

  snap.node_count = valid_bins;
  if (sum_m_rad2 > 0.0L) {
    snap.eps_tan =
        std::sqrt(static_cast<double>(sum_m_tan2 / sum_m_rad2));
  }
  if (valid_bins > 0) {
    const long double inv = 1.0L / static_cast<long double>(valid_bins);
    const long double mean = sum_rad * inv;
    const long double variance = std::max(
        0.0L, sum_rad2 * inv - mean * mean);
    snap.a_rad_mean = static_cast<double>(mean);
    snap.a_rad_std = std::sqrt(static_cast<double>(variance));
    if (std::abs(static_cast<double>(mean)) > 0.0) {
      snap.eps_rad =
          snap.a_rad_std / std::abs(static_cast<double>(mean));
    }
  }
  return snap;
}

double shell_drive_sym_work_from_snapshot(
    const ShellDriveSymDiagSnapshot& snap,
    const double dt,
    const std::vector<double>& old_v_r,
    const std::vector<double>& old_v_z,
    const std::vector<double>& new_v_r,
    const std::vector<double>& new_v_z) {
  if (!snap.valid || !(dt > 0.0) || old_v_r.size() != snap.force_r.size() ||
      old_v_z.size() != snap.force_r.size() ||
      new_v_r.size() != snap.force_r.size() ||
      new_v_z.size() != snap.force_r.size()) {
    return 0.0;
  }
  long double work_rate = 0.0L;
  for (std::size_t n = 0; n < snap.force_r.size(); ++n) {
    const double u_r = 0.5 * (old_v_r[n] + new_v_r[n]);
    const double u_z = 0.5 * (old_v_z[n] + new_v_z[n]);
    if (!std::isfinite(u_r) || !std::isfinite(u_z)) {
      continue;
    }
    work_rate += static_cast<long double>(u_r) *
                 static_cast<long double>(snap.force_r[n]);
    work_rate += static_cast<long double>(u_z) *
                 static_cast<long double>(snap.force_z[n]);
  }
  return dt * static_cast<double>(work_rate);
}

void emit_shell_drive_sym_diag(
    const core::State& state,
    const core::Config& cfg,
    const double t_op,
    const double dt,
    const diagnostics::EnergyTotals& energy_before,
    const ShellDriveSymDiagSnapshot& snap,
    const core::NodeField1D& old_v_r,
    const core::NodeField1D& old_v_z) {
  if (!snap.valid) {
    return;
  }
  const diagnostics::EnergyTotals energy_after =
      diagnostics::compute_energy_totals_2d(state);
  const double E_before = energy_before.E_int_e + energy_before.E_int_i +
                          energy_before.E_kin;
  const double E_after = energy_after.E_int_e + energy_after.E_int_i +
                         energy_after.E_kin;
  const double dE = E_after - E_before;
  const std::vector<double> old_r = copy_field_to_vector(old_v_r);
  const std::vector<double> old_z = copy_field_to_vector(old_v_z);
  const std::vector<double> new_r = copy_field_to_vector(state.v_r);
  const std::vector<double> new_z = copy_field_to_vector(state.v_z);
  const double W_drive =
      shell_drive_sym_work_from_snapshot(snap, dt, old_r, old_z, new_r, new_z);
  const double R_W = dE - W_drive;
  const double denom =
      std::max({std::abs(dE), std::abs(W_drive), 1.0e-300});
  const double R_W_rel = R_W / denom;

  std::ostringstream summary;
  summary << "[shell_drive_sym] step=" << state.step
          << " t=" << format_scientific(t_op)
          << " dt=" << format_scientific(dt)
          << " eval_time=" << format_scientific(snap.eval_time)
          << " source=drive_force_half"
          << " geometry=midpoint"
          << " topology=" << (snap.multiblock ? "multiblock_shell" : "structured")
          << " local_i=" << snap.local_i
          << " bins=" << snap.node_count << "/" << (snap.n_j + 1)
          << " p_ext=" << format_scientific(snap.p_ext)
          << " eps_tan=" << format_scientific(snap.eps_tan)
          << " eps_rad=" << format_scientific(snap.eps_rad)
          << " a_rad_mean=" << format_scientific(snap.a_rad_mean)
          << " a_rad_std=" << format_scientific(snap.a_rad_std)
          << " dE_K_plus_Eint=" << format_scientific(dE)
          << " W_drive=" << format_scientific(W_drive)
          << " R_W=" << format_scientific(R_W)
          << " R_W_rel=" << format_scientific(R_W_rel)
          << " note=separate_drive_force_buffer";
  core::log_info(summary.str());

  std::ostringstream bins;
  bins << "[shell_drive_sym] bins step=" << state.step
       << " local_i=" << snap.local_i << " a_rad_by_local_j=[";
  for (std::size_t j = 0; j < snap.a_rad_bins.size(); ++j) {
    if (j > 0) {
      bins << ",";
    }
    bins << j << ":" << format_scientific(snap.a_rad_bins[j]);
  }
  bins << "]";
  core::log_info(bins.str());

  (void)cfg;
}

bool is_tri_fan_center_perturbation_topology(const core::State& state) {
  return state.mesh.dim == 2 &&
         state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane &&
         state.mesh.polar_center_treatment == mesh::PolarCenterTreatment::TriFan;
}

bool is_multiblock_center_perturbation_topology(const core::State& state) {
  return state.mesh.dim == 2 && state.mesh.topo.multiblock.has_value();
}

bool tri_fan_center_perturbation_history_due(const core::State& state,
                                             const core::Config& cfg,
                                             const double dt) {
  const int next_step = state.step + 1;
  const double next_t = state.t + dt;
  const bool by_step =
      (cfg.output.history_every > 0) && (next_step % cfg.output.history_every == 0);
  bool by_time = false;
  if (cfg.output.history_every_s > 0.0 && state.t_next_history >= 0.0) {
    const double eps =
        1.0e-14 * std::max(std::abs(next_t), cfg.output.history_every_s);
    by_time = next_t >= (state.t_next_history - eps);
  }
  return by_step || by_time;
}

double finite_signed_ratio_or_zero(const double current, const double reference) {
  if (reference == 0.0) {
    return 0.0;
  }
  const double ratio = current / reference;
  return std::isfinite(ratio) ? ratio : 0.0;
}

// Sign of the planar signed area of cell c's quad, read from the BASE
// coordinates. The trial volume below is shoelace-family and flips sign
// with the node winding (spherical-polar meshes are clockwise-wound); the
// base-quad orientation is applied to every trial volume so that a winding
// flip during the trial step still reads as a collapsed cell and is
// rejected, matching rz_geometric_cfl. On positive-winding meshes the
// factor is exactly 1.0 and the limiter is bit-identical.
double quad_winding_orientation_from_nodes(const std::vector<double>& r,
                                           const std::vector<double>& z,
                                           const int nz,
                                           const int c) {
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n[4] = {i * stride + j, (i + 1) * stride + j,
                    (i + 1) * stride + (j + 1), i * stride + (j + 1)};
  double a2 = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    a2 += r[static_cast<std::size_t>(n[k])] * z[static_cast<std::size_t>(n[kp])] -
          r[static_cast<std::size_t>(n[kp])] * z[static_cast<std::size_t>(n[k])];
  }
  return (a2 >= 0.0) ? 1.0 : -1.0;
}

double rz_quad_volume_from_nodes(const std::vector<double>& r,
                                 const std::vector<double>& z,
                                 const int nr,
                                 const int nz,
                                 const int c,
                                 const double orientation) {
  (void)nr;
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);

  const double rc[4] = {r[n00], r[n10], r[n11], r[n01]};
  const double zc[4] = {z[n00], z[n10], z[n11], z[n01]};
  const double centroid_r = 0.25 * (rc[0] + rc[1] + rc[2] + rc[3]);
  const double centroid_z = 0.25 * (zc[0] + zc[1] + zc[2] + zc[3]);

  double vol_sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    const double rk = rc[k] - centroid_r;
    const double rkp1 = rc[kp1] - centroid_r;
    const double zk = zc[k] - centroid_z;
    const double zkp1 = zc[kp1] - centroid_z;
    const double cross_local = rk * zkp1 - rkp1 * zk;
    const double weight = rk + rkp1 + 3.0 * centroid_r;
    vol_sum += cross_local * weight;
  }

  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  return orientation * (pi_over_three * vol_sum);
}

struct WorkSplitRowSummary {
  bool valid = false;
  int i = -1;
  int global_i = -1;
  int layer_j0 = -1;
  int layer_j1 = -1;
  int target_j0 = -1;
  int target_j1 = -1;
  int layer_width_cells = 0;
  double layer_width_cm = 0.0;
  double sum_W_P = 0.0;
  double sum_W_Q = 0.0;
  double mean_Te = 0.0;
  double max_Te = 0.0;
  double mean_Te_noQ = 0.0;
  double max_Te_noQ = 0.0;
  double mean_Tiso = 0.0;
  double max_Tiso = 0.0;
  int upstream_count = 0;
};

double work_split_cell_z_center(const std::vector<double>& z,
                                const int nr,
                                const int nz,
                                const int c) {
  (void)nr;
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  const int n00 = i * stride + j;
  const int n10 = (i + 1) * stride + j;
  const int n11 = (i + 1) * stride + (j + 1);
  const int n01 = i * stride + (j + 1);
  return 0.25 * (z[n00] + z[n10] + z[n11] + z[n01]);
}

double work_split_cv_e_fallback(const core::Config& cfg) {
  if (cfg.materials.materials.empty()) {
    return 0.0;
  }
  const auto& mat = cfg.materials.materials.front();
  const double gm1 = mat.ideal_gas_gamma - 1.0;
  const double z = (cfg.materials.zbar.model == "fixed" &&
                    cfg.materials.zbar.fixed_value >= 0.0)
                       ? cfg.materials.zbar.fixed_value
                       : mat.Z;
  if (!(gm1 > 0.0) || !(mat.A > 0.0) || !(z > 0.0)) {
    return 0.0;
  }
  return z * kEvToErg / (mat.A * kProtonMass * gm1);
}

double work_split_cv_i_fallback(const core::Config& cfg) {
  if (cfg.materials.materials.empty()) {
    return 0.0;
  }
  const auto& mat = cfg.materials.materials.front();
  const double gm1 = mat.ideal_gas_gamma - 1.0;
  if (!(gm1 > 0.0) || !(mat.A > 0.0)) {
    return 0.0;
  }
  return kEvToErg / (mat.A * kProtonMass * gm1);
}

bool work_split_plot_due(const core::State& state,
                         const core::Config& cfg,
                         const double dt) {
  const bool by_step =
      (cfg.output.plot_every > 0) && (state.step % cfg.output.plot_every == 0);
  const double t_after = state.t + dt;
  const double interval = cfg.output.plot_every_s;
  const double eps =
      (interval > 0.0) ? 1.0e-12 * std::max(1.0, std::abs(interval)) : 0.0;
  const bool by_time =
      (interval > 0.0) && (state.t_next_plot >= 0.0) &&
      (t_after + eps >= state.t_next_plot);
  const double t_end_eps =
      (cfg.main.t_end > 0.0)
          ? 1.0e-12 * std::max(1.0, std::abs(cfg.main.t_end))
          : 0.0;
  const bool final_step =
      (cfg.main.t_end > 0.0) && (t_after + t_end_eps >= cfg.main.t_end);
  const int interval_steps =
      cfg.numerics.hydro.work_split_audit_cell_every_n_steps;
  const bool by_audit_interval =
      (interval_steps > 0) && (state.step % interval_steps == 0);
  return by_step || by_time || final_step || by_audit_interval;
}

std::pair<double, double> infer_work_split_upstream_state(
    const int i,
    const int nz,
    const int j0,
    const int j1,
    const bool downstream_plus,
    const std::vector<double>& rho,
    const std::vector<double>& Te) {
  constexpr int margin = 4;
  int lo = margin;
  int hi = nz - margin - 1;
  if (j0 >= 0 && j1 >= 0) {
    if (downstream_plus) {
      lo = margin;
      hi = std::max(margin, j0 - 1);
    } else {
      lo = std::min(nz - margin - 1, j1 + 1);
      hi = nz - margin - 1;
    }
  }
  int best_j = -1;
  double best_T = std::numeric_limits<double>::infinity();
  for (int j = lo; j <= hi; ++j) {
    const double T = Te[i * nz + j];
    if (std::isfinite(T) && T > 0.0 && T < best_T) {
      best_T = T;
      best_j = j;
    }
  }
  if (best_j < 0) {
    for (int j = margin; j <= std::min(nz - margin - 1, margin + 4); ++j) {
      const double T = Te[i * nz + j];
      if (std::isfinite(T) && T > 0.0 && T < best_T) {
        best_T = T;
        best_j = j;
      }
    }
  }
  if (best_j < 0) {
    return {30.0, 2.0};
  }
  double sum_T = 0.0;
  double sum_rho = 0.0;
  int count = 0;
  for (int j = std::max(margin, best_j - 2);
       j <= std::min(nz - margin - 1, best_j + 2);
       ++j) {
    const int c = i * nz + j;
    if (Te[c] > 0.0 && rho[c] > 0.0 &&
        std::isfinite(Te[c]) && std::isfinite(rho[c])) {
      sum_T += Te[c];
      sum_rho += rho[c];
      ++count;
    }
  }
  if (count <= 0) {
    return {30.0, 2.0};
  }
  return {sum_T / static_cast<double>(count),
          sum_rho / static_cast<double>(count)};
}

void append_work_split_audit_2d_rz(
    const core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    const double dt,
    const char* hydro_half,
    const core::CellField1D& e_old,
    const core::CellField1D& ei_old,
    const core::CellField1D& V_old,
    const core::CellField1D& P_half,
    const core::CellField1D& Pi_half,
    const core::CellField1D& Q_half,
    const core::CellField1D& div_u_half,
    const std::int8_t* d_hydro_active,
    const int q_heat_to_electron) {
  if (!cfg.numerics.hydro.work_split_audit_2d_rz ||
      state.mesh.dim != 2 || !cfg.main.two_temperature ||
      state.rho.empty()) {
    return;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (n_cells <= 0 || static_cast<int>(state.rho.size()) != n_cells) {
    return;
  }

  static std::vector<double> cumulative_WQ;
  static int cumulative_size = 0;
  static int cumulative_rank = -1;
  static int last_step = -1;
  if ((state.step == 0 && last_step != 0) || cumulative_size != n_cells ||
      cumulative_rank != part.rank) {
    cumulative_WQ.assign(static_cast<std::size_t>(n_cells), 0.0);
    cumulative_size = n_cells;
    cumulative_rank = part.rank;
  }
  last_step = state.step;

  const std::vector<double> rho = copy_field_to_vector(state.rho);
  const std::vector<double> mass = copy_field_to_vector(state.mass);
  const std::vector<double> vol = copy_field_to_vector(state.vol);
  const std::vector<double> vol_old = copy_field_to_vector(V_old);
  const std::vector<double> ee_new = copy_field_to_vector(state.ee);
  const std::vector<double> ei_new = copy_field_to_vector(state.ei);
  const std::vector<double> ee_n = copy_field_to_vector(e_old);
  const std::vector<double> ei_n = copy_field_to_vector(ei_old);
  const std::vector<double> Pe = copy_field_to_vector(P_half);
  const std::vector<double> Pi = copy_field_to_vector(Pi_half);
  const std::vector<double> Q = copy_field_to_vector(Q_half);
  const std::vector<double> div_u = copy_field_to_vector(div_u_half);
  const std::vector<double> Te = copy_field_to_vector(state.Te);
  const std::vector<double> Ti = copy_field_to_vector(state.Ti);
  const std::vector<double> z_node = copy_field_to_vector(state.x_z);
  const std::vector<double> cv_e =
      state.cv_e.empty() ? std::vector<double>{} : copy_field_to_vector(state.cv_e);
  const std::vector<double> cv_i =
      state.cv_i.empty() ? std::vector<double>{} : copy_field_to_vector(state.cv_i);

  std::vector<std::int8_t> hydro_active;
  if (d_hydro_active != nullptr) {
    hydro_active.resize(static_cast<std::size_t>(n_cells), 0);
    cuda_check(cudaMemcpy(hydro_active.data(), d_hydro_active,
                          static_cast<std::size_t>(n_cells) * sizeof(std::int8_t),
                          cudaMemcpyDeviceToHost),
               "Hydro2D work_split_audit hydro_active D2H failed");
  }

  const double cv_e_default = work_split_cv_e_fallback(cfg);
  const double cv_i_default = work_split_cv_i_fallback(cfg);
  const double gamma =
      cfg.materials.materials.empty()
          ? (5.0 / 3.0)
          : cfg.materials.materials.front().ideal_gas_gamma;
  const double gm1 = std::max(gamma - 1.0, 1.0e-12);
  const bool emit_cells = work_split_plot_due(state, cfg, dt);
  const bool all_rows = cfg.numerics.hydro.work_split_audit_all_rows;
  const int requested_mid_global_i = 16;
  int mid_i = requested_mid_global_i - part.global_offset_r;
  if (mid_i < 0 || mid_i >= nr) {
    mid_i = nr / 2;
  }

  const std::filesystem::path diag_dir =
      (cfg.output.directory.empty()
           ? std::filesystem::path(".")
           : std::filesystem::path(cfg.output.directory)) /
      "diag";
  std::filesystem::create_directories(diag_dir);
  std::ostringstream filename;
  filename << "work_split_audit_2d_rz_rank" << std::setw(4)
           << std::setfill('0') << part.rank << ".csv";
  const std::filesystem::path path = diag_dir / filename.str();
  const bool write_header =
      !std::filesystem::exists(path) || std::filesystem::file_size(path) == 0;
  std::ofstream out(path, std::ios::out | std::ios::app);
  if (!out) {
    tenryu::core::log_warning("Hydro2D work_split_audit failed to open " +
                              path.string());
    return;
  }
  out << std::scientific << std::setprecision(16);
  if (write_header) {
    out << "record_type,step,time_s,dt_s,half,rank,scope,global_i,local_i,"
           "global_j,local_j,z_cm,layer_j0,layer_j1,target_j0,target_j1,"
           "shock_layer_member,upstream_half_member,downstream_target_candidate,"
           "rho,Te_eV,Ti_eV,T_iso_inferred_eV,T_iso_const_eV,Te_noQ_eV,"
           "div_u,dV_cm3,W_Pe_erg,W_Pi_erg,W_P_erg,W_Q_erg,qei_erg,"
           "delta_Ee_erg,delta_Ei_erg,sum_W_P_upstream_erg,"
           "sum_W_Q_upstream_erg,wq_fraction_upstream,mean_Te_upstream_eV,"
           "max_Te_upstream_eV,mean_Te_noQ_upstream_eV,"
           "max_Te_noQ_upstream_eV,mean_Tiso_upstream_eV,"
           "max_Tiso_upstream_eV,layer_width_cells,layer_width_cm,"
           "T_up_inferred_eV,rho_up_inferred_gcc\n";
  }

  auto active_cell = [&](const int c) {
    return hydro_active.empty() || hydro_active[static_cast<std::size_t>(c)] != 0;
  };
  auto cell_z = [&](const int c) {
    return work_split_cell_z_center(z_node, nr, nz, c);
  };
  auto cv_total = [&](const int c) {
    const double cve =
        (!cv_e.empty() && cv_e[static_cast<std::size_t>(c)] > 0.0)
            ? cv_e[static_cast<std::size_t>(c)]
            : cv_e_default;
    const double cvi =
        (!cv_i.empty() && cv_i[static_cast<std::size_t>(c)] > 0.0)
            ? cv_i[static_cast<std::size_t>(c)]
            : cv_i_default;
    return std::max(cve + cvi, 1.0e-300);
  };

  WorkSplitRowSummary radial{};
  radial.valid = true;
  radial.i = -1;
  radial.global_i = -1;
  radial.layer_j0 = -1;
  radial.layer_j1 = -1;
  radial.target_j0 = -1;
  radial.target_j1 = -1;
  radial.max_Te = -std::numeric_limits<double>::infinity();
  radial.max_Te_noQ = -std::numeric_limits<double>::infinity();
  radial.max_Tiso = -std::numeric_limits<double>::infinity();
  WorkSplitRowSummary mid_summary{};

  for (int i = 0; i < nr; ++i) {
    constexpr int margin = 4;
    std::vector<double> chi(static_cast<std::size_t>(nz), 0.0);
    int peak_j = -1;
    double peak_chi = 0.0;
    for (int j = margin; j < nz - margin; ++j) {
      const int c = i * nz + j;
      if (!active_cell(c)) {
        continue;
      }
      const int jm = i * nz + (j - 1);
      const int jp = i * nz + (j + 1);
      const double p_m = Pe[jm] + Pi[jm];
      const double p_p = Pe[jp] + Pi[jp];
      const bool rising = (rho[jp] > rho[jm]) && (p_p > p_m) &&
                          (Te[jp] > Te[jm]);
      if (!rising) {
        continue;
      }
      chi[static_cast<std::size_t>(j)] = std::max(0.0, -div_u[c]);
      if (chi[static_cast<std::size_t>(j)] > peak_chi) {
        peak_chi = chi[static_cast<std::size_t>(j)];
        peak_j = j;
      }
    }

    int j0 = -1;
    int j1 = -1;
    if (peak_j >= 0 && peak_chi > 0.0) {
      j0 = peak_j;
      j1 = peak_j;
      while (j0 > margin && chi[static_cast<std::size_t>(j0 - 1)] > 0.0) {
        --j0;
      }
      while (j1 < nz - margin - 1 &&
             chi[static_cast<std::size_t>(j1 + 1)] > 0.0) {
        ++j1;
      }
    }

    const int before_j = (j0 > margin) ? (j0 - 1) : margin;
    const int after_j = (j1 >= 0 && j1 < nz - margin - 1) ? (j1 + 1)
                                                          : (nz - margin - 1);
    const bool downstream_plus =
        (j0 >= 0 && j1 >= 0)
            ? (rho[i * nz + after_j] >= rho[i * nz + before_j])
            : true;
    int target_j0 = -1;
    int target_j1 = -1;
    if (j0 >= 0 && j1 >= 0) {
      if (downstream_plus) {
        target_j0 = std::min(nz - margin - 1, j1 + 1);
        target_j1 = std::min(nz - margin - 1, j1 + 3);
      } else {
        target_j0 = std::max(margin, j0 - 3);
        target_j1 = std::max(margin, j0 - 1);
      }
    }

    const auto [T_up, rho_up] =
        infer_work_split_upstream_state(i, nz, j0, j1, downstream_plus, rho, Te);
    const int upstream_mid =
        (j0 >= 0 && j1 >= 0) ? ((j0 + j1) / 2) : -1;

    WorkSplitRowSummary row{};
    row.valid = true;
    row.i = i;
    row.global_i = part.global_offset_r + i;
    row.layer_j0 = j0;
    row.layer_j1 = j1;
    row.target_j0 = target_j0;
    row.target_j1 = target_j1;
    row.layer_width_cells = (j0 >= 0 && j1 >= j0) ? (j1 - j0 + 1) : 0;
    if (row.layer_width_cells > 0) {
      row.layer_width_cm = std::abs(cell_z(i * nz + j1) - cell_z(i * nz + j0));
    }
    row.max_Te = -std::numeric_limits<double>::infinity();
    row.max_Te_noQ = -std::numeric_limits<double>::infinity();
    row.max_Tiso = -std::numeric_limits<double>::infinity();

    auto write_cell = [&](const int j,
                          const bool shock_member,
                          const bool upstream_member,
                          const bool target_member,
                          const double T_iso,
                          const double T_iso_const,
                          const double Te_noQ,
                          const double W_Pe,
                          const double W_Pi,
                          const double W_Q,
                          const double qei_erg,
                          const double dEe,
                          const double dEi) {
      const int c = i * nz + j;
      out << "cell," << state.step << ',' << state.t << ',' << dt << ','
          << (hydro_half != nullptr ? hydro_half : "") << ',' << part.rank
          << ',' << (all_rows ? "row" : "mid") << ','
          << (part.global_offset_r + i) << ',' << i << ','
          << (part.global_offset_z + j) << ',' << j << ',' << cell_z(c) << ','
          << j0 << ',' << j1 << ',' << target_j0 << ',' << target_j1 << ','
          << (shock_member ? 1 : 0) << ',' << (upstream_member ? 1 : 0) << ','
          << (target_member ? 1 : 0) << ',' << rho[c] << ',' << Te[c] << ','
          << Ti[c] << ',' << T_iso << ',' << T_iso_const << ',' << Te_noQ
          << ',' << div_u[c] << ',' << (vol[c] - vol_old[c]) << ','
          << W_Pe << ',' << W_Pi << ',' << (W_Pe + W_Pi) << ',' << W_Q
          << ',' << qei_erg << ',' << dEe << ',' << dEi
          << ",,,,,,,,,,," << T_up << ',' << rho_up << '\n';
    };

    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      if (!active_cell(c)) {
        continue;
      }
      const double dV = vol[c] - vol_old[c];
      const double W_Pe = -Pe[c] * dV;
      const double W_Pi = -Pi[c] * dV;
      const double W_Q = -Q[c] * dV;
      cumulative_WQ[static_cast<std::size_t>(c)] += W_Q;
      const double dEe = mass[c] * (ee_new[c] - ee_n[c]);
      const double dEi = mass[c] * (ei_new[c] - ei_n[c]);
      const double W_Q_e = (q_heat_to_electron != 0) ? W_Q : 0.0;
      const double W_Q_i = (q_heat_to_electron != 0) ? 0.0 : W_Q;
      const double qei_from_e = W_Pe + W_Q_e - dEe;
      const double qei_from_i = dEi - W_Pi - W_Q_i;
      const double qei_erg = 0.5 * (qei_from_e + qei_from_i);
      const double rho_ratio =
          (rho_up > 0.0 && rho[c] > 0.0) ? (rho[c] / rho_up) : 1.0;
      const double T_iso = T_up * std::pow(std::max(rho_ratio, 1.0e-300), gm1);
      const double T_iso_const =
          30.0 * std::pow(std::max(rho[c] / 2.0, 1.0e-300), gm1);
      const double Te_noQ =
          Te[c] - cumulative_WQ[static_cast<std::size_t>(c)] /
                      (std::max(mass[c], 1.0e-300) * cv_total(c));
      const bool shock_member = (j0 >= 0 && j >= j0 && j <= j1);
      const bool upstream_member =
          (j0 >= 0 && j1 >= 0) &&
          (downstream_plus ? (j >= j0 && j <= upstream_mid)
                           : (j >= upstream_mid && j <= j1));
      const bool target_member =
          (target_j0 >= 0 && target_j1 >= 0 && j >= target_j0 && j <= target_j1);

      if (upstream_member) {
        row.sum_W_P += W_Pe + W_Pi;
        row.sum_W_Q += W_Q;
        row.mean_Te += Te[c];
        row.max_Te = std::max(row.max_Te, Te[c]);
        row.mean_Te_noQ += Te_noQ;
        row.max_Te_noQ = std::max(row.max_Te_noQ, Te_noQ);
        row.mean_Tiso += T_iso;
        row.max_Tiso = std::max(row.max_Tiso, T_iso);
        ++row.upstream_count;

        radial.sum_W_P += W_Pe + W_Pi;
        radial.sum_W_Q += W_Q;
        radial.mean_Te += Te[c];
        radial.max_Te = std::max(radial.max_Te, Te[c]);
        radial.mean_Te_noQ += Te_noQ;
        radial.max_Te_noQ = std::max(radial.max_Te_noQ, Te_noQ);
        radial.mean_Tiso += T_iso;
        radial.max_Tiso = std::max(radial.max_Tiso, T_iso);
        ++radial.upstream_count;
      }

      if (emit_cells && (all_rows || i == mid_i)) {
        write_cell(j, shock_member, upstream_member, target_member, T_iso,
                   T_iso_const, Te_noQ, W_Pe, W_Pi, W_Q, qei_erg, dEe, dEi);
      }
    }

    if (row.upstream_count > 0) {
      const double inv = 1.0 / static_cast<double>(row.upstream_count);
      row.mean_Te *= inv;
      row.mean_Te_noQ *= inv;
      row.mean_Tiso *= inv;
    } else {
      row.max_Te = 0.0;
      row.max_Te_noQ = 0.0;
      row.max_Tiso = 0.0;
    }
    if (i == mid_i) {
      mid_summary = row;
    }
  }

  if (radial.upstream_count > 0) {
    const double inv = 1.0 / static_cast<double>(radial.upstream_count);
    radial.mean_Te *= inv;
    radial.mean_Te_noQ *= inv;
    radial.mean_Tiso *= inv;
  } else {
    radial.max_Te = 0.0;
    radial.max_Te_noQ = 0.0;
    radial.max_Tiso = 0.0;
  }

  auto write_summary = [&](const char* scope, const WorkSplitRowSummary& s) {
    const double denom = s.sum_W_P + s.sum_W_Q;
    const double frac = (std::abs(denom) > 0.0) ? (s.sum_W_Q / denom) : 0.0;
    out << "summary," << state.step << ',' << state.t << ',' << dt << ','
        << (hydro_half != nullptr ? hydro_half : "") << ',' << part.rank << ','
        << scope << ',' << s.global_i << ',' << s.i
        << ",-1,-1,0.0," << s.layer_j0 << ',' << s.layer_j1 << ','
        << s.target_j0 << ',' << s.target_j1
        << ",0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
        << s.sum_W_P << ',' << s.sum_W_Q << ',' << frac << ','
        << s.mean_Te << ',' << s.max_Te << ',' << s.mean_Te_noQ << ','
        << s.max_Te_noQ << ',' << s.mean_Tiso << ',' << s.max_Tiso << ','
        << s.layer_width_cells << ',' << s.layer_width_cm << ",0,0\n";
  };
  write_summary("mid", mid_summary);
  write_summary("radial_sum", radial);
}

TrialVolumeCflResult check_trial_lagrangian_volume_cfl_host(
    const int nr,
    const int nz,
    const std::vector<double>& r_base,
    const std::vector<double>& z_base,
    const std::vector<double>& v_r,
    const std::vector<double>& v_z,
    const std::vector<double>& vol_current,
    const double dt,
    const double floor_fraction,
    const double shrink_fraction,
    const std::vector<std::int8_t>* hydro_active,
    const parallel::Reduction* reduction) {
  TENRYU_ASSERT(dt > 0.0, "trial-volume CFL requires dt > 0");
  TENRYU_ASSERT(floor_fraction > 0.0 && floor_fraction <= 1.0,
                "trial-volume CFL floor_fraction must be in (0, 1]");
  TENRYU_ASSERT(shrink_fraction > 0.0 && shrink_fraction < 1.0,
                "trial-volume CFL shrink_fraction must be in (0, 1)");
  const int n_cells = nr * nz;
  const int n_nodes = (nr + 1) * (nz + 1);
  TENRYU_ASSERT(static_cast<int>(vol_current.size()) == n_cells,
                "trial-volume CFL requires vol_current size == n_cells");
  TENRYU_ASSERT(static_cast<int>(r_base.size()) == n_nodes &&
                    static_cast<int>(z_base.size()) == n_nodes &&
                    static_cast<int>(v_r.size()) == n_nodes &&
                    static_cast<int>(v_z.size()) == n_nodes,
                "trial-volume CFL requires node field sizes == n_nodes");
  TENRYU_ASSERT(hydro_active == nullptr || hydro_active->empty() ||
                    static_cast<int>(hydro_active->size()) == n_cells,
                "trial-volume CFL hydro_active size mismatch");

  std::vector<double> r_trial(r_base.size(), 0.0);
  std::vector<double> z_trial(z_base.size(), 0.0);
  for (std::size_t n = 0; n < r_base.size(); ++n) {
    r_trial[n] = r_base[n] + dt * v_r[n];
    z_trial[n] = z_base[n] + dt * v_z[n];
  }

  double local_min_ratio = std::numeric_limits<double>::infinity();
  int first_failing_cell = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (hydro_active != nullptr && !hydro_active->empty() &&
        (*hydro_active)[static_cast<std::size_t>(c)] == 0) {
      continue;
    }
    const double orientation =
        quad_winding_orientation_from_nodes(r_base, z_base, nz, c);
    const double vol_trial =
        rz_quad_volume_from_nodes(r_trial, z_trial, nr, nz, c, orientation);
    const double denom = std::max(vol_current[static_cast<std::size_t>(c)], 1.0e-300);
    const double ratio = vol_trial / denom;
    local_min_ratio = std::min(local_min_ratio, ratio);
    if (first_failing_cell < 0 && ratio < floor_fraction) {
      first_failing_cell = c;
    }
  }
  if (!std::isfinite(local_min_ratio)) {
    local_min_ratio = 1.0;
  }

  const double global_min_ratio =
      reduction != nullptr ? reduction->allreduce_min(local_min_ratio) : local_min_ratio;
  TrialVolumeCflResult result{};
  result.admissible = global_min_ratio >= floor_fraction;
  result.min_trial_vol_ratio = global_min_ratio;
  result.suggested_dt = result.admissible ? dt : dt * shrink_fraction;
  result.first_failing_cell = result.admissible ? -1 : first_failing_cell;
  return result;
}

TrialVolumeCflResult check_trial_lagrangian_volume_cfl_fields(
    const int nr,
    const int nz,
    const core::NodeField1D& r_base,
    const core::NodeField1D& z_base,
    const core::NodeField1D& v_r,
    const core::NodeField1D& v_z,
    const core::CellField1D& vol_current,
    const double dt,
    const double floor_fraction,
    const double shrink_fraction,
    const std::vector<std::int8_t>* hydro_active,
    const parallel::Reduction* reduction) {
  return check_trial_lagrangian_volume_cfl_host(
      nr, nz, copy_field_to_vector(r_base), copy_field_to_vector(z_base),
      copy_field_to_vector(v_r), copy_field_to_vector(v_z),
      copy_field_to_vector(vol_current), dt, floor_fraction, shrink_fraction,
      hydro_active, reduction);
}

std::int8_t* upload_hydro_active(const char* pool_tag,
                                 const std::vector<std::int8_t>& hydro_active) {
  if (hydro_active.empty()) {
    return nullptr;
  }
  std::int8_t* d_active = nullptr;
  if (pool_tag != nullptr) {
    d_active = static_cast<std::int8_t*>(core::device_scratch_acquire(
        pool_tag, hydro_active.size() * sizeof(std::int8_t)));
  } else {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_active),
                          hydro_active.size() * sizeof(std::int8_t)),
               "Hydro2D: cudaMalloc hydro_active failed");
  }
  cuda_check(cudaMemcpy(d_active, hydro_active.data(),
                        hydro_active.size() * sizeof(std::int8_t),
                        cudaMemcpyHostToDevice),
             "Hydro2D: cudaMemcpy hydro_active failed");
  return d_active;
}

std::vector<std::int8_t> make_central_macro_effective_active(
    const core::State& state) {
  if (!central_pseudo_core::active(state) &&
      !pole_angular_derefine::active(state) &&
      state.evacuated_cells.inactive_member_mask.empty()) {
    return {};
  }
  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<std::int8_t> effective(static_cast<std::size_t>(n_cells), 1);
  if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "Hydro2D central macro active mask size mismatch");
    effective = state.hydro_active;
  }
  const auto apply_inactive = [&](const std::vector<std::uint8_t>& inactive) {
    if (inactive.size() != static_cast<std::size_t>(n_cells)) {
      return;
    }
    for (int c = 0; c < n_cells; ++c) {
      if (inactive[static_cast<std::size_t>(c)] != 0U) {
        effective[static_cast<std::size_t>(c)] = 0;
      }
    }
  };
  apply_inactive(state.central_pseudo_core.inactive_member_mask);
  apply_inactive(state.pole_angular_derefine.inactive_member_mask);
  apply_inactive(state.evacuated_cells.inactive_member_mask);
  if (effective == state.hydro_active && state.hydro_active.empty()) {
    return {};
  }
  return effective;
}

std::uint8_t* upload_uint8_vector(const char* pool_tag,
                                  const std::vector<std::uint8_t>& values,
                                  const char* const malloc_message,
                                  const char* const memcpy_message) {
  if (values.empty()) {
    return nullptr;
  }
  std::uint8_t* d_values = nullptr;
  if (pool_tag != nullptr) {
    d_values = static_cast<std::uint8_t*>(core::device_scratch_acquire(
        pool_tag, values.size() * sizeof(std::uint8_t)));
  } else {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_values),
                          values.size() * sizeof(std::uint8_t)),
               malloc_message);
  }
  cuda_check(cudaMemcpy(d_values, values.data(),
                        values.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             memcpy_message);
  return d_values;
}

std::uint8_t* upload_cell_nverts_if_nonquad(const char* pool_tag,
                                            const core::State& state) {
  const auto& cell_nverts = state.mesh.cell_nverts;
  if (cell_nverts.size() != state.mass.size()) {
    return nullptr;
  }
  const bool has_nonquad = std::any_of(cell_nverts.begin(), cell_nverts.end(),
                                       [](const std::uint8_t nverts) {
                                         return nverts != 4U;
                                       });
  return has_nonquad ? upload_uint8_vector(pool_tag,
                                           cell_nverts,
                                           "Hydro2D: cudaMalloc cell_nverts failed",
                                           "Hydro2D: cudaMemcpy cell_nverts failed")
                     : nullptr;
}

std::uint8_t* upload_node_flags_if_constraints(const char* pool_tag,
                                               const core::State& state) {
  const auto& node_flags = state.mesh.topo.node_flags;
  if (node_flags.size() != state.x_r.size()) {
    return nullptr;
  }
  const bool has_constraints = std::any_of(node_flags.begin(), node_flags.end(),
                                           [](const std::uint8_t flags) {
                                             return (flags & (kNodeCenterFlag |
                                                              kNodePoleAxisFlag |
                                                              mesh::NODE_AXIS |
                                                              mesh::NODE_INNER_PHYSICAL_BOUNDARY |
                                                              mesh::NODE_OUTER_PHYSICAL_BOUNDARY)) !=
                                                    0U;
                                           });
  return has_constraints ? upload_uint8_vector(pool_tag,
                                               node_flags,
                                               "Hydro2D: cudaMalloc node_flags failed",
                                               "Hydro2D: cudaMemcpy node_flags failed")
                         : nullptr;
}

void enforce_node_axis_state(core::State& state,
                             const std::uint8_t* d_node_flags) {
  if (d_node_flags == nullptr) {
    return;
  }
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  enforce_node_axis_state_kernel<<<nw.blocks(), 256>>>(
      state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data(),
      d_node_flags, nw.begin, nw.end, n_nodes);
  sync_kernel("Hydro2D enforce node axis state kernel failed");
}

void sync_state_supply_energy_from_temperature(core::State& state,
                                               const core::Config& cfg,
                                               const bool use_two_temp,
                                               const HydroTableViews& eos_views) {
  if (!has_state_supply_bc(cfg) || state.state_supply_mask.empty()) {
    return;
  }
  const auto& b = cfg.numerics.hydro.boundary_2d;
  const int bottom_active = b.z_bottom_cfg.supply_active(state.t) ? 1 : 0;
  const int top_active = b.z_top_cfg.supply_active(state.t) ? 1 : 0;
  if (bottom_active == 0 && top_active == 0) {
    return;
  }
  std::int8_t* d_mask = upload_hydro_active("h2d:state_supply_energy_mask",
                                            state.state_supply_mask);
  const int n_cells = static_cast<int>(state.rho.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const auto& mat = cfg.materials.materials.front();
  state_supply_energy_from_temperature_kernel<<<cw.blocks(), 256>>>(
      d_mask, state.ee.data(), state.ei.data(), state.rho.data(), state.zbar.data(),
      state.Te.data(), state.Ti.data(), cw.begin, cw.end, state.mesh.topo.nz,
      bottom_active, top_active,
      mat.ideal_gas_gamma, mat.A, mat.Z, cfg.numerics.floors.Te,
      cfg.numerics.floors.Ti,
      eos_views.tab_ion, eos_views.tab_ele, eos_views.tab_total, use_two_temp);
  sync_kernel("Hydro2D state_supply energy-from-temperature kernel failed");
}

void upload_mesh_cache(const core::State& state, MeshDeviceCache& cache) {
  TENRYU_ASSERT(state.corner_stride == state.mesh.corner_stride,
                "Hydro2D mesh cache corner stride mismatch");
  const std::size_t expected_svec_size =
      state.rho.size() *
      static_cast<std::size_t>(state.mesh.corner_stride);
  TENRYU_ASSERT(state.mesh.cell_Svec_r.size() == expected_svec_size &&
                    state.mesh.cell_Svec_z.size() == expected_svec_size,
                "Hydro2D mesh cache Svec size mismatch");
  cache.Svec_r.reset(expected_svec_size);
  cache.Svec_z.reset(expected_svec_size);
  cache.area.reset(state.rho.size());
  TENRYU_ASSERT(
      state.mesh.cell_Svec_r_device.size() == expected_svec_size &&
          state.mesh.cell_Svec_z_device.size() == expected_svec_size &&
          state.mesh.cell_area_device.size() == state.rho.size(),
      "Hydro2D mesh cache: device geometry size mismatch");
  static const bool b1_diag = std::getenv("TENRYU_B1_DIAG") != nullptr;
  if (b1_diag) {
    const_cast<core::State&>(state).mesh.materialize_host_svec();
    static int b1_diag_calls = 0;
    ++b1_diag_calls;
    std::vector<double> dev_svec_r(expected_svec_size);
    std::vector<double> dev_svec_z(expected_svec_size);
    std::vector<double> dev_area(state.rho.size());
    cuda_check(cudaMemcpy(dev_svec_r.data(),
                          state.mesh.cell_Svec_r_device.data(),
                          expected_svec_size * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "b1diag svec_r pull failed");
    cuda_check(cudaMemcpy(dev_svec_z.data(),
                          state.mesh.cell_Svec_z_device.data(),
                          expected_svec_size * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "b1diag svec_z pull failed");
    cuda_check(cudaMemcpy(dev_area.data(),
                          state.mesh.cell_area_device.data(),
                          state.rho.size() * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "b1diag area pull failed");
    std::size_t n_diff_r = 0, n_diff_z = 0, n_diff_a = 0;
    std::size_t first_r = 0, first_z = 0, first_a = 0;
    for (std::size_t i = 0; i < expected_svec_size; ++i) {
      if (dev_svec_r[i] != state.mesh.cell_Svec_r[i]) {
        if (n_diff_r == 0) first_r = i;
        ++n_diff_r;
      }
      if (dev_svec_z[i] != state.mesh.cell_Svec_z[i]) {
        if (n_diff_z == 0) first_z = i;
        ++n_diff_z;
      }
    }
    for (std::size_t i = 0; i < dev_area.size(); ++i) {
      if (dev_area[i] != state.mesh.cell_area[i]) {
        if (n_diff_a == 0) first_a = i;
        ++n_diff_a;
      }
    }
    if (n_diff_r != 0 || n_diff_z != 0 || n_diff_a != 0) {
      const int stride = state.mesh.corner_stride;
      std::ostringstream os;
      os << "[b1diag] call=" << b1_diag_calls
         << " DIVERGED svec_r=" << n_diff_r << " (first idx " << first_r
         << " cell " << (first_r / static_cast<std::size_t>(stride))
         << " slot " << (first_r % static_cast<std::size_t>(stride))
         << " dev=" << std::setprecision(17) << dev_svec_r[first_r]
         << " host=" << state.mesh.cell_Svec_r[first_r] << ")"
         << " svec_z=" << n_diff_z << " area=" << n_diff_a;
      tenryu::core::log_warning(os.str());
    } else {
      tenryu::core::log_warning("[b1diag] call=" +
                                std::to_string(b1_diag_calls) + " identical");
    }
  }
  cuda_check(cudaMemcpy(cache.Svec_r.data(),
                        state.mesh.cell_Svec_r_device.data(),
                        expected_svec_size * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D mesh cache Svec_r D2D failed");
  cuda_check(cudaMemcpy(cache.Svec_z.data(),
                        state.mesh.cell_Svec_z_device.data(),
                        expected_svec_size * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D mesh cache Svec_z D2D failed");
  cuda_check(cudaMemcpy(cache.area.data(),
                        state.mesh.cell_area_device.data(),
                        state.rho.size() * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D mesh cache area D2D failed");
}

void apply_axis_motion_preflight(core::NodeField1D& v_r,
                                 double* coupled_v_r,
                                 const core::NodeField1D& x_r_old,
                                 const core::NodeField1D& x_z_old,
                                 const core::NodeField1D& v_z,
                                 const int nr,
                                 const int nz,
                                 const double dt,
                                 const double margin_floor_fraction) {
  if (margin_floor_fraction <= 0.0 || nr < 1 || nz < 1) {
    return;
  }

  core::NodeField1D node_scale;
  node_scale.reset(static_cast<std::size_t>(nz + 1));
  node_scale.fill(1.0);
  const int blocks_j = (nz + 255) / 256;
  t21_axis_motion_preflight_kernel<<<blocks_j, 256>>>(
      node_scale.data(), x_r_old.data(), x_z_old.data(), v_r.data(), v_z.data(),
      nz, dt, margin_floor_fraction);
  const int blocks_nodes_j = (nz + 1 + 255) / 256;
  t21_apply_axis_motion_scale_kernel<<<blocks_nodes_j, 256>>>(
      v_r.data(), coupled_v_r, node_scale.data(), nz);
  sync_kernel("Hydro2D axis motion preflight kernels failed");
}

void zero_state_supply_z_position_velocity(core::NodeField1D& pos_v_z,
                                           const int nr,
                                           const int nz,
                                           const bool bottom_active,
                                           const bool top_active) {
  if (!bottom_active && !top_active) {
    return;
  }

  const int blocks = (nr + 1 + 255) / 256;
  zero_state_supply_z_position_velocity_kernel<<<blocks, 256>>>(
      pos_v_z.data(), nr, nz,
      bottom_active ? 1 : 0,
      top_active ? 1 : 0);
  sync_kernel("Hydro2D state_supply z position velocity kernel failed");
}

void zero_state_supply_z_position_velocity(core::NodeField1D& pos_v_z,
                                           const core::Config& cfg,
                                           const int nr,
                                           const int nz) {
  if (!has_state_supply_bc(cfg)) {
    return;
  }

  const auto& b = cfg.numerics.hydro.boundary_2d;
  zero_state_supply_z_position_velocity(
      pos_v_z, nr, nz,
      b.z_bottom_cfg.is_state_supply(),
      b.z_top_cfg.is_state_supply());
}

double axis_z_monotone_sigma(const std::vector<double>& z_axis,
                             const std::vector<double>& vz_axis,
                             const double dt) {
  constexpr double eps_z = 1.0e-12;
  double sigma = 1.0;
  const std::size_t n_axis = z_axis.size();
  if (n_axis < 2U) {
    return sigma;
  }
  for (std::size_t j = 0; j + 1U < n_axis; ++j) {
    const double gap = z_axis[j + 1U] - z_axis[j];
    const double closing = dt * (vz_axis[j] - vz_axis[j + 1U]);
    if (closing > 0.0) {
      sigma = std::min(sigma, (1.0 - eps_z) * gap / closing);
    }
  }
  if (!std::isfinite(sigma) || sigma < 0.0) {
    return 0.0;
  }
  return std::min(1.0, sigma);
}

double constrain_axis_z_position_velocity(core::State& state,
                                          const core::Config& cfg,
                                          core::NodeField1D& pos_v_z,
                                          const core::NodeField1D& x_r_old,
                                          const core::NodeField1D& x_z_old,
                                          const double dt,
                                          const bool count_engaged) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nz < 0) {
    return 1.0;
  }

  const int blocks_axis = (nz + 1 + 255) / 256;
  if (cfg.numerics.ale.axis_z_motion != "lagrangian_tangential") {
    state.axis_lagrangian_tangential_last_sigma = 1.0;
    return 1.0;
  }

  if (count_engaged) {
    ++state.axis_lagrangian_tangential_engaged_count;
  }

  std::vector<double> z_host(static_cast<std::size_t>(nz + 1), 0.0);
  std::vector<double> vz_host(static_cast<std::size_t>(nz + 1), 0.0);
  cuda_check(cudaMemcpy(z_host.data(), x_z_old.data(),
                        static_cast<std::size_t>(nz + 1) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D axis-z copy z host failed");
  cuda_check(cudaMemcpy(vz_host.data(), pos_v_z.data(),
                        static_cast<std::size_t>(nz + 1) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D axis-z copy velocity host failed");

  const double sigma_monotone = axis_z_monotone_sigma(z_host, vz_host, dt);
  core::NodeField1D delta_r;
  core::NodeField1D delta_z;
  const int n_nodes = static_cast<int>(state.x_r.size());
  delta_r.reset(static_cast<std::size_t>(n_nodes));
  delta_z.reset(static_cast<std::size_t>(n_nodes));
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  build_axis_only_z_displacement_kernel<<<nw.blocks(), 256>>>(
      delta_r.data(), delta_z.data(), pos_v_z.data(), nw.begin, nw.end, n_nodes,
      nz, dt);
  sync_kernel("Hydro2D axis-z displacement build failed");

  tenryu::mesh::CandidateMeshAdmissibilityFloors floors;
  const tenryu::mesh::LineSearchResult ls =
      tenryu::mesh::linesearch_largest_admissible_sigma(
          x_r_old.data(), x_z_old.data(), delta_r.data(), delta_z.data(),
          sigma_monotone, 1.0e-3, 20, nr, nz, floors, nullptr,
          state.x_r_reference.size() == state.x_r.size() &&
                  state.x_z_reference.size() == state.x_z.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size() &&
                  state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr);
  const double sigma = ls.sigma_accepted;
  state.axis_lagrangian_tangential_last_sigma = sigma;

  constexpr double eps_floor = 1.0e-3;
  if (!(sigma >= eps_floor)) {
    if (count_engaged) {
      ++state.axis_lagrangian_tangential_ineffective_count;
    }
    zero_axis_z_position_velocity_kernel<<<blocks_axis, 256>>>(pos_v_z.data(), nz);
    sync_kernel("Hydro2D ineffective axis-z position velocity kernel failed");
    if (count_engaged && cfg.main.verbosity == "verbose") {
      tenryu::core::log_warning(
          "[axis-phase8b] step " + std::to_string(state.step) +
          ": sigma collapsed to " + format_scientific(sigma) + " (target " +
          format_scientific(sigma_monotone) + ") — treating as fixed for this step");
    }
    return 0.0;
  }

  scale_axis_z_position_velocity_kernel<<<blocks_axis, 256>>>(
      pos_v_z.data(), nz, sigma);
  sync_kernel("Hydro2D scale axis-z position velocity kernel failed");
  return sigma;
}

const char* hydro_failure_stage_name(
    const tenryu::coupling::HydroFailureStage stage) {
  switch (stage) {
    case tenryu::coupling::HydroFailureStage::Default:
      return "default";
    case tenryu::coupling::HydroFailureStage::StepStart:
      return "step_start";
    case tenryu::coupling::HydroFailureStage::PredictorCommit:
      return "predictor_commit";
    case tenryu::coupling::HydroFailureStage::PostPredictor:
      return "post_predictor";
    case tenryu::coupling::HydroFailureStage::PostCorrector:
      return "post_corrector";
    case tenryu::coupling::HydroFailureStage::RetryRestoreValidate:
      return "retry_restore_validate";
  }
  return "unknown";
}

const char* retry_action_hint_name(
    const tenryu::coupling::RetryActionHint action) {
  switch (action) {
    case tenryu::coupling::RetryActionHint::None:
      return "none";
    case tenryu::coupling::RetryActionHint::ReduceDtOnly:
      return "reduce_dt_only";
    case tenryu::coupling::RetryActionHint::RepairOnly:
      return "repair_only";
    case tenryu::coupling::RetryActionHint::ForceAxisSpinePlusLocalAle:
      return "force_axis_spine_plus_local_ale";
    case tenryu::coupling::RetryActionHint::ForceBoundaryPatchRepair:
      return "force_boundary_patch_repair";
    case tenryu::coupling::RetryActionHint::ForceCdLocalRezone:
      return "force_cd_local_rezone";
    case tenryu::coupling::RetryActionHint::ForceInteriorMultiNodeRepair:
      return "force_interior_multi_node_repair";
    case tenryu::coupling::RetryActionHint::ForceFullWinslow:
      return "force_full_winslow";
  }
  return "unknown";
}

void populate_mesh_quality_dt_failure(
    tenryu::coupling::HydroStepResult& result,
    const core::State& state,
    const core::Config& cfg,
    const MeshQualityDtLimit& limit,
    const tenryu::coupling::HydroFailureStage stage) {
  result.admissible = false;
  result.retry_required = true;
  result.reason = mesh_quality_metric_string(limit.metric);
  result.stage = stage;
  result.first_failing_cell = limit.first_cell;
  result.first_failing_corner = limit.first_corner_or_gauss;
  result.first_failing_i =
      limit.first_cell < 0 ? -1 : limit.first_cell / state.mesh.topo.nz;
  result.first_failing_j =
      limit.first_cell < 0 ? -1 : limit.first_cell % state.mesh.topo.nz;
  result.min_metric = limit.sigma_safe;
  result.failing_value = limit.sigma_safe;
  result.trial_scale = limit.sigma_safe;
  result.suggested_dt = std::max(cfg.numerics.dt.min_s, limit.suggested_dt);
  int band = cfg.numerics.hydro.axis_guard_band_cells;
  bool in_band = result.first_failing_i >= 0 && result.first_failing_i <= band;
  result.retry_action =
      in_band
          ? tenryu::coupling::RetryActionHint::ForceAxisSpinePlusLocalAle
          : tenryu::coupling::RetryActionHint::ReduceDtOnly;
  if (result.first_failing_i == 0) {
    result.regime = MeshFailureRegime::AxisFace;
  } else if (in_band) {
    result.regime = MeshFailureRegime::AxisBand;
  }
}

void populate_path_guard_failure(
    tenryu::coupling::HydroStepResult& result,
    const core::State& state,
    const core::Config& cfg,
    const MeshQualityDtLimit& limit,
    const tenryu::coupling::HydroFailureStage failure_stage,
    const double full_step_dt) {
  populate_mesh_quality_dt_failure(
      result, state, cfg, limit, failure_stage);
  const double dt_new = 0.5 * limit.sigma_safe * full_step_dt;
  result.min_metric = limit.min_value;
  result.failing_value = limit.min_value;
  result.suggested_dt = dt_new;
  result.retry_action = tenryu::coupling::RetryActionHint::ReduceDtOnly;
  result.geometry_soft_failure.valid = true;
  result.geometry_soft_failure.cell = limit.first_cell;
  result.geometry_soft_failure.corner_or_slot =
      limit.first_corner_or_gauss;
  result.geometry_soft_failure.predicate =
      mesh_quality_predicate_string(limit.metric);
  result.geometry_soft_failure.min_value = limit.min_value;
  result.geometry_soft_failure.sigma_safe = limit.sigma_safe;
}

bool multiblock_path_admissibility_precommit_enabled(
    const core::State& state,
    const core::Config& cfg) {
  return cfg.numerics.ale.multiblock_path_admissibility_enabled &&
         tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh) &&
         state.mesh.topo.multiblock.has_value();
}

bool multiblock_path_admissibility_failed(
    const tenryu::mesh::PathAdmissibilityResult& path_check) {
  return path_check.first_failing_cell != -1 &&
         path_check.first_failing_lambda >= 0.0 &&
         path_check.first_failing_lambda < 1.0;
}

std::string format_multiblock_path_admissibility_location(
    const core::State& state,
    const core::Config& cfg,
    const tenryu::mesh::PathAdmissibilityResult& path_check);

std::string format_multiblock_path_admissibility_anatomy(
    const tenryu::mesh::PathAdmissibilityResult& path_check);

tenryu::hydro::pole_angular_coarsen::Overlay build_pole_path_overlay(
    const core::State& state,
    const core::Config& cfg) {
  auto derefine_overlay = tenryu::hydro::pole_angular_derefine::path_overlay(
      const_cast<core::State&>(state), cfg);
  if (derefine_overlay.active) {
    return derefine_overlay;
  }
  auto motion_overlay =
      tenryu::hydro::pole_angular_coarsen::build_motion_overlay(state, cfg);
  if (motion_overlay.active) {
    return motion_overlay;
  }
  return tenryu::hydro::pole_angular_coarsen::build_overlay(state, cfg);
}

std::string format_pole_motion_taper(
    const tenryu::hydro::pole_angular_coarsen::MotionPilotResult& result) {
  std::ostringstream oss;
  oss << ", profile=" << result.profile
      << ", transition_rows=" << result.transition_rows
      << ", free_q_le=" << result.q_free
      << ", full_q_ge=" << result.q_full
      << ", alpha_rows=";
  if (result.taper_rows.empty()) {
    oss << "none";
  } else {
    for (std::size_t k = 0; k < result.taper_rows.size(); ++k) {
      if (k > 0) {
        oss << "/";
      }
      oss << "q" << result.taper_rows[k].q << "=" << std::fixed
          << std::setprecision(6) << result.taper_rows[k].alpha;
    }
  }
  return oss.str();
}

std::string format_pole_motion_transition_range(
    const tenryu::hydro::pole_angular_coarsen::MotionPilotResult& result) {
  if (result.transition_i_min < 0 ||
      result.transition_i_max < result.transition_i_min) {
    return "none";
  }
  return std::to_string(result.transition_i_min) + ":" +
         std::to_string(result.transition_i_max + 1);
}

tenryu::hydro::pole_angular_coarsen::MotionPilotResult
apply_pole_motion_pilot_position_velocity(
    const core::State& state,
    const core::Config& cfg,
    const tenryu::hydro::pole_angular_coarsen::Overlay& source_overlay,
    const core::NodeField1D& r_old,
    const core::NodeField1D& z_old,
    core::NodeField1D& pos_v_r,
    core::NodeField1D& pos_v_z,
    const double dt,
    const char* stage) {
  tenryu::hydro::pole_angular_coarsen::MotionPilotResult result;
  result.active = source_overlay.active;
  if (!source_overlay.active) {
    return result;
  }

  static bool marker_logged = false;
  if (!marker_logged) {
    marker_logged = true;
    tenryu::core::log_warning(
        "[pole_motion_pilot] geometry/path-only pilot active: overwriting "
        "POLAR_SHELL pole-band mesh position velocity w from accepted coarse "
        "macro motion; material velocity, CCH force/work, and conservation "
        "accounting are unchanged");
  }

  const int n_nodes = static_cast<int>(r_old.size());
  const int n_cells = state.mesh.topo.n_cells;
  auto xr_old = copy_field_to_vector(r_old);
  auto xz_old = copy_field_to_vector(z_old);
  auto pos_r = copy_field_to_vector(pos_v_r);
  auto pos_z = copy_field_to_vector(pos_v_z);
  std::vector<double> xr_new(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> xz_new(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    xr_new[static_cast<std::size_t>(n)] =
        xr_old[static_cast<std::size_t>(n)] +
        dt * pos_r[static_cast<std::size_t>(n)];
    xz_new[static_cast<std::size_t>(n)] =
        xz_old[static_cast<std::size_t>(n)] +
        dt * pos_z[static_cast<std::size_t>(n)];
  }

  tenryu::hydro::pole_angular_coarsen::Overlay accepted_overlay;
  std::vector<double> xr_reference;
  std::vector<double> xz_reference;
  const double floor_rel =
      std::max(0.0, cfg.numerics.ale.path_admissibility_floor);
  if (floor_rel > 0.0 &&
      state.x_r_reference.size() == static_cast<std::size_t>(n_nodes) &&
      state.x_z_reference.size() == static_cast<std::size_t>(n_nodes)) {
    xr_reference = copy_field_to_vector(state.x_r_reference);
    xz_reference = copy_field_to_vector(state.x_z_reference);
  }
  const auto cover = tenryu::mesh::prepare_pole_coarsen_overlay_host(
      &source_overlay,
      n_cells,
      xr_old,
      xz_old,
      xr_new,
      xz_new,
      floor_rel,
      accepted_overlay,
      xr_reference.empty() ? nullptr : &xr_reference,
      xz_reference.empty() ? nullptr : &xz_reference);
  if (cover.first_failing_cell != -1) {
    tenryu::core::log_warning(
        std::string("[pole_motion_pilot] accepted macro cover failed at ") +
        stage + ", no mesh-velocity overwrite applied, first_cell=" +
        std::to_string(cover.first_failing_cell) + ", lambda=" +
        format_scientific(cover.first_failing_lambda) +
        format_multiblock_path_admissibility_location(state, cfg, cover) +
        format_multiblock_path_admissibility_anatomy(cover));
    return result;
  }

  result = tenryu::hydro::pole_angular_coarsen::apply_coherent_motion_host(
      accepted_overlay, xr_old, xz_old, pos_r, pos_z, dt);
  if (!result.applied) {
    return result;
  }

  pos_v_r.copy_from_host(pos_r);
  pos_v_z.copy_from_host(pos_z);
  static bool apply_logged = false;
  if (!apply_logged) {
    apply_logged = true;
    tenryu::core::log_warning(
        std::string("[pole_motion_pilot] first coherent mesh-velocity "
                    "overwrite stage=") +
        stage + ", q_range=" + std::to_string(accepted_overlay.q_min) + ":" +
        std::to_string(accepted_overlay.q_max + 1) + ", level_max=" +
        std::to_string(accepted_overlay.level_max) + ", macros=" +
        std::to_string(result.macro_count) + ", nodes=" +
        std::to_string(result.nodes_overwritten) + ", transition_i_range=" +
        format_pole_motion_transition_range(result) +
        ", transition_nodes=" + std::to_string(result.transition_nodes) +
        format_pole_motion_taper(result));
  }
  return result;
}

const char* multiblock_location_block_name(const mesh::BlockRole role) {
  switch (role) {
    case mesh::BlockRole::CENTRAL_CORE:
      return "core";
    case mesh::BlockRole::BRIDGE:
      return "bridge";
    case mesh::BlockRole::NORTH_FAN:
      return "north_fan";
    case mesh::BlockRole::EAST_FAN:
      return "east_fan";
    case mesh::BlockRole::SOUTH_FAN:
      return "south_fan";
    case mesh::BlockRole::POLAR_SHELL:
      return "shell";
  }
  return "unknown";
}

const mesh::BlockInfo* multiblock_cell_block(
    const mesh::MultiBlockTopology& mb,
    const int cell,
    int* block_id_out) {
  if (cell >= 0 && mb.cell_block_id.size() > static_cast<std::size_t>(cell)) {
    const int block_id = mb.cell_block_id[static_cast<std::size_t>(cell)];
    if (block_id >= 0 &&
        block_id < static_cast<int>(mb.blocks.size())) {
      const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
      if (cell >= block.cell_begin &&
          cell < block.cell_begin + block.cell_count) {
        if (block_id_out != nullptr) {
          *block_id_out = block_id;
        }
        return &block;
      }
    }
  }
  for (int block_id = 0; block_id < static_cast<int>(mb.blocks.size());
       ++block_id) {
    const auto& block = mb.blocks[static_cast<std::size_t>(block_id)];
    if (cell >= block.cell_begin &&
        cell < block.cell_begin + block.cell_count) {
      if (block_id_out != nullptr) {
        *block_id_out = block_id;
      }
      return &block;
    }
  }
  return nullptr;
}

const char* multiblock_angular_sector(const mesh::BlockRole role,
                                      const int local_j,
                                      const int n_c) {
  switch (role) {
    case mesh::BlockRole::CENTRAL_CORE:
      return "central";
    case mesh::BlockRole::NORTH_FAN:
      return "north";
    case mesh::BlockRole::EAST_FAN:
      return "east";
    case mesh::BlockRole::SOUTH_FAN:
      return "south";
    case mesh::BlockRole::BRIDGE:
    case mesh::BlockRole::POLAR_SHELL:
      if (n_c > 0 && local_j >= 0) {
        if (local_j < n_c) {
          return "north";
        }
        if (local_j < 3 * n_c) {
          return "east";
        }
        if (local_j < 4 * n_c) {
          return "south";
        }
      }
      return "unknown";
  }
  return "unknown";
}

std::string multiblock_cell_boundary_label(const mesh::MultiBlockTopology& mb,
                                           const int cell) {
  if (cell < 0 ||
      mb.face_adj_csr_offsets.size() <= static_cast<std::size_t>(cell + 1) ||
      mb.face_bc_tags.empty()) {
    return "unknown";
  }
  const int off = mb.face_adj_csr_offsets[static_cast<std::size_t>(cell)];
  const int next = mb.face_adj_csr_offsets[static_cast<std::size_t>(cell) + 1U];
  if (off < 0 || next < off ||
      next > static_cast<int>(mb.face_bc_tags.size())) {
    return "unknown";
  }
  for (int p = off; p < next; ++p) {
    const auto kind = static_cast<mesh::BoundaryKind>(
        mb.face_bc_tags[static_cast<std::size_t>(p)]);
    if (kind == mesh::BoundaryKind::SEAM_CORE_BRIDGE ||
        kind == mesh::BoundaryKind::SEAM_BRIDGE_SHELL) {
      return mesh::boundary_kind_name(kind);
    }
  }
  for (int p = off; p < next; ++p) {
    const auto kind = static_cast<mesh::BoundaryKind>(
        mb.face_bc_tags[static_cast<std::size_t>(p)]);
    if (kind != mesh::BoundaryKind::Interior) {
      return mesh::boundary_kind_name(kind);
    }
  }
  return "interior";
}

std::string format_multiblock_path_admissibility_location(
    const core::State& state,
    const core::Config& cfg,
    const tenryu::mesh::PathAdmissibilityResult& path_check) {
  if (path_check.first_failing_cell ==
      tenryu::mesh::kPoleAngularCoarsenSentinelCell) {
    std::string reason = "unknown";
    if (path_check.macro_has_anatomy != 0) {
      const int area_kind = static_cast<int>(
          tenryu::mesh::PathAdmissibilityMetricKind::AREA_OR_RZ_VOLUME);
      const int edge_kind = static_cast<int>(
          tenryu::mesh::PathAdmissibilityMetricKind::EDGE_CROSS);
      if (!std::isfinite(path_check.macro_v_old) ||
          !std::isfinite(path_check.macro_v_new) ||
          path_check.macro_v_old == 0.0) {
        reason = "nonfinite_or_vold0";
      } else if (path_check.first_failing_metric_kind == area_kind) {
        reason = "rz_volume_path";
      } else if (path_check.first_failing_metric_kind == edge_kind) {
        reason = "simple_loop_path";
      } else {
        reason = "none";
      }
    }
    std::string out = ", pole_coarsen_boundary=1, macro_reason=" + reason;
    if (path_check.macro_has_anatomy != 0) {
      out += ", macro_v_old=" + format_scientific(path_check.macro_v_old) +
             ", macro_v_new=" + format_scientific(path_check.macro_v_new) +
             ", macro_floor=" + format_scientific(path_check.macro_floor) +
             ", span=" + std::to_string(path_check.macro_span) +
             ", level=" + std::to_string(path_check.macro_level) +
             ", q_range=" + std::to_string(path_check.macro_q_begin) + ":" +
             std::to_string(path_check.macro_q_end) +
             ", j_range=" + std::to_string(path_check.macro_j_begin) + ":" +
             std::to_string(path_check.macro_j_end) +
             ", skipped_nodes=" +
             std::to_string(path_check.macro_skipped_nodes) +
             ", macro_simple_ok=" +
             std::to_string(path_check.macro_simple_ok);
      if (path_check.macro_simple_ok == 0) {
        out += ", viol_nodes=" +
               std::to_string(path_check.macro_viol_node_a) + "/" +
               std::to_string(path_check.macro_viol_node_b) +
               ", viol_theta=" +
               format_scientific(path_check.macro_viol_theta_a) + "/" +
               format_scientific(path_check.macro_viol_theta_b);
      }
    }
    return out;
  }
  if (path_check.first_failing_cell ==
      tenryu::mesh::kCentralMacroCoreSentinelCell) {
    // Terminal-failure forensics (always on: macro rejections are rare and
    // the volume-floor vs simple-loop distinction plus the angular location
    // of the violating segments is decision-critical for the terminal phase).
    std::string reason = "unknown";
    if (path_check.macro_has_anatomy != 0) {
      if (!std::isfinite(path_check.macro_v_old) ||
          !std::isfinite(path_check.macro_v_new) ||
          path_check.macro_v_old <= 0.0) {
        reason = "nonfinite_or_vold0";
      } else if (path_check.macro_v_new <= path_check.macro_floor) {
        reason = "volume_floor";
      } else if (path_check.macro_simple_ok == 0) {
        reason = "simple_loop";
      } else {
        reason = "none";
      }
    }
    std::string out = ", macro_core_boundary=1, macro_reason=" + reason;
    if (path_check.macro_has_anatomy != 0) {
      out += ", macro_v_old=" + format_scientific(path_check.macro_v_old) +
             ", macro_v_new=" + format_scientific(path_check.macro_v_new) +
             ", macro_floor=" + format_scientific(path_check.macro_floor) +
             ", macro_simple_ok=" +
             std::to_string(path_check.macro_simple_ok);
      if (path_check.macro_simple_ok == 0) {
        out += ", viol_nodes=" +
               std::to_string(path_check.macro_viol_node_a) + "/" +
               std::to_string(path_check.macro_viol_node_b) +
               ", viol_theta=" +
               format_scientific(path_check.macro_viol_theta_a) + "/" +
               format_scientific(path_check.macro_viol_theta_b);
      }
    }
    return out;
  }
  if (!cfg.numerics.debug.trace_mesh_motion ||
      path_check.first_failing_cell == -1 ||
      !state.mesh.topo.multiblock.has_value()) {
    return "";
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int cell = path_check.first_failing_cell;
  int block_id = -1;
  const mesh::BlockInfo* block = multiblock_cell_block(mb, cell, &block_id);
  if (block == nullptr || block->n_j_cells <= 0) {
    return ", block=unknown, layer=-1, angular_k=-1";
  }
  const int local = cell - block->cell_begin;
  if (local < 0 || local >= block->cell_count) {
    return ", block=unknown, layer=-1, angular_k=-1";
  }
  const int local_i = local / block->n_j_cells;
  const int local_j = local - local_i * block->n_j_cells;
  if (cfg.mesh.topology_scheme ==
      core::TopologyScheme::MULTIBLOCK_CART_CORE_POLAR_SHELL) {
    return ", block=" +
           std::string(multiblock_location_block_name(block->role)) +
           ", layer=" + std::to_string(local_i) +
           ", angular_k=" + std::to_string(local_j);
  }
  const int n_c = cfg.mesh.multiblock_cart_core_n_c;
  return ", block_role=" +
         std::string(multiblock_location_block_name(block->role)) +
         ", block_id=" + std::to_string(block_id) +
         ", local_i=" + std::to_string(local_i) +
         ", local_j=" + std::to_string(local_j) +
         ", seam_sector=" +
         multiblock_angular_sector(block->role, local_j, n_c) +
         ", boundary=" + multiblock_cell_boundary_label(mb, cell);
}

std::string format_multiblock_path_admissibility_anatomy(
    const tenryu::mesh::PathAdmissibilityResult& path_check) {
  if (!tenryu::mesh::path_admissibility_anatomy_enabled()) {
    return "";
  }
  return ", path_source=" +
         std::string(tenryu::mesh::path_admissibility_source_kind_name(
             path_check.path_source_kind)) +
         ", metric_kind=" +
         std::string(tenryu::mesh::path_admissibility_metric_kind_name(
             path_check.first_failing_metric_kind)) +
         ", metric_block=" +
         std::to_string(path_check.first_failing_block_id) +
         ", metric_local_i=" +
         std::to_string(path_check.first_failing_local_i) +
         ", metric_local_j=" +
         std::to_string(path_check.first_failing_local_j) +
         ", old_geometry_inadmissible=" +
         std::to_string(path_check.old_geometry_inadmissible) +
         ", macro_repair_required=" +
         std::to_string(path_check.macro_repair_required) +
         ", pole_axis_radial_order_inversion=" +
         std::to_string(path_check.pole_axis_radial_order_inversion);
}

void populate_multiblock_path_admissibility_failure(
    tenryu::coupling::HydroStepResult& result,
    const core::State& state,
    const core::Config& cfg,
    const tenryu::mesh::PathAdmissibilityResult& path_check,
    const double dt,
    const tenryu::coupling::HydroFailureStage stage) {
  observe_path_admissibility_margin(result, path_check);
  const double lambda =
      std::isfinite(path_check.first_failing_lambda)
          ? std::min(std::max(path_check.first_failing_lambda, 0.0), 1.0)
          : 0.0;
  const bool macro_repair_required =
      path_check.macro_repair_required != 0;
  const bool pole_axis_radial_order_inversion =
      path_check.pole_axis_radial_order_inversion != 0;
  result.admissible = false;
  result.retry_required = true;
  if (pole_axis_radial_order_inversion) {
    result.reason = "pole_axis_radial_order_inversion";
  } else if (macro_repair_required) {
    result.reason = "macro_repair_required";
  } else if (path_check.first_failing_cell ==
             tenryu::mesh::kCentralMacroCoreSentinelCell) {
    result.reason = "macro_core_path_admissibility";
  } else if (path_check.first_failing_cell ==
             tenryu::mesh::kPoleAngularCoarsenSentinelCell) {
    result.reason = "pole_coarsen_path_admissibility";
  } else {
    result.reason = "multiblock_path_admissibility";
  }
  result.first_failing_cell = path_check.first_failing_cell;
  result.first_failing_corner = -1;
  result.min_metric = path_check.min_margin;
  result.failing_value = path_check.min_margin;
  result.min_corner_j = path_check.min_margin;
  result.trial_scale = lambda;
  result.path_source_kind = path_check.path_source_kind;
  result.path_metric_kind = path_check.first_failing_metric_kind;
  result.path_block_id = path_check.first_failing_block_id;
  result.path_local_i = path_check.first_failing_local_i;
  result.path_local_j = path_check.first_failing_local_j;
  result.path_old_geometry_inadmissible =
      path_check.old_geometry_inadmissible;
  result.macro_q_begin = path_check.macro_q_begin;
  result.macro_q_end = path_check.macro_q_end;
  result.macro_j_begin = path_check.macro_j_begin;
  result.macro_j_end = path_check.macro_j_end;
  result.macro_span = path_check.macro_span;
  result.macro_level = path_check.macro_level;
  result.stage = stage;
  result.retry_action =
      (macro_repair_required || pole_axis_radial_order_inversion)
          ? tenryu::coupling::RetryActionHint::RepairOnly
          : tenryu::coupling::RetryActionHint::ReduceDtOnly;
  result.geometry_failure_kind =
      tenryu::mesh::MeshGeometryFailureKind::TopologyInvariantViolation;
  if (path_check.first_failing_cell >= 0 && state.mesh.topo.nz > 0) {
    result.first_failing_i = path_check.first_failing_cell / state.mesh.topo.nz;
    result.first_failing_j = path_check.first_failing_cell % state.mesh.topo.nz;
  }
  result.suggested_dt =
      (macro_repair_required || pole_axis_radial_order_inversion)
          ? 0.0
          : std::max(cfg.numerics.dt.min_s,
                     cfg.numerics.ale.dt_rejection_factor * lambda * dt);
}

tenryu::coupling::RetryActionHint geometry_retry_action_for_stage(
    const tenryu::coupling::HydroFailureStage stage) {
  if (stage == tenryu::coupling::HydroFailureStage::StepStart ||
      stage == tenryu::coupling::HydroFailureStage::RetryRestoreValidate) {
    return tenryu::coupling::RetryActionHint::RepairOnly;
  }
  return tenryu::coupling::RetryActionHint::ReduceDtOnly;
}

tenryu::coupling::HydroStepResult hydro_result_from_geometry_failure(
    const tenryu::mesh::MeshGeometryResult& geometry,
    const tenryu::coupling::HydroFailureStage stage) {
  tenryu::coupling::HydroStepResult result{};
  result.admissible = false;
  result.retry_required = true;
  result.reason = "mesh_geometry";
  result.first_failing_cell = geometry.failing_cell;
  result.first_failing_corner = -1;
  result.min_metric = geometry.failing_value;
  result.suggested_dt = 0.0;
  result.stage = stage;
  result.retry_action = geometry_retry_action_for_stage(stage);
  result.geometry_failure_kind = geometry.kind;
  result.first_failing_i = geometry.failing_i;
  result.first_failing_j = geometry.failing_j;
  result.failing_value = geometry.failing_value;
  result.min_cell_vol = geometry.min_cell_vol;
  result.min_cell_area = geometry.min_cell_area;
  return result;
}

std::string format_hydro_geometry_failure_record(
    const tenryu::coupling::HydroStepResult& result) {
  std::ostringstream oss;
  oss << "stage=" << hydro_failure_stage_name(result.stage)
      << " kind="
      << tenryu::mesh::mesh_geometry_failure_kind_name(
             result.geometry_failure_kind)
      << " failing_cell=" << result.first_failing_cell
      << " failing_i=" << result.first_failing_i
      << " failing_j=" << result.first_failing_j
      << " failing_value=" << format_scientific(result.failing_value)
      << " min_cell_vol=" << format_scientific(result.min_cell_vol)
      << " min_cell_area=" << format_scientific(result.min_cell_area)
      << " retry_action=" << retry_action_hint_name(result.retry_action);
  return oss.str();
}

bool compute_density_guard_diag_enabled() {
  const char* const env = std::getenv("TENRYU_I1B_COMPUTE_DENSITY_GUARD_DIAG");
  return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

bool compute_density_diag_cell_nodes(const core::State& state,
                                     const int c,
                                     int nodes[4],
                                     int* active_nverts_out) {
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlots; ++k) {
    nodes[k] = -1;
  }
  const int n_cells = static_cast<int>(state.mass.size());
  if (c < 0 || c >= n_cells) {
    return false;
  }
  const int active_nverts =
      state.mesh.cell_nverts.size() == static_cast<std::size_t>(n_cells)
          ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
          : mesh::kMeshTopoCellStorageSlots;
  if (active_nverts_out != nullptr) {
    *active_nverts_out = active_nverts;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    if (mb.cell_node_csr_offsets.size() < static_cast<std::size_t>(c + 2)) {
      return false;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
    if (off < 0 || end - off < active_nverts ||
        end > static_cast<int>(mb.cell_node_csr_indices.size())) {
      return false;
    }
    for (int k = 0; k < active_nverts; ++k) {
      nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    }
    return true;
  }
  const int nz = state.mesh.topo.nz;
  if (nz <= 0) {
    return false;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  nodes[0] = i * stride + j;
  nodes[1] = (i + 1) * stride + j;
  nodes[2] = (i + 1) * stride + (j + 1);
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    nodes[3] = i * stride + (j + 1);
  }
  return true;
}

double compute_density_diag_geometry_volume(const std::vector<double>& r,
                                            const std::vector<double>& z,
                                            const int nodes[4],
                                            const int active_nverts) {
  for (int k = 0; k < active_nverts; ++k) {
    if (nodes[k] < 0 || static_cast<std::size_t>(nodes[k]) >= r.size() ||
        static_cast<std::size_t>(nodes[k]) >= z.size()) {
      return std::numeric_limits<double>::quiet_NaN();
    }
  }
  if (active_nverts == 3) {
    return tenryu::mesh::rz_triangle_volume_exact(
        r[static_cast<std::size_t>(nodes[0])],
        z[static_cast<std::size_t>(nodes[0])],
        r[static_cast<std::size_t>(nodes[1])],
        z[static_cast<std::size_t>(nodes[1])],
        r[static_cast<std::size_t>(nodes[2])],
        z[static_cast<std::size_t>(nodes[2])]);
  }
  return tenryu::mesh::rz_quad_volume_exact(
      r[static_cast<std::size_t>(nodes[0])],
      z[static_cast<std::size_t>(nodes[0])],
      r[static_cast<std::size_t>(nodes[1])],
      z[static_cast<std::size_t>(nodes[1])],
      r[static_cast<std::size_t>(nodes[2])],
      z[static_cast<std::size_t>(nodes[2])],
      r[static_cast<std::size_t>(nodes[3])],
      z[static_cast<std::size_t>(nodes[3])]);
}

bool compute_density_diag_mask_has(const std::vector<std::uint8_t>& mask,
                                   const int c) {
  return c >= 0 && static_cast<std::size_t>(c) < mask.size() &&
         mask[static_cast<std::size_t>(c)] != 0U;
}

void emit_compute_density_guard_diag(
    const core::State& state,
    const core::Config& cfg,
    const tenryu::coupling::HydroFailureStage stage,
    const bool shell_subcycle_committed_hint) {
  if (!compute_density_guard_diag_enabled()) {
    return;
  }
  const int n_cells = static_cast<int>(state.mass.size());
  if (n_cells <= 0 || state.vol.size() != state.mass.size()) {
    return;
  }

  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> r;
  std::vector<double> z;
  state.mass.copy_to_host(mass);
  state.vol.copy_to_host(vol);
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);

  const bool shell_subcycle_env = shell_subcycle_requested();
  int emitted = 0;
  constexpr int kMaxEmit = 8;
  for (int c = 0; c < n_cells && emitted < kMaxEmit; ++c) {
    const double v = vol[static_cast<std::size_t>(c)];
    const double m = mass[static_cast<std::size_t>(c)];
    const double rho = (v > 0.0) ? (m / v)
                                 : std::numeric_limits<double>::quiet_NaN();
    if (v > 0.0 && std::isfinite(v) && std::isfinite(m) &&
        std::isfinite(rho) && rho > 0.0) {
      continue;
    }

    int block_id = -1;
    int local_i = -1;
    int local_j = -1;
    const mesh::BlockInfo* block = nullptr;
    if (state.mesh.topo.multiblock.has_value()) {
      const auto& mb = *state.mesh.topo.multiblock;
      block = multiblock_cell_block(mb, c, &block_id);
      if (block != nullptr && block->n_j_cells > 0) {
        const int local = c - block->cell_begin;
        if (local >= 0 && local < block->cell_count) {
          local_i = local / block->n_j_cells;
          local_j = local - local_i * block->n_j_cells;
        }
      }
    } else if (state.mesh.topo.nz > 0) {
      local_i = c / state.mesh.topo.nz;
      local_j = c - local_i * state.mesh.topo.nz;
    }

    int nodes[4] = {-1, -1, -1, -1};
    int active_nverts = mesh::kMeshTopoCellStorageSlots;
    const bool have_nodes =
        compute_density_diag_cell_nodes(state, c, nodes, &active_nverts);
    const double v_geom =
        have_nodes ? compute_density_diag_geometry_volume(r, z, nodes,
                                                          active_nverts)
                   : std::numeric_limits<double>::quiet_NaN();

    const bool shell_cell =
        block != nullptr && block->role == mesh::BlockRole::POLAR_SHELL;
    const bool central_member = compute_density_diag_mask_has(
        state.central_pseudo_core.member_mask, c);
    const bool deref_member = compute_density_diag_mask_has(
        state.pole_angular_derefine.member_mask, c);
    const bool cap_cell =
        state.mesh.topo.multiblock.has_value() &&
        state.mesh.topo.multiblock->has_trifan_cap && block != nullptr &&
        block->role == mesh::BlockRole::CENTRAL_CORE;
    const char* region = "other";
    if (shell_cell) {
      region = "POLAR_SHELL";
    } else if (central_member) {
      region = "central_pseudo_core_member";
    } else if (deref_member) {
      region = "pole_angular_deref_member";
    } else if (cap_cell) {
      region = "cap";
    }

    std::ostringstream os;
    os << std::setprecision(17)
       << "[compute_density_guard_diag]"
       << " step=" << state.step
       << " t=" << state.t
       << " stage=" << hydro_failure_stage_name(stage)
       << " cell_id=" << c
       << " block_id=" << block_id
       << " block_role="
       << (block != nullptr ? multiblock_location_block_name(block->role)
                            : "single_block")
       << " local_i=" << local_i
       << " local_j=" << local_j
       << " region=" << region
       << " shell_subcycle_scope=" << (shell_subcycle_env && shell_cell ? 1 : 0)
       << " touched_by_subcycle_this_step="
       << (shell_subcycle_committed_hint && shell_cell ? 1 : 0)
       << " V_c=" << v
       << " V_geom_signed=" << v_geom
       << " m_c=" << m
       << " rho_c=" << rho
       << " active_nverts=" << active_nverts
       << " central_member=" << (central_member ? 1 : 0)
       << " deref_member=" << (deref_member ? 1 : 0)
       << " cap_member=" << (cap_cell ? 1 : 0)
       << " cell_is_void="
       << (c >= 0 && static_cast<std::size_t>(c) < state.cell_is_void.size()
               ? static_cast<int>(state.cell_is_void[static_cast<std::size_t>(c)])
               : -1);
    for (int k = 0; k < mesh::kMeshTopoCellStorageSlots; ++k) {
      const int n = nodes[k];
      const double rk =
          n >= 0 && static_cast<std::size_t>(n) < r.size()
              ? r[static_cast<std::size_t>(n)]
              : std::numeric_limits<double>::quiet_NaN();
      const double zk =
          n >= 0 && static_cast<std::size_t>(n) < z.size()
              ? z[static_cast<std::size_t>(n)]
              : std::numeric_limits<double>::quiet_NaN();
      os << " node" << k << "=" << n
         << " r" << k << "=" << rk
         << " z" << k << "=" << zk;
    }
    if (state.mesh.topo.multiblock.has_value() && block != nullptr) {
      os << " seam_sector="
         << multiblock_angular_sector(block->role, local_j,
                                      cfg.mesh.multiblock_cart_core_n_c)
         << " boundary="
         << multiblock_cell_boundary_label(*state.mesh.topo.multiblock, c);
    }
    tenryu::core::log_warning(os.str());
    ++emitted;
  }
}

tenryu::coupling::HydroStepResult refresh_geometry_and_density_impl(
    core::State& state,
    const core::Config& cfg,
    const tenryu::coupling::HydroFailureStage stage,
    int* rho_clamp_count = nullptr,
    tenryu::coupling::ProfileObservability* observability = nullptr,
    const bool shell_subcycle_committed_hint = false) {
  tenryu::coupling::HydroStepResult result{};
  const bool retry_can_handle_geometry =
      cfg.numerics.hydro.driver_full_step_retry_enabled ||
      tenryu::core::effective_mesh_geometry_soft_fail(cfg) ||
      i1b_path_guard_enabled();
  tenryu::mesh::MeshGeometryCheckOptions opts{};
  opts.policy = retry_can_handle_geometry
                    ? tenryu::mesh::MeshGeometryFailurePolicy::SoftReturn
                    : tenryu::mesh::MeshGeometryFailurePolicy::HardAssert;
  std::vector<std::uint8_t> inactive_cell_mask;
  const auto combine_inactive_mask =
      [&](const std::vector<std::uint8_t>& mask) {
        if (mask.empty()) {
          return;
        }
        TENRYU_ASSERT(
            mask.size() ==
                static_cast<std::size_t>(state.mesh.topo.n_cells),
            "Hydro2D geometry inactive cell mask size mismatch");
        if (inactive_cell_mask.empty()) {
          inactive_cell_mask.assign(mask.size(), 0U);
        }
        for (std::size_t c = 0; c < mask.size(); ++c) {
          inactive_cell_mask[c] =
              static_cast<std::uint8_t>(inactive_cell_mask[c] | mask[c]);
        }
      };
  combine_inactive_mask(state.central_pseudo_core.inactive_member_mask);
  combine_inactive_mask(state.pole_angular_derefine.inactive_member_mask);
  // Evacuated cells stay SUBJECT to the geometry admissibility check (signed
  // volume / inversion). Consultation #21 §7.1 exempts them only from QUALITY
  // penalties, handled via the d_hydro_force_active channel. Measured incident:
  // the inactive spacer at (0,283) was winslow-inverted when this check skipped
  // it (ledger A474).
  if (!inactive_cell_mask.empty()) {
    opts.inactive_cell_mask = inactive_cell_mask.data();
  }
  const tenryu::mesh::MeshGeometryResult geometry =
      state.mesh.recompute_geometry_checked(opts);
  if (!geometry.admissible) {
    if (observability != nullptr) {
      observability->note_mesh_geometry_failure(geometry);
    }
    result = hydro_result_from_geometry_failure(geometry, stage);
    if (cfg.numerics.hydro.driver_full_step_retry_enabled) {
      result.reason = "geometry_inversion";
      result.retry_action =
          tenryu::coupling::RetryActionHint::ForceFullWinslow;
    }
    return result;
  }
  if (state.vol.size() == 0) {
    state.vol = state.mesh.cell_vol;  // one-time init sizing path
  } else {
    TENRYU_ASSERT(
        state.mesh.cell_vol_device.size() == state.vol.size(),
        "Hydro2D refresh: cell_vol_device/state.vol size mismatch");
    cuda_check(cudaMemcpy(state.vol.data(),
                          state.mesh.cell_vol_device.data(),
                          state.vol.size() * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D refresh: cell_vol D2D failed");
  }

  TENRYU_ASSERT(state.mass.size() == state.vol.size(),
                "Hydro2D requires mass/volume size consistency");
  TENRYU_ASSERT(state.cell_is_void.size() == state.mass.size(),
                "Hydro2D requires cell_is_void/mass size consistency");
  const int n_cells = static_cast<int>(state.mass.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  // Ghost-inclusive compute window (OPEN-HYDRO-CORNER): ghost rho must track
  // the owner bitwise (inputs mass/vol are exchange-/geometry-fresh there);
  // the clamp counter stays owned-windowed via cw.
  const core::State::LaunchWindow dw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  emit_compute_density_guard_diag(
      state, cfg, stage, shell_subcycle_committed_hint);
  std::uint8_t* d_cell_is_void = nullptr;
  d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "h2d:cell_is_void_refresh",
      static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_cell_is_void,
                        state.cell_is_void.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "Hydro2D: cudaMemcpy cell_is_void failed");
  const FloorClampExclusionMasks exclusion_masks =
      floor_clamp_exclusion_masks(state);
  compute_density_kernel<<<dw.blocks(), 256>>>(
      state.rho.data(), state.mass.data(), state.vol.data(), d_cell_is_void,
      exclusion_masks.central_member, exclusion_masks.central_passive,
      exclusion_masks.pole_member,
      dw.begin, dw.end, cw.begin, cw.end, n_cells, cfg.numerics.floors.rho,
      rho_clamp_count);
  sync_kernel("Hydro2D compute_density kernel failed");
  return result;
}

bool recompute_corner_mass_each_step(const core::Config& cfg) {
  return cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled &&
         !corner_mass_lagrangian_invariant_enabled(cfg);
}

void ensure_corner_mass_2d(core::State& state,
                           const std::uint8_t* d_cell_nverts,
                           const int corner_mass_convention,
                           const bool canonical_subzonal_partition,
                           const bool force_recompute,
                           const bool lagrangian_invariant,
                           const bool aw_compatible_force_work,
                           const bool polar_tier_equal_planar_area) {
  const int n_cells = static_cast<int>(state.mass.size());
  const std::size_t expected_size =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  state.corner_mass_is_lagrangian_invariant = lagrangian_invariant;
  bool recompute = force_recompute;
  if (state.corner_mass.size() != expected_size) {
    TENRYU_ASSERT(!lagrangian_invariant || !state.corner_mass_initialized,
                  "Hydro2D invariant corner_mass size changed after initialization");
    state.corner_mass.reset(expected_size);
    state.corner_mass_initialized = false;
    recompute = true;
  }
  if (lagrangian_invariant && !state.corner_mass_initialized) {
    recompute = true;
  }
  if (expected_size == 0) {
    return;
  }
  if (lagrangian_invariant && state.corner_mass_initialized && !force_recompute) {
    return;
  }
  if (!recompute) {
    return;
  }

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "Hydro2D multiblock corner mass requires cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "Hydro2D multiblock corner mass requires cell-node CSR indices");
    compute_corner_mass_2d_multiblock_kernel<<<cw.blocks(), 256>>>(
        state.corner_mass.data(), state.mass.data(), state.x_r.data(),
        state.x_z.data(), state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(), d_cell_nverts,
        rz::corner_mass_fallback_device_recorder(), cw.begin, cw.end,
        n_cells, lagrangian_invariant, polar_tier_equal_planar_area,
        state.corner_stride);
    sync_kernel("Hydro2D compute_corner_mass_multiblock kernel failed");
  } else {
    // A124(b) contract: SPEC §6.4 makes aw_compatible_force_work imply the
    // canonical subzonal partition (P1), so a structured corner-mass request
    // carrying the AW flag without it can only come from a direct-built
    // config that bypassed namelist validation — and would select a
    // partition no validated deck can reach. Fail loudly instead.
    TENRYU_ASSERT(!aw_compatible_force_work || canonical_subzonal_partition,
                  "Hydro2D structured corner mass: aw_compatible_force_work "
                  "requires the canonical subzonal partition (A124b)");
    // Option C: cache corner masses for the ghost i-planes too — the
    // interface node-mass scatter reads them, and for the Lagrangian-
    // invariant cache the ghost copies stay exactly equal to the
    // neighbor's frozen owned values (mass/x are halo-valid at cache
    // time). Owned-only caching left ghost corner masses at ZERO, which
    // silently re-halved the interface node mass after the scatter-window
    // fix. Full window at P==1 (fw == cw) keeps the serial path
    // byte-identical.
    const core::State::LaunchWindow fw =
        state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
    const int button_outer_node_ring = button_outer_node_ring_or_disabled(state);
    compute_corner_mass_2d_kernel<<<fw.blocks(), 256>>>(
        state.corner_mass.data(), state.mass.data(), state.x_r.data(), state.x_z.data(),
        d_cell_nverts, rz::corner_mass_fallback_device_recorder(),
        corner_mass_convention, canonical_subzonal_partition, fw.begin,
        fw.end, state.mesh.topo.nr, state.mesh.topo.nz,
        aw_compatible_force_work, button_outer_node_ring,
        state.corner_stride);
    sync_kernel("Hydro2D compute_corner_mass kernel failed");
  }
  state.corner_mass_initialized = true;
}

void launch_compute_node_activity_2d(
    std::uint8_t* const node_active,
    const std::int8_t* const d_hydro_active,
    const core::State& state) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                      static_cast<std::size_t>(n_nodes) + 1U,
                  "Hydro2D multiblock node activity requires reverse CSR offsets");
    compute_node_activity_2d_multiblock_kernel<<<nw.blocks(), 256>>>(
        node_active,
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        nw.begin, nw.end, n_nodes);
  } else {
    compute_node_activity_2d_kernel<<<cw.blocks(), 256>>>(
        node_active, d_hydro_active, cw.begin, cw.end, state.mesh.topo.nr,
        state.mesh.topo.nz);
  }
}

void launch_compute_node_mass_2d(double* const node_mass,
                                 const core::State& state,
                                 const core::Config& cfg,
                                 const std::uint8_t* const d_cell_nverts,
                                 const std::int8_t* const d_hydro_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  const int button_outer_node_ring = button_outer_node_ring_or_disabled(state);
  const std::int8_t* const node_mass_active =
      (button_outer_node_ring >= 1) ? d_hydro_active : nullptr;
  if (state.mesh.topo.multiblock.has_value()) {
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                      static_cast<std::size_t>(n_nodes) + 1U,
                  "Hydro2D multiblock node mass requires reverse CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                      state.mesh.multiblock_reverse_csr_node_corners.size(),
                  "Hydro2D multiblock node mass reverse CSR payload mismatch");
    const bool central_core_control_volume_fallback =
        state.mesh.topo.multiblock->block_count == 5;
    const int node_mass_mode = equal_split_node_mass_mode(
        cfg.numerics.hydro.axis_node_mass_convention);
    compute_node_mass_2d_multiblock_kernel<<<nw.blocks(), 256>>>(
        node_mass,
        state.corner_mass.data(),
        (central_core_control_volume_fallback || node_mass_mode != 0)
            ? state.mass.data()
            : nullptr,
        state.x_r.data(),
        state.x_z.data(),
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        state.mesh.multiblock_reverse_csr_node_corners.data(),
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts,
        node_mass_active,
        nw.begin,
        nw.end,
        n_nodes,
        central_core_control_volume_fallback,
        cfg.numerics.hydro.bbs_axis_policy_enabled, node_mass_mode,
        state.corner_stride);
  } else {
    // Cell->node atomicAdd scatter: the shared interface node plane needs
    // corner masses from BOTH adjacent cell planes, so the window must
    // include the ghost i-planes (Option C; corner_mass is the init-cached
    // Lagrangian invariant, globally valid on every rank). An owned-only
    // window halves the interface node mass => accel = F/m doubles
    // (root-caused via the parallel_fld_2d P=2 gate, v_z x2.02 at i=32).
    const core::State::LaunchWindow fw =
        state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
    compute_node_mass_2d_kernel<<<fw.blocks(), 256>>>(
        node_mass, state.corner_mass.data(), state.mass.data(), state.x_r.data(),
        state.x_z.data(), d_cell_nverts, node_mass_active, fw.begin,
        fw.end, state.mesh.topo.nr, state.mesh.topo.nz,
        button_outer_node_ring, state.corner_stride);
  }
}

void launch_compute_node_planar_mass_2d(
    core::State& state,
    const std::uint8_t* const d_cell_nverts,
    const std::int8_t* const d_hydro_active,
    const int blocks_cells) {
  state.node_planar_mass.reset(state.x_r.size());
  cuda_check(cudaMemset(state.node_planar_mass.data(), 0,
                        state.node_planar_mass.size() * sizeof(double)),
             "Hydro2D zero node planar mass failed");
  if (state.mesh.topo.multiblock.has_value()) {
    const int n_nodes = static_cast<int>(state.x_r.size());
    const int blocks_nodes = (n_nodes + 255) / 256;
    if (state.mesh.corner_stride <= mesh::kMeshTopoCellStorageSlotsMax) {
      compute_node_planar_mass_2d_multiblock_kernel<<<blocks_nodes, 256>>>(
          state.node_planar_mass.data(), state.rho.data(), state.x_r.data(),
          state.x_z.data(),
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          state.mesh.multiblock_reverse_csr_node_cells.data(),
          state.mesh.multiblock_reverse_csr_node_corners.data(),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(), d_cell_nverts,
          d_hydro_active, n_nodes);
    } else {
      compute_node_planar_mass_2d_multiblock_kernel<
          mesh::kMeshTopoCellStorageSlotsMaxGeneral><<<blocks_nodes, 256>>>(
          state.node_planar_mass.data(), state.rho.data(), state.x_r.data(),
          state.x_z.data(),
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          state.mesh.multiblock_reverse_csr_node_cells.data(),
          state.mesh.multiblock_reverse_csr_node_corners.data(),
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(), d_cell_nverts,
          d_hydro_active, n_nodes);
    }
  } else {
    compute_node_planar_mass_2d_kernel<<<blocks_cells, 256>>>(
        state.node_planar_mass.data(), state.rho.data(), state.x_r.data(),
        state.x_z.data(), d_hydro_active, state.mesh.topo.nr,
        state.mesh.topo.nz);
  }
}

void launch_compute_accel_from_force_2d(
    double* const accel_r,
    double* const accel_z,
    const double* const force_r,
    const double* const force_z,
    const core::NodeField1D& node_mass,
    const std::uint8_t* const d_node_active,
    const std::uint8_t* const d_node_flags,
    const int n_nodes,
    const core::State& state,
    const core::Config& cfg) {
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  switch (cfg.numerics.hydro.rz_momentum_scheme_id) {
    case 0:
      compute_accel_from_force_kernel<0><<<nw.blocks(), 256>>>(
          accel_r, accel_z, force_r, force_z, node_mass.data(), nullptr,
          d_node_active, d_node_flags, nw.begin, nw.end, n_nodes);
      break;
    case 1:
      TENRYU_ASSERT(state.node_planar_mass.size() ==
                        static_cast<std::size_t>(n_nodes),
                    "Hydro2D area-weighted acceleration requires node planar mass");
      compute_accel_from_force_kernel<1><<<nw.blocks(), 256>>>(
          accel_r, accel_z, force_r, force_z, node_mass.data(),
          state.node_planar_mass.data(), d_node_active, d_node_flags, nw.begin,
          nw.end, n_nodes);
      if (d_node_flags != nullptr) {
        aggregate_center_axial_accel_kernel<<<1, 1>>>(
            accel_r, accel_z, force_z, state.node_planar_mass.data(),
            d_node_flags, d_node_active, nw.begin, nw.end);
      }
      break;
    default:
      TENRYU_ASSERT(false, "Hydro2D invalid resolved RZ momentum scheme");
  }
}

void launch_apply_boundary_accel_constraints(
    double* const accel_r,
    double* const accel_z,
    const std::uint8_t* const d_node_flags,
    const core::State& state,
    const Boundary2DType r_inner_type,
    const Boundary2DType r_outer_type,
    const Boundary2DType z_bottom_type,
    const Boundary2DType z_top_type) {
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  if (state.mesh.topo.multiblock.has_value()) {
    apply_boundary_accel_constraints_multiblock_kernel<<<nw.blocks(), 256>>>(
        accel_r,
        accel_z,
        state.x_r.data(),
        state.x_z.data(),
        d_node_flags,
        nw.begin,
        nw.end,
        n_nodes,
        static_cast<int>(r_inner_type),
        static_cast<int>(r_outer_type),
        static_cast<int>(z_bottom_type),
        static_cast<int>(z_top_type));
  } else {
    apply_boundary_accel_constraints_kernel<<<nw.blocks(), 256>>>(
        accel_r,
        accel_z,
        d_node_flags,
        nw.begin,
        nw.end,
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        static_cast<int>(r_outer_type),
        static_cast<int>(z_bottom_type),
        static_cast<int>(z_top_type));
  }
}

void launch_apply_evac_contact_accel_constraints(
    core::State& state,
    double* const accel_r,
    double* const accel_z,
    const double* const node_mass_resolved) {
  const int n_pairs = state.evacuated_cells.n_active_pairs;
  if (n_pairs <= 0) {
    return;
  }
  apply_evac_contact_accel_constraints_kernel<<<1, 1>>>(
      accel_r,
      accel_z,
      node_mass_resolved,
      state.evacuated_cells.d_contact_pair_nodes.data(),
      state.evacuated_cells.d_contact_pair_normal.data(),
      state.evacuated_cells.d_contact_pair_lambda.data(),
      n_pairs);
}

void launch_apply_evac_contact_velocity_constraints(
    core::State& state,
    double* const v_r,
    double* const v_z,
    const double* const node_mass_resolved,
    const double* const x_r,
    const double* const x_z,
    const double* const cell_sound_speed,
    const std::int8_t* const hydro_active,
    const double dt,
    double* const pair_dk_accum,
    const double mortar_position_drift_beta) {
  const int n_pairs = state.evacuated_cells.n_active_pairs;
  const int n_rows = state.evacuated_cells.n_contact_rows;
  if (n_pairs <= 0 && n_rows <= 0) {
    return;
  }
  const int n_active_cells =
      state.evacuated_cells.n_contact_active_cells;
  TENRYU_ASSERT(
      (n_pairs <= 0 && n_active_cells == 0) ||
          (n_active_cells > 0 && cell_sound_speed != nullptr &&
           state.evacuated_cells.d_inactive_member_mask.size() ==
               static_cast<std::size_t>(state.mesh.topo.n_cells) &&
           state.evacuated_cells.d_contact_active_cells.size() ==
               static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_active_axis.size() ==
               static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_face_nodes.size() ==
               4U * static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_face_normal_ref.size() ==
               2U * static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_face_g0.size() ==
               static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_mortar_g_hold.size() ==
               core::kEvacContactMortarRowCapacity *
                   static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_mortar_g_hold_valid.size() ==
               core::kEvacContactMortarRowCapacity *
                   static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_mortar_drift_count.size() == 3U &&
           state.evacuated_cells.d_contact_mortar_drift_max_ucorr.size() == 1U &&
           state.evacuated_cells.d_contact_active_vol_at_engagement.size() ==
               static_cast<std::size_t>(n_active_cells) &&
           (state.evacuated_cells.d_contact_active_devolumized.size() == 0U ||
            state.evacuated_cells.d_contact_active_devolumized.size() ==
                static_cast<std::size_t>(n_active_cells)) &&
           state.evacuated_cells.d_contact_active_pair_dk_owner.size() ==
               static_cast<std::size_t>(n_active_cells) &&
           state.evacuated_cells.d_contact_volume_projection_count.size() ==
               2U),
      "Hydro2D evacuated-cell contact active device state mismatch");
  TENRYU_ASSERT(
      n_rows <= 0 ||
          (state.evacuated_cells.d_contact_row_nodes.size() ==
               3U * static_cast<std::size_t>(n_rows) &&
           state.evacuated_cells.d_contact_row_xi.size() ==
               static_cast<std::size_t>(n_rows) &&
           state.evacuated_cells.d_contact_row_normal.size() ==
               2U * static_cast<std::size_t>(n_rows) &&
           state.evacuated_cells.d_contact_row_dk.size() ==
               static_cast<std::size_t>(n_rows)),
      "Hydro2D evacuated-cell contact row device state mismatch");
  TENRYU_ASSERT(
      state.evacuated_cells.d_node_axis_alias.size() == 0U ||
          state.evacuated_cells.d_node_axis_alias.size() ==
              static_cast<std::size_t>(state.mesh.topo.n_nodes),
      "Hydro2D evacuated-cell node axis alias device state mismatch");
  apply_evac_contact_velocity_constraints_kernel<<<1, 1>>>(
      v_r,
      v_z,
      node_mass_resolved,
      x_r,
      x_z,
      cell_sound_speed,
      state.evacuated_cells.d_inactive_member_mask.data(),
      hydro_active,
      n_pairs > 0 ? state.evacuated_cells.d_contact_pair_nodes.data()
                  : nullptr,
      n_pairs > 0 ? state.evacuated_cells.d_contact_pair_normal.data()
                  : nullptr,
      state.evacuated_cells.d_contact_pair_row_covered.size() ==
              static_cast<std::size_t>(n_pairs)
          ? state.evacuated_cells.d_contact_pair_row_covered.data()
          : nullptr,
      n_pairs > 0 ? pair_dk_accum : nullptr,
      n_pairs,
      dt,
      state.evacuated_cells.d_contact_active_cells.data(),
      state.evacuated_cells.d_contact_active_axis.data(),
      state.evacuated_cells.d_contact_face_nodes.data(),
      state.evacuated_cells.d_contact_face_normal_ref.data(),
      state.evacuated_cells.d_contact_face_g0.data(),
      state.evacuated_cells.d_contact_active_vol_at_engagement.data(),
      state.evacuated_cells.d_contact_active_devolumized.size() != 0U
          ? state.evacuated_cells.d_contact_active_devolumized.data()
          : nullptr,
      state.evacuated_cells.d_contact_active_pair_dk_owner.data(),
      state.evacuated_cells.d_contact_mortar_g_hold.data(),
      state.evacuated_cells.d_contact_mortar_g_hold_valid.data(),
      mortar_position_drift_beta,
      state.evacuated_cells.d_contact_mortar_drift_count.data(),
      state.evacuated_cells.d_contact_mortar_drift_max_ucorr.data(),
      state.evacuated_cells.d_contact_volume_projection_count.data(),
      n_active_cells,
      state.mesh.topo.nr,
      state.mesh.topo.nz,
      state.evacuated_cells.d_node_axis_alias.size() != 0U
          ? state.evacuated_cells.d_node_axis_alias.data()
          : nullptr,
      n_rows > 0 ? state.evacuated_cells.d_contact_row_nodes.data()
                 : nullptr,
      n_rows > 0 ? state.evacuated_cells.d_contact_row_xi.data() : nullptr,
      n_rows > 0 ? state.evacuated_cells.d_contact_row_normal.data()
                 : nullptr,
      n_rows > 0 ? state.evacuated_cells.d_contact_row_dk.data() : nullptr,
      n_rows > 0 ? n_rows : 0);
  // The constraint kernel writes the resolved surviving node only, so re-synchronize fused pair velocities before subsequent position integration.
  enforce_node_axis_aliases(state);
}

}  // namespace

namespace {

void enforce_1t_closure(
    core::State& state,
    const core::Config& cfg,
    const HydroTableViews& eos_views = HydroTableViews{},
    const std::int8_t* hydro_active = nullptr) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro2D requires at least one material");
  const auto& mat = cfg.materials.materials.front();

  const int n_cells = static_cast<int>(state.rho.size());
  // Ghost-inclusive (OPEN-HYDRO-CORNER): ghost cells must carry the same
  // closure outputs (Pe/Pi/Te/Ti/cv) as their owner or interface corner
  // forces lose bitwise cancellation. Inputs (rho/ee/ei/zbar) are
  // exchange-fresh at ghosts. Serial identity: fw == [0, n_cells).
  const core::State::LaunchWindow fw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
  double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
  enforce_1t_closure_kernel<<<fw.blocks(), 256>>>(
      state.ee.data(), state.ei.data(), state.Te.data(), state.Ti.data(),
      state.Pe.data(), state.Pi.data(), state.rho.data(), state.zbar.data(),
      hydro_active, fw.begin, fw.end, n_cells,
      mat.ideal_gas_gamma, mat.A, mat.Z, cfg.numerics.floors.Te,
      eos_views.tab_total, eos_views.spline_total, eos_views.jet_total,
      use_helmholtz_spline_backend(eos_views), use_helmholtz_jet_backend(eos_views),
      cfg.numerics.hydro.eos_writeback, cv_e_ptr, cv_i_ptr);
  sync_kernel("Hydro2D enforce_1t_closure kernel failed");
}

void enforce_2t_closure(
    core::State& state,
    const core::Config& cfg,
    const HydroTableViews& eos_views = HydroTableViews{},
    const std::int8_t* hydro_active = nullptr) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro2D requires at least one material");
  const auto& mat = cfg.materials.materials.front();

  const int n_cells = static_cast<int>(state.rho.size());
  // Ghost-inclusive (OPEN-HYDRO-CORNER): see enforce_1t_closure.
  const core::State::LaunchWindow fw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
  double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
  enforce_2t_closure_kernel<<<fw.blocks(), 256>>>(
      state.ee.data(), state.ei.data(), state.Te.data(), state.Ti.data(),
      state.Pe.data(), state.Pi.data(), state.rho.data(), state.zbar.data(),
      hydro_active, fw.begin, fw.end, n_cells,
      mat.ideal_gas_gamma, mat.A, mat.Z, cfg.numerics.floors.Te, cfg.numerics.floors.Ti,
      eos_views.tab_ion, eos_views.tab_ele, eos_views.spline_total, eos_views.jet_total,
      use_helmholtz_spline_backend(eos_views), use_helmholtz_jet_backend(eos_views),
      cfg.numerics.hydro.eos_writeback, cv_e_ptr, cv_i_ptr);
  sync_kernel("Hydro2D enforce_2t_closure kernel failed");
}

void enforce_eos_closure(core::State& state,
                         const core::Config& cfg,
                         const bool use_two_temp,
                         const HydroTableViews& eos_views = HydroTableViews{},
                         const std::int8_t* hydro_active = nullptr) {
  if (use_two_temp) {
    enforce_2t_closure(state, cfg, eos_views, hydro_active);
  } else {
    enforce_1t_closure(state, cfg, eos_views, hydro_active);
  }
}

void compute_cell_sound_speed(core::CellField1D& cs,
                              const core::State& state,
                              const core::Config& cfg,
                              const bool use_two_temp,
                              const HydroTableViews& eos_views = HydroTableViews{},
                              const std::int8_t* hydro_active = nullptr) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "Hydro2D requires at least one material");
  const auto& mat = cfg.materials.materials.front();

  cs.reset(state.rho.size());
  const int n_cells = static_cast<int>(state.rho.size());
  // Ghost-inclusive (OPEN-HYDRO-CORNER fix): cs feeds the AV Q at ghost
  // cells; inputs (rho/Te/Ti) are exchange-fresh there. Serial identity.
  const core::State::LaunchWindow cw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  const double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
  const double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
  if (use_two_temp) {
    compute_sound_speed_2t_kernel<<<cw.blocks(), 256>>>(
        cs.data(), state.rho.data(), state.ee.data(), state.ei.data(), state.Pe.data(),
        state.Pi.data(), state.Te.data(), state.Ti.data(), eos_views.tab_ion,
        eos_views.tab_ele, eos_views.spline_total, eos_views.jet_total, cv_i_ptr, cv_e_ptr,
        hydro_active, cw.begin, cw.end, n_cells,
        use_helmholtz_spline_backend(eos_views),
        use_helmholtz_jet_backend(eos_views),
        mat.ideal_gas_gamma);
  } else {
    compute_sound_speed_1t_kernel<<<cw.blocks(), 256>>>(
        cs.data(), state.ee.data(), state.Te.data(), state.rho.data(), eos_views.tab_total,
        eos_views.spline_total, eos_views.jet_total, cv_e_ptr, hydro_active,
        cw.begin, cw.end, n_cells,
        use_helmholtz_spline_backend(eos_views), use_helmholtz_jet_backend(eos_views),
        mat.ideal_gas_gamma);
  }
  sync_kernel("Hydro2D compute_sound_speed kernel failed");
}

bool has_pressure_boundary(const core::Config& cfg) {
  const auto r_outer = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  const auto z_bottom = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom);
  const auto z_top = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top);
  return r_outer == Boundary2DType::PRESSURE ||
         z_bottom == Boundary2DType::PRESSURE ||
         z_top == Boundary2DType::PRESSURE;
}

double copy_device_double_at(const double* values,
                             const int index,
                             const char* message) {
  if (values == nullptr || index < 0) {
    return 0.0;
  }
  double out = 0.0;
  cuda_check(cudaMemcpy(&out,
                        values + index,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             message);
  return out;
}

int copy_device_int_at(const int* values,
                       const int index,
                       const char* message) {
  if (values == nullptr || index < 0) {
    return 0;
  }
  int out = 0;
  cuda_check(cudaMemcpy(&out,
                        values + index,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             message);
  return out;
}

bool trace_cell_nodes(const core::State& state,
                      const int c,
                      int nodes[4],
                      int* active_nverts_out = nullptr) {
  for (int k = 0; k < mesh::kMeshTopoCellStorageSlots; ++k) {
    nodes[k] = -1;
  }
  if (c < 0 || c >= static_cast<int>(state.rho.size())) {
    return false;
  }
  const int active_nverts =
      state.mesh.cell_nverts.size() == state.rho.size()
          ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
          : mesh::kMeshTopoCellStorageSlots;
  if (active_nverts_out != nullptr) {
    *active_nverts_out = active_nverts;
  }
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    if (mb.cell_node_csr_offsets.size() < static_cast<std::size_t>(c + 2)) {
      return false;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
    if (end - off < 4 ||
        mb.cell_node_csr_indices.size() < static_cast<std::size_t>(off + 4)) {
      return false;
    }
    for (int k = 0; k < active_nverts; ++k) {
      nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
    }
    return true;
  }

  const int nz = state.mesh.topo.nz;
  if (nz <= 0) {
    return false;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const int stride = nz + 1;
  nodes[0] = i * stride + j;
  nodes[1] = (i + 1) * stride + j;
  nodes[2] = (i + 1) * stride + (j + 1);
  if (active_nverts == mesh::kMeshTopoCellStorageSlots) {
    nodes[3] = i * stride + (j + 1);
  }
  return true;
}

struct AxisWallCornerRef {
  int cell = -1;
  int index = -1;
};

struct AxisWallEdgeRef {
  int edge = -1;
  int cell_a = -1;
  int cell_b = -1;
  double sign = 0.0;
};

struct AxisWallNodeScatter {
  int node = -1;
  std::vector<AxisWallCornerRef> corners;
  std::vector<AxisWallEdgeRef> edges;
};

struct AxisWallWatch {
  int cell = -1;
  int active_nverts = 0;
  std::array<AxisWallNodeScatter, 4> nodes;
  bool topology_ready = false;
  bool initial_area_set = false;
  double initial_area = 0.0;
};

const std::vector<int>& axiswall_diag_cell_ids() {
  static const std::vector<int> cells = [] {
    std::vector<int> parsed;
    const char* const raw = std::getenv("TENRYU_AXISWALL_DIAG");
    if (raw == nullptr || raw[0] == '\0') {
      return parsed;
    }
    const char* token = raw;
    while (*token != '\0') {
      const char* token_end = token;
      while (*token_end != '\0' && *token_end != ',') {
        ++token_end;
      }
      char* parsed_end = nullptr;
      const long value = std::strtol(token, &parsed_end, 10);
      const char* trailing = parsed_end;
      while (trailing < token_end &&
             std::isspace(static_cast<unsigned char>(*trailing)) != 0) {
        ++trailing;
      }
      if (parsed_end != token && trailing == token_end && value >= 0 &&
          value <= std::numeric_limits<int>::max()) {
        const int cell = static_cast<int>(value);
        if (std::find(parsed.begin(), parsed.end(), cell) == parsed.end()) {
          parsed.push_back(cell);
        }
      }
      token = (*token_end == ',') ? token_end + 1 : token_end;
    }
    return parsed;
  }();
  return cells;
}

int axiswall_diag_every() {
  static const int every = [] {
    const char* const raw = std::getenv("TENRYU_AXISWALL_DIAG_EVERY");
    if (raw == nullptr || raw[0] == '\0') {
      return 100;
    }
    char* end = nullptr;
    const long value = std::strtol(raw, &end, 10);
    return end != raw && *end == '\0' && value > 0 &&
                   value <= std::numeric_limits<int>::max()
               ? static_cast<int>(value)
               : 100;
  }();
  return every;
}

std::vector<AxisWallWatch>& axiswall_diag_watches() {
  static std::vector<AxisWallWatch> watches = [] {
    std::vector<AxisWallWatch> out;
    for (const int cell : axiswall_diag_cell_ids()) {
      AxisWallWatch watch;
      watch.cell = cell;
      out.push_back(std::move(watch));
    }
    return out;
  }();
  return watches;
}

bool axiswall_cell_active(const std::vector<std::int8_t>& hydro_active,
                          const int cell) {
  return cell >= 0 &&
         (hydro_active.empty() ||
          (static_cast<std::size_t>(cell) < hydro_active.size() &&
           hydro_active[static_cast<std::size_t>(cell)] != 0));
}

bool axiswall_multiblock_face_nodes(const core::State& state,
                                    const mesh::UniqueOrientedFace& face,
                                    int* node0,
                                    int* node1) {
  if (!state.mesh.topo.multiblock.has_value() || face.cell_a < 0 ||
      face.cell_a >= static_cast<int>(state.rho.size())) {
    return false;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int active_nverts =
      mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, face.cell_a);
  int corner0 = -1;
  int corner1 = -1;
  if (active_nverts == 4) {
    if (face.local_a == 0) {
      corner0 = 0;
      corner1 = 3;
    } else if (face.local_a == 1) {
      corner0 = 1;
      corner1 = 2;
    } else if (face.local_a == 2) {
      corner0 = 0;
      corner1 = 1;
    } else if (face.local_a == 3) {
      corner0 = 3;
      corner1 = 2;
    } else {
      return false;
    }
  } else if (!mesh::mesh_topo_active_local_face_corners(
                 active_nverts, face.local_a, &corner0, &corner1)) {
    return false;
  }
  if (mb.cell_node_csr_offsets.size() <
      static_cast<std::size_t>(face.cell_a + 2)) {
    return false;
  }
  const int off =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(face.cell_a)];
  if (off < 0 ||
      static_cast<std::size_t>(off + std::max(corner0, corner1)) >=
          mb.cell_node_csr_indices.size()) {
    return false;
  }
  *node0 =
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner0)];
  *node1 =
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner1)];
  return true;
}

void axiswall_add_edge_ref(AxisWallWatch& watch,
                           const int edge,
                           const int node0,
                           const int node1,
                           const int cell_a,
                           const int cell_b) {
  for (int k = 0; k < watch.active_nverts; ++k) {
    auto& node = watch.nodes[static_cast<std::size_t>(k)];
    if (node.node == node0) {
      node.edges.push_back(AxisWallEdgeRef{edge, cell_a, cell_b, -1.0});
    } else if (node.node == node1) {
      node.edges.push_back(AxisWallEdgeRef{edge, cell_a, cell_b, 1.0});
    }
  }
}

void initialize_axiswall_watch_topology(AxisWallWatch& watch,
                                        const core::State& state) {
  int nodes[4] = {-1, -1, -1, -1};
  int active_nverts = 0;
  TENRYU_ASSERT(trace_cell_nodes(state, watch.cell, nodes, &active_nverts),
                "TENRYU_AXISWALL_DIAG watched cell is out of range");
  TENRYU_ASSERT(active_nverts == 3 || active_nverts == 4,
                "TENRYU_AXISWALL_DIAG supports triangle/quad watched cells");
  watch.active_nverts = active_nverts;
  for (int k = 0; k < active_nverts; ++k) {
    watch.nodes[static_cast<std::size_t>(k)].node = nodes[k];
  }

  const int n_cells = static_cast<int>(state.rho.size());
  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "TENRYU_AXISWALL_DIAG requires multiblock cell-node CSR");
    for (int cell = 0; cell < n_cells; ++cell) {
      const int off =
          mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
      const int nverts =
          mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
      TENRYU_ASSERT(off >= 0 &&
                        static_cast<std::size_t>(off + nverts) <=
                            mb.cell_node_csr_indices.size(),
                    "TENRYU_AXISWALL_DIAG invalid multiblock cell-node CSR");
      for (int corner = 0; corner < nverts; ++corner) {
        const int node =
            mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner)];
        for (int k = 0; k < active_nverts; ++k) {
          if (watch.nodes[static_cast<std::size_t>(k)].node == node) {
            watch.nodes[static_cast<std::size_t>(k)].corners.push_back(
                AxisWallCornerRef{
                    cell, cell * state.corner_stride + corner});
          }
        }
      }
    }
    int edge = 0;
    for (const auto& face : mb.unique_internal_faces) {
      int node0 = -1;
      int node1 = -1;
      TENRYU_ASSERT(
          axiswall_multiblock_face_nodes(state, face, &node0, &node1),
          "TENRYU_AXISWALL_DIAG invalid multiblock internal face");
      axiswall_add_edge_ref(
          watch, edge, node0, node1, face.cell_a, face.cell_b);
      ++edge;
    }
    for (const auto& face : mb.boundary_faces) {
      int node0 = -1;
      int node1 = -1;
      TENRYU_ASSERT(
          axiswall_multiblock_face_nodes(state, face, &node0, &node1),
          "TENRYU_AXISWALL_DIAG invalid multiblock boundary face");
      axiswall_add_edge_ref(watch, edge, node0, node1, face.cell_a, -1);
      ++edge;
    }
  } else {
    const int nr = state.mesh.topo.nr;
    const int nz = state.mesh.topo.nz;
    const int stride = nz + 1;
    for (int cell = 0; cell < n_cells; ++cell) {
      const int i = cell / nz;
      const int j = cell - i * nz;
      const int cell_nodes[4] = {
          i * stride + j,
          (i + 1) * stride + j,
          (i + 1) * stride + (j + 1),
          i * stride + (j + 1),
      };
      for (int corner = 0; corner < 4; ++corner) {
        for (int k = 0; k < active_nverts; ++k) {
          if (watch.nodes[static_cast<std::size_t>(k)].node ==
              cell_nodes[corner]) {
            watch.nodes[static_cast<std::size_t>(k)].corners.push_back(
                AxisWallCornerRef{
                    cell, cell * state.corner_stride + corner});
          }
        }
      }
    }
    const int n_radial = nr * (nz + 1);
    const int n_edges = n_radial + (nr + 1) * nz;
    for (int edge = 0; edge < n_edges; ++edge) {
      int node0 = -1;
      int node1 = -1;
      int cell_a = -1;
      int cell_b = -1;
      if (edge < n_radial) {
        const int i = edge / (nz + 1);
        const int j = edge - i * (nz + 1);
        node0 = i * stride + j;
        node1 = (i + 1) * stride + j;
        if (j > 0) {
          cell_a = i * nz + (j - 1);
        }
        if (j < nz) {
          cell_b = i * nz + j;
        }
      } else {
        const int angular = edge - n_radial;
        const int i = angular / nz;
        const int j = angular - i * nz;
        node0 = i * stride + j;
        node1 = i * stride + (j + 1);
        if (i > 0) {
          cell_a = (i - 1) * nz + j;
        }
        if (i < nr) {
          cell_b = i * nz + j;
        }
      }
      axiswall_add_edge_ref(
          watch, edge, node0, node1, cell_a, cell_b);
    }
  }
  watch.topology_ready = true;
}

std::uint8_t axiswall_copy_node_active(const std::uint8_t* values,
                                       const int node) {
  std::uint8_t out = 0U;
  cuda_check(cudaMemcpy(&out,
                        values + node,
                        sizeof(std::uint8_t),
                        cudaMemcpyDeviceToHost),
             "Hydro2D axiswall copy node_active failed");
  return out;
}

int axiswall_velocity_bc_mode(const Boundary2DType type) {
  if (type == Boundary2DType::FIXED) {
    return pole_axis::kVelocityBcFixed;
  }
  if (type == Boundary2DType::REFLECT) {
    return pole_axis::kVelocityBcReflect;
  }
  if (type == Boundary2DType::STATE_SUPPLY) {
    return pole_axis::kVelocityBcStateSupply;
  }
  return pole_axis::kVelocityBcFree;
}

void axiswall_remove_spherical_normal(double* vector_r,
                                      double* vector_z,
                                      const double r,
                                      const double z) {
  const double s = std::sqrt(r * r + z * z);
  if (s > 0.0) {
    const double inv_s = 1.0 / s;
    const double normal_r = r * inv_s;
    const double normal_z = z * inv_s;
    const double normal_component =
        *vector_r * normal_r + *vector_z * normal_z;
    *vector_r -= normal_component * normal_r;
    *vector_z -= normal_component * normal_z;
  }
}

void axiswall_apply_multiblock_constraint(const core::State& state,
                                          const core::Config& cfg,
                                          const int node,
                                          const double r,
                                          const double z,
                                          double* accel_r,
                                          double* accel_z) {
  if (state.mesh.topo.node_flags.size() <= static_cast<std::size_t>(node)) {
    return;
  }
  const std::uint8_t flags =
      state.mesh.topo.node_flags[static_cast<std::size_t>(node)];
  if ((flags & kNodeCenterFlag) != 0U) {
    *accel_r = 0.0;
    *accel_z = 0.0;
    return;
  }
  if ((flags & (mesh::NODE_AXIS | kNodePoleAxisFlag)) != 0U) {
    *accel_r = 0.0;
  }
  if ((flags & mesh::NODE_INNER_PHYSICAL_BOUNDARY) != 0U) {
    const Boundary2DType inner =
        parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_inner);
    if (inner == Boundary2DType::PINNED) {
      *accel_r = 0.0;
      *accel_z = 0.0;
      return;
    }
    if (inner == Boundary2DType::AXIS) {
      axiswall_remove_spherical_normal(accel_r, accel_z, r, z);
      return;
    }
  }
  if ((flags & mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U) {
    const Boundary2DType outer =
        parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
    if (outer == Boundary2DType::FIXED) {
      *accel_r = 0.0;
      *accel_z = 0.0;
    } else if (outer == Boundary2DType::REFLECT ||
               outer == Boundary2DType::STATE_SUPPLY ||
               (outer != Boundary2DType::FREE &&
                outer != Boundary2DType::PRESSURE)) {
      axiswall_remove_spherical_normal(accel_r, accel_z, r, z);
    }
  }
}

void axiswall_force_to_accel(const core::State& state,
                             const core::Config& cfg,
                             const int node,
                             const bool node_active,
                             const double node_mass,
                             const double r,
                             const double z,
                             const double force_r,
                             const double force_z,
                             double* accel_r,
                             double* accel_z) {
  const std::uint8_t flags =
      state.mesh.topo.node_flags.size() > static_cast<std::size_t>(node)
          ? state.mesh.topo.node_flags[static_cast<std::size_t>(node)]
          : 0U;
  if (!node_active || !(node_mass > 0.0) ||
      (flags & kNodeCenterFlag) != 0U) {
    *accel_r = 0.0;
    *accel_z = 0.0;
    return;
  }
  *accel_r = force_r / node_mass;
  *accel_z = force_z / node_mass;
  if (state.mesh.topo.multiblock.has_value()) {
    axiswall_apply_multiblock_constraint(
        state, cfg, node, r, z, accel_r, accel_z);
    return;
  }
  const int nz = state.mesh.topo.nz;
  const int i = node / (nz + 1);
  const int j = node - i * (nz + 1);
  const std::uint8_t* node_flags =
      state.mesh.topo.node_flags.empty()
          ? nullptr
          : state.mesh.topo.node_flags.data();
  pole_axis::apply_2d_boundary_vector_constraints(
      *accel_r,
      *accel_z,
      node_flags,
      node,
      i,
      j,
      state.mesh.topo.nr,
      nz,
      axiswall_velocity_bc_mode(
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer)),
      axiswall_velocity_bc_mode(
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom)),
      axiswall_velocity_bc_mode(
          parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top)),
      true);
}

double axiswall_signed_area(const std::array<double, 4>& r,
                            const std::array<double, 4>& z,
                            const int nverts) {
  double twice_area = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int next = (k + 1) % nverts;
    twice_area += r[static_cast<std::size_t>(k)] *
                      z[static_cast<std::size_t>(next)] -
                  r[static_cast<std::size_t>(next)] *
                      z[static_cast<std::size_t>(k)];
  }
  return 0.5 * twice_area;
}

double axiswall_area_rate(const std::array<double, 4>& r,
                          const std::array<double, 4>& z,
                          const std::array<double, 4>& velocity_r,
                          const std::array<double, 4>& velocity_z,
                          const int nverts) {
  double twice_rate = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int prev = (k + nverts - 1) % nverts;
    const int next = (k + 1) % nverts;
    twice_rate +=
        velocity_r[static_cast<std::size_t>(k)] *
            (z[static_cast<std::size_t>(next)] -
             z[static_cast<std::size_t>(prev)]) +
        velocity_z[static_cast<std::size_t>(k)] *
            (r[static_cast<std::size_t>(prev)] -
             r[static_cast<std::size_t>(next)]);
  }
  return 0.5 * twice_rate;
}

void emit_axiswall_diag(const core::State& state,
                        const core::Config& cfg,
                        const double eval_time,
                        const double dt,
                        const core::NodeField1D& r,
                        const core::NodeField1D& z,
                        const core::NodeField1D& velocity_r,
                        const core::NodeField1D& velocity_z,
                        const core::NodeField1D& node_mass,
                        const std::uint8_t* d_node_active,
                        const std::vector<std::int8_t>& hydro_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  TENRYU_ASSERT(d_node_active != nullptr,
                "TENRYU_AXISWALL_DIAG requires node activity");
  TENRYU_ASSERT(state.corner_force_p_r.size() == n_corners &&
                    state.corner_force_p_z.size() == n_corners &&
                    state.corner_force_sub_r.size() == n_corners &&
                    state.corner_force_sub_z.size() == n_corners,
                "TENRYU_AXISWALL_DIAG requires compatible corner-force buffers");
  TENRYU_ASSERT(
      cfg.numerics.hydro.av_model == core::AvModel::CswEdge ||
          cfg.numerics.hydro.av_model == core::AvModel::CswEdgeCsw98,
      "TENRYU_AXISWALL_DIAG requires CSW edge artificial viscosity");
  TENRYU_ASSERT(state.edge_force_av_r.size() ==
                        state.edge_force_av_z.size() &&
                    node_mass.size() == static_cast<std::size_t>(n_nodes),
                "TENRYU_AXISWALL_DIAG force/mass buffer size mismatch");

  bool emitted = false;
  for (auto& watch : axiswall_diag_watches()) {
    if (!watch.topology_ready) {
      initialize_axiswall_watch_topology(watch, state);
    }
    std::array<double, 4> node_r{};
    std::array<double, 4> node_z{};
    std::array<double, 4> node_velocity_r{};
    std::array<double, 4> node_velocity_z{};
    std::array<double, 4> accel_p_r{};
    std::array<double, 4> accel_p_z{};
    std::array<double, 4> accel_q_r{};
    std::array<double, 4> accel_q_z{};
    std::array<double, 4> accel_sz_r{};
    std::array<double, 4> accel_sz_z{};

    for (int k = 0; k < watch.active_nverts; ++k) {
      const AxisWallNodeScatter& scatter =
          watch.nodes[static_cast<std::size_t>(k)];
      const int node = scatter.node;
      TENRYU_ASSERT(node >= 0 && node < n_nodes,
                    "TENRYU_AXISWALL_DIAG watched node is out of range");
      node_r[static_cast<std::size_t>(k)] = copy_device_double_at(
          r.data(), node, "Hydro2D axiswall copy node r failed");
      node_z[static_cast<std::size_t>(k)] = copy_device_double_at(
          z.data(), node, "Hydro2D axiswall copy node z failed");
      node_velocity_r[static_cast<std::size_t>(k)] =
          copy_device_double_at(
              velocity_r.data(),
              node,
              "Hydro2D axiswall copy node velocity_r failed");
      node_velocity_z[static_cast<std::size_t>(k)] =
          copy_device_double_at(
              velocity_z.data(),
              node,
              "Hydro2D axiswall copy node velocity_z failed");
      const double mass = copy_device_double_at(
          node_mass.data(), node, "Hydro2D axiswall copy node mass failed");
      const bool active =
          axiswall_copy_node_active(d_node_active, node) != 0U;

      double force_p_r = 0.0;
      double force_p_z = 0.0;
      double force_sz_r = 0.0;
      double force_sz_z = 0.0;
      for (const AxisWallCornerRef& corner : scatter.corners) {
        if (!axiswall_cell_active(hydro_active, corner.cell)) {
          continue;
        }
        force_p_r += copy_device_double_at(
            state.corner_force_p_r.data(),
            corner.index,
            "Hydro2D axiswall copy pressure corner force_r failed");
        force_p_z += copy_device_double_at(
            state.corner_force_p_z.data(),
            corner.index,
            "Hydro2D axiswall copy pressure corner force_z failed");
        force_sz_r += copy_device_double_at(
            state.corner_force_sub_r.data(),
            corner.index,
            "Hydro2D axiswall copy subzonal corner force_r failed");
        force_sz_z += copy_device_double_at(
            state.corner_force_sub_z.data(),
            corner.index,
            "Hydro2D axiswall copy subzonal corner force_z failed");
      }

      double force_q_r = 0.0;
      double force_q_z = 0.0;
      for (const AxisWallEdgeRef& edge : scatter.edges) {
        if (!axiswall_cell_active(hydro_active, edge.cell_a) &&
            !axiswall_cell_active(hydro_active, edge.cell_b)) {
          continue;
        }
        force_q_r +=
            edge.sign *
            copy_device_double_at(
                state.edge_force_av_r.data(),
                edge.edge,
                "Hydro2D axiswall copy edge AV force_r failed");
        force_q_z +=
            edge.sign *
            copy_device_double_at(
                state.edge_force_av_z.data(),
                edge.edge,
                "Hydro2D axiswall copy edge AV force_z failed");
      }

      axiswall_force_to_accel(
          state,
          cfg,
          node,
          active,
          mass,
          node_r[static_cast<std::size_t>(k)],
          node_z[static_cast<std::size_t>(k)],
          force_p_r,
          force_p_z,
          &accel_p_r[static_cast<std::size_t>(k)],
          &accel_p_z[static_cast<std::size_t>(k)]);
      axiswall_force_to_accel(
          state,
          cfg,
          node,
          active,
          mass,
          node_r[static_cast<std::size_t>(k)],
          node_z[static_cast<std::size_t>(k)],
          force_q_r,
          force_q_z,
          &accel_q_r[static_cast<std::size_t>(k)],
          &accel_q_z[static_cast<std::size_t>(k)]);
      axiswall_force_to_accel(
          state,
          cfg,
          node,
          active,
          mass,
          node_r[static_cast<std::size_t>(k)],
          node_z[static_cast<std::size_t>(k)],
          force_sz_r,
          force_sz_z,
          &accel_sz_r[static_cast<std::size_t>(k)],
          &accel_sz_z[static_cast<std::size_t>(k)]);
    }

    const double area =
        axiswall_signed_area(node_r, node_z, watch.active_nverts);
    if (!watch.initial_area_set) {
      watch.initial_area = area;
      watch.initial_area_set = true;
    }
    const double dA_u =
        dt * axiswall_area_rate(
                 node_r,
                 node_z,
                 node_velocity_r,
                 node_velocity_z,
                 watch.active_nverts);
    const double half_dt_squared = 0.5 * dt * dt;
    const double dA_p =
        half_dt_squared *
        axiswall_area_rate(
            node_r, node_z, accel_p_r, accel_p_z, watch.active_nverts);
    const double dA_q =
        half_dt_squared *
        axiswall_area_rate(
            node_r, node_z, accel_q_r, accel_q_z, watch.active_nverts);
    const double dA_sz =
        half_dt_squared *
        axiswall_area_rate(
            node_r, node_z, accel_sz_r, accel_sz_z, watch.active_nverts);
    const double area_ratio =
        watch.initial_area != 0.0
            ? area / watch.initial_area
            : std::numeric_limits<double>::infinity();
    if (area_ratio < 0.5 || state.step % axiswall_diag_every() == 0) {
      std::fprintf(stderr,
                   "[axiswall] step=%d t=%.6e cell=%d A=%.6e A0=%.6e "
                   "dA_u=%.3e dA_p=%.3e dA_q=%.3e dA_sz=%.3e\n",
                   state.step,
                   eval_time,
                   watch.cell,
                   area,
                   watch.initial_area,
                   dA_u,
                   dA_p,
                   dA_q,
                   dA_sz);
      emitted = true;
    }
  }
  if (emitted) {
    std::fflush(stderr);
  }
}

bool pole_axis_planar_diag_enabled() {
  const char* env = std::getenv("TENRYU_I1B_POLE_AXIS_DIAG");
  return env != nullptr && env[0] != '\0' && std::string(env) != "0";
}

bool pole_axis_bbsw_enabled(const core::Config& cfg) {
  static const char* const env = std::getenv(pole_axis_bbsw::kEnvEnable);
  static const bool env_present = env != nullptr && env[0] != '\0';
  static const bool env_enabled = env_present && std::string(env) != "0";
  const bool enabled =
      env_present ? env_enabled : cfg.numerics.ale.pole_axis_bbsw_enabled;
  if (enabled && cfg.mesh.shell_polar_cap_dendrite) {
    throw core::namelist::ConfigError(
        "pole-axis BBSW is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  return enabled;
}

std::vector<int> pole_axis_bbsw_hybrid_cart_axis_nodes(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  if (!mesh::mesh_topo_has_polar_tier_cart_center(cfg.mesh)) {
    return {};
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "hybrid pole-axis BBSW requires multiblock topology");
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(node_r.size() == static_cast<std::size_t>(n_nodes) &&
                    node_z.size() == static_cast<std::size_t>(n_nodes),
                "hybrid pole-axis BBSW requires node-sized coordinates");
  const int n_aw_axis_prefix_nodes = mesh::mesh_topo_n_aw_axis_prefix_nodes(
      *state.mesh.topo.multiblock);
  TENRYU_ASSERT(n_aw_axis_prefix_nodes <= n_nodes,
                "hybrid pole-axis BBSW AW-axis prefix exceeds node count");
  std::vector<int> nodes;
  for (int node = n_aw_axis_prefix_nodes; node < n_nodes; ++node) {
    if (node_r[static_cast<std::size_t>(node)] == 0.0) {
      nodes.push_back(node);
    }
  }
  std::sort(nodes.begin(), nodes.end(), [&](const int a, const int b) {
    const double za = node_z[static_cast<std::size_t>(a)];
    const double zb = node_z[static_cast<std::size_t>(b)];
    return za == zb ? a < b : za < zb;
  });
  return nodes;
}

int pole_axis_planar_diag_every() {
  const char* env = std::getenv("TENRYU_I1B_POLE_AXIS_DIAG_EVERY");
  if (env == nullptr || env[0] == '\0') {
    return 20;
  }
  return std::max(1, std::atoi(env));
}

bool pole_axis_planar_diag_due(const core::State& state,
                               const core::Config& cfg) {
  const int every = pole_axis_planar_diag_every();
  const bool enabled = pole_axis_planar_diag_enabled();
  if (enabled && cfg.mesh.shell_polar_cap_dendrite) {
    throw core::namelist::ConfigError(
        "pole-axis planar diagnostics are not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  return enabled && every > 0 && state.step % every == 0 &&
         state.mesh.topo.multiblock.has_value();
}

double planar_diag_at_or_nan(const std::vector<double>& values, const int n) {
  return n >= 0 && static_cast<std::size_t>(n) < values.size()
             ? values[static_cast<std::size_t>(n)]
             : std::numeric_limits<double>::quiet_NaN();
}

int planar_diag_u8_or_minus_one(const std::vector<std::uint8_t>& values,
                                const int n) {
  return n >= 0 && static_cast<std::size_t>(n) < values.size()
             ? static_cast<int>(values[static_cast<std::size_t>(n)])
             : -1;
}

double planar_diag_triangle_area2(const double r0,
                                  const double z0,
                                  const double r1,
                                  const double z1,
                                  const double r2,
                                  const double z2) {
  return r0 * z1 - r1 * z0 + r1 * z2 - r2 * z1 + r2 * z0 - r0 * z2;
}

void planar_diag_corner_vectors(const double* r,
                                const double* z,
                                const int nverts,
                                const double orientation_sign,
                                double* cvec_r,
                                double* cvec_z) {
  for (int k = 0; k < nverts; ++k) {
    const int km1 = (k + nverts - 1) % nverts;
    const int kp1 = (k + 1) % nverts;
    cvec_r[k] = 0.5 * orientation_sign * (z[kp1] - z[km1]);
    cvec_z[k] = 0.5 * orientation_sign * (r[km1] - r[kp1]);
  }
}

void planar_diag_corner_areas(const double* r,
                              const double* z,
                              const int nverts,
                              const double orientation_sign,
                              double* area) {
  if (nverts <= 0) {
    return;
  }

  double rc = 0.0;
  double zc = 0.0;
  for (int k = 0; k < nverts; ++k) {
    rc += r[k];
    zc += z[k];
  }
  rc /= static_cast<double>(nverts);
  zc /= static_cast<double>(nverts);

  if (nverts == 4) {
    double edge_area[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
      const int kp1 = (k + 1) & 3;
      const double a2 = planar_diag_triangle_area2(
          r[k], z[k], r[kp1], z[kp1], rc, zc);
      edge_area[k] = std::fabs(0.5 * orientation_sign * a2);
    }
    area[0] = (5.0 * edge_area[3] + 5.0 * edge_area[0] +
               edge_area[1] + edge_area[2]) /
              12.0;
    area[1] = (edge_area[3] + 5.0 * edge_area[0] +
               5.0 * edge_area[1] + edge_area[2]) /
              12.0;
    area[2] = (edge_area[3] + edge_area[0] +
               5.0 * edge_area[1] + 5.0 * edge_area[2]) /
              12.0;
    area[3] = (5.0 * edge_area[3] + edge_area[0] +
               edge_area[1] + 5.0 * edge_area[2]) /
              12.0;
    return;
  }

  double area2 = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    area2 += r[k] * z[kp1] - r[kp1] * z[k];
  }
  const double total_area = std::fabs(0.5 * orientation_sign * area2);
  const double uniform = total_area / static_cast<double>(nverts);
  for (int k = 0; k < nverts; ++k) {
    area[k] = uniform;
  }
}

double planar_diag_cell_orientation_sign(const core::State& state,
                                         const int c) {
  if (!state.mesh.topo.multiblock.has_value()) {
    return 1.0;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (c >= 0 && static_cast<std::size_t>(c) < mb.cell_orientation_sign.size() &&
      mb.cell_orientation_sign[static_cast<std::size_t>(c)] < 0) {
    return -1.0;
  }
  return 1.0;
}

double planar_diag_cell_density(const std::vector<double>& rho,
                                const std::vector<double>& mass,
                                const std::vector<double>& vol,
                                const int c) {
  double density = planar_diag_at_or_nan(rho, c);
  if (std::isfinite(density) && density > 0.0) {
    return density;
  }
  const double m = planar_diag_at_or_nan(mass, c);
  const double v = planar_diag_at_or_nan(vol, c);
  if (std::isfinite(m) && m >= 0.0 && std::isfinite(v) && v > 0.0) {
    density = m / v;
  }
  return (std::isfinite(density) && density > 0.0) ? density : 0.0;
}

double planar_diag_corner_density(const std::vector<double>& rho,
                                  const std::vector<double>& mass,
                                  const std::vector<double>& vol,
                                  const std::vector<double>& corner_mass,
                                  const int c,
                                  const int corner,
                                  const int active_nverts,
                                  const int corner_stride,
                                  const double* r,
                                  const double* z) {
  double density = planar_diag_cell_density(rho, mass, vol, c);
  if (active_nverts == 4 &&
      corner_mass.size() >=
          (static_cast<std::size_t>(c) + 1U) *
              static_cast<std::size_t>(corner_stride)) {
    double v_corner[4] = {0.0, 0.0, 0.0, 0.0};
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], v_corner);
    const double v = v_corner[corner];
    const double m = corner_mass[static_cast<std::size_t>(c) *
                                     static_cast<std::size_t>(corner_stride) +
                                 static_cast<std::size_t>(corner)];
    if (std::isfinite(m) && m >= 0.0 && std::isfinite(v) && v > 0.0) {
      density = m / v;
    }
  }
  return (std::isfinite(density) && density > 0.0) ? density : 0.0;
}

double planar_diag_macro_pressure(
    const core::PoleAngularDerefineMacroState& macro,
    const std::vector<double>& mass,
    const std::vector<double>& pressure) {
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (const int c : macro.member_cells) {
    const double m = planar_diag_at_or_nan(mass, c);
    const double p = planar_diag_at_or_nan(pressure, c);
    if (std::isfinite(m) && m > 0.0 && std::isfinite(p)) {
      numerator += static_cast<long double>(m) * static_cast<long double>(p);
      denominator += static_cast<long double>(m);
    }
  }
  if (!(denominator > 0.0L)) {
    return 0.0;
  }
  const double p = static_cast<double>(numerator / denominator);
  return (std::isfinite(p) && p > 0.0) ? p : 0.0;
}

double planar_diag_macro_density(
    const core::PoleAngularDerefineMacroState& macro,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    const std::vector<double>& mass,
    const std::vector<double>& rho,
    const std::vector<double>& vol) {
  const int n_boundary = static_cast<int>(macro.boundary_nodes_ordered.size());
  if (n_boundary < 3) {
    return 0.0;
  }
  std::vector<double> r(static_cast<std::size_t>(n_boundary));
  std::vector<double> z(static_cast<std::size_t>(n_boundary));
  for (int k = 0; k < n_boundary; ++k) {
    const int n = macro.boundary_nodes_ordered[static_cast<std::size_t>(k)];
    r[static_cast<std::size_t>(k)] = planar_diag_at_or_nan(node_r, n);
    z[static_cast<std::size_t>(k)] = planar_diag_at_or_nan(node_z, n);
  }

  long double m_sum = 0.0L;
  long double rho_weighted = 0.0L;
  long double rho_weight = 0.0L;
  for (const int c : macro.member_cells) {
    const double m = planar_diag_at_or_nan(mass, c);
    const double rho_c = planar_diag_cell_density(rho, mass, vol, c);
    if (std::isfinite(m) && m > 0.0) {
      m_sum += static_cast<long double>(m);
      if (std::isfinite(rho_c) && rho_c > 0.0) {
        rho_weighted += static_cast<long double>(m) *
                        static_cast<long double>(rho_c);
        rho_weight += static_cast<long double>(m);
      }
    }
  }

  const double v = std::fabs(rz::rz_polygon_volume_exact(
      r.data(), z.data(), n_boundary));
  if (std::isfinite(v) && v > 0.0 && m_sum >= 0.0L) {
    const double density = static_cast<double>(m_sum / static_cast<long double>(v));
    if (std::isfinite(density) && density > 0.0) {
      return density;
    }
  }
  if (rho_weight > 0.0L) {
    const double density = static_cast<double>(rho_weighted / rho_weight);
    if (std::isfinite(density) && density > 0.0) {
      return density;
    }
  }
  return 0.0;
}

bool planar_diag_pole_deref_macro_contains_node(const core::State& state,
                                                const int q,
                                                const int j) {
  const auto& pc = state.pole_angular_derefine;
  if (!pc.configured || !pc.built || !pc.valid) {
    return false;
  }
  for (const auto& macro : pc.macros) {
    if (q >= macro.local_i_begin && q <= macro.local_i_end &&
        j >= macro.local_j_begin && j <= macro.local_j_end) {
      return true;
    }
  }
  return false;
}

bool planar_diag_mask_has(const std::vector<std::uint8_t>& mask, const int n) {
  return n >= 0 && static_cast<std::size_t>(n) < mask.size() &&
         mask[static_cast<std::size_t>(n)] != 0U;
}

struct PoleAxisPlanarAccum {
  long double mhat = 0.0L;
  long double fhat_z = 0.0L;
};

constexpr double kPlanarCswRzEdgeCoeffPi =
    3.1415926535897932384626433832795028841971693993751;

bool planar_face_corners(const int active_nverts,
                         const int local,
                         int* corner0,
                         int* corner1) {
  if (active_nverts == 3) {
    return mesh::mesh_topo_active_local_face_corners(
        active_nverts, local, corner0, corner1);
  }
  if (local == 0) {
    *corner0 = 0;
    *corner1 = 3;
  } else if (local == 1) {
    *corner0 = 1;
    *corner1 = 2;
  } else if (local == 2) {
    *corner0 = 0;
    *corner1 = 1;
  } else if (local == 3) {
    *corner0 = 3;
    *corner1 = 2;
  } else {
    return false;
  }
  return true;
}

void add_csw_edge_av_planar_axis_force(
    const core::State& state,
    const core::Config& cfg,
    const std::vector<double>& node_r,
    std::vector<PoleAxisPlanarAccum>& accum) {
  if ((cfg.numerics.hydro.av_model != core::AvModel::CswEdge &&
       cfg.numerics.hydro.av_model != core::AvModel::CswEdgeCsw98) ||
      !state.mesh.topo.multiblock.has_value() || state.edge_force_av_z.empty()) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  if (mb.cell_node_csr_offsets.empty() || mb.cell_node_csr_indices.empty()) {
    return;
  }
  std::vector<double> edge_force_z;
  state.edge_force_av_z.copy_to_host(edge_force_z);
  const int n_nodes = static_cast<int>(accum.size());

  auto add_face = [&](const int edge_index, const int c, const int local) {
    if (edge_index < 0 || static_cast<std::size_t>(edge_index) >= edge_force_z.size() ||
        c < 0 || static_cast<std::size_t>(c + 1) >= mb.cell_node_csr_offsets.size()) {
      return;
    }
    const int active_nverts =
        mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
    int corner0 = 0;
    int corner1 = 0;
    if (!planar_face_corners(active_nverts, local, &corner0, &corner1)) {
      return;
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    if (off < 0 || static_cast<std::size_t>(off + std::max(corner0, corner1)) >=
                       mb.cell_node_csr_indices.size()) {
      return;
    }
    const int n0 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner0)];
    const int n1 = mb.cell_node_csr_indices[static_cast<std::size_t>(off + corner1)];
    if (n0 < 0 || n1 < 0 || n0 >= n_nodes || n1 >= n_nodes ||
        static_cast<std::size_t>(n0) >= node_r.size() ||
        static_cast<std::size_t>(n1) >= node_r.size()) {
      return;
    }
    const double coeff =
        kPlanarCswRzEdgeCoeffPi *
        (node_r[static_cast<std::size_t>(n0)] +
         node_r[static_cast<std::size_t>(n1)]);
    const double fz_planar =
        (std::isfinite(coeff) && std::fabs(coeff) > 0.0)
            ? edge_force_z[static_cast<std::size_t>(edge_index)] / coeff
            : 0.0;
    if (!std::isfinite(fz_planar)) {
      return;
    }
    accum[static_cast<std::size_t>(n0)].fhat_z -=
        static_cast<long double>(fz_planar);
    accum[static_cast<std::size_t>(n1)].fhat_z +=
        static_cast<long double>(fz_planar);
  };

  const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
  for (int f = 0; f < n_internal; ++f) {
    const mesh::UniqueOrientedFace& face =
        mb.unique_internal_faces[static_cast<std::size_t>(f)];
    add_face(f, face.cell_a, face.local_a);
  }
  const int n_boundary = static_cast<int>(mb.boundary_faces.size());
  for (int f = 0; f < n_boundary; ++f) {
    const mesh::UniqueOrientedFace& face =
        mb.boundary_faces[static_cast<std::size_t>(f)];
    add_face(n_internal + f, face.cell_a, face.local_a);
  }
}

void add_outer_pressure_planar_axis_force(
    const core::State& state,
    const core::Config& cfg,
    const double eval_time,
    const mesh::BlockInfo& shell,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z,
    std::vector<PoleAxisPlanarAccum>& accum) {
  const auto r_outer_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  if (r_outer_type != Boundary2DType::PRESSURE || !state.pressure_drive_1d.has_value() ||
      shell.owned_node_begin < 0 || shell.n_i_cells <= 0 ||
      shell.n_j_cells <= 0) {
    return;
  }
  const int n_nodes = static_cast<int>(accum.size());
  const int stride = shell.n_j_cells + 1;
  const int shell_node_count = (shell.n_i_cells + 1) * stride;
  if (shell.owned_node_begin + shell_node_count >
      static_cast<int>(node_r.size())) {
    return;
  }

  const double p_ext = state.pressure_drive_1d->eval(eval_time);
  const auto& pcfg =
      cfg.numerics.hydro.pressure_drive_perturbation;
  PressureDrivePerturbationParams pp{};
  if (pcfg.enabled) {
    pp = core::make_pressure_drive_perturbation_params(pcfg);
  }
  const double* shell_r = node_r.data() + shell.owned_node_begin;
  const double* shell_z = node_z.data() + shell.owned_node_begin;
  for (int j = 0; j < shell.n_j_cells; ++j) {
    const int n0 = shell.owned_node_begin + shell.n_i_cells * stride + j;
    const int n1 = n0 + 1;
    if (n0 < 0 || n1 < 0 || n0 >= n_nodes || n1 >= n_nodes) {
      continue;
    }
    double p_seg = p_ext;
    if (pcfg.enabled) {
      const int shell_n0 = shell.n_i_cells * stride + j;
      const int shell_n1 = shell_n0 + 1;
      const double rm = 0.5 * (shell_r[shell_n0] + shell_r[shell_n1]);
      const double zm = 0.5 * (shell_z[shell_n0] + shell_z[shell_n1]);
      // Host/device g evaluations may differ in the last ULP; this path has no parity contract.
      p_seg = p_ext * pressure_drive_perturbation_g(pp, rm, zm);
    }
    const pole_axis_bbsw::PlanarEndpointAreaVectors areas =
        pole_axis_bbsw::outer_boundary_endpoint_area_vectors(
            shell_r, shell_z, shell.n_i_cells, shell.n_j_cells, j);
    accum[static_cast<std::size_t>(n0)].fhat_z +=
        -static_cast<long double>(p_seg) *
        static_cast<long double>(areas.node0.z);
    accum[static_cast<std::size_t>(n1)].fhat_z +=
        -static_cast<long double>(p_seg) *
        static_cast<long double>(areas.node1.z);
  }
}

void emit_pole_axis_planar_diag(const core::State& state,
                                const core::Config& cfg,
                                const double eval_time,
                                const core::CellField1D& cell_pq,
                                const core::NodeField1D& node_mass,
                                const core::NodeField1D& force_z,
                                const core::NodeField1D& accel_z,
                                const std::int8_t* d_hydro_active,
                                const std::uint8_t* d_node_active,
                                const bool compatible_mode,
                                const bool selected_dynamics_call) {
  if (!selected_dynamics_call || !pole_axis_planar_diag_due(state, cfg)) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "pole_axis_planar requires multiblock topology");
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  const int n_i = shell.n_i_cells;
  const int n_j = shell.n_j_cells;
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  if (n_i <= 0 || n_j <= 0 || n_cells <= 0 || n_nodes <= 0) {
    return;
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> velocity_z;
  std::vector<double> node_mass_true;
  std::vector<double> force_z_true;
  std::vector<double> accel_z_current;
  std::vector<double> rho;
  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> pressure;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> qvisc;
  std::vector<double> corner_mass;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.v_z.copy_to_host(velocity_z);
  node_mass.copy_to_host(node_mass_true);
  force_z.copy_to_host(force_z_true);
  accel_z.copy_to_host(accel_z_current);
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(mass);
  state.vol.copy_to_host(vol);
  cell_pq.copy_to_host(pressure);
  if (state.Pe.size() == cell_pq.size()) {
    state.Pe.copy_to_host(Pe);
  }
  if (state.Pi.size() == cell_pq.size()) {
    state.Pi.copy_to_host(Pi);
  }
  if (state.Qvisc.size() == cell_pq.size()) {
    state.Qvisc.copy_to_host(qvisc);
  }
  if (state.corner_mass.size() ==
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(state.corner_stride)) {
    state.corner_mass.copy_to_host(corner_mass);
  }

  std::vector<std::int8_t> hydro_active;
  if (d_hydro_active != nullptr) {
    hydro_active.resize(static_cast<std::size_t>(n_cells), 0);
    cuda_check(cudaMemcpy(hydro_active.data(),
                          d_hydro_active,
                          hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyDeviceToHost),
               "pole_axis_planar copy hydro_active failed");
  }
  std::vector<std::uint8_t> node_active;
  if (d_node_active != nullptr) {
    node_active.resize(static_cast<std::size_t>(n_nodes), 0U);
    cuda_check(cudaMemcpy(node_active.data(),
                          d_node_active,
                          node_active.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "pole_axis_planar copy node_active failed");
  }

  auto cell_active = [&](const int c) {
    return hydro_active.empty() ||
           (c >= 0 && static_cast<std::size_t>(c) < hydro_active.size() &&
            hydro_active[static_cast<std::size_t>(c)] != 0);
  };

  std::vector<double> planar_pressure(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> raw_planar_pressure(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    double raw_p = 0.0;
    if (static_cast<std::size_t>(c) < Pe.size()) {
      raw_p += Pe[static_cast<std::size_t>(c)];
    }
    if (static_cast<std::size_t>(c) < Pi.size()) {
      raw_p += Pi[static_cast<std::size_t>(c)];
    }
    if (static_cast<std::size_t>(c) < qvisc.size()) {
      raw_p += qvisc[static_cast<std::size_t>(c)];
    }
    raw_planar_pressure[static_cast<std::size_t>(c)] =
        std::isfinite(raw_p) ? raw_p : 0.0;

    if (cell_active(c)) {
      double p = planar_diag_at_or_nan(pressure, c);
      if (compatible_mode && static_cast<std::size_t>(c) < qvisc.size()) {
        const double q = qvisc[static_cast<std::size_t>(c)];
        if (std::isfinite(q)) {
          p += q;
        }
      }
      planar_pressure[static_cast<std::size_t>(c)] =
          std::isfinite(p) ? p : 0.0;
    }
  }

  std::vector<PoleAxisPlanarAccum> accum(static_cast<std::size_t>(n_nodes));
  for (int c = 0; c < n_cells; ++c) {
    if (!cell_active(c)) {
      continue;
    }
    int nodes[4] = {-1, -1, -1, -1};
    int active_nverts = mesh::kMeshTopoCellStorageSlots;
    if (!trace_cell_nodes(state, c, nodes, &active_nverts)) {
      continue;
    }
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    double z[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < active_nverts; ++k) {
      r[k] = planar_diag_at_or_nan(node_r, nodes[k]);
      z[k] = planar_diag_at_or_nan(node_z, nodes[k]);
    }
    const double orientation = planar_diag_cell_orientation_sign(state, c);
    double area[4] = {0.0, 0.0, 0.0, 0.0};
    double cvec_r[4] = {0.0, 0.0, 0.0, 0.0};
    double cvec_z[4] = {0.0, 0.0, 0.0, 0.0};
    planar_diag_corner_areas(r, z, active_nverts, orientation, area);
    planar_diag_corner_vectors(r, z, active_nverts, orientation, cvec_r, cvec_z);
    const double p = planar_pressure[static_cast<std::size_t>(c)];
    for (int k = 0; k < active_nverts; ++k) {
      const int n = nodes[k];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const double density =
          planar_diag_corner_density(rho, mass, vol, corner_mass, c, k,
                                     active_nverts, state.corner_stride, r, z);
      auto& a = accum[static_cast<std::size_t>(n)];
      a.mhat += static_cast<long double>(density) *
                static_cast<long double>(area[k]);
      a.fhat_z += static_cast<long double>(p) *
                  static_cast<long double>(cvec_z[k]);
    }
  }

  const auto& pc = state.pole_angular_derefine;
  if (pc.configured && pc.built && pc.valid) {
    for (const auto& macro : pc.macros) {
      const int n_boundary =
          static_cast<int>(macro.boundary_nodes_ordered.size());
      if (n_boundary < 3) {
        continue;
      }
      std::vector<double> r(static_cast<std::size_t>(n_boundary), 0.0);
      std::vector<double> z(static_cast<std::size_t>(n_boundary), 0.0);
      for (int k = 0; k < n_boundary; ++k) {
        const int n = macro.boundary_nodes_ordered[static_cast<std::size_t>(k)];
        r[static_cast<std::size_t>(k)] = planar_diag_at_or_nan(node_r, n);
        z[static_cast<std::size_t>(k)] = planar_diag_at_or_nan(node_z, n);
      }
      std::vector<double> area(static_cast<std::size_t>(n_boundary), 0.0);
      std::vector<double> cvec_r(static_cast<std::size_t>(n_boundary), 0.0);
      std::vector<double> cvec_z(static_cast<std::size_t>(n_boundary), 0.0);
      planar_diag_corner_areas(r.data(), z.data(), n_boundary, 1.0,
                               area.data());
      planar_diag_corner_vectors(r.data(), z.data(), n_boundary, 1.0,
                                 cvec_r.data(), cvec_z.data());
      const double density =
          planar_diag_macro_density(macro, node_r, node_z, mass, rho, vol);
      double p_macro = planar_diag_macro_pressure(macro, mass, planar_pressure);
      if (!(p_macro > 0.0) || !std::isfinite(p_macro)) {
        p_macro = planar_diag_macro_pressure(macro, mass, raw_planar_pressure);
      }
      for (int k = 0; k < n_boundary; ++k) {
        const int n = macro.boundary_nodes_ordered[static_cast<std::size_t>(k)];
        if (n < 0 || n >= n_nodes) {
          continue;
        }
        auto& a = accum[static_cast<std::size_t>(n)];
        a.mhat += static_cast<long double>(density) *
                  static_cast<long double>(area[static_cast<std::size_t>(k)]);
        a.fhat_z += static_cast<long double>(p_macro) *
                    static_cast<long double>(cvec_z[static_cast<std::size_t>(k)]);
      }
    }
  }

  add_csw_edge_av_planar_axis_force(state, cfg, node_r, accum);
  add_outer_pressure_planar_axis_force(
      state, cfg, eval_time, shell, node_r, node_z, accum);

  int shell_block_id = -1;
  const auto& blocks = state.mesh.topo.multiblock->blocks;
  for (int b = 0; b < static_cast<int>(blocks.size()); ++b) {
    if (&blocks[static_cast<std::size_t>(b)] == &shell) {
      shell_block_id = b;
      break;
    }
  }

  const int stride = n_j + 1;
  const auto shell_node = [&](const int q, const int j) {
    return shell.owned_node_begin + q * stride + j;
  };
  const int north_ref = shell_node(n_i, 0);
  const int south_ref = shell_node(n_i, n_j);
  const double z_c = 0.5 * (planar_diag_at_or_nan(node_z, north_ref) +
                            planar_diag_at_or_nan(node_z, south_ref));
  const int columns[2] = {0, n_j};
  for (const int j : columns) {
    for (int q = 0; q <= n_i; ++q) {
      const int n = shell_node(q, j);
      const int outer = (q < n_i) ? shell_node(q + 1, j) : -1;
      const double s = std::abs(planar_diag_at_or_nan(node_z, n) - z_c);
      const double s_outer =
          outer >= 0 ? std::abs(planar_diag_at_or_nan(node_z, outer) - z_c)
                     : std::numeric_limits<double>::quiet_NaN();
      const double gap = outer >= 0 ? s_outer - s
                                    : std::numeric_limits<double>::quiet_NaN();
      const auto& a = accum[static_cast<std::size_t>(n)];
      const double mhat = static_cast<double>(a.mhat);
      const double fhat_z = static_cast<double>(a.fhat_z);
      const double az_planar =
          (std::isfinite(mhat) && mhat > 0.0 && std::isfinite(fhat_z))
              ? fhat_z / mhat
              : std::numeric_limits<double>::quiet_NaN();
      const bool pole_member =
          planar_diag_pole_deref_macro_contains_node(state, q, j);
      const bool pole_boundary =
          planar_diag_mask_has(pc.boundary_node_mask, n);

      std::ostringstream os;
      os << std::setprecision(17)
         << "[pole_axis_planar]"
         << " step=" << state.step
         << " phase=post_accel_kernel"
         << " block=" << shell_block_id
         << " node_id=" << n
         << " local_i=" << q
         << " local_j=" << j
         << " node_active=" << planar_diag_u8_or_minus_one(node_active, n)
         << " node_mass_true=" << planar_diag_at_or_nan(node_mass_true, n)
         << " force_z_true=" << planar_diag_at_or_nan(force_z_true, n)
         << " accel_z_current=" << planar_diag_at_or_nan(accel_z_current, n)
         << " mhat_planar=" << mhat
         << " fhat_z_planar=" << fhat_z
         << " a_z_planar=" << az_planar
         << " v_z=" << planar_diag_at_or_nan(velocity_z, n)
         << " gap=" << gap
         << " pole_deref_member=" << (pole_member ? 1 : 0)
         << " pole_deref_boundary=" << (pole_boundary ? 1 : 0);
      core::log_warning(os.str());
    }
  }
}

std::vector<PoleAxisPlanarAccum> compute_pole_axis_planar_accum_host(
    const core::State& state,
    const core::Config& cfg,
    const mesh::BlockInfo& shell,
    const double eval_time,
    const core::CellField1D& cell_pq,
    const std::int8_t* d_hydro_active,
    const bool compatible_mode,
    std::vector<double>& node_r,
    std::vector<double>& node_z,
    std::vector<std::uint8_t>& node_active,
    const std::uint8_t* d_node_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  node_r.clear();
  node_z.clear();
  node_active.clear();
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  if (d_node_active != nullptr && n_nodes > 0) {
    node_active.resize(static_cast<std::size_t>(n_nodes), 0U);
    cuda_check(cudaMemcpy(node_active.data(),
                          d_node_active,
                          node_active.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "pole_axis_bbsw copy node_active failed");
  }

  std::vector<double> rho;
  std::vector<double> mass;
  std::vector<double> vol;
  std::vector<double> pressure;
  std::vector<double> Pe;
  std::vector<double> Pi;
  std::vector<double> qvisc;
  std::vector<double> corner_mass;
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(mass);
  state.vol.copy_to_host(vol);
  cell_pq.copy_to_host(pressure);
  if (state.Pe.size() == cell_pq.size()) {
    state.Pe.copy_to_host(Pe);
  }
  if (state.Pi.size() == cell_pq.size()) {
    state.Pi.copy_to_host(Pi);
  }
  if (state.Qvisc.size() == cell_pq.size()) {
    state.Qvisc.copy_to_host(qvisc);
  }
  if (state.corner_mass.size() ==
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(state.corner_stride)) {
    state.corner_mass.copy_to_host(corner_mass);
  }

  std::vector<std::int8_t> hydro_active;
  if (d_hydro_active != nullptr && n_cells > 0) {
    hydro_active.resize(static_cast<std::size_t>(n_cells), 0);
    cuda_check(cudaMemcpy(hydro_active.data(),
                          d_hydro_active,
                          hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyDeviceToHost),
               "pole_axis_bbsw copy hydro_active failed");
  }
  auto cell_active = [&](const int c) {
    return hydro_active.empty() ||
           (c >= 0 && static_cast<std::size_t>(c) < hydro_active.size() &&
            hydro_active[static_cast<std::size_t>(c)] != 0);
  };

  std::vector<double> planar_pressure(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> raw_planar_pressure(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    double raw_p = 0.0;
    if (static_cast<std::size_t>(c) < Pe.size()) {
      raw_p += Pe[static_cast<std::size_t>(c)];
    }
    if (static_cast<std::size_t>(c) < Pi.size()) {
      raw_p += Pi[static_cast<std::size_t>(c)];
    }
    if (static_cast<std::size_t>(c) < qvisc.size()) {
      raw_p += qvisc[static_cast<std::size_t>(c)];
    }
    raw_planar_pressure[static_cast<std::size_t>(c)] =
        std::isfinite(raw_p) ? raw_p : 0.0;
    if (cell_active(c)) {
      double p = planar_diag_at_or_nan(pressure, c);
      if (compatible_mode && static_cast<std::size_t>(c) < qvisc.size()) {
        const double q = qvisc[static_cast<std::size_t>(c)];
        if (std::isfinite(q)) {
          p += q;
        }
      }
      planar_pressure[static_cast<std::size_t>(c)] =
          std::isfinite(p) ? p : 0.0;
    }
  }

  std::vector<PoleAxisPlanarAccum> accum(static_cast<std::size_t>(n_nodes));
  for (int c = 0; c < n_cells; ++c) {
    if (!cell_active(c)) {
      continue;
    }
    int nodes[4] = {-1, -1, -1, -1};
    int active_nverts = mesh::kMeshTopoCellStorageSlots;
    if (!trace_cell_nodes(state, c, nodes, &active_nverts)) {
      continue;
    }
    double r[4] = {0.0, 0.0, 0.0, 0.0};
    double z[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < active_nverts; ++k) {
      r[k] = planar_diag_at_or_nan(node_r, nodes[k]);
      z[k] = planar_diag_at_or_nan(node_z, nodes[k]);
    }
    const double orientation = planar_diag_cell_orientation_sign(state, c);
    double area[4] = {0.0, 0.0, 0.0, 0.0};
    double cvec_r[4] = {0.0, 0.0, 0.0, 0.0};
    double cvec_z[4] = {0.0, 0.0, 0.0, 0.0};
    planar_diag_corner_areas(r, z, active_nverts, orientation, area);
    planar_diag_corner_vectors(r, z, active_nverts, orientation, cvec_r, cvec_z);
    const double p = planar_pressure[static_cast<std::size_t>(c)];
    for (int k = 0; k < active_nverts; ++k) {
      const int n = nodes[k];
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      const double density =
          planar_diag_corner_density(rho, mass, vol, corner_mass, c, k,
                                     active_nverts, state.corner_stride, r, z);
      auto& a = accum[static_cast<std::size_t>(n)];
      a.mhat += static_cast<long double>(density) *
                static_cast<long double>(area[k]);
      a.fhat_z += static_cast<long double>(p) *
                  static_cast<long double>(cvec_z[k]);
    }
  }

  const auto& pc = state.pole_angular_derefine;
  if (pc.configured && pc.built && pc.valid) {
    for (const auto& macro : pc.macros) {
      const int n_boundary =
          static_cast<int>(macro.boundary_nodes_ordered.size());
      if (n_boundary < 3) {
        continue;
      }
      std::vector<double> r(static_cast<std::size_t>(n_boundary), 0.0);
      std::vector<double> z(static_cast<std::size_t>(n_boundary), 0.0);
      for (int k = 0; k < n_boundary; ++k) {
        const int n = macro.boundary_nodes_ordered[static_cast<std::size_t>(k)];
        r[static_cast<std::size_t>(k)] = planar_diag_at_or_nan(node_r, n);
        z[static_cast<std::size_t>(k)] = planar_diag_at_or_nan(node_z, n);
      }
      std::vector<double> area(static_cast<std::size_t>(n_boundary), 0.0);
      std::vector<double> cvec_r(static_cast<std::size_t>(n_boundary), 0.0);
      std::vector<double> cvec_z(static_cast<std::size_t>(n_boundary), 0.0);
      planar_diag_corner_areas(r.data(), z.data(), n_boundary, 1.0,
                               area.data());
      planar_diag_corner_vectors(r.data(), z.data(), n_boundary, 1.0,
                                 cvec_r.data(), cvec_z.data());
      const double density =
          planar_diag_macro_density(macro, node_r, node_z, mass, rho, vol);
      double p_macro = planar_diag_macro_pressure(macro, mass, planar_pressure);
      if (!(p_macro > 0.0) || !std::isfinite(p_macro)) {
        p_macro = planar_diag_macro_pressure(macro, mass, raw_planar_pressure);
      }
      for (int k = 0; k < n_boundary; ++k) {
        const int n = macro.boundary_nodes_ordered[static_cast<std::size_t>(k)];
        if (n < 0 || n >= n_nodes) {
          continue;
        }
        auto& a = accum[static_cast<std::size_t>(n)];
        a.mhat += static_cast<long double>(density) *
                  static_cast<long double>(area[static_cast<std::size_t>(k)]);
        a.fhat_z += static_cast<long double>(p_macro) *
                    static_cast<long double>(cvec_z[static_cast<std::size_t>(k)]);
      }
    }
  }

  add_csw_edge_av_planar_axis_force(state, cfg, node_r, accum);
  add_outer_pressure_planar_axis_force(
      state, cfg, eval_time, shell, node_r, node_z, accum);
  return accum;
}

void apply_pole_axis_bbsw_acceleration(
    core::NodeField1D& accel_r,
    core::NodeField1D& accel_z,
    core::State& state,
    const core::Config& cfg,
    const double eval_time,
    const core::CellField1D& cell_pq,
    const std::int8_t* d_hydro_active,
    const std::uint8_t* d_node_active,
    const bool compatible_mode,
    const bool selected_dynamics_call) {
  if (!selected_dynamics_call || !pole_axis_bbsw_enabled(cfg) ||
      !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  if (shell.n_i_cells <= 0 || shell.n_j_cells <= 0) {
    return;
  }
  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<std::uint8_t> node_active;
  const std::vector<PoleAxisPlanarAccum> accum =
      compute_pole_axis_planar_accum_host(state,
                                          cfg,
                                          shell,
                                          eval_time,
                                          cell_pq,
                                          d_hydro_active,
                                          compatible_mode,
                                          node_r,
                                          node_z,
                                          node_active,
                                          d_node_active);
  const int n_nodes = static_cast<int>(state.x_r.size());
  std::vector<double> ar;
  std::vector<double> az;
  accel_r.copy_to_host(ar);
  accel_z.copy_to_host(az);
  if (static_cast<int>(ar.size()) != n_nodes ||
      static_cast<int>(az.size()) != n_nodes) {
    return;
  }
  if (mesh::mesh_topo_has_polar_tier_cart_center(cfg.mesh)) {
    const std::vector<int> nodes = pole_axis_bbsw_hybrid_cart_axis_nodes(
        state, cfg, node_r, node_z);
    for (const int n : nodes) {
      if (n < 0 || n >= n_nodes) {
        continue;
      }
      if (!node_active.empty() &&
          node_active[static_cast<std::size_t>(n)] == 0U) {
        continue;
      }
      const auto& a = accum[static_cast<std::size_t>(n)];
      const double mhat = static_cast<double>(a.mhat);
      const double fhat_z = static_cast<double>(a.fhat_z);
      if (!(std::isfinite(mhat) && mhat > 0.0 && std::isfinite(fhat_z))) {
        continue;
      }
      ar[static_cast<std::size_t>(n)] = 0.0;
      az[static_cast<std::size_t>(n)] = fhat_z / mhat;
    }
  } else {
    const int stride = shell.n_j_cells + 1;
    const int columns[2] = {0, shell.n_j_cells};
    for (const int j : columns) {
      for (int q = 0; q <= shell.n_i_cells; ++q) {
        const int n = shell.owned_node_begin + q * stride + j;
        if (n < 0 || n >= n_nodes) {
          continue;
        }
        if (!node_active.empty() &&
            node_active[static_cast<std::size_t>(n)] == 0U) {
          continue;
        }
        const auto& a = accum[static_cast<std::size_t>(n)];
        const double mhat = static_cast<double>(a.mhat);
        const double fhat_z = static_cast<double>(a.fhat_z);
        if (!(std::isfinite(mhat) && mhat > 0.0 && std::isfinite(fhat_z))) {
          continue;
        }
        ar[static_cast<std::size_t>(n)] = 0.0;
        az[static_cast<std::size_t>(n)] = fhat_z / mhat;
      }
    }
  }
  accel_r.copy_from_host(ar.data());
  accel_z.copy_from_host(az.data());
}

void apply_pole_axis_bbsw_order_constraint(
    core::State& state,
    const core::Config& cfg,
    const double eval_time,
    const core::CellField1D& cell_pq,
    const std::int8_t* d_hydro_active,
    const std::uint8_t* d_node_active,
    const bool compatible_mode,
    const core::NodeField1D& x_z_old,
    core::NodeField1D& pos_v_r,
    core::NodeField1D& pos_v_z,
    core::NodeField1D* midpoint_v_r,
    core::NodeField1D* midpoint_v_z,
    core::NodeField1D* final_v_r,
    core::NodeField1D* final_v_z,
    const core::NodeField1D* old_v_z,
    const double dt) {
  if (!pole_axis_bbsw_enabled(cfg) || !(dt > 0.0) ||
      !state.mesh.topo.multiblock.has_value()) {
    return;
  }
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  if (shell.n_i_cells <= 0 || shell.n_j_cells <= 0) {
    return;
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<std::uint8_t> node_active;
  const std::vector<PoleAxisPlanarAccum> accum =
      compute_pole_axis_planar_accum_host(state,
                                          cfg,
                                          shell,
                                          eval_time,
                                          cell_pq,
                                          d_hydro_active,
                                          compatible_mode,
                                          node_r,
                                          node_z,
                                          node_active,
                                          d_node_active);
  const int n_nodes = static_cast<int>(state.x_r.size());
  std::vector<double> z_old;
  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> mid_r;
  std::vector<double> mid_z;
  std::vector<double> fin_r;
  std::vector<double> fin_z;
  std::vector<double> old_z_vel;
  x_z_old.copy_to_host(z_old);
  pos_v_r.copy_to_host(pos_r);
  pos_v_z.copy_to_host(pos_z);
  if (midpoint_v_r != nullptr && midpoint_v_r->data() != pos_v_r.data()) {
    midpoint_v_r->copy_to_host(mid_r);
  }
  if (midpoint_v_z != nullptr && midpoint_v_z->data() != pos_v_z.data()) {
    midpoint_v_z->copy_to_host(mid_z);
  }
  if (final_v_r != nullptr) {
    final_v_r->copy_to_host(fin_r);
  }
  if (final_v_z != nullptr) {
    final_v_z->copy_to_host(fin_z);
  }
  if (old_v_z != nullptr) {
    old_v_z->copy_to_host(old_z_vel);
  }
  if (static_cast<int>(z_old.size()) != n_nodes ||
      static_cast<int>(pos_r.size()) != n_nodes ||
      static_cast<int>(pos_z.size()) != n_nodes) {
    return;
  }

  if (mesh::mesh_topo_has_polar_tier_cart_center(cfg.mesh)) {
    const std::vector<int> nodes = pole_axis_bbsw_hybrid_cart_axis_nodes(
        state, cfg, node_r, z_old);
    const int n_axis = static_cast<int>(nodes.size());
    if (n_axis <= 0) {
      return;
    }
    std::vector<double> z_hat(static_cast<std::size_t>(n_axis), 0.0);
    std::vector<double> weight(static_cast<std::size_t>(n_axis), 1.0e-300);
    std::vector<double> delta(
        static_cast<std::size_t>(std::max(0, n_axis - 1)), 0.0);
    for (int q = 0; q < n_axis; ++q) {
      const int n = nodes[static_cast<std::size_t>(q)];
      if (!node_active.empty() &&
          node_active[static_cast<std::size_t>(n)] == 0U) {
        return;
      }
      z_hat[static_cast<std::size_t>(q)] =
          z_old[static_cast<std::size_t>(n)] +
          dt * pos_z[static_cast<std::size_t>(n)];
      const double mhat =
          static_cast<double>(accum[static_cast<std::size_t>(n)].mhat);
      weight[static_cast<std::size_t>(q)] =
          (std::isfinite(mhat) && mhat > 0.0) ? mhat : 1.0e-300;
    }
    for (int q = 0; q + 1 < n_axis; ++q) {
      const int n0 = nodes[static_cast<std::size_t>(q)];
      const int n1 = nodes[static_cast<std::size_t>(q + 1)];
      delta[static_cast<std::size_t>(q)] = pole_axis_bbsw::hard_gap(
          z_old[static_cast<std::size_t>(n0)],
          z_old[static_cast<std::size_t>(n1)]);
    }
    const std::vector<double> z_proj =
        pole_axis_bbsw::project_axis_positions_with_hard_gaps(
            z_hat, weight, delta);
    bool changed = false;
    for (int q = 0; q < n_axis; ++q) {
      const int n = nodes[static_cast<std::size_t>(q)];
      const double u_mid =
          (z_proj[static_cast<std::size_t>(q)] -
           z_old[static_cast<std::size_t>(n)]) /
          dt;
      if (!std::isfinite(u_mid)) {
        continue;
      }
      pos_r[static_cast<std::size_t>(n)] = 0.0;
      pos_z[static_cast<std::size_t>(n)] = u_mid;
      if (!mid_r.empty()) {
        mid_r[static_cast<std::size_t>(n)] = 0.0;
      }
      if (!mid_z.empty()) {
        mid_z[static_cast<std::size_t>(n)] = u_mid;
      }
      if (!fin_r.empty()) {
        fin_r[static_cast<std::size_t>(n)] = 0.0;
      }
      if (!fin_z.empty()) {
        const double old_u =
            (!old_z_vel.empty() &&
             static_cast<std::size_t>(n) < old_z_vel.size())
                ? old_z_vel[static_cast<std::size_t>(n)]
                : 0.0;
        fin_z[static_cast<std::size_t>(n)] = 2.0 * u_mid - old_u;
      }
      changed = true;
    }
    if (!changed) {
      return;
    }
    pos_v_r.copy_from_host(pos_r.data());
    pos_v_z.copy_from_host(pos_z.data());
    if (midpoint_v_r != nullptr && midpoint_v_r->data() != pos_v_r.data() &&
        !mid_r.empty()) {
      midpoint_v_r->copy_from_host(mid_r.data());
    }
    if (midpoint_v_z != nullptr && midpoint_v_z->data() != pos_v_z.data() &&
        !mid_z.empty()) {
      midpoint_v_z->copy_from_host(mid_z.data());
    }
    if (final_v_r != nullptr && !fin_r.empty()) {
      final_v_r->copy_from_host(fin_r.data());
    }
    if (final_v_z != nullptr && !fin_z.empty()) {
      final_v_z->copy_from_host(fin_z.data());
    }
    return;
  }

  const int stride = shell.n_j_cells + 1;
  const auto shell_node = [&](const int q, const int j) {
    return shell.owned_node_begin + q * stride + j;
  };
  const int north_ref = shell_node(shell.n_i_cells, 0);
  const int south_ref = shell_node(shell.n_i_cells, shell.n_j_cells);
  if (north_ref < 0 || south_ref < 0 || north_ref >= n_nodes ||
      south_ref >= n_nodes) {
    return;
  }
  const double z_c = 0.5 * (z_old[static_cast<std::size_t>(north_ref)] +
                            z_old[static_cast<std::size_t>(south_ref)]);

  const int columns[2] = {0, shell.n_j_cells};
  bool changed = false;
  for (const int j : columns) {
    const int n_axis = shell.n_i_cells + 1;
    std::vector<int> nodes(static_cast<std::size_t>(n_axis), -1);
    std::vector<double> s_old(static_cast<std::size_t>(n_axis), 0.0);
    std::vector<double> s_hat(static_cast<std::size_t>(n_axis), 0.0);
    std::vector<double> weight(static_cast<std::size_t>(n_axis), 1.0e-300);
    std::vector<double> sign(static_cast<std::size_t>(n_axis), 1.0);
    std::vector<double> delta(static_cast<std::size_t>(std::max(0, n_axis - 1)),
                              0.0);
    bool usable = true;
    for (int q = 0; q < n_axis; ++q) {
      const int n = shell_node(q, j);
      nodes[static_cast<std::size_t>(q)] = n;
      if (n < 0 || n >= n_nodes ||
          (!node_active.empty() &&
           node_active[static_cast<std::size_t>(n)] == 0U)) {
        usable = false;
        break;
      }
      const double dz = z_old[static_cast<std::size_t>(n)] - z_c;
      sign[static_cast<std::size_t>(q)] = (dz >= 0.0) ? 1.0 : -1.0;
      s_old[static_cast<std::size_t>(q)] = std::fabs(dz);
      s_hat[static_cast<std::size_t>(q)] =
          s_old[static_cast<std::size_t>(q)] +
          dt * sign[static_cast<std::size_t>(q)] *
              pos_z[static_cast<std::size_t>(n)];
      const double mhat =
          static_cast<double>(accum[static_cast<std::size_t>(n)].mhat);
      weight[static_cast<std::size_t>(q)] =
          (std::isfinite(mhat) && mhat > 0.0) ? mhat : 1.0e-300;
    }
    if (!usable) {
      continue;
    }
    for (int q = 0; q + 1 < n_axis; ++q) {
      delta[static_cast<std::size_t>(q)] =
          pole_axis_bbsw::hard_gap(s_old[static_cast<std::size_t>(q)],
                                   s_old[static_cast<std::size_t>(q + 1)]);
    }
    const std::vector<double> s_proj =
        pole_axis_bbsw::project_axis_positions_with_hard_gaps(
            s_hat, weight, delta);
    for (int q = 0; q < n_axis; ++q) {
      const int n = nodes[static_cast<std::size_t>(q)];
      const double u_mid =
          sign[static_cast<std::size_t>(q)] *
          (s_proj[static_cast<std::size_t>(q)] -
           s_old[static_cast<std::size_t>(q)]) /
          dt;
      if (std::isfinite(u_mid)) {
        pos_r[static_cast<std::size_t>(n)] = 0.0;
        pos_z[static_cast<std::size_t>(n)] = u_mid;
        if (!mid_r.empty()) {
          mid_r[static_cast<std::size_t>(n)] = 0.0;
        }
        if (!mid_z.empty()) {
          mid_z[static_cast<std::size_t>(n)] = u_mid;
        }
        if (!fin_r.empty()) {
          fin_r[static_cast<std::size_t>(n)] = 0.0;
        }
        if (!fin_z.empty()) {
          const double old_u =
              (!old_z_vel.empty() &&
               static_cast<std::size_t>(n) < old_z_vel.size())
                  ? old_z_vel[static_cast<std::size_t>(n)]
                  : 0.0;
          fin_z[static_cast<std::size_t>(n)] = 2.0 * u_mid - old_u;
        }
        changed = true;
      }
    }
  }
  if (!changed) {
    return;
  }
  pos_v_r.copy_from_host(pos_r.data());
  pos_v_z.copy_from_host(pos_z.data());
  if (midpoint_v_r != nullptr && midpoint_v_r->data() != pos_v_r.data() &&
      !mid_r.empty()) {
    midpoint_v_r->copy_from_host(mid_r.data());
  }
  if (midpoint_v_z != nullptr && midpoint_v_z->data() != pos_v_z.data() &&
      !mid_z.empty()) {
    midpoint_v_z->copy_from_host(mid_z.data());
  }
  if (final_v_r != nullptr && !fin_r.empty()) {
    final_v_r->copy_from_host(fin_r.data());
  }
  if (final_v_z != nullptr && !fin_z.empty()) {
    final_v_z->copy_from_host(fin_z.data());
  }
}

void apply_pole_axis_bbsw_work_correction(
    core::State& state,
    const core::Config& cfg,
    const double dt,
    const core::NodeField1D& old_v_z,
    const core::NodeField1D& final_v_z,
    const core::NodeField1D& node_mass,
    const core::NodeField1D& true_force_z,
    const std::uint8_t* d_node_active,
    const std::int8_t* d_hydro_active,
    const bool compatible_mode) {
  if (!pole_axis_bbsw_enabled(cfg) || !compatible_mode || !(dt > 0.0) ||
      !state.mesh.topo.multiblock.has_value() || true_force_z.empty()) {
    return;
  }
  const mesh::BlockInfo& shell =
      mesh::mesh_topo_multiblock_polar_shell_block(*state.mesh.topo.multiblock);
  if (shell.n_i_cells <= 0 || shell.n_j_cells <= 0) {
    return;
  }

  const int n_nodes = static_cast<int>(state.x_r.size());
  const int n_cells = static_cast<int>(state.rho.size());
  std::vector<double> old_v;
  std::vector<double> new_v;
  std::vector<double> mass_node;
  std::vector<double> force_true;
  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> rho;
  std::vector<double> cell_mass;
  std::vector<double> vol;
  std::vector<double> corner_mass;
  std::vector<double> work_p;
  old_v_z.copy_to_host(old_v);
  final_v_z.copy_to_host(new_v);
  node_mass.copy_to_host(mass_node);
  true_force_z.copy_to_host(force_true);
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.rho.copy_to_host(rho);
  state.mass.copy_to_host(cell_mass);
  state.vol.copy_to_host(vol);
  state.work_p_per_cell.copy_to_host(work_p);
  if (state.corner_mass.size() ==
      static_cast<std::size_t>(n_cells) *
          static_cast<std::size_t>(state.corner_stride)) {
    state.corner_mass.copy_to_host(corner_mass);
  }
  if (static_cast<int>(old_v.size()) != n_nodes ||
      static_cast<int>(new_v.size()) != n_nodes ||
      static_cast<int>(mass_node.size()) != n_nodes ||
      static_cast<int>(force_true.size()) != n_nodes ||
      static_cast<int>(work_p.size()) != n_cells) {
    return;
  }

  std::vector<std::uint8_t> node_active;
  if (d_node_active != nullptr && n_nodes > 0) {
    node_active.resize(static_cast<std::size_t>(n_nodes), 0U);
    cuda_check(cudaMemcpy(node_active.data(),
                          d_node_active,
                          node_active.size() * sizeof(std::uint8_t),
                          cudaMemcpyDeviceToHost),
               "pole_axis_bbsw work correction copy node_active failed");
  }
  std::vector<std::int8_t> hydro_active;
  if (d_hydro_active != nullptr && n_cells > 0) {
    hydro_active.resize(static_cast<std::size_t>(n_cells), 0);
    cuda_check(cudaMemcpy(hydro_active.data(),
                          d_hydro_active,
                          hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyDeviceToHost),
               "pole_axis_bbsw work correction copy hydro_active failed");
  }
  auto cell_active = [&](const int c) {
    return hydro_active.empty() ||
           (c >= 0 && static_cast<std::size_t>(c) < hydro_active.size() &&
            hydro_active[static_cast<std::size_t>(c)] != 0);
  };

  std::vector<int> reverse_offsets;
  std::vector<int> reverse_cells;
  std::vector<int> reverse_corners;
  state.mesh.multiblock_reverse_csr_node_offsets.copy_to_host(reverse_offsets);
  state.mesh.multiblock_reverse_csr_node_cells.copy_to_host(reverse_cells);
  state.mesh.multiblock_reverse_csr_node_corners.copy_to_host(reverse_corners);
  if (reverse_offsets.size() != static_cast<std::size_t>(n_nodes) + 1U) {
    return;
  }

  std::vector<int> owned_axis_nodes;
  if (mesh::mesh_topo_has_polar_tier_cart_center(cfg.mesh)) {
    owned_axis_nodes = pole_axis_bbsw_hybrid_cart_axis_nodes(
        state, cfg, node_r, node_z);
  } else {
    const int stride = shell.n_j_cells + 1;
    const int columns[2] = {0, shell.n_j_cells};
    owned_axis_nodes.reserve(
        static_cast<std::size_t>(2 * (shell.n_i_cells + 1)));
    for (const int j : columns) {
      for (int q = 0; q <= shell.n_i_cells; ++q) {
        owned_axis_nodes.push_back(shell.owned_node_begin + q * stride + j);
      }
    }
  }
  bool changed = false;
  for (const int n : owned_axis_nodes) {
    if (n < 0 || n >= n_nodes ||
        (!node_active.empty() &&
         node_active[static_cast<std::size_t>(n)] == 0U)) {
      continue;
    }
    const double m_node = mass_node[static_cast<std::size_t>(n)];
    const double v_old = old_v[static_cast<std::size_t>(n)];
    const double v_new = new_v[static_cast<std::size_t>(n)];
    const double f_true = force_true[static_cast<std::size_t>(n)];
    if (!(std::isfinite(m_node) && m_node > 0.0 && std::isfinite(v_old) &&
          std::isfinite(v_new) && std::isfinite(f_true))) {
      continue;
    }
    const double f_eff = m_node * (v_new - v_old) / dt;
    const double u_mid = 0.5 * (v_old + v_new);
    const double work_rate = -u_mid * (f_eff - f_true);
    if (!std::isfinite(work_rate) || work_rate == 0.0) {
      continue;
    }

    struct IncidentWeight {
      int cell = -1;
      double weight = 0.0;
    };
    std::vector<IncidentWeight> incident;
    const int off = reverse_offsets[static_cast<std::size_t>(n)];
    const int end = reverse_offsets[static_cast<std::size_t>(n + 1)];
    double weight_sum = 0.0;
    for (int p = off; p < end; ++p) {
      if (p < 0 || static_cast<std::size_t>(p) >= reverse_cells.size() ||
          static_cast<std::size_t>(p) >= reverse_corners.size()) {
        continue;
      }
      const int c = reverse_cells[static_cast<std::size_t>(p)];
      const int corner = reverse_corners[static_cast<std::size_t>(p)];
      if (c < 0 || c >= n_cells || !cell_active(c)) {
        continue;
      }
      int nodes[4] = {-1, -1, -1, -1};
      int active_nverts = mesh::kMeshTopoCellStorageSlots;
      if (!trace_cell_nodes(state, c, nodes, &active_nverts) ||
          corner < 0 || corner >= active_nverts) {
        continue;
      }
      double r[4] = {0.0, 0.0, 0.0, 0.0};
      double z[4] = {0.0, 0.0, 0.0, 0.0};
      for (int k = 0; k < active_nverts; ++k) {
        r[k] = planar_diag_at_or_nan(node_r, nodes[k]);
        z[k] = planar_diag_at_or_nan(node_z, nodes[k]);
      }
      double area[4] = {0.0, 0.0, 0.0, 0.0};
      planar_diag_corner_areas(r,
                               z,
                               active_nverts,
                               planar_diag_cell_orientation_sign(state, c),
                               area);
      const double density =
          planar_diag_corner_density(rho,
                                     cell_mass,
                                     vol,
                                     corner_mass,
                                     c,
                                     corner,
                                     active_nverts,
                                     state.corner_stride,
                                     r,
                                     z);
      double weight = density * area[corner];
      if (!(std::isfinite(weight) && weight > 0.0)) {
        weight = 1.0;
      }
      incident.push_back(IncidentWeight{c, weight});
      weight_sum += weight;
    }
    if (!(weight_sum > 0.0) || incident.empty()) {
      continue;
    }
    for (const IncidentWeight entry : incident) {
      work_p[static_cast<std::size_t>(entry.cell)] +=
          work_rate * entry.weight / weight_sum;
    }
    changed = true;
  }
  if (changed) {
    state.work_p_per_cell.copy_from_host(work_p.data());
  }
}

void trace_compatible_force_diag(const core::State& state,
                                 const core::Config& cfg,
                                 const core::CellField1D& cell_pq,
                                 const core::NodeField1D& node_mass,
                                 const core::NodeField1D& force_r,
                                 const core::NodeField1D& force_z) {
  if (!mesh_trace::enabled(state, cfg) ||
      state.step >= cfg.numerics.debug.trace_max_steps) {
    return;
  }
  const int c = cfg.numerics.debug.trace_mesh_cell;
  int nodes[4] = {-1, -1, -1, -1};
  int active_nverts = mesh::kMeshTopoCellStorageSlots;
  if (!trace_cell_nodes(state, c, nodes, &active_nverts)) {
    return;
  }

  const double pq = copy_device_double_at(
      cell_pq.data(), c, "Hydro2D compatible force diag copy cell_pq failed");
  const double vol = copy_device_double_at(
      state.vol.data(), c, "Hydro2D compatible force diag copy vol failed");
  const double mass = copy_device_double_at(
      state.mass.data(), c, "Hydro2D compatible force diag copy mass failed");
  double corner_sum = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    corner_sum += copy_device_double_at(
        state.corner_mass.data(),
        c * state.corner_stride + k,
        "Hydro2D compatible force diag copy corner_mass failed");
  }

  std::fprintf(stderr,
               "[compat-force-diag step=%d cell=%d] cell_pq=%.17e vol=%.17e "
               "mass=%.17e corner_mass_sum=%.17e\n",
               state.step,
               c,
               pq,
               vol,
               mass,
               corner_sum);
  for (int k = 0; k < active_nverts; ++k) {
    const int n = nodes[k];
    if (n < 0) {
      continue;
    }
    const double m_node = copy_device_double_at(
        node_mass.data(), n, "Hydro2D compatible force diag copy node_mass failed");
    const double fr = copy_device_double_at(
        force_r.data(), n, "Hydro2D compatible force diag copy force_r failed");
    const double fz = copy_device_double_at(
        force_z.data(), n, "Hydro2D compatible force diag copy force_z failed");
    const double ar = (m_node != 0.0) ? (fr / m_node)
                                     : std::numeric_limits<double>::quiet_NaN();
    const double az = (m_node != 0.0) ? (fz / m_node)
                                     : std::numeric_limits<double>::quiet_NaN();
    std::fprintf(stderr,
                 "[compat-force-diag step=%d cell=%d node=%d] "
                 "m_node=%.17e F_r=%.17e F_z=%.17e a_r=%.17e a_z=%.17e\n",
                 state.step,
                 c,
                 n,
                 m_node,
                 fr,
                 fz,
                 ar,
                 az);
    if (state.mesh.topo.multiblock.has_value() &&
        state.mesh.multiblock_reverse_csr_node_offsets.size() >
            static_cast<std::size_t>(n + 1) &&
        state.mesh.multiblock_reverse_csr_node_cells.size() ==
            state.mesh.multiblock_reverse_csr_node_corners.size()) {
      const int off = copy_device_int_at(
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          n,
          "Hydro2D compatible force diag copy reverse CSR offset failed");
      const int end = copy_device_int_at(
          state.mesh.multiblock_reverse_csr_node_offsets.data(),
          n + 1,
          "Hydro2D compatible force diag copy reverse CSR offset end failed");
      const int incident_count =
          static_cast<int>(state.mesh.multiblock_reverse_csr_node_cells.size());
      const auto& mb = *state.mesh.topo.multiblock;
      for (int p = off; p < end && p < incident_count; ++p) {
        const int incident_cell = copy_device_int_at(
            state.mesh.multiblock_reverse_csr_node_cells.data(),
            p,
            "Hydro2D compatible force diag copy reverse CSR cell failed");
        const int corner = copy_device_int_at(
            state.mesh.multiblock_reverse_csr_node_corners.data(),
            p,
            "Hydro2D compatible force diag copy reverse CSR corner failed");
        const int idx = incident_cell * state.corner_stride + corner;
        const int block_id =
            (incident_cell >= 0 &&
             incident_cell < static_cast<int>(mb.cell_block_id.size()))
                ? mb.cell_block_id[static_cast<std::size_t>(incident_cell)]
                : -1;
        const double cell_p = copy_device_double_at(
            cell_pq.data(),
            incident_cell,
            "Hydro2D compatible force diag copy incident cell_p failed");
        const double svec_z =
            (idx >= 0 && idx < static_cast<int>(state.mesh.cell_Svec_z.size()))
                ? state.mesh.cell_Svec_z[static_cast<std::size_t>(idx)]
                : 0.0;
        const double corner_p_z = copy_device_double_at(
            state.corner_force_p_z.data(),
            idx,
            "Hydro2D compatible force diag copy incident corner_force_p_z failed");
        const double corner_sub_z = copy_device_double_at(
            state.corner_force_sub_z.data(),
            idx,
            "Hydro2D compatible force diag copy incident corner_force_sub_z failed");
        std::fprintf(stderr,
                     "[compat-force-incident step=%d node=%d cell=%d corner=%d] "
                     "block_id=%d cell_p=%.17e Svec_z=%.17e p_Svec_z=%.17e "
                     "corner_p_z=%.17e corner_sub_z=%.17e\n",
                     state.step,
                     n,
                     incident_cell,
                     corner,
                     block_id,
                     cell_p,
                     svec_z,
                     cell_p * svec_z,
                     corner_p_z,
                     corner_sub_z);
      }
    }
  }
  std::fflush(stderr);
}

enum class CompatibleForceDumpStage {
  Pressure,
  Subzonal,
  Av,
};

struct CompatibleForceDumpCornerRef {
  int cell = -1;
  int index = -1;
};

struct CompatibleForceDumpEdgeRef {
  int edge = -1;
  int cell_a = -1;
  int cell_b = -1;
  int node0 = -1;
  int node1 = -1;
};

struct CompatibleForceDumpNode {
  int node = -1;
  std::vector<CompatibleForceDumpCornerRef> corners;
  std::vector<CompatibleForceDumpEdgeRef> edges;
};

struct CompatibleForceDumpContext {
  std::vector<CompatibleForceDumpNode> nodes;
  bool topology_ready = false;
  int n_cells = 0;
  int n_nodes = 0;
  int corner_stride = 0;
};

const std::vector<int>& compatible_force_dump_node_ids() {
  static const std::vector<int> nodes = [] {
    std::vector<int> parsed;
    const char* const raw = std::getenv("TENRYU_I1B_FORCE_DUMP");
    if (raw == nullptr || raw[0] == '\0') {
      return parsed;
    }
    const char* token = raw;
    while (*token != '\0') {
      const char* token_end = token;
      while (*token_end != '\0' && *token_end != ',') {
        ++token_end;
      }
      char* parsed_end = nullptr;
      const long value = std::strtol(token, &parsed_end, 10);
      const char* trailing = parsed_end;
      while (trailing < token_end &&
             std::isspace(static_cast<unsigned char>(*trailing)) != 0) {
        ++trailing;
      }
      if (parsed_end != token && trailing == token_end && value >= 0 &&
          value <= std::numeric_limits<int>::max()) {
        const int node = static_cast<int>(value);
        if (std::find(parsed.begin(), parsed.end(), node) == parsed.end()) {
          parsed.push_back(node);
        }
      }
      token = (*token_end == ',') ? token_end + 1 : token_end;
    }
    return parsed;
  }();
  return nodes;
}

CompatibleForceDumpContext& compatible_force_dump_context() {
  static CompatibleForceDumpContext context = [] {
    CompatibleForceDumpContext out;
    for (const int node : compatible_force_dump_node_ids()) {
      CompatibleForceDumpNode watch;
      watch.node = node;
      out.nodes.push_back(std::move(watch));
    }
    return out;
  }();
  return context;
}

void initialize_compatible_force_dump_topology(
    CompatibleForceDumpContext& context,
    const core::State& state) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "TENRYU_I1B_FORCE_DUMP requires a multiblock mesh");
  context.n_cells = static_cast<int>(state.rho.size());
  context.n_nodes = static_cast<int>(state.x_r.size());
  context.corner_stride = state.corner_stride;
  TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                    static_cast<std::size_t>(context.n_nodes) + 1U,
                "TENRYU_I1B_FORCE_DUMP requires reverse CSR node offsets");
  TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                    state.mesh.multiblock_reverse_csr_node_corners.size(),
                "TENRYU_I1B_FORCE_DUMP reverse CSR payload mismatch");

  std::vector<int> reverse_offsets;
  std::vector<int> reverse_cells;
  std::vector<int> reverse_corners;
  state.mesh.multiblock_reverse_csr_node_offsets.copy_to_host(reverse_offsets);
  state.mesh.multiblock_reverse_csr_node_cells.copy_to_host(reverse_cells);
  state.mesh.multiblock_reverse_csr_node_corners.copy_to_host(reverse_corners);
  for (CompatibleForceDumpNode& watch : context.nodes) {
    TENRYU_ASSERT(watch.node >= 0 && watch.node < context.n_nodes,
                  "TENRYU_I1B_FORCE_DUMP watched node is out of range");
    const int off = reverse_offsets[static_cast<std::size_t>(watch.node)];
    const int end = reverse_offsets[static_cast<std::size_t>(watch.node + 1)];
    TENRYU_ASSERT(off >= 0 && end >= off &&
                      static_cast<std::size_t>(end) <= reverse_cells.size(),
                  "TENRYU_I1B_FORCE_DUMP invalid reverse CSR node slice");
    for (int p = off; p < end; ++p) {
      const int cell = reverse_cells[static_cast<std::size_t>(p)];
      const int corner = reverse_corners[static_cast<std::size_t>(p)];
      TENRYU_ASSERT(cell >= 0 && cell < context.n_cells && corner >= 0 &&
                        corner < context.corner_stride,
                    "TENRYU_I1B_FORCE_DUMP invalid reverse CSR entry");
      watch.corners.push_back(CompatibleForceDumpCornerRef{
          cell, cell * context.corner_stride + corner});
    }
  }

  const auto& mb = *state.mesh.topo.multiblock;
  const auto add_edge = [&](const int edge,
                            const mesh::UniqueOrientedFace& face,
                            const int cell_b) {
    int node0 = -1;
    int node1 = -1;
    TENRYU_ASSERT(axiswall_multiblock_face_nodes(
                      state, face, &node0, &node1),
                  "TENRYU_I1B_FORCE_DUMP invalid multiblock face");
    for (CompatibleForceDumpNode& watch : context.nodes) {
      if (watch.node == node0 || watch.node == node1) {
        watch.edges.push_back(CompatibleForceDumpEdgeRef{
            edge, face.cell_a, cell_b, node0, node1});
      }
    }
  };
  int edge = 0;
  for (const mesh::UniqueOrientedFace& face : mb.unique_internal_faces) {
    add_edge(edge, face, face.cell_b);
    ++edge;
  }
  for (const mesh::UniqueOrientedFace& face : mb.boundary_faces) {
    add_edge(edge, face, -1);
    ++edge;
  }
  context.topology_ready = true;
}

bool compatible_force_dump_cell_active(
    const std::vector<std::int8_t>& hydro_active,
    const int cell) {
  return cell >= 0 &&
         (hydro_active.empty() ||
          (static_cast<std::size_t>(cell) < hydro_active.size() &&
           hydro_active[static_cast<std::size_t>(cell)] != 0));
}

void compatible_force_dump_edge_lift_alpha(const core::Config& cfg,
                                           const double r0,
                                           const double r1,
                                           double* alpha0,
                                           double* alpha1) {
  *alpha0 = 1.0;
  *alpha1 = 1.0;
  if (!cfg.numerics.hydro.csw_rz_lift_enabled ||
      !cfg.numerics.hydro.aw_compatible_force_work || !(r0 > 0.0) ||
      !(r1 > 0.0)) {
    return;
  }
  const double rbar = 0.5 * (r0 + r1);
  if (!(rbar <= cfg.numerics.hydro.csw_rz_lift_guard_ratio *
                    std::fmin(r0, r1))) {
    return;
  }
  *alpha0 = rbar / r0;
  *alpha1 = rbar / r1;
}

std::vector<double> compatible_force_dump_corner_densities(
    const core::State& state,
    const core::Config& cfg,
    const int cell) {
  int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  int active_nverts = mesh::kMeshTopoCellStorageSlots;
  TENRYU_ASSERT(trace_cell_nodes(state, cell, nodes, &active_nverts) &&
                    (active_nverts == 3 || active_nverts == 4 ||
                     active_nverts == 5),
                "TENRYU_I1B_FORCE_DUMP invalid adjacent cell topology");

  double r[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double z[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  double corner_mass[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  for (int k = 0; k < active_nverts; ++k) {
    r[k] = copy_device_double_at(
        state.x_r.data(), nodes[k],
        "TENRYU_I1B_FORCE_DUMP copy corner node r failed");
    z[k] = copy_device_double_at(
        state.x_z.data(), nodes[k],
        "TENRYU_I1B_FORCE_DUMP copy corner node z failed");
    corner_mass[k] = copy_device_double_at(
        state.corner_mass.data(), cell * state.corner_stride + k,
        "TENRYU_I1B_FORCE_DUMP copy corner mass failed");
  }

  double corner_volume[mesh::kMeshTopoCellStorageSlotsMaxGeneral] = {};
  if (active_nverts == 3) {
    if (mesh::mesh_topo_polar_tier_family(cfg.mesh)) {
      rz::compute_triangle_corner_volumes_equal_planar_area(
          r[0], z[0], r[1], z[1], r[2], z[2], corner_volume);
    } else {
      const double rc = (r[0] + r[1] + r[2]) / 3.0;
      const double zc = (z[0] + z[1] + z[2]) / 3.0;
      for (int k = 0; k < 3; ++k) {
        const int kp = (k + 1) % 3;
        const int km = (k + 2) % 3;
        const double r_sub[4] = {
            r[k], 0.5 * (r[k] + r[kp]), rc, 0.5 * (r[km] + r[k])};
        const double z_sub[4] = {
            z[k], 0.5 * (z[k] + z[kp]), zc, 0.5 * (z[km] + z[k])};
        corner_volume[k] =
            std::fabs(rz::rz_polygon_volume_exact(r_sub, z_sub, 4));
      }
    }
  } else if (active_nverts == 5) {
    PentagonPoint x[5];
    for (int k = 0; k < 5; ++k) {
      x[k] = {r[k], z[k]};
    }
    pentagon_corner_rz_volumes(x, corner_volume);
  } else {
    rz::compute_quad_corner_volumes_exact_subpolygon(
        r[0], z[0], r[1], z[1], r[2], z[2], r[3], z[3], corner_volume);
  }

  std::vector<double> density(static_cast<std::size_t>(active_nverts), 0.0);
  int origin_corner = -1;
  if (active_nverts == 3 &&
      mesh::mesh_topo_polar_tier_family(cfg.mesh)) {
    for (int k = 0; k < 3; ++k) {
      if (r[k] == 0.0 && z[k] == 0.0) {
        origin_corner = k;
        break;
      }
    }
  }
  for (int k = 0; k < active_nverts; ++k) {
    if (k == origin_corner) {
      continue;
    }
    density[static_cast<std::size_t>(k)] =
        (active_nverts == 5 ? corner_mass[k] : std::fmax(corner_mass[k], 0.0)) /
        corner_volume[k];
  }
  return density;
}

void dump_compatible_stage_forces(const core::State& state,
                                  const core::Config& cfg,
                                  const std::int8_t* d_hydro_active,
                                  const CompatibleForceDumpStage stage) {
  CompatibleForceDumpContext& context = compatible_force_dump_context();
  if (context.nodes.empty()) {
    return;
  }
  cuda_check(cudaDeviceSynchronize(),
             "TENRYU_I1B_FORCE_DUMP stage synchronization failed");
  if (!context.topology_ready) {
    initialize_compatible_force_dump_topology(context, state);
  }
  TENRYU_ASSERT(context.n_cells == static_cast<int>(state.rho.size()) &&
                    context.n_nodes == static_cast<int>(state.x_r.size()) &&
                    context.corner_stride == state.corner_stride,
                "TENRYU_I1B_FORCE_DUMP topology changed after initialization");

  std::vector<std::int8_t> hydro_active;
  if (d_hydro_active != nullptr && context.n_cells > 0) {
    hydro_active.resize(static_cast<std::size_t>(context.n_cells), 0);
    cuda_check(cudaMemcpy(hydro_active.data(),
                          d_hydro_active,
                          hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyDeviceToHost),
               "TENRYU_I1B_FORCE_DUMP copy hydro_active failed");
  }

  std::vector<double> first_r;
  std::vector<double> first_z;
  std::vector<double> second_r;
  std::vector<double> second_z;
  std::vector<double> av_r;
  std::vector<double> av_z;
  std::vector<double> node_r;
  const bool corner_av =
      cfg.numerics.hydro.av_model == core::AvModel::MimeticTensorV1 ||
      cfg.numerics.hydro.av_model == core::AvModel::ScalarVnrLegacy;
  state.corner_force_p_r.copy_to_host(first_r);
  state.corner_force_p_z.copy_to_host(first_z);
  if (stage != CompatibleForceDumpStage::Pressure) {
    state.corner_force_sub_r.copy_to_host(second_r);
    state.corner_force_sub_z.copy_to_host(second_z);
  }
  if (stage == CompatibleForceDumpStage::Av && corner_av) {
    state.corner_force_q_r.copy_to_host(av_r);
    state.corner_force_q_z.copy_to_host(av_z);
  } else if (stage == CompatibleForceDumpStage::Av) {
    state.edge_force_av_r.copy_to_host(av_r);
    state.edge_force_av_z.copy_to_host(av_z);
    if (cfg.numerics.hydro.csw_rz_lift_enabled &&
        cfg.numerics.hydro.aw_compatible_force_work) {
      state.x_r.copy_to_host(node_r);
    }
  }

  const char* const stage_name =
      stage == CompatibleForceDumpStage::Pressure
          ? "pressure"
          : (stage == CompatibleForceDumpStage::Subzonal ? "subzonal" : "av");
  for (const CompatibleForceDumpNode& watch : context.nodes) {
    double force_r = 0.0;
    double force_z = 0.0;
    for (const CompatibleForceDumpCornerRef& corner : watch.corners) {
      if (!compatible_force_dump_cell_active(hydro_active, corner.cell)) {
        continue;
      }
      if (stage == CompatibleForceDumpStage::Pressure) {
        force_r += first_r[static_cast<std::size_t>(corner.index)];
        force_z += first_z[static_cast<std::size_t>(corner.index)];
      } else {
        force_r += first_r[static_cast<std::size_t>(corner.index)] +
                   second_r[static_cast<std::size_t>(corner.index)];
        force_z += first_z[static_cast<std::size_t>(corner.index)] +
                   second_z[static_cast<std::size_t>(corner.index)];
      }
    }
    if (stage == CompatibleForceDumpStage::Av && corner_av) {
      double force_av_r = 0.0;
      double force_av_z = 0.0;
      for (const CompatibleForceDumpCornerRef& corner : watch.corners) {
        if (!compatible_force_dump_cell_active(hydro_active, corner.cell)) {
          continue;
        }
        force_av_r += av_r[static_cast<std::size_t>(corner.index)];
        force_av_z += av_z[static_cast<std::size_t>(corner.index)];
      }
      force_r += force_av_r;
      force_z += force_av_z;
    } else if (stage == CompatibleForceDumpStage::Av) {
      for (const CompatibleForceDumpEdgeRef& edge : watch.edges) {
        if (!compatible_force_dump_cell_active(hydro_active, edge.cell_a) &&
            !compatible_force_dump_cell_active(hydro_active, edge.cell_b)) {
          continue;
        }
        double alpha0 = 1.0;
        double alpha1 = 1.0;
        if (!node_r.empty()) {
          compatible_force_dump_edge_lift_alpha(
              cfg,
              node_r[static_cast<std::size_t>(edge.node0)],
              node_r[static_cast<std::size_t>(edge.node1)],
              &alpha0,
              &alpha1);
        }
        const double weight = watch.node == edge.node0 ? -alpha0 : alpha1;
        force_r += weight * av_r[static_cast<std::size_t>(edge.edge)];
        force_z += weight * av_z[static_cast<std::size_t>(edge.edge)];
      }
    }
    std::fprintf(stderr,
                 "[force-dump] step=%d stage=%s node=%d F_r=%.17e F_z=%.17e\n",
                 state.step,
                 stage_name,
                 watch.node,
                 force_r,
                 force_z);
  }
  if (stage == CompatibleForceDumpStage::Subzonal) {
    std::fprintf(
        stderr,
        "[force-dump] step=%d corner_rho_spread_max=%.17e cell=%d corner=%d\n",
        state.step,
        state.max_corner_density_spread_step,
        state.max_corner_density_spread_cell_id,
        state.max_corner_density_spread_corner_id);
    // Goal: confirm/refute a ~1e-4 uniform-density mismatch between cached
    // initialization corner masses and runtime corner volumes at junction rings.
    for (const CompatibleForceDumpNode& watch : context.nodes) {
      for (const CompatibleForceDumpCornerRef& corner : watch.corners) {
        const double rho_zonal = copy_device_double_at(
            state.rho.data(), corner.cell,
            "TENRYU_I1B_FORCE_DUMP copy zonal density failed");
        const std::vector<double> corner_density =
            compatible_force_dump_corner_densities(state, cfg, corner.cell);
        std::fprintf(stderr,
                     "[force-dump] step=%d node=%d cell=%d rho_zonal=%.17e "
                     "corner_rho=",
                     state.step,
                     watch.node,
                     corner.cell,
                     rho_zonal);
        for (std::size_t k = 0; k < corner_density.size(); ++k) {
          std::fprintf(stderr, "%s%.17e", k == 0U ? "" : ",",
                       corner_density[k]);
        }
        std::fprintf(stderr, "\n");
      }
    }
  }
  std::fflush(stderr);
}

void emit_drive_layer_diag(const core::State& state,
                           const double eval_time,
                           const core::NodeField1D& force_r,
                           const core::NodeField1D& force_z) {
  if (state.mesh.topo.multiblock.has_value()) {
    std::ostringstream os;
    os << "[drive-layer] step=" << state.step
       << " t=" << format_scientific(eval_time)
       << " skipped=multiblock";
    core::log_info(os.str());
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_cells =
      static_cast<std::size_t>(nr) * static_cast<std::size_t>(nz);
  const std::size_t n_nodes =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  if (nr < 3 || nz < 1 || state.rho.size() != n_cells ||
      state.work_p_per_cell.size() != n_cells ||
      state.work_sub_per_cell.size() != n_cells ||
      state.work_av_per_cell.size() != n_cells ||
      state.v_r.size() != n_nodes || state.v_z.size() != n_nodes ||
      state.x_r.size() != n_nodes || state.x_z.size() != n_nodes ||
      force_r.size() != n_nodes || force_z.size() != n_nodes) {
    std::ostringstream os;
    os << "[drive-layer] step=" << state.step
       << " t=" << format_scientific(eval_time)
       << " skipped=invalid-structured-layout";
    core::log_info(os.str());
    return;
  }

  sync_kernel("Hydro2D drive-layer diagnostic pre-copy failed");
  const std::vector<double> rho = copy_field_to_vector(state.rho);
  const std::vector<double> work_p =
      copy_field_to_vector(state.work_p_per_cell);
  const std::vector<double> work_sub =
      copy_field_to_vector(state.work_sub_per_cell);
  const std::vector<double> work_av =
      copy_field_to_vector(state.work_av_per_cell);
  const std::vector<double> velocity_r = copy_field_to_vector(state.v_r);
  const std::vector<double> velocity_z = copy_field_to_vector(state.v_z);
  const std::vector<double> x_r = copy_field_to_vector(state.x_r);
  const std::vector<double> x_z = copy_field_to_vector(state.x_z);
  const std::vector<double> total_force_r = copy_field_to_vector(force_r);
  const std::vector<double> total_force_z = copy_field_to_vector(force_z);
  const std::vector<double> planar_mass =
      state.node_planar_mass.size() == n_nodes
          ? copy_field_to_vector(state.node_planar_mass)
          : std::vector<double>{};

  const int outer_row = nr - 1;
  int j_star = 0;
  double max_abs_work_av =
      std::abs(work_av[static_cast<std::size_t>(outer_row * nz)]);
  long double sum_wp = 0.0L;
  long double sum_ws = 0.0L;
  long double sum_wav = 0.0L;
  long double sum_abs_wav = 0.0L;
  for (int j = 0; j < nz; ++j) {
    const std::size_t c =
        static_cast<std::size_t>(outer_row * nz + j);
    const double abs_work_av = std::abs(work_av[c]);
    if (abs_work_av > max_abs_work_av) {
      max_abs_work_av = abs_work_av;
      j_star = j;
    }
    sum_wp += static_cast<long double>(work_p[c]);
    sum_ws += static_cast<long double>(work_sub[c]);
    sum_wav += static_cast<long double>(work_av[c]);
    sum_abs_wav += static_cast<long double>(abs_work_av);
  }

  const std::array<int, 4> candidates = {j_star, 0, nz / 2, nz - 1};
  std::vector<int> columns;
  columns.reserve(candidates.size());
  for (const int j : candidates) {
    if (std::find(columns.begin(), columns.end(), j) == columns.end()) {
      columns.push_back(j);
    }
  }

  const int node_stride = nz + 1;
  for (const int j : columns) {
    for (int i = nr - 3; i < nr; ++i) {
      const std::size_t c = static_cast<std::size_t>(i * nz + j);
      const std::size_t node_lo =
          static_cast<std::size_t>(i * node_stride + j);
      const std::size_t node_hi =
          static_cast<std::size_t>((i + 1) * node_stride + j);
      const double dr = std::hypot(x_r[node_hi] - x_r[node_lo],
                                   x_z[node_hi] - x_z[node_lo]);
      std::ostringstream os;
      os << "[drive-layer] step=" << state.step
         << " t=" << format_scientific(eval_time)
         << " i=" << i
         << " j=" << j
         << " rho=" << format_scientific(rho[c])
         << " wp=" << format_scientific(work_p[c])
         << " ws=" << format_scientific(work_sub[c])
         << " wav=" << format_scientific(work_av[c])
         << " ur_lo=" << format_scientific(velocity_r[node_lo])
         << " ur_hi=" << format_scientific(velocity_r[node_hi])
         << " uz_lo=" << format_scientific(velocity_z[node_lo])
         << " uz_hi=" << format_scientific(velocity_z[node_hi])
         << " dr=" << format_scientific(dr);
      core::log_info(os.str());
    }
    const std::size_t boundary_node =
        static_cast<std::size_t>(nr * node_stride + j);
    std::ostringstream boundary;
    boundary << "[drive-layer-bnode] step=" << state.step
             << " j=" << j
             << " ur=" << format_scientific(velocity_r[boundary_node])
             << " uz=" << format_scientific(velocity_z[boundary_node])
             << " r=" << format_scientific(x_r[boundary_node])
             << " z=" << format_scientific(x_z[boundary_node]);
    core::log_info(boundary.str());
  }

  const double mean_abs_wav =
      static_cast<double>(sum_abs_wav / static_cast<long double>(nz));
  const double max_row_ratio_wav =
      mean_abs_wav > 0.0 ? max_abs_work_av / mean_abs_wav : 0.0;
  std::ostringstream summary;
  summary << "[drive-layer-sum] step=" << state.step
          << " row=" << outer_row
          << " sum_wp=" << format_scientific(static_cast<double>(sum_wp))
          << " sum_ws=" << format_scientific(static_cast<double>(sum_ws))
          << " sum_wav=" << format_scientific(static_cast<double>(sum_wav))
          << " max_row_ratio_wav="
          << format_scientific(max_row_ratio_wav);
  core::log_info(summary.str());

  if (!planar_mass.empty() && nz >= 2) {
    const std::array<int, 2> pole_j = {0, nz};
    const std::array<int, 2> peer_j = {1, nz - 1};
    const std::array<const char*, 2> pole_name = {"theta0", "theta_pi"};
    for (int pole = 0; pole < 2; ++pole) {
      const std::size_t p =
          static_cast<std::size_t>(nr * node_stride + pole_j[pole]);
      const std::size_t q =
          static_cast<std::size_t>(nr * node_stride + peer_j[pole]);
      if (state.mesh.topo.node_flags.size() != n_nodes ||
          (state.mesh.topo.node_flags[p] & mesh::NODE_POLE_AXIS) == 0U) {
        continue;
      }
      const double mass_p = planar_mass[p];
      const double mass_q = planar_mass[q];
      const double eta_m =
          mass_p != 0.0
              ? mass_q / (2.0 * mass_p)
              : std::numeric_limits<double>::quiet_NaN();
      std::ostringstream mass_diag;
      mass_diag << "[drive-layer-axis-mass] step=" << state.step
                << " pole=" << pole_name[pole]
                << " P=" << p
                << " Q=" << q
                << " M_A_P=" << format_scientific(mass_p)
                << " M_A_Q=" << format_scientific(mass_q)
                << " eta_M=" << format_scientific(eta_m);
      core::log_info(mass_diag.str());
    }
  }

  (void)total_force_r;
  (void)total_force_z;
}

void assert_compatible_pre_accel_inputs(const double* force_r,
                                        const double* force_z,
                                        const double* node_mass,
                                        const std::uint8_t* d_node_active,
                                        const std::uint8_t* d_node_flags,
                                        const int n_nodes,
                                        const core::State::LaunchWindow& nw,
                                        const double mass_floor) {
  if (n_nodes <= 0) {
    return;
  }
  int* d_bad_node = nullptr;
  int* d_bad_code = nullptr;
  d_bad_node = static_cast<int*>(
      core::device_scratch_acquire("h2d:preaccel_bad_node", sizeof(int)));
  d_bad_code = static_cast<int*>(
      core::device_scratch_acquire("h2d:preaccel_bad_code", sizeof(int)));
  const int minus_one = -1;
  const int zero = 0;
  cuda_check(cudaMemcpy(d_bad_node, &minus_one, sizeof(int), cudaMemcpyHostToDevice),
             "Hydro2D compatible pre-accel init bad_node failed");
  cuda_check(cudaMemcpy(d_bad_code, &zero, sizeof(int), cudaMemcpyHostToDevice),
             "Hydro2D compatible pre-accel init bad_code failed");

  validate_compatible_pre_accel_kernel<<<nw.blocks(), 256>>>(
      d_bad_node,
      d_bad_code,
      force_r,
      force_z,
      node_mass,
      d_node_active,
      d_node_flags,
      nw.begin,
      nw.end,
      n_nodes,
      mass_floor);
  cuda_check(cudaGetLastError(),
             "Hydro2D compatible pre-accel validation launch failed");
  cuda_check(core::debug_kernel_sync(),
             "Hydro2D compatible pre-accel validation failed");

  int bad_node = -1;
  int bad_code = 0;
  cuda_check(cudaMemcpy(&bad_node, d_bad_node, sizeof(int), cudaMemcpyDeviceToHost),
             "Hydro2D compatible pre-accel copy bad_node failed");
  cuda_check(cudaMemcpy(&bad_code, d_bad_code, sizeof(int), cudaMemcpyDeviceToHost),
             "Hydro2D compatible pre-accel copy bad_code failed");
  if (bad_node >= 0) {
    std::ostringstream os;
    os << "Hydro2D compatible pre-accel invalid input at node=" << bad_node
       << " code=" << bad_code << " mass_floor=" << mass_floor;
    TENRYU_ASSERT(false, os.str());
  }
}

void capture_and_zero_polar_tier_origin_force(
    core::State& state,
    const core::Config& cfg,
    double* const force_r,
    double* const force_z) {
  // Deliberately scheme-exact: the hybrid has no CENTER_FAN origin to capture.
  if (cfg.mesh.topology_scheme !=
      core::TopologyScheme::MULTIBLOCK_POLAR_TIER) {
    return;
  }
  // The installed ReALE tessellation has no NODE_CENTER origin to capture or pin.
  if (state.mesh.topo.general_polygonal) {
    return;
  }

  TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(state.mesh.topo.n_nodes),
                "polar-tier topology requires node flags for every node");
  int origin_node = -1;
  for (int n = 0; n < state.mesh.topo.n_nodes; ++n) {
    const std::uint8_t flags =
        state.mesh.topo.node_flags[static_cast<std::size_t>(n)];
    if ((flags & mesh::NODE_CENTER) == 0U) {
      continue;
    }
    TENRYU_ASSERT(origin_node < 0,
                  "polar-tier topology must have exactly one NODE_CENTER");
    origin_node = n;
  }
  TENRYU_ASSERT(origin_node >= 0,
                "polar-tier topology requires a NODE_CENTER origin");

  auto& diag = state.polar_tier_origin_force_diag;
  diag.reset();
  diag.node = origin_node;
  cuda_check(cudaMemcpy(&diag.raw_force_r,
                        force_r + origin_node,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D polar-tier origin force_r capture failed");
  cuda_check(cudaMemcpy(&diag.raw_force_z,
                        force_z + origin_node,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D polar-tier origin force_z capture failed");
  diag.valid = true;

  cuda_check(cudaMemset(force_r + origin_node, 0, sizeof(double)),
             "Hydro2D polar-tier origin force_r zero failed");
  cuda_check(cudaMemset(force_z + origin_node, 0, sizeof(double)),
             "Hydro2D polar-tier origin force_z zero failed");
}

void compute_acceleration_2d(core::NodeField1D& accel_r,
                             core::NodeField1D& accel_z,
                             core::State& state,
                             const core::Config& cfg,
                             const double eval_time,
                             const MeshDeviceCache& mesh_cache,
                             const core::CellField1D& cell_pq,
                             const std::uint8_t* d_node_active,
                             const std::uint8_t* d_node_flags,
                             const core::NodeField1D& node_mass,
                             const std::int8_t* d_hydro_active,
                             const HydroEOSContext* eos_ctx = nullptr,
                             const bool include_boundary_pressure = true,
                             const bool trace_pressure_bc = false,
                             const bool use_compatible_force_buffers = true,
                             const core::CellField1D* cs_for_av = nullptr,
                             const bool include_compatible_av = true,
                             core::NodeField1D* audit_force_r = nullptr,
                             core::NodeField1D* audit_force_z = nullptr,
                             const double central_pseudo_core_impulse_dt = 0.0,
                             const bool capture_pole_axis_diag = false,
                             core::NodeField1D* pole_axis_bbsw_true_force_z =
                                 nullptr,
                             const bool include_plasma_viscosity = true,
                             const bool aw_axis_slave_theta0_active = false,
                             const bool aw_axis_slave_theta_pi_active = false,
                             const SubzonalPressureProjectionDebugBuffers*
                                 subzonal_projection_debug = nullptr,
                             double* r_momentum_source_force_sum = nullptr,
                             const bool force_midpoint_compatible = false,
                             const core::CellField1D* midpoint_scalar_q = nullptr,
                             const char* psc_diag_tag = "untagged") {
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);

  core::NodeField1D force_r{"h2d:accel:force_r"};
  core::NodeField1D force_z{"h2d:accel:force_z"};
  force_r.reset(state.x_r.size());
  force_z.reset(state.x_r.size());

  compatible::ensure_compatible_force_work_buffers(
      state, cfg, force_midpoint_compatible);
  compatible::zero_compatible_force_work_buffers(state);
  const bool button_center_enabled =
      state.mesh.button_center && state.mesh.button_center->enabled;
  const bool compatible_mode =
      use_compatible_force_buffers && !button_center_enabled &&
      (force_midpoint_compatible ||
       compatible::compatible_force_work_enabled(cfg));
  if (compatible_mode) {
    const bool force_dump_enabled =
        !compatible_force_dump_node_ids().empty();
    compatible::launch_assemble_pressure_corner_forces_2d(
        state, cell_pq, mesh_cache.Svec_r, mesh_cache.Svec_z, d_hydro_active,
        cfg);
    if (force_dump_enabled) {
      dump_compatible_stage_forces(
          state, cfg, d_hydro_active, CompatibleForceDumpStage::Pressure);
    }
    if (force_midpoint_compatible &&
        cfg.numerics.hydro.av_model == core::AvModel::ScalarVnrLegacy) {
      TENRYU_ASSERT(midpoint_scalar_q != nullptr,
                    "midpoint scalar AV requires the stage Q buffer");
      compatible::launch_assemble_scalar_av_corner_forces_2d(
          state, *midpoint_scalar_q, mesh_cache.Svec_r, mesh_cache.Svec_z,
          d_hydro_active);
      if (force_dump_enabled) {
        dump_compatible_stage_forces(
            state, cfg, d_hydro_active, CompatibleForceDumpStage::Av);
      }
    }
    compute_compatible_subzonal_pressure_force_2d(
        state, cfg, eos_ctx, d_hydro_active, aw_axis_slave_theta0_active,
        aw_axis_slave_theta_pi_active, subzonal_projection_debug);
    if (force_dump_enabled) {
      dump_compatible_stage_forces(
          state, cfg, d_hydro_active, CompatibleForceDumpStage::Subzonal);
    }
    central_pseudo_core::zero_member_compatible_cell_buffers(state);
    pole_angular_derefine::zero_member_compatible_cell_buffers(state);
    compatible::launch_compute_compatible_work_2d(
        state, cfg, state.v_r.data(), state.v_z.data(), d_hydro_active);
    update_compatible_subzonal_pressure_observability_2d(state, cfg);
    if (include_compatible_av) {
      const core::CellField1D& av_cs =
          (cs_for_av != nullptr && !cs_for_av->empty()) ? *cs_for_av : state.cs;
      if (cfg.numerics.hydro.av_model ==
          core::AvModel::MimeticTensorV1) {
        compatible::launch_compute_tensor_av_2d(
            state, cfg, av_cs, state.v_r.data(), state.v_z.data(),
            d_hydro_active, aw_axis_slave_theta0_active,
            aw_axis_slave_theta_pi_active);
        central_pseudo_core::zero_member_compatible_cell_buffers(state);
        pole_angular_derefine::zero_member_compatible_cell_buffers(state);
      } else {
        compatible::launch_compute_csw_edge_av_2d(
            state, cfg, av_cs, state.v_r.data(), state.v_z.data(),
            d_hydro_active, aw_axis_slave_theta0_active,
            aw_axis_slave_theta_pi_active, node_mass.data(), state.dt);
      }
      if (force_dump_enabled &&
          cfg.numerics.hydro.av_model != core::AvModel::ScalarVnrLegacy) {
        dump_compatible_stage_forces(
            state, cfg, d_hydro_active, CompatibleForceDumpStage::Av);
      }
      central_pseudo_core::zero_member_edge_av_forces(state);
      pole_angular_derefine::zero_member_edge_av_forces(state);
    }
    if (central_pseudo_core_impulse_dt > 0.0) {
      central_pseudo_core::emit_phase_ledger(
          state, cfg, "E_after_member_force_zero", "midpoint_force");
    }
    compatible::launch_sum_compatible_forces_to_nodes_2d(
        force_r.data(), force_z.data(), d_hydro_active, state, cfg);
    if (include_compatible_av &&
        cfg.numerics.hydro.av_model ==
        core::AvModel::MimeticTensorV1) {
      compatible::launch_sum_tensor_av_corner_forces_to_nodes_2d(
          force_r.data(), force_z.data(), d_hydro_active, state);
    }
    if (force_midpoint_compatible &&
        cfg.numerics.hydro.av_model == core::AvModel::ScalarVnrLegacy) {
      compatible::launch_sum_scalar_av_corner_forces_to_nodes_2d(
          force_r.data(), force_z.data(), d_hydro_active, state);
    }
    if (include_boundary_pressure) {
      central_pseudo_core::add_boundary_pressure_force(
          state, cfg, cell_pq, force_r.data(), force_z.data(),
          central_pseudo_core_impulse_dt, psc_diag_tag);
      pole_angular_derefine::add_boundary_pressure_force(
          state, cfg, cell_pq, force_r.data(), force_z.data(),
          central_pseudo_core_impulse_dt);
    }
    if (central_pseudo_core_impulse_dt > 0.0) {
      central_pseudo_core::emit_phase_ledger(
          state, cfg, "E_after_boundary_force", "midpoint_force");
    }
    trace_compatible_force_diag(state, cfg, cell_pq, node_mass, force_r, force_z);
  } else {
    detail::launch_compute_force_from_cells_2d(
        force_r.data(), force_z.data(), cell_pq.data(), mesh_cache.Svec_r.data(),
        mesh_cache.Svec_z.data(), d_hydro_active, state, cfg);
  }

  const braginskii::Params visc_params = braginskii::params_from_config(cfg);
  if (include_plasma_viscosity && visc_params.enabled) {
    TENRYU_ASSERT(!button_center_enabled,
                  "plasma_viscosity 2D does not support button_center (v1)");
    TENRYU_ASSERT(!cfg.numerics.ale.central_pseudo_core_enabled,
                  "plasma_viscosity 2D does not support central_pseudo_core (v1)");
    TENRYU_ASSERT(!pole_angular_derefine::active(state),
                  "plasma_viscosity 2D does not support pole_angular_derefine (v1)");
    const int n_cells_visc = static_cast<int>(state.rho.size());
    braginskii::ensure_visc_buffers_2d(state, n_cells_visc,
                                       visc_params.species != 0);
    braginskii::zero_visc_buffers_2d(state);
    braginskii::Topo2D visc_topo;
    core::DeviceArray<std::uint8_t> d_visc_cell_nverts{
        "h2d:accel:visc_cell_nverts"};
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      visc_topo.multiblock = true;
      visc_topo.n_cells = n_cells_visc;
      visc_topo.n_nodes = n_nodes;
      visc_topo.cell_node_csr_offsets =
          state.mesh.multiblock_cell_node_csr_offsets.data();
      visc_topo.cell_node_csr_indices =
          state.mesh.multiblock_cell_node_csr_indices.data();
      visc_topo.rev_node_offsets =
          state.mesh.multiblock_reverse_csr_node_offsets.data();
      visc_topo.rev_node_cells =
          state.mesh.multiblock_reverse_csr_node_cells.data();
      visc_topo.rev_node_corners =
          state.mesh.multiblock_reverse_csr_node_corners.data();
      const bool has_visc_tri =
          state.mesh.cell_nverts.size() ==
              static_cast<std::size_t>(n_cells_visc) &&
          std::any_of(state.mesh.cell_nverts.begin(),
                      state.mesh.cell_nverts.end(),
                      [](const std::uint8_t nverts) {
                        return nverts == 3U;
                      });
      if (has_visc_tri) {
        d_visc_cell_nverts.reset(static_cast<std::size_t>(n_cells_visc));
        d_visc_cell_nverts.copy_from_host(state.mesh.cell_nverts);
        visc_topo.cell_nverts = d_visc_cell_nverts.data();
      } else {
        visc_topo.cell_nverts = nullptr;
      }
    } else {
      visc_topo.nr = state.mesh.topo.nr;
      visc_topo.nz = state.mesh.topo.nz;
    }
    braginskii::compute_corner_forces_2d(
        state.corner_force_visc_r.data(), state.corner_force_visc_z.data(),
        state.visc_heat_rate_per_cell.data(),
        visc_params.species != 0 ? state.visc_heat_rate_e_per_cell.data()
                                 : nullptr,
        state.x_r.data(), state.x_z.data(), state.v_r.data(),
        state.v_z.data(), state.rho.data(),
        state.Ti.empty() ? nullptr : state.Ti.data(),
        state.Te.empty() ? nullptr : state.Te.data(),
        state.A_eff.empty() ? nullptr : state.A_eff.data(),
        state.zbar.empty() ? nullptr : state.zbar.data(),
        state.vol.data(), d_hydro_active, visc_topo, visc_params);
    if (compatible_mode) {
      braginskii::compute_work_visc_2d(
          state.work_visc_per_cell.data(), state.corner_force_visc_r.data(),
          state.corner_force_visc_z.data(), state.v_r.data(),
          state.v_z.data(), d_hydro_active, visc_topo);
    }
    braginskii::add_corner_forces_to_nodes_2d(
        force_r.data(), force_z.data(), state.corner_force_visc_r.data(),
        state.corner_force_visc_z.data(), d_hydro_active, visc_topo);
  }

  const auto r_outer_type = parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  if (include_boundary_pressure && r_outer_type == Boundary2DType::PRESSURE) {
    TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                  "2D pressure boundary requires pressure_drive_1d table");
    const double p_ext = state.pressure_drive_1d->eval(eval_time);
    const auto& pcfg =
        cfg.numerics.hydro.pressure_drive_perturbation;
    PressureDrivePerturbationParams pp{};
    if (pcfg.enabled) {
      pp = core::make_pressure_drive_perturbation_params(pcfg);
    }
    const bool rz_exact_endpoint =
        state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane;
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      mesh::mesh_topo_assert_multiblock_outermost_polar_shell_outer_boundary(
          state.mesh.topo);
      const int shell_node_begin =
          mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
              state.mesh.topo);
      const int nr_shell =
          mesh::mesh_topo_multiblock_outermost_polar_shell_nr(state.mesh.topo);
      const int nz_polar =
          mesh::mesh_topo_multiblock_outermost_polar_shell_nz(state.mesh.topo);
      detail::launch_multiblock_polar_shell_pressure_forces(
          force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
          shell_node_begin, nr_shell, nz_polar, p_ext, rz_exact_endpoint,
          node_mass.data(), cfg.numerics.hydro.rz_momentum_scheme_id, pp,
          pcfg.enabled);
      if (trace_pressure_bc && mesh_trace::enabled(state, cfg)) {
        core::NodeField1D bc_force_r{"h2d:accel:bc_force_r_a"};
        core::NodeField1D bc_force_z{"h2d:accel:bc_force_z_a"};
        bc_force_r.reset(state.x_r.size());
        bc_force_z.reset(state.x_r.size());
        detail::launch_multiblock_polar_shell_pressure_forces(
            bc_force_r.data(),
            bc_force_z.data(),
            state.x_r.data(),
            state.x_z.data(),
            shell_node_begin,
            nr_shell,
            nz_polar,
            p_ext,
            rz_exact_endpoint,
            node_mass.data(),
            cfg.numerics.hydro.rz_momentum_scheme_id,
            pp,
            pcfg.enabled);
        sync_kernel("Hydro2D trace pressure BC multiblock force failed");
        mesh_trace::trace_pressure_bc(state,
                                      cfg,
                                      p_ext,
                                      nz_polar,
                                      bc_force_r,
                                      bc_force_z);
      }
    } else {
      detail::launch_r_outer_boundary_pressure_forces(
          force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
          state.mesh.topo.nr, state.mesh.topo.nz, p_ext, rz_exact_endpoint,
          cfg.numerics.hydro.rz_momentum_scheme_id, pp, pcfg.enabled);
      if (trace_pressure_bc && mesh_trace::enabled(state, cfg)) {
        core::NodeField1D bc_force_r{"h2d:accel:bc_force_r_b"};
        core::NodeField1D bc_force_z{"h2d:accel:bc_force_z_b"};
        bc_force_r.reset(state.x_r.size());
        bc_force_z.reset(state.x_r.size());
        detail::launch_r_outer_boundary_pressure_forces(
            bc_force_r.data(),
            bc_force_z.data(),
            state.x_r.data(),
            state.x_z.data(),
            state.mesh.topo.nr,
            state.mesh.topo.nz,
            p_ext,
            rz_exact_endpoint,
            cfg.numerics.hydro.rz_momentum_scheme_id,
            pp,
            pcfg.enabled);
        sync_kernel("Hydro2D trace pressure BC force failed");
        mesh_trace::trace_pressure_bc(state,
                                      cfg,
                                      p_ext,
                                      state.mesh.topo.nz,
                                      bc_force_r,
                                      bc_force_z);
      }
    }
  } else if (include_boundary_pressure &&
             r_outer_type == Boundary2DType::REFLECT &&
             !mesh::mesh_topo_is_multiblock(cfg.mesh) &&
             state.mesh.logical ==
                 mesh::LogicalMesh2D::SphericalPolarHalfplane) {
    core::CellField1D midpoint_mirror_pq{
        "h2d:accel:midpoint_mirror_pq"};
    const double* mirror_pq = cell_pq.data();
    if (force_midpoint_compatible &&
        cfg.numerics.hydro.av_model == core::AvModel::ScalarVnrLegacy) {
      TENRYU_ASSERT(midpoint_scalar_q != nullptr,
                    "midpoint mirror force requires the stage Q buffer");
      midpoint_mirror_pq.reset(state.rho.size());
      const int blocks_cells =
          (static_cast<int>(state.rho.size()) + 255) / 256;
      build_cell_pressure_plus_q_kernel<<<blocks_cells, 256>>>(
          midpoint_mirror_pq.data(), cell_pq.data(),
          midpoint_scalar_q->data(), d_hydro_active,
          static_cast<int>(state.rho.size()));
      mirror_pq = midpoint_mirror_pq.data();
    }
    // v1 mirror closure is single-block wedge only; multiblock reflect keeps its node-flags treatment.
    detail::launch_r_outer_boundary_mirror_forces(
        force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
        mirror_pq, state.mesh.topo.nr, state.mesh.topo.nz, true,
        cfg.numerics.hydro.rz_momentum_scheme_id);
  }

  static const int drive_layer_diag_every = [] {
    const char* const raw = std::getenv("TENRYU_DRIVE_LAYER_DIAG");
    const int value =
        raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
    return value > 0 ? value : 0;
  }();
  if (drive_layer_diag_every > 0 && compatible_mode &&
      state.step % drive_layer_diag_every == 0) {
    static int last_drive_layer_diag_step =
        std::numeric_limits<int>::min();
    if (last_drive_layer_diag_step != state.step) {
      last_drive_layer_diag_step = state.step;
      emit_drive_layer_diag(state, eval_time, force_r, force_z);
    }
  }

  capture_and_zero_polar_tier_origin_force(
      state, cfg, force_r.data(), force_z.data());

  if (compatible_mode) {
    constexpr double kCompatibleNodeMassFloor = 1.0e-300;
    assert_compatible_pre_accel_inputs(force_r.data(),
                                       force_z.data(),
                                       node_mass.data(),
                                       d_node_active,
                                       d_node_flags,
                                       n_nodes,
                                       nw,
                                       kCompatibleNodeMassFloor);
  }
  if (audit_force_r != nullptr && audit_force_z != nullptr) {
    audit_force_r->reset(state.x_r.size());
    audit_force_z->reset(state.x_r.size());
    cuda_check(cudaMemcpy(audit_force_r->data(),
                          force_r.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D cap-energy-audit copy force_r failed");
    cuda_check(cudaMemcpy(audit_force_z->data(),
                          force_z.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D cap-energy-audit copy force_z failed");
  }
  if (pole_axis_bbsw_true_force_z != nullptr && pole_axis_bbsw_enabled(cfg)) {
    pole_axis_bbsw_true_force_z->reset(state.x_r.size());
    cuda_check(cudaMemcpy(pole_axis_bbsw_true_force_z->data(),
                          force_z.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D pole-axis BBSW copy true force_z failed");
  }
  if (r_momentum_source_force_sum != nullptr) {
    TENRYU_ASSERT(compatible_mode,
                  "r-momentum source audit requires compatible force path");
    *r_momentum_source_force_sum =
        cap_energy_audit_reduce_sum(force_r.data(), n_nodes);
  }

  launch_compute_accel_from_force_2d(
      accel_r.data(), accel_z.data(), force_r.data(), force_z.data(), node_mass,
      d_node_active, d_node_flags, n_nodes, state, cfg);

  emit_pole_axis_planar_diag(state,
                             cfg,
                             eval_time,
                             cell_pq,
                             node_mass,
                             force_z,
                             accel_z,
                             d_hydro_active,
                             d_node_active,
                             compatible_mode,
                             capture_pole_axis_diag);

  apply_pole_axis_bbsw_acceleration(accel_r,
                                    accel_z,
                                    state,
                                    cfg,
                                    eval_time,
                                    cell_pq,
                                    d_hydro_active,
                                    d_node_active,
                                    compatible_mode,
                                    capture_pole_axis_diag);

  launch_apply_boundary_accel_constraints(
      accel_r.data(),
      accel_z.data(),
      d_node_flags,
      state,
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_inner),
      r_outer_type,
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom),
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top));
  if (!state.mesh.topo.multiblock.has_value()) {
    const double* const resolved_node_mass =
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data();
    launch_apply_evac_contact_accel_constraints(
        state, accel_r.data(), accel_z.data(), resolved_node_mass);
  }

  sync_kernel("Hydro2D compute_acceleration_2d kernels failed");
  if (const char* adbg = std::getenv("TENRYU_H2D_DEBUG_ACCEL")) {
    // Diagnostic-only (OPEN-HYDRO-CORNER localization): raw per-rank
    // dumps of the assembled node force/mass/accel, overwritten each
    // call (run with max_steps=1 for the step-1 snapshot).
    const int nn = static_cast<int>(state.x_r.size());
    const char* dbg_rank_env = std::getenv("OMPI_COMM_WORLD_RANK");
    const int dbg_rank = (dbg_rank_env != nullptr) ? std::atoi(dbg_rank_env)
                                                   : 0;
    const auto dump_nodes = [&](const char* tag, const double* dptr) {
      std::vector<double> h(static_cast<std::size_t>(nn));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(nn) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H2D accel debug dump D2H failed");
      const std::string path = std::string(adbg) + "_" + tag + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      FILE* fp = std::fopen(path.c_str(), "wb");
      if (fp != nullptr) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    dump_nodes("fr", force_r.data());
    dump_nodes("fz", force_z.data());
    dump_nodes("nm", node_mass.data());
    dump_nodes("ar", accel_r.data());
    dump_nodes("az", accel_z.data());
    const auto dump_cells = [&](const char* tag, const double* dptr,
                                const int count) {
      std::vector<double> h(static_cast<std::size_t>(count));
      cuda_check(cudaMemcpy(h.data(), dptr,
                            static_cast<std::size_t>(count) * sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "H2D accel debug cell dump D2H failed");
      const std::string path = std::string(adbg) + "_" + tag + "_r" +
                               std::to_string(dbg_rank) + ".bin";
      FILE* fp = std::fopen(path.c_str(), "wb");
      if (fp != nullptr) {
        std::fwrite(h.data(), sizeof(double), h.size(), fp);
        std::fclose(fp);
      }
    };
    const int ncc = static_cast<int>(state.rho.size());
    dump_cells("pq", cell_pq.data(), ncc);
    dump_cells("svr", mesh_cache.Svec_r.data(), ncc * 4);
    dump_cells("svz", mesh_cache.Svec_z.data(), ncc * 4);
  }
}

void recompute_midpoint_corner_work_2d(
    core::State& state,
    const core::Config& cfg,
    const double* work_velocity_r,
    const double* work_velocity_z,
    const std::int8_t* d_hydro_active) {
  TENRYU_ASSERT(!state.mesh.topo.multiblock.has_value(),
                "midpoint_v1 corner work is structured-only");
  compatible::launch_compute_compatible_work_2d(
      state, cfg, work_velocity_r, work_velocity_z, d_hydro_active);
  update_compatible_subzonal_pressure_observability_2d(state, cfg);
  if (cfg.numerics.hydro.av_model == core::AvModel::ScalarVnrLegacy) {
    compatible::launch_compute_scalar_av_compatible_work_2d(
        state, work_velocity_r, work_velocity_z, d_hydro_active);
  }
  const braginskii::Params visc_params = braginskii::params_from_config(cfg);
  if (visc_params.enabled) {
    braginskii::Topo2D topo;
    topo.nr = state.mesh.topo.nr;
    topo.nz = state.mesh.topo.nz;
    braginskii::compute_work_visc_2d(
        state.work_visc_per_cell.data(), state.corner_force_visc_r.data(),
        state.corner_force_visc_z.data(), work_velocity_r, work_velocity_z,
        d_hydro_active, topo);
  }
  sync_kernel("Hydro2D midpoint compatible work recompute failed");
}

template <int RZ_SCHEME>
__global__ void pressure_boundary_work_dot_kernel(
    double* __restrict__ work,
    const double* __restrict__ force_r,
    const double* __restrict__ force_z,
    const double* __restrict__ velocity_r,
    const double* __restrict__ velocity_z,
    const double* __restrict__ node_mass,
    const double* __restrict__ node_planar_mass,
    const int n_begin,
    const int n_end,
    const int n_nodes) {
  const int n = n_begin + blockIdx.x * blockDim.x + threadIdx.x;
  if (n >= n_end) {
    return;
  }
  double conjugate_scale = 1.0;
  if constexpr (RZ_SCHEME == 1) {
    // AWS advances velocity with F_A / M_A, while E_total measures nodal
    // kinetic energy with M_RZ.  Therefore the force conjugate to that
    // kinetic-energy basis is (M_RZ / M_A) F_A.
    const double planar_mass = node_planar_mass[n];
    conjugate_scale = planar_mass > 0.0 ? node_mass[n] / planar_mass : 0.0;
  }
  work[n] = conjugate_scale *
            (force_r[n] * velocity_r[n] + force_z[n] * velocity_z[n]);
}

double compute_pressure_boundary_work_step_2d(
    const core::State& state,
    const core::Config& cfg,
    const double dt,
    const double eval_time,
    const core::NodeField1D& velocity_r,
    const core::NodeField1D& velocity_z,
    const core::NodeField1D& node_mass) {
  if (!(dt > 0.0)) {
    return 0.0;
  }
  const auto r_outer_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  if (r_outer_type != Boundary2DType::PRESSURE) {
    return 0.0;
  }
  TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                "2D pressure boundary requires pressure_drive_1d table");
  const int n_nodes = static_cast<int>(state.x_r.size());
  if (n_nodes <= 0) {
    return 0.0;
  }

  core::NodeField1D force_r{"h2d:pbwork:force_r"};
  core::NodeField1D force_z{"h2d:pbwork:force_z"};
  core::NodeField1D work{"h2d:pbwork:work"};
  force_r.reset(state.x_r.size());
  force_z.reset(state.x_r.size());
  work.reset(state.x_r.size());

  const double p_ext = state.pressure_drive_1d->eval(eval_time);
  const auto& pcfg =
      cfg.numerics.hydro.pressure_drive_perturbation;
  PressureDrivePerturbationParams pp{};
  if (pcfg.enabled) {
    pp = core::make_pressure_drive_perturbation_params(pcfg);
  }
  const bool rz_exact_endpoint =
      state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane;
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    mesh::mesh_topo_assert_multiblock_outermost_polar_shell_outer_boundary(
        state.mesh.topo);
    const int shell_node_begin =
        mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
            state.mesh.topo);
    const int nr_shell =
        mesh::mesh_topo_multiblock_outermost_polar_shell_nr(state.mesh.topo);
    const int nz_polar =
        mesh::mesh_topo_multiblock_outermost_polar_shell_nz(state.mesh.topo);
    detail::launch_multiblock_polar_shell_pressure_forces(
        force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
        shell_node_begin, nr_shell, nz_polar, p_ext, rz_exact_endpoint,
        node_mass.data(), cfg.numerics.hydro.rz_momentum_scheme_id, pp,
        pcfg.enabled);
  } else {
    detail::launch_r_outer_boundary_pressure_forces(
        force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
        state.mesh.topo.nr, state.mesh.topo.nz, p_ext, rz_exact_endpoint,
        cfg.numerics.hydro.rz_momentum_scheme_id, pp, pcfg.enabled);
  }

  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  switch (cfg.numerics.hydro.rz_momentum_scheme_id) {
    case 0:
      pressure_boundary_work_dot_kernel<0><<<nw.blocks(), 256>>>(
          work.data(), force_r.data(), force_z.data(), velocity_r.data(),
          velocity_z.data(), node_mass.data(), nullptr, nw.begin, nw.end,
          n_nodes);
      break;
    case 1:
      TENRYU_ASSERT(state.node_planar_mass.size() ==
                        static_cast<std::size_t>(n_nodes),
                    "Hydro2D AWS boundary work requires node planar mass");
      pressure_boundary_work_dot_kernel<1><<<nw.blocks(), 256>>>(
          work.data(), force_r.data(), force_z.data(), velocity_r.data(),
          velocity_z.data(), node_mass.data(), state.node_planar_mass.data(),
          nw.begin, nw.end, n_nodes);
      break;
    default:
      TENRYU_ASSERT(false, "Hydro2D invalid resolved RZ momentum scheme");
  }
  sync_kernel("Hydro2D pressure-boundary work audit failed");

  std::vector<double> work_host;
  work.copy_to_host(work_host);
  long double sum = 0.0L;
  for (int n = nw.begin; n < nw.end; ++n) {
    sum += static_cast<long double>(work_host[static_cast<std::size_t>(n)]);
  }
  return dt * static_cast<double>(sum);
}

double copy_node_scalar(const double* values, const int node) {
  if (values == nullptr || node < 0) {
    return 0.0;
  }
  double out = 0.0;
  cuda_check(cudaMemcpy(&out,
                        values + node,
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "Hydro2D mesh-forensics node scalar copy failed");
  return out;
}

void populate_mesh_forensics_sample(
    tenryu::coupling::HydroStepResult& result,
    const core::State& state,
    const core::NodeField1D& r_old,
    const core::NodeField1D& z_old,
    const core::NodeField1D& r_candidate,
    const core::NodeField1D& z_candidate,
    const core::NodeField1D& u_old_r,
    const core::NodeField1D& u_old_z,
    const core::NodeField1D& u_half_r,
    const core::NodeField1D& u_half_z,
    const core::NodeField1D& u_new_r,
    const core::NodeField1D& u_new_z,
    const core::NodeField1D& a_pressure_r,
    const core::NodeField1D& a_pressure_z,
    const core::NodeField1D& a_qvisc_r,
    const core::NodeField1D& a_qvisc_z,
    const core::NodeField1D& a_hourglass_r,
    const core::NodeField1D& a_hourglass_z,
    const core::NodeField1D& node_mass) {
  const int cell = result.first_failing_cell;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const bool split_polar_shell =
      state.mesh.topo.multiblock.has_value() &&
      mesh::mesh_topo_multiblock_polar_shell_block_count(
          *state.mesh.topo.multiblock) > 1;
  int nodes[4] = {-1, -1, -1, -1};
  int sample_count = 4;
  if (split_polar_shell) {
    if (cell < 0 || cell >= state.mesh.topo.n_cells) {
      return;
    }
    const mesh::MultiBlockTopology& mb = *state.mesh.topo.multiblock;
    if (mb.cell_node_csr_offsets.size() !=
            static_cast<std::size_t>(state.mesh.topo.n_cells) + 1U ||
        mb.cell_node_csr_indices.empty()) {
      return;
    }
    const int off =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int nverts =
        mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, cell);
    sample_count = std::min(4, nverts);
    if (off < 0 || sample_count <= 0 ||
        static_cast<std::size_t>(off + sample_count) >
            mb.cell_node_csr_indices.size()) {
      return;
    }
    for (int k = 0; k < sample_count; ++k) {
      nodes[k] =
          mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (nodes[k] < 0 || nodes[k] >= state.mesh.topo.n_nodes) {
        return;
      }
    }
  } else {
    if (cell < 0 || nr <= 0 || nz <= 0 || cell >= nr * nz) {
      return;
    }
    const int i = cell / nz;
    const int j = cell - i * nz;
    nodes[0] = rz_node_index_2d(i, j, nz);
    nodes[1] = rz_node_index_2d(i + 1, j, nz);
    nodes[2] = rz_node_index_2d(i + 1, j + 1, nz);
    nodes[3] = rz_node_index_2d(i, j + 1, nz);
  }
  for (int k = 0; k < sample_count; ++k) {
    auto& out = result.mesh_forensics_nodes[static_cast<std::size_t>(k)];
    const int n = nodes[k];
    out.r_old = copy_node_scalar(r_old.data(), n);
    out.z_old = copy_node_scalar(z_old.data(), n);
    out.r_candidate = copy_node_scalar(r_candidate.data(), n);
    out.z_candidate = copy_node_scalar(z_candidate.data(), n);
    out.u_r_old = copy_node_scalar(u_old_r.data(), n);
    out.u_z_old = copy_node_scalar(u_old_z.data(), n);
    out.u_r_half = copy_node_scalar(u_half_r.data(), n);
    out.u_z_half = copy_node_scalar(u_half_z.data(), n);
    out.u_r_new = copy_node_scalar(u_new_r.data(), n);
    out.u_z_new = copy_node_scalar(u_new_z.data(), n);
    out.a_r_pressure =
        a_pressure_r.empty() ? 0.0 : copy_node_scalar(a_pressure_r.data(), n);
    out.a_z_pressure =
        a_pressure_z.empty() ? 0.0 : copy_node_scalar(a_pressure_z.data(), n);
    out.a_r_qvisc = a_qvisc_r.empty() ? 0.0 : copy_node_scalar(a_qvisc_r.data(), n);
    out.a_z_qvisc = a_qvisc_z.empty() ? 0.0 : copy_node_scalar(a_qvisc_z.data(), n);
    out.a_r_hourglass =
        a_hourglass_r.empty() ? 0.0 : copy_node_scalar(a_hourglass_r.data(), n);
    out.a_z_hourglass =
        a_hourglass_z.empty() ? 0.0 : copy_node_scalar(a_hourglass_z.data(), n);
    out.node_mass = node_mass.empty() ? 0.0 : copy_node_scalar(node_mass.data(), n);
    out.corner_mass_into_cell =
        state.corner_mass.empty() ? 0.0
                                  : copy_node_scalar(
                                        state.corner_mass.data(),
                                        cell * state.corner_stride + k);
  }
  result.mesh_forensics_sample_valid = true;
}

}  // namespace

void launch_compute_node_mass_for_cfl_2d(
    double* const node_mass,
    const core::State& state,
    const core::Config& cfg,
    const std::uint8_t* const d_cell_nverts,
    const std::int8_t* const d_hydro_active) {
  if (state.mesh.topo.multiblock.has_value() &&
      !state.mesh.multiblock_reverse_csr_node_offsets.empty()) {
    detail::launch_compute_node_mass_2d_multiblock(
        node_mass,
        state.corner_mass.data(),
        state.mass.data(),
        state.x_r.data(),
        state.mesh.multiblock_reverse_csr_node_offsets.data(),
        state.mesh.multiblock_reverse_csr_node_cells.data(),
        state.mesh.multiblock_reverse_csr_node_corners.data(),
        static_cast<int>(state.x_r.size()),
        equal_split_node_mass_mode(
            cfg.numerics.hydro.axis_node_mass_convention),
        state.corner_stride);
  } else {
    launch_compute_node_mass_2d(node_mass, state, cfg, d_cell_nverts,
                                d_hydro_active);
  }
}

double compute_node_r_momentum_2d(const core::State& state,
                                  const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  if (n_cells <= 0 || n_nodes <= 0) {
    return 0.0;
  }
  const int blocks_nodes = (n_nodes + 255) / 256;

  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_nonquad("h2d:r_momentum_audit:cell_nverts", state);
  std::int8_t* d_hydro_active =
      upload_hydro_active("h2d:r_momentum_audit:hydro_active",
                          state.hydro_active);
  std::int8_t* d_hydro_force_active = d_hydro_active;
  const std::vector<std::int8_t> central_macro_effective_active =
      make_central_macro_effective_active(state);
  if (!central_macro_effective_active.empty()) {
    d_hydro_force_active = upload_hydro_active(
        "h2d:r_momentum_audit:hydro_force_active",
        central_macro_effective_active);
  }

  core::NodeField1D node_mass{"h2d:r_momentum_audit:node_mass"};
  node_mass.reset(static_cast<std::size_t>(n_nodes));
  launch_compute_node_mass_2d(
      node_mass.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  sync_kernel("Hydro2D r-momentum source-audit compute_node_mass failed");

  core::NodeField1D momentum_r{"h2d:r_momentum_audit:momentum_r"};
  momentum_r.reset(static_cast<std::size_t>(n_nodes));
  r_momentum_source_audit_node_momentum_kernel<<<blocks_nodes, 256>>>(
      momentum_r.data(), node_mass.data(), state.v_r.data(), n_nodes);
  sync_kernel("Hydro2D r-momentum source-audit node momentum failed");
  return cap_energy_audit_reduce_sum(momentum_r.data(), n_nodes);
}

tenryu::coupling::HydroStepResult refresh_geometry_and_density(
    core::State& state,
    const core::Config& cfg,
    const tenryu::coupling::HydroFailureStage stage,
    int* rho_clamp_count,
    tenryu::coupling::ProfileObservability* observability) {
  return refresh_geometry_and_density_impl(
      state, cfg, stage, rho_clamp_count, observability);
}

void validate_geometry_after_retry_restore(core::State& state,
                                            const core::Config& cfg) {
  (void)cfg;
  tenryu::mesh::MeshGeometryCheckOptions opts{};
  opts.policy = tenryu::mesh::MeshGeometryFailurePolicy::SoftReturn;
  const tenryu::mesh::MeshGeometryResult geometry =
      state.mesh.recompute_geometry_checked(opts);
  if (geometry.admissible) {
    return;
  }

  const tenryu::coupling::HydroStepResult result =
      hydro_result_from_geometry_failure(
          geometry,
          tenryu::coupling::HydroFailureStage::RetryRestoreValidate);
  const std::string record = format_hydro_geometry_failure_record(result);
  tenryu::core::log_warning(
      "Hydro2D retry restore geometry validation failed: " + record);
  TENRYU_ASSERT(false,
                "Hydro2D retry restore geometry validation failed: " + record);
}

// Applies the conservative AV cap q <= k * max(Pe + Pi, 0) in place. For
// Pe+Pi <= 0, q is intentionally set to zero so vacuum/cold cells do not retain
// artificial-viscosity pressure. Legacy av_qcap_center_band_only is consumed by
// namelist parsing; runtime dispatch uses the effective av_qcap_scope enum.
void qcap_apply_in_place(core::CellField1D& Qvisc,
                         const core::Config& cfg,
                         const core::State& state,
                         const std::int8_t* d_hydro_active,
                         core::CellField1D* Qvisc_per_material = nullptr) {
  const double qcap_over_p = cfg.numerics.hydro.av_qcap_over_p;
  if (!(qcap_over_p > 0.0)) {
    return;
  }

  const core::AvQcapScope qcap_scope = cfg.numerics.hydro.av_qcap_scope;
  if (qcap_scope == core::AvQcapScope::TRI_FAN_RADIAL_INDEX &&
      !(state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane &&
        state.mesh.polar_center_treatment == mesh::PolarCenterTreatment::TriFan)) {
    return;
  }

  const int n_cells = static_cast<int>(Qvisc.size());
  TENRYU_ASSERT(Qvisc.size() == state.Pe.size(),
                "Hydro2D AV qcap requires Qvisc/Pe size match");
  TENRYU_ASSERT(Qvisc.size() == state.Pi.size(),
                "Hydro2D AV qcap requires Qvisc/Pi size match");
  TENRYU_ASSERT(n_cells == 0 || state.mesh.topo.nz > 0,
                "Hydro2D AV qcap requires a valid 2D mesh topology");
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const std::size_t n_cell_mat =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  const bool per_material_hydro_enabled =
      cfg.main.two_temperature &&
      cfg.numerics.materials.per_material_conservation_enabled &&
      n_mat > 0 &&
      state.mass_per_material.size() == n_cell_mat &&
      state.Ee_per_material.size() == n_cell_mat &&
      state.Ei_per_material.size() == n_cell_mat &&
      state.volFrac.size() == n_cell_mat;
  TENRYU_ASSERT(!per_material_hydro_enabled || Qvisc_per_material != nullptr,
                "Hydro2D AV qcap per-material mode requires Qvisc_per_material");
  if (Qvisc_per_material != nullptr) {
    TENRYU_ASSERT(Qvisc_per_material->size() == n_cell_mat,
                  "Hydro2D AV qcap requires per-material Qvisc size match");
  }
  if (n_cells == 0) {
    return;
  }

  const double* d_cell_centroid_r = nullptr;
  double r_match_threshold = 0.0;
  if (qcap_scope == core::AvQcapScope::CENTROID_R_LE_R_MATCH) {
    TENRYU_ASSERT(std::isfinite(cfg.mesh.multiblock_cart_core_r_match),
                  "Hydro2D AV qcap centroid-r scope requires finite "
                  "Mesh.multiblock_cart_core_r_match");
    TENRYU_ASSERT(state.mesh.cell_centroid_r_device.size() == Qvisc.size(),
                  "Hydro2D AV qcap centroid-r scope requires device "
                  "cell_centroid_r mirror");
    d_cell_centroid_r = state.mesh.cell_centroid_r_device.data();
    r_match_threshold = cfg.mesh.multiblock_cart_core_r_match;
  }

  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  qcap_apply_2d_kernel<<<cw.blocks(), 256>>>(
      Qvisc.data(),
      Qvisc_per_material != nullptr ? Qvisc_per_material->data() : nullptr,
      state.Pe.data(),
      state.Pi.data(),
      d_hydro_active,
      cw.begin,
      cw.end,
      n_cells,
      state.mesh.topo.nz,
      n_mat,
      qcap_over_p,
      static_cast<int>(qcap_scope),
      cfg.numerics.hydro.tri_fan_center_cfl_band_radial_index,
      d_cell_centroid_r,
      r_match_threshold);
  sync_kernel("Hydro2D AV qcap kernel failed");
}

namespace detail {

void launch_compute_node_activity_2d_multiblock(
    std::uint8_t* const node_active,
    const int* const reverse_csr_node_offsets,
    const int n_nodes_total) {
  const int blocks = (n_nodes_total + 255) / 256;
  compute_node_activity_2d_multiblock_kernel<<<blocks, 256>>>(
      node_active, reverse_csr_node_offsets, 0, n_nodes_total, n_nodes_total);
}

void launch_compute_node_mass_2d_multiblock(
    double* const node_mass,
    const double* const corner_mass,
    const double* const mass_cell,
    const double* const x_r,
    const int* const reverse_csr_node_offsets,
    const int* const reverse_csr_node_cells,
    const int* const reverse_csr_node_corners,
    const int n_nodes_total,
    const int equal_split_node_mass_mode,
    const int corner_stride) {
  const int blocks = (n_nodes_total + 255) / 256;
  compute_node_mass_2d_multiblock_kernel<<<blocks, 256>>>(
      node_mass,
      corner_mass,
      mass_cell,
      x_r,
      nullptr,
      reverse_csr_node_offsets,
      reverse_csr_node_cells,
      reverse_csr_node_corners,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      0,
      n_nodes_total,
      n_nodes_total,
      false,
      false,
      equal_split_node_mass_mode,
      corner_stride);
}

void launch_compute_node_mass_2d_multiblock(
    double* const node_mass,
    const double* const corner_mass,
    const double* const mass_cell,
    const double* const x_r,
    const int* const reverse_csr_node_offsets,
    const int* const reverse_csr_node_cells,
    const int* const reverse_csr_node_corners,
    const int n_nodes_total,
    const int equal_split_node_mass_mode) {
  launch_compute_node_mass_2d_multiblock(
      node_mass, corner_mass, mass_cell, x_r, reverse_csr_node_offsets,
      reverse_csr_node_cells, reverse_csr_node_corners, n_nodes_total,
      equal_split_node_mass_mode, mesh::kMeshTopoCellStorageSlots);
}

void launch_apply_boundary_accel_constraints_multiblock(
    double* const accel_r,
    double* const accel_z,
    const double* const x_r,
    const double* const x_z,
    const std::uint8_t* const node_flags,
    const int n_nodes_total,
    const int r_outer_type,
    const int z_bottom_type,
    const int z_top_type) {
  const int blocks = (n_nodes_total + 255) / 256;
  apply_boundary_accel_constraints_multiblock_kernel<<<blocks, 256>>>(
      accel_r,
      accel_z,
      x_r,
      x_z,
      node_flags,
      0,
      n_nodes_total,
      n_nodes_total,
      static_cast<int>(Boundary2DType::AXIS),
      r_outer_type,
      z_bottom_type,
      z_top_type);
}

void launch_compute_cell_dVdt_2d(
    double* const dVdt,
    const double* const v_r,
    const double* const v_z,
    const double* const Svec_r,
    const double* const Svec_z,
    const core::State& state,
    const core::Config& cfg,
    const std::uint8_t* const d_cell_nverts,
    const std::int8_t* const hydro_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  // Ghost-inclusive window (OPEN-HYDRO-CORNER fix, design doc §6l): dVdt
  // feeds div_u -> dl -> Qvisc, and the corner-force scatter reads pq =
  // Pe+Pi+Q at GHOST cells; the owned-only window left ghost Q at zero
  // (the entry exchange's previous-step Q is overwritten by the full
  // -range recompute from zeroed div_u), which kicked the interface x
  // z-boundary corner nodes wherever Q != 0. Inputs (node v/x, Svec) are
  // ghost-ring fresh. Serial: identity window.
  const core::State::LaunchWindow cw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "Hydro2D multiblock dVdt requires multiblock topology");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                      static_cast<std::size_t>(n_cells) + 1U,
                  "Hydro2D multiblock dVdt requires cell-node CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                      static_cast<std::size_t>(n_cells) *
                          static_cast<std::size_t>(state.mesh.corner_stride),
                  "Hydro2D multiblock dVdt requires cell-node CSR indices");
    compute_cell_dVdt_2d_multiblock_kernel<<<cw.blocks(), 256>>>(
        dVdt,
        v_r,
        v_z,
        Svec_r,
        Svec_z,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_nverts,
        cw.begin, cw.end, n_cells, state.corner_stride);
  } else {
    const int button_outer_node_ring = button_outer_node_ring_or_disabled(state);
    const std::int8_t* const dVdt_active =
        (button_outer_node_ring >= 1) ? hydro_active : nullptr;
    compute_cell_dVdt_kernel<<<cw.blocks(), 256>>>(
        dVdt, v_r, v_z, Svec_r, Svec_z, dVdt_active, state.x_r.data(),
        state.x_z.data(), cw.begin, cw.end, state.mesh.topo.nr,
        state.mesh.topo.nz, button_outer_node_ring, state.corner_stride);
  }
}

void launch_compute_cell_dVdt_2d(
    double* const dVdt,
    const double* const v_r,
    const double* const v_z,
    const double* const Svec_r,
    const double* const Svec_z,
    const core::State& state,
    const core::Config& cfg,
    const std::int8_t* const hydro_active) {
  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_nonquad(nullptr, state);
  launch_compute_cell_dVdt_2d(
      dVdt, v_r, v_z, Svec_r, Svec_z, state, cfg, d_cell_nverts,
      hydro_active);
  if (d_cell_nverts != nullptr) {
    cuda_check(cudaFree(d_cell_nverts),
               "Hydro2D: cudaFree cell_nverts failed");
    d_cell_nverts = nullptr;
  }
}

void launch_compute_cell_dVdt_2d(
    double* const dVdt,
    const double* const v_r,
    const double* const v_z,
    const double* const Svec_r,
    const double* const Svec_z,
    const core::State& state,
    const core::Config& cfg) {
  launch_compute_cell_dVdt_2d(
      dVdt, v_r, v_z, Svec_r, Svec_z, state, cfg, nullptr);
}

void launch_compute_force_from_cells_2d(
    double* const force_r,
    double* const force_z,
    const double* const cell_pq,
    const double* const Svec_r,
    const double* const Svec_z,
    const std::int8_t* const hydro_active,
    const core::State& state,
    const core::Config& cfg) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow fw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                  "Hydro2D multiblock force requires multiblock topology");
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_offsets.size() ==
                      static_cast<std::size_t>(n_nodes) + 1U,
                  "Hydro2D multiblock force requires reverse CSR offsets");
    TENRYU_ASSERT(state.mesh.multiblock_reverse_csr_node_cells.size() ==
                      state.mesh.multiblock_reverse_csr_node_corners.size(),
                  "Hydro2D multiblock force reverse CSR payload mismatch");
    switch (cfg.numerics.hydro.rz_momentum_scheme_id) {
      case 0:
        compute_force_from_cells_2d_multiblock_kernel<0><<<nw.blocks(), 256>>>(
            force_r,
            force_z,
            cell_pq,
            Svec_r,
            Svec_z,
            state.x_r.data(),
            state.x_z.data(),
            hydro_active,
            state.mesh.multiblock_reverse_csr_node_offsets.data(),
            state.mesh.multiblock_reverse_csr_node_cells.data(),
            state.mesh.multiblock_reverse_csr_node_corners.data(),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            nw.begin, nw.end, n_nodes, state.corner_stride);
        break;
      case 1:
        TENRYU_ASSERT(
            std::all_of(
                state.mesh.cell_nverts.begin(),
                state.mesh.cell_nverts.end(),
                [](const std::uint8_t nverts) { return nverts == 4U; }),
            "area_weighted_symmetric v1 requires all-quad meshes (tri-fan caps unsupported)");
        compute_force_from_cells_2d_multiblock_kernel<1><<<nw.blocks(), 256>>>(
            force_r,
            force_z,
            cell_pq,
            Svec_r,
            Svec_z,
            state.x_r.data(),
            state.x_z.data(),
            hydro_active,
            state.mesh.multiblock_reverse_csr_node_offsets.data(),
            state.mesh.multiblock_reverse_csr_node_cells.data(),
            state.mesh.multiblock_reverse_csr_node_corners.data(),
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            nw.begin, nw.end, n_nodes, state.corner_stride);
        break;
      default:
        TENRYU_ASSERT(false, "Hydro2D invalid resolved RZ momentum scheme");
    }
  } else {
    const int button_outer_node_ring = button_outer_node_ring_or_disabled(state);
    switch (cfg.numerics.hydro.rz_momentum_scheme_id) {
      case 0:
        compute_force_from_cells_kernel<0><<<fw.blocks(), 256>>>(
            force_r,
            force_z,
            cell_pq,
            Svec_r,
            Svec_z,
            state.x_r.data(),
            state.x_z.data(),
            hydro_active,
            fw.begin,
            fw.end,
            state.mesh.topo.nr,
            state.mesh.topo.nz,
            button_outer_node_ring,
            state.corner_stride);
        break;
      case 1:
        compute_force_from_cells_kernel<1><<<fw.blocks(), 256>>>(
            force_r,
            force_z,
            cell_pq,
            Svec_r,
            Svec_z,
            state.x_r.data(),
            state.x_z.data(),
            hydro_active,
            fw.begin,
            fw.end,
            state.mesh.topo.nr,
            state.mesh.topo.nz,
            button_outer_node_ring,
            state.corner_stride);
        break;
      default:
        TENRYU_ASSERT(false, "Hydro2D invalid resolved RZ momentum scheme");
    }
  }
}

void compute_hydro_2d_force_work_audit_fields(
    core::State& state,
    const core::Config& cfg,
    const double eval_time,
    core::NodeField1D& force_r,
    core::NodeField1D& force_z,
    core::NodeField1D& accel_r,
    core::NodeField1D& accel_z,
    core::NodeField1D& node_mass,
    core::CellField1D& cell_pq,
    core::CellField1D& dVdt,
    core::CellField1D& div_u,
    core::CellField1D& dl,
    core::CellField1D& cs) {
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "Hydro2D force/work audit requires 2D_RZ geometry");
  TENRYU_ASSERT(state.mesh.dim == 2,
                "Hydro2D force/work audit requires a 2D mesh");
  TENRYU_ASSERT(cfg.numerics.hydro.av_type == "vnr",
                "Hydro2D force/work audit supports only VNR artificial viscosity");
  TENRYU_ASSERT(!cfg.numerics.materials.per_material_conservation_enabled,
                "Hydro2D force/work audit does not support per-material hydro");
  if (state.rho.empty()) {
    return;
  }

  validate_boundary_2d(cfg);

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  const core::State::LaunchWindow pw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  const HydroTableViews eos_views = select_hydro_table_views(nullptr);
  const bool use_two_temp = cfg.main.two_temperature;

  std::int8_t* d_hydro_active =
      upload_hydro_active("h2d:hydro_active", state.hydro_active);
  std::int8_t* d_hydro_force_active = d_hydro_active;
  std::vector<std::int8_t> central_macro_effective_active;
  const std::vector<std::int8_t>* hydro_force_active_host = &state.hydro_active;
  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_nonquad("h2d:cell_nverts", state);
  std::uint8_t* d_node_flags =
      upload_node_flags_if_constraints("h2d:node_flags", state);
  std::uint8_t* d_node_active = nullptr;
  auto cleanup = [&]() {
    if (d_node_active != nullptr) {
      d_node_active = nullptr;
    }
    if (d_hydro_force_active != nullptr && d_hydro_force_active != d_hydro_active) {
      d_hydro_force_active = nullptr;
    }
    if (d_hydro_active != nullptr) {
      d_hydro_active = nullptr;
    }
    if (d_cell_nverts != nullptr) {
      d_cell_nverts = nullptr;
    }
    if (d_node_flags != nullptr) {
      d_node_flags = nullptr;
    }
  };

  const tenryu::coupling::HydroStepResult geometry_result =
      refresh_geometry_and_density_impl(
          state,
          cfg,
          tenryu::coupling::HydroFailureStage::StepStart);
  TENRYU_ASSERT(geometry_result.admissible,
                "Hydro2D force/work audit geometry refresh failed");

  MeshDeviceCache mesh_cache;
  upload_mesh_cache(state, mesh_cache);
  ensure_corner_mass_2d(state,
                        d_cell_nverts,
                        static_cast<int>(
                            cfg.numerics.hydro.corner_mass_convention),
                        cfg.numerics.hydro.subzonal_pressure_enabled,
                        recompute_corner_mass_each_step(cfg),
                        corner_mass_lagrangian_invariant_enabled(cfg),
                        cfg.numerics.hydro.aw_compatible_force_work,
                        mesh::mesh_topo_polar_tier_family(cfg.mesh));
  ensure_hourglass_subzonal_masses_2d(state, cfg, false);
  pole_angular_derefine::ensure_built(state, cfg);
  ale::csr_optionb_canonicalize_corner_mass_basis(state, cfg);
  central_pseudo_core::aggregate_state(state, cfg, "force_work_audit", false);
  pole_angular_derefine::aggregate_state(state, cfg, "force_work_audit", false);
  central_macro_effective_active = make_central_macro_effective_active(state);
  if (!central_macro_effective_active.empty()) {
    d_hydro_force_active = upload_hydro_active("h2d:hydro_force_active",
                                               central_macro_effective_active);
    hydro_force_active_host = &central_macro_effective_active;
  }

  enforce_eos_closure(state, cfg, use_two_temp, eos_views,
                      d_hydro_force_active);

  d_node_active = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "h2d:audit_node_active",
      static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t)));
  cuda_check(cudaMemset(d_node_active,
                        0,
                        static_cast<std::size_t>(n_nodes) * sizeof(std::uint8_t)),
             "Hydro2D force/work audit cudaMemset node_active failed");
  launch_compute_node_activity_2d(d_node_active, d_hydro_force_active, state);
  sync_kernel("Hydro2D force/work audit compute_node_activity failed");

  node_mass.reset(static_cast<std::size_t>(n_nodes));
  launch_compute_node_mass_2d(
      node_mass.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  sync_kernel("Hydro2D force/work audit compute_node_mass failed");
  const int blocks_cells = (n_cells + 255) / 256;
  if (cfg.numerics.hydro.rz_momentum_scheme_id == 1) {
    const int blocks_cells = (n_cells + 255) / 256;
    launch_compute_node_planar_mass_2d(
        state, d_cell_nverts, d_hydro_force_active, blocks_cells);
    sync_kernel("Hydro2D force/work audit compute_node_planar_mass failed");
  }

  compute_cell_sound_speed(cs, state, cfg, use_two_temp, eos_views,
                           d_hydro_force_active);

  dVdt.reset(static_cast<std::size_t>(n_cells));
  div_u.reset(static_cast<std::size_t>(n_cells));
  dl.reset(static_cast<std::size_t>(n_cells));
  // OPEN-HYDRO-CORNER fix: the AV input chain must cover the ghost ring
  // (the corner-force scatter reads Q there). Serial identity window.
  const core::State::LaunchWindow av_fw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  launch_compute_cell_dVdt_2d(
      dVdt.data(), state.v_r.data(), state.v_z.data(), mesh_cache.Svec_r.data(),
      mesh_cache.Svec_z.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  compute_cell_div_u_kernel<<<av_fw.blocks(), 256>>>(
      div_u.data(), dVdt.data(), state.vol.data(), av_fw.begin, av_fw.end, n_cells);
  compute_cell_length_scale_kernel<<<av_fw.blocks(), 256>>>(
      dl.data(), mesh_cache.area.data(), av_fw.begin, av_fw.end, n_cells);
  sync_kernel("Hydro2D force/work audit dVdt/div_u/dl failed");

  const bool av_eos_aware_active =
      cfg.numerics.hydro.av_eos_aware &&
      (eos_views.tab_total.n_rho > 0 || eos_views.tab_ion.n_rho > 0 ||
       eos_views.tab_ele.n_rho > 0 || eos_views.spline_total.n_rho > 0 ||
       eos_views.jet_total.n_rho > 0);
  ArtificialViscosity av(cfg.numerics.hydro.av_linear,
                         cfg.numerics.hydro.av_quadratic,
                         1.0,
                         1.0,
                         av_eos_aware_active,
                         cfg.numerics.hydro.av_eos_gamma1_ref,
                         cfg.numerics.hydro.av_eos_boost_max,
                         ArtificialViscosityType::Vnr);
  if (cfg.numerics.hydro.av_model == core::AvModel::ScalarVnrLegacy) {
    av.compute_Q_2d(state.Qvisc, state.rho, dl, div_u, state.Pe, state.Pi, cs,
                    *hydro_force_active_host);
    qcap_apply_in_place(state.Qvisc, cfg, state, d_hydro_force_active);
  } else {
    cuda_check(cudaMemset(state.Qvisc.data(),
                          0,
                          static_cast<std::size_t>(n_cells) * sizeof(double)),
               "Hydro2D force/work audit zero compatible-mode Qvisc failed");
  }

  cell_pq.reset(static_cast<std::size_t>(n_cells));
  build_cell_pq_kernel<<<pw.blocks(), 256>>>(
      cell_pq.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
      d_hydro_force_active, pw.begin, pw.end, n_cells);
  sync_kernel("Hydro2D force/work audit build_cell_pq failed");

  force_r.reset(static_cast<std::size_t>(n_nodes));
  force_z.reset(static_cast<std::size_t>(n_nodes));
  accel_r.reset(static_cast<std::size_t>(n_nodes));
  accel_z.reset(static_cast<std::size_t>(n_nodes));
  if (cfg.numerics.hydro.aw_compatible_force_work) {
    const AwAxisSlaveStructuredLines aw_axis_slave_lines =
        detect_aw_axis_slave_structured_lines(state, cfg);
    compute_acceleration_2d(
        accel_r, accel_z, state, cfg, eval_time, mesh_cache, cell_pq,
        d_node_active, d_node_flags, node_mass, d_hydro_force_active, nullptr,
        true, false, true, &cs, true, &force_r, &force_z, 0.0, false, nullptr,
        true, aw_axis_slave_lines.theta0_active,
        aw_axis_slave_lines.theta_pi_active, nullptr, nullptr, false, nullptr,
        "force_work_audit");
    // Same rationale as the fall-through path below: the force umbrella runs
    // the compatible-work corrector BEFORE the csw edge-AV assembly, whose
    // apply re-zeroes work_av_per_cell and leaves planar assembly-stage
    // estimates. Re-run the corrector so the audit fields expose the
    // corrector-final exact RZ work from the fully assembled forces.
    if (compatible::compatible_force_work_enabled(cfg) &&
        !(state.mesh.button_center && state.mesh.button_center->enabled)) {
      compatible::launch_compute_compatible_work_2d(
          state, cfg, state.v_r.data(), state.v_z.data(), d_hydro_active);
    }
    cleanup();
    return;
  }

  launch_compute_force_from_cells_2d(
      force_r.data(), force_z.data(), cell_pq.data(), mesh_cache.Svec_r.data(),
      mesh_cache.Svec_z.data(), d_hydro_force_active, state, cfg);

  // The force umbrella runs the compatible-work corrector BEFORE the csw
  // edge-AV assembly, whose apply then re-zeroes work_av_per_cell and leaves
  // planar assembly-stage estimates in the ledger (production instead
  // consumes the midpoint corrector stage downstream). Re-run the corrector
  // here so the audit fields expose the corrector-final exact RZ work,
  // recomputed from the fully assembled forces, for all three terms.
  if (compatible::compatible_force_work_enabled(cfg) &&
      !(state.mesh.button_center && state.mesh.button_center->enabled)) {
    compatible::launch_compute_compatible_work_2d(
        state, cfg, state.v_r.data(), state.v_z.data(), d_hydro_active);
  }

  const auto r_outer_type =
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer);
  if (r_outer_type == Boundary2DType::PRESSURE) {
    TENRYU_ASSERT(state.pressure_drive_1d.has_value(),
                  "Hydro2D force/work audit pressure boundary requires table");
    const double p_ext = state.pressure_drive_1d->eval(eval_time);
    const auto& pcfg =
        cfg.numerics.hydro.pressure_drive_perturbation;
    PressureDrivePerturbationParams pp{};
    if (pcfg.enabled) {
      pp = core::make_pressure_drive_perturbation_params(pcfg);
    }
    const bool rz_exact_endpoint =
        state.mesh.logical == mesh::LogicalMesh2D::SphericalPolarHalfplane;
    if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
      mesh::mesh_topo_assert_multiblock_outermost_polar_shell_outer_boundary(
          state.mesh.topo);
      launch_multiblock_polar_shell_pressure_forces(
          force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
          mesh::mesh_topo_multiblock_outermost_polar_shell_node_offset(
              state.mesh.topo),
          mesh::mesh_topo_multiblock_outermost_polar_shell_nr(state.mesh.topo),
          mesh::mesh_topo_multiblock_outermost_polar_shell_nz(state.mesh.topo),
          p_ext,
          rz_exact_endpoint, node_mass.data(),
          cfg.numerics.hydro.rz_momentum_scheme_id, pp, pcfg.enabled);
    } else {
      launch_r_outer_boundary_pressure_forces(
          force_r.data(), force_z.data(), state.x_r.data(), state.x_z.data(),
          state.mesh.topo.nr, state.mesh.topo.nz, p_ext, rz_exact_endpoint,
          cfg.numerics.hydro.rz_momentum_scheme_id, pp, pcfg.enabled);
    }
  }

  launch_compute_accel_from_force_2d(
      accel_r.data(), accel_z.data(), force_r.data(), force_z.data(), node_mass,
      d_node_active, d_node_flags, n_nodes, state, cfg);

  launch_apply_boundary_accel_constraints(
      accel_r.data(),
      accel_z.data(),
      d_node_flags,
      state,
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_inner),
      r_outer_type,
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_bottom),
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.z_top));
  if (!state.mesh.topo.multiblock.has_value()) {
    const double* const resolved_node_mass =
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data();
    launch_apply_evac_contact_accel_constraints(
        state, accel_r.data(), accel_z.data(), resolved_node_mass);
  }

  sync_kernel("Hydro2D force/work audit force/accel failed");
  cleanup();
}

void accumulate_tri_fan_center_perturbation_diag(
    core::State& state,
    const core::Config& cfg,
    const core::NodeField1D& node_mass,
    const core::NodeField1D& velocity_r,
    const core::NodeField1D& velocity_z,
    const core::NodeField1D& accel_pressure_r,
    const core::NodeField1D& accel_pressure_z,
    const core::NodeField1D& accel_q_r,
    const core::NodeField1D& accel_q_z) {
  if (cfg.numerics.hydro.center_perturbation_diag_scope !=
      core::CenterPerturbationDiagScope::TRI_FAN_FIRST_RING) {
    return;
  }
  state.tri_fan_center_perturbation_diag.reset();
  if (!is_tri_fan_center_perturbation_topology(state)) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (nr <= 0 || nz <= 0 || n_cells <= 0 || n_nodes <= 0 ||
      static_cast<int>(state.Qvisc.size()) != n_cells ||
      static_cast<int>(state.Pe.size()) != n_cells ||
      static_cast<int>(state.Pi.size()) != n_cells ||
      static_cast<int>(state.vol.size()) != n_cells ||
      static_cast<int>(state.x_r.size()) != n_nodes ||
      static_cast<int>(state.x_z.size()) != n_nodes ||
      static_cast<int>(state.x_r_initial.size()) != n_nodes ||
      static_cast<int>(state.x_z_initial.size()) != n_nodes ||
      static_cast<int>(node_mass.size()) != n_nodes ||
      static_cast<int>(velocity_r.size()) != n_nodes ||
      static_cast<int>(velocity_z.size()) != n_nodes ||
      static_cast<int>(accel_pressure_r.size()) != n_nodes ||
      static_cast<int>(accel_pressure_z.size()) != n_nodes ||
      static_cast<int>(accel_q_r.size()) != n_nodes ||
      static_cast<int>(accel_q_z.size()) != n_nodes) {
    return;
  }

  const std::vector<double> mass_n = copy_field_to_vector(node_mass);
  const std::vector<double> u_r = copy_field_to_vector(velocity_r);
  const std::vector<double> u_z = copy_field_to_vector(velocity_z);
  const std::vector<double> a_p_r = copy_field_to_vector(accel_pressure_r);
  const std::vector<double> a_p_z = copy_field_to_vector(accel_pressure_z);
  const std::vector<double> a_q_r = copy_field_to_vector(accel_q_r);
  const std::vector<double> a_q_z = copy_field_to_vector(accel_q_z);

  double mean_u_r = 0.0;
  double mean_u_z = 0.0;
  for (int j = 0; j <= nz; ++j) {
    const int n = rz_node_index_2d(1, j, nz);
    mean_u_r += u_r[static_cast<std::size_t>(n)];
    mean_u_z += u_z[static_cast<std::size_t>(n)];
  }
  const double inv_ring_count = 1.0 / static_cast<double>(nz + 1);
  mean_u_r *= inv_ring_count;
  mean_u_z *= inv_ring_count;

  // First-ring perturbation split for NUMERICS T5:
  // u'_n = u_n - <u>_{i=1}; Edot' sums m_n u'_n dot a^source_n.
  double Edot_prime_P = 0.0;
  double Edot_prime_Q = 0.0;
  double Eprime_k = 0.0;
  for (int j = 0; j <= nz; ++j) {
    const int n = rz_node_index_2d(1, j, nz);
    const auto idx = static_cast<std::size_t>(n);
    const double m = mass_n[idx];
    if (!(m > 0.0) || !std::isfinite(m)) {
      continue;
    }
    const double du_r = u_r[idx] - mean_u_r;
    const double du_z = u_z[idx] - mean_u_z;
    Edot_prime_P += m * (du_r * a_p_r[idx] + du_z * a_p_z[idx]);
    Edot_prime_Q += m * (du_r * a_q_r[idx] + du_z * a_q_z[idx]);
    Eprime_k += 0.5 * m * (du_r * du_r + du_z * du_z);
  }

  const std::vector<double> qvisc = copy_field_to_vector(state.Qvisc);
  const std::vector<double> pe = copy_field_to_vector(state.Pe);
  const std::vector<double> pi = copy_field_to_vector(state.Pi);
  const std::vector<double> vol = copy_field_to_vector(state.vol);
  const std::vector<double> r = copy_field_to_vector(state.x_r);
  const std::vector<double> z = copy_field_to_vector(state.x_z);
  const std::vector<double> r_initial = copy_field_to_vector(state.x_r_initial);
  const std::vector<double> z_initial = copy_field_to_vector(state.x_z_initial);

  const int band_i =
      std::min(nr - 1,
               std::max(0, cfg.numerics.hydro.tri_fan_center_cfl_band_radial_index));
  double max_q_over_p = -std::numeric_limits<double>::infinity();
  double min_corner_J = std::numeric_limits<double>::infinity();
  double min_cell_volume = std::numeric_limits<double>::infinity();
  bool q_observed = false;
  bool corner_observed = false;
  bool volume_observed = false;
  const bool have_cell_nverts =
      static_cast<int>(state.mesh.cell_nverts.size()) == n_cells;

  auto active_cell = [&](const int c) {
    return state.hydro_active.empty() ||
           state.hydro_active[static_cast<std::size_t>(c)] != 0;
  };
  auto observe_corner_ratio = [&](const double ratio) {
    if (!std::isfinite(ratio)) {
      return;
    }
    min_corner_J = corner_observed ? std::min(min_corner_J, ratio) : ratio;
    corner_observed = true;
  };

  for (int i = 0; i <= band_i; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      if (!active_cell(c)) {
        continue;
      }
      const auto cidx = static_cast<std::size_t>(c);
      const double p = std::max(pe[cidx] + pi[cidx], 1.0e-30);
      const double q_over_p = qvisc[cidx] / p;
      if (std::isfinite(q_over_p)) {
        max_q_over_p = q_observed ? std::max(max_q_over_p, q_over_p) : q_over_p;
        q_observed = true;
      }
      if (std::isfinite(vol[cidx])) {
        min_cell_volume =
            volume_observed ? std::min(min_cell_volume, vol[cidx]) : vol[cidx];
        volume_observed = true;
      }

      const int n00 = rz_node_index_2d(i, j, nz);
      const int n10 = rz_node_index_2d(i + 1, j, nz);
      const int n11 = rz_node_index_2d(i + 1, j + 1, nz);
      const int n01 = rz_node_index_2d(i, j + 1, nz);
      const int active_nverts =
          have_cell_nverts
              ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
              : mesh::kMeshTopoCellStorageSlots;
      const bool tri = active_nverts == 3;
      if (tri) {
        const double j0 = corner_jacobian_cross2(
            r_initial[static_cast<std::size_t>(n10)] -
                r_initial[static_cast<std::size_t>(n00)],
            z_initial[static_cast<std::size_t>(n10)] -
                z_initial[static_cast<std::size_t>(n00)],
            r_initial[static_cast<std::size_t>(n11)] -
                r_initial[static_cast<std::size_t>(n00)],
            z_initial[static_cast<std::size_t>(n11)] -
                z_initial[static_cast<std::size_t>(n00)]);
        const double j1 = corner_jacobian_cross2(
            r[static_cast<std::size_t>(n10)] - r[static_cast<std::size_t>(n00)],
            z[static_cast<std::size_t>(n10)] - z[static_cast<std::size_t>(n00)],
            r[static_cast<std::size_t>(n11)] - r[static_cast<std::size_t>(n00)],
            z[static_cast<std::size_t>(n11)] - z[static_cast<std::size_t>(n00)]);
        observe_corner_ratio(finite_signed_ratio_or_zero(j1, j0));
        continue;
      }

      const int nodes[4] = {n00, n10, n11, n01};
      double rr0[4];
      double zz0[4];
      double rr1[4];
      double zz1[4];
      for (int k = 0; k < 4; ++k) {
        const auto n = static_cast<std::size_t>(nodes[k]);
        rr0[k] = r_initial[n];
        zz0[k] = z_initial[n];
        rr1[k] = r[n];
        zz1[k] = z[n];
      }
      for (int corner = 0; corner < 4; ++corner) {
        observe_corner_ratio(finite_signed_ratio_or_zero(
            corner_jacobian_from_quad(rr1, zz1, corner),
            corner_jacobian_from_quad(rr0, zz0, corner)));
      }
    }
  }

  auto& diag = state.tri_fan_center_perturbation_diag;
  diag.valid = true;
  diag.Edot_prime_P = Edot_prime_P;
  diag.Edot_prime_Q = Edot_prime_Q;
  diag.Eprime_k = Eprime_k;
  diag.max_q_over_p = q_observed ? max_q_over_p : 0.0;
  diag.min_corner_J = corner_observed ? min_corner_J : 0.0;
  diag.min_cell_volume = volume_observed ? min_cell_volume : 0.0;
}

void accumulate_centroid_r_innermost_bins_perturbation_diag(
    core::State& state,
    const core::Config& cfg,
    const core::NodeField1D& node_mass,
    const core::NodeField1D& velocity_r,
    const core::NodeField1D& velocity_z,
    const core::NodeField1D& accel_pressure_r,
    const core::NodeField1D& accel_pressure_z,
    const core::NodeField1D& accel_q_r,
    const core::NodeField1D& accel_q_z) {
  if (cfg.numerics.hydro.center_perturbation_diag_scope !=
      core::CenterPerturbationDiagScope::CENTROID_R_INNERMOST_BINS) {
    return;
  }
  state.tri_fan_center_perturbation_diag.reset();
  if (!is_multiblock_center_perturbation_topology(state)) {
    return;
  }

  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int ntheta = 4 * cfg.mesh.multiblock_cart_core_n_c;
  const int bridge_layers = cfg.mesh.multiblock_cart_core_bridge_layers;
  const int radial_bins =
      cfg.numerics.hydro.center_perturbation_diag_radial_bins;
  if (n_cells <= 0 || n_nodes <= 0 || ntheta <= 0 || bridge_layers <= 0 ||
      radial_bins <= 0 ||
      static_cast<int>(mb.blocks.size()) != mb.block_count ||
      static_cast<int>(mb.cell_block_id.size()) != n_cells ||
      static_cast<int>(mb.cell_node_csr_offsets.size()) != n_cells + 1 ||
      static_cast<int>(mb.cell_node_csr_indices.size()) !=
          state.mesh.corner_stride * n_cells ||
      mb.n_cells_core < 0 || mb.n_cells_bridge < 0 || mb.n_cells_shell < 0 ||
      mb.n_cells_core + mb.n_cells_bridge + mb.n_cells_shell != n_cells ||
      mb.n_cells_bridge != bridge_layers * ntheta ||
      static_cast<int>(state.Qvisc.size()) != n_cells ||
      static_cast<int>(state.Pe.size()) != n_cells ||
      static_cast<int>(state.Pi.size()) != n_cells ||
      static_cast<int>(state.vol.size()) != n_cells ||
      static_cast<int>(state.x_r.size()) != n_nodes ||
      static_cast<int>(state.x_z.size()) != n_nodes ||
      static_cast<int>(state.x_r_initial.size()) != n_nodes ||
      static_cast<int>(state.x_z_initial.size()) != n_nodes ||
      static_cast<int>(node_mass.size()) != n_nodes ||
      static_cast<int>(velocity_r.size()) != n_nodes ||
      static_cast<int>(velocity_z.size()) != n_nodes ||
      static_cast<int>(accel_pressure_r.size()) != n_nodes ||
      static_cast<int>(accel_pressure_z.size()) != n_nodes ||
      static_cast<int>(accel_q_r.size()) != n_nodes ||
      static_cast<int>(accel_q_z.size()) != n_nodes) {
    return;
  }

  auto active_cell = [&](const int c) {
    return state.hydro_active.empty() ||
           state.hydro_active[static_cast<std::size_t>(c)] != 0;
  };
  const bool have_cell_nverts =
      static_cast<int>(state.mesh.cell_nverts.size()) == n_cells;

  auto radial_bin = [&](const int c) {
    const mesh::BlockInfo* block = multiblock_cell_block(mb, c, nullptr);
    if (block == nullptr || block->n_j_cells <= 0) {
      return std::numeric_limits<int>::max();
    }
    const int local = c - block->cell_begin;
    if (local < 0 || local >= block->cell_count) {
      return std::numeric_limits<int>::max();
    }
    const int local_i = local / block->n_j_cells;
    if (local_i < 0 || local_i >= block->n_i_cells) {
      return std::numeric_limits<int>::max();
    }
    switch (block->role) {
      case mesh::BlockRole::CENTRAL_CORE:
        return 0;
      case mesh::BlockRole::BRIDGE:
      case mesh::BlockRole::NORTH_FAN:
      case mesh::BlockRole::EAST_FAN:
      case mesh::BlockRole::SOUTH_FAN:
        return 1 + local_i;
      case mesh::BlockRole::POLAR_SHELL:
        return 1 + bridge_layers + local_i;
    }
    return std::numeric_limits<int>::max();
  };

  std::vector<int> selected_cells;
  selected_cells.reserve(static_cast<std::size_t>(n_cells));
  std::vector<std::uint8_t> selected_node(static_cast<std::size_t>(n_nodes), 0);
  for (int c = 0; c < n_cells; ++c) {
    if (!active_cell(c) || radial_bin(c) >= radial_bins) {
      continue;
    }
    selected_cells.push_back(c);
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int next = mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    if (next - off != state.mesh.corner_stride || off < 0 ||
        next > static_cast<int>(mb.cell_node_csr_indices.size())) {
      return;
    }
    const int active_nverts =
        have_cell_nverts
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : mesh::kMeshTopoCellStorageSlots;
    for (int p = off; p < off + active_nverts; ++p) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(p)];
      if (n < 0 || n >= n_nodes) {
        return;
      }
      selected_node[static_cast<std::size_t>(n)] = 1;
    }
  }
  if (selected_cells.empty()) {
    return;
  }

  const std::vector<double> mass_n = copy_field_to_vector(node_mass);
  const std::vector<double> u_r = copy_field_to_vector(velocity_r);
  const std::vector<double> u_z = copy_field_to_vector(velocity_z);
  const std::vector<double> a_p_r = copy_field_to_vector(accel_pressure_r);
  const std::vector<double> a_p_z = copy_field_to_vector(accel_pressure_z);
  const std::vector<double> a_q_r = copy_field_to_vector(accel_q_r);
  const std::vector<double> a_q_z = copy_field_to_vector(accel_q_z);

  double mass_sum = 0.0;
  double mean_u_r = 0.0;
  double mean_u_z = 0.0;
  for (int n = 0; n < n_nodes; ++n) {
    if (selected_node[static_cast<std::size_t>(n)] == 0) {
      continue;
    }
    const auto idx = static_cast<std::size_t>(n);
    const double m = mass_n[idx];
    if (!(m > 0.0) || !std::isfinite(m)) {
      continue;
    }
    mass_sum += m;
    mean_u_r += m * u_r[idx];
    mean_u_z += m * u_z[idx];
  }
  if (!(mass_sum > 0.0) || !std::isfinite(mass_sum)) {
    return;
  }
  mean_u_r /= mass_sum;
  mean_u_z /= mass_sum;

  double Edot_prime_P = 0.0;
  double Edot_prime_Q = 0.0;
  double Eprime_k = 0.0;
  for (int n = 0; n < n_nodes; ++n) {
    if (selected_node[static_cast<std::size_t>(n)] == 0) {
      continue;
    }
    const auto idx = static_cast<std::size_t>(n);
    const double m = mass_n[idx];
    if (!(m > 0.0) || !std::isfinite(m)) {
      continue;
    }
    const double du_r = u_r[idx] - mean_u_r;
    const double du_z = u_z[idx] - mean_u_z;
    Edot_prime_P += m * (du_r * a_p_r[idx] + du_z * a_p_z[idx]);
    Edot_prime_Q += m * (du_r * a_q_r[idx] + du_z * a_q_z[idx]);
    Eprime_k += 0.5 * m * (du_r * du_r + du_z * du_z);
  }

  const std::vector<double> qvisc = copy_field_to_vector(state.Qvisc);
  const std::vector<double> pe = copy_field_to_vector(state.Pe);
  const std::vector<double> pi = copy_field_to_vector(state.Pi);
  const std::vector<double> vol = copy_field_to_vector(state.vol);
  const std::vector<double> r = copy_field_to_vector(state.x_r);
  const std::vector<double> z = copy_field_to_vector(state.x_z);
  const std::vector<double> r_initial = copy_field_to_vector(state.x_r_initial);
  const std::vector<double> z_initial = copy_field_to_vector(state.x_z_initial);

  double max_q_over_p = -std::numeric_limits<double>::infinity();
  double min_corner_J = std::numeric_limits<double>::infinity();
  double min_cell_volume = std::numeric_limits<double>::infinity();
  bool q_observed = false;
  bool corner_observed = false;
  bool volume_observed = false;

  auto observe_corner_ratio = [&](const double ratio) {
    if (!std::isfinite(ratio)) {
      return;
    }
    min_corner_J = corner_observed ? std::min(min_corner_J, ratio) : ratio;
    corner_observed = true;
  };

  for (const int c : selected_cells) {
    const auto cidx = static_cast<std::size_t>(c);
    const double p = std::max(pe[cidx] + pi[cidx], 1.0e-30);
    const double q_over_p = qvisc[cidx] / p;
    if (std::isfinite(q_over_p)) {
      max_q_over_p = q_observed ? std::max(max_q_over_p, q_over_p) : q_over_p;
      q_observed = true;
    }
    if (std::isfinite(vol[cidx])) {
      min_cell_volume =
          volume_observed ? std::min(min_cell_volume, vol[cidx]) : vol[cidx];
      volume_observed = true;
    }

    const int off = mb.cell_node_csr_offsets[cidx];
    const int active_nverts =
        have_cell_nverts
            ? mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c)
            : mesh::kMeshTopoCellStorageSlots;
    int nodes[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double rr0[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double zz0[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double rr1[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    double zz1[mesh::kMeshTopoCellStorageSlotsMaxGeneral];
    for (int k = 0; k < active_nverts; ++k) {
      nodes[k] = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      const auto nidx = static_cast<std::size_t>(nodes[k]);
      rr0[k] = r_initial[nidx];
      zz0[k] = z_initial[nidx];
      rr1[k] = r[nidx];
      zz1[k] = z[nidx];
    }
    if (active_nverts == 3) {
      const double j0 = corner_jacobian_cross2(rr0[1] - rr0[0],
                                               zz0[1] - zz0[0],
                                               rr0[2] - rr0[0],
                                               zz0[2] - zz0[0]);
      const double j1 = corner_jacobian_cross2(rr1[1] - rr1[0],
                                               zz1[1] - zz1[0],
                                               rr1[2] - rr1[0],
                                               zz1[2] - zz1[0]);
      observe_corner_ratio(finite_signed_ratio_or_zero(j1, j0));
    } else {
      for (int corner = 0; corner < 4; ++corner) {
        observe_corner_ratio(finite_signed_ratio_or_zero(
            corner_jacobian_from_quad(rr1, zz1, corner),
            corner_jacobian_from_quad(rr0, zz0, corner)));
      }
    }
  }

  auto& diag = state.tri_fan_center_perturbation_diag;
  diag.valid = true;
  diag.Edot_prime_P = Edot_prime_P;
  diag.Edot_prime_Q = Edot_prime_Q;
  diag.Eprime_k = Eprime_k;
  diag.max_q_over_p = q_observed ? max_q_over_p : 0.0;
  diag.min_corner_J = corner_observed ? min_corner_J : 0.0;
  diag.min_cell_volume = volume_observed ? min_cell_volume : 0.0;
}

}  // namespace detail

TrialVolumeCflResult check_trial_lagrangian_volume_cfl(
    const core::State& state,
    const double dt,
    const double floor_fraction,
    const double shrink_fraction,
    const parallel::Reduction* reduction) {
  if (state.dt_prev_hydro <= 0.0) {
    return TrialVolumeCflResult{true, 1.0, dt, -1};
  }
  if (state.mesh.topo.multiblock.has_value()) {
    return TrialVolumeCflResult{true, 1.0, dt, -1};
  }
  TENRYU_ASSERT(state.mesh.dim == 2,
                "trial-volume CFL requires 2D_RZ mesh geometry");
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  bool bottom_state_supply = false;
  bool top_state_supply = false;
  if (nr > 0 && nz > 0 &&
      state.state_supply_mask.size() == static_cast<std::size_t>(nr * nz)) {
    for (int i = 0; i < nr; ++i) {
      bottom_state_supply =
          bottom_state_supply ||
          state.state_supply_mask[static_cast<std::size_t>(i * nz)] != 0;
      top_state_supply =
          top_state_supply ||
          state.state_supply_mask[static_cast<std::size_t>(i * nz + (nz - 1))] != 0;
    }
  }
  core::NodeField1D position_v_z{"h2d:trialcfl:position_v_z"};
  const core::NodeField1D* trial_v_z = &state.v_z;
  if (bottom_state_supply || top_state_supply) {
    const int n_nodes = static_cast<int>(state.v_z.size());
    position_v_z.reset(n_nodes);
    cuda_check(cudaMemcpy(position_v_z.data(), state.v_z.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy trial-volume position z velocity failed");
    zero_state_supply_z_position_velocity(position_v_z, nr, nz, bottom_state_supply,
                                          top_state_supply);
    trial_v_z = &position_v_z;
  }
  return check_trial_lagrangian_volume_cfl_fields(
      nr, nz, state.x_r, state.x_z,
      state.v_r, *trial_v_z, state.vol, dt, floor_fraction, shrink_fraction,
      &state.hydro_active, reduction);
}

void Hydro2D::prepare_initial_sound_speed(core::State& state,
                                          const core::Config& cfg,
                                          const HydroEOSContext* eos_ctx) const {
  if (state.rho.empty()) {
    return;
  }

  const HydroTableViews eos_views = select_hydro_table_views(eos_ctx);

  const bool use_two_temp = cfg.main.two_temperature;
  enforce_eos_closure(state, cfg, use_two_temp, eos_views);

  core::CellField1D cs_n;
  compute_cell_sound_speed(cs_n, state, cfg, use_two_temp, eos_views);
  state.cs = std::move(cs_n);
}

void Hydro2D::prepare_initial_corner_mass(core::State& state,
                                          const core::Config& cfg) const {
  if (state.mass.empty()) {
    return;
  }
  const int n_cells = static_cast<int>(state.mass.size());
  const int blocks_cells = (n_cells + 255) / 256;
  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_nonquad(nullptr, state);
  ensure_corner_mass_2d(state,
                        d_cell_nverts,
                        static_cast<int>(
                            cfg.numerics.hydro.corner_mass_convention),
                        cfg.numerics.hydro.subzonal_pressure_enabled,
                        recompute_corner_mass_each_step(cfg),
                        corner_mass_lagrangian_invariant_enabled(cfg),
                        cfg.numerics.hydro.aw_compatible_force_work,
                        mesh::mesh_topo_polar_tier_family(cfg.mesh));
  if (d_cell_nverts != nullptr) {
    cuda_check(cudaFree(d_cell_nverts),
               "Hydro2D: cudaFree cell_nverts failed");
    d_cell_nverts = nullptr;
  }
}

void Hydro2D::reclose_after_energy_edit(
    core::State& state,
    const core::Config& cfg,
    const HydroEOSContext* eos_ctx) const {
  if (state.rho.empty()) {
    return;
  }
  const HydroTableViews eos_views = select_hydro_table_views(eos_ctx);
  const bool use_two_temp = cfg.main.two_temperature;
  const std::int8_t* hydro_active = state.hydro_active_device_ptr();
  enforce_eos_closure(state, cfg, use_two_temp, eos_views, hydro_active);
  core::CellField1D cs_new;
  compute_cell_sound_speed(cs_new, state, cfg, use_two_temp, eos_views,
                           hydro_active);
  state.cs = std::move(cs_new);
}

void Hydro2D::apply_state_supply_boundary(core::State& state,
                                          const core::Config& cfg,
                                          const HydroEOSContext* eos_ctx,
                                          const bool accumulate_tally) const {
  if (!has_state_supply_bc(cfg) || state.rho.empty()) {
    return;
  }

  const HydroTableViews eos_views = select_hydro_table_views(eos_ctx);
  const bool use_two_temp = cfg.main.two_temperature;
  apply_state_supply_zonal_override(state, cfg, accumulate_tally);
  sync_state_supply_energy_from_temperature(state, cfg, use_two_temp, eos_views);
  enforce_eos_closure(state, cfg, use_two_temp, eos_views);
  tally_state_supply_delta_E(state, cfg, accumulate_tally);
  core::CellField1D cs_new;
  compute_cell_sound_speed(cs_new, state, cfg, use_two_temp, eos_views);
  state.cs = std::move(cs_new);
}

tenryu::coupling::HydroStepResult Hydro2D::lagrangian_step(
    core::State& state,
    const double dt,
    const core::Config& cfg,
    const double t_op,
    double* E_floor_injected,
    int* clamp_count,
    int* rho_clamp_count,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    cudaStream_t stream,
    const char* hydro_half,
    const HydroEOSContext* eos_ctx,
    const parallel::Reduction* reduction,
    MeshRegimeDeviceCache* mesh_regime_cache,
    tenryu::coupling::ProfileObservability* observability,
    double* r_momentum_source_impulse) const {
  tenryu::coupling::HydroStepResult result;
  if (r_momentum_source_impulse != nullptr) {
    *r_momentum_source_impulse = 0.0;
  }
  TENRYU_ASSERT(dt > 0.0, "Hydro2D::lagrangian_step requires dt > 0");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ", "Hydro2D requires 2D_RZ geometry");
  TENRYU_ASSERT(state.mesh.dim == 2, "Hydro2D requires 2D mesh");
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "Hydro2D requires matching node coordinate arrays");
  TENRYU_ASSERT(state.v_r.size() == state.v_z.size(),
                "Hydro2D requires matching node velocity arrays");
  TENRYU_ASSERT(state.x_r.size() == state.v_r.size(),
                "Hydro2D requires matching node arrays");
  TENRYU_ASSERT(state.mesh.topo.n_nodes == static_cast<int>(state.x_r.size()),
                "Hydro2D requires mesh/state node consistency");
  TENRYU_ASSERT(state.mesh.topo.n_cells == static_cast<int>(state.rho.size()),
                "Hydro2D requires mesh/state cell consistency");
  TENRYU_ASSERT(cfg.numerics.hydro.av_type == "vnr",
                "Numerics.hydro.av_type=\"riemann\" is supported only in 1D_SPH");
  const std::string& av_heat_to = cfg.numerics.hydro.av_heat_to;
  TENRYU_ASSERT(av_heat_to == "ion" || av_heat_to == "electron",
                "Numerics.hydro.av_heat_to must be \"ion\" or \"electron\" in v1.0");
  mesh::MeshSvecLazyScope svec_lazy_scope(state.mesh);
  const int q_heat_to_electron = (av_heat_to == "electron") ? 1 : 0;
  double local_E_floor = 0.0;
  int local_clamp_count = 0;
  int local_rho_clamp_count = 0;
  clear_av_max_diagnostic(state);

  if (state.rho.empty()) {
    return result;
  }

  validate_boundary_2d(cfg);
  const bool midpoint_v1_requested =
      cfg.numerics.hydro.time_integration ==
      core::HydroTimeIntegration::MidpointV1;
  const bool midpoint_v1_multiblock =
      midpoint_v1_requested && mesh::mesh_topo_is_multiblock(cfg.mesh);
  if (midpoint_v1_multiblock && state.step == 0) {
    static std::atomic_flag midpoint_multiblock_note_once =
        ATOMIC_FLAG_INIT;
    if (!midpoint_multiblock_note_once.test_and_set()) {
      core::log_info(
          "Hydro2D midpoint_v1 is stage-1 structured-only; "
          "multiblock/CSR remains on pc_v0");
    }
  }
  const bool midpoint_v1_special_path =
      midpoint_v1_requested && !midpoint_v1_multiblock &&
      (cfg.numerics.hydro.hllc_z_flux_2d_rz ||
       (state.mesh.button_center && state.mesh.button_center->enabled));
  if (midpoint_v1_special_path && state.step == 0) {
    static std::atomic_flag midpoint_special_path_note_once =
        ATOMIC_FLAG_INIT;
    if (!midpoint_special_path_note_once.test_and_set()) {
      core::log_info(
          "Hydro2D midpoint_v1 stage-1 compatible path excludes HLLC/button-center; "
          "the selected special path keeps its existing integration");
    }
  }
  const bool midpoint_v1 =
      midpoint_v1_requested && !midpoint_v1_multiblock &&
      !midpoint_v1_special_path;

  const int n_cells = static_cast<int>(state.rho.size());
  const int n_nodes = static_cast<int>(state.x_r.size());
  std::unique_ptr<CarrierHookContext> carrier_hook_context;
  const auto ensure_carrier_hook_context = [&]() -> CarrierHookContext& {
    if (!carrier_hook_context) {
      carrier_hook_context = std::make_unique<CarrierHookContext>(state);
    }
    return *carrier_hook_context;
  };
  const bool path_guard_enabled = i1b_path_guard_enabled();
  const int blocks_cells = (n_cells + 255) / 256;
  const int blocks_nodes = (n_nodes + 255) / 256;
  const core::State::LaunchWindow cw = state.owned_cell_window(n_cells);
  const core::State::LaunchWindow nw = state.owned_node_window(n_nodes);
  const core::State::LaunchWindow pw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  const bool axis_lagrangian_tangential =
      cfg.numerics.ale.axis_z_motion == "lagrangian_tangential";
  const bool state_supply_active = has_state_supply_bc(cfg);
  static const bool aw_axis_snap_env_enabled = [] {
    const char* const raw = std::getenv("TENRYU_AW_AXIS_SNAP");
    return raw == nullptr || std::string(raw) != "0";
  }();
  static const int drive_layer_diag_every = [] {
    const char* const raw = std::getenv("TENRYU_DRIVE_LAYER_DIAG");
    const int value =
        raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
    return value > 0 ? value : 0;
  }();
  static const int shadow_replay_step = [] {
    const char* const raw = std::getenv("TENRYU_SHADOW_REPLAY");
    if (raw == nullptr || raw[0] == '\0') {
      return -1;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return -1;
    }
    return static_cast<int>(parsed);
  }();
  static bool shadow_replay_printed = false;
  const AwAxisSlaveStructuredLines aw_axis_slave_lines =
      detect_aw_axis_slave_structured_lines(state, cfg);
  aw_axis_slave_theta0_active_ = aw_axis_slave_lines.theta0_active;
  aw_axis_slave_theta_pi_active_ = aw_axis_slave_lines.theta_pi_active;
  const bool aw_axis_slave_active =
      aw_axis_slave_theta0_active_ || aw_axis_slave_theta_pi_active_;
  const bool aw_axis_slave_multiblock_candidate =
      aw_axis_slave_enabled() &&
      cfg.numerics.hydro.aw_compatible_force_work &&
      state.mesh.topo.multiblock.has_value() &&
      mesh::mesh_topo_polar_tier_family(cfg.mesh);
  bool aw_axis_slave_multiblock_active = false;
  const int aw_axis_slave_first_i =
      aw_axis_slave_active &&
              (state.mesh.topo.node_flags[static_cast<std::size_t>(
                   state.mesh.topo.node_index(0, 0))] &
               mesh::NODE_CENTER) != 0U
          ? 1
          : 0;
  if (aw_axis_slave_active &&
      aw_axis_slave_kappa_saved_.size() != state.x_r.size()) {
    aw_axis_slave_kappa_saved_.reset(state.x_r.size());
  }
  mesh_trace::trace_start(state, cfg, dt);
  mesh_trace::trace_cell0_geometry(state, cfg, "start");

  const HydroTableViews eos_views = select_hydro_table_views(eos_ctx);

  std::int8_t* d_hydro_active =
      upload_hydro_active("h2d:hydro_active", state.hydro_active);
  std::int8_t* d_hydro_force_active = d_hydro_active;
  std::vector<std::int8_t> central_macro_effective_active;
  const std::vector<std::int8_t>* hydro_force_active_host = &state.hydro_active;
  std::vector<std::int8_t> hydro_active_restore;
  bool hydro_active_overridden = false;
  const auto restore_hydro_active = [&]() {
    if (hydro_active_overridden) {
      state.hydro_active = std::move(hydro_active_restore);
      state.note_hydro_active_host_write();
      hydro_active_overridden = false;
    }
  };
  std::uint8_t* d_cell_nverts =
      upload_cell_nverts_if_nonquad("h2d:cell_nverts", state);
  std::uint8_t* d_node_flags =
      upload_node_flags_if_constraints("h2d:node_flags", state);
  MeshRegimeDeviceCache local_mesh_regime_cache;
  MeshRegimeDeviceCache* active_mesh_regime_cache =
      mesh_regime_cache != nullptr ? mesh_regime_cache : &local_mesh_regime_cache;
  if (part.n_ranks > 1 && bufs != nullptr) {
    // OPEN-HYDRO-CORNER: rho/Pe/Pi/Te/Ti/zbar joined the entry exchange so
    // ghost cells enter the step bitwise-equal to their owner's end-of-step
    // state; the ghost-inclusive density/closure recompute below then keeps
    // them owner-consistent within the step (zbar has no in-step hydro
    // producer, so the exchange is its only ghost refresh on hydro-only
    // decks).
    double* field_ptrs[10] = {state.ee.data(),   state.ei.data(),
                              state.mass.data(), state.Qvisc.data(),
                              state.rho.data(),  state.Pe.data(),
                              state.Pi.data(),   state.Te.data(),
                              state.Ti.data(),   state.zbar.data()};
    parallel::exchange_cell_fields(part, *bufs, field_ptrs, 10, n_cells, stream,
                                   1);
    if (d_hydro_active != nullptr) {
      std::int8_t* hydro_active_ptr = d_hydro_active;
      parallel::exchange_int8_fields(part, *bufs, &hydro_active_ptr, 1, n_cells,
                                     stream, 1);
      // Write back halo-exchanged hydro_active to host for AV Q computation.
      cuda_check(cudaMemcpy(state.hydro_active.data(), hydro_active_ptr,
                            state.hydro_active.size() * sizeof(std::int8_t),
                            cudaMemcpyDeviceToHost),
                 "Hydro2D: cudaMemcpy hydro_active D2H failed");
    }
  }
  const bool mesh_attr_active =
      diagnostics::mesh_attribution::begin_step(state, cfg, state.t, hydro_half);
  const auto& mesh_forensics_cfg =
      cfg.numerics.diagnostics.mesh_degeneracy_forensics;
  const bool mesh_forensics_active =
      mesh_forensics_cfg.enabled ||
      mesh_forensics_cfg.velocity_history_enabled;
  const auto emit_mesh_attr_failure = [&](const tenryu::coupling::HydroStepResult& failure,
                                          const double* candidate_r = nullptr,
                                          const double* candidate_z = nullptr) {
    if (mesh_attr_active) {
      diagnostics::mesh_attribution::emit_failure_record(
          state,
          cfg,
          failure.first_failing_cell,
          failure.first_failing_corner,
          candidate_r,
          candidate_z);
    }
  };
  int* d_rho_clamp_count = nullptr;
  const int zero_i = 0;
  d_rho_clamp_count = static_cast<int*>(
      core::device_scratch_acquire("h2d:rho_clamp_count", sizeof(int)));
  cuda_check(cudaMemcpy(d_rho_clamp_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Hydro2D: init rho_clamp_count failed");
  std::uint8_t* d_node_active = nullptr;
  const auto cleanup_precommit_retry_buffers = [&]() {
    restore_hydro_active();
    if (d_node_active != nullptr) {
      d_node_active = nullptr;
    }
    d_rho_clamp_count = nullptr;
    if (d_hydro_force_active != nullptr && d_hydro_force_active != d_hydro_active) {
      d_hydro_force_active = nullptr;
    }
    if (d_hydro_active != nullptr) {
      d_hydro_active = nullptr;
    }
    if (d_cell_nverts != nullptr) {
      d_cell_nverts = nullptr;
    }
    if (d_node_flags != nullptr) {
      d_node_flags = nullptr;
    }
  };

  result = refresh_geometry_and_density(
      state,
      cfg,
      tenryu::coupling::HydroFailureStage::StepStart,
      d_rho_clamp_count,
      observability);
  if (!result.admissible) {
    emit_mesh_attr_failure(result);
    cleanup_precommit_retry_buffers();
    return result;
  }
  const Ring7SeamRemapResult ring7_remap =
      apply_ring7_seam_quotient_remap(state,
                                      cfg,
                                      eos_ctx,
                                      state.v_r.data(),
                                      state.v_z.data(),
                                      dt,
                                      reduction,
                                      "pre_step");
  if (ring7_remap.dt_reduction_required) {
    result.admissible = false;
    result.retry_required = true;
    result.reason = "mesh_quality_rz_volume";
    result.stage = tenryu::coupling::HydroFailureStage::StepStart;
    result.retry_action = tenryu::coupling::RetryActionHint::ReduceDtOnly;
    result.min_metric = ring7_remap.min_eta_V;
    result.first_failing_cell = ring7_remap.request_cell;
    result.suggested_dt = ring7_remap.suggested_dt;
    result.trial_scale = dt > 0.0 ? ring7_remap.suggested_dt / dt : 0.0;
    result.rz_volume_failed = true;
    cleanup_precommit_retry_buffers();
    return result;
  }
  MeshDeviceCache mesh_cache{"h2d:meshcache:svec_r", "h2d:meshcache:svec_z", "h2d:meshcache:area"};
  upload_mesh_cache(state, mesh_cache);
  ensure_corner_mass_2d(state,
                        d_cell_nverts,
                        static_cast<int>(
                            cfg.numerics.hydro.corner_mass_convention),
                        cfg.numerics.hydro.subzonal_pressure_enabled,
                        recompute_corner_mass_each_step(cfg),
                        corner_mass_lagrangian_invariant_enabled(cfg),
                        cfg.numerics.hydro.aw_compatible_force_work,
                        mesh::mesh_topo_polar_tier_family(cfg.mesh));
  if (aw_axis_slave_multiblock_candidate) {
    const mesh::MultiBlockTopology* const topology =
        &*state.mesh.topo.multiblock;
    if (aw_axis_slave_multiblock_topology_ != topology) {
      const AwAxisSlaveMasterList master =
          build_aw_axis_slave_master_list_multiblock(state, cfg);
      TENRYU_ASSERT(master.p.size() == master.q.size(),
                    "AW axis multiblock master list size mismatch");
      aw_axis_slave_p_.reset(master.p.size());
      aw_axis_slave_q_.reset(master.q.size());
      aw_axis_slave_multiblock_kappa_saved_.reset(master.p.size());
      aw_axis_slave_p_.copy_from_host(master.p);
      aw_axis_slave_q_.copy_from_host(master.q);
      aw_axis_slave_multiblock_topology_ = topology;
    }
    aw_axis_slave_multiblock_active = !aw_axis_slave_p_.empty();
  }
  central_pseudo_core::maybe_absorb_first_active_ring(state, cfg);
  pole_angular_derefine::ensure_built(state, cfg);
  ale::csr_optionb_canonicalize_corner_mass_basis(state, cfg);
  ale::record_estep_trace_boundary(
      state, cfg, part, reduction, "step_start", dt, t_op);
  const bool trace_mesh_motion = mesh_trace::enabled(state, cfg);
  const std::uint64_t subzonal_hash_before =
      trace_mesh_motion ? mesh_trace::coordinate_hash(state) : 0ULL;
  ensure_hourglass_subzonal_masses_2d(state, cfg, false);
  ale::csr_optionb_canonicalize_corner_mass_basis(state, cfg);
  if (trace_mesh_motion) {
    const std::uint64_t subzonal_hash_after =
        mesh_trace::coordinate_hash(state);
    mesh_trace::trace_corner_mass_canary(state,
                                         cfg,
                                         subzonal_hash_before,
                                         subzonal_hash_after);
  }
  central_pseudo_core::aggregate_state(state, cfg, "hydro_step_start", false);
  pole_angular_derefine::aggregate_state(state, cfg, "hydro_step_start", false);
  conservation_audit::emit_stage(state, "hydro_step_start_after_aggregate");
  central_macro_effective_active = make_central_macro_effective_active(state);
  if (!central_macro_effective_active.empty()) {
    d_hydro_force_active = upload_hydro_active("h2d:hydro_force_active",
                                               central_macro_effective_active);
    hydro_force_active_host = &central_macro_effective_active;
    hydro_active_restore = state.hydro_active;
    state.hydro_active = central_macro_effective_active;
    state.note_hydro_active_host_write();
    hydro_active_overridden = true;
  }
  if (!cfg.numerics.hydro.regime_aware_corner_j_guard_enabled) {
    active_mesh_regime_cache->invalidate();
  } else if (!active_mesh_regime_cache->valid()) {
    classify_mesh_regimes(state,
                          dt,
                          cfg.numerics.hydro.axis_guard_band_cells,
                          *active_mesh_regime_cache,
                          d_hydro_force_active,
                          cfg.numerics.has_physical_rz_axis);
  }

  const bool use_two_temp = cfg.main.two_temperature;
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const std::size_t n_cell_mat =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  const bool per_material_hydro_enabled =
      use_two_temp && cfg.numerics.materials.per_material_conservation_enabled &&
      n_mat > 0 && state.mass_per_material.size() == n_cell_mat &&
      state.Ee_per_material.size() == n_cell_mat &&
      state.Ei_per_material.size() == n_cell_mat &&
      state.volFrac.size() == n_cell_mat;
  if (cfg.numerics.materials.per_material_conservation_enabled) {
    TENRYU_ASSERT(per_material_hydro_enabled,
                  "Hydro2D per-material mode requires 2T and complete per-material arrays");
  }
  const bool compatible_energy_mode =
      midpoint_v1 || compatible::compatible_force_work_enabled(cfg);
  TENRYU_ASSERT(!central_pseudo_core::configured(cfg) || compatible_energy_mode,
                "central pseudo-core Exp1 requires compatible force/work hydro");
  const bool cap_energy_audit = cap_energy_audit_enabled();
  const bool i1b_spurious_sensor = i1b_spurious_sensor_enabled();
  TENRYU_ASSERT(!(compatible_energy_mode && per_material_hydro_enabled),
                "Hydro2D compatible force/work energy update does not support per-material conservation");
  if (per_material_hydro_enabled) {
    bool any_table_backed = false;
    for (const auto& material : cfg.materials.materials) {
      any_table_backed =
          any_table_backed || material.eos_tables != nullptr || material.eos_model != "ideal_gas";
    }
    if (any_table_backed) {
      TENRYU_ASSERT(eos_ctx != nullptr,
                    "Hydro2D per-material mode requires HydroEOSContext for table-backed materials");
      TENRYU_ASSERT(eos_ctx->n_materials >= n_mat,
                    "Hydro2D per-material EOS context material count mismatch");
    }
  }
  const auto refresh_or_enforce_closure = [&](const bool force_invalidate_all) {
    if (per_material_hydro_enabled) {
      per_material::refresh_per_material_derived_cell_fields(
          state, cfg, eos_ctx, force_invalidate_all);
    } else {
      enforce_eos_closure(state, cfg, use_two_temp, eos_views,
                          d_hydro_force_active);
    }
  };
  std::unique_ptr<core::Config> midpoint_stage_cfg;
  core::CellField1D midpoint_stage_ee{"h2d:step:midpoint_stage_ee"};
  core::CellField1D midpoint_stage_ei{"h2d:step:midpoint_stage_ei"};
  if (midpoint_v1) {
    midpoint_stage_cfg = std::make_unique<core::Config>(cfg);
    midpoint_stage_cfg->numerics.hydro.eos_writeback = false;
    midpoint_stage_ee.reset(n_cells);
    midpoint_stage_ei.reset(n_cells);
  }
  const auto with_midpoint_stage_energy = [&](auto&& operation) {
    TENRYU_ASSERT(midpoint_v1,
                  "midpoint stage-energy scope requires midpoint_v1");
    std::swap(state.ee, midpoint_stage_ee);
    std::swap(state.ei, midpoint_stage_ei);
    try {
      operation();
    } catch (...) {
      std::swap(state.ei, midpoint_stage_ei);
      std::swap(state.ee, midpoint_stage_ee);
      throw;
    }
    std::swap(state.ei, midpoint_stage_ei);
    std::swap(state.ee, midpoint_stage_ee);
  };
  const auto refresh_midpoint_stage_closure = [&]() {
    TENRYU_ASSERT(midpoint_stage_cfg != nullptr,
                  "midpoint stage EOS config is not initialized");
    with_midpoint_stage_energy([&]() {
      enforce_eos_closure(state, *midpoint_stage_cfg, use_two_temp, eos_views,
                          d_hydro_force_active);
    });
  };
  const auto compute_sound_speed_for_step = [&](core::CellField1D& cs_field) {
    if (per_material_hydro_enabled) {
      cs_field.reset(n_cells);
      cuda_check(cudaMemcpy(cs_field.data(),
                            state.cs.data(),
                            static_cast<std::size_t>(n_cells) * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "Hydro2D: copy per-material projected sound speed failed");
    } else {
      compute_cell_sound_speed(cs_field, state, cfg, use_two_temp, eos_views,
                               d_hydro_force_active);
    }
  };
  const bool apply_1t_energy_renorm =
      !midpoint_v1 && !use_two_temp && !has_pressure_boundary(cfg);
  // Option-C staged intra-step ghost refreshes (2D analog of the 1D scheme,
  // design doc mpi_w2_hydro1d_conversion_spec.md): derived caches exchanged
  // after each derive block so stencil consumers read exact ghosts.
  const auto exchange_cell_ghosts_2d = [&](std::initializer_list<double*> fields) {
    if (part.n_ranks <= 1 || bufs == nullptr) {
      return;
    }
    double* ptrs[8];
    int n_ptrs = 0;
    for (double* ptr : fields) {
      if (ptr != nullptr) {
        ptrs[n_ptrs++] = ptr;
      }
    }
    if (n_ptrs > 0) {
      parallel::exchange_cell_fields(part, *bufs, ptrs, n_ptrs, n_cells,
                                     stream, 1);
    }
  };
  const auto allreduce_sum_scalar_2d = [&](const double value) -> double {
    if (part.n_ranks <= 1) {
      return value;
    }
    const parallel::Reduction reduction(part.n_ranks);
    return reduction.allreduce_sum(value);
  };
  const auto exchange_mid_nodes_2d = [&]() {
    if (part.n_ranks <= 1 || bufs == nullptr) {
      return;
    }
    double* node_ptrs[4] = {state.x_r.data(), state.x_z.data(),
                            state.v_r.data(), state.v_z.data()};
    parallel::exchange_node_fields(part, *bufs, node_ptrs, 4, n_nodes, stream,
                                   2);
  };
  refresh_or_enforce_closure(false);
  if (cfg.numerics.hydro.hllc_z_flux_2d_rz) {
    TENRYU_ASSERT(cfg.numerics.hydro.total_energy_remap_2d_rz,
                  "Hydro2D HLLC z flux requires total_energy_remap_2d_rz=true");
    const HllcZFlux2DRZResult hllc_result =
        apply_hllc_z_flux_2d_rz(state,
                                cfg,
                                dt,
                                d_hydro_active,
                                part.rank,
                                hydro_half);
    local_E_floor += std::max(hllc_result.E_floor_injected, 0.0);
    local_clamp_count += std::max(hllc_result.clamp_count, 0);
    local_rho_clamp_count += std::max(hllc_result.rho_clamp_count, 0);

    (void)state_supply_active;
    refresh_or_enforce_closure(false);
    core::CellField1D cs_hllc;
    compute_sound_speed_for_step(cs_hllc);
    state.cs = std::move(cs_hllc);

    if (state.vol_prev_hydro.size() != state.vol.size()) {
      state.vol_prev_hydro.reset(state.vol.size());
    }
    cuda_check(cudaMemcpy(state.vol_prev_hydro.data(),
                          state.vol.data(),
                          static_cast<std::size_t>(n_cells) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D HLLC: copy previous hydro volume snapshot failed");
    state.dt_prev_hydro = dt;
    if (mesh_attr_active) {
      diagnostics::mesh_attribution::end_step_success(state, cfg);
    }
    if (E_floor_injected != nullptr) {
      *E_floor_injected += std::max(local_E_floor, 0.0);
    }
    if (clamp_count != nullptr) {
      *clamp_count += std::max(local_clamp_count, 0);
    }
    if (rho_clamp_count != nullptr) {
      *rho_clamp_count += std::max(local_rho_clamp_count, 0);
    }
    cleanup_precommit_retry_buffers();
    return result;
  }
  core::CellField1D Te_n_for_qei{"h2d:step:Te_n_for_qei"};
  core::CellField1D Ti_n_for_qei{"h2d:step:Ti_n_for_qei"};
  if (use_two_temp && !per_material_hydro_enabled &&
      cfg.numerics.hydro.qei_evaluate_at_t_n) {
    Te_n_for_qei.reset(n_cells);
    Ti_n_for_qei.reset(n_cells);
    cuda_check(cudaMemcpy(Te_n_for_qei.data(), state.Te.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D: copy Te_n_for_qei failed");
    cuda_check(cudaMemcpy(Ti_n_for_qei.data(), state.Ti.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D: copy Ti_n_for_qei failed");
  }
  const double E_before = apply_1t_energy_renorm
                              ? allreduce_sum_scalar_2d(
                                    diagnostics::compute_energy_budget_2d(state).E_total)
                              : 0.0;
  const bool total_energy_identity_check =
      cfg.numerics.hydro.total_energy_identity_check;
  const diagnostics::EnergyTotals total_energy_identity_before =
      total_energy_identity_check
          ? diagnostics::compute_energy_totals_2d(state)
          : diagnostics::EnergyTotals{};
  double total_energy_identity_W_ext = 0.0;
  const bool shell_drive_sym_diag_active =
      shell_drive_sym_diag_due(state, cfg);
  diagnostics::EnergyTotals shell_drive_sym_energy_before{};
  if (shell_drive_sym_diag_active) {
    shell_drive_sym_energy_before = diagnostics::compute_energy_totals_2d(state);
  }

  d_node_active = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "h2d:node_active", n_nodes * sizeof(std::uint8_t)));
  cuda_check(cudaMemset(d_node_active, 0, n_nodes * sizeof(std::uint8_t)),
             "Hydro2D: cudaMemset node_active failed");
  launch_compute_node_activity_2d(d_node_active, d_hydro_force_active, state);
  sync_kernel("Hydro2D compute_node_activity kernel failed");

  core::NodeField1D node_mass{"h2d:step:node_mass"};
  node_mass.reset(n_nodes);
  launch_compute_node_mass_2d(
      node_mass.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  sync_kernel("Hydro2D compute_node_mass kernel failed");
  if (cfg.numerics.hydro.rz_momentum_scheme_id == 1) {
    launch_compute_node_planar_mass_2d(
        state, d_cell_nverts, d_hydro_force_active, blocks_cells);
    sync_kernel("Hydro2D compute_node_planar_mass kernel failed");
  }
  mesh_trace::trace_activity(state,
                             cfg,
                             d_node_active,
                             node_mass,
                             d_hydro_force_active);

  core::CellField1D cs_n{"h2d:step:cs_n"};
  compute_sound_speed_for_step(cs_n);
  exchange_cell_ghosts_2d({state.rho.data(), state.vol.data(), state.Te.data(),
                           state.Ti.data(), state.Pe.data(), state.Pi.data(),
                           cs_n.data()});

  core::CellField1D dVdt_n{"h2d:step:dVdt_n"};
  core::CellField1D div_u_n{"h2d:step:div_u_n"};
  core::CellField1D dl_n{"h2d:step:dl_n"};
  dVdt_n.reset(n_cells);
  div_u_n.reset(n_cells);
  dl_n.reset(n_cells);
  // OPEN-HYDRO-CORNER fix: the AV input chain must cover the ghost ring
  // (the corner-force scatter reads Q there). Serial identity window.
  const core::State::LaunchWindow av_fw =
      state.owned_cell_window_ghost(n_cells, state.mesh.topo.nz);
  detail::launch_compute_cell_dVdt_2d(
      dVdt_n.data(), state.v_r.data(), state.v_z.data(), mesh_cache.Svec_r.data(),
      mesh_cache.Svec_z.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  compute_cell_div_u_kernel<<<av_fw.blocks(), 256>>>(
      div_u_n.data(), dVdt_n.data(), state.vol.data(), av_fw.begin, av_fw.end, n_cells);
  compute_cell_length_scale_kernel<<<av_fw.blocks(), 256>>>(
      dl_n.data(), mesh_cache.area.data(), av_fw.begin, av_fw.end, n_cells);
  sync_kernel("Hydro2D compute dVdt/div_u/dl kernels failed");

  const bool av_eos_aware_active =
      cfg.numerics.hydro.av_eos_aware &&
      (eos_views.tab_total.n_rho > 0 || eos_views.tab_ion.n_rho > 0 ||
       eos_views.tab_ele.n_rho > 0 || eos_views.spline_total.n_rho > 0 ||
       eos_views.jet_total.n_rho > 0);
  ArtificialViscosity av(cfg.numerics.hydro.av_linear, cfg.numerics.hydro.av_quadratic,
                         1.0, 1.0, av_eos_aware_active,
                         cfg.numerics.hydro.av_eos_gamma1_ref,
                         cfg.numerics.hydro.av_eos_boost_max,
                         ArtificialViscosityType::Vnr);
  core::CellField1D Qvisc_per_material{"h2d:step:Qvisc_per_material"};
  if (per_material_hydro_enabled) {
    Qvisc_per_material.reset(n_cell_mat);
  }
  const auto compute_artificial_viscosity_2d =
      [&](const core::CellField1D& dl,
          const core::CellField1D& div_u,
          const core::CellField1D& cs) {
        if (cfg.numerics.hydro.av_model != core::AvModel::ScalarVnrLegacy) {
          cuda_check(cudaMemset(state.Qvisc.data(),
                                0,
                                static_cast<std::size_t>(n_cells) * sizeof(double)),
                     "Hydro2D zero compatible-mode scalar Qvisc failed");
          if (per_material_hydro_enabled) {
            cuda_check(cudaMemset(Qvisc_per_material.data(),
                                  0,
                                  n_cell_mat * sizeof(double)),
                       "Hydro2D zero compatible-mode per-material Qvisc failed");
          }
          return;
        }
        if (per_material_hydro_enabled) {
          av.compute_q_per_material_2d(
              state.Qvisc, Qvisc_per_material, state, dl, div_u, cfg, eos_ctx,
              *hydro_force_active_host);
        } else {
          av.compute_Q_2d(state.Qvisc, state.rho, dl, div_u, state.Pe, state.Pi, cs,
                          *hydro_force_active_host);
        }
      };
  compute_artificial_viscosity_2d(dl_n, div_u_n, cs_n);
  qcap_apply_in_place(state.Qvisc,
                      cfg,
                      state,
                      d_hydro_force_active,
                      per_material_hydro_enabled ? &Qvisc_per_material : nullptr);
  exchange_cell_ghosts_2d({state.Qvisc.data()});
  diagnostics::mesh_diag::dump_hydro_cells(state,
                                           cfg,
                                           "pre_hydro",
                                           dt,
                                           state.x_r,
                                           state.x_z,
                                           state.v_r,
                                           state.v_z,
                                           &cs_n,
                                           &div_u_n,
                                           "not_recorded_in_hydro",
                                           hydro_half);
  ale_diag::emit_identity_field_diag(
      state, cfg, node_mass, "pre_hydro", t_op, dt, part.rank, &cs_n);

  core::NodeField1D r_old{"h2d:step:r_old"};
  core::NodeField1D z_old{"h2d:step:z_old"};
  core::NodeField1D ur_old{"h2d:step:ur_old"};
  core::NodeField1D uz_old{"h2d:step:uz_old"};
  core::CellField1D V_old{"h2d:step:V_old"};
  core::CellField1D e_old{"h2d:step:e_old"};
  core::CellField1D ei_old{"h2d:step:ei_old"};
  core::CellField1D Pe_old{"h2d:step:Pe_old"};
  core::CellField1D Pi_old{"h2d:step:Pi_old"};
  r_old.reset(n_nodes);
  z_old.reset(n_nodes);
  ur_old.reset(n_nodes);
  uz_old.reset(n_nodes);
  V_old.reset(n_cells);
  e_old.reset(n_cells);
  Pe_old.reset(n_cells);
  Pi_old.reset(n_cells);
  if (use_two_temp || midpoint_v1) {
    ei_old.reset(n_cells);
  }

  cuda_check(cudaMemcpy(r_old.data(), state.x_r.data(), n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy r_old failed");
  cuda_check(cudaMemcpy(z_old.data(), state.x_z.data(), n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy z_old failed");
  cuda_check(cudaMemcpy(ur_old.data(), state.v_r.data(), n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy ur_old failed");
  cuda_check(cudaMemcpy(uz_old.data(), state.v_z.data(), n_nodes * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy uz_old failed");
  emit_origin_trace_step_begin(state, uz_old.data());
  cuda_check(cudaMemcpy(V_old.data(), state.vol.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy V_old failed");
  cuda_check(cudaMemcpy(e_old.data(), state.ee.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy e_old failed");
  cuda_check(cudaMemcpy(Pe_old.data(), state.Pe.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy Pe_old failed");
  cuda_check(cudaMemcpy(Pi_old.data(), state.Pi.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy Pi_old failed");
  if (use_two_temp || midpoint_v1) {
    cuda_check(cudaMemcpy(ei_old.data(), state.ei.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy ei_old failed");
  }

  core::NodeField1D attr_pressure_n_r{"h2d:step:attr_pressure_n_r"};
  core::NodeField1D attr_pressure_n_z{"h2d:step:attr_pressure_n_z"};
  core::NodeField1D attr_q_n_r{"h2d:step:attr_q_n_r"};
  core::NodeField1D attr_q_n_z{"h2d:step:attr_q_n_z"};
  core::CellField1D pq_n{"h2d:step:pq_n"};
  pq_n.reset(n_cells);
  if (compatible_energy_mode) {
    build_cell_pressure_kernel<<<pw.blocks(), 256>>>(
        pq_n.data(), state.Pe.data(), state.Pi.data(), d_hydro_force_active,
        pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D build pressure force-source n kernel failed");
  } else {
    build_cell_pq_kernel<<<pw.blocks(), 256>>>(
        pq_n.data(), state.Pe.data(), state.Pi.data(), state.Qvisc.data(),
        d_hydro_force_active, pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D build_cell_pq n kernel failed");
  }

  core::NodeField1D a_n_r{"h2d:step:a_n_r"};
  core::NodeField1D a_n_z{"h2d:step:a_n_z"};
  a_n_r.reset(n_nodes);
  a_n_z.reset(n_nodes);
  compute_acceleration_2d(a_n_r, a_n_z, state, cfg, t_op, mesh_cache, pq_n,
                          d_node_active, d_node_flags, node_mass, d_hydro_force_active,
                          eos_ctx, true, true, true, &cs_n, true, nullptr, nullptr,
                          0.0, false, nullptr, true,
                          aw_axis_slave_theta0_active_,
                          aw_axis_slave_theta_pi_active_, nullptr, nullptr,
                          midpoint_v1,
                          midpoint_v1 ? &state.Qvisc : nullptr, "dyn_a");
  static const bool axiswall_diag_enabled = [] {
    const char* const raw = std::getenv("TENRYU_AXISWALL_DIAG");
    return raw != nullptr && raw[0] != '\0';
  }();
  if (axiswall_diag_enabled) {
    TENRYU_ASSERT(
        compatible_energy_mode,
        "TENRYU_AXISWALL_DIAG requires compatible force/work assembly");
    const core::NodeField1D& axiswall_node_mass =
        cfg.numerics.hydro.rz_momentum_scheme_id == 1
            ? state.node_planar_mass
            : node_mass;
    emit_axiswall_diag(state,
                       cfg,
                       t_op,
                       dt,
                       r_old,
                       z_old,
                       ur_old,
                       uz_old,
                       axiswall_node_mass,
                       d_node_active,
                       *hydro_force_active_host);
  }
  mesh_trace::trace_post_pressure_accel(state,
                                        cfg,
                                        a_n_r,
                                        a_n_z,
                                        node_mass,
                                        d_hydro_force_active);
  core::NodeField1D a_hg_n_r{"h2d:step:a_hg_n_r"};
  core::NodeField1D a_hg_n_z{"h2d:step:a_hg_n_z"};
  core::NodeField1D a_nonhg_n_r{"h2d:step:a_nonhg_n_r"};
  core::NodeField1D a_nonhg_n_z{"h2d:step:a_nonhg_n_z"};
  if (subzonal_hourglass_enabled(cfg)) {
    if (midpoint_v1) {
      a_nonhg_n_r.reset(n_nodes);
      a_nonhg_n_z.reset(n_nodes);
      cuda_check(cudaMemcpy(a_nonhg_n_r.data(), a_n_r.data(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy non-hourglass a_n_r failed");
      cuda_check(cudaMemcpy(a_nonhg_n_z.data(), a_n_z.data(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy non-hourglass a_n_z failed");
    }
    core::CellField1D pressure_n{"h2d:step:pressure_n"};
    pressure_n.reset(n_cells);
    build_cell_pressure_kernel<<<pw.blocks(), 256>>>(
        pressure_n.data(), state.Pe.data(), state.Pi.data(),
        d_hydro_force_active, pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D build_cell_pressure n hourglass kernel failed");
    compute_anti_hourglass_acceleration_2d(a_hg_n_r,
                                           a_hg_n_z,
                                           nullptr,
                                           state,
                                           cfg,
                                           pressure_n,
                                           node_mass,
                                           a_n_r,
                                           a_n_z,
                                           d_hydro_force_active,
                                           dt,
                                           false,
                                           d_node_flags);
    mesh_trace::trace_post_hg_accel(state,
                                    cfg,
                                    a_hg_n_r,
                                    a_hg_n_z,
                                    pressure_n,
                                    d_hydro_force_active,
                                    dt);
    add_hourglass_acceleration_2d(a_n_r, a_n_z, a_hg_n_r, a_hg_n_z);
    // §22 contact must be last; hourglass leaked ~0.04 nm/step into the held gap.
    launch_apply_evac_contact_accel_constraints(
        state,
        a_n_r.data(),
        a_n_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data());
  }
  mesh_trace::trace_post_total_accel(state, cfg, a_n_r, a_n_z);
  if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
      state.carrier_node_class.empty()) { /* skip */
  } else {
    ensure_carrier_hook_context().apply_condensation(
        a_n_r.data(), a_n_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 1
            ? state.node_planar_mass.data()
            : node_mass.data());
  }
  core::NodeField1D u_half_r{"h2d:step:u_half_r"};
  core::NodeField1D u_half_z{"h2d:step:u_half_z"};
  core::NodeField1D predictor_work_v_r{"h2d:step:predictor_work_v_r"};
  core::NodeField1D predictor_work_v_z{"h2d:step:predictor_work_v_z"};
  core::CellField1D hourglass_work_n_erg{"h2d:step:hourglass_work_n_erg"};
  if (midpoint_v1) {
    u_half_r.reset(n_nodes);
    u_half_z.reset(n_nodes);
    // §22: the constraint must hold at consumption; modifier-chasing is insufficient.
    launch_apply_evac_contact_accel_constraints(
        state,
        a_n_r.data(),
        a_n_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data());
    launch_update_node_velocity_2d(
        u_half_r.data(), u_half_z.data(), ur_old.data(), uz_old.data(),
        a_n_r.data(), a_n_z.data(), d_node_active, d_node_flags, n_nodes,
        0.5 * dt, state, cfg);
    evacuated_cell_contact_probe(state, cfg, "post_vel_kernel");
    emit_origin_trace(state, "post_predictor_uhalf", u_half_z.data());
    predictor_work_v_r.reset(n_nodes);
    predictor_work_v_z.reset(n_nodes);
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        predictor_work_v_r.data() + nw.begin, ur_old.data() + nw.begin,
        u_half_r.data() + nw.begin, nw.count());
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        predictor_work_v_z.data() + nw.begin, uz_old.data() + nw.begin,
        u_half_z.data() + nw.begin, nw.count());
    sync_kernel("Hydro2D midpoint predictor velocity average failed");

    recompute_midpoint_corner_work_2d(
        state, cfg, predictor_work_v_r.data(), predictor_work_v_z.data(),
        d_hydro_force_active);
    cuda_check(cudaMemcpy(midpoint_stage_ee.data(), e_old.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint restore predictor ee failed");
    cuda_check(cudaMemcpy(midpoint_stage_ei.data(), ei_old.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint restore predictor ei failed");
    with_midpoint_stage_energy([&]() {
      apply_compatible_energy_work_2d(
          state, cfg, true, Pe_old, Pi_old, use_two_temp, 0.5 * dt,
          d_hydro_force_active, nullptr, nullptr);
      if (subzonal_hourglass_enabled(cfg) &&
          cfg.numerics.hydro.hourglass.compatible_work_enabled) {
        core::CellField1D pressure_n{"h2d:step:midpoint_pressure_n"};
        pressure_n.reset(n_cells);
        build_cell_pressure_kernel<<<blocks_cells, 256>>>(
            pressure_n.data(), Pe_old.data(), Pi_old.data(),
            d_hydro_force_active, pw.begin, pw.end, n_cells);
        compute_anti_hourglass_acceleration_2d(
            a_hg_n_r, a_hg_n_z, &hourglass_work_n_erg, state, cfg, pressure_n,
            node_mass, a_nonhg_n_r, a_nonhg_n_z, d_hydro_force_active, dt, true,
            d_node_flags, predictor_work_v_r.data(),
            predictor_work_v_z.data(), 0.5 * dt);
        apply_hourglass_work_2d(state, cfg, hourglass_work_n_erg, use_two_temp,
                                d_hydro_force_active, nullptr, nullptr);
      }
    });
  }
  const auto center_perturbation_scope =
      cfg.numerics.hydro.center_perturbation_diag_scope;
  bool center_perturbation_active = false;
  if (center_perturbation_scope != core::CenterPerturbationDiagScope::DISABLED) {
    const bool topology_matches =
        (center_perturbation_scope ==
             core::CenterPerturbationDiagScope::TRI_FAN_FIRST_RING &&
         is_tri_fan_center_perturbation_topology(state)) ||
        (center_perturbation_scope ==
             core::CenterPerturbationDiagScope::CENTROID_R_INNERMOST_BINS &&
         is_multiblock_center_perturbation_topology(state));
    center_perturbation_active =
        tri_fan_center_perturbation_history_due(state, cfg, dt) && topology_matches;
    if (!center_perturbation_active) {
      state.tri_fan_center_perturbation_diag.reset();
    }
  }
  const bool source_accel_diagnostics_active =
      mesh_attr_active || mesh_forensics_active || center_perturbation_active;
  if (source_accel_diagnostics_active) {
    core::CellField1D attr_pressure_n{"h2d:step:attr_pressure_n"};
    core::CellField1D attr_q_n{"h2d:step:attr_q_n"};
    attr_pressure_n.reset(n_cells);
    attr_q_n.reset(n_cells);
    build_cell_pressure_kernel<<<pw.blocks(), 256>>>(
        attr_pressure_n.data(), state.Pe.data(), state.Pi.data(),
        d_hydro_force_active, pw.begin, pw.end, n_cells);
    build_cell_q_kernel<<<cw.blocks(), 256>>>(
        attr_q_n.data(), state.Qvisc.data(), d_hydro_force_active, cw.begin,
        cw.end, n_cells);
    sync_kernel("Hydro2D mesh attribution build source pressure/Q failed");
    attr_pressure_n_r.reset(n_nodes);
    attr_pressure_n_z.reset(n_nodes);
    attr_q_n_r.reset(n_nodes);
    attr_q_n_z.reset(n_nodes);
    compute_acceleration_2d(attr_pressure_n_r, attr_pressure_n_z, state, cfg, t_op,
                            mesh_cache, attr_pressure_n, d_node_active, d_node_flags,
                            node_mass, d_hydro_force_active, eos_ctx, true, false, true,
                            &cs_n, false, nullptr, nullptr, 0.0, false, nullptr,
                            false, aw_axis_slave_theta0_active_,
                            aw_axis_slave_theta_pi_active_, nullptr, nullptr,
                            false, nullptr, "attr_p");
    compute_acceleration_2d(attr_q_n_r, attr_q_n_z, state, cfg, t_op, mesh_cache,
                            attr_q_n, d_node_active, d_node_flags, node_mass,
                            d_hydro_force_active, eos_ctx, false, false, false,
                            nullptr, true, nullptr, nullptr, 0.0, false, nullptr,
                            false, aw_axis_slave_theta0_active_,
                            aw_axis_slave_theta_pi_active_, nullptr, nullptr,
                            false, nullptr, "attr_q");
  }
  if (center_perturbation_active) {
    if (center_perturbation_scope ==
        core::CenterPerturbationDiagScope::TRI_FAN_FIRST_RING) {
      detail::accumulate_tri_fan_center_perturbation_diag(state,
                                                          cfg,
                                                          node_mass,
                                                          ur_old,
                                                          uz_old,
                                                          attr_pressure_n_r,
                                                          attr_pressure_n_z,
                                                          attr_q_n_r,
                                                          attr_q_n_z);
    } else if (center_perturbation_scope ==
               core::CenterPerturbationDiagScope::CENTROID_R_INNERMOST_BINS) {
      detail::accumulate_centroid_r_innermost_bins_perturbation_diag(
          state,
          cfg,
          node_mass,
          ur_old,
          uz_old,
          attr_pressure_n_r,
          attr_pressure_n_z,
          attr_q_n_r,
          attr_q_n_z);
    }
  }
  if (mesh_attr_active) {
    auto& attr = diagnostics::mesh_attribution::global_workspace();
    attr.clear_lagrangian_sources();
    attr.record_kinematic_carry(ur_old.data(), uz_old.data(), d_node_active, n_nodes,
                                0.5 * dt);
    attr.record_acceleration_displacement(
        diagnostics::mesh_attribution::MeshDeformSource::HydroPressure,
        attr_pressure_n_r.data(),
        attr_pressure_n_z.data(),
        d_node_active,
        n_nodes,
        0.25 * dt * dt);
    attr.record_acceleration_displacement(
        diagnostics::mesh_attribution::MeshDeformSource::ArtificialViscosity,
        attr_q_n_r.data(),
        attr_q_n_z.data(),
        d_node_active,
        n_nodes,
        0.25 * dt * dt);
  }

  if (!midpoint_v1) {
    u_half_r.reset(n_nodes);
    u_half_z.reset(n_nodes);
    // §22: the constraint must hold at consumption; modifier-chasing is insufficient.
    launch_apply_evac_contact_accel_constraints(
        state,
        a_n_r.data(),
        a_n_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data());
    launch_update_node_velocity_2d(
        u_half_r.data(), u_half_z.data(), ur_old.data(), uz_old.data(),
        a_n_r.data(), a_n_z.data(), d_node_active, d_node_flags,
        n_nodes, 0.5 * dt, state, cfg);
    evacuated_cell_contact_probe(state, cfg, "post_vel_kernel");
    emit_origin_trace(state, "post_predictor_uhalf", u_half_z.data());
  }
  core::NodeField1D& predictor_base_r =
      midpoint_v1 ? predictor_work_v_r : u_half_r;
  core::NodeField1D& predictor_base_z =
      midpoint_v1 ? predictor_work_v_z : u_half_z;
  apply_axis_motion_preflight(
      predictor_base_r, midpoint_v1 ? u_half_r.data() : nullptr, r_old, z_old,
      predictor_base_z, state.mesh.topo.nr, state.mesh.topo.nz, 0.5 * dt,
      cfg.numerics.hydro.axis_motion_floor_fraction);
  const auto predictor_pole_motion_overlay =
      tenryu::hydro::pole_angular_coarsen::build_motion_overlay(state, cfg);
  core::NodeField1D pos_half_r{"h2d:step:pos_half_r"};
  core::NodeField1D pos_half_z{"h2d:step:pos_half_z"};
  double* predictor_pos_r = predictor_base_r.data();
  double* predictor_pos_z = predictor_base_z.data();
  if (predictor_pole_motion_overlay.active) {
    pos_half_r.reset(n_nodes);
    cuda_check(cudaMemcpy(pos_half_r.data(), predictor_base_r.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy predictor position r velocity failed");
    predictor_pos_r = pos_half_r.data();
  }
  if (state_supply_active || axis_lagrangian_tangential ||
      predictor_pole_motion_overlay.active) {
    pos_half_z.reset(n_nodes);
    cuda_check(cudaMemcpy(pos_half_z.data(), predictor_base_z.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy predictor position z velocity failed");
    if (state_supply_active) {
      zero_state_supply_z_position_velocity(pos_half_z, cfg, state.mesh.topo.nr,
                                            state.mesh.topo.nz);
    }
    if (axis_lagrangian_tangential) {
      constrain_axis_z_position_velocity(state, cfg, pos_half_z, r_old, z_old,
                                         0.5 * dt, true);
    }
    predictor_pos_z = pos_half_z.data();
  }
  if (predictor_pole_motion_overlay.active) {
    apply_pole_motion_pilot_position_velocity(
        state,
        cfg,
        predictor_pole_motion_overlay,
        r_old,
        z_old,
        pos_half_r,
        pos_half_z,
        0.5 * dt,
        "predictor");
  }
  core::NodeField1D* predictor_pos_r_field =
      predictor_pole_motion_overlay.active ? &pos_half_r : &predictor_base_r;
  core::NodeField1D* predictor_pos_z_field =
      (state_supply_active || axis_lagrangian_tangential ||
       predictor_pole_motion_overlay.active)
          ? &pos_half_z
          : &predictor_base_z;
  apply_pole_axis_bbsw_order_constraint(state,
                                        cfg,
                                        t_op,
                                        pq_n,
                                        d_hydro_force_active,
                                        d_node_active,
                                        compatible_energy_mode,
                                        z_old,
                                        *predictor_pos_r_field,
                                        *predictor_pos_z_field,
                                        &u_half_r,
                                        &u_half_z,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        0.5 * dt);
  if (aw_axis_slave_active) {
    launch_apply_aw_axis_velocity_slave_2d(
        u_half_r.data(),
        u_half_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        true);
    if (drive_layer_diag_every > 0 && state.step <= 10) {
      const int i = state.mesh.topo.nr;
      const int stride = state.mesh.topo.nz + 1;
      const auto emit_axis_slave_dbg =
          [&](const char* const pole, const int p, const int q) {
            double uz_p = 0.0;
            double ur_q = 0.0;
            double uz_q = 0.0;
            double kappa = 0.0;
            cuda_check(cudaMemcpy(&uz_p,
                                  u_half_z.data() + p,
                                  sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "Hydro2D axis slave dbg copy uzP failed");
            cuda_check(cudaMemcpy(&ur_q,
                                  u_half_r.data() + q,
                                  sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "Hydro2D axis slave dbg copy urQ failed");
            cuda_check(cudaMemcpy(&uz_q,
                                  u_half_z.data() + q,
                                  sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "Hydro2D axis slave dbg copy uzQ failed");
            cuda_check(cudaMemcpy(&kappa,
                                  aw_axis_slave_kappa_saved_.data() + p,
                                  sizeof(double),
                                  cudaMemcpyDeviceToHost),
                       "Hydro2D axis slave dbg copy kappa failed");
            std::ostringstream os;
            os << "[axis-slave-dbg] step=" << state.step
               << " pole=" << pole
               << " i=" << i
               << " P=" << p
               << " uzP=" << uz_p
               << " Q=" << q
               << " urQ=" << ur_q
               << " uzQ=" << uz_q
               << " kappa=" << kappa;
            core::log_info(os.str());
          };
      if (aw_axis_slave_theta0_active_) {
        const int p = i * stride;
        emit_axis_slave_dbg("theta0", p, p + 1);
      }
      if (aw_axis_slave_theta_pi_active_) {
        const int p = i * stride + state.mesh.topo.nz;
        emit_axis_slave_dbg("theta_pi", p, p - 1);
      }
    }
    if (predictor_pos_r_field->data() != u_half_r.data() ||
        predictor_pos_z_field->data() != u_half_z.data()) {
      launch_apply_aw_axis_velocity_slave_2d(
          predictor_pos_r_field->data(),
          predictor_pos_z_field->data(),
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_kappa_saved_.data(),
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active_,
          aw_axis_slave_theta_pi_active_,
          false);
    }
    sync_kernel("Hydro2D predictor AW axis velocity slave failed");
  }
  if (aw_axis_slave_multiblock_active) {
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        u_half_r.data(),
        u_half_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        true);
    if (predictor_pos_r_field->data() != u_half_r.data() ||
        predictor_pos_z_field->data() != u_half_z.data()) {
      launch_apply_aw_axis_velocity_slave_2d_multiblock(
          predictor_pos_r_field->data(),
          predictor_pos_z_field->data(),
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_multiblock_kappa_saved_.data(),
          aw_axis_slave_p_.data(),
          aw_axis_slave_q_.data(),
          static_cast<int>(aw_axis_slave_p_.size()),
          false);
    }
    sync_kernel("Hydro2D predictor multiblock AW axis velocity slave failed");
  }
  conservation_audit::emit_stage(state, "bbsw_predictor_constraint");
  predictor_pos_r = predictor_pos_r_field->data();
  predictor_pos_z = predictor_pos_z_field->data();
  // Trial-stage projection; heat is owned by the post-boundary finalizer (consultation #23 §7.4).
  const double* const predictor_commit_node_mass =
      cfg.numerics.hydro.rz_momentum_scheme_id == 0
          ? node_mass.data()
          : state.node_planar_mass.data();
  launch_apply_evac_contact_velocity_constraints(
      state, u_half_r.data(), u_half_z.data(), predictor_commit_node_mass,
      state.x_r.data(), state.x_z.data(), cs_n.data(),
      d_hydro_force_active,
      dt,
      nullptr,
      cfg.numerics.ale.evacuated_cell.closure_contact
          .mortar_position_drift_beta);
  copy_array_kernel<<<nw.blocks(), 256>>>(state.v_r.data() + nw.begin,
                                          u_half_r.data() + nw.begin,
                                          nw.count());
  copy_array_kernel<<<nw.blocks(), 256>>>(state.v_z.data() + nw.begin,
                                          u_half_z.data() + nw.begin,
                                          nw.count());
  evacuated_cell_contact_probe(state, cfg, "post_vcopy");
  apply_axis_contact_guard("predictor", cfg, r_old, predictor_pos_r, node_mass,
                           d_node_active, d_node_flags, n_nodes, nw,
                           0.5 * dt, state.step);
  if (aw_axis_slave_active) {
    launch_apply_aw_axis_velocity_slave_2d(
        u_half_r.data(),
        u_half_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        false);
    if (predictor_pos_r_field->data() != u_half_r.data() ||
        predictor_pos_z_field->data() != u_half_z.data()) {
      launch_apply_aw_axis_velocity_slave_2d(
          predictor_pos_r,
          predictor_pos_z,
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_kappa_saved_.data(),
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active_,
          aw_axis_slave_theta_pi_active_,
          false);
    }
    copy_array_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(), u_half_r.data(), n_nodes);
    copy_array_kernel<<<blocks_nodes, 256>>>(
        state.v_z.data(), u_half_z.data(), n_nodes);
    sync_kernel("Hydro2D predictor AW axis velocity re-slave failed");
  }
  if (aw_axis_slave_multiblock_active) {
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        u_half_r.data(),
        u_half_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        false);
    if (predictor_pos_r_field->data() != u_half_r.data() ||
        predictor_pos_z_field->data() != u_half_z.data()) {
      launch_apply_aw_axis_velocity_slave_2d_multiblock(
          predictor_pos_r,
          predictor_pos_z,
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_multiblock_kappa_saved_.data(),
          aw_axis_slave_p_.data(),
          aw_axis_slave_q_.data(),
          static_cast<int>(aw_axis_slave_p_.size()),
          false);
    }
    copy_array_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(), u_half_r.data(), n_nodes);
    copy_array_kernel<<<blocks_nodes, 256>>>(
        state.v_z.data(), u_half_z.data(), n_nodes);
    sync_kernel(
        "Hydro2D predictor multiblock AW axis velocity re-slave failed");
  }
  if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
      state.carrier_node_class.empty()) { /* skip */
  } else {
    auto& carrier = ensure_carrier_hook_context();
    carrier.apply_reconstruct(nullptr, nullptr, u_half_r.data(), u_half_z.data());
    if (predictor_pos_r_field->data() != u_half_r.data() ||
        predictor_pos_z_field->data() != u_half_z.data()) {
      carrier.apply_reconstruct(nullptr, nullptr, predictor_pos_r, predictor_pos_z);
    }
    carrier.apply_reconstruct(nullptr, nullptr, state.v_r.data(), state.v_z.data());
  }
  emit_ring7_seam_scaffold_diagnostics(state,
                                        cfg,
                                        r_old.data(),
                                        z_old.data(),
                                        predictor_pos_r,
                                        predictor_pos_z,
                                        0.5 * dt,
                                        reduction,
                                        "predictor");
  if (cfg.numerics.hydro.mesh_quality_dt_cfl_enabled || path_guard_enabled) {
    const MeshQualityDtLimit pred_limit = compute_mesh_quality_dt_limit(
        state,
        cfg,
        r_old.data(),
        z_old.data(),
        predictor_pos_r,
        predictor_pos_z,
        0.5 * dt,
        reduction,
        path_guard_enabled);
    emit_ring7_pole_cap_recomputed_limiter_validation(state,
                                                      cfg,
                                                      pred_limit,
                                                      0.5 * dt,
                                                      "predictor");
    if (!pred_limit.admissible) {
      if (path_guard_enabled) {
        populate_path_guard_failure(
            result,
            state,
            cfg,
            pred_limit,
            tenryu::coupling::HydroFailureStage::PredictorCommit,
            dt);
      } else {
        populate_mesh_quality_dt_failure(
            result,
            state,
            cfg,
            pred_limit,
            tenryu::coupling::HydroFailureStage::PredictorCommit);
      }
      if (!path_guard_enabled &&
          pred_limit.metric == MeshQualityFailingMetric::RzVolume) {
        record_ring7_failed_mesh_velocity(state,
                                          cfg,
                                          pred_limit.first_cell,
                                          pred_limit.sigma_safe,
                                          0.5 * dt,
                                          predictor_pos_r,
                                          predictor_pos_z);
      }
      if (mesh_forensics_active) {
        core::NodeField1D candidate_r{"h2d:step:candidate_r_a"};
        core::NodeField1D candidate_z{"h2d:step:candidate_z_a"};
        candidate_r.reset(n_nodes);
        candidate_z.reset(n_nodes);
        commit_position_2d_kernel<<<nw.blocks(), 256>>>(
            candidate_r.data(), candidate_z.data(), r_old.data(), z_old.data(),
            predictor_pos_r, predictor_pos_z, d_node_active, d_node_flags,
            nw.begin, nw.end, n_nodes, 0.5 * dt);
        sync_kernel("Hydro2D mesh forensics predictor candidate failed");
        populate_mesh_forensics_sample(result,
                                       state,
                                       r_old,
                                       z_old,
                                       candidate_r,
                                       candidate_z,
                                       ur_old,
                                       uz_old,
                                       u_half_r,
                                       u_half_z,
                                       state.v_r,
                                       state.v_z,
                                       attr_pressure_n_r,
                                       attr_pressure_n_z,
                                       attr_q_n_r,
                                       attr_q_n_z,
                                       a_hg_n_r,
                                       a_hg_n_z,
                                       node_mass);
      }
      if (!path_guard_enabled) {
        tenryu::core::log_warning(
            "Hydro2D mesh-quality dt CFL rejected predictor trial at step=" +
            std::to_string(state.step) + ", reason=" +
            std::string(result.reason != nullptr ? result.reason : "") +
            ", first_cell=" + std::to_string(result.first_failing_cell) +
            ", first_corner_or_gauss=" +
            std::to_string(result.first_failing_corner) +
            ", sigma_safe=" + format_scientific(pred_limit.sigma_safe) +
            ", suggested_dt=" + format_scientific(result.suggested_dt));
      }
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  if (multiblock_path_admissibility_precommit_enabled(state, cfg)) {
    const auto pole_overlay = build_pole_path_overlay(state, cfg);
    const auto* pole_overlay_ptr = pole_overlay.active ? &pole_overlay : nullptr;
    const auto path_check = tenryu::mesh::evaluate_path_admissibility(
        state,
        cfg,
        r_old.data(),
        z_old.data(),
        predictor_pos_r,
        predictor_pos_z,
        0.5 * dt,
        cfg.numerics.ale.path_admissibility_floor,
        pole_overlay_ptr,
        state.x_r_reference.size() == state.x_r.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    observe_path_admissibility_margin(result, path_check);
    const bool forecast_due = mesh_forecast_or_shell_replay_due(state);
    if (forecast_due) {
      const auto forecast = tenryu::mesh::evaluate_mesh_forecast(
          state,
          cfg,
          r_old.data(),
          z_old.data(),
          predictor_pos_r,
          predictor_pos_z,
          0.5 * dt,
          cfg.numerics.ale.path_admissibility_floor,
          mesh_forecast_diag_config().q_warn,
          path_check,
          pole_overlay_ptr);
      populate_mesh_forecast_result(result, forecast);
      if (mesh_forecast_diag_due(state)) {
        emit_mesh_forecast_diag(state, "predictor", 0.5 * dt, path_check,
                                forecast);
      }
      tenryu::hydro::ale::maybe_run_shell_rezone_replay_probe(
          state,
          cfg,
          forecast,
          r_old.data(),
          z_old.data(),
          predictor_pos_r,
          predictor_pos_z,
          0.5 * dt,
          "predictor",
          eos_ctx,
          reduction);
    }
    if (multiblock_path_admissibility_failed(path_check)) {
      populate_multiblock_path_admissibility_failure(
          result,
          state,
          cfg,
          path_check,
          dt,
          tenryu::coupling::HydroFailureStage::PredictorCommit);
      ++state.multiblock_path_dt_rejection_count;
      if (mesh_forensics_active) {
        core::NodeField1D candidate_r{"h2d:step:candidate_r_b"};
        core::NodeField1D candidate_z{"h2d:step:candidate_z_b"};
        candidate_r.reset(n_nodes);
        candidate_z.reset(n_nodes);
        commit_position_2d_kernel<<<nw.blocks(), 256>>>(
            candidate_r.data(), candidate_z.data(), r_old.data(), z_old.data(),
            predictor_pos_r, predictor_pos_z, d_node_active, d_node_flags,
            nw.begin, nw.end, n_nodes, 0.5 * dt);
        sync_kernel("Hydro2D mesh forensics predictor path candidate failed");
        populate_mesh_forensics_sample(result,
                                       state,
                                       r_old,
                                       z_old,
                                       candidate_r,
                                       candidate_z,
                                       ur_old,
                                       uz_old,
                                       u_half_r,
                                       u_half_z,
                                       state.v_r,
                                       state.v_z,
                                       attr_pressure_n_r,
                                       attr_pressure_n_z,
                                       attr_q_n_r,
                                       attr_q_n_z,
                                       a_hg_n_r,
                                       a_hg_n_z,
                                       node_mass);
      }
      tenryu::core::log_warning(
          "Hydro2D multiblock path admissibility rejected predictor trial at step=" +
          std::to_string(state.step) + ", first_cell=" +
          std::to_string(path_check.first_failing_cell) + ", lambda=" +
          format_scientific(path_check.first_failing_lambda) +
          format_multiblock_path_admissibility_location(state, cfg, path_check) +
          format_multiblock_path_admissibility_anatomy(path_check) +
          ", min_margin=" + format_scientific(path_check.min_margin) +
          ", suggested_dt=" + format_scientific(result.suggested_dt));
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  commit_position_2d_kernel<<<nw.blocks(), 256>>>(
      state.x_r.data(), state.x_z.data(), r_old.data(), z_old.data(),
      predictor_pos_r, predictor_pos_z, d_node_active, d_node_flags, nw.begin,
      nw.end, n_nodes, 0.5 * dt);
  sync_kernel("Hydro2D predictor_update kernel failed");

  apply_boundary_2d(state, cfg, t_op + 0.5 * dt);
  copy_array_kernel<<<nw.blocks(), 256>>>(u_half_r.data() + nw.begin,
                                          state.v_r.data() + nw.begin,
                                          nw.count());
  copy_array_kernel<<<nw.blocks(), 256>>>(u_half_z.data() + nw.begin,
                                          state.v_z.data() + nw.begin,
                                          nw.count());
  sync_kernel("Hydro2D copy u_half kernels failed");
  if (state_supply_active) {
    restore_state_supply_material_velocity(state, cfg);
  }
  enforce_node_axis_state(state, d_node_flags);
  if (aw_axis_slave_active) {
    launch_apply_aw_axis_velocity_slave_2d(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        false);
    if (aw_axis_snap_env_enabled) {
      launch_snap_aw_axis_coordinates_2d(
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_kappa_saved_.data(),
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active_,
          aw_axis_slave_theta_pi_active_);
    }
    copy_array_kernel<<<blocks_nodes, 256>>>(
        u_half_r.data(), state.v_r.data(), n_nodes);
    copy_array_kernel<<<blocks_nodes, 256>>>(
        u_half_z.data(), state.v_z.data(), n_nodes);
    sync_kernel("Hydro2D post-boundary predictor AW axis closure failed");
  }
  if (aw_axis_slave_multiblock_active) {
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        false);
    if (aw_axis_snap_env_enabled) {
      launch_snap_aw_axis_coordinates_2d_multiblock(
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_multiblock_kappa_saved_.data(),
          aw_axis_slave_p_.data(),
          aw_axis_slave_q_.data(),
          static_cast<int>(aw_axis_slave_p_.size()));
    }
    copy_array_kernel<<<blocks_nodes, 256>>>(
        u_half_r.data(), state.v_r.data(), n_nodes);
    copy_array_kernel<<<blocks_nodes, 256>>>(
        u_half_z.data(), state.v_z.data(), n_nodes);
    sync_kernel(
        "Hydro2D post-boundary predictor multiblock AW axis closure failed");
  }
  if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
      state.carrier_node_class.empty()) { /* skip */
  } else {
    auto& carrier = ensure_carrier_hook_context();
    carrier.apply_reconstruct(state.x_r.data(), state.x_z.data(),
                              state.v_r.data(), state.v_z.data());
    carrier.apply_reconstruct(nullptr, nullptr, u_half_r.data(), u_half_z.data());
  }
  // Consultation #23 §11.4: finalize contact after the last stage-velocity writer.
  launch_apply_evac_contact_velocity_constraints(
      state, state.v_r.data(), state.v_z.data(),
      cfg.numerics.hydro.rz_momentum_scheme_id == 0
          ? node_mass.data()
          : state.node_planar_mass.data(),
      state.x_r.data(), state.x_z.data(),
      cs_n.data(),
      d_hydro_force_active,
      dt,
      state.evacuated_cells.d_contact_pair_dk.data(),
      cfg.numerics.ale.evacuated_cell.closure_contact
          .mortar_position_drift_beta);
  if (!midpoint_v1) {
    evacuated_cell_contact_probe(state, cfg, "post_vel_predictor");
  }
  central_pseudo_core::rebuild_virtual_member_geometry(
      state, cfg, "post_predictor");
  // OPEN-HYDRO-CORNER: ghost node coords must be refreshed BEFORE the
  // half-position geometry/density refresh and mesh-cache upload below;
  // the predictor commit moves owned nodes only, so until this exchange
  // the neighbor's boundary columns are one motion stale and every
  // derived quantity (Svec/vol/rho) forks per rank.
  exchange_mid_nodes_2d();
  if (cap_energy_audit && !midpoint_v1) {
    result.cap_energy_audit_W_ext_step =
        compute_pressure_boundary_work_step_2d(
            state, cfg, dt, t_op + 0.5 * dt, u_half_r, u_half_z, node_mass);
  }

  if (cfg.numerics.hydro.in_hydro_corner_j_guard_enabled ||
      cfg.numerics.hydro.in_hydro_gauss_j_guard_enabled ||
      cfg.numerics.hydro.in_hydro_rz_volume_guard_enabled) {
    result = compute_in_hydro_candidate_mesh_guard(
        state,
        dt,
        r_old.data(),
        z_old.data(),
        state.x_r.data(),
        state.x_z.data(),
        cfg,
        tenryu::coupling::HydroFailureStage::PostPredictor,
        reduction,
        d_hydro_force_active,
        cfg.numerics.hydro.regime_aware_corner_j_guard_enabled
            ? active_mesh_regime_cache->current()
            : nullptr);
    if (!result.admissible) {
      if (mesh_forensics_active) {
        populate_mesh_forensics_sample(result,
                                       state,
                                       r_old,
                                       z_old,
                                       state.x_r,
                                       state.x_z,
                                       ur_old,
                                       uz_old,
                                       u_half_r,
                                       u_half_z,
                                       state.v_r,
                                       state.v_z,
                                       attr_pressure_n_r,
                                       attr_pressure_n_z,
                                       attr_q_n_r,
                                       attr_q_n_z,
                                       a_hg_n_r,
                                       a_hg_n_z,
                                       node_mass);
      }
      emit_mesh_attr_failure(result);
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  result = refresh_geometry_and_density(
      state,
      cfg,
      tenryu::coupling::HydroFailureStage::PostPredictor,
      midpoint_v1 ? nullptr : d_rho_clamp_count,
      observability);
  if (!result.admissible) {
    emit_mesh_attr_failure(result);
    cleanup_precommit_retry_buffers();
    return result;
  }
  upload_mesh_cache(state, mesh_cache);
  central_pseudo_core::aggregate_state(state, cfg, "post_predictor", false);
  pole_angular_derefine::aggregate_state(state, cfg, "post_predictor", false);
  conservation_audit::emit_stage(state, "post_predictor_after_aggregate");
  if (midpoint_v1) {
    refresh_midpoint_stage_closure();
  } else {
    refresh_or_enforce_closure(false);
  }
  if (mesh_attr_active) {
    auto& attr = diagnostics::mesh_attribution::global_workspace();
    attr.clear_lagrangian_sources();
    attr.record_kinematic_carry(ur_old.data(), uz_old.data(), d_node_active, n_nodes,
                                dt);
    attr.record_acceleration_displacement(
        diagnostics::mesh_attribution::MeshDeformSource::HydroPressure,
        attr_pressure_n_r.data(),
        attr_pressure_n_z.data(),
        d_node_active,
        n_nodes,
        0.5 * dt * dt);
    attr.record_acceleration_displacement(
        diagnostics::mesh_attribution::MeshDeformSource::ArtificialViscosity,
        attr_q_n_r.data(),
        attr_q_n_z.data(),
        d_node_active,
        n_nodes,
        0.5 * dt * dt);
  }

  core::CellField1D cs_half{"h2d:step:cs_half"};
  if (midpoint_v1) {
    with_midpoint_stage_energy(
        [&]() { compute_sound_speed_for_step(cs_half); });
  } else {
    compute_sound_speed_for_step(cs_half);
  }
  exchange_cell_ghosts_2d({state.rho.data(), state.vol.data(), state.Te.data(),
                           state.Ti.data(), state.Pe.data(), state.Pi.data(),
                           cs_half.data()});

  core::CellField1D dVdt_half{"h2d:step:dVdt_half"};
  core::CellField1D div_u_half{"h2d:step:div_u_half"};
  core::CellField1D dl_half{"h2d:step:dl_half"};
  dVdt_half.reset(n_cells);
  div_u_half.reset(n_cells);
  dl_half.reset(n_cells);
  // OPEN-HYDRO-CORNER fix: av_fw (declared at the first AV cluster in
  // this scope) covers the ghost ring here too.
  detail::launch_compute_cell_dVdt_2d(
      dVdt_half.data(), state.v_r.data(), state.v_z.data(), mesh_cache.Svec_r.data(),
      mesh_cache.Svec_z.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  compute_cell_div_u_kernel<<<av_fw.blocks(), 256>>>(
      div_u_half.data(), dVdt_half.data(), state.vol.data(), av_fw.begin, av_fw.end,
      n_cells);
  compute_cell_length_scale_kernel<<<av_fw.blocks(), 256>>>(
      dl_half.data(), mesh_cache.area.data(), av_fw.begin, av_fw.end, n_cells);
  sync_kernel("Hydro2D compute half dVdt/div_u/dl kernels failed");

  compute_artificial_viscosity_2d(dl_half, div_u_half, cs_half);
  qcap_apply_in_place(state.Qvisc,
                      cfg,
                      state,
                      d_hydro_force_active,
                      per_material_hydro_enabled ? &Qvisc_per_material : nullptr);
  exchange_cell_ghosts_2d({state.Qvisc.data()});

  core::CellField1D P_half{"h2d:step:P_half"};
  core::CellField1D Pi_half{"h2d:step:Pi_half"};
  core::CellField1D Q_half{"h2d:step:Q_half"};
  core::CellField1D V_half{"h2d:step:V_half"};
  core::CellField1D rho_half{"h2d:step:rho_half"};
  core::CellField1D Te_half{"h2d:step:Te_half"};
  core::CellField1D Ti_half{"h2d:step:Ti_half"};
  P_half.reset(n_cells);
  Pi_half.reset(n_cells);
  Q_half.reset(n_cells);
  V_half.reset(n_cells);
  if (use_two_temp) {
    rho_half.reset(n_cells);
    Te_half.reset(n_cells);
    Ti_half.reset(n_cells);
  }
  if (midpoint_v1) {
    cuda_check(cudaMemcpy(P_half.data(), state.Pe.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint copy P_half failed");
    cuda_check(cudaMemcpy(Pi_half.data(), state.Pi.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint copy Pi_half failed");
  } else {
    // P_half/Pi_half feed the corrector pq build on the GHOST-extended window
    // (unlike 1D, whose corrector pq reads state.Pe/Pi directly): average over
    // pw so interface ghost cells carry real half-pressures. Inputs are
    // ghost-valid (Pe_old from the previous tail exchange, Pe from ex1.5b).
    // Root cause of the 2D Noh P=2 growth (2e-7@10 steps -> 1e-4@347): zeroed
    // ghost P_half under the owned-window average.
    average_arrays_kernel<<<pw.blocks(), 256>>>(
        P_half.data() + pw.begin, Pe_old.data() + pw.begin,
        state.Pe.data() + pw.begin, pw.count());
    average_arrays_kernel<<<pw.blocks(), 256>>>(
        Pi_half.data() + pw.begin, Pi_old.data() + pw.begin,
        state.Pi.data() + pw.begin, pw.count());
  }
  cuda_check(cudaMemcpy(Q_half.data(), state.Qvisc.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy Q_half failed");
  cuda_check(cudaMemcpy(V_half.data(), state.vol.data(), n_cells * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy V_half failed");
  sync_kernel("Hydro2D: build P_half/Pi_half kernels failed");
  if (use_two_temp) {
    cuda_check(cudaMemcpy(rho_half.data(), state.rho.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy rho_half failed");
    cuda_check(cudaMemcpy(Te_half.data(), state.Te.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy Te_half failed");
    cuda_check(cudaMemcpy(Ti_half.data(), state.Ti.data(), n_cells * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy Ti_half failed");
  }

  core::CellField1D pq_half{"h2d:step:pq_half"};
  pq_half.reset(n_cells);
  if (compatible_energy_mode) {
    build_cell_pressure_kernel<<<pw.blocks(), 256>>>(
        pq_half.data(), P_half.data(), Pi_half.data(), d_hydro_force_active,
        pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D build pressure force-source half kernel failed");
  } else {
    build_cell_pq_kernel<<<pw.blocks(), 256>>>(
        pq_half.data(), P_half.data(), Pi_half.data(), state.Qvisc.data(),
        d_hydro_force_active, pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D build_cell_pq half kernel failed");
  }

  core::NodeField1D attr_pressure_half_r{"h2d:step:attr_pressure_half_r"};
  core::NodeField1D attr_pressure_half_z{"h2d:step:attr_pressure_half_z"};
  core::NodeField1D attr_q_half_r{"h2d:step:attr_q_half_r"};
  core::NodeField1D attr_q_half_z{"h2d:step:attr_q_half_z"};
  if (mesh_forensics_active) {
    core::CellField1D attr_pressure_half{"h2d:step:attr_pressure_half"};
    core::CellField1D attr_q_half{"h2d:step:attr_q_half"};
    attr_pressure_half.reset(n_cells);
    attr_q_half.reset(n_cells);
    build_cell_pressure_kernel<<<pw.blocks(), 256>>>(
        attr_pressure_half.data(), P_half.data(), Pi_half.data(),
        d_hydro_force_active, pw.begin, pw.end, n_cells);
    build_cell_q_kernel<<<cw.blocks(), 256>>>(
        attr_q_half.data(), Q_half.data(), d_hydro_force_active, cw.begin,
        cw.end, n_cells);
    sync_kernel("Hydro2D mesh forensics build half source pressure/Q failed");
    attr_pressure_half_r.reset(n_nodes);
    attr_pressure_half_z.reset(n_nodes);
    attr_q_half_r.reset(n_nodes);
    attr_q_half_z.reset(n_nodes);
    compute_acceleration_2d(attr_pressure_half_r, attr_pressure_half_z, state,
                            cfg, t_op + 0.5 * dt, mesh_cache,
                            attr_pressure_half, d_node_active, d_node_flags, node_mass,
                            d_hydro_force_active, eos_ctx, true, false, true, &cs_half,
                            false, nullptr, nullptr, 0.0, false, nullptr, false,
                            aw_axis_slave_theta0_active_,
                            aw_axis_slave_theta_pi_active_, nullptr, nullptr,
                            false, nullptr, "attr_p_half");
    compute_acceleration_2d(attr_q_half_r, attr_q_half_z, state, cfg,
                            t_op + 0.5 * dt, mesh_cache, attr_q_half,
                            d_node_active, d_node_flags, node_mass, d_hydro_force_active,
                            eos_ctx, false, false, false, nullptr, true, nullptr,
                            nullptr, 0.0, false, nullptr, false,
                            aw_axis_slave_theta0_active_,
                            aw_axis_slave_theta_pi_active_, nullptr, nullptr,
                            false, nullptr, "attr_q_half");
  }

  core::NodeField1D a_half_r{"h2d:step:a_half_r"};
  core::NodeField1D a_half_z{"h2d:step:a_half_z"};
  core::NodeField1D cap_energy_audit_force_half_r{"h2d:step:cap_energy_audit_force_half_r"};
  core::NodeField1D cap_energy_audit_force_half_z{"h2d:step:cap_energy_audit_force_half_z"};
  core::NodeField1D pole_axis_bbsw_true_force_half_z{"h2d:step:pole_axis_bbsw_true_force_half_z"};
  double r_momentum_source_force_sum = 0.0;
  a_half_r.reset(n_nodes);
  a_half_z.reset(n_nodes);
  const bool shadow_replay_due =
      shadow_replay_step >= 0 && !shadow_replay_printed &&
      state.step == shadow_replay_step && compatible_energy_mode &&
      cfg.numerics.hydro.aw_compatible_force_work &&
      cfg.numerics.hydro.subzonal_pressure_enabled &&
      !state.mesh.topo.multiblock.has_value();
  StructuredAwShadowProjectionStorage shadow_projection_storage;
  SubzonalPressureProjectionDebugBuffers shadow_projection_debug{};
  const SubzonalPressureProjectionDebugBuffers* shadow_projection_debug_ptr =
      nullptr;
  if (shadow_replay_due) {
    shadow_projection_storage.reset(static_cast<std::size_t>(n_cells));
    shadow_projection_debug = shadow_projection_storage.device_buffers();
    shadow_projection_debug_ptr = &shadow_projection_debug;
  }
  compute_acceleration_2d(a_half_r, a_half_z, state, cfg, t_op + 0.5 * dt,
                          mesh_cache, pq_half, d_node_active, d_node_flags, node_mass,
                          d_hydro_force_active, eos_ctx, true, false, true, &cs_half,
                          true,
                          (cap_energy_audit && compatible_energy_mode)
                              ? &cap_energy_audit_force_half_r
                              : nullptr,
                          (cap_energy_audit && compatible_energy_mode)
                              ? &cap_energy_audit_force_half_z
                              : nullptr,
                          midpoint_v1 ? 0.0 : dt,
                          true,
                          &pole_axis_bbsw_true_force_half_z,
                          true,
                          aw_axis_slave_theta0_active_,
                          aw_axis_slave_theta_pi_active_,
                          shadow_projection_debug_ptr,
                          (!midpoint_v1 && r_momentum_source_impulse != nullptr)
                              ? &r_momentum_source_force_sum
                              : nullptr,
                          midpoint_v1,
                          midpoint_v1 ? &state.Qvisc : nullptr, "dyn_half");
  if (!midpoint_v1 && r_momentum_source_impulse != nullptr) {
    *r_momentum_source_impulse = dt * r_momentum_source_force_sum;
  }
  if (shadow_replay_due) {
    shadow_replay_printed = emit_structured_aw_shadow_replay(
        state,
        cfg,
        eos_ctx,
        d_hydro_force_active,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        shadow_projection_storage);
  }
  ShellDriveSymDiagSnapshot shell_drive_sym_snapshot{};
  if (shell_drive_sym_diag_active) {
    shell_drive_sym_snapshot =
        capture_shell_drive_sym_diag_snapshot(
            state, cfg, t_op + 0.5 * dt, node_mass);
  }
  core::NodeField1D a_hg_half_r{"h2d:step:a_hg_half_r"};
  core::NodeField1D a_hg_half_z{"h2d:step:a_hg_half_z"};
  core::NodeField1D a_nonhg_half_r{"h2d:step:a_nonhg_half_r"};
  core::NodeField1D a_nonhg_half_z{"h2d:step:a_nonhg_half_z"};
  core::CellField1D hourglass_work_erg{"h2d:step:hourglass_work_erg"};
  if (subzonal_hourglass_enabled(cfg)) {
    if (midpoint_v1) {
      a_nonhg_half_r.reset(n_nodes);
      a_nonhg_half_z.reset(n_nodes);
      cuda_check(cudaMemcpy(a_nonhg_half_r.data(), a_half_r.data(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy non-hourglass a_half_r failed");
      cuda_check(cudaMemcpy(a_nonhg_half_z.data(), a_half_z.data(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy non-hourglass a_half_z failed");
    }
    core::CellField1D pressure_half_for_hg{"h2d:step:pressure_half_for_hg"};
    pressure_half_for_hg.reset(n_cells);
    build_cell_pressure_kernel<<<pw.blocks(), 256>>>(
        pressure_half_for_hg.data(), P_half.data(), Pi_half.data(),
        d_hydro_force_active, pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D build_cell_pressure half hourglass kernel failed");
    compute_anti_hourglass_acceleration_2d(
        a_hg_half_r,
        a_hg_half_z,
        (!midpoint_v1 &&
         cfg.numerics.hydro.hourglass.compatible_work_enabled)
            ? &hourglass_work_erg
            : nullptr,
        state,
        cfg,
        pressure_half_for_hg,
        node_mass,
        a_half_r,
        a_half_z,
        d_hydro_force_active,
        dt,
        cfg.numerics.hydro.hourglass.compatible_work_enabled,
        d_node_flags);
    add_hourglass_acceleration_2d(a_half_r, a_half_z, a_hg_half_r, a_hg_half_z);
    // §22 contact must be last; hourglass leaked ~0.04 nm/step into the held gap.
    launch_apply_evac_contact_accel_constraints(
        state,
        a_half_r.data(),
        a_half_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data());
  }

  if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
      state.carrier_node_class.empty()) { /* skip */
  } else {
    ensure_carrier_hook_context().apply_condensation(
        a_half_r.data(), a_half_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 1
            ? state.node_planar_mass.data()
            : node_mass.data());
  }

  if (midpoint_v1) {
    core::NodeField1D u_provisional_r{"h2d:step:u_provisional_r"};
    core::NodeField1D u_provisional_z{"h2d:step:u_provisional_z"};
    core::NodeField1D ubar_provisional_r{"h2d:step:ubar_provisional_r"};
    core::NodeField1D ubar_provisional_z{"h2d:step:ubar_provisional_z"};
    core::NodeField1D x_provisional_r{"h2d:step:x_provisional_r"};
    core::NodeField1D x_provisional_z{"h2d:step:x_provisional_z"};
    u_provisional_r.reset(n_nodes);
    u_provisional_z.reset(n_nodes);
    ubar_provisional_r.reset(n_nodes);
    ubar_provisional_z.reset(n_nodes);
    x_provisional_r.reset(n_nodes);
    x_provisional_z.reset(n_nodes);
    launch_update_node_velocity_2d(
        u_provisional_r.data(), u_provisional_z.data(), ur_old.data(),
        uz_old.data(), a_half_r.data(), a_half_z.data(), d_node_active,
        d_node_flags, n_nodes, dt, state, cfg);
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        ubar_provisional_r.data() + nw.begin, ur_old.data() + nw.begin,
        u_provisional_r.data() + nw.begin, nw.count());
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        ubar_provisional_z.data() + nw.begin, uz_old.data() + nw.begin,
        u_provisional_z.data() + nw.begin, nw.count());
    core::NodeField1D provisional_position_v_z{
        "h2d:step:provisional_position_v_z"};
    const double* provisional_position_v_z_ptr =
        ubar_provisional_z.data();
    if (state_supply_active || axis_lagrangian_tangential) {
      provisional_position_v_z.reset(n_nodes);
      cuda_check(cudaMemcpy(provisional_position_v_z.data(),
                            ubar_provisional_z.data(),
                            n_nodes * sizeof(double),
                            cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy provisional position v_z failed");
      if (state_supply_active) {
        zero_state_supply_z_position_velocity(
            provisional_position_v_z, cfg, state.mesh.topo.nr,
            state.mesh.topo.nz);
      }
      if (axis_lagrangian_tangential) {
        constrain_axis_z_position_velocity(
            state, cfg, provisional_position_v_z, r_old, z_old, dt, false);
      }
      provisional_position_v_z_ptr = provisional_position_v_z.data();
    }
    if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
        state.carrier_node_class.empty()) { /* skip */
    } else {
      auto& carrier = ensure_carrier_hook_context();
      carrier.apply_reconstruct(nullptr, nullptr, u_provisional_r.data(),
                                u_provisional_z.data());
      carrier.apply_reconstruct(nullptr, nullptr, ubar_provisional_r.data(),
                                ubar_provisional_z.data());
      if (!provisional_position_v_z.empty()) {
        carrier.apply_reconstruct(nullptr, nullptr, nullptr,
                                  provisional_position_v_z.data());
      }
    }
    commit_position_2d_kernel<<<blocks_nodes, 256>>>(
        x_provisional_r.data(), x_provisional_z.data(), r_old.data(),
        z_old.data(), ubar_provisional_r.data(), provisional_position_v_z_ptr,
        d_node_active, d_node_flags, nw.begin, nw.end, n_nodes, dt);
    sync_kernel("Hydro2D midpoint provisional full update failed");

    recompute_midpoint_corner_work_2d(
        state, cfg, ubar_provisional_r.data(), ubar_provisional_z.data(),
        d_hydro_force_active);
    cuda_check(cudaMemcpy(midpoint_stage_ee.data(), e_old.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint restore provisional ee failed");
    cuda_check(cudaMemcpy(midpoint_stage_ei.data(), ei_old.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint restore provisional ei failed");
    with_midpoint_stage_energy([&]() {
      apply_compatible_energy_work_2d(
          state, cfg, true, P_half, Pi_half, use_two_temp, dt,
          d_hydro_force_active, nullptr, nullptr);
      if (subzonal_hourglass_enabled(cfg) &&
          cfg.numerics.hydro.hourglass.compatible_work_enabled) {
        core::CellField1D pressure_half_for_hg{
            "h2d:step:midpoint_pressure_half_provisional"};
        pressure_half_for_hg.reset(n_cells);
        build_cell_pressure_kernel<<<blocks_cells, 256>>>(
            pressure_half_for_hg.data(), P_half.data(), Pi_half.data(),
            d_hydro_force_active, pw.begin, pw.end, n_cells);
        compute_anti_hourglass_acceleration_2d(
            a_hg_half_r, a_hg_half_z, &hourglass_work_erg, state, cfg,
            pressure_half_for_hg, node_mass, a_nonhg_half_r, a_nonhg_half_z,
            d_hydro_force_active, dt, true, d_node_flags,
            ubar_provisional_r.data(), ubar_provisional_z.data(), dt);
        apply_hourglass_work_2d(state, cfg, hourglass_work_erg, use_two_temp,
                                d_hydro_force_active, nullptr, nullptr);
      }
    });

    launch_update_node_velocity_2d(
        u_half_r.data(), u_half_z.data(), ur_old.data(), uz_old.data(),
        a_half_r.data(), a_half_z.data(), d_node_active, d_node_flags, n_nodes,
        0.5 * dt, state, cfg);
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        predictor_work_v_r.data() + nw.begin, ur_old.data() + nw.begin,
        u_half_r.data() + nw.begin, nw.count());
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        predictor_work_v_z.data() + nw.begin, uz_old.data() + nw.begin,
        u_half_z.data() + nw.begin, nw.count());
    // Trial-stage projection; heat is owned by the post-boundary finalizer (consultation #23 §7.4).
    const double* const corrected_half_commit_node_mass =
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data();
    launch_apply_evac_contact_velocity_constraints(
        state,
        u_half_r.data(),
        u_half_z.data(),
        corrected_half_commit_node_mass,
        state.x_r.data(),
        state.x_z.data(),
        cs_half.data(),
        d_hydro_force_active,
        dt,
        nullptr,
        cfg.numerics.ale.evacuated_cell.closure_contact
            .mortar_position_drift_beta);
    copy_array_kernel<<<blocks_nodes, 256>>>(
        state.v_r.data(), u_half_r.data(), n_nodes);
    copy_array_kernel<<<blocks_nodes, 256>>>(
        state.v_z.data(), u_half_z.data(), n_nodes);
    evacuated_cell_contact_probe(state, cfg, "post_vcopy");
    sync_kernel("Hydro2D midpoint corrected-half kinematics failed");

    recompute_midpoint_corner_work_2d(
        state, cfg, predictor_work_v_r.data(), predictor_work_v_z.data(),
        d_hydro_force_active);
    cuda_check(cudaMemcpy(midpoint_stage_ee.data(), e_old.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint restore corrected-half ee failed");
    cuda_check(cudaMemcpy(midpoint_stage_ei.data(), ei_old.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint restore corrected-half ei failed");
    with_midpoint_stage_energy([&]() {
      apply_compatible_energy_work_2d(
          state, cfg, true, P_half, Pi_half, use_two_temp, 0.5 * dt,
          d_hydro_force_active, nullptr, nullptr);
      if (subzonal_hourglass_enabled(cfg) &&
          cfg.numerics.hydro.hourglass.compatible_work_enabled) {
        core::CellField1D pressure_half_for_hg{
            "h2d:step:midpoint_pressure_half_corrected"};
        pressure_half_for_hg.reset(n_cells);
        build_cell_pressure_kernel<<<blocks_cells, 256>>>(
            pressure_half_for_hg.data(), P_half.data(), Pi_half.data(),
            d_hydro_force_active, pw.begin, pw.end, n_cells);
        compute_anti_hourglass_acceleration_2d(
            a_hg_half_r, a_hg_half_z, &hourglass_work_erg, state, cfg,
            pressure_half_for_hg, node_mass, a_nonhg_half_r, a_nonhg_half_z,
            d_hydro_force_active, dt, true, d_node_flags,
            predictor_work_v_r.data(), predictor_work_v_z.data(), 0.5 * dt);
        apply_hourglass_work_2d(state, cfg, hourglass_work_erg, use_two_temp,
                                d_hydro_force_active, nullptr, nullptr);
      }
    });

    average_arrays_kernel<<<blocks_nodes, 256>>>(
        state.x_r.data() + nw.begin, r_old.data() + nw.begin,
        x_provisional_r.data() + nw.begin, nw.count());
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        state.x_z.data() + nw.begin, z_old.data() + nw.begin,
        x_provisional_z.data() + nw.begin, nw.count());
    sync_kernel("Hydro2D midpoint corrected-half geometry failed");
    evacuated_cell_contact_probe(state, cfg, "pre_apply_boundary");
    apply_boundary_2d(state, cfg, t_op + 0.5 * dt);
    evacuated_cell_contact_probe(state, cfg, "post_apply_boundary");
    if (state_supply_active) {
      restore_state_supply_material_velocity(state, cfg);
    }
    enforce_node_axis_state(state, d_node_flags);
    evacuated_cell_contact_probe(state, cfg, "post_axis_enforce");
    if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
        state.carrier_node_class.empty()) { /* skip */
    } else {
      auto& carrier = ensure_carrier_hook_context();
      carrier.apply_reconstruct(state.x_r.data(), state.x_z.data(),
                                state.v_r.data(), state.v_z.data());
      carrier.apply_reconstruct(nullptr, nullptr, u_half_r.data(), u_half_z.data());
    }
    // Consultation #23 §11.4: finalize contact after the last stage-velocity writer.
    launch_apply_evac_contact_velocity_constraints(
        state, state.v_r.data(), state.v_z.data(),
        cfg.numerics.hydro.rz_momentum_scheme_id == 0
            ? node_mass.data()
            : state.node_planar_mass.data(),
        state.x_r.data(), state.x_z.data(),
        cs_half.data(),
        d_hydro_force_active,
        dt,
        state.evacuated_cells.d_contact_pair_dk.data(),
        cfg.numerics.ale.evacuated_cell.closure_contact
            .mortar_position_drift_beta);
    evacuated_cell_contact_probe(state, cfg, "post_carrier");
    evacuated_cell_contact_probe(state, cfg, "post_vel_predictor");
    result = refresh_geometry_and_density(
        state, cfg, tenryu::coupling::HydroFailureStage::PostPredictor,
        nullptr, observability);
    if (!result.admissible) {
      emit_mesh_attr_failure(result);
      cleanup_precommit_retry_buffers();
      return result;
    }
    upload_mesh_cache(state, mesh_cache);
    refresh_midpoint_stage_closure();

    with_midpoint_stage_energy(
        [&]() { compute_sound_speed_for_step(cs_half); });
    detail::launch_compute_cell_dVdt_2d(
        dVdt_half.data(), state.v_r.data(), state.v_z.data(),
        mesh_cache.Svec_r.data(), mesh_cache.Svec_z.data(), state, cfg,
        d_cell_nverts, d_hydro_force_active);
    compute_cell_div_u_kernel<<<blocks_cells, 256>>>(
        div_u_half.data(), dVdt_half.data(), state.vol.data(), av_fw.begin,
        av_fw.end, n_cells);
    compute_cell_length_scale_kernel<<<blocks_cells, 256>>>(
        dl_half.data(), mesh_cache.area.data(), av_fw.begin, av_fw.end,
        n_cells);
    compute_artificial_viscosity_2d(dl_half, div_u_half, cs_half);
    qcap_apply_in_place(state.Qvisc, cfg, state, d_hydro_force_active,
                        per_material_hydro_enabled ? &Qvisc_per_material
                                                   : nullptr);

    cuda_check(cudaMemcpy(P_half.data(), state.Pe.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint copy corrected P_half failed");
    cuda_check(cudaMemcpy(Pi_half.data(), state.Pi.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint copy corrected Pi_half failed");
    cuda_check(cudaMemcpy(Q_half.data(), state.Qvisc.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint copy corrected Q_half failed");
    cuda_check(cudaMemcpy(V_half.data(), state.vol.data(),
                          n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
               "Hydro2D midpoint copy corrected V_half failed");
    if (use_two_temp) {
      cuda_check(cudaMemcpy(rho_half.data(), state.rho.data(),
                            n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy corrected rho_half failed");
      cuda_check(cudaMemcpy(Te_half.data(), state.Te.data(),
                            n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy corrected Te_half failed");
      cuda_check(cudaMemcpy(Ti_half.data(), state.Ti.data(),
                            n_cells * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy corrected Ti_half failed");
    }
    build_cell_pressure_kernel<<<blocks_cells, 256>>>(
        pq_half.data(), P_half.data(), Pi_half.data(), d_hydro_force_active,
        pw.begin, pw.end, n_cells);
    sync_kernel("Hydro2D midpoint corrected-half closure failed");

    compute_acceleration_2d(
        a_half_r, a_half_z, state, cfg, t_op + 0.5 * dt, mesh_cache, pq_half,
        d_node_active, d_node_flags, node_mass, d_hydro_force_active, eos_ctx,
        true, false, true, &cs_half, true,
        (cap_energy_audit && compatible_energy_mode)
            ? &cap_energy_audit_force_half_r
            : nullptr,
        (cap_energy_audit && compatible_energy_mode)
            ? &cap_energy_audit_force_half_z
            : nullptr,
        dt, true, &pole_axis_bbsw_true_force_half_z, true,
        aw_axis_slave_theta0_active_, aw_axis_slave_theta_pi_active_, nullptr,
        r_momentum_source_impulse != nullptr ? &r_momentum_source_force_sum
                                             : nullptr,
        true, &state.Qvisc, "dyn_half_corrected");
    if (r_momentum_source_impulse != nullptr) {
      *r_momentum_source_impulse = dt * r_momentum_source_force_sum;
    }
    if (shell_drive_sym_diag_active) {
      shell_drive_sym_snapshot =
          capture_shell_drive_sym_diag_snapshot(
              state, cfg, t_op + 0.5 * dt, node_mass);
    }
    if (subzonal_hourglass_enabled(cfg)) {
      a_nonhg_half_r.reset(n_nodes);
      a_nonhg_half_z.reset(n_nodes);
      cuda_check(cudaMemcpy(a_nonhg_half_r.data(), a_half_r.data(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy corrected non-hourglass a_r failed");
      cuda_check(cudaMemcpy(a_nonhg_half_z.data(), a_half_z.data(),
                            n_nodes * sizeof(double), cudaMemcpyDeviceToDevice),
                 "Hydro2D midpoint copy corrected non-hourglass a_z failed");
      core::CellField1D pressure_half_for_hg{
          "h2d:step:midpoint_pressure_half_final"};
      pressure_half_for_hg.reset(n_cells);
      build_cell_pressure_kernel<<<blocks_cells, 256>>>(
          pressure_half_for_hg.data(), P_half.data(), Pi_half.data(),
          d_hydro_force_active, pw.begin, pw.end, n_cells);
      compute_anti_hourglass_acceleration_2d(
          a_hg_half_r, a_hg_half_z, nullptr, state, cfg,
          pressure_half_for_hg, node_mass, a_nonhg_half_r, a_nonhg_half_z,
          d_hydro_force_active, dt, false, d_node_flags);
      add_hourglass_acceleration_2d(
          a_half_r, a_half_z, a_hg_half_r, a_hg_half_z);
      // §22 contact must be last; hourglass leaked ~0.04 nm/step into the held gap.
      launch_apply_evac_contact_accel_constraints(
          state,
          a_half_r.data(),
          a_half_z.data(),
          cfg.numerics.hydro.rz_momentum_scheme_id == 0
              ? node_mass.data()
              : state.node_planar_mass.data());
    }
    if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
        state.carrier_node_class.empty()) { /* skip */
    } else {
      ensure_carrier_hook_context().apply_condensation(
          a_half_r.data(), a_half_z.data(),
          cfg.numerics.hydro.rz_momentum_scheme_id == 1
              ? state.node_planar_mass.data()
              : node_mass.data());
    }
  }

  // §22: the constraint must hold at consumption; modifier-chasing is insufficient.
  launch_apply_evac_contact_accel_constraints(
      state,
      a_half_r.data(),
      a_half_z.data(),
      cfg.numerics.hydro.rz_momentum_scheme_id == 0
          ? node_mass.data()
          : state.node_planar_mass.data());
  launch_update_node_velocity_2d(
      state.v_r.data(), state.v_z.data(), ur_old.data(), uz_old.data(),
      a_half_r.data(), a_half_z.data(), d_node_active, d_node_flags,
      n_nodes, dt, state, cfg);
  evacuated_cell_contact_probe(state, cfg, "post_vel_kernel_corr");
  emit_origin_trace(state, "c1_update_v", state.v_z.data());
  emit_origin_trace(state, "post_corrector_v", state.v_z.data());
  if (midpoint_v1) {
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        u_half_r.data() + nw.begin, ur_old.data() + nw.begin,
        state.v_r.data() + nw.begin, nw.count());
    emit_origin_trace(state, "c2_avg_r", state.v_z.data());
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        u_half_z.data() + nw.begin, uz_old.data() + nw.begin,
        state.v_z.data() + nw.begin, nw.count());
    emit_origin_trace(state, "c3_avg_z", state.v_z.data());
  }
  pole_axis_diag::capture_post_accel_velocity(state);
  mesh_trace::trace_post_velocity(state, cfg);
  apply_axis_motion_preflight(u_half_r, state.v_r.data(), r_old, z_old, u_half_z,
                              state.mesh.topo.nr, state.mesh.topo.nz, dt,
                              cfg.numerics.hydro.axis_motion_floor_fraction);
  emit_origin_trace(state, "c4_axis_preflight", state.v_z.data());
  const auto corrector_pole_motion_overlay =
      tenryu::hydro::pole_angular_coarsen::build_motion_overlay(state, cfg);
  core::NodeField1D pos_corrector_r{"h2d:step:pos_corrector_r"};
  core::NodeField1D* corrector_pos_r_ptr = &u_half_r;
  if (corrector_pole_motion_overlay.active) {
    pos_corrector_r.reset(n_nodes);
    cuda_check(cudaMemcpy(pos_corrector_r.data(), u_half_r.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy corrector position r velocity failed");
    emit_origin_trace(state, "c5_copy_pos_r", state.v_z.data());
    corrector_pos_r_ptr = &pos_corrector_r;
  }
  core::NodeField1D pos_corrector_z{"h2d:step:pos_corrector_z"};
  core::NodeField1D* corrector_pos_z_ptr = &u_half_z;
  if (state_supply_active || axis_lagrangian_tangential ||
      corrector_pole_motion_overlay.active) {
    pos_corrector_z.reset(n_nodes);
    cuda_check(cudaMemcpy(pos_corrector_z.data(), u_half_z.data(),
                          static_cast<std::size_t>(n_nodes) * sizeof(double),
                          cudaMemcpyDeviceToDevice),
               "Hydro2D: copy corrector position z velocity failed");
    emit_origin_trace(state, "c6_copy_pos_z", state.v_z.data());
    if (state_supply_active) {
      zero_state_supply_z_position_velocity(pos_corrector_z, cfg, state.mesh.topo.nr,
                                            state.mesh.topo.nz);
      emit_origin_trace(state, "c7_supply_pos_z", state.v_z.data());
    }
    if (axis_lagrangian_tangential) {
      constrain_axis_z_position_velocity(state, cfg, pos_corrector_z, r_old, z_old,
                                         dt, false);
      emit_origin_trace(state, "c8_phase8b_axis_z", state.v_z.data());
    }
    corrector_pos_z_ptr = &pos_corrector_z;
  }
  if (corrector_pole_motion_overlay.active) {
    apply_pole_motion_pilot_position_velocity(
        state,
        cfg,
        corrector_pole_motion_overlay,
        r_old,
        z_old,
        *corrector_pos_r_ptr,
        *corrector_pos_z_ptr,
        dt,
        "corrector");
    emit_origin_trace(state, "c9_pole_motion", state.v_z.data());
  }
  apply_pole_axis_bbsw_order_constraint(state,
                                        cfg,
                                        t_op + 0.5 * dt,
                                        pq_half,
                                        d_hydro_force_active,
                                        d_node_active,
                                        compatible_energy_mode,
                                        z_old,
                                        *corrector_pos_r_ptr,
                                        *corrector_pos_z_ptr,
                                        &u_half_r,
                                        &u_half_z,
                                        &state.v_r,
                                        &state.v_z,
                                        &uz_old,
                                        dt);
  emit_origin_trace(state, "c10_bbsw_constraint", state.v_z.data());
  if (aw_axis_slave_active) {
    launch_apply_aw_axis_velocity_slave_2d(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        true);
    emit_origin_trace(state, "c11_aw_struct_v", state.v_z.data());
    launch_apply_aw_axis_velocity_slave_2d(
        corrector_pos_r_ptr->data(),
        corrector_pos_z_ptr->data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        false);
    emit_origin_trace(state, "c12_aw_struct_pos", state.v_z.data());
    sync_kernel("Hydro2D corrector AW axis velocity slave failed");
  }
  if (aw_axis_slave_multiblock_active) {
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        true);
    emit_origin_trace(state, "c13_aw_mb_v", state.v_z.data());
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        corrector_pos_r_ptr->data(),
        corrector_pos_z_ptr->data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        false);
    emit_origin_trace(state, "c14_aw_mb_pos", state.v_z.data());
    sync_kernel("Hydro2D corrector multiblock AW axis velocity slave failed");
    emit_origin_trace(state, "post_aw_slave", state.v_z.data());
  }
  if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
      state.carrier_node_class.empty()) { /* skip */
  } else {
    auto& carrier = ensure_carrier_hook_context();
    carrier.apply_reconstruct(nullptr, nullptr, state.v_r.data(), state.v_z.data());
    carrier.apply_reconstruct(nullptr, nullptr, u_half_r.data(), u_half_z.data());
    if (corrector_pos_r_ptr->data() != u_half_r.data() ||
        corrector_pos_z_ptr->data() != u_half_z.data()) {
      carrier.apply_reconstruct(nullptr, nullptr, corrector_pos_r_ptr->data(),
                                corrector_pos_z_ptr->data());
    }
  }
  conservation_audit::emit_stage(state, "bbsw_corrector_constraint");
  if (total_energy_identity_check || midpoint_v1 ||
      parse_boundary_2d_type(cfg.numerics.hydro.boundary_2d.r_outer) ==
          Boundary2DType::PRESSURE) {
    core::NodeField1D identity_work_v_r{
        "h2d:step:identity_work_v_r"};
    core::NodeField1D identity_work_v_z{
        "h2d:step:identity_work_v_z"};
    identity_work_v_r.reset(n_nodes);
    identity_work_v_z.reset(n_nodes);
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        identity_work_v_r.data() + nw.begin, ur_old.data() + nw.begin,
        state.v_r.data() + nw.begin, nw.count());
    emit_origin_trace(state, "c15_identity_avg_r", state.v_z.data());
    average_arrays_kernel<<<blocks_nodes, 256>>>(
        identity_work_v_z.data() + nw.begin, uz_old.data() + nw.begin,
        state.v_z.data() + nw.begin, nw.count());
    emit_origin_trace(state, "c16_identity_avg_z", state.v_z.data());
    total_energy_identity_W_ext = compute_pressure_boundary_work_step_2d(
        state, cfg, dt, t_op + 0.5 * dt, identity_work_v_r,
        identity_work_v_z, node_mass);
    if (cap_energy_audit && midpoint_v1) {
      result.cap_energy_audit_W_ext_step =
          total_energy_identity_W_ext;
    }
    if (midpoint_v1 && subzonal_hourglass_enabled(cfg) &&
        cfg.numerics.hydro.hourglass.compatible_work_enabled) {
      core::CellField1D pressure_half_for_hg{
          "h2d:step:midpoint_pressure_half_work"};
      pressure_half_for_hg.reset(n_cells);
      build_cell_pressure_kernel<<<blocks_cells, 256>>>(
          pressure_half_for_hg.data(), P_half.data(), Pi_half.data(),
          d_hydro_force_active, pw.begin, pw.end, n_cells);
      compute_anti_hourglass_acceleration_2d(
          a_hg_half_r, a_hg_half_z, &hourglass_work_erg, state, cfg,
          pressure_half_for_hg, node_mass, a_nonhg_half_r, a_nonhg_half_z,
          d_hydro_force_active, dt, true, d_node_flags,
          identity_work_v_r.data(), identity_work_v_z.data(), dt);
    }
  }
  apply_axis_contact_guard("corrector", cfg, r_old, corrector_pos_r_ptr->data(),
                           node_mass, d_node_active, d_node_flags, n_nodes,
                           nw, dt, state.step);
  emit_origin_trace(state, "c17_contact_guard", state.v_z.data());
  if (aw_axis_slave_active) {
    launch_apply_aw_axis_velocity_slave_2d(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        false);
    emit_origin_trace(state, "c18_aw_struct_reslave_v", state.v_z.data());
    launch_apply_aw_axis_velocity_slave_2d(
        corrector_pos_r_ptr->data(),
        corrector_pos_z_ptr->data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        false);
    emit_origin_trace(state, "c19_aw_struct_reslave_pos", state.v_z.data());
    sync_kernel("Hydro2D corrector AW axis velocity re-slave failed");
  }
  if (aw_axis_slave_multiblock_active) {
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        false);
    emit_origin_trace(state, "c20_aw_mb_reslave_v", state.v_z.data());
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        corrector_pos_r_ptr->data(),
        corrector_pos_z_ptr->data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        false);
    emit_origin_trace(state, "c21_aw_mb_reslave_pos", state.v_z.data());
    sync_kernel(
        "Hydro2D corrector multiblock AW axis velocity re-slave failed");
  }
  core::NodeField1D& corrector_pos_r = *corrector_pos_r_ptr;
  core::NodeField1D& corrector_pos_z = *corrector_pos_z_ptr;
  emit_ring7_seam_scaffold_diagnostics(state,
                                        cfg,
                                        r_old.data(),
                                        z_old.data(),
                                        corrector_pos_r.data(),
                                        corrector_pos_z.data(),
                                        dt,
                                        reduction,
                                        "corrector");
  diagnostics::mesh_diag::dump_hydro_trial(state,
                                           cfg,
                                           "post_corrector_pre_commit",
                                           dt,
                                           r_old,
                                           z_old,
                                           corrector_pos_r,
                                           corrector_pos_z,
                                           &cs_half,
                                           &div_u_half,
                                           "not_recorded_in_hydro",
                                           hydro_half);
  const auto emit_mesh_attr_corrector_trial_failure =
      [&](const tenryu::coupling::HydroStepResult& failure) {
        if (!mesh_attr_active) {
          return;
        }
        core::NodeField1D candidate_r{"h2d:step:candidate_r_c"};
        core::NodeField1D candidate_z{"h2d:step:candidate_z_c"};
        candidate_r.reset(n_nodes);
        candidate_z.reset(n_nodes);
        commit_position_2d_kernel<<<nw.blocks(), 256>>>(
            candidate_r.data(), candidate_z.data(), r_old.data(), z_old.data(),
            corrector_pos_r.data(), corrector_pos_z.data(), d_node_active, d_node_flags,
            nw.begin, nw.end, n_nodes, dt);
        emit_origin_trace(state, "c22_attr_candidate_commit", state.v_z.data());
        sync_kernel("Hydro2D mesh attribution corrector candidate failed");
        emit_mesh_attr_failure(failure, candidate_r.data(), candidate_z.data());
      };
  if (cfg.numerics.hydro.mesh_quality_dt_cfl_enabled || path_guard_enabled) {
    const MeshQualityDtLimit limit = compute_mesh_quality_dt_limit(
        state,
        cfg,
        r_old.data(),
        z_old.data(),
        corrector_pos_r.data(),
        corrector_pos_z.data(),
        dt,
        reduction,
        path_guard_enabled);
    emit_ring7_pole_cap_recomputed_limiter_validation(state,
                                                      cfg,
                                                      limit,
                                                      dt,
                                                      "corrector");
    if (!limit.admissible) {
      if (path_guard_enabled) {
        populate_path_guard_failure(
            result,
            state,
            cfg,
            limit,
            tenryu::coupling::HydroFailureStage::PostCorrector,
            dt);
      } else {
        populate_mesh_quality_dt_failure(
            result,
            state,
            cfg,
            limit,
            tenryu::coupling::HydroFailureStage::PostCorrector);
      }
      if (!path_guard_enabled &&
          limit.metric == MeshQualityFailingMetric::RzVolume) {
        record_ring7_failed_mesh_velocity(state,
                                          cfg,
                                          limit.first_cell,
                                          limit.sigma_safe,
                                          dt,
                                          corrector_pos_r.data(),
                                          corrector_pos_z.data());
      }
      if (mesh_forensics_active) {
        core::NodeField1D candidate_r{"h2d:step:candidate_r_d"};
        core::NodeField1D candidate_z{"h2d:step:candidate_z_d"};
        candidate_r.reset(n_nodes);
        candidate_z.reset(n_nodes);
        commit_position_2d_kernel<<<nw.blocks(), 256>>>(
            candidate_r.data(), candidate_z.data(), r_old.data(), z_old.data(),
            corrector_pos_r.data(), corrector_pos_z.data(), d_node_active, d_node_flags,
            nw.begin, nw.end, n_nodes, dt);
        emit_origin_trace(state, "c23_quality_candidate_commit", state.v_z.data());
        sync_kernel("Hydro2D mesh forensics corrector candidate failed");
        populate_mesh_forensics_sample(result,
                                       state,
                                       r_old,
                                       z_old,
                                       candidate_r,
                                       candidate_z,
                                       ur_old,
                                       uz_old,
                                       u_half_r,
                                       u_half_z,
                                       state.v_r,
                                       state.v_z,
                                       attr_pressure_half_r,
                                       attr_pressure_half_z,
                                       attr_q_half_r,
                                       attr_q_half_z,
                                       a_hg_half_r,
                                       a_hg_half_z,
                                       node_mass);
      }
      if (!path_guard_enabled) {
        tenryu::core::log_warning(
            "Hydro2D mesh-quality dt CFL rejected corrector trial at step=" +
            std::to_string(state.step) + ", reason=" +
            std::string(result.reason != nullptr ? result.reason : "") +
            ", first_cell=" + std::to_string(result.first_failing_cell) +
            ", first_corner_or_gauss=" +
            std::to_string(result.first_failing_corner) +
            ", sigma_safe=" + format_scientific(limit.sigma_safe) +
            ", suggested_dt=" + format_scientific(result.suggested_dt));
      }
      emit_mesh_attr_corrector_trial_failure(result);
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  if (multiblock_path_admissibility_precommit_enabled(state, cfg)) {
    const auto pole_overlay = build_pole_path_overlay(state, cfg);
    const auto* pole_overlay_ptr = pole_overlay.active ? &pole_overlay : nullptr;
    const auto path_check = tenryu::mesh::evaluate_path_admissibility(
        state,
        cfg,
        r_old.data(),
        z_old.data(),
        corrector_pos_r.data(),
        corrector_pos_z.data(),
        dt,
        cfg.numerics.ale.path_admissibility_floor,
        pole_overlay_ptr,
        state.x_r_reference.size() == state.x_r.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    observe_path_admissibility_margin(result, path_check);
    const bool forecast_due = mesh_forecast_or_shell_replay_due(state);
    if (forecast_due) {
      const auto forecast = tenryu::mesh::evaluate_mesh_forecast(
          state,
          cfg,
          r_old.data(),
          z_old.data(),
          corrector_pos_r.data(),
          corrector_pos_z.data(),
          dt,
          cfg.numerics.ale.path_admissibility_floor,
          mesh_forecast_diag_config().q_warn,
          path_check,
          pole_overlay_ptr);
      populate_mesh_forecast_result(result, forecast);
      if (mesh_forecast_diag_due(state)) {
        emit_mesh_forecast_diag(state, "corrector", dt, path_check, forecast);
      }
      tenryu::hydro::ale::maybe_run_shell_rezone_replay_probe(
          state,
          cfg,
          forecast,
          r_old.data(),
          z_old.data(),
          corrector_pos_r.data(),
          corrector_pos_z.data(),
          dt,
          "corrector",
          eos_ctx,
          reduction);
    }
    if (multiblock_path_admissibility_failed(path_check)) {
      populate_multiblock_path_admissibility_failure(
          result,
          state,
          cfg,
          path_check,
          dt,
          tenryu::coupling::HydroFailureStage::PostCorrector);
      ++state.multiblock_path_dt_rejection_count;
      if (mesh_forensics_active) {
        core::NodeField1D candidate_r{"h2d:step:candidate_r_e"};
        core::NodeField1D candidate_z{"h2d:step:candidate_z_e"};
        candidate_r.reset(n_nodes);
        candidate_z.reset(n_nodes);
        commit_position_2d_kernel<<<nw.blocks(), 256>>>(
            candidate_r.data(), candidate_z.data(), r_old.data(), z_old.data(),
            corrector_pos_r.data(), corrector_pos_z.data(), d_node_active, d_node_flags,
            nw.begin, nw.end, n_nodes, dt);
        emit_origin_trace(state, "c24_path_candidate_commit", state.v_z.data());
        sync_kernel("Hydro2D mesh forensics corrector path candidate failed");
        populate_mesh_forensics_sample(result,
                                       state,
                                       r_old,
                                       z_old,
                                       candidate_r,
                                       candidate_z,
                                       ur_old,
                                       uz_old,
                                       u_half_r,
                                       u_half_z,
                                       state.v_r,
                                       state.v_z,
                                       attr_pressure_half_r,
                                       attr_pressure_half_z,
                                       attr_q_half_r,
                                       attr_q_half_z,
                                       a_hg_half_r,
                                       a_hg_half_z,
                                       node_mass);
      }
      tenryu::core::log_warning(
          "Hydro2D multiblock path admissibility rejected corrector trial at step=" +
          std::to_string(state.step) + ", first_cell=" +
          std::to_string(path_check.first_failing_cell) + ", lambda=" +
          format_scientific(path_check.first_failing_lambda) +
          format_multiblock_path_admissibility_location(state, cfg, path_check) +
          format_multiblock_path_admissibility_anatomy(path_check) +
          ", min_margin=" + format_scientific(path_check.min_margin) +
          ", suggested_dt=" + format_scientific(result.suggested_dt));
      emit_mesh_attr_corrector_trial_failure(result);
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  // PR 4: when mesh_quality_dt_cfl_enabled is true, compute_mesh_quality_dt_limit
  // supersedes this endpoint-shrink trial-volume CFL. The old path is retained
  // for legacy decks; endpoint shrink is insufficient for state-sensitive
  // axis-cell failures because it does not check the full candidate trajectory.
  if (cfg.numerics.hydro.trial_volume_cfl_enabled && state.dt_prev_hydro > 0.0 &&
      !cfg.numerics.hydro.mesh_quality_dt_cfl_enabled) {
    const auto trial = check_trial_lagrangian_volume_cfl_fields(
        state.mesh.topo.nr, state.mesh.topo.nz, r_old, z_old, corrector_pos_r, corrector_pos_z,
        V_old, dt, cfg.numerics.hydro.trial_volume_cfl_floor_fraction,
        cfg.numerics.hydro.trial_volume_cfl_shrink_fraction,
        hydro_force_active_host, reduction);
    diagnostics::mesh_diag::dump_trial_volume_cfl(state,
                                                  cfg,
                                                  true,
                                                  trial.admissible,
                                                  trial.min_trial_vol_ratio,
                                                  trial.suggested_dt,
                                                  trial.first_failing_cell);
    if (!trial.admissible) {
      const double suggested_dt = std::max(trial.suggested_dt,
                                           cfg.numerics.dt.min_s);
      tenryu::core::log_warning(
          "Hydro2D trial-volume CFL rejected corrector trial at step=" +
          std::to_string(state.step) + ", first_cell=" +
          std::to_string(trial.first_failing_cell) + ", min_trial_vol_ratio=" +
          format_scientific(trial.min_trial_vol_ratio) + ", floor_fraction=" +
          format_scientific(cfg.numerics.hydro.trial_volume_cfl_floor_fraction) +
          ", suggested_dt=" + format_scientific(suggested_dt) +
          (cfg.numerics.hydro.driver_full_step_retry_enabled
               ? " (driver retry requested)"
               : " (diagnostic-only: driver step retry is disabled)"));
      if (cfg.numerics.hydro.driver_full_step_retry_enabled) {
        result.admissible = false;
        result.retry_required = true;
        result.reason = "trial_volume_cfl";
        result.first_failing_cell = trial.first_failing_cell;
        result.first_failing_corner = -1;
        result.first_failing_i =
            trial.first_failing_cell < 0
                ? -1
                : trial.first_failing_cell / state.mesh.topo.nz;
        result.first_failing_j =
            trial.first_failing_cell < 0
                ? -1
                : trial.first_failing_cell % state.mesh.topo.nz;
        result.min_metric = trial.min_trial_vol_ratio;
        result.suggested_dt = suggested_dt;
        emit_mesh_attr_corrector_trial_failure(result);
        cleanup_precommit_retry_buffers();
        return result;
      }
    }
  } else {
    diagnostics::mesh_diag::dump_trial_volume_cfl(state,
                                                  cfg,
                                                  false,
                                                  true,
                                                  1.0,
                                                  dt,
                                                  -1);
  }
  if (cfg.numerics.hydro.corner_jacobian_ale_trigger_enabled) {
    const auto corner_trial = check_trial_lagrangian_corner_jacobian_fields(
        state.mesh.topo.nr, state.mesh.topo.nz, r_old, z_old, corrector_pos_r, corrector_pos_z,
        dt, cfg.numerics.hydro.corner_jacobian_floor_eps, reduction,
        *hydro_force_active_host, cfg.numerics.hydro.axis_margin_guard_enabled,
        cfg.numerics.has_physical_rz_axis,
        state.evacuated_cells.cell_axis_edge_collapsed.empty()
            ? nullptr
            : &state.evacuated_cells.cell_axis_edge_collapsed,
        state.evacuated_cells.geometry_policy_exempt_cells.empty()
            ? nullptr
            : &state.evacuated_cells.geometry_policy_exempt_cells);
    if (!corner_trial.admissible) {
      double jdot = std::numeric_limits<double>::quiet_NaN();
      double jdot_n = std::numeric_limits<double>::quiet_NaN();
      double jdot_t = std::numeric_limits<double>::quiet_NaN();
      double chi_t = std::numeric_limits<double>::quiet_NaN();
      const int failing_cell = corner_trial.first_failing_cell;
      const int failing_corner = corner_trial.first_failing_corner;
      if (failing_cell >= 0 &&
          failing_cell < state.mesh.topo.nr * state.mesh.topo.nz &&
          failing_corner >= 0 && failing_corner < 4) {
        const int failing_i = failing_cell / state.mesh.topo.nz;
        const int failing_j = failing_cell % state.mesh.topo.nz;
        const int cell_nodes[4] = {
            state.mesh.topo.node_index(failing_i, failing_j),
            state.mesh.topo.node_index(failing_i + 1, failing_j),
            state.mesh.topo.node_index(failing_i + 1, failing_j + 1),
            state.mesh.topo.node_index(failing_i, failing_j + 1)};
        double node_r[4] = {};
        double node_z[4] = {};
        double node_v_r[4] = {};
        double node_v_z[4] = {};
        for (int k = 0; k < 4; ++k) {
          cuda_check(cudaMemcpy(&node_r[k],
                                r_old.data() + cell_nodes[k],
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "corner-J shear telemetry: copy node r failed");
          cuda_check(cudaMemcpy(&node_z[k],
                                z_old.data() + cell_nodes[k],
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "corner-J shear telemetry: copy node z failed");
          cuda_check(cudaMemcpy(&node_v_r[k],
                                state.v_r.data() + cell_nodes[k],
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "corner-J shear telemetry: copy node v_r failed");
          cuda_check(cudaMemcpy(&node_v_z[k],
                                state.v_z.data() + cell_nodes[k],
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "corner-J shear telemetry: copy node v_z failed");
        }

        const int seam_node_0 = state.mesh.topo.node_index(0, failing_j);
        const int seam_node_1 = state.mesh.topo.node_index(0, failing_j + 1);
        double seam_r[2] = {};
        double seam_z[2] = {};
        cuda_check(cudaMemcpy(&seam_r[0],
                              r_old.data() + seam_node_0,
                              sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "corner-J shear telemetry: copy seam r0 failed");
        cuda_check(cudaMemcpy(&seam_z[0],
                              z_old.data() + seam_node_0,
                              sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "corner-J shear telemetry: copy seam z0 failed");
        cuda_check(cudaMemcpy(&seam_r[1],
                              r_old.data() + seam_node_1,
                              sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "corner-J shear telemetry: copy seam r1 failed");
        cuda_check(cudaMemcpy(&seam_z[1],
                              z_old.data() + seam_node_1,
                              sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "corner-J shear telemetry: copy seam z1 failed");

        const double tangent_r = seam_r[1] - seam_r[0];
        const double tangent_z = seam_z[1] - seam_z[0];
        const double tangent_norm = std::hypot(tangent_r, tangent_z);
        if (tangent_norm > 0.0 && std::isfinite(tangent_norm)) {
          const double t_hat_r = tangent_r / tangent_norm;
          const double t_hat_z = tangent_z / tangent_norm;
          const double n_hat_r = -t_hat_z;
          const double n_hat_z = t_hat_r;
          const int previous = (failing_corner + 3) & 3;
          const int next = (failing_corner + 1) & 3;
          const double e_minus_r = node_r[previous] - node_r[failing_corner];
          const double e_minus_z = node_z[previous] - node_z[failing_corner];
          const double e_plus_r = node_r[next] - node_r[failing_corner];
          const double e_plus_z = node_z[next] - node_z[failing_corner];
          const double dv_next_r = node_v_r[next] - node_v_r[failing_corner];
          const double dv_next_z = node_v_z[next] - node_v_z[failing_corner];
          const double dv_previous_r =
              node_v_r[previous] - node_v_r[failing_corner];
          const double dv_previous_z =
              node_v_z[previous] - node_v_z[failing_corner];
          const auto cross = [](const double a_r,
                                const double a_z,
                                const double b_r,
                                const double b_z) {
            return a_r * b_z - a_z * b_r;
          };
          jdot = cross(dv_next_r, dv_next_z, e_minus_r, e_minus_z) +
                 cross(e_plus_r,
                       e_plus_z,
                       dv_previous_r,
                       dv_previous_z);
          const auto projected_jdot = [&](const double direction_r,
                                          const double direction_z) {
            const double next_projection =
                dv_next_r * direction_r + dv_next_z * direction_z;
            const double previous_projection =
                dv_previous_r * direction_r +
                dv_previous_z * direction_z;
            return cross(next_projection * direction_r,
                         next_projection * direction_z,
                         e_minus_r,
                         e_minus_z) +
                   cross(e_plus_r,
                         e_plus_z,
                         previous_projection * direction_r,
                         previous_projection * direction_z);
          };
          jdot_t = projected_jdot(t_hat_r, t_hat_z);
          jdot_n = projected_jdot(n_hat_r, n_hat_z);
          chi_t = std::max(-jdot_t, 0.0) /
                  (std::max(-jdot, 0.0) + 1.0e-300);
        }
      }
      char corner_rate_telemetry[160];
      std::snprintf(corner_rate_telemetry,
                    sizeof(corner_rate_telemetry),
                    " chi_t=%.3f Jdot_n=%.3e Jdot_t=%.3e",
                    chi_t,
                    jdot_n,
                    jdot_t);
      tenryu::core::log_warning(
          "Hydro2D corner-J guard detected inadmissible corrector trial at step=" +
          std::to_string(state.step) + ", first_cell=" +
          std::to_string(corner_trial.first_failing_cell) + ", first_corner=" +
          std::to_string(corner_trial.first_failing_corner) + ", min_current_J=" +
          format_scientific(corner_trial.min_current_j) + ", min_trial_J=" +
          format_scientific(corner_trial.min_trial_j) + ", min_trial_ratio=" +
          format_scientific(corner_trial.min_trial_ratio) + ", suggested_dt=" +
          format_scientific(corner_trial.suggested_dt) + corner_rate_telemetry +
          (cfg.numerics.hydro.driver_full_step_retry_enabled
               ? " (driver retry requested)"
               : " (diagnostic-only: driver step retry is disabled)"));
      if (cfg.numerics.hydro.driver_full_step_retry_enabled) {
        result.admissible = false;
        result.retry_required = true;
        result.reason = "corner_j";
        result.first_failing_cell = corner_trial.first_failing_cell;
        result.first_failing_corner = corner_trial.first_failing_corner;
        result.first_failing_i =
            corner_trial.first_failing_cell < 0
                ? -1
                : corner_trial.first_failing_cell / state.mesh.topo.nz;
        result.first_failing_j =
            corner_trial.first_failing_cell < 0
                ? -1
                : corner_trial.first_failing_cell % state.mesh.topo.nz;
        int band = cfg.numerics.hydro.axis_guard_band_cells;
        bool in_band =
            result.first_failing_i >= 0 && result.first_failing_i <= band;
        result.retry_action =
            in_band
                ? tenryu::coupling::RetryActionHint::ForceAxisSpinePlusLocalAle
                : tenryu::coupling::RetryActionHint::ReduceDtOnly;
        if (result.first_failing_i == 0) {
          result.regime = MeshFailureRegime::AxisFace;
        } else if (in_band) {
          result.regime = MeshFailureRegime::AxisBand;
        }
        result.min_metric =
            std::min(corner_trial.min_trial_j, corner_trial.min_current_j);
        result.suggested_dt = corner_trial.suggested_dt;
        emit_mesh_attr_corrector_trial_failure(result);
        cleanup_precommit_retry_buffers();
        return result;
      }
    }
  }
  if (cfg.numerics.hydro.axis_projection.enabled) {
    axis_projection::observe_precommit(
        state,
        cfg,
        r_old.data(),
        z_old.data(),
        corrector_pos_r.data(),
        corrector_pos_z.data(),
        use_two_temp ? ei_old.data() : e_old.data(),
        dt,
        t_op + dt,
        reduction,
        part.rank);
  }
  commit_position_2d_kernel<<<nw.blocks(), 256>>>(
      state.x_r.data(), state.x_z.data(), r_old.data(), z_old.data(),
      corrector_pos_r.data(), corrector_pos_z.data(), d_node_active, d_node_flags,
      nw.begin, nw.end, n_nodes, dt);
  emit_origin_trace(state, "c25_commit_pos", state.v_z.data());
  sync_kernel("Hydro2D corrector_update kernel failed");
  apply_boundary_2d(state, cfg, t_op + dt);
  emit_origin_trace(state, "c26_boundary", state.v_z.data());
  if (state_supply_active) {
    restore_state_supply_material_velocity(state, cfg);
    emit_origin_trace(state, "c27_supply_restore", state.v_z.data());
  }
  enforce_node_axis_state(state, d_node_flags);
  emit_origin_trace(state, "c28_enforce_axis", state.v_z.data());
  emit_origin_trace(state, "post_enforce", state.v_z.data());
  if (aw_axis_slave_active) {
    launch_apply_aw_axis_velocity_slave_2d(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_kappa_saved_.data(),
        state.mesh.topo.nr,
        state.mesh.topo.nz,
        aw_axis_slave_first_i,
        aw_axis_slave_theta0_active_,
        aw_axis_slave_theta_pi_active_,
        false);
    if (aw_axis_snap_env_enabled) {
      launch_snap_aw_axis_coordinates_2d(
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_kappa_saved_.data(),
          state.mesh.topo.nr,
          state.mesh.topo.nz,
          aw_axis_slave_first_i,
          aw_axis_slave_theta0_active_,
          aw_axis_slave_theta_pi_active_);
    }
    sync_kernel("Hydro2D post-boundary corrector AW axis closure failed");
  }
  if (aw_axis_slave_multiblock_active) {
    launch_apply_aw_axis_velocity_slave_2d_multiblock(
        state.v_r.data(),
        state.v_z.data(),
        state.x_r.data(),
        state.x_z.data(),
        aw_axis_slave_multiblock_kappa_saved_.data(),
        aw_axis_slave_p_.data(),
        aw_axis_slave_q_.data(),
        static_cast<int>(aw_axis_slave_p_.size()),
        false);
    if (aw_axis_snap_env_enabled) {
      launch_snap_aw_axis_coordinates_2d_multiblock(
          state.x_r.data(),
          state.x_z.data(),
          aw_axis_slave_multiblock_kappa_saved_.data(),
          aw_axis_slave_p_.data(),
          aw_axis_slave_q_.data(),
          static_cast<int>(aw_axis_slave_p_.size()));
    }
    sync_kernel(
        "Hydro2D post-boundary corrector multiblock AW axis closure failed");
  }
  if (!state.carrier_domain_active || !state.boundary_carrier.valid ||
      state.carrier_node_class.empty()) { /* skip */
  } else {
    ensure_carrier_hook_context().apply_reconstruct(
        state.x_r.data(), state.x_z.data(), state.v_r.data(), state.v_z.data());
  }
  // Consultation #23 §11.4: finalize contact after the last stage-velocity writer.
  launch_apply_evac_contact_velocity_constraints(
      state, state.v_r.data(), state.v_z.data(),
      cfg.numerics.hydro.rz_momentum_scheme_id == 0
          ? node_mass.data()
          : state.node_planar_mass.data(),
      state.x_r.data(), state.x_z.data(),
      cs_half.data(),
      d_hydro_force_active,
      dt,
      state.evacuated_cells.d_contact_pair_dk.data(),
      cfg.numerics.ale.evacuated_cell.closure_contact
          .mortar_position_drift_beta);
  evacuated_cell_contact_probe(state, cfg, "post_vel_corrector");
  central_pseudo_core::rebuild_virtual_member_geometry(
      state, cfg, "post_corrector");
  // OPEN-HYDRO-CORNER: same as the predictor — exchange ghost node coords
  // before the post-corrector geometry refresh and work/energy stages.
  exchange_mid_nodes_2d();
  mesh_trace::trace_post_lagrange_position(state, cfg, r_old, z_old);
  mesh_trace::trace_cell0_geometry(state, cfg, "post_lagrange_position");
  if (multiblock_path_admissibility_precommit_enabled(state, cfg)) {
    const auto pole_overlay = build_pole_path_overlay(state, cfg);
    const auto* pole_overlay_ptr = pole_overlay.active ? &pole_overlay : nullptr;
    const auto path_check = tenryu::mesh::evaluate_path_admissibility(
        state,
        cfg,
        r_old.data(),
        z_old.data(),
        state.x_r.data(),
        state.x_z.data(),
        cfg.numerics.ale.path_admissibility_floor,
        pole_overlay_ptr,
        state.x_r_reference.size() == state.x_r.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    observe_path_admissibility_margin(result, path_check);
    const bool forecast_due = mesh_forecast_post_commit_due(state);
    if (forecast_due) {
      const auto forecast = tenryu::mesh::evaluate_mesh_forecast(
          state,
          cfg,
          r_old.data(),
          z_old.data(),
          state.x_r.data(),
          state.x_z.data(),
          cfg.numerics.ale.path_admissibility_floor,
          mesh_forecast_diag_config().q_warn,
          path_check,
          pole_overlay_ptr);
      populate_mesh_forecast_result(result, forecast);
      if (mesh_forecast_diag_due(state)) {
        emit_mesh_forecast_diag(state, "post_corrector_commit", dt, path_check,
                                forecast);
      }
      result.shell_subcycle_committed =
          tenryu::hydro::ale::maybe_run_shell_rezone_replay_probe(
              state,
              cfg,
              forecast,
              state.x_r.data(),
              state.x_z.data(),
              corrector_pos_r.data(),
              corrector_pos_z.data(),
              dt,
              "post_corrector_commit",
              eos_ctx,
              reduction);
    }
    if (multiblock_path_admissibility_failed(path_check)) {
      populate_multiblock_path_admissibility_failure(
          result,
          state,
          cfg,
          path_check,
          dt,
          tenryu::coupling::HydroFailureStage::PostCorrector);
      ++state.multiblock_path_dt_rejection_count;
      tenryu::core::log_warning(
          "[multiblock-path-admissibility] step=" +
          std::to_string(state.step) + " first_cell=" +
          std::to_string(path_check.first_failing_cell) + " lambda=" +
          format_scientific(path_check.first_failing_lambda) +
          format_multiblock_path_admissibility_location(state, cfg, path_check) +
          format_multiblock_path_admissibility_anatomy(path_check) +
          " min_margin=" + format_scientific(path_check.min_margin) +
          " dt: " + format_scientific(dt) + " -> " +
          format_scientific(result.suggested_dt));
      emit_mesh_attr_failure(result);
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  if (cfg.numerics.hydro.in_hydro_corner_j_guard_enabled ||
      cfg.numerics.hydro.in_hydro_gauss_j_guard_enabled ||
      cfg.numerics.hydro.in_hydro_rz_volume_guard_enabled) {
    result = compute_in_hydro_candidate_mesh_guard(
        state,
        dt,
        r_old.data(),
        z_old.data(),
        state.x_r.data(),
        state.x_z.data(),
        cfg,
        tenryu::coupling::HydroFailureStage::PostCorrector,
        reduction,
        d_hydro_force_active,
        cfg.numerics.hydro.regime_aware_corner_j_guard_enabled
            ? active_mesh_regime_cache->current()
            : nullptr);
    if (!result.admissible) {
      if (mesh_forensics_active) {
        populate_mesh_forensics_sample(result,
                                       state,
                                       r_old,
                                       z_old,
                                       state.x_r,
                                       state.x_z,
                                       ur_old,
                                       uz_old,
                                       u_half_r,
                                       u_half_z,
                                       state.v_r,
                                       state.v_z,
                                       attr_pressure_half_r,
                                       attr_pressure_half_z,
                                       attr_q_half_r,
                                       attr_q_half_z,
                                       a_hg_half_r,
                                       a_hg_half_z,
                                       node_mass);
      }
      emit_mesh_attr_failure(result);
      cleanup_precommit_retry_buffers();
      return result;
    }
  }
  result = refresh_geometry_and_density_impl(
      state,
      cfg,
      tenryu::coupling::HydroFailureStage::PostCorrector,
      d_rho_clamp_count,
      observability,
      result.shell_subcycle_committed);
  if (!result.admissible) {
    emit_mesh_attr_failure(result);
    cleanup_precommit_retry_buffers();
    return result;
  }
  upload_mesh_cache(state, mesh_cache);
  ensure_hourglass_subzonal_masses_2d(state, cfg, false);
  ale::csr_optionb_canonicalize_corner_mass_basis(state, cfg);
  conservation_audit::emit_stage(state, "post_corrector_lagrange_geometry");
  if (i1b_spurious_sensor) {
    result.i1b_hydro_nonaffine_sensor =
        compute_hydro_nonaffine_ke_sensor_2d(
            state, d_hydro_active, i1b_spurious_sensor_top_k());
  }

  if (compatible_energy_mode) {
    core::NodeField1D compatible_work_v_r{"h2d:step:compatible_work_v_r"};
    core::NodeField1D compatible_work_v_z{"h2d:step:compatible_work_v_z"};
    compatible_work_v_r.reset(n_nodes);
    compatible_work_v_z.reset(n_nodes);
    average_arrays_kernel<<<nw.blocks(), 256>>>(
        compatible_work_v_r.data() + nw.begin, ur_old.data() + nw.begin,
        state.v_r.data() + nw.begin, nw.count());
    average_arrays_kernel<<<nw.blocks(), 256>>>(
        compatible_work_v_z.data() + nw.begin, uz_old.data() + nw.begin,
        state.v_z.data() + nw.begin, nw.count());
    sync_kernel("Hydro2D compatible work velocity average failed");
    if (midpoint_v1) {
      recompute_midpoint_corner_work_2d(
          state, cfg, compatible_work_v_r.data(),
          compatible_work_v_z.data(), d_hydro_force_active);
    } else {
      compatible::launch_compute_compatible_work_2d(
          state, cfg, compatible_work_v_r.data(), compatible_work_v_z.data(),
          d_hydro_force_active);
      update_compatible_subzonal_pressure_observability_2d(state, cfg);
    }
    sync_kernel("Hydro2D compatible corrector work recompute failed");
    apply_pole_axis_bbsw_work_correction(state,
                                         cfg,
                                         dt,
                                         uz_old,
                                         state.v_z,
                                         node_mass,
                                         pole_axis_bbsw_true_force_half_z,
                                         d_node_active,
                                         d_hydro_force_active,
                                         compatible_energy_mode);
    central_pseudo_core::set_boundary_pressure_work(
        state,
        cfg,
        dt,
        node_mass,
        ur_old.data(),
        uz_old.data(),
        state.v_r.data(),
        state.v_z.data());
    pole_angular_derefine::set_boundary_pressure_work(
        state,
        cfg,
        dt,
        ur_old.data(),
        uz_old.data(),
        state.v_r.data(),
        state.v_z.data());
    if (cap_energy_audit) {
      const CapEnergyAuditCompatibleTerms audit_terms =
          compute_cap_energy_audit_compatible_terms(
              state,
              dt,
              braginskii::params_from_config(cfg).enabled
                  ? state.work_visc_per_cell.data()
                  : nullptr,
              compatible_work_v_r,
              compatible_work_v_z,
              cap_energy_audit_force_half_r,
              cap_energy_audit_force_half_z,
              ur_old,
              uz_old,
              node_mass,
              d_node_active,
              d_node_flags);
      result.cap_energy_audit_W_pq = audit_terms.W_pq;
      result.cap_energy_audit_W_uF = audit_terms.W_uF;
      result.cap_energy_audit_W_apex_pinned = audit_terms.W_apex_pinned;
    }
    conservation_audit::emit_stage(state, "post_compatible_work_setup");
  }

  double* d_hydro_floor = nullptr;
  int* d_hydro_clamp_count = nullptr;
  const double zero_d = 0.0;
  d_hydro_floor = static_cast<double*>(
      core::device_scratch_acquire("h2d:hydro_floor", sizeof(double)));
  d_hydro_clamp_count = static_cast<int*>(
      core::device_scratch_acquire("h2d:hydro_clamp_count", sizeof(int)));
  cuda_check(cudaMemcpy(d_hydro_floor, &zero_d, sizeof(double), cudaMemcpyHostToDevice),
             "Hydro2D: init hydro floor failed");
  cuda_check(cudaMemcpy(d_hydro_clamp_count, &zero_i, sizeof(int), cudaMemcpyHostToDevice),
             "Hydro2D: init hydro clamp_count failed");

  core::CellField1D compatible_zero_energy_source{"h2d:step:compatible_zero_energy_source"};
  if (compatible_energy_mode) {
    compatible_zero_energy_source.reset(n_cells);
  }
  const core::CellField1D& Pe_energy_half =
      compatible_energy_mode ? compatible_zero_energy_source : P_half;
  const core::CellField1D& Pi_energy_half =
      compatible_energy_mode ? compatible_zero_energy_source : Pi_half;
  const core::CellField1D& Q_energy_half =
      compatible_energy_mode ? compatible_zero_energy_source : Q_half;
  const FloorClampExclusionMasks exclusion_masks =
      floor_clamp_exclusion_masks(state);

  if (use_two_temp && per_material_hydro_enabled) {
    Hydro2DMaterialParams* d_params = nullptr;
    std::uint8_t* d_te_valid = nullptr;
    std::uint8_t* d_ti_valid = nullptr;
    unsigned long long* d_counts = nullptr;

    const std::vector<Hydro2DMaterialParams> h_params =
        make_hydro2d_material_params(cfg);
    d_params = static_cast<Hydro2DMaterialParams*>(core::device_scratch_acquire(
        "h2d:permat_params", h_params.size() * sizeof(Hydro2DMaterialParams)));
    cuda_check(cudaMemcpy(d_params,
                          h_params.data(),
                          h_params.size() * sizeof(Hydro2DMaterialParams),
                          cudaMemcpyHostToDevice),
               "Hydro2D: copy per-material energy params failed");
    const bool lazy_cache_enabled = cfg.numerics.materials.lazy_cache_te_m_enabled;
    if (lazy_cache_enabled) {
      TENRYU_ASSERT(state.Te_per_material.size() == n_cell_mat,
                    "Hydro2D per-material energy requires Te_per_material");
      TENRYU_ASSERT(state.Ti_per_material.size() == n_cell_mat,
                    "Hydro2D per-material energy requires Ti_per_material");
      TENRYU_ASSERT(state.Te_per_material_valid.size() == n_cell_mat,
                    "Hydro2D per-material energy requires Te valid flags");
      TENRYU_ASSERT(state.Ti_per_material_valid.size() == n_cell_mat,
                    "Hydro2D per-material energy requires Ti valid flags");
      d_te_valid = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "h2d:permat_te_valid", n_cell_mat * sizeof(std::uint8_t)));
      d_ti_valid = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "h2d:permat_ti_valid", n_cell_mat * sizeof(std::uint8_t)));
      cuda_check(cudaMemcpy(d_te_valid,
                            state.Te_per_material_valid.data(),
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice),
                 "Hydro2D: copy per-material Te valid failed");
      cuda_check(cudaMemcpy(d_ti_valid,
                            state.Ti_per_material_valid.data(),
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyHostToDevice),
                 "Hydro2D: copy per-material Ti valid failed");
    }
    d_counts = static_cast<unsigned long long*>(core::device_scratch_acquire(
        "h2d:permat_counts",
        per_material::kPerMaterialCounterCount * sizeof(unsigned long long)));
    cuda_check(cudaMemset(d_counts,
                          0,
                          per_material::kPerMaterialCounterCount *
                              sizeof(unsigned long long)),
               "Hydro2D: cudaMemset per-material energy counts failed");

    const auto* d_electron_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_electron_views
                                                              : nullptr;
    const auto* d_ion_views =
        (eos_ctx != nullptr && eos_ctx->n_materials >= n_mat) ? eos_ctx->d_ion_views
                                                              : nullptr;
    const int blocks_cm = (static_cast<int>(n_cell_mat) + 255) / 256;
    energy_update_with_old_volume_2t_per_material_kernel<<<blocks_cm, 256>>>(
        state.Ee_per_material.data(),
        state.Ei_per_material.data(),
        state.mass_per_material.data(),
        state.volFrac.data(),
        state.vol.data(),
        V_old.data(),
        V_half.data(),
        Qvisc_per_material.data(),
        lazy_cache_enabled ? state.Te_per_material.data() : nullptr,
        lazy_cache_enabled ? state.Ti_per_material.data() : nullptr,
        d_te_valid,
        d_ti_valid,
        d_counts,
        d_electron_views,
        d_ion_views,
        d_params,
        d_hydro_active,
        exclusion_masks.central_member,
        exclusion_masks.central_passive,
        exclusion_masks.pole_member,
        n_cells,
        n_mat,
        cfg.numerics.floors.Te,
        cfg.numerics.floors.Ti,
        cfg.numerics.materials.presence_threshold_volfrac,
        cfg.numerics.materials.presence_threshold_mass_density_g_per_cc,
        q_heat_to_electron,
        lazy_cache_enabled,
        cfg.materials.low_density_extrapolation,
        d_hydro_floor,
        d_hydro_clamp_count);
    sync_kernel("Hydro2D per-material energy_update kernel failed");
    state.dispatch_counters.per_material_kernel_call_count.fetch_add(
        1, std::memory_order_relaxed);

    if (lazy_cache_enabled) {
      cuda_check(cudaMemcpy(state.Te_per_material_valid.data(),
                            d_te_valid,
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyDeviceToHost),
                 "Hydro2D: copy per-material Te valid back failed");
      cuda_check(cudaMemcpy(state.Ti_per_material_valid.data(),
                            d_ti_valid,
                            n_cell_mat * sizeof(std::uint8_t),
                            cudaMemcpyDeviceToHost),
                 "Hydro2D: copy per-material Ti valid back failed");
    }
    unsigned long long h_counts[per_material::kPerMaterialCounterCount] = {};
    cuda_check(cudaMemcpy(h_counts,
                          d_counts,
                          per_material::kPerMaterialCounterCount *
                              sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost),
               "Hydro2D: copy per-material energy counts failed");
    state.dispatch_counters.eos_inverse_call_count.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[per_material::kPerMaterialCounterEOSInverse]),
        std::memory_order_relaxed);
    state.dispatch_counters.lazy_cache_te_m_hit_count.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[per_material::kPerMaterialCounterLazyCacheHit]),
        std::memory_order_relaxed);
    state.dispatch_counters.lazy_cache_te_m_miss_count.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[per_material::kPerMaterialCounterLazyCacheMiss]),
        std::memory_order_relaxed);
    state.dispatch_counters.eos_table_validity_violations.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[per_material::kPerMaterialCounterEOSTableValidityViolation]),
        std::memory_order_relaxed);
    state.dispatch_counters.presence_absent_events.fetch_add(
        static_cast<std::uint64_t>(
            h_counts[per_material::kPerMaterialCounterPresenceAbsent]),
        std::memory_order_relaxed);
    per_material::refresh_per_material_derived_cell_fields(state, cfg, eos_ctx, true);

  } else if (use_two_temp) {
    const auto& mat = cfg.materials.materials.front();
    const double* cv_e_ptr = state.cv_e.empty() ? nullptr : state.cv_e.data();
    const double* cv_i_ptr = state.cv_i.empty() ? nullptr : state.cv_i.data();
    const double* qei_Te =
        cfg.numerics.hydro.qei_evaluate_at_t_n ? Te_n_for_qei.data()
                                               : Te_half.data();
    const double* qei_Ti =
        cfg.numerics.hydro.qei_evaluate_at_t_n ? Ti_n_for_qei.data()
                                               : Ti_half.data();
    energy_update_with_old_volume_2t_kernel<<<cw.blocks(), 256>>>(
        state.ee.data(), state.ei.data(), e_old.data(), ei_old.data(), rho_half.data(),
        qei_Te, qei_Ti, state.vol.data(), V_old.data(),
        state.mass.data(), Pe_energy_half.data(), Pi_energy_half.data(),
        Q_energy_half.data(),
        state.zbar.data(), d_hydro_force_active, exclusion_masks.central_member,
        exclusion_masks.central_passive, exclusion_masks.pole_member, cw.begin,
        cw.end, n_cells, dt,
        mat.ideal_gas_gamma, mat.A, mat.Z, cfg.numerics.hydro.qei_multiplier,
        eos_views.tab_ion, eos_views.tab_ele,
        cv_e_ptr, cv_i_ptr, q_heat_to_electron, d_hydro_floor, d_hydro_clamp_count);
  } else {
    energy_update_with_old_volume_kernel<<<cw.blocks(), 256>>>(
        state.ee.data(), e_old.data(), state.vol.data(), V_old.data(),
        state.mass.data(), Pe_energy_half.data(), Q_energy_half.data(),
        d_hydro_force_active, exclusion_masks.central_member,
        exclusion_masks.central_passive, exclusion_masks.pole_member, cw.begin,
        cw.end, n_cells, d_hydro_floor, d_hydro_clamp_count);
  }
  if (!(use_two_temp && per_material_hydro_enabled)) {
    sync_kernel("Hydro2D energy_update kernel failed");
  }
  {
    const braginskii::Params visc_energy_params =
        braginskii::params_from_config(cfg);
    if (visc_energy_params.enabled && !compatible_energy_mode) {
      TENRYU_ASSERT(!per_material_hydro_enabled,
                    "plasma_viscosity 2D does not support per-material decks (v1)");
      TENRYU_ASSERT(state.visc_heat_rate_per_cell.size() ==
                        static_cast<std::size_t>(n_cells),
                    "legacy viscous heat requires the heat-rate buffer");
      apply_visc_heat_kernel<<<cw.blocks(), 256>>>(
          state.ee.data(), state.ei.data(), state.mass.data(),
          state.visc_heat_rate_per_cell.data(), d_hydro_force_active,
          cw.begin, cw.end, n_cells, dt, use_two_temp ? 1 : 0);
      sync_kernel("Hydro2D viscous heat kernel failed");
      if (visc_energy_params.species != 0) {
        TENRYU_ASSERT(state.visc_heat_rate_e_per_cell.size() ==
                          static_cast<std::size_t>(n_cells),
                      "legacy electron viscous heat requires the e-heat "
                      "buffer");
        // Electron-viscous heating goes to the electron energy in 2T and
        // to the single (total) matter energy in 1T — both are state.ee,
        // i.e. exactly the use_two_temp=0 routing of this kernel.
        apply_visc_heat_kernel<<<cw.blocks(), 256>>>(
            state.ee.data(), state.ei.data(), state.mass.data(),
            state.visc_heat_rate_e_per_cell.data(), d_hydro_force_active,
            cw.begin, cw.end, n_cells, dt, /*use_two_temp=*/0);
        sync_kernel("Hydro2D electron viscous heat kernel failed");
      }
    }
  }
  launch_apply_compatible_energy_work_2d(state,
                                         cfg,
                                         P_half,
                                         Pi_half,
                                         use_two_temp,
                                         dt,
                                         d_hydro_force_active,
                                         d_hydro_floor,
                                         d_hydro_clamp_count,
                                         midpoint_v1);
  if (subzonal_hourglass_enabled(cfg) &&
      cfg.numerics.hydro.hourglass.compatible_work_enabled) {
    apply_hourglass_work_2d(state,
                            cfg,
                            hourglass_work_erg,
                            use_two_temp,
                            d_hydro_force_active,
                            d_hydro_floor,
                            d_hydro_clamp_count);
  }
  central_pseudo_core::emit_phase_ledger(
      state, cfg, "E_after_lagrange_update", "post_energy_update");
  central_pseudo_core::apply_boundary_pressure_work(state, cfg, use_two_temp);
  pole_angular_derefine::apply_boundary_pressure_work(state, cfg, use_two_temp);
  if (shell_drive_sym_diag_active) {
    emit_shell_drive_sym_diag(state,
                              cfg,
                              t_op,
                              dt,
                              shell_drive_sym_energy_before,
                              shell_drive_sym_snapshot,
                              ur_old,
                              uz_old);
  }
  central_pseudo_core::emit_phase_ledger(
      state, cfg, "E_after_boundary_work", "post_boundary_work");
  central_pseudo_core::aggregate_state(state, cfg, "post_energy", true);
  pole_angular_derefine::maintain_pole_spans(state, cfg, "post_energy");
  pole_angular_derefine::aggregate_state(state, cfg, "post_energy", true);
  conservation_audit::emit_stage(state, "post_energy_after_aggregate");

  cuda_check(cudaMemcpy(&local_E_floor, d_hydro_floor, sizeof(double), cudaMemcpyDeviceToHost),
             "Hydro2D: copy hydro floor failed");
  cuda_check(cudaMemcpy(&local_clamp_count,
                        d_hydro_clamp_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Hydro2D: copy hydro clamp_count failed");
  cuda_check(cudaMemcpy(&local_rho_clamp_count,
                        d_rho_clamp_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "Hydro2D: copy rho_clamp_count failed");
  if (apply_1t_energy_renorm) {
    const double E_after =
        allreduce_sum_scalar_2d(diagnostics::compute_energy_budget_2d(state).E_total);
    const double delta_E = E_before - E_after;
    double rel_corr = 0.0;
    if (std::abs(E_before) > 0.0) {
      rel_corr = std::abs(delta_E) / std::abs(E_before);
      if (rel_corr > 1.0e-12) {
        tenryu::core::log_info("Hydro2D energy correction rel=" +
                               format_scientific(rel_corr));
      }
    }

    if (std::abs(delta_E) > 0.0) {
      double* d_active_mass = nullptr;
      double* d_active_internal = nullptr;
      d_active_mass = static_cast<double*>(
          core::device_scratch_acquire("h2d:renorm_active_mass", sizeof(double)));
      d_active_internal = static_cast<double*>(core::device_scratch_acquire(
          "h2d:renorm_active_internal", sizeof(double)));
      cuda_check(cudaMemset(d_active_mass, 0, sizeof(double)),
                 "Hydro2D: cudaMemset active_mass failed");
      cuda_check(cudaMemset(d_active_internal, 0, sizeof(double)),
                 "Hydro2D: cudaMemset active_internal failed");

      reduce_active_mass_internal_kernel<<<cw.blocks(), 256>>>(
          state.mass.data(), state.ee.data(), d_hydro_force_active, cw.begin,
          cw.end, n_cells, d_active_mass, d_active_internal);
      sync_kernel("Hydro2D reduce_active_mass_internal kernel failed");

      double active_mass = 0.0;
      double active_internal = 0.0;
      cuda_check(cudaMemcpy(&active_mass, d_active_mass, sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Hydro2D: memcpy active_mass failed");
      cuda_check(cudaMemcpy(&active_internal, d_active_internal, sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "Hydro2D: memcpy active_internal failed");
      active_mass = allreduce_sum_scalar_2d(active_mass);
      active_internal = allreduce_sum_scalar_2d(active_internal);

      if (active_internal > 0.0) {
        const double scale = (active_internal + delta_E) / active_internal;
        scale_active_energy_kernel<<<cw.blocks(), 256>>>(
            state.ee.data(), d_hydro_force_active, cw.begin, cw.end, n_cells,
            scale);
        sync_kernel("Hydro2D scale_active_energy kernel failed");
      } else if (active_mass > 0.0) {
        const double de = delta_E / active_mass;
        shift_active_energy_kernel<<<cw.blocks(), 256>>>(
            state.ee.data(), d_hydro_force_active, cw.begin, cw.end, n_cells,
            de);
        sync_kernel("Hydro2D shift_active_energy kernel failed");
      }

      int first_active = -1;
      if (hydro_force_active_host->empty()) {
        first_active = 0;
      } else {
        const std::size_t scan_begin = static_cast<std::size_t>(cw.begin);
        const std::size_t scan_end = std::min(
            static_cast<std::size_t>(cw.end), hydro_force_active_host->size());
        for (std::size_t c = scan_begin; c < scan_end; ++c) {
          if ((*hydro_force_active_host)[c] != 0) {
            first_active = static_cast<int>(c);
            break;
          }
        }
      }
      if (part.n_ranks > 1) {
        const parallel::Reduction reduction(part.n_ranks);
        double key = first_active >= 0 ? static_cast<double>(first_active)
                                       : 1.0e18;
        key = reduction.allreduce_min(key);
        first_active = key < 1.0e18 ? static_cast<int>(key) : -1;
      }

      if (first_active >= 0) {
        const double E_corrected = allreduce_sum_scalar_2d(
            diagnostics::compute_energy_budget_2d(state).E_total);
        const double residual = E_before - E_corrected;
        const bool owns_first_active =
            first_active >= cw.begin && first_active < cw.end;
        if (std::abs(residual) > 0.0 && owns_first_active) {
          double ee_first = 0.0;
          double m_first = 0.0;
          cuda_check(cudaMemcpy(&ee_first,
                                state.ee.data() + first_active,
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "Hydro2D: copy first_active ee failed");
          cuda_check(cudaMemcpy(&m_first,
                                state.mass.data() + first_active,
                                sizeof(double),
                                cudaMemcpyDeviceToHost),
                     "Hydro2D: copy first_active mass failed");
          if (m_first > 0.0) {
            const double delta_ee = residual / m_first;
            const double rel_delta =
                std::abs(delta_ee) / std::max(std::abs(ee_first), 1.0e-30);
            if (rel_delta > 0.1) {
              tenryu::core::log_warning(
                  "Hydro2D residual correction concentrated in first active cell: rel_delta_ee=" +
                  format_scientific(rel_delta));
            }
          }
          add_first_active_residual_kernel<<<1, 1>>>(
              state.ee.data(), state.mass.data(), first_active, residual);
          sync_kernel("Hydro2D add_first_active_residual kernel failed");
        }
      }
    }
  }

  if (state_supply_active) {
    apply_state_supply_zonal_override(state, cfg);
    sync_state_supply_energy_from_temperature(state, cfg, use_two_temp, eos_views);
  }
  if (!per_material_hydro_enabled) {
    refresh_or_enforce_closure(false);
  }
  if (cfg.numerics.hydro.work_split_audit_2d_rz && use_two_temp &&
      !per_material_hydro_enabled) {
    append_work_split_audit_2d_rz(state,
                                  cfg,
                                  part,
                                  dt,
                                  hydro_half,
                                  e_old,
                                  ei_old,
                                  V_old,
                                  P_half,
                                  Pi_half,
                                  Q_half,
                                  div_u_half,
                                  d_hydro_force_active,
                                  q_heat_to_electron);
  }
  if (state_supply_active) {
    tally_state_supply_delta_E(state, cfg);
  }
  core::CellField1D cs_new;
  compute_sound_speed_for_step(cs_new);
  exchange_cell_ghosts_2d({state.rho.data(), state.vol.data(), state.Te.data(),
                           state.Ti.data(), state.Pe.data(), state.Pi.data(),
                           cs_new.data()});
  // OPEN-HYDRO-CORNER tail: final node coords/velocities must be exchanged
  // BEFORE the end-of-step AV recompute below — dVdt/div_u at ghost cells
  // read ghost-node velocities, and next step's pq consumes the stored
  // Qvisc (measured: gate Qvisc band violations at interface cells while
  // state fields held their tiers).
  exchange_mid_nodes_2d();

  core::CellField1D dVdt_new{"h2d:step:dVdt_new"};
  core::CellField1D div_u_new{"h2d:step:div_u_new"};
  core::CellField1D dl_new{"h2d:step:dl_new"};
  dVdt_new.reset(n_cells);
  div_u_new.reset(n_cells);
  dl_new.reset(n_cells);
  detail::launch_compute_cell_dVdt_2d(
      dVdt_new.data(), state.v_r.data(), state.v_z.data(), mesh_cache.Svec_r.data(),
      mesh_cache.Svec_z.data(), state, cfg, d_cell_nverts, d_hydro_force_active);
  compute_cell_div_u_kernel<<<cw.blocks(), 256>>>(
      div_u_new.data(), dVdt_new.data(), state.vol.data(), cw.begin, cw.end,
      n_cells);
  compute_cell_length_scale_kernel<<<cw.blocks(), 256>>>(
      dl_new.data(), mesh_cache.area.data(), cw.begin, cw.end, n_cells);
  sync_kernel("Hydro2D compute new dVdt/div_u/dl kernels failed");

  compute_artificial_viscosity_2d(dl_new, div_u_new, cs_new);
  qcap_apply_in_place(state.Qvisc,
                      cfg,
                      state,
                      d_hydro_force_active,
                      per_material_hydro_enabled ? &Qvisc_per_material : nullptr);
  update_av_max_diagnostic(state, cs_new, div_u_new, dl_new,
                           d_hydro_active);
  diagnostics::mesh_diag::dump_hydro_cells(state,
                                           cfg,
                                           "post_hydro",
                                           dt,
                                           state.x_r,
                                           state.x_z,
                                           state.v_r,
                                           state.v_z,
                                           &cs_new,
                                           &div_u_new,
                                           "not_recorded_in_hydro",
                                           hydro_half);
  state.cs = std::move(cs_new);
  if (total_energy_identity_check) {
    const diagnostics::EnergyTotals after =
        diagnostics::compute_energy_totals_2d(state);
    double terms[6] = {
        total_energy_identity_before.E_kin,
        total_energy_identity_before.E_int_e +
            total_energy_identity_before.E_int_i,
        after.E_kin,
        after.E_int_e + after.E_int_i,
        total_energy_identity_W_ext,
        local_E_floor,
    };
    if (reduction != nullptr) {
      reduction->allreduce_sum(terms, 6);
    }
    const double delta_K = terms[2] - terms[0];
    const double delta_U = terms[3] - terms[1];
    const double W_ext = terms[4];
    const double E_floor = terms[5];
    const double residual = delta_K + delta_U - W_ext - E_floor;
    const double scale =
        std::max({terms[0] + terms[1], std::abs(W_ext), 1.0e-300});
    if (!(std::abs(residual) <= 1.0e-11 * scale)) {
      std::ostringstream os;
      os << std::setprecision(17)
         << "Hydro2D total-energy identity failed: R=" << residual
         << " scale=" << scale << " delta_K=" << delta_K
         << " delta_U=" << delta_U << " W_ext=" << W_ext
         << " E_floor=" << E_floor;
      TENRYU_ASSERT(false, os.str());
    }
  }
  if (part.n_ranks > 1 && bufs != nullptr) {
    double* node_field_ptrs[4] = {state.x_r.data(), state.x_z.data(), state.v_r.data(),
                                  state.v_z.data()};
    parallel::exchange_node_fields(part, *bufs, node_field_ptrs, 4, n_nodes, stream,
                                   2);
  }

  if (E_floor_injected != nullptr) {
    *E_floor_injected += std::max(local_E_floor, 0.0);
  }
  if (clamp_count != nullptr) {
    *clamp_count += std::max(local_clamp_count, 0);
  }
  if (rho_clamp_count != nullptr) {
    *rho_clamp_count += std::max(local_rho_clamp_count, 0);
  }

  if (state.vol_prev_hydro.size() != state.vol.size()) {
    state.vol_prev_hydro.reset(state.vol.size());
  }
  cuda_check(cudaMemcpy(state.vol_prev_hydro.data(),
                        V_old.data(),
                        static_cast<std::size_t>(n_cells) * sizeof(double),
                        cudaMemcpyDeviceToDevice),
             "Hydro2D: copy previous hydro volume snapshot failed");
  state.dt_prev_hydro = dt;
  ale::csr_optionb_canonicalize_corner_mass_basis(state, cfg);
  ale::record_estep_trace_boundary(
      state, cfg, part, reduction, "post_lagrangian", dt, t_op + dt);
  ale_diag::emit_identity_field_diag(
      state, cfg, node_mass, "post_lagrange", t_op + dt, dt, part.rank);
  ale_diag::capture_and_emit_mover_post_hydro(
      state, cfg, r_old, z_old, node_mass, dt, t_op + dt, part.rank);
  conservation_audit::emit_stage(state, "hydro_step_end");

  if (mesh_attr_active) {
    diagnostics::mesh_attribution::end_step_success(state, cfg);
  }
  result.pressure_boundary_work_step = total_energy_identity_W_ext;
  restore_hydro_active();
  emit_origin_trace(state, "step_end", state.v_z.data());
  return result;
}

}  // namespace tenryu::hydro
