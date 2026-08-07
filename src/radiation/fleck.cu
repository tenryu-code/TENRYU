#include "radiation/fleck.cuh"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/ionmix_reader.cuh"
#include "materials/tmat_reader.hpp"
#include "radiation/fleck_bodies.cuh"
#include "radiation/groups.cuh"
#include "radiation/planck_table.cuh"
#include "core/state.hpp"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr int kDtRadOpacityConstant = fleck_bodies::kDtRadOpacityConstant;
constexpr int kDtRadOpacityFreqDepMarshak =
    fleck_bodies::kDtRadOpacityFreqDepMarshak;
constexpr int kDtRadOpacityTableNLTE = fleck_bodies::kDtRadOpacityTableNLTE;
constexpr int kReduceBlock = fleck_bodies::kReduceBlock;
using fleck_bodies::DtRadCellCandidate;
using fleck_bodies::DtRadLimitDeviceResult;
using fleck_bodies::compute_dt_rad_candidates_kernel_body;
using fleck_bodies::reduce_dt_rad_candidates_kernel_body;

struct DtRadPlanckCache {
  PlanckTable table;
  std::vector<double> bounds_eV;
  int n_groups = 0;
  int n_T = 0;
  double T_min = 0.0;
  double T_max = 0.0;
  bool valid = false;

  [[nodiscard]] bool matches(const int groups,
                             const int n_T_in,
                             const double T_min_in,
                             const double T_max_in,
                             const std::vector<double>& bounds) const {
    return valid && n_groups == groups && n_T == n_T_in && T_min == T_min_in &&
           T_max == T_max_in && bounds_eV == bounds;
  }

  PlanckTable& get(const Groups& groups,
                   const int n_T_in,
                   const double T_min_in,
                   const double T_max_in,
                   const std::vector<double>& bounds) {
    if (!matches(groups.num_groups(), n_T_in, T_min_in, T_max_in, bounds)) {
      table.build(groups, n_T_in, T_min_in, T_max_in);
      bounds_eV = bounds;
      n_groups = groups.num_groups();
      n_T = n_T_in;
      T_min = T_min_in;
      T_max = T_max_in;
      valid = true;
    }
    return table;
  }
};

struct DtRadDeviceOpacityCache {
  double* d_log_temps = nullptr;
  double* d_log_numdens = nullptr;
  double* d_kappa_PA = nullptr;
  double* d_kappa_PE = nullptr;
  double* d_kappa_R = nullptr;
  int ntemp = 0;
  int ndens = 0;
  int ngroups = 0;
  double log_T_min = 0.0;
  double log_T_max = 0.0;
  double log_ni_min = 0.0;
  double log_ni_max = 0.0;
  double T_min = 0.0;
  double T_max = 0.0;
  std::string key;

  ~DtRadDeviceOpacityCache() {
    release();
  }

  void release() {
    if (d_kappa_R != nullptr) {
      cuda_check(cudaFree(d_kappa_R), "dt_rad opacity cudaFree kappa_R failed");
      d_kappa_R = nullptr;
    }
    if (d_kappa_PE != nullptr) {
      cuda_check(cudaFree(d_kappa_PE), "dt_rad opacity cudaFree kappa_PE failed");
      d_kappa_PE = nullptr;
    }
    if (d_kappa_PA != nullptr) {
      cuda_check(cudaFree(d_kappa_PA), "dt_rad opacity cudaFree kappa_PA failed");
      d_kappa_PA = nullptr;
    }
    if (d_log_numdens != nullptr) {
      cuda_check(cudaFree(d_log_numdens), "dt_rad opacity cudaFree log_numdens failed");
      d_log_numdens = nullptr;
    }
    if (d_log_temps != nullptr) {
      cuda_check(cudaFree(d_log_temps), "dt_rad opacity cudaFree log_temps failed");
      d_log_temps = nullptr;
    }
    ntemp = 0;
    ndens = 0;
    ngroups = 0;
    log_T_min = 0.0;
    log_T_max = 0.0;
    log_ni_min = 0.0;
    log_ni_max = 0.0;
    T_min = 0.0;
    T_max = 0.0;
    key.clear();
  }

  void upload(const std::string& new_key, const materials::IonmixOpacityData& host) {
    if (key == new_key && d_kappa_PE != nullptr) {
      return;
    }
    TENRYU_ASSERT(host.ntemp > 0, "dt_rad opacity upload requires host.ntemp > 0");
    TENRYU_ASSERT(host.ndens > 0, "dt_rad opacity upload requires host.ndens > 0");
    TENRYU_ASSERT(host.ngroups > 0, "dt_rad opacity upload requires host.ngroups > 0");
    TENRYU_ASSERT(host.log_temps.size() == static_cast<std::size_t>(host.ntemp),
                  "dt_rad opacity upload log_temps size mismatch");
    TENRYU_ASSERT(host.log_numdens.size() == static_cast<std::size_t>(host.ndens),
                  "dt_rad opacity upload log_numdens size mismatch");
    const std::size_t n_table =
        static_cast<std::size_t>(host.ngroups) * static_cast<std::size_t>(host.ndens) *
        static_cast<std::size_t>(host.ntemp);
    TENRYU_ASSERT(host.kappa_PA.size() == n_table,
                  "dt_rad opacity upload kappa_PA size mismatch");
    TENRYU_ASSERT(host.kappa_PE.size() == n_table,
                  "dt_rad opacity upload kappa_PE size mismatch");
    TENRYU_ASSERT(host.kappa_R.size() == n_table,
                  "dt_rad opacity upload kappa_R size mismatch");
    TENRYU_ASSERT(!host.temps_eV.empty(), "dt_rad opacity upload requires temperatures");
    TENRYU_ASSERT(!host.numdens_cm3.empty(), "dt_rad opacity upload requires densities");

    release();
    const auto upload_vector = [](double** dst,
                                  const std::vector<double>& src,
                                  const char* malloc_message,
                                  const char* copy_message) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), sizeof(double) * src.size()),
                 malloc_message);
      cuda_check(cudaMemcpy(*dst,
                            src.data(),
                            sizeof(double) * src.size(),
                            cudaMemcpyHostToDevice),
                 copy_message);
    };

    upload_vector(&d_log_temps,
                  host.log_temps,
                  "dt_rad opacity cudaMalloc log_temps failed",
                  "dt_rad opacity copy log_temps failed");
    upload_vector(&d_log_numdens,
                  host.log_numdens,
                  "dt_rad opacity cudaMalloc log_numdens failed",
                  "dt_rad opacity copy log_numdens failed");
    upload_vector(&d_kappa_PA,
                  host.kappa_PA,
                  "dt_rad opacity cudaMalloc kappa_PA failed",
                  "dt_rad opacity copy kappa_PA failed");
    upload_vector(&d_kappa_PE,
                  host.kappa_PE,
                  "dt_rad opacity cudaMalloc kappa_PE failed",
                  "dt_rad opacity copy kappa_PE failed");
    upload_vector(&d_kappa_R,
                  host.kappa_R,
                  "dt_rad opacity cudaMalloc kappa_R failed",
                  "dt_rad opacity copy kappa_R failed");

    ntemp = host.ntemp;
    ndens = host.ndens;
    ngroups = host.ngroups;
    log_T_min = host.log_temps.front();
    log_T_max = host.log_temps.back();
    log_ni_min = host.log_numdens.front();
    log_ni_max = host.log_numdens.back();
    T_min = host.temps_eV.front();
    T_max = host.temps_eV.back();
    key = new_key;
  }

  [[nodiscard]] materials::IonmixOpacityDeviceView view() const {
    materials::IonmixOpacityDeviceView out{};
    out.log_temps = d_log_temps;
    out.log_numdens = d_log_numdens;
    out.kappa_PA = d_kappa_PA;
    out.kappa_PE = d_kappa_PE;
    out.kappa_R = d_kappa_R;
    out.ntemp = ntemp;
    out.ndens = ndens;
    out.ngroups = ngroups;
    out.log_T_min = log_T_min;
    out.log_T_max = log_T_max;
    out.log_ni_min = log_ni_min;
    out.log_ni_max = log_ni_max;
    out.T_min = T_min;
    out.T_max = T_max;
    return out;
  }
};

struct DtRadWorkspace {
  DtRadCellCandidate* d_candidates = nullptr;
  DtRadLimitDeviceResult* d_result = nullptr;
  std::uint8_t* d_cell_is_void = nullptr;
  int candidate_capacity = 0;
  int void_capacity = 0;

  ~DtRadWorkspace() {
    release();
  }

  void release() {
    if (d_cell_is_void != nullptr) {
      cuda_check(cudaFree(d_cell_is_void), "dt_rad workspace cudaFree void failed");
      d_cell_is_void = nullptr;
    }
    if (d_result != nullptr) {
      cuda_check(cudaFree(d_result), "dt_rad workspace cudaFree result failed");
      d_result = nullptr;
    }
    if (d_candidates != nullptr) {
      cuda_check(cudaFree(d_candidates), "dt_rad workspace cudaFree candidates failed");
      d_candidates = nullptr;
    }
    candidate_capacity = 0;
    void_capacity = 0;
  }

  void ensure(const int n_cells, const bool needs_void) {
    if (candidate_capacity < n_cells) {
      if (d_candidates != nullptr) {
        cuda_check(cudaFree(d_candidates), "dt_rad workspace cudaFree old candidates failed");
        d_candidates = nullptr;
      }
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_candidates),
                            sizeof(DtRadCellCandidate) *
                                static_cast<std::size_t>(n_cells)),
                 "dt_rad workspace cudaMalloc candidates failed");
      candidate_capacity = n_cells;
    }
    if (d_result == nullptr) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_result),
                            sizeof(DtRadLimitDeviceResult)),
                 "dt_rad workspace cudaMalloc result failed");
    }
    if (needs_void && void_capacity < n_cells) {
      if (d_cell_is_void != nullptr) {
        cuda_check(cudaFree(d_cell_is_void), "dt_rad workspace cudaFree old void failed");
        d_cell_is_void = nullptr;
      }
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_is_void),
                            sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)),
                 "dt_rad workspace cudaMalloc void failed");
      void_capacity = n_cells;
    }
  }
};

__global__ void compute_dt_rad_candidates_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const double* __restrict__ state_cv_e,
    const int state_cv_e_size,
    const std::uint8_t* __restrict__ cell_is_void,
    const materials::IonmixOpacityDeviceView table,
    const PlanckTableDeviceView planck,
    DtRadCellCandidate* __restrict__ candidates,
    const int n_cells,
    const int n_groups,
    const int opacity_model,
    const double kappa_a,
    const double E_min_eV,
    const double E_max_eV,
    const double sigma_cap,
    const double T_floor,
    const double rho_floor,
    const double alpha,
    const double f_min,
    const double cv_e_override,
    const double gm1,
    const double A,
    const int linearized_planck,
    const double infinity) {
  const int c = blockIdx.x;
  const int tid = threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  extern __shared__ double smem[];
  compute_dt_rad_candidates_kernel_body(
      c, tid, smem, rho, Te, zbar, state_cv_e, state_cv_e_size,
      cell_is_void, table, planck, candidates, n_cells, n_groups,
      opacity_model, kappa_a, E_min_eV, E_max_eV, sigma_cap, T_floor,
      rho_floor, alpha, f_min, cv_e_override, gm1, A, linearized_planck,
      infinity);
}

__global__ void reduce_dt_rad_candidates_kernel(
    const DtRadCellCandidate* __restrict__ candidates,
    const int n_cells,
    DtRadLimitDeviceResult* __restrict__ result,
    const double infinity) {
  const int tid = threadIdx.x;
  reduce_dt_rad_candidates_kernel_body(tid, candidates, n_cells, result,
                                       infinity);
}

__global__ void compute_fleck_kernel(const double* __restrict__ rho,
                                     const double* __restrict__ Te,
                                     const double* __restrict__ zbar,
                                     const std::uint8_t* __restrict__ cell_is_void,
                                     const double* __restrict__ sigma_a,
                                     const PlanckTableDeviceView planck,
                                     double* __restrict__ f_fleck,
                                     double* __restrict__ sigma_a_eff,
                                     double* __restrict__ sigma_s_eff,
                                     const int n_cells,
                                     const int n_groups,
                                     const double dt,
                                     const double alpha,
                                     const double f_max,
                                     const double cv_e_override,
                                     const double gamma,
                                     const double A,
                                     const int linearized_planck,
                                     const double* __restrict__ state_cv_e) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int base = c * n_groups;
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    for (int g = 0; g < n_groups; ++g) {
      f_fleck[base + g] = 1.0;
    }
    for (int g = 0; g < n_groups; ++g) {
      sigma_a_eff[base + g] = 0.0;
      sigma_s_eff[base + g] = 0.0;
    }
    return;
  }

  const double rho_c = rho[c];
  const double Te_c = fmax(Te[c], 1.0e-12);

  double Cv_e = -1.0;
  if (cv_e_override > 0.0) {
    Cv_e = cv_e_override;
  } else if (state_cv_e != nullptr && state_cv_e[c] > 0.0) {
    // Table EOS path: state_cv_e is mass-specific [erg/(g*eV)].
    Cv_e = fmax(rho_c, 0.0) * state_cv_e[c];
  } else {
    const double z = fmax(zbar[c], 0.0);
    const double gm1 = fmax(gamma - 1.0, 1.0e-12);
    const double cv_mass_e = z * tenryu::core::constants::eV_to_erg /
                             (fmax(A, 1.0e-12) * tenryu::core::constants::proton_mass * gm1);
    Cv_e = fmax(rho_c, 0.0) * cv_mass_e;
  }
  Cv_e = fmax(Cv_e, 1.0e-30);

  double sigma_aP = 0.0;
  if (n_groups == 1) {
    sigma_aP = fmax(sigma_a[base], 0.0);
  } else {
    for (int g = 0; g < n_groups; ++g) {
      const double sigma_g = fmax(sigma_a[base + g], 0.0);
      const double b_g = fmax(planck.interpolate_b(g, Te_c), 0.0);
      sigma_aP += sigma_g * b_g;
    }
  }

  double beta = 0.0;
  if (linearized_planck && cv_e_override > 0.0) {
    beta = 1.0;
  } else {
    beta = 4.0 * tenryu::core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
  }
  if (cv_e_override <= 0.0 &&
      (state_cv_e == nullptr || !(state_cv_e[c] > 0.0))) {
    // Safety cap for ideal-gas fallback Cv path:
    // keeping beta <= 1 guarantees Fleck f in [0, 1] under rounding/low-Cv extremes
    // and avoids unphysical super-emission behavior (f > 1).
    beta = fmin(beta, 1.0);
  }
  const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
  const double z =
      alpha_safe * beta * tenryu::core::constants::c_light * sigma_aP * dt;
  const double f_std = 1.0 / (1.0 + z);

  // McClarren-Urbatsch modified Fleck: stronger effective scattering in stiff cells.
  constexpr double kFleckBlendZ0 = 2.0;
  constexpr double kFleckBlendZ1 = 10.0;
  const double t_blend = (z <= kFleckBlendZ0)
                             ? 0.0
                             : (z >= kFleckBlendZ1)
                                   ? 1.0
                                   : (z - kFleckBlendZ0) /
                                         (kFleckBlendZ1 - kFleckBlendZ0);
  const double w = t_blend * t_blend * (3.0 - 2.0 * t_blend);
  const double f_mu = exp(-z);
  double f = (1.0 - w) * f_std + w * f_mu;
  f = fmin(f, f_max);
  f = fmax(f, 0.0);
  for (int g = 0; g < n_groups; ++g) {
    f_fleck[base + g] = f;
  }

  for (int g = 0; g < n_groups; ++g) {
    const double sigma = sigma_a[base + g];
    sigma_a_eff[base + g] = f * sigma;
    sigma_s_eff[base + g] = (1.0 - f) * sigma;
  }
}

}  // namespace

void compute_fleck_and_sigma_eff_cuda(const FleckView& view,
                                      const core::Config& cfg) {
  TENRYU_ASSERT(view.rho != nullptr, "compute_fleck requires rho");
  TENRYU_ASSERT(view.Te != nullptr, "compute_fleck requires Te");
  TENRYU_ASSERT(view.zbar != nullptr, "compute_fleck requires zbar");
  TENRYU_ASSERT(view.sigma_a != nullptr, "compute_fleck requires sigma_a");
  TENRYU_ASSERT(view.f_fleck != nullptr, "compute_fleck requires f_fleck");
  TENRYU_ASSERT(view.sigma_a_eff != nullptr,
                "compute_fleck requires sigma_a_eff");
  TENRYU_ASSERT(view.sigma_s_eff != nullptr,
                "compute_fleck requires sigma_s_eff");
  TENRYU_ASSERT(view.cell_is_void != nullptr,
                "compute_fleck requires cell_is_void");
  TENRYU_ASSERT(view.n_cells >= 0, "compute_fleck requires n_cells >= 0");
  TENRYU_ASSERT(view.n_groups >= 1, "compute_fleck requires n_groups >= 1");
  if (view.n_groups > 1) {
    TENRYU_ASSERT(view.planck.b_g != nullptr,
                  "compute_fleck requires planck table for multigroup");
    TENRYU_ASSERT(view.planck.T_grid != nullptr,
                  "compute_fleck requires planck temperature grid for multigroup");
  }

  const int mat_idx = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(mat_idx >= 0, "compute_fleck requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(mat_idx)];
  const double cv_e_override = mat.cv_e_override;

  constexpr int kBlock = 256;
  const int grid = (view.n_cells + kBlock - 1) / kBlock;
  if (grid > 0) {
    std::uint8_t* d_cell_is_void = nullptr;
    const std::size_t bytes_void =
        sizeof(std::uint8_t) * static_cast<std::size_t>(view.n_cells);
    d_cell_is_void = static_cast<std::uint8_t*>(
        core::device_scratch_acquire("fleck:cell_is_void", bytes_void));
    cuda_check(cudaMemcpy(d_cell_is_void,
                          view.cell_is_void,
                          bytes_void,
                          cudaMemcpyHostToDevice),
               "compute_fleck copy cell_is_void failed");
    compute_fleck_kernel<<<grid, kBlock>>>(view.rho,
                                           view.Te,
                                           view.zbar,
                                           d_cell_is_void,
                                           view.sigma_a,
                                           view.planck,
                                           view.f_fleck,
                                           view.sigma_a_eff,
                                           view.sigma_s_eff,
                                           view.n_cells,
                                           view.n_groups,
                                           view.dt,
                                           cfg.radiation.imc.alpha,
                                           cfg.radiation.imc.f_max,
                                           cv_e_override,
                                           mat.ideal_gas_gamma,
                                           mat.A,
                                           cfg.radiation.imc.linearized_planck ? 1 : 0,
                                           view.state_cv_e);
    cuda_check(cudaGetLastError(), "compute_fleck kernel launch failed");
    cuda_check(cudaDeviceSynchronize(), "compute_fleck kernel execution failed");
  }
}

double compute_dt_rad_limit(const core::State& state,
                            const core::Config& cfg) {
  return compute_dt_rad_limit(state, cfg, nullptr);
}

double compute_dt_rad_limit(const core::State& state,
                            const core::Config& cfg,
                            DtRadDiagnostics* diag) {
  if (diag != nullptr) {
    *diag = DtRadDiagnostics{};
  }
  if (!cfg.radiation.enabled || state.rho.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  if (cfg.materials.materials.empty()) {
    return std::numeric_limits<double>::infinity();
  }

  const int mat_idx = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(mat_idx >= 0,
                "compute_dt_rad_limit requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(mat_idx)];
  const bool use_freq_dep_marshak = (mat.opacity_model == "freq_dep_marshak");
  const bool use_nlte_table =
      (mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat");
  const double kappa_a = std::max(0.0, mat.kappa_a_constant);
  if (!use_freq_dep_marshak && !use_nlte_table && kappa_a <= 0.0) {
    return std::numeric_limits<double>::infinity();
  }

  TENRYU_ASSERT(state.rho.size() <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "compute_dt_rad_limit n_cells exceeds int range");
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(state.Te.size() >= state.rho.size(),
                "compute_dt_rad_limit Te size mismatch");
  TENRYU_ASSERT(state.cv_e.size() <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "compute_dt_rad_limit cv_e size exceeds int range");
  if (mat.cv_e_override <= 0.0) {
    TENRYU_ASSERT(state.zbar.size() >= state.rho.size(),
                  "compute_dt_rad_limit zbar size mismatch");
  }
  const int n_groups = std::max(cfg.radiation.groups, 1);
  std::vector<double> bounds_eV = cfg.radiation.group_bounds_eV;
  double E_min_eV = 0.0;
  double E_max_eV = 0.0;
  if (use_freq_dep_marshak || use_nlte_table) {
    if (bounds_eV.empty() || static_cast<int>(bounds_eV.size()) != n_groups + 1) {
      const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
      const double Tmin = std::max(range[0], 1.0e-12);
      const double Tmax = std::max(range[1], Tmin * 1.000001);
      bounds_eV = Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
    }
  }
  if (use_freq_dep_marshak) {
    TENRYU_ASSERT(static_cast<int>(bounds_eV.size()) == n_groups + 1,
                  "compute_dt_rad_limit requires valid group bounds for freq_dep_marshak");
    E_min_eV = std::max(bounds_eV.front(), 1.0e-12);
    E_max_eV = std::max(bounds_eV.back(), E_min_eV * 1.000001);
  }

  PlanckTableDeviceView planck_view{};
  materials::IonmixOpacityDeviceView table_view{};
  if (use_nlte_table) {
    TENRYU_ASSERT(static_cast<int>(bounds_eV.size()) == n_groups + 1,
                  "compute_dt_rad_limit requires valid group bounds for table_nlte/tmat");
    const Groups groups(bounds_eV);
    const auto& planck_cfg = cfg.radiation.planck_fraction;
    const int planck_n_T = std::max(planck_cfg.compute_N_T, 2);
    const std::vector<double> planck_range =
        resolve_compute_T_range_eV(cfg, false);
    const double planck_T_min = planck_range[0];
    const double planck_T_max = planck_range[1];
    static DtRadPlanckCache planck_cache;
    PlanckTable& planck =
        planck_cache.get(groups, planck_n_T, planck_T_min, planck_T_max, bounds_eV);
    planck_view = planck.device_view();
    TENRYU_ASSERT(planck_view.n_groups == n_groups,
                  "compute_dt_rad_limit Planck/group mismatch");
    TENRYU_ASSERT(planck_view.n_T == planck_n_T,
                  "compute_dt_rad_limit Planck temperature grid mismatch");
    TENRYU_ASSERT(n_groups == 1 ||
                      (planck_view.T_grid != nullptr && planck_view.b_g != nullptr),
                  "compute_dt_rad_limit requires Planck table for multigroup NLTE");

    const std::string table_key = mat.opacity_model + "\n" + mat.opacity_file;
    static std::string cached_table_key;
    static std::unique_ptr<materials::IonmixOpacityData> cached_table;
    if (!cached_table || cached_table_key != table_key) {
      if (mat.opacity_model == "tmat") {
        const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
        TENRYU_ASSERT(tmat.opacity.has_value(),
                      "compute_dt_rad_limit tmat opacity requires /opacity payload");
        cached_table = std::make_unique<materials::IonmixOpacityData>(
            materials::tmat_to_ionmix_opacity(*tmat.opacity, true));
      } else {
        cached_table = std::make_unique<materials::IonmixOpacityData>(
            materials::load_ionmix_opacity(mat.opacity_file));
      }
      cached_table_key = table_key;
    }
    TENRYU_ASSERT(cached_table != nullptr,
                  "compute_dt_rad_limit failed to initialize NLTE opacity table");
    TENRYU_ASSERT(cached_table->ngroups == n_groups,
                  "compute_dt_rad_limit NLTE table group count mismatch");
    static DtRadDeviceOpacityCache device_table_cache;
    device_table_cache.upload(cached_table_key, *cached_table);
    table_view = device_table_cache.view();
    TENRYU_ASSERT(state.cell_is_void.size() >= static_cast<std::size_t>(n_cells),
                  "compute_dt_rad_limit cell_is_void size mismatch");
  }

  const double alpha = cfg.radiation.imc.alpha;
  const double f_min = cfg.numerics.dt.f_min_fleck;
  if (!(alpha > 0.0) || !(f_min > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }

  const double gm1 = std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12);
  const double A = std::max(mat.A, 1.0e-12);
  const double rho_floor =
      (std::isfinite(cfg.numerics.floors.rho) && cfg.numerics.floors.rho > 0.0)
          ? cfg.numerics.floors.rho
          : 1.0e-30;
  const bool linearized_planck =
      cfg.radiation.imc.linearized_planck && mat.cv_e_override > 0.0;

  int opacity_model = kDtRadOpacityConstant;
  if (use_freq_dep_marshak) {
    opacity_model = kDtRadOpacityFreqDepMarshak;
  } else if (use_nlte_table) {
    opacity_model = kDtRadOpacityTableNLTE;
  }

  static DtRadWorkspace workspace;
  workspace.ensure(n_cells, use_nlte_table);
  std::uint8_t* d_cell_is_void = nullptr;
  if (use_nlte_table) {
    const std::size_t bytes_void =
        sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
    cuda_check(cudaMemcpy(workspace.d_cell_is_void,
                          state.cell_is_void.data(),
                          bytes_void,
                          cudaMemcpyHostToDevice),
               "compute_dt_rad_limit copy cell_is_void failed");
    d_cell_is_void = workspace.d_cell_is_void;
  }

  const double infinity = std::numeric_limits<double>::infinity();
  const int cv_e_size = static_cast<int>(state.cv_e.size());
  const std::size_t shared_doubles = 2U * static_cast<std::size_t>(n_groups);
  const std::size_t shared_bytes = shared_doubles * sizeof(double);
  TENRYU_ASSERT(shared_bytes <= 32U * 1024U,
                "dt_rad NLTE shared memory exceeds 32 KiB safety cap; reduce n_groups");
  compute_dt_rad_candidates_kernel<<<n_cells, 32, shared_bytes>>>(
      state.rho.data(),
      state.Te.data(),
      state.zbar.data(),
      state.cv_e.data(),
      cv_e_size,
      d_cell_is_void,
      table_view,
      planck_view,
      workspace.d_candidates,
      n_cells,
      n_groups,
      opacity_model,
      kappa_a,
      E_min_eV,
      E_max_eV,
      cfg.numerics.safety.opacity_cap,
      cfg.numerics.floors.Te,
      rho_floor,
      alpha,
      f_min,
      mat.cv_e_override,
      gm1,
      A,
      linearized_planck ? 1 : 0,
      infinity);
  cuda_check(cudaGetLastError(), "compute_dt_rad_limit candidate kernel launch failed");
  reduce_dt_rad_candidates_kernel<<<1, kReduceBlock>>>(workspace.d_candidates,
                                                       n_cells,
                                                       workspace.d_result,
                                                       infinity);
  cuda_check(cudaGetLastError(), "compute_dt_rad_limit reduce kernel launch failed");

  DtRadLimitDeviceResult result{};
  cuda_check(cudaMemcpy(&result,
                        workspace.d_result,
                        sizeof(result),
                        cudaMemcpyDeviceToHost),
             "compute_dt_rad_limit copy result failed");

  double dt_rad = result.dt_rad;
  if (diag != nullptr && result.limiting_cell >= 0) {
    diag->limiting_cell = result.limiting_cell;
    diag->limiting_sigma_P = result.sigma_P;
    diag->limiting_beta = result.beta;
    diag->limiting_Cv_e = result.Cv_e;
    diag->limiting_Te = result.Te;
    diag->limiting_rho = result.rho;
  }

  if (use_nlte_table && result.max_sigma_P > 0.0 &&
      (!std::isfinite(dt_rad) || dt_rad <= 0.0)) {
    const double safety = (1.0 - f_min) / (f_min * alpha);
    if (safety > 0.0) {
      const double dt_sigma_bound =
          safety / (tenryu::core::constants::c_light * result.max_sigma_P);
      if (std::isfinite(dt_sigma_bound) && dt_sigma_bound > 0.0) {
        dt_rad = std::min(dt_rad, dt_sigma_bound);
      }
    }
  }

  if (diag != nullptr) {
    diag->dt_rad = dt_rad;
  }
  return dt_rad;
}

}  // namespace tenryu::radiation
