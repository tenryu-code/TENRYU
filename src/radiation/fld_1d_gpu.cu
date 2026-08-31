#include "radiation/fld_1d_gpu.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

#include "mesh/geometry_1d.cuh"

#include <cuda_runtime.h>
#include <cusparse.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "materials/eos_device_table.hpp"
#include "materials/eos_table.hpp"
#include "materials/ionmix_reader.cuh"
#include "materials/ionmix_reader.hpp"
#include "materials/opacity.cuh"
#include "materials/tmat_reader.hpp"
#include "radiation/fleck.cuh"
#include "radiation/fld_1d_bodies.cuh"
#include "radiation/group_structure.hpp"
#include "radiation/groups.cuh"
#include "radiation/nlte_coeffs.cuh"
#include "radiation/phi1_poles.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void cusparse_check(const cusparseStatus_t status, const char* message) {
  TENRYU_ASSERT(status == CUSPARSE_STATUS_SUCCESS, message);
}

struct FldCusparseCache {
  cusparseHandle_t handle = nullptr;

  ~FldCusparseCache() {
    if (handle != nullptr) {
      static_cast<void>(cusparseDestroy(handle));
      handle = nullptr;
    }
  }
};

FldCusparseCache& fld_cusparse_cache() {
  static FldCusparseCache cache;
  return cache;
}

__host__ __device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double x) {
  return isfinite(x) ? fmax(x, 0.0) : 0.0;
}

__host__ __device__ inline double positive_or_floor(const double x,
                                                    const double floor_value) {
  const double xf = finite_or_zero(x);
  return (xf > floor_value) ? xf : floor_value;
}

__host__ __device__ inline double safe_pow4(const double T) {
  const double Tc = fmin(fmax(finite_or_zero(T), 0.0), 1.0e6);
  const double T2 = Tc * Tc;
  return T2 * T2;
}

__device__ inline void atomic_max_nonnegative_double(double* address,
                                                     const double value) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val = __double_as_longlong(value);
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      break;
    }
    assumed = old;
  }
}

__device__ double flux_limiter_lambda(const double R, const int limiter) {
  const double r = fmax(finite_or_zero(R), 0.0);
  if (limiter == 2) {
    return 1.0 / 3.0;
  }
  if (limiter == 1) {
    return 1.0 / sqrt(9.0 + r * r);
  }
  if (r < 1.0e-3) {
    // Levermore-Pomraning series (coth r = 1/r + r/3 - r^3/45 + 2r^5/945):
    // the raw form below loses up to ~1e-4 relative accuracy to cancellation
    // for r near 1e-6; the omitted series term is O(r^6/4725) < 3e-22 here.
    const double r2 = r * r;
    return 1.0 / 3.0 - r2 / 45.0 + 2.0 * r2 * r2 / 945.0;
  }
  if (r > 50.0) {
    // coth(r) - 1 < 4e-44 for r > 50: (1 - 1/r)/r is exact to double
    // precision and continuous at the crossover (the bare 1/r tail had a
    // 2% jump at r = 50).
    return (1.0 - 1.0 / r) / r;
  }
  return (1.0 / tanh(r) - 1.0 / r) / r;
}

int limiter_id(const std::string& limiter) {
  if (limiter == "larsen") {
    return 1;
  }
  if (limiter == "none") {
    return 2;
  }
  return 0;
}

struct DeviceOpacityTable {
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

  DeviceOpacityTable() = default;
  DeviceOpacityTable(const DeviceOpacityTable&) = delete;
  DeviceOpacityTable& operator=(const DeviceOpacityTable&) = delete;

  ~DeviceOpacityTable() {
    release();
  }

  void upload(const materials::IonmixOpacityData& host) {
    release();
    TENRYU_ASSERT(host.ntemp > 0 && host.ndens > 0 && host.ngroups > 0,
                  "FLD DeviceOpacityTable requires non-empty opacity table");
    const std::size_t n_table =
        static_cast<std::size_t>(host.ngroups) *
        static_cast<std::size_t>(host.ndens) *
        static_cast<std::size_t>(host.ntemp);
    TENRYU_ASSERT(host.kappa_PA.size() == n_table &&
                      host.kappa_PE.size() == n_table &&
                      host.kappa_R.size() == n_table,
                  "FLD DeviceOpacityTable opacity table size mismatch");

    const auto upload_vec = [](double** dst,
                               const std::vector<double>& src,
                               const char* label) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(dst),
                            sizeof(double) * src.size()),
                 label);
      cuda_check(cudaMemcpy(*dst,
                            src.data(),
                            sizeof(double) * src.size(),
                            cudaMemcpyHostToDevice),
                 label);
    };
    upload_vec(&d_log_temps, host.log_temps, "FLD upload log_temps failed");
    upload_vec(&d_log_numdens, host.log_numdens, "FLD upload log_numdens failed");
    upload_vec(&d_kappa_PA, host.kappa_PA, "FLD upload kappa_PA failed");
    upload_vec(&d_kappa_PE, host.kappa_PE, "FLD upload kappa_PE failed");
    upload_vec(&d_kappa_R, host.kappa_R, "FLD upload kappa_R failed");
    ntemp = host.ntemp;
    ndens = host.ndens;
    ngroups = host.ngroups;
    log_T_min = host.log_temps.front();
    log_T_max = host.log_temps.back();
    log_ni_min = host.log_numdens.front();
    log_ni_max = host.log_numdens.back();
    T_min = host.temps_eV.front();
    T_max = host.temps_eV.back();
  }

  void release() {
    if (d_kappa_R != nullptr) {
      cuda_check(cudaFree(d_kappa_R), "FLD free kappa_R failed");
      d_kappa_R = nullptr;
    }
    if (d_kappa_PE != nullptr) {
      cuda_check(cudaFree(d_kappa_PE), "FLD free kappa_PE failed");
      d_kappa_PE = nullptr;
    }
    if (d_kappa_PA != nullptr) {
      cuda_check(cudaFree(d_kappa_PA), "FLD free kappa_PA failed");
      d_kappa_PA = nullptr;
    }
    if (d_log_numdens != nullptr) {
      cuda_check(cudaFree(d_log_numdens), "FLD free log_numdens failed");
      d_log_numdens = nullptr;
    }
    if (d_log_temps != nullptr) {
      cuda_check(cudaFree(d_log_temps), "FLD free log_temps failed");
      d_log_temps = nullptr;
    }
    ntemp = 0;
    ndens = 0;
    ngroups = 0;
  }

  materials::IonmixOpacityDeviceView view() const {
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

struct NlteOpacityCache {
  std::unique_ptr<materials::IonmixOpacityData> host;
  DeviceOpacityTable device;
  std::string file;
  int groups = 0;
};

NlteOpacityCache& nlte_cache() {
  static NlteOpacityCache cache;
  return cache;
}

struct MultiMatOpacityEntry {
  std::unique_ptr<materials::IonmixOpacityData> host;
  DeviceOpacityTable device;
  std::string file;
  int groups = 0;
  bool is_lte = false;
};

struct MatOpacityDesc {
  // kind: 0 = constant/void, 1 = LTE table, 2 = NLTE table.
  int kind = 0;
  double kappa_planck = 0.0;
  double kappa_rosseland = 0.0;
  double inv_A_mp = 0.0;
  materials::IonmixOpacityDeviceView view;
};

struct MultiMatOpacityCache {
  std::vector<MultiMatOpacityEntry> entries;
  core::DeviceArray<MatOpacityDesc> device_descs;
  std::string signature;
};

MultiMatOpacityCache& multimat_opacity_cache() {
  static MultiMatOpacityCache cache;
  return cache;
}

struct FldGroupBoundsCache {
  std::vector<double> last_bounds;
  bool valid = false;
};

FldGroupBoundsCache& fld_group_bounds_cache() {
  static FldGroupBoundsCache cache;
  return cache;
}

class FldElectronEOSTableCache {
 public:
  materials::DeviceEOSTableView view_for(const materials::EOSTableTriplet& tables) {
    if (tables_ != &tables) {
      electron_.upload(tables.electron);
      tables_ = &tables;
    }
    return electron_.view();
  }

 private:
  const materials::EOSTableTriplet* tables_ = nullptr;
  materials::DeviceEOSTable electron_;
};

FldElectronEOSTableCache& fld_electron_eos_table_cache() {
  static FldElectronEOSTableCache cache;
  return cache;
}

materials::DeviceEOSTableView fld_electron_eos_device_view(
    const materials::EOSTableTriplet* tables) {
  if (tables == nullptr) {
    return materials::DeviceEOSTableView{};
  }
  return fld_electron_eos_table_cache().view_for(*tables);
}

__device__ inline bool has_electron_eos_table(
    const materials::DeviceEOSTableView& electron_eos) {
  return electron_eos.n_rho > 0 && electron_eos.n_T > 0 &&
         electron_eos.P_table != nullptr &&
         electron_eos.e_table != nullptr &&
         electron_eos.cv_table != nullptr;
}

__device__ inline double eos_log_temperature(const double T,
                                             const double temperature_floor_eV) {
  return log(fmax(finite_or_zero(T), fmax(temperature_floor_eV, 1.0e-30)));
}

void ensure_nlte_table_uploaded(const core::Config& cfg,
                                const core::Config::MaterialsConfig::MatDef& mat,
                                const int n_groups) {
  auto& cache = nlte_cache();
  if (cache.host != nullptr && cache.file == mat.opacity_file &&
      cache.groups == n_groups) {
    return;
  }
  if (mat.opacity_model == "tmat") {
    const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
    TENRYU_ASSERT(tmat.opacity.has_value(),
                  "FLD tmat opacity.model requires /opacity payload");
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::tmat_to_ionmix_opacity(*tmat.opacity));
  } else {
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::load_ionmix_opacity(mat.opacity_file));
  }
  if (cfg.radiation.group_repack_hard_xray) {
    std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
    if (target_bounds.empty()) {
      const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
      target_bounds =
          repack_group_bounds_for_hard_xray(n_groups, range);
    }
    *cache.host = resample_opacity_groups_to_bounds(*cache.host, target_bounds);
  }
  TENRYU_ASSERT(cache.host->ngroups == n_groups,
                "FLD NLTE opacity table group count must match Radiation.groups");
  cache.device.upload(*cache.host);
  cache.file = mat.opacity_file;
  cache.groups = n_groups;
}

void ensure_multimat_opacity_uploaded(const core::Config& cfg,
                                      const int n_groups) {
  std::string signature;
  for (const auto& mat : cfg.materials.materials) {
    signature += mat.opacity_model + ":" + mat.opacity_file + ";";
  }

  auto& cache = multimat_opacity_cache();
  bool reusable = cache.signature == signature &&
                  cache.entries.size() == cfg.materials.materials.size();
  if (reusable) {
    for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
      if (!cfg.materials.materials[m].is_void &&
          cfg.materials.materials[m].opacity_model == "tmat" &&
          cache.entries[m].groups != n_groups) {
        reusable = false;
        break;
      }
    }
  }
  if (reusable) {
    return;
  }

  cache.entries =
      std::vector<MultiMatOpacityEntry>(cfg.materials.materials.size());
  std::vector<MatOpacityDesc> descs(cfg.materials.materials.size());
  for (std::size_t m = 0; m < cfg.materials.materials.size(); ++m) {
    const auto& mat = cfg.materials.materials[m];
    auto& entry = cache.entries[m];
    auto& desc = descs[m];
    entry.file = mat.opacity_file;
    desc.inv_A_mp =
        1.0 / (std::max(mat.A, 1.0e-12) * core::constants::proton_mass);

    if (mat.is_void) {
      desc.kind = 0;
      desc.kappa_planck = 0.0;
      desc.kappa_rosseland = 0.0;
      continue;
    }
    if (mat.opacity_model == "tmat") {
      const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
      TENRYU_ASSERT(tmat.opacity.has_value(),
                    "FLD tmat opacity.model requires /opacity payload");
      entry.host = std::make_unique<materials::IonmixOpacityData>(
          materials::tmat_to_ionmix_opacity(*tmat.opacity));
      if (cfg.radiation.group_repack_hard_xray) {
        std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
        if (target_bounds.empty()) {
          const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
          target_bounds = repack_group_bounds_for_hard_xray(n_groups, range);
        }
        *entry.host =
            resample_opacity_groups_to_bounds(*entry.host, target_bounds);
      }
      TENRYU_ASSERT(
          entry.host->ngroups == n_groups,
          "FLD multi-material opacity table group count must match Radiation.groups");
      entry.device.upload(*entry.host);
      entry.groups = n_groups;
      entry.is_lte = tmat.opacity->is_lte;
      desc.kind = entry.is_lte ? 1 : 2;
      desc.view = entry.device.view();
      continue;
    }

    desc.kind = 0;
    desc.kappa_planck = (mat.kappa_planck_override >= 0.0)
                            ? mat.kappa_planck_override
                            : std::max(0.0, mat.kappa_a_constant);
    desc.kappa_rosseland = std::max(0.0, mat.kappa_a_constant);
  }

  cache.device_descs.reset(descs.size());
  cache.device_descs.copy_from_host(descs);
  cache.signature = std::move(signature);
}

__global__ void eval_opacity_multimat_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const int* __restrict__ cell_material_index,
    const MatOpacityDesc* __restrict__ descs,
    int n_materials,
    double kappa_floor,
    double kappa_cap,
    double temperature_floor_eV,
    double* __restrict__ sigma_a,
    double* __restrict__ sigma_pe,
    double* __restrict__ sigma_R,
    int n_cells,
    int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int m = min(max(cell_material_index[c], 0), n_materials - 1);
  const MatOpacityDesc desc = descs[m];
  if (desc.kind == 2) {
    return;  // NLTE material: the per-material NLTE launch fills this cell
  }
  const double rho_c = rho[c];
  const double rho_safe = (rho_c >= 0.0) ? rho_c : 0.0;
  const double sigma_min = rho_safe * fmax(kappa_floor, 0.0);
  const double sigma_max = rho_safe * fmax(kappa_cap, 0.0);
  const int base = c * n_groups;

  if (desc.kind == 0) {
    const double sigma_a_c = fmin(
        fmax(fmax(rho_safe * desc.kappa_planck, 0.0), sigma_min), sigma_max);
    const double sigma_R_c = fmin(
        fmax(fmax(rho_safe * desc.kappa_rosseland, 0.0), sigma_min), sigma_max);
    for (int g = 0; g < n_groups; ++g) {
      sigma_a[base + g] = sigma_a_c;
      sigma_pe[base + g] = sigma_a_c;
      sigma_R[base + g] = sigma_R_c;
    }
    return;
  }

  const double log_ni = log(fmax(rho_safe * desc.inv_A_mp, 1.0e-300));
  const double log_T = log(fmax(Te[c], temperature_floor_eV));
  for (int g = 0; g < n_groups; ++g) {
    const double kappa_pa =
        desc.view.interpolate(desc.view.kappa_PA, g, log_ni, log_T);
    const double kappa_pe =
        desc.view.interpolate(desc.view.kappa_PE, g, log_ni, log_T);
    const double kappa_r =
        desc.view.interpolate(desc.view.kappa_R, g, log_ni, log_T);
    sigma_a[base + g] = fmin(
        fmax(fmax(rho_safe * kappa_pa, 0.0), sigma_min), sigma_max);
    sigma_pe[base + g] = fmin(
        fmax(fmax(rho_safe * kappa_pe, 0.0), sigma_min), sigma_max);
    sigma_R[base + g] = fmin(
        fmax(fmax(rho_safe * kappa_r, 0.0), sigma_min), sigma_max);
  }
}

__global__ void build_nlte_cell_mask_kernel(
    const int* __restrict__ cell_material_index,
    const MatOpacityDesc* __restrict__ descs,
    int n_materials,
    std::uint8_t* __restrict__ mask,
    int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) return;
  const int m = min(max(cell_material_index[c], 0), n_materials - 1);
  mask[c] = (descs[m].kind == 2) ? 1U : 0U;
}

__device__ inline void build_eta_from_planck_kernel_body(
    const int idx,
    const double* __restrict__ Te,
    const double* __restrict__ sigma_a,
    PlanckTableDeviceView planck,
    double* __restrict__ eta,
    int n_cells,
    int n_groups,
    double temperature_floor_eV,
    const std::uint8_t* __restrict__ cell_eval_skip) {
  const int c = idx / n_groups;
  if (cell_eval_skip != nullptr && cell_eval_skip[c] != 0U) {
    return;
  }
  const int g = idx - c * n_groups;
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T2 = T * T;
  const double B = core::constants::a_eV * T2 * T2 *
                   fmax(planck.interpolate_b(g, T), 0.0);
  eta[idx] = core::constants::c_light * fmax(finite_or_zero(sigma_a[idx]), 0.0) * B;
}

__global__ void build_eta_from_planck_kernel(
    const double* __restrict__ Te,
    const double* __restrict__ sigma_a,
    PlanckTableDeviceView planck,
    double* __restrict__ eta,
    int n_cells,
    int n_groups,
    double temperature_floor_eV,
    const std::uint8_t* __restrict__ cell_eval_skip) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  build_eta_from_planck_kernel_body(idx, Te, sigma_a, planck, eta, n_cells,
                                    n_groups, temperature_floor_eV,
                                    cell_eval_skip);
}

__device__ double harmonic_positive(const double a, const double b) {
  if (!(a > 0.0) || !(b > 0.0)) {
    return 0.0;
  }
  return 2.0 / (1.0 / a + 1.0 / b);
}

// Face-centered flux-limiter fix (Turner & Stone 2001, Fig. 2:
// D lives on zone faces; CASTRO II Sec. 6.4: the limiter is what lets a
// radiation front propagate at ~c). The previous cell-centered lambda +
// harmonic averaging collapsed the face D to ~2*D_cold at a hot/cold front
// (R evaluated with the receiver cell's E), throttling the face flux to
// ~c*E_cold and stalling Marshak fronts; uniform planar cell volumes expose
// the stall while spherical center volumes masked it. The face form uses
// the face-averaged E and the face gradient; in the smooth optically-thick
// limit (lambda == 1/3 on both sides) it reproduces the old harmonic form
// exactly: harmonic(c/(3 sL), c/(3 sR)) == c / (3 * 0.5 * (sL + sR)).
// f is the interior face index (1 <= f <= n_cells-1): cells f-1 | f.
__device__ double fld_face_diffusion_coeff(const double* __restrict__ x_r,
                                           const double* __restrict__ rad_E,
                                           const double* __restrict__ sigma_R,
                                           const int f,
                                           const int g,
                                           const int n_groups,
                                           const double sigma_floor,
                                           const int limiter) {
  const int lcg = (f - 1) * n_groups + g;
  const int rcg = f * n_groups + g;
  const double El = fmax(finite_or_zero(rad_E[lcg]), 0.0);
  const double Er = fmax(finite_or_zero(rad_E[rcg]), 0.0);
  const double xl = 0.5 * (x_r[f - 1] + x_r[f]);
  const double xr = 0.5 * (x_r[f] + x_r[f + 1]);
  const double dist = fmax(xr - xl, 1.0e-300);
  const double sl = fmax(finite_or_zero(sigma_R[lcg]), sigma_floor);
  const double sr = fmax(finite_or_zero(sigma_R[rcg]), sigma_floor);
  const double sigma_face = 0.5 * (sl + sr);
  const double E_face = fmax(0.5 * (El + Er), 1.0e-300);
  const double R = fabs(Er - El) / (dist * sigma_face * E_face);
  const double lambda = flux_limiter_lambda(R, limiter);
  return core::constants::c_light * lambda / sigma_face;
}

bool use_fld_fleck(const core::Config::MaterialsConfig::MatDef& mat) {
  return mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat" ||
         mat.opacity_model == "constant" || mat.opacity_model == "power_law";
}

__global__ void compute_fleck_for_fld_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,
    const double* __restrict__ state_cv_e,
    double* __restrict__ f_fleck,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double alpha,
    const double cv_e_override,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const materials::DeviceEOSTableView electron_eos,
    const int use_table_cv,
    const double temperature_floor_eV,
    const double* __restrict__ rad_E_old,
    const PlanckTableDeviceView planck,
    const int fleck_beta_secant,
    const int fleck_form_exp,
    const std::uint8_t* __restrict__ cell_eval_skip) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  fld_1d_bodies::compute_fleck_for_fld_kernel_body(c, rho, Te, zbar, cell_is_void, sigma_a,
                                    state_cv_e, f_fleck, n_cells, n_groups, dt,
                                    alpha, cv_e_override, gamma_eff, A_eff,
                                    electron_eos, use_table_cv,
                                    temperature_floor_eV, rad_E_old, planck,
                                    fleck_beta_secant, fleck_form_exp,
                                    cell_eval_skip);
}

// 1D FLD outer-boundary kinds, parsed from
// Radiation.multigroup_diffusion.boundary.outer_r (NUMERICS §6.7 1D BC).
// Inner r=0 is always reflecting by spherical symmetry.
constexpr int kFld1dOuterVacuum = 0;
constexpr int kFld1dOuterReflect = 1;
constexpr int kFld1dOuterMarshak = 2;

int parse_fld_1d_outer_bc(const std::string& value) {
  if (value == "vacuum") {
    return kFld1dOuterVacuum;
  }
  if (value == "reflect" || value == "reflective") {
    return kFld1dOuterReflect;
  }
  if (value == "marshak") {
    return kFld1dOuterMarshak;
  }
  // Namelist validation is the primary gate; this is a defensive backstop.
  TENRYU_ASSERT(false,
                "FLD 1D: unsupported multigroup_diffusion.boundary.outer_r");
  return kFld1dOuterVacuum;
}

// Outgoing boundary-leakage coefficient entering the outer-cell diagonal and
// the escaped-energy tally: vacuum uses the half-range escape cE/2 (documented
// 1D convention), marshak uses the Milne/Marshak cE/4 paired with an incoming
// flux on the RHS, reflect leaks nothing.
__host__ __device__ inline double fld_1d_outer_leak_coeff(const int bc) {
  if (bc == kFld1dOuterVacuum) {
    return 0.5 * core::constants::c_light;
  }
  if (bc == kFld1dOuterMarshak) {
    return 0.25 * core::constants::c_light;
  }
  return 0.0;
}

// Per-group incoming Marshak flux [erg/cm^2/s]. Priority: blackbody drive at
// T_r (F_inc,g = (c/4) a_eV T_r^4 b_g(T_r)); otherwise a grey constant flux
// (validated to G==1) assigned to group 0.
__device__ inline void compute_marshak_finc_kernel_body(
    const int g,
    const PlanckTableDeviceView planck,
    const double T_r_eV,
    const double const_flux,
    double* __restrict__ finc,
    const int n_groups) {
  if (T_r_eV > 0.0) {
    const double T2 = T_r_eV * T_r_eV;
    const double b = fmax(planck.interpolate_b(g, T_r_eV), 0.0);
    finc[g] =
        0.25 * core::constants::c_light * core::constants::a_eV * T2 * T2 * b;
    return;
  }
  finc[g] = (g == 0) ? fmax(const_flux, 0.0) : 0.0;
}

__global__ void compute_marshak_finc_kernel(const PlanckTableDeviceView planck,
                                            const double T_r_eV,
                                            const double const_flux,
                                            double* __restrict__ finc,
                                            const int n_groups) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= n_groups) {
    return;
  }
  compute_marshak_finc_kernel_body(g, planck, T_r_eV, const_flux, finc,
                                   n_groups);
}

template <int GEOM>
__device__ inline void assemble_fld_tridiag_kernel_body(
    const int idx,
    const double* __restrict__ x_r,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ sigma_pa,
    const double* __restrict__ fleck,
    const double* __restrict__ eta,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ rad_E_iter,
    const double* __restrict__ sigma_R,
    double* __restrict__ lower,
    double* __restrict__ diag,
    double* __restrict__ upper,
    double* __restrict__ rhs,
    int n_cells,
    int n_groups,
    double dt,
    int outer_bc,
    const double* __restrict__ marshak_finc,
    double volume_source_rate,
    double volume_source_r_max,
    double sigma_floor,
    int limiter,
    const int exchange_off) {
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  const int cg = c * n_groups + g;
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  const double sigma = fmax(finite_or_zero(sigma_removal[cg]), 0.0);
  double d = V;
  if (exchange_off == 0) {
    d += dt * core::constants::c_light * sigma * V;
  }
  double l = 0.0;
  double u = 0.0;
  if (c > 0) {
    const double Df = fld_face_diffusion_coeff(
        x_r, rad_E_iter, sigma_R, c, g, n_groups, sigma_floor, limiter);
    const double xf = x_r[c];
    double area;
    if constexpr (GEOM == 0) {
      area = 4.0 * kPi * xf * xf;
    } else {
      area = tenryu::mesh::geometry_1d_face_area(GEOM, xf);
    }
    const double xl = 0.5 * (x_r[c - 1] + x_r[c]);
    const double xc = 0.5 * (x_r[c] + x_r[c + 1]);
    const double dist = fmax(xc - xl, 1.0e-300);
    const double coef = dt * area * Df / dist;
    d += coef;
    l = -coef;
  }
  if (c + 1 < n_cells) {
    const double Df = fld_face_diffusion_coeff(
        x_r, rad_E_iter, sigma_R, c + 1, g, n_groups, sigma_floor, limiter);
    const double xf = x_r[c + 1];
    double area;
    if constexpr (GEOM == 0) {
      area = 4.0 * kPi * xf * xf;
    } else {
      area = tenryu::mesh::geometry_1d_face_area(GEOM, xf);
    }
    const double xc = 0.5 * (x_r[c] + x_r[c + 1]);
    const double xr = 0.5 * (x_r[c + 1] + x_r[c + 2]);
    const double dist = fmax(xr - xc, 1.0e-300);
    const double coef = dt * area * Df / dist;
    d += coef;
    u = -coef;
  } else {
    const double r_outer = x_r[n_cells];
    double area;
    if constexpr (GEOM == 0) {
      area = 4.0 * kPi * r_outer * r_outer;
    } else {
      area = tenryu::mesh::geometry_1d_face_area(GEOM, r_outer);
    }
    d += dt * area * fld_1d_outer_leak_coeff(outer_bc);
  }
  lower[idx] = l;
  diag[idx] = d;
  upper[idx] = u;
  const double E_old = fmax(finite_or_zero(rad_E_old[cg]), 0.0);
  double source = (exchange_off == 0) ? fmax(finite_or_zero(eta[cg]), 0.0) : 0.0;
  if (exchange_off == 0 && fleck != nullptr && sigma_pa != nullptr) {
    // fleck is a [n_cells x n_groups] field like every other cg array here;
    // indexing it with the bare cell index read a wrong element for G > 1.
    const double f = fmin(fmax(finite_or_zero(fleck[cg]), 0.0), 1.0);
    const double sigma_raw = fmax(finite_or_zero(sigma_pa[cg]), 0.0);
    source = f * source +
             (1.0 - f) * core::constants::c_light * sigma_raw * E_old;
  }
  double rhs_val = V * E_old + dt * V * source;
  // External radiation volume source (Su-Olson class). Not part of the matter
  // emission linearization, so it is added outside the Fleck blend. Namelist
  // validation restricts it to G==1; the g==0 guard keeps an invalid
  // multigroup config from multiplying the source by G.
  if (g == 0 && volume_source_rate > 0.0 && volume_source_r_max > 0.0) {
    const double r_c = 0.5 * (x_r[c] + x_r[c + 1]);
    if (r_c <= volume_source_r_max) {
      rhs_val += dt * V * volume_source_rate;
    }
  }
  if (outer_bc == kFld1dOuterMarshak && marshak_finc != nullptr &&
      c + 1 == n_cells) {
    const double r_outer = x_r[n_cells];
    double area;
    if constexpr (GEOM == 0) {
      area = 4.0 * kPi * r_outer * r_outer;
    } else {
      area = tenryu::mesh::geometry_1d_face_area(GEOM, r_outer);
    }
    rhs_val += dt * area * fmax(finite_or_zero(marshak_finc[g]), 0.0);
  }
  rhs[idx] = rhs_val;
}

template <int GEOM>
__global__ void assemble_fld_tridiag_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ sigma_pa,
    const double* __restrict__ fleck,
    const double* __restrict__ eta,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ rad_E_iter,
    const double* __restrict__ sigma_R,
    double* __restrict__ lower,
    double* __restrict__ diag,
    double* __restrict__ upper,
    double* __restrict__ rhs,
    int n_cells,
    int n_groups,
    double dt,
    int outer_bc,
    const double* __restrict__ marshak_finc,
    double volume_source_rate,
    double volume_source_r_max,
    double sigma_floor,
    int limiter,
    const int exchange_off) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  assemble_fld_tridiag_kernel_body<GEOM>(
      idx, x_r, vol, sigma_removal, sigma_pa, fleck, eta, rad_E_old, rad_E_iter,
      sigma_R, lower, diag, upper, rhs, n_cells, n_groups, dt, outer_bc,
      marshak_finc, volume_source_rate, volume_source_r_max, sigma_floor,
      limiter, exchange_off);
}

__device__ inline void publish_solution_kernel_body(
    const int idx,
    const double* __restrict__ rhs,
    double* __restrict__ rad_E,
    int n_cells,
    int n_groups,
    int* __restrict__ nonfinite_count) {
  const int g = idx / n_cells;
  const int c = idx - g * n_cells;
  if (nonfinite_count != nullptr && !isfinite(rhs[idx])) {
    atomicAdd(nonfinite_count, 1);
  }
  rad_E[c * n_groups + g] = fmax(finite_or_zero(rhs[idx]), 0.0);
}

__global__ void publish_solution_kernel(const double* __restrict__ rhs,
                                        double* __restrict__ rad_E,
                                        int n_cells,
                                        int n_groups,
                                        int* __restrict__ nonfinite_count) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  publish_solution_kernel_body(idx, rhs, rad_E, n_cells, n_groups,
                               nonfinite_count);
}

__device__ inline void snapshot_Te_kernel_body(const int c,
                                               const double* __restrict__ Te,
                                               double* __restrict__ Te_old,
                                               int n_cells) {
  Te_old[c] = finite_or_zero(Te[c]);
}

__global__ void snapshot_Te_kernel(const double* __restrict__ Te,
                                   double* __restrict__ Te_old,
                                   int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  snapshot_Te_kernel_body(c, Te, Te_old, n_cells);
}

template <bool EOS_TAIL>
__device__ inline void update_matter_body(
    const int cell,
    const int lane,
    double* smem,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ Te_old,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_pe,
    const double* __restrict__ rad_E,
    const double* __restrict__ fleck,      // nullable [n_cells*n_groups]
    const double* __restrict__ rad_E_old,  // required when fleck != nullptr
    PlanckTableDeviceView planck,
    materials::DeviceEOSTableView electron_eos,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    int n_cells,
    int n_groups,
    double dt,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    double cv_e_override,
    double temperature_floor_eV,
    int has_cv_e,
    int max_newton_iterations,
    double newton_tolerance) {
  const int c = cell;
  const int tid = lane;

  double* s_F_g = smem;
  double* s_dF_g = smem + n_groups;

  __shared__ double s_T;
  __shared__ double s_step_T_ref;
  __shared__ double s_iter_T_prev;
  __shared__ double s_rho_c;
  __shared__ double s_cv_mass;
  __shared__ double s_dt_safe;
  __shared__ double s_Cv;
  __shared__ double s_e_e_ref;
  __shared__ int s_break;
  __shared__ int s_use_table_eos;
  __shared__ materials::RhoBracket s_eos_rho_bracket;

  if (tid == 0) {
    const double rho_c = positive_or_floor(rho[c], 1.0e-300);
    s_rho_c = rho_c;
    s_step_T_ref = fmax(finite_or_zero(Te_old[c]), temperature_floor_eV);
    s_iter_T_prev = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
    s_T = s_iter_T_prev;
    double cv_mass = 0.0;
    if (has_cv_e != 0 && cv_e != nullptr && cv_e[c] > 0.0 && isfinite(cv_e[c])) {
      cv_mass = cv_e[c];
    } else if (cv_e_override > 0.0) {
      cv_mass = cv_e_override / rho_c;
    } else {
      // Shared per-cell effective properties (multi-material fix; was materials[0]'s
      // scalar A/gamma for every cell — wrong in multi-material decks).
      const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
      const double gm1 = fmax(gamma_c - 1.0, 1.0e-12);
      const double z = fmax(finite_or_zero(zbar[c]), 0.0);
      cv_mass = z * core::constants::eV_to_erg /
                (fmax(A_eff[c], 1.0e-12) * core::constants::proton_mass * gm1);
    }
    s_cv_mass = fmax(finite_or_zero(cv_mass), 1.0e-300);
    s_dt_safe = fmax(dt, 1.0e-300);
    s_Cv = fmax(rho_c * s_cv_mass, 1.0e-300);
    s_use_table_eos = has_electron_eos_table(electron_eos) ? 1 : 0;
    if (s_use_table_eos != 0) {
      s_eos_rho_bracket = materials::find_rho_bracket(electron_eos, rho_c);
      if constexpr (EOS_TAIL) {
        s_e_e_ref = materials::device_eos_eval_with_high_t_tail(
                        electron_eos, s_eos_rho_bracket, s_step_T_ref,
                        temperature_floor_eV)
                        .e;
      } else {
        s_e_e_ref = materials::device_eos_energy(
            electron_eos,
            s_eos_rho_bracket,
            eos_log_temperature(s_step_T_ref, temperature_floor_eV));
      }
    } else {
      s_e_e_ref = 0.0;
    }
    s_break = 0;
  }
  __syncthreads();

  const int max_iter = (max_newton_iterations > 1) ? max_newton_iterations : 1;
  const double tol = fmax(newton_tolerance, 0.0);
  for (int iter = 0; iter < max_iter; ++iter) {
    if (s_break != 0) {
      break;
    }
    const double T = s_T;
    const double T4 = safe_pow4(T);
    const double T3 = (T > 0.0) ? (T4 / T) : 0.0;

    for (int g = tid; g < n_groups; g += blockDim.x) {
      const int idx = c * n_groups + g;
      const double sigma_pa = nonnegative_finite(sigma_a[idx]);
      const double sigma_pe_g =
          (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
      const double E = nonnegative_finite(rad_E[idx]);
      const double b = fmax(planck.interpolate_b(g, T), 0.0);
      const double B = core::constants::a_eV * T4 * b;
      // Fleck-consistent exchange (W-B conservation fix): the E-equation
      // emits f*eta + (1-f)*c*sigma*E_old, so the matter must lose exactly
      // that blend — booking the unblended c*sigma*B here created
      // O((1-f)*c*sigma*dt*|E_old-B|) phantom energy per step.
      double f = 1.0;
      double E_old_g = 0.0;
      if (fleck != nullptr) {
        f = fmin(fmax(finite_or_zero(fleck[idx]), 0.0), 1.0);
        E_old_g = (rad_E_old != nullptr)
                      ? fmax(finite_or_zero(rad_E_old[idx]), 0.0)
                      : 0.0;
      }
      const double emit_g = f * core::constants::c_light * sigma_pe_g * B +
                            (1.0 - f) * core::constants::c_light * sigma_pe_g *
                                E_old_g;
      s_F_g[g] = -(core::constants::c_light * sigma_pa * E - emit_g);
      s_dF_g[g] = f * core::constants::c_light * sigma_pe_g *
                  4.0 * core::constants::a_eV * T3 * b;
    }
    __syncthreads();

    if (tid == 0) {
      double F = 0.0;
      double dF = 0.0;
      if (s_use_table_eos != 0) {
        double e_e_T;
        double cv_e_T;
        if constexpr (EOS_TAIL) {
          const auto th = materials::device_eos_eval_with_high_t_tail(
              electron_eos, s_eos_rho_bracket, T, temperature_floor_eV);
          e_e_T = th.e;
          cv_e_T = fmax(nonnegative_finite(th.cv), 1.0e-300);
        } else {
          const double logT = eos_log_temperature(T, temperature_floor_eV);
          e_e_T =
              materials::device_eos_energy(electron_eos, s_eos_rho_bracket, logT);
          cv_e_T = fmax(
              nonnegative_finite(
                  materials::device_eos_cv(electron_eos, s_eos_rho_bracket, logT)),
              1.0e-300);
        }
        F = s_rho_c * (e_e_T - s_e_e_ref) / s_dt_safe;
        dF = s_rho_c * cv_e_T / s_dt_safe;
      } else {
        F = s_Cv * (T - s_step_T_ref) / s_dt_safe;
        dF = s_Cv / s_dt_safe;
      }
      for (int g = 0; g < n_groups; ++g) {
        F += s_F_g[g];
        dF += s_dF_g[g];
      }
      if (!(isfinite(F) && isfinite(dF)) || !(dF > 0.0)) {
        s_break = 1;
      } else {
        double dT = -F / dF;
        const double max_step = 0.5 * fmax(T, temperature_floor_eV);
        dT = fmin(fmax(dT, -max_step), max_step);
        const double T_next = fmax(T + dT, temperature_floor_eV);
        const double rel = fabs(T_next - T) / fmax(T_next, temperature_floor_eV);
        s_T = T_next;
        if (rel <= tol) {
          s_break = 1;
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const double T = s_T;
    Te[c] = T;
    if (s_use_table_eos != 0) {
      if constexpr (EOS_TAIL) {
        const auto th = materials::device_eos_eval_with_high_t_tail(
            electron_eos, s_eos_rho_bracket, T, temperature_floor_eV);
        ee[c] = th.e;
        Pe[c] = th.P;
      } else {
        const double logT = eos_log_temperature(T, temperature_floor_eV);
        ee[c] =
            materials::device_eos_energy(electron_eos, s_eos_rho_bracket, logT);
        Pe[c] =
            materials::device_eos_pressure(electron_eos, s_eos_rho_bracket, logT);
      }
    } else {
      ee[c] = finite_or_zero(ee[c]) + s_cv_mass * (T - s_iter_T_prev);
      Pe[c] = fmax(fmax(gamma_eff[c], 1.0 + 1.0e-12) - 1.0, 1.0e-12) * s_rho_c *
              finite_or_zero(ee[c]);
    }
    delta_T_rel[c] = fabs(T - s_iter_T_prev) / fmax(T, temperature_floor_eV);
  }
  __syncthreads();

  const double T_final = s_T;
  const double Tn4 = safe_pow4(T_final);
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  for (int g = tid; g < n_groups; g += blockDim.x) {
    const int idx = c * n_groups + g;
    const double sigma_pa = nonnegative_finite(sigma_a[idx]);
    const double sigma_pe_g =
        (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
    const double E = nonnegative_finite(rad_E[idx]);
    const double b = fmax(planck.interpolate_b(g, T_final), 0.0);
    double f = 1.0;
    double E_old_g = 0.0;
    if (fleck != nullptr) {
      f = fmin(fmax(finite_or_zero(fleck[idx]), 0.0), 1.0);
      E_old_g = (rad_E_old != nullptr)
                    ? fmax(finite_or_zero(rad_E_old[idx]), 0.0)
                    : 0.0;
    }
    rad_dep[idx] = dt * V * core::constants::c_light * sigma_pa * E;
    rad_emit[idx] =
        dt * V *
        (f * core::constants::c_light * sigma_pe_g * core::constants::a_eV *
             Tn4 * b +
         (1.0 - f) * core::constants::c_light * sigma_pe_g * E_old_g);
  }
}

template <bool EOS_TAIL>
__global__ void update_matter_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ Te_old,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_pe,
    const double* __restrict__ rad_E,
    const double* __restrict__ fleck,      // nullable [n_cells*n_groups]
    const double* __restrict__ rad_E_old,  // required when fleck != nullptr
    PlanckTableDeviceView planck,
    materials::DeviceEOSTableView electron_eos,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    int n_cells,
    int n_groups,
    double dt,
    const double* __restrict__ A_eff,
    const double* __restrict__ gamma_eff,
    double cv_e_override,
    double temperature_floor_eV,
    int has_cv_e,
    int max_newton_iterations,
    double newton_tolerance) {
  const int c = blockIdx.x;
  const int tid = threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  extern __shared__ double smem[];
  update_matter_body<EOS_TAIL>(
      c, tid, smem, rho, vol, zbar, cv_e, Te_old, sigma_a, sigma_pe, rad_E,
      fleck, rad_E_old, planck, electron_eos, Te, ee, Pe, rad_dep, rad_emit,
      delta_T_rel, n_cells, n_groups, dt, A_eff, gamma_eff, cv_e_override,
      temperature_floor_eV, has_cv_e, max_newton_iterations,
      newton_tolerance);
}

// rung-2 exp_rosenbrock source substep (grey): exact frozen-coefficient
// direct transfer q = h*phi_1(-(1+beta)h)*(E - aT^4) applied symmetrically
// (E -= q, U_e += q) — design docs/design/fleck_exp_source_20260716.md §3.
// Coefficient chain and ee/Pe closure mirror update_matter_body; the phi_1
// evaluator is the shared fld_1d_bodies::fleck_form_phi1 (phi_1(-x)).
template <bool EOS_TAIL>
__device__ inline void exp_source_transfer_body(
    const int c,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ state_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,
    double* __restrict__ rad_E,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    const int n_cells,
    const double dt,
    const double cv_e_override,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const materials::DeviceEOSTableView electron_eos,
    const double temperature_floor_eV) {
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    rad_dep[c] = 0.0;
    rad_emit[c] = 0.0;
    delta_T_rel[c] = 0.0;
    return;
  }
  const double rho_c = positive_or_floor(rho[c], 1.0e-300);
  const double T_n = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const bool use_table = has_electron_eos_table(electron_eos);
  materials::RhoBracket br{};
  double cv_mass = 0.0;
  if (use_table) {
    br = materials::find_rho_bracket(electron_eos, rho_c);
    cv_mass = fmax(
        nonnegative_finite(materials::device_eos_cv(
            electron_eos, br,
            eos_log_temperature(T_n, temperature_floor_eV))),
        1.0e-300);
  } else if (state_cv_e != nullptr && state_cv_e[c] > 0.0 &&
             isfinite(state_cv_e[c])) {
    cv_mass = state_cv_e[c];
  } else if (cv_e_override > 0.0) {
    cv_mass = cv_e_override / rho_c;
  } else {
    const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
    const double gm1 = fmax(gamma_c - 1.0, 1.0e-12);
    const double z = fmax(finite_or_zero(zbar[c]), 0.0);
    cv_mass = z * core::constants::eV_to_erg /
              (fmax(A_eff[c], 1.0e-12) * core::constants::proton_mass * gm1);
  }
  const double Cv = fmax(rho_c * fmax(finite_or_zero(cv_mass), 1.0e-300),
                         1.0e-300);
  const double sigma = nonnegative_finite(sigma_a[c]);
  const double h = core::constants::c_light * sigma * fmax(dt, 0.0);
  const double T2 = T_n * T_n;
  const double B_n = core::constants::a_eV * T2 * T2;
  double beta = 4.0 * core::constants::a_eV * T_n * T2 / Cv;
  if (!isfinite(beta) || beta < 0.0) {
    beta = 0.0;
  }
  const double x = (1.0 + beta) * h;
  const double E_n = fmax(finite_or_zero(rad_E[c]), 0.0);
  const double q = h * fld_1d_bodies::fleck_form_phi1(x) * (E_n - B_n);
  rad_E[c] = fmax(E_n - q, 0.0);
  double T_new = T_n;
  if (use_table) {
    double e_n;
    if constexpr (EOS_TAIL) {
      e_n = fmax(finite_or_zero(ee[c]), 0.0);
    } else {
      e_n = materials::device_eos_energy(
          electron_eos, br, eos_log_temperature(T_n, temperature_floor_eV));
    }
    const double e_new = e_n + q / rho_c;
    double T = T_n;
    for (int it = 0; it < 50; ++it) {
      double eT;
      double cvT;
      if constexpr (EOS_TAIL) {
        const auto th = materials::device_eos_eval_with_high_t_tail(
            electron_eos, br, T, temperature_floor_eV);
        eT = th.e;
        cvT = fmax(nonnegative_finite(th.cv), 1.0e-300);
      } else {
        const double logT = eos_log_temperature(T, temperature_floor_eV);
        eT =
            materials::device_eos_energy(electron_eos, br, logT);
        cvT = fmax(
            nonnegative_finite(
                materials::device_eos_cv(electron_eos, br, logT)),
            1.0e-300);
      }
      double dT = (e_new - eT) / cvT;
      const double max_step = 0.5 * fmax(T, temperature_floor_eV);
      dT = fmin(fmax(dT, -max_step), max_step);
      const double T_next = fmax(T + dT, temperature_floor_eV);
      const bool done =
          fabs(T_next - T) <= 1.0e-12 * fmax(T_next, temperature_floor_eV);
      T = T_next;
      if (done) {
        break;
      }
    }
    T_new = T;
    if constexpr (EOS_TAIL) {
      ee[c] = fmax(e_n + q / rho_c, 0.0);
      const auto th = materials::device_eos_eval_with_high_t_tail(
          electron_eos, br, T_new, temperature_floor_eV);
      Pe[c] = th.P;
    } else {
      const double logT = eos_log_temperature(T_new, temperature_floor_eV);
      ee[c] = materials::device_eos_energy(electron_eos, br, logT);
      Pe[c] = materials::device_eos_pressure(electron_eos, br, logT);
    }
  } else {
    T_new = fmax(T_n + (q / rho_c) / fmax(cv_mass, 1.0e-300),
                 temperature_floor_eV);
    ee[c] = finite_or_zero(ee[c]) + q / rho_c;
    Pe[c] = fmax(fmax(gamma_eff[c], 1.0 + 1.0e-12) - 1.0, 1.0e-12) * rho_c *
            finite_or_zero(ee[c]);
  }
  Te[c] = T_new;
  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  // Net-accounting ledger (the exact map defines the net transfer only):
  // q > 0 = matter gains (deposition), q < 0 = matter loses (emission).
  rad_dep[c] = V * fmax(q, 0.0);
  rad_emit[c] = V * fmax(-q, 0.0);
  delta_T_rel[c] = fabs(T_new - T_n) / fmax(T_new, temperature_floor_eV);
}

template <bool EOS_TAIL>
__global__ void exp_source_transfer_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ state_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,
    double* __restrict__ rad_E,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    const int n_cells,
    const double dt,
    const double cv_e_override,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const materials::DeviceEOSTableView electron_eos,
    const double temperature_floor_eV) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  exp_source_transfer_body<EOS_TAIL>(
      c, rho, vol, zbar, state_cv_e, cell_is_void, sigma_a, rad_E, Te, ee, Pe,
      rad_dep, rad_emit, delta_T_rel, n_cells, dt, cv_e_override, gamma_eff,
      A_eff, electron_eos, temperature_floor_eV);
}

// rung-2 multigroup exp_rosenbrock source substep (design doc
// docs/design/exp_mg_phi1_20260717.md §1): exact conservative transfer on
// the conserved hyperplane via the G x G diagonal-plus-rank-one reduction,
//   K = -X - X*gamma*1^T (dimensionless), r_g = xi_g (b_g B_n - E_g),
//   dE = phi1(K) r,  dU = -sum_g dE_g,
// with phi1 evaluated by the certified fixed-pole rational set
// (phi1_poles.hpp; per pole a Sherman-Morrison O(G) solve). gamma_g =
// d(b_g B)/dU = (b_g 4 a T^3 + B_n db_g/dT)/Cv keeps second order under
// varying Planck fractions (verdict §5.4). Single thread per cell;
// fixed-order sequential sums (deterministic). v1 safeguard: on a
// non-finite or radiation-negative trial the CELL transfer is rejected
// (dE=0) and counted — a documented deviation from the verdict's
// per-cell Fleck fallback (structurally heavier; revisit if gates demand).
struct Phi1PoleSet {
  double theta_re[phi1poles::kNPoles];
  double theta_im[phi1poles::kNPoles];
  double omega_re[phi1poles::kNPoles];
  double omega_im[phi1poles::kNPoles];
};

inline Phi1PoleSet make_phi1_pole_set() {
  Phi1PoleSet s{};
  for (int k = 0; k < phi1poles::kNPoles; ++k) {
    s.theta_re[k] = phi1poles::kThetaRe[k];
    s.theta_im[k] = phi1poles::kThetaIm[k];
    s.omega_re[k] = phi1poles::kOmegaRe[k];
    s.omega_im[k] = phi1poles::kOmegaIm[k];
  }
  return s;
}

template <bool EOS_TAIL>
__device__ inline void exp_mg_source_transfer_body(
    const int c,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ state_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,   // [n_cells*n_groups]
    double* __restrict__ rad_E,           // [n_cells*n_groups]
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,         // [n_cells*n_groups]
    double* __restrict__ rad_emit,        // [n_cells*n_groups]
    double* __restrict__ delta_T_rel,
    int* __restrict__ reject_count,       // single int, atomicAdd on reject
    int* __restrict__ reject_info,    // [0]=slot cursor; slots of 3 ints {cell, reason, group}, capacity 8
    double* __restrict__ reject_vals, // slots of 3 doubles {E_g, dE_or_dU, Te_n}, capacity 8
    const int n_cells,
    const int n_groups,
    const double dt,
    const double cv_e_override,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const materials::DeviceEOSTableView electron_eos,
    const PlanckTableDeviceView planck,
    const double temperature_floor_eV,
    const Phi1PoleSet poles) {
  constexpr int kMaxG = 96;
  const int base = c * n_groups;
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    for (int g = 0; g < n_groups; ++g) {
      rad_dep[base + g] = 0.0;
      rad_emit[base + g] = 0.0;
    }
    delta_T_rel[c] = 0.0;
    return;
  }
  const double rho_c = positive_or_floor(rho[c], 1.0e-300);
  const double T_n = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const bool use_table = has_electron_eos_table(electron_eos);
  materials::RhoBracket br{};
  double cv_mass = 0.0;
  if (use_table) {
    br = materials::find_rho_bracket(electron_eos, rho_c);
    cv_mass = fmax(
        nonnegative_finite(materials::device_eos_cv(
            electron_eos, br,
            eos_log_temperature(T_n, temperature_floor_eV))),
        1.0e-300);
  } else if (state_cv_e != nullptr && state_cv_e[c] > 0.0 &&
             isfinite(state_cv_e[c])) {
    cv_mass = state_cv_e[c];
  } else if (cv_e_override > 0.0) {
    cv_mass = cv_e_override / rho_c;
  } else {
    const double gamma_c = fmax(gamma_eff[c], 1.0 + 1.0e-12);
    const double gm1 = fmax(gamma_c - 1.0, 1.0e-12);
    const double z = fmax(finite_or_zero(zbar[c]), 0.0);
    cv_mass = z * core::constants::eV_to_erg /
              (fmax(A_eff[c], 1.0e-12) * core::constants::proton_mass * gm1);
  }
  const double Cv = fmax(rho_c * fmax(finite_or_zero(cv_mass), 1.0e-300),
                         1.0e-300);
  const double T2 = T_n * T_n;
  const double B_n = core::constants::a_eV * T2 * T2;
  const double fourAT3 = 4.0 * core::constants::a_eV * T_n * T2;

  double xi[kMaxG];
  double gam[kMaxG];
  double rr[kMaxG];
  double dE[kMaxG];
  if (n_groups > kMaxG) {
    // Structurally prevented by the builder cap; guard anyway.
    if (reject_count != nullptr) {
      atomicAdd(reject_count, 1);
    }
    return;
  }
  const double eps_T = 1.0e-3;
  const double T_lo = fmax(T_n * (1.0 - eps_T), temperature_floor_eV);
  const double T_hi = T_n * (1.0 + eps_T);
  bool ok = true;
  int rej_reason = 0;
  int rej_g = -1;
  double rej_E = 0.0;
  double rej_dE = 0.0;
  double E_sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    const double sg = nonnegative_finite(sigma_a[base + g]);
    xi[g] = core::constants::c_light * sg * fmax(dt, 0.0);
    const double b_g = fmax(planck.interpolate_b(g, T_n), 0.0);
    const double b_hi = fmax(planck.interpolate_b(g, T_hi), 0.0);
    const double b_lo = fmax(planck.interpolate_b(g, T_lo), 0.0);
    const double db_dT = (b_hi - b_lo) / fmax(T_hi - T_lo, 1.0e-300);
    gam[g] = (b_g * fourAT3 + B_n * db_dT) / Cv;
    if (!isfinite(gam[g])) {
      gam[g] = 0.0;
    }
    const double E_g = fmax(finite_or_zero(rad_E[base + g]), 0.0);
    E_sum += E_g;
    rr[g] = xi[g] * (b_g * B_n - E_g);
    dE[g] = 0.0;
    if (!isfinite(xi[g]) || !isfinite(rr[g])) {
      if (rej_reason == 0) {
        rej_reason = 1;
        rej_g = g;
        rej_E = E_g;
        rej_dE = 0.0;
      }
      ok = false;
    }
  }
  if (ok) {
    for (int k = 0; k < phi1poles::kNPoles; ++k) {
      const double thr = poles.theta_re[k];
      const double thi = poles.theta_im[k];
      // Sherman-Morrison: d_g = theta + xi_g; p=r/d; q=(xi*gam)/d;
      // P = sum p; S = 1 + sum q; x = p - q*P/S; dE += 2 Re(omega * x).
      double P_re = 0.0, P_im = 0.0, S_re = 1.0, S_im = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        const double dre = thr + xi[g];
        const double dim = thi;
        const double den = dre * dre + dim * dim;
        // p = r/d  (r real)
        const double p_re = rr[g] * dre / den;
        const double p_im = -rr[g] * dim / den;
        // q = u/d, u = xi*gam (real)
        const double u = xi[g] * gam[g];
        const double q_re = u * dre / den;
        const double q_im = -u * dim / den;
        P_re += p_re; P_im += p_im;
        S_re += q_re; S_im += q_im;
      }
      // ratio = P/S
      const double sden = S_re * S_re + S_im * S_im;
      const double rat_re = (P_re * S_re + P_im * S_im) / sden;
      const double rat_im = (P_im * S_re - P_re * S_im) / sden;
      const double owre = poles.omega_re[k];
      const double owim = poles.omega_im[k];
      for (int g = 0; g < n_groups; ++g) {
        const double dre = thr + xi[g];
        const double dim = thi;
        const double den = dre * dre + dim * dim;
        const double p_re = rr[g] * dre / den;
        const double p_im = -rr[g] * dim / den;
        const double u = xi[g] * gam[g];
        const double q_re = u * dre / den;
        const double q_im = -u * dim / den;
        const double x_re = p_re - (q_re * rat_re - q_im * rat_im);
        const double x_im = p_im - (q_re * rat_im + q_im * rat_re);
        dE[g] += 2.0 * (owre * x_re - owim * x_im);
      }
    }
    // Trial admissibility (verdict §4.7 items 3-4, v1 reject-and-count):
    double dU = 0.0;
    // Underflow-tail guard: groups whose content AND increment are both
    // vanishingly small relative to the cell's physical energy scale
    // (Wien-tail underflow, observed at 1e-60..1e-106 vs B_n) carry no
    // physics; a relative negativity test is meaningless there. Clamp
    // them to nonnegativity instead of rejecting the whole cell. dU is
    // accumulated from the CLAMPED dE, so conservation stays exact.
    const double tiny_scale = 1.0e-20 * fmax(B_n, E_sum);
    for (int g = 0; g < n_groups; ++g) {
      const double E_g = fmax(finite_or_zero(rad_E[base + g]), 0.0);
      if (isfinite(dE[g]) && E_g <= tiny_scale &&
          fabs(dE[g]) <= tiny_scale) {
        dE[g] = fmax(dE[g], -E_g);
      }
      const double E_new = E_g + dE[g];
      const double scale = fmax(fmax(E_g, fabs(dE[g])), 1.0e-300);
      if (!isfinite(dE[g]) || E_new < -1.0e-10 * scale) {
        if (rej_reason == 0) {
          rej_reason = 2;
          rej_g = g;
          rej_E = E_g;
          rej_dE = dE[g];
        }
        ok = false;
      }
      dU -= dE[g];
    }
    if (ok && !isfinite(dU)) {
      rej_reason = 3;
      rej_g = -1;
      rej_E = 0.0;
      rej_dE = dU;
      ok = false;
    }
    if (ok) {
      const double V = fmax(finite_or_zero(vol[c]), 0.0);
      for (int g = 0; g < n_groups; ++g) {
        const double E_g = fmax(finite_or_zero(rad_E[base + g]), 0.0);
        rad_E[base + g] = fmax(E_g + dE[g], 0.0);
        rad_dep[base + g] = V * fmax(-dE[g], 0.0);
        rad_emit[base + g] = V * fmax(dE[g], 0.0);
      }
      double T_new = T_n;
      if (use_table) {
        double e_n;
        if constexpr (EOS_TAIL) {
          e_n = fmax(finite_or_zero(ee[c]), 0.0);
        } else {
          e_n = materials::device_eos_energy(
              electron_eos, br, eos_log_temperature(T_n, temperature_floor_eV));
        }
        const double e_new = e_n + dU / rho_c;
        double T = T_n;
        for (int it = 0; it < 50; ++it) {
          double eT;
          double cvT;
          if constexpr (EOS_TAIL) {
            const auto th = materials::device_eos_eval_with_high_t_tail(
                electron_eos, br, T, temperature_floor_eV);
            eT = th.e;
            cvT = fmax(nonnegative_finite(th.cv), 1.0e-300);
          } else {
            const double logT = eos_log_temperature(T, temperature_floor_eV);
            eT =
                materials::device_eos_energy(electron_eos, br, logT);
            cvT = fmax(
                nonnegative_finite(
                    materials::device_eos_cv(electron_eos, br, logT)),
                1.0e-300);
          }
          double dT = (e_new - eT) / cvT;
          const double max_step = 0.5 * fmax(T, temperature_floor_eV);
          dT = fmin(fmax(dT, -max_step), max_step);
          const double T_next = fmax(T + dT, temperature_floor_eV);
          const bool done =
              fabs(T_next - T) <= 1.0e-12 * fmax(T_next, temperature_floor_eV);
          T = T_next;
          if (done) {
            break;
          }
        }
        T_new = T;
        if constexpr (EOS_TAIL) {
          ee[c] = fmax(e_new, 0.0);
          const auto th = materials::device_eos_eval_with_high_t_tail(
              electron_eos, br, T_new, temperature_floor_eV);
          Pe[c] = th.P;
        } else {
          const double logT = eos_log_temperature(T_new, temperature_floor_eV);
          ee[c] = materials::device_eos_energy(electron_eos, br, logT);
          Pe[c] = materials::device_eos_pressure(electron_eos, br, logT);
        }
      } else {
        T_new = fmax(T_n + (dU / rho_c) / fmax(cv_mass, 1.0e-300),
                     temperature_floor_eV);
        ee[c] = finite_or_zero(ee[c]) + dU / rho_c;
        Pe[c] = fmax(fmax(gamma_eff[c], 1.0 + 1.0e-12) - 1.0, 1.0e-12) *
                rho_c * finite_or_zero(ee[c]);
      }
      Te[c] = T_new;
      delta_T_rel[c] = fabs(T_new - T_n) / fmax(T_new, temperature_floor_eV);
      return;
    }
  }
  // Reject path: no transfer this step for this cell; count it loudly.
  for (int g = 0; g < n_groups; ++g) {
    rad_dep[base + g] = 0.0;
    rad_emit[base + g] = 0.0;
  }
  delta_T_rel[c] = 0.0;
  if (reject_count != nullptr) {
    atomicAdd(reject_count, 1);
  }
  if (reject_info != nullptr && reject_vals != nullptr) {
    const int slot = atomicAdd(&reject_info[0], 1);
    if (slot < 8) {
      reject_info[1 + 3 * slot] = c;
      reject_info[2 + 3 * slot] = rej_reason;
      reject_info[3 + 3 * slot] = rej_g;
      reject_vals[3 * slot] = rej_E;
      reject_vals[3 * slot + 1] = rej_dE;
      reject_vals[3 * slot + 2] = T_n;
    }
  }
}

template <bool EOS_TAIL>
__global__ void exp_mg_source_transfer_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ zbar,
    const double* __restrict__ state_cv_e,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,
    double* __restrict__ rad_E,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    int* __restrict__ reject_count,
    int* __restrict__ reject_info,
    double* __restrict__ reject_vals,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double cv_e_override,
    const double* __restrict__ gamma_eff,
    const double* __restrict__ A_eff,
    const materials::DeviceEOSTableView electron_eos,
    const PlanckTableDeviceView planck,
    const double temperature_floor_eV,
    const Phi1PoleSet poles) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  exp_mg_source_transfer_body<EOS_TAIL>(
      c, rho, vol, zbar, state_cv_e, cell_is_void, sigma_a, rad_E, Te, ee, Pe,
      rad_dep, rad_emit, delta_T_rel, reject_count, reject_info, reject_vals,
      n_cells, n_groups, dt, cv_e_override, gamma_eff, A_eff, electron_eos,
      planck, temperature_floor_eV, poles);
}

__device__ inline void max_reduce_kernel_body(const int i,
                                              const int tid,
                                              double* shared,
                                              const double* __restrict__ values,
                                              double* __restrict__ out,
                                              int n) {
  shared[tid] = (i < n) ? fmax(finite_or_zero(values[i]), 0.0) : 0.0;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmax(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomic_max_nonnegative_double(out, shared[0]);
  }
}

__global__ void max_reduce_kernel(const double* __restrict__ values,
                                  double* __restrict__ out,
                                  int n) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  max_reduce_kernel_body(i, tid, shared, values, out, n);
}

// Device-expressed outer-convergence predicate (persistent-kernel prep, task #49):
// pack[0] = residual (max |delta_T|), pack[1] holds converged flag as a double
// (1.0 / 0.0) so one 16-byte read returns both. The comparison lives ON DEVICE so a
// future device-resident outer loop reuses the same decision code; the host loop
// below consumes the flag without re-deriving it.
__device__ inline void fld_outer_converged_kernel_body(const int idx,
                                                       const double* reduction_work,
                                                       const double outer_tol,
                                                       double* pack2) {
  const double residual = reduction_work[0];
  pack2[0] = residual;
  pack2[1] = (residual < outer_tol) ? 1.0 : 0.0;
}

__global__ void fld_outer_converged_kernel(const double* reduction_work,
                                           const double outer_tol,
                                           double* pack2) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) {
    return;
  }
  fld_outer_converged_kernel_body(idx, reduction_work, outer_tol, pack2);
}

__global__ void reduce_outer_check_kernel(const double* __restrict__ values,
                                          const int n,
                                          const double outer_tol,
                                          double* pack2) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  double local_max = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    local_max = fmax(local_max, fmax(finite_or_zero(values[i]), 0.0));
  }
  shared[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmax(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double residual = shared[0];
    pack2[0] = residual;
    pack2[1] = (residual < outer_tol) ? 1.0 : 0.0;
  }
}

template <int GEOM>
__device__ inline void escaped_energy_kernel_body(
    const int g,
    const int tid,
    double* shared,
    const double* __restrict__ x_r,
    const double* __restrict__ rad_E,
    double* __restrict__ out,
    int n_cells,
    int n_groups,
    double dt,
    int outer_bc) {
  double local = 0.0;
  if (g < n_groups) {
    const double r = x_r[n_cells];
    double area;
    if constexpr (GEOM == 0) {
      area = 4.0 * kPi * r * r;
    } else {
      area = tenryu::mesh::geometry_1d_face_area(GEOM, r);
    }
    const double E = fmax(finite_or_zero(rad_E[(n_cells - 1) * n_groups + g]), 0.0);
    local = dt * area * fld_1d_outer_leak_coeff(outer_bc) * E;
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, shared[0]);
  }
}

template <int GEOM>
__global__ void escaped_energy_kernel(const double* __restrict__ x_r,
                                      const double* __restrict__ rad_E,
                                      double* __restrict__ out,
                                      int n_cells,
                                      int n_groups,
                                      double dt,
                                      int outer_bc) {
  if (n_cells <= 0) {
    return;
  }
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int g = blockIdx.x * blockDim.x + tid;
  escaped_energy_kernel_body<GEOM>(g, tid, shared, x_r, rad_E, out, n_cells,
                                   n_groups, dt, outer_bc);
}

// Energy injected by the external radiation volume source this step [erg]:
// sum over cells with r_center <= r_max of dt*V*rate (matches the RHS
// injection in assemble_fld_tridiag_kernel, which applies it to g==0 only).
__global__ void volume_source_energy_kernel(const double* __restrict__ x_r,
                                            const double* __restrict__ vol,
                                            double* __restrict__ out,
                                            int n_cells,
                                            double dt,
                                            double rate,
                                            double r_max) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int c = blockIdx.x * blockDim.x + tid;
  double local = 0.0;
  if (c < n_cells) {
    const double r_c = 0.5 * (x_r[c] + x_r[c + 1]);
    if (r_c <= r_max) {
      local = dt * fmax(finite_or_zero(vol[c]), 0.0) * rate;
    }
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, shared[0]);
  }
}

__global__ void max_reduced_flux_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ rad_E,
    const double* __restrict__ sigma_R,
    double* __restrict__ out,
    int n_cells,
    int n_groups,
    double sigma_floor,
    int limiter) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int n_faces = (n_cells > 1) ? (n_cells - 1) : 0;
  const int total = n_faces * n_groups;
  double local = 0.0;
  if (idx < total) {
    const int f = idx / n_groups;
    const int g = idx - f * n_groups;
    const int c0 = f;
    const int c1 = f + 1;
    const double xc0 = 0.5 * (x_r[c0] + x_r[c0 + 1]);
    const double xc1 = 0.5 * (x_r[c1] + x_r[c1 + 1]);
    const double dist = fmax(xc1 - xc0, 1.0e-300);
    const int i0 = c0 * n_groups + g;
    const int i1 = c1 * n_groups + g;
    const double E0 = fmax(finite_or_zero(rad_E[i0]), 0.0);
    const double E1 = fmax(finite_or_zero(rad_E[i1]), 0.0);
    const double Df = fld_face_diffusion_coeff(
        x_r, rad_E, sigma_R, f + 1, g, n_groups, sigma_floor, limiter);
    const double F = -Df * (E1 - E0) / dist;
    const double Eface = fmax(0.5 * (E0 + E1), 1.0e-300);
    const double ratio = fabs(F) / (core::constants::c_light * Eface);
    local = (ratio >= 0.0 && isfinite(ratio)) ? ratio : 0.0;
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmax(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomic_max_nonnegative_double(out, shared[0]);
  }
}

__global__ void fv_interior_flux_residual_kernel(
    const double* __restrict__ vol,
    const double* __restrict__ sigma_a,
    const double* __restrict__ lower,
    const double* __restrict__ diag,
    const double* __restrict__ upper,
    double* __restrict__ out,
    int n_cells,
    int n_groups,
    double dt) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int total = n_cells * n_groups;
  double local = 0.0;
  if (idx < total) {
    const int g = idx / n_cells;
    const int c = idx - g * n_cells;
    if (c != 0 && c + 1 < n_cells) {
      const int cg = c * n_groups + g;
      const double base_diag =
          fmax(finite_or_zero(vol[c]), 0.0) +
          dt * core::constants::c_light *
              fmax(finite_or_zero(sigma_a[cg]), 0.0) *
              fmax(finite_or_zero(vol[c]), 0.0);
      const double flux_sum = lower[idx] + (diag[idx] - base_diag) + upper[idx];
      local = fabs(finite_or_zero(flux_sum));
    }
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmax(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomic_max_nonnegative_double(out, shared[0]);
  }
}

void ensure_state_buffers(core::State& state, const int n_cells, const int n_groups) {
  const std::size_t n_total =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  state.rad_E.reset(n_total);
  state.rad_dep.reset(n_total);
  state.rad_emit.reset(n_total);
  state.rad_E_old.reset(n_total);
  state.fld_sigma_a.reset(n_total);
  state.fld_sigma_pe.reset(n_total);
  state.fld_sigma_R.reset(n_total);
  state.fld_eta.reset(n_total);
  state.fld_D_cell.reset(n_total);
  state.fld_lower.reset(n_total);
  state.fld_diag.reset(n_total);
  state.fld_upper.reset(n_total);
  state.fld_rhs.reset(n_total);
  state.fld_Te_old.reset(static_cast<std::size_t>(n_cells));
  state.fld_delta_T.reset(static_cast<std::size_t>(n_cells));
  state.fld_reduction_work.reset(1);
}

void initialize_rad_E_old_if_needed(core::State& state,
                                    const int n_cells,
                                    const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes == 0U) {
    return;
  }
  if (state.step == 0 || state.holo_ale_invalidated) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "FLD initialize rad_E_old failed");
  }
}

std::vector<double> radiation_group_bounds(const core::Config& cfg,
                                           const int n_groups) {
  if (static_cast<int>(cfg.radiation.group_bounds_eV.size()) == n_groups + 1) {
    return cfg.radiation.group_bounds_eV;
  }
  const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
  const double Tmin = std::max(range[0], 1.0e-3);
  const double Tmax = std::max(range[1], Tmin * 1.001);
  return Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
}

void evaluate_fld_opacity_and_emission(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const int n_cells,
    const int n_groups,
    const double dt,
    const bool sigma_already_evaluated = false,
    const bool first_outer_iter = true,
    const bool defer_clamp_warning = false,
    int* const deferred_clamp_counts = nullptr) {
  const auto& fld = cfg.radiation.multigroup_diffusion;
  const int total = n_cells * n_groups;
  const auto n_nonvoid_materials = std::count_if(
      cfg.materials.materials.begin(), cfg.materials.materials.end(),
      [](const auto& material) { return !material.is_void; });
  const bool any_table = std::any_of(
      cfg.materials.materials.begin(), cfg.materials.materials.end(),
      [](const auto& material) {
        return !material.is_void &&
               (material.opacity_model == "tmat" ||
                material.opacity_model == "table_nlte");
      });
  bool multimat_sigma_done = false;
  std::uint8_t* d_nlte_mask = nullptr;
  if (n_nonvoid_materials > 1 && any_table) {
    ensure_multimat_opacity_uploaded(cfg, n_groups);
    state.ensure_cell_material_props(cfg);
    auto& cache = multimat_opacity_cache();
    const int opacity_grid = (n_cells + kBlock - 1) / kBlock;
    if (opacity_grid > 0) {
      eval_opacity_multimat_kernel<<<opacity_grid, kBlock>>>(
          state.rho.data(),
          state.Te.data(),
          state.cell_material_index.data(),
          cache.device_descs.data(),
          static_cast<int>(cfg.materials.materials.size()),
          fld.opacity_floor,
          fld.opacity_cap,
          cfg.numerics.floors.Te,
          state.fld_sigma_a.data(),
          state.fld_sigma_pe.data(),
          state.fld_sigma_R.data(),
          n_cells,
          n_groups);
      cuda_check(cudaGetLastError(),
                 "FLD multi-material opacity kernel launch failed");
    }
    const bool any_nlte = std::any_of(
        cache.entries.begin(), cache.entries.end(),
        [&](const auto& entry) {
          const std::size_t m = static_cast<std::size_t>(
              &entry - cache.entries.data());
          const auto& material = cfg.materials.materials[m];
          return !material.is_void && material.opacity_model == "tmat" &&
                 !entry.is_lte;
        });
    if (any_nlte) {
      d_nlte_mask = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "fld1d:multimat_nlte_mask", static_cast<std::size_t>(n_cells)));
      if (opacity_grid > 0) {
        build_nlte_cell_mask_kernel<<<opacity_grid, kBlock>>>(
            state.cell_material_index.data(),
            cache.device_descs.data(),
            static_cast<int>(cfg.materials.materials.size()),
            d_nlte_mask,
            n_cells);
        cuda_check(cudaGetLastError(),
                   "FLD multi-material NLTE mask kernel launch failed");
      }

      const std::size_t scratch_size = static_cast<std::size_t>(total);
      state.fld_nlte_f_work.reset(scratch_size);
      state.fld_nlte_sigma_eff_work.reset(scratch_size);
      state.fld_nlte_sigma_s_eff_work.reset(scratch_size);
      state.fld_nlte_eta_cdf_work.reset(scratch_size);
      state.fld_nlte_lambda_work.reset(scratch_size);
      const double* cv_e_ptr =
          (state.cv_e.size() == static_cast<std::size_t>(n_cells))
              ? state.cv_e.data()
              : nullptr;
      int* pinned_counts = nullptr;
      if (defer_clamp_warning) {
        pinned_counts = deferred_clamp_counts;
        if (pinned_counts == nullptr) {
          pinned_counts = static_cast<int*>(core::host_pinned_scratch_acquire(
              "fld1d:nlte_clamp_counts", 3 * sizeof(int)));
        }
      }
      NlteCoeffsDeviceResult result{};
      for (std::size_t m = 0; m < cache.entries.size(); ++m) {
        if (cache.entries[m].is_lte || cfg.materials.materials[m].is_void ||
            cfg.materials.materials[m].opacity_model != "tmat") {
          continue;
        }
        const auto& material = cfg.materials.materials[m];
        const auto material_result = compute_nlte_coefficients_cuda_with_pe(
            state.rho.data(),
            state.Te.data(),
            state.zbar.data(),
            cv_e_ptr,
            state.cell_is_void.data(),
            state.cell_is_void.size(),
            cache.entries[m].device.view(),
            planck.device_view(),
            n_cells,
            n_groups,
            dt,
            std::max(material.A, 1.0e-12),
            1.0,
            0.0,
            1.0,
            material.lambda_fd_delta_rel,
            material.lambda_fd_abs_min,
            fld.opacity_cap,
            material.cv_e_override,
            cfg.numerics.floors.Te,
            std::max(material.ideal_gas_gamma - 1.0, 1.0e-12),
            false,
            false,
            false,
            state.fld_nlte_f_work.data(),
            state.fld_sigma_a.data(),
            state.fld_sigma_pe.data(),
            state.fld_sigma_R.data(),
            state.fld_nlte_sigma_eff_work.data(),
            state.fld_nlte_sigma_s_eff_work.data(),
            state.fld_nlte_eta_cdf_work.data(),
            state.fld_eta.data(),
            state.fld_nlte_lambda_work.data(),
            0,
            false,
            pinned_counts,
            defer_clamp_warning && !first_outer_iter,
            state.cell_material_index.data(),
            static_cast<int>(m));
        result.negative_alpha_clamp_count +=
            material_result.negative_alpha_clamp_count;
        result.negative_eta_clamp_count +=
            material_result.negative_eta_clamp_count;
        result.nan_inf_count += material_result.nan_inf_count;
      }
      if (!defer_clamp_warning &&
          (result.nan_inf_count != 0 || result.negative_eta_clamp_count != 0)) {
        core::log_warning("FLD NLTE coefficient clamp counts: nan_inf=" +
                          std::to_string(result.nan_inf_count) +
                          ", eta=" +
                          std::to_string(result.negative_eta_clamp_count));
      }
    }
    multimat_sigma_done = true;
  }
  const bool use_nlte =
      !multimat_sigma_done &&
      (mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat");
  if (use_nlte) {
    ensure_nlte_table_uploaded(cfg, mat, n_groups);
    const std::size_t scratch_size = static_cast<std::size_t>(total);
    state.fld_nlte_f_work.reset(scratch_size);
    state.fld_nlte_sigma_eff_work.reset(scratch_size);
    state.fld_nlte_sigma_s_eff_work.reset(scratch_size);
    state.fld_nlte_eta_cdf_work.reset(scratch_size);
    state.fld_nlte_lambda_work.reset(scratch_size);
    auto& cache = nlte_cache();
    const double* cv_e_ptr =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    int* pinned_counts = nullptr;
    if (defer_clamp_warning) {
      pinned_counts = deferred_clamp_counts;
      if (pinned_counts == nullptr) {
        pinned_counts = static_cast<int*>(core::host_pinned_scratch_acquire(
            "fld1d:nlte_clamp_counts", 3 * sizeof(int)));
      }
    }
    const auto result = compute_nlte_coefficients_cuda_with_pe(
        state.rho.data(),
        state.Te.data(),
        state.zbar.data(),
        cv_e_ptr,
        state.cell_is_void.data(),
        state.cell_is_void.size(),
        cache.device.view(),
        planck.device_view(),
        n_cells,
        n_groups,
        dt,
        std::max(mat.A, 1.0e-12),
        1.0,
        0.0,
        1.0,
        mat.lambda_fd_delta_rel,
        mat.lambda_fd_abs_min,
        fld.opacity_cap,
        mat.cv_e_override,
        cfg.numerics.floors.Te,
        std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12),
        false,
        false,
        false,
        state.fld_nlte_f_work.data(),
        state.fld_sigma_a.data(),
        state.fld_sigma_pe.data(),
        state.fld_sigma_R.data(),
        state.fld_nlte_sigma_eff_work.data(),
        state.fld_nlte_sigma_s_eff_work.data(),
        state.fld_nlte_eta_cdf_work.data(),
        state.fld_eta.data(),
        state.fld_nlte_lambda_work.data(),
        0,
        false,
        pinned_counts,
        defer_clamp_warning && !first_outer_iter);
    if (!defer_clamp_warning &&
        (result.nan_inf_count != 0 || result.negative_eta_clamp_count != 0)) {
      core::log_warning("FLD NLTE coefficient clamp counts: nan_inf=" +
                        std::to_string(result.nan_inf_count) +
                        ", eta=" +
                        std::to_string(result.negative_eta_clamp_count));
    }
    return;
  }

  if (!multimat_sigma_done && !sigma_already_evaluated) {
    materials::OpacityEvalView opacity_view{};
    opacity_view.rho = state.rho.data();
    opacity_view.Te = state.Te.data();
    opacity_view.sigma_a = state.fld_sigma_a.data();
    opacity_view.sigma_R = state.fld_sigma_R.data();
    opacity_view.n_cells = n_cells;
    opacity_view.n_groups = n_groups;
    opacity_view.opacity_model =
        (mat.opacity_model == "freq_dep_marshak")
            ? materials::kOpacityModelFreqDepMarshak
            : ((mat.opacity_model == "power_law")
                   ? materials::kOpacityModelPowerLaw
                   : materials::kOpacityModelConstant);
    opacity_view.kappa_planck_const =
        (mat.kappa_planck_override >= 0.0)
            ? mat.kappa_planck_override
            : std::max(0.0, mat.kappa_a_constant);
    opacity_view.kappa_rosseland_const = std::max(0.0, mat.kappa_a_constant);
    opacity_view.kappa_floor = fld.opacity_floor;
    opacity_view.kappa_cap = fld.opacity_cap;
    opacity_view.power_law_kappa0 = mat.opacity_power_law_kappa0_cm2_g;
    opacity_view.power_law_alpha_T = mat.opacity_power_law_alpha_T;
    opacity_view.power_law_lambda_rho = mat.opacity_power_law_lambda_rho;
    opacity_view.power_law_T_ref_eV = mat.opacity_power_law_T_ref_eV;
    opacity_view.power_law_rho_ref = mat.opacity_power_law_rho_ref_g_cc;
    if (n_nonvoid_materials > 1) {
      state.ensure_cell_material_props(cfg);
      opacity_view.kappa_planck_cell = state.kappa_planck_eff.data();
      opacity_view.kappa_rosseland_cell = state.kappa_rosseland_eff.data();
    }
    std::vector<double> bounds;
    if (opacity_view.opacity_model == materials::kOpacityModelFreqDepMarshak) {
      bounds = radiation_group_bounds(cfg, n_groups);
      state.fld_group_bounds_work.reset(bounds.size());
      auto& cache = fld_group_bounds_cache();
      if (!(cache.valid && bounds == cache.last_bounds &&
            state.fld_group_bounds_work.size() == bounds.size())) {
        cuda_check(cudaMemcpy(state.fld_group_bounds_work.data(),
                              bounds.data(),
                              sizeof(double) * bounds.size(),
                              cudaMemcpyHostToDevice),
                   "FLD upload group bounds failed");
        cache.last_bounds = bounds;
        cache.valid = true;
      }
      opacity_view.group_bounds_eV = state.fld_group_bounds_work.data();
    }
    materials::evaluate_opacity_cuda(opacity_view, nullptr);
  }
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    build_eta_from_planck_kernel<<<grid, kBlock>>>(state.Te.data(),
                                                   state.fld_sigma_a.data(),
                                                   planck.device_view(),
                                                   state.fld_eta.data(),
                                                   n_cells,
                                                   n_groups,
                                                   cfg.numerics.floors.Te,
                                                   d_nlte_mask);
    cuda_check(cudaGetLastError(), "FLD eta kernel launch failed");
  }
  if (multimat_sigma_done || mat.opacity_model == "constant" ||
      mat.opacity_model == "power_law") {
    const std::size_t scratch_size = static_cast<std::size_t>(total);
    state.fld_nlte_f_work.reset(scratch_size);
    if (!multimat_sigma_done) {
      cuda_check(cudaMemcpy(state.fld_sigma_pe.data(),
                            state.fld_sigma_a.data(),
                            sizeof(double) * scratch_size,
                            cudaMemcpyDeviceToDevice),
                 "FLD copy constant sigma_pe failed");
    }
    std::uint8_t* d_cell_is_void = nullptr;
    if (state.cell_is_void.size() == static_cast<std::size_t>(n_cells)) {
      const std::size_t void_bytes =
          sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
      d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "fld_1d_gpu:evaluate_fld_opacity_and_emission:d_cell_is_void",
          void_bytes));
      cuda_check(cudaMemcpy(d_cell_is_void,
                            state.cell_is_void.data(),
                            void_bytes,
                            cudaMemcpyHostToDevice),
                 "FLD Fleck copy cell_is_void failed");
    }
    const int fleck_grid = (n_cells + kBlock - 1) / kBlock;
    if (fleck_grid > 0) {
      // Diagnostic-only hook (W-I dt-sensitivity isolation): scales the
      // Fleck z = alpha*beta*c*sigma*dt without changing dt. Default 1.0
      // (bit-identical); never set in production decks.
      const char* fleck_alpha_env = std::getenv("TENRYU_FLD_FLECK_ALPHA");
      const double fleck_alpha =
          (fleck_alpha_env != nullptr && fleck_alpha_env[0] != 0)
              ? std::atof(fleck_alpha_env)
              : 1.0;
      compute_fleck_for_fld_kernel<<<fleck_grid, kBlock>>>(
          state.rho.data(),
          state.Te.data(),
          state.zbar.data(),
          d_cell_is_void,
          state.fld_sigma_a.data(),
          (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                   : nullptr,
          state.fld_nlte_f_work.data(),
          n_cells,
          n_groups,
          dt,
          fleck_alpha,
          mat.cv_e_override,
          state.gamma_eff.data(),
          state.A_eff.data(),
          (mat.hydro_eos_backend != "exact_ideal_gas")
              ? fld_electron_eos_device_view(mat.eos_tables.get())
              : materials::DeviceEOSTableView{},
          (fld.fleck_cv_source == "table") ? 1 : 0,
          cfg.numerics.floors.Te,
          state.rad_E.data(),
          planck.device_view(),
          (fld.fleck_beta == "secant") ? 1
          : (fld.fleck_beta == "guard") ? 2
                                        : 0,
          (fld.fleck_form == "exp_phi1") ? 1 : 0,
          d_nlte_mask);
      cuda_check(cudaGetLastError(), "FLD Fleck kernel launch failed");
      cuda_check(core::debug_kernel_sync(), "FLD Fleck kernel sync failed");
    }
  }
}

void assemble_tridiag(core::State& state,
                      const int n_cells,
                      const int n_groups,
                      const double dt,
                      const bool use_fleck,
                      const int outer_bc,
                      const double* marshak_finc,
                      const double volume_source_rate,
                      const double volume_source_r_max,
                      const double sigma_floor,
                      const int limiter,
                      const int exchange_off = 0) {
  const int total = n_cells * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  const int geom_code = state.mesh.geometry_code;
  if (grid > 0) {
    auto launch_assemble = [&](auto geom_tag) {
      constexpr int GEOM = decltype(geom_tag)::value;
      assemble_fld_tridiag_kernel<GEOM><<<grid, kBlock>>>(
        state.x_r.data(),
        state.vol.data(),
        state.fld_sigma_a.data(),
        use_fleck ? state.fld_sigma_a.data() : nullptr,
        use_fleck ? state.fld_nlte_f_work.data() : nullptr,
        state.fld_eta.data(),
        state.rad_E_old.data(),
        state.rad_E.data(),
        state.fld_sigma_R.data(),
        state.fld_lower.data(),
        state.fld_diag.data(),
        state.fld_upper.data(),
        state.fld_rhs.data(),
        n_cells,
        n_groups,
        dt,
        outer_bc,
        marshak_finc,
        volume_source_rate,
        volume_source_r_max,
        sigma_floor,
        limiter,
        exchange_off);
    };
    switch (geom_code) {
      case 1:
        launch_assemble(std::integral_constant<int, 1>{});
        break;
      case 2:
        launch_assemble(std::integral_constant<int, 2>{});
        break;
      default:
        launch_assemble(std::integral_constant<int, 0>{});
        break;
    }
    cuda_check(cudaGetLastError(), "FLD tridiag assembly launch failed");
  }
}

void solve_tridiag(core::State& state, const int n_cells, const int n_groups) {
  auto& cache = fld_cusparse_cache();
  if (cache.handle == nullptr) {
    cusparse_check(cusparseCreate(&cache.handle), "FLD cusparseCreate failed");
  }
  static const bool use_gtsv_loop = [] {
    const char* value = std::getenv("TENRYU_FLD_GTSV_LOOP");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  static const bool solve_check = [] {
    const char* value = std::getenv("TENRYU_FLD_SOLVE_CHECK");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  struct SolveCheckSnapshots {
    std::vector<double> lower;
    std::vector<double> diag;
    std::vector<double> upper;
    std::vector<double> rhs;
  };
  std::unique_ptr<SolveCheckSnapshots> solve_check_snapshots;
  std::size_t buffer_size = 0U;
  if (use_gtsv_loop) {
    cusparse_check(cusparseDgtsv2_bufferSizeExt(cache.handle,
                                                n_cells,
                                                1,
                                                state.fld_lower.data(),
                                                state.fld_diag.data(),
                                                state.fld_upper.data(),
                                                state.fld_rhs.data(),
                                                n_cells,
                                                &buffer_size),
                  "FLD cuSPARSE buffer size failed");
  } else {
    cusparse_check(cusparseDgtsv2StridedBatch_bufferSizeExt(
                      cache.handle,
                      n_cells,
                      state.fld_lower.data(),
                      state.fld_diag.data(),
                      state.fld_upper.data(),
                      state.fld_rhs.data(),
                      n_groups,
                      n_cells,
                      &buffer_size),
                  "FLD cuSPARSE buffer size failed");
  }
  const std::size_t buffer_doubles =
      (buffer_size + sizeof(double) - 1U) / sizeof(double);
  state.fld_cusparse_buffer.reset(buffer_doubles);
  if (solve_check) {
    const std::size_t solve_size =
        static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(n_cells);
    solve_check_snapshots = std::make_unique<SolveCheckSnapshots>();
    solve_check_snapshots->lower.resize(solve_size);
    solve_check_snapshots->diag.resize(solve_size);
    solve_check_snapshots->upper.resize(solve_size);
    solve_check_snapshots->rhs.resize(solve_size);
    const std::size_t solve_bytes = solve_size * sizeof(double);
    cuda_check(cudaMemcpy(solve_check_snapshots->lower.data(),
                          state.fld_lower.data(),
                          solve_bytes,
                          cudaMemcpyDeviceToHost),
               "FLD solve check lower snapshot failed");
    cuda_check(cudaMemcpy(solve_check_snapshots->diag.data(),
                          state.fld_diag.data(),
                          solve_bytes,
                          cudaMemcpyDeviceToHost),
               "FLD solve check diagonal snapshot failed");
    cuda_check(cudaMemcpy(solve_check_snapshots->upper.data(),
                          state.fld_upper.data(),
                          solve_bytes,
                          cudaMemcpyDeviceToHost),
               "FLD solve check upper snapshot failed");
    cuda_check(cudaMemcpy(solve_check_snapshots->rhs.data(),
                          state.fld_rhs.data(),
                          solve_bytes,
                          cudaMemcpyDeviceToHost),
               "FLD solve check RHS snapshot failed");
  }
  if (use_gtsv_loop) {
    for (int g = 0; g < n_groups; ++g) {
      cusparse_check(
          cusparseDgtsv2(cache.handle,
                         n_cells,
                         1,
                         state.fld_lower.data() + g * n_cells,
                         state.fld_diag.data() + g * n_cells,
                         state.fld_upper.data() + g * n_cells,
                         state.fld_rhs.data() + g * n_cells,
                         n_cells,
                         state.fld_cusparse_buffer.data()),
          "FLD cuSPARSE tridiagonal solve failed");
    }
  } else {
    cusparse_check(cusparseDgtsv2StridedBatch(cache.handle,
                                              n_cells,
                                              state.fld_lower.data(),
                                              state.fld_diag.data(),
                                              state.fld_upper.data(),
                                              state.fld_rhs.data(),
                                              n_groups,
                                              n_cells,
                                              state.fld_cusparse_buffer.data()),
                  "FLD cuSPARSE tridiagonal solve failed");
  }
  if (solve_check) {
    const std::size_t solve_size = solve_check_snapshots->rhs.size();
    std::vector<double> solution(solve_size);
    cuda_check(cudaMemcpy(solution.data(),
                          state.fld_rhs.data(),
                          solve_size * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "FLD solve check solution copy failed");
    std::size_t hit_count = 0U;
    auto fmt_d = [](double v){ char b[32]; std::snprintf(b, sizeof(b), "%.17g", v); return std::string(b); };
    for (int g = 0; g < n_groups; ++g) {
      for (int c = 0; c < n_cells; ++c) {
        const std::size_t i = static_cast<std::size_t>(g) *
                                  static_cast<std::size_t>(n_cells) +
                              static_cast<std::size_t>(c);
        const double x = solution[i];
        if (!std::isfinite(x) || std::fabs(x) > 1.0e30) {
          ++hit_count;
          if (hit_count <= 8U) {
            const double diag_left =
                (c > 0) ? solve_check_snapshots->diag[i - 1U]
                        : std::numeric_limits<double>::quiet_NaN();
            const double diag_right =
                (c + 1 < n_cells)
                    ? solve_check_snapshots->diag[i + 1U]
                    : std::numeric_limits<double>::quiet_NaN();
            core::log_error(
                "FLD solve check: g=" + std::to_string(g) +
                " c=" + std::to_string(c) + " x=" + fmt_d(x) +
                " | inputs dl=" + fmt_d(solve_check_snapshots->lower[i]) +
                " d=" + fmt_d(solve_check_snapshots->diag[i]) +
                " du=" + fmt_d(solve_check_snapshots->upper[i]) +
                " rhs=" + fmt_d(solve_check_snapshots->rhs[i]) +
                " | c-1 d=" + fmt_d(diag_left) +
                " c+1 d=" + fmt_d(diag_right));
          }
        }
      }
    }
    if (hit_count > 0U) {
      core::log_error("FLD solve check: total_hits=" + std::to_string(hit_count) +
                      " n_cells=" + std::to_string(n_cells));
      const std::vector<double>* const input_arrays[] = {
          &solve_check_snapshots->lower,
          &solve_check_snapshots->diag,
          &solve_check_snapshots->upper,
          &solve_check_snapshots->rhs};
      const char* const input_array_names[] = {"lower", "diag", "upper", "rhs"};
      std::size_t input_bad_count = 0U;
      for (int g = 0; g < n_groups; ++g) {
        for (int c = 0; c < n_cells; ++c) {
          const std::size_t i = static_cast<std::size_t>(g) *
                                    static_cast<std::size_t>(n_cells) +
                                static_cast<std::size_t>(c);
          const double diag_left =
              (c > 0) ? solve_check_snapshots->diag[i - 1U]
                      : std::numeric_limits<double>::quiet_NaN();
          const double diag_right =
              (c + 1 < n_cells)
                  ? solve_check_snapshots->diag[i + 1U]
                  : std::numeric_limits<double>::quiet_NaN();
          for (std::size_t arr = 0U; arr < 4U; ++arr) {
            const double v = (*input_arrays[arr])[i];
            if (!std::isfinite(v) || std::fabs(v) > 1.0e30) {
              ++input_bad_count;
              if (input_bad_count <= 8U) {
                core::log_error(
                    "FLD solve check INPUT: g=" + std::to_string(g) +
                    " c=" + std::to_string(c) +
                    " arr=" + input_array_names[arr] + " v=" + fmt_d(v) +
                    " | d[c-1]=" + fmt_d(diag_left) +
                    " d[c]=" + fmt_d(solve_check_snapshots->diag[i]) +
                    " d[c+1]=" + fmt_d(diag_right) +
                    " rhs[c]=" + fmt_d(solve_check_snapshots->rhs[i]));
              }
            }
          }
        }
      }
      core::log_error("FLD solve check INPUT: total_bad=" +
                      std::to_string(input_bad_count));
      TENRYU_ASSERT(false, "FLD solve produced non-finite/huge solution");
    }
  }
}

double copy_scalar_from_device(core::State& state) {
  double value = 0.0;
  cuda_check(cudaMemcpy(&value,
                        state.fld_reduction_work.data(),
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "FLD copy reduction scalar failed");
  return value;
}

void zero_reduction_scalar(core::State& state) {
  cuda_check(cudaMemset(state.fld_reduction_work.data(), 0, sizeof(double)),
             "FLD zero reduction scalar failed");
}

struct FldOuterCheck {
  double residual = std::numeric_limits<double>::infinity();
  bool converged = false;
};

void launch_outer_check(core::State& state, const int n_cells,
                        const double outer_tol, double* const d_pack) {
  reduce_outer_check_kernel<<<1, kBlock>>>(state.fld_delta_T.data(), n_cells,
                                           outer_tol, d_pack);
  cuda_check(cudaGetLastError(), "FLD outer predicate launch failed");
}

FldOuterCheck reduce_outer_check(core::State& state, const int n_cells,
                                 const double outer_tol) {
  double* d_pack = static_cast<double*>(core::device_scratch_acquire(
      "fld_1d:outer_check_pack", 2 * sizeof(double)));
  launch_outer_check(state, n_cells, outer_tol, d_pack);
  double h_pack[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(h_pack, d_pack, sizeof(h_pack), cudaMemcpyDeviceToHost),
             "FLD outer check read failed");
  FldOuterCheck out;
  out.residual = h_pack[0];
  out.converged = h_pack[1] != 0.0;
  return out;
}

double compute_escaped_energy(core::State& state,
                              const int n_cells,
                              const int n_groups,
                              const double dt,
                              const int outer_bc) {
  zero_reduction_scalar(state);
  const int grid = (n_groups + kBlock - 1) / kBlock;
  const int geom_code = state.mesh.geometry_code;
  if (grid > 0) {
    auto launch_escaped = [&](auto geom_tag) {
      constexpr int GEOM = decltype(geom_tag)::value;
      escaped_energy_kernel<GEOM><<<grid, kBlock>>>(
          state.x_r.data(),
          state.rad_E.data(),
          state.fld_reduction_work.data(),
          n_cells,
          n_groups,
          dt,
          outer_bc);
    };
    switch (geom_code) {
      case 1:
        launch_escaped(std::integral_constant<int, 1>{});
        break;
      case 2:
        launch_escaped(std::integral_constant<int, 2>{});
        break;
      default:
        launch_escaped(std::integral_constant<int, 0>{});
        break;
    }
    cuda_check(cudaGetLastError(), "FLD escaped energy launch failed");
  }
  return copy_scalar_from_device(state);
}

double compute_volume_source_energy(core::State& state,
                                    const int n_cells,
                                    const double dt,
                                    const double rate,
                                    const double r_max) {
  if (n_cells <= 0 || rate <= 0.0 || r_max <= 0.0) {
    return 0.0;
  }
  zero_reduction_scalar(state);
  const int grid = (n_cells + kBlock - 1) / kBlock;
  volume_source_energy_kernel<<<grid, kBlock>>>(state.x_r.data(),
                                                state.vol.data(),
                                                state.fld_reduction_work.data(),
                                                n_cells,
                                                dt,
                                                rate,
                                                r_max);
  cuda_check(cudaGetLastError(), "FLD volume source energy launch failed");
  return copy_scalar_from_device(state);
}

void copy_rad_E_to_old(core::State& state, const int n_cells, const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes > 0U) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "FLD update rad_E_old failed");
  }
}

}  // namespace

void advance_radiation_step_fld_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const double dt) {
  TENRYU_ASSERT(cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion,
                "advance_radiation_step_fld_1d requires multigroup_diffusion mode");
  TENRYU_ASSERT(state.mesh.dim == 1 || cfg.main.dimension == "1D_SPH",
                "advance_radiation_step_fld_1d requires 1D_SPH state");
  TENRYU_ASSERT(dt > 0.0, "advance_radiation_step_fld_1d requires dt > 0");
  TENRYU_ASSERT(!state.rho.empty(), "advance_radiation_step_fld_1d requires cells");
  TENRYU_ASSERT(!state.x_r.empty(),
                "advance_radiation_step_fld_1d requires 1D node radii");
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_cells + 1),
                "FLD requires x_r size n_cells+1");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "FLD requires Planck table group count to match Radiation.groups");

  ensure_state_buffers(state, n_cells, n_groups);
  // Fleck-blend reference field: snapshot at the START of THIS solve.
  // (Historic end-of-previous-solve snapshot assumed nothing modifies rad_E
  // between solves — the gamma_r hydro-half payments broke that contract:
  // a stale-high E_old made the blend inject ~(1-f) c sigma (E_old-E_now)
  // per step. Start-of-solve copy is byte-identical for every no-coupling
  // config and correct under coupling.)
  copy_rad_E_to_old(state, n_cells, n_groups);
  // Matter coupling consumes the shared per-cell effective properties (multi-material fix)
  // properties (radiation can run with hydro disabled, so ensure here too).
  state.ensure_cell_material_props(cfg);

  const int cell_grid = (n_cells + kBlock - 1) / kBlock;
  if (cell_grid > 0) {
    snapshot_Te_kernel<<<cell_grid, kBlock>>>(state.Te.data(),
                                              state.fld_Te_old.data(),
                                              n_cells);
    cuda_check(cudaGetLastError(), "FLD snapshot Te launch failed");
  }

  state.fld_converged = false;
  state.fld_outer_residual = std::numeric_limits<double>::infinity();
  state.fld_outer_iterations = 0;
  state.fld_escaped_step = 0.0;
  state.fld_marshak_in_step = 0.0;
  state.fld_volume_source_in_step = 0.0;

  const auto& fld = cfg.radiation.multigroup_diffusion;
  int max_iter = std::max(fld.max_outer_iterations, 1);
  // Diagnostic override (W-I outer-convergence audit); never set in
  // production decks.
  const char* max_outer_env = std::getenv("TENRYU_FLD_MAX_OUTER_OVERRIDE");
  if (max_outer_env != nullptr && max_outer_env[0] != 0) {
    const int v = std::atoi(max_outer_env);
    if (v > 0) {
      max_iter = v;
    }
  }
  const bool use_nlte = use_fld_fleck(mat);
  // W-I AFI mode: the Fleck f-blend is dropped at its two CONSUMPTION
  // points (assembly pseudo-scattering and the matter-side blend) so the
  // outer iteration converges the fully implicit emission (Larsen, Kumar
  // & Morel, JCP 238 (2013)). The f field itself may still be computed
  // upstream (constant-opacity path) — it is simply not consumed. The
  // NLTE/tmat opacity plumbing (sigma_pe) is physics and stays.
  const bool use_fleck_blend = use_nlte && (fld.fleck_mode != "afi");

  // 1D boundary handling (NUMERICS §6.7 1D BC): inner r=0 is always
  // reflecting; the outer face honors multigroup_diffusion.boundary.outer_r.
  const int outer_bc = parse_fld_1d_outer_bc(fld.boundary.outer_r);
  core::DeviceArray<double> marshak_finc;
  double marshak_in_flux_total = 0.0;  // sum_g F_inc,g [erg/cm^2/s]
  if (outer_bc == kFld1dOuterMarshak) {
    // Indirect-drive hohlraum-style boundary: constant marshak_Tr_eV wins when
    // set; otherwise the deck callable frozen at init (state.marshak_Tr_1d,
    // no runtime Python) is evaluated at solve entry (state.t) — same
    // precedence/convention as the IMC emitter (source.cu emit_marshak).
    double T_r_eV = cfg.radiation.boundary.marshak_Tr_eV;
    if (!(T_r_eV > 0.0) && state.marshak_Tr_1d.has_value()) {
      T_r_eV = state.marshak_Tr_1d->eval(state.t);
    }
    double const_flux = fld.marshak.flux_erg_per_cm2_s;
    if (fld.marshak.flux_pulse_duration_s >= 0.0 &&
        state.t >= fld.marshak.flux_pulse_duration_s) {
      const_flux = 0.0;  // pulsed grey drive expired (same rule as 2D_RZ)
    }
    // Namelist validation guarantees exactly one of {marshak_Tr_eV > 0,
    // flux_erg_per_cm2_s > 0}; const_flux may legitimately be 0 here after a
    // pulsed grey drive expires.
    marshak_finc = core::DeviceArray<double>(static_cast<std::size_t>(n_groups));
    const int finc_grid = (n_groups + kBlock - 1) / kBlock;
    compute_marshak_finc_kernel<<<finc_grid, kBlock>>>(planck.device_view(),
                                                       T_r_eV,
                                                       const_flux,
                                                       marshak_finc.data(),
                                                       n_groups);
    cuda_check(cudaGetLastError(), "FLD marshak finc launch failed");
    if (T_r_eV > 0.0) {
      const double T2 = T_r_eV * T_r_eV;
      // Renormalized b_g sums to 1, so the group sum is exactly (c/4) a T^4.
      marshak_in_flux_total =
          0.25 * core::constants::c_light * core::constants::a_eV * T2 * T2;
    } else {
      marshak_in_flux_total = std::max(const_flux, 0.0);
    }
  }
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);

  static const bool rad_ebal_enabled = [] {
    const char* s = std::getenv("TENRYU_HYDRO_EBAL");
    return s != nullptr && std::atoi(s) != 0;
  }();
  const auto rad_ebal_matter = [&]() -> double {
    if (!rad_ebal_enabled) {
      return 0.0;
    }
    const std::size_t n = state.ee.size();
    std::vector<double> ee_h(n, 0.0), rho_h(n, 0.0), vol_h(n, 0.0);
    state.ee.copy_to_host(ee_h.data());
    state.rho.copy_to_host(rho_h.data());
    state.vol.copy_to_host(vol_h.data());
    double u = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      u += ee_h[i] * rho_h[i] * vol_h[i];
    }
    return u;
  };
  const auto rad_ebal_field = [&]() -> double {
    if (!rad_ebal_enabled) {
      return 0.0;
    }
    const std::size_t ncg = state.rad_E.size();
    const std::size_t nc = state.vol.size();
    if (ncg == 0 || nc == 0) {
      return 0.0;
    }
    std::vector<double> E_h(ncg, 0.0), vol_h(nc, 0.0);
    state.rad_E.copy_to_host(E_h.data());
    state.vol.copy_to_host(vol_h.data());
    const std::size_t ng = ncg / nc;
    double u = 0.0;
    for (std::size_t c = 0; c < nc; ++c) {
      double s = 0.0;
      for (std::size_t g = 0; g < ng; ++g) {
        s += E_h[c * ng + g];
      }
      u += s * vol_h[c];
    }
    return u;
  };
  double rad_ebal_Um0 = rad_ebal_matter();
  double rad_ebal_Ur0 = rad_ebal_field();
  double rad_ebal_Um1 = 0.0;
  double rad_ebal_Ur1 = 0.0;
  int* const d_publish_nonfinite_count = static_cast<int*>(
      core::device_scratch_acquire("fld1d:publish_nonfinite_count", sizeof(int)));
  cuda_check(cudaMemset(d_publish_nonfinite_count, 0, sizeof(int)),
             "FLD publish non-finite counter reset failed");

  if (fld.source_integrator == "exp_rosenbrock") {
    // rung-2 Lie split: exact frozen source transfer, then a pure diffusion
    // solve with the exchange stripped; no outer Picard (design doc §3).
    evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells,
                                      n_groups, dt);
    std::uint8_t* d_cell_is_void = nullptr;
    if (state.cell_is_void.size() == static_cast<std::size_t>(n_cells)) {
      const std::size_t void_bytes =
          sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
      d_cell_is_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
          "fld_1d_gpu:evaluate_fld_opacity_and_emission:d_cell_is_void",
          void_bytes));
      cuda_check(cudaMemcpy(d_cell_is_void,
                            state.cell_is_void.data(),
                            void_bytes,
                            cudaMemcpyHostToDevice),
                 "FLD Fleck copy cell_is_void failed");
    }
    const int cell_grid = (n_cells + kBlock - 1) / kBlock;
    if (cell_grid > 0) {
      if (n_groups == 1) {
        const bool eos_tail =
            (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
        if (eos_tail) {
          exp_source_transfer_kernel<true><<<cell_grid, kBlock>>>(
              state.rho.data(),
              state.vol.data(),
              state.zbar.data(),
              (state.cv_e.size() == static_cast<std::size_t>(n_cells))
                  ? state.cv_e.data()
                  : nullptr,
              d_cell_is_void,
              state.fld_sigma_a.data(),
              state.rad_E.data(),
              state.Te.data(),
              state.ee.data(),
              state.Pe.data(),
              state.rad_dep.data(),
              state.rad_emit.data(),
              state.fld_delta_T.data(),
              n_cells,
              dt,
              mat.cv_e_override,
              state.gamma_eff.data(),
              state.A_eff.data(),
              (mat.hydro_eos_backend != "exact_ideal_gas")
                  ? fld_electron_eos_device_view(mat.eos_tables.get())
                  : materials::DeviceEOSTableView{},
              cfg.numerics.floors.Te);
        } else {
          exp_source_transfer_kernel<false><<<cell_grid, kBlock>>>(
              state.rho.data(),
              state.vol.data(),
              state.zbar.data(),
              (state.cv_e.size() == static_cast<std::size_t>(n_cells))
                  ? state.cv_e.data()
                  : nullptr,
              d_cell_is_void,
              state.fld_sigma_a.data(),
              state.rad_E.data(),
              state.Te.data(),
              state.ee.data(),
              state.Pe.data(),
              state.rad_dep.data(),
              state.rad_emit.data(),
              state.fld_delta_T.data(),
              n_cells,
              dt,
              mat.cv_e_override,
              state.gamma_eff.data(),
              state.A_eff.data(),
              (mat.hydro_eos_backend != "exact_ideal_gas")
                  ? fld_electron_eos_device_view(mat.eos_tables.get())
                  : materials::DeviceEOSTableView{},
              cfg.numerics.floors.Te);
        }
        cuda_check(cudaGetLastError(), "FLD exp source kernel launch failed");
      } else {
        static int* d_exp_mg_reject = nullptr;
        static int* d_exp_mg_reject_info = nullptr;
        static double* d_exp_mg_reject_vals = nullptr;
        if (d_exp_mg_reject == nullptr) {
          cuda_check(cudaMalloc(&d_exp_mg_reject, sizeof(int)),
                     "FLD exp mg reject counter alloc failed");
        }
        if (d_exp_mg_reject_info == nullptr) {
          cuda_check(cudaMalloc(&d_exp_mg_reject_info, 25 * sizeof(int)),
                     "FLD exp mg reject info alloc failed");
        }
        if (d_exp_mg_reject_vals == nullptr) {
          cuda_check(cudaMalloc(&d_exp_mg_reject_vals, 24 * sizeof(double)),
                     "FLD exp mg reject vals alloc failed");
        }
        cuda_check(cudaMemset(d_exp_mg_reject, 0, sizeof(int)),
                   "FLD exp mg reject counter reset failed");
        cuda_check(cudaMemset(d_exp_mg_reject_info, 0, 25 * sizeof(int)),
                   "FLD exp mg reject info reset failed");
        cuda_check(cudaMemset(d_exp_mg_reject_vals, 0, 24 * sizeof(double)),
                   "FLD exp mg reject vals reset failed");
        const Phi1PoleSet poles = make_phi1_pole_set();
        const bool eos_tail =
            (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
        if (eos_tail) {
          exp_mg_source_transfer_kernel<true><<<cell_grid, kBlock>>>(
              state.rho.data(),
              state.vol.data(),
              state.zbar.data(),
              (state.cv_e.size() == static_cast<std::size_t>(n_cells))
                  ? state.cv_e.data()
                  : nullptr,
              d_cell_is_void,
              state.fld_sigma_a.data(),
              state.rad_E.data(),
              state.Te.data(),
              state.ee.data(),
              state.Pe.data(),
              state.rad_dep.data(),
              state.rad_emit.data(),
              state.fld_delta_T.data(),
              d_exp_mg_reject,
              d_exp_mg_reject_info,
              d_exp_mg_reject_vals,
              n_cells,
              n_groups,
              dt,
              mat.cv_e_override,
              state.gamma_eff.data(),
              state.A_eff.data(),
              (mat.hydro_eos_backend != "exact_ideal_gas")
                  ? fld_electron_eos_device_view(mat.eos_tables.get())
                  : materials::DeviceEOSTableView{},
              planck.device_view(),
              cfg.numerics.floors.Te,
              poles);
        } else {
          exp_mg_source_transfer_kernel<false><<<cell_grid, kBlock>>>(
              state.rho.data(),
              state.vol.data(),
              state.zbar.data(),
              (state.cv_e.size() == static_cast<std::size_t>(n_cells))
                  ? state.cv_e.data()
                  : nullptr,
              d_cell_is_void,
              state.fld_sigma_a.data(),
              state.rad_E.data(),
              state.Te.data(),
              state.ee.data(),
              state.Pe.data(),
              state.rad_dep.data(),
              state.rad_emit.data(),
              state.fld_delta_T.data(),
              d_exp_mg_reject,
              d_exp_mg_reject_info,
              d_exp_mg_reject_vals,
              n_cells,
              n_groups,
              dt,
              mat.cv_e_override,
              state.gamma_eff.data(),
              state.A_eff.data(),
              (mat.hydro_eos_backend != "exact_ideal_gas")
                  ? fld_electron_eos_device_view(mat.eos_tables.get())
                  : materials::DeviceEOSTableView{},
              planck.device_view(),
              cfg.numerics.floors.Te,
              poles);
        }
        cuda_check(cudaGetLastError(), "FLD exp mg source kernel launch failed");
        int rejects = 0;
        cuda_check(cudaMemcpy(&rejects, d_exp_mg_reject, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "FLD exp mg reject counter D2H failed");
        if (rejects > 0) {
          int info[25] = {0};
          double vals[24] = {0.0};
          cuda_check(cudaMemcpy(info, d_exp_mg_reject_info, sizeof(info),
                                cudaMemcpyDeviceToHost),
                     "FLD exp mg reject info D2H failed");
          cuda_check(cudaMemcpy(vals, d_exp_mg_reject_vals, sizeof(vals),
                                cudaMemcpyDeviceToHost),
                     "FLD exp mg reject vals D2H failed");
          std::string msg =
              "FLD exp_rosenbrock mg: rejected transfer in " +
              std::to_string(rejects) +
              " cells this step (no exchange applied there)";
          const int n_slots = std::min(info[0], 8);
          for (int s = 0; s < n_slots; ++s) {
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          " [cell=%d reason=%d g=%d Te=%.3e E_g=%.3e dE=%.3e]",
                          info[1 + 3 * s], info[2 + 3 * s], info[3 + 3 * s],
                          vals[3 * s + 2], vals[3 * s], vals[3 * s + 1]);
            msg += buf;
          }
          core::log_warning(msg);
        }
      }
    }
    // Diffusion substep initial condition = the post-source field.
    cuda_check(cudaMemcpy(state.rad_E_old.data(), state.rad_E.data(), bytes,
                          cudaMemcpyDeviceToDevice),
               "FLD exp source E copy failed");
    rad_ebal_Um1 = rad_ebal_matter();
    rad_ebal_Ur1 = rad_ebal_field();
    assemble_tridiag(state, n_cells, n_groups, dt, /*use_fleck=*/false,
                     outer_bc,
                     marshak_finc.empty() ? nullptr : marshak_finc.data(),
                     cfg.radiation.volume_source_rate,
                     cfg.radiation.volume_source_x_max,
                     cfg.radiation.multigroup_diffusion.opacity_floor,
                     limiter_id(cfg.radiation.multigroup_diffusion.flux_limiter),
                     /*exchange_off=*/1);
    solve_tridiag(state, n_cells, n_groups);
    const int total = n_cells * n_groups;
    const int total_grid = (total + kBlock - 1) / kBlock;
    if (total_grid > 0) {
      publish_solution_kernel<<<total_grid, kBlock>>>(state.fld_rhs.data(),
                                                      state.rad_E.data(),
                                                      n_cells,
                                                      n_groups,
                                                      d_publish_nonfinite_count);
      cuda_check(cudaGetLastError(), "FLD publish solution launch failed");
    }
    state.fld_outer_residual = 0.0;
    state.fld_outer_iterations = 1;
    state.fld_converged = true;
  } else {
    const bool sigma_iteration_invariant = mat.opacity_model == "constant";
    if (sigma_iteration_invariant) {
      evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells,
                                        n_groups, dt);
    }
    const bool pipeline_opacity_model =
        mat.opacity_model == "constant" || mat.opacity_model == "table_nlte" ||
        mat.opacity_model == "tmat";
    const char* outer_no_pipeline_env =
        std::getenv("TENRYU_FLD_OUTER_NO_PIPELINE");
    const bool outer_pipeline_disabled =
        outer_no_pipeline_env != nullptr && std::atoi(outer_no_pipeline_env) != 0;
    const bool use_outer_pipeline =
        pipeline_opacity_model && !outer_pipeline_disabled;
    const bool has_nlte_clamp_counts =
        mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";

    const auto launch_outer_iteration = [&](const int iter,
                                            int* const deferred_clamp_counts) {
      evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells,
                                        n_groups, dt,
                                        sigma_iteration_invariant,
                                        /*first_outer_iter=*/iter == 0,
                                        /*defer_clamp_warning=*/true,
                                        deferred_clamp_counts);
      assemble_tridiag(state, n_cells, n_groups, dt, use_fleck_blend, outer_bc,
                       marshak_finc.empty() ? nullptr : marshak_finc.data(),
                       cfg.radiation.volume_source_rate,
                       cfg.radiation.volume_source_x_max,
                       cfg.radiation.multigroup_diffusion.opacity_floor,
                       limiter_id(cfg.radiation.multigroup_diffusion.flux_limiter));
      solve_tridiag(state, n_cells, n_groups);
      const int total = n_cells * n_groups;
      const int total_grid = (total + kBlock - 1) / kBlock;
      if (total_grid > 0) {
        publish_solution_kernel<<<total_grid, kBlock>>>(state.fld_rhs.data(),
                                                        state.rad_E.data(),
                                                        n_cells,
                                                        n_groups,
                                                        d_publish_nonfinite_count);
        cuda_check(cudaGetLastError(), "FLD publish solution launch failed");
      }
      if (n_cells > 0) {
        const std::size_t shared_doubles = 2U * static_cast<std::size_t>(n_groups);
        const std::size_t shared_bytes = shared_doubles * sizeof(double);
        TENRYU_ASSERT(shared_bytes <= 32U * 1024U,
                      "FLD update_matter shared memory exceeds 32 KiB safety cap; "
                      "reduce n_groups");
        const bool eos_tail =
            (cfg.numerics.hydro.eos_closure_mode == "energy_authoritative");
        if (eos_tail) {
          update_matter_kernel<true><<<n_cells, 32, shared_bytes>>>(
              state.rho.data(),
              state.vol.data(),
              state.zbar.data(),
              (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                       : nullptr,
              state.fld_Te_old.data(),
              state.fld_sigma_a.data(),
              use_nlte ? state.fld_sigma_pe.data() : nullptr,
              state.rad_E.data(),
              use_fleck_blend ? state.fld_nlte_f_work.data() : nullptr,
              state.rad_E_old.data(),
              planck.device_view(),
              (mat.hydro_eos_backend != "exact_ideal_gas")
                  ? fld_electron_eos_device_view(mat.eos_tables.get())
                  : materials::DeviceEOSTableView{},
              state.Te.data(),
              state.ee.data(),
              state.Pe.data(),
              state.rad_dep.data(),
              state.rad_emit.data(),
              state.fld_delta_T.data(),
              n_cells,
              n_groups,
              dt,
              state.A_eff.data(),
              state.gamma_eff.data(),
              mat.cv_e_override,
              cfg.numerics.floors.Te,
              state.cv_e.size() == static_cast<std::size_t>(n_cells) ? 1 : 0,
              std::max(fld.max_outer_iterations, 1),
              fld.outer_tol);
        } else {
          update_matter_kernel<false><<<n_cells, 32, shared_bytes>>>(
              state.rho.data(),
              state.vol.data(),
              state.zbar.data(),
              (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                       : nullptr,
              state.fld_Te_old.data(),
              state.fld_sigma_a.data(),
              use_nlte ? state.fld_sigma_pe.data() : nullptr,
              state.rad_E.data(),
              use_fleck_blend ? state.fld_nlte_f_work.data() : nullptr,
              state.rad_E_old.data(),
              planck.device_view(),
              (mat.hydro_eos_backend != "exact_ideal_gas")
                  ? fld_electron_eos_device_view(mat.eos_tables.get())
                  : materials::DeviceEOSTableView{},
              state.Te.data(),
              state.ee.data(),
              state.Pe.data(),
              state.rad_dep.data(),
              state.rad_emit.data(),
              state.fld_delta_T.data(),
              n_cells,
              n_groups,
              dt,
              state.A_eff.data(),
              state.gamma_eff.data(),
              mat.cv_e_override,
              cfg.numerics.floors.Te,
              state.cv_e.size() == static_cast<std::size_t>(n_cells) ? 1 : 0,
              std::max(fld.max_outer_iterations, 1),
              fld.outer_tol);
        }
        cuda_check(cudaGetLastError(), "FLD matter update launch failed");
      }
    };

    const auto emit_nlte_clamp_warning = [](const int* const pinned_counts) {
      if (pinned_counts[2] != 0 || pinned_counts[1] != 0) {
        core::log_warning("FLD NLTE coefficient clamp counts: nan_inf=" +
                          std::to_string(pinned_counts[2]) + ", eta=" +
                          std::to_string(pinned_counts[1]));
      }
    };

    if (!use_outer_pipeline) {
      for (int iter = 0; iter < max_iter; ++iter) {
        launch_outer_iteration(iter, nullptr);
        const FldOuterCheck check =
            reduce_outer_check(state, n_cells, fld.outer_tol);
        state.fld_outer_residual = check.residual;
        state.fld_outer_iterations = iter + 1;
        if (has_nlte_clamp_counts) {
          int* pinned_counts = static_cast<int*>(core::host_pinned_scratch_acquire(
              "fld1d:nlte_clamp_counts", 3 * sizeof(int)));
          emit_nlte_clamp_warning(pinned_counts);
        }
        if (check.converged) {
          state.fld_converged = true;
          break;
        }
      }
    } else {
      constexpr int kPipelineSlots = 2;
      cudaStream_t const outer_stream = nullptr;
      double* d_outer_pack = static_cast<double*>(core::device_scratch_acquire(
          "fld_1d:outer_check_pack", 2 * sizeof(double)));
      void* d_iteration_void = nullptr;
      void* d_nlte_clamp_counts = nullptr;
      if (has_nlte_clamp_counts) {
        d_iteration_void = core::device_scratch_acquire(
            "nlte:cell_is_void",
            sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells));
        d_nlte_clamp_counts = core::device_scratch_acquire(
            "nlte:clamp_counts", 3 * sizeof(int));
      } else {
        d_iteration_void = core::device_scratch_acquire(
            "fld_1d_gpu:evaluate_fld_opacity_and_emission:d_cell_is_void",
            sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells));
      }
      double* pinned_outer_packs = static_cast<double*>(
          core::host_pinned_scratch_acquire(
              "fld1d:outer_pipeline_check_packs",
              kPipelineSlots * 2 * sizeof(double)));
      int* pinned_clamp_counts = nullptr;
      if (has_nlte_clamp_counts) {
        pinned_clamp_counts = static_cast<int*>(core::host_pinned_scratch_acquire(
            "fld1d:outer_pipeline_clamp_counts",
            kPipelineSlots * 3 * sizeof(int)));
      }
      cudaEvent_t pull_events[kPipelineSlots] = {nullptr, nullptr};
      for (int slot = 0; slot < kPipelineSlots; ++slot) {
        cuda_check(cudaEventCreateWithFlags(&pull_events[slot],
                                            cudaEventDisableTiming),
                   "FLD outer pipeline event creation failed");
      }

      const auto enqueue_outer_pull = [&](const int slot) {
        cuda_check(cudaMemcpyAsync(pinned_outer_packs + 2 * slot,
                                   d_outer_pack,
                                   2 * sizeof(double),
                                   cudaMemcpyDeviceToHost,
                                   outer_stream),
                   "FLD outer pipeline check read failed");
        cuda_check(cudaEventRecord(pull_events[slot], outer_stream),
                   "FLD outer pipeline event record failed");
      };

      launch_outer_iteration(
          0, has_nlte_clamp_counts ? pinned_clamp_counts : nullptr);
      launch_outer_check(state, n_cells, fld.outer_tol, d_outer_pack);
      enqueue_outer_pull(0);

      struct MutationSpan {
        void* device_ptr;
        std::size_t bytes;
      };
      std::vector<MutationSpan> mutation_spans;
      const auto add_field = [&](auto& field) {
        mutation_spans.push_back(
            {field.data(), field.size() * sizeof(*field.data())});
      };
      add_field(state.fld_eta);
      add_field(state.fld_nlte_f_work);
      add_field(state.fld_sigma_pe);
      if (has_nlte_clamp_counts) {
        add_field(state.fld_sigma_a);
        add_field(state.fld_sigma_R);
        add_field(state.fld_nlte_sigma_eff_work);
        add_field(state.fld_nlte_sigma_s_eff_work);
        add_field(state.fld_nlte_eta_cdf_work);
        add_field(state.fld_nlte_lambda_work);
      }
      add_field(state.fld_lower);
      add_field(state.fld_diag);
      add_field(state.fld_upper);
      add_field(state.fld_rhs);
      add_field(state.fld_cusparse_buffer);
      add_field(state.rad_E);
      add_field(state.Te);
      add_field(state.ee);
      add_field(state.Pe);
      add_field(state.rad_dep);
      add_field(state.rad_emit);
      add_field(state.fld_delta_T);
      mutation_spans.push_back({d_outer_pack, 2 * sizeof(double)});
      mutation_spans.push_back(
          {d_iteration_void,
           sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells)});
      if (d_nlte_clamp_counts != nullptr) {
        mutation_spans.push_back({d_nlte_clamp_counts, 3 * sizeof(int)});
      }

      std::size_t snapshot_bytes = 0U;
      for (const MutationSpan& span : mutation_spans) {
        snapshot_bytes += span.bytes;
      }
      auto* snapshot_arena = static_cast<std::uint8_t*>(
          core::device_scratch_acquire("fld1d:outer_pipeline_snapshots",
                                       kPipelineSlots * snapshot_bytes));
      const auto copy_snapshot = [&](const int slot,
                                     const bool restore) {
        std::size_t offset = static_cast<std::size_t>(slot) * snapshot_bytes;
        for (const MutationSpan& span : mutation_spans) {
          void* const arena_ptr = snapshot_arena + offset;
          cuda_check(
              restore
                  ? cudaMemcpyAsync(span.device_ptr,
                                    arena_ptr,
                                    span.bytes,
                                    cudaMemcpyDeviceToDevice,
                                    outer_stream)
                  : cudaMemcpyAsync(arena_ptr,
                                    span.device_ptr,
                                    span.bytes,
                                    cudaMemcpyDeviceToDevice,
                                    outer_stream),
              restore ? "FLD outer pipeline restore failed"
                      : "FLD outer pipeline snapshot failed");
          offset += span.bytes;
        }
      };

      for (int iter = 0; iter < max_iter; ++iter) {
        const int slot = iter % kPipelineSlots;
        const bool speculated = iter + 1 < max_iter;
        if (speculated) {
          copy_snapshot(slot, /*restore=*/false);
          const int next_slot = (iter + 1) % kPipelineSlots;
          launch_outer_iteration(
              iter + 1,
              has_nlte_clamp_counts
                  ? pinned_clamp_counts + 3 * next_slot
                  : nullptr);
          launch_outer_check(state, n_cells, fld.outer_tol, d_outer_pack);
          enqueue_outer_pull(next_slot);
        }

        cuda_check(cudaEventSynchronize(pull_events[slot]),
                   "FLD outer pipeline event synchronize failed");
        const double* const h_pack = pinned_outer_packs + 2 * slot;
        state.fld_outer_residual = h_pack[0];
        state.fld_outer_iterations = iter + 1;
        if (has_nlte_clamp_counts) {
          emit_nlte_clamp_warning(pinned_clamp_counts + 3 * slot);
        }
        if (h_pack[1] != 0.0) {
          state.fld_converged = true;
          if (speculated) {
            copy_snapshot(slot, /*restore=*/true);
            cuda_check(cudaStreamSynchronize(outer_stream),
                       "FLD outer pipeline rollback synchronize failed");
          }
          break;
        }
      }

      for (cudaEvent_t& event : pull_events) {
        cuda_check(cudaEventDestroy(event),
                   "FLD outer pipeline event destruction failed");
      }
    }

    rad_ebal_Um1 = rad_ebal_matter();
    rad_ebal_Ur1 = rad_ebal_field();

    if (!state.fld_converged) {
      // Silent non-convergence hid the Fleck/AFI accuracy question (W-I):
      // surface it, rate-limited.
      static int fld_outer_warn_count = 0;
      ++fld_outer_warn_count;
      if (fld_outer_warn_count <= 5 || fld_outer_warn_count % 500 == 0) {
        core::log_warning(
            "FLD 1D outer iteration hit max_outer_iterations=" +
            std::to_string(max_iter) + " without reaching outer_tol=" +
            std::to_string(fld.outer_tol) + " (residual=" +
            std::to_string(state.fld_outer_residual) + ", warning #" +
            std::to_string(fld_outer_warn_count) + ")");
      }
    }
  }

  int publish_nonfinite_count = 0;
  cuda_check(cudaMemcpy(&publish_nonfinite_count,
                        d_publish_nonfinite_count,
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "FLD publish non-finite counter read failed");
  if (publish_nonfinite_count > 0) {
    core::log_error("FLD publish: " +
                    std::to_string(publish_nonfinite_count) +
                    " non-finite solver-solution entries sanitized this step");
    if (cfg.numerics.safety.nan_fatal) {
      TENRYU_ASSERT(false,
                    "FLD solver produced non-finite radiation energy "
                    "(nan_fatal=true)");
    }
  }

  if (bytes > 0U) {
    state.fld_escaped_step =
        compute_escaped_energy(state, n_cells, n_groups, dt, outer_bc);
    if (outer_bc == kFld1dOuterMarshak && marshak_in_flux_total > 0.0) {
      double r_outer = 0.0;
      cuda_check(cudaMemcpy(&r_outer,
                            state.x_r.data() + n_cells,
                            sizeof(double),
                            cudaMemcpyDeviceToHost),
                 "FLD marshak outer radius copy failed");
      const double area =
          (state.mesh.geometry_code == 0)
              ? (4.0 * kPi * r_outer * r_outer)
              : tenryu::mesh::geometry_1d_face_area(state.mesh.geometry_code,
                                                    r_outer);
      state.fld_marshak_in_step = dt * area * marshak_in_flux_total;
    }
    state.fld_volume_source_in_step =
        compute_volume_source_energy(state,
                                     n_cells,
                                     dt,
                                     cfg.radiation.volume_source_rate,
                                     cfg.radiation.volume_source_x_max);
  }
  state.holo_ale_invalidated = false;
  if (rad_ebal_enabled) {
    const double Um2 = rad_ebal_matter();
    const double Ur2 = rad_ebal_field();
    std::ostringstream rad_os;
    rad_os << std::scientific << std::setprecision(9)
           << "[rad_ebal] dUm_src=" << (rad_ebal_Um1 - rad_ebal_Um0)
           << " dUr_src=" << (rad_ebal_Ur1 - rad_ebal_Ur0)
           << " dUm_rest=" << (Um2 - rad_ebal_Um1)
           << " dUr_rest=" << (Ur2 - rad_ebal_Ur1);
    core::log_info(rad_os.str());
  }
}

double fld_compute_max_reduced_flux_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  ensure_state_buffers(state, n_cells, n_groups);
  evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells, n_groups, 1.0e-30);
  zero_reduction_scalar(state);
  const int total = std::max(n_cells - 1, 0) * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    max_reduced_flux_kernel<<<grid, kBlock>>>(state.x_r.data(),
                                              state.rad_E.data(),
                                              state.fld_sigma_R.data(),
                                              state.fld_reduction_work.data(),
                                              n_cells,
                                              n_groups,
                                              cfg.radiation.multigroup_diffusion.opacity_floor,
                                              limiter_id(cfg.radiation.multigroup_diffusion.flux_limiter));
    cuda_check(cudaGetLastError(), "FLD reduced flux launch failed");
  }
  return copy_scalar_from_device(state);
}

double fld_compute_fv_uniform_residual_1d(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const double dt) {
  const int n_cells = static_cast<int>(state.rho.size());
  const int n_groups = std::max(cfg.radiation.groups, 1);
  ensure_state_buffers(state, n_cells, n_groups);
  evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells, n_groups, dt);
  // Audit path: interior-row consistency only; the marshak RHS injection is
  // irrelevant here (finc=nullptr) but the boundary diagonal must match the
  // production operator, so the parsed outer BC is passed through.
  assemble_tridiag(state, n_cells, n_groups, dt,
                   use_fld_fleck(mat) &&
                       cfg.radiation.multigroup_diffusion.fleck_mode != "afi",
                   parse_fld_1d_outer_bc(
                       cfg.radiation.multigroup_diffusion.boundary.outer_r),
                   nullptr,
                   cfg.radiation.volume_source_rate,
                   cfg.radiation.volume_source_x_max,
                   cfg.radiation.multigroup_diffusion.opacity_floor,
                   limiter_id(cfg.radiation.multigroup_diffusion.flux_limiter));
  zero_reduction_scalar(state);
  const int total = n_cells * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    fv_interior_flux_residual_kernel<<<grid, kBlock>>>(
        state.vol.data(),
        state.fld_sigma_a.data(),
        state.fld_lower.data(),
        state.fld_diag.data(),
        state.fld_upper.data(),
        state.fld_reduction_work.data(),
        n_cells,
        n_groups,
        dt);
    cuda_check(cudaGetLastError(), "FLD FV residual launch failed");
  }
  return copy_scalar_from_device(state);
}

}  // namespace tenryu::radiation
