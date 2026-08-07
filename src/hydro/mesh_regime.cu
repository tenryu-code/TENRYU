#include "hydro/mesh_regime.cuh"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "hydro/rz_corner_mass.cuh"

namespace tenryu::hydro {
namespace {

constexpr double kCdScoreLo = 0.1;
constexpr double kCdScoreHi = 0.9;
constexpr double kCdMinHigh = 0.3;
constexpr double kCdMinLow = 0.2;
constexpr float kInteriorSmoothThreshold = 0.5F;
constexpr float kInteriorCdThreshold = 0.85F;
constexpr double kTiny = 1.0e-300;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__host__ __device__ double saturate01(const double x) {
  return fmin(1.0, fmax(0.0, x));
}

__host__ __device__ int rz_node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

__host__ __device__ double cross2(const double ar,
                                  const double az,
                                  const double br,
                                  const double bz) {
  return ar * bz - az * br;
}

__host__ __device__ double signed_area4(const double* r, const double* z) {
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    sum += r[k] * z[kp] - r[kp] * z[k];
  }
  return 0.5 * sum;
}

__host__ __device__ double signed_rz_quad_volume(const double* r,
                                                 const double* z) {
  const double rc = 0.25 * (r[0] + r[1] + r[2] + r[3]);
  const double zc = 0.25 * (z[0] + z[1] + z[2] + z[3]);
  double sum = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const double rk = r[k] - rc;
    const double rkp = r[kp] - rc;
    const double zk = z[k] - zc;
    const double zkp = z[kp] - zc;
    const double cross = rk * zkp - rkp * zk;
    const double weight = rk + rkp + 3.0 * rc;
    sum += cross * weight;
  }
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  return pi_over_three * sum;
}

__host__ __device__ double min_margin_update(double current,
                                             const double value) {
  return fmin(current, value);
}

__device__ double cell_center_r(const double* __restrict__ x_r,
                                const int i,
                                const int j,
                                const int nz) {
  const int n00 = rz_node_index(i, j, nz);
  const int n10 = rz_node_index(i + 1, j, nz);
  const int n11 = rz_node_index(i + 1, j + 1, nz);
  const int n01 = rz_node_index(i, j + 1, nz);
  return 0.25 * (x_r[n00] + x_r[n10] + x_r[n11] + x_r[n01]);
}

__device__ double cell_center_z(const double* __restrict__ x_z,
                                const int i,
                                const int j,
                                const int nz) {
  const int n00 = rz_node_index(i, j, nz);
  const int n10 = rz_node_index(i + 1, j, nz);
  const int n11 = rz_node_index(i + 1, j + 1, nz);
  const int n01 = rz_node_index(i, j + 1, nz);
  return 0.25 * (x_z[n00] + x_z[n10] + x_z[n11] + x_z[n01]);
}

__device__ double safe_log_rho(const double* __restrict__ rho,
                               const int c) {
  return log(fmax(rho[c], kTiny));
}

__device__ double derivative_log_rho_r(const double* __restrict__ rho,
                                       const double* __restrict__ x_r,
                                       const int i,
                                       const int j,
                                       const int nr,
                                       const int nz) {
  if (nr <= 1) {
    return 0.0;
  }
  int im = i;
  int ip = i;
  if (i <= 0) {
    ip = 1;
  } else if (i >= nr - 1) {
    im = nr - 2;
  } else {
    im = i - 1;
    ip = i + 1;
  }
  const double xl = cell_center_r(x_r, im, j, nz);
  const double xr = cell_center_r(x_r, ip, j, nz);
  const double dx = xr - xl;
  if (!(fabs(dx) > 0.0) || !isfinite(dx)) {
    return 0.0;
  }
  return (safe_log_rho(rho, ip * nz + j) - safe_log_rho(rho, im * nz + j)) / dx;
}

__device__ double derivative_log_rho_z(const double* __restrict__ rho,
                                       const double* __restrict__ x_z,
                                       const int i,
                                       const int j,
                                       const int nz) {
  if (nz <= 1) {
    return 0.0;
  }
  int jm = j;
  int jp = j;
  if (j <= 0) {
    jp = 1;
  } else if (j >= nz - 1) {
    jm = nz - 2;
  } else {
    jm = j - 1;
    jp = j + 1;
  }
  const double zl = cell_center_z(x_z, i, jm, nz);
  const double zr = cell_center_z(x_z, i, jp, nz);
  const double dz = zr - zl;
  if (!(fabs(dz) > 0.0) || !isfinite(dz)) {
    return 0.0;
  }
  return (safe_log_rho(rho, i * nz + jp) - safe_log_rho(rho, i * nz + jm)) / dz;
}

__device__ MeshTopoTag topology_tag(const int i,
                                    const int j,
                                    const int nr,
                                    const int nz,
                                    const int axis_guard_band_cells,
                                    const bool has_physical_rz_axis) {
  if (has_physical_rz_axis && i == 0) {
    return MeshTopoTag::AxisFace;
  }
  if (has_physical_rz_axis && axis_guard_band_cells > 0 &&
      i <= axis_guard_band_cells) {
    return MeshTopoTag::AxisBand;
  }
  const bool radial_outer = (i == nr - 1);
  const bool z_bottom = (j == 0);
  const bool z_top = (j == nz - 1);
  if (radial_outer && (z_bottom || z_top)) {
    return MeshTopoTag::CornerBoundary;
  }
  if (radial_outer) {
    return MeshTopoTag::RadialOuterBoundary;
  }
  if (z_bottom) {
    return MeshTopoTag::ZBottomBoundary;
  }
  if (z_top) {
    return MeshTopoTag::ZTopBoundary;
  }
  return MeshTopoTag::Interior;
}

__device__ bool domain_boundary_topology(const MeshTopoTag topo) {
  return topo == MeshTopoTag::RadialOuterBoundary ||
         topo == MeshTopoTag::ZBottomBoundary ||
         topo == MeshTopoTag::ZTopBoundary ||
         topo == MeshTopoTag::CornerBoundary;
}

__device__ double cell_dvdt(const double* __restrict__ v_r,
                            const double* __restrict__ v_z,
                            const double* __restrict__ cell_Svec_r,
                            const double* __restrict__ cell_Svec_z,
                            const int i,
                            const int j,
                            const int nz,
                            const int c) {
  const int n00 = rz_node_index(i, j, nz);
  const int n10 = rz_node_index(i + 1, j, nz);
  const int n11 = rz_node_index(i + 1, j + 1, nz);
  const int n01 = rz_node_index(i, j + 1, nz);
  const int idx = 4 * c;
  double out = 0.0;
  out += v_r[n00] * cell_Svec_r[idx + 0] + v_z[n00] * cell_Svec_z[idx + 0];
  out += v_r[n10] * cell_Svec_r[idx + 1] + v_z[n10] * cell_Svec_z[idx + 1];
  out += v_r[n11] * cell_Svec_r[idx + 2] + v_z[n11] * cell_Svec_z[idx + 2];
  out += v_r[n01] * cell_Svec_r[idx + 3] + v_z[n01] * cell_Svec_z[idx + 3];
  return out;
}

__device__ bool cell_active_or_unmasked(const std::uint8_t* __restrict__ cell_is_void,
                                        const std::int8_t* __restrict__ hydro_active,
                                        const int c) {
  return (cell_is_void == nullptr || cell_is_void[c] == 0U) &&
         (hydro_active == nullptr || hydro_active[c] != 0);
}

__device__ int infer_button_outer_node_ring(
    const std::uint8_t* __restrict__ cell_is_void,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz) {
  for (int i = 1; i < nr; ++i) {
    const int c = i * nz;
    if (cell_active_or_unmasked(cell_is_void, hydro_active, c)) {
      return i;
    }
  }
  return -1;
}

__device__ double button_cell_dvdt(
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int outer_ring,
    const int nz) {
  double centroid_r = 0.0;
  double centroid_z = 0.0;
  rz::button_polygon_area_centroid_from_nodes(
      x_r, x_z, outer_ring, nz, &centroid_r, &centroid_z);
  const int nverts = nz + 1;
  double out = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int n = rz::button_seam_node_index(outer_ring, k, nz);
    double sr = 0.0;
    double sz = 0.0;
    rz::button_polygon_svec_from_nodes(x_r, x_z, outer_ring, nz, k,
                                       centroid_r, centroid_z, &sr, &sz);
    out += v_r[n] * sr + v_z[n] * sz;
  }
  return out;
}

__device__ CellRegime void_cell_regime() {
  CellRegime regime{};
  regime.topo_tag = static_cast<std::uint8_t>(MeshTopoTag::Inactive);
  regime.primary_regime =
      static_cast<std::uint8_t>(MeshFailureRegime::VoidOrInactive);
  regime.trigger_scale_threshold = kInteriorSmoothThreshold;
  return regime;
}

}  // namespace

__host__ __device__ AxisMarginPredicateResult evaluate_axis_margin_predicate_quad(
    const double* r_current,
    const double* z_current,
    const double* r_trial,
    const double* z_trial,
    const double floor_eps) {
  AxisMarginPredicateResult result{};
  result.min_margin = INFINITY;

  double rt[4] = {r_trial[0], r_trial[1], r_trial[2], r_trial[3]};
  const double zt[4] = {z_trial[0], z_trial[1], z_trial[2], z_trial[3]};
  rt[0] = 0.0;
  rt[3] = 0.0;

  const double axis_scale =
      fmax(fmax(fabs(r_current[0]), fabs(r_current[3])), 1e-30);
  const double axis_floor = floor_eps * axis_scale;
  const double axis_margin = axis_floor - fmax(fabs(r_trial[0]), fabs(r_trial[3]));
  result.min_margin = min_margin_update(result.min_margin, axis_margin);
  if (!(axis_margin >= 0.0) || !isfinite(axis_margin)) {
    result.admissible = false;
    result.failed_condition = 1;
    return result;
  }

  const double r_scale = fmax(fmax(fabs(r_current[1]), fabs(r_current[2])), 1e-30);
  const double r_floor = floor_eps * r_scale;
  const double r_margin = fmin(rt[1], rt[2]) - r_floor;
  result.min_margin = min_margin_update(result.min_margin, r_margin);
  if (!(rt[1] > r_floor && rt[2] > r_floor) || !isfinite(r_margin)) {
    result.admissible = false;
    result.failed_condition = 2;
    return result;
  }

  const double dz0 = z_current[3] - z_current[0];
  const double z_floor = floor_eps * fmax(fabs(dz0), 1e-30);
  const double orient = (dz0 >= 0.0) ? 1.0 : -1.0;
  const double z_margin = orient * (zt[3] - zt[0]) - z_floor;
  result.min_margin = min_margin_update(result.min_margin, z_margin);
  if (!(z_margin > 0.0) || !isfinite(z_margin)) {
    result.admissible = false;
    result.failed_condition = 3;
    return result;
  }

  const double area_current = signed_area4(r_current, z_current);
  const double area_floor = floor_eps * fmax(fabs(area_current), 1e-30);
  const double area_margin = signed_area4(rt, zt) - area_floor;
  result.min_margin = min_margin_update(result.min_margin, area_margin);
  if (!(area_margin > 0.0) || !isfinite(area_margin)) {
    result.admissible = false;
    result.failed_condition = 4;
    return result;
  }

  const double vol_current = signed_rz_quad_volume(r_current, z_current);
  const double vol_floor = floor_eps * fmax(fabs(vol_current), 1e-30);
  const double vol_margin = signed_rz_quad_volume(rt, zt) - vol_floor;
  result.min_margin = min_margin_update(result.min_margin, vol_margin);
  if (!(vol_margin > 0.0) || !isfinite(vol_margin)) {
    result.admissible = false;
    result.failed_condition = 5;
    return result;
  }

  result.admissible = true;
  result.failed_condition = 0;
  return result;
}

__global__ void classify_mesh_regimes_kernel(
    CellRegime* __restrict__ out,
    std::uint8_t* __restrict__ previous_primary_regime,
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ cell_area,
    const double* __restrict__ cell_Svec_r,
    const double* __restrict__ cell_Svec_z,
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ v_r,
    const double* __restrict__ v_z,
    const std::uint8_t* __restrict__ cell_is_void,
    const std::int8_t* __restrict__ hydro_active,
    const int nr,
    const int nz,
    const double dt,
    const int axis_guard_band_cells,
    const bool has_physical_rz_axis) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }

  if ((cell_is_void != nullptr && cell_is_void[c] != 0U) ||
      (hydro_active != nullptr && hydro_active[c] == 0)) {
    out[c] = void_cell_regime();
    if (previous_primary_regime != nullptr) {
      previous_primary_regime[c] =
          static_cast<std::uint8_t>(MeshFailureRegime::VoidOrInactive);
    }
    return;
  }

  const int i = c / nz;
  const int j = c - i * nz;
  const MeshTopoTag topo =
      topology_tag(i, j, nr, nz, axis_guard_band_cells,
                   has_physical_rz_axis);

  const double dr_log = derivative_log_rho_r(rho, x_r, i, j, nr, nz);
  const double dz_log = derivative_log_rho_z(rho, x_z, i, j, nz);
  const double h = (cell_area[c] > 0.0 && isfinite(cell_area[c]))
                       ? sqrt(cell_area[c])
                       : 0.0;
  const double grad_sensor = h * sqrt(dr_log * dr_log + dz_log * dz_log);
  const double grad_score =
      saturate01((grad_sensor - kCdScoreLo) / (kCdScoreHi - kCdScoreLo));

  const bool button_like_c0 =
      c == 0 && cell_area[c] > 0.0 && cell_Svec_r[0] == 0.0 &&
      cell_Svec_z[0] == 0.0 && cell_Svec_r[1] == 0.0 &&
      cell_Svec_z[1] == 0.0 && cell_Svec_r[2] == 0.0 &&
      cell_Svec_z[2] == 0.0 && cell_Svec_r[3] == 0.0 &&
      cell_Svec_z[3] == 0.0;
  const int button_outer_ring =
      button_like_c0 ? infer_button_outer_node_ring(cell_is_void, hydro_active,
                                                   nr, nz)
                     : -1;
  const double dVdt =
      (button_outer_ring >= 1)
          ? button_cell_dvdt(v_r, v_z, x_r, x_z, button_outer_ring, nz)
          : cell_dvdt(v_r, v_z, cell_Svec_r, cell_Svec_z, i, j, nz, c);
  const double div_u = (vol[c] > 0.0 && isfinite(vol[c])) ? dVdt / vol[c] : 0.0;
  const double compression_sensor = fmax(0.0, -dt * div_u);
  const double compression_score =
      saturate01((compression_sensor - kCdScoreLo) / (kCdScoreHi - kCdScoreLo));
  const float cd_score = static_cast<float>(grad_score * compression_score);

  const MeshFailureRegime prev_primary =
      previous_primary_regime == nullptr
          ? MeshFailureRegime::Unknown
          : static_cast<MeshFailureRegime>(previous_primary_regime[c]);
  const bool previous_cd = prev_primary == MeshFailureRegime::InteriorCD;
  const bool interior_cd =
      previous_cd ? (cd_score >= static_cast<float>(kCdMinLow))
                  : (cd_score > static_cast<float>(kCdMinHigh));

  MeshFailureRegime primary = MeshFailureRegime::InteriorSmooth;
  if (topo == MeshTopoTag::AxisFace) {
    primary = MeshFailureRegime::AxisFace;
  } else if (topo == MeshTopoTag::AxisBand) {
    primary = MeshFailureRegime::AxisBand;
  } else if (domain_boundary_topology(topo)) {
    primary = MeshFailureRegime::DomainBoundary;
  } else if (interior_cd) {
    primary = MeshFailureRegime::InteriorCD;
  }

  CellRegime regime{};
  regime.topo_tag = static_cast<std::uint8_t>(topo);
  regime.cd_score = cd_score;
  regime.compression_score = static_cast<float>(compression_score);
  regime.boundary_score = domain_boundary_topology(topo) ? 1.0F : 0.0F;
  regime.trigger_scale_threshold =
      (primary == MeshFailureRegime::InteriorCD) ? kInteriorCdThreshold
                                                 : kInteriorSmoothThreshold;
  regime.primary_regime = static_cast<std::uint8_t>(primary);
  out[c] = regime;
  if (previous_primary_regime != nullptr) {
    previous_primary_regime[c] = regime.primary_regime;
  }
}

MeshRegimeDeviceCache::~MeshRegimeDeviceCache() {
  release();
}

MeshRegimeDeviceCache::MeshRegimeDeviceCache(
    MeshRegimeDeviceCache&& other) noexcept {
  current_ = other.current_;
  previous_primary_ = other.previous_primary_;
  size_ = other.size_;
  valid_ = other.valid_;
  other.current_ = nullptr;
  other.previous_primary_ = nullptr;
  other.size_ = 0;
  other.valid_ = false;
}

MeshRegimeDeviceCache& MeshRegimeDeviceCache::operator=(
    MeshRegimeDeviceCache&& other) noexcept {
  if (this != &other) {
    release();
    current_ = other.current_;
    previous_primary_ = other.previous_primary_;
    size_ = other.size_;
    valid_ = other.valid_;
    other.current_ = nullptr;
    other.previous_primary_ = nullptr;
    other.size_ = 0;
    other.valid_ = false;
  }
  return *this;
}

void MeshRegimeDeviceCache::ensure_size(const std::size_t count) {
  if (count == size_ && current_ != nullptr && previous_primary_ != nullptr) {
    return;
  }
  release();
  size_ = count;
  valid_ = false;
  if (size_ == 0) {
    return;
  }
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&current_),
                        size_ * sizeof(CellRegime)),
             "mesh regime: cudaMalloc current failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&previous_primary_),
                        size_ * sizeof(std::uint8_t)),
             "mesh regime: cudaMalloc previous primary failed");
  cuda_check(cudaMemset(current_, 0, size_ * sizeof(CellRegime)),
             "mesh regime: cudaMemset current failed");
  cuda_check(cudaMemset(previous_primary_, 0, size_ * sizeof(std::uint8_t)),
             "mesh regime: cudaMemset previous primary failed");
}

void MeshRegimeDeviceCache::release() {
  if (current_ != nullptr) {
    cuda_check(cudaFree(current_), "mesh regime: cudaFree current failed");
    current_ = nullptr;
  }
  if (previous_primary_ != nullptr) {
    cuda_check(cudaFree(previous_primary_),
               "mesh regime: cudaFree previous primary failed");
    previous_primary_ = nullptr;
  }
  size_ = 0;
  valid_ = false;
}

void classify_mesh_regimes(const core::State& state,
                           const double dt,
                           const int axis_guard_band_cells,
                           MeshRegimeDeviceCache& cache,
                           const std::int8_t* d_hydro_active,
                           const bool has_physical_rz_axis) {
  if (state.mesh.dim != 2 || !(dt > 0.0) || !std::isfinite(dt)) {
    cache.invalidate();
    return;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_cells = nr * nz;
  if (n_cells <= 0) {
    cache.invalidate();
    return;
  }
  TENRYU_ASSERT(state.rho.size() == static_cast<std::size_t>(n_cells) &&
                    state.vol.size() == static_cast<std::size_t>(n_cells),
                "mesh regime classifier requires rho/vol size == n_cells");
  TENRYU_ASSERT(state.mesh.cell_area.size() == static_cast<std::size_t>(n_cells),
                "mesh regime classifier requires cell_area size == n_cells");
  TENRYU_ASSERT(state.mesh.cell_Svec_r.size() == static_cast<std::size_t>(4 * n_cells) &&
                    state.mesh.cell_Svec_z.size() == static_cast<std::size_t>(4 * n_cells),
                "mesh regime classifier requires Svec size == 4*n_cells");
  TENRYU_ASSERT(state.cell_is_void.size() == static_cast<std::size_t>(n_cells),
                "mesh regime classifier requires cell_is_void size == n_cells");

  cache.ensure_size(static_cast<std::size_t>(n_cells));

  double* d_area = nullptr;
  double* d_svec_r = nullptr;
  double* d_svec_z = nullptr;
  std::uint8_t* d_void = nullptr;
  std::int8_t* d_active_owned = nullptr;
  const std::int8_t* d_active = d_hydro_active;

  d_area = static_cast<double*>(core::device_scratch_acquire(
      "mregime:cell_area", state.mesh.cell_area.size() * sizeof(double)));
  d_svec_r = static_cast<double*>(core::device_scratch_acquire(
      "mregime:svec_r", state.mesh.cell_Svec_r.size() * sizeof(double)));
  d_svec_z = static_cast<double*>(core::device_scratch_acquire(
      "mregime:svec_z", state.mesh.cell_Svec_z.size() * sizeof(double)));
  d_void = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "mregime:cell_is_void",
      state.cell_is_void.size() * sizeof(std::uint8_t)));
  cuda_check(cudaMemcpy(d_area, state.mesh.cell_area.data(),
                        state.mesh.cell_area.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "mesh regime: copy cell_area failed");
  cuda_check(cudaMemcpy(d_svec_r, state.mesh.cell_Svec_r.data(),
                        state.mesh.cell_Svec_r.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "mesh regime: copy Svec_r failed");
  cuda_check(cudaMemcpy(d_svec_z, state.mesh.cell_Svec_z.data(),
                        state.mesh.cell_Svec_z.size() * sizeof(double),
                        cudaMemcpyHostToDevice),
             "mesh regime: copy Svec_z failed");
  cuda_check(cudaMemcpy(d_void, state.cell_is_void.data(),
                        state.cell_is_void.size() * sizeof(std::uint8_t),
                        cudaMemcpyHostToDevice),
             "mesh regime: copy cell_is_void failed");

  if (d_active == nullptr && !state.hydro_active.empty()) {
    d_active_owned = static_cast<std::int8_t*>(core::device_scratch_acquire(
        "mregime:hydro_active",
        state.hydro_active.size() * sizeof(std::int8_t)));
    cuda_check(cudaMemcpy(d_active_owned, state.hydro_active.data(),
                          state.hydro_active.size() * sizeof(std::int8_t),
                          cudaMemcpyHostToDevice),
               "mesh regime: copy hydro_active failed");
    d_active = d_active_owned;
  }

  const int blocks = (n_cells + 255) / 256;
  classify_mesh_regimes_kernel<<<blocks, 256>>>(
      cache.current(), cache.previous_primary(), state.rho.data(), state.vol.data(),
      d_area, d_svec_r, d_svec_z, state.x_r.data(), state.x_z.data(),
      state.v_r.data(), state.v_z.data(), d_void, d_active, nr, nz, dt,
      axis_guard_band_cells, has_physical_rz_axis);
  cuda_check(cudaGetLastError(), "mesh regime classifier kernel launch failed");
  cuda_check(core::debug_kernel_sync(), "mesh regime classifier kernel failed");

  cache.mark_valid();
}

std::vector<CellRegime> copy_mesh_regimes_to_host(
    const MeshRegimeDeviceCache& cache) {
  std::vector<CellRegime> host(cache.size());
  if (cache.size() == 0) {
    return host;
  }
  cuda_check(cudaMemcpy(host.data(), cache.current(),
                        cache.size() * sizeof(CellRegime),
                        cudaMemcpyDeviceToHost),
             "mesh regime: copy current to host failed");
  return host;
}

}  // namespace tenryu::hydro
