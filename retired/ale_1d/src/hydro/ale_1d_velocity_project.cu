#include "hydro/ale_1d_velocity_project.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <vector>

#include <cub/cub.cuh>

#include "core/error.hpp"

namespace tenryu::hydro::ale1d {
namespace {

constexpr int kBlockSize = 256;
constexpr double kTiny = 1.0e-300;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

int blocks_for(const int n) {
  return (n + kBlockSize - 1) / kBlockSize;
}

bool closure_dump_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("TENRYU_ALE1D_CLOSURE_DUMP");
    return value != nullptr && value[0] == '1' && value[1] == '\0';
  }();
  return enabled;
}

__host__ __device__ double minmod3(const double a, const double b, const double c) {
  if (a > 0.0 && b > 0.0 && c > 0.0) {
    return fmin(a, fmin(b, c));
  }
  if (a < 0.0 && b < 0.0 && c < 0.0) {
    return -fmin(fabs(a), fmin(fabs(b), fabs(c)));
  }
  return 0.0;
}

__host__ __device__ double slope_scale_for_face_value(const double q_face,
                                                      const double q_cell,
                                                      const double q_min,
                                                      const double q_max) {
  if (q_face > q_max) {
    const double denom = q_face - q_cell;
    return denom > kTiny ? fmax(0.0, fmin(1.0, (q_max - q_cell) / denom))
                         : 0.0;
  }
  if (q_face < q_min) {
    const double denom = q_face - q_cell;
    return denom < -kTiny ? fmax(0.0, fmin(1.0, (q_min - q_cell) / denom))
                          : 0.0;
  }
  return 1.0;
}

// Limited slope in the mass coordinate of a cell value q_i with neighbours
// q_{i-1}, q_{i+1} (cell centres (m_{i-1} + m_i)/2 and (m_i + m_{i+1})/2
// apart): generalized minmod with theta, scaled so that the values at the
// cell's two faces stay within the neighbours' range.
__device__ double mass_coordinate_slope(const double qm,
                                        const double qi,
                                        const double qp,
                                        const double mm,
                                        const double mi,
                                        const double mp,
                                        const double theta) {
  if (!isfinite(qm) || !isfinite(qi) || !isfinite(qp) || !(mm > 0.0) ||
      !(mi > 0.0) || !(mp > 0.0)) {
    return 0.0;
  }
  const double d_minus = 0.5 * (mm + mi);
  const double d_plus = 0.5 * (mi + mp);
  const double s = minmod3(theta * (qi - qm) / d_minus,
                           (qp - qm) / (d_minus + d_plus),
                           theta * (qp - qi) / d_plus);
  if (s == 0.0 || !isfinite(s)) {
    return 0.0;
  }
  const double q_l = qi - 0.5 * s * mi;
  const double q_r = qi + 0.5 * s * mi;
  const double q_min = fmin(qm, fmin(qi, qp));
  const double q_max = fmax(qm, fmax(qi, qp));
  const double alpha = fmin(slope_scale_for_face_value(q_l, qi, q_min, q_max),
                            slope_scale_for_face_value(q_r, qi, q_min, q_max));
  return alpha * s;
}

// Slopes of the two node velocities each cell carries: psi^L_i = v_i and
// psi^R_i = v_{i+1}. Boundary cells and cells next to a face with phi = 0
// (pinned/protected, or a first-order mass basis) stay first order, as in
// the remap of the conserved fields.
__global__ void half_index_shift_slopes_kernel(const double* __restrict__ v_old,
                                               const double* __restrict__ mass_old,
                                               const double* __restrict__ phi_face,
                                               double* __restrict__ slope_left,
                                               double* __restrict__ slope_right,
                                               const int n,
                                               const double theta) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  slope_left[i] = 0.0;
  slope_right[i] = 0.0;
  if (i <= 0 || i >= n - 1 || phi_face[i] == 0.0 || phi_face[i + 1] == 0.0) {
    return;
  }
  const double mm = mass_old[i - 1];
  const double mi = mass_old[i];
  const double mp = mass_old[i + 1];
  slope_left[i] =
      mass_coordinate_slope(v_old[i - 1], v_old[i], v_old[i + 1], mm, mi, mp, theta);
  slope_right[i] =
      mass_coordinate_slope(v_old[i], v_old[i + 1], v_old[i + 2], mm, mi, mp, theta);
}

// Flux of a cell quantity through face j: the face mass flux F_j times the
// donor's linear reconstruction at the swept mass's centroid (F > 0 takes
// the leftmost F of cell j, F < 0 the rightmost |F| of cell j-1), blended
// with the donor value by phi_j. Returns false (no flux) for F = 0 or a face
// without a donor cell.
__device__ bool swept_value(const int j,
                            const int field,
                            const double* __restrict__ v_old,
                            const double* __restrict__ mass_old,
                            const double* __restrict__ mass_flux,
                            const double* __restrict__ phi_face,
                            const double* __restrict__ slope_left,
                            const double* __restrict__ slope_right,
                            const int n,
                            double* value) {
  const double F = mass_flux[j];
  int d = -1;
  if (F > 0.0) {
    d = j;
  } else if (F < 0.0) {
    d = j - 1;
  }
  if (d < 0 || d >= n) {
    return false;
  }
  const double offset = (F > 0.0) ? 0.5 * (F - mass_old[d]) : 0.5 * (mass_old[d] + F);
  const double q = (field == 0) ? v_old[d] : v_old[d + 1];
  const double s = (field == 0) ? slope_left[d] : slope_right[d];
  *value = q + phi_face[j] * s * offset;
  return true;
}

// psi^n_i = psi^o_i + [F_{i+1} (psi_{i+1/2} - psi^o_i) - F_i (psi_{i-1/2} -
// psi^o_i)] / m^n_i: the conservative update m^n psi^n = m^o psi^o +
// F_{i+1} psi_{i+1/2} - F_i psi_{i-1/2} written with m^o = m^n - F_{i+1} + F_i,
// so that a uniform psi stays exactly uniform.
__global__ void half_index_shift_remap_kernel(const double* __restrict__ v_old,
                                              const double* __restrict__ mass_old,
                                              const double* __restrict__ mass_new,
                                              const double* __restrict__ mass_flux,
                                              const double* __restrict__ phi_face,
                                              const double* __restrict__ slope_left,
                                              const double* __restrict__ slope_right,
                                              double* __restrict__ psi_left,
                                              double* __restrict__ psi_right,
                                              double* __restrict__ p_old,
                                              double* __restrict__ p_new,
                                              double* __restrict__ u_new,
                                              const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double m_new = mass_new[i];
  const double f_plus = mass_flux[i + 1];
  const double f_minus = mass_flux[i];
  double psi[2];
  for (int field = 0; field < 2; ++field) {
    const double psi_old = (field == 0) ? v_old[i] : v_old[i + 1];
    double value = psi_old;
    if (m_new > kTiny) {
      double q_plus = 0.0;
      double q_minus = 0.0;
      const double dq_plus =
          swept_value(i + 1, field, v_old, mass_old, mass_flux, phi_face, slope_left,
                      slope_right, n, &q_plus)
              ? f_plus * (q_plus - psi_old)
              : 0.0;
      const double dq_minus =
          swept_value(i, field, v_old, mass_old, mass_flux, phi_face, slope_left,
                      slope_right, n, &q_minus)
              ? f_minus * (q_minus - psi_old)
              : 0.0;
      value = psi_old + (dq_plus - dq_minus) / m_new;
    }
    psi[field] = value;
  }
  psi_left[i] = psi[0];
  psi_right[i] = psi[1];
  p_old[i] = mass_old[i] * (0.5 * (v_old[i] + v_old[i + 1]));
  const double p = 0.5 * m_new * (psi[0] + psi[1]);
  p_new[i] = p;
  u_new[i] = (m_new > kTiny) ? p / m_new : 0.0;
}

// v_j = (m_j psi^L_j + m_{j-1} psi^R_{j-1}) / (m_j + m_{j-1}), evaluated as
// psi^R_{j-1} + m_j (psi^L_j - psi^R_{j-1}) / (m_j + m_{j-1}) so that equal
// values return exactly; the centre node is at rest and the outer node takes
// the last cell's right value.
__global__ void half_index_shift_node_kernel(const double* __restrict__ mass_new,
                                             const double* __restrict__ psi_left,
                                             const double* __restrict__ psi_right,
                                             double* __restrict__ v_new,
                                             const int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }
  if (j == 0) {
    v_new[0] = 0.0;
    return;
  }
  if (j == n) {
    v_new[n] = psi_right[n - 1];
    return;
  }
  const double m_right = fmax(mass_new[j], 0.0);
  const double m_left = fmax(mass_new[j - 1], 0.0);
  const double denom = m_right + m_left;
  const double weight_right = (denom > kTiny) ? m_right / denom : 0.5;
  v_new[j] = psi_right[j - 1] + weight_right * (psi_left[j] - psi_right[j - 1]);
}

__global__ void kinetic_energy_kernel(const double* __restrict__ mass,
                                      const double* __restrict__ v,
                                      double* __restrict__ ke_node,
                                      const int n) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j > n) {
    return;
  }
  const double left = (j > 0) ? mass[j - 1] : 0.0;
  const double right = (j < n) ? mass[j] : 0.0;
  const double m_node = 0.5 * (left + right);
  ke_node[j] = 0.5 * m_node * v[j] * v[j];
}

__global__ void deposit_kinetic_closure_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    const double* __restrict__ ke_remap,
    const double* __restrict__ mass_new,
    const double* __restrict__ v_new,
    double* __restrict__ deposited,
    const int n,
    const int two_temperature) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  deposited[i] = 0.0;

  const double m = fmax(mass_new[i], 0.0);
  if (!(m > 0.0)) {
    return;
  }
  const double ee_before = ee[i];
  const double ei_before = ei[i];
  const double e_e_raw = fmax(ee_before, 0.0);
  const double e_i_raw = fmax(ei_before, 0.0);
  const double floor_ee = fmin(0.0, ee_before);
  const double floor_ei = fmin(0.0, ei_before);

  const double kinetic_new =
      0.25 * m * (v_new[i] * v_new[i] + v_new[i + 1] * v_new[i + 1]);
  double dE = ke_remap[i] - kinetic_new;
  if (!isfinite(dE) || dE == 0.0) {
    return;
  }

  const double e_sum = e_e_raw + (two_temperature != 0 ? e_i_raw : 0.0);
  if (dE < 0.0) {
    const double removable = m * e_sum;
    if (!(removable > 0.0)) {
      return;
    }
    dE = fmax(dE, -removable);
  }

  const double de = dE / m;
  if (two_temperature == 0) {
    ee[i] = fmax(ee_before + de, floor_ee);
    deposited[i] = m * (ee[i] - ee_before);
    return;
  }

  const double f_e =
      (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
  ee[i] = fmax(ee_before + f_e * de, floor_ee);
  ei[i] = fmax(ei_before + (1.0 - f_e) * de, floor_ei);
  deposited[i] =
      m * ((ee[i] - ee_before) + (ei[i] - ei_before));
}

__global__ void compute_kinetic_closure_deficit_capacity_kernel(
    double* __restrict__ deficit,
    double* __restrict__ capacity,
    const double* __restrict__ ee,
    const double* __restrict__ ei,
    const double* __restrict__ ke_remap,
    const double* __restrict__ mass_new,
    const double* __restrict__ v_new,
    const double e_floor,
    const int n,
    const int two_temperature) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  deficit[i] = 0.0;
  capacity[i] = 0.0;
  const double m = fmax(mass_new[i], 0.0);
  if (!(m > 0.0)) {
    return;
  }
  const double ee_before = ee[i];
  const double ei_before = ei[i];
  const double e_e_raw = fmax(ee_before, 0.0);
  const double e_i_raw = fmax(ei_before, 0.0);
  const double floor_e = fmax(e_floor, 0.0);
  const double floor_ee = fmin(floor_e, ee_before);
  const double floor_ei = fmin(floor_e, ei_before);

  const double kinetic_new =
      0.25 * m * (v_new[i] * v_new[i] + v_new[i + 1] * v_new[i + 1]);
  double dE = ke_remap[i] - kinetic_new;
  if (!isfinite(dE)) {
    dE = 0.0;
  }

  const double de = dE / m;
  if (two_temperature == 0) {
    const double ee_tent = ee_before + de;
    deficit[i] = fmax(0.0, floor_ee - ee_tent) * m;
    capacity[i] = fmax(0.0, ee_tent - floor_ee) * m;
    return;
  }

  const double e_sum = e_e_raw + e_i_raw;
  const double f_e =
      (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
  const double ee_tent = ee_before + f_e * de;
  const double ei_tent = ei_before + (1.0 - f_e) * de;
  const double D_e = fmax(0.0, floor_ee - ee_tent) * m;
  const double D_i = fmax(0.0, floor_ei - ei_tent) * m;
  const double C_e = fmax(0.0, ee_tent - floor_ee) * m;
  const double C_i = fmax(0.0, ei_tent - floor_ei) * m;
  deficit[i] = D_e + D_i;
  capacity[i] = C_e + C_i;
}

__global__ void apply_kinetic_closure_redistribution_kernel(
    double* __restrict__ ee,
    double* __restrict__ ei,
    double* __restrict__ deposited,
    const double* __restrict__ ke_remap,
    const double* __restrict__ mass_new,
    const double* __restrict__ v_new,
    const double e_floor,
    const double absorption_factor,
    const int n,
    const int two_temperature) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }

  deposited[i] = 0.0;
  const double m = fmax(mass_new[i], 0.0);
  if (!(m > 0.0)) {
    return;
  }
  const double ee_before = ee[i];
  const double ei_before = ei[i];
  const double e_e_raw = fmax(ee_before, 0.0);
  const double e_i_raw = fmax(ei_before, 0.0);
  const double floor_e = fmax(e_floor, 0.0);
  const double floor_ee = fmin(floor_e, ee_before);
  const double floor_ei = fmin(floor_e, ei_before);

  const double kinetic_new =
      0.25 * m * (v_new[i] * v_new[i] + v_new[i + 1] * v_new[i + 1]);
  double dE = ke_remap[i] - kinetic_new;
  if (!isfinite(dE)) {
    dE = 0.0;
  }

  const double de = dE / m;
  double ee_after = ee_before;
  double ei_after = ei_before;

  if (two_temperature == 0) {
    const double tentative_e_e = ee_before + de;
    const double D_e = fmax(0.0, floor_ee - tentative_e_e) * m;
    const double C_e = fmax(0.0, tentative_e_e - floor_ee) * m;
    ee_after = (D_e > 0.0)
                   ? floor_ee
                   : fmax(floor_ee,
                          tentative_e_e - absorption_factor * C_e / m);
    ei_after = ei_before;
  } else {
    const double e_sum = e_e_raw + e_i_raw;
    const double f_e =
        (e_sum > 0.0 && isfinite(e_sum)) ? (e_e_raw / e_sum) : 0.5;
    const double tentative_e_e = ee_before + f_e * de;
    const double tentative_e_i = ei_before + (1.0 - f_e) * de;
    const double D_e = fmax(0.0, floor_ee - tentative_e_e) * m;
    const double D_i = fmax(0.0, floor_ei - tentative_e_i) * m;
    const double C_e = fmax(0.0, tentative_e_e - floor_ee) * m;
    const double C_i = fmax(0.0, tentative_e_i - floor_ei) * m;
    ee_after = (D_e > 0.0)
                   ? floor_ee
                   : fmax(floor_ee,
                          tentative_e_e - absorption_factor * C_e / m);
    ei_after = (D_i > 0.0)
                   ? floor_ei
                   : fmax(floor_ei,
                          tentative_e_i - absorption_factor * C_i / m);
  }

  ee[i] = ee_after;
  ei[i] = ei_after;
  deposited[i] =
      m * ((ee_after - ee_before) + (ei_after - ei_before));
}

double reduce_sum(const double* input,
                  const int n,
                  Ale1dVelocityProjectScratch& scratch,
                  cudaStream_t stream,
                  const char* label) {
  if (n <= 0) {
    return 0.0;
  }
  std::size_t temp_bytes = 0;
  cuda_check(cub::DeviceReduce::Sum(nullptr, temp_bytes, input,
                                    scratch.reduce_out.data(), n, stream),
             label);
  if (scratch.reduce_temp.size() < temp_bytes) {
    scratch.reduce_temp.resize(temp_bytes);
  }
  cuda_check(cub::DeviceReduce::Sum(scratch.reduce_temp.data(), temp_bytes, input,
                                    scratch.reduce_out.data(), n, stream),
             label);
  double host = 0.0;
  cuda_check(cudaMemcpyAsync(&host, scratch.reduce_out.data(), sizeof(double),
                             cudaMemcpyDeviceToHost, stream),
             label);
  cuda_check(cudaStreamSynchronize(stream), label);
  return host;
}

}  // namespace

void Ale1dVelocityProjectScratch::resize(const int n_cells) {
  TENRYU_ASSERT(n_cells >= 0,
                "ALE1D velocity projection n_cells must be nonnegative");
  const auto n = static_cast<std::size_t>(n_cells);
  p_old_cell.resize(n);
  p_new_cell.resize(n);
  u_new_cell.resize(n);
  v_new_node.resize(n + 1U);
  psi_left.resize(n);
  psi_right.resize(n);
  slope_left.resize(n);
  slope_right.resize(n);
  ke_node_old.resize(n + 1U);
  ke_node_new.resize(n + 1U);
  deficit.resize(n);
  capacity.resize(n);
  deposited.resize(n);
  reduce_out.resize(1U);
}

bool Ale1dVelocityProjectScratch::size_matches(const int n_cells) const {
  if (n_cells < 0) {
    return false;
  }
  const auto n = static_cast<std::size_t>(n_cells);
  return p_old_cell.size() == n && p_new_cell.size() == n &&
         u_new_cell.size() == n && v_new_node.size() == n + 1U &&
         psi_left.size() == n && psi_right.size() == n && slope_left.size() == n &&
         slope_right.size() == n && ke_node_old.size() == n + 1U &&
         ke_node_new.size() == n + 1U && deficit.size() == n && capacity.size() == n &&
         deposited.size() == n && reduce_out.size() == 1U;
}

Ale1dVelocityProjectResult project_velocity(
    const core::State& state,
    const double* mass_new,
    const double* mass_flux,
    const double* phi_face,
    const double limiter_theta,
    const bool ke_conservation_closure,
    const bool two_temperature,
    const double* ke_remap,
    double* ee_new,
    double* ei_new,
    Ale1dVelocityProjectScratch& scratch) {
  Ale1dVelocityProjectResult result;
  const int n = static_cast<int>(scratch.p_old_cell.size());
  if (n <= 0 || mass_new == nullptr || mass_flux == nullptr || phi_face == nullptr) {
    return result;
  }
  TENRYU_ASSERT(scratch.size_matches(n),
                "ALE1D velocity projection scratch size mismatch");
  TENRYU_ASSERT(state.x_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D velocity projection requires n_cells+1 old nodes");
  TENRYU_ASSERT(state.v_r.size() >= static_cast<std::size_t>(n + 1),
                "ALE1D velocity projection requires n_cells+1 velocities");
  TENRYU_ASSERT(state.mass.size() >= static_cast<std::size_t>(n),
                "ALE1D velocity projection requires old mass");
  if (ke_conservation_closure) {
    TENRYU_ASSERT(ke_remap != nullptr,
                  "ALE1D velocity projection requires remapped kinetic energy");
    TENRYU_ASSERT(ee_new != nullptr && ei_new != nullptr,
                  "ALE1D velocity projection requires candidate internal energy");
  }

  cudaStream_t stream = nullptr;
  half_index_shift_slopes_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      state.v_r.data(), state.mass.data(), phi_face, scratch.slope_left.data(),
      scratch.slope_right.data(), n, limiter_theta);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection slope launch failed");
  half_index_shift_remap_kernel<<<blocks_for(n), kBlockSize, 0, stream>>>(
      state.v_r.data(), state.mass.data(), mass_new, mass_flux, phi_face,
      scratch.slope_left.data(), scratch.slope_right.data(), scratch.psi_left.data(),
      scratch.psi_right.data(), scratch.p_old_cell.data(), scratch.p_new_cell.data(),
      scratch.u_new_cell.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection remap launch failed");
  half_index_shift_node_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      mass_new, scratch.psi_left.data(), scratch.psi_right.data(),
      scratch.v_new_node.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection node launch failed");

  if (ke_conservation_closure) {
    // Deposit into the remapped-but-uncommitted candidate. This moves the KE
    // discrepancy into internal energy before diagnostics, so
    // global_total_energy_rel_err retains its KE + internal + radiation meaning.
    constexpr double kClosureSpecificEnergyFloor = 0.0;
    compute_kinetic_closure_deficit_capacity_kernel
        <<<blocks_for(n), kBlockSize, 0, stream>>>(
            scratch.deficit.data(),
            scratch.capacity.data(),
            ee_new,
            ei_new,
            ke_remap,
            mass_new,
            scratch.v_new_node.data(),
            kClosureSpecificEnergyFloor,
            n,
            two_temperature ? 1 : 0);
    cuda_check(cudaGetLastError(),
               "ALE1D velocity projection KE closure deficit/capacity launch "
               "failed");
    const double global_deficit = reduce_sum(
        scratch.deficit.data(), n, scratch, stream,
        "ALE1D velocity projection KE closure deficit reduction failed");
    const double global_capacity = reduce_sum(
        scratch.capacity.data(), n, scratch, stream,
        "ALE1D velocity projection KE closure capacity reduction failed");

    double absorption_factor = 0.0;
    if (global_deficit > 0.0 && global_capacity > 0.0) {
      absorption_factor = std::min(1.0, global_deficit / global_capacity);
    }

    apply_kinetic_closure_redistribution_kernel
        <<<blocks_for(n), kBlockSize, 0, stream>>>(
            ee_new,
            ei_new,
            scratch.deposited.data(),
            ke_remap,
            mass_new,
            scratch.v_new_node.data(),
            kClosureSpecificEnergyFloor,
            absorption_factor,
            n,
            two_temperature ? 1 : 0);
    cuda_check(cudaGetLastError(),
               "ALE1D velocity projection KE closure launch failed");
    result.ke_closure_deposited = reduce_sum(
        scratch.deposited.data(), n, scratch, stream,
        "ALE1D velocity projection KE closure reduction failed");
  }

  kinetic_energy_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      state.mass.data(), state.v_r.data(), scratch.ke_node_old.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection old KE launch failed");
  kinetic_energy_kernel<<<blocks_for(n + 1), kBlockSize, 0, stream>>>(
      mass_new, scratch.v_new_node.data(), scratch.ke_node_new.data(), n);
  cuda_check(cudaGetLastError(),
             "ALE1D velocity projection new KE launch failed");

  result.kinetic_energy_old = reduce_sum(
      scratch.ke_node_old.data(), n + 1, scratch, stream,
      "ALE1D velocity projection old KE reduction failed");
  result.kinetic_energy_new = reduce_sum(
      scratch.ke_node_new.data(), n + 1, scratch, stream,
      "ALE1D velocity projection new KE reduction failed");
  if (ke_conservation_closure) {
    static int closure_dump_count = 0;
    if (closure_dump_enabled() && closure_dump_count < 2) {
      const int invocation = ++closure_dump_count;
      std::vector<double> ke_remap_host(static_cast<std::size_t>(n));
      std::vector<double> deposited_host(static_cast<std::size_t>(n));
      std::vector<double> ee_new_host(static_cast<std::size_t>(n));
      std::vector<double> ei_new_host(static_cast<std::size_t>(n));
      std::vector<double> mass_new_host(static_cast<std::size_t>(n));
      std::vector<double> v_new_host(static_cast<std::size_t>(n + 1));
      cuda_check(cudaMemcpyAsync(ke_remap_host.data(),
                                 ke_remap,
                                 static_cast<std::size_t>(n) * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "ALE1D closure dump ke_remap download failed");
      cuda_check(cudaMemcpyAsync(deposited_host.data(),
                                 scratch.deposited.data(),
                                 static_cast<std::size_t>(n) * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "ALE1D closure dump deposited download failed");
      cuda_check(cudaMemcpyAsync(ee_new_host.data(),
                                 ee_new,
                                 static_cast<std::size_t>(n) * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "ALE1D closure dump ee_new download failed");
      cuda_check(cudaMemcpyAsync(ei_new_host.data(),
                                 ei_new,
                                 static_cast<std::size_t>(n) * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "ALE1D closure dump ei_new download failed");
      cuda_check(cudaMemcpyAsync(mass_new_host.data(),
                                 mass_new,
                                 static_cast<std::size_t>(n) * sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "ALE1D closure dump mass_new download failed");
      cuda_check(cudaMemcpyAsync(v_new_host.data(),
                                 scratch.v_new_node.data(),
                                 static_cast<std::size_t>(n + 1) *
                                     sizeof(double),
                                 cudaMemcpyDeviceToHost,
                                 stream),
                 "ALE1D closure dump v_new download failed");
      cuda_check(cudaStreamSynchronize(stream),
                 "ALE1D closure dump synchronization failed");

      double ke_remap_total = 0.0;
      double ke_new_cell_total = 0.0;
      for (int i = 0; i < n; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        ke_remap_total += ke_remap_host[idx];
        ke_new_cell_total +=
            0.25 * mass_new_host[idx] *
            (v_new_host[idx] * v_new_host[idx] +
             v_new_host[idx + 1U] * v_new_host[idx + 1U]);
      }

      std::ostringstream totals;
      totals << std::scientific << std::setprecision(6)
             << "[ale1d-ke-closure-dump] invocation=" << invocation
             << " KE_old=" << result.kinetic_energy_old
             << " KE_new=" << result.kinetic_energy_new
             << " deposited_total=" << result.ke_closure_deposited
             << " ke_remap_total=" << ke_remap_total
             << " KE_new_cell_total=" << ke_new_cell_total;
      core::log_info(totals.str());

      std::vector<int> order(static_cast<std::size_t>(n));
      std::iota(order.begin(), order.end(), 0);
      const int top_count = std::min(n, 5);
      std::partial_sort(
          order.begin(),
          order.begin() + top_count,
          order.end(),
          [&deposited_host](const int lhs, const int rhs) {
            const double lhs_abs =
                std::abs(deposited_host[static_cast<std::size_t>(lhs)]);
            const double rhs_abs =
                std::abs(deposited_host[static_cast<std::size_t>(rhs)]);
            return lhs_abs == rhs_abs ? lhs < rhs : lhs_abs > rhs_abs;
          });
      for (int rank = 0; rank < top_count; ++rank) {
        const int i = order[static_cast<std::size_t>(rank)];
        const auto idx = static_cast<std::size_t>(i);
        const double kinetic_new_cell =
            0.25 * mass_new_host[idx] *
            (v_new_host[idx] * v_new_host[idx] +
             v_new_host[idx + 1U] * v_new_host[idx + 1U]);
        std::ostringstream cell;
        cell << std::scientific << std::setprecision(6)
             << "[ale1d-ke-closure-dump-cell] invocation=" << invocation
             << " index=" << i
             << " ke_remap=" << ke_remap_host[idx]
             << " K_new_cell=" << kinetic_new_cell
             << " deposited=" << deposited_host[idx]
             << " ee_new=" << ee_new_host[idx]
             << " ei_new=" << ei_new_host[idx]
             << " mass_new=" << mass_new_host[idx];
        core::log_info(cell.str());
      }
    }
    const double diff = std::abs(result.kinetic_energy_new +
                                 result.ke_closure_deposited -
                                 result.kinetic_energy_old);
    result.kinetic_energy_drift_rel =
        diff / std::max(result.kinetic_energy_old, kTiny);
  } else {
    const double diff =
        std::abs(result.kinetic_energy_new - result.kinetic_energy_old);
    result.kinetic_energy_drift_rel =
        result.kinetic_energy_old > kTiny ? diff / result.kinetic_energy_old
                                          : diff;
  }
  result.success = true;
  return result;
}

}  // namespace tenryu::hydro::ale1d
