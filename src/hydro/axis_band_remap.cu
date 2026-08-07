#include "hydro/axis_band_remap.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "burn/burn_constants.hpp"
#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/rz_quad_volume.cuh"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::ale {
namespace {

constexpr double kTiny = 1.0e-300;
constexpr double kFluxTiny = 1.0e-30;
constexpr double kWarnRel = 1.0e-8;
// Hard-rollback threshold for conservation residuals (k04 R14). The band
// remap is a serial host computation, so residuals are deterministic and
// roundoff-level (<< 1e-8) unless a genuine defect is present.
constexpr double kConsFailRel = 1.0e-8;

struct DeviceBuffer {
  ~DeviceBuffer() {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
  }

  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  bool reset(const std::size_t count) {
    size = count;
    if (count == 0) {
      return true;
    }
    return cudaMalloc(reinterpret_cast<void**>(&data),
                      count * sizeof(double)) == cudaSuccess;
  }

  bool copy_from_host(const std::vector<double>& src) const {
    if (size == 0) {
      return true;
    }
    return src.size() == size &&
           cudaMemcpy(data, src.data(), size * sizeof(double),
                      cudaMemcpyHostToDevice) == cudaSuccess;
  }

  double* data = nullptr;
  std::size_t size = 0;
};

template <typename T, typename = void>
struct HasAxisBandRadiationFlag : std::false_type {};

template <typename T>
struct HasAxisBandRadiationFlag<
    T,
    std::void_t<decltype(
        std::declval<const T&>()
            .numerics.ale.axis_band_managed_remap_include_radiation_groups)>>
    : std::true_type {};

template <typename T>
bool include_radiation_groups_impl(const T& cfg, std::true_type) {
  return cfg.numerics.ale.axis_band_managed_remap_include_radiation_groups;
}

template <typename T>
bool include_radiation_groups_impl(const T&, std::false_type) {
  return true;
}

bool include_radiation_groups(const core::Config& cfg) {
  return include_radiation_groups_impl(
      cfg, HasAxisBandRadiationFlag<core::Config>{});
}

std::size_t checked_count(const int a, const int b, const char* what) {
  TENRYU_ASSERT(a >= 0 && b >= 0, what);
  return static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
}

std::size_t node_idx(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * (nz + 1) + j);
}

std::size_t cell_idx(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * nz + j);
}

double clamp01_host(const double x) {
  if (!std::isfinite(x)) {
    return 0.0;
  }
  return std::min(1.0, std::max(0.0, x));
}

template <typename Field>
std::vector<double> copy_host_field(const Field& field, const char* name) {
  std::vector<double> out;
  field.copy_to_host(out);
  TENRYU_ASSERT(out.size() == field.size(),
                (std::string("axis band remap copy failed: ") + name).c_str());
  return out;
}

double rz_signed_quad_area(const double r0,
                           const double z0,
                           const double r1,
                           const double z1,
                           const double r2,
                           const double z2,
                           const double r3,
                           const double z3) {
  return 0.5 * ((r0 * z1 - r1 * z0) + (r1 * z2 - r2 * z1) +
                (r2 * z3 - r3 * z2) + (r3 * z0 - r0 * z3));
}

template <bool FixedSign>
double swept_volume_r_face_t(const std::vector<double>& r_old,
                             const std::vector<double>& z_old,
                             const std::vector<double>& r_new,
                             const std::vector<double>& z_new,
                             const int i_face,
                             const int j,
                             const int nz) {
  const std::size_t n0 = node_idx(i_face, j, nz);
  const std::size_t n1 = node_idx(i_face, j + 1, nz);
  const double raw = detail::rz_signed_quad_volume(
      r_old[n0], z_old[n0], r_new[n0], z_new[n0], r_new[n1], z_new[n1],
      r_old[n1], z_old[n1]);
  return FixedSign ? -raw : raw;
}

template <bool FixedSign>
double swept_volume_z_face_t(const std::vector<double>& r_old,
                             const std::vector<double>& z_old,
                             const std::vector<double>& r_new,
                             const std::vector<double>& z_new,
                             const int i,
                             const int j_face,
                             const int nz) {
  const std::size_t n0 = node_idx(i, j_face, nz);
  const std::size_t n1 = node_idx(i + 1, j_face, nz);
  const double raw = detail::rz_signed_quad_volume(
      r_old[n0], z_old[n0], r_old[n1], z_old[n1], r_new[n1], z_new[n1],
      r_new[n0], z_new[n0]);
  return FixedSign ? -raw : raw;
}

double van_leer(const double dl, const double dr) {
  if (!std::isfinite(dl) || !std::isfinite(dr) || dl * dr <= 0.0) {
    return 0.0;
  }
  const double denom = dl + dr;
  if (std::abs(denom) < kFluxTiny) {
    return 0.0;
  }
  return 2.0 * dl * dr / denom;
}

double intensive_at(const std::vector<double>& q,
                    const std::vector<double>& vol,
                    const int i,
                    const int j,
                    const int nz) {
  const std::size_t c = cell_idx(i, j, nz);
  return (vol[c] != 0.0) ? (q[c] / vol[c]) : 0.0;
}

double limited_face_intensive_r(const std::vector<double>& q,
                                const std::vector<double>& vol,
                                const int donor_i,
                                const int j,
                                const int K,
                                const int nz,
                                const bool has_physical_rz_axis,
                                const bool right_face) {
  const double phi_d = intensive_at(q, vol, donor_i, j, nz);
  double slope = 0.0;
  if (K > 1) {
    if (donor_i == 0) {
      slope = has_physical_rz_axis ? (intensive_at(q, vol, 1, j, nz) - phi_d)
                                   : 0.0;
    } else if (donor_i == K - 1) {
      slope = phi_d - intensive_at(q, vol, K - 2, j, nz);
    } else {
      slope = van_leer(phi_d - intensive_at(q, vol, donor_i - 1, j, nz),
                       intensive_at(q, vol, donor_i + 1, j, nz) - phi_d);
    }
  }
  return phi_d + (right_face ? 0.5 : -0.5) * slope;
}

double limited_face_intensive_z(const std::vector<double>& q,
                                const std::vector<double>& vol,
                                const int i,
                                const int donor_j,
                                const int nz,
                                const bool top_face) {
  const double phi_d = intensive_at(q, vol, i, donor_j, nz);
  double slope = 0.0;
  if (nz > 1) {
    if (donor_j == 0) {
      slope = intensive_at(q, vol, i, 1, nz) - phi_d;
    } else if (donor_j == nz - 1) {
      slope = phi_d - intensive_at(q, vol, i, nz - 2, nz);
    } else {
      slope = van_leer(phi_d - intensive_at(q, vol, i, donor_j - 1, nz),
                       intensive_at(q, vol, i, donor_j + 1, nz) - phi_d);
    }
  }
  return phi_d + (top_face ? 0.5 : -0.5) * slope;
}

void apply_r_sweep_to_field(std::vector<double>& q,
                            const std::vector<double>& vol_old,
                            const std::vector<double>& dV_faces,
                            const int K,
                            const int nz,
                            const bool has_physical_rz_axis,
                            const bool donor_sign_fixed) {
  std::vector<double> q_new = q;
  for (int j = 0; j < nz; ++j) {
    for (int i_face = 1; i_face < K; ++i_face) {
      const double dV =
          dV_faces[static_cast<std::size_t>(i_face * nz + j)];
      if (std::abs(dV) < kFluxTiny) {
        continue;
      }
      const int i_left = i_face - 1;
      const int i_right = i_face;
      if (donor_sign_fixed) {
        const int donor_i = (dV > 0.0) ? i_left : i_right;
        const bool donor_right_face = (dV > 0.0);
        const double q_face = limited_face_intensive_r(q, vol_old, donor_i, j,
                                                       K, nz,
                                                       has_physical_rz_axis,
                                                       donor_right_face);
        const double flux = q_face * dV;
        q_new[cell_idx(i_left, j, nz)] -= flux;
        q_new[cell_idx(i_right, j, nz)] += flux;
      } else {
        const int donor_i = (dV > 0.0) ? i_right : i_left;
        const bool donor_right_face = !(dV > 0.0);
        const double q_face = limited_face_intensive_r(q, vol_old, donor_i, j,
                                                       K, nz,
                                                       has_physical_rz_axis,
                                                       donor_right_face);
        const double flux = -q_face * dV;
        q_new[cell_idx(i_left, j, nz)] -= flux;
        q_new[cell_idx(i_right, j, nz)] += flux;
      }
    }
  }
  q.swap(q_new);
}

void apply_z_sweep_to_field(std::vector<double>& q,
                            const std::vector<double>& vol_old,
                            const std::vector<double>& dV_faces,
                            const int K,
                            const int nz,
                            const bool donor_sign_fixed) {
  std::vector<double> q_new = q;
  for (int i = 0; i < K; ++i) {
    for (int j_face = 1; j_face < nz; ++j_face) {
      const double dV =
          dV_faces[static_cast<std::size_t>(i * (nz + 1) + j_face)];
      if (std::abs(dV) < kFluxTiny) {
        continue;
      }
      const int j_low = j_face - 1;
      const int j_high = j_face;
      if (donor_sign_fixed) {
        const int donor_j = (dV > 0.0) ? j_low : j_high;
        const bool donor_top_face = (dV > 0.0);
        const double q_face = limited_face_intensive_z(q, vol_old, i, donor_j,
                                                       nz, donor_top_face);
        const double flux = q_face * dV;
        q_new[cell_idx(i, j_low, nz)] -= flux;
        q_new[cell_idx(i, j_high, nz)] += flux;
      } else {
        const int donor_j = (dV > 0.0) ? j_high : j_low;
        const bool donor_top_face = !(dV > 0.0);
        const double q_face = limited_face_intensive_z(q, vol_old, i, donor_j,
                                                       nz, donor_top_face);
        const double flux = -q_face * dV;
        q_new[cell_idx(i, j_low, nz)] -= flux;
        q_new[cell_idx(i, j_high, nz)] += flux;
      }
    }
  }
  q.swap(q_new);
}

void apply_r_sweep_to_fields(std::vector<double*>& fields,
                             const std::vector<double>& vol_old,
                             const std::vector<double>& dV_faces,
                             const int K,
                             const int nz,
                             const bool has_physical_rz_axis,
                             const bool donor_sign_fixed) {
  const std::size_t band_cells =
      static_cast<std::size_t>(K) * static_cast<std::size_t>(nz);
  for (double* field : fields) {
    std::vector<double> q(field, field + band_cells);
    apply_r_sweep_to_field(q, vol_old, dV_faces, K, nz,
                           has_physical_rz_axis, donor_sign_fixed);
    std::copy(q.begin(), q.end(), field);
  }
}

void apply_z_sweep_to_fields(std::vector<double*>& fields,
                             const std::vector<double>& vol_old,
                             const std::vector<double>& dV_faces,
                             const int K,
                             const int nz,
                             const bool donor_sign_fixed) {
  const std::size_t band_cells =
      static_cast<std::size_t>(K) * static_cast<std::size_t>(nz);
  for (double* field : fields) {
    std::vector<double> q(field, field + band_cells);
    apply_z_sweep_to_field(q, vol_old, dV_faces, K, nz, donor_sign_fixed);
    std::copy(q.begin(), q.end(), field);
  }
}

double finite_positive_min(const std::vector<double>& v) {
  double out = std::numeric_limits<double>::infinity();
  for (const double x : v) {
    if (!std::isfinite(x)) {
      return -std::numeric_limits<double>::infinity();
    }
    out = std::min(out, x);
  }
  return out;
}

double finite_min(const std::vector<double>& v) {
  double out = std::numeric_limits<double>::infinity();
  for (const double x : v) {
    if (!std::isfinite(x)) {
      return -std::numeric_limits<double>::infinity();
    }
    out = std::min(out, x);
  }
  return out;
}

double min_specific_internal(const std::vector<double>& mass,
                             const std::vector<double>& e_e,
                             const std::vector<double>& e_i,
                             const bool two_temperature) {
  double out = std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < mass.size(); ++c) {
    if (!std::isfinite(mass[c]) || !(mass[c] > 0.0) ||
        !std::isfinite(e_e[c]) ||
        (two_temperature && !std::isfinite(e_i[c]))) {
      return -std::numeric_limits<double>::infinity();
    }
    double e_specific = e_e[c] / mass[c];
    if (two_temperature) {
      e_specific = std::min(e_specific, e_i[c] / mass[c]);
    }
    out = std::min(out, e_specific);
  }
  return out;
}

double sum_vector(const std::vector<double>& v) {
  double out = 0.0;
  for (const double x : v) {
    out += x;
  }
  return out;
}

double reduced_sum(const parallel::Reduction* reduction, const double local) {
  return (reduction != nullptr) ? reduction->allreduce_sum(local) : local;
}

double reduced_min(const parallel::Reduction* reduction, const double local) {
  return (reduction != nullptr) ? reduction->allreduce_min(local) : local;
}

double reduced_max(const parallel::Reduction* reduction, const double local) {
  return (reduction != nullptr) ? reduction->allreduce_max(local) : local;
}

double rel_delta(const double after, const double before) {
  const double denom = std::max(std::abs(before), kTiny);
  return (after - before) / denom;
}

struct Projection {
  std::vector<double> v_r_node;
  std::vector<double> v_z_node;
  double kinetic = 0.0;
};

void compute_corner_mass(const std::vector<double>& mass,
                         const std::vector<double>& r_nodes,
                         const int K,
                         const int nz,
                         std::vector<double>& corner_mass) {
  const std::size_t band_cells =
      static_cast<std::size_t>(K) * static_cast<std::size_t>(nz);
  corner_mass.assign(band_cells * 4U, 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = cell_idx(i, j, nz);
      const double m_cell = std::max(mass[c], 0.0);
      const double R_L =
          0.5 * (r_nodes[node_idx(i, j, nz)] +
                 r_nodes[node_idx(i, j + 1, nz)]);
      const double R_R =
          0.5 * (r_nodes[node_idx(i + 1, j, nz)] +
                 r_nodes[node_idx(i + 1, j + 1, nz)]);
      const double denom = (R_L + R_R) * 6.0;
      if (!(std::isfinite(denom) && denom > 0.0)) {
        const double q = 0.25 * m_cell;
        corner_mass[c * 4U + 0U] = q;
        corner_mass[c * 4U + 1U] = q;
        corner_mass[c * 4U + 2U] = q;
        corner_mass[c * 4U + 3U] = q;
        continue;
      }
      const double w_L = (2.0 * R_L + R_R) / denom;
      const double w_R = (R_L + 2.0 * R_R) / denom;
      corner_mass[c * 4U + 0U] = w_L * m_cell;
      corner_mass[c * 4U + 1U] = w_R * m_cell;
      corner_mass[c * 4U + 2U] = w_R * m_cell;
      corner_mass[c * 4U + 3U] = w_L * m_cell;
    }
  }
}

double node_kinetic(const std::vector<double>& corner_mass,
                    const std::vector<double>& v_r_node,
                    const std::vector<double>& v_z_node,
                    const int K,
                    const int nz) {
  double out = 0.0;
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = cell_idx(i, j, nz);
      const std::size_t n00 = node_idx(i, j, nz);
      const std::size_t n10 = node_idx(i + 1, j, nz);
      const std::size_t n11 = node_idx(i + 1, j + 1, nz);
      const std::size_t n01 = node_idx(i, j + 1, nz);
      const std::size_t base = c * 4U;
      const std::size_t nodes[4] = {n00, n10, n11, n01};
      for (int q = 0; q < 4; ++q) {
        const double vr = v_r_node[nodes[q]];
        const double vz = v_z_node[nodes[q]];
        out += 0.5 * corner_mass[base + static_cast<std::size_t>(q)] *
               (vr * vr + vz * vz);
      }
    }
  }
  return out;
}

Projection project_cell_momentum_to_nodes(const std::vector<double>& mass,
                                          const std::vector<double>& mom_r,
                                          const std::vector<double>& mom_z,
                                          const std::vector<double>& r_nodes,
                                          const int K,
                                          const int nz) {
  const std::size_t band_nodes =
      static_cast<std::size_t>(K + 1) * static_cast<std::size_t>(nz + 1);
  Projection out;
  out.v_r_node.assign(band_nodes, 0.0);
  out.v_z_node.assign(band_nodes, 0.0);

  std::vector<double> corner_mass;
  compute_corner_mass(mass, r_nodes, K, nz, corner_mass);

  std::vector<double> v_r_cell(mass.size(), 0.0);
  std::vector<double> v_z_cell(mass.size(), 0.0);
  for (std::size_t c = 0; c < mass.size(); ++c) {
    if (std::isfinite(mass[c]) && mass[c] > 0.0) {
      v_r_cell[c] = mom_r[c] / mass[c];
      v_z_cell[c] = mom_z[c] / mass[c];
    }
  }

  for (int i = 0; i <= K; ++i) {
    for (int j = 0; j <= nz; ++j) {
      double m_sum = 0.0;
      double pr_sum = 0.0;
      double pz_sum = 0.0;
      for (int di = -1; di <= 0; ++di) {
        for (int dj = -1; dj <= 0; ++dj) {
          const int ic = i + di;
          const int jc = j + dj;
          if (ic < 0 || ic >= K || jc < 0 || jc >= nz) {
            continue;
          }
          const std::size_t c = cell_idx(ic, jc, nz);
          int corner = 3;
          if (di == 0 && dj == 0) {
            corner = 0;
          } else if (di == -1 && dj == 0) {
            corner = 1;
          } else if (di == -1 && dj == -1) {
            corner = 2;
          }
          const double m =
              std::max(corner_mass[c * 4U + static_cast<std::size_t>(corner)],
                       0.0);
          m_sum += m;
          pr_sum += m * v_r_cell[c];
          pz_sum += m * v_z_cell[c];
        }
      }
      const std::size_t n = node_idx(i, j, nz);
      if (m_sum > 0.0) {
        out.v_r_node[n] = pr_sum / m_sum;
        out.v_z_node[n] = pz_sum / m_sum;
      }
    }
  }

  out.kinetic = node_kinetic(corner_mass, out.v_r_node, out.v_z_node, K, nz);
  return out;
}

double source_node_kinetic(const std::vector<double>& mass,
                           const std::vector<double>& r_nodes,
                           const std::vector<double>& v_r_nodes,
                           const std::vector<double>& v_z_nodes,
                           const int K,
                           const int nz) {
  std::vector<double> corner_mass;
  compute_corner_mass(mass, r_nodes, K, nz, corner_mass);
  return node_kinetic(corner_mass, v_r_nodes, v_z_nodes, K, nz);
}

void log_conservation_warning(const AxisBandRemapResult& result) {
  const double max_abs =
      std::max({std::abs(result.mass_delta_rel),
                std::abs(result.E_int_e_delta_rel),
                std::abs(result.E_int_i_delta_rel),
                std::abs(result.E_kin_delta_rel),
                std::abs(result.E_rad_delta_rel),
                std::abs(result.mom_r_delta_scaled),
                std::abs(result.mom_z_delta_scaled)});
  if (!(max_abs > kWarnRel)) {
    return;
  }
  std::ostringstream os;
  os << "axis band remap conservation diagnostic exceeded warning threshold: "
     << "mass=" << result.mass_delta_rel
     << " E_int_e=" << result.E_int_e_delta_rel
     << " E_int_i=" << result.E_int_i_delta_rel
     << " E_kin=" << result.E_kin_delta_rel
     << " E_rad=" << result.E_rad_delta_rel
     << " mom_r_scaled=" << result.mom_r_delta_scaled
     << " mom_z_scaled=" << result.mom_z_delta_scaled;
  core::log_warning(os.str());
}

bool sync_device_scratch(const DeviceBuffer& d_mass,
                         const DeviceBuffer& d_mom_r,
                         const DeviceBuffer& d_mom_z,
                         const DeviceBuffer& d_E_int_e,
                         const DeviceBuffer& d_E_int_i,
                         const DeviceBuffer& d_volfrac_mass,
                         const DeviceBuffer& d_rad_E_ext,
                         const DeviceBuffer& d_vol_intermediate,
                         const DeviceBuffer& d_vol_final,
                         const std::vector<double>& mass,
                         const std::vector<double>& mom_r,
                         const std::vector<double>& mom_z,
                         const std::vector<double>& E_int_e,
                         const std::vector<double>& E_int_i,
                         const std::vector<double>& volfrac_mass,
                         const std::vector<double>& rad_E_ext,
                         const std::vector<double>& vol_intermediate,
                         const std::vector<double>& vol_final) {
  return d_mass.copy_from_host(mass) &&
         d_mom_r.copy_from_host(mom_r) &&
         d_mom_z.copy_from_host(mom_z) &&
         d_E_int_e.copy_from_host(E_int_e) &&
         d_E_int_i.copy_from_host(E_int_i) &&
         d_volfrac_mass.copy_from_host(volfrac_mass) &&
         d_rad_E_ext.copy_from_host(rad_E_ext) &&
         d_vol_intermediate.copy_from_host(vol_intermediate) &&
         d_vol_final.copy_from_host(vol_final);
}

}  // namespace

AxisBandRemapResult apply_axis_band_remap(
    core::State& state,
    const core::Config& cfg,
    const int K,
    const parallel::Reduction* reduction) {
  AxisBandRemapResult result;
  result.K = K;

  TENRYU_ASSERT(state.mesh.dim == 2, "axis band remap requires 2D mesh");
  TENRYU_ASSERT(K >= 1, "axis band remap requires K >= 1");
  TENRYU_ASSERT(K <= state.mesh.topo.nr,
                "axis band remap requires K <= mesh nr");
  TENRYU_ASSERT(state.mesh.topo.nz > 0,
                "axis band remap requires mesh nz > 0");

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const std::size_t n_cells =
      checked_count(nr, nz, "axis band remap invalid cell count");
  const std::size_t n_nodes =
      checked_count(nr + 1, nz + 1, "axis band remap invalid node count");
  const std::size_t band_cells =
      checked_count(K, nz, "axis band remap invalid band cell count");
  const std::size_t band_nodes =
      checked_count(K + 1, nz + 1, "axis band remap invalid band node count");

  TENRYU_ASSERT(state.x_r.size() >= n_nodes &&
                    state.x_z.size() >= n_nodes &&
                    state.v_r.size() >= n_nodes &&
                    state.v_z.size() >= n_nodes,
                "axis band remap node field size mismatch");
  TENRYU_ASSERT(state.rho.size() >= n_cells &&
                    state.mass.size() >= n_cells &&
                    state.ee.size() >= n_cells &&
                    state.ei.size() >= n_cells &&
                    state.vol.size() >= n_cells,
                "axis band remap cell field size mismatch");

  const int n_materials =
      state.volFrac.empty()
          ? 0
          : static_cast<int>(state.volFrac.size() / n_cells);
  TENRYU_ASSERT(state.volFrac.empty() || state.volFrac.size() % n_cells == 0,
                "axis band remap volFrac size must be n_cells*n_materials");

  const bool include_radiation =
      include_radiation_groups(cfg) && !state.rad_E.empty();
  const int n_groups =
      include_radiation ? static_cast<int>(state.rad_E.size() / n_cells) : 0;
  TENRYU_ASSERT(!include_radiation || state.rad_E.size() % n_cells == 0,
                "axis band remap rad_E size must be n_cells*n_groups");
  const bool swept_volume_sign_fixed =
      cfg.numerics.ale.swept_volume_sign_fixed;
  const bool two_temperature = cfg.main.two_temperature;
  const bool remap_gas_tracer =
      state.gas_tracer_initialized && state.gas_tracer_Y.size() == n_cells;
  const bool remap_burn_species =
      cfg.burn.enabled &&
      state.burn_n_host.size() ==
          n_cells * static_cast<std::size_t>(tenryu::burn::kNumSpecies);
  const bool remap_hot_e_eps =
      !state.hot_e_eps_cum_host.empty() &&
      state.hot_e_eps_cum_host.size() == n_cells;
  const bool remap_burn_eps =
      !state.burn_eps_cum_host.empty() &&
      state.burn_eps_cum_host.size() == n_cells;

  std::vector<double> x_r = copy_host_field(state.x_r, "x_r");
  std::vector<double> x_z = copy_host_field(state.x_z, "x_z");
  std::vector<double> v_r = copy_host_field(state.v_r, "v_r");
  std::vector<double> v_z = copy_host_field(state.v_z, "v_z");
  std::vector<double> vol = copy_host_field(state.vol, "vol");
  std::vector<double> rho = copy_host_field(state.rho, "rho");
  std::vector<double> mass_field = copy_host_field(state.mass, "mass");
  std::vector<double> ee = copy_host_field(state.ee, "ee");
  std::vector<double> ei = copy_host_field(state.ei, "ei");
  std::vector<double> volFrac;
  std::vector<double> rad_E;
  std::vector<double> gas_tracer_Y;
  if (n_materials > 0) {
    volFrac = copy_host_field(state.volFrac, "volFrac");
  }
  if (n_groups > 0) {
    rad_E = copy_host_field(state.rad_E, "rad_E");
  }
  if (remap_gas_tracer) {
    gas_tracer_Y = copy_host_field(state.gas_tracer_Y, "gas_tracer_Y");
  }

  std::vector<double> r_old(band_nodes, 0.0);
  std::vector<double> z_old(band_nodes, 0.0);
  std::vector<double> vr_old(band_nodes, 0.0);
  std::vector<double> vz_old(band_nodes, 0.0);
  for (int i = 0; i <= K; ++i) {
    for (int j = 0; j <= nz; ++j) {
      const std::size_t nb = node_idx(i, j, nz);
      const std::size_t ns = node_idx(i, j, nz);
      r_old[nb] = x_r[ns];
      z_old[nb] = x_z[ns];
      vr_old[nb] = v_r[ns];
      vz_old[nb] = v_z[ns];
    }
  }

  bool target_invalid = false;
  std::vector<double> r_target(band_nodes, 0.0);
  std::vector<double> z_target(band_nodes, 0.0);
  for (int j = 0; j <= nz; ++j) {
    const double R_K = x_r[node_idx(K, j, nz)];
    const double Z_K = x_z[node_idx(K, j, nz)];
    if (!std::isfinite(R_K) || !(R_K > 0.0) || !std::isfinite(Z_K)) {
      target_invalid = true;
    }
    for (int i = 0; i <= K; ++i) {
      const double frac = static_cast<double>(i) / static_cast<double>(K);
      const std::size_t n = node_idx(i, j, nz);
      r_target[n] = R_K * std::sqrt(frac);
      z_target[n] = Z_K;
    }
  }
  for (int j = 0; j < nz; ++j) {
    const double dz =
        x_z[node_idx(K, j + 1, nz)] - x_z[node_idx(K, j, nz)];
    if (!std::isfinite(dz) || !(dz > 0.0)) {
      target_invalid = true;
    }
  }
  if (reduced_max(reduction, target_invalid ? 1.0 : 0.0) > 0.5) {
    result.failure = AxisBandRemapFailure::TargetMeshInfeasible;
    return result;
  }

  bool mesh_changed = false;
  for (std::size_t n = 0; n < band_nodes; ++n) {
    if (r_old[n] != r_target[n] || z_old[n] != z_target[n]) {
      mesh_changed = true;
      break;
    }
  }

  std::vector<double> cell_vol_old(band_cells, 0.0);
  std::vector<double> cell_mass(band_cells, 0.0);
  std::vector<double> cell_mom_r(band_cells, 0.0);
  std::vector<double> cell_mom_z(band_cells, 0.0);
  std::vector<double> cell_E_int_e(band_cells, 0.0);
  std::vector<double> cell_E_int_i(band_cells, 0.0);
  std::vector<double> cell_volFrac_mass(
      band_cells * static_cast<std::size_t>(n_materials), 0.0);
  std::vector<double> cell_rad_E_ext(
      band_cells * static_cast<std::size_t>(n_groups), 0.0);
  std::vector<double> cell_gas_tracer_mass(
      remap_gas_tracer ? band_cells : 0U, 0.0);
  std::vector<double> cell_burn_species_mass(
      remap_burn_species
          ? band_cells *
                static_cast<std::size_t>(tenryu::burn::kNumSpecies)
          : 0U,
      0.0);
  std::vector<double> cell_hot_e_eps_mass(
      remap_hot_e_eps ? band_cells : 0U, 0.0);
  std::vector<double> cell_burn_eps_mass(
      remap_burn_eps ? band_cells : 0U, 0.0);
  std::vector<double> cell_vol_intermediate(band_cells, 0.0);
  std::vector<double> cell_vol_final(band_cells, 0.0);

  DeviceBuffer d_cell_mass;
  DeviceBuffer d_cell_mom_r;
  DeviceBuffer d_cell_mom_z;
  DeviceBuffer d_cell_E_int_e;
  DeviceBuffer d_cell_E_int_i;
  DeviceBuffer d_cell_volFrac_mass;
  DeviceBuffer d_cell_rad_E_ext;
  DeviceBuffer d_cell_vol_intermediate;
  DeviceBuffer d_cell_vol_final;
  const bool scratch_alloc_ok =
      d_cell_mass.reset(band_cells) &&
      d_cell_mom_r.reset(band_cells) &&
      d_cell_mom_z.reset(band_cells) &&
      d_cell_E_int_e.reset(band_cells) &&
      d_cell_E_int_i.reset(band_cells) &&
      d_cell_volFrac_mass.reset(cell_volFrac_mass.size()) &&
      d_cell_rad_E_ext.reset(cell_rad_E_ext.size()) &&
      d_cell_vol_intermediate.reset(band_cells) &&
      d_cell_vol_final.reset(band_cells);
  if (!scratch_alloc_ok) {
    result.failure = AxisBandRemapFailure::ScratchAllocationFailed;
    return result;
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t b = cell_idx(i, j, nz);
      const std::size_t c = cell_idx(i, j, nz);
      const double V = vol[c];
      const double m = rho[c] * V;
      cell_vol_old[b] = V;
      cell_mass[b] = m;
      cell_E_int_e[b] = m * ee[c];
      cell_E_int_i[b] = m * ei[c];
      for (int mat = 0; mat < n_materials; ++mat) {
        cell_volFrac_mass[static_cast<std::size_t>(mat) * band_cells + b] =
            m * volFrac[c * static_cast<std::size_t>(n_materials) +
                        static_cast<std::size_t>(mat)];
      }
      for (int g = 0; g < n_groups; ++g) {
        cell_rad_E_ext[static_cast<std::size_t>(g) * band_cells + b] =
            rad_E[c * static_cast<std::size_t>(n_groups) +
                  static_cast<std::size_t>(g)] *
            V;
      }
      if (remap_gas_tracer) {
        cell_gas_tracer_mass[b] = m * clamp01_host(gas_tracer_Y[c]);
      }
      if (remap_burn_species) {
        for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
          cell_burn_species_mass[static_cast<std::size_t>(s) * band_cells + b] =
              m * std::fmax(
                      state.burn_n_host[
                          c * static_cast<std::size_t>(
                                  tenryu::burn::kNumSpecies) +
                          static_cast<std::size_t>(s)],
                      0.0);
        }
      }
      if (remap_hot_e_eps) {
        cell_hot_e_eps_mass[b] =
            m * std::fmax(state.hot_e_eps_cum_host[c], 0.0);
      }
      if (remap_burn_eps) {
        cell_burn_eps_mass[b] =
            m * std::fmax(state.burn_eps_cum_host[c], 0.0);
      }
    }
  }

  std::vector<double> corner_mass_old;
  compute_corner_mass(cell_mass, r_old, K, nz, corner_mass_old);
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t b = cell_idx(i, j, nz);
      const std::size_t base = b * 4U;
      const std::size_t n00 = node_idx(i, j, nz);
      const std::size_t n10 = node_idx(i + 1, j, nz);
      const std::size_t n11 = node_idx(i + 1, j + 1, nz);
      const std::size_t n01 = node_idx(i, j + 1, nz);
      const double m_sum = corner_mass_old[base + 0U] +
                           corner_mass_old[base + 1U] +
                           corner_mass_old[base + 2U] +
                           corner_mass_old[base + 3U];
      if (m_sum > 0.0) {
        const double vc_r =
            (corner_mass_old[base + 0U] * vr_old[n00] +
             corner_mass_old[base + 1U] * vr_old[n10] +
             corner_mass_old[base + 2U] * vr_old[n11] +
             corner_mass_old[base + 3U] * vr_old[n01]) /
            m_sum;
        const double vc_z =
            (corner_mass_old[base + 0U] * vz_old[n00] +
             corner_mass_old[base + 1U] * vz_old[n10] +
             corner_mass_old[base + 2U] * vz_old[n11] +
             corner_mass_old[base + 3U] * vz_old[n01]) /
            m_sum;
        cell_mom_r[b] = cell_mass[b] * vc_r;
        cell_mom_z[b] = cell_mass[b] * vc_z;
      }
    }
  }

  const double mass_before = reduced_sum(reduction, sum_vector(cell_mass));
  const double mom_r_before = reduced_sum(reduction, sum_vector(cell_mom_r));
  const double mom_z_before = reduced_sum(reduction, sum_vector(cell_mom_z));
  double v_scale_local = 0.0;
  for (std::size_t n = 0; n < band_nodes; ++n) {
    v_scale_local = std::max(
        v_scale_local,
        std::max(std::abs(vr_old[n]), std::abs(vz_old[n])));
  }
  const double v_scale = reduced_max(reduction, v_scale_local);
  const double e_e_before = reduced_sum(reduction, sum_vector(cell_E_int_e));
  const double e_i_before = reduced_sum(reduction, sum_vector(cell_E_int_i));
  const double e_rad_before =
      reduced_sum(reduction, sum_vector(cell_rad_E_ext));
  const double e_kin_before = reduced_sum(
      reduction, source_node_kinetic(cell_mass, r_old, vr_old, vz_old, K, nz));

  if (!sync_device_scratch(d_cell_mass, d_cell_mom_r, d_cell_mom_z,
                           d_cell_E_int_e, d_cell_E_int_i,
                           d_cell_volFrac_mass, d_cell_rad_E_ext,
                           d_cell_vol_intermediate, d_cell_vol_final,
                           cell_mass, cell_mom_r, cell_mom_z,
                           cell_E_int_e, cell_E_int_i,
                           cell_volFrac_mass, cell_rad_E_ext,
                           cell_vol_intermediate, cell_vol_final)) {
    result.failure = AxisBandRemapFailure::ScratchAllocationFailed;
    return result;
  }

  std::vector<double> r_mid = r_target;
  std::vector<double> z_mid = z_old;
  std::vector<double> dV_r_faces(
      static_cast<std::size_t>(K + 1) * static_cast<std::size_t>(nz), 0.0);
  for (int i_face = 0; i_face <= K; ++i_face) {
    for (int j = 0; j < nz; ++j) {
      dV_r_faces[static_cast<std::size_t>(i_face * nz + j)] =
          swept_volume_sign_fixed
              ? swept_volume_r_face_t<true>(
                    r_old, z_old, r_mid, z_mid, i_face, j, nz)
              : swept_volume_r_face_t<false>(
                    r_old, z_old, r_mid, z_mid, i_face, j, nz);
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < nz; ++j) {
      if (swept_volume_sign_fixed) {
        cell_vol_intermediate[cell_idx(i, j, nz)] =
            cell_vol_old[cell_idx(i, j, nz)] -
            dV_r_faces[static_cast<std::size_t>((i + 1) * nz + j)] +
            dV_r_faces[static_cast<std::size_t>(i * nz + j)];
      } else {
        cell_vol_intermediate[cell_idx(i, j, nz)] =
            cell_vol_old[cell_idx(i, j, nz)] +
            dV_r_faces[static_cast<std::size_t>((i + 1) * nz + j)] -
            dV_r_faces[static_cast<std::size_t>(i * nz + j)];
      }
    }
  }

  std::vector<double*> fields = {
      cell_mass.data(), cell_mom_r.data(), cell_mom_z.data(),
      cell_E_int_e.data(), cell_E_int_i.data()};
  for (int mat = 0; mat < n_materials; ++mat) {
    fields.push_back(cell_volFrac_mass.data() +
                     static_cast<std::size_t>(mat) * band_cells);
  }
  for (int g = 0; g < n_groups; ++g) {
    fields.push_back(cell_rad_E_ext.data() +
                     static_cast<std::size_t>(g) * band_cells);
  }
  if (remap_gas_tracer) {
    fields.push_back(cell_gas_tracer_mass.data());
  }
  if (remap_burn_species) {
    for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
      fields.push_back(cell_burn_species_mass.data() +
                       static_cast<std::size_t>(s) * band_cells);
    }
  }
  if (remap_hot_e_eps) {
    fields.push_back(cell_hot_e_eps_mass.data());
  }
  if (remap_burn_eps) {
    fields.push_back(cell_burn_eps_mass.data());
  }

  apply_r_sweep_to_fields(
      fields, cell_vol_old, dV_r_faces, K, nz,
      cfg.numerics.has_physical_rz_axis, swept_volume_sign_fixed);

  if (!sync_device_scratch(d_cell_mass, d_cell_mom_r, d_cell_mom_z,
                           d_cell_E_int_e, d_cell_E_int_i,
                           d_cell_volFrac_mass, d_cell_rad_E_ext,
                           d_cell_vol_intermediate, d_cell_vol_final,
                           cell_mass, cell_mom_r, cell_mom_z,
                           cell_E_int_e, cell_E_int_i,
                           cell_volFrac_mass, cell_rad_E_ext,
                           cell_vol_intermediate, cell_vol_final)) {
    result.failure = AxisBandRemapFailure::ScratchAllocationFailed;
    return result;
  }

  result.min_v_intermediate =
      reduced_min(reduction, finite_positive_min(cell_vol_intermediate));
  result.min_cell_mass = reduced_min(reduction, finite_positive_min(cell_mass));
  const double min_E_e_r = finite_positive_min(cell_E_int_e);
  const double min_E_i_r = finite_positive_min(cell_E_int_i);
  const double min_E_r =
      reduced_min(reduction,
                  two_temperature ? std::min(min_E_e_r, min_E_i_r) : min_E_e_r);
  const double min_E_i_r_1t =
      two_temperature ? 0.0 : reduced_min(reduction, min_E_i_r);
  if (!(result.min_v_intermediate > 0.0)) {
    result.failure = AxisBandRemapFailure::IntermediatePositivityViolation;
    return result;
  }
  if (!(result.min_cell_mass > 0.0)) {
    result.failure = AxisBandRemapFailure::RemappedMassNegative;
    return result;
  }
  if (!(min_E_r > 0.0) ||
      (!two_temperature && !(min_E_i_r_1t >= 0.0))) {
    result.failure = AxisBandRemapFailure::RemappedInternalEnergyNegative;
    return result;
  }
  if (n_groups > 0) {
    const double min_rad_r =
        reduced_min(reduction, finite_min(cell_rad_E_ext));
    if (!(min_rad_r >= 0.0)) {
      result.failure = AxisBandRemapFailure::RemappedRadiationNegative;
      return result;
    }
  }

  std::vector<double> dV_z_faces(
      static_cast<std::size_t>(K) * static_cast<std::size_t>(nz + 1), 0.0);
  for (int i = 0; i < K; ++i) {
    for (int j_face = 0; j_face <= nz; ++j_face) {
      dV_z_faces[static_cast<std::size_t>(i * (nz + 1) + j_face)] =
          swept_volume_sign_fixed
              ? swept_volume_z_face_t<true>(
                    r_mid, z_mid, r_target, z_target, i, j_face, nz)
              : swept_volume_z_face_t<false>(
                    r_mid, z_mid, r_target, z_target, i, j_face, nz);
    }
  }

  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < nz; ++j) {
      if (swept_volume_sign_fixed) {
        cell_vol_final[cell_idx(i, j, nz)] =
            cell_vol_intermediate[cell_idx(i, j, nz)] -
            dV_z_faces[static_cast<std::size_t>(i * (nz + 1) + j + 1)] +
            dV_z_faces[static_cast<std::size_t>(i * (nz + 1) + j)];
      } else {
        cell_vol_final[cell_idx(i, j, nz)] =
            cell_vol_intermediate[cell_idx(i, j, nz)] +
            dV_z_faces[static_cast<std::size_t>(i * (nz + 1) + j + 1)] -
            dV_z_faces[static_cast<std::size_t>(i * (nz + 1) + j)];
      }
    }
  }

  apply_z_sweep_to_fields(
      fields, cell_vol_intermediate, dV_z_faces, K, nz,
      swept_volume_sign_fixed);

  if (!sync_device_scratch(d_cell_mass, d_cell_mom_r, d_cell_mom_z,
                           d_cell_E_int_e, d_cell_E_int_i,
                           d_cell_volFrac_mass, d_cell_rad_E_ext,
                           d_cell_vol_intermediate, d_cell_vol_final,
                           cell_mass, cell_mom_r, cell_mom_z,
                           cell_E_int_e, cell_E_int_i,
                           cell_volFrac_mass, cell_rad_E_ext,
                           cell_vol_intermediate, cell_vol_final)) {
    result.failure = AxisBandRemapFailure::ScratchAllocationFailed;
    return result;
  }

  const Projection projection = project_cell_momentum_to_nodes(
      cell_mass, cell_mom_r, cell_mom_z, r_target, K, nz);
  const double mass_after = reduced_sum(reduction, sum_vector(cell_mass));
  const double e_e_after = reduced_sum(reduction, sum_vector(cell_E_int_e));
  const double e_i_after = reduced_sum(reduction, sum_vector(cell_E_int_i));
  const double e_rad_after =
      reduced_sum(reduction, sum_vector(cell_rad_E_ext));
  const double e_kin_after = reduced_sum(reduction, projection.kinetic);
  const double mom_r_after = reduced_sum(reduction, sum_vector(cell_mom_r));
  const double mom_z_after = reduced_sum(reduction, sum_vector(cell_mom_z));
  const double mom_scale =
      std::max({std::abs(mom_r_before), std::abs(mom_z_before),
                mass_before * v_scale, kTiny});

  result.mass_delta_rel = rel_delta(mass_after, mass_before);
  result.mom_r_delta_scaled = (mom_r_after - mom_r_before) / mom_scale;
  result.mom_z_delta_scaled = (mom_z_after - mom_z_before) / mom_scale;
  result.E_int_e_delta_rel = rel_delta(e_e_after, e_e_before);
  result.E_int_i_delta_rel = rel_delta(e_i_after, e_i_before);
  result.E_kin_delta_rel = rel_delta(e_kin_after, e_kin_before);
  result.E_rad_delta_rel = (n_groups > 0) ? rel_delta(e_rad_after, e_rad_before) : 0.0;

  result.min_v_final = reduced_min(reduction, finite_positive_min(cell_vol_final));
  result.min_cell_mass = reduced_min(reduction, finite_positive_min(cell_mass));
  result.min_specific_e =
      reduced_min(reduction,
                  min_specific_internal(cell_mass, cell_E_int_e, cell_E_int_i,
                                        two_temperature));
  const double min_E_e_z = finite_positive_min(cell_E_int_e);
  const double min_E_i_z = finite_positive_min(cell_E_int_i);
  const double min_E_z =
      reduced_min(reduction,
                  two_temperature ? std::min(min_E_e_z, min_E_i_z) : min_E_e_z);
  const double min_E_i_z_1t =
      two_temperature ? 0.0 : reduced_min(reduction, min_E_i_z);

  if (!(result.min_v_final > 0.0)) {
    result.failure = AxisBandRemapFailure::FinalPositivityViolation;
    log_conservation_warning(result);
    return result;
  }
  if (!(result.min_cell_mass > 0.0)) {
    result.failure = AxisBandRemapFailure::RemappedMassNegative;
    log_conservation_warning(result);
    return result;
  }
  if (!(min_E_z > 0.0) || !(result.min_specific_e > 0.0) ||
      (!two_temperature && !(min_E_i_z_1t >= 0.0))) {
    result.failure = AxisBandRemapFailure::RemappedInternalEnergyNegative;
    log_conservation_warning(result);
    return result;
  }
  if (n_groups > 0) {
    const double min_rad_z =
        reduced_min(reduction, finite_min(cell_rad_E_ext));
    if (!(min_rad_z >= 0.0)) {
      result.failure = AxisBandRemapFailure::RemappedRadiationNegative;
      log_conservation_warning(result);
      return result;
    }
  }
  // Hard conservation gate (k04 R14): mass, internal energies, radiation
  // energy, and band cell momentum must close to roundoff. The projection
  // kinetic-energy delta is intentionally excluded — the mass-weighted
  // cell-to-node projection is not KE-conserving by construction and its
  // delta stays a warning-level diagnostic. The E_rad term is gated only
  // when the band held radiation energy before the remap; otherwise the
  // groupwise nonnegativity gate above is the guard.
  const double cons_max_abs =
      std::max({std::abs(result.mass_delta_rel),
                std::abs(result.E_int_e_delta_rel),
                std::abs(result.E_int_i_delta_rel),
                (e_rad_before > 0.0) ? std::abs(result.E_rad_delta_rel) : 0.0,
                std::abs(result.mom_r_delta_scaled),
                std::abs(result.mom_z_delta_scaled)});
  if (!(cons_max_abs <= kConsFailRel)) {
    result.failure = AxisBandRemapFailure::ConservationViolation;
    log_conservation_warning(result);
    return result;
  }

  if (mesh_changed) {
    for (int i = 0; i <= K; ++i) {
      for (int j = 0; j <= nz; ++j) {
        const std::size_t n = node_idx(i, j, nz);
        x_r[n] = r_target[n];
        x_z[n] = z_target[n];
        v_r[n] = projection.v_r_node[n];
        v_z[n] = projection.v_z_node[n];
      }
    }

    for (int i = 0; i < K; ++i) {
      for (int j = 0; j < nz; ++j) {
        const std::size_t b = cell_idx(i, j, nz);
        const std::size_t c = cell_idx(i, j, nz);
        vol[c] = cell_vol_final[b];
        mass_field[c] = cell_mass[b];
        rho[c] = cell_mass[b] / cell_vol_final[b];
        ee[c] = cell_E_int_e[b] / cell_mass[b];
        ei[c] = cell_E_int_i[b] / cell_mass[b];
        for (int mat = 0; mat < n_materials; ++mat) {
          volFrac[c * static_cast<std::size_t>(n_materials) +
                  static_cast<std::size_t>(mat)] =
              cell_volFrac_mass[static_cast<std::size_t>(mat) * band_cells + b] /
              cell_mass[b];
        }
        for (int g = 0; g < n_groups; ++g) {
          rad_E[c * static_cast<std::size_t>(n_groups) +
                static_cast<std::size_t>(g)] =
              cell_rad_E_ext[static_cast<std::size_t>(g) * band_cells + b] /
              cell_vol_final[b];
        }
        const double mass_denom = std::max(cell_mass[b], kTiny);
        if (remap_gas_tracer) {
          gas_tracer_Y[c] =
              clamp01_host(cell_gas_tracer_mass[b] / mass_denom);
        }
        if (remap_burn_species) {
          for (int s = 0; s < tenryu::burn::kNumSpecies; ++s) {
            state.burn_n_host[
                c * static_cast<std::size_t>(tenryu::burn::kNumSpecies) +
                static_cast<std::size_t>(s)] =
                std::fmax(
                    cell_burn_species_mass[
                        static_cast<std::size_t>(s) * band_cells + b] /
                        mass_denom,
                    0.0);
          }
        }
        if (remap_hot_e_eps) {
          state.hot_e_eps_cum_host[c] =
              std::fmax(cell_hot_e_eps_mass[b] / mass_denom, 0.0);
        }
        if (remap_burn_eps) {
          state.burn_eps_cum_host[c] =
              std::fmax(cell_burn_eps_mass[b] / mass_denom, 0.0);
        }
      }
    }

    state.x_r.copy_from_host(x_r);
    state.x_z.copy_from_host(x_z);
    state.v_r.copy_from_host(v_r);
    state.v_z.copy_from_host(v_z);
    state.vol.copy_from_host(vol);
    state.rho.copy_from_host(rho);
    state.mass.copy_from_host(mass_field);
    state.ee.copy_from_host(ee);
    state.ei.copy_from_host(ei);
    if (n_materials > 0) {
      state.volFrac.copy_from_host(volFrac);
    }
    if (n_groups > 0) {
      state.rad_E.copy_from_host(rad_E);
    }
    if (remap_gas_tracer) {
      state.gas_tracer_Y.copy_from_host(gas_tracer_Y);
    }

    if (state.mesh.cell_vol.size() == n_cells) {
      for (int i = 0; i < K; ++i) {
        for (int j = 0; j < nz; ++j) {
          const std::size_t c = cell_idx(i, j, nz);
          state.mesh.cell_vol[c] = cell_vol_final[c];
          if (state.mesh.cell_centroid_r.size() == n_cells &&
              state.mesh.cell_centroid_z.size() == n_cells) {
            const std::size_t n00 = node_idx(i, j, nz);
            const std::size_t n10 = node_idx(i + 1, j, nz);
            const std::size_t n11 = node_idx(i + 1, j + 1, nz);
            const std::size_t n01 = node_idx(i, j + 1, nz);
            state.mesh.cell_centroid_r[c] =
                0.25 * (r_target[n00] + r_target[n10] +
                        r_target[n11] + r_target[n01]);
            state.mesh.cell_centroid_z[c] =
                0.25 * (z_target[n00] + z_target[n10] +
                        z_target[n11] + z_target[n01]);
          }
          if (state.mesh.cell_area.size() == n_cells) {
            const std::size_t n00 = node_idx(i, j, nz);
            const std::size_t n10 = node_idx(i + 1, j, nz);
            const std::size_t n11 = node_idx(i + 1, j + 1, nz);
            const std::size_t n01 = node_idx(i, j + 1, nz);
            state.mesh.cell_area[c] = std::abs(rz_signed_quad_area(
                r_target[n00], z_target[n00],
                r_target[n10], z_target[n10],
                r_target[n11], z_target[n11],
                r_target[n01], z_target[n01]));
          }
        }
      }
    }
  }

  result.succeeded = true;
  result.failure = AxisBandRemapFailure::None;
  log_conservation_warning(result);
  return result;
}

}  // namespace tenryu::hydro::ale
