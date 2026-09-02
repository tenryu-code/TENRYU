#include "coupling/staged_dt_gate.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "core/config.hpp"
#include "core/error.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::coupling {
namespace {

constexpr double kTiny = 1.0e-300;
constexpr int kCsw98MaxSideVecs =
    mesh::kMeshTopoCellStorageSlotsMaxGeneral;

struct CswGateParams {
  int axisline_av_enabled = 0;
  int axistouch_av_off = 0;
  double axis_eps_cm = 0.0;
};

bool aw_axisline_av_enabled() {
  static const bool enabled = [] {
    const char* const raw = std::getenv("TENRYU_AW_AXISLINE_AV");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
  }();
  return enabled;
}

bool axistouch_av_off_enabled() {
  static const bool enabled = [] {
    const char* const raw = std::getenv("TENRYU_I1B_AXISTOUCH_AV_OFF");
    return raw != nullptr && raw[0] == '1' && raw[1] == '\0';
  }();
  return enabled;
}

bool multiblock_aw_axis_line_edge(const double* node_r,
                                  const int n0,
                                  const int n1,
                                  const bool aw_planar,
                                  const CswGateParams params) {
  if (params.axisline_av_enabled != 0 || !aw_planar) {
    return false;
  }
  return node_r[n0] <= params.axis_eps_cm &&
         node_r[n1] <= params.axis_eps_cm;
}

bool multiblock_axis_touch_edge(const double* node_r,
                                const int n0,
                                const int n1,
                                const bool aw_planar,
                                const CswGateParams params) {
  return params.axistouch_av_off != 0 && aw_planar &&
         (node_r[n0] <= params.axis_eps_cm ||
          node_r[n1] <= params.axis_eps_cm);
}

// D7: these C2 helpers are a host-only verbatim arithmetic transcription of
// compatible_av_csw.cuh. They intentionally remain copied here, with parity
// coverage, instead of sharing a header with the device implementation.
inline double csw98_winding_orientation(const double* r,
                                        const double* z,
                                        const int nverts) {
  double area2 = 0.0;
  double scale = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp = (k + 1 == nverts) ? 0 : (k + 1);
    area2 += r[k] * z[kp] - r[kp] * z[k];
    scale += r[k] * r[k] + z[k] * z[k];
  }
  const double threshold = 64.0 * 2.220446049250313e-16 * scale;
  if (!(std::abs(area2) > threshold)) {
    ::tenryu::core::tenryu_abort(
        "fabs(area2) > threshold",
        "csw98_winding_orientation: zero/near-zero signed area (degenerate cell) "
        "must not silently orient +1 (AI-review Amendment A)",
        __FILE__, __LINE__);
  }
  return (area2 >= 0.0) ? 1.0 : -1.0;
}

inline void csw98_rz_corner_gradients(const double* r,
                                      const double* z,
                                      const int nverts,
                                      double* a_r,
                                      double* a_z) {
  constexpr double pi_over_three =
      1.0471975511965977461542144610931676280657231331250352736615;
  const double orientation = csw98_winding_orientation(r, z, nverts);
  for (int k = 0; k < nverts; ++k) {
    const int km = (k + nverts - 1) % nverts;
    const int kp = (k + 1) % nverts;
    a_r[k] = orientation * (pi_over_three *
        (z[kp] * (r[k] + r[kp]) + (r[k] * z[kp] - r[kp] * z[k]) -
         z[km] * (r[km] + r[k]) + (r[km] * z[k] - r[k] * z[km])));
    a_z[k] = orientation * (pi_over_three *
        (r[km] * (r[km] + r[k]) - r[kp] * (r[k] + r[kp])));
  }
}

inline void csw98_planar_corner_gradients(const double* r,
                                          const double* z,
                                          const int nverts,
                                          double* a_r,
                                          double* a_z) {
  const double orientation = csw98_winding_orientation(r, z, nverts);
  for (int k = 0; k < nverts; ++k) {
    const int km = (k + nverts - 1) % nverts;
    const int kp = (k + 1) % nverts;
    a_r[k] = 0.5 * orientation * (z[kp] - z[km]);
    a_z[k] = 0.5 * orientation * (r[km] - r[kp]);
  }
}

inline void csw98_c2_side_svecs(const double* r,
                                const double* z,
                                const int nverts,
                                double* s_r,
                                double* s_z,
                                const bool aw_planar = false) {
  if (!(nverts >= 3 && nverts <= kCsw98MaxSideVecs)) {
    ::tenryu::core::tenryu_abort(
        "nverts >= 3 && nverts <= kCsw98MaxSideVecs",
        "csw98_c2_side_svecs: vertex count outside supported range "
        "must not overrun local side-vector storage",
        __FILE__, __LINE__);
  }
  double a_r[kCsw98MaxSideVecs];
  double a_z[kCsw98MaxSideVecs];
  if (aw_planar) {
    csw98_planar_corner_gradients(r, z, nverts, a_r, a_z);
  } else {
    csw98_rz_corner_gradients(r, z, nverts, a_r, a_z);
  }
  double mean_r = 0.0;
  double mean_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    mean_r += a_r[k];
    mean_z += a_z[k];
  }
  const double inv = 1.0 / static_cast<double>(nverts);
  mean_r *= inv;
  mean_z *= inv;
  double s0_r = 0.0;
  double s0_z = 0.0;
  for (int j = 1; j < nverts; ++j) {
    const double w = static_cast<double>(nverts - j);
    s0_r += w * (a_r[j] - mean_r);
    s0_z += w * (a_z[j] - mean_z);
  }
  s_r[0] = s0_r * inv;
  s_z[0] = s0_z * inv;
  for (int k = 1; k < nverts; ++k) {
    s_r[k] = s_r[k - 1] - (a_r[k] - mean_r);
    s_z[k] = s_z[k - 1] - (a_z[k] - mean_z);
  }
}

inline bool csw98_c2_svec_for_side(const double* r,
                                   const double* z,
                                   const int nverts,
                                   const int ca,
                                   const int cb,
                                   double* out_r,
                                   double* out_z,
                                   const bool aw_planar = false) {
  if (!(nverts >= 3 && nverts <= kCsw98MaxSideVecs)) {
    ::tenryu::core::tenryu_abort(
        "nverts >= 3 && nverts <= kCsw98MaxSideVecs",
        "csw98_c2_svec_for_side: vertex count outside supported range "
        "must not overrun local side-vector storage",
        __FILE__, __LINE__);
  }
  double s_r[kCsw98MaxSideVecs];
  double s_z[kCsw98MaxSideVecs];
  csw98_c2_side_svecs(r, z, nverts, s_r, s_z, aw_planar);
  if (cb == ((ca + 1) % nverts)) {
    *out_r = s_r[ca];
    *out_z = s_z[ca];
    return true;
  }
  if (ca == ((cb + 1) % nverts)) {
    *out_r = -s_r[cb];
    *out_z = -s_z[cb];
    return true;
  }
  *out_r = 0.0;
  *out_z = 0.0;
  return false;
}

void update_minimum(const double candidate,
                    const int cell,
                    double& family_min,
                    StagedDtResult& result) {
  if (candidate < family_min) {
    family_min = candidate;
  }
  if (candidate < result.min_dt) {
    result.min_dt = candidate;
    result.binding_cell = cell;
  }
}

}  // namespace

StagedDtResult evaluate_staged_hydro_dt(
    const int* csr_offsets,
    const int* csr_indices,
    const unsigned char* nverts,
    const int n_cells,
    const int n_nodes,
    const double* node_r,
    const double* node_z,
    const double* velocity_r,
    const double* velocity_z,
    const core::Config& cfg) {
  StagedDtResult result;
  if (cfg.numerics.ale.rezone_min_dt_s == 0.0) {
    return result;
  }

  TENRYU_ASSERT(csr_offsets != nullptr && csr_indices != nullptr &&
                    nverts != nullptr && node_r != nullptr &&
                    node_z != nullptr && velocity_r != nullptr &&
                    velocity_z != nullptr,
                "staged dt gate requires non-null staged arrays");
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "staged dt gate requires a non-empty staged mesh");

  const auto av_model = cfg.numerics.hydro.av_model;
  const bool edge_av_enabled =
      cfg.main.dim == 2 &&
      av_model == core::AvModel::CswEdgeCsw98 &&
      !cfg.numerics.hydro.csw_pole_floor_enabled &&
      !cfg.numerics.hydro.csw_pole_desens_enabled;
  const double coefficient = cfg.numerics.hydro.av_cfl_coefficient;
  const bool evaluate_edge_av = edge_av_enabled && coefficient > 0.0;
  const bool aw_planar = cfg.numerics.hydro.aw_compatible_force_work;
  const double crossing_dt_safety =
      cfg.numerics.hydro.crossing_dt_safety;
  const CswGateParams params{
      aw_axisline_av_enabled() ? 1 : 0,
      axistouch_av_off_enabled() ? 1 : 0,
      cfg.numerics.axis_eps_cm};

  for (int cell = 0; cell < n_cells; ++cell) {
    const int nv = static_cast<int>(nverts[cell]);
    TENRYU_ASSERT(nv >= 3 && nv <= kCsw98MaxSideVecs,
                  "staged dt gate cell vertex count outside supported range");
    const int begin = csr_offsets[cell];
    TENRYU_ASSERT(begin >= 0 && csr_offsets[cell + 1] >= begin + nv,
                  "staged dt gate CSR span is shorter than active ring");

    int nodes[kCsw98MaxSideVecs];
    double cell_r[kCsw98MaxSideVecs];
    double cell_z[kCsw98MaxSideVecs];
    for (int corner = 0; corner < nv; ++corner) {
      const int node = csr_indices[begin + corner];
      TENRYU_ASSERT(node >= 0 && node < n_nodes,
                    "staged dt gate CSR node outside staged mesh");
      nodes[corner] = node;
      cell_r[corner] = node_r[node];
      cell_z[corner] = node_z[node];
    }

    for (int corner0 = 0; corner0 < nv; ++corner0) {
      const int corner1 = (corner0 + 1) % nv;
      const int n0 = nodes[corner0];
      const int n1 = nodes[corner1];
      const double dx_r = node_r[n1] - node_r[n0];
      const double dx_z = node_z[n1] - node_z[n0];
      const double du_r = velocity_r[n1] - velocity_r[n0];
      const double du_z = velocity_z[n1] - velocity_z[n0];
      const double dx_mag = std::hypot(dx_r, dx_z);
      const double du_mag = std::hypot(du_r, du_z);

      if (crossing_dt_safety > 0.0 && dx_mag > 0.0) {
        // cfl.cu's 1-D guard lifted to a ring edge: dr=|dx| and closing is
        // the relative velocity projected onto the inward edge direction.
        const double closing = -(du_r * dx_r + du_z * dx_z) / dx_mag;
        if (closing > 0.0) {
          update_minimum(crossing_dt_safety * dx_mag / closing,
                         cell, result.crossing_dt, result);
        }
      }

      if (!evaluate_edge_av ||
          multiblock_aw_axis_line_edge(node_r, n0, n1, aw_planar, params) ||
          multiblock_axis_touch_edge(node_r, n0, n1, aw_planar, params) ||
          !(dx_mag > kTiny) || !(du_mag > kTiny)) {
        continue;
      }

      double s_r = 0.0;
      double s_z = 0.0;
      if (!csw98_c2_svec_for_side(cell_r, cell_z, nv, corner0, corner1,
                                  &s_r, &s_z, aw_planar)) {
        continue;
      }
      const double compression = du_r * s_r + du_z * s_z;
      if (!(compression < 0.0) || !std::isfinite(compression)) {
        continue;
      }

      const double s_mag = std::hypot(s_r, s_z);
      if (!(s_mag > 0.0)) {
        continue;
      }
      const double proj = std::abs(compression) / (du_mag * s_mag);
      // D7: psi=0 disables the limiter conservatively for the host gate, so
      // du_eff_host >= du_eff_device and the staged bound biases to rejection.
      const double psi = 0.0;
      const double psi_clamped = std::min(1.0, std::max(0.0, psi));
      const double du_eff =
          du_mag * std::max(0.0, 1.0 - psi_clamped) * proj;
      if (du_eff > kTiny && std::isfinite(du_eff) && dx_mag > 0.0) {
        update_minimum(coefficient * dx_mag / du_eff,
                       cell, result.edge_av_dt, result);
      }
    }
  }

  return result;
}

}  // namespace tenryu::coupling
