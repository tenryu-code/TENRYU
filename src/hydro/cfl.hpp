#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cmath>
#include <limits>

#include "core/config.hpp"
#include "core/state.hpp"

#if defined(__CUDACC__)
#define TENRYU_CFL_HOST_DEVICE __host__ __device__
#else
#define TENRYU_CFL_HOST_DEVICE
#endif

namespace tenryu::parallel {
class Reduction;
}

namespace tenryu::hydro {

namespace detail {

TENRYU_CFL_HOST_DEVICE inline double
csr_polygon_characteristic_length_from_nodes(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const int* __restrict__ node_ids,
    const int nv) {
  if (x_r == nullptr || x_z == nullptr || node_ids == nullptr || nv < 3) {
    return 0.0;
  }

  double cross_sum = 0.0;
  double perimeter = 0.0;
  double centroid_r_sum = 0.0;
  double centroid_z_sum = 0.0;
  for (int k = 0; k < nv; ++k) {
    const int kp = (k + 1 == nv) ? 0 : (k + 1);
    const int n = node_ids[k];
    const int np = node_ids[kp];
    const double cross = x_r[n] * x_z[np] - x_r[np] * x_z[n];
    const double edge = std::hypot(x_r[np] - x_r[n], x_z[np] - x_z[n]);
    if (!std::isfinite(edge) || !std::isfinite(cross)) {
      return 0.0;
    }
    if (!(edge > 0.0)) {
      if (n == np) {
        continue;
      }
      return 0.0;
    }
    cross_sum += cross;
    perimeter += edge;
    centroid_r_sum += (x_r[n] + x_r[np]) * cross;
    centroid_z_sum += (x_z[n] + x_z[np]) * cross;
  }

  const double area = 0.5 * std::fabs(cross_sum);
  if (!(area > 0.0) || !(perimeter > 0.0) || !std::isfinite(area) ||
      !std::isfinite(perimeter) || !std::isfinite(centroid_r_sum) ||
      !std::isfinite(centroid_z_sum)) {
    return 0.0;
  }

  const double inv_centroid_denom = 1.0 / (3.0 * cross_sum);
  const double centroid_r = centroid_r_sum * inv_centroid_denom;
  const double centroid_z = centroid_z_sum * inv_centroid_denom;
  if (!std::isfinite(inv_centroid_denom) || !std::isfinite(centroid_r) ||
      !std::isfinite(centroid_z)) {
    return 0.0;
  }

  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < nv; ++k) {
    const int kp = (k + 1 == nv) ? 0 : (k + 1);
    const int n = node_ids[k];
    const int np = node_ids[kp];
    const double ar = x_r[n];
    const double az = x_z[n];
    const double br = x_r[np];
    const double bz = x_z[np];
    const double ab = std::hypot(br - ar, bz - az);
    const double bc = std::hypot(centroid_r - br, centroid_z - bz);
    const double ca = std::hypot(ar - centroid_r, az - centroid_z);
    const double tri_area2 =
        std::fabs((br - ar) * (centroid_z - az) -
                  (bz - az) * (centroid_r - ar));
    if (!std::isfinite(tri_area2) || !std::isfinite(ab) ||
        !std::isfinite(bc) || !std::isfinite(ca)) {
      return 0.0;
    }
    if (!(ab > 0.0)) {
      if (n == np) {
        continue;
      }
      return 0.0;
    }
    if (!(tri_area2 > 0.0) || !(bc > 0.0) || !(ca > 0.0)) {
      return 0.0;
    }
    min_altitude = std::fmin(min_altitude, tri_area2 / ab);
    min_altitude = std::fmin(min_altitude, tri_area2 / bc);
    min_altitude = std::fmin(min_altitude, tri_area2 / ca);
  }

  const double h_2ap = 2.0 * area / perimeter;
  const double h = std::fmin(h_2ap, min_altitude);
  return (h > 0.0 && std::isfinite(h)) ? h : 0.0;
}

inline double quad_center_cfl_h(const std::array<double, 4>& r,
                                const std::array<double, 4>& z,
                                double* h_2ap_out,
                                double* min_altitude_out) {
  double area2 = 0.0;
  double perimeter = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp1 = (k + 1) & 3;
    area2 += r[static_cast<std::size_t>(k)] * z[static_cast<std::size_t>(kp1)] -
             r[static_cast<std::size_t>(kp1)] * z[static_cast<std::size_t>(k)];
    perimeter += std::hypot(r[static_cast<std::size_t>(kp1)] -
                                r[static_cast<std::size_t>(k)],
                            z[static_cast<std::size_t>(kp1)] -
                                z[static_cast<std::size_t>(k)]);
  }
  const double area = 0.5 * std::abs(area2);
  if (!(area > 0.0) || !(perimeter > 0.0) ||
      !std::isfinite(area) || !std::isfinite(perimeter)) {
    if (h_2ap_out != nullptr) {
      *h_2ap_out = 0.0;
    }
    if (min_altitude_out != nullptr) {
      *min_altitude_out = 0.0;
    }
    return 0.0;
  }

  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 4; ++k) {
    const int e0 = (k + 1) & 3;
    const int e1 = (k + 2) & 3;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double edge_len = std::hypot(er, ez);
    if (!(edge_len > 0.0) || !std::isfinite(edge_len)) {
      if (h_2ap_out != nullptr) {
        *h_2ap_out = 0.0;
      }
      if (min_altitude_out != nullptr) {
        *min_altitude_out = 0.0;
      }
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / edge_len;
    min_altitude = std::min(min_altitude, altitude);
  }

  const double h_2ap = 2.0 * area / perimeter;
  if (h_2ap_out != nullptr) {
    *h_2ap_out = h_2ap;
  }
  if (min_altitude_out != nullptr) {
    *min_altitude_out = min_altitude;
  }
  const double h = std::min(h_2ap, min_altitude);
  return (h > 0.0 && std::isfinite(h)) ? h : 0.0;
}

inline double quad_center_cfl_h(const std::array<double, 4>& r,
                                const std::array<double, 4>& z) {
  return quad_center_cfl_h(r, z, nullptr, nullptr);
}

inline double active_polygon_center_cfl_h(const std::array<double, 4>& r,
                                          const std::array<double, 4>& z,
                                          const int active_nverts,
                                          double* h_2ap_out,
                                          double* min_altitude_out) {
  if (active_nverts != 3) {
    return quad_center_cfl_h(r, z, h_2ap_out, min_altitude_out);
  }

  double area2 = 0.0;
  double perimeter = 0.0;
  for (int k = 0; k < 3; ++k) {
    const int kp1 = (k + 1) % 3;
    area2 += r[static_cast<std::size_t>(k)] * z[static_cast<std::size_t>(kp1)] -
             r[static_cast<std::size_t>(kp1)] * z[static_cast<std::size_t>(k)];
    perimeter += std::hypot(r[static_cast<std::size_t>(kp1)] -
                                r[static_cast<std::size_t>(k)],
                            z[static_cast<std::size_t>(kp1)] -
                                z[static_cast<std::size_t>(k)]);
  }
  const double area = 0.5 * std::abs(area2);
  if (!(area > 0.0) || !(perimeter > 0.0) ||
      !std::isfinite(area) || !std::isfinite(perimeter)) {
    if (h_2ap_out != nullptr) {
      *h_2ap_out = 0.0;
    }
    if (min_altitude_out != nullptr) {
      *min_altitude_out = 0.0;
    }
    return 0.0;
  }

  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < 3; ++k) {
    const int e0 = (k + 1) % 3;
    const int e1 = (k + 2) % 3;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double edge_len = std::hypot(er, ez);
    if (!(edge_len > 0.0) || !std::isfinite(edge_len)) {
      if (h_2ap_out != nullptr) {
        *h_2ap_out = 0.0;
      }
      if (min_altitude_out != nullptr) {
        *min_altitude_out = 0.0;
      }
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / edge_len;
    min_altitude = std::min(min_altitude, altitude);
  }

  const double h_2ap = 2.0 * area / perimeter;
  if (h_2ap_out != nullptr) {
    *h_2ap_out = h_2ap;
  }
  if (min_altitude_out != nullptr) {
    *min_altitude_out = min_altitude;
  }
  const double h = std::min(h_2ap, min_altitude);
  return (h > 0.0 && std::isfinite(h)) ? h : 0.0;
}

inline double active_polygon_center_cfl_h(const std::array<double, 4>& r,
                                          const std::array<double, 4>& z,
                                          const int active_nverts) {
  return active_polygon_center_cfl_h(r, z, active_nverts, nullptr, nullptr);
}

inline double min_altitude_from_polygon(
    const std::array<double, mesh::kMeshTopoCellStorageSlotsMax>& r,
    const std::array<double, mesh::kMeshTopoCellStorageSlotsMax>& z,
    const int active_nverts) {
  double min_altitude = std::numeric_limits<double>::infinity();
  for (int k = 0; k < active_nverts; ++k) {
    const int e0 = (k + 1) % active_nverts;
    const int e1 = (k + 2) % active_nverts;
    const double er = r[static_cast<std::size_t>(e1)] -
                      r[static_cast<std::size_t>(e0)];
    const double ez = z[static_cast<std::size_t>(e1)] -
                      z[static_cast<std::size_t>(e0)];
    const double edge_len = std::hypot(er, ez);
    if (!(edge_len > 0.0) || !std::isfinite(edge_len)) {
      return 0.0;
    }
    const double pr = r[static_cast<std::size_t>(k)] -
                      r[static_cast<std::size_t>(e0)];
    const double pz = z[static_cast<std::size_t>(k)] -
                      z[static_cast<std::size_t>(e0)];
    const double altitude = std::abs(pr * ez - pz * er) / edge_len;
    if (!std::isfinite(altitude)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    min_altitude = std::min(min_altitude, altitude);
  }
  return min_altitude;
}

inline double active_polygon_center_cfl_h(
    const std::array<double, mesh::kMeshTopoCellStorageSlotsMax>& r,
    const std::array<double, mesh::kMeshTopoCellStorageSlotsMax>& z,
    const int active_nverts,
    double* h_2ap_out,
    double* min_altitude_out) {
  if (active_nverts == 3 || active_nverts == 4) {
    std::array<double, 4> legacy_r{};
    std::array<double, 4> legacy_z{};
    for (int k = 0; k < active_nverts; ++k) {
      legacy_r[static_cast<std::size_t>(k)] =
          r[static_cast<std::size_t>(k)];
      legacy_z[static_cast<std::size_t>(k)] =
          z[static_cast<std::size_t>(k)];
    }
    return active_polygon_center_cfl_h(
        legacy_r, legacy_z, active_nverts, h_2ap_out, min_altitude_out);
  }
  if (active_nverts < 5 ||
      active_nverts > mesh::kMeshTopoCellStorageSlotsMax) {
    if (h_2ap_out != nullptr) {
      *h_2ap_out = 0.0;
    }
    if (min_altitude_out != nullptr) {
      *min_altitude_out = 0.0;
    }
    return 0.0;
  }

  double area2 = 0.0;
  double perimeter = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const int kp1 = (k + 1) % active_nverts;
    area2 += r[static_cast<std::size_t>(k)] *
                 z[static_cast<std::size_t>(kp1)] -
             r[static_cast<std::size_t>(kp1)] *
                 z[static_cast<std::size_t>(k)];
    perimeter += std::hypot(r[static_cast<std::size_t>(kp1)] -
                                r[static_cast<std::size_t>(k)],
                            z[static_cast<std::size_t>(kp1)] -
                                z[static_cast<std::size_t>(k)]);
  }
  const double area = 0.5 * std::abs(area2);
  if (!(area > 0.0) || !(perimeter > 0.0) ||
      !std::isfinite(area) || !std::isfinite(perimeter)) {
    if (h_2ap_out != nullptr) {
      *h_2ap_out = 0.0;
    }
    if (min_altitude_out != nullptr) {
      *min_altitude_out = 0.0;
    }
    return 0.0;
  }

  const double min_altitude =
      min_altitude_from_polygon(r, z, active_nverts);
  const double h_2ap = 2.0 * area / perimeter;
  if (h_2ap_out != nullptr) {
    *h_2ap_out = h_2ap;
  }
  if (min_altitude_out != nullptr) {
    *min_altitude_out = min_altitude;
  }
  const double h = std::min(h_2ap, min_altitude);
  return (h > 0.0 && std::isfinite(h)) ? h : 0.0;
}

inline double active_polygon_center_cfl_h(
    const std::array<double, mesh::kMeshTopoCellStorageSlotsMax>& r,
    const std::array<double, mesh::kMeshTopoCellStorageSlotsMax>& z,
    const int active_nverts) {
  return active_polygon_center_cfl_h(
      r, z, active_nverts, nullptr, nullptr);
}

}  // namespace detail

#undef TENRYU_CFL_HOST_DEVICE

double compute_volume_rate_cfl_dt(const core::State& state,
                                  double dt_in,
                                  double dt_used_prev,
                                  double threshold,
                                  const parallel::Reduction* reduction,
                                  int* exact_argmin_cell = nullptr);

double compute_rz_geometric_cfl_dt(const core::State& state,
                                   const core::Config& cfg,
                                   double dt_proposed);

double compute_rz_geometric_cfl_dt(const core::State& state,
                                   const core::Config& cfg,
                                   double dt_proposed,
                                   const double* vhalf_r,
                                   const double* vhalf_z,
                                   int* argmin_cell = nullptr);

// Reset the volume-rate CFL Lagrangian history after accepted ALE.
// ALE rezone/remap motion is non-Lagrangian, while the volume-rate CFL sensor
// measures Lagrangian flow between hydro snapshots.  Copying current volume
// into the baseline prevents one polluted post-ALE measurement.
void reset_volume_rate_cfl_history_after_ale(core::State& state);

struct HydroCflWinnerInfo {
  double dt = std::numeric_limits<double>::infinity();
  int cell_id = -1;
  int i = -1;
  int j = -1;
  double dt_at_cell = std::numeric_limits<double>::infinity();
  double dl_at_cell = 0.0;
  double cs_at_cell = 0.0;
  double rho_at_cell = 0.0;
  double u_z_at_cell = 0.0;
};

enum class HydroMinTermClass : int {
  AcousticCell = 0,
  EdgeAv = 1,
  EdgeCrossing = 2,
  Other = 3,
};

enum class HydroMinOtherTerm : int {
  None = 0,
  PostShock,
  ArtificialHeat,
  SubzonalPressure,
  AxisMargin,
  TriFanCenter,
  CornerJPredict,
  PoleAxisContact,
  RzGeometric,
  VolumeRate,
  PolarSlavingStiffness,
  CentralPseudoCoreAcoustic,
  PoleAngularDerefineAcoustic,
  EdgeAccelDisplacement,
};

enum class HydroMinTermDetail : int {
  Unknown = 0,
  AcousticCell,
  CswEdgeAv,
  Csw98EdgeAv,
  TensorAv,
  EdgeCrossing,
};

struct HydroMinContributor {
  HydroMinTermClass term_class = HydroMinTermClass::Other;
  HydroMinOtherTerm other_term = HydroMinOtherTerm::None;
  HydroMinTermDetail term_detail = HydroMinTermDetail::Unknown;
  int cell_id = -1;
  // Raw (pre-cell_id_stable-remap) cell index of the argmin; equals cell_id
  // when no stable remap applies. Consumers that address live mesh arrays
  // (e.g. the dt-floor absorption retry) must use this, not cell_id.
  int cell_id_raw = -1;
  int edge_id = -1;
  int node0 = -1;
  int node1 = -1;
  int other_node = -1;
  double length = 0.0;
  double du = 0.0;
  double acceleration = 0.0;
  double coefficient = 0.0;
  double raw_dt = std::numeric_limits<double>::infinity();
  double rho = 0.0;
  double mu = 0.0;
  double polar_lambda = 0.0;
  double polar_sigma = 0.0;
};

struct TriFanCenterCflInfo {
  double dt = std::numeric_limits<double>::infinity();
  int cell_id = -1;
  int i = -1;
  int j = -1;
  double h = 0.0;
  double h_2ap = 0.0;
  double min_altitude = 0.0;
  double c_eff = 0.0;
  double q_over_p = 0.0;
  double r4[4] = {};
  double z4[4] = {};
  int block_id = -1;
  int nverts = 0;
  double r8[mesh::kMeshTopoCellStorageSlotsMax] = {};
  double z8[mesh::kMeshTopoCellStorageSlotsMax] = {};
};

struct HydroDtDiagnostics {
  double dt = std::numeric_limits<double>::infinity();
  double acoustic_dt = std::numeric_limits<double>::infinity();
  double post_shock_dt = std::numeric_limits<double>::infinity();
  // 1D node-crossing guard (Numerics.hydro.crossing_dt_safety; 2026-07-26 review).
  double crossing_dt = std::numeric_limits<double>::infinity();
  double edge_accel_dt = std::numeric_limits<double>::infinity();
  // 1D artificial-heat row-sum bound (av_heat_C, VNR chi; 2026-07-26 review).
  double art_heat_dt = std::numeric_limits<double>::infinity();
  double axis_margin_dt = std::numeric_limits<double>::infinity();
  double edge_av_dt = std::numeric_limits<double>::infinity();
  double subzonal_pressure_dt = std::numeric_limits<double>::infinity();
  double rz_geometric_dt = std::numeric_limits<double>::infinity();
  double volume_rate_dt = std::numeric_limits<double>::infinity();
  double corner_j_predict_dt = std::numeric_limits<double>::infinity();
  HydroCflWinnerInfo cfl_winner;
  TriFanCenterCflInfo tri_fan_center_cfl;
  HydroMinContributor min_contributor;
};

double compute_dt_hydro(const core::State& state, const core::Config& cfg);

HydroDtDiagnostics compute_dt_hydro_diagnostics(const core::State& state,
                                                const core::Config& cfg,
                                                bool track_min_contributor = false);

struct HydroDtArgmin {
  double dt = std::numeric_limits<double>::infinity();
  int argmin_cell = -1;
  double sqrt_area_at_argmin = 0.0;
  double cs_at_argmin = 0.0;
  int argmin_i = -1;
  int argmin_j = -1;
  double rho_at_argmin = 0.0;
  double u_z_at_argmin = 0.0;
};

HydroDtArgmin compute_dt_hydro_argmin(const core::State& state,
                                      const core::Config& cfg);

struct AxisMarginDtArgmin {
  double dt = std::numeric_limits<double>::infinity();
  double scale = 1.0;
  TriFanCenterCflInfo center_cfl;
};

struct VolumeRateDtArgmin {
  double dt = std::numeric_limits<double>::infinity();
  int argmin_cell = -1;
  double frac_rate_at_argmin = 0.0;
};

double compute_dt_hydro_acoustic(const core::State& state,
                                 const core::Config& cfg);

TriFanCenterCflInfo compute_centroid_r_center_cfl_dt(
    const core::State& state,
    const core::Config& cfg);

AxisMarginDtArgmin compute_dt_hydro_axis_margin_argmin(
    const core::State& state,
    const core::Config& cfg);

VolumeRateDtArgmin compute_dt_hydro_volume_rate_argmin(
    const core::State& state,
    const core::Config& cfg);

}  // namespace tenryu::hydro
