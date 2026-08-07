#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

namespace tenryu::laser::sector_ps {

// Sheet bookkeeping folds at the turning point; the caustic index is carried separately — Follett's caution that CBET/field evaluation must respect the caustic is enforced downstream via the field-limiter flag (in_limiter_zone), v1 design §11 constraint (iii). This is a documented v1 deviation from a strict caustic-fold bookkeeping.

// One traced ray of the reference beam, as a polyline in the trace
// meridional plane, ordered along the propagation path.
struct RayPath {
  int ray_index;
  double impact_parameter;
  std::vector<double> r;
  std::vector<double> theta;
  std::vector<double> P;
  std::vector<double> area;
  // Optional exact wavevector angle at every node. When empty, alpha is
  // reconstructed from the polyline by finite differences.
  std::vector<double> alpha;
};

struct PhaseSpaceParams {
  double caustic_area_rel_tol = 1.0e-3;
  double bouguer_tol = 1.0e-10;
  double bouguer_tol_fd = 1.0e-3;
};

struct RayAnnotation {
  int n_turning;
  int caustic_index;
  bool ambiguous;
  double bouguer_max_drift;
};

struct ExclusionLedger {
  double excluded_power_fraction;
  int n_excluded_rays;
};

struct PumpLookup {
  bool valid;
  double I;
  double alpha;
};

struct CrossingView {
  double theta;
  double alpha;
  double P;
  double area;
  int ray_index;
  bool in_limiter_zone;
};

struct FlatTable {
  int n_shells;
  std::vector<int> offsets;          // size 2*n_shells+1, CSR over bins
  std::vector<double> theta;         // per crossing, bin-concatenated
  std::vector<double> alpha;
  std::vector<double> P;
  std::vector<double> area;
  std::vector<std::int32_t> ray_index;
  std::vector<std::uint8_t> in_limiter;
};

struct PhaseSpaceTable {
  PhaseSpaceTable() = default;

 private:
  struct Impl;
  std::shared_ptr<const Impl> impl_;

  friend PhaseSpaceTable build_table(
      const std::vector<RayPath>& rays,
      const std::vector<double>& shell_r,
      const std::vector<double>& eps_r,
      const PhaseSpaceParams& p,
      std::vector<RayAnnotation>& annotations_out);
  friend ExclusionLedger exclusion_ledger(const PhaseSpaceTable& table);
  friend PumpLookup lookup(const PhaseSpaceTable& table, int shell_index,
                           int sheet, double theta_p);
  friend const std::vector<CrossingView>& crossings(
      const PhaseSpaceTable& table, int shell_index, int sheet);
  friend FlatTable flatten_table(const PhaseSpaceTable& table);
  friend void flatten_table_into(const PhaseSpaceTable& table,
                                 FlatTable& flat);
};

PhaseSpaceTable build_table(const std::vector<RayPath>& rays,
                            const std::vector<double>& shell_r,
                            const std::vector<double>& eps_r,
                            const PhaseSpaceParams& p,
                            std::vector<RayAnnotation>& annotations_out);

ExclusionLedger exclusion_ledger(const PhaseSpaceTable& table);

// Follett Eq. (16), in the pump-beam frame.
double pump_polar_angle(double theta_s, double theta_i);

// Follett Eq. (21), for a port at (theta_i, phi_i).
double pump_polar_angle_3d(double theta_s, double theta_i, double phi_i);

// Follett Eq. (17), for an in-plane seed-pump crossing.
double crossing_cos_2d(double alpha_s, double alpha_p,
                       double theta_s, double theta_i);

// Components in (r_hat, theta_hat) of the beam frame.
std::array<double, 2> meridional_direction(double alpha, int sheet);

// I is interpolated from P_cross / area_cross at each bracketing crossing,
// rather than from separately interpolated P_cross and area_cross.
// Crossings in the limiter zone are excluded from the interpolation set.
PumpLookup lookup(const PhaseSpaceTable& table, int shell_index, int sheet,
                  double theta_p);

// Stored in ascending theta order with ray_index as the tie-break. Diagnostic
// enumeration retains crossings in the limiter zone.
const std::vector<CrossingView>& crossings(const PhaseSpaceTable& table,
                                           int shell_index, int sheet);

FlatTable flatten_table(const PhaseSpaceTable& table);
void flatten_table_into(const PhaseSpaceTable& table, FlatTable& flat);

}  // namespace tenryu::laser::sector_ps
