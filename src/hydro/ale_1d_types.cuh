#pragma once

namespace tenryu::hydro::ale1d {

enum class FeatureKind {
  LaserAbsorption,
  AblationFront,
  Shock,
  MaterialInterface,
  CenterHotspot
};

struct Ale1dFeature {
  FeatureKind kind = FeatureKind::CenterHotspot;
  double x_center = 0.0;
  double r_center = 0.0;
  double sigma_x = 0.0;
  double sigma_r = 0.0;
  double confidence = 0.0;
  double target_cells = 0.0;
  int peak_cell_or_face = -1;
  bool pinned_face = false;
};

enum class Ale1dSkipReason {
  None,
  Disabled,
  WrongGeometry,
  ParticleModeUnsupported,
  NTooSmall,
  ProtectedFractionTooHigh,
  MovableSegmentTooSmall,
  BenefitTooSmall,
  CandidateInvalid,
  ConservationRejected,
  DtPenaltyTooLarge
};

const char* to_string(Ale1dSkipReason r);

struct Ale1dStepResult {
  bool applied = false;
  bool quality_triggered = false;
  bool floor_triggered = false;
  bool cadence_triggered = false;
  bool converged = true;
  bool remap_rejected = false;
  Ale1dSkipReason skip_reason = Ale1dSkipReason::None;
  double max_dr_ratio = 1.0;
  double mass_conservation_rel_err = 0.0;
  double energy_conservation_rel_err = 0.0;
  double radiation_conservation_rel_err = 0.0;
  double kinetic_energy_drift_rel = 0.0;
  double radiation_smearing_metric = 0.0;
  int n_protected_nodes = 0;
};

enum class RemapInterpolationMode {
  PinnedZeroFlux,
  FirstOrderDonor,
  LimitedHighOrder
};

struct FieldRemapPolicy {
  RemapInterpolationMode mode = RemapInterpolationMode::FirstOrderDonor;
  int low_order_band_cells = 2;
  bool requires_positivity = true;
  bool reset_after_commit = false;
};

}  // namespace tenryu::hydro::ale1d
