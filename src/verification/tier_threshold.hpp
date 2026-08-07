#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <exception>

namespace tenryu::verification {

inline constexpr std::size_t kThresholdTierCount = 9;

enum class ThresholdTier : std::uint8_t {
  deterministic_short = 1,
  deterministic_long = 2,
  monte_carlo_tally = 3,
  monte_carlo_packet = 4,
  smooth_mms_convergence_rate = 5,
  shock_convergence_rate = 6,
  anisotropy_acceptance = 7,
  ulp_pure_geometry = 8,
  c_eps_n_op = 9,
};

enum class ThresholdUnit : std::uint8_t {
  dimensionless_relative = 1,
  dimensionless_rate = 2,
  ulp_count = 3,
  exact_closed_form = 4,
  roundoff_event = 5,
  c_eps_operation_count = 6,
};

enum class ReductionMode : std::uint8_t {
  atomic_add = 1,
  kahan = 2,
  pairwise = 3,
};

struct TierThreshold {
  ThresholdTier tier = ThresholdTier::deterministic_short;
  const char* name = "";
  ThresholdUnit primary_unit = ThresholdUnit::dimensionless_relative;
  ThresholdUnit relative_tolerance_unit = ThresholdUnit::dimensionless_relative;

  double relative_tolerance = 0.0;
  double relative_target = 0.0;
  double mass_relative_tolerance = 0.0;
  double mass_relative_target = 0.0;
  double energy_relative_tolerance = 0.0;
  double energy_relative_target = 0.0;
  double convergence_rate_min = 0.0;
  double smooth_region_rate_min = 0.0;
  double systematic_bias_max_exclusive = 0.0;
  double confidence_level = 0.0;
  double production_relative_target = 0.0;
  int production_min_r = 0;
  int production_min_z = 0;
  double ulp_tolerance = 0.0;
  double roundoff_operation_count = 0.0;
  double c_eps_multiplier = 0.0;

  bool shock_position_convergence_required = false;
  bool exact_closed_form_required = false;
  bool event_roundoff_required = false;
  bool c_eps_n_op_required = false;
};

class UnknownThresholdTier final : public std::exception {
 public:
  [[nodiscard]] const char* what() const noexcept override;
};

class UnknownReductionMode final : public std::exception {
 public:
  [[nodiscard]] const char* what() const noexcept override;
};

[[nodiscard]] const std::array<ThresholdTier, kThresholdTierCount>& threshold_tiers();
[[nodiscard]] const TierThreshold& threshold_for(ThresholdTier tier);

[[nodiscard]] double c_eps_n_op_tolerance(std::size_t operation_count,
                                          ReductionMode mode,
                                          double C = 16.0);

}  // namespace tenryu::verification
