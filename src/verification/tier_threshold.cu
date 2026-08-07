#include "verification/tier_threshold.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace tenryu::verification {
namespace {

constexpr TierThreshold deterministic_short() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::deterministic_short;
  threshold.name = "deterministic-short";
  threshold.primary_unit = ThresholdUnit::dimensionless_relative;
  threshold.mass_relative_target = 1.0e-13;
  threshold.mass_relative_tolerance = 1.0e-12;
  threshold.energy_relative_target = 1.0e-12;
  threshold.energy_relative_tolerance = 1.0e-11;
  return threshold;
}

constexpr TierThreshold deterministic_long() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::deterministic_long;
  threshold.name = "deterministic-long";
  threshold.primary_unit = ThresholdUnit::dimensionless_relative;
  threshold.mass_relative_target = 1.0e-10;
  threshold.mass_relative_tolerance = 1.0e-8;
  threshold.energy_relative_target = 1.0e-8;
  threshold.energy_relative_tolerance = 1.0e-8;
  return threshold;
}

constexpr TierThreshold monte_carlo_tally() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::monte_carlo_tally;
  threshold.name = "monte-carlo-tally";
  threshold.primary_unit = ThresholdUnit::dimensionless_relative;
  threshold.relative_target = 5.0e-3;
  threshold.relative_tolerance = 1.0e-2;
  threshold.systematic_bias_max_exclusive = 1.0e-3;
  threshold.confidence_level = 0.95;
  return threshold;
}

constexpr TierThreshold monte_carlo_packet() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::monte_carlo_packet;
  threshold.name = "monte-carlo-packet";
  threshold.primary_unit = ThresholdUnit::roundoff_event;
  threshold.relative_tolerance = std::numeric_limits<double>::epsilon();
  threshold.roundoff_operation_count = 1.0;
  threshold.event_roundoff_required = true;
  return threshold;
}

constexpr TierThreshold smooth_mms_convergence_rate() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::smooth_mms_convergence_rate;
  threshold.name = "smooth-MMS-convergence-rate";
  threshold.primary_unit = ThresholdUnit::dimensionless_rate;
  threshold.convergence_rate_min = 1.8;
  return threshold;
}

constexpr TierThreshold shock_convergence_rate() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::shock_convergence_rate;
  threshold.name = "shock-convergence-rate";
  threshold.primary_unit = ThresholdUnit::dimensionless_rate;
  threshold.convergence_rate_min = 0.8;
  threshold.smooth_region_rate_min = 1.5;
  threshold.shock_position_convergence_required = true;
  return threshold;
}

constexpr TierThreshold anisotropy_acceptance() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::anisotropy_acceptance;
  threshold.name = "anisotropy-acceptance";
  threshold.primary_unit = ThresholdUnit::dimensionless_relative;
  threshold.relative_tolerance = 5.0e-2;
  threshold.production_relative_target = 2.0e-2;
  threshold.production_min_r = 256;
  threshold.production_min_z = 512;
  return threshold;
}

constexpr TierThreshold ulp_pure_geometry() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::ulp_pure_geometry;
  threshold.name = "ULP-pure-geometry";
  threshold.primary_unit = ThresholdUnit::exact_closed_form;
  threshold.ulp_tolerance = 0.0;
  threshold.exact_closed_form_required = true;
  return threshold;
}

constexpr TierThreshold c_eps_n_op() {
  TierThreshold threshold{};
  threshold.tier = ThresholdTier::c_eps_n_op;
  threshold.name = "C*eps*N_op";
  threshold.primary_unit = ThresholdUnit::c_eps_operation_count;
  threshold.c_eps_multiplier = 16.0;
  threshold.c_eps_n_op_required = true;
  return threshold;
}

static constexpr std::array<ThresholdTier, kThresholdTierCount> kTiers = {
    ThresholdTier::deterministic_short,
    ThresholdTier::deterministic_long,
    ThresholdTier::monte_carlo_tally,
    ThresholdTier::monte_carlo_packet,
    ThresholdTier::smooth_mms_convergence_rate,
    ThresholdTier::shock_convergence_rate,
    ThresholdTier::anisotropy_acceptance,
    ThresholdTier::ulp_pure_geometry,
    ThresholdTier::c_eps_n_op,
};

static constexpr std::array<TierThreshold, kThresholdTierCount> kThresholds = {
    deterministic_short(),
    deterministic_long(),
    monte_carlo_tally(),
    monte_carlo_packet(),
    smooth_mms_convergence_rate(),
    shock_convergence_rate(),
    anisotropy_acceptance(),
    ulp_pure_geometry(),
    c_eps_n_op(),
};

double reduction_operation_scale(const std::size_t operation_count,
                                 const ReductionMode mode) {
  if (operation_count == 0) {
    return 0.0;
  }

  switch (mode) {
    case ReductionMode::atomic_add:
      return static_cast<double>(operation_count);
    case ReductionMode::kahan:
    case ReductionMode::pairwise:
      return std::max(1.0, std::log2(static_cast<double>(operation_count)));
  }

  throw UnknownReductionMode{};
}

}  // namespace

const char* UnknownThresholdTier::what() const noexcept {
  return "unknown verification threshold tier";
}

const char* UnknownReductionMode::what() const noexcept {
  return "unknown verification reduction mode";
}

const std::array<ThresholdTier, kThresholdTierCount>& threshold_tiers() {
  return kTiers;
}

const TierThreshold& threshold_for(const ThresholdTier tier) {
  const auto ordinal = static_cast<std::uint8_t>(tier);
  if (ordinal == 0 || ordinal > kThresholds.size()) {
    throw UnknownThresholdTier{};
  }
  return kThresholds[static_cast<std::size_t>(ordinal - 1)];
}

double c_eps_n_op_tolerance(const std::size_t operation_count,
                            const ReductionMode mode,
                            const double C) {
  return C * std::numeric_limits<double>::epsilon() *
         reduction_operation_scale(operation_count, mode);
}

}  // namespace tenryu::verification
