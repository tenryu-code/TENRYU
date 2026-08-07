#include "burn/screening.hpp"

#include <atomic>
#include <string>

#include "burn/screening_device.cuh"
#include "core/error.hpp"

namespace tenryu::burn {
namespace {

void warn_salpeter_invalid_once() {
  static std::atomic<bool> warned{false};
  bool expected = false;
  if (warned.compare_exchange_strong(expected, true,
                                     std::memory_order_relaxed)) {
    core::log_warning(
        "burn screening: non-finite or non-positive Salpeter inputs; "
        "screening factor forced to 1 for the affected evaluations");
  }
}

void warn_salpeter_strong_once() {
  static std::atomic<bool> warned{false};
  bool expected = false;
  if (warned.compare_exchange_strong(expected, true,
                                     std::memory_order_relaxed)) {
    core::log_warning(
        "burn screening: Salpeter h exceeded " +
        std::to_string(screening_device_detail::kSalpeterHMax) +
        " (strong screening, outside the weak-screening model); factor "
        "capped at exp(" +
        std::to_string(screening_device_detail::kSalpeterHMax) + ")");
  }
}

}  // namespace

void burn_screening_emit_warnings(const unsigned int warning_flags) {
  if ((warning_flags & kScreeningWarningInvalid) != 0U) {
    warn_salpeter_invalid_once();
  }
  if ((warning_flags & kScreeningWarningStrong) != 0U) {
    warn_salpeter_strong_once();
  }
}

void burn_screening_factors(const ScreeningMode mode, const double Ti_eV,
                            const double Te_eV, const double ne_cm3,
                            const double n_species[], double F[]) {
  unsigned int warning_flags = 0U;
  burn_screening_factors_core(mode, Ti_eV, Te_eV, ne_cm3, n_species, F,
                              &warning_flags);
  burn_screening_emit_warnings(warning_flags);
}

}  // namespace tenryu::burn
