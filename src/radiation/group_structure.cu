#include "radiation/group_structure.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kKappaFloor = 1.0e-100;

[[nodiscard]] double group_rep_energy(const double lo, const double hi) {
  if (lo > 0.0 && hi > lo) {
    return std::sqrt(lo * hi);
  }
  return 0.5 * std::max(hi, 0.0);
}

[[nodiscard]] bool same_bounds(const std::vector<double>& a,
                               const std::vector<double>& b) {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double interpolate_group_value(const std::vector<double>& reps,
                                             const std::vector<double>& table,
                                             const int source_group_count,
                                             const int ndens,
                                             const int ntemp,
                                             const double target_E,
                                             const int d,
                                             const int t) {
  if (source_group_count <= 1 || reps.empty()) {
    return table[static_cast<std::size_t>(d) * static_cast<std::size_t>(ntemp) +
                 static_cast<std::size_t>(t)];
  }

  int g0 = 0;
  int g1 = 0;
  double u = 0.0;
  if (target_E <= reps.front()) {
    g0 = 0;
    g1 = 0;
  } else if (target_E >= reps.back()) {
    g0 = source_group_count - 1;
    g1 = g0;
  } else {
    const auto upper = std::upper_bound(reps.begin(), reps.end(), target_E);
    g1 = static_cast<int>(upper - reps.begin());
    g0 = std::max(0, g1 - 1);
    const double log0 = std::log(std::max(reps[static_cast<std::size_t>(g0)],
                                          std::numeric_limits<double>::min()));
    const double log1 = std::log(std::max(reps[static_cast<std::size_t>(g1)],
                                          std::numeric_limits<double>::min()));
    const double logE = std::log(std::max(target_E, std::numeric_limits<double>::min()));
    u = (log1 > log0) ? ((logE - log0) / (log1 - log0)) : 0.0;
    u = std::clamp(u, 0.0, 1.0);
  }

  const auto idx = [ndens, ntemp](const int g, const int dd, const int tt) {
    return static_cast<std::size_t>(g) * static_cast<std::size_t>(ndens) *
               static_cast<std::size_t>(ntemp) +
           static_cast<std::size_t>(dd) * static_cast<std::size_t>(ntemp) +
           static_cast<std::size_t>(tt);
  };
  const double k0 = table[idx(g0, d, t)];
  const double k1 = table[idx(g1, d, t)];
  if (g0 == g1) {
    return k0;
  }
  if (k0 > kKappaFloor && k1 > kKappaFloor) {
    return std::exp((1.0 - u) * std::log(k0) + u * std::log(k1));
  }
  return (1.0 - u) * std::max(k0, 0.0) + u * std::max(k1, 0.0);
}

void resample_field(const materials::IonmixOpacityData& input,
                    const std::vector<double>& target_bounds_eV,
                    const std::vector<double>& source_reps,
                    const std::vector<double>& src,
                    std::vector<double>* dst) {
  TENRYU_ASSERT(dst != nullptr, "resample_field requires output");
  const int n_groups = input.ngroups;
  const int ndens = input.ndens;
  const int ntemp = input.ntemp;
  TENRYU_ASSERT(src.size() == static_cast<std::size_t>(n_groups) *
                                  static_cast<std::size_t>(ndens) *
                                  static_cast<std::size_t>(ntemp),
                "resample_field source table size mismatch");
  dst->assign(static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(ndens) *
                  static_cast<std::size_t>(ntemp),
              0.0);
  for (int g = 0; g < n_groups; ++g) {
    const double target_E = group_rep_energy(target_bounds_eV[static_cast<std::size_t>(g)],
                                             target_bounds_eV[static_cast<std::size_t>(g + 1)]);
    for (int d = 0; d < ndens; ++d) {
      for (int t = 0; t < ntemp; ++t) {
        const std::size_t out_idx = input.flat_index(g, d, t);
        (*dst)[out_idx] = interpolate_group_value(source_reps,
                                                  src,
                                                  n_groups,
                                                  ndens,
                                                  ntemp,
                                                  target_E,
                                                  d,
                                                  t);
      }
    }
  }
}

}  // namespace

materials::IonmixOpacityData resample_opacity_groups_to_bounds(
    const materials::IonmixOpacityData& input,
    const std::vector<double>& target_bounds_eV) {
  TENRYU_ASSERT(input.ngroups > 0 && input.ndens > 0 && input.ntemp > 0,
                "resample_opacity_groups_to_bounds requires non-empty opacity table");
  TENRYU_ASSERT(target_bounds_eV.size() == static_cast<std::size_t>(input.ngroups + 1),
                "resample_opacity_groups_to_bounds target bounds size mismatch");
  if (same_bounds(input.bounds_eV, target_bounds_eV)) {
    return input;
  }

  materials::IonmixOpacityData out = input;
  out.bounds_eV = target_bounds_eV;

  std::vector<double> source_reps(static_cast<std::size_t>(input.ngroups), 0.0);
  for (int g = 0; g < input.ngroups; ++g) {
    source_reps[static_cast<std::size_t>(g)] =
        group_rep_energy(input.bounds_eV[static_cast<std::size_t>(g)],
                         input.bounds_eV[static_cast<std::size_t>(g + 1)]);
  }

  resample_field(input, target_bounds_eV, source_reps, input.kappa_R, &out.kappa_R);
  resample_field(input, target_bounds_eV, source_reps, input.kappa_PA, &out.kappa_PA);
  resample_field(input, target_bounds_eV, source_reps, input.kappa_PE, &out.kappa_PE);
  return out;
}

}  // namespace tenryu::radiation
