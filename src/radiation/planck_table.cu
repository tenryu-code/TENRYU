#include "radiation/planck_table.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "materials/eos_table.hpp"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

bool valid_range(const std::vector<double>& range) {
  return range.size() == 2U && std::isfinite(range[0]) && std::isfinite(range[1]) &&
         range[0] > 0.0 && range[1] > range[0];
}

void expand_range(std::vector<double>& range, const double lo, const double hi) {
  if (!(std::isfinite(lo) && std::isfinite(hi) && lo > 0.0 && hi > lo)) {
    return;
  }
  if (range.empty()) {
    range = {lo, hi};
    return;
  }
  range[0] = std::min(range[0], lo);
  range[1] = std::max(range[1], hi);
}

std::vector<double> eos_temperature_range(const core::Config& cfg) {
  std::vector<double> range;
  for (const auto& mat : cfg.materials.materials) {
    if (mat.is_void || !mat.eos_tables) {
      continue;
    }
    const auto& t_grid = mat.eos_tables->electron.T_grid_eV;
    if (t_grid.size() >= 2U) {
      expand_range(range, t_grid.front(), t_grid.back());
    }
  }
  return range;
}

std::string format_range(const std::vector<double>& range) {
  return "[" + std::to_string(range[0]) + ", " + std::to_string(range[1]) + "]";
}

// x f(x) for the normalized Planck density in x = E/T,
// f(x) = (15/pi^4) x^3/(e^x - 1) (same branches as Groups' Planck kernel):
// d/d ln T of the fraction above x is x f(x).
double planck_x_density(const double x) {
  constexpr double kPi = 3.141592653589793238462643383279502884;
  constexpr double kNorm = 15.0 / (kPi * kPi * kPi * kPi);
  if (!(x > 0.0) || !std::isfinite(x)) {
    return 0.0;
  }
  double kernel = 0.0;
  if (x < 1.0e-3) {
    kernel = x * x - 0.5 * x * x * x;
  } else if (x > 500.0) {
    kernel = x * x * x * std::exp(-x);
  } else {
    kernel = (x * x * x) / std::expm1(x);
  }
  return kNorm * x * kernel;
}

}  // namespace

PlanckTable::~PlanckTable() {
  release();
}

PlanckTable::PlanckTable(PlanckTable&& other) noexcept {
  *this = std::move(other);
}

PlanckTable& PlanckTable::operator=(PlanckTable&& other) noexcept {
  if (this != &other) {
    release();

    n_T_ = other.n_T_;
    n_groups_ = other.n_groups_;
    constant_in_T_ = other.constant_in_T_;
    T_grid_host_ = std::move(other.T_grid_host_);
    b_g_host_ = std::move(other.b_g_host_);
    cdf_g_host_ = std::move(other.cdf_g_host_);
    tail_g_host_ = std::move(other.tail_g_host_);
    ln_cdf_g_host_ = std::move(other.ln_cdf_g_host_);
    ln_tail_g_host_ = std::move(other.ln_tail_g_host_);
    dln_cdf_g_host_ = std::move(other.dln_cdf_g_host_);
    dln_tail_g_host_ = std::move(other.dln_tail_g_host_);
    ln_T_grid_host_ = std::move(other.ln_T_grid_host_);
    d_T_grid_ = other.d_T_grid_;
    d_b_g_ = other.d_b_g_;
    d_cdf_g_ = other.d_cdf_g_;
    d_tail_g_ = other.d_tail_g_;
    d_ln_cdf_g_ = other.d_ln_cdf_g_;
    d_ln_tail_g_ = other.d_ln_tail_g_;
    d_dln_cdf_g_ = other.d_dln_cdf_g_;
    d_dln_tail_g_ = other.d_dln_tail_g_;
    d_ln_T_grid_ = other.d_ln_T_grid_;

    other.n_T_ = 0;
    other.n_groups_ = 1;
    other.constant_in_T_ = false;
    other.d_T_grid_ = nullptr;
    other.d_b_g_ = nullptr;
    other.d_cdf_g_ = nullptr;
    other.d_tail_g_ = nullptr;
    other.d_ln_cdf_g_ = nullptr;
    other.d_ln_tail_g_ = nullptr;
    other.d_dln_cdf_g_ = nullptr;
    other.d_dln_tail_g_ = nullptr;
    other.d_ln_T_grid_ = nullptr;
  }
  return *this;
}

void PlanckTable::release() {
  for (double** p : {&d_tail_g_, &d_ln_cdf_g_, &d_ln_tail_g_, &d_dln_cdf_g_, &d_dln_tail_g_,
                     &d_ln_T_grid_}) {
    if (*p != nullptr) {
      cuda_check(cudaFree(*p), "PlanckTable cudaFree interpolation table failed");
      *p = nullptr;
    }
  }
  if (d_cdf_g_ != nullptr) {
    cuda_check(cudaFree(d_cdf_g_), "PlanckTable cudaFree cdf failed");
    d_cdf_g_ = nullptr;
  }
  if (d_b_g_ != nullptr) {
    cuda_check(cudaFree(d_b_g_), "PlanckTable cudaFree b_g failed");
    d_b_g_ = nullptr;
  }
  if (d_T_grid_ != nullptr) {
    cuda_check(cudaFree(d_T_grid_), "PlanckTable cudaFree T_grid failed");
    d_T_grid_ = nullptr;
  }
}

void PlanckTable::build(const Groups& groups,
                        const int n_T,
                        const double T_min_eV,
                        const double T_max_eV) {
  TENRYU_ASSERT(n_T > 1, "PlanckTable::build requires n_T > 1");
  TENRYU_ASSERT(T_min_eV > 0.0, "PlanckTable::build requires T_min > 0");
  TENRYU_ASSERT(T_max_eV > T_min_eV, "PlanckTable::build requires T_max > T_min");

  release();

  n_T_ = n_T;
  n_groups_ = groups.num_groups();
  TENRYU_ASSERT(n_groups_ >= 1, "PlanckTable::build requires at least one group");

  T_grid_host_.assign(static_cast<std::size_t>(n_T_), 0.0);
  b_g_host_.assign(static_cast<std::size_t>(n_T_ * n_groups_), 0.0);
  cdf_g_host_.assign(static_cast<std::size_t>(n_T_ * n_groups_), 0.0);
  dln_cdf_g_host_.assign(static_cast<std::size_t>(n_T_ * n_groups_), 0.0);
  dln_tail_g_host_.assign(static_cast<std::size_t>(n_T_ * n_groups_), 0.0);
  std::vector<double> xf_bound(static_cast<std::size_t>(n_groups_ + 1), 0.0);

  const double log_min = std::log(T_min_eV);
  const double log_max = std::log(T_max_eV);
  std::vector<double> raw_b(static_cast<std::size_t>(n_groups_), 0.0);
  // 2026-07-26 review: the renormalization below silently folds
  // the out-of-range Planck tails back into the configured groups. Track the
  // raw coverage per table temperature so a group structure that misses a
  // significant emission fraction is surfaced at build time (diagnostic only
  // — the renormalized b_g values are unchanged).
  constexpr double kTailDeficitInfo = 1.0e-3;
  double tail_deficit_max = 0.0;
  double tail_deficit_max_T = T_min_eV;
  double T_cov_lo = -1.0;
  double T_cov_hi = -1.0;
  for (int k = 0; k < n_T_; ++k) {
    const double u = static_cast<double>(k) / static_cast<double>(n_T_ - 1);
    const double T = std::exp((1.0 - u) * log_min + u * log_max);
    T_grid_host_[static_cast<std::size_t>(k)] = T;

    if (n_groups_ == 1) {
      b_g_host_[static_cast<std::size_t>(k)] = 1.0;
      cdf_g_host_[static_cast<std::size_t>(k)] = 1.0;
      continue;
    }

    double sum = 0.0;
    for (int g = 0; g < n_groups_; ++g) {
      const double b = std::max(0.0, groups.planck_fraction_raw(g, T));
      raw_b[static_cast<std::size_t>(g)] = b;
      sum += b;
    }
    const double tail_deficit = 1.0 - sum;
    if (tail_deficit > tail_deficit_max) {
      tail_deficit_max = tail_deficit;
      tail_deficit_max_T = T;
    }
    if (tail_deficit <= kTailDeficitInfo) {
      if (T_cov_lo < 0.0) {
        T_cov_lo = T;
      }
      T_cov_hi = T;
    }
    if (sum > 0.0) {
      // Planck fractions b_g are renormalized to sum to 1 over configured
      // groups. Energy from low/high-energy tails outside group boundaries is
      // redistributed into represented groups. This is correct for finite-
      // group IMC: total emitted energy must be allocated entirely among
      // available groups. Users should ensure group boundaries cover the
      // physically relevant spectrum to minimize tail redistribution.
      const double inv_sum = 1.0 / sum;
      for (int g = 0; g < n_groups_; ++g) {
        b_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] =
            raw_b[static_cast<std::size_t>(g)] * inv_sum;
      }
    } else {
      for (int g = 0; g < n_groups_; ++g) {
        b_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] = 0.0;
      }
    }
#ifndef NDEBUG
    {
      double check_sum = 0.0;
      for (int g = 0; g < n_groups_; ++g) {
        check_sum += b_g_host_[static_cast<std::size_t>(k * n_groups_ + g)];
      }
      TENRYU_ASSERT(std::abs(check_sum - 1.0) < 1.0e-12,
                    "PlanckTable::build b_g normalization check failed");
    }
#endif

    double cdf = 0.0;
    for (int g = 0; g < n_groups_; ++g) {
      cdf += b_g_host_[static_cast<std::size_t>(k * n_groups_ + g)];
      cdf_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] = cdf;
    }
    cdf_g_host_[static_cast<std::size_t>(k * n_groups_ + (n_groups_ - 1))] = 1.0;

    // Node slopes for the Hermite interpolation in ln T: with raw
    // cumulative RC_g, raw tail RD_g and raw sum S (so C = RC/S, D = RD/S),
    // d RC_g/du = xf(x_0) - xf(x_{g+1}), d RD_g/du = xf(x_{g+1}) - xf(x_G),
    // d S/du = xf(x_0) - xf(x_G), xf(x) = x f(x), x_k = E_k/T.
    if (sum > 0.0) {
      for (int kb = 0; kb <= n_groups_; ++kb) {
        const double E = (kb < n_groups_) ? groups.energy_lo(kb) : groups.energy_hi(n_groups_ - 1);
        xf_bound[static_cast<std::size_t>(kb)] = planck_x_density(E / T);
      }
      const double dS = xf_bound.front() - xf_bound.back();
      constexpr double kSlopeFloor = 1.0e-300;
      double rc = 0.0;
      for (int g = 0; g < n_groups_; ++g) {
        rc += raw_b[static_cast<std::size_t>(g)];
        const double drc = xf_bound.front() - xf_bound[static_cast<std::size_t>(g + 1)];
        dln_cdf_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] =
            (rc > kSlopeFloor) ? (drc / rc - dS / sum) : 0.0;
      }
      double rd = 0.0;
      for (int g = n_groups_ - 1; g >= 0; --g) {
        const double drd = xf_bound[static_cast<std::size_t>(g + 1)] - xf_bound.back();
        dln_tail_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] =
            (rd > kSlopeFloor) ? (drd / rd - dS / sum) : 0.0;
        rd += raw_b[static_cast<std::size_t>(g)];
      }
    }
  }

  if (n_groups_ > 1) {
    char buf[256];
    if (T_cov_lo < 0.0) {
      std::snprintf(buf, sizeof(buf),
                    "PlanckTable: configured group bounds capture < %.1f%% of "
                    "the Planck emission at EVERY table temperature "
                    "(max tail deficit %.3e at T = %.3e eV) — group range is "
                    "likely mis-sized for this problem",
                    100.0 * (1.0 - kTailDeficitInfo), tail_deficit_max,
                    tail_deficit_max_T);
      core::log_warning(buf);
    } else {
      std::snprintf(buf, sizeof(buf),
                    "PlanckTable: group Planck coverage (tail deficit <= "
                    "%.0e) holds for T in [%.3e, %.3e] eV of table range "
                    "[%.3e, %.3e] eV; max tail deficit %.3e at T = %.3e eV",
                    kTailDeficitInfo, T_cov_lo, T_cov_hi, T_min_eV, T_max_eV,
                    tail_deficit_max, tail_deficit_max_T);
      core::log_info(buf);
    }
  }

  finalize_and_upload("PlanckTable::build");
}

void PlanckTable::finalize_and_upload(const char* who) {
  const std::size_t n_T = static_cast<std::size_t>(n_T_);
  const std::size_t n_g = static_cast<std::size_t>(n_groups_);
  TENRYU_ASSERT(T_grid_host_.size() == n_T && b_g_host_.size() == n_T * n_g &&
                    cdf_g_host_.size() == n_T * n_g && dln_cdf_g_host_.size() == n_T * n_g &&
                    dln_tail_g_host_.size() == n_T * n_g,
                "PlanckTable::finalize_and_upload size mismatch");
  // Tails as suffix sums of b_g (not 1 - C_g, which cancels for the small
  // Wien tails), and the logs the interpolation mixes geometrically.
  constexpr double kLnFloor = 1.0e-300;
  constant_in_T_ = true;
  for (std::size_t k = 1; k < n_T && constant_in_T_; ++k) {
    for (std::size_t g = 0; g < n_g; ++g) {
      if (b_g_host_[k * n_g + g] != b_g_host_[g]) {
        constant_in_T_ = false;
        break;
      }
    }
  }
  tail_g_host_.assign(n_T * n_g, 0.0);
  ln_cdf_g_host_.assign(n_T * n_g, 0.0);
  ln_tail_g_host_.assign(n_T * n_g, 0.0);
  ln_T_grid_host_.assign(n_T, 0.0);
  for (std::size_t k = 0; k < n_T; ++k) {
    ln_T_grid_host_[k] = std::log(T_grid_host_[k]);
    double tail = 0.0;
    for (std::size_t g = n_g; g-- > 0;) {
      tail_g_host_[k * n_g + g] = tail;
      tail += b_g_host_[k * n_g + g];
    }
    for (std::size_t g = 0; g < n_g; ++g) {
      ln_cdf_g_host_[k * n_g + g] = std::log(std::max(cdf_g_host_[k * n_g + g], kLnFloor));
      ln_tail_g_host_[k * n_g + g] = std::log(std::max(tail_g_host_[k * n_g + g], kLnFloor));
    }
  }

  const auto upload = [&](double** dst, const std::vector<double>& src, const char* what) {
    const std::string msg_alloc = std::string(who) + " cudaMalloc " + what + " failed";
    const std::string msg_copy = std::string(who) + " copy " + what + " failed";
    cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), sizeof(double) * src.size()),
               msg_alloc.c_str());
    cuda_check(cudaMemcpy(*dst, src.data(), sizeof(double) * src.size(), cudaMemcpyHostToDevice),
               msg_copy.c_str());
  };
  upload(&d_T_grid_, T_grid_host_, "T_grid");
  upload(&d_b_g_, b_g_host_, "b_g");
  upload(&d_cdf_g_, cdf_g_host_, "cdf");
  upload(&d_tail_g_, tail_g_host_, "tail");
  upload(&d_ln_cdf_g_, ln_cdf_g_host_, "ln_cdf");
  upload(&d_ln_tail_g_, ln_tail_g_host_, "ln_tail");
  upload(&d_dln_cdf_g_, dln_cdf_g_host_, "dln_cdf");
  upload(&d_dln_tail_g_, dln_tail_g_host_, "dln_tail");
  upload(&d_ln_T_grid_, ln_T_grid_host_, "ln_T_grid");
}

void PlanckTable::build_constant_fractions(const Groups& groups,
                                           const std::vector<double>& fractions,
                                           const double T_min_eV,
                                           const double T_max_eV) {
  TENRYU_ASSERT(T_min_eV > 0.0,
                "PlanckTable::build_constant_fractions requires T_min > 0");
  TENRYU_ASSERT(T_max_eV > T_min_eV,
                "PlanckTable::build_constant_fractions requires T_max > T_min");

  release();

  n_T_ = 2;
  n_groups_ = groups.num_groups();
  TENRYU_ASSERT(n_groups_ >= 1,
                "PlanckTable::build_constant_fractions requires >= 1 group");
  TENRYU_ASSERT(static_cast<int>(fractions.size()) == n_groups_,
                "PlanckTable::build_constant_fractions fractions size mismatch");
  double sum = 0.0;
  for (const double f : fractions) {
    TENRYU_ASSERT(std::isfinite(f) && f >= 0.0,
                  "PlanckTable::build_constant_fractions fractions must be "
                  "finite and non-negative");
    sum += f;
  }
  TENRYU_ASSERT(std::abs(sum - 1.0) < 1.0e-12,
                "PlanckTable::build_constant_fractions fractions must sum to 1");

  T_grid_host_ = {T_min_eV, T_max_eV};
  b_g_host_.assign(static_cast<std::size_t>(2 * n_groups_), 0.0);
  cdf_g_host_.assign(static_cast<std::size_t>(2 * n_groups_), 0.0);
  dln_cdf_g_host_.assign(static_cast<std::size_t>(2 * n_groups_), 0.0);
  dln_tail_g_host_.assign(static_cast<std::size_t>(2 * n_groups_), 0.0);
  for (int k = 0; k < 2; ++k) {
    double cdf = 0.0;
    for (int g = 0; g < n_groups_; ++g) {
      const double b = fractions[static_cast<std::size_t>(g)] / sum;
      b_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] = b;
      cdf += b;
      cdf_g_host_[static_cast<std::size_t>(k * n_groups_ + g)] = cdf;
    }
    cdf_g_host_[static_cast<std::size_t>(k * n_groups_ + (n_groups_ - 1))] = 1.0;
  }

  finalize_and_upload("PlanckTable::build_constant_fractions");
}

std::vector<double> resolve_compute_T_range_eV(const core::Config& cfg,
                                               const bool log_auto_derivation) {
  if (!cfg.radiation.compute_T_range_eV.empty()) {
    TENRYU_ASSERT(valid_range(cfg.radiation.compute_T_range_eV),
                  "Radiation.compute_T_range_eV must satisfy [Tmin>0, Tmax>Tmin]");
    return cfg.radiation.compute_T_range_eV;
  }

  std::vector<double> range = eos_temperature_range(cfg);
  std::string source = "EOS table";
  if (!valid_range(range)) {
    range = cfg.radiation.opacity_T_range_eV;
    source = "opacity table";
  }
  if (!valid_range(range)) {
    range = {1.0e-2, 1.0e3};
    source = "fallback default";
  }

  if (log_auto_derivation) {
    core::log_info("Radiation.compute_T_range_eV auto-derived from " + source +
                   ": " + format_range(range) + " eV");
  }
  return range;
}

PlanckTable build_planck_table_from_config(const core::Config& cfg) {
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::vector<double> compute_T_range_eV =
      resolve_compute_T_range_eV(cfg, true);
  std::vector<double> bounds = cfg.radiation.group_bounds_eV;
  if (bounds.empty() || static_cast<int>(bounds.size()) != n_groups + 1) {
    const double Tmin = std::max(compute_T_range_eV[0], 1.0e-3);
    const double Tmax = std::max(compute_T_range_eV[1], Tmin * 1.001);
    bounds = Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
  }

  Groups groups(bounds);
  const auto& planck_cfg = cfg.radiation.planck_fraction;
  if (planck_cfg.method == "tabulate") {
    core::log_warning("Radiation.groups.planck_fraction.method=\"tabulate\" is stored but not "
                      "implemented yet; falling back to computed Planck fractions");
  }

  PlanckTable planck;
  planck.build(groups,
               std::max(planck_cfg.compute_N_T, 2),
               compute_T_range_eV[0],
               compute_T_range_eV[1]);
  return planck;
}

}  // namespace tenryu::radiation
