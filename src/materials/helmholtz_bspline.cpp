#include "materials/helmholtz_bspline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "materials/helmholtz_spline.hpp"

namespace tenryu::materials {
namespace {

struct BasisEval1D {
  std::vector<int> indices;
  std::vector<double> value;
  std::vector<double> deriv1;
  std::vector<double> deriv2;
};

struct ErrorStats {
  double max_rel = 0.0;
  double rho_at_max = 0.0;
  double T_at_max = 0.0;
  bool initialized = false;

  void update(const double rel, const double rho, const double T) {
    if (!initialized || rel > max_rel) {
      max_rel = rel;
      rho_at_max = rho;
      T_at_max = T;
      initialized = true;
    }
  }
};

struct PositivityStats {
  double min_value = std::numeric_limits<double>::infinity();
  double rho_at_min = 0.0;
  double T_at_min = 0.0;
  std::size_t negative_count = 0;
  bool initialized = false;

  void update(const double value, const double rho, const double T) {
    if (!initialized || value < min_value) {
      min_value = value;
      rho_at_min = rho;
      T_at_min = T;
      initialized = true;
    }
    if (value <= 0.0 || !std::isfinite(value)) {
      ++negative_count;
    }
  }
};

struct TemperatureInversionStats {
  ErrorStats all{};
  ErrorStats low{};
  std::size_t bracket_failures = 0;
  std::size_t convergence_failures = 0;
};

struct TemperatureInversionResult {
  double T = 0.0;
  bool bracketed = false;
  bool converged = false;
};

struct MatrixStats {
  double diag_min = std::numeric_limits<double>::infinity();
  double diag_max = 0.0;
  double coeff_min = std::numeric_limits<double>::infinity();
  double coeff_max = -std::numeric_limits<double>::infinity();
  double coeff_mean = 0.0;
  double rel_residual = 0.0;
  double rel_data_residual = 0.0;
};

struct SolveAttempt {
  bool success = false;
  std::vector<double> coeffs;
  MatrixStats stats{};
  std::size_t cv_nonpositive = std::numeric_limits<std::size_t>::max();
  std::size_t drho_nonpositive = std::numeric_limits<std::size_t>::max();
  std::size_t cs2_nonpositive = std::numeric_limits<std::size_t>::max();
};

struct StabilityCounts {
  std::size_t cv_nonpositive = 0;
  std::size_t drho_nonpositive = 0;
  std::size_t cs2_nonpositive = 0;
};

enum class ConstraintKind {
  Cv,
  Drho,
};

struct ConstraintSpec {
  ConstraintKind kind = ConstraintKind::Cv;
  std::size_t flat_index = 0;
  double rhs = 0.0;
  std::vector<std::size_t> indices;
  std::vector<double> values;
};

[[nodiscard]] HelmholtzBSplineThermo eval_thermo_host(const HelmholtzBSpline1T& fit,
                                                      double rho,
                                                      double T_eV);
[[nodiscard]] bool cholesky_factor_inplace(std::vector<double>& matrix, std::size_t n);
void cholesky_solve_inplace(const std::vector<double>& factor,
                            std::vector<double>& rhs,
                            std::size_t n);

[[nodiscard]] std::string_view constraint_kind_name(const ConstraintKind kind) {
  switch (kind) {
    case ConstraintKind::Cv:
      return "cv";
    case ConstraintKind::Drho:
      return "drho";
  }
  return "unknown";
}

[[nodiscard]] std::string format_error_stat(const ErrorStats& s) {
  std::ostringstream oss;
  if (!s.initialized) {
    oss << "n/a";
  } else {
    oss << s.max_rel << " @ rho=" << s.rho_at_max << ", T=" << s.T_at_max << " eV";
  }
  return oss.str();
}

[[nodiscard]] std::string format_positivity_stat(const PositivityStats& s) {
  std::ostringstream oss;
  if (!s.initialized) {
    oss << "n/a";
  } else {
    oss << "min=" << s.min_value << " @ rho=" << s.rho_at_min << ", T=" << s.T_at_min
        << " eV, nonpositive=" << s.negative_count;
  }
  return oss.str();
}

[[nodiscard]] std::string format_matrix_stat(const MatrixStats& s) {
  std::ostringstream oss;
  oss << "diag[min,max]=(" << s.diag_min << ", " << s.diag_max << "), coeff[min,max,mean]=("
      << s.coeff_min << ", " << s.coeff_max << ", " << s.coeff_mean
      << "), rel_residual=" << s.rel_residual
      << ", rel_data_residual=" << s.rel_data_residual;
  return oss.str();
}

[[nodiscard]] std::size_t coeff_index(const std::size_t ix,
                                      const std::size_t iy,
                                      const std::size_t n_basis_x) {
  return iy * n_basis_x + ix;
}

[[nodiscard]] std::vector<double> reduce_grid_stride(const std::vector<double>& grid,
                                                     const int stride_raw) {
  TENRYU_ASSERT(!grid.empty(), "reduce_grid_stride requires non-empty grid");
  const std::size_t stride = static_cast<std::size_t>(std::max(stride_raw, 1));
  std::vector<double> reduced;
  reduced.reserve(grid.size() / stride + 2);
  for (std::size_t i = 0; i < grid.size(); i += stride) {
    reduced.push_back(grid[i]);
  }
  if (reduced.back() != grid.back()) {
    reduced.push_back(grid.back());
  }
  return reduced;
}

[[nodiscard]] double median_abs(std::vector<double> values) {
  if (values.empty()) {
    return 0.0;
  }
  for (double& v : values) {
    v = std::fabs(v);
  }
  const std::size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
  return values[mid];
}

[[nodiscard]] std::vector<double> build_open_clamped_knot_vector(const std::vector<double>& grid,
                                                                 const int degree) {
  TENRYU_ASSERT(grid.size() >= 2, "HelmholtzBSpline requires at least two grid points");
  TENRYU_ASSERT(degree >= 1, "HelmholtzBSpline degree must be >= 1");

  const std::size_t n_basis = grid.size() + static_cast<std::size_t>(degree) - 1;
  std::vector<double> knots(n_basis + static_cast<std::size_t>(degree) + 1, 0.0);
  std::size_t k = 0;
  for (int r = 0; r <= degree; ++r) {
    knots[k++] = grid.front();
  }
  for (std::size_t i = 1; i + 1 < grid.size(); ++i) {
    knots[k++] = grid[i];
  }
  for (int r = 0; r <= degree; ++r) {
    knots[k++] = grid.back();
  }
  TENRYU_ASSERT(k == knots.size(), "HelmholtzBSpline knot construction size mismatch");
  return knots;
}

[[nodiscard]] int find_span(const std::vector<double>& knots,
                            const int degree,
                            const std::size_t n_basis,
                            const double u_raw) {
  TENRYU_ASSERT(n_basis >= static_cast<std::size_t>(degree + 1),
                "HelmholtzBSpline n_basis too small");
  const int n = static_cast<int>(n_basis) - 1;
  const double u_min = knots[static_cast<std::size_t>(degree)];
  const double u_max = knots[n + 1];
  const double u = std::clamp(u_raw, u_min, u_max);
  if (u >= u_max) {
    return n;
  }
  if (u <= u_min) {
    return degree;
  }
  int low = degree;
  int high = n + 1;
  int mid = (low + high) / 2;
  while (u < knots[static_cast<std::size_t>(mid)] ||
         u >= knots[static_cast<std::size_t>(mid + 1)]) {
    if (u < knots[static_cast<std::size_t>(mid)]) {
      high = mid;
    } else {
      low = mid;
    }
    mid = (low + high) / 2;
  }
  return mid;
}

void ders_basis_funs(const int span,
                     const double u,
                     const int degree,
                     const int max_deriv,
                     const std::vector<double>& knots,
                     std::vector<double>& ders) {
  const int p = degree;
  const int n = max_deriv;
  ders.assign(static_cast<std::size_t>(n + 1) * static_cast<std::size_t>(p + 1), 0.0);
  std::vector<double> ndu(static_cast<std::size_t>(p + 1) * static_cast<std::size_t>(p + 1),
                          0.0);
  std::vector<double> left(static_cast<std::size_t>(p + 1), 0.0);
  std::vector<double> right(static_cast<std::size_t>(p + 1), 0.0);
  auto ndu_at = [&](const int i, const int j) -> double& {
    return ndu[static_cast<std::size_t>(i) * static_cast<std::size_t>(p + 1) +
               static_cast<std::size_t>(j)];
  };
  auto ders_at = [&](const int k, const int j) -> double& {
    return ders[static_cast<std::size_t>(k) * static_cast<std::size_t>(p + 1) +
                static_cast<std::size_t>(j)];
  };

  ndu_at(0, 0) = 1.0;
  for (int j = 1; j <= p; ++j) {
    left[static_cast<std::size_t>(j)] =
        u - knots[static_cast<std::size_t>(span + 1 - j)];
    right[static_cast<std::size_t>(j)] =
        knots[static_cast<std::size_t>(span + j)] - u;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      ndu_at(j, r) = right[static_cast<std::size_t>(r + 1)] +
                     left[static_cast<std::size_t>(j - r)];
      const double temp =
          (std::fabs(ndu_at(j, r)) > 1.0e-300) ? (ndu_at(r, j - 1) / ndu_at(j, r)) : 0.0;
      ndu_at(r, j) = saved + right[static_cast<std::size_t>(r + 1)] * temp;
      saved = left[static_cast<std::size_t>(j - r)] * temp;
    }
    ndu_at(j, j) = saved;
  }

  for (int j = 0; j <= p; ++j) {
    ders_at(0, j) = ndu_at(j, p);
  }

  std::vector<double> a(2 * static_cast<std::size_t>(p + 1), 0.0);
  auto a_at = [&](const int s, const int j) -> double& {
    return a[static_cast<std::size_t>(s) * static_cast<std::size_t>(p + 1) +
             static_cast<std::size_t>(j)];
  };
  for (int r = 0; r <= p; ++r) {
    int s1 = 0;
    int s2 = 1;
    a_at(0, 0) = 1.0;
    for (int k = 1; k <= n; ++k) {
      double d = 0.0;
      const int rk = r - k;
      const int pk = p - k;
      int j1 = 0;
      int j2 = 0;
      if (r >= k) {
        a_at(s2, 0) = a_at(s1, 0) / ndu_at(pk + 1, rk);
        d = a_at(s2, 0) * ndu_at(rk, pk);
      }
      if (rk >= -1) {
        j1 = 1;
      } else {
        j1 = -rk;
      }
      if (r - 1 <= pk) {
        j2 = k - 1;
      } else {
        j2 = p - r;
      }
      for (int j = j1; j <= j2; ++j) {
        a_at(s2, j) = (a_at(s1, j) - a_at(s1, j - 1)) / ndu_at(pk + 1, rk + j);
        d += a_at(s2, j) * ndu_at(rk + j, pk);
      }
      if (r <= pk) {
        a_at(s2, k) = -a_at(s1, k - 1) / ndu_at(pk + 1, r);
        d += a_at(s2, k) * ndu_at(r, pk);
      }
      ders_at(k, r) = d;
      std::swap(s1, s2);
    }
  }

  int factorial_scale = p;
  for (int k = 1; k <= n; ++k) {
    for (int j = 0; j <= p; ++j) {
      ders_at(k, j) *= factorial_scale;
    }
    factorial_scale *= (p - k);
  }
}

[[nodiscard]] BasisEval1D evaluate_basis_1d(const std::vector<double>& knots,
                                            const int degree,
                                            const std::size_t n_basis,
                                            const double u_raw) {
  BasisEval1D out;
  out.indices.resize(static_cast<std::size_t>(degree + 1));
  out.value.resize(static_cast<std::size_t>(degree + 1), 0.0);
  out.deriv1.resize(static_cast<std::size_t>(degree + 1), 0.0);
  out.deriv2.resize(static_cast<std::size_t>(degree + 1), 0.0);

  const double u_min = knots[static_cast<std::size_t>(degree)];
  const double u_max = knots[n_basis];
  double u = std::clamp(u_raw, u_min, u_max);
  // Derivatives at clamped end knots are one-sided; evaluate from the interior limit.
  if (u <= u_min) {
    u = std::nextafter(u_min, u_max);
  } else if (u >= u_max) {
    u = std::nextafter(u_max, u_min);
  }
  const int span = find_span(knots, degree, n_basis, u);
  std::vector<double> ders;
  ders_basis_funs(span, u, degree, 2, knots, ders);
  for (int j = 0; j <= degree; ++j) {
    out.indices[static_cast<std::size_t>(j)] = span - degree + j;
    out.value[static_cast<std::size_t>(j)] = ders[static_cast<std::size_t>(j)];
    out.deriv1[static_cast<std::size_t>(j)] =
        ders[static_cast<std::size_t>(degree + 1 + j)];
    out.deriv2[static_cast<std::size_t>(j)] =
        ders[2 * static_cast<std::size_t>(degree + 1) + static_cast<std::size_t>(j)];
  }
  return out;
}

void accumulate_sparse_row(std::vector<double>& ata,
                           std::vector<double>& rhs,
                           const std::size_t n_coeffs,
                           const std::vector<std::size_t>& indices,
                           const std::vector<double>& values,
                           const double target) {
  TENRYU_ASSERT(indices.size() == values.size(), "HelmholtzBSpline sparse row size mismatch");
  for (std::size_t a = 0; a < indices.size(); ++a) {
    const std::size_t ia = indices[a];
    const double va = values[a];
    rhs[ia] += va * target;
    double* row = ata.data() + ia * n_coeffs;
    for (std::size_t b = 0; b < indices.size(); ++b) {
      row[indices[b]] += va * values[b];
    }
  }
}

void add_second_difference_regularization(std::vector<double>& ata,
                                          std::vector<double>& rhs,
                                          const std::size_t n_basis_x,
                                          const std::size_t n_basis_y,
                                          const double lambda) {
  if (!(lambda > 0.0)) {
    return;
  }
  const double w = std::sqrt(lambda);
  const std::size_t n_coeffs = n_basis_x * n_basis_y;
  std::vector<std::size_t> indices(3);
  std::vector<double> values = {w, -2.0 * w, w};

  for (std::size_t iy = 0; iy < n_basis_y; ++iy) {
    for (std::size_t ix = 1; ix + 1 < n_basis_x; ++ix) {
      indices[0] = coeff_index(ix - 1, iy, n_basis_x);
      indices[1] = coeff_index(ix, iy, n_basis_x);
      indices[2] = coeff_index(ix + 1, iy, n_basis_x);
      accumulate_sparse_row(ata, rhs, n_coeffs, indices, values, 0.0);
    }
  }
  for (std::size_t iy = 1; iy + 1 < n_basis_y; ++iy) {
    for (std::size_t ix = 0; ix < n_basis_x; ++ix) {
      indices[0] = coeff_index(ix, iy - 1, n_basis_x);
      indices[1] = coeff_index(ix, iy, n_basis_x);
      indices[2] = coeff_index(ix, iy + 1, n_basis_x);
      accumulate_sparse_row(ata, rhs, n_coeffs, indices, values, 0.0);
    }
  }
}

void add_gauge_condition(std::vector<double>& ata,
                         const std::size_t n_coeffs,
                         const std::size_t gauge_index,
                         const double gauge_weight) {
  if (!(gauge_weight > 0.0)) {
    return;
  }
  ata[gauge_index * n_coeffs + gauge_index] += gauge_weight * gauge_weight;
}

void add_cv_reference_rows(std::vector<double>& ata,
                           std::vector<double>& rhs,
                           const std::size_t n_coeff,
                           const std::vector<BasisEval1D>& bx,
                           const std::vector<BasisEval1D>& by,
                           const std::vector<double>& cv_targets,
                           const std::size_t n_basis_x,
                           const double phi_scale,
                           const double cv_floor,
                           const double lambda_cv) {
  if (!(lambda_cv > 0.0)) {
    return;
  }
  const double lambda_sqrt = std::sqrt(lambda_cv);
  std::vector<std::size_t> row_indices;
  std::vector<double> row_values;
  row_indices.reserve(36);
  row_values.reserve(36);
  for (std::size_t j = 0; j < by.size(); ++j) {
    for (std::size_t i = 0; i < bx.size(); ++i) {
      const std::size_t idx = j * bx.size() + i;
      const double wCv = lambda_sqrt / std::max(std::fabs(cv_targets[idx]), cv_floor);
      row_indices.clear();
      row_values.clear();
      for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
        for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
          const std::size_t cidx =
              coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                          static_cast<std::size_t>(by[j].indices[ay]),
                          n_basis_x);
          row_indices.push_back(cidx);
          row_values.push_back(wCv * (-phi_scale) * bx[i].value[ax] *
                               (by[j].deriv1[ay] + by[j].deriv2[ay]));
        }
      }
      accumulate_sparse_row(ata, rhs, n_coeff, row_indices, row_values, wCv * cv_targets[idx]);
    }
  }
}

void add_drho_reference_rows(std::vector<double>& ata,
                             std::vector<double>& rhs,
                             const std::size_t n_coeff,
                             const std::vector<BasisEval1D>& bx,
                             const std::vector<BasisEval1D>& by,
                             const std::vector<double>& drho_targets,
                             const std::vector<double>& T_grid_eV,
                             const std::size_t n_basis_x,
                             const double phi_scale,
                             const double drho_floor,
                             const double lambda_drho) {
  if (!(lambda_drho > 0.0)) {
    return;
  }
  const double lambda_sqrt = std::sqrt(lambda_drho);
  std::vector<std::size_t> row_indices;
  std::vector<double> row_values;
  row_indices.reserve(36);
  row_values.reserve(36);
  for (std::size_t j = 0; j < by.size(); ++j) {
    const double T = T_grid_eV[j];
    for (std::size_t i = 0; i < bx.size(); ++i) {
      const std::size_t idx = j * bx.size() + i;
      const double wR = lambda_sqrt / std::max(std::fabs(drho_targets[idx]), drho_floor);
      row_indices.clear();
      row_values.clear();
      for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
        for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
          const std::size_t cidx =
              coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                          static_cast<std::size_t>(by[j].indices[ay]),
                          n_basis_x);
          row_indices.push_back(cidx);
          row_values.push_back(wR * T * phi_scale *
                               (bx[i].deriv1[ax] + bx[i].deriv2[ax]) * by[j].value[ay]);
        }
      }
      accumulate_sparse_row(ata, rhs, n_coeff, row_indices, row_values, wR * drho_targets[idx]);
    }
  }
}

[[nodiscard]] ConstraintSpec build_constraint_spec(
    const ConstraintKind kind,
    const std::size_t flat_idx,
    const std::vector<BasisEval1D>& bx,
    const std::vector<BasisEval1D>& by,
    const std::vector<double>& T_grid_eV,
    const std::size_t n_basis_x,
    const double phi_scale,
    const double rhs_value) {
  const std::size_t nx = bx.size();
  TENRYU_ASSERT(nx > 0 && by.size() > 0, "HelmholtzBSpline constraint build requires basis data");
  const std::size_t i = flat_idx % nx;
  const std::size_t j = flat_idx / nx;
  TENRYU_ASSERT(j < by.size(), "HelmholtzBSpline constraint build index out of bounds");

  ConstraintSpec out;
  out.kind = kind;
  out.flat_index = flat_idx;
  out.rhs = rhs_value;
  out.indices.reserve((bx[i].indices.size()) * (by[j].indices.size()));
  out.values.reserve((bx[i].indices.size()) * (by[j].indices.size()));
  const double T = T_grid_eV[j];

  for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
    for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
      const std::size_t cidx =
          coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                      static_cast<std::size_t>(by[j].indices[ay]),
                      n_basis_x);
      double value = 0.0;
      if (kind == ConstraintKind::Cv) {
        value = (-phi_scale) * bx[i].value[ax] * (by[j].deriv1[ay] + by[j].deriv2[ay]);
      } else {
        value = T * phi_scale * (bx[i].deriv1[ax] + bx[i].deriv2[ax]) * by[j].value[ay];
      }
      out.indices.push_back(cidx);
      out.values.push_back(value);
    }
  }
  return out;
}

void dense_row_from_constraint(const ConstraintSpec& spec,
                               const std::vector<double>& scale,
                               const std::size_t n_coeff,
                               std::vector<double>& dense_row) {
  dense_row.assign(n_coeff, 0.0);
  for (std::size_t k = 0; k < spec.indices.size(); ++k) {
    const std::size_t idx = spec.indices[k];
    dense_row[idx] = spec.values[k] * scale[idx];
  }
}

[[nodiscard]] bool solve_spd_with_jitter(const std::vector<double>& matrix,
                                         std::vector<double>& rhs,
                                         const std::size_t n) {
  double diag_mean = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    diag_mean += std::fabs(matrix[i * n + i]);
  }
  diag_mean /= std::max<std::size_t>(n, 1);
  double jitter = 0.0;
  for (int attempt = 0; attempt < 8; ++attempt) {
    std::vector<double> trial = matrix;
    if (jitter > 0.0) {
      for (std::size_t i = 0; i < n; ++i) {
        trial[i * n + i] += jitter;
      }
    }
    if (cholesky_factor_inplace(trial, n)) {
      cholesky_solve_inplace(trial, rhs, n);
      return true;
    }
    jitter = (attempt == 0) ? (1.0e-12 * std::max(diag_mean, 1.0)) : (jitter * 100.0);
  }
  return false;
}

void add_cv_violation_penalties(std::vector<double>& ata,
                                std::vector<double>& rhs,
                                const std::size_t n_coeff,
                                const std::vector<BasisEval1D>& bx,
                                const std::vector<BasisEval1D>& by,
                                const std::vector<char>& violated,
                                const std::vector<double>& eps_targets,
                                const std::size_t n_basis_x,
                                const double phi_scale,
                                const double penalty_weight) {
  if (!(penalty_weight > 0.0)) {
    return;
  }
  std::vector<std::size_t> row_indices;
  std::vector<double> row_values;
  row_indices.reserve(36);
  row_values.reserve(36);
  for (std::size_t j = 0; j < by.size(); ++j) {
    for (std::size_t i = 0; i < bx.size(); ++i) {
      const std::size_t idx = j * bx.size() + i;
      if (!violated[idx]) {
        continue;
      }
      row_indices.clear();
      row_values.clear();
      for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
        for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
          const std::size_t cidx =
              coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                          static_cast<std::size_t>(by[j].indices[ay]),
                          n_basis_x);
          row_indices.push_back(cidx);
          row_values.push_back(penalty_weight * (-phi_scale) * bx[i].value[ax] *
                               (by[j].deriv1[ay] + by[j].deriv2[ay]));
        }
      }
      accumulate_sparse_row(ata, rhs, n_coeff, row_indices, row_values,
                            penalty_weight * eps_targets[idx]);
    }
  }
}

void add_drho_violation_penalties(std::vector<double>& ata,
                                  std::vector<double>& rhs,
                                  const std::size_t n_coeff,
                                  const std::vector<BasisEval1D>& bx,
                                  const std::vector<BasisEval1D>& by,
                                  const std::vector<char>& violated,
                                  const std::vector<double>& eps_targets,
                                  const std::vector<double>& T_grid_eV,
                                  const std::size_t n_basis_x,
                                  const double phi_scale,
                                  const double penalty_weight) {
  if (!(penalty_weight > 0.0)) {
    return;
  }
  std::vector<std::size_t> row_indices;
  std::vector<double> row_values;
  row_indices.reserve(36);
  row_values.reserve(36);
  for (std::size_t j = 0; j < by.size(); ++j) {
    const double T = T_grid_eV[j];
    for (std::size_t i = 0; i < bx.size(); ++i) {
      const std::size_t idx = j * bx.size() + i;
      if (!violated[idx]) {
        continue;
      }
      row_indices.clear();
      row_values.clear();
      for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
        for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
          const std::size_t cidx =
              coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                          static_cast<std::size_t>(by[j].indices[ay]),
                          n_basis_x);
          row_indices.push_back(cidx);
          row_values.push_back(penalty_weight * T * phi_scale *
                               (bx[i].deriv1[ax] + bx[i].deriv2[ax]) * by[j].value[ay]);
        }
      }
      accumulate_sparse_row(ata, rhs, n_coeff, row_indices, row_values,
                            penalty_weight * eps_targets[idx]);
    }
  }
}

[[nodiscard]] bool cholesky_factor_inplace(std::vector<double>& matrix, const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j <= i; ++j) {
      double sum = matrix[i * n + j];
      for (std::size_t k = 0; k < j; ++k) {
        sum -= matrix[i * n + k] * matrix[j * n + k];
      }
      if (i == j) {
        if (!(sum > 0.0) || !std::isfinite(sum)) {
          return false;
        }
        matrix[i * n + i] = std::sqrt(sum);
      } else {
        matrix[i * n + j] = sum / matrix[j * n + j];
      }
    }
    for (std::size_t j = i + 1; j < n; ++j) {
      matrix[i * n + j] = 0.0;
    }
  }
  return true;
}

void cholesky_solve_inplace(const std::vector<double>& factor,
                            std::vector<double>& rhs,
                            const std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    double sum = rhs[i];
    for (std::size_t j = 0; j < i; ++j) {
      sum -= factor[i * n + j] * rhs[j];
    }
    rhs[i] = sum / factor[i * n + i];
  }

  for (std::size_t ii = 0; ii < n; ++ii) {
    const std::size_t i = n - 1 - ii;
    double sum = rhs[i];
    for (std::size_t j = i + 1; j < n; ++j) {
      sum -= factor[j * n + i] * rhs[j];
    }
    rhs[i] = sum / factor[i * n + i];
  }
}

[[nodiscard]] double compute_relative_residual(const std::vector<double>& ata,
                                               const std::vector<double>& rhs,
                                               const std::vector<double>& x,
                                               const std::size_t n) {
  double num2 = 0.0;
  double den2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double ax = 0.0;
    const double* row = ata.data() + i * n;
    for (std::size_t j = 0; j < n; ++j) {
      ax += row[j] * x[j];
    }
    const double r = ax - rhs[i];
    num2 += r * r;
    den2 += rhs[i] * rhs[i];
  }
  return std::sqrt(num2 / std::max(den2, 1.0e-300));
}

[[nodiscard]] double compute_data_relative_residual(const HelmholtzBSpline1T& fit,
                                                    const EOSTable& table,
                                                    const double p_floor,
                                                    const double e_floor) {
  double num2 = 0.0;
  double den2 = 0.0;
  for (std::size_t j = 0; j < table.n_T(); ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < table.n_rho(); ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = table.rho_grid[i];
      const HelmholtzBSplineEval phi = fit.eval(table.log_rho_grid[i], table.log_T_grid[j]);
      const double p_eval = rho * T * phi.phi_x;
      const double e_eval = -T * phi.phi_y;
      const double wP = 1.0 / std::max(std::fabs(table.P_table[idx]), p_floor);
      const double wE = 1.0 / std::max(std::fabs(table.e_table[idx]), e_floor);
      const double rP = wP * (p_eval - table.P_table[idx]);
      const double rE = wE * (e_eval - table.e_table[idx]);
      num2 += rP * rP + rE * rE;
      den2 += wP * wP * table.P_table[idx] * table.P_table[idx] +
              wE * wE * table.e_table[idx] * table.e_table[idx];
    }
  }
  return std::sqrt(num2 / std::max(den2, 1.0e-300));
}

[[nodiscard]] StabilityCounts evaluate_stability_counts(const HelmholtzBSpline1T& fit,
                                                        const EOSTable& table) {
  StabilityCounts out{};
  for (std::size_t j = 0; j < table.n_T(); ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < table.n_rho(); ++i) {
      const HelmholtzBSplineThermo thermo = eval_thermo_host(fit, table.rho_grid[i], T);
      if (!(thermo.cv > 0.0) || !std::isfinite(thermo.cv)) {
        ++out.cv_nonpositive;
      }
      if (!(thermo.dP_drho_T > 0.0) || !std::isfinite(thermo.dP_drho_T)) {
        ++out.drho_nonpositive;
      }
      const double cs2 = thermo.sound_speed * thermo.sound_speed;
      if (!(cs2 > 0.0) || !std::isfinite(cs2)) {
        ++out.cs2_nonpositive;
      }
    }
  }
  return out;
}

[[nodiscard]] HelmholtzBSplineThermo eval_thermo_host(const HelmholtzBSpline1T& fit,
                                                      const double rho,
                                                      const double T_eV) {
  HelmholtzBSplineThermo out{};
  if (!(rho > 0.0) || !(T_eV > 0.0) || fit.empty()) {
    return out;
  }
  const HelmholtzBSplineEval phi = fit.eval(std::log(rho), std::log(T_eV));
  const double cv = -(phi.phi_y + phi.phi_yy);
  const double dP_drho_T = T_eV * (phi.phi_x + phi.phi_xx);
  const double dP_dT_rho = rho * (phi.phi_x + phi.phi_xy);
  const double cs2 = dP_drho_T +
                     T_eV * dP_dT_rho * dP_dT_rho /
                         std::max(rho * rho * std::max(cv, 1.0e-30), 1.0e-30);
  out.pressure = rho * T_eV * phi.phi_x;
  out.energy = -T_eV * phi.phi_y;
  out.cv = cv;
  out.dP_drho_T = dP_drho_T;
  out.dP_dT_rho = dP_dT_rho;
  out.sound_speed = (cs2 > 0.0 && std::isfinite(cs2)) ? std::sqrt(cs2) : 0.0;
  return out;
}

[[nodiscard]] TemperatureInversionResult temperature_from_energy_host(
    const HelmholtzBSpline1T& fit,
    const double rho,
    const double e_target) {
  TemperatureInversionResult out{};
  if (!(rho > 0.0) || fit.empty()) {
    return out;
  }

  const double log_rho =
      std::clamp(std::log(rho), fit.log_rho_grid.front(), fit.log_rho_grid.back());
  double y_lo = fit.log_T_grid.front();
  double y_hi = fit.log_T_grid.back();
  double e_lo = eval_thermo_host(fit, rho, std::exp(y_lo)).energy;
  if (e_target <= e_lo) {
    out.T = std::exp(y_lo);
    out.bracketed = true;
    out.converged = true;
    return out;
  }
  const double e_hi_global = eval_thermo_host(fit, rho, std::exp(y_hi)).energy;
  if (e_target >= e_hi_global) {
    out.T = std::exp(y_hi);
    out.bracketed = true;
    out.converged = true;
    return out;
  }

  double y_prev = y_lo;
  double e_prev = e_lo;
  for (std::size_t j = 1; j < fit.log_T_grid.size(); ++j) {
    const double y_curr = fit.log_T_grid[j];
    const double e_curr = eval_thermo_host(fit, rho, std::exp(y_curr)).energy;
    if ((e_target >= e_prev && e_target <= e_curr) ||
        (e_target <= e_prev && e_target >= e_curr)) {
      y_lo = y_prev;
      y_hi = y_curr;
      e_lo = e_prev;
      double e_hi = e_curr;
      out.bracketed = true;
      if (e_hi < e_lo) {
        std::swap(y_lo, y_hi);
        std::swap(e_lo, e_hi);
      }
      if (e_target <= e_lo) {
        out.T = std::exp(y_lo);
        out.converged = true;
        return out;
      }
      if (e_target >= e_hi) {
        out.T = std::exp(y_hi);
        out.converged = true;
        return out;
      }
      double y = y_lo + (e_target - e_lo) * (y_hi - y_lo) / std::max(e_hi - e_lo, 1.0e-30);
      for (int iter = 0; iter < 32; ++iter) {
        const HelmholtzBSplineThermo thermo = eval_thermo_host(fit, rho, std::exp(y));
        const double g = thermo.energy - e_target;
        if (std::fabs(g) <= 1.0e-10 * std::max(std::fabs(e_target), 1.0)) {
          out.T = std::exp(y);
          out.converged = true;
          return out;
        }
        if (g > 0.0) {
          y_hi = y;
        } else {
          y_lo = y;
        }
        const double gp = std::exp(y) * thermo.cv;
        double y_new = 0.5 * (y_lo + y_hi);
        if (std::fabs(gp) > 1.0e-30 && std::isfinite(gp)) {
          const double y_trial = y - g / gp;
          if (y_trial > y_lo && y_trial < y_hi) {
            y_new = y_trial;
          }
        }
        y = y_new;
      }
      out.T = std::exp(0.5 * (y_lo + y_hi));
      return out;
    }
    y_prev = y_curr;
    e_prev = e_curr;
  }

  out.T = std::exp(fit.log_T_grid.back());
  return out;
}

void log_fit_report(const HelmholtzBSpline1T& fit,
                    const EOSTable& table,
                    const HelmholtzBSplineFitOptions& options,
                    const std::string_view label) {
  ErrorStats p_all{};
  ErrorStats e_all{};
  ErrorStats p_low{};
  ErrorStats e_low{};
  PositivityStats cv_stats{};
  PositivityStats cs2_stats{};
  TemperatureInversionStats inv_stats{};

  const double p_floor =
      std::max(options.relative_floor_fraction * median_abs(table.P_table), 1.0e-30);
  const double e_floor =
      std::max(options.relative_floor_fraction * median_abs(table.e_table), 1.0e-30);

  for (std::size_t j = 0; j < table.n_T(); ++j) {
    const double T = table.T_grid_eV[j];
    const bool is_low_T = (T >= 0.1 && T <= 10.0);
    for (std::size_t i = 0; i < table.n_rho(); ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = table.rho_grid[i];
      const HelmholtzBSplineThermo thermo = eval_thermo_host(fit, rho, T);
      const double p_rel =
          std::fabs(thermo.pressure - table.P_table[idx]) /
          std::max(std::fabs(table.P_table[idx]), p_floor);
      const double e_rel =
          std::fabs(thermo.energy - table.e_table[idx]) /
          std::max(std::fabs(table.e_table[idx]), e_floor);
      p_all.update(p_rel, rho, T);
      e_all.update(e_rel, rho, T);
      if (is_low_T) {
        p_low.update(p_rel, rho, T);
        e_low.update(e_rel, rho, T);
      }
      cv_stats.update(thermo.cv, rho, T);
      const double cs2 = thermo.sound_speed * thermo.sound_speed;
      cs2_stats.update(cs2, rho, T);

      const TemperatureInversionResult inv =
          temperature_from_energy_host(fit, rho, table.e_table[idx]);
      if (!inv.bracketed) {
        ++inv_stats.bracket_failures;
      }
      if (!inv.converged) {
        ++inv_stats.convergence_failures;
      }
      const double t_rel = std::fabs(inv.T - T) / std::max(std::fabs(T), 1.0e-30);
      inv_stats.all.update(t_rel, rho, T);
      if (is_low_T) {
        inv_stats.low.update(t_rel, rho, T);
      }
    }
  }

  core::log_info("HelmholtzBSpline[" + std::string(label) +
                 "] fit options: degree=" + std::to_string(options.degree) +
                 ", lambda_second_diff=" + std::to_string(options.lambda_second_diff) +
                 ", lambda_ref_cv=" + std::to_string(options.lambda_ref_cv) +
                 ", lambda_ref_drho=" + std::to_string(options.lambda_ref_drho) +
                 ", knot_stride=(" + std::to_string(options.knot_stride_rho) + ", " +
                 std::to_string(options.knot_stride_T) + ")" +
                 ", qp_iterations=" + std::to_string(fit.total_qp_iterations) +
                 ", active_set_size=" + std::to_string(fit.final_active_set_size) +
                 ", gauge_weight=" + std::to_string(options.gauge_weight) +
                 ", relative_floor_fraction=" +
                 std::to_string(options.relative_floor_fraction) +
                 ", phi_scale=" + std::to_string(fit.phi_scale));
  core::log_info("HelmholtzBSpline[" + std::string(label) +
                 "] node relerr: P=" + format_error_stat(p_all) +
                 ", e=" + format_error_stat(e_all));
  core::log_info("HelmholtzBSpline[" + std::string(label) +
                 "] low-T (0.1-10 eV) node relerr: P=" + format_error_stat(p_low) +
                 ", e=" + format_error_stat(e_low));
  core::log_info("HelmholtzBSpline[" + std::string(label) +
                 "] positivity: cv{" + format_positivity_stat(cv_stats) + "}, cs2{" +
                 format_positivity_stat(cs2_stats) + "}");
  core::log_info("HelmholtzBSpline[" + std::string(label) +
                 "] T_from_e node inversion: relerr(all)=" +
                 format_error_stat(inv_stats.all) + ", relerr(low-T)=" +
                 format_error_stat(inv_stats.low) + ", bracket_failures=" +
                 std::to_string(inv_stats.bracket_failures) + ", convergence_failures=" +
                 std::to_string(inv_stats.convergence_failures));

  if (!table.empty()) {
    const std::size_t i = table.n_rho() / 2;
    const std::size_t j = table.n_T() / 2;
    const std::size_t idx = table.flat_index(i, j);
    const double rho = table.rho_grid[i];
    const double T = table.T_grid_eV[j];
    const HelmholtzBSplineEval phi = fit.eval(table.log_rho_grid[i], table.log_T_grid[j]);
    const double phi_x_target = table.P_table[idx] / std::max(rho * T, 1.0e-30);
    const double phi_y_target = -table.e_table[idx] / std::max(T, 1.0e-30);
    const double cv_target =
        (table.cv_table.size() == table.n_rho() * table.n_T()) ? table.cv_table[idx] : 0.0;
    core::log_info("HelmholtzBSpline[" + std::string(label) +
                   "] center node: rho=" + std::to_string(rho) + ", T=" + std::to_string(T) +
                   ", phi_x(eval/target)=(" + std::to_string(phi.phi_x) + ", " +
                   std::to_string(phi_x_target) + "), phi_y(eval/target)=(" +
                   std::to_string(phi.phi_y) + ", " + std::to_string(phi_y_target) +
                   "), P(eval/data)=(" + std::to_string(rho * T * phi.phi_x) + ", " +
                   std::to_string(table.P_table[idx]) + "), e(eval/data)=(" +
                   std::to_string(-T * phi.phi_y) + ", " + std::to_string(table.e_table[idx]) +
                   "), cv(eval/target)=(" + std::to_string(-(phi.phi_y + phi.phi_yy)) + ", " +
                   std::to_string(cv_target) + ")");

    const BasisEval1D bx = evaluate_basis_1d(fit.knot_x, fit.degree, fit.n_basis_x,
                                             table.log_rho_grid[i]);
    const BasisEval1D by = evaluate_basis_1d(fit.knot_y, fit.degree, fit.n_basis_y,
                                             table.log_T_grid[j]);
    double p_row_max = 0.0;
    double e_row_max = 0.0;
    double p_row_sumabs = 0.0;
    double e_row_sumabs = 0.0;
    for (std::size_t ax = 0; ax < bx.indices.size(); ++ax) {
      for (std::size_t ay = 0; ay < by.indices.size(); ++ay) {
        const double p_val = bx.deriv1[ax] * by.value[ay];
        const double e_val = bx.value[ax] * by.deriv1[ay];
        p_row_max = std::max(p_row_max, std::fabs(p_val));
        e_row_max = std::max(e_row_max, std::fabs(e_val));
        p_row_sumabs += std::fabs(p_val);
        e_row_sumabs += std::fabs(e_val);
      }
    }
    const double wP = 1.0 / std::max(std::fabs(table.P_table[idx]), p_floor);
    const double wE = 1.0 / std::max(std::fabs(table.e_table[idx]), e_floor);
    core::log_info("HelmholtzBSpline[" + std::string(label) +
                   "] center row support: nnz=" +
                   std::to_string(bx.indices.size() * by.indices.size()) +
                   ", P_row{|a|max,sumabs}=(" +
                   std::to_string(wP * fit.phi_scale * p_row_max) + ", " +
                   std::to_string(wP * fit.phi_scale * p_row_sumabs) +
                   "), E_row{|a|max,sumabs}=(" +
                   std::to_string(wE * fit.phi_scale * e_row_max) + ", " +
                   std::to_string(wE * fit.phi_scale * e_row_sumabs) + ")");
  }
}

void log_matrix_report(const MatrixStats& stats, const std::string_view label) {
  core::log_info("HelmholtzBSpline[" + std::string(label) +
                 "] matrix/coeff stats: " + format_matrix_stat(stats));
}

[[nodiscard]] EOSTable build_ideal_gas_table(const std::vector<double>& rho_grid,
                                             const std::vector<double>& T_grid_eV) {
  EOSTable table;
  table.rho_grid = rho_grid;
  table.T_grid_eV = T_grid_eV;
  table.log_rho_grid.resize(rho_grid.size());
  table.log_T_grid.resize(T_grid_eV.size());
  for (std::size_t i = 0; i < rho_grid.size(); ++i) {
    table.log_rho_grid[i] = std::log(rho_grid[i]);
  }
  for (std::size_t j = 0; j < T_grid_eV.size(); ++j) {
    table.log_T_grid[j] = std::log(T_grid_eV[j]);
  }
  table.P_table.resize(rho_grid.size() * T_grid_eV.size(), 0.0);
  table.e_table.resize(table.P_table.size(), 0.0);
  table.cv_table.resize(table.P_table.size(), 1.5);
  for (std::size_t j = 0; j < T_grid_eV.size(); ++j) {
    for (std::size_t i = 0; i < rho_grid.size(); ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = rho_grid[i];
      const double T = T_grid_eV[j];
      table.P_table[idx] = rho * T;
      table.e_table[idx] = 1.5 * T;
    }
  }
  return table;
}

[[nodiscard]] std::vector<double> build_cv_targets(const EOSTable& table) {
  std::vector<double> cv(table.n_rho() * table.n_T(), 0.0);
  if (table.cv_table.size() == cv.size()) {
    return table.cv_table;
  }
  for (std::size_t i = 0; i < table.n_rho(); ++i) {
    for (std::size_t j = 0; j < table.n_T(); ++j) {
      const std::size_t idx = table.flat_index(i, j);
      if (table.n_T() == 1) {
        cv[idx] = 0.0;
      } else if (j == 0) {
        const std::size_t idx1 = table.flat_index(i, j + 1);
        cv[idx] = (table.e_table[idx1] - table.e_table[idx]) /
                  std::max(table.T_grid_eV[j + 1] - table.T_grid_eV[j], 1.0e-30);
      } else if (j + 1 == table.n_T()) {
        const std::size_t idx0 = table.flat_index(i, j - 1);
        cv[idx] = (table.e_table[idx] - table.e_table[idx0]) /
                  std::max(table.T_grid_eV[j] - table.T_grid_eV[j - 1], 1.0e-30);
      } else {
        const std::size_t idx0 = table.flat_index(i, j - 1);
        const std::size_t idx1 = table.flat_index(i, j + 1);
        cv[idx] = (table.e_table[idx1] - table.e_table[idx0]) /
                  std::max(table.T_grid_eV[j + 1] - table.T_grid_eV[j - 1], 1.0e-30);
      }
    }
  }
  return cv;
}

[[nodiscard]] bool run_self_test_case(const EOSTable& table,
                                      const HelmholtzBSplineFitOptions& options,
                                      const std::string_view label,
                                      const double cv_tol) {
  HelmholtzBSpline1T fit;
  fit.build_from_table(table, options, label);

  bool pass = true;
  for (std::size_t j = 0; j < table.n_T(); ++j) {
    for (std::size_t i = 0; i < table.n_rho(); ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const HelmholtzBSplineThermo thermo =
          fit.eval_thermo(table.rho_grid[i], table.T_grid_eV[j]);
      const double p_rel =
          std::fabs(thermo.pressure - table.P_table[idx]) / std::max(table.P_table[idx], 1.0e-30);
      const double e_rel =
          std::fabs(thermo.energy - table.e_table[idx]) / std::max(table.e_table[idx], 1.0e-30);
      const double t_inv =
          fit.temperature_from_energy(table.rho_grid[i], table.e_table[idx]);
      const double t_rel =
          std::fabs(t_inv - table.T_grid_eV[j]) / std::max(table.T_grid_eV[j], 1.0e-30);
      if (!(p_rel < 1.0e-6 && e_rel < 1.0e-6 && std::fabs(thermo.cv - 1.5) < cv_tol &&
            t_rel < 1.0e-6)) {
        pass = false;
      }
    }
  }
  core::log_info("HelmholtzBSpline[" + std::string(label) + "] " +
                 std::string(pass ? "PASSED" : "FAILED"));
  return pass;
}

}  // namespace

void HelmholtzBSpline1T::clear() {
  degree = 5;
  log_rho_grid.clear();
  log_T_grid.clear();
  knot_x.clear();
  knot_y.clear();
  n_basis_x = 0;
  n_basis_y = 0;
  phi_scale = 1.0;
  effective_lambda_constraint = 0.0;
  final_active_set_size = 0;
  total_qp_iterations = 0;
  coeffs.clear();
}

void HelmholtzBSpline1T::build_from_table(const EOSTable& table,
                                          const HelmholtzBSplineFitOptions& options,
                                          const std::string_view label) {
  clear();
  if (table.empty()) {
    return;
  }

  TENRYU_ASSERT(options.degree == 5,
                "HelmholtzBSpline Phase 1 currently supports quintic degree=5 only");
  TENRYU_ASSERT(table.log_rho_grid.size() == table.n_rho() &&
                    table.log_T_grid.size() == table.n_T(),
                "HelmholtzBSpline grid size mismatch");
  TENRYU_ASSERT(table.P_table.size() == table.n_rho() * table.n_T() &&
                    table.e_table.size() == table.n_rho() * table.n_T(),
                "HelmholtzBSpline table size mismatch");

  degree = options.degree;
  log_rho_grid = table.log_rho_grid;
  log_T_grid = table.log_T_grid;
  const std::vector<double> reduced_log_rho =
      reduce_grid_stride(log_rho_grid, options.knot_stride_rho);
  const std::vector<double> reduced_log_T =
      reduce_grid_stride(log_T_grid, options.knot_stride_T);
  knot_x = build_open_clamped_knot_vector(reduced_log_rho, degree);
  knot_y = build_open_clamped_knot_vector(reduced_log_T, degree);
  n_basis_x = reduced_log_rho.size() + static_cast<std::size_t>(degree) - 1;
  n_basis_y = reduced_log_T.size() + static_cast<std::size_t>(degree) - 1;
  const std::size_t n_coeff = n_basis_x * n_basis_y;
  coeffs.assign(n_coeff, 0.0);

  std::vector<BasisEval1D> bx(log_rho_grid.size());
  std::vector<BasisEval1D> by(log_T_grid.size());
  for (std::size_t i = 0; i < log_rho_grid.size(); ++i) {
    bx[i] = evaluate_basis_1d(knot_x, degree, n_basis_x, log_rho_grid[i]);
  }
  for (std::size_t j = 0; j < log_T_grid.size(); ++j) {
    by[j] = evaluate_basis_1d(knot_y, degree, n_basis_y, log_T_grid[j]);
  }

  HelmholtzSpline1T ref_spline;
  ref_spline.build_from_table(table, std::string(label) + "_reference");

  std::vector<double> phi_x_targets(table.n_rho() * table.n_T(), 0.0);
  std::vector<double> phi_y_targets(table.n_rho() * table.n_T(), 0.0);
  std::vector<double> cv_ref(table.n_rho() * table.n_T(), 0.0);
  std::vector<double> drho_ref(table.n_rho() * table.n_T(), 0.0);
  std::vector<double> p_abs_samples;
  std::vector<double> e_abs_samples;
  std::vector<double> cv_abs_samples;
  std::vector<double> drho_abs_samples;
  p_abs_samples.reserve(table.n_rho() * table.n_T());
  e_abs_samples.reserve(table.n_rho() * table.n_T());
  cv_abs_samples.reserve(table.n_rho() * table.n_T());
  drho_abs_samples.reserve(table.n_rho() * table.n_T());
  double max_abs_phi_x = 0.0;
  double max_abs_phi_y = 0.0;
  double max_abs_cv = 0.0;
  double max_abs_drho = 0.0;
  for (std::size_t j = 0; j < table.n_T(); ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < table.n_rho(); ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = table.rho_grid[i];
      phi_x_targets[idx] = table.P_table[idx] / std::max(rho * T, 1.0e-30);
      phi_y_targets[idx] = -table.e_table[idx] / std::max(T, 1.0e-30);
      cv_ref[idx] = ref_spline.e_dy_nodes[idx] / std::max(T, 1.0e-30);
      drho_ref[idx] = ref_spline.P_dx_nodes[idx] / std::max(rho, 1.0e-30);
      p_abs_samples.push_back(table.P_table[idx]);
      e_abs_samples.push_back(table.e_table[idx]);
      cv_abs_samples.push_back(cv_ref[idx]);
      drho_abs_samples.push_back(drho_ref[idx]);
      max_abs_phi_x = std::max(max_abs_phi_x, std::fabs(phi_x_targets[idx]));
      max_abs_phi_y = std::max(max_abs_phi_y, std::fabs(phi_y_targets[idx]));
      max_abs_cv = std::max(max_abs_cv, std::fabs(cv_ref[idx]));
      max_abs_drho = std::max(max_abs_drho, std::fabs(drho_ref[idx]));
    }
  }
  phi_scale = std::max({1.0, max_abs_phi_x, max_abs_phi_y});
  const double p_floor =
      std::max(options.relative_floor_fraction * median_abs(std::move(p_abs_samples)), 1.0e-30);
  const double e_floor =
      std::max(options.relative_floor_fraction * median_abs(std::move(e_abs_samples)), 1.0e-30);
  const double cv_floor =
      std::max(options.relative_floor_fraction * median_abs(std::move(cv_abs_samples)), 1.0e-30);
  const double drho_floor =
      std::max(options.relative_floor_fraction * median_abs(std::move(drho_abs_samples)),
               1.0e-30);

  std::vector<double> ata_base(n_coeff * n_coeff, 0.0);
  std::vector<double> rhs_base(n_coeff, 0.0);
  std::vector<std::size_t> row_indices;
  std::vector<double> row_values;
  row_indices.reserve(static_cast<std::size_t>((degree + 1) * (degree + 1)));
  row_values.reserve(static_cast<std::size_t>((degree + 1) * (degree + 1)));

  for (std::size_t j = 0; j < table.n_T(); ++j) {
    const double T = table.T_grid_eV[j];
    for (std::size_t i = 0; i < table.n_rho(); ++i) {
      const std::size_t idx = table.flat_index(i, j);
      const double rho = table.rho_grid[i];
      const double wP = 1.0 / std::max(std::fabs(table.P_table[idx]), p_floor);
      const double wE = 1.0 / std::max(std::fabs(table.e_table[idx]), e_floor);

      row_indices.clear();
      row_values.clear();
      for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
        for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
          const std::size_t cidx =
              coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                          static_cast<std::size_t>(by[j].indices[ay]),
                          n_basis_x);
          row_indices.push_back(cidx);
          row_values.push_back(wP * phi_scale * bx[i].deriv1[ax] * by[j].value[ay]);
        }
      }
      accumulate_sparse_row(ata_base, rhs_base, n_coeff, row_indices, row_values,
                            wP * phi_x_targets[idx]);

      row_indices.clear();
      row_values.clear();
      for (std::size_t ax = 0; ax < bx[i].indices.size(); ++ax) {
        for (std::size_t ay = 0; ay < by[j].indices.size(); ++ay) {
          const std::size_t cidx =
              coeff_index(static_cast<std::size_t>(bx[i].indices[ax]),
                          static_cast<std::size_t>(by[j].indices[ay]),
                          n_basis_x);
          row_indices.push_back(cidx);
          row_values.push_back(wE * phi_scale * bx[i].value[ax] * by[j].deriv1[ay]);
        }
      }
      accumulate_sparse_row(ata_base, rhs_base, n_coeff, row_indices, row_values,
                            wE * phi_y_targets[idx]);
    }
  }

  add_drho_reference_rows(ata_base, rhs_base, n_coeff, bx, by, drho_ref, table.T_grid_eV,
                          n_basis_x, phi_scale, drho_floor, options.lambda_ref_drho);
  add_cv_reference_rows(ata_base, rhs_base, n_coeff, bx, by, cv_ref, n_basis_x, phi_scale,
                        cv_floor, options.lambda_ref_cv);
  add_second_difference_regularization(ata_base, rhs_base, n_basis_x, n_basis_y,
                                       options.lambda_second_diff);
  add_gauge_condition(ata_base,
                      n_coeff,
                      coeff_index(n_basis_x / 2, n_basis_y / 2, n_basis_x),
                      options.gauge_weight);

  MatrixStats base_stats{};
  for (std::size_t i = 0; i < n_coeff; ++i) {
    const double d = std::fabs(ata_base[i * n_coeff + i]);
    base_stats.diag_min = std::min(base_stats.diag_min, d);
    base_stats.diag_max = std::max(base_stats.diag_max, d);
  }

  std::vector<double> scale(n_coeff, 1.0);
  for (std::size_t i = 0; i < n_coeff; ++i) {
    scale[i] = 1.0 / std::sqrt(std::max(std::fabs(ata_base[i * n_coeff + i]), 1.0e-300));
  }

  std::vector<double> ata_scaled_base = ata_base;
  std::vector<double> rhs_scaled_base = rhs_base;
  for (std::size_t i = 0; i < n_coeff; ++i) {
    rhs_scaled_base[i] *= scale[i];
    double* row = ata_scaled_base.data() + i * n_coeff;
    for (std::size_t j = 0; j < n_coeff; ++j) {
      row[j] *= scale[i] * scale[j];
    }
  }
  if (options.diagonal_jitter > 0.0) {
    double diag_mean = 0.0;
    for (std::size_t i = 0; i < n_coeff; ++i) {
      diag_mean += std::fabs(ata_scaled_base[i * n_coeff + i]);
    }
    diag_mean /= std::max<std::size_t>(n_coeff, 1);
    const double jitter = options.diagonal_jitter * std::max(diag_mean, 1.0);
    for (std::size_t i = 0; i < n_coeff; ++i) {
      ata_scaled_base[i * n_coeff + i] += jitter;
    }
  }

  std::vector<double> h_factor = ata_scaled_base;
  TENRYU_ASSERT(cholesky_factor_inplace(h_factor, n_coeff),
                "HelmholtzBSpline base Cholesky factorization failed");

  std::vector<double> y_unconstrained = rhs_scaled_base;
  cholesky_solve_inplace(h_factor, y_unconstrained, n_coeff);

  const double eps_cv =
      std::max(options.positivity_eps_fraction * std::max(median_abs(cv_ref), 1.0), 1.0e-30);
  const double eps_drho =
      std::max(options.positivity_eps_fraction * std::max(median_abs(drho_ref), 1.0), 1.0e-30);
  const double violation_tol = 0.01 * std::max(eps_cv, eps_drho);
  const double lambda_tol = 1.0e-10;

  auto make_attempt_from_scaled = [&](const std::vector<double>& y_scaled) -> SolveAttempt {
    SolveAttempt attempt{};
    attempt.coeffs.resize(n_coeff);
    for (std::size_t i = 0; i < n_coeff; ++i) {
      attempt.coeffs[i] = y_scaled[i] * scale[i];
      attempt.stats.coeff_min = std::min(attempt.stats.coeff_min, attempt.coeffs[i]);
      attempt.stats.coeff_max = std::max(attempt.stats.coeff_max, attempt.coeffs[i]);
      attempt.stats.coeff_mean += attempt.coeffs[i];
    }
    attempt.stats.coeff_mean /= std::max<std::size_t>(n_coeff, 1);
    attempt.stats.diag_min = base_stats.diag_min;
    attempt.stats.diag_max = base_stats.diag_max;
    attempt.stats.rel_residual =
        compute_relative_residual(ata_scaled_base, rhs_scaled_base, y_scaled, n_coeff);
    HelmholtzBSpline1T tmp = *this;
    tmp.coeffs = attempt.coeffs;
    attempt.stats.rel_data_residual = compute_data_relative_residual(tmp, table, p_floor, e_floor);
    const StabilityCounts counts = evaluate_stability_counts(tmp, table);
    attempt.cv_nonpositive = counts.cv_nonpositive;
    attempt.drho_nonpositive = counts.drho_nonpositive;
    attempt.cs2_nonpositive = counts.cs2_nonpositive;
    attempt.success = true;
    return attempt;
  };

  auto solve_with_active_constraints =
      [&](const std::vector<ConstraintSpec>& active_constraints,
          std::vector<double>& y_scaled,
          std::vector<double>& multipliers) -> bool {
    y_scaled = y_unconstrained;
    multipliers.clear();
    if (active_constraints.empty()) {
      return true;
    }

    const std::size_t m = active_constraints.size();
    std::vector<std::vector<double>> dense_rows(m, std::vector<double>(n_coeff, 0.0));
    std::vector<std::vector<double>> w_cols(m, std::vector<double>(n_coeff, 0.0));
    std::vector<double> schur(m * m, 0.0);
    std::vector<double> rhs_schur(m, 0.0);

    for (std::size_t a = 0; a < m; ++a) {
      dense_row_from_constraint(active_constraints[a], scale, n_coeff, dense_rows[a]);
      w_cols[a] = dense_rows[a];
      cholesky_solve_inplace(h_factor, w_cols[a], n_coeff);
      double gy0 = 0.0;
      for (std::size_t i = 0; i < n_coeff; ++i) {
        gy0 += dense_rows[a][i] * y_unconstrained[i];
      }
      rhs_schur[a] = active_constraints[a].rhs - gy0;
    }

    for (std::size_t a = 0; a < m; ++a) {
      for (std::size_t b = a; b < m; ++b) {
        double value = 0.0;
        for (std::size_t i = 0; i < n_coeff; ++i) {
          value += dense_rows[a][i] * w_cols[b][i];
        }
        schur[a * m + b] = value;
        schur[b * m + a] = value;
      }
    }

    multipliers = rhs_schur;
    if (!solve_spd_with_jitter(schur, multipliers, m)) {
      return false;
    }

    for (std::size_t a = 0; a < m; ++a) {
      const double lambda = multipliers[a];
      for (std::size_t i = 0; i < n_coeff; ++i) {
        y_scaled[i] += w_cols[a][i] * lambda;
      }
    }
    return true;
  };

  SolveAttempt best{};
  bool have_best = false;
  std::size_t best_active_set_size = 0;
  int best_iterations = 0;
  auto better_attempt = [](const SolveAttempt& lhs, const SolveAttempt& rhs) {
    if (lhs.cv_nonpositive != rhs.cv_nonpositive) {
      return lhs.cv_nonpositive < rhs.cv_nonpositive;
    }
    if (lhs.drho_nonpositive != rhs.drho_nonpositive) {
      return lhs.drho_nonpositive < rhs.drho_nonpositive;
    }
    if (lhs.cs2_nonpositive != rhs.cs2_nonpositive) {
      return lhs.cs2_nonpositive < rhs.cs2_nonpositive;
    }
    return lhs.stats.rel_data_residual < rhs.stats.rel_data_residual;
  };

  std::vector<ConstraintSpec> active_constraints;
  std::vector<char> active_cv(table.n_rho() * table.n_T(), 0);
  std::vector<char> active_drho(table.n_rho() * table.n_T(), 0);
  int completed_iterations = 0;
  for (int iter = 0; iter < std::max(options.max_constraint_iters, 1); ++iter) {
    ++completed_iterations;
    std::vector<double> y_scaled;
    std::vector<double> multipliers;
    TENRYU_ASSERT(solve_with_active_constraints(active_constraints, y_scaled, multipliers),
                  "HelmholtzBSpline active-set KKT solve failed");

    bool removed = false;
    if (!multipliers.empty()) {
      double min_lambda = multipliers.front();
      std::size_t min_idx = 0;
      for (std::size_t a = 1; a < multipliers.size(); ++a) {
        if (multipliers[a] < min_lambda) {
          min_lambda = multipliers[a];
          min_idx = a;
        }
      }
      if (min_lambda < -lambda_tol) {
        const ConstraintSpec removed_constraint = active_constraints[min_idx];
        if (removed_constraint.kind == ConstraintKind::Cv) {
          active_cv[removed_constraint.flat_index] = 0;
        } else {
          active_drho[removed_constraint.flat_index] = 0;
        }
        active_constraints.erase(active_constraints.begin() +
                                 static_cast<std::ptrdiff_t>(min_idx));
        core::log_info("HelmholtzBSpline[" + std::string(label) +
                       "] active-set iter=" + std::to_string(iter) +
                       ", active_set_size=" + std::to_string(active_constraints.size()) +
                       ", worst_violation=release_lambda(" + std::to_string(min_lambda) +
                       "), worst_constraint_type=" +
                       std::string(constraint_kind_name(removed_constraint.kind)));
        removed = true;
      }
    }
    if (removed) {
      continue;
    }

    const SolveAttempt attempt = make_attempt_from_scaled(y_scaled);
    TENRYU_ASSERT(attempt.success, "HelmholtzBSpline active-set attempt failed");
    if (!have_best || better_attempt(attempt, best)) {
      best = attempt;
      best_active_set_size = active_constraints.size();
      best_iterations = completed_iterations;
      have_best = true;
    }
    HelmholtzBSpline1T tmp = *this;
    tmp.coeffs = attempt.coeffs;
    double worst_violation = std::numeric_limits<double>::infinity();
    ConstraintKind worst_kind = ConstraintKind::Cv;
    std::size_t worst_flat_idx = std::numeric_limits<std::size_t>::max();
    for (std::size_t j = 0; j < table.n_T(); ++j) {
      const double T = table.T_grid_eV[j];
      for (std::size_t i = 0; i < table.n_rho(); ++i) {
        const std::size_t idx = table.flat_index(i, j);
        const HelmholtzBSplineThermo thermo = eval_thermo_host(tmp, table.rho_grid[i], T);
        const double cv_violation = thermo.cv - eps_cv;
        const double drho_violation = thermo.dP_drho_T - eps_drho;
        if (!active_cv[idx] && cv_violation < worst_violation) {
          worst_violation = cv_violation;
          worst_kind = ConstraintKind::Cv;
          worst_flat_idx = idx;
        }
        if (!active_drho[idx] && drho_violation < worst_violation) {
          worst_violation = drho_violation;
          worst_kind = ConstraintKind::Drho;
          worst_flat_idx = idx;
        }
      }
    }

    core::log_info("HelmholtzBSpline[" + std::string(label) + "] active-set iter=" +
                   std::to_string(iter) + ", active_set_size=" +
                   std::to_string(active_constraints.size()) + ", worst_violation=" +
                   std::to_string(worst_violation) + ", worst_constraint_type=" +
                   std::string(constraint_kind_name(worst_kind)) + ", cv_nonpositive=" +
                   std::to_string(attempt.cv_nonpositive) + ", drho_nonpositive=" +
                   std::to_string(attempt.drho_nonpositive) + ", cs2_nonpositive=" +
                   std::to_string(attempt.cs2_nonpositive) + ", rel_data_residual=" +
                   std::to_string(attempt.stats.rel_data_residual));

    if (worst_flat_idx == std::numeric_limits<std::size_t>::max() ||
        worst_violation >= -violation_tol) {
      best = attempt;
      best_active_set_size = active_constraints.size();
      best_iterations = completed_iterations;
      have_best = true;
      break;
    }

    if (worst_kind == ConstraintKind::Cv) {
      active_cv[worst_flat_idx] = 1;
      active_constraints.push_back(
          build_constraint_spec(ConstraintKind::Cv, worst_flat_idx, bx, by, table.T_grid_eV,
                                n_basis_x, phi_scale, eps_cv));
    } else {
      active_drho[worst_flat_idx] = 1;
      active_constraints.push_back(
          build_constraint_spec(ConstraintKind::Drho, worst_flat_idx, bx, by, table.T_grid_eV,
                                n_basis_x, phi_scale, eps_drho));
    }
  }
  TENRYU_ASSERT(have_best, "HelmholtzBSpline failed to produce any candidate fit");
  coeffs = best.coeffs;
  effective_lambda_constraint = 0.0;
  final_active_set_size = best_active_set_size;
  total_qp_iterations = best_iterations;
  core::log_info("HelmholtzBSpline[" + std::string(label) + "] active-set final: iterations=" +
                 std::to_string(total_qp_iterations) + ", active_set_size=" +
                 std::to_string(final_active_set_size) + ", cv_nonpositive=" +
                 std::to_string(best.cv_nonpositive) + ", drho_nonpositive=" +
                 std::to_string(best.drho_nonpositive) + ", cs2_nonpositive=" +
                 std::to_string(best.cs2_nonpositive) + ", rel_data_residual=" +
                 std::to_string(best.stats.rel_data_residual));
  log_matrix_report(best.stats, label);
  log_fit_report(*this, table, options, label);
}

bool HelmholtzBSpline1T::self_test() {
  HelmholtzBSplineFitOptions options;
  options.lambda_second_diff = 1.0e-2;
  options.gauge_weight = 1.0;
  options.diagonal_jitter = 1.0e-14;
  const EOSTable small =
      build_ideal_gas_table({1.0, 2.0, 4.0, 8.0, 16.0}, {1.0, 2.0, 4.0, 8.0, 16.0});
  bool pass = run_self_test_case(small, options, "self_test_small", 1.0e-2);

  std::vector<double> rho_large(81);
  std::vector<double> T_large(50);
  const double log_rho_min = std::log(1.0e-4);
  const double log_rho_max = std::log(1.0e2);
  const double log_T_min = std::log(1.0e-1);
  const double log_T_max = std::log(1.0e3);
  for (std::size_t i = 0; i < rho_large.size(); ++i) {
    const double a = static_cast<double>(i) / static_cast<double>(rho_large.size() - 1);
    rho_large[i] = std::exp((1.0 - a) * log_rho_min + a * log_rho_max);
  }
  for (std::size_t j = 0; j < T_large.size(); ++j) {
    const double a = static_cast<double>(j) / static_cast<double>(T_large.size() - 1);
    T_large[j] = std::exp((1.0 - a) * log_T_min + a * log_T_max);
  }
  const EOSTable large = build_ideal_gas_table(rho_large, T_large);
  pass = run_self_test_case(large, options, "self_test_81x50", 5.0e-2) && pass;

  core::log_info(std::string("HelmholtzBSpline[self_test] ") + (pass ? "PASSED" : "FAILED"));
  return pass;
}

HelmholtzBSplineEval HelmholtzBSpline1T::eval(const double log_rho_raw,
                                              const double log_T_raw) const {
  HelmholtzBSplineEval out{};
  if (empty()) {
    return out;
  }

  const BasisEval1D bx = evaluate_basis_1d(
      knot_x, degree, n_basis_x, std::clamp(log_rho_raw, log_rho_grid.front(), log_rho_grid.back()));
  const BasisEval1D by = evaluate_basis_1d(
      knot_y, degree, n_basis_y, std::clamp(log_T_raw, log_T_grid.front(), log_T_grid.back()));

  for (std::size_t ax = 0; ax < bx.indices.size(); ++ax) {
    for (std::size_t ay = 0; ay < by.indices.size(); ++ay) {
      const double c = coeffs[coeff_index(static_cast<std::size_t>(bx.indices[ax]),
                                          static_cast<std::size_t>(by.indices[ay]),
                                          n_basis_x)];
      out.phi += c * bx.value[ax] * by.value[ay];
      out.phi_x += c * bx.deriv1[ax] * by.value[ay];
      out.phi_y += c * bx.value[ax] * by.deriv1[ay];
      out.phi_xx += c * bx.deriv2[ax] * by.value[ay];
      out.phi_yy += c * bx.value[ax] * by.deriv2[ay];
      out.phi_xy += c * bx.deriv1[ax] * by.deriv1[ay];
    }
  }
  out.phi *= phi_scale;
  out.phi_x *= phi_scale;
  out.phi_y *= phi_scale;
  out.phi_xx *= phi_scale;
  out.phi_yy *= phi_scale;
  out.phi_xy *= phi_scale;
  return out;
}

HelmholtzBSplineThermo HelmholtzBSpline1T::eval_thermo(const double rho,
                                                       const double T_eV) const {
  return eval_thermo_host(*this, rho, T_eV);
}

double HelmholtzBSpline1T::temperature_from_energy(const double rho,
                                                   const double e_target) const {
  return temperature_from_energy_host(*this, rho, e_target).T;
}

}  // namespace tenryu::materials
