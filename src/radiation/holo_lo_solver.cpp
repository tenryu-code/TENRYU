#include "radiation/holo_lo_solver.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "materials/eos_table.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kFourPi = 12.56637061435917295385;
constexpr double kSigmaFloor = 1.0e-30;
constexpr double kEnergyFloor = 1.0e-30;
constexpr double kConservationRelTol = 1.0e-8;
constexpr double kQDNegativeGroupRelTol = 1.0e-2;
constexpr double kQDNegativeCellRelTol = 5.0e-3;

[[nodiscard]] bool finite(const double value) {
  return std::isfinite(value);
}

[[nodiscard]] double finite_or_zero(const double value) {
  return finite(value) ? value : 0.0;
}

[[nodiscard]] double nonnegative_finite(const double value) {
  return finite(value) ? std::max(value, 0.0) : 0.0;
}

[[nodiscard]] std::size_t idx(const int cell, const int group, const int n_groups) {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
         static_cast<std::size_t>(group);
}

[[nodiscard]] bool lo_cell_active(const HoloLOInputs& in, const int cell) {
  return in.cell_active == nullptr || in.cell_active[cell] != 0U;
}

[[nodiscard]] bool lo_material_commit_cell(const HoloLOInputs& in, const int cell) {
  return in.lo_coupled != nullptr && in.lo_coupled[cell] != 0U;
}

[[nodiscard]] double face_current_energy(const HoloLOInputs& in,
                                         const int face,
                                         const int group) {
  if (in.face_current == nullptr || face < 0 || face > in.n_cells ||
      group < 0 || group >= in.n_groups) {
    return 0.0;
  }
  return finite_or_zero(in.face_current[idx(face, group, in.n_groups)]);
}

[[nodiscard]] double face_current_dt(const HoloLOInputs& in) {
  const double current_dt = finite_or_zero(in.face_current_dt);
  return (current_dt > 0.0) ? current_dt : in.dt;
}

[[nodiscard]] double cell_radiation_sum(const std::vector<double>& E,
                                        const int cell,
                                        const int n_groups) {
  if (cell < 0 || n_groups <= 0) {
    return 0.0;
  }
  double sum = 0.0;
  for (int g = 0; g < n_groups; ++g) {
    sum += nonnegative_finite(E[idx(cell, g, n_groups)]);
  }
  return sum;
}

void add_corrected_source_diagnostic(double& rad_dep,
                                     double& rad_emit,
                                     const long double gross_dep,
                                     const long double gross_emit,
                                     const double net_source) {
  if (!finite(net_source)) {
    return;
  }

  const auto use_net_split = [&]() {
    return std::pair<double, double>{(net_source > 0.0) ? net_source : 0.0,
                                    (net_source < 0.0) ? -net_source : 0.0};
  };

  double dep_add = static_cast<double>(std::max(gross_dep, 0.0L));
  double emit_add = static_cast<double>(std::max(gross_emit, 0.0L));
  if (!finite(dep_add) || !finite(emit_add)) {
    const auto split = use_net_split();
    dep_add = split.first;
    emit_add = split.second;
  } else {
    const double correction = net_source - (dep_add - emit_add);
    if (finite(correction)) {
      if (correction >= 0.0) {
        dep_add += correction;
      } else {
        emit_add -= correction;
      }
    }
    const double corrected_net = dep_add - emit_add;
    const double net_tol =
        1.0e-12 * std::max(std::abs(net_source), kEnergyFloor);
    if (!finite(corrected_net) || std::abs(corrected_net - net_source) > net_tol) {
      const auto split = use_net_split();
      dep_add = split.first;
      emit_add = split.second;
    }
  }

  rad_dep += dep_add;
  rad_emit += emit_add;
}

void record_failure(HoloLOResult& out,
                    const HoloLOFailureStage stage,
                    const int cell,
                    const int group,
                    const HoloLOInputs& in,
                    const std::vector<double>& E,
                    const std::vector<double>& Te,
                    const double residual) {
  if (out.failure_stage != HoloLOFailureStage::None) {
    return;
  }
  out.failure_stage = stage;
  out.failure_cell = cell;
  out.failure_group = group;
  if (cell >= 0 && cell < in.n_cells) {
    out.failure_E_sum = cell_radiation_sum(E, cell, in.n_groups);
    out.failure_T = finite_or_zero(Te[static_cast<std::size_t>(cell)]);
  }
  out.failure_residual = residual;
}

[[nodiscard]] double harmonic_mean(const double a, const double b) {
  const double denom = a + b;
  return (denom > 0.0) ? (2.0 * a * b / denom) : 0.0;
}

[[nodiscard]] double cell_center(const HoloLOInputs& in, const int c) {
  return 0.5 * (finite_or_zero(in.node_r[c]) + finite_or_zero(in.node_r[c + 1]));
}

[[nodiscard]] double face_area(const HoloLOInputs& in, const int face) {
  const double r = std::max(finite_or_zero(in.node_r[face]), 0.0);
  return kFourPi * r * r;
}

[[nodiscard]] double face_coupling(const HoloLOInputs& in, const int face, const int group) {
  const int left = face - 1;
  const int right = face;
  const double dr = cell_center(in, right) - cell_center(in, left);
  if (!(dr > 0.0)) {
    return 0.0;
  }
  const double sigma_left =
      std::max(nonnegative_finite(in.sigma_R[idx(left, group, in.n_groups)]), kSigmaFloor);
  const double sigma_right =
      std::max(nonnegative_finite(in.sigma_R[idx(right, group, in.n_groups)]), kSigmaFloor);
  const double D_left = core::constants::c_light / (3.0 * sigma_left);
  const double D_right = core::constants::c_light / (3.0 * sigma_right);
  const double D_face = harmonic_mean(D_left, D_right);
  return face_area(in, face) * D_face / dr;
}

struct QDFaceCoefficients {
  double area = 0.0;
  double dF_dE_left = 0.0;
  double dF_dE_right = 0.0;
  double F_const = 0.0;
  bool valid = false;
};

[[nodiscard]] double qd_chi(const HoloLOInputs& in, const int cell, const int group) {
  if (in.chi == nullptr) {
    return 1.0 / 3.0;
  }
  // The origin-touching cell is isotropic by spherical symmetry.
  if (cell == 0) {
    return 1.0 / 3.0;
  }
  const double value = finite_or_zero(in.chi[idx(cell, group, in.n_groups)]);
  if (!(value > 0.0)) {
    return 1.0 / 3.0;
  }
  return std::clamp(value, 1.0 / 3.0, 1.0);
}

[[nodiscard]] double qd_old_flux(const HoloLOInputs& in, const int face, const int group) {
  if (in.F_lo == nullptr || face < 0 || face > in.n_cells ||
      group < 0 || group >= in.n_groups) {
    return 0.0;
  }
  return finite_or_zero(in.F_lo[idx(face, group, in.n_groups)]);
}

[[nodiscard]] double qd_geometry_correction(const HoloLOInputs& in,
                                            const int left,
                                            const int face,
                                            const double f_avg,
                                            const double dr) {
  const double r_face = std::max(finite_or_zero(in.node_r[face]), 0.0);
  const double r_min_reg = dr;
  const double r_reg = std::max(r_face, r_min_reg);
  return (3.0 * f_avg - 1.0) / (2.0 * r_reg);
}

[[nodiscard]] QDFaceCoefficients qd_face_coefficients(const HoloLOInputs& in,
                                                      const int face,
                                                      const int group) {
  QDFaceCoefficients out{};
  const int left = face - 1;
  const int right = face;
  const double dr = cell_center(in, right) - cell_center(in, left);
  if (!(dr > 0.0)) {
    return out;
  }

  const double sigma_left =
      std::max(nonnegative_finite(in.sigma_R[idx(left, group, in.n_groups)]), kSigmaFloor);
  const double sigma_right =
      std::max(nonnegative_finite(in.sigma_R[idx(right, group, in.n_groups)]), kSigmaFloor);
  const double sigma_face = 0.5 * (sigma_left + sigma_right);
  const double beta = 1.0 / (core::constants::c_light * in.dt);
  const double denom = beta + sigma_face;
  if (!(denom > 0.0) || !finite(denom)) {
    return out;
  }

  const double f_left = qd_chi(in, left, group);
  const double f_right = qd_chi(in, right, group);
  const double f_avg = 0.5 * (f_left + f_right);
  const double geom = qd_geometry_correction(in, left, face, f_avg, dr);
  const double gamma = 1.0 / denom;
  const double transport = core::constants::c_light * gamma;
  out.area = face_area(in, face);
  out.dF_dE_left = transport * (f_left / dr - geom);
  out.dF_dE_right = transport * (-f_right / dr - geom);
  out.F_const = gamma * beta * qd_old_flux(in, face, group);
  out.valid = finite(out.area) && out.area >= 0.0 &&
              finite(out.dF_dE_left) && finite(out.dF_dE_right) &&
              finite(out.F_const);
  return out;
}

[[nodiscard]] double qd_face_flux(const QDFaceCoefficients& coeff,
                                  const double E_left,
                                  const double E_right) {
  return coeff.F_const + coeff.dF_dE_left * E_left +
         coeff.dF_dE_right * E_right;
}

[[nodiscard]] double qd_negative_energy_tolerance(
    const HoloLOInputs& in,
    const std::vector<double>& E_initial,
    const std::vector<double>& B,
    const int cell,
    const int group) {
  const std::size_t key = idx(cell, group, in.n_groups);
  const double group_scale =
      std::max(nonnegative_finite(E_initial[key]),
               nonnegative_finite(B[static_cast<std::size_t>(cell)]));
  const double cell_scale = cell_radiation_sum(E_initial, cell, in.n_groups);
  return std::max(kQDNegativeGroupRelTol * group_scale,
                  kQDNegativeCellRelTol * cell_scale);
}

[[nodiscard]] bool solve_tridiagonal(std::vector<double>& lower,
                                     std::vector<double>& diag,
                                     std::vector<double>& upper,
                                     std::vector<double>& rhs,
                                     std::vector<double>& sol) {
  const int n = static_cast<int>(diag.size());
  sol.assign(static_cast<std::size_t>(std::max(n, 0)), 0.0);
  if (n <= 0) {
    return true;
  }

  for (int i = 1; i < n; ++i) {
    const double pivot = diag[static_cast<std::size_t>(i - 1)];
    if (!(pivot > 0.0) || !finite(pivot)) {
      return false;
    }
    const double factor = lower[static_cast<std::size_t>(i)] / pivot;
    diag[static_cast<std::size_t>(i)] -= factor * upper[static_cast<std::size_t>(i - 1)];
    rhs[static_cast<std::size_t>(i)] -= factor * rhs[static_cast<std::size_t>(i - 1)];
  }

  if (!(diag.back() > 0.0) || !finite(diag.back())) {
    return false;
  }
  sol.back() = rhs.back() / diag.back();
  if (!finite(sol.back())) {
    return false;
  }
  for (int i = n - 2; i >= 0; --i) {
    const double pivot = diag[static_cast<std::size_t>(i)];
    if (!(pivot > 0.0) || !finite(pivot)) {
      return false;
    }
    sol[static_cast<std::size_t>(i)] =
        (rhs[static_cast<std::size_t>(i)] -
         upper[static_cast<std::size_t>(i)] * sol[static_cast<std::size_t>(i + 1)]) /
        pivot;
    if (!finite(sol[static_cast<std::size_t>(i)])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool solve_tridiagonal_general(std::vector<double>& lower,
                                             std::vector<double>& diag,
                                             std::vector<double>& upper,
                                             std::vector<double>& rhs,
                                             std::vector<double>& sol) {
  const int n = static_cast<int>(diag.size());
  sol.assign(static_cast<std::size_t>(std::max(n, 0)), 0.0);
  if (n <= 0) {
    return true;
  }

  const auto bad_pivot = [](const double pivot) {
    return !finite(pivot) || std::abs(pivot) < 1.0e-30;
  };

  for (int i = 1; i < n; ++i) {
    const double pivot = diag[static_cast<std::size_t>(i - 1)];
    if (bad_pivot(pivot)) {
      return false;
    }
    const double factor = lower[static_cast<std::size_t>(i)] / pivot;
    diag[static_cast<std::size_t>(i)] -= factor * upper[static_cast<std::size_t>(i - 1)];
    rhs[static_cast<std::size_t>(i)] -= factor * rhs[static_cast<std::size_t>(i - 1)];
  }

  if (bad_pivot(diag.back())) {
    return false;
  }
  sol.back() = rhs.back() / diag.back();
  if (!finite(sol.back())) {
    return false;
  }
  for (int i = n - 2; i >= 0; --i) {
    const double pivot = diag[static_cast<std::size_t>(i)];
    if (bad_pivot(pivot)) {
      return false;
    }
    sol[static_cast<std::size_t>(i)] =
        (rhs[static_cast<std::size_t>(i)] -
         upper[static_cast<std::size_t>(i)] * sol[static_cast<std::size_t>(i + 1)]) /
        pivot;
    if (!finite(sol[static_cast<std::size_t>(i)])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] double planck_b(const HoloLOInputs& in, const int group, const double T) {
  if (in.n_groups <= 1) {
    return 1.0;
  }
  return std::max(in.planck->interpolate_b_host(group, T), 0.0);
}

[[nodiscard]] double planck_B(const HoloLOInputs& in, const int group, const double T) {
  const double T2 = T * T;
  const double T4 = T2 * T2;
  if (!finite(T4)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return core::constants::a_eV * T4 * planck_b(in, group, T);
}

[[nodiscard]] double ideal_cv(const HoloLOInputs& in,
                              const int cell,
                              const std::vector<double>& ee,
                              const std::vector<double>& Te) {
  double cv = (in.cv_e != nullptr) ? finite_or_zero(in.cv_e[cell]) : 0.0;
  if (!(cv > 0.0)) {
    cv = finite_or_zero(in.cv_e_const);
  }
  if (!(cv > 0.0) && Te[static_cast<std::size_t>(cell)] > 0.0 &&
      ee[static_cast<std::size_t>(cell)] > 0.0) {
    cv = ee[static_cast<std::size_t>(cell)] / Te[static_cast<std::size_t>(cell)];
  }
  return cv;
}

[[nodiscard]] double cell_mass(const HoloLOInputs& in, const int cell) {
  const double m_state = (in.mass != nullptr) ? finite_or_zero(in.mass[cell]) : 0.0;
  if (m_state > 0.0) {
    return m_state;
  }
  return finite_or_zero(in.rho[cell]) * finite_or_zero(in.vol[cell]);
}

[[nodiscard]] double material_pressure(const HoloLOInputs& in,
                                       const int cell,
                                       const double T,
                                       const double e_new) {
  const double rho = std::max(finite_or_zero(in.rho[cell]), kEnergyFloor);
  if (in.material_model == HoloLOMaterialModel::TableEOS) {
    return in.electron_eos->pressure(rho, T);
  }
  return std::max(in.pressure_gamma_minus_one, 0.0) * rho * e_new;
}

[[nodiscard]] bool solve_global_group(const HoloLOInputs& in,
                                      const int group,
                                      const std::vector<double>& E_initial,
                                      const std::vector<double>& Te,
                                      std::vector<double>& E_work,
                                      std::vector<double>& rad_dep_lo,
                                      std::vector<double>& rad_emit_lo,
                                      std::vector<double>& source_matter_delta_cell,
                                      long double& global_source_matter_delta,
                                      long double& boundary_E_in,
                                      long double& boundary_E_out,
                                      long double& boundary_limited_E,
                                      HoloLOResult& out) {
  const int n = in.n_cells;
  std::vector<double> B(static_cast<std::size_t>(n), 0.0);
  std::vector<double> alpha(static_cast<std::size_t>(n), 0.0);
  std::vector<double> face_couplings(static_cast<std::size_t>(n + 1), 0.0);
  std::vector<int> dirichlet_neighbor_face(static_cast<std::size_t>(n + 1), -1);
  std::vector<double> boundary_sink_cell(static_cast<std::size_t>(n), 0.0);
  std::vector<double> boundary_current_cell(static_cast<std::size_t>(n), 0.0);
  std::vector<double> boundary_available(static_cast<std::size_t>(n), 0.0);
  std::vector<double> E_group(static_cast<std::size_t>(n), 0.0);

  const double T_floor = std::max(finite_or_zero(in.temperature_floor_eV), 1.0e-12);
  for (int c = 0; c < n; ++c) {
    const std::size_t key = idx(c, group, in.n_groups);
    const double V = finite_or_zero(in.vol[c]);
    const double E0 = E_initial[key];
    if (!(V > 0.0) || !finite(E0) || E0 < 0.0) {
      record_failure(out, HoloLOFailureStage::SpatialSolve, c, group, in, E_work, Te, E0);
      return false;
    }
    E_group[static_cast<std::size_t>(c)] = E0;
    boundary_available[static_cast<std::size_t>(c)] = V * E0;
  }

  const double current_dt = face_current_dt(in);
  const double current_to_step_energy =
      (current_dt > 0.0) ? (in.dt / current_dt) : 1.0;
  for (int first = 0; first < n;) {
    while (first < n && !lo_cell_active(in, first)) {
      ++first;
    }
    if (first >= n) {
      break;
    }
    int last = first;
    while (last + 1 < n && lo_cell_active(in, last + 1)) {
      ++last;
    }

    const int patch_n = last - first + 1;
    std::vector<double> lower(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> diag(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> upper(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> boundary_positive(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> boundary_out_request(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> sol;

    for (int local = 0; local < patch_n; ++local) {
      const int c = first + local;
      const std::size_t local_us = static_cast<std::size_t>(local);
      const std::size_t c_us = static_cast<std::size_t>(c);
      const std::size_t key = idx(c, group, in.n_groups);
      const double V = finite_or_zero(in.vol[c]);
      const double E0 = E_initial[key];
      const double time_coeff = V / in.dt;
      const double sigma = nonnegative_finite(in.sigma_P[key]);
      alpha[c_us] = core::constants::c_light * sigma * in.dt;
      B[c_us] = planck_B(in,
                         group,
                         std::max(finite_or_zero(Te[c_us]), T_floor));
      if (!finite(alpha[c_us]) || !finite(B[c_us])) {
        record_failure(out,
                       HoloLOFailureStage::SourceSolve,
                       c,
                       group,
                       in,
                       E_work,
                       Te,
                       B[c_us]);
        return false;
      }

      diag[local_us] = time_coeff * (1.0 + alpha[c_us]);
      rhs[local_us] = time_coeff * (E0 + alpha[c_us] * B[c_us]);
      if (in.consistency_source != nullptr) {
        rhs[local_us] +=
            in.consistency_alpha * finite_or_zero(in.consistency_source[key]);
      }
    }

    for (int face = first + 1; face <= last; ++face) {
      const double coupling = face_coupling(in, face, group);
      if (!finite(coupling) || coupling < 0.0) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       face - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       coupling);
        return false;
      }
      const int left_local = face - 1 - first;
      const int right_local = face - first;
      face_couplings[static_cast<std::size_t>(face)] = coupling;
      diag[static_cast<std::size_t>(left_local)] += coupling;
      upper[static_cast<std::size_t>(left_local)] -= coupling;
      diag[static_cast<std::size_t>(right_local)] += coupling;
      lower[static_cast<std::size_t>(right_local)] -= coupling;
    }

    const auto add_boundary_current = [&](const int local, const double signed_energy) {
      if (!finite(signed_energy) || signed_energy == 0.0) {
        return;
      }
      const std::size_t local_us = static_cast<std::size_t>(local);
      if (signed_energy > 0.0) {
        boundary_positive[local_us] += signed_energy;
      } else {
        boundary_out_request[local_us] += -signed_energy;
      }
    };

    if (in.face_current != nullptr) {
      if (first > 0) {
        add_boundary_current(0, face_current_energy(in, first, group));
      }
      if (last + 1 < n) {
        add_boundary_current(patch_n - 1, -face_current_energy(in, last + 1, group));
      }
    } else {
      const auto add_dirichlet_boundary = [&](const int local, const int face,
                                              const int neighbor_cell) {
        const std::size_t local_us = static_cast<std::size_t>(local);
        const int c = first + local;
        const double coupling = face_coupling(in, face, group);
        const double E_neighbor =
            nonnegative_finite(E_initial[idx(neighbor_cell, group, in.n_groups)]);
        if (!finite(coupling) || coupling < 0.0 || !finite(E_neighbor)) {
          record_failure(out,
                         HoloLOFailureStage::SpatialSolve,
                         c,
                         group,
                         in,
                         E_work,
                         Te,
                         coupling);
          return false;
        }
        diag[local_us] += coupling;
        rhs[local_us] += coupling * E_neighbor;
        face_couplings[static_cast<std::size_t>(face)] = coupling;
        dirichlet_neighbor_face[static_cast<std::size_t>(face)] = neighbor_cell;
        return true;
      };

      if (first > 0 && !lo_cell_active(in, first - 1)) {
        if (!add_dirichlet_boundary(0, first, first - 1)) {
          return false;
        }
      }
      if (last + 1 < n && !lo_cell_active(in, last + 1)) {
        if (!add_dirichlet_boundary(patch_n - 1, last + 1, last + 1)) {
          return false;
        }
      }
    }

    for (int local = 0; local < patch_n; ++local) {
      const std::size_t local_us = static_cast<std::size_t>(local);
      const int c = first + local;
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double positive = std::max(boundary_positive[local_us], 0.0);
      const double requested_out = std::max(boundary_out_request[local_us], 0.0);
      if (positive == 0.0 && requested_out == 0.0) {
        continue;
      }
      const double available =
          std::max(boundary_available[c_us], 0.0) + positive;
      const double allowed_out = std::min(requested_out, available);
      const double limited = requested_out - allowed_out;
      boundary_available[c_us] = std::max(available - allowed_out, 0.0);
      const double signed_limited = positive - allowed_out;
      rhs[local_us] += signed_limited / current_dt;
      boundary_current_cell[c_us] += signed_limited * current_to_step_energy;
      boundary_E_in += static_cast<long double>(positive * current_to_step_energy);
      boundary_E_out += static_cast<long double>(allowed_out * current_to_step_energy);
      boundary_limited_E += static_cast<long double>(limited * current_to_step_energy);
    }

    if (last == n - 1 && in.has_physical_outer_vacuum) {
      const double outer_sink = face_area(in, n) * 0.25 * core::constants::c_light;
      if (!finite(outer_sink) || outer_sink < 0.0) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       n - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       outer_sink);
        return false;
      }
      diag.back() += outer_sink;
      boundary_sink_cell[static_cast<std::size_t>(n - 1)] += outer_sink;
    }

    if (!solve_tridiagonal(lower, diag, upper, rhs, sol)) {
      record_failure(out,
                     HoloLOFailureStage::SpatialSolve,
                     first,
                     group,
                     in,
                     E_work,
                     Te,
                     std::numeric_limits<double>::quiet_NaN());
      return false;
    }

    for (int local = 0; local < patch_n; ++local) {
      const int c = first + local;
      const std::size_t c_us = static_cast<std::size_t>(c);
      const std::size_t key = idx(c, group, in.n_groups);
      const double E0 = E_initial[key];
      const double E_new = sol[static_cast<std::size_t>(local)];
      if (!finite(E_new) || E_new < -1.0e-12 * std::max(E0, 1.0)) {
        record_failure(out, HoloLOFailureStage::SpatialSolve, c, group, in, E_work, Te, E_new);
        return false;
      }
      E_group[c_us] = std::max(E_new, 0.0);
    }

    first = last + 1;
  }

  for (int c = 0; c < n; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const std::size_t key = idx(c, group, in.n_groups);
    if (!lo_cell_active(in, c)) {
      E_work[key] = E_initial[key];
      continue;
    }
    const double E0 = E_initial[key];
    E_work[key] = E_group[c_us];
    const double V = finite_or_zero(in.vol[c]);
    long double diffusion_rate = 0.0L;
    if (c > 0) {
      diffusion_rate +=
          static_cast<long double>(face_couplings[c_us]) *
          static_cast<long double>(E_group[c_us] - E_group[c_us - 1U]);
    }
    if (c + 1 < n) {
      diffusion_rate +=
          static_cast<long double>(face_couplings[c_us + 1U]) *
          static_cast<long double>(E_group[c_us] - E_group[c_us + 1U]);
    }
    if (boundary_sink_cell[c_us] > 0.0) {
      diffusion_rate +=
          static_cast<long double>(boundary_sink_cell[c_us]) *
          static_cast<long double>(E_group[c_us]);
    }
    const long double source_matter_delta_ld =
        static_cast<long double>(boundary_current_cell[c_us]) -
        static_cast<long double>(V) *
            static_cast<long double>(E_group[c_us] - E0) -
        static_cast<long double>(in.dt) * diffusion_rate;
    const double source_matter_delta = static_cast<double>(source_matter_delta_ld);
    if (!finite(source_matter_delta)) {
      record_failure(out,
                     HoloLOFailureStage::SourceSolve,
                     c,
                     group,
                     in,
                     E_work,
                     Te,
                     source_matter_delta);
      return false;
    }
    source_matter_delta_cell[c_us] += source_matter_delta;
    global_source_matter_delta += source_matter_delta_ld;
    add_corrected_source_diagnostic(
        rad_dep_lo[key],
        rad_emit_lo[key],
        static_cast<long double>(alpha[c_us]) * static_cast<long double>(E_work[key]) *
            static_cast<long double>(V),
        static_cast<long double>(alpha[c_us]) * static_cast<long double>(B[c_us]) *
            static_cast<long double>(V),
        source_matter_delta);
    const auto accumulate_dirichlet_face_balance = [&](const int face) {
      const int neighbor_cell = dirichlet_neighbor_face[static_cast<std::size_t>(face)];
      if (neighbor_cell < 0) {
        return;
      }
      const double coupling = face_couplings[static_cast<std::size_t>(face)];
      const double E_neighbor =
          nonnegative_finite(E_initial[idx(neighbor_cell, group, in.n_groups)]);
      boundary_E_in += static_cast<long double>(coupling) *
                       static_cast<long double>(E_neighbor) *
                       static_cast<long double>(in.dt);
      boundary_E_out += static_cast<long double>(coupling) *
                        static_cast<long double>(E_group[c_us]) *
                        static_cast<long double>(in.dt);
    };
    if (c > 0) {
      accumulate_dirichlet_face_balance(c);
    }
    if (c + 1 < n) {
      accumulate_dirichlet_face_balance(c + 1);
    }
    if (boundary_sink_cell[c_us] > 0.0) {
      boundary_E_out +=
          static_cast<long double>(boundary_sink_cell[c_us]) *
          static_cast<long double>(E_group[c_us]) *
          static_cast<long double>(in.dt);
    }
  }

  out.boundary_E_in = static_cast<double>(boundary_E_in);
  out.boundary_E_out = static_cast<double>(boundary_E_out);
  out.boundary_limited_E = static_cast<double>(boundary_limited_E);
  out.solver_iterations = std::max(out.solver_iterations, 1);
  return true;
}

[[nodiscard]] bool solve_global_group_qd(const HoloLOInputs& in,
                                         const int group,
                                         const std::vector<double>& E_initial,
                                         const std::vector<double>& Te,
                                         std::vector<double>& E_work,
                                         std::vector<double>& rad_dep_lo,
                                         std::vector<double>& rad_emit_lo,
                                         std::vector<double>& source_matter_delta_cell,
                                         long double& global_source_matter_delta,
                                         long double& boundary_E_in,
                                         long double& boundary_E_out,
                                         long double& boundary_limited_E,
                                         HoloLOResult& out) {
  struct BoundaryCurrentEvent {
    int local = -1;
    int face = -1;
    int active_side = 0;  // +1: active cell is right of face, -1: active is left.
    double signed_energy = 0.0;
  };

  const int n = in.n_cells;
  std::vector<double> B(static_cast<std::size_t>(n), 0.0);
  std::vector<double> alpha(static_cast<std::size_t>(n), 0.0);
  std::vector<double> face_fluxes(static_cast<std::size_t>(n + 1), 0.0);
  std::vector<std::uint8_t> qd_transport_face(static_cast<std::size_t>(n + 1), 0U);
  std::vector<std::uint8_t> qd_solved_face(static_cast<std::size_t>(n + 1), 0U);
  std::vector<std::uint8_t> current_boundary_face(static_cast<std::size_t>(n + 1), 0U);
  std::vector<int> dirichlet_active_cell(static_cast<std::size_t>(n + 1), -1);
  std::vector<double> current_face_energy(static_cast<std::size_t>(n + 1), 0.0);
  std::vector<double> boundary_sink_cell(static_cast<std::size_t>(n), 0.0);
  std::vector<double> boundary_current_cell(static_cast<std::size_t>(n), 0.0);
  std::vector<double> boundary_available(static_cast<std::size_t>(n), 0.0);
  std::vector<double> E_group(static_cast<std::size_t>(n), 0.0);

  const double T_floor = std::max(finite_or_zero(in.temperature_floor_eV), 1.0e-12);
  for (int c = 0; c < n; ++c) {
    const std::size_t key = idx(c, group, in.n_groups);
    const double V = finite_or_zero(in.vol[c]);
    const double E0 = E_initial[key];
    if (!(V > 0.0) || !finite(E0) || E0 < 0.0) {
      record_failure(out, HoloLOFailureStage::SpatialSolve, c, group, in, E_work, Te, E0);
      return false;
    }
    E_group[static_cast<std::size_t>(c)] = E0;
    boundary_available[static_cast<std::size_t>(c)] = V * E0;
  }

  const double current_dt = face_current_dt(in);
  const double current_to_step_energy =
      (current_dt > 0.0) ? (in.dt / current_dt) : 1.0;
  for (int first = 0; first < n;) {
    while (first < n && !lo_cell_active(in, first)) {
      ++first;
    }
    if (first >= n) {
      break;
    }
    int last = first;
    while (last + 1 < n && lo_cell_active(in, last + 1)) {
      ++last;
    }

    const int patch_n = last - first + 1;
    const int system_n = 2 * patch_n - 1;
    std::vector<double> lower(static_cast<std::size_t>(system_n), 0.0);
    std::vector<double> diag(static_cast<std::size_t>(system_n), 0.0);
    std::vector<double> upper(static_cast<std::size_t>(system_n), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(system_n), 0.0);
    std::vector<double> boundary_positive(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> boundary_out_request(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<double> boundary_out_allowed(static_cast<std::size_t>(patch_n), 0.0);
    std::vector<BoundaryCurrentEvent> boundary_current_events;
    std::vector<double> sol;
    const auto e_index = [](const int local) { return 2 * local; };
    const auto f_index = [](const int left_local) { return 2 * left_local + 1; };

    for (int local = 0; local < patch_n; ++local) {
      const int c = first + local;
      const std::size_t e_us = static_cast<std::size_t>(e_index(local));
      const std::size_t c_us = static_cast<std::size_t>(c);
      const std::size_t key = idx(c, group, in.n_groups);
      const double V = finite_or_zero(in.vol[c]);
      const double E0 = E_initial[key];
      const double time_coeff = V / in.dt;
      const double sigma = nonnegative_finite(in.sigma_P[key]);
      alpha[c_us] = core::constants::c_light * sigma * in.dt;
      B[c_us] = planck_B(in,
                         group,
                         std::max(finite_or_zero(Te[c_us]), T_floor));
      if (!finite(alpha[c_us]) || !finite(B[c_us])) {
        record_failure(out,
                       HoloLOFailureStage::SourceSolve,
                       c,
                       group,
                       in,
                       E_work,
                       Te,
                       B[c_us]);
        return false;
      }

      diag[e_us] = time_coeff * (1.0 + alpha[c_us]);
      rhs[e_us] = time_coeff * (E0 + alpha[c_us] * B[c_us]);
      if (in.consistency_source != nullptr) {
        rhs[e_us] +=
            in.consistency_alpha * finite_or_zero(in.consistency_source[key]);
      }
    }

    const auto add_qd_face = [&](const int face,
                                 const int left_local,
                                 const int right_local) {
      const int left = face - 1;
      const int right = face;
      const double dr = cell_center(in, right) - cell_center(in, left);
      if (!(dr > 0.0)) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       face - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       dr);
        return false;
      }

      const double sigma_left =
          std::max(nonnegative_finite(in.sigma_R[idx(left, group, in.n_groups)]),
                   kSigmaFloor);
      const double sigma_right =
          std::max(nonnegative_finite(in.sigma_R[idx(right, group, in.n_groups)]),
                   kSigmaFloor);
      const double sigma_face = 0.5 * (sigma_left + sigma_right);
      const double beta = 1.0 / (core::constants::c_light * in.dt);
      const double denom = beta + sigma_face;
      const double area = face_area(in, face);
      if (!(denom > 0.0) || !finite(denom) || !finite(area) || area < 0.0) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       face - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       denom);
        return false;
      }

      const double f_left = qd_chi(in, left, group);
      const double f_right = qd_chi(in, right, group);
      const double f_avg = 0.5 * (f_left + f_right);
      const double geom = qd_geometry_correction(in, left, face, f_avg, dr);
      const double geom_rhs =
          geom * (E_initial[idx(left, group, in.n_groups)] +
                  E_initial[idx(right, group, in.n_groups)]);
      const int e_left = e_index(left_local);
      const int e_right = e_index(right_local);
      const int f_row = f_index(left_local);
      const std::size_t e_left_us = static_cast<std::size_t>(e_left);
      const std::size_t e_right_us = static_cast<std::size_t>(e_right);
      const std::size_t f_us = static_cast<std::size_t>(f_row);

      upper[e_left_us] += area;
      lower[e_right_us] -= area;
      lower[f_us] = core::constants::c_light * (-f_left / dr);
      diag[f_us] = denom;
      upper[f_us] = core::constants::c_light * (f_right / dr);
      rhs[f_us] = beta * qd_old_flux(in, face, group) -
                  core::constants::c_light * geom_rhs;
      qd_transport_face[static_cast<std::size_t>(face)] = 1U;
      qd_solved_face[static_cast<std::size_t>(face)] = 1U;
      return true;
    };

    for (int face = first + 1; face <= last; ++face) {
      const int left_local = face - 1 - first;
      const int right_local = face - first;
      if (!add_qd_face(face, left_local, right_local)) {
        return false;
      }
    }

    const auto add_boundary_current = [&](const int local,
                                          const int face,
                                          const int active_side,
                                          const double signed_energy) {
      if (!finite(signed_energy) || signed_energy == 0.0) {
        return;
      }
      const std::size_t local_us = static_cast<std::size_t>(local);
      if (signed_energy > 0.0) {
        boundary_positive[local_us] += signed_energy;
      } else {
        boundary_out_request[local_us] += -signed_energy;
      }
      boundary_current_events.push_back(
          BoundaryCurrentEvent{local, face, active_side, signed_energy});
    };

    if (in.face_current != nullptr) {
      if (first > 0) {
        add_boundary_current(0, first, 1, face_current_energy(in, first, group));
      }
      if (last + 1 < n) {
        add_boundary_current(patch_n - 1,
                             last + 1,
                             -1,
                             -face_current_energy(in, last + 1, group));
      }
    } else {
      const auto add_dirichlet_boundary = [&](const int local, const int face,
                                              const int neighbor_cell) {
        const std::size_t e_us = static_cast<std::size_t>(e_index(local));
        const int c = first + local;
        const QDFaceCoefficients coeff = qd_face_coefficients(in, face, group);
        const double E_neighbor =
            nonnegative_finite(E_initial[idx(neighbor_cell, group, in.n_groups)]);
        if (!coeff.valid || !finite(E_neighbor)) {
          record_failure(out,
                         HoloLOFailureStage::SpatialSolve,
                         c,
                         group,
                         in,
                         E_work,
                         Te,
                         coeff.dF_dE_left);
          return false;
        }
        if (c == face) {
          diag[e_us] -= coeff.area * coeff.dF_dE_right;
          rhs[e_us] += coeff.area *
                       (coeff.dF_dE_left * E_neighbor + coeff.F_const);
        } else {
          diag[e_us] += coeff.area * coeff.dF_dE_left;
          rhs[e_us] -= coeff.area *
                       (coeff.dF_dE_right * E_neighbor + coeff.F_const);
        }
        qd_transport_face[static_cast<std::size_t>(face)] = 1U;
        dirichlet_active_cell[static_cast<std::size_t>(face)] = c;
        return true;
      };

      if (first > 0 && !lo_cell_active(in, first - 1)) {
        if (!add_dirichlet_boundary(0, first, first - 1)) {
          return false;
        }
      }
      if (last + 1 < n && !lo_cell_active(in, last + 1)) {
        if (!add_dirichlet_boundary(patch_n - 1, last + 1, last + 1)) {
          return false;
        }
      }
    }

    for (int local = 0; local < patch_n; ++local) {
      const std::size_t local_us = static_cast<std::size_t>(local);
      const int c = first + local;
      const std::size_t c_us = static_cast<std::size_t>(c);
      const double positive = std::max(boundary_positive[local_us], 0.0);
      const double requested_out = std::max(boundary_out_request[local_us], 0.0);
      if (positive == 0.0 && requested_out == 0.0) {
        continue;
      }
      const double available =
          std::max(boundary_available[c_us], 0.0) + positive;
      const double allowed_out = std::min(requested_out, available);
      const double limited = requested_out - allowed_out;
      boundary_available[c_us] = std::max(available - allowed_out, 0.0);
      boundary_out_allowed[local_us] = allowed_out;
      const double signed_limited = positive - allowed_out;
      rhs[static_cast<std::size_t>(e_index(local))] += signed_limited / current_dt;
      boundary_current_cell[c_us] += signed_limited * current_to_step_energy;
      boundary_E_in += static_cast<long double>(positive * current_to_step_energy);
      boundary_E_out += static_cast<long double>(allowed_out * current_to_step_energy);
      boundary_limited_E += static_cast<long double>(limited * current_to_step_energy);
    }

    for (const BoundaryCurrentEvent& event : boundary_current_events) {
      if (event.local < 0 || event.local >= patch_n) {
        continue;
      }
      const std::size_t local_us = static_cast<std::size_t>(event.local);
      double signed_limited = event.signed_energy;
      if (event.signed_energy < 0.0) {
        const double requested_out = std::max(boundary_out_request[local_us], 0.0);
        const double out_scale =
            (requested_out > 0.0) ? (boundary_out_allowed[local_us] / requested_out) : 0.0;
        signed_limited = event.signed_energy * out_scale;
      }
      const double oriented_energy =
          (event.active_side > 0) ? signed_limited : -signed_limited;
      current_face_energy[static_cast<std::size_t>(event.face)] += oriented_energy;
      current_boundary_face[static_cast<std::size_t>(event.face)] = 1U;
    }

    if (last == n - 1 && in.has_physical_outer_vacuum) {
      const double outer_sink = face_area(in, n) * 0.25 * core::constants::c_light;
      if (!finite(outer_sink) || outer_sink < 0.0) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       n - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       outer_sink);
        return false;
      }
      diag[static_cast<std::size_t>(e_index(patch_n - 1))] += outer_sink;
      boundary_sink_cell[static_cast<std::size_t>(n - 1)] += outer_sink;
    }

    if (!solve_tridiagonal_general(lower, diag, upper, rhs, sol)) {
      record_failure(out,
                     HoloLOFailureStage::SpatialSolve,
                     first,
                     group,
                     in,
                     E_work,
                     Te,
                     std::numeric_limits<double>::quiet_NaN());
      return false;
    }

    for (int local = 0; local < patch_n; ++local) {
      const int c = first + local;
      const std::size_t c_us = static_cast<std::size_t>(c);
      const std::size_t key = idx(c, group, in.n_groups);
      const double E0 = E_initial[key];
      const double E_new = sol[static_cast<std::size_t>(e_index(local))];
      if (!finite(E_new)) {
        record_failure(out, HoloLOFailureStage::SpatialSolve, c, group, in, E_work, Te, E_new);
        return false;
      }
      // Multigroup QD can produce small group-wise undershoots while the cell
      // total remains positive; clamp only sub-cell-significant negatives.
      const double negative_tol =
          qd_negative_energy_tolerance(in, E_initial, B, c, group);
      if (E_new < -negative_tol) {
        record_failure(out, HoloLOFailureStage::SpatialSolve, c, group, in, E_work, Te, E_new);
        return false;
      }
      E_group[c_us] = std::max(E_new, 0.0);
    }
    for (int face = first + 1; face <= last; ++face) {
      const int left_local = face - 1 - first;
      const double flux = sol[static_cast<std::size_t>(f_index(left_local))];
      if (!finite(flux)) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       face - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       flux);
        return false;
      }
      face_fluxes[static_cast<std::size_t>(face)] = flux;
    }

    first = last + 1;
  }

  for (int face = 1; face < n; ++face) {
    const std::size_t face_us = static_cast<std::size_t>(face);
    if (qd_transport_face[face_us] != 0U) {
      if (qd_solved_face[face_us] != 0U) {
        if (finite(face_fluxes[face_us])) {
          continue;
        }
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       face - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       face_fluxes[face_us]);
        return false;
      }
      const QDFaceCoefficients coeff = qd_face_coefficients(in, face, group);
      const double flux =
          qd_face_flux(coeff,
                       E_group[static_cast<std::size_t>(face - 1)],
                       E_group[static_cast<std::size_t>(face)]);
      if (!coeff.valid || !finite(flux)) {
        record_failure(out,
                       HoloLOFailureStage::SpatialSolve,
                       face - 1,
                       group,
                       in,
                       E_work,
                       Te,
                       flux);
        return false;
      }
      face_fluxes[face_us] = flux;
    } else if (current_boundary_face[face_us] != 0U) {
      const double A = face_area(in, face);
      if (A > 0.0 && finite(A) && current_dt > 0.0) {
        face_fluxes[face_us] = current_face_energy[face_us] / (current_dt * A);
      }
    }
  }
  face_fluxes[0] = 0.0;
  if (in.has_physical_outer_vacuum && n > 0) {
    face_fluxes[static_cast<std::size_t>(n)] =
        0.25 * core::constants::c_light * E_group[static_cast<std::size_t>(n - 1)];
  } else {
    face_fluxes[static_cast<std::size_t>(n)] = 0.0;
  }

  for (int c = 0; c < n; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    const std::size_t key = idx(c, group, in.n_groups);
    if (!lo_cell_active(in, c)) {
      E_work[key] = E_initial[key];
      continue;
    }
    const double E0 = E_initial[key];
    E_work[key] = E_group[c_us];
    const double V = finite_or_zero(in.vol[c]);
    long double transport_rate = 0.0L;
    if (c > 0 && qd_transport_face[c_us] != 0U) {
      transport_rate -= static_cast<long double>(face_area(in, c)) *
                        static_cast<long double>(face_fluxes[c_us]);
    }
    if (c + 1 < n && qd_transport_face[c_us + 1U] != 0U) {
      transport_rate += static_cast<long double>(face_area(in, c + 1)) *
                        static_cast<long double>(face_fluxes[c_us + 1U]);
    }
    if (boundary_sink_cell[c_us] > 0.0) {
      transport_rate +=
          static_cast<long double>(boundary_sink_cell[c_us]) *
          static_cast<long double>(E_group[c_us]);
    }
    const long double source_matter_delta_ld =
        static_cast<long double>(boundary_current_cell[c_us]) -
        static_cast<long double>(V) *
            static_cast<long double>(E_group[c_us] - E0) -
        static_cast<long double>(in.dt) * transport_rate;
    const double source_matter_delta = static_cast<double>(source_matter_delta_ld);
    if (!finite(source_matter_delta)) {
      record_failure(out,
                     HoloLOFailureStage::SourceSolve,
                     c,
                     group,
                     in,
                     E_work,
                     Te,
                     source_matter_delta);
      return false;
    }
    source_matter_delta_cell[c_us] += source_matter_delta;
    global_source_matter_delta += source_matter_delta_ld;
    add_corrected_source_diagnostic(
        rad_dep_lo[key],
        rad_emit_lo[key],
        static_cast<long double>(alpha[c_us]) * static_cast<long double>(E_work[key]) *
            static_cast<long double>(V),
        static_cast<long double>(alpha[c_us]) * static_cast<long double>(B[c_us]) *
            static_cast<long double>(V),
        source_matter_delta);
    const auto accumulate_dirichlet_face_balance = [&](const int face) {
      if (dirichlet_active_cell[static_cast<std::size_t>(face)] != c) {
        return;
      }
      const double energy =
          face_area(in, face) * face_fluxes[static_cast<std::size_t>(face)] * in.dt;
      if (c == face - 1) {
        if (energy >= 0.0) {
          boundary_E_out += static_cast<long double>(energy);
        } else {
          boundary_E_in += static_cast<long double>(-energy);
        }
      } else {
        if (energy >= 0.0) {
          boundary_E_in += static_cast<long double>(energy);
        } else {
          boundary_E_out += static_cast<long double>(-energy);
        }
      }
    };
    if (c > 0) {
      accumulate_dirichlet_face_balance(c);
    }
    if (c + 1 < n) {
      accumulate_dirichlet_face_balance(c + 1);
    }
    if (boundary_sink_cell[c_us] > 0.0) {
      boundary_E_out +=
          static_cast<long double>(boundary_sink_cell[c_us]) *
          static_cast<long double>(E_group[c_us]) *
          static_cast<long double>(in.dt);
    }
  }

  if (in.F_lo != nullptr) {
    for (int face = 0; face <= n; ++face) {
      in.F_lo[idx(face, group, in.n_groups)] =
          finite_or_zero(face_fluxes[static_cast<std::size_t>(face)]);
    }
  }
  out.boundary_E_in = static_cast<double>(boundary_E_in);
  out.boundary_E_out = static_cast<double>(boundary_E_out);
  out.boundary_limited_E = static_cast<double>(boundary_limited_E);
  out.solver_iterations = std::max(out.solver_iterations, 1);
  return true;
}

[[nodiscard]] bool apply_material_delta_cell(const HoloLOInputs& in,
                                             const int cell,
                                             const double matter_delta,
                                             std::vector<double>& ee,
                                             std::vector<double>& Te,
                                             std::vector<double>& Pe,
                                             const std::vector<double>& E,
                                             HoloLOResult& out) {
  if (!lo_material_commit_cell(in, cell)) {
    return true;
  }

  const double m = cell_mass(in, cell);
  const double e_old = ee[static_cast<std::size_t>(cell)];
  if (!(m > 0.0) || !finite(e_old)) {
    record_failure(out, HoloLOFailureStage::SourceSolve, cell, -1, in, E, Te, e_old);
    return false;
  }

  const double T_floor = std::max(finite_or_zero(in.temperature_floor_eV), 1.0e-12);
  double applied_delta = matter_delta;
  double e_floor = 0.0;
  double cv = 0.0;
  double e_new = e_old;
  double T_new = T_floor;
  const double rho = std::max(finite_or_zero(in.rho[cell]), kEnergyFloor);
  if (in.material_model == HoloLOMaterialModel::TableEOS) {
    e_floor = in.electron_eos->energy(rho, T_floor);
    if (!finite(e_floor)) {
      record_failure(out, HoloLOFailureStage::SourceSolve, cell, -1, in, E, Te, e_floor);
      return false;
    }
  } else {
    cv = ideal_cv(in, cell, ee, Te);
    if (!(cv > 0.0) || !finite(cv)) {
      record_failure(out, HoloLOFailureStage::SourceSolve, cell, -1, in, E, Te, cv);
      return false;
    }
    e_floor = cv * T_floor;
  }

  if (applied_delta < 0.0) {
    const double max_removal = m * std::max(e_old - e_floor, 0.0);
    if (finite(max_removal) && std::abs(applied_delta) > max_removal) {
      applied_delta = -max_removal;
    }
  }

  const double e_trial = e_old + applied_delta / m;
  if (!finite(e_trial)) {
    record_failure(out, HoloLOFailureStage::SourceSolve, cell, -1, in, E, Te, e_trial);
    return false;
  }

  if (in.material_model == HoloLOMaterialModel::TableEOS) {
    e_new = std::max(e_trial, e_floor);
    T_new = std::max(in.electron_eos->temperature_from_energy(rho, e_new), T_floor);
    e_new = in.electron_eos->energy(rho, T_new);
  } else {
    e_new = std::max(e_trial, e_floor);
    T_new = e_new / cv;
  }

  if (!finite(e_new) || !finite(T_new) || !(T_new > 0.0)) {
    record_failure(out, HoloLOFailureStage::SourceSolve, cell, -1, in, E, Te, e_new);
    return false;
  }

  ee[static_cast<std::size_t>(cell)] = e_new;
  Te[static_cast<std::size_t>(cell)] = T_new;
  Pe[static_cast<std::size_t>(cell)] = material_pressure(in, cell, T_new, e_new);
  out.matter_delta += m * (e_new - e_old);
  return true;
}

void validate_inputs(const HoloLOInputs& in) {
  if (in.n_cells <= 0) {
    return;
  }
  TENRYU_ASSERT(in.dimension == HoloGeometryDimension::Spherical1D,
                "solve_holo_lo_1d_cpu supports Spherical1D geometry only in v1");
  TENRYU_ASSERT(in.n_groups > 0, "solve_holo_lo_1d_cpu requires n_groups > 0");
  TENRYU_ASSERT(in.dt > 0.0, "solve_holo_lo_1d_cpu requires dt > 0");
  TENRYU_ASSERT(std::isfinite(in.consistency_alpha) &&
                    in.consistency_alpha >= 0.0 &&
                    in.consistency_alpha <= 1.0,
                "solve_holo_lo_1d_cpu requires consistency_alpha in [0, 1]");
  TENRYU_ASSERT(in.E_lo != nullptr, "solve_holo_lo_1d_cpu requires E_lo");
  TENRYU_ASSERT(in.ee != nullptr, "solve_holo_lo_1d_cpu requires ee");
  TENRYU_ASSERT(in.Te != nullptr, "solve_holo_lo_1d_cpu requires Te");
  TENRYU_ASSERT(in.Pe != nullptr, "solve_holo_lo_1d_cpu requires Pe");
  TENRYU_ASSERT(in.sigma_P != nullptr, "solve_holo_lo_1d_cpu requires sigma_P");
  TENRYU_ASSERT(in.sigma_R != nullptr, "solve_holo_lo_1d_cpu requires sigma_R");
  TENRYU_ASSERT(in.rho != nullptr, "solve_holo_lo_1d_cpu requires rho");
  TENRYU_ASSERT(in.vol != nullptr, "solve_holo_lo_1d_cpu requires vol");
  TENRYU_ASSERT(in.node_r != nullptr, "solve_holo_lo_1d_cpu requires node_r");
  TENRYU_ASSERT(in.n_groups <=
                    std::numeric_limits<int>::max() / std::max(in.n_cells, 1),
                "solve_holo_lo_1d_cpu n_cells*n_groups overflow");
  if (in.n_groups > 1) {
    TENRYU_ASSERT(in.planck != nullptr,
                  "solve_holo_lo_1d_cpu requires planck table for multigroup solve");
  }
  if (in.face_current != nullptr) {
    TENRYU_ASSERT(in.face_current_dt > 0.0,
                  "solve_holo_lo_1d_cpu requires face_current_dt > 0 with face_current");
  }
  if (in.material_model == HoloLOMaterialModel::TableEOS) {
    TENRYU_ASSERT(in.electron_eos != nullptr,
                  "solve_holo_lo_1d_cpu table EOS path requires electron_eos");
  }
}

}  // namespace

const char* holo_lo_failure_stage_name(const HoloLOFailureStage stage) {
  switch (stage) {
    case HoloLOFailureStage::None:
      return "none";
    case HoloLOFailureStage::SpatialSolve:
      return "spatial";
    case HoloLOFailureStage::SourceSolve:
      return "source";
    case HoloLOFailureStage::Conservation:
      return "conservation";
  }
  return "unknown";
}

HoloLOResult solve_holo_lo_1d_cpu(const HoloLOInputs& in) {
  return solve_holo_lo_1d_cpu(in, false);
}

HoloLOResult solve_holo_lo_1d_cpu(const HoloLOInputs& in,
                                  const bool use_quasidiffusion) {
  HoloLOResult out{};
  if (in.n_cells <= 0) {
    return out;
  }
  validate_inputs(in);

  const std::size_t n_cells = static_cast<std::size_t>(in.n_cells);
  const std::size_t n_groups = static_cast<std::size_t>(in.n_groups);
  const std::size_t n_total = n_cells * n_groups;
  std::vector<double> E_initial(in.E_lo, in.E_lo + n_total);
  std::vector<double> E_work = E_initial;
  std::vector<double> ee_work(in.ee, in.ee + n_cells);
  std::vector<double> Te_work(in.Te, in.Te + n_cells);
  std::vector<double> Pe_work(in.Pe, in.Pe + n_cells);
  std::vector<double> rad_dep_work(n_total, 0.0);
  std::vector<double> rad_emit_work(n_total, 0.0);
  std::vector<double> source_matter_delta_cell(n_cells, 0.0);
  long double global_source_matter_delta = 0.0L;
  long double boundary_E_in = 0.0L;
  long double boundary_E_out = 0.0L;
  long double boundary_limited_E = 0.0L;

  if (in.rad_dep_lo != nullptr) {
    std::copy(in.rad_dep_lo, in.rad_dep_lo + n_total, rad_dep_work.begin());
  }
  if (in.rad_emit_lo != nullptr) {
    std::copy(in.rad_emit_lo, in.rad_emit_lo + n_total, rad_emit_work.begin());
  }
  if (in.matter_delta_lo_cell != nullptr) {
    std::fill(in.matter_delta_lo_cell, in.matter_delta_lo_cell + n_cells, 0.0);
  }

  for (int g = 0; g < in.n_groups; ++g) {
    const bool solved =
        use_quasidiffusion
            ? solve_global_group_qd(in,
                                    g,
                                    E_initial,
                                    Te_work,
                                    E_work,
                                    rad_dep_work,
                                    rad_emit_work,
                                    source_matter_delta_cell,
                                    global_source_matter_delta,
                                    boundary_E_in,
                                    boundary_E_out,
                                    boundary_limited_E,
                                    out)
            : solve_global_group(in,
                                 g,
                                 E_initial,
                                 Te_work,
                                 E_work,
                                 rad_dep_work,
                                 rad_emit_work,
                                 source_matter_delta_cell,
                                 global_source_matter_delta,
                                 boundary_E_in,
                                 boundary_E_out,
                                 boundary_limited_E,
                                 out);
    if (!solved) {
      ++out.failures;
      return out;
    }
  }

  for (int c = 0; c < in.n_cells; ++c) {
    if (!apply_material_delta_cell(in,
                                   c,
                                   source_matter_delta_cell[static_cast<std::size_t>(c)],
                                   ee_work,
                                   Te_work,
                                   Pe_work,
                                   E_work,
                                   out)) {
      ++out.failures;
      return out;
    }
  }
  long double rad_delta = 0.0L;
  for (int c = 0; c < in.n_cells; ++c) {
    const double V = finite_or_zero(in.vol[c]);
    for (int g = 0; g < in.n_groups; ++g) {
      const std::size_t key = idx(c, g, in.n_groups);
      rad_delta += static_cast<long double>(V) *
                   static_cast<long double>(E_work[key] - E_initial[key]);
    }
  }
  out.rad_delta = static_cast<double>(rad_delta);

  out.boundary_E_in = static_cast<double>(boundary_E_in);
  out.boundary_E_out = static_cast<double>(boundary_E_out);
  out.boundary_limited_E = static_cast<double>(boundary_limited_E);
  const long double boundary_net = boundary_E_in - boundary_E_out;
  const long double conservation_error =
      global_source_matter_delta + rad_delta - boundary_net;
  out.conservation_error = static_cast<double>(conservation_error);
  const double balance_scale =
      std::max({static_cast<double>(std::abs(global_source_matter_delta)),
                static_cast<double>(std::abs(rad_delta)),
                static_cast<double>(std::abs(boundary_net)), kEnergyFloor});
  if (!finite(out.conservation_error) ||
      std::abs(out.conservation_error) > kConservationRelTol * balance_scale) {
    record_failure(out,
                   HoloLOFailureStage::Conservation,
                   -1,
                   -1,
                   in,
                   E_work,
                   Te_work,
                   out.conservation_error);
    ++out.failures;
    return out;
  }

  std::copy(E_work.begin(), E_work.end(), in.E_lo);
  std::copy(ee_work.begin(), ee_work.end(), in.ee);
  std::copy(Te_work.begin(), Te_work.end(), in.Te);
  std::copy(Pe_work.begin(), Pe_work.end(), in.Pe);
  if (in.rad_dep_lo != nullptr) {
    std::copy(rad_dep_work.begin(), rad_dep_work.end(), in.rad_dep_lo);
  }
  if (in.rad_emit_lo != nullptr) {
    std::copy(rad_emit_work.begin(), rad_emit_work.end(), in.rad_emit_lo);
  }
  if (in.matter_delta_lo_cell != nullptr) {
    std::copy(source_matter_delta_cell.begin(),
              source_matter_delta_cell.end(),
              in.matter_delta_lo_cell);
  }
  return out;
}

}  // namespace tenryu::radiation
