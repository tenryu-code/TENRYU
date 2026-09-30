#include "radiation/ddmc_coefficients.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>

#include "core/axis_tolerance.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kMilneLambda = 0.7104;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kPi = 3.14159265358979323846;

const char* face_name_2d(const int face) {
  switch (face) {
    case 0:
      return "r_inner";
    case 1:
      return "r_outer";
    case 2:
      return "z_bottom";
    case 3:
      return "z_top";
    default:
      return "unknown_face";
  }
}

void warn_negative_leak_coeff(const std::int64_t cell,
                              const int group,
                              const char* face_label,
                              const double leak_coeff) {
  static int neg_leak_count = 0;
  if (neg_leak_count++ >= 10) {
    return;
  }

  std::cerr << "[WARN] Negative DDMC leak coefficient in cell " << cell
            << " group " << group << " (" << face_label << "): " << leak_coeff
            << " (clamped to 0)\n";
}

int node_index_2d_rz(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

int cell_index_2d_rz(const int i, const int j, const int nz) {
  return i * nz + j;
}

void face_endpoints_2d_rz(const int face,
                          const double r0,
                          const double z0,
                          const double r1,
                          const double z1,
                          const double r2,
                          const double z2,
                          const double r3,
                          const double z3,
                          double* ra,
                          double* za,
                          double* rb,
                          double* zb) {
  if (face == 0) {
    *ra = r0;
    *za = z0;
    *rb = r3;
    *zb = z3;
  } else if (face == 1) {
    *ra = r1;
    *za = z1;
    *rb = r2;
    *zb = z2;
  } else if (face == 2) {
    *ra = r0;
    *za = z0;
    *rb = r1;
    *zb = z1;
  } else {
    *ra = r3;
    *za = z3;
    *rb = r2;
    *zb = z2;
  }
}

}  // namespace

std::size_t DDMCCoefficients::cell_index(const std::int64_t cell, const int group) const {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups_) +
         static_cast<std::size_t>(group);
}

std::size_t DDMCCoefficients::face_index(const std::int64_t face, const int group) const {
  return static_cast<std::size_t>(face) * static_cast<std::size_t>(n_groups_) +
         static_cast<std::size_t>(group);
}

double ConstantOpacityProvider::sigma_rosseland(int,
                                                int group,
                                                double rho,
                                                double) const {
  if (group < 0 || group >= static_cast<int>(kappa_R_.size())) {
    return 0.0;
  }
  return std::max(rho, 0.0) * std::max(kappa_R_[static_cast<std::size_t>(group)], 0.0);
}

DDMCCoefficients::DDMCCoefficients(const std::int64_t n_cells,
                                   const int n_groups,
                                   const double sigma_floor,
                                   const double sigma_cap)
    : n_cells_(n_cells),
      n_groups_(n_groups),
      sigma_floor_(std::max(sigma_floor, 1.0e-30)),
      sigma_cap_(std::max(sigma_cap, std::max(sigma_floor, 1.0e-30))),
      cell_data_(static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups)),
      face_sigma_minus_(static_cast<std::size_t>(n_cells + 1) *
                        static_cast<std::size_t>(n_groups),
                        0.0),
      face_sigma_plus_(static_cast<std::size_t>(n_cells + 1) *
                       static_cast<std::size_t>(n_groups),
                       0.0) {}

double DDMCCoefficients::clamp_sigma(const double sigma) const {
  const double sigma_cap = std::max(sigma_cap_, sigma_floor_);
  return std::clamp(sigma, sigma_floor_, sigma_cap);
}

double DDMCCoefficients::compute_diffusion_coeff(const double sigma_R) {
  const double sigma_safe = std::max(sigma_R, 1.0e-30);
  return 1.0 / (3.0 * sigma_safe);
}

double DDMCCoefficients::compute_face_temperature(const double T_left_eV,
                                                  const double T_right_eV) {
  const double Tl = std::max(T_left_eV, 0.0);
  const double Tr = std::max(T_right_eV, 0.0);
  const double T4 = 0.5 * (Tl * Tl * Tl * Tl + Tr * Tr * Tr * Tr);
  return std::pow(std::max(T4, 0.0), 0.25);
}

void DDMCCoefficients::compute_1d(const std::vector<double>& node_r,
                                  const std::vector<double>& rho,
                                  const std::vector<double>& Te,
                                  const std::vector<double>& sigma_R_center,
                                  const ModeSelector& mode_selector,
                                  const DDMCBoundaryType bc_inner,
                                  const DDMCBoundaryType bc_outer,
                                  const bool m_matrix_check,
                                  const OpacityProvider* opacity_provider) {
  std::vector<double> dx(static_cast<std::size_t>(n_cells_), 0.0);
  for (std::int64_t c = 0; c < n_cells_; ++c) {
    const std::size_t c_us = static_cast<std::size_t>(c);
    dx[c_us] = std::max(node_r[c_us + 1] - node_r[c_us], 0.0);
  }

  for (std::int64_t face = 0; face <= n_cells_; ++face) {
    for (int g = 0; g < n_groups_; ++g) {
      const std::size_t idx = face_index(face, g);

      if (face == 0) {
        const double sigma = clamp_sigma(sigma_R_center[cell_index(0, g)]);
        face_sigma_minus_[idx] = sigma;
        face_sigma_plus_[idx] = sigma;
        continue;
      }
      if (face == n_cells_) {
        const double sigma = clamp_sigma(sigma_R_center[cell_index(n_cells_ - 1, g)]);
        face_sigma_minus_[idx] = sigma;
        face_sigma_plus_[idx] = sigma;
        continue;
      }

      const std::int64_t c_left = face - 1;
      const std::int64_t c_right = face;
      const double T_face =
          compute_face_temperature(Te[static_cast<std::size_t>(c_left)],
                                   Te[static_cast<std::size_t>(c_right)]);

      if (opacity_provider != nullptr) {
        const double sigma_left = opacity_provider->sigma_rosseland(
            static_cast<int>(c_left), g, rho[static_cast<std::size_t>(c_left)], T_face);
        const double sigma_right = opacity_provider->sigma_rosseland(
            static_cast<int>(c_right), g, rho[static_cast<std::size_t>(c_right)], T_face);
        face_sigma_minus_[idx] = clamp_sigma(sigma_left);
        face_sigma_plus_[idx] = clamp_sigma(sigma_right);
      } else {
        face_sigma_minus_[idx] = clamp_sigma(sigma_R_center[cell_index(c_left, g)]);
        face_sigma_plus_[idx] = clamp_sigma(sigma_R_center[cell_index(c_right, g)]);
      }
    }
  }

  for (std::int64_t c = 0; c < n_cells_; ++c) {
    for (int g = 0; g < n_groups_; ++g) {
      const std::size_t cg = cell_index(c, g);
      CellDDMCData data{};

      if (mode_selector.get_mode(c, g) != TransportMode::DDMC) {
        if (c > 0) {
          data.neighbor_face[0] = static_cast<int>(c - 1);
        }
        if (c + 1 < n_cells_) {
          data.neighbor_face[1] = static_cast<int>(c + 1);
        }
        data.bc_face[0] = (c == 0) ? bc_inner : DDMCBoundaryType::Internal;
        data.bc_face[1] = (c == n_cells_ - 1) ? bc_outer : DDMCBoundaryType::Internal;
        sync_legacy_aliases(data);
        cell_data_[cg] = data;
        continue;
      }

      const double dx_i = std::max(dx[static_cast<std::size_t>(c)], 1.0e-30);

      if (c == 0) {
        data.bc_left = bc_inner;
      } else if (mode_selector.get_mode(c - 1, g) == TransportMode::DDMC) {
        data.bc_left = DDMCBoundaryType::Internal;
      } else {
        data.bc_left = DDMCBoundaryType::Interface;
      }
      data.bc_face[0] = data.bc_left;
      data.neighbor_face[0] = (c > 0) ? static_cast<int>(c - 1) : -1;

      if (c == n_cells_ - 1) {
        data.bc_right = bc_outer;
      } else if (mode_selector.get_mode(c + 1, g) == TransportMode::DDMC) {
        data.bc_right = DDMCBoundaryType::Internal;
      } else {
        data.bc_right = DDMCBoundaryType::Interface;
      }
      data.bc_face[1] = data.bc_right;
      data.neighbor_face[1] = (c + 1 < n_cells_) ? static_cast<int>(c + 1) : -1;

      if (data.bc_left == DDMCBoundaryType::Internal && c > 0) {
        const double dx_left = std::max(dx[static_cast<std::size_t>(c - 1)], 1.0e-30);
        const double sigma_plus = face_sigma_plus_[face_index(c, g)];
        const double sigma_minus = face_sigma_minus_[face_index(c, g)];
        const double denom = 3.0 * dx_i * (sigma_plus * dx_i + sigma_minus * dx_left);
        if (denom > 0.0) {
          data.sigma_leak_left = 2.0 / denom;
        }
      } else if (data.bc_left == DDMCBoundaryType::Vacuum) {
        const double sigma_tr = clamp_sigma(sigma_R_center[cell_index(c, g)]);
        const double D = compute_diffusion_coeff(sigma_tr);
        data.sigma_leak_left = compute_vacuum_leak(D, dx_i, sigma_tr);
      } else if (data.bc_left == DDMCBoundaryType::Interface) {
        // DDMC-side face Rosseland opacity at interface face `c` (between c-1 and c).
        // When no explicit provider is passed, face_sigma_plus_ is still populated from
        // cell-centered sigma_R_center on the DDMC side.
        const double sigma_tr = clamp_sigma(face_sigma_plus_[face_index(c, g)]);
        data.sigma_leak_left = compute_interface_leak(sigma_tr, dx_i);
      }

      if (data.bc_right == DDMCBoundaryType::Internal && c + 1 < n_cells_) {
        const double dx_right = std::max(dx[static_cast<std::size_t>(c + 1)], 1.0e-30);
        const double sigma_minus = face_sigma_minus_[face_index(c + 1, g)];
        const double sigma_plus = face_sigma_plus_[face_index(c + 1, g)];
        const double denom = 3.0 * dx_i * (sigma_minus * dx_i + sigma_plus * dx_right);
        if (denom > 0.0) {
          data.sigma_leak_right = 2.0 / denom;
        }
      } else if (data.bc_right == DDMCBoundaryType::Vacuum) {
        const double sigma_tr = clamp_sigma(sigma_R_center[cell_index(c, g)]);
        const double D = compute_diffusion_coeff(sigma_tr);
        data.sigma_leak_right = compute_vacuum_leak(D, dx_i, sigma_tr);
      } else if (data.bc_right == DDMCBoundaryType::Interface) {
        // DDMC-side face Rosseland opacity at interface face `c+1` (between c and c+1).
        // When no explicit provider is passed, face_sigma_minus_ is still populated from
        // cell-centered sigma_R_center on the DDMC side.
        const double sigma_tr = clamp_sigma(face_sigma_minus_[face_index(c + 1, g)]);
        data.sigma_leak_right = compute_interface_leak(sigma_tr, dx_i);
      }

      if (!m_matrix_check) {
        if (data.sigma_leak_left < 0.0) {
          warn_negative_leak_coeff(c, g, "left", data.sigma_leak_left);
        }
        if (data.sigma_leak_right < 0.0) {
          warn_negative_leak_coeff(c, g, "right", data.sigma_leak_right);
        }
        data.sigma_leak_left = std::max(data.sigma_leak_left, 0.0);
        data.sigma_leak_right = std::max(data.sigma_leak_right, 0.0);
      }
      data.sigma_leak_face[0] = data.sigma_leak_left;
      data.sigma_leak_face[1] = data.sigma_leak_right;

      if (data.bc_left == DDMCBoundaryType::Internal ||
          data.bc_left == DDMCBoundaryType::Interface) {
        data.sigma_leak_out += data.sigma_leak_left;
      } else if (data.bc_left == DDMCBoundaryType::Vacuum) {
        data.sigma_leak_bnd += data.sigma_leak_left;
      }

      if (data.bc_right == DDMCBoundaryType::Internal ||
          data.bc_right == DDMCBoundaryType::Interface) {
        data.sigma_leak_out += data.sigma_leak_right;
      } else if (data.bc_right == DDMCBoundaryType::Vacuum) {
        data.sigma_leak_bnd += data.sigma_leak_right;
      }

      sync_legacy_aliases(data);
      cell_data_[cg] = data;
    }
  }
}

const CellDDMCData& DDMCCoefficients::get_cell_data(const std::int64_t cell,
                                                    const int group) const {
  static const CellDDMCData kEmpty{};
  if (cell < 0 || cell >= n_cells_ || group < 0 || group >= n_groups_) {
    return kEmpty;
  }
  return cell_data_[cell_index(cell, group)];
}

double DDMCCoefficients::get_face_sigma_minus(const std::int64_t face,
                                              const int group) const {
  if (face < 0 || face > n_cells_ || group < 0 || group >= n_groups_) {
    return 0.0;
  }
  return face_sigma_minus_[face_index(face, group)];
}

double DDMCCoefficients::get_face_sigma_plus(const std::int64_t face,
                                             const int group) const {
  if (face < 0 || face > n_cells_ || group < 0 || group >= n_groups_) {
    return 0.0;
  }
  return face_sigma_plus_[face_index(face, group)];
}

double DDMCCoefficients::compute_vacuum_leak(const double D,
                                             const double dx,
                                             const double sigma_tr) const {
  if (dx <= 0.0 || sigma_tr <= 0.0) {
    return 0.0;
  }
  const double d_ext = kMilneLambda / sigma_tr;
  const double denom = dx * (dx + d_ext);
  if (denom <= 0.0) {
    return 0.0;
  }
  return D / denom;
}

double DDMCCoefficients::compute_interface_leak(const double sigma_R,
                                                const double dx) const {
  if (dx <= 0.0) {
    return 0.0;
  }
  const double denom = 3.0 * clamp_sigma(sigma_R) * dx + 6.0 * kMilneLambda;
  if (denom <= 0.0) {
    return 0.0;
  }
  return (1.0 / dx) * (2.0 / denom);
}

void DDMCCoefficients::sync_legacy_aliases(CellDDMCData& data) const {
  data.sigma_leak_left = data.sigma_leak_face[0];
  data.sigma_leak_right = data.sigma_leak_face[1];
  data.bc_left = data.bc_face[0];
  data.bc_right = data.bc_face[1];
}

void DDMCCoefficients::compute_2d(const std::vector<double>& node_r,
                                  const std::vector<double>& node_z,
                                  const int nr,
                                  const int nz,
                                  const std::vector<double>& cell_vol,
                                  const std::vector<double>& rho,
                                  const std::vector<double>& Te,
                                  const std::vector<double>& sigma_R_center,
                                  const ModeSelector& mode_selector,
                                  const DDMCBoundaryType bc_inner_r,
                                  const DDMCBoundaryType bc_outer_r,
                                  const DDMCBoundaryType bc_bottom_z,
                                  const DDMCBoundaryType bc_top_z,
                                  const bool m_matrix_check,
                                  const OpacityProvider* opacity_provider) {
  if (nr <= 0 || nz <= 0) {
    return;
  }
  if (static_cast<std::int64_t>(nr) * static_cast<std::int64_t>(nz) != n_cells_) {
    return;
  }
  if (node_r.size() != node_z.size()) {
    return;
  }

  const int n_nodes_expected = (nr + 1) * (nz + 1);
  if (static_cast<int>(node_r.size()) != n_nodes_expected) {
    return;
  }

  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::int64_t c = static_cast<std::int64_t>(cell_index_2d_rz(i, j, nz));
      const std::size_t c_us = static_cast<std::size_t>(c);
      const int n00 = node_index_2d_rz(i, j, nz);
      const int n10 = node_index_2d_rz(i + 1, j, nz);
      const int n11 = node_index_2d_rz(i + 1, j + 1, nz);
      const int n01 = node_index_2d_rz(i, j + 1, nz);

      const double r0 = node_r[static_cast<std::size_t>(n00)];
      const double z0 = node_z[static_cast<std::size_t>(n00)];
      const double r1 = node_r[static_cast<std::size_t>(n10)];
      const double z1 = node_z[static_cast<std::size_t>(n10)];
      const double r2 = node_r[static_cast<std::size_t>(n11)];
      const double z2 = node_z[static_cast<std::size_t>(n11)];
      const double r3 = node_r[static_cast<std::size_t>(n01)];
      const double z3 = node_z[static_cast<std::size_t>(n01)];

      const double r_max = std::max(std::max(r0, r1), std::max(r2, r3));
      const double V_i = std::max(cell_vol[c_us], 0.0);

      for (int g = 0; g < n_groups_; ++g) {
        const std::size_t cg = cell_index(c, g);
        CellDDMCData data{};

        // Geometry and topology metadata are filled for all cells to support diagnostics.
        for (int face = 0; face < CellDDMCData::kFaceCount; ++face) {
          int neighbor = -1;
          DDMCBoundaryType bc = DDMCBoundaryType::Internal;
          if (face == 0) {
            if (i == 0) {
              bc = bc_inner_r;
            } else {
              neighbor = cell_index_2d_rz(i - 1, j, nz);
            }
          } else if (face == 1) {
            if (i == nr - 1) {
              bc = bc_outer_r;
            } else {
              neighbor = cell_index_2d_rz(i + 1, j, nz);
            }
          } else if (face == 2) {
            if (j == 0) {
              bc = bc_bottom_z;
            } else {
              neighbor = cell_index_2d_rz(i, j - 1, nz);
            }
          } else {
            if (j == nz - 1) {
              bc = bc_top_z;
            } else {
              neighbor = cell_index_2d_rz(i, j + 1, nz);
            }
          }
          data.neighbor_face[face] = neighbor;
          data.bc_face[face] = bc;

          double ra = 0.0;
          double za = 0.0;
          double rb = 0.0;
          double zb = 0.0;
          face_endpoints_2d_rz(face, r0, z0, r1, z1, r2, z2, r3, z3, &ra, &za, &rb, &zb);
          const double L_m = std::hypot(rb - ra, zb - za);
          const double R_bar = 0.5 * (ra + rb);

          bool axis_face = false;
          if (face == 0 && i == 0 &&
              std::abs(R_bar) <= core::axis_tolerance(r_max)) {
            axis_face = true;
            data.A_face[face] = 0.0;
          } else {
            data.A_face[face] = kTwoPi * std::max(R_bar, 0.0) * std::max(L_m, 0.0);
          }

          if (V_i > 0.0 && L_m > 0.0) {
            if (axis_face && r_max > 0.0) {
              data.delta_x_face[face] = V_i / std::max(kPi * r_max * L_m, 1.0e-30);
            } else if (data.A_face[face] > 0.0) {
              data.delta_x_face[face] = V_i / data.A_face[face];
            }
          }
        }

        if (mode_selector.get_mode(c, g) != TransportMode::DDMC) {
          sync_legacy_aliases(data);
          cell_data_[cg] = data;
          continue;
        }

        const double sigma_center = clamp_sigma(sigma_R_center[cg]);
        for (int face = 0; face < CellDDMCData::kFaceCount; ++face) {
          DDMCBoundaryType bc = data.bc_face[face];
          const int neighbor = data.neighbor_face[face];
          if (neighbor >= 0) {
            if (mode_selector.get_mode(neighbor, g) == TransportMode::DDMC) {
              bc = DDMCBoundaryType::Internal;
            } else {
              bc = DDMCBoundaryType::Interface;
            }
          }
          data.bc_face[face] = bc;

          if (bc == DDMCBoundaryType::Reflective || data.A_face[face] <= 0.0) {
            data.sigma_leak_face[face] = 0.0;
            continue;
          }

          const double dx_m = std::max(data.delta_x_face[face], 0.0);
          if (dx_m <= 0.0 || V_i <= 0.0) {
            data.sigma_leak_face[face] = 0.0;
            continue;
          }

          double sigma_leak = 0.0;
          if (bc == DDMCBoundaryType::Internal && neighbor >= 0) {
            const std::size_t neighbor_us = static_cast<std::size_t>(neighbor);
            const double V_j = std::max(cell_vol[neighbor_us], 0.0);
            const double dx_neighbor = V_j / data.A_face[face];
            if (dx_neighbor > 0.0) {
              double sigma_face_i = sigma_center;
              double sigma_face_j = clamp_sigma(sigma_R_center[cell_index(neighbor, g)]);
              if (opacity_provider != nullptr) {
                const double Te_face =
                    compute_face_temperature(Te[c_us], Te[neighbor_us]);
                const double sigma_i_face = opacity_provider->sigma_rosseland(
                    static_cast<int>(c), g, rho[c_us], Te_face);
                const double sigma_j_face = opacity_provider->sigma_rosseland(
                    neighbor, g, rho[neighbor_us], Te_face);
                sigma_face_i = clamp_sigma(sigma_i_face);
                sigma_face_j = clamp_sigma(sigma_j_face);
              }
              const double denom =
                  3.0 * V_i * (sigma_face_i * dx_m + sigma_face_j * dx_neighbor);
              if (denom > 0.0) {
                sigma_leak = (2.0 * data.A_face[face]) / denom;
              }
            }
          } else if (bc == DDMCBoundaryType::Vacuum) {
            double sigma_tr = sigma_center;
            if (opacity_provider != nullptr) {
              const double sigma_i_face =
                  opacity_provider->sigma_rosseland(static_cast<int>(c), g, rho[c_us], Te[c_us]);
              sigma_tr = clamp_sigma(sigma_i_face);
            }
            const double D = 1.0 / (3.0 * sigma_tr);
            const double d_ext = kMilneLambda / sigma_tr;
            const double denom = V_i * (dx_m + d_ext);
            if (denom > 0.0) {
              sigma_leak = data.A_face[face] * D / denom;
            }
          } else if (bc == DDMCBoundaryType::Interface) {
            double sigma_tr = sigma_center;
            if (opacity_provider != nullptr) {
              double Te_face = Te[c_us];
              if (neighbor >= 0) {
                Te_face =
                    compute_face_temperature(Te[c_us], Te[static_cast<std::size_t>(neighbor)]);
              }
              const double sigma_i_face =
                  opacity_provider->sigma_rosseland(static_cast<int>(c), g, rho[c_us], Te_face);
              sigma_tr = clamp_sigma(sigma_i_face);
            }
            const double denom = V_i * (3.0 * sigma_tr * dx_m + 6.0 * kMilneLambda);
            if (denom > 0.0) {
              sigma_leak = (2.0 * data.A_face[face]) / denom;
            }
          }
          data.sigma_leak_face[face] = sigma_leak;
          if (!m_matrix_check) {
            if (data.sigma_leak_face[face] < 0.0) {
              warn_negative_leak_coeff(c,
                                       g,
                                       face_name_2d(face),
                                       data.sigma_leak_face[face]);
            }
            data.sigma_leak_face[face] = std::max(data.sigma_leak_face[face], 0.0);
          }
        }

        for (int face = 0; face < CellDDMCData::kFaceCount; ++face) {
          const double leak = std::max(data.sigma_leak_face[face], 0.0);
          if (data.bc_face[face] == DDMCBoundaryType::Internal ||
              data.bc_face[face] == DDMCBoundaryType::Interface) {
            data.sigma_leak_out += leak;
          } else if (data.bc_face[face] == DDMCBoundaryType::Vacuum) {
            data.sigma_leak_bnd += leak;
          }
        }
        sync_legacy_aliases(data);
        cell_data_[cg] = data;
      }
    }
  }
}

}  // namespace tenryu::radiation
