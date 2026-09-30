#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/ddmc_coefficients.hpp"
#include "radiation/mmatrix_check.hpp"
#include "radiation/mode_selector.hpp"

using tenryu::radiation::DDMCBoundaryType;
using tenryu::radiation::DDMCCoefficients;
using tenryu::radiation::ModeSelector;
using tenryu::radiation::ModeSelectorConfig;
using tenryu::radiation::TransportMode;

namespace {

std::vector<double> make_node_field_2d(const std::vector<double>& r_nodes,
                                       const std::vector<double>& z_nodes,
                                       const bool is_r) {
  const int nr = static_cast<int>(r_nodes.size()) - 1;
  const int nz = static_cast<int>(z_nodes.size()) - 1;
  std::vector<double> field(static_cast<std::size_t>((nr + 1) * (nz + 1)), 0.0);
  for (int i = 0; i <= nr; ++i) {
    for (int j = 0; j <= nz; ++j) {
      const int n = i * (nz + 1) + j;
      field[static_cast<std::size_t>(n)] = is_r ? r_nodes[static_cast<std::size_t>(i)]
                                                 : z_nodes[static_cast<std::size_t>(j)];
    }
  }
  return field;
}

std::vector<double> make_cell_vol_2d(const std::vector<double>& r_nodes,
                                     const std::vector<double>& z_nodes) {
  constexpr double kPi = 3.14159265358979323846;
  const int nr = static_cast<int>(r_nodes.size()) - 1;
  const int nz = static_cast<int>(z_nodes.size()) - 1;
  std::vector<double> vol(static_cast<std::size_t>(nr * nz), 0.0);
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const double r_lo = r_nodes[static_cast<std::size_t>(i)];
      const double r_hi = r_nodes[static_cast<std::size_t>(i + 1)];
      const double z_lo = z_nodes[static_cast<std::size_t>(j)];
      const double z_hi = z_nodes[static_cast<std::size_t>(j + 1)];
      const int c = i * nz + j;
      vol[static_cast<std::size_t>(c)] =
          kPi * (r_hi * r_hi - r_lo * r_lo) * (z_hi - z_lo);
    }
  }
  return vol;
}

}  // namespace

TEST_CASE("DDMC coefficients 2D_RZ face geometry and normalization",
          "[radiation][ddmc][coefficients][2d]") {
  constexpr int nr = 2;
  constexpr int nz = 2;
  constexpr int n_cells = nr * nz;
  constexpr int n_groups = 1;
  constexpr double kPi = 3.14159265358979323846;

  const std::vector<double> r_nodes = {0.0, 0.5, 1.0};
  const std::vector<double> z_nodes = {0.0, 1.0, 2.0};
  const auto node_r = make_node_field_2d(r_nodes, z_nodes, true);
  const auto node_z = make_node_field_2d(r_nodes, z_nodes, false);
  const auto cell_vol = make_cell_vol_2d(r_nodes, z_nodes);

  const std::vector<double> rho(n_cells, 1.0);
  const std::vector<double> Te(n_cells, 10.0);
  const std::vector<double> sigma_R(n_cells, 50.0);
  const std::vector<double> sigma_a = sigma_R;
  const std::vector<double> fleck_f(n_cells, 0.05);

  ModeSelectorConfig mode_cfg{};
  mode_cfg.tau_ddmc = 3.0;
  mode_cfg.omega_ddmc = 0.9;
  mode_cfg.sigma_floor = 1.0e-20;
  ModeSelector mode(n_cells, n_groups, mode_cfg);
  mode.compute_modes_2d_rz(node_r, node_z, nr, nz, sigma_R, fleck_f, sigma_a);
  for (int c = 0; c < n_cells; ++c) {
    REQUIRE(mode.get_mode(c, 0) == TransportMode::DDMC);
  }

  DDMCCoefficients coeff(n_cells, n_groups, 1.0e-20);
  coeff.compute_2d(node_r,
                   node_z,
                   nr,
                   nz,
                   cell_vol,
                   rho,
                   Te,
                   sigma_R,
                   mode,
                   DDMCBoundaryType::Reflective,
                   DDMCBoundaryType::Reflective,
                   DDMCBoundaryType::Vacuum,
                   DDMCBoundaryType::Vacuum,
                   true,
                   nullptr);

  // Axis-contact cell: face 0 area must vanish and no axis leakage.
  const auto& axis_cell = coeff.get_cell_data(0, 0);  // i=0, j=0
  REQUIRE(axis_cell.A_face[0] == Catch::Approx(0.0).margin(1.0e-30));
  REQUIRE(axis_cell.sigma_leak_face[0] == Catch::Approx(0.0).margin(1.0e-30));
  REQUIRE(axis_cell.delta_x_face[0] > 0.0);  // axis-safe surrogate width

  // Non-axis face area check: cell(i=1,j=0), left face at R=0.5, L=1 => A=pi.
  const int c_non_axis = 1 * nz + 0;
  const auto& non_axis = coeff.get_cell_data(c_non_axis, 0);
  REQUIRE(non_axis.A_face[0] == Catch::Approx(kPi).epsilon(1.0e-12));

  // Probability normalization over absorb + four face leaks.
  std::vector<double> sigma_a_eff(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    sigma_a_eff[static_cast<std::size_t>(c)] =
        fleck_f[static_cast<std::size_t>(c)] * sigma_a[static_cast<std::size_t>(c)];
  }

  for (int c = 0; c < n_cells; ++c) {
    const auto& cell = coeff.get_cell_data(c, 0);
    const double sigma_abs = sigma_a_eff[static_cast<std::size_t>(c)];
    double sigma_tot = sigma_abs;
    for (int face = 0; face < 4; ++face) {
      sigma_tot += std::max(cell.sigma_leak_face[face], 0.0);
    }
    REQUIRE(sigma_tot > 0.0);

    double prob_sum = sigma_abs / sigma_tot;
    for (int face = 0; face < 4; ++face) {
      prob_sum += std::max(cell.sigma_leak_face[face], 0.0) / sigma_tot;
    }
    REQUIRE(prob_sum == Catch::Approx(1.0).epsilon(1.0e-13));
  }

  auto mode_for_mmatrix = mode;
  const auto mmatrix =
      tenryu::radiation::check_mmatrix_condition(coeff, mode_for_mmatrix, sigma_a_eff);
  REQUIRE(mmatrix.total_violations == 0);
}

