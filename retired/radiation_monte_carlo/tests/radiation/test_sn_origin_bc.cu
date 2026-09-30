#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "radiation/sn_transport_1d.hpp"

namespace {

std::vector<double> spherical_nodes(const int n_cells, const double dr) {
  std::vector<double> node(static_cast<std::size_t>(n_cells + 1), 0.0);
  for (int i = 0; i <= n_cells; ++i) {
    node[static_cast<std::size_t>(i)] = dr * static_cast<double>(i);
  }
  return node;
}

std::vector<double> spherical_volumes(const std::vector<double>& node) {
  constexpr double four_pi_over_three = 4.18879020478639098462;
  std::vector<double> vol(node.size() - 1U, 0.0);
  for (std::size_t i = 0; i < vol.size(); ++i) {
    vol[i] = four_pi_over_three *
             (node[i + 1U] * node[i + 1U] * node[i + 1U] -
              node[i] * node[i] * node[i]);
  }
  return vol;
}

}  // namespace

TEST_CASE("S_N origin parity flag preserves investigated legacy sweep",
          "[radiation][sn][origin]") {
  constexpr int n_cells = 4;
  constexpr int n_groups = 1;
  const std::vector<double> node = spherical_nodes(n_cells, 0.25);
  const std::vector<double> vol = spherical_volumes(node);
  std::vector<double> sigma_a(static_cast<std::size_t>(n_cells), 0.2);
  std::vector<double> sigma_s(static_cast<std::size_t>(n_cells), 0.0);
  std::vector<double> source(static_cast<std::size_t>(n_cells), 1.0e12);

  tenryu::radiation::SNTransport1DConfig legacy_cfg{};
  legacy_cfg.n_angles = 8;
  legacy_cfg.max_iterations = 100;
  legacy_cfg.convergence_tol = 1.0e-13;
  legacy_cfg.origin_parity_only = false;

  tenryu::radiation::SNTransport1DConfig parity_cfg = legacy_cfg;
  parity_cfg.origin_parity_only = true;

  const auto legacy = tenryu::radiation::solve_sn_transport_1d(
      sigma_a.data(), sigma_s.data(), source.data(), node.data(), vol.data(),
      n_cells, n_groups, legacy_cfg);
  const auto parity = tenryu::radiation::solve_sn_transport_1d(
      sigma_a.data(), sigma_s.data(), source.data(), node.data(), vol.data(),
      n_cells, n_groups, parity_cfg);

  REQUIRE(legacy.converged);
  REQUIRE(parity.converged);
  REQUIRE(legacy.E_sn.size() == parity.E_sn.size());
  for (std::size_t i = 0; i < legacy.E_sn.size(); ++i) {
    CHECK(parity.E_sn[i] == legacy.E_sn[i]);
  }
}
