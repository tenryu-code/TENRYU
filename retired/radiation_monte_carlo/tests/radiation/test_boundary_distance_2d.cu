#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/boundary_distance_2d.cuh"

namespace {

std::vector<double> make_uniform_nodes(const int n,
                                       const double x_min,
                                       const double x_max) {
  std::vector<double> x(static_cast<std::size_t>(n + 1), 0.0);
  const double dx = (x_max - x_min) / static_cast<double>(n);
  for (int i = 0; i <= n; ++i) {
    x[static_cast<std::size_t>(i)] = x_min + static_cast<double>(i) * dx;
  }
  return x;
}

void build_rect_nodes(const int nr,
                      const int nz,
                      const double r_min,
                      const double r_max,
                      const double z_min,
                      const double z_max,
                      std::vector<double>* node_r,
                      std::vector<double>* node_z) {
  const auto r = make_uniform_nodes(nr, r_min, r_max);
  const auto z = make_uniform_nodes(nz, z_min, z_max);

  node_r->assign(static_cast<std::size_t>((nr + 1) * (nz + 1)), 0.0);
  node_z->assign(static_cast<std::size_t>((nr + 1) * (nz + 1)), 0.0);
  for (int i = 0; i <= nr; ++i) {
    for (int j = 0; j <= nz; ++j) {
      const int n = i * (nz + 1) + j;
      (*node_r)[static_cast<std::size_t>(n)] = r[static_cast<std::size_t>(i)];
      (*node_z)[static_cast<std::size_t>(n)] = z[static_cast<std::size_t>(j)];
    }
  }
}

}  // namespace

TEST_CASE("boundary_distance_2d selects right boundary and neighbor correctly",
          "[radiation][imc][2d]") {
  std::vector<double> node_r;
  std::vector<double> node_z;
  build_rect_nodes(2, 1, 0.0, 2.0, 0.0, 1.0, &node_r, &node_z);

  const auto hit_neighbor = tenryu::radiation::boundary_distance_2d_rz(
      1.5, 0.5, -1.0, 0.0, 0.0, 1, 2, 1, node_r.data(), node_z.data());
  REQUIRE(hit_neighbor.face == 0);
  REQUIRE_FALSE(hit_neighbor.is_boundary);
  REQUIRE(hit_neighbor.neighbor == 0);
  REQUIRE(hit_neighbor.s == Catch::Approx(0.5).epsilon(1.0e-12));

  const auto hit_outer = tenryu::radiation::boundary_distance_2d_rz(
      1.5, 0.5, +1.0, 0.0, 0.0, 1, 2, 1, node_r.data(), node_z.data());
  REQUIRE(hit_outer.face == 1);
  REQUIRE(hit_outer.is_boundary);
  REQUIRE(hit_outer.neighbor == -1);
  REQUIRE(hit_outer.s == Catch::Approx(0.5).epsilon(1.0e-12));
}

TEST_CASE("boundary_distance_2d chooses correct face for diagonal ray",
          "[radiation][imc][2d]") {
  std::vector<double> node_r;
  std::vector<double> node_z;
  build_rect_nodes(1, 1, 0.0, 1.0, 0.0, 1.0, &node_r, &node_z);

  const auto hit = tenryu::radiation::boundary_distance_2d_rz(
      0.6, 0.4, 0.2, 0.8, 0.0, 0, 1, 1, node_r.data(), node_z.data());
  REQUIRE(hit.face == 3);
  REQUIRE(hit.is_boundary);
  REQUIRE(hit.s == Catch::Approx(0.75).epsilon(1.0e-12));
}

TEST_CASE("boundary_distance_2d is stable for grazing incidence",
          "[radiation][imc][2d]") {
  std::vector<double> node_r;
  std::vector<double> node_z;
  build_rect_nodes(1, 1, 0.0, 1.0, 0.0, 1.0, &node_r, &node_z);

  const auto hit = tenryu::radiation::boundary_distance_2d_rz(
      0.25, 0.5, 1.0, 1.0e-15, 0.0, 0, 1, 1, node_r.data(), node_z.data());
  REQUIRE(std::isfinite(hit.s));
  REQUIRE(hit.s > 0.0);
  REQUIRE(hit.face == 1);
}

TEST_CASE("boundary_distance_2d handles axis contact and phi-driven curvature",
          "[radiation][imc][2d]") {
  std::vector<double> node_r;
  std::vector<double> node_z;
  build_rect_nodes(1, 1, 0.0, 1.0, 0.0, 1.0, &node_r, &node_z);

  const auto hit_axis = tenryu::radiation::boundary_distance_2d_rz(
      0.2, 0.5, -1.0, 0.0, 0.0, 0, 1, 1, node_r.data(), node_z.data());
  REQUIRE(hit_axis.face == 0);
  REQUIRE(hit_axis.is_boundary);
  REQUIRE(hit_axis.s == Catch::Approx(0.2).epsilon(1.0e-12));

  const auto hit_phi = tenryu::radiation::boundary_distance_2d_rz(
      0.5, 0.5, 0.0, 0.0, 1.0, 0, 1, 1, node_r.data(), node_z.data());
  REQUIRE(hit_phi.face == 1);
  REQUIRE(hit_phi.is_boundary);
  REQUIRE(hit_phi.s == Catch::Approx(std::sqrt(0.75)).epsilon(1.0e-12));
}
