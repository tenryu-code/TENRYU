#include <math.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "radiation/boundary_distance_2d.cuh"
#include "radiation/face_geometry_2d.cuh"

using namespace tenryu::radiation;

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTightTol = 1.0e-14;

struct QuadCell {
  double r00 = 0.0;
  double z00 = 0.0;
  double r10 = 0.0;
  double z10 = 0.0;
  double r11 = 0.0;
  double z11 = 0.0;
  double r01 = 0.0;
  double z01 = 0.0;
};

struct Direction3 {
  double r = 0.0;
  double z = 0.0;
  double phi = 0.0;
};

[[nodiscard]] FaceGeom2D face_geom(const QuadCell& cell, const int face_id) {
  return compute_face_geom(face_id,
                           cell.r00,
                           cell.z00,
                           cell.r10,
                           cell.z10,
                           cell.r11,
                           cell.z11,
                           cell.r01,
                           cell.z01);
}

[[nodiscard]] double norm2(const double x, const double y) {
  return std::sqrt(x * x + y * y);
}

[[nodiscard]] double norm3(const double x, const double y, const double z) {
  return std::sqrt(x * x + y * y + z * z);
}

void require_unit_normal_orthogonal_tangent(const FaceGeom2D& geom,
                                            const double tol) {
  REQUIRE(norm2(geom.nr, geom.nz) == Catch::Approx(1.0).margin(tol));
  REQUIRE(norm2(geom.tr, geom.tz) == Catch::Approx(1.0).margin(tol));
  REQUIRE(geom.nr * geom.tr + geom.nz * geom.tz == Catch::Approx(0.0).margin(tol));
}

[[nodiscard]] std::array<double, 2> centroid(const QuadCell& cell) {
  return {
      0.25 * (cell.r00 + cell.r10 + cell.r11 + cell.r01),
      0.25 * (cell.z00 + cell.z10 + cell.z11 + cell.z01),
  };
}

void require_outward_normals(const QuadCell& cell) {
  const auto c = centroid(cell);
  for (int face_id = 0; face_id < 4; ++face_id) {
    const FaceGeom2D geom = face_geom(cell, face_id);
    require_unit_normal_orthogonal_tangent(geom, kTightTol);
    const double z_mid = 0.5 * (geom.z1 + geom.z2);
    const double dot = (geom.r_mid - c[0]) * geom.nr + (z_mid - c[1]) * geom.nz;
    REQUIRE(dot > 0.0);
  }
}

void build_single_cell_nodes(const QuadCell& cell,
                             std::array<double, 4>* node_r,
                             std::array<double, 4>* node_z) {
  // node index = i * (nz + 1) + j with nr=nz=1
  (*node_r)[0] = cell.r00;  // (i=0,j=0)
  (*node_z)[0] = cell.z00;
  (*node_r)[1] = cell.r01;  // (i=0,j=1)
  (*node_z)[1] = cell.z01;
  (*node_r)[2] = cell.r10;  // (i=1,j=0)
  (*node_z)[2] = cell.z10;
  (*node_r)[3] = cell.r11;  // (i=1,j=1)
  (*node_z)[3] = cell.z11;
}

[[nodiscard]] double analytic_intersection_s_horizontal(const QuadCell& cell,
                                                        const double r0,
                                                        const double z0) {
  const double denom = cell.z11 - cell.z10;
  REQUIRE(std::abs(denom) > 0.0);
  const double t = (z0 - cell.z10) / denom;
  REQUIRE(t >= 0.0);
  REQUIRE(t <= 1.0);
  const double r_face = cell.r10 + t * (cell.r11 - cell.r10);
  return r_face - r0;
}

[[nodiscard]] Direction3 sample_ddmc_to_imc_direction_cosine(std::mt19937* rng,
                                                             const double nr,
                                                             const double nz,
                                                             const double tr,
                                                             const double tz) {
  std::uniform_real_distribution<double> uniform01(0.0, 1.0);
  const double xi_mu = std::max(uniform01(*rng), 1.0e-16);
  const double xi_phi = std::min(std::max(uniform01(*rng), 0.0), 1.0 - 1.0e-16);

  const double mu = std::sqrt(std::clamp(xi_mu, 0.0, 1.0));
  const double phi = 2.0 * kPi * xi_phi;
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  const double tangent = sin_theta * std::cos(phi);

  Direction3 dir{};
  dir.r = mu * nr + tangent * tr;
  dir.z = mu * nz + tangent * tz;
  dir.phi = sin_theta * std::sin(phi);
  return dir;
}

[[nodiscard]] double ks_cosine_mu_distribution(std::vector<double> mu_samples) {
  std::sort(mu_samples.begin(), mu_samples.end());
  const double n = static_cast<double>(mu_samples.size());
  double d_max = 0.0;
  for (std::size_t i = 0; i < mu_samples.size(); ++i) {
    const double x = mu_samples[i];
    const double f = std::clamp(x * x, 0.0, 1.0);  // CDF for p(mu)=2mu on [0,1]
    const double fn_lo = static_cast<double>(i) / n;
    const double fn_hi = static_cast<double>(i + 1) / n;
    d_max = std::max(d_max, std::abs(f - fn_lo));
    d_max = std::max(d_max, std::abs(fn_hi - f));
  }
  return d_max;
}

}  // namespace

TEST_CASE("face_geometry_2d_normal", "[radiation][face_geometry_2d]") {
  SECTION("rectangular cell") {
    const QuadCell cell{
        1.0, 0.0,  // v0
        3.0, 0.0,  // v1
        3.0, 2.0,  // v2
        1.0, 2.0,  // v3
    };

    const FaceGeom2D f0 = face_geom(cell, 0);
    const FaceGeom2D f1 = face_geom(cell, 1);
    const FaceGeom2D f2 = face_geom(cell, 2);
    const FaceGeom2D f3 = face_geom(cell, 3);

    REQUIRE(f0.r1 == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f0.z1 == Catch::Approx(2.0).margin(kTightTol));
    REQUIRE(f0.r2 == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f0.z2 == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f0.nr == Catch::Approx(-1.0).margin(kTightTol));
    REQUIRE(f0.nz == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f0.tr == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f0.tz == Catch::Approx(-1.0).margin(kTightTol));
    REQUIRE(f0.length == Catch::Approx(2.0).margin(kTightTol));
    require_unit_normal_orthogonal_tangent(f0, kTightTol);

    REQUIRE(f1.r1 == Catch::Approx(3.0).margin(kTightTol));
    REQUIRE(f1.z1 == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f1.r2 == Catch::Approx(3.0).margin(kTightTol));
    REQUIRE(f1.z2 == Catch::Approx(2.0).margin(kTightTol));
    REQUIRE(f1.nr == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f1.nz == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f1.tr == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f1.tz == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f1.length == Catch::Approx(2.0).margin(kTightTol));
    require_unit_normal_orthogonal_tangent(f1, kTightTol);

    REQUIRE(f2.r1 == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f2.z1 == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f2.r2 == Catch::Approx(3.0).margin(kTightTol));
    REQUIRE(f2.z2 == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f2.nr == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f2.nz == Catch::Approx(-1.0).margin(kTightTol));
    REQUIRE(f2.tr == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f2.tz == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f2.length == Catch::Approx(2.0).margin(kTightTol));
    require_unit_normal_orthogonal_tangent(f2, kTightTol);

    REQUIRE(f3.r1 == Catch::Approx(3.0).margin(kTightTol));
    REQUIRE(f3.z1 == Catch::Approx(2.0).margin(kTightTol));
    REQUIRE(f3.r2 == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f3.z2 == Catch::Approx(2.0).margin(kTightTol));
    REQUIRE(f3.nr == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f3.nz == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(f3.tr == Catch::Approx(-1.0).margin(kTightTol));
    REQUIRE(f3.tz == Catch::Approx(0.0).margin(kTightTol));
    REQUIRE(f3.length == Catch::Approx(2.0).margin(kTightTol));
    require_unit_normal_orthogonal_tangent(f3, kTightTol);
  }

  SECTION("trapezoid cell") {
    const QuadCell cell{
        1.0, 0.0,   // v0
        3.0, 0.0,   // v1
        2.5, 2.0,   // v2
        1.5, 2.0,   // v3
    };
    require_outward_normals(cell);
  }

  SECTION("parallelogram cell") {
    const QuadCell cell{
        1.0, 0.0,  // v0
        3.0, 1.0,  // v1
        2.0, 3.0,  // v2
        0.0, 2.0,  // v3
    };
    require_outward_normals(cell);
  }
}

TEST_CASE("face_geometry_2d_boundary_distance_2d_skew",
          "[radiation][face_geometry_2d]") {
  SECTION("rectangle regression") {
    const QuadCell cell{
        1.0, 0.0,  // v0
        3.0, 0.0,  // v1
        3.0, 2.0,  // v2
        1.0, 2.0,  // v3
    };

    std::array<double, 4> node_r{};
    std::array<double, 4> node_z{};
    build_single_cell_nodes(cell, &node_r, &node_z);

    const BoundaryHit2D hit = boundary_distance_2d_rz(
        2.0, 1.0, 1.0, 0.0, 0.0, 0, 1, 1, node_r.data(), node_z.data());

    REQUIRE(hit.face == 1);
    REQUIRE(hit.s == Catch::Approx(1.0).margin(1.0e-14));
    REQUIRE(hit.face != 2);
  }

  SECTION("trapezoid analytic intersection") {
    const QuadCell cell{
        1.0, 0.0,   // v0
        3.0, 0.0,   // v1
        2.5, 2.0,   // v2
        1.5, 2.0,   // v3
    };

    std::array<double, 4> node_r{};
    std::array<double, 4> node_z{};
    build_single_cell_nodes(cell, &node_r, &node_z);

    const double expected_s = analytic_intersection_s_horizontal(cell, 2.0, 1.0);
    const BoundaryHit2D hit = boundary_distance_2d_rz(
        2.0, 1.0, 1.0, 0.0, 0.0, 0, 1, 1, node_r.data(), node_z.data());

    REQUIRE(hit.face == 1);
    REQUIRE(hit.s == Catch::Approx(expected_s).margin(1.0e-14));
  }

  SECTION("trapezoid various directions") {
    const QuadCell cell{
        1.0, 0.0,   // v0
        3.0, 0.0,   // v1
        2.5, 2.0,   // v2
        1.5, 2.0,   // v3
    };

    std::array<double, 4> node_r{};
    std::array<double, 4> node_z{};
    build_single_cell_nodes(cell, &node_r, &node_z);

    struct RayCase {
      double omega_r;
      double omega_z;
      int expected_face;
    };
    const std::array<RayCase, 3> rays = {{
        {1.0, 1.0, 1},       // diagonal
        {-1.0, 0.1, 0},      // toward left skew edge
        {1.0, 1.0e-12, 1},   // nearly parallel to top/bottom edge
    }};

    for (const RayCase ray : rays) {
      const BoundaryHit2D hit = boundary_distance_2d_rz(
          2.0, 1.0, ray.omega_r, ray.omega_z, 0.0, 0, 1, 1, node_r.data(), node_z.data());
      REQUIRE(hit.s > 0.0);
      REQUIRE(hit.face == ray.expected_face);
    }
  }
}

TEST_CASE("face_geometry_2d_reflect_skew_cell", "[radiation][face_geometry_2d]") {
  SECTION("axis-aligned normal") {
    {
      double dir_r = 0.5;
      double dir_z = 0.5;
      reflect_direction_2d(dir_r, dir_z, 1.0, 0.0);
      REQUIRE(dir_r == Catch::Approx(-0.5).margin(kTightTol));
      REQUIRE(dir_z == Catch::Approx(0.5).margin(kTightTol));
    }
    {
      double dir_r = 0.5;
      double dir_z = 0.5;
      reflect_direction_2d(dir_r, dir_z, 0.0, 1.0);
      REQUIRE(dir_r == Catch::Approx(0.5).margin(kTightTol));
      REQUIRE(dir_z == Catch::Approx(-0.5).margin(kTightTol));
    }
  }

  SECTION("tilted normal") {
    const double nr = std::sqrt(3.0) * 0.5;
    const double nz = 0.5;
    double dir_r = 1.0;
    double dir_z = 0.0;
    const double mu_in = face_mu_2d(dir_r, dir_z, nr, nz);

    const double expected_r = dir_r - 2.0 * mu_in * nr;
    const double expected_z = dir_z - 2.0 * mu_in * nz;

    reflect_direction_2d(dir_r, dir_z, nr, nz);
    const double mu_out = face_mu_2d(dir_r, dir_z, nr, nz);

    REQUIRE(dir_r == Catch::Approx(expected_r).margin(kTightTol));
    REQUIRE(dir_z == Catch::Approx(expected_z).margin(kTightTol));
    REQUIRE(norm2(dir_r, dir_z) == Catch::Approx(1.0).margin(kTightTol));
    REQUIRE(std::abs(mu_in) == Catch::Approx(std::abs(mu_out)).margin(kTightTol));
  }

  SECTION("unit vector preservation for tilted normals") {
    const std::array<std::array<double, 2>, 3> normals = {{
        {std::sqrt(3.0) * 0.5, 0.5},
        {0.6, 0.8},
        {-0.8, 0.6},
    }};
    const std::array<std::array<double, 2>, 5> directions = {{
        {1.0, 0.0},
        {0.0, 1.0},
        {std::sqrt(2.0) * 0.5, std::sqrt(2.0) * 0.5},
        {-0.8, 0.6},
        {0.3, -std::sqrt(1.0 - 0.09)},
    }};

    for (const auto& normal : normals) {
      const double n_norm = norm2(normal[0], normal[1]);
      const double nr = normal[0] / n_norm;
      const double nz = normal[1] / n_norm;
      for (const auto& dir : directions) {
        const double d_norm = norm2(dir[0], dir[1]);
        double dir_r = dir[0] / d_norm;
        double dir_z = dir[1] / d_norm;
        reflect_direction_2d(dir_r, dir_z, nr, nz);
        REQUIRE(norm2(dir_r, dir_z) == Catch::Approx(1.0).margin(kTightTol));
      }
    }
  }
}

TEST_CASE("face_geometry_2d_push_off_face_2d", "[radiation][face_geometry_2d]") {
  SECTION("axis-aligned push") {
    constexpr double eps = 1.0e-12;

    {
      double r = 2.0;
      double z = 1.0;
      push_off_face_2d(r, z, 1.0, 0.0, eps);
      REQUIRE(r == Catch::Approx(2.0 + eps).margin(kTightTol));
      REQUIRE(z == Catch::Approx(1.0).margin(kTightTol));
    }

    {
      double r = 2.0;
      double z = 1.0;
      push_off_face_2d(r, z, 0.0, 1.0, eps);
      REQUIRE(r == Catch::Approx(2.0).margin(kTightTol));
      REQUIRE(z == Catch::Approx(1.0 + eps).margin(kTightTol));
    }
  }

  SECTION("tilted push") {
    constexpr double eps = 1.0e-12;
    double r = 2.0;
    double z = 1.0;
    push_off_face_2d(r, z, 0.6, 0.8, eps);
    REQUIRE(r == Catch::Approx(2.0 + 0.6e-12).margin(kTightTol));
    REQUIRE(z == Catch::Approx(1.0 + 0.8e-12).margin(kTightTol));
  }
}

TEST_CASE("face_geometry_2d_ddmc_to_imc_direction_skew",
          "[radiation][face_geometry_2d]") {
  constexpr int n_samples = 100000;
  const double nr = std::sqrt(3.0) * 0.5;
  const double nz = 0.5;
  const double tr = -nz;
  const double tz = nr;

  std::mt19937 rng(20260224U);
  std::vector<double> mu_samples;
  mu_samples.reserve(static_cast<std::size_t>(n_samples));
  double mu_sum = 0.0;

  for (int i = 0; i < n_samples; ++i) {
    const Direction3 dir = sample_ddmc_to_imc_direction_cosine(&rng, nr, nz, tr, tz);
    const double mu = face_mu_2d(dir.r, dir.z, nr, nz);
    REQUIRE(mu > 0.0);
    REQUIRE(norm3(dir.r, dir.z, dir.phi) == Catch::Approx(1.0).margin(kTightTol));
    mu_sum += mu;
    mu_samples.push_back(mu);
  }

  const double mean_mu = mu_sum / static_cast<double>(n_samples);
  REQUIRE(mean_mu == Catch::Approx(2.0 / 3.0).margin(0.01));

  const double ks = ks_cosine_mu_distribution(mu_samples);
  REQUIRE(ks < 0.01);
}
