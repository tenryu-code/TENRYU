#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "radiation/boundary.cuh"
#include "radiation/difference_residualization.cuh"
#include "radiation/particle_pool.cuh"
#include "radiation/tally.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_tally");
}

}  // namespace

TEST_CASE("Tally zero and finalize", "[radiation][tally]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 1;

  tenryu::core::GroupField1D rad_dep(n_cells * n_groups);
  tenryu::core::GroupField1D rad_E(n_cells * n_groups);
  tenryu::core::CellField1D vol(n_cells);
  vol = std::vector<double>{2.0, 4.0};

  double* d_rad_E_tally = nullptr;
  double* d_E_escape = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally), sizeof(double) * n_cells));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_E_escape), sizeof(double) * n_groups));

  tenryu::radiation::zero_tallies_cuda(rad_dep.data(),
                                       d_rad_E_tally,
                                       d_E_escape,
                                       n_cells,
                                       n_groups);

  std::vector<double> tally = {2.0, 8.0};
  cuda_check(cudaMemcpy(d_rad_E_tally,
                        tally.data(),
                        sizeof(double) * tally.size(),
                        cudaMemcpyHostToDevice));

  const double dt = 1.0e-9;
  tenryu::radiation::tally_finalize_cuda(rad_E.data(),
                                         d_rad_E_tally,
                                         vol.data(),
                                         n_cells,
                                         n_groups,
                                         dt);

  std::vector<double> rad_E_host(n_cells, 0.0);
  rad_E.copy_to_host(rad_E_host.data());

  REQUIRE(rad_E_host[0] == Catch::Approx(2.0 / (2.0 * tenryu::core::constants::c_light * dt))
                               .epsilon(1.0e-12));
  REQUIRE(rad_E_host[1] == Catch::Approx(8.0 / (4.0 * tenryu::core::constants::c_light * dt))
                               .epsilon(1.0e-12));

  cuda_check(cudaFree(d_E_escape));
  cuda_check(cudaFree(d_rad_E_tally));
}

TEST_CASE("Reference absorption preseed adds deterministic deposition",
          "[radiation][tally][difference]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 1;
  const double dt = 2.0e-14;

  tenryu::core::GroupField1D rad_dep(n_cells * n_groups);
  tenryu::core::GroupField1D sigma_a_eff(n_cells * n_groups);
  tenryu::core::GroupField1D E_ref(n_cells * n_groups);
  tenryu::core::CellField1D vol(n_cells);

  rad_dep = std::vector<double>{1.0, -2.0};
  sigma_a_eff = std::vector<double>{2.0, 0.5};
  E_ref = std::vector<double>{3.0, 4.0};
  vol = std::vector<double>{5.0, 7.0};

  tenryu::radiation::preseed_reference_absorption_cuda(rad_dep.data(),
                                                       sigma_a_eff.data(),
                                                       E_ref.data(),
                                                       vol.data(),
                                                       n_cells,
                                                       n_groups,
                                                       dt);

  std::vector<double> dep_host(n_cells, 0.0);
  rad_dep.copy_to_host(dep_host.data());

  const double dep0 = tenryu::core::constants::c_light * 2.0 * 3.0 * 5.0 * dt;
  const double dep1 = tenryu::core::constants::c_light * 0.5 * 4.0 * 7.0 * dt;
  REQUIRE(dep_host[0] == Catch::Approx(1.0 + dep0).epsilon(1.0e-12));
  REQUIRE(dep_host[1] == Catch::Approx(-2.0 + dep1).epsilon(1.0e-12));
}

TEST_CASE("HOLO Prr finalize normalizes moment chi and coverage",
          "[radiation][tally][holo]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 2;
  constexpr int n_total = n_cells * n_groups;
  const double dt = 2.0e-9;

  tenryu::core::GroupField1D Prr(n_total);
  tenryu::core::GroupField1D chi(n_total);
  tenryu::core::GroupField1D coverage(n_total);
  tenryu::core::GroupField1D rad_E_tally(n_total);
  tenryu::core::CellField1D vol(n_cells);

  Prr = std::vector<double>{1.5, 2.0, 5.0, 6.0};
  coverage = std::vector<double>{6.0, 4.0, 7.0, 8.0};
  rad_E_tally = std::vector<double>{6.0, 8.0, 10.0, 12.0};
  vol = std::vector<double>{2.0, 4.0};

  std::vector<std::uint8_t> holo_core = {1U, 0U};
  std::uint8_t* d_holo_core = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_holo_core),
                        sizeof(std::uint8_t) * holo_core.size()));
  cuda_check(cudaMemcpy(d_holo_core,
                        holo_core.data(),
                        sizeof(std::uint8_t) * holo_core.size(),
                        cudaMemcpyHostToDevice));

  tenryu::radiation::holo_prr_finalize_cuda(Prr.data(),
                                            chi.data(),
                                            coverage.data(),
                                            Prr.data(),
                                            coverage.data(),
                                            rad_E_tally.data(),
                                            vol.data(),
                                            d_holo_core,
                                            n_cells,
                                            n_groups,
                                            dt,
                                            1.0e-300);

  std::vector<double> Prr_host(n_total, 0.0);
  std::vector<double> chi_host(n_total, 0.0);
  std::vector<double> coverage_host(n_total, 0.0);
  Prr.copy_to_host(Prr_host.data());
  chi.copy_to_host(chi_host.data());
  coverage.copy_to_host(coverage_host.data());

  const double norm0 = 2.0 * tenryu::core::constants::c_light * dt;
  REQUIRE(Prr_host[0] == Catch::Approx(1.5 / norm0).epsilon(1.0e-12));
  REQUIRE(Prr_host[1] == Catch::Approx(2.0 / norm0).epsilon(1.0e-12));
  REQUIRE(chi_host[0] == Catch::Approx(0.25).epsilon(1.0e-12));
  REQUIRE(chi_host[1] == Catch::Approx(0.25).epsilon(1.0e-12));
  REQUIRE(coverage_host[0] == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(coverage_host[1] == Catch::Approx(0.5).epsilon(1.0e-12));
  REQUIRE(Prr_host[2] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(chi_host[2] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(coverage_host[2] == Catch::Approx(0.0).margin(1.0e-15));
  REQUIRE(Prr_host[3] == Catch::Approx(0.0).margin(1.0e-15));

  cuda_check(cudaFree(d_holo_core));
}

TEST_CASE("HOLO consistency source combines temporal and face defects",
          "[radiation][tally][holo]") {
  constexpr int n_cells = 3;
  constexpr int n_groups = 1;
  constexpr int n_total = n_cells * n_groups;
  constexpr double dt = 2.0;

  tenryu::core::GroupField1D consistency_source(n_total);
  tenryu::core::GroupField1D E_HO(n_total);
  tenryu::core::GroupField1D E_LO_pred(n_total);
  tenryu::core::GroupField1D E_old(n_total);
  tenryu::core::GroupField1D sigma_R(n_total);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::NodeField1D node_r(n_cells + 1);
  tenryu::core::CellField1D lo_weight(n_cells);
  tenryu::core::GroupField1D face_current(n_cells + 1);

  E_HO = std::vector<double>{2.0, 4.0, 8.0};
  E_LO_pred = std::vector<double>{1.0, 3.0, 8.0};
  E_old = std::vector<double>{0.0, 0.0, 0.0};
  sigma_R = std::vector<double>{1.0e300, 1.0e300, 1.0e300};
  vol = std::vector<double>{1.0, 1.0, 1.0};
  node_r = std::vector<double>{0.0, 1.0, 2.0, 3.0};
  lo_weight = std::vector<double>{1.0, 1.0, 0.0};
  face_current = std::vector<double>{0.0, 4.0, 8.0, 0.0};

  tenryu::radiation::compute_holo_consistency_cuda(consistency_source.data(),
                                                   E_HO.data(),
                                                   E_LO_pred.data(),
                                                   E_old.data(),
                                                   face_current.data(),
                                                   nullptr,
                                                   sigma_R.data(),
                                                   vol.data(),
                                                   node_r.data(),
                                                   lo_weight.data(),
                                                   n_cells,
                                                   n_groups,
                                                   dt);

  std::vector<double> source_host(n_total, 0.0);
  consistency_source.copy_to_host(source_host.data());

  const std::vector<double> expected{2.5, -1.5, 0.0};
  for (int i = 0; i < n_total; ++i) {
    REQUIRE(source_host[static_cast<std::size_t>(i)] ==
            Catch::Approx(expected[static_cast<std::size_t>(i)]).margin(1.0e-12));
  }
}

TEST_CASE("Difference tally finalize adds reference before physical clamp",
          "[radiation][tally][difference]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 1;
  const double dt = 1.0e-9;

  tenryu::core::GroupField1D rad_E(n_cells * n_groups);
  tenryu::core::GroupField1D E_ref_avg(n_cells * n_groups);
  tenryu::core::GroupField1D residual_E(n_cells * n_groups);
  tenryu::core::CellField1D vol(n_cells);
  vol = std::vector<double>{2.0, 4.0};
  E_ref_avg = std::vector<double>{3.0, 1.0};

  const std::vector<double> residual_density = {-1.0, -2.0};
  std::vector<double> tally(n_cells, 0.0);
  for (int c = 0; c < n_cells; ++c) {
    tally[static_cast<std::size_t>(c)] =
        residual_density[static_cast<std::size_t>(c)] *
        (c == 0 ? 2.0 : 4.0) * tenryu::core::constants::c_light * dt;
  }

  double* d_rad_E_tally = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_rad_E_tally),
                        sizeof(double) * tally.size()));
  cuda_check(cudaMemcpy(d_rad_E_tally,
                        tally.data(),
                        sizeof(double) * tally.size(),
                        cudaMemcpyHostToDevice));

  tenryu::radiation::tally_finalize_cuda(rad_E.data(),
                                         d_rad_E_tally,
                                         vol.data(),
                                         n_cells,
                                         n_groups,
                                         dt,
                                         nullptr,
                                         nullptr,
                                         E_ref_avg.data(),
                                         residual_E.data());

  std::vector<double> rad_E_host(n_cells, 0.0);
  std::vector<double> residual_host(n_cells, 0.0);
  rad_E.copy_to_host(rad_E_host.data());
  residual_E.copy_to_host(residual_host.data());

  REQUIRE(residual_host[0] == Catch::Approx(-1.0).epsilon(1.0e-12));
  REQUIRE(rad_E_host[0] == Catch::Approx(2.0).epsilon(1.0e-12));
  REQUIRE(residual_host[1] == Catch::Approx(-2.0).epsilon(1.0e-12));
  REQUIRE(rad_E_host[1] == Catch::Approx(0.0).margin(1.0e-14));

  cuda_check(cudaFree(d_rad_E_tally));
}

TEST_CASE("HOLO acceptance uses binary core ownership and re-centers difference reservoir",
          "[radiation][tally][holo][difference]") {
  constexpr int n_cells = 3;
  constexpr int n_groups = 2;
  constexpr int n_total = n_cells * n_groups;

  tenryu::core::GroupField1D rad_E(n_total);
  tenryu::core::GroupField1D E_LO(n_total);
  tenryu::core::GroupField1D E_ref(n_total);
  tenryu::core::GroupField1D previous_reference_U(n_total);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::GroupField1D residual_E(n_total);
  std::uint8_t* d_core_mask = nullptr;

  rad_E = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  E_LO = std::vector<double>{10.0, 20.0, 30.0, 40.0, 50.0, 60.0};
  const std::vector<std::uint8_t> core_mask{1U, 1U, 0U};
  E_ref = std::vector<double>{1.0, 1.0, 2.0, 2.0, 5.0, 7.0};
  previous_reference_U = std::vector<double>{100.0, 200.0, 300.0,
                                             400.0, 500.0, 600.0};
  vol = std::vector<double>{2.0, 3.0, 4.0};
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_core_mask),
                        sizeof(std::uint8_t) * core_mask.size()));
  cuda_check(cudaMemcpy(d_core_mask,
                        core_mask.data(),
                        sizeof(std::uint8_t) * core_mask.size(),
                        cudaMemcpyHostToDevice));

  tenryu::radiation::holo_accept_radiation_cuda(rad_E.data(),
                                                E_LO.data(),
                                                d_core_mask,
                                                n_cells,
                                                n_groups,
                                                nullptr,
                                                nullptr);
  tenryu::radiation::holo_set_reference_U_cuda(previous_reference_U.data(),
                                               rad_E.data(),
                                               vol.data(),
                                               d_core_mask,
                                               n_cells,
                                               n_groups);
  tenryu::radiation::difference_reproject_residual_cuda(residual_E.data(),
                                                        rad_E.data(),
                                                        E_ref.data(),
                                                        n_cells,
                                                        n_groups,
                                                        d_core_mask);

  std::vector<double> rad_E_host(n_total, 0.0);
  std::vector<double> previous_reference_host(n_total, 0.0);
  std::vector<double> residual_host(n_total, 0.0);
  rad_E.copy_to_host(rad_E_host.data());
  previous_reference_U.copy_to_host(previous_reference_host.data());
  residual_E.copy_to_host(residual_host.data());

  const std::vector<double> expected_E{10.0, 20.0, 30.0, 40.0, 5.0, 6.0};
  const std::vector<double> expected_previous_reference{20.0, 40.0, 90.0,
                                                        120.0, 500.0, 600.0};
  const std::vector<double> expected_residual{9.0, 19.0, 28.0, 38.0, 0.0, 0.0};
  for (int i = 0; i < n_total; ++i) {
    REQUIRE(rad_E_host[static_cast<std::size_t>(i)] ==
            Catch::Approx(expected_E[static_cast<std::size_t>(i)]).epsilon(1.0e-12));
    REQUIRE(previous_reference_host[static_cast<std::size_t>(i)] ==
            Catch::Approx(expected_previous_reference[static_cast<std::size_t>(i)])
                .epsilon(1.0e-12));
    REQUIRE(residual_host[static_cast<std::size_t>(i)] ==
            Catch::Approx(expected_residual[static_cast<std::size_t>(i)])
                .epsilon(1.0e-12));
  }
  cuda_check(cudaFree(d_core_mask));
}

TEST_CASE("HOLO DF recenter kills live census in LO core cells",
          "[radiation][holo][difference][pool]") {
  constexpr int n_cells = 3;
  constexpr int n_particles = 5;

  tenryu::radiation::PhotonPool pool;
  pool.allocate(n_particles);
  pool.n_alive = n_particles;
  pool.n_census = n_particles;

  const std::vector<std::int32_t> cell_id{0, 1, 2, 1, -1};
  const std::vector<std::uint8_t> alive{
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
      tenryu::radiation::kAlive,
      tenryu::radiation::kDead,
      tenryu::radiation::kAlive};
  const std::vector<std::uint8_t> holo_core{0U, 1U, 0U};

  cuda_check(cudaMemcpy(pool.cell_id,
                        cell_id.data(),
                        sizeof(std::int32_t) * cell_id.size(),
                        cudaMemcpyHostToDevice));
  cuda_check(cudaMemcpy(pool.alive,
                        alive.data(),
                        sizeof(std::uint8_t) * alive.size(),
                        cudaMemcpyHostToDevice));
  std::uint8_t* d_holo_core = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_holo_core),
                        sizeof(std::uint8_t) * holo_core.size()));
  cuda_check(cudaMemcpy(d_holo_core,
                        holo_core.data(),
                        sizeof(std::uint8_t) * holo_core.size(),
                        cudaMemcpyHostToDevice));

  const int killed = tenryu::radiation::kill_difference_census_in_holo_core_cuda(
      pool, d_holo_core, n_particles, n_cells);

  std::vector<std::uint8_t> alive_host(n_particles, tenryu::radiation::kDead);
  cuda_check(cudaMemcpy(alive_host.data(),
                        pool.alive,
                        sizeof(std::uint8_t) * alive_host.size(),
                        cudaMemcpyDeviceToHost));

  REQUIRE(killed == 1);
  REQUIRE(alive_host[0] == tenryu::radiation::kAlive);
  REQUIRE(alive_host[1] == tenryu::radiation::kDead);
  REQUIRE(alive_host[2] == tenryu::radiation::kAlive);
  REQUIRE(alive_host[3] == tenryu::radiation::kDead);
  REQUIRE(alive_host[4] == tenryu::radiation::kAlive);

  cuda_check(cudaFree(d_holo_core));
}

TEST_CASE("Reference face transport conserves closed 1D domain",
          "[radiation][tally][difference]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 1;
  const double dt = 1.0e-14;

  tenryu::core::GroupField1D E_ref(n_cells * n_groups);
  tenryu::core::GroupField1D sigma_R(n_cells * n_groups);
  tenryu::core::NodeField1D node_r(n_cells + 1);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::GroupField1D U_ref_end(n_cells * n_groups);
  tenryu::core::GroupField1D delta_U(n_cells * n_groups);
  tenryu::core::GroupField1D E_ref_avg(n_cells * n_groups);
  tenryu::core::GroupField1D ref_face_current((n_cells + 1) * n_groups);

  E_ref = std::vector<double>{4.0, 1.0};
  sigma_R = std::vector<double>{0.0, 0.0};
  node_r = std::vector<double>{0.0, 1.0, 2.0};
  vol = std::vector<double>{1.0, 1.0};

  const auto result = tenryu::radiation::reference_face_transport_1d_cuda(
      U_ref_end.data(),
      delta_U.data(),
      E_ref_avg.data(),
      E_ref.data(),
      sigma_R.data(),
      node_r.data(),
      vol.data(),
      nullptr,
      nullptr,
      n_cells,
      n_groups,
      dt,
      tenryu::radiation::kBoundaryReflect,
      tenryu::radiation::kBoundaryReflect,
      nullptr,
      ref_face_current.data());

  std::vector<double> U_host(n_cells, 0.0);
  std::vector<double> delta_host(n_cells, 0.0);
  std::vector<double> face_host(n_cells + 1, 0.0);
  U_ref_end.copy_to_host(U_host.data());
  delta_U.copy_to_host(delta_host.data());
  ref_face_current.copy_to_host(face_host.data());

  const double Q = tenryu::core::constants::c_light * 0.25 *
                   (4.0 * 3.141592653589793238462643383279502884) *
                   (4.0 - 1.0) * dt;
  REQUIRE(delta_host[0] == Catch::Approx(-Q).epsilon(1.0e-12));
  REQUIRE(delta_host[1] == Catch::Approx(Q).epsilon(1.0e-12));
  REQUIRE(face_host[0] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(face_host[1] == Catch::Approx(Q).epsilon(1.0e-12));
  REQUIRE(face_host[2] == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE((U_host[0] + U_host[1]) == Catch::Approx(5.0).epsilon(1.0e-12));
  REQUIRE(result.E_escape == Catch::Approx(0.0).margin(1.0e-14));
  REQUIRE(result.E_source == Catch::Approx(0.0).margin(1.0e-14));
}

TEST_CASE("Reference face AP limiter matches thin and thick limits",
          "[radiation][tally][difference]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 1;
  const double dt = 1.0e-14;

  tenryu::core::GroupField1D E_ref(n_cells * n_groups);
  tenryu::core::GroupField1D sigma_R(n_cells * n_groups);
  tenryu::core::NodeField1D node_r(n_cells + 1);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::GroupField1D U_ref_end(n_cells * n_groups);
  tenryu::core::GroupField1D delta_U(n_cells * n_groups);
  tenryu::core::GroupField1D E_ref_avg(n_cells * n_groups);

  E_ref = std::vector<double>{4.0, 1.0};
  node_r = std::vector<double>{0.0, 1.0, 2.0};
  vol = std::vector<double>{1.0, 1.0};

  sigma_R = std::vector<double>{0.0, 0.0};
  (void)tenryu::radiation::reference_face_transport_1d_cuda(
      U_ref_end.data(),
      delta_U.data(),
      E_ref_avg.data(),
      E_ref.data(),
      sigma_R.data(),
      node_r.data(),
      vol.data(),
      nullptr,
      nullptr,
      n_cells,
      n_groups,
      dt,
      tenryu::radiation::kBoundaryReflect,
      tenryu::radiation::kBoundaryReflect);
  std::vector<double> thin_delta(n_cells, 0.0);
  delta_U.copy_to_host(thin_delta.data());

  sigma_R = std::vector<double>{1.0e6, 1.0e6};
  (void)tenryu::radiation::reference_face_transport_1d_cuda(
      U_ref_end.data(),
      delta_U.data(),
      E_ref_avg.data(),
      E_ref.data(),
      sigma_R.data(),
      node_r.data(),
      vol.data(),
      nullptr,
      nullptr,
      n_cells,
      n_groups,
      dt,
      tenryu::radiation::kBoundaryReflect,
      tenryu::radiation::kBoundaryReflect);
  std::vector<double> thick_delta(n_cells, 0.0);
  delta_U.copy_to_host(thick_delta.data());

  constexpr double tau = 1.0e6;
  const double thick_limit = 1.0 / (0.75 * tau);
  REQUIRE(thin_delta[0] < 0.0);
  REQUIRE(thick_delta[0] < 0.0);
  REQUIRE(std::abs(thick_delta[0] / thin_delta[0]) ==
          Catch::Approx(thick_limit).epsilon(1.0e-12));
}

TEST_CASE("Reference face transport accounts outer vacuum leakage",
          "[radiation][tally][difference]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 1;
  const double dt = 1.0e-14;

  tenryu::core::GroupField1D E_ref(n_cells * n_groups);
  tenryu::core::GroupField1D sigma_R(n_cells * n_groups);
  tenryu::core::NodeField1D node_r(n_cells + 1);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::GroupField1D U_ref_end(n_cells * n_groups);
  tenryu::core::GroupField1D delta_U(n_cells * n_groups);
  tenryu::core::GroupField1D E_ref_avg(n_cells * n_groups);

  E_ref = std::vector<double>{2.0};
  sigma_R = std::vector<double>{0.0};
  node_r = std::vector<double>{0.0, 1.0};
  vol = std::vector<double>{1.0};

  const auto result = tenryu::radiation::reference_face_transport_1d_cuda(
      U_ref_end.data(),
      delta_U.data(),
      E_ref_avg.data(),
      E_ref.data(),
      sigma_R.data(),
      node_r.data(),
      vol.data(),
      nullptr,
      nullptr,
      n_cells,
      n_groups,
      dt,
      tenryu::radiation::kBoundaryReflect,
      tenryu::radiation::kBoundaryVacuum);

  std::vector<double> delta_host(n_cells, 0.0);
  delta_U.copy_to_host(delta_host.data());

  const double expected_escape =
      tenryu::core::constants::c_light * 0.25 *
      (4.0 * 3.141592653589793238462643383279502884) * 2.0 * dt;
  REQUIRE(delta_host[0] == Catch::Approx(-expected_escape).epsilon(1.0e-12));
  REQUIRE(result.E_escape == Catch::Approx(expected_escape).epsilon(1.0e-12));
  REQUIRE(result.E_source == Catch::Approx(0.0).margin(1.0e-14));
}
