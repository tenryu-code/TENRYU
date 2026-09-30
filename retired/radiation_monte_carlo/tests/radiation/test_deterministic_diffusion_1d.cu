#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/field.hpp"
#include "core/error.hpp"
#include "numerics/rkl2_sts.hpp"
#include "radiation/deterministic_diffusion_1d.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_deterministic_diffusion_1d");
}

std::uint8_t* upload_mask(const std::vector<std::uint8_t>& mask) {
  std::uint8_t* d_mask = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask),
                        sizeof(std::uint8_t) * mask.size()));
  cuda_check(cudaMemcpy(d_mask,
                        mask.data(),
                        sizeof(std::uint8_t) * mask.size(),
                        cudaMemcpyHostToDevice));
  return d_mask;
}

std::vector<double> spherical_nodes(const int n, const double r0, const double dr) {
  std::vector<double> node(static_cast<std::size_t>(n + 1), 0.0);
  for (int i = 0; i <= n; ++i) {
    node[static_cast<std::size_t>(i)] = r0 + dr * static_cast<double>(i);
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

double weighted_energy(const std::vector<double>& E, const std::vector<double>& vol) {
  long double sum = 0.0L;
  for (std::size_t i = 0; i < E.size(); ++i) {
    sum += static_cast<long double>(E[i]) * static_cast<long double>(vol[i]);
  }
  return static_cast<double>(sum);
}

}  // namespace

TEST_CASE("RKL2 coefficients match Meyer small-stage values",
          "[radiation][deterministic_diffusion]") {
  const auto coeff = tenryu::numerics::compute_rkl2_coefficients(3, 0.0);
  REQUIRE(coeff.s == 3);
  REQUIRE(coeff.mu_tilde[1] == Catch::Approx(2.0 / 15.0).epsilon(1.0e-14));
  REQUIRE(coeff.mu[2] == Catch::Approx(1.5).epsilon(1.0e-14));
  REQUIRE(coeff.mu_tilde[2] == Catch::Approx(0.6).epsilon(1.0e-14));
  REQUIRE(coeff.nu[2] == Catch::Approx(-0.5).epsilon(1.0e-14));
  REQUIRE(coeff.gamma_tilde[2] == Catch::Approx(-0.4).epsilon(1.0e-14));
  REQUIRE(coeff.mu[3] == Catch::Approx(25.0 / 12.0).epsilon(1.0e-14));
  REQUIRE(coeff.mu_tilde[3] == Catch::Approx(5.0 / 6.0).epsilon(1.0e-14));
  REQUIRE(coeff.nu[3] == Catch::Approx(-5.0 / 6.0).epsilon(1.0e-14));
  REQUIRE(coeff.gamma_tilde[3] == Catch::Approx(-5.0 / 9.0).epsilon(1.0e-14));
}

TEST_CASE("RKL2 stage estimate uses stability capacity",
          "[radiation][deterministic_diffusion]") {
  REQUIRE(tenryu::numerics::estimate_rkl2_stages(1.0, 1.0, 1.0) == 2);
  REQUIRE(tenryu::numerics::estimate_rkl2_stages(64.0, 1.0, 1.0) == 16);
  REQUIRE(tenryu::numerics::estimate_rkl2_stages(256.0, 1.0, 1.0) == 32);
  REQUIRE(tenryu::numerics::estimate_rkl2_subcycles(64.0, 1.0, 8, 1.0) == 4);
}

TEST_CASE("1D pseudo-planar RKL2 diffusion follows cosine decay",
          "[radiation][deterministic_diffusion]") {
  constexpr int n_cells = 128;
  constexpr int n_groups = 1;
  constexpr double r0 = 1.0e8;
  constexpr double length = 1.0;
  constexpr double dr = length / static_cast<double>(n_cells);
  constexpr double D = 1.0;
  constexpr double dt = 1.0e-3;
  const std::vector<double> node = spherical_nodes(n_cells, r0, dr);
  const std::vector<double> vol = spherical_volumes(node);
  const double sigma_R = tenryu::core::constants::c_light / (3.0 * D);

  std::vector<double> E(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const double x = (static_cast<double>(c) + 0.5) * dr;
    E[static_cast<std::size_t>(c)] =
        2.0 + 0.1 * std::cos(3.14159265358979323846 * x / length);
  }

  tenryu::core::GroupField1D d_E(n_cells * n_groups);
  tenryu::core::GroupField1D d_sigma(n_cells * n_groups);
  tenryu::core::CellField1D d_vol(n_cells);
  tenryu::core::NodeField1D d_node(n_cells + 1);
  d_E.copy_from_host(E.data());
  d_sigma.copy_from_host(std::vector<double>(static_cast<std::size_t>(n_cells), sigma_R));
  d_vol.copy_from_host(vol.data());
  d_node.copy_from_host(node.data());
  std::uint8_t* d_mask =
      upload_mask(std::vector<std::uint8_t>(static_cast<std::size_t>(n_cells), 1U));

  tenryu::radiation::DiffusionStepInputs in{};
  in.diff_E = d_E.data();
  in.sigma_R = d_sigma.data();
  in.vol = d_vol.data();
  in.node_r = d_node.data();
  in.diff_cell = d_mask;
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt = dt;
  in.bc_inner = 0;
  in.bc_outer = 0;
  const auto coeff = tenryu::numerics::compute_rkl2_coefficients(16, 0.0);
  (void)tenryu::radiation::deterministic_diffusion_step_1d(in, coeff);

  std::vector<double> out(static_cast<std::size_t>(n_cells), 0.0);
  d_E.copy_to_host(out.data());
  const double decay =
      std::exp(-D * 3.14159265358979323846 * 3.14159265358979323846 * dt /
               (length * length));
  double l2 = 0.0;
  double ref_l2 = 0.0;
  for (int c = 0; c < n_cells; ++c) {
    const double x = (static_cast<double>(c) + 0.5) * dr;
    const double ref =
        2.0 + 0.1 * decay * std::cos(3.14159265358979323846 * x / length);
    const double err = out[static_cast<std::size_t>(c)] - ref;
    l2 += err * err;
    ref_l2 += ref * ref;
  }
  REQUIRE(std::sqrt(l2 / ref_l2) < 2.0e-3);
  cuda_check(cudaFree(d_mask));
}

TEST_CASE("1D spherical diffusion conserves energy with reflective boundaries",
          "[radiation][deterministic_diffusion]") {
  constexpr int n_cells = 32;
  constexpr int n_groups = 1;
  constexpr double D = 0.2;
  constexpr double dt = 2.0e-5;
  const std::vector<double> node = spherical_nodes(n_cells, 0.0, 1.0 / n_cells);
  const std::vector<double> vol = spherical_volumes(node);
  const double sigma_R = tenryu::core::constants::c_light / (3.0 * D);

  std::vector<double> E(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const double x = (static_cast<double>(c) + 0.5) / static_cast<double>(n_cells);
    E[static_cast<std::size_t>(c)] = 1.0 + 0.5 * std::exp(-200.0 * (x - 0.4) * (x - 0.4));
  }
  const double E_before = weighted_energy(E, vol);

  tenryu::core::GroupField1D d_E(n_cells * n_groups);
  tenryu::core::GroupField1D d_sigma(n_cells * n_groups);
  tenryu::core::CellField1D d_vol(n_cells);
  tenryu::core::NodeField1D d_node(n_cells + 1);
  d_E.copy_from_host(E.data());
  d_sigma.copy_from_host(std::vector<double>(static_cast<std::size_t>(n_cells), sigma_R));
  d_vol.copy_from_host(vol.data());
  d_node.copy_from_host(node.data());
  std::uint8_t* d_mask =
      upload_mask(std::vector<std::uint8_t>(static_cast<std::size_t>(n_cells), 1U));

  tenryu::radiation::DiffusionStepInputs in{};
  in.diff_E = d_E.data();
  in.sigma_R = d_sigma.data();
  in.vol = d_vol.data();
  in.node_r = d_node.data();
  in.diff_cell = d_mask;
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt = dt;
  in.bc_inner = 0;
  in.bc_outer = 0;
  const auto coeff = tenryu::numerics::compute_rkl2_coefficients(12, 0.0);
  const auto result = tenryu::radiation::deterministic_diffusion_step_1d(in, coeff);

  std::vector<double> out(static_cast<std::size_t>(n_cells), 0.0);
  d_E.copy_to_host(out.data());
  const double E_after = weighted_energy(out, vol);
  REQUIRE(E_after == Catch::Approx(E_before).epsilon(1.0e-10));
  REQUIRE(result.E_leaked == Catch::Approx(0.0).margin(1.0e-10 * E_before));
  cuda_check(cudaFree(d_mask));
}

TEST_CASE("1D vacuum boundary leaks deterministic diffusion energy",
          "[radiation][deterministic_diffusion]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 1;
  constexpr double D = 0.1;
  constexpr double dt = 1.0e-14;
  const std::vector<double> node = {1.0, 2.0};
  const std::vector<double> vol = spherical_volumes(node);
  const std::vector<double> E = {5.0};
  const double sigma_R = tenryu::core::constants::c_light / (3.0 * D);

  tenryu::core::GroupField1D d_E(n_cells * n_groups);
  tenryu::core::GroupField1D d_sigma(n_cells * n_groups);
  tenryu::core::CellField1D d_vol(n_cells);
  tenryu::core::NodeField1D d_node(n_cells + 1);
  d_E.copy_from_host(E.data());
  d_sigma.copy_from_host(std::vector<double>{sigma_R});
  d_vol.copy_from_host(vol.data());
  d_node.copy_from_host(node.data());
  std::uint8_t* d_mask = upload_mask({1U});

  tenryu::radiation::DiffusionStepInputs in{};
  in.diff_E = d_E.data();
  in.sigma_R = d_sigma.data();
  in.vol = d_vol.data();
  in.node_r = d_node.data();
  in.diff_cell = d_mask;
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt = dt;
  in.bc_inner = 0;
  in.bc_outer = 1;
  const auto coeff = tenryu::numerics::compute_rkl2_coefficients(4, 0.0);
  const auto result = tenryu::radiation::deterministic_diffusion_step_1d(in, coeff);

  std::vector<double> out(1, 0.0);
  d_E.copy_to_host(out.data());
  const double E_before = E[0] * vol[0];
  const double E_after = out[0] * vol[0];
  REQUIRE(E_after < E_before);
  REQUIRE(result.E_leaked > 0.0);
  REQUIRE(result.E_leaked == Catch::Approx(E_before - E_after).epsilon(1.0e-10));
  cuda_check(cudaFree(d_mask));
}

TEST_CASE("1D RKL2 face-current source deposits once",
          "[radiation][deterministic_diffusion]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 1;
  constexpr double D = 0.1;
  constexpr double dt = 1.0e-6;
  constexpr double J_in = 3.0;
  const std::vector<double> node = {1.0, 2.0};
  const std::vector<double> vol = spherical_volumes(node);
  const double sigma_R = tenryu::core::constants::c_light / (3.0 * D);

  tenryu::core::GroupField1D d_E(n_cells * n_groups);
  tenryu::core::GroupField1D d_sigma(n_cells * n_groups);
  tenryu::core::CellField1D d_vol(n_cells);
  tenryu::core::NodeField1D d_node(n_cells + 1);
  d_E.copy_from_host(std::vector<double>{0.0});
  d_sigma.copy_from_host(std::vector<double>{sigma_R});
  d_vol.copy_from_host(vol.data());
  d_node.copy_from_host(node.data());
  std::uint8_t* d_mask = upload_mask({1U});
  double* d_face_current = nullptr;
  const std::vector<double> face_current = {J_in, 0.0};
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_face_current),
                        sizeof(double) * face_current.size()));
  cuda_check(cudaMemcpy(d_face_current,
                        face_current.data(),
                        sizeof(double) * face_current.size(),
                        cudaMemcpyHostToDevice));

  tenryu::radiation::DiffusionStepInputs in{};
  in.diff_E = d_E.data();
  in.sigma_R = d_sigma.data();
  in.vol = d_vol.data();
  in.node_r = d_node.data();
  in.diff_cell = d_mask;
  in.face_current_in = d_face_current;
  in.face_current_dt = dt;
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt = dt;
  in.bc_inner = 0;
  in.bc_outer = 0;
  const auto coeff = tenryu::numerics::compute_rkl2_coefficients(8, 0.05);
  const auto result = tenryu::radiation::deterministic_diffusion_step_1d(in, coeff);

  std::vector<double> out(1, 0.0);
  d_E.copy_to_host(out.data());
  const double E_after = out[0] * vol[0];
  REQUIRE(E_after == Catch::Approx(J_in).epsilon(1.0e-10));
  REQUIRE(result.E_after == Catch::Approx(J_in).epsilon(1.0e-10));
  REQUIRE(result.E_leaked == Catch::Approx(0.0).margin(1.0e-10));
  std::vector<double> face_after(2, -1.0);
  cuda_check(cudaMemcpy(face_after.data(),
                        d_face_current,
                        sizeof(double) * face_after.size(),
                        cudaMemcpyDeviceToHost));
  REQUIRE(face_after[0] == Catch::Approx(0.0).margin(0.0));
  REQUIRE(face_after[1] == Catch::Approx(0.0).margin(0.0));
  cuda_check(cudaFree(d_face_current));
  cuda_check(cudaFree(d_mask));
}

TEST_CASE("1D face-current source deposits when RKL2 operator is skipped",
          "[radiation][deterministic_diffusion]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 1;
  constexpr double D = 0.1;
  constexpr double dt = 1.0e-6;
  constexpr double J_in = 3.0;
  const std::vector<double> node = {1.0, 2.0};
  const std::vector<double> vol = spherical_volumes(node);
  const double sigma_R = tenryu::core::constants::c_light / (3.0 * D);

  tenryu::core::GroupField1D d_E(n_cells * n_groups);
  tenryu::core::GroupField1D d_sigma(n_cells * n_groups);
  tenryu::core::CellField1D d_vol(n_cells);
  tenryu::core::NodeField1D d_node(n_cells + 1);
  d_E.copy_from_host(std::vector<double>{0.0});
  d_sigma.copy_from_host(std::vector<double>{sigma_R});
  d_vol.copy_from_host(vol.data());
  d_node.copy_from_host(node.data());
  std::uint8_t* d_mask = upload_mask({1U});
  double* d_face_current = nullptr;
  const std::vector<double> face_current = {J_in, 0.0};
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_face_current),
                        sizeof(double) * face_current.size()));
  cuda_check(cudaMemcpy(d_face_current,
                        face_current.data(),
                        sizeof(double) * face_current.size(),
                        cudaMemcpyHostToDevice));

  tenryu::radiation::DiffusionStepInputs in{};
  in.diff_E = d_E.data();
  in.sigma_R = d_sigma.data();
  in.vol = d_vol.data();
  in.node_r = d_node.data();
  in.diff_cell = d_mask;
  in.face_current_in = d_face_current;
  in.face_current_dt = dt;
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt = dt;
  in.bc_inner = 0;
  in.bc_outer = 0;
  const auto result =
      tenryu::radiation::deterministic_diffusion_step_1d(in, 0, 0.05, 0.8);

  std::vector<double> out(1, 0.0);
  d_E.copy_to_host(out.data());
  const double E_after = out[0] * vol[0];
  REQUIRE(E_after == Catch::Approx(J_in).epsilon(1.0e-10));
  REQUIRE(result.E_after == Catch::Approx(J_in).epsilon(1.0e-10));
  REQUIRE(result.E_leaked == Catch::Approx(0.0).margin(1.0e-10));
  std::vector<double> face_after(2, -1.0);
  cuda_check(cudaMemcpy(face_after.data(),
                        d_face_current,
                        sizeof(double) * face_after.size(),
                        cudaMemcpyDeviceToHost));
  REQUIRE(face_after[0] == Catch::Approx(0.0).margin(0.0));
  REQUIRE(face_after[1] == Catch::Approx(0.0).margin(0.0));
  cuda_check(cudaFree(d_face_current));
  cuda_check(cudaFree(d_mask));
}
