#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "radiation/diffusion_source_solve.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_diffusion_source_solve");
}

std::uint8_t* upload_diffusion_mask(const std::vector<std::uint8_t>& mask) {
  std::uint8_t* d_mask = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_mask),
                        sizeof(std::uint8_t) * mask.size()));
  cuda_check(cudaMemcpy(d_mask,
                        mask.data(),
                        sizeof(std::uint8_t) * mask.size(),
                        cudaMemcpyHostToDevice));
  return d_mask;
}

}  // namespace

TEST_CASE("Diffusion source solve leaves equilibrium cell stationary",
          "[radiation][diffusion_source]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 1;
  constexpr double T0 = 2.0;
  constexpr double rho0 = 1.0;
  constexpr double vol0 = 3.0;
  constexpr double cv_e = 5.0;
  constexpr double gm1 = 2.0 / 3.0;
  constexpr double sigma_P = 1.0e-10;
  constexpr double dt = 1.0e-9;

  const double B0 = tenryu::core::constants::a_eV * T0 * T0 * T0 * T0;

  tenryu::core::GroupField1D diff_E(n_cells * n_groups);
  tenryu::core::GroupField1D sigma(n_cells * n_groups);
  tenryu::core::GroupField1D rad_dep(n_cells * n_groups);
  tenryu::core::GroupField1D rad_emit(n_cells * n_groups);
  tenryu::core::CellField1D ee(n_cells);
  tenryu::core::CellField1D Te(n_cells);
  tenryu::core::CellField1D Pe(n_cells);
  tenryu::core::CellField1D rho(n_cells);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::CellField1D mass(n_cells);

  diff_E = std::vector<double>{B0};
  sigma = std::vector<double>{sigma_P};
  ee = std::vector<double>{cv_e * T0};
  Te = std::vector<double>{T0};
  Pe = std::vector<double>{gm1 * rho0 * cv_e * T0};
  rho = std::vector<double>{rho0};
  vol = std::vector<double>{vol0};
  mass = std::vector<double>{rho0 * vol0};

  std::uint8_t* d_diff_cell = upload_diffusion_mask({1U});

  tenryu::radiation::DiffusionSourceSolveInputs in{};
  in.diff_E = diff_E.data();
  in.ee = ee.data();
  in.Te = Te.data();
  in.Pe = Pe.data();
  in.sigma_P = sigma.data();
  in.vol = vol.data();
  in.rho = rho.data();
  in.mass = mass.data();
  in.diff_cell = d_diff_cell;
  in.rad_dep = rad_dep.data();
  in.rad_emit = rad_emit.data();
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt_s = dt;
  in.cv_e_const = cv_e;
  in.pressure_gamma_minus_one = gm1;
  in.temperature_floor_eV = 1.0e-6;

  const auto result = tenryu::radiation::diffusion_source_solve_cuda(in);

  std::vector<double> out_E(1, 0.0);
  std::vector<double> out_ee(1, 0.0);
  std::vector<double> out_Te(1, 0.0);
  std::vector<double> out_Pe(1, 0.0);
  std::vector<double> out_dep(1, 0.0);
  std::vector<double> out_emit(1, 0.0);
  diff_E.copy_to_host(out_E.data());
  ee.copy_to_host(out_ee.data());
  Te.copy_to_host(out_Te.data());
  Pe.copy_to_host(out_Pe.data());
  rad_dep.copy_to_host(out_dep.data());
  rad_emit.copy_to_host(out_emit.data());

  REQUIRE(result.n_failures == 0);
  REQUIRE(out_Te[0] == Catch::Approx(T0).epsilon(1.0e-12));
  REQUIRE(out_ee[0] == Catch::Approx(cv_e * T0).epsilon(1.0e-12));
  REQUIRE(out_Pe[0] == Catch::Approx(gm1 * rho0 * cv_e * T0).epsilon(1.0e-12));
  REQUIRE(out_E[0] == Catch::Approx(B0).epsilon(1.0e-12));
  REQUIRE(out_dep[0] == Catch::Approx(out_emit[0]).epsilon(1.0e-12));

  cuda_check(cudaFree(d_diff_cell));
}

TEST_CASE("Diffusion source solve conserves cell energy while heating matter",
          "[radiation][diffusion_source]") {
  constexpr int n_cells = 1;
  constexpr int n_groups = 1;
  constexpr double T0 = 1.0;
  constexpr double rho0 = 1.0;
  constexpr double vol0 = 1.0;
  constexpr double cv_e = 1.0;
  constexpr double gm1 = 2.0 / 3.0;
  constexpr double sigma_P = 1.0;
  constexpr double dt = 1.0e-11;

  const double B0 = tenryu::core::constants::a_eV * T0 * T0 * T0 * T0;
  const double E_old = 2.0 * B0;
  const double ee_old = cv_e * T0;
  const double total_before = rho0 * vol0 * ee_old + vol0 * E_old;

  tenryu::core::GroupField1D diff_E(n_cells * n_groups);
  tenryu::core::GroupField1D sigma(n_cells * n_groups);
  tenryu::core::GroupField1D rad_dep(n_cells * n_groups);
  tenryu::core::GroupField1D rad_emit(n_cells * n_groups);
  tenryu::core::CellField1D ee(n_cells);
  tenryu::core::CellField1D Te(n_cells);
  tenryu::core::CellField1D Pe(n_cells);
  tenryu::core::CellField1D rho(n_cells);
  tenryu::core::CellField1D vol(n_cells);
  tenryu::core::CellField1D mass(n_cells);

  diff_E = std::vector<double>{E_old};
  sigma = std::vector<double>{sigma_P};
  ee = std::vector<double>{ee_old};
  Te = std::vector<double>{T0};
  Pe = std::vector<double>{gm1 * rho0 * ee_old};
  rho = std::vector<double>{rho0};
  vol = std::vector<double>{vol0};
  mass = std::vector<double>{rho0 * vol0};

  std::uint8_t* d_diff_cell = upload_diffusion_mask({1U});

  tenryu::radiation::DiffusionSourceSolveInputs in{};
  in.diff_E = diff_E.data();
  in.ee = ee.data();
  in.Te = Te.data();
  in.Pe = Pe.data();
  in.sigma_P = sigma.data();
  in.vol = vol.data();
  in.rho = rho.data();
  in.mass = mass.data();
  in.diff_cell = d_diff_cell;
  in.rad_dep = rad_dep.data();
  in.rad_emit = rad_emit.data();
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  in.dt_s = dt;
  in.cv_e_const = cv_e;
  in.pressure_gamma_minus_one = gm1;
  in.temperature_floor_eV = 1.0e-6;

  const auto result = tenryu::radiation::diffusion_source_solve_cuda(in);

  std::vector<double> out_E(1, 0.0);
  std::vector<double> out_ee(1, 0.0);
  std::vector<double> out_Te(1, 0.0);
  diff_E.copy_to_host(out_E.data());
  ee.copy_to_host(out_ee.data());
  Te.copy_to_host(out_Te.data());

  const double total_after = rho0 * vol0 * out_ee[0] + vol0 * out_E[0];

  REQUIRE(result.n_failures == 0);
  REQUIRE(out_Te[0] > T0);
  REQUIRE(out_E[0] < E_old);
  REQUIRE(total_after == Catch::Approx(total_before).epsilon(1.0e-10));

  cuda_check(cudaFree(d_diff_cell));
}
