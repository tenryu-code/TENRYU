#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "radiation/fleck.cuh"

namespace {

void cuda_check(const cudaError_t err) {
  TENRYU_ASSERT(err == cudaSuccess, "CUDA failure in test_fleck");
}

}  // namespace

TEST_CASE("Fleck factor value", "[radiation][fleck]") {
  tenryu::core::Config cfg;
  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "m";
  mat.A = 1.0;
  mat.ideal_gas_gamma = 5.0 / 3.0;
  cfg.materials.materials = {mat};
  cfg.radiation.imc.alpha = 1.0;
  cfg.radiation.imc.f_max = 1.0;

  tenryu::core::CellField1D rho(1);
  tenryu::core::CellField1D Te(1);
  tenryu::core::CellField1D zbar(1);
  rho = std::vector<double>{1.0};
  Te = std::vector<double>{10.0};
  zbar = std::vector<double>{1.0};

  double* d_sigma_a = nullptr;
  double* d_f = nullptr;
  double* d_sigma_a_eff = nullptr;
  double* d_sigma_s_eff = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_f), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_a_eff), sizeof(double)));
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_sigma_s_eff), sizeof(double)));

  const double sigma_a = 2.0;
  cuda_check(cudaMemcpy(d_sigma_a, &sigma_a, sizeof(double), cudaMemcpyHostToDevice));
  std::vector<std::uint8_t> host_void(1, 0U);

  tenryu::radiation::FleckView view;
  view.rho = rho.data();
  view.Te = Te.data();
  view.zbar = zbar.data();
  view.cell_is_void = host_void.data();
  view.sigma_a = d_sigma_a;
  view.f_fleck = d_f;
  view.sigma_a_eff = d_sigma_a_eff;
  view.sigma_s_eff = d_sigma_s_eff;
  view.n_cells = 1;
  view.n_groups = 1;
  view.dt = 1.0e-12;

  tenryu::radiation::compute_fleck_and_sigma_eff_cuda(view, cfg);

  double f = 0.0;
  double sigma_a_eff = 0.0;
  double sigma_s_eff = 0.0;
  cuda_check(cudaMemcpy(&f, d_f, sizeof(double), cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&sigma_a_eff,
                        d_sigma_a_eff,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));
  cuda_check(cudaMemcpy(&sigma_s_eff,
                        d_sigma_s_eff,
                        sizeof(double),
                        cudaMemcpyDeviceToHost));

  const double gm1 = mat.ideal_gas_gamma - 1.0;
  const double cv_mass_e = tenryu::core::constants::eV_to_erg /
                           (mat.A * tenryu::core::constants::proton_mass * gm1);
  const double Cv_e = cv_mass_e;
  const double beta = 4.0 * tenryu::core::constants::a_eV * 10.0 * 10.0 * 10.0 / Cv_e;
  const double f_ref = 1.0 / (1.0 + tenryu::core::constants::c_light * beta * sigma_a * 1.0e-12);

  REQUIRE(f == Catch::Approx(f_ref).epsilon(1.0e-12));
  REQUIRE(sigma_a_eff == Catch::Approx(f * sigma_a).epsilon(1.0e-12));
  REQUIRE(sigma_s_eff == Catch::Approx((1.0 - f) * sigma_a).epsilon(1.0e-12));

  cuda_check(cudaFree(d_sigma_s_eff));
  cuda_check(cudaFree(d_sigma_a_eff));
  cuda_check(cudaFree(d_f));
  cuda_check(cudaFree(d_sigma_a));
}
