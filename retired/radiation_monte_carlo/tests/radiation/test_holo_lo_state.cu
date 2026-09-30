#include <limits>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cuda_runtime.h>

#include "radiation/holo_lo_state.cuh"

namespace {

void require_cuda(const cudaError_t err) {
  REQUIRE(err == cudaSuccess);
}

}  // namespace

TEST_CASE("HOLO LO initialization preserves finite positive global state",
          "[radiation][holo]") {
  constexpr int n_cells = 2;
  constexpr int n_groups = 3;
  constexpr int n_values = n_cells * n_groups;

  const std::vector<double> initial_E_LO = {
      1.0,
      -2.0,
      std::numeric_limits<double>::quiet_NaN(),
      4.0,
      0.0,
      std::numeric_limits<double>::infinity()};

  double* d_E_LO = nullptr;
  require_cuda(cudaMalloc(reinterpret_cast<void**>(&d_E_LO), sizeof(double) * n_values));
  require_cuda(cudaMemcpy(d_E_LO,
                          initial_E_LO.data(),
                          sizeof(double) * n_values,
                          cudaMemcpyHostToDevice));

  tenryu::radiation::initialize_holo_lo_state_cuda(d_E_LO, n_cells, n_groups);

  std::vector<double> E_LO(n_values, 0.0);
  require_cuda(cudaMemcpy(E_LO.data(),
                          d_E_LO,
                          sizeof(double) * n_values,
                          cudaMemcpyDeviceToHost));
  require_cuda(cudaFree(d_E_LO));

  REQUIRE(E_LO[0] == Catch::Approx(1.0));
  REQUIRE(E_LO[1] == Catch::Approx(0.0));
  REQUIRE(E_LO[2] == Catch::Approx(0.0));
  REQUIRE(E_LO[3] == Catch::Approx(4.0));
  REQUIRE(E_LO[4] == Catch::Approx(0.0));
  REQUIRE(E_LO[5] == Catch::Approx(0.0));
}
