#include "coupling/dt_controller_device.cuh"

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::coupling {
namespace {

__global__ void dt_controller_check_kernel(
    tenryu::coupling::DtLadderIn in,
    double* out2) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const auto r = tenryu::coupling::dt_ladder_eval(in);
    out2[0] = r.dt_chosen;
    out2[1] = static_cast<double>(r.limiter);
  }
}

}  // namespace

void run_dt_controller_check(const DtLadderIn& in, double out[2]) {
  double* d_out2 = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_out2),
                        2 * sizeof(double)));
  dt_controller_check_kernel<<<1, 1>>>(in, d_out2);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(out, d_out2, 2 * sizeof(double),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_out2));
}

}  // namespace tenryu::coupling
