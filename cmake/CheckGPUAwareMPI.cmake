if(NOT TENRYU_ENABLE_MPI)
  set(TENRYU_GPU_AWARE_MPI_COMPILE 0)
  return()
endif()

set(TENRYU_GPU_AWARE_MPI_COMPILE 0)

set(_tenryu_gpu_aware_mpi_probe_source [[
#include <cuda_runtime.h>
#include <mpi.h>

int main() {
  double* device_buffer = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device_buffer), sizeof(double)) != cudaSuccess) {
    return 1;
  }

  MPI_Send(device_buffer, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
  cudaFree(device_buffer);
  return 0;
}
]])

try_compile(TENRYU_GPU_AWARE_MPI_COMPILE_OK
  SOURCE_FROM_CONTENT tenryu_gpu_aware_mpi_probe.cpp "${_tenryu_gpu_aware_mpi_probe_source}"
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  LINK_LIBRARIES MPI::MPI_CXX CUDA::cudart
  OUTPUT_VARIABLE TENRYU_GPU_AWARE_MPI_COMPILE_LOG
)

if(TENRYU_GPU_AWARE_MPI_COMPILE_OK)
  set(TENRYU_GPU_AWARE_MPI_COMPILE 1)
  message(STATUS "GPU-aware MPI compile probe: supported")
else()
  set(TENRYU_GPU_AWARE_MPI_COMPILE 0)
  message(STATUS "GPU-aware MPI compile probe: not supported (host-staging fallback)")
endif()

unset(_tenryu_gpu_aware_mpi_probe_source)
