#include "core/device_pack.hpp"

#include <cstddef>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"

namespace tenryu::core {
namespace {

constexpr int kBlockSize = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

__global__ void gather_fields_kernel(const double* s0,
                                     const double* s1,
                                     const double* s2,
                                     const double* s3,
                                     const double* s4,
                                     const double* s5,
                                     const double* s6,
                                     const double* s7,
                                     const int k,
                                     const int n,
                                     double* dst) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= k * n) {
    return;
  }
  const int a = idx / n;
  const int j = idx - a * n;
  switch (a) {
    case 0:
      dst[idx] = s0[j];
      break;
    case 1:
      dst[idx] = s1[j];
      break;
    case 2:
      dst[idx] = s2[j];
      break;
    case 3:
      dst[idx] = s3[j];
      break;
    case 4:
      dst[idx] = s4[j];
      break;
    case 5:
      dst[idx] = s5[j];
      break;
    case 6:
      dst[idx] = s6[j];
      break;
    case 7:
      dst[idx] = s7[j];
      break;
  }
}

__global__ void scatter_fields_kernel(const double* src,
                                      const int k,
                                      const int n,
                                      double* dst0,
                                      double* dst1,
                                      double* dst2,
                                      double* dst3,
                                      double* dst4,
                                      double* dst5,
                                      double* dst6,
                                      double* dst7) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= k * n) {
    return;
  }
  const int a = idx / n;
  const int j = idx - a * n;
  switch (a) {
    case 0:
      dst0[j] = src[idx];
      break;
    case 1:
      dst1[j] = src[idx];
      break;
    case 2:
      dst2[j] = src[idx];
      break;
    case 3:
      dst3[j] = src[idx];
      break;
    case 4:
      dst4[j] = src[idx];
      break;
    case 5:
      dst5[j] = src[idx];
      break;
    case 6:
      dst6[j] = src[idx];
      break;
    case 7:
      dst7[j] = src[idx];
      break;
  }
}

__global__ void gather_two_scalars_kernel(const double* a,
                                          const double* b,
                                          double* dst) {
  const int idx = threadIdx.x;
  if (idx == 0) {
    dst[0] = *a;
  } else if (idx == 1) {
    dst[1] = *b;
  }
}

}  // namespace

void pack_pull_fields(const double* const* d_srcs,
                      const int k,
                      const int n,
                      double* h_dst,
                      const char* tag) {
  TENRYU_ASSERT(k >= 1 && k <= 8,
                "pack_pull_fields requires 1 <= k <= 8");
  const int total = k * n;
  auto* stage = static_cast<double*>(device_scratch_acquire(
      tag, static_cast<std::size_t>(total) * sizeof(double)));
  const double* srcs[8] = {};
  for (int i = 0; i < k; ++i) {
    srcs[i] = d_srcs[i];
  }
  const int grid = (total + kBlockSize - 1) / kBlockSize;
  gather_fields_kernel<<<grid, kBlockSize>>>(srcs[0], srcs[1], srcs[2],
                                              srcs[3], srcs[4], srcs[5],
                                              srcs[6], srcs[7], k, n, stage);
  cuda_check(cudaGetLastError(), "pack_pull_fields gather launch failed");
  cuda_check(cudaMemcpy(h_dst,
                        stage,
                        static_cast<std::size_t>(total) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "pack_pull_fields readback failed");
}

void pack_push_fields(const double* h_src,
                      const int k,
                      const int n,
                      double* const* d_dsts,
                      const char* tag) {
  TENRYU_ASSERT(k >= 1 && k <= 8,
                "pack_push_fields requires 1 <= k <= 8");
  const int total = k * n;
  auto* stage = static_cast<double*>(device_scratch_acquire(
      tag, static_cast<std::size_t>(total) * sizeof(double)));
  cuda_check(cudaMemcpy(stage,
                        h_src,
                        static_cast<std::size_t>(total) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "pack_push_fields upload failed");
  double* dsts[8] = {};
  for (int i = 0; i < k; ++i) {
    dsts[i] = d_dsts[i];
  }
  const int grid = (total + kBlockSize - 1) / kBlockSize;
  scatter_fields_kernel<<<grid, kBlockSize>>>(stage, k, n, dsts[0], dsts[1],
                                               dsts[2], dsts[3], dsts[4],
                                               dsts[5], dsts[6], dsts[7]);
  cuda_check(cudaGetLastError(), "pack_push_fields scatter launch failed");
}

void pull_two_scalars(const double* d_a,
                      const double* d_b,
                      double* h_out2,
                      const char* tag) {
  auto* stage = static_cast<double*>(
      device_scratch_acquire(tag, 2 * sizeof(double)));
  gather_two_scalars_kernel<<<1, 2>>>(d_a, d_b, stage);
  cuda_check(cudaGetLastError(), "pull_two_scalars gather launch failed");
  cuda_check(cudaMemcpy(h_out2,
                        stage,
                        2 * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "pull_two_scalars readback failed");
}

}  // namespace tenryu::core
