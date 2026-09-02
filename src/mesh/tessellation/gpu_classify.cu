#include "mesh/tessellation/gpu_classify.hpp"

#include "mesh/tessellation/gpu_circumcenter.hpp"

#include "core/error.hpp"
#include "core/device_scratch.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>

namespace tenryu::mesh::tess {
namespace {

struct DApprox {
  double value;
  double bound;
};

inline constexpr double kApproxSlack =
    1.0 + 4.0 * 2.220446049250313e-16;
inline constexpr double kApproxEps = 2.220446049250313e-16;
inline constexpr double kClassifyContractionSafety = 2.0;

__forceinline__ __device__ DApprox d_approx_of_terms(
    const double* terms,
    const int count) {
  double value = 0.0;
  double magnitude = 0.0;
  for (int i = 0; i < count; ++i) {
    value += terms[i];
    magnitude += fabs(terms[i]);
  }
  const double term_count = static_cast<double>(count > 0 ? count - 1 : 0);
  const double bound =
      term_count * kApproxEps * magnitude *
      (1.0 + term_count * kApproxEps) * kApproxSlack * kApproxSlack;
  return DApprox{value, bound};
}

__forceinline__ __device__ DApprox d_approx_sub(const DApprox& a,
                                                 const DApprox& b) {
  const double value = a.value - b.value;
  const double bound =
      (a.bound + b.bound + kApproxEps * fabs(value)) * kApproxSlack;
  return DApprox{value, bound};
}

__forceinline__ __device__ DApprox d_approx_add(const DApprox& a,
                                                 const DApprox& b) {
  const double value = a.value + b.value;
  const double bound =
      (a.bound + b.bound + kApproxEps * fabs(value)) * kApproxSlack;
  return DApprox{value, bound};
}

__forceinline__ __device__ DApprox d_approx_mul(const DApprox& a,
                                                 const DApprox& b) {
  const double value = a.value * b.value;
  const double bound =
      (fabs(a.value) * b.bound + fabs(b.value) * a.bound +
       a.bound * b.bound + kApproxEps * fabs(value)) *
      kApproxSlack;
  return DApprox{value, bound};
}

__forceinline__ __device__ bool d_certain_sign(const DApprox a, int& sign) {
  if (isfinite(a.value) && isfinite(a.bound) &&
      fabs(a.value) > kClassifyContractionSafety * a.bound) {
    sign = a.value > 0.0 ? 1 : -1;
    return true;
  }
  return false;
}

__global__ void gpu_classify_kernel(
    const std::size_t n_nodes,
    const std::int32_t* node_rep_triangle,
    const double* x_terms,
    const std::int32_t* x_count,
    const double* y_terms,
    const std::int32_t* y_count,
    const double* w_terms,
    const std::int32_t* w_count,
    const std::uint8_t* fallback,
    const double* segments_rz,
    const std::size_t n_segments,
    std::uint8_t* status) {
  const std::size_t node =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (node >= n_nodes) {
    return;
  }

  const std::size_t triangle =
      static_cast<std::size_t>(node_rep_triangle[node]);
  if (fallback[triangle] != 0) {
    status[node] = 2;
    return;
  }

  const DApprox x = d_approx_of_terms(
      x_terms + triangle * kGpuCcXYCap, x_count[triangle]);
  const DApprox y = d_approx_of_terms(
      y_terms + triangle * kGpuCcXYCap, y_count[triangle]);
  const DApprox w = d_approx_of_terms(
      w_terms + triangle * kGpuCcWCap, w_count[triangle]);

  int winding_number = 0;
  for (std::size_t segment = 0; segment < n_segments; ++segment) {
    const double* endpoints = segments_rz + 4 * segment;
    const double a_r = endpoints[0];
    const double a_z = endpoints[1];
    const double b_r = endpoints[2];
    const double b_z = endpoints[3];

    int sign_a = 0;
    int sign_b = 0;
    const DApprox z_a =
        d_approx_sub(d_approx_mul(w, DApprox{a_z, 0.0}), y);
    const DApprox z_b =
        d_approx_sub(d_approx_mul(w, DApprox{b_z, 0.0}), y);
    if (!d_certain_sign(z_a, sign_a) ||
        !d_certain_sign(z_b, sign_b)) {
      status[node] = 2;
      return;
    }

    const double edge_r_value = b_r - a_r;
    const double edge_z_value = b_z - a_z;
    const DApprox edge_r{
        edge_r_value,
        kApproxEps * fabs(edge_r_value) * kApproxSlack};
    const DApprox edge_z{
        edge_z_value,
        kApproxEps * fabs(edge_z_value) * kApproxSlack};
    const DApprox point_r =
        d_approx_sub(x, d_approx_mul(w, DApprox{a_r, 0.0}));
    const DApprox point_z =
        d_approx_sub(y, d_approx_mul(w, DApprox{a_z, 0.0}));
    const DApprox determinant = d_approx_sub(
        d_approx_mul(edge_r, point_z),
        d_approx_mul(edge_z, point_r));
    int side = 0;
    if (!d_certain_sign(determinant, side)) {
      status[node] = 2;
      return;
    }

    if (sign_a <= 0 && sign_b > 0 && side > 0) {
      ++winding_number;
    } else if (sign_b <= 0 && sign_a > 0 && side < 0) {
      --winding_number;
    }
  }
  status[node] = winding_number != 0 ? 1 : 0;
}

void cuda_assert(const cudaError_t error, const char* message) {
  TENRYU_ASSERT(error == cudaSuccess, message);
}

}  // namespace

GpuClassifyView gpu_classify_batch(
    const std::size_t n_nodes,
    const std::int32_t* node_rep_triangle,
    const std::size_t n_triangles,
    const double* segments_rz,
    const std::size_t n_segments) {
  if (n_nodes == 0) {
    return {};
  }

  TENRYU_ASSERT(gpu_circumcenter_last_batch_count() == n_triangles,
                "tess_gpu_classify_stale_batch");

  const std::size_t xy_term_bytes =
      n_triangles * kGpuCcXYCap * sizeof(double);
  const std::size_t w_term_bytes =
      n_triangles * kGpuCcWCap * sizeof(double);
  const std::size_t count_bytes = n_triangles * sizeof(std::int32_t);
  const std::size_t fallback_bytes = n_triangles * sizeof(std::uint8_t);
  const std::size_t reps_bytes = n_nodes * sizeof(std::int32_t);
  const std::size_t segments_bytes = 4 * n_segments * sizeof(double);
  const std::size_t status_bytes = n_nodes * sizeof(std::uint8_t);

  auto* x_terms_d = static_cast<double*>(
      core::device_scratch_acquire("tess_gpu_cc_xterms_d", xy_term_bytes));
  auto* x_count_d = static_cast<std::int32_t*>(
      core::device_scratch_acquire("tess_gpu_cc_xcount_d", count_bytes));
  auto* y_terms_d = static_cast<double*>(
      core::device_scratch_acquire("tess_gpu_cc_yterms_d", xy_term_bytes));
  auto* y_count_d = static_cast<std::int32_t*>(
      core::device_scratch_acquire("tess_gpu_cc_ycount_d", count_bytes));
  auto* w_terms_d = static_cast<double*>(
      core::device_scratch_acquire("tess_gpu_cc_wterms_d", w_term_bytes));
  auto* w_count_d = static_cast<std::int32_t*>(
      core::device_scratch_acquire("tess_gpu_cc_wcount_d", count_bytes));
  auto* fallback_d = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "tess_gpu_cc_fallback_d", fallback_bytes));

  auto* reps_d = static_cast<std::int32_t*>(
      core::device_scratch_acquire("tess_gpu_cls_reps_d", reps_bytes));
  auto* segments_d = static_cast<double*>(
      core::device_scratch_acquire("tess_gpu_cls_segs_d", segments_bytes));
  auto* status_d = static_cast<std::uint8_t*>(
      core::device_scratch_acquire("tess_gpu_cls_status_d", status_bytes));
  auto* reps_h = static_cast<std::int32_t*>(
      core::host_pinned_scratch_acquire("tess_gpu_cls_reps_h", reps_bytes));
  auto* segments_h = static_cast<double*>(core::host_pinned_scratch_acquire(
      "tess_gpu_cls_segs_h", segments_bytes));
  auto* status_h = static_cast<std::uint8_t*>(core::host_pinned_scratch_acquire(
      "tess_gpu_cls_status_h", status_bytes));

  const auto gpu_begin = std::chrono::steady_clock::now();
  std::memcpy(reps_h, node_rep_triangle, reps_bytes);
  std::memcpy(segments_h, segments_rz, segments_bytes);
  cuda_assert(cudaMemcpy(reps_d, reps_h, reps_bytes, cudaMemcpyHostToDevice),
              "tess_gpu_cls_reps_h2d");
  cuda_assert(cudaMemcpy(segments_d, segments_h, segments_bytes,
                         cudaMemcpyHostToDevice),
              "tess_gpu_cls_segs_h2d");

  constexpr int kThreadsPerBlock = 128;
  const unsigned int blocks = static_cast<unsigned int>(
      (n_nodes + kThreadsPerBlock - 1) / kThreadsPerBlock);
  gpu_classify_kernel<<<blocks, kThreadsPerBlock>>>(
      n_nodes, reps_d, x_terms_d, x_count_d, y_terms_d, y_count_d,
      w_terms_d, w_count_d, fallback_d, segments_d, n_segments, status_d);
  cuda_assert(cudaGetLastError(), "tess_gpu_cls_kernel_launch");
  cuda_assert(cudaDeviceSynchronize(), "tess_gpu_cls_kernel_sync");
  cuda_assert(cudaMemcpy(status_h, status_d, status_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cls_status_d2h");
  const auto gpu_end = std::chrono::steady_clock::now();

  GpuClassifyView view;
  view.status = status_h;
  view.count = n_nodes;
  view.gpu_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_begin).count();
  for (std::size_t node = 0; node < n_nodes; ++node) {
    if (status_h[node] == 0) {
      ++view.certified_outside;
    } else if (status_h[node] == 1) {
      ++view.certified_inside;
    } else {
      ++view.uncertain;
    }
  }
  return view;
}

}  // namespace tenryu::mesh::tess
