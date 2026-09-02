#include "mesh/tessellation/gpu_circumcenter.hpp"

#include "core/error.hpp"
#include "core/device_scratch.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstring>

namespace tenryu::mesh::tess {
namespace {

inline constexpr int kFixedExpCap = 64;
std::size_t g_last_batch_count = 0;

struct FixedExp {
  double terms[kFixedExpCap];
  int size = 0;
};

__forceinline__ __device__ void fx_copy(const FixedExp& src,
                                        FixedExp& dst) {
  dst.size = src.size;
  for (int i = 0; i < src.size; ++i) {
    dst.terms[i] = src.terms[i];
  }
}

__forceinline__ __device__ void fx_push(FixedExp& expansion,
                                        const double value,
                                        bool& overflow) {
  if (expansion.size == kFixedExpCap) {
    overflow = true;
    return;
  }
  expansion.terms[expansion.size++] = value;
}

__forceinline__ __device__ void two_sum(const double a,
                                        const double b,
                                        double& sum,
                                        double& error) {
  sum = a + b;
  const double b_virtual = sum - a;
  const double a_virtual = sum - b_virtual;
  const double b_roundoff = b - b_virtual;
  const double a_roundoff = a - a_virtual;
  error = a_roundoff + b_roundoff;
}

__forceinline__ __device__ void two_diff(const double a,
                                         const double b,
                                         double& difference,
                                         double& error) {
  difference = a - b;
  const double b_virtual = a - difference;
  const double a_virtual = difference + b_virtual;
  const double b_roundoff = b_virtual - b;
  const double a_roundoff = a - a_virtual;
  error = a_roundoff + b_roundoff;
}

__forceinline__ __device__ void two_product(const double a,
                                            const double b,
                                            double& product,
                                            double& error) {
  product = a * b;
  error = fma(a, b, -product);
}

__forceinline__ __device__ void append_nonzero(FixedExp& expansion,
                                               const double value,
                                               bool& overflow) {
  if (value != 0.0) {
    fx_push(expansion, value, overflow);
  }
}

__forceinline__ __device__ void fx_sum(const FixedExp& lhs,
                                       const FixedExp& rhs,
                                       FixedExp& out,
                                       bool& overflow) {
  if (overflow) {
    out.size = 0;
    return;
  }
  if (lhs.size == 0) {
    fx_copy(rhs, out);
    return;
  }
  if (rhs.size == 0) {
    fx_copy(lhs, out);
    return;
  }

  out.size = 0;
  int lhs_index = 0;
  int rhs_index = 0;
  double accumulator = 0.0;
  if (rhs_index == rhs.size ||
      (lhs_index != lhs.size &&
       fabs(lhs.terms[lhs_index]) <= fabs(rhs.terms[rhs_index]))) {
    accumulator = lhs.terms[lhs_index++];
  } else {
    accumulator = rhs.terms[rhs_index++];
  }

  while (lhs_index != lhs.size || rhs_index != rhs.size) {
    double component = 0.0;
    if (rhs_index == rhs.size ||
        (lhs_index != lhs.size &&
         fabs(lhs.terms[lhs_index]) <= fabs(rhs.terms[rhs_index]))) {
      component = lhs.terms[lhs_index++];
    } else {
      component = rhs.terms[rhs_index++];
    }
    double sum = 0.0;
    double error = 0.0;
    two_sum(accumulator, component, sum, error);
    append_nonzero(out, error, overflow);
    if (overflow) {
      out.size = 0;
      return;
    }
    accumulator = sum;
  }
  if (accumulator != 0.0 || out.size == 0) {
    fx_push(out, accumulator, overflow);
  }
}

__forceinline__ __device__ void fx_negate(FixedExp& expansion) {
  for (int i = 0; i < expansion.size; ++i) {
    expansion.terms[i] = -expansion.terms[i];
  }
}

__forceinline__ __device__ void fx_difference(const FixedExp& lhs,
                                              const FixedExp& rhs,
                                              FixedExp& out,
                                              FixedExp* scratch,
                                              bool& overflow) {
  // scratch[0]: negated.
  FixedExp& negated = scratch[0];
  fx_copy(rhs, negated);
  fx_negate(negated);
  fx_sum(lhs, negated, out, overflow);
}

__forceinline__ __device__ void fx_exact_difference(const double a,
                                                    const double b,
                                                    FixedExp& out,
                                                    bool& overflow) {
  double difference = 0.0;
  double error = 0.0;
  two_diff(a, b, difference, error);
  out.size = 0;
  append_nonzero(out, error, overflow);
  if (overflow) {
    out.size = 0;
    return;
  }
  if (difference != 0.0 || out.size == 0) {
    fx_push(out, difference, overflow);
  }
}

__forceinline__ __device__ void fx_scale(const FixedExp& expansion,
                                         const double scale,
                                         FixedExp& out,
                                         FixedExp* scratch,
                                         bool& overflow) {
  // scratch[0..1]: result ping-pong; scratch[2]: term.
  scratch[0].size = 0;
  scratch[1].size = 0;
  int current = 0;
  for (int i = 0; i < expansion.size; ++i) {
    double product = 0.0;
    double error = 0.0;
    two_product(expansion.terms[i], scale, product, error);
    FixedExp& term = scratch[2];
    term.size = 0;
    append_nonzero(term, error, overflow);
    if (overflow) {
      out.size = 0;
      return;
    }
    if (product != 0.0 || term.size == 0) {
      fx_push(term, product, overflow);
    }
    if (overflow) {
      out.size = 0;
      return;
    }
    fx_sum(scratch[current], term, scratch[1 - current], overflow);
    if (overflow) {
      out.size = 0;
      return;
    }
    current = 1 - current;
  }
  fx_copy(scratch[current], out);
}

__forceinline__ __device__ void fx_product(const FixedExp& lhs,
                                           const FixedExp& rhs,
                                           FixedExp& out,
                                           FixedExp* scratch,
                                           bool& overflow) {
  // scratch[0..1]: result ping-pong; scratch[2]: scaled;
  // scratch[3..5]: fx_scale scratch.
  scratch[0].size = 0;
  scratch[1].size = 0;
  int current = 0;
  for (int i = 0; i < rhs.size; ++i) {
    FixedExp& scaled = scratch[2];
    fx_scale(lhs, rhs.terms[i], scaled, scratch + 3, overflow);
    if (overflow) {
      out.size = 0;
      return;
    }
    fx_sum(scratch[current], scaled, scratch[1 - current], overflow);
    if (overflow) {
      out.size = 0;
      return;
    }
    current = 1 - current;
  }
  fx_copy(scratch[current], out);
}

__forceinline__ __device__ void fx_lift(const FixedExp& x,
                                        const FixedExp& y,
                                        FixedExp& out,
                                        FixedExp* scratch,
                                        bool& overflow) {
  // scratch[0]: x_squared; scratch[1]: y_squared;
  // scratch[2..7]: fx_product scratch.
  FixedExp& x_squared = scratch[0];
  FixedExp& y_squared = scratch[1];
  fx_product(x, x, x_squared, scratch + 2, overflow);
  fx_product(y, y, y_squared, scratch + 2, overflow);
  fx_sum(x_squared, y_squared, out, overflow);
}

__forceinline__ __device__ void fx_cross(const FixedExp& ax,
                                         const FixedExp& ay,
                                         const FixedExp& bx,
                                         const FixedExp& by,
                                         FixedExp& out,
                                         FixedExp* scratch,
                                         bool& overflow) {
  // scratch[0]: ax_by; scratch[1]: ay_bx;
  // scratch[2..7]: fx_product scratch or fx_difference scratch.
  FixedExp& ax_by = scratch[0];
  FixedExp& ay_bx = scratch[1];
  fx_product(ax, by, ax_by, scratch + 2, overflow);
  fx_product(ay, bx, ay_bx, scratch + 2, overflow);
  fx_difference(ax_by, ay_bx, out, scratch + 2, overflow);
}

__forceinline__ __device__ int fx_sign(const FixedExp& expansion) {
  for (int i = expansion.size - 1; i >= 0; --i) {
    if (expansion.terms[i] > 0.0) {
      return 1;
    }
    if (expansion.terms[i] < 0.0) {
      return -1;
    }
  }
  return 0;
}

__forceinline__ __device__ void fx_from_scalar(FixedExp& out,
                                               const double value) {
  out.size = 1;
  out.terms[0] = value;
}

__forceinline__ __device__ bool fx_value(const FixedExp& expansion,
                                         double& value) {
  value = 0.0;
  for (int i = 0; i < expansion.size; ++i) {
    value += expansion.terms[i];
    if (!isfinite(value)) {
      return false;
    }
  }
  return true;
}

__forceinline__ __device__ void mark_fallback(const std::size_t triangle,
                                              std::int32_t* x_count,
                                              std::int32_t* y_count,
                                              std::int32_t* w_count,
                                              std::uint8_t* fallback) {
  x_count[triangle] = 0;
  y_count[triangle] = 0;
  w_count[triangle] = 0;
  fallback[triangle] = 1;
}

__global__ void gpu_circumcenter_kernel(
    const std::size_t n_triangles,
    const std::int32_t* sorted_site_triplets,
    const double* site_rz,
    double* x_terms,
    std::int32_t* x_count,
    double* y_terms,
    std::int32_t* y_count,
    double* w_terms,
    std::int32_t* w_count,
    double* constructed,
    std::uint8_t* fallback) {
  const std::size_t triangle =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (triangle >= n_triangles) {
    return;
  }

  const std::int32_t a_index = sorted_site_triplets[3 * triangle];
  const std::int32_t b_index = sorted_site_triplets[3 * triangle + 1];
  const std::int32_t c_index = sorted_site_triplets[3 * triangle + 2];
  const double a_r = site_rz[2 * static_cast<std::size_t>(a_index)];
  const double a_z = site_rz[2 * static_cast<std::size_t>(a_index) + 1];
  const double b_r = site_rz[2 * static_cast<std::size_t>(b_index)];
  const double b_z = site_rz[2 * static_cast<std::size_t>(b_index) + 1];
  const double c_r = site_rz[2 * static_cast<std::size_t>(c_index)];
  const double c_z = site_rz[2 * static_cast<std::size_t>(c_index) + 1];

  bool overflow = false;
  // 22 named slots + 8 helper-scratch slots. The single array is the
  // point: element addresses are explicit index arithmetic, so the
  // compiler cannot misallocate distinct expansions onto one slot
  // (measured failure mode of the aggregate-local layout).
  FixedExp ws[30];
  FixedExp* scratch = ws + 22;
  FixedExp& ar = ws[0];
  FixedExp& az = ws[1];
  FixedExp& br = ws[2];
  FixedExp& bz = ws[3];
  FixedExp& cr = ws[4];
  FixedExp& cz = ws[5];
  FixedExp& a_lift = ws[6];
  FixedExp& b_lift = ws[7];
  FixedExp& c_lift = ws[8];
  FixedExp& difference = ws[9];
  FixedExp& product0 = ws[10];
  FixedExp& product1 = ws[11];
  FixedExp& product2 = ws[12];
  FixedExp& sum01 = ws[13];
  FixedExp& numerator_r = ws[14];
  FixedExp& numerator_z = ws[15];
  FixedExp& ba_r = ws[16];
  FixedExp& ba_z = ws[17];
  FixedExp& ca_r = ws[18];
  FixedExp& ca_z = ws[19];
  FixedExp& cross = ws[20];
  FixedExp& denominator = ws[21];
  fx_from_scalar(ar, a_r);
  fx_from_scalar(az, a_z);
  fx_from_scalar(br, b_r);
  fx_from_scalar(bz, b_z);
  fx_from_scalar(cr, c_r);
  fx_from_scalar(cz, c_z);

  fx_lift(ar, az, a_lift, scratch, overflow);
  fx_lift(br, bz, b_lift, scratch, overflow);
  fx_lift(cr, cz, c_lift, scratch, overflow);
  if (overflow) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }

  fx_exact_difference(b_z, c_z, difference, overflow);
  fx_product(a_lift, difference, product0, scratch, overflow);
  fx_exact_difference(c_z, a_z, difference, overflow);
  fx_product(b_lift, difference, product1, scratch, overflow);
  fx_sum(product0, product1, sum01, overflow);
  fx_exact_difference(a_z, b_z, difference, overflow);
  fx_product(c_lift, difference, product2, scratch, overflow);
  fx_sum(sum01, product2, numerator_r, overflow);
  if (overflow) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }

  fx_exact_difference(c_r, b_r, difference, overflow);
  fx_product(a_lift, difference, product0, scratch, overflow);
  fx_exact_difference(a_r, c_r, difference, overflow);
  fx_product(b_lift, difference, product1, scratch, overflow);
  fx_sum(product0, product1, sum01, overflow);
  fx_exact_difference(b_r, a_r, difference, overflow);
  fx_product(c_lift, difference, product2, scratch, overflow);
  fx_sum(sum01, product2, numerator_z, overflow);
  if (overflow) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }

  fx_exact_difference(b_r, a_r, ba_r, overflow);
  fx_exact_difference(b_z, a_z, ba_z, overflow);
  fx_exact_difference(c_r, a_r, ca_r, overflow);
  fx_exact_difference(c_z, a_z, ca_z, overflow);
  fx_cross(ba_r, ba_z, ca_r, ca_z, cross, scratch, overflow);
  fx_scale(cross, 2.0, denominator, scratch, overflow);
  if (overflow) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }

  const int w_sign = fx_sign(denominator);
  if (w_sign == 0) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }
  if (w_sign < 0) {
    fx_negate(numerator_r);
    fx_negate(numerator_z);
    fx_negate(denominator);
  }

  double x_value = 0.0;
  double y_value = 0.0;
  double w_value = 0.0;
  if (!fx_value(numerator_r, x_value) ||
      !fx_value(numerator_z, y_value) ||
      !fx_value(denominator, w_value) || !(w_value > 0.0)) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }
  const double constructed_r = x_value / w_value;
  const double constructed_z = y_value / w_value;
  if (!isfinite(constructed_r) || !isfinite(constructed_z)) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }

  if (numerator_r.size > kGpuCcXYCap ||
      numerator_z.size > kGpuCcXYCap ||
      denominator.size > kGpuCcWCap || overflow) {
    mark_fallback(triangle, x_count, y_count, w_count, fallback);
    return;
  }

  const std::size_t x_offset = triangle * kGpuCcXYCap;
  const std::size_t y_offset = triangle * kGpuCcXYCap;
  const std::size_t w_offset = triangle * kGpuCcWCap;
  for (int i = 0; i < numerator_r.size; ++i) {
    x_terms[x_offset + static_cast<std::size_t>(i)] = numerator_r.terms[i];
  }
  for (int i = 0; i < numerator_z.size; ++i) {
    y_terms[y_offset + static_cast<std::size_t>(i)] = numerator_z.terms[i];
  }
  for (int i = 0; i < denominator.size; ++i) {
    w_terms[w_offset + static_cast<std::size_t>(i)] = denominator.terms[i];
  }
  x_count[triangle] = static_cast<std::int32_t>(numerator_r.size);
  y_count[triangle] = static_cast<std::int32_t>(numerator_z.size);
  w_count[triangle] = static_cast<std::int32_t>(denominator.size);
  constructed[2 * triangle] = constructed_r;
  constructed[2 * triangle + 1] = constructed_z;
  fallback[triangle] = 0;
}

void cuda_assert(const cudaError_t error, const char* message) {
  TENRYU_ASSERT(error == cudaSuccess, message);
}

}  // namespace

std::size_t gpu_circumcenter_last_batch_count() {
  return g_last_batch_count;
}

GpuCircumcenterBatchView gpu_circumcenter_batch(
    const std::size_t n_triangles,
    const std::int32_t* sorted_site_triplets,
    const double* site_rz,
    const std::size_t n_sites) {
  if (n_triangles == 0) {
    return {};
  }

  const std::size_t triangle_bytes =
      3 * n_triangles * sizeof(std::int32_t);
  const std::size_t site_bytes = 2 * n_sites * sizeof(double);
  const std::size_t xy_term_bytes =
      n_triangles * kGpuCcXYCap * sizeof(double);
  const std::size_t w_term_bytes =
      n_triangles * kGpuCcWCap * sizeof(double);
  const std::size_t count_bytes = n_triangles * sizeof(std::int32_t);
  const std::size_t constructed_bytes =
      2 * n_triangles * sizeof(double);
  const std::size_t fallback_bytes = n_triangles * sizeof(std::uint8_t);

  auto* triangles_d = static_cast<std::int32_t*>(
      core::device_scratch_acquire("tess_gpu_cc_tris_d", triangle_bytes));
  auto* sites_d = static_cast<double*>(
      core::device_scratch_acquire("tess_gpu_cc_sites_d", site_bytes));
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
  auto* constructed_d = static_cast<double*>(core::device_scratch_acquire(
      "tess_gpu_cc_constructed_d", constructed_bytes));
  auto* fallback_d = static_cast<std::uint8_t*>(core::device_scratch_acquire(
      "tess_gpu_cc_fallback_d", fallback_bytes));

  auto* triangles_h = static_cast<std::int32_t*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_tris_h", triangle_bytes));
  auto* sites_h = static_cast<double*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_sites_h", site_bytes));
  auto* x_terms_h = static_cast<double*>(core::host_pinned_scratch_acquire(
      "tess_gpu_cc_xterms_h", xy_term_bytes));
  auto* x_count_h = static_cast<std::int32_t*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_xcount_h", count_bytes));
  auto* y_terms_h = static_cast<double*>(core::host_pinned_scratch_acquire(
      "tess_gpu_cc_yterms_h", xy_term_bytes));
  auto* y_count_h = static_cast<std::int32_t*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_ycount_h", count_bytes));
  auto* w_terms_h = static_cast<double*>(core::host_pinned_scratch_acquire(
      "tess_gpu_cc_wterms_h", w_term_bytes));
  auto* w_count_h = static_cast<std::int32_t*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_wcount_h", count_bytes));
  auto* constructed_h = static_cast<double*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_constructed_h",
                                        constructed_bytes));
  auto* fallback_h = static_cast<std::uint8_t*>(
      core::host_pinned_scratch_acquire("tess_gpu_cc_fallback_h",
                                        fallback_bytes));

  const auto gpu_begin = std::chrono::steady_clock::now();
  std::memcpy(triangles_h, sorted_site_triplets, triangle_bytes);
  std::memcpy(sites_h, site_rz, site_bytes);
  cuda_assert(cudaMemcpy(triangles_d, triangles_h, triangle_bytes,
                         cudaMemcpyHostToDevice),
              "tess_gpu_cc_tris_h2d");
  cuda_assert(cudaMemcpy(sites_d, sites_h, site_bytes,
                         cudaMemcpyHostToDevice),
              "tess_gpu_cc_sites_h2d");

  constexpr int kThreadsPerBlock = 128;
  const unsigned int blocks = static_cast<unsigned int>(
      (n_triangles + kThreadsPerBlock - 1) / kThreadsPerBlock);
  gpu_circumcenter_kernel<<<blocks, kThreadsPerBlock>>>(
      n_triangles, triangles_d, sites_d, x_terms_d, x_count_d, y_terms_d,
      y_count_d, w_terms_d, w_count_d, constructed_d, fallback_d);
  cuda_assert(cudaGetLastError(), "tess_gpu_cc_kernel_launch");
  cuda_assert(cudaDeviceSynchronize(), "tess_gpu_cc_kernel_sync");
  const auto gpu_end = std::chrono::steady_clock::now();

  const auto d2h_begin = std::chrono::steady_clock::now();
  cuda_assert(cudaMemcpy(x_terms_h, x_terms_d, xy_term_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_xterms_d2h");
  cuda_assert(cudaMemcpy(x_count_h, x_count_d, count_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_xcount_d2h");
  cuda_assert(cudaMemcpy(y_terms_h, y_terms_d, xy_term_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_yterms_d2h");
  cuda_assert(cudaMemcpy(y_count_h, y_count_d, count_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_ycount_d2h");
  cuda_assert(cudaMemcpy(w_terms_h, w_terms_d, w_term_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_wterms_d2h");
  cuda_assert(cudaMemcpy(w_count_h, w_count_d, count_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_wcount_d2h");
  cuda_assert(cudaMemcpy(constructed_h, constructed_d, constructed_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_constructed_d2h");
  cuda_assert(cudaMemcpy(fallback_h, fallback_d, fallback_bytes,
                         cudaMemcpyDeviceToHost),
              "tess_gpu_cc_fallback_d2h");
  const auto d2h_end = std::chrono::steady_clock::now();

  GpuCircumcenterBatchView view;
  view.x_terms = x_terms_h;
  view.x_count = x_count_h;
  view.y_terms = y_terms_h;
  view.y_count = y_count_h;
  view.w_terms = w_terms_h;
  view.w_count = w_count_h;
  view.constructed = constructed_h;
  view.fallback = fallback_h;
  view.count = n_triangles;
  view.gpu_ms =
      std::chrono::duration<double, std::milli>(gpu_end - gpu_begin).count();
  view.d2h_ms =
      std::chrono::duration<double, std::milli>(d2h_end - d2h_begin).count();
  g_last_batch_count = n_triangles;
  return view;
}

}  // namespace tenryu::mesh::tess
