#pragma once

#include <cstdint>

// The pump lookup of the port_section chi build: the intensity and the meridional angle alpha of
// the records of one table bin at the polar angle theta_p, by linear interpolation between the
// two records (not in a limiter zone) that bracket it. A bin's records are sorted by
// (theta, ray index) (build_s1_table_device, sector_ps::build_table), limiter records included.
// Returns false when the bin has no record outside the limiters or theta_p lies beyond its
// largest theta (the shadow).

namespace tenryu::laser::port_section {

// The interpolation once the bracketing records are known (shared by both searches below).
__device__ inline bool chi_lookup_from_records(const double* theta, const double* alpha,
                                               const double* power, const double* area,
                                               const std::uint8_t* in_limiter, const int hi,
                                               const double theta_p, double* intensity_out,
                                               double* alpha_out) {
  if (theta[hi] == theta_p) {
    *intensity_out = power[hi] / area[hi];
    *alpha_out = alpha[hi];
    return true;
  }
  int lo = hi - 1;
  while (in_limiter[lo] != 0) {
    --lo;
  }
  const double weight = (theta_p - theta[lo]) / (theta[hi] - theta[lo]);
  const double lower_intensity = power[lo] / area[lo];
  const double upper_intensity = power[hi] / area[hi];
  *intensity_out = lower_intensity + weight * (upper_intensity - lower_intensity);
  *alpha_out = alpha[lo] + weight * (alpha[hi] - alpha[lo]);
  return true;
}

// The first record of the bin outside the limiters and the bin's end; false when the bin has no
// record outside the limiters or theta_p is beyond the last such record.
__device__ inline bool chi_lookup_ends(const int* offsets, const double* theta,
                                       const std::uint8_t* in_limiter, const int bin,
                                       const double theta_p, int* first_out, int* end_out) {
  const int begin = offsets[bin];
  const int end = offsets[bin + 1];
  int first = begin;
  while (first < end && in_limiter[first] != 0) {
    ++first;
  }
  if (first == end) {
    return false;
  }
  int last = end - 1;
  while (last >= first && in_limiter[last] != 0) {
    --last;
  }
  if (theta_p > theta[last]) {
    return false;
  }
  *first_out = first;
  *end_out = end;
  return true;
}

// The search of the reference kernel: the upper record by a scan from the first.
__device__ inline bool chi_lookup_linear(const int* offsets, const double* theta,
                                         const double* alpha, const double* power,
                                         const double* area, const std::uint8_t* in_limiter,
                                         const int bin, const double theta_p,
                                         double* intensity_out, double* alpha_out) {
  int first = 0;
  int end = 0;
  if (!chi_lookup_ends(offsets, theta, in_limiter, bin, theta_p, &first, &end)) {
    return false;
  }
  if (theta_p <= theta[first]) {
    *intensity_out = power[first] / area[first];
    *alpha_out = alpha[first];
    return true;
  }
  int hi = first + 1;
  while (hi < end && (in_limiter[hi] != 0 || theta[hi] < theta_p)) {
    ++hi;
  }
  if (hi == end) {
    return false;
  }
  return chi_lookup_from_records(theta, alpha, power, area, in_limiter, hi, theta_p,
                                 intensity_out, alpha_out);
}

// The same lookup with the upper record found by bisection: in [first + 1, end) the records with
// theta < theta_p come first (sorted bin), so the first record at or above theta_p is a lower
// bound; the scan's record is the first one from there outside the limiters. Same records,
// same arithmetic, same result.
__device__ inline bool chi_lookup(const int* offsets, const double* theta, const double* alpha,
                                  const double* power, const double* area,
                                  const std::uint8_t* in_limiter, const int bin,
                                  const double theta_p, double* intensity_out,
                                  double* alpha_out) {
  int first = 0;
  int end = 0;
  if (!chi_lookup_ends(offsets, theta, in_limiter, bin, theta_p, &first, &end)) {
    return false;
  }
  if (theta_p <= theta[first]) {
    *intensity_out = power[first] / area[first];
    *alpha_out = alpha[first];
    return true;
  }
  int lo = first + 1;
  int count = end - lo;
  while (count > 0) {
    const int step = count / 2;
    const int mid = lo + step;
    if (theta[mid] < theta_p) {
      lo = mid + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  int hi = lo;
  while (hi < end && in_limiter[hi] != 0) {
    ++hi;
  }
  if (hi == end) {
    return false;
  }
  return chi_lookup_from_records(theta, alpha, power, area, in_limiter, hi, theta_p,
                                 intensity_out, alpha_out);
}

}  // namespace tenryu::laser::port_section
