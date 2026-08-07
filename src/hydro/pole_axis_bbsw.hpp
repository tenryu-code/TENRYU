#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tenryu::hydro::pole_axis_bbsw {

constexpr const char* kEnvEnable = "TENRYU_I1B_POLE_AXIS_BBSW";
constexpr double kHardGapFp = 64.0;
constexpr double kAxisContactC = 0.5;

struct PlanarAreaVector {
  double r = 0.0;
  double z = 0.0;
};

struct PlanarEndpointAreaVectors {
  PlanarAreaVector node0;
  PlanarAreaVector node1;
};

inline double hard_gap(const double s0, const double s1) {
  constexpr double eps = std::numeric_limits<double>::epsilon();
  const double L = std::max({std::fabs(s0), std::fabs(s1), 1.0e-300});
  return kHardGapFp * eps * L;
}

inline double triangle_area2(const double r0,
                             const double z0,
                             const double r1,
                             const double z1,
                             const double r2,
                             const double z2) {
  return r0 * z1 - r1 * z0 + r1 * z2 - r2 * z1 + r2 * z0 - r0 * z2;
}

inline void corner_vectors(const double* r,
                           const double* z,
                           const int nverts,
                           const double orientation_sign,
                           double* cvec_r,
                           double* cvec_z) {
  for (int k = 0; k < nverts; ++k) {
    const int km1 = (k + nverts - 1) % nverts;
    const int kp1 = (k + 1) % nverts;
    cvec_r[k] = 0.5 * orientation_sign * (z[kp1] - z[km1]);
    cvec_z[k] = 0.5 * orientation_sign * (r[km1] - r[kp1]);
  }
}

inline void corner_areas(const double* r,
                         const double* z,
                         const int nverts,
                         const double orientation_sign,
                         double* area) {
  if (nverts <= 0) {
    return;
  }
  double rc = 0.0;
  double zc = 0.0;
  for (int k = 0; k < nverts; ++k) {
    rc += r[k];
    zc += z[k];
  }
  rc /= static_cast<double>(nverts);
  zc /= static_cast<double>(nverts);

  if (nverts == 4) {
    double edge_area[4] = {0.0, 0.0, 0.0, 0.0};
    for (int k = 0; k < 4; ++k) {
      const int kp1 = (k + 1) & 3;
      const double a2 =
          triangle_area2(r[k], z[k], r[kp1], z[kp1], rc, zc);
      edge_area[k] = std::fabs(0.5 * orientation_sign * a2);
    }
    area[0] = (5.0 * edge_area[3] + 5.0 * edge_area[0] +
               edge_area[1] + edge_area[2]) /
              12.0;
    area[1] = (edge_area[3] + 5.0 * edge_area[0] +
               5.0 * edge_area[1] + edge_area[2]) /
              12.0;
    area[2] = (edge_area[3] + edge_area[0] +
               5.0 * edge_area[1] + 5.0 * edge_area[2]) /
              12.0;
    area[3] = (5.0 * edge_area[3] + edge_area[0] +
               edge_area[1] + 5.0 * edge_area[2]) /
              12.0;
    return;
  }

  double area2 = 0.0;
  for (int k = 0; k < nverts; ++k) {
    const int kp1 = (k + 1) % nverts;
    area2 += r[k] * z[kp1] - r[kp1] * z[k];
  }
  const double total_area = std::fabs(0.5 * orientation_sign * area2);
  const double uniform = total_area / static_cast<double>(nverts);
  for (int k = 0; k < nverts; ++k) {
    area[k] = uniform;
  }
}

inline PlanarEndpointAreaVectors outer_boundary_endpoint_area_vectors(
    const double* x_r,
    const double* x_z,
    const int nr,
    const int nz,
    const int j) {
  const int stride = nz + 1;
  const int n0 = nr * stride + j;
  const int n1 = nr * stride + (j + 1);
  const int n_inner0 = (nr - 1) * stride + j;
  const int n_inner1 = (nr - 1) * stride + (j + 1);
  const double r0 = x_r[n0];
  const double z0 = x_z[n0];
  const double r1 = x_r[n1];
  const double z1 = x_z[n1];
  const double dr = r1 - r0;
  const double dz = z1 - z0;

  PlanarEndpointAreaVectors areas{};
  areas.node0.r = 0.5 * dz;
  areas.node0.z = -0.5 * dr;
  areas.node1.r = 0.5 * dz;
  areas.node1.z = -0.5 * dr;

  const double face_r = 0.5 * (r0 + r1);
  const double face_z = 0.5 * (z0 + z1);
  const double cell_r = 0.25 * (x_r[n_inner0] + r0 + r1 + x_r[n_inner1]);
  const double cell_z = 0.25 * (x_z[n_inner0] + z0 + z1 + x_z[n_inner1]);
  const double outward_dot =
      dz * (face_r - cell_r) - dr * (face_z - cell_z);
  if (outward_dot < 0.0) {
    areas.node0.r = -areas.node0.r;
    areas.node0.z = -areas.node0.z;
    areas.node1.r = -areas.node1.r;
    areas.node1.z = -areas.node1.z;
  }
  return areas;
}

inline std::vector<double> weighted_isotonic_non_decreasing(
    const std::vector<double>& target,
    const std::vector<double>& weight) {
  const std::size_t n = target.size();
  std::vector<double> out(n, 0.0);
  if (n == 0U) {
    return out;
  }

  struct Block {
    std::size_t begin = 0U;
    std::size_t end = 0U;
    long double w = 0.0L;
    long double wx = 0.0L;
  };
  std::vector<Block> blocks;
  blocks.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double wi = (std::isfinite(weight[i]) && weight[i] > 0.0)
                          ? weight[i]
                          : 1.0e-300;
    const double xi = std::isfinite(target[i]) ? target[i] : 0.0;
    blocks.push_back(Block{i, i + 1U, wi, wi * static_cast<long double>(xi)});
    while (blocks.size() >= 2U) {
      Block& b = blocks.back();
      Block& a = blocks[blocks.size() - 2U];
      const long double ma = a.wx / a.w;
      const long double mb = b.wx / b.w;
      if (ma <= mb) {
        break;
      }
      a.end = b.end;
      a.w += b.w;
      a.wx += b.wx;
      blocks.pop_back();
    }
  }

  for (const Block& block : blocks) {
    const double value = static_cast<double>(block.wx / block.w);
    for (std::size_t i = block.begin; i < block.end; ++i) {
      out[i] = value;
    }
  }
  return out;
}

inline std::vector<double> project_axis_positions_with_hard_gaps(
    const std::vector<double>& s_hat,
    const std::vector<double>& weight,
    const std::vector<double>& delta) {
  const std::size_t n = s_hat.size();
  std::vector<double> offset(n, 0.0);
  for (std::size_t i = 1; i < n; ++i) {
    offset[i] = offset[i - 1U] + delta[i - 1U];
  }
  std::vector<double> shifted(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    shifted[i] = s_hat[i] - offset[i];
  }
  std::vector<double> projected =
      weighted_isotonic_non_decreasing(shifted, weight);
  for (std::size_t i = 0; i < n; ++i) {
    projected[i] += offset[i];
  }
  return projected;
}

}  // namespace tenryu::hydro::pole_axis_bbsw
