#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "core/error.hpp"

namespace tenryu::mesh::assembly {

// An ordered open polyline of N+1 nodes along one patch boundary.
struct BoundaryChain {
  std::vector<double> r;
  std::vector<double> z;  // Same size >= 2.
};

// Structured patch of (ni+1) x (nj+1) nodes; node (i,j) at index
// i*(nj+1)+j.
struct TfiPatch {
  int ni = 0;
  int nj = 0;
  std::vector<double> r;
  std::vector<double> z;
};

// Discrete Gordon–Hall transfinite interpolation with index-uniform blending
// (the boundary chains carry all grading; xi_i = i/ni, eta_j = j/nj):
//   x(i,j) = (1-eta)*B(i) + eta*T(i) + (1-xi)*L(j) + xi*R(j)
//          - [(1-xi)(1-eta)*P00 + xi(1-eta)*P10
//             + (1-xi)eta*P01 + xi*eta*P11]
// Chain layout: bottom = eta=0 (i=0..ni), top = eta=1 (i=0..ni),
//               left = xi=0 (j=0..nj), right = xi=1 (j=0..nj).
// Corner consistency is asserted with exact equality:
//   bottom[0]==left[0]==P00, bottom[ni]==right[0]==P10,
//   top[0]==left[nj]==P01,   top[ni]==right[nj]==P11.
// Boundary nodes of the output are copied verbatim from the chains.
inline TfiPatch tfi_patch(const BoundaryChain& bottom,
                          const BoundaryChain& top,
                          const BoundaryChain& left,
                          const BoundaryChain& right) {
  const auto validate_chain = [](const BoundaryChain& chain) {
    TENRYU_ASSERT(chain.r.size() == chain.z.size(),
                  "boundary coordinate arrays must have matching sizes");
    TENRYU_ASSERT(chain.r.size() >= 2U,
                  "boundary chains must contain at least two nodes");
    for (std::size_t k = 0; k < chain.r.size(); ++k) {
      TENRYU_ASSERT(std::isfinite(chain.r[k]) && std::isfinite(chain.z[k]),
                    "boundary coordinates must be finite");
    }
  };
  validate_chain(bottom);
  validate_chain(top);
  validate_chain(left);
  validate_chain(right);

  TENRYU_ASSERT(bottom.r.size() == top.r.size(),
                "bottom and top boundary sizes must match");
  TENRYU_ASSERT(left.r.size() == right.r.size(),
                "left and right boundary sizes must match");

  const std::size_t ni_size = bottom.r.size() - 1U;
  const std::size_t nj_size = left.r.size() - 1U;
  TENRYU_ASSERT(ni_size <= static_cast<std::size_t>(
                               std::numeric_limits<int>::max()),
                "patch i interval count exceeds int range");
  TENRYU_ASSERT(nj_size <= static_cast<std::size_t>(
                               std::numeric_limits<int>::max()),
                "patch j interval count exceeds int range");
  const int ni = static_cast<int>(ni_size);
  const int nj = static_cast<int>(nj_size);

  TENRYU_ASSERT(bottom.r.front() == left.r.front() &&
                    bottom.z.front() == left.z.front(),
                "bottom-left corner coordinates must match exactly");
  TENRYU_ASSERT(bottom.r.back() == right.r.front() &&
                    bottom.z.back() == right.z.front(),
                "bottom-right corner coordinates must match exactly");
  TENRYU_ASSERT(top.r.front() == left.r.back() &&
                    top.z.front() == left.z.back(),
                "top-left corner coordinates must match exactly");
  TENRYU_ASSERT(top.r.back() == right.r.back() &&
                    top.z.back() == right.z.back(),
                "top-right corner coordinates must match exactly");

  const std::size_t node_count =
      (ni_size + 1U) * (nj_size + 1U);
  TfiPatch patch;
  patch.ni = ni;
  patch.nj = nj;
  patch.r.resize(node_count);
  patch.z.resize(node_count);

  const double p00_r = bottom.r.front();
  const double p00_z = bottom.z.front();
  const double p10_r = bottom.r.back();
  const double p10_z = bottom.z.back();
  const double p01_r = top.r.front();
  const double p01_z = top.z.front();
  const double p11_r = top.r.back();
  const double p11_z = top.z.back();

  for (int i = 0; i <= ni; ++i) {
    const std::size_t i_index = static_cast<std::size_t>(i);
    const double xi = static_cast<double>(i) / static_cast<double>(ni);
    for (int j = 0; j <= nj; ++j) {
      const std::size_t j_index = static_cast<std::size_t>(j);
      const std::size_t index =
          i_index * (nj_size + 1U) + j_index;

      if (j == 0) {
        patch.r[index] = bottom.r[i_index];
        patch.z[index] = bottom.z[i_index];
      } else if (j == nj) {
        patch.r[index] = top.r[i_index];
        patch.z[index] = top.z[i_index];
      } else if (i == 0) {
        patch.r[index] = left.r[j_index];
        patch.z[index] = left.z[j_index];
      } else if (i == ni) {
        patch.r[index] = right.r[j_index];
        patch.z[index] = right.z[j_index];
      } else {
        const double eta =
            static_cast<double>(j) / static_cast<double>(nj);
        const double corner_r =
            (1.0 - xi) * (1.0 - eta) * p00_r +
            xi * (1.0 - eta) * p10_r +
            (1.0 - xi) * eta * p01_r + xi * eta * p11_r;
        const double corner_z =
            (1.0 - xi) * (1.0 - eta) * p00_z +
            xi * (1.0 - eta) * p10_z +
            (1.0 - xi) * eta * p01_z + xi * eta * p11_z;
        patch.r[index] =
            (1.0 - eta) * bottom.r[i_index] + eta * top.r[i_index] +
            (1.0 - xi) * left.r[j_index] + xi * right.r[j_index] -
            corner_r;
        patch.z[index] =
            (1.0 - eta) * bottom.z[i_index] + eta * top.z[i_index] +
            (1.0 - xi) * left.z[j_index] + xi * right.z[j_index] -
            corner_z;
      }
    }
  }

  return patch;
}

// Minimum signed corner-triangle Jacobian over all cells of the patch, in the
// planar (r,z) metric. At corner k with neighbors k+1 and k-1:
//   J_k = (x_{k+1}-x_k) x (x_{k-1}-x_k).
inline double min_corner_jacobian(const TfiPatch& patch) {
  TENRYU_ASSERT(patch.ni >= 1 && patch.nj >= 1,
                "patch interval counts must be positive");
  const std::size_t ni_size = static_cast<std::size_t>(patch.ni);
  const std::size_t nj_size = static_cast<std::size_t>(patch.nj);
  const std::size_t node_count =
      (ni_size + 1U) * (nj_size + 1U);
  TENRYU_ASSERT(patch.r.size() == node_count &&
                    patch.z.size() == node_count,
                "patch coordinate arrays must match its dimensions");
  for (std::size_t k = 0; k < node_count; ++k) {
    TENRYU_ASSERT(std::isfinite(patch.r[k]) && std::isfinite(patch.z[k]),
                  "patch coordinates must be finite");
  }

  const auto index = [nj_size](const int i, const int j) {
    return static_cast<std::size_t>(i) * (nj_size + 1U) +
           static_cast<std::size_t>(j);
  };
  double minimum = std::numeric_limits<double>::infinity();
  for (int i = 0; i < patch.ni; ++i) {
    for (int j = 0; j < patch.nj; ++j) {
      const std::size_t corners[4] = {
          index(i, j), index(i + 1, j), index(i + 1, j + 1),
          index(i, j + 1)};
      for (int k = 0; k < 4; ++k) {
        const std::size_t current = corners[k];
        const std::size_t next = corners[(k + 1) % 4];
        const std::size_t previous = corners[(k + 3) % 4];
        const double next_r = patch.r[next] - patch.r[current];
        const double next_z = patch.z[next] - patch.z[current];
        const double previous_r = patch.r[previous] - patch.r[current];
        const double previous_z = patch.z[previous] - patch.z[current];
        const double jacobian =
            next_r * previous_z - next_z * previous_r;
        minimum = std::min(minimum, jacobian);
      }
    }
  }
  return minimum;
}

// Corner where two straight offset families meet at point P (v1: straight
// walls; curved walls arrive with the junction templates). Family X is the
// stack of lines parallel to its wall at signed depths measured along its
// UNIT outward normal n_X:  offset-X line at depth d = { x : (x-P).n_X = d }.
struct OffsetCornerSpec {
  double corner_r = 0.0, corner_z = 0.0;      // P
  double normal_a_r = 0.0, normal_a_z = 0.0;  // unit normal of family A
  double normal_b_r = 0.0, normal_b_z = 0.0;  // unit normal of family B
  std::vector<double> depths_a;               // ascending, depths_a[0] == 0.0 exactly
  std::vector<double> depths_b;               // ascending, depths_b[0] == 0.0 exactly
};

// Node (k,l) of the returned patch (ni = depths_a.size()-1 as the i/xi
// direction, nj = depths_b.size()-1 as j/eta) is the intersection of
// offset-A line k with offset-B line l, by Cramer closed form:
//   det = nA_r*nB_z - nA_z*nB_r          (assert |det| >= min_sin)
//   C_r = P_r + (dA*nB_z - dB*nA_z)/det
//   C_z = P_z + (dB*nA_r - dA*nB_r)/det
// Node (0,0) is P VERBATIM (copied, not computed). Row k with dA=0 lies on
// wall A; column l with dB=0 lies on wall B (within FP).
// Patch orientation follows sign(det); pick the (A,B) handedness with det > 0 for a positively-oriented patch (or reverse afterwards).
// min_sin default = 0.199368 (sin 11.5 deg, the TIP_CORNER conditioning gate).
inline TfiPatch offset_corner_patch(const OffsetCornerSpec& spec,
                                    double min_sin = 0.199368) {
  TENRYU_ASSERT(std::isfinite(spec.corner_r) &&
                    std::isfinite(spec.corner_z),
                "offset corner coordinates must be finite");
  TENRYU_ASSERT(std::isfinite(spec.normal_a_r) &&
                    std::isfinite(spec.normal_a_z) &&
                    std::isfinite(spec.normal_b_r) &&
                    std::isfinite(spec.normal_b_z),
                "offset corner normals must be finite");
  TENRYU_ASSERT(
      std::abs(std::hypot(spec.normal_a_r, spec.normal_a_z) - 1.0) <=
          1.0e-12,
      "offset corner family A normal must be unit length to 1e-12");
  TENRYU_ASSERT(
      std::abs(std::hypot(spec.normal_b_r, spec.normal_b_z) - 1.0) <=
          1.0e-12,
      "offset corner family B normal must be unit length to 1e-12");
  TENRYU_ASSERT(spec.depths_a.size() >= 2U && spec.depths_b.size() >= 2U,
                "offset corner depth ladders must contain at least two nodes");
  TENRYU_ASSERT(spec.depths_a.front() == 0.0 &&
                    spec.depths_b.front() == 0.0,
                "offset corner depth ladders must start at exactly zero");
  for (std::size_t k = 0; k < spec.depths_a.size(); ++k) {
    TENRYU_ASSERT(std::isfinite(spec.depths_a[k]),
                  "offset corner family A depths must be finite");
    if (k > 0U) {
      TENRYU_ASSERT(spec.depths_a[k - 1U] < spec.depths_a[k],
                    "offset corner family A depths must be ascending");
    }
  }
  for (std::size_t l = 0; l < spec.depths_b.size(); ++l) {
    TENRYU_ASSERT(std::isfinite(spec.depths_b[l]),
                  "offset corner family B depths must be finite");
    if (l > 0U) {
      TENRYU_ASSERT(spec.depths_b[l - 1U] < spec.depths_b[l],
                    "offset corner family B depths must be ascending");
    }
  }
  TENRYU_ASSERT(std::isfinite(min_sin) && min_sin > 0.0,
                "offset corner min_sin must be positive and finite");

  const double det =
      spec.normal_a_r * spec.normal_b_z -
      spec.normal_a_z * spec.normal_b_r;
  TENRYU_ASSERT(std::abs(det) >= min_sin,
                "offset corner conditioning gate requires |det| >= min_sin");

  const std::size_t ni_size = spec.depths_a.size() - 1U;
  const std::size_t nj_size = spec.depths_b.size() - 1U;
  TENRYU_ASSERT(ni_size <= static_cast<std::size_t>(
                               std::numeric_limits<int>::max()),
                "offset corner i interval count exceeds int range");
  TENRYU_ASSERT(nj_size <= static_cast<std::size_t>(
                               std::numeric_limits<int>::max()),
                "offset corner j interval count exceeds int range");

  TfiPatch patch;
  patch.ni = static_cast<int>(ni_size);
  patch.nj = static_cast<int>(nj_size);
  const std::size_t node_count =
      (ni_size + 1U) * (nj_size + 1U);
  patch.r.resize(node_count);
  patch.z.resize(node_count);
  for (std::size_t k = 0; k <= ni_size; ++k) {
    for (std::size_t l = 0; l <= nj_size; ++l) {
      const std::size_t index = k * (nj_size + 1U) + l;
      if (k == 0U && l == 0U) {
        patch.r[index] = spec.corner_r;
        patch.z[index] = spec.corner_z;
      } else {
        const double depth_a = spec.depths_a[k];
        const double depth_b = spec.depths_b[l];
        patch.r[index] =
            spec.corner_r +
            (depth_a * spec.normal_b_z -
             depth_b * spec.normal_a_z) /
                det;
        patch.z[index] =
            spec.corner_z +
            (depth_b * spec.normal_a_r -
             depth_a * spec.normal_b_r) /
                det;
      }
    }
  }
  return patch;
}

// Fixed-sweep Jacobi Winslow smoothing of a structured patch. Nodes with
// pinned[i*(nj+1)+j] != 0 never move (boundaries are typically pinned by the
// caller). Interior update per sweep (double-buffered Jacobi — order-free,
// deterministic): with central differences at the current iterate,
//   x_xi  = 0.5*(x[i+1][j] - x[i-1][j])
//   x_eta = 0.5*(x[i][j+1] - x[i][j-1])
//   alpha = x_eta . x_eta, beta = x_xi . x_eta, gamma = x_xi . x_xi.
inline void smooth_patch_winslow(TfiPatch& patch,
                                 const std::vector<std::uint8_t>& pinned,
                                 const int n_sweeps,
                                 const double omega = 0.5) {
  TENRYU_ASSERT(patch.ni >= 1 && patch.nj >= 1,
                "patch interval counts must be positive");
  TENRYU_ASSERT(n_sweeps >= 0,
                "Winslow sweep count must be non-negative");
  TENRYU_ASSERT(std::isfinite(omega) && omega > 0.0 && omega < 2.0,
                "Winslow relaxation factor must be finite and in (0, 2)");

  const std::size_t ni_size = static_cast<std::size_t>(patch.ni);
  const std::size_t nj_size = static_cast<std::size_t>(patch.nj);
  const std::size_t node_count =
      (ni_size + 1U) * (nj_size + 1U);
  TENRYU_ASSERT(patch.r.size() == node_count &&
                    patch.z.size() == node_count,
                "patch coordinate arrays must match its dimensions");
  TENRYU_ASSERT(pinned.size() == node_count,
                "Winslow pin mask must match the patch dimensions");
  for (std::size_t k = 0; k < node_count; ++k) {
    TENRYU_ASSERT(std::isfinite(patch.r[k]) && std::isfinite(patch.z[k]),
                  "patch coordinates must be finite");
  }

  const auto index = [nj_size](const int i, const int j) {
    return static_cast<std::size_t>(i) * (nj_size + 1U) +
           static_cast<std::size_t>(j);
  };
  std::vector<double> next_r = patch.r;
  std::vector<double> next_z = patch.z;
  for (int sweep = 0; sweep < n_sweeps; ++sweep) {
    next_r = patch.r;
    next_z = patch.z;
    for (int i = 1; i < patch.ni; ++i) {
      for (int j = 1; j < patch.nj; ++j) {
        const std::size_t center = index(i, j);
        if (pinned[center] != 0U) {
          continue;
        }

        const std::size_t i_plus = index(i + 1, j);
        const std::size_t i_minus = index(i - 1, j);
        const std::size_t j_plus = index(i, j + 1);
        const std::size_t j_minus = index(i, j - 1);
        const std::size_t i_plus_j_plus = index(i + 1, j + 1);
        const std::size_t i_minus_j_plus = index(i - 1, j + 1);
        const std::size_t i_plus_j_minus = index(i + 1, j - 1);
        const std::size_t i_minus_j_minus = index(i - 1, j - 1);

        const double x_xi_r =
            0.5 * (patch.r[i_plus] - patch.r[i_minus]);
        const double x_xi_z =
            0.5 * (patch.z[i_plus] - patch.z[i_minus]);
        const double x_eta_r =
            0.5 * (patch.r[j_plus] - patch.r[j_minus]);
        const double x_eta_z =
            0.5 * (patch.z[j_plus] - patch.z[j_minus]);
        const double alpha = x_eta_r * x_eta_r + x_eta_z * x_eta_z;
        const double beta = x_xi_r * x_eta_r + x_xi_z * x_eta_z;
        const double gamma = x_xi_r * x_xi_r + x_xi_z * x_xi_z;
        const double alpha_plus_gamma = alpha + gamma;
        if (alpha_plus_gamma <= 0.0 ||
            !std::isfinite(alpha_plus_gamma)) {
          continue;
        }

        const double cross_r =
            patch.r[i_plus_j_plus] - patch.r[i_minus_j_plus] -
            patch.r[i_plus_j_minus] + patch.r[i_minus_j_minus];
        const double cross_z =
            patch.z[i_plus_j_plus] - patch.z[i_minus_j_plus] -
            patch.z[i_plus_j_minus] + patch.z[i_minus_j_minus];
        const double denominator = 2.0 * alpha_plus_gamma;
        const double new_r =
            (alpha * (patch.r[i_plus] + patch.r[i_minus]) +
             gamma * (patch.r[j_plus] + patch.r[j_minus]) -
             0.5 * beta * cross_r) /
            denominator;
        const double new_z =
            (alpha * (patch.z[i_plus] + patch.z[i_minus]) +
             gamma * (patch.z[j_plus] + patch.z[j_minus]) -
             0.5 * beta * cross_z) /
            denominator;
        next_r[center] =
            patch.r[center] + omega * (new_r - patch.r[center]);
        next_z[center] =
            patch.z[center] + omega * (new_z - patch.z[center]);
      }
    }
    patch.r.swap(next_r);
    patch.z.swap(next_z);
  }
}

// Positive-Jacobian safeguarded blend between two same-shape patches. Tests
// t = 1, 1/2, ..., 1/256 and returns `before` verbatim if none is admissible.
inline TfiPatch positive_jacobian_blend(const TfiPatch& before,
                                        const TfiPatch& after) {
  const auto validate_patch = [](const TfiPatch& patch) {
    TENRYU_ASSERT(patch.ni >= 1 && patch.nj >= 1,
                  "patch interval counts must be positive");
    const std::size_t ni_size = static_cast<std::size_t>(patch.ni);
    const std::size_t nj_size = static_cast<std::size_t>(patch.nj);
    const std::size_t node_count =
        (ni_size + 1U) * (nj_size + 1U);
    TENRYU_ASSERT(patch.r.size() == node_count &&
                      patch.z.size() == node_count,
                  "patch coordinate arrays must match its dimensions");
    for (std::size_t k = 0; k < node_count; ++k) {
      TENRYU_ASSERT(std::isfinite(patch.r[k]) && std::isfinite(patch.z[k]),
                    "patch coordinates must be finite");
    }
  };
  validate_patch(before);
  validate_patch(after);
  TENRYU_ASSERT(before.ni == after.ni && before.nj == after.nj,
                "blend patches must have matching dimensions");

  TfiPatch candidate;
  candidate.ni = before.ni;
  candidate.nj = before.nj;
  candidate.r.resize(before.r.size());
  candidate.z.resize(before.z.size());
  double t = 1.0;
  for (int halvings = 0; halvings <= 8; ++halvings) {
    for (std::size_t k = 0; k < before.r.size(); ++k) {
      candidate.r[k] = before.r[k] + t * (after.r[k] - before.r[k]);
      candidate.z[k] = before.z[k] + t * (after.z[k] - before.z[k]);
    }
    if (min_corner_jacobian(candidate) > 0.0) {
      return candidate;
    }
    t *= 0.5;
  }
  return before;
}

// Hermite column patch. Tangents are full derivatives with respect to eta;
// boundary rows are copied verbatim from the supplied chains.
inline TfiPatch hermite_column_patch(
    const BoundaryChain& bottom,
    const BoundaryChain& top,
    const std::vector<double>& d_bottom_r,
    const std::vector<double>& d_bottom_z,
    const std::vector<double>& d_top_r,
    const std::vector<double>& d_top_z,
    const int nj) {
  const auto validate_chain = [](const BoundaryChain& chain) {
    TENRYU_ASSERT(chain.r.size() == chain.z.size(),
                  "boundary coordinate arrays must have matching sizes");
    TENRYU_ASSERT(chain.r.size() >= 2U,
                  "boundary chains must contain at least two nodes");
    for (std::size_t k = 0; k < chain.r.size(); ++k) {
      TENRYU_ASSERT(std::isfinite(chain.r[k]) && std::isfinite(chain.z[k]),
                    "boundary coordinates must be finite");
    }
  };
  validate_chain(bottom);
  validate_chain(top);
  TENRYU_ASSERT(bottom.r.size() == top.r.size(),
                "bottom and top boundary sizes must match");
  TENRYU_ASSERT(d_bottom_r.size() == bottom.r.size() &&
                    d_bottom_z.size() == bottom.r.size() &&
                    d_top_r.size() == bottom.r.size() &&
                    d_top_z.size() == bottom.r.size(),
                "Hermite tangent arrays must match the boundary sizes");
  TENRYU_ASSERT(nj >= 1,
                "Hermite patch j interval count must be positive");
  for (std::size_t k = 0; k < bottom.r.size(); ++k) {
    TENRYU_ASSERT(std::isfinite(d_bottom_r[k]) &&
                      std::isfinite(d_bottom_z[k]) &&
                      std::isfinite(d_top_r[k]) &&
                      std::isfinite(d_top_z[k]),
                  "Hermite tangents must be finite");
  }

  const std::size_t ni_size = bottom.r.size() - 1U;
  TENRYU_ASSERT(ni_size <= static_cast<std::size_t>(
                               std::numeric_limits<int>::max()),
                "patch i interval count exceeds int range");
  const int ni = static_cast<int>(ni_size);
  const std::size_t nj_size = static_cast<std::size_t>(nj);
  TfiPatch patch;
  patch.ni = ni;
  patch.nj = nj;
  patch.r.resize((ni_size + 1U) * (nj_size + 1U));
  patch.z.resize((ni_size + 1U) * (nj_size + 1U));

  for (int i = 0; i <= ni; ++i) {
    const std::size_t i_index = static_cast<std::size_t>(i);
    const std::size_t column_begin = i_index * (nj_size + 1U);
    for (int j = 0; j <= nj; ++j) {
      const std::size_t index =
          column_begin + static_cast<std::size_t>(j);
      if (j == 0) {
        patch.r[index] = bottom.r[i_index];
        patch.z[index] = bottom.z[i_index];
      } else if (j == nj) {
        patch.r[index] = top.r[i_index];
        patch.z[index] = top.z[i_index];
      } else {
        const double eta =
            static_cast<double>(j) / static_cast<double>(nj);
        const double eta2 = eta * eta;
        const double eta3 = eta2 * eta;
        const double h00 = 2.0 * eta3 - 3.0 * eta2 + 1.0;
        const double h10 = eta3 - 2.0 * eta2 + eta;
        const double h01 = -2.0 * eta3 + 3.0 * eta2;
        const double h11 = eta3 - eta2;
        patch.r[index] = h00 * bottom.r[i_index] +
                         h10 * d_bottom_r[i_index] +
                         h01 * top.r[i_index] + h11 * d_top_r[i_index];
        patch.z[index] = h00 * bottom.z[i_index] +
                         h10 * d_bottom_z[i_index] +
                         h01 * top.z[i_index] + h11 * d_top_z[i_index];
      }
    }
  }
  return patch;
}

}  // namespace tenryu::mesh::assembly
