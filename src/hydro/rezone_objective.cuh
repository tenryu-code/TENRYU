#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale::m1 {

inline constexpr int kRezoneMaxCellVerts = 8;
inline constexpr int kRezoneMaxQuadraturePoints = 8;
inline constexpr int kRezoneMaxBacktrackHalvings = 12;
inline constexpr double kBarrierActivationFraction = 0.1;

// M1 objective geometry is planar dA=dr dz. No 2*pi*r factor enters these
// weights; the RZ physical-volume measure belongs to other registry fields.
struct RezoneQuadratureTarget {
  double w00 = 0.0;
  double w01 = 0.0;
  double w10 = 0.0;
  double w11 = 0.0;
  double winv00 = 0.0;
  double winv01 = 0.0;
  double winv10 = 0.0;
  double winv11 = 0.0;
  double det_w = 0.0;
  double omega = 0.0;
  double j_min = 0.0;
};

struct RezoneCellObjective {
  double weighted_sum = 0.0;
  double weight_sum = 0.0;
  double value = 0.0;
  double min_det_a = INFINITY;
  bool feasible = true;
};

struct RezoneMeshView {
  int n_nodes = 0;
  int n_cells = 0;
  int topology_stride = 4;
  int quadrature_stride = kRezoneMaxQuadraturePoints;

  const int* cell_node_offsets = nullptr;
  const int* cell_node_indices = nullptr;
  const std::uint8_t* cell_nverts = nullptr;

  const int* node_cell_offsets = nullptr;
  const int* node_cells = nullptr;
  const int* node_corners = nullptr;

  // Frozen inner-solve inputs.
  const RezoneQuadratureTarget* quadrature_targets = nullptr;
  const double* h = nullptr;
  const double* x_lagrangian_r = nullptr;
  const double* x_lagrangian_z = nullptr;
  const std::uint8_t* axis_mask = nullptr;

  // Ring-neighbor terms are centered only where both neighbors exist.
  // theta_touch_* is the reverse CSR from a node to every centered term
  // containing that node (center, left neighbor, or right neighbor).
  const int* theta_left = nullptr;
  const int* theta_right = nullptr;
  const int* theta_touch_offsets = nullptr;
  const int* theta_touch_centers = nullptr;
};

struct RezoneObjectiveParams {
  double gamma_align = 0.0;
  double lambda_m = 0.0;
  double theta_reg = 0.0;
  double beta = 0.0;
  double hessian_damping = 1.0e-10;
  int objective_node_count = 0;
};

struct RezoneLocalObjective {
  double value = INFINITY;
  double metric_numerator = 0.0;
  double metric_denominator = 0.0;
  double tether = 0.0;
  double theta = 0.0;
  double barrier = 0.0;
  double min_det_a = INFINITY;
  bool feasible = false;
};

enum class RezoneNodeSkipReason : std::uint8_t {
  kNone = 0,
  kInvalidInput,
  kBounds,
  kIncidentCellValidity,
  kBarrier,
  kThetaPairInvalid,
  kObjectiveInvalid,
  kZeroGradient,
  kHessianInvalid,
  kStepInvalid,
  kNoDecrease,
};

struct RezoneNodeDerivatives {
  RezoneLocalObjective objective{};
  double gradient_r = 0.0;
  double gradient_z = 0.0;
  double hessian_rr = 0.0;
  double hessian_rz = 0.0;
  double hessian_zz = 0.0;
  RezoneNodeSkipReason skip_reason = RezoneNodeSkipReason::kInvalidInput;
};

struct RezoneNodeNewtonStep {
  double delta_r = 0.0;
  double delta_z = 0.0;
  double objective_before = INFINITY;
  double objective_after = INFINITY;
  double gradient_r = 0.0;
  double gradient_z = 0.0;
  double proposed_step_norm = 0.0;
  int halvings = 0;
  bool blocked = false;
  RezoneNodeSkipReason skip_reason = RezoneNodeSkipReason::kNone;
};

struct RezoneSweepResult {
  double J_before = INFINITY;
  double J_after = INFINITY;
  int moved_nodes = 0;
  int blocked_nodes = 0;
  int halvings = 0;
};

struct RezoneRingNeighborPairs {
  std::vector<int> left;
  std::vector<int> right;
  std::vector<int> touch_offsets;
  std::vector<int> touch_centers;
};

namespace ring_detail {

inline void add_ring_neighbor_terms(
    RezoneRingNeighborPairs& pairs,
    const std::vector<int>& ring) {
  if (ring.size() < 3U) {
    return;
  }
  for (std::size_t j = 1; j + 1U < ring.size(); ++j) {
    const int center = ring[j];
    const int left = ring[j - 1U];
    const int right = ring[j + 1U];
    TENRYU_ASSERT(center >= 0 &&
                      center < static_cast<int>(pairs.left.size()) &&
                      left >= 0 &&
                      left < static_cast<int>(pairs.left.size()) &&
                      right >= 0 &&
                      right < static_cast<int>(pairs.left.size()),
                  "M1 ring-neighbor node id out of range");
    TENRYU_ASSERT(
        (pairs.left[static_cast<std::size_t>(center)] < 0 &&
         pairs.right[static_cast<std::size_t>(center)] < 0) ||
            (pairs.left[static_cast<std::size_t>(center)] == left &&
             pairs.right[static_cast<std::size_t>(center)] == right),
        "M1 ring-neighbor center has inconsistent within-block neighbors");
    pairs.left[static_cast<std::size_t>(center)] = left;
    pairs.right[static_cast<std::size_t>(center)] = right;
  }
}

inline std::vector<int> block_cell_ring(
    const mesh::MultiBlockTopology& mb,
    const std::vector<std::uint8_t>& cell_nverts,
    const int topology_stride,
    const mesh::BlockInfo& block,
    const int row,
    const bool upper) {
  std::vector<int> ring;
  const int first_cell =
      block.cell_begin + row * block.n_j_cells;
  for (int j = 0; j < block.n_j_cells; ++j) {
    const int cell = first_cell + j;
    TENRYU_ASSERT(
        j == 0 ||
            mb.cell_id_stable[static_cast<std::size_t>(cell)] ==
                mb.cell_id_stable[static_cast<std::size_t>(cell - 1)] + 1,
        "M1 ring-neighbor block row requires consecutive stable cell ids");
    const int begin =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
    const int end =
        mb.cell_node_csr_offsets[static_cast<std::size_t>(cell) + 1U];
    TENRYU_ASSERT(end - begin == topology_stride,
                  "M1 ring-neighbor CSR width must match topology stride");
    const int nverts =
        cell_nverts.empty()
            ? mesh::kMeshTopoCellStorageSlots
            : mesh::mesh_topo_cell_active_nverts(cell_nverts, cell);
    TENRYU_ASSERT(nverts == 4 || nverts == 5,
                  "M1 ring-neighbor builder supports quad/pentagon rows");
    if (!upper) {
      ring.push_back(
          mb.cell_node_csr_indices[static_cast<std::size_t>(begin)]);
      if (j + 1 == block.n_j_cells) {
        ring.push_back(mb.cell_node_csr_indices[
            static_cast<std::size_t>(begin + nverts - 1)]);
      }
      continue;
    }
    ring.push_back(mb.cell_node_csr_indices[
        static_cast<std::size_t>(begin + 1)]);
    if (nverts == 5) {
      ring.push_back(mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + 2)]);
    }
    if (j + 1 == block.n_j_cells) {
      ring.push_back(mb.cell_node_csr_indices[
          static_cast<std::size_t>(begin + nverts - 2)]);
    }
  }
  return ring;
}

inline void build_touch_csr(RezoneRingNeighborPairs& pairs) {
  const int n_nodes = static_cast<int>(pairs.left.size());
  std::vector<std::vector<int>> touching(
      static_cast<std::size_t>(n_nodes));
  for (int center = 0; center < n_nodes; ++center) {
    const int left = pairs.left[static_cast<std::size_t>(center)];
    const int right = pairs.right[static_cast<std::size_t>(center)];
    if (left < 0 || right < 0) {
      continue;
    }
    touching[static_cast<std::size_t>(left)].push_back(center);
    touching[static_cast<std::size_t>(center)].push_back(center);
    touching[static_cast<std::size_t>(right)].push_back(center);
  }
  pairs.touch_offsets.assign(static_cast<std::size_t>(n_nodes) + 1U, 0);
  for (int node = 0; node < n_nodes; ++node) {
    auto& centers = touching[static_cast<std::size_t>(node)];
    std::sort(centers.begin(), centers.end());
    centers.erase(std::unique(centers.begin(), centers.end()), centers.end());
    pairs.touch_offsets[static_cast<std::size_t>(node) + 1U] =
        pairs.touch_offsets[static_cast<std::size_t>(node)] +
        static_cast<int>(centers.size());
    pairs.touch_centers.insert(
        pairs.touch_centers.end(), centers.begin(), centers.end());
  }
}

}  // namespace ring_detail

inline RezoneRingNeighborPairs rezone_build_ring_neighbor_pairs(
    const mesh::MeshTopology& topology,
    const std::vector<std::uint8_t>& cell_nverts,
    const int topology_stride) {
  TENRYU_ASSERT(topology.n_nodes >= 0,
                "M1 ring-neighbor builder requires non-negative node count");
  RezoneRingNeighborPairs pairs;
  pairs.left.assign(static_cast<std::size_t>(topology.n_nodes), -1);
  pairs.right.assign(static_cast<std::size_t>(topology.n_nodes), -1);

  if (!topology.multiblock.has_value()) {
    TENRYU_ASSERT(
        topology.n_nodes == (topology.nr + 1) * (topology.nz + 1),
        "M1 structured ring-neighbor builder requires logical node storage");
    for (int i = 0; i <= topology.nr; ++i) {
      std::vector<int> ring;
      ring.reserve(static_cast<std::size_t>(topology.nz + 1));
      for (int j = 0; j <= topology.nz; ++j) {
        ring.push_back(topology.node_index(i, j));
      }
      ring_detail::add_ring_neighbor_terms(pairs, ring);
    }
    ring_detail::build_touch_csr(pairs);
    return pairs;
  }

  const auto& mb = *topology.multiblock;
  TENRYU_ASSERT(mb.cell_id_stable.size() ==
                    static_cast<std::size_t>(topology.n_cells),
                "M1 ring-neighbor builder requires stable cell ids");
  for (const mesh::BlockInfo& block : mb.blocks) {
    if (block.role != mesh::BlockRole::POLAR_SHELL &&
        block.role != mesh::BlockRole::PENTAGON_BELT) {
      continue;
    }
    for (int row = 0; row < block.n_i_cells; ++row) {
      ring_detail::add_ring_neighbor_terms(
          pairs,
          ring_detail::block_cell_ring(
              mb, cell_nverts, topology_stride, block, row, false));
      ring_detail::add_ring_neighbor_terms(
          pairs,
          ring_detail::block_cell_ring(
              mb, cell_nverts, topology_stride, block, row, true));
    }
  }
  ring_detail::build_touch_csr(pairs);
  return pairs;
}

namespace detail {

__host__ __device__ inline bool finite(const double x) {
#if defined(__CUDA_ARCH__)
  return isfinite(x);
#else
  return std::isfinite(x);
#endif
}

__host__ __device__ inline int cell_nverts(
    const std::uint8_t* const cell_nverts_ptr,
    const int cell) {
  return cell_nverts_ptr == nullptr ? 4
                                    : static_cast<int>(cell_nverts_ptr[cell]);
}

__host__ __device__ inline int quadrature_count(const int nverts) {
  return nverts == 4 ? 4 : nverts;
}

__host__ __device__ inline double determinant(
    const double a00,
    const double a01,
    const double a10,
    const double a11) {
  return a00 * a11 - a01 * a10;
}

__host__ __device__ inline void bilinear_derivative_weights(
    const int q,
    double dxi[4],
    double deta[4]) {
  // Same 2x2 Gauss constant and Q1 derivative idiom as
  // corner_jacobian_quality.cuh, re-declared here to keep this library uncoupled.
  constexpr double g =
      0.57735026918962576450914878050195745565;
  const double xi[4] = {-g, g, g, -g};
  const double eta[4] = {-g, -g, g, g};
  const double x = xi[q];
  const double e = eta[q];

  dxi[0] = -0.25 * (1.0 - e);
  dxi[1] = 0.25 * (1.0 - e);
  dxi[2] = 0.25 * (1.0 + e);
  dxi[3] = -0.25 * (1.0 + e);
  deta[0] = -0.25 * (1.0 - x);
  deta[1] = -0.25 * (1.0 + x);
  deta[2] = 0.25 * (1.0 + x);
  deta[3] = 0.25 * (1.0 - x);
}

__host__ __device__ inline bool matrix_inverse(
    const double a00,
    const double a01,
    const double a10,
    const double a11,
    double& inv00,
    double& inv01,
    double& inv10,
    double& inv11,
    double& det) {
  det = determinant(a00, a01, a10, a11);
  if (!finite(det) || !(det > 0.0)) {
    return false;
  }
  const double inv_det = 1.0 / det;
  inv00 = a11 * inv_det;
  inv01 = -a01 * inv_det;
  inv10 = -a10 * inv_det;
  inv11 = a00 * inv_det;
  return finite(inv00) && finite(inv01) && finite(inv10) && finite(inv11);
}

__host__ __device__ inline void polygon_vertex_mean(
    const double* const r,
    const double* const z,
    const int nverts,
    double& mean_r,
    double& mean_z) {
  mean_r = 0.0;
  mean_z = 0.0;
  for (int k = 0; k < nverts; ++k) {
    mean_r += r[k];
    mean_z += z[k];
  }
  mean_r /= static_cast<double>(nverts);
  mean_z /= static_cast<double>(nverts);
}

__host__ __device__ inline bool build_jacobian(
    const double* const r,
    const double* const z,
    const int nverts,
    const int q,
    const int moved_corner,
    double& a00,
    double& a01,
    double& a10,
    double& a11,
    double& coeff0,
    double& coeff1) {
  if (nverts < 3 || nverts > kRezoneMaxCellVerts) {
    return false;
  }

  if (nverts == 4) {
    if (q < 0 || q >= 4) {
      return false;
    }
    double dxi[4]{};
    double deta[4]{};
    bilinear_derivative_weights(q, dxi, deta);
    a00 = 0.0;
    a01 = 0.0;
    a10 = 0.0;
    a11 = 0.0;
    for (int k = 0; k < 4; ++k) {
      a00 += r[k] * dxi[k];
      a01 += r[k] * deta[k];
      a10 += z[k] * dxi[k];
      a11 += z[k] * deta[k];
    }
    coeff0 = moved_corner >= 0 ? dxi[moved_corner] : 0.0;
    coeff1 = moved_corner >= 0 ? deta[moved_corner] : 0.0;
    return finite(a00) && finite(a01) && finite(a10) && finite(a11);
  }

  if (q < 0 || q >= nverts) {
    return false;
  }
  const int next = (q + 1 == nverts) ? 0 : q + 1;
  double star_r = 0.0;
  double star_z = 0.0;
  polygon_vertex_mean(r, z, nverts, star_r, star_z);
  a00 = r[q] - star_r;
  a01 = r[next] - star_r;
  a10 = z[q] - star_z;
  a11 = z[next] - star_z;

  const double star_coefficient = moved_corner >= 0
                                      ? -1.0 / static_cast<double>(nverts)
                                      : 0.0;
  coeff0 = star_coefficient + (moved_corner == q ? 1.0 : 0.0);
  coeff1 = star_coefficient + (moved_corner == next ? 1.0 : 0.0);
  return finite(a00) && finite(a01) && finite(a10) && finite(a11);
}

struct QuadratureEvaluation {
  double metric = INFINITY;
  double det_a = -INFINITY;
  double metric_gradient_r = 0.0;
  double metric_gradient_z = 0.0;
  double det_gradient_r = 0.0;
  double det_gradient_z = 0.0;
  bool feasible = false;
};

struct ThetaEvaluation {
  double penalty = 0.0;
  double ratio = 0.0;
  double ratio_gradient_r = 0.0;
  double ratio_gradient_z = 0.0;
  bool feasible = false;
};

__host__ __device__ inline ThetaEvaluation evaluate_theta_term(
    const RezoneMeshView& view,
    const double* const current_r,
    const double* const current_z,
    const int center,
    const int moved_node,
    const double trial_r,
    const double trial_z) {
  ThetaEvaluation out{};
  if (center < 0 || center >= view.n_nodes ||
      view.theta_left == nullptr || view.theta_right == nullptr ||
      current_r == nullptr || current_z == nullptr) {
    return out;
  }
  const int left = view.theta_left[center];
  const int right = view.theta_right[center];
  if (left < 0 || left >= view.n_nodes ||
      right < 0 || right >= view.n_nodes) {
    return out;
  }
  const double center_r =
      center == moved_node ? trial_r : current_r[center];
  const double center_z =
      center == moved_node ? trial_z : current_z[center];
  const double left_r =
      left == moved_node ? trial_r : current_r[left];
  const double left_z =
      left == moved_node ? trial_z : current_z[left];
  const double right_r =
      right == moved_node ? trial_r : current_r[right];
  const double right_z =
      right == moved_node ? trial_z : current_z[right];
  const double dl_r = center_r - left_r;
  const double dl_z = center_z - left_z;
  const double dr_r = center_r - right_r;
  const double dr_z = center_z - right_z;
  const double d_left = sqrt(dl_r * dl_r + dl_z * dl_z);
  const double d_right = sqrt(dr_r * dr_r + dr_z * dr_z);
  const double sum = d_left + d_right;
  if (!finite(d_left) || !finite(d_right) || !(d_left > 0.0) ||
      !(d_right > 0.0) || !finite(sum) || !(sum > 0.0)) {
    return out;
  }

  const double difference = d_left - d_right;
  out.ratio = difference / sum;
  out.penalty = out.ratio * out.ratio;
  if (moved_node >= 0) {
    double grad_left_r = 0.0;
    double grad_left_z = 0.0;
    double grad_right_r = 0.0;
    double grad_right_z = 0.0;
    if (moved_node == center) {
      grad_left_r = dl_r / d_left;
      grad_left_z = dl_z / d_left;
      grad_right_r = dr_r / d_right;
      grad_right_z = dr_z / d_right;
    } else if (moved_node == left) {
      grad_left_r = -dl_r / d_left;
      grad_left_z = -dl_z / d_left;
    } else if (moved_node == right) {
      grad_right_r = -dr_r / d_right;
      grad_right_z = -dr_z / d_right;
    } else {
      return out;
    }
    const double inv_sum2 = 1.0 / (sum * sum);
    out.ratio_gradient_r =
        ((grad_left_r - grad_right_r) * sum -
         difference * (grad_left_r + grad_right_r)) *
        inv_sum2;
    out.ratio_gradient_z =
        ((grad_left_z - grad_right_z) * sum -
         difference * (grad_left_z + grad_right_z)) *
        inv_sum2;
  }
  out.feasible = finite(out.penalty) && finite(out.ratio_gradient_r) &&
                 finite(out.ratio_gradient_z);
  return out;
}

__host__ __device__ inline QuadratureEvaluation evaluate_quadrature(
    const double* const r,
    const double* const z,
    const int nverts,
    const int q,
    const int moved_corner,
    const RezoneQuadratureTarget& target,
    const double gamma_align) {
  QuadratureEvaluation out{};
  double a00 = 0.0;
  double a01 = 0.0;
  double a10 = 0.0;
  double a11 = 0.0;
  double coeff0 = 0.0;
  double coeff1 = 0.0;
  if (!build_jacobian(r, z, nverts, q, moved_corner,
                      a00, a01, a10, a11, coeff0, coeff1)) {
    return out;
  }

  out.det_a = determinant(a00, a01, a10, a11);
  if (!finite(out.det_a) || !(out.det_a > 0.0) ||
      !finite(target.j_min) || !(target.j_min > 0.0) ||
      out.det_a < target.j_min) {
    return out;
  }

  const double t00 = a00 * target.winv00 + a01 * target.winv10;
  const double t01 = a00 * target.winv01 + a01 * target.winv11;
  const double t10 = a10 * target.winv00 + a11 * target.winv10;
  const double t11 = a10 * target.winv01 + a11 * target.winv11;
  const double det_t = determinant(t00, t01, t10, t11);
  if (!finite(det_t) || !(det_t > 0.0)) {
    return out;
  }

  const double frobenius2 =
      t00 * t00 + t01 * t01 + t10 * t10 + t11 * t11;
  const double mu = frobenius2 / (2.0 * det_t);
  const double dm00 = t00 / det_t - mu * t11 / det_t;
  const double dm01 = t01 / det_t + mu * t10 / det_t;
  const double dm10 = t10 / det_t + mu * t01 / det_t;
  const double dm11 = t11 / det_t - mu * t00 / det_t;

  const double d00 = t00 - 1.0;
  const double d01 = t01;
  const double d10 = t10;
  const double d11 = t11 - 1.0;
  const double align =
      gamma_align * (d00 * d00 + d01 * d01 + d10 * d10 + d11 * d11);

  const double gt00 = dm00 + 2.0 * gamma_align * d00;
  const double gt01 = dm01 + 2.0 * gamma_align * d01;
  const double gt10 = dm10 + 2.0 * gamma_align * d10;
  const double gt11 = dm11 + 2.0 * gamma_align * d11;

  // For T=A W^{-1},
  //   d mu_shape/dT = T/det(T) - mu_shape T^{-T},
  //   d ||T-I||_F^2/dT = 2(T-I), and
  //   d f/dA = (d f/dT) W^{-T}.
  const double ga00 = gt00 * target.winv00 + gt01 * target.winv01;
  const double ga01 = gt00 * target.winv10 + gt01 * target.winv11;
  const double ga10 = gt10 * target.winv00 + gt11 * target.winv01;
  const double ga11 = gt10 * target.winv10 + gt11 * target.winv11;

  out.metric = mu + align;
  out.metric_gradient_r = ga00 * coeff0 + ga01 * coeff1;
  out.metric_gradient_z = ga10 * coeff0 + ga11 * coeff1;
  out.det_gradient_r = a11 * coeff0 - a10 * coeff1;
  out.det_gradient_z = -a01 * coeff0 + a00 * coeff1;
  out.feasible = finite(out.metric) && finite(out.metric_gradient_r) &&
                 finite(out.metric_gradient_z) &&
                 finite(out.det_gradient_r) &&
                 finite(out.det_gradient_z);
  return out;
}

__host__ __device__ inline bool load_cell(
    const RezoneMeshView& view,
    const double* const current_r,
    const double* const current_z,
    const int cell,
    const int moved_node,
    const double trial_r,
    const double trial_z,
    double r[kRezoneMaxCellVerts],
    double z[kRezoneMaxCellVerts],
    int& nverts) {
  if (cell < 0 || cell >= view.n_cells ||
      view.cell_node_offsets == nullptr ||
      view.cell_node_indices == nullptr ||
      current_r == nullptr || current_z == nullptr) {
    return false;
  }
  nverts = cell_nverts(view.cell_nverts, cell);
  if (nverts < 3 || nverts > kRezoneMaxCellVerts) {
    return false;
  }
  const int begin = view.cell_node_offsets[cell];
  const int end = view.cell_node_offsets[cell + 1];
  if (begin < 0 || end - begin != view.topology_stride ||
      view.topology_stride < nverts) {
    return false;
  }
  for (int k = 0; k < nverts; ++k) {
    const int node = view.cell_node_indices[begin + k];
    if (node < 0 || node >= view.n_nodes) {
      return false;
    }
    r[k] = node == moved_node ? trial_r : current_r[node];
    z[k] = node == moved_node ? trial_z : current_z[node];
    if (!finite(r[k]) || !finite(z[k])) {
      return false;
    }
  }
  return true;
}

__host__ __device__ inline const RezoneQuadratureTarget& target_at(
    const RezoneMeshView& view,
    const int cell,
    const int q) {
  return view.quadrature_targets[cell * view.quadrature_stride + q];
}

__host__ __device__ inline int objective_node_count(
    const RezoneMeshView& view,
    const RezoneObjectiveParams& params) {
  return params.objective_node_count > 0 ? params.objective_node_count
                                         : view.n_nodes;
}

}  // namespace detail

__host__ __device__ inline bool rezone_build_cell_quadrature_targets(
    const double* const reference_r,
    const double* const reference_z,
    const int nverts,
    const double* const j_min_q,
    RezoneQuadratureTarget* const targets) {
  if (reference_r == nullptr || reference_z == nullptr || targets == nullptr ||
      nverts < 3 || nverts > kRezoneMaxCellVerts) {
    return false;
  }

  const int nq = detail::quadrature_count(nverts);
  for (int q = 0; q < nq; ++q) {
    double w00 = 0.0;
    double w01 = 0.0;
    double w10 = 0.0;
    double w11 = 0.0;
    double unused0 = 0.0;
    double unused1 = 0.0;
    if (!detail::build_jacobian(reference_r, reference_z, nverts, q, -1,
                                w00, w01, w10, w11,
                                unused0, unused1)) {
      return false;
    }

    RezoneQuadratureTarget target{};
    target.w00 = w00;
    target.w01 = w01;
    target.w10 = w10;
    target.w11 = w11;
    if (!detail::matrix_inverse(
            w00, w01, w10, w11,
            target.winv00, target.winv01,
            target.winv10, target.winv11, target.det_w)) {
      return false;
    }

    if (nverts == 4) {
      target.omega = 1.0;
    } else {
      // Star-P1 uses (x*,v_q,v_q+1), with x* the vertex mean. A and W
      // both use edge-column Jacobians, so their determinants are consistently
      // twice-area and the factor cancels in T=A W^{-1}. The one-point
      // rule is located at the triangle centroid; A is P1-constant, so the
      // centroid coordinates need not be stored. Its planar quadrature weight
      // is the reference star-triangle area.
      target.omega = 0.5 * target.det_w;
    }
    target.j_min =
        j_min_q == nullptr
            ? fmax(1.0e-12 * target.det_w, 1.0e-300)
            : j_min_q[q];
    if (!detail::finite(target.omega) || !(target.omega > 0.0) ||
        !detail::finite(target.j_min) || !(target.j_min > 0.0)) {
      return false;
    }

    // TODO(M1): compose W with the frozen shock-target rotation.
    targets[q] = target;
  }
  return true;
}

__host__ __device__ inline RezoneCellObjective rezone_cell_objective(
    const double* const current_r,
    const double* const current_z,
    const double* const reference_r,
    const double* const reference_z,
    const int nverts,
    const double gamma_align,
    const double* const j_min_q = nullptr) {
  RezoneCellObjective out{};
  if (current_r == nullptr || current_z == nullptr) {
    out.feasible = false;
    out.value = INFINITY;
    out.weighted_sum = INFINITY;
    out.min_det_a = -INFINITY;
    return out;
  }
  RezoneQuadratureTarget targets[kRezoneMaxQuadraturePoints]{};
  if (!rezone_build_cell_quadrature_targets(
          reference_r, reference_z, nverts, j_min_q, targets)) {
    out.feasible = false;
    out.value = INFINITY;
    out.weighted_sum = INFINITY;
    out.min_det_a = -INFINITY;
    return out;
  }

  const int nq = detail::quadrature_count(nverts);
  for (int q = 0; q < nq; ++q) {
    const detail::QuadratureEvaluation evaluation =
        detail::evaluate_quadrature(
            current_r, current_z, nverts, q, -1, targets[q], gamma_align);
    out.min_det_a = fmin(out.min_det_a, evaluation.det_a);
    if (!evaluation.feasible) {
      out.feasible = false;
      out.value = INFINITY;
      out.weighted_sum = INFINITY;
      return out;
    }
    const double weight = targets[q].omega * targets[q].det_w;
    out.weighted_sum += weight * evaluation.metric;
    out.weight_sum += weight;
  }

  if (!(out.weight_sum > 0.0) || !detail::finite(out.weighted_sum) ||
      !detail::finite(out.weight_sum)) {
    out.feasible = false;
    out.value = INFINITY;
    return out;
  }
  // mu_shape is the reciprocal of the existing
  // multiblock_winslow_corner_condition_quality form
  // 2J/(|a|^2+|b|^2) in ale_rezone.cuh.
  out.value = out.weighted_sum / out.weight_sum;
  return out;
}

__host__ __device__ inline RezoneNodeDerivatives
rezone_node_local_derivatives(
    const RezoneMeshView& view,
    const double* const current_r,
    const double* const current_z,
    const int node,
    double trial_r,
    const double trial_z,
    const RezoneObjectiveParams& params) {
  RezoneNodeDerivatives out{};
  if (node < 0 || node >= view.n_nodes ||
      view.node_cell_offsets == nullptr || view.node_cells == nullptr ||
      view.node_corners == nullptr || view.quadrature_targets == nullptr ||
      view.h == nullptr || view.x_lagrangian_r == nullptr ||
      view.x_lagrangian_z == nullptr || current_r == nullptr ||
      current_z == nullptr || !detail::finite(trial_r) ||
      !detail::finite(trial_z)) {
    return out;
  }
  out.skip_reason = RezoneNodeSkipReason::kNone;

  const bool axis = view.axis_mask != nullptr && view.axis_mask[node] != 0U;
  if (axis) {
    trial_r = 0.0;
  } else if (trial_r < 0.0) {
    out.skip_reason = RezoneNodeSkipReason::kBounds;
    return out;
  }

  const int incident_begin = view.node_cell_offsets[node];
  const int incident_end = view.node_cell_offsets[node + 1];
  if (incident_begin < 0 || incident_end < incident_begin) {
    out.skip_reason = RezoneNodeSkipReason::kIncidentCellValidity;
    return out;
  }

  out.objective.feasible = true;
  for (int p = incident_begin; p < incident_end; ++p) {
    const int cell = view.node_cells[p];
    const int moved_corner = view.node_corners[p];
    double r[kRezoneMaxCellVerts]{};
    double z[kRezoneMaxCellVerts]{};
    int nverts = 0;
    if (!detail::load_cell(view, current_r, current_z, cell, node,
                           trial_r, trial_z, r, z, nverts) ||
        moved_corner < 0 || moved_corner >= nverts) {
      out.objective.feasible = false;
      out.skip_reason = RezoneNodeSkipReason::kIncidentCellValidity;
      break;
    }
    const int nq = detail::quadrature_count(nverts);
    for (int q = 0; q < nq; ++q) {
      const RezoneQuadratureTarget& target =
          detail::target_at(view, cell, q);
      const detail::QuadratureEvaluation evaluation =
          detail::evaluate_quadrature(
              r, z, nverts, q, moved_corner, target, params.gamma_align);
      out.objective.min_det_a =
          fmin(out.objective.min_det_a, evaluation.det_a);
      if (!evaluation.feasible) {
        out.objective.feasible = false;
        out.skip_reason =
            detail::finite(evaluation.det_a) &&
                    detail::finite(target.j_min) &&
                    evaluation.det_a < target.j_min
                ? RezoneNodeSkipReason::kBarrier
                : RezoneNodeSkipReason::kIncidentCellValidity;
        break;
      }
      const double weight = target.omega * target.det_w;
      out.objective.metric_numerator += weight * evaluation.metric;
      out.objective.metric_denominator += weight;
      const double act = kBarrierActivationFraction * target.det_w;
      if (params.beta != 0.0 && evaluation.det_a < act) {
        out.objective.barrier +=
            params.beta * log(act / evaluation.det_a);
      }
    }
    if (!out.objective.feasible) {
      break;
    }
  }

  const double h = view.h[node];
  const int nv = detail::objective_node_count(view, params);
  if (!out.objective.feasible || !detail::finite(h) || !(h > 0.0) ||
      nv <= 0 || !(out.objective.metric_denominator > 0.0)) {
    out.objective.feasible = false;
    out.objective.value = INFINITY;
    if (out.skip_reason == RezoneNodeSkipReason::kNone) {
      out.skip_reason = RezoneNodeSkipReason::kIncidentCellValidity;
    }
    return out;
  }

  const double dr_l = trial_r - view.x_lagrangian_r[node];
  const double dz_l = trial_z - view.x_lagrangian_z[node];
  const double tether_scale =
      params.lambda_m / (static_cast<double>(nv) * h * h);
  out.objective.tether =
      tether_scale * (dr_l * dr_l + dz_l * dz_l);

  // theta bunching lies in the null space of cell-local terms (subzonal merit-1.0 experiment + external consult, 2026-07-28); this neighbor-aware spacing penalty is the prescribed cure.
  if (params.theta_reg != 0.0) {
    if (view.theta_touch_offsets == nullptr ||
        view.theta_touch_centers == nullptr) {
      out.objective.feasible = false;
      out.objective.value = INFINITY;
      out.skip_reason = RezoneNodeSkipReason::kThetaPairInvalid;
      return out;
    }
    const int theta_begin = view.theta_touch_offsets[node];
    const int theta_end = view.theta_touch_offsets[node + 1];
    if (theta_begin < 0 || theta_end < theta_begin) {
      out.objective.feasible = false;
      out.objective.value = INFINITY;
      out.skip_reason = RezoneNodeSkipReason::kThetaPairInvalid;
      return out;
    }
    const double theta_scale =
        params.theta_reg / static_cast<double>(nv);
    for (int p = theta_begin; p < theta_end; ++p) {
      const detail::ThetaEvaluation theta =
          detail::evaluate_theta_term(
              view, current_r, current_z, view.theta_touch_centers[p],
              node, trial_r, trial_z);
      if (!theta.feasible) {
        out.objective.feasible = false;
        out.objective.value = INFINITY;
        out.skip_reason = RezoneNodeSkipReason::kThetaPairInvalid;
        return out;
      }
      out.objective.theta += theta_scale * theta.penalty;
      const double gradient_scale =
          2.0 * theta_scale * theta.ratio;
      out.gradient_r += gradient_scale * theta.ratio_gradient_r;
      out.gradient_z += gradient_scale * theta.ratio_gradient_z;
      // Gauss-Newton Hessian of sqrt(theta_scale) * ratio.
      out.hessian_rr +=
          2.0 * theta_scale * theta.ratio_gradient_r *
          theta.ratio_gradient_r;
      out.hessian_rz +=
          2.0 * theta_scale * theta.ratio_gradient_r *
          theta.ratio_gradient_z;
      out.hessian_zz +=
          2.0 * theta_scale * theta.ratio_gradient_z *
          theta.ratio_gradient_z;
    }
  }

  out.objective.value =
      out.objective.metric_numerator / out.objective.metric_denominator +
      out.objective.tether + out.objective.theta + out.objective.barrier;
  if (!detail::finite(out.objective.value)) {
    out.objective.feasible = false;
    out.objective.value = INFINITY;
    out.skip_reason = RezoneNodeSkipReason::kObjectiveInvalid;
    return out;
  }

  for (int p = incident_begin; p < incident_end; ++p) {
    const int cell = view.node_cells[p];
    const int moved_corner = view.node_corners[p];
    double r[kRezoneMaxCellVerts]{};
    double z[kRezoneMaxCellVerts]{};
    int nverts = 0;
    if (!detail::load_cell(view, current_r, current_z, cell, node,
                           trial_r, trial_z, r, z, nverts)) {
      out.objective.feasible = false;
      out.objective.value = INFINITY;
      out.skip_reason = RezoneNodeSkipReason::kIncidentCellValidity;
      return out;
    }
    const int nq = detail::quadrature_count(nverts);
    for (int q = 0; q < nq; ++q) {
      const RezoneQuadratureTarget& target =
          detail::target_at(view, cell, q);
      const detail::QuadratureEvaluation evaluation =
          detail::evaluate_quadrature(
              r, z, nverts, q, moved_corner, target, params.gamma_align);
      if (!evaluation.feasible) {
        out.objective.feasible = false;
        out.objective.value = INFINITY;
        out.skip_reason =
            detail::finite(evaluation.det_a) &&
                    detail::finite(target.j_min) &&
                    evaluation.det_a < target.j_min
                ? RezoneNodeSkipReason::kBarrier
                : RezoneNodeSkipReason::kIncidentCellValidity;
        return out;
      }
      const double normalized_weight =
          target.omega * target.det_w /
          out.objective.metric_denominator;
      const double gm_r =
          normalized_weight * evaluation.metric_gradient_r;
      const double gm_z =
          normalized_weight * evaluation.metric_gradient_z;
      const double act = kBarrierActivationFraction * target.det_w;
      const double gb_r =
          params.beta != 0.0 && evaluation.det_a < act
              ? -params.beta * evaluation.det_gradient_r /
                    evaluation.det_a
              : 0.0;
      const double gb_z =
          params.beta != 0.0 && evaluation.det_a < act
              ? -params.beta * evaluation.det_gradient_z /
                    evaluation.det_a
              : 0.0;
      const double gq_r = gm_r + gb_r;
      const double gq_z = gm_z + gb_z;
      out.gradient_r += gq_r;
      out.gradient_z += gq_z;

      // Positive-semidefinite scalar-residual Gauss-Newton approximation:
      // H_q ~= grad(f_q) grad(f_q)^T. The tether Hessian below is exact.
      out.hessian_rr += gq_r * gq_r;
      out.hessian_rz += gq_r * gq_z;
      out.hessian_zz += gq_z * gq_z;
    }
  }

  out.gradient_r += 2.0 * tether_scale * dr_l;
  out.gradient_z += 2.0 * tether_scale * dz_l;
  out.hessian_rr += 2.0 * tether_scale;
  out.hessian_zz += 2.0 * tether_scale;
  return out;
}

__host__ __device__ inline RezoneLocalObjective
rezone_node_local_objective(
    const RezoneMeshView& view,
    const double* const current_r,
    const double* const current_z,
    const int node,
    const double trial_r,
    const double trial_z,
    const RezoneObjectiveParams& params) {
  return rezone_node_local_derivatives(
             view, current_r, current_z, node, trial_r, trial_z, params)
      .objective;
}

__host__ __device__ inline RezoneNodeNewtonStep rezone_node_newton_step(
    const RezoneMeshView& view,
    const double* const current_r,
    const double* const current_z,
    const int node,
    const RezoneObjectiveParams& params) {
  RezoneNodeNewtonStep step{};
  if (node < 0 || node >= view.n_nodes || current_r == nullptr ||
      current_z == nullptr) {
    step.blocked = true;
    step.skip_reason = RezoneNodeSkipReason::kInvalidInput;
    return step;
  }

  const bool axis = view.axis_mask != nullptr && view.axis_mask[node] != 0U;
  const double base_r = axis ? 0.0 : current_r[node];
  const double base_z = current_z[node];
  const RezoneNodeDerivatives derivatives =
      rezone_node_local_derivatives(
          view, current_r, current_z, node, base_r, base_z, params);
  step.objective_before = derivatives.objective.value;
  step.objective_after = derivatives.objective.value;
  step.gradient_r = derivatives.gradient_r;
  step.gradient_z = derivatives.gradient_z;
  if (!derivatives.objective.feasible) {
    step.blocked = true;
    step.skip_reason = derivatives.skip_reason;
    return step;
  }

  const double active_gradient_r = axis ? 0.0 : derivatives.gradient_r;
  const double gradient2 =
      active_gradient_r * active_gradient_r +
      derivatives.gradient_z * derivatives.gradient_z;
  if (!detail::finite(gradient2) ||
      gradient2 <= 1.0e-30 * fmax(1.0, derivatives.objective.value *
                                          derivatives.objective.value)) {
    step.skip_reason = RezoneNodeSkipReason::kZeroGradient;
    return step;
  }

  const double damping = fmax(0.0, params.hessian_damping);
  double raw_delta_r = 0.0;
  double raw_delta_z = 0.0;
  if (axis) {
    const double hzz = derivatives.hessian_zz + damping;
    if (!detail::finite(hzz) || !(hzz > 0.0)) {
      step.blocked = true;
      step.skip_reason = RezoneNodeSkipReason::kHessianInvalid;
      return step;
    }
    raw_delta_z = -derivatives.gradient_z / hzz;
  } else {
    const double hrr = derivatives.hessian_rr + damping;
    const double hrz = derivatives.hessian_rz;
    const double hzz = derivatives.hessian_zz + damping;
    const double det_h = hrr * hzz - hrz * hrz;
    if (!detail::finite(det_h) || !(det_h > 0.0)) {
      step.blocked = true;
      step.skip_reason = RezoneNodeSkipReason::kHessianInvalid;
      return step;
    }
    raw_delta_r =
        (-hzz * derivatives.gradient_r +
         hrz * derivatives.gradient_z) /
        det_h;
    raw_delta_z =
        (hrz * derivatives.gradient_r -
         hrr * derivatives.gradient_z) /
        det_h;
  }

  if (!detail::finite(raw_delta_r) || !detail::finite(raw_delta_z)) {
    step.blocked = true;
    step.skip_reason = RezoneNodeSkipReason::kStepInvalid;
    return step;
  }
  step.proposed_step_norm =
      sqrt(raw_delta_r * raw_delta_r + raw_delta_z * raw_delta_z);
  double h_local = INFINITY;
  const int incident_begin = view.node_cell_offsets[node];
  const int incident_end = view.node_cell_offsets[node + 1];
  for (int p = incident_begin; p < incident_end; ++p) {
    const int cell = view.node_cells[p];
    const int corner = view.node_corners[p];
    const int nverts = detail::cell_nverts(view.cell_nverts, cell);
    const int begin = view.cell_node_offsets[cell];
    const int previous_node =
        view.cell_node_indices[begin + (corner + nverts - 1) % nverts];
    const int next_node =
        view.cell_node_indices[begin + (corner + 1) % nverts];
    const double previous_dr = current_r[previous_node] - base_r;
    const double previous_dz = current_z[previous_node] - base_z;
    const double next_dr = current_r[next_node] - base_r;
    const double next_dz = current_z[next_node] - base_z;
    h_local =
        fmin(h_local,
             sqrt(previous_dr * previous_dr + previous_dz * previous_dz));
    h_local =
        fmin(h_local, sqrt(next_dr * next_dr + next_dz * next_dz));
  }
  const double max_step_norm = 0.25 * h_local;
  if (step.proposed_step_norm > max_step_norm) {
    const double step_scale = max_step_norm / step.proposed_step_norm;
    raw_delta_r *= step_scale;
    raw_delta_z *= step_scale;
  }

  double scale = 1.0;
  RezoneNodeSkipReason rejection_reason =
      RezoneNodeSkipReason::kNoDecrease;
  for (int halvings = 0; halvings <= kRezoneMaxBacktrackHalvings;
       ++halvings) {
    const double trial_r = axis ? 0.0 : base_r + scale * raw_delta_r;
    const double trial_z = base_z + scale * raw_delta_z;
    if (axis || trial_r >= 0.0) {
      const RezoneNodeDerivatives trial_derivatives =
          rezone_node_local_derivatives(
              view, current_r, current_z, node, trial_r, trial_z, params);
      const RezoneLocalObjective& trial = trial_derivatives.objective;
      if (trial.feasible && trial.value < step.objective_before) {
        step.delta_r = axis ? 0.0 : trial_r - base_r;
        step.delta_z = trial_z - base_z;
        step.objective_after = trial.value;
        step.halvings = halvings;
        return step;
      }
      rejection_reason =
          trial.feasible ? RezoneNodeSkipReason::kNoDecrease
                         : trial_derivatives.skip_reason;
    } else {
      rejection_reason = RezoneNodeSkipReason::kBounds;
    }
    if (halvings < kRezoneMaxBacktrackHalvings) {
      scale *= 0.5;
    }
  }

  step.halvings = kRezoneMaxBacktrackHalvings;
  step.blocked = true;
  step.skip_reason = rejection_reason;
  return step;
}

inline const char* rezone_node_skip_reason_name(
    const RezoneNodeSkipReason reason) {
  switch (reason) {
    case RezoneNodeSkipReason::kNone:
      return "accepted";
    case RezoneNodeSkipReason::kInvalidInput:
      return "invalid input";
    case RezoneNodeSkipReason::kBounds:
      return "radial bounds";
    case RezoneNodeSkipReason::kIncidentCellValidity:
      return "incident-cell bounds/validity";
    case RezoneNodeSkipReason::kBarrier:
      return "incident-cell barrier";
    case RezoneNodeSkipReason::kThetaPairInvalid:
      return "theta pair invalid";
    case RezoneNodeSkipReason::kObjectiveInvalid:
      return "objective invalid";
    case RezoneNodeSkipReason::kZeroGradient:
      return "zero gradient";
    case RezoneNodeSkipReason::kHessianInvalid:
      return "Hessian invalid";
    case RezoneNodeSkipReason::kStepInvalid:
      return "Newton step invalid";
    case RezoneNodeSkipReason::kNoDecrease:
      return "no local objective decrease";
  }
  return "unknown";
}

inline RezoneLocalObjective rezone_global_objective(
    const RezoneMeshView& view,
    const double* const current_r,
    const double* const current_z,
    const RezoneObjectiveParams& params) {
  RezoneLocalObjective out{};
  out.feasible = true;
  if (current_r == nullptr || current_z == nullptr ||
      view.quadrature_targets == nullptr || view.h == nullptr ||
      view.x_lagrangian_r == nullptr || view.x_lagrangian_z == nullptr) {
    out.feasible = false;
    return out;
  }

  for (int node = 0; node < view.n_nodes; ++node) {
    if (!detail::finite(current_r[node]) || !detail::finite(current_z[node]) ||
        current_r[node] < 0.0 ||
        (view.axis_mask != nullptr && view.axis_mask[node] != 0U &&
         current_r[node] != 0.0)) {
      out.feasible = false;
      return out;
    }
  }

  for (int cell = 0; cell < view.n_cells; ++cell) {
    double r[kRezoneMaxCellVerts]{};
    double z[kRezoneMaxCellVerts]{};
    int nverts = 0;
    if (!detail::load_cell(view, current_r, current_z, cell, -1,
                           0.0, 0.0, r, z, nverts)) {
      out.feasible = false;
      return out;
    }
    const int nq = detail::quadrature_count(nverts);
    for (int q = 0; q < nq; ++q) {
      const RezoneQuadratureTarget& target =
          detail::target_at(view, cell, q);
      const detail::QuadratureEvaluation evaluation =
          detail::evaluate_quadrature(
              r, z, nverts, q, -1, target, params.gamma_align);
      out.min_det_a = fmin(out.min_det_a, evaluation.det_a);
      if (!evaluation.feasible) {
        out.feasible = false;
        return out;
      }
      const double weight = target.omega * target.det_w;
      out.metric_numerator += weight * evaluation.metric;
      out.metric_denominator += weight;
      const double act = kBarrierActivationFraction * target.det_w;
      if (params.beta != 0.0 && evaluation.det_a < act) {
        out.barrier += params.beta * log(act / evaluation.det_a);
      }
    }
  }

  const int nv = detail::objective_node_count(view, params);
  if (!(out.metric_denominator > 0.0) || nv <= 0) {
    out.feasible = false;
    return out;
  }
  for (int node = 0; node < view.n_nodes; ++node) {
    const double h = view.h[node];
    if (!detail::finite(h) || !(h > 0.0)) {
      out.feasible = false;
      return out;
    }
    const double dr = current_r[node] - view.x_lagrangian_r[node];
    const double dz = current_z[node] - view.x_lagrangian_z[node];
    out.tether += params.lambda_m * (dr * dr + dz * dz) /
                  (static_cast<double>(nv) * h * h);
  }

  if (params.theta_reg != 0.0) {
    if (view.theta_left == nullptr || view.theta_right == nullptr) {
      out.feasible = false;
      return out;
    }
    const double theta_scale =
        params.theta_reg / static_cast<double>(nv);
    for (int center = 0; center < view.n_nodes; ++center) {
      if (view.theta_left[center] < 0 || view.theta_right[center] < 0) {
        continue;
      }
      const detail::ThetaEvaluation theta =
          detail::evaluate_theta_term(
              view, current_r, current_z, center, -1, 0.0, 0.0);
      if (!theta.feasible) {
        out.feasible = false;
        return out;
      }
      out.theta += theta_scale * theta.penalty;
    }
  }

  out.value = out.metric_numerator / out.metric_denominator +
              out.tether + out.theta + out.barrier;
  out.feasible = detail::finite(out.value);
  return out;
}

inline RezoneSweepResult rezone_sweep(
    const RezoneMeshView& view,
    double* const current_r,
    double* const current_z,
    const std::uint8_t* const node_mask,
    const RezoneObjectiveParams& params) {
  RezoneSweepResult result{};
  const RezoneLocalObjective before =
      rezone_global_objective(view, current_r, current_z, params);
  result.J_before = before.value;
  result.J_after = before.value;
  if (!before.feasible || current_r == nullptr || current_z == nullptr) {
    return result;
  }

  std::vector<double> old_r(
      current_r, current_r + static_cast<std::size_t>(view.n_nodes));
  std::vector<double> old_z(
      current_z, current_z + static_cast<std::size_t>(view.n_nodes));
  std::vector<double> staged_r = old_r;
  std::vector<double> staged_z = old_z;

  bool any_step = false;
  for (int node = 0; node < view.n_nodes; ++node) {
    if (node_mask != nullptr && node_mask[node] == 0U) {
      continue;
    }
    const RezoneNodeNewtonStep step =
        rezone_node_newton_step(
            view, old_r.data(), old_z.data(), node, params);
    result.halvings += step.halvings;
    if (step.blocked) {
      ++result.blocked_nodes;
      continue;
    }
    staged_r[static_cast<std::size_t>(node)] =
        old_r[static_cast<std::size_t>(node)] + step.delta_r;
    staged_z[static_cast<std::size_t>(node)] =
        old_z[static_cast<std::size_t>(node)] + step.delta_z;
    if (view.axis_mask != nullptr && view.axis_mask[node] != 0U) {
      staged_r[static_cast<std::size_t>(node)] = 0.0;
    }
    any_step = any_step || step.delta_r != 0.0 || step.delta_z != 0.0;
  }

  if (!any_step) {
    return result;
  }

  std::vector<double> candidate_r(static_cast<std::size_t>(view.n_nodes));
  std::vector<double> candidate_z(static_cast<std::size_t>(view.n_nodes));
  double scale = 1.0;
  for (int halvings = 0; halvings <= kRezoneMaxBacktrackHalvings;
       ++halvings) {
    for (int node = 0; node < view.n_nodes; ++node) {
      candidate_r[static_cast<std::size_t>(node)] =
          old_r[static_cast<std::size_t>(node)] +
          scale * (staged_r[static_cast<std::size_t>(node)] -
                   old_r[static_cast<std::size_t>(node)]);
      candidate_z[static_cast<std::size_t>(node)] =
          old_z[static_cast<std::size_t>(node)] +
          scale * (staged_z[static_cast<std::size_t>(node)] -
                   old_z[static_cast<std::size_t>(node)]);
      if (view.axis_mask != nullptr && view.axis_mask[node] != 0U) {
        candidate_r[static_cast<std::size_t>(node)] = 0.0;
      }
    }
    const RezoneLocalObjective after =
        rezone_global_objective(
            view, candidate_r.data(), candidate_z.data(), params);
    if (after.feasible && after.value < before.value) {
      for (int node = 0; node < view.n_nodes; ++node) {
        current_r[node] = candidate_r[static_cast<std::size_t>(node)];
        current_z[node] = candidate_z[static_cast<std::size_t>(node)];
        if (current_r[node] != old_r[static_cast<std::size_t>(node)] ||
            current_z[node] != old_z[static_cast<std::size_t>(node)]) {
          ++result.moved_nodes;
        }
      }
      result.J_after = after.value;
      result.halvings += halvings;
      return result;
    }
    if (halvings < kRezoneMaxBacktrackHalvings) {
      scale *= 0.5;
    }
  }

  result.halvings += kRezoneMaxBacktrackHalvings;
  return result;
}

}  // namespace tenryu::hydro::ale::m1
