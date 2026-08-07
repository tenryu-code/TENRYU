#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/boundary_2d.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/candidate_mesh_admissibility.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale {

struct TriFanTrackingReferenceResult {
  bool applicable = false;
  bool installed = false;
  double sigma_accepted = 0.0;
  int linesearch_iters = 0;
  bool identity_follow_fallback = false;
  int identity_follow_fallback_count = 0;
  double outer_band_max_log_volume_ratio = 0.0;
  double angular_spread_over_h_max = 0.0;
  int shock_width_proxy_rings = 0;
  double max_limited_delta = 0.0;
  double max_limiter_excess = 0.0;
  std::vector<double> raw_ring_scale;
  std::vector<double> filtered_ring_scale;
  tenryu::mesh::CandidateMeshQuality final_quality{};
};

struct MultiblockDifferentialBandScaleResult {
  bool applicable = false;
  bool built = false;
  int band_count = 0;
  int limiter_excess_count = 0;
  int monotone_clamp_count = 0;
  double s_max_initial = 0.0;
  double max_limited_delta = 0.0;
  double max_limiter_excess = 0.0;
  std::vector<int> band_sample_count;
  std::vector<double> initial_band_scale;
  std::vector<double> raw_band_scale;
  std::vector<double> corrected_band_scale;
  std::vector<double> band_displacement;
  std::vector<double> raw_band_spread;
  std::vector<std::uint8_t> limiter_excess_band;
  std::vector<std::uint8_t> monotone_clamped_band;
};

namespace tracking_reference_detail {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

inline double median_in_place(std::vector<double>& values) {
  TENRYU_ASSERT(!values.empty(), "tracking reference median requires samples");
  std::sort(values.begin(), values.end());
  const std::size_t n = values.size();
  if ((n & 1U) != 0U) {
    return values[n / 2U];
  }
  return 0.5 * (values[n / 2U - 1U] + values[n / 2U]);
}

inline bool finite_nonnegative(const double x) {
  return std::isfinite(x) && x >= 0.0;
}

inline int count_bits(unsigned int mask) {
  int count = 0;
  while (mask != 0U) {
    count += static_cast<int>(mask & 1U);
    mask >>= 1U;
  }
  return count;
}

inline int reference_cell_active_nverts(const tenryu::core::State& state,
                                        const int c) {
  if (state.mesh.cell_nverts.empty()) {
    return tenryu::mesh::kMeshTopoCellStorageSlots;
  }
  return tenryu::mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
}

inline void validate_multiblock_reference_seam_metadata(
    const tenryu::mesh::MultiBlockTopology& mb) {
  TENRYU_ASSERT(mb.block_count > 0 && mb.block_count <= 31,
                "multiblock reference xi requires 1..31 topology blocks");
  TENRYU_ASSERT(!mb.seams.empty(),
                "multiblock reference xi requires seam metadata");
  int seam_index_end = 0;
  for (const tenryu::mesh::SeamInfo& seam : mb.seams) {
    TENRYU_ASSERT(seam.block_a >= 0 && seam.block_a < mb.block_count,
                  "multiblock reference xi seam block_a out of range");
    TENRYU_ASSERT(seam.block_b >= 0 && seam.block_b < mb.block_count,
                  "multiblock reference xi seam block_b out of range");
    TENRYU_ASSERT(seam.orientation == 1 || seam.orientation == -1,
                  "multiblock reference xi seam orientation must be +/-1");
    TENRYU_ASSERT(seam.index_begin == seam_index_end,
                  "multiblock reference xi seam indices must be contiguous");
    TENRYU_ASSERT(seam.index_count > 0,
                  "multiblock reference xi seam index_count must be positive");
    seam_index_end += seam.index_count;
  }
}

struct ReferenceXiSample {
  double s0 = 0.0;
  double xi = 0.0;
};

inline void validate_multiblock_reference_xi_monotone(
    const std::vector<double>& s0,
    const std::vector<double>& xi,
    const double xi_tol) {
  std::vector<ReferenceXiSample> samples(s0.size());
  for (std::size_t n = 0; n < s0.size(); ++n) {
    samples[n] = {s0[n], xi[n]};
  }
  std::sort(samples.begin(), samples.end(),
            [](const ReferenceXiSample& a, const ReferenceXiSample& b) {
              return a.s0 < b.s0;
            });
  double xi_prev = -std::numeric_limits<double>::infinity();
  for (const ReferenceXiSample& sample : samples) {
    TENRYU_ASSERT(std::isfinite(sample.s0) && std::isfinite(sample.xi),
                  "multiblock reference xi monotonicity requires finite samples");
    TENRYU_ASSERT(sample.xi + xi_tol >= xi_prev,
                  "multiblock reference xi is not monotone in initial radius");
    xi_prev = std::max(xi_prev, sample.xi);
  }
}

inline void validate_multiblock_reference_xi_seams(
    const tenryu::core::State& state,
    const std::vector<double>& r0,
    const std::vector<double>& z0,
    const std::vector<double>& xi,
    const double s_max,
    const double xi_tol) {
  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  const int n_nodes = topo.n_nodes;
  const int n_cells = topo.n_cells;

  validate_multiblock_reference_seam_metadata(mb);
  TENRYU_ASSERT(mb.cell_block_id.size() == static_cast<std::size_t>(n_cells),
                "multiblock reference xi requires cell block ids");
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "multiblock reference xi requires cell-node CSR offsets");

  std::vector<unsigned int> node_block_mask(static_cast<std::size_t>(n_nodes), 0U);
  for (int c = 0; c < n_cells; ++c) {
    const int block_id = mb.cell_block_id[static_cast<std::size_t>(c)];
    TENRYU_ASSERT(block_id >= 0 && block_id < mb.block_count,
                  "multiblock reference xi cell block id out of range");
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int next = mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    TENRYU_ASSERT(next - off == state.mesh.corner_stride,
                  "multiblock reference xi cell-node CSR width must match mesh "
                  "stride");
    TENRYU_ASSERT(off >= 0 && static_cast<std::size_t>(next) <=
                                  mb.cell_node_csr_indices.size(),
                  "multiblock reference xi cell-node CSR index out of range");
    const int active_nverts = reference_cell_active_nverts(state, c);
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "multiblock reference xi cell node out of range");
      node_block_mask[static_cast<std::size_t>(n)] |=
          1U << static_cast<unsigned int>(block_id);
    }
  }

  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    if (count_bits(node_block_mask[idx]) > 1) {
      TENRYU_ASSERT(std::isfinite(xi[idx]),
                    "multiblock reference xi seam-shared node is non-finite");
    }
  }

  const double geom_tol = xi_tol * std::max(s_max, 1.0e-300);
  std::vector<int> order(static_cast<std::size_t>(n_nodes), 0);
  for (int n = 0; n < n_nodes; ++n) {
    order[static_cast<std::size_t>(n)] = n;
  }
  std::sort(order.begin(), order.end(), [&r0](const int a, const int b) {
    return r0[static_cast<std::size_t>(a)] <
           r0[static_cast<std::size_t>(b)];
  });
  for (int ia = 0; ia < n_nodes; ++ia) {
    const int a = order[static_cast<std::size_t>(ia)];
    const std::size_t au = static_cast<std::size_t>(a);
    for (int ib = ia + 1; ib < n_nodes; ++ib) {
      const int b = order[static_cast<std::size_t>(ib)];
      const std::size_t bu = static_cast<std::size_t>(b);
      const double dr = r0[bu] - r0[au];
      if (dr > geom_tol) {
        break;
      }
      const double dz = z0[bu] - z0[au];
      if (std::abs(dz) > geom_tol) {
        continue;
      }
      if (std::hypot(dr, dz) <= geom_tol) {
        TENRYU_ASSERT(std::abs(xi[bu] - xi[au]) <= xi_tol,
                      "multiblock reference xi seam-equivalent nodes disagree");
      }
    }
  }
}

inline double ring_median_radius(const std::vector<double>& r,
                                 const std::vector<double>& z,
                                 const std::vector<std::uint8_t>& flags,
                                 const int i,
                                 const int nz,
                                 const bool exclude_poles) {
  if (i == 0) {
    return 0.0;
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(nz + 1));
  for (int j = 0; j <= nz; ++j) {
    const int n = node_index(i, j, nz);
    const auto idx = static_cast<std::size_t>(n);
    const std::uint8_t f = flags.empty() ? tenryu::mesh::NODE_NONE : flags[idx];
    if ((f & tenryu::mesh::NODE_CENTER) != 0U) {
      continue;
    }
    if (exclude_poles && (f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
      continue;
    }
    const double s = std::hypot(r[idx], z[idx]);
    if (std::isfinite(s)) {
      samples.push_back(s);
    }
  }
  if (!samples.empty()) {
    return median_in_place(samples);
  }
  if (exclude_poles) {
    return ring_median_radius(r, z, flags, i, nz, false);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

inline bool nearly_equal_scale(const double a, const double b) {
  if (!std::isfinite(a) || !std::isfinite(b)) {
    return false;
  }
  const double scale = std::max({1.0, std::abs(a), std::abs(b)});
  return std::abs(a - b) <= 16.0 * std::numeric_limits<double>::epsilon() * scale;
}

inline double local_node_scale(const std::vector<double>& r,
                               const std::vector<double>& z,
                               const int nr,
                               const int nz,
                               const int i,
                               const int j) {
  const int n = node_index(i, j, nz);
  const auto idx = static_cast<std::size_t>(n);
  double h = std::numeric_limits<double>::infinity();
  const auto observe = [&](const int ii, const int jj) {
    if (ii < 0 || ii > nr || jj < 0 || jj > nz) {
      return;
    }
    const int m = node_index(ii, jj, nz);
    const auto midx = static_cast<std::size_t>(m);
    const double dr = r[midx] - r[idx];
    const double dz = z[midx] - z[idx];
    const double d = std::hypot(dr, dz);
    if (std::isfinite(d) && d > 0.0) {
      h = std::min(h, d);
    }
  };
  observe(i - 1, j);
  observe(i + 1, j);
  observe(i, j - 1);
  observe(i, j + 1);
  return std::isfinite(h) ? h : 0.0;
}

inline void unit_vector_from_initial(const std::vector<double>& r0,
                                     const std::vector<double>& z0,
                                     const int i,
                                     const int j,
                                     const int nz,
                                     double& e_r,
                                     double& e_z) {
  const int n = node_index(i, j, nz);
  const auto idx = static_cast<std::size_t>(n);
  const double s0 = std::hypot(r0[idx], z0[idx]);
  if (std::isfinite(s0) && s0 > 0.0) {
    e_r = r0[idx] / s0;
    e_z = z0[idx] / s0;
    return;
  }
  constexpr double pi = 3.141592653589793238462643383279502884;
  const double theta = pi * static_cast<double>(j) / static_cast<double>(nz);
  e_r = std::sin(theta);
  e_z = std::cos(theta);
}

inline double projected_ring_median_radius(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<double>& r0,
    const std::vector<double>& z0,
    const std::vector<std::uint8_t>& flags,
    const int i,
    const int nz,
    const bool exclude_poles) {
  if (i == 0) {
    return 0.0;
  }
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(nz + 1));
  for (int j = 0; j <= nz; ++j) {
    const int n = node_index(i, j, nz);
    const auto idx = static_cast<std::size_t>(n);
    const std::uint8_t f = flags.empty() ? tenryu::mesh::NODE_NONE : flags[idx];
    if ((f & tenryu::mesh::NODE_CENTER) != 0U) {
      continue;
    }
    if (exclude_poles && (f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
      continue;
    }
    double e_r = 0.0;
    double e_z = 1.0;
    unit_vector_from_initial(r0, z0, i, j, nz, e_r, e_z);
    const double p = r[idx] * e_r + z[idx] * e_z;
    if (std::isfinite(p)) {
      samples.push_back(p);
    }
  }
  if (!samples.empty()) {
    return median_in_place(samples);
  }
  if (exclude_poles) {
    return projected_ring_median_radius(r, z, r0, z0, flags, i, nz, false);
  }
  return std::numeric_limits<double>::quiet_NaN();
}

inline double local_ring_spacing(const std::vector<double>& ring,
                                 const std::vector<double>& fallback,
                                 const int i,
                                 const int nr) {
  double h = std::numeric_limits<double>::infinity();
  const auto observe = [&](const double d_raw) {
    const double d = std::abs(d_raw);
    if (std::isfinite(d) && d > 0.0) {
      h = std::min(h, d);
    }
  };
  if (i > 0) {
    observe(ring[static_cast<std::size_t>(i)] -
            ring[static_cast<std::size_t>(i - 1)]);
  }
  if (i < nr) {
    observe(ring[static_cast<std::size_t>(i + 1)] -
            ring[static_cast<std::size_t>(i)]);
  }
  if (i > 0) {
    observe(fallback[static_cast<std::size_t>(i)] -
            fallback[static_cast<std::size_t>(i - 1)]);
  }
  if (i < nr) {
    observe(fallback[static_cast<std::size_t>(i + 1)] -
            fallback[static_cast<std::size_t>(i)]);
  }
  return std::isfinite(h) ? h : 0.0;
}

inline double edge_ring_spacing(const std::vector<double>& ring,
                                const std::vector<double>& fallback,
                                const int edge) {
  double h = std::numeric_limits<double>::infinity();
  const auto observe = [&](const double d_raw) {
    const double d = std::abs(d_raw);
    if (std::isfinite(d) && d > 0.0) {
      h = std::min(h, d);
    }
  };
  observe(ring[static_cast<std::size_t>(edge)] -
          ring[static_cast<std::size_t>(edge - 1)]);
  observe(fallback[static_cast<std::size_t>(edge)] -
          fallback[static_cast<std::size_t>(edge - 1)]);
  return std::isfinite(h) ? h : 0.0;
}

inline bool monotone_project_ring_scale(
    std::vector<double>& s,
    const std::vector<double>& center,
    const std::vector<double>& cap,
    const std::vector<double>& edge_floor,
    double& max_delta,
    double& max_excess) {
  const int nr = static_cast<int>(s.size()) - 1;
  if (nr <= 0) {
    return false;
  }
  const double outer = center[static_cast<std::size_t>(nr)];
  if (!std::isfinite(outer) || !(outer > 0.0)) {
    return false;
  }

  std::vector<double> feasible_hi(static_cast<std::size_t>(nr + 1), outer);
  feasible_hi[static_cast<std::size_t>(nr)] = outer;
  for (int i = nr - 1; i >= 1; --i) {
    const double hi_cap =
        center[static_cast<std::size_t>(i)] + cap[static_cast<std::size_t>(i)];
    const double hi_next =
        feasible_hi[static_cast<std::size_t>(i + 1)] -
        edge_floor[static_cast<std::size_t>(i + 1)];
    feasible_hi[static_cast<std::size_t>(i)] = std::min(hi_cap, hi_next);
  }

  s[0] = 0.0;
  for (int i = 1; i < nr; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const double lo_cap = center[idx] - cap[idx];
    const double lo_prev =
        s[static_cast<std::size_t>(i - 1)] + edge_floor[idx];
    const double lo = std::max(lo_cap, lo_prev);
    const double hi = feasible_hi[idx];
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) {
      return false;
    }
    const double candidate = std::isfinite(s[idx]) ? s[idx] : center[idx];
    s[idx] = std::clamp(candidate, lo, hi);
  }
  s[static_cast<std::size_t>(nr)] = outer;
  for (int i = 1; i <= nr; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const double spacing =
        s[idx] - s[static_cast<std::size_t>(i - 1)];
    if (!std::isfinite(spacing) || spacing < edge_floor[idx]) {
      return false;
    }
    const double delta = std::abs(s[idx] - center[idx]);
    const double excess = delta - cap[idx];
    max_delta = std::max(max_delta, delta);
    max_excess = std::max(max_excess, excess);
  }
  return true;
}

inline bool multiblock_differential_reference_diag_enabled() {
  const char* const raw = std::getenv("TENRYU_I1B_DIFFREF_DIAG");
  if (raw == nullptr) {
    return false;
  }
  const std::string value(raw);
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "yes" || value == "YES" || value == "on" ||
         value == "ON";
}

inline int reference_xi_band(const double xi, const int band_count) {
  TENRYU_ASSERT(std::isfinite(xi),
                "multiblock differential reference xi must be finite");
  TENRYU_ASSERT(band_count > 0,
                "multiblock differential reference band_count must be positive");
  const double scaled = xi * static_cast<double>(band_count);
  return std::clamp(static_cast<int>(std::floor(scaled)), 0, band_count - 1);
}

inline bool fill_missing_band_values(std::vector<double>& values,
                                     const std::vector<int>& counts) {
  const int n = static_cast<int>(values.size());
  TENRYU_ASSERT(counts.size() == values.size(),
                "multiblock differential reference band counts size mismatch");
  std::vector<int> known;
  known.reserve(values.size());
  for (int i = 0; i < n; ++i) {
    if (counts[static_cast<std::size_t>(i)] > 0 &&
        std::isfinite(values[static_cast<std::size_t>(i)])) {
      known.push_back(i);
    }
  }
  if (known.empty()) {
    return false;
  }

  const int first = known.front();
  for (int i = 0; i < first; ++i) {
    values[static_cast<std::size_t>(i)] =
        values[static_cast<std::size_t>(first)];
  }
  for (std::size_t k = 0; k + 1U < known.size(); ++k) {
    const int lo = known[k];
    const int hi = known[k + 1U];
    const double vlo = values[static_cast<std::size_t>(lo)];
    const double vhi = values[static_cast<std::size_t>(hi)];
    const double inv_span = 1.0 / static_cast<double>(hi - lo);
    for (int i = lo + 1; i < hi; ++i) {
      const double t = static_cast<double>(i - lo) * inv_span;
      values[static_cast<std::size_t>(i)] = (1.0 - t) * vlo + t * vhi;
    }
  }
  const int last = known.back();
  for (int i = last + 1; i < n; ++i) {
    values[static_cast<std::size_t>(i)] =
        values[static_cast<std::size_t>(last)];
  }
  return true;
}

inline double local_band_spacing(const std::vector<double>& band,
                                 const std::vector<double>& fallback,
                                 const int i) {
  const int n = static_cast<int>(band.size());
  double h = std::numeric_limits<double>::infinity();
  const auto observe = [&](const double d_raw) {
    const double d = std::abs(d_raw);
    if (std::isfinite(d) && d > 0.0) {
      h = std::min(h, d);
    }
  };
  if (i > 0) {
    observe(band[static_cast<std::size_t>(i)] -
            band[static_cast<std::size_t>(i - 1)]);
    observe(fallback[static_cast<std::size_t>(i)] -
            fallback[static_cast<std::size_t>(i - 1)]);
  }
  if (i + 1 < n) {
    observe(band[static_cast<std::size_t>(i + 1)] -
            band[static_cast<std::size_t>(i)]);
    observe(fallback[static_cast<std::size_t>(i + 1)] -
            fallback[static_cast<std::size_t>(i)]);
  }
  return std::isfinite(h) ? h : 0.0;
}

inline double edge_band_spacing(const std::vector<double>& band,
                                const std::vector<double>& fallback,
                                const int edge) {
  double h = std::numeric_limits<double>::infinity();
  const auto observe = [&](const double d_raw) {
    const double d = std::abs(d_raw);
    if (std::isfinite(d) && d > 0.0) {
      h = std::min(h, d);
    }
  };
  observe(band[static_cast<std::size_t>(edge)] -
          band[static_cast<std::size_t>(edge - 1)]);
  observe(fallback[static_cast<std::size_t>(edge)] -
          fallback[static_cast<std::size_t>(edge - 1)]);
  return std::isfinite(h) ? h : 0.0;
}

inline bool band_scale_raw_admissible(
    const std::vector<double>& raw,
    const std::vector<double>& edge_floor,
    const double first_floor) {
  const int n = static_cast<int>(raw.size());
  if (n <= 0 || edge_floor.size() != raw.size()) {
    return false;
  }
  if (!std::isfinite(raw.front()) || raw.front() < first_floor) {
    return false;
  }
  for (int i = 1; i < n; ++i) {
    const double spacing =
        raw[static_cast<std::size_t>(i)] -
        raw[static_cast<std::size_t>(i - 1)];
    if (!std::isfinite(raw[static_cast<std::size_t>(i)]) ||
        !std::isfinite(spacing) || spacing < edge_floor[static_cast<std::size_t>(i)]) {
      return false;
    }
  }
  return true;
}

inline bool monotone_project_band_scale(
    std::vector<double>& s,
    const std::vector<double>& center,
    const std::vector<double>& cap,
    const std::vector<double>& edge_floor,
    const double first_floor,
    double& max_delta,
    double& max_excess,
    int& monotone_clamp_count,
    std::vector<std::uint8_t>& monotone_clamped) {
  const int n = static_cast<int>(s.size());
  if (n <= 1 || center.size() != s.size() || cap.size() != s.size() ||
      edge_floor.size() != s.size()) {
    return false;
  }
  monotone_clamped.assign(s.size(), 0U);
  const double outer = center.back();
  if (!std::isfinite(outer) || !(outer > 0.0)) {
    return false;
  }

  std::vector<double> feasible_hi(static_cast<std::size_t>(n), outer);
  feasible_hi[static_cast<std::size_t>(n - 1)] = outer;
  for (int i = n - 2; i >= 0; --i) {
    const auto idx = static_cast<std::size_t>(i);
    const double hi_cap = center[idx] + cap[idx];
    const double hi_next =
        feasible_hi[static_cast<std::size_t>(i + 1)] -
        edge_floor[static_cast<std::size_t>(i + 1)];
    feasible_hi[idx] = std::min(hi_cap, hi_next);
  }

  for (int i = 0; i < n - 1; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const double lo_cap = center[idx] - cap[idx];
    const double lo_prev =
        i == 0 ? first_floor
               : s[static_cast<std::size_t>(i - 1)] + edge_floor[idx];
    const double lo = std::max(lo_cap, lo_prev);
    const double hi = feasible_hi[idx];
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) {
      return false;
    }
    const double before = s[idx];
    const double candidate = std::isfinite(before) ? before : center[idx];
    s[idx] = std::clamp(candidate, lo, hi);
    if (!nearly_equal_scale(before, s[idx])) {
      monotone_clamped[idx] = 1U;
      ++monotone_clamp_count;
    }
  }
  const double outer_before = s[static_cast<std::size_t>(n - 1)];
  s[static_cast<std::size_t>(n - 1)] = outer;
  if (!nearly_equal_scale(outer_before, outer)) {
    monotone_clamped[static_cast<std::size_t>(n - 1)] = 1U;
    ++monotone_clamp_count;
  }

  for (int i = 0; i < n; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    if (!std::isfinite(s[idx])) {
      return false;
    }
    if (i > 0) {
      const double spacing =
          s[idx] - s[static_cast<std::size_t>(i - 1)];
      if (!std::isfinite(spacing) || spacing < edge_floor[idx]) {
        return false;
      }
    }
    const double delta = std::abs(s[idx] - center[idx]);
    const double excess = delta - cap[idx];
    max_delta = std::max(max_delta, delta);
    if (excess > 0.0) {
      max_excess = std::max(max_excess, excess);
    }
  }
  return true;
}

inline void log_multiblock_differential_band_scales(
    const tenryu::core::State& state,
    const MultiblockDifferentialBandScaleResult& result) {
  for (int i = 0; i < result.band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    tenryu::core::log_info(
        std::string("[multiblock_diffref_band] step=") +
        std::to_string(state.step) +
        " band=" + std::to_string(i) +
        " samples=" + std::to_string(result.band_sample_count[idx]) +
        " s_i_raw=" + std::to_string(result.raw_band_scale[idx]) +
        " S_i=" + std::to_string(result.corrected_band_scale[idx]) +
        " d_i=" + std::to_string(result.band_displacement[idx]) +
        " raw_spread=" + std::to_string(result.raw_band_spread[idx]) +
        " limiter_excess_band=" +
        std::to_string(result.limiter_excess_band[idx] != 0U ? 1 : 0) +
        " monotone_clamped_band=" +
        std::to_string(result.monotone_clamped_band[idx] != 0U ? 1 : 0) +
        " limiter_excess_count=" +
        std::to_string(result.limiter_excess_count) +
        " monotone_clamp_count=" +
        std::to_string(result.monotone_clamp_count));
  }
}

inline void build_radial_target_from_ring_scale(
    const std::vector<double>& ring_scale,
    const std::vector<double>& r0,
    const std::vector<double>& z0,
    const std::vector<std::uint8_t>& flags,
    const int nr,
    const int nz,
    std::vector<double>& r_target,
    std::vector<double>& z_target) {
  for (int i = 0; i <= nr; ++i) {
    const double s = ring_scale[static_cast<std::size_t>(i)];
    for (int j = 0; j <= nz; ++j) {
      const int n = node_index(i, j, nz);
      const auto idx = static_cast<std::size_t>(n);
      const std::uint8_t f = flags[idx];
      if ((f & tenryu::mesh::NODE_CENTER) != 0U) {
        r_target[idx] = 0.0;
        z_target[idx] = 0.0;
        continue;
      }
      double e_r = 0.0;
      double e_z = 1.0;
      unit_vector_from_initial(r0, z0, i, j, nz, e_r, e_z);
      r_target[idx] = s * e_r;
      z_target[idx] = s * e_z;
      if ((f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
        r_target[idx] = 0.0;
      }
    }
  }
}

inline bool build_seamless_converging_ring_scale(
    const std::vector<double>& r_lag,
    const std::vector<double>& z_lag,
    const std::vector<double>& r0,
    const std::vector<double>& z0,
    const std::vector<std::uint8_t>& flags,
    const tenryu::core::Config& cfg,
    const int nr,
    const int nz,
    TriFanTrackingReferenceResult& result) {
  const auto& ale = cfg.numerics.ale;
  TENRYU_ASSERT(nearly_equal_scale(ale.tri_fan_tracking_reference_beta, 1.0),
                "tri_fan seamless_converging beta history is not implemented");

  std::vector<double> r_ring(static_cast<std::size_t>(nr + 1), 0.0);
  std::vector<double> q_ring(static_cast<std::size_t>(nr + 1), 0.0);
  for (int i = 1; i <= nr; ++i) {
    const double projected =
        projected_ring_median_radius(r_lag, z_lag, r0, z0, flags, i, nz, true);
    const double initial =
        ring_median_radius(r0, z0, flags, i, nz, true);
    if (!finite_nonnegative(projected) || !(std::isfinite(initial) && initial > 0.0)) {
      return false;
    }
    r_ring[static_cast<std::size_t>(i)] = projected;
    q_ring[static_cast<std::size_t>(i)] = initial;
    result.raw_ring_scale[static_cast<std::size_t>(i)] = projected;
  }
  if (!(r_ring[static_cast<std::size_t>(nr)] > 0.0)) {
    return false;
  }

  std::vector<double> alpha(static_cast<std::size_t>(nr + 1), 1.0);
  for (int i = 1; i <= nr; ++i) {
    alpha[static_cast<std::size_t>(i)] =
        r_ring[static_cast<std::size_t>(i)] / q_ring[static_cast<std::size_t>(i)];
  }
  alpha[0] = alpha[1];

  std::vector<double> edge_weight(static_cast<std::size_t>(nr), 1.0);
  const double g0 = ale.tri_fan_tracking_reference_g0;
  for (int i = 0; i < nr; ++i) {
    const double g = std::abs(alpha[static_cast<std::size_t>(i + 1)] -
                              alpha[static_cast<std::size_t>(i)]);
    const double x = g / g0;
    edge_weight[static_cast<std::size_t>(i)] = 1.0 / (1.0 + x * x);
  }

  std::vector<double> ring_weight(static_cast<std::size_t>(nr + 1), 0.0);
  for (int i = 1; i < nr; ++i) {
    ring_weight[static_cast<std::size_t>(i)] =
        std::min(edge_weight[static_cast<std::size_t>(i - 1)],
                 edge_weight[static_cast<std::size_t>(i)]);
  }

  std::vector<double> fit = r_ring;
  std::vector<double> tmp = fit;
  constexpr int kSmoothPasses = 8;
  for (int pass = 0; pass < kSmoothPasses; ++pass) {
    tmp = fit;
    for (int i = 1; i < nr; ++i) {
      const double w = 0.5 * ring_weight[static_cast<std::size_t>(i)];
      const double smooth =
          0.5 * (fit[static_cast<std::size_t>(i - 1)] +
                 fit[static_cast<std::size_t>(i + 1)]);
      tmp[static_cast<std::size_t>(i)] =
          fit[static_cast<std::size_t>(i)] +
          w * (smooth - fit[static_cast<std::size_t>(i)]);
    }
    tmp[0] = 0.0;
    tmp[static_cast<std::size_t>(nr)] = r_ring[static_cast<std::size_t>(nr)];
    fit.swap(tmp);
  }

  std::vector<double> s = r_ring;
  std::vector<double> cap(static_cast<std::size_t>(nr + 1), 0.0);
  const double nu = ale.tri_fan_tracking_reference_nu;
  const double eps_v = ale.tri_fan_tracking_reference_eps_v;
  for (int i = 1; i < nr; ++i) {
    const double h = local_ring_spacing(r_ring, q_ring, i, nr);
    const double volume_relax = 1.0 + 2.0 * ring_weight[static_cast<std::size_t>(i)];
    cap[static_cast<std::size_t>(i)] =
        std::min(nu * h, eps_v * volume_relax * h);
    const double raw_d =
        ring_weight[static_cast<std::size_t>(i)] *
        (fit[static_cast<std::size_t>(i)] - r_ring[static_cast<std::size_t>(i)]);
    const double d =
        std::clamp(raw_d,
                   -cap[static_cast<std::size_t>(i)],
                   cap[static_cast<std::size_t>(i)]);
    s[static_cast<std::size_t>(i)] = r_ring[static_cast<std::size_t>(i)] + d;
  }
  s[0] = 0.0;
  s[static_cast<std::size_t>(nr)] = r_ring[static_cast<std::size_t>(nr)];

  double scale = std::abs(r_ring[static_cast<std::size_t>(nr)]);
  scale = std::max(scale, std::abs(q_ring[static_cast<std::size_t>(nr)]));
  for (int i = 1; i <= nr; ++i) {
    scale = std::max(scale, std::abs(r_ring[static_cast<std::size_t>(i)]));
    scale = std::max(scale, std::abs(q_ring[static_cast<std::size_t>(i)]));
  }
  const double tiny_floor =
      64.0 * std::numeric_limits<double>::epsilon() * std::max(scale, 1.0e-300);
  std::vector<double> edge_floor(static_cast<std::size_t>(nr + 1), tiny_floor);
  for (int i = 1; i <= nr; ++i) {
    const double h = edge_ring_spacing(r_ring, q_ring, i);
    const double rel_floor = cfg.numerics.ale.reference_corner_j_floor_rel * h;
    edge_floor[static_cast<std::size_t>(i)] =
        std::max(tiny_floor, std::isfinite(rel_floor) ? rel_floor : 0.0);
  }

  if (!monotone_project_ring_scale(s,
                                   r_ring,
                                   cap,
                                   edge_floor,
                                   result.max_limited_delta,
                                   result.max_limiter_excess)) {
    return false;
  }
  result.filtered_ring_scale = s;
  return true;
}

inline std::vector<double> compute_reference_volumes(
    const std::vector<double>& r_ref,
    const std::vector<double>& z_ref,
    const std::vector<std::uint8_t>& cell_nverts,
    const int nr,
    const int nz) {
  std::vector<double> vol(static_cast<std::size_t>(nr * nz), 0.0);
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      const int n00 = node_index(i, j, nz);
      const int n10 = node_index(i + 1, j, nz);
      const int n11 = node_index(i + 1, j + 1, nz);
      const int n01 = node_index(i, j + 1, nz);
      const int active_nverts =
          tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
      const double signed_vol =
          active_nverts == 3
              ? tenryu::mesh::rz_triangle_volume_exact(
                    r_ref[static_cast<std::size_t>(n00)],
                    z_ref[static_cast<std::size_t>(n00)],
                    r_ref[static_cast<std::size_t>(n10)],
                    z_ref[static_cast<std::size_t>(n10)],
                    r_ref[static_cast<std::size_t>(n11)],
                    z_ref[static_cast<std::size_t>(n11)])
              : tenryu::mesh::rz_quad_volume_exact(
                    r_ref[static_cast<std::size_t>(n00)],
                    z_ref[static_cast<std::size_t>(n00)],
                    r_ref[static_cast<std::size_t>(n10)],
                    z_ref[static_cast<std::size_t>(n10)],
                    r_ref[static_cast<std::size_t>(n11)],
                    z_ref[static_cast<std::size_t>(n11)],
                    r_ref[static_cast<std::size_t>(n01)],
                    z_ref[static_cast<std::size_t>(n01)]);
      const double v = -signed_vol;
      TENRYU_ASSERT(std::isfinite(v) && v > 0.0,
                    "tri_fan tracking reference produced invalid reference volume");
      vol[static_cast<std::size_t>(c)] = v;
    }
  }
  return vol;
}

inline double outer_band_max_log_volume_ratio(
    const std::vector<double>& vol_ref,
    const std::vector<double>& vol_lag,
    const int nr,
    const int nz) {
  double max_abs_log = 0.0;
  for (int i = 0; i < nr; ++i) {
    if (!(i > nr / 2)) {
      continue;
    }
    double ref_sum = 0.0;
    double lag_sum = 0.0;
    for (int j = 0; j < nz; ++j) {
      const auto idx = static_cast<std::size_t>(i * nz + j);
      ref_sum += vol_ref[idx];
      lag_sum += vol_lag[idx];
    }
    if (std::isfinite(ref_sum) && std::isfinite(lag_sum) &&
        ref_sum > 0.0 && lag_sum > 0.0) {
      max_abs_log = std::max(max_abs_log, std::abs(std::log(ref_sum / lag_sum)));
    }
  }
  return max_abs_log;
}

inline double angular_spread_over_h_max(
    const std::vector<double>& r_lag,
    const std::vector<double>& z_lag,
    const std::vector<double>& r0,
    const std::vector<double>& z0,
    const std::vector<std::uint8_t>& flags,
    const int nr,
    const int nz) {
  std::vector<double> r_ring(static_cast<std::size_t>(nr + 1), 0.0);
  std::vector<double> q_ring(static_cast<std::size_t>(nr + 1), 0.0);
  for (int i = 1; i <= nr; ++i) {
    r_ring[static_cast<std::size_t>(i)] =
        projected_ring_median_radius(r_lag, z_lag, r0, z0, flags, i, nz, true);
    q_ring[static_cast<std::size_t>(i)] =
        ring_median_radius(r0, z0, flags, i, nz, true);
  }

  double max_ratio = 0.0;
  for (int i = 1; i <= nr; ++i) {
    double sum = 0.0;
    double sum2 = 0.0;
    int count = 0;
    for (int j = 0; j <= nz; ++j) {
      const int n = node_index(i, j, nz);
      const auto idx = static_cast<std::size_t>(n);
      const std::uint8_t f = flags.empty() ? tenryu::mesh::NODE_NONE : flags[idx];
      if ((f & tenryu::mesh::NODE_CENTER) != 0U ||
          (f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
        continue;
      }
      double e_r = 0.0;
      double e_z = 1.0;
      unit_vector_from_initial(r0, z0, i, j, nz, e_r, e_z);
      const double p = r_lag[idx] * e_r + z_lag[idx] * e_z;
      if (!std::isfinite(p)) {
        continue;
      }
      sum += p;
      sum2 += p * p;
      ++count;
    }
    if (count <= 1) {
      continue;
    }
    const double inv_count = 1.0 / static_cast<double>(count);
    const double mean = sum * inv_count;
    const double variance = std::max(0.0, sum2 * inv_count - mean * mean);
    const double h = local_ring_spacing(r_ring, q_ring, i, nr);
    if (std::isfinite(h) && h > 0.0) {
      max_ratio = std::max(max_ratio, std::sqrt(variance) / h);
    }
  }
  return max_ratio;
}

inline int shock_width_proxy_rings(
    const std::vector<double>& r_lag,
    const std::vector<double>& z_lag,
    const std::vector<double>& r0,
    const std::vector<double>& z0,
    const std::vector<std::uint8_t>& flags,
    const double g0,
    const int nr,
    const int nz) {
  if (!(g0 > 0.0) || !std::isfinite(g0)) {
    return 0;
  }
  std::vector<double> alpha(static_cast<std::size_t>(nr + 1), 1.0);
  for (int i = 1; i <= nr; ++i) {
    const double projected =
        projected_ring_median_radius(r_lag, z_lag, r0, z0, flags, i, nz, true);
    const double initial = ring_median_radius(r0, z0, flags, i, nz, true);
    if (std::isfinite(projected) && std::isfinite(initial) && initial > 0.0) {
      alpha[static_cast<std::size_t>(i)] = projected / initial;
    }
  }
  if (nr > 0) {
    alpha[0] = alpha[1];
  }
  int count = 0;
  for (int i = 0; i < nr; ++i) {
    const double jump = std::abs(alpha[static_cast<std::size_t>(i + 1)] -
                                 alpha[static_cast<std::size_t>(i)]);
    if (std::isfinite(jump) && jump > g0) {
      ++count;
    }
  }
  return count;
}

inline tenryu::mesh::CandidateMeshQuality reference_displacement_quality(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const std::vector<double>& delta_r,
    const std::vector<double>& delta_z,
    const int nr,
    const int nz,
    const int n_nodes,
    const int n_cells,
    double& sigma_accepted,
    int& linesearch_iters) {
  double* d_delta_r = nullptr;
  double* d_delta_z = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  tracking_reference_detail::cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_delta_r), node_bytes),
      "tri_fan reference cudaMalloc delta_r failed");
  tracking_reference_detail::cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_delta_z), node_bytes),
      "tri_fan reference cudaMalloc delta_z failed");
  tracking_reference_detail::cuda_check(
      cudaMemcpy(d_delta_r, delta_r.data(), node_bytes, cudaMemcpyHostToDevice),
      "tri_fan reference delta_r H2D failed");
  tracking_reference_detail::cuda_check(
      cudaMemcpy(d_delta_z, delta_z.data(), node_bytes, cudaMemcpyHostToDevice),
      "tri_fan reference delta_z H2D failed");
  tracking_reference_detail::cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_cell_nverts),
                 static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)),
      "tri_fan reference cudaMalloc cell_nverts failed");
  tracking_reference_detail::cuda_check(
      cudaMemcpy(d_cell_nverts,
                 state.mesh.cell_nverts.data(),
                 static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                 cudaMemcpyHostToDevice),
      "tri_fan reference cell_nverts H2D failed");

  tenryu::mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;
  tenryu::mesh::CandidateMeshQuality quality =
      tenryu::mesh::evaluate_candidate_mesh_quality(state.x_r.data(),
                                                    state.x_z.data(),
                                                    d_delta_r,
                                                    d_delta_z,
                                                    1.0,
                                                    nr,
                                                    nz,
                                                    floors,
                                                    d_cell_nverts,
                                                    state.x_r_reference.size() ==
                                                                state.x_r.size() &&
                                                            state.x_z_reference.size() ==
                                                                state.x_z.size()
                                                        ? state.x_r_reference.data()
                                                        : nullptr,
                                                    state.x_r_reference.size() ==
                                                                state.x_r.size() &&
                                                            state.x_z_reference.size() ==
                                                                state.x_z.size()
                                                        ? state.x_z_reference.data()
                                                        : nullptr);
  sigma_accepted = 1.0;
  linesearch_iters = 0;
  if (!quality.admissible()) {
    const auto ls = tenryu::mesh::linesearch_largest_admissible_sigma(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        1.0,
        0.0,
        cfg.numerics.ale.reference_linesearch_max_iters,
        nr,
        nz,
        floors,
        d_cell_nverts,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    sigma_accepted = ls.sigma_accepted;
    linesearch_iters = ls.iters_used;
    quality = ls.quality;
  }
  if (sigma_accepted == 0.0 && !quality.admissible()) {
    quality = tenryu::mesh::evaluate_candidate_mesh_quality(state.x_r.data(),
                                                            state.x_z.data(),
                                                            d_delta_r,
                                                            d_delta_z,
                                                            0.0,
                                                            nr,
                                                            nz,
                                                            floors,
                                                            d_cell_nverts,
                                                            state.x_r_reference.size() ==
                                                                        state.x_r.size() &&
                                                                    state.x_z_reference.size() ==
                                                                        state.x_z.size()
                                                                ? state.x_r_reference.data()
                                                                : nullptr,
                                                            state.x_r_reference.size() ==
                                                                        state.x_r.size() &&
                                                                    state.x_z_reference.size() ==
                                                                        state.x_z.size()
                                                                ? state.x_z_reference.data()
                                                                : nullptr);
    TENRYU_ASSERT(quality.admissible(),
                  "tri_fan reference current mesh is inadmissible");
  }

  tracking_reference_detail::cuda_check(
      cudaFree(d_cell_nverts), "tri_fan reference cudaFree cell_nverts failed");
  tracking_reference_detail::cuda_check(
      cudaFree(d_delta_z), "tri_fan reference cudaFree delta_z failed");
  tracking_reference_detail::cuda_check(
      cudaFree(d_delta_r), "tri_fan reference cudaFree delta_r failed");
  return quality;
}

inline double cross2_host(const double ar,
                          const double az,
                          const double br,
                          const double bz) {
  return ar * bz - az * br;
}

inline double cell_corner_jacobian_ratio_host(
    const std::vector<double>& r,
    const std::vector<double>& z,
    const std::vector<std::uint8_t>& cell_nverts,
    const int ci,
    const int cj,
    const int nz) {
  const int c = ci * nz + cj;
  const int n00 = node_index(ci, cj, nz);
  const int n10 = node_index(ci + 1, cj, nz);
  const int n11 = node_index(ci + 1, cj + 1, nz);
  const int n01 = node_index(ci, cj + 1, nz);
  const int active_nverts =
      tenryu::mesh::mesh_topo_cell_active_nverts(cell_nverts, c);
  const int nodes[4] = {n00, n10, n11, n01};
  double rr[4]{};
  double zz[4]{};
  for (int k = 0; k < active_nverts; ++k) {
    const auto idx = static_cast<std::size_t>(nodes[k]);
    rr[k] = r[idx];
    zz[k] = z[idx];
  }

  double corner_j[4]{};
  if (active_nverts == 3) {
    for (int k = 0; k < 3; ++k) {
      const int kp = (k + 1) % 3;
      const int km = (k + 2) % 3;
      corner_j[k] = cross2_host(rr[kp] - rr[k],
                                zz[kp] - zz[k],
                                rr[km] - rr[k],
                                zz[km] - zz[k]);
    }
  } else {
    corner_j[0] = cross2_host(rr[1] - rr[0], zz[1] - zz[0],
                              rr[3] - rr[0], zz[3] - zz[0]);
    corner_j[1] = cross2_host(rr[1] - rr[0], zz[1] - zz[0],
                              rr[2] - rr[1], zz[2] - zz[1]);
    corner_j[2] = cross2_host(rr[2] - rr[3], zz[2] - zz[3],
                              rr[2] - rr[1], zz[2] - zz[1]);
    corner_j[3] = cross2_host(rr[2] - rr[3], zz[2] - zz[3],
                              rr[3] - rr[0], zz[3] - zz[0]);
  }

  double min_j = std::numeric_limits<double>::infinity();
  double max_j = 0.0;
  for (int k = 0; k < active_nverts; ++k) {
    const double j = -corner_j[k];
    if (!std::isfinite(j) || !(j > 0.0)) {
      return -std::numeric_limits<double>::infinity();
    }
    min_j = std::min(min_j, j);
    max_j = std::max(max_j, j);
  }
  if (!(max_j > 0.0) || !std::isfinite(min_j) || !std::isfinite(max_j)) {
    return -std::numeric_limits<double>::infinity();
  }
  return min_j / max_j;
}

inline double min_bulk_corner_jacobian_ratio(
    const std::vector<double>& r_ref,
    const std::vector<double>& z_ref,
    const std::vector<std::uint8_t>& cell_nverts,
    const int nr,
    const int nz,
    const int center_node_ring_max) {
  double min_ratio = std::numeric_limits<double>::infinity();
  const int first_bulk_cell =
      center_node_ring_max >= nr ? nr : center_node_ring_max + 1;
  for (int ci = first_bulk_cell; ci < nr; ++ci) {
    for (int cj = 0; cj < nz; ++cj) {
      min_ratio =
          std::min(min_ratio,
                   cell_corner_jacobian_ratio_host(
                       r_ref, z_ref, cell_nverts, ci, cj, nz));
    }
  }
  return min_ratio;
}

inline double multiblock_cell_orientation_sign(
    const tenryu::mesh::MultiBlockTopology& mb,
    const int cell) {
  const auto idx = static_cast<std::size_t>(cell);
  TENRYU_ASSERT(idx < mb.cell_orientation_sign.size(),
                "multiblock differential reference orientation sign index out of range");
  const int sign = mb.cell_orientation_sign[idx];
  TENRYU_ASSERT(sign == 1 || sign == -1,
                "multiblock differential reference orientation sign must be +/-1");
  return static_cast<double>(sign);
}

inline double interpolate_band_displacement(
    const std::vector<double>& band_displacement,
    const double xi) {
  const int band_count = static_cast<int>(band_displacement.size());
  TENRYU_ASSERT(band_count > 0,
                "multiblock differential reference displacement bands are empty");
  const int band = reference_xi_band(xi, band_count);
  const double inv_k = 1.0 / static_cast<double>(band_count);
  const double center = (static_cast<double>(band) + 0.5) * inv_k;
  if (band == 0 && xi <= center) {
    return band_displacement.front();
  }
  if (band == band_count - 1 && xi >= center) {
    return band_displacement.back();
  }

  if (xi < center) {
    const int lo = band - 1;
    const double lo_center = (static_cast<double>(lo) + 0.5) * inv_k;
    const double t = (xi - lo_center) / (center - lo_center);
    return (1.0 - t) * band_displacement[static_cast<std::size_t>(lo)] +
           t * band_displacement[static_cast<std::size_t>(band)];
  }

  const int hi = band + 1;
  const double hi_center = (static_cast<double>(hi) + 0.5) * inv_k;
  const double t = (xi - center) / (hi_center - center);
  return (1.0 - t) * band_displacement[static_cast<std::size_t>(band)] +
         t * band_displacement[static_cast<std::size_t>(hi)];
}

inline std::vector<double> multiblock_node_spacing(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const std::vector<double>& r,
    const std::vector<double>& z) {
  const auto& topo = state.mesh.topo;
  const int n_nodes = topo.n_nodes;
  const int n_cells = topo.n_cells;
  TENRYU_ASSERT(r.size() == static_cast<std::size_t>(n_nodes) &&
                    z.size() == static_cast<std::size_t>(n_nodes),
                "multiblock differential reference node spacing size mismatch");

  std::vector<double> h(static_cast<std::size_t>(n_nodes),
                        std::numeric_limits<double>::infinity());
  for (int c = 0; c < n_cells; ++c) {
    const auto nodes = tenryu::mesh::mesh_topo_cell_corner_nodes_n(
        topo, state.mesh.cell_nverts, c, cfg.mesh);
    const int active_nverts = nodes.count;
    for (int k = 0; k < active_nverts; ++k) {
      const int n0 = nodes.values[static_cast<std::size_t>(k)];
      const int n1 =
          nodes.values[static_cast<std::size_t>((k + 1) % active_nverts)];
      TENRYU_ASSERT(n0 >= 0 && n0 < n_nodes && n1 >= 0 && n1 < n_nodes,
                    "multiblock differential reference cell node out of range");
      const auto i0 = static_cast<std::size_t>(n0);
      const auto i1 = static_cast<std::size_t>(n1);
      const double d = std::hypot(r[i1] - r[i0], z[i1] - z[i0]);
      if (std::isfinite(d) && d > 0.0) {
        h[i0] = std::min(h[i0], d);
        h[i1] = std::min(h[i1], d);
      }
    }
  }
  for (double& value : h) {
    if (!std::isfinite(value)) {
      value = 0.0;
    }
  }
  return h;
}

inline bool multiblock_has_tri_cell_nverts(
    const tenryu::core::State& state) {
  const int n_cells = state.mesh.topo.n_cells;
  if (state.mesh.cell_nverts.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }
  return std::any_of(state.mesh.cell_nverts.begin(),
                     state.mesh.cell_nverts.end(),
                     [](const std::uint8_t nverts) { return nverts == 3U; });
}

inline std::vector<double> compute_multiblock_reference_volumes(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const std::vector<double>& r_ref,
    const std::vector<double>& z_ref) {
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "multiblock differential reference requires multiblock topology");
  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  const int n_nodes = topo.n_nodes;
  const int n_cells = topo.n_cells;
  TENRYU_ASSERT(r_ref.size() == static_cast<std::size_t>(n_nodes) &&
                    z_ref.size() == static_cast<std::size_t>(n_nodes),
                "multiblock differential reference volume node size mismatch");
  std::vector<double> vol(static_cast<std::size_t>(n_cells), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    const auto nodes =
        tenryu::mesh::mesh_topo_cell_corner_nodes(topo, c, cfg.mesh);
    const int active_nverts = reference_cell_active_nverts(state, c);
    double rr[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    double zz[tenryu::mesh::kMeshTopoCellStorageSlots]{};
    for (int k = 0; k < active_nverts; ++k) {
      const int n = nodes[static_cast<std::size_t>(k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "multiblock differential reference volume node out of range");
      rr[k] = r_ref[static_cast<std::size_t>(n)];
      zz[k] = z_ref[static_cast<std::size_t>(n)];
    }
    const double v_signed =
        active_nverts == 3
            ? tenryu::hydro::rz::rz_polygon_volume_exact(rr, zz, 3)
            : tenryu::mesh::rz_quad_volume_exact(
                  rr[0], zz[0], rr[1], zz[1],
                  rr[2], zz[2], rr[3], zz[3]);
    const double v = multiblock_cell_orientation_sign(mb, c) * v_signed;
    if (state.mesh.is_geometry_exempt_cell(c)) {
      // Virtual pseudo-core member geometry may legitimately fold; it has no
      // physical volume role in the reference machinery.
      vol[static_cast<std::size_t>(c)] =
          std::isfinite(v) ? std::max(v, 0.0) : 0.0;
      continue;
    }
    TENRYU_ASSERT(std::isfinite(v) && v > 0.0,
                  "multiblock differential reference produced invalid cell volume");
    vol[static_cast<std::size_t>(c)] = v;
  }
  return vol;
}

inline std::vector<double> multiblock_band_volume_ratios(
    const tenryu::core::State& state,
    const MultiblockDifferentialBandScaleResult& bands,
    const std::vector<double>& vol_ref,
    const std::vector<double>& vol_lag) {
  std::vector<double> xi_cell;
  state.ref_xi_cell.copy_to_host(xi_cell);
  std::vector<double> ref_sum(static_cast<std::size_t>(bands.band_count), 0.0);
  std::vector<double> lag_sum(static_cast<std::size_t>(bands.band_count), 0.0);
  const int n_cells = state.mesh.topo.n_cells;
  for (int c = 0; c < n_cells; ++c) {
    const int band = reference_xi_band(
        xi_cell[static_cast<std::size_t>(c)], bands.band_count);
    const auto idx = static_cast<std::size_t>(band);
    ref_sum[idx] += vol_ref[static_cast<std::size_t>(c)];
    lag_sum[idx] += vol_lag[static_cast<std::size_t>(c)];
  }
  std::vector<double> ratio(static_cast<std::size_t>(bands.band_count), 0.0);
  for (int i = 0; i < bands.band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    ratio[idx] = (lag_sum[idx] > 0.0 && std::isfinite(lag_sum[idx]))
                     ? ref_sum[idx] / lag_sum[idx]
                     : std::numeric_limits<double>::quiet_NaN();
  }
  return ratio;
}

inline void log_multiblock_differential_reference_install(
    const tenryu::core::State& state,
    const MultiblockDifferentialBandScaleResult& bands,
    const tenryu::mesh::CandidateMeshQuality& quality,
    const double sigma_accepted,
    const int linesearch_iters,
    const int cap_hit_count,
    const int binding_cell_count,
    const int binding_first_bad_cell,
    const int binding_first_bad_stable_cell,
    const int n_cells,
    const std::vector<double>& band_volume_ratio) {
  std::ostringstream summary;
  summary << "[multiblock_diffref_install] step=" << state.step
          << " sigma_accepted=" << sigma_accepted
          << " linesearch_iters=" << linesearch_iters
          << " cap_hit_count=" << cap_hit_count
          << " binding_cell_count=" << binding_cell_count
          << " binding_cell_fraction="
          << (n_cells > 0
                  ? static_cast<double>(binding_cell_count) /
                        static_cast<double>(n_cells)
                  : 0.0)
          << " binding_first_bad_cell=" << binding_first_bad_cell
          << " binding_first_bad_stable_cell=" << binding_first_bad_stable_cell
          << " min_rz_volume_rel=" << quality.min_rz_volume_rel
          << " min_corner_j_rel=" << quality.min_corner_j_rel
          << " min_gauss_j_rel=" << quality.min_gauss_j_rel;
  tenryu::core::log_info(summary.str());

  for (int i = 0; i < bands.band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    std::ostringstream line;
    line << "[multiblock_diffref_install_band] step=" << state.step
         << " band=" << i
         << " d_i=" << bands.band_displacement[idx]
         << " V_ref_over_V_lag=" << band_volume_ratio[idx];
    tenryu::core::log_info(line.str());
  }
}

}  // namespace tracking_reference_detail

inline bool multiblock_reference_xi_built(
    const tenryu::core::State& state) {
  const std::size_t n_nodes =
      state.mesh.topo.n_nodes > 0
          ? static_cast<std::size_t>(state.mesh.topo.n_nodes)
          : state.x_r_initial.size();
  const std::size_t n_cells =
      state.mesh.topo.n_cells > 0
          ? static_cast<std::size_t>(state.mesh.topo.n_cells)
          : state.vol.size();
  return n_nodes > 0U && n_cells > 0U &&
         state.ref_xi_node.size() == n_nodes &&
         state.ref_xi_cell.size() == n_cells &&
         state.ref_dir0_r.size() == n_nodes &&
         state.ref_dir0_z.size() == n_nodes;
}

inline void build_multiblock_reference_xi_initial(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  if (multiblock_reference_xi_built(state)) {
    return;
  }

  TENRYU_ASSERT(cfg.main.dim == 2 && state.mesh.dim == 2,
                "multiblock reference xi requires 2D mesh state");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "multiblock reference xi requires 2D_RZ geometry");
  TENRYU_ASSERT(tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh),
                "multiblock reference xi requires multiblock config");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "multiblock reference xi requires multiblock topology");

  const auto& topo = state.mesh.topo;
  const int n_nodes = topo.n_nodes;
  const int n_cells = topo.n_cells;
  TENRYU_ASSERT(n_nodes > 0 && n_cells > 0,
                "multiblock reference xi requires non-empty topology");
  TENRYU_ASSERT(state.x_r_initial.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_initial.size() == static_cast<std::size_t>(n_nodes),
                "multiblock reference xi requires initial node storage");
  TENRYU_ASSERT(topo.node_flags.size() == static_cast<std::size_t>(n_nodes),
                "multiblock reference xi requires node flags");
  TENRYU_ASSERT(state.mesh.cell_nverts.empty() ||
                    state.mesh.cell_nverts.size() ==
                        static_cast<std::size_t>(n_cells),
                "multiblock reference xi requires cell_nverts size match");

  const double xi_tol =
      cfg.numerics.ale.multiblock_differential_reference_xi_seam_tol;
  TENRYU_ASSERT(std::isfinite(xi_tol) && xi_tol >= 0.0,
                "multiblock reference xi seam tolerance must be finite >= 0");

  std::vector<double> r0;
  std::vector<double> z0;
  state.x_r_initial.copy_to_host(r0);
  state.x_z_initial.copy_to_host(z0);

  std::vector<double> s0(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> xi_node(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> dir0_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> dir0_z(static_cast<std::size_t>(n_nodes), 0.0);

  double s_max = 0.0;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    TENRYU_ASSERT(std::isfinite(r0[idx]) && std::isfinite(z0[idx]),
                  "multiblock reference xi initial coordinates must be finite");
    s0[idx] = std::hypot(r0[idx], z0[idx]);
    TENRYU_ASSERT(std::isfinite(s0[idx]),
                  "multiblock reference xi initial radius must be finite");
    s_max = std::max(s_max, s0[idx]);
  }
  TENRYU_ASSERT(s_max > 0.0 && std::isfinite(s_max),
                "multiblock reference xi requires positive initial radius span");

  const double inv_s_max = 1.0 / s_max;
  const double dir_eps =
      64.0 * std::numeric_limits<double>::epsilon() * s_max;
  int outer_node_count = 0;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    xi_node[idx] = s0[idx] * inv_s_max;
    TENRYU_ASSERT(std::isfinite(xi_node[idx]) &&
                      xi_node[idx] >= 0.0 && xi_node[idx] <= 1.0,
                  "multiblock reference xi node value out of [0,1]");

    const std::uint8_t flags = topo.node_flags[idx];
    if (s0[idx] > dir_eps) {
      dir0_r[idx] = r0[idx] / s0[idx];
      dir0_z[idx] = z0[idx] / s0[idx];
    } else if ((flags & tenryu::mesh::NODE_AXIS) != 0U && z0[idx] != 0.0) {
      dir0_r[idx] = 0.0;
      dir0_z[idx] = (z0[idx] > 0.0) ? 1.0 : -1.0;
    } else {
      dir0_r[idx] = 0.0;
      dir0_z[idx] = 0.0;
    }
    const double dir_norm = std::hypot(dir0_r[idx], dir0_z[idx]);
    TENRYU_ASSERT(dir_norm == 0.0 ||
                      std::abs(dir_norm - 1.0) <=
                          64.0 * std::numeric_limits<double>::epsilon(),
                  "multiblock reference xi director norm is invalid");
    if ((flags & tenryu::mesh::NODE_CENTER) != 0U) {
      TENRYU_ASSERT(xi_node[idx] <= xi_tol,
                    "multiblock reference xi center node must be zero");
    }
    if ((flags & tenryu::mesh::NODE_OUTER_PHYSICAL_BOUNDARY) != 0U) {
      ++outer_node_count;
      TENRYU_ASSERT(std::abs(xi_node[idx] - 1.0) <= xi_tol,
                    "multiblock reference xi outer node must be one");
    }
  }
  TENRYU_ASSERT(outer_node_count > 0,
                "multiblock reference xi requires outer boundary nodes");

  tracking_reference_detail::validate_multiblock_reference_xi_monotone(
      s0, xi_node, xi_tol);
  tracking_reference_detail::validate_multiblock_reference_xi_seams(
      state, r0, z0, xi_node, s_max, xi_tol);

  std::vector<double> xi_cell(static_cast<std::size_t>(n_cells), 0.0);
  const auto& mb = *topo.multiblock;
  for (int c = 0; c < n_cells; ++c) {
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int active_nverts =
        tracking_reference_detail::reference_cell_active_nverts(state, c);
    TENRYU_ASSERT(active_nverts > 0,
                  "multiblock reference xi cell must have active nodes");
    double sum = 0.0;
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      TENRYU_ASSERT(n >= 0 && n < n_nodes,
                    "multiblock reference xi cell node out of range");
      sum += xi_node[static_cast<std::size_t>(n)];
    }
    xi_cell[static_cast<std::size_t>(c)] =
        sum / static_cast<double>(active_nverts);
  }

  state.ref_xi_node.reset(static_cast<std::size_t>(n_nodes));
  state.ref_xi_cell.reset(static_cast<std::size_t>(n_cells));
  state.ref_dir0_r.reset(static_cast<std::size_t>(n_nodes));
  state.ref_dir0_z.reset(static_cast<std::size_t>(n_nodes));
  state.ref_xi_node.copy_from_host(xi_node);
  state.ref_xi_cell.copy_from_host(xi_cell);
  state.ref_dir0_r.copy_from_host(dir0_r);
  state.ref_dir0_z.copy_from_host(dir0_z);
}

inline MultiblockDifferentialBandScaleResult
build_multiblock_differential_band_scales(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  MultiblockDifferentialBandScaleResult result;
  if (!cfg.numerics.ale.multiblock_differential_reference_enabled) {
    return result;
  }
  result.applicable = true;

  TENRYU_ASSERT(cfg.main.dim == 2 && state.mesh.dim == 2,
                "multiblock differential reference requires 2D mesh state");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "multiblock differential reference requires 2D_RZ geometry");
  TENRYU_ASSERT(tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh),
                "multiblock differential reference requires multiblock config");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "multiblock differential reference requires multiblock topology");

  if (!multiblock_reference_xi_built(state)) {
    build_multiblock_reference_xi_initial(state, cfg);
  }

  const auto& topo = state.mesh.topo;
  const int n_nodes = topo.n_nodes;
  TENRYU_ASSERT(n_nodes > 0,
                "multiblock differential reference requires non-empty topology");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes),
                "multiblock differential reference requires current node storage");
  TENRYU_ASSERT(state.x_r_initial.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z_initial.size() == static_cast<std::size_t>(n_nodes),
                "multiblock differential reference requires initial node storage");

  const auto& ale = cfg.numerics.ale;
  const int band_count = ale.multiblock_differential_reference_band_count;
  TENRYU_ASSERT(band_count > 1,
                "multiblock differential reference requires at least two bands");
  result.band_count = band_count;
  result.band_sample_count.assign(static_cast<std::size_t>(band_count), 0);
  result.initial_band_scale.assign(static_cast<std::size_t>(band_count), 0.0);
  result.raw_band_scale.assign(static_cast<std::size_t>(band_count), 0.0);
  result.corrected_band_scale.assign(static_cast<std::size_t>(band_count), 0.0);
  result.band_displacement.assign(static_cast<std::size_t>(band_count), 0.0);
  result.raw_band_spread.assign(static_cast<std::size_t>(band_count), 0.0);
  result.limiter_excess_band.assign(static_cast<std::size_t>(band_count), 0U);
  result.monotone_clamped_band.assign(static_cast<std::size_t>(band_count), 0U);

  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> r0;
  std::vector<double> z0;
  std::vector<double> xi_node;
  std::vector<double> dir0_r;
  std::vector<double> dir0_z;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);
  state.x_r_initial.copy_to_host(r0);
  state.x_z_initial.copy_to_host(z0);
  state.ref_xi_node.copy_to_host(xi_node);
  state.ref_dir0_r.copy_to_host(dir0_r);
  state.ref_dir0_z.copy_to_host(dir0_z);

  std::vector<std::vector<double>> raw_samples(
      static_cast<std::size_t>(band_count));
  std::vector<std::vector<double>> initial_samples(
      static_cast<std::size_t>(band_count));
  std::vector<double> raw_min(static_cast<std::size_t>(band_count),
                              std::numeric_limits<double>::infinity());
  std::vector<double> raw_max(static_cast<std::size_t>(band_count),
                              -std::numeric_limits<double>::infinity());

  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    TENRYU_ASSERT(std::isfinite(r[idx]) && std::isfinite(z[idx]) &&
                      std::isfinite(r0[idx]) && std::isfinite(z0[idx]),
                  "multiblock differential reference node coordinates must be finite");
    TENRYU_ASSERT(std::isfinite(dir0_r[idx]) && std::isfinite(dir0_z[idx]),
                  "multiblock differential reference directors must be finite");
    const int band =
        tracking_reference_detail::reference_xi_band(xi_node[idx], band_count);
    const auto b = static_cast<std::size_t>(band);
    const double p = r[idx] * dir0_r[idx] + z[idx] * dir0_z[idx];
    const double p0 = r0[idx] * dir0_r[idx] + z0[idx] * dir0_z[idx];
    const double s0 = std::hypot(r0[idx], z0[idx]);
    TENRYU_ASSERT(std::isfinite(p) && std::isfinite(p0),
                  "multiblock differential reference projection must be finite");
    TENRYU_ASSERT(std::isfinite(s0),
                  "multiblock differential reference initial radius must be finite");
    raw_samples[b].push_back(p);
    initial_samples[b].push_back(p0);
    raw_min[b] = std::min(raw_min[b], p);
    raw_max[b] = std::max(raw_max[b], p);
    ++result.band_sample_count[b];
    result.s_max_initial = std::max(result.s_max_initial, s0);
  }
  TENRYU_ASSERT(result.s_max_initial > 0.0 && std::isfinite(result.s_max_initial),
                "multiblock differential reference requires positive initial span");

  for (int i = 0; i < band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    if (result.band_sample_count[idx] > 0) {
      result.raw_band_scale[idx] =
          tracking_reference_detail::median_in_place(raw_samples[idx]);
      result.initial_band_scale[idx] =
          tracking_reference_detail::median_in_place(initial_samples[idx]);
      result.raw_band_spread[idx] = raw_max[idx] - raw_min[idx];
    }
  }
  if (!tracking_reference_detail::fill_missing_band_values(
          result.raw_band_scale, result.band_sample_count) ||
      !tracking_reference_detail::fill_missing_band_values(
          result.initial_band_scale, result.band_sample_count)) {
    return result;
  }

  const double scale =
      std::max(std::abs(result.s_max_initial), 1.0e-300);
  const double tiny_floor =
      64.0 * std::numeric_limits<double>::epsilon() * scale;
  const double first_floor =
      ale.multiblock_differential_reference_s_cap_min_rel *
      result.s_max_initial;
  std::vector<double> edge_floor(static_cast<std::size_t>(band_count),
                                 tiny_floor);
  for (int i = 1; i < band_count; ++i) {
    const double h = tracking_reference_detail::edge_band_spacing(
        result.raw_band_scale, result.initial_band_scale, i);
    const double rel_floor =
        ale.multiblock_differential_reference_eps_v * h;
    edge_floor[static_cast<std::size_t>(i)] =
        std::max(tiny_floor, std::isfinite(rel_floor) ? rel_floor : 0.0);
  }

  if (tracking_reference_detail::band_scale_raw_admissible(
          result.raw_band_scale, edge_floor, first_floor)) {
    result.corrected_band_scale = result.raw_band_scale;
    result.built = true;
    if (tracking_reference_detail::multiblock_differential_reference_diag_enabled()) {
      tracking_reference_detail::log_multiblock_differential_band_scales(
          state, result);
    }
    return result;
  }

  std::vector<double> alpha(static_cast<std::size_t>(band_count), 0.0);
  for (int i = 0; i < band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const double denom = result.initial_band_scale[idx];
    if (std::abs(denom) > tiny_floor) {
      alpha[idx] = result.raw_band_scale[idx] / denom;
    } else {
      alpha[idx] = std::abs(result.raw_band_scale[idx]) <= tiny_floor
                       ? 0.0
                       : result.raw_band_scale[idx] / tiny_floor;
    }
  }

  std::vector<double> edge_weight(
      static_cast<std::size_t>(band_count - 1), 1.0);
  const double g0 = ale.multiblock_differential_reference_smoothing_g0;
  for (int i = 0; i + 1 < band_count; ++i) {
    const double g = std::abs(alpha[static_cast<std::size_t>(i + 1)] -
                              alpha[static_cast<std::size_t>(i)]);
    const double x = g / g0;
    edge_weight[static_cast<std::size_t>(i)] = 1.0 / (1.0 + x * x);
  }

  std::vector<double> fit = result.raw_band_scale;
  std::vector<double> tmp = fit;
  constexpr int kSmoothPasses = 8;
  for (int pass = 0; pass < kSmoothPasses; ++pass) {
    tmp = fit;
    for (int i = 1; i + 1 < band_count; ++i) {
      const double wl = edge_weight[static_cast<std::size_t>(i - 1)];
      const double wr = edge_weight[static_cast<std::size_t>(i)];
      tmp[static_cast<std::size_t>(i)] =
          fit[static_cast<std::size_t>(i)] +
          0.25 * (wl * (fit[static_cast<std::size_t>(i - 1)] -
                        fit[static_cast<std::size_t>(i)]) +
                  wr * (fit[static_cast<std::size_t>(i + 1)] -
                        fit[static_cast<std::size_t>(i)]));
    }
    tmp.front() = result.raw_band_scale.front();
    tmp.back() = result.raw_band_scale.back();
    fit.swap(tmp);
  }

  std::vector<double> s = result.raw_band_scale;
  std::vector<double> cap(static_cast<std::size_t>(band_count), 0.0);
  const double nu = ale.multiblock_differential_reference_nu;
  for (int i = 0; i < band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const double h = tracking_reference_detail::local_band_spacing(
        result.raw_band_scale, result.initial_band_scale, i);
    cap[idx] = nu * h;
    if (i == band_count - 1) {
      cap[idx] = 0.0;
      s[idx] = result.raw_band_scale[idx];
      continue;
    }
    const double raw_d = fit[idx] - result.raw_band_scale[idx];
    if (std::abs(raw_d) > cap[idx]) {
      result.limiter_excess_band[idx] = 1U;
      ++result.limiter_excess_count;
      result.max_limiter_excess =
          std::max(result.max_limiter_excess, std::abs(raw_d) - cap[idx]);
    }
    const double d = std::clamp(raw_d, -cap[idx], cap[idx]);
    s[idx] = result.raw_band_scale[idx] + d;
    result.max_limited_delta =
        std::max(result.max_limited_delta, std::abs(d));
  }
  s.back() = result.raw_band_scale.back();

  if (!tracking_reference_detail::monotone_project_band_scale(
          s,
          result.raw_band_scale,
          cap,
          edge_floor,
          first_floor,
          result.max_limited_delta,
          result.max_limiter_excess,
          result.monotone_clamp_count,
          result.monotone_clamped_band)) {
    result.corrected_band_scale = s;
    for (int i = 0; i < band_count; ++i) {
      const auto idx = static_cast<std::size_t>(i);
      result.band_displacement[idx] =
          result.corrected_band_scale[idx] - result.raw_band_scale[idx];
    }
    result.built = true;
    if (tracking_reference_detail::multiblock_differential_reference_diag_enabled()) {
      tracking_reference_detail::log_multiblock_differential_band_scales(
          state, result);
    }
    return result;
  }

  result.corrected_band_scale = s;
  for (int i = 0; i < band_count; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    result.band_displacement[idx] =
        result.corrected_band_scale[idx] - result.raw_band_scale[idx];
  }
  result.built = true;
  if (tracking_reference_detail::multiblock_differential_reference_diag_enabled()) {
    tracking_reference_detail::log_multiblock_differential_band_scales(
        state, result);
  }
  return result;
}

inline bool install_multiblock_differential_reference(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double* sigma_accepted_out = nullptr) {
  if (sigma_accepted_out != nullptr) {
    *sigma_accepted_out = 0.0;
  }
  if (!multiblock_reference_xi_built(state)) {
    build_multiblock_reference_xi_initial(state, cfg);
  }

  MultiblockDifferentialBandScaleResult bands =
      build_multiblock_differential_band_scales(state, cfg);
  if (!bands.applicable) {
    return false;
  }
  TENRYU_ASSERT(bands.built,
                "multiblock differential reference band scales did not build");
  TENRYU_ASSERT(bands.band_count > 0 &&
                    bands.band_displacement.size() ==
                        static_cast<std::size_t>(bands.band_count),
                "multiblock differential reference band displacement mismatch");

  TENRYU_ASSERT(cfg.main.dim == 2 && state.mesh.dim == 2,
                "multiblock differential reference install requires 2D mesh state");
  TENRYU_ASSERT(cfg.main.dimension == "2D_RZ",
                "multiblock differential reference install requires 2D_RZ geometry");
  TENRYU_ASSERT(tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh),
                "multiblock differential reference install requires multiblock config");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "multiblock differential reference install requires multiblock topology");

  const auto& topo = state.mesh.topo;
  const auto& mb = *topo.multiblock;
  const int n_nodes = topo.n_nodes;
  const int n_cells = topo.n_cells;
  TENRYU_ASSERT(n_nodes > 0 && n_cells > 0,
                "multiblock differential reference install requires non-empty topology");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes),
                "multiblock differential reference install requires current nodes");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "multiblock differential reference install requires reference nodes");
  TENRYU_ASSERT(state.cell_vol_initial.size() == state.vol.size() &&
                    state.cell_vol_initial.size() ==
                        static_cast<std::size_t>(n_cells),
                "multiblock differential reference install requires reference volumes");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "multiblock differential reference requires device cell-node CSR offsets");
  TENRYU_ASSERT(state.mesh.multiblock_cell_node_csr_indices.size() ==
                    static_cast<std::size_t>(n_cells) *
                        tenryu::mesh::kMeshTopoCellStorageSlots,
                "multiblock differential reference requires device cell-node CSR indices");
  TENRYU_ASSERT(mb.cell_id_stable.size() == static_cast<std::size_t>(n_cells),
                "multiblock differential reference requires stable cell ids");
  TENRYU_ASSERT(mb.cell_orientation_sign.size() ==
                    static_cast<std::size_t>(n_cells),
                "multiblock differential reference requires orientation signs");

  std::vector<double> r_lag;
  std::vector<double> z_lag;
  std::vector<double> xi_node;
  std::vector<double> dir0_r;
  std::vector<double> dir0_z;
  state.x_r.copy_to_host(r_lag);
  state.x_z.copy_to_host(z_lag);
  state.ref_xi_node.copy_to_host(xi_node);
  state.ref_dir0_r.copy_to_host(dir0_r);
  state.ref_dir0_z.copy_to_host(dir0_z);
  TENRYU_ASSERT(xi_node.size() == static_cast<std::size_t>(n_nodes) &&
                    dir0_r.size() == static_cast<std::size_t>(n_nodes) &&
                    dir0_z.size() == static_cast<std::size_t>(n_nodes),
                "multiblock differential reference xi/director size mismatch");

  const std::vector<double> node_h =
      tracking_reference_detail::multiblock_node_spacing(
          state, cfg, r_lag, z_lag);
  std::vector<double> r_capped = r_lag;
  std::vector<double> z_capped = z_lag;
  std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
  const double nu = cfg.numerics.ale.multiblock_differential_reference_nu;
  const double eps =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(std::abs(bands.s_max_initial), 1.0e-300);
  int cap_hit_count = 0;
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    TENRYU_ASSERT(std::isfinite(r_lag[idx]) && std::isfinite(z_lag[idx]) &&
                      std::isfinite(dir0_r[idx]) && std::isfinite(dir0_z[idx]),
                  "multiblock differential reference target node is non-finite");
    const double d =
        tracking_reference_detail::interpolate_band_displacement(
            bands.band_displacement, xi_node[idx]);
    const double dr = d * dir0_r[idx];
    const double dz = d * dir0_z[idx];
    const double dist = std::hypot(dr, dz);
    double chi = 1.0;
    if (dist > 0.0 && std::isfinite(dist)) {
      chi = std::min(1.0, (nu * node_h[idx]) / std::max(dist, eps));
    }
    if (!std::isfinite(chi) || chi < 0.0) {
      chi = 0.0;
    }
    if (chi < 1.0) {
      ++cap_hit_count;
    }
    r_capped[idx] = r_lag[idx] + chi * dr;
    z_capped[idx] = z_lag[idx] + chi * dz;
    delta_r[idx] = r_capped[idx] - r_lag[idx];
    delta_z[idx] = z_capped[idx] - z_lag[idx];
  }

  tenryu::core::DeviceArray<double> d_delta_r(
      static_cast<std::size_t>(n_nodes));
  tenryu::core::DeviceArray<double> d_delta_z(
      static_cast<std::size_t>(n_nodes));
  tenryu::core::DeviceArray<int> d_cell_id_stable(mb.cell_id_stable.size());
  tenryu::core::DeviceArray<int> d_cell_orientation_sign(
      mb.cell_orientation_sign.size());
  tenryu::core::DeviceArray<std::uint8_t> d_cell_nverts;
  d_delta_r.copy_from_host(delta_r);
  d_delta_z.copy_from_host(delta_z);
  d_cell_id_stable.copy_from_host(mb.cell_id_stable);
  d_cell_orientation_sign.copy_from_host(mb.cell_orientation_sign);
  const std::uint8_t* d_cell_nverts_ptr = nullptr;
  if (tracking_reference_detail::multiblock_has_tri_cell_nverts(state)) {
    d_cell_nverts.reset(state.mesh.cell_nverts.size());
    d_cell_nverts.copy_from_host(state.mesh.cell_nverts);
    d_cell_nverts_ptr = d_cell_nverts.data();
  }

  tenryu::mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;

  tenryu::mesh::CandidateMeshQuality unit_quality =
      tenryu::mesh::evaluate_candidate_mesh_quality_csr(
          state.x_r.data(),
          state.x_z.data(),
          d_delta_r.data(),
          d_delta_z.data(),
          1.0,
          n_cells,
          state.mesh.corner_stride,
          state.mesh.multiblock_cell_node_csr_offsets.data(),
          state.mesh.multiblock_cell_node_csr_indices.data(),
          d_cell_id_stable.data(),
          d_cell_orientation_sign.data(),
          floors,
          d_cell_nverts_ptr,
          nullptr,
          nullptr,
          0,
          state.x_r_reference.size() == state.x_r.size() &&
                  state.x_z_reference.size() == state.x_z.size()
              ? state.x_r_reference.data()
              : nullptr,
          state.x_r_reference.size() == state.x_r.size() &&
                  state.x_z_reference.size() == state.x_z.size()
              ? state.x_z_reference.data()
              : nullptr);
  tenryu::mesh::CandidateMeshQuality final_quality = unit_quality;
  double sigma_accepted = 1.0;
  int linesearch_iters = 0;
  if (!unit_quality.admissible()) {
    const tenryu::mesh::LineSearchResult ls =
        tenryu::mesh::linesearch_largest_admissible_sigma_csr(
            state.x_r.data(),
            state.x_z.data(),
            d_delta_r.data(),
            d_delta_z.data(),
            1.0,
            0.0,
            cfg.numerics.ale.reference_linesearch_max_iters,
            n_cells,
            state.mesh.corner_stride,
            state.mesh.multiblock_cell_node_csr_offsets.data(),
            state.mesh.multiblock_cell_node_csr_indices.data(),
            d_cell_id_stable.data(),
            d_cell_orientation_sign.data(),
            floors,
            d_cell_nverts_ptr,
            nullptr,
            nullptr,
            0,
            state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size()
                ? state.x_r_reference.data()
                : nullptr,
            state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size()
                ? state.x_z_reference.data()
                : nullptr);
    sigma_accepted = ls.sigma_accepted;
    linesearch_iters = ls.iters_used;
    final_quality = ls.quality;
  }
  if (sigma_accepted == 0.0 && !final_quality.admissible()) {
    final_quality = tenryu::mesh::evaluate_candidate_mesh_quality_csr(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r.data(),
        d_delta_z.data(),
        0.0,
        n_cells,
        state.mesh.corner_stride,
        state.mesh.multiblock_cell_node_csr_offsets.data(),
        state.mesh.multiblock_cell_node_csr_indices.data(),
        d_cell_id_stable.data(),
        d_cell_orientation_sign.data(),
        floors,
        d_cell_nverts_ptr,
        nullptr,
        nullptr,
        0,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    TENRYU_ASSERT(final_quality.admissible(),
                  "multiblock differential reference current mesh is inadmissible");
  }

  std::vector<double> r_ref = r_lag;
  std::vector<double> z_ref = z_lag;
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    r_ref[idx] = r_lag[idx] + sigma_accepted * delta_r[idx];
    z_ref[idx] = z_lag[idx] + sigma_accepted * delta_z[idx];
  }
  if (sigma_accepted_out != nullptr) {
    *sigma_accepted_out = sigma_accepted;
  }

  const std::vector<double> vol_ref =
      tracking_reference_detail::compute_multiblock_reference_volumes(
          state, cfg, r_ref, z_ref);
  if (tracking_reference_detail::multiblock_differential_reference_diag_enabled()) {
    const std::vector<double> vol_lag =
        tracking_reference_detail::compute_multiblock_reference_volumes(
            state, cfg, r_lag, z_lag);
    const std::vector<double> band_volume_ratio =
        tracking_reference_detail::multiblock_band_volume_ratios(
            state, bands, vol_ref, vol_lag);
    const int binding_cell_count =
        unit_quality.admissible() || unit_quality.first_bad_cell < 0 ? 0 : 1;
    tracking_reference_detail::log_multiblock_differential_reference_install(
        state,
        bands,
        final_quality,
        sigma_accepted,
        linesearch_iters,
        cap_hit_count,
        binding_cell_count,
        unit_quality.first_bad_cell,
        unit_quality.first_bad_stable_cell,
        n_cells,
        band_volume_ratio);
  }

  state.x_r_reference.copy_from_host(r_ref);
  state.x_z_reference.copy_from_host(z_ref);
  state.cell_vol_initial.copy_from_host(vol_ref);
  return true;
}

inline bool prepare_multiblock_differential_reference_if_enabled(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    double* sigma_accepted_out = nullptr) {
  if (sigma_accepted_out != nullptr) {
    *sigma_accepted_out = 0.0;
  }
  if (!cfg.numerics.ale.multiblock_differential_reference_enabled ||
      !tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return false;
  }
  return install_multiblock_differential_reference(state, cfg,
                                                  sigma_accepted_out);
}

inline bool tri_fan_tracking_reference_applicable(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  return cfg.numerics.ale.tri_fan_tracking_reference_enabled &&
         cfg.mesh.polar_center_treatment == "tri_fan" &&
         !tenryu::mesh::mesh_topo_is_multiblock(cfg.mesh) &&
         tenryu::mesh::has_tri_fan_center_topology(state.mesh);
}

inline TriFanTrackingReferenceResult install_tri_fan_tracking_reference_if_enabled(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  TriFanTrackingReferenceResult result;
  if (!tri_fan_tracking_reference_applicable(state, cfg)) {
    return result;
  }
  result.applicable = true;

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_nodes > 0 && n_cells > 0,
                "tri_fan tracking reference requires non-empty 2D topology");
  TENRYU_ASSERT(state.mesh.topo.node_flags.size() ==
                    static_cast<std::size_t>(n_nodes),
                "tri_fan tracking reference requires node flags");
  TENRYU_ASSERT(state.mesh.cell_nverts.size() ==
                    static_cast<std::size_t>(n_cells),
                "tri_fan tracking reference requires cell_nverts");
  TENRYU_ASSERT(state.x_r_initial.size() == state.x_r.size() &&
                    state.x_z_initial.size() == state.x_z.size(),
                "tri_fan tracking reference requires initial node storage");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "tri_fan tracking reference requires reference node storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == state.vol.size(),
                "tri_fan tracking reference requires reference volumes");

  std::vector<double> r_lag;
  std::vector<double> z_lag;
  std::vector<double> r0;
  std::vector<double> z0;
  std::vector<double> r_prev;
  std::vector<double> z_prev;
  state.x_r.copy_to_host(r_lag);
  state.x_z.copy_to_host(z_lag);
  state.x_r_initial.copy_to_host(r0);
  state.x_z_initial.copy_to_host(z0);
  state.x_r_reference.copy_to_host(r_prev);
  state.x_z_reference.copy_to_host(z_prev);

  const auto& flags = state.mesh.topo.node_flags;
  result.raw_ring_scale.assign(static_cast<std::size_t>(nr + 1), 0.0);
  result.filtered_ring_scale.assign(static_cast<std::size_t>(nr + 1), 0.0);

  std::vector<double> r_target(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> z_target(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> r_limited = r_lag;
  std::vector<double> z_limited = z_lag;
  const bool seamless_mode =
      cfg.numerics.ale.tri_fan_tracking_reference_mode == "seamless_converging";

  if (cfg.numerics.ale.tri_fan_tracking_reference_mode == "legacy_lagging") {
    const double omega = cfg.numerics.ale.tri_fan_tracking_reference_omega;
    for (int i = 1; i <= nr; ++i) {
      const double raw =
          tracking_reference_detail::ring_median_radius(r_lag, z_lag, flags, i, nz, true);
      const double prev =
          tracking_reference_detail::ring_median_radius(r_prev, z_prev, flags, i, nz, true);
      const double initial =
          tracking_reference_detail::ring_median_radius(r0, z0, flags, i, nz, true);
      result.raw_ring_scale[static_cast<std::size_t>(i)] =
          tracking_reference_detail::finite_nonnegative(raw) ? raw : initial;

      double filtered = result.raw_ring_scale[static_cast<std::size_t>(i)];
      const bool have_prev = tracking_reference_detail::finite_nonnegative(prev);
      const bool prev_is_initial =
          have_prev && tracking_reference_detail::nearly_equal_scale(prev, initial);
      if (have_prev && !prev_is_initial &&
          tracking_reference_detail::finite_nonnegative(
              result.raw_ring_scale[static_cast<std::size_t>(i)])) {
        filtered = (1.0 - omega) * prev +
                   omega * result.raw_ring_scale[static_cast<std::size_t>(i)];
      }
      if (!tracking_reference_detail::finite_nonnegative(filtered)) {
        filtered =
            tracking_reference_detail::finite_nonnegative(initial) ? initial : 0.0;
      }
      result.filtered_ring_scale[static_cast<std::size_t>(i)] = filtered;
    }

    result.raw_ring_scale[0] = 0.0;
    result.filtered_ring_scale[0] = 0.0;
    for (int i = 1; i <= nr; ++i) {
      double& s = result.filtered_ring_scale[static_cast<std::size_t>(i)];
      const double prev = result.filtered_ring_scale[static_cast<std::size_t>(i - 1)];
      if (!(s >= prev)) {
        s = std::nextafter(prev, std::numeric_limits<double>::infinity());
      }
    }

    tracking_reference_detail::build_radial_target_from_ring_scale(
        result.filtered_ring_scale, r0, z0, flags, nr, nz, r_target, z_target);

    const double nu = cfg.numerics.ale.tri_fan_tracking_reference_nu;
    for (int i = 0; i <= nr; ++i) {
      for (int j = 0; j <= nz; ++j) {
        const int n = tracking_reference_detail::node_index(i, j, nz);
        const auto idx = static_cast<std::size_t>(n);
        const std::uint8_t f = flags[idx];
        if ((f & tenryu::mesh::NODE_CENTER) != 0U) {
          r_limited[idx] = 0.0;
          z_limited[idx] = 0.0;
          continue;
        }
        const double dr = r_target[idx] - r_lag[idx];
        const double dz = z_target[idx] - z_lag[idx];
        const double dist = std::hypot(dr, dz);
        const double h =
            tracking_reference_detail::local_node_scale(r_lag, z_lag, nr, nz, i, j);
        double chi = 1.0;
        if (dist > 0.0 && std::isfinite(dist)) {
          chi = std::min(1.0, (nu * h) / dist);
        }
        if (!std::isfinite(chi) || chi < 0.0) {
          chi = 0.0;
        }
        r_limited[idx] = r_lag[idx] + chi * dr;
        z_limited[idx] = z_lag[idx] + chi * dz;
        const double limited_dist =
            std::hypot(r_limited[idx] - r_lag[idx], z_limited[idx] - z_lag[idx]);
        result.max_limited_delta = std::max(result.max_limited_delta, limited_dist);
        result.max_limiter_excess =
            std::max(result.max_limiter_excess, limited_dist - nu * h);
        if ((f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
          r_limited[idx] = 0.0;
        }
      }
    }
  } else {
    TENRYU_ASSERT(
        cfg.numerics.ale.tri_fan_tracking_reference_mode == "seamless_converging",
        "unknown tri_fan tracking reference mode");
    const bool built =
        tracking_reference_detail::build_seamless_converging_ring_scale(
            r_lag, z_lag, r0, z0, flags, cfg, nr, nz, result);
    if (built) {
      for (int i = 0; i <= nr; ++i) {
        const double d =
            result.filtered_ring_scale[static_cast<std::size_t>(i)] -
            result.raw_ring_scale[static_cast<std::size_t>(i)];
        for (int j = 0; j <= nz; ++j) {
          const int n = tracking_reference_detail::node_index(i, j, nz);
          const auto idx = static_cast<std::size_t>(n);
          const std::uint8_t f = flags[idx];
          if ((f & tenryu::mesh::NODE_CENTER) != 0U) {
            r_target[idx] = 0.0;
            z_target[idx] = 0.0;
            r_limited[idx] = 0.0;
            z_limited[idx] = 0.0;
            continue;
          }
          double e_r = 0.0;
          double e_z = 1.0;
          tracking_reference_detail::unit_vector_from_initial(
              r0, z0, i, j, nz, e_r, e_z);
          r_target[idx] = r_lag[idx] + d * e_r;
          z_target[idx] = z_lag[idx] + d * e_z;
          r_limited[idx] = r_target[idx];
          z_limited[idx] = z_target[idx];
          if ((f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
            r_target[idx] = 0.0;
            r_limited[idx] = 0.0;
          }
        }
      }
    } else {
      r_target = r_lag;
      z_target = z_lag;
      r_limited = r_lag;
      z_limited = z_lag;
      result.identity_follow_fallback = true;
      result.identity_follow_fallback_count = 1;
    }
  }

  std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    delta_r[idx] = r_limited[idx] - r_lag[idx];
    delta_z[idx] = z_limited[idx] - z_lag[idx];
  }

  double* d_delta_r = nullptr;
  double* d_delta_z = nullptr;
  std::uint8_t* d_cell_nverts = nullptr;
  const std::size_t node_bytes = static_cast<std::size_t>(n_nodes) * sizeof(double);
  tracking_reference_detail::cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_delta_r), node_bytes),
      "tri_fan tracking reference cudaMalloc delta_r failed");
  tracking_reference_detail::cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_delta_z), node_bytes),
      "tri_fan tracking reference cudaMalloc delta_z failed");
  tracking_reference_detail::cuda_check(
      cudaMemcpy(d_delta_r, delta_r.data(), node_bytes, cudaMemcpyHostToDevice),
      "tri_fan tracking reference delta_r H2D failed");
  tracking_reference_detail::cuda_check(
      cudaMemcpy(d_delta_z, delta_z.data(), node_bytes, cudaMemcpyHostToDevice),
      "tri_fan tracking reference delta_z H2D failed");
  tracking_reference_detail::cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&d_cell_nverts),
                 static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t)),
      "tri_fan tracking reference cudaMalloc cell_nverts failed");
  tracking_reference_detail::cuda_check(
      cudaMemcpy(d_cell_nverts,
                 state.mesh.cell_nverts.data(),
                 static_cast<std::size_t>(n_cells) * sizeof(std::uint8_t),
                 cudaMemcpyHostToDevice),
      "tri_fan tracking reference cell_nverts H2D failed");

  tenryu::mesh::CandidateMeshAdmissibilityFloors floors;
  floors.volume_rel = cfg.numerics.ale.reference_volume_floor_rel;
  floors.corner_j_rel = cfg.numerics.ale.reference_corner_j_floor_rel;
  floors.gauss_j_rel = cfg.numerics.ale.reference_gauss_j_floor_rel;
  result.final_quality = tenryu::mesh::evaluate_candidate_mesh_quality(
      state.x_r.data(),
      state.x_z.data(),
      d_delta_r,
      d_delta_z,
      1.0,
      nr,
      nz,
      floors,
      d_cell_nverts,
      state.x_r_reference.size() == state.x_r.size() &&
              state.x_z_reference.size() == state.x_z.size()
          ? state.x_r_reference.data()
          : nullptr,
      state.x_r_reference.size() == state.x_r.size() &&
              state.x_z_reference.size() == state.x_z.size()
          ? state.x_z_reference.data()
          : nullptr);
  result.sigma_accepted = 1.0;
  if (!result.final_quality.admissible()) {
    const auto ls = tenryu::mesh::linesearch_largest_admissible_sigma(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        1.0,
        0.0,
        cfg.numerics.ale.reference_linesearch_max_iters,
        nr,
        nz,
        floors,
        d_cell_nverts,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    result.sigma_accepted = ls.sigma_accepted;
    result.linesearch_iters = ls.iters_used;
    result.final_quality = ls.quality;
  }
  if (result.sigma_accepted == 0.0 && !result.final_quality.admissible()) {
    result.final_quality = tenryu::mesh::evaluate_candidate_mesh_quality(
        state.x_r.data(),
        state.x_z.data(),
        d_delta_r,
        d_delta_z,
        0.0,
        nr,
        nz,
        floors,
        d_cell_nverts,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_r_reference.data()
            : nullptr,
        state.x_r_reference.size() == state.x_r.size() &&
                state.x_z_reference.size() == state.x_z.size()
            ? state.x_z_reference.data()
            : nullptr);
    TENRYU_ASSERT(result.final_quality.admissible(),
                  "tri_fan tracking reference current mesh is inadmissible");
  }
  tracking_reference_detail::cuda_check(
      cudaFree(d_cell_nverts),
      "tri_fan tracking reference cudaFree cell_nverts failed");
  tracking_reference_detail::cuda_check(
      cudaFree(d_delta_z),
      "tri_fan tracking reference cudaFree delta_z failed");
  tracking_reference_detail::cuda_check(
      cudaFree(d_delta_r),
      "tri_fan tracking reference cudaFree delta_r failed");

  std::vector<double> r_ref = r_lag;
  std::vector<double> z_ref = z_lag;
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    r_ref[idx] = r_lag[idx] + result.sigma_accepted * delta_r[idx];
    z_ref[idx] = z_lag[idx] + result.sigma_accepted * delta_z[idx];
    const std::uint8_t f = flags[idx];
    if ((f & tenryu::mesh::NODE_CENTER) != 0U) {
      r_ref[idx] = 0.0;
      z_ref[idx] = 0.0;
    } else if ((f & tenryu::mesh::NODE_POLE_AXIS) != 0U) {
      r_ref[idx] = 0.0;
    }
  }

  const std::vector<double> vol_ref =
      tracking_reference_detail::compute_reference_volumes(
          r_ref, z_ref, state.mesh.cell_nverts, nr, nz);
  if (seamless_mode) {
    const std::vector<double> vol_lag =
        tracking_reference_detail::compute_reference_volumes(
            r_lag, z_lag, state.mesh.cell_nverts, nr, nz);
    result.outer_band_max_log_volume_ratio =
        tracking_reference_detail::outer_band_max_log_volume_ratio(
            vol_ref, vol_lag, nr, nz);
    result.angular_spread_over_h_max =
        tracking_reference_detail::angular_spread_over_h_max(
            r_lag, z_lag, r0, z0, flags, nr, nz);
    result.shock_width_proxy_rings =
        tracking_reference_detail::shock_width_proxy_rings(
            r_lag,
            z_lag,
            r0,
            z0,
            flags,
            cfg.numerics.ale.tri_fan_tracking_reference_g0,
            nr,
            nz);
    tenryu::core::log_info(
        std::string("[tri_fan_tracking_reference][seamless] step=") +
        std::to_string(state.step) +
        " outer_band_max_abs_log_Vref_over_Vlag=" +
        std::to_string(result.outer_band_max_log_volume_ratio) +
        " angular_spread_over_h_max=" +
        std::to_string(result.angular_spread_over_h_max) +
        " sigma_accepted=" + std::to_string(result.sigma_accepted) +
        " linesearch_iters=" + std::to_string(result.linesearch_iters) +
        " identity_follow_fallback=" +
        std::to_string(result.identity_follow_fallback ? 1 : 0) +
        " identity_follow_fallback_count=" +
        std::to_string(result.identity_follow_fallback_count) +
        " shock_width_proxy_rings=" +
        std::to_string(result.shock_width_proxy_rings));
  }
  state.x_r_reference.copy_from_host(r_ref);
  state.x_z_reference.copy_from_host(z_ref);
  state.cell_vol_initial.copy_from_host(vol_ref);
  result.installed = true;
  return result;
}

inline bool tri_fan_lagrangian_bulk_reference_applicable(
    const tenryu::core::State& state,
    const tenryu::core::Config& cfg) {
  if (cfg.numerics.ale.tri_fan_tracking_reference_enabled &&
      cfg.numerics.ale.tri_fan_tracking_reference_mode == "seamless_converging") {
    return false;
  }
  if (!cfg.numerics.ale.conservative_remap_lagrangian_bulk_enabled ||
      !cfg.numerics.ale.conservative_remap_enabled ||
      cfg.numerics.ale.conservative_remap_target != "reference" ||
      cfg.main.dimension != "2D_RZ" ||
      state.mesh.dim != 2 ||
      !tenryu::mesh::mesh_topo_is_single_block(cfg.mesh) ||
      cfg.mesh.logical_mesh_2d != "spherical_polar_halfplane" ||
      state.mesh.logical != tenryu::mesh::LogicalMesh2D::SphericalPolarHalfplane ||
      cfg.mesh.polar_center_treatment != "tri_fan" ||
      state.mesh.polar_center_treatment != tenryu::mesh::PolarCenterTreatment::TriFan ||
      !tenryu::mesh::has_tri_fan_center_topology(state.mesh)) {
    return false;
  }
  return tenryu::hydro::parse_boundary_2d_type(
             cfg.numerics.hydro.boundary_2d.r_outer) ==
         tenryu::hydro::Boundary2DType::PRESSURE;
}

inline TriFanTrackingReferenceResult
finalize_tri_fan_lagrangian_bulk_reference_if_enabled(
    tenryu::core::State& state,
    const tenryu::core::Config& cfg,
    const bool tracking_reference_installed) {
  TriFanTrackingReferenceResult result;
  if (!tri_fan_lagrangian_bulk_reference_applicable(state, cfg)) {
    return result;
  }
  result.applicable = true;

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_nodes > 0 && n_cells > 0,
                "tri_fan lagrangian bulk reference requires non-empty topology");
  TENRYU_ASSERT(state.mesh.cell_nverts.size() ==
                    static_cast<std::size_t>(n_cells),
                "tri_fan lagrangian bulk reference requires cell_nverts");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "tri_fan lagrangian bulk reference requires reference storage");
  TENRYU_ASSERT(state.cell_vol_initial.size() == state.vol.size(),
                "tri_fan lagrangian bulk reference requires reference volumes");

  std::vector<double> r_lag;
  std::vector<double> z_lag;
  std::vector<double> r_ref;
  std::vector<double> z_ref;
  state.x_r.copy_to_host(r_lag);
  state.x_z.copy_to_host(z_lag);
  state.x_r_reference.copy_to_host(r_ref);
  state.x_z_reference.copy_to_host(z_ref);

  const int center_node_ring_max =
      cfg.numerics.ale.conservative_remap_lagrangian_bulk_center_node_ring_max;
  if (!tracking_reference_installed) {
    r_ref = r_lag;
    z_ref = z_lag;
  } else {
    const int first_bulk_node_ring =
        center_node_ring_max >= nr ? nr + 1 : center_node_ring_max + 1;
    for (int i = first_bulk_node_ring; i <= nr; ++i) {
      for (int j = 0; j <= nz; ++j) {
        const int n = tracking_reference_detail::node_index(i, j, nz);
        const auto idx = static_cast<std::size_t>(n);
        r_ref[idx] = r_lag[idx];
        z_ref[idx] = z_lag[idx];
      }
    }
  }

  std::vector<double> delta_r(static_cast<std::size_t>(n_nodes), 0.0);
  std::vector<double> delta_z(static_cast<std::size_t>(n_nodes), 0.0);
  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    delta_r[idx] = r_ref[idx] - r_lag[idx];
    delta_z[idx] = z_ref[idx] - z_lag[idx];
  }

  result.final_quality =
      tracking_reference_detail::reference_displacement_quality(state,
                                                                cfg,
                                                                delta_r,
                                                                delta_z,
                                                                nr,
                                                                nz,
                                                                n_nodes,
                                                                n_cells,
                                                                result.sigma_accepted,
                                                                result.linesearch_iters);

  for (int n = 0; n < n_nodes; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    r_ref[idx] = r_lag[idx] + result.sigma_accepted * delta_r[idx];
    z_ref[idx] = z_lag[idx] + result.sigma_accepted * delta_z[idx];
  }

  const std::vector<double> vol_ref =
      tracking_reference_detail::compute_reference_volumes(
          r_ref, z_ref, state.mesh.cell_nverts, nr, nz);
  state.x_r_reference.copy_from_host(r_ref);
  state.x_z_reference.copy_from_host(z_ref);
  state.cell_vol_initial.copy_from_host(vol_ref);
  result.installed = true;

  constexpr double kBulkCornerJRatioWarnFloor = 1.0e-3;
  const double min_bulk_corner_j_ratio =
      tracking_reference_detail::min_bulk_corner_jacobian_ratio(
          r_ref,
          z_ref,
          state.mesh.cell_nverts,
          nr,
          nz,
          center_node_ring_max);
  if (min_bulk_corner_j_ratio < kBulkCornerJRatioWarnFloor) {
    tenryu::core::log_warning(
        std::string("[conservative_remap_lagrangian_bulk] bulk cell ci>M "
                    "approaching tangle; consider raising center_node_ring_max "
                    "(M=") +
        std::to_string(center_node_ring_max) +
        ", min_corner_j_ratio=" + std::to_string(min_bulk_corner_j_ratio) + ")");
  }
  return result;
}

}  // namespace tenryu::hydro::ale
