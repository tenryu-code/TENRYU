#include "hydro/refinement_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro {
namespace {

struct LohnerTerms {
  double numerator = 0.0;
  double denominator = 0.0;
};

struct BlockView {
  int cell_begin = 0;
  int n_i = 0;  // slow (radial/r) direction
  int n_j = 0;  // fast (angular/z) direction
};

enum class MetricMode { POLAR_ARC, COORDINATE };

LohnerTerms lohner_terms_1d(const double um2,
                            const double u0,
                            const double up2,
                            const double eps) {
  return {
      std::abs(up2 - 2.0 * u0 + um2),
      std::abs(up2 - u0) + std::abs(u0 - um2) +
          eps * (std::abs(up2) + 2.0 * std::abs(u0) + std::abs(um2)),
  };
}

// Metric factor from a signed center-to-center denominator.
// Orientation-invariant: block layer/column indexing may run either
// way on the tier meshes (inner tiers index layers inward), and every
// Lohner term is squared, so only the physical distance matters.
// Degenerate or non-finite geometry contributes nothing (factor 0).
double metric_factor(const double signed_denominator) {
  const double distance = std::abs(signed_denominator);
  if (!std::isfinite(distance) || distance <= 0.0) {
    return 0.0;
  }
  return 1.0 / distance;
}

template <typename Tag>
std::vector<double> copy_field(const core::Field1D<Tag>& field) {
  std::vector<double> out;
  field.copy_to_host(out);
  return out;
}

struct LohnerFirstDerivatives {
  explicit LohnerFirstDerivatives(const std::size_t size)
      : delu1(size, 0.0),
        delua1(size, 0.0),
        delu2d(size, 0.0),
        delua2d(size, 0.0) {}

  std::vector<double> delu1;
  std::vector<double> delua1;
  std::vector<double> delu2d;
  std::vector<double> delua2d;
};

double lohner_second_pass_cell(const LohnerFirstDerivatives& first,
                               const int cell,
                               const int n_j,
                               const double fx,
                               const double fy,
                               const bool radial_metric_valid,
                               const bool angular_metric_valid,
                               const double eps) {
  double numerator = 0.0;
  double denominator = 0.0;
  const auto accumulate_term = [&](const std::vector<double>& delu,
                                   const std::vector<double>& delua,
                                   const int minus,
                                   const int plus,
                                   const double factor) {
    const double d2 =
        (delu[static_cast<std::size_t>(plus)] -
         delu[static_cast<std::size_t>(minus)]) *
        factor;
    const double d3 =
        (std::abs(delu[static_cast<std::size_t>(plus)]) +
         std::abs(delu[static_cast<std::size_t>(minus)])) *
        factor;
    const double d4 =
        (delua[static_cast<std::size_t>(plus)] +
         delua[static_cast<std::size_t>(minus)]) *
        factor;
    numerator += d2 * d2;
    const double filtered = d3 + eps * d4;
    denominator += filtered * filtered;
  };

  if (radial_metric_valid) {
    accumulate_term(first.delu1,
                    first.delua1,
                    cell - n_j,
                    cell + n_j,
                    fx);
  }
  if (angular_metric_valid) {
    accumulate_term(
        first.delu1, first.delua1, cell - 1, cell + 1, fy);
  }
  if (radial_metric_valid) {
    accumulate_term(first.delu2d,
                    first.delua2d,
                    cell - n_j,
                    cell + n_j,
                    fx);
  }
  if (angular_metric_valid) {
    accumulate_term(
        first.delu2d, first.delua2d, cell - 1, cell + 1, fy);
  }

  if (denominator == 0.0) {
    // FLASH uses HUGE when numerator is nonzero; cap that per-cell field at 1.
    return numerator != 0.0 ? 1.0 : 0.0;
  }
  return std::sqrt(numerator / denominator);
}

void cell_centroid(const mesh::MultiBlockTopology& mb,
                   const int cell,
                   const std::vector<double>& node_r,
                   const std::vector<double>& node_z,
                   double& centroid_r,
                   double& centroid_z) {
  const int offset =
      mb.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  centroid_r = 0.0;
  centroid_z = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int node =
        mb.cell_node_csr_indices[static_cast<std::size_t>(offset + k)];
    centroid_r += node_r[static_cast<std::size_t>(node)];
    centroid_z += node_z[static_cast<std::size_t>(node)];
  }
  centroid_r *= 0.25;
  centroid_z *= 0.25;
}

void structured_cell_centroid(const mesh::MeshTopology& topo,
                              const int cell,
                              const std::vector<double>& node_r,
                              const std::vector<double>& node_z,
                              double& centroid_r,
                              double& centroid_z) {
  const int ci = cell / topo.nz;
  const int cj = cell - ci * topo.nz;
  centroid_r = 0.0;
  centroid_z = 0.0;
  for (int di = 0; di < 2; ++di) {
    for (int dj = 0; dj < 2; ++dj) {
      const int node = topo.node_index(ci + di, cj + dj);
      centroid_r += node_r[static_cast<std::size_t>(node)];
      centroid_z += node_z[static_cast<std::size_t>(node)];
    }
  }
  centroid_r *= 0.25;
  centroid_z *= 0.25;
}

void evaluate_block(const BlockView block,
                    const MetricMode metric_mode,
                    const std::vector<double>& rho,
                    const std::vector<double>& pressure,
                    const std::vector<double>& centroid_s,
                    const std::vector<double>& centroid_theta,
                    const std::vector<double>& centroid_r,
                    const std::vector<double>& centroid_z,
                    const double filter_eps,
                    const double detect_cutoff,
                    std::vector<double>& refine_error,
                    double& max_e,
                    std::size_t& n_computed,
                    std::size_t& n_above_02,
                    std::size_t& n_above_05,
                    std::size_t& n_above_08,
                    double& front_radius_weighted_cm,
                    double& front_weight) {
  const int n_i = block.n_i;
  const int n_j = block.n_j;
  const int block_cell_count = n_i * n_j;
  n_computed += static_cast<std::size_t>(block_cell_count);
  std::vector<double> block_error(
      static_cast<std::size_t>(block_cell_count), 0.0);

  const auto metric_denominators = [&](const int cell,
                                       double& radial_denominator,
                                       double& angular_denominator) {
    if (metric_mode == MetricMode::POLAR_ARC) {
      radial_denominator =
          centroid_s[static_cast<std::size_t>(cell + n_j)] -
          centroid_s[static_cast<std::size_t>(cell - n_j)];
      angular_denominator =
          centroid_s[static_cast<std::size_t>(cell)] *
          (centroid_theta[static_cast<std::size_t>(cell + 1)] -
           centroid_theta[static_cast<std::size_t>(cell - 1)]);
    } else {
      radial_denominator =
          centroid_r[static_cast<std::size_t>(cell + n_j)] -
          centroid_r[static_cast<std::size_t>(cell - n_j)];
      angular_denominator =
          centroid_z[static_cast<std::size_t>(cell + 1)] -
          centroid_z[static_cast<std::size_t>(cell - 1)];
    }
  };

  const auto accumulate_field = [&](const std::vector<double>& field) {
    LohnerFirstDerivatives first(
        static_cast<std::size_t>(block_cell_count));
    for (int layer = 1; layer + 1 < n_i; ++layer) {
      for (int column = 1; column + 1 < n_j; ++column) {
        const int local_cell = layer * n_j + column;
        const int cell = block.cell_begin + local_cell;
        double radial_denominator = 0.0;
        double angular_denominator = 0.0;
        metric_denominators(
            cell, radial_denominator, angular_denominator);
        const double fx = metric_factor(radial_denominator);
        if (fx != 0.0) {
          first.delu1[static_cast<std::size_t>(local_cell)] =
              (field[static_cast<std::size_t>(cell + n_j)] -
               field[static_cast<std::size_t>(cell - n_j)]) *
              fx;
          first.delua1[static_cast<std::size_t>(local_cell)] =
              (std::abs(field[static_cast<std::size_t>(cell + n_j)]) +
               std::abs(field[static_cast<std::size_t>(cell - n_j)])) *
              fx;
        }

        const double fy = metric_factor(angular_denominator);
        if (fy != 0.0) {
          first.delu2d[static_cast<std::size_t>(local_cell)] =
              (field[static_cast<std::size_t>(cell + 1)] -
               field[static_cast<std::size_t>(cell - 1)]) *
              fy;
          first.delua2d[static_cast<std::size_t>(local_cell)] =
              (std::abs(field[static_cast<std::size_t>(cell + 1)]) +
               std::abs(field[static_cast<std::size_t>(cell - 1)])) *
              fy;
        }
      }
    }

    for (int layer = 2; layer + 2 < n_i; ++layer) {
      for (int column = 2; column + 2 < n_j; ++column) {
        const int local_cell = layer * n_j + column;
        const int cell = block.cell_begin + local_cell;
        double radial_denominator = 0.0;
        double angular_denominator = 0.0;
        metric_denominators(
            cell, radial_denominator, angular_denominator);
        const double fx = metric_factor(radial_denominator);
        const double fy = metric_factor(angular_denominator);
        const bool radial_metric_valid = fx != 0.0;
        const bool angular_metric_valid = fy != 0.0;
        const double field_error = lohner_second_pass_cell(
            first,
            local_cell,
            n_j,
            fx,
            fy,
            radial_metric_valid,
            angular_metric_valid,
            filter_eps);
        block_error[static_cast<std::size_t>(local_cell)] =
            std::max(block_error[static_cast<std::size_t>(local_cell)],
                     field_error);
      }
    }
  };

  accumulate_field(rho);
  accumulate_field(pressure);
  for (int layer = 2; layer + 2 < n_i; ++layer) {
    for (int column = 2; column + 2 < n_j; ++column) {
      const int cell = block.cell_begin + layer * n_j + column;
      const int local_cell = layer * n_j + column;
      const double cell_e = block_error[static_cast<std::size_t>(local_cell)];
      refine_error[static_cast<std::size_t>(cell)] = cell_e;
      max_e = std::max(max_e, cell_e);
      n_above_02 += cell_e > 0.2 ? 1U : 0U;
      n_above_05 += cell_e > 0.5 ? 1U : 0U;
      n_above_08 += cell_e > 0.8 ? 1U : 0U;

      if (cell_e > detect_cutoff) {
        constexpr double kRadiansToDegrees =
            57.295779513082320876798154814105170332405472466564;
        const double theta_deg =
            centroid_theta[static_cast<std::size_t>(cell)] *
            kRadiansToDegrees;
        if (theta_deg >= 60.0 && theta_deg <= 120.0) {
          front_radius_weighted_cm +=
              cell_e * centroid_s[static_cast<std::size_t>(cell)];
          front_weight += cell_e;
        }
      }
    }
  }
}

}  // namespace

double lohner_metric_factor(const double signed_denominator) {
  return metric_factor(signed_denominator);
}

double lohner_ratio_1d(const double um2,
                       const double u0,
                       const double up2,
                       const double eps) {
  const LohnerTerms terms = lohner_terms_1d(um2, u0, up2, eps);
  return terms.denominator == 0.0 ? 0.0
                                  : terms.numerator / terms.denominator;
}

double lohner_e_2d_flat(const double u[5][5], const double eps) {
  constexpr int n_i = 5;
  constexpr int n_j = 5;
  LohnerFirstDerivatives first(n_i * n_j);
  for (int layer = 1; layer + 1 < n_i; ++layer) {
    for (int column = 1; column + 1 < n_j; ++column) {
      const int cell = layer * n_j + column;
      first.delu1[static_cast<std::size_t>(cell)] =
          u[layer + 1][column] - u[layer - 1][column];
      first.delua1[static_cast<std::size_t>(cell)] =
          std::abs(u[layer + 1][column]) +
          std::abs(u[layer - 1][column]);
      first.delu2d[static_cast<std::size_t>(cell)] =
          u[layer][column + 1] - u[layer][column - 1];
      first.delua2d[static_cast<std::size_t>(cell)] =
          std::abs(u[layer][column + 1]) +
          std::abs(u[layer][column - 1]);
    }
  }
  constexpr int center = 2 * n_j + 2;
  return lohner_second_pass_cell(
      first, center, n_j, 1.0, 1.0, true, true, eps);
}

void refinement_estimator_step(core::State& state, const core::Config& cfg) {
  const auto& estimator = cfg.numerics.diagnostics.refinement_estimator;
  if (!estimator.enabled || state.step % estimator.every != 0) {
    return;
  }

  const std::size_t n_cells = state.rho.size();
  const bool has_multiblock = state.mesh.topo.multiblock.has_value();
  if (!has_multiblock &&
      (state.mesh.dim != 2 || state.mesh.topo.nr < 5 ||
       state.mesh.topo.nz < 5 ||
       n_cells != static_cast<std::size_t>(state.mesh.topo.nr) *
                      static_cast<std::size_t>(state.mesh.topo.nz))) {
    static bool warned_missing_structured_grid = false;
    if (!warned_missing_structured_grid) {
      warned_missing_structured_grid = true;
      core::log_warning(
          "[refine_est] structured 2D grid unavailable; estimator skipped");
    }
    return;
  }

  if (state.refine_error.size() != n_cells) {
    state.refine_error.resize(n_cells);
  }
  std::fill(state.refine_error.begin(), state.refine_error.end(), 0.0);

  const std::vector<double> rho = copy_field(state.rho);
  const std::vector<double> pe = copy_field(state.Pe);
  const std::vector<double> pi = copy_field(state.Pi);
  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  std::vector<double> pressure(n_cells, 0.0);
  for (std::size_t c = 0; c < n_cells; ++c) {
    pressure[c] = pe[c] + pi[c];
  }

  double max_e = 0.0;
  std::size_t n_computed = 0;
  std::size_t n_above_02 = 0;
  std::size_t n_above_05 = 0;
  std::size_t n_above_08 = 0;
  double front_radius_weighted_cm = 0.0;
  double front_weight = 0.0;

  std::vector<double> centroid_s(n_cells, 0.0);
  std::vector<double> centroid_theta(n_cells, 0.0);
  std::vector<double> centroid_r;
  std::vector<double> centroid_z;

  if (has_multiblock) {
    const auto& mb = *state.mesh.topo.multiblock;
    std::vector<unsigned char> qualifying_blocks(mb.blocks.size(), 0U);
    for (std::size_t b = 0; b < mb.blocks.size(); ++b) {
      const auto& block = mb.blocks[b];
      if (block.cell_count != block.n_i_cells * block.n_j_cells) {
        static bool warned_skipped_block = false;
        if (!warned_skipped_block) {
          warned_skipped_block = true;
          core::log_warning(
              "[refine_est] block " + std::to_string(b) +
              " cell count does not match n_i*n_j; block skipped");
        }
        continue;
      }

      qualifying_blocks[b] = 1U;
      for (int local_cell = 0; local_cell < block.cell_count; ++local_cell) {
        const int cell = block.cell_begin + local_cell;
        double cell_centroid_r = 0.0;
        double cell_centroid_z = 0.0;
        cell_centroid(
            mb, cell, node_r, node_z, cell_centroid_r, cell_centroid_z);
        centroid_s[static_cast<std::size_t>(cell)] =
            std::hypot(cell_centroid_r, cell_centroid_z);
        centroid_theta[static_cast<std::size_t>(cell)] =
            std::atan2(cell_centroid_r, cell_centroid_z);
      }
    }

    for (std::size_t b = 0; b < mb.blocks.size(); ++b) {
      if (qualifying_blocks[b] == 0U) {
        continue;
      }
      const auto& block = mb.blocks[b];
      evaluate_block({block.cell_begin, block.n_i_cells, block.n_j_cells},
                     MetricMode::POLAR_ARC,
                     rho,
                     pressure,
                     centroid_s,
                     centroid_theta,
                     centroid_r,
                     centroid_z,
                     estimator.filter_eps,
                     estimator.detect_cutoff,
                     state.refine_error,
                     max_e,
                     n_computed,
                     n_above_02,
                     n_above_05,
                     n_above_08,
                     front_radius_weighted_cm,
                     front_weight);
    }
  } else {
    const MetricMode metric_mode =
        cfg.mesh.logical_mesh_2d.rfind("spherical_polar", 0) == 0
            ? MetricMode::POLAR_ARC
            : MetricMode::COORDINATE;
    if (metric_mode == MetricMode::COORDINATE) {
      centroid_r.resize(n_cells, 0.0);
      centroid_z.resize(n_cells, 0.0);
    }
    for (std::size_t cell = 0; cell < n_cells; ++cell) {
      double cell_centroid_r = 0.0;
      double cell_centroid_z = 0.0;
      structured_cell_centroid(state.mesh.topo,
                               static_cast<int>(cell),
                               node_r,
                               node_z,
                               cell_centroid_r,
                               cell_centroid_z);
      centroid_s[cell] = std::hypot(cell_centroid_r, cell_centroid_z);
      centroid_theta[cell] = std::atan2(cell_centroid_r, cell_centroid_z);
      if (metric_mode == MetricMode::COORDINATE) {
        centroid_r[cell] = cell_centroid_r;
        centroid_z[cell] = cell_centroid_z;
      }
    }
    evaluate_block({0, state.mesh.topo.nr, state.mesh.topo.nz},
                   metric_mode,
                   rho,
                   pressure,
                   centroid_s,
                   centroid_theta,
                   centroid_r,
                   centroid_z,
                   estimator.filter_eps,
                   estimator.detect_cutoff,
                   state.refine_error,
                   max_e,
                   n_computed,
                   n_above_02,
                   n_above_05,
                   n_above_08,
                   front_radius_weighted_cm,
                   front_weight);
  }

  const double inv_computed =
      n_computed > 0 ? 1.0 / static_cast<double>(n_computed) : 0.0;
  const double front_r_e_um =
      front_weight > 0.0
          ? 1.0e4 * front_radius_weighted_cm / front_weight
          : 0.0;

  std::ostringstream line;
  line << "[refine_est] step=" << state.step << " t=" << std::scientific
       << std::setprecision(17) << state.t << " maxE=" << std::fixed
       << std::setprecision(6) << max_e
       << " f02=" << static_cast<double>(n_above_02) * inv_computed
       << " f05=" << static_cast<double>(n_above_05) * inv_computed
       << " f08=" << static_cast<double>(n_above_08) * inv_computed
       << " front_rE_um=" << front_r_e_um;
  core::log_info(line.str());
}

}  // namespace tenryu::hydro
