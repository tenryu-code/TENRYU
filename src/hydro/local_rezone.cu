#include "hydro/local_rezone.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include "core/error.hpp"
#include "hydro/corner_jacobian_quality.cuh"
#include "mesh/mesh.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"

namespace tenryu::hydro::ale {
namespace {

struct CellPatch {
  int i0 = 0;
  int i1 = 0;
  int j0 = 0;
  int j1 = 0;
};

constexpr int kAxisVariationalMaxPicardPasses = 3;
constexpr int kAxisVariationalMaxFeasibilityIterations = 600;
constexpr double kAxisVariationalFloorEps = 1.0e-30;

struct LinearConstraint {
  std::vector<double> a;
  double b = 0.0;
  int cell = -1;
  int corner = -1;
};

struct AxisVariationalDof {
  int node = -1;
  int i = -1;
  int j = -1;
  bool is_r = false;
};

struct AxisVariationalContext {
  int nr = 0;
  int nz = 0;
  int affected_i0 = 0;
  int affected_i1 = -1;
  int affected_j0 = 0;
  int affected_j1 = -1;
  double coord_scale = 1.0;
  double coord_eps = 0.0;
  double j_floor = detail::kJFloor;
  double area_floor = kAxisVariationalFloorEps;
  double volume_floor = kAxisVariationalFloorEps;
  std::vector<double> r_base;
  std::vector<double> z_base;
  std::vector<int> r_dof;
  std::vector<int> z_dof;
  std::vector<AxisVariationalDof> dofs;
  std::vector<int> affected_cells;
};

int node_index(const int i, const int j, const int nz) {
  return rz_node_index_2d(i, j, nz);
}

CellPatch patch_from_focus(const int focus_cell,
                           const int nr,
                           const int nz,
                           const int radius_i,
                           const int radius_j) {
  int ci = (nr > 0) ? nr / 2 : 0;
  int cj = (nz > 0) ? nz / 2 : 0;
  if (focus_cell >= 0 && nz > 0) {
    ci = focus_cell / nz;
    cj = focus_cell - ci * nz;
  }
  ci = std::clamp(ci, 0, std::max(nr - 1, 0));
  cj = std::clamp(cj, 0, std::max(nz - 1, 0));
  CellPatch patch;
  patch.i0 = std::clamp(ci - std::max(radius_i, 0), 0, std::max(nr - 1, 0));
  patch.i1 = std::clamp(ci + std::max(radius_i, 0), 0, std::max(nr - 1, 0));
  patch.j0 = std::clamp(cj - std::max(radius_j, 0), 0, std::max(nz - 1, 0));
  patch.j1 = std::clamp(cj + std::max(radius_j, 0), 0, std::max(nz - 1, 0));
  return patch;
}

void exchange_nodes_if_needed(core::State& state,
                              const parallel::PartitionInfo& part,
                              parallel::CommBuffers* bufs) {
  if (part.n_ranks > 1 && bufs != nullptr) {
    double* node_ptrs[2] = {state.x_r.data(), state.x_z.data()};
    parallel::exchange_node_fields(
        part, *bufs, node_ptrs, 2, state.mesh.topo.n_nodes, nullptr, 6);
  }
}

double reference_r(const core::Config& cfg, const int i, const int nr) {
  const double dr = (cfg.mesh.r_max - cfg.mesh.r_min) /
                    static_cast<double>(std::max(nr, 1));
  return cfg.mesh.r_min + dr * static_cast<double>(i);
}

double reference_z(const core::Config& cfg, const int j, const int nz) {
  const double dz = (cfg.mesh.z_max - cfg.mesh.z_min) /
                    static_cast<double>(std::max(nz, 1));
  return cfg.mesh.z_min + dz * static_cast<double>(j);
}

double eval_corner_j(const std::vector<double>& r,
                     const std::vector<double>& z,
                     const int nr,
                     const int nz,
                     const int ci,
                     const int cj,
                     const int corner) {
  (void)nr;
  const int nodes[4] = {node_index(ci, cj, nz),
                        node_index(ci + 1, cj, nz),
                        node_index(ci + 1, cj + 1, nz),
                        node_index(ci, cj + 1, nz)};
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    rr[k] = r[static_cast<std::size_t>(nodes[k])];
    zz[k] = z[static_cast<std::size_t>(nodes[k])];
  }
  return corner_jacobian_from_quad(rr, zz, corner);
}

double min_patch_corner_j(const std::vector<double>& r,
                          const std::vector<double>& z,
                          const int nr,
                          const int nz,
                          const CellPatch& patch) {
  double min_j = std::numeric_limits<double>::infinity();
  for (int ci = patch.i0; ci <= patch.i1; ++ci) {
    for (int cj = patch.j0; cj <= patch.j1; ++cj) {
      for (int corner = 0; corner < 4; ++corner) {
        const double j = eval_corner_j(r, z, nr, nz, ci, cj, corner);
        if (!std::isfinite(j)) {
          return -std::numeric_limits<double>::infinity();
        }
        min_j = std::min(min_j, j);
      }
    }
  }
  return min_j == std::numeric_limits<double>::infinity() ? 1.0 : min_j;
}

double axis_variational_coord_value(const AxisVariationalContext& ctx,
                                    const int node,
                                    const bool is_r,
                                    const std::vector<double>& x_values) {
  const int dof = is_r ? ctx.r_dof[static_cast<std::size_t>(node)]
                       : ctx.z_dof[static_cast<std::size_t>(node)];
  if (dof >= 0) {
    return x_values[static_cast<std::size_t>(dof)];
  }
  return is_r ? ctx.r_base[static_cast<std::size_t>(node)]
              : ctx.z_base[static_cast<std::size_t>(node)];
}

double eval_axis_variational_corner_j(const AxisVariationalContext& ctx,
                                      const int ci,
                                      const int cj,
                                      const int corner,
                                      const std::vector<double>& x_values) {
  const int nodes[4] = {node_index(ci, cj, ctx.nz),
                        node_index(ci + 1, cj, ctx.nz),
                        node_index(ci + 1, cj + 1, ctx.nz),
                        node_index(ci, cj + 1, ctx.nz)};
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    rr[k] = axis_variational_coord_value(ctx, nodes[k], true, x_values);
    zz[k] = axis_variational_coord_value(ctx, nodes[k], false, x_values);
  }
  return corner_jacobian_from_quad(rr, zz, corner);
}

double eval_axis_variational_area(const AxisVariationalContext& ctx,
                                  const int ci,
                                  const int cj,
                                  const std::vector<double>& x_values) {
  const int nodes[4] = {node_index(ci, cj, ctx.nz),
                        node_index(ci + 1, cj, ctx.nz),
                        node_index(ci + 1, cj + 1, ctx.nz),
                        node_index(ci, cj + 1, ctx.nz)};
  double rr[4];
  double zz[4];
  for (int k = 0; k < 4; ++k) {
    rr[k] = axis_variational_coord_value(ctx, nodes[k], true, x_values);
    zz[k] = axis_variational_coord_value(ctx, nodes[k], false, x_values);
  }
  const double cross = rr[0] * zz[1] - zz[0] * rr[1] +
                       rr[1] * zz[2] - zz[1] * rr[2] +
                       rr[2] * zz[3] - zz[2] * rr[3] +
                       rr[3] * zz[0] - zz[3] * rr[0];
  return 0.5 * std::fabs(cross);
}

double eval_axis_variational_volume(const AxisVariationalContext& ctx,
                                    const int ci,
                                    const int cj,
                                    const std::vector<double>& x_values) {
  const int n00 = node_index(ci, cj, ctx.nz);
  const int n10 = node_index(ci + 1, cj, ctx.nz);
  const int n11 = node_index(ci + 1, cj + 1, ctx.nz);
  const int n01 = node_index(ci, cj + 1, ctx.nz);
  return detail::rz_signed_quad_volume(
      axis_variational_coord_value(ctx, n00, true, x_values),
      axis_variational_coord_value(ctx, n00, false, x_values),
      axis_variational_coord_value(ctx, n10, true, x_values),
      axis_variational_coord_value(ctx, n10, false, x_values),
      axis_variational_coord_value(ctx, n11, true, x_values),
      axis_variational_coord_value(ctx, n11, false, x_values),
      axis_variational_coord_value(ctx, n01, true, x_values),
      axis_variational_coord_value(ctx, n01, false, x_values));
}

double axis_variational_fd_step(const AxisVariationalContext& ctx,
                                const std::vector<double>& x_ref,
                                const int dof) {
  return std::max(1.0e-8 * ctx.coord_scale,
                  64.0 * std::numeric_limits<double>::epsilon() *
                      std::max(ctx.coord_scale,
                               std::fabs(x_ref[static_cast<std::size_t>(dof)])));
}

template <typename Eval>
bool add_axis_variational_linearized_constraint(
    const AxisVariationalContext& ctx,
    const std::vector<double>& x_ref,
    const double floor,
    const int cell,
    const int corner,
    const Eval& eval,
    std::vector<LinearConstraint>& constraints,
    int* impossible_constraint) {
  const int ndof = static_cast<int>(ctx.dofs.size());
  const double value0 = eval(x_ref);
  if (!std::isfinite(value0)) {
    if (impossible_constraint != nullptr) {
      *impossible_constraint = static_cast<int>(constraints.size());
    }
    return false;
  }

  LinearConstraint con;
  con.a.assign(static_cast<std::size_t>(ndof), 0.0);
  con.cell = cell;
  con.corner = corner;
  double dot_grad_x = 0.0;
  double norm2 = 0.0;
  for (int d = 0; d < ndof; ++d) {
    std::vector<double> x_plus = x_ref;
    const double h = axis_variational_fd_step(ctx, x_ref, d);
    x_plus[static_cast<std::size_t>(d)] += h;
    const double value_plus = eval(x_plus);
    if (!std::isfinite(value_plus)) {
      if (impossible_constraint != nullptr) {
        *impossible_constraint = static_cast<int>(constraints.size());
      }
      return false;
    }
    const double grad = (value_plus - value0) / h;
    con.a[static_cast<std::size_t>(d)] = grad;
    dot_grad_x += grad * x_ref[static_cast<std::size_t>(d)];
    norm2 += grad * grad;
  }
  if (norm2 <= std::numeric_limits<double>::min()) {
    if (value0 < floor) {
      if (impossible_constraint != nullptr) {
        *impossible_constraint = static_cast<int>(constraints.size());
      }
      return false;
    }
    return true;
  }
  con.b = floor - value0 + dot_grad_x;
  constraints.push_back(con);
  return true;
}

void add_axis_variational_difference_constraint(
    const AxisVariationalContext& ctx,
    const int positive_node,
    const int negative_node,
    const bool is_r,
    const double min_spacing,
    std::vector<LinearConstraint>& constraints) {
  LinearConstraint con;
  con.a.assign(ctx.dofs.size(), 0.0);
  double fixed = 0.0;
  const int positive_dof = is_r
                               ? ctx.r_dof[static_cast<std::size_t>(positive_node)]
                               : ctx.z_dof[static_cast<std::size_t>(positive_node)];
  if (positive_dof >= 0) {
    con.a[static_cast<std::size_t>(positive_dof)] += 1.0;
  } else {
    fixed += is_r ? ctx.r_base[static_cast<std::size_t>(positive_node)]
                  : ctx.z_base[static_cast<std::size_t>(positive_node)];
  }
  const int negative_dof = is_r
                               ? ctx.r_dof[static_cast<std::size_t>(negative_node)]
                               : ctx.z_dof[static_cast<std::size_t>(negative_node)];
  if (negative_dof >= 0) {
    con.a[static_cast<std::size_t>(negative_dof)] -= 1.0;
  } else {
    fixed -= is_r ? ctx.r_base[static_cast<std::size_t>(negative_node)]
                  : ctx.z_base[static_cast<std::size_t>(negative_node)];
  }
  con.b = min_spacing - fixed;
  constraints.push_back(con);
}

double compute_axis_variational_j_floor(const core::Config& cfg,
                                        const AxisVariationalContext& ctx,
                                        const std::vector<double>& x_values) {
  double max_abs_j = 0.0;
  for (const int c : ctx.affected_cells) {
    const int ci = c / ctx.nz;
    const int cj = c - ci * ctx.nz;
    for (int corner = 0; corner < 4; ++corner) {
      const double j = eval_axis_variational_corner_j(ctx, ci, cj, corner, x_values);
      if (std::isfinite(j)) {
        max_abs_j = std::max(max_abs_j, std::fabs(j));
      }
    }
  }
  const double floor_eps =
      std::max(cfg.numerics.hydro.corner_jacobian_floor_eps, 0.0);
  return std::max(detail::kJFloor, floor_eps * max_abs_j);
}

bool build_axis_variational_constraints(
    const AxisVariationalContext& ctx,
    const std::vector<double>& x_ref,
    std::vector<LinearConstraint>& constraints,
    int* impossible_constraint) {
  constraints.clear();
  if (impossible_constraint != nullptr) {
    *impossible_constraint = -1;
  }

  for (const int c : ctx.affected_cells) {
    const int ci = c / ctx.nz;
    const int cj = c - ci * ctx.nz;
    for (int corner = 0; corner < 4; ++corner) {
      const auto eval = [&](const std::vector<double>& x_values) {
        return eval_axis_variational_corner_j(ctx, ci, cj, corner, x_values);
      };
      if (!add_axis_variational_linearized_constraint(ctx,
                                                      x_ref,
                                                      ctx.j_floor,
                                                      c,
                                                      corner,
                                                      eval,
                                                      constraints,
                                                      impossible_constraint)) {
        return false;
      }
    }
  }

  for (int i = ctx.affected_i0; i <= ctx.affected_i1; ++i) {
    for (int j = ctx.affected_j0; j <= ctx.affected_j1 + 1; ++j) {
      if (i < 0 || i >= ctx.nr || j < 0 || j > ctx.nz) {
        continue;
      }
      add_axis_variational_difference_constraint(ctx,
                                                 node_index(i + 1, j, ctx.nz),
                                                 node_index(i, j, ctx.nz),
                                                 true,
                                                 ctx.coord_eps,
                                                 constraints);
    }
  }

  for (int i = std::max(1, ctx.affected_i0); i <= ctx.affected_i1 + 1; ++i) {
    for (int j = ctx.affected_j0; j <= ctx.affected_j1; ++j) {
      if (i < 1 || i > ctx.nr || j < 0 || j >= ctx.nz) {
        continue;
      }
      add_axis_variational_difference_constraint(ctx,
                                                 node_index(i, j + 1, ctx.nz),
                                                 node_index(i, j, ctx.nz),
                                                 false,
                                                 ctx.coord_eps,
                                                 constraints);
    }
  }

  for (const int c : ctx.affected_cells) {
    const int ci = c / ctx.nz;
    const int cj = c - ci * ctx.nz;
    const auto eval_area = [&](const std::vector<double>& x_values) {
      return eval_axis_variational_area(ctx, ci, cj, x_values);
    };
    if (!add_axis_variational_linearized_constraint(ctx,
                                                    x_ref,
                                                    ctx.area_floor,
                                                    c,
                                                    -1,
                                                    eval_area,
                                                    constraints,
                                                    impossible_constraint)) {
      return false;
    }
    const auto eval_volume = [&](const std::vector<double>& x_values) {
      return eval_axis_variational_volume(ctx, ci, cj, x_values);
    };
    if (!add_axis_variational_linearized_constraint(ctx,
                                                    x_ref,
                                                    ctx.volume_floor,
                                                    c,
                                                    -1,
                                                    eval_volume,
                                                    constraints,
                                                    impossible_constraint)) {
      return false;
    }
  }

  return true;
}

bool solve_axis_variational_feasibility(
    const AxisVariationalContext& ctx,
    const std::vector<LinearConstraint>& constraints,
    std::vector<double>& x_values,
    int* iterations,
    int* infeasible_constraint) {
  const int ndof = static_cast<int>(ctx.dofs.size());
  const double slack_tol = 1.0e-24;
  if (iterations != nullptr) {
    *iterations = 0;
  }
  if (infeasible_constraint != nullptr) {
    *infeasible_constraint = -1;
  }
  for (int iter = 0; iter < kAxisVariationalMaxFeasibilityIterations; ++iter) {
    double worst_slack = std::numeric_limits<double>::infinity();
    int worst = -1;
    for (int cidx = 0; cidx < static_cast<int>(constraints.size()); ++cidx) {
      const LinearConstraint& con = constraints[static_cast<std::size_t>(cidx)];
      double lhs = 0.0;
      double norm2 = 0.0;
      for (int d = 0; d < ndof; ++d) {
        const double a = con.a[static_cast<std::size_t>(d)];
        lhs += a * x_values[static_cast<std::size_t>(d)];
        norm2 += a * a;
      }
      const double slack = lhs - con.b;
      if (slack < worst_slack) {
        worst_slack = slack;
        worst = cidx;
      }
      if (slack < -slack_tol && norm2 <= std::numeric_limits<double>::min()) {
        if (iterations != nullptr) {
          *iterations += iter + 1;
        }
        if (infeasible_constraint != nullptr) {
          *infeasible_constraint = cidx;
        }
        return false;
      }
    }
    if (worst < 0 || worst_slack >= -slack_tol) {
      if (iterations != nullptr) {
        *iterations += iter + 1;
      }
      return true;
    }
    const LinearConstraint& con = constraints[static_cast<std::size_t>(worst)];
    double lhs = 0.0;
    double norm2 = 0.0;
    for (int d = 0; d < ndof; ++d) {
      const double a = con.a[static_cast<std::size_t>(d)];
      lhs += a * x_values[static_cast<std::size_t>(d)];
      norm2 += a * a;
    }
    if (norm2 <= std::numeric_limits<double>::min()) {
      if (iterations != nullptr) {
        *iterations += iter + 1;
      }
      if (infeasible_constraint != nullptr) {
        *infeasible_constraint = worst;
      }
      return false;
    }
    const double alpha = (con.b - lhs) / norm2;
    for (int d = 0; d < ndof; ++d) {
      x_values[static_cast<std::size_t>(d)] +=
          alpha * con.a[static_cast<std::size_t>(d)];
    }
  }
  if (iterations != nullptr) {
    *iterations += kAxisVariationalMaxFeasibilityIterations;
  }
  if (infeasible_constraint != nullptr) {
    *infeasible_constraint = -1;
  }
  return false;
}

bool exact_axis_variational_corner_j_valid(const AxisVariationalContext& ctx,
                                           const std::vector<double>& x_values,
                                           double* min_corner_j) {
  double local_min = std::numeric_limits<double>::infinity();
  for (const int c : ctx.affected_cells) {
    const int ci = c / ctx.nz;
    const int cj = c - ci * ctx.nz;
    for (int corner = 0; corner < 4; ++corner) {
      const double j = eval_axis_variational_corner_j(ctx, ci, cj, corner, x_values);
      if (!std::isfinite(j) || j < ctx.j_floor) {
        if (min_corner_j != nullptr) {
          *min_corner_j = std::isfinite(j) ? j : -1.0;
        }
        return false;
      }
      local_min = std::min(local_min, j);
    }
  }
  if (min_corner_j != nullptr) {
    *min_corner_j =
        (local_min == std::numeric_limits<double>::infinity()) ? 1.0 : local_min;
  }
  return true;
}

void mark_axis_band_moves(const CellPatch& patch,
                          const int nr,
                          const int nz,
                          std::vector<std::uint8_t>& move_r,
                          std::vector<std::uint8_t>& move_z) {
  for (int i = patch.i0; i <= patch.i1 + 1; ++i) {
    for (int j = patch.j0; j <= patch.j1 + 1; ++j) {
      if (i < 0 || i > nr || j < 0 || j > nz) {
        continue;
      }
      const int id = node_index(i, j, nz);
      if (i == 0) {
        move_r[static_cast<std::size_t>(id)] = 0;
        continue;
      }
      move_r[static_cast<std::size_t>(id)] = 1;
      move_z[static_cast<std::size_t>(id)] = 1;
    }
  }
}

void mark_boundary_moves(const CellPatch& patch,
                         const int nr,
                         const int nz,
                         std::vector<std::uint8_t>& move_r,
                         std::vector<std::uint8_t>& move_z) {
  for (int i = patch.i0; i <= patch.i1 + 1; ++i) {
    for (int j = patch.j0; j <= patch.j1 + 1; ++j) {
      if (i < 0 || i > nr || j < 0 || j > nz || i == 0) {
        continue;
      }
      const bool patch_edge =
          i == patch.i0 || i == patch.i1 + 1 || j == patch.j0 || j == patch.j1 + 1;
      const bool z_domain = j == 0 || j == nz;
      const bool r_outer = i == nr;
      const int id = node_index(i, j, nz);
      if (z_domain && !r_outer) {
        move_r[static_cast<std::size_t>(id)] = 1;
      } else if (r_outer && !z_domain) {
        move_z[static_cast<std::size_t>(id)] = 1;
      } else if (!patch_edge && !z_domain && !r_outer) {
        move_r[static_cast<std::size_t>(id)] = 1;
        move_z[static_cast<std::size_t>(id)] = 1;
      }
    }
  }
}

void mark_interior_moves(const CellPatch& patch,
                         const int nr,
                         const int nz,
                         std::vector<std::uint8_t>& move_r,
                         std::vector<std::uint8_t>& move_z) {
  for (int i = patch.i0 + 1; i <= patch.i1; ++i) {
    for (int j = patch.j0 + 1; j <= patch.j1; ++j) {
      if (i <= 0 || i >= nr || j <= 0 || j >= nz) {
        continue;
      }
      const int id = node_index(i, j, nz);
      move_r[static_cast<std::size_t>(id)] = 1;
      move_z[static_cast<std::size_t>(id)] = 1;
    }
  }
}

bool apply_reference_projection(core::State& state,
                                const core::Config& cfg,
                                const CellPatch& patch,
                                const std::vector<std::uint8_t>& move_r,
                                const std::vector<std::uint8_t>& move_z,
                                double* min_corner_j) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  std::vector<double> r;
  std::vector<double> z;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);
  const std::vector<double> r_original = r;
  const std::vector<double> z_original = z;

  for (int i = patch.i0; i <= patch.i1 + 1; ++i) {
    for (int j = patch.j0; j <= patch.j1 + 1; ++j) {
      if (i < 0 || i > nr || j < 0 || j > nz) {
        continue;
      }
      const int id = node_index(i, j, nz);
      if (cfg.numerics.has_physical_rz_axis && i == 0) {
        r[static_cast<std::size_t>(id)] = 0.0;
      }
      if (move_r[static_cast<std::size_t>(id)] != 0) {
        r[static_cast<std::size_t>(id)] = reference_r(cfg, i, nr);
      }
      if (move_z[static_cast<std::size_t>(id)] != 0) {
        z[static_cast<std::size_t>(id)] = reference_z(cfg, j, nz);
      }
    }
  }

  const double span = std::max(std::abs(cfg.mesh.r_max - cfg.mesh.r_min),
                               std::abs(cfg.mesh.z_max - cfg.mesh.z_min));
  const double eps = std::max(1.0, span) * 64.0 * std::numeric_limits<double>::epsilon();
  for (int iter = 0; iter < 8; ++iter) {
    for (int j = patch.j0; j <= patch.j1 + 1; ++j) {
      for (int i = std::max(1, patch.i0); i <= patch.i1 + 1; ++i) {
        const int id = node_index(i, j, nz);
        const int prev = node_index(i - 1, j, nz);
        if (move_r[static_cast<std::size_t>(id)] != 0 &&
            r[static_cast<std::size_t>(id)] <=
                r[static_cast<std::size_t>(prev)] + eps) {
          r[static_cast<std::size_t>(id)] =
              r[static_cast<std::size_t>(prev)] + eps;
        }
      }
    }
    for (int i = patch.i0; i <= patch.i1 + 1; ++i) {
      for (int j = std::max(1, patch.j0); j <= patch.j1 + 1; ++j) {
        const int id = node_index(i, j, nz);
        const int prev = node_index(i, j - 1, nz);
        if (move_z[static_cast<std::size_t>(id)] != 0 &&
            z[static_cast<std::size_t>(id)] <=
                z[static_cast<std::size_t>(prev)] + eps) {
          z[static_cast<std::size_t>(id)] =
              z[static_cast<std::size_t>(prev)] + eps;
        }
      }
    }
  }

  const double min_j = min_patch_corner_j(r, z, nr, nz, patch);
  if (min_corner_j != nullptr) {
    *min_corner_j = min_j;
  }
  if (!(min_j > detail::kJFloor)) {
    state.x_r.copy_from_host(r_original);
    state.x_z.copy_from_host(z_original);
    return false;
  }
  state.x_r.copy_from_host(r);
  state.x_z.copy_from_host(z);
  return true;
}

RezoneResult finish_projection(core::State& state,
                               const core::Config& cfg,
                               const parallel::PartitionInfo& part,
                               parallel::CommBuffers* bufs,
                               const parallel::Reduction* reduction,
                               const int iterations,
                               const double residual) {
  RezoneResult out;
  out.triggered = true;
  out.converged = true;
  out.iterations = iterations;
  out.residual = residual;
  exchange_nodes_if_needed(state, part, bufs);
  bool post_tangle = false;
  out.min_quality =
      compute_min_quality(state, cfg, &post_tangle, &out.min_quality_cell_post);
  if (detail::evacuated_constraint_masks_present(state)) {
    out.min_quality = compute_min_quality(state, cfg, nullptr, nullptr, true);
  }
  out.mesh_tangle = post_tangle;
  out.final_axis_margin =
      compute_axis_margin_min(state, cfg.numerics.has_physical_rz_axis).min_margin;
  if (reduction != nullptr) {
    out.min_quality = reduction->allreduce_min(out.min_quality);
    out.mesh_tangle = reduction->allreduce_max(out.mesh_tangle ? 1.0 : 0.0) > 0.5;
    out.final_axis_margin = reduction->allreduce_min(out.final_axis_margin);
  }
  if (out.mesh_tangle) {
    out.converged = false;
  }
  return out;
}

}  // namespace

RezoneResult run_axis_spine_plus_local_rezone(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request,
    const CellRegime*) {
  RezoneResult pre;
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return pre;
  }
  pre.min_quality =
      compute_min_quality(state, cfg, &pre.mesh_tangle, &pre.min_quality_cell_pre);
  if (!cfg.numerics.has_physical_rz_axis) {
    return pre;
  }
  constexpr double kAxisSpineRepairAlpha = 0.5;
  constexpr int kAxisSpineRepairMaxSweeps = 10;
  constexpr double kAxisSpineRepairTolZ = 1.0e-12;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  std::vector<double> r_original;
  std::vector<double> z_original;
  state.x_r.copy_to_host(r_original);
  state.x_z.copy_to_host(z_original);
  double max_dz = 0.0;
  for (int sweep = 0; sweep < kAxisSpineRepairMaxSweeps; ++sweep) {
    max_dz = minimal_axis_z_repair(state.x_r,
                                   state.x_z,
                                   nr,
                                   nz,
                                   state.evacuated_cells.n_contact_active_cells > 0
                                       ? state.evacuated_cells.d_contact_active_cells.data()
                                       : nullptr,
                                   state.evacuated_cells.n_contact_active_cells,
                                   nz,
                                   kAxisSpineRepairAlpha,
                                   1,
                                   kAxisSpineRepairTolZ);
    if (max_dz < kAxisSpineRepairTolZ) {
      break;
    }
  }
  CellPatch patch = patch_from_focus(request.focus_cell,
                                     nr,
                                     nz,
                                     cfg.numerics.hydro.axis_guard_band_cells,
                                     request.patch_radius_j);
  patch.i0 = 0;
  patch.i1 = std::min(nr - 1, std::max(0, cfg.numerics.hydro.axis_guard_band_cells));
  std::vector<std::uint8_t> move_r(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0);
  std::vector<std::uint8_t> move_z(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0);
  mark_axis_band_moves(patch, nr, nz, move_r, move_z);
  double min_j = 1.0;
  if (!apply_reference_projection(state, cfg, patch, move_r, move_z, &min_j)) {
    state.x_r.copy_from_host(r_original);
    state.x_z.copy_from_host(z_original);
    pre.triggered = false;
    pre.converged = false;
    pre.residual = min_j;
    return pre;
  }
  return finish_projection(
      state, cfg, part, bufs, reduction, kAxisSpineRepairMaxSweeps, max_dz);
}

RezoneResult run_axis_variational_projection(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request,
    const CellRegime*) {
  RezoneResult pre;
  if (mesh::mesh_topo_is_multiblock(cfg.mesh)) {
    return pre;
  }
  pre.min_quality =
      compute_min_quality(state, cfg, &pre.mesh_tangle, &pre.min_quality_cell_pre);
  if (!cfg.numerics.has_physical_rz_axis) {
    return pre;
  }
  constexpr double kAxisSpineRepairAlpha = 0.5;
  constexpr int kAxisSpineRepairMaxSweeps = 10;
  constexpr double kAxisSpineRepairTolZ = 1.0e-12;
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    pre.triggered = false;
    pre.converged = true;
    pre.residual = 0.0;
    return pre;
  }

  std::vector<double> r_original;
  std::vector<double> z_original;
  state.x_r.copy_to_host(r_original);
  state.x_z.copy_to_host(z_original);

  double max_dz = 0.0;
  for (int sweep = 0; sweep < kAxisSpineRepairMaxSweeps; ++sweep) {
    max_dz = minimal_axis_z_repair(state.x_r,
                                   state.x_z,
                                   nr,
                                   nz,
                                   state.evacuated_cells.n_contact_active_cells > 0
                                       ? state.evacuated_cells.d_contact_active_cells.data()
                                       : nullptr,
                                   state.evacuated_cells.n_contact_active_cells,
                                   nz,
                                   kAxisSpineRepairAlpha,
                                   1,
                                   kAxisSpineRepairTolZ);
    if (max_dz < kAxisSpineRepairTolZ) {
      break;
    }
  }

  std::vector<double> r_lag;
  std::vector<double> z_lag;
  state.x_r.copy_to_host(r_lag);
  state.x_z.copy_to_host(z_lag);
  TENRYU_ASSERT(r_lag.size() == z_lag.size(),
                "axis variational projection requires matching node fields");
  TENRYU_ASSERT(static_cast<int>(r_lag.size()) >= (nr + 1) * (nz + 1),
                "axis variational projection node field size is smaller than mesh");
  for (int j = 0; j <= nz; ++j) {
    r_lag[static_cast<std::size_t>(node_index(0, j, nz))] = 0.0;
  }

  std::vector<std::uint8_t> contact_frozen_nodes(r_lag.size(), 0U);
  for (const core::EvacContactSlot& slot : state.contact_graph.records) {
    if (slot.state != core::EvacContactState::kActive) {
      continue;
    }
    for (int p = 0; p < 2; ++p) {
      if (slot.pair_engaged[p] == 0U) {
        continue;
      }
      const int pair_nodes[2] = {slot.node_a[p], slot.node_b[p]};
      for (const int node : pair_nodes) {
        if (node >= 0 &&
            node < static_cast<int>(contact_frozen_nodes.size())) {
          contact_frozen_nodes[static_cast<std::size_t>(node)] = 1U;
        }
      }
    }
  }

  CellPatch patch = patch_from_focus(request.focus_cell,
                                     nr,
                                     nz,
                                     cfg.numerics.hydro.axis_guard_band_cells,
                                     request.patch_radius_j);
  patch.i0 = 0;
  patch.i1 = std::min(nr - 1, std::max(0, cfg.numerics.hydro.axis_guard_band_cells));

  AxisVariationalContext ctx;
  ctx.nr = nr;
  ctx.nz = nz;
  ctx.affected_i0 = patch.i0;
  ctx.affected_i1 = std::min(nr - 1, patch.i1 + 1);
  ctx.affected_j0 = std::max(0, patch.j0 - 1);
  ctx.affected_j1 = std::min(nz - 1, patch.j1 + 1);
  ctx.r_base = r_lag;
  ctx.z_base = z_lag;
  ctx.r_dof.assign(r_lag.size(), -1);
  ctx.z_dof.assign(z_lag.size(), -1);
  const double domain_span =
      std::max(std::fabs(cfg.mesh.r_max - cfg.mesh.r_min),
               std::fabs(cfg.mesh.z_max - cfg.mesh.z_min));
  ctx.coord_scale = std::max(1.0, domain_span);
  ctx.coord_eps = 64.0 * std::numeric_limits<double>::epsilon() * ctx.coord_scale;
  ctx.area_floor = kAxisVariationalFloorEps;
  ctx.volume_floor = kAxisVariationalFloorEps;

  for (int i = 1; i <= patch.i1 + 1; ++i) {
    for (int j = patch.j0; j <= patch.j1 + 1; ++j) {
      if (i <= 0 || i >= nr || j <= 0 || j >= nz) {
        continue;
      }
      const int id = node_index(i, j, nz);
      if (contact_frozen_nodes[static_cast<std::size_t>(id)] != 0U) {
        continue;
      }
      ctx.r_dof[static_cast<std::size_t>(id)] =
          static_cast<int>(ctx.dofs.size());
      ctx.dofs.push_back(AxisVariationalDof{id, i, j, true});
      ctx.z_dof[static_cast<std::size_t>(id)] =
          static_cast<int>(ctx.dofs.size());
      ctx.dofs.push_back(AxisVariationalDof{id, i, j, false});
    }
  }

  const std::vector<std::uint8_t>& contact_active_mask =
      state.evacuated_cells.contact_active_mask;
  const std::vector<std::uint8_t>& axis_edge_collapsed =
      state.evacuated_cells.cell_axis_edge_collapsed;
  for (int ci = ctx.affected_i0; ci <= ctx.affected_i1; ++ci) {
    for (int cj = ctx.affected_j0; cj <= ctx.affected_j1; ++cj) {
      if (ci < 0 || ci >= nr || cj < 0 || cj >= nz) {
        continue;
      }
      const int cell = ci * nz + cj;
      if (cell < static_cast<int>(contact_active_mask.size()) &&
          contact_active_mask[static_cast<std::size_t>(cell)] != 0U) {
        continue;
      }
      if (cell < static_cast<int>(axis_edge_collapsed.size()) &&
          axis_edge_collapsed[static_cast<std::size_t>(cell)] != 0U) {
        continue;
      }
      ctx.affected_cells.push_back(cell);
    }
  }

  if (ctx.dofs.empty() || ctx.affected_cells.empty()) {
    state.x_r.copy_from_host(r_original);
    state.x_z.copy_from_host(z_original);
    pre.triggered = false;
    pre.converged = true;
    pre.residual = 0.0;
    return pre;
  }

  std::vector<double> x_lag(ctx.dofs.size(), 0.0);
  std::vector<double> x_target(ctx.dofs.size(), 0.0);
  for (int d = 0; d < static_cast<int>(ctx.dofs.size()); ++d) {
    const AxisVariationalDof& dof = ctx.dofs[static_cast<std::size_t>(d)];
    const double lag = dof.is_r ? ctx.r_base[static_cast<std::size_t>(dof.node)]
                                : ctx.z_base[static_cast<std::size_t>(dof.node)];
    x_lag[static_cast<std::size_t>(d)] = lag;
    x_target[static_cast<std::size_t>(d)] = lag;
  }
  ctx.j_floor = compute_axis_variational_j_floor(cfg, ctx, x_lag);

  std::vector<double> x = x_lag;
  std::vector<double> best_x = x_lag;
  std::vector<LinearConstraint> constraints;
  int total_iterations = 0;
  int infeasible_constraint = -1;
  double min_corner_j = 1.0;
  (void)exact_axis_variational_corner_j_valid(ctx, x, &min_corner_j);
  bool solved = false;

  for (int picard = 0; picard < kAxisVariationalMaxPicardPasses; ++picard) {
    if (!build_axis_variational_constraints(ctx,
                                            x,
                                            constraints,
                                            &infeasible_constraint)) {
      break;
    }

    std::vector<double> candidate = x_target;
    int iterations = 0;
    if (!solve_axis_variational_feasibility(ctx,
                                            constraints,
                                            candidate,
                                            &iterations,
                                            &infeasible_constraint)) {
      total_iterations += iterations;
      (void)exact_axis_variational_corner_j_valid(ctx, candidate, &min_corner_j);
      break;
    }
    total_iterations += iterations;
    x = candidate;
    double candidate_min_j = 1.0;
    if (exact_axis_variational_corner_j_valid(ctx, x, &candidate_min_j)) {
      solved = true;
      best_x = x;
      min_corner_j = candidate_min_j;
      break;
    }
    min_corner_j = candidate_min_j;
  }

  if (!solved) {
    state.x_r.copy_from_host(r_original);
    state.x_z.copy_from_host(z_original);
    pre.triggered = false;
    pre.converged = false;
    pre.iterations = total_iterations;
    pre.residual = min_corner_j;
    return pre;
  }

  std::vector<double> r_candidate = ctx.r_base;
  std::vector<double> z_candidate = ctx.z_base;
  for (int d = 0; d < static_cast<int>(ctx.dofs.size()); ++d) {
    const AxisVariationalDof& dof = ctx.dofs[static_cast<std::size_t>(d)];
    if (dof.is_r) {
      r_candidate[static_cast<std::size_t>(dof.node)] =
          best_x[static_cast<std::size_t>(d)];
    } else {
      z_candidate[static_cast<std::size_t>(dof.node)] =
          best_x[static_cast<std::size_t>(d)];
    }
  }
  for (int j = 0; j <= nz; ++j) {
    const int id = node_index(0, j, nz);
    r_candidate[static_cast<std::size_t>(id)] = 0.0;
    z_candidate[static_cast<std::size_t>(id)] =
        ctx.z_base[static_cast<std::size_t>(id)];
  }
  state.x_r.copy_from_host(r_candidate);
  state.x_z.copy_from_host(z_candidate);
  return finish_projection(state, cfg, part, bufs, reduction, total_iterations, min_corner_j);
}

RezoneResult run_boundary_patch_projection(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request) {
  RezoneResult pre;
  pre.min_quality =
      compute_min_quality(state, cfg, &pre.mesh_tangle, &pre.min_quality_cell_pre);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const CellPatch patch = patch_from_focus(request.focus_cell,
                                           nr,
                                           nz,
                                           request.patch_radius_i,
                                           request.patch_radius_j);
  std::vector<std::uint8_t> move_r(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0);
  std::vector<std::uint8_t> move_z(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0);
  mark_boundary_moves(patch, nr, nz, move_r, move_z);
  double min_j = 1.0;
  if (!apply_reference_projection(state, cfg, patch, move_r, move_z, &min_j)) {
    if (cfg.numerics.ale.multi_node_boundary_repair_enabled) {
      AleMinQualityCell fail_cell;
      int fail_corner = -1;
      double ignored_min_j = 1.0;
      (void)compute_corner_post_tangle(
          state, cfg, &fail_cell, &ignored_min_j, &fail_corner);
      const LocalBoundaryRepairResult repair =
          try_multi_node_boundary_repair(state, cfg, fail_cell, fail_corner);
      if (repair.applied) {
        return finish_projection(state,
                                 cfg,
                                 part,
                                 bufs,
                                 reduction,
                                 repair.solver_iterations,
                                 repair.min_corner_j);
      }
    }
    pre.triggered = false;
    pre.converged = false;
    pre.residual = min_j;
    return pre;
  }
  return finish_projection(state, cfg, part, bufs, reduction, 1, min_j);
}

RezoneResult run_interior_multi_node_projection(
    core::State& state,
    const core::Config& cfg,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs,
    const parallel::Reduction* reduction,
    const AleRequest& request) {
  RezoneResult pre;
  pre.min_quality =
      compute_min_quality(state, cfg, &pre.mesh_tangle, &pre.min_quality_cell_pre);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  const CellPatch patch = patch_from_focus(request.focus_cell,
                                           nr,
                                           nz,
                                           std::max(1, request.patch_radius_i),
                                           std::max(1, request.patch_radius_j));
  std::vector<std::uint8_t> move_r(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0);
  std::vector<std::uint8_t> move_z(static_cast<std::size_t>(state.mesh.topo.n_nodes), 0);
  mark_interior_moves(patch, nr, nz, move_r, move_z);
  double min_j = 1.0;
  if (!apply_reference_projection(state, cfg, patch, move_r, move_z, &min_j)) {
    pre.triggered = false;
    pre.converged = false;
    pre.residual = min_j;
    return pre;
  }
  return finish_projection(state, cfg, part, bufs, reduction, 1, min_j);
}

}  // namespace tenryu::hydro::ale
