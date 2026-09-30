#include "radiation/radiation_step.hpp"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "radiation/fld_1d_gpu.cuh"
#include "radiation/fld_2d_rz_gpu.cuh"
#include "radiation/groups.cuh"
#include "radiation/sn_transport_1d_gpu.cuh"
#include "radiation/sn_transport_2d_gpu.cuh"

namespace tenryu::radiation {
namespace {

[[nodiscard]] bool equal_double_bits(const std::vector<double>& lhs, const std::vector<double>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.empty() || std::memcmp(lhs.data(), rhs.data(), sizeof(double) * lhs.size()) == 0);
}

bool is_supported_2d_bc_outer(const std::string& mode) {
  return mode == "vacuum" || mode == "reflect";
}

bool is_supported_2d_bc_z(const std::string& mode) {
  return mode == "vacuum" || mode == "reflect" || mode == "marshak";
}

}  // namespace

void RadiationStep::set_last_overshoot_metrics(const std::int64_t count, const double max_ratio) {
  last_overshoot_count_ = count;
  last_overshoot_max_ = max_ratio;
}

void RadiationStep::step(core::State& state,
                         const core::Config& cfg,
                         const double dt,
                         const parallel::PartitionInfo& part,
                         parallel::CommBuffers* bufs,
                         const double drive_time_s) {
  TENRYU_ASSERT(bufs != nullptr || part.n_ranks <= 1,
                "RadiationStep::step requires CommBuffers when n_ranks > 1");
  last_overshoot_count_ = 0;
  last_overshoot_max_ = 0.0;
  if (!cfg.radiation.enabled || dt <= 0.0 || state.rho.empty()) {
    return;
  }
  if (cfg.materials.materials.empty()) {
    return;
  }
  if (state.mesh.dim == 2) {
    TENRYU_ASSERT(cfg.radiation.boundary.inner_r == "reflect",
                  "2D_RZ radiation boundary inner_r must be \"reflect\"");
    TENRYU_ASSERT(is_supported_2d_bc_outer(cfg.radiation.boundary.outer_r),
                  "2D_RZ radiation boundary outer_r must be \"vacuum\" or \"reflect\"");
    TENRYU_ASSERT(is_supported_2d_bc_z(cfg.radiation.boundary.bottom_z),
                  "2D_RZ radiation boundary bottom_z must be \"vacuum\", \"reflect\", or \"marshak\"");
    TENRYU_ASSERT(is_supported_2d_bc_z(cfg.radiation.boundary.top_z),
                  "2D_RZ radiation boundary top_z must be \"vacuum\", \"reflect\", or \"marshak\"");
  }

  TENRYU_ASSERT(state.rho.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "RadiationStep::step n_cells exceeds int range");
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(static_cast<int>(state.cell_is_void.size()) == n_cells,
                "RadiationStep::step requires cell_is_void size to match n_cells");
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const std::size_t n_cells_us = static_cast<std::size_t>(n_cells);
  const std::size_t n_groups_us = static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(n_cells == 0 || n_groups_us <= (std::numeric_limits<std::size_t>::max() / n_cells_us),
                "RadiationStep::step n_cells*n_groups overflow");
  TENRYU_ASSERT(n_cells_us * n_groups_us <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "RadiationStep::step n_cells*n_groups exceeds int range");

  const int mat_idx = cfg.materials.first_nonvoid_material_index();
  TENRYU_ASSERT(mat_idx >= 0, "RadiationStep::step requires at least one non-void material");
  const auto& mat = cfg.materials.materials[static_cast<std::size_t>(mat_idx)];

  std::vector<double> bounds = cfg.radiation.group_bounds_eV;
  if (bounds.empty() || static_cast<int>(bounds.size()) != n_groups + 1) {
    const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
    const double Tmin = std::max(range[0], 1.0e-3);
    const double Tmax = std::max(range[1], Tmin * 1.001);
    bounds = Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
  }
  Groups groups(bounds);
  const auto& planck_cfg = cfg.radiation.planck_fraction;
  if (planck_cfg.method == "tabulate") {
    core::log_warning("Radiation.groups.planck_fraction.method=\"tabulate\" is stored but not "
                      "implemented yet; falling back to computed Planck fractions");
  }
  const int planck_n_T = std::max(planck_cfg.compute_N_T, 2);
  const std::vector<double> planck_range = resolve_compute_T_range_eV(cfg, false);
  const double planck_T_min = planck_range[0];
  const double planck_T_max = planck_range[1];
  const int planck_n_groups = groups.num_groups();
  if (!planck_cache_valid_ || planck_cache_n_groups_ != planck_n_groups || planck_cache_n_T_ != planck_n_T ||
      planck_cache_T_min_eV_ != planck_T_min || planck_cache_T_max_eV_ != planck_T_max ||
      !equal_double_bits(planck_cache_bounds_eV_, bounds)) {
    planck_cache_.build(groups, planck_n_T, planck_T_min, planck_T_max);
    planck_cache_bounds_eV_ = bounds;
    planck_cache_n_groups_ = planck_n_groups;
    planck_cache_n_T_ = planck_n_T;
    planck_cache_T_min_eV_ = planck_T_min;
    planck_cache_T_max_eV_ = planck_T_max;
    planck_cache_valid_ = true;
  }
  const PlanckTable& planck = planck_cache_;

  if (cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion) {
    if (state.mesh.dim == 1) {
      advance_radiation_step_fld_1d(state, cfg, planck, mat, dt, drive_time_s);
      return;
    }
    advance_radiation_step_fld_2d_rz(state, cfg, planck, mat, dt, part, bufs);
    return;
  }
  TENRYU_ASSERT(cfg.radiation.mode == core::RadiationMode::SnTransport,
                "RadiationStep::step: Radiation.mode must be multigroup_diffusion or sn_transport");
  if (state.mesh.dim == 1) {
    advance_radiation_step_sn_1d(state, cfg, planck, mat, dt, drive_time_s);
    return;
  }
  advance_radiation_step_sn_2d_rz(state, cfg, planck, mat, dt, part, bufs);
}

}  // namespace tenryu::radiation
