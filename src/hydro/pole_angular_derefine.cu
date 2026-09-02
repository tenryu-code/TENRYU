#include "hydro/pole_angular_derefine.cuh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include "core/error.hpp"
#include "diagnostics/diagnostics.hpp"
#include "hydro/central_pseudo_core.cuh"
#include "hydro/conservation_audit.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "mesh/mesh.hpp"
#include "mesh/path_admissibility.cuh"

namespace tenryu::hydro::pole_angular_derefine {
namespace {

constexpr const char* kEnvEnable =
    "TENRYU_I1B_POLAR_SHELL_ANGULAR_DEREFINE";
constexpr double kPi =
    3.1415926535897932384626433832795028841971693993751;
constexpr double kChi = 1.0;
constexpr int kLevelMax = 3;
constexpr double kTracerPureTol = 1.0e-8;

inline void cuda_check(const cudaError_t err, const char* const message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline bool env_flag_enabled(const char* name) {
  const char* raw = std::getenv(name);
  return raw != nullptr && raw[0] != '\0' && raw[0] != '0';
}

inline bool is_supported_topology(const core::Config& cfg) {
  return cfg.main.dimension == "2D_RZ" &&
         mesh::mesh_topo_is_multiblock(cfg.mesh);
}

template <typename Tag>
std::vector<double> copy_field(const core::Field1D<Tag>& field) {
  std::vector<double> out;
  field.copy_to_host(out);
  return out;
}

int polar_shell_block_id(const mesh::MultiBlockTopology& mb) {
  int block_id = -1;
  for (int b = 0; b < static_cast<int>(mb.blocks.size()); ++b) {
    if (mb.blocks[static_cast<std::size_t>(b)].role ==
        mesh::BlockRole::POLAR_SHELL) {
      TENRYU_ASSERT(block_id < 0,
                    "polar shell angular de-refine found multiple shell blocks");
      block_id = b;
    }
  }
  return block_id;
}

int shell_cell_id(const mesh::BlockInfo& shell, const int i, const int j) {
  TENRYU_ASSERT(i >= 0 && i < shell.n_i_cells &&
                    j >= 0 && j < shell.n_j_cells,
                "polar shell angular de-refine cell index out of range");
  return shell.cell_begin + i * shell.n_j_cells + j;
}

int shell_node_id(const mesh::BlockInfo& shell, const int i, const int j) {
  TENRYU_ASSERT(i >= 0 && i <= shell.n_i_cells &&
                    j >= 0 && j <= shell.n_j_cells,
                "polar shell angular de-refine node index out of range");
  return shell.owned_node_begin + i * (shell.n_j_cells + 1) + j;
}

int shell_node_id(const core::PoleAngularDerefineState& pc,
                  const int i,
                  const int j) {
  TENRYU_ASSERT(i >= 0 && i <= pc.n_i_cells &&
                    j >= 0 && j <= pc.n_j_cells,
                "polar shell angular de-refine node index out of range");
  return pc.owned_node_begin + i * (pc.n_j_cells + 1) + j;
}

std::array<int, 4> multiblock_cell_nodes(const mesh::MultiBlockTopology& mb,
                                         const int c) {
  const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
  return {{
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 0)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 1)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 2)],
      mb.cell_node_csr_indices[static_cast<std::size_t>(off + 3)],
  }};
}

int active_nverts(const mesh::Mesh& mesh, const int c) {
  if (mesh.cell_nverts.size() == static_cast<std::size_t>(mesh.topo.n_cells)) {
    return mesh::mesh_topo_cell_active_nverts(mesh.cell_nverts, c);
  }
  return mesh::kMeshTopoCellStorageSlots;
}

double distance_node(const std::vector<double>& node_r,
                     const std::vector<double>& node_z,
                     const int a,
                     const int b) {
  const double dr = node_r[static_cast<std::size_t>(a)] -
                    node_r[static_cast<std::size_t>(b)];
  const double dz = node_z[static_cast<std::size_t>(a)] -
                    node_z[static_cast<std::size_t>(b)];
  return std::hypot(dr, dz);
}

double node_radius(const std::vector<double>& node_r,
                   const std::vector<double>& node_z,
                   const int n) {
  return std::hypot(node_r[static_cast<std::size_t>(n)],
                    node_z[static_cast<std::size_t>(n)]);
}

bool macro_material_pure(const mesh::BlockInfo& shell,
                         const std::vector<double>& tracer,
                         const int i,
                         const int j0,
                         const int j1) {
  if (tracer.empty()) {
    return true;
  }
  double min_y = std::numeric_limits<double>::infinity();
  double max_y = -std::numeric_limits<double>::infinity();
  for (int j = j0; j < j1; ++j) {
    const int c = shell_cell_id(shell, i, j);
    const double y = tracer[static_cast<std::size_t>(c)];
    min_y = std::min(min_y, y);
    max_y = std::max(max_y, y);
  }
  if (!std::isfinite(min_y) || !std::isfinite(max_y)) {
    return false;
  }
  return (max_y - min_y) <= kTracerPureTol || min_y >= 1.0 - kTracerPureTol ||
         max_y <= kTracerPureTol;
}

int span_for_level(const int level) {
  TENRYU_ASSERT(level >= 0 && level < 30,
                "polar shell angular de-refine level out of range");
  return 1 << level;
}

int level_for_span(const int span) {
  int level = 0;
  int value = 1;
  while (value < span) {
    value <<= 1;
    ++level;
  }
  return level;
}

int max_pole_span(const int n_j_cells) {
  return std::min(span_for_level(kLevelMax), std::max(1, n_j_cells / 2));
}

int dyadic_span_at_least(const int span, const int max_span) {
  if (max_span < 2 || span < 2) {
    return 0;
  }
  int value = 2;
  while (value < span && value < max_span) {
    value <<= 1;
  }
  return std::min(value, max_span);
}

double pole_span_ratio(const mesh::BlockInfo& shell,
                       const std::vector<double>& node_r,
                       const std::vector<double>& node_z,
                       const int i,
                       const bool low_pole,
                       const int span) {
  const int j0 = low_pole ? 0 : shell.n_j_cells - span;
  const int j1 = low_pole ? span : shell.n_j_cells;
  const int n00 = shell_node_id(shell, i, j0);
  const int n01 = shell_node_id(shell, i, j1);
  const int n10 = shell_node_id(shell, i + 1, j0);
  const int n11 = shell_node_id(shell, i + 1, j1);
  const double r_mean =
      0.25 * (node_radius(node_r, node_z, n00) +
              node_radius(node_r, node_z, n01) +
              node_radius(node_r, node_z, n10) +
              node_radius(node_r, node_z, n11));
  const double dr =
      0.5 * (distance_node(node_r, node_z, n00, n10) +
             distance_node(node_r, node_z, n01, n11));
  const double dtheta =
      kPi * static_cast<double>(span) /
      static_cast<double>(std::max(1, shell.n_j_cells));
  if (!(dr > 0.0) || !std::isfinite(dr)) {
    return std::numeric_limits<double>::infinity();
  }
  return (r_mean * dtheta) / (kChi * dr);
}

int select_pole_span(const mesh::BlockInfo& shell,
                     const std::vector<double>& node_r,
                     const std::vector<double>& node_z,
                     const int i,
                     const bool low_pole) {
  const int max_span = max_pole_span(shell.n_j_cells);
  for (int trial = 2; trial <= max_span; trial <<= 1) {
    if (pole_span_ratio(shell, node_r, node_z, i, low_pole, trial) >= 1.0) {
      return trial;
    }
  }
  return max_span >= 2 ? max_span : 0;
}

double boundary_loop_volume(const core::PoleAngularDerefineMacroState& macro,
                            const std::vector<double>& node_r,
                            const std::vector<double>& node_z) {
  const auto& boundary_nodes = macro.boundary_nodes_ordered;
  const int n_boundary = static_cast<int>(boundary_nodes.size());
  TENRYU_ASSERT(n_boundary >= 3,
                "polar shell angular de-refine boundary needs >=3 nodes");
  std::vector<double> r(static_cast<std::size_t>(n_boundary));
  std::vector<double> z(static_cast<std::size_t>(n_boundary));
  for (int k = 0; k < n_boundary; ++k) {
    const int n = boundary_nodes[static_cast<std::size_t>(k)];
    TENRYU_ASSERT(n >= 0 && static_cast<std::size_t>(n) < node_r.size(),
                  "polar shell angular de-refine boundary node out of range");
    r[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
    z[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
  }
  const double volume = rz::rz_polygon_volume_exact(r.data(), z.data(),
                                                    n_boundary);
  TENRYU_ASSERT(std::isfinite(volume) && volume > 0.0,
                "polar shell angular de-refine macro volume must be positive");
  return volume;
}

void orient_macro_boundary_positive(core::PoleAngularDerefineMacroState& macro,
                                    const std::vector<double>& node_r,
                                    const std::vector<double>& node_z) {
  const int n_boundary =
      static_cast<int>(macro.boundary_nodes_ordered.size());
  std::vector<double> r(static_cast<std::size_t>(n_boundary));
  std::vector<double> z(static_cast<std::size_t>(n_boundary));
  for (int k = 0; k < n_boundary; ++k) {
    const int n = macro.boundary_nodes_ordered[static_cast<std::size_t>(k)];
    r[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
    z[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
  }
  const double volume = rz::rz_polygon_volume_exact(r.data(), z.data(),
                                                    n_boundary);
  TENRYU_ASSERT(std::isfinite(volume) && volume != 0.0,
                "polar shell angular de-refine macro boundary is degenerate");
  if (volume < 0.0) {
    std::reverse(macro.boundary_nodes_ordered.begin(),
                 macro.boundary_nodes_ordered.end());
  }
}

void initialize_scatter_weights(core::PoleAngularDerefineMacroState& macro,
                                const std::vector<double>& mass) {
  const std::size_t n_members = macro.member_cells.size();
  TENRYU_ASSERT(n_members > 0U,
                "polar shell angular de-refine scatter needs member cells");
  if (macro.member_scatter_weights.size() == n_members) {
    return;
  }
  long double raw_sum = 0.0L;
  std::vector<long double> raw(n_members, 0.0L);
  bool mass_ok = true;
  for (std::size_t i = 0; i < n_members; ++i) {
    const int c = macro.member_cells[i];
    const double m = mass[static_cast<std::size_t>(c)];
    if (!(std::isfinite(m) && m > 0.0)) {
      mass_ok = false;
      break;
    }
    raw[i] = static_cast<long double>(m);
    raw_sum += raw[i];
  }
  if (!mass_ok || !(raw_sum > 0.0L)) {
    raw_sum = static_cast<long double>(n_members);
    std::fill(raw.begin(), raw.end(), 1.0L);
  }
  macro.member_scatter_weights.assign(n_members, 0.0);
  long double prefix = 0.0L;
  for (std::size_t i = 0; i + 1U < n_members; ++i) {
    macro.member_scatter_weights[i] =
        static_cast<double>(raw[i] / raw_sum);
    prefix += static_cast<long double>(macro.member_scatter_weights[i]);
  }
  macro.member_scatter_weights[n_members - 1U] =
      static_cast<double>(1.0L - prefix);
  if (!(macro.member_scatter_weights[n_members - 1U] > 0.0 &&
        std::isfinite(macro.member_scatter_weights[n_members - 1U]))) {
    const double uniform = 1.0 / static_cast<double>(n_members);
    std::fill(macro.member_scatter_weights.begin(),
              macro.member_scatter_weights.end(),
              uniform);
  }
}

double electron_energy_fraction(const double Ue,
                                const double Ui,
                                const bool has_ion_energy) {
  if (!has_ion_energy) {
    return 1.0;
  }
  const double U_total = Ue + Ui;
  return (std::isfinite(U_total) && U_total > 0.0) ? Ue / U_total : 0.5;
}

struct MemberSums {
  long double M = 0.0L;
  long double MY = 0.0L;
  long double Ue = 0.0L;
  long double Ui = 0.0L;
  long double V = 0.0L;
};

MemberSums sum_member_state(const core::PoleAngularDerefineMacroState& macro,
                            const std::vector<double>& mass,
                            const std::vector<double>& vol,
                            const std::vector<double>& ee,
                            const std::vector<double>& ei,
                            const std::vector<double>& tracer) {
  MemberSums sums{};
  for (const int c : macro.member_cells) {
    const std::size_t idx = static_cast<std::size_t>(c);
    sums.M += static_cast<long double>(mass[idx]);
    sums.V += static_cast<long double>(vol[idx]);
    sums.Ue += static_cast<long double>(mass[idx]) *
               static_cast<long double>(ee[idx]);
    if (!ei.empty()) {
      sums.Ui += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(ei[idx]);
    }
    if (!tracer.empty()) {
      sums.MY += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(tracer[idx]);
    }
  }
  return sums;
}

std::vector<double> member_corner_mass_host(const core::State& state) {
  const int n_cells = state.mesh.topo.n_cells;
  const std::size_t expected = static_cast<std::size_t>(n_cells) * 4U;
  TENRYU_ASSERT(state.corner_mass.size() == expected,
                "polar shell angular de-refine requires state.corner_mass basis");
  return copy_field(state.corner_mass);
}

void write_macro_mirrors(core::State& state,
                         const core::PoleAngularDerefineMacroState& macro,
                         std::vector<double>& mass,
                         std::vector<double>& vol,
                         std::vector<double>& rho,
                         std::vector<double>& ee,
                         std::vector<double>& ei,
                         std::vector<double>& tracer) {
  TENRYU_ASSERT(macro.M_c > 0.0 && macro.V_c > 0.0,
                "polar shell angular de-refine mirror needs positive macro state");
  TENRYU_ASSERT(macro.member_scatter_weights.size() == macro.member_cells.size(),
                "polar shell angular de-refine scatter weights not initialized");
  const double rho_c = macro.M_c / macro.V_c;
  long double m_prefix = 0.0L;
  long double v_prefix = 0.0L;
  long double ue_prefix = 0.0L;
  long double ui_prefix = 0.0L;
  long double my_prefix = 0.0L;
  for (std::size_t k = 0; k < macro.member_cells.size(); ++k) {
    const int c = macro.member_cells[k];
    const std::size_t idx = static_cast<std::size_t>(c);
    const bool last = k + 1U == macro.member_cells.size();
    const long double w =
        static_cast<long double>(macro.member_scatter_weights[k]);
    const long double m_i =
        last ? static_cast<long double>(macro.M_c) - m_prefix
             : w * static_cast<long double>(macro.M_c);
    const long double V_i =
        last ? static_cast<long double>(macro.V_c) - v_prefix
             : w * static_cast<long double>(macro.V_c);
    const long double Ue_i =
        last ? static_cast<long double>(macro.Ue_c) - ue_prefix
             : w * static_cast<long double>(macro.Ue_c);
    const long double Ui_i =
        last ? static_cast<long double>(macro.Ui_c) - ui_prefix
             : w * static_cast<long double>(macro.Ui_c);
    const long double MY_i =
        last ? static_cast<long double>(macro.M_Y_c) - my_prefix
             : w * static_cast<long double>(macro.M_Y_c);
    TENRYU_ASSERT(m_i > 0.0L && V_i > 0.0L,
                  "polar shell angular de-refine mirror produced bad storage");
    mass[idx] = static_cast<double>(m_i);
    vol[idx] = static_cast<double>(V_i);
    rho[idx] = rho_c;
    ee[idx] = static_cast<double>(Ue_i / m_i);
    if (!ei.empty()) {
      ei[idx] = static_cast<double>(Ui_i / m_i);
    }
    if (!tracer.empty()) {
      tracer[idx] = static_cast<double>(MY_i / m_i);
    }
    m_prefix += static_cast<long double>(mass[idx]);
    v_prefix += static_cast<long double>(vol[idx]);
    ue_prefix += static_cast<long double>(mass[idx]) *
                 static_cast<long double>(ee[idx]);
    if (!ei.empty()) {
      ui_prefix += static_cast<long double>(mass[idx]) *
                   static_cast<long double>(ei[idx]);
    }
    if (!tracer.empty()) {
      my_prefix += static_cast<long double>(mass[idx]) *
                   static_cast<long double>(tracer[idx]);
    }
  }
}

void upload_overlay(core::State& state) {
  auto& pc = state.pole_angular_derefine;
  pc.d_member_mask.reset(pc.member_mask.size());
  pc.d_member_mask.copy_from_host(pc.member_mask);
  pc.d_inactive_member_mask.reset(pc.inactive_member_mask.size());
  pc.d_inactive_member_mask.copy_from_host(pc.inactive_member_mask);
  pc.d_boundary_node_mask.reset(pc.boundary_node_mask.size());
  pc.d_boundary_node_mask.copy_from_host(pc.boundary_node_mask);
  pc.d_boundary_nodes_flat.reset(pc.boundary_nodes_flat.size());
  pc.d_boundary_nodes_flat.copy_from_host(pc.boundary_nodes_flat);
  pc.d_boundary_node_offsets.reset(pc.boundary_node_offsets.size());
  pc.d_boundary_node_offsets.copy_from_host(pc.boundary_node_offsets);
  pc.d_boundary_impulse_r.reset(pc.boundary_nodes_flat.size());
  pc.d_boundary_impulse_z.reset(pc.boundary_nodes_flat.size());
}

core::PoleAngularDerefineMacroState make_macro(
    const mesh::BlockInfo& shell,
    const int block_id,
    const int i,
    const int j0,
    const int j1,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  core::PoleAngularDerefineMacroState macro;
  macro.block_id = block_id;
  macro.local_i_begin = i;
  macro.local_i_end = i + 1;
  macro.local_j_begin = j0;
  macro.local_j_end = j1;
  macro.span = j1 - j0;
  macro.level = level_for_span(macro.span);
  macro.skipped_nodes = 2 * std::max(0, macro.span - 1);
  macro.representative_cell = shell_cell_id(shell, i, j0);
  for (int j = j0; j < j1; ++j) {
    macro.member_cells.push_back(shell_cell_id(shell, i, j));
  }
  macro.boundary_nodes_ordered = {
      shell_node_id(shell, i, j0),
      shell_node_id(shell, i, j1),
      shell_node_id(shell, i + 1, j1),
      shell_node_id(shell, i + 1, j0),
  };
  orient_macro_boundary_positive(macro, node_r, node_z);
  return macro;
}

bool cells_available(const core::PoleAngularDerefineMacroState& macro,
                     const std::vector<std::uint8_t>& mask) {
  for (const int c : macro.member_cells) {
    if (mask[static_cast<std::size_t>(c)] != 0U) {
      return false;
    }
  }
  return true;
}

void mark_macro(core::PoleAngularDerefineState& pc,
                const core::PoleAngularDerefineMacroState& macro) {
  for (const int c : macro.member_cells) {
    pc.member_mask[static_cast<std::size_t>(c)] = 1U;
    pc.inactive_member_mask[static_cast<std::size_t>(c)] = 1U;
  }
  for (const int n : macro.boundary_nodes_ordered) {
    pc.boundary_node_mask[static_cast<std::size_t>(n)] = 1U;
    pc.boundary_nodes_flat.push_back(n);
  }
}

bool same_pole_side(const core::PoleAngularDerefineState& pc,
                    const core::PoleAngularDerefineMacroState& a,
                    const int j_begin,
                    const int j_end) {
  const bool low = j_begin == 0;
  const bool high = j_end == pc.n_j_cells;
  if (!(low || high)) {
    return false;
  }
  return low ? a.local_j_begin == 0 : a.local_j_end == pc.n_j_cells;
}

bool intervals_overlap(const int a0, const int a1, const int b0, const int b1) {
  return std::max(a0, b0) < std::min(a1, b1);
}

void clear_boundary_work(core::PoleAngularDerefineMacroState& macro) {
  macro.boundary_work_pending = false;
  macro.boundary_work_valid = false;
  macro.boundary_work_dU = 0.0;
  macro.boundary_work_chi_e = 1.0;
  macro.boundary_work_W_old = 0.0;
  macro.boundary_work_W_new = 0.0;
  macro.boundary_work_W_mid = 0.0;
  macro.boundary_work_residual = 0.0;
}

void refresh_geometry_exempt_cells_impl(core::State& state) {
  const int n_cells = state.mesh.topo.n_cells;
  if (n_cells <= 0) {
    state.mesh.geometry_exempt_cells.clear();
    return;
  }
  std::vector<std::uint8_t> exempt(static_cast<std::size_t>(n_cells), 0U);
  bool any = false;
  const auto apply_inactive = [&](const std::vector<std::uint8_t>& mask) {
    if (mask.size() != static_cast<std::size_t>(n_cells)) {
      return;
    }
    for (int c = 0; c < n_cells; ++c) {
      if (mask[static_cast<std::size_t>(c)] != 0U) {
        exempt[static_cast<std::size_t>(c)] = 1U;
        any = true;
      }
    }
  };
  apply_inactive(state.central_pseudo_core.inactive_member_mask);
  apply_inactive(state.pole_angular_derefine.inactive_member_mask);
  if (any) {
    state.mesh.geometry_exempt_cells = std::move(exempt);
  } else {
    state.mesh.geometry_exempt_cells.clear();
  }
}

void rebuild_overlay_masks(core::State& state) {
  auto& pc = state.pole_angular_derefine;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  pc.member_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  pc.inactive_member_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  pc.boundary_node_mask.assign(static_cast<std::size_t>(n_nodes), 0U);
  pc.boundary_nodes_flat.clear();
  pc.boundary_node_offsets.clear();
  pc.boundary_node_offsets.push_back(0);
  std::sort(pc.macros.begin(),
            pc.macros.end(),
            [](const auto& a, const auto& b) {
              if (a.local_i_begin != b.local_i_begin) {
                return a.local_i_begin < b.local_i_begin;
              }
              return a.local_j_begin < b.local_j_begin;
            });
  for (auto& macro : pc.macros) {
    clear_boundary_work(macro);
    mark_macro(pc, macro);
    pc.boundary_node_offsets.push_back(
        static_cast<int>(pc.boundary_nodes_flat.size()));
  }
  pc.macro_count = static_cast<int>(pc.macros.size());
  pc.valid = !pc.macros.empty();
  pc.representative_cell =
      pc.valid ? pc.macros.front().representative_cell : -1;
  refresh_geometry_exempt_cells_impl(state);
}

void assert_macro_mask_coverage(const core::State& state) {
  if (!assertions_enabled()) {
    return;
  }
  const auto& pc = state.pole_angular_derefine;
  if (!pc.valid || pc.macros.empty()) {
    return;
  }
  const int n_cells = state.mesh.topo.n_cells;
  TENRYU_ASSERT(pc.member_mask.size() == static_cast<std::size_t>(n_cells) &&
                    pc.inactive_member_mask.size() ==
                        static_cast<std::size_t>(n_cells),
                "polar shell angular de-refine mask size mismatch");
  TENRYU_ASSERT(state.mesh.geometry_exempt_cells.size() ==
                    static_cast<std::size_t>(n_cells),
                "polar shell angular de-refine geometry exemption size mismatch");
  for (const auto& macro : pc.macros) {
    TENRYU_ASSERT(macro.local_i_end == macro.local_i_begin + 1,
                  "polar shell angular de-refine macro must span one q row");
    const int span = macro.local_j_end - macro.local_j_begin;
    TENRYU_ASSERT(span == macro.span &&
                      static_cast<int>(macro.member_cells.size()) == span,
                  "polar shell angular de-refine macro member span mismatch");
    for (int j = macro.local_j_begin; j < macro.local_j_end; ++j) {
      const int c = pc.cell_begin + macro.local_i_begin * pc.n_j_cells + j;
      TENRYU_ASSERT(c >= 0 && c < n_cells,
                    "polar shell angular de-refine member cell out of range");
      const auto idx = static_cast<std::size_t>(c);
      TENRYU_ASSERT(pc.member_mask[idx] != 0U &&
                        pc.inactive_member_mask[idx] != 0U &&
                        state.mesh.geometry_exempt_cells[idx] != 0U,
                    "polar shell angular de-refine macro member mask gap");
      TENRYU_ASSERT(std::find(macro.member_cells.begin(),
                              macro.member_cells.end(),
                              c) != macro.member_cells.end(),
                    "polar shell angular de-refine macro member list gap");
    }
  }
}

void sync_overlay_masks_after_mutation(core::State& state) {
  rebuild_overlay_masks(state);
  assert_macro_mask_coverage(state);
  upload_overlay(state);
}

bool canonicalize_short_pole_edge(
    core::PoleAngularDerefineMacroState& macro,
    const int n_j_cells,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  if (macro.boundary_nodes_ordered.size() <= 3U ||
      !(macro.local_j_begin == 0 || macro.local_j_end == n_j_cells)) {
    return false;
  }
  double axis_scale = 0.0;
  for (std::size_t k = 0; k < macro.boundary_nodes_ordered.size(); ++k) {
    const int a = macro.boundary_nodes_ordered[k];
    const int b =
        macro.boundary_nodes_ordered[(k + 1U) %
                                     macro.boundary_nodes_ordered.size()];
    axis_scale =
        std::max(axis_scale, std::abs(node_r[static_cast<std::size_t>(a)]));
    axis_scale =
        std::max(axis_scale, std::abs(node_z[static_cast<std::size_t>(a)]));
    axis_scale = std::max(axis_scale, distance_node(node_r, node_z, a, b));
  }
  const double axis_eps =
      std::max(1.0e-12, 1.0e-9 * std::max(axis_scale, 1.0e-30));
  int finite_axis_edges = 0;
  for (std::size_t k = 0; k < macro.boundary_nodes_ordered.size(); ++k) {
    const int a = macro.boundary_nodes_ordered[k];
    const int b =
        macro.boundary_nodes_ordered[(k + 1U) %
                                     macro.boundary_nodes_ordered.size()];
    if (std::abs(node_r[static_cast<std::size_t>(a)]) <= 10.0 * axis_eps &&
        std::abs(node_r[static_cast<std::size_t>(b)]) <= 10.0 * axis_eps &&
        distance_node(node_r, node_z, a, b) > axis_eps) {
      ++finite_axis_edges;
    }
  }
  const bool preserve_axis_edge =
      macro.local_i_end == macro.local_i_begin + 1 && finite_axis_edges == 1;
  if (!preserve_axis_edge) {
    const auto alias =
        tenryu::mesh::path_admissibility_detail::canonicalize_loop_node_aliases(
            node_r,
            node_z,
            macro.boundary_nodes_ordered,
            macro.single_apex_boundary ? macro.canonical_apex_node : -1,
            macro.single_apex_boundary);
    if (alias.loop_nodes.size() >= 3U &&
        alias.loop_nodes != macro.boundary_nodes_ordered) {
      macro.boundary_nodes_ordered = alias.loop_nodes;
      orient_macro_boundary_positive(macro, node_r, node_z);
      return true;
    }
  }
  double scale = 0.0;
  for (std::size_t k = 0; k < macro.boundary_nodes_ordered.size(); ++k) {
    const int a = macro.boundary_nodes_ordered[k];
    const int b =
        macro.boundary_nodes_ordered[(k + 1U) %
                                     macro.boundary_nodes_ordered.size()];
    scale = std::max(scale, node_radius(node_r, node_z, a));
    scale = std::max(scale, node_radius(node_r, node_z, b));
    scale = std::max(scale, distance_node(node_r, node_z, a, b));
  }
  const double eps = std::max(1.0e-12, 1.0e-9 * std::max(scale, 1.0e-30));
  for (std::size_t k = 0; k < macro.boundary_nodes_ordered.size(); ++k) {
    const std::size_t next =
        (k + 1U) % macro.boundary_nodes_ordered.size();
    const int a = macro.boundary_nodes_ordered[k];
    const int b = macro.boundary_nodes_ordered[next];
    if (distance_node(node_r, node_z, a, b) <= eps) {
      macro.boundary_nodes_ordered.erase(
          macro.boundary_nodes_ordered.begin() +
          static_cast<std::ptrdiff_t>(next));
      orient_macro_boundary_positive(macro, node_r, node_z);
      return true;
    }
  }
  return false;
}

struct KineticProjectionResult {
  double K_old = 0.0;
  double K_new = 0.0;
  double dU = 0.0;
  double chi_e = 1.0;
};

KineticProjectionResult project_macro_kinetic_energy(
    core::State& state,
    const std::vector<int>& macro_indices) {
  KineticProjectionResult result{};
  auto& pc = state.pole_angular_derefine;
  if (macro_indices.empty() || pc.macros.empty()) {
    return result;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "polar shell angular de-refine KE projection requires multiblock");
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = static_cast<int>(state.v_r.size());
  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> ee = copy_field(state.ee);
  std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                            : copy_field(state.ei);
  std::vector<double> v_r = copy_field(state.v_r);
  std::vector<double> v_z = copy_field(state.v_z);
  const std::vector<double> corner_mass = member_corner_mass_host(state);

  std::vector<long double> total_m(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> total_pr(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> total_pz(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> member_m(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> member_pr(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> member_pz(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> proj_m(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> proj_pr(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<long double> proj_pz(static_cast<std::size_t>(n_nodes), 0.0L);
  std::vector<std::uint8_t> affected(static_cast<std::size_t>(n_nodes), 0U);

  for (int c = 0; c < n_cells; ++c) {
    const std::array<int, 4> nodes = multiblock_cell_nodes(mb, c);
    const int nverts = active_nverts(state.mesh, c);
    for (int k = 0; k < nverts; ++k) {
      const int n = nodes[static_cast<std::size_t>(k)];
      const long double cm = static_cast<long double>(
          corner_mass[static_cast<std::size_t>(4 * c + k)]);
      total_m[static_cast<std::size_t>(n)] += cm;
      total_pr[static_cast<std::size_t>(n)] +=
          cm * static_cast<long double>(v_r[static_cast<std::size_t>(n)]);
      total_pz[static_cast<std::size_t>(n)] +=
          cm * static_cast<long double>(v_z[static_cast<std::size_t>(n)]);
    }
  }

  long double Ue_members = 0.0L;
  long double Ui_members = 0.0L;
  long double M_members = 0.0L;
  for (const int macro_index : macro_indices) {
    TENRYU_ASSERT(macro_index >= 0 &&
                      macro_index < static_cast<int>(pc.macros.size()),
                  "polar shell angular de-refine KE macro index out of range");
    const auto& macro = pc.macros[static_cast<std::size_t>(macro_index)];
    for (const int c : macro.member_cells) {
      const std::array<int, 4> nodes = multiblock_cell_nodes(mb, c);
      const int nverts = active_nverts(state.mesh, c);
      const double cell_m = mass[static_cast<std::size_t>(c)];
      M_members += static_cast<long double>(cell_m);
      Ue_members += static_cast<long double>(cell_m) *
                    static_cast<long double>(ee[static_cast<std::size_t>(c)]);
      if (!ei.empty()) {
        Ui_members += static_cast<long double>(cell_m) *
                      static_cast<long double>(
                          ei[static_cast<std::size_t>(c)]);
      }
      for (int k = 0; k < nverts; ++k) {
        const int n = nodes[static_cast<std::size_t>(k)];
        const std::size_t n_u = static_cast<std::size_t>(n);
        const long double cm = static_cast<long double>(
            corner_mass[static_cast<std::size_t>(4 * c + k)]);
        member_m[n_u] += cm;
        member_pr[n_u] += cm * static_cast<long double>(v_r[n_u]);
        member_pz[n_u] += cm * static_cast<long double>(v_z[n_u]);
        affected[n_u] = 1U;

        const int rel = n - pc.owned_node_begin;
        TENRYU_ASSERT(rel >= 0,
                      "polar shell angular de-refine member node outside shell");
        const int stride = pc.n_j_cells + 1;
        const int ii = rel / stride;
        const int jj = rel - ii * stride;
        TENRYU_ASSERT(ii == macro.local_i_begin || ii == macro.local_i_end,
                      "polar shell angular de-refine member node not on radial side");
        TENRYU_ASSERT(jj >= macro.local_j_begin && jj <= macro.local_j_end,
                      "polar shell angular de-refine member node outside macro span");
        const double s =
            static_cast<double>(jj - macro.local_j_begin) /
            static_cast<double>(macro.local_j_end - macro.local_j_begin);
        const int n0 = shell_node_id(pc, ii, macro.local_j_begin);
        const int n1 = shell_node_id(pc, ii, macro.local_j_end);
        const long double w0 = static_cast<long double>(1.0 - s);
        const long double w1 = static_cast<long double>(s);
        const long double pr = cm * static_cast<long double>(v_r[n_u]);
        const long double pz = cm * static_cast<long double>(v_z[n_u]);
        proj_m[static_cast<std::size_t>(n0)] += w0 * cm;
        proj_pr[static_cast<std::size_t>(n0)] += w0 * pr;
        proj_pz[static_cast<std::size_t>(n0)] += w0 * pz;
        proj_m[static_cast<std::size_t>(n1)] += w1 * cm;
        proj_pr[static_cast<std::size_t>(n1)] += w1 * pr;
        proj_pz[static_cast<std::size_t>(n1)] += w1 * pz;
        affected[static_cast<std::size_t>(n0)] = 1U;
        affected[static_cast<std::size_t>(n1)] = 1U;
      }
    }
  }

  long double K_old = 0.0L;
  long double K_new = 0.0L;
  for (int n = 0; n < n_nodes; ++n) {
    const std::size_t idx = static_cast<std::size_t>(n);
    if (affected[idx] == 0U) {
      continue;
    }
    const long double old_m = total_m[idx];
    const long double old_pr = total_pr[idx];
    const long double old_pz = total_pz[idx];
    if (old_m > 0.0L) {
      K_old += 0.5L * (old_pr * old_pr + old_pz * old_pz) / old_m;
    }
    const long double new_m = old_m - member_m[idx] + proj_m[idx];
    const long double new_pr = old_pr - member_pr[idx] + proj_pr[idx];
    const long double new_pz = old_pz - member_pz[idx] + proj_pz[idx];
    if (new_m > 0.0L) {
      v_r[idx] = static_cast<double>(new_pr / new_m);
      v_z[idx] = static_cast<double>(new_pz / new_m);
      K_new += 0.5L * (new_pr * new_pr + new_pz * new_pz) / new_m;
    } else {
      v_r[idx] = 0.0;
      v_z[idx] = 0.0;
    }
  }

  double dU = static_cast<double>(K_old - K_new);
  const double scale =
      std::max({std::abs(static_cast<double>(K_old)),
                std::abs(static_cast<double>(K_new)),
                1.0});
  if (dU < 0.0 && std::abs(dU) <= 1.0e-12 * scale) {
    dU = 0.0;
  }
  TENRYU_ASSERT(std::isfinite(dU) && dU >= 0.0,
                "polar shell angular de-refine activation KE closure went negative");
  const double chi_e =
      electron_energy_fraction(static_cast<double>(Ue_members),
                               static_cast<double>(Ui_members),
                               !ei.empty());
  if (dU > 0.0 && M_members > 0.0L) {
    for (const int macro_index : macro_indices) {
      const auto& macro = pc.macros[static_cast<std::size_t>(macro_index)];
      for (const int c : macro.member_cells) {
        const std::size_t idx = static_cast<std::size_t>(c);
        const double w =
            mass[idx] / static_cast<double>(M_members);
        const double dU_i = dU * w;
        ee[idx] += chi_e * dU_i / mass[idx];
        if (!ei.empty()) {
          ei[idx] += (1.0 - chi_e) * dU_i / mass[idx];
        }
      }
    }
  }
  state.v_r.copy_from_host(v_r);
  state.v_z.copy_from_host(v_z);
  state.ee.copy_from_host(ee);
  if (!ei.empty()) {
    state.ei.copy_from_host(ei);
  }
  result.K_old = static_cast<double>(K_old);
  result.K_new = static_cast<double>(K_new);
  result.dU = dU;
  result.chi_e = chi_e;
  return result;
}

void project_activation_kinetic_energy(core::State& state,
                                       const core::Config& cfg) {
  (void)cfg;
  auto& pc = state.pole_angular_derefine;
  if (pc.activation_ke_valid || pc.macros.empty()) {
    return;
  }
  std::vector<int> macro_indices;
  macro_indices.reserve(pc.macros.size());
  for (int m = 0; m < static_cast<int>(pc.macros.size()); ++m) {
    macro_indices.push_back(m);
  }
  const KineticProjectionResult result =
      project_macro_kinetic_energy(state, macro_indices);
  pc.activation_ke_K_old = result.K_old;
  pc.activation_ke_K_new = result.K_new;
  pc.activation_ke_dU = result.dU;
  pc.activation_ke_chi_e = result.chi_e;
  pc.activation_ke_valid = true;
}

struct ScalarTotals {
  double mass = 0.0;
  double tracer_mass = 0.0;
  double Ue = 0.0;
  double Ui = 0.0;
  double E_total = 0.0;
};

ScalarTotals scalar_totals(const core::State& state) {
  ScalarTotals totals{};
  const std::vector<double> mass = copy_field(state.mass);
  const std::vector<double> ee = copy_field(state.ee);
  const std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                                  : copy_field(state.ei);
  const std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  for (std::size_t c = 0; c < mass.size(); ++c) {
    totals.mass += mass[c];
    totals.Ue += mass[c] * ee[c];
    if (!ei.empty()) {
      totals.Ui += mass[c] * ei[c];
    }
    if (!tracer.empty()) {
      totals.tracer_mass += mass[c] * tracer[c];
    }
  }
  totals.E_total = diagnostics::compute_energy_budget_2d(state).E_total;
  return totals;
}

double relative_delta(const double value, const double ref) {
  return (value - ref) / std::max(std::abs(ref), 1.0e-300);
}

bool replace_pole_macro_span(core::State& state,
                             const core::Config& cfg,
                             const mesh::BlockInfo& shell,
                             const int q,
                             const bool low_pole,
                             const int requested_span,
                             const char* aggregate_phase,
                             const char* log_tag) {
  auto& pc = state.pole_angular_derefine;
  const int max_span = max_pole_span(pc.n_j_cells);
  const int target_span = std::min(max_span, requested_span);
  if (target_span < 2) {
    return false;
  }

  std::size_t source_index = pc.macros.size();
  for (std::size_t m = 0; m < pc.macros.size(); ++m) {
    const auto& candidate = pc.macros[m];
    if (candidate.local_i_begin == q &&
        ((low_pole && candidate.local_j_begin == 0) ||
         (!low_pole && candidate.local_j_end == pc.n_j_cells))) {
      source_index = m;
      break;
    }
  }
  if (source_index == pc.macros.size()) {
    return false;
  }

  const auto source = pc.macros[source_index];
  const int old_span = source.local_j_end - source.local_j_begin;
  if (target_span <= old_span || old_span < 2) {
    return false;
  }
  const int new_j_begin = low_pole ? 0 : pc.n_j_cells - target_span;
  const int new_j_end = low_pole ? target_span : pc.n_j_cells;
  if (new_j_begin < 0 || new_j_end > pc.n_j_cells ||
      new_j_end <= new_j_begin) {
    return false;
  }

  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  const std::vector<double> mass = copy_field(state.mass);
  const std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  if (!macro_material_pure(shell, tracer, q, new_j_begin, new_j_end)) {
    return false;
  }

  auto replacement =
      make_macro(shell, pc.block_id, q, new_j_begin, new_j_end, node_r, node_z);
  initialize_scatter_weights(replacement, mass);

  std::vector<core::PoleAngularDerefineMacroState> grown;
  grown.reserve(pc.macros.size());
  std::vector<std::uint8_t> used(
      static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
  bool removed = false;
  for (const auto& existing : pc.macros) {
    const bool replace_this =
        existing.local_i_begin == q &&
        same_pole_side(pc, existing, source.local_j_begin, source.local_j_end) &&
        intervals_overlap(existing.local_j_begin,
                          existing.local_j_end,
                          new_j_begin,
                          new_j_end);
    if (replace_this) {
      removed = true;
      continue;
    }
    for (const int c : existing.member_cells) {
      if (used[static_cast<std::size_t>(c)] != 0U) {
        return false;
      }
      used[static_cast<std::size_t>(c)] = 1U;
    }
    grown.push_back(existing);
  }
  if (!removed) {
    return false;
  }
  for (const int c : replacement.member_cells) {
    if (used[static_cast<std::size_t>(c)] != 0U) {
      return false;
    }
    used[static_cast<std::size_t>(c)] = 1U;
  }

  const ScalarTotals before = scalar_totals(state);
  const int replacement_index = static_cast<int>(grown.size());
  grown.push_back(std::move(replacement));
  pc.macros = std::move(grown);
  const KineticProjectionResult ke =
      project_macro_kinetic_energy(state, std::vector<int>{replacement_index});
  sync_overlay_masks_after_mutation(state);
  aggregate_state(state,
                  cfg,
                  aggregate_phase != nullptr ? aggregate_phase : "span_growth",
                  false);
  const ScalarTotals after = scalar_totals(state);
  const double dM_growth = after.mass - before.mass;
  const double dMY_growth = after.tracer_mass - before.tracer_mass;
  const double mass_tol =
      std::max(1.0e-30,
               64.0 * std::numeric_limits<double>::epsilon() *
                   std::max(std::abs(before.mass), 1.0e-300));
  TENRYU_ASSERT(std::isfinite(dM_growth) &&
                    std::abs(dM_growth) <= mass_tol,
                "polar shell angular de-refine span growth changed mass");

  const double ratio =
      pole_span_ratio(shell, node_r, node_z, q, low_pole, target_span);
  char message[1024];
  std::snprintf(message,
                sizeof(message),
                "[%s] step=%d t=%.17e phase=%s pole=%s q=%d "
                "span_old=%d span_new=%d j_range=%d:%d members=%d "
                "r_dtheta_over_chidr=%.17e dM_growth=%.17e "
                "dMY_growth=%.17e dU=%.17e K_old=%.17e K_new=%.17e",
                log_tag != nullptr ? log_tag
                                   : "pole_angular_derefine_span_growth",
                state.step,
                state.t,
                aggregate_phase != nullptr ? aggregate_phase : "",
                low_pole ? "low" : "high",
                q,
                old_span,
                target_span,
                new_j_begin,
                new_j_end,
                target_span,
                ratio,
                dM_growth,
                dMY_growth,
                ke.dU,
                ke.K_old,
                ke.K_new);
  core::log_warning(message);

  const int span_warn_bound = std::max(1, pc.n_j_cells / 4);
  if (target_span > span_warn_bound) {
    char warn[512];
    std::snprintf(warn,
                  sizeof(warn),
                  "[pole_angular_derefine_ball_dominance_warn] pole=%s q=%d "
                  "span=%d bound=%d n_j=%d",
                  low_pole ? "low" : "high",
                  q,
                  target_span,
                  span_warn_bound,
                  pc.n_j_cells);
    core::log_warning(warn);
  }
  return true;
}

struct SpanGrowthRequest {
  int q = -1;
  bool low_pole = true;
  int target_span = 0;
};

__global__ void zero_member_compatible_cell_buffers_kernel(
    double* __restrict__ corner_force_p_r,
    double* __restrict__ corner_force_p_z,
    double* __restrict__ corner_force_sub_r,
    double* __restrict__ corner_force_sub_z,
    double* __restrict__ corner_force_q_r,
    double* __restrict__ corner_force_q_z,
    double* __restrict__ work_p,
    double* __restrict__ work_sub,
    double* __restrict__ work_av,
    const std::uint8_t* __restrict__ member_mask,
    const int n_cells,
    const int corner_stride) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  const int base = 4 * c;
  for (int k = 0; k < 4; ++k) {
    corner_force_p_r[base + k] = 0.0;
    corner_force_p_z[base + k] = 0.0;
    corner_force_sub_r[base + k] = 0.0;
    corner_force_sub_z[base + k] = 0.0;
  }
  if (corner_force_q_r != nullptr) {
    const int q_base = c * corner_stride;
    for (int k = 0; k < corner_stride; ++k) {
      corner_force_q_r[q_base + k] = 0.0;
      corner_force_q_z[q_base + k] = 0.0;
    }
  }
  work_p[c] = 0.0;
  work_sub[c] = 0.0;
  work_av[c] = 0.0;
}

__global__ void zero_member_work_kernel(double* __restrict__ work_p,
                                        double* __restrict__ work_sub,
                                        double* __restrict__ work_av,
                                        const std::uint8_t* __restrict__ member_mask,
                                        const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  work_p[c] = 0.0;
  work_sub[c] = 0.0;
  work_av[c] = 0.0;
}

__global__ void zero_member_aw_pressure_work_force_kernel(
    double* __restrict__ corner_force_p_rz_r,
    double* __restrict__ corner_force_p_rz_z,
    const std::uint8_t* __restrict__ member_mask,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells || member_mask == nullptr || member_mask[c] == 0U) {
    return;
  }
  const int base = 4 * c;
  for (int k = 0; k < 4; ++k) {
    corner_force_p_rz_r[base + k] = 0.0;
    corner_force_p_rz_z[base + k] = 0.0;
  }
}

__global__ void zero_internal_member_edge_forces_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    const int* __restrict__ cell_a,
    const int* __restrict__ cell_b,
    const std::uint8_t* __restrict__ member_mask,
    const int n_edges,
    const int offset) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_edges || member_mask == nullptr) {
    return;
  }
  const int a = cell_a[e];
  const int b = cell_b[e];
  if ((a >= 0 && member_mask[a] != 0U) || (b >= 0 && member_mask[b] != 0U)) {
    edge_force_r[offset + e] = 0.0;
    edge_force_z[offset + e] = 0.0;
  }
}

__global__ void zero_boundary_member_edge_forces_kernel(
    double* __restrict__ edge_force_r,
    double* __restrict__ edge_force_z,
    const int* __restrict__ cell,
    const std::uint8_t* __restrict__ member_mask,
    const int n_edges,
    const int offset) {
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= n_edges || member_mask == nullptr) {
    return;
  }
  const int c = cell[e];
  if (c >= 0 && member_mask[c] != 0U) {
    edge_force_r[offset + e] = 0.0;
    edge_force_z[offset + e] = 0.0;
  }
}

__global__ void scatter_boundary_force_kernel(
    double* __restrict__ force_r,
    double* __restrict__ force_z,
    double* __restrict__ impulse_r,
    double* __restrict__ impulse_z,
    const int* __restrict__ nodes,
    const double* __restrict__ boundary_force_r,
    const double* __restrict__ boundary_force_z,
    const int n,
    const double impulse_dt) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= n) {
    return;
  }
  const int node = nodes[k];
  const double fr = boundary_force_r[k];
  const double fz = boundary_force_z[k];
  atomicAdd(force_r + node, fr);
  atomicAdd(force_z + node, fz);
  impulse_r[k] = impulse_dt * fr;
  impulse_z[k] = impulse_dt * fz;
}

double macro_pressure_from_cells(
    const core::PoleAngularDerefineMacroState& macro,
    const std::vector<double>& mass,
    const std::vector<double>& pressure) {
  long double numerator = 0.0L;
  long double denominator = 0.0L;
  for (const int c : macro.member_cells) {
    const std::size_t idx = static_cast<std::size_t>(c);
    const double m = mass[idx];
    const double p = pressure[idx];
    if (std::isfinite(m) && m > 0.0 && std::isfinite(p)) {
      numerator += static_cast<long double>(m) * static_cast<long double>(p);
      denominator += static_cast<long double>(m);
    }
  }
  if (!(denominator > 0.0L)) {
    return 0.0;
  }
  const double p = static_cast<double>(numerator / denominator);
  return (std::isfinite(p) && p > 0.0) ? p : 0.0;
}

void build_boundary_force_host(const core::PoleAngularDerefineState& pc,
                               const std::vector<double>& node_r,
                               const std::vector<double>& node_z,
                               const std::vector<double>& mass,
                               const std::vector<double>& pressure,
                               std::vector<double>& force_r,
                               std::vector<double>& force_z) {
  force_r.assign(pc.boundary_nodes_flat.size(), 0.0);
  force_z.assign(pc.boundary_nodes_flat.size(), 0.0);
  for (std::size_t m = 0; m < pc.macros.size(); ++m) {
    const auto& macro = pc.macros[m];
    const int begin = pc.boundary_node_offsets[m];
    const int end = pc.boundary_node_offsets[m + 1U];
    const int n_boundary = end - begin;
    std::vector<double> r(static_cast<std::size_t>(n_boundary));
    std::vector<double> z(static_cast<std::size_t>(n_boundary));
    for (int k = 0; k < n_boundary; ++k) {
      const int n = pc.boundary_nodes_flat[static_cast<std::size_t>(begin + k)];
      r[static_cast<std::size_t>(k)] = node_r[static_cast<std::size_t>(n)];
      z[static_cast<std::size_t>(k)] = node_z[static_cast<std::size_t>(n)];
    }
    double centroid_r = 0.0;
    double centroid_z = 0.0;
    rz::rz_polygon_area_centroid_exact(r.data(), z.data(), n_boundary,
                                       &centroid_r, &centroid_z);
    const double p_c = macro_pressure_from_cells(macro, mass, pressure);
    for (int k = 0; k < n_boundary; ++k) {
      double sr = 0.0;
      double sz = 0.0;
      rz::rz_polygon_svec_exact(r.data(), z.data(), n_boundary, k,
                                centroid_r, centroid_z, 1.0, &sr, &sz);
      force_r[static_cast<std::size_t>(begin + k)] = p_c * sr;
      force_z[static_cast<std::size_t>(begin + k)] = p_c * sz;
    }
  }
}

}  // namespace

void refresh_geometry_exempt_cells(core::State& state) {
  refresh_geometry_exempt_cells_impl(state);
}

bool configured(const core::Config& cfg) {
  const bool enabled = env_flag_enabled(kEnvEnable);
  if (enabled && cfg.mesh.shell_polar_cap_dendrite) {
    throw core::namelist::ConfigError(
        "pole angular derefine is not supported with "
        "Mesh.shell_polar_cap_dendrite=true (pending shell-chain generalization)");
  }
  return is_supported_topology(cfg) && enabled;
}

bool assertions_enabled() {
  return env_flag_enabled(kEnvEnable);
}

bool active(const core::State& state) {
  return state.pole_angular_derefine.configured &&
         state.pole_angular_derefine.built &&
         state.pole_angular_derefine.valid;
}

void ensure_built(core::State& state, const core::Config& cfg) {
  if (!configured(cfg)) {
    return;
  }
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::ensure_built(state, cfg);
  }
  auto& pc = state.pole_angular_derefine;
  pc.configured = true;
  if (pc.built) {
    refresh_geometry_exempt_cells_impl(state);
    return;
  }
  pc.built = true;
  pc.valid = false;
  if (!state.mesh.topo.multiblock.has_value() || state.mesh.topo.n_cells <= 0) {
    return;
  }
  const auto& mb = *state.mesh.topo.multiblock;
  const int block_id = polar_shell_block_id(mb);
  if (block_id < 0) {
    return;
  }
  const auto& shell = mb.blocks[static_cast<std::size_t>(block_id)];
  TENRYU_ASSERT(shell.role == mesh::BlockRole::POLAR_SHELL,
                "polar shell angular de-refine requires POLAR_SHELL block");
  TENRYU_ASSERT(shell.n_i_cells > 0 && shell.n_j_cells >= 4,
                "polar shell angular de-refine needs a non-empty shell grid");
  TENRYU_ASSERT(shell.owned_node_count ==
                    (shell.n_i_cells + 1) * (shell.n_j_cells + 1),
                "polar shell angular de-refine requires contiguous shell nodes");

  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  const std::size_t expected_corner_mass =
      static_cast<std::size_t>(n_cells) * 4U;
  if (state.corner_mass.size() != expected_corner_mass) {
    pc.built = false;
    pc.valid = false;
    return;
  }
  pc.block_id = block_id;
  pc.n_i_cells = shell.n_i_cells;
  pc.n_j_cells = shell.n_j_cells;
  pc.cell_begin = shell.cell_begin;
  pc.owned_node_begin = shell.owned_node_begin;
  pc.member_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  pc.inactive_member_mask.assign(static_cast<std::size_t>(n_cells), 0U);
  std::vector<std::uint8_t> unavailable_cells = pc.inactive_member_mask;
  pc.boundary_node_mask.assign(static_cast<std::size_t>(n_nodes), 0U);
  pc.boundary_nodes_flat.clear();
  pc.boundary_node_offsets.clear();
  pc.boundary_node_offsets.push_back(0);
  pc.macros.clear();

  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  const std::vector<double> mass = copy_field(state.mass);
  const std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);

  for (int i = 0; i < shell.n_i_cells; ++i) {
    for (const bool low_pole : {true, false}) {
      const int span = select_pole_span(shell, node_r, node_z, i, low_pole);
      if (span < 2 || span > shell.n_j_cells) {
        continue;
      }
      const int j0 = low_pole ? 0 : shell.n_j_cells - span;
      const int j1 = low_pole ? span : shell.n_j_cells;
      if (!macro_material_pure(shell, tracer, i, j0, j1)) {
        continue;
      }
      auto macro = make_macro(shell, block_id, i, j0, j1, node_r, node_z);
      if (!cells_available(macro, unavailable_cells)) {
        continue;
      }
      initialize_scatter_weights(macro, mass);
      mark_macro(pc, macro);
      for (const int c : macro.member_cells) {
        unavailable_cells[static_cast<std::size_t>(c)] = 1U;
      }
      pc.macros.push_back(std::move(macro));
      pc.boundary_node_offsets.push_back(
          static_cast<int>(pc.boundary_nodes_flat.size()));
    }
  }

  pc.macro_count = static_cast<int>(pc.macros.size());
  pc.valid = !pc.macros.empty();
  refresh_geometry_exempt_cells_impl(state);
  if (!pc.valid) {
    return;
  }
  pc.representative_cell = pc.macros.front().representative_cell;
  assert_macro_mask_coverage(state);
  upload_overlay(state);
  project_activation_kinetic_energy(state, cfg);
  aggregate_state(state, cfg, "activation", false);
  const auto totals = scalar_totals(state);
  pc.E_total_ref = totals.E_total;
  pc.mass_ref = totals.mass;
  pc.tracer_mass_ref = totals.tracer_mass;
  pc.Ue_ref = totals.Ue;
  pc.Ui_ref = totals.Ui;
  std::fprintf(stderr,
               "[pole_angular_derefine] built step=%d macros=%d block=%d "
               "activation_dU=%.17e K_old=%.17e K_new=%.17e\n",
               state.step,
               pc.macro_count,
               pc.block_id,
               pc.activation_ke_dU,
               pc.activation_ke_K_old,
               pc.activation_ke_K_new);
}

void aggregate_state(core::State& state,
                     const core::Config& cfg,
                     const char* phase,
                     const bool emit_audit) {
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return;
  }
  auto& pc = state.pole_angular_derefine;
  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> vol = copy_field(state.vol);
  std::vector<double> rho = copy_field(state.rho);
  std::vector<double> ee = copy_field(state.ee);
  std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                            : copy_field(state.ei);
  std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  for (auto& macro : pc.macros) {
    const MemberSums sums = sum_member_state(macro, mass, vol, ee, ei, tracer);
    TENRYU_ASSERT(sums.M > 0.0L,
                  "polar shell angular de-refine aggregate needs positive mass");
    macro.M_c = static_cast<double>(sums.M);
    macro.M_Y_c = static_cast<double>(sums.MY);
    macro.Ue_c = static_cast<double>(sums.Ue);
    macro.Ui_c = static_cast<double>(sums.Ui);
    macro.V_c = boundary_loop_volume(macro, node_r, node_z);
    write_macro_mirrors(state, macro, mass, vol, rho, ee, ei, tracer);
  }
  state.mass.copy_from_host(mass.data());
  state.vol.copy_from_host(vol.data());
  state.rho.copy_from_host(rho.data());
  state.ee.copy_from_host(ee.data());
  if (!ei.empty()) {
    state.ei.copy_from_host(ei.data());
  }
  if (!tracer.empty()) {
    state.gas_tracer_Y.copy_from_host(tracer.data());
  }
  if (conservation_audit::enabled()) {
    std::string audit_stage = std::string("pole_aggregate:") +
                              (phase != nullptr ? phase : "unspecified");
    conservation_audit::emit_stage(state, audit_stage.c_str());
  }

  if (emit_audit) {
    const auto totals = scalar_totals(state);
    std::fprintf(stderr,
                 "[pole_angular_derefine_audit] step=%d t=%.17e phase=%s "
                 "dM=%.17e dMY=%.17e dUe=%.17e dUi=%.17e dE=%.17e "
                 "relE=%.17e\n",
                 state.step,
                 state.t,
                 phase != nullptr ? phase : "",
                 totals.mass - pc.mass_ref,
                 totals.tracer_mass - pc.tracer_mass_ref,
                 totals.Ue - pc.Ue_ref,
                 totals.Ui - pc.Ui_ref,
                 totals.E_total - pc.E_total_ref,
                 relative_delta(totals.E_total, pc.E_total_ref));
  }
}

bool maintain_pole_spans(core::State& state,
                         const core::Config& cfg,
                         const char* phase) {
  if (!configured(cfg)) {
    return false;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return false;
  }
  auto& pc = state.pole_angular_derefine;
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "polar shell angular de-refine span maintenance requires multiblock");
  const auto& mb = *state.mesh.topo.multiblock;
  if (pc.block_id < 0 || pc.block_id >= static_cast<int>(mb.blocks.size())) {
    return false;
  }
  const auto& shell = mb.blocks[static_cast<std::size_t>(pc.block_id)];
  const int max_span = max_pole_span(pc.n_j_cells);
  if (max_span < 2) {
    return false;
  }

  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  std::vector<SpanGrowthRequest> requests;
  requests.reserve(pc.macros.size());
  for (const auto& macro : pc.macros) {
    const bool low_pole = macro.local_j_begin == 0;
    const bool high_pole = macro.local_j_end == pc.n_j_cells;
    if (!low_pole && !high_pole) {
      continue;
    }
    const int old_span = macro.local_j_end - macro.local_j_begin;
    const int selected_span =
        select_pole_span(shell, node_r, node_z, macro.local_i_begin, low_pole);
    const int target_span = std::min(max_span, selected_span);
    if (old_span >= 2 && target_span > old_span) {
      SpanGrowthRequest request;
      request.q = macro.local_i_begin;
      request.low_pole = low_pole;
      request.target_span = target_span;
      requests.push_back(request);
    }
  }

  bool changed = false;
  for (const auto& request : requests) {
    changed |= replace_pole_macro_span(state,
                                       cfg,
                                       shell,
                                       request.q,
                                       request.low_pole,
                                       request.target_span,
                                       phase != nullptr ? phase
                                                        : "span_maintain",
                                       "pole_angular_derefine_maintain");
  }
  return changed;
}

const std::uint8_t* member_mask_device(core::State& state,
                                       const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state) ? state.pole_angular_derefine.d_member_mask.data()
                       : nullptr;
}

const std::uint8_t* inactive_member_mask_device(core::State& state,
                                                const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state)
             ? state.pole_angular_derefine.d_inactive_member_mask.data()
             : nullptr;
}

const std::uint8_t* boundary_node_mask_device(core::State& state,
                                              const core::Config& cfg) {
  if (!configured(cfg)) {
    return nullptr;
  }
  ensure_built(state, cfg);
  return active(state) ? state.pole_angular_derefine.d_boundary_node_mask.data()
                       : nullptr;
}

const std::uint8_t* combined_inactive_mask_device(
    core::State& state,
    const core::Config& cfg,
    core::DeviceArray<std::uint8_t>& scratch) {
  if (central_pseudo_core::configured(cfg)) {
    central_pseudo_core::ensure_built(state, cfg);
  }
  ensure_built(state, cfg);
  return combined_inactive_mask_device(state, scratch);
}

const std::uint8_t* combined_inactive_mask_device(
    core::State& state,
    core::DeviceArray<std::uint8_t>& scratch) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0) {
    return nullptr;
  }
  std::vector<std::uint8_t> mask(static_cast<std::size_t>(n_cells), 0U);
  bool any = false;
  if (state.central_pseudo_core.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (state.central_pseudo_core.inactive_member_mask[static_cast<std::size_t>(c)] !=
          0U) {
        mask[static_cast<std::size_t>(c)] = 1U;
        any = true;
      }
    }
  }
  if (state.pole_angular_derefine.inactive_member_mask.size() ==
      static_cast<std::size_t>(n_cells)) {
    for (int c = 0; c < n_cells; ++c) {
      if (state.pole_angular_derefine
              .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
        mask[static_cast<std::size_t>(c)] = 1U;
        any = true;
      }
    }
  }
  if (!any) {
    return nullptr;
  }
  scratch.reset(mask.size());
  scratch.copy_from_host(mask);
  return scratch.data();
}

std::vector<std::int8_t> apply_inactive_to_active_mask(
    const core::State& state,
    const std::vector<std::int8_t>* base_active) {
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 ||
      state.pole_angular_derefine.inactive_member_mask.size() !=
          static_cast<std::size_t>(n_cells)) {
    return {};
  }
  std::vector<std::int8_t> active(static_cast<std::size_t>(n_cells), 1);
  if (base_active != nullptr && !base_active->empty()) {
    TENRYU_ASSERT(base_active->size() == static_cast<std::size_t>(n_cells),
                  "polar shell angular de-refine base active size mismatch");
    active = *base_active;
  } else if (!state.hydro_active.empty()) {
    TENRYU_ASSERT(state.hydro_active.size() == static_cast<std::size_t>(n_cells),
                  "polar shell angular de-refine hydro_active size mismatch");
    active = state.hydro_active;
  }
  for (int c = 0; c < n_cells; ++c) {
    if (state.pole_angular_derefine
            .inactive_member_mask[static_cast<std::size_t>(c)] != 0U) {
      active[static_cast<std::size_t>(c)] = 0;
    }
  }
  return active;
}

tenryu::hydro::pole_angular_coarsen::Overlay path_overlay(
    core::State& state,
    const core::Config& cfg) {
  tenryu::hydro::pole_angular_coarsen::Overlay overlay;
  if (!configured(cfg)) {
    return overlay;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return overlay;
  }
  const auto& pc = state.pole_angular_derefine;
  overlay.active = true;
  overlay.skip_fine_child_paths = true;
  overlay.supports_deref_macro_repair = true;
  overlay.block_id = pc.block_id;
  overlay.representative_cell = pc.representative_cell;
  overlay.q_min = 0;
  overlay.q_max = pc.n_i_cells - 1;
  overlay.level_max = kLevelMax;
  overlay.n_i_cells = pc.n_i_cells;
  overlay.n_j_cells = pc.n_j_cells;
  overlay.cell_begin = pc.cell_begin;
  overlay.owned_node_begin = pc.owned_node_begin;
  overlay.pole_coarsen_inactive_fine_mask = pc.inactive_member_mask;
  overlay.macros.reserve(pc.macros.size());
  for (const auto& macro_state : pc.macros) {
    tenryu::hydro::pole_angular_coarsen::Macro macro;
    macro.block_id = macro_state.block_id;
    macro.representative_cell = macro_state.representative_cell;
    macro.local_i_begin = macro_state.local_i_begin;
    macro.local_i_end = macro_state.local_i_end;
    macro.local_j_begin = macro_state.local_j_begin;
    macro.local_j_end = macro_state.local_j_end;
    macro.level = macro_state.level;
    macro.span = macro_state.span;
    macro.skipped_nodes = macro_state.skipped_nodes;
    macro.single_apex_boundary = macro_state.single_apex_boundary;
    macro.canonical_apex_node = macro_state.canonical_apex_node;
    macro.boundary_nodes_ordered = macro_state.boundary_nodes_ordered;
    overlay.macros.push_back(std::move(macro));
  }
  return overlay;
}

bool repair_macro_boundary(core::State& state,
                           const core::Config& cfg,
                           const int local_i_begin,
                           const int local_j_begin,
                           const int local_j_end) {
  if (!configured(cfg)) {
    return false;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return false;
  }
  auto& pc = state.pole_angular_derefine;
  if (local_i_begin < 0 || local_i_begin >= pc.n_i_cells ||
      local_j_end <= local_j_begin ||
      !(local_j_begin == 0 || local_j_end == pc.n_j_cells)) {
    return false;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "polar shell angular de-refine repair requires multiblock");
  const auto& mb = *state.mesh.topo.multiblock;
  const auto& shell = mb.blocks[static_cast<std::size_t>(pc.block_id)];
  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  const std::vector<double> mass = copy_field(state.mass);

  for (auto& macro : pc.macros) {
    if (macro.local_i_begin == local_i_begin &&
        macro.local_j_begin == local_j_begin &&
        macro.local_j_end == local_j_end &&
        canonicalize_short_pole_edge(macro, pc.n_j_cells, node_r, node_z)) {
      sync_overlay_masks_after_mutation(state);
      aggregate_state(state, cfg, "macro_repair_canonicalize", false);
      core::log_warning(
          "[pole_angular_derefine_repair] canonicalized pole edge q=" +
          std::to_string(local_i_begin) + " j=" +
          std::to_string(local_j_begin) + ":" +
          std::to_string(local_j_end));
      return true;
    }
  }

  const int old_span = local_j_end - local_j_begin;
  const int max_span =
      std::min(span_for_level(kLevelMax), std::max(1, pc.n_j_cells / 2));
  if (old_span < 2 || old_span >= max_span) {
    return false;
  }
  const int new_span = std::min(max_span, old_span * 2);
  const bool low_pole = local_j_begin == 0;
  const int new_j_begin = low_pole ? 0 : pc.n_j_cells - new_span;
  const int new_j_end = low_pole ? new_span : pc.n_j_cells;
  const std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  if (!macro_material_pure(
          shell, tracer, local_i_begin, new_j_begin, new_j_end)) {
    return false;
  }

  auto replacement = make_macro(shell,
                                pc.block_id,
                                local_i_begin,
                                new_j_begin,
                                new_j_end,
                                node_r,
                                node_z);
  initialize_scatter_weights(replacement, mass);

  std::vector<core::PoleAngularDerefineMacroState> repaired;
  repaired.reserve(pc.macros.size());
  bool removed = false;
  std::vector<std::uint8_t> used(
      static_cast<std::size_t>(state.mesh.topo.n_cells), 0U);
  for (const auto& macro : pc.macros) {
    const bool replace_this =
        macro.local_i_begin == local_i_begin &&
        same_pole_side(pc, macro, local_j_begin, local_j_end) &&
        intervals_overlap(macro.local_j_begin,
                          macro.local_j_end,
                          new_j_begin,
                          new_j_end);
    if (replace_this) {
      removed = true;
      continue;
    }
    for (const int c : macro.member_cells) {
      if (used[static_cast<std::size_t>(c)] != 0U) {
        return false;
      }
      used[static_cast<std::size_t>(c)] = 1U;
    }
    repaired.push_back(macro);
  }
  if (!removed) {
    return false;
  }
  for (const int c : replacement.member_cells) {
    if (used[static_cast<std::size_t>(c)] != 0U) {
      return false;
    }
    used[static_cast<std::size_t>(c)] = 1U;
  }
  repaired.push_back(std::move(replacement));
  pc.macros = std::move(repaired);
  sync_overlay_masks_after_mutation(state);
  aggregate_state(state, cfg, "macro_repair_coarsen", false);
  core::log_warning(
      "[pole_angular_derefine_repair] coarsened q=" +
      std::to_string(local_i_begin) + " j=" +
      std::to_string(local_j_begin) + ":" +
      std::to_string(local_j_end) + " span=" + std::to_string(old_span) +
      " -> j=" + std::to_string(new_j_begin) + ":" +
      std::to_string(new_j_end) + " span=" + std::to_string(new_span));
  return true;
}

bool extend_for_adjacent_active_child(core::State& state,
                                      const core::Config& cfg,
                                      const int failing_cell,
                                      const bool allow_next_span) {
  if (!configured(cfg) || failing_cell < 0) {
    return false;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return false;
  }
  auto& pc = state.pole_angular_derefine;
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "polar shell angular de-refine extension requires multiblock");
  const auto& mb = *state.mesh.topo.multiblock;
  if (pc.block_id < 0 || pc.block_id >= static_cast<int>(mb.blocks.size())) {
    return false;
  }
  const auto& shell = mb.blocks[static_cast<std::size_t>(pc.block_id)];
  if (failing_cell < shell.cell_begin ||
      failing_cell >= shell.cell_begin + shell.cell_count ||
      pc.inactive_member_mask.size() !=
          static_cast<std::size_t>(state.mesh.topo.n_cells) ||
      pc.inactive_member_mask[static_cast<std::size_t>(failing_cell)] != 0U) {
    return false;
  }
  const int local = failing_cell - shell.cell_begin;
  const int q = local / shell.n_j_cells;
  const int j = local - q * shell.n_j_cells;
  if (q < 0 || q >= shell.n_i_cells || j < 0 || j >= shell.n_j_cells) {
    return false;
  }

  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  const int max_span = max_pole_span(pc.n_j_cells);

  for (std::size_t macro_index = 0; macro_index < pc.macros.size();
       ++macro_index) {
    const auto& macro = pc.macros[macro_index];
    if (macro.local_i_begin != q) {
      continue;
    }
    const bool low_pole = macro.local_j_begin == 0 && j >= macro.local_j_end;
    const bool high_pole =
        macro.local_j_end == pc.n_j_cells && j < macro.local_j_begin;
    if (!low_pole && !high_pole) {
      continue;
    }

    const int old_span = macro.local_j_end - macro.local_j_begin;
    int selected_span = select_pole_span(shell, node_r, node_z, q, low_pole);
    if (allow_next_span && selected_span <= old_span && old_span < max_span) {
      selected_span = std::min(max_span, old_span * 2);
    }
    const int required_span = low_pole ? j + 1 : pc.n_j_cells - j;
    if (required_span > max_span) {
      continue;
    }
    const int required_dyadic = dyadic_span_at_least(required_span, max_span);
    const int new_span =
        std::max(std::min(max_span, selected_span), required_dyadic);
    if (old_span < 2 || new_span <= old_span) {
      continue;
    }
    if (replace_pole_macro_span(state,
                                cfg,
                                shell,
                                q,
                                low_pole,
                                new_span,
                                "macro_extend",
                                "pole_angular_derefine_extend")) {
      return true;
    }
  }
  return false;
}

double acoustic_dt(core::State& state, const core::Config& cfg) {
  if (!configured(cfg)) {
    return std::numeric_limits<double>::infinity();
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return std::numeric_limits<double>::infinity();
  }
  aggregate_state(state, cfg, "cfl_acoustic", false);
  const auto& pc = state.pole_angular_derefine;
  std::vector<double> cs =
      state.cs.empty() ? std::vector<double>{} : copy_field(state.cs);
  std::vector<double> node_r = copy_field(state.x_r);
  std::vector<double> node_z = copy_field(state.x_z);
  double dt = std::numeric_limits<double>::infinity();
  for (const auto& macro : pc.macros) {
    double h = std::numeric_limits<double>::infinity();
    for (std::size_t k = 0; k < macro.boundary_nodes_ordered.size(); ++k) {
      const int a = macro.boundary_nodes_ordered[k];
      const int b =
          macro.boundary_nodes_ordered[(k + 1U) %
                                       macro.boundary_nodes_ordered.size()];
      h = std::min(h, distance_node(node_r, node_z, a, b));
    }
    if (!(h > 0.0) || !std::isfinite(h)) {
      continue;
    }
    const int c0 = macro.representative_cell;
    const double c_eff =
        !cs.empty() ? std::max(0.0, cs[static_cast<std::size_t>(c0)]) : 0.0;
    if (!(c_eff > 0.0) || !std::isfinite(c_eff)) {
      continue;
    }
    dt = std::min(dt,
                  cfg.numerics.dt.cfl_hydro * h /
                      (c_eff * (1.0 + cfg.numerics.hydro.av_linear)));
  }
  return dt;
}

void assert_active_cell(const core::State& state, const int cell, const char* op) {
  if (!assertions_enabled() || cell < 0) {
    return;
  }
  const auto& inactive = state.pole_angular_derefine.inactive_member_mask;
  if (inactive.empty()) {
    return;
  }
  TENRYU_ASSERT(static_cast<std::size_t>(cell) >= inactive.size() ||
                    inactive[static_cast<std::size_t>(cell)] == 0U,
                std::string(op != nullptr ? op : "operator") +
                    " touched inactive polar shell de-refine child");
}

void zero_member_compatible_cell_buffers(core::State& state) {
  if (!active(state)) {
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  if (n_cells <= 0 || state.corner_force_p_r.empty() ||
      state.corner_force_sub_r.empty() || state.work_p_per_cell.empty()) {
    return;
  }
  const int blocks = (n_cells + 255) / 256;
  const std::size_t n_corners =
      static_cast<std::size_t>(n_cells) *
      static_cast<std::size_t>(state.corner_stride);
  const bool has_corner_q = state.corner_force_q_r.size() == n_corners &&
                            state.corner_force_q_z.size() == n_corners;
  zero_member_compatible_cell_buffers_kernel<<<blocks, 256>>>(
      state.corner_force_p_r.data(),
      state.corner_force_p_z.data(),
      state.corner_force_sub_r.data(),
      state.corner_force_sub_z.data(),
      has_corner_q ? state.corner_force_q_r.data() : nullptr,
      has_corner_q ? state.corner_force_q_z.data() : nullptr,
      state.work_p_per_cell.data(),
      state.work_sub_per_cell.data(),
      state.work_av_per_cell.data(),
      state.pole_angular_derefine.d_member_mask.data(),
      n_cells,
      state.corner_stride);
  cuda_check(cudaGetLastError(),
             "polar shell angular de-refine zero compatible buffers launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "polar shell angular de-refine zero compatible buffers failed");
  if (!state.corner_force_p_rz_r.empty()) {
    zero_member_aw_pressure_work_force_kernel<<<blocks, 256>>>(
        state.corner_force_p_rz_r.data(),
        state.corner_force_p_rz_z.data(),
        state.pole_angular_derefine.d_member_mask.data(),
        n_cells);
    cuda_check(
        cudaGetLastError(),
        "polar shell angular de-refine zero member AW pressure work force launch failed");
    cuda_check(
        cudaDeviceSynchronize(),
        "polar shell angular de-refine zero member AW pressure work force failed");
  }
}

void zero_member_edge_av_forces(core::State& state) {
  if (!active(state) || state.edge_force_av_r.empty()) {
    return;
  }
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "polar shell angular de-refine edge AV zero needs multiblock");
  const auto& mb = *state.mesh.topo.multiblock;
  const int n_internal = static_cast<int>(mb.unique_internal_faces.size());
  const int n_boundary = static_cast<int>(mb.boundary_faces.size());
  if (n_internal > 0) {
    const int blocks = (n_internal + 255) / 256;
    zero_internal_member_edge_forces_kernel<<<blocks, 256>>>(
        state.edge_force_av_r.data(),
        state.edge_force_av_z.data(),
        thrust::raw_pointer_cast(mb.d_unique_face_cell_a.data()),
        thrust::raw_pointer_cast(mb.d_unique_face_cell_b.data()),
        state.pole_angular_derefine.d_member_mask.data(),
        n_internal,
        0);
  }
  if (n_boundary > 0) {
    const int blocks = (n_boundary + 255) / 256;
    zero_boundary_member_edge_forces_kernel<<<blocks, 256>>>(
        state.edge_force_av_r.data(),
        state.edge_force_av_z.data(),
        thrust::raw_pointer_cast(mb.d_boundary_face_cell.data()),
        state.pole_angular_derefine.d_member_mask.data(),
        n_boundary,
        n_internal);
  }
  cuda_check(cudaGetLastError(),
             "polar shell angular de-refine zero edge AV launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "polar shell angular de-refine zero edge AV failed");
}

void add_boundary_pressure_force(core::State& state,
                                 const core::Config& cfg,
                                 const core::CellField1D& cell_pressure,
                                 double* force_r,
                                 double* force_z,
                                 const double impulse_dt) {
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  auto& pc = state.pole_angular_derefine;
  if (!active(state) || pc.boundary_nodes_flat.empty()) {
    return;
  }
  std::vector<double> node_r = copy_field(state.x_r);
  std::vector<double> node_z = copy_field(state.x_z);
  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> pressure = copy_field(cell_pressure);
  std::vector<double> force_host_r;
  std::vector<double> force_host_z;
  build_boundary_force_host(pc, node_r, node_z, mass, pressure,
                            force_host_r, force_host_z);
  core::DeviceArray<double> d_force_r(force_host_r.size());
  core::DeviceArray<double> d_force_z(force_host_z.size());
  d_force_r.copy_from_host(force_host_r);
  d_force_z.copy_from_host(force_host_z);
  const int n = static_cast<int>(pc.boundary_nodes_flat.size());
  const int blocks = (n + 255) / 256;
  scatter_boundary_force_kernel<<<blocks, 256>>>(
      force_r,
      force_z,
      pc.d_boundary_impulse_r.data(),
      pc.d_boundary_impulse_z.data(),
      pc.d_boundary_nodes_flat.data(),
      d_force_r.data(),
      d_force_z.data(),
      n,
      impulse_dt);
  cuda_check(cudaGetLastError(),
             "polar shell angular de-refine boundary force launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "polar shell angular de-refine boundary force failed");
}

void set_boundary_pressure_work(core::State& state,
                                const core::Config& cfg,
                                const double dt,
                                const double* old_velocity_r,
                                const double* old_velocity_z,
                                const double* new_velocity_r,
                                const double* new_velocity_z) {
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state) || state.work_p_per_cell.empty()) {
    return;
  }
  auto& pc = state.pole_angular_derefine;
  for (auto& macro : pc.macros) {
    macro.boundary_work_pending = false;
    macro.boundary_work_valid = false;
  }
  if (pc.d_boundary_impulse_r.empty() || pc.d_boundary_impulse_z.empty() ||
      !(dt > 0.0)) {
    zero_member_compatible_cell_buffers(state);
    return;
  }
  const int n_cells = static_cast<int>(state.rho.size());
  const int cell_blocks = (n_cells + 255) / 256;
  zero_member_work_kernel<<<cell_blocks, 256>>>(
      state.work_p_per_cell.data(),
      state.work_sub_per_cell.data(),
      state.work_av_per_cell.data(),
      pc.d_member_mask.data(),
      n_cells);
  cuda_check(cudaGetLastError(),
             "polar shell angular de-refine zero work launch failed");
  cuda_check(cudaDeviceSynchronize(),
             "polar shell angular de-refine zero work failed");

  const int n_nodes = static_cast<int>(state.x_r.size());
  TENRYU_ASSERT(old_velocity_r != nullptr && old_velocity_z != nullptr &&
                    new_velocity_r != nullptr && new_velocity_z != nullptr,
                "polar shell angular de-refine pressure work needs velocities");
  std::vector<double> impulse_r;
  std::vector<double> impulse_z;
  std::vector<double> old_r(static_cast<std::size_t>(n_nodes));
  std::vector<double> old_z(static_cast<std::size_t>(n_nodes));
  std::vector<double> new_r(static_cast<std::size_t>(n_nodes));
  std::vector<double> new_z(static_cast<std::size_t>(n_nodes));
  pc.d_boundary_impulse_r.copy_to_host(impulse_r);
  pc.d_boundary_impulse_z.copy_to_host(impulse_z);
  cuda_check(cudaMemcpy(old_r.data(), old_velocity_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "polar shell angular de-refine copy old vr failed");
  cuda_check(cudaMemcpy(old_z.data(), old_velocity_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "polar shell angular de-refine copy old vz failed");
  cuda_check(cudaMemcpy(new_r.data(), new_velocity_r,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "polar shell angular de-refine copy new vr failed");
  cuda_check(cudaMemcpy(new_z.data(), new_velocity_z,
                        static_cast<std::size_t>(n_nodes) * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "polar shell angular de-refine copy new vz failed");

  for (std::size_t m = 0; m < pc.macros.size(); ++m) {
    auto& macro = pc.macros[m];
    const int begin = pc.boundary_node_offsets[m];
    const int end = pc.boundary_node_offsets[m + 1U];
    long double dot_old = 0.0L;
    long double dot_new = 0.0L;
    for (int k = begin; k < end; ++k) {
      const int n = pc.boundary_nodes_flat[static_cast<std::size_t>(k)];
      const long double Ir =
          static_cast<long double>(impulse_r[static_cast<std::size_t>(k)]);
      const long double Iz =
          static_cast<long double>(impulse_z[static_cast<std::size_t>(k)]);
      dot_old +=
          Ir * static_cast<long double>(old_r[static_cast<std::size_t>(n)]) +
          Iz * static_cast<long double>(old_z[static_cast<std::size_t>(n)]);
      dot_new +=
          Ir * static_cast<long double>(new_r[static_cast<std::size_t>(n)]) +
          Iz * static_cast<long double>(new_z[static_cast<std::size_t>(n)]);
    }
    const long double dot_mid = 0.5L * (dot_old + dot_new);
    macro.boundary_work_W_old = static_cast<double>(-dot_old);
    macro.boundary_work_W_new = static_cast<double>(-dot_new);
    macro.boundary_work_W_mid = static_cast<double>(-dot_mid);
    macro.boundary_work_dU = macro.boundary_work_W_mid;
    macro.boundary_work_residual =
        static_cast<double>(static_cast<long double>(macro.boundary_work_dU) +
                            dot_mid);
    macro.boundary_work_chi_e =
        electron_energy_fraction(macro.Ue_c, macro.Ui_c, !state.ei.empty());
    macro.boundary_work_valid = true;
    macro.boundary_work_pending = true;
  }
}

void apply_boundary_pressure_work(core::State& state,
                                  const core::Config& cfg,
                                  const bool use_two_temp) {
  if (!configured(cfg)) {
    return;
  }
  ensure_built(state, cfg);
  if (!active(state)) {
    return;
  }
  auto& pc = state.pole_angular_derefine;
  std::vector<double> mass = copy_field(state.mass);
  std::vector<double> vol = copy_field(state.vol);
  std::vector<double> rho = copy_field(state.rho);
  std::vector<double> ee = copy_field(state.ee);
  std::vector<double> ei = state.ei.empty() ? std::vector<double>{}
                                            : copy_field(state.ei);
  std::vector<double> tracer =
      state.gas_tracer_Y.empty() ? std::vector<double>{}
                                 : copy_field(state.gas_tracer_Y);
  const std::vector<double> node_r = copy_field(state.x_r);
  const std::vector<double> node_z = copy_field(state.x_z);
  bool any = false;
  double boundary_work_delta = 0.0;
  for (auto& macro : pc.macros) {
    if (!macro.boundary_work_pending) {
      continue;
    }
    const MemberSums sums = sum_member_state(macro, mass, vol, ee, ei, tracer);
    TENRYU_ASSERT(sums.M > 0.0L,
                  "polar shell angular de-refine work needs positive mass");
    macro.M_c = static_cast<double>(sums.M);
    macro.M_Y_c = static_cast<double>(sums.MY);
    macro.V_c = boundary_loop_volume(macro, node_r, node_z);
    if (use_two_temp && !ei.empty()) {
      const double chi_e = macro.boundary_work_chi_e;
      macro.Ue_c = static_cast<double>(sums.Ue) +
                   chi_e * macro.boundary_work_dU;
      macro.Ui_c = static_cast<double>(sums.Ui) +
                   (1.0 - chi_e) * macro.boundary_work_dU;
    } else {
      macro.Ue_c = static_cast<double>(sums.Ue) + macro.boundary_work_dU;
      macro.Ui_c = !ei.empty() ? static_cast<double>(sums.Ui) : 0.0;
    }
    TENRYU_ASSERT(std::isfinite(macro.Ue_c) && std::isfinite(macro.Ui_c),
                  "polar shell angular de-refine work produced bad energy");
    write_macro_mirrors(state, macro, mass, vol, rho, ee, ei, tracer);
    boundary_work_delta += macro.boundary_work_dU;
    macro.boundary_work_pending = false;
    any = true;
  }
  if (!any) {
    return;
  }
  state.mass.copy_from_host(mass.data());
  state.vol.copy_from_host(vol.data());
  state.rho.copy_from_host(rho.data());
  state.ee.copy_from_host(ee.data());
  if (!ei.empty()) {
    state.ei.copy_from_host(ei.data());
  }
  if (!tracer.empty()) {
    state.gas_tracer_Y.copy_from_host(tracer.data());
  }
  conservation_audit::emit_stage(
      state, "pole_boundary_pressure_work", boundary_work_delta);
}

}  // namespace tenryu::hydro::pole_angular_derefine
