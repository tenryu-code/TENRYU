#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale_reference_diagnostics {
namespace detail {

inline std::string format_value(const double value) {
  std::ostringstream oss;
  oss << std::scientific << std::setprecision(16) << value;
  return oss.str();
}

inline double percentile_sorted(const std::vector<double>& sorted,
                                const double q) {
  if (sorted.empty()) {
    return 0.0;
  }
  const double x = q * static_cast<double>(sorted.size() - 1U);
  const std::size_t lo = static_cast<std::size_t>(std::floor(x));
  const std::size_t hi = std::min(lo + 1U, sorted.size() - 1U);
  const double a = x - static_cast<double>(lo);
  return sorted[lo] * (1.0 - a) + sorted[hi] * a;
}

}  // namespace detail

inline void sample_reference_pre_remap(core::State& state,
                                       const double dt,
                                       const int remap_call) {
  if (remap_call > 5 && (remap_call % 50) != 0) {
    return;
  }

  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  if (n_cells <= 0 || n_nodes <= 0 || !state.mesh.topo.multiblock.has_value()) {
    return;
  }

  const std::size_t n_cells_size = static_cast<std::size_t>(n_cells);
  const std::size_t n_nodes_size = static_cast<std::size_t>(n_nodes);
  if (state.gas_tracer_Y.empty() || !state.gas_tracer_initialized ||
      state.gas_tracer_Y.size() != n_cells_size) {
    std::ostringstream oss;
    oss << "[diff_ref_diag] step=" << state.step
        << " remap_call=" << remap_call
        << " diff_ref_diag: gas tracer unavailable";
    core::log_info(oss.str());
    return;
  }

  TENRYU_ASSERT(state.vol.size() == n_cells_size &&
                    state.rho.size() == n_cells_size,
                "diff_ref_diag requires cell volume and density");
  TENRYU_ASSERT(state.x_r.size() == n_nodes_size &&
                    state.x_z.size() == n_nodes_size &&
                    state.x_r_reference.size() == n_nodes_size &&
                    state.x_z_reference.size() == n_nodes_size &&
                    state.v_r.size() == n_nodes_size &&
                    state.v_z.size() == n_nodes_size,
                "diff_ref_diag requires node coordinates, references, and velocities");

  const auto& mb = *state.mesh.topo.multiblock;
  TENRYU_ASSERT(mb.cell_node_csr_offsets.size() == n_cells_size + 1U,
                "diff_ref_diag requires host cell-node CSR offsets");

  std::vector<double> gas_tracer;
  std::vector<double> vol;
  std::vector<double> rho;
  std::vector<double> x_r;
  std::vector<double> x_z;
  std::vector<double> x_r_ref;
  std::vector<double> x_z_ref;
  std::vector<double> v_r;
  std::vector<double> v_z;
  state.gas_tracer_Y.copy_to_host(gas_tracer);
  state.vol.copy_to_host(vol);
  state.rho.copy_to_host(rho);
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.x_r_reference.copy_to_host(x_r_ref);
  state.x_z_reference.copy_to_host(x_z_ref);
  state.v_r.copy_to_host(v_r);
  state.v_z.copy_to_host(v_z);

  long double gas_volume = 0.0L;
  std::vector<double> gas_rho;
  std::vector<std::uint8_t> gas_node(n_nodes_size, static_cast<std::uint8_t>(0));
  for (int c = 0; c < n_cells; ++c) {
    if (!(gas_tracer[static_cast<std::size_t>(c)] > 0.5)) {
      continue;
    }
    const double vc = vol[static_cast<std::size_t>(c)];
    const double rhoc = rho[static_cast<std::size_t>(c)];
    if (std::isfinite(vc) && vc > 0.0) {
      gas_volume += static_cast<long double>(vc);
    }
    if (std::isfinite(rhoc)) {
      gas_rho.push_back(rhoc);
    }
    const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
    const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c) + 1U];
    const int available_nverts = (end > off) ? (end - off) : 0;
    int active_nverts = ::tenryu::mesh::mesh_topo_cell_active_nverts(
        state.mesh.cell_nverts, c);
    active_nverts = std::min(active_nverts, available_nverts);
    for (int k = 0; k < active_nverts; ++k) {
      const int n = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
      if (n >= 0 && n < n_nodes) {
        gas_node[static_cast<std::size_t>(n)] = static_cast<std::uint8_t>(1);
      }
    }
  }

  if (gas_rho.empty() || !(gas_volume > 0.0L)) {
    std::ostringstream oss;
    oss << "[diff_ref_diag] step=" << state.step
        << " remap_call=" << remap_call
        << " diff_ref_diag: gas tracer unavailable";
    core::log_info(oss.str());
    return;
  }

  std::sort(gas_rho.begin(), gas_rho.end());
  const double rho_p50 = detail::percentile_sorted(gas_rho, 0.5);
  if (!state.diff_ref_diag_baseline_initialized) {
    state.diff_ref_diag_gas_mesh_volume_initial = static_cast<double>(gas_volume);
    state.diff_ref_diag_gas_rho_p50_initial = rho_p50;
    state.diff_ref_diag_baseline_initialized = true;
  }

  std::vector<double> anti_compression;
  anti_compression.reserve(n_nodes_size);
  constexpr double eps = 1.0e-300;
  for (int n = 0; n < n_nodes; ++n) {
    if (gas_node[static_cast<std::size_t>(n)] == 0U) {
      continue;
    }
    const double r = x_r[static_cast<std::size_t>(n)];
    const double z = x_z[static_cast<std::size_t>(n)];
    const double radius = std::hypot(r, z);
    if (!(std::isfinite(radius) && radius > eps)) {
      continue;
    }
    const double er = r / radius;
    const double ez = z / radius;
    const double u_radial =
        v_r[static_cast<std::size_t>(n)] * er +
        v_z[static_cast<std::size_t>(n)] * ez;
    if (!(std::isfinite(u_radial) && u_radial < 0.0)) {
      continue;
    }
    const double numerator =
        (x_r_ref[static_cast<std::size_t>(n)] - r) * er +
        (x_z_ref[static_cast<std::size_t>(n)] - z) * ez;
    const double denominator = -u_radial * dt + eps;
    const double value = numerator / denominator;
    if (std::isfinite(value)) {
      anti_compression.push_back(value);
    }
  }
  std::sort(anti_compression.begin(), anti_compression.end());

  const double C_mesh_gas =
      state.diff_ref_diag_gas_mesh_volume_initial /
      static_cast<double>(gas_volume);
  const double C_rho_bulk =
      rho_p50 / state.diff_ref_diag_gas_rho_p50_initial;
  const double A_p50 = detail::percentile_sorted(anti_compression, 0.5);
  const double A_p90 = detail::percentile_sorted(anti_compression, 0.9);
  const double A_max = anti_compression.empty() ? 0.0 : anti_compression.back();

  std::ostringstream oss;
  oss << "[diff_ref_diag] step=" << state.step
      << " remap_call=" << remap_call
      << " C_mesh_gas=" << detail::format_value(C_mesh_gas)
      << " C_rho_bulk=" << detail::format_value(C_rho_bulk)
      << " A_n_p50=" << detail::format_value(A_p50)
      << " A_n_p90=" << detail::format_value(A_p90)
      << " A_n_max=" << detail::format_value(A_max)
      << " A_n_count=" << anti_compression.size();
  core::log_info(oss.str());
}

}  // namespace tenryu::hydro::ale_reference_diagnostics
