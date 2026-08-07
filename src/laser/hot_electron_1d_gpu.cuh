#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "laser/hot_electron_1d.cuh"

namespace tenryu::laser::hot_electron {

// Device execution of the cone pipeline. Consumes DEVICE pointers for the
// per-cell plasma fields (rho/zbar/A_eff/Te_eV/r_nodes, all n_cells or
// n_cells+1 doubles resident on the GPU) and host-side sources/config.
// Appends per-cell POWER into dep_power_cell (HOST vector) and fills the
// same DepositResult contract as deposit_hot_electrons_cone_1d.
DepositResult deposit_hot_electrons_cone_1d_device(
    const HotEChannelSpec& spec,
    Geometry1D geom,
    const std::vector<HotESource>& sources,
    const double* d_rho, const double* d_zbar, const double* d_A_eff,
    const double* d_Te_eV, const std::vector<std::uint8_t>& cell_is_void_host,
    const double* d_r_nodes, int n_nodes,
    cudaStream_t stream,
    std::vector<double>& dep_power_cell);

// Shorthand-resolution overload (kept: the unit-tested single-channel equivalence surface).
DepositResult deposit_hot_electrons_cone_1d_device(
    const core::Config::LaserConfig::HotElectronConfig& cfg,
    Geometry1D geom,
    const std::vector<HotESource>& sources,
    const double* d_rho, const double* d_zbar, const double* d_A_eff,
    const double* d_Te_eV, const std::vector<std::uint8_t>& cell_is_void_host,
    const double* d_r_nodes, int n_nodes,
    cudaStream_t stream,
    std::vector<double>& dep_power_cell);

}  // namespace tenryu::laser::hot_electron
