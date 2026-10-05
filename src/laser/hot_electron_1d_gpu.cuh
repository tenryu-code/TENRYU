#pragma once

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "laser/hot_electron_1d.cuh"

namespace tenryu::laser::hot_electron {

// One chord of the cone pipeline: a source's launch radius (planar: x), the direction cosine to the
// radial (slab) axis, and the power the chord carries (the source's P_hot times the node weight).
struct ConeChordJob {
  double r_s;
  double mu_dir;
  double P_chord;
};

// The chords of device jobs [n_jobs], one thread per (job, group): each marches its chord through
// the cells and writes the power it deposits in each cell into its row (rows [n_jobs * n_groups]
// [n_cells], zeroed here), the power that leaves the mesh into escaped [n_rows] and its RK
// substep-cap hits into caps [n_rows]; then out_cell [n_cells] = the rows summed per cell in row
// order. Device inputs: groups [n_groups], the plasma fields [n_cells], the void mask [n_cells] and
// the nodes [n_nodes].
void cone_chords_device(const ConeChordJob* jobs, int n_jobs, const GroupSpec* groups,
                        int n_groups, const double* rho, const double* zbar, const double* A_eff,
                        const double* Te_eV, const std::uint8_t* cell_is_void,
                        const double* r_nodes, int n_nodes, bool planar, double T_h_erg,
                        double* rows, double* escaped, int* caps, double* out_cell,
                        cudaStream_t stream);

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
