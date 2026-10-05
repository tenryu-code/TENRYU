#pragma once

// 1D CBET stages that ran on the host every step, on the device (NUMERICS §5.10): the effective
// mass number and the plasma pack of the cells (formerly cbet_stage_cell_fields on the hydro
// mirror, which stays as the reference of the tests), the per-cell exchange maps of the snapshot
// with their closure check (formerly fill_cbet_viz_fields in laser.cu), and the port_section
// outputs after the solve: the outgoing power per port and the hot-electron capture sums
// (formerly host loops in laser.cu). The file is compiled without floating-point contraction
// (-fmad=false), so the device arithmetic repeats the host's operation by operation.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "laser/cbet.cuh"

namespace tenryu::laser {

// ws.cell_A_eff[c]: the effective mass number of cell c as laser_mesh.cu compute_cell_effective_A
// gives it — the volume-fraction-weighted harmonic mean of the materials' A, or material_A when the
// state carries no volume fractions or the cell's mean is not a positive finite number.
void cbet_stage_cell_A_eff_device(CbetWorkspace& ws, const core::State& state, double material_A,
                                  const std::vector<double>& material_A_list,
                                  cudaStream_t stream);

// The per-cell plasma pack of the 1D solve (cell_chi_pref, cell_c_a, cell_u_r, cell_k_bar,
// cell_vol, cell_mask; cbet_stage_cell_fields) from the device state and ws.cell_A_eff.
void cbet_stage_cell_fields_device(CbetWorkspace& ws, const core::State& state,
                                   const core::Config::LaserConfig& laser, double lambda0_cm,
                                   cudaStream_t stream);

// Flags of the step stages, read back by cbet_read_step_outputs.
enum CbetStepFlag : int {
  kCbetFlagClosure = 1 << 0,       // a cell's final dQ + IAW violates the action closure
  kCbetFlagCellVolume = 1 << 1,    // a cell volume is not positive and finite
  kCbetFlagRecordOffsets = 1 << 2, // a ray's record offsets disagree with its record count
  kCbetFlagCaptureChannel = 1 << 3,// a capture row names a channel outside the configuration
  kCbetFlagCaptureCell = 1 << 4,   // a capture row names a cell outside the mesh
};

// Prepares the step's readback for n_config_channels capture channels and clears its flags; call
// once per step before cbet_viz_fields_device and ps_outputs_device.
void cbet_begin_step_outputs(CbetWorkspace& ws, int n_config_channels, cudaStream_t stream);

// The exchange maps of the final iteration (formerly fill_cbet_viz_fields): per cell
// ws.viz_gross = 0.5 sum_g |dQ| / V and ws.viz_net_inbound = (sum of dQ over the inbound groups) / V
// [erg/s/cm^3], with the closure check |sum_g dQ + IAW| <= 1e-9 max(sum_g |dQ| + |IAW|, 1e-300)
// (IAW = ws.iaw_cell in port_section, zero otherwise) and the volume check. The groups are
// g = 0 .. G-1; group g is inbound when (g mod (n_branches n_bins)) / n_bins == 0.
void cbet_viz_fields_device(CbetWorkspace& ws, int G, int n_branches, int n_bins,
                            bool port_section, cudaStream_t stream);

// port_section after the solve: ws.ps_outgoing[port] = the final record weight of every ray of the
// port, summed in ray order; with hot-electron capture, ws.ps_capture_pcross[port * n_config +
// channel] and the per-channel capture sums (read back by cbet_read_step_outputs), accumulated in
// the order (port, ray, capture channel) of the former host loop. Cell radii are the centres of
// the node radii x_r. Without capture, ws.ps_capture_pcross is zero.
void ps_outputs_device(CbetWorkspace& ws, const double* x_r, int n_cells, int n_config_channels,
                       bool capture_on, cudaStream_t stream);

// The step's small results, read back with one copy and one synchronization.
struct CbetStepReadback {
  int flags = 0;
  double dq_abs_max = 0.0;  // max over cells and groups of |dQ|
  double dq_abs_sum = 0.0;  // sum over cells (in cell order) of sum_g |dQ|
  std::vector<double> capture_sum_P;    // [n_config_channels] sum of P_cross
  std::vector<double> capture_sum_Pr;   // [n_config_channels] sum of r_cell P_cross
  std::vector<double> capture_sum_Pmu;  // [n_config_channels] sum of mu P_cross
  double capture_banked_power = 0.0;    // sum of P_cross (1 - (1 - eta)) over the rows
};

CbetStepReadback cbet_read_step_outputs(CbetWorkspace& ws, int n_config_channels,
                                        cudaStream_t stream);

// Copies the outputs that stay on the device (the exchange maps; the port_section outgoing power
// and capture per port) into the State, for a snapshot or a checkpoint. Outputs not computed yet
// leave the State's vectors as they are.
void cbet_sync_output_fields(core::State& state, const CbetWorkspace& ws);

}  // namespace tenryu::laser
