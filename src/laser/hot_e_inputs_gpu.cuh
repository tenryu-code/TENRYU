#pragma once

// The inputs of the hot-electron eta model of the 1D laser step (NUMERICS §5.11.3), computed on
// the device: per cell the electron density, n_e / n_c and the cell centre; per channel the
// evaluation surface (the outermost crossing of n_e = eval_nc_fraction n_c), the electron
// temperature there, the shell nearest to it and the density scale |d ln n_e / dr| from the
// weighted fit (hot_e_eta::fit_kappa_abs_um); in port_section, the reference beam's angular profile
// at that shell from the device phase-space table, the illumination metric of the port layout
// (port_section::illumination_at_shell) and the common-wave drive with its sky map
// (port_section::common_wave_drive). These ran on the host every step (laser.cu laser_step on the
// hydro mirror and the host tables, which stay as the references of the tests). The arithmetic
// follows the host's operation by operation without floating-point contraction; the results differ
// only by the last-place differences of the device transcendental functions, and the decisions
// they feed (a common-wave cluster's members, a profile bin) can change only where a value lies
// within such a difference of the decision's threshold.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/state.hpp"
#include "laser/port_geometry.hpp"
#include "laser/port_section_s1_gpu.cuh"

namespace tenryu::laser::hot_e_inputs {

// What a channel needs this step.
struct ChannelSpec {
  double eval_nc_fraction = 0.25;
  bool illumination = false;   // port_section with illumination_metric = "equivalent_area"
  bool common_wave = false;    // port_section TPD channel with tpd_overlap_mode = "common_wave_cluster"
  bool sky_map = false;        // the channel whose common-wave grid the snapshot keeps
  double delta_theta_deg = 0.0;
};

// Per-channel results read back for the host eta update.
struct ChannelResult {
  bool valid = false;            // an evaluation surface and a positive fitted scale exist
  double Te_s_eV = 0.0;
  double kappa_um = 0.0;
  double eval_radius_cm = 0.0;
  int eval_shell = -1;
  // illumination (when requested and the table has the shell)
  bool illumination_valid = false;
  double f_illum2 = 0.0;
  double f_union = 0.0;
  // common-wave drive (when requested or for the sky map)
  bool drive_computed = false;
  double I_drive = 0.0;
  double I_lower = 0.0;
  double I_upper = 0.0;
  int n_sigma_mode = 0;
};

// Grid sizes of the host functions (port_section_overlap.hpp defaults).
constexpr int kIlluminationMuProfile = 64;
constexpr int kIlluminationMuGrid = 128;
constexpr int kIlluminationPhiGrid = 64;
constexpr double kIlluminationCutFraction = 0.01;
constexpr int kCommonWaveMuGrid = 64;
constexpr int kCommonWavePhiGrid = 32;
constexpr int kCommonWaveMuProfile = 64;

class Workspace {
 public:
  Workspace();
  ~Workspace();
  Workspace(const Workspace&) = delete;
  Workspace& operator=(const Workspace&) = delete;
  struct Impl;
  Impl* impl() const { return impl_; }

 private:
  Impl* impl_ = nullptr;
};

// Per cell: n_e (zero in void cells), n_hat = n_e / max(n_crit, 1e-30) and the cell centre, from the
// device state and the cells' effective mass number A_eff (device, [n_cells]).
void stage_cells(Workspace& ws, const core::State& state, const double* A_eff, double n_crit,
                 cudaStream_t stream);

// The cell arrays of stage_cells (device, valid until the next stage_cells).
const double* cell_n_hat(const Workspace& ws);
const double* cell_r(const Workspace& ws);

// The ports of the layout (sorted as the PortTable), uploaded when they change.
void stage_ports(Workspace& ws, const port_geom::PortTable& ports);

// Per channel: the evaluation surface (n_e = eval_nc_fraction max(n_crit, 1e-30)), Te, nearest
// shell and fitted scale; with a device table (nullptr when no table was built yet) and an
// evaluation shell, the illumination and common-wave drive the specs ask for (requires
// stage_ports). One readback for all channels.
std::vector<ChannelResult> evaluate_channels(Workspace& ws, const core::State& state,
                                             const std::vector<ChannelSpec>& specs,
                                             const port_section::S1DeviceTable* table,
                                             double n_crit, cudaStream_t stream);

// The sky map of the channel with sky_map = true, when the last evaluate_channels computed it: the
// mu-major grids [kCommonWaveMuGrid * kCommonWavePhiGrid] of I_tot (the per-point sum over the
// ports), I_cw and the winning cluster's member count, in erg/s/cm^2, copied device to device into
// the caller's arrays. False (nothing copied) when the last call computed none.
bool copy_sky_map(const Workspace& ws, double* I_tot, double* I_cw, double* n_sigma,
                  cudaStream_t stream);

// The sky grid's mu and phi nodes (host copies; empty before the first sky map).
const std::vector<double>& sky_mu(const Workspace& ws);
const std::vector<double>& sky_phi(const Workspace& ws);

}  // namespace tenryu::laser::hot_e_inputs
