#pragma once

// The 1D hot-electron transport of the laser step on the device (laser.cu laser_step after the
// trace, NUMERICS §5.11): the trace's capture rows of every beam kept on the device; per config
// channel the model sums (the captured power and its radius moment) and the captures of the
// channels that make hot electrons, reduced to sources on (cell, axis-angle bin) aggregates
// (hot_electron::reduce_captures); the cone chords of each source (the band quadrature of
// hot_electron::build_band_nodes, the tables of the channel's spec made once on the host) marched by
// hot_electron::cone_chords_device, or the radial march (hot_electron::deposit_hot_electrons_
// radial_1d); the power per cell; and the per-cell preheat diagnostics with the explicit-source dt
// limit. The captures, the sources, the radial march and the diagnostics ran on the host every step
// on copies of the capture rows and of the plasma fields, and the cone's chord list was built there.
// The arithmetic repeats the host's operation by operation without floating-point contraction (the
// file is compiled with -fmad=false), the sums the host accumulated in an order accumulated in the
// same order by one thread: the cone's power and sums in the order of the device pipeline laser_step
// ran (hot_electron::deposit_hot_electrons_cone_1d_device), the radial march's in the order of the
// host march. The radial march's stopping power now runs on the device, whose erf, exp and log can
// differ from the host's in the last place.

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/state.hpp"
#include "laser/hot_electron_1d.cuh"

namespace tenryu::laser::hot_e_transport_1d {

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

// A step's capture lists of the n_config_channels channels emptied and their model sums zeroed.
void begin_step(Workspace& ws, int n_config_channels, cudaStream_t stream);

// The trace's capture staging of the step: rows [n_beams][rays_per_beam][n_capture_channels][4]
// {hit, r, mu, P} zeroed.
void begin_trace(Workspace& ws, int n_capture_channels, int n_beams, int rays_per_beam,
                 cudaStream_t stream);

// A traced beam's capture rows (device [n_rays][n_capture_channels][4]) into the staging.
void stage_beam(Workspace& ws, int beam, const double* rows, int n_rays, cudaStream_t stream);
// A folded beam: the anchor beam's rows copied to it.
void replay_beam(Workspace& ws, int anchor_beam, int beam, cudaStream_t stream);

// The staged rows in the order beam, ray, capture channel: per config channel (config_index of each
// capture channel) the model sums P and r P when model_sums is set, and the captures {r, mu,
// eta_eff P} of the capture channels with eta_eff > 0 appended to the channel's list.
void collect_trace_captures(Workspace& ws, const std::vector<int>& config_index,
                            const std::vector<double>& eta_eff, bool model_sums,
                            cudaStream_t stream);

// Captures given on the host (radial_absorption_1d, port_section), the channel's list.
void add_host_captures(Workspace& ws, int config_channel,
                       const std::vector<hot_electron::RayCapture>& captures);

// The model sums of collect_trace_captures (P and r P per config channel, zero without a trace
// collection) and whether any channel has a capture; one synchronisation of `stream`.
void read_capture_state(Workspace& ws, std::vector<double>* sum_P, std::vector<double>* sum_Pr,
                        bool* any_captures, cudaStream_t stream);

struct ChannelResult {
  bool active = false;
  int n_sources = 0;
  double P_hot = 0.0;
  double P_deposited = 0.0;
  double P_escaped = 0.0;
  double P_residual_inner = 0.0;
  double r_source_mean = 0.0;
  double conservation_resid = 0.0;
  int substep_cap_hits = 0;
};

// The transport of every config channel with captures, in channel order, into the power per cell
// (device [n_cells], zeroed first; power_cell below), with the host's warnings (captures outside
// the hydro mesh, the conservation check) in the host's order: the spec of each channel, the
// mesh's geometry_code (0 spherical, 1 cylindrical, 2 planar) and the angular model (cone, or
// radial). A_eff_cone is the A_eff the cone chords read (the State's), and A_eff_radial the one the
// radial march reads (the laser's effective mass number). Device state: rho, zbar, Te, x_r, the
// void mask. Returns the per-channel results (one synchronisation per channel for its sizes).
std::vector<ChannelResult> transport(Workspace& ws, const core::State& state,
                                     const std::vector<hot_electron::HotEChannelSpec>& specs,
                                     int geometry_code, bool cone, const double* A_eff_cone,
                                     const double* A_eff_radial, cudaStream_t stream);

// The hot-electron power per cell of the last transport (device [n_cells]).
const double* power_cell(const Workspace& ws);

// laser_step's per-cell diagnostics of the last transport: Q = P / vol where P > 0 and vol > 0 (0
// elsewhere), eps_cum += P dt / (rho vol), and the explicit-source dt limit min(limit rho vol
// max(ee, 0) / P). eps_cum is kept on the device, uploaded from its host copy when something else
// changed that copy (a restart, a retried or rolled-back step); Q and eps_cum come back into
// state.hot_e_Q_host and state.hot_e_eps_cum_host. Returns the dt limit (infinity when no cell
// limits it).
double diagnostics(Workspace& ws, core::State& state, double dt, double explicit_source_limit,
                   cudaStream_t stream);

}  // namespace tenryu::laser::hot_e_transport_1d
