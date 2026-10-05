#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/state.hpp"
#include "laser/cbet_lm_fields.cuh"

namespace tenryu::laser {

// CBET v1 (design: the internal design note cbet_1d_v1_design_20260707.md).
// Conservative pairwise cell exchange between angular ray groups on the 1D_SPH
// ray bundle, with a 2D_RZ mode that recomputes pair coupling and deposits to
// LaserMesh nodes. All floating-point tallies are atomic-free / fixed-order and
// bitwise deterministic for a fixed build+device; the only atomics are integer
// event counters (order-independent sums) and the 2D final node deposit
// (atomicAdd(double), covered by the documented 2D noise-band gates).

struct CbetSolveResult {
  int iterations = 0;
  bool converged = false;
  double conv_final = 0.0;
  double exchanged_power = 0.0;      // erg/s, sum over pairs of |exchange| (last iter)
  double E_iaw_rate = 0.0;           // erg/s, signed port_section IAW ledger rate
  double ledger_residual_rel = 0.0;  // |sum dQ| / max(sum |dQ|, tiny)
  unsigned long long clamp_count = 0;
  unsigned long long capped_pairs = 0;
  long long overflow_rays = 0;
  double clamped_power = 0.0;  // erg/s injected by positivity clamps in the final pass
  std::vector<double> dq_host;  // [n_cells*dq_G], final-iteration net exchange
  int dq_G = 0;
  int dq_G_ref = 0;
  int dq_n_ports = 0;
  int dq_n_bins = 0;
  int dq_n_branches = 0;
  // In 1D, build_keys maps rec_mu <= 0 to branch 0 (inbound); for state g,
  // branch = (g % (dq_n_branches*dq_n_bins)) / dq_n_bins.
};

struct CbetPortSectionArgs {
  int n_ports;
  const double* chi_host;         // [n_cells * n_pairs_ps] from build_chi_ps
  const double* chi_device = nullptr;  // device ptr; when set, chi_host may be null and no H2D occurs
  const double* omega_state_host; // [G_ps]
  const double* port_weight_host; // [n_ports]
  const std::int16_t* pair_p;     // [n_pairs_ps]
  const std::int16_t* pair_q;     // [n_pairs_ps]
  const std::int32_t* pair_index; // [G_ps * G_ps]
  double* traj_power = nullptr;               // [traj_n_output_rays * traj_max_steps]
  const std::int32_t* traj_rec_idx = nullptr; // parallel to traj_power, device-only
  const int* traj_step_count = nullptr;       // [traj_n_output_rays]
  double* traj_rec_ratio = nullptr;           // [traj_n_output_rays * cap_per_ray]
  long long traj_ray_offset = 0;
  int traj_n_output_rays = 0;
  int traj_output_stride = 1;
  int traj_max_steps = 0;
  int viz_port = 0;
};

struct CbetWorkspace {
  // geometry of the problem staged into the workspace
  int n_rays_total = 0;
  int cap_per_ray = 0;
  int n_cells = 0;
  int n_beams = 0;
  int n_bins = 0;
  int n_branches = 2;  // direction branches per (beam,bin): 1D sign(mu)=2, 2D quadrant of (a,c)=4
  bool dim2d = false;  // 2D_RZ mode: node-deposit output path, no chi cache, no dep_rows
  int n_nodes = 0;     // 2D: laser-mesh node count (deposit rows are per node)
  int nz_cells = 0;    // 2D: LM cells along z; cell id = ci * nz_cells + cj
  int n_nodes_z = 0;   // 2D: node stride; node(i,j) = i * n_nodes_z + j
  int n_groups = 0;  // state G (legacy beam groups; ps_mode = n_ports*G_ref)
  int n_pairs = 0;   // G*(G-1)/2
  long long n_records_live = 0;

  // --- port_section expanded-state mode (1D only; design §13) ---
  bool ps_mode = false;
  int n_ports = 0;  // canonical port count
  int G_ref = 0;    // reference groups = n_branches * n_bins (keys/segments stay G_ref)
  double* rec_w_ps = nullptr;    // [n_ports * n_rays_total * cap_per_ray]
  double* rec_em1 = nullptr;     // [n_rays_total * cap_per_ray] expm1(-S/2) of each record
  // final pass: the IB deposit of each half record [n_ports * live records * 2] and the
  // unabsorbed and clamp-injected power of each (port, ray) [n_ports * n_rays_total], added to the
  // rows in port order by cbet_ps_deposit_rows_kernel
  double* ps_dep_terms = nullptr;
  double* ps_port_unabs = nullptr;
  double* ps_port_clamp = nullptr;
  double* port_weight = nullptr; // [n_ports] device copy of power weights
  // ps-mode hot-e capture (port_section + eta_mode=model only)
  int ps_n_channels = 0;                 // config channels (<= kMaxSources)
  double* ps_cell_nhat = nullptr;        // [n_cells] staged n_e/n_crit
  double* ps_capture_thresh = nullptr;   // [ps_n_channels] f_s ascending order
  std::int32_t* ps_capture_order = nullptr; // [ps_n_channels] config index in ascending-f_s order
  double* ps_one_minus_eta = nullptr;    // [ps_n_channels] in the SAME ascending order
  double* ps_capture_stage = nullptr;    // [n_ports * n_rays_total * ps_n_channels * 4]
                                         // rows {hit, P_cross_raw, mu_at_capture, r_hint}

  // per-record SoA stripes [n_rays_total * cap_per_ray]
  std::int32_t* rec_cell = nullptr;
  float* rec_mu = nullptr;
  double* rec_ds = nullptr;
  double* rec_S = nullptr;
  double* rec_w = nullptr;
  // per-record extra stripes (2D only; nullptr in 1D)
  float* rec_c = nullptr;    // axial direction cosine c = dz/ds (rec_mu holds radial a = dR/ds)
  float* rec_w00 = nullptr;  // ds-weighted mean bilinear corner weights of the record's cell
  float* rec_w10 = nullptr;  //   (BilinearWeights convention of the record's (cell.i, cell.j);
  float* rec_w01 = nullptr;  //    w11 is derived as max(0, 1 - w00 - w10 - w01))
  // per-ray meta [n_rays_total]
  std::int32_t* rec_count = nullptr;
  std::int32_t* ray_group_base = nullptr;  // beam*(n_branches*n_bins) + bin
  double* ray_P0 = nullptr;
  std::uint8_t* ray_overflow = nullptr;
  // sort/segment view
  std::int64_t* ray_rec_offset = nullptr;  // [n_rays_total + 1] exclusive scan of rec_count
  std::int64_t* sort_key = nullptr;        // [live records]
  std::int32_t* sort_slot = nullptr;       // [live records]
  std::int64_t* seg_offsets = nullptr;     // [G*n_cells + 1]
  // per-(g,c) tallies, layout idx = c*G + g
  double* L = nullptr;
  double* L_prev = nullptr;
  double* Mmu = nullptr;
  // per-(g,c) axial moment (2D only)
  double* Mc = nullptr;
  double* ds_max_gc = nullptr;
  double* fcap = nullptr;
  double* dQ = nullptr;
  double* iaw_partial = nullptr;  // [n_cells*n_groups], gainer-only IAW rate
  double* exch_partial = nullptr;
  double* iaw_cell = nullptr;  // [n_cells], signed port_section IAW rate [erg/s]
  // pair tables
  double* chi = nullptr;            // [n_cells * n_pairs]
  std::int16_t* pair_p = nullptr;   // [n_pairs]
  std::int16_t* pair_q = nullptr;   // [n_pairs]
  std::int32_t* pair_index = nullptr;  // [G*G] -> pair id (or -1 on diagonal)
  double* omega_group = nullptr;    // [G]
  // per-cell plasma pack [n_cells]
  double* cell_chi_pref = nullptr;  // (lambda0 e^2/(c^3 m_e)) * (nh/(1-nh)) * <Z>/(<Z>Te+3Ti) [cm s/erg]
  double* cell_c_a = nullptr;       // [cm/s]
  double* cell_u_r = nullptr;       // [cm/s]
  // per-cell axial flow (2D only; cell_u_r then holds the radial component)
  double* cell_u_z = nullptr;
  double* cell_k_bar = nullptr;     // n_refr * omega0 / c [1/cm]
  double* cell_vol = nullptr;       // [cm^3]
  std::uint8_t* cell_mask = nullptr;
  // deposit outputs
  double* dep_rows = nullptr;    // [n_rays_total * n_cells]
  // 2D deposit output: per-beam node rows [n_beams * n_nodes] (atomic corner adds)
  double* dep_nodes = nullptr;
  double* unabs_rows = nullptr;  // [n_rays_total]
  double* clamp_rows = nullptr;  // [n_rays_total] final-pass clamp-injected power per ray
  // device scalars/counters
  double* d_scalars = nullptr;                 // [8]
  double* stats_partial = nullptr;             // [5 * 256] partial sums of the iteration statistics
  double* d_iaw_sum = nullptr;                 // [1]
  unsigned long long* d_counters = nullptr;    // [4]
  // host-side beam bookkeeping
  std::vector<int> beam_ray_offset;
  std::vector<int> beam_ray_count;
  std::vector<double> beam_omega;

  // grow-only capacities (element counts). pair_p/pair_q [n_pairs], pair_index
  // [G*G], and omega_group [G] have different lengths and MUST track separate
  // capacities: a single shared capacity lets a modest G growth skip the
  // reallocation of the shorter arrays while the H2D staging writes the new
  // (larger) element counts — a device buffer overrun (2026-07-26 review).
  std::size_t cap_records = 0;
  std::size_t cap_rays = 0;
  std::size_t cap_gc = 0;
  std::size_t cap_chi = 0;
  std::size_t cap_pair_list = 0;   // pair_p / pair_q [n_pairs]
  std::size_t cap_pair_index = 0;  // pair_index [G*G]
  std::size_t cap_omega = 0;       // omega_group [G]
  std::size_t cap_cells = 0;
  std::size_t cap_rows = 0;
  std::size_t cap_dep_nodes = 0;
  std::size_t cap_rec_w_ps = 0;
  std::size_t cap_rec_em1 = 0;
  std::size_t cap_ps_dep_terms = 0;
  std::size_t cap_ps_port_unabs = 0;
  std::size_t cap_ps_port_clamp = 0;
  std::size_t cap_port_weight = 0;
  std::size_t cap_ps_cell_nhat = 0;
  std::size_t cap_ps_capture_thresh = 0;
  std::size_t cap_ps_capture_order = 0;
  std::size_t cap_ps_one_minus_eta = 0;
  std::size_t cap_ps_capture_stage = 0;

  // --- 1D step stages on the device and their outputs (cbet_stage_gpu.cuh) ---
  double* cell_A_eff = nullptr;          // [n_cells] effective mass number of the cell
  double* material_A_device = nullptr;   // [n_materials] the laser mesh's material A list
  std::vector<double> material_A_uploaded;  // host copy of material_A_device
  double* viz_gross = nullptr;           // [n_cells] 0.5 sum_g |dQ| / V of the final iteration
  double* viz_net_inbound = nullptr;     // [n_cells] sum of dQ over the inbound groups / V
  double* viz_dq_abs_cell = nullptr;     // [n_cells] sum_g |dQ| (audit)
  double* viz_dq_max_cell = nullptr;     // [n_cells] max_g |dQ| (audit)
  double* ps_outgoing = nullptr;         // [n_ports] outgoing power per port [erg/s]
  double* ps_capture_pcross = nullptr;   // [n_ports * n config channels] captured power [erg/s]
  std::int32_t* ps_capture_row_config = nullptr;  // [capture rows] config channel a row adds to, or -1
  double* step_out = nullptr;            // device side of the step readback (cbet_read_step_outputs)
  int* step_flags = nullptr;             // device error flags of the step stages
  double* step_out_host = nullptr;       // pinned host side of step_out
  int* step_flags_host = nullptr;        // pinned host side of step_flags
  std::size_t cap_cell_outputs = 0;      // cell_A_eff
  std::size_t cap_viz = 0;               // viz_gross, viz_net_inbound, viz_dq_abs_cell, viz_dq_max_cell
  std::size_t cap_material_A = 0;
  std::size_t cap_ps_outgoing = 0;
  std::size_t cap_ps_capture_pcross = 0;
  std::size_t cap_ps_capture_row_config = 0;
  std::size_t cap_step_out = 0;
  int ps_capture_config_channels = 0;    // config channels of ps_capture_pcross
  bool viz_valid = false;                // viz_* hold a step's maps
  bool ps_outputs_valid = false;         // ps_outgoing / ps_capture_pcross hold a step's values
  // port_section tables staged for the solve: uploaded again only when the values or the device
  // array change (a legacy-mode preparation writes the same arrays and resets these).
  template <typename T>
  struct StagedUpload {
    std::vector<T> values;
    const void* device = nullptr;
  };
  StagedUpload<double> ps_staged_omega;
  StagedUpload<double> ps_staged_weight;
  StagedUpload<std::int16_t> ps_staged_pair_p;
  StagedUpload<std::int16_t> ps_staged_pair_q;
  StagedUpload<std::int32_t> ps_staged_pair_index;
  // Legacy mode: the beams' frequencies on the device, from which a kernel fills omega_group.
  double* beam_omega_device = nullptr;   // [n_beams]
  std::size_t cap_beam_omega = 0;
  StagedUpload<double> legacy_staged_beam_omega;
  // The solve's scalar readbacks: the record total after the scan (device), and the pinned host
  // side of the per-iteration statistics and of the final counters and sums.
  struct SolveReadback {
    std::int64_t records_total;
    double stats[5];
    unsigned long long counters[4];  // [0] clamps, [1] capped pairs, [2] overflowed rays
    double iaw_sum;
    double clamped_power;
  };
  std::int64_t* d_records_total = nullptr;  // [1]
  SolveReadback* solve_host = nullptr;      // pinned
  void reset_ps_staged_uploads() {
    ps_staged_omega = {};
    ps_staged_weight = {};
    ps_staged_pair_p = {};
    ps_staged_pair_q = {};
    ps_staged_pair_index = {};
  }

  CbetWorkspace() = default;
  ~CbetWorkspace();
  CbetWorkspace(const CbetWorkspace&) = delete;
  CbetWorkspace& operator=(const CbetWorkspace&) = delete;
  CbetWorkspace(CbetWorkspace&&) = delete;
  CbetWorkspace& operator=(CbetWorkspace&&) = delete;

  void release();
};

// Signed 2D_RZ pair coupling for orientation "p gains from q when positive".
// Exactly odd under (p<->q): base_cos/bb are symmetric, domega and kdotu negate
// exactly, P(g) is FP-odd. Two-point Z2 closure over the mirror-symmetric azimuthal
// residual b = sqrt(1 - a^2 - c^2) (spec §2.3) replaces the 1D n_phi quadrature.
TENRYU_HOST_DEVICE inline double cbet_pair_coupling_2d(const double a_p,
                                        const double c_p,
                                        const double a_q,
                                        const double c_q,
                                        const double omega_p,
                                        const double omega_q,
                                        const double chi_pref,
                                        const double c_a,
                                        const double u_r,
                                        const double u_z,
                                        const double k_bar,
                                        const double f_cbet,
                                        const double alpha_iaw,
                                        const double k_a_floor,
                                        const double test_chi) {
  if (test_chi > 0.0) {
    return test_chi;
  }
  const double b_p = ::sqrt(::fmax(0.0, 1.0 - a_p * a_p - c_p * c_p));
  const double b_q = ::sqrt(::fmax(0.0, 1.0 - a_q * a_q - c_q * c_q));
  const double domega = omega_q - omega_p;
  const double kdotu = k_bar * ((a_q - a_p) * u_r + (c_q - c_p) * u_z);
  const double ca_safe = ::fmax(c_a, 1.0e-30);
  const double base_cos = a_p * a_q + c_p * c_q;
  const double bb = b_p * b_q;
  double acc = 0.0;
#pragma unroll
  for (int s = 0; s < 2; ++s) {
    const double cospsi = (s == 0) ? (base_cos + bb) : (base_cos - bb);
    const double ka2 = ::fmax(2.0 * (1.0 - cospsi), 0.0);
    const double ka = k_bar * ::sqrt(ka2);
    if (!(ka >= k_a_floor * k_bar) || !(ka > 0.0)) {
      continue;
    }
    const double g = (domega - kdotu) / (ka * ca_safe);
    const double ga = g * alpha_iaw;
    const double one_g2 = 1.0 - g * g;
    const double denom = ga * ga + one_g2 * one_g2;
    const double Pg = ga / denom;
    const double pol = 0.25 * f_cbet * (1.0 + cospsi * cospsi);
    acc += pol * Pg;
  }
  return chi_pref * (0.5 * acc);
}

CbetWorkspace& global_cbet_workspace();
void invalidate_global_cbet_workspace();

// Sizes + allocates (grow-only) all buffers and host-builds the static pair
// tables (pair_p/pair_q/pair_index) and beam bookkeeping storage. Zeroes
// rec_count and ray_overflow. Must be called before recording rays.
void cbet_workspace_prepare(CbetWorkspace& ws,
                            int n_rays_total,
                            int cap_per_ray,
                            int n_cells,
                            int n_beams,
                            int n_impact_bins,
                            cudaStream_t stream = nullptr,
                            int n_branches = 2,
                            bool dim2d = false,
                            int n_nodes = 0,
                            int nz_cells = 0,
                            int n_nodes_z = 0,
                            int n_ports = 0,
                            int ps_n_channels = 0);

// Registers one beam's rays: fills ray_group_base for [ray_offset, ray_offset+count)
// (bin = local_k * n_bins / count), copies device ray powers into ray_P0, and
// records beam omega (2*pi*c/lambda_beam_cm).
void cbet_stage_ray_meta(CbetWorkspace& ws,
                         int beam_index,
                         int ray_offset,
                         int count,
                         double omega_beam,
                         const double* d_ray_power,
                         cudaStream_t stream = nullptr);

// Constants of the per-cell plasma pack for a laser wavelength (shared by the host and device
// stages so that both use the same values).
struct CbetCellFieldConstants {
  double c = 0.0;            // speed of light [cm/s]
  double eV = 0.0;           // erg per eV
  double omega0 = 0.0;       // laser angular frequency [1/s]
  double n_crit = 0.0;       // critical electron density [cm^-3]
  double lam_pref = 0.0;     // lambda0 e^2 / (c^3 m_e) [cm s]
  double proton_mass = 0.0;  // [g]
};
CbetCellFieldConstants cbet_cell_field_constants(double lambda0_cm);

// Builds the per-cell plasma pack from the 1D hydro mirror + State (Ti, node
// velocities, cell volumes are copied device->host->device here; all cgs+eV).
// The production 1D step uses cbet_stage_cell_fields_device (cbet_stage_gpu.cuh); this host
// version is the reference of its tests.
void cbet_stage_cell_fields(CbetWorkspace& ws,
                            const core::State& state,
                            const HydroMirror1D& mirror,
                            const core::Config::LaserConfig& laser,
                            double lambda0_cm,
                            cudaStream_t stream = nullptr);

// Builds the per-LM-cell plasma pack for the 2D solve from host node fields
// (4-corner means; torus volumes; coverage/void/cutoff mask). All cgs+eV.
void cbet_stage_cell_fields_2d(CbetWorkspace& ws,
                               const CbetLmFields& lm,
                               const core::Config::LaserConfig& laser,
                               double lambda0_cm,
                               cudaStream_t stream = nullptr);

// Runs the fixed-point exchange iteration and the final depositing propagate.
// On return dep_rows/unabs_rows hold per-ray absorbed power [erg/s] per cell and
// terminal unabsorbed power [erg/s]. rec_* stripes must be staged beforehand.
CbetSolveResult cbet_solve_and_deposit(CbetWorkspace& ws,
                                       const core::Config::LaserConfig::CbetConfig& cfg,
                                       cudaStream_t stream = nullptr,
                                       const CbetPortSectionArgs* ps = nullptr,
                                       bool collect_dq = false);

// Deterministically sums rows[first .. first+count) and ADDS the result into *d_out
// (single-block strided fixed-order reduction; bitwise stable for fixed build+device).
void cbet_sum_rows_add(const double* d_rows, int first, int count, double* d_out,
                       cudaStream_t stream = nullptr);

}  // namespace tenryu::laser
