#pragma once

#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/cap_energy_audit.hpp"

namespace tenryu::hydro {
struct HydroEOSContext;
}

namespace tenryu::parallel {
class Reduction;
}

namespace tenryu::hydro::ale {

struct CsrConsAuditContext {
  bool enabled = false;
  const tenryu::parallel::Reduction* reduction = nullptr;
  int step = 0;
  double t = 0.0;
  double dt = 0.0;
  int remap_id = 0;
  int trigger_cell = -1;
  double min_path_margin = 0.0;
};

bool csr_cons_audit_env_enabled();
void set_csr_cons_audit_context(const CsrConsAuditContext& context);
void clear_csr_cons_audit_context();

#if defined(__CUDACC__)
__device__ double rz_limited_slope_r(int i,
                                     int j,
                                     int nr,
                                     int nz,
                                     const double* Q,
                                     const double* x_r,
                                     const std::uint8_t* cell_nverts);
__device__ double rz_limited_slope_z(int i,
                                     int j,
                                     int nr,
                                     int nz,
                                     const double* Q,
                                     const double* x_z,
                                     const std::uint8_t* cell_nverts);
__device__ double rz_barth_jespersen_limiter(double slope,
                                             double Q_K,
                                             double Q_min,
                                             double Q_max,
                                             double dx);
__device__ double rz_reconstructed_face_value_r(int K,
                                                int Kp,
                                                double swept_volume,
                                                const double* Q,
                                                const double* x_r_lag,
                                                const double* x_z_lag,
                                                const double* x_r_ref,
                                                const double* x_z_ref,
                                                int i_face,
                                                int j,
                                                int nr,
                                                int nz,
                                                double Q_floor,
                                                const std::uint8_t* cell_nverts);
__device__ double rz_reconstructed_face_value_z(int K,
                                                int Kp,
                                                double swept_volume,
                                                const double* Q,
                                                const double* x_r_lag,
                                                const double* x_z_lag,
                                                const double* x_r_ref,
                                                const double* x_z_ref,
                                                int i,
                                                int j_face,
                                                int nr,
                                                int nz,
                                                double Q_floor,
                                                const std::uint8_t* cell_nverts);
#endif

struct AleRemap2DRZResult {
  bool applied = false;
  // Total stored-mass closure of this remap: (sum_post - sum_pre)/sum_pre
  // over ALL cells, measured unconditionally around the CSR remap body. A
  // committed remap must close at roundoff; a violation marks fabricated
  // mass (measured: +1.119e-1 total in ONE center-patch Winslow fire) and
  // the driver rejects the step via the full-step retry snapshot.
  double mass_closure_rel = 0.0;
  double boundary_mass_flux_z_bottom = 0.0;
  double boundary_mass_flux_z_top = 0.0;
  double boundary_momentum_flux_z_bottom = 0.0;
  double boundary_momentum_flux_z_top = 0.0;
  double boundary_energy_flux_z_bottom = 0.0;
  double boundary_energy_flux_z_top = 0.0;
  double mass_floor_delta = 0.0;
  double E_floor_injected = 0.0;
  double E_eint_floor_deposit = 0.0;
  double E_active_floor = 0.0;
  double E_redistribution_unresolved = 0.0;
  int n_eint_floor_hits = 0;
  int n_active_floor_hits = 0;
  double cap_energy_audit_D_K = 0.0;
  double eta_contact_step = 0.0;
  tenryu::hydro::I1BSpuriousSensorSummary i1b_ale_ke_sensor{};
  // Basis-coherent transport (TENRYU_I1B_OPTIONB_COHERENT): the component's
  // post-projection (V-paired) corner masses, exported so the install writes
  // literally the same numbers the velocity re-recover divided by. Empty
  // unless the coherent path ran this remap.
  std::vector<double> optionb_coherent_corner_mass;
  bool replay_transport_momentum_valid = false;
  double replay_transport_pr = 0.0;
  double replay_transport_pz = 0.0;
  double replay_raw_pi0_r = 0.0;
  double replay_raw_pi0_z = 0.0;
  double replay_raw_pi1_r = 0.0;
  double replay_raw_pi1_z = 0.0;
  double replay_assm_residual_r = 0.0;
  double replay_assm_residual_z = 0.0;
  double replay_rec_residual_r = 0.0;
  double replay_rec_residual_z = 0.0;
  int replay_discarded_dual_faces = 0;
  double replay_discarded_dual_mass = 0.0;
  double replay_discarded_dual_pi_r = 0.0;
  double replay_discarded_dual_pi_z = 0.0;
};

struct AleRemap2DRZOverrides {
  bool force_total_energy_remap = false;
  bool force_optionb_velocity_authority = false;
  bool force_optionb_coherent = false;
  bool force_optionb_coherent_transport = false;
  bool force_optionb_coherent_rerecover = false;
  bool allow_polar_shell_derefine = false;
  bool force_optionb_corner_mass_install = false;
  bool collect_replay_diagnostics = false;
  double near_massless_velocity_mass_floor = 0.0;
  const std::uint8_t* active_cell_mask = nullptr;
  const std::uint8_t* corner_transport_cell_mask = nullptr;
  const std::uint8_t* energy_closure_cell_mask = nullptr;
  // When non-null: node mask (1 = frozen) OR-ed into the velocity-projection
  // freeze — frozen nodes keep their velocities bitwise through the remap.
  // v1 restriction: mutually exclusive with the core_freeze velocity mask.
  const std::uint8_t* velocity_projection_frozen_node_mask = nullptr;
  const std::uint8_t* active_node_velocity_mask = nullptr;
};

struct CsrOptionBFaceColoring {
  int n_colors = 0;
  bool pathological = false;
  std::vector<int> face_color;
};

struct CsrOptionBCornerVelocityRemapResult {
  bool applied = false;
  int n_face_colors = 0;
  int n_internal_faces = 0;
  long long total_internal_face_packets = 0;  // total internal-face packets attempted (fallback-fraction denominator)
  long long total_filter_cells = 0;           // total cells passed through the hourglass filter
  int fallback_packets = 0;
  int expanded_stencil_packets = 0;
  int expanded_ring1_packets = 0;
  int expanded_ring2_packets = 0;
  int expanded_failed_packets = 0;
  int centroid_out_of_donor_packets = 0;
  int centroid_on_boundary_packets = 0;
  int centroid_far_packets = 0;
  int receiver_vertex_out_of_donor_packets = 0;
  int invalid_input_packets = 0;
  int skipped_packets = 0;
  int filter_invalid_cells = 0;
  int filter_degenerate_cells = 0;
  double alpha_min = 1.0;
  bool pathological_face_coloring = false;
  bool replay_transport_momentum_valid = false;
  double replay_transport_pr = 0.0;
  double replay_transport_pz = 0.0;
  double replay_raw_pi0_r = 0.0;
  double replay_raw_pi0_z = 0.0;
  double replay_raw_pi1_r = 0.0;
  double replay_raw_pi1_z = 0.0;
  double replay_assm_residual_r = 0.0;
  double replay_assm_residual_z = 0.0;
  double replay_rec_residual_r = 0.0;
  double replay_rec_residual_z = 0.0;
  int replay_discarded_dual_faces = 0;
  double replay_discarded_dual_mass = 0.0;
  double replay_discarded_dual_pi_r = 0.0;
  double replay_discarded_dual_pi_z = 0.0;
};

struct CsrOptionBCornerVelocityRemapBuffers {
  core::DeviceArray<double> corner_mass;
  core::DeviceArray<double> corner_p_r;
  core::DeviceArray<double> corner_p_z;
  core::DeviceArray<double> node_mass;
  core::DeviceArray<double> node_p_r;
  core::DeviceArray<double> node_p_z;
  core::DeviceArray<double> v_r;
  core::DeviceArray<double> v_z;
  core::DeviceArray<double> cell_mass;
  // Basis-coherent transport only: snapshot of the transported ledger taken
  // BEFORE the in-place V-pairing projection (corner_mass then holds the
  // projected/installed product); consumed by the basis-defect and gap-form
  // audit instruments. Empty unless TENRYU_I1B_OPTIONB_COHERENT ran.
  core::DeviceArray<double> corner_mass_transported;

  void reset(int n_cells, int n_nodes);
};

CsrOptionBFaceColoring csr_optionb_build_internal_face_coloring(
    const tenryu::mesh::MultiBlockTopology& mb,
    int n_cells);

bool csr_optionb_validate_internal_face_coloring(
    const tenryu::mesh::MultiBlockTopology& mb,
    int n_cells,
    const CsrOptionBFaceColoring& coloring);

bool csr_optionb_corner_velocity_remap_env_enabled();
bool csr_optionb_velocity_authority_enabled(const core::Config& cfg);
bool csr_optionb_coherent_enabled(const core::Config& cfg);

void csr_optionb_canonicalize_corner_mass_basis(core::State& state,
                                                const core::Config& cfg);

CsrOptionBCornerVelocityRemapResult
csr_optionb_corner_velocity_remap_component(
    const core::State& state,
    const core::Config& cfg,
    CsrOptionBCornerVelocityRemapBuffers& buffers,
    bool force_apply = false,
    bool force_swept_volume_sign_fixed = false,
    const double* target_cell_mass = nullptr,
    const double* source_v_r_override = nullptr,
    const double* source_v_z_override = nullptr,
    bool disable_limiters_for_audit = false,
    bool allow_polar_shell_derefine = false,
    bool force_optionb_coherent = false,
    bool force_optionb_coherent_transport = false,
    bool force_optionb_coherent_rerecover = false,
    bool collect_replay_diagnostics = false,
    double near_massless_velocity_mass_floor = 0.0,
    const std::uint8_t* override_inactive_cell_mask = nullptr,
    const std::uint8_t* assembly_cell_mask = nullptr,
    const std::uint8_t* active_node_velocity_mask = nullptr,
    const std::uint8_t* target_cell_mass_mask = nullptr,
    const std::uint8_t* discard_reference_inactive_cell_mask = nullptr);

AleRemap2DRZResult ale_remap_2d_rz(core::State& state,
                                   const core::Config& cfg,
                                   const HydroEOSContext* eos_ctx,
                                   double dt,
                                   const std::uint8_t* core_freeze_frozen_node_mask = nullptr,
                                   const AleRemap2DRZOverrides& overrides = {});

// Step-scoped worst |mass_closure_rel| across every CSR remap invocation
// (Winslow center-patch, per-block, axis-rezone fires, repair routes). The
// ALE phase resets it on entry and harvests it into AleStepResult; the
// driver-level repair routes harvest it directly after their own remaps.
void reset_remap_mass_closure_step_max();
double remap_mass_closure_step_max();

namespace ale_velcoherence {

void sample(const core::State& state,
            const core::Config& cfg,
            const char* checkpoint);

}  // namespace ale_velcoherence

}  // namespace tenryu::hydro::ale
