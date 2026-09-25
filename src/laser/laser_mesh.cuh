#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/device_error_flags.cuh"
#include "core/state.hpp"
#include "laser/bilinear_interpolation.cuh"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::laser {
struct LaserPhysExtOptions;
struct LaserZeffMaterial;
}

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::laser {

struct HydroMirror1D {
  std::vector<double> rho;
  std::vector<double> Te;
  std::vector<double> zbar;
  std::vector<double> r_edges;
  std::vector<double> A_eff;
  std::vector<std::uint8_t> cell_is_void;
};

struct LaserMesh {
  static constexpr int kTraceStepHistSize = 5;

  int nr = 0;
  int nz = 0;
  int n_nodes_r = 0;
  int n_nodes_z = 0;
  int nr_capacity = 0;
  int nz_capacity = 0;
  int n_nodes_r_capacity = 0;
  int n_nodes_z_capacity = 0;
  int underuse_steps = 0;
  int steps_since_realloc = 0;

  double* node_R = nullptr;        // [n_nodes_r]
  double* node_Z = nullptr;        // [n_nodes_z]
  double* n_e_hat = nullptr;       // [n_nodes_r * n_nodes_z]
  double* n_e_hat_raw = nullptr;   // [n_nodes_r * n_nodes_z] unclipped ne/nc
  double* T_e = nullptr;           // [n_nodes_r * n_nodes_z]
  double* Zbar = nullptr;          // [n_nodes_r * n_nodes_z]
  double* smooth_kappa_factor = nullptr;  // [n_nodes_r * n_nodes_z] precomputed IB smooth A
  double* grad_n_hat_R = nullptr;  // [n_nodes_r * n_nodes_z]
  double* grad_n_hat_Z = nullptr;  // [n_nodes_r * n_nodes_z]
  // Radial profile of the 1D traces [radial_n_nodes <= radial_capacity]:
  // since 2026-09-24 on the hydro-anchored nodes of map_from_hydro_1d
  // (laser_mesh_bodies::build_trace_profile_nodes_1d); before, the axis
  // column of the 2D laser mesh.
  double* radial_node_r = nullptr;
  double* radial_n_hat = nullptr;
  double* radial_n_hat_raw = nullptr;
  double* radial_smooth_kappa = nullptr;
  double* radial_T_e = nullptr;
  double* radial_Zbar = nullptr;
  double* radial_dn_dr = nullptr;
  int radial_n_nodes = 0;
  int radial_capacity = 0;
  // Set by map_from_hydro_1d: compute_smooth_kappa then fills
  // radial_smooth_kappa on the profile nodes.
  bool trace_profile_1d = false;
  // The inputs of the last map_from_hydro_1d that its trace profile uses
  // (map_trace_profile_1d builds the profile from them).
  struct TraceProfileMap1D {
    int n_cells = 0;
    int outer_surface_cell = -1;
    int fcrit_cell = -1;
    int use_ghost_corona = 0;
    int critical_clip = 0;
    double n_crit_safe = 1.0;
    double ghost_ne_inner = 0.0;
    double ghost_scale_length = 0.0;
    double ghost_ne_min = 0.0;
    double r_surface_outer = 0.0;
    double r_ghost_outer = 0.0;
    double Te_anchor = 0.0;
    double zbar_anchor = 0.0;
  };
  TraceProfileMap1D trace_profile_map;
  // Mesh.geometry_1d of the mapped hydro state (mesh::Geometry1D: 0 sphere,
  // 1 cylinder, 2 slab), set by map_from_hydro_1d.
  int geometry_code = 0;
  // A device double holding 0 (the Z of the profile nodes).
  double* radial_zero = nullptr;
  double* deposit = nullptr;       // [n_nodes_r * n_nodes_z] [erg/s]
  double* hydro_A_eff_device = nullptr;       // [hydro_cell_capacity]
  std::uint8_t* hydro_cell_is_void_device = nullptr;  // [hydro_cell_capacity]
  int hydro_cell_capacity = 0;
  double* zeff_table_dev = nullptr;
  int zeff_table_capacity = 0;
  int* scratch_step_histogram = nullptr;  // [kTraceStepHistSize]
  void* scratch_step_tally_slab = nullptr;  // [40] owns five 8-byte step tally slots.
  // Non-owning aliases into scratch_step_tally_slab slots 0..4 (offsets 0,8,16,24,32);
  // never cudaFree them individually.
  double* scratch_unabsorbed = nullptr;                         // [1] slot 0
  unsigned long long* scratch_tail_closure_count = nullptr;      // [1] slot 1
  double* scratch_tail_closure_absorbed_power = nullptr;         // [1] slot 2
  unsigned long long* scratch_critical_surface_hit_count = nullptr;  // [1] slot 3
  double* scratch_ra_power_total = nullptr;                      // [1] slot 4
  core::DeviceErrorFlags* scratch_error_flags = nullptr;              // [1]
  unsigned char* scratch_step_pack_device = nullptr;  // persistent scalar/tiny D2H pack
  unsigned char* scratch_step_pack_host = nullptr;    // pinned host mirror
  std::size_t scratch_step_pack_capacity = 0;
  int* scratch_per_ray_step_count = nullptr;       // [scratch_per_ray_step_capacity]
  int* scratch_sorted_step_count = nullptr;        // [scratch_per_ray_step_capacity]
  int* scratch_per_warp_step_max = nullptr;        // [scratch_per_warp_step_capacity]
  unsigned long long* scratch_per_warp_step_sum = nullptr;  // [scratch_per_warp_step_capacity]
  double* hot_e_capture = nullptr;
  int scratch_per_ray_step_capacity = 0;
  int scratch_per_warp_step_capacity = 0;
  int hot_e_capture_capacity = 0;

  double R_max = 0.0;
  double Z_min = 0.0;
  double Z_max = 0.0;
  double n_crit = 0.0;
  double n_hat_margin = 0.9999;
  double dx_min = 0.0;
  double target_radius = 0.0;
  double material_A = 1.0;
  std::vector<double> material_A_list;
  std::shared_ptr<void> port_section_state;
  bool ghost_corona_enabled = false;
  int ghost_n_out = 0;
  double ghost_ne_min_frac = 0.03;
  double ghost_ne_max_frac = 0.99;
  double ghost_Te_min_eV = 50.0;
  double ghost_zbar_min = 1.0;
  double ghost_zbar_max = 4.0;
  int ghost_handoff_cells = 4;
  double ghost_handoff_decay = 1.5;
  bool ghost_transition_enabled = false;
  double ghost_transition_resolved_nhat = 0.9;
  int ghost_transition_resolved_cells = 3;
  double ghost_transition_density_exponent = 1.0;
  double last_ghost_transition_blend = 0.0;
  int last_ghost_transition_resolved_cells = 0;
  double last_ghost_width = 0.0;  // ghost corona width of the last 1D map [cm]; 0 = inactive
  // Laser turn-on time [s] (laser_turn_on_time_s), filled on the first 1D map;
  // negative until then. The ghost corona width grows as c_s (t - t_on).
  double laser_turn_on_s = -1.0;
  double last_trace_unabsorbed_power = 0.0;
  double last_transfer_blocked_power = 0.0;
  double last_unabsorbed_power = 0.0;
  double last_ra_power = 0.0;
  double last_commanded_energy = 0.0;
  std::int64_t last_tail_closure_count = 0;
  double last_tail_closure_absorbed_power = 0.0;
  std::int64_t last_critical_surface_hit_count = 0;
  double last_cbet_exchanged_power = 0.0;
  double last_cbet_ledger_residual = 0.0;
  double last_cbet_conv_final = 0.0;
  std::int64_t last_cbet_clamp_count = 0;
  std::int64_t last_cbet_overflow_rays = 0;
  int last_cbet_iterations = 0;
  bool last_cbet_converged = true;
  std::vector<std::vector<int>> ray_steps_previous;
  std::vector<std::vector<int>> ray_steps_output;
  std::vector<std::vector<int>> ray_order;

  LaserMesh() = default;
  ~LaserMesh();
  LaserMesh(const LaserMesh&) = delete;
  LaserMesh& operator=(const LaserMesh&) = delete;
  LaserMesh(LaserMesh&& other) noexcept;
  LaserMesh& operator=(LaserMesh&& other) noexcept;

  void release();
  [[nodiscard]] bool is_allocated() const;
  [[nodiscard]] int n_nodes() const;

  void allocate(int nr_cells, int nz_cells);
  void ensure_capacity(int nr_new, int nz_new);
  // Radial profile arrays for at least n_nodes nodes (their content is not
  // kept).
  void ensure_radial_capacity(int n_nodes);
  void clear_deposit(cudaStream_t stream = nullptr) const;
  void ensure_step_scratch();
  void ensure_step_pack(std::size_t bytes);
  void ensure_per_ray_step_scratch(int n_rays);
  void ensure_hot_e_capture(int n_rays, int n_channels);
  void clear_step_scratch(cudaStream_t stream = nullptr) const;

  TENRYU_HOST_DEVICE inline int node_index(const int i, const int j) const {
    return i * n_nodes_z + j;
  }

  TENRYU_HOST_DEVICE inline bool is_outside(const double R, const double Z) const {
    return (R < 0.0 || R > R_max || Z < Z_min || Z > Z_max);
  }

  TENRYU_HOST_DEVICE inline double interp_n_hat(const double R, const double Z) const {
    const BilinearCell c =
        BilinearInterp::locate_cell(node_R, node_Z, n_nodes_r, n_nodes_z, R, Z);
    const BilinearWeights w = BilinearInterp::compute_weights(c.xi, c.eta);
    return BilinearInterp::interpolate(n_e_hat, n_nodes_z, c, w);
  }

  TENRYU_HOST_DEVICE inline void interp_grad(const double R,
                                             const double Z,
                                             double& dndR,
                                             double& dndZ) const {
    const BilinearCell c =
        BilinearInterp::locate_cell(node_R, node_Z, n_nodes_r, n_nodes_z, R, Z);
    const BilinearWeights w = BilinearInterp::compute_weights(c.xi, c.eta);
    BilinearInterp::interpolate_gradient(grad_n_hat_R, grad_n_hat_Z, n_nodes_z, c, w, dndR,
                                         dndZ);
  }

  TENRYU_HOST_DEVICE inline void interp_state(const double R,
                                              const double Z,
                                              double& n_hat,
                                              double& Te,
                                              double& zbar) const {
    const BilinearCell c =
        BilinearInterp::locate_cell(node_R, node_Z, n_nodes_r, n_nodes_z, R, Z);
    const BilinearWeights w = BilinearInterp::compute_weights(c.xi, c.eta);
    n_hat = BilinearInterp::interpolate(n_e_hat, n_nodes_z, c, w);
    Te = BilinearInterp::interpolate(T_e, n_nodes_z, c, w);
    zbar = BilinearInterp::interpolate(Zbar, n_nodes_z, c, w);
  }
};

LaserMesh create_from_config(const core::Config& cfg);

// Earliest time [s] at which any beam's frozen power table is positive: the
// abscissa before the first positive sample (the piecewise-linear power rises
// from there), or the first abscissa when that sample is positive. 0 when no
// table has a positive sample.
double laser_turn_on_time_s(const core::State& state);

void upload_zeff_table(LaserMesh& mesh,
                       const double* host_ratio,
                       int n);

void build_hydro_mirror_1d(const LaserMesh& mesh,
                           const core::State& state,
                           HydroMirror1D& mirror);

void map_from_hydro_1d(LaserMesh& mesh,
                       const core::State& state,
                       const core::Config::LaserConfig& laser_cfg,
                       cudaStream_t stream = nullptr);

// Material of every node of the 1D laser mesh (multi-material decks with a
// per-material collision charge, 2026-09-24). map_from_hydro_1d sizes the
// node arrays for the mesh it maps: node_material / node_zcoll [n_nodes],
// radial_node_material / radial_zcoll [radial_n_nodes].
struct LaserNodeMaterial1D {
  const int* cell_material_index = nullptr;           // hydro cells [n_cells]
  const LaserZeffMaterial* zeff_materials = nullptr;  // [n_materials]
  int n_materials = 0;
  core::DeviceArray<int> node_material;
  core::DeviceArray<double> node_zcoll;
  // The trace profile's nodes [radial_n_nodes].
  core::DeviceArray<int> radial_node_material;
  core::DeviceArray<double> radial_zcoll;
};

void map_from_hydro_1d(LaserMesh& mesh,
                       const core::State& state,
                       const core::Config::LaserConfig& laser_cfg,
                       const HydroMirror1D& hydro,
                       cudaStream_t stream = nullptr,
                       LaserNodeMaterial1D* node_material = nullptr);

// The nodes of the 1D traces' radial profile
// (laser_mesh_bodies::build_trace_profile_nodes_1d) placed on the device by one
// block in parallel over the hydro cells, bitwise the serial placement (which
// runs for meshes outside the parallel placement's conditions): device inputs
// as build_trace_profile_nodes_1d takes them, the nodes into out (at most
// capacity) and their count into *d_count.
void place_trace_profile_nodes_1d(const double* r_edges,
                                  const std::uint8_t* cell_is_void,
                                  int n_cells,
                                  int outer_surface_cell,
                                  int fcrit_cell,
                                  const double* rho,
                                  const double* zbar,
                                  const double* A_eff_cell,
                                  double n_crit_safe,
                                  int critical_clip,
                                  double n_hat_margin,
                                  const double* graded_r,
                                  int n_graded,
                                  double* out,
                                  int capacity,
                                  int* d_count,
                                  int subdiv,
                                  bool planar,
                                  cudaStream_t stream = nullptr);

// The 1D traces' radial profile of the last map_from_hydro_1d (its graded
// nodes and mesh.trace_profile_map, the hydro state it mapped): nodes
// (laser_mesh_bodies::build_trace_profile_nodes_1d with subdiv parts per half
// cell), n_hat, n_hat_raw, T_e, Zbar, node materials and dn/dr on them.
// map_from_hydro_1d calls it with subdiv = 1; a larger subdiv is a
// measure-only refinement (the laser trace comparison diagnostic).
void map_trace_profile_1d(LaserMesh& mesh,
                          const core::State& state,
                          int subdiv,
                          cudaStream_t stream = nullptr,
                          LaserNodeMaterial1D* node_material = nullptr);

// The IB factor radial_smooth_kappa on the profile nodes (the profile part of
// compute_smooth_kappa).
void compute_trace_profile_kappa_1d(LaserMesh& mesh,
                                    double lambda_cm,
                                    double eps_n,
                                    double coulomb_log_floor,
                                    cudaStream_t stream = nullptr,
                                    const LaserPhysExtOptions* phys_ext = nullptr,
                                    const LaserNodeMaterial1D* node_material = nullptr);

// Measure-only: the radial profile as it was before 2026-09-24, the Z = 0
// column of the 2D laser mesh (node_R, n_hat, n_hat_raw, the IB factor, T_e
// and the node collision charge copied from the column; dn/dr between its
// nodes). map_trace_profile_1d and compute_trace_profile_kappa_1d restore the
// hydro-anchored profile.
void extract_axis_column_profile_1d(LaserMesh& mesh,
                                    cudaStream_t stream = nullptr,
                                    LaserNodeMaterial1D* node_material = nullptr);

// Under MPI (Option C, NUMERICS §12.4.2 partition-of-unity): pass the
// partition + reducer so each LM node is computed by the single rank
// owning its located base hydro cell (the 4-cell interpolation stencil
// is ghost-1-reachable and halo-fresh) and the LM field arrays are
// Allreduce(SUM)-assembled — every rank ends with the identical,
// owner-true LaserMesh before tracing. Serial defaults keep the P=1
// byte path.
void map_from_hydro_2d(LaserMesh& mesh,
                       const core::State& state,
                       const core::Config::LaserConfig& laser_cfg,
                       cudaStream_t stream = nullptr,
                       const parallel::PartitionInfo* part = nullptr,
                       const parallel::Reduction* reduction = nullptr);

void compute_gradients(LaserMesh& mesh, cudaStream_t stream = nullptr);
void compute_smooth_kappa(LaserMesh& mesh,
                          double lambda_cm,
                          double eps_n,
                          double coulomb_log_floor,
                          cudaStream_t stream = nullptr,
                          const LaserPhysExtOptions* phys_ext = nullptr,
                          const LaserNodeMaterial1D* node_material = nullptr);

double compute_min_cell_spacing(const LaserMesh& mesh);

}  // namespace tenryu::laser
