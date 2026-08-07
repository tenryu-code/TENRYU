#pragma once

#include "core/state.hpp"
#include "parallel/reduction.hpp"

namespace tenryu::diagnostics {

struct EnergyTotals {
  double E_int_e = 0.0;
  double E_int_i = 0.0;
  double E_kin = 0.0;
  // W-J-2 diagnostic: nodal-form kinetic (1D only; 2D copies E_kin).
  double E_kin_nodal = 0.0;
};

struct EnergyBudget {
  // 14-component budget terms (all [erg])
  double E_int_e = 0.0;
  double E_int_i = 0.0;
  double E_kin = 0.0;
  double E_rad = 0.0;
  double E_rad_mesh_advection = 0.0;
  double E_laser_in = 0.0;
  double E_Marshak_in = 0.0;
  double E_volume_in = 0.0;
  double E_burn_in = 0.0;
  double E_laser_esc = 0.0;
  double E_cbet_iaw = 0.0;
  double E_rad_esc = 0.0;
  double E_numerical_loss = 0.0;
  double E_pdV_bdry = 0.0;
  double E_floor = 0.0;
  double E_safety = 0.0;
  double E_redistribution_unresolved = 0.0;
  // Signed solver correction term; emitted in diagnostics output as energy/E_solver.
  double E_solver = 0.0;

  // Backward-compatible summary terms
  double E_kinetic = 0.0;
  double E_internal = 0.0;
  double E_total = 0.0;

  // Step diagnostics
  double dE_total = 0.0;
  double epsilon_budget = 0.0;
  double E_denom = 0.0;
};

struct ConservationResiduals {
  double mass = 0.0;
  double r_momentum = 0.0;
  double z_momentum = 0.0;
  double vol_closure = 0.0;
  bool valid = false;
};

struct EnergyBudgetStepInput {
  EnergyTotals before{};
  EnergyTotals after{};
  double E_rad_before = 0.0;
  double E_rad_after = 0.0;
  double E_rad_mesh_advection = 0.0;
  double E_laser_in = 0.0;
  double E_Marshak_in = 0.0;
  double E_volume_in = 0.0;
  double E_burn_in = 0.0;
  double E_laser_esc = 0.0;
  double E_cbet_iaw = 0.0;
  double E_rad_esc = 0.0;
  double E_numerical_loss = 0.0;
  double E_pdV_bdry = 0.0;
  double E_floor = 0.0;
  double E_safety = 0.0;
  double E_redistribution_unresolved = 0.0;
  // Signed solver correction term; emitted in diagnostics output as energy/E_solver.
  double E_solver = 0.0;
  const parallel::Reduction* reduction = nullptr;
};

EnergyTotals compute_energy_totals_1d(const core::State& state);
EnergyTotals compute_energy_totals_2d(const core::State& state);

// --- W-R1 slot (read-deferred) audit reductions --------------------------
// Block-sum geometry shared with the internal reduce helpers: one slot
// holds k * energy_reduce_blocks(n) doubles, quantity-major.
int energy_reduce_blocks(int n);
// Same per-quantity block reductions as the internal batched reduce, block
// sums written into d_slot (quantity-major). NO device sync, NO readback --
// the caller materializes later from a host copy of the slot.
void reduce_device_sums_to_slot(const double* const* d_contribs,
                                int k,
                                int n,
                                double* d_slot);
// Host twin of the internal final sum: ascending long-double per quantity
// over `blocks` block sums. blocks == 0 yields 0.0 for every quantity.
void materialize_reduced_sums(const double* h_block_sums,
                              int k,
                              int blocks,
                              double* out);
// Slot variant of compute_energy_totals_1d: identical contribution
// kernel, block sums for {int_e, int_i, kin, kin_nodal} into d_slot
// (4 * energy_reduce_blocks(n_cells) doubles). Returns the block count
// used (0 for an empty state -- materializing then yields zeros).
int compute_energy_totals_1d_to_slot(const core::State& state,
                                     double* d_slot);
// Slot variant of compute_energy_totals_2d: identical contribution
// kernels, block sums for {int_e, int_i, kin} into d_slot
// (3 * energy_reduce_blocks(n_cells) doubles). Returns the block count
// used (0 for an empty state -- materializing then yields zeros).
int compute_energy_totals_2d_to_slot(const core::State& state,
                                     double* d_slot);
// Host materialization of a 4-quantity 1D slot readback.
EnergyTotals materialize_energy_totals_1d_from_packet(const double* h_slot,
                                                      int blocks);
// Host materialization of a 3-quantity slot readback; E_kin_nodal :=
// E_kin exactly as compute_energy_totals_2d does.
EnergyTotals materialize_energy_totals_2d(const double* h_slot,
                                          int blocks);

EnergyBudget compute_step_energy_budget(const EnergyBudgetStepInput& input);

}  // namespace tenryu::diagnostics
