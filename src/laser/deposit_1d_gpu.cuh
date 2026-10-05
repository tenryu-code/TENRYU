#pragma once

// The 1D laser cell deposit of a step on the device: the beams' deposits added in order (laser.cu
// laser_step's total_dep_1d, formerly a host vector filled from a copy of every beam's deposit) and
// the redistribution to the cells that can absorb it (deposit_transfer.cu
// apply_deposit_redistribution_1d: blocked receivers, the ghost-corona hand-off, the smoothing passes,
// the hot-electron power, the conservation check), which ran on the host on a hydro mirror. The
// arithmetic repeats the host's operation by operation without floating-point contraction (the file
// is compiled with -fmad=false); the hand-off weights use the device exp and pow, which can differ
// from the host's in the last place, and the long double sums of the host (the blocked power and the
// conservation checks) are double-double sums here.

#include <cstdint>

#include <cuda_runtime.h>

#include "core/state.hpp"
#include "laser/laser_map_1d_gpu.cuh"

namespace tenryu::laser::deposit_1d {

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

// Zeroes the step's accumulated deposit [n_cells] (erg/s).
void begin(Workspace& ws, int n_cells, cudaStream_t stream);
// total += src, per cell (src device [n_cells]).
void add(Workspace& ws, const double* src, cudaStream_t stream);
// total = src / divisor, per cell (the skip path's energy to power).
void assign_divided(Workspace& ws, const double* src, double divisor, cudaStream_t stream);
// The accumulated deposit (device [n_cells]).
double* total(const Workspace& ws);
// A device copy of src kept for the beam fold (laser_step's fold_dep), and its addition.
void keep_fold(Workspace& ws, const double* src, cudaStream_t stream);
void add_fold(Workspace& ws, cudaStream_t stream);
const double* fold(const Workspace& ws);

// Host scalars of a redistribution.
struct Inputs {
  double dt = 0.0;
  double conservation_tol = 1.0e-10;
  int smooth_passes = 0;
  double smooth_alpha = 0.0;
  int ghost_enabled = 0;
  int transition_enabled = 0;
  int handoff_cells = 4;
  double handoff_decay = 1.5;
  double transition_resolved_nhat = 0.9;
  int transition_resolved_cells = 3;
  double transition_density_exponent = 1.0;
  double n_crit = 1.0;
  int owned_begin = 0;   // the cells this rank writes: [owned_begin, owned_end)
  int owned_end = 0;
};

struct Result {
  double blocked_power = 0.0;        // power with no receiver (erg/s, not negative)
  double transition_blend = 0.0;
  int resolved_cells = 0;
  double sum_input = 0.0;            // the deposit given (erg/s)
  double conservation_rel = 0.0;     // |given + hot-e - (cells + blocked)| / |given + hot-e|
  int smoothing_ran = 0;
  double smoothing_sum_before = 0.0;
  double smoothing_rel = 0.0;
  double energy_sum = 0.0;           // the sum of the energies written to state.laser_dep (erg)
};

// apply_deposit_redistribution_1d on the accumulated deposit (overwritten), for dt > 0: writes
// state.laser_dep = power * dt on the owned cells and zero elsewhere. map holds this step's map of
// the same state (laser_map_1d::map_scalars: the deposit densities and the allowed supercritical
// cell); hot_e_extra is the hot-electron power per cell (device [n_cells]) or nullptr. One
// synchronisation of `stream` (the scalars).
Result redistribute(Workspace& ws, core::State& state, const laser_map_1d::Workspace& map,
                    const laser_map_1d::MapScalars& m, const double* hot_e_extra,
                    const Inputs& in, cudaStream_t stream);

}  // namespace tenryu::laser::deposit_1d
