#pragma once

#include <cstddef>

#include <cuda_runtime.h>

#include "burn/corman_diffusion.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"

namespace tenryu::burn {

// Boundary classes for the four 2D_RZ domain faces: 0 = zero-flux
// (axis/reflect), 1 = Milne escape (free-type).
struct Corman2DBc {
  int r_inner = 0;
  int r_outer = 0;
  int z_bottom = 0;
  int z_top = 0;
};

// One implicit multigroup cascade step for ONE charged-product slot on the
// structured 2D RZ mesh. N_dev is the group-major density spectrum
// [n_groups * n_cells] (cells c = i*nz + j). Node coords x_r/x_z are the
// (nr+1)*(nz+1) State node arrays. dep_e/dep_i accumulate erg per cell.
// Deterministic: serial tally kernels, fixed-iteration CG with fixed-shape
// reductions. Throws (TENRYU_ASSERT) if CG hits the iteration cap without
// converging (fail-closed; no silent saturation).
// MPI (Option C, M18d d2): pass the partition/comm context and the owned
// cell window for the distributed CG (owned-masked dots + Allreduce,
// per-iteration search-direction ghost exchange) and the owned-window
// tallies. Serial defaults keep the P=1 byte path.
CormanStepResult corman_diffusion_2d_step(
    const CormanParams& p, int nr, int nz, double species_A, double species_Z,
    double E_birth_keV, const double* x_r_dev, const double* x_z_dev,
    const double* vol_dev, const double* rho_dev, const double* Te_eV_dev,
    const double* Ti_eV_dev, const double* ne_dev, const double* S_birth_dev,
    double dt_s, double* N_dev, double* dep_e_dev, double* dep_i_dev,
    const Corman2DBc& bc, cudaStream_t stream,
    const parallel::PartitionInfo* mpi_part = nullptr,
    parallel::CommBuffers* mpi_bufs = nullptr, int mpi_c_begin = 0,
    int mpi_c_end = 0);

void corman2d_scale_rows_by_cell(double* dst, const double* src,
                                 const double* rho, int G, int J);
void corman2d_divide_rows_by_cell(double* dst, const double* src,
                                  const double* rho, int G, int J);

}  // namespace tenryu::burn
