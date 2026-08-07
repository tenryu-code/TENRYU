#pragma once

#include <string>
#include <vector>

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace tenryu::parallel {

// Decomposition layout (Option C, docs/design/mpi_m18_20_20260717.md §3):
// every rank allocates and initializes the GLOBAL-size arrays; the rank
// computes only its owned slab [local_cell_range); ghost cells are the
// neighbor-owned one-layer strips adjacent to the owned slab, stored at
// their natural global indices and refreshed by halo exchange. There is no
// separate per-rank local array layout. local_array_nr/nz therefore equal
// the global array extents (kept for call-site compatibility).
struct PartitionInfo {
  int rank = 0;
  int n_ranks = 1;
  int cart_coords[2] = {0, 0};
  int cart_dims[2] = {1, 1};
  int local_cell_range[2][2] = {{0, 0}, {0, 0}};
  int local_node_range[2][2] = {{0, 0}, {0, 0}};
  int ghost_layers = 0;
  int n_ghost_cells = 0;
  int nr_local = 0;
  int nz_local = 0;
  int neighbor_ranks[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  int global_offset_r = 0;
  int global_offset_z = 0;
  int global_nr = 0;
  int global_nz = 0;  // logical z extent (1 for 1D_SPH)
  int local_array_nr = 0;
  int local_array_nz = 0;
#if TENRYU_ENABLE_MPI
  MPI_Comm cart_comm = MPI_COMM_NULL;
#endif

  bool has_left_boundary() const { return neighbor_ranks[0] < 0; }
  bool has_right_boundary() const { return neighbor_ranks[1] < 0; }
  bool has_bottom_boundary() const { return neighbor_ranks[2] < 0; }
  bool has_top_boundary() const { return neighbor_ranks[3] < 0; }
  bool has_axis() const { return cart_coords[0] == 0; }
  bool is_slab() const { return cart_dims[0] == 1 || cart_dims[1] == 1; }

  bool owns_cell(const int i, const int j) const {
    return i >= local_cell_range[0][0] && i < local_cell_range[0][1] &&
           j >= local_cell_range[1][0] && j < local_cell_range[1][1];
  }
  // Flat global cell index c = i * max(global_nz, 1) + j.
  bool owns_cell_flat(const int c) const {
    const int nz = global_nz > 0 ? global_nz : 1;
    return owns_cell(c / nz, c % nz);
  }
};

class Partition {
 public:
  static PartitionInfo build(int global_nr, int global_nz, int dim,
                             const std::string& method,
                             const std::vector<int>& user_dims,
                             int min_cells_per_rank);
  static PartitionInfo build_single(int global_nr, int global_nz, int dim);
  // World rank/size without building a partition (1/0 when MPI is off or
  // not initialized).
  static void query_world(int* rank, int* n_ranks);

  // Backward compatibility: old API.
  Partition() = default;
  PartitionInfo info() const { return build_single(0, 0, 1); }
};

}  // namespace tenryu::parallel
