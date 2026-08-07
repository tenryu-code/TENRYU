#include "parallel/partition.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>

#include "core/error.hpp"

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace tenryu::parallel {
namespace {

constexpr int kLeft = 0;
constexpr int kRight = 1;
constexpr int kBottom = 2;
constexpr int kTop = 3;
constexpr int kNE = 4;
constexpr int kNW = 5;
constexpr int kSE = 6;
constexpr int kSW = 7;

struct AxisSplit {
  int start = 0;
  int end = 0;
  int local = 0;
};

std::string to_lower(std::string text) {
  for (char& ch : text) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return text;
}

AxisSplit split_axis(const int global_n, const int n_parts, const int part) {
  TENRYU_ASSERT(n_parts > 0, "split_axis requires n_parts > 0");
  TENRYU_ASSERT(part >= 0 && part < n_parts, "split_axis part index is out of range");

  const int safe_global = global_n > 0 ? global_n : 0;
  const int base = safe_global / n_parts;
  const int rem = safe_global % n_parts;

  AxisSplit out;
  out.local = base + (part < rem ? 1 : 0);
  out.start = part * base + std::min(part, rem);
  out.end = out.start + out.local;
  return out;
}

int logical_nz_for_1d(const int global_nr) {
  return global_nr > 0 ? 1 : 0;
}

bool is_prime(const int value) {
  if (value <= 1) {
    return false;
  }
  if (value <= 3) {
    return true;
  }
  if ((value % 2) == 0) {
    return false;
  }
  for (int factor = 3; static_cast<int64_t>(factor) * factor <= value;
       factor += 2) {
    if ((value % factor) == 0) {
      return false;
    }
  }
  return true;
}

std::array<int, 2> choose_auto_cart_dims(const int n_ranks, const int global_nr,
                                         const int global_nz) {
  int best_pr = 1;
  int best_pz = n_ranks > 0 ? n_ranks : 1;
  int64_t best_cost = std::numeric_limits<int64_t>::max();
  int best_aspect = std::numeric_limits<int>::max();

  for (int pr = 1; pr <= n_ranks; ++pr) {
    if ((n_ranks % pr) != 0) {
      continue;
    }
    const int pz = n_ranks / pr;
    const int64_t cost = static_cast<int64_t>(pr) * global_nz +
                         static_cast<int64_t>(pz) * global_nr;
    const int aspect = std::abs(pr - pz);
    if (cost < best_cost ||
        (cost == best_cost &&
         (aspect < best_aspect ||
          (aspect == best_aspect && pr < best_pr)))) {
      best_pr = pr;
      best_pz = pz;
      best_cost = cost;
      best_aspect = aspect;
    }
  }

  return {best_pr, best_pz};
}

void set_local_ranges(PartitionInfo& info, const AxisSplit& r_split,
                      const AxisSplit& z_split, const int dim,
                      const int global_nr, const int global_nz) {
  info.global_nr = global_nr;
  info.global_nz = global_nz;
  info.local_cell_range[0][0] = r_split.start;
  info.local_cell_range[0][1] = r_split.end;
  info.local_cell_range[1][0] = z_split.start;
  info.local_cell_range[1][1] = z_split.end;

  info.local_node_range[0][0] = r_split.start;
  info.local_node_range[0][1] = global_nr > 0 ? (r_split.end + 1) : 0;
  info.local_node_range[1][0] = z_split.start;
  info.local_node_range[1][1] =
      dim == 2 ? (global_nz > 0 ? (z_split.end + 1) : 0) : z_split.end;

  info.nr_local = r_split.local;
  info.nz_local = z_split.local;
  info.global_offset_r = r_split.start;
  info.global_offset_z = z_split.start;
}

void finalize_ghost_layout(PartitionInfo& info, const int dim) {
  // Option C layout: arrays are global-size on every rank, so the array
  // extents equal the global extents regardless of rank count. Ghost cells
  // are the neighbor-owned strips adjacent to the owned slab (one layer per
  // existing neighbor), counted here for buffer sizing/diagnostics.
  (void)dim;
  info.local_array_nr = info.global_nr;
  info.local_array_nz = info.global_nz;
  if (info.n_ranks <= 1) {
    info.ghost_layers = 0;
    info.n_ghost_cells = 0;
    return;
  }

  // g=2: the widest production stencils are the 1D checkerboard PQ filter
  // (rho/active at i±2) and the electron odd-even flux (cells j-2..j+1)
  // (design doc mpi_m18_20_20260717.md §6b.1).
  info.ghost_layers = 2;
  const int g = info.ghost_layers;
  if (info.cart_dims[0] > 1) {
    TENRYU_ASSERT(info.nr_local >= 2 * g,
                  "partition: owned r-extent must be >= 2*ghost_layers; "
                  "reduce rank count or raise min_cells_per_rank");
  }
  if (info.cart_dims[1] > 1) {
    TENRYU_ASSERT(info.nz_local >= 2 * g,
                  "partition: owned z-extent must be >= 2*ghost_layers; "
                  "reduce rank count or raise min_cells_per_rank");
  }
  int ghost_cells = 0;
  if (info.neighbor_ranks[kLeft] >= 0) ghost_cells += g * info.nz_local;
  if (info.neighbor_ranks[kRight] >= 0) ghost_cells += g * info.nz_local;
  if (info.neighbor_ranks[kBottom] >= 0) ghost_cells += g * info.nr_local;
  if (info.neighbor_ranks[kTop] >= 0) ghost_cells += g * info.nr_local;
  if (info.neighbor_ranks[kNE] >= 0) ghost_cells += g * g;
  if (info.neighbor_ranks[kNW] >= 0) ghost_cells += g * g;
  if (info.neighbor_ranks[kSE] >= 0) ghost_cells += g * g;
  if (info.neighbor_ranks[kSW] >= 0) ghost_cells += g * g;
  info.n_ghost_cells = ghost_cells;
}

void fill_1d_neighbors(PartitionInfo& info) {
  info.neighbor_ranks[kLeft] = info.rank > 0 ? (info.rank - 1) : -1;
  info.neighbor_ranks[kRight] =
      (info.rank + 1) < info.n_ranks ? (info.rank + 1) : -1;
  info.neighbor_ranks[kBottom] = -1;
  info.neighbor_ranks[kTop] = -1;
  info.neighbor_ranks[kNE] = -1;
  info.neighbor_ranks[kNW] = -1;
  info.neighbor_ranks[kSE] = -1;
  info.neighbor_ranks[kSW] = -1;
}

#if TENRYU_ENABLE_MPI
std::string mpi_failure_message(const char* call) {
  return std::string(call) + " failed while building PartitionInfo";
}
#endif

int cart_rank_or_boundary(const PartitionInfo& info, const int pr,
                          const int pz) {
  if (pr < 0 || pr >= info.cart_dims[0] || pz < 0 || pz >= info.cart_dims[1]) {
    return -1;
  }

#if TENRYU_ENABLE_MPI
  TENRYU_ASSERT(info.cart_comm != MPI_COMM_NULL,
                "cart_comm must be valid for 2D multi-rank decomposition");
  int coords[2] = {pr, pz};
  int neighbor_rank = -1;
  const int err = MPI_Cart_rank(info.cart_comm, coords, &neighbor_rank);
  TENRYU_ASSERT(err == MPI_SUCCESS, mpi_failure_message("MPI_Cart_rank"));
  return neighbor_rank;
#else
  return pr * info.cart_dims[1] + pz;
#endif
}

void fill_2d_neighbors(PartitionInfo& info) {
  const int pr = info.cart_coords[0];
  const int pz = info.cart_coords[1];

  info.neighbor_ranks[kLeft] = cart_rank_or_boundary(info, pr - 1, pz);
  info.neighbor_ranks[kRight] = cart_rank_or_boundary(info, pr + 1, pz);
  info.neighbor_ranks[kBottom] = cart_rank_or_boundary(info, pr, pz - 1);
  info.neighbor_ranks[kTop] = cart_rank_or_boundary(info, pr, pz + 1);
  info.neighbor_ranks[kNE] = cart_rank_or_boundary(info, pr + 1, pz + 1);
  info.neighbor_ranks[kNW] = cart_rank_or_boundary(info, pr - 1, pz + 1);
  info.neighbor_ranks[kSE] = cart_rank_or_boundary(info, pr + 1, pz - 1);
  info.neighbor_ranks[kSW] = cart_rank_or_boundary(info, pr - 1, pz - 1);
}

void query_rank_info(int* rank, int* n_ranks) {
  TENRYU_ASSERT(rank != nullptr, "query_rank_info rank pointer must not be null");
  TENRYU_ASSERT(n_ranks != nullptr,
                "query_rank_info n_ranks pointer must not be null");
  *rank = 0;
  *n_ranks = 1;

#if TENRYU_ENABLE_MPI
  int mpi_initialized = 0;
  const int initialized_err = MPI_Initialized(&mpi_initialized);
  TENRYU_ASSERT(initialized_err == MPI_SUCCESS,
                "MPI_Initialized failed in Partition::build");
  if (!mpi_initialized) {
    return;
  }

  const int rank_err = MPI_Comm_rank(MPI_COMM_WORLD, rank);
  TENRYU_ASSERT(rank_err == MPI_SUCCESS,
                mpi_failure_message("MPI_Comm_rank(MPI_COMM_WORLD)"));
  const int size_err = MPI_Comm_size(MPI_COMM_WORLD, n_ranks);
  TENRYU_ASSERT(size_err == MPI_SUCCESS,
                mpi_failure_message("MPI_Comm_size(MPI_COMM_WORLD)"));
#endif
}

}  // namespace

void Partition::query_world(int* rank, int* n_ranks) {
  query_rank_info(rank, n_ranks);
}

PartitionInfo Partition::build_single(const int global_nr, const int global_nz,
                                      const int dim) {
  TENRYU_ASSERT(dim == 1 || dim == 2,
                "Partition::build_single expects dim == 1 or dim == 2");
  TENRYU_ASSERT(global_nr >= 0, "Partition::build_single requires global_nr >= 0");
  TENRYU_ASSERT(global_nz >= 0, "Partition::build_single requires global_nz >= 0");

  PartitionInfo info{};

  const AxisSplit r_split = split_axis(global_nr, 1, 0);
  const int z_cells = dim == 2 ? global_nz : logical_nz_for_1d(global_nr);
  const AxisSplit z_split = split_axis(z_cells, 1, 0);
  set_local_ranges(info, r_split, z_split, dim, global_nr, z_cells);

  info.cart_coords[0] = 0;
  info.cart_coords[1] = 0;
  info.cart_dims[0] = 1;
  info.cart_dims[1] = 1;

  if (dim == 1) {
    fill_1d_neighbors(info);
  } else {
    fill_2d_neighbors(info);
  }

  finalize_ghost_layout(info, dim);
  return info;
}

PartitionInfo Partition::build(const int global_nr, const int global_nz,
                               const int dim, const std::string& method,
                               const std::vector<int>& user_dims,
                               const int min_cells_per_rank) {
  TENRYU_ASSERT(dim == 1 || dim == 2, "Partition::build expects dim == 1 or dim == 2");
  TENRYU_ASSERT(global_nr >= 0, "Partition::build requires global_nr >= 0");
  TENRYU_ASSERT(global_nz >= 0, "Partition::build requires global_nz >= 0");
  TENRYU_ASSERT(min_cells_per_rank >= 1,
                "Partition::build requires min_cells_per_rank >= 1");

  const std::string method_lower = to_lower(method);
  TENRYU_ASSERT(method_lower.empty() || method_lower == "slab" ||
                    method_lower == "cartesian",
                "Partition::build method must be \"slab\" or \"cartesian\"");

  int rank = 0;
  int n_ranks = 1;
  query_rank_info(&rank, &n_ranks);
  if (n_ranks <= 1) {
    return build_single(global_nr, global_nz, dim);
  }

  PartitionInfo info{};
  info.rank = rank;
  info.n_ranks = n_ranks;

  if (dim == 1) {
    const int64_t required_cells =
        static_cast<int64_t>(n_ranks) * min_cells_per_rank;
    const int64_t available_cells = global_nr;
    if (available_cells < required_cells) {
      TENRYU_ASSERT(
          false,
          "Partition::build 1D slab min-cells violation: N_r=" +
              std::to_string(global_nr) + ", P=" + std::to_string(n_ranks) +
              ", n_min=" + std::to_string(min_cells_per_rank) +
              ". Require N_r >= P * n_min.");
    }

    info.cart_coords[0] = rank;
    info.cart_coords[1] = 0;
    info.cart_dims[0] = n_ranks;
    info.cart_dims[1] = 1;

    const AxisSplit r_split = split_axis(global_nr, n_ranks, rank);
    const int z_cells = logical_nz_for_1d(global_nr);
    const AxisSplit z_split = split_axis(z_cells, 1, 0);
    set_local_ranges(info, r_split, z_split, dim, global_nr, z_cells);
    fill_1d_neighbors(info);
    finalize_ghost_layout(info, dim);
    return info;
  }

  int pr = 1;
  int pz = n_ranks;
  if (!user_dims.empty()) {
    TENRYU_ASSERT(user_dims.size() == 2,
                  "Partition::build user_dims must contain exactly two entries");
    TENRYU_ASSERT(user_dims[0] > 0 && user_dims[1] > 0,
                  "Partition::build user_dims entries must be positive");
    const int64_t user_product =
        static_cast<int64_t>(user_dims[0]) * user_dims[1];
    TENRYU_ASSERT(user_product == n_ranks,
                  "Partition::build user_dims product must equal n_ranks");
    pr = user_dims[0];
    pz = user_dims[1];
  } else {
    const auto dims = choose_auto_cart_dims(n_ranks, global_nr, global_nz);
    pr = dims[0];
    pz = dims[1];
    if (is_prime(n_ranks) && (pr == 1 || pz == 1)) {
      core::log_warning("WARNING: prime rank count P=" + std::to_string(n_ranks) +
                        " results in 1D decomposition. Recommend composite rank "
                        "count for better load balance.");
    }
  }

  TENRYU_ASSERT(pr <= global_nr,
                "Partition::build 2D: ranks in R exceed global cells");
  TENRYU_ASSERT(pz <= global_nz,
                "Partition::build 2D: ranks in Z exceed global cells");
  TENRYU_ASSERT(global_nr / pr >= min_cells_per_rank,
                "Partition::build 2D: too few R cells per rank");
  TENRYU_ASSERT(global_nz / pz >= min_cells_per_rank,
                "Partition::build 2D: too few Z cells per rank");

  info.cart_dims[0] = pr;
  info.cart_dims[1] = pz;

#if TENRYU_ENABLE_MPI
  int mpi_initialized = 0;
  const int initialized_err = MPI_Initialized(&mpi_initialized);
  TENRYU_ASSERT(initialized_err == MPI_SUCCESS,
                "MPI_Initialized failed before MPI_Cart_create");
  TENRYU_ASSERT(mpi_initialized,
                "MPI_Cart_create requires MPI to be initialized");

  int dims[2] = {pr, pz};
  int periods[2] = {0, 0};
  MPI_Comm cart_comm = MPI_COMM_NULL;
  const int create_err =
      MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart_comm);
  TENRYU_ASSERT(create_err == MPI_SUCCESS, mpi_failure_message("MPI_Cart_create"));
  TENRYU_ASSERT(cart_comm != MPI_COMM_NULL,
                "MPI_Cart_create returned MPI_COMM_NULL");
  info.cart_comm = cart_comm;

  int cart_rank = 0;
  const int rank_err = MPI_Comm_rank(info.cart_comm, &cart_rank);
  TENRYU_ASSERT(rank_err == MPI_SUCCESS, mpi_failure_message("MPI_Comm_rank(cart)"));
  int coords[2] = {0, 0};
  const int coords_err = MPI_Cart_coords(info.cart_comm, cart_rank, 2, coords);
  TENRYU_ASSERT(coords_err == MPI_SUCCESS,
                mpi_failure_message("MPI_Cart_coords"));
  info.cart_coords[0] = coords[0];
  info.cart_coords[1] = coords[1];
#else
  const int safe_pz = pz > 0 ? pz : 1;
  info.cart_coords[0] = rank / safe_pz;
  info.cart_coords[1] = rank % safe_pz;
#endif

  const AxisSplit r_split = split_axis(global_nr, info.cart_dims[0], info.cart_coords[0]);
  const AxisSplit z_split = split_axis(global_nz, info.cart_dims[1], info.cart_coords[1]);
  set_local_ranges(info, r_split, z_split, dim, global_nr, global_nz);
  fill_2d_neighbors(info);
  finalize_ghost_layout(info, dim);
  return info;
}

}  // namespace tenryu::parallel
