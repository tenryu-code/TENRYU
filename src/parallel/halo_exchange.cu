#include <cstdlib>
#include <cstdio>
#include <chrono>
#include <atomic>
#include "parallel/halo_exchange.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"

#if TENRYU_ENABLE_MPI
#include <mpi.h>
#endif

namespace {
// TENRYU_COMM_PROFILE (design doc §6q.3): env-gated per-entry-point call
// counters. Zero effect without the env (one branch per call). MPI-
// independent on purpose: the wrapper entry points below compile in both
// MPI modes, so these must too (the OFF build broke when they lived
// inside the #if — caught by audit_heisenbug_4config_smoke).
struct CommProfSlot {
  const char* name;
  std::atomic<long> calls{0};
  std::atomic<double> total_s{0.0};
};
inline bool comm_prof_enabled() {
  static const bool on = std::getenv("TENRYU_COMM_PROFILE") != nullptr;
  return on;
}
inline void comm_prof_add(CommProfSlot& slot, const double dt_s) {
  slot.calls.fetch_add(1, std::memory_order_relaxed);
  double cur = slot.total_s.load(std::memory_order_relaxed);
  while (!slot.total_s.compare_exchange_weak(cur, cur + dt_s)) {
  }
}
struct CommProfRegistry {
  std::vector<CommProfSlot*> slots;
  ~CommProfRegistry() {
    if (!comm_prof_enabled()) {
      return;
    }
    for (CommProfSlot* s : slots) {
      if (s->calls.load() == 0) {
        continue;
      }
      const long c = s->calls.load();
      const double t = s->total_s.load();
      std::fprintf(stderr,
                   "[comm-profile] %s calls=%ld total=%.3fs avg=%.3fms\n",
                   s->name, c, t, c > 0 ? 1e3 * t / c : 0.0);
    }
  }
};
CommProfRegistry& comm_prof_registry() {
  static CommProfRegistry reg;
  return reg;
}
struct CommProfTimer {
  CommProfSlot& slot;
  std::chrono::steady_clock::time_point t0;
  bool armed;
  explicit CommProfTimer(CommProfSlot& s) : slot(s), armed(comm_prof_enabled()) {
    if (armed) {
      t0 = std::chrono::steady_clock::now();
    }
  }
  ~CommProfTimer() {
    if (armed) {
      const auto t1 = std::chrono::steady_clock::now();
      comm_prof_add(slot,
                    std::chrono::duration<double>(t1 - t0).count());
    }
  }
};
}  // namespace

namespace tenryu::parallel {
namespace {

// Option C layout (docs/design/mpi_m18_20_20260717.md §3): all field arrays
// are GLOBAL-size on every rank. Halo exchange sends the owned edge strips
// and receives the neighbor-owned strips, both addressed at their natural
// global indices. Regions below are therefore expressed in global (i, j)
// with flat index = i * stride + j.
struct ExchangeRegion {
  int i0 = 0;
  int j0 = 0;
  int ni = 0;
  int nj = 0;
};

struct ExchangeGeometry {
  int ghost = 0;
  // Owned index range in the global array ([begin, end), cells or nodes).
  int i_begin = 0;
  int i_end = 0;
  int j_begin = 0;
  int j_end = 0;
  int stride = 0;      // flat = i * stride + j
  int array_ni = 0;    // global array extents (for the field_size check)
  int array_nj = 0;
  bool node_mode = false;  // nodes use the owner-overwrite region scheme
};

#if TENRYU_ENABLE_MPI
constexpr int kThreadsPerBlock = 256;

bool is_direction_active(const PartitionInfo& part, const int direction) {
  if (direction == LEFT || direction == RIGHT) {
    return true;
  }
  if (part.cart_dims[1] == 1) {
    return false;
  }
  return true;
}

int opposite_direction(const int direction) {
  switch (direction) {
    case LEFT:
      return RIGHT;
    case RIGHT:
      return LEFT;
    case BOTTOM:
      return TOP;
    case TOP:
      return BOTTOM;
    case NE:
      return SW;
    case NW:
      return SE;
    case SE:
      return NW;
    case SW:
      return NE;
    default:
      return direction;
  }
}

int halo_tag(const int phase_id, const int direction) {
  return phase_id * 1000 + direction * 100;
}
#endif

ExchangeRegion clip_region(const ExchangeRegion& region,
                           const ExchangeGeometry& geom) {
  ExchangeRegion out = region;
  if (out.i0 < 0) {
    out.ni += out.i0;
    out.i0 = 0;
  }
  if (out.j0 < 0) {
    out.nj += out.j0;
    out.j0 = 0;
  }
  if (out.i0 + out.ni > geom.array_ni) {
    out.ni = geom.array_ni - out.i0;
  }
  if (out.j0 + out.nj > geom.array_nj) {
    out.nj = geom.array_nj - out.j0;
  }
  if (out.ni < 0) out.ni = 0;
  if (out.nj < 0) out.nj = 0;
  return out;
}

// Cell fields: send the owned edge strips (g layers just inside the owned
// slab); receive the neighbor-owned strips (g layers just outside). The
// transverse extent is the owned range only; diagonal ghosts are covered by
// the explicit corner phase, so face messages carry no stale corner data.
std::array<ExchangeRegion, MAX_NEIGHBORS> build_cell_send_regions(
    const ExchangeGeometry& geom) {
  std::array<ExchangeRegion, MAX_NEIGHBORS> out{};
  const int g = geom.ghost;
  const int ni = geom.i_end - geom.i_begin;
  const int nj = geom.j_end - geom.j_begin;
  if (g <= 0 || ni <= 0 || nj <= 0) {
    return out;
  }

  out[LEFT] = ExchangeRegion{geom.i_begin, geom.j_begin, g, nj};
  out[RIGHT] = ExchangeRegion{geom.i_end - g, geom.j_begin, g, nj};
  out[BOTTOM] = ExchangeRegion{geom.i_begin, geom.j_begin, ni, g};
  out[TOP] = ExchangeRegion{geom.i_begin, geom.j_end - g, ni, g};

  out[NE] = ExchangeRegion{geom.i_end - g, geom.j_end - g, g, g};
  out[NW] = ExchangeRegion{geom.i_begin, geom.j_end - g, g, g};
  out[SE] = ExchangeRegion{geom.i_end - g, geom.j_begin, g, g};
  out[SW] = ExchangeRegion{geom.i_begin, geom.j_begin, g, g};
  return out;
}

std::array<ExchangeRegion, MAX_NEIGHBORS> build_cell_recv_regions(
    const ExchangeGeometry& geom) {
  std::array<ExchangeRegion, MAX_NEIGHBORS> out{};
  const int g = geom.ghost;
  const int ni = geom.i_end - geom.i_begin;
  const int nj = geom.j_end - geom.j_begin;
  if (g <= 0 || ni <= 0 || nj <= 0) {
    return out;
  }

  out[LEFT] = clip_region(ExchangeRegion{geom.i_begin - g, geom.j_begin, g, nj}, geom);
  out[RIGHT] = clip_region(ExchangeRegion{geom.i_end, geom.j_begin, g, nj}, geom);
  out[BOTTOM] = clip_region(ExchangeRegion{geom.i_begin, geom.j_begin - g, ni, g}, geom);
  out[TOP] = clip_region(ExchangeRegion{geom.i_begin, geom.j_end, ni, g}, geom);

  out[NE] = clip_region(ExchangeRegion{geom.i_end, geom.j_end, g, g}, geom);
  out[NW] = clip_region(ExchangeRegion{geom.i_begin - g, geom.j_end, g, g}, geom);
  out[SE] = clip_region(ExchangeRegion{geom.i_end, geom.j_begin - g, g, g}, geom);
  out[SW] = clip_region(ExchangeRegion{geom.i_begin - g, geom.j_begin - g, g, g}, geom);
  return out;
}

// Node fields (slab decompositions only): the shared node plane at a slab
// interface is stored by both ranks; the lower-coordinate side owns it
// (NUMERICS §12.1.3 min-rank rule). Regions are asymmetric:
//  - toward the lower side (LEFT/BOTTOM): send my g interior planes past the
//    shared plane; receive the shared plane (owner overwrite) plus the
//    peer's g interior planes.
//  - toward the upper side (RIGHT/TOP): send my g interior planes plus the
//    shared plane I own; receive the peer's g interior planes beyond it.
// Sizes pair up: my LEFT send (g) <-> peer RIGHT recv (g); my LEFT recv
// (g+1) <-> peer RIGHT send (g+1).
std::array<ExchangeRegion, MAX_NEIGHBORS> build_node_send_regions(
    const ExchangeGeometry& geom) {
  std::array<ExchangeRegion, MAX_NEIGHBORS> out{};
  const int g = geom.ghost;
  const int ni = geom.i_end - geom.i_begin;
  const int nj = geom.j_end - geom.j_begin;
  if (g <= 0 || ni <= 0 || nj <= 0) {
    return out;
  }

  out[LEFT] = ExchangeRegion{geom.i_begin + 1, geom.j_begin, g, nj};
  out[RIGHT] = ExchangeRegion{geom.i_end - 1 - g, geom.j_begin, g + 1, nj};
  out[BOTTOM] = ExchangeRegion{geom.i_begin, geom.j_begin + 1, ni, g};
  out[TOP] = ExchangeRegion{geom.i_begin, geom.j_end - 1 - g, ni, g + 1};
  return out;
}

std::array<ExchangeRegion, MAX_NEIGHBORS> build_node_recv_regions(
    const ExchangeGeometry& geom) {
  std::array<ExchangeRegion, MAX_NEIGHBORS> out{};
  const int g = geom.ghost;
  const int ni = geom.i_end - geom.i_begin;
  const int nj = geom.j_end - geom.j_begin;
  if (g <= 0 || ni <= 0 || nj <= 0) {
    return out;
  }

  out[LEFT] =
      clip_region(ExchangeRegion{geom.i_begin - g, geom.j_begin, g + 1, nj}, geom);
  out[RIGHT] = clip_region(ExchangeRegion{geom.i_end, geom.j_begin, g, nj}, geom);
  out[BOTTOM] =
      clip_region(ExchangeRegion{geom.i_begin, geom.j_begin - g, ni, g + 1}, geom);
  out[TOP] = clip_region(ExchangeRegion{geom.i_begin, geom.j_end, ni, g}, geom);
  return out;
}

#if TENRYU_ENABLE_MPI
template <typename T>
std::size_t direction_bytes(const ExchangeRegion& region, const int n_fields) {
  if (region.ni <= 0 || region.nj <= 0 || n_fields <= 0) {
    return 0;
  }
  const std::size_t cell_count =
      static_cast<std::size_t>(region.ni) * static_cast<std::size_t>(region.nj);
  return cell_count * static_cast<std::size_t>(n_fields) * sizeof(T);
}

template <typename T>
__global__ void pack_halo_kernel(T* __restrict__ send_buffer,
                                 T* const* __restrict__ field_ptrs,
                                 const int n_fields, const int i0,
                                 const int j0, const int ni, const int nj,
                                 const int stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = ni * nj;
  if (idx >= total) {
    return;
  }

  const int local_i = idx / nj;
  const int local_j = idx % nj;
  const int i = i0 + local_i;
  const int j = j0 + local_j;
  const int flat = i * stride + j;
  const int base = idx * n_fields;
  for (int field = 0; field < n_fields; ++field) {
    send_buffer[base + field] = field_ptrs[field][flat];
  }
}

template <typename T>
__global__ void unpack_halo_kernel(const T* __restrict__ recv_buffer,
                                   T* const* __restrict__ field_ptrs,
                                   const int n_fields, const int i0,
                                   const int j0, const int ni, const int nj,
                                   const int stride) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = ni * nj;
  if (idx >= total) {
    return;
  }

  const int local_i = idx / nj;
  const int local_j = idx % nj;
  const int i = i0 + local_i;
  const int j = j0 + local_j;
  const int flat = i * stride + j;
  const int base = idx * n_fields;
  for (int field = 0; field < n_fields; ++field) {
    field_ptrs[field][flat] = recv_buffer[base + field];
  }
}

template <typename T>
void launch_pack(CommBuffers& buffers, T* const* device_field_ptrs,
                 const ExchangeRegion& region, const int direction,
                 const int n_fields, const int stride,
                 cudaStream_t stream) {
  const int total = region.ni * region.nj;
  if (total <= 0) {
    return;
  }

  T* const send_ptr = buffers.send_halo[direction].as<T>();
  const int blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  pack_halo_kernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      send_ptr, device_field_ptrs, n_fields, region.i0, region.j0, region.ni,
      region.nj, stride);
}

template <typename T>
void launch_unpack(CommBuffers& buffers, T* const* device_field_ptrs,
                   const ExchangeRegion& region, const int direction,
                   const int n_fields, const int stride,
                   cudaStream_t stream) {
  const int total = region.ni * region.nj;
  if (total <= 0) {
    return;
  }

  const T* const recv_ptr = buffers.recv_halo[direction].as<T>();
  const int blocks = (total + kThreadsPerBlock - 1) / kThreadsPerBlock;
  unpack_halo_kernel<T><<<blocks, kThreadsPerBlock, 0, stream>>>(
      recv_ptr, device_field_ptrs, n_fields, region.i0, region.j0, region.ni,
      region.nj, stride);
}

std::string mpi_error_message(const char* operation) {
  return std::string(operation) + " failed in halo exchange";
}

MPI_Comm halo_comm(const PartitionInfo& part) {
  if (part.cart_comm != MPI_COMM_NULL) {
    return part.cart_comm;
  }
  return MPI_COMM_WORLD;
}

template <typename T>
void exchange_phase(const PartitionInfo& part, CommBuffers& buffers,
                    T* const* device_field_ptrs, const int n_fields,
                    const int phase_id,
                    const std::array<ExchangeRegion, MAX_NEIGHBORS>& send_regions,
                    const std::array<ExchangeRegion, MAX_NEIGHBORS>& recv_regions,
                    const int* directions, const int n_directions,
                    const bool use_gpu_aware_mpi, const int stride,
                    cudaStream_t stream) {
  std::array<int, 4> active_dirs{};
  std::array<std::size_t, 4> active_send_bytes{};
  std::array<std::size_t, 4> active_recv_bytes{};
  int active_count = 0;

  for (int idx = 0; idx < n_directions; ++idx) {
    const int direction = directions[idx];
    if (!is_direction_active(part, direction)) {
      continue;
    }
    if (part.neighbor_ranks[direction] < 0) {
      continue;
    }

    const std::size_t send_bytes =
        direction_bytes<T>(send_regions[direction], n_fields);
    const std::size_t recv_bytes =
        direction_bytes<T>(recv_regions[direction], n_fields);
    if (send_bytes == 0 && recv_bytes == 0) {
      continue;
    }

    if (send_bytes > 0) {
      launch_pack(buffers, device_field_ptrs, send_regions[direction], direction,
                  n_fields, stride, stream);
    }
    active_dirs[active_count] = direction;
    active_send_bytes[active_count] = send_bytes;
    active_recv_bytes[active_count] = recv_bytes;
    ++active_count;
  }

  if (active_count == 0) {
    return;
  }

  if (use_gpu_aware_mpi) {
    CUDA_CHECK(cudaStreamSynchronize(stream));
  } else {
    for (int idx = 0; idx < active_count; ++idx) {
      const int direction = active_dirs[idx];
      if (active_send_bytes[idx] == 0) {
        continue;
      }
      CUDA_CHECK(cudaMemcpyAsync(
          buffers.host_send[direction].ptr, buffers.send_halo[direction].ptr,
          active_send_bytes[idx], cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  std::array<MPI_Request, 8> requests{};
  int request_count = 0;
  const MPI_Comm comm = halo_comm(part);

  for (int idx = 0; idx < active_count; ++idx) {
    const int direction = active_dirs[idx];
    if (active_recv_bytes[idx] == 0) {
      continue;
    }
    const int recv_direction = opposite_direction(direction);
    TENRYU_ASSERT(
        active_recv_bytes[idx] <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "halo exchange message exceeds MPI int byte-count limit");
    const int mpi_bytes = static_cast<int>(active_recv_bytes[idx]);
    void* recv_ptr = use_gpu_aware_mpi ? buffers.recv_halo[direction].ptr
                                       : buffers.host_recv[direction].ptr;
    const int recv_err =
        MPI_Irecv(recv_ptr, mpi_bytes, MPI_BYTE, part.neighbor_ranks[direction],
                  halo_tag(phase_id, recv_direction), comm,
                  &requests[request_count++]);
    TENRYU_ASSERT(recv_err == MPI_SUCCESS, mpi_error_message("MPI_Irecv"));
  }

  for (int idx = 0; idx < active_count; ++idx) {
    const int direction = active_dirs[idx];
    if (active_send_bytes[idx] == 0) {
      continue;
    }
    TENRYU_ASSERT(
        active_send_bytes[idx] <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
        "halo exchange message exceeds MPI int byte-count limit");
    const int mpi_bytes = static_cast<int>(active_send_bytes[idx]);
    const void* send_ptr = use_gpu_aware_mpi ? buffers.send_halo[direction].ptr
                                             : buffers.host_send[direction].ptr;
    const int send_err =
        MPI_Isend(const_cast<void*>(send_ptr), mpi_bytes, MPI_BYTE,
                  part.neighbor_ranks[direction], halo_tag(phase_id, direction),
                  comm, &requests[request_count++]);
    TENRYU_ASSERT(send_err == MPI_SUCCESS, mpi_error_message("MPI_Isend"));
  }

  const int wait_err = MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE);
  TENRYU_ASSERT(wait_err == MPI_SUCCESS, mpi_error_message("MPI_Waitall"));

  if (!use_gpu_aware_mpi) {
    for (int idx = 0; idx < active_count; ++idx) {
      const int direction = active_dirs[idx];
      if (active_recv_bytes[idx] == 0) {
        continue;
      }
      CUDA_CHECK(cudaMemcpyAsync(
          buffers.recv_halo[direction].ptr, buffers.host_recv[direction].ptr,
          active_recv_bytes[idx], cudaMemcpyHostToDevice, stream));
    }
  }

  for (int idx = 0; idx < active_count; ++idx) {
    const int direction = active_dirs[idx];
    if (active_recv_bytes[idx] == 0) {
      continue;
    }
    launch_unpack(buffers, device_field_ptrs, recv_regions[direction], direction,
                  n_fields, stride, stream);
  }
}
#endif

template <typename T>
void exchange_fields_impl(const PartitionInfo& part, CommBuffers& buffers,
                          T* const* field_ptrs, const int n_fields,
                          const int field_size, const int phase_id,
                          const ExchangeGeometry& geom, cudaStream_t stream) {
  if (part.n_ranks <= 1) {
    return;
  }
  if (n_fields <= 0) {
    return;
  }

  TENRYU_ASSERT(field_ptrs != nullptr,
                "halo exchange requires a non-null field_ptrs array");
  TENRYU_ASSERT(geom.ghost > 0,
                "halo exchange requires ghost_layers > 0 for multi-rank runs");
  TENRYU_ASSERT(geom.array_ni > 0 && geom.array_nj > 0,
                "halo exchange requires positive global array extents");
  TENRYU_ASSERT(geom.stride > 0, "halo exchange requires a positive stride");
  const int required_size = geom.array_ni * geom.array_nj;
  TENRYU_ASSERT(field_size >= required_size,
                "halo exchange field_size is smaller than the global array size");
  if (geom.node_mode) {
    TENRYU_ASSERT(part.is_slab(),
                  "node halo exchange supports slab decompositions only "
                  "(single-axis split) in v1");
  }

#if TENRYU_ENABLE_MPI
  bool use_gpu_aware_mpi = buffers.gpu_aware_mpi;
#if TENRYU_GPU_AWARE_MPI_COMPILE == 0
  use_gpu_aware_mpi = false;
#endif
  if (!use_gpu_aware_mpi && buffers.gpu_aware_mpi) {
    buffers.gpu_aware_mpi = false;
  }

  const auto send_regions =
      geom.node_mode ? build_node_send_regions(geom) : build_cell_send_regions(geom);
  const auto recv_regions =
      geom.node_mode ? build_node_recv_regions(geom) : build_cell_recv_regions(geom);

  std::size_t required_bytes = 0;
  for (int direction = 0; direction < MAX_NEIGHBORS; ++direction) {
    if (!is_direction_active(part, direction)) {
      continue;
    }
    if (part.neighbor_ranks[direction] < 0) {
      continue;
    }
    required_bytes =
        std::max(required_bytes, direction_bytes<T>(send_regions[direction], n_fields));
    required_bytes =
        std::max(required_bytes, direction_bytes<T>(recv_regions[direction], n_fields));
  }
  if (required_bytes == 0) {
    return;
  }

  buffers.resize_if_needed(required_bytes);

  DeviceArray device_field_ptrs_storage;
  device_field_ptrs_storage.resize(static_cast<std::size_t>(n_fields) * sizeof(T*));
  CUDA_CHECK(cudaMemcpyAsync(device_field_ptrs_storage.ptr, field_ptrs,
                             static_cast<std::size_t>(n_fields) * sizeof(T*),
                             cudaMemcpyHostToDevice, stream));
  T* const* device_field_ptrs = device_field_ptrs_storage.as<T*>();

  constexpr int kFaceDirections[4] = {LEFT, RIGHT, BOTTOM, TOP};
  constexpr int kCornerDirections[4] = {NE, NW, SE, SW};

  exchange_phase(part, buffers, device_field_ptrs, n_fields, phase_id,
                 send_regions, recv_regions, kFaceDirections, 4,
                 use_gpu_aware_mpi, geom.stride, stream);
  if (!geom.node_mode) {
    exchange_phase(part, buffers, device_field_ptrs, n_fields, phase_id,
                   send_regions, recv_regions, kCornerDirections, 4,
                   use_gpu_aware_mpi, geom.stride, stream);
  }

  CUDA_CHECK(cudaGetLastError());
#else
  (void)buffers;
  (void)field_ptrs;
  (void)phase_id;
  (void)stream;
#endif
}

ExchangeGeometry cell_geometry(const PartitionInfo& part) {
  ExchangeGeometry geom{};
  geom.ghost = part.ghost_layers;
  geom.i_begin = part.local_cell_range[0][0];
  geom.i_end = part.local_cell_range[0][1];
  geom.j_begin = part.local_cell_range[1][0];
  geom.j_end = part.local_cell_range[1][1];
  geom.array_ni = part.global_nr;
  geom.array_nj = part.global_nz > 0 ? part.global_nz : 1;
  geom.stride = geom.array_nj;
  geom.node_mode = false;
  return geom;
}

ExchangeGeometry node_geometry(const PartitionInfo& part) {
  ExchangeGeometry geom{};
  geom.ghost = part.ghost_layers;
  geom.i_begin = part.local_node_range[0][0];
  geom.i_end = part.local_node_range[0][1];
  geom.j_begin = part.local_node_range[1][0];
  geom.j_end = part.local_node_range[1][1];
  const bool is_1d = (part.local_node_range[1][1] - part.local_node_range[1][0]) ==
                     (part.local_cell_range[1][1] - part.local_cell_range[1][0]);
  geom.array_ni = part.global_nr + 1;
  geom.array_nj = is_1d ? (part.global_nz > 0 ? part.global_nz : 1)
                        : (part.global_nz + 1);
  geom.stride = geom.array_nj;
  geom.node_mode = true;
  return geom;
}

}  // namespace

void exchange_cell_fields(const PartitionInfo& part, CommBuffers& buffers,
                          double* const* field_ptrs, const int n_fields,
                          const int field_size, cudaStream_t stream,
                          const int phase_id) {
  static CommProfSlot comm_prof_slot_x_cell{"x_cell"};
  static const bool comm_prof_reg_x_cell =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_x_cell),
       true);
  (void)comm_prof_reg_x_cell;
  CommProfTimer comm_prof_timer_x_cell(comm_prof_slot_x_cell);
  exchange_fields_impl(part, buffers, field_ptrs, n_fields, field_size, phase_id,
                       cell_geometry(part), stream);
}

void exchange_int8_fields(const PartitionInfo& part, CommBuffers& buffers,
                          int8_t* const* field_ptrs, const int n_fields,
                          const int field_size, cudaStream_t stream,
                          const int phase_id) {
  static CommProfSlot comm_prof_slot_x_int8{"x_int8"};
  static const bool comm_prof_reg_x_int8 =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_x_int8),
       true);
  (void)comm_prof_reg_x_int8;
  CommProfTimer comm_prof_timer_x_int8(comm_prof_slot_x_int8);
  exchange_fields_impl(part, buffers, field_ptrs, n_fields, field_size, phase_id,
                       cell_geometry(part), stream);
}

void exchange_node_fields(const PartitionInfo& part, CommBuffers& buffers,
                          double* const* field_ptrs, const int n_fields,
                          const int field_size, cudaStream_t stream,
                          const int phase_id) {
  static CommProfSlot comm_prof_slot_x_node{"x_node"};
  static const bool comm_prof_reg_x_node =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_x_node),
       true);
  (void)comm_prof_reg_x_node;
  CommProfTimer comm_prof_timer_x_node(comm_prof_slot_x_node);
  exchange_fields_impl(part, buffers, field_ptrs, n_fields, field_size, phase_id,
                       node_geometry(part), stream);
}

void exchange_cell_strips_scaled(const PartitionInfo& part,
                                 CommBuffers& buffers, double* field,
                                 const int elems_per_cell, const int phase_id) {
  static CommProfSlot comm_prof_slot_x_strips{"x_strips"};
  static const bool comm_prof_reg_x_strips =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_x_strips),
       true);
  (void)comm_prof_reg_x_strips;
  CommProfTimer comm_prof_timer_x_strips(comm_prof_slot_x_strips);
  if (part.n_ranks <= 1 || field == nullptr || elems_per_cell <= 0) {
    return;
  }
#if TENRYU_ENABLE_MPI
  TENRYU_ASSERT(part.is_slab() && part.cart_dims[1] == 1,
                "exchange_cell_strips_scaled supports r-slab decompositions "
                "only in v1");
  const int nz = part.global_nz > 0 ? part.global_nz : 1;
  const int g = part.ghost_layers;
  const int i0 = part.local_cell_range[0][0];
  const int i1 = part.local_cell_range[0][1];
  const std::size_t strip_elems = static_cast<std::size_t>(g) * nz *
                                  static_cast<std::size_t>(elems_per_cell);
  const std::size_t strip_bytes = strip_elems * sizeof(double);
  TENRYU_ASSERT(strip_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "scaled strip exceeds MPI int byte-count limit");

  bool use_gpu_aware_mpi = buffers.gpu_aware_mpi;
#if TENRYU_GPU_AWARE_MPI_COMPILE == 0
  use_gpu_aware_mpi = false;
#endif
  const MPI_Comm comm = halo_comm(part);
  const auto strip_ptr = [&](const int cell_row) {
    return field + static_cast<std::size_t>(cell_row) * nz * elems_per_cell;
  };

  std::array<MPI_Request, 4> requests{};
  int request_count = 0;
  // Host staging (single pinned buffer per direction slot reused).
  std::array<std::vector<double>, 2> host_send;
  std::array<std::vector<double>, 2> host_recv;
  const int dirs[2] = {LEFT, RIGHT};
  // send: LEFT -> owned rows [i0, i0+g); RIGHT -> [i1-g, i1)
  // recv: LEFT ghost [i0-g, i0); RIGHT ghost [i1, i1+g)
  for (int k = 0; k < 2; ++k) {
    const int direction = dirs[k];
    if (part.neighbor_ranks[direction] < 0) {
      continue;
    }
    double* recv_dst = (direction == LEFT) ? strip_ptr(i0 - g) : strip_ptr(i1);
    void* recv_ptr = recv_dst;
    if (!use_gpu_aware_mpi) {
      host_recv[k].resize(strip_elems);
      recv_ptr = host_recv[k].data();
    }
    const int recv_err = MPI_Irecv(
        recv_ptr, static_cast<int>(strip_bytes), MPI_BYTE,
        part.neighbor_ranks[direction],
        halo_tag(phase_id, opposite_direction(direction)), comm,
        &requests[request_count++]);
    TENRYU_ASSERT(recv_err == MPI_SUCCESS,
                  mpi_error_message("MPI_Irecv(scaled strips)"));
  }
  for (int k = 0; k < 2; ++k) {
    const int direction = dirs[k];
    if (part.neighbor_ranks[direction] < 0) {
      continue;
    }
    double* send_src = (direction == LEFT) ? strip_ptr(i0) : strip_ptr(i1 - g);
    const void* send_ptr = send_src;
    if (!use_gpu_aware_mpi) {
      host_send[k].resize(strip_elems);
      CUDA_CHECK(cudaMemcpy(host_send[k].data(), send_src, strip_bytes,
                            cudaMemcpyDeviceToHost));
      send_ptr = host_send[k].data();
    }
    const int send_err = MPI_Isend(
        const_cast<void*>(send_ptr), static_cast<int>(strip_bytes), MPI_BYTE,
        part.neighbor_ranks[direction], halo_tag(phase_id, direction), comm,
        &requests[request_count++]);
    TENRYU_ASSERT(send_err == MPI_SUCCESS,
                  mpi_error_message("MPI_Isend(scaled strips)"));
  }
  const int wait_err =
      MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE);
  TENRYU_ASSERT(wait_err == MPI_SUCCESS,
                mpi_error_message("MPI_Waitall(scaled strips)"));
  if (!use_gpu_aware_mpi) {
    for (int k = 0; k < 2; ++k) {
      const int direction = dirs[k];
      if (part.neighbor_ranks[direction] < 0 || host_recv[k].empty()) {
        continue;
      }
      double* recv_dst = (direction == LEFT) ? strip_ptr(i0 - g) : strip_ptr(i1);
      CUDA_CHECK(cudaMemcpy(recv_dst, host_recv[k].data(), strip_bytes,
                            cudaMemcpyHostToDevice));
    }
  }
#else
  (void)buffers;
  (void)phase_id;
#endif
}

void sendrecv_add_planes(const PartitionInfo& part, CommBuffers& buffers,
                         double* base, const int plane_elems,
                         const std::size_t left_offset,
                         const std::size_t right_offset,
                         const int phase_id) {
  sendrecv_add_planes_asym(part, buffers, base, plane_elems, left_offset,
                           left_offset, right_offset, right_offset,
                           phase_id);
}

void sendrecv_add_planes_asym(const PartitionInfo& part,
                              CommBuffers& buffers, double* base,
                              const int plane_elems,
                              const std::size_t send_left_offset,
                              const std::size_t recv_left_offset,
                              const std::size_t send_right_offset,
                              const std::size_t recv_right_offset,
                              const int phase_id) {
  static CommProfSlot comm_prof_slot_x_planes{"x_planes"};
  static const bool comm_prof_reg_x_planes =
      (comm_prof_registry().slots.push_back(&comm_prof_slot_x_planes),
       true);
  (void)comm_prof_reg_x_planes;
  CommProfTimer comm_prof_timer_x_planes(comm_prof_slot_x_planes);
  if (part.n_ranks <= 1 || base == nullptr || plane_elems <= 0) {
    return;
  }
#if TENRYU_ENABLE_MPI
  TENRYU_ASSERT(part.is_slab() && part.cart_dims[1] == 1,
                "sendrecv_add_planes supports r-slab decompositions only "
                "in v1");
  const std::size_t plane_bytes =
      static_cast<std::size_t>(plane_elems) * sizeof(double);
  TENRYU_ASSERT(
      plane_bytes <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
      "interface plane exceeds MPI int byte-count limit");
  const MPI_Comm comm = halo_comm(part);
  const int dirs[2] = {LEFT, RIGHT};
  const std::size_t send_offsets[2] = {send_left_offset, send_right_offset};
  const std::size_t recv_offsets[2] = {recv_left_offset, recv_right_offset};
  // Host staging throughout (the add happens host-side; planes are tiny).
  std::array<std::vector<double>, 2> host_send;
  std::array<std::vector<double>, 2> host_recv;
  std::array<MPI_Request, 4> requests{};
  int request_count = 0;
  for (int k = 0; k < 2; ++k) {
    const int direction = dirs[k];
    if (part.neighbor_ranks[direction] < 0) {
      continue;
    }
    host_recv[k].resize(static_cast<std::size_t>(plane_elems));
    const int recv_err = MPI_Irecv(
        host_recv[k].data(), static_cast<int>(plane_bytes), MPI_BYTE,
        part.neighbor_ranks[direction],
        halo_tag(phase_id, opposite_direction(direction)), comm,
        &requests[request_count++]);
    TENRYU_ASSERT(recv_err == MPI_SUCCESS,
                  mpi_error_message("MPI_Irecv(add planes)"));
  }
  for (int k = 0; k < 2; ++k) {
    const int direction = dirs[k];
    if (part.neighbor_ranks[direction] < 0) {
      continue;
    }
    host_send[k].resize(static_cast<std::size_t>(plane_elems));
    CUDA_CHECK(cudaMemcpy(host_send[k].data(), base + send_offsets[k],
                          plane_bytes, cudaMemcpyDeviceToHost));
    const int send_err = MPI_Isend(
        host_send[k].data(), static_cast<int>(plane_bytes), MPI_BYTE,
        part.neighbor_ranks[direction], halo_tag(phase_id, direction), comm,
        &requests[request_count++]);
    TENRYU_ASSERT(send_err == MPI_SUCCESS,
                  mpi_error_message("MPI_Isend(add planes)"));
  }
  const int wait_err =
      MPI_Waitall(request_count, requests.data(), MPI_STATUSES_IGNORE);
  TENRYU_ASSERT(wait_err == MPI_SUCCESS,
                mpi_error_message("MPI_Waitall(add planes)"));
  for (int k = 0; k < 2; ++k) {
    const int direction = dirs[k];
    if (part.neighbor_ranks[direction] < 0 || host_recv[k].empty()) {
      continue;
    }
    // In-place completion at the RECV slot: its current local content
    // plus the neighbor's message (for the symmetric completion the two
    // partials are disjoint by construction; for the asym transfer the
    // local recv-slot content is zero, so the add is an owner->ghost
    // copy).
    std::vector<double> local(static_cast<std::size_t>(plane_elems));
    CUDA_CHECK(cudaMemcpy(local.data(), base + recv_offsets[k], plane_bytes,
                          cudaMemcpyDeviceToHost));
    for (int e = 0; e < plane_elems; ++e) {
      local[static_cast<std::size_t>(e)] +=
          host_recv[k][static_cast<std::size_t>(e)];
    }
    CUDA_CHECK(cudaMemcpy(base + recv_offsets[k], local.data(), plane_bytes,
                          cudaMemcpyHostToDevice));
  }
#else
  (void)buffers;
  (void)send_left_offset;
  (void)recv_left_offset;
  (void)send_right_offset;
  (void)recv_right_offset;
  (void)phase_id;
#endif
}

void HaloExchange::exchange() {
}

}  // namespace tenryu::parallel
