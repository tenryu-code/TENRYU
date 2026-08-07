#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"

namespace tenryu::parallel {

void exchange_cell_fields(const PartitionInfo& part, CommBuffers& buffers,
                          double* const* field_ptrs, int n_fields,
                          int field_size, cudaStream_t stream,
                          int phase_id = 1);

void exchange_int8_fields(const PartitionInfo& part, CommBuffers& buffers,
                          int8_t* const* field_ptrs, int n_fields,
                          int field_size, cudaStream_t stream,
                          int phase_id = 1);

void exchange_node_fields(const PartitionInfo& part, CommBuffers& buffers,
                          double* const* field_ptrs, int n_fields,
                          int field_size, cudaStream_t stream,
                          int phase_id = 2);

// r-slab-only exchange of a cell field with elems_per_cell interleaved
// components (cell-major minor layout, e.g. rad_E[c*G+g]): the owned edge
// strips and ghost strips are CONTIGUOUS memory under the r-major flat
// index, so this exchanges plain contiguous blocks (no pack kernel).
void exchange_cell_strips_scaled(const PartitionInfo& part,
                                 CommBuffers& buffers, double* field,
                                 int elems_per_cell, int phase_id = 4);

// r-slab-only SUM-completion of shared interface planes: both ranks
// adjacent to an interface hold DISJOINT partial contributions at the
// same global plane (e.g. SN unique r-face fluxes, where forward and
// backward directions are written by opposite sides); each side sends
// its contiguous plane [offset, offset + plane_elems) and ADDS the
// received partial in place, so both end with the full sum. left_offset
// addresses this rank's lower interface plane, right_offset its upper
// one (matching the neighbor's opposite offset by construction). No-op
// when n_ranks <= 1.
void sendrecv_add_planes(const PartitionInfo& part, CommBuffers& buffers,
                         double* base, int plane_elems,
                         std::size_t left_offset, std::size_t right_offset,
                         int phase_id = 9);

// Generalized form with separate send/recv offsets per side: send
// base[send_*_offset, +plane_elems) to that neighbor and ADD its message
// into base[recv_*_offset, +plane_elems). With recv slots that are zero
// on the receiver (e.g. a face plane wholly owned by the neighbor) the
// add acts as an owner->ghost transfer. The symmetric-completion form
// above is the send==recv special case.
void sendrecv_add_planes_asym(const PartitionInfo& part,
                              CommBuffers& buffers, double* base,
                              int plane_elems, std::size_t send_left_offset,
                              std::size_t recv_left_offset,
                              std::size_t send_right_offset,
                              std::size_t recv_right_offset,
                              int phase_id = 9);

class HaloExchange {
 public:
  HaloExchange() = default;

  void exchange();
};

}  // namespace tenryu::parallel
