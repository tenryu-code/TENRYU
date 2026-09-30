#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::parallel {

struct alignas(8) ParticleEmigrant {
  double position[3];
  double direction[3];
  double energy;
  double weight;
  double time_remain;
  double birth_energy;
  std::uint64_t global_id;
  std::uint32_t rng_counter;
  std::int32_t cell_id_src;
  std::uint16_t group;
  std::int8_t sign;
  std::int8_t leak_face;
  std::uint8_t mode;
  std::int8_t padding[3];
};

static_assert(sizeof(ParticleEmigrant) == 104,
              "ParticleEmigrant must be 104 bytes");

struct EmigrantBuffer {
  ParticleEmigrant* data = nullptr;
  std::int32_t* dest_rank = nullptr;
  int count = 0;
  int capacity = 0;
  int per_dest_count[8] = {};
  int per_dest_offset[8] = {};
  static constexpr int BYTES_PER_PARTICLE = 104;

  EmigrantBuffer() = default;
  EmigrantBuffer(const EmigrantBuffer&) = delete;
  EmigrantBuffer& operator=(const EmigrantBuffer&) = delete;
  EmigrantBuffer(EmigrantBuffer&& other) noexcept;
  EmigrantBuffer& operator=(EmigrantBuffer&& other) noexcept;

  void resize_if_needed(int required);
  ~EmigrantBuffer();
};

int detect_emigrants(const radiation::PhotonPool& pool, const PartitionInfo& part,
                     EmigrantBuffer& emigrants, cudaStream_t stream);

void exchange_emigrants(const PartitionInfo& part, CommBuffers& buffers,
                        const EmigrantBuffer& send, EmigrantBuffer& recv,
                        cudaStream_t stream = nullptr);

void merge_immigrants(radiation::PhotonPool& pool, const EmigrantBuffer& recv,
                      bool sort_by_global_id, cudaStream_t stream = nullptr);

class ParticleMigration {
 public:
  ParticleMigration() = default;

  void migrate();
};

}  // namespace tenryu::parallel
