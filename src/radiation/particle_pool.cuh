#pragma once

#include <cstdint>

namespace tenryu::radiation {

enum ParticleMode : std::uint8_t {
  kModeIMC = 0,
  kModeDDMC = 1,
  kModeRW = 2,
};

enum ParticleStatus : std::uint8_t {
  kDead = 0,
  kAlive = 1,
  kOverflow = 2,
};

struct PhotonPool {
  // Position and direction.
  double* pos_r = nullptr;
  double* pos_z = nullptr;
  double* dir_r = nullptr;
  double* dir_z = nullptr;
  double* dir_phi = nullptr;

  // Scalars.
  double* energy = nullptr;
  double* weight = nullptr;
  double* time_remain = nullptr;
  double* birth_energy = nullptr;
  std::int8_t* sign = nullptr;  // +1 or -1, default +1

  // Identifiers.
  std::uint64_t* global_id = nullptr;
  std::uint32_t* rng_counter = nullptr;
  std::int32_t* cell_id = nullptr;
  std::uint16_t* group_id = nullptr;
  std::uint8_t* mode = nullptr;
  std::uint8_t* alive = nullptr;

  int capacity = 0;
  int n_alive = 0;
  int n_census = 0;

  PhotonPool() = default;
  ~PhotonPool();

  PhotonPool(const PhotonPool&) = delete;
  PhotonPool& operator=(const PhotonPool&) = delete;

  PhotonPool(PhotonPool&& other) noexcept;
  PhotonPool& operator=(PhotonPool&& other) noexcept;

  void allocate(int initial_capacity);
  void release();
  void clear();
  void reserve(int required_capacity, int max_pool_size);

  void swap(PhotonPool& other) noexcept;
};

}  // namespace tenryu::radiation
