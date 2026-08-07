#include "radiation/particle_pool.cuh"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void alloc_field(T** ptr, const int capacity, const char* message) {
  cuda_check(cudaMalloc(reinterpret_cast<void**>(ptr),
                        sizeof(T) * static_cast<std::size_t>(capacity)),
             message);
}

template <typename T>
void free_field(T** ptr, const char* message) {
  if (*ptr != nullptr) {
    cuda_check(cudaFree(*ptr), message);
    *ptr = nullptr;
  }
}

}  // namespace

PhotonPool::~PhotonPool() {
  release();
}

PhotonPool::PhotonPool(PhotonPool&& other) noexcept {
  *this = std::move(other);
}

PhotonPool& PhotonPool::operator=(PhotonPool&& other) noexcept {
  if (this != &other) {
    release();

    pos_r = other.pos_r;
    pos_z = other.pos_z;
    dir_r = other.dir_r;
    dir_z = other.dir_z;
    dir_phi = other.dir_phi;
    energy = other.energy;
    weight = other.weight;
    time_remain = other.time_remain;
    birth_energy = other.birth_energy;
    sign = other.sign;
    global_id = other.global_id;
    rng_counter = other.rng_counter;
    cell_id = other.cell_id;
    group_id = other.group_id;
    mode = other.mode;
    alive = other.alive;
    capacity = other.capacity;
    n_alive = other.n_alive;
    n_census = other.n_census;

    other.pos_r = nullptr;
    other.pos_z = nullptr;
    other.dir_r = nullptr;
    other.dir_z = nullptr;
    other.dir_phi = nullptr;
    other.energy = nullptr;
    other.weight = nullptr;
    other.time_remain = nullptr;
    other.birth_energy = nullptr;
    other.sign = nullptr;
    other.global_id = nullptr;
    other.rng_counter = nullptr;
    other.cell_id = nullptr;
    other.group_id = nullptr;
    other.mode = nullptr;
    other.alive = nullptr;
    other.capacity = 0;
    other.n_alive = 0;
    other.n_census = 0;
  }
  return *this;
}

void PhotonPool::allocate(const int initial_capacity) {
  TENRYU_ASSERT(initial_capacity > 0, "PhotonPool::allocate requires capacity > 0");
  release();

  alloc_field(&pos_r, initial_capacity, "PhotonPool cudaMalloc pos_r failed");
  alloc_field(&pos_z, initial_capacity, "PhotonPool cudaMalloc pos_z failed");
  alloc_field(&dir_r, initial_capacity, "PhotonPool cudaMalloc dir_r failed");
  alloc_field(&dir_z, initial_capacity, "PhotonPool cudaMalloc dir_z failed");
  alloc_field(&dir_phi, initial_capacity, "PhotonPool cudaMalloc dir_phi failed");
  alloc_field(&energy, initial_capacity, "PhotonPool cudaMalloc energy failed");
  alloc_field(&weight, initial_capacity, "PhotonPool cudaMalloc weight failed");
  alloc_field(&time_remain, initial_capacity,
              "PhotonPool cudaMalloc time_remain failed");
  alloc_field(&birth_energy, initial_capacity,
              "PhotonPool cudaMalloc birth_energy failed");
  alloc_field(&sign, initial_capacity, "PhotonPool cudaMalloc sign failed");
  cuda_check(cudaMemset(sign, 1, sizeof(std::int8_t) * static_cast<std::size_t>(initial_capacity)),
             "PhotonPool cudaMemset sign failed");
  alloc_field(&global_id, initial_capacity, "PhotonPool cudaMalloc global_id failed");
  alloc_field(&rng_counter, initial_capacity,
              "PhotonPool cudaMalloc rng_counter failed");
  alloc_field(&cell_id, initial_capacity, "PhotonPool cudaMalloc cell_id failed");
  alloc_field(&group_id, initial_capacity, "PhotonPool cudaMalloc group_id failed");
  alloc_field(&mode, initial_capacity, "PhotonPool cudaMalloc mode failed");
  alloc_field(&alive, initial_capacity, "PhotonPool cudaMalloc alive failed");

  capacity = initial_capacity;
  n_alive = 0;
  n_census = 0;
}

void PhotonPool::release() {
  free_field(&alive, "PhotonPool cudaFree alive failed");
  free_field(&mode, "PhotonPool cudaFree mode failed");
  free_field(&group_id, "PhotonPool cudaFree group_id failed");
  free_field(&cell_id, "PhotonPool cudaFree cell_id failed");
  free_field(&rng_counter, "PhotonPool cudaFree rng_counter failed");
  free_field(&global_id, "PhotonPool cudaFree global_id failed");
  free_field(&sign, "PhotonPool cudaFree sign failed");
  free_field(&birth_energy, "PhotonPool cudaFree birth_energy failed");
  free_field(&time_remain, "PhotonPool cudaFree time_remain failed");
  free_field(&weight, "PhotonPool cudaFree weight failed");
  free_field(&energy, "PhotonPool cudaFree energy failed");
  free_field(&dir_phi, "PhotonPool cudaFree dir_phi failed");
  free_field(&dir_z, "PhotonPool cudaFree dir_z failed");
  free_field(&dir_r, "PhotonPool cudaFree dir_r failed");
  free_field(&pos_z, "PhotonPool cudaFree pos_z failed");
  free_field(&pos_r, "PhotonPool cudaFree pos_r failed");
  capacity = 0;
  n_alive = 0;
  n_census = 0;
}

void PhotonPool::clear() {
  n_alive = 0;
  n_census = 0;
}

void PhotonPool::reserve(const int required_capacity, const int max_pool_size) {
  TENRYU_ASSERT(required_capacity >= 0,
                "PhotonPool::reserve requires required_capacity >= 0");
  TENRYU_ASSERT(max_pool_size > 0,
                "PhotonPool::reserve requires max_pool_size > 0");

  if (required_capacity <= capacity) {
    return;
  }

  int new_capacity = std::max(1, capacity);
  while (new_capacity < required_capacity) {
    new_capacity = std::min(max_pool_size, new_capacity * 2);
    if (new_capacity == max_pool_size) {
      break;
    }
  }

  TENRYU_ASSERT(new_capacity >= required_capacity,
                "PhotonPool reserve exceeded max_pool_size");

  PhotonPool tmp;
  tmp.allocate(new_capacity);

  const int n_copy = n_alive;
  if (n_copy > 0) {
    const std::size_t bytes_d = sizeof(double) * static_cast<std::size_t>(n_copy);
    const std::size_t bytes_u64 = sizeof(std::uint64_t) * static_cast<std::size_t>(n_copy);
    const std::size_t bytes_u32 = sizeof(std::uint32_t) * static_cast<std::size_t>(n_copy);
    const std::size_t bytes_i32 = sizeof(std::int32_t) * static_cast<std::size_t>(n_copy);
    const std::size_t bytes_i8 = sizeof(std::int8_t) * static_cast<std::size_t>(n_copy);
    const std::size_t bytes_u16 = sizeof(std::uint16_t) * static_cast<std::size_t>(n_copy);
    const std::size_t bytes_u8 = sizeof(std::uint8_t) * static_cast<std::size_t>(n_copy);

    cuda_check(cudaMemcpy(tmp.pos_r, pos_r, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy pos_r failed");
    cuda_check(cudaMemcpy(tmp.pos_z, pos_z, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy pos_z failed");
    cuda_check(cudaMemcpy(tmp.dir_r, dir_r, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy dir_r failed");
    cuda_check(cudaMemcpy(tmp.dir_z, dir_z, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy dir_z failed");
    cuda_check(cudaMemcpy(tmp.dir_phi, dir_phi, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy dir_phi failed");
    cuda_check(cudaMemcpy(tmp.energy, energy, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy energy failed");
    cuda_check(cudaMemcpy(tmp.weight, weight, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy weight failed");
    cuda_check(cudaMemcpy(tmp.time_remain, time_remain, bytes_d, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy time_remain failed");
    cuda_check(cudaMemcpy(tmp.birth_energy,
                          birth_energy,
                          bytes_d,
                          cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy birth_energy failed");
    cuda_check(cudaMemcpy(tmp.sign, sign, bytes_i8, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy sign failed");
    cuda_check(cudaMemcpy(tmp.global_id,
                          global_id,
                          bytes_u64,
                          cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy global_id failed");
    cuda_check(cudaMemcpy(tmp.rng_counter,
                          rng_counter,
                          bytes_u32,
                          cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy rng_counter failed");
    cuda_check(cudaMemcpy(tmp.cell_id, cell_id, bytes_i32, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy cell_id failed");
    cuda_check(cudaMemcpy(tmp.group_id,
                          group_id,
                          bytes_u16,
                          cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy group_id failed");
    cuda_check(cudaMemcpy(tmp.mode, mode, bytes_u8, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy mode failed");
    cuda_check(cudaMemcpy(tmp.alive, alive, bytes_u8, cudaMemcpyDeviceToDevice),
               "PhotonPool reserve copy alive failed");
  }

  tmp.n_alive = n_alive;
  tmp.n_census = n_census;
  swap(tmp);
}

void PhotonPool::swap(PhotonPool& other) noexcept {
  std::swap(pos_r, other.pos_r);
  std::swap(pos_z, other.pos_z);
  std::swap(dir_r, other.dir_r);
  std::swap(dir_z, other.dir_z);
  std::swap(dir_phi, other.dir_phi);
  std::swap(energy, other.energy);
  std::swap(weight, other.weight);
  std::swap(time_remain, other.time_remain);
  std::swap(birth_energy, other.birth_energy);
  std::swap(sign, other.sign);
  std::swap(global_id, other.global_id);
  std::swap(rng_counter, other.rng_counter);
  std::swap(cell_id, other.cell_id);
  std::swap(group_id, other.group_id);
  std::swap(mode, other.mode);
  std::swap(alive, other.alive);
  std::swap(capacity, other.capacity);
  std::swap(n_alive, other.n_alive);
  std::swap(n_census, other.n_census);
}

}  // namespace tenryu::radiation
