#include "materials/cold_equilibrium_device.hpp"

#include <cstddef>
#include <utility>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::materials {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

}  // namespace

DeviceColdEquilibriumTable::~DeviceColdEquilibriumTable() {
  free_all();
}

DeviceColdEquilibriumTable::DeviceColdEquilibriumTable(
    DeviceColdEquilibriumTable&& other) noexcept {
  *this = std::move(other);
}

DeviceColdEquilibriumTable& DeviceColdEquilibriumTable::operator=(
    DeviceColdEquilibriumTable&& other) noexcept {
  if (this != &other) {
    free_all();
    view_ = other.view_;
    other.view_ = ColdEquilibriumView{};
  }
  return *this;
}

void DeviceColdEquilibriumTable::upload(const ColdEquilibriumTable& cpu_table) {
  free_all();
  if (cpu_table.empty()) {
    return;
  }

  const ColdEquilibriumView host_view = cpu_table.view();
  const std::size_t bytes = sizeof(double) * cpu_table.v_knots.size();
  double* d_v_knots = nullptr;
  double* d_PN_knots = nullptr;
  double* d_C0_knots = nullptr;
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_v_knots), bytes),
             "DeviceColdEquilibriumTable::upload cudaMalloc v_knots failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_PN_knots), bytes),
             "DeviceColdEquilibriumTable::upload cudaMalloc PN_knots failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_C0_knots), bytes),
             "DeviceColdEquilibriumTable::upload cudaMalloc C0_knots failed");
  cuda_check(cudaMemcpy(d_v_knots, cpu_table.v_knots.data(), bytes, cudaMemcpyHostToDevice),
             "DeviceColdEquilibriumTable::upload cudaMemcpy v_knots failed");
  cuda_check(cudaMemcpy(d_PN_knots, cpu_table.PN_knots.data(), bytes, cudaMemcpyHostToDevice),
             "DeviceColdEquilibriumTable::upload cudaMemcpy PN_knots failed");
  cuda_check(cudaMemcpy(d_C0_knots, cpu_table.C0_knots.data(), bytes, cudaMemcpyHostToDevice),
             "DeviceColdEquilibriumTable::upload cudaMemcpy C0_knots failed");
  view_ = host_view;
  view_.v_knots = d_v_knots;
  view_.PN_knots = d_PN_knots;
  view_.C0_knots = d_C0_knots;
}

ColdEquilibriumView DeviceColdEquilibriumTable::view() const {
  return view_;
}

void DeviceColdEquilibriumTable::free_all() {
  if (view_.C0_knots != nullptr) {
    cuda_check(cudaFree(const_cast<double*>(view_.C0_knots)),
               "DeviceColdEquilibriumTable::free_all cudaFree C0_knots failed");
  }
  if (view_.PN_knots != nullptr) {
    cuda_check(cudaFree(const_cast<double*>(view_.PN_knots)),
               "DeviceColdEquilibriumTable::free_all cudaFree PN_knots failed");
  }
  if (view_.v_knots != nullptr) {
    cuda_check(cudaFree(const_cast<double*>(view_.v_knots)),
               "DeviceColdEquilibriumTable::free_all cudaFree v_knots failed");
  }
  view_ = ColdEquilibriumView{};
}

}  // namespace tenryu::materials
