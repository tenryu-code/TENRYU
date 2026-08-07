#include "materials/helmholtz_jet_device.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::materials {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline int checked_int(const std::size_t value, const char* message) {
  TENRYU_ASSERT(value <= static_cast<std::size_t>(std::numeric_limits<int>::max()), message);
  return static_cast<int>(value);
}

}  // namespace

DeviceHelmholtzJet::~DeviceHelmholtzJet() {
  free_all();
}

DeviceHelmholtzJet::DeviceHelmholtzJet(DeviceHelmholtzJet&& other) noexcept {
  *this = std::move(other);
}

DeviceHelmholtzJet& DeviceHelmholtzJet::operator=(DeviceHelmholtzJet&& other) noexcept {
  if (this != &other) {
    free_all();
    d_log_rho_grid_ = other.d_log_rho_grid_;
    d_log_T_grid_ = other.d_log_T_grid_;
    d_cell_coeffs_ = other.d_cell_coeffs_;
    n_rho_ = other.n_rho_;
    n_T_ = other.n_T_;
    log_rho_min_ = other.log_rho_min_;
    log_rho_max_ = other.log_rho_max_;
    log_T_min_ = other.log_T_min_;
    log_T_max_ = other.log_T_max_;
    d_log_rho_inv_ = other.d_log_rho_inv_;
    d_log_T_inv_ = other.d_log_T_inv_;
    other.d_log_rho_grid_ = nullptr;
    other.d_log_T_grid_ = nullptr;
    other.d_cell_coeffs_ = nullptr;
    other.n_rho_ = 0;
    other.n_T_ = 0;
    other.log_rho_min_ = 0.0;
    other.log_rho_max_ = 0.0;
    other.log_T_min_ = 0.0;
    other.log_T_max_ = 0.0;
    other.d_log_rho_inv_ = 0.0;
    other.d_log_T_inv_ = 0.0;
  }
  return *this;
}

void DeviceHelmholtzJet::upload(const HelmholtzJetEOS& cpu_jet) {
  free_all();
  if (cpu_jet.empty()) {
    return;
  }
  const int n_rho = checked_int(cpu_jet.n_rho, "DeviceHelmholtzJet::upload n_rho overflow");
  const int n_T = checked_int(cpu_jet.n_T, "DeviceHelmholtzJet::upload n_T overflow");
  const std::size_t grid_rho_bytes = sizeof(double) * static_cast<std::size_t>(n_rho);
  const std::size_t grid_T_bytes = sizeof(double) * static_cast<std::size_t>(n_T);
  const std::size_t coeff_bytes = sizeof(double) * cpu_jet.cell_coeffs.size();

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_log_rho_grid_), grid_rho_bytes),
             "DeviceHelmholtzJet::upload cudaMalloc log_rho_grid failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_log_T_grid_), grid_T_bytes),
             "DeviceHelmholtzJet::upload cudaMalloc log_T_grid failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cell_coeffs_), coeff_bytes),
             "DeviceHelmholtzJet::upload cudaMalloc cell_coeffs failed");

  cuda_check(cudaMemcpy(d_log_rho_grid_,
                        cpu_jet.log_rho_grid.data(),
                        grid_rho_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzJet::upload cudaMemcpy log_rho_grid failed");
  cuda_check(cudaMemcpy(d_log_T_grid_,
                        cpu_jet.log_T_grid.data(),
                        grid_T_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzJet::upload cudaMemcpy log_T_grid failed");
  cuda_check(cudaMemcpy(d_cell_coeffs_,
                        cpu_jet.cell_coeffs.data(),
                        coeff_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzJet::upload cudaMemcpy cell_coeffs failed");

  n_rho_ = n_rho;
  n_T_ = n_T;
  log_rho_min_ = cpu_jet.log_rho_grid.front();
  log_rho_max_ = cpu_jet.log_rho_grid.back();
  log_T_min_ = cpu_jet.log_T_grid.front();
  log_T_max_ = cpu_jet.log_T_grid.back();

  if (n_rho_ > 1) {
    const double d = cpu_jet.log_rho_grid[1] - cpu_jet.log_rho_grid[0];
    bool uniform = true;
    for (int i = 2; i < n_rho_; ++i) {
      if (std::fabs((cpu_jet.log_rho_grid[static_cast<std::size_t>(i)] -
                     cpu_jet.log_rho_grid[static_cast<std::size_t>(i - 1)]) -
                    d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_rho_inv_ = uniform ? 1.0 / d : 0.0;
  }
  if (n_T_ > 1) {
    const double d = cpu_jet.log_T_grid[1] - cpu_jet.log_T_grid[0];
    bool uniform = true;
    for (int i = 2; i < n_T_; ++i) {
      if (std::fabs((cpu_jet.log_T_grid[static_cast<std::size_t>(i)] -
                     cpu_jet.log_T_grid[static_cast<std::size_t>(i - 1)]) -
                    d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_T_inv_ = uniform ? 1.0 / d : 0.0;
  }
}

HelmholtzJetDeviceView DeviceHelmholtzJet::view() const {
  HelmholtzJetDeviceView v{};
  v.log_rho_grid = d_log_rho_grid_;
  v.log_T_grid = d_log_T_grid_;
  v.cell_coeffs = d_cell_coeffs_;
  v.n_rho = n_rho_;
  v.n_T = n_T_;
  v.log_rho_min = log_rho_min_;
  v.log_rho_max = log_rho_max_;
  v.log_T_min = log_T_min_;
  v.log_T_max = log_T_max_;
  v.d_log_rho_inv = d_log_rho_inv_;
  v.d_log_T_inv = d_log_T_inv_;
  return v;
}

void DeviceHelmholtzJet::free_all() {
  if (d_cell_coeffs_ != nullptr) {
    cuda_check(cudaFree(d_cell_coeffs_), "DeviceHelmholtzJet::free_all cudaFree cell_coeffs");
    d_cell_coeffs_ = nullptr;
  }
  if (d_log_T_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_T_grid_), "DeviceHelmholtzJet::free_all cudaFree log_T_grid");
    d_log_T_grid_ = nullptr;
  }
  if (d_log_rho_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_rho_grid_), "DeviceHelmholtzJet::free_all cudaFree log_rho_grid");
    d_log_rho_grid_ = nullptr;
  }
  n_rho_ = 0;
  n_T_ = 0;
  log_rho_min_ = 0.0;
  log_rho_max_ = 0.0;
  log_T_min_ = 0.0;
  log_T_max_ = 0.0;
  d_log_rho_inv_ = 0.0;
  d_log_T_inv_ = 0.0;
}

}  // namespace tenryu::materials
