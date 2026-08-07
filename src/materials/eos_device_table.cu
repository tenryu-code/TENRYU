#include "materials/eos_device_table.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "materials/eos_table.hpp"

namespace tenryu::materials {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline int checked_int(const std::size_t value, const char* message) {
  TENRYU_ASSERT(value <= static_cast<std::size_t>(std::numeric_limits<int>::max()), message);
  return static_cast<int>(value);
}

bool table_supports_rho_e_reclosure(const EOSTable& table) {
  if (table.empty() || table.n_T() < 2 || table.e_table.empty()) {
    return false;
  }
  const std::size_t n_rho = table.n_rho();
  const std::size_t n_T = table.n_T();
  for (std::size_t i = 0; i < n_rho; ++i) {
    double prev = table.e_table[table.flat_index(i, 0)];
    if (!std::isfinite(prev)) {
      return false;
    }
    bool increasing = false;
    for (std::size_t j = 1; j < n_T; ++j) {
      const double curr = table.e_table[table.flat_index(i, j)];
      const double scale = std::max(std::max(std::abs(prev), std::abs(curr)), 1.0);
      const double tol = 1.0e-12 * scale;
      if (!std::isfinite(curr) || curr + tol < prev) {
        return false;
      }
      increasing = increasing || (curr > prev + tol);
      prev = curr;
    }
    if (!increasing) {
      return false;
    }
  }
  return true;
}

}  // namespace

DeviceEOSTable::~DeviceEOSTable() {
  free_all();
}

DeviceEOSTable::DeviceEOSTable(DeviceEOSTable&& other) noexcept {
  *this = std::move(other);
}

DeviceEOSTable& DeviceEOSTable::operator=(DeviceEOSTable&& other) noexcept {
  if (this != &other) {
    free_all();

    d_log_rho_grid_ = other.d_log_rho_grid_;
    d_log_T_grid_ = other.d_log_T_grid_;
    d_P_table_ = other.d_P_table_;
    d_e_table_ = other.d_e_table_;
    d_cv_table_ = other.d_cv_table_;
    n_rho_ = other.n_rho_;
    n_T_ = other.n_T_;
    log_rho_min_ = other.log_rho_min_;
    log_rho_max_ = other.log_rho_max_;
    log_T_min_ = other.log_T_min_;
    log_T_max_ = other.log_T_max_;
    d_log_rho_inv_ = other.d_log_rho_inv_;
    d_log_T_inv_ = other.d_log_T_inv_;
    supports_rho_e_reclosure_ = other.supports_rho_e_reclosure_;

    other.d_log_rho_grid_ = nullptr;
    other.d_log_T_grid_ = nullptr;
    other.d_P_table_ = nullptr;
    other.d_e_table_ = nullptr;
    other.d_cv_table_ = nullptr;
    other.n_rho_ = 0;
    other.n_T_ = 0;
    other.log_rho_min_ = 0.0;
    other.log_rho_max_ = 0.0;
    other.log_T_min_ = 0.0;
    other.log_T_max_ = 0.0;
    other.d_log_rho_inv_ = 0.0;
    other.d_log_T_inv_ = 0.0;
    other.supports_rho_e_reclosure_ = 0u;
  }
  return *this;
}

void DeviceEOSTable::upload(const EOSTable& cpu_table) {
  free_all();
  if (cpu_table.empty()) {
    n_rho_ = 0;
    return;
  }

  const int n_rho = checked_int(cpu_table.n_rho(), "DeviceEOSTable::upload n_rho overflow");
  const int n_T = checked_int(cpu_table.n_T(), "DeviceEOSTable::upload n_T overflow");
  TENRYU_ASSERT(cpu_table.log_rho_grid.size() == static_cast<std::size_t>(n_rho),
                "DeviceEOSTable::upload log_rho_grid size mismatch");
  TENRYU_ASSERT(cpu_table.log_T_grid.size() == static_cast<std::size_t>(n_T),
                "DeviceEOSTable::upload log_T_grid size mismatch");
  TENRYU_ASSERT(cpu_table.P_table.size() == static_cast<std::size_t>(n_rho) *
                                          static_cast<std::size_t>(n_T),
                "DeviceEOSTable::upload P_table size mismatch");
  TENRYU_ASSERT(cpu_table.e_table.size() == static_cast<std::size_t>(n_rho) *
                                          static_cast<std::size_t>(n_T),
                "DeviceEOSTable::upload e_table size mismatch");
  TENRYU_ASSERT(cpu_table.cv_table.size() == static_cast<std::size_t>(n_rho) *
                                           static_cast<std::size_t>(n_T),
                "DeviceEOSTable::upload cv_table size mismatch");

  const std::size_t grid_rho_bytes = sizeof(double) * static_cast<std::size_t>(n_rho);
  const std::size_t grid_T_bytes = sizeof(double) * static_cast<std::size_t>(n_T);
  const std::size_t table_bytes =
      sizeof(double) * static_cast<std::size_t>(n_rho) * static_cast<std::size_t>(n_T);

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_log_rho_grid_), grid_rho_bytes),
             "DeviceEOSTable::upload cudaMalloc log_rho_grid failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_log_T_grid_), grid_T_bytes),
             "DeviceEOSTable::upload cudaMalloc log_T_grid failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P_table_), table_bytes),
             "DeviceEOSTable::upload cudaMalloc P_table failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_table_), table_bytes),
             "DeviceEOSTable::upload cudaMalloc e_table failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_cv_table_), table_bytes),
             "DeviceEOSTable::upload cudaMalloc cv_table failed");

  cuda_check(cudaMemcpy(d_log_rho_grid_,
                        cpu_table.log_rho_grid.data(),
                        grid_rho_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceEOSTable::upload cudaMemcpy log_rho_grid failed");
  cuda_check(cudaMemcpy(d_log_T_grid_,
                        cpu_table.log_T_grid.data(),
                        grid_T_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceEOSTable::upload cudaMemcpy log_T_grid failed");
  cuda_check(cudaMemcpy(d_P_table_,
                        cpu_table.P_table.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceEOSTable::upload cudaMemcpy P_table failed");
  cuda_check(cudaMemcpy(d_e_table_,
                        cpu_table.e_table.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceEOSTable::upload cudaMemcpy e_table failed");
  cuda_check(cudaMemcpy(d_cv_table_,
                        cpu_table.cv_table.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceEOSTable::upload cudaMemcpy cv_table failed");

  n_rho_ = n_rho;
  n_T_ = n_T;
  log_rho_min_ = cpu_table.log_rho_grid.front();
  log_rho_max_ = cpu_table.log_rho_grid.back();
  log_T_min_ = cpu_table.log_T_grid.front();
  log_T_max_ = cpu_table.log_T_grid.back();
  supports_rho_e_reclosure_ = table_supports_rho_e_reclosure(cpu_table) ? 1u : 0u;

  if (n_rho_ > 1) {
    const double d = cpu_table.log_rho_grid[1] - cpu_table.log_rho_grid[0];
    bool uniform = true;
    for (int i = 2; i < n_rho_; ++i) {
      if (std::fabs((cpu_table.log_rho_grid[static_cast<std::size_t>(i)] -
                     cpu_table.log_rho_grid[static_cast<std::size_t>(i - 1)]) -
                    d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_rho_inv_ = uniform ? 1.0 / d : 0.0;
  } else {
    d_log_rho_inv_ = 0.0;
  }

  if (n_T_ > 1) {
    const double d = cpu_table.log_T_grid[1] - cpu_table.log_T_grid[0];
    bool uniform = true;
    for (int i = 2; i < n_T_; ++i) {
      if (std::fabs((cpu_table.log_T_grid[static_cast<std::size_t>(i)] -
                     cpu_table.log_T_grid[static_cast<std::size_t>(i - 1)]) -
                    d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_T_inv_ = uniform ? 1.0 / d : 0.0;
  } else {
    d_log_T_inv_ = 0.0;
  }
}

DeviceEOSTableView DeviceEOSTable::view() const {
  DeviceEOSTableView v{};
  v.log_rho_grid = d_log_rho_grid_;
  v.log_T_grid = d_log_T_grid_;
  v.P_table = d_P_table_;
  v.e_table = d_e_table_;
  v.cv_table = d_cv_table_;
  v.n_rho = n_rho_;
  v.n_T = n_T_;
  v.log_rho_min = log_rho_min_;
  v.log_rho_max = log_rho_max_;
  v.log_T_min = log_T_min_;
  v.log_T_max = log_T_max_;
  v.d_log_rho_inv = d_log_rho_inv_;
  v.d_log_T_inv = d_log_T_inv_;
  v.supports_rho_e_reclosure = supports_rho_e_reclosure_;
  return v;
}

void DeviceEOSTable::free_all() {
  if (d_cv_table_ != nullptr) {
    cuda_check(cudaFree(d_cv_table_), "DeviceEOSTable::free_all cudaFree cv_table failed");
    d_cv_table_ = nullptr;
  }
  if (d_e_table_ != nullptr) {
    cuda_check(cudaFree(d_e_table_), "DeviceEOSTable::free_all cudaFree e_table failed");
    d_e_table_ = nullptr;
  }
  if (d_P_table_ != nullptr) {
    cuda_check(cudaFree(d_P_table_), "DeviceEOSTable::free_all cudaFree P_table failed");
    d_P_table_ = nullptr;
  }
  if (d_log_T_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_T_grid_), "DeviceEOSTable::free_all cudaFree log_T_grid failed");
    d_log_T_grid_ = nullptr;
  }
  if (d_log_rho_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_rho_grid_),
               "DeviceEOSTable::free_all cudaFree log_rho_grid failed");
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
  supports_rho_e_reclosure_ = 0u;
}

}  // namespace tenryu::materials
