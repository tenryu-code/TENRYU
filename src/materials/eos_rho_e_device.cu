#include "materials/eos_rho_e_device.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "materials/eos_rho_e_table.hpp"

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

DeviceEOSRhoETable::~DeviceEOSRhoETable() {
  free_all();
}

DeviceEOSRhoETable::DeviceEOSRhoETable(DeviceEOSRhoETable&& other) noexcept {
  *this = std::move(other);
}

DeviceEOSRhoETable& DeviceEOSRhoETable::operator=(DeviceEOSRhoETable&& other) noexcept {
  if (this != &other) {
    free_all();

    d_log_rho_grid_ = other.d_log_rho_grid_;
    d_log_e_grid_ = other.d_log_e_grid_;
    d_P_table_ = other.d_P_table_;
    d_T_table_ = other.d_T_table_;
    d_P_xx_ = other.d_P_xx_;
    d_P_yy_ = other.d_P_yy_;
    d_P_xxyy_ = other.d_P_xxyy_;
    d_T_xx_ = other.d_T_xx_;
    d_T_yy_ = other.d_T_yy_;
    d_T_xxyy_ = other.d_T_xxyy_;
    use_log_grid_ = other.use_log_grid_;
    n_rho_ = other.n_rho_;
    n_e_ = other.n_e_;
    log_rho_min_ = other.log_rho_min_;
    log_rho_max_ = other.log_rho_max_;
    log_e_min_ = other.log_e_min_;
    log_e_max_ = other.log_e_max_;
    d_log_rho_inv_ = other.d_log_rho_inv_;
    d_log_e_inv_ = other.d_log_e_inv_;

    other.d_log_rho_grid_ = nullptr;
    other.d_log_e_grid_ = nullptr;
    other.d_P_table_ = nullptr;
    other.d_T_table_ = nullptr;
    other.d_P_xx_ = nullptr;
    other.d_P_yy_ = nullptr;
    other.d_P_xxyy_ = nullptr;
    other.d_T_xx_ = nullptr;
    other.d_T_yy_ = nullptr;
    other.d_T_xxyy_ = nullptr;
    other.use_log_grid_ = true;
    other.n_rho_ = 0;
    other.n_e_ = 0;
    other.log_rho_min_ = 0.0;
    other.log_rho_max_ = 0.0;
    other.log_e_min_ = 0.0;
    other.log_e_max_ = 0.0;
    other.d_log_rho_inv_ = 0.0;
    other.d_log_e_inv_ = 0.0;
  }
  return *this;
}

void DeviceEOSRhoETable::upload(const EOSRhoETable& cpu_table) {
  free_all();
  if (cpu_table.empty()) {
    n_rho_ = 0;
    return;
  }

  const int n_rho = checked_int(cpu_table.n_rho(), "DeviceEOSRhoETable::upload n_rho overflow");
  const int n_e = checked_int(cpu_table.n_e(), "DeviceEOSRhoETable::upload n_e overflow");
  TENRYU_ASSERT(cpu_table.log_rho_grid().size() == static_cast<std::size_t>(n_rho),
                "DeviceEOSRhoETable::upload log_rho_grid size mismatch");
  TENRYU_ASSERT(cpu_table.log_e_grid().size() == static_cast<std::size_t>(n_e),
                "DeviceEOSRhoETable::upload log_e_grid size mismatch");
  TENRYU_ASSERT(cpu_table.P_table().size() == static_cast<std::size_t>(n_rho) *
                                             static_cast<std::size_t>(n_e),
                "DeviceEOSRhoETable::upload P_table size mismatch");
  TENRYU_ASSERT(cpu_table.T_table().size() == static_cast<std::size_t>(n_rho) *
                                             static_cast<std::size_t>(n_e),
                "DeviceEOSRhoETable::upload T_table size mismatch");
  TENRYU_ASSERT(cpu_table.P_xx().size() == static_cast<std::size_t>(n_rho) *
                                           static_cast<std::size_t>(n_e) &&
                    cpu_table.P_yy().size() == static_cast<std::size_t>(n_rho) *
                                                   static_cast<std::size_t>(n_e) &&
                    cpu_table.P_xxyy().size() == static_cast<std::size_t>(n_rho) *
                                                     static_cast<std::size_t>(n_e) &&
                    cpu_table.T_xx().size() == static_cast<std::size_t>(n_rho) *
                                                   static_cast<std::size_t>(n_e) &&
                    cpu_table.T_yy().size() == static_cast<std::size_t>(n_rho) *
                                                   static_cast<std::size_t>(n_e) &&
                    cpu_table.T_xxyy().size() == static_cast<std::size_t>(n_rho) *
                                                     static_cast<std::size_t>(n_e),
                "DeviceEOSRhoETable::upload spline field size mismatch");

  const std::size_t grid_rho_bytes = sizeof(double) * static_cast<std::size_t>(n_rho);
  const std::size_t grid_e_bytes = sizeof(double) * static_cast<std::size_t>(n_e);
  const std::size_t table_bytes =
      sizeof(double) * static_cast<std::size_t>(n_rho) * static_cast<std::size_t>(n_e);

  const auto alloc_and_copy = [&](double** dst,
                                  const double* src,
                                  const std::size_t bytes,
                                  const char* malloc_message,
                                  const char* memcpy_message) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), bytes), malloc_message);
    cuda_check(cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice), memcpy_message);
  };

  alloc_and_copy(&d_log_rho_grid_,
                 cpu_table.log_rho_grid().data(),
                 grid_rho_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc log_rho_grid failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy log_rho_grid failed");
  alloc_and_copy(&d_log_e_grid_,
                 cpu_table.log_e_grid().data(),
                 grid_e_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc log_e_grid failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy log_e_grid failed");
  alloc_and_copy(&d_P_table_,
                 cpu_table.P_table().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc P_table failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy P_table failed");
  alloc_and_copy(&d_T_table_,
                 cpu_table.T_table().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc T_table failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy T_table failed");
  alloc_and_copy(&d_P_xx_,
                 cpu_table.P_xx().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc P_xx failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy P_xx failed");
  alloc_and_copy(&d_P_yy_,
                 cpu_table.P_yy().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc P_yy failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy P_yy failed");
  alloc_and_copy(&d_P_xxyy_,
                 cpu_table.P_xxyy().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc P_xxyy failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy P_xxyy failed");
  alloc_and_copy(&d_T_xx_,
                 cpu_table.T_xx().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc T_xx failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy T_xx failed");
  alloc_and_copy(&d_T_yy_,
                 cpu_table.T_yy().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc T_yy failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy T_yy failed");
  alloc_and_copy(&d_T_xxyy_,
                 cpu_table.T_xxyy().data(),
                 table_bytes,
                 "DeviceEOSRhoETable::upload cudaMalloc T_xxyy failed",
                 "DeviceEOSRhoETable::upload cudaMemcpy T_xxyy failed");

  n_rho_ = n_rho;
  n_e_ = n_e;
  use_log_grid_ = cpu_table.use_log_grid();
  log_rho_min_ = cpu_table.log_rho_grid().front();
  log_rho_max_ = cpu_table.log_rho_grid().back();
  log_e_min_ = cpu_table.log_e_grid().front();
  log_e_max_ = cpu_table.log_e_grid().back();

  if (n_rho_ > 1) {
    const double d = cpu_table.log_rho_grid()[1] - cpu_table.log_rho_grid()[0];
    bool uniform = true;
    for (int i = 2; i < n_rho_; ++i) {
      if (std::fabs((cpu_table.log_rho_grid()[static_cast<std::size_t>(i)] -
                     cpu_table.log_rho_grid()[static_cast<std::size_t>(i - 1)]) -
                    d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_rho_inv_ = uniform ? 1.0 / d : 0.0;
  } else {
    d_log_rho_inv_ = 0.0;
  }

  if (n_e_ > 1) {
    const double d = cpu_table.log_e_grid()[1] - cpu_table.log_e_grid()[0];
    bool uniform = true;
    for (int i = 2; i < n_e_; ++i) {
      if (std::fabs((cpu_table.log_e_grid()[static_cast<std::size_t>(i)] -
                     cpu_table.log_e_grid()[static_cast<std::size_t>(i - 1)]) -
                    d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_e_inv_ = uniform ? 1.0 / d : 0.0;
  } else {
    d_log_e_inv_ = 0.0;
  }
}

EOSRhoEDeviceView DeviceEOSRhoETable::view() const {
  EOSRhoEDeviceView v{};
  v.log_rho_grid = d_log_rho_grid_;
  v.log_e_grid = d_log_e_grid_;
  v.P_table = d_P_table_;
  v.T_table = d_T_table_;
  v.P_xx = d_P_xx_;
  v.P_yy = d_P_yy_;
  v.P_xxyy = d_P_xxyy_;
  v.T_xx = d_T_xx_;
  v.T_yy = d_T_yy_;
  v.T_xxyy = d_T_xxyy_;
  v.use_log_grid = use_log_grid_;
  v.n_rho = n_rho_;
  v.n_e = n_e_;
  v.log_rho_min = log_rho_min_;
  v.log_rho_max = log_rho_max_;
  v.log_e_min = log_e_min_;
  v.log_e_max = log_e_max_;
  v.d_log_rho_inv = d_log_rho_inv_;
  v.d_log_e_inv = d_log_e_inv_;
  return v;
}

void DeviceEOSRhoETable::free_all() {
  if (d_T_xxyy_ != nullptr) {
    cuda_check(cudaFree(d_T_xxyy_),
               "DeviceEOSRhoETable::free_all cudaFree T_xxyy failed");
    d_T_xxyy_ = nullptr;
  }
  if (d_T_yy_ != nullptr) {
    cuda_check(cudaFree(d_T_yy_), "DeviceEOSRhoETable::free_all cudaFree T_yy failed");
    d_T_yy_ = nullptr;
  }
  if (d_T_xx_ != nullptr) {
    cuda_check(cudaFree(d_T_xx_), "DeviceEOSRhoETable::free_all cudaFree T_xx failed");
    d_T_xx_ = nullptr;
  }
  if (d_P_xxyy_ != nullptr) {
    cuda_check(cudaFree(d_P_xxyy_),
               "DeviceEOSRhoETable::free_all cudaFree P_xxyy failed");
    d_P_xxyy_ = nullptr;
  }
  if (d_P_yy_ != nullptr) {
    cuda_check(cudaFree(d_P_yy_), "DeviceEOSRhoETable::free_all cudaFree P_yy failed");
    d_P_yy_ = nullptr;
  }
  if (d_P_xx_ != nullptr) {
    cuda_check(cudaFree(d_P_xx_), "DeviceEOSRhoETable::free_all cudaFree P_xx failed");
    d_P_xx_ = nullptr;
  }
  if (d_T_table_ != nullptr) {
    cuda_check(cudaFree(d_T_table_), "DeviceEOSRhoETable::free_all cudaFree T_table failed");
    d_T_table_ = nullptr;
  }
  if (d_P_table_ != nullptr) {
    cuda_check(cudaFree(d_P_table_), "DeviceEOSRhoETable::free_all cudaFree P_table failed");
    d_P_table_ = nullptr;
  }
  if (d_log_e_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_e_grid_),
               "DeviceEOSRhoETable::free_all cudaFree log_e_grid failed");
    d_log_e_grid_ = nullptr;
  }
  if (d_log_rho_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_rho_grid_),
               "DeviceEOSRhoETable::free_all cudaFree log_rho_grid failed");
    d_log_rho_grid_ = nullptr;
  }

  n_rho_ = 0;
  n_e_ = 0;
  use_log_grid_ = true;
  log_rho_min_ = 0.0;
  log_rho_max_ = 0.0;
  log_e_min_ = 0.0;
  log_e_max_ = 0.0;
  d_log_rho_inv_ = 0.0;
  d_log_e_inv_ = 0.0;
}

}  // namespace tenryu::materials
