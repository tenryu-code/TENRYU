#include "materials/helmholtz_spline_device.hpp"

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

DeviceHelmholtzSpline::~DeviceHelmholtzSpline() {
  free_all();
}

DeviceHelmholtzSpline::DeviceHelmholtzSpline(DeviceHelmholtzSpline&& other) noexcept {
  *this = std::move(other);
}

DeviceHelmholtzSpline& DeviceHelmholtzSpline::operator=(DeviceHelmholtzSpline&& other) noexcept {
  if (this != &other) {
    free_all();

    d_log_rho_grid_ = other.d_log_rho_grid_;
    d_log_T_grid_ = other.d_log_T_grid_;
    d_P_nodes_ = other.d_P_nodes_;
    d_P_dx_nodes_ = other.d_P_dx_nodes_;
    d_P_dy_nodes_ = other.d_P_dy_nodes_;
    d_P_dxdy_nodes_ = other.d_P_dxdy_nodes_;
    d_e_nodes_ = other.d_e_nodes_;
    d_e_dx_nodes_ = other.d_e_dx_nodes_;
    d_e_dy_nodes_ = other.d_e_dy_nodes_;
    d_e_dxdy_nodes_ = other.d_e_dxdy_nodes_;
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
    other.d_P_nodes_ = nullptr;
    other.d_P_dx_nodes_ = nullptr;
    other.d_P_dy_nodes_ = nullptr;
    other.d_P_dxdy_nodes_ = nullptr;
    other.d_e_nodes_ = nullptr;
    other.d_e_dx_nodes_ = nullptr;
    other.d_e_dy_nodes_ = nullptr;
    other.d_e_dxdy_nodes_ = nullptr;
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

void DeviceHelmholtzSpline::upload(const HelmholtzSpline1T& cpu_spline) {
  free_all();
  if (cpu_spline.empty()) {
    n_rho_ = 0;
    return;
  }

  const int n_rho = checked_int(cpu_spline.n_rho(),
                                "DeviceHelmholtzSpline::upload n_rho overflow");
  const int n_T = checked_int(cpu_spline.n_T(),
                              "DeviceHelmholtzSpline::upload n_T overflow");
  const std::size_t table_size =
      static_cast<std::size_t>(n_rho) * static_cast<std::size_t>(n_T);
  TENRYU_ASSERT(cpu_spline.P_nodes.size() == table_size &&
                    cpu_spline.P_dx_nodes.size() == table_size &&
                    cpu_spline.P_dy_nodes.size() == table_size &&
                    cpu_spline.P_dxdy_nodes.size() == table_size &&
                    cpu_spline.e_nodes.size() == table_size &&
                    cpu_spline.e_dx_nodes.size() == table_size &&
                    cpu_spline.e_dy_nodes.size() == table_size &&
                    cpu_spline.e_dxdy_nodes.size() == table_size,
                "DeviceHelmholtzSpline::upload table size mismatch");

  const std::size_t grid_rho_bytes = sizeof(double) * static_cast<std::size_t>(n_rho);
  const std::size_t grid_T_bytes = sizeof(double) * static_cast<std::size_t>(n_T);
  const std::size_t table_bytes = sizeof(double) * table_size;

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_log_rho_grid_), grid_rho_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc log_rho_grid failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_log_T_grid_), grid_T_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc log_T_grid failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc P_nodes failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P_dx_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc P_dx_nodes failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P_dy_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc P_dy_nodes failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_P_dxdy_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc P_dxdy_nodes failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc e_nodes failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_dx_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc e_dx_nodes failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_dy_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc e_dy_nodes failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_e_dxdy_nodes_), table_bytes),
             "DeviceHelmholtzSpline::upload cudaMalloc e_dxdy_nodes failed");

  cuda_check(cudaMemcpy(d_log_rho_grid_,
                        cpu_spline.log_rho_grid.data(),
                        grid_rho_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy log_rho_grid failed");
  cuda_check(cudaMemcpy(d_log_T_grid_,
                        cpu_spline.log_T_grid.data(),
                        grid_T_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy log_T_grid failed");

  cuda_check(cudaMemcpy(d_P_nodes_,
                        cpu_spline.P_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy P_nodes failed");
  cuda_check(cudaMemcpy(d_P_dx_nodes_,
                        cpu_spline.P_dx_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy P_dx_nodes failed");
  cuda_check(cudaMemcpy(d_P_dy_nodes_,
                        cpu_spline.P_dy_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy P_dy_nodes failed");
  cuda_check(cudaMemcpy(d_P_dxdy_nodes_,
                        cpu_spline.P_dxdy_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy P_dxdy_nodes failed");

  cuda_check(cudaMemcpy(d_e_nodes_,
                        cpu_spline.e_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy e_nodes failed");
  cuda_check(cudaMemcpy(d_e_dx_nodes_,
                        cpu_spline.e_dx_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy e_dx_nodes failed");
  cuda_check(cudaMemcpy(d_e_dy_nodes_,
                        cpu_spline.e_dy_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy e_dy_nodes failed");
  cuda_check(cudaMemcpy(d_e_dxdy_nodes_,
                        cpu_spline.e_dxdy_nodes.data(),
                        table_bytes,
                        cudaMemcpyHostToDevice),
             "DeviceHelmholtzSpline::upload cudaMemcpy e_dxdy_nodes failed");

  n_rho_ = n_rho;
  n_T_ = n_T;
  log_rho_min_ = cpu_spline.log_rho_grid.front();
  log_rho_max_ = cpu_spline.log_rho_grid.back();
  log_T_min_ = cpu_spline.log_T_grid.front();
  log_T_max_ = cpu_spline.log_T_grid.back();

  if (n_rho_ > 1) {
    const double d = cpu_spline.log_rho_grid[1] - cpu_spline.log_rho_grid[0];
    bool uniform = true;
    for (int i = 2; i < n_rho_; ++i) {
      if (std::fabs((cpu_spline.log_rho_grid[static_cast<std::size_t>(i)] -
                     cpu_spline.log_rho_grid[static_cast<std::size_t>(i - 1)]) -
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
    const double d = cpu_spline.log_T_grid[1] - cpu_spline.log_T_grid[0];
    bool uniform = true;
    for (int i = 2; i < n_T_; ++i) {
      if (std::fabs((cpu_spline.log_T_grid[static_cast<std::size_t>(i)] -
                     cpu_spline.log_T_grid[static_cast<std::size_t>(i - 1)]) -
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

HelmholtzSplineDeviceView DeviceHelmholtzSpline::view() const {
  HelmholtzSplineDeviceView v{};
  v.log_rho_grid = d_log_rho_grid_;
  v.log_T_grid = d_log_T_grid_;
  v.P_nodes = d_P_nodes_;
  v.P_dx_nodes = d_P_dx_nodes_;
  v.P_dy_nodes = d_P_dy_nodes_;
  v.P_dxdy_nodes = d_P_dxdy_nodes_;
  v.e_nodes = d_e_nodes_;
  v.e_dx_nodes = d_e_dx_nodes_;
  v.e_dy_nodes = d_e_dy_nodes_;
  v.e_dxdy_nodes = d_e_dxdy_nodes_;
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

void DeviceHelmholtzSpline::free_all() {
  if (d_e_dxdy_nodes_ != nullptr) {
    cuda_check(cudaFree(d_e_dxdy_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree e_dxdy_nodes failed");
    d_e_dxdy_nodes_ = nullptr;
  }
  if (d_e_dy_nodes_ != nullptr) {
    cuda_check(cudaFree(d_e_dy_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree e_dy_nodes failed");
    d_e_dy_nodes_ = nullptr;
  }
  if (d_e_dx_nodes_ != nullptr) {
    cuda_check(cudaFree(d_e_dx_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree e_dx_nodes failed");
    d_e_dx_nodes_ = nullptr;
  }
  if (d_e_nodes_ != nullptr) {
    cuda_check(cudaFree(d_e_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree e_nodes failed");
    d_e_nodes_ = nullptr;
  }
  if (d_P_dxdy_nodes_ != nullptr) {
    cuda_check(cudaFree(d_P_dxdy_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree P_dxdy_nodes failed");
    d_P_dxdy_nodes_ = nullptr;
  }
  if (d_P_dy_nodes_ != nullptr) {
    cuda_check(cudaFree(d_P_dy_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree P_dy_nodes failed");
    d_P_dy_nodes_ = nullptr;
  }
  if (d_P_dx_nodes_ != nullptr) {
    cuda_check(cudaFree(d_P_dx_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree P_dx_nodes failed");
    d_P_dx_nodes_ = nullptr;
  }
  if (d_P_nodes_ != nullptr) {
    cuda_check(cudaFree(d_P_nodes_),
               "DeviceHelmholtzSpline::free_all cudaFree P_nodes failed");
    d_P_nodes_ = nullptr;
  }
  if (d_log_T_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_T_grid_),
               "DeviceHelmholtzSpline::free_all cudaFree log_T_grid failed");
    d_log_T_grid_ = nullptr;
  }
  if (d_log_rho_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_rho_grid_),
               "DeviceHelmholtzSpline::free_all cudaFree log_rho_grid failed");
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
