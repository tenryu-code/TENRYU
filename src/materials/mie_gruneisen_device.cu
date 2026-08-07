#include "materials/mie_gruneisen_device.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "materials/mie_gruneisen.hpp"

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

DeviceMieGruneisen::~DeviceMieGruneisen() {
  free_all();
}

DeviceMieGruneisen::DeviceMieGruneisen(DeviceMieGruneisen&& other) noexcept {
  *this = std::move(other);
}

DeviceMieGruneisen& DeviceMieGruneisen::operator=(DeviceMieGruneisen&& other) noexcept {
  if (this != &other) {
    free_all();
    d_log_rho_grid_ = other.d_log_rho_grid_;
    d_gamma_ion_ = other.d_gamma_ion_;
    d_gamma_ele_ = other.d_gamma_ele_;
    d_p_ref_ion_ = other.d_p_ref_ion_;
    d_p_ref_ele_ = other.d_p_ref_ele_;
    d_e_ref_ion_ = other.d_e_ref_ion_;
    d_e_ref_ele_ = other.d_e_ref_ele_;
    d_dgamma_ion_dlogrho_ = other.d_dgamma_ion_dlogrho_;
    d_dgamma_ele_dlogrho_ = other.d_dgamma_ele_dlogrho_;
    d_dp_ref_ion_dlogrho_ = other.d_dp_ref_ion_dlogrho_;
    d_dp_ref_ele_dlogrho_ = other.d_dp_ref_ele_dlogrho_;
    d_de_ref_ion_dlogrho_ = other.d_de_ref_ion_dlogrho_;
    d_de_ref_ele_dlogrho_ = other.d_de_ref_ele_dlogrho_;
    n_rho_ = other.n_rho_;
    log_rho_min_ = other.log_rho_min_;
    log_rho_max_ = other.log_rho_max_;
    d_log_rho_inv_ = other.d_log_rho_inv_;

    other.d_log_rho_grid_ = nullptr;
    other.d_gamma_ion_ = nullptr;
    other.d_gamma_ele_ = nullptr;
    other.d_p_ref_ion_ = nullptr;
    other.d_p_ref_ele_ = nullptr;
    other.d_e_ref_ion_ = nullptr;
    other.d_e_ref_ele_ = nullptr;
    other.d_dgamma_ion_dlogrho_ = nullptr;
    other.d_dgamma_ele_dlogrho_ = nullptr;
    other.d_dp_ref_ion_dlogrho_ = nullptr;
    other.d_dp_ref_ele_dlogrho_ = nullptr;
    other.d_de_ref_ion_dlogrho_ = nullptr;
    other.d_de_ref_ele_dlogrho_ = nullptr;
    other.n_rho_ = 0;
    other.log_rho_min_ = 0.0;
    other.log_rho_max_ = 0.0;
    other.d_log_rho_inv_ = 0.0;
  }
  return *this;
}

void DeviceMieGruneisen::upload(const MieGruneisenEOS& cpu_eos) {
  free_all();
  if (cpu_eos.empty()) {
    return;
  }

  const int n_rho = checked_int(cpu_eos.n_rho(), "DeviceMieGruneisen::upload n_rho overflow");
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rho);

  const auto alloc_and_copy = [&](double** dst,
                                  const std::vector<double>& src,
                                  const char* malloc_message,
                                  const char* memcpy_message) {
    TENRYU_ASSERT(src.size() == static_cast<std::size_t>(n_rho),
                  "DeviceMieGruneisen::upload field size mismatch");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(dst), bytes), malloc_message);
    cuda_check(cudaMemcpy(*dst, src.data(), bytes, cudaMemcpyHostToDevice), memcpy_message);
  };

  alloc_and_copy(&d_log_rho_grid_, cpu_eos.log_rho_grid(),
                 "DeviceMieGruneisen::upload cudaMalloc log_rho_grid failed",
                 "DeviceMieGruneisen::upload cudaMemcpy log_rho_grid failed");
  alloc_and_copy(&d_gamma_ion_, cpu_eos.gamma_ion(),
                 "DeviceMieGruneisen::upload cudaMalloc gamma_ion failed",
                 "DeviceMieGruneisen::upload cudaMemcpy gamma_ion failed");
  alloc_and_copy(&d_gamma_ele_, cpu_eos.gamma_ele(),
                 "DeviceMieGruneisen::upload cudaMalloc gamma_ele failed",
                 "DeviceMieGruneisen::upload cudaMemcpy gamma_ele failed");
  alloc_and_copy(&d_p_ref_ion_, cpu_eos.p_ref_ion(),
                 "DeviceMieGruneisen::upload cudaMalloc p_ref_ion failed",
                 "DeviceMieGruneisen::upload cudaMemcpy p_ref_ion failed");
  alloc_and_copy(&d_p_ref_ele_, cpu_eos.p_ref_ele(),
                 "DeviceMieGruneisen::upload cudaMalloc p_ref_ele failed",
                 "DeviceMieGruneisen::upload cudaMemcpy p_ref_ele failed");
  alloc_and_copy(&d_e_ref_ion_, cpu_eos.e_ref_ion(),
                 "DeviceMieGruneisen::upload cudaMalloc e_ref_ion failed",
                 "DeviceMieGruneisen::upload cudaMemcpy e_ref_ion failed");
  alloc_and_copy(&d_e_ref_ele_, cpu_eos.e_ref_ele(),
                 "DeviceMieGruneisen::upload cudaMalloc e_ref_ele failed",
                 "DeviceMieGruneisen::upload cudaMemcpy e_ref_ele failed");
  alloc_and_copy(&d_dgamma_ion_dlogrho_, cpu_eos.dgamma_ion_dlogrho(),
                 "DeviceMieGruneisen::upload cudaMalloc dgamma_ion failed",
                 "DeviceMieGruneisen::upload cudaMemcpy dgamma_ion failed");
  alloc_and_copy(&d_dgamma_ele_dlogrho_, cpu_eos.dgamma_ele_dlogrho(),
                 "DeviceMieGruneisen::upload cudaMalloc dgamma_ele failed",
                 "DeviceMieGruneisen::upload cudaMemcpy dgamma_ele failed");
  alloc_and_copy(&d_dp_ref_ion_dlogrho_, cpu_eos.dp_ref_ion_dlogrho(),
                 "DeviceMieGruneisen::upload cudaMalloc dp_ref_ion failed",
                 "DeviceMieGruneisen::upload cudaMemcpy dp_ref_ion failed");
  alloc_and_copy(&d_dp_ref_ele_dlogrho_, cpu_eos.dp_ref_ele_dlogrho(),
                 "DeviceMieGruneisen::upload cudaMalloc dp_ref_ele failed",
                 "DeviceMieGruneisen::upload cudaMemcpy dp_ref_ele failed");
  alloc_and_copy(&d_de_ref_ion_dlogrho_, cpu_eos.de_ref_ion_dlogrho(),
                 "DeviceMieGruneisen::upload cudaMalloc de_ref_ion failed",
                 "DeviceMieGruneisen::upload cudaMemcpy de_ref_ion failed");
  alloc_and_copy(&d_de_ref_ele_dlogrho_, cpu_eos.de_ref_ele_dlogrho(),
                 "DeviceMieGruneisen::upload cudaMalloc de_ref_ele failed",
                 "DeviceMieGruneisen::upload cudaMemcpy de_ref_ele failed");

  n_rho_ = n_rho;
  log_rho_min_ = cpu_eos.log_rho_grid().front();
  log_rho_max_ = cpu_eos.log_rho_grid().back();
  if (n_rho_ > 1) {
    const double d = cpu_eos.log_rho_grid()[1] - cpu_eos.log_rho_grid()[0];
    bool uniform = true;
    for (int i = 2; i < n_rho_; ++i) {
      const double step =
          cpu_eos.log_rho_grid()[static_cast<std::size_t>(i)] -
          cpu_eos.log_rho_grid()[static_cast<std::size_t>(i - 1)];
      if (std::fabs(step - d) > 1.0e-10 * std::fabs(d)) {
        uniform = false;
        break;
      }
    }
    d_log_rho_inv_ = uniform ? (1.0 / d) : 0.0;
  }
}

MieGruneisenDeviceView DeviceMieGruneisen::view() const {
  MieGruneisenDeviceView out{};
  out.log_rho_grid = d_log_rho_grid_;
  out.gamma_ion = d_gamma_ion_;
  out.gamma_ele = d_gamma_ele_;
  out.p_ref_ion = d_p_ref_ion_;
  out.p_ref_ele = d_p_ref_ele_;
  out.e_ref_ion = d_e_ref_ion_;
  out.e_ref_ele = d_e_ref_ele_;
  out.dgamma_ion_dlogrho = d_dgamma_ion_dlogrho_;
  out.dgamma_ele_dlogrho = d_dgamma_ele_dlogrho_;
  out.dp_ref_ion_dlogrho = d_dp_ref_ion_dlogrho_;
  out.dp_ref_ele_dlogrho = d_dp_ref_ele_dlogrho_;
  out.de_ref_ion_dlogrho = d_de_ref_ion_dlogrho_;
  out.de_ref_ele_dlogrho = d_de_ref_ele_dlogrho_;
  out.n_rho = n_rho_;
  out.log_rho_min = log_rho_min_;
  out.log_rho_max = log_rho_max_;
  out.d_log_rho_inv = d_log_rho_inv_;
  return out;
}

void DeviceMieGruneisen::free_all() {
  if (d_de_ref_ele_dlogrho_ != nullptr) {
    cuda_check(cudaFree(d_de_ref_ele_dlogrho_),
               "DeviceMieGruneisen::free_all cudaFree de_ref_ele failed");
    d_de_ref_ele_dlogrho_ = nullptr;
  }
  if (d_de_ref_ion_dlogrho_ != nullptr) {
    cuda_check(cudaFree(d_de_ref_ion_dlogrho_),
               "DeviceMieGruneisen::free_all cudaFree de_ref_ion failed");
    d_de_ref_ion_dlogrho_ = nullptr;
  }
  if (d_dp_ref_ele_dlogrho_ != nullptr) {
    cuda_check(cudaFree(d_dp_ref_ele_dlogrho_),
               "DeviceMieGruneisen::free_all cudaFree dp_ref_ele failed");
    d_dp_ref_ele_dlogrho_ = nullptr;
  }
  if (d_dp_ref_ion_dlogrho_ != nullptr) {
    cuda_check(cudaFree(d_dp_ref_ion_dlogrho_),
               "DeviceMieGruneisen::free_all cudaFree dp_ref_ion failed");
    d_dp_ref_ion_dlogrho_ = nullptr;
  }
  if (d_dgamma_ele_dlogrho_ != nullptr) {
    cuda_check(cudaFree(d_dgamma_ele_dlogrho_),
               "DeviceMieGruneisen::free_all cudaFree dgamma_ele failed");
    d_dgamma_ele_dlogrho_ = nullptr;
  }
  if (d_dgamma_ion_dlogrho_ != nullptr) {
    cuda_check(cudaFree(d_dgamma_ion_dlogrho_),
               "DeviceMieGruneisen::free_all cudaFree dgamma_ion failed");
    d_dgamma_ion_dlogrho_ = nullptr;
  }
  if (d_e_ref_ele_ != nullptr) {
    cuda_check(cudaFree(d_e_ref_ele_),
               "DeviceMieGruneisen::free_all cudaFree e_ref_ele failed");
    d_e_ref_ele_ = nullptr;
  }
  if (d_e_ref_ion_ != nullptr) {
    cuda_check(cudaFree(d_e_ref_ion_),
               "DeviceMieGruneisen::free_all cudaFree e_ref_ion failed");
    d_e_ref_ion_ = nullptr;
  }
  if (d_p_ref_ele_ != nullptr) {
    cuda_check(cudaFree(d_p_ref_ele_),
               "DeviceMieGruneisen::free_all cudaFree p_ref_ele failed");
    d_p_ref_ele_ = nullptr;
  }
  if (d_p_ref_ion_ != nullptr) {
    cuda_check(cudaFree(d_p_ref_ion_),
               "DeviceMieGruneisen::free_all cudaFree p_ref_ion failed");
    d_p_ref_ion_ = nullptr;
  }
  if (d_gamma_ele_ != nullptr) {
    cuda_check(cudaFree(d_gamma_ele_),
               "DeviceMieGruneisen::free_all cudaFree gamma_ele failed");
    d_gamma_ele_ = nullptr;
  }
  if (d_gamma_ion_ != nullptr) {
    cuda_check(cudaFree(d_gamma_ion_),
               "DeviceMieGruneisen::free_all cudaFree gamma_ion failed");
    d_gamma_ion_ = nullptr;
  }
  if (d_log_rho_grid_ != nullptr) {
    cuda_check(cudaFree(d_log_rho_grid_),
               "DeviceMieGruneisen::free_all cudaFree log_rho_grid failed");
    d_log_rho_grid_ = nullptr;
  }
  n_rho_ = 0;
  log_rho_min_ = 0.0;
  log_rho_max_ = 0.0;
  d_log_rho_inv_ = 0.0;
}

}  // namespace tenryu::materials
