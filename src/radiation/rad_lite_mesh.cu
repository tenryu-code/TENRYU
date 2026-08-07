#include "radiation/rad_lite_mesh.hpp"

#include <algorithm>

#include <cuda_runtime.h>

#include "core/error.hpp"

namespace tenryu::radiation {
namespace {

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

constexpr int kMaxHydroPerRad = 50;
constexpr double kSigmaFloor = 1.0e-30;

[[nodiscard]] inline int flatten_index(const int cell, const int group, const int n_groups) {
  return cell * n_groups + group;
}

[[nodiscard]] bool can_merge_edge(const int left_cell,
                                  const int n_groups,
                                  const double* sigma_a_eff,
                                  const double* sigma_s_eff,
                                  const int8_t* ddmc_mode,
                                  const int* material_id,
                                  const double sigma_ratio_max) {
  const int right_cell = left_cell + 1;

  if (material_id != nullptr && material_id[left_cell] != material_id[right_cell]) {
    return false;
  }

  if (ddmc_mode != nullptr) {
    for (int g = 0; g < n_groups; ++g) {
      const int left_idx = flatten_index(left_cell, g, n_groups);
      const int right_idx = flatten_index(right_cell, g, n_groups);
      if (ddmc_mode[left_idx] != ddmc_mode[right_idx]) {
        return false;
      }
    }
  }

  for (int g = 0; g < n_groups; ++g) {
    const int left_idx = flatten_index(left_cell, g, n_groups);
    const int right_idx = flatten_index(right_cell, g, n_groups);
    const double sigma_t_left = sigma_a_eff[left_idx] + sigma_s_eff[left_idx];
    const double sigma_t_right = sigma_a_eff[right_idx] + sigma_s_eff[right_idx];
    const double sigma_t_max = std::max(sigma_t_left, sigma_t_right);
    const double sigma_t_min = std::min(sigma_t_left, sigma_t_right);
    const double ratio = sigma_t_max / std::max(sigma_t_min, kSigmaFloor);
    if (!(ratio < sigma_ratio_max)) {
      return false;
    }
  }

  return true;
}

__global__ void redistribute_tallies_kernel(double* hydro_dep,
                                            double* hydro_tl,
                                            const double* rad_dep,
                                            const double* rad_tl,
                                            const int32_t* hydro_to_rad,
                                            const double* w_dep,
                                            const double* w_tl,
                                            const int n_hydro,
                                            const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_hydro * n_groups) {
    return;
  }
  const int i = tid / n_groups;     // hydro cell
  const int g = tid % n_groups;     // group
  const int r = hydro_to_rad[i];    // rad cell
  hydro_dep[tid] = rad_dep[r * n_groups + g] * w_dep[tid];
  hydro_tl[tid] = rad_tl[r * n_groups + g] * w_tl[i];
}

__global__ void remap_cell_ids_to_rad_kernel(int32_t* cell_id,
                                             const int32_t* hydro_to_rad,
                                             const int n) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const int32_t c = cell_id[tid];
  if (c >= 0) {
    cell_id[tid] = hydro_to_rad[c];
  }
}

__global__ void remap_cell_ids_to_hydro_kernel(int32_t* cell_id,
                                               const double* pos_r,
                                               const double* hydro_node_r,
                                               const int32_t* rad_h_begin,
                                               const int32_t* rad_h_end,
                                               const int n,
                                               const int n_rad) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) {
    return;
  }
  const int32_t rad_c = cell_id[tid];
  if (rad_c < 0 || rad_c >= n_rad) {
    return;  // sentinel or invalid -> keep
  }

  const int32_t h_begin = rad_h_begin[rad_c];
  const int32_t h_end = rad_h_end[rad_c];
  if (h_end - h_begin <= 1) {
    // Single hydro cell in this rad cell
    cell_id[tid] = h_begin;
    return;
  }

  // Binary search for hydro cell containing pos_r[tid]
  const double r = pos_r[tid];
  // DDMC sentinel particles have NaN position -> map to first hydro cell in rad cell
  if (!isfinite(r)) {
    cell_id[tid] = h_begin;
    return;
  }
  int32_t lo = h_begin;
  int32_t hi = h_end - 1;
  while (lo < hi) {
    const int32_t mid = lo + (hi - lo) / 2;
    if (hydro_node_r[mid + 1] <= r) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }

  // Clamp to valid range
  if (lo < h_begin) {
    lo = h_begin;
  }
  if (lo >= h_end) {
    lo = h_end - 1;
  }
  cell_id[tid] = lo;
}

}  // namespace

RadLiteMesh1D build_rad_lite_mesh(const int n_cells,
                                  const int n_groups,
                                  const double* node_r,
                                  const double* vol,
                                  const double* sigma_a_eff,
                                  const double* sigma_s_eff,
                                  const double* Te,
                                  const int8_t* ddmc_mode,
                                  const int* material_id,
                                  const double sigma_ratio_max) {
  RadLiteMesh1D mesh;
  mesh.n_hydro = n_cells;
  mesh.n_groups = n_groups;

  if (n_cells <= 0 || n_groups <= 0 || node_r == nullptr || vol == nullptr ||
      sigma_a_eff == nullptr || sigma_s_eff == nullptr || Te == nullptr) {
    return mesh;
  }

  mesh.hydro_to_rad.assign(n_cells, 0);

  int h_begin = 0;
  while (h_begin < n_cells) {
    int h_end = h_begin + 1;
    while (h_end < n_cells) {
      const int current_count = h_end - h_begin;
      if (current_count >= kMaxHydroPerRad) {
        break;
      }
      if (!can_merge_edge(h_end - 1,
                          n_groups,
                          sigma_a_eff,
                          sigma_s_eff,
                          ddmc_mode,
                          material_id,
                          sigma_ratio_max)) {
        break;
      }
      ++h_end;
    }

    const int32_t rad_idx = static_cast<int32_t>(mesh.rad_h_begin.size());
    mesh.rad_h_begin.push_back(h_begin);
    mesh.rad_h_end.push_back(h_end);
    for (int i = h_begin; i < h_end; ++i) {
      mesh.hydro_to_rad[i] = rad_idx;
    }
    h_begin = h_end;
  }

  mesh.n_rad = static_cast<int>(mesh.rad_h_begin.size());
  mesh.rad_node_r.resize(mesh.n_rad + 1);
  for (int r = 0; r < mesh.n_rad; ++r) {
    mesh.rad_node_r[r] = node_r[mesh.rad_h_begin[r]];
  }
  mesh.rad_node_r[mesh.n_rad] = node_r[n_cells];

  mesh.sigma_a_eff_rad.assign(mesh.n_rad * n_groups, 0.0);
  mesh.sigma_s_eff_rad.assign(mesh.n_rad * n_groups, 0.0);
  mesh.Te_rad.assign(mesh.n_rad, 0.0);
  mesh.w_dep.assign(n_cells * n_groups, 0.0);
  mesh.w_tl.assign(n_cells, 0.0);

  for (int r = 0; r < mesh.n_rad; ++r) {
    const int begin = mesh.rad_h_begin[r];
    const int end = mesh.rad_h_end[r];
    const int span = end - begin;

    double vol_sum = 0.0;
    double Te_vol_sum = 0.0;
    for (int i = begin; i < end; ++i) {
      const double Vi = vol[i];
      vol_sum += Vi;
      Te_vol_sum += Vi * Te[i];
    }

    if (vol_sum > 0.0) {
      mesh.Te_rad[r] = Te_vol_sum / vol_sum;
      for (int i = begin; i < end; ++i) {
        mesh.w_tl[i] = vol[i] / vol_sum;
      }
    } else if (span > 0) {
      const double uniform = 1.0 / static_cast<double>(span);
      for (int i = begin; i < end; ++i) {
        mesh.w_tl[i] = uniform;
      }
    }

    for (int g = 0; g < n_groups; ++g) {
      double sigma_a_vol_sum = 0.0;
      double sigma_s_vol_sum = 0.0;
      double dep_denom = 0.0;
      for (int i = begin; i < end; ++i) {
        const int idx = flatten_index(i, g, n_groups);
        const double Vi = vol[i];
        sigma_a_vol_sum += Vi * sigma_a_eff[idx];
        sigma_s_vol_sum += Vi * sigma_s_eff[idx];
        dep_denom += Vi * sigma_a_eff[idx];
      }

      const int ridx = flatten_index(r, g, n_groups);
      if (vol_sum > 0.0) {
        mesh.sigma_a_eff_rad[ridx] = sigma_a_vol_sum / vol_sum;
        mesh.sigma_s_eff_rad[ridx] = sigma_s_vol_sum / vol_sum;
      }

      if (dep_denom > 0.0) {
        for (int i = begin; i < end; ++i) {
          const int idx = flatten_index(i, g, n_groups);
          mesh.w_dep[idx] = vol[i] * sigma_a_eff[idx] / dep_denom;
        }
      } else if (vol_sum > 0.0) {
        for (int i = begin; i < end; ++i) {
          const int idx = flatten_index(i, g, n_groups);
          mesh.w_dep[idx] = vol[i] / vol_sum;
        }
      } else if (span > 0) {
        const double uniform = 1.0 / static_cast<double>(span);
        for (int i = begin; i < end; ++i) {
          const int idx = flatten_index(i, g, n_groups);
          mesh.w_dep[idx] = uniform;
        }
      }
    }
  }

  mesh.enabled = (mesh.n_rad < mesh.n_hydro);
  return mesh;
}

void redistribute_tallies_cuda(double* hydro_dep,
                               double* hydro_tl,
                               const double* rad_dep,
                               const double* rad_tl,
                               const int32_t* hydro_to_rad,
                               const double* w_dep,
                               const double* w_tl,
                               const int n_hydro,
                               const int n_groups) {
  if (hydro_dep == nullptr || hydro_tl == nullptr || rad_dep == nullptr || rad_tl == nullptr ||
      hydro_to_rad == nullptr || w_dep == nullptr || w_tl == nullptr || n_hydro <= 0 ||
      n_groups <= 0) {
    return;
  }

  constexpr int kBlock = 256;
  const int n_total = n_hydro * n_groups;
  const int grid = (n_total + kBlock - 1) / kBlock;
  if (grid > 0) {
    redistribute_tallies_kernel<<<grid, kBlock>>>(
        hydro_dep, hydro_tl, rad_dep, rad_tl, hydro_to_rad, w_dep, w_tl, n_hydro, n_groups);
    cuda_check(cudaGetLastError(), "redistribute_tallies kernel launch failed");
  }
}

RadDeviceData prepare_rad_device_data(const RadLiteMesh1D& mesh,
                                      const double* hydro_node_r,
                                      const double* hydro_vol,
                                      const double* sigma_R_hydro,
                                      const double* sigma_a_hydro,
                                      const double* fleck_f_hydro,
                                      const int8_t* ddmc_mode_hydro,
                                      const double* eta_cdf_hydro) {
  (void)eta_cdf_hydro;

  RadDeviceData data;
  const int n_rad = mesh.n_rad;
  const int n_hydro = mesh.n_hydro;
  const int n_groups = mesh.n_groups;
  if (n_rad <= 0 || n_hydro <= 0 || n_groups <= 0 || hydro_node_r == nullptr ||
      hydro_vol == nullptr || sigma_a_hydro == nullptr || fleck_f_hydro == nullptr) {
    return data;
  }

  const size_t n_rad_groups = static_cast<size_t>(n_rad) * static_cast<size_t>(n_groups);
  const size_t n_hydro_groups = static_cast<size_t>(n_hydro) * static_cast<size_t>(n_groups);

  if (mesh.hydro_to_rad.size() != static_cast<size_t>(n_hydro) ||
      mesh.rad_h_begin.size() != static_cast<size_t>(n_rad) ||
      mesh.rad_h_end.size() != static_cast<size_t>(n_rad) ||
      mesh.rad_node_r.size() != static_cast<size_t>(n_rad + 1) ||
      mesh.sigma_a_eff_rad.size() != n_rad_groups || mesh.sigma_s_eff_rad.size() != n_rad_groups ||
      mesh.Te_rad.size() != static_cast<size_t>(n_rad) || mesh.w_dep.size() != n_hydro_groups ||
      mesh.w_tl.size() != static_cast<size_t>(n_hydro)) {
    return data;
  }

  std::vector<double> vol_rad(n_rad, 0.0);
  std::vector<double> sigma_R_rad(n_rad_groups, 0.0);
  std::vector<double> sigma_a_rad(n_rad_groups, 0.0);
  std::vector<double> cell_dx_rad(n_rad, 0.0);
  std::vector<double> fleck_f_rad(n_rad, 0.0);
  std::vector<int8_t> ddmc_mode_rad(n_rad_groups, 0);

  for (int r = 0; r < n_rad; ++r) {
    int32_t h_begin = mesh.rad_h_begin[r];
    int32_t h_end = mesh.rad_h_end[r];
    if (h_begin < 0) {
      h_begin = 0;
    }
    if (h_begin > n_hydro) {
      h_begin = n_hydro;
    }
    if (h_end < h_begin) {
      h_end = h_begin;
    }
    if (h_end > n_hydro) {
      h_end = n_hydro;
    }

    cell_dx_rad[r] = mesh.rad_node_r[r + 1] - mesh.rad_node_r[r];
    const int span = h_end - h_begin;

    double vol_sum = 0.0;
    double fleck_vol_sum = 0.0;
    double fleck_plain_sum = 0.0;
    for (int i = h_begin; i < h_end; ++i) {
      const double Vi = hydro_vol[i];
      vol_sum += Vi;
      fleck_vol_sum += Vi * fleck_f_hydro[i];
      fleck_plain_sum += fleck_f_hydro[i];
    }
    vol_rad[r] = vol_sum;

    if (vol_sum > 0.0) {
      fleck_f_rad[r] = fleck_vol_sum / vol_sum;
    } else if (span > 0) {
      fleck_f_rad[r] = fleck_plain_sum / static_cast<double>(span);
    }

    for (int g = 0; g < n_groups; ++g) {
      const int ridx = flatten_index(r, g, n_groups);
      if (ddmc_mode_hydro != nullptr && span > 0) {
        ddmc_mode_rad[ridx] = ddmc_mode_hydro[flatten_index(h_begin, g, n_groups)];
      }

      double sigma_a_vol_sum = 0.0;
      double sigma_R_vol_sum = 0.0;
      double sigma_a_plain_sum = 0.0;
      double sigma_R_plain_sum = 0.0;
      for (int i = h_begin; i < h_end; ++i) {
        const int idx = flatten_index(i, g, n_groups);
        const double Vi = hydro_vol[i];
        sigma_a_vol_sum += Vi * sigma_a_hydro[idx];
        sigma_a_plain_sum += sigma_a_hydro[idx];
        if (sigma_R_hydro != nullptr) {
          sigma_R_vol_sum += Vi * sigma_R_hydro[idx];
          sigma_R_plain_sum += sigma_R_hydro[idx];
        }
      }

      if (vol_sum > 0.0) {
        sigma_a_rad[ridx] = sigma_a_vol_sum / vol_sum;
        if (sigma_R_hydro != nullptr) {
          sigma_R_rad[ridx] = sigma_R_vol_sum / vol_sum;
        }
      } else if (span > 0) {
        sigma_a_rad[ridx] = sigma_a_plain_sum / static_cast<double>(span);
        if (sigma_R_hydro != nullptr) {
          sigma_R_rad[ridx] = sigma_R_plain_sum / static_cast<double>(span);
        }
      }
    }
  }

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.sigma_a_eff), n_rad_groups * sizeof(double)),
             "cudaMalloc sigma_a_eff failed");
  cuda_check(cudaMemcpy(data.sigma_a_eff,
                        mesh.sigma_a_eff_rad.data(),
                        n_rad_groups * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy sigma_a_eff failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.sigma_s_eff), n_rad_groups * sizeof(double)),
             "cudaMalloc sigma_s_eff failed");
  cuda_check(cudaMemcpy(data.sigma_s_eff,
                        mesh.sigma_s_eff_rad.data(),
                        n_rad_groups * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy sigma_s_eff failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.Te), static_cast<size_t>(n_rad) * sizeof(double)),
             "cudaMalloc Te failed");
  cuda_check(cudaMemcpy(data.Te,
                        mesh.Te_rad.data(),
                        static_cast<size_t>(n_rad) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy Te failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.vol), static_cast<size_t>(n_rad) * sizeof(double)),
             "cudaMalloc vol failed");
  cuda_check(cudaMemcpy(
                 data.vol, vol_rad.data(), static_cast<size_t>(n_rad) * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy vol failed");

  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&data.node_r), static_cast<size_t>(n_rad + 1) * sizeof(double)),
      "cudaMalloc node_r failed");
  cuda_check(cudaMemcpy(data.node_r,
                        mesh.rad_node_r.data(),
                        static_cast<size_t>(n_rad + 1) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy node_r failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.sigma_R), n_rad_groups * sizeof(double)),
             "cudaMalloc sigma_R failed");
  cuda_check(cudaMemcpy(
                 data.sigma_R, sigma_R_rad.data(), n_rad_groups * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy sigma_R failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.sigma_a), n_rad_groups * sizeof(double)),
             "cudaMalloc sigma_a failed");
  cuda_check(cudaMemcpy(
                 data.sigma_a, sigma_a_rad.data(), n_rad_groups * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy sigma_a failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.cell_dx), static_cast<size_t>(n_rad) * sizeof(double)),
             "cudaMalloc cell_dx failed");
  cuda_check(cudaMemcpy(data.cell_dx,
                        cell_dx_rad.data(),
                        static_cast<size_t>(n_rad) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy cell_dx failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.fleck_f), static_cast<size_t>(n_rad) * sizeof(double)),
             "cudaMalloc fleck_f failed");
  cuda_check(cudaMemcpy(data.fleck_f,
                        fleck_f_rad.data(),
                        static_cast<size_t>(n_rad) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy fleck_f failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.ddmc_mode), n_rad_groups * sizeof(int8_t)),
             "cudaMalloc ddmc_mode failed");
  cuda_check(cudaMemcpy(data.ddmc_mode,
                        ddmc_mode_rad.data(),
                        n_rad_groups * sizeof(int8_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy ddmc_mode failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.rad_dep), n_rad_groups * sizeof(double)),
             "cudaMalloc rad_dep failed");
  cuda_check(cudaMemset(data.rad_dep, 0, n_rad_groups * sizeof(double)), "cudaMemset rad_dep failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.rad_E_tally), n_rad_groups * sizeof(double)),
             "cudaMalloc rad_E_tally failed");
  cuda_check(
      cudaMemset(data.rad_E_tally, 0, n_rad_groups * sizeof(double)), "cudaMemset rad_E_tally failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.hydro_to_rad),
                        static_cast<size_t>(n_hydro) * sizeof(int32_t)),
             "cudaMalloc hydro_to_rad failed");
  cuda_check(cudaMemcpy(data.hydro_to_rad,
                        mesh.hydro_to_rad.data(),
                        static_cast<size_t>(n_hydro) * sizeof(int32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy hydro_to_rad failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.w_dep), n_hydro_groups * sizeof(double)),
             "cudaMalloc w_dep failed");
  cuda_check(cudaMemcpy(
                 data.w_dep, mesh.w_dep.data(), n_hydro_groups * sizeof(double), cudaMemcpyHostToDevice),
             "cudaMemcpy w_dep failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.w_tl), static_cast<size_t>(n_hydro) * sizeof(double)),
             "cudaMalloc w_tl failed");
  cuda_check(cudaMemcpy(data.w_tl,
                        mesh.w_tl.data(),
                        static_cast<size_t>(n_hydro) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy w_tl failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.hydro_node_r),
                        static_cast<size_t>(n_hydro + 1) * sizeof(double)),
             "cudaMalloc hydro_node_r failed");
  cuda_check(cudaMemcpy(data.hydro_node_r,
                        hydro_node_r,
                        static_cast<size_t>(n_hydro + 1) * sizeof(double),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy hydro_node_r failed");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&data.rad_h_begin),
                        static_cast<size_t>(n_rad) * sizeof(int32_t)),
             "cudaMalloc rad_h_begin failed");
  cuda_check(cudaMemcpy(data.rad_h_begin,
                        mesh.rad_h_begin.data(),
                        static_cast<size_t>(n_rad) * sizeof(int32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy rad_h_begin failed");

  cuda_check(
      cudaMalloc(reinterpret_cast<void**>(&data.rad_h_end), static_cast<size_t>(n_rad) * sizeof(int32_t)),
      "cudaMalloc rad_h_end failed");
  cuda_check(cudaMemcpy(data.rad_h_end,
                        mesh.rad_h_end.data(),
                        static_cast<size_t>(n_rad) * sizeof(int32_t),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy rad_h_end failed");

  data.eta_cdf = nullptr;
  data.n_rad = n_rad;
  data.n_hydro = n_hydro;
  data.n_groups = n_groups;
  data.active = true;
  return data;
}

void free_rad_device_data(RadDeviceData& data) {
  if (data.sigma_a_eff != nullptr) {
    cuda_check(cudaFree(data.sigma_a_eff), "cudaFree sigma_a_eff failed");
    data.sigma_a_eff = nullptr;
  }
  if (data.sigma_s_eff != nullptr) {
    cuda_check(cudaFree(data.sigma_s_eff), "cudaFree sigma_s_eff failed");
    data.sigma_s_eff = nullptr;
  }
  if (data.Te != nullptr) {
    cuda_check(cudaFree(data.Te), "cudaFree Te failed");
    data.Te = nullptr;
  }
  if (data.vol != nullptr) {
    cuda_check(cudaFree(data.vol), "cudaFree vol failed");
    data.vol = nullptr;
  }
  if (data.node_r != nullptr) {
    cuda_check(cudaFree(data.node_r), "cudaFree node_r failed");
    data.node_r = nullptr;
  }
  if (data.sigma_R != nullptr) {
    cuda_check(cudaFree(data.sigma_R), "cudaFree sigma_R failed");
    data.sigma_R = nullptr;
  }
  if (data.sigma_a != nullptr) {
    cuda_check(cudaFree(data.sigma_a), "cudaFree sigma_a failed");
    data.sigma_a = nullptr;
  }
  if (data.cell_dx != nullptr) {
    cuda_check(cudaFree(data.cell_dx), "cudaFree cell_dx failed");
    data.cell_dx = nullptr;
  }
  if (data.fleck_f != nullptr) {
    cuda_check(cudaFree(data.fleck_f), "cudaFree fleck_f failed");
    data.fleck_f = nullptr;
  }
  if (data.ddmc_mode != nullptr) {
    cuda_check(cudaFree(data.ddmc_mode), "cudaFree ddmc_mode failed");
    data.ddmc_mode = nullptr;
  }
  if (data.eta_cdf != nullptr) {
    cuda_check(cudaFree(data.eta_cdf), "cudaFree eta_cdf failed");
    data.eta_cdf = nullptr;
  }
  if (data.rad_dep != nullptr) {
    cuda_check(cudaFree(data.rad_dep), "cudaFree rad_dep failed");
    data.rad_dep = nullptr;
  }
  if (data.rad_E_tally != nullptr) {
    cuda_check(cudaFree(data.rad_E_tally), "cudaFree rad_E_tally failed");
    data.rad_E_tally = nullptr;
  }
  if (data.hydro_to_rad != nullptr) {
    cuda_check(cudaFree(data.hydro_to_rad), "cudaFree hydro_to_rad failed");
    data.hydro_to_rad = nullptr;
  }
  if (data.w_dep != nullptr) {
    cuda_check(cudaFree(data.w_dep), "cudaFree w_dep failed");
    data.w_dep = nullptr;
  }
  if (data.w_tl != nullptr) {
    cuda_check(cudaFree(data.w_tl), "cudaFree w_tl failed");
    data.w_tl = nullptr;
  }
  if (data.hydro_node_r != nullptr) {
    cuda_check(cudaFree(data.hydro_node_r), "cudaFree hydro_node_r failed");
    data.hydro_node_r = nullptr;
  }
  if (data.rad_h_begin != nullptr) {
    cuda_check(cudaFree(data.rad_h_begin), "cudaFree rad_h_begin failed");
    data.rad_h_begin = nullptr;
  }
  if (data.rad_h_end != nullptr) {
    cuda_check(cudaFree(data.rad_h_end), "cudaFree rad_h_end failed");
    data.rad_h_end = nullptr;
  }

  data.n_rad = 0;
  data.n_hydro = 0;
  data.n_groups = 0;
  data.active = false;
}

void remap_cell_ids_to_rad_cuda(int32_t* cell_id, const int32_t* hydro_to_rad, const int n) {
  if (cell_id == nullptr || hydro_to_rad == nullptr || n <= 0) {
    return;
  }

  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  if (grid > 0) {
    remap_cell_ids_to_rad_kernel<<<grid, kBlock>>>(cell_id, hydro_to_rad, n);
    cuda_check(cudaGetLastError(), "remap_cell_ids_to_rad kernel launch failed");
  }
}

void remap_cell_ids_to_hydro_cuda(int32_t* cell_id,
                                  const double* pos_r,
                                  const double* hydro_node_r,
                                  const int32_t* rad_h_begin,
                                  const int32_t* rad_h_end,
                                  const int n,
                                  const int n_rad) {
  if (cell_id == nullptr || pos_r == nullptr || hydro_node_r == nullptr || rad_h_begin == nullptr ||
      rad_h_end == nullptr || n <= 0 || n_rad <= 0) {
    return;
  }

  constexpr int kBlock = 256;
  const int grid = (n + kBlock - 1) / kBlock;
  if (grid > 0) {
    remap_cell_ids_to_hydro_kernel<<<grid, kBlock>>>(
        cell_id, pos_r, hydro_node_r, rad_h_begin, rad_h_end, n, n_rad);
    cuda_check(cudaGetLastError(), "remap_cell_ids_to_hydro kernel launch failed");
  }
}

}  // namespace tenryu::radiation
