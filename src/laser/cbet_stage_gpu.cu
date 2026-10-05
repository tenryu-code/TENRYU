#include "laser/cbet_stage_gpu.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "core/device_ordered_sum.cuh"
#include "core/device_pack.hpp"
#include "core/error.hpp"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic below repeats the host
// loops it replaces (cbet.cu cbet_stage_cell_fields, laser_mesh.cu compute_cell_effective_A and
// the former laser.cu fill_cbet_viz_fields, fill_port_section_outgoing_power and capture loop)
// operation by operation, so contraction into fused multiply-adds must not change it.

namespace tenryu::laser {
namespace {

constexpr int kThreads = 256;

inline void stage_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure_capacity(T** ptr, std::size_t* capacity, const std::size_t needed, const char* message) {
  if (needed <= *capacity && *ptr != nullptr) {
    return;
  }
  if (*ptr != nullptr) {
    static_cast<void>(cudaFree(*ptr));
    *ptr = nullptr;
  }
  *capacity = 0;
  if (needed == 0) {
    return;
  }
  stage_check(cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), message);
  *capacity = needed;
}

int blocks_for(const long long n) {
  return static_cast<int>(std::max<long long>(1, (n + kThreads - 1) / kThreads));
}

// std::max(a, b) of the host loops: b when a < b, else a (a NaN first argument stays NaN).
__device__ inline double host_max(const double a, const double b) { return (a < b) ? b : a; }

__global__ void cell_A_eff_kernel(const int n_cells, const int n_materials,
                                  const double* __restrict__ volfrac,
                                  const double* __restrict__ material_A,
                                  const double material_A_default, double* __restrict__ A_eff) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells) {
    return;
  }
  double A = host_max(material_A_default, 1.0e-12);
  if (volfrac != nullptr) {
    const long long base = static_cast<long long>(cell) * n_materials;
    double frac_sum = 0.0;
    double inv_A_c = 0.0;
    for (int m = 0; m < n_materials; ++m) {
      const double frac = host_max(volfrac[base + m], 0.0);
      frac_sum += frac;
      const double A_m = host_max(material_A[m], 1.0e-12);
      inv_A_c += frac / A_m;
    }
    if (frac_sum > 1.0e-30) {
      inv_A_c /= frac_sum;
    }
    if (::isfinite(inv_A_c) && inv_A_c > 1.0e-30) {
      A = 1.0 / inv_A_c;
    }
  }
  A_eff[cell] = A;
}

struct CellFieldArgs {
  double eV;
  double c;
  double omega0;
  double n_crit;
  double lam_pref;
  double proton_mass;
  double eps_n;
  double cutoff;
};

__global__ void cell_fields_kernel(const int n_cells, const CellFieldArgs k,
                                   const double* __restrict__ rho_in,
                                   const double* __restrict__ zbar_in,
                                   const double* __restrict__ A_eff_in,
                                   const double* __restrict__ Te_in,
                                   const double* __restrict__ Ti_in,
                                   const double* __restrict__ v_r,
                                   const double* __restrict__ vol_in,
                                   const std::uint8_t* __restrict__ cell_is_void,
                                   double* __restrict__ chi_pref, double* __restrict__ c_a,
                                   double* __restrict__ u_r, double* __restrict__ k_bar,
                                   double* __restrict__ vol_out, std::uint8_t* __restrict__ mask) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells) {
    return;
  }
  const double rho = rho_in[i];
  const double zbar = host_max(zbar_in[i], 0.0);
  const double A_eff = host_max(A_eff_in[i], 1.0e-30);
  const double Te_erg = host_max(Te_in[i], 0.0) * k.eV;
  const double Ti_erg = host_max(Ti_in[i], 0.0) * k.eV;
  const double u_r_cell = 0.5 * (v_r[i] + v_r[i + 1]);
  const double ne = rho * zbar / (A_eff * k.proton_mass);
  const double nh_raw = ne / k.n_crit;
  const double vol = vol_in[i];
  const bool is_void = (cell_is_void != nullptr && cell_is_void[i] != 0);
  const bool finite_fields = ::isfinite(rho) && ::isfinite(zbar) && ::isfinite(A_eff) &&
                             ::isfinite(Te_erg) && ::isfinite(Ti_erg) && ::isfinite(u_r_cell) &&
                             ::isfinite(vol);
  const bool active = finite_fields && !is_void && rho > 0.0 && zbar > 0.0 && Te_erg > 0.0 &&
                      nh_raw > 0.0 && nh_raw < k.cutoff && vol > 0.0;
  mask[i] = active ? 1 : 0;
  if (active) {
    const double denom_T = zbar * Te_erg + 3.0 * Ti_erg;
    const double one_minus = host_max(1.0 - nh_raw, k.eps_n);
    chi_pref[i] = k.lam_pref * (nh_raw / one_minus) * (zbar / host_max(denom_T, 1.0e-300));
    c_a[i] = ::sqrt(host_max(denom_T, 0.0) / (A_eff * k.proton_mass));
    k_bar[i] = ::sqrt(host_max(1.0 - nh_raw, k.eps_n)) * k.omega0 / k.c;
    u_r[i] = u_r_cell;
  } else {
    chi_pref[i] = 0.0;
    c_a[i] = 0.0;
    u_r[i] = 0.0;
    k_bar[i] = 0.0;
  }
  vol_out[i] = vol;
}

__global__ void viz_fields_kernel(const int n_cells, const int G, const int n_branches,
                                  const int n_bins, const double* __restrict__ dQ,
                                  const double* __restrict__ iaw_cell,
                                  const double* __restrict__ cell_vol,
                                  double* __restrict__ gross, double* __restrict__ net_inbound,
                                  double* __restrict__ dq_abs_cell,
                                  double* __restrict__ dq_max_cell, int* __restrict__ flags) {
  const int cell = blockIdx.x * blockDim.x + threadIdx.x;
  if (cell >= n_cells) {
    return;
  }
  constexpr double kClosureTiny = 1.0e-300;
  double sum = 0.0;
  double sum_abs = 0.0;
  double sum_inbound = 0.0;
  double dq_max = 0.0;
  const long long row = static_cast<long long>(cell) * G;
  for (int g = 0; g < G; ++g) {
    const double dq = dQ[row + g];
    const double dq_abs = ::fabs(dq);
    dq_max = host_max(dq_max, dq_abs);
    sum += dq;
    sum_abs += dq_abs;
    const int branch = (g % (n_branches * n_bins)) / n_bins;
    if (branch == 0) {
      sum_inbound += dq;
    }
  }
  const double iaw = (iaw_cell != nullptr) ? iaw_cell[cell] : 0.0;
  if (!(::fabs(sum + iaw) <= 1.0e-9 * host_max(sum_abs + ::fabs(iaw), kClosureTiny))) {
    atomicOr(flags, kCbetFlagClosure);
  }
  const double vol = cell_vol[cell];
  if (!(::isfinite(vol) && vol > 0.0)) {
    atomicOr(flags, kCbetFlagCellVolume);
  }
  gross[cell] = 0.5 * sum_abs / vol;
  net_inbound[cell] = sum_inbound / vol;
  dq_abs_cell[cell] = sum_abs;
  dq_max_cell[cell] = dq_max;
}

constexpr int kAuditThreads = 256;
constexpr int kAuditPerThread = 4;

// One block: the audit over the cells (step_out[0] = max, step_out[1] = sum), with the values of
// the former one-thread loop in cell order. The fold max = host_max(max, x) from 0 keeps the largest
// non-NaN value (it never becomes NaN: a NaN x is passed over), so each thread folds a contiguous
// run of cells and the runs' results are folded in order; the sum adds the cells in order
// (core::device_ordered, the zero cells skipped: the running sum starts at +0).
__global__ void viz_audit_kernel(const int n_cells, const double* __restrict__ dq_abs_cell,
                                 const double* __restrict__ dq_max_cell,
                                 double* __restrict__ step_out) {
  __shared__ double sh_values[kAuditThreads * kAuditPerThread];
  __shared__ int sh_scan[kAuditThreads];
  __shared__ double sh_max[kAuditThreads];
  const int t = static_cast<int>(threadIdx.x);
  const int per_thread = (n_cells + kAuditThreads - 1) / kAuditThreads;
  const int begin = t * per_thread;
  const int end = ::min(n_cells, begin + per_thread);
  double local_max = 0.0;
  for (int cell = begin; cell < end; ++cell) {
    local_max = host_max(local_max, dq_max_cell[cell]);
  }
  sh_max[t] = local_max;
  const double dq_sum = core::device_ordered::block_ordered_sum_nonzero<kAuditThreads, kAuditPerThread>(
      dq_abs_cell, n_cells, 0.0, sh_values, sh_scan);
  __syncthreads();
  if (t == 0) {
    double dq_max = 0.0;
    for (int i = 0; i < kAuditThreads; ++i) {
      dq_max = host_max(dq_max, sh_max[i]);
    }
    step_out[0] = dq_max;
    step_out[1] = dq_sum;
  }
}

// One thread per port: the final record weight of every ray, summed in ray order.
__global__ void outgoing_power_kernel(const int n_ports, const int n_rays, const int cap_per_ray,
                                      const std::int32_t* __restrict__ rec_count,
                                      const std::int64_t* __restrict__ ray_rec_offset,
                                      const double* __restrict__ rec_w_ps,
                                      double* __restrict__ outgoing, int* __restrict__ flags) {
  const int port = blockIdx.x * blockDim.x + threadIdx.x;
  if (port >= n_ports) {
    return;
  }
  const long long records_per_port = static_cast<long long>(n_rays) * cap_per_ray;
  const double* port_w = rec_w_ps + static_cast<long long>(port) * records_per_port;
  double sum = 0.0;
  for (int ray = 0; ray < n_rays; ++ray) {
    const std::int64_t offset_count = ray_rec_offset[ray + 1] - ray_rec_offset[ray];
    const int count = rec_count[ray];
    if (offset_count != static_cast<std::int64_t>(count)) {
      atomicOr(flags, kCbetFlagRecordOffsets);
    }
    if (count <= 0) {
      continue;
    }
    sum += port_w[static_cast<long long>(ray) * cap_per_ray + (count - 1)];
  }
  outgoing[port] = sum;
}

// The capture rows (4 doubles each) are in the order (port, ray, capture channel) of the former
// host loop. One thread per row: the config channel the row's power adds to, or -1 for a row that
// adds nothing (not captured, or with a channel or cell out of range, which also sets the flag the
// loop set).
__global__ void capture_classify_kernel(const long long n_rows, const int n_channels,
                                        const int n_config, const int n_cells,
                                        const double* __restrict__ stage,
                                        const std::int32_t* __restrict__ capture_order,
                                        std::int32_t* __restrict__ row_config,
                                        int* __restrict__ flags) {
  const long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }
  const double* capture = stage + row * 4;
  std::int32_t config_out = -1;
  if (capture[0] > 0.5 && capture[1] > 0.0) {
    const int config_index = capture_order[row % n_channels];
    const int cell = static_cast<int>(capture[3]);
    if (config_index < 0 || config_index >= n_config) {
      atomicOr(flags, kCbetFlagCaptureChannel);
    } else if (cell < 0 || cell >= n_cells) {
      atomicOr(flags, kCbetFlagCaptureCell);
    } else {
      config_out = config_index;
    }
  }
  row_config[row] = config_out;
}

constexpr int kCaptureThreads = 256;
constexpr int kCapturePerThread = 4;

// The sums over the rows that add (row_config >= 0), each in row order with the former loop's
// expressions, one block per former thread: block c < K (K = n_config) the sums of P, r P and mu P
// of config channel c over all rows (step_out[3 + c], [3 + K + c], [3 + 2K + c]); block
// K + port * K + c the power port `port` captured into channel c over the port's rows (pcross,
// zeroed before); block K + n_ports * K the banked power over all rows (step_out[2]). The threads
// form the terms of the rows that add (the products as the loop rounded them) and list them in
// row order, kCaptureThreads * kCapturePerThread rows at a time; thread 32 s adds sum s's terms in
// that order.
__global__ void capture_sum_kernel(const int n_ports, const int n_rays, const int n_channels,
                                   const int n_config, const double* __restrict__ stage,
                                   const std::int32_t* __restrict__ row_config,
                                   const double* __restrict__ one_minus_eta,
                                   const double* __restrict__ x_r, double* __restrict__ pcross,
                                   double* __restrict__ step_out) {
  constexpr int kChunk = kCaptureThreads * kCapturePerThread;
  __shared__ double sh_terms[3][kChunk];
  __shared__ int sh_scan[kCaptureThreads];
  const int t = static_cast<int>(threadIdx.x);
  const int b = static_cast<int>(blockIdx.x);
  const long long rows_per_port = static_cast<long long>(n_rays) * n_channels;
  const long long n_rows = static_cast<long long>(n_ports) * rows_per_port;
  // the block's sums: 0 = the config channel's P, r P and mu P; 1 = a port's captured power; 2 = the
  // banked power
  int kind = 0;
  int config_index = b;
  long long first = 0;
  long long end = n_rows;
  double* pcross_out = nullptr;
  if (b >= n_config + n_ports * n_config) {
    kind = 2;
    config_index = -1;
  } else if (b >= n_config) {
    kind = 1;
    const int port = (b - n_config) / n_config;
    config_index = (b - n_config) % n_config;
    first = static_cast<long long>(port) * rows_per_port;
    end = first + rows_per_port;
    pcross_out = pcross + static_cast<long long>(port) * n_config + config_index;
  }
  const int n_sums = (kind == 0) ? 3 : 1;
  const int sum = t / 32;
  const bool adder = (t % 32 == 0) && sum < n_sums;
  double acc = (kind == 1 && t == 0) ? *pcross_out : 0.0;
  for (long long chunk = first; chunk < end; chunk += kChunk) {
    double terms[kCapturePerThread][3];
    bool adds[kCapturePerThread];
    int local = 0;
#pragma unroll
    for (int j = 0; j < kCapturePerThread; ++j) {
      const long long row = chunk + static_cast<long long>(t) * kCapturePerThread + j;
      adds[j] = false;
      if (row < end) {
        const std::int32_t row_c = row_config[row];
        adds[j] = (kind == 2) ? (row_c >= 0) : (row_c == config_index);
      }
      if (adds[j]) {
        const double* capture = stage + row * 4;
        if (kind == 0) {
          const int cell = static_cast<int>(capture[3]);
          terms[j][0] = capture[1];
          const double r_cell = 0.5 * (x_r[cell] + x_r[cell + 1]);
          terms[j][1] = r_cell * capture[1];
          terms[j][2] = capture[2] * capture[1];
        } else if (kind == 1) {
          terms[j][0] = capture[1];
        } else {
          terms[j][0] = capture[1] * (1.0 - one_minus_eta[row % n_channels]);
        }
        ++local;
      }
    }
    int total = 0;
    int at = core::device_ordered::block_exclusive_prefix<kCaptureThreads>(local, sh_scan, &total);
#pragma unroll
    for (int j = 0; j < kCapturePerThread; ++j) {
      if (adds[j]) {
        for (int s = 0; s < n_sums; ++s) {
          sh_terms[s][at] = terms[j][s];
        }
        ++at;
      }
    }
    __syncthreads();
    if (adder) {
      for (int m = 0; m < total; ++m) {
        acc += sh_terms[sum][m];
      }
    }
    __syncthreads();
  }
  if (!adder) {
    return;
  }
  if (kind == 0) {
    step_out[3 + sum * n_config + config_index] = acc;
  } else if (kind == 1) {
    *pcross_out = acc;
  } else {
    step_out[2] = acc;
  }
}

void ensure_step_buffers(CbetWorkspace& ws, const std::size_t n_out) {
  if (ws.step_flags == nullptr) {
    stage_check(cudaMalloc(reinterpret_cast<void**>(&ws.step_flags), sizeof(int)),
                "cbet step flags alloc");
  }
  if (ws.step_flags_host == nullptr) {
    stage_check(cudaMallocHost(reinterpret_cast<void**>(&ws.step_flags_host), sizeof(int)),
                "cbet step flags pinned alloc");
  }
  if (n_out > ws.cap_step_out || ws.step_out == nullptr || ws.step_out_host == nullptr) {
    if (ws.step_out != nullptr) {
      static_cast<void>(cudaFree(ws.step_out));
      ws.step_out = nullptr;
    }
    if (ws.step_out_host != nullptr) {
      static_cast<void>(cudaFreeHost(ws.step_out_host));
      ws.step_out_host = nullptr;
    }
    const std::size_t capacity = std::max<std::size_t>(n_out, 8U);
    stage_check(cudaMalloc(reinterpret_cast<void**>(&ws.step_out), capacity * sizeof(double)),
                "cbet step outputs alloc");
    stage_check(cudaMallocHost(reinterpret_cast<void**>(&ws.step_out_host),
                               capacity * sizeof(double)),
                "cbet step outputs pinned alloc");
    ws.cap_step_out = capacity;
  }
}

std::size_t step_out_size(const int n_config_channels) {
  return 3U + 3U * static_cast<std::size_t>(std::max(n_config_channels, 0));
}

}  // namespace

void cbet_stage_cell_A_eff_device(CbetWorkspace& ws, const core::State& state,
                                  const double material_A,
                                  const std::vector<double>& material_A_list,
                                  cudaStream_t stream) {
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(n_cells > 0, "cbet A_eff: empty mesh");
  ensure_capacity(&ws.cell_A_eff, &ws.cap_cell_outputs, static_cast<std::size_t>(n_cells),
                  "cbet cell A_eff alloc");
  const int n_materials = static_cast<int>(material_A_list.size());
  const bool have_fractions =
      n_materials > 0 &&
      state.volFrac.size() == static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_materials);
  if (have_fractions) {
    const bool changed = ws.material_A_uploaded.size() != material_A_list.size() ||
                         std::memcmp(ws.material_A_uploaded.data(), material_A_list.data(),
                                     material_A_list.size() * sizeof(double)) != 0 ||
                         ws.material_A_device == nullptr;
    if (changed) {
      ensure_capacity(&ws.material_A_device, &ws.cap_material_A, material_A_list.size(),
                      "cbet material A alloc");
      ws.material_A_uploaded = material_A_list;
      stage_check(cudaMemcpyAsync(ws.material_A_device, ws.material_A_uploaded.data(),
                                  ws.material_A_uploaded.size() * sizeof(double),
                                  cudaMemcpyHostToDevice, stream),
                  "cbet material A H2D failed");
    }
  }
  cell_A_eff_kernel<<<blocks_for(n_cells), kThreads, 0, stream>>>(
      n_cells, n_materials, have_fractions ? state.volFrac.data() : nullptr,
      have_fractions ? ws.material_A_device : nullptr, material_A, ws.cell_A_eff);
  stage_check(cudaGetLastError(), "cbet A_eff launch failed");
}

void cbet_stage_cell_fields_device(CbetWorkspace& ws, const core::State& state,
                                   const core::Config::LaserConfig& laser, const double lambda0_cm,
                                   cudaStream_t stream) {
  const int n_cells = ws.n_cells;
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) == n_cells &&
                    static_cast<int>(state.zbar.size()) == n_cells &&
                    static_cast<int>(state.Te.size()) == n_cells &&
                    static_cast<int>(state.Ti.size()) == n_cells &&
                    static_cast<int>(state.vol.size()) == n_cells &&
                    static_cast<int>(state.v_r.size()) >= n_cells + 1,
                "cbet cell fields: state size mismatch");
  TENRYU_ASSERT(ws.cell_A_eff != nullptr && ws.cap_cell_outputs >= static_cast<std::size_t>(n_cells),
                "cbet cell fields: A_eff not staged");
  const CbetCellFieldConstants constants = cbet_cell_field_constants(lambda0_cm);
  CellFieldArgs args{};
  args.eV = constants.eV;
  args.c = constants.c;
  args.omega0 = constants.omega0;
  args.n_crit = constants.n_crit;
  args.lam_pref = constants.lam_pref;
  args.proton_mass = constants.proton_mass;
  args.eps_n = laser.absorption.eps_n;
  args.cutoff = laser.cbet.ne_frac_cutoff;
  const std::uint8_t* void_mask =
      state.cell_is_void.empty() ? nullptr : core::device_cell_is_void(state.cell_is_void);
  cell_fields_kernel<<<blocks_for(n_cells), kThreads, 0, stream>>>(
      n_cells, args, state.rho.data(), state.zbar.data(), ws.cell_A_eff, state.Te.data(),
      state.Ti.data(), state.v_r.data(), state.vol.data(), void_mask, ws.cell_chi_pref,
      ws.cell_c_a, ws.cell_u_r, ws.cell_k_bar, ws.cell_vol, ws.cell_mask);
  stage_check(cudaGetLastError(), "cbet cell fields launch failed");
}

void cbet_viz_fields_device(CbetWorkspace& ws, const int G, const int n_branches, const int n_bins,
                            const bool port_section, cudaStream_t stream) {
  TENRYU_ASSERT(G > 0, "CBET visualization requires a positive dQ state dimension");
  TENRYU_ASSERT(n_bins > 0 && n_branches > 0,
                "CBET visualization requires positive branch/bin dimensions");
  const int n_cells = ws.n_cells;
  TENRYU_ASSERT(ws.step_out != nullptr && ws.step_flags != nullptr,
                "CBET visualization before cbet_begin_step_outputs");
  // The four maps share one capacity (all [n_cells]); reallocate them together.
  const bool grow = ws.viz_gross == nullptr || ws.viz_net_inbound == nullptr ||
                    ws.viz_dq_abs_cell == nullptr || ws.viz_dq_max_cell == nullptr ||
                    ws.cap_viz < static_cast<std::size_t>(n_cells);
  if (grow) {
    for (double** p : {&ws.viz_gross, &ws.viz_net_inbound, &ws.viz_dq_abs_cell, &ws.viz_dq_max_cell}) {
      std::size_t capacity = 0;
      ensure_capacity(p, &capacity, static_cast<std::size_t>(n_cells), "cbet viz alloc");
    }
    ws.cap_viz = static_cast<std::size_t>(n_cells);
  }
  viz_fields_kernel<<<blocks_for(n_cells), kThreads, 0, stream>>>(
      n_cells, G, n_branches, n_bins, ws.dQ, port_section ? ws.iaw_cell : nullptr, ws.cell_vol,
      ws.viz_gross, ws.viz_net_inbound, ws.viz_dq_abs_cell, ws.viz_dq_max_cell, ws.step_flags);
  stage_check(cudaGetLastError(), "cbet viz launch failed");
  viz_audit_kernel<<<1, kAuditThreads, 0, stream>>>(n_cells, ws.viz_dq_abs_cell,
                                                    ws.viz_dq_max_cell, ws.step_out);
  stage_check(cudaGetLastError(), "cbet viz audit launch failed");
  ws.viz_valid = true;
}

void ps_outputs_device(CbetWorkspace& ws, const double* x_r, const int n_cells,
                       const int n_config_channels, const bool capture_on, cudaStream_t stream) {
  TENRYU_ASSERT(ws.ps_mode && ws.n_ports > 0, "port_section outputs: not a port_section workspace");
  TENRYU_ASSERT(n_config_channels >= 0, "port_section outputs: negative channel count");
  ensure_capacity(&ws.ps_outgoing, &ws.cap_ps_outgoing, static_cast<std::size_t>(ws.n_ports),
                  "cbet outgoing alloc");
  const std::size_t n_pcross =
      static_cast<std::size_t>(ws.n_ports) * static_cast<std::size_t>(n_config_channels);
  ensure_capacity(&ws.ps_capture_pcross, &ws.cap_ps_capture_pcross, std::max<std::size_t>(n_pcross, 1U),
                  "cbet capture pcross alloc");
  TENRYU_ASSERT(ws.step_out != nullptr && ws.step_flags != nullptr &&
                    ws.cap_step_out >= step_out_size(n_config_channels),
                "port_section outputs before cbet_begin_step_outputs");
  ws.ps_capture_config_channels = n_config_channels;
  outgoing_power_kernel<<<blocks_for(ws.n_ports), kThreads, 0, stream>>>(
      ws.n_ports, ws.n_rays_total, ws.cap_per_ray, ws.rec_count, ws.ray_rec_offset, ws.rec_w_ps,
      ws.ps_outgoing, ws.step_flags);
  stage_check(cudaGetLastError(), "cbet outgoing power launch failed");
  stage_check(cudaMemsetAsync(ws.ps_capture_pcross, 0, std::max<std::size_t>(n_pcross, 1U) * sizeof(double),
                              stream),
              "cbet capture pcross reset");
  if (capture_on) {
    TENRYU_ASSERT(ws.ps_n_channels > 0, "port_section capture: no capture channel");
    const long long n_rows = static_cast<long long>(ws.n_ports) * ws.n_rays_total * ws.ps_n_channels;
    ensure_capacity(&ws.ps_capture_row_config, &ws.cap_ps_capture_row_config,
                    static_cast<std::size_t>(std::max<long long>(n_rows, 1)),
                    "cbet capture row config alloc");
    capture_classify_kernel<<<blocks_for(n_rows), kThreads, 0, stream>>>(
        n_rows, ws.ps_n_channels, n_config_channels, n_cells, ws.ps_capture_stage,
        ws.ps_capture_order, ws.ps_capture_row_config, ws.step_flags);
    stage_check(cudaGetLastError(), "cbet capture classify launch failed");
    const int n_sums = n_config_channels + ws.n_ports * n_config_channels + 1;
    capture_sum_kernel<<<n_sums, kCaptureThreads, 0, stream>>>(
        ws.n_ports, ws.n_rays_total, ws.ps_n_channels, n_config_channels, ws.ps_capture_stage,
        ws.ps_capture_row_config, ws.ps_one_minus_eta, x_r, ws.ps_capture_pcross, ws.step_out);
    stage_check(cudaGetLastError(), "cbet capture sum launch failed");
  }
  ws.ps_outputs_valid = true;
}

void cbet_begin_step_outputs(CbetWorkspace& ws, const int n_config_channels, cudaStream_t stream) {
  TENRYU_ASSERT(n_config_channels >= 0, "cbet step outputs: negative channel count");
  ensure_step_buffers(ws, step_out_size(n_config_channels));
  stage_check(cudaMemsetAsync(ws.step_flags, 0, sizeof(int), stream), "cbet step flags reset");
  stage_check(cudaMemsetAsync(ws.step_out, 0, step_out_size(n_config_channels) * sizeof(double), stream),
              "cbet step outputs reset");
}

CbetStepReadback cbet_read_step_outputs(CbetWorkspace& ws, const int n_config_channels,
                                        cudaStream_t stream) {
  const std::size_t n_out = step_out_size(n_config_channels);
  TENRYU_ASSERT(ws.step_out != nullptr && ws.step_flags != nullptr && ws.cap_step_out >= n_out,
                "cbet step readback before the step stages");
  stage_check(cudaMemcpyAsync(ws.step_out_host, ws.step_out, n_out * sizeof(double),
                              cudaMemcpyDeviceToHost, stream),
              "cbet step outputs D2H failed");
  stage_check(cudaMemcpyAsync(ws.step_flags_host, ws.step_flags, sizeof(int),
                              cudaMemcpyDeviceToHost, stream),
              "cbet step flags D2H failed");
  stage_check(cudaStreamSynchronize(stream), "cbet step readback sync failed");
  CbetStepReadback out;
  out.flags = *ws.step_flags_host;
  out.dq_abs_max = ws.step_out_host[0];
  out.dq_abs_sum = ws.step_out_host[1];
  out.capture_banked_power = ws.step_out_host[2];
  const std::size_t K = static_cast<std::size_t>(std::max(n_config_channels, 0));
  out.capture_sum_P.assign(ws.step_out_host + 3, ws.step_out_host + 3 + K);
  out.capture_sum_Pr.assign(ws.step_out_host + 3 + K, ws.step_out_host + 3 + 2 * K);
  out.capture_sum_Pmu.assign(ws.step_out_host + 3 + 2 * K, ws.step_out_host + 3 + 3 * K);
  return out;
}

void cbet_sync_output_fields(core::State& state, const CbetWorkspace& ws) {
  if (ws.viz_valid) {
    const std::size_t n = static_cast<std::size_t>(ws.n_cells);
    state.cbet_gross_exchange.resize(n);
    state.cbet_net_to_inbound.resize(n);
    stage_check(cudaMemcpy(state.cbet_gross_exchange.data(), ws.viz_gross, n * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "cbet gross exchange D2H failed");
    stage_check(cudaMemcpy(state.cbet_net_to_inbound.data(), ws.viz_net_inbound, n * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "cbet net exchange D2H failed");
  }
  if (ws.ps_outputs_valid) {
    state.ps_port_outgoing_power.resize(static_cast<std::size_t>(ws.n_ports));
    stage_check(cudaMemcpy(state.ps_port_outgoing_power.data(), ws.ps_outgoing,
                           state.ps_port_outgoing_power.size() * sizeof(double),
                           cudaMemcpyDeviceToHost),
                "cbet outgoing power D2H failed");
    state.ps_port_capture_pcross.resize(static_cast<std::size_t>(ws.n_ports) *
                                        static_cast<std::size_t>(ws.ps_capture_config_channels));
    if (!state.ps_port_capture_pcross.empty()) {
      stage_check(cudaMemcpy(state.ps_port_capture_pcross.data(), ws.ps_capture_pcross,
                             state.ps_port_capture_pcross.size() * sizeof(double),
                             cudaMemcpyDeviceToHost),
                  "cbet capture pcross D2H failed");
    }
  }
}

}  // namespace tenryu::laser
