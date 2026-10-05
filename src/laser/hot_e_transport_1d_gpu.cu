#include "laser/hot_e_transport_1d_gpu.cuh"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "core/constants.hpp"
#include "core/device_pack.hpp"
#include "core/error.hpp"
#include "core/device_ordered_sum.cuh"
#include "laser/hot_electron_1d_gpu.cuh"

// Compiled with -fmad=false (src/laser/CMakeLists.txt): the arithmetic repeats laser.cu laser_step's
// hot-electron block and hot_electron_1d.cpp (reduce_captures, build_band_nodes,
// deposit_hot_electrons_radial_1d) and hot_electron_1d_gpu.cu (deposit_hot_electrons_cone_1d_device)
// operation by operation.

namespace tenryu::laser::hot_e_transport_1d {
namespace {

using hot_electron::ConeChordJob;
using hot_electron::GroupSpec;
using hot_electron::HotEChannelSpec;
using hot_electron::kAxisBins;
using hot_electron::kMuxEpsilon;

constexpr int kThreads = 256;

inline void check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void ensure(T** ptr, std::size_t* capacity, const std::size_t needed, const char* message) {
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
  check(cudaMalloc(reinterpret_cast<void**>(ptr), needed * sizeof(T)), message);
  *capacity = needed;
}

int blocks_for(const long long n) {
  return static_cast<int>(std::max<long long>(1, (n + kThreads - 1) / kThreads));
}

__device__ inline double host_max(const double a, const double b) { return (a < b) ? b : a; }
__device__ inline double host_min(const double a, const double b) { return (b < a) ? b : a; }

// ---------------------------------------------------------------------------------------------
// captures

__device__ inline int block_exclusive_prefix(const int value, int* sh, int* total) {
  return core::device_ordered::block_exclusive_prefix<kThreads>(value, sh, total);
}

constexpr int kCollectPerThread = 8;
constexpr int kCollectChunk = kThreads * kCollectPerThread;

// One block per config channel: the staged rows in the order beam, ray, capture channel, a chunk
// at a time. The captures of the capture channels with eta_eff > 0 are written to the channel's
// list in that order (each thread a run of entries, the runs' counts combined in order); the
// model sums P and r P are added in that order by one thread from a compacted copy in shared
// memory.
__global__ void collect_kernel(const double* __restrict__ stage, const int n_beams,
                               const int rays_per_beam, const int n_k,
                               const int* __restrict__ config_index,
                               const double* __restrict__ eta_eff, const int n_config,
                               const int model_sums, const int list_capacity,
                               double* __restrict__ list_r, double* __restrict__ list_mu,
                               double* __restrict__ list_P, int* __restrict__ list_count,
                               double* __restrict__ sums /* [n_config][2] */) {
  __shared__ int sh_scan[kThreads];
  __shared__ double sh_P[kCollectChunk];
  __shared__ double sh_rP[kCollectChunk];
  const int ch = blockIdx.x;
  if (ch >= n_config) {
    return;
  }
  const int t = threadIdx.x;
  const long long n_entries = static_cast<long long>(n_beams) * rays_per_beam * n_k;
  double* r_out = list_r + static_cast<long long>(ch) * list_capacity;
  double* mu_out = list_mu + static_cast<long long>(ch) * list_capacity;
  double* P_out = list_P + static_cast<long long>(ch) * list_capacity;
  double sum_P = 0.0;   // thread 0's
  double sum_Pr = 0.0;
  int count = 0;        // the list's length so far (the same in every thread)
  for (long long chunk = 0; chunk < n_entries; chunk += kCollectChunk) {
    const long long begin = chunk + static_cast<long long>(t) * kCollectPerThread;
    int n_list = 0;
    int n_model = 0;
    for (int j = 0; j < kCollectPerThread; ++j) {
      const long long e = begin + j;
      if (e >= n_entries) {
        break;
      }
      const int k = static_cast<int>(e % n_k);
      if (config_index[k] != ch) {
        continue;
      }
      const double* cap = stage + e * 4;
      if (cap[0] > 0.5 && cap[3] > 0.0) {
        ++n_model;
        if (eta_eff[k] > 0.0) {
          ++n_list;
        }
      }
    }
    int list_total = 0;
    const int list_offset = block_exclusive_prefix(n_list, sh_scan, &list_total);
    int model_total = 0;
    const int model_offset = block_exclusive_prefix(n_model, sh_scan, &model_total);
    int li = count + list_offset;
    int mi = model_offset;
    for (int j = 0; j < kCollectPerThread; ++j) {
      const long long e = begin + j;
      if (e >= n_entries) {
        break;
      }
      const int k = static_cast<int>(e % n_k);
      if (config_index[k] != ch) {
        continue;
      }
      const double* cap = stage + e * 4;
      if (cap[0] > 0.5 && cap[3] > 0.0) {
        sh_P[mi] = cap[3];
        sh_rP[mi] = cap[1] * cap[3];
        ++mi;
        if (eta_eff[k] > 0.0) {
          r_out[li] = cap[1];
          mu_out[li] = cap[2];
          P_out[li] = eta_eff[k] * cap[3];
          ++li;
        }
      }
    }
    __syncthreads();
    if (t == 0 && model_sums != 0) {
      for (int m = 0; m < model_total; ++m) {
        sum_P += sh_P[m];
        sum_Pr += sh_rP[m];
      }
    }
    count += list_total;
    __syncthreads();
  }
  if (t == 0) {
    list_count[ch] = count;
    sums[2 * ch] = sum_P;
    sums[2 * ch + 1] = sum_Pr;
  }
}

// hot_electron::reduce_captures, in three launches. Here, one thread per capture: its (cell,
// axis-angle bin) and whether it lies outside the mesh (captures without power: no bin), the
// clamped axis cosine kept for the bin's moment, and the bin marked as having captures.
__global__ void bin_ids_kernel(const int n_config, const int list_capacity,
                               const double* __restrict__ list_r, double* __restrict__ list_mu,
                               const double* __restrict__ list_P,
                               const int* __restrict__ list_count,
                               const double* __restrict__ r_nodes, const int n_cells,
                               int* __restrict__ cap_bin, std::uint8_t* __restrict__ cap_outside,
                               std::uint8_t* __restrict__ bin_mark /* [n_config][n_cells][16] */) {
  const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ch = static_cast<int>(id / list_capacity);
  const int i = static_cast<int>(id % list_capacity);
  if (ch >= n_config || i >= list_count[ch]) {
    return;
  }
  const long long at = static_cast<long long>(ch) * list_capacity + i;
  const double P_hot = list_P[at];
  if (!(P_hot > 0.0)) {
    cap_bin[at] = -1;
    cap_outside[at] = 0U;
    return;
  }
  const double r_s = list_r[at];
  cap_outside[at] = (r_s < r_nodes[0] || r_s > r_nodes[n_cells]) ? 1U : 0U;
  // c = n_cells - 1; while (c > 0 && r_s < r_nodes[c]) --c;  (the nodes increase)
  int c = n_cells - 1;
  if (c > 0 && r_s < r_nodes[c]) {
    int lo = 0;   // r_s < r_nodes[c] fails at lo (or lo == 0)
    int hi = c;   // r_s < r_nodes[hi] holds
    while (hi - lo > 1) {
      const int mid = lo + (hi - lo) / 2;
      if (r_s < r_nodes[mid]) {
        hi = mid;
      } else {
        lo = mid;
      }
    }
    c = lo;
  }
  double mu = list_mu[at];
  if (mu < -1.0) {
    mu = -1.0;
  }
  if (mu > 1.0) {
    mu = 1.0;
  }
  list_mu[at] = mu;
  int b = static_cast<int>((mu + 1.0) * 0.5 * kAxisBins);
  if (b >= kAxisBins) {
    b = kAxisBins - 1;
  }
  if (b < 0) {
    b = 0;
  }
  const int bin = c * kAxisBins + b;
  cap_bin[at] = bin;
  bin_mark[static_cast<long long>(ch) * n_cells * kAxisBins + bin] = 1U;
}

// One block per channel: the bins with captures in (cell, bin) order, the sources' order (each
// thread a run of bins, the runs' counts combined in order); and, by the first thread, the captures
// outside the mesh counted and their power summed in the captures' order (the host's warning).
__global__ void bin_list_kernel(const int n_config, const int n_cells, const int list_capacity,
                                const std::uint8_t* __restrict__ bin_mark,
                                const int* __restrict__ list_count,
                                const double* __restrict__ list_P,
                                const std::uint8_t* __restrict__ cap_outside,
                                const int source_capacity, int* __restrict__ src_bin,
                                int* __restrict__ src_count, long long* __restrict__ out_of_domain,
                                double* __restrict__ out_of_domain_power) {
  __shared__ int sh_scan[kThreads];
  const int ch = blockIdx.x;
  if (ch >= n_config) {
    return;
  }
  const int t = threadIdx.x;
  const int n_bins = n_cells * kAxisBins;
  const int chunk = (n_bins + kThreads - 1) / kThreads;
  const int begin = ::min(n_bins, t * chunk);
  const int end = ::min(n_bins, begin + chunk);
  const std::uint8_t* mark = bin_mark + static_cast<long long>(ch) * n_bins;
  int local = 0;
  for (int i = begin; i < end; ++i) {
    local += (mark[i] != 0U) ? 1 : 0;
  }
  int total = 0;
  int offset = block_exclusive_prefix(local, sh_scan, &total);
  int* out = src_bin + static_cast<long long>(ch) * source_capacity;
  for (int i = begin; i < end; ++i) {
    if (mark[i] != 0U) {
      out[offset] = i;
      ++offset;
    }
  }
  if (t == 0) {
    src_count[ch] = total;
    long long outside = 0;
    double outside_power = 0.0;
    const long long base = static_cast<long long>(ch) * list_capacity;
    const int count = list_count[ch];
    for (int i = 0; i < count; ++i) {
      if (cap_outside[base + i] != 0U) {
        ++outside;
        outside_power += list_P[base + i];
      }
    }
    out_of_domain[ch] = outside;
    out_of_domain_power[ch] = outside_power;
  }
}

// One thread per (channel, source): the bin's sums over its captures in the captures' order (P,
// P mu, P r, as the host's bins accumulated them), then the source {cell, r = P r / P,
// mu = P mu / P, P}.
__global__ void source_sums_kernel(const int n_config, const int list_capacity,
                                   const double* __restrict__ list_r,
                                   const double* __restrict__ list_mu,
                                   const double* __restrict__ list_P,
                                   const int* __restrict__ list_count,
                                   const int* __restrict__ cap_bin, const int source_capacity,
                                   const int* __restrict__ src_bin,
                                   const int* __restrict__ src_count, int* __restrict__ src_cell,
                                   double* __restrict__ src_r, double* __restrict__ src_mu,
                                   double* __restrict__ src_P) {
  const long long id = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int ch = static_cast<int>(id / source_capacity);
  const int s = static_cast<int>(id % source_capacity);
  if (ch >= n_config || s >= src_count[ch]) {
    return;
  }
  const long long out = static_cast<long long>(ch) * source_capacity + s;
  const int bin = src_bin[out];
  const long long base = static_cast<long long>(ch) * list_capacity;
  const int count = list_count[ch];
  double P = 0.0;
  double P_mu = 0.0;
  double P_r = 0.0;
  for (int i = 0; i < count; ++i) {
    if (cap_bin[base + i] != bin) {
      continue;
    }
    const double P_hot = list_P[base + i];
    P += P_hot;
    P_mu += P_hot * list_mu[base + i];
    P_r += P_hot * list_r[base + i];
  }
  src_cell[out] = bin / kAxisBins;
  src_r[out] = P_r / P;
  src_mu[out] = P_mu / P;
  src_P[out] = P;
}

// ---------------------------------------------------------------------------------------------
// cone chords

// hot_electron::build_band_nodes of a channel, made on the host once per spec: mode 0 the
// Gauss-Legendre band, 1 a ring at mu_lo, 2 the single node {mu_axis, 1}.
struct BandTable {
  int mode;
  int n_mu;
  int n_phi;
  const double* mu;      // [n_mu] (mode 1: one entry, mu_lo)
  const double* smu;     // [n_mu] sqrt(max(1 - mu^2, 0))
  const double* weight;  // [n_mu] wmu / n_phi (mode 1: 1 / n_phi)
  const double* cos_phi; // [n_phi]
};

struct ConeChannelState {
  double P_hot;
  double Pr;
  double P_deposited;
  double P_escaped;
  int n_sources;
  int n_jobs;
  int active;
  int done;  // finished without chords (no jobs or no groups)
};

// One thread: the sources' power and radius moment, then the chords of every source and node in
// order (the in-plane chords of a slab deposit in their source cell here), as the host built them.
__global__ void cone_jobs_kernel(const int ch, const int source_capacity,
                                 const int* __restrict__ src_cell, const double* __restrict__ src_r,
                                 const double* __restrict__ src_mu, const double* __restrict__ src_P,
                                 const int* __restrict__ src_count, const BandTable band,
                                 const int planar, const int n_groups, const int n_cells,
                                 const std::uint8_t* __restrict__ cell_is_void,
                                 double* __restrict__ dep, ConeChordJob* __restrict__ jobs,
                                 ConeChannelState* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const long long base = static_cast<long long>(ch) * source_capacity;
  const int n_sources = src_count[ch];
  ConeChannelState st{};
  for (int s = 0; s < n_sources; ++s) {
    st.P_hot += src_P[base + s];
    st.Pr += src_P[base + s] * src_r[base + s];
  }
  st.n_sources = n_sources;
  if (n_sources == 0 || !(st.P_hot > 0.0)) {
    *out = st;
    return;
  }
  st.active = 1;
  int n_jobs = 0;
  for (int s = 0; s < n_sources; ++s) {
    const double P_source = src_P[base + s];
    const double mu_axis = src_mu[base + s];
    const int source_cell = src_cell[base + s];
    const double sa = ::sqrt(host_max(1.0 - mu_axis * mu_axis, 0.0));
    const int n_mu_nodes = (band.mode == 2) ? 1 : band.n_mu;
    const int n_phi_nodes = (band.mode == 2) ? 1 : band.n_phi;
    for (int i = 0; i < n_mu_nodes; ++i) {
      for (int jp = 0; jp < n_phi_nodes; ++jp) {
        double mu_dir;
        double weight;
        if (band.mode == 2) {
          mu_dir = mu_axis;
          weight = 1.0;
        } else {
          mu_dir = band.mu[i] * mu_axis + band.smu[i] * sa * band.cos_phi[jp];
          weight = band.weight[i];
        }
        const double P_chord = P_source * weight;
        if (!(P_chord > 0.0)) {
          continue;
        }
        if (planar != 0 && ::fabs(mu_dir) < kMuxEpsilon) {
          // an in-plane chord thermalizes in its source cell
          if (source_cell >= 0 && source_cell < n_cells &&
              !(cell_is_void != nullptr && cell_is_void[source_cell] != 0U)) {
            dep[source_cell] += P_chord;
            st.P_deposited += P_chord;
          } else {
            st.P_escaped += P_chord;
          }
          continue;
        }
        jobs[n_jobs] = ConeChordJob{src_r[base + s], mu_dir, P_chord};
        ++n_jobs;
      }
    }
  }
  st.n_jobs = n_jobs;
  st.done = (n_jobs == 0 || n_groups == 0) ? 1 : 0;
  *out = st;
}

constexpr int kSumPerThread = 8;

// One block: the chords' power into the cells (each cell once) and the sums the host made: the
// deposited power over the cells in cell order and the escaped power over the rows in row order
// (one thread, from the nonzero entries compacted in order), the cap hits (integers); the
// conservation residual.
__global__ void cone_finish_kernel(const int n_cells, const int n_rows,
                                   const double* __restrict__ out_cell,
                                   const double* __restrict__ escaped, const int* __restrict__ caps,
                                   double* __restrict__ dep, const ConeChannelState* __restrict__ st,
                                   double* __restrict__ result /* [5] */) {
  __shared__ double sh_values[kThreads * kSumPerThread];
  __shared__ int sh_scan[kThreads];
  const int t = threadIdx.x;
  const ConeChannelState state = *st;
  double P_deposited = state.P_deposited;
  double P_escaped = state.P_escaped;
  int cap_local = 0;
  if (state.done == 0) {
    for (int c = t; c < n_cells; c += kThreads) {
      dep[c] += out_cell[c];
    }
    P_deposited = core::device_ordered::block_ordered_sum_nonzero<kThreads, kSumPerThread>(
        out_cell, n_cells, P_deposited, sh_values, sh_scan);
    P_escaped = core::device_ordered::block_ordered_sum_nonzero<kThreads, kSumPerThread>(
        escaped, n_rows, P_escaped, sh_values, sh_scan);
    for (int row = t; row < n_rows; row += kThreads) {
      cap_local += caps[row];
    }
  }
  int cap_hits = 0;
  static_cast<void>(block_exclusive_prefix(cap_local, sh_scan, &cap_hits));
  if (t != 0) {
    return;
  }
  const double P_hot = state.P_hot;
  result[0] = P_deposited;
  result[1] = P_escaped;
  result[2] = ::fabs(P_deposited + P_escaped - P_hot) / host_max(P_hot, 1.0e-300);
  result[3] = static_cast<double>(cap_hits);
  result[4] = state.Pr / P_hot;
}

// ---------------------------------------------------------------------------------------------
// radial march (hot_electron::deposit_hot_electrons_radial_1d)

struct RadialSetup {
  double P_hot;
  double Pr;
  int n_sources;
  int outer_cell;
  double outer_r;
  int inner_cell;  // the first real cell (the deposit_residual inner boundary)
};

__global__ void radial_setup_kernel(const int ch, const int source_capacity,
                                    const int* __restrict__ src_cell,
                                    const double* __restrict__ src_r,
                                    const double* __restrict__ src_P,
                                    const int* __restrict__ src_count, const int n_cells,
                                    const std::uint8_t* __restrict__ cell_is_void,
                                    RadialSetup* __restrict__ out) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const long long base = static_cast<long long>(ch) * source_capacity;
  RadialSetup st{};
  st.n_sources = src_count[ch];
  st.outer_cell = -1;
  int outer = -1;
  for (int s = 0; s < st.n_sources; ++s) {
    st.P_hot += src_P[base + s];
    st.Pr += src_P[base + s] * src_r[base + s];
    if (outer < 0 || src_r[base + s] > src_r[base + outer] ||
        (src_r[base + s] == src_r[base + outer] && src_cell[base + s] > src_cell[base + outer])) {
      outer = s;
    }
  }
  if (outer >= 0) {
    st.outer_cell = src_cell[base + outer];
    st.outer_r = src_r[base + outer];
  }
  st.inner_cell = -1;
  for (int c = 0; c < n_cells; ++c) {
    if (!(cell_is_void != nullptr && cell_is_void[c] != 0U)) {
      st.inner_cell = c;
      break;
    }
  }
  *out = st;
}

// One block per group, its first thread marching: the inward march from the outermost source
// through the cells (row [group][cell] the power deposited in each), the energy left at the inner
// end and the cap hits. The march of a group is a serial chain of double-precision work; a block
// each spreads the groups over the multiprocessors and their double-precision units (one warp of
// groups would share the units of one).
__global__ void radial_march_kernel(const GroupSpec* __restrict__ groups, const int n_groups,
                                    const RadialSetup* __restrict__ setup,
                                    const double* __restrict__ rho, const double* __restrict__ zbar,
                                    const double* __restrict__ A_eff,
                                    const double* __restrict__ Te_eV,
                                    const std::uint8_t* __restrict__ cell_is_void,
                                    const double* __restrict__ r_nodes, const int n_cells,
                                    const double T_h_erg, double* __restrict__ rows,
                                    double* __restrict__ E_left, double* __restrict__ Ndot_out,
                                    int* __restrict__ cap_hits) {
  const int g = blockIdx.x;
  if (g >= n_groups || threadIdx.x != 0) {
    return;
  }
  double* row = rows + static_cast<long long>(g) * n_cells;
  for (int c = 0; c < n_cells; ++c) {
    row[c] = 0.0;
  }
  E_left[g] = 0.0;
  Ndot_out[g] = 0.0;
  cap_hits[g] = 0;
  const GroupSpec group = groups[g];
  if (!(group.weight > 0.0) || !(group.E_rep > 0.0)) {
    return;
  }
  const RadialSetup st = *setup;
  const double Ndot = st.P_hot * group.weight / group.E_rep;
  double E = group.E_rep;
  int caps = 0;
  for (int c = st.outer_cell; c >= 0; --c) {
    if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
      continue;
    }
    const double r_hi = (c == st.outer_cell)
                            ? host_min(host_max(st.outer_r, r_nodes[c]), r_nodes[c + 1])
                            : r_nodes[c + 1];
    const double dSigma = rho[c] * (r_hi - r_nodes[c]);
    const double Te_erg = host_max(Te_eV[c], 0.0) * core::constants::eV_to_erg;
    double ne = 0.0;
    if ((A_eff[c] > 0.0) && (rho[c] > 0.0)) {
      ne = rho[c] * host_max(zbar[c], 0.0) / (A_eff[c] * core::constants::proton_mass);
    }
    const double E_floor = host_max(2.0 * Te_erg, 1.0e-3 * T_h_erg);
    const double rho_c = rho[c];
    const auto stopping = [=](const double Ee) -> double {
      return hot_electron::stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
    };
    const double E_out = hot_electron::march_cell(E, dSigma, stopping, E_floor,
                                                  hot_electron::kSubstepEnergyFraction,
                                                  hot_electron::kMaxSubstepsPerCell, &caps);
    row[c] = Ndot * (E - E_out);
    E = E_out;
    if (!(E > 0.0)) {
      break;
    }
  }
  E_left[g] = E;
  Ndot_out[g] = Ndot;
  cap_hits[g] = caps;
}

// One thread: the rows into the cells and the sums in the host's order (per group the cells from
// the source inward, then the energy left at the inner end).
__global__ void radial_finish_kernel(const GroupSpec* __restrict__ groups, const int n_groups,
                                     const RadialSetup* __restrict__ setup,
                                     const std::uint8_t* __restrict__ cell_is_void,
                                     const int n_cells, const double* __restrict__ rows,
                                     const double* __restrict__ E_left,
                                     const double* __restrict__ Ndot, const int* __restrict__ caps,
                                     const int inner_escape, double* __restrict__ dep,
                                     double* __restrict__ result /* [7] */) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  const RadialSetup st = *setup;
  double P_deposited = 0.0;
  double P_escaped = 0.0;
  double P_residual_inner = 0.0;
  int cap_hits = 0;
  for (int g = 0; g < n_groups; ++g) {
    if (!(groups[g].weight > 0.0) || !(groups[g].E_rep > 0.0)) {
      continue;
    }
    const double* row = rows + static_cast<long long>(g) * n_cells;
    for (int c = st.outer_cell; c >= 0; --c) {
      if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
        continue;
      }
      // the cells the march did not reach hold zero (the host stopped adding there)
      dep[c] += row[c];
      P_deposited += row[c];
    }
    cap_hits += caps[g];
    if (E_left[g] > 0.0) {
      const double residual = Ndot[g] * E_left[g];
      if (inner_escape != 0) {
        P_escaped += residual;
      } else if (st.inner_cell >= 0) {
        dep[st.inner_cell] += residual;
        P_deposited += residual;
        P_residual_inner += residual;
      } else {
        result[6] = 1.0;  // the host asserts: deposit_residual needs a real cell
      }
    }
  }
  result[0] = P_deposited;
  result[1] = P_escaped;
  result[2] = ::fabs(P_deposited + P_escaped - st.P_hot) / host_max(st.P_hot, 1.0e-300);
  result[3] = static_cast<double>(cap_hits);
  result[4] = st.Pr / st.P_hot;
  result[5] = P_residual_inner;
}

__global__ void min_kernel(const int n, const double* __restrict__ values,
                           double* __restrict__ out) {
  __shared__ double sh[kThreads];
  const int t = threadIdx.x;
  double m = CUDART_INF;
  for (int i = t; i < n; i += blockDim.x) {
    m = host_min(m, values[i]);  // no NaN among the values
  }
  sh[t] = m;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (t < stride) {
      sh[t] = host_min(sh[t], sh[t + stride]);
    }
    __syncthreads();
  }
  if (t == 0) {
    *out = sh[0];
  }
}

// ---------------------------------------------------------------------------------------------
// diagnostics

__global__ void diagnostics_kernel(const int n_cells, const double* __restrict__ power,
                                   const double* __restrict__ vol, const double* __restrict__ rho,
                                   const double* __restrict__ ee, const double dt,
                                   const double limit, double* __restrict__ Q,
                                   double* __restrict__ eps_cum, double* __restrict__ candidate) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  candidate[c] = CUDART_INF;
  const double P_cell = power[c];
  if (!(P_cell > 0.0)) {
    Q[c] = 0.0;
    return;
  }
  const double vol_c = vol[c];
  const double rho_c = rho[c];
  Q[c] = 0.0;  // the host copy was zeroed at the start of the laser step
  if (vol_c > 0.0) {
    Q[c] = P_cell / vol_c;
    if (rho_c > 0.0) {
      eps_cum[c] += P_cell * dt / (rho_c * vol_c);
      const double e_cell = rho_c * vol_c * host_max(ee[c], 0.0);
      if (e_cell > 0.0) {
        const double cand = limit * e_cell / P_cell;
        if (cand < CUDART_INF) {  // the host's `if (cand < dt_lim)` never takes a NaN
          candidate[c] = cand;
        }
      }
    }
  }
}

}  // namespace

namespace {

// The groups of one channel's spec and, for the cone, its band quadrature without the source's axis
// (hot_electron::build_band_nodes), made on the host once per spec and kept on the device.
struct SpecTables {
  bool built = false;
  bool cone = false;
  hot_electron::HotEChannelSpec spec{};
  int n_groups = 0;
  int mode = 0;
  int band_n_mu = 0;
  GroupSpec* groups = nullptr;  // device [n_groups]
  double* band = nullptr;       // device [3 band_n_mu + n_phi]: mu, smu, weight, cos_phi
  std::size_t cap_groups = 0;
  std::size_t cap_band = 0;

  bool matches(const HotEChannelSpec& s, const bool for_cone) const {
    return built && cone == for_cone && spec.T_hot_erg == s.T_hot_erg &&
           spec.n_energy_groups == s.n_energy_groups && spec.E_min_over_Th == s.E_min_over_Th &&
           spec.E_max_over_Th == s.E_max_over_Th && spec.mu_lo == s.mu_lo &&
           spec.mu_hi == s.mu_hi && spec.n_mu == s.n_mu && spec.n_phi == s.n_phi;
  }

  BandTable band_table() const {
    BandTable t{};
    t.mode = mode;
    t.n_mu = band_n_mu;
    t.n_phi = spec.n_phi;
    t.mu = band;
    t.smu = band + band_n_mu;
    t.weight = band + 2 * band_n_mu;
    t.cos_phi = band + 3 * band_n_mu;
    return t;
  }
};

void build_spec_tables(SpecTables& t, const HotEChannelSpec& spec, const bool cone) {
  t.built = false;
  t.cone = cone;
  t.spec = spec;
  const std::vector<GroupSpec> groups = hot_electron::build_groups(
      spec.T_hot_erg, spec.n_energy_groups, spec.E_min_over_Th, spec.E_max_over_Th);
  t.n_groups = static_cast<int>(groups.size());
  std::vector<double> band;
  t.mode = 0;
  t.band_n_mu = 0;
  if (cone) {
    // hot_electron::build_band_nodes, the parts that do not depend on the source's axis
    double mu_hi = spec.mu_hi;
    double mu_lo = spec.mu_lo;
    if (mu_hi > 1.0) {
      mu_hi = 1.0;
    }
    if (mu_lo < -1.0) {
      mu_lo = -1.0;
    }
    TENRYU_ASSERT(mu_lo <= mu_hi, "hot_electron band nodes require mu_lo <= mu_hi");
    std::vector<double> mu;
    std::vector<double> smu;
    std::vector<double> weight;
    const int n_phi = spec.n_phi;
    if (!(mu_lo < mu_hi)) {
      if (!(mu_lo < 1.0)) {
        t.mode = 2;  // the single node {mu_axis, 1}
      } else {
        t.mode = 1;  // a ring at mu_lo
        mu.push_back(mu_lo);
        smu.push_back(std::sqrt(std::max(1.0 - mu_lo * mu_lo, 0.0)));
        weight.push_back(1.0 / static_cast<double>(n_phi));
      }
    } else {
      std::vector<double> gx;
      std::vector<double> gw;
      hot_electron::gauss_legendre(spec.n_mu, gx, gw);
      const double half_span = 0.5 * (mu_hi - mu_lo);
      const double mid = 0.5 * (mu_hi + mu_lo);
      for (int i = 0; i < spec.n_mu; ++i) {
        const double m = mid + half_span * gx[static_cast<std::size_t>(i)];
        const double wmu = gw[static_cast<std::size_t>(i)] * 0.5;  // GL weights sum to 2
        mu.push_back(m);
        smu.push_back(std::sqrt(std::max(1.0 - m * m, 0.0)));
        weight.push_back(wmu / static_cast<double>(n_phi));
      }
    }
    t.band_n_mu = static_cast<int>(mu.size());
    band.insert(band.end(), mu.begin(), mu.end());
    band.insert(band.end(), smu.begin(), smu.end());
    band.insert(band.end(), weight.begin(), weight.end());
    if (t.mode != 2) {
      for (int jp = 0; jp < n_phi; ++jp) {
        const double phi = 2.0 * hot_electron::kPi * (static_cast<double>(jp) + 0.5) /
                           static_cast<double>(n_phi);
        band.push_back(std::cos(phi));
      }
    }
  }
  ensure(&t.groups, &t.cap_groups, std::max<std::size_t>(groups.size(), 1U), "hot-e groups alloc");
  ensure(&t.band, &t.cap_band, std::max<std::size_t>(band.size(), 1U), "hot-e band alloc");
  if (!groups.empty()) {
    check(cudaMemcpy(t.groups, groups.data(), groups.size() * sizeof(GroupSpec),
                     cudaMemcpyHostToDevice),
          "hot-e groups H2D");
  }
  if (!band.empty()) {
    check(cudaMemcpy(t.band, band.data(), band.size() * sizeof(double), cudaMemcpyHostToDevice),
          "hot-e band H2D");
  }
  t.built = true;
}

}  // namespace

struct Workspace::Impl {
  int n_config = 0;
  // the trace's staging
  int n_k = 0;
  int n_beams = 0;
  int rays = 0;
  double* stage = nullptr;
  std::size_t cap_stage = 0;
  // the capture lists [n_config][list_capacity], their counts and the model sums [n_config][2]
  int list_capacity = 0;
  double* list_r = nullptr;
  double* list_mu = nullptr;
  double* list_P = nullptr;
  std::size_t cap_list_r = 0;
  std::size_t cap_list_mu = 0;
  std::size_t cap_list_P = 0;
  int* list_count = nullptr;
  std::size_t cap_list_count = 0;
  double* sums = nullptr;
  std::size_t cap_sums = 0;
  int* config_index = nullptr;
  double* eta_eff = nullptr;
  std::size_t cap_config_index = 0;
  std::size_t cap_eta_eff = 0;
  bool trace_collected = false;
  std::vector<std::vector<hot_electron::RayCapture>> host_captures;
  std::vector<int> counts_host;  // the list counts of the last read_capture_state
  // the bins, the sources and the chords
  int n_cells = 0;
  int* cap_bin = nullptr;                 // [n_config][list_capacity]
  std::uint8_t* cap_outside = nullptr;    // [n_config][list_capacity]
  std::uint8_t* bin_mark = nullptr;       // [n_config][n_cells][kAxisBins]
  int* src_bin = nullptr;                 // [n_config][source_capacity]
  std::size_t cap_cap_bin = 0;
  std::size_t cap_cap_outside = 0;
  std::size_t cap_bin_mark = 0;
  std::size_t cap_src_bin = 0;
  long long* out_of_domain = nullptr;
  double* out_of_domain_power = nullptr;
  std::size_t cap_ood = 0;
  std::size_t cap_ood_power = 0;
  int source_capacity = 0;
  int* src_cell = nullptr;
  double* src_r = nullptr;
  double* src_mu = nullptr;
  double* src_P = nullptr;
  int* src_count = nullptr;
  std::size_t cap_src_cell = 0;
  std::size_t cap_src_r = 0;
  std::size_t cap_src_mu = 0;
  std::size_t cap_src_P = 0;
  std::size_t cap_src_count = 0;
  ConeChordJob* jobs = nullptr;
  std::size_t cap_jobs = 0;
  double* rows = nullptr;
  std::size_t cap_rows = 0;
  double* escaped = nullptr;
  std::size_t cap_escaped = 0;
  int* caps = nullptr;
  std::size_t cap_caps = 0;
  double* out_cell = nullptr;
  std::size_t cap_out_cell = 0;
  ConeChannelState* cone_state = nullptr;
  RadialSetup* radial_setup = nullptr;
  double* E_left = nullptr;
  double* Ndot = nullptr;
  std::size_t cap_E_left = 0;
  std::size_t cap_Ndot = 0;
  double* result = nullptr;  // [8]
  std::vector<SpecTables> spec_tables;
  // the power per cell and the diagnostics
  double* power = nullptr;
  std::size_t cap_power = 0;
  double* Q = nullptr;
  double* eps = nullptr;
  double* candidate = nullptr;
  double* dt_min = nullptr;
  std::size_t cap_Q = 0;
  std::size_t cap_eps = 0;
  std::size_t cap_candidate = 0;
  int eps_cells = 0;
  std::vector<double> eps_host_mirror;  // eps_cum as it last came back to the host
  unsigned char* pinned = nullptr;
  std::size_t cap_pinned = 0;

  unsigned char* pinned_bytes(const std::size_t bytes) {
    if (bytes > cap_pinned || pinned == nullptr) {
      if (pinned != nullptr) {
        // no copy may still read the old buffer
        check(cudaDeviceSynchronize(), "hot-e pinned regrow sync");
        static_cast<void>(cudaFreeHost(pinned));
        pinned = nullptr;
      }
      cap_pinned = 0;
      check(cudaMallocHost(reinterpret_cast<void**>(&pinned), bytes), "hot-e pinned alloc");
      cap_pinned = bytes;
    }
    return pinned;
  }

  void ensure_lists(const int capacity) {
    list_capacity = capacity;
    const std::size_t n_list =
        static_cast<std::size_t>(std::max(n_config, 1)) * static_cast<std::size_t>(capacity);
    ensure(&list_r, &cap_list_r, n_list, "hot-e capture list alloc");
    ensure(&list_mu, &cap_list_mu, n_list, "hot-e capture list alloc");
    ensure(&list_P, &cap_list_P, n_list, "hot-e capture list alloc");
  }

  ~Impl() {
    void* ptrs[] = {stage,  list_r,       list_mu,    list_P,        list_count,
                    sums,   config_index, eta_eff,    cap_bin,       out_of_domain,
                    out_of_domain_power,  src_cell,   src_r,         src_mu,
                    src_P,  src_count,    jobs,       rows,          escaped,
                    caps,   out_cell,     cone_state, radial_setup,  E_left,
                    Ndot,   result,       power,      Q,             eps,
                    candidate,            dt_min,     cap_outside,   bin_mark,
                    src_bin};
    for (void* ptr : ptrs) {
      if (ptr != nullptr) {
        static_cast<void>(cudaFree(ptr));
      }
    }
    for (SpecTables& t : spec_tables) {
      if (t.groups != nullptr) {
        static_cast<void>(cudaFree(t.groups));
      }
      if (t.band != nullptr) {
        static_cast<void>(cudaFree(t.band));
      }
    }
    if (pinned != nullptr) {
      static_cast<void>(cudaFreeHost(pinned));
    }
  }
};

Workspace::Workspace() : impl_(new Impl) {}
Workspace::~Workspace() { delete impl_; }

void begin_step(Workspace& ws, const int n_config_channels, cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(n_config_channels >= 0, "hot-e transport: negative channel count");
  w.n_config = n_config_channels;
  w.n_k = 0;
  w.n_beams = 0;
  w.rays = 0;
  if (w.list_capacity < 16 || w.list_r == nullptr) {
    w.ensure_lists(16);
  } else {
    w.ensure_lists(w.list_capacity);  // the lists of more channels
  }
  const std::size_t n_ch = static_cast<std::size_t>(std::max(n_config_channels, 1));
  ensure(&w.list_count, &w.cap_list_count, n_ch, "hot-e capture count alloc");
  ensure(&w.sums, &w.cap_sums, 2U * n_ch, "hot-e sums alloc");
  check(cudaMemsetAsync(w.list_count, 0, n_ch * sizeof(int), stream), "hot-e count reset");
  check(cudaMemsetAsync(w.sums, 0, 2U * n_ch * sizeof(double), stream), "hot-e sums reset");
  w.trace_collected = false;
  w.host_captures.assign(static_cast<std::size_t>(n_config_channels), {});
  w.counts_host.assign(static_cast<std::size_t>(n_config_channels), 0);
}

void begin_trace(Workspace& ws, const int n_capture_channels, const int n_beams,
                 const int rays_per_beam, cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(n_capture_channels >= 0 && n_beams >= 0 && rays_per_beam >= 0,
                "hot-e transport: negative staging sizes");
  w.n_k = n_capture_channels;
  w.n_beams = n_beams;
  w.rays = rays_per_beam;
  const std::size_t n_stage = static_cast<std::size_t>(n_beams) *
                              static_cast<std::size_t>(rays_per_beam) *
                              static_cast<std::size_t>(n_capture_channels) * 4U;
  ensure(&w.stage, &w.cap_stage, std::max<std::size_t>(n_stage, 1U), "hot-e stage alloc");
  if (n_stage > 0) {
    check(cudaMemsetAsync(w.stage, 0, n_stage * sizeof(double), stream), "hot-e stage reset");
  }
  // a channel's list holds at most every staged capture
  const long long most = static_cast<long long>(n_beams) * rays_per_beam * n_capture_channels;
  TENRYU_ASSERT(most <= (1LL << 30), "hot-e transport: capture staging too large");
  const int capacity = std::max(static_cast<int>(most), 16);
  if (capacity > w.list_capacity) {
    w.ensure_lists(capacity);
  }
}

void stage_beam(Workspace& ws, const int beam, const double* rows, const int n_rays,
                cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(beam >= 0 && beam < w.n_beams && n_rays >= 0 && n_rays <= w.rays,
                "hot-e transport: beam staging out of range");
  const std::size_t row = static_cast<std::size_t>(w.rays) * static_cast<std::size_t>(w.n_k) * 4U;
  const std::size_t count =
      static_cast<std::size_t>(n_rays) * static_cast<std::size_t>(w.n_k) * 4U;
  if (count == 0) {
    return;
  }
  check(cudaMemcpyAsync(w.stage + static_cast<std::size_t>(beam) * row, rows,
                        count * sizeof(double), cudaMemcpyDeviceToDevice, stream),
        "hot-e capture stage D2D");
}

void replay_beam(Workspace& ws, const int anchor_beam, const int beam, cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(anchor_beam >= 0 && anchor_beam < w.n_beams && beam >= 0 && beam < w.n_beams,
                "hot-e transport: beam replay out of range");
  const std::size_t row = static_cast<std::size_t>(w.rays) * static_cast<std::size_t>(w.n_k) * 4U;
  if (row == 0 || beam == anchor_beam) {
    return;
  }
  check(cudaMemcpyAsync(w.stage + static_cast<std::size_t>(beam) * row,
                        w.stage + static_cast<std::size_t>(anchor_beam) * row,
                        row * sizeof(double), cudaMemcpyDeviceToDevice, stream),
        "hot-e capture replay D2D");
}

void collect_trace_captures(Workspace& ws, const std::vector<int>& config_index,
                            const std::vector<double>& eta_eff, const bool model_sums,
                            cudaStream_t stream) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(static_cast<int>(config_index.size()) == w.n_k &&
                    static_cast<int>(eta_eff.size()) == w.n_k,
                "hot-e transport: capture channel tables size mismatch");
  for (const int ci : config_index) {
    TENRYU_ASSERT(ci >= 0 && ci < w.n_config, "hot-e transport: config index out of range");
  }
  w.trace_collected = true;
  if (w.n_k == 0 || w.n_config == 0) {
    return;
  }
  ensure(&w.config_index, &w.cap_config_index, static_cast<std::size_t>(w.n_k),
         "hot-e config index alloc");
  ensure(&w.eta_eff, &w.cap_eta_eff, static_cast<std::size_t>(w.n_k), "hot-e eta alloc");
  // staged through pinned memory: the async copies read it after this returns
  const std::size_t bytes = static_cast<std::size_t>(w.n_k) * (sizeof(int) + sizeof(double));
  unsigned char* pinned = w.pinned_bytes(std::max<std::size_t>(bytes, 128U));
  check(cudaStreamSynchronize(stream), "hot-e pinned reuse sync");
  std::memcpy(pinned, config_index.data(), config_index.size() * sizeof(int));
  std::memcpy(pinned + config_index.size() * sizeof(int), eta_eff.data(),
              eta_eff.size() * sizeof(double));
  check(cudaMemcpyAsync(w.config_index, pinned, config_index.size() * sizeof(int),
                        cudaMemcpyHostToDevice, stream),
        "hot-e config index H2D");
  check(cudaMemcpyAsync(w.eta_eff, pinned + config_index.size() * sizeof(int),
                        eta_eff.size() * sizeof(double), cudaMemcpyHostToDevice, stream),
        "hot-e eta H2D");
  collect_kernel<<<w.n_config, kThreads, 0, stream>>>(
      w.stage, w.n_beams, w.rays, w.n_k, w.config_index, w.eta_eff, w.n_config,
      model_sums ? 1 : 0, w.list_capacity, w.list_r, w.list_mu, w.list_P, w.list_count, w.sums);
  check(cudaGetLastError(), "hot-e collect launch");
}

void add_host_captures(Workspace& ws, const int config_channel,
                       const std::vector<hot_electron::RayCapture>& captures) {
  auto& w = *ws.impl();
  TENRYU_ASSERT(config_channel >= 0 && config_channel < w.n_config,
                "hot-e transport: host capture channel out of range");
  auto& list = w.host_captures[static_cast<std::size_t>(config_channel)];
  list.insert(list.end(), captures.begin(), captures.end());
}

void read_capture_state(Workspace& ws, std::vector<double>* sum_P, std::vector<double>* sum_Pr,
                        bool* any_captures, cudaStream_t stream) {
  auto& w = *ws.impl();
  const std::size_t n_ch = static_cast<std::size_t>(w.n_config);
  std::size_t host_most = 0;
  for (const auto& list : w.host_captures) {
    host_most = std::max(host_most, list.size());
  }
  if (host_most > 0) {
    TENRYU_ASSERT(!w.trace_collected,
                  "hot-e transport: trace and host captures in the same step");
    if (host_most > static_cast<std::size_t>(w.list_capacity)) {
      w.ensure_lists(static_cast<int>(host_most));
    }
    std::vector<double> r;
    std::vector<double> mu;
    std::vector<double> P;
    for (std::size_t ch = 0; ch < n_ch; ++ch) {
      const auto& list = w.host_captures[ch];
      w.counts_host[ch] = static_cast<int>(list.size());
      if (list.empty()) {
        continue;
      }
      r.resize(list.size());
      mu.resize(list.size());
      P.resize(list.size());
      for (std::size_t i = 0; i < list.size(); ++i) {
        r[i] = list[i].r_s;
        mu[i] = list[i].mu_axis;
        P[i] = list[i].P_hot;
      }
      const std::size_t base = ch * static_cast<std::size_t>(w.list_capacity);
      check(cudaMemcpyAsync(w.list_r + base, r.data(), r.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream),
            "hot-e host captures H2D");
      check(cudaMemcpyAsync(w.list_mu + base, mu.data(), mu.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream),
            "hot-e host captures H2D");
      check(cudaMemcpyAsync(w.list_P + base, P.data(), P.size() * sizeof(double),
                            cudaMemcpyHostToDevice, stream),
            "hot-e host captures H2D");
      // pageable sources: wait before the vectors are reused
      check(cudaStreamSynchronize(stream), "hot-e host captures sync");
    }
    check(cudaMemcpyAsync(w.list_count, w.counts_host.data(), n_ch * sizeof(int),
                          cudaMemcpyHostToDevice, stream),
          "hot-e host capture counts H2D");
    check(cudaStreamSynchronize(stream), "hot-e host capture counts sync");
  }
  sum_P->assign(n_ch, 0.0);
  sum_Pr->assign(n_ch, 0.0);
  if (n_ch > 0) {
    const std::size_t sums_bytes = n_ch * 2U * sizeof(double);
    unsigned char* pinned = w.pinned_bytes(sums_bytes + n_ch * sizeof(int));
    check(cudaMemcpyAsync(pinned, w.sums, sums_bytes, cudaMemcpyDeviceToHost, stream),
          "hot-e sums D2H");
    check(cudaMemcpyAsync(pinned + sums_bytes, w.list_count, n_ch * sizeof(int),
                          cudaMemcpyDeviceToHost, stream),
          "hot-e counts D2H");
    check(cudaStreamSynchronize(stream), "hot-e capture state sync");
    std::vector<double> sums(2U * n_ch);
    std::memcpy(sums.data(), pinned, sums_bytes);
    std::memcpy(w.counts_host.data(), pinned + sums_bytes, n_ch * sizeof(int));
    for (std::size_t ch = 0; ch < n_ch; ++ch) {
      (*sum_P)[ch] = sums[2U * ch];
      (*sum_Pr)[ch] = sums[2U * ch + 1U];
    }
  }
  bool any = false;
  for (const int count : w.counts_host) {
    any = any || count > 0;
  }
  *any_captures = any;
}

std::vector<ChannelResult> transport(Workspace& ws, const core::State& state,
                                     const std::vector<HotEChannelSpec>& specs,
                                     const int geometry_code, const bool cone,
                                     const double* A_eff_cone, const double* A_eff_radial,
                                     cudaStream_t stream) {
  auto& w = *ws.impl();
  const int n_cells = static_cast<int>(state.rho.size());
  TENRYU_ASSERT(n_cells > 0 && static_cast<int>(state.x_r.size()) == n_cells + 1,
                "hot-e transport: state size mismatch");
  TENRYU_ASSERT(static_cast<int>(specs.size()) == w.n_config,
                "hot-e transport: one spec per config channel");
  TENRYU_ASSERT(state.cell_is_void.size() == static_cast<std::size_t>(n_cells),
                "hot_electron transport cell_is_void size mismatch");
  TENRYU_ASSERT(cone ? (A_eff_cone != nullptr) : (A_eff_radial != nullptr),
                "hot-e transport: A_eff not given");
  w.n_cells = n_cells;
  const std::size_t nn = static_cast<std::size_t>(n_cells);
  ensure(&w.power, &w.cap_power, nn, "hot-e power alloc");
  check(cudaMemsetAsync(w.power, 0, nn * sizeof(double), stream), "hot-e power reset");
  std::vector<ChannelResult> results(static_cast<std::size_t>(w.n_config));
  if (w.n_config == 0) {
    return results;
  }
  const std::size_t n_ch = static_cast<std::size_t>(w.n_config);
  const std::size_t n_marks = n_ch * nn * static_cast<std::size_t>(kAxisBins);
  ensure(&w.bin_mark, &w.cap_bin_mark, n_marks, "hot-e bin marks alloc");
  check(cudaMemsetAsync(w.bin_mark, 0, n_marks, stream), "hot-e bin marks reset");
  const std::size_t n_list = n_ch * static_cast<std::size_t>(w.list_capacity);
  ensure(&w.cap_bin, &w.cap_cap_bin, n_list, "hot-e capture bins alloc");
  ensure(&w.cap_outside, &w.cap_cap_outside, n_list, "hot-e capture bins alloc");
  ensure(&w.out_of_domain, &w.cap_ood, n_ch, "hot-e domain alloc");
  ensure(&w.out_of_domain_power, &w.cap_ood_power, n_ch, "hot-e domain alloc");
  const double* r_nodes = state.x_r.data();
  bin_ids_kernel<<<blocks_for(static_cast<long long>(n_list)), kThreads, 0, stream>>>(
      w.n_config, w.list_capacity, w.list_r, w.list_mu, w.list_P, w.list_count, r_nodes, n_cells,
      w.cap_bin, w.cap_outside, w.bin_mark);
  check(cudaGetLastError(), "hot-e bin ids launch");
  w.source_capacity = n_cells * kAxisBins;
  const std::size_t n_src = n_ch * static_cast<std::size_t>(w.source_capacity);
  ensure(&w.src_bin, &w.cap_src_bin, n_src, "hot-e sources alloc");
  ensure(&w.src_cell, &w.cap_src_cell, n_src, "hot-e sources alloc");
  ensure(&w.src_r, &w.cap_src_r, n_src, "hot-e sources alloc");
  ensure(&w.src_mu, &w.cap_src_mu, n_src, "hot-e sources alloc");
  ensure(&w.src_P, &w.cap_src_P, n_src, "hot-e sources alloc");
  ensure(&w.src_count, &w.cap_src_count, n_ch, "hot-e sources alloc");
  bin_list_kernel<<<w.n_config, kThreads, 0, stream>>>(
      w.n_config, n_cells, w.list_capacity, w.bin_mark, w.list_count, w.list_P, w.cap_outside,
      w.source_capacity, w.src_bin, w.src_count, w.out_of_domain, w.out_of_domain_power);
  check(cudaGetLastError(), "hot-e bin list launch");
  source_sums_kernel<<<blocks_for(static_cast<long long>(n_src)), kThreads, 0, stream>>>(
      w.n_config, w.list_capacity, w.list_r, w.list_mu, w.list_P, w.list_count, w.cap_bin,
      w.source_capacity, w.src_bin, w.src_count, w.src_cell, w.src_r, w.src_mu, w.src_P);
  check(cudaGetLastError(), "hot-e source sums launch");
  // the source counts and the clamped captures, for the sizes and the warnings
  const std::size_t off_ood = n_ch * sizeof(int);
  const std::size_t off_ood_power = off_ood + n_ch * sizeof(long long);
  unsigned char* pinned =
      w.pinned_bytes(std::max<std::size_t>(off_ood_power + n_ch * sizeof(double), 128U));
  check(cudaMemcpyAsync(pinned, w.src_count, n_ch * sizeof(int), cudaMemcpyDeviceToHost, stream),
        "hot-e source counts D2H");
  check(cudaMemcpyAsync(pinned + off_ood, w.out_of_domain, n_ch * sizeof(long long),
                        cudaMemcpyDeviceToHost, stream),
        "hot-e domain D2H");
  check(cudaMemcpyAsync(pinned + off_ood_power, w.out_of_domain_power, n_ch * sizeof(double),
                        cudaMemcpyDeviceToHost, stream),
        "hot-e domain D2H");
  check(cudaStreamSynchronize(stream), "hot-e source counts sync");
  std::vector<int> n_sources(n_ch);
  std::vector<long long> outside(n_ch);
  std::vector<double> outside_power(n_ch);
  std::memcpy(n_sources.data(), pinned, n_ch * sizeof(int));
  std::memcpy(outside.data(), pinned + off_ood, n_ch * sizeof(long long));
  std::memcpy(outside_power.data(), pinned + off_ood_power, n_ch * sizeof(double));

  const std::uint8_t* mask = core::device_cell_is_void(state.cell_is_void);
  if (w.spec_tables.size() < n_ch) {
    w.spec_tables.resize(n_ch);
  }
  if (w.result == nullptr) {
    check(cudaMalloc(reinterpret_cast<void**>(&w.result), 8 * sizeof(double)), "hot-e result alloc");
  }
  if (w.cone_state == nullptr) {
    check(cudaMalloc(reinterpret_cast<void**>(&w.cone_state), sizeof(ConeChannelState)),
          "hot-e cone state alloc");
  }
  if (w.radial_setup == nullptr) {
    check(cudaMalloc(reinterpret_cast<void**>(&w.radial_setup), sizeof(RadialSetup)),
          "hot-e radial setup alloc");
  }
  for (std::size_t ch = 0; ch < n_ch; ++ch) {
    ChannelResult& res = results[ch];
    if (w.counts_host[ch] == 0) {
      continue;  // no captures in this channel
    }
    if (outside[ch] > 0) {
      core::log_warning("hot_electron reduce_captures: " + std::to_string(outside[ch]) +
                        " capture(s) outside the hydro mesh (" +
                        std::to_string(outside_power[ch]) +
                        " erg/s) clamped into the boundary cell");
    }
    if (n_sources[ch] == 0) {
      continue;
    }
    const HotEChannelSpec& spec = specs[ch];
    if (cone) {
      TENRYU_ASSERT(geometry_code != 1,
                    "hot_electron cone with cylindrical geometry (validation gap)");
    }
    SpecTables& tables = w.spec_tables[ch];
    if (!tables.matches(spec, cone)) {
      build_spec_tables(tables, spec, cone);
    }
    if (cone) {
      const bool planar = (geometry_code == 2);
      const std::size_t nodes_per_source =
          (tables.mode == 2)
              ? 1U
              : static_cast<std::size_t>(tables.band_n_mu) * static_cast<std::size_t>(spec.n_phi);
      ensure(&w.jobs, &w.cap_jobs,
             std::max<std::size_t>(static_cast<std::size_t>(n_sources[ch]) * nodes_per_source, 1U),
             "hot-e jobs alloc");
      cone_jobs_kernel<<<1, 1, 0, stream>>>(
          static_cast<int>(ch), w.source_capacity, w.src_cell, w.src_r, w.src_mu, w.src_P,
          w.src_count, tables.band_table(), planar ? 1 : 0, tables.n_groups, n_cells, mask,
          w.power, w.jobs, w.cone_state);
      check(cudaGetLastError(), "hot-e cone jobs launch");
      unsigned char* st_bytes = w.pinned_bytes(128U);
      check(cudaMemcpyAsync(st_bytes, w.cone_state, sizeof(ConeChannelState),
                            cudaMemcpyDeviceToHost, stream),
            "hot-e cone state D2H");
      check(cudaStreamSynchronize(stream), "hot-e cone state sync");
      ConeChannelState st{};
      std::memcpy(&st, st_bytes, sizeof(ConeChannelState));
      if (st.active == 0) {
        continue;
      }
      int n_rows = 0;
      if (st.done == 0) {
        n_rows = st.n_jobs * tables.n_groups;
        const std::size_t rows_bytes = static_cast<std::size_t>(n_rows) * nn * sizeof(double);
        TENRYU_ASSERT(rows_bytes <= (4ULL << 30),
                      "hot_electron device row scratch exceeds 4 GiB "
                      "(n_jobs*n_groups*n_cells); reduce hot_electron n_mu/n_phi/"
                      "n_energy_groups or the capture spread");
        ensure(&w.rows, &w.cap_rows, static_cast<std::size_t>(n_rows) * nn, "hot-e rows alloc");
        ensure(&w.escaped, &w.cap_escaped, static_cast<std::size_t>(n_rows), "hot-e rows alloc");
        ensure(&w.caps, &w.cap_caps, static_cast<std::size_t>(n_rows), "hot-e rows alloc");
        ensure(&w.out_cell, &w.cap_out_cell, nn, "hot-e out alloc");
        hot_electron::cone_chords_device(w.jobs, st.n_jobs, tables.groups, tables.n_groups,
                                         state.rho.data(), state.zbar.data(), A_eff_cone,
                                         state.Te.data(), mask, r_nodes, n_cells + 1, planar,
                                         spec.T_hot_erg, w.rows, w.escaped, w.caps, w.out_cell,
                                         stream);
      }
      cone_finish_kernel<<<1, kThreads, 0, stream>>>(n_cells, n_rows, w.out_cell, w.escaped,
                                                     w.caps, w.power, w.cone_state, w.result);
      check(cudaGetLastError(), "hot-e cone finish launch");
      double* out = reinterpret_cast<double*>(w.pinned_bytes(128U));
      check(cudaMemcpyAsync(out, w.result, 5 * sizeof(double), cudaMemcpyDeviceToHost, stream),
            "hot-e cone result D2H");
      check(cudaStreamSynchronize(stream), "hot-e cone result sync");
      res.active = true;
      res.n_sources = st.n_sources;
      res.P_hot = st.P_hot;
      res.P_deposited = out[0];
      res.P_escaped = out[1];
      res.conservation_resid = out[2];
      res.substep_cap_hits = static_cast<int>(out[3]);
      res.r_source_mean = out[4];
      if (res.conservation_resid > 1.0e-8) {
        core::log_warning("hot_electron cone conservation check failed");
      }
    } else {
      radial_setup_kernel<<<1, 1, 0, stream>>>(static_cast<int>(ch), w.source_capacity,
                                               w.src_cell, w.src_r, w.src_P, w.src_count,
                                               n_cells, mask, w.radial_setup);
      check(cudaGetLastError(), "hot-e radial setup launch");
      unsigned char* st_bytes = w.pinned_bytes(128U);
      check(cudaMemcpyAsync(st_bytes, w.radial_setup, sizeof(RadialSetup),
                            cudaMemcpyDeviceToHost, stream),
            "hot-e radial setup D2H");
      check(cudaStreamSynchronize(stream), "hot-e radial setup sync");
      RadialSetup setup{};
      std::memcpy(&setup, st_bytes, sizeof(RadialSetup));
      if (setup.n_sources == 0 || !(setup.P_hot > 0.0)) {
        continue;
      }
      TENRYU_ASSERT(setup.outer_cell >= 0 && setup.outer_cell < n_cells,
                    "hot_electron radial source cell out of range");
      const int n_groups = tables.n_groups;
      const std::size_t n_g = static_cast<std::size_t>(std::max(n_groups, 1));
      ensure(&w.rows, &w.cap_rows, n_g * nn, "hot-e radial rows alloc");
      ensure(&w.E_left, &w.cap_E_left, n_g, "hot-e radial alloc");
      ensure(&w.Ndot, &w.cap_Ndot, n_g, "hot-e radial alloc");
      ensure(&w.caps, &w.cap_caps, n_g, "hot-e radial alloc");
      check(cudaMemsetAsync(w.result, 0, 8 * sizeof(double), stream), "hot-e result reset");
      if (n_groups > 0) {
        radial_march_kernel<<<n_groups, 32, 0, stream>>>(
            tables.groups, n_groups, w.radial_setup, state.rho.data(), state.zbar.data(),
            A_eff_radial, state.Te.data(), mask, r_nodes, n_cells, spec.T_hot_erg, w.rows,
            w.E_left, w.Ndot, w.caps);
        check(cudaGetLastError(), "hot-e radial march launch");
      }
      radial_finish_kernel<<<1, 1, 0, stream>>>(tables.groups, n_groups, w.radial_setup, mask,
                                                n_cells, w.rows, w.E_left, w.Ndot, w.caps,
                                                spec.inner_escape ? 1 : 0, w.power, w.result);
      check(cudaGetLastError(), "hot-e radial finish launch");
      double* out = reinterpret_cast<double*>(w.pinned_bytes(128U));
      check(cudaMemcpyAsync(out, w.result, 7 * sizeof(double), cudaMemcpyDeviceToHost, stream),
            "hot-e radial result D2H");
      check(cudaStreamSynchronize(stream), "hot-e radial result sync");
      TENRYU_ASSERT(out[6] == 0.0,
                    "hot_electron radial deposit_residual requires a non-void cell");
      res.active = true;
      res.n_sources = setup.n_sources;
      res.P_hot = setup.P_hot;
      res.P_deposited = out[0];
      res.P_escaped = out[1];
      res.conservation_resid = out[2];
      res.substep_cap_hits = static_cast<int>(out[3]);
      res.r_source_mean = out[4];
      res.P_residual_inner = out[5];
      if (res.conservation_resid > 1.0e-8) {
        core::log_warning("hot_electron radial conservation check failed");
      }
    }
  }
  return results;
}

const double* power_cell(const Workspace& ws) { return ws.impl()->power; }

double diagnostics(Workspace& ws, core::State& state, const double dt,
                   const double explicit_source_limit, cudaStream_t stream) {
  auto& w = *ws.impl();
  const int n_cells = w.n_cells;
  TENRYU_ASSERT(n_cells == static_cast<int>(state.rho.size()) && w.power != nullptr,
                "hot-e diagnostics before a transport");
  const std::size_t nn = static_cast<std::size_t>(n_cells);
  if (state.hot_e_Q_host.size() != nn) {
    state.hot_e_Q_host.assign(nn, 0.0);
  }
  if (state.hot_e_eps_cum_host.size() != nn) {
    state.hot_e_eps_cum_host.assign(nn, 0.0);
  }
  ensure(&w.Q, &w.cap_Q, nn, "hot-e Q alloc");
  ensure(&w.candidate, &w.cap_candidate, nn, "hot-e candidate alloc");
  const bool eps_resized = (w.eps_cells != n_cells) || nn > w.cap_eps || w.eps == nullptr;
  ensure(&w.eps, &w.cap_eps, nn, "hot-e eps alloc");
  if (w.dt_min == nullptr) {
    check(cudaMalloc(reinterpret_cast<void**>(&w.dt_min), sizeof(double)), "hot-e dt alloc");
  }
  if (eps_resized || state.hot_e_eps_cum_host != w.eps_host_mirror) {
    // a restart, a retried or rolled-back step, or a new mesh changed the host copy
    check(cudaMemcpyAsync(w.eps, state.hot_e_eps_cum_host.data(), nn * sizeof(double),
                          cudaMemcpyHostToDevice, stream),
          "hot-e eps H2D");
  }
  w.eps_cells = n_cells;
  diagnostics_kernel<<<blocks_for(n_cells), kThreads, 0, stream>>>(
      n_cells, w.power, state.vol.data(), state.rho.data(), state.ee.data(), dt,
      explicit_source_limit, w.Q, w.eps, w.candidate);
  check(cudaGetLastError(), "hot-e diagnostics launch");
  min_kernel<<<1, kThreads, 0, stream>>>(n_cells, w.candidate, w.dt_min);
  check(cudaGetLastError(), "hot-e dt limit launch");
  double* out = reinterpret_cast<double*>(w.pinned_bytes(128U));
  check(cudaMemcpyAsync(out, w.dt_min, sizeof(double), cudaMemcpyDeviceToHost, stream),
        "hot-e dt limit D2H");
  check(cudaMemcpyAsync(state.hot_e_Q_host.data(), w.Q, nn * sizeof(double),
                        cudaMemcpyDeviceToHost, stream),
        "hot-e Q D2H");
  check(cudaMemcpyAsync(state.hot_e_eps_cum_host.data(), w.eps, nn * sizeof(double),
                        cudaMemcpyDeviceToHost, stream),
        "hot-e eps D2H");
  check(cudaStreamSynchronize(stream), "hot-e diagnostics sync");
  w.eps_host_mirror = state.hot_e_eps_cum_host;
  return *out;
}

}  // namespace tenryu::laser::hot_e_transport_1d
