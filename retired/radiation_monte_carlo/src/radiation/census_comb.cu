#include "radiation/census_comb.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include "core/error.hpp"
#include "core/rng/philox_cpu.hpp"
#include "radiation/particle_pool.cuh"

namespace tenryu::radiation {
namespace {

constexpr int kThreadsPerBlock = 256;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void d2h(std::vector<T>& host, const T* ptr, const int n, const char* msg) {
  host.resize(static_cast<std::size_t>(n));
  if (n > 0) {
    cuda_check(cudaMemcpy(host.data(),
                          ptr,
                          sizeof(T) * static_cast<std::size_t>(n),
                          cudaMemcpyDeviceToHost),
               msg);
  }
}

template <typename T>
void h2d(T* ptr, const std::vector<T>& host, const char* msg) {
  if (!host.empty()) {
    cuda_check(cudaMemcpy(ptr,
                          host.data(),
                          sizeof(T) * host.size(),
                          cudaMemcpyHostToDevice),
               msg);
  }
}

__device__ inline double atomic_add_double(double* address, const double value) {
#if __CUDA_ARCH__ >= 600
  return atomicAdd(address, value);
#else
  auto* address_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_ull;
  unsigned long long int assumed = 0ULL;
  do {
    assumed = old;
    old = atomicCAS(address_ull,
                    assumed,
                    __double_as_longlong(
                        value + __longlong_as_double(static_cast<long long>(assumed))));
  } while (assumed != old);
  return __longlong_as_double(static_cast<long long>(old));
#endif
}

template <typename T>
void d2d_copy(T* dst, const T* src, const int n, const char* msg) {
  if (n > 0) {
    cuda_check(cudaMemcpy(dst,
                          src,
                          sizeof(T) * static_cast<std::size_t>(n),
                          cudaMemcpyDeviceToDevice),
               msg);
  }
}

struct PhotonPoolReadView {
  const double* pos_r = nullptr;
  const double* pos_z = nullptr;
  const double* dir_r = nullptr;
  const double* dir_z = nullptr;
  const double* dir_phi = nullptr;
  const double* energy = nullptr;
  const double* weight = nullptr;
  const double* time_remain = nullptr;
  const double* birth_energy = nullptr;
  const std::int8_t* sign = nullptr;
  const std::uint64_t* global_id = nullptr;
  const std::uint32_t* rng_counter = nullptr;
  const std::int32_t* cell_id = nullptr;
  const std::uint16_t* group_id = nullptr;
  const std::uint8_t* mode = nullptr;
  const std::uint8_t* alive = nullptr;
};

struct PhotonPoolWriteView {
  double* pos_r = nullptr;
  double* pos_z = nullptr;
  double* dir_r = nullptr;
  double* dir_z = nullptr;
  double* dir_phi = nullptr;
  double* energy = nullptr;
  double* weight = nullptr;
  double* time_remain = nullptr;
  double* birth_energy = nullptr;
  std::int8_t* sign = nullptr;
  std::uint64_t* global_id = nullptr;
  std::uint32_t* rng_counter = nullptr;
  std::int32_t* cell_id = nullptr;
  std::uint16_t* group_id = nullptr;
  std::uint8_t* mode = nullptr;
  std::uint8_t* alive = nullptr;
};

PhotonPoolReadView make_read_view(const PhotonPool& pool) {
  PhotonPoolReadView view;
  view.pos_r = pool.pos_r;
  view.pos_z = pool.pos_z;
  view.dir_r = pool.dir_r;
  view.dir_z = pool.dir_z;
  view.dir_phi = pool.dir_phi;
  view.energy = pool.energy;
  view.weight = pool.weight;
  view.time_remain = pool.time_remain;
  view.birth_energy = pool.birth_energy;
  view.sign = pool.sign;
  view.global_id = pool.global_id;
  view.rng_counter = pool.rng_counter;
  view.cell_id = pool.cell_id;
  view.group_id = pool.group_id;
  view.mode = pool.mode;
  view.alive = pool.alive;
  return view;
}

PhotonPoolWriteView make_write_view(PhotonPool& pool) {
  PhotonPoolWriteView view;
  view.pos_r = pool.pos_r;
  view.pos_z = pool.pos_z;
  view.dir_r = pool.dir_r;
  view.dir_z = pool.dir_z;
  view.dir_phi = pool.dir_phi;
  view.energy = pool.energy;
  view.weight = pool.weight;
  view.time_remain = pool.time_remain;
  view.birth_energy = pool.birth_energy;
  view.sign = pool.sign;
  view.global_id = pool.global_id;
  view.rng_counter = pool.rng_counter;
  view.cell_id = pool.cell_id;
  view.group_id = pool.group_id;
  view.mode = pool.mode;
  view.alive = pool.alive;
  return view;
}

struct KahanSum {
  double sum = 0.0;
  double c = 0.0;

  void add(const double x) {
    const double y = x - c;
    const double t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
};

struct SortedEntry {
  std::uint64_t key = 0;
  int src = 0;
};

struct BinInfo {
  std::uint64_t key = 0;
  int begin = 0;
  int end = 0;
  int n = 0;
  double E_bin = 0.0;
  double score = 0.0;
  int target = 0;
  double frac = 0.0;
};

[[nodiscard]] inline double clamp_energy(const double E) {
  return (std::isfinite(E) && E > 0.0) ? E : 0.0;
}

void add_numerical_loss(double* const d_loss, const double loss_abs) {
  if (d_loss == nullptr || !(loss_abs > 0.0) || !std::isfinite(loss_abs)) {
    return;
  }
  double host_loss = 0.0;
  cuda_check(cudaMemcpy(&host_loss, d_loss, sizeof(host_loss), cudaMemcpyDeviceToHost),
             "census_comb copy E_numerical_loss (D2H) failed");
  host_loss += loss_abs;
  cuda_check(cudaMemcpy(d_loss, &host_loss, sizeof(host_loss), cudaMemcpyHostToDevice),
             "census_comb copy E_numerical_loss (H2D) failed");
}

[[nodiscard]] double draw_u0(const std::uint32_t seed,
                             const int step,
                             const std::uint64_t counter,
                             const int m_res) {
  TENRYU_ASSERT(m_res > 0, "census_comb draw_u0 requires m_res > 0");
  const std::uint64_t key =
      static_cast<std::uint64_t>(seed) ^
      static_cast<std::uint64_t>(static_cast<std::uint32_t>(step));
  core::rng::PhiloxCpu rng(key, 0ULL, counter, 0U);
  double u = rng.uniform();
  const double one_minus = std::nextafter(1.0, 0.0);
  if (!(u > 0.0) || !std::isfinite(u)) {
    u = one_minus;
  }
  if (u >= 1.0) {
    u = one_minus;
  }
  return u / static_cast<double>(m_res);
}

std::vector<int> systematic_residual_counts(const std::vector<double>& residual,
                                            const int m_res,
                                            const std::uint32_t seed,
                                            const int step,
                                            const std::uint64_t counter) {
  const int n = static_cast<int>(residual.size());
  std::vector<int> inc(static_cast<std::size_t>(n), 0);
  if (n == 0 || m_res <= 0) {
    return inc;
  }

  KahanSum sum_residual;
  for (const double r : residual) {
    if (r > 0.0 && std::isfinite(r)) {
      sum_residual.add(r);
    }
  }

  if (!(sum_residual.sum > 0.0)) {
    std::vector<int> order(static_cast<std::size_t>(n), 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](const int a, const int b) {
      const double ra = residual[static_cast<std::size_t>(a)];
      const double rb = residual[static_cast<std::size_t>(b)];
      if (ra != rb) {
        return ra > rb;
      }
      return a < b;
    });
    for (int k = 0; k < m_res; ++k) {
      const int idx = order[static_cast<std::size_t>(k % n)];
      ++inc[static_cast<std::size_t>(idx)];
    }
    return inc;
  }

  std::vector<double> cdf(static_cast<std::size_t>(n), 0.0);
  KahanSum cdf_sum;
  for (int i = 0; i < n; ++i) {
    const double r = residual[static_cast<std::size_t>(i)];
    const double p = (r > 0.0 && std::isfinite(r)) ? (r / sum_residual.sum) : 0.0;
    cdf_sum.add(p);
    cdf[static_cast<std::size_t>(i)] = cdf_sum.sum;
  }
  cdf.back() = 1.0;

  const double u0 = draw_u0(seed, step, counter, m_res);
  const double one_minus = std::nextafter(1.0, 0.0);
  int j = 0;
  for (int k = 0; k < m_res; ++k) {
    double u = u0 + static_cast<double>(k) / static_cast<double>(m_res);
    if (u >= 1.0) {
      u = one_minus;
    }
    while (j + 1 < n && u > cdf[static_cast<std::size_t>(j)]) {
      ++j;
    }
    ++inc[static_cast<std::size_t>(j)];
  }
  return inc;
}

__global__ void detect_bins_kernel(const std::int32_t* __restrict__ cell_id,
                                   const std::uint16_t* __restrict__ group_id,
                                   const std::uint8_t* __restrict__ mode,
                                   const double* __restrict__ energy,
                                   int* __restrict__ bin_id,
                                   int* __restrict__ bin_flag,
                                   const int n_alive,
                                   const int n_cells,
                                   const int n_groups) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_alive) {
    return;
  }
  (void)energy;
  (void)bin_id;

  const std::uint64_t cg = static_cast<std::uint64_t>(n_cells) *
                           static_cast<std::uint64_t>(n_groups);
  const std::uint64_t key =
      static_cast<std::uint64_t>(mode[tid]) * cg +
      static_cast<std::uint64_t>(cell_id[tid]) * static_cast<std::uint64_t>(n_groups) +
      static_cast<std::uint64_t>(group_id[tid]);

  int is_start = 0;
  if (tid == 0) {
    is_start = 1;
  } else {
    const std::uint64_t prev_key =
        static_cast<std::uint64_t>(mode[tid - 1]) * cg +
        static_cast<std::uint64_t>(cell_id[tid - 1]) * static_cast<std::uint64_t>(n_groups) +
        static_cast<std::uint64_t>(group_id[tid - 1]);
    is_start = (key != prev_key) ? 1 : 0;
  }
  bin_flag[tid] = is_start;
}

__global__ void bin_boundaries_and_energy_kernel(const int* __restrict__ bin_id,
                                                 const int* __restrict__ bin_flag,
                                                 const std::uint8_t* __restrict__ mode,
                                                 const double* __restrict__ energy,
                                                 int* __restrict__ bin_start,
                                                 std::uint8_t* __restrict__ bin_mode,
                                                 double* __restrict__ bin_E,
                                                 const int n_alive) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_alive) {
    return;
  }

  const int b = bin_id[tid] - (1 - bin_flag[tid]);
  if (bin_flag[tid] == 1) {
    bin_start[b] = tid;
    bin_mode[b] = mode[tid];
  }

  const double E = energy[tid];
  if (isfinite(E) && E > 0.0) {
    atomic_add_double(&bin_E[b], E);
  }
}

__global__ void gather_equalized_kernel(const PhotonPoolReadView src,
                                        const PhotonPoolWriteView dst,
                                        const int* __restrict__ gather_idx,
                                        const std::uint8_t* __restrict__ gid_override_mask,
                                        const std::uint64_t* __restrict__ gid_override_value,
                                        const int* __restrict__ bin_target,
                                        const int* __restrict__ bin_prefix,
                                        const double* __restrict__ bin_E,
                                        const int* __restrict__ bin_count,
                                        const int n_bins,
                                        const int n_target) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_target) {
    return;
  }

  const int src_idx = gather_idx[j];

  dst.pos_r[j] = src.pos_r[src_idx];
  dst.pos_z[j] = src.pos_z[src_idx];
  dst.dir_r[j] = src.dir_r[src_idx];
  dst.dir_z[j] = src.dir_z[src_idx];
  dst.dir_phi[j] = src.dir_phi[src_idx];
  if (gid_override_mask[j] != 0U) {
    dst.global_id[j] = gid_override_value[j];
    dst.rng_counter[j] = 0U;
  } else {
    dst.global_id[j] = src.global_id[src_idx];
    dst.rng_counter[j] = src.rng_counter[src_idx];
  }
  dst.cell_id[j] = src.cell_id[src_idx];
  dst.group_id[j] = src.group_id[src_idx];
  dst.mode[j] = src.mode[src_idx];
  dst.sign[j] = src.sign[src_idx];
  dst.alive[j] = static_cast<std::uint8_t>(kAlive);

  int lo = 0;
  int hi = n_bins - 1;
  while (lo < hi) {
    const int mid = (lo + hi + 1) / 2;
    if (bin_prefix[mid] <= j) {
      lo = mid;
    } else {
      hi = mid - 1;
    }
  }
  const int b = lo;

  const int count_b = bin_count[b];
  const int target_b = bin_target[b];
  if (target_b < count_b && target_b > 0) {
    const double E_unit = bin_E[b] / static_cast<double>(target_b);
    dst.energy[j] = E_unit;
    dst.weight[j] = 1.0;
    dst.birth_energy[j] = E_unit;
    dst.time_remain[j] = 0.0;
  } else {
    dst.energy[j] = src.energy[src_idx];
    dst.weight[j] = src.weight[src_idx];
    dst.birth_energy[j] = src.birth_energy[src_idx];
    dst.time_remain[j] = src.time_remain[src_idx];
  }
}

struct EssBinMark {
  int bin = 0;
  int count = 0;
  int split_factor = 1;
  int tier = 2;
  double ess = 0.0;
  double importance = 0.0;
};

__global__ void gather_split_kernel(const PhotonPoolReadView src,
                                    const PhotonPoolWriteView dst,
                                    const int* __restrict__ gather_idx,
                                    const int* __restrict__ split_factor,
                                    const std::uint8_t* __restrict__ gid_override_mask,
                                    const std::uint64_t* __restrict__ gid_override_value,
                                    const int n_out) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n_out) {
    return;
  }

  const int src_idx = gather_idx[j];
  const int factor = max(split_factor[j], 1);

  dst.pos_r[j] = src.pos_r[src_idx];
  dst.pos_z[j] = src.pos_z[src_idx];
  dst.dir_r[j] = src.dir_r[src_idx];
  dst.dir_z[j] = src.dir_z[src_idx];
  dst.dir_phi[j] = src.dir_phi[src_idx];
  dst.cell_id[j] = src.cell_id[src_idx];
  dst.group_id[j] = src.group_id[src_idx];
  dst.mode[j] = src.mode[src_idx];
  dst.sign[j] = src.sign[src_idx];
  dst.alive[j] = static_cast<std::uint8_t>(kAlive);
  dst.time_remain[j] = src.time_remain[src_idx];

  if (factor > 1) {
    const double inv_factor = 1.0 / static_cast<double>(factor);
    dst.energy[j] = src.energy[src_idx] * inv_factor;
    dst.weight[j] = src.weight[src_idx] * inv_factor;
    dst.birth_energy[j] = src.birth_energy[src_idx] * inv_factor;
  } else {
    dst.energy[j] = src.energy[src_idx];
    dst.weight[j] = src.weight[src_idx];
    dst.birth_energy[j] = src.birth_energy[src_idx];
  }

  if (gid_override_mask[j] != 0U) {
    dst.global_id[j] = gid_override_value[j];
    dst.rng_counter[j] = 0U;
  } else {
    dst.global_id[j] = src.global_id[src_idx];
    dst.rng_counter[j] = src.rng_counter[src_idx];
  }
}

}  // namespace

CensusCombResult census_comb(PhotonPool& pool,
                             const int n_alive,
                             const int n_cells,
                             const int n_groups,
                             const CensusCombConfig& cfg,
                             const std::uint64_t step_base_gid,
                             const std::uint64_t n_emit_total,
                             const std::uint32_t seed,
                             const int step,
                             double* const d_E_numerical_loss) {
  TENRYU_ASSERT(n_alive >= 0, "census_comb requires n_alive >= 0");
  TENRYU_ASSERT(n_cells >= 1, "census_comb requires n_cells >= 1");
  TENRYU_ASSERT(n_groups >= 1, "census_comb requires n_groups >= 1");
  TENRYU_ASSERT(cfg.max_particles >= 1, "census_comb requires max_particles >= 1");
  TENRYU_ASSERT(cfg.min_per_bin >= 1, "census_comb requires min_per_bin >= 1");

  CensusCombResult out;
  const int raw_target = static_cast<int>(
      std::floor(cfg.target_fraction * static_cast<double>(cfg.max_particles)));
  out.target_count = std::max(1, std::min(cfg.max_particles, raw_target));

  if (n_alive == 0) {
    pool.n_alive = 0;
    pool.n_census = 0;
    return out;
  }

  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> weight;
  std::vector<double> time_remain;
  std::vector<double> birth_energy;
  std::vector<std::int8_t> sign;
  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;

  d2h(pos_r, pool.pos_r, n_alive, "census_comb copy pos_r failed");
  d2h(pos_z, pool.pos_z, n_alive, "census_comb copy pos_z failed");
  d2h(dir_r, pool.dir_r, n_alive, "census_comb copy dir_r failed");
  d2h(dir_z, pool.dir_z, n_alive, "census_comb copy dir_z failed");
  d2h(dir_phi, pool.dir_phi, n_alive, "census_comb copy dir_phi failed");
  d2h(energy, pool.energy, n_alive, "census_comb copy energy failed");
  d2h(weight, pool.weight, n_alive, "census_comb copy weight failed");
  d2h(time_remain,
      pool.time_remain,
      n_alive,
      "census_comb copy time_remain failed");
  d2h(birth_energy,
      pool.birth_energy,
      n_alive,
      "census_comb copy birth_energy failed");
  d2h(sign, pool.sign, n_alive, "census_comb copy sign failed");
  d2h(global_id, pool.global_id, n_alive, "census_comb copy global_id failed");
  d2h(rng_counter,
      pool.rng_counter,
      n_alive,
      "census_comb copy rng_counter failed");
  d2h(cell_id, pool.cell_id, n_alive, "census_comb copy cell_id failed");
  d2h(group_id, pool.group_id, n_alive, "census_comb copy group_id failed");
  d2h(mode, pool.mode, n_alive, "census_comb copy mode failed");
  d2h(alive, pool.alive, n_alive, "census_comb copy alive failed");

  const std::uint64_t cell_group_span = static_cast<std::uint64_t>(n_cells) *
                                        static_cast<std::uint64_t>(n_groups);
  std::vector<SortedEntry> entries;
  entries.reserve(static_cast<std::size_t>(n_alive));
  double dropped_abs = 0.0;
  for (int src = 0; src < n_alive; ++src) {
    const std::size_t s = static_cast<std::size_t>(src);
    const bool valid = (alive[s] == kAlive) && (cell_id[s] >= 0) &&
                       (cell_id[s] < n_cells) &&
                       (static_cast<int>(group_id[s]) < n_groups) &&
                       ((mode[s] == kModeIMC) || (mode[s] == kModeDDMC) ||
                        (mode[s] == kModeRW));
    if (!valid) {
      const double E = energy[s];
      if (std::isfinite(E) && E != 0.0) {
        dropped_abs += std::abs(E);
      }
      continue;
    }

    const std::uint64_t key =
        static_cast<std::uint64_t>(mode[s]) * cell_group_span +
        static_cast<std::uint64_t>(cell_id[s]) * static_cast<std::uint64_t>(n_groups) +
        static_cast<std::uint64_t>(group_id[s]);
    entries.push_back({key, src});
  }

  std::sort(entries.begin(), entries.end(), [](const SortedEntry& a, const SortedEntry& b) {
    if (a.key != b.key) {
      return a.key < b.key;
    }
    return a.src < b.src;
  });

  if (entries.empty()) {
    std::vector<std::uint8_t> alive_out(static_cast<std::size_t>(n_alive), kDead);
    h2d(pool.alive, alive_out, "census_comb write alive failed");
    add_numerical_loss(d_E_numerical_loss, dropped_abs);
    pool.n_alive = 0;
    pool.n_census = 0;
    return out;
  }

  std::vector<BinInfo> bins;
  bins.reserve(entries.size());
  KahanSum E_before_sum;
  int begin = 0;
  const int n_entries = static_cast<int>(entries.size());
  while (begin < n_entries) {
    const std::uint64_t key = entries[static_cast<std::size_t>(begin)].key;
    int end = begin + 1;
    while (end < n_entries && entries[static_cast<std::size_t>(end)].key == key) {
      ++end;
    }

    KahanSum E_bin_sum;
    for (int i = begin; i < end; ++i) {
      const int src = entries[static_cast<std::size_t>(i)].src;
      E_bin_sum.add(clamp_energy(energy[static_cast<std::size_t>(src)]));
    }

    BinInfo bin;
    bin.key = key;
    bin.begin = begin;
    bin.end = end;
    bin.n = end - begin;
    bin.E_bin = E_bin_sum.sum;
    bins.push_back(bin);
    E_before_sum.add(bin.E_bin);
    begin = end;
  }
  out.E_before = E_before_sum.sum;

  KahanSum score_sum;
  for (BinInfo& bin : bins) {
    const std::uint64_t mode_from_key =
        (cell_group_span > 0) ? (bin.key / cell_group_span) : 0ULL;
    const double alpha =
        ((mode_from_key == static_cast<std::uint64_t>(kModeDDMC)) ||
         (mode_from_key == static_cast<std::uint64_t>(kModeRW)))
                             ? cfg.mode_weight_ddmc
                             : cfg.mode_weight_imc;
    bin.score = alpha * bin.E_bin;
    score_sum.add(bin.score);
  }

  if (!(score_sum.sum > 0.0)) {
    out.zero_score = true;
    out.n_alive_out = 0;
    out.E_after = 0.0;
    out.E_killed_bins = out.E_before;
    add_numerical_loss(d_E_numerical_loss, out.E_killed_bins + dropped_abs);
    core::log_warning("census_comb: all bins zero energy, killed " +
                      std::to_string(n_entries) + " particles");
    std::vector<std::uint8_t> alive_out(static_cast<std::size_t>(n_alive), kDead);
    h2d(pool.alive, alive_out, "census_comb write alive failed");
    pool.n_alive = 0;
    pool.n_census = 0;
    return out;
  }

  const int B = static_cast<int>(bins.size());
  const int N_min = cfg.min_per_bin;
  const int N_max = out.target_count;
  const std::int64_t N_base = static_cast<std::int64_t>(B) * static_cast<std::int64_t>(N_min);

  if (N_base > static_cast<std::int64_t>(N_max)) {
    out.emergency = true;
    std::int64_t n_det = 0;
    std::vector<double> residual(static_cast<std::size_t>(B), 0.0);

    for (int b = 0; b < B; ++b) {
      const double w = static_cast<double>(N_max) * bins[static_cast<std::size_t>(b)].score /
                       score_sum.sum;
      const int n_floor = std::max(0, static_cast<int>(std::floor(w)));
      bins[static_cast<std::size_t>(b)].target = n_floor;
      n_det += n_floor;
      residual[static_cast<std::size_t>(b)] =
          std::max(0.0, w - static_cast<double>(n_floor));
    }

    if (n_det > static_cast<std::int64_t>(N_max)) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const int ta = bins[static_cast<std::size_t>(a)].target;
        const int tb = bins[static_cast<std::size_t>(b)].target;
        if (ta != tb) {
          return ta > tb;
        }
        return bins[static_cast<std::size_t>(a)].key <
               bins[static_cast<std::size_t>(b)].key;
      });
      while (n_det > static_cast<std::int64_t>(N_max)) {
        bool changed = false;
        for (const int idx : order) {
          BinInfo& bin = bins[static_cast<std::size_t>(idx)];
          if (bin.target > 0) {
            --bin.target;
            --n_det;
            changed = true;
            if (n_det == static_cast<std::int64_t>(N_max)) {
              break;
            }
          }
        }
        if (!changed) {
          break;
        }
      }
    }

    const int m_res =
        static_cast<int>(static_cast<std::int64_t>(N_max) - n_det);
    if (m_res > 0) {
      const std::uint64_t counter = 2ULL * cell_group_span;
      const std::vector<int> inc =
          systematic_residual_counts(residual, m_res, seed, step, counter);
      for (int b = 0; b < B; ++b) {
        bins[static_cast<std::size_t>(b)].target += inc[static_cast<std::size_t>(b)];
      }
    }

    int target_sum = 0;
    for (const BinInfo& bin : bins) {
      target_sum += bin.target;
    }
    if (target_sum < N_max) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        if (residual[static_cast<std::size_t>(a)] !=
            residual[static_cast<std::size_t>(b)]) {
          return residual[static_cast<std::size_t>(a)] >
                 residual[static_cast<std::size_t>(b)];
        }
        return bins[static_cast<std::size_t>(a)].key <
               bins[static_cast<std::size_t>(b)].key;
      });
      const int missing = N_max - target_sum;
      for (int i = 0; i < missing; ++i) {
        const int idx = order[static_cast<std::size_t>(i % B)];
        ++bins[static_cast<std::size_t>(idx)].target;
      }
    } else if (target_sum > N_max) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const int ta = bins[static_cast<std::size_t>(a)].target;
        const int tb = bins[static_cast<std::size_t>(b)].target;
        if (ta != tb) {
          return ta > tb;
        }
        return bins[static_cast<std::size_t>(a)].key <
               bins[static_cast<std::size_t>(b)].key;
      });
      int excess = target_sum - N_max;
      for (const int idx : order) {
        BinInfo& bin = bins[static_cast<std::size_t>(idx)];
        while (excess > 0 && bin.target > 0) {
          --bin.target;
          --excess;
        }
        if (excess == 0) {
          break;
        }
      }
    }
  } else {
    const int N_extra = N_max - static_cast<int>(N_base);
    int target_sum = 0;
    for (BinInfo& bin : bins) {
      const double raw_extra =
          static_cast<double>(N_extra) * bin.score / score_sum.sum;
      const int floor_extra = std::max(0, static_cast<int>(std::floor(raw_extra)));
      bin.target = N_min + floor_extra;
      bin.frac = raw_extra - static_cast<double>(floor_extra);
      if (!(bin.frac > 0.0)) {
        bin.frac = 0.0;
      }
      target_sum += bin.target;
    }

    int remaining = N_max - target_sum;
    if (remaining > 0) {
      TENRYU_ASSERT(remaining <= B,
                    "census_comb largest-remainder remainder exceeds bin count");
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const double fa = bins[static_cast<std::size_t>(a)].frac;
        const double fb = bins[static_cast<std::size_t>(b)].frac;
        if (fa != fb) {
          return fa > fb;
        }
        return bins[static_cast<std::size_t>(a)].key <
               bins[static_cast<std::size_t>(b)].key;
      });
      for (int i = 0; i < remaining; ++i) {
        const int idx = order[static_cast<std::size_t>(i)];
        ++bins[static_cast<std::size_t>(idx)].target;
      }
    } else if (remaining < 0) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const int ta = bins[static_cast<std::size_t>(a)].target;
        const int tb = bins[static_cast<std::size_t>(b)].target;
        if (ta != tb) {
          return ta > tb;
        }
        return bins[static_cast<std::size_t>(a)].key <
               bins[static_cast<std::size_t>(b)].key;
      });
      for (const int idx : order) {
        BinInfo& bin = bins[static_cast<std::size_t>(idx)];
        while (remaining < 0 && bin.target > 0) {
          --bin.target;
          ++remaining;
        }
        if (remaining == 0) {
          break;
        }
      }
    }
  }

  std::vector<double> out_pos_r;
  std::vector<double> out_pos_z;
  std::vector<double> out_dir_r;
  std::vector<double> out_dir_z;
  std::vector<double> out_dir_phi;
  std::vector<double> out_energy;
  std::vector<double> out_weight;
  std::vector<double> out_time_remain;
  std::vector<double> out_birth_energy;
  std::vector<std::int8_t> out_sign;
  std::vector<std::uint64_t> out_global_id;
  std::vector<std::uint32_t> out_rng_counter;
  std::vector<std::int32_t> out_cell_id;
  std::vector<std::uint16_t> out_group_id;
  std::vector<std::uint8_t> out_mode;

  out_pos_r.reserve(entries.size());
  out_pos_z.reserve(entries.size());
  out_dir_r.reserve(entries.size());
  out_dir_z.reserve(entries.size());
  out_dir_phi.reserve(entries.size());
  out_energy.reserve(entries.size());
  out_weight.reserve(entries.size());
  out_time_remain.reserve(entries.size());
  out_birth_energy.reserve(entries.size());
  out_sign.reserve(entries.size());
  out_global_id.reserve(entries.size());
  out_rng_counter.reserve(entries.size());
  out_cell_id.reserve(entries.size());
  out_group_id.reserve(entries.size());
  out_mode.reserve(entries.size());

  const double nan = std::numeric_limits<double>::quiet_NaN();
  KahanSum E_after_sum;
  int n_combed_bins = 0;
  int n_killed_bins = 0;
  double E_killed_bins = 0.0;
  std::uint64_t dup_offset = 0ULL;
  constexpr std::uint64_t kStepLocalLimit = (1ULL << 40);
  TENRYU_ASSERT(n_emit_total < kStepLocalLimit,
                "census_comb requires n_emit_total < 2^40");

  for (const BinInfo& bin : bins) {
    const int n_bin = bin.n;
    const int m = bin.target;

    if (!(bin.E_bin > 0.0)) {
      ++n_killed_bins;
      ++n_combed_bins;
      E_killed_bins += bin.E_bin;
      continue;
    }

    if (m <= 0) {
      ++n_killed_bins;
      ++n_combed_bins;
      E_killed_bins += bin.E_bin;
      continue;
    }

    if (n_bin <= m) {
      for (int i = bin.begin; i < bin.end; ++i) {
        const int src = entries[static_cast<std::size_t>(i)].src;
        const std::size_t s = static_cast<std::size_t>(src);
        out_pos_r.push_back(pos_r[s]);
        out_pos_z.push_back(pos_z[s]);
        out_dir_r.push_back(dir_r[s]);
        out_dir_z.push_back(dir_z[s]);
        out_dir_phi.push_back(dir_phi[s]);
        out_energy.push_back(energy[s]);
        out_weight.push_back(weight[s]);
        out_time_remain.push_back(time_remain[s]);
        out_birth_energy.push_back(birth_energy[s]);
        out_sign.push_back(sign[s]);
        out_global_id.push_back(global_id[s]);
        out_rng_counter.push_back(rng_counter[s]);
        out_cell_id.push_back(cell_id[s]);
        out_group_id.push_back(group_id[s]);
        out_mode.push_back(mode[s]);
      }
      E_after_sum.add(bin.E_bin);
      continue;
    }

    ++n_combed_bins;
    std::vector<double> E_clamped(static_cast<std::size_t>(n_bin), 0.0);
    std::vector<int> n_copy(static_cast<std::size_t>(n_bin), 0);
    int n_det = 0;
    for (int i = 0; i < n_bin; ++i) {
      const int src = entries[static_cast<std::size_t>(bin.begin + i)].src;
      const double Ei = clamp_energy(energy[static_cast<std::size_t>(src)]);
      E_clamped[static_cast<std::size_t>(i)] = Ei;
      const double wi = Ei / bin.E_bin;
      const int ni = std::max(
          0,
          static_cast<int>(std::floor(static_cast<double>(m) * wi)));
      n_copy[static_cast<std::size_t>(i)] = ni;
      n_det += ni;
    }

    if (n_det > m) {
      std::vector<int> order(static_cast<std::size_t>(n_bin), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const int na = n_copy[static_cast<std::size_t>(a)];
        const int nb = n_copy[static_cast<std::size_t>(b)];
        if (na != nb) {
          return na > nb;
        }
        const double Ea = E_clamped[static_cast<std::size_t>(a)];
        const double Eb = E_clamped[static_cast<std::size_t>(b)];
        if (Ea != Eb) {
          return Ea > Eb;
        }
        return a < b;
      });
      while (n_det > m) {
        bool changed = false;
        for (const int idx : order) {
          int& ni = n_copy[static_cast<std::size_t>(idx)];
          if (ni > 0) {
            --ni;
            --n_det;
            changed = true;
            if (n_det == m) {
              break;
            }
          }
        }
        if (!changed) {
          break;
        }
      }
    }

    const int m_res = m - n_det;
    if (m_res > 0) {
      std::vector<double> residual(static_cast<std::size_t>(n_bin), 0.0);
      for (int i = 0; i < n_bin; ++i) {
        const double wi = E_clamped[static_cast<std::size_t>(i)] / bin.E_bin;
        const double ri = static_cast<double>(m) * wi -
                          static_cast<double>(n_copy[static_cast<std::size_t>(i)]);
        residual[static_cast<std::size_t>(i)] = std::max(0.0, ri);
      }
      const std::vector<int> inc =
          systematic_residual_counts(residual, m_res, seed, step, bin.key);
      for (int i = 0; i < n_bin; ++i) {
        n_copy[static_cast<std::size_t>(i)] += inc[static_cast<std::size_t>(i)];
      }
    }

    int n_out_bin = 0;
    for (const int ni : n_copy) {
      n_out_bin += ni;
    }
    if (n_out_bin < m) {
      std::vector<int> order(static_cast<std::size_t>(n_bin), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const double Ea = E_clamped[static_cast<std::size_t>(a)];
        const double Eb = E_clamped[static_cast<std::size_t>(b)];
        if (Ea != Eb) {
          return Ea > Eb;
        }
        return a < b;
      });
      int missing = m - n_out_bin;
      for (int i = 0; i < missing; ++i) {
        const int idx = order[static_cast<std::size_t>(i % n_bin)];
        ++n_copy[static_cast<std::size_t>(idx)];
      }
      n_out_bin = m;
    } else if (n_out_bin > m) {
      std::vector<int> order(static_cast<std::size_t>(n_bin), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        const int na = n_copy[static_cast<std::size_t>(a)];
        const int nb = n_copy[static_cast<std::size_t>(b)];
        if (na != nb) {
          return na > nb;
        }
        return a < b;
      });
      int excess = n_out_bin - m;
      for (const int idx : order) {
        int& ni = n_copy[static_cast<std::size_t>(idx)];
        while (excess > 0 && ni > 0) {
          --ni;
          --excess;
        }
        if (excess == 0) {
          break;
        }
      }
    }

    std::vector<int> selected;
    selected.reserve(static_cast<std::size_t>(m));
    for (int i = 0; i < n_bin; ++i) {
      const int ni = n_copy[static_cast<std::size_t>(i)];
      for (int c = 0; c < ni; ++c) {
        selected.push_back(i);
      }
    }
    if (static_cast<int>(selected.size()) > m) {
      selected.resize(static_cast<std::size_t>(m));
    } else if (static_cast<int>(selected.size()) < m) {
      while (static_cast<int>(selected.size()) < m) {
        selected.push_back(0);
      }
    }

    const std::size_t bin_out_begin = out_energy.size();
    for (int i = 0; i < m; ++i) {
      const int loc = selected[static_cast<std::size_t>(i)];
      const int src = entries[static_cast<std::size_t>(bin.begin + loc)].src;
      const std::size_t s = static_cast<std::size_t>(src);
      const std::uint8_t m_src = mode[s];
      if (m_src == kModeDDMC) {
        out_pos_r.push_back(nan);
        out_pos_z.push_back(nan);
        out_dir_r.push_back(nan);
        out_dir_z.push_back(nan);
        out_dir_phi.push_back(nan);
      } else {
        out_pos_r.push_back(pos_r[s]);
        out_pos_z.push_back(pos_z[s]);
        out_dir_r.push_back(dir_r[s]);
        out_dir_z.push_back(dir_z[s]);
        out_dir_phi.push_back(dir_phi[s]);
      }

      if (n_copy[static_cast<std::size_t>(loc)] == 1) {
        out_global_id.push_back(global_id[s]);
        out_rng_counter.push_back(rng_counter[s]);
      } else {
        TENRYU_ASSERT(n_emit_total + dup_offset < kStepLocalLimit,
                      "census_comb duplicate local id exceeds 2^40");
        TENRYU_ASSERT(step_base_gid <=
                          std::numeric_limits<std::uint64_t>::max() -
                              (n_emit_total + dup_offset),
                      "census_comb global_id overflow");
        out_global_id.push_back(step_base_gid + n_emit_total + dup_offset);
        out_rng_counter.push_back(0U);
        ++dup_offset;
      }

      out_energy.push_back(0.0);
      out_weight.push_back(1.0);
      out_time_remain.push_back(0.0);
      out_birth_energy.push_back(0.0);
      out_sign.push_back(sign[s]);
      out_cell_id.push_back(cell_id[s]);
      out_group_id.push_back(group_id[s]);
      out_mode.push_back(m_src);
    }

    const double E_unit = bin.E_bin / static_cast<double>(m);
    for (int i = 0; i < m; ++i) {
      double Ei = E_unit;
      if (i == m - 1) {
        Ei = bin.E_bin - static_cast<double>(m - 1) * E_unit;
      }
      const std::size_t dst = bin_out_begin + static_cast<std::size_t>(i);
      out_energy[dst] = Ei;
      out_birth_energy[dst] = Ei;
    }
    E_after_sum.add(bin.E_bin);
  }

  out.n_alive_out = static_cast<int>(out_energy.size());
  out.n_combed_bins = n_combed_bins;
  out.E_after = E_after_sum.sum;
  out.E_killed_bins = E_killed_bins;

  if (out.emergency) {
    core::log_warning("census_comb: emergency allocation path engaged (N_base > N_max); "
                      "killed bins=" +
                      std::to_string(n_killed_bins));
  }

  add_numerical_loss(d_E_numerical_loss, dropped_abs + out.E_killed_bins);

  h2d(pool.pos_r, out_pos_r, "census_comb write pos_r failed");
  h2d(pool.pos_z, out_pos_z, "census_comb write pos_z failed");
  h2d(pool.dir_r, out_dir_r, "census_comb write dir_r failed");
  h2d(pool.dir_z, out_dir_z, "census_comb write dir_z failed");
  h2d(pool.dir_phi, out_dir_phi, "census_comb write dir_phi failed");
  h2d(pool.energy, out_energy, "census_comb write energy failed");
  h2d(pool.weight, out_weight, "census_comb write weight failed");
  h2d(pool.time_remain,
      out_time_remain,
      "census_comb write time_remain failed");
  h2d(pool.birth_energy,
      out_birth_energy,
      "census_comb write birth_energy failed");
  h2d(pool.sign, out_sign, "census_comb write sign failed");
  h2d(pool.global_id, out_global_id, "census_comb write global_id failed");
  h2d(pool.rng_counter,
      out_rng_counter,
      "census_comb write rng_counter failed");
  h2d(pool.cell_id, out_cell_id, "census_comb write cell_id failed");
  h2d(pool.group_id, out_group_id, "census_comb write group_id failed");
  h2d(pool.mode, out_mode, "census_comb write mode failed");

  std::vector<std::uint8_t> alive_out(static_cast<std::size_t>(n_alive), kDead);
  for (int i = 0; i < out.n_alive_out; ++i) {
    alive_out[static_cast<std::size_t>(i)] = kAlive;
  }
  h2d(pool.alive, alive_out, "census_comb write alive failed");

  pool.n_alive = out.n_alive_out;
  pool.n_census = out.n_alive_out;
  return out;
}

CensusCombResult census_comb_gpu(PhotonPool& pool,
                                 const int n_alive,
                                 const int n_cells,
                                 const int n_groups,
                                 const CensusCombConfig& cfg,
                                 const std::uint64_t step_base_gid,
                                 const std::uint64_t n_emit_total,
                                 const std::uint32_t seed,
                                 const int step,
                                 double* const d_E_numerical_loss) {
  TENRYU_ASSERT(n_alive >= 0, "census_comb_gpu requires n_alive >= 0");
  TENRYU_ASSERT(n_cells >= 1, "census_comb_gpu requires n_cells >= 1");
  TENRYU_ASSERT(n_groups >= 1, "census_comb_gpu requires n_groups >= 1");
  TENRYU_ASSERT(cfg.max_particles >= 1, "census_comb_gpu requires max_particles >= 1");
  TENRYU_ASSERT(cfg.min_per_bin >= 1, "census_comb_gpu requires min_per_bin >= 1");
  constexpr std::uint64_t kStepLocalLimit = (1ULL << 40);
  TENRYU_ASSERT(n_emit_total < kStepLocalLimit,
                "census_comb_gpu requires n_emit_total < 2^40");

  CensusCombResult out;
  const int raw_target = static_cast<int>(
      std::floor(cfg.target_fraction * static_cast<double>(cfg.max_particles)));
  out.target_count = std::max(1, std::min(cfg.max_particles, raw_target));

  if (n_alive == 0) {
    pool.n_alive = 0;
    pool.n_census = 0;
    return out;
  }

  int* d_bin_flag = nullptr;
  int* d_bin_id = nullptr;
  void* d_scan_temp = nullptr;
  std::size_t scan_temp_bytes = 0;
  int* d_bin_start = nullptr;
  std::uint8_t* d_bin_mode = nullptr;
  double* d_bin_E = nullptr;
  int* d_bin_target = nullptr;
  int* d_bin_prefix = nullptr;
  int* d_bin_count = nullptr;
  int* d_gather_idx = nullptr;
  std::uint8_t* d_gid_override_mask = nullptr;
  std::uint64_t* d_gid_override_value = nullptr;

  auto free_if = [](void* ptr, const char* msg) {
    if (ptr != nullptr) {
      cuda_check(cudaFree(ptr), msg);
    }
  };
  auto cleanup = [&]() {
    free_if(d_gid_override_value,
            "census_comb_gpu cudaFree d_gid_override_value failed");
    free_if(d_gid_override_mask,
            "census_comb_gpu cudaFree d_gid_override_mask failed");
    free_if(d_gather_idx, "census_comb_gpu cudaFree d_gather_idx failed");
    free_if(d_bin_count, "census_comb_gpu cudaFree d_bin_count failed");
    free_if(d_bin_prefix, "census_comb_gpu cudaFree d_bin_prefix failed");
    free_if(d_bin_target, "census_comb_gpu cudaFree d_bin_target failed");
    free_if(d_bin_E, "census_comb_gpu cudaFree d_bin_E failed");
    free_if(d_bin_mode, "census_comb_gpu cudaFree d_bin_mode failed");
    free_if(d_bin_start, "census_comb_gpu cudaFree d_bin_start failed");
    free_if(d_scan_temp, "census_comb_gpu cudaFree d_scan_temp failed");
    free_if(d_bin_id, "census_comb_gpu cudaFree d_bin_id failed");
    free_if(d_bin_flag, "census_comb_gpu cudaFree d_bin_flag failed");
    d_gid_override_value = nullptr;
    d_gid_override_mask = nullptr;
    d_gather_idx = nullptr;
    d_bin_count = nullptr;
    d_bin_prefix = nullptr;
    d_bin_target = nullptr;
    d_bin_E = nullptr;
    d_bin_mode = nullptr;
    d_bin_start = nullptr;
    d_scan_temp = nullptr;
    d_bin_id = nullptr;
    d_bin_flag = nullptr;
  };

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_flag),
                        sizeof(int) * static_cast<std::size_t>(n_alive)),
             "census_comb_gpu cudaMalloc d_bin_flag failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_id),
                        sizeof(int) * static_cast<std::size_t>(n_alive)),
             "census_comb_gpu cudaMalloc d_bin_id failed");

  const int blocks = (n_alive + kThreadsPerBlock - 1) / kThreadsPerBlock;
  detect_bins_kernel<<<blocks, kThreadsPerBlock>>>(pool.cell_id,
                                                    pool.group_id,
                                                    pool.mode,
                                                    pool.energy,
                                                    d_bin_id,
                                                    d_bin_flag,
                                                    n_alive,
                                                    n_cells,
                                                    n_groups);
  cuda_check(cudaGetLastError(), "census_comb_gpu launch detect_bins_kernel failed");

  cuda_check(cub::DeviceScan::ExclusiveSum(nullptr,
                                           scan_temp_bytes,
                                           d_bin_flag,
                                           d_bin_id,
                                           n_alive),
             "census_comb_gpu CUB scan temp-storage query failed");
  if (scan_temp_bytes > 0) {
    cuda_check(cudaMalloc(&d_scan_temp, scan_temp_bytes),
               "census_comb_gpu cudaMalloc d_scan_temp failed");
  }
  cuda_check(cub::DeviceScan::ExclusiveSum(d_scan_temp,
                                           scan_temp_bytes,
                                           d_bin_flag,
                                           d_bin_id,
                                           n_alive),
             "census_comb_gpu CUB exclusive scan failed");
  cuda_check(cudaGetLastError(), "census_comb_gpu launch CUB exclusive scan failed");

  int h_last_id = 0;
  int h_last_flag = 0;
  cuda_check(cudaMemcpy(&h_last_id,
                        d_bin_id + (n_alive - 1),
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu copy last bin_id failed");
  cuda_check(cudaMemcpy(&h_last_flag,
                        d_bin_flag + (n_alive - 1),
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu copy last bin_flag failed");
  const int B = h_last_id + h_last_flag;
  TENRYU_ASSERT(B >= 1, "census_comb_gpu requires at least one bin for n_alive > 0");

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_start),
                        sizeof(int) * static_cast<std::size_t>(B + 1)),
             "census_comb_gpu cudaMalloc d_bin_start failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_mode),
                        sizeof(std::uint8_t) * static_cast<std::size_t>(B)),
             "census_comb_gpu cudaMalloc d_bin_mode failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_E),
                        sizeof(double) * static_cast<std::size_t>(B)),
             "census_comb_gpu cudaMalloc d_bin_E failed");
  cuda_check(cudaMemset(d_bin_E, 0, sizeof(double) * static_cast<std::size_t>(B)),
             "census_comb_gpu cudaMemset d_bin_E failed");

  bin_boundaries_and_energy_kernel<<<blocks, kThreadsPerBlock>>>(d_bin_id,
                                                                  d_bin_flag,
                                                                  pool.mode,
                                                                  pool.energy,
                                                                  d_bin_start,
                                                                  d_bin_mode,
                                                                  d_bin_E,
                                                                  n_alive);
  cuda_check(cudaGetLastError(),
             "census_comb_gpu launch bin_boundaries_and_energy_kernel failed");
  cuda_check(cudaMemcpy(d_bin_start + B,
                        &n_alive,
                        sizeof(int),
                        cudaMemcpyHostToDevice),
             "census_comb_gpu write bin_start sentinel failed");

  std::vector<int> h_bin_start(static_cast<std::size_t>(B + 1), 0);
  std::vector<std::uint8_t> h_mode_sample(static_cast<std::size_t>(B), 0U);
  cuda_check(cudaMemcpy(h_bin_start.data(),
                        d_bin_start,
                        sizeof(int) * static_cast<std::size_t>(B + 1),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu copy bin_start (D2H) failed");
  cuda_check(cudaMemcpy(h_mode_sample.data(),
                        d_bin_mode,
                        sizeof(std::uint8_t) * static_cast<std::size_t>(B),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu copy bin_mode (D2H) failed");

  std::vector<int> h_bin_count(static_cast<std::size_t>(B), 0);
  for (int b = 0; b < B; ++b) {
    h_bin_count[static_cast<std::size_t>(b)] =
        h_bin_start[static_cast<std::size_t>(b + 1)] -
        h_bin_start[static_cast<std::size_t>(b)];
  }

  // D2H energy, cell_id, group_id for deterministic bin_E and bin_key computation.
  // GPU atomicAdd bin_E is non-deterministic in summation order; bin_key must
  // match CPU census_comb's key for systematic_residual_counts reproducibility.
  std::vector<double> h_energy(static_cast<std::size_t>(n_alive), 0.0);
  std::vector<std::int32_t> h_cell_id(static_cast<std::size_t>(n_alive), 0);
  std::vector<std::uint16_t> h_group_id(static_cast<std::size_t>(n_alive), 0U);
  cuda_check(cudaMemcpy(h_energy.data(),
                        pool.energy,
                        sizeof(double) * static_cast<std::size_t>(n_alive),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu D2H energy failed");
  cuda_check(cudaMemcpy(h_cell_id.data(),
                        pool.cell_id,
                        sizeof(std::int32_t) * static_cast<std::size_t>(n_alive),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu D2H cell_id failed");
  cuda_check(cudaMemcpy(h_group_id.data(),
                        pool.group_id,
                        sizeof(std::uint16_t) * static_cast<std::size_t>(n_alive),
                        cudaMemcpyDeviceToHost),
             "census_comb_gpu D2H group_id failed");

  // Compute bin energies from D2H'd individual energies (deterministic, sequential)
  const std::uint64_t cell_group_span =
      static_cast<std::uint64_t>(n_cells) * static_cast<std::uint64_t>(n_groups);
  std::vector<double> h_bin_E(static_cast<std::size_t>(B), 0.0);
  std::vector<std::uint64_t> h_bin_key(static_cast<std::size_t>(B), 0ULL);
  for (int b = 0; b < B; ++b) {
    const int bs = h_bin_start[static_cast<std::size_t>(b)];
    const int nb = h_bin_count[static_cast<std::size_t>(b)];
    KahanSum cpu_sum;
    for (int i = 0; i < nb; ++i) {
      cpu_sum.add(clamp_energy(h_energy[static_cast<std::size_t>(bs + i)]));
    }
    h_bin_E[static_cast<std::size_t>(b)] = cpu_sum.sum;
    // Compute bucket key from first particle in bin (matches CPU census_comb key)
    h_bin_key[static_cast<std::size_t>(b)] =
        static_cast<std::uint64_t>(h_mode_sample[static_cast<std::size_t>(b)]) *
            cell_group_span +
        static_cast<std::uint64_t>(h_cell_id[static_cast<std::size_t>(bs)]) *
            static_cast<std::uint64_t>(n_groups) +
        static_cast<std::uint64_t>(h_group_id[static_cast<std::size_t>(bs)]);
  }

  // Upload deterministic bin_E to device for gather kernel energy equalization
  cuda_check(cudaMemcpy(d_bin_E,
                        h_bin_E.data(),
                        sizeof(double) * static_cast<std::size_t>(B),
                        cudaMemcpyHostToDevice),
             "census_comb_gpu upload deterministic bin_E (H2D) failed");

  std::vector<double> h_score(static_cast<std::size_t>(B), 0.0);
  double score_sum = 0.0;
  double E_before = 0.0;
  for (int b = 0; b < B; ++b) {
    const double alpha =
        ((h_mode_sample[static_cast<std::size_t>(b)] == static_cast<std::uint8_t>(kModeDDMC)) ||
         (h_mode_sample[static_cast<std::size_t>(b)] == static_cast<std::uint8_t>(kModeRW)))
            ? cfg.mode_weight_ddmc
            : cfg.mode_weight_imc;
    const double score = alpha * h_bin_E[static_cast<std::size_t>(b)];
    h_score[static_cast<std::size_t>(b)] = score;
    score_sum += score;
    E_before += h_bin_E[static_cast<std::size_t>(b)];
  }
  out.E_before = E_before;

  if (!(score_sum > 0.0)) {
    out.zero_score = true;
    out.n_alive_out = 0;
    out.E_after = 0.0;
    out.E_killed_bins = out.E_before;
    add_numerical_loss(d_E_numerical_loss, out.E_killed_bins);
    cuda_check(cudaMemset(pool.alive,
                          static_cast<int>(kDead),
                          sizeof(std::uint8_t) * static_cast<std::size_t>(n_alive)),
               "census_comb_gpu zero-score cudaMemset alive failed");
    pool.n_alive = 0;
    pool.n_census = 0;
    cleanup();
    return out;
  }

  const int N_max = out.target_count;
  const int N_min_per_bin = cfg.min_per_bin;
  const std::int64_t N_base = static_cast<std::int64_t>(B) *
                              static_cast<std::int64_t>(N_min_per_bin);
  std::vector<int> h_target(static_cast<std::size_t>(B), 0);

  if (N_base > static_cast<std::int64_t>(N_max)) {
    out.emergency = true;
    int target_sum = 0;
    std::vector<double> h_frac(static_cast<std::size_t>(B), 0.0);
    for (int b = 0; b < B; ++b) {
      const double w = static_cast<double>(N_max) *
                       h_score[static_cast<std::size_t>(b)] / score_sum;
      const int n_floor = std::max(0, static_cast<int>(std::floor(w)));
      h_target[static_cast<std::size_t>(b)] = n_floor;
      h_frac[static_cast<std::size_t>(b)] = w - static_cast<double>(n_floor);
      target_sum += n_floor;
    }

    int remaining = N_max - target_sum;
    if (remaining > 0) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        if (h_frac[static_cast<std::size_t>(a)] != h_frac[static_cast<std::size_t>(b)]) {
          return h_frac[static_cast<std::size_t>(a)] >
                 h_frac[static_cast<std::size_t>(b)];
        }
        return a < b;
      });
      for (int i = 0; i < remaining && i < B; ++i) {
        ++h_target[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
      }
    } else if (remaining < 0) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        if (h_target[static_cast<std::size_t>(a)] !=
            h_target[static_cast<std::size_t>(b)]) {
          return h_target[static_cast<std::size_t>(a)] >
                 h_target[static_cast<std::size_t>(b)];
        }
        return a < b;
      });
      for (const int idx : order) {
        int& t = h_target[static_cast<std::size_t>(idx)];
        while (remaining < 0 && t > 0) {
          --t;
          ++remaining;
        }
        if (remaining == 0) {
          break;
        }
      }
    }
  } else {
    const int N_extra = N_max - static_cast<int>(N_base);
    int target_sum = 0;
    std::vector<double> h_frac(static_cast<std::size_t>(B), 0.0);
    for (int b = 0; b < B; ++b) {
      const double raw_extra =
          static_cast<double>(N_extra) * h_score[static_cast<std::size_t>(b)] / score_sum;
      const int floor_extra = std::max(0, static_cast<int>(std::floor(raw_extra)));
      h_target[static_cast<std::size_t>(b)] = N_min_per_bin + floor_extra;
      h_frac[static_cast<std::size_t>(b)] = raw_extra - static_cast<double>(floor_extra);
      target_sum += h_target[static_cast<std::size_t>(b)];
    }

    int remaining = N_max - target_sum;
    if (remaining > 0) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        if (h_frac[static_cast<std::size_t>(a)] != h_frac[static_cast<std::size_t>(b)]) {
          return h_frac[static_cast<std::size_t>(a)] >
                 h_frac[static_cast<std::size_t>(b)];
        }
        return a < b;
      });
      for (int i = 0; i < remaining && i < B; ++i) {
        ++h_target[static_cast<std::size_t>(order[static_cast<std::size_t>(i)])];
      }
    } else if (remaining < 0) {
      std::vector<int> order(static_cast<std::size_t>(B), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b) {
        if (h_target[static_cast<std::size_t>(a)] !=
            h_target[static_cast<std::size_t>(b)]) {
          return h_target[static_cast<std::size_t>(a)] >
                 h_target[static_cast<std::size_t>(b)];
        }
        return a < b;
      });
      for (const int idx : order) {
        int& t = h_target[static_cast<std::size_t>(idx)];
        while (remaining < 0 && t > 0) {
          --t;
          ++remaining;
        }
        if (remaining == 0) {
          break;
        }
      }
    }
  }

  for (int b = 0; b < B; ++b) {
    if (!(h_bin_E[static_cast<std::size_t>(b)] > 0.0)) {
      h_target[static_cast<std::size_t>(b)] = 0;
      continue;
    }
    h_target[static_cast<std::size_t>(b)] =
        std::min(h_target[static_cast<std::size_t>(b)],
                 h_bin_count[static_cast<std::size_t>(b)]);
  }

  std::vector<int> h_prefix(static_cast<std::size_t>(B), 0);
  for (int b = 1; b < B; ++b) {
    h_prefix[static_cast<std::size_t>(b)] =
        h_prefix[static_cast<std::size_t>(b - 1)] +
        h_target[static_cast<std::size_t>(b - 1)];
  }
  const int N_target = h_prefix.back() + h_target.back();

  int n_combed_bins = 0;
  double E_after = 0.0;
  double E_killed_bins = 0.0;
  for (int b = 0; b < B; ++b) {
    const int t = h_target[static_cast<std::size_t>(b)];
    const int n_bin = h_bin_count[static_cast<std::size_t>(b)];
    const double E_bin = h_bin_E[static_cast<std::size_t>(b)];
    if (t <= 0 && E_bin > 0.0) {
      E_killed_bins += E_bin;
    } else {
      E_after += E_bin;
    }
    if (t < n_bin) {
      ++n_combed_bins;
    }
  }

  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_target),
                        sizeof(int) * static_cast<std::size_t>(B)),
             "census_comb_gpu cudaMalloc d_bin_target failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_prefix),
                        sizeof(int) * static_cast<std::size_t>(B)),
             "census_comb_gpu cudaMalloc d_bin_prefix failed");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_bin_count),
                        sizeof(int) * static_cast<std::size_t>(B)),
             "census_comb_gpu cudaMalloc d_bin_count failed");
  cuda_check(cudaMemcpy(d_bin_target,
                        h_target.data(),
                        sizeof(int) * static_cast<std::size_t>(B),
                        cudaMemcpyHostToDevice),
             "census_comb_gpu copy bin_target (H2D) failed");
  cuda_check(cudaMemcpy(d_bin_prefix,
                        h_prefix.data(),
                        sizeof(int) * static_cast<std::size_t>(B),
                        cudaMemcpyHostToDevice),
             "census_comb_gpu copy bin_prefix (H2D) failed");
  cuda_check(cudaMemcpy(d_bin_count,
                        h_bin_count.data(),
                        sizeof(int) * static_cast<std::size_t>(B),
                        cudaMemcpyHostToDevice),
             "census_comb_gpu copy bin_count (H2D) failed");

  // h_energy already D2H'd and h_bin_E already recomputed above

  std::vector<int> h_gather_idx(static_cast<std::size_t>(N_target), 0);
  std::vector<std::uint8_t> h_gid_override_mask(static_cast<std::size_t>(N_target), 0U);
  std::vector<std::uint64_t> h_gid_override_value(static_cast<std::size_t>(N_target), 0ULL);
  std::uint64_t dup_offset = 0ULL;
  int out_pos = 0;
  for (int b = 0; b < B; ++b) {
    const int bin_start = h_bin_start[static_cast<std::size_t>(b)];
    const int n_bin = h_bin_count[static_cast<std::size_t>(b)];
    const int target = h_target[static_cast<std::size_t>(b)];
    const double E_bin = h_bin_E[static_cast<std::size_t>(b)];
    if (target <= 0) {
      continue;
    }

    if (target >= n_bin) {
      for (int rank = 0; rank < n_bin; ++rank) {
        TENRYU_ASSERT(out_pos < N_target,
                      "census_comb_gpu gather index overflow (non-combed)");
        h_gather_idx[static_cast<std::size_t>(out_pos++)] = bin_start + rank;
      }
      continue;
    }

    std::vector<double> E_clamped(static_cast<std::size_t>(n_bin), 0.0);
    std::vector<int> n_copy(static_cast<std::size_t>(n_bin), 0);
    int n_det = 0;
    for (int i = 0; i < n_bin; ++i) {
      const int src_idx = bin_start + i;
      const double Ei = clamp_energy(h_energy[static_cast<std::size_t>(src_idx)]);
      E_clamped[static_cast<std::size_t>(i)] = Ei;
      const double wi = (E_bin > 0.0) ? (Ei / E_bin) : 0.0;
      const int ni = std::max(
          0,
          static_cast<int>(std::floor(static_cast<double>(target) * wi)));
      n_copy[static_cast<std::size_t>(i)] = ni;
      n_det += ni;
    }

    if (n_det > target) {
      std::vector<int> order(static_cast<std::size_t>(n_bin), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b_idx) {
        const int na = n_copy[static_cast<std::size_t>(a)];
        const int nb = n_copy[static_cast<std::size_t>(b_idx)];
        if (na != nb) {
          return na > nb;
        }
        const double Ea = E_clamped[static_cast<std::size_t>(a)];
        const double Eb = E_clamped[static_cast<std::size_t>(b_idx)];
        if (Ea != Eb) {
          return Ea > Eb;
        }
        return a < b_idx;
      });
      while (n_det > target) {
        bool changed = false;
        for (const int idx : order) {
          int& ni = n_copy[static_cast<std::size_t>(idx)];
          if (ni > 0) {
            --ni;
            --n_det;
            changed = true;
            if (n_det == target) {
              break;
            }
          }
        }
        if (!changed) {
          break;
        }
      }
    }

    const int m_res = target - n_det;
    if (m_res > 0) {
      std::vector<double> residual(static_cast<std::size_t>(n_bin), 0.0);
      for (int i = 0; i < n_bin; ++i) {
        const double wi = (E_bin > 0.0)
                              ? (E_clamped[static_cast<std::size_t>(i)] / E_bin)
                              : 0.0;
        const double ri = static_cast<double>(target) * wi -
                          static_cast<double>(n_copy[static_cast<std::size_t>(i)]);
        residual[static_cast<std::size_t>(i)] = std::max(0.0, ri);
      }
      const std::vector<int> inc = systematic_residual_counts(
          residual,
          m_res,
          seed,
          step,
          h_bin_key[static_cast<std::size_t>(b)]);
      for (int i = 0; i < n_bin; ++i) {
        n_copy[static_cast<std::size_t>(i)] += inc[static_cast<std::size_t>(i)];
      }
    }

    int n_out_bin = 0;
    for (const int ni : n_copy) {
      n_out_bin += ni;
    }
    if (n_out_bin < target) {
      std::vector<int> order(static_cast<std::size_t>(n_bin), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b_idx) {
        const double Ea = E_clamped[static_cast<std::size_t>(a)];
        const double Eb = E_clamped[static_cast<std::size_t>(b_idx)];
        if (Ea != Eb) {
          return Ea > Eb;
        }
        return a < b_idx;
      });
      const int missing = target - n_out_bin;
      for (int i = 0; i < missing; ++i) {
        const int idx = order[static_cast<std::size_t>(i % n_bin)];
        ++n_copy[static_cast<std::size_t>(idx)];
      }
      n_out_bin = target;
    } else if (n_out_bin > target) {
      std::vector<int> order(static_cast<std::size_t>(n_bin), 0);
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&](const int a, const int b_idx) {
        const int na = n_copy[static_cast<std::size_t>(a)];
        const int nb = n_copy[static_cast<std::size_t>(b_idx)];
        if (na != nb) {
          return na > nb;
        }
        return a < b_idx;
      });
      int excess = n_out_bin - target;
      for (const int idx : order) {
        int& ni = n_copy[static_cast<std::size_t>(idx)];
        while (excess > 0 && ni > 0) {
          --ni;
          --excess;
        }
        if (excess == 0) {
          break;
        }
      }
    }

    for (int i = 0; i < n_bin; ++i) {
      const int ni = n_copy[static_cast<std::size_t>(i)];
      for (int c = 0; c < ni; ++c) {
        TENRYU_ASSERT(out_pos < N_target,
                      "census_comb_gpu gather index overflow (combed)");
        const int out_idx = out_pos++;
        h_gather_idx[static_cast<std::size_t>(out_idx)] = bin_start + i;
        if (ni > 1) {
          TENRYU_ASSERT(n_emit_total + dup_offset < kStepLocalLimit,
                        "census_comb_gpu duplicate local id exceeds 2^40");
          TENRYU_ASSERT(step_base_gid <=
                            std::numeric_limits<std::uint64_t>::max() -
                                (n_emit_total + dup_offset),
                        "census_comb_gpu global_id overflow");
          h_gid_override_mask[static_cast<std::size_t>(out_idx)] = 1U;
          h_gid_override_value[static_cast<std::size_t>(out_idx)] =
              step_base_gid + n_emit_total + dup_offset;
          ++dup_offset;
        }
      }
    }
  }
  TENRYU_ASSERT(out_pos == N_target,
                "census_comb_gpu gather index count mismatch");

  if (N_target > 0) {
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gather_idx),
                          sizeof(int) * static_cast<std::size_t>(N_target)),
               "census_comb_gpu cudaMalloc d_gather_idx failed");
    cuda_check(cudaMemcpy(d_gather_idx,
                          h_gather_idx.data(),
                          sizeof(int) * static_cast<std::size_t>(N_target),
                          cudaMemcpyHostToDevice),
               "census_comb_gpu H2D gather_idx failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gid_override_mask),
                          sizeof(std::uint8_t) * static_cast<std::size_t>(N_target)),
               "census_comb_gpu cudaMalloc d_gid_override_mask failed");
    cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_gid_override_value),
                          sizeof(std::uint64_t) * static_cast<std::size_t>(N_target)),
               "census_comb_gpu cudaMalloc d_gid_override_value failed");
    cuda_check(cudaMemcpy(d_gid_override_mask,
                          h_gid_override_mask.data(),
                          sizeof(std::uint8_t) * static_cast<std::size_t>(N_target),
                          cudaMemcpyHostToDevice),
               "census_comb_gpu H2D gid_override_mask failed");
    cuda_check(cudaMemcpy(d_gid_override_value,
                          h_gid_override_value.data(),
                          sizeof(std::uint64_t) * static_cast<std::size_t>(N_target),
                          cudaMemcpyHostToDevice),
               "census_comb_gpu H2D gid_override_value failed");
  }

  PhotonPool temp_pool;
  if (N_target > 0) {
    temp_pool.allocate(N_target);
    const PhotonPoolReadView src_view = make_read_view(pool);
    const PhotonPoolWriteView dst_view = make_write_view(temp_pool);
    const int compact_blocks = (N_target + kThreadsPerBlock - 1) / kThreadsPerBlock;
    gather_equalized_kernel<<<compact_blocks, kThreadsPerBlock>>>(src_view,
                                                                   dst_view,
                                                                   d_gather_idx,
                                                                   d_gid_override_mask,
                                                                   d_gid_override_value,
                                                                   d_bin_target,
                                                                   d_bin_prefix,
                                                                   d_bin_E,
                                                                   d_bin_count,
                                                                   B,
                                                                   N_target);
    cuda_check(cudaGetLastError(),
               "census_comb_gpu launch gather_equalized_kernel failed");
    cuda_check(cudaDeviceSynchronize(),
               "census_comb_gpu cudaDeviceSynchronize(gather) failed");

    d2d_copy(pool.pos_r,
             temp_pool.pos_r,
             N_target,
             "census_comb_gpu copy back pos_r failed");
    d2d_copy(pool.pos_z,
             temp_pool.pos_z,
             N_target,
             "census_comb_gpu copy back pos_z failed");
    d2d_copy(pool.dir_r,
             temp_pool.dir_r,
             N_target,
             "census_comb_gpu copy back dir_r failed");
    d2d_copy(pool.dir_z,
             temp_pool.dir_z,
             N_target,
             "census_comb_gpu copy back dir_z failed");
    d2d_copy(pool.dir_phi,
             temp_pool.dir_phi,
             N_target,
             "census_comb_gpu copy back dir_phi failed");
    d2d_copy(pool.energy,
             temp_pool.energy,
             N_target,
             "census_comb_gpu copy back energy failed");
    d2d_copy(pool.weight,
             temp_pool.weight,
             N_target,
             "census_comb_gpu copy back weight failed");
    d2d_copy(pool.time_remain,
             temp_pool.time_remain,
             N_target,
             "census_comb_gpu copy back time_remain failed");
    d2d_copy(pool.birth_energy,
             temp_pool.birth_energy,
             N_target,
             "census_comb_gpu copy back birth_energy failed");
    d2d_copy(pool.sign,
             temp_pool.sign,
             N_target,
             "census_comb_gpu copy back sign failed");
    d2d_copy(pool.global_id,
             temp_pool.global_id,
             N_target,
             "census_comb_gpu copy back global_id failed");
    d2d_copy(pool.rng_counter,
             temp_pool.rng_counter,
             N_target,
             "census_comb_gpu copy back rng_counter failed");
    d2d_copy(pool.cell_id,
             temp_pool.cell_id,
             N_target,
             "census_comb_gpu copy back cell_id failed");
    d2d_copy(pool.group_id,
             temp_pool.group_id,
             N_target,
             "census_comb_gpu copy back group_id failed");
    d2d_copy(pool.mode,
             temp_pool.mode,
             N_target,
             "census_comb_gpu copy back mode failed");
    d2d_copy(pool.alive,
             temp_pool.alive,
             N_target,
             "census_comb_gpu copy back alive failed");
  }

  if (N_target < n_alive) {
    cuda_check(cudaMemset(pool.alive + N_target,
                          static_cast<int>(kDead),
                          sizeof(std::uint8_t) *
                              static_cast<std::size_t>(n_alive - N_target)),
               "census_comb_gpu cudaMemset tail alive failed");
  }

  add_numerical_loss(d_E_numerical_loss, E_killed_bins);
  cleanup();

  out.n_alive_out = N_target;
  out.n_combed_bins = n_combed_bins;
  out.E_after = E_after;
  out.E_killed_bins = E_killed_bins;

  pool.n_alive = N_target;
  pool.n_census = N_target;
  return out;
}

EssFloorResult census_ess_floor_gpu(PhotonPool& pool,
                                    const int n_alive,
                                    const int n_cells,
                                    const int n_groups,
                                    const CensusCombConfig& cfg,
                                    const std::vector<double>& importance,
                                    const std::uint64_t step_base_gid,
                                    const std::uint64_t n_emit_total,
                                    const std::uint32_t seed,
                                    const int step) {
  TENRYU_ASSERT(n_alive >= 0, "census_ess_floor_gpu requires n_alive >= 0");
  TENRYU_ASSERT(n_cells >= 1, "census_ess_floor_gpu requires n_cells >= 1");
  TENRYU_ASSERT(n_groups >= 1, "census_ess_floor_gpu requires n_groups >= 1");
  TENRYU_ASSERT(cfg.max_split_factor >= 1,
                "census_ess_floor_gpu requires max_split_factor >= 1");
  constexpr std::uint64_t kStepLocalLimit = (1ULL << 40);
  TENRYU_ASSERT(n_emit_total < kStepLocalLimit,
                "census_ess_floor_gpu requires n_emit_total < 2^40");

  EssFloorResult out;
  out.n_alive_out = n_alive;
  if (n_alive == 0 || importance.empty()) {
    pool.n_alive = n_alive;
    pool.n_census = n_alive;
    return out;
  }

  (void)seed;
  (void)step;

  const std::size_t n_keys =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  TENRYU_ASSERT(importance.size() == n_keys,
                "census_ess_floor_gpu importance size mismatch");
  TENRYU_ASSERT(pool.capacity >= n_alive,
                "census_ess_floor_gpu requires capacity >= n_alive");

  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> weight;
  std::vector<double> time_remain;
  std::vector<double> birth_energy;
  std::vector<std::int8_t> sign;
  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;

  d2h(pos_r, pool.pos_r, n_alive, "census_ess_floor_gpu copy pos_r failed");
  d2h(pos_z, pool.pos_z, n_alive, "census_ess_floor_gpu copy pos_z failed");
  d2h(dir_r, pool.dir_r, n_alive, "census_ess_floor_gpu copy dir_r failed");
  d2h(dir_z, pool.dir_z, n_alive, "census_ess_floor_gpu copy dir_z failed");
  d2h(dir_phi,
      pool.dir_phi,
      n_alive,
      "census_ess_floor_gpu copy dir_phi failed");
  d2h(energy, pool.energy, n_alive, "census_ess_floor_gpu copy energy failed");
  d2h(weight, pool.weight, n_alive, "census_ess_floor_gpu copy weight failed");
  d2h(time_remain,
      pool.time_remain,
      n_alive,
      "census_ess_floor_gpu copy time_remain failed");
  d2h(birth_energy,
      pool.birth_energy,
      n_alive,
      "census_ess_floor_gpu copy birth_energy failed");
  d2h(sign, pool.sign, n_alive, "census_ess_floor_gpu copy sign failed");
  d2h(global_id,
      pool.global_id,
      n_alive,
      "census_ess_floor_gpu copy global_id failed");
  d2h(rng_counter,
      pool.rng_counter,
      n_alive,
      "census_ess_floor_gpu copy rng_counter failed");
  d2h(cell_id,
      pool.cell_id,
      n_alive,
      "census_ess_floor_gpu copy cell_id failed");
  d2h(group_id,
      pool.group_id,
      n_alive,
      "census_ess_floor_gpu copy group_id failed");
  d2h(mode, pool.mode, n_alive, "census_ess_floor_gpu copy mode failed");
  d2h(alive, pool.alive, n_alive, "census_ess_floor_gpu copy alive failed");

  auto importance_value = [&](const std::size_t idx) {
    const double v = importance[idx];
    return (std::isfinite(v) && v > 0.0) ? v : 0.0;
  };

  std::vector<std::uint8_t> tier(n_keys, 2U);
  std::vector<int> group_order(static_cast<std::size_t>(n_groups), 0);
  std::iota(group_order.begin(), group_order.end(), 0);
  for (int c = 0; c < n_cells; ++c) {
    const std::size_t cell_base =
        static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups);
    double total_importance = 0.0;
    for (int g = 0; g < n_groups; ++g) {
      total_importance += importance_value(cell_base + static_cast<std::size_t>(g));
    }
    if (!(total_importance > 0.0)) {
      continue;
    }

    std::iota(group_order.begin(), group_order.end(), 0);
    std::sort(group_order.begin(), group_order.end(), [&](const int a, const int b) {
      const double ia = importance_value(cell_base + static_cast<std::size_t>(a));
      const double ib = importance_value(cell_base + static_cast<std::size_t>(b));
      if (ia != ib) {
        return ia > ib;
      }
      return a < b;
    });

    double cumulative = 0.0;
    for (const int g : group_order) {
      cumulative += importance_value(cell_base + static_cast<std::size_t>(g)) /
                    total_importance;
      const std::size_t idx = cell_base + static_cast<std::size_t>(g);
      if (cumulative <= 0.50) {
        tier[idx] = 0U;
      } else if (cumulative <= 0.90) {
        tier[idx] = 1U;
      }
    }
  }

  std::vector<int> bin_count(n_keys, 0);
  std::vector<double> sum_E(n_keys, 0.0);
  std::vector<double> sum_E2(n_keys, 0.0);
  std::vector<double> ess_value(n_keys, 0.0);
  KahanSum E_before_sum;
  for (int i = 0; i < n_alive; ++i) {
    TENRYU_ASSERT(alive[static_cast<std::size_t>(i)] == static_cast<std::uint8_t>(kAlive),
                  "census_ess_floor_gpu requires alive particles in [0, n_alive)");
    const int cell = cell_id[static_cast<std::size_t>(i)];
    const int group = static_cast<int>(group_id[static_cast<std::size_t>(i)]);
    TENRYU_ASSERT(cell >= 0 && cell < n_cells,
                  "census_ess_floor_gpu requires valid cell_id");
    TENRYU_ASSERT(group >= 0 && group < n_groups,
                  "census_ess_floor_gpu requires valid group_id");

    const std::size_t key =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
        static_cast<std::size_t>(group);
    ++bin_count[key];
    const double Ei = clamp_energy(energy[static_cast<std::size_t>(i)]);
    sum_E[key] += Ei;
    sum_E2[key] += Ei * Ei;
    E_before_sum.add(Ei);
  }
  out.E_before = E_before_sum.sum;
  out.E_after = out.E_before;

  std::vector<int> desired_split(n_keys, 1);
  std::int64_t total_new = 0;
  for (std::size_t key = 0; key < n_keys; ++key) {
    if (bin_count[key] <= 0) {
      continue;
    }
    if (tier[key] >= 2U) {
      continue;
    }
    if (!(sum_E[key] > 0.0) || !(sum_E2[key] > 0.0)) {
      continue;
    }

    const double ess = (sum_E[key] * sum_E[key]) / sum_E2[key];
    ess_value[key] = ess;
    const double ess_target = (tier[key] == 0U) ? cfg.ess_min_tier0 : cfg.ess_min_tier1;
    if (!(ess < ess_target)) {
      continue;
    }

    int factor = static_cast<int>(std::ceil(ess_target / std::max(ess, 0.01)));
    factor = std::max(1, std::min(cfg.max_split_factor, factor));
    desired_split[key] = factor;
    total_new += static_cast<std::int64_t>(factor - 1) *
                 static_cast<std::int64_t>(bin_count[key]);
  }

  std::vector<int> split_factor = desired_split;
  const int available_new = std::max(pool.capacity - n_alive, 0);
  if (total_new > static_cast<std::int64_t>(available_new) && total_new > 0) {
    const double scale =
        (available_new > 0)
            ? (static_cast<double>(available_new) / static_cast<double>(total_new))
            : 0.0;
    std::vector<double> fractional_level(n_keys, 0.0);
    std::int64_t scaled_new = 0;
    for (std::size_t key = 0; key < n_keys; ++key) {
      if (desired_split[key] <= 1) {
        split_factor[key] = 1;
        continue;
      }
      const double scaled_levels =
          static_cast<double>(desired_split[key] - 1) * scale;
      const int base_levels =
          std::max(0, static_cast<int>(std::floor(scaled_levels)));
      split_factor[key] = 1 + std::min(desired_split[key] - 1, base_levels);
      fractional_level[key] = scaled_levels - static_cast<double>(base_levels);
      scaled_new += static_cast<std::int64_t>(split_factor[key] - 1) *
                    static_cast<std::int64_t>(bin_count[key]);
    }

    int remaining_new =
        std::max(available_new - static_cast<int>(scaled_new), 0);
    std::vector<std::size_t> order;
    order.reserve(n_keys);
    for (std::size_t key = 0; key < n_keys; ++key) {
      if (split_factor[key] < desired_split[key]) {
        order.push_back(key);
      }
    }
    std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
      if (fractional_level[a] != fractional_level[b]) {
        return fractional_level[a] > fractional_level[b];
      }
      if (tier[a] != tier[b]) {
        return tier[a] < tier[b];
      }
      if (ess_value[a] != ess_value[b]) {
        return ess_value[a] < ess_value[b];
      }
      return a < b;
    });

    bool progressed = true;
    while (remaining_new > 0 && progressed) {
      progressed = false;
      for (const std::size_t key : order) {
        const int cost = bin_count[key];
        if (split_factor[key] >= desired_split[key] || cost > remaining_new) {
          continue;
        }
        ++split_factor[key];
        remaining_new -= cost;
        progressed = true;
        if (remaining_new == 0) {
          break;
        }
      }
    }
  }

  int n_split_bins = 0;
  std::int64_t final_new = 0;
  for (std::size_t key = 0; key < n_keys; ++key) {
    if (split_factor[key] > 1) {
      ++n_split_bins;
      final_new += static_cast<std::int64_t>(split_factor[key] - 1) *
                   static_cast<std::int64_t>(bin_count[key]);
    }
  }

  TENRYU_ASSERT(final_new <= static_cast<std::int64_t>(available_new),
                "census_ess_floor_gpu split allocation exceeded capacity");
  const int n_out = n_alive + static_cast<int>(final_new);
  TENRYU_ASSERT(n_out <= pool.capacity,
                "census_ess_floor_gpu output exceeds pool capacity");

  out.n_split_bins = n_split_bins;
  out.n_alive_out = n_out;
  if (n_out == n_alive) {
    pool.n_alive = n_alive;
    pool.n_census = n_alive;
    return out;
  }

  std::vector<double> pos_r_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> pos_z_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> dir_r_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> dir_z_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> dir_phi_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> energy_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> weight_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> time_remain_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<double> birth_energy_out(static_cast<std::size_t>(n_out), 0.0);
  std::vector<std::int8_t> sign_out(static_cast<std::size_t>(n_out), 1);
  std::vector<std::uint64_t> global_id_out(static_cast<std::size_t>(n_out), 0ULL);
  std::vector<std::uint32_t> rng_counter_out(static_cast<std::size_t>(n_out), 0U);
  std::vector<std::int32_t> cell_id_out(static_cast<std::size_t>(n_out), 0);
  std::vector<std::uint16_t> group_id_out(static_cast<std::size_t>(n_out), 0U);
  std::vector<std::uint8_t> mode_out(static_cast<std::size_t>(n_out), 0U);
  std::vector<std::uint8_t> alive_out(static_cast<std::size_t>(n_out), kAlive);

  std::uint64_t dup_offset = 0ULL;
  int dst = 0;
  for (int i = 0; i < n_alive; ++i) {
    const int cell = cell_id[static_cast<std::size_t>(i)];
    const int group = static_cast<int>(group_id[static_cast<std::size_t>(i)]);
    const std::size_t key =
        static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups) +
        static_cast<std::size_t>(group);
    const int factor = std::max(split_factor[key], 1);
    const double inv_factor = 1.0 / static_cast<double>(factor);
    for (int copy = 0; copy < factor; ++copy) {
      TENRYU_ASSERT(dst < n_out,
                    "census_ess_floor_gpu output fill exceeded expected size");
      const std::size_t src = static_cast<std::size_t>(i);
      const std::size_t out_idx = static_cast<std::size_t>(dst++);
      pos_r_out[out_idx] = pos_r[src];
      pos_z_out[out_idx] = pos_z[src];
      dir_r_out[out_idx] = dir_r[src];
      dir_z_out[out_idx] = dir_z[src];
      dir_phi_out[out_idx] = dir_phi[src];
      time_remain_out[out_idx] = time_remain[src];
      sign_out[out_idx] = sign[src];
      cell_id_out[out_idx] = cell_id[src];
      group_id_out[out_idx] = group_id[src];
      mode_out[out_idx] = mode[src];
      alive_out[out_idx] = static_cast<std::uint8_t>(kAlive);

      if (factor > 1) {
        energy_out[out_idx] = energy[src] * inv_factor;
        weight_out[out_idx] = weight[src] * inv_factor;
        birth_energy_out[out_idx] = birth_energy[src] * inv_factor;
      } else {
        energy_out[out_idx] = energy[src];
        weight_out[out_idx] = weight[src];
        birth_energy_out[out_idx] = birth_energy[src];
      }

      if (copy == 0) {
        global_id_out[out_idx] = global_id[src];
        rng_counter_out[out_idx] = rng_counter[src];
      } else {
        TENRYU_ASSERT(n_emit_total + dup_offset < kStepLocalLimit,
                      "census_ess_floor_gpu duplicate local id exceeds 2^40");
        TENRYU_ASSERT(step_base_gid <=
                          std::numeric_limits<std::uint64_t>::max() -
                              (n_emit_total + dup_offset),
                      "census_ess_floor_gpu global_id overflow");
        global_id_out[out_idx] = step_base_gid + n_emit_total + dup_offset;
        rng_counter_out[out_idx] = 0U;
        ++dup_offset;
      }
    }
  }
  TENRYU_ASSERT(dst == n_out,
                "census_ess_floor_gpu output fill count mismatch");

  h2d(pool.pos_r, pos_r_out, "census_ess_floor_gpu upload pos_r failed");
  h2d(pool.pos_z, pos_z_out, "census_ess_floor_gpu upload pos_z failed");
  h2d(pool.dir_r, dir_r_out, "census_ess_floor_gpu upload dir_r failed");
  h2d(pool.dir_z, dir_z_out, "census_ess_floor_gpu upload dir_z failed");
  h2d(pool.dir_phi, dir_phi_out, "census_ess_floor_gpu upload dir_phi failed");
  h2d(pool.energy, energy_out, "census_ess_floor_gpu upload energy failed");
  h2d(pool.weight, weight_out, "census_ess_floor_gpu upload weight failed");
  h2d(pool.time_remain,
      time_remain_out,
      "census_ess_floor_gpu upload time_remain failed");
  h2d(pool.birth_energy,
      birth_energy_out,
      "census_ess_floor_gpu upload birth_energy failed");
  h2d(pool.sign, sign_out, "census_ess_floor_gpu upload sign failed");
  h2d(pool.global_id,
      global_id_out,
      "census_ess_floor_gpu upload global_id failed");
  h2d(pool.rng_counter,
      rng_counter_out,
      "census_ess_floor_gpu upload rng_counter failed");
  h2d(pool.cell_id, cell_id_out, "census_ess_floor_gpu upload cell_id failed");
  h2d(pool.group_id,
      group_id_out,
      "census_ess_floor_gpu upload group_id failed");
  h2d(pool.mode, mode_out, "census_ess_floor_gpu upload mode failed");
  h2d(pool.alive, alive_out, "census_ess_floor_gpu upload alive failed");

  pool.n_alive = n_out;
  pool.n_census = n_out;
  return out;
}

}  // namespace tenryu::radiation
