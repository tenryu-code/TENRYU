#include "burn/mc_transport.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <curand_kernel.h>

#include "burn/burn_constants.hpp"
#include "burn/corman_diffusion.cuh"
#include "core/error.hpp"

namespace tenryu::burn {
namespace {

constexpr int kBlock = 128;
constexpr int kSlots = 6;
constexpr int kTotals = 3;
constexpr int kTotalSourced = 0;
constexpr int kTotalEscaped = 1;
constexpr int kTotalInflight = 2;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
struct DeviceTemp {
  T* ptr = nullptr;

  DeviceTemp() = default;

  DeviceTemp(const std::size_t count, const char* message) {
    if (count > 0U) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(&ptr),
                            count * sizeof(T)),
                 message);
    }
  }

  ~DeviceTemp() {
    if (ptr != nullptr) {
      static_cast<void>(cudaFree(ptr));
      ptr = nullptr;
    }
  }

  DeviceTemp(const DeviceTemp&) = delete;
  DeviceTemp& operator=(const DeviceTemp&) = delete;
};

__host__ __device__ inline double slot_birth_MeV(const int slot) {
  constexpr double t[kSlots] = {3.540, 1.010, 3.023,
                                0.820, 3.690, 14.663};
  return t[slot];
}

__host__ __device__ inline int slot_species_id(const int slot) {
  constexpr int t[kSlots] = {kHe4, kT, kP, kHe3, kHe4, kP};
  return t[slot];
}

__host__ __device__ inline double slot_charge_Z(const int slot) {
  constexpr double t[kSlots] = {2.0, 1.0, 1.0, 2.0, 2.0, 1.0};
  return t[slot];
}

__device__ inline double clamp_unit(const double x) {
  return (x < -1.0) ? -1.0 : ((x > 1.0) ? 1.0 : x);
}

__device__ int locate_cell(const double r, const double* __restrict__ r_node,
                           const int n_cells) {
  if (r <= r_node[0]) {
    return 0;
  }
  if (r >= r_node[n_cells]) {
    return n_cells - 1;
  }
  int lo = 0;
  int hi = n_cells;
  while (lo + 1 < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (r >= r_node[mid]) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (lo < n_cells) ? lo : (n_cells - 1);
}

struct Chord {
  double s = 0.0;
  int face = 1;
};

__device__ Chord next_chord(const double r, const double mu,
                            const double* __restrict__ r_node,
                            const int j) {
  const double mu_c = clamp_unit(mu);
  const double one_minus_mu2 = fmax(0.0, 1.0 - mu_c * mu_c);
  const double b2 = r * r * one_minus_mu2;
  const double b = sqrt(b2);
  const double r_inner = r_node[j];
  const double r_outer = r_node[j + 1];
  Chord c;

  // For j == 0, r_inner == 0 and b < r_inner never triggers. A ray through
  // the origin therefore continues to the outward intersection naturally.
  if (mu_c < 0.0 && b < r_inner) {
    const double root = sqrt(fmax(r_inner * r_inner - b2, 0.0));
    c.s = -r * mu_c - root;
    c.face = -1;
  } else {
    const double root = sqrt(fmax(r_outer * r_outer - b2, 0.0));
    c.s = -r * mu_c + root;
    c.face = 1;
  }
  if (!(c.s >= 0.0)) {
    c.s = 0.0;
  }
  return c;
}

__global__ void spawn_kernel(
    McParams p, int n_cells, unsigned long long step_index,
    const double* __restrict__ r_node, const double* __restrict__ S_birth,
    const double* __restrict__ vol, double dt_s, double* __restrict__ r_p,
    double* __restrict__ mu_p, double* __restrict__ E_p,
    double* __restrict__ w_p, int* __restrict__ slot_p,
    unsigned char* __restrict__ alive_p, int capacity,
    int* __restrict__ live_count, int* __restrict__ overflow,
    double* __restrict__ totals) {
  const std::size_t idx =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(kSlots) *
      static_cast<std::size_t>(p.particles_per_cell);
  if (idx >= total) {
    return;
  }
  const int sample = static_cast<int>(
      idx % static_cast<std::size_t>(p.particles_per_cell));
  const std::size_t cell_slot =
      idx / static_cast<std::size_t>(p.particles_per_cell);
  const int slot = static_cast<int>(cell_slot % kSlots);
  const int cell = static_cast<int>(cell_slot / kSlots);
  const double source = S_birth[slot * n_cells + cell];
  if (!(source > 0.0)) {
    return;
  }
  const double weight =
      source * vol[cell] * dt_s / static_cast<double>(p.particles_per_cell);
  if (!(weight > 0.0)) {
    return;
  }

  const int out = atomicAdd(live_count, 1);
  if (out >= capacity) {
    atomicExch(overflow, 1);
    return;
  }

  const unsigned long long gid =
      mc_transport_global_id(cell, slot, sample, p.particles_per_cell);
  curandStatePhilox4_32_10_t state;
  curand_init(p.seed ^ gid, step_index, 0ULL, &state);
  const double xi_r = curand_uniform_double(&state);
  const double xi_mu = curand_uniform_double(&state);
  const double r_lo = r_node[cell];
  const double r_hi = r_node[cell + 1];
  const double r3 = r_lo * r_lo * r_lo +
                    xi_r * (r_hi * r_hi * r_hi - r_lo * r_lo * r_lo);
  const double E = slot_birth_MeV(slot) * kMeVToErg;

  r_p[out] = cbrt(r3);
  mu_p[out] = 2.0 * xi_mu - 1.0;
  E_p[out] = E;
  w_p[out] = weight;
  slot_p[out] = slot;
  alive_p[out] = 1U;
  atomicAdd(&totals[kTotalSourced], weight * E);
}

__global__ void transport_kernel(
    McParams p, int n_cells, int live_count,
    const double* __restrict__ r_node, const double* __restrict__ rho,
    const double* __restrict__ Te_eV, const double* __restrict__ Ti_eV,
    const double* __restrict__ ne, double dt_s, double* __restrict__ r_p,
    double* __restrict__ mu_p, double* __restrict__ E_p,
    const double* __restrict__ w_p, const int* __restrict__ slot_p,
    unsigned char* __restrict__ alive_p, double* __restrict__ dep_e,
    double* __restrict__ dep_i, double* __restrict__ totals) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= live_count || alive_p[i] == 0U) {
    return;
  }

  double r = r_p[i];
  double mu = clamp_unit(mu_p[i]);
  double E = E_p[i];
  const double weight = w_p[i];
  const int slot = slot_p[i];
  const double A = species_A(slot_species_id(slot));
  const double Z = slot_charge_Z(slot);
  const double mass = A * corman_detail::kProtonMassG;
  const double E_min = p.E_min_keV * corman_detail::kKeVToErg;
  constexpr double tiny_r = 1.0e-300;

  int j = locate_cell(r, r_node, n_cells);
  double dt_rem = dt_s;
  bool live = true;

  while (live && dt_rem > 0.0 && E > 0.0) {
    if (E <= E_min) {
      atomicAdd(&dep_i[j], weight * E);
      E = 0.0;
      live = false;
      break;
    }

    const double lnLe = corman_electron_log(Te_eV[j], ne[j]);
    const double E_keV = E / corman_detail::kKeVToErg;
    const double lnLI =
        corman_ion_log(A, Z, E_keV, rho[j], Te_eV[j], Ti_eV[j], ne[j]);
    const double tE = corman_tE(A, Z, Te_eV[j], ne[j], lnLe);
    const double gamma = corman_gamma(A, Z, rho[j], lnLI);
    const double sqrtE = sqrt(E);
    const double electron =
        (tE > 0.0 && isfinite(tE)) ? (E / tE) : 0.0;
    const double ion = (gamma > 0.0) ? (gamma / sqrtE) : 0.0;
    const double Edot = electron + ion;
    const double v = sqrt(2.0 * E / mass);
    const double ds_E =
        (Edot > 0.0 && v > 0.0)
            ? (p.dE_frac * E / Edot * v)
            : INFINITY;
    const Chord chord = next_chord(r, mu, r_node, j);
    const double s_time = v * dt_rem;
    double s = chord.s;
    bool crossed = true;
    if (ds_E < s) {
      s = ds_E;
      crossed = false;
    }
    if (s_time < s) {
      s = s_time;
      crossed = false;
    }

    double dE = (Edot > 0.0 && v > 0.0) ? (Edot * s / v) : 0.0;
    if (dE > E) {
      dE = E;
    }
    const double fi =
        (Edot > 0.0) ? fmin(1.0, fmax(0.0, ion / Edot)) : 0.0;
    const double e_loss = weight * dE;
    const double e_i = e_loss * fi;
    const double e_e = e_loss - e_i;
    if (e_e != 0.0) {
      atomicAdd(&dep_e[j], e_e);
    }
    if (e_i != 0.0) {
      atomicAdd(&dep_i[j], e_i);
    }

    const double r2_new = fmax(0.0, r * r + s * s + 2.0 * r * s * mu);
    const double r_new = sqrt(r2_new);
    const double mu_new = (s + r * mu) / fmax(r_new, tiny_r);
    r = r_new;
    mu = clamp_unit(mu_new);
    E -= dE;
    if (E < 0.0) {
      E = 0.0;
    }
    if (v > 0.0) {
      dt_rem -= s / v;
    }

    if (crossed && chord.face > 0 && j + 1 >= n_cells) {
      atomicAdd(&totals[kTotalEscaped], weight * E);
      live = false;
      break;
    }
    if (E <= E_min) {
      atomicAdd(&dep_i[j], weight * E);
      E = 0.0;
      live = false;
      break;
    }
    if (crossed) {
      if (chord.face > 0) {
        ++j;
      } else if (j > 0) {
        --j;
      }
    }
  }

  if (live) {
    r_p[i] = r;
    mu_p[i] = mu;
    E_p[i] = E;
    alive_p[i] = 1U;
    atomicAdd(&totals[kTotalInflight], weight * E);
  } else {
    alive_p[i] = 0U;
  }
}

__global__ void compact_kernel(
    int old_count, const int* __restrict__ new_index,
    const double* __restrict__ r_in, const double* __restrict__ mu_in,
    const double* __restrict__ E_in, const double* __restrict__ w_in,
    const int* __restrict__ slot_in, const unsigned char* __restrict__ alive_in,
    double* __restrict__ r_out, double* __restrict__ mu_out,
    double* __restrict__ E_out, double* __restrict__ w_out,
    int* __restrict__ slot_out, unsigned char* __restrict__ alive_out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= old_count || alive_in[i] == 0U) {
    return;
  }
  const int out = new_index[i];
  r_out[out] = r_in[i];
  mu_out[out] = mu_in[i];
  E_out[out] = E_in[i];
  w_out[out] = w_in[i];
  slot_out[out] = slot_in[i];
  alive_out[out] = 1U;
}

void compact_particles(const int old_count, int* live_count_inout,
                       double* r_p, double* mu_p, double* E_p, double* w_p,
                       int* slot_p, unsigned char* alive_p,
                       cudaStream_t stream) {
  if (old_count <= 0) {
    *live_count_inout = 0;
    return;
  }

  std::vector<unsigned char> alive(static_cast<std::size_t>(old_count), 0U);
  cuda_check(cudaMemcpyAsync(alive.data(), alive_p,
                             alive.size() * sizeof(unsigned char),
                             cudaMemcpyDeviceToHost, stream),
             "MC transport copy alive flags failed");
  cuda_check(cudaStreamSynchronize(stream),
             "MC transport alive flag synchronize failed");

  std::vector<int> index(static_cast<std::size_t>(old_count), -1);
  int live_count = 0;
  for (int i = 0; i < old_count; ++i) {
    if (alive[static_cast<std::size_t>(i)] != 0U) {
      index[static_cast<std::size_t>(i)] = live_count;
      ++live_count;
    }
  }
  if (live_count == old_count) {
    *live_count_inout = live_count;
    return;
  }
  if (live_count == 0) {
    *live_count_inout = 0;
    return;
  }

  // v2 perf follow-up: replace this host-assisted path with device compaction.
  DeviceTemp<int> d_index(static_cast<std::size_t>(old_count),
                          "MC transport allocate compaction index failed");
  DeviceTemp<double> d_r(static_cast<std::size_t>(live_count),
                         "MC transport allocate compact r failed");
  DeviceTemp<double> d_mu(static_cast<std::size_t>(live_count),
                          "MC transport allocate compact mu failed");
  DeviceTemp<double> d_E(static_cast<std::size_t>(live_count),
                         "MC transport allocate compact E failed");
  DeviceTemp<double> d_w(static_cast<std::size_t>(live_count),
                         "MC transport allocate compact weight failed");
  DeviceTemp<int> d_slot(static_cast<std::size_t>(live_count),
                         "MC transport allocate compact slot failed");
  DeviceTemp<unsigned char> d_alive(
      static_cast<std::size_t>(live_count),
      "MC transport allocate compact alive failed");

  cuda_check(cudaMemcpyAsync(d_index.ptr, index.data(),
                             index.size() * sizeof(int),
                             cudaMemcpyHostToDevice, stream),
             "MC transport upload compaction index failed");
  const int grid = (old_count + kBlock - 1) / kBlock;
  compact_kernel<<<grid, kBlock, 0, stream>>>(
      old_count, d_index.ptr, r_p, mu_p, E_p, w_p, slot_p, alive_p, d_r.ptr,
      d_mu.ptr, d_E.ptr, d_w.ptr, d_slot.ptr, d_alive.ptr);
  cuda_check(cudaGetLastError(), "MC transport compaction kernel launch failed");

  const std::size_t bytes_d = static_cast<std::size_t>(live_count) * sizeof(double);
  const std::size_t bytes_i = static_cast<std::size_t>(live_count) * sizeof(int);
  const std::size_t bytes_u =
      static_cast<std::size_t>(live_count) * sizeof(unsigned char);
  cuda_check(cudaMemcpyAsync(r_p, d_r.ptr, bytes_d, cudaMemcpyDeviceToDevice,
                             stream),
             "MC transport copy compact r failed");
  cuda_check(cudaMemcpyAsync(mu_p, d_mu.ptr, bytes_d, cudaMemcpyDeviceToDevice,
                             stream),
             "MC transport copy compact mu failed");
  cuda_check(cudaMemcpyAsync(E_p, d_E.ptr, bytes_d, cudaMemcpyDeviceToDevice,
                             stream),
             "MC transport copy compact E failed");
  cuda_check(cudaMemcpyAsync(w_p, d_w.ptr, bytes_d, cudaMemcpyDeviceToDevice,
                             stream),
             "MC transport copy compact weight failed");
  cuda_check(cudaMemcpyAsync(slot_p, d_slot.ptr, bytes_i,
                             cudaMemcpyDeviceToDevice, stream),
             "MC transport copy compact slot failed");
  cuda_check(cudaMemcpyAsync(alive_p, d_alive.ptr, bytes_u,
                             cudaMemcpyDeviceToDevice, stream),
             "MC transport copy compact alive failed");
  cuda_check(cudaStreamSynchronize(stream),
             "MC transport compaction synchronize failed");
  *live_count_inout = live_count;
}

}  // namespace

McStepResult mc_transport_step(
    const McParams& p, const int n_cells, const long long step_index,
    const double* r_node_dev, const double* rho_dev, const double* Te_eV_dev,
    const double* Ti_eV_dev, const double* ne_dev, const double* S_birth_dev,
    const double* vol_dev, const double dt_s, double* r_p, double* mu_p,
    double* E_p, double* w_p, int* slot_p, unsigned char* alive_p,
    const int capacity, int* live_count_inout, double* dep_e_dev,
    double* dep_i_dev, cudaStream_t stream) {
  McStepResult result;
  if (n_cells <= 0) {
    return result;
  }

  TENRYU_ASSERT(p.particles_per_cell > 0,
                "MC transport particles_per_cell must be positive");
  TENRYU_ASSERT(p.E_min_keV > 0.0 && p.dE_frac > 0.0,
                "MC transport energy controls are invalid");
  TENRYU_ASSERT(step_index >= 0, "MC transport step_index must be nonnegative");
  TENRYU_ASSERT(dt_s >= 0.0, "MC transport dt must be nonnegative");
  TENRYU_ASSERT(capacity >= 0 && live_count_inout != nullptr,
                "MC transport capacity/count are invalid");
  TENRYU_ASSERT(*live_count_inout >= 0 && *live_count_inout <= capacity,
                "MC transport live count outside pool capacity");
  TENRYU_ASSERT(r_node_dev != nullptr && rho_dev != nullptr &&
                    Te_eV_dev != nullptr && Ti_eV_dev != nullptr &&
                    ne_dev != nullptr && S_birth_dev != nullptr &&
                    vol_dev != nullptr && r_p != nullptr && mu_p != nullptr &&
                    E_p != nullptr && w_p != nullptr && slot_p != nullptr &&
                    alive_p != nullptr && dep_e_dev != nullptr &&
                    dep_i_dev != nullptr,
                "MC transport received a null pointer");

  DeviceTemp<int> d_live_count(1U, "MC transport allocate live count failed");
  DeviceTemp<int> d_overflow(1U, "MC transport allocate overflow flag failed");
  DeviceTemp<double> d_totals(static_cast<std::size_t>(kTotals),
                              "MC transport allocate totals failed");
  const int initial_count = *live_count_inout;
  const int zero_i = 0;
  cuda_check(cudaMemcpyAsync(d_live_count.ptr, &initial_count, sizeof(int),
                             cudaMemcpyHostToDevice, stream),
             "MC transport upload live count failed");
  cuda_check(cudaMemcpyAsync(d_overflow.ptr, &zero_i, sizeof(int),
                             cudaMemcpyHostToDevice, stream),
             "MC transport upload overflow flag failed");
  cuda_check(cudaMemsetAsync(d_totals.ptr, 0, kTotals * sizeof(double), stream),
             "MC transport zero totals failed");

  const std::size_t birth_threads =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(kSlots) *
      static_cast<std::size_t>(p.particles_per_cell);
  TENRYU_ASSERT(birth_threads <=
                    static_cast<std::size_t>(std::numeric_limits<int>::max()),
                "MC transport birth launch exceeds supported grid size");
  if (birth_threads > 0U) {
    const int grid =
        static_cast<int>((birth_threads + kBlock - 1U) / kBlock);
    spawn_kernel<<<grid, kBlock, 0, stream>>>(
        p, n_cells, static_cast<unsigned long long>(step_index), r_node_dev,
        S_birth_dev, vol_dev, dt_s, r_p, mu_p, E_p, w_p, slot_p, alive_p,
        capacity, d_live_count.ptr, d_overflow.ptr, d_totals.ptr);
    cuda_check(cudaGetLastError(), "MC transport spawn kernel launch failed");
  }

  int post_spawn_count = 0;
  int overflow = 0;
  cuda_check(cudaMemcpyAsync(&post_spawn_count, d_live_count.ptr, sizeof(int),
                             cudaMemcpyDeviceToHost, stream),
             "MC transport copy spawned live count failed");
  cuda_check(cudaMemcpyAsync(&overflow, d_overflow.ptr, sizeof(int),
                             cudaMemcpyDeviceToHost, stream),
             "MC transport copy overflow flag failed");
  cuda_check(cudaStreamSynchronize(stream),
             "MC transport spawn synchronize failed");
  result.overflow = (overflow != 0);
  TENRYU_ASSERT(!result.overflow,
                "MC transport particle pool capacity overflow");
  TENRYU_ASSERT(post_spawn_count <= capacity,
                "MC transport spawned live count exceeds capacity");

  if (post_spawn_count > 0) {
    const int grid = (post_spawn_count + kBlock - 1) / kBlock;
    transport_kernel<<<grid, kBlock, 0, stream>>>(
        p, n_cells, post_spawn_count, r_node_dev, rho_dev, Te_eV_dev,
        Ti_eV_dev, ne_dev, dt_s, r_p, mu_p, E_p, w_p, slot_p, alive_p,
        dep_e_dev, dep_i_dev, d_totals.ptr);
    cuda_check(cudaGetLastError(),
               "MC transport transport kernel launch failed");
  }

  compact_particles(post_spawn_count, live_count_inout, r_p, mu_p, E_p, w_p,
                    slot_p, alive_p, stream);

  double totals[kTotals] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpyAsync(totals, d_totals.ptr, sizeof(totals),
                             cudaMemcpyDeviceToHost, stream),
             "MC transport copy totals failed");
  cuda_check(cudaStreamSynchronize(stream),
             "MC transport totals synchronize failed");

  result.sourced_erg = totals[kTotalSourced];
  result.escaped_erg = totals[kTotalEscaped];
  result.inflight_erg = totals[kTotalInflight];
  result.live_particles = *live_count_inout;
  return result;
}

}  // namespace tenryu::burn
