#include "radiation/ddmc_transport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/rng/philox_cpu.hpp"
#include "parallel/partition.hpp"

namespace tenryu::radiation {
namespace {

constexpr double kSigmaTotFloor = 1.0e-30;
constexpr double kLeakTol = 1.0e-14;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr int kMaxEventsDdmc = 100000;

void warn_negative_leak_coeff(const std::int64_t cell,
                              const int group,
                              const char* side,
                              const double leak_coeff,
                              const bool clamped_to_zero) {
  static int neg_leak_count = 0;
  if (neg_leak_count++ >= 10) {
    return;
  }

  std::cerr << "[WARN] Negative DDMC leak coefficient in cell " << cell
            << " group " << group << " (" << side << "): " << leak_coeff;
  if (clamped_to_zero) {
    std::cerr << " (clamped to 0)";
  } else {
    std::cerr << " (triggered guard path)";
  }
  std::cerr << '\n';
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename T>
void copy_device_to_host(std::vector<T>& host,
                         const T* device,
                         const int n,
                         const char* message) {
  host.resize(static_cast<std::size_t>(n));
  if (n <= 0) {
    return;
  }
  cuda_check(cudaMemcpy(host.data(),
                        device,
                        sizeof(T) * static_cast<std::size_t>(n),
                        cudaMemcpyDeviceToHost),
             message);
}

template <typename T>
void copy_host_to_device(T* device,
                         const std::vector<T>& host,
                         const char* message) {
  if (host.empty()) {
    return;
  }
  cuda_check(cudaMemcpy(device,
                        host.data(),
                        sizeof(T) * host.size(),
                        cudaMemcpyHostToDevice),
             message);
}

void sample_isotropic_direction_1d(core::rng::PhiloxCpu& rng,
                                   double* dir_r,
                                   double* dir_z,
                                   double* dir_phi) {
  const double mu = 2.0 * rng.uniform() - 1.0;
  const double phi = kTwoPi * rng.uniform();
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  *dir_r = mu;
  *dir_z = sin_theta * std::cos(phi);
  *dir_phi = sin_theta * std::sin(phi);
}

void sample_volume_uniform_position_1d(core::rng::PhiloxCpu& rng,
                                       const double r_lo,
                                       const double r_hi,
                                       double* pos_r,
                                       double* pos_z) {
  const double r_lo3 = r_lo * r_lo * r_lo;
  const double r_hi3 = r_hi * r_hi * r_hi;
  const double xi_r = rng.uniform();
  if (r_hi > r_lo) {
    *pos_r = std::max(std::cbrt(r_lo3 + xi_r * (r_hi3 - r_lo3)), 0.0);
  } else {
    *pos_r = std::max(0.5 * (r_lo + r_hi), 0.0);
  }
  *pos_z = 0.0;
}

}  // namespace

DDMCTransport1D::DDMCTransport1D(const std::int64_t n_cells,
                                 const int n_groups,
                                 const bool interface_exit_half_isotropic)
    : n_cells_(n_cells),
      n_groups_(n_groups),
      interface_exit_half_isotropic_(interface_exit_half_isotropic) {}

std::size_t DDMCTransport1D::index(const std::int64_t cell, const int group) const {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups_) +
         static_cast<std::size_t>(group);
}

void DDMCTransport1D::process_range(PhotonPool& pool,
                                    const int start_index,
                                    const int end_index,
                                    const double dt,
                                    const std::uint64_t step_number,
                                    const std::uint64_t user_seed,
                                    const std::vector<double>& node_r,
                                    const ModeSelector& mode_selector,
                                    const DDMCCoefficients& coefficients,
                                    const std::vector<double>& sigma_a_eff,
                                    const std::vector<double>* const sigma_s_eff,
                                    const std::vector<double>* const eta_cdf,
                                    std::vector<double>& rad_dep,
                                    std::vector<double>& rad_E_tally,
                                    std::vector<double>& E_escape,
                                    double* const E_numerical_loss,
                                    DDMCMomentumEstimator* momentum,
                                    const tenryu::parallel::PartitionInfo* const partition) {
  diagnostics_ = DDMCDiagnostics{};
  TENRYU_ASSERT(mode_selector.n_cells() == n_cells_,
                "DDMC transport cell-count mismatch");
  TENRYU_ASSERT(mode_selector.n_groups() == n_groups_,
                "DDMC transport group-count mismatch");

  const int n_particles = pool.n_alive;
  if (n_particles <= 0 || start_index >= end_index) {
    return;
  }

  std::vector<double> pos_r;
  std::vector<double> pos_z;
  std::vector<double> dir_r;
  std::vector<double> dir_z;
  std::vector<double> dir_phi;
  std::vector<double> energy;
  std::vector<double> time_remain;
  // Note: DDMC does not use particle 'weight' field directly.
  // Weight is only relevant for IMC transport. DDMC operates on
  // particle energy via absorption/re-emission events.

  std::vector<std::uint64_t> global_id;
  std::vector<std::uint32_t> rng_counter;
  std::vector<std::int32_t> cell_id;
  std::vector<std::uint16_t> group_id;
  std::vector<std::uint8_t> mode;
  std::vector<std::uint8_t> alive;

  copy_device_to_host(pos_r, pool.pos_r, n_particles, "ddmc D2H pos_r failed");
  copy_device_to_host(pos_z, pool.pos_z, n_particles, "ddmc D2H pos_z failed");
  copy_device_to_host(dir_r, pool.dir_r, n_particles, "ddmc D2H dir_r failed");
  copy_device_to_host(dir_z, pool.dir_z, n_particles, "ddmc D2H dir_z failed");
  copy_device_to_host(dir_phi, pool.dir_phi, n_particles, "ddmc D2H dir_phi failed");
  copy_device_to_host(energy, pool.energy, n_particles, "ddmc D2H energy failed");
  copy_device_to_host(time_remain,
                      pool.time_remain,
                      n_particles,
                      "ddmc D2H time_remain failed");

  copy_device_to_host(global_id,
                      pool.global_id,
                      n_particles,
                      "ddmc D2H global_id failed");
  copy_device_to_host(rng_counter,
                      pool.rng_counter,
                      n_particles,
                      "ddmc D2H rng_counter failed");
  copy_device_to_host(cell_id, pool.cell_id, n_particles, "ddmc D2H cell_id failed");
  copy_device_to_host(group_id,
                      pool.group_id,
                      n_particles,
                      "ddmc D2H group_id failed");
  copy_device_to_host(mode, pool.mode, n_particles, "ddmc D2H mode failed");
  copy_device_to_host(alive, pool.alive, n_particles, "ddmc D2H alive failed");

  const int begin = std::max(start_index, 0);
  const int end = std::min(end_index, n_particles);
  const int ghost_layers =
      (partition != nullptr) ? std::max(partition->ghost_layers, 0) : 0;
  int nr_local = static_cast<int>(n_cells_) - 2 * ghost_layers;
  if (partition != nullptr && partition->nr_local > 0) {
    nr_local = partition->nr_local;
  }
  nr_local = std::max(nr_local, 0);
  const int owned_begin = ghost_layers;
  const int owned_end = ghost_layers + nr_local;
  const bool use_nlte_scatter =
      sigma_s_eff != nullptr && eta_cdf != nullptr &&
      sigma_s_eff->size() == sigma_a_eff.size() && eta_cdf->size() == sigma_a_eff.size();
  int neighbor_ranks[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  if (partition != nullptr) {
    for (int dir = 0; dir < 8; ++dir) {
      neighbor_ranks[dir] = partition->neighbor_ranks[dir];
    }
  }
  const double nan = std::numeric_limits<double>::quiet_NaN();

  for (int p = begin; p < end; ++p) {
    const std::size_t p_us = static_cast<std::size_t>(p);
    if (alive[static_cast<std::size_t>(p)] != kAlive ||
        mode[static_cast<std::size_t>(p)] != kModeDDMC) {
      continue;
    }

    ++diagnostics_.processed;

    int events = 0;
    auto c = static_cast<std::int64_t>(cell_id[static_cast<std::size_t>(p)]);
    auto g = static_cast<int>(group_id[static_cast<std::size_t>(p)]);

    if (c < 0 || c >= n_cells_ || g < 0 || g >= n_groups_) {
      const double E_bad = energy[static_cast<std::size_t>(p)];
      if (E_bad > 0.0 && E_numerical_loss != nullptr) {
        *E_numerical_loss += E_bad;
      }
      alive[static_cast<std::size_t>(p)] = kDead;
      energy[static_cast<std::size_t>(p)] = 0.0;
      continue;
    }

    double E = energy[static_cast<std::size_t>(p)];
    double t_rem = time_remain[static_cast<std::size_t>(p)];
    if (!std::isfinite(E) || !std::isfinite(t_rem)) {
      if (std::isfinite(E) && E > 0.0 && E_numerical_loss != nullptr) {
        *E_numerical_loss += E;
      }
      alive[static_cast<std::size_t>(p)] = kDead;
      energy[static_cast<std::size_t>(p)] = 0.0;
      time_remain[static_cast<std::size_t>(p)] = 0.0;
      continue;
    }
    if (E <= 0.0) {
      if (E < 0.0 && std::isfinite(E) && E_numerical_loss != nullptr) {
        *E_numerical_loss += -E;
      }
      alive[static_cast<std::size_t>(p)] = kDead;
      energy[static_cast<std::size_t>(p)] = 0.0;
      time_remain[static_cast<std::size_t>(p)] = 0.0;
      continue;
    }
    if (t_rem <= 0.0) {
      t_rem = dt;
    }

    core::rng::PhiloxCpu rng(global_id[static_cast<std::size_t>(p)],
                             user_seed,
                             step_number,
                             rng_counter[static_cast<std::size_t>(p)]);
    bool preserved_census = false;
    auto mark_emigrant = [&](const int face) {
      const std::int64_t src_cell = c;
      if (src_cell >= 0) {
        const std::size_t src_us = static_cast<std::size_t>(src_cell);
        if (src_us + 1 < node_r.size()) {
          pos_r[p_us] = 0.5 * (node_r[src_us] + node_r[src_us + 1]);
        } else {
          pos_r[p_us] = nan;
        }
      } else {
        pos_r[p_us] = nan;
      }
      pos_z[p_us] = 0.0;
      dir_r[p_us] = nan;
      dir_z[p_us] = nan;
      dir_phi[p_us] = nan;
      mode[p_us] = kModeDDMC;
      alive[p_us] = kAlive;
      c = static_cast<std::int64_t>(-(100 + face));
    };

    while (alive[static_cast<std::size_t>(p)] == kAlive &&
           mode[static_cast<std::size_t>(p)] == kModeDDMC &&
           t_rem > 0.0 && events < kMaxEventsDdmc) {
      ++events;

      if (!std::isfinite(E) || !std::isfinite(t_rem)) {
        if (std::isfinite(E) && E > 0.0 && E_numerical_loss != nullptr) {
          *E_numerical_loss += E;
        }
        E = 0.0;
        t_rem = 0.0;
        alive[static_cast<std::size_t>(p)] = kDead;
        break;
      }
      if (E <= 0.0) {
        if (E < 0.0 && std::isfinite(E) && E_numerical_loss != nullptr) {
          *E_numerical_loss += -E;
        }
        E = 0.0;
        t_rem = 0.0;
        alive[static_cast<std::size_t>(p)] = kDead;
        break;
      }

      if (c < 0 || c >= n_cells_ || g < 0 || g >= n_groups_) {
        if (E > 0.0 && E_numerical_loss != nullptr) {
          *E_numerical_loss += E;
        }
        alive[static_cast<std::size_t>(p)] = kDead;
        E = 0.0;
        break;
      }

      const std::size_t cg = index(c, g);
      const auto& cell = coefficients.get_cell_data(c, g);
      const double sigma_a = std::max(sigma_a_eff[cg], 0.0);
      const double sigma_s =
          use_nlte_scatter ? std::max((*sigma_s_eff)[cg], 0.0) : 0.0;
      if (cell.sigma_leak_left < 0.0) {
        warn_negative_leak_coeff(c,
                                 g,
                                 "left",
                                 cell.sigma_leak_left,
                                 cell.sigma_leak_left >= -kLeakTol);
      }
      if (cell.sigma_leak_right < 0.0) {
        warn_negative_leak_coeff(c,
                                 g,
                                 "right",
                                 cell.sigma_leak_right,
                                 cell.sigma_leak_right >= -kLeakTol);
      }
      if (cell.sigma_leak_left < -kLeakTol || cell.sigma_leak_right < -kLeakTol) {
        // Negative DDMC leakage is invalid when transport reaches this point.
        rad_dep[cg] += E;
        E = 0.0;
        alive[static_cast<std::size_t>(p)] = kDead;
        ++diagnostics_.absorbed;
        break;
      }
      const double sigma_leak_left = std::max(cell.sigma_leak_left, 0.0);
      const double sigma_leak_right = std::max(cell.sigma_leak_right, 0.0);
      const bool left_is_vacuum = (cell.bc_left == DDMCBoundaryType::Vacuum);
      const bool right_is_vacuum = (cell.bc_right == DDMCBoundaryType::Vacuum);
      const double sigma_leak_left_vacuum = left_is_vacuum ? sigma_leak_left : 0.0;
      const double sigma_leak_right_vacuum = right_is_vacuum ? sigma_leak_right : 0.0;
      const double sigma_leak_left_out = left_is_vacuum ? 0.0 : sigma_leak_left;
      const double sigma_leak_right_out = right_is_vacuum ? 0.0 : sigma_leak_right;
      const double sigma_leak_bnd = sigma_leak_left_vacuum + sigma_leak_right_vacuum;
      // NUMERICS §7.5: sigma_tot = sigma_a_eff + sigma_leak_left_out +
      // sigma_leak_right_out + sigma_leak_bnd. True NLTE DDMC adds local
      // effective-scatter removal sigma_s_eff and resamples g_out from eta_cdf.
      const double sigma_tot =
          sigma_a + sigma_s + sigma_leak_left_out + sigma_leak_right_out + sigma_leak_bnd;

      if (sigma_tot <= kSigmaTotFloor) {
        rad_E_tally[cg] += tenryu::core::constants::c_light * E * t_rem;
        t_rem = 0.0;
        ++diagnostics_.sigma_tot_zero;
        break;
      }

      const double dt_evt = sample_ddmc_event_time(sigma_tot, rng.uniform());
      const double dt_res = std::min(dt_evt, t_rem);
      rad_E_tally[cg] += tenryu::core::constants::c_light * E * dt_res;

      if (dt_evt >= t_rem) {
        t_rem = 0.0;
        ++diagnostics_.census;
        preserved_census = true;
        break;
      }

      t_rem -= dt_evt;

      const DDMCEvent event =
          select_ddmc_event(sigma_a,
                            sigma_s,
                            sigma_leak_left_out,
                            sigma_leak_right_out,
                            sigma_leak_bnd,
                            sigma_tot,
                            rng.uniform());

      if (event == DDMCEvent::Absorb) {
        rad_dep[cg] += E;
        E = 0.0;
        alive[static_cast<std::size_t>(p)] = kDead;
        ++diagnostics_.absorbed;
        break;
      }

      if (event == DDMCEvent::Scatter) {
        ++diagnostics_.scattered;
        const std::size_t cell_base =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups_);
        const int g_new = sample_group_from_cdf(
            eta_cdf->data() + static_cast<std::ptrdiff_t>(cell_base), n_groups_, rng.uniform());
        if (mode_selector.get_mode(c, g_new) != TransportMode::DDMC) {
          mode[static_cast<std::size_t>(p)] = kModeIMC;
          g = g_new;
          ++diagnostics_.converted_to_imc;
          sample_volume_uniform_position_1d(
              rng,
              node_r[static_cast<std::size_t>(c)],
              node_r[static_cast<std::size_t>(c + 1)],
              &pos_r[static_cast<std::size_t>(p)],
              &pos_z[static_cast<std::size_t>(p)]);
          sample_isotropic_direction_1d(rng,
                                        &dir_r[static_cast<std::size_t>(p)],
                                        &dir_z[static_cast<std::size_t>(p)],
                                        &dir_phi[static_cast<std::size_t>(p)]);
          break;
        }
        g = g_new;
        continue;
      }

      if (event == DDMCEvent::LeakLeft) {
        ++diagnostics_.leak_left;
        if (momentum != nullptr) {
          momentum->tally_face_flux(c, g, -1, E);
        }
        const std::int64_t next = c - 1;
        if (ghost_layers > 0) {
          const bool next_is_ghost =
              next < static_cast<std::int64_t>(owned_begin) ||
              next >= static_cast<std::int64_t>(owned_end);
          if (next_is_ghost && neighbor_ranks[0] >= 0) {
            mark_emigrant(0);
            break;
          } else if (next_is_ghost && neighbor_ranks[0] < 0) {
            // Safety: physical boundary reached via internal BC — absorb as fallback.
            rad_dep[cg] += E;
            E = 0.0;
            alive[static_cast<std::size_t>(p)] = kDead;
            ++diagnostics_.absorbed;
            break;
          }
        }

        if (cell.bc_left == DDMCBoundaryType::Internal) {
          c -= 1;
          continue;
        }
        if (cell.bc_left == DDMCBoundaryType::Interface) {
          c -= 1;
          mode[static_cast<std::size_t>(p)] = kModeIMC;
          ++diagnostics_.converted_to_imc;
          const double xi_mu = rng.uniform();
          const double mu = interface_exit_half_isotropic_
                                ? std::clamp(xi_mu, 0.0, 1.0)
                                : std::sqrt(std::max(xi_mu, 0.0));
          const double xi_phi = rng.uniform();
          const double phi = kTwoPi * xi_phi;
          const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
          if (c >= 0 && c < n_cells_) {
            // c refers to the new (left) cell after c -= 1; interface is its right face.
            pos_r[static_cast<std::size_t>(p)] =
                node_r[static_cast<std::size_t>(c + 1)];
            pos_z[static_cast<std::size_t>(p)] = 0.0;
            dir_r[static_cast<std::size_t>(p)] = -mu;
            dir_z[static_cast<std::size_t>(p)] = sin_theta * std::cos(phi);
            dir_phi[static_cast<std::size_t>(p)] = sin_theta * std::sin(phi);
          } else {
            alive[static_cast<std::size_t>(p)] = kDead;
            E_escape[static_cast<std::size_t>(g)] += E;
            E = 0.0;
          }
          break;
        }
        if (cell.bc_left == DDMCBoundaryType::Vacuum) {
          E_escape[static_cast<std::size_t>(g)] += E;
          alive[static_cast<std::size_t>(p)] = kDead;
          E = 0.0;
          ++diagnostics_.leak_boundary;
          break;
        }
        // Reflective leakage should be unreachable (Sigma_leak=0 at reflective boundaries).
        rad_dep[cg] += E;
        E = 0.0;
        alive[static_cast<std::size_t>(p)] = kDead;
        ++diagnostics_.absorbed;
        break;
      }

      if (event == DDMCEvent::LeakRight) {
        ++diagnostics_.leak_right;
        if (momentum != nullptr) {
          momentum->tally_face_flux(c, g, +1, E);
        }
        const std::int64_t next = c + 1;
        if (ghost_layers > 0) {
          const bool next_is_ghost =
              next < static_cast<std::int64_t>(owned_begin) ||
              next >= static_cast<std::int64_t>(owned_end);
          if (next_is_ghost && neighbor_ranks[1] >= 0) {
            mark_emigrant(1);
            break;
          } else if (next_is_ghost && neighbor_ranks[1] < 0) {
            // Safety: physical boundary reached via internal BC — absorb as fallback.
            rad_dep[cg] += E;
            E = 0.0;
            alive[static_cast<std::size_t>(p)] = kDead;
            ++diagnostics_.absorbed;
            break;
          }
        }

        if (cell.bc_right == DDMCBoundaryType::Internal) {
          c += 1;
          continue;
        }
        if (cell.bc_right == DDMCBoundaryType::Interface) {
          c += 1;
          mode[static_cast<std::size_t>(p)] = kModeIMC;
          ++diagnostics_.converted_to_imc;
          const double xi_mu = rng.uniform();
          const double mu = interface_exit_half_isotropic_
                                ? std::clamp(xi_mu, 0.0, 1.0)
                                : std::sqrt(std::max(xi_mu, 0.0));
          const double xi_phi = rng.uniform();
          const double phi = kTwoPi * xi_phi;
          const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
          if (c >= 0 && c < n_cells_) {
            // c refers to the new (right) cell after c += 1; interface is its left face.
            pos_r[static_cast<std::size_t>(p)] = node_r[static_cast<std::size_t>(c)];
            pos_z[static_cast<std::size_t>(p)] = 0.0;
            dir_r[static_cast<std::size_t>(p)] = mu;
            dir_z[static_cast<std::size_t>(p)] = sin_theta * std::cos(phi);
            dir_phi[static_cast<std::size_t>(p)] = sin_theta * std::sin(phi);
          } else {
            alive[static_cast<std::size_t>(p)] = kDead;
            E_escape[static_cast<std::size_t>(g)] += E;
            E = 0.0;
          }
          break;
        }
        if (cell.bc_right == DDMCBoundaryType::Vacuum) {
          E_escape[static_cast<std::size_t>(g)] += E;
          alive[static_cast<std::size_t>(p)] = kDead;
          E = 0.0;
          ++diagnostics_.leak_boundary;
          break;
        }
        // Reflective leakage should be unreachable (Sigma_leak=0 at reflective boundaries).
        rad_dep[cg] += E;
        E = 0.0;
        alive[static_cast<std::size_t>(p)] = kDead;
        ++diagnostics_.absorbed;
        break;
      }

      if (event == DDMCEvent::LeakBoundary) {
        E_escape[static_cast<std::size_t>(g)] += E;
        alive[static_cast<std::size_t>(p)] = kDead;
        E = 0.0;
        ++diagnostics_.leak_boundary;
        break;
      }

      if (event == DDMCEvent::Census) {
        t_rem = 0.0;
        ++diagnostics_.census;
        preserved_census = true;
        break;
      }
    }

    if (events >= kMaxEventsDdmc && alive[static_cast<std::size_t>(p)] == kAlive &&
        mode[static_cast<std::size_t>(p)] == kModeDDMC && !preserved_census) {
      ++diagnostics_.max_events_reached;
      if (c >= 0 && c < n_cells_ && g >= 0 && g < n_groups_) {
        const std::size_t cg = index(c, g);
        rad_dep[cg] += E;
      } else if (E > 0.0 && E_numerical_loss != nullptr) {
        *E_numerical_loss += E;
      }
      E = 0.0;
      alive[static_cast<std::size_t>(p)] = kDead;
    }

    energy[static_cast<std::size_t>(p)] = E;
    time_remain[static_cast<std::size_t>(p)] = std::max(t_rem, 0.0);
    cell_id[static_cast<std::size_t>(p)] = static_cast<std::int32_t>(c);
    group_id[static_cast<std::size_t>(p)] = static_cast<std::uint16_t>(g);
    rng_counter[static_cast<std::size_t>(p)] = rng.counter();
  }

  copy_host_to_device(pool.pos_r, pos_r, "ddmc H2D pos_r failed");
  copy_host_to_device(pool.pos_z, pos_z, "ddmc H2D pos_z failed");
  copy_host_to_device(pool.dir_r, dir_r, "ddmc H2D dir_r failed");
  copy_host_to_device(pool.dir_z, dir_z, "ddmc H2D dir_z failed");
  copy_host_to_device(pool.dir_phi, dir_phi, "ddmc H2D dir_phi failed");
  copy_host_to_device(pool.energy, energy, "ddmc H2D energy failed");
  copy_host_to_device(pool.time_remain,
                      time_remain,
                      "ddmc H2D time_remain failed");
  copy_host_to_device(pool.rng_counter,
                      rng_counter,
                      "ddmc H2D rng_counter failed");
  copy_host_to_device(pool.cell_id, cell_id, "ddmc H2D cell_id failed");
  copy_host_to_device(pool.group_id, group_id, "ddmc H2D group_id failed");
  copy_host_to_device(pool.mode, mode, "ddmc H2D mode failed");
  copy_host_to_device(pool.alive, alive, "ddmc H2D alive failed");
}

}  // namespace tenryu::radiation
