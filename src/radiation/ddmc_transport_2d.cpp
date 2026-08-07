#include "radiation/ddmc_transport.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "core/rng/philox_cpu.hpp"
#include "parallel/partition.hpp"
#include "face_geometry_2d.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kSigmaTotFloor = 1.0e-30;
constexpr double kLeakTol = 1.0e-14;
constexpr double kTwoPi = 6.28318530717958647692;
constexpr double kGeomEps = 1.0e-12;
constexpr int kMaxEventsDdmc = 100000;

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

int node_index_2d_rz(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

struct CellVertices2D {
  double r00 = 0.0;
  double z00 = 0.0;
  double r10 = 0.0;
  double z10 = 0.0;
  double r11 = 0.0;
  double z11 = 0.0;
  double r01 = 0.0;
  double z01 = 0.0;
};

CellVertices2D get_cell_vertices_2d(const std::int64_t cell,
                                    const int nz,
                                    const std::vector<double>& node_r,
                                    const std::vector<double>& node_z) {
  const int i = static_cast<int>(cell) / nz;
  const int j = static_cast<int>(cell) - i * nz;
  const int n00 = node_index_2d_rz(i, j, nz);
  const int n10 = node_index_2d_rz(i + 1, j, nz);
  const int n11 = node_index_2d_rz(i + 1, j + 1, nz);
  const int n01 = node_index_2d_rz(i, j + 1, nz);
  CellVertices2D vertices{};
  vertices.r00 = node_r[static_cast<std::size_t>(n00)];
  vertices.z00 = node_z[static_cast<std::size_t>(n00)];
  vertices.r10 = node_r[static_cast<std::size_t>(n10)];
  vertices.z10 = node_z[static_cast<std::size_t>(n10)];
  vertices.r11 = node_r[static_cast<std::size_t>(n11)];
  vertices.z11 = node_z[static_cast<std::size_t>(n11)];
  vertices.r01 = node_r[static_cast<std::size_t>(n01)];
  vertices.z01 = node_z[static_cast<std::size_t>(n01)];
  return vertices;
}

void cell_center_2d(const CellVertices2D& vertices, double* const r, double* const z) {
  *r = 0.25 * (vertices.r00 + vertices.r10 + vertices.r11 + vertices.r01);
  *z = 0.25 * (vertices.z00 + vertices.z10 + vertices.z11 + vertices.z01);
}

double normal_sign_toward_point(const FaceGeom2D& geom,
                                const double target_r,
                                const double target_z) {
  const double z_mid = 0.5 * (geom.z1 + geom.z2);
  const double dot =
      (target_r - geom.r_mid) * geom.nr + (target_z - z_mid) * geom.nz;
  return (dot >= 0.0) ? 1.0 : -1.0;
}

double sample_face_param_r_weighted(const FaceGeom2D& geom, const double xi) {
  const double R1 = geom.r1;
  const double R2 = geom.r2;
  const double eps_R =
      1.0e-10 * std::max({std::abs(R1), std::abs(R2), 1.0e-20});
  const double xi_clamped = std::clamp(xi, 0.0, 1.0);
  if (std::abs(R2 - R1) < eps_R) {
    return xi_clamped;
  }

  const double R1_sq = R1 * R1;
  const double R2_sq = R2 * R2;
  const double term = std::max(R1_sq + xi_clamped * (R2_sq - R1_sq), 0.0);
  const double t = (-R1 + std::sqrt(term)) / (R2 - R1);
  return std::clamp(t, 0.0, 1.0);
}

void bilinear_map(const double eta,
                  const double zeta,
                  const double r00,
                  const double z00,
                  const double r10,
                  const double z10,
                  const double r11,
                  const double z11,
                  const double r01,
                  const double z01,
                  double* r_out,
                  double* z_out) {
  const double n00 = (1.0 - eta) * (1.0 - zeta);
  const double n10 = eta * (1.0 - zeta);
  const double n11 = eta * zeta;
  const double n01 = (1.0 - eta) * zeta;
  *r_out = n00 * r00 + n10 * r10 + n11 * r11 + n01 * r01;
  *z_out = n00 * z00 + n10 * z10 + n11 * z11 + n01 * z01;
}

void sample_isotropic_direction_2d(core::rng::PhiloxCpu& rng,
                                   double* dir_r,
                                   double* dir_z,
                                   double* dir_phi) {
  const double mu_z = 2.0 * rng.uniform() - 1.0;
  const double phi = kTwoPi * rng.uniform();
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu_z * mu_z));
  *dir_r = sin_theta * std::cos(phi);
  *dir_z = mu_z;
  *dir_phi = sin_theta * std::sin(phi);
}

void sample_volume_uniform_position_2d(core::rng::PhiloxCpu& rng,
                                       const CellVertices2D& vertices,
                                       double* pos_r,
                                       double* pos_z) {
  const double r_max_cell = std::max({vertices.r00, vertices.r10, vertices.r11, vertices.r01});
  double r_p = 0.25 * (vertices.r00 + vertices.r10 + vertices.r11 + vertices.r01);
  double z_p = 0.25 * (vertices.z00 + vertices.z10 + vertices.z11 + vertices.z01);
  bool accepted = false;
  for (int n_try = 0; n_try < 64; ++n_try) {
    const double xi_eta = rng.uniform();
    const double xi_zeta = rng.uniform();
    const double xi_reject = rng.uniform();
    bilinear_map(xi_eta,
                 xi_zeta,
                 vertices.r00,
                 vertices.z00,
                 vertices.r10,
                 vertices.z10,
                 vertices.r11,
                 vertices.z11,
                 vertices.r01,
                 vertices.z01,
                 &r_p,
                 &z_p);
    const double w = (r_max_cell > 0.0) ? std::clamp(r_p / r_max_cell, 0.0, 1.0) : 1.0;
    if (xi_reject <= w) {
      accepted = true;
      break;
    }
  }
  if (!accepted) {
    r_p = std::max(r_p, 0.0);
  }
  *pos_r = std::max(r_p, 0.0);
  *pos_z = z_p;
}

}  // namespace

DDMCTransport2D::DDMCTransport2D(const std::int64_t n_cells,
                                 const int n_groups,
                                 const int nr,
                                 const int nz,
                                 const bool interface_exit_half_isotropic)
    : n_cells_(n_cells),
      n_groups_(n_groups),
      nr_(nr),
      nz_(nz),
      interface_exit_half_isotropic_(interface_exit_half_isotropic) {}

std::size_t DDMCTransport2D::index(const std::int64_t cell, const int group) const {
  return static_cast<std::size_t>(cell) * static_cast<std::size_t>(n_groups_) +
         static_cast<std::size_t>(group);
}

int DDMCTransport2D::neighbor_from_face(const std::int64_t cell, const int face) const {
  if (cell < 0 || cell >= n_cells_) {
    return -1;
  }
  const int c = static_cast<int>(cell);
  const int i = c / nz_;
  const int j = c - i * nz_;
  if (face == 0) {
    return (i > 0) ? ((i - 1) * nz_ + j) : -1;
  }
  if (face == 1) {
    return (i + 1 < nr_) ? ((i + 1) * nz_ + j) : -1;
  }
  if (face == 2) {
    return (j > 0) ? (i * nz_ + (j - 1)) : -1;
  }
  if (face == 3) {
    return (j + 1 < nz_) ? (i * nz_ + (j + 1)) : -1;
  }
  return -1;
}

void DDMCTransport2D::sample_ddmc_to_imc_direction(const double xi_mu,
                                                   const double xi_phi,
                                                   const double nr,
                                                   const double nz,
                                                   const double tr,
                                                   const double tz,
                                                   double* dir_r,
                                                   double* dir_z,
                                                   double* dir_phi) const {
  const double mu = interface_exit_half_isotropic_
                        ? std::clamp(xi_mu, 0.0, 1.0)
                        : std::sqrt(std::max(xi_mu, 0.0));
  const double phi = kTwoPi * xi_phi;
  const double sin_theta = std::sqrt(std::max(0.0, 1.0 - mu * mu));
  const double tangent = sin_theta * std::cos(phi);
  *dir_r = mu * nr + tangent * tr;
  *dir_z = mu * nz + tangent * tz;
  *dir_phi = sin_theta * std::sin(phi);
}

void DDMCTransport2D::process_range(PhotonPool& pool,
                                    const int start_index,
                                    const int end_index,
                                    const double dt,
                                    const std::uint64_t step_number,
                                    const std::uint64_t user_seed,
                                    const std::vector<double>& node_r,
                                    const std::vector<double>& node_z,
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
                "DDMC transport2d cell-count mismatch");
  TENRYU_ASSERT(mode_selector.n_groups() == n_groups_,
                "DDMC transport2d group-count mismatch");

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

  copy_device_to_host(pos_r, pool.pos_r, n_particles, "ddmc2d D2H pos_r failed");
  copy_device_to_host(pos_z, pool.pos_z, n_particles, "ddmc2d D2H pos_z failed");
  copy_device_to_host(dir_r, pool.dir_r, n_particles, "ddmc2d D2H dir_r failed");
  copy_device_to_host(dir_z, pool.dir_z, n_particles, "ddmc2d D2H dir_z failed");
  copy_device_to_host(dir_phi, pool.dir_phi, n_particles, "ddmc2d D2H dir_phi failed");
  copy_device_to_host(energy, pool.energy, n_particles, "ddmc2d D2H energy failed");
  copy_device_to_host(time_remain,
                      pool.time_remain,
                      n_particles,
                      "ddmc2d D2H time_remain failed");

  copy_device_to_host(global_id,
                      pool.global_id,
                      n_particles,
                      "ddmc2d D2H global_id failed");
  copy_device_to_host(rng_counter,
                      pool.rng_counter,
                      n_particles,
                      "ddmc2d D2H rng_counter failed");
  copy_device_to_host(cell_id, pool.cell_id, n_particles, "ddmc2d D2H cell_id failed");
  copy_device_to_host(group_id,
                      pool.group_id,
                      n_particles,
                      "ddmc2d D2H group_id failed");
  copy_device_to_host(mode, pool.mode, n_particles, "ddmc2d D2H mode failed");
  copy_device_to_host(alive, pool.alive, n_particles, "ddmc2d D2H alive failed");

  const int begin = std::max(start_index, 0);
  const int end = std::min(end_index, n_particles);
  const int ghost_layers =
      (partition != nullptr) ? std::max(partition->ghost_layers, 0) : 0;
  int nr_local = nr_ - 2 * ghost_layers;
  int nz_local = nz_ - 2 * ghost_layers;
  if (partition != nullptr && partition->nr_local > 0) {
    nr_local = partition->nr_local;
  }
  if (partition != nullptr && partition->nz_local > 0) {
    nz_local = partition->nz_local;
  }
  nr_local = std::max(nr_local, 0);
  nz_local = std::max(nz_local, 0);
  const int owned_r_begin = ghost_layers;
  const int owned_r_end = ghost_layers + nr_local;
  const int owned_z_begin = ghost_layers;
  const int owned_z_end = ghost_layers + nz_local;
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
    if (alive[p_us] != kAlive || mode[p_us] != kModeDDMC) {
      continue;
    }

    ++diagnostics_.processed;
    int events = 0;
    auto c = static_cast<std::int64_t>(cell_id[p_us]);
    auto g = static_cast<int>(group_id[p_us]);

    if (c < 0 || c >= n_cells_ || g < 0 || g >= n_groups_) {
      const double E_bad = energy[p_us];
      if (E_bad > 0.0 && E_numerical_loss != nullptr) {
        *E_numerical_loss += E_bad;
      }
      alive[p_us] = kDead;
      energy[p_us] = 0.0;
      continue;
    }

    double E = energy[p_us];
    double t_rem = time_remain[p_us];
    if (!std::isfinite(E) || !std::isfinite(t_rem)) {
      if (std::isfinite(E) && E > 0.0 && E_numerical_loss != nullptr) {
        *E_numerical_loss += E;
      }
      alive[p_us] = kDead;
      energy[p_us] = 0.0;
      time_remain[p_us] = 0.0;
      continue;
    }
    if (E <= 0.0) {
      if (E < 0.0 && std::isfinite(E) && E_numerical_loss != nullptr) {
        *E_numerical_loss += -E;
      }
      alive[p_us] = kDead;
      energy[p_us] = 0.0;
      time_remain[p_us] = 0.0;
      continue;
    }
    if (t_rem <= 0.0) {
      t_rem = dt;
    }

    core::rng::PhiloxCpu rng(global_id[p_us],
                             user_seed,
                             step_number,
                             rng_counter[p_us]);
    bool preserved_census = false;
    auto mark_emigrant = [&](const int face) {
      if (c >= 0 && c < n_cells_) {
        const CellVertices2D src_vertices = get_cell_vertices_2d(c, nz_, node_r, node_z);
        cell_center_2d(src_vertices, &pos_r[p_us], &pos_z[p_us]);
      } else {
        pos_r[p_us] = nan;
        pos_z[p_us] = nan;
      }
      dir_r[p_us] = nan;
      dir_z[p_us] = nan;
      dir_phi[p_us] = nan;
      mode[p_us] = kModeDDMC;
      alive[p_us] = kAlive;
      c = static_cast<std::int64_t>(-(100 + face));
    };

    while (alive[p_us] == kAlive && mode[p_us] == kModeDDMC && t_rem > 0.0 &&
           events < kMaxEventsDdmc) {
      ++events;

      if (!std::isfinite(E) || !std::isfinite(t_rem)) {
        if (std::isfinite(E) && E > 0.0 && E_numerical_loss != nullptr) {
          *E_numerical_loss += E;
        }
        E = 0.0;
        t_rem = 0.0;
        alive[p_us] = kDead;
        break;
      }
      if (E <= 0.0) {
        if (E < 0.0 && std::isfinite(E) && E_numerical_loss != nullptr) {
          *E_numerical_loss += -E;
        }
        E = 0.0;
        t_rem = 0.0;
        alive[p_us] = kDead;
        break;
      }

      if (c < 0 || c >= n_cells_ || g < 0 || g >= n_groups_) {
        if (E > 0.0 && E_numerical_loss != nullptr) {
          *E_numerical_loss += E;
        }
        alive[p_us] = kDead;
        E = 0.0;
        break;
      }

      const std::size_t cg = index(c, g);
      const auto& cell = coefficients.get_cell_data(c, g);
      const double sigma_a = std::max(sigma_a_eff[cg], 0.0);
      const double sigma_s =
          use_nlte_scatter ? std::max((*sigma_s_eff)[cg], 0.0) : 0.0;
      double sigma_tot = sigma_a + sigma_s;
      for (int face = 0; face < CellDDMCData::kFaceCount; ++face) {
        if (cell.sigma_leak_face[face] < -kLeakTol) {
          rad_dep[cg] += E;
          E = 0.0;
          alive[p_us] = kDead;
          ++diagnostics_.absorbed;
          break;
        }
        sigma_tot += std::max(cell.sigma_leak_face[face], 0.0);
      }
      if (alive[p_us] != kAlive) {
        break;
      }

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

      const double xi = rng.uniform() * sigma_tot;
      if (xi < sigma_a) {
        rad_dep[cg] += E;
        E = 0.0;
        alive[p_us] = kDead;
        ++diagnostics_.absorbed;
        break;
      }
      if (use_nlte_scatter && xi < sigma_a + sigma_s) {
        ++diagnostics_.scattered;
        const std::size_t cell_base =
            static_cast<std::size_t>(c) * static_cast<std::size_t>(n_groups_);
        const int g_new = sample_group_from_cdf(
            eta_cdf->data() + static_cast<std::ptrdiff_t>(cell_base), n_groups_, rng.uniform());
        if (mode_selector.get_mode(c, g_new) != TransportMode::DDMC) {
          mode[p_us] = kModeIMC;
          g = g_new;
          ++diagnostics_.converted_to_imc;
          const CellVertices2D sample_vertices = get_cell_vertices_2d(c, nz_, node_r, node_z);
          sample_volume_uniform_position_2d(
              rng, sample_vertices, &pos_r[p_us], &pos_z[p_us]);
          sample_isotropic_direction_2d(
              rng, &dir_r[p_us], &dir_z[p_us], &dir_phi[p_us]);
          break;
        }
        g = g_new;
        continue;
      }
      double cdf = sigma_a + sigma_s;
      int selected_face = -1;
      for (int face = 0; face < CellDDMCData::kFaceCount; ++face) {
        cdf += std::max(cell.sigma_leak_face[face], 0.0);
        if (xi <= cdf + 1.0e-15) {
          selected_face = face;
          break;
        }
      }

      if (selected_face < 0) {
        rad_dep[cg] += E;
        E = 0.0;
        alive[p_us] = kDead;
        ++diagnostics_.absorbed;
        break;
      }

      if (selected_face == 0) {
        ++diagnostics_.leak_left;
      } else if (selected_face == 1) {
        ++diagnostics_.leak_right;
      } else if (selected_face == 2) {
        ++diagnostics_.leak_face2;
      } else {
        ++diagnostics_.leak_face3;
      }

      if (momentum != nullptr) {
        if (selected_face == 0) {
          momentum->tally_face_flux(c, g, -1, E);
        } else if (selected_face == 1) {
          momentum->tally_face_flux(c, g, +1, E);
        } else if (selected_face == 2) {
          momentum->tally_face_flux(c, g, -1, E);
        } else if (selected_face == 3) {
          momentum->tally_face_flux(c, g, +1, E);
        }
      }

      int next = cell.neighbor_face[selected_face];
      if (next < 0 || next >= n_cells_) {
        next = neighbor_from_face(c, selected_face);
      }
      if (ghost_layers > 0 && next >= 0 && next < n_cells_) {
        const int ni = next / nz_;
        const int nj = next - ni * nz_;
        const bool next_is_ghost =
            ni < owned_r_begin || ni >= owned_r_end || nj < owned_z_begin ||
            nj >= owned_z_end;
        if (next_is_ghost && neighbor_ranks[selected_face] >= 0) {
          mark_emigrant(selected_face);
          break;
        } else if (next_is_ghost && neighbor_ranks[selected_face] < 0) {
          // Safety: physical boundary reached via internal BC - should be
          // unreachable (sigma_leak=0 at reflective). Absorb as fallback.
          rad_dep[cg] += E;
          E = 0.0;
          alive[p_us] = kDead;
          ++diagnostics_.absorbed;
          break;
        }
      }

      const DDMCBoundaryType bc = cell.bc_face[selected_face];
      if (bc == DDMCBoundaryType::Vacuum) {
        E_escape[static_cast<std::size_t>(g)] += E;
        alive[p_us] = kDead;
        E = 0.0;
        ++diagnostics_.leak_boundary;
        break;
      }

      if (bc == DDMCBoundaryType::Reflective) {
        const CellVertices2D reflect_vertices =
            get_cell_vertices_2d(c, nz_, node_r, node_z);
        const FaceGeom2D reflect_geom =
            compute_face_geom(selected_face,
                              reflect_vertices.r00,
                              reflect_vertices.z00,
                              reflect_vertices.r10,
                              reflect_vertices.z10,
                              reflect_vertices.r11,
                              reflect_vertices.z11,
                              reflect_vertices.r01,
                              reflect_vertices.z01);
        TENRYU_ASSERT(reflect_geom.length > 0.0,
                      "DDMC2D invalid reflective face geometry");
        reflect_direction_2d(
            dir_r[p_us], dir_z[p_us], reflect_geom.nr, reflect_geom.nz);
        continue;
      }

      if (next < 0 || next >= n_cells_) {
        E_escape[static_cast<std::size_t>(g)] += E;
        alive[p_us] = kDead;
        E = 0.0;
        ++diagnostics_.leak_boundary;
        break;
      }

      if (bc == DDMCBoundaryType::Internal) {
        c = static_cast<std::int64_t>(next);
        continue;
      }

      // DDMC->IMC conversion through an interface face.
      const CellVertices2D leak_vertices =
          get_cell_vertices_2d(c, nz_, node_r, node_z);
      const FaceGeom2D leak_geom =
          compute_face_geom(selected_face,
                            leak_vertices.r00,
                            leak_vertices.z00,
                            leak_vertices.r10,
                            leak_vertices.z10,
                            leak_vertices.r11,
                            leak_vertices.z11,
                            leak_vertices.r01,
                            leak_vertices.z01);
      TENRYU_ASSERT(leak_geom.length > 0.0,
                    "DDMC2D invalid interface face geometry");
      const std::int64_t imc_cell = static_cast<std::int64_t>(next);
      mode[p_us] = kModeIMC;
      ++diagnostics_.converted_to_imc;

      const CellVertices2D imc_vertices =
          get_cell_vertices_2d(imc_cell, nz_, node_r, node_z);
      double imc_r_center = 0.0;
      double imc_z_center = 0.0;
      cell_center_2d(imc_vertices, &imc_r_center, &imc_z_center);
      const double normal_sign =
          normal_sign_toward_point(leak_geom, imc_r_center, imc_z_center);
      const double nr_to_imc = normal_sign * leak_geom.nr;
      const double nz_to_imc = normal_sign * leak_geom.nz;

      const double t_face = sample_face_param_r_weighted(leak_geom, rng.uniform());
      pos_r[p_us] = leak_geom.r1 + t_face * (leak_geom.r2 - leak_geom.r1);
      pos_z[p_us] = leak_geom.z1 + t_face * (leak_geom.z2 - leak_geom.z1);
      push_off_face_2d(
          pos_r[p_us], pos_z[p_us], nr_to_imc, nz_to_imc, kGeomEps);
      pos_r[p_us] = std::max(pos_r[p_us], 0.0);

      sample_ddmc_to_imc_direction(rng.uniform(),
                                   rng.uniform(),
                                   nr_to_imc,
                                   nz_to_imc,
                                   leak_geom.tr,
                                   leak_geom.tz,
                                   &dir_r[p_us],
                                   &dir_z[p_us],
                                   &dir_phi[p_us]);
      c = imc_cell;
      break;
    }

    if (events >= kMaxEventsDdmc && alive[p_us] == kAlive &&
        mode[p_us] == kModeDDMC && !preserved_census) {
      ++diagnostics_.max_events_reached;
      if (c >= 0 && c < n_cells_ && g >= 0 && g < n_groups_) {
        const std::size_t cg = index(c, g);
        rad_dep[cg] += E;
      } else if (E > 0.0 && E_numerical_loss != nullptr) {
        *E_numerical_loss += E;
      }
      E = 0.0;
      alive[p_us] = kDead;
    }

    energy[p_us] = E;
    time_remain[p_us] = std::max(t_rem, 0.0);
    cell_id[p_us] = static_cast<std::int32_t>(c);
    group_id[p_us] = static_cast<std::uint16_t>(g);
    rng_counter[p_us] = rng.counter();
  }

  copy_host_to_device(pool.pos_r, pos_r, "ddmc2d H2D pos_r failed");
  copy_host_to_device(pool.pos_z, pos_z, "ddmc2d H2D pos_z failed");
  copy_host_to_device(pool.dir_r, dir_r, "ddmc2d H2D dir_r failed");
  copy_host_to_device(pool.dir_z, dir_z, "ddmc2d H2D dir_z failed");
  copy_host_to_device(pool.dir_phi, dir_phi, "ddmc2d H2D dir_phi failed");
  copy_host_to_device(pool.energy, energy, "ddmc2d H2D energy failed");
  copy_host_to_device(pool.time_remain, time_remain, "ddmc2d H2D time_remain failed");
  copy_host_to_device(pool.rng_counter, rng_counter, "ddmc2d H2D rng_counter failed");
  copy_host_to_device(pool.cell_id, cell_id, "ddmc2d H2D cell_id failed");
  copy_host_to_device(pool.group_id, group_id, "ddmc2d H2D group_id failed");
  copy_host_to_device(pool.mode, mode, "ddmc2d H2D mode failed");
  copy_host_to_device(pool.alive, alive, "ddmc2d H2D alive failed");
}

}  // namespace tenryu::radiation
