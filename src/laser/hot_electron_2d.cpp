#include "laser/hot_electron_2d.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::laser::hot_electron {
namespace {

double norm3(const double v[3]) {
  return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void normalize3(double v[3]) {
  const double n = norm3(v);
  if (n > 0.0) {
    v[0] /= n;
    v[1] /= n;
    v[2] /= n;
  } else {
    v[0] = 0.0;
    v[1] = 0.0;
    v[2] = 1.0;
  }
}

void cross3(const double a[3], const double b[3], double c[3]) {
  c[0] = a[1] * b[2] - a[2] * b[1];
  c[1] = a[2] * b[0] - a[0] * b[2];
  c[2] = a[0] * b[1] - a[1] * b[0];
}

int default_max_segments_2d(const MeshView2D& view) {
  const int n_side =
      static_cast<int>(std::ceil(std::sqrt(static_cast<double>(view.n_cells))));
  return 8 * std::max(64, n_side);
}

int chi_bin_from_k(const double kR, const double kZ) {
  const double chi = std::atan2(kR, kZ);
  int bin = static_cast<int>((chi + kPi) / (2.0 * kPi) * kChiBins);
  if (bin < 0) {
    bin = 0;
  }
  if (bin >= kChiBins) {
    bin = kChiBins - 1;
  }
  return bin;
}

struct ReducedCaptureEntry {
  int cell = -1;
  int chi_bin = 0;
  int kphi_bin = 0;
  double R = 0.0;
  double Z = 0.0;
  double k[3] = {0.0, 0.0, 1.0};
  double P = 0.0;
};

bool entry_less(const ReducedCaptureEntry& a, const ReducedCaptureEntry& b) {
  if (a.cell != b.cell) return a.cell < b.cell;
  if (a.chi_bin != b.chi_bin) return a.chi_bin < b.chi_bin;
  if (a.kphi_bin != b.kphi_bin) return a.kphi_bin < b.kphi_bin;
  if (a.R != b.R) return a.R < b.R;
  if (a.Z != b.Z) return a.Z < b.Z;
  if (a.k[0] != b.k[0]) return a.k[0] < b.k[0];
  if (a.k[1] != b.k[1]) return a.k[1] < b.k[1];
  if (a.k[2] != b.k[2]) return a.k[2] < b.k[2];
  return a.P < b.P;
}

struct SegmentLocal {
  double rho = 0.0;
  double ne = 0.0;
  double Te_erg = 0.0;
  double E_floor = 0.0;
};

}  // namespace

MeshView2D MeshView2DStorage::view(const double* node_R, const double* node_Z,
                                   const int n_cells) const {
  MeshView2D out;
  out.n_cells = n_cells;
  out.cell_nodes = cell_nodes.data();
  out.cell_neighbor = cell_neighbor.data();
  out.node_R = node_R;
  out.node_Z = node_Z;
  return out;
}

MeshView2DStorage build_single_block_view(const int nr, const int nz) {
  TENRYU_ASSERT(nr > 0 && nz > 0, "hot_electron 2d view requires positive extents");
  MeshView2DStorage storage;
  const int n_cells = nr * nz;
  storage.cell_nodes.assign(static_cast<std::size_t>(n_cells) * 4, 0);
  storage.cell_neighbor.assign(static_cast<std::size_t>(n_cells) * 4, -1);
  const auto node_id = [nz](const int i, const int j) {
    return i * (nz + 1) + j;
  };
  const auto cell_id = [nz](const int i, const int j) {
    return i * nz + j;
  };
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = cell_id(i, j);
      storage.cell_nodes[static_cast<std::size_t>(c) * 4 + 0] = node_id(i, j);
      storage.cell_nodes[static_cast<std::size_t>(c) * 4 + 1] = node_id(i + 1, j);
      storage.cell_nodes[static_cast<std::size_t>(c) * 4 + 2] = node_id(i + 1, j + 1);
      storage.cell_nodes[static_cast<std::size_t>(c) * 4 + 3] = node_id(i, j + 1);

      storage.cell_neighbor[static_cast<std::size_t>(c) * 4 + 0] =
          (j > 0) ? cell_id(i, j - 1) : -1;
      storage.cell_neighbor[static_cast<std::size_t>(c) * 4 + 1] =
          (i + 1 < nr) ? cell_id(i + 1, j) : -1;
      storage.cell_neighbor[static_cast<std::size_t>(c) * 4 + 2] =
          (j + 1 < nz) ? cell_id(i, j + 1) : -1;
      storage.cell_neighbor[static_cast<std::size_t>(c) * 4 + 3] =
          (i > 0) ? cell_id(i - 1, j) : -1;
    }
  }
  return storage;
}

std::vector<DirNode> build_band_dirs_3d(const double axis_in[3], double mu_lo,
                                        double mu_hi, const int n_mu,
                                        const int n_phi) {
  std::vector<DirNode> nodes;
  double axis[3] = {axis_in[0], axis_in[1], axis_in[2]};
  normalize3(axis);
  if (mu_hi > 1.0) {
    mu_hi = 1.0;
  }
  if (mu_lo < -1.0) {
    mu_lo = -1.0;
  }
  TENRYU_ASSERT(mu_lo <= mu_hi, "hot_electron 3d band dirs require mu_lo <= mu_hi");

  const double zhat[3] = {0.0, 0.0, 1.0};
  double e1[3] = {0.0, 0.0, 0.0};
  cross3(axis, zhat, e1);
  if (norm3(e1) > 1.0e-12) {
    normalize3(e1);
  } else {
    e1[0] = 1.0;
    e1[1] = 0.0;
    e1[2] = 0.0;
  }
  double e2[3] = {0.0, 0.0, 0.0};
  cross3(axis, e1, e2);
  normalize3(e2);

  const auto append_node = [&](const double mu, const double phi, const double weight) {
    const double smu = std::sqrt(std::max(1.0 - mu * mu, 0.0));
    DirNode node;
    const double cp = std::cos(phi);
    const double sp = std::sin(phi);
    for (int k = 0; k < 3; ++k) {
      node.om[k] = mu * axis[k] + smu * (cp * e1[k] + sp * e2[k]);
    }
    normalize3(node.om);
    node.weight = weight;
    nodes.push_back(node);
  };

  if (!(mu_lo < mu_hi)) {
    if (!(mu_lo < 1.0)) {
      DirNode node;
      node.om[0] = axis[0];
      node.om[1] = axis[1];
      node.om[2] = axis[2];
      node.weight = 1.0;
      nodes.push_back(node);
      return nodes;
    }
    const double mu = mu_lo;
    nodes.reserve(static_cast<std::size_t>(n_phi));
    for (int jp = 0; jp < n_phi; ++jp) {
      const double phi =
          2.0 * kPi * (static_cast<double>(jp) + 0.5) / static_cast<double>(n_phi);
      append_node(mu, phi, 1.0 / static_cast<double>(n_phi));
    }
    return nodes;
  }

  std::vector<double> gx;
  std::vector<double> gw;
  gauss_legendre(n_mu, gx, gw);
  const double half_span = 0.5 * (mu_hi - mu_lo);
  const double mid = 0.5 * (mu_hi + mu_lo);
  nodes.reserve(static_cast<std::size_t>(n_mu) * static_cast<std::size_t>(n_phi));
  for (int i = 0; i < n_mu; ++i) {
    const double mu = mid + half_span * gx[static_cast<std::size_t>(i)];
    const double wmu = gw[static_cast<std::size_t>(i)] * 0.5;
    for (int jp = 0; jp < n_phi; ++jp) {
      const double phi =
          2.0 * kPi * (static_cast<double>(jp) + 0.5) / static_cast<double>(n_phi);
      append_node(mu, phi, wmu / static_cast<double>(n_phi));
    }
  }
  return nodes;
}

ReduceResult2D reduce_captures_2d(const std::vector<RayCapture2D>& caps,
                                  const MeshView2D& v) {
  ReduceResult2D result;
  if (caps.empty() || v.n_cells <= 0) {
    return result;
  }
  std::vector<ReducedCaptureEntry> entries;
  entries.reserve(caps.size());
  for (const RayCapture2D& cap : caps) {
    if (!(cap.P_hot > 0.0)) {
      continue;
    }
    const int cell = locate_cell_2d(v, cap.R_s, cap.Z_s, 0);
    if (cell < 0) {
      result.P_locate_failed += cap.P_hot;
      ++result.n_locate_failed;
      continue;
    }
    ReducedCaptureEntry entry;
    entry.cell = cell;
    entry.chi_bin = chi_bin_from_k(cap.kR, cap.kZ);
    entry.kphi_bin = (std::abs(cap.kphi) < kPhiClassEdge) ? 0 : 1;
    entry.R = cap.R_s;
    entry.Z = cap.Z_s;
    entry.k[0] = cap.kR;
    entry.k[1] = cap.kphi;
    entry.k[2] = cap.kZ;
    normalize3(entry.k);
    entry.P = cap.P_hot;
    entries.push_back(entry);
  }
  std::sort(entries.begin(), entries.end(), entry_less);

  struct Bin {
    double P = 0.0;
    double PR = 0.0;
    double PZ = 0.0;
    double Pk[3] = {0.0, 0.0, 0.0};
    double maxP = -1.0;
    double maxK[3] = {0.0, 0.0, 1.0};
  };
  std::vector<Bin> bins(static_cast<std::size_t>(v.n_cells) * kChiBins * 2);
  for (const ReducedCaptureEntry& entry : entries) {
    const std::size_t idx =
        (static_cast<std::size_t>(entry.cell) * kChiBins +
         static_cast<std::size_t>(entry.chi_bin)) *
            2 +
        static_cast<std::size_t>(entry.kphi_bin);
    Bin& bin = bins[idx];
    bin.P += entry.P;
    bin.PR += entry.P * entry.R;
    bin.PZ += entry.P * entry.Z;
    for (int k = 0; k < 3; ++k) {
      bin.Pk[k] += entry.P * entry.k[k];
    }
    if (entry.P > bin.maxP) {
      bin.maxP = entry.P;
      bin.maxK[0] = entry.k[0];
      bin.maxK[1] = entry.k[1];
      bin.maxK[2] = entry.k[2];
    }
  }

  for (int cell = 0; cell < v.n_cells; ++cell) {
    for (int chi = 0; chi < kChiBins; ++chi) {
      for (int kphi = 0; kphi < 2; ++kphi) {
        const std::size_t idx =
            (static_cast<std::size_t>(cell) * kChiBins + static_cast<std::size_t>(chi)) *
                2 +
            static_cast<std::size_t>(kphi);
        const Bin& bin = bins[idx];
        if (!(bin.P > 0.0)) {
          continue;
        }
        HotESource2D src;
        src.cell = cell;
        src.R_s = bin.PR / bin.P;
        src.Z_s = bin.PZ / bin.P;
        src.k[0] = bin.Pk[0] / bin.P;
        src.k[1] = bin.Pk[1] / bin.P;
        src.k[2] = bin.Pk[2] / bin.P;
        if (norm3(src.k) < 1.0e-12) {
          src.k[0] = bin.maxK[0];
          src.k[1] = bin.maxK[1];
          src.k[2] = bin.maxK[2];
        }
        normalize3(src.k);
        src.P_hot = bin.P;
        result.sources.push_back(src);
      }
    }
  }
  return result;
}

DepositResult2D deposit_hot_electrons_cone_2d(
    const HotEChannelSpec& spec,
    const std::vector<HotESource2D>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const MeshView2D& view,
    const double mesh_scale,
    std::vector<double>& dep_power_cell,
    int max_segments) {
  DepositResult2D result;
  TENRYU_ASSERT(view.n_cells > 0, "hot_electron 2d cone requires cells");
  TENRYU_ASSERT(view.cell_nodes != nullptr, "hot_electron 2d cone missing cell nodes");
  TENRYU_ASSERT(view.cell_neighbor != nullptr, "hot_electron 2d cone missing neighbors");
  TENRYU_ASSERT(view.node_R != nullptr, "hot_electron 2d cone missing node R");
  TENRYU_ASSERT(view.node_Z != nullptr, "hot_electron 2d cone missing node Z");
  const std::size_t n = static_cast<std::size_t>(view.n_cells);
  TENRYU_ASSERT(rho.size() == n, "hot_electron 2d cone rho size mismatch");
  TENRYU_ASSERT(zbar.size() == n, "hot_electron 2d cone zbar size mismatch");
  TENRYU_ASSERT(A_eff.size() == n, "hot_electron 2d cone A_eff size mismatch");
  TENRYU_ASSERT(Te_eV.size() == n, "hot_electron 2d cone Te_eV size mismatch");
  TENRYU_ASSERT(cell_is_void.size() == n, "hot_electron 2d cone cell_is_void size mismatch");
  TENRYU_ASSERT(dep_power_cell.size() == n, "hot_electron 2d cone dep_power_cell size mismatch");

  double PR = 0.0;
  for (const HotESource2D& source : sources) {
    result.P_hot += source.P_hot;
    PR += source.P_hot * source.R_s;
  }
  if (sources.empty() || !(result.P_hot > 0.0)) {
    return result;
  }
  result.n_sources = static_cast<int>(sources.size());
  result.r_source_mean = PR / result.P_hot;

  const double T_h_erg = spec.T_hot_erg;
  const std::vector<GroupSpec> groups =
      build_groups(T_h_erg, spec.n_energy_groups, spec.E_min_over_Th,
                   spec.E_max_over_Th);
  const double kProtonMass = core::constants::proton_mass;
  std::vector<SegmentLocal> locals(n);
  for (std::size_t c = 0; c < n; ++c) {
    locals[c].rho = rho[c];
    locals[c].Te_erg = std::max(Te_eV[c], 0.0) * core::constants::eV_to_erg;
    locals[c].E_floor = std::max(2.0 * locals[c].Te_erg, 1.0e-3 * T_h_erg);
    if (A_eff[c] > 0.0 && rho[c] > 0.0) {
      locals[c].ne = rho[c] * std::max(zbar[c], 0.0) / (A_eff[c] * kProtonMass);
    }
  }

  if (max_segments <= 0) {
    max_segments = default_max_segments_2d(view);
  }
  for (const HotESource2D& source : sources) {
    TENRYU_ASSERT(source.cell >= 0 && source.cell < view.n_cells,
                  "hot_electron 2d source cell out of range");
    const std::vector<DirNode> dirs =
        build_band_dirs_3d(source.k, spec.mu_lo, spec.mu_hi, spec.n_mu, spec.n_phi);
    for (const DirNode& dir : dirs) {
      const double P_chord = source.P_hot * dir.weight;
      if (!(P_chord > 0.0)) {
        continue;
      }
      const Chord3D chord(source.R_s, source.Z_s, dir.om);
      ChordWalkerRZ walker(view, chord, source.cell, mesh_scale, max_segments);
      std::vector<std::pair<int, double>> segments;
      segments.reserve(static_cast<std::size_t>(max_segments));
      bool exited = false;
      bool stuck = false;
      int terminal_cell = source.cell;
      for (;;) {
        double ds = 0.0;
        int cell = -1;
        const WalkStatus status = walker.next(ds, cell);
        if (status == WalkStatus::Segment) {
          if (cell >= 0 && cell < view.n_cells && ds > 0.0) {
            segments.push_back({cell, ds});
          }
          if (walker.exited_boundary) {
            exited = true;
            break;
          }
          continue;
        }
        if (status == WalkStatus::Stuck) {
          stuck = true;
          terminal_cell = cell;
        } else {
          exited = true;
        }
        break;
      }
      result.walker_relocations += walker.relocate_attempts;
      if (stuck) {
        ++result.walker_stuck_terminations;
      }

      for (const GroupSpec& g : groups) {
        if (!(g.weight > 0.0) || !(g.E_rep > 0.0)) {
          continue;
        }
        const double Ndot = P_chord * g.weight / g.E_rep;
        double E = g.E_rep;
        std::vector<double> row(segments.size(), 0.0);
        for (std::size_t iseg = 0; iseg < segments.size(); ++iseg) {
          const int cell = segments[iseg].first;
          const std::size_t c = static_cast<std::size_t>(cell);
          if (cell_is_void[c]) {
            continue;
          }
          const SegmentLocal& local = locals[c];
          const double dSigma = local.rho * segments[iseg].second;
          const auto stopping = [&](const double Ee) -> double {
            return stopping_power_erg_cm2_per_g(Ee, local.ne, local.Te_erg, local.rho);
          };
          const double E_out =
              march_cell(E, dSigma, stopping, local.E_floor,
                         kSubstepEnergyFraction, kMaxSubstepsPerCell,
                         &result.substep_cap_hits);
          row[iseg] += Ndot * (E - E_out);
          E = E_out;
          if (!(E > 0.0)) {
            break;
          }
        }
        for (std::size_t iseg = 0; iseg < row.size(); ++iseg) {
          if (row[iseg] != 0.0) {
            const int cell = segments[iseg].first;
            dep_power_cell[static_cast<std::size_t>(cell)] += row[iseg];
            result.P_deposited += row[iseg];
          }
        }
        if (E > 0.0) {
          const double residual = Ndot * E;
          if (stuck) {
            TENRYU_ASSERT(terminal_cell >= 0 && terminal_cell < view.n_cells,
                          "hot_electron 2d stuck terminal cell out of range");
            dep_power_cell[static_cast<std::size_t>(terminal_cell)] += residual;
            result.P_deposited += residual;
          } else if (exited) {
            result.P_escaped += residual;
          }
        }
      }
    }
  }
  result.conservation_resid =
      std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
      std::max(result.P_hot, 1.0e-300);
  if (result.conservation_resid > 1.0e-8) {
    core::log_warning("hot_electron 2d cone conservation check failed");
  }
  result.active = true;
  return result;
}

}  // namespace tenryu::laser::hot_electron
