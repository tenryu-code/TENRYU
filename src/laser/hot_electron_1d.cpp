#include "laser/hot_electron_1d.cuh"

#include <algorithm>

#include "core/constants.hpp"
#include "core/error.hpp"

namespace tenryu::laser::hot_electron {
namespace {
inline double phi1(const double u) { return (1.0 + u) * std::exp(-u); }
inline double phi2(const double u) { return (2.0 + 2.0 * u + u * u) * std::exp(-u); }
inline double moment1(const double a, const double b, const double T) {
  return T * T * (phi1(a / T) - phi1(b / T));
}
inline double moment2(const double a, const double b, const double T) {
  return T * T * T * (phi2(a / T) - phi2(b / T));
}
}  // namespace

std::vector<GroupSpec> build_groups(const double T_hot_erg, const int n_groups,
                                    const double e_min_over_th, const double e_max_over_th) {
  std::vector<GroupSpec> groups;
  if (!(T_hot_erg > 0.0) || n_groups < 1 || !(e_min_over_th > 0.0) ||
      !(e_max_over_th > e_min_over_th)) {
    return groups;
  }
  const double E_min = e_min_over_th * T_hot_erg;
  const double E_max = e_max_over_th * T_hot_erg;
  const double window = moment1(E_min, E_max, T_hot_erg);
  if (!(window > 0.0)) {
    return groups;
  }
  groups.resize(static_cast<std::size_t>(n_groups));
  const double ratio = E_max / E_min;
  double e_lo = E_min;
  for (int g = 0; g < n_groups; ++g) {
    const double e_hi =
        (g + 1 == n_groups)
            ? E_max
            : E_min * std::pow(ratio, static_cast<double>(g + 1) / static_cast<double>(n_groups));
    const double I_g = moment1(e_lo, e_hi, T_hot_erg);
    GroupSpec spec;
    spec.weight = I_g / window;
    spec.E_rep = (I_g > 0.0) ? moment2(e_lo, e_hi, T_hot_erg) / I_g : std::sqrt(e_lo * e_hi);
    groups[static_cast<std::size_t>(g)] = spec;
    e_lo = e_hi;
  }
  return groups;
}

void gauss_legendre(const int n, std::vector<double>& x, std::vector<double>& w) {
  x.assign(static_cast<std::size_t>(n), 0.0);
  w.assign(static_cast<std::size_t>(n), 0.0);
  const int m = (n + 1) / 2;
  for (int i = 0; i < m; ++i) {
    double z = std::cos(kPi * (static_cast<double>(i) + 0.75) /
                        (static_cast<double>(n) + 0.5));
    double pp = 0.0;
    for (int iter = 0; iter < 100; ++iter) {
      double p1 = 1.0;
      double p2 = 0.0;
      for (int k = 0; k < n; ++k) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * k + 1.0) * z * p2 - static_cast<double>(k) * p3) /
             (static_cast<double>(k) + 1.0);
      }
      pp = static_cast<double>(n) * (z * p1 - p2) / (z * z - 1.0);
      const double z1 = z;
      z = z1 - p1 / pp;
      if (std::abs(z - z1) < 1.0e-15) {
        break;
      }
    }
    x[static_cast<std::size_t>(i)] = -z;
    x[static_cast<std::size_t>(n - 1 - i)] = z;
    const double wi = 2.0 / ((1.0 - z * z) * pp * pp);
    w[static_cast<std::size_t>(i)] = wi;
    w[static_cast<std::size_t>(n - 1 - i)] = wi;
  }
}

std::vector<AngleNode> build_band_nodes(const double mu_axis, double mu_lo,
                                        double mu_hi, const int n_mu, const int n_phi) {
  std::vector<AngleNode> nodes;
  if (mu_hi > 1.0) {
    mu_hi = 1.0;
  }
  if (mu_lo < -1.0) {
    mu_lo = -1.0;
  }
  TENRYU_ASSERT(mu_lo <= mu_hi, "hot_electron band nodes require mu_lo <= mu_hi");
  const double sa = std::sqrt(std::max(1.0 - mu_axis * mu_axis, 0.0));
  if (!(mu_lo < mu_hi)) {
    if (!(mu_lo < 1.0)) {
      nodes.push_back(AngleNode{mu_axis, 1.0});
      return nodes;
    }
    const double mu = mu_lo;
    const double smu = std::sqrt(std::max(1.0 - mu * mu, 0.0));
    nodes.reserve(static_cast<std::size_t>(n_phi));
    for (int jp = 0; jp < n_phi; ++jp) {
      const double phi = 2.0 * kPi * (static_cast<double>(jp) + 0.5) /
                         static_cast<double>(n_phi);
      const double mu_dir = mu * mu_axis + smu * sa * std::cos(phi);
      nodes.push_back(AngleNode{mu_dir, 1.0 / static_cast<double>(n_phi)});
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
    const double wmu = gw[static_cast<std::size_t>(i)] * 0.5;  // GL weights sum to 2
    const double smu = std::sqrt(std::max(1.0 - mu * mu, 0.0));
    for (int jp = 0; jp < n_phi; ++jp) {
      const double phi = 2.0 * kPi * (static_cast<double>(jp) + 0.5) /
                         static_cast<double>(n_phi);
      const double mu_dir = mu * mu_axis + smu * sa * std::cos(phi);
      nodes.push_back(AngleNode{mu_dir, wmu / static_cast<double>(n_phi)});
    }
  }
  return nodes;
}

std::vector<AngleNode> build_cone_nodes(const double mu_axis, const double theta_div_rad,
                                        const int n_mu, const int n_phi) {
  const double mu0 = std::cos(theta_div_rad);
  return build_band_nodes(mu_axis, mu0, 1.0, n_mu, n_phi);
}

namespace {
inline double band_mu_from_deg(const double angle_deg) {
  return std::cos(angle_deg * kPi / 180.0);
}
}  // namespace

HotEChannelSpec make_channel_spec(
    const core::Config::LaserConfig::HotElectronConfig& he,
    const core::Config::LaserConfig::HotEChannelConfig& channel) {
  HotEChannelSpec spec;
  spec.T_hot_erg = channel.T_hot_eV * core::constants::eV_to_erg;
  spec.n_energy_groups = channel.n_energy_groups;
  spec.E_min_over_Th = channel.E_min_over_Th;
  spec.E_max_over_Th = channel.E_max_over_Th;
  if (channel.mechanism == "tpd") {
    double ang_lo_deg = channel.tpd_theta_deg - channel.tpd_delta_deg;
    if (ang_lo_deg < 0.0) {
      ang_lo_deg = 0.0;
    }
    double ang_hi_deg = channel.tpd_theta_deg + channel.tpd_delta_deg;
    if (ang_hi_deg > 180.0) {
      ang_hi_deg = 180.0;
    }
    spec.mu_lo = band_mu_from_deg(ang_hi_deg);
    spec.mu_hi = (ang_lo_deg > 0.0) ? band_mu_from_deg(ang_lo_deg) : 1.0;
  } else {
    spec.mu_lo = band_mu_from_deg(channel.theta_div_deg);
    spec.mu_hi = 1.0;
  }
  spec.n_mu = channel.n_mu;
  spec.n_phi = channel.n_phi;
  spec.inner_escape = (he.inner_bc == "escape");
  return spec;
}

HotEChannelSpec make_channel_spec_from_shorthand(
    const core::Config::LaserConfig::HotElectronConfig& he) {
  HotEChannelSpec spec;
  spec.T_hot_erg = he.T_hot_eV * core::constants::eV_to_erg;
  spec.n_energy_groups = he.n_energy_groups;
  spec.E_min_over_Th = he.E_min_over_Th;
  spec.E_max_over_Th = he.E_max_over_Th;
  spec.mu_lo = band_mu_from_deg(he.theta_div_deg);
  spec.mu_hi = 1.0;
  spec.n_mu = he.n_mu;
  spec.n_phi = he.n_phi;
  spec.inner_escape = (he.inner_bc == "escape");
  return spec;
}

std::vector<HotESource> reduce_captures(const std::vector<RayCapture>& captures,
                                        const std::vector<double>& r_nodes) {
  std::vector<HotESource> out;
  if (captures.empty() || r_nodes.size() < 2) {
    return out;
  }
  const int n_cells = static_cast<int>(r_nodes.size()) - 1;
  struct Bin {
    double P = 0.0;
    double P_mu = 0.0;
    double P_r = 0.0;
  };
  std::vector<Bin> bins(static_cast<std::size_t>(n_cells) * kAxisBins);
  long long out_of_domain = 0;
  double out_of_domain_power = 0.0;
  for (const RayCapture& cap : captures) {
    if (!(cap.P_hot > 0.0)) {
      continue;
    }
    if (cap.r_s < r_nodes.front() || cap.r_s > r_nodes.back()) {
      // Captures outside the hydro mesh (e.g. in the ghost corona beyond the
      // outer edge) are clamped into the boundary cell below; count them so
      // the handoff is visible instead of silent (AI review k10-S3).
      ++out_of_domain;
      out_of_domain_power += cap.P_hot;
    }
    int c = n_cells - 1;
    while (c > 0 && cap.r_s < r_nodes[static_cast<std::size_t>(c)]) {
      --c;
    }
    double mu = cap.mu_axis;
    if (mu < -1.0) mu = -1.0;
    if (mu > 1.0) mu = 1.0;
    int b = static_cast<int>((mu + 1.0) * 0.5 * kAxisBins);
    if (b >= kAxisBins) b = kAxisBins - 1;
    if (b < 0) b = 0;
    Bin& bin = bins[static_cast<std::size_t>(c) * kAxisBins + static_cast<std::size_t>(b)];
    bin.P += cap.P_hot;
    bin.P_mu += cap.P_hot * mu;
    bin.P_r += cap.P_hot * cap.r_s;
  }
  if (out_of_domain > 0) {
    core::log_warning("hot_electron reduce_captures: " +
                      std::to_string(out_of_domain) +
                      " capture(s) outside the hydro mesh (" +
                      std::to_string(out_of_domain_power) +
                      " erg/s) clamped into the boundary cell");
  }
  for (int c = 0; c < n_cells; ++c) {
    for (int b = 0; b < kAxisBins; ++b) {
      const Bin& bin = bins[static_cast<std::size_t>(c) * kAxisBins + static_cast<std::size_t>(b)];
      if (!(bin.P > 0.0)) {
        continue;
      }
      HotESource src;
      src.cell = c;
      src.r_s = bin.P_r / bin.P;
      src.mu_axis = bin.P_mu / bin.P;
      src.P_hot = bin.P;
      out.push_back(src);
    }
  }
  return out;
}

DepositResult deposit_hot_electrons_cone_1d(
    const HotEChannelSpec& spec,
    const Geometry1D geom,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell) {
  DepositResult result;
  TENRYU_ASSERT(r_nodes.size() >= 2, "hot_electron cone requires at least two nodes");
  const std::size_t n = r_nodes.size() - 1;
  const int n_nodes = static_cast<int>(r_nodes.size());
  TENRYU_ASSERT(rho.size() == n, "hot_electron cone rho size mismatch");
  TENRYU_ASSERT(zbar.size() == n, "hot_electron cone zbar size mismatch");
  TENRYU_ASSERT(A_eff.size() == n, "hot_electron cone A_eff size mismatch");
  TENRYU_ASSERT(Te_eV.size() == n, "hot_electron cone Te_eV size mismatch");
  TENRYU_ASSERT(cell_is_void.size() == n, "hot_electron cone cell_is_void size mismatch");
  TENRYU_ASSERT(dep_power_cell.size() == n, "hot_electron cone dep_power_cell size mismatch");

  double Pr = 0.0;
  for (const HotESource& source : sources) {
    result.P_hot += source.P_hot;
    Pr += source.P_hot * source.r_s;
  }
  if (sources.empty() || !(result.P_hot > 0.0)) {
    return result;
  }
  result.n_sources = static_cast<int>(sources.size());
  result.r_source_mean = Pr / result.P_hot;

  const double T_h_erg = spec.T_hot_erg;
  const std::vector<GroupSpec> groups =
      build_groups(T_h_erg, spec.n_energy_groups, spec.E_min_over_Th, spec.E_max_over_Th);
  std::vector<double> row(n, 0.0);
  const double kProtonMass = core::constants::proton_mass;
  const auto electron_density = [&](const std::size_t c) -> double {
    const double A = A_eff[c];
    if (!(A > 0.0) || !(rho[c] > 0.0)) {
      return 0.0;
    }
    return rho[c] * std::max(zbar[c], 0.0) / (A * kProtonMass);
  };

  for (const HotESource& source : sources) {
    const std::vector<AngleNode> nodes =
        build_band_nodes(source.mu_axis, spec.mu_lo, spec.mu_hi, spec.n_mu, spec.n_phi);
    for (const AngleNode& node : nodes) {
      const double P_chord = source.P_hot * node.weight;
      if (!(P_chord > 0.0)) {
        continue;
      }
      if (geom == Geometry1D::planar &&
          std::abs(node.mu_dir) < kMuxEpsilon) {
        // In-plane chord in an infinite lateral slab: the path length in the
        // source cell diverges, so the chord thermalizes in place. The old
        // walker classified this as escape (unphysical for a slab).
        const std::size_t sc = static_cast<std::size_t>(source.cell);
        if (source.cell >= 0 && sc < n && !cell_is_void[sc]) {
          dep_power_cell[sc] += P_chord;
          result.P_deposited += P_chord;
        } else {
          result.P_escaped += P_chord;
        }
        continue;
      }
      double escaped_chord = 0.0;
      for (const GroupSpec& g : groups) {
        if (!(g.weight > 0.0) || !(g.E_rep > 0.0)) {
          continue;
        }
        const double Ndot = P_chord * g.weight / g.E_rep;
        double E = g.E_rep;
        if (geom == Geometry1D::spherical) {
          ChordWalkerSph walker(source.r_s, source.r_s * node.mu_dir,
                                r_nodes.data(), n_nodes);
          double ds = 0.0;
          int cell = -1;
          while (walker.next(ds, cell)) {
            const std::size_t c = static_cast<std::size_t>(cell);
            if (cell_is_void[c]) {
              continue;
            }
            const double dSigma = rho[c] * ds;
            const double Te_erg = std::max(Te_eV[c], 0.0) * core::constants::eV_to_erg;
            const double ne = electron_density(c);
            const double E_floor = std::max(2.0 * Te_erg, 1.0e-3 * T_h_erg);
            const double rho_c = rho[c];
            const auto stopping = [&](const double Ee) -> double {
              return stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
            };
            const double E_out = march_cell(E, dSigma, stopping, E_floor,
                                            kSubstepEnergyFraction,
                                            kMaxSubstepsPerCell,
                                            &result.substep_cap_hits);
            row[c] += Ndot * (E - E_out);
            E = E_out;
            if (!(E > 0.0)) {
              break;
            }
          }
        } else {
          ChordWalkerPln walker(source.r_s, node.mu_dir, r_nodes.data(), n_nodes);
          double ds = 0.0;
          int cell = -1;
          while (walker.next(ds, cell)) {
            const std::size_t c = static_cast<std::size_t>(cell);
            if (cell_is_void[c]) {
              continue;
            }
            const double dSigma = rho[c] * ds;
            const double Te_erg = std::max(Te_eV[c], 0.0) * core::constants::eV_to_erg;
            const double ne = electron_density(c);
            const double E_floor = std::max(2.0 * Te_erg, 1.0e-3 * T_h_erg);
            const double rho_c = rho[c];
            const auto stopping = [&](const double Ee) -> double {
              return stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
            };
            const double E_out = march_cell(E, dSigma, stopping, E_floor,
                                            kSubstepEnergyFraction,
                                            kMaxSubstepsPerCell,
                                            &result.substep_cap_hits);
            row[c] += Ndot * (E - E_out);
            E = E_out;
            if (!(E > 0.0)) {
              break;
            }
          }
        }
        if (E > 0.0) {
          escaped_chord += Ndot * E;
        }
      }
      for (std::size_t i = 0; i < n; ++i) {
        if (row[i] != 0.0) {
          dep_power_cell[i] += row[i];
          result.P_deposited += row[i];
        }
      }
      std::fill(row.begin(), row.end(), 0.0);
      result.P_escaped += escaped_chord;
    }
  }

  result.conservation_resid =
      std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
      std::max(result.P_hot, 1.0e-300);
  if (result.conservation_resid > 1.0e-8) {
    core::log_warning("hot_electron cone conservation check failed");
  }
  result.active = true;
  return result;
}

DepositResult deposit_hot_electrons_cone_1d(
    const core::Config::LaserConfig::HotElectronConfig& cfg,
    const Geometry1D geom,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell) {
  return deposit_hot_electrons_cone_1d(make_channel_spec_from_shorthand(cfg), geom,
                                       sources, rho, zbar, A_eff, Te_eV,
                                       cell_is_void, r_nodes, dep_power_cell);
}

DepositResult deposit_hot_electrons_radial_1d(
    const HotEChannelSpec& spec,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell) {
  DepositResult result;
  TENRYU_ASSERT(r_nodes.size() >= 2, "hot_electron radial requires at least two nodes");
  const std::size_t n = r_nodes.size() - 1;
  TENRYU_ASSERT(rho.size() == n, "hot_electron radial rho size mismatch");
  TENRYU_ASSERT(zbar.size() == n, "hot_electron radial zbar size mismatch");
  TENRYU_ASSERT(A_eff.size() == n, "hot_electron radial A_eff size mismatch");
  TENRYU_ASSERT(Te_eV.size() == n, "hot_electron radial Te_eV size mismatch");
  TENRYU_ASSERT(cell_is_void.size() == n, "hot_electron radial cell_is_void size mismatch");
  TENRYU_ASSERT(dep_power_cell.size() == n, "hot_electron radial dep_power_cell size mismatch");

  const HotESource* outermost = nullptr;
  double Pr = 0.0;
  for (const HotESource& source : sources) {
    result.P_hot += source.P_hot;
    Pr += source.P_hot * source.r_s;
    if (outermost == nullptr || source.r_s > outermost->r_s ||
        (source.r_s == outermost->r_s && source.cell > outermost->cell)) {
      outermost = &source;
    }
  }
  if (sources.empty() || !(result.P_hot > 0.0)) {
    return result;
  }
  TENRYU_ASSERT(outermost != nullptr, "hot_electron radial requires a source");
  TENRYU_ASSERT(outermost->cell >= 0 && outermost->cell < static_cast<int>(n),
                "hot_electron radial source cell out of range");
  result.n_sources = static_cast<int>(sources.size());
  // Power-weighted mean per the DepositResult contract (AI review k10-S4);
  // the march itself still starts all power at the outermost source
  // (documented radial-mode merge rule, NUMERICS 5.11).
  result.r_source_mean = Pr / result.P_hot;

  const double T_h_erg = spec.T_hot_erg;
  const std::vector<GroupSpec> groups =
      build_groups(T_h_erg, spec.n_energy_groups, spec.E_min_over_Th, spec.E_max_over_Th);
  const double kProtonMass = core::constants::proton_mass;
  const auto electron_density = [&](const std::size_t c) -> double {
    const double A = A_eff[c];
    if (!(A > 0.0) || !(rho[c] > 0.0)) {
      return 0.0;
    }
    return rho[c] * std::max(zbar[c], 0.0) / (A * kProtonMass);
  };

  for (const GroupSpec& g : groups) {
    if (!(g.weight > 0.0) || !(g.E_rep > 0.0)) {
      continue;
    }
    const double Ndot = result.P_hot * g.weight / g.E_rep;
    double E = g.E_rep;
    for (int c_int = outermost->cell; c_int >= 0; --c_int) {
      const std::size_t c = static_cast<std::size_t>(c_int);
      if (cell_is_void[c]) {
        continue;
      }
      // Partial first cell: the inward march starts at r_s, not at the outer
      // face of the source cell (AI review k10-S6).
      const double r_hi = (c_int == outermost->cell)
                              ? std::min(std::max(outermost->r_s, r_nodes[c]),
                                         r_nodes[c + 1])
                              : r_nodes[c + 1];
      const double dSigma = rho[c] * (r_hi - r_nodes[c]);
      const double Te_erg = std::max(Te_eV[c], 0.0) * core::constants::eV_to_erg;
      const double ne = electron_density(c);
      const double E_floor = std::max(2.0 * Te_erg, 1.0e-3 * T_h_erg);
      const double rho_c = rho[c];
      const auto stopping = [&](const double Ee) -> double {
        return stopping_power_erg_cm2_per_g(Ee, ne, Te_erg, rho_c);
      };
      const double E_out = march_cell(E, dSigma, stopping, E_floor,
                                      kSubstepEnergyFraction,
                                      kMaxSubstepsPerCell,
                                      &result.substep_cap_hits);
      const double deposited = Ndot * (E - E_out);
      dep_power_cell[c] += deposited;
      result.P_deposited += deposited;
      E = E_out;
      if (!(E > 0.0)) {
        break;
      }
    }
    if (E > 0.0) {
      const double residual = Ndot * E;
      if (spec.inner_escape) {
        result.P_escaped += residual;
      } else {
        int inner_cell = -1;
        for (std::size_t c = 0; c < n; ++c) {
          if (!cell_is_void[c]) {
            inner_cell = static_cast<int>(c);
            break;
          }
        }
        TENRYU_ASSERT(inner_cell >= 0,
                      "hot_electron radial deposit_residual requires a non-void cell");
        dep_power_cell[static_cast<std::size_t>(inner_cell)] += residual;
        result.P_deposited += residual;
        result.P_residual_inner += residual;
      }
    }
  }

  result.conservation_resid =
      std::abs(result.P_deposited + result.P_escaped - result.P_hot) /
      std::max(result.P_hot, 1.0e-300);
  if (result.conservation_resid > 1.0e-8) {
    core::log_warning("hot_electron radial conservation check failed");
  }
  result.active = true;
  return result;
}

DepositResult deposit_hot_electrons_radial_1d(
    const core::Config::LaserConfig::HotElectronConfig& cfg,
    const std::vector<HotESource>& sources,
    const std::vector<double>& rho,
    const std::vector<double>& zbar,
    const std::vector<double>& A_eff,
    const std::vector<double>& Te_eV,
    const std::vector<std::uint8_t>& cell_is_void,
    const std::vector<double>& r_nodes,
    std::vector<double>& dep_power_cell) {
  return deposit_hot_electrons_radial_1d(make_channel_spec_from_shorthand(cfg), sources,
                                         rho, zbar, A_eff, Te_eV, cell_is_void,
                                         r_nodes, dep_power_cell);
}

}  // namespace tenryu::laser::hot_electron
