#include "hydro/central_core_1d_submodel.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>

namespace tenryu::hydro::core1d {

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFourPiOver3 = 4.0 * kPi / 3.0;

Core1DParams g_params;

double env_double(const char* name, const double fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return std::atof(raw);
}

int env_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return std::atoi(raw);
}

double av_c1() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_AV_C1");
  if (raw != nullptr && raw[0] != '\0') {
    static const double v = std::atof(raw);
    return v;
  }
  return g_params.av_c1;
}

double av_c2() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_AV_C2");
  if (raw != nullptr && raw[0] != '\0') {
    static const double v = std::atof(raw);
    return v;
  }
  return g_params.av_c2;
}

double cfl_number() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_CFL");
  if (raw != nullptr && raw[0] != '\0') {
    static const double v = std::atof(raw);
    return v;
  }
  return g_params.cfl;
}

int max_substeps() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_MAX_SUBSTEPS");
  if (raw != nullptr && raw[0] != '\0') {
    static const int v = std::atoi(raw);
    return v;
  }
  return g_params.max_substeps;
}

int build_shell_target() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_BUILD_SHELLS");
  if (raw != nullptr && raw[0] != '\0') {
    static const int v = std::atoi(raw);
    return v;
  }
  return g_params.build_shells;
}

double piston_speed_cap_factor() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_PISTON_CAP");
  if (raw != nullptr && raw[0] != '\0') {
    static const double v = std::atof(raw);
    return v;
  }
  return g_params.piston_cap;
}

struct Core1D {
  bool active = false;
  bool piston_aligned = false;
  // Terminal free-surface mode (set_free_outer): standard massive-face VNR
  // conventions replace the coupled massless-face lumping. One-way switch.
  bool free_outer = false;
  int last_step = -1;
  // staggered grid: N cells, N+1 edges; r[0] = 0 pinned.
  std::vector<double> r;
  std::vector<double> u;
  std::vector<double> m;
  std::vector<double> e;
  std::vector<double> Y;
  // ledger
  double injected_mass = 0.0;
  double injected_energy = 0.0;   // internal + kinetic of injected shells
  double piston_work = 0.0;       // cumulative discrete outer-boundary work
  long long total_substeps = 0;
  long long positivity_guard_halvings = 0;
  int energy_guard_halvings = 0;
  int dt_floor_hits = 0;
  // windowed closure attribution (reset at each ledger print)
  double win_U0 = 0.0;
  double win_K0 = 0.0;
  double win_W0 = 0.0;
  double win_inj0 = 0.0;
  int win_rollbacks = 0;
  int win_fresh_snaps = 0;
  int win_absorbs = 0;
  bool win_init = false;
};

std::map<const void*, Core1D>& registry() {
  static std::map<const void*, Core1D> instances;
  return instances;
}

double cell_volume(const double r0, const double r1) {
  return kFourPiOver3 * (r1 * r1 * r1 - r0 * r0 * r0);
}

double radius_of_volume(const double v) {
  return std::cbrt(std::max(v, 0.0) / kFourPiOver3);
}

double total_internal(const Core1D& c) {
  double s = 0.0;
  for (std::size_t j = 0; j < c.m.size(); ++j) {
    s += c.m[j] * c.e[j];
  }
  return s;
}

double total_kinetic(const Core1D& c) {
  const std::size_t n = c.m.size();
  double s = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    // Massless outer face convention: interior nodes carry half of each
    // neighbor cell, the last interior node carries the outer cell's full
    // mass, the face itself carries none. Free-outer mode: standard
    // half-half everywhere, the face carries half the outer cell (added
    // after the loop).
    double mn;
    if (i == 0) {
      mn = 0.5 * c.m[0];
    } else if (i == n - 1 && !c.free_outer) {
      mn = 0.5 * c.m[i - 1] + c.m[i];
    } else {
      mn = 0.5 * (c.m[i - 1] + c.m[i]);
    }
    s += 0.5 * mn * c.u[i] * c.u[i];
  }
  if (c.free_outer && n > 0) {
    s += 0.5 * (0.5 * c.m[n - 1]) * c.u[n] * c.u[n];
  }
  return s;
}

// One explicit VNR substep with compatible work on the supported paths.
// Outer boundary: dynamic_outer=true evolves the outer node under the
// external-vs-face pressure difference (no velocity override); false is the
// legacy kinematic piston (bc = velocity).
// Returns the substep dt actually taken (0 on refusal).
double substep(Core1D& c,
               const double gamma,
               const bool dynamic_outer,
               const double bc,
               const double dt_cap) {
  static int refuse_dumps = 0;
  const std::vector<double>* refuse_unew = nullptr;
  const std::size_t n = c.m.size();
  std::vector<double> P(n), q(n), cs(n);
  for (std::size_t j = 0; j < n; ++j) {
    const double vol = cell_volume(c.r[j], c.r[j + 1]);
    const double rho = c.m[j] / std::max(vol, 1.0e-300);
    P[j] = (gamma - 1.0) * rho * c.e[j];
    cs[j] = std::sqrt(std::max(gamma * P[j] / rho, 0.0));
    const double du = c.u[j + 1] - c.u[j];
    q[j] = du < 0.0
               ? rho * (av_c1() * cs[j] * std::abs(du) + av_c2() * du * du)
               : 0.0;
  }
  const auto dump_refuse = [&](const char* mode,
                               const double dt_now,
                               const int halvings,
                               const long long cross_i) {
    if (refuse_dumps >= 12) {
      return;
    }
    ++refuse_dumps;
    double dr_min = std::numeric_limits<double>::infinity();
    std::size_t dr_min_j = 0;
    double e_min = std::numeric_limits<double>::infinity();
    std::size_t e_min_j = 0;
    std::size_t e_neg = 0;
    double P_min = std::numeric_limits<double>::infinity();
    double P_max = -std::numeric_limits<double>::infinity();
    double q_max = -std::numeric_limits<double>::infinity();
    double cs_max = -std::numeric_limits<double>::infinity();
    for (std::size_t j = 0; j < n; ++j) {
      const double dr = c.r[j + 1] - c.r[j];
      if (dr < dr_min) {
        dr_min = dr;
        dr_min_j = j;
      }
      if (c.e[j] < e_min) {
        e_min = c.e[j];
        e_min_j = j;
      }
      if (c.e[j] < 0.0) {
        ++e_neg;
      }
      P_min = std::min(P_min, P[j]);
      P_max = std::max(P_max, P[j]);
      q_max = std::max(q_max, q[j]);
      cs_max = std::max(cs_max, cs[j]);
    }
    double u_absmax = 0.0;
    std::size_t u_absmax_i = 0;
    for (std::size_t i = 0; i <= n; ++i) {
      const double u_abs = std::abs(c.u[i]);
      if (u_abs > u_absmax) {
        u_absmax = u_abs;
        u_absmax_i = i;
      }
    }
    double unew_absmax = 0.0;
    std::size_t unew_absmax_i = 0;
    if (refuse_unew != nullptr) {
      for (std::size_t i = 0; i <= n; ++i) {
        const double unew_abs = std::abs((*refuse_unew)[i]);
        if (unew_abs > unew_absmax) {
          unew_absmax = unew_abs;
          unew_absmax_i = i;
        }
      }
    }
    std::fprintf(
        stderr,
        "[core1d_substep_refuse] mode=%s dt_cap=%.6e dt=%.6e n=%zu "
        "dr_min=%.6e@%zu e_min=%.6e@%zu e_neg=%zu P_min=%.6e "
        "P_max=%.6e q_max=%.6e cs_max=%.6e u_absmax=%.6e@%zu "
        "unew_absmax=%.6e@%zu r0=%.6e r_out=%.6e halvings=%d "
        "cross_i=%lld\n",
        mode, dt_cap, dt_now, n, dr_min, dr_min_j, e_min, e_min_j, e_neg,
        P_min, P_max, q_max, cs_max, u_absmax, u_absmax_i, unew_absmax,
        unew_absmax_i, c.r[0], c.r[n], halvings, cross_i);
  };
  double dt = dt_cap;
  for (std::size_t j = 0; j < n; ++j) {
    const double dr = c.r[j + 1] - c.r[j];
    const double du = std::abs(c.u[j + 1] - c.u[j]);
    const double lim = cfl_number() * dr / (cs[j] + du + 1.0e-30);
    dt = std::min(dt, lim);
  }
  if (!(dt > 0.0) || !std::isfinite(dt)) {
    dump_refuse("nonfinite_dt", dt, 0, -1);
    return 0.0;
  }
  std::vector<double> unew(n + 1);
  std::vector<double> du;
  refuse_unew = &unew;
  // MASSLESS outer face (verdict #6 Q3-1, converged after measuring both
  // failure modes): the outer cell's full mass lumps onto its INNER node,
  // so the kinematically driven face moves zero mass — the override does
  // no nodal kinetic work (the slosh pump is gone) while the volume stays
  // tethered to the 2D boundary (the pure force-driven face un-tethers and
  // freely expands: measured dV divergence to -7e-3 cm^3 and 386% residual).
  const auto compute_kick = [&](const double kick_dt) {
    unew[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
      // Free-outer mode uses the standard half-half node mass everywhere; the
      // coupled massless-face mode lumps the outer cell's full mass onto its
      // inner node (the face then moves zero mass). NOTE the legacy
      // dynamic_outer path keeps the lumped interior AND a half-mass face —
      // the outer cell's inertia is counted 1.5x, a plausible source of its
      // measured instability; the free mode is the consistent scheme.
      const double mn = (i == n - 1 && !c.free_outer)
                            ? 0.5 * c.m[i - 1] + c.m[i]
                            : 0.5 * (c.m[i - 1] + c.m[i]);
      const double A = 4.0 * kPi * c.r[i] * c.r[i];
      const double f = -((P[i] + q[i]) - (P[i - 1] + q[i - 1])) * A;
      unew[i] = c.u[i] + kick_dt * f / std::max(mn, 1.0e-300);
    }
    if (c.free_outer) {
      // Terminal free surface: massive face (half the outer cell), driven by
      // the face-vs-external pressure difference; bc carries P_ext [dyn/cm^2].
      const double m_out = std::max(0.5 * c.m[n - 1], 1.0e-300);
      const double A_out = 4.0 * kPi * c.r[n] * c.r[n];
      const double f_out = ((P[n - 1] + q[n - 1]) - bc) * A_out;
      unew[n] = c.u[n] + kick_dt * f_out / m_out;
    } else if (dynamic_outer) {
      // Experimental force-driven outer node (env; measured unstable —
      // kept for study only). Known inconsistency: total_kinetic() excludes
      // this kicked face's KE while the lumped inner node retains the outer
      // cell's full mass. W4d-7 intentionally leaves this unsupported mode's
      // numerics unchanged.
      const double m_out = std::max(0.5 * c.m[n - 1], 1.0e-300);
      const double A_out = 4.0 * kPi * c.r[n] * c.r[n];
      const double f_out = -(bc - (P[n - 1] + q[n - 1])) * A_out;
      unew[n] = c.u[n] + kick_dt * f_out / m_out;
    } else {
      unew[n] = bc;  // massless face: prescribed constraint velocity
    }
  };
  // Positivity retries recompute the kick from the pre-kick velocities so
  // momentum, position, compatible work, and piston work share the final dt.
  bool motion_ok = false;
  int halvings = 0;
  long long last_cross_i = -1;
  for (int guard = 0; guard <= 40; ++guard) {
    compute_kick(dt);
    bool ok = true;
    double prev = 0.0;
    for (std::size_t i = 0; i <= n; ++i) {
      const double rn = c.r[i] + dt * unew[i];
      if (i > 0 && rn <= prev) {
        ok = false;
        last_cross_i = static_cast<long long>(i);
        break;
      }
      prev = rn;
    }
    if (ok && !(dynamic_outer && !c.free_outer)) {
      du.resize(n);
      std::vector<double> ubar(n + 1, 0.0);
      for (std::size_t i = 0; i <= n; ++i) {
        ubar[i] = 0.5 * (c.u[i] + unew[i]);
      }
      if (!c.free_outer) {
        ubar[n] = bc;
      }
      for (std::size_t j = 0; j < n; ++j) {
        const double A_left = 4.0 * kPi * c.r[j] * c.r[j];
        const double A_right = 4.0 * kPi * c.r[j + 1] * c.r[j + 1];
        du[j] = dt * (P[j] + q[j]) *
                (A_left * ubar[j] - A_right * ubar[j + 1]);
      }
      bool energy_ok = true;
      double worst_deficit = -std::numeric_limits<double>::infinity();
      long long worst_cell = -1;
      for (std::size_t j = 0; j < n; ++j) {
        const double margin =
            64.0 * std::numeric_limits<double>::epsilon() *
            std::max({std::abs(c.e[j]),
                      0.5 * std::max(c.u[j] * c.u[j],
                                     c.u[j + 1] * c.u[j + 1]),
                      cs[j] * cs[j]});
        const double trial_e = c.e[j] + du[j] / c.m[j];
        if (!(trial_e > margin)) {
          energy_ok = false;
          const double deficit = margin - trial_e;
          if (worst_cell < 0 || deficit > worst_deficit) {
            worst_deficit = deficit;
            worst_cell = static_cast<long long>(j);
          }
        }
      }
      if (!energy_ok) {
        last_cross_i = worst_cell;
        if (guard == 40) {
          break;
        }
        dt *= 0.5;
        ++halvings;
        ++c.positivity_guard_halvings;
        ++c.energy_guard_halvings;
        if (dt < 1.0e-22) {
          ++c.dt_floor_hits;
          dump_refuse("energy_floor", dt, halvings, last_cross_i);
          return 0.0;
        }
        continue;
      }
    }
    if (ok) {
      motion_ok = true;
      break;
    }
    if (guard == 40) {
      break;
    }
    dt *= 0.5;
    ++halvings;
    ++c.positivity_guard_halvings;
    if (dt < 1.0e-22) {
      ++c.dt_floor_hits;
      dump_refuse("guard_floor", dt, halvings, last_cross_i);
      return 0.0;
    }
  }
  if (!motion_ok) {
    ++c.dt_floor_hits;
    dump_refuse("guard_exhaust", dt, 41, last_cross_i);
    return 0.0;
  }
  std::vector<double> rnew(n + 1);
  for (std::size_t i = 0; i <= n; ++i) {
    rnew[i] = c.r[i] + dt * unew[i];
  }
  rnew[0] = 0.0;
  if (dynamic_outer && !c.free_outer) {
    // Study-only legacy dynamic mode is outside W4d-7 scope.
    for (std::size_t j = 0; j < n; ++j) {
      const double v0 = cell_volume(c.r[j], c.r[j + 1]);
      const double v1 = cell_volume(rnew[j], rnew[j + 1]);
      c.e[j] =
          std::max(c.e[j] - (P[j] + q[j]) * (v1 - v0) / c.m[j], 1.0e-30);
    }
  } else {
    // For sigma_j = P_j + q_j, the kick at each massive interior node is
    //   M_i (u_i^+ - u_i^-) = dt A_i (sigma_{i-1} - sigma_i).
    // Multiplication by ubar_i = (u_i^- + u_i^+)/2 gives its exact kinetic
    // work. Attributing the opposite work to the two face-owning cells gives
    //   dU_j = dt sigma_j (A_j ubar_j - A_{j+1} ubar_{j+1}).
    // Summing cells cancels every interior face against dK. The center has
    // A_0 = 0. At the massive free face, ubar_n is the kick midpoint and the
    // uncancelled external term is -dt P_ext A_n ubar_n. At the massless
    // coupled face there is no KE: its prescribed velocity bc is the face
    // velocity, so the uncancelled flux is -dt sigma_{n-1} A_n bc.
    for (std::size_t j = 0; j < n; ++j) {
      c.e[j] += du[j] / c.m[j];
    }
  }
  // Piston work done ON the sub-model by the moving outer face. Free mode:
  // the external pressure (bc) is the only outside agent -- the face's own
  // (P+q) exchange with the outer cell is internal (cell U <-> face KE) and
  // must not be booked as injection. Pair the exact external component of
  // the momentum kick, -bc*A_old, with the kinetic-energy midpoint velocity
  // and the accepted guard dt that produced both unew and rnew.
  {
    if (c.free_outer) {
      const double A_old = 4.0 * kPi * c.r[n] * c.r[n];
      const double u_mid = 0.5 * (c.u[n] + unew[n]);
      c.piston_work += -bc * A_old * u_mid * dt;
    } else if (dynamic_outer) {
      // Study-only legacy booking; see the known face-KE exclusion above.
      const double r_mid = 0.5 * (c.r[n] + rnew[n]);
      const double A = 4.0 * kPi * r_mid * r_mid;
      c.piston_work += -(P[n - 1] + q[n - 1]) * A * unew[n] * dt;
    } else {
      // The prescribed massless-face flux uses the same pre-step area family
      // as the interior compatible identity and the constraint velocity bc.
      const double A_old = 4.0 * kPi * c.r[n] * c.r[n];
      c.piston_work += -(P[n - 1] + q[n - 1]) * A_old * bc * dt;
    }
  }
  c.r.swap(rnew);
  c.u.swap(unew);
  ++c.total_substeps;
  return dt;
}

}  // namespace

void set_params(const Core1DParams& p) {
  static bool injected = false;
  if (injected) {
    return;
  }
  g_params = p;
  injected = true;
  std::fprintf(stderr,
               "[core1d] params injected enabled=%d build_shells=%d "
               "split_append=%d av_c1=%.17g av_c2=%.17g cfl=%.17g "
               "piston_cap=%.17g max_substeps=%d\n",
               g_params.enabled ? 1 : 0,
               g_params.build_shells,
               g_params.split_append,
               g_params.av_c1,
               g_params.av_c2,
               g_params.cfl,
               g_params.piston_cap,
               g_params.max_substeps);
}

bool enabled() {
  // Env snapshot is safe to cache; the injected-params fallback must stay a
  // LIVE read — a static latch here can freeze the pre-injection default if
  // any caller runs before set_params (ordering-fragile).
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_SUBMODEL");
  if (raw != nullptr && raw[0] != '\0') {
    return raw[0] != '0';
  }
  return g_params.enabled;
}

void build(const void* key,
           std::vector<BuildCell> cells,
           const double V_c,
           const double gamma) {
  (void)gamma;
  if (!enabled() || cells.empty() || !(V_c > 0.0)) {
    return;
  }
  Core1D& c = registry()[key];
  c = Core1D{};
  std::sort(cells.begin(), cells.end(),
            [](const BuildCell& a, const BuildCell& b) { return a.r < b.r; });
  double m_total = 0.0;
  double v_total = 0.0;
  for (const BuildCell& bc : cells) {
    m_total += bc.m;
    v_total += bc.v;
  }
  if (!(m_total > 0.0) || !(v_total > 0.0)) {
    return;
  }
  const int n_shell =
      std::max(4, std::min<int>(build_shell_target(),
                                static_cast<int>(cells.size())));
  const double m_per = m_total / n_shell;
  c.r.push_back(0.0);
  double acc_m = 0.0, acc_me = 0.0, acc_mY = 0.0, acc_v = 0.0, v_cum = 0.0;
  for (std::size_t i = 0; i < cells.size(); ++i) {
    acc_m += cells[i].m;
    acc_me += cells[i].m * cells[i].e;
    acc_mY += cells[i].m * cells[i].Y;
    acc_v += cells[i].v;
    const bool last = (i + 1 == cells.size());
    if (acc_m >= m_per || last) {
      v_cum += acc_v;
      c.m.push_back(acc_m);
      c.e.push_back(acc_me / acc_m);
      c.Y.push_back(std::min(std::max(acc_mY / acc_m, 0.0), 1.0));
      // edge radii purely from the cumulative member volume; the one-time
      // outer-face alignment to the macro boundary loop volume happens on
      // the first advance (a static t=0 geometry redefinition, logged, not
      // a dynamic compression).
      c.r.push_back(radius_of_volume(v_cum));
      acc_m = acc_me = acc_mY = acc_v = 0.0;
    }
  }
  c.u.assign(c.r.size(), 0.0);
  c.injected_mass = m_total;
  c.injected_energy = total_internal(c);
  c.active = true;
  std::fprintf(stderr,
               "[core1d] built shells=%zu M=%.6e g V=%.6e cm3 U=%.6e erg\n",
               c.m.size(), m_total, V_c, c.injected_energy);
}

bool built(const void* key) {
  const auto it = registry().find(key);
  return it != registry().end() && it->second.active;
}

void reset(const void* key) {
  registry().erase(key);
}

void absorb_event_single(const void* key,
                         double mass,
                         double e_specific,
                         double u_radial,
                         double gas_fraction,
                         double shell_volume);

bool dist_append_enabled() {
  static const char* raw = std::getenv("TENRYU_I1B_CORE_1D_DIST_APPEND");
  if (raw != nullptr && raw[0] != '\0') {
    return raw[0] != '0';
  }
  return g_params.dist_append;
}

void absorb_event_distribution(const void* key,
                               std::vector<ShellSample> samples) {
  if (!enabled() || samples.empty()) {
    return;
  }
  // Innermost-first: most negative u_r penetrates deepest; appending in
  // ascending u_r order stacks the sub-shells physically (each
  // absorb_event_single appends at the current outer face).
  std::sort(samples.begin(), samples.end(),
            [](const ShellSample& a, const ShellSample& b) {
              return a.u_r < b.u_r;
            });
  // Cap the sub-shell count: merge ADJACENT samples (u_r order) into bins,
  // conserving (M, P_r, U, MY, V) per bin; within-bin velocity variance
  // thermalizes into the bin's specific internal energy (graceful
  // degradation to the pooled append at cap=1).
  static const int cap = [] {
    const char* raw = std::getenv("TENRYU_I1B_CORE_1D_DIST_BINS");
    const int v = raw != nullptr && raw[0] != '\0' ? std::atoi(raw) : 0;
    return v > 0 ? v : 64;
  }();
  const int n = static_cast<int>(samples.size());
  const int bins = std::min(n, cap);
  int emitted = 0;
  double m_total = 0.0;
  for (int b = 0; b < bins; ++b) {
    const int lo = (n * b) / bins;
    const int hi = (n * (b + 1)) / bins;
    double M = 0.0, P = 0.0, U = 0.0, K = 0.0, MY = 0.0, V = 0.0;
    for (int i = lo; i < hi; ++i) {
      const ShellSample& c = samples[static_cast<std::size_t>(i)];
      M += c.m;
      P += c.m * c.u_r;
      U += c.m * c.e;
      K += 0.5 * c.m * c.u_r * c.u_r;
      MY += c.m * c.Y;
      V += c.V;
    }
    if (!(M > 0.0) || !(V > 0.0)) {
      continue;
    }
    const double u = P / M;
    const double e_spec = (U + std::max(K - 0.5 * M * u * u, 0.0)) / M;
    absorb_event_single(key, M, e_spec, u, MY / M, V);
    ++emitted;
    m_total += M;
  }
  std::fprintf(stderr,
               "[core1d] dist append: samples=%d bins=%d emitted=%d "
               "M=%.6e u_range=[%.4e,%.4e]\n",
               n,
               bins,
               emitted,
               m_total,
               samples.front().u_r,
               samples.back().u_r);
}

void absorb_event(const void* key,
                  const double mass,
                  const double e_specific,
                  const double u_radial,
                  const double gas_fraction,
                  const double shell_volume) {
  if (!enabled()) {
    return;
  }
  // Split massive appends into equal sub-shells so shocks can form INSIDE
  // the absorbed material (dyncore20: two pooled mega-shells carried the
  // whole capsule shell; with no internal pressure-gradient resolution the
  // slab rams the cushion under-dissipatively and over-compresses it by
  // +59% vs the resolved 1D reference).
  static const int split_n = [] {
    const char* raw = std::getenv("TENRYU_I1B_CORE_1D_SPLIT_APPEND");
    const int v = raw != nullptr && raw[0] != '\0'
                      ? std::atoi(raw)
                      : g_params.split_append;
    return v > 1 ? v : 1;
  }();
  {
    const auto it0 = registry().find(key);
    if (split_n > 1 && it0 != registry().end() && it0->second.active &&
        mass > 0.0 &&
        mass > 0.05 * std::max(it0->second.injected_mass, 1.0e-300)) {
      for (int k = 0; k < split_n; ++k) {
        absorb_event_single(key, mass / split_n, e_specific, u_radial,
                            gas_fraction, shell_volume / split_n);
      }
      return;
    }
  }
  absorb_event_single(key, mass, e_specific, u_radial, gas_fraction,
                      shell_volume);
}

void absorb_event_single(const void* key,
                         const double mass,
                         const double e_specific,
                         const double u_radial,
                         const double gas_fraction,
                         const double shell_volume) {
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return;
  }
  Core1D& c = it->second;
  if (!(mass > 0.0) || !std::isfinite(e_specific) || !(shell_volume > 0.0)) {
    return;
  }
  const double r_out_new =
      radius_of_volume(cell_volume(0.0, c.r.back()) + shell_volume);
  const double K_before = total_kinetic(c);
  const double m_prev = c.m.back();
  const double u_face_old = c.u.back();
  c.m.push_back(mass);
  c.e.push_back(std::max(e_specific, 1.0e-30));
  c.Y.push_back(std::min(std::max(gas_fraction, 0.0), 1.0));
  c.r.push_back(std::max(r_out_new, c.r.back() * (1.0 + 1.0e-12)));
  // Massless-face convention: the OLD face node becomes the interior owner
  // of (half prev cell + the new shell); land the arriving radial momentum
  // there by a conserving merge. The NEW face continues kinematically.
  {
    const double m_own = 0.5 * m_prev + mass;
    c.u.back() = (0.5 * m_prev * c.u[c.u.size() - 2] + mass * u_radial) /
                 std::max(m_own, 1.0e-300);
  }
  c.u.push_back(u_face_old);
  c.injected_mass += mass;
  // Book the DISCRETE kinetic-energy change of the append (the interface
  // node's mass reassignment shifts K beyond the naive 0.5*m*u^2), so the
  // ledger stays exact by construction.
  c.injected_energy +=
      mass * std::max(e_specific, 1.0e-30) + (total_kinetic(c) - K_before);
  // Invalidate the step snapshot: a failure-retry absorption re-enters the
  // same 2D step, and rolling back to the pre-absorb snapshot would pair
  // shorter r/u/e arrays with the grown m/Y. The next begin_step takes a
  // fresh post-absorb snapshot instead; the aborted attempt's piston motion
  // is self-correcting (the next advance snaps to the restored V_c target).
  c.last_step = -1;
  ++c.win_absorbs;
  std::fprintf(stderr,
               "[core1d] absorb shell m=%.4e e=%.4e u_r=%.4e Y=%.3f "
               "r_out=%.5e shells=%zu K_before=%.6e K_after=%.6e "
               "booked=%.6e\n",
               mass, e_specific, u_radial, gas_fraction, c.r.back(),
               c.m.size(), K_before, total_kinetic(c),
               mass * std::max(e_specific, 1.0e-30) +
                   (total_kinetic(c) - K_before));
}

bool begin_step(const void* key, const int step) {
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return false;
  }
  Core1D& c = it->second;
  if (c.last_step == step) {
    // Same 2D step re-entry (double invocation of the hydro_step_start
    // aggregate, or a driver full-step retry): the sub-model has already
    // advanced this step. Re-advancing with rollback was measured to leak
    // kinetic energy (window attribution: closure == dK); skipping keeps
    // one advance per step and a deterministic outer-face pressure for the
    // retried 2D attempt. The piston self-corrects to the restored V_c on
    // the next step.
    ++c.win_rollbacks;
    return false;
  }
  c.last_step = step;
  ++c.win_fresh_snaps;
  return true;
}

void advance_to_volume(const void* key,
                       const double V_c,
                       const double dt_2d,
                       const double gamma) {
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return;
  }
  Core1D& c = it->second;
  if (!(V_c > 0.0) || !(dt_2d > 0.0)) {
    return;
  }
  if (c.free_outer) {
    // Terminal free surface: there is no 2D volume to track any more; a
    // kinematic override here would silently re-tether the face.
    static bool warned = false;
    if (!warned) {
      warned = true;
      std::fprintf(stderr,
                   "[core1d] advance_to_volume ignored: free-outer mode\n");
    }
    return;
  }
  const double r_target = radius_of_volume(V_c);
  if (!c.piston_aligned) {
    // One-time static alignment of the discretization's outer face to the
    // macro boundary loop volume (t=0 geometry redefinition, not dynamics;
    // energy books start from the post-alignment state).
    const double floor_r = c.r[c.r.size() - 2] * (1.0 + 1.0e-12);
    const double mismatch =
        (r_target - c.r.back()) / std::max(c.r.back(), 1.0e-300);
    c.r.back() = std::max(r_target, floor_r);
    c.piston_aligned = true;
    std::fprintf(stderr,
                 "[core1d] piston aligned: outer-face mismatch %.3e "
                 "(member-volume sum vs boundary loop)\n",
                 mismatch);
    return;
  }
  double remaining = dt_2d;
  int steps = 0;
  while (remaining > 1.0e-30 && steps < max_substeps()) {
    // Recompute the piston velocity every substep so the endpoint converges
    // to the target volume without a bookkeeping-free position snap. Cap
    // the slam rate at a physical multiple of the outer cell signal speed.
    const std::size_t n = c.m.size();
    double u_piston = (r_target - c.r.back()) / remaining;
    {
      const double vol = cell_volume(c.r[n - 1], c.r[n]);
      const double rho = c.m[n - 1] / std::max(vol, 1.0e-300);
      const double P = (gamma - 1.0) * rho * c.e[n - 1];
      const double cs = std::sqrt(std::max(gamma * P / rho, 0.0));
      const double cap =
          piston_speed_cap_factor() * (cs + std::abs(c.u[n - 1]) + 1.0e-30);
      u_piston = std::min(std::max(u_piston, -cap), cap);
    }
    const double dt = substep(c, gamma, false, u_piston, remaining);
    if (!(dt > 0.0)) {
      break;
    }
    remaining -= dt;
    ++steps;
  }
  const double resid =
      std::abs(c.r.back() - r_target) / std::max(r_target, 1.0e-300);
  if (resid > 1.0e-3) {
    std::fprintf(stderr,
                 "[core1d] piston lag: r_out=%.6e target=%.6e rel=%.3e "
                 "substeps=%d\n",
                 c.r.back(), r_target, resid, steps);
  }
}

void advance_dynamic(const void* key,
                     const double P_ext,
                     const double dt_2d,
                     const double gamma) {
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return;
  }
  Core1D& c = it->second;
  if (!(dt_2d > 0.0) || !std::isfinite(P_ext) || P_ext < 0.0) {
    return;
  }
  if (!c.piston_aligned) {
    // First-contact geometry note only: the dynamic boundary needs no
    // alignment (volume tracking is diagnostic), but keep the flag so the
    // kinematic fallback stays consistent if toggled.
    c.piston_aligned = true;
  }
  double remaining = dt_2d;
  int steps = 0;
  while (remaining > 1.0e-30 && steps < max_substeps()) {
    const double dt = substep(c, gamma, true, P_ext, remaining);
    if (!(dt > 0.0)) {
      ++c.dt_floor_hits;
      break;
    }
    remaining -= dt;
    ++steps;
  }
}

void set_free_outer(const void* key, const double gamma) {
  (void)gamma;
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return;
  }
  Core1D& c = it->second;
  if (c.free_outer) {
    return;
  }
  // The convention switch moves the outer cell's half mass from the last
  // interior node (velocity u[n-1]) onto the face (velocity u[n]); the
  // definition jump in total kinetic energy is booked as an explicit
  // injection so the conservation ledger closes exactly across the switch.
  const double k_before = total_kinetic(c);
  c.free_outer = true;
  const double k_after = total_kinetic(c);
  c.injected_energy += k_after - k_before;
  std::fprintf(stderr,
               "[core1d] free-outer mode ON: K reattribution dK=%.6e erg "
               "(massless coupled face -> massive free face), u_face=%.6e "
               "u_last=%.6e\n",
               k_after - k_before,
               c.u.back(),
               c.u[c.u.size() - 2]);
}

bool free_outer(const void* key) {
  const auto it = registry().find(key);
  return it != registry().end() && it->second.active &&
         it->second.free_outer;
}

long long substep_count(const void* key) {
  const auto it = registry().find(key);
  return it != registry().end() ? it->second.total_substeps : 0LL;
}

long long positivity_guard_halving_count(const void* key) {
  const auto it = registry().find(key);
  return it != registry().end() ? it->second.positivity_guard_halvings : 0LL;
}

double outer_face_pressure(const void* key, const double gamma) {
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return 0.0;
  }
  const Core1D& c = it->second;
  const std::size_t n = c.m.size();
  const double vol = cell_volume(c.r[n - 1], c.r[n]);
  const double rho = c.m[n - 1] / std::max(vol, 1.0e-300);
  const double P = (gamma - 1.0) * rho * c.e[n - 1];
  const double du = c.u[n] - c.u[n - 1];
  const double cs = std::sqrt(std::max(gamma * P / rho, 0.0));
  const double q =
      du < 0.0 ? rho * (av_c1() * cs * std::abs(du) + av_c2() * du * du)
               : 0.0;
  const double p_face = P + q;
  return (std::isfinite(p_face) && p_face > 0.0) ? p_face : 0.0;
}

double outer_face_pressure_static(const void* key, const double gamma) {
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return 0.0;
  }
  const Core1D& c = it->second;
  const std::size_t n = c.m.size();
  const double vol = cell_volume(c.r[n - 1], c.r[n]);
  const double rho = c.m[n - 1] / std::max(vol, 1.0e-300);
  const double P = (gamma - 1.0) * rho * c.e[n - 1];
  return (std::isfinite(P) && P > 0.0) ? P : 0.0;
}

double piston_work_total(const void* key) {
  const auto it = registry().find(key);
  return (it != registry().end() && it->second.active)
             ? it->second.piston_work
             : 0.0;
}

double current_volume(const void* key) {
  const auto it = registry().find(key);
  return (it != registry().end() && it->second.active)
             ? cell_volume(0.0, it->second.r.back())
             : 0.0;
}

double internal_energy_total(const void* key) {
  const auto it = registry().find(key);
  return (it != registry().end() && it->second.active)
             ? total_internal(it->second)
             : 0.0;
}

double kinetic_energy_total(const void* key) {
  const auto it = registry().find(key);
  return (it != registry().end() && it->second.active)
             ? total_kinetic(it->second)
             : 0.0;
}

TailDiagnostics tail_diagnostics(const void* key, const double gamma) {
  TailDiagnostics out;
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active ||
      it->second.m.empty()) {
    return out;
  }
  const Core1D& c = it->second;
  const std::size_t n = c.m.size();
  std::vector<double> pressure(n, 0.0);
  std::vector<double> center(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    const double vol = cell_volume(c.r[j], c.r[j + 1]);
    const double rho = c.m[j] / std::max(vol, 1.0e-300);
    pressure[j] = (gamma - 1.0) * rho * c.e[j];
    center[j] = 0.5 * (c.r[j] + c.r[j + 1]);
    if (!std::isfinite(pressure[j]) || !std::isfinite(center[j])) {
      return TailDiagnostics{};
    }
    out.p_max = std::max(out.p_max, pressure[j]);
  }
  std::size_t shock_shell = 0U;
  double max_abs_dpdr = -1.0;
  for (std::size_t j = 1; j < n; ++j) {
    const double dr = center[j] - center[j - 1U];
    if (!(dr > 0.0)) {
      return TailDiagnostics{};
    }
    const double abs_dpdr =
        std::abs(pressure[j] - pressure[j - 1U]) / dr;
    if (abs_dpdr > max_abs_dpdr) {
      max_abs_dpdr = abs_dpdr;
      shock_shell = j;
    }
  }
  out.r_shock = c.r[shock_shell + 1U];
  out.u_outer = c.u.back();
  out.valid = std::isfinite(out.r_shock) && std::isfinite(out.u_outer) &&
              std::isfinite(out.p_max);
  return out;
}

GasView gas_view(const void* key, const double gamma) {
  GasView out;
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return out;
  }
  const Core1D& c = it->second;
  out.valid = true;
  for (std::size_t j = 0; j < c.m.size(); ++j) {
    const double vol = cell_volume(c.r[j], c.r[j + 1]);
    const double rho = c.m[j] / std::max(vol, 1.0e-300);
    out.V_gas += c.Y[j] * vol;
    out.M_gas += c.Y[j] * c.m[j];
    out.U_gas += c.Y[j] * c.m[j] * c.e[j];
    if (c.Y[j] > 0.5) {
      out.rhoR_gas += rho * (c.r[j + 1] - c.r[j]);
      out.r_gas_outer = c.r[j + 1];
    }
  }
  {
    const double vol0 = cell_volume(c.r[0], c.r[1]);
    const double rho0 = c.m[0] / std::max(vol0, 1.0e-300);
    out.p_center = (gamma - 1.0) * rho0 * c.e[0];
    out.e_center = c.e[0];
  }
  return out;
}

void emit_ledger(const void* key,
                 const int step,
                 const double t,
                 const double gamma) {
  static const int every = env_int("TENRYU_I1B_CORE_1D_LEDGER_EVERY", 0);
  if (every <= 0 || (step % every) != 0) {
    return;
  }
  const auto it = registry().find(key);
  if (it == registry().end() || !it->second.active) {
    return;
  }
  Core1D& c = it->second;
  const double U = total_internal(c);
  const double K = total_kinetic(c);
  const double budget = c.injected_energy + c.piston_work;
  const GasView g = gas_view(key, gamma);
  std::fprintf(stderr,
               "[core1d_ledger] step=%d t=%.6e shells=%zu M=%.9e "
               "U=%.6e K=%.6e budget=%.6e resid_rel=%.3e "
               "Vgas=%.6e p_c=%.4e substeps=%lld dt_floor=%d\n",
               step, t, c.m.size(), c.injected_mass, U, K, budget,
               (U + K - budget) / std::max(std::abs(budget), 1.0e-300),
               g.V_gas, g.p_center, c.total_substeps, c.dt_floor_hits);
  if (c.win_init) {
    const double dU = U - c.win_U0;
    const double dK = K - c.win_K0;
    const double dW = c.piston_work - c.win_W0;
    const double dInj = c.injected_energy - c.win_inj0;
    std::fprintf(stderr,
                 "[core1d_ledger_win] step=%d dU=%.4e dK=%.4e dW=%.4e "
                 "dInj=%.4e closure=%.4e rollbacks=%d fresh=%d absorbs=%d\n",
                 step, dU, dK, dW, dInj, dU + dK - dW - dInj,
                 c.win_rollbacks, c.win_fresh_snaps, c.win_absorbs);
  }
  c.win_U0 = U;
  c.win_K0 = K;
  c.win_W0 = c.piston_work;
  c.win_inj0 = c.injected_energy;
  c.win_rollbacks = 0;
  c.win_fresh_snaps = 0;
  c.win_absorbs = 0;
  c.win_init = true;
}

}  // namespace tenryu::hydro::core1d
