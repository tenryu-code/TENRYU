#include "diagnostics/mesh_diag_dump.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "hydro/rz_corner_mass.cuh"
#include "hydro/rz_quad_volume.cuh"

namespace tenryu::diagnostics::mesh_diag {
namespace {

struct EnvConfig {
  bool enabled = false;
  int step0 = 0;
  int step1 = 2000000000;
  int every = 0;
  int i_lo = 0;
  int i_hi = std::numeric_limits<int>::max();
  int j_lo = 0;
  int j_hi = std::numeric_limits<int>::max();
  std::string outdir;
};

struct Window {
  int i_lo = 0;
  int i_hi = -1;
  int j_lo = 0;
  int j_hi = -1;
};

struct GeometryMetrics {
  std::array<double, 4> corner_j{};
  std::array<double, 4> face_length{};
  double min_corner_j = 0.0;
  double max_face_length = 0.0;
  double planar_area = 0.0;
  double rz_volume = 0.0;
  double h_min = 0.0;
};

int parse_env_int(const char* name, const int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(raw, &end, 10);
  if (end == raw || *end != '\0') {
    return fallback;
  }
  if (value < static_cast<long>(std::numeric_limits<int>::min()) ||
      value > static_cast<long>(std::numeric_limits<int>::max())) {
    return fallback;
  }
  return static_cast<int>(value);
}

EnvConfig read_env_config() {
  EnvConfig cfg;
  const char* enabled = std::getenv("TENRYU_MESH_DIAG");
  cfg.enabled = (enabled != nullptr && std::string(enabled) == "1");
  if (!cfg.enabled) {
    return cfg;
  }
  cfg.step0 = parse_env_int("TENRYU_MESH_DIAG_STEP0", cfg.step0);
  cfg.step1 = parse_env_int("TENRYU_MESH_DIAG_STEP1", cfg.step1);
  cfg.every = parse_env_int("TENRYU_MESH_DIAG_EVERY", cfg.every);
  cfg.i_lo = parse_env_int("TENRYU_MESH_DIAG_I_LO", cfg.i_lo);
  cfg.i_hi = parse_env_int("TENRYU_MESH_DIAG_I_HI", cfg.i_hi);
  cfg.j_lo = parse_env_int("TENRYU_MESH_DIAG_J_LO", cfg.j_lo);
  cfg.j_hi = parse_env_int("TENRYU_MESH_DIAG_J_HI", cfg.j_hi);
  const char* explicit_outdir = std::getenv("TENRYU_MESH_DIAG_OUTDIR");
  if (explicit_outdir != nullptr && explicit_outdir[0] != '\0') {
    cfg.outdir = explicit_outdir;
  } else if (const char* tenryu_outdir = std::getenv("TENRYU_OUTDIR");
             tenryu_outdir != nullptr && tenryu_outdir[0] != '\0') {
    cfg.outdir = (std::filesystem::path(tenryu_outdir) / "diag").string();
  }
  return cfg;
}

const EnvConfig& env_config() {
  static const EnvConfig cfg = read_env_config();
  return cfg;
}

bool active_step(const core::State& state) {
  const EnvConfig& env = env_config();
  if (!env.enabled) {
    return false;
  }
  const bool in_window = state.step >= env.step0 && state.step <= env.step1;
  if (env.every <= 0) {
    return in_window;
  }
  const bool default_window = env.step0 == 0 && env.step1 == 2000000000;
  const bool every_step = state.step >= 0 && (state.step % env.every) == 0;
  return default_window ? every_step : (in_window || every_step);
}

Window active_window(const core::State& state) {
  const EnvConfig& env = env_config();
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  Window w;
  if (nr <= 0 || nz <= 0 || env.i_lo > nr - 1 || env.i_hi < 0 ||
      env.j_lo > nz - 1 || env.j_hi < 0) {
    return w;
  }
  w.i_lo = std::max(env.i_lo, 0);
  w.i_hi = std::min(env.i_hi, nr - 1);
  w.j_lo = std::max(env.j_lo, 0);
  w.j_hi = std::min(env.j_hi, nz - 1);
  return w;
}

bool valid_window(const Window& w) {
  return w.i_lo <= w.i_hi && w.j_lo <= w.j_hi;
}

std::string output_dir(const core::Config& cfg) {
  const EnvConfig& env = env_config();
  if (!env.outdir.empty()) {
    return env.outdir;
  }
  return (std::filesystem::path(cfg.output.directory) / "diag").string();
}

std::ofstream open_step_file(const core::State& state, const core::Config& cfg) {
  const std::filesystem::path dir(output_dir(cfg));
  std::filesystem::create_directories(dir);
  std::ostringstream name;
  name << "mesh_diag_step" << std::setw(6) << std::setfill('0') << state.step
       << ".jsonl";
  std::ofstream os(dir / name.str(), std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), "mesh diagnostics failed to open JSONL output");
  os << std::scientific << std::setprecision(17);
  return os;
}

bool i1b_polar_pole_diag_active() {
  const char* enabled = std::getenv("TENRYU_I1B_POLAR_POLE_DIAG");
  return enabled != nullptr && std::string(enabled) == "1";
}

std::ofstream open_i1b_polar_pole_file(const core::State& state) {
  const std::filesystem::path dir = std::filesystem::path("tmp") / "diagnostics";
  std::filesystem::create_directories(dir);
  std::ostringstream name;
  name << "i1b_polar_pole_step" << std::setw(6) << std::setfill('0')
       << state.step << ".jsonl";
  std::ofstream os(dir / name.str(), std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), "I1-B polar pole diagnostics failed to open JSONL output");
  os << std::scientific << std::setprecision(17);
  return os;
}

template <typename Field>
std::vector<double> copy_field(const Field& field) {
  std::vector<double> out;
  if (!field.empty()) {
    field.copy_to_host(out);
  }
  return out;
}

double field_value(const std::vector<double>& field, const int idx) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= field.size()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  return field[static_cast<std::size_t>(idx)];
}

std::uint8_t cell_nverts_value(const core::State& state, const int c) {
  if (c >= 0 && static_cast<std::size_t>(c) < state.mesh.cell_nverts.size()) {
    return state.mesh.cell_nverts[static_cast<std::size_t>(c)];
  }
  return 4U;
}

int cell_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

double cross2(const double ar, const double az, const double br, const double bz) {
  return ar * bz - az * br;
}

double face_length(const double r0, const double z0, const double r1, const double z1) {
  return std::hypot(r1 - r0, z1 - z0);
}

GeometryMetrics compute_geometry(const std::vector<double>& r,
                                 const std::vector<double>& z,
                                 const std::vector<double>& vol,
                                 const int nr,
                                 const int nz,
                                 const int c) {
  (void)nr;
  const int i = c / nz;
  const int j = c - i * nz;
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const double rr[4] = {field_value(r, n00), field_value(r, n10),
                        field_value(r, n11), field_value(r, n01)};
  const double zz[4] = {field_value(z, n00), field_value(z, n10),
                        field_value(z, n11), field_value(z, n01)};

  GeometryMetrics g;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    const int km = (k + 3) & 3;
    g.corner_j[static_cast<std::size_t>(k)] =
        cross2(rr[kp] - rr[k], zz[kp] - zz[k], rr[km] - rr[k], zz[km] - zz[k]);
    g.face_length[static_cast<std::size_t>(k)] =
        face_length(rr[k], zz[k], rr[kp], zz[kp]);
  }
  g.min_corner_j = *std::min_element(g.corner_j.begin(), g.corner_j.end());
  g.max_face_length = *std::max_element(g.face_length.begin(), g.face_length.end());
  double shoelace = 0.0;
  for (int k = 0; k < 4; ++k) {
    const int kp = (k + 1) & 3;
    shoelace += rr[k] * zz[kp] - rr[kp] * zz[k];
  }
  g.planar_area = 0.5 * shoelace;
  g.rz_volume = tenryu::hydro::ale::detail::rz_signed_quad_volume(
      rr[0], zz[0], rr[1], zz[1], rr[2], zz[2], rr[3], zz[3]);
  const double v_for_altitude =
      static_cast<std::size_t>(c) < vol.size() ? vol[static_cast<std::size_t>(c)]
                                               : g.rz_volume;
  g.h_min = (g.max_face_length > 0.0) ? (2.0 * v_for_altitude / g.max_face_length)
                                      : std::numeric_limits<double>::quiet_NaN();
  return g;
}

void cell_nodes(const int i, const int j, const int nz, const int nverts, int* nodes) {
  nodes[0] = node_index(i, j, nz);
  nodes[1] = node_index(i + 1, j, nz);
  nodes[2] = node_index(i + 1, j + 1, nz);
  nodes[3] = (nverts == 3) ? -1 : node_index(i, j + 1, nz);
}

double exact_rz_volume_for_cell(const std::vector<double>& r,
                                const std::vector<double>& z,
                                const int i,
                                const int j,
                                const int nz,
                                const int nverts) {
  int nodes[4] = {-1, -1, -1, -1};
  cell_nodes(i, j, nz, nverts, nodes);
  double rr[4] = {0.0, 0.0, 0.0, 0.0};
  double zz[4] = {0.0, 0.0, 0.0, 0.0};
  for (int k = 0; k < nverts; ++k) {
    rr[k] = field_value(r, nodes[k]);
    zz[k] = field_value(z, nodes[k]);
  }
  return tenryu::hydro::rz::rz_polygon_volume_exact(rr, zz, nverts);
}

double quad_gauss_j_min(const std::array<double, 4>& r,
                        const std::array<double, 4>& z) {
  constexpr double q = 0.577350269189625764509148780501957456;
  const double xi[4] = {-q, q, q, -q};
  const double eta[4] = {-q, -q, q, q};
  double out = std::numeric_limits<double>::infinity();
  for (int p = 0; p < 4; ++p) {
    const double dN_dxi[4] = {
        -0.25 * (1.0 - eta[p]),
        0.25 * (1.0 - eta[p]),
        0.25 * (1.0 + eta[p]),
        -0.25 * (1.0 + eta[p]),
    };
    const double dN_deta[4] = {
        -0.25 * (1.0 - xi[p]),
        -0.25 * (1.0 + xi[p]),
        0.25 * (1.0 + xi[p]),
        0.25 * (1.0 - xi[p]),
    };
    double dr_dxi = 0.0;
    double dz_dxi = 0.0;
    double dr_deta = 0.0;
    double dz_deta = 0.0;
    for (int k = 0; k < 4; ++k) {
      dr_dxi += dN_dxi[k] * r[static_cast<std::size_t>(k)];
      dz_dxi += dN_dxi[k] * z[static_cast<std::size_t>(k)];
      dr_deta += dN_deta[k] * r[static_cast<std::size_t>(k)];
      dz_deta += dN_deta[k] * z[static_cast<std::size_t>(k)];
    }
    out = std::min(out, dr_dxi * dz_deta - dz_dxi * dr_deta);
  }
  return out;
}

double quad_gauss_j_min_for_cell(const std::vector<double>& r,
                                 const std::vector<double>& z,
                                 const int i,
                                 const int j,
                                 const int nz) {
  int nodes[4] = {-1, -1, -1, -1};
  cell_nodes(i, j, nz, 4, nodes);
  const std::array<double, 4> rr = {
      field_value(r, nodes[0]),
      field_value(r, nodes[1]),
      field_value(r, nodes[2]),
      field_value(r, nodes[3]),
  };
  const std::array<double, 4> zz = {
      field_value(z, nodes[0]),
      field_value(z, nodes[1]),
      field_value(z, nodes[2]),
      field_value(z, nodes[3]),
  };
  return quad_gauss_j_min(rr, zz);
}

std::string json_escape(const char* input) {
  std::string out;
  if (input == nullptr) {
    return out;
  }
  for (const char ch : std::string(input)) {
    switch (ch) {
      case '\\':
        out += "\\\\";
        break;
      case '"':
        out += "\\\"";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        out += ch;
        break;
    }
  }
  return out;
}

void write_number(std::ostream& os, const char* key, const double value) {
  os << ",\"" << key << "\":";
  if (std::isfinite(value)) {
    os << value;
  } else {
    os << "null";
  }
}

void write_string(std::ostream& os, const char* key, const char* value) {
  os << ",\"" << key << "\":\"" << json_escape(value) << "\"";
}

void write_bool(std::ostream& os, const char* key, const bool value) {
  os << ",\"" << key << "\":" << (value ? "true" : "false");
}

void write_common(std::ostream& os,
                  const core::State& state,
                  const core::Config& cfg,
                  const char* phase,
                  const double dt) {
  os << "{\"step\":" << state.step;
  write_number(os, "t", state.t);
  write_number(os, "state_dt", state.dt);
  write_number(os, "dt", dt);
  write_string(os, "phase", phase);
  write_bool(os, "has_physical_rz_axis",
             cfg.numerics.has_physical_rz_axis);
}

void dump_cells_from_host(const core::State& state,
                          const core::Config& cfg,
                          const char* phase,
                          const double dt,
                          const std::vector<double>& r,
                          const std::vector<double>& z,
                          const std::vector<double>& vr,
                          const std::vector<double>& vz,
                          const std::vector<double>& cs,
                          const std::vector<double>& div_u,
                          const char* dt_limiter_source,
                          const char* hydro_half) {
  const Window w = active_window(state);
  if (!valid_window(w)) {
    return;
  }
  std::vector<double> rho = copy_field(state.rho);
  std::vector<double> pe = copy_field(state.Pe);
  std::vector<double> pi = copy_field(state.Pi);
  std::vector<double> q = copy_field(state.Qvisc);
  std::vector<double> te = copy_field(state.Te);
  std::vector<double> ti = copy_field(state.Ti);
  std::vector<double> vol = copy_field(state.vol);
  std::ofstream os = open_step_file(state, cfg);
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  for (int i = w.i_lo; i <= w.i_hi; ++i) {
    for (int j = w.j_lo; j <= w.j_hi; ++j) {
      const int c = cell_index(i, j, nz);
      const GeometryMetrics g = compute_geometry(r, z, vol, nr, nz, c);
      const double rho_c = field_value(rho, c);
      const double pe_c = field_value(pe, c);
      const double pi_c = field_value(pi, c);
      const double p_c = pe_c + pi_c;
      const double q_c = field_value(q, c);
      write_common(os, state, cfg, phase, dt);
      os << ",\"i\":" << i << ",\"j\":" << j << ",\"c\":" << c;
      write_string(os, "dt_limiter_source", dt_limiter_source);
      write_string(os, "hydro_half", hydro_half);
      write_number(os, "rho", rho_c);
      write_number(os, "P_e", pe_c);
      write_number(os, "P_i", pi_c);
      write_number(os, "P_total", p_c);
      write_number(os, "Qvisc", q_c);
      write_number(os, "Q_over_rho", q_c / std::max(std::abs(rho_c), 1.0e-300));
      write_number(os, "Q_over_P", q_c / std::max(std::abs(p_c), 1.0e-300));
      write_number(os, "Te", field_value(te, c));
      write_number(os, "Ti", field_value(ti, c));
      write_number(os, "c_s", field_value(cs, c));
      write_number(os, "div_u", field_value(div_u, c));
      write_number(os, "vol", field_value(vol, c));
      write_number(os, "rz_volume_from_nodes", g.rz_volume);
      write_number(os, "planar_area", g.planar_area);
      write_number(os, "corner_J0", g.corner_j[0]);
      write_number(os, "corner_J1", g.corner_j[1]);
      write_number(os, "corner_J2", g.corner_j[2]);
      write_number(os, "corner_J3", g.corner_j[3]);
      write_number(os, "min_corner_J", g.min_corner_j);
      write_number(os, "face_length0", g.face_length[0]);
      write_number(os, "face_length1", g.face_length[1]);
      write_number(os, "face_length2", g.face_length[2]);
      write_number(os, "face_length3", g.face_length[3]);
      write_number(os, "max_face_length", g.max_face_length);
      write_number(os, "h_min", g.h_min);
      os << "}\n";
    }
  }

  const std::string node_phase = std::string(phase) + "_node";
  for (int i = w.i_lo; i <= w.i_hi + 1; ++i) {
    for (int j = w.j_lo; j <= w.j_hi + 1; ++j) {
      const int node_id = node_index(i, j, nz);
      write_common(os, state, cfg, node_phase.c_str(), dt);
      os << ",\"node_i\":" << i << ",\"node_j\":" << j
         << ",\"node_id\":" << node_id;
      write_string(os, "hydro_half", hydro_half);
      write_number(os, "x_r", field_value(r, node_id));
      write_number(os, "x_z", field_value(z, node_id));
      write_number(os, "v_r", field_value(vr, node_id));
      write_number(os, "v_z", field_value(vz, node_id));
      os << "}\n";
    }
  }
}

bool enabled_for_step_impl(const core::State& state) {
  if (!active_step(state) || state.mesh.dim != 2) {
    return false;
  }
  if (state.mesh.topo.nr <= 0 || state.mesh.topo.nz <= 0) {
    return false;
  }
  return valid_window(active_window(state));
}

}  // namespace

bool enabled_for_step(const core::State& state, const core::Config& cfg) {
  (void)cfg;
  return enabled_for_step_impl(state);
}

void dump_i1b_polar_pole_diag(const core::State& state,
                              const core::Config& cfg,
                              const I1BPolarPoleAleCounters& counters) {
  (void)cfg;
  if (!i1b_polar_pole_diag_active() || state.mesh.dim != 2 ||
      state.mesh.topo.nr <= 0 || state.mesh.topo.nz <= 0) {
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  std::vector<double> r = copy_field(state.x_r);
  std::vector<double> z = copy_field(state.x_z);
  std::vector<double> r0 = copy_field(state.x_r_initial);
  std::vector<double> z0 = copy_field(state.x_z_initial);
  std::vector<double> v_r = copy_field(state.v_r);
  std::vector<double> v_z = copy_field(state.v_z);
  std::vector<double> vol0 = copy_field(state.cell_vol_initial);
  if (r0.size() != r.size() || z0.size() != z.size()) {
    r0 = r;
    z0 = z;
  }

  std::ofstream os = open_i1b_polar_pole_file(state);
  const int i_lo = std::max(0, nr - 3);
  const int i_hi = nr - 1;
  const int lower_j_hi = std::min(nz - 1, 3);
  const int upper_j_lo = std::max(0, nz - 4);
  for (int i = i_lo; i <= i_hi; ++i) {
    for (int band = 0; band < 2; ++band) {
      const int j_lo = (band == 0) ? 0 : upper_j_lo;
      const int j_hi = (band == 0) ? lower_j_hi : (nz - 1);
      for (int j = j_lo; j <= j_hi; ++j) {
        if (band == 1 && j <= lower_j_hi) {
          continue;
        }
        const int c = cell_index(i, j, nz);
        const int nverts = static_cast<int>(cell_nverts_value(state, c));
        const GeometryMetrics g = compute_geometry(r, z, std::vector<double>{}, nr, nz, c);
        const GeometryMetrics g0 = compute_geometry(r0, z0, std::vector<double>{}, nr, nz, c);
        const double exact_v = exact_rz_volume_for_cell(r, z, i, j, nz, nverts);
        const double exact_v0 = (static_cast<std::size_t>(c) < vol0.size() &&
                                 std::abs(vol0[static_cast<std::size_t>(c)]) > 0.0)
                                    ? vol0[static_cast<std::size_t>(c)]
                                    : exact_rz_volume_for_cell(r0, z0, i, j, nz, nverts);
        const double vol_ratio =
            std::abs(exact_v) / std::max(std::abs(exact_v0), 1.0e-300);
        const double gauss_j_min = (nverts == 4)
                                       ? quad_gauss_j_min_for_cell(r, z, i, j, nz)
                                       : g.min_corner_j;
        const double gauss_j_min0 = (nverts == 4)
                                        ? quad_gauss_j_min_for_cell(r0, z0, i, j, nz)
                                        : g0.min_corner_j;
        const double gauss_j_ratio =
            std::abs(gauss_j_min) / std::max(std::abs(gauss_j_min0), 1.0e-300);
        const double corner_j_ratio =
            std::abs(g.min_corner_j) / std::max(std::abs(g0.min_corner_j), 1.0e-300);
        const int pole_j = (j < nz / 2) ? 0 : nz;
        const int pole_i = std::min(i + 1, nr);
        const int pole_node = node_index(pole_i, pole_j, nz);

        write_common(os, state, cfg, counters.phase, state.dt);
        write_number(os, "time", state.t);
        os << ",\"i\":" << i << ",\"j\":" << j << ",\"c\":" << c
           << ",\"nverts\":" << nverts
           << ",\"pole_node_i\":" << pole_i
           << ",\"pole_node_j\":" << pole_j
           << ",\"pole_node_id\":" << pole_node;
        write_number(os, "exact_rz_volume", exact_v);
        write_number(os, "exact_rz_volume0", exact_v0);
        write_number(os, "exact_rz_volume_ratio", vol_ratio);
        write_number(os, "gauss_J_min", gauss_j_min);
        write_number(os, "gauss_J_ratio", gauss_j_ratio);
        write_number(os, "corner_J_min", g.min_corner_j);
        write_number(os, "corner_J_ratio", corner_j_ratio);
        write_number(os, "pole_node_u_r", field_value(v_r, pole_node));
        write_number(os, "pole_node_u_z", field_value(v_z, pole_node));
        write_bool(os, "axis_guard_trigger", counters.axis_guard_trigger);
        write_bool(os, "force_rezone_input", counters.force_rezone_input);
        write_bool(os, "forced_rezone_fired", counters.forced_rezone_fired);
        write_bool(os, "rezone_triggered", counters.rezone_triggered);
        write_bool(os, "retry_request_path", counters.retry_request_path);
        write_bool(os, "escape_valve_or_repair_fired",
                   counters.escape_valve_or_repair_fired);
        write_string(os, "force_reason", counters.force_reason);
        os << "}\n";
      }
    }
  }
}

void dump_hydro_cells(const core::State& state,
                      const core::Config& cfg,
                      const char* phase,
                      const double dt,
                      const core::NodeField1D& x_r,
                      const core::NodeField1D& x_z,
                      const core::NodeField1D& v_r,
                      const core::NodeField1D& v_z,
                      const core::CellField1D* cs,
                      const core::CellField1D* div_u,
                      const char* dt_limiter_source,
                      const char* hydro_half) {
  if (!enabled_for_step_impl(state)) {
    return;
  }
  dump_cells_from_host(state, cfg, phase, dt, copy_field(x_r), copy_field(x_z),
                       copy_field(v_r), copy_field(v_z),
                       cs == nullptr ? std::vector<double>{} : copy_field(*cs),
                       div_u == nullptr ? std::vector<double>{} : copy_field(*div_u),
                       dt_limiter_source,
                       hydro_half);
}

void dump_hydro_trial(const core::State& state,
                      const core::Config& cfg,
                      const char* phase,
                      const double dt,
                      const core::NodeField1D& r_base,
                      const core::NodeField1D& z_base,
                      const core::NodeField1D& pos_v_r,
                      const core::NodeField1D& pos_v_z,
                      const core::CellField1D* cs,
                      const core::CellField1D* div_u,
                      const char* dt_limiter_source,
                      const char* hydro_half) {
  if (!enabled_for_step_impl(state)) {
    return;
  }
  std::vector<double> r = copy_field(r_base);
  std::vector<double> z = copy_field(z_base);
  std::vector<double> vr = copy_field(pos_v_r);
  std::vector<double> vz = copy_field(pos_v_z);
  const std::size_t n = std::min({r.size(), z.size(), vr.size(), vz.size()});
  for (std::size_t idx = 0; idx < n; ++idx) {
    r[idx] += dt * vr[idx];
    z[idx] += dt * vz[idx];
  }
  dump_cells_from_host(state, cfg, phase, dt, r, z, vr, vz,
                       cs == nullptr ? std::vector<double>{} : copy_field(*cs),
                       div_u == nullptr ? std::vector<double>{} : copy_field(*div_u),
                       dt_limiter_source,
                       hydro_half);
}

void dump_trial_volume_cfl(const core::State& state,
                           const core::Config& cfg,
                           const bool enabled,
                           const bool admissible,
                           const double min_trial_vol_ratio,
                           const double suggested_dt,
                           const int first_failing_cell) {
  if (!enabled_for_step_impl(state)) {
    return;
  }
  std::ofstream os = open_step_file(state, cfg);
  write_common(os, state, cfg, "hydro_trial_volume_cfl", state.dt);
  write_bool(os, "enabled", enabled);
  write_bool(os, "admissible", admissible);
  write_number(os, "min_trial_vol_ratio", min_trial_vol_ratio);
  write_number(os, "suggested_dt", suggested_dt);
  os << ",\"first_failing_cell\":" << first_failing_cell;
  if (first_failing_cell >= 0 && state.mesh.topo.nz > 0) {
    os << ",\"first_failing_i\":" << first_failing_cell / state.mesh.topo.nz
       << ",\"first_failing_j\":" << first_failing_cell % state.mesh.topo.nz;
  }
  os << "}\n";
}

void dump_ale_predictive_acceptance(const core::State& state,
                                    const core::Config& cfg,
                                    const bool active,
                                    const bool feasible,
                                    const char* failure_class,
                                    const int axis_failure_count,
                                    const int cell_vol_failure_count,
                                    const int first_axis_failing_j,
                                    const int first_vol_failing_c,
                                    const double candidate_axis_margin_min,
                                    const double trial_axis_margin_min,
                                    const double candidate_cell_vol_min,
                                    const double trial_cell_vol_min) {
  if (!enabled_for_step_impl(state)) {
    return;
  }
  std::ofstream os = open_step_file(state, cfg);
  write_common(os, state, cfg, "ale_predictive_accept", state.dt);
  write_bool(os, "active", active);
  write_bool(os, "feasible", feasible);
  write_string(os, "failure_class", failure_class);
  os << ",\"axis_failure_count\":" << axis_failure_count
     << ",\"cell_vol_failure_count\":" << cell_vol_failure_count
     << ",\"first_axis_failing_j\":" << first_axis_failing_j
     << ",\"first_vol_failing_c\":" << first_vol_failing_c;
  if (first_vol_failing_c >= 0 && state.mesh.topo.nz > 0) {
    os << ",\"first_vol_failing_i\":" << first_vol_failing_c / state.mesh.topo.nz
       << ",\"first_vol_failing_j\":" << first_vol_failing_c % state.mesh.topo.nz;
  }
  write_number(os, "candidate_axis_margin_min", candidate_axis_margin_min);
  write_number(os, "trial_axis_margin_min", trial_axis_margin_min);
  write_number(os, "candidate_cell_vol_min", candidate_cell_vol_min);
  write_number(os, "trial_cell_vol_min", trial_cell_vol_min);
  os << "}\n";
}

void dump_ale_backtrack_iter(const core::State& state,
                             const core::Config& cfg,
                             const int attempt,
                             const double lambda,
                             const bool post_tangle,
                             const bool post_corner_tangle,
                             const double trial_quality,
                             const double min_corner_j,
                             const int post_min_quality_cell_c,
                             const int post_min_quality_cell_i,
                             const int post_min_quality_cell_j,
                             const int post_corner_fail_cell_c,
                             const int post_corner_fail_cell_i,
                             const int post_corner_fail_cell_j,
                             const int post_corner_fail_corner,
                             const bool accepted,
                             const char* rejection_reason) {
  if (!enabled_for_step_impl(state)) {
    return;
  }
  std::ofstream os = open_step_file(state, cfg);
  write_common(os, state, cfg, "ale_backtrack_iter", state.dt);
  os << ",\"attempt\":" << attempt;
  write_number(os, "lambda", lambda);
  write_bool(os, "post_tangle", post_tangle);
  write_bool(os, "post_corner_tangle", post_corner_tangle);
  write_number(os, "trial_quality", trial_quality);
  write_number(os, "min_corner_J", min_corner_j);
  os << ",\"post_min_quality_cell_c\":" << post_min_quality_cell_c
     << ",\"post_min_quality_cell_i\":" << post_min_quality_cell_i
     << ",\"post_min_quality_cell_j\":" << post_min_quality_cell_j;
  os << ",\"post_corner_fail_cell_c\":" << post_corner_fail_cell_c
     << ",\"post_corner_fail_cell_i\":" << post_corner_fail_cell_i
     << ",\"post_corner_fail_cell_j\":" << post_corner_fail_cell_j
     << ",\"post_corner_fail_corner\":" << post_corner_fail_corner;
  write_bool(os, "accepted", accepted);
  write_string(os, "rejection_reason", rejection_reason);
  os << "}\n";
}

void dump_ale_rezone_end(const core::State& state,
                         const core::Config& cfg,
                         const int rollback_count,
                         const char* last_reason,
                         const double axis_margin_min,
                         const bool accepted_axis_inflow_this_event,
                         const int accepted_axis_inflow_count,
                         const double accepted_axis_inflow_max,
                         const int rezone_iterations,
                         const int min_quality_cell_pre_c,
                         const int min_quality_cell_pre_i,
                         const int min_quality_cell_pre_j,
                         const int min_quality_cell_post_c,
                         const int min_quality_cell_post_i,
                         const int min_quality_cell_post_j) {
  if (!enabled_for_step_impl(state)) {
    return;
  }
  std::ofstream os = open_step_file(state, cfg);
  write_common(os, state, cfg, "ale_rezone_end", state.dt);
  os << ",\"rollback_count\":" << rollback_count;
  write_string(os, "last_reason", last_reason);
  write_number(os, "axis_margin_min", axis_margin_min);
  write_bool(os, "accepted_axis_inflow_this_event", accepted_axis_inflow_this_event);
  os << ",\"accepted_axis_inflow_count\":" << accepted_axis_inflow_count;
  write_number(os, "accepted_axis_inflow_max", accepted_axis_inflow_max);
  os << ",\"rezone_iterations\":" << rezone_iterations
     << ",\"min_quality_cell_pre_c\":" << min_quality_cell_pre_c
     << ",\"min_quality_cell_pre_i\":" << min_quality_cell_pre_i
     << ",\"min_quality_cell_pre_j\":" << min_quality_cell_pre_j
     << ",\"min_quality_cell_post_c\":" << min_quality_cell_post_c
     << ",\"min_quality_cell_post_i\":" << min_quality_cell_post_i
     << ",\"min_quality_cell_post_j\":" << min_quality_cell_post_j
     << "}\n";
}

}  // namespace tenryu::diagnostics::mesh_diag
