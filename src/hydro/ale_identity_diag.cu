#include "hydro/ale_identity_diag.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "mesh/mesh.hpp"

namespace tenryu::hydro::ale_diag {
namespace {

constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct FieldDigest {
  std::uint64_t hash = kFnvOffset;
  double linf = 0.0;
  std::size_t nonfinite_count = 0U;
};

struct MoverSnapshot {
  bool valid = false;
  int step = -1;
  int rank = 0;
  std::string path;
  std::vector<double> node_mass;
  std::vector<double> x_old_r;
  std::vector<double> x_old_z;
  std::vector<double> x_lag_r;
  std::vector<double> x_lag_z;
  std::vector<double> u_move_r;
  std::vector<double> u_move_z;
  std::vector<std::uint8_t> gas_node;
  std::vector<std::uint8_t> shell_node;
};

struct MoverBandStats {
  const char* band = "all";
  int count = 0;
  double mass_sum = 0.0;
  double C_stored = 0.0;
  double C_move = 0.0;
  double du_linf = 0.0;
  double du_mass_l2_sum = 0.0;
};

MoverSnapshot g_mover_snapshot;

template <typename Field>
std::vector<double> copy_field(const Field& field) {
  std::vector<double> host;
  field.copy_to_host(host);
  return host;
}

void fnv_append_double(std::uint64_t& hash, const double value) {
  std::uint64_t bits = 0ULL;
  static_assert(sizeof(bits) == sizeof(value), "double hash size mismatch");
  std::memcpy(&bits, &value, sizeof(bits));
  hash ^= bits;
  hash *= kFnvPrime;
}

FieldDigest digest_values(const std::vector<double>& values) {
  FieldDigest out;
  for (const double value : values) {
    fnv_append_double(out.hash, value);
    if (std::isfinite(value)) {
      out.linf = std::max(out.linf, std::abs(value));
    } else {
      ++out.nonfinite_count;
    }
  }
  return out;
}

FieldDigest digest_scalar(const double value) {
  std::vector<double> values{value};
  return digest_values(values);
}

std::string json_escape(const std::string& value) {
  std::string out;
  out.reserve(value.size() + 8U);
  for (const char ch : value) {
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

std::string hash_hex(const std::uint64_t hash) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return os.str();
}

void write_json_number(std::ostream& os, const char* key, const double value) {
  os << ",\"" << key << "\":";
  if (std::isfinite(value)) {
    os << value;
  } else {
    os << "null";
  }
}

void write_json_string(std::ostream& os, const char* key, const std::string& value) {
  os << ",\"" << key << "\":\"" << json_escape(value) << "\"";
}

std::filesystem::path diag_path(const char* explicit_env,
                                const char* file_name) {
  if (const char* explicit_path = std::getenv(explicit_env);
      explicit_path != nullptr && explicit_path[0] != '\0') {
    return std::filesystem::path(explicit_path);
  }
  if (const char* explicit_outdir = std::getenv("TENRYU_ALE_DIAG_OUTDIR");
      explicit_outdir != nullptr && explicit_outdir[0] != '\0') {
    return std::filesystem::path(explicit_outdir) / file_name;
  }
  if (const char* tenryu_outdir = std::getenv("TENRYU_OUTDIR");
      tenryu_outdir != nullptr && tenryu_outdir[0] != '\0') {
    return std::filesystem::path(tenryu_outdir) / "diag" / file_name;
  }
  return std::filesystem::path(file_name);
}

void append_jsonl(const char* tag,
                  const char* explicit_env,
                  const char* file_name,
                  const std::string& line) {
  const std::filesystem::path path = diag_path(explicit_env, file_name);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream os(path, std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), std::string(tag) + " failed to open JSONL output");
  os << line << '\n';
  core::log_info(std::string(tag) + " " + line);
}

bool diag_enabled(const core::Config& cfg) {
  return cfg.numerics.ale.ale_mover_diag;
}

void emit_identity_record(const core::State& state,
                          const char* stage,
                          const char* field,
                          const FieldDigest& digest,
                          const std::size_t count,
                          const double t_s,
                          const double dt_s,
                          const int rank) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(17);
  os << "{\"tag\":\"ale_identity_diag\",\"record\":\"ale_identity_diag\"";
  os << ",\"step\":" << state.step;
  os << ",\"rank\":" << rank;
  write_json_string(os, "stage", stage != nullptr ? stage : "unknown");
  write_json_string(os, "field", field != nullptr ? field : "unknown");
  write_json_number(os, "t_s", t_s);
  write_json_number(os, "dt_s", dt_s);
  os << ",\"count\":" << count;
  write_json_string(os, "hash", hash_hex(digest.hash));
  write_json_number(os, "linf", digest.linf);
  os << ",\"nonfinite_count\":" << digest.nonfinite_count;
  os << "}";
  append_jsonl("[ale_identity_diag]",
               "TENRYU_ALE_IDENTITY_DIAG_PATH",
               "ale_identity_diag.jsonl",
               os.str());
}

template <typename Field>
void emit_field(const core::State& state,
                const char* stage,
                const char* field_name,
                const Field& field,
                const double t_s,
                const double dt_s,
                const int rank) {
  const std::vector<double> values = copy_field(field);
  emit_identity_record(
      state, stage, field_name, digest_values(values), values.size(), t_s, dt_s, rank);
}

bool active_material_cell(const core::State& state, const int c) {
  const std::size_t idx = static_cast<std::size_t>(c);
  if (!state.hydro_active.empty() && idx < state.hydro_active.size() &&
      state.hydro_active[idx] == 0) {
    return false;
  }
  if (!state.cell_is_void.empty() && idx < state.cell_is_void.size() &&
      state.cell_is_void[idx] != 0U) {
    return false;
  }
  return true;
}

int active_nverts(const core::State& state, const int c) {
  if (state.mesh.cell_nverts.size() == static_cast<std::size_t>(state.mesh.topo.n_cells)) {
    return mesh::mesh_topo_cell_active_nverts(state.mesh.cell_nverts, c);
  }
  return 4;
}

void mark_band_node(std::vector<std::uint8_t>& band, const int node) {
  if (node >= 0 && static_cast<std::size_t>(node) < band.size()) {
    band[static_cast<std::size_t>(node)] = 1U;
  }
}

void classify_mover_bands(const core::State& state,
                          const core::Config& cfg,
                          MoverSnapshot& snap) {
  const int n_nodes = state.mesh.topo.n_nodes;
  const int n_cells = state.mesh.topo.n_cells;
  snap.gas_node.assign(static_cast<std::size_t>(n_nodes), 0U);
  snap.shell_node.assign(static_cast<std::size_t>(n_nodes), 0U);
  if (n_nodes <= 0 || n_cells <= 0) {
    return;
  }
  const double R_g_cm = cfg.numerics.diagnostics.hotspot_gas.R_g_cm;
  const bool use_radius = std::isfinite(R_g_cm) && R_g_cm > 0.0;
  const bool use_tracer =
      !use_radius && state.gas_tracer_Y.size() == static_cast<std::size_t>(n_cells);
  if (!use_radius && !use_tracer) {
    return;
  }
  const std::vector<double> gas_tracer =
      use_tracer ? copy_field(state.gas_tracer_Y) : std::vector<double>{};
  const double cut = std::clamp(cfg.numerics.ale.core_freeze_tracer_cut, 0.0, 1.0);

  if (state.mesh.topo.multiblock.has_value()) {
    const auto& mb = *state.mesh.topo.multiblock;
    if (mb.cell_node_csr_offsets.size() <
        static_cast<std::size_t>(n_cells + 1)) {
      return;
    }
    for (int c = 0; c < n_cells; ++c) {
      if (!active_material_cell(state, c)) {
        continue;
      }
      const int off = mb.cell_node_csr_offsets[static_cast<std::size_t>(c)];
      const int end = mb.cell_node_csr_offsets[static_cast<std::size_t>(c + 1)];
      const int available_nverts = std::max(end - off, 0);
      const int nverts = std::min(active_nverts(state, c), available_nverts);
      double rc = 0.0;
      double zc = 0.0;
      for (int k = 0; k < nverts; ++k) {
        const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        if (node >= 0 && static_cast<std::size_t>(node) < snap.x_lag_r.size() &&
            static_cast<std::size_t>(node) < snap.x_lag_z.size()) {
          rc += snap.x_lag_r[static_cast<std::size_t>(node)];
          zc += snap.x_lag_z[static_cast<std::size_t>(node)];
        }
      }
      if (nverts > 0) {
        rc /= static_cast<double>(nverts);
        zc /= static_cast<double>(nverts);
      }
      const bool gas =
          use_radius ? (std::hypot(rc, zc) < R_g_cm)
                     : (gas_tracer[static_cast<std::size_t>(c)] >= cut);
      for (int k = 0; k < nverts; ++k) {
        const int node = mb.cell_node_csr_indices[static_cast<std::size_t>(off + k)];
        mark_band_node(gas ? snap.gas_node : snap.shell_node, node);
      }
    }
    return;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const int c = i * nz + j;
      if (!active_material_cell(state, c)) {
        continue;
      }
      const int n0 = i * (nz + 1) + j;
      const int n1 = (i + 1) * (nz + 1) + j;
      const int n2 = (i + 1) * (nz + 1) + (j + 1);
      const int n3 = i * (nz + 1) + (j + 1);
      const double rc =
          0.25 * (snap.x_lag_r[static_cast<std::size_t>(n0)] +
                  snap.x_lag_r[static_cast<std::size_t>(n1)] +
                  snap.x_lag_r[static_cast<std::size_t>(n2)] +
                  snap.x_lag_r[static_cast<std::size_t>(n3)]);
      const double zc =
          0.25 * (snap.x_lag_z[static_cast<std::size_t>(n0)] +
                  snap.x_lag_z[static_cast<std::size_t>(n1)] +
                  snap.x_lag_z[static_cast<std::size_t>(n2)] +
                  snap.x_lag_z[static_cast<std::size_t>(n3)]);
      const bool gas =
          use_radius ? (std::hypot(rc, zc) < R_g_cm)
                     : (gas_tracer[static_cast<std::size_t>(c)] >= cut);
      auto& band = gas ? snap.gas_node : snap.shell_node;
      mark_band_node(band, n0);
      mark_band_node(band, n1);
      mark_band_node(band, n2);
      mark_band_node(band, n3);
    }
  }
}

MoverBandStats compute_mover_band(const MoverSnapshot& snap,
                                  const std::vector<double>& u_stored_r,
                                  const std::vector<double>& u_stored_z,
                                  const char* band,
                                  const std::vector<std::uint8_t>* mask) {
  MoverBandStats out;
  out.band = band;
  const std::size_t n =
      std::min({snap.node_mass.size(),
                snap.x_lag_r.size(),
                snap.x_lag_z.size(),
                snap.u_move_r.size(),
                snap.u_move_z.size(),
                u_stored_r.size(),
                u_stored_z.size()});
  for (std::size_t i = 0; i < n; ++i) {
    if (mask != nullptr && (i >= mask->size() || (*mask)[i] == 0U)) {
      continue;
    }
    const double m = snap.node_mass[i];
    const double rr = snap.x_lag_r[i];
    const double rz = snap.x_lag_z[i];
    const double us_r = u_stored_r[i];
    const double us_z = u_stored_z[i];
    const double um_r = snap.u_move_r[i];
    const double um_z = snap.u_move_z[i];
    const double du = std::hypot(us_r - um_r, us_z - um_z);
    ++out.count;
    out.mass_sum += m;
    out.C_stored -= m * (rr * us_r + rz * us_z);
    out.C_move -= m * (rr * um_r + rz * um_z);
    out.du_linf = std::max(out.du_linf, du);
    out.du_mass_l2_sum += m * du * du;
  }
  return out;
}

void emit_mover_record(const core::State& state,
                       const MoverSnapshot& snap,
                       const char* stage,
                       const MoverBandStats& stats,
                       const double t_s,
                       const double dt_s) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(17);
  os << "{\"tag\":\"ale_mover_diag\",\"record\":\"ale_mover_diag\"";
  os << ",\"step\":" << state.step;
  os << ",\"rank\":" << snap.rank;
  write_json_string(os, "path", snap.path);
  write_json_string(os, "stage", stage != nullptr ? stage : "unknown");
  write_json_string(os, "band", stats.band);
  write_json_number(os, "t_s", t_s);
  write_json_number(os, "dt_s", dt_s);
  os << ",\"count\":" << stats.count;
  write_json_number(os, "mass_sum", stats.mass_sum);
  write_json_number(os, "C_stored", stats.C_stored);
  write_json_number(os, "C_move", stats.C_move);
  write_json_number(os, "delta_C", stats.C_stored - stats.C_move);
  write_json_number(os, "du_linf", stats.du_linf);
  const double du_mass_l2 =
      stats.mass_sum > 0.0 ? std::sqrt(stats.du_mass_l2_sum / stats.mass_sum) : 0.0;
  write_json_number(os, "du_mass_l2", du_mass_l2);
  os << "}";
  append_jsonl("[ale_mover_diag]",
               "TENRYU_ALE_MOVER_DIAG_PATH",
               "ale_mover_diag.jsonl",
               os.str());
}

void emit_mover_records(const core::State& state,
                        const MoverSnapshot& snap,
                        const std::vector<double>& u_stored_r,
                        const std::vector<double>& u_stored_z,
                        const char* stage,
                        const double t_s,
                        const double dt_s) {
  emit_mover_record(
      state,
      snap,
      stage,
      compute_mover_band(snap, u_stored_r, u_stored_z, "all", nullptr),
      t_s,
      dt_s);
  emit_mover_record(
      state,
      snap,
      stage,
      compute_mover_band(snap, u_stored_r, u_stored_z, "gas", &snap.gas_node),
      t_s,
      dt_s);
  emit_mover_record(
      state,
      snap,
      stage,
      compute_mover_band(snap, u_stored_r, u_stored_z, "shell", &snap.shell_node),
      t_s,
      dt_s);
}

}  // namespace

void emit_identity_field_diag(const core::State& state,
                              const core::Config& cfg,
                              const core::NodeField1D& node_mass,
                              const char* stage,
                              const double t_s,
                              const double dt_s,
                              const int rank,
                              const core::CellField1D* cs_override) {
  if (!diag_enabled(cfg)) {
    return;
  }
  emit_field(state, stage, "x_r", state.x_r, t_s, dt_s, rank);
  emit_field(state, stage, "x_z", state.x_z, t_s, dt_s, rank);
  emit_field(state, stage, "v_r", state.v_r, t_s, dt_s, rank);
  emit_field(state, stage, "v_z", state.v_z, t_s, dt_s, rank);
  emit_field(state, stage, "vol", state.vol, t_s, dt_s, rank);
  emit_field(state, stage, "rho", state.rho, t_s, dt_s, rank);
  emit_field(state, stage, "mass", state.mass, t_s, dt_s, rank);
  emit_field(state, stage, "ee", state.ee, t_s, dt_s, rank);
  emit_field(state, stage, "ei", state.ei, t_s, dt_s, rank);
  emit_field(state, stage, "Te", state.Te, t_s, dt_s, rank);
  emit_field(state, stage, "Ti", state.Ti, t_s, dt_s, rank);
  emit_field(state, stage, "Pe", state.Pe, t_s, dt_s, rank);
  emit_field(state, stage, "Pi", state.Pi, t_s, dt_s, rank);
  emit_field(state, stage, "cs", cs_override != nullptr ? *cs_override : state.cs,
             t_s, dt_s, rank);
  emit_field(state, stage, "Qvisc", state.Qvisc, t_s, dt_s, rank);
  emit_field(state, stage, "corner_mass", state.corner_mass, t_s, dt_s, rank);
  emit_field(state, stage, "node_mass", node_mass, t_s, dt_s, rank);
  emit_field(state, stage, "vol_prev_hydro", state.vol_prev_hydro, t_s, dt_s, rank);
  const FieldDigest dt_digest = digest_scalar(state.dt_prev_hydro);
  emit_identity_record(
      state, stage, "dt_prev_hydro", dt_digest, 1U, t_s, dt_s, rank);
}

void capture_and_emit_mover_post_hydro(const core::State& state,
                                       const core::Config& cfg,
                                       const core::NodeField1D& r_old,
                                       const core::NodeField1D& z_old,
                                       const core::NodeField1D& node_mass,
                                       const double dt_s,
                                       const double t_s,
                                       const int rank) {
  if (!diag_enabled(cfg) || !(dt_s > 0.0)) {
    return;
  }
  MoverSnapshot snap;
  snap.valid = true;
  snap.step = state.step;
  snap.rank = rank;
  snap.path = (cfg.mesh.motion == "ale" && cfg.numerics.ale.enabled) ? "ale"
                                                                      : "lagrangian";
  snap.node_mass = copy_field(node_mass);
  snap.x_old_r = copy_field(r_old);
  snap.x_old_z = copy_field(z_old);
  snap.x_lag_r = copy_field(state.x_r);
  snap.x_lag_z = copy_field(state.x_z);
  snap.u_move_r.resize(snap.x_lag_r.size(), 0.0);
  snap.u_move_z.resize(snap.x_lag_z.size(), 0.0);
  const std::size_t n =
      std::min({snap.x_old_r.size(), snap.x_old_z.size(), snap.x_lag_r.size(),
                snap.x_lag_z.size()});
  for (std::size_t i = 0; i < n; ++i) {
    snap.u_move_r[i] = (snap.x_lag_r[i] - snap.x_old_r[i]) / dt_s;
    snap.u_move_z[i] = (snap.x_lag_z[i] - snap.x_old_z[i]) / dt_s;
  }
  classify_mover_bands(state, cfg, snap);
  const std::vector<double> u_stored_r = copy_field(state.v_r);
  const std::vector<double> u_stored_z = copy_field(state.v_z);
  g_mover_snapshot = std::move(snap);
  emit_mover_records(state,
                     g_mover_snapshot,
                     u_stored_r,
                     u_stored_z,
                     g_mover_snapshot.path == "ale" ? "ale_pre_projection"
                                                    : "post_hydro",
                     t_s,
                     dt_s);
}

void emit_mover_post_projection(const core::State& state,
                                const core::Config& cfg,
                                const double dt_s,
                                const double t_s) {
  if (!diag_enabled(cfg) || !g_mover_snapshot.valid ||
      g_mover_snapshot.step != state.step) {
    return;
  }
  const std::vector<double> u_stored_r = copy_field(state.v_r);
  const std::vector<double> u_stored_z = copy_field(state.v_z);
  emit_mover_records(
      state, g_mover_snapshot, u_stored_r, u_stored_z, "ale_post_projection", t_s, dt_s);
}

}  // namespace tenryu::hydro::ale_diag
