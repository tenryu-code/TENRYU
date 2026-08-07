#include "diagnostics/mesh_degeneracy_forensics.hpp"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include "core/error.hpp"
#include "hydro/rz_quad_volume.cuh"

namespace tenryu::diagnostics {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

bool starts_with(const std::string& value, const char* prefix) {
  const std::string p(prefix);
  return value.size() >= p.size() && value.compare(0, p.size(), p) == 0;
}

bool eligible_reason(const std::string& reason) {
  return starts_with(reason, "mesh_quality_") || starts_with(reason, "in_hydro_");
}

double sanitize_sigma(double sigma) {
  if (!(sigma >= 0.0) || !std::isfinite(sigma)) {
    return 0.0;
  }
  return std::min(sigma, 1.0);
}

double cross2(const double ar, const double az, const double br, const double bz) {
  return ar * bz - az * br;
}

double field_value(const std::vector<double>& values, const int idx) {
  if (idx < 0 || static_cast<std::size_t>(idx) >= values.size()) {
    return 0.0;
  }
  return values[static_cast<std::size_t>(idx)];
}

template <typename Field>
std::vector<double> copy_field(const Field& field) {
  std::vector<double> out;
  if (!field.empty()) {
    field.copy_to_host(out);
  }
  return out;
}

void bilinear_derivatives_at_center(const std::array<double, 4>& values,
                                    double& dxi,
                                    double& deta) {
  dxi = 0.25 * (-values[0] + values[1] + values[2] - values[3]);
  deta = 0.25 * (-values[0] - values[1] + values[2] + values[3]);
}

void compute_div_vorticity(const std::array<double, 4>& r,
                           const std::array<double, 4>& z,
                           const std::array<double, 4>& vr,
                           const std::array<double, 4>& vz,
                           double& div_u,
                           double& vorticity) {
  double r_xi;
  double r_eta;
  double z_xi;
  double z_eta;
  double ur_xi;
  double ur_eta;
  double uz_xi;
  double uz_eta;
  bilinear_derivatives_at_center(r, r_xi, r_eta);
  bilinear_derivatives_at_center(z, z_xi, z_eta);
  bilinear_derivatives_at_center(vr, ur_xi, ur_eta);
  bilinear_derivatives_at_center(vz, uz_xi, uz_eta);
  const double jac = r_xi * z_eta - r_eta * z_xi;
  if (!(std::abs(jac) > 0.0) || !std::isfinite(jac)) {
    div_u = 0.0;
    vorticity = 0.0;
    return;
  }
  const double dur_dr = (ur_xi * z_eta - ur_eta * z_xi) / jac;
  const double dur_dz = (-ur_xi * r_eta + ur_eta * r_xi) / jac;
  const double duz_dr = (uz_xi * z_eta - uz_eta * z_xi) / jac;
  const double duz_dz = (-uz_xi * r_eta + uz_eta * r_xi) / jac;
  div_u = dur_dr + duz_dz;
  vorticity = dur_dz - duz_dr;
}

std::string json_escape(const std::string& input) {
  std::string out;
  for (const char ch : input) {
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

void write_int(std::ostream& os, const char* key, const int value) {
  os << ",\"" << key << "\":" << value;
}

void write_bool(std::ostream& os, const char* key, const bool value) {
  os << ",\"" << key << "\":" << (value ? "true" : "false");
}

void write_string(std::ostream& os, const char* key, const std::string& value) {
  os << ",\"" << key << "\":\"" << json_escape(value) << "\"";
}

void write_number_array(std::ostream& os,
                        const char* key,
                        const std::array<double, 4>& values) {
  os << ",\"" << key << "\":[";
  for (int i = 0; i < 4; ++i) {
    if (i > 0) {
      os << ",";
    }
    if (std::isfinite(values[static_cast<std::size_t>(i)])) {
      os << values[static_cast<std::size_t>(i)];
    } else {
      os << "null";
    }
  }
  os << "]";
}

void write_string_array(std::ostream& os,
                        const char* key,
                        const std::array<std::string, 4>& values) {
  os << ",\"" << key << "\":[";
  for (int i = 0; i < 4; ++i) {
    if (i > 0) {
      os << ",";
    }
    os << "\"" << json_escape(values[static_cast<std::size_t>(i)]) << "\"";
  }
  os << "]";
}

bool cell_ij_from_linear(const core::State& state,
                         const int cell,
                         int& i,
                         int& j) {
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (state.mesh.dim != 2 || nr <= 0 || nz <= 0 || cell < 0 ||
      cell >= nr * nz) {
    return false;
  }
  i = cell / nz;
  j = cell - i * nz;
  return true;
}

bool cell_node_indices(const core::State& state,
                       const int cell,
                       std::array<int, 4>& nodes,
                       int& i,
                       int& j) {
  if (!cell_ij_from_linear(state, cell, i, j)) {
    return false;
  }
  const int stride = state.mesh.topo.nz + 1;
  nodes = {
      i * stride + j,
      (i + 1) * stride + j,
      (i + 1) * stride + (j + 1),
      i * stride + (j + 1),
  };
  const int n_nodes = state.mesh.topo.n_nodes;
  for (const int node : nodes) {
    if (node < 0 || node >= n_nodes) {
      return false;
    }
  }
  return true;
}

void write_node(std::ostream& os,
                const MeshDegeneracyForensicsRecord::NodeData& node) {
  os << "{";
  os << "\"r_old\":";
  if (std::isfinite(node.r_old)) {
    os << node.r_old;
  } else {
    os << "null";
  }
  write_number(os, "z_old", node.z_old);
  write_number(os, "r_candidate", node.r_candidate);
  write_number(os, "z_candidate", node.z_candidate);
  write_number(os, "dr", node.dr);
  write_number(os, "dz", node.dz);
  write_number(os, "dr_over_dt", node.dr_over_dt);
  write_number(os, "dz_over_dt", node.dz_over_dt);
  write_number(os, "u_r_old", node.u_r_old);
  write_number(os, "u_z_old", node.u_z_old);
  write_number(os, "u_r_half", node.u_r_half);
  write_number(os, "u_z_half", node.u_z_half);
  write_number(os, "u_r_new", node.u_r_new);
  write_number(os, "u_z_new", node.u_z_new);
  write_number(os, "a_r_pressure", node.a_r_pressure);
  write_number(os, "a_z_pressure", node.a_z_pressure);
  write_number(os, "a_r_qvisc", node.a_r_qvisc);
  write_number(os, "a_z_qvisc", node.a_z_qvisc);
  write_number(os, "a_r_hourglass", node.a_r_hourglass);
  write_number(os, "a_z_hourglass", node.a_z_hourglass);
  write_number(os, "node_mass", node.node_mass);
  write_number(os, "corner_mass_into_cell", node.corner_mass_into_cell);
  os << "}";
}

int selected_corner_for_record(const MeshDegeneracyForensicsContext& context) {
  if (context.failing_corner >= 0 && context.failing_corner < 4) {
    return context.failing_corner;
  }
  std::array<double, 4> r0{};
  std::array<double, 4> z0{};
  std::array<double, 4> r1{};
  std::array<double, 4> z1{};
  for (int k = 0; k < 4; ++k) {
    r0[static_cast<std::size_t>(k)] = context.nodes[static_cast<std::size_t>(k)].r_old;
    z0[static_cast<std::size_t>(k)] = context.nodes[static_cast<std::size_t>(k)].z_old;
    r1[static_cast<std::size_t>(k)] =
        context.nodes[static_cast<std::size_t>(k)].r_candidate;
    z1[static_cast<std::size_t>(k)] =
        context.nodes[static_cast<std::size_t>(k)].z_candidate;
  }
  int corner = 0;
  double min_j = compute_corner_j_at_sigma(r0, z0, r1, z1, 0, 1.0);
  for (int c = 1; c < 4; ++c) {
    const double j = compute_corner_j_at_sigma(r0, z0, r1, z1, c, 1.0);
    if (j < min_j) {
      min_j = j;
      corner = c;
    }
  }
  return corner;
}

double floor_rel_for_reason(const core::Config& cfg, const std::string& reason) {
  if (reason == "mesh_quality_gauss_j") {
    return cfg.numerics.hydro.mesh_quality_dt_gauss_j_floor_rel;
  }
  if (reason == "mesh_quality_rz_volume") {
    return cfg.numerics.hydro.mesh_quality_dt_rz_volume_floor_rel;
  }
  if (reason == "in_hydro_gauss_j") {
    return cfg.numerics.hydro.in_hydro_gauss_j_floor_rel;
  }
  if (reason == "in_hydro_rz_volume") {
    return cfg.numerics.hydro.in_hydro_rz_volume_floor_rel;
  }
  if (starts_with(reason, "in_hydro_")) {
    return cfg.numerics.hydro.corner_jacobian_floor_eps;
  }
  return cfg.numerics.hydro.mesh_quality_dt_corner_j_floor_rel;
}

MeshDegeneracyForensicsState::PhaseSnapshot* select_phase_snapshot(
    MeshDegeneracyForensicsState& state,
    const char* phase) {
  if (phase == nullptr) {
    return nullptr;
  }
  if (std::strcmp(phase, "post_lagrange") == 0) {
    return &state.post_lagrange_snapshot;
  }
  if (std::strcmp(phase, "post_ale_rezone") == 0) {
    return &state.post_ale_rezone_snapshot;
  }
  if (std::strcmp(phase, "post_remap") == 0) {
    return &state.post_remap_snapshot;
  }
  return nullptr;
}

const MeshDegeneracyForensicsState::PhaseSnapshot* select_phase_snapshot(
    const MeshDegeneracyForensicsState& state,
    const char* phase) {
  if (phase == nullptr) {
    return nullptr;
  }
  if (std::strcmp(phase, "post_lagrange") == 0) {
    return &state.post_lagrange_snapshot;
  }
  if (std::strcmp(phase, "post_ale_rezone") == 0) {
    return &state.post_ale_rezone_snapshot;
  }
  if (std::strcmp(phase, "post_remap") == 0) {
    return &state.post_remap_snapshot;
  }
  return nullptr;
}

void clear_phase_snapshot(MeshDegeneracyForensicsState::PhaseSnapshot& snapshot) {
  snapshot.valid = false;
  snapshot.ran = false;
  snapshot.step = -1;
  snapshot.r.clear();
  snapshot.z.clear();
}

double corner_j_from_positions(const std::array<double, 4>& r,
                               const std::array<double, 4>& z,
                               const int corner) {
  if (corner < 0 || corner >= 4) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  return cross2(r[static_cast<std::size_t>(kp)] -
                    r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(kp)] -
                    z[static_cast<std::size_t>(corner)],
                r[static_cast<std::size_t>(km)] -
                    r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(km)] -
                    z[static_cast<std::size_t>(corner)]);
}

double linearized_corner_j_delta(const std::array<double, 4>& r,
                                 const std::array<double, 4>& z,
                                 const std::array<double, 4>& dr,
                                 const std::array<double, 4>& dz,
                                 const int corner) {
  if (corner < 0 || corner >= 4) {
    return 0.0;
  }
  const int qp = (corner + 1) & 3;
  const int qm = (corner + 3) & 3;
  const auto q = static_cast<std::size_t>(corner);
  const auto p = static_cast<std::size_t>(qp);
  const auto m = static_cast<std::size_t>(qm);
  return (z[m] - z[q]) * dr[p] - (r[m] - r[q]) * dz[p] -
         (z[p] - z[q]) * dr[m] + (r[p] - r[q]) * dz[m] +
         (z[p] - z[m]) * dr[q] + (r[m] - r[p]) * dz[q];
}

std::array<double, 4> corner_j_all(const std::array<double, 4>& r,
                                   const std::array<double, 4>& z) {
  std::array<double, 4> out{};
  for (int q = 0; q < 4; ++q) {
    out[static_cast<std::size_t>(q)] = corner_j_from_positions(r, z, q);
  }
  return out;
}

void write_corner_j_source_budget(
    std::ostream& os,
    const MeshDegeneracyForensicsRecord::CornerJSourceBudget& budget) {
  bool first = true;
  const auto sep = [&]() {
    if (!first) {
      os << ",";
    }
    first = false;
  };
  const auto write_array = [&](const char* key,
                               const std::array<double, 4>& values) {
    sep();
    os << "\"" << key << "\":[";
    for (int i = 0; i < 4; ++i) {
      if (i > 0) {
        os << ",";
      }
      const double value = values[static_cast<std::size_t>(i)];
      if (std::isfinite(value)) {
        os << value;
      } else {
        os << "null";
      }
    }
    os << "]";
  };
  const auto write_budget_bool = [&](const char* key, const bool value) {
    sep();
    os << "\"" << key << "\":" << (value ? "true" : "false");
  };
  const auto write_budget_int = [&](const char* key, const int value) {
    sep();
    os << "\"" << key << "\":" << value;
  };
  const auto write_budget_number = [&](const char* key, const double value) {
    sep();
    os << "\"" << key << "\":";
    if (std::isfinite(value)) {
      os << value;
    } else {
      os << "null";
    }
  };
  os << "{";
  write_array("dJ_pressure", budget.dJ_pressure);
  write_array("dJ_scalar_qvisc", budget.dJ_scalar_qvisc);
  write_array("dJ_anti_hourglass", budget.dJ_anti_hourglass);
  write_array("dJ_predictor", budget.dJ_predictor);
  write_array("dJ_corrector", budget.dJ_corrector);
  write_array("dJ_ale_rezone", budget.dJ_ale_rezone);
  write_array("dJ_remap", budget.dJ_remap);
  write_array("J_post_predictor", budget.J_post_predictor);
  write_array("J_post_corrector", budget.J_post_corrector);
  write_array("J_post_ale_rezone", budget.J_post_ale_rezone);
  write_array("J_post_remap", budget.J_post_remap);
  write_budget_bool("predictor_ran", budget.predictor_ran);
  write_budget_bool("corrector_ran", budget.corrector_ran);
  write_budget_bool("ale_rezone_ran", budget.ale_rezone_ran);
  write_budget_bool("remap_ran", budget.remap_ran);
  write_budget_bool("anti_hourglass_active", budget.anti_hourglass_active);
  write_budget_int("dominant_source_corner", budget.dominant_source_corner);
  write_budget_int("dominant_source_kind", budget.dominant_source_kind);
  write_budget_number("dominant_source_dJ", budget.dominant_source_dJ);
  os << "}";
}

}  // namespace

MeshDegeneracyTriggerDecision MeshDegeneracyForensicsState::observe_failure(
    const core::Config::NumericsConfig::DiagnosticsConfig::
        MeshDegeneracyForensicsConfig& cfg,
    const MeshDegeneracyFailureEvent& event) {
  MeshDegeneracyTriggerDecision decision;
  decision.total_dumps_emitted = total_dumps_emitted;
  if (first_failure_cell_c < 0 && event.cell >= 0 &&
      eligible_reason(event.reason)) {
    first_failure_cell_c = event.cell;
  }
  if (!cfg.enabled) {
    return decision;
  }

  decision.eligible_reason = eligible_reason(event.reason);
  if (!decision.eligible_reason) {
    if (event.step != active_step) {
      active_step = event.step;
      same_cell_consecutive_count = 0;
      last_cell = -1;
      last_corner = -1;
      last_stage = -1;
      last_reason.clear();
    }
    return decision;
  }

  const double sigma = sanitize_sigma(event.sigma_safe);
  const bool same_step = event.step == active_step;
  const bool same_key = same_step && event.cell == last_cell &&
                        event.corner == last_corner && event.stage == last_stage;
  if (!same_key) {
    active_step = event.step;
    last_cell = event.cell;
    last_corner = event.corner;
    last_stage = event.stage;
    last_reason = event.reason;
    same_cell_consecutive_count = 1;
    sigma_band_min_recent = sigma;
    sigma_band_max_recent = sigma;
  } else {
    ++same_cell_consecutive_count;
    sigma_band_min_recent = std::min(sigma_band_min_recent, sigma);
    sigma_band_max_recent = std::max(sigma_band_max_recent, sigma);
    last_reason = event.reason;
  }

  const double band_width = sigma_band_max_recent - sigma_band_min_recent;
  const double band_scale = std::max(std::abs(sigma_band_max_recent), 1.0e-30);
  decision.same_cell_consecutive_count = same_cell_consecutive_count;
  decision.sigma_band_min_recent = sigma_band_min_recent;
  decision.sigma_band_max_recent = sigma_band_max_recent;
  decision.sigma_dt_invariant_confirmed =
      same_cell_consecutive_count >= 2 && band_width <= 0.1 * band_scale;

  const bool count_trigger = same_cell_consecutive_count >= cfg.same_cell_count;
  const bool sigma_trigger = sigma < cfg.sigma_threshold;
  const bool cap_allows = total_dumps_emitted < cfg.max_dumps_per_run;
  decision.emit = (count_trigger || sigma_trigger) && cap_allows;
  if (decision.emit) {
    ++total_dumps_emitted;
  }
  decision.total_dumps_emitted = total_dumps_emitted;
  return decision;
}

void MeshDegeneracyForensicsState::reset_corner_j_source_budget_phase_snapshots(
    const int step) {
  (void)step;
  clear_phase_snapshot(post_lagrange_snapshot);
  clear_phase_snapshot(post_ale_rezone_snapshot);
  clear_phase_snapshot(post_remap_snapshot);
}

void MeshDegeneracyForensicsState::snapshot_corner_j_source_budget_phase(
    const core::State& state,
    const char* phase,
    const bool ran) {
  auto* snapshot = select_phase_snapshot(*this, phase);
  if (snapshot == nullptr) {
    return;
  }
  snapshot->valid = false;
  snapshot->ran = ran;
  snapshot->step = state.step;
  snapshot->r.clear();
  snapshot->z.clear();
  if (state.mesh.dim != 2 || state.x_r.empty() ||
      state.x_r.size() != state.x_z.size()) {
    return;
  }
  state.x_r.copy_to_host(snapshot->r);
  state.x_z.copy_to_host(snapshot->z);
  snapshot->valid = snapshot->r.size() == snapshot->z.size();
}

bool MeshDegeneracyForensicsState::extract_corner_j_source_budget_phase_positions(
    const char* phase,
    const int cell,
    const int nz,
    MeshDegeneracyForensicsPhasePositions& out) const {
  out = MeshDegeneracyForensicsPhasePositions{};
  const auto* snapshot = select_phase_snapshot(*this, phase);
  if (snapshot == nullptr || !snapshot->valid || cell < 0 || nz <= 0) {
    return false;
  }
  const int i = cell / nz;
  const int j = cell - i * nz;
  const int stride = nz + 1;
  const int nodes[4] = {
      i * stride + j,
      (i + 1) * stride + j,
      (i + 1) * stride + (j + 1),
      i * stride + (j + 1),
  };
  const int n_nodes = static_cast<int>(snapshot->r.size());
  for (int k = 0; k < 4; ++k) {
    if (nodes[k] < 0 || nodes[k] >= n_nodes) {
      out = MeshDegeneracyForensicsPhasePositions{};
      return false;
    }
    const auto idx = static_cast<std::size_t>(k);
    const auto node = static_cast<std::size_t>(nodes[k]);
    out.r[idx] = snapshot->r[node];
    out.z[idx] = snapshot->z[node];
  }
  out.valid = true;
  out.ran = snapshot->ran;
  return true;
}

double compute_corner_j_at_sigma(const std::array<double, 4>& r_old,
                                 const std::array<double, 4>& z_old,
                                 const std::array<double, 4>& r_candidate,
                                 const std::array<double, 4>& z_candidate,
                                 const int corner,
                                 const double sigma) {
  if (corner < 0 || corner >= 4) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::array<double, 4> r{};
  std::array<double, 4> z{};
  for (int k = 0; k < 4; ++k) {
    const auto idx = static_cast<std::size_t>(k);
    r[idx] = r_old[idx] + sigma * (r_candidate[idx] - r_old[idx]);
    z[idx] = z_old[idx] + sigma * (z_candidate[idx] - z_old[idx]);
  }
  const int kp = (corner + 1) & 3;
  const int km = (corner + 3) & 3;
  return cross2(r[static_cast<std::size_t>(kp)] - r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(kp)] - z[static_cast<std::size_t>(corner)],
                r[static_cast<std::size_t>(km)] - r[static_cast<std::size_t>(corner)],
                z[static_cast<std::size_t>(km)] - z[static_cast<std::size_t>(corner)]);
}

BilinearFitCoefficients fit_hourglass_amplitude(
    const std::array<double, 4>& values) {
  BilinearFitCoefficients out;
  out.a0 = 0.25 * (values[0] + values[1] + values[2] + values[3]);
  out.a_xi = 0.25 * (-values[0] + values[1] + values[2] - values[3]);
  out.a_eta = 0.25 * (-values[0] - values[1] + values[2] + values[3]);
  out.a_xi_eta = 0.25 * (values[0] - values[1] + values[2] - values[3]);
  return out;
}

double recompute_corner_j_floor_root(const std::array<double, 4>& r_old,
                                     const std::array<double, 4>& z_old,
                                     const std::array<double, 4>& r_candidate,
                                     const std::array<double, 4>& z_candidate,
                                     const int corner,
                                     const double j_floor) {
  const auto f = [&](const double sigma) {
    return compute_corner_j_at_sigma(r_old, z_old, r_candidate, z_candidate,
                                     corner, sigma) - j_floor;
  };
  double prev_sigma = 0.0;
  double prev = f(prev_sigma);
  if (!std::isfinite(prev) || prev <= 0.0) {
    return 0.0;
  }
  constexpr int kScan = 128;
  for (int s = 1; s <= kScan; ++s) {
    const double sigma = static_cast<double>(s) / static_cast<double>(kScan);
    const double value = f(sigma);
    if (!std::isfinite(value)) {
      return 0.0;
    }
    if (value <= 0.0) {
      double lo = prev_sigma;
      double hi = sigma;
      for (int iter = 0; iter < 80; ++iter) {
        const double mid = 0.5 * (lo + hi);
        if (f(mid) > 0.0) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      return hi;
    }
    prev_sigma = sigma;
    prev = value;
  }
  (void)prev;
  return 1.0;
}

double compute_local_total_energy_residual(
    const double kinetic_energy_change_4_nodes,
    const double PdV_work,
    const double viscous_work,
    const double force_dot_velocity_work) {
  (void)PdV_work;
  (void)viscous_work;
  return kinetic_energy_change_4_nodes - force_dot_velocity_work;
}

MeshDegeneracyForensicsRecord::CornerJSourceBudget
compute_corner_j_source_budget(const MeshDegeneracyForensicsContext& context) {
  MeshDegeneracyForensicsRecord::CornerJSourceBudget budget;
  budget.predictor_ran =
      context.stage == "predictor" || context.stage == "corrector";
  budget.corrector_ran = context.stage == "corrector";
  budget.ale_rezone_ran =
      context.post_lagrange.valid && context.post_ale_rezone.valid &&
      context.post_ale_rezone.ran;
  budget.remap_ran = context.post_ale_rezone.valid && context.post_remap.valid &&
                     context.post_remap.ran;
  budget.anti_hourglass_active = context.anti_hourglass_active;

  std::array<double, 4> r0{};
  std::array<double, 4> z0{};
  std::array<double, 4> r_candidate{};
  std::array<double, 4> z_candidate{};
  std::array<double, 4> r_post_predictor{};
  std::array<double, 4> z_post_predictor{};
  std::array<double, 4> r_post_corrector{};
  std::array<double, 4> z_post_corrector{};
  std::array<double, 4> dr{};
  std::array<double, 4> dz{};
  const double dt =
      (context.dt > 0.0 && std::isfinite(context.dt)) ? context.dt : 0.0;
  for (int k = 0; k < 4; ++k) {
    const auto idx = static_cast<std::size_t>(k);
    const auto& node = context.nodes[idx];
    r0[idx] = node.r_old;
    z0[idx] = node.z_old;
    r_candidate[idx] = node.r_candidate;
    z_candidate[idx] = node.z_candidate;
    r_post_predictor[idx] = r0[idx];
    z_post_predictor[idx] = z0[idx];
    if (budget.predictor_ran && dt > 0.0) {
      r_post_predictor[idx] = r0[idx] + 0.5 * dt * node.u_r_half;
      z_post_predictor[idx] = z0[idx] + 0.5 * dt * node.u_z_half;
    }
    r_post_corrector[idx] =
        budget.corrector_ran ? r_candidate[idx] : r_post_predictor[idx];
    z_post_corrector[idx] =
        budget.corrector_ran ? z_candidate[idx] : z_post_predictor[idx];
  }

  const double source_coeff =
      (context.stage == "predictor") ? 0.25 * dt * dt : 0.5 * dt * dt;
  const auto accumulate_accel_source =
      [&](auto accel_r, auto accel_z, std::array<double, 4>& out) {
        if (dt == 0.0) {
          return;
        }
        for (int k = 0; k < 4; ++k) {
          const auto idx = static_cast<std::size_t>(k);
          const auto& node = context.nodes[idx];
          dr[idx] = source_coeff * accel_r(node);
          dz[idx] = source_coeff * accel_z(node);
        }
        for (int q = 0; q < 4; ++q) {
          out[static_cast<std::size_t>(q)] =
              linearized_corner_j_delta(r0, z0, dr, dz, q);
        }
      };
  accumulate_accel_source(
      [](const MeshDegeneracyForensicsRecord::NodeData& node) {
        return node.a_r_pressure;
      },
      [](const MeshDegeneracyForensicsRecord::NodeData& node) {
        return node.a_z_pressure;
      },
      budget.dJ_pressure);
  accumulate_accel_source(
      [](const MeshDegeneracyForensicsRecord::NodeData& node) {
        return node.a_r_qvisc;
      },
      [](const MeshDegeneracyForensicsRecord::NodeData& node) {
        return node.a_z_qvisc;
      },
      budget.dJ_scalar_qvisc);
  if (budget.anti_hourglass_active) {
    accumulate_accel_source(
        [](const MeshDegeneracyForensicsRecord::NodeData& node) {
          return node.a_r_hourglass;
        },
        [](const MeshDegeneracyForensicsRecord::NodeData& node) {
          return node.a_z_hourglass;
        },
        budget.dJ_anti_hourglass);
  }

  if (budget.predictor_ran) {
    for (int k = 0; k < 4; ++k) {
      const auto idx = static_cast<std::size_t>(k);
      dr[idx] = r_post_predictor[idx] - r0[idx];
      dz[idx] = z_post_predictor[idx] - z0[idx];
    }
    for (int q = 0; q < 4; ++q) {
      budget.dJ_predictor[static_cast<std::size_t>(q)] =
          linearized_corner_j_delta(r0, z0, dr, dz, q);
    }
  }
  if (budget.corrector_ran) {
    for (int k = 0; k < 4; ++k) {
      const auto idx = static_cast<std::size_t>(k);
      dr[idx] = r_post_corrector[idx] - r_post_predictor[idx];
      dz[idx] = z_post_corrector[idx] - z_post_predictor[idx];
    }
    for (int q = 0; q < 4; ++q) {
      budget.dJ_corrector[static_cast<std::size_t>(q)] =
          linearized_corner_j_delta(r_post_predictor, z_post_predictor, dr, dz, q);
    }
  }

  budget.J_post_predictor = corner_j_all(r_post_predictor, z_post_predictor);
  budget.J_post_corrector = corner_j_all(r_post_corrector, z_post_corrector);
  budget.J_post_ale_rezone = budget.J_post_corrector;
  budget.J_post_remap = budget.J_post_corrector;

  if (context.post_lagrange.valid && context.post_ale_rezone.valid) {
    budget.J_post_ale_rezone =
        corner_j_all(context.post_ale_rezone.r, context.post_ale_rezone.z);
    if (budget.ale_rezone_ran) {
      for (int k = 0; k < 4; ++k) {
        const auto idx = static_cast<std::size_t>(k);
        dr[idx] = context.post_ale_rezone.r[idx] - context.post_lagrange.r[idx];
        dz[idx] = context.post_ale_rezone.z[idx] - context.post_lagrange.z[idx];
      }
      for (int q = 0; q < 4; ++q) {
        budget.dJ_ale_rezone[static_cast<std::size_t>(q)] =
            linearized_corner_j_delta(
                context.post_lagrange.r, context.post_lagrange.z, dr, dz, q);
      }
    }
  }
  if (context.post_remap.valid) {
    budget.J_post_remap = corner_j_all(context.post_remap.r, context.post_remap.z);
    if (budget.remap_ran) {
      const auto& base = context.post_ale_rezone.valid ? context.post_ale_rezone
                                                       : context.post_lagrange;
      if (base.valid) {
        for (int k = 0; k < 4; ++k) {
          const auto idx = static_cast<std::size_t>(k);
          dr[idx] = context.post_remap.r[idx] - base.r[idx];
          dz[idx] = context.post_remap.z[idx] - base.z[idx];
        }
        for (int q = 0; q < 4; ++q) {
          budget.dJ_remap[static_cast<std::size_t>(q)] =
              linearized_corner_j_delta(base.r, base.z, dr, dz, q);
        }
      }
    }
  } else {
    budget.J_post_remap = budget.J_post_ale_rezone;
  }

  const auto consider = [&](const int kind, const int corner, const double value) {
    if (value < budget.dominant_source_dJ) {
      budget.dominant_source_dJ = value;
      budget.dominant_source_kind = kind;
      budget.dominant_source_corner = corner;
    }
  };
  for (int q = 0; q < 4; ++q) {
    const auto idx = static_cast<std::size_t>(q);
    consider(0, q, budget.dJ_pressure[idx]);
    consider(1, q, budget.dJ_scalar_qvisc[idx]);
    consider(2, q, budget.dJ_anti_hourglass[idx]);
    const double source_sum =
        budget.dJ_pressure[idx] + budget.dJ_scalar_qvisc[idx] +
        budget.dJ_anti_hourglass[idx];
    if (budget.predictor_ran) {
      consider(3, q, budget.dJ_predictor[idx] - source_sum);
    }
    if (budget.corrector_ran) {
      consider(4, q, budget.dJ_corrector[idx] - source_sum);
    }
    if (budget.ale_rezone_ran) {
      consider(5, q, budget.dJ_ale_rezone[idx]);
    }
    if (budget.remap_ran) {
      consider(6, q, budget.dJ_remap[idx]);
    }
  }
  return budget;
}

MeshDegeneracyForensicsRecord build_mesh_degeneracy_forensics_record(
    const core::State& state,
    const core::Config& cfg,
    const MeshDegeneracyForensicsContext& context) {
  MeshDegeneracyForensicsRecord record;
  record.step = state.step;
  record.retry_index = context.retry_index;
  record.dt = context.dt;
  record.t = context.t;
  record.stage = context.stage;
  record.reason = context.reason;
  record.cell = context.cell;
  record.i = context.i;
  record.j = context.j;
  record.failing_corner = context.failing_corner;
  record.sigma_root_reported = sanitize_sigma(context.sigma_root_reported);
  record.nodes = context.nodes;
  record.same_cell_consecutive_count = context.same_cell_consecutive_count;
  record.sigma_band_min_recent = context.sigma_band_min_recent;
  record.sigma_band_max_recent = context.sigma_band_max_recent;
  record.sigma_dt_invariant_confirmed = context.sigma_dt_invariant_confirmed;
  record.corner_j_source_budget_enabled =
      cfg.numerics.diagnostics.mesh_degeneracy_forensics
          .corner_j_source_budget_enabled;
  if (record.corner_j_source_budget_enabled) {
    record.corner_j_source_budget = compute_corner_j_source_budget(context);
  }

  std::array<double, 4> r0{};
  std::array<double, 4> z0{};
  std::array<double, 4> r1{};
  std::array<double, 4> z1{};
  std::array<double, 4> vr{};
  std::array<double, 4> vz{};
  std::array<double, 4> pos_r{};
  std::array<double, 4> pos_z{};
  for (int k = 0; k < 4; ++k) {
    const auto idx = static_cast<std::size_t>(k);
    auto& node = record.nodes[idx];
    node.dr = node.r_candidate - node.r_old;
    node.dz = node.z_candidate - node.z_old;
    if (context.dt > 0.0 && std::isfinite(context.dt)) {
      node.dr_over_dt = node.dr / context.dt;
      node.dz_over_dt = node.dz / context.dt;
    }
    r0[idx] = node.r_old;
    z0[idx] = node.z_old;
    r1[idx] = node.r_candidate;
    z1[idx] = node.z_candidate;
    vr[idx] = node.dr_over_dt;
    vz[idx] = node.dz_over_dt;
    pos_r[idx] = node.r_candidate;
    pos_z[idx] = node.z_candidate;
  }

  const int j_corner = selected_corner_for_record(context);
  record.J0 = compute_corner_j_at_sigma(r0, z0, r1, z1, j_corner, 0.0);
  record.J1 = compute_corner_j_at_sigma(r0, z0, r1, z1, j_corner, 1.0);
  record.J_floor =
      (record.J0 > 0.0 ? floor_rel_for_reason(cfg, context.reason) * record.J0 : 0.0);
  record.J_at_sigma_0 = record.J0;
  record.J_at_sigma_010 =
      compute_corner_j_at_sigma(r0, z0, r1, z1, j_corner, 0.1);
  record.J_at_sigma_safe = compute_corner_j_at_sigma(
      r0, z0, r1, z1, j_corner, record.sigma_root_reported);
  record.J_at_sigma_050 =
      compute_corner_j_at_sigma(r0, z0, r1, z1, j_corner, 0.5);
  record.J_at_sigma_100 = record.J1;
  record.dJ_dsigma_at_0 =
      4.0 * record.J_at_sigma_050 - 3.0 * record.J0 - record.J1;
  record.sigma_root_recomputed =
      recompute_corner_j_floor_root(r0, z0, r1, z1, j_corner, record.J_floor);

  const auto pos_fit_r = fit_hourglass_amplitude(pos_r);
  const auto pos_fit_z = fit_hourglass_amplitude(pos_z);
  const auto vel_fit_r = fit_hourglass_amplitude(vr);
  const auto vel_fit_z = fit_hourglass_amplitude(vz);
  record.position_hourglass_amplitude_r = pos_fit_r.a_xi_eta;
  record.position_hourglass_amplitude_z = pos_fit_z.a_xi_eta;
  record.velocity_hourglass_amplitude_r = vel_fit_r.a_xi_eta;
  record.velocity_hourglass_amplitude_z = vel_fit_z.a_xi_eta;
  const double hg_pos_norm =
      std::hypot(record.position_hourglass_amplitude_r,
                 record.position_hourglass_amplitude_z);
  const double affine_norm =
      std::hypot(pos_fit_r.a_xi, pos_fit_z.a_xi) +
      std::hypot(pos_fit_r.a_eta, pos_fit_z.a_eta);
  record.hourglass_to_affine_gradient_ratio =
      hg_pos_norm / std::max(affine_norm, 1.0e-300);

  const int c = context.cell;
  const auto rho = copy_field(state.rho);
  const auto te = copy_field(state.Te);
  const auto ti = copy_field(state.Ti);
  const auto pe = copy_field(state.Pe);
  const auto pi = copy_field(state.Pi);
  const auto q = copy_field(state.Qvisc);
  const auto cs = copy_field(state.cs);
  const auto vol = copy_field(state.vol);
  const auto volfrac = copy_field(state.volFrac);
  const auto laser_dep = copy_field(state.laser_dep);
  const auto plic_normal_r = copy_field(state.plic_normal_r);
  const auto plic_normal_z = copy_field(state.plic_normal_z);
  const auto plic_centroid_r = copy_field(state.plic_centroid_r);
  const auto plic_centroid_z = copy_field(state.plic_centroid_z);
  record.rho_failing_cell = field_value(rho, c);
  record.Te_failing_cell = field_value(te, c);
  record.Ti_failing_cell = field_value(ti, c);
  record.pressure_failing_cell = field_value(pe, c) + field_value(pi, c);
  record.sound_speed_failing_cell = field_value(cs, c);
  record.Q_visc_failing_cell = field_value(q, c);
  record.laser_deposition_failing_cell = field_value(laser_dep, c);
  record.plic_normal_r = field_value(plic_normal_r, c);
  record.plic_normal_z = field_value(plic_normal_z, c);
  record.plic_centroid_r = field_value(plic_centroid_r, c);
  record.plic_centroid_z = field_value(plic_centroid_z, c);
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  for (int m = 0; m < 4; ++m) {
    if (m < n_mat) {
      record.material_names[static_cast<std::size_t>(m)] =
          cfg.materials.materials[static_cast<std::size_t>(m)].name;
      record.volfrac_failing_cell[static_cast<std::size_t>(m)] =
          field_value(volfrac, c * n_mat + m);
    } else {
      record.material_names[static_cast<std::size_t>(m)].clear();
      record.volfrac_failing_cell[static_cast<std::size_t>(m)] = 0.0;
    }
  }
  int present_materials = 0;
  for (double vf : record.volfrac_failing_cell) {
    if (vf > 1.0e-12) {
      ++present_materials;
    }
  }
  record.is_mixed = present_materials > 1;

  compute_div_vorticity(r1, z1, vr, vz, record.div_u_failing_cell,
                        record.vorticity_failing_cell);
  const double hg_vel_norm =
      std::hypot(record.velocity_hourglass_amplitude_r,
                 record.velocity_hourglass_amplitude_z);
  record.hourglass_to_sound_speed_ratio =
      hg_vel_norm / std::max(std::abs(record.sound_speed_failing_cell), 1.0e-300);

  double ke_old = 0.0;
  double ke_new = 0.0;
  for (const auto& node : record.nodes) {
    const double mass = std::max(node.node_mass, 0.0);
    ke_old += 0.5 * mass * (node.u_r_old * node.u_r_old +
                            node.u_z_old * node.u_z_old);
    ke_new += 0.5 * mass * (node.u_r_new * node.u_r_new +
                            node.u_z_new * node.u_z_new);
    const double u_mid_r = 0.5 * (node.u_r_old + node.u_r_new);
    const double u_mid_z = 0.5 * (node.u_z_old + node.u_z_new);
    record.force_dot_velocity_work +=
        mass * ((node.a_r_pressure + node.a_r_qvisc + node.a_r_hourglass) *
                    u_mid_r +
                (node.a_z_pressure + node.a_z_qvisc + node.a_z_hourglass) *
                    u_mid_z) *
        context.dt;
  }
  record.kinetic_energy_change_4_nodes = ke_new - ke_old;
  const double v_old = field_value(vol, c);
  const double v_candidate = tenryu::hydro::ale::detail::rz_signed_quad_volume(
      r1[0], z1[0], r1[1], z1[1], r1[2], z1[2], r1[3], z1[3]);
  const double dV = v_candidate - v_old;
  record.PdV_work = -record.pressure_failing_cell * dV;
  record.viscous_work = -record.Q_visc_failing_cell * dV;
  record.local_total_energy_residual = compute_local_total_energy_residual(
      record.kinetic_energy_change_4_nodes,
      record.PdV_work,
      record.viscous_work,
      record.force_dot_velocity_work);
  return record;
}

std::string serialize_mesh_degeneracy_forensics_record_jsonl(
    const MeshDegeneracyForensicsRecord& record) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(17);
  os << "{\"step\":" << record.step;
  write_int(os, "retry_index", record.retry_index);
  write_number(os, "dt", record.dt);
  write_number(os, "t", record.t);
  write_string(os, "stage", record.stage);
  write_string(os, "reason", record.reason);
  write_int(os, "cell", record.cell);
  write_int(os, "i", record.i);
  write_int(os, "j", record.j);
  write_int(os, "failing_corner", record.failing_corner);
  write_number(os, "J0", record.J0);
  write_number(os, "J_floor", record.J_floor);
  write_number(os, "J1", record.J1);
  write_number(os, "dJ_dsigma_at_0", record.dJ_dsigma_at_0);
  write_number(os, "sigma_root_reported", record.sigma_root_reported);
  write_number(os, "sigma_root_recomputed", record.sigma_root_recomputed);
  write_number(os, "J_at_sigma_0", record.J_at_sigma_0);
  write_number(os, "J_at_sigma_010", record.J_at_sigma_010);
  write_number(os, "J_at_sigma_safe", record.J_at_sigma_safe);
  write_number(os, "J_at_sigma_050", record.J_at_sigma_050);
  write_number(os, "J_at_sigma_100", record.J_at_sigma_100);
  os << ",\"nodes\":[";
  for (int k = 0; k < 4; ++k) {
    if (k > 0) {
      os << ",";
    }
    write_node(os, record.nodes[static_cast<std::size_t>(k)]);
  }
  os << "]";
  write_number(os,
               "position_hourglass_amplitude_r",
               record.position_hourglass_amplitude_r);
  write_number(os,
               "position_hourglass_amplitude_z",
               record.position_hourglass_amplitude_z);
  write_number(os,
               "velocity_hourglass_amplitude_r",
               record.velocity_hourglass_amplitude_r);
  write_number(os,
               "velocity_hourglass_amplitude_z",
               record.velocity_hourglass_amplitude_z);
  write_number(os,
               "hourglass_to_affine_gradient_ratio",
               record.hourglass_to_affine_gradient_ratio);
  write_number(os,
               "hourglass_to_sound_speed_ratio",
               record.hourglass_to_sound_speed_ratio);
  write_number_array(os, "volfrac_failing_cell", record.volfrac_failing_cell);
  write_string_array(os, "material_names", record.material_names);
  write_bool(os, "is_mixed", record.is_mixed);
  write_number(os, "plic_normal_r", record.plic_normal_r);
  write_number(os, "plic_normal_z", record.plic_normal_z);
  write_number(os, "plic_centroid_r", record.plic_centroid_r);
  write_number(os, "plic_centroid_z", record.plic_centroid_z);
  write_number(os, "rho_failing_cell", record.rho_failing_cell);
  write_number(os, "Te_failing_cell", record.Te_failing_cell);
  write_number(os, "Ti_failing_cell", record.Ti_failing_cell);
  write_number(os, "pressure_failing_cell", record.pressure_failing_cell);
  write_number(os, "sound_speed_failing_cell", record.sound_speed_failing_cell);
  write_number(os, "Q_visc_failing_cell", record.Q_visc_failing_cell);
  write_number(os, "div_u_failing_cell", record.div_u_failing_cell);
  write_number(os, "vorticity_failing_cell", record.vorticity_failing_cell);
  write_number(os, "laser_deposition_failing_cell",
               record.laser_deposition_failing_cell);
  write_number(os, "PdV_work", record.PdV_work);
  write_number(os, "viscous_work", record.viscous_work);
  write_number(os, "force_dot_velocity_work", record.force_dot_velocity_work);
  write_number(os,
               "kinetic_energy_change_4_nodes",
               record.kinetic_energy_change_4_nodes);
  write_number(os,
               "local_total_energy_residual",
               record.local_total_energy_residual);
  if (record.corner_j_source_budget_enabled) {
    write_bool(os, "corner_j_source_budget_enabled", true);
    os << ",\"corner_j_source_budget\":";
    write_corner_j_source_budget(os, record.corner_j_source_budget);
  }
  write_int(os,
            "same_cell_consecutive_count",
            record.same_cell_consecutive_count);
  write_number(os, "sigma_band_min_recent", record.sigma_band_min_recent);
  write_number(os, "sigma_band_max_recent", record.sigma_band_max_recent);
  write_bool(os,
             "sigma_dt_invariant_confirmed",
             record.sigma_dt_invariant_confirmed);
  os << "}";
  return os.str();
}

std::filesystem::path mesh_degeneracy_forensics_output_path(
    const core::Config& cfg) {
  const auto& forensics = cfg.numerics.diagnostics.mesh_degeneracy_forensics;
  const std::filesystem::path dir =
      forensics.output_dir.empty() ? std::filesystem::path(cfg.output.directory)
                                   : std::filesystem::path(forensics.output_dir);
  return dir / "mesh_degeneracy_forensics.jsonl";
}

void append_mesh_degeneracy_forensics_record_jsonl(
    const core::Config& cfg,
    const MeshDegeneracyForensicsRecord& record) {
  const std::filesystem::path path = mesh_degeneracy_forensics_output_path(cfg);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream os(path, std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(), "mesh_degeneracy_forensics failed to open JSONL output");
  os << serialize_mesh_degeneracy_forensics_record_jsonl(record) << "\n";
}

std::filesystem::path mesh_degeneracy_velocity_history_output_path(
    const core::Config& cfg,
    const int i,
    const int j) {
  const auto& forensics = cfg.numerics.diagnostics.mesh_degeneracy_forensics;
  const std::filesystem::path dir =
      forensics.output_dir.empty() ? std::filesystem::path(cfg.output.directory)
                                   : std::filesystem::path(forensics.output_dir);
  return dir / ("mesh_degeneracy_velocity_history_" + std::to_string(i) + "_" +
                std::to_string(j) + ".jsonl");
}

VelocityHistoryRecord build_velocity_history_record(const core::State& state,
                                                    const int cell,
                                                    std::string phase,
                                                    const int step,
                                                    const double t,
                                                    const double dt) {
  VelocityHistoryRecord record;
  record.step = step;
  record.t = t;
  record.dt = dt;
  record.cell = cell;
  record.phase = std::move(phase);

  std::array<int, 4> nodes{};
  int i = -1;
  int j = -1;
  if (!cell_node_indices(state, cell, nodes, i, j) ||
      state.x_r.size() != state.x_z.size() ||
      state.v_r.size() != state.v_z.size() ||
      state.x_r.size() != state.v_r.size() ||
      state.x_r.size() < static_cast<std::size_t>(state.mesh.topo.n_nodes)) {
    return record;
  }
  record.i = i;
  record.j = j;

  std::vector<double> r;
  std::vector<double> z;
  std::vector<double> vr;
  std::vector<double> vz;
  state.x_r.copy_to_host(r);
  state.x_z.copy_to_host(z);
  state.v_r.copy_to_host(vr);
  state.v_z.copy_to_host(vz);

  for (int k = 0; k < 4; ++k) {
    const auto out_idx = static_cast<std::size_t>(k);
    const auto node = static_cast<std::size_t>(nodes[out_idx]);
    record.r_old[out_idx] = r[node];
    record.z_old[out_idx] = z[node];
    record.u_r_old[out_idx] = vr[node];
    record.u_z_old[out_idx] = vz[node];
  }

  const auto vel_fit_r = fit_hourglass_amplitude(record.u_r_old);
  const auto vel_fit_z = fit_hourglass_amplitude(record.u_z_old);
  record.velocity_hourglass_amplitude_r = vel_fit_r.a_xi_eta;
  record.velocity_hourglass_amplitude_z = vel_fit_z.a_xi_eta;
  const double hg_norm =
      std::hypot(record.velocity_hourglass_amplitude_r,
                 record.velocity_hourglass_amplitude_z);
  const double affine_norm =
      std::hypot(vel_fit_r.a_xi, vel_fit_z.a_xi) +
      std::hypot(vel_fit_r.a_eta, vel_fit_z.a_eta);
  record.hourglass_to_affine_gradient_ratio =
      hg_norm / std::max(affine_norm, 1.0e-300);
  return record;
}

std::string serialize_velocity_history_record_jsonl(
    const VelocityHistoryRecord& record) {
  std::ostringstream os;
  os << std::scientific << std::setprecision(17);
  os << "{\"step\":" << record.step;
  write_number(os, "t", record.t);
  write_number(os, "dt", record.dt);
  write_int(os, "cell", record.cell);
  write_int(os, "i", record.i);
  write_int(os, "j", record.j);
  write_number_array(os, "r_old", record.r_old);
  write_number_array(os, "z_old", record.z_old);
  write_number_array(os, "u_r_old", record.u_r_old);
  write_number_array(os, "u_z_old", record.u_z_old);
  write_number(os,
               "velocity_hourglass_amplitude_r",
               record.velocity_hourglass_amplitude_r);
  write_number(os,
               "velocity_hourglass_amplitude_z",
               record.velocity_hourglass_amplitude_z);
  write_number(os,
               "hourglass_to_affine_gradient_ratio",
               record.hourglass_to_affine_gradient_ratio);
  write_string(os, "phase", record.phase);
  os << "}";
  return os.str();
}

void append_velocity_history_record_jsonl(const core::Config& cfg,
                                          const VelocityHistoryRecord& record) {
  const std::filesystem::path path =
      mesh_degeneracy_velocity_history_output_path(cfg, record.i, record.j);
  std::filesystem::create_directories(path.parent_path());
  std::ofstream os(path, std::ios::out | std::ios::app);
  TENRYU_ASSERT(os.good(),
                "mesh_degeneracy_velocity_history failed to open JSONL output");
  os << serialize_velocity_history_record_jsonl(record) << "\n";
}

std::vector<int> velocity_history_cells_for_target(const core::Config& cfg,
                                                   const int target_cell) {
  std::vector<int> cells;
  const auto& forensics = cfg.numerics.diagnostics.mesh_degeneracy_forensics;
  const int nr = cfg.mesh.nr;
  const int nz = cfg.mesh.nz;
  if (nr <= 0 || nz <= 0 || target_cell < 0 || target_cell >= nr * nz) {
    return cells;
  }
  const int i0 = target_cell / nz;
  const int j0 = target_cell - i0 * nz;
  cells.push_back(target_cell);
  if (!forensics.velocity_history_include_1_ring) {
    return cells;
  }
  for (int dj = -1; dj <= 1; ++dj) {
    for (int di = -1; di <= 1; ++di) {
      if (di == 0 && dj == 0) {
        continue;
      }
      const int i = i0 + di;
      const int j = j0 + dj;
      if (i >= 0 && i < nr && j >= 0 && j < nz) {
        cells.push_back(i * nz + j);
      }
    }
  }
  return cells;
}

int emit_velocity_history(const core::State& state,
                          const core::Config& cfg,
                          const int target_cell,
                          const std::string& phase,
                          const int step,
                          const double t,
                          const double dt,
                          MeshDegeneracyForensicsState* forensics_state) {
  const auto& forensics = cfg.numerics.diagnostics.mesh_degeneracy_forensics;
  if (!forensics.velocity_history_enabled ||
      forensics.velocity_history_max_records <= 0) {
    return 0;
  }

  int emitted = 0;
  const std::vector<int> cells =
      velocity_history_cells_for_target(cfg, target_cell);
  for (const int cell : cells) {
    if (forensics_state != nullptr &&
        forensics_state->velocity_history_records_emitted >=
            forensics.velocity_history_max_records) {
      break;
    }
    const VelocityHistoryRecord record =
        build_velocity_history_record(state, cell, phase, step, t, dt);
    if (record.i < 0 || record.j < 0) {
      continue;
    }
    append_velocity_history_record_jsonl(cfg, record);
    if (forensics_state != nullptr) {
      ++forensics_state->velocity_history_records_emitted;
    }
    ++emitted;
  }
  return emitted;
}

int emit_velocity_history(const core::State& state,
                          const core::Config& cfg,
                          const int target_cell,
                          const std::string& phase) {
  MeshDegeneracyForensicsState local_state;
  return emit_velocity_history(state,
                               cfg,
                               target_cell,
                               phase,
                               state.step,
                               state.t,
                               state.dt,
                               &local_state);
}

}  // namespace tenryu::diagnostics
