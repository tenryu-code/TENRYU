#include "hydro/shadow_homothety_diag.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>
#include <vector>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/mesh.hpp"
#include "parallel/partition.hpp"

namespace tenryu::hydro {
namespace {

struct VectorRz {
  double r = 0.0;
  double z = 0.0;
};

struct ShadowHomothetySample {
  int step = -1;
  double t = 0.0;
  double a_driver = 0.0;
  double ann1_relative = 0.0;
  double ann2_relative = 0.0;
  double homothety_rate = 0.0;
  double epsilon_median = 0.0;
  double epsilon_p95 = 0.0;
  double feather_area = 0.0;
  double feather_volume = 0.0;
  double feather_area_rate = 0.0;
  double feather_volume_rate = 0.0;
  double feather_radial_velocity = 0.0;
  double rplus_radial_velocity = 0.0;
  double radial_velocity_jump = 0.0;
  double interface_strain = 0.0;
  double rplus_min_volume_ratio = 0.0;
};

struct ShadowHomothetyPostLagrangeSample {
  int step = -1;
  double t = 0.0;
  double a_plag = 0.0;
  double a_inc = 0.0;
  double a_cumulative = 1.0;
  double instantaneous_rate = 0.0;
  double epsilon_median = 0.0;
  double epsilon_p95 = 0.0;
};

struct ShadowHomothetyState {
  bool initialized = false;
  bool multirank_unsupported = false;
  bool multirank_logged = false;
  bool flushed = false;
  bool has_recorded_step = false;
  bool previous_sample_valid = false;
  bool last_sample_valid = false;
  bool last_post_lagrange_sample_valid = false;
  int rank = 0;
  int latest_step = -1;
  int last_recorded_post_lagrange_step = -1;
  double latest_t = 0.0;
  double previous_t = 0.0;
  double previous_a_driver = 0.0;
  double previous_feather_area = 0.0;
  double previous_post_lagrange_t = 0.0;
  double previous_end_scale = 1.0;
  double post_lagrange_cumulative = 1.0;
  const core::State* live_state = nullptr;
  std::vector<double> template_node_r;
  std::vector<double> template_node_z;
  std::vector<double> initial_volume;
  std::vector<double> initial_mass;
  std::vector<int> cell_node_offsets;
  std::vector<int> cell_node_indices;
  std::vector<std::uint8_t> cell_nverts;
  std::vector<VectorRz> template_centroid;
  std::vector<double> initial_radius;
  std::vector<int> driver_cells;
  std::vector<int> annulus1_cells;
  std::vector<int> annulus2_cells;
  std::vector<int> feather_cells;
  std::vector<int> rplus_cells;
  std::vector<int> wideband_cells;
  std::vector<double> wideband_theta0_deg;
  std::vector<double> previous_feather_volume;
  std::vector<double> previous_rplus_volume;
  ShadowHomothetySample last_sample;
  ShadowHomothetyPostLagrangeSample last_post_lagrange_sample;
};

bool shadow_homothety_enabled() {
  static const bool enabled = [] {
    const char* raw = std::getenv("TENRYU_I1B_SHADOW_HOMOTHETY");
    return raw != nullptr && std::strcmp(raw, "1") == 0;
  }();
  return enabled;
}

int shadow_homothety_cadence_steps() {
  static const int cadence_steps = [] {
    const char* raw = std::getenv("TENRYU_I1B_SHADOW_CADENCE");
    if (raw == nullptr || raw[0] == '\0') {
      return 50;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 1 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
      return 50;
    }
    return static_cast<int>(parsed);
  }();
  return cadence_steps;
}

int shadow_homothety_top_k() {
  static const int top_k = [] {
    const char* raw = std::getenv("TENRYU_I1B_SHADOW_TOPK");
    if (raw == nullptr || raw[0] == '\0') {
      return 0;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw, &end, 10);
    if (end == raw || *end != '\0' || parsed < 1 ||
        parsed > static_cast<long>(std::numeric_limits<int>::max())) {
      return 0;
    }
    return static_cast<int>(parsed);
  }();
  return top_k;
}

ShadowHomothetyState& shadow_homothety_state() {
  static ShadowHomothetyState state;
  return state;
}

double quiet_nan() {
  return std::numeric_limits<double>::quiet_NaN();
}

int active_nverts(const ShadowHomothetyState& diagnostic, const int cell) {
  return mesh::mesh_topo_cell_active_nverts(diagnostic.cell_nverts, cell);
}

void validate_cell_csr(const ShadowHomothetyState& diagnostic,
                       const int cell,
                       const int n_nodes) {
  TENRYU_ASSERT(cell >= 0 &&
                    static_cast<std::size_t>(cell) + 1U <
                        diagnostic.cell_node_offsets.size(),
                "shadow homothety cell-node CSR offset is missing");
  const int offset =
      diagnostic.cell_node_offsets[static_cast<std::size_t>(cell)];
  const int end =
      diagnostic.cell_node_offsets[static_cast<std::size_t>(cell) + 1U];
  const int nverts = active_nverts(diagnostic, cell);
  TENRYU_ASSERT(offset >= 0 && end - offset >= nverts &&
                    static_cast<std::size_t>(end) <=
                        diagnostic.cell_node_indices.size(),
                "shadow homothety cell-node CSR is incomplete");
  for (int corner = 0; corner < nverts; ++corner) {
    const int node = diagnostic.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    TENRYU_ASSERT(node >= 0 && node < n_nodes,
                  "shadow homothety cell node is out of range");
  }
}

VectorRz cell_vertex_mean(const ShadowHomothetyState& diagnostic,
                          const int cell,
                          const std::vector<double>& node_r,
                          const std::vector<double>& node_z) {
  const int offset =
      diagnostic.cell_node_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(diagnostic, cell);
  double sum_r = 0.0;
  double sum_z = 0.0;
  for (int corner = 0; corner < nverts; ++corner) {
    const int node = diagnostic.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    sum_r += node_r[static_cast<std::size_t>(node)];
    sum_z += node_z[static_cast<std::size_t>(node)];
  }
  const double inverse_count = 1.0 / static_cast<double>(nverts);
  return {sum_r * inverse_count, sum_z * inverse_count};
}

double planar_cell_area(const ShadowHomothetyState& diagnostic,
                        const int cell,
                        const std::vector<double>& node_r,
                        const std::vector<double>& node_z) {
  const int offset =
      diagnostic.cell_node_offsets[static_cast<std::size_t>(cell)];
  const int nverts = active_nverts(diagnostic, cell);
  double twice_area = 0.0;
  for (int corner = 0; corner < nverts; ++corner) {
    const int next = (corner + 1) % nverts;
    const int node0 = diagnostic.cell_node_indices[
        static_cast<std::size_t>(offset + corner)];
    const int node1 = diagnostic.cell_node_indices[
        static_cast<std::size_t>(offset + next)];
    twice_area += node_r[static_cast<std::size_t>(node0)] *
                      node_z[static_cast<std::size_t>(node1)] -
                  node_r[static_cast<std::size_t>(node1)] *
                      node_z[static_cast<std::size_t>(node0)];
  }
  return 0.5 * std::abs(twice_area);
}

double dot(const VectorRz& lhs, const VectorRz& rhs) {
  return lhs.r * rhs.r + lhs.z * rhs.z;
}

double weighted_scale(const ShadowHomothetyState& diagnostic,
                      const std::vector<int>& cells,
                      const std::vector<VectorRz>& current_centroid) {
  double numerator = 0.0;
  double denominator = 0.0;
  for (const int cell : cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double weight = diagnostic.initial_volume[index];
    numerator += weight *
                 dot(diagnostic.template_centroid[index],
                     current_centroid[index]);
    denominator += weight * dot(diagnostic.template_centroid[index],
                                diagnostic.template_centroid[index]);
  }
  return denominator > 0.0 ? numerator / denominator : quiet_nan();
}

double order_statistic(std::vector<double> values, const std::size_t index) {
  if (values.empty()) {
    return quiet_nan();
  }
  const std::size_t bounded_index = std::min(index, values.size() - 1U);
  std::nth_element(values.begin(), values.begin() + bounded_index,
                   values.end());
  return values[bounded_index];
}

double median(std::vector<double> values) {
  const std::size_t index = values.size() / 2U;
  return order_statistic(std::move(values), index);
}

double p95(std::vector<double> values) {
  const std::size_t index = static_cast<std::size_t>(
      0.95 * static_cast<double>(values.size()));
  return order_statistic(std::move(values), index);
}

double median_radial_velocity(const std::vector<int>& cells,
                              const std::vector<VectorRz>& current_centroid,
                              const std::vector<VectorRz>& centroid_velocity) {
  std::vector<double> radial_velocity;
  radial_velocity.reserve(cells.size());
  for (const int cell : cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double radius = std::hypot(current_centroid[index].r,
                                     current_centroid[index].z);
    if (!(radius > 0.0) || !std::isfinite(radius)) {
      continue;
    }
    const double value =
        dot(centroid_velocity[index], current_centroid[index]) / radius;
    if (std::isfinite(value)) {
      radial_velocity.push_back(value);
    }
  }
  return median(std::move(radial_velocity));
}

void emit_wideband_top_k(
    const ShadowHomothetyState& diagnostic,
    const int step,
    const double t,
    const std::vector<VectorRz>& current_centroid,
    const std::vector<VectorRz>& centroid_velocity,
    const std::vector<double>& current_volume) {
  const int requested_top_k = shadow_homothety_top_k();
  if (requested_top_k == 0 || diagnostic.rank != 0) {
    return;
  }
  TENRYU_ASSERT(diagnostic.wideband_cells.size() ==
                    diagnostic.wideband_theta0_deg.size(),
                "shadow homothety wideband metadata size changed");

  const std::size_t wideband_count = diagnostic.wideband_cells.size();
  std::vector<double> radial_velocity(wideband_count);
  std::vector<double> volume_ratio(wideband_count);
  std::vector<std::size_t> indices(wideband_count);
  for (std::size_t slot = 0; slot < wideband_count; ++slot) {
    const int cell = diagnostic.wideband_cells[slot];
    const std::size_t index = static_cast<std::size_t>(cell);
    const double radius = std::hypot(current_centroid[index].r,
                                     current_centroid[index].z);
    radial_velocity[slot] =
        dot(centroid_velocity[index], current_centroid[index]) / radius;
    volume_ratio[slot] =
        current_volume[index] / diagnostic.initial_volume[index];
    indices[slot] = slot;
  }

  const std::size_t top_count = std::min(
      wideband_count, static_cast<std::size_t>(requested_top_k));
  std::partial_sort(
      indices.begin(), indices.begin() + top_count, indices.end(),
      [&radial_velocity, &diagnostic](const std::size_t lhs,
                                      const std::size_t rhs) {
        const double lhs_magnitude =
            std::isnan(radial_velocity[lhs])
                ? -std::numeric_limits<double>::infinity()
                : std::abs(radial_velocity[lhs]);
        const double rhs_magnitude =
            std::isnan(radial_velocity[rhs])
                ? -std::numeric_limits<double>::infinity()
                : std::abs(radial_velocity[rhs]);
        if (lhs_magnitude != rhs_magnitude) {
          return lhs_magnitude > rhs_magnitude;
        }
        return diagnostic.wideband_cells[lhs] <
               diagnostic.wideband_cells[rhs];
      });

  for (std::size_t rank = 0; rank < top_count; ++rank) {
    const std::size_t slot = indices[rank];
    const int cell = diagnostic.wideband_cells[slot];
    const std::size_t index = static_cast<std::size_t>(cell);
    std::ostringstream line;
    line << "[shadow-homothety-topk] step=" << step
         << " t=" << std::scientific << std::setprecision(9) << t
         << " k=" << rank
         << " cell=" << cell
         << " r0=" << std::setprecision(6)
         << diagnostic.initial_radius[index]
         << " th0_deg=" << std::fixed << std::setprecision(3)
         << diagnostic.wideband_theta0_deg[slot]
         << " ur=" << std::scientific << std::setprecision(6)
         << radial_velocity[slot]
         << " vratio=" << volume_ratio[slot];
    core::log_info(line.str());
  }
}

void emit_sample(const ShadowHomothetyState& diagnostic,
                 const ShadowHomothetySample& sample,
                 const bool final) {
  if (diagnostic.rank != 0) {
    return;
  }
  std::ostringstream line;
  line << std::scientific
       << (final ? "[shadow-homothety-end] FINAL"
                 : "[shadow-homothety-end]")
       << " step=" << sample.step
       << " t=" << std::setprecision(9) << sample.t
       << " a_D=" << std::setprecision(12) << sample.a_driver
       << " ann1_rel=" << std::setprecision(3) << sample.ann1_relative
       << " ann2_rel=" << sample.ann2_relative
       << " H=" << std::setprecision(6) << sample.homothety_rate
       << " eps_med=" << std::setprecision(3) << sample.epsilon_median
       << " eps_p95=" << sample.epsilon_p95
       << " Af=" << std::setprecision(6) << sample.feather_area
       << " Vf=" << sample.feather_volume
       << " dAf_dt=" << std::setprecision(3) << sample.feather_area_rate
       << " dVf_dt=" << sample.feather_volume_rate
       << " ur_F=" << std::setprecision(6)
       << sample.feather_radial_velocity
       << " ur_Rplus=" << sample.rplus_radial_velocity
       << " dufr=" << sample.radial_velocity_jump
       << " If=" << sample.interface_strain
       << " Vmin_ratio=" << sample.rplus_min_volume_ratio;
  core::log_info(line.str());
}

void emit_post_lagrange_sample(
    const ShadowHomothetyState& diagnostic,
    const ShadowHomothetyPostLagrangeSample& sample,
    const bool final) {
  if (diagnostic.rank != 0) {
    return;
  }
  std::ostringstream line;
  line << std::scientific
       << (final ? "[shadow-homothety-plag] FINAL"
                 : "[shadow-homothety-plag]")
       << " step=" << sample.step
       << " t=" << std::setprecision(9) << sample.t
       << " a_plag=" << std::setprecision(12) << sample.a_plag
       << " a_inc=" << std::setprecision(6) << sample.a_inc
       << " A_cum=" << std::setprecision(12) << sample.a_cumulative
       << " H_inst=" << std::setprecision(6) << sample.instantaneous_rate
       << " eps_med=" << std::setprecision(3) << sample.epsilon_median
       << " eps_p95=" << sample.epsilon_p95;
  core::log_info(line.str());
}

std::vector<VectorRz> capture_position_centroids(
    const ShadowHomothetyState& diagnostic,
    const core::State& state) {
  const std::size_t n_cells = diagnostic.template_centroid.size();
  const std::size_t n_nodes = diagnostic.template_node_r.size();
  TENRYU_ASSERT(state.x_r.size() == n_nodes && state.x_z.size() == n_nodes,
                "shadow homothety node position field size changed");

  std::vector<double> node_r;
  std::vector<double> node_z;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);

  std::vector<VectorRz> current_centroid(n_cells);
  for (std::size_t index = 0; index < n_cells; ++index) {
    current_centroid[index] = cell_vertex_mean(
        diagnostic, static_cast<int>(index), node_r, node_z);
  }
  return current_centroid;
}

void capture_post_lagrange_epsilon(
    const ShadowHomothetyState& diagnostic,
    const std::vector<VectorRz>& current_centroid,
    ShadowHomothetyPostLagrangeSample& sample) {
  std::vector<double> epsilon;
  epsilon.reserve(diagnostic.driver_cells.size());
  for (const int cell : diagnostic.driver_cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double denominator = dot(diagnostic.template_centroid[index],
                                   diagnostic.template_centroid[index]);
    const double cell_scale =
        dot(diagnostic.template_centroid[index], current_centroid[index]) /
        denominator;
    epsilon.push_back(
        std::abs(cell_scale - sample.a_plag) /
        std::max(sample.a_plag, 1.0e-30));
  }
  sample.epsilon_median = median(epsilon);
  sample.epsilon_p95 = p95(std::move(epsilon));
}

ShadowHomothetySample capture_sample(ShadowHomothetyState& diagnostic,
                                     const core::State& state,
                                     const int step,
                                     const double t) {
  const std::size_t n_cells = diagnostic.template_centroid.size();
  const std::size_t n_nodes = diagnostic.template_node_r.size();
  TENRYU_ASSERT(state.x_r.size() == n_nodes && state.x_z.size() == n_nodes &&
                    state.v_r.size() == n_nodes && state.v_z.size() == n_nodes,
                "shadow homothety node field size changed");
  TENRYU_ASSERT(state.vol.size() == n_cells && state.mass.size() == n_cells,
                "shadow homothety cell field size changed");

  std::vector<double> node_r;
  std::vector<double> node_z;
  std::vector<double> velocity_r;
  std::vector<double> velocity_z;
  std::vector<double> current_volume;
  std::vector<double> current_mass;
  state.x_r.copy_to_host(node_r);
  state.x_z.copy_to_host(node_z);
  state.v_r.copy_to_host(velocity_r);
  state.v_z.copy_to_host(velocity_z);
  state.vol.copy_to_host(current_volume);
  state.mass.copy_to_host(current_mass);
  TENRYU_ASSERT(current_mass.size() == n_cells,
                "shadow homothety mass copy is incomplete");

  std::vector<VectorRz> current_centroid(n_cells);
  std::vector<VectorRz> centroid_velocity(n_cells);
  for (std::size_t index = 0; index < n_cells; ++index) {
    const int cell = static_cast<int>(index);
    current_centroid[index] =
        cell_vertex_mean(diagnostic, cell, node_r, node_z);
    centroid_velocity[index] =
        cell_vertex_mean(diagnostic, cell, velocity_r, velocity_z);
  }

  ShadowHomothetySample sample;
  sample.step = step;
  sample.t = t;
  sample.a_driver = weighted_scale(
      diagnostic, diagnostic.driver_cells, current_centroid);
  const double a_ann1 = weighted_scale(
      diagnostic, diagnostic.annulus1_cells, current_centroid);
  const double a_ann2 = weighted_scale(
      diagnostic, diagnostic.annulus2_cells, current_centroid);
  sample.ann1_relative =
      (a_ann1 - sample.a_driver) / sample.a_driver;
  sample.ann2_relative =
      (a_ann2 - sample.a_driver) / sample.a_driver;

  std::vector<double> epsilon;
  epsilon.reserve(diagnostic.driver_cells.size());
  for (const int cell : diagnostic.driver_cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double denominator = dot(diagnostic.template_centroid[index],
                                   diagnostic.template_centroid[index]);
    const double cell_scale =
        dot(diagnostic.template_centroid[index], current_centroid[index]) /
        denominator;
    epsilon.push_back(
        std::abs(cell_scale - sample.a_driver) /
        std::max(sample.a_driver, 1.0e-30));
  }
  sample.epsilon_median = median(epsilon);
  sample.epsilon_p95 = p95(std::move(epsilon));

  double sample_dt = 0.0;
  if (diagnostic.previous_sample_valid) {
    sample_dt = t - diagnostic.previous_t;
    if (sample_dt > 0.0) {
      sample.homothety_rate =
          -(sample.a_driver - diagnostic.previous_a_driver) /
          (sample_dt * sample.a_driver);
    }
  }

  std::vector<double> feather_volume;
  feather_volume.reserve(diagnostic.feather_cells.size());
  for (const int cell : diagnostic.feather_cells) {
    const std::size_t index = static_cast<std::size_t>(cell);
    sample.feather_area +=
        planar_cell_area(diagnostic, cell, node_r, node_z);
    sample.feather_volume += current_volume[index];
    feather_volume.push_back(current_volume[index]);
  }
  if (diagnostic.previous_sample_valid && sample_dt > 0.0) {
    TENRYU_ASSERT(diagnostic.previous_feather_volume.size() ==
                      feather_volume.size(),
                  "shadow homothety feather snapshot size changed");
    double volume_delta = 0.0;
    for (std::size_t slot = 0; slot < feather_volume.size(); ++slot) {
      volume_delta +=
          feather_volume[slot] - diagnostic.previous_feather_volume[slot];
    }
    sample.feather_area_rate =
        (sample.feather_area - diagnostic.previous_feather_area) / sample_dt;
    sample.feather_volume_rate = volume_delta / sample_dt;
  }

  sample.feather_radial_velocity = median_radial_velocity(
      diagnostic.feather_cells, current_centroid, centroid_velocity);
  sample.rplus_radial_velocity = median_radial_velocity(
      diagnostic.rplus_cells, current_centroid, centroid_velocity);
  sample.radial_velocity_jump =
      sample.rplus_radial_velocity - sample.feather_radial_velocity;

  std::vector<double> rplus_volume;
  std::vector<double> interface_strain;
  rplus_volume.reserve(diagnostic.rplus_cells.size());
  interface_strain.reserve(diagnostic.rplus_cells.size());
  double minimum_volume_ratio = std::numeric_limits<double>::infinity();
  for (std::size_t slot = 0; slot < diagnostic.rplus_cells.size(); ++slot) {
    const int cell = diagnostic.rplus_cells[slot];
    const std::size_t index = static_cast<std::size_t>(cell);
    const double volume = current_volume[index];
    rplus_volume.push_back(volume);
    if (diagnostic.initial_volume[index] > 0.0 && std::isfinite(volume)) {
      minimum_volume_ratio =
          std::min(minimum_volume_ratio,
                   volume / diagnostic.initial_volume[index]);
    }
    if (diagnostic.previous_sample_valid && sample_dt > 0.0) {
      TENRYU_ASSERT(diagnostic.previous_rplus_volume.size() ==
                        diagnostic.rplus_cells.size(),
                    "shadow homothety RPLUS snapshot size changed");
      const double previous_volume = diagnostic.previous_rplus_volume[slot];
      if (volume > 0.0 && previous_volume > 0.0 &&
          std::isfinite(volume) && std::isfinite(previous_volume)) {
        interface_strain.push_back(
            (std::log(volume) - std::log(previous_volume)) / sample_dt);
      }
    }
  }
  if (diagnostic.previous_sample_valid && sample_dt > 0.0) {
    sample.interface_strain = median(std::move(interface_strain));
  }
  sample.rplus_min_volume_ratio =
      std::isfinite(minimum_volume_ratio) ? minimum_volume_ratio : quiet_nan();

  emit_wideband_top_k(diagnostic, step, t, current_centroid,
                      centroid_velocity, current_volume);

  diagnostic.previous_sample_valid = true;
  diagnostic.previous_t = t;
  diagnostic.previous_a_driver = sample.a_driver;
  diagnostic.previous_feather_area = sample.feather_area;
  diagnostic.previous_feather_volume = std::move(feather_volume);
  diagnostic.previous_rplus_volume = std::move(rplus_volume);
  return sample;
}

void initialize_diagnostic(ShadowHomothetyState& diagnostic,
                           const core::State& state,
                           const core::Config& cfg,
                           const int rank) {
  TENRYU_ASSERT(state.mesh.dim == 2,
                "shadow homothety requires a two-dimensional mesh");
  TENRYU_ASSERT(state.mesh.topo.multiblock.has_value(),
                "shadow homothety requires multiblock topology");
  const int n_cells = state.mesh.topo.n_cells;
  const int n_nodes = state.mesh.topo.n_nodes;
  TENRYU_ASSERT(n_cells > 0 && n_nodes > 0,
                "shadow homothety requires nonempty mesh topology");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>(n_nodes) &&
                    state.x_z.size() == static_cast<std::size_t>(n_nodes) &&
                    state.mass.size() == static_cast<std::size_t>(n_cells) &&
                    state.vol.size() == static_cast<std::size_t>(n_cells),
                "shadow homothety initial field sizes are incomplete");

  state.x_r.copy_to_host(diagnostic.template_node_r);
  state.x_z.copy_to_host(diagnostic.template_node_z);
  state.mass.copy_to_host(diagnostic.initial_mass);
  state.vol.copy_to_host(diagnostic.initial_volume);
  state.mesh.multiblock_cell_node_csr_offsets.copy_to_host(
      diagnostic.cell_node_offsets);
  state.mesh.multiblock_cell_node_csr_indices.copy_to_host(
      diagnostic.cell_node_indices);
  diagnostic.cell_nverts = state.mesh.cell_nverts;
  TENRYU_ASSERT(diagnostic.cell_node_offsets.size() ==
                    static_cast<std::size_t>(n_cells) + 1U,
                "shadow homothety cell-node CSR offsets are incomplete");
  TENRYU_ASSERT(diagnostic.cell_nverts.empty() ||
                    diagnostic.cell_nverts.size() ==
                        static_cast<std::size_t>(n_cells),
                "shadow homothety active vertex counts are incomplete");

  diagnostic.template_centroid.resize(static_cast<std::size_t>(n_cells));
  diagnostic.initial_radius.resize(static_cast<std::size_t>(n_cells));
  std::vector<std::vector<int>> node_cells(static_cast<std::size_t>(n_nodes));
  for (int cell = 0; cell < n_cells; ++cell) {
    validate_cell_csr(diagnostic, cell, n_nodes);
    const std::size_t index = static_cast<std::size_t>(cell);
    diagnostic.template_centroid[index] = cell_vertex_mean(
        diagnostic, cell, diagnostic.template_node_r,
        diagnostic.template_node_z);
    diagnostic.initial_radius[index] =
        std::hypot(diagnostic.template_centroid[index].r,
                   diagnostic.template_centroid[index].z);
    const int offset = diagnostic.cell_node_offsets[index];
    const int nverts = active_nverts(diagnostic, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = diagnostic.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      node_cells[static_cast<std::size_t>(node)].push_back(cell);
    }
  }

  std::vector<std::uint8_t> excluded(static_cast<std::size_t>(n_cells), 0U);
  for (int cell = 0; cell < n_cells; ++cell) {
    if (active_nverts(diagnostic, cell) == 4) {
      continue;
    }
    excluded[static_cast<std::size_t>(cell)] = 1U;
    const int offset =
        diagnostic.cell_node_offsets[static_cast<std::size_t>(cell)];
    const int nverts = active_nverts(diagnostic, cell);
    for (int corner = 0; corner < nverts; ++corner) {
      const int node = diagnostic.cell_node_indices[
          static_cast<std::size_t>(offset + corner)];
      for (const int neighbor : node_cells[static_cast<std::size_t>(node)]) {
        excluded[static_cast<std::size_t>(neighbor)] = 1U;
      }
    }
  }

  const double outer_radius = cfg.numerics.ale.euler_window.rad_out;
  const double transition_width =
      cfg.numerics.ale.euler_window.transition_width;
  for (int cell = 0; cell < n_cells; ++cell) {
    const std::size_t index = static_cast<std::size_t>(cell);
    const double radius = diagnostic.initial_radius[index];
    const VectorRz centroid = diagnostic.template_centroid[index];
    const bool active = std::isfinite(centroid.r) &&
                        std::isfinite(centroid.z) &&
                        diagnostic.initial_volume[index] > 0.0 &&
                        std::isfinite(diagnostic.initial_volume[index]);
    const bool regular = excluded[index] == 0U;
    if (active && radius > outer_radius && radius <= 2.2e-3) {
      diagnostic.wideband_cells.push_back(cell);
      diagnostic.wideband_theta0_deg.push_back(
          std::atan2(centroid.r, centroid.z) * 180.0 / std::acos(-1.0));
    }
    if (active && regular && radius >= 0.25 * outer_radius &&
        radius <= 0.75 * outer_radius) {
      diagnostic.driver_cells.push_back(cell);
    }
    if (active && regular && radius >= 0.78 * outer_radius &&
        radius <= 0.88 * outer_radius) {
      diagnostic.annulus1_cells.push_back(cell);
    }
    if (active && regular && radius >= 0.88 * outer_radius &&
        radius <= 0.98 * outer_radius) {
      diagnostic.annulus2_cells.push_back(cell);
    }
    if (radius >= outer_radius - transition_width &&
        radius <= outer_radius) {
      diagnostic.feather_cells.push_back(cell);
    }
    if (radius > outer_radius &&
        radius <= outer_radius + 3.0 * transition_width) {
      diagnostic.rplus_cells.push_back(cell);
    }
  }

  diagnostic.rank = rank;
  diagnostic.live_state = &state;
  diagnostic.previous_post_lagrange_t = state.t;
  diagnostic.initialized = true;
  if (rank == 0) {
    std::ostringstream manifest;
    manifest << std::scientific << std::setprecision(9)
             << "[shadow-homothety] manifest R_out=" << outer_radius
             << " W=" << transition_width
             << " nD=" << diagnostic.driver_cells.size()
             << " nAnn1=" << diagnostic.annulus1_cells.size()
             << " nAnn2=" << diagnostic.annulus2_cells.size()
             << " nF=" << diagnostic.feather_cells.size()
             << " nRplus=" << diagnostic.rplus_cells.size()
             << " nWide=" << diagnostic.wideband_cells.size();
    core::log_info(manifest.str());
  }
}

}  // namespace

void shadow_homothety_run_start(const core::State& state,
                                const core::Config& cfg,
                                const int rank) {
  if (!shadow_homothety_enabled()) {
    return;
  }
  ShadowHomothetyState& diagnostic = shadow_homothety_state();
  if (diagnostic.initialized || diagnostic.multirank_unsupported) {
    return;
  }
  int world_rank = 0;
  int n_ranks = 1;
  parallel::Partition::query_world(&world_rank, &n_ranks);
  (void)world_rank;
  if (n_ranks > 1) {
    diagnostic.multirank_unsupported = true;
    if (rank == 0 && !diagnostic.multirank_logged) {
      core::log_info("[shadow-homothety] multirank-unsupported");
      diagnostic.multirank_logged = true;
    }
    return;
  }
  initialize_diagnostic(diagnostic, state, cfg, rank);
}

void shadow_homothety_record_post_lagrange(const core::State& state,
                                           const int step,
                                           const double t) {
  if (!shadow_homothety_enabled()) {
    return;
  }
  ShadowHomothetyState& diagnostic = shadow_homothety_state();
  if (!diagnostic.initialized || diagnostic.multirank_unsupported ||
      diagnostic.flushed ||
      step == diagnostic.last_recorded_post_lagrange_step) {
    return;
  }

  const std::vector<VectorRz> current_centroid =
      capture_position_centroids(diagnostic, state);
  ShadowHomothetyPostLagrangeSample sample;
  sample.step = step;
  sample.t = t;
  sample.a_plag = weighted_scale(
      diagnostic, diagnostic.driver_cells, current_centroid);
  sample.a_inc = sample.a_plag / diagnostic.previous_end_scale;
  diagnostic.post_lagrange_cumulative *= sample.a_inc;
  sample.a_cumulative = diagnostic.post_lagrange_cumulative;
  const double dt_step = t - diagnostic.previous_post_lagrange_t;
  if (dt_step > 0.0) {
    sample.instantaneous_rate = -(sample.a_inc - 1.0) / dt_step;
  }
  if (step % shadow_homothety_cadence_steps() == 0) {
    capture_post_lagrange_epsilon(diagnostic, current_centroid, sample);
    emit_post_lagrange_sample(diagnostic, sample, false);
  } else {
    sample.epsilon_median = quiet_nan();
    sample.epsilon_p95 = quiet_nan();
  }

  diagnostic.previous_post_lagrange_t = t;
  diagnostic.last_recorded_post_lagrange_step = step;
  diagnostic.last_post_lagrange_sample = sample;
  diagnostic.last_post_lagrange_sample_valid = true;
}

void shadow_homothety_record_step(const core::State& state,
                                  const int step,
                                  const double t) {
  if (!shadow_homothety_enabled()) {
    return;
  }
  ShadowHomothetyState& diagnostic = shadow_homothety_state();
  if (!diagnostic.initialized || diagnostic.multirank_unsupported ||
      diagnostic.flushed) {
    return;
  }
  diagnostic.live_state = &state;
  diagnostic.latest_step = step;
  diagnostic.latest_t = t;
  diagnostic.has_recorded_step = true;
  const std::vector<VectorRz> current_centroid =
      capture_position_centroids(diagnostic, state);
  diagnostic.previous_end_scale = weighted_scale(
      diagnostic, diagnostic.driver_cells, current_centroid);
  if (step % shadow_homothety_cadence_steps() != 0) {
    return;
  }
  diagnostic.last_sample = capture_sample(diagnostic, state, step, t);
  diagnostic.last_sample_valid = true;
  emit_sample(diagnostic, diagnostic.last_sample, false);
}

void shadow_homothety_flush() {
  if (!shadow_homothety_enabled()) {
    return;
  }
  ShadowHomothetyState& diagnostic = shadow_homothety_state();
  if (!diagnostic.initialized || diagnostic.multirank_unsupported ||
      diagnostic.flushed) {
    return;
  }
  if (diagnostic.has_recorded_step &&
      (!diagnostic.last_sample_valid ||
       diagnostic.last_sample.step != diagnostic.latest_step)) {
    TENRYU_ASSERT(diagnostic.live_state != nullptr,
                  "shadow homothety final state is unavailable");
    diagnostic.last_sample =
        capture_sample(diagnostic, *diagnostic.live_state,
                       diagnostic.latest_step, diagnostic.latest_t);
    diagnostic.last_sample_valid = true;
  }
  if (diagnostic.last_sample_valid) {
    emit_sample(diagnostic, diagnostic.last_sample, true);
  }
  if (diagnostic.last_post_lagrange_sample_valid) {
    emit_post_lagrange_sample(
        diagnostic, diagnostic.last_post_lagrange_sample, true);
  }
  diagnostic.flushed = true;
}

}  // namespace tenryu::hydro
