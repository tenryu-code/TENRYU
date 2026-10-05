#include "laser/port_section_s1_host.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/constants.hpp"
#include "core/error.hpp"
#include "laser/sector_adapter.hpp"

namespace tenryu::laser::port_section {
namespace {

template <typename T>
std::vector<T> download(const T* device, const std::size_t count, const char* message) {
  std::vector<T> host(count);
  if (count > 0) {
    TENRYU_ASSERT(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost) ==
                      cudaSuccess,
                  message);
  }
  return host;
}

struct IncidentAttenuationContext {
  const std::vector<double>* rec_S = nullptr;
};

// Former laser.cu apply_incident_attenuation.
double apply_incident_attenuation(const double power_in, const std::int64_t record_index,
                                  const void* context) {
  const auto* attenuation = static_cast<const IncidentAttenuationContext*>(context);
  const double S_half = 0.5 * (*attenuation->rec_S)[static_cast<std::size_t>(record_index)];
  double power = power_in;
  const double dP_first = -power * std::expm1(-S_half);
  power -= dP_first;
  const double dP_second = -power * std::expm1(-S_half);
  power -= dP_second;
  return power;
}

}  // namespace

S1HostReference build_s1_table_host_reference(const S1DeviceInput& input) {
  const int n_rays = input.n_rays;
  const int cap_per_ray = input.cap_per_ray;
  const int n_cells = input.n_cells;
  const std::size_t stripe_size = static_cast<std::size_t>(n_rays) * static_cast<std::size_t>(cap_per_ray);
  const auto rec_count = download(input.rec_count, static_cast<std::size_t>(n_rays), "s1 reference: rec_count");
  const auto stripe_cell = download(input.rec_cell, stripe_size, "s1 reference: rec_cell");
  const auto stripe_mu = download(input.rec_mu, stripe_size, "s1 reference: rec_mu");
  const auto stripe_ds = download(input.rec_ds, stripe_size, "s1 reference: rec_ds");
  const auto stripe_S = download(input.rec_S, stripe_size, "s1 reference: rec_S");
  const auto ray_P0 = download(input.ray_P0, static_cast<std::size_t>(n_rays), "s1 reference: ray_P0");
  const auto ray_group_base =
      download(input.ray_group_base, static_cast<std::size_t>(n_rays), "s1 reference: ray_group_base");
  const auto r_edges =
      download(input.r_edges, static_cast<std::size_t>(n_cells) + 1U, "s1 reference: r_edges");
  const auto rho = download(input.rho, static_cast<std::size_t>(n_cells), "s1 reference: rho");
  const auto zbar = download(input.zbar, static_cast<std::size_t>(n_cells), "s1 reference: zbar");
  const auto A_eff = download(input.A_eff, static_cast<std::size_t>(n_cells), "s1 reference: A_eff");

  // Compaction of the record stripes (former laser.cu build_port_section_diagnostics).
  std::vector<std::int64_t> ray_rec_offset(static_cast<std::size_t>(n_rays) + 1, 0);
  for (int ray = 0; ray < n_rays; ++ray) {
    const int count = std::clamp(static_cast<int>(rec_count[static_cast<std::size_t>(ray)]), 0, cap_per_ray);
    ray_rec_offset[static_cast<std::size_t>(ray) + 1] = ray_rec_offset[static_cast<std::size_t>(ray)] + count;
  }
  const std::size_t n_records = static_cast<std::size_t>(ray_rec_offset.back());
  std::vector<std::int32_t> rec_cell(n_records);
  std::vector<float> rec_mu(n_records);
  std::vector<double> rec_ds(n_records);
  std::vector<double> rec_S(n_records);
  for (int ray = 0; ray < n_rays; ++ray) {
    const std::size_t source = static_cast<std::size_t>(ray) * static_cast<std::size_t>(cap_per_ray);
    const std::size_t destination = static_cast<std::size_t>(ray_rec_offset[static_cast<std::size_t>(ray)]);
    const std::size_t count = static_cast<std::size_t>(ray_rec_offset[static_cast<std::size_t>(ray) + 1] -
                                                       ray_rec_offset[static_cast<std::size_t>(ray)]);
    std::copy_n(stripe_cell.data() + source, count, rec_cell.data() + destination);
    std::copy_n(stripe_mu.data() + source, count, rec_mu.data() + destination);
    std::copy_n(stripe_ds.data() + source, count, rec_ds.data() + destination);
    std::copy_n(stripe_S.data() + source, count, rec_S.data() + destination);
  }

  std::vector<double> cell_r_center(static_cast<std::size_t>(n_cells));
  std::vector<double> cell_eps(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    const std::size_t index = static_cast<std::size_t>(cell);
    cell_r_center[index] = 0.5 * (r_edges[index] + r_edges[index + 1]);
    const double z = std::max(zbar[index], 0.0);
    const double a = std::max(A_eff[index], 1.0e-30);
    const double ne = rho[index] * z / (a * core::constants::proton_mass);
    cell_eps[index] = std::max(0.0, 1.0 - ne / input.n_crit);
  }

  std::vector<double> impact_parameter(static_cast<std::size_t>(n_rays));
  for (int ray = 0; ray < n_rays; ++ray) {
    impact_parameter[static_cast<std::size_t>(ray)] = input.impact_spacing * (static_cast<double>(ray) + 0.5);
  }

  const IncidentAttenuationContext attenuation_context{&rec_S};
  sector_adapter::AdapterInput adapter_input{};
  adapter_input.n_rays = n_rays;
  adapter_input.ray_rec_offset = ray_rec_offset.data();
  adapter_input.rec_cell = rec_cell.data();
  adapter_input.rec_mu = rec_mu.data();
  adapter_input.rec_ds = rec_ds.data();
  adapter_input.ray_P0 = ray_P0.data();
  adapter_input.ray_impact_parameter = impact_parameter.data();
  adapter_input.cell_r_center = cell_r_center.data();
  adapter_input.r_edges = r_edges.data();
  adapter_input.cell_eps = cell_eps.data();
  adapter_input.n_cells = n_cells;
  adapter_input.attenuate = apply_incident_attenuation;
  adapter_input.attenuation_context = &attenuation_context;
  std::vector<int> source_ray_indices;
  const std::vector<sector_ps::RayPath> ray_paths =
      sector_adapter::build_ray_paths(adapter_input, &source_ray_indices);

  S1HostReference result;
  result.n_paths = static_cast<int>(ray_paths.size());
  result.ray_bin.resize(source_ray_indices.size());
  for (std::size_t ray = 0; ray < source_ray_indices.size(); ++ray) {
    result.ray_bin[ray] =
        ray_group_base[static_cast<std::size_t>(source_ray_indices[ray])] % input.n_bins;
  }

  std::vector<double> shell_eps(r_edges.size(), 0.0);
  for (std::size_t shell = 0; shell < r_edges.size(); ++shell) {
    const double radius = r_edges[shell];
    if (radius <= cell_r_center.front()) {
      shell_eps[shell] = cell_eps.front();
      continue;
    }
    if (radius >= cell_r_center.back()) {
      shell_eps[shell] = cell_eps.back();
      continue;
    }
    const auto upper = std::upper_bound(cell_r_center.begin(), cell_r_center.end(), radius);
    const std::size_t hi = static_cast<std::size_t>(upper - cell_r_center.begin());
    const std::size_t lo = hi - 1;
    const double weight = (radius - cell_r_center[lo]) / (cell_r_center[hi] - cell_r_center[lo]);
    shell_eps[shell] = cell_eps[lo] + weight * (cell_eps[hi] - cell_eps[lo]);
  }

  sector_ps::PhaseSpaceParams params{};
  params.caustic_area_rel_tol = input.caustic_area_rel_tol;
  params.bouguer_tol = 0.5;
  params.bouguer_tol_fd = 0.5;
  std::vector<sector_ps::RayAnnotation> annotations;
  const sector_ps::PhaseSpaceTable table =
      sector_ps::build_table(ray_paths, r_edges, shell_eps, params, annotations);
  result.ledger = sector_ps::exclusion_ledger(table);
  for (const sector_ps::RayPath& path : ray_paths) {
    result.total_nodes += static_cast<long long>(path.r.size());
  }
  for (const sector_ps::RayAnnotation& annotation : annotations) {
    result.bouguer_drift_max = std::max(result.bouguer_drift_max, annotation.bouguer_max_drift);
  }
  result.flat = sector_ps::flatten_table(table);
  result.total_crossings = static_cast<long long>(result.flat.theta.size());

  // Former laser.cu fill_port_section_ray_map.
  constexpr int kThetaBins = kRayMapThetaBins;
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kErgPerSToW = 1.0e-7;
  const std::size_t n_shells = r_edges.size();
  result.ray_map.assign(n_shells * static_cast<std::size_t>(kThetaBins) * 2U, 0.0);
  for (std::size_t shell = 0; shell < n_shells; ++shell) {
    for (int sheet = 0; sheet < 2; ++sheet) {
      std::array<double, kThetaBins> weighted_intensity{};
      std::array<double, kThetaBins> power{};
      for (const sector_ps::CrossingView& crossing :
           sector_ps::crossings(table, static_cast<int>(shell), sheet)) {
        const double scaled = std::clamp(crossing.theta, 0.0, kPi) * static_cast<double>(kThetaBins) / kPi;
        const int bin = std::clamp(static_cast<int>(std::floor(scaled)), 0, kThetaBins - 1);
        const double intensity = crossing.P / crossing.area;
        weighted_intensity[static_cast<std::size_t>(bin)] += crossing.P * intensity;
        power[static_cast<std::size_t>(bin)] += crossing.P;
      }
      for (int bin = 0; bin < kThetaBins; ++bin) {
        const std::size_t index =
            (shell * static_cast<std::size_t>(kThetaBins) + static_cast<std::size_t>(bin)) * 2U +
            static_cast<std::size_t>(sheet);
        if (power[static_cast<std::size_t>(bin)] > 0.0) {
          result.ray_map[index] =
              weighted_intensity[static_cast<std::size_t>(bin)] / power[static_cast<std::size_t>(bin)] * kErgPerSToW;
        }
      }
    }
  }
  return result;
}

}  // namespace tenryu::laser::port_section
