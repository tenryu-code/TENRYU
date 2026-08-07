#include "diagnostics/diagnostics.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "laser/laser_mesh.cuh"

namespace tenryu::diagnostics {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

template <typename Tag>
std::vector<double> to_host(const core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

template <typename T>
std::vector<T> to_host_device_array(const core::DeviceArray<T>& field) {
  std::vector<T> host;
  field.copy_to_host(host);
  return host;
}

bool is_sorted_non_decreasing(const std::vector<double>& values) {
  return std::is_sorted(values.begin(), values.end());
}

bool extract_structured_nodes_2d(const core::State& state,
                                 std::vector<double>& node_r,
                                 std::vector<double>& node_z) {
  if (state.mesh.dim != 2) {
    return false;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return false;
  }
  const int stride = nz + 1;
  const int n_nodes = (nr + 1) * (nz + 1);
  if (static_cast<int>(state.x_r.size()) != n_nodes ||
      static_cast<int>(state.x_z.size()) != n_nodes) {
    return false;
  }

  const auto x_r = to_host(state.x_r);
  const auto x_z = to_host(state.x_z);

  node_r.assign(static_cast<std::size_t>(nr + 1), 0.0);
  node_z.assign(static_cast<std::size_t>(nz + 1), 0.0);
  for (int i = 0; i <= nr; ++i) {
    node_r[static_cast<std::size_t>(i)] = x_r[static_cast<std::size_t>(i * stride)];
  }
  for (int j = 0; j <= nz; ++j) {
    node_z[static_cast<std::size_t>(j)] = x_z[static_cast<std::size_t>(j)];
  }

  return is_sorted_non_decreasing(node_r) && is_sorted_non_decreasing(node_z);
}

inline double clamp01_host(const double x) {
  if (!std::isfinite(x)) {
    return 0.0;
  }
  return std::min(1.0, std::max(0.0, x));
}

double first_nonvoid_ideal_gamma(const core::Config& cfg) {
  const int mat = cfg.materials.first_nonvoid_material_index();
  if (mat >= 0 &&
      static_cast<std::size_t>(mat) < cfg.materials.materials.size()) {
    const double gamma =
        cfg.materials.materials[static_cast<std::size_t>(mat)].ideal_gas_gamma;
    if (gamma > 1.0 && std::isfinite(gamma)) {
      return gamma;
    }
  }
  return 5.0 / 3.0;
}

double central_macro_boundary_mean_radius(
    const core::CentralPseudoCoreState& pc,
    const std::vector<double>& node_r,
    const std::vector<double>& node_z) {
  const int n_boundary = static_cast<int>(pc.boundary_nodes_ordered.size());
  if (n_boundary <= 0) {
    return 0.0;
  }
  long double r_sum = 0.0L;
  long double z_sum = 0.0L;
  int count = 0;
  for (const int n : pc.boundary_nodes_ordered) {
    if (n < 0 || static_cast<std::size_t>(n) >= node_r.size() ||
        static_cast<std::size_t>(n) >= node_z.size()) {
      continue;
    }
    r_sum += static_cast<long double>(node_r[static_cast<std::size_t>(n)]);
    z_sum += static_cast<long double>(node_z[static_cast<std::size_t>(n)]);
    ++count;
  }
  if (count <= 0) {
    return 0.0;
  }
  const double r = static_cast<double>(r_sum / static_cast<long double>(count));
  const double z = static_cast<double>(z_sum / static_cast<long double>(count));
  return std::hypot(r, z);
}

double weighted_quantile(std::vector<std::pair<double, double>> samples,
                         const double fraction) {
  if (samples.empty()) {
    return 0.0;
  }
  samples.erase(std::remove_if(samples.begin(),
                               samples.end(),
                               [](const auto& sample) {
                                 return !(sample.second > 0.0) ||
                                        !std::isfinite(sample.first) ||
                                        !std::isfinite(sample.second);
                               }),
                samples.end());
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(),
            samples.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
  long double total = 0.0L;
  for (const auto& sample : samples) {
    total += static_cast<long double>(sample.second);
  }
  if (!(total > 0.0L)) {
    return 0.0;
  }
  const long double target =
      std::min(1.0, std::max(0.0, fraction)) * total;
  long double cumulative = 0.0L;
  for (const auto& sample : samples) {
    cumulative += static_cast<long double>(sample.second);
    if (cumulative >= target) {
      return sample.first;
    }
  }
  return samples.back().first;
}

inline double finite_or_zero_host(const double value) {
  return std::isfinite(value) ? value : 0.0;
}

struct CellVelocityHostView {
  std::vector<double> v_r;
  std::vector<double> v_z;
  std::vector<int> cell_node_offsets;
  std::vector<int> cell_node_indices;
  bool has_multiblock_cells = false;
  int dim = 1;
  int nr = 0;
  int nz = 1;
  std::size_t n_cells = 0;
};

CellVelocityHostView make_cell_velocity_host_view(const core::State& state,
                                                  const std::size_t n_cells) {
  CellVelocityHostView view{};
  view.dim = state.mesh.dim;
  view.nr = state.mesh.topo.nr;
  view.nz = state.mesh.topo.nz;
  view.n_cells = n_cells;
  view.v_r = to_host(state.v_r);
  if (state.mesh.dim == 2) {
    view.v_z = to_host(state.v_z);
    if (state.mesh.topo.multiblock.has_value()) {
      const auto& mb = *state.mesh.topo.multiblock;
      if (mb.cell_node_csr_offsets.size() == n_cells + 1U &&
          !mb.cell_node_csr_indices.empty()) {
        view.cell_node_offsets = mb.cell_node_csr_offsets;
        view.cell_node_indices = mb.cell_node_csr_indices;
        view.has_multiblock_cells = true;
      }
    }
    if (!view.has_multiblock_cells &&
        state.mesh.multiblock_cell_node_csr_offsets.size() == n_cells + 1U &&
        !state.mesh.multiblock_cell_node_csr_indices.empty()) {
      view.cell_node_offsets =
          to_host_device_array(state.mesh.multiblock_cell_node_csr_offsets);
      view.cell_node_indices =
          to_host_device_array(state.mesh.multiblock_cell_node_csr_indices);
      view.has_multiblock_cells =
          view.cell_node_offsets.size() == n_cells + 1U &&
          !view.cell_node_indices.empty();
    }
  }
  return view;
}

double cell_velocity_sq_host(const CellVelocityHostView& view,
                             const std::size_t c) {
  if (view.dim == 1) {
    if (view.v_r.size() != view.n_cells + 1U || c + 1U >= view.v_r.size()) {
      return 0.0;
    }
    const double vr =
        0.5 * (finite_or_zero_host(view.v_r[c]) +
               finite_or_zero_host(view.v_r[c + 1U]));
    return vr * vr;
  }

  if (view.dim != 2 || view.v_r.empty() || view.v_z.empty()) {
    return 0.0;
  }

  if (view.has_multiblock_cells &&
      c + 1U < view.cell_node_offsets.size()) {
    const int begin = view.cell_node_offsets[c];
    const int end = view.cell_node_offsets[c + 1U];
    if (begin >= 0 && end > begin &&
        static_cast<std::size_t>(end) <= view.cell_node_indices.size()) {
      long double vr = 0.0L;
      long double vz = 0.0L;
      int count = 0;
      for (int k = begin; k < end; ++k) {
        const int node = view.cell_node_indices[static_cast<std::size_t>(k)];
        if (node < 0 ||
            static_cast<std::size_t>(node) >= view.v_r.size() ||
            static_cast<std::size_t>(node) >= view.v_z.size()) {
          continue;
        }
        vr += static_cast<long double>(
            finite_or_zero_host(view.v_r[static_cast<std::size_t>(node)]));
        vz += static_cast<long double>(
            finite_or_zero_host(view.v_z[static_cast<std::size_t>(node)]));
        ++count;
      }
      if (count > 0) {
        const long double inv = 1.0L / static_cast<long double>(count);
        vr *= inv;
        vz *= inv;
        return static_cast<double>(vr * vr + vz * vz);
      }
    }
  }

  if (view.nr <= 0 || view.nz <= 0 ||
      view.v_r.size() != static_cast<std::size_t>((view.nr + 1) * (view.nz + 1)) ||
      view.v_z.size() != view.v_r.size()) {
    return 0.0;
  }
  const int i = static_cast<int>(c / static_cast<std::size_t>(view.nz));
  const int j = static_cast<int>(c - static_cast<std::size_t>(i * view.nz));
  if (i < 0 || i >= view.nr || j < 0 || j >= view.nz) {
    return 0.0;
  }
  const int stride = view.nz + 1;
  const std::size_t n00 = static_cast<std::size_t>(i * stride + j);
  const std::size_t n10 = static_cast<std::size_t>((i + 1) * stride + j);
  const std::size_t n11 = static_cast<std::size_t>((i + 1) * stride + j + 1);
  const std::size_t n01 = static_cast<std::size_t>(i * stride + j + 1);
  const double vr =
      0.25 * (finite_or_zero_host(view.v_r[n00]) +
              finite_or_zero_host(view.v_r[n10]) +
              finite_or_zero_host(view.v_r[n11]) +
              finite_or_zero_host(view.v_r[n01]));
  const double vz =
      0.25 * (finite_or_zero_host(view.v_z[n00]) +
              finite_or_zero_host(view.v_z[n10]) +
              finite_or_zero_host(view.v_z[n11]) +
              finite_or_zero_host(view.v_z[n01]));
  return vr * vr + vz * vz;
}

double sample_rho_2d(const std::vector<double>& node_r,
                     const std::vector<double>& node_z,
                     const std::vector<double>& rho,
                     const int nz,
                     const double r,
                     const double z) {
  if (node_r.size() < 2 || node_z.size() < 2 || rho.empty()) {
    return 0.0;
  }
  if (r < node_r.front() || r >= node_r.back()) {
    return 0.0;
  }
  if (z < node_z.front() || z >= node_z.back()) {
    return 0.0;
  }

  const auto r_it = std::upper_bound(node_r.begin(), node_r.end(), r);
  const auto z_it = std::upper_bound(node_z.begin(), node_z.end(), z);
  if (r_it == node_r.begin() || z_it == node_z.begin()) {
    return 0.0;
  }

  const int i = static_cast<int>(std::distance(node_r.begin(), r_it) - 1);
  const int j = static_cast<int>(std::distance(node_z.begin(), z_it) - 1);
  const std::size_t idx = static_cast<std::size_t>(i * nz + j);
  if (idx >= rho.size()) {
    return 0.0;
  }
  return rho[idx];
}

double integrate_rhoR_1d(const core::State& state,
                         const bool shell_only,
                         const double shell_threshold,
                         const std::vector<double>* rho_weight = nullptr) {
  if (state.mesh.dim != 1 || state.rho.empty()) {
    return 0.0;
  }

  const auto rho = to_host(state.rho);
  if (rho_weight != nullptr && rho_weight->size() != rho.size()) {
    return 0.0;
  }
  const auto nodes = to_host(state.x_r);
  if (nodes.size() != rho.size() + 1) {
    return 0.0;
  }

  double integral = 0.0;
  for (std::size_t c = 0; c < rho.size(); ++c) {
    const double dr = std::max(nodes[c + 1] - nodes[c], 0.0);
    if (dr <= 0.0) {
      continue;
    }
    const double weight =
        (rho_weight != nullptr) ? clamp01_host((*rho_weight)[c]) : 1.0;
    const double rho_masked = rho[c] * weight;
    const double rho_c =
        shell_only && rho[c] < shell_threshold ? 0.0 : rho_masked;
    integral += rho_c * dr;
  }
  return integral;
}

double ray_s_max(const std::vector<double>& node_r,
                 const std::vector<double>& node_z,
                 const double dir_r,
                 const double dir_z) {
  if (node_r.size() < 2 || node_z.size() < 2) {
    return 0.0;
  }

  const double r_min = node_r.front();
  const double r_max = node_r.back();
  const double z_min = node_z.front();
  const double z_max = node_z.back();

  if (!(r_min <= 0.0 && 0.0 <= r_max && z_min <= 0.0 && 0.0 <= z_max)) {
    return 0.0;
  }

  double s_max = std::numeric_limits<double>::infinity();
  if (dir_r > 1.0e-14) {
    s_max = std::min(s_max, r_max / dir_r);
  }
  if (dir_z > 1.0e-14) {
    s_max = std::min(s_max, z_max / dir_z);
  }
  if (dir_z < -1.0e-14) {
    s_max = std::min(s_max, z_min / dir_z);
  }

  if (!std::isfinite(s_max) || s_max <= 0.0) {
    return 0.0;
  }
  return s_max;
}

double integrate_rhoR_ray_2d(const std::vector<double>& node_r,
                             const std::vector<double>& node_z,
                             const std::vector<double>& rho,
                             const int nz,
                             const double angle_deg,
                             const bool shell_only,
                             const double shell_threshold) {
  const double theta = angle_deg * kPi / 180.0;
  const double dir_r = std::cos(theta);
  const double dir_z = std::sin(theta);
  if (dir_r <= 0.0 && std::abs(dir_z) <= 1.0e-14) {
    return 0.0;
  }

  const double s_max = ray_s_max(node_r, node_z, dir_r, dir_z);
  if (s_max <= 0.0) {
    return 0.0;
  }

  constexpr int kSamples = 512;
  const double ds = s_max / static_cast<double>(kSamples);

  const auto apply_shell = [&](const double rho_value) {
    if (!shell_only) {
      return rho_value;
    }
    return (rho_value >= shell_threshold) ? rho_value : 0.0;
  };

  double prev_s = 0.0;
  double prev = apply_shell(sample_rho_2d(node_r, node_z, rho, nz, 0.0, 0.0));
  double sum = 0.0;

  for (int k = 1; k <= kSamples; ++k) {
    const double s = ds * static_cast<double>(k);
    const double r = dir_r * s;
    const double z = dir_z * s;
    const double curr = apply_shell(sample_rho_2d(node_r, node_z, rho, nz, r, z));
    sum += 0.5 * (prev + curr) * (s - prev_s);
    prev_s = s;
    prev = curr;
  }

  return sum;
}

double find_shell_radius_1d(const core::State& state,
                            const double rho_threshold) {
  if (state.mesh.dim != 1 || state.rho.empty()) {
    return 0.0;
  }

  const auto rho = to_host(state.rho);
  const auto nodes = to_host(state.x_r);
  if (nodes.size() != rho.size() + 1) {
    return 0.0;
  }

  double radius = 0.0;
  for (std::size_t c = 0; c < rho.size(); ++c) {
    if (rho[c] >= rho_threshold) {
      radius = std::max(radius, nodes[c + 1]);
    }
  }
  return radius;
}

double find_isodensity_radius_2d(const std::vector<double>& node_r,
                                 const std::vector<double>& node_z,
                                 const std::vector<double>& rho,
                                 const int nz,
                                 const double angle_deg,
                                 const double rho_threshold) {
  const double theta = angle_deg * kPi / 180.0;
  const double dir_r = std::cos(theta);
  const double dir_z = std::sin(theta);
  const double s_max = ray_s_max(node_r, node_z, dir_r, dir_z);
  if (s_max <= 0.0) {
    return 0.0;
  }

  constexpr int kSamples = 1024;
  const double ds = s_max / static_cast<double>(kSamples);

  double prev_s = 0.0;
  double prev_rho = sample_rho_2d(node_r, node_z, rho, nz, 0.0, 0.0);
  double radius = (prev_rho >= rho_threshold) ? 0.0 : 0.0;

  for (int k = 1; k <= kSamples; ++k) {
    const double s = ds * static_cast<double>(k);
    const double r = dir_r * s;
    const double z = dir_z * s;
    const double curr_rho = sample_rho_2d(node_r, node_z, rho, nz, r, z);

    if (curr_rho >= rho_threshold) {
      radius = s;
    }
    if (prev_rho >= rho_threshold && curr_rho < rho_threshold) {
      const double denom = prev_rho - curr_rho;
      if (denom > 1.0e-30) {
        const double frac = (prev_rho - rho_threshold) / denom;
        const double cross = prev_s + frac * (s - prev_s);
        radius = std::max(radius, cross);
      } else {
        radius = std::max(radius, prev_s);
      }
    }

    prev_s = s;
    prev_rho = curr_rho;
  }

  return radius;
}

double find_inner_isodensity_radius_2d(const std::vector<double>& node_r,
                                       const std::vector<double>& node_z,
                                       const std::vector<double>& rho,
                                       const int nz,
                                       const double angle_deg,
                                       const double rho_threshold) {
  const double theta = angle_deg * kPi / 180.0;
  const double dir_r = std::cos(theta);
  const double dir_z = std::sin(theta);
  const double s_max = ray_s_max(node_r, node_z, dir_r, dir_z);
  if (s_max <= 0.0) {
    return 0.0;
  }

  constexpr int kSamples = 1024;
  const double ds = s_max / static_cast<double>(kSamples);

  double prev_s = 0.0;
  double prev_rho = sample_rho_2d(node_r, node_z, rho, nz, 0.0, 0.0);
  if (prev_rho >= rho_threshold) {
    return 0.0;
  }

  for (int k = 1; k <= kSamples; ++k) {
    const double s = ds * static_cast<double>(k);
    const double r = dir_r * s;
    const double z = dir_z * s;
    const double curr_rho = sample_rho_2d(node_r, node_z, rho, nz, r, z);
    if (prev_rho < rho_threshold && curr_rho >= rho_threshold) {
      const double denom = curr_rho - prev_rho;
      if (denom > 1.0e-30) {
        const double frac = (rho_threshold - prev_rho) / denom;
        return prev_s + frac * (s - prev_s);
      }
      return s;
    }
    prev_s = s;
    prev_rho = curr_rho;
  }

  return 0.0;
}

double find_outer_isodensity_radius_1d(const std::vector<double>& nodes,
                                       const std::vector<double>& rho,
                                       const double rho_threshold) {
  if (nodes.size() != rho.size() + 1 || rho.empty()) {
    return 0.0;
  }

  double radius = 0.0;
  for (std::size_t c = 0; c < rho.size(); ++c) {
    if (rho[c] >= rho_threshold) {
      radius = std::max(radius, nodes[c + 1]);
    }
  }
  return radius;
}

double find_inner_isodensity_radius_1d(const std::vector<double>& nodes,
                                       const std::vector<double>& rho,
                                       const double rho_threshold) {
  if (nodes.size() != rho.size() + 1 || rho.empty()) {
    return 0.0;
  }

  for (std::size_t c = 0; c < rho.size(); ++c) {
    if (rho[c] >= rho_threshold) {
      return nodes[c];
    }
  }
  return 0.0;
}

double legendre_p(const int ell, const double x) {
  if (ell == 0) {
    return 1.0;
  }
  if (ell == 1) {
    return x;
  }

  double p_nm2 = 1.0;
  double p_nm1 = x;
  for (int l = 2; l <= ell; ++l) {
    const double p_n = ((2.0 * l - 1.0) * x * p_nm1 - (l - 1.0) * p_nm2) /
                       static_cast<double>(l);
    p_nm2 = p_nm1;
    p_nm1 = p_n;
  }
  return p_nm1;
}

double sum_laser_dep(const core::State& state) {
  if (state.laser_dep.empty()) {
    return 0.0;
  }
  const auto dep = to_host(state.laser_dep);
  long double sum = 0.0L;
  for (const double value : dep) {
    sum += static_cast<long double>(value);
  }
  return static_cast<double>(sum);
}

double compute_density_proxy_radius(const laser::LaserMesh* laser_mesh,
                                    const double* field,
                                    const double threshold_n_hat) {
  if (laser_mesh == nullptr || !laser_mesh->is_allocated() ||
      laser_mesh->n_nodes_r <= 0 || laser_mesh->n_nodes_z <= 0 ||
      field == nullptr || laser_mesh->node_R == nullptr) {
    return 0.0;
  }

  std::vector<double> node_r(static_cast<std::size_t>(laser_mesh->n_nodes_r), 0.0);
  std::vector<double> n_hat(static_cast<std::size_t>(laser_mesh->n_nodes()), 0.0);
  cuda_check(cudaMemcpy(node_r.data(),
                        laser_mesh->node_R,
                        node_r.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "compute_laser_pattern copy node_R failed");
  cuda_check(cudaMemcpy(n_hat.data(),
                        field,
                        n_hat.size() * sizeof(double),
                        cudaMemcpyDeviceToHost),
             "compute_laser_pattern copy n_e_hat failed");

  double radius = -1.0;
  for (std::size_t n = 0; n < n_hat.size(); ++n) {
    if (n_hat[n] < threshold_n_hat) {
      continue;
    }
    const int i = static_cast<int>(n) / laser_mesh->n_nodes_z;
    if (i < 0 || i >= laser_mesh->n_nodes_r) {
      continue;
    }
    const double r = node_r[static_cast<std::size_t>(i)];
    if (r >= 0.0) {
      radius = std::max(radius, r);
    }
  }

  if (!(radius >= 0.0) || !std::isfinite(radius)) {
    return 0.0;
  }
  return radius;
}

double compute_critical_surface_r(const laser::LaserMesh* laser_mesh) {
  return compute_density_proxy_radius(laser_mesh, laser_mesh != nullptr ? laser_mesh->n_e_hat_raw
                                                                        : nullptr,
                                      1.0);
}

double compute_near_critical_r(const laser::LaserMesh* laser_mesh) {
  constexpr double kNearCriticalNhat = 0.9;
  return compute_density_proxy_radius(laser_mesh, laser_mesh != nullptr ? laser_mesh->n_e_hat_raw
                                                                        : nullptr,
                                      kNearCriticalNhat);
}

double compute_absorption_weighted_r_1d(const core::State& state) {
  if (state.mesh.dim != 1 || state.laser_dep.empty() || state.x_r.empty()) {
    return 0.0;
  }
  const auto dep = to_host(state.laser_dep);
  const auto nodes = to_host(state.x_r);
  if (nodes.size() != dep.size() + 1) {
    return 0.0;
  }

  long double weighted_sum = 0.0L;
  long double dep_sum = 0.0L;
  for (std::size_t c = 0; c < dep.size(); ++c) {
    const double q = dep[c];
    if (!(q > 0.0)) {
      continue;
    }
    const double r_center = 0.5 * (nodes[c] + nodes[c + 1]);
    weighted_sum += static_cast<long double>(q) * static_cast<long double>(r_center);
    dep_sum += static_cast<long double>(q);
  }
  if (!(dep_sum > 0.0L)) {
    return 0.0;
  }
  return static_cast<double>(weighted_sum / dep_sum);
}

double compute_absorption_weighted_r_2d(const core::State& state) {
  if (state.mesh.dim != 2 || state.laser_dep.empty() || state.x_r.empty()) {
    return 0.0;
  }
  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  if (nr <= 0 || nz <= 0) {
    return 0.0;
  }

  const auto dep = to_host(state.laser_dep);
  const auto x_r = to_host(state.x_r);
  const int stride = nz + 1;
  const int n_cells = nr * nz;
  if (static_cast<int>(dep.size()) != n_cells) {
    return 0.0;
  }

  long double weighted_sum = 0.0L;
  long double dep_sum = 0.0L;
  for (int i = 0; i < nr; ++i) {
    for (int j = 0; j < nz; ++j) {
      const std::size_t c = static_cast<std::size_t>(i * nz + j);
      const double q = dep[c];
      if (!(q > 0.0)) {
        continue;
      }
      const int n00 = i * stride + j;
      const int n10 = (i + 1) * stride + j;
      const int n01 = i * stride + (j + 1);
      const int n11 = (i + 1) * stride + (j + 1);
      const double r_center =
          0.25 * (x_r[static_cast<std::size_t>(n00)] + x_r[static_cast<std::size_t>(n10)] +
                  x_r[static_cast<std::size_t>(n01)] + x_r[static_cast<std::size_t>(n11)]);
      weighted_sum += static_cast<long double>(q) * static_cast<long double>(r_center);
      dep_sum += static_cast<long double>(q);
    }
  }

  if (!(dep_sum > 0.0L)) {
    return 0.0;
  }
  return static_cast<double>(weighted_sum / dep_sum);
}

double compute_absorption_weighted_r(const core::State& state) {
  if (state.mesh.dim == 1) {
    return compute_absorption_weighted_r_1d(state);
  }
  if (state.mesh.dim == 2) {
    return compute_absorption_weighted_r_2d(state);
  }
  return 0.0;
}

}  // namespace

EnergyBudget compute_energy_budget_1d(const core::State& state) {
  const EnergyTotals totals = compute_energy_totals_1d(state);
  EnergyBudget budget{};
  budget.E_int_e = totals.E_int_e;
  budget.E_int_i = totals.E_int_i;
  budget.E_kin = totals.E_kin;
  budget.E_kinetic = totals.E_kin;
  budget.E_internal = totals.E_int_e + totals.E_int_i;
  budget.E_total = budget.E_internal + budget.E_kin;
  return budget;
}

EnergyBudget compute_energy_budget_2d(const core::State& state) {
  const EnergyTotals totals = compute_energy_totals_2d(state);
  EnergyBudget budget{};
  budget.E_int_e = totals.E_int_e;
  budget.E_int_i = totals.E_int_i;
  budget.E_kin = totals.E_kin;
  budget.E_kinetic = totals.E_kin;
  budget.E_internal = totals.E_int_e + totals.E_int_i;
  budget.E_total = budget.E_internal + budget.E_kin;
  return budget;
}

double relative_total_energy_error(const EnergyBudget& reference,
                                   const EnergyBudget& current) {
  const double denom = std::abs(reference.E_total);
  TENRYU_ASSERT(denom > 0.0,
                "Energy reference total must be non-zero for relative error");
  return std::abs(current.E_total - reference.E_total) / denom;
}

bool check_energy_conservation(const EnergyBudget& reference,
                               const EnergyBudget& current,
                               const double rel_tol) {
  TENRYU_ASSERT(rel_tol >= 0.0, "Energy conservation tolerance must be >= 0");
  return relative_total_energy_error(reference, current) <= rel_tol;
}

ArealDensityDiagnostics compute_areal_density(const core::State& state,
                                              const core::Config& cfg) {
  ArealDensityDiagnostics out{};
  out.angles_deg = cfg.diagnostics.areal_density.angles_deg;
  if (out.angles_deg.empty()) {
    out.angles_deg = {0.0};
  }
  out.rhoR.assign(out.angles_deg.size(), 0.0);

  if (state.rho.empty()) {
    return out;
  }

  const auto rho = to_host(state.rho);
  std::vector<double> gas_tracer_Y;
  const bool has_hotspot_tracer =
      state.gas_tracer_initialized &&
      state.gas_tracer_Y.size() == rho.size();
  if (has_hotspot_tracer) {
    gas_tracer_Y = to_host(state.gas_tracer_Y);
    out.rhoR_hotspot_tracer.assign(out.angles_deg.size(), 0.0);
  }
  const double rho_max = *std::max_element(rho.begin(), rho.end());
  const bool shell_only = (cfg.diagnostics.areal_density.r_range == "shell");
  const double shell_threshold = 0.1 * rho_max;

  if (state.mesh.dim == 1) {
    const double rhoR = integrate_rhoR_1d(state, shell_only, shell_threshold);
    std::fill(out.rhoR.begin(), out.rhoR.end(), rhoR);
    if (has_hotspot_tracer) {
      const double rhoR_hotspot =
          integrate_rhoR_1d(state, false, shell_threshold, &gas_tracer_Y);
      std::fill(out.rhoR_hotspot_tracer.begin(),
                out.rhoR_hotspot_tracer.end(),
                rhoR_hotspot);
    }
    return out;
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  if (!extract_structured_nodes_2d(state, node_r, node_z)) {
    return out;
  }

  const int nz = state.mesh.topo.nz;
  std::vector<double> rho_hotspot;
  if (has_hotspot_tracer) {
    rho_hotspot.assign(rho.size(), 0.0);
    for (std::size_t c = 0; c < rho.size(); ++c) {
      rho_hotspot[c] = rho[c] * clamp01_host(gas_tracer_Y[c]);
    }
  }
  for (std::size_t a = 0; a < out.angles_deg.size(); ++a) {
    out.rhoR[a] = integrate_rhoR_ray_2d(node_r,
                                        node_z,
                                        rho,
                                        nz,
                                        out.angles_deg[a],
                                        shell_only,
                                        shell_threshold);
    if (has_hotspot_tracer) {
      out.rhoR_hotspot_tracer[a] =
          integrate_rhoR_ray_2d(node_r,
                                node_z,
                                rho_hotspot,
                                nz,
                                out.angles_deg[a],
                                false,
                                shell_threshold);
    }
  }
  return out;
}

SphericityDiagnostics compute_sphericity(const core::State& state,
                                         const core::Config& cfg) {
  SphericityDiagnostics out{};
  out.modes = cfg.diagnostics.sphericity.modes;
  if (out.modes.empty()) {
    out.modes = {0, 2, 4};
  }
  out.coefficients.assign(out.modes.size(), 0.0);

  if (state.rho.empty()) {
    return out;
  }

  const auto rho = to_host(state.rho);
  const double rho_max = *std::max_element(rho.begin(), rho.end());
  const double rho_threshold = std::max(cfg.diagnostics.sphericity.rho_threshold,
                                        0.1 * rho_max);

  if (state.mesh.dim == 1) {
    const double radius = find_shell_radius_1d(state, rho_threshold);
    for (std::size_t k = 0; k < out.modes.size(); ++k) {
      out.coefficients[k] = (out.modes[k] == 0) ? radius : 0.0;
    }
    return out;
  }

  std::vector<double> node_r;
  std::vector<double> node_z;
  if (!extract_structured_nodes_2d(state, node_r, node_z)) {
    return out;
  }

  const int nz = state.mesh.topo.nz;
  constexpr int kSamples = 64;
  std::vector<double> mu(static_cast<std::size_t>(kSamples), 0.0);
  std::vector<double> radius(static_cast<std::size_t>(kSamples), 0.0);

  for (int s = 0; s < kSamples; ++s) {
    const double mu_s = static_cast<double>(s) / static_cast<double>(kSamples - 1);
    const double theta = std::acos(std::clamp(mu_s, 0.0, 1.0));
    const double angle_deg = theta * 180.0 / kPi;
    mu[static_cast<std::size_t>(s)] = mu_s;
    radius[static_cast<std::size_t>(s)] =
        find_isodensity_radius_2d(node_r, node_z, rho, nz, angle_deg, rho_threshold);
  }

  for (std::size_t k = 0; k < out.modes.size(); ++k) {
    const int ell = out.modes[k];
    if (ell < 0) {
      out.coefficients[k] = 0.0;
      continue;
    }

    double integral = 0.0;
    for (int s = 1; s < kSamples; ++s) {
      const std::size_t s0 = static_cast<std::size_t>(s - 1);
      const std::size_t s1 = static_cast<std::size_t>(s);
      const double dmu = mu[s1] - mu[s0];
      const double f0 = radius[s0] * legendre_p(ell, mu[s0]);
      const double f1 = radius[s1] * legendre_p(ell, mu[s1]);
      integral += 0.5 * (f0 + f1) * dmu;
    }
    out.coefficients[k] = (2.0 * static_cast<double>(ell) + 1.0) * integral;
  }

  return out;
}

LaserPatternDiagnostics compute_laser_pattern(const core::State& state,
                                              const core::Config& cfg,
                                              const laser::LaserMesh* laser_mesh) {
  LaserPatternDiagnostics out{};
  if (!cfg.laser.enabled) {
    // Laser fields are identically zero on laser-off runs; reproduce the
    // zero-input evaluation without the device pulls.
    if (cfg.diagnostics.laser_pattern.per_beam && !cfg.laser.beams.empty()) {
      out.absorbed_fraction_per_beam.assign(cfg.laser.beams.size(), 0.0);
    }
    return out;
  }
  out.absorbed_total = sum_laser_dep(state);
  if (state.dt > 0.0) {
    out.absorbed_power_total = out.absorbed_total / state.dt;
  }
  out.critical_surface_r = compute_critical_surface_r(laser_mesh);
  out.near_critical_r = compute_near_critical_r(laser_mesh);
  out.absorption_weighted_r = compute_absorption_weighted_r(state);
  if (laser_mesh != nullptr) {
    if (state.dt > 0.0) {
      out.commanded_power_total =
          std::max(0.0, laser_mesh->last_commanded_energy / state.dt);
    }
    out.trace_unabsorbed_power_total = std::max(0.0, laser_mesh->last_trace_unabsorbed_power);
    out.unabsorbed_power_total = std::max(0.0, laser_mesh->last_unabsorbed_power);
    out.transfer_blocked_power_total = std::max(0.0, laser_mesh->last_transfer_blocked_power);
    out.tail_closure_count = std::max<std::int64_t>(0, laser_mesh->last_tail_closure_count);
    out.tail_closure_absorbed_power_total =
        std::max(0.0, laser_mesh->last_tail_closure_absorbed_power);
    out.cbet_exchanged_power_total = std::max(0.0, laser_mesh->last_cbet_exchanged_power);
    out.cbet_ledger_residual_rel = laser_mesh->last_cbet_ledger_residual;
    out.cbet_iterations = static_cast<std::int64_t>(laser_mesh->last_cbet_iterations);
    out.cbet_clamp_count = std::max<std::int64_t>(0, laser_mesh->last_cbet_clamp_count);
    out.critical_surface_hit_count =
        std::max<std::int64_t>(0, laser_mesh->last_critical_surface_hit_count);
    if (out.commanded_power_total > 0.0) {
      out.trace_absorption_efficiency_total = std::clamp(
          1.0 - out.trace_unabsorbed_power_total / out.commanded_power_total, 0.0, 1.0);
    }
    if (out.commanded_power_total > 0.0) {
      out.absorption_efficiency_total = std::clamp(
          out.absorbed_power_total / out.commanded_power_total, 0.0, 1.0);
    }
    out.corona_transition_blend = laser_mesh->last_ghost_transition_blend;
    out.corona_transition_resolved_cells =
        static_cast<std::int64_t>(laser_mesh->last_ghost_transition_resolved_cells);
  }

  if (!cfg.diagnostics.laser_pattern.per_beam) {
    return out;
  }

  const std::size_t n_beams = cfg.laser.beams.size();
  if (n_beams == 0) {
    return out;
  }

  out.absorbed_fraction_per_beam.assign(n_beams, 0.0);
  if (!(out.absorbed_total > 0.0)) {
    return out;
  }

  std::vector<double> weights(n_beams, 0.0);
  for (std::size_t b = 0; b < n_beams; ++b) {
    if (b < state.laser_waveforms.size() && state.laser_waveforms[b].n_points > 0) {
      weights[b] = std::max(0.0, state.laser_waveforms[b].eval(state.t));
    } else {
      weights[b] = 1.0;
    }
  }

  const double w_sum = std::accumulate(weights.begin(), weights.end(), 0.0);
  if (!(w_sum > 0.0)) {
    const double uniform = 1.0 / static_cast<double>(n_beams);
    std::fill(out.absorbed_fraction_per_beam.begin(),
              out.absorbed_fraction_per_beam.end(),
              uniform);
    return out;
  }

  for (std::size_t b = 0; b < n_beams; ++b) {
    out.absorbed_fraction_per_beam[b] = weights[b] / w_sum;
  }
  return out;
}

void initialize_hotspot_gas_tracer(core::State& state,
                                   const core::Config& cfg) {
  const auto& hot = cfg.numerics.diagnostics.hotspot_gas;
  if (!hot.enabled) {
    return;
  }
  if (state.gas_tracer_initialized) {
    return;
  }
  if (state.step != 0) {
    core::log_warning(
        "[hotspot_gas] tracer initialization skipped on nonzero-step state; "
        "restart continuation is not supported for this diagnostic chunk");
    return;
  }
  const std::size_t n = state.mass.size();
  if (n == 0 || state.rho.size() != n || state.vol.size() != n) {
    core::log_warning("[hotspot_gas] tracer initialization skipped: cell state is empty");
    return;
  }
  if (state.gas_tracer_Y.empty()) {
    state.gas_tracer_Y.reset(n);
  }
  if (state.mesh.cell_centroid_r.size() != n ||
      (state.mesh.dim == 2 && state.mesh.cell_centroid_z.size() != n)) {
    state.mesh.node_r = state.x_r.data();
    state.mesh.node_z = (state.mesh.dim == 2) ? state.x_z.data() : nullptr;
    state.mesh.recompute_geometry();
  }
  if (state.mesh.cell_centroid_r.size() != n ||
      (state.mesh.dim == 2 && state.mesh.cell_centroid_z.size() != n)) {
    core::log_warning(
        "[hotspot_gas] tracer initialization skipped: cell centroids are unavailable");
    return;
  }

  const auto mass = to_host(state.mass);
  const auto vol = to_host(state.vol);
  const auto rho = to_host(state.rho);
  const auto ee = (state.ee.size() == n) ? to_host(state.ee) : std::vector<double>{};
  const auto ei = (state.ei.size() == n) ? to_host(state.ei) : std::vector<double>{};
  const CellVelocityHostView velocity_view =
      make_cell_velocity_host_view(state, n);
  std::vector<double> tracer(n, 0.0);
  long double M0 = 0.0L;
  long double V0 = 0.0L;
  long double U0 = 0.0L;
  long double K0 = 0.0L;
  std::size_t tagged_cells = 0;
  std::vector<std::pair<double, double>> radius_samples;
  std::vector<std::pair<double, double>> rho_samples;
  radius_samples.reserve(n);
  rho_samples.reserve(n);
  for (std::size_t c = 0; c < n; ++c) {
    const double r = state.mesh.cell_centroid_r[c];
    const double z =
        (state.mesh.dim == 2) ? state.mesh.cell_centroid_z[c] : 0.0;
    const double rr = std::hypot(r, z);
    if (rr < hot.R_g_cm) {
      tracer[c] = 1.0;
      const double m = std::max(mass[c], 0.0);
      M0 += static_cast<long double>(m);
      V0 += static_cast<long double>(std::max(vol[c], 0.0));
      if (c < rho.size() && rho[c] > 0.0 && std::isfinite(rho[c])) {
        radius_samples.emplace_back(rr, m);
        rho_samples.emplace_back(rho[c], m);
      }
      if (c < ee.size()) {
        U0 += static_cast<long double>(m) *
              static_cast<long double>(finite_or_zero_host(ee[c]));
      }
      if (c < ei.size()) {
        U0 += static_cast<long double>(m) *
              static_cast<long double>(finite_or_zero_host(ei[c]));
      }
      K0 += 0.5L * static_cast<long double>(m) *
            static_cast<long double>(cell_velocity_sq_host(velocity_view, c));
      ++tagged_cells;
    }
  }
  state.gas_tracer_Y.copy_from_host(tracer.data());
  state.gas_tracer_initialized = true;
  state.gas_tracer_mass_initial = static_cast<double>(M0);
  if (M0 > 0.0L && V0 > 0.0L) {
    state.hotspot_rho_bar_initial = static_cast<double>(M0 / V0);
  }
  const double R50_initial = weighted_quantile(radius_samples, 0.50);
  if (hot.R_g_cm > 0.0 && R50_initial > 0.0 && std::isfinite(R50_initial)) {
    state.hotspot_cr50_initial = hot.R_g_cm / R50_initial;
  }
  state.hotspot_rho50_initial = weighted_quantile(rho_samples, 0.50);
  state.hotspot_internal_energy_initial_erg = static_cast<double>(U0);
  state.hotspot_kinetic_energy_initial_erg = static_cast<double>(K0);

  std::ostringstream msg;
  msg << "[hotspot_gas] initialized R_g_cm=" << hot.R_g_cm
      << " tagged_cells=" << tagged_cells
      << " M_g0=" << state.gas_tracer_mass_initial << " g"
      << " rho_bar0=" << state.hotspot_rho_bar_initial << " g/cm3"
      << " E_hotspot0="
      << (state.hotspot_internal_energy_initial_erg +
          state.hotspot_kinetic_energy_initial_erg)
      << " erg";
  core::log_info(msg.str());
}

IcfShellDiagnostics compute_icf_shell_diagnostics(
    const core::State& state,
    const core::Config& cfg,
    const double R_initial_cm) {
  IcfShellDiagnostics out{};
  if (state.rho.empty()) {
    return out;
  }

  const auto rho = to_host(state.rho);
  const auto mass = to_host(state.mass);
  const bool has_mass = (mass.size() == rho.size());
  const double rho_peak = *std::max_element(rho.begin(), rho.end());
  if (!(rho_peak > 0.0) || !std::isfinite(rho_peak)) {
    return out;
  }

  const double shell_mean_threshold = 0.1 * rho_peak;
  const double rho_outer_threshold =
      cfg.numerics.diagnostics.icf.rho_outer_threshold_g_per_cc > 0.0
          ? cfg.numerics.diagnostics.icf.rho_outer_threshold_g_per_cc
          : 0.1 * rho_peak;
  const double rho_inner_threshold =
      cfg.numerics.diagnostics.icf.rho_inner_threshold_g_per_cc > 0.0
          ? cfg.numerics.diagnostics.icf.rho_inner_threshold_g_per_cc
          : 0.5 * rho_peak;

  double R_outer = 0.0;
  double R_inner = 0.0;
  long double weighted_radius_sum = 0.0L;
  long double weight_sum = 0.0L;

  if (state.mesh.dim == 1) {
    const auto nodes = to_host(state.x_r);
    if (nodes.size() != rho.size() + 1) {
      return out;
    }
    for (std::size_t c = 0; c < rho.size(); ++c) {
      if (rho[c] < shell_mean_threshold) {
        continue;
      }
      const double r = 0.5 * (nodes[c] + nodes[c + 1]);
      const double w = has_mass ? std::max(mass[c], 0.0) : 1.0;
      weighted_radius_sum += static_cast<long double>(w) *
                             static_cast<long double>(r);
      weight_sum += static_cast<long double>(w);
    }
    R_outer = find_outer_isodensity_radius_1d(nodes, rho, rho_outer_threshold);
    R_inner = find_inner_isodensity_radius_1d(nodes, rho, rho_inner_threshold);
  } else if (state.mesh.dim == 2) {
    std::vector<double> node_r;
    std::vector<double> node_z;
    if (!extract_structured_nodes_2d(state, node_r, node_z)) {
      return out;
    }
    const int nr = state.mesh.topo.nr;
    const int nz = state.mesh.topo.nz;
    if (nr <= 0 || nz <= 0 ||
        static_cast<int>(rho.size()) != nr * nz) {
      return out;
    }

    for (int i = 0; i < nr; ++i) {
      const double r_center = 0.5 * (node_r[static_cast<std::size_t>(i)] +
                                     node_r[static_cast<std::size_t>(i + 1)]);
      for (int j = 0; j < nz; ++j) {
        const std::size_t c = static_cast<std::size_t>(i * nz + j);
        if (rho[c] < shell_mean_threshold) {
          continue;
        }
        const double w = has_mass ? std::max(mass[c], 0.0) : 1.0;
        weighted_radius_sum += static_cast<long double>(w) *
                               static_cast<long double>(r_center);
        weight_sum += static_cast<long double>(w);
      }
    }
    R_outer =
        find_isodensity_radius_2d(node_r, node_z, rho, nz, 0.0, rho_outer_threshold);
    R_inner = find_inner_isodensity_radius_2d(
        node_r, node_z, rho, nz, 0.0, rho_inner_threshold);
  } else {
    return out;
  }

  if (!(weight_sum > 0.0L)) {
    return out;
  }

  const double R_shell = static_cast<double>(weighted_radius_sum / weight_sum);
  const double thickness = R_outer - R_inner;
  if (!(R_shell > 0.0) || !(thickness > 0.0) ||
      !std::isfinite(R_shell) || !std::isfinite(thickness)) {
    return out;
  }

  const double R_initial =
      (R_initial_cm > 0.0 && std::isfinite(R_initial_cm)) ? R_initial_cm : R_shell;
  out.valid = true;
  out.R_shell_cm = R_shell;
  out.shell_thickness_cm = thickness;
  out.IFAR = R_shell / thickness;
  out.CR = R_initial / R_shell;
  out.R_initial_cm = R_initial;
  return out;
}

HotspotGasDiagnostics compute_hotspot_gas_diagnostics(
    const core::State& state,
    const core::Config& cfg) {
  HotspotGasDiagnostics out{};
  const auto& hot = cfg.numerics.diagnostics.hotspot_gas;
  if (!hot.enabled || state.gas_tracer_Y.empty()) {
    return out;
  }
  const bool two_temperature = cfg.main.two_temperature;
  const std::size_t n = state.gas_tracer_Y.size();
  if (state.mass.size() != n || state.rho.size() != n || state.vol.size() != n ||
      state.Pe.size() != n || state.Pi.size() != n ||
      state.Te.size() != n || state.ee.size() != n || state.ei.size() != n ||
      (two_temperature && state.Ti.size() != n) ||
      state.mesh.cell_centroid_r.size() != n ||
      (state.mesh.dim == 2 && state.mesh.cell_centroid_z.size() != n)) {
    return out;
  }

  const auto Y = to_host(state.gas_tracer_Y);
  const auto mass = to_host(state.mass);
  const auto rho = to_host(state.rho);
  const auto vol = to_host(state.vol);
  const auto Pe = to_host(state.Pe);
  const auto Pi = to_host(state.Pi);
  const auto Te = to_host(state.Te);
  const auto Ti = two_temperature ? to_host(state.Ti) : std::vector<double>{};
  const auto ee = to_host(state.ee);
  const auto ei = to_host(state.ei);
  const CellVelocityHostView velocity_view =
      make_cell_velocity_host_view(state, n);
  const double gamma = first_nonvoid_ideal_gamma(cfg);
  const auto& pc = state.central_pseudo_core;
  const bool macro_core_active =
      pc.configured && pc.built && pc.valid &&
      pc.inactive_member_mask.size() == n &&
      pc.M_c > 0.0 && pc.V_c > 0.0 &&
      std::isfinite(pc.M_c) && std::isfinite(pc.V_c);

  long double M = 0.0L;
  long double YV = 0.0L;
  long double rr2_mass_sum = 0.0L;
  long double rho_mass_sum = 0.0L;
  long double p_mass_sum = 0.0L;
  long double K_mass_sum = 0.0L;
  long double Te_mass_sum = 0.0L;
  long double Te_weight_sum = 0.0L;
  long double Ti_mass_sum = 0.0L;
  long double Ti_weight_sum = 0.0L;
  long double Ue_sum = 0.0L;
  long double Ui_sum = 0.0L;
  long double kinetic_sum = 0.0L;
  std::vector<std::pair<double, double>> radius_samples;
  std::vector<std::pair<double, double>> rho_samples;
  std::vector<std::pair<double, double>> p_samples;
  std::vector<std::pair<double, double>> K_samples;
  std::vector<std::pair<double, double>> Te_samples;
  std::vector<std::pair<double, double>> Ti_samples;
  radius_samples.reserve(n);
  rho_samples.reserve(n);
  p_samples.reserve(n);
  K_samples.reserve(n);
  Te_samples.reserve(n);
  Ti_samples.reserve(n);

  for (std::size_t c = 0; c < n; ++c) {
    if (macro_core_active && pc.inactive_member_mask[c] != 0U) {
      continue;
    }
    const double y = clamp01_host(Y[c]);
    const double m = std::max(mass[c], 0.0);
    const double q = m * y;
    if (!(q > 0.0) || !std::isfinite(q)) {
      continue;
    }
    const double r = state.mesh.cell_centroid_r[c];
    const double z =
        (state.mesh.dim == 2) ? state.mesh.cell_centroid_z[c] : 0.0;
    const double rr = std::hypot(r, z);
    const double rho_c = rho[c];
    const double p_c = Pe[c] + Pi[c];
    const double Te_c = Te[c];
    const double Ti_c = two_temperature ? Ti[c] : 0.0;
    if (!std::isfinite(rr) || !(rho_c > 0.0) || !std::isfinite(rho_c) ||
        !(p_c >= 0.0) || !std::isfinite(p_c)) {
      continue;
    }
    const double K_c = (p_c > 0.0) ? p_c / std::pow(rho_c, gamma) : 0.0;
    M += static_cast<long double>(q);
    YV += static_cast<long double>(y) *
          static_cast<long double>(std::max(vol[c], 0.0));
    rr2_mass_sum += static_cast<long double>(q) *
                    static_cast<long double>(rr * rr);
    rho_mass_sum += static_cast<long double>(q) *
                    static_cast<long double>(rho_c);
    p_mass_sum += static_cast<long double>(q) *
                  static_cast<long double>(p_c);
    K_mass_sum += static_cast<long double>(q) *
                  static_cast<long double>(K_c);
    if (Te_c >= 0.0 && std::isfinite(Te_c)) {
      Te_mass_sum += static_cast<long double>(q) *
                     static_cast<long double>(Te_c);
      Te_weight_sum += static_cast<long double>(q);
      Te_samples.emplace_back(Te_c, q);
    }
    if (two_temperature && Ti_c >= 0.0 && std::isfinite(Ti_c)) {
      Ti_mass_sum += static_cast<long double>(q) *
                     static_cast<long double>(Ti_c);
      Ti_weight_sum += static_cast<long double>(q);
      Ti_samples.emplace_back(Ti_c, q);
    }
    Ue_sum += static_cast<long double>(q) *
              static_cast<long double>(finite_or_zero_host(ee[c]));
    Ui_sum += static_cast<long double>(q) *
              static_cast<long double>(finite_or_zero_host(ei[c]));
    kinetic_sum += 0.5L * static_cast<long double>(q) *
                   static_cast<long double>(cell_velocity_sq_host(velocity_view, c));
    radius_samples.emplace_back(rr, q);
    rho_samples.emplace_back(rho_c, q);
    p_samples.emplace_back(p_c, q);
    K_samples.emplace_back(K_c, q);
  }

  if (macro_core_active) {
    const double M_Y_c =
        std::isfinite(pc.M_Y_c) ? std::max(pc.M_Y_c, 0.0) : 0.0;
    const double y_c = clamp01_host(M_Y_c / pc.M_c);
    const double q_c = y_c * pc.M_c;
    // Pressure-equilibrium partial volume of the absorbed gas: under the
    // pooled same-gamma ideal-gas closure V_gas/V_c equals the gas share of
    // the internal energy, NOT the mass fraction (the mass fraction assigns
    // the cold dense shell's specific volume to the hot gas and makes CR_V
    // jump discontinuously at mixed absorb events). Fall back to the mass
    // fraction when the energy fraction has not been initialized.
    const double f_gas_energy =
        std::isfinite(pc.U_gas_frac_c) ? clamp01_host(pc.U_gas_frac_c) : y_c;
    double V_c_gas = f_gas_energy * pc.V_c;
    // Stratified 1D sub-model override: the hydro layer publishes the
    // sub-model's shell-resolved gas partial volume into the state (no
    // pooled-equilibrium estimate, no diagnostics->hydro dependency).
    if (std::isfinite(pc.core1d_V_gas_c) && pc.core1d_V_gas_c > 0.0) {
      V_c_gas = pc.core1d_V_gas_c;
    }
    if (q_c > 0.0 && V_c_gas > 0.0 &&
        std::isfinite(q_c) && std::isfinite(V_c_gas)) {
      std::vector<double> node_r;
      std::vector<double> node_z;
      state.x_r.copy_to_host(node_r);
      state.x_z.copy_to_host(node_z);
      const double rho_c = pc.M_c / pc.V_c;
      const double V_c0 =
          (pc.V_c_initial > 0.0 && std::isfinite(pc.V_c_initial))
              ? pc.V_c_initial
              : 0.0;
      const double rho_c0 =
          (pc.rho_c_initial > 0.0 && std::isfinite(pc.rho_c_initial))
              ? pc.rho_c_initial
              : ((V_c0 > 0.0) ? (pc.M_c / V_c0) : 0.0);
      const double e_c =
          std::max((pc.Ue_c + pc.Ui_c) / std::max(pc.M_c, 1.0e-300), 0.0);
      const double p_c_raw =
          (gamma > 1.0) ? (gamma - 1.0) * rho_c * e_c : 0.0;
      const double p_c =
          (p_c_raw >= 0.0 && std::isfinite(p_c_raw)) ? p_c_raw : 0.0;
      const double K_c_raw =
          (p_c > 0.0 && rho_c > 0.0) ? p_c / std::pow(rho_c, gamma) : 0.0;
      const double K_c =
          (K_c_raw >= 0.0 && std::isfinite(K_c_raw)) ? K_c_raw : 0.0;
      const double rr_c_raw =
          central_macro_boundary_mean_radius(pc, node_r, node_z);
      const double rr_c = std::isfinite(rr_c_raw) ? rr_c_raw : 0.0;

      M += static_cast<long double>(q_c);
      YV += static_cast<long double>(V_c_gas);
      rr2_mass_sum += static_cast<long double>(q_c) *
                      static_cast<long double>(rr_c * rr_c);
      rho_mass_sum += static_cast<long double>(q_c) *
                      static_cast<long double>(rho_c);
      p_mass_sum += static_cast<long double>(q_c) *
                    static_cast<long double>(p_c);
      K_mass_sum += static_cast<long double>(q_c) *
                    static_cast<long double>(K_c);
      const int rep = pc.representative_cell;
      if (rep >= 0 && static_cast<std::size_t>(rep) < Te.size()) {
        const double Te_c = Te[static_cast<std::size_t>(rep)];
        if (Te_c >= 0.0 && std::isfinite(Te_c)) {
          Te_mass_sum += static_cast<long double>(q_c) *
                         static_cast<long double>(Te_c);
          Te_weight_sum += static_cast<long double>(q_c);
          Te_samples.emplace_back(Te_c, q_c);
        }
      }
      if (two_temperature && rep >= 0 &&
          static_cast<std::size_t>(rep) < Ti.size()) {
        const double Ti_c = Ti[static_cast<std::size_t>(rep)];
        if (Ti_c >= 0.0 && std::isfinite(Ti_c)) {
          Ti_mass_sum += static_cast<long double>(q_c) *
                         static_cast<long double>(Ti_c);
          Ti_weight_sum += static_cast<long double>(q_c);
          Ti_samples.emplace_back(Ti_c, q_c);
        }
      }
      Ue_sum += static_cast<long double>(y_c) *
                static_cast<long double>(finite_or_zero_host(pc.Ue_c));
      Ui_sum += static_cast<long double>(y_c) *
                static_cast<long double>(finite_or_zero_host(pc.Ui_c));
      radius_samples.emplace_back(rr_c, q_c);
      rho_samples.emplace_back(rho_c, q_c);
      p_samples.emplace_back(p_c, q_c);
      K_samples.emplace_back(K_c, q_c);

      out.macro_core_valid = true;
      out.macro_core_mass_g = pc.M_c;
      out.macro_core_tracer_mass_g = q_c;
      out.macro_core_gas_energy_frac = f_gas_energy;
      out.macro_core_volume_cm3 = pc.V_c;
      out.macro_core_volume_initial_cm3 = V_c0;
      out.macro_core_rho_gcc = rho_c;
      out.macro_core_rho_initial_gcc = rho_c0;
      if (V_c0 > 0.0 && pc.V_c > 0.0) {
        out.CR_V_macro = std::cbrt(V_c0 / pc.V_c);
      }
    }
  }

  if (!(M > 0.0L)) {
    return out;
  }

  out.valid = true;
  out.R_g_cm = hot.R_g_cm;
  out.gas_mass_initial_g = state.gas_tracer_mass_initial;
  out.gas_mass_g = static_cast<double>(M);
  if (state.gas_tracer_mass_initial > 0.0 &&
      std::isfinite(state.gas_tracer_mass_initial)) {
    out.gas_mass_rel_drift =
        (out.gas_mass_g - state.gas_tracer_mass_initial) /
        state.gas_tracer_mass_initial;
  }

  out.R50_cm = weighted_quantile(radius_samples, 0.50);
  out.R90_cm = weighted_quantile(radius_samples, 0.90);
  out.R95_cm = weighted_quantile(radius_samples, 0.95);
  out.R99_cm = weighted_quantile(radius_samples, 0.99);
  if (hot.R_g_cm > 0.0) {
    out.CR50 = (out.R50_cm > 0.0) ? hot.R_g_cm / out.R50_cm : 0.0;
    out.CR90 = (out.R90_cm > 0.0) ? hot.R_g_cm / out.R90_cm : 0.0;
    out.CR95 = (out.R95_cm > 0.0) ? hot.R_g_cm / out.R95_cm : 0.0;
    out.CR99 = (out.R99_cm > 0.0) ? hot.R_g_cm / out.R99_cm : 0.0;
  }
  out.Rrms_cm =
      std::sqrt(std::max(0.0L, (5.0L / 3.0L) * rr2_mass_sum / M));
  out.CRrms = (hot.R_g_cm > 0.0 && out.Rrms_cm > 0.0)
                  ? hot.R_g_cm / out.Rrms_cm
                  : 0.0;
  out.rho_bar_volume_gcc = (YV > 0.0L) ? static_cast<double>(M / YV) : 0.0;
  out.rho_mean_mass_gcc = static_cast<double>(rho_mass_sum / M);
  out.rho50_gcc = weighted_quantile(rho_samples, 0.50);
  const double rho_bar_initial =
      (state.hotspot_rho_bar_initial > 0.0 &&
       std::isfinite(state.hotspot_rho_bar_initial))
          ? state.hotspot_rho_bar_initial
          : out.rho_bar_volume_gcc;
  const double CR50_initial =
      (state.hotspot_cr50_initial > 0.0 &&
       std::isfinite(state.hotspot_cr50_initial))
          ? state.hotspot_cr50_initial
          : out.CR50;
  const double rho50_initial =
      (state.hotspot_rho50_initial > 0.0 &&
       std::isfinite(state.hotspot_rho50_initial))
          ? state.hotspot_rho50_initial
          : out.rho50_gcc;
  out.rho_bar_initial_gcc = rho_bar_initial;
  if (rho_bar_initial > 0.0 && std::isfinite(rho_bar_initial) &&
      out.rho_bar_volume_gcc >= 0.0 && std::isfinite(out.rho_bar_volume_gcc)) {
    out.CR_V = std::cbrt(out.rho_bar_volume_gcc / rho_bar_initial);
  }
  if (CR50_initial > 0.0 && std::isfinite(CR50_initial) &&
      out.CR50 >= 0.0 && std::isfinite(out.CR50)) {
    out.C_R50_norm = out.CR50 / CR50_initial;
  }
  if (rho50_initial > 0.0 && std::isfinite(rho50_initial) &&
      out.rho50_gcc >= 0.0 && std::isfinite(out.rho50_gcc)) {
    out.CR_rho50 = std::cbrt(out.rho50_gcc / rho50_initial);
  }
  out.rho90_gcc = weighted_quantile(rho_samples, 0.90);
  out.rho95_gcc = weighted_quantile(rho_samples, 0.95);
  out.rho99_gcc = weighted_quantile(rho_samples, 0.99);
  out.p_mean_dyn_cm2 = static_cast<double>(p_mass_sum / M);
  out.p50_dyn_cm2 = weighted_quantile(p_samples, 0.50);
  out.p90_dyn_cm2 = weighted_quantile(p_samples, 0.90);
  out.p95_dyn_cm2 = weighted_quantile(p_samples, 0.95);
  out.p99_dyn_cm2 = weighted_quantile(p_samples, 0.99);
  out.Te_mean_eV =
      (Te_weight_sum > 0.0L) ? static_cast<double>(Te_mass_sum / Te_weight_sum) : 0.0;
  out.Te_p10_eV = weighted_quantile(Te_samples, 0.10);
  out.Te_p50_eV = weighted_quantile(Te_samples, 0.50);
  out.Te_p90_eV = weighted_quantile(Te_samples, 0.90);
  out.Ti_valid = two_temperature;
  if (two_temperature) {
    out.Ti_mean_eV =
        (Ti_weight_sum > 0.0L) ? static_cast<double>(Ti_mass_sum / Ti_weight_sum) : 0.0;
    out.Ti_p10_eV = weighted_quantile(Ti_samples, 0.10);
    out.Ti_p50_eV = weighted_quantile(Ti_samples, 0.50);
    out.Ti_p90_eV = weighted_quantile(Ti_samples, 0.90);
  }
  out.hotspot_internal_energy_erg = static_cast<double>(Ue_sum + Ui_sum);
  out.hotspot_kinetic_energy_erg = static_cast<double>(kinetic_sum);
  out.hotspot_total_energy_erg =
      out.hotspot_internal_energy_erg + out.hotspot_kinetic_energy_erg;
  out.hotspot_internal_energy_initial_erg =
      std::isfinite(state.hotspot_internal_energy_initial_erg)
          ? state.hotspot_internal_energy_initial_erg
          : out.hotspot_internal_energy_erg;
  out.hotspot_kinetic_energy_initial_erg =
      std::isfinite(state.hotspot_kinetic_energy_initial_erg)
          ? state.hotspot_kinetic_energy_initial_erg
          : out.hotspot_kinetic_energy_erg;
  out.hotspot_total_energy_initial_erg =
      out.hotspot_internal_energy_initial_erg +
      out.hotspot_kinetic_energy_initial_erg;
  out.hotspot_work_proxy_internal_erg =
      out.hotspot_internal_energy_erg -
      out.hotspot_internal_energy_initial_erg;
  out.hotspot_work_proxy_kinetic_erg =
      out.hotspot_kinetic_energy_erg -
      out.hotspot_kinetic_energy_initial_erg;
  out.hotspot_work_proxy_total_erg =
      out.hotspot_total_energy_erg -
      out.hotspot_total_energy_initial_erg;
  out.K_mean = static_cast<double>(K_mass_sum / M);
  out.K50 = weighted_quantile(K_samples, 0.50);
  out.K90 = weighted_quantile(K_samples, 0.90);
  out.K95 = weighted_quantile(K_samples, 0.95);
  out.K99 = weighted_quantile(K_samples, 0.99);
  return out;
}

PhaseResolvedEnergyDiagnostics compute_phase_resolved_energy_diagnostic(
    const EnergyTotals& pre_hydro,
    const EnergyTotals& post_hydro,
    const EnergyTotals& post_ale) {
  PhaseResolvedEnergyDiagnostics out{};
  out.valid = true;
  out.internal_pre_hydro = pre_hydro.E_int_e + pre_hydro.E_int_i;
  out.internal_post_hydro = post_hydro.E_int_e + post_hydro.E_int_i;
  out.internal_post_ale = post_ale.E_int_e + post_ale.E_int_i;
  out.kinetic_pre_hydro = pre_hydro.E_kin;
  out.kinetic_post_hydro = post_hydro.E_kin;
  out.kinetic_post_ale = post_ale.E_kin;
  out.E_pre_hydro = out.internal_pre_hydro + out.kinetic_pre_hydro;
  out.E_post_hydro = out.internal_post_hydro + out.kinetic_post_hydro;
  out.E_post_ale = out.internal_post_ale + out.kinetic_post_ale;
  out.dE_hydro = out.E_post_hydro - out.E_pre_hydro;
  out.dE_ale = out.E_post_ale - out.E_post_hydro;
  out.dE_hydro_internal = out.internal_post_hydro - out.internal_pre_hydro;
  out.dE_ale_internal = out.internal_post_ale - out.internal_post_hydro;
  out.dE_hydro_kinetic = out.kinetic_post_hydro - out.kinetic_pre_hydro;
  out.dE_ale_kinetic = out.kinetic_post_ale - out.kinetic_post_hydro;
  return out;
}

void finalize_ale_closure_audit(AleClosureAuditDiagnostics& audit) {
  if (!audit.valid) {
    return;
  }

  audit.F_floor = audit.sum_dI_after_floor - audit.sum_dI_raw;
  audit.residual_K0_cellcorner_node =
      audit.K0_cellcorner - audit.K0_node_from_corner;
  audit.residual_K0_node_budget = audit.K0_node_from_corner - audit.K0_budget;
  audit.residual_K_remap = audit.K_remap_total - audit.K0_scalar_total;
  audit.residual_I_remap = audit.I_raw - audit.I0;
  audit.residual_closure_target =
      audit.sum_dI_raw - (audit.K_remap_total - audit.K_node_postBC);
  audit.dE_ale_predicted =
      audit.residual_I_remap + audit.residual_K_remap + audit.F_floor +
      (audit.K_post_budget - audit.K_node_postBC);
  audit.residual_dE_ale = audit.dE_ale_predicted - audit.dE_ale_total;

  audit.mechanism_code = 0;
  audit.k_remap_defect_dominant = false;
  audit.floor_dominant = false;
  audit.diagnostic_mismatch_dominant = false;
  const double dE_scale = std::abs(audit.dE_ale_total);
  if (!(dE_scale > 0.0) || !std::isfinite(dE_scale)) {
    return;
  }
  const double dominant = 0.1 * dE_scale;
  if (std::abs(audit.residual_K_remap) > dominant) {
    audit.k_remap_defect_dominant = true;
    audit.mechanism_code = 1;
  } else if (audit.F_floor > dominant) {
    audit.floor_dominant = true;
    audit.mechanism_code = 2;
  } else if (std::abs(audit.K_post_budget - audit.K_node_postBC) > dominant) {
    audit.diagnostic_mismatch_dominant = true;
    audit.mechanism_code = 3;
  }
}

namespace {

const char* ale_closure_audit_mechanism_label(const int mechanism_code) {
  switch (mechanism_code) {
    case 1:
      return "K_remap_defect";
    case 2:
      return "floor_dominant";
    case 3:
      return "mismatch";
    default:
      return "unknown";
  }
}

}  // namespace

void log_energy_budget_step(const EnergyBudget& budget,
                            const core::Config& cfg,
                            const int step,
                            const double t) {
  if (!cfg.diagnostics.enabled || !cfg.diagnostics.energy_budget.enabled) {
    return;
  }
  if (cfg.diagnostics.every > 1 && (step % cfg.diagnostics.every) != 0) {
    return;
  }

  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[diag:energy] step=" << step
      << " t=" << t
      << " eps_budget=" << budget.epsilon_budget
      << " dE_total=" << budget.dE_total
      << " E_redistribution_unresolved="
      << budget.E_redistribution_unresolved;
  core::log_debug(oss.str());

  static int eps_warn_count = 0;
  // First-step residual is dominated by discretization of the initial state.
  if (step > 1 &&
      budget.epsilon_budget > cfg.diagnostics.energy_budget.warn_threshold) {
    ++eps_warn_count;
    if (eps_warn_count == 1 || eps_warn_count % 100 == 0) {
      core::log_warning("Energy budget epsilon exceeded threshold: eps=" +
                        std::to_string(budget.epsilon_budget) +
                        ", threshold=" +
                        std::to_string(cfg.diagnostics.energy_budget.warn_threshold) +
                        " (warning #" + std::to_string(eps_warn_count) + ")");
    }
  }

  const double denom = std::max(budget.E_denom, 1.0e-20);
  const double floor_ratio = std::abs(budget.E_floor) / denom;
  const double safety_ratio = std::abs(budget.E_safety) / denom;
  const double redistribution_ratio =
      std::abs(budget.E_redistribution_unresolved) / denom;
  static int floor_warn_count = 0;
  if (floor_ratio > 1.0e-3) {
    ++floor_warn_count;
    if (floor_warn_count == 1 || floor_warn_count % 100 == 0) {
      core::log_warning("Energy floor contribution is large: E_floor/E_denom=" +
                        std::to_string(floor_ratio) +
                        " (warning #" + std::to_string(floor_warn_count) + ")");
    }
  }
  static int safety_warn_count = 0;
  if (safety_ratio > 1.0e-3) {
    ++safety_warn_count;
    if (safety_warn_count == 1 || safety_warn_count % 100 == 0) {
      core::log_warning("Energy safety contribution is large: E_safety/E_denom=" +
                        std::to_string(safety_ratio) +
                        " (warning #" + std::to_string(safety_warn_count) + ")");
    }
  }
  static int redistribution_warn_count = 0;
  if (redistribution_ratio > 1.0e-3) {
    ++redistribution_warn_count;
    if (redistribution_warn_count == 1 || redistribution_warn_count % 100 == 0) {
      core::log_warning(
          "Energy redistribution unresolved contribution is large: "
          "E_redistribution_unresolved/E_denom=" +
          std::to_string(redistribution_ratio) +
          " (warning #" + std::to_string(redistribution_warn_count) + ")");
    }
  }
}

void log_ale_closure_audit_step(const AleClosureAuditDiagnostics& audit,
                                const EnergyBudget& budget,
                                const core::Config& cfg,
                                const int step,
                                const double t) {
  if (!cfg.numerics.ale.ke_conservation_closure_audit || !audit.valid) {
    return;
  }
  if (!(budget.epsilon_budget > 1.0e-3)) {
    return;
  }

  std::ostringstream oss;
  oss << std::scientific << std::setprecision(6);
  oss << "[ale-closure-audit] step=" << step
      << " t=" << t
      << " K0_scalar=" << audit.K0_scalar_total
      << " K_remap=" << audit.K_remap_total
      << " delta_K=" << audit.residual_K_remap
      << " I_raw-I0=" << audit.residual_I_remap
      << " F_floor=" << audit.F_floor
      << " K_post_budget-K_post_closure="
      << (audit.K_post_budget - audit.K_node_postBC)
      << " mechanism="
      << ale_closure_audit_mechanism_label(audit.mechanism_code);
  core::log_warning(oss.str());

  if (audit.k_remap_defect_dominant) {
    core::log_warning("[ale-closure-audit] K-scalar-remap defect dominant");
  } else if (audit.floor_dominant) {
    core::log_warning("[ale-closure-audit] floor/positivity dominant");
  } else if (audit.diagnostic_mismatch_dominant) {
    core::log_warning("[ale-closure-audit] diagnostic mismatch dominant");
  }
}

}  // namespace tenryu::diagnostics
