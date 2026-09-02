#include "hydro/carrier_constraint_adapter.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/state.hpp"
#include "mesh/voronoi_rz.cuh"

namespace tenryu::hydro {
namespace {

bool carrier_hook_timing_enabled() {
  static const bool enabled =
      std::getenv("TENRYU_CARRIER_HOOK_TIMING") != nullptr;
  return enabled;
}

void copy_device_to_host(double* host, const double* device,
                         const std::size_t count, const char* label) {
  const cudaError_t err = cudaMemcpy(host, device, count * sizeof(double),
                                     cudaMemcpyDeviceToHost);
  TENRYU_ASSERT(err == cudaSuccess,
                std::string(label) + ": " + cudaGetErrorString(err));
}

void copy_host_to_device(double* device, const double* host,
                         const std::size_t count, const char* label) {
  const cudaError_t err = cudaMemcpy(device, host, count * sizeof(double),
                                     cudaMemcpyHostToDevice);
  TENRYU_ASSERT(err == cudaSuccess,
                std::string(label) + ": " + cudaGetErrorString(err));
}

}  // namespace

CarrierHookContext::CarrierHookContext(const core::State& state)
    : node_count_(state.x_r.size()),
      carrier_(state.boundary_carrier),
      timing_enabled_(carrier_hook_timing_enabled()) {
  TENRYU_ASSERT(state.carrier_node_class.size() == node_count_,
                "carrier node-class size mismatch");
  TENRYU_ASSERT(state.carrier_node_edge.size() == node_count_,
                "carrier node-edge size mismatch");
  TENRYU_ASSERT(state.carrier_node_lambda.size() == node_count_,
                "carrier node-lambda size mismatch");

  boundary_nodes_.reserve(carrier_.masters.size() + node_count_);
  for (const mesh::CarrierVertex& master : carrier_.masters) {
    boundary_nodes_.push_back(master.mesh_node);
  }
  for (std::size_t node = 0; node < node_count_; ++node) {
    if (state.carrier_node_class[node] !=
        static_cast<std::uint8_t>(
            mesh::voronoi::TessNodeClass::kCarrierSlave)) {
      continue;
    }
    slave_node_.push_back(static_cast<int>(node));
    slave_edge_.push_back(state.carrier_node_edge[node]);
    slave_lambda_.push_back(state.carrier_node_lambda[node]);
    boundary_nodes_.push_back(static_cast<int>(node));
  }

  const CarrierSlaveView slaves{slave_node_.data(), slave_edge_.data(),
                                slave_lambda_.data(),
                                static_cast<int>(slave_node_.size())};
  validate_carrier_slave_view(carrier_, slaves, node_count_);
  host_scratch_ = static_cast<double*>(core::host_pinned_scratch_acquire(
      "h2d:carrier_hook_host", 5 * node_count_ * sizeof(double)));
}

CarrierHookContext::~CarrierHookContext() {
  if (timing_enabled_ &&
      (reconstruct_calls_ != 0 || condensation_calls_ != 0)) {
    std::fprintf(stderr,
                 "[carrier_hooks] recon_ms=%.2f recon_calls=%d cond_ms=%.2f "
                 "cond_calls=%d\n",
                 reconstruct_ms_, reconstruct_calls_, condensation_ms_,
                 condensation_calls_);
  }
}

void CarrierHookContext::apply_reconstruct(double* d_pos_r, double* d_pos_z,
                                           double* d_vel_r, double* d_vel_z) {
  const auto start = timing_enabled_ ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  double* const pos_r = host_scratch_;
  double* const pos_z = pos_r + node_count_;
  double* const vel_r = pos_z + node_count_;
  double* const vel_z = vel_r + node_count_;
  double* const device[4] = {d_pos_r, d_pos_z, d_vel_r, d_vel_z};
  double* const host[4] = {pos_r, pos_z, vel_r, vel_z};
  const char* const d2h_label[4] = {
      "carrier reconstruct pos_r D2H failed",
      "carrier reconstruct pos_z D2H failed",
      "carrier reconstruct vel_r D2H failed",
      "carrier reconstruct vel_z D2H failed"};
  const char* const h2d_label[4] = {
      "carrier reconstruct pos_r H2D failed",
      "carrier reconstruct pos_z H2D failed",
      "carrier reconstruct vel_r H2D failed",
      "carrier reconstruct vel_z H2D failed"};

  for (int component = 0; component < 4; ++component) {
    if (device[component] != nullptr) {
      copy_device_to_host(host[component], device[component], node_count_,
                          d2h_label[component]);
    } else {
      std::fill(host[component], host[component] + node_count_, 0.0);
    }
  }

  const CarrierSlaveView slaves{slave_node_.data(), slave_edge_.data(),
                                slave_lambda_.data(),
                                static_cast<int>(slave_node_.size())};
  reconstruct_slaves(carrier_, slaves,
                     std::span<double>(pos_r, node_count_),
                     std::span<double>(pos_z, node_count_),
                     std::span<double>(vel_r, node_count_),
                     std::span<double>(vel_z, node_count_));

  for (int component = 0; component < 4; ++component) {
    if (device[component] != nullptr) {
      copy_host_to_device(device[component], host[component], node_count_,
                          h2d_label[component]);
    }
  }
  if (timing_enabled_) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    reconstruct_ms_ +=
        std::chrono::duration<double, std::milli>(elapsed).count();
    ++reconstruct_calls_;
  }
}

void CarrierHookContext::apply_condensation(double* d_accel_r,
                                            double* d_accel_z,
                                            const double* d_node_mass) {
  const auto start = timing_enabled_ ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
  TENRYU_ASSERT(d_accel_r != nullptr && d_accel_z != nullptr &&
                    d_node_mass != nullptr,
                "carrier condensation requires acceleration and mass arrays");
  double* const accel_r = host_scratch_;
  double* const accel_z = accel_r + node_count_;
  double* const node_mass = accel_z + node_count_;
  double* const force_r = node_mass + node_count_;
  double* const force_z = force_r + node_count_;

  copy_device_to_host(accel_r, d_accel_r, node_count_,
                      "carrier condensation accel_r D2H failed");
  copy_device_to_host(accel_z, d_accel_z, node_count_,
                      "carrier condensation accel_z D2H failed");
  copy_device_to_host(node_mass, d_node_mass, node_count_,
                      "carrier condensation mass D2H failed");
  for (const int node : boundary_nodes_) {
    force_r[node] = node_mass[node] * accel_r[node];
    force_z[node] = node_mass[node] * accel_z[node];
  }

  const CarrierSlaveView slaves{slave_node_.data(), slave_edge_.data(),
                                slave_lambda_.data(),
                                static_cast<int>(slave_node_.size())};
  const CondensedBoundarySystem system =
      condense_boundary_forces_and_masses(
          carrier_, slaves, std::span<const double>(node_mass, node_count_),
          std::span<const double>(force_r, node_count_),
          std::span<const double>(force_z, node_count_));
  std::vector<double> master_accel_r(carrier_.masters.size());
  std::vector<double> master_accel_z(carrier_.masters.size());
  solve_condensed_masters(system, master_accel_r, master_accel_z);
  for (std::size_t master = 0; master < carrier_.masters.size(); ++master) {
    const int node = carrier_.masters[master].mesh_node;
    accel_r[node] = master_accel_r[master];
    accel_z[node] = master_accel_z[master];
  }

  copy_host_to_device(d_accel_r, accel_r, node_count_,
                      "carrier condensation accel_r H2D failed");
  copy_host_to_device(d_accel_z, accel_z, node_count_,
                      "carrier condensation accel_z H2D failed");
  if (timing_enabled_) {
    const auto elapsed = std::chrono::steady_clock::now() - start;
    condensation_ms_ +=
        std::chrono::duration<double, std::milli>(elapsed).count();
    ++condensation_calls_;
  }
}

}  // namespace tenryu::hydro
