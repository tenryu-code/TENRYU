#pragma once

#include <cstddef>
#include <vector>

#include "hydro/carrier_constraint.hpp"

namespace tenryu::core {
struct State;
}

namespace tenryu::hydro {

struct CarrierHookContext {
  explicit CarrierHookContext(const core::State& state);
  ~CarrierHookContext();

  void apply_reconstruct(double* d_pos_r, double* d_pos_z,
                         double* d_vel_r, double* d_vel_z);
  void apply_condensation(double* d_accel_r, double* d_accel_z,
                          const double* d_node_mass);

 private:
  std::size_t node_count_ = 0;
  mesh::BoundaryCarrier carrier_;
  std::vector<int> slave_node_;
  std::vector<int> slave_edge_;
  std::vector<double> slave_lambda_;
  std::vector<int> boundary_nodes_;
  double* host_scratch_ = nullptr;
  bool timing_enabled_ = false;
  double reconstruct_ms_ = 0.0;
  double condensation_ms_ = 0.0;
  int reconstruct_calls_ = 0;
  int condensation_calls_ = 0;
};

}  // namespace tenryu::hydro
