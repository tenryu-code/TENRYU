#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <functional>
#include <utility>
#include <vector>

#include "core/error.hpp"
#include "mesh/interface_ladder.hpp"

namespace tenryu::mesh::assembly {

struct JunctionVertex {
  int id = -1;
  double r = 0.0;
  double z = 0.0;
};

// Analytic curve segment parameterized by arclength s in [0, length].
struct InterfaceCurve {
  std::function<void(double s, double* r, double* z)> eval;
  double length = 0.0;
};

struct InterfaceContract {
  int id = -1;
  InterfaceCurve curve;
  int vertex_begin = -1;
  int vertex_end = -1;
  std::vector<double> interior_landmarks;
  SizingProposal side_a;
  SizingProposal side_b;
  int n_intervals = 0;
  std::vector<double> s_coords;
  std::vector<int> node_ids;
};

class InterfaceRegistry {
 public:
  int add_vertex(const double r, const double z) {
    for (const JunctionVertex& vertex : vertices_) {
      TENRYU_ASSERT(!(bitwise_equal(vertex.r, r) &&
                      bitwise_equal(vertex.z, z)),
                    "junction vertex coordinates must be unique");
    }
    const int id = static_cast<int>(vertices_.size());
    vertices_.push_back(JunctionVertex{id, r, z});
    return id;
  }

  int add_contract(InterfaceContract contract) {
    TENRYU_ASSERT(contract.id >= 0,
                  "interface contract id must be non-negative");
    const auto position = std::lower_bound(
        contracts_.begin(), contracts_.end(), contract.id,
        [](const InterfaceContract& existing, const int id) {
          return existing.id < id;
        });
    TENRYU_ASSERT(position == contracts_.end() || position->id != contract.id,
                  "interface contract id must be unique");
    TENRYU_ASSERT(contract.curve.length > 0.0,
                  "interface curve length must be positive");
    TENRYU_ASSERT(static_cast<bool>(contract.curve.eval),
                  "interface curve evaluator must be set");
    TENRYU_ASSERT(contract.vertex_begin >= 0 &&
                      static_cast<std::size_t>(contract.vertex_begin) <
                          vertices_.size(),
                  "interface begin vertex id must exist");
    TENRYU_ASSERT(contract.vertex_end >= 0 &&
                      static_cast<std::size_t>(contract.vertex_end) <
                          vertices_.size(),
                  "interface end vertex id must exist");

    double previous = 0.0;
    for (const double landmark : contract.interior_landmarks) {
      TENRYU_ASSERT(landmark > previous && landmark < contract.curve.length,
                    "interface landmarks must be strictly increasing and interior");
      previous = landmark;
    }
    TENRYU_ASSERT(contract.n_intervals >= 1,
                  "interface interval count must be positive");
    TENRYU_ASSERT(
        static_cast<std::size_t>(contract.n_intervals) >=
            contract.interior_landmarks.size() + 1U,
        "interface interval count must cover every landmark interval");

    const int id = contract.id;
    contracts_.insert(position, std::move(contract));
    return id;
  }

  void finalize() {
    TENRYU_ASSERT(!finalized_,
                  "interface registry must only be finalized once");

    node_r_.resize(static_cast<std::size_t>(n_nodes()));
    node_z_.resize(static_cast<std::size_t>(n_nodes()));
    for (const JunctionVertex& vertex : vertices_) {
      node_r_[static_cast<std::size_t>(vertex.id)] = vertex.r;
      node_z_[static_cast<std::size_t>(vertex.id)] = vertex.z;
    }

    int next_node_id = n_vertices();
    for (InterfaceContract& interface : contracts_) {
      const MergedSizing merged =
          merge_sizing(interface.side_a, interface.side_b);
      TENRYU_ASSERT(merged.feasible,
                    "interface sizing proposals must be feasible");
      const double h = merged.h;
      interface.s_coords = equidistribute_with_landmarks(
          interface.curve.length, [h](const double) { return h; },
          interface.interior_landmarks, interface.n_intervals);

      interface.node_ids.resize(
          static_cast<std::size_t>(interface.n_intervals) + 1U);
      interface.node_ids.front() = interface.vertex_begin;
      interface.node_ids.back() = interface.vertex_end;

      const JunctionVertex& begin =
          vertices_[static_cast<std::size_t>(interface.vertex_begin)];
      const JunctionVertex& end =
          vertices_[static_cast<std::size_t>(interface.vertex_end)];
      const double endpoint_tolerance =
          1.0e-9 * std::max(interface.curve.length, 1.0);

      double curve_r = 0.0;
      double curve_z = 0.0;
      interface.curve.eval(0.0, &curve_r, &curve_z);
      TENRYU_ASSERT(std::abs(curve_r - begin.r) <= endpoint_tolerance &&
                        std::abs(curve_z - begin.z) <= endpoint_tolerance,
                    "interface curve begin must match its junction vertex");
      interface.curve.eval(interface.curve.length, &curve_r, &curve_z);
      TENRYU_ASSERT(std::abs(curve_r - end.r) <= endpoint_tolerance &&
                        std::abs(curve_z - end.z) <= endpoint_tolerance,
                    "interface curve end must match its junction vertex");

      for (int j = 1; j < interface.n_intervals; ++j) {
        const int node_id = next_node_id++;
        interface.node_ids[static_cast<std::size_t>(j)] = node_id;
        interface.curve.eval(
            interface.s_coords[static_cast<std::size_t>(j)],
            &node_r_[static_cast<std::size_t>(node_id)],
            &node_z_[static_cast<std::size_t>(node_id)]);
      }
    }
    finalized_ = true;
  }

  int n_vertices() const { return static_cast<int>(vertices_.size()); }

  int n_nodes() const {
    int count = n_vertices();
    for (const InterfaceContract& interface : contracts_) {
      count += interface.n_intervals - 1;
    }
    return count;
  }

  const std::vector<double>& node_r() const { return node_r_; }

  const std::vector<double>& node_z() const { return node_z_; }

  const InterfaceContract& contract(const int id) const {
    const auto position = std::lower_bound(
        contracts_.begin(), contracts_.end(), id,
        [](const InterfaceContract& existing, const int requested_id) {
          return existing.id < requested_id;
        });
    TENRYU_ASSERT(position != contracts_.end() && position->id == id,
                  "interface contract id must exist");
    return *position;
  }

  static std::vector<int> reversed(const std::vector<int>& ids) {
    return std::vector<int>(ids.rbegin(), ids.rend());
  }

 private:
  static bool bitwise_equal(const double lhs, const double rhs) {
    return std::memcmp(&lhs, &rhs, sizeof(double)) == 0;
  }

  std::vector<JunctionVertex> vertices_;
  std::vector<InterfaceContract> contracts_;
  std::vector<double> node_r_;
  std::vector<double> node_z_;
  bool finalized_ = false;
};

}  // namespace tenryu::mesh::assembly
