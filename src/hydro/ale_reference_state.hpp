#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::ale {
namespace reference_state_detail {

constexpr std::uint64_t kFnv1aOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnv1aPrime = 1099511628211ULL;

inline void fnv1a_append_bytes(std::uint64_t& hash,
                               const void* data,
                               const std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    hash ^= static_cast<std::uint64_t>(bytes[i]);
    hash *= kFnv1aPrime;
  }
}

inline void fnv1a_append_field(std::uint64_t& hash,
                               const tenryu::core::NodeField1D& field) {
  std::vector<double> values;
  field.copy_to_host(values);
  fnv1a_append_bytes(hash,
                     values.data(),
                     values.size() * sizeof(double));
}

}  // namespace reference_state_detail

inline void reanchor_moving_reference(tenryu::core::State& state) {
  TENRYU_ASSERT(state.x_r.size() == state.x_z.size(),
                "ALE reference re-anchor requires paired current coordinates");
  TENRYU_ASSERT(state.x_r_reference.size() == state.x_r.size() &&
                    state.x_z_reference.size() == state.x_z.size(),
                "ALE reference re-anchor requires node-sized moving reference");

  std::vector<double> x_r;
  std::vector<double> x_z;
  state.x_r.copy_to_host(x_r);
  state.x_z.copy_to_host(x_z);
  state.x_r_reference.copy_from_host(x_r);
  state.x_z_reference.copy_from_host(x_z);
  ++state.reference_epoch;
}

inline std::uint64_t reference_state_hash(
    const tenryu::core::State& state) {
  std::uint64_t hash = reference_state_detail::kFnv1aOffset;
  reference_state_detail::fnv1a_append_bytes(
      hash, &state.reference_epoch, sizeof(state.reference_epoch));
  reference_state_detail::fnv1a_append_field(hash, state.x_r_initial);
  reference_state_detail::fnv1a_append_field(hash, state.x_z_initial);
  reference_state_detail::fnv1a_append_field(hash, state.x_r_reference);
  reference_state_detail::fnv1a_append_field(hash, state.x_z_reference);
  reference_state_detail::fnv1a_append_field(hash, state.x_r_shock_target);
  reference_state_detail::fnv1a_append_field(hash, state.x_z_shock_target);
  return hash;
}

}  // namespace tenryu::hydro::ale
