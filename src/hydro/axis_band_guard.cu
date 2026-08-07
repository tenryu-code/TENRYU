#include "hydro/axis_band_guard.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"

namespace tenryu::hydro::ale {
namespace {

std::size_t checked_count(const int a, const int b, const char* what) {
  TENRYU_ASSERT(a >= 0 && b >= 0, what);
  return static_cast<std::size_t>(a) * static_cast<std::size_t>(b);
}

void validate_band(const core::State& state, const int K) {
  TENRYU_ASSERT(state.mesh.dim == 2, "axis band snapshot requires 2D mesh");
  TENRYU_ASSERT(K >= 1, "axis band snapshot requires K >= 1");
  TENRYU_ASSERT(K <= state.mesh.topo.nr,
                "axis band snapshot requires K <= mesh nr");
  TENRYU_ASSERT(state.mesh.topo.nz >= 1,
                "axis band snapshot requires mesh nz >= 1");
}

int material_count(const core::State& state) {
  const std::size_t n_cells = checked_count(state.mesh.topo.nr, state.mesh.topo.nz,
                                            "axis band snapshot invalid mesh size");
  if (state.volFrac.empty()) {
    return 0;
  }
  TENRYU_ASSERT(n_cells > 0, "axis band snapshot requires non-empty cell mesh");
  TENRYU_ASSERT(state.volFrac.size() % n_cells == 0,
                "axis band snapshot volFrac size must be n_cells*n_materials");
  return static_cast<int>(state.volFrac.size() / n_cells);
}

int radiation_group_count(const core::State& state, const bool include_radiation) {
  if (!include_radiation || state.rad_E.empty()) {
    return 0;
  }
  const std::size_t n_cells = checked_count(state.mesh.topo.nr, state.mesh.topo.nz,
                                            "axis band snapshot invalid mesh size");
  TENRYU_ASSERT(n_cells > 0, "axis band snapshot requires non-empty cell mesh");
  TENRYU_ASSERT(state.rad_E.size() % n_cells == 0,
                "axis band snapshot rad_E size must be n_cells*n_groups");
  return static_cast<int>(state.rad_E.size() / n_cells);
}

template <typename Field>
void append_capture_desc(std::vector<core::TransactionBufferDesc>& descs,
                         Field& field,
                         const std::size_t count,
                         const char* name) {
  TENRYU_ASSERT(field.size() >= count,
                (std::string("axis band snapshot field too small: ") + name).c_str());
  descs.push_back({name, field.data(), count * sizeof(double)});
}

template <typename Field>
void append_restore_desc(std::vector<core::TransactionBufferDesc>& descs,
                         Field& field,
                         const std::size_t count,
                         const char* name) {
  TENRYU_ASSERT(field.size() >= count,
                (std::string("axis band snapshot restore field too small: ") +
                 name)
                    .c_str());
  descs.push_back({name, field.data(), count * sizeof(double)});
}

std::vector<core::TransactionBufferDesc> capture_descs(
    core::State& state,
    const std::size_t node_count,
    const std::size_t cell_count,
    const std::size_t mat_count,
    const std::size_t group_count) {
  std::vector<core::TransactionBufferDesc> descs;
  descs.reserve(11);
  append_capture_desc(descs, state.x_r, node_count, "x_r");
  append_capture_desc(descs, state.x_z, node_count, "x_z");
  append_capture_desc(descs, state.v_r, node_count, "v_r");
  append_capture_desc(descs, state.v_z, node_count, "v_z");
  append_capture_desc(descs, state.vol, cell_count, "vol");
  append_capture_desc(descs, state.rho, cell_count, "rho");
  append_capture_desc(descs, state.mass, cell_count, "mass");
  append_capture_desc(descs, state.ee, cell_count, "ee");
  append_capture_desc(descs, state.ei, cell_count, "ei");
  if (mat_count > 0) {
    append_capture_desc(descs, state.volFrac, mat_count, "volFrac");
  }
  if (group_count > 0) {
    append_capture_desc(descs, state.rad_E, group_count, "rad_E");
  }
  return descs;
}

std::vector<core::TransactionBufferDesc> restore_descs(
    core::State& state,
    const std::size_t node_count,
    const std::size_t cell_count,
    const std::size_t mat_count,
    const std::size_t group_count) {
  std::vector<core::TransactionBufferDesc> descs;
  descs.reserve(11);
  append_restore_desc(descs, state.x_r, node_count, "x_r");
  append_restore_desc(descs, state.x_z, node_count, "x_z");
  append_restore_desc(descs, state.v_r, node_count, "v_r");
  append_restore_desc(descs, state.v_z, node_count, "v_z");
  append_restore_desc(descs, state.vol, cell_count, "vol");
  append_restore_desc(descs, state.rho, cell_count, "rho");
  append_restore_desc(descs, state.mass, cell_count, "mass");
  append_restore_desc(descs, state.ee, cell_count, "ee");
  append_restore_desc(descs, state.ei, cell_count, "ei");
  if (mat_count > 0) {
    append_restore_desc(descs, state.volFrac, mat_count, "volFrac");
  }
  if (group_count > 0) {
    append_restore_desc(descs, state.rad_E, group_count, "rad_E");
  }
  return descs;
}

}  // namespace

void AxisBandGuard::capture(core::State& state,
                            const int K,
                            const bool include_radiation,
                            const cudaStream_t stream) {
  validate_band(state, K);

  const int nz = state.mesh.topo.nz;
  const int n_materials = material_count(state);
  const int n_groups = radiation_group_count(state, include_radiation);
  const std::size_t node_count =
      checked_count(K + 1, nz + 1, "axis band snapshot invalid node count");
  const std::size_t cell_count =
      checked_count(K, nz, "axis band snapshot invalid cell count");
  const std::size_t mat_count =
      cell_count * static_cast<std::size_t>(n_materials);
  const std::size_t group_count =
      cell_count * static_cast<std::size_t>(n_groups);
  const auto descs =
      capture_descs(state, node_count, cell_count, mat_count, group_count);

  const core::TransactionError error = tx_.capture(descs, stream);
  TENRYU_ASSERT(error == core::TransactionError::kNone,
                "AxisBandGuard capture failed");

  K_ = K;
  nz_ = nz;
  n_materials_ = n_materials;
  n_groups_ = n_groups;
  include_radiation_ = include_radiation;
  captured_bytes_.clear();
  captured_ptrs_.clear();
  captured_bytes_.reserve(descs.size());
  captured_ptrs_.reserve(descs.size());
  for (const auto& desc : descs) {
    captured_bytes_.push_back(desc.bytes);
    captured_ptrs_.push_back(desc.live_ptr);
  }
}

void AxisBandGuard::restore(core::State& state, const cudaStream_t stream) {
  TENRYU_ASSERT(tx_.captured(),
                "AxisBandGuard restore requires a captured transaction");

  validate_band(state, K_);
  TENRYU_ASSERT(nz_ == state.mesh.topo.nz,
                "axis band snapshot restore nz mismatch");
  TENRYU_ASSERT(n_materials_ == material_count(state),
                "axis band snapshot restore n_materials mismatch");
  TENRYU_ASSERT(n_groups_ ==
                    radiation_group_count(state, include_radiation_),
                "axis band snapshot restore n_groups mismatch");

  const std::size_t node_count =
      checked_count(K_ + 1, nz_ + 1, "axis band snapshot invalid node count");
  const std::size_t cell_count =
      checked_count(K_, nz_, "axis band snapshot invalid cell count");
  const std::size_t mat_count =
      cell_count * static_cast<std::size_t>(n_materials_);
  const std::size_t group_count =
      cell_count * static_cast<std::size_t>(n_groups_);
  const auto descs =
      restore_descs(state, node_count, cell_count, mat_count, group_count);

  TENRYU_ASSERT(descs.size() == captured_bytes_.size(),
                "AxisBandGuard descriptor count changed after capture");
  TENRYU_ASSERT(descs.size() == captured_ptrs_.size(),
                "AxisBandGuard captured pointer count mismatch");

  bool pointers_unchanged = true;
  for (std::size_t i = 0; i < descs.size(); ++i) {
    TENRYU_ASSERT(
        descs[i].bytes == captured_bytes_[i],
        std::string("AxisBandGuard byte-size mismatch for field ") + descs[i].name);
    pointers_unchanged =
        pointers_unchanged && descs[i].live_ptr == captured_ptrs_[i];
  }

  if (pointers_unchanged) {
    tx_.record_gate("axis_band_rollback", true);
    TENRYU_ASSERT(tx_.commit(stream),
                  "AxisBandGuard rollback commit failed");
  } else {
    for (const auto& desc : descs) {
      if (desc.bytes == 0) {
        continue;
      }
      void* const shadow = tx_.shadow_ptr(desc.name);
      TENRYU_ASSERT(
          shadow != nullptr,
          std::string("AxisBandGuard missing shadow slot for field ") + desc.name);
      CUDA_CHECK(cudaMemcpyAsync(desc.live_ptr, shadow, desc.bytes,
                                 cudaMemcpyDeviceToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    tx_.discard();
  }
}

void AxisBandGuard::accept() {
  if (tx_.captured()) {
    tx_.discard();
  }
}

bool AxisBandGuard::captured() const {
  return tx_.captured();
}

}  // namespace tenryu::hydro::ale
