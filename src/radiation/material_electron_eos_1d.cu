#include "radiation/material_electron_eos_1d.hpp"

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

#include "core/error.hpp"
#include "materials/eos_device_table.hpp"
#include "materials/eos_table.hpp"
#include "materials/material_closure.hpp"

namespace tenryu::radiation {
namespace {

void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

// Every material's electron table on the device and a device array of views
// indexed by the Config material slot (multi-material closure, 2026-09-14;
// moved from the FLD solver 2026-09-24).
class MaterialElectronEOSCache {
 public:
  ~MaterialElectronEOSCache() {
    if (d_views_ != nullptr) {
      static_cast<void>(cudaFree(d_views_));
    }
  }

  const materials::DeviceEOSTableView* device_views_for(const core::Config& cfg) {
    const auto& mats = cfg.materials.materials;
    std::vector<const materials::EOSTableTriplet*> keys(mats.size(), nullptr);
    for (std::size_t m = 0; m < mats.size(); ++m) {
      keys[m] = mats[m].eos_tables.get();
    }
    if (keys != keys_ || d_views_ == nullptr) {
      tables_.clear();
      tables_.resize(mats.size());
      std::vector<materials::DeviceEOSTableView> host_views(mats.size());
      for (std::size_t m = 0; m < mats.size(); ++m) {
        // A material on the exact ideal-gas hydro backend closes with the
        // ideal gas: empty view (2026-09-24; its retained tables used to
        // serve its cells' matter update in multi-material runs).
        if (mats[m].eos_tables && mats[m].hydro_eos_backend != "exact_ideal_gas") {
          tables_[m].upload(mats[m].eos_tables->electron);
        }
        host_views[m] = tables_[m].view();
      }
      if (d_views_ != nullptr) {
        cuda_check(cudaFree(d_views_), "material electron view array free failed");
        d_views_ = nullptr;
      }
      if (!host_views.empty()) {
        cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_views_),
                              host_views.size() * sizeof(materials::DeviceEOSTableView)),
                   "material electron view array alloc failed");
        cuda_check(cudaMemcpy(d_views_, host_views.data(),
                              host_views.size() * sizeof(materials::DeviceEOSTableView),
                              cudaMemcpyHostToDevice),
                   "material electron view array H2D failed");
      }
      keys_ = keys;
    }
    return d_views_;
  }

  [[nodiscard]] int n_materials() const {
    return static_cast<int>(keys_.size());
  }

 private:
  std::vector<const materials::EOSTableTriplet*> keys_;
  std::vector<materials::DeviceEOSTable> tables_;
  materials::DeviceEOSTableView* d_views_ = nullptr;
};

MaterialElectronEOSCache& material_electron_eos_cache() {
  static MaterialElectronEOSCache cache;
  return cache;
}

}  // namespace

materials::CellEOSTableSelector cell_electron_table_selector_1d(core::State& state,
                                                                const core::Config& cfg,
                                                                const int n_cells) {
  materials::CellEOSTableSelector selector;
  int n_nonvoid = 0;
  bool any_table = false;
  for (const auto& m : cfg.materials.materials) {
    if (!m.is_void) {
      ++n_nonvoid;
    }
    if (m.eos_tables && m.hydro_eos_backend != "exact_ideal_gas") {
      any_table = true;
    }
  }
  const materials::MaterialClosureParams* closure_params =
      materials::selector_closure_params(cfg);
  if (n_nonvoid <= 1 || (!any_table && closure_params == nullptr) || n_cells <= 0) {
    return selector;
  }
  state.ensure_cell_material_props(cfg);
  if (state.cell_material_index.size() != static_cast<std::size_t>(n_cells)) {
    return selector;
  }
  auto& cache = material_electron_eos_cache();
  selector.electron_views = cache.device_views_for(cfg);
  selector.n_materials = cache.n_materials();
  selector.cell_material_index = state.cell_material_index.data();
  selector.closure_params = closure_params;
  return selector;
}

}  // namespace tenryu::radiation
