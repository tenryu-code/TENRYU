#include "materials/material_closure.hpp"

#include <cuda_runtime.h>

#include <cstddef>

#include "core/config.hpp"
#include "core/error.hpp"

namespace tenryu::materials {
namespace {

bool same_params(const MaterialClosureParams& a, const MaterialClosureParams& b) {
  return a.Z == b.Z && a.A == b.A && a.gamma == b.gamma &&
         a.cv_e_override == b.cv_e_override && a.eos_T_ref_eV == b.eos_T_ref_eV &&
         a.hydro_backend_kind == b.hydro_backend_kind && a.is_void == b.is_void;
}

// Upload of the per-material closure parameters, kept for the process (the
// materials of a run are immutable).
class MaterialClosureParamsCache {
 public:
  ~MaterialClosureParamsCache() {
    if (d_params_ != nullptr) {
      static_cast<void>(cudaFree(d_params_));
    }
  }

  const MaterialClosureParams* device_params(const std::vector<MaterialClosureParams>& params) {
    if (params.empty()) {
      return nullptr;
    }
    bool same = (d_params_ != nullptr && params.size() == host_.size());
    for (std::size_t m = 0; same && m < params.size(); ++m) {
      same = same_params(params[m], host_[m]);
    }
    if (same) {
      return d_params_;
    }
    if (d_params_ != nullptr) {
      TENRYU_ASSERT(cudaFree(d_params_) == cudaSuccess,
                    "material closure parameter array free failed");
      d_params_ = nullptr;
    }
    const std::size_t bytes = params.size() * sizeof(MaterialClosureParams);
    TENRYU_ASSERT(cudaMalloc(reinterpret_cast<void**>(&d_params_), bytes) == cudaSuccess,
                  "material closure parameter array alloc failed");
    TENRYU_ASSERT(
        cudaMemcpy(d_params_, params.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess,
        "material closure parameter array H2D failed");
    host_ = params;
    return d_params_;
  }

 private:
  std::vector<MaterialClosureParams> host_;
  MaterialClosureParams* d_params_ = nullptr;
};

MaterialClosureParamsCache& closure_params_cache() {
  static MaterialClosureParamsCache cache;
  return cache;
}

}  // namespace

std::uint8_t hydro_backend_kind_of(const core::Config& cfg, const int material_index) {
  const auto& mats = cfg.materials.materials;
  if (material_index < 0 || material_index >= static_cast<int>(mats.size())) {
    return kHydroBackendLegacy;
  }
  const auto& mat = mats[static_cast<std::size_t>(material_index)];
  const std::string& backend = mat.hydro_eos_backend;
  if (backend == "exact_ideal_gas") {
    return kHydroBackendExactIdealGas;
  }
  if (!mat.eos_tables) {
    return kHydroBackendLegacy;
  }
  if (backend == "rho_e_table") {
    return kHydroBackendRhoETable;
  }
  if (backend == "mie_gruneisen") {
    return kHydroBackendMieGruneisen;
  }
  if (backend == "helmholtz_spline") {
    return kHydroBackendHelmholtzSpline;
  }
  if (backend == "helmholtz_jet") {
    return kHydroBackendHelmholtzJet;
  }
  return kHydroBackendLegacy;
}

std::vector<MaterialClosureParams> material_closure_params(const core::Config& cfg) {
  const auto& mats = cfg.materials.materials;
  std::vector<MaterialClosureParams> params(mats.size());
  for (std::size_t m = 0; m < mats.size(); ++m) {
    MaterialClosureParams& p = params[m];
    p.Z = mats[m].Z;
    p.A = mats[m].A;
    p.gamma = mats[m].ideal_gas_gamma;
    p.cv_e_override = mats[m].cv_e_override;
    p.eos_T_ref_eV = mats[m].eos_T_ref_eV;
    p.hydro_backend_kind = hydro_backend_kind_of(cfg, static_cast<int>(m));
    p.is_void = mats[m].is_void ? 1u : 0u;
  }
  return params;
}

namespace {

template <typename Differs>
bool nonvoid_materials_differ(const core::Config& cfg, Differs differs) {
  if (cfg.main.dim != 1) {
    return false;
  }
  const std::vector<MaterialClosureParams> params = material_closure_params(cfg);
  const MaterialClosureParams* first = nullptr;
  for (const MaterialClosureParams& p : params) {
    if (p.is_void != 0u) {
      continue;
    }
    if (first == nullptr) {
      first = &p;
    } else if (differs(*first, p)) {
      return true;
    }
  }
  return false;
}

}  // namespace

bool material_closure_params_vary(const core::Config& cfg) {
  return nonvoid_materials_differ(
      cfg, [](const MaterialClosureParams& a, const MaterialClosureParams& b) {
        return a.hydro_backend_kind != b.hydro_backend_kind ||
               a.cv_e_override != b.cv_e_override || a.eos_T_ref_eV != b.eos_T_ref_eV;
      });
}

bool hydro_backend_kinds_vary(const core::Config& cfg) {
  return nonvoid_materials_differ(
      cfg, [](const MaterialClosureParams& a, const MaterialClosureParams& b) {
        return a.hydro_backend_kind != b.hydro_backend_kind;
      });
}

bool material_A_or_Z_vary(const core::Config& cfg) {
  return nonvoid_materials_differ(
      cfg, [](const MaterialClosureParams& a, const MaterialClosureParams& b) {
        return a.A != b.A || a.Z != b.Z;
      });
}

const MaterialClosureParams* material_closure_params_device(const core::Config& cfg) {
  if (cfg.main.dim != 1 || cfg.materials.materials.empty()) {
    return nullptr;
  }
  return closure_params_cache().device_params(material_closure_params(cfg));
}

const MaterialClosureParams* selector_closure_params(const core::Config& cfg) {
  return material_closure_params_vary(cfg) ? material_closure_params_device(cfg) : nullptr;
}

}  // namespace tenryu::materials
