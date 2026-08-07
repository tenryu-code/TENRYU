#include "hydro/eos_context.hpp"

#include <cuda_runtime.h>

#include <utility>

#include "core/config.hpp"
#include "core/error.hpp"
#include "materials/eos_rho_e_table.hpp"
#include "materials/eos_table.hpp"
#include "materials/helmholtz_jet.hpp"
#include "materials/helmholtz_bspline.hpp"
#include "materials/helmholtz_spline.hpp"
#include "materials/mie_gruneisen.hpp"

namespace tenryu::hydro {

namespace {
inline void cuda_check(const cudaError_t err, const char* msg) {
  TENRYU_ASSERT(err == cudaSuccess, msg);
}
}  // namespace

void HydroEOSContext::initialize(const core::Config& cfg) {
  destroy();
  const auto& mats = cfg.materials.materials;
  n_materials = static_cast<int>(mats.size());
  if (n_materials == 0) return;

  ion.resize(static_cast<std::size_t>(n_materials));
  electron.resize(static_cast<std::size_t>(n_materials));
  total.resize(static_cast<std::size_t>(n_materials));
  total_rho_e.resize(static_cast<std::size_t>(n_materials));
  ion_helmholtz.resize(static_cast<std::size_t>(n_materials));
  electron_helmholtz.resize(static_cast<std::size_t>(n_materials));
  total_helmholtz.resize(static_cast<std::size_t>(n_materials));
  total_helmholtz_jet.resize(static_cast<std::size_t>(n_materials));
  mie_gruneisen.resize(static_cast<std::size_t>(n_materials));
  hydro_backend_kind.assign(static_cast<std::size_t>(n_materials), 0u);
  rho_e_reclosure_supported.assign(static_cast<std::size_t>(n_materials), 0u);

  any_table = false;
  any_helmholtz_spline = false;
  any_helmholtz_jet = false;
  for (int m = 0; m < n_materials; ++m) {
    const auto& mat = mats[static_cast<std::size_t>(m)];
    if (mat.eos_tables) {
      ion[static_cast<std::size_t>(m)].upload(mat.eos_tables->ion);
      electron[static_cast<std::size_t>(m)].upload(mat.eos_tables->electron);
      total[static_cast<std::size_t>(m)].upload(mat.eos_tables->total);
      any_table = true;
    }
    if (mat.eos_model == "ideal_gas") {
      rho_e_reclosure_supported[static_cast<std::size_t>(m)] = 1u;
    } else if (mat.eos_model == "tmat" && mat.hydro_eos_backend == "legacy" &&
               mat.eos_tables) {
      const bool supports =
          cfg.main.two_temperature
              ? (ion[static_cast<std::size_t>(m)].supports_rho_e_reclosure() &&
                 electron[static_cast<std::size_t>(m)].supports_rho_e_reclosure())
              : total[static_cast<std::size_t>(m)].supports_rho_e_reclosure();
      rho_e_reclosure_supported[static_cast<std::size_t>(m)] = supports ? 1u : 0u;
    }
    if (mat.eos_tables && mat.hydro_eos_backend == "rho_e_table") {
      core::log_info("Hydro rho-e table backend active for material '" + mat.name +
                     "' (precomputed total P(rho,e), T(rho,e) surrogate only; " +
                     std::string(cfg.numerics.hydro.rho_e_linear_grid ? "linear" : "log") +
                     " rho-e grid).");
      materials::EOSRhoETable rho_e_table;
      rho_e_table.build_from_rhoT_table(mat.eos_tables->total,
                                        cfg.numerics.hydro.rho_e_linear_grid);
      total_rho_e[static_cast<std::size_t>(m)].upload(rho_e_table);
      hydro_backend_kind[static_cast<std::size_t>(m)] = 4u;
    }
    if (mat.hydro_eos_backend == "exact_ideal_gas") {
      core::log_info("Hydro exact ideal-gas closure backend active for material '" + mat.name +
                     "' (1D diagnostic; raw tables retained for radiation/transport).");
      hydro_backend_kind[static_cast<std::size_t>(m)] = 3u;
    }
    if (mat.eos_tables && mat.hydro_eos_backend == "mie_gruneisen") {
      core::log_info("Hydro trajectory-fitted Mie-Gruneisen backend active for material '" +
                     mat.name + "' (1D 2T diagnostic; hydro P/cs use affine branch closure).");
      materials::MieGruneisenEOS mg;
      mg.build_from_tables(mat.eos_tables->ion, mat.eos_tables->electron,
                           mat.mg_T_ref_eV, mat.mg_dT_rel, mat.name);
      mie_gruneisen[static_cast<std::size_t>(m)].upload(mg);
      hydro_backend_kind[static_cast<std::size_t>(m)] = 5u;
    }
    // Quintic B-spline Helmholtz fit test (Phase 1 diagnostic)
    if (mat.eos_tables && mat.hydro_eos_backend == "helmholtz_spline") {
      // Phase 1: also run quintic B-spline fit for diagnostic
      core::log_info("Testing Helmholtz quintic B-spline fit for material '" + mat.name + "'...");
      materials::HelmholtzBSpline1T bspline_fit;
      materials::HelmholtzBSplineFitOptions opts;
      bspline_fit.build_from_table(mat.eos_tables->total, opts, "total_bspline");
      core::log_info("Helmholtz quintic B-spline fit complete for material '" + mat.name + "'.");
    }
    if (mat.eos_tables && mat.hydro_eos_backend == "helmholtz_spline") {
      core::log_info("Hydro bicubic EOS spline backend active for material '" + mat.name +
                     "' (compat key: helmholtz_spline, total-EOS surrogate only).");
      const materials::HelmholtzSplineTriplet splines =
          materials::build_helmholtz_spline_triplet(*mat.eos_tables);
      total_helmholtz[static_cast<std::size_t>(m)].upload(splines.total);
      hydro_backend_kind[static_cast<std::size_t>(m)] = 1u;
      any_helmholtz_spline = true;
    }
    if (mat.eos_tables && mat.hydro_eos_backend == "helmholtz_jet") {
      core::log_info("Hydro Helmholtz jet backend active for material '" + mat.name +
                     "' (total-EOS projected-jet surrogate only).");
      materials::HelmholtzJetEOS jet;
      jet.build_from_table(mat.eos_tables->total, "total_jet");
      total_helmholtz_jet[static_cast<std::size_t>(m)].upload(jet);
      hydro_backend_kind[static_cast<std::size_t>(m)] = 2u;
      any_helmholtz_jet = true;
    }
    // else: DeviceEOSTable default-constructed -> empty -> n_rho==0 -> ideal gas fallback
  }

  if (!any_table) return;

  // Upload view arrays to device
  const std::size_t view_bytes = sizeof(materials::DeviceEOSTableView) *
                                 static_cast<std::size_t>(n_materials);
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_ion_views), view_bytes),
             "HydroEOSContext::initialize cudaMalloc d_ion_views");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_electron_views), view_bytes),
             "HydroEOSContext::initialize cudaMalloc d_electron_views");
  cuda_check(cudaMalloc(reinterpret_cast<void**>(&d_total_views), view_bytes),
             "HydroEOSContext::initialize cudaMalloc d_total_views");

  std::vector<materials::DeviceEOSTableView> h_ion_views(static_cast<std::size_t>(n_materials));
  std::vector<materials::DeviceEOSTableView> h_ele_views(static_cast<std::size_t>(n_materials));
  std::vector<materials::DeviceEOSTableView> h_total_views(
      static_cast<std::size_t>(n_materials));
  for (int m = 0; m < n_materials; ++m) {
    h_ion_views[static_cast<std::size_t>(m)] = ion[static_cast<std::size_t>(m)].view();
    h_ele_views[static_cast<std::size_t>(m)] = electron[static_cast<std::size_t>(m)].view();
    h_total_views[static_cast<std::size_t>(m)] = total[static_cast<std::size_t>(m)].view();
  }
  cuda_check(cudaMemcpy(d_ion_views, h_ion_views.data(), view_bytes, cudaMemcpyHostToDevice),
             "HydroEOSContext::initialize cudaMemcpy d_ion_views");
  cuda_check(cudaMemcpy(d_electron_views, h_ele_views.data(), view_bytes, cudaMemcpyHostToDevice),
             "HydroEOSContext::initialize cudaMemcpy d_electron_views");
  cuda_check(cudaMemcpy(d_total_views, h_total_views.data(), view_bytes, cudaMemcpyHostToDevice),
             "HydroEOSContext::initialize cudaMemcpy d_total_views");
}

void HydroEOSContext::destroy() {
  if (d_total_views) {
    cudaFree(d_total_views);
    d_total_views = nullptr;
  }
  if (d_electron_views) {
    cudaFree(d_electron_views);
    d_electron_views = nullptr;
  }
  if (d_ion_views) {
    cudaFree(d_ion_views);
    d_ion_views = nullptr;
  }
  ion.clear();
  electron.clear();
  total.clear();
  total_rho_e.clear();
  ion_helmholtz.clear();
  electron_helmholtz.clear();
  total_helmholtz.clear();
  total_helmholtz_jet.clear();
  mie_gruneisen.clear();
  hydro_backend_kind.clear();
  rho_e_reclosure_supported.clear();
  n_materials = 0;
  any_table = false;
  any_helmholtz_spline = false;
  any_helmholtz_jet = false;
}

HydroEOSContext::~HydroEOSContext() {
  destroy();
}

HydroEOSContext::HydroEOSContext(HydroEOSContext&& o) noexcept
    : ion(std::move(o.ion)),
      electron(std::move(o.electron)),
      total(std::move(o.total)),
      total_rho_e(std::move(o.total_rho_e)),
      ion_helmholtz(std::move(o.ion_helmholtz)),
      electron_helmholtz(std::move(o.electron_helmholtz)),
      total_helmholtz(std::move(o.total_helmholtz)),
      total_helmholtz_jet(std::move(o.total_helmholtz_jet)),
      mie_gruneisen(std::move(o.mie_gruneisen)),
      hydro_backend_kind(std::move(o.hydro_backend_kind)),
      rho_e_reclosure_supported(std::move(o.rho_e_reclosure_supported)),
      d_ion_views(o.d_ion_views),
      d_electron_views(o.d_electron_views),
      d_total_views(o.d_total_views),
      n_materials(o.n_materials),
      any_table(o.any_table),
      any_helmholtz_spline(o.any_helmholtz_spline),
      any_helmholtz_jet(o.any_helmholtz_jet) {
  o.d_ion_views = nullptr;
  o.d_electron_views = nullptr;
  o.d_total_views = nullptr;
  o.n_materials = 0;
  o.any_table = false;
  o.any_helmholtz_spline = false;
  o.any_helmholtz_jet = false;
}

HydroEOSContext& HydroEOSContext::operator=(HydroEOSContext&& o) noexcept {
  if (this != &o) {
    destroy();
    ion = std::move(o.ion);
    electron = std::move(o.electron);
    total = std::move(o.total);
    total_rho_e = std::move(o.total_rho_e);
    ion_helmholtz = std::move(o.ion_helmholtz);
    electron_helmholtz = std::move(o.electron_helmholtz);
    total_helmholtz = std::move(o.total_helmholtz);
    total_helmholtz_jet = std::move(o.total_helmholtz_jet);
    mie_gruneisen = std::move(o.mie_gruneisen);
    hydro_backend_kind = std::move(o.hydro_backend_kind);
    rho_e_reclosure_supported = std::move(o.rho_e_reclosure_supported);
    d_ion_views = o.d_ion_views;
    d_electron_views = o.d_electron_views;
    d_total_views = o.d_total_views;
    n_materials = o.n_materials;
    any_table = o.any_table;
    any_helmholtz_spline = o.any_helmholtz_spline;
    any_helmholtz_jet = o.any_helmholtz_jet;
    o.d_ion_views = nullptr;
    o.d_electron_views = nullptr;
    o.d_total_views = nullptr;
    o.n_materials = 0;
    o.any_table = false;
    o.any_helmholtz_spline = false;
    o.any_helmholtz_jet = false;
  }
  return *this;
}

bool HydroEOSContext::material_uses_helmholtz(const int material_index) const {
  return material_index >= 0 &&
         material_index < static_cast<int>(hydro_backend_kind.size()) &&
         (hydro_backend_kind[static_cast<std::size_t>(material_index)] == 1u ||
          hydro_backend_kind[static_cast<std::size_t>(material_index)] == 2u);
}

std::uint8_t HydroEOSContext::material_hydro_backend_kind(const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(hydro_backend_kind.size())) {
    return 0u;
  }
  return hydro_backend_kind[static_cast<std::size_t>(material_index)];
}

bool HydroEOSContext::material_supports_rho_e_reclosure(const int material_index) const {
  if (material_index < 0 ||
      material_index >= static_cast<int>(rho_e_reclosure_supported.size())) {
    return false;
  }
  return rho_e_reclosure_supported[static_cast<std::size_t>(material_index)] != 0u;
}

materials::DeviceEOSTableView HydroEOSContext::ion_view(const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(ion.size())) {
    return materials::DeviceEOSTableView{};
  }
  return ion[static_cast<std::size_t>(material_index)].view();
}

materials::DeviceEOSTableView HydroEOSContext::electron_view(const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(electron.size())) {
    return materials::DeviceEOSTableView{};
  }
  return electron[static_cast<std::size_t>(material_index)].view();
}

materials::DeviceEOSTableView HydroEOSContext::total_view(const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(total.size())) {
    return materials::DeviceEOSTableView{};
  }
  return total[static_cast<std::size_t>(material_index)].view();
}

materials::EOSRhoEDeviceView HydroEOSContext::total_rho_e_view(const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(total_rho_e.size())) {
    return materials::EOSRhoEDeviceView{};
  }
  return total_rho_e[static_cast<std::size_t>(material_index)].view();
}

materials::HelmholtzSplineDeviceView HydroEOSContext::ion_helmholtz_view(
    const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(ion_helmholtz.size())) {
    return materials::HelmholtzSplineDeviceView{};
  }
  return ion_helmholtz[static_cast<std::size_t>(material_index)].view();
}

materials::HelmholtzSplineDeviceView HydroEOSContext::electron_helmholtz_view(
    const int material_index) const {
  if (material_index < 0 ||
      material_index >= static_cast<int>(electron_helmholtz.size())) {
    return materials::HelmholtzSplineDeviceView{};
  }
  return electron_helmholtz[static_cast<std::size_t>(material_index)].view();
}

materials::HelmholtzSplineDeviceView HydroEOSContext::total_helmholtz_view(
    const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(total_helmholtz.size())) {
    return materials::HelmholtzSplineDeviceView{};
  }
  return total_helmholtz[static_cast<std::size_t>(material_index)].view();
}

materials::HelmholtzJetDeviceView HydroEOSContext::total_helmholtz_jet_view(
    const int material_index) const {
  if (material_index < 0 ||
      material_index >= static_cast<int>(total_helmholtz_jet.size())) {
    return materials::HelmholtzJetDeviceView{};
  }
  return total_helmholtz_jet[static_cast<std::size_t>(material_index)].view();
}

materials::MieGruneisenDeviceView HydroEOSContext::mie_gruneisen_view(
    const int material_index) const {
  if (material_index < 0 || material_index >= static_cast<int>(mie_gruneisen.size())) {
    return materials::MieGruneisenDeviceView{};
  }
  return mie_gruneisen[static_cast<std::size_t>(material_index)].view();
}

}  // namespace tenryu::hydro
