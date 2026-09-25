#pragma once

// Per-material opacity description and the multi-material opacity kernel of
// the 1D radiation solvers: every cell's opacities from its materials at their
// partial densities, mixed by Materials.opacity_mix_rule (NUMERICS §1.1.5).
// Shared by the FLD solver (fld_1d_gpu.cu) and the S_N solver
// (sn_transport_1d_gpu.cu, 2026-09-24). The kernels are in an unnamed
// namespace: each including translation unit has its own copy.

#include "materials/ionmix_reader.cuh"
#include "materials/opacity_eval.cuh"

namespace tenryu::radiation {
namespace {

struct MatOpacityDesc {
  // kind: 0 = constant/void, 1 = LTE table, 2 = NLTE table (tmat or
  // table_nlte; filled by the per-material NLTE launch), 3 = power law
  // (grey), 4 = frequency-dependent Marshak opacity (2026-09-24: kinds 3 and
  // 4 and table_nlte in multi-material decks).
  int kind = 0;
  int is_void = 0;
  double kappa_planck = 0.0;
  double kappa_rosseland = 0.0;
  // Physical scattering opacity [cm^2/g] (S_N; 0 for the tables, unused by
  // FLD).
  double kappa_scatter = 0.0;
  double inv_A_mp = 0.0;
  // Power law kappa = kappa0 (T/T_ref)^-alpha_T (rho/rho_ref)^lambda_rho.
  double pl_kappa0 = 0.0;
  double pl_alpha_T = 0.0;
  double pl_lambda_rho = 0.0;
  double pl_T_ref = 1.0;
  double pl_rho_ref = 1.0;
  materials::IonmixOpacityDeviceView view;
};

#ifdef __CUDACC__

// Opacities [cm^2/g] of one material in group g at density rho_m and
// temperature T for the multi-material kernel. kind 0: the constants; 1/2:
// the table (kind 2 cells of an NLTE-dominant material are overwritten by the
// NLTE launch); 3: the grey power law; 4: the frequency-dependent Marshak
// coefficients, which are absorption coefficients [1/cm] independent of the
// density, divided by rho_m.
__device__ inline void multimat_material_kappas(const MatOpacityDesc& d,
                                                const int g,
                                                const double rho_m,
                                                const double log_ni,
                                                const double log_T,
                                                const double T,
                                                const double* __restrict__ group_bounds,
                                                double* k_pa,
                                                double* k_pe,
                                                double* k_r) {
  if (d.kind == 0) {
    *k_pa = d.kappa_planck;
    *k_pe = d.kappa_planck;
    *k_r = d.kappa_rosseland;
  } else if (d.kind == 3) {
    const double kappa =
        (rho_m > 0.0) ? d.pl_kappa0 * pow(T / d.pl_T_ref, -d.pl_alpha_T) *
                            pow(rho_m / d.pl_rho_ref, d.pl_lambda_rho)
                      : 0.0;
    *k_pa = kappa;
    *k_pe = kappa;
    *k_r = kappa;
  } else if (d.kind == 4) {
    double sigma_p = 0.0;
    double sigma_r = 0.0;
    const double lo = group_bounds[g];
    const double hi = group_bounds[g + 1];
    if (rho_m > 0.0 && isfinite(lo) && isfinite(hi) && lo >= 0.0 && hi > lo) {
      materials::freq_dep_marshak_group_sigmas(fmax(T, 1.0e-6), lo, hi, &sigma_p, &sigma_r);
    }
    const double inv_rho = (rho_m > 0.0) ? 1.0 / rho_m : 0.0;
    *k_pa = sigma_p * inv_rho;
    *k_pe = sigma_p * inv_rho;
    *k_r = sigma_r * inv_rho;
  } else {
    *k_pa = d.view.interpolate(d.view.kappa_PA, g, log_ni, log_T);
    *k_pe = d.view.interpolate(d.view.kappa_PE, g, log_ni, log_T);
    *k_r = d.view.interpolate(d.view.kappa_R, g, log_ni, log_T);
  }
}

// Exact volume-partition mixing: sigma = rho * sum_m w_m * kappa_m, evaluated
// at each material's partial density rho_m = rho*w_m/f_m. Mass fractions are
// preferred, with volume fractions as fallback. NLTE-dominant cells are filled
// by the NLTE launch using the dominant-material approximation.
template <int kMixRule>
__global__ void eval_opacity_multimat_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const int* __restrict__ cell_material_index,
    const double* __restrict__ mass_per_material,  // [n_cells*n_materials] or nullptr
    const double* __restrict__ vol_frac,           // [n_cells*n_materials] or nullptr
    const MatOpacityDesc* __restrict__ descs,
    int n_materials,
    double kappa_floor,
    double kappa_cap,
    double temperature_floor_eV,
    double* __restrict__ sigma_a,
    double* __restrict__ sigma_pe,
    double* __restrict__ sigma_R,
    int n_cells,
    int n_groups,
    const double* __restrict__ group_bounds) {
  constexpr int mix_rule = kMixRule;
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  const int m_dom = min(max(cell_material_index[c], 0), n_materials - 1);
  const MatOpacityDesc desc = descs[m_dom];
  if (desc.kind == 2) {
    return;  // NLTE material: the per-material NLTE launch fills this cell
  }
  const double rho_c = rho[c];
  const double rho_safe = (rho_c >= 0.0) ? rho_c : 0.0;
  const double sigma_min = rho_safe * fmax(kappa_floor, 0.0);
  const double sigma_max = rho_safe * fmax(kappa_cap, 0.0);
  const int base = c * n_groups;

  constexpr int kMaxMats = 16;
  // Raw weights: mass fractions when tracked, volume fractions otherwise,
  // dominant-only as the last resort. Void never participates.
  double raw[kMaxMats];
  double wsum = 0.0;
  for (int m = 0; m < n_materials; ++m) {
    double r = 0.0;
    if (!descs[m].is_void) {
      if (mass_per_material != nullptr) {
        r = fmax(mass_per_material[c * n_materials + m], 0.0);
      } else if (vol_frac != nullptr) {
        r = fmax(vol_frac[c * n_materials + m], 0.0);
      } else {
        r = (m == m_dom) ? 1.0 : 0.0;
      }
    }
    raw[m] = r;
    wsum += r;
  }

  if (!(wsum > 0.0)) {
    if (desc.kind == 0) {
      const double sigma_a_c = fmin(
          fmax(fmax(rho_safe * desc.kappa_planck, 0.0), sigma_min), sigma_max);
      const double sigma_R_c = fmin(
          fmax(fmax(rho_safe * desc.kappa_rosseland, 0.0), sigma_min), sigma_max);
      for (int g = 0; g < n_groups; ++g) {
        sigma_a[base + g] = sigma_a_c;
        sigma_pe[base + g] = sigma_a_c;
        sigma_R[base + g] = sigma_R_c;
      }
      return;
    }

    const double log_ni = log(fmax(rho_safe * desc.inv_A_mp, 1.0e-300));
    const double log_T = log(fmax(Te[c], temperature_floor_eV));
    for (int g = 0; g < n_groups; ++g) {
      double kappa_pa = 0.0;
      double kappa_pe = 0.0;
      double kappa_r = 0.0;
      multimat_material_kappas(desc, g, rho_safe, log_ni, log_T, Te[c], group_bounds,
                               &kappa_pa, &kappa_pe, &kappa_r);
      sigma_a[base + g] = fmin(
          fmax(fmax(rho_safe * kappa_pa, 0.0), sigma_min), sigma_max);
      sigma_pe[base + g] = fmin(
          fmax(fmax(rho_safe * kappa_pe, 0.0), sigma_min), sigma_max);
      sigma_R[base + g] = fmin(
          fmax(fmax(rho_safe * kappa_r, 0.0), sigma_min), sigma_max);
    }
    return;
  }

  double w[kMaxMats];
  double log_ni_m[kMaxMats];
  double rho_part[kMaxMats];
  for (int m = 0; m < n_materials; ++m) {
    w[m] = raw[m] / wsum;
    rho_part[m] = rho_safe;
    if (w[m] > 0.0 && descs[m].kind != 0) {
      double rho_m = rho_safe;
      if (vol_frac != nullptr) {
        const double f = fmax(vol_frac[c * n_materials + m], 0.0);
        if (f > 1.0e-12) {
          rho_m = rho_safe * w[m] / f;
        }
      }
      rho_part[m] = rho_m;
      log_ni_m[m] = log(fmax(rho_m * descs[m].inv_A_mp, 1.0e-300));
    } else {
      log_ni_m[m] = 0.0;
    }
  }
  const double log_T = log(fmax(Te[c], temperature_floor_eV));
  // Materials.opacity_mix_rule (NUMERICS §1.1.5): 0 linear; 1 Planck linear,
  // Rosseland harmonic (1/kappa_R = sum w_m / kappa_R,m, each kappa_R,m
  // floored at 1e-20 cm^2/g); 2 the largest material opacity (2026-09-24).
  constexpr double kHarmonicKappaFloor = 1.0e-20;
  for (int g = 0; g < n_groups; ++g) {
    double kappa_pa = 0.0;
    double kappa_pe = 0.0;
    double kappa_r = 0.0;
    [[maybe_unused]] double inv_kappa_r = 0.0;
    for (int m = 0; m < n_materials; ++m) {
      if (!(w[m] > 0.0)) continue;
      const MatOpacityDesc& d = descs[m];
      double k_pa = 0.0;
      double k_pe = 0.0;
      double k_r = 0.0;
      multimat_material_kappas(d, g, rho_part[m], log_ni_m[m], log_T, Te[c], group_bounds,
                               &k_pa, &k_pe, &k_r);
      if constexpr (mix_rule == 2) {
        kappa_pa = fmax(kappa_pa, k_pa);
        kappa_pe = fmax(kappa_pe, k_pe);
        kappa_r = fmax(kappa_r, k_r);
      } else {
        kappa_pa += w[m] * k_pa;
        kappa_pe += w[m] * k_pe;
        if constexpr (mix_rule == 1) {
          inv_kappa_r += w[m] / fmax(k_r, kHarmonicKappaFloor);
        } else {
          kappa_r += w[m] * k_r;
        }
      }
    }
    if constexpr (mix_rule == 1) {
      kappa_r = (inv_kappa_r > 0.0) ? 1.0 / inv_kappa_r : 0.0;
    }
    sigma_a[base + g] = fmin(
        fmax(fmax(rho_safe * kappa_pa, 0.0), sigma_min), sigma_max);
    sigma_pe[base + g] = fmin(
        fmax(fmax(rho_safe * kappa_pe, 0.0), sigma_min), sigma_max);
    sigma_R[base + g] = fmin(
        fmax(fmax(rho_safe * kappa_r, 0.0), sigma_min), sigma_max);
  }
}

// Physical scattering of a multi-material cell for S_N: sigma_s = rho sum_m
// w_m kappa_s,m (the materials' scattering coefficients add at their partial
// densities; weights as in eval_opacity_multimat_kernel), bounded like the
// single-material constant path (rho * [kappa_floor, kappa_cap]). A cell
// whose dominant material is a table (kinds 1 and 2, which have no
// scattering, as the single-material table path) takes no floor.
__global__ void eval_scattering_multimat_kernel(
    const double* __restrict__ rho,
    const int* __restrict__ cell_material_index,
    const double* __restrict__ mass_per_material,  // [n_cells*n_materials] or nullptr
    const double* __restrict__ vol_frac,           // [n_cells*n_materials] or nullptr
    const MatOpacityDesc* __restrict__ descs,
    int n_materials,
    double kappa_floor,
    double kappa_cap,
    double* __restrict__ sigma_s,
    int n_cells,
    int n_groups) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int m_dom = min(max(cell_material_index[c], 0), n_materials - 1);
  constexpr int kMaxMats = 16;
  double raw[kMaxMats];
  double wsum = 0.0;
  for (int m = 0; m < n_materials; ++m) {
    double r = 0.0;
    if (!descs[m].is_void) {
      if (mass_per_material != nullptr) {
        r = fmax(mass_per_material[c * n_materials + m], 0.0);
      } else if (vol_frac != nullptr) {
        r = fmax(vol_frac[c * n_materials + m], 0.0);
      } else {
        r = (m == m_dom) ? 1.0 : 0.0;
      }
    }
    raw[m] = r;
    wsum += r;
  }
  double kappa_s = descs[m_dom].kappa_scatter;
  if (wsum > 0.0) {
    kappa_s = 0.0;
    for (int m = 0; m < n_materials; ++m) {
      const double w = raw[m] / wsum;
      if (w > 0.0) {
        kappa_s += w * descs[m].kappa_scatter;
      }
    }
  }
  const double rho_c = rho[c];
  const double rho_safe = (rho_c >= 0.0) ? rho_c : 0.0;
  const bool table_cell = descs[m_dom].kind == 1 || descs[m_dom].kind == 2;
  const double sigma_min = table_cell ? 0.0 : rho_safe * fmax(kappa_floor, 0.0);
  const double sigma_max = rho_safe * fmax(kappa_cap, 0.0);
  const double value = fmin(fmax(fmax(rho_safe * kappa_s, 0.0), sigma_min), sigma_max);
  const int base = c * n_groups;
  for (int g = 0; g < n_groups; ++g) {
    sigma_s[base + g] = value;
  }
}

#endif  // __CUDACC__

}  // namespace
}  // namespace tenryu::radiation
