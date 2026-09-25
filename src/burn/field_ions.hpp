#pragma once

#include <cstddef>
#include <vector>

#include "burn/burn_constants.hpp"
#include "burn/deposition.cuh"
#include "core/constants.hpp"
#include "core/macros.hpp"

namespace tenryu::burn {

// Per-ion means over the ions a fast charged fusion product slows down on
// (NUMERICS §14.3, §14.7), all fully ionized: mass number A, charge Z, Z^2
// and Z^2/A. The default is equimolar DT.
struct FieldIons {
  double A_bar = 2.51505;
  double z_bar = 1.0;
  double z2_bar = 1.0;
  double z2_over_A_bar = 0.5 / 2.0141 + 0.5 / 3.0160;
};

// One kind of ion of a mixture: weight (any scale common to the mixture,
// e.g. ions per unit mass), mass number and charge.
struct IonKind {
  double w = 0.0;
  double A = 0.0;
  double Z = 0.0;
};

// Per-ion means of a mixture. The weights are normalized before the sums,
// so equal weights of D and T give field_ions_equimolar_dt() bit for bit.
// Returns `fallback` when no kind has a positive weight.
TENRYU_HOST_DEVICE inline FieldIons field_ions_from_kinds(
    const IonKind* kinds, const int n_kinds, const FieldIons& fallback) {
  double total = 0.0;
  for (int k = 0; k < n_kinds; ++k) {
    if (kinds[k].w > 0.0 && kinds[k].A > 0.0) {
      total += kinds[k].w;
    }
  }
  if (!(total > 0.0)) {
    return fallback;
  }
  FieldIons f;
  f.A_bar = 0.0;
  f.z_bar = 0.0;
  f.z2_bar = 0.0;
  f.z2_over_A_bar = 0.0;
  for (int k = 0; k < n_kinds; ++k) {
    const IonKind& ion = kinds[k];
    if (!(ion.w > 0.0 && ion.A > 0.0)) {
      continue;
    }
    const double x = ion.w / total;
    const double z2 = ion.Z * ion.Z;
    f.A_bar += x * ion.A;
    f.z_bar += x * ion.Z;
    f.z2_bar += x * z2;
    f.z2_over_A_bar += x * (z2 / ion.A);
  }
  return f;
}

TENRYU_HOST_DEVICE inline FieldIons field_ions_equimolar_dt() {
  const IonKind dt[2] = {{1.0, species_A(kD), species_Z(kD)},
                         {1.0, species_A(kT), species_Z(kT)}};
  return field_ions_from_kinds(dt, 2, FieldIons{});
}

// Fuel mixture of the burn number fractions Burn.x_D / x_T / x_He3.
TENRYU_HOST_DEVICE inline FieldIons field_ions_from_fractions(
    const double x_D, const double x_T, const double x_He3) {
  const IonKind fuel[3] = {{x_D, species_A(kD), species_Z(kD)},
                           {x_T, species_A(kT), species_Z(kT)},
                           {x_He3, species_A(kHe3), species_Z(kHe3)}};
  return field_ions_from_kinds(fuel, 3, field_ions_equimolar_dt());
}

// Factors of the local range fit (NUMERICS §14.3) relative to equimolar DT:
// electrons per unit mass (fe) and sum of Z^2/A per unit mass (fi). Both are
// exactly 1 for field_ions_equimolar_dt().
TENRYU_HOST_DEVICE inline FraleyRangeMedium fraley_range_medium(
    const FieldIons& f) {
  const FieldIons dt = field_ions_equimolar_dt();
  FraleyRangeMedium m;
  m.fe = (f.z_bar / f.A_bar) / (dt.z_bar / dt.A_bar);
  m.fi = (f.z2_over_A_bar / f.A_bar) / (dt.z2_over_A_bar / dt.A_bar);
  return m;
}

// A material of the cell mixture. Its ions join the field when `field` is
// set: non-fuel, non-void materials (fuel ions come from the burn
// inventory, which follows depletion and ash).
struct FieldIonMaterial {
  double A = 0.0;
  double Z = 0.0;
  bool field = false;
};

// Field ions of every cell (NUMERICS §14.7): the burn inventory species
// (specific inventories Y_s [1/g], kNumSpecies per cell, cell-major; an
// empty vector means none) weighted by Y_s m_p, and the field materials
// weighted by vf_m / A_m (the inventory's convention that a material's
// mass fraction is its volume fraction). Cells with neither take
// `fallback`.
inline void cell_field_ions(const std::vector<double>& burn_Y,
                            const double* volFrac, const int n_mat,
                            const std::vector<FieldIonMaterial>& materials,
                            const int n_cells, const FieldIons& fallback,
                            std::vector<FieldIons>& out) {
  const std::size_t n = static_cast<std::size_t>(n_cells > 0 ? n_cells : 0);
  out.assign(n, fallback);
  const bool have_inventory =
      burn_Y.size() >= n * static_cast<std::size_t>(kNumSpecies);
  std::vector<IonKind> kinds(static_cast<std::size_t>(kNumSpecies) +
                             static_cast<std::size_t>(n_mat > 0 ? n_mat : 0));
  for (std::size_t c = 0; c < n; ++c) {
    int n_kinds = 0;
    if (have_inventory) {
      for (int s = 0; s < kNumSpecies; ++s) {
        kinds[static_cast<std::size_t>(n_kinds++)] =
            IonKind{burn_Y[c * static_cast<std::size_t>(kNumSpecies) +
                           static_cast<std::size_t>(s)] *
                        core::constants::proton_mass,
                    species_A(s), species_Z(s)};
      }
    }
    if (volFrac != nullptr) {
      for (int m = 0; m < n_mat; ++m) {
        const FieldIonMaterial& mat =
            materials[static_cast<std::size_t>(m)];
        if (!mat.field || !(mat.A > 0.0)) {
          continue;
        }
        const double vf =
            volFrac[c * static_cast<std::size_t>(n_mat) +
                    static_cast<std::size_t>(m)];
        if (!(vf > 0.0)) {
          continue;
        }
        kinds[static_cast<std::size_t>(n_kinds++)] =
            IonKind{vf / mat.A, mat.A, mat.Z};
      }
    }
    out[c] = field_ions_from_kinds(kinds.data(), n_kinds, fallback);
  }
}

// Per-cell field ions for the slowing-down kernels: kFieldIonCellValues
// values per cell, cell-major {A_bar, z2_bar, z2_over_A_bar} (the
// slowing-down does not use z_bar). A null pointer means the uniform value
// everywhere.
inline constexpr int kFieldIonCellValues = 3;

struct FieldIonCells {
  const double* values = nullptr;
};

inline void pack_field_ion_cells(const std::vector<FieldIons>& cells,
                                 std::vector<double>& packed) {
  packed.resize(cells.size() *
                static_cast<std::size_t>(kFieldIonCellValues));
  for (std::size_t c = 0; c < cells.size(); ++c) {
    double* const v =
        packed.data() + c * static_cast<std::size_t>(kFieldIonCellValues);
    v[0] = cells[c].A_bar;
    v[1] = cells[c].z2_bar;
    v[2] = cells[c].z2_over_A_bar;
  }
}

TENRYU_HOST_DEVICE inline FieldIons field_ions_at(const FieldIonCells& cells,
                                                  const FieldIons& uniform,
                                                  const int j) {
  if (cells.values == nullptr) {
    return uniform;
  }
  const double* const v = cells.values + kFieldIonCellValues * j;
  FieldIons f = uniform;
  f.A_bar = v[0];
  f.z2_bar = v[1];
  f.z2_over_A_bar = v[2];
  return f;
}

}  // namespace tenryu::burn
