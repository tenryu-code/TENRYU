#pragma once

// The per-cell parts of the 1D burn phase that the driver computed on the host around the device
// stage and the charged-product transport, on the device (burn_inputs_1d_gpu.cu, compiled with
// -fmad=false so that every product is rounded before it is added, as on the host): the cells'
// radial velocities from the nodes', the field ions of each cell with their local range fit
// factors (NUMERICS §14.3, §14.7), the cumulative specific burn heating and neutron count, the
// electron density of the transport, the slots that hold particles, and the deposits, heating
// rates and explicit-source dt limit after the transport. Each launches on the default stream;
// only slot_nonzero_flags_1d_device and burn_transport_finish_1d_device return values to the
// host.

#include <vector>

#include "burn/field_ions.hpp"

namespace tenryu::burn {

// The materials of the field-ion mixture (burn_cell_field_ions) in device memory:
// kFieldMaterialValues doubles per material, {A, Z, 1 when its ions join the field else 0}.
inline constexpr int kFieldMaterialValues = 3;

// The packed form of `materials` for FieldIonSetup::materials.
void pack_field_ion_materials(const std::vector<FieldIonMaterial>& materials,
                              std::vector<double>& packed);

// The materials of the field-ion mixture and the fuel fallback.
struct FieldIonSetup {
  int n_mat = 0;
  const double* materials = nullptr;  // device, kFieldMaterialValues * n_mat
  FieldIons fallback;
};

// v_cell[c] = 0.5 (v_node[c] + v_node[c + 1]).
void cell_velocities_1d_device(const double* v_node, int n_cells, double* v_cell);

// The field ions of every cell (cell_field_ions): the inventory species of burn_y (kNumSpecies per
// cell; nullptr: none) weighted by Y_s m_p and the field materials weighted by vf_m / A_m
// (volFrac: setup.n_mat per cell; nullptr: none); the fallback for a cell with neither. fe[c],
// fi[c] = their fraley_range_medium; field_cells (nullptr: not written) =
// kFieldIonCellValues per cell, as pack_field_ion_cells packs them.
void range_medium_1d_device(const double* burn_y, const double* volFrac, const FieldIonSetup& setup,
                            int n_cells, double* fe, double* fi, double* field_cells = nullptr);

// eps_cum[c] += (dE_e[c] + dE_i[c]) / (rho[c] vol[c]) where rho vol > 1e-30 (eps_cum nullptr: not
// updated), and (when both are given) neutron_cum[c] += births[c].
void burn_cumulative_1d_device(const double* rho, const double* vol, const double* dE_e,
                               const double* dE_i, const double* births, int n_cells,
                               double* eps_cum, double* neutron_cum);

// ne[c] = zbar[c] rho[c] / (A_eff[c] m_p) where A_eff[c] m_p > 0, else 0.
void electron_density_1d_device(const double* zbar, const double* rho, const double* A_eff,
                                int n_cells, double* ne);

// Whether each product slot holds a nonzero entry (a NaN counts): flags[s] for the birth sources
// S_birth (6 slots of n_cells), flags[6 + s] for the in-flight spectra N (6 slots of
// n_groups * n_cells; nullptr: 0). One copy to the host.
void slot_nonzero_flags_1d_device(const double* S_birth, const double* N, int n_cells,
                                  int n_groups, int flags[12]);

// The deposits of the charged-product transport and their bookkeeping (the driver's loops after
// the transport): dE = dep + nh where nh is given (nh_e, nh_i: the reaction stage's neutron
// heating; dep_e, dep_i are updated); eps_cum += (dE_e + dE_i) / (rho vol) where rho vol > 1e-30;
// Q = dE / (vol dt) where vol > 0 and dt > 0, else 0; and the explicit-source limit
// min over the cells with P = (dE_e + dE_i) / dt > 0 and e = rho vol max(ee + ei, 0) > 0 of
// explicit_source_limit e / P (infinity when none). Returns {sum of dE_e, sum of dE_i, limit},
// the sums in cell order.
struct BurnTransportTotals {
  double dep_e = 0.0;
  double dep_i = 0.0;
  double dt_limit = 0.0;
};
BurnTransportTotals burn_transport_finish_1d_device(double* dep_e, double* dep_i,
                                                    const double* nh_e, const double* nh_i,
                                                    const double* rho, const double* vol,
                                                    const double* ee, const double* ei,
                                                    int n_cells, double dt,
                                                    double explicit_source_limit,
                                                    double* eps_cum, double* Q_e, double* Q_i);

}  // namespace tenryu::burn
