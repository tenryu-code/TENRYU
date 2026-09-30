#pragma once

#include <cstdint>

#include "radiation/holo_geometry.hpp"

namespace tenryu::materials {
struct EOSTable;
}  // namespace tenryu::materials

namespace tenryu::radiation {

class PlanckTable;

enum class HoloLOMaterialModel {
  IdealGas,
  TableEOS,
};

enum class HoloLOFailureStage {
  None,
  SpatialSolve,
  SourceSolve,
  Conservation,
};

struct HoloLOInputs {
  HoloGeometryDimension dimension = HoloGeometryDimension::Spherical1D;
  double* E_lo = nullptr;           // [n_cells * G], in/out [erg/cm^3]
  double* ee = nullptr;             // [n_cells], in/out electron specific energy [erg/g]
  double* Te = nullptr;             // [n_cells], in/out [eV]
  double* Pe = nullptr;             // [n_cells], out [erg/cm^3]
  const double* sigma_P = nullptr;  // [n_cells * G], [1/cm]
  const double* sigma_R = nullptr;  // [n_cells * G], [1/cm]
  const double* chi = nullptr;      // [n_cells * G], Eddington factor f_E = P_rr/E, optional
  double* F_lo = nullptr;           // [(n_cells + 1) * G], in/out face flux [erg/(cm^2 s)], optional
  const double* rho = nullptr;      // [n_cells], [g/cm^3]
  const double* mass = nullptr;     // [n_cells], optional [g]
  const double* vol = nullptr;      // [n_cells], [cm^3]
  const double* node_r = nullptr;   // [n_cells + 1], [cm]
  const std::uint8_t* lo_coupled = nullptr; // [n_cells], optional core material commit mask
  const std::uint8_t* cell_active = nullptr; // [n_cells], optional LO patch solve mask
  const double* lo_weight = nullptr; // [n_cells], optional HOLO blend/patch weight
  const double* consistency_source = nullptr; // [n_cells * G], optional HOLO source [erg/s]
  double consistency_alpha = 1.0;    // relaxation coefficient for consistency source
  const double* face_current = nullptr; // [(n_cells + 1) * G], optional signed step tally [erg]
  double face_current_dt = 0.0;     // [s], normalization interval for face_current
  bool has_physical_outer_vacuum = true;
  double* rad_dep_lo = nullptr;     // [n_cells * G], optional accumulated [erg]
  double* rad_emit_lo = nullptr;    // [n_cells * G], optional accumulated [erg]
  double* matter_delta_lo_cell = nullptr; // [n_cells], optional LO candidate source [erg]
  const double* cv_e = nullptr;     // [n_cells], optional mass cv [erg/(g*eV)]
  const materials::EOSTable* electron_eos = nullptr;
  const PlanckTable* planck = nullptr;
  int n_cells = 0;
  int n_groups = 1;
  double dt = 0.0;
  double cv_e_const = 0.0;
  double pressure_gamma_minus_one = 2.0 / 3.0;
  double temperature_floor_eV = 1.0e-3;
  HoloLOMaterialModel material_model = HoloLOMaterialModel::IdealGas;
};

struct HoloLOResult {
  double matter_delta = 0.0;        // [erg], positive when matter gains energy
  double rad_delta = 0.0;           // [erg], positive when LO radiation gains energy
  double boundary_E_in = 0.0;       // [erg]
  double boundary_E_out = 0.0;      // [erg]
  double boundary_limited_E = 0.0;  // [erg], unused without internal face-current BCs
  double conservation_error = 0.0;  // [erg], global LO source + radiation - boundary net
  int solver_iterations = 0;
  int failures = 0;
  HoloLOFailureStage failure_stage = HoloLOFailureStage::None;
  int failure_cell = -1;
  int failure_group = -1;
  double failure_E_sum = 0.0;       // [erg/cm^3], group-summed E in failed cell
  double failure_T = 0.0;           // [eV]
  double failure_residual = 0.0;    // [erg] for source/conservation failures
};

[[nodiscard]] const char* holo_lo_failure_stage_name(HoloLOFailureStage stage);

HoloLOResult solve_holo_lo_1d_cpu(const HoloLOInputs& in);
HoloLOResult solve_holo_lo_1d_cpu(const HoloLOInputs& in, bool use_quasidiffusion);

}  // namespace tenryu::radiation
