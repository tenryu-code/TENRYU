#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "core/config.hpp"
#include "core/state.hpp"
#include "diagnostics/radial_fourier_audit_v2_mode.hpp"

namespace tenryu::diagnostics {

constexpr int kRadialFourierFieldCount = 6;

enum class RadialFourierStageId : std::uint8_t {
  HydroLag = 0,
  ArtificialViscosity = 1,
  Winslow = 2,
  Remap = 3,
  BoundaryFill = 4,
  FldSolve = 5,
  NewtonSource = 6,
  Positivity = 7,
  DtController = 8,
};

enum class RadialFourierStagePhase : std::uint8_t {
  Before = 0,
  After = 1,
};

enum class RadialFourierFieldId : std::uint8_t {
  Rho = 0,
  Te = 1,
  Ti = 2,
  Ur = 3,
  Uz = 4,
  Erad = 5,
};

enum class RadialFourierComplexFieldId : std::uint8_t {
  Rho = 0,
  M = 1,
  V = 2,
  MOverV = 3,
  Pr = 4,
  Pz = 5,
  Ur = 6,
  Uz = 7,
  Ee = 8,
  Ei = 9,
  Erad = 10,
  Te = 11,
  Ti = 12,
  Xr = 13,
  Xz = 14,
  Ar = 15,
  Az = 16,
  DVSwpt = 17,
  Qvisc = 18,
  LambdaFld = 19,
  RFld = 20,
  KappaEff = 21,
  FFleck = 22,
  NewtonIters = 23,
  NewtonResidual = 24,
  NotAvailable = 255,
};

enum class FldSubstageAuditSubstageId : std::uint8_t {
  EradBeforeFldEntry = 0,
  OpacityBeforeDBuild = 1,
  DCellAfterDBuild = 2,
  StateSupplyBoundaryCoeff = 3,
  StateSupplyDtCoeff = 4,
  StateSupplyBoundarySource = 5,
  BoundaryDiagContribution = 6,
  RhsAfterAssembly = 7,
  CgTrueResidual = 8,
  EradAfterCgSolve = 9,
  EradAfterNewtonSource = 10,
};

enum class FldSubstageAuditFieldId : std::uint8_t {
  Erad = 0,
  Rho = 1,
  SigmaR = 2,
  DCell = 3,
  StateSupplyBoundaryCoeff = 4,
  StateSupplyDtCoeff = 5,
  StateSupplyBoundarySource = 6,
  BoundaryDiagContribution = 7,
  Rhs = 8,
  CgTrueResidual = 9,
};

enum class FldSubstageAuditFieldLayout : std::uint8_t {
  CellScalar = 0,
  CellMajorGroup = 1,
  GroupMajor = 2,
};

enum class FldSubstageAuditNormalization : std::uint8_t {
  MeanSubtractedUnweightedRawSum = 0,
};

#if TENRYU_RFA_V2_MODE_IS_OFF
inline constexpr RadialFourierAuditV2Mode kRadialFourierAuditV2Mode =
    RadialFourierAuditV2Mode::Off;
#elif TENRYU_RFA_V2_MODE_IS_STUB
inline constexpr RadialFourierAuditV2Mode kRadialFourierAuditV2Mode =
    RadialFourierAuditV2Mode::Stub;
#elif TENRYU_RFA_V2_MODE_IS_DUMMY_BUFFER
inline constexpr RadialFourierAuditV2Mode kRadialFourierAuditV2Mode =
    RadialFourierAuditV2Mode::DummyBuffer;
#else
inline constexpr RadialFourierAuditV2Mode kRadialFourierAuditV2Mode =
    RadialFourierAuditV2Mode::Full;
#endif

struct RadialFourierFieldMaximum {
  double A_max = 0.0;
  int m_max = 0;
  int j_max = 0;
};

struct RadialFourierResult {
  bool valid = false;
  int nr = 0;
  int nz = 0;
  int n_modes = 0;
  std::array<RadialFourierFieldMaximum, kRadialFourierFieldCount> fields{};
};

struct RadialFourierAuditRecord {
  bool valid = false;
  std::uint64_t cycle = 0;
  double t_s = 0.0;
  std::uint8_t stage_id = 0;
  std::uint8_t stage_phase = 0;
  std::array<RadialFourierFieldMaximum, kRadialFourierFieldCount> fields{};
};

struct RadialFourierComplexCoeff {
  std::uint8_t field_id = 0;
  int m = 0;
  int j = 0;
  double mean_unw = 0.0;
  double cre_unw = 0.0;
  double cim_unw = 0.0;
  double amp_unw = 0.0;
  double phase_unw = 0.0;
  double mean_vol = 0.0;
  double cre_vol = 0.0;
  double cim_vol = 0.0;
  double amp_vol = 0.0;
  double phase_vol = 0.0;
  double q_min_j = 0.0;
  double q_max_j = 0.0;
  double wsum_vol = 0.0;
};

struct RadialFourierComplexResult {
  bool valid = false;
  int nr = 0;
  int nz = 0;
  std::vector<RadialFourierComplexCoeff> coeffs;
};

struct RadialFourierComplexAuditRecord {
  bool valid = false;
  std::uint64_t cycle = 0;
  double t_s = 0.0;
  double dt_cycle = 0.0;
  std::uint8_t stage_id = 0;
  std::uint8_t stage_phase = 0;
  std::vector<RadialFourierComplexCoeff> coeffs;
};

struct FldSubstageAuditRecord {
  bool valid = false;
  std::uint64_t cycle = 0;
  double t_s = 0.0;
  double dt_cycle = 0.0;
  std::uint8_t substage_id = 0;
  std::uint8_t field_id = 0;
  std::uint8_t normalization_kind = 0;
  int m = 0;
  int j = 0;
  int group = 0;
  int outer_iter = 0;
  int nr = 0;
  int nz = 0;
  double cre = 0.0;
  double cim = 0.0;
  double amplitude = 0.0;
  double phase = 0.0;
  double mean = 0.0;
  double q_min_j = 0.0;
  double q_max_j = 0.0;
  double normalization = 0.0;
  double solver_residual_l2_rel = 0.0;
  double solver_residual_max = 0.0;
};

const char* radial_fourier_stage_name(RadialFourierStageId stage);
const char* radial_fourier_field_name(RadialFourierFieldId field);
const char* radial_fourier_complex_field_name(RadialFourierComplexFieldId field);

std::vector<FldSubstageAuditRecord> compute_fld_substage_fixed_mode_audit(
    const double* device_field,
    FldSubstageAuditFieldLayout layout,
    int nr,
    int nz,
    int n_groups,
    FldSubstageAuditSubstageId substage,
    FldSubstageAuditFieldId field,
    int outer_iter,
    const std::vector<int>& m_targets,
    const std::vector<int>& j_targets,
    double solver_residual_l2_rel = 0.0,
    double solver_residual_max = 0.0);

RadialFourierResult compute_radial_fourier_audit(
    const core::State& state,
    const core::Config& cfg);

RadialFourierComplexResult compute_radial_fourier_complex_audit(
    const core::State& state,
    const core::Config& cfg);

RadialFourierAuditRecord make_radial_fourier_audit_record(
    const RadialFourierResult& result,
    std::uint64_t cycle,
    double t_s,
    RadialFourierStageId stage,
    RadialFourierStagePhase phase);

RadialFourierComplexAuditRecord make_radial_fourier_complex_audit_record(
    const RadialFourierComplexResult& result,
    std::uint64_t cycle,
    double t_s,
    double dt_cycle,
    RadialFourierStageId stage,
    RadialFourierStagePhase phase);

}  // namespace tenryu::diagnostics
