#include "radiation/fld_2d_rz_gpu.cuh"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "core/constants.hpp"
#include "core/device_scratch.hpp"
#include "core/error.hpp"
#include "core/kernel_guard.hpp"
#include "core/namelist/errors.hpp"
#include "materials/eos_device_table.hpp"
#include "materials/eos_table.hpp"
#include "materials/ionmix_reader.cuh"
#include "materials/ionmix_reader.hpp"
#include "materials/opacity.cuh"
#include "materials/tmat_reader.hpp"
#include "parallel/comm_buffers.hpp"
#include "parallel/halo_exchange.hpp"
#include "parallel/partition.hpp"
#include "parallel/reduction.hpp"
#include "radiation/amgx_solver.hpp"
#include "radiation/fld_substage_audit_drain.hpp"
#include "radiation/fleck.cuh"
#include "radiation/group_structure.hpp"
#include "radiation/groups.cuh"
#include "radiation/nlte_coeffs.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr int kBlock = 256;
constexpr int kDotSingleBlockMaxBlocks = 16;
constexpr int kCsrEntriesPerRow = 5;
constexpr int kFldBcReflect = 0;
constexpr int kFldBcVacuum = 1;
constexpr int kFldBcMarshak = 2;
constexpr int kFldBcStateSupply = 3;
constexpr int kStateSupplyBoundaryPolicyLocalDCurrent = 0;
constexpr int kStateSupplyBoundaryPolicyHarmonicGhostDTest = 1;
constexpr int kStateSupplyBoundaryPolicyRadialMeanDTest = 2;
// Phase 2a-1.5d: Minimum cell-center distance for FLD interior face
// coupling. Below this, the mesh is considered degenerate (pinch or
// inversion); the face contributes no diffusion flux. 1e-12 cm is well
// below typical ICF mesh scales (1e-4 cm cell size) and represents a
// sub-atomic length where Lagrangian mesh has clearly lost physical
// meaning.
constexpr double kFldFaceDistMin = 1.0e-12;  // cm

enum class FldPrecondMode { Diagonal, ZLine, RGmg };

inline int dot_block_count(const int n) {
  return (n <= 0) ? 0 : (n + kBlock - 1) / kBlock;
}

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

inline void cusparse_check(const cusparseStatus_t status, const char* message) {
  TENRYU_ASSERT(status == CUSPARSE_STATUS_SUCCESS, message);
}

struct RgmgLevel {
  parallel::DeviceArray diag;
  parallel::DeviceArray ar_left;
  parallel::DeviceArray ar_right;
  parallel::DeviceArray az_m;
  parallel::DeviceArray az_p;
  parallel::DeviceArray b;
  parallel::DeviceArray x;
  parallel::DeviceArray res;
  parallel::DeviceArray corr;
  int nr = 0;
  int nz = 0;
  int n_groups = 0;
};

struct FldCgGraphKey {
  int n_rows = 0;
  int nnz = 0;
  const double* values = nullptr;
  const int* row_offsets = nullptr;
  const int* col_indices = nullptr;
  const double* x = nullptr;
  const double* diag_inv = nullptr;
  const double* r = nullptr;
  const double* z = nullptr;
  const double* p = nullptr;
  const double* Ap = nullptr;
  const double* pingpong = nullptr;
  const double* pAp = nullptr;
  const void* status = nullptr;
  const int* iter_counter = nullptr;
  const void* spmv_buffer = nullptr;
  const double* dot_partials = nullptr;
  int precond_mode_id = 0;  // 0=diag, 1=zline, 2=rgmg
  const void* zline_dl = nullptr;
  const void* zline_d = nullptr;
  const void* zline_du = nullptr;
  const void* gtsv_buffer = nullptr;
  std::vector<const void*> precond_ptrs;  // rgmg: every level array baked in
  bool operator==(const FldCgGraphKey&) const = default;
};

struct Fld2DWorkspace {
  cusparseHandle_t handle = nullptr;

  parallel::DeviceArray d_row_offsets;
  parallel::DeviceArray d_col_indices;
  parallel::DeviceArray d_values;
  parallel::DeviceArray d_rhs;
  parallel::DeviceArray d_diag_inv;
  parallel::DeviceArray d_x;

  parallel::DeviceArray d_r;
  parallel::DeviceArray d_z;
  parallel::DeviceArray d_p;
  parallel::DeviceArray d_Ap;
  parallel::DeviceArray d_spmv;
  parallel::DeviceArray d_scalar;
  parallel::DeviceArray d_dot_partials;
  parallel::DeviceArray d_line_dl;
  parallel::DeviceArray d_line_d;
  parallel::DeviceArray d_line_du;
  parallel::DeviceArray d_gtsv2_buffer;
  parallel::DeviceArray d_cg_rz_pingpong;
  parallel::DeviceArray d_cg_pAp;
  parallel::DeviceArray d_cg_status;
  parallel::DeviceArray d_diag_scalars;
  parallel::DeviceArray d_diag_ints;
  parallel::DeviceArray d_diag_record;
  parallel::DeviceArray d_rogue_records;
  parallel::DeviceArray d_audit_m_targets;
  parallel::DeviceArray d_audit_j_targets;
  parallel::DeviceArray d_audit_records;
  parallel::DeviceArray d_audit_boundary_coeff;
  parallel::DeviceArray d_audit_boundary_dt_coeff;
  parallel::DeviceArray d_audit_boundary_source;
  parallel::DeviceArray d_audit_boundary_diag;
  // Anderson acceleration history (outer_accel="anderson"): ring of the last
  // m+1 outer iterates u_j (Te inputs) and residuals f_j = G(u_j) - u_j.
  parallel::DeviceArray d_aa_u[5];
  parallel::DeviceArray d_aa_f[5];
  std::vector<RgmgLevel> rgmg_levels;
  parallel::DeviceArray rgmg_gtsv_buffer;
  int rgmg_n_levels = 0;

  cusparseSpMatDescr_t mat = nullptr;
  cusparseDnVecDescr_t vec_x = nullptr;
  cusparseDnVecDescr_t vec_p = nullptr;
  cusparseDnVecDescr_t vec_Ap = nullptr;
  // §6q.9 stage 2: per-group owned-row CSR views (row = g*n_cells + c).
  std::vector<cusparseSpMatDescr_t> mat_owned;
  std::vector<cusparseDnVecDescr_t> vec_Ap_owned;
  parallel::DeviceArray d_owned_row_offsets;
  int owned_view_c_begin = -1;
  int owned_view_c_end = -1;

  int desc_n_rows = 0;
  int desc_nnz = 0;
  const int* desc_row_offsets_ptr = nullptr;
  const int* desc_col_indices_ptr = nullptr;
  const double* desc_values_ptr = nullptr;
  const double* desc_x_ptr = nullptr;
  const double* desc_p_ptr = nullptr;
  const double* desc_Ap_ptr = nullptr;

  // W1 (2026-07-09): CUDA-graph replay of the CG inter-check block.
  // cg_stream is a default-flags (blocking) stream: legacy default-stream
  // semantics order the replayed graph against all surrounding
  // default-stream work, so the kernel execution order is identical to
  // the eager loop.
  cudaStream_t cg_stream = nullptr;
  cudaGraphExec_t cg_graph_exec = nullptr;
  bool cg_graph_capture_failed = false;
  parallel::DeviceArray d_cg_iter;
  FldCgGraphKey cg_graph_key;

  // Option-C MPI context for the advance call in flight (set at the top of
  // advance_radiation_step_fld_2d_rz; the part/bufs pointers are owned by
  // the driver and valid only for that call). n_ranks == 1 keeps every
  // serial code path byte-identical; direct solver calls from tests never
  // populate this, so they run the serial paths.
  const parallel::PartitionInfo* mpi_part = nullptr;
  parallel::CommBuffers* mpi_bufs = nullptr;
  int mpi_n_ranks = 1;
  int mpi_c_begin = 0;
  int mpi_c_end = 0;    // owned cells [c_begin, c_end)
  int mpi_n_cells = 0;  // global cell count (per-group row stride)
  int mpi_n_groups = 0;

  ~Fld2DWorkspace() {
    if (cg_graph_exec != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(cg_graph_exec));
      cg_graph_exec = nullptr;
    }
    if (cg_stream != nullptr) {
      static_cast<void>(cudaStreamDestroy(cg_stream));
      cg_stream = nullptr;
    }
    if (vec_Ap != nullptr) {
      static_cast<void>(cusparseDestroyDnVec(vec_Ap));
      vec_Ap = nullptr;
    }
    if (vec_p != nullptr) {
      static_cast<void>(cusparseDestroyDnVec(vec_p));
      vec_p = nullptr;
    }
    if (vec_x != nullptr) {
      static_cast<void>(cusparseDestroyDnVec(vec_x));
      vec_x = nullptr;
    }
    if (mat != nullptr) {
      static_cast<void>(cusparseDestroySpMat(mat));
      mat = nullptr;
    }
    if (handle != nullptr) {
      static_cast<void>(cusparseDestroy(handle));
      handle = nullptr;
    }
  }
};

Fld2DWorkspace& fld_2d_workspace() {
  static Fld2DWorkspace ws;
  return ws;
}

double* ensure_dot_partials(Fld2DWorkspace& ws, const int nblocks) {
  const std::size_t bytes =
      static_cast<std::size_t>(nblocks) * sizeof(double);
  if (ws.d_dot_partials.capacity < bytes) {
    ws.d_dot_partials.resize(bytes);
  }
  return ws.d_dot_partials.as<double>();
}

// TENRYU_FLD_CG_NO_GRAPH=1 forces the eager CG loop (the pre-W1 code
// path, byte-identical); default (unset/0) enables graph replay for the
// Diagonal-preconditioner CG.
bool fld_cg_graph_disabled() {
  static const bool disabled = [] {
    const char* v = std::getenv("TENRYU_FLD_CG_NO_GRAPH");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  return disabled;
}

int fld_audit_reduction_thread_count(const int nr) {
  int threads = 1;
  while (threads < nr && threads < 1024) {
    threads <<= 1;
  }
  return threads;
}

struct CgDiagnostics {
  double min_pAp_value = std::numeric_limits<double>::infinity();
  int count_nonpos_pAp = 0;
  int count_nonfinite_alpha_beta_rz = 0;
  double recurrent_resid_first_check = 0.0;
  double recurrent_resid_last_check = 0.0;
  double cap_exit_true_resid_rel = 0.0;
  double cap_exit_true_resid_l2_abs = 0.0;
  int cap_exit_true_resid_group = -1;
  bool cap_exit_unconverged = false;
  int iters_executed = 0;
};

struct PostPublishDiagnostics {
  double true_residual_l2_rel = 0.0;
  double true_residual_l2_abs = 0.0;
  double true_residual_max = 0.0;
  double E_solver_abs = 0.0;
  int true_residual_max_group = -1;
};

PostPublishDiagnostics compute_post_publish_solver_diagnostics(
    const double* rhs,
    const double* x_pub,
    int n_rows,
    int n_cells = 0,
    int n_groups = 0);

struct CgDeviceStatus {
  double min_pAp;
  int breakdown_iter;
  int count_nonpos_pAp;
  int count_nonfinite;
};

struct RogueRecord {
  double value = 0.0;
  int cell_idx = -1;
  int group_idx = -1;
};

struct FldFaceSymmetryRecord {
  double value = 0.0;
  double c_lr = 0.0;
  double c_rl = 0.0;
  double energy_delta = 0.0;
  int cell_l = -1;
  int cell_r = -1;
  int direction = -1;
};

struct FldPerRowDefectRecord {
  double value = 0.0;
  double row_defect = 0.0;
  double V_op = 0.0;
  double V_state = 0.0;
  int cell_idx = -1;
  int group_idx = -1;
};

struct FldTraceRecord {
  int cell_matched;
  int i;
  int j;
  int group;
  double Te_in;
  double E_old_in;
  double sigma_pa;
  double sigma_removal;
  double B_T;
  double eta;
  double fleck_f;
  double D_cell;
  double rhs_V_E_old;
  double rhs_dt_V_f_eta;
  double rhs_dt_V_one_minus_f_csE;
  double rhs_boundary;
  double rhs_total;
  double diag_V;
  double diag_csigV;
  double diag_face_total;
  double diag_total;
  double offdiag_iL;
  double offdiag_iR;
  double offdiag_jB;
  double offdiag_jT;
  double x_raw;
  double x_pub;
  double clamp_delta;
  double Arow_x;
  double Arow_x_minus_rhs;
  double xL;
  double xR;
  double xB;
  double xT;
  double rad_E_delta;
};
static_assert(sizeof(FldTraceRecord) % 8 == 0, "alignment check");

struct FldTraceConfig {
  int cell = -1;
  int group = 0;
  int from_step = 0;
  int num_steps = INT_MAX;
};

struct FldRowIdentityDiagnostics {
  double sum_csr_residual = 0.0;
  double sum_matrix_formula_residual = 0.0;
  double sum_tally_formula_residual = 0.0;
  double sum_face_div = 0.0;
  double sum_abs_face_div = 0.0;
  double sum_emit_kernel = 0.0;
  double sum_dep_kernel = 0.0;
  double sum_emit_formula = 0.0;
  double sum_dep_formula = 0.0;
  RogueRecord max_delta_csr_vs_matrix;
  RogueRecord max_delta_matrix_vs_tally;
  RogueRecord max_delta_emit_kernel_vs_formula;
  RogueRecord max_delta_dep_kernel_vs_formula;
};

struct EscapeBreakdownTotals {
  double total_outer_r = 0.0;
  double total_z_bottom = 0.0;
  double total_z_top = 0.0;
  double total_vacuum_outer_r = 0.0;
  double total_vacuum_z_bottom = 0.0;
  double total_vacuum_z_top = 0.0;
  double sum_signed_delta_outer_r = 0.0;
  double sum_signed_delta_z_bottom = 0.0;
  double sum_signed_delta_z_top = 0.0;
  double sum_abs_delta = 0.0;
  int count_delta_gt_1 = 0;
  int count_delta_gt_10 = 0;
  int count_delta_gt_100 = 0;
  double max_per_cell_outer_r = 0.0;
  int max_cell_outer_r_i = -1;
  int max_cell_outer_r_j = -1;
  double max_per_cell_z_bottom = 0.0;
  int max_cell_z_bottom_i = -1;
  int max_cell_z_bottom_j = -1;
  double max_per_cell_z_top = 0.0;
  int max_cell_z_top_i = -1;
  int max_cell_z_top_j = -1;
  double max_boundary_diag_escape_delta = 0.0;
  int max_boundary_diag_escape_delta_i = -1;
  int max_boundary_diag_escape_delta_j = -1;
  int max_boundary_diag_escape_delta_g = -1;
  double max_pos_delta = 0.0;
  int max_pos_delta_i = -1;
  int max_pos_delta_j = -1;
  double max_neg_delta = 0.0;
  int max_neg_delta_i = -1;
  int max_neg_delta_j = -1;
  double csr_diag_value = 0.0;
  double V_op = 0.0;
  double dt_c_sigma_V = 0.0;
  double interior_diag_sum = 0.0;
  double csr_boundary_diag = 0.0;
  double formula_boundary_coef = 0.0;
  double rad_E_at_cell = 0.0;
  double sigma_a_at_cell = 0.0;
  double rho_at_cell = 0.0;
  double D_cell_at_cell = 0.0;
};

struct EscapeBreakdownComponentRecord {
  double csr_diag_value = 0.0;
  double V_op = 0.0;
  double dt_c_sigma_V = 0.0;
  double interior_diag_sum = 0.0;
  double csr_boundary_diag = 0.0;
  double formula_boundary_coef = 0.0;
  double rad_E_at_cell = 0.0;
  double sigma_a_at_cell = 0.0;
  double rho_at_cell = 0.0;
  double D_cell_at_cell = 0.0;
};

struct FldFaceTraceSideRecord {
  int valid = 0;
  int side = -1;
  int cell_i = -1;
  int cell_j = -1;
  int neighbor_i = -1;
  int neighbor_j = -1;
  int group = -1;
  int csr_col = -1;
  int coef_finite = 0;
  int coef_fallback = 0;
  int csr_mismatch = 0;
  int D_self_finite = 0;
  int D_neighbor_finite = 0;
  int D_face_finite = 0;
  double D_self = 0.0;
  double D_neighbor = 0.0;
  double D_face = 0.0;
  double area_face = 0.0;
  double dist_face = 0.0;
  double coef = 0.0;
  double csr_value = 0.0;
  double expected_csr_value = 0.0;
  double self_rho = 0.0;
  double self_sigma = 0.0;
  double self_rad_E = 0.0;
  double neighbor_rho = 0.0;
  double neighbor_sigma = 0.0;
  double neighbor_rad_E = 0.0;
};
static_assert(sizeof(FldFaceTraceSideRecord) % 8 == 0, "alignment check");

struct FldFaceGlobalMaxRecord {
  double value = 0.0;
  double D_self = 0.0;
  double D_neighbor = 0.0;
  double D_face = 0.0;
  double area_face = 0.0;
  double dist_face = 0.0;
  double area_dist_ratio = 0.0;
  double coef = 0.0;
  int cell_i = -1;
  int cell_j = -1;
  int neighbor_i = -1;
  int neighbor_j = -1;
  int group = -1;
  int side = -1;
};
static_assert(sizeof(FldFaceGlobalMaxRecord) % 8 == 0, "alignment check");

struct FldFaceGlobalMaxDiagnostics {
  RogueRecord max_D_cell;
  FldFaceGlobalMaxRecord max_D_face;
  FldFaceGlobalMaxRecord max_coef;
  FldFaceGlobalMaxRecord max_area_dist_ratio;
};

int parse_env_int_once(const char* name, const int fallback) {
  const char* text = std::getenv(name);
  if (text == nullptr || text[0] == '\0') {
    return fallback;
  }
  char* end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (end == text) {
    return fallback;
  }
  if (value < static_cast<long>(INT_MIN)) {
    return INT_MIN;
  }
  if (value > static_cast<long>(INT_MAX)) {
    return INT_MAX;
  }
  return static_cast<int>(value);
}

const FldTraceConfig& fld_trace_config() {
  static const FldTraceConfig config{
      parse_env_int_once("TENRYU_FLD_TRACE_CELL", -1),
      parse_env_int_once("TENRYU_FLD_TRACE_GROUP", 0),
      parse_env_int_once("TENRYU_FLD_TRACE_FROM_STEP", 0),
      parse_env_int_once("TENRYU_FLD_TRACE_NUM_STEPS", INT_MAX)};
  return config;
}

int fld_row_identity_step() {
  static const int target_step =
      parse_env_int_once("TENRYU_FLD_ROW_IDENTITY_STEP", INT_MAX);
  return target_step;
}

int fld_escape_breakdown_step() {
  static const int target_step =
      parse_env_int_once("TENRYU_FLD_ESCAPE_BREAKDOWN_STEP", INT_MAX);
  return target_step;
}

int fld_face_trace_cell() {
  static const int target_cell =
      parse_env_int_once("TENRYU_FLD_FACE_TRACE_CELL", -1);
  return target_cell;
}

int fld_face_trace_step() {
  static const int target_step =
      parse_env_int_once("TENRYU_FLD_FACE_TRACE_STEP", INT_MAX);
  return target_step;
}

bool fld_trace_step_active(const FldTraceConfig& trace,
                           const bool verbose,
                           const int step,
                           const int n_cells,
                           const int n_groups) {
  if (!verbose || trace.cell < 0 || trace.cell >= n_cells ||
      trace.group < 0 || trace.group >= n_groups || trace.num_steps <= 0) {
    return false;
  }
  const long long rel =
      static_cast<long long>(step) - static_cast<long long>(trace.from_step);
  return rel >= 0 && rel < static_cast<long long>(trace.num_steps);
}

constexpr int kRogueMaxAbsXRaw = 0;
constexpr int kRogueMinXRaw = 1;
constexpr int kRogueMaxRadE = 2;
constexpr int kRogueMaxRTrue = 3;
constexpr int kRogueMaxESolverRow = 4;
constexpr int kNumRogueRecords = 5;

__host__ __device__ inline double finite_or_zero(const double x) {
  return isfinite(x) ? x : 0.0;
}

__host__ __device__ inline double nonnegative_finite(const double x) {
  return isfinite(x) ? fmax(x, 0.0) : 0.0;
}

__host__ __device__ inline double positive_or_floor(const double x,
                                                    const double floor_value) {
  const double xf = finite_or_zero(x);
  return (xf > floor_value) ? xf : floor_value;
}

__host__ __device__ inline int cell_index(const int i, const int j, const int nz) {
  return i * nz + j;
}

__host__ __device__ inline double fld_boundary_leakage_coeff(
    const int bc) {
  if (bc == kFldBcVacuum) {
    return 0.5 * core::constants::c_light;
  }
  if (bc == kFldBcMarshak) {
    return 0.25 * core::constants::c_light;
  }
  if (bc == kFldBcStateSupply) {
    return 0.0;
  }
  return 0.0;
}

int fld_boundary_code(const std::string& value) {
  if (value == "reflect") {
    return kFldBcReflect;
  }
  if (value == "vacuum") {
    return kFldBcVacuum;
  }
  if (value == "marshak") {
    return kFldBcMarshak;
  }
  if (value == "state_supply") {
    return kFldBcStateSupply;
  }
  TENRYU_ASSERT(false, "unsupported FLD boundary type");
  return kFldBcReflect;
}

__host__ __device__ inline int node_index(const int i, const int j, const int nz) {
  return i * (nz + 1) + j;
}

// Surface area of a straight RZ mesh edge revolved around the symmetry axis.
__host__ __device__ inline double fld_edge_area_rz(const double r0,
                                                   const double r1,
                                                   const double z0,
                                                   const double z1) {
  const double r0n = fmax(finite_or_zero(r0), 0.0);
  const double r1n = fmax(finite_or_zero(r1), 0.0);
  const double dr = r1n - r0n;
  const double dz_e = finite_or_zero(z1) - finite_or_zero(z0);
  const double slant = hypot(dr, dz_e);
  return kPi * (r0n + r1n) * slant;
}

__device__ inline void atomic_max_nonnegative_double(double* address,
                                                     const double value) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val = __double_as_longlong(value);
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      break;
    }
    assumed = old;
  }
}

__device__ inline void atomic_min_double(double* address,
                                         const double value) {
  if (!isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) > value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      break;
    }
    assumed = old;
  }
}

__device__ inline void atomic_max_record(RogueRecord* records,
                                         const int slot,
                                         const double value,
                                         const int cell_idx,
                                         const int group_idx) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  RogueRecord* record = records + slot;
  auto* ull = reinterpret_cast<unsigned long long*>(&record->value);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      record->cell_idx = cell_idx;
      record->group_idx = group_idx;
      break;
    }
    assumed = old;
  }
}

__device__ inline void atomic_max_cell_record(RogueRecord* record,
                                              const double value,
                                              const int cell_idx) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(&record->value);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      record->cell_idx = cell_idx;
      record->group_idx = -1;
      break;
    }
    assumed = old;
  }
}

__device__ inline bool atomic_max_cell_record_updated(RogueRecord* record,
                                                      const double value,
                                                      const int cell_idx,
                                                      const int group_idx) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return false;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(&record->value);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      record->cell_idx = cell_idx;
      record->group_idx = group_idx;
      return true;
    }
    assumed = old;
  }
  return false;
}

__device__ inline void atomic_min_record(RogueRecord* records,
                                         const int slot,
                                         const double value,
                                         const int cell_idx,
                                         const int group_idx) {
  if (!isfinite(value)) {
    return;
  }
  RogueRecord* record = records + slot;
  auto* ull = reinterpret_cast<unsigned long long*>(&record->value);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) > value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      record->cell_idx = cell_idx;
      record->group_idx = group_idx;
      break;
    }
    assumed = old;
  }
}

__device__ inline void atomic_max_face_symmetry_record(
    FldFaceSymmetryRecord* record,
    const double value,
    const double c_lr,
    const double c_rl,
    const double energy_delta,
    const int cell_l,
    const int cell_r,
    const int direction) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(&record->value);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      record->c_lr = c_lr;
      record->c_rl = c_rl;
      record->energy_delta = energy_delta;
      record->cell_l = cell_l;
      record->cell_r = cell_r;
      record->direction = direction;
      break;
    }
    assumed = old;
  }
}

__device__ inline void atomic_max_per_row_defect_record(
    FldPerRowDefectRecord* record,
    const double value,
    const double row_defect,
    const double V_op,
    const double V_state,
    const int cell_idx,
    const int group_idx) {
  if (!(value >= 0.0) || !isfinite(value)) {
    return;
  }
  auto* ull = reinterpret_cast<unsigned long long*>(&record->value);
  unsigned long long old = *ull;
  unsigned long long assumed = old;
  const unsigned long long val =
      static_cast<unsigned long long>(__double_as_longlong(value));
  while (__longlong_as_double(static_cast<long long>(assumed)) < value) {
    old = atomicCAS(ull, assumed, val);
    if (old == assumed) {
      record->row_defect = row_defect;
      record->V_op = V_op;
      record->V_state = V_state;
      record->cell_idx = cell_idx;
      record->group_idx = group_idx;
      break;
    }
    assumed = old;
  }
}

__device__ double flux_limiter_lambda(const double R, const int limiter) {
  const double r = fmax(finite_or_zero(R), 0.0);
  if (limiter == 2) {
    return 1.0 / 3.0;
  }
  if (limiter == 1) {
    return 1.0 / sqrt(9.0 + r * r);
  }
  if (r < 1.0e-6) {
    return 1.0 / 3.0;
  }
  if (r > 50.0) {
    return 1.0 / r;
  }
  return (1.0 / tanh(r) - 1.0 / r) / r;
}

int limiter_id(const std::string& limiter) {
  if (limiter == "larsen") {
    return 1;
  }
  if (limiter == "none") {
    return 2;
  }
  return 0;
}

int state_supply_boundary_policy_id(const std::string& policy) {
  if (policy == "harmonic_ghost_D_test") {
    return kStateSupplyBoundaryPolicyHarmonicGhostDTest;
  }
  if (policy == "radial_mean_D_test") {
    return kStateSupplyBoundaryPolicyRadialMeanDTest;
  }
  return kStateSupplyBoundaryPolicyLocalDCurrent;
}

struct DeviceOpacityTable {
  double* d_log_temps = nullptr;
  double* d_log_numdens = nullptr;
  double* d_kappa_PA = nullptr;
  double* d_kappa_PE = nullptr;
  double* d_kappa_R = nullptr;
  int ntemp = 0;
  int ndens = 0;
  int ngroups = 0;
  double log_T_min = 0.0;
  double log_T_max = 0.0;
  double log_ni_min = 0.0;
  double log_ni_max = 0.0;
  double T_min = 0.0;
  double T_max = 0.0;

  DeviceOpacityTable() = default;
  DeviceOpacityTable(const DeviceOpacityTable&) = delete;
  DeviceOpacityTable& operator=(const DeviceOpacityTable&) = delete;

  ~DeviceOpacityTable() {
    release();
  }

  void upload(const materials::IonmixOpacityData& host) {
    release();
    TENRYU_ASSERT(host.ntemp > 0 && host.ndens > 0 && host.ngroups > 0,
                  "FLD2D DeviceOpacityTable requires non-empty opacity table");
    const std::size_t n_table =
        static_cast<std::size_t>(host.ngroups) *
        static_cast<std::size_t>(host.ndens) *
        static_cast<std::size_t>(host.ntemp);
    TENRYU_ASSERT(host.kappa_PA.size() == n_table &&
                      host.kappa_PE.size() == n_table &&
                      host.kappa_R.size() == n_table,
                  "FLD2D DeviceOpacityTable opacity table size mismatch");
    const auto upload_vec = [](double** dst,
                               const std::vector<double>& src,
                               const char* label) {
      cuda_check(cudaMalloc(reinterpret_cast<void**>(dst),
                            sizeof(double) * src.size()),
                 label);
      cuda_check(cudaMemcpy(*dst,
                            src.data(),
                            sizeof(double) * src.size(),
                            cudaMemcpyHostToDevice),
                 label);
    };
    upload_vec(&d_log_temps, host.log_temps, "FLD2D upload log_temps failed");
    upload_vec(&d_log_numdens, host.log_numdens, "FLD2D upload log_numdens failed");
    upload_vec(&d_kappa_PA, host.kappa_PA, "FLD2D upload kappa_PA failed");
    upload_vec(&d_kappa_PE, host.kappa_PE, "FLD2D upload kappa_PE failed");
    upload_vec(&d_kappa_R, host.kappa_R, "FLD2D upload kappa_R failed");
    ntemp = host.ntemp;
    ndens = host.ndens;
    ngroups = host.ngroups;
    log_T_min = host.log_temps.front();
    log_T_max = host.log_temps.back();
    log_ni_min = host.log_numdens.front();
    log_ni_max = host.log_numdens.back();
    T_min = host.temps_eV.front();
    T_max = host.temps_eV.back();
  }

  void release() {
    if (d_kappa_R != nullptr) {
      cuda_check(cudaFree(d_kappa_R), "FLD2D free kappa_R failed");
      d_kappa_R = nullptr;
    }
    if (d_kappa_PE != nullptr) {
      cuda_check(cudaFree(d_kappa_PE), "FLD2D free kappa_PE failed");
      d_kappa_PE = nullptr;
    }
    if (d_kappa_PA != nullptr) {
      cuda_check(cudaFree(d_kappa_PA), "FLD2D free kappa_PA failed");
      d_kappa_PA = nullptr;
    }
    if (d_log_numdens != nullptr) {
      cuda_check(cudaFree(d_log_numdens), "FLD2D free log_numdens failed");
      d_log_numdens = nullptr;
    }
    if (d_log_temps != nullptr) {
      cuda_check(cudaFree(d_log_temps), "FLD2D free log_temps failed");
      d_log_temps = nullptr;
    }
    ntemp = 0;
    ndens = 0;
    ngroups = 0;
  }

  materials::IonmixOpacityDeviceView view() const {
    materials::IonmixOpacityDeviceView out{};
    out.log_temps = d_log_temps;
    out.log_numdens = d_log_numdens;
    out.kappa_PA = d_kappa_PA;
    out.kappa_PE = d_kappa_PE;
    out.kappa_R = d_kappa_R;
    out.ntemp = ntemp;
    out.ndens = ndens;
    out.ngroups = ngroups;
    out.log_T_min = log_T_min;
    out.log_T_max = log_T_max;
    out.log_ni_min = log_ni_min;
    out.log_ni_max = log_ni_max;
    out.T_min = T_min;
    out.T_max = T_max;
    return out;
  }
};

struct NlteOpacityCache {
  std::unique_ptr<materials::IonmixOpacityData> host;
  DeviceOpacityTable device;
  std::string file;
  int groups = 0;
};

NlteOpacityCache& nlte_cache() {
  static NlteOpacityCache cache;
  return cache;
}

struct FldGroupBoundsCache {
  std::vector<double> last_bounds;
  bool valid = false;
};

FldGroupBoundsCache& fld_group_bounds_cache() {
  static FldGroupBoundsCache cache;
  return cache;
}

class FldElectronEOSTableCache {
 public:
  materials::DeviceEOSTableView view_for(const materials::EOSTableTriplet& tables) {
    if (tables_ != &tables) {
      electron_.upload(tables.electron);
      tables_ = &tables;
    }
    return electron_.view();
  }

 private:
  const materials::EOSTableTriplet* tables_ = nullptr;
  materials::DeviceEOSTable electron_;
};

FldElectronEOSTableCache& fld_electron_eos_table_cache() {
  static FldElectronEOSTableCache cache;
  return cache;
}

materials::DeviceEOSTableView fld_electron_eos_device_view(
    const materials::EOSTableTriplet* tables) {
  if (tables == nullptr) {
    return materials::DeviceEOSTableView{};
  }
  return fld_electron_eos_table_cache().view_for(*tables);
}

__device__ inline bool has_electron_eos_table(
    const materials::DeviceEOSTableView& electron_eos) {
  return electron_eos.n_rho > 0 && electron_eos.n_T > 0 &&
         electron_eos.P_table != nullptr &&
         electron_eos.e_table != nullptr &&
         electron_eos.cv_table != nullptr;
}

__device__ inline double eos_log_temperature(const double T,
                                             const double temperature_floor_eV) {
  return log(fmax(finite_or_zero(T), fmax(temperature_floor_eV, 1.0e-30)));
}

void ensure_nlte_table_uploaded(const core::Config& cfg,
                                const core::Config::MaterialsConfig::MatDef& mat,
                                const int n_groups) {
  auto& cache = nlte_cache();
  if (cache.host != nullptr && cache.file == mat.opacity_file &&
      cache.groups == n_groups) {
    return;
  }
  if (mat.opacity_model == "tmat") {
    const materials::TmatFile tmat = materials::load_tmat(mat.opacity_file);
    TENRYU_ASSERT(tmat.opacity.has_value(),
                  "FLD2D tmat opacity.model requires /opacity payload");
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::tmat_to_ionmix_opacity(*tmat.opacity));
  } else {
    cache.host = std::make_unique<materials::IonmixOpacityData>(
        materials::load_ionmix_opacity(mat.opacity_file));
  }
  if (cfg.radiation.group_repack_hard_xray) {
    std::vector<double> target_bounds = cfg.radiation.group_bounds_eV;
    if (target_bounds.empty()) {
      const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
      target_bounds =
          repack_group_bounds_for_hard_xray(n_groups, range);
    }
    *cache.host = resample_opacity_groups_to_bounds(*cache.host, target_bounds);
  }
  TENRYU_ASSERT(cache.host->ngroups == n_groups,
                "FLD2D NLTE opacity table group count must match Radiation.groups");
  cache.device.upload(*cache.host);
  cache.file = mat.opacity_file;
  cache.groups = n_groups;
}

struct CellGeometryRZ {
  double r_left = 0.0;
  double r_right = 0.0;
  double z_bottom = 0.0;
  double z_top = 0.0;
  double r_center = 0.0;
  double z_center = 0.0;
  double area_iL = 0.0;
  double area_iR = 0.0;
  double area_jB = 0.0;
  double area_jT = 0.0;
  double V_op = 0.0;
};

__device__ inline CellGeometryRZ rect_cell_geometry_v2(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const int i,
    const int j,
    const int nz) {
  const int n00 = node_index(i, j, nz);
  const int n10 = node_index(i + 1, j, nz);
  const int n11 = node_index(i + 1, j + 1, nz);
  const int n01 = node_index(i, j + 1, nz);
  const double r00 = finite_or_zero(x_r[n00]);
  const double r10 = finite_or_zero(x_r[n10]);
  const double r11 = finite_or_zero(x_r[n11]);
  const double r01 = finite_or_zero(x_r[n01]);
  const double z00 = finite_or_zero(x_z[n00]);
  const double z10 = finite_or_zero(x_z[n10]);
  const double z11 = finite_or_zero(x_z[n11]);
  const double z01 = finite_or_zero(x_z[n01]);

  CellGeometryRZ geom;
  geom.r_left = fmax(0.5 * (r00 + r01), 0.0);
  geom.r_right = fmax(0.5 * (r10 + r11), 0.0);
  geom.z_bottom = 0.5 * (z00 + z10);
  geom.z_top = 0.5 * (z01 + z11);
  geom.r_center = 0.25 * (r00 + r10 + r11 + r01);
  geom.z_center = 0.25 * (z00 + z10 + z11 + z01);
  geom.area_iL = fld_edge_area_rz(r00, r01, z00, z01);
  geom.area_iR = fld_edge_area_rz(r10, r11, z10, z11);
  geom.area_jB = fld_edge_area_rz(r00, r10, z00, z10);
  geom.area_jT = fld_edge_area_rz(r01, r11, z01, z11);
  const int c = cell_index(i, j, nz);
  const double V_input = nonnegative_finite(vol[c]);
  const double dz = fmax(fabs(geom.z_top - geom.z_bottom), 0.0);
  const double area_z_mid =
      kPi * fmax(geom.r_right * geom.r_right - geom.r_left * geom.r_left, 0.0);
  geom.V_op = (V_input > 0.0) ? V_input : area_z_mid * dz;
  return geom;
}

__host__ __device__ inline double fld_state_supply_E(const double T_supply_eV) {
  const double T = nonnegative_finite(T_supply_eV);
  const double T2 = T * T;
  return core::constants::a_eV * T2 * T2;
}

__device__ double harmonic_positive(const double a,
                                    const double b,
                                    int* __restrict__ skip_D_count = nullptr);

__device__ inline double fld_state_supply_radial_mean_D(
    const double* __restrict__ D_cells,
    const int nr,
    const int nz,
    const int n_groups,
    const int boundary_j,
    const int g) {
  if (D_cells == nullptr || nr <= 0 || nz <= 0 || n_groups <= 0 || g < 0 ||
      g >= n_groups || boundary_j < 0 || boundary_j >= nz) {
    return 0.0;
  }
  double sum = 0.0;
  for (int ii = 0; ii < nr; ++ii) {
    const int c = cell_index(ii, boundary_j, nz);
    sum += nonnegative_finite(D_cells[c * n_groups + g]);
  }
  return nonnegative_finite(sum / static_cast<double>(nr));
}

__device__ inline double fld_state_supply_ghost_D(
    const double rho_cell,
    const double sigma_R_cell,
    const double rho_ghost,
    const int limiter) {
  const double rho_local = nonnegative_finite(rho_cell);
  const double sigma_R_local = nonnegative_finite(sigma_R_cell);
  const double kappa_R_local =
      (rho_local > 0.0) ? (sigma_R_local / rho_local) : 0.0;
  const double sigma_R_ghost = kappa_R_local * nonnegative_finite(rho_ghost);
  if (!(sigma_R_ghost > 0.0)) {
    return 0.0;
  }
  const double lambda_ghost = flux_limiter_lambda(0.0, limiter);
  return core::constants::c_light * lambda_ghost / sigma_R_ghost;
}

__device__ inline double fld_state_supply_boundary_coeff(
    const CellGeometryRZ& geom,
    const double D_cell,
    const bool top_face,
    const int policy = kStateSupplyBoundaryPolicyLocalDCurrent,
    const double rho_cell = 0.0,
    const double sigma_R_cell = 0.0,
    const double rho_ghost = 0.0,
    const double* __restrict__ D_cells = nullptr,
    const int nr = 0,
    const int nz = 0,
    const int n_groups = 1,
    const int boundary_j = 0,
    const int g = 0,
    const int limiter = 0) {
  const double area = top_face ? geom.area_jT : geom.area_jB;
  const double d_raw = top_face ? (geom.z_top - geom.z_center)
                                : (geom.z_center - geom.z_bottom);
  const double d_face = fmax(fabs(finite_or_zero(d_raw)), kFldFaceDistMin);
  if (policy == kStateSupplyBoundaryPolicyLocalDCurrent) {
    const double coeff = area * nonnegative_finite(D_cell) / d_face;
    return (isfinite(coeff) && coeff > 0.0) ? coeff : 0.0;
  }
  double D_face = 0.0;
  if (policy == kStateSupplyBoundaryPolicyHarmonicGhostDTest) {
    const double D_ghost =
        fld_state_supply_ghost_D(rho_cell, sigma_R_cell, rho_ghost, limiter);
    D_face = harmonic_positive(nonnegative_finite(D_cell), D_ghost, nullptr);
  } else if (policy == kStateSupplyBoundaryPolicyRadialMeanDTest) {
    D_face = fld_state_supply_radial_mean_D(
        D_cells, nr, nz, n_groups, boundary_j, g);
  }
  const double coeff = area * D_face / d_face;
  return (isfinite(coeff) && coeff > 0.0) ? coeff : 0.0;
}

__global__ void build_eta_from_planck_kernel(
    const double* __restrict__ Te,
    const double* __restrict__ sigma_a,
    PlanckTableDeviceView planck,
    double* __restrict__ eta,
    int n_cells,
    int n_groups,
    double temperature_floor_eV) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const double T = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
  const double T2 = T * T;
  const double B = core::constants::a_eV * T2 * T2 *
                   fmax(planck.interpolate_b(g, T), 0.0);
  eta[idx] = core::constants::c_light *
             fmax(finite_or_zero(sigma_a[idx]), 0.0) * B;
}

__global__ void compute_cell_centers_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    double* __restrict__ rc_out,
    double* __restrict__ zc_out,
    int nr,
    int nz) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  rc_out[c] = geom.r_center;
  zc_out[c] = geom.z_center;
}

__global__ void compute_d_cell_2d_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ rad_E,
    const double* __restrict__ sigma_R,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    double* __restrict__ D_cell,
    int nr,
    int nz,
    int n_groups,
    double sigma_floor,
    int limiter) {
  (void)x_r;
  (void)x_z;
  (void)vol;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const int i = c / nz;
  const int j = c - i * nz;
  const double E = fmax(finite_or_zero(rad_E[idx]), 0.0);
  double dEdr = 0.0;
  double dEdz = 0.0;
  if (nr > 1) {
    const int il = (i > 0) ? (i - 1) : i;
    const int ir = (i + 1 < nr) ? (i + 1) : i;
    const int cl = cell_index(il, j, nz);
    const int cr = cell_index(ir, j, nz);
    const double El = fmax(finite_or_zero(rad_E[cl * n_groups + g]), 0.0);
    const double Er = fmax(finite_or_zero(rad_E[cr * n_groups + g]), 0.0);
    const double rl = rc_cache[cl];
    const double rr = rc_cache[cr];
    dEdr = (Er - El) / fmax(rr - rl, 1.0e-300);
  }
  if (nz > 1) {
    const int jb = (j > 0) ? (j - 1) : j;
    const int jt = (j + 1 < nz) ? (j + 1) : j;
    const int cb = cell_index(i, jb, nz);
    const int ct = cell_index(i, jt, nz);
    const double Eb = fmax(finite_or_zero(rad_E[cb * n_groups + g]), 0.0);
    const double Et = fmax(finite_or_zero(rad_E[ct * n_groups + g]), 0.0);
    const double zb = zc_cache[cb];
    const double zt = zc_cache[ct];
    dEdz = (Et - Eb) / fmax(zt - zb, 1.0e-300);
  }
  const double sigma = fmax(finite_or_zero(sigma_R[idx]), sigma_floor);
  const double grad = sqrt(dEdr * dEdr + dEdz * dEdz);
  const double R = grad / (sigma * fmax(E, 1.0e-300));
  const double lambda = flux_limiter_lambda(R, limiter);
  D_cell[idx] = core::constants::c_light * lambda / sigma;
}

__device__ double harmonic_positive(const double a,
                                    const double b,
                                    int* __restrict__ skip_D_count) {
  if (!(a > 0.0) || !(b > 0.0)) {
    if (skip_D_count != nullptr) {
      atomicAdd(skip_D_count, 1);
    }
    return 0.0;
  }
  return 2.0 / (1.0 / a + 1.0 / b);
}

bool use_fld_fleck(const core::Config::MaterialsConfig::MatDef& mat) {
  return mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat" ||
         mat.opacity_model == "constant" || mat.opacity_model == "power_law";
}

__global__ void compute_fleck_for_fld_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ Te,
    const double* __restrict__ zbar,
    const std::uint8_t* __restrict__ cell_is_void,
    const double* __restrict__ sigma_a,
    const double* __restrict__ state_cv_e,
    double* __restrict__ f_fleck,
    const int n_cells,
    const int n_groups,
    const double dt,
    const double alpha,
    const double cv_e_override,
    const double gamma,
    const double A,
    const materials::DeviceEOSTableView electron_eos,
    const int use_table_cv,
    const double temperature_floor_eV) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  const int base = c * n_groups;
  if (cell_is_void != nullptr && cell_is_void[c] != 0U) {
    for (int g = 0; g < n_groups; ++g) {
      f_fleck[base + g] = 1.0;
    }
    return;
  }

  const double rho_c = rho[c];
  const double Te_c = fmax(Te[c], 1.0e-12);
  double Cv_e = -1.0;
  if (use_table_cv != 0 && has_electron_eos_table(electron_eos)) {
    const materials::RhoBracket br = materials::find_rho_bracket(electron_eos, rho_c);
    const double cv_mass = materials::device_eos_cv(
        electron_eos, br, eos_log_temperature(Te_c, temperature_floor_eV));
    if (isfinite(cv_mass) && cv_mass > 0.0) {
      Cv_e = fmax(rho_c, 0.0) * cv_mass;
    }
  }
  if (Cv_e <= 0.0) {
    if (cv_e_override > 0.0) {
      Cv_e = cv_e_override;
    } else if (state_cv_e != nullptr && state_cv_e[c] > 0.0) {
      Cv_e = fmax(rho_c, 0.0) * state_cv_e[c];
    } else {
      const double z_atom = fmax(zbar[c], 0.0);
      const double gm1 = fmax(gamma - 1.0, 1.0e-12);
      const double cv_mass_e =
          z_atom * core::constants::eV_to_erg /
          (fmax(A, 1.0e-12) * core::constants::proton_mass * gm1);
      Cv_e = fmax(rho_c, 0.0) * cv_mass_e;
    }
  }
  Cv_e = fmax(Cv_e, 1.0e-30);

  const double beta =
      4.0 * core::constants::a_eV * Te_c * Te_c * Te_c / Cv_e;
  const double alpha_safe = (alpha > 0.0) ? alpha : 1.0;
  constexpr double kBlendZ0 = 2.0;
  constexpr double kBlendZ1 = 10.0;
  for (int g = 0; g < n_groups; ++g) {
    const double sigma_g = fmax(sigma_a[base + g], 0.0);
    const double z =
        alpha_safe * beta * core::constants::c_light * sigma_g * dt;
    const double f_std = 1.0 / (1.0 + z);
    const double f_mu = exp(-z);
    const double t = (z <= kBlendZ0)
                         ? 0.0
                         : (z >= kBlendZ1) ? 1.0
                                           : (z - kBlendZ0) / (kBlendZ1 - kBlendZ0);
    const double w = t * t * (3.0 - 2.0 * t);
    double f = (1.0 - w) * f_std + w * f_mu;
    if (!isfinite(f) || f < 0.0) {
      f = 0.0;
    }
    if (f > 1.0) {
      f = 1.0;
    }
    f_fleck[base + g] = f;
  }
}

__device__ void add_offdiag(double coef,
                            int row,
                            int col,
                            int* __restrict__ col_indices,
                            double* __restrict__ values,
                            int slot,
                            double* diag,
                            int* __restrict__ face_skip_nonfinite_count) {
  const int base = row * kCsrEntriesPerRow;
  if (isfinite(coef) && coef > 0.0) {
    *diag += coef;
    col_indices[base + slot] = col;
    values[base + slot] = -coef;
  } else {
    if (!isfinite(coef) && face_skip_nonfinite_count != nullptr) {
      atomicAdd(face_skip_nonfinite_count, 1);
    }
    col_indices[base + slot] = row;
    values[base + slot] = 0.0;
  }
}

__device__ double fld_rhs_source_rate(const double* __restrict__ eta,
                                      const double* __restrict__ fleck,
                                      const double* __restrict__ sigma_pa,
                                      const double* __restrict__ rad_E_old,
                                      int cg,
                                      int c) {
  const double E_old = nonnegative_finite(rad_E_old[cg]);
  double source = nonnegative_finite(eta[cg]);
  if (fleck != nullptr && sigma_pa != nullptr) {
    // BUG-11: fleck is a cg array (see nlte_coeffs.cu layout contract);
    // the bare-cell read consumed group 0's f for every group when G > 1.
    const double f = fmin(fmax(finite_or_zero(fleck[cg]), 0.0), 1.0);
    source = f * source +
             (1.0 - f) * core::constants::c_light *
                 nonnegative_finite(sigma_pa[cg]) * E_old;
  }
  return source;
}

__global__ void assemble_fld_2d_csr_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ sigma_pa,
    const double* __restrict__ fleck,
    const double* __restrict__ eta,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ D_cell,
    const double* __restrict__ Te,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    int* __restrict__ row_offsets,
    int* __restrict__ col_indices,
    double* __restrict__ values,
    double* __restrict__ rhs,
    double* __restrict__ diag_inv,
    int nr,
    int nz,
    int n_groups,
    double dt,
    int outer_r_bc,
    int z_bottom_bc,
    int z_top_bc,
    double marshak_flux_erg_per_cm2_s,
    double T_supply_z_bottom_eV,
    double T_supply_z_top_eV,
    int* __restrict__ face_skip_D_count,
    int* __restrict__ face_skip_nonfinite_count,
    int* __restrict__ face_skip_dist_floor_count,
    int* __restrict__ diag_fallback_count,
    int trace_row,
    FldTraceRecord* __restrict__ trace_record,
    const double* __restrict__ rho = nullptr,
    const double* __restrict__ sigma_R = nullptr,
    int state_supply_boundary_policy =
        kStateSupplyBoundaryPolicyLocalDCurrent,
    int limiter = 0,
    double rho_supply_z_bottom = 0.0,
    double rho_supply_z_top = 0.0) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int n_rows = n_cells * n_groups;
  if (row > n_rows) {
    return;
  }
  row_offsets[row] = row * kCsrEntriesPerRow;
  if (row == n_rows) {
    return;
  }
  const int g = row / n_cells;
  const int c = row - g * n_cells;
  const int i = c / nz;
  const int j = c - i * nz;
  const int cg = c * n_groups + g;
  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  const double rc = geom.r_center;
  const double zc = geom.z_center;
  const double V = geom.V_op;

  const int base = row * kCsrEntriesPerRow;
  const double sig = fmax(finite_or_zero(sigma_removal[cg]), 0.0);
  const double diag_V = V;
  const double diag_csigV = dt * core::constants::c_light * sig * V;
  double diag = V + dt * core::constants::c_light * sig * V;
  double boundary_source = 0.0;
  col_indices[base] = row;
  values[base] = 0.0;

  if (i > 0) {
    const int cn = cell_index(i - 1, j, nz);
    const int ng = g * n_cells + cn;
    const double rn = rc_cache[cn];
    const double raw_dist = rc - rn;
    if (!(raw_dist > kFldFaceDistMin)) {
      if (face_skip_dist_floor_count != nullptr) {
        atomicAdd(face_skip_dist_floor_count, 1);
      }
      col_indices[base + 1] = row;
      values[base + 1] = 0.0;
    } else {
      const double Df = harmonic_positive(
          D_cell[cn * n_groups + g], D_cell[cg], face_skip_D_count);
      add_offdiag(dt * geom.area_iL * Df / raw_dist,
                  row, ng, col_indices, values, 1, &diag,
                  face_skip_nonfinite_count);
    }
  } else {
    col_indices[base + 1] = row;
    values[base + 1] = 0.0;
  }
  if (i + 1 < nr) {
    const int cn = cell_index(i + 1, j, nz);
    const int ng = g * n_cells + cn;
    const double rn = rc_cache[cn];
    const double raw_dist = rn - rc;
    if (!(raw_dist > kFldFaceDistMin)) {
      if (face_skip_dist_floor_count != nullptr) {
        atomicAdd(face_skip_dist_floor_count, 1);
      }
      col_indices[base + 2] = row;
      values[base + 2] = 0.0;
    } else {
      const double Df = harmonic_positive(
          D_cell[cg], D_cell[cn * n_groups + g], face_skip_D_count);
      add_offdiag(dt * geom.area_iR * Df / raw_dist,
                  row, ng, col_indices, values, 2, &diag,
                  face_skip_nonfinite_count);
    }
  } else {
    diag += dt * geom.area_iR * fld_boundary_leakage_coeff(outer_r_bc);
    col_indices[base + 2] = row;
    values[base + 2] = 0.0;
  }
  if (j > 0) {
    const int cn = cell_index(i, j - 1, nz);
    const int ng = g * n_cells + cn;
    const double zn = zc_cache[cn];
    const double raw_dist = zc - zn;
    if (!(raw_dist > kFldFaceDistMin)) {
      if (face_skip_dist_floor_count != nullptr) {
        atomicAdd(face_skip_dist_floor_count, 1);
      }
      col_indices[base + 3] = row;
      values[base + 3] = 0.0;
    } else {
      const double Df = harmonic_positive(
          D_cell[cn * n_groups + g], D_cell[cg], face_skip_D_count);
      add_offdiag(dt * geom.area_jB * Df / raw_dist,
                  row, ng, col_indices, values, 3, &diag,
                  face_skip_nonfinite_count);
    }
  } else {
    if (z_bottom_bc == kFldBcStateSupply) {
      const double coeff =
          fld_state_supply_boundary_coeff(
              geom,
              D_cell[cg],
              false,
              state_supply_boundary_policy,
              (rho != nullptr) ? rho[c] : 0.0,
              (sigma_R != nullptr) ? sigma_R[cg] : 0.0,
              rho_supply_z_bottom,
              D_cell,
              nr,
              nz,
              n_groups,
              0,
              g,
              limiter);
      const double dt_coeff = dt * coeff;
      diag += dt_coeff;
      boundary_source +=
          dt_coeff * fld_state_supply_E(T_supply_z_bottom_eV);
    } else {
      diag += dt * geom.area_jB * fld_boundary_leakage_coeff(z_bottom_bc);
    }
    if (z_bottom_bc == kFldBcMarshak) {
      boundary_source += dt * geom.area_jB * marshak_flux_erg_per_cm2_s;
    }
    col_indices[base + 3] = row;
    values[base + 3] = 0.0;
  }
  if (j + 1 < nz) {
    const int cn = cell_index(i, j + 1, nz);
    const int ng = g * n_cells + cn;
    const double zn = zc_cache[cn];
    const double raw_dist = zn - zc;
    if (!(raw_dist > kFldFaceDistMin)) {
      if (face_skip_dist_floor_count != nullptr) {
        atomicAdd(face_skip_dist_floor_count, 1);
      }
      col_indices[base + 4] = row;
      values[base + 4] = 0.0;
    } else {
      const double Df = harmonic_positive(
          D_cell[cg], D_cell[cn * n_groups + g], face_skip_D_count);
      add_offdiag(dt * geom.area_jT * Df / raw_dist,
                  row, ng, col_indices, values, 4, &diag,
                  face_skip_nonfinite_count);
    }
  } else {
    if (z_top_bc == kFldBcStateSupply) {
      const double coeff =
          fld_state_supply_boundary_coeff(
              geom,
              D_cell[cg],
              true,
              state_supply_boundary_policy,
              (rho != nullptr) ? rho[c] : 0.0,
              (sigma_R != nullptr) ? sigma_R[cg] : 0.0,
              rho_supply_z_top,
              D_cell,
              nr,
              nz,
              n_groups,
              nz - 1,
              g,
              limiter);
      const double dt_coeff = dt * coeff;
      diag += dt_coeff;
      boundary_source += dt_coeff * fld_state_supply_E(T_supply_z_top_eV);
    } else {
      diag += dt * geom.area_jT * fld_boundary_leakage_coeff(z_top_bc);
    }
    if (z_top_bc == kFldBcMarshak) {
      boundary_source += dt * geom.area_jT * marshak_flux_erg_per_cm2_s;
    }
    col_indices[base + 4] = row;
    values[base + 4] = 0.0;
  }

  if (!(isfinite(diag) && diag > 0.0) && diag_fallback_count != nullptr) {
    atomicAdd(diag_fallback_count, 1);
  }
  values[base] = (isfinite(diag) && diag > 0.0) ? diag : 1.0;
  diag_inv[row] = 1.0 / values[base];
  const double E_old = nonnegative_finite(rad_E_old[cg]);
  const double sigma_pa_value =
      (sigma_pa != nullptr) ? nonnegative_finite(sigma_pa[cg]) : sig;
  // BUG-11: cg read (see nlte_coeffs.cu layout contract).
  const double fleck_f =
      (fleck != nullptr) ? fmin(fmax(finite_or_zero(fleck[cg]), 0.0), 1.0) : 1.0;
  const double eta_value = nonnegative_finite(eta[cg]);
  const double source = fleck_f * eta_value +
                        (1.0 - fleck_f) * core::constants::c_light *
                            sigma_pa_value * E_old;
  const double rhs_V_E_old = V * E_old;
  const double rhs_dt_V_f_eta = dt * V * fleck_f * eta_value;
  const double rhs_dt_V_one_minus_f_csE =
      dt * V * (1.0 - fleck_f) * core::constants::c_light *
      sigma_pa_value * E_old;
  rhs[row] = rhs_V_E_old + dt * V * source + boundary_source;
  if (!isfinite(rhs[row])) {
    rhs[row] = 0.0;
  }
  if (trace_record != nullptr && row == trace_row) {
    const double Te_in = nonnegative_finite(Te[c]);
    const double T2 = Te_in * Te_in;
    trace_record->cell_matched = 1;
    trace_record->i = i;
    trace_record->j = j;
    trace_record->group = g;
    trace_record->Te_in = Te_in;
    trace_record->E_old_in = E_old;
    trace_record->sigma_pa = sigma_pa_value;
    trace_record->sigma_removal = sig;
    trace_record->B_T = core::constants::a_eV * T2 * T2;
    trace_record->eta = eta_value;
    trace_record->fleck_f = fleck_f;
    trace_record->D_cell = nonnegative_finite(D_cell[cg]);
    trace_record->rhs_V_E_old = rhs_V_E_old;
    trace_record->rhs_dt_V_f_eta = rhs_dt_V_f_eta;
    trace_record->rhs_dt_V_one_minus_f_csE = rhs_dt_V_one_minus_f_csE;
    trace_record->rhs_boundary = boundary_source;
    trace_record->rhs_total = rhs[row];
    trace_record->diag_V = diag_V;
    trace_record->diag_csigV = diag_csigV;
    trace_record->diag_face_total = diag - diag_V - diag_csigV;
    trace_record->diag_total = values[base];
    trace_record->offdiag_iL = values[base + 1];
    trace_record->offdiag_iR = values[base + 2];
    trace_record->offdiag_jB = values[base + 3];
    trace_record->offdiag_jT = values[base + 4];
  }
}

__device__ double fld_audit_field_value(
    const double* __restrict__ field,
    const int layout,
    const int c,
    const int g,
    const int n_cells,
    const int n_groups) {
  if (field == nullptr) {
    return 0.0;
  }
  if (layout ==
      static_cast<int>(diagnostics::FldSubstageAuditFieldLayout::CellScalar)) {
    return finite_or_zero(field[c]);
  }
  if (layout ==
      static_cast<int>(diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup)) {
    return finite_or_zero(field[c * n_groups + g]);
  }
  if (layout ==
      static_cast<int>(diagnostics::FldSubstageAuditFieldLayout::GroupMajor)) {
    return finite_or_zero(field[g * n_cells + c]);
  }
  return 0.0;
}

__global__ void fld_substage_audit_top_boundary_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ D_cell,
    const double* __restrict__ rho,
    const double* __restrict__ sigma_R,
    double* __restrict__ coeff_out,
    double* __restrict__ dt_coeff_out,
    double* __restrict__ source_out,
    double* __restrict__ diag_out,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    const int z_top_bc,
    const double T_supply_z_top_eV,
    const int state_supply_boundary_policy,
    const int limiter,
    const double rho_supply_z_top) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  const int i = c / nz;
  const int j = c - i * nz;
  double coeff = 0.0;
  double dt_coeff = 0.0;
  double source = 0.0;
  double diag = 0.0;
  if (j + 1 == nz && z_top_bc == kFldBcStateSupply) {
    const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
    coeff = fld_state_supply_boundary_coeff(
        geom,
        D_cell[idx],
        true,
        state_supply_boundary_policy,
        (rho != nullptr) ? rho[c] : 0.0,
        (sigma_R != nullptr) ? sigma_R[idx] : 0.0,
        rho_supply_z_top,
        D_cell,
        nr,
        nz,
        n_groups,
        nz - 1,
        g,
        limiter);
    dt_coeff = dt * coeff;
    source = dt_coeff * fld_state_supply_E(T_supply_z_top_eV);
    diag = dt_coeff;
  }
  coeff_out[idx] = coeff;
  dt_coeff_out[idx] = dt_coeff;
  source_out[idx] = source;
  diag_out[idx] = diag;
}

__global__ void fld_substage_audit_fourier_kernel(
    const double* __restrict__ field,
    const int layout,
    const int* __restrict__ m_targets,
    const int* __restrict__ j_targets,
    const int n_m,
    const int n_j,
    const int nr,
    const int nz,
    const int n_groups,
    const std::uint8_t substage_id,
    const std::uint8_t field_id,
    const int outer_iter,
    const double solver_residual_l2_rel,
    const double solver_residual_max,
    diagnostics::FldSubstageAuditRecord* __restrict__ records) {
  const int target = blockIdx.x;
  const int total = n_groups * n_m * n_j;
  if (target >= total) {
    return;
  }
  const int tid = threadIdx.x;
  const int j_idx = target % n_j;
  const int m_idx = (target / n_j) % n_m;
  const int g = target / (n_j * n_m);
  const int m = m_targets[m_idx];
  const int j = j_targets[j_idx];
  const int n_cells = nr * nz;

  extern __shared__ double shared[];
  double* sum = shared;
  double* qmin = shared + blockDim.x;
  double* qmax = shared + 2 * blockDim.x;

  double local_sum = 0.0;
  double local_min = 1.0e300;
  double local_max = -1.0e300;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double q =
        fld_audit_field_value(field, layout, c, g, n_cells, n_groups);
    local_sum += q;
    local_min = fmin(local_min, q);
    local_max = fmax(local_max, q);
  }
  sum[tid] = local_sum;
  qmin[tid] = local_min;
  qmax[tid] = local_max;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum[tid] += sum[tid + stride];
      qmin[tid] = fmin(qmin[tid], qmin[tid + stride]);
      qmax[tid] = fmax(qmax[tid], qmax[tid + stride]);
    }
    __syncthreads();
  }

  const double mean = sum[0] / static_cast<double>(nr);
  const double row_min = qmin[0];
  const double row_max = qmax[0];
  __syncthreads();

  double local_cre = 0.0;
  double local_cim = 0.0;
  for (int i = tid; i < nr; i += blockDim.x) {
    const int c = i * nz + j;
    const double q =
        fld_audit_field_value(field, layout, c, g, n_cells, n_groups);
    const double residual = q - mean;
    const double theta =
        2.0 * kPi * static_cast<double>(m * i) / static_cast<double>(nr);
    local_cre += residual * cos(theta);
    local_cim -= residual * sin(theta);
  }
  sum[tid] = local_cre;
  qmin[tid] = local_cim;
  __syncthreads();

  for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum[tid] += sum[tid + stride];
      qmin[tid] += qmin[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    auto& rec = records[target];
    rec.valid = true;
    rec.substage_id = substage_id;
    rec.field_id = field_id;
    rec.normalization_kind = static_cast<std::uint8_t>(
        diagnostics::FldSubstageAuditNormalization::MeanSubtractedUnweightedRawSum);
    rec.m = m;
    rec.j = j;
    rec.group = g;
    rec.outer_iter = outer_iter;
    rec.nr = nr;
    rec.nz = nz;
    rec.cre = sum[0];
    rec.cim = qmin[0];
    rec.amplitude = sqrt(rec.cre * rec.cre + rec.cim * rec.cim);
    rec.phase = atan2(rec.cim, rec.cre);
    rec.mean = mean;
    rec.q_min_j = row_min;
    rec.q_max_j = row_max;
    rec.normalization = static_cast<double>(nr);
    rec.solver_residual_l2_rel = solver_residual_l2_rel;
    rec.solver_residual_max = solver_residual_max;
  }
}

__global__ void init_solution_kernel(const double* __restrict__ rad_E_old,
                                     double* __restrict__ x,
                                     int n_cells,
                                     int n_groups) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }
  const int g = row / n_cells;
  const int c = row - g * n_cells;
  x[row] = fmax(finite_or_zero(rad_E_old[c * n_groups + g]), 0.0);
}

__global__ void fld_trace_post_solve_kernel(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ rhs,
    const double* __restrict__ x,
    const double* __restrict__ vol,
    int n_groups,
    int n_cells,
    int nz,
    int trace_row,
    int trace_cell,
    int trace_group,
    FldTraceRecord* __restrict__ rec) {
  if (rec == nullptr || trace_row < 0 ||
      threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  rec->x_raw = finite_or_zero(x[trace_row]);
  rec->x_pub = fmax(rec->x_raw, 0.0);
  rec->clamp_delta = (rec->x_raw < 0.0)
                         ? fmax(finite_or_zero(vol[trace_cell]), 0.0) *
                               (rec->x_pub - rec->x_raw)
                         : 0.0;

  const int row_start = row_offsets[trace_row];
  const int row_end = row_offsets[trace_row + 1];
  double sum = 0.0;
  for (int k = row_start; k < row_end; ++k) {
    const int col = col_indices[k];
    if (col >= 0 && col < n_cells * n_groups) {
      sum += finite_or_zero(values[k]) * finite_or_zero(x[col]);
    }
  }
  rec->Arow_x = sum;
  rec->Arow_x_minus_rhs = sum - finite_or_zero(rhs[trace_row]);
  rec->rhs_total = finite_or_zero(rhs[trace_row]);

  const int trace_i = trace_cell / nz;
  const int trace_j = trace_cell - trace_i * nz;
  const int group_offset = trace_group * n_cells;
  rec->xL = (trace_i > 0) ? finite_or_zero(x[group_offset + trace_cell - nz]) : 0.0;
  rec->xR = (trace_cell + nz < n_cells)
                ? finite_or_zero(x[group_offset + trace_cell + nz])
                : 0.0;
  rec->xB = (trace_j > 0) ? finite_or_zero(x[group_offset + trace_cell - 1]) : 0.0;
  rec->xT = (trace_j + 1 < nz) ? finite_or_zero(x[group_offset + trace_cell + 1])
                               : 0.0;
  rec->rad_E_delta = fabs(rec->x_pub - rec->E_old_in);
  rec->cell_matched = 1;
}

__global__ void aa_diff_kernel(const double* __restrict__ g,
                               const double* __restrict__ u,
                               double* __restrict__ f,
                               const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    f[i] = g[i] - u[i];
  }
}

// u_next[i] = u[i] + beta*f[i] - sum_j gamma[j]*(dU_j[i] + beta*dF_j[i]),
// with dU_j/dF_j formed on the fly from the ring buffers; p <= 4.
// Non-finite results fall back to the raw Newton output g[i]; Te floor is
// applied by the caller argument te_floor.
__global__ void aa_mix_kernel(const double* __restrict__ u,
                              const double* __restrict__ f,
                              const double* __restrict__ u_hist0,
                              const double* __restrict__ u_hist1,
                              const double* __restrict__ u_hist2,
                              const double* __restrict__ u_hist3,
                              const double* __restrict__ u_hist4,
                              const double* __restrict__ f_hist0,
                              const double* __restrict__ f_hist1,
                              const double* __restrict__ f_hist2,
                              const double* __restrict__ f_hist3,
                              const double* __restrict__ f_hist4,
                              const double g0,
                              const double g1,
                              const double g2,
                              const double g3,
                              const int p,
                              const double beta,
                              const double te_floor,
                              const double* __restrict__ g_raw,
                              double* __restrict__ te_out,
                              const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double* u_hist[5] = {u_hist0, u_hist1, u_hist2, u_hist3, u_hist4};
  const double* f_hist[5] = {f_hist0, f_hist1, f_hist2, f_hist3, f_hist4};
  const double gamma[4] = {g0, g1, g2, g3};
  double val = u[i] + beta * f[i];
  for (int j = 0; j < p; ++j) {
    const double du = u_hist[j + 1][i] - u_hist[j][i];
    const double df = f_hist[j + 1][i] - f_hist[j][i];
    val -= gamma[j] * (du + beta * df);
  }
  if (!isfinite(val)) {
    val = g_raw[i];
  }
  te_out[i] = fmax(val, te_floor);
}

__global__ void compute_initial_residual_kernel(const double* __restrict__ rhs,
                                                const double* __restrict__ Ap,
                                                const double* __restrict__ diag_inv,
                                                double* __restrict__ r,
                                                double* __restrict__ z,
                                                double* __restrict__ p,
                                                int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const double ri = finite_or_zero(rhs[i]) - finite_or_zero(Ap[i]);
  const double zi = finite_or_zero(diag_inv[i]) * ri;
  r[i] = ri;
  z[i] = zi;
  p[i] = zi;
}

__global__ void compute_initial_residual_only_kernel(
    const double* __restrict__ rhs,
    const double* __restrict__ Ap,
    double* __restrict__ r,
    int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  r[i] = finite_or_zero(rhs[i]) - finite_or_zero(Ap[i]);
}

__global__ void build_zline_tridiag_from_csr_kernel(
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    double* __restrict__ dl,
    double* __restrict__ d,
    double* __restrict__ du,
    int nr,
    int nz,
    int n_cells,
    int n_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }
  const int base = row * kCsrEntriesPerRow;
  const int local = row % n_cells;
  const int j = local % nz;
  d[row] = values[base + 0];
  dl[row] = (j > 0 && col_indices[base + 3] == row - 1) ? values[base + 3] : 0.0;
  du[row] =
      (j < nz - 1 && col_indices[base + 4] == row + 1) ? values[base + 4] : 0.0;
}

__global__ void extract_fld_2d_csr_stencil_kernel(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    double* __restrict__ diag,
    double* __restrict__ ar_left,
    double* __restrict__ ar_right,
    double* __restrict__ az_m,
    double* __restrict__ az_p,
    int nr,
    int nz,
    int n_groups) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }
  const int base = row_offsets[row];
  diag[row] = values[base + 0];
  ar_left[row] =
      (col_indices[base + 1] == row - nz) ? values[base + 1] : 0.0;
  ar_right[row] =
      (col_indices[base + 2] == row + nz) ? values[base + 2] : 0.0;
  az_m[row] = (col_indices[base + 3] == row - 1) ? values[base + 3] : 0.0;
  az_p[row] = (col_indices[base + 4] == row + 1) ? values[base + 4] : 0.0;
}

__global__ void aggregate_fld_2d_stencil_pairwise_r_galerkin_kernel(
    const double* __restrict__ diag,
    const double* __restrict__ ar_left,
    const double* __restrict__ ar_right,
    const double* __restrict__ az_m,
    const double* __restrict__ az_p,
    double* __restrict__ diag_c,
    double* __restrict__ ar_left_c,
    double* __restrict__ ar_right_c,
    double* __restrict__ az_m_c,
    double* __restrict__ az_p_c,
    int nr,
    int nz,
    int n_groups) {
  const int row_c = blockIdx.x * blockDim.x + threadIdx.x;
  const int nr_c = nr / 2;
  const int n_cells = nr * nz;
  const int n_cells_c = nr_c * nz;
  const int n_rows_c = n_cells_c * n_groups;
  if (row_c >= n_rows_c) {
    return;
  }
  const int g = row_c / n_cells_c;
  const int local_c = row_c - g * n_cells_c;
  const int I = local_c / nz;
  const int j = local_c - I * nz;
  const int row_e = g * n_cells + (2 * I) * nz + j;
  const int row_o = row_e + nz;

  ar_left_c[row_c] = ar_left[row_e];
  ar_right_c[row_c] = ar_right[row_o];
  az_m_c[row_c] = az_m[row_e] + az_m[row_o];
  az_p_c[row_c] = az_p[row_e] + az_p[row_o];
  diag_c[row_c] = diag[row_e] + diag[row_o] + ar_right[row_e] + ar_left[row_o];
}

__global__ void fld_2d_stencil_spmv_kernel(
    const double* __restrict__ diag,
    const double* __restrict__ ar_left,
    const double* __restrict__ ar_right,
    const double* __restrict__ az_m,
    const double* __restrict__ az_p,
    const double* __restrict__ x,
    double* __restrict__ y,
    int nr,
    int nz,
    int n_groups) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }
  const int local = row % n_cells;
  const int i = local / nz;
  const int j = local - i * nz;
  double sum = diag[row] * x[row];
  if (i > 0) {
    sum += ar_left[row] * x[row - nz];
  }
  if (i + 1 < nr) {
    sum += ar_right[row] * x[row + nz];
  }
  if (j > 0) {
    sum += az_m[row] * x[row - 1];
  }
  if (j + 1 < nz) {
    sum += az_p[row] * x[row + 1];
  }
  y[row] = sum;
}

__global__ void fld_2d_stencil_residual_kernel(
    const double* __restrict__ diag,
    const double* __restrict__ ar_left,
    const double* __restrict__ ar_right,
    const double* __restrict__ az_m,
    const double* __restrict__ az_p,
    const double* __restrict__ b,
    const double* __restrict__ x,
    double* __restrict__ r,
    int nr,
    int nz,
    int n_groups) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }
  const int local = row % n_cells;
  const int i = local / nz;
  const int j = local - i * nz;
  double ax = diag[row] * x[row];
  if (i > 0) {
    ax += ar_left[row] * x[row - nz];
  }
  if (i + 1 < nr) {
    ax += ar_right[row] * x[row + nz];
  }
  if (j > 0) {
    ax += az_m[row] * x[row - 1];
  }
  if (j + 1 < nz) {
    ax += az_p[row] * x[row + 1];
  }
  r[row] = b[row] - ax;
}

__global__ void fld_2d_restrict_pairwise_r_kernel(
    const double* __restrict__ r_f,
    double* __restrict__ r_c,
    int nr_fine,
    int nz,
    int n_groups) {
  const int row_c = blockIdx.x * blockDim.x + threadIdx.x;
  const int nr_c = nr_fine / 2;
  const int n_cells_f = nr_fine * nz;
  const int n_cells_c = nr_c * nz;
  const int n_rows_c = n_cells_c * n_groups;
  if (row_c >= n_rows_c) {
    return;
  }
  const int g = row_c / n_cells_c;
  const int local_c = row_c - g * n_cells_c;
  const int I = local_c / nz;
  const int j = local_c - I * nz;
  const int row_e = g * n_cells_f + (2 * I) * nz + j;
  const int row_o = row_e + nz;
  r_c[row_c] = r_f[row_e] + r_f[row_o];
}

__global__ void fld_2d_prolong_add_pairwise_r_kernel(
    const double* __restrict__ e_c,
    double* __restrict__ x_f,
    int nr_fine,
    int nz,
    int n_groups) {
  const int row_c = blockIdx.x * blockDim.x + threadIdx.x;
  const int nr_c = nr_fine / 2;
  const int n_cells_f = nr_fine * nz;
  const int n_cells_c = nr_c * nz;
  const int n_rows_c = n_cells_c * n_groups;
  if (row_c >= n_rows_c) {
    return;
  }
  const int g = row_c / n_cells_c;
  const int local_c = row_c - g * n_cells_c;
  const int I = local_c / nz;
  const int j = local_c - I * nz;
  const int row_e = g * n_cells_f + (2 * I) * nz + j;
  const int row_o = row_e + nz;
  x_f[row_e] += e_c[row_c];
  x_f[row_o] += e_c[row_c];
}

__global__ void fld_2d_axpy_kernel(const double alpha,
                                   const double* __restrict__ x,
                                   double* __restrict__ y,
                                   const int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  y[i] += alpha * x[i];
}

void extract_fld_2d_csr_stencil(const int* row_offsets,
                                const int* col_indices,
                                const double* values,
                                double* diag,
                                double* ar_left,
                                double* ar_right,
                                double* az_m,
                                double* az_p,
                                const int nr,
                                const int nz,
                                const int n_groups) {
  TENRYU_ASSERT(row_offsets != nullptr && col_indices != nullptr &&
                    values != nullptr && diag != nullptr && ar_left != nullptr &&
                    ar_right != nullptr && az_m != nullptr && az_p != nullptr,
                "FLD2D MG CSR stencil extraction requires non-null pointers");
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG CSR stencil extraction requires positive dimensions");
  const int n_rows = nr * nz * n_groups;
  const int grid = (n_rows + kBlock - 1) / kBlock;
  if (grid > 0) {
    extract_fld_2d_csr_stencil_kernel<<<grid, kBlock>>>(row_offsets,
                                                        col_indices,
                                                        values,
                                                        diag,
                                                        ar_left,
                                                        ar_right,
                                                        az_m,
                                                        az_p,
                                                        nr,
                                                        nz,
                                                        n_groups);
    cuda_check(cudaGetLastError(), "FLD2D MG CSR stencil extraction launch failed");
  }
}

void aggregate_fld_2d_stencil_pairwise_r_galerkin(
    const double* diag,
    const double* ar_left,
    const double* ar_right,
    const double* az_m,
    const double* az_p,
    double* diag_c,
    double* ar_left_c,
    double* ar_right_c,
    double* az_m_c,
    double* az_p_c,
    const int nr,
    const int nz,
    const int n_groups) {
  TENRYU_ASSERT(diag != nullptr && ar_left != nullptr && ar_right != nullptr &&
                    az_m != nullptr && az_p != nullptr && diag_c != nullptr &&
                    ar_left_c != nullptr && ar_right_c != nullptr &&
                    az_m_c != nullptr && az_p_c != nullptr,
                "FLD2D MG Galerkin aggregation requires non-null pointers");
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG Galerkin aggregation requires positive dimensions");
  TENRYU_ASSERT((nr % 2) == 0,
                "FLD2D MG Galerkin aggregation requires even nr");
  const int n_rows_c = (nr / 2) * nz * n_groups;
  const int grid = (n_rows_c + kBlock - 1) / kBlock;
  if (grid > 0) {
    aggregate_fld_2d_stencil_pairwise_r_galerkin_kernel<<<grid, kBlock>>>(
        diag,
        ar_left,
        ar_right,
        az_m,
        az_p,
        diag_c,
        ar_left_c,
        ar_right_c,
        az_m_c,
        az_p_c,
        nr,
        nz,
        n_groups);
    cuda_check(cudaGetLastError(), "FLD2D MG Galerkin aggregation launch failed");
  }
}

void fld_2d_stencil_spmv(const double* diag,
                         const double* ar_left,
                         const double* ar_right,
                         const double* az_m,
                         const double* az_p,
                         const double* x,
                         double* y,
                         const int nr,
                         const int nz,
                         const int n_groups) {
  TENRYU_ASSERT(diag != nullptr && ar_left != nullptr && ar_right != nullptr &&
                    az_m != nullptr && az_p != nullptr && x != nullptr &&
                    y != nullptr,
                "FLD2D MG stencil SpMV requires non-null pointers");
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG stencil SpMV requires positive dimensions");
  const int n_rows = nr * nz * n_groups;
  const int grid = (n_rows + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_stencil_spmv_kernel<<<grid, kBlock>>>(diag,
                                                 ar_left,
                                                 ar_right,
                                                 az_m,
                                                 az_p,
                                                 x,
                                                 y,
                                                 nr,
                                                 nz,
                                                 n_groups);
    cuda_check(cudaGetLastError(), "FLD2D MG stencil SpMV launch failed");
  }
}

void fld_2d_stencil_residual(const double* diag,
                             const double* ar_left,
                             const double* ar_right,
                             const double* az_m,
                             const double* az_p,
                             const double* b,
                             const double* x,
                             double* r,
                             const int nr,
                             const int nz,
                             const int n_groups) {
  TENRYU_ASSERT(diag != nullptr && ar_left != nullptr && ar_right != nullptr &&
                    az_m != nullptr && az_p != nullptr && b != nullptr &&
                    x != nullptr && r != nullptr,
                "FLD2D MG stencil residual requires non-null pointers");
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG stencil residual requires positive dimensions");
  const int n_rows = nr * nz * n_groups;
  const int grid = (n_rows + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_stencil_residual_kernel<<<grid, kBlock>>>(diag,
                                                     ar_left,
                                                     ar_right,
                                                     az_m,
                                                     az_p,
                                                     b,
                                                     x,
                                                     r,
                                                     nr,
                                                     nz,
                                                     n_groups);
    cuda_check(cudaGetLastError(), "FLD2D MG stencil residual launch failed");
  }
}

void fld_2d_stencil_residual(cudaStream_t stream,
                             const double* diag,
                             const double* ar_left,
                             const double* ar_right,
                             const double* az_m,
                             const double* az_p,
                             const double* b,
                             const double* x,
                             double* r,
                             const int nr,
                             const int nz,
                             const int n_groups) {
  TENRYU_ASSERT(diag != nullptr && ar_left != nullptr && ar_right != nullptr &&
                    az_m != nullptr && az_p != nullptr && b != nullptr &&
                    x != nullptr && r != nullptr,
                "FLD2D MG stencil residual requires non-null pointers");
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG stencil residual requires positive dimensions");
  const int n_rows = nr * nz * n_groups;
  const int grid = (n_rows + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_stencil_residual_kernel<<<grid, kBlock, 0, stream>>>(diag,
                                                                ar_left,
                                                                ar_right,
                                                                az_m,
                                                                az_p,
                                                                b,
                                                                x,
                                                                r,
                                                                nr,
                                                                nz,
                                                                n_groups);
    cuda_check(cudaGetLastError(), "FLD2D MG stencil residual launch failed");
  }
}

void fld_2d_restrict_pairwise_r(const double* r_f,
                                double* r_c,
                                const int nr_fine,
                                const int nz,
                                const int n_groups) {
  TENRYU_ASSERT(r_f != nullptr && r_c != nullptr,
                "FLD2D MG pairwise-r restriction requires non-null pointers");
  TENRYU_ASSERT(nr_fine > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG pairwise-r restriction requires positive dimensions");
  TENRYU_ASSERT((nr_fine % 2) == 0,
                "FLD2D MG pairwise-r restriction requires even nr");
  const int n_rows_c = (nr_fine / 2) * nz * n_groups;
  const int grid = (n_rows_c + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_restrict_pairwise_r_kernel<<<grid, kBlock>>>(
        r_f, r_c, nr_fine, nz, n_groups);
    cuda_check(cudaGetLastError(),
               "FLD2D MG pairwise-r restriction launch failed");
  }
}

void fld_2d_restrict_pairwise_r(cudaStream_t stream,
                                const double* r_f,
                                double* r_c,
                                const int nr_fine,
                                const int nz,
                                const int n_groups) {
  TENRYU_ASSERT(r_f != nullptr && r_c != nullptr,
                "FLD2D MG pairwise-r restriction requires non-null pointers");
  TENRYU_ASSERT(nr_fine > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG pairwise-r restriction requires positive dimensions");
  TENRYU_ASSERT((nr_fine % 2) == 0,
                "FLD2D MG pairwise-r restriction requires even nr");
  const int n_rows_c = (nr_fine / 2) * nz * n_groups;
  const int grid = (n_rows_c + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_restrict_pairwise_r_kernel<<<grid, kBlock, 0, stream>>>(
        r_f, r_c, nr_fine, nz, n_groups);
    cuda_check(cudaGetLastError(),
               "FLD2D MG pairwise-r restriction launch failed");
  }
}

void fld_2d_prolong_add_pairwise_r(const double* e_c,
                                   double* x_f,
                                   const int nr_fine,
                                   const int nz,
                                   const int n_groups) {
  TENRYU_ASSERT(e_c != nullptr && x_f != nullptr,
                "FLD2D MG pairwise-r prolong-add requires non-null pointers");
  TENRYU_ASSERT(nr_fine > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG pairwise-r prolong-add requires positive dimensions");
  TENRYU_ASSERT((nr_fine % 2) == 0,
                "FLD2D MG pairwise-r prolong-add requires even nr");
  const int n_rows_c = (nr_fine / 2) * nz * n_groups;
  const int grid = (n_rows_c + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_prolong_add_pairwise_r_kernel<<<grid, kBlock>>>(
        e_c, x_f, nr_fine, nz, n_groups);
    cuda_check(cudaGetLastError(),
               "FLD2D MG pairwise-r prolong-add launch failed");
  }
}

void fld_2d_prolong_add_pairwise_r(cudaStream_t stream,
                                   const double* e_c,
                                   double* x_f,
                                   const int nr_fine,
                                   const int nz,
                                   const int n_groups) {
  TENRYU_ASSERT(e_c != nullptr && x_f != nullptr,
                "FLD2D MG pairwise-r prolong-add requires non-null pointers");
  TENRYU_ASSERT(nr_fine > 0 && nz > 0 && n_groups > 0,
                "FLD2D MG pairwise-r prolong-add requires positive dimensions");
  TENRYU_ASSERT((nr_fine % 2) == 0,
                "FLD2D MG pairwise-r prolong-add requires even nr");
  const int n_rows_c = (nr_fine / 2) * nz * n_groups;
  const int grid = (n_rows_c + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_prolong_add_pairwise_r_kernel<<<grid, kBlock, 0, stream>>>(
        e_c, x_f, nr_fine, nz, n_groups);
    cuda_check(cudaGetLastError(),
               "FLD2D MG pairwise-r prolong-add launch failed");
  }
}

void fld_2d_axpy(const double alpha,
                 const double* x,
                 double* y,
                 const int n) {
  TENRYU_ASSERT(x != nullptr && y != nullptr,
                "FLD2D MG axpy requires non-null pointers");
  TENRYU_ASSERT(n >= 0, "FLD2D MG axpy requires non-negative row count");
  const int grid = (n + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_axpy_kernel<<<grid, kBlock>>>(alpha, x, y, n);
    cuda_check(cudaGetLastError(), "FLD2D MG axpy launch failed");
  }
}

void fld_2d_axpy(cudaStream_t stream,
                 const double alpha,
                 const double* x,
                 double* y,
                 const int n) {
  TENRYU_ASSERT(x != nullptr && y != nullptr,
                "FLD2D MG axpy requires non-null pointers");
  TENRYU_ASSERT(n >= 0, "FLD2D MG axpy requires non-negative row count");
  const int grid = (n + kBlock - 1) / kBlock;
  if (grid > 0) {
    fld_2d_axpy_kernel<<<grid, kBlock, 0, stream>>>(alpha, x, y, n);
    cuda_check(cudaGetLastError(), "FLD2D MG axpy launch failed");
  }
}

__global__ void dot_single_block_kernel(const double* __restrict__ a,
                                        const double* __restrict__ b,
                                        double* __restrict__ out,
                                        int n) {
  // Single-block fixed-order reduction: deterministic run-to-run and no
  // cross-block atomic accumulation.
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  double local = 0.0;
  for (int i = tid; i < n; i += blockDim.x) {
    local += finite_or_zero(a[i]) * finite_or_zero(b[i]);
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *out = shared[0];
  }
}

__global__ void dot_partials_kernel(const double* __restrict__ a,
                                    const double* __restrict__ b,
                                    double* __restrict__ partials,
                                    int n) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  shared[tid] =
      (i < n) ? finite_or_zero(a[i]) * finite_or_zero(b[i]) : 0.0;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void dot_finalize_kernel(const double* __restrict__ partials,
                                    double* __restrict__ out,
                                    int nblocks) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  double local = 0.0;
  for (int i = tid; i < nblocks; i += blockDim.x) {
    local += partials[i];
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    *out = shared[0];
  }
}

// Owned-rows dot partials for the Option-C distributed CG: solver vectors
// are group-major (row = g*n_cells + c), so the owned rows form one
// contiguous cell range per group. At a full window (c_begin=0,
// owned=n_cells) the thread->element map degenerates to dot_partials_kernel,
// but single-rank runs keep the original kernels entirely (P==1 bit gate).
__global__ void owned_offsets_shift_kernel(const int* __restrict__ src,
                                           int* __restrict__ dst,
                                           const int row_lo,
                                           const int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i <= count) {
    dst[i] = src[row_lo + i] - src[row_lo];
  }
}

__global__ void dot_owned_partials_kernel(const double* __restrict__ a,
                                          const double* __restrict__ b,
                                          double* __restrict__ partials,
                                          int n_cells,
                                          int c_begin,
                                          int owned,
                                          int total) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  double v = 0.0;
  if (idx < total) {
    const int g = idx / owned;
    const int row = g * n_cells + c_begin + (idx - g * owned);
    v = finite_or_zero(a[row]) * finite_or_zero(b[row]);
  }
  shared[tid] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    partials[blockIdx.x] = shared[0];
  }
}

__global__ void cg_update_x_r_kernel(double* __restrict__ x,
                                     double* __restrict__ r,
                                     const double* __restrict__ p,
                                     const double* __restrict__ Ap,
                                     const double* __restrict__ d_rz,
                                     const double* __restrict__ d_pAp,
                                     CgDeviceStatus* __restrict__ status,
                                     int iter,
                                     int n) {
  __shared__ double alpha_s;
  __shared__ int skip_update_s;
  if (threadIdx.x == 0) {
    const int prior_breakdown = status->breakdown_iter;
    const double pAp_s = finite_or_zero(*d_pAp);
    const double rz_s = finite_or_zero(*d_rz);
    const bool breakdown = !(fabs(pAp_s) > 1.0e-300);
    const double alpha_value = breakdown ? 0.0 : rz_s / pAp_s;
    alpha_s = alpha_value;
    skip_update_s = (prior_breakdown >= 0 || breakdown) ? 1 : 0;
    if (blockIdx.x == 0 && prior_breakdown < 0) {
      if (isfinite(pAp_s)) {
        status->min_pAp = fmin(status->min_pAp, pAp_s);
        if (pAp_s <= 0.0) {
          ++status->count_nonpos_pAp;
        }
      } else {
        ++status->count_nonfinite;
      }
      if (!breakdown && (!isfinite(alpha_value) || !isfinite(rz_s))) {
        ++status->count_nonfinite;
      }
      if (breakdown) {
        atomicCAS(&status->breakdown_iter, -1, iter);
      }
    }
  }
  __syncthreads();
  if (skip_update_s != 0) {
    return;
  }
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  // Phase 2a-1.5: removed in-loop fmax(..., 0.0) clamp on x.
  // Clamp inside CG breaks the invariant r = b - A x because r is updated
  // by the unclamped linear recurrence below. Positivity is now enforced
  // at publish time with diagnostics.
  x[i] = finite_or_zero(x[i]) + alpha_s * finite_or_zero(p[i]);
  r[i] = finite_or_zero(r[i]) - alpha_s * finite_or_zero(Ap[i]);
}

__global__ void cg_apply_preconditioner_kernel(const double* __restrict__ r,
                                               const double* __restrict__ diag_inv,
                                               double* __restrict__ z,
                                               int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    z[i] = finite_or_zero(diag_inv[i]) * finite_or_zero(r[i]);
  }
}

__global__ void cg_update_p_kernel(double* __restrict__ p,
                                   const double* __restrict__ z,
                                   const double* __restrict__ d_rz_next,
                                   const double* __restrict__ d_rz,
                                   CgDeviceStatus* __restrict__ status,
                                   int n) {
  __shared__ double beta_s;
  __shared__ int skip_update_s;
  if (threadIdx.x == 0) {
    const int prior_breakdown = status->breakdown_iter;
    const double rzn_s = finite_or_zero(*d_rz_next);
    const double rz_s = finite_or_zero(*d_rz);
    const double beta_value = (fabs(rz_s) > 1.0e-300) ? (rzn_s / rz_s) : 0.0;
    beta_s = beta_value;
    skip_update_s = (prior_breakdown >= 0) ? 1 : 0;
    if (blockIdx.x == 0 && prior_breakdown < 0 &&
        (!isfinite(beta_value) || !isfinite(rzn_s))) {
      ++status->count_nonfinite;
    }
  }
  __syncthreads();
  if (skip_update_s != 0) {
    return;
  }
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    p[i] = finite_or_zero(z[i]) + beta_s * finite_or_zero(p[i]);
  }
}

__global__ void cg_bump_iter_kernel(int* __restrict__ d_iter) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    ++(*d_iter);
  }
}

// Graph-path variant of cg_update_x_r_kernel. Kernel parameters are baked
// into an instantiated CUDA graph, so the changing iteration index cannot
// be passed by value; it is read from a device counter instead
// (cg_bump_iter_kernel increments it inside the same graph). The body is
// kept VERBATIM from cg_update_x_r_kernel except that `iter` is loaded
// from *d_iter_p; the eager kernel stays untouched so the fallback path
// is byte-identical to the pre-W1 binary.
__global__ void cg_update_x_r_from_dev_kernel(
    double* __restrict__ x,
    double* __restrict__ r,
    const double* __restrict__ p,
    const double* __restrict__ Ap,
    const double* __restrict__ d_rz,
    const double* __restrict__ d_pAp,
    CgDeviceStatus* __restrict__ status,
    const int* __restrict__ d_iter_p,
    int n) {
  __shared__ double alpha_s;
  __shared__ int skip_update_s;
  if (threadIdx.x == 0) {
    const int iter = *d_iter_p;
    const int prior_breakdown = status->breakdown_iter;
    const double pAp_s = finite_or_zero(*d_pAp);
    const double rz_s = finite_or_zero(*d_rz);
    const bool breakdown = !(fabs(pAp_s) > 1.0e-300);
    const double alpha_value = breakdown ? 0.0 : rz_s / pAp_s;
    alpha_s = alpha_value;
    skip_update_s = (prior_breakdown >= 0 || breakdown) ? 1 : 0;
    if (blockIdx.x == 0 && prior_breakdown < 0) {
      if (isfinite(pAp_s)) {
        status->min_pAp = fmin(status->min_pAp, pAp_s);
        if (pAp_s <= 0.0) {
          ++status->count_nonpos_pAp;
        }
      } else {
        ++status->count_nonfinite;
      }
      if (!breakdown && (!isfinite(alpha_value) || !isfinite(rz_s))) {
        ++status->count_nonfinite;
      }
      if (breakdown) {
        atomicCAS(&status->breakdown_iter, -1, iter);
      }
    }
  }
  __syncthreads();
  if (skip_update_s != 0) {
    return;
  }
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  // Phase 2a-1.5 contract as in cg_update_x_r_kernel: no in-loop clamp.
  x[i] = finite_or_zero(x[i]) + alpha_s * finite_or_zero(p[i]);
  r[i] = finite_or_zero(r[i]) - alpha_s * finite_or_zero(Ap[i]);
}

__global__ void publish_with_projection_kernel(double* __restrict__ x,
                                               double* __restrict__ rad_E,
                                               const double* __restrict__ vol,
                                               double* __restrict__ clamp_hits,
                                               double* __restrict__ clamp_energy_delta,
                                               double* __restrict__ min_x_raw,
                                               const int c_begin,
                                               const int c_end,
                                               int n_cells,
                                               int n_groups) {
  // Solver rows are group-major (row = g*n_cells + c): owned rows are
  // strided per group, so the window iterates (group x owned-cell) pairs.
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  const int owned = c_end - c_begin;
  if (owned <= 0 || k >= owned * n_groups) {
    return;
  }
  const int g = k / owned;
  const int c = c_begin + (k - g * owned);
  const int row = g * n_cells + c;
  const double x_raw = finite_or_zero(x[row]);
  const double x_pub = fmax(x_raw, 0.0);
  if (x_raw < 0.0) {
    atomicAdd(clamp_hits, 1.0);
    atomicAdd(clamp_energy_delta,
              fmax(finite_or_zero(vol[c]), 0.0) * (x_pub - x_raw));
    atomic_min_double(min_x_raw, x_raw);
  }
  x[row] = x_pub;
  rad_E[c * n_groups + g] = x_pub;
}

__global__ void dcell_stats_kernel(const double* __restrict__ D_cell,
                                   double* __restrict__ out,
                                   int* __restrict__ counts,
                                   int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  const double value = D_cell[idx];
  if (!isfinite(value)) {
    atomicAdd(counts + 1, 1);
    return;
  }
  atomic_min_double(out, value);
  atomic_max_nonnegative_double(out + 1, value);
  if (value == 0.0) {
    atomicAdd(counts, 1);
  }
}

__global__ void csr_matrix_stats_kernel(const int* __restrict__ col_indices,
                                        const double* __restrict__ values,
                                        const double* __restrict__ rhs,
                                        double* __restrict__ out,
                                        int* __restrict__ counts,
                                        int n_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }
  const int base = row * kCsrEntriesPerRow;
  const double diag = values[base];
  double offdiag_abs_sum = 0.0;
  int nonfinite_count = 0;
  if (!isfinite(diag)) {
    ++nonfinite_count;
  }
  if (!isfinite(rhs[row])) {
    ++nonfinite_count;
  }
  for (int slot = 1; slot < kCsrEntriesPerRow; ++slot) {
    const double aij = values[base + slot];
    if (isfinite(aij)) {
      offdiag_abs_sum += fabs(aij);
    } else {
      ++nonfinite_count;
    }
  }
  if (nonfinite_count != 0) {
    atomicAdd(counts + 1, nonfinite_count);
  }
  if (diag > 0.0 && isfinite(diag)) {
    atomic_min_double(out, diag);
    atomic_max_nonnegative_double(out + 1, diag);
    if (diag - offdiag_abs_sum <= 0.0) {
      atomicAdd(counts, 1);
    }
    double s_i = 0.0;
    for (int slot = 1; slot < kCsrEntriesPerRow; ++slot) {
      const int col = col_indices[base + slot];
      if (col == row || col < 0 || col >= n_rows) {
        continue;
      }
      const double aij = values[base + slot];
      const double ajj = values[col * kCsrEntriesPerRow];
      if (isfinite(aij) && isfinite(ajj) && ajj > 0.0) {
        s_i += fabs(aij) / sqrt(diag * ajj);
      }
    }
    atomic_min_double(out + 2, 1.0 - s_i);
    atomic_max_nonnegative_double(out + 3, 1.0 + s_i);
  }
}

__global__ void pair_symmetry_audit_kernel(const int* __restrict__ col_indices,
                                           const double* __restrict__ values,
                                           double* __restrict__ max_diff,
                                           int* __restrict__ violation_count,
                                           int n_rows) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_rows * (kCsrEntriesPerRow - 1);
  if (idx >= total) {
    return;
  }
  const int row = idx / (kCsrEntriesPerRow - 1);
  const int slot = 1 + idx - row * (kCsrEntriesPerRow - 1);
  const int base = row * kCsrEntriesPerRow;
  const int col = col_indices[base + slot];
  if (col == row || col < 0 || col >= n_rows) {
    return;
  }
  const double aij = values[base + slot];
  double aji = 0.0;
  const int col_base = col * kCsrEntriesPerRow;
  bool found_back = false;
  for (int back_slot = 1; back_slot < kCsrEntriesPerRow; ++back_slot) {
    if (col_indices[col_base + back_slot] == row) {
      aji = values[col_base + back_slot];
      found_back = true;
      break;
    }
  }
  if (!found_back && aij == 0.0) {
    return;
  }
  const double diff = fabs(finite_or_zero(aij) - finite_or_zero(aji));
  atomic_max_nonnegative_double(max_diff, diff);
  const double scale = fmax(fmax(fabs(finite_or_zero(aij)),
                                fabs(finite_or_zero(aji))),
                           1.0);
  if (diff > 1.0e-12 * scale) {
    atomicAdd(violation_count, 1);
  }
}

__global__ void compute_operator_residual_sum_kernel(
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ x,
    const double* __restrict__ rhs,
    double* __restrict__ sum_residual,
    int n_rows) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n_rows) {
    return;
  }
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  double Ax = 0.0;
  for (int k = row_start; k < row_end; ++k) {
    const int col = col_indices[k];
    if (col >= 0 && col < n_rows) {
      Ax += finite_or_zero(values[k]) * finite_or_zero(x[col]);
    }
  }
  atomicAdd(sum_residual, Ax - finite_or_zero(rhs[row]));
}

__global__ void compute_rad_delta_sum_kernel(const double* __restrict__ rad_E,
                                             const double* __restrict__ rad_E_old,
                                             const double* __restrict__ vol,
                                             double* __restrict__ sum_delta,
                                             int n_cells,
                                             int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const double delta =
      fmax(finite_or_zero(vol[c]), 0.0) *
      (finite_or_zero(rad_E[idx]) - finite_or_zero(rad_E_old[idx]));
  atomicAdd(sum_delta, delta);
}

__global__ void compute_emit_minus_dep_sum_kernel(
    const double* __restrict__ rad_emit,
    const double* __restrict__ rad_dep,
    double* __restrict__ sum_emit_minus_dep,
    int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  atomicAdd(sum_emit_minus_dep,
            finite_or_zero(rad_emit[idx]) - finite_or_zero(rad_dep[idx]));
}

__global__ void compute_per_row_defect_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_a,
    const double* __restrict__ eta,
    const double* __restrict__ fleck,
    const double* __restrict__ rad_E,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ D_cell,
    int nr,
    int nz,
    int n_groups,
    double dt,
    int outer_r_bc,
    int z_bottom_bc,
    int z_top_bc,
    double T_supply_z_bottom_eV,
    double T_supply_z_top_eV,
    double* __restrict__ sum_defect,
    FldPerRowDefectRecord* __restrict__ max_record,
    double* __restrict__ sum_vol_diff,
    int* __restrict__ counts,
    const double* __restrict__ rho = nullptr,
    const double* __restrict__ sigma_R = nullptr,
    int state_supply_boundary_policy =
        kStateSupplyBoundaryPolicyLocalDCurrent,
    int limiter = 0,
    double rho_supply_z_bottom = 0.0,
    double rho_supply_z_top = 0.0) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }

  const int g = row / n_cells;
  const int c = row - g * n_cells;
  const int i = c / nz;
  const int j = c - i * nz;
  const int cg = c * n_groups + g;

  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  const double V_op = geom.V_op;

  const double V_state = nonnegative_finite(vol[c]);
  const double vol_diff = fabs(V_op - V_state);
  if (vol_diff > 1.0e-30 * fmax(V_op, V_state)) {
    atomicAdd(counts, 1);
    atomicAdd(sum_vol_diff, vol_diff);
  }

  const double sigma_pa_value = nonnegative_finite(sigma_a[cg]);
  const double E = nonnegative_finite(rad_E[cg]);
  const double E_old = nonnegative_finite(rad_E_old[cg]);
  const double source = fld_rhs_source_rate(eta, fleck, sigma_a, rad_E_old, cg, c);
  const double dep = dt * V_op * core::constants::c_light * sigma_pa_value * E;
  const double emit = dt * V_op * source;
  const double rad_delta = V_op * (E - E_old);

  double escape = 0.0;
  if (i + 1 == nr) {
    escape += dt * geom.area_iR * fld_boundary_leakage_coeff(outer_r_bc) * E;
  }
  if (j == 0) {
    if (z_bottom_bc == kFldBcStateSupply) {
      const double coeff =
          fld_state_supply_boundary_coeff(
              geom,
              D_cell[cg],
              false,
              state_supply_boundary_policy,
              rho[c],
              (sigma_R != nullptr) ? sigma_R[cg] : 0.0,
              rho_supply_z_bottom,
              D_cell,
              nr,
              nz,
              n_groups,
              0,
              g,
              limiter);
      escape += dt * coeff * (E - fld_state_supply_E(T_supply_z_bottom_eV));
    } else {
      escape += dt * geom.area_jB * fld_boundary_leakage_coeff(z_bottom_bc) * E;
    }
  }
  if (j + 1 == nz) {
    if (z_top_bc == kFldBcStateSupply) {
      const double coeff =
          fld_state_supply_boundary_coeff(
              geom,
              D_cell[cg],
              true,
              state_supply_boundary_policy,
              rho[c],
              (sigma_R != nullptr) ? sigma_R[cg] : 0.0,
              rho_supply_z_top,
              D_cell,
              nr,
              nz,
              n_groups,
              nz - 1,
              g,
              limiter);
      escape += dt * coeff * (E - fld_state_supply_E(T_supply_z_top_eV));
    } else {
      escape += dt * geom.area_jT * fld_boundary_leakage_coeff(z_top_bc) * E;
    }
  }

  const double row_defect = rad_delta + escape - (emit - dep);
  const double abs_defect = fabs(row_defect);
  atomicAdd(sum_defect, row_defect);
  if (abs_defect > 1.0e-3) {
    atomicAdd(counts + 1, 1);
  }
  if (abs_defect > 1.0e-2) {
    atomicAdd(counts + 2, 1);
  }
  if (abs_defect > 1.0e-1) {
    atomicAdd(counts + 3, 1);
  }
  if (abs_defect > 1.0) {
    atomicAdd(counts + 4, 1);
  }
  if (abs_defect > 10.0) {
    atomicAdd(counts + 5, 1);
  }
  atomic_max_per_row_defect_record(max_record,
                                   abs_defect,
                                   row_defect,
                                   V_op,
                                   V_state,
                                   c,
                                   g);
}

__global__ void compute_row_identity_diagnostic_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ sigma_a,
    const double* __restrict__ eta,
    const double* __restrict__ fleck,
    const double* __restrict__ rad_E,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ rad_emit,
    const double* __restrict__ rad_dep,
    const double* __restrict__ D_cell,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const double* __restrict__ x_solution,
    const double* __restrict__ rhs,
    int nr,
    int nz,
    int n_groups,
    double dt,
    int outer_r_bc,
    int z_bottom_bc,
    int z_top_bc,
    double marshak_flux,
    double T_supply_z_bottom_eV,
    double T_supply_z_top_eV,
    double* __restrict__ sums,
    RogueRecord* __restrict__ records,
    const double* __restrict__ rho = nullptr,
    const double* __restrict__ sigma_R = nullptr,
    int state_supply_boundary_policy =
        kStateSupplyBoundaryPolicyLocalDCurrent,
    int limiter = 0,
    double rho_supply_z_bottom = 0.0,
    double rho_supply_z_top = 0.0) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }

  const int g = row / n_cells;
  const int c = row - g * n_cells;
  const int i = c / nz;
  const int j = c - i * nz;
  const int cg = c * n_groups + g;

  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  const double rc = geom.r_center;
  const double zc = geom.z_center;
  const double V_op = geom.V_op;

  const double V_state = nonnegative_finite(vol[c]);
  (void)V_state;
  const double sigma_pa = nonnegative_finite(sigma_a[cg]);
  const double sigma_removal = sigma_pa;
  const double E_old = nonnegative_finite(rad_E_old[cg]);
  const double E_new = nonnegative_finite(rad_E[cg]);
  const double eta_v = nonnegative_finite(eta[cg]);
  // BUG-11: mirror the assembly's cg read (fleck is a [n_cells x n_groups]
  // array; layout contract in nlte_coeffs.cu) so this diagnostic recomputes
  // exactly what assemble_fld_2d_csr_kernel assembles for G > 1.
  const double f = (fleck != nullptr)
                       ? fmin(fmax(finite_or_zero(fleck[cg]), 0.0), 1.0)
                       : 1.0;
  const double source = f * eta_v +
                        (1.0 - f) * core::constants::c_light * sigma_pa * E_old;
  const double rho_cell = (rho != nullptr) ? rho[c] : 0.0;
  const double sigma_R_cell = (sigma_R != nullptr) ? sigma_R[cg] : 0.0;
  const double state_supply_z_bottom_coeff =
      (j == 0 && z_bottom_bc == kFldBcStateSupply)
          ? fld_state_supply_boundary_coeff(geom,
                                            D_cell[cg],
                                            false,
                                            state_supply_boundary_policy,
                                            rho_cell,
                                            sigma_R_cell,
                                            rho_supply_z_bottom,
                                            D_cell,
                                            nr,
                                            nz,
                                            n_groups,
                                            0,
                                            g,
                                            limiter)
          : 0.0;
  const double state_supply_z_top_coeff =
      (j + 1 == nz && z_top_bc == kFldBcStateSupply)
          ? fld_state_supply_boundary_coeff(geom,
                                            D_cell[cg],
                                            true,
                                            state_supply_boundary_policy,
                                            rho_cell,
                                            sigma_R_cell,
                                            rho_supply_z_top,
                                            D_cell,
                                            nr,
                                            nz,
                                            n_groups,
                                            nz - 1,
                                            g,
                                            limiter)
          : 0.0;

  double boundary_diag = 0.0;
  if (i + 1 == nr) {
    boundary_diag += dt * geom.area_iR * fld_boundary_leakage_coeff(outer_r_bc);
  }
  if (j == 0) {
    if (z_bottom_bc == kFldBcStateSupply) {
      boundary_diag += dt * state_supply_z_bottom_coeff;
    } else {
      boundary_diag += dt * geom.area_jB * fld_boundary_leakage_coeff(z_bottom_bc);
    }
  }
  if (j + 1 == nz) {
    if (z_top_bc == kFldBcStateSupply) {
      boundary_diag += dt * state_supply_z_top_coeff;
    } else {
      boundary_diag += dt * geom.area_jT * fld_boundary_leakage_coeff(z_top_bc);
    }
  }

  double boundary_source = 0.0;
  if (j == 0 && z_bottom_bc == kFldBcMarshak) {
    boundary_source += dt * geom.area_jB * marshak_flux;
  } else if (j == 0 && z_bottom_bc == kFldBcStateSupply) {
    boundary_source +=
        dt * state_supply_z_bottom_coeff *
        fld_state_supply_E(T_supply_z_bottom_eV);
  }
  if (j + 1 == nz && z_top_bc == kFldBcMarshak) {
    boundary_source += dt * geom.area_jT * marshak_flux;
  } else if (j + 1 == nz && z_top_bc == kFldBcStateSupply) {
    boundary_source += dt * state_supply_z_top_coeff *
                       fld_state_supply_E(T_supply_z_top_eV);
  }

  double face_div = 0.0;
  if (i > 0) {
    const int cn = cell_index(i - 1, j, nz);
    const double rn = rc_cache[cn];
    const double raw_dist = rc - rn;
    if (raw_dist > kFldFaceDistMin) {
      const double Df = harmonic_positive(D_cell[cn * n_groups + g], D_cell[cg],
                                          nullptr);
      const double coef = dt * geom.area_iL * Df / raw_dist;
      if (isfinite(coef) && coef > 0.0) {
        face_div +=
            coef * (E_new - nonnegative_finite(rad_E[cn * n_groups + g]));
      }
    }
  }
  if (i + 1 < nr) {
    const int cn = cell_index(i + 1, j, nz);
    const double rn = rc_cache[cn];
    const double raw_dist = rn - rc;
    if (raw_dist > kFldFaceDistMin) {
      const double Df = harmonic_positive(D_cell[cg], D_cell[cn * n_groups + g],
                                          nullptr);
      const double coef = dt * geom.area_iR * Df / raw_dist;
      if (isfinite(coef) && coef > 0.0) {
        face_div +=
            coef * (E_new - nonnegative_finite(rad_E[cn * n_groups + g]));
      }
    }
  }
  if (j > 0) {
    const int cn = cell_index(i, j - 1, nz);
    const double zn = zc_cache[cn];
    const double raw_dist = zc - zn;
    if (raw_dist > kFldFaceDistMin) {
      const double Df = harmonic_positive(D_cell[cn * n_groups + g], D_cell[cg],
                                          nullptr);
      const double coef = dt * geom.area_jB * Df / raw_dist;
      if (isfinite(coef) && coef > 0.0) {
        face_div +=
            coef * (E_new - nonnegative_finite(rad_E[cn * n_groups + g]));
      }
    }
  }
  if (j + 1 < nz) {
    const int cn = cell_index(i, j + 1, nz);
    const double zn = zc_cache[cn];
    const double raw_dist = zn - zc;
    if (raw_dist > kFldFaceDistMin) {
      const double Df = harmonic_positive(D_cell[cg], D_cell[cn * n_groups + g],
                                          nullptr);
      const double coef = dt * geom.area_jT * Df / raw_dist;
      if (isfinite(coef) && coef > 0.0) {
        face_div +=
            coef * (E_new - nonnegative_finite(rad_E[cn * n_groups + g]));
      }
    }
  }

  const double matrix_LHS_per_row =
      (V_op + dt * core::constants::c_light * sigma_removal * V_op +
       boundary_diag) *
          E_new +
      face_div;
  const double matrix_b_per_row =
      V_op * E_old + dt * V_op * source + boundary_source;
  const double matrix_formula_residual =
      matrix_LHS_per_row - matrix_b_per_row;

  double csr_Ax = 0.0;
  const int row_start = row_offsets[row];
  const int row_end = row_offsets[row + 1];
  for (int k = row_start; k < row_end; ++k) {
    const int col = col_indices[k];
    if (col >= 0 && col < n_rows) {
      csr_Ax += finite_or_zero(values[k]) * finite_or_zero(x_solution[col]);
    }
  }
  const double csr_residual = csr_Ax - finite_or_zero(rhs[row]);

  const double rad_delta = V_op * (E_new - E_old);
  const double escape_per_row = boundary_diag * E_new;
  const double dep_kernel = nonnegative_finite(rad_dep[cg]);
  const double emit_kernel = nonnegative_finite(rad_emit[cg]);
  const double tally_formula_residual =
      rad_delta + dep_kernel + escape_per_row + face_div -
      emit_kernel - boundary_source;

  const double emit_formula = dt * V_op * source;
  const double dep_formula =
      dt * V_op * core::constants::c_light * sigma_pa * E_new;

  const double delta_csr_vs_matrix =
      csr_residual - matrix_formula_residual;
  const double delta_matrix_vs_tally =
      matrix_formula_residual - tally_formula_residual;
  const double delta_emit_kernel_vs_formula =
      emit_kernel - emit_formula;
  const double delta_dep_kernel_vs_formula =
      dep_kernel - dep_formula;

  atomicAdd(sums + 0, csr_residual);
  atomicAdd(sums + 1, matrix_formula_residual);
  atomicAdd(sums + 2, tally_formula_residual);
  atomicAdd(sums + 3, face_div);
  atomicAdd(sums + 4, fabs(face_div));
  atomicAdd(sums + 5, emit_kernel);
  atomicAdd(sums + 6, dep_kernel);
  atomicAdd(sums + 7, emit_formula);
  atomicAdd(sums + 8, dep_formula);

  atomic_max_record(records, 0, fabs(delta_csr_vs_matrix), c, g);
  atomic_max_record(records, 1, fabs(delta_matrix_vs_tally), c, g);
  atomic_max_record(records, 2, fabs(delta_emit_kernel_vs_formula), c, g);
  atomic_max_record(records, 3, fabs(delta_dep_kernel_vs_formula), c, g);
}

__device__ inline void fill_face_trace_side_record(
    const double* __restrict__ rho,
    const double* __restrict__ rad_E,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ D_cell,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const int n_cells,
    const int n_groups,
    const int nz,
    const int row,
    const int g,
    const int c,
    const int i,
    const int j,
    const int cn,
    const int ni,
    const int nj,
    const int side,
    const int slot,
    const double area,
    const double raw_dist,
    const double dt,
    FldFaceTraceSideRecord* __restrict__ rec) {
  rec->valid = 1;
  rec->side = side;
  rec->cell_i = i;
  rec->cell_j = j;
  rec->neighbor_i = ni;
  rec->neighbor_j = nj;
  rec->group = g;
  const int cg = c * n_groups + g;
  const int cng = cn * n_groups + g;
  const double D_self = D_cell[cg];
  const double D_neighbor = D_cell[cng];
  const bool degenerate = !(raw_dist > kFldFaceDistMin);
  const double D_face = harmonic_positive(D_self, D_neighbor, nullptr);
  const double coef = degenerate ? 0.0 : dt * area * D_face / raw_dist;
  const int base = row * kCsrEntriesPerRow;
  const double csr_value = values[base + slot];
  const int csr_col = col_indices[base + slot];
  const bool coef_ok = isfinite(coef) && coef > 0.0;
  const double expected = coef_ok ? -coef : 0.0;
  const double diff = fabs(finite_or_zero(csr_value) - expected);
  const double scale = fmax(fmax(fabs(expected), fabs(finite_or_zero(csr_value))),
                           1.0);
  rec->csr_col = csr_col;
  rec->coef_finite = isfinite(coef) ? 1 : 0;
  rec->coef_fallback = coef_ok ? 0 : 1;
  rec->csr_mismatch = (diff > 1.0e-12 * scale) ? 1 : 0;
  rec->D_self_finite = isfinite(D_self) ? 1 : 0;
  rec->D_neighbor_finite = isfinite(D_neighbor) ? 1 : 0;
  rec->D_face_finite = isfinite(D_face) ? 1 : 0;
  rec->D_self = finite_or_zero(D_self);
  rec->D_neighbor = finite_or_zero(D_neighbor);
  rec->D_face = finite_or_zero(D_face);
  rec->area_face = finite_or_zero(area);
  rec->dist_face = finite_or_zero(raw_dist);
  rec->coef = finite_or_zero(coef);
  rec->csr_value = finite_or_zero(csr_value);
  rec->expected_csr_value = expected;
  rec->self_rho = nonnegative_finite(rho[c]);
  rec->self_sigma = nonnegative_finite(sigma_removal[cg]);
  rec->self_rad_E = nonnegative_finite(rad_E[cg]);
  rec->neighbor_rho = nonnegative_finite(rho[cn]);
  rec->neighbor_sigma = nonnegative_finite(sigma_removal[cng]);
  rec->neighbor_rad_E = nonnegative_finite(rad_E[cng]);
  (void)n_cells;
}

__global__ void compute_face_coef_trace_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ rho,
    const double* __restrict__ rad_E,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ D_cell,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    const int target_cell,
    FldFaceTraceSideRecord* __restrict__ records) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int n_cells = nr * nz;
  if (target_cell < 0 || target_cell >= n_cells || n_groups <= 0) {
    return;
  }
  const int g = 0;
  const int c = target_cell;
  const int i = c / nz;
  const int j = c - i * nz;
  const int row = g * n_cells + c;
  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  const double rc = geom.r_center;
  const double zc = geom.z_center;
  for (int side = 0; side < 4; ++side) {
    records[side].valid = 0;
    records[side].side = side;
    records[side].cell_i = i;
    records[side].cell_j = j;
    records[side].neighbor_i = -1;
    records[side].neighbor_j = -1;
    records[side].group = g;
    records[side].csr_col = col_indices[row * kCsrEntriesPerRow + side + 1];
    records[side].csr_value =
        finite_or_zero(values[row * kCsrEntriesPerRow + side + 1]);
  }
  if (i > 0) {
    const int cn = cell_index(i - 1, j, nz);
    const double dist = rc - rc_cache[cn];
    fill_face_trace_side_record(rho, rad_E, sigma_removal, D_cell, col_indices,
                                values, n_cells, n_groups, nz, row, g, c, i,
                                j, cn, i - 1, j, 0, 1, geom.area_iL, dist, dt,
                                records + 0);
  }
  if (i + 1 < nr) {
    const int cn = cell_index(i + 1, j, nz);
    const double dist = rc_cache[cn] - rc;
    fill_face_trace_side_record(rho, rad_E, sigma_removal, D_cell, col_indices,
                                values, n_cells, n_groups, nz, row, g, c, i,
                                j, cn, i + 1, j, 1, 2, geom.area_iR, dist, dt,
                                records + 1);
  }
  if (j > 0) {
    const int cn = cell_index(i, j - 1, nz);
    const double dist = zc - zc_cache[cn];
    fill_face_trace_side_record(rho, rad_E, sigma_removal, D_cell, col_indices,
                                values, n_cells, n_groups, nz, row, g, c, i,
                                j, cn, i, j - 1, 2, 3, geom.area_jB, dist, dt,
                                records + 2);
  }
  if (j + 1 < nz) {
    const int cn = cell_index(i, j + 1, nz);
    const double dist = zc_cache[cn] - zc;
    fill_face_trace_side_record(rho, rad_E, sigma_removal, D_cell, col_indices,
                                values, n_cells, n_groups, nz, row, g, c, i,
                                j, cn, i, j + 1, 3, 4, geom.area_jT, dist, dt,
                                records + 3);
  }
}

__global__ void compute_face_global_max_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ D_cell,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    RogueRecord* __restrict__ max_D_cell,
    FldFaceGlobalMaxRecord* __restrict__ max_D_face,
    FldFaceGlobalMaxRecord* __restrict__ max_coef,
    FldFaceGlobalMaxRecord* __restrict__ max_area_dist_ratio) {
  if (threadIdx.x != 0 || blockIdx.x != 0) {
    return;
  }
  const int n_cells = nr * nz;
  const int n_r_faces = (nr > 1) ? ((nr - 1) * nz) : 0;
  const int n_z_faces = (nz > 1) ? (nr * (nz - 1)) : 0;
  const int face_total = (n_r_faces + n_z_faces) * n_groups;
  const int cell_total = n_cells * n_groups;
  const int total = cell_total + face_total;
  for (int idx = 0; idx < total; ++idx) {
    if (idx < cell_total) {
      const int c = idx / n_groups;
      const int g = idx - c * n_groups;
      const double value = D_cell[idx];
      if (isfinite(value) && value > max_D_cell->value) {
        max_D_cell->value = value;
        max_D_cell->cell_idx = c;
        max_D_cell->group_idx = g;
      }
      continue;
    }

    const int face_idx = idx - cell_total;
    const int face = face_idx / n_groups;
    const int g = face_idx - face * n_groups;
    int c0 = -1;
    int c1 = -1;
    int i0 = -1;
    int j0 = -1;
    int i1 = -1;
    int j1 = -1;
    int side = -1;
    double area = 0.0;
    double dist = 0.0;
    if (face < n_r_faces) {
      const int f = face;
      i0 = f / nz;
      j0 = f - i0 * nz;
      i1 = i0 + 1;
      j1 = j0;
      side = 1;
      c0 = cell_index(i0, j0, nz);
      c1 = cell_index(i1, j1, nz);
      const CellGeometryRZ geom =
          rect_cell_geometry_v2(x_r, x_z, vol, i0, j0, nz);
      area = geom.area_iR;
      dist = rc_cache[c1] - rc_cache[c0];
    } else {
      const int f = face - n_r_faces;
      i0 = f / (nz - 1);
      j0 = f - i0 * (nz - 1);
      i1 = i0;
      j1 = j0 + 1;
      side = 3;
      c0 = cell_index(i0, j0, nz);
      c1 = cell_index(i1, j1, nz);
      const CellGeometryRZ geom =
          rect_cell_geometry_v2(x_r, x_z, vol, i0, j0, nz);
      area = geom.area_jT;
      dist = zc_cache[c1] - zc_cache[c0];
    }
    if (!(dist > kFldFaceDistMin)) {
      continue;
    }
    const double D_self = D_cell[c0 * n_groups + g];
    const double D_neighbor = D_cell[c1 * n_groups + g];
    const double D_face = harmonic_positive(D_self, D_neighbor, nullptr);
    const double coef = dt * area * D_face / dist;
    const double area_dist_ratio = area / dist;
    FldFaceGlobalMaxRecord* records[3] = {
        max_D_face, max_coef, max_area_dist_ratio};
    const double values_to_compare[3] = {D_face, coef, area_dist_ratio};
    for (int rec_idx = 0; rec_idx < 3; ++rec_idx) {
      if (isfinite(values_to_compare[rec_idx]) &&
          values_to_compare[rec_idx] > records[rec_idx]->value) {
        records[rec_idx]->value = values_to_compare[rec_idx];
        records[rec_idx]->D_self = finite_or_zero(D_self);
        records[rec_idx]->D_neighbor = finite_or_zero(D_neighbor);
        records[rec_idx]->D_face = finite_or_zero(D_face);
        records[rec_idx]->area_face = finite_or_zero(area);
        records[rec_idx]->dist_face = finite_or_zero(dist);
        records[rec_idx]->area_dist_ratio = finite_or_zero(area_dist_ratio);
        records[rec_idx]->coef = finite_or_zero(coef);
        records[rec_idx]->cell_i = i0;
        records[rec_idx]->cell_j = j0;
        records[rec_idx]->neighbor_i = i1;
        records[rec_idx]->neighbor_j = j1;
        records[rec_idx]->group = g;
        records[rec_idx]->side = side;
      }
    }
  }
}

__global__ void fld_face_symmetry_audit_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ rad_E,
    const double* __restrict__ D_cell,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    FldFaceSymmetryRecord* __restrict__ record,
    int nr,
    int nz,
    int n_groups,
    double dt) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_r_faces = (nr > 1) ? ((nr - 1) * nz) : 0;
  const int n_z_faces = (nz > 1) ? (nr * (nz - 1)) : 0;
  const int total = (n_r_faces + n_z_faces) * n_groups;
  if (idx >= total) {
    return;
  }
  const int face = idx / n_groups;
  const int g = idx - face * n_groups;
  int cell_l = 0;
  int cell_r = 0;
  int direction = 0;
  double area_lr = 0.0;
  double area_rl = 0.0;
  double dist = 0.0;
  if (face < n_r_faces) {
    const int f = face;
    const int i = f / nz;
    const int j = f - i * nz;
    cell_l = cell_index(i, j, nz);
    cell_r = cell_index(i + 1, j, nz);
    direction = 0;
    const CellGeometryRZ geom_l =
        rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
    const CellGeometryRZ geom_r =
        rect_cell_geometry_v2(x_r, x_z, vol, i + 1, j, nz);
    area_lr = geom_l.area_iR;
    area_rl = geom_r.area_iL;
    dist = finite_or_zero(rc_cache[cell_r]) - finite_or_zero(rc_cache[cell_l]);
  } else {
    const int f = face - n_r_faces;
    const int i = f / (nz - 1);
    const int j = f - i * (nz - 1);
    cell_l = cell_index(i, j, nz);
    cell_r = cell_index(i, j + 1, nz);
    direction = 1;
    const CellGeometryRZ geom_l =
        rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
    const CellGeometryRZ geom_r =
        rect_cell_geometry_v2(x_r, x_z, vol, i, j + 1, nz);
    area_lr = geom_l.area_jT;
    area_rl = geom_r.area_jB;
    dist = finite_or_zero(zc_cache[cell_r]) - finite_or_zero(zc_cache[cell_l]);
  }
  if (!(dist > kFldFaceDistMin)) {
    return;
  }
  const int idx_l = cell_l * n_groups + g;
  const int idx_r = cell_r * n_groups + g;
  const double Df = harmonic_positive(D_cell[idx_l], D_cell[idx_r]);
  const double c_lr = dt * area_lr * Df / dist;
  const double c_rl = dt * area_rl * Df / dist;
  const double energy_delta =
      finite_or_zero(rad_E[idx_l]) - finite_or_zero(rad_E[idx_r]);
  const double defect = fabs((c_lr - c_rl) * energy_delta);
  atomic_max_face_symmetry_record(record,
                                  defect,
                                  c_lr,
                                  c_rl,
                                  energy_delta,
                                  cell_l,
                                  cell_r,
                                  direction);
}

__global__ void rogue_x_raw_kernel(const double* __restrict__ x,
                                   RogueRecord* __restrict__ records,
                                   int n_cells,
                                   int n_groups) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }
  const int g = row / n_cells;
  const int c = row - g * n_cells;
  const double x_raw = finite_or_zero(x[row]);
  atomic_max_record(records, kRogueMaxAbsXRaw, fabs(x_raw), c, g);
  atomic_min_record(records, kRogueMinXRaw, x_raw, c, g);
}

__global__ void rogue_rad_E_kernel(const double* __restrict__ rad_E,
                                   RogueRecord* __restrict__ records,
                                   int n_cells,
                                   int n_groups) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = n_cells * n_groups;
  if (idx >= total) {
    return;
  }
  const int c = idx / n_groups;
  const int g = idx - c * n_groups;
  atomic_max_record(records,
                    kRogueMaxRadE,
                    nonnegative_finite(rad_E[idx]),
                    c,
                    g);
}

__global__ void rogue_residual_kernel(const double* __restrict__ r_true,
                                      RogueRecord* __restrict__ records,
                                      int n_cells,
                                      int n_groups) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_rows = n_cells * n_groups;
  if (row >= n_rows) {
    return;
  }
  const int g = row / n_cells;
  const int c = row - g * n_cells;
  const double value = fabs(finite_or_zero(r_true[row]));
  atomic_max_record(records, kRogueMaxRTrue, value, c, g);
  atomic_max_record(records, kRogueMaxESolverRow, value, c, g);
}

__global__ void snapshot_Te_kernel(const double* __restrict__ Te,
                                   double* __restrict__ Te_old,
                                   int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c < n_cells) {
    Te_old[c] = finite_or_zero(Te[c]);
  }
}

__global__ void update_matter_kernel(
    const double* __restrict__ rho,
    const double* __restrict__ vol,
    const double* __restrict__ mass,
    const double* __restrict__ zbar,
    const double* __restrict__ cv_e,
    const double* __restrict__ Te_old,
    const double* __restrict__ sigma_a,
    const double* __restrict__ sigma_pe,
    const double* __restrict__ eta,
    const double* __restrict__ fleck,
    const double* __restrict__ rad_E_old,
    const double* __restrict__ rad_E,
    PlanckTableDeviceView planck,
    materials::DeviceEOSTableView electron_eos,
    double* __restrict__ Te,
    double* __restrict__ ee,
    double* __restrict__ Pe,
    double* __restrict__ Ee_per_material,
    const double* __restrict__ mass_per_material,
    double* __restrict__ rad_dep,
    double* __restrict__ rad_emit,
    double* __restrict__ delta_T_rel,
    double* __restrict__ fld_fleck,
    const int c_begin,
    int n_cells,
    int n_groups,
    int n_mat,
    double dt,
    double A,
    double gamma,
    double cv_e_override,
    double temperature_floor_eV,
    int has_cv_e,
    int max_newton_iterations,
    double newton_tolerance,
    bool diagnostic_mode,
    int* __restrict__ newton_converged_count,
    int* __restrict__ newton_invalid_count,
    int* __restrict__ newton_cap_hit_count,
    int* __restrict__ newton_reject_count,
    double* __restrict__ newton_resid_abs_max,
    double* __restrict__ newton_resid_rel_max,
    double* __restrict__ newton_reject_resid_rel_max) {
  const int c = c_begin + blockIdx.x;
  const int tid = threadIdx.x;
  if (c >= n_cells) {
    return;
  }

  extern __shared__ double smem[];
  double* s_F_g = smem;
  double* s_S_g = s_F_g + n_groups;
  double* s_dF_g = s_S_g + n_groups;

  __shared__ double s_T;
  __shared__ double s_step_T_ref;
  __shared__ double s_iter_T_prev;
  __shared__ double s_rho_c;
  __shared__ double s_cv_mass;
  __shared__ double s_dt_safe;
  __shared__ double s_Cv;
  __shared__ double s_e_e_ref;
  __shared__ double s_last_resid_abs;
  __shared__ double s_last_resid_rel;
  __shared__ int s_break;
  __shared__ int s_exit_reason;
  __shared__ int s_use_table_eos;
  __shared__ materials::RhoBracket s_eos_rho_bracket;

  if (tid == 0) {
    const double rho_c = positive_or_floor(rho[c], 1.0e-300);
    s_rho_c = rho_c;
    s_step_T_ref = fmax(finite_or_zero(Te_old[c]), temperature_floor_eV);
    s_iter_T_prev = fmax(finite_or_zero(Te[c]), temperature_floor_eV);
    s_T = s_iter_T_prev;
    double cv_mass = 0.0;
    if (has_cv_e != 0 && cv_e != nullptr && cv_e[c] > 0.0 && isfinite(cv_e[c])) {
      cv_mass = cv_e[c];
    } else if (cv_e_override > 0.0) {
      cv_mass = cv_e_override / rho_c;
    } else {
      const double gm1 = fmax(gamma - 1.0, 1.0e-12);
      const double z = fmax(finite_or_zero(zbar[c]), 0.0);
      cv_mass = z * core::constants::eV_to_erg /
                (fmax(A, 1.0e-12) * core::constants::proton_mass * gm1);
    }
    s_cv_mass = fmax(finite_or_zero(cv_mass), 1.0e-300);
    s_dt_safe = fmax(dt, 1.0e-300);
    s_Cv = fmax(rho_c * s_cv_mass, 1.0e-300);
    s_use_table_eos = has_electron_eos_table(electron_eos) ? 1 : 0;
    if (s_use_table_eos != 0) {
      s_eos_rho_bracket = materials::find_rho_bracket(electron_eos, rho_c);
      s_e_e_ref = materials::device_eos_energy(
          electron_eos,
          s_eos_rho_bracket,
          eos_log_temperature(s_step_T_ref, temperature_floor_eV));
    } else {
      s_e_e_ref = 0.0;
    }
    s_break = 0;
    s_exit_reason = 0;
    s_last_resid_abs = 0.0;
    s_last_resid_rel = 0.0;
  }
  __syncthreads();

  const int max_iter = (max_newton_iterations > 1) ? max_newton_iterations : 1;
  const double tol = fmax(newton_tolerance, 0.0);
  for (int iter = 0; iter < max_iter; ++iter) {
    if (s_break != 0) {
      break;
    }
    const double T = s_T;
    const double T2 = T * T;
    const double T4 = T2 * T2;
    const double T3 = (T > 0.0) ? (T4 / T) : 0.0;

    for (int g = tid; g < n_groups; g += blockDim.x) {
      const int idx = c * n_groups + g;
      const double sigma_pa = nonnegative_finite(sigma_a[idx]);
      const double sigma_pe_g =
          (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx]) : sigma_pa;
      const double E = nonnegative_finite(rad_E[idx]);
      const double dep_rate = core::constants::c_light * sigma_pa * E;
      const double b =
          (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T), 0.0);
      const double B = core::constants::a_eV * T4 * b;
      double f = 1.0;
      double E_old_g = 0.0;
      if (fleck != nullptr) {
        f = fmin(fmax(finite_or_zero(fleck[idx]), 0.0), 1.0);
        E_old_g = (rad_E_old != nullptr)
                      ? fmax(finite_or_zero(rad_E_old[idx]), 0.0)
                      : 0.0;
      }
      const double emit_g = f * core::constants::c_light * sigma_pe_g * B +
                            (1.0 - f) * core::constants::c_light *
                                sigma_pe_g * E_old_g;
      s_F_g[g] = -(dep_rate - emit_g);
      s_dF_g[g] = f * core::constants::c_light * sigma_pe_g * 4.0 *
                  core::constants::a_eV * T3 * b;
      s_S_g[g] = fmax(fabs(emit_g), fabs(dep_rate));
    }
    __syncthreads();

    if (tid == 0) {
      double F = 0.0;
      double dF = 0.0;
      if (s_use_table_eos != 0) {
        const double logT = eos_log_temperature(T, temperature_floor_eV);
        const double e_e_T =
            materials::device_eos_energy(electron_eos, s_eos_rho_bracket, logT);
        const double cv_e_T = fmax(
            nonnegative_finite(
                materials::device_eos_cv(electron_eos, s_eos_rho_bracket, logT)),
            1.0e-300);
        F = s_rho_c * (e_e_T - s_e_e_ref) / s_dt_safe;
        dF = s_rho_c * cv_e_T / s_dt_safe;
      } else {
        F = s_Cv * (T - s_step_T_ref) / s_dt_safe;
        dF = s_Cv / s_dt_safe;
      }
      const double material_rate = F;
      for (int g = 0; g < n_groups; ++g) {
        F += s_F_g[g];
        dF += s_dF_g[g];
      }
      double S_residual = 0.0;
      for (int g = 0; g < n_groups; ++g) {
        S_residual = fmax(S_residual, s_S_g[g]);
      }
      S_residual = fmax(fmax(S_residual, fabs(material_rate)), 1.0e-300);
      s_last_resid_abs = fabs(F);
      s_last_resid_rel = s_last_resid_abs / S_residual;
      if (!(isfinite(F) && isfinite(dF)) || !(dF > 0.0)) {
        s_break = 1;
        s_exit_reason = 2;
      } else {
        double dT = -F / dF;
        const double max_step = 0.5 * fmax(T, temperature_floor_eV);
        dT = fmin(fmax(dT, -max_step), max_step);
        const double T_next = fmax(T + dT, temperature_floor_eV);
        const double rel = fabs(T_next - T) / fmax(T_next, temperature_floor_eV);
        double material_rate_next = 0.0;
        if (s_use_table_eos != 0) {
          const double logT_next = eos_log_temperature(T_next, temperature_floor_eV);
          const double e_e_T_next = materials::device_eos_energy(
              electron_eos, s_eos_rho_bracket, logT_next);
          material_rate_next = s_rho_c * (e_e_T_next - s_e_e_ref) / s_dt_safe;
        } else {
          material_rate_next = s_Cv * (T_next - s_step_T_ref) / s_dt_safe;
        }
        double F_next = material_rate_next;
        S_residual = 0.0;
        const double T_next2 = T_next * T_next;
        const double T_next4 = T_next2 * T_next2;
        for (int g = 0; g < n_groups; ++g) {
          const int idx = c * n_groups + g;
          const double sigma_pa = nonnegative_finite(sigma_a[idx]);
          const double sigma_pe_g =
              (sigma_pe != nullptr) ? nonnegative_finite(sigma_pe[idx])
                                    : sigma_pa;
          const double E = nonnegative_finite(rad_E[idx]);
          const double dep_rate = core::constants::c_light * sigma_pa * E;
          const double b =
              (n_groups == 1) ? 1.0 : fmax(planck.interpolate_b(g, T_next), 0.0);
          const double B = core::constants::a_eV * T_next4 * b;
          double f = 1.0;
          double E_old_g = 0.0;
          if (fleck != nullptr) {
            f = fmin(fmax(finite_or_zero(fleck[idx]), 0.0), 1.0);
            E_old_g = (rad_E_old != nullptr)
                          ? fmax(finite_or_zero(rad_E_old[idx]), 0.0)
                          : 0.0;
          }
          const double emit_g = f * core::constants::c_light * sigma_pe_g * B +
                                (1.0 - f) * core::constants::c_light *
                                    sigma_pe_g * E_old_g;
          F_next += -(dep_rate - emit_g);
          S_residual = fmax(S_residual, fmax(fabs(emit_g), fabs(dep_rate)));
        }
        S_residual = fmax(fmax(S_residual, fabs(material_rate_next)), 1.0e-300);
        s_last_resid_abs = fabs(F_next);
        s_last_resid_rel = s_last_resid_abs / S_residual;
        s_T = T_next;
        const bool dual_criterion_pass =
            (rel <= tol) && (s_last_resid_rel <= fmax(tol, 1.0e-12));
        const bool numerical_fixed_point = (rel < 1.0e-15);
        if (dual_criterion_pass || numerical_fixed_point) {
          s_break = 1;
          s_exit_reason = 1;
        }
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    const bool newton_accepted = (s_exit_reason == 1);
    const double T = newton_accepted ? s_T : s_iter_T_prev;
    const double ee_before = finite_or_zero(ee[c]);
    Te[c] = T;
    if (s_use_table_eos != 0) {
      const double logT = eos_log_temperature(T, temperature_floor_eV);
      ee[c] = materials::device_eos_energy(electron_eos, s_eos_rho_bracket, logT);
      Pe[c] = materials::device_eos_pressure(electron_eos, s_eos_rho_bracket, logT);
    } else {
      ee[c] = finite_or_zero(ee[c]) + s_cv_mass * (T - s_iter_T_prev);
      Pe[c] = fmax(gamma - 1.0, 1.0e-12) * s_rho_c * finite_or_zero(ee[c]);
    }
    if (Ee_per_material != nullptr && mass_per_material != nullptr &&
        mass != nullptr && n_mat > 0) {
      double sum_mass_m = 0.0;
      for (int m = 0; m < n_mat; ++m) {
        const int idx = c * n_mat + m;
        const double mass_m = mass_per_material[idx];
        if (mass_m > 0.0 && isfinite(mass_m)) {
          sum_mass_m += mass_m;
        }
      }
      const double mass_c = mass[c];
      if (sum_mass_m > 0.0 && mass_c > 0.0 && isfinite(mass_c)) {
        const double dE_cell = mass_c * (finite_or_zero(ee[c]) - ee_before);
        if (isfinite(dE_cell) && dE_cell != 0.0) {
          for (int m = 0; m < n_mat; ++m) {
            const int idx = c * n_mat + m;
            const double mass_m = mass_per_material[idx];
            if (mass_m > 0.0 && isfinite(mass_m)) {
              const double dE_m = dE_cell * (mass_m / sum_mass_m);
              Ee_per_material[idx] = fmax(Ee_per_material[idx] + dE_m, 0.0);
            }
          }
        }
      }
    }
    delta_T_rel[c] = newton_accepted
                         ? fabs(T - s_iter_T_prev) /
                               fmax(T, temperature_floor_eV)
                         : 1.0;
    if (fld_fleck != nullptr) {
      // BUG-11: fleck is a cg array; the per-cell diagnostic takes the cell's
      // group-0 slot (producers broadcast one gray f per cell to all groups).
      const double f = (fleck != nullptr)
                           ? fmin(fmax(finite_or_zero(fleck[c * n_groups]), 0.0), 1.0)
                           : 1.0;
      fld_fleck[c] = f;
    }
    if (diagnostic_mode) {
      if (s_exit_reason == 1 && newton_converged_count != nullptr) {
        atomicAdd(newton_converged_count, 1);
      } else if (s_exit_reason == 2 && newton_invalid_count != nullptr) {
        atomicAdd(newton_invalid_count, 1);
      } else if (s_exit_reason == 0 && newton_cap_hit_count != nullptr) {
        atomicAdd(newton_cap_hit_count, 1);
      }
      if (!newton_accepted) {
        if (newton_reject_count != nullptr) {
          atomicAdd(newton_reject_count, 1);
        }
        if (newton_reject_resid_rel_max != nullptr) {
          atomic_max_nonnegative_double(newton_reject_resid_rel_max,
                                        s_last_resid_rel);
        }
      } else {
        if (newton_resid_abs_max != nullptr) {
          atomic_max_nonnegative_double(newton_resid_abs_max, s_last_resid_abs);
        }
        if (newton_resid_rel_max != nullptr) {
          atomic_max_nonnegative_double(newton_resid_rel_max, s_last_resid_rel);
        }
      }
    }
  }
  __syncthreads();

  const double V = fmax(finite_or_zero(vol[c]), 0.0);
  for (int g = tid; g < n_groups; g += blockDim.x) {
    const int idx = c * n_groups + g;
    const double sigma_pa = nonnegative_finite(sigma_a[idx]);
    const double E = nonnegative_finite(rad_E[idx]);
    const double source =
        fld_rhs_source_rate(eta, fleck, sigma_a, rad_E_old, idx, c);
    rad_dep[idx] = dt * V * core::constants::c_light * sigma_pa * E;
    rad_emit[idx] = dt * V * source;
  }
}

__global__ void max_reduce_kernel(const double* __restrict__ values,
                                  double* __restrict__ out,
                                  int n) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int i = blockIdx.x * blockDim.x + tid;
  shared[tid] = (i < n) ? fmax(finite_or_zero(values[i]), 0.0) : 0.0;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmax(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomic_max_nonnegative_double(out, shared[0]);
  }
}

// Owned-rows window (Option C): iterate the per-group owned row ranges;
// at a full window (c_begin=0, owned=n_cells) the row map is the identity
// over [0, total) — single-rank behavior is bit-identical.
__global__ void post_publish_residual_kernel(const double* __restrict__ rhs,
                                             double* __restrict__ Ax_or_r_true,
                                             double* __restrict__ out,
                                             int n_cells,
                                             int c_begin,
                                             int owned,
                                             int total) {
  __shared__ double shared_sum[kBlock];
  __shared__ double shared_max[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  double local_sum = 0.0;
  double local_max = 0.0;
  if (idx < total) {
    const int g = idx / owned;
    const int i = g * n_cells + c_begin + (idx - g * owned);
    const double residual =
        finite_or_zero(Ax_or_r_true[i]) - finite_or_zero(rhs[i]);
    const double r_true = -residual;
    Ax_or_r_true[i] = r_true;
    local_sum = residual;
    local_max = fabs(r_true);
  }
  shared_sum[tid] = local_sum;
  shared_max[tid] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_sum[tid] += shared_sum[tid + stride];
      shared_max[tid] = fmax(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, shared_sum[0]);
    atomic_max_nonnegative_double(out + 1, shared_max[0]);
  }
}

__global__ void max_group_major_residual_record_kernel(
    const double* __restrict__ r_true,
    RogueRecord* __restrict__ record,
    int n_cells,
    int n_groups,
    int c_begin,
    int owned) {
  const int total = owned * n_groups;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int g = idx / owned;
  const int c = c_begin + (idx - g * owned);
  const int row = g * n_cells + c;
  atomic_max_record(record, 0, fabs(finite_or_zero(r_true[row])), c, g);
}

// Owned-cells window (Option C): each physical boundary face is owned by
// exactly one rank under the r-slab split, so the owned-windowed sums are a
// disjoint partition of the global tally (the driver's step-budget
// Allreduce completes them). Full window at P==1 is the identity map.
__global__ void escaped_energy_2d_kernel(const double* __restrict__ x_r,
                                         const double* __restrict__ x_z,
                                         const double* __restrict__ vol,
                                         const double* __restrict__ rad_E,
                                         const double* __restrict__ D_cell,
                                         double* __restrict__ out,
                                         int c_begin,
                                         int c_end,
                                         int nr,
                                         int nz,
                                         int n_groups,
                                         double dt,
                                         int outer_r_bc,
                                         int z_bottom_bc,
                                         int z_top_bc,
                                         double T_supply_z_bottom_eV,
                                         double T_supply_z_top_eV,
                                         const double* __restrict__ rho = nullptr,
                                         const double* __restrict__ sigma_R = nullptr,
                                         int state_supply_boundary_policy =
                                             kStateSupplyBoundaryPolicyLocalDCurrent,
                                         int limiter = 0,
                                         double rho_supply_z_bottom = 0.0,
                                         double rho_supply_z_top = 0.0) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int total = (c_end - c_begin) * n_groups;
  double local = 0.0;
  if (idx < total) {
    const int c_local = idx / n_groups;
    const int g = idx - c_local * n_groups;
    const int c = c_begin + c_local;
    const int e = c * n_groups + g;
    const int i = c / nz;
    const int j = c - i * nz;
    if (i + 1 == nr || j == 0 || j + 1 == nz) {
      const CellGeometryRZ geom =
          rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
      double leakage_area_coeff = 0.0;
      if (i + 1 == nr) {
        leakage_area_coeff +=
            geom.area_iR * fld_boundary_leakage_coeff(outer_r_bc);
      }
      if (j == 0) {
        if (z_bottom_bc == kFldBcStateSupply) {
          const double coeff =
              fld_state_supply_boundary_coeff(
                  geom,
                  D_cell[e],
                  false,
                  state_supply_boundary_policy,
                  (rho != nullptr) ? rho[c] : 0.0,
                  (sigma_R != nullptr) ? sigma_R[e] : 0.0,
                  rho_supply_z_bottom,
                  D_cell,
                  nr,
                  nz,
                  n_groups,
                  0,
                  g,
                  limiter);
          local += dt * fmax(coeff *
                                 (fmax(finite_or_zero(rad_E[e]), 0.0) -
                                  fld_state_supply_E(T_supply_z_bottom_eV)),
                             0.0);
        } else {
          leakage_area_coeff +=
              geom.area_jB * fld_boundary_leakage_coeff(z_bottom_bc);
        }
      }
      if (j + 1 == nz) {
        if (z_top_bc == kFldBcStateSupply) {
          const double coeff =
              fld_state_supply_boundary_coeff(
                  geom,
                  D_cell[e],
                  true,
                  state_supply_boundary_policy,
                  (rho != nullptr) ? rho[c] : 0.0,
                  (sigma_R != nullptr) ? sigma_R[e] : 0.0,
                  rho_supply_z_top,
                  D_cell,
                  nr,
                  nz,
                  n_groups,
                  nz - 1,
                  g,
                  limiter);
          local += dt * fmax(coeff *
                                 (fmax(finite_or_zero(rad_E[e]), 0.0) -
                                  fld_state_supply_E(T_supply_z_top_eV)),
                             0.0);
        } else {
          leakage_area_coeff += geom.area_jT * fld_boundary_leakage_coeff(z_top_bc);
        }
      }
      const double E = fmax(finite_or_zero(rad_E[e]), 0.0);
      local += dt * leakage_area_coeff * E;
    }
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, shared[0]);
  }
}

__global__ void state_supply_flux_2d_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ rad_E,
    const double* __restrict__ D_cell,
    double* __restrict__ out_in_out_net,
    int c_begin,
    int c_end,
    int nr,
    int nz,
    int n_groups,
    double dt,
    int z_bottom_bc,
    int z_top_bc,
    double T_supply_z_bottom_eV,
    double T_supply_z_top_eV,
    const double* __restrict__ rho = nullptr,
    const double* __restrict__ sigma_R = nullptr,
    int state_supply_boundary_policy =
        kStateSupplyBoundaryPolicyLocalDCurrent,
    int limiter = 0,
    double rho_supply_z_bottom = 0.0,
    double rho_supply_z_top = 0.0) {
  __shared__ double shared_in[kBlock];
  __shared__ double shared_out[kBlock];
  __shared__ double shared_net[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int total = (c_end - c_begin) * n_groups;
  double local_in = 0.0;
  double local_out = 0.0;
  double local_net = 0.0;
  if (idx < total) {
    const int c_local = idx / n_groups;
    const int g = idx - c_local * n_groups;
    const int c = c_begin + c_local;
    const int e = c * n_groups + g;
    const int i = c / nz;
    const int j = c - i * nz;
    if ((j == 0 && z_bottom_bc == kFldBcStateSupply) ||
        (j + 1 == nz && z_top_bc == kFldBcStateSupply)) {
      const CellGeometryRZ geom =
          rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
      const double E = fmax(finite_or_zero(rad_E[e]), 0.0);
      if (j == 0 && z_bottom_bc == kFldBcStateSupply) {
        const double coeff =
            fld_state_supply_boundary_coeff(
                geom,
                D_cell[e],
                false,
                state_supply_boundary_policy,
                (rho != nullptr) ? rho[c] : 0.0,
                (sigma_R != nullptr) ? sigma_R[e] : 0.0,
                rho_supply_z_bottom,
                D_cell,
                nr,
                nz,
                n_groups,
                0,
                g,
                limiter);
        const double net =
            dt * coeff * (E - fld_state_supply_E(T_supply_z_bottom_eV));
        local_net += net;
        local_out += fmax(net, 0.0);
        local_in += fmax(-net, 0.0);
      }
      if (j + 1 == nz && z_top_bc == kFldBcStateSupply) {
        const double coeff =
            fld_state_supply_boundary_coeff(
                geom,
                D_cell[e],
                true,
                state_supply_boundary_policy,
                (rho != nullptr) ? rho[c] : 0.0,
                (sigma_R != nullptr) ? sigma_R[e] : 0.0,
                rho_supply_z_top,
                D_cell,
                nr,
                nz,
                n_groups,
                nz - 1,
                g,
                limiter);
        const double net =
            dt * coeff * (E - fld_state_supply_E(T_supply_z_top_eV));
        local_net += net;
        local_out += fmax(net, 0.0);
        local_in += fmax(-net, 0.0);
      }
    }
  }
  shared_in[tid] = local_in;
  shared_out[tid] = local_out;
  shared_net[tid] = local_net;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_in[tid] += shared_in[tid + stride];
      shared_out[tid] += shared_out[tid + stride];
      shared_net[tid] += shared_net[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out_in_out_net + 0, shared_in[0]);
    atomicAdd(out_in_out_net + 1, shared_out[0]);
    atomicAdd(out_in_out_net + 2, shared_net[0]);
  }
}

__global__ void compute_escape_breakdown_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const double* __restrict__ rho,
    const double* __restrict__ rad_E,
    const double* __restrict__ sigma_removal,
    const double* __restrict__ D_cell,
    const int* __restrict__ col_indices,
    const double* __restrict__ values,
    int nr,
    int nz,
    int n_groups,
    double dt,
    int outer_r_bc,
    int z_bottom_bc,
    int z_top_bc,
    double rho_vac_threshold,
    double* __restrict__ totals,
    RogueRecord* __restrict__ records,
    EscapeBreakdownComponentRecord* __restrict__ component_record) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int n_cells = nr * nz;
  if (c >= n_cells) {
    return;
  }
  const int i = c / nz;
  const int j = c - i * nz;
  if (i + 1 != nr && j != 0 && j + 1 != nz) {
    return;
  }

  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  const double outer_coeff =
      (i + 1 == nr) ? dt * geom.area_iR *
                          fld_boundary_leakage_coeff(outer_r_bc)
                    : 0.0;
  const double z_bottom_coeff =
      (j == 0) ? dt * geom.area_jB *
                     fld_boundary_leakage_coeff(z_bottom_bc)
               : 0.0;
  const double z_top_coeff =
      (j + 1 == nz) ? dt * geom.area_jT *
                          fld_boundary_leakage_coeff(z_top_bc)
                    : 0.0;
  const double boundary_coeff =
      outer_coeff + z_bottom_coeff + z_top_coeff;
  if (!(boundary_coeff > 0.0)) {
    return;
  }

  double outer_escape = 0.0;
  double z_bottom_escape = 0.0;
  double z_top_escape = 0.0;
  double csr_boundary_escape = 0.0;
  double formula_boundary_escape = 0.0;
  double signed_delta_outer_r = 0.0;
  double signed_delta_z_bottom = 0.0;
  double signed_delta_z_top = 0.0;
  double max_group_abs_delta = 0.0;
  int max_group_delta_g = -1;
  EscapeBreakdownComponentRecord local_component;
  const double V = geom.V_op;
  for (int g = 0; g < n_groups; ++g) {
    const int cg = c * n_groups + g;
    const int row = g * n_cells + c;
    const double E = nonnegative_finite(rad_E[cg]);
    outer_escape += outer_coeff * E;
    z_bottom_escape += z_bottom_coeff * E;
    z_top_escape += z_top_coeff * E;
    formula_boundary_escape += boundary_coeff * E;

    const int base = row * kCsrEntriesPerRow;
    double interior_diag = 0.0;
    for (int slot = 1; slot < kCsrEntriesPerRow; ++slot) {
      if (col_indices[base + slot] != row) {
        interior_diag += fmax(-finite_or_zero(values[base + slot]), 0.0);
      }
    }
    const double sig = nonnegative_finite(sigma_removal[cg]);
    const double dt_c_sigma_V = dt * core::constants::c_light * sig * V;
    const double csr_boundary_diag =
        finite_or_zero(values[base]) - V - dt_c_sigma_V - interior_diag;
    const double csr_boundary_group_escape = csr_boundary_diag * E;
    csr_boundary_escape += csr_boundary_group_escape;
    const double group_delta =
        csr_boundary_group_escape - boundary_coeff * E;
    const double boundary_delta_scale =
        (boundary_coeff > 0.0) ? group_delta / boundary_coeff : 0.0;
    signed_delta_outer_r += outer_coeff * boundary_delta_scale;
    signed_delta_z_bottom += z_bottom_coeff * boundary_delta_scale;
    signed_delta_z_top += z_top_coeff * boundary_delta_scale;
    const double abs_group_delta = fabs(group_delta);
    if (abs_group_delta > max_group_abs_delta) {
      max_group_abs_delta = abs_group_delta;
      max_group_delta_g = g;
      local_component.csr_diag_value = finite_or_zero(values[base]);
      local_component.V_op = V;
      local_component.dt_c_sigma_V = dt_c_sigma_V;
      local_component.interior_diag_sum = interior_diag;
      local_component.csr_boundary_diag = csr_boundary_diag;
      local_component.formula_boundary_coef = boundary_coeff;
      local_component.rad_E_at_cell = E;
      local_component.sigma_a_at_cell = sig;
      local_component.rho_at_cell = nonnegative_finite(rho[c]);
      local_component.D_cell_at_cell = nonnegative_finite(D_cell[cg]);
    }
  }

  atomicAdd(totals + 0, outer_escape);
  atomicAdd(totals + 1, z_bottom_escape);
  atomicAdd(totals + 2, z_top_escape);
  const bool is_vacuum = nonnegative_finite(rho[c]) < rho_vac_threshold;
  if (is_vacuum) {
    atomicAdd(totals + 3, outer_escape);
    atomicAdd(totals + 4, z_bottom_escape);
    atomicAdd(totals + 5, z_top_escape);
  }
  atomicAdd(totals + 6, signed_delta_outer_r);
  atomicAdd(totals + 7, signed_delta_z_bottom);
  atomicAdd(totals + 8, signed_delta_z_top);

  atomic_max_cell_record(records + 0, outer_escape, c);
  atomic_max_cell_record(records + 1, z_bottom_escape, c);
  atomic_max_cell_record(records + 2, z_top_escape, c);
  const double delta = csr_boundary_escape - formula_boundary_escape;
  const double abs_delta = fabs(delta);
  atomicAdd(totals + 9, abs_delta);
  if (abs_delta > 1.0) {
    atomicAdd(totals + 10, 1.0);
  }
  if (abs_delta > 10.0) {
    atomicAdd(totals + 11, 1.0);
  }
  if (abs_delta > 100.0) {
    atomicAdd(totals + 12, 1.0);
  }
  if (atomic_max_cell_record_updated(records + 3, abs_delta, c,
                                     max_group_delta_g)) {
    *component_record = local_component;
  }
  if (delta > 0.0) {
    atomic_max_cell_record(records + 4, delta, c);
  } else if (delta < 0.0) {
    atomic_max_cell_record(records + 5, -delta, c);
  }
}

// Owned-radial window (Option C): the z boundary faces span every radial
// column, so the owned i range partitions the tally across ranks.
__global__ void marshak_in_energy_2d_kernel(
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    double* __restrict__ out,
    int i_begin,
    int i_end,
    int nr,
    int nz,
    double dt,
    int z_bottom_bc,
    int z_top_bc,
    double marshak_flux_erg_per_cm2_s) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int i = i_begin + blockIdx.x * blockDim.x + tid;
  double local = 0.0;
  if (i < i_end && marshak_flux_erg_per_cm2_s > 0.0) {
    double area = 0.0;
    if (z_bottom_bc == kFldBcMarshak) {
      const CellGeometryRZ geom_bottom =
          rect_cell_geometry_v2(x_r, x_z, vol, i, 0, nz);
      area += geom_bottom.area_jB;
    }
    if (z_top_bc == kFldBcMarshak) {
      const CellGeometryRZ geom_top =
          rect_cell_geometry_v2(x_r, x_z, vol, i, nz - 1, nz);
      area += geom_top.area_jT;
    }
    local = dt * marshak_flux_erg_per_cm2_s * area;
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomicAdd(out, shared[0]);
  }
}

// Additive RHS post-pass for Tr(t)-driven marshak z faces (spec §3). The
// existing assembly kernel stays byte-untouched (W-G1 FMA-reassociation
// lesson); boundary rows receive their per-group incident source AFTER
// assembly, once per outer iteration. One thread per (face, i, g); no
// atomics (each CSR row is touched by exactly one thread).
__global__ void add_marshak_tr_rhs_2d_kernel(
    double* __restrict__ rhs,
    const double* __restrict__ finc,  // [2*n_groups]: bottom face then top
    const double* __restrict__ x_r,
    const double* __restrict__ x_z,
    const double* __restrict__ vol,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    const int bottom_active,
    const int top_active) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int per_face = nr * n_groups;
  if (idx >= 2 * per_face) {
    return;
  }
  const int face = idx / per_face;
  const int rem = idx - face * per_face;
  const int i = rem / n_groups;
  const int g = rem - i * n_groups;
  if (face == 0 && bottom_active == 0) {
    return;
  }
  if (face == 1 && top_active == 0) {
    return;
  }
  const int j = (face == 0) ? 0 : (nz - 1);
  const CellGeometryRZ geom = rect_cell_geometry_v2(x_r, x_z, vol, i, j, nz);
  const double area = (face == 0) ? geom.area_jB : geom.area_jT;
  const double f = finc[face * n_groups + g];
  if (!(area > 0.0) || !(f > 0.0)) {
    return;
  }
  const int n_cells = nr * nz;
  const int row = g * n_cells + cell_index(i, j, nz);
  rhs[row] += dt * area * f;
}

__global__ void max_reduced_flux_2d_kernel(
    const double* __restrict__ rad_E,
    const double* __restrict__ D_cell,
    const double* __restrict__ rc_cache,
    const double* __restrict__ zc_cache,
    double* __restrict__ out,
    int nr,
    int nz,
    int n_groups) {
  __shared__ double shared[kBlock];
  const int tid = threadIdx.x;
  const int idx = blockIdx.x * blockDim.x + tid;
  const int n_r_faces = (nr > 1) ? ((nr - 1) * nz) : 0;
  const int n_z_faces = (nz > 1) ? (nr * (nz - 1)) : 0;
  const int total = (n_r_faces + n_z_faces) * n_groups;
  double local = 0.0;
  if (idx < total) {
    const int face = idx / n_groups;
    const int g = idx - face * n_groups;
    int c0 = 0;
    int c1 = 0;
    double dist = 1.0e-300;
    if (face < n_r_faces) {
      const int f = face;
      const int i = f / nz;
      const int j = f - i * nz;
      c0 = cell_index(i, j, nz);
      c1 = cell_index(i + 1, j, nz);
    } else {
      const int f = face - n_r_faces;
      const int i = f / (nz - 1);
      const int j = f - i * (nz - 1);
      c0 = cell_index(i, j, nz);
      c1 = cell_index(i, j + 1, nz);
    }
    const double r0 = rc_cache[c0];
    const double z0 = zc_cache[c0];
    const double r1 = rc_cache[c1];
    const double z1 = zc_cache[c1];
    dist = fmax(sqrt((r1 - r0) * (r1 - r0) + (z1 - z0) * (z1 - z0)), 1.0e-300);
    const int i0g = c0 * n_groups + g;
    const int i1g = c1 * n_groups + g;
    const double E0 = fmax(finite_or_zero(rad_E[i0g]), 0.0);
    const double E1 = fmax(finite_or_zero(rad_E[i1g]), 0.0);
    const double Df = harmonic_positive(D_cell[i0g], D_cell[i1g]);
    const double F = -Df * (E1 - E0) / dist;
    const double Eface = fmax(0.5 * (E0 + E1), 1.0e-300);
    const double ratio = fabs(F) / (core::constants::c_light * Eface);
    local = (ratio >= 0.0 && isfinite(ratio)) ? ratio : 0.0;
  }
  shared[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] = fmax(shared[tid], shared[tid + stride]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    atomic_max_nonnegative_double(out, shared[0]);
  }
}

void ensure_state_buffers(core::State& state, const int n_cells, const int n_groups) {
  const std::size_t n_total =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups);
  state.rad_E.reset(n_total);
  state.rad_dep.reset(n_total);
  state.rad_emit.reset(n_total);
  state.rad_E_old.reset(n_total);
  state.fld_sigma_a.reset(n_total);
  state.fld_sigma_pe.reset(n_total);
  state.fld_sigma_R.reset(n_total);
  state.fld_eta.reset(n_total);
  state.fld_D_cell.reset(n_total);
  state.fld_Te_old.reset(static_cast<std::size_t>(n_cells));
  state.fld_delta_T.reset(static_cast<std::size_t>(n_cells));
  state.fld_fleck.reset(static_cast<std::size_t>(n_cells));
  state.fld_reduction_work.reset(1);
}

void initialize_rad_E_old_if_needed(core::State& state,
                                    const int n_cells,
                                    const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes == 0U) {
    return;
  }
  if (state.step == 0 || state.holo_ale_invalidated) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "FLD2D initialize rad_E_old failed");
  }
}

void compute_cell_centers(core::State& state, const int nr, const int nz) {
  const int n_cells = nr * nz;
  const bool resized = (static_cast<int>(state.fld_cell_rc.size()) != n_cells) ||
                       (static_cast<int>(state.fld_cell_zc.size()) != n_cells);
  if (resized) {
    state.fld_cell_rc.reset(static_cast<std::size_t>(n_cells));
    state.fld_cell_zc.reset(static_cast<std::size_t>(n_cells));
  }
  // Phase 2a-1.5c: Always recompute. The previous cache-skip logic
  // (resize / step==0 / holo_ale_invalidated) was unsafe because normal
  // 2D Lagrangian hydro updates state.x_r/x_z every step WITHOUT setting
  // holo_ale_invalidated. Stale rc/zc caches → asymmetric face distances
  // (rc_cache[neighbor] vs current rc) → operator non-self-adjointness
  // → sum_face_div leak that scales with mesh deformation.
  // Cost: O(n_cells) kernel per FLD step; negligible vs Picard outer cost.
  const int center_grid = (n_cells + kBlock - 1) / kBlock;
  if (center_grid > 0) {
    compute_cell_centers_kernel<<<center_grid, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.fld_cell_rc.data(),
        state.fld_cell_zc.data(),
        nr,
        nz);
    cuda_check(cudaGetLastError(), "FLD2D cell centers launch failed");
  }
}

std::vector<double> radiation_group_bounds(const core::Config& cfg,
                                           const int n_groups) {
  if (static_cast<int>(cfg.radiation.group_bounds_eV.size()) == n_groups + 1) {
    return cfg.radiation.group_bounds_eV;
  }
  const std::vector<double> range = resolve_compute_T_range_eV(cfg, false);
  const double Tmin = std::max(range[0], 1.0e-3);
  const double Tmax = std::max(range[1], Tmin * 1.001);
  return Groups::make_log_uniform_bounds(n_groups, Tmin, Tmax);
}

void evaluate_fld_opacity_and_emission(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const int n_cells,
    const int n_groups,
    const double dt) {
  const auto& fld = cfg.radiation.multigroup_diffusion;
  const int total = n_cells * n_groups;
  const bool use_nlte =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  if (use_nlte) {
    ensure_nlte_table_uploaded(cfg, mat, n_groups);
    const std::size_t scratch_size = static_cast<std::size_t>(total);
    state.fld_nlte_f_work.reset(scratch_size);
    state.fld_nlte_sigma_eff_work.reset(scratch_size);
    state.fld_nlte_sigma_s_eff_work.reset(scratch_size);
    state.fld_nlte_eta_cdf_work.reset(scratch_size);
    state.fld_nlte_lambda_work.reset(scratch_size);
    auto& cache = nlte_cache();
    const double* cv_e_ptr =
        (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                 : nullptr;
    const auto result = compute_nlte_coefficients_cuda_with_pe(
        state.rho.data(),
        state.Te.data(),
        state.zbar.data(),
        cv_e_ptr,
        state.cell_is_void.data(),
        state.cell_is_void.size(),
        cache.device.view(),
        planck.device_view(),
        n_cells,
        n_groups,
        dt,
        std::max(mat.A, 1.0e-12),
        1.0,
        0.0,
        1.0,
        mat.lambda_fd_delta_rel,
        mat.lambda_fd_abs_min,
        fld.opacity_cap,
        mat.cv_e_override,
        cfg.numerics.floors.Te,
        std::max(mat.ideal_gas_gamma - 1.0, 1.0e-12),
        false,
        false,
        false,
        state.fld_nlte_f_work.data(),
        state.fld_sigma_a.data(),
        state.fld_sigma_pe.data(),
        state.fld_sigma_R.data(),
        state.fld_nlte_sigma_eff_work.data(),
        state.fld_nlte_sigma_s_eff_work.data(),
        state.fld_nlte_eta_cdf_work.data(),
        state.fld_eta.data(),
        state.fld_nlte_lambda_work.data(),
        0);
    if (result.nan_inf_count != 0 || result.negative_eta_clamp_count != 0) {
      core::log_warning("FLD2D NLTE coefficient clamp counts: nan_inf=" +
                        std::to_string(result.nan_inf_count) +
                        ", eta=" +
                        std::to_string(result.negative_eta_clamp_count));
    }
    return;
  }

  materials::OpacityEvalView opacity_view{};
  opacity_view.rho = state.rho.data();
  opacity_view.Te = state.Te.data();
  opacity_view.sigma_a = state.fld_sigma_a.data();
  opacity_view.sigma_R = state.fld_sigma_R.data();
  opacity_view.n_cells = n_cells;
  opacity_view.n_groups = n_groups;
  opacity_view.opacity_model =
      (mat.opacity_model == "freq_dep_marshak")
          ? materials::kOpacityModelFreqDepMarshak
          : ((mat.opacity_model == "power_law")
                 ? materials::kOpacityModelPowerLaw
                 : materials::kOpacityModelConstant);
  opacity_view.kappa_planck_const =
      (mat.kappa_planck_override >= 0.0)
          ? mat.kappa_planck_override
          : std::max(0.0, mat.kappa_a_constant);
  opacity_view.kappa_rosseland_const = std::max(0.0, mat.kappa_a_constant);
  opacity_view.kappa_floor = fld.opacity_floor;
  opacity_view.kappa_cap = fld.opacity_cap;
  opacity_view.power_law_kappa0 = mat.opacity_power_law_kappa0_cm2_g;
  opacity_view.power_law_alpha_T = mat.opacity_power_law_alpha_T;
  opacity_view.power_law_lambda_rho = mat.opacity_power_law_lambda_rho;
  opacity_view.power_law_T_ref_eV = mat.opacity_power_law_T_ref_eV;
  opacity_view.power_law_rho_ref = mat.opacity_power_law_rho_ref_g_cc;
  const auto n_nonvoid_materials = std::count_if(
      cfg.materials.materials.begin(), cfg.materials.materials.end(),
      [](const auto& m) { return !m.is_void; });
  if (n_nonvoid_materials > 1) {
    state.ensure_cell_material_props(cfg);
    opacity_view.kappa_planck_cell = state.kappa_planck_eff.data();
    opacity_view.kappa_rosseland_cell = state.kappa_rosseland_eff.data();
  }
  std::vector<double> bounds;
  if (opacity_view.opacity_model == materials::kOpacityModelFreqDepMarshak) {
    bounds = radiation_group_bounds(cfg, n_groups);
    state.fld_group_bounds_work.reset(bounds.size());
    auto& cache = fld_group_bounds_cache();
    if (!(cache.valid && bounds == cache.last_bounds &&
          state.fld_group_bounds_work.size() == bounds.size())) {
      cuda_check(cudaMemcpy(state.fld_group_bounds_work.data(),
                            bounds.data(),
                            sizeof(double) * bounds.size(),
                            cudaMemcpyHostToDevice),
                 "FLD2D upload group bounds failed");
      cache.last_bounds = bounds;
      cache.valid = true;
    }
    opacity_view.group_bounds_eV = state.fld_group_bounds_work.data();
  }
  materials::evaluate_opacity_cuda(opacity_view, nullptr);
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    build_eta_from_planck_kernel<<<grid, kBlock>>>(state.Te.data(),
                                                   state.fld_sigma_a.data(),
                                                   planck.device_view(),
                                                   state.fld_eta.data(),
                                                   n_cells,
                                                   n_groups,
                                                   cfg.numerics.floors.Te);
    cuda_check(cudaGetLastError(), "FLD2D eta kernel launch failed");
  }
  if (mat.opacity_model == "constant" || mat.opacity_model == "power_law") {
    const std::size_t scratch_size = static_cast<std::size_t>(total);
    state.fld_nlte_f_work.reset(scratch_size);
    cuda_check(cudaMemcpy(state.fld_sigma_pe.data(),
                          state.fld_sigma_a.data(),
                          sizeof(double) * scratch_size,
                          cudaMemcpyDeviceToDevice),
               "FLD2D copy constant sigma_pe failed");
    std::uint8_t* d_cell_is_void = nullptr;
    if (state.cell_is_void.size() == static_cast<std::size_t>(n_cells)) {
      const std::size_t void_bytes =
          sizeof(std::uint8_t) * static_cast<std::size_t>(n_cells);
      d_cell_is_void = static_cast<std::uint8_t*>(
          core::device_scratch_acquire("fld2d:fleck_cell_is_void", void_bytes));
      cuda_check(cudaMemcpy(d_cell_is_void,
                            state.cell_is_void.data(),
                            void_bytes,
                            cudaMemcpyHostToDevice),
                 "FLD2D Fleck copy cell_is_void failed");
    }
    const int fleck_grid = (n_cells + kBlock - 1) / kBlock;
    if (fleck_grid > 0) {
      compute_fleck_for_fld_kernel<<<fleck_grid, kBlock>>>(
          state.rho.data(),
          state.Te.data(),
          state.zbar.data(),
          d_cell_is_void,
          state.fld_sigma_a.data(),
          (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                   : nullptr,
          state.fld_nlte_f_work.data(),
          n_cells,
          n_groups,
          dt,
          1.0,
          mat.cv_e_override,
          std::max(mat.ideal_gas_gamma, 1.0 + 1.0e-12),
          std::max(mat.A, 1.0e-12),
          (mat.hydro_eos_backend != "exact_ideal_gas")
              ? fld_electron_eos_device_view(mat.eos_tables.get())
              : materials::DeviceEOSTableView{},
          (fld.fleck_cv_source == "table") ? 1 : 0,
          cfg.numerics.floors.Te);
      cuda_check(cudaGetLastError(), "FLD2D Fleck kernel launch failed");
      cuda_check(core::debug_kernel_sync(), "FLD2D Fleck kernel sync failed");
    }
  }
}

void compute_diffusion_coefficients(core::State& state,
                                    const core::Config& cfg,
                                    const int nr,
                                    const int nz,
                                    const int n_groups,
                                    const double* rc_cache,
                                    const double* zc_cache) {
  TENRYU_ASSERT(rc_cache != nullptr && zc_cache != nullptr,
                "FLD2D D_cell requires cached cell centers");
  const int total = nr * nz * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    compute_d_cell_2d_kernel<<<grid, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.rad_E.data(),
        state.fld_sigma_R.data(),
        rc_cache,
        zc_cache,
        state.fld_D_cell.data(),
        nr,
        nz,
        n_groups,
        cfg.radiation.multigroup_diffusion.opacity_floor,
        limiter_id(cfg.radiation.multigroup_diffusion.flux_limiter));
    cuda_check(cudaGetLastError(), "FLD2D D_cell launch failed");
  }
}

void compute_diffusion_coefficients(core::State& state,
                                    const core::Config& cfg,
                                    const int nr,
                                    const int nz,
                                    const int n_groups) {
  compute_cell_centers(state, nr, nz);
  compute_diffusion_coefficients(state,
                                 cfg,
                                 nr,
                                 nz,
                                 n_groups,
                                 state.fld_cell_rc.data(),
                                 state.fld_cell_zc.data());
}

double copy_reduction_scalar(core::State& state) {
  double value = 0.0;
  cuda_check(cudaMemcpy(&value,
                        state.fld_reduction_work.data(),
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy reduction scalar failed");
  return value;
}

void zero_reduction_scalar(core::State& state) {
  cuda_check(cudaMemset(state.fld_reduction_work.data(), 0, sizeof(double)),
             "FLD2D zero reduction scalar failed");
}

std::vector<int> fld_substage_audit_modes(const int nr) {
  std::vector<int> modes;
  const int max_mode = nr / 2;
  const int raw_modes[2] = {14, 15};
  for (const int m : raw_modes) {
    if (m >= 0 && m <= max_mode) {
      modes.push_back(m);
    }
  }
  return modes;
}

std::vector<int> fld_substage_audit_rows(const int nz) {
  std::vector<int> rows;
  const int raw_rows[3] = {nz - 1, nz - 2, nz - 3};
  for (const int j : raw_rows) {
    if (j >= 0 && j < nz) {
      rows.push_back(j);
    }
  }
  return rows;
}

void append_fld_substage_audit_field(
    Fld2DWorkspace& ws,
    const double* const field,
    const diagnostics::FldSubstageAuditFieldLayout layout,
    const diagnostics::FldSubstageAuditSubstageId substage,
    const diagnostics::FldSubstageAuditFieldId field_id,
    const int nr,
    const int nz,
    const int n_groups,
    const int outer_iter,
    const double solver_residual_l2_rel = 0.0,
    const double solver_residual_max = 0.0) {
  if (field == nullptr || nr <= 0 || nz <= 0 || n_groups <= 0) {
    return;
  }
  const std::vector<int> modes = fld_substage_audit_modes(nr);
  const std::vector<int> rows = fld_substage_audit_rows(nz);
  if (modes.empty() || rows.empty()) {
    return;
  }
  const std::size_t n_m = modes.size();
  const std::size_t n_j = rows.size();
  const std::size_t n_records =
      static_cast<std::size_t>(n_groups) * n_m * n_j;
  ws.d_audit_m_targets.resize(n_m * sizeof(int));
  ws.d_audit_j_targets.resize(n_j * sizeof(int));
  ws.d_audit_records.resize(n_records *
                            sizeof(diagnostics::FldSubstageAuditRecord));
  cuda_check(cudaMemcpy(ws.d_audit_m_targets.as<int>(),
                        modes.data(),
                        n_m * sizeof(int),
                        cudaMemcpyHostToDevice),
             "FLD2D substage audit copy m targets failed");
  cuda_check(cudaMemcpy(ws.d_audit_j_targets.as<int>(),
                        rows.data(),
                        n_j * sizeof(int),
                        cudaMemcpyHostToDevice),
             "FLD2D substage audit copy j targets failed");

  const int threads = fld_audit_reduction_thread_count(nr);
  const std::size_t shared_bytes =
      3U * static_cast<std::size_t>(threads) * sizeof(double);
  fld_substage_audit_fourier_kernel<<<static_cast<int>(n_records),
                                      threads,
                                      shared_bytes>>>(
      field,
      static_cast<int>(layout),
      ws.d_audit_m_targets.as<int>(),
      ws.d_audit_j_targets.as<int>(),
      static_cast<int>(n_m),
      static_cast<int>(n_j),
      nr,
      nz,
      n_groups,
      static_cast<std::uint8_t>(substage),
      static_cast<std::uint8_t>(field_id),
      outer_iter,
      solver_residual_l2_rel,
      solver_residual_max,
      ws.d_audit_records.as<diagnostics::FldSubstageAuditRecord>());
  cuda_check(cudaGetLastError(), "FLD2D substage audit Fourier launch failed");

  std::vector<diagnostics::FldSubstageAuditRecord> host_records(n_records);
  cuda_check(cudaMemcpy(host_records.data(),
                        ws.d_audit_records.as<diagnostics::FldSubstageAuditRecord>(),
                        n_records *
                            sizeof(diagnostics::FldSubstageAuditRecord),
                        cudaMemcpyDeviceToHost),
             "FLD2D substage audit copy records failed");
  auto& batch = fld_substage_audit_batch();
  batch.insert(batch.end(), host_records.begin(), host_records.end());
}

void compute_fld_substage_boundary_audit_fields(
    Fld2DWorkspace& ws,
    const core::State& state,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    const int z_top_bc,
    const double T_supply_z_top_eV,
    const int state_supply_boundary_policy,
    const int fld_limiter_id,
    const double rho_supply_z_top) {
  const int n_rows = nr * nz * n_groups;
  if (n_rows <= 0) {
    return;
  }
  const std::size_t bytes =
      sizeof(double) * static_cast<std::size_t>(n_rows);
  ws.d_audit_boundary_coeff.resize(bytes);
  ws.d_audit_boundary_dt_coeff.resize(bytes);
  ws.d_audit_boundary_source.resize(bytes);
  ws.d_audit_boundary_diag.resize(bytes);
  const int grid = (n_rows + kBlock - 1) / kBlock;
  fld_substage_audit_top_boundary_kernel<<<grid, kBlock>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.fld_D_cell.data(),
      state.rho.data(),
      state.fld_sigma_R.data(),
      ws.d_audit_boundary_coeff.as<double>(),
      ws.d_audit_boundary_dt_coeff.as<double>(),
      ws.d_audit_boundary_source.as<double>(),
      ws.d_audit_boundary_diag.as<double>(),
      nr,
      nz,
      n_groups,
      dt,
      z_top_bc,
      T_supply_z_top_eV,
      state_supply_boundary_policy,
      fld_limiter_id,
      rho_supply_z_top);
  cuda_check(cudaGetLastError(),
             "FLD2D substage audit top-boundary launch failed");
}

void reset_fld_step_diagnostics(core::State& state) {
  state.fld_clamp_hits_step = 0;
  state.fld_clamp_energy_delta_step = 0.0;
  state.fld_min_x_raw_step = 0.0;
  state.fld_cg_true_residual_l2_rel_step = 0.0;
  state.fld_cg_true_residual_max_step = 0.0;
  state.fld_E_solver_step = 0.0;
  state.fld_cg_true_residual_l2_rel_RAW_step = 0.0;
  state.fld_cg_true_residual_max_RAW_step = 0.0;
  state.fld_E_solver_RAW_step = 0.0;
  state.fld_csr_diag_min_pos_step = std::numeric_limits<double>::infinity();
  state.fld_csr_diag_max_step = 0.0;
  state.fld_csr_weak_diag_dom_count_step = 0;
  state.fld_csr_nonfinite_count_step = 0;
  state.fld_gershgorin_lower_min_step = std::numeric_limits<double>::infinity();
  state.fld_gershgorin_upper_max_step = 0.0;
  state.fld_cg_pAp_min_step = std::numeric_limits<double>::infinity();
  state.fld_cg_nonpos_pAp_count_step = 0;
  state.fld_cg_nonfinite_count_step = 0;
  state.fld_cg_recurrent_resid_last_check_max_step = 0.0;
  state.fld_Dcell_min_step = std::numeric_limits<double>::infinity();
  state.fld_Dcell_max_step = 0.0;
  state.fld_Dcell_zero_count_step = 0;
  state.fld_Dcell_nonfinite_count_step = 0;
  state.fld_face_skip_D_count_step = 0;
  state.fld_face_skip_nonfinite_count_step = 0;
  state.fld_face_skip_dist_count_step = 0;
  state.fld_diag_fallback_count_step = 0;
  state.fld_pair_symmetry_max_diff_step = 0.0;
  state.fld_pair_symmetry_violation_count_step = 0;
  state.fld_newton_converged_count_step = 0;
  state.fld_newton_cap_hit_count_step = 0;
  state.fld_newton_invalid_count_step = 0;
  state.fld_newton_resid_abs_max_step = 0.0;
  state.fld_newton_resid_rel_max_step = 0.0;
  state.fld_newton_reject_count_step = 0;
  state.fld_newton_reject_resid_rel_max_step = 0.0;
  state.fld_rogue_max_abs_x_raw_step = 0.0;
  state.fld_rogue_max_abs_x_raw_cell_step = -1;
  state.fld_rogue_max_abs_x_raw_group_step = -1;
  state.fld_rogue_min_x_raw_step = 0.0;
  state.fld_rogue_min_x_raw_cell_step = -1;
  state.fld_rogue_min_x_raw_group_step = -1;
  state.fld_rogue_max_rad_E_step = 0.0;
  state.fld_rogue_max_rad_E_cell_step = -1;
  state.fld_rogue_max_rad_E_group_step = -1;
  state.fld_rogue_max_r_true_step = 0.0;
  state.fld_rogue_max_r_true_cell_step = -1;
  state.fld_rogue_max_r_true_group_step = -1;
  state.fld_rogue_max_E_solver_row_step = 0.0;
  state.fld_rogue_max_E_solver_row_cell_step = -1;
  state.fld_rogue_max_E_solver_row_group_step = -1;
  state.fld_escaped_step = 0.0;
  state.fld_marshak_in_step = 0.0;
  state.fld_state_supply_in_step = 0.0;
  state.fld_state_supply_out_step = 0.0;
  state.fld_state_supply_net_step = 0.0;
}

void initialize_rogue_records(Fld2DWorkspace& ws) {
  RogueRecord records[kNumRogueRecords] = {};
  for (int i = 0; i < kNumRogueRecords; ++i) {
    records[i].value = 0.0;
    records[i].cell_idx = -1;
    records[i].group_idx = -1;
  }
  records[kRogueMinXRaw].value = std::numeric_limits<double>::infinity();
  ws.d_rogue_records.resize(sizeof(records));
  cuda_check(cudaMemcpy(ws.d_rogue_records.ptr,
                        records,
                        sizeof(records),
                        cudaMemcpyHostToDevice),
             "FLD2D initialize rogue diagnostics failed");
}

void copy_rogue_records_to_state(core::State& state, Fld2DWorkspace& ws) {
  if (ws.d_rogue_records.size < sizeof(RogueRecord) * kNumRogueRecords) {
    return;
  }
  RogueRecord records[kNumRogueRecords] = {};
  cuda_check(cudaMemcpy(records,
                        ws.d_rogue_records.ptr,
                        sizeof(records),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy rogue diagnostics failed");
  state.fld_rogue_max_abs_x_raw_step = records[kRogueMaxAbsXRaw].value;
  state.fld_rogue_max_abs_x_raw_cell_step = records[kRogueMaxAbsXRaw].cell_idx;
  state.fld_rogue_max_abs_x_raw_group_step = records[kRogueMaxAbsXRaw].group_idx;
  if (records[kRogueMinXRaw].cell_idx >= 0) {
    state.fld_rogue_min_x_raw_step = records[kRogueMinXRaw].value;
    state.fld_rogue_min_x_raw_cell_step = records[kRogueMinXRaw].cell_idx;
    state.fld_rogue_min_x_raw_group_step = records[kRogueMinXRaw].group_idx;
  }
  state.fld_rogue_max_rad_E_step = records[kRogueMaxRadE].value;
  state.fld_rogue_max_rad_E_cell_step = records[kRogueMaxRadE].cell_idx;
  state.fld_rogue_max_rad_E_group_step = records[kRogueMaxRadE].group_idx;
  state.fld_rogue_max_r_true_step = records[kRogueMaxRTrue].value;
  state.fld_rogue_max_r_true_cell_step = records[kRogueMaxRTrue].cell_idx;
  state.fld_rogue_max_r_true_group_step = records[kRogueMaxRTrue].group_idx;
  state.fld_rogue_max_E_solver_row_step = records[kRogueMaxESolverRow].value;
  state.fld_rogue_max_E_solver_row_cell_step =
      records[kRogueMaxESolverRow].cell_idx;
  state.fld_rogue_max_E_solver_row_group_step =
      records[kRogueMaxESolverRow].group_idx;
}

void update_dcell_diagnostics(core::State& state,
                              Fld2DWorkspace& ws,
                              const int n_total) {
  if (n_total <= 0) {
    return;
  }
  const double init_doubles[2] = {std::numeric_limits<double>::infinity(), 0.0};
  ws.d_diag_scalars.resize(sizeof(init_doubles));
  ws.d_diag_ints.resize(2U * sizeof(int));
  cuda_check(cudaMemcpy(ws.d_diag_scalars.ptr,
                        init_doubles,
                        sizeof(init_doubles),
                        cudaMemcpyHostToDevice),
             "FLD2D initialize D_cell diagnostic scalars failed");
  cuda_check(cudaMemset(ws.d_diag_ints.ptr, 0, 2U * sizeof(int)),
             "FLD2D zero D_cell diagnostic counts failed");
  const int grid = (n_total + kBlock - 1) / kBlock;
  dcell_stats_kernel<<<grid, kBlock>>>(state.fld_D_cell.data(),
                                       ws.d_diag_scalars.as<double>(),
                                       ws.d_diag_ints.as<int>(),
                                       n_total);
  cuda_check(cudaGetLastError(), "FLD2D D_cell diagnostics launch failed");
  double values[2] = {std::numeric_limits<double>::infinity(), 0.0};
  int counts[2] = {0, 0};
  cuda_check(cudaMemcpy(values,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy D_cell diagnostic scalars failed");
  cuda_check(cudaMemcpy(counts,
                        ws.d_diag_ints.as<int>(),
                        sizeof(counts),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy D_cell diagnostic counts failed");
  if (std::isfinite(values[0])) {
    state.fld_Dcell_min_step = std::min(state.fld_Dcell_min_step, values[0]);
  }
  state.fld_Dcell_max_step = std::max(state.fld_Dcell_max_step,
                                      finite_or_zero(values[1]));
  state.fld_Dcell_zero_count_step =
      std::max(state.fld_Dcell_zero_count_step, counts[0]);
  state.fld_Dcell_nonfinite_count_step =
      std::max(state.fld_Dcell_nonfinite_count_step, counts[1]);
}

void update_csr_matrix_diagnostics(core::State& state,
                                   Fld2DWorkspace& ws,
                                   const int n_rows) {
  if (n_rows <= 0) {
    return;
  }
  const double init_doubles[4] = {
      std::numeric_limits<double>::infinity(),
      0.0,
      std::numeric_limits<double>::infinity(),
      0.0};
  ws.d_diag_scalars.resize(sizeof(init_doubles));
  ws.d_diag_ints.resize(2U * sizeof(int));
  cuda_check(cudaMemcpy(ws.d_diag_scalars.ptr,
                        init_doubles,
                        sizeof(init_doubles),
                        cudaMemcpyHostToDevice),
             "FLD2D initialize CSR diagnostic scalars failed");
  cuda_check(cudaMemset(ws.d_diag_ints.ptr, 0, 2U * sizeof(int)),
             "FLD2D zero CSR diagnostic counts failed");
  const int grid = (n_rows + kBlock - 1) / kBlock;
  csr_matrix_stats_kernel<<<grid, kBlock>>>(ws.d_col_indices.as<int>(),
                                            ws.d_values.as<double>(),
                                            ws.d_rhs.as<double>(),
                                            ws.d_diag_scalars.as<double>(),
                                            ws.d_diag_ints.as<int>(),
                                            n_rows);
  cuda_check(cudaGetLastError(), "FLD2D CSR diagnostics launch failed");
  double values[4] = {
      std::numeric_limits<double>::infinity(),
      0.0,
      std::numeric_limits<double>::infinity(),
      0.0};
  int counts[2] = {0, 0};
  cuda_check(cudaMemcpy(values,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy CSR diagnostic scalars failed");
  cuda_check(cudaMemcpy(counts,
                        ws.d_diag_ints.as<int>(),
                        sizeof(counts),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy CSR diagnostic counts failed");
  if (std::isfinite(values[0])) {
    state.fld_csr_diag_min_pos_step =
        std::min(state.fld_csr_diag_min_pos_step, values[0]);
  }
  state.fld_csr_diag_max_step =
      std::max(state.fld_csr_diag_max_step, finite_or_zero(values[1]));
  if (std::isfinite(values[2])) {
    state.fld_gershgorin_lower_min_step =
        std::min(state.fld_gershgorin_lower_min_step, values[2]);
  }
  state.fld_gershgorin_upper_max_step =
      std::max(state.fld_gershgorin_upper_max_step, finite_or_zero(values[3]));
  state.fld_csr_weak_diag_dom_count_step += counts[0];
  state.fld_csr_nonfinite_count_step += counts[1];
}

void update_pair_symmetry_diagnostics(core::State& state,
                                      Fld2DWorkspace& ws,
                                      const int n_rows) {
  if (n_rows <= 0) {
    return;
  }
  ws.d_diag_scalars.resize(sizeof(double));
  ws.d_diag_ints.resize(sizeof(int));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, sizeof(double)),
             "FLD2D zero pair-symmetry scalar failed");
  cuda_check(cudaMemset(ws.d_diag_ints.ptr, 0, sizeof(int)),
             "FLD2D zero pair-symmetry count failed");
  const int total = n_rows * (kCsrEntriesPerRow - 1);
  const int grid = (total + kBlock - 1) / kBlock;
  pair_symmetry_audit_kernel<<<grid, kBlock>>>(ws.d_col_indices.as<int>(),
                                               ws.d_values.as<double>(),
                                               ws.d_diag_scalars.as<double>(),
                                               ws.d_diag_ints.as<int>(),
                                               n_rows);
  cuda_check(cudaGetLastError(), "FLD2D pair-symmetry diagnostics launch failed");
  double max_diff = 0.0;
  int count = 0;
  cuda_check(cudaMemcpy(&max_diff,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(double),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy pair-symmetry scalar failed");
  cuda_check(cudaMemcpy(&count,
                        ws.d_diag_ints.as<int>(),
                        sizeof(int),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy pair-symmetry count failed");
  state.fld_pair_symmetry_max_diff_step =
      std::max(state.fld_pair_symmetry_max_diff_step, finite_or_zero(max_diff));
  state.fld_pair_symmetry_violation_count_step += count;
}

double device_dot(const double* a,
                  const double* b,
                  const int n,
                  parallel::DeviceArray& scalar) {
  scalar.resize(sizeof(double));
  if (n <= 0) {
    cuda_check(cudaMemset(scalar.ptr, 0, sizeof(double)), "FLD2D zero dot failed");
  } else if (dot_block_count(n) <= kDotSingleBlockMaxBlocks) {
    dot_single_block_kernel<<<1, kBlock>>>(a, b, scalar.as<double>(), n);
    cuda_check(cudaGetLastError(), "FLD2D dot launch failed");
  } else {
    auto& ws = fld_2d_workspace();
    const int nblocks = dot_block_count(n);
    double* partials = ensure_dot_partials(ws, nblocks);
    dot_partials_kernel<<<nblocks, kBlock>>>(a, b, partials, n);
    cuda_check(cudaGetLastError(), "FLD2D dot partials launch failed");
    dot_finalize_kernel<<<1, kBlock>>>(partials, scalar.as<double>(), nblocks);
    cuda_check(cudaGetLastError(), "FLD2D dot finalize launch failed");
  }
  double value = 0.0;
  cuda_check(cudaMemcpy(&value, scalar.as<double>(), sizeof(double), cudaMemcpyDeviceToHost),
             "FLD2D copy dot failed");
  return finite_or_zero(value);
}

void device_dot_to(const double* a,
                   const double* b,
                   const int n,
                   double* out) {
  if (n <= 0) {
    cuda_check(cudaMemset(out, 0, sizeof(double)), "FLD2D zero device dot failed");
  } else if (dot_block_count(n) <= kDotSingleBlockMaxBlocks) {
    dot_single_block_kernel<<<1, kBlock>>>(a, b, out, n);
    cuda_check(cudaGetLastError(), "FLD2D device dot launch failed");
  } else {
    auto& ws = fld_2d_workspace();
    const int nblocks = dot_block_count(n);
    double* partials = ensure_dot_partials(ws, nblocks);
    dot_partials_kernel<<<nblocks, kBlock>>>(a, b, partials, n);
    cuda_check(cudaGetLastError(), "FLD2D device dot partials launch failed");
    dot_finalize_kernel<<<1, kBlock>>>(partials, out, nblocks);
    cuda_check(cudaGetLastError(), "FLD2D device dot finalize launch failed");
  }
}

// Stream-parameterized twin of device_dot_to for CUDA-graph capture.
// Must NOT allocate: the caller pre-sizes ws.d_dot_partials before
// capture begins (allocation is illegal inside stream capture).
void device_dot_to_stream(cudaStream_t stream,
                          Fld2DWorkspace& ws,
                          const double* a,
                          const double* b,
                          const int n,
                          double* out) {
  if (n <= 0) {
    cuda_check(cudaMemsetAsync(out, 0, sizeof(double), stream),
               "FLD2D zero device dot (stream) failed");
  } else if (dot_block_count(n) <= kDotSingleBlockMaxBlocks) {
    dot_single_block_kernel<<<1, kBlock, 0, stream>>>(a, b, out, n);
    cuda_check(cudaGetLastError(),
               "FLD2D device dot (stream) launch failed");
  } else {
    const int nblocks = dot_block_count(n);
    double* partials = ws.d_dot_partials.as<double>();
    dot_partials_kernel<<<nblocks, kBlock, 0, stream>>>(a, b, partials, n);
    cuda_check(cudaGetLastError(),
               "FLD2D device dot partials (stream) launch failed");
    dot_finalize_kernel<<<1, kBlock, 0, stream>>>(partials, out, nblocks);
    cuda_check(cudaGetLastError(),
               "FLD2D device dot finalize (stream) launch failed");
  }
}

// Owned-rows masked dot + MPI_Allreduce(SUM): the T-sum-class global scalar
// for the Option-C distributed CG (NUMERICS §12.7.3 — global-sum order is
// rank-count dependent, run-to-run stable). Returns the GLOBAL value on the
// host; when d_out is non-null the global value is also uploaded there so
// device-resident consumers (alpha/beta update kernels, breakdown status)
// read rank-uniform scalars. Caller guards ws.mpi_n_ranks > 1.
double device_dot_owned_global(Fld2DWorkspace& ws,
                               const double* a,
                               const double* b,
                               double* d_out) {
  const int owned = ws.mpi_c_end - ws.mpi_c_begin;
  const int total = owned * ws.mpi_n_groups;
  double local = 0.0;
  if (total > 0) {
    const int nblocks = dot_block_count(total);
    double* partials = ensure_dot_partials(ws, nblocks);
    dot_owned_partials_kernel<<<nblocks, kBlock>>>(
        a, b, partials, ws.mpi_n_cells, ws.mpi_c_begin, owned, total);
    cuda_check(cudaGetLastError(), "FLD2D owned dot launch failed");
    ws.d_scalar.resize(sizeof(double));
    dot_finalize_kernel<<<1, kBlock>>>(partials, ws.d_scalar.as<double>(),
                                       nblocks);
    cuda_check(cudaGetLastError(), "FLD2D owned dot finalize launch failed");
    cuda_check(cudaMemcpy(&local, ws.d_scalar.as<double>(), sizeof(double),
                          cudaMemcpyDeviceToHost),
               "FLD2D copy owned dot failed");
    local = finite_or_zero(local);
  }
  const parallel::Reduction reduction(ws.mpi_n_ranks);
  const double global = reduction.allreduce_sum(local);
  if (d_out != nullptr) {
    cuda_check(cudaMemcpy(d_out, &global, sizeof(double),
                          cudaMemcpyHostToDevice),
               "FLD2D upload global dot failed");
  }
  return global;
}

void prepare_zline_preconditioner(Fld2DWorkspace& ws,
                                  const int* col_indices,
                                  const double* values,
                                  double* z,
                                  const int n_rows,
                                  const int nr,
                                  const int nz,
                                  const int n_groups) {
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D z-line preconditioner requires positive dimensions");
  const int n_cells = nr * nz;
  TENRYU_ASSERT(n_rows == n_cells * n_groups,
                "FLD2D z-line preconditioner row count mismatch");
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rows);
  ws.d_line_dl.resize(bytes);
  ws.d_line_d.resize(bytes);
  ws.d_line_du.resize(bytes);

  const int grid = (n_rows + kBlock - 1) / kBlock;
  build_zline_tridiag_from_csr_kernel<<<grid, kBlock>>>(col_indices,
                                                        values,
                                                        ws.d_line_dl.as<double>(),
                                                        ws.d_line_d.as<double>(),
                                                        ws.d_line_du.as<double>(),
                                                        nr,
                                                        nz,
                                                        n_cells,
                                                        n_rows);
  cuda_check(cudaGetLastError(), "FLD2D build z-line tridiag launch failed");

  std::size_t buffer_size = 0U;
  cusparse_check(cusparseDgtsv2StridedBatch_bufferSizeExt(
                    ws.handle,
                    nz,
                    ws.d_line_dl.as<double>(),
                    ws.d_line_d.as<double>(),
                    ws.d_line_du.as<double>(),
                    z,
                    n_groups * nr,
                    nz,
                    &buffer_size),
                "FLD2D z-line cuSPARSE buffer size failed");
  ws.d_gtsv2_buffer.resize(buffer_size);
}

void apply_zline_preconditioner(Fld2DWorkspace& ws,
                                const double* r,
                                double* z,
                                const int n_rows,
                                const int nz,
                                const int batch_count,
                                const int row_offset = 0) {
  // §6q.8 owned-row windowing: row_offset/batch_count select the owned
  // z-line span (r-slab lines are contiguous); serial passes 0/full.
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rows);
  // gtsv2StridedBatch treats dl/d/du as const (verified CUDA 12.0/12.6): read the
  // master coefficient arrays directly; only the RHS must be copied (gtsv overwrites
  // z in place with the solution).
  cuda_check(cudaMemcpyAsync(z + row_offset, r + row_offset, bytes,
                             cudaMemcpyDeviceToDevice, 0),
             "FLD2D copy z-line rhs failed");
  cusparse_check(cusparseDgtsv2StridedBatch(ws.handle,
                                            nz,
                                            ws.d_line_dl.as<double>() + row_offset,
                                            ws.d_line_d.as<double>() + row_offset,
                                            ws.d_line_du.as<double>() + row_offset,
                                            z + row_offset,
                                            batch_count,
                                            nz,
                                            ws.d_gtsv2_buffer.ptr),
                "FLD2D z-line preconditioner solve failed");
}

void apply_zline_preconditioner(cudaStream_t stream,
                                Fld2DWorkspace& ws,
                                const double* r,
                                double* z,
                                const int n_rows,
                                const int nz,
                                const int batch_count) {
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rows);
  // gtsv2StridedBatch treats dl/d/du as const (verified CUDA 12.0/12.6): read the
  // master coefficient arrays directly; only the RHS must be copied (gtsv overwrites
  // z in place with the solution).
  cuda_check(cudaMemcpyAsync(z, r, bytes, cudaMemcpyDeviceToDevice, stream),
             "FLD2D copy z-line rhs failed");
  cusparse_check(cusparseDgtsv2StridedBatch(ws.handle,
                                            nz,
                                            ws.d_line_dl.as<double>(),
                                            ws.d_line_d.as<double>(),
                                            ws.d_line_du.as<double>(),
                                            z,
                                            batch_count,
                                            nz,
                                            ws.d_gtsv2_buffer.ptr),
                "FLD2D z-line preconditioner solve failed");
}

int rgmg_level_n_rows(const RgmgLevel& level) {
  return level.nr * level.nz * level.n_groups;
}

void resize_rgmg_level(RgmgLevel& level,
                       const int nr,
                       const int nz,
                       const int n_groups) {
  level.nr = nr;
  level.nz = nz;
  level.n_groups = n_groups;
  const std::size_t bytes =
      sizeof(double) *
      static_cast<std::size_t>(rgmg_level_n_rows(level));
  level.diag.resize(bytes);
  level.ar_left.resize(bytes);
  level.ar_right.resize(bytes);
  level.az_m.resize(bytes);
  level.az_p.resize(bytes);
  level.b.resize(bytes);
  level.x.resize(bytes);
  level.res.resize(bytes);
  level.corr.resize(bytes);
}

void build_rgmg_hierarchy(Fld2DWorkspace& ws,
                          const int* row_offsets,
                          const int* col_indices,
                          const double* values,
                          const int nr,
                          const int nz,
                          const int n_groups) {
  TENRYU_ASSERT(row_offsets != nullptr && col_indices != nullptr &&
                    values != nullptr,
                "FLD2D RGMG hierarchy requires non-null fine CSR pointers");
  TENRYU_ASSERT(nr > 0 && nz > 0 && n_groups > 0,
                "FLD2D RGMG hierarchy requires positive dimensions");
  TENRYU_ASSERT((nr & (nr - 1)) == 0,
                "FLD2D RGMG hierarchy requires power-of-two nr");
  if (ws.handle == nullptr) {
    cusparse_check(cusparseCreate(&ws.handle), "FLD2D cusparseCreate failed");
  }

  int n_levels = 1;
  for (int nr_l = nr; nr_l > 1; nr_l >>= 1) {
    ++n_levels;
  }
  ws.rgmg_levels.resize(static_cast<std::size_t>(n_levels));
  ws.rgmg_n_levels = n_levels;

  std::size_t max_gtsv_buffer = 0U;
  for (int l = 0; l < n_levels; ++l) {
    RgmgLevel& level = ws.rgmg_levels[static_cast<std::size_t>(l)];
    resize_rgmg_level(level, nr >> l, nz, n_groups);
  }

  RgmgLevel& level0 = ws.rgmg_levels[0];
  extract_fld_2d_csr_stencil(row_offsets,
                             col_indices,
                             values,
                             level0.diag.as<double>(),
                             level0.ar_left.as<double>(),
                             level0.ar_right.as<double>(),
                             level0.az_m.as<double>(),
                             level0.az_p.as<double>(),
                             nr,
                             nz,
                             n_groups);
  for (int l = 1; l < n_levels; ++l) {
    RgmgLevel& fine = ws.rgmg_levels[static_cast<std::size_t>(l - 1)];
    RgmgLevel& coarse = ws.rgmg_levels[static_cast<std::size_t>(l)];
    aggregate_fld_2d_stencil_pairwise_r_galerkin(fine.diag.as<double>(),
                                                 fine.ar_left.as<double>(),
                                                 fine.ar_right.as<double>(),
                                                 fine.az_m.as<double>(),
                                                 fine.az_p.as<double>(),
                                                 coarse.diag.as<double>(),
                                                 coarse.ar_left.as<double>(),
                                                 coarse.ar_right.as<double>(),
                                                 coarse.az_m.as<double>(),
                                                 coarse.az_p.as<double>(),
                                                 fine.nr,
                                                 fine.nz,
                                                 fine.n_groups);
  }

  for (int l = 0; l < n_levels; ++l) {
    RgmgLevel& level = ws.rgmg_levels[static_cast<std::size_t>(l)];
    std::size_t buffer_size = 0U;
    cusparse_check(cusparseDgtsv2StridedBatch_bufferSizeExt(
                      ws.handle,
                      level.nz,
                      level.az_m.as<double>(),
                      level.diag.as<double>(),
                      level.az_p.as<double>(),
                      level.corr.as<double>(),
                      level.n_groups * level.nr,
                      level.nz,
                      &buffer_size),
                  "FLD2D RGMG cuSPARSE buffer size failed");
    max_gtsv_buffer = std::max(max_gtsv_buffer, buffer_size);
  }
  ws.rgmg_gtsv_buffer.resize(max_gtsv_buffer);
}

void rgmg_solve_z_lines(Fld2DWorkspace& ws,
                        RgmgLevel& level,
                        double* rhs_solution) {
  TENRYU_ASSERT(rhs_solution != nullptr,
                "FLD2D RGMG z-line solve requires non-null RHS");
  cusparse_check(cusparseDgtsv2StridedBatch(ws.handle,
                                            level.nz,
                                            level.az_m.as<double>(),
                                            level.diag.as<double>(),
                                            level.az_p.as<double>(),
                                            rhs_solution,
                                            level.n_groups * level.nr,
                                            level.nz,
                                            ws.rgmg_gtsv_buffer.ptr),
                "FLD2D RGMG z-line solve failed");
}

void rgmg_solve_z_lines(cudaStream_t stream,
                        Fld2DWorkspace& ws,
                        RgmgLevel& level,
                        double* rhs_solution) {
  TENRYU_ASSERT(rhs_solution != nullptr,
                "FLD2D RGMG z-line solve requires non-null RHS");
  cusparse_check(cusparseDgtsv2StridedBatch(ws.handle,
                                            level.nz,
                                            level.az_m.as<double>(),
                                            level.diag.as<double>(),
                                            level.az_p.as<double>(),
                                            rhs_solution,
                                            level.n_groups * level.nr,
                                            level.nz,
                                            ws.rgmg_gtsv_buffer.ptr),
                "FLD2D RGMG z-line solve failed");
}

void rgmg_smooth(Fld2DWorkspace& ws,
                 const int level_index,
                 const double omega,
                 const bool x_is_zero = false) {
  TENRYU_ASSERT(level_index >= 0 && level_index < ws.rgmg_n_levels,
                "FLD2D RGMG smoother level index out of range");
  TENRYU_ASSERT(std::isfinite(omega) && omega > 0.0,
                "FLD2D RGMG smoother requires positive finite omega");
  RgmgLevel& level = ws.rgmg_levels[static_cast<std::size_t>(level_index)];
  const int n_rows = rgmg_level_n_rows(level);
  const std::size_t bytes =
      sizeof(double) * static_cast<std::size_t>(n_rows);
  if (x_is_zero) {
    cuda_check(cudaMemcpyAsync(level.corr.as<double>(),
                          level.b.as<double>(),
                          bytes,
                          cudaMemcpyDeviceToDevice,
                          0),
               "FLD2D RGMG copy zero-x smoother RHS failed");
  } else {
    fld_2d_stencil_residual(level.diag.as<double>(),
                            level.ar_left.as<double>(),
                            level.ar_right.as<double>(),
                            level.az_m.as<double>(),
                            level.az_p.as<double>(),
                            level.b.as<double>(),
                            level.x.as<double>(),
                            level.res.as<double>(),
                            level.nr,
                            level.nz,
                            level.n_groups);
    cuda_check(cudaMemcpyAsync(level.corr.as<double>(),
                          level.res.as<double>(),
                          bytes,
                          cudaMemcpyDeviceToDevice,
                          0),
               "FLD2D RGMG copy smoother residual failed");
  }
  rgmg_solve_z_lines(ws, level, level.corr.as<double>());
  fld_2d_axpy(omega, level.corr.as<double>(), level.x.as<double>(), n_rows);
}

void rgmg_smooth(cudaStream_t stream,
                 Fld2DWorkspace& ws,
                 const int level_index,
                 const double omega,
                 const bool x_is_zero) {
  TENRYU_ASSERT(level_index >= 0 && level_index < ws.rgmg_n_levels,
                "FLD2D RGMG smoother level index out of range");
  TENRYU_ASSERT(std::isfinite(omega) && omega > 0.0,
                "FLD2D RGMG smoother requires positive finite omega");
  RgmgLevel& level = ws.rgmg_levels[static_cast<std::size_t>(level_index)];
  const int n_rows = rgmg_level_n_rows(level);
  const std::size_t bytes =
      sizeof(double) * static_cast<std::size_t>(n_rows);
  if (x_is_zero) {
    cuda_check(cudaMemcpyAsync(level.corr.as<double>(),
                          level.b.as<double>(),
                          bytes,
                          cudaMemcpyDeviceToDevice,
                          stream),
               "FLD2D RGMG copy zero-x smoother RHS failed");
  } else {
    fld_2d_stencil_residual(stream,
                            level.diag.as<double>(),
                            level.ar_left.as<double>(),
                            level.ar_right.as<double>(),
                            level.az_m.as<double>(),
                            level.az_p.as<double>(),
                            level.b.as<double>(),
                            level.x.as<double>(),
                            level.res.as<double>(),
                            level.nr,
                            level.nz,
                            level.n_groups);
    cuda_check(cudaMemcpyAsync(level.corr.as<double>(),
                          level.res.as<double>(),
                          bytes,
                          cudaMemcpyDeviceToDevice,
                          stream),
               "FLD2D RGMG copy smoother residual failed");
  }
  rgmg_solve_z_lines(stream, ws, level, level.corr.as<double>());
  fld_2d_axpy(stream,
              omega,
              level.corr.as<double>(),
              level.x.as<double>(),
              n_rows);
}

void rgmg_coarsest_exact_solve(Fld2DWorkspace& ws, const int level_index) {
  TENRYU_ASSERT(level_index >= 0 && level_index < ws.rgmg_n_levels,
                "FLD2D RGMG coarsest level index out of range");
  RgmgLevel& level = ws.rgmg_levels[static_cast<std::size_t>(level_index)];
  TENRYU_ASSERT(level.nr == 1,
                "FLD2D RGMG coarsest exact solve requires nr == 1");
  const int n_rows = rgmg_level_n_rows(level);
  const std::size_t bytes =
      sizeof(double) * static_cast<std::size_t>(n_rows);
  cuda_check(cudaMemcpyAsync(level.x.as<double>(),
                        level.b.as<double>(),
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        0),
             "FLD2D RGMG copy coarsest RHS failed");
  rgmg_solve_z_lines(ws, level, level.x.as<double>());
}

void rgmg_coarsest_exact_solve(cudaStream_t stream,
                               Fld2DWorkspace& ws,
                               const int level_index) {
  TENRYU_ASSERT(level_index >= 0 && level_index < ws.rgmg_n_levels,
                "FLD2D RGMG coarsest level index out of range");
  RgmgLevel& level = ws.rgmg_levels[static_cast<std::size_t>(level_index)];
  TENRYU_ASSERT(level.nr == 1,
                "FLD2D RGMG coarsest exact solve requires nr == 1");
  const int n_rows = rgmg_level_n_rows(level);
  const std::size_t bytes =
      sizeof(double) * static_cast<std::size_t>(n_rows);
  cuda_check(cudaMemcpyAsync(level.x.as<double>(),
                        level.b.as<double>(),
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        stream),
             "FLD2D RGMG copy coarsest RHS failed");
  rgmg_solve_z_lines(stream, ws, level, level.x.as<double>());
}

void rgmg_vcycle(Fld2DWorkspace& ws, const double omega) {
  TENRYU_ASSERT(ws.rgmg_n_levels > 0,
                "FLD2D RGMG V-cycle requires a built hierarchy");
  TENRYU_ASSERT(ws.rgmg_n_levels ==
                    static_cast<int>(ws.rgmg_levels.size()),
                "FLD2D RGMG level count mismatch");
  RgmgLevel& finest = ws.rgmg_levels[0];
  cuda_check(cudaMemsetAsync(finest.x.ptr,
                        0,
                        sizeof(double) *
                            static_cast<std::size_t>(rgmg_level_n_rows(finest)),
                        0),
             "FLD2D RGMG zero finest x failed");

  const int last = ws.rgmg_n_levels - 1;
  for (int l = 0; l < last; ++l) {
    RgmgLevel& fine = ws.rgmg_levels[static_cast<std::size_t>(l)];
    RgmgLevel& coarse = ws.rgmg_levels[static_cast<std::size_t>(l + 1)];
    rgmg_smooth(ws, l, omega, true);
    fld_2d_stencil_residual(fine.diag.as<double>(),
                            fine.ar_left.as<double>(),
                            fine.ar_right.as<double>(),
                            fine.az_m.as<double>(),
                            fine.az_p.as<double>(),
                            fine.b.as<double>(),
                            fine.x.as<double>(),
                            fine.res.as<double>(),
                            fine.nr,
                            fine.nz,
                            fine.n_groups);
    fld_2d_restrict_pairwise_r(fine.res.as<double>(),
                               coarse.b.as<double>(),
                               fine.nr,
                               fine.nz,
                               fine.n_groups);
    cuda_check(cudaMemsetAsync(coarse.x.ptr,
                          0,
                          sizeof(double) *
                              static_cast<std::size_t>(
                                  rgmg_level_n_rows(coarse)),
                          0),
               "FLD2D RGMG zero coarse x failed");
  }

  rgmg_coarsest_exact_solve(ws, last);

  for (int l = last - 1; l >= 0; --l) {
    RgmgLevel& fine = ws.rgmg_levels[static_cast<std::size_t>(l)];
    RgmgLevel& coarse = ws.rgmg_levels[static_cast<std::size_t>(l + 1)];
    fld_2d_prolong_add_pairwise_r(coarse.x.as<double>(),
                                  fine.x.as<double>(),
                                  fine.nr,
                                  fine.nz,
                                  fine.n_groups);
    rgmg_smooth(ws, l, omega, false);
  }
}

void rgmg_vcycle(cudaStream_t stream, Fld2DWorkspace& ws, const double omega) {
  TENRYU_ASSERT(ws.rgmg_n_levels > 0,
                "FLD2D RGMG V-cycle requires a built hierarchy");
  TENRYU_ASSERT(ws.rgmg_n_levels ==
                    static_cast<int>(ws.rgmg_levels.size()),
                "FLD2D RGMG level count mismatch");
  RgmgLevel& finest = ws.rgmg_levels[0];
  cuda_check(cudaMemsetAsync(finest.x.ptr,
                        0,
                        sizeof(double) *
                            static_cast<std::size_t>(rgmg_level_n_rows(finest)),
                        stream),
             "FLD2D RGMG zero finest x failed");

  const int last = ws.rgmg_n_levels - 1;
  for (int l = 0; l < last; ++l) {
    RgmgLevel& fine = ws.rgmg_levels[static_cast<std::size_t>(l)];
    RgmgLevel& coarse = ws.rgmg_levels[static_cast<std::size_t>(l + 1)];
    rgmg_smooth(stream, ws, l, omega, true);
    fld_2d_stencil_residual(stream,
                            fine.diag.as<double>(),
                            fine.ar_left.as<double>(),
                            fine.ar_right.as<double>(),
                            fine.az_m.as<double>(),
                            fine.az_p.as<double>(),
                            fine.b.as<double>(),
                            fine.x.as<double>(),
                            fine.res.as<double>(),
                            fine.nr,
                            fine.nz,
                            fine.n_groups);
    fld_2d_restrict_pairwise_r(stream,
                               fine.res.as<double>(),
                               coarse.b.as<double>(),
                               fine.nr,
                               fine.nz,
                               fine.n_groups);
    cuda_check(cudaMemsetAsync(coarse.x.ptr,
                          0,
                          sizeof(double) *
                              static_cast<std::size_t>(
                                  rgmg_level_n_rows(coarse)),
                          stream),
               "FLD2D RGMG zero coarse x failed");
  }

  rgmg_coarsest_exact_solve(stream, ws, last);

  for (int l = last - 1; l >= 0; --l) {
    RgmgLevel& fine = ws.rgmg_levels[static_cast<std::size_t>(l)];
    RgmgLevel& coarse = ws.rgmg_levels[static_cast<std::size_t>(l + 1)];
    fld_2d_prolong_add_pairwise_r(stream,
                                  coarse.x.as<double>(),
                                  fine.x.as<double>(),
                                  fine.nr,
                                  fine.nz,
                                  fine.n_groups);
    rgmg_smooth(stream, ws, l, omega, false);
  }
}

void rgmg_apply(Fld2DWorkspace& ws,
                const double* r,
                double* z,
                const int n_rows,
                const double omega) {
  RgmgLevel& finest = ws.rgmg_levels[0];
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rows);
  cuda_check(cudaMemcpyAsync(finest.b.as<double>(),
                        r,
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        0),
             "FLD2D RGMG copy rhs failed");
  rgmg_vcycle(ws, omega);
  cuda_check(cudaMemcpyAsync(z,
                        finest.x.as<double>(),
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        0),
             "FLD2D RGMG copy solution failed");
}

void rgmg_apply(cudaStream_t stream,
                Fld2DWorkspace& ws,
                const double* r,
                double* z,
                const int n_rows,
                const double omega) {
  RgmgLevel& finest = ws.rgmg_levels[0];
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rows);
  cuda_check(cudaMemcpyAsync(finest.b.as<double>(),
                        r,
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        stream),
             "FLD2D RGMG copy rhs failed");
  rgmg_vcycle(stream, ws, omega);
  cuda_check(cudaMemcpyAsync(z,
                        finest.x.as<double>(),
                        bytes,
                        cudaMemcpyDeviceToDevice,
                        stream),
             "FLD2D RGMG copy solution failed");
}

// Builds (or reuses) the instantiated graph for one CG inter-check block:
//   tail(k) + [bump, head, tail](k+1..k+3) + [bump, head](k+4)
// where head = SpMV + dot(p,Ap) + update_x_r and tail = precond +
// dot(r,z) + update_p. Entry always happens at k === 0 (mod 4), so the rz
// ping-pong parity pattern is the same for every block and one graph
// serves all replays (and all solves while the key pointers hold).
// Diagonal preconditioner only. Returns false (and latches the eager
// fallback) if capture or instantiation fails.
bool ensure_cg_block_graph(Fld2DWorkspace& ws,
                           double* x,
                           double* r,
                           double* z,
                           double* p,
                           double* Ap,
                           const double* diag_inv,
                           double* d_cg_rz_pingpong,
                           double* d_cg_pAp,
                           CgDeviceStatus* d_cg_status,
                           const int* row_offsets,
                           const int* col_indices,
                           const double* values,
                           const int n_rows,
                           const int nnz,
                           const int grid,
                           const FldPrecondMode precond_mode,
                           const int nz,
                           const int zline_batch_count,
                           const double rgmg_omega) {
  if (ws.cg_graph_capture_failed) {
    return false;
  }
  ws.d_cg_iter.resize(sizeof(int));
  // Pre-size the dot scratch: no allocation is legal during capture, and
  // the captured dot kernels bake the partials pointer, so it must be
  // final (grow-only across ALL solves sharing this workspace) before the
  // key snapshot below.
  if (dot_block_count(n_rows) > kDotSingleBlockMaxBlocks) {
    ensure_dot_partials(ws, dot_block_count(n_rows));
  }
  FldCgGraphKey key;
  key.n_rows = n_rows;
  key.nnz = nnz;
  key.values = values;
  key.row_offsets = row_offsets;
  key.col_indices = col_indices;
  key.x = x;
  key.diag_inv = diag_inv;
  key.r = r;
  key.z = z;
  key.p = p;
  key.Ap = Ap;
  key.pingpong = d_cg_rz_pingpong;
  key.pAp = d_cg_pAp;
  key.status = d_cg_status;
  key.iter_counter = ws.d_cg_iter.as<int>();
  key.spmv_buffer = ws.d_spmv.ptr;
  key.dot_partials = ws.d_dot_partials.as<double>();
  key.precond_mode_id =
      precond_mode == FldPrecondMode::Diagonal
          ? 0
          : (precond_mode == FldPrecondMode::ZLine ? 1 : 2);
  if (precond_mode == FldPrecondMode::ZLine) {
    key.zline_dl = ws.d_line_dl.ptr;
    key.zline_d = ws.d_line_d.ptr;
    key.zline_du = ws.d_line_du.ptr;
    key.gtsv_buffer = ws.d_gtsv2_buffer.ptr;
  } else if (precond_mode == FldPrecondMode::RGmg) {
    key.gtsv_buffer = ws.rgmg_gtsv_buffer.ptr;
    key.precond_ptrs.clear();
    key.precond_ptrs.reserve(
        static_cast<std::size_t>(ws.rgmg_n_levels) * 9U);
    for (int l = 0; l < ws.rgmg_n_levels; ++l) {
      const RgmgLevel& lev = ws.rgmg_levels[static_cast<std::size_t>(l)];
      key.precond_ptrs.push_back(lev.diag.ptr);
      key.precond_ptrs.push_back(lev.ar_left.ptr);
      key.precond_ptrs.push_back(lev.ar_right.ptr);
      key.precond_ptrs.push_back(lev.az_m.ptr);
      key.precond_ptrs.push_back(lev.az_p.ptr);
      key.precond_ptrs.push_back(lev.b.ptr);
      key.precond_ptrs.push_back(lev.x.ptr);
      key.precond_ptrs.push_back(lev.res.ptr);
      key.precond_ptrs.push_back(lev.corr.ptr);
    }
  }
  if (ws.cg_graph_exec != nullptr && key == ws.cg_graph_key) {
    return true;
  }
  if (ws.cg_stream == nullptr &&
      cudaStreamCreate(&ws.cg_stream) != cudaSuccess) {
    ws.cg_stream = nullptr;
    ws.cg_graph_capture_failed = true;
    static_cast<void>(cudaGetLastError());
    core::log_warning(
        "FLD2D CG graph: stream creation failed; using the eager CG loop");
    return false;
  }
  if (ws.cg_graph_exec != nullptr) {
    static_cast<void>(cudaGraphExecDestroy(ws.cg_graph_exec));
    ws.cg_graph_exec = nullptr;
  }
  int* d_iter = ws.d_cg_iter.as<int>();
  const double one = 1.0;
  const double zero = 0.0;
  cusparse_check(cusparseSetStream(ws.handle, ws.cg_stream),
                 "FLD2D CG graph cusparseSetStream failed");
  bool spmv_ok = true;
  cudaGraph_t graph = nullptr;
  cudaError_t capture_err =
      cudaStreamBeginCapture(ws.cg_stream, cudaStreamCaptureModeThreadLocal);
  if (capture_err == cudaSuccess) {
    const auto capture_tail = [&](const int parity) {
      double* d_rz = d_cg_rz_pingpong + parity;
      double* d_rz_next = d_cg_rz_pingpong + (1 - parity);
      if (precond_mode == FldPrecondMode::Diagonal) {
        cg_apply_preconditioner_kernel<<<grid, kBlock, 0, ws.cg_stream>>>(
            r, diag_inv, z, n_rows);
      } else if (precond_mode == FldPrecondMode::ZLine) {
        apply_zline_preconditioner(
            ws.cg_stream, ws, r, z, n_rows, nz, zline_batch_count);
      } else {
        rgmg_apply(ws.cg_stream, ws, r, z, n_rows, rgmg_omega);
      }
      device_dot_to_stream(ws.cg_stream, ws, r, z, n_rows, d_rz_next);
      cg_update_p_kernel<<<grid, kBlock, 0, ws.cg_stream>>>(
          p, z, d_rz_next, d_rz, d_cg_status, n_rows);
    };
    const auto capture_head = [&](const int parity) {
      cg_bump_iter_kernel<<<1, 1, 0, ws.cg_stream>>>(d_iter);
      if (cusparseSpMV(ws.handle,
                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                       &one,
                       ws.mat,
                       ws.vec_p,
                       &zero,
                       ws.vec_Ap,
                       CUDA_R_64F,
                       CUSPARSE_SPMV_ALG_DEFAULT,
                       ws.d_spmv.ptr) != CUSPARSE_STATUS_SUCCESS) {
        spmv_ok = false;
      }
      device_dot_to_stream(ws.cg_stream, ws, p, Ap, n_rows, d_cg_pAp);
      cg_update_x_r_from_dev_kernel<<<grid, kBlock, 0, ws.cg_stream>>>(
          x, r, p, Ap, d_cg_rz_pingpong + parity, d_cg_pAp, d_cg_status,
          d_iter, n_rows);
    };
    // Block-entry iteration k is always even (k = 4, 8, ...).
    capture_tail(0);
    capture_head(1);
    capture_tail(1);
    capture_head(0);
    capture_tail(0);
    capture_head(1);
    capture_tail(1);
    capture_head(0);
    capture_err = cudaStreamEndCapture(ws.cg_stream, &graph);
  }
  cusparse_check(cusparseSetStream(ws.handle, nullptr),
                 "FLD2D CG graph cusparseSetStream restore failed");
  bool ok = capture_err == cudaSuccess && graph != nullptr && spmv_ok;
  if (ok) {
    ok = cudaGraphInstantiate(&ws.cg_graph_exec, graph, 0) == cudaSuccess &&
         ws.cg_graph_exec != nullptr;
  }
  if (graph != nullptr) {
    static_cast<void>(cudaGraphDestroy(graph));
  }
  if (!ok) {
    if (ws.cg_graph_exec != nullptr) {
      static_cast<void>(cudaGraphExecDestroy(ws.cg_graph_exec));
      ws.cg_graph_exec = nullptr;
    }
    ws.cg_graph_key = FldCgGraphKey{};
    ws.cg_graph_capture_failed = true;
    static_cast<void>(cudaGetLastError());
    core::log_warning(
        "FLD2D CG graph capture failed; using the eager CG loop");
    return false;
  }
  ws.cg_graph_key = key;
  return true;
}

int solve_cusparse_cg_jacobi(const int* row_offsets,
                             const int* col_indices,
                             const double* values,
                             const double* rhs,
                             const double* diag_inv,
                             double* x,
                             const int n_rows,
                             const int nnz,
                             const double tolerance,
                             const bool tol_norm_rhs,
                             const int cg_max_iter,
                             const bool verbose,
                             CgDiagnostics* diagnostics,
                             const FldPrecondMode precond_mode =
                                 FldPrecondMode::Diagonal,
                             const int nr = 0,
                             const int nz = 0,
                             const int n_groups = 0,
                             const double rgmg_omega = 0.67) {
  if (n_rows <= 0) {
    if (verbose && diagnostics != nullptr) {
      diagnostics->iters_executed = 0;
    }
    return 0;
  }
  auto& ws = fld_2d_workspace();
  if (ws.handle == nullptr) {
    cusparse_check(cusparseCreate(&ws.handle), "FLD2D cusparseCreate failed");
  }
  // Option-C distributed CG (design doc mpi_m18c_fld2d_cg_spec.md): the
  // SpMV stays full-size (non-owned rows hold finite garbage assembled from
  // stale-but-finite inputs and are never read through masked dots), the p
  // search direction gets a per-iteration ghost-strip refresh so owned-row
  // SpMV reads valid neighbor columns, and EVERY inner product becomes an
  // owned-masked dot + Allreduce so alpha/beta/convergence/breakdown are
  // rank-uniform. Single-rank runs keep the serial paths byte-identical.
  const bool mpi_active = ws.mpi_n_ranks > 1;
  TENRYU_ASSERT(!mpi_active ||
                    (ws.mpi_part != nullptr && ws.mpi_bufs != nullptr &&
                     ws.mpi_n_cells > 0 && ws.mpi_n_groups > 0 &&
                     n_rows == ws.mpi_n_cells * ws.mpi_n_groups &&
                     ws.mpi_c_begin >= 0 && ws.mpi_c_end > ws.mpi_c_begin &&
                     ws.mpi_c_end <= ws.mpi_n_cells),
                "FLD2D distributed CG: workspace MPI context mismatch "
                "(silent serial fallback under MPI would desynchronize "
                "ranks)");
  FldPrecondMode effective_precond = precond_mode;
  if (mpi_active && precond_mode == FldPrecondMode::RGmg) {
    // RGmg semicoarsens in r across the slab interfaces (rank-local
    // hierarchies would mix garbage rows into owned corrections); fall back
    // to the z-line preconditioner under MPI (v1.1 item, redesign doc §2).
    static bool warned_rgmg_mpi = false;
    if (!warned_rgmg_mpi) {
      core::log_warning(
          "FLD2D RGMG preconditioner is not rank-decomposed; falling back "
          "to the z-line preconditioner under MPI");
      warned_rgmg_mpi = true;
    }
    effective_precond = FldPrecondMode::ZLine;
  }
  const std::size_t bytes = sizeof(double) * static_cast<std::size_t>(n_rows);
  ws.d_r.resize(bytes);
  ws.d_z.resize(bytes);
  ws.d_p.resize(bytes);
  ws.d_Ap.resize(bytes);
  ws.d_cg_rz_pingpong.resize(2 * sizeof(double));
  ws.d_cg_pAp.resize(sizeof(double));
  ws.d_cg_status.resize(sizeof(CgDeviceStatus));
  const int cg_dot_blocks = dot_block_count(n_rows);
  if (cg_dot_blocks > kDotSingleBlockMaxBlocks) {
    ensure_dot_partials(ws, cg_dot_blocks);
  }
  double* r = ws.d_r.as<double>();
  double* z = ws.d_z.as<double>();
  double* p = ws.d_p.as<double>();
  double* Ap = ws.d_Ap.as<double>();
  double* d_cg_rz_pingpong = ws.d_cg_rz_pingpong.as<double>();
  double* d_cg_pAp = ws.d_cg_pAp.as<double>();
  CgDeviceStatus* d_cg_status = ws.d_cg_status.as<CgDeviceStatus>();
  const bool use_zline = effective_precond == FldPrecondMode::ZLine;
  const bool use_rgmg = effective_precond == FldPrecondMode::RGmg;
  const int zline_batch_count = use_zline ? n_groups * nr : 0;
  if (use_zline) {
    prepare_zline_preconditioner(
        ws, col_indices, values, z, n_rows, nr, nz, n_groups);
  } else if (use_rgmg) {
    build_rgmg_hierarchy(
        ws, row_offsets, col_indices, values, nr, nz, n_groups);
  }

  const bool descriptors_changed =
      ws.mat == nullptr || ws.vec_x == nullptr || ws.vec_p == nullptr ||
      ws.vec_Ap == nullptr || ws.desc_n_rows != n_rows ||
      ws.desc_nnz != nnz || ws.desc_row_offsets_ptr != row_offsets ||
      ws.desc_col_indices_ptr != col_indices || ws.desc_values_ptr != values ||
      ws.desc_x_ptr != x || ws.desc_p_ptr != p || ws.desc_Ap_ptr != Ap;
  if (descriptors_changed) {
    cusparseSpMatDescr_t new_mat = nullptr;
    cusparseDnVecDescr_t new_vec_x = nullptr;
    cusparseDnVecDescr_t new_vec_p = nullptr;
    cusparseDnVecDescr_t new_vec_Ap = nullptr;
    const auto destroy_new_descriptors = [&]() {
      if (new_vec_Ap != nullptr) {
        static_cast<void>(cusparseDestroyDnVec(new_vec_Ap));
        new_vec_Ap = nullptr;
      }
      if (new_vec_p != nullptr) {
        static_cast<void>(cusparseDestroyDnVec(new_vec_p));
        new_vec_p = nullptr;
      }
      if (new_vec_x != nullptr) {
        static_cast<void>(cusparseDestroyDnVec(new_vec_x));
        new_vec_x = nullptr;
      }
      if (new_mat != nullptr) {
        static_cast<void>(cusparseDestroySpMat(new_mat));
        new_mat = nullptr;
      }
    };
    cusparseStatus_t status =
        cusparseCreateCsr(&new_mat,
                          n_rows,
                          n_rows,
                          nnz,
                          const_cast<int*>(row_offsets),
                          const_cast<int*>(col_indices),
                          const_cast<double*>(values),
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_32I,
                          CUSPARSE_INDEX_BASE_ZERO,
                          CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      destroy_new_descriptors();
      cusparse_check(status, "FLD2D create CSR failed");
    }
    status = cusparseCreateDnVec(&new_vec_x, n_rows, x, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      destroy_new_descriptors();
      cusparse_check(status, "FLD2D create x vector failed");
    }
    status = cusparseCreateDnVec(&new_vec_p, n_rows, p, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      destroy_new_descriptors();
      cusparse_check(status, "FLD2D create p vector failed");
    }
    status = cusparseCreateDnVec(&new_vec_Ap, n_rows, Ap, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      destroy_new_descriptors();
      cusparse_check(status, "FLD2D create Ap vector failed");
    }
    if (ws.vec_Ap != nullptr) {
      static_cast<void>(cusparseDestroyDnVec(ws.vec_Ap));
    }
    if (ws.vec_p != nullptr) {
      static_cast<void>(cusparseDestroyDnVec(ws.vec_p));
    }
    if (ws.vec_x != nullptr) {
      static_cast<void>(cusparseDestroyDnVec(ws.vec_x));
    }
    if (ws.mat != nullptr) {
      static_cast<void>(cusparseDestroySpMat(ws.mat));
    }
    ws.mat = new_mat;
    ws.vec_x = new_vec_x;
    ws.vec_p = new_vec_p;
    ws.vec_Ap = new_vec_Ap;
    ws.desc_n_rows = n_rows;
    ws.desc_nnz = nnz;
    ws.desc_row_offsets_ptr = row_offsets;
    ws.desc_col_indices_ptr = col_indices;
    ws.desc_values_ptr = values;
    ws.desc_x_ptr = x;
    ws.desc_p_ptr = p;
    ws.desc_Ap_ptr = Ap;
  }
  // §6q.9 stage 2: per-group owned-row CSR views (row-independent CSR SpMV
  // over a row subrange is bit-identical to the same rows of the full
  // product, so tiers are untouched; columns stay global-width and read the
  // full-length p/x whose ghost strips are refreshed per iteration).
  if (mpi_active) {
    const bool owned_views_stale =
        descriptors_changed ||
        ws.mat_owned.size() != static_cast<std::size_t>(n_groups) ||
        ws.owned_view_c_begin != ws.mpi_c_begin ||
        ws.owned_view_c_end != ws.mpi_c_end;
    if (owned_views_stale) {
      for (auto& m : ws.mat_owned) {
        if (m != nullptr) {
          static_cast<void>(cusparseDestroySpMat(m));
        }
      }
      for (auto& v : ws.vec_Ap_owned) {
        if (v != nullptr) {
          static_cast<void>(cusparseDestroyDnVec(v));
        }
      }
      ws.mat_owned.assign(static_cast<std::size_t>(n_groups), nullptr);
      ws.vec_Ap_owned.assign(static_cast<std::size_t>(n_groups), nullptr);
      const int owned_cells = ws.mpi_c_end - ws.mpi_c_begin;
      ws.d_owned_row_offsets.resize(sizeof(int) *
                                    static_cast<std::size_t>(n_groups) *
                                    static_cast<std::size_t>(owned_cells + 1));
      int* owned_offsets_base = ws.d_owned_row_offsets.as<int>();
      for (int gsp = 0; gsp < n_groups; ++gsp) {
        const int row_lo = gsp * ws.mpi_n_cells + ws.mpi_c_begin;
        int* dst = owned_offsets_base +
                   static_cast<std::size_t>(gsp) *
                       static_cast<std::size_t>(owned_cells + 1);
        const int shift_grid = (owned_cells + 1 + kBlock - 1) / kBlock;
        owned_offsets_shift_kernel<<<shift_grid, kBlock>>>(
            row_offsets, dst, row_lo, owned_cells);
        cuda_check(cudaGetLastError(),
                   "FLD2D owned offsets shift launch failed");
        int base_and_end[2] = {0, 0};
        cuda_check(cudaMemcpy(&base_and_end[0], row_offsets + row_lo,
                              sizeof(int), cudaMemcpyDeviceToHost),
                   "FLD2D owned offsets base D2H failed");
        cuda_check(cudaMemcpy(&base_and_end[1],
                              row_offsets + row_lo + owned_cells, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "FLD2D owned offsets end D2H failed");
        const int nnz_g = base_and_end[1] - base_and_end[0];
        cusparse_check(
            cusparseCreateCsr(&ws.mat_owned[static_cast<std::size_t>(gsp)],
                              owned_cells, n_rows, nnz_g, dst,
                              const_cast<int*>(col_indices) + base_and_end[0],
                              const_cast<double*>(values) + base_and_end[0],
                              CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F),
            "FLD2D create owned CSR view failed");
        cusparse_check(
            cusparseCreateDnVec(&ws.vec_Ap_owned[static_cast<std::size_t>(gsp)],
                                owned_cells, Ap + row_lo, CUDA_R_64F),
            "FLD2D create owned Ap vector failed");
      }
      ws.owned_view_c_begin = ws.mpi_c_begin;
      ws.owned_view_c_end = ws.mpi_c_end;
    }
  }
  const double one = 1.0;
  const double zero = 0.0;
  std::size_t spmv_buffer_size = 0U;
  cusparse_check(cusparseSpMV_bufferSize(ws.handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one,
                                         ws.mat,
                                         ws.vec_x,
                                         &zero,
                                         ws.vec_Ap,
                                         CUDA_R_64F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         &spmv_buffer_size),
                "FLD2D SpMV buffer size failed");
  if (mpi_active) {
    for (int gsp = 0; gsp < n_groups; ++gsp) {
      std::size_t owned_buf = 0U;
      cusparse_check(
          cusparseSpMV_bufferSize(
              ws.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
              ws.mat_owned[static_cast<std::size_t>(gsp)], ws.vec_p, &zero,
              ws.vec_Ap_owned[static_cast<std::size_t>(gsp)], CUDA_R_64F,
              CUSPARSE_SPMV_ALG_DEFAULT, &owned_buf),
          "FLD2D owned SpMV buffer size failed");
      spmv_buffer_size = std::max(spmv_buffer_size, owned_buf);
    }
  }
  if (ws.d_spmv.size != spmv_buffer_size) {
    ws.d_spmv.resize(spmv_buffer_size);
  }
  if (mpi_active) {
    for (int gsp = 0; gsp < n_groups; ++gsp) {
      cusparse_check(
          cusparseSpMV(ws.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                       ws.mat_owned[static_cast<std::size_t>(gsp)], ws.vec_x,
                       &zero,
                       ws.vec_Ap_owned[static_cast<std::size_t>(gsp)],
                       CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, ws.d_spmv.ptr),
          "FLD2D initial owned SpMV failed");
    }
  } else {
    cusparse_check(cusparseSpMV(ws.handle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &one,
                                ws.mat,
                                ws.vec_x,
                                &zero,
                                ws.vec_Ap,
                                CUDA_R_64F,
                                CUSPARSE_SPMV_ALG_DEFAULT,
                                ws.d_spmv.ptr),
                  "FLD2D initial SpMV failed");
  }

  const int grid = (n_rows + kBlock - 1) / kBlock;
  // §6q.8 owned-row windowing (M20): every per-iteration vector op and the
  // z-line preconditioner run over the OWNED spans only. The solver row
  // ordering is GROUP-MAJOR (row = g*n_cells + c — see
  // dot_owned_partials_kernel), so the owned rows form G contiguous spans
  // [g*n_cells + c_begin, g*n_cells + c_end); we launch one windowed call
  // per group. The SpMV stays library-full-size (stage 2); non-owned rows
  // keep stale-but-finite garbage exactly as before — dots are owned-masked
  // and the p ghost strips are refreshed per iteration, so P>1 results are
  // expected bit-identical to the full-span launches. Serial: one span ==
  // the full vector.
  const int cg_span_cells =
      mpi_active ? (ws.mpi_c_end - ws.mpi_c_begin) : n_rows;
  const int cg_span_groups = mpi_active ? n_groups : 1;
  const int cg_span_rows =
      mpi_active ? cg_span_cells : n_rows;
  const int ogrid = (cg_span_rows + kBlock - 1) / kBlock;
  const auto cg_span_lo = [&](const int gsp) {
    return mpi_active ? gsp * ws.mpi_n_cells + ws.mpi_c_begin : 0;
  };
  const int zline_batch_owned =
      mpi_active ? cg_span_cells / nz : zline_batch_count;
  if (effective_precond == FldPrecondMode::Diagonal) {
    for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
      const int lo = cg_span_lo(gsp);
      compute_initial_residual_kernel<<<ogrid, kBlock>>>(
          rhs + lo, Ap + lo, diag_inv + lo, r + lo, z + lo, p + lo,
          cg_span_rows);
    }
    cuda_check(cudaGetLastError(), "FLD2D initial residual launch failed");
  } else {
    for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
      const int lo = cg_span_lo(gsp);
      compute_initial_residual_only_kernel<<<ogrid, kBlock>>>(
          rhs + lo, Ap + lo, r + lo, cg_span_rows);
    }
    cuda_check(cudaGetLastError(), "FLD2D initial residual-only launch failed");
    if (use_zline) {
      for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
        apply_zline_preconditioner(ws, r, z, cg_span_rows, nz,
                                   zline_batch_owned, cg_span_lo(gsp));
      }
    } else {
      rgmg_apply(ws, r, z, n_rows, rgmg_omega);
    }
    cuda_check(cudaMemcpy(p, z, bytes, cudaMemcpyDeviceToDevice),
               "FLD2D initialize z-line search direction failed");
  }
  const double r0 = std::sqrt(
      std::max(mpi_active ? device_dot_owned_global(ws, r, r, nullptr)
                          : device_dot(r, r, n_rows, ws.d_scalar),
               0.0));
  double tol_denom = r0;
  if (tol_norm_rhs) {
    const double b_norm = std::sqrt(
        std::max(mpi_active ? device_dot_owned_global(ws, rhs, rhs, nullptr)
                            : device_dot(rhs, rhs, n_rows, ws.d_scalar),
                 0.0));
    tol_denom = (b_norm > 0.0) ? b_norm : 1.0;
    if (r0 <= tolerance * tol_denom || !(r0 > 0.0)) {
      if (verbose && diagnostics != nullptr) {
        diagnostics->iters_executed = 0;
      }
      return 0;
    }
  } else {
    if (r0 <= tolerance || !(r0 > 0.0)) {
      if (verbose && diagnostics != nullptr) {
        diagnostics->iters_executed = 0;
      }
      return 0;
    }
  }
  const CgDeviceStatus initial_cg_status{
      std::numeric_limits<double>::infinity(), -1, 0, 0};
  cuda_check(cudaMemcpy(d_cg_status,
                        &initial_cg_status,
                        sizeof(initial_cg_status),
                        cudaMemcpyHostToDevice),
             "FLD2D init CG device status failed");
  if (mpi_active) {
    device_dot_owned_global(ws, r, z, d_cg_rz_pingpong);
  } else {
    device_dot_to(r, z, n_rows, d_cg_rz_pingpong);
  }
  // Per-iteration ghost refresh of the search direction: owned-row SpMV
  // reads p at the neighbor-owned ghost strips, which the tail's update_p
  // corrupts (z is garbage there). Group-major vector => one cell-field
  // exchange per group block.
  std::vector<double*> mpi_p_group_ptrs;
  if (mpi_active) {
    mpi_p_group_ptrs.resize(static_cast<std::size_t>(ws.mpi_n_groups));
    for (int g = 0; g < ws.mpi_n_groups; ++g) {
      mpi_p_group_ptrs[static_cast<std::size_t>(g)] =
          p + static_cast<std::size_t>(g) *
                  static_cast<std::size_t>(ws.mpi_n_cells);
    }
  }
  const auto exchange_p_ghosts = [&]() {
    if (!mpi_active) {
      return;
    }
    parallel::exchange_cell_fields(*ws.mpi_part, *ws.mpi_bufs,
                                   mpi_p_group_ptrs.data(), ws.mpi_n_groups,
                                   ws.mpi_n_cells, nullptr, 6);
  };
  const auto copy_cg_status = [&]() {
    CgDeviceStatus host_status{};
    cuda_check(cudaMemcpy(&host_status,
                          d_cg_status,
                          sizeof(host_status),
                          cudaMemcpyDeviceToHost),
               "FLD2D copy CG device status failed");
    return host_status;
  };
  const auto copy_breakdown_iter = [&]() {
    int breakdown_iter = -1;
    const char* status_bytes = reinterpret_cast<const char*>(d_cg_status);
    cuda_check(cudaMemcpy(&breakdown_iter,
                          status_bytes +
                              offsetof(CgDeviceStatus, breakdown_iter),
                          sizeof(breakdown_iter),
                          cudaMemcpyDeviceToHost),
               "FLD2D copy CG breakdown status failed");
    return breakdown_iter;
  };
  const auto finalize_cg_diagnostics = [&](const int iters_executed) {
    if (verbose && diagnostics != nullptr) {
      const CgDeviceStatus host_status = copy_cg_status();
      diagnostics->min_pAp_value = host_status.min_pAp;
      diagnostics->count_nonpos_pAp = host_status.count_nonpos_pAp;
      diagnostics->count_nonfinite_alpha_beta_rz = host_status.count_nonfinite;
      diagnostics->iters_executed = iters_executed;
    }
  };
  constexpr int kCheckEveryIterThreshold = 4;
  const int max_iter = std::max(50, std::min(cg_max_iter, n_rows));
  // W1 (2026-07-09): between residual checks the loop body is a fixed
  // kernel sequence over device-resident scalars; graph replay removes
  // ~9 host launch ops per iteration. The eager lambdas below reproduce
  // the pre-W1 loop body verbatim; graph mode replays the identical
  // kernel sequence (bit-identity gated).
  const auto run_head = [&](const int iter) {
    exchange_p_ghosts();
    if (mpi_active) {
      for (int gsp = 0; gsp < n_groups; ++gsp) {
        cusparse_check(
            cusparseSpMV(ws.handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                         ws.mat_owned[static_cast<std::size_t>(gsp)],
                         ws.vec_p, &zero,
                         ws.vec_Ap_owned[static_cast<std::size_t>(gsp)],
                         CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT,
                         ws.d_spmv.ptr),
            "FLD2D CG owned SpMV failed");
      }
    } else {
      cusparse_check(cusparseSpMV(ws.handle,
                                  CUSPARSE_OPERATION_NON_TRANSPOSE,
                                  &one,
                                  ws.mat,
                                  ws.vec_p,
                                  &zero,
                                  ws.vec_Ap,
                                  CUDA_R_64F,
                                  CUSPARSE_SPMV_ALG_DEFAULT,
                                  ws.d_spmv.ptr),
                    "FLD2D CG SpMV failed");
    }
    if (mpi_active) {
      device_dot_owned_global(ws, p, Ap, d_cg_pAp);
    } else {
      device_dot_to(p, Ap, n_rows, d_cg_pAp);
    }
    double* d_rz = d_cg_rz_pingpong + (iter & 1);
    for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
      const int lo = cg_span_lo(gsp);
      cg_update_x_r_kernel<<<ogrid, kBlock>>>(
          x + lo, r + lo, p + lo, Ap + lo, d_rz, d_cg_pAp, d_cg_status, iter,
          cg_span_rows);
    }
    cuda_check(cudaGetLastError(), "FLD2D CG update x/r launch failed");
  };
  const auto run_tail = [&](const int iter) {
    double* d_rz = d_cg_rz_pingpong + (iter & 1);
    double* d_rz_next = d_cg_rz_pingpong + ((iter + 1) & 1);
    if (effective_precond == FldPrecondMode::Diagonal) {
      for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
        const int lo = cg_span_lo(gsp);
        cg_apply_preconditioner_kernel<<<ogrid, kBlock>>>(
            r + lo, diag_inv + lo, z + lo, cg_span_rows);
      }
      cuda_check(cudaGetLastError(), "FLD2D CG preconditioner launch failed");
    } else {
      if (use_zline) {
        for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
          apply_zline_preconditioner(ws, r, z, cg_span_rows, nz,
                                     zline_batch_owned, cg_span_lo(gsp));
        }
      } else {
        rgmg_apply(ws, r, z, n_rows, rgmg_omega);
      }
    }
    if (mpi_active) {
      device_dot_owned_global(ws, r, z, d_rz_next);
    } else {
      device_dot_to(r, z, n_rows, d_rz_next);
    }
    for (int gsp = 0; gsp < cg_span_groups; ++gsp) {
      const int lo = cg_span_lo(gsp);
      cg_update_p_kernel<<<ogrid, kBlock>>>(
          p + lo, z + lo, d_rz_next, d_rz, d_cg_status, cg_span_rows);
    }
    cuda_check(cudaGetLastError(), "FLD2D CG update p launch failed");
  };
  // Residual-norm convergence check costs a D2H sync; check early iters and
  // then every 4 iters (plus the last iter) to reduce GPU pipeline stalls.
  const auto check_convergence = [&](const int iter, bool& returned) -> int {
    const int breakdown_iter = copy_breakdown_iter();
    if (breakdown_iter >= 0) {
      finalize_cg_diagnostics(breakdown_iter + 1);
      returned = true;
      return breakdown_iter + 1;
    }
    const double r_norm = std::sqrt(
        std::max(mpi_active ? device_dot_owned_global(ws, r, r, nullptr)
                            : device_dot(r, r, n_rows, ws.d_scalar),
                 0.0));
    const double rel = r_norm / tol_denom;
    if (verbose && diagnostics != nullptr && std::isfinite(rel)) {
      if (iter + 1 >= kCheckEveryIterThreshold &&
          diagnostics->recurrent_resid_first_check == 0.0) {
        diagnostics->recurrent_resid_first_check = rel;
      }
      diagnostics->recurrent_resid_last_check = rel;
    }
    if (rel <= tolerance) {
      finalize_cg_diagnostics(iter + 1);
      if (std::getenv("TENRYU_FLD_CG_TELEMETRY") != nullptr) {
        // §16.3 companion telemetry (recorded only, never gated): per-solve
        // converged iteration count for iteration-normalized scaling
        // analysis (§6q.12) and the reproducible-reduction audit lane.
        std::fprintf(stderr, "[cg-telemetry] iters=%d rel=%.3e\n", iter + 1,
                     rel);
      }
      returned = true;
      return iter + 1;
    }
    returned = false;
    return 0;
  };
  const auto finalize_cap_exit = [&]() -> int {
    finalize_cg_diagnostics(max_iter);
    if (diagnostics != nullptr) {
      diagnostics->iters_executed = max_iter;
      int n_cells_diag = 0;
      if (nr > 0 && nz > 0) {
        n_cells_diag = nr * nz;
      } else if (n_groups > 0 && (n_rows % n_groups) == 0) {
        n_cells_diag = n_rows / n_groups;
      }
      const PostPublishDiagnostics true_diagnostics =
          compute_post_publish_solver_diagnostics(rhs,
                                                  x,
                                                  n_rows,
                                                  n_cells_diag,
                                                  n_groups);
      const double denom = std::max(tol_denom, 1.0e-300);
      diagnostics->cap_exit_true_resid_l2_abs =
          true_diagnostics.true_residual_l2_abs;
      diagnostics->cap_exit_true_resid_rel =
          true_diagnostics.true_residual_l2_abs / denom;
      diagnostics->cap_exit_true_resid_group =
          true_diagnostics.true_residual_max_group;
      diagnostics->cap_exit_unconverged =
          !(diagnostics->cap_exit_true_resid_rel <= tolerance);
    }
    return max_iter;
  };
  // CUDA-graph replay is incompatible with the per-iteration MPI exchange
  // and host Allreduce round-trips; force the eager loop under MPI.
  const bool graph_enabled = !fld_cg_graph_disabled() && !mpi_active;
  for (int iter = 0; iter < max_iter; ++iter) {
    run_head(iter);
    const bool check_residual_now =
        (iter < kCheckEveryIterThreshold) || ((iter & 3) == 0) ||
        (iter == max_iter - 1);
    if (check_residual_now) {
      bool returned = false;
      const int rc = check_convergence(iter, returned);
      if (returned) {
        return rc;
      }
      if (graph_enabled && iter >= kCheckEveryIterThreshold &&
          (iter & 3) == 0 && iter + 4 <= max_iter - 1 &&
          ensure_cg_block_graph(ws,
                                x,
                                r,
                                z,
                                p,
                                Ap,
                                diag_inv,
                                d_cg_rz_pingpong,
                                d_cg_pAp,
                                d_cg_status,
                                row_offsets,
                                col_indices,
                                values,
                                n_rows,
                                nnz,
                                grid,
                                effective_precond,
                                nz,
                                zline_batch_count,
                                rgmg_omega)) {
        // Graph mode. Each replay executes tail(k) + iters k+1..k+3 +
        // head(k+4) — exactly the eager kernel sequence between the check
        // at k and the check at k+4.
        int k = iter;
        cuda_check(cudaMemcpy(ws.d_cg_iter.ptr,
                              &k,
                              sizeof(int),
                              cudaMemcpyHostToDevice),
                   "FLD2D CG graph iter init failed");
        while (k + 4 <= max_iter - 1) {
          cuda_check(cudaGraphLaunch(ws.cg_graph_exec, ws.cg_stream),
                     "FLD2D CG graph launch failed");
          k += 4;
          bool returned_b = false;
          const int rc_b = check_convergence(k, returned_b);
          if (returned_b) {
            return rc_b;
          }
        }
        // Eager remainder: the pending tail of iter k, then at most three
        // full iterations whose only remaining check is the forced one at
        // max_iter - 1 (none of k+1..k+3 satisfies (iter & 3) == 0).
        run_tail(k);
        for (int iter_c = k + 1; iter_c < max_iter; ++iter_c) {
          run_head(iter_c);
          if (iter_c == max_iter - 1) {
            bool returned_c = false;
            const int rc_c = check_convergence(iter_c, returned_c);
            if (returned_c) {
              return rc_c;
            }
          }
          run_tail(iter_c);
        }
        return finalize_cap_exit();
      }
    }
    run_tail(iter);
  }
  return finalize_cap_exit();
}

PostPublishDiagnostics compute_post_publish_solver_diagnostics(const double* rhs,
                                                               const double* x_pub,
                                                               const int n_rows,
                                                               const int n_cells,
                                                               const int n_groups) {
  PostPublishDiagnostics diagnostics;
  if (n_rows <= 0) {
    return diagnostics;
  }
  auto& ws = fld_2d_workspace();
  TENRYU_ASSERT(ws.handle != nullptr && ws.mat != nullptr && ws.vec_x != nullptr &&
                    ws.vec_Ap != nullptr && ws.desc_x_ptr == x_pub,
                "FLD2D post-publish diagnostics require active cuSPARSE descriptors");
  const double one = 1.0;
  const double zero = 0.0;
  std::size_t spmv_buffer_size = 0U;
  cusparse_check(cusparseSpMV_bufferSize(ws.handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one,
                                         ws.mat,
                                         ws.vec_x,
                                         &zero,
                                         ws.vec_Ap,
                                         CUDA_R_64F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         &spmv_buffer_size),
                "FLD2D diagnostic SpMV buffer size failed");
  if (ws.d_spmv.size != spmv_buffer_size) {
    ws.d_spmv.resize(spmv_buffer_size);
  }
  cusparse_check(cusparseSpMV(ws.handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &one,
                              ws.mat,
                              ws.vec_x,
                              &zero,
                              ws.vec_Ap,
                              CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT,
                              ws.d_spmv.ptr),
                "FLD2D diagnostic SpMV failed");

  // Option C: mask the residual reductions to owned rows and Allreduce so
  // the cap-exit verdict (and its policy="fail" abort) is rank-uniform.
  const bool mpi_active = ws.mpi_n_ranks > 1;
  TENRYU_ASSERT(!mpi_active ||
                    (ws.mpi_n_cells > 0 && ws.mpi_n_groups > 0 &&
                     n_rows == ws.mpi_n_cells * ws.mpi_n_groups),
                "FLD2D post-publish diagnostics: MPI context mismatch");
  const int pp_n_cells = mpi_active ? ws.mpi_n_cells : n_rows;
  const int pp_c0 = mpi_active ? ws.mpi_c_begin : 0;
  const int pp_owned = mpi_active ? (ws.mpi_c_end - ws.mpi_c_begin) : n_rows;
  const int pp_total = mpi_active ? pp_owned * ws.mpi_n_groups : n_rows;
  ws.d_diag_scalars.resize(2U * sizeof(double));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, 2U * sizeof(double)),
             "FLD2D zero diagnostic scalars failed");
  const int grid = (pp_total + kBlock - 1) / kBlock;
  post_publish_residual_kernel<<<grid, kBlock>>>(
      rhs, ws.d_Ap.as<double>(), ws.d_diag_scalars.as<double>(),
      pp_n_cells, pp_c0, pp_owned, pp_total);
  cuda_check(cudaGetLastError(), "FLD2D diagnostic residual launch failed");

  double reduced[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(reduced,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(reduced),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy diagnostic scalars failed");
  const double r_l2 = std::sqrt(std::max(
      mpi_active
          ? device_dot_owned_global(ws, ws.d_Ap.as<double>(),
                                    ws.d_Ap.as<double>(), nullptr)
          : device_dot(ws.d_Ap.as<double>(), ws.d_Ap.as<double>(), n_rows,
                       ws.d_scalar),
      0.0));
  const double rhs_l2 = std::sqrt(std::max(
      mpi_active ? device_dot_owned_global(ws, rhs, rhs, nullptr)
                 : device_dot(rhs, rhs, n_rows, ws.d_scalar),
      0.0));
  if (mpi_active) {
    const parallel::Reduction reduction(ws.mpi_n_ranks);
    reduced[0] = reduction.allreduce_sum(finite_or_zero(reduced[0]));
    reduced[1] = reduction.allreduce_max(finite_or_zero(reduced[1]));
  }
  diagnostics.true_residual_l2_rel = r_l2 / std::max(rhs_l2, 1.0e-300);
  diagnostics.true_residual_l2_abs = r_l2;
  diagnostics.true_residual_max = fmax(finite_or_zero(reduced[1]), 0.0);
  diagnostics.E_solver_abs = fabs(finite_or_zero(reduced[0]));
  if (n_cells > 0 && n_groups > 0 && n_rows == n_cells * n_groups) {
    RogueRecord initial_record{};
    initial_record.value = 0.0;
    initial_record.cell_idx = -1;
    initial_record.group_idx = -1;
    ws.d_diag_record.resize(sizeof(RogueRecord));
    cuda_check(cudaMemcpy(ws.d_diag_record.ptr,
                          &initial_record,
                          sizeof(initial_record),
                          cudaMemcpyHostToDevice),
               "FLD2D initialize true-residual record failed");
    max_group_major_residual_record_kernel<<<grid, kBlock>>>(
        ws.d_Ap.as<double>(),
        ws.d_diag_record.as<RogueRecord>(),
        n_cells,
        n_groups,
        mpi_active ? pp_c0 : 0,
        mpi_active ? pp_owned : n_cells);
    cuda_check(cudaGetLastError(),
               "FLD2D true-residual max record launch failed");
    RogueRecord record{};
    cuda_check(cudaMemcpy(&record,
                          ws.d_diag_record.ptr,
                          sizeof(record),
                          cudaMemcpyDeviceToHost),
               "FLD2D copy true-residual max record failed");
    diagnostics.true_residual_max_group = record.group_idx;
  }
  return diagnostics;
}

PostPublishDiagnostics compute_post_publish_solver_diagnostics_from_csr(
    const int* row_offsets,
    const int* col_indices,
    const double* values,
    const double* rhs,
    double* x_pub,
    const int n_rows,
    const int nnz) {
  PostPublishDiagnostics diagnostics;
  if (n_rows <= 0) {
    return diagnostics;
  }
  auto& ws = fld_2d_workspace();
  if (ws.handle == nullptr) {
    cusparse_check(cusparseCreate(&ws.handle), "FLD2D cusparseCreate failed");
  }
  ws.d_Ap.resize(sizeof(double) * static_cast<std::size_t>(n_rows));

  cusparseSpMatDescr_t mat = nullptr;
  cusparseDnVecDescr_t vec_x = nullptr;
  cusparseDnVecDescr_t vec_Ap = nullptr;
  cusparse_check(cusparseCreateCsr(&mat,
                                   n_rows,
                                   n_rows,
                                   nnz,
                                   const_cast<int*>(row_offsets),
                                   const_cast<int*>(col_indices),
                                   const_cast<double*>(values),
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_BASE_ZERO,
                                   CUDA_R_64F),
                "FLD2D audit create CSR failed");
  cusparse_check(cusparseCreateDnVec(&vec_x, n_rows, x_pub, CUDA_R_64F),
                "FLD2D audit create x vector failed");
  cusparse_check(cusparseCreateDnVec(&vec_Ap,
                                     n_rows,
                                     ws.d_Ap.as<double>(),
                                     CUDA_R_64F),
                "FLD2D audit create Ap vector failed");

  const double one = 1.0;
  const double zero = 0.0;
  std::size_t spmv_buffer_size = 0U;
  cusparse_check(cusparseSpMV_bufferSize(ws.handle,
                                         CUSPARSE_OPERATION_NON_TRANSPOSE,
                                         &one,
                                         mat,
                                         vec_x,
                                         &zero,
                                         vec_Ap,
                                         CUDA_R_64F,
                                         CUSPARSE_SPMV_ALG_DEFAULT,
                                         &spmv_buffer_size),
                "FLD2D audit SpMV buffer size failed");
  if (ws.d_spmv.size != spmv_buffer_size) {
    ws.d_spmv.resize(spmv_buffer_size);
  }
  cusparse_check(cusparseSpMV(ws.handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &one,
                              mat,
                              vec_x,
                              &zero,
                              vec_Ap,
                              CUDA_R_64F,
                              CUSPARSE_SPMV_ALG_DEFAULT,
                              ws.d_spmv.ptr),
                "FLD2D audit SpMV failed");
  static_cast<void>(cusparseDestroyDnVec(vec_Ap));
  static_cast<void>(cusparseDestroyDnVec(vec_x));
  static_cast<void>(cusparseDestroySpMat(mat));

  const bool mpi_active = ws.mpi_n_ranks > 1;
  TENRYU_ASSERT(!mpi_active ||
                    (ws.mpi_n_cells > 0 && ws.mpi_n_groups > 0 &&
                     n_rows == ws.mpi_n_cells * ws.mpi_n_groups),
                "FLD2D audit diagnostics: MPI context mismatch");
  const int pp_n_cells = mpi_active ? ws.mpi_n_cells : n_rows;
  const int pp_c0 = mpi_active ? ws.mpi_c_begin : 0;
  const int pp_owned = mpi_active ? (ws.mpi_c_end - ws.mpi_c_begin) : n_rows;
  const int pp_total = mpi_active ? pp_owned * ws.mpi_n_groups : n_rows;
  ws.d_diag_scalars.resize(2U * sizeof(double));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, 2U * sizeof(double)),
             "FLD2D zero audit diagnostic scalars failed");
  const int grid = (pp_total + kBlock - 1) / kBlock;
  post_publish_residual_kernel<<<grid, kBlock>>>(
      rhs, ws.d_Ap.as<double>(), ws.d_diag_scalars.as<double>(),
      pp_n_cells, pp_c0, pp_owned, pp_total);
  cuda_check(cudaGetLastError(), "FLD2D audit residual launch failed");

  double reduced[2] = {0.0, 0.0};
  cuda_check(cudaMemcpy(reduced,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(reduced),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy audit diagnostic scalars failed");
  const double r_l2 = std::sqrt(std::max(
      mpi_active
          ? device_dot_owned_global(ws, ws.d_Ap.as<double>(),
                                    ws.d_Ap.as<double>(), nullptr)
          : device_dot(ws.d_Ap.as<double>(), ws.d_Ap.as<double>(), n_rows,
                       ws.d_scalar),
      0.0));
  const double rhs_l2 = std::sqrt(std::max(
      mpi_active ? device_dot_owned_global(ws, rhs, rhs, nullptr)
                 : device_dot(rhs, rhs, n_rows, ws.d_scalar),
      0.0));
  if (mpi_active) {
    const parallel::Reduction reduction(ws.mpi_n_ranks);
    reduced[0] = reduction.allreduce_sum(finite_or_zero(reduced[0]));
    reduced[1] = reduction.allreduce_max(finite_or_zero(reduced[1]));
  }
  diagnostics.true_residual_l2_rel = r_l2 / std::max(rhs_l2, 1.0e-300);
  diagnostics.true_residual_l2_abs = r_l2;
  diagnostics.true_residual_max = fmax(finite_or_zero(reduced[1]), 0.0);
  diagnostics.E_solver_abs = fabs(finite_or_zero(reduced[0]));
  return diagnostics;
}

void copy_rad_E_to_old(core::State& state, const int n_cells, const int n_groups) {
  const std::size_t bytes =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_groups) *
      sizeof(double);
  if (bytes > 0U) {
    cuda_check(cudaMemcpy(state.rad_E_old.data(),
                          state.rad_E.data(),
                          bytes,
                          cudaMemcpyDeviceToDevice),
               "FLD2D update rad_E_old failed");
  }
}

double reduce_max_delta_T(core::State& state, const int n_cells) {
  zero_reduction_scalar(state);
  const core::State::LaunchWindow dw = state.owned_cell_window(n_cells);
  const int grid = (dw.count() + kBlock - 1) / kBlock;
  if (grid > 0) {
    max_reduce_kernel<<<grid, kBlock>>>(state.fld_delta_T.data() + dw.begin,
                                        state.fld_reduction_work.data(),
                                        dw.count());
    cuda_check(cudaGetLastError(), "FLD2D residual reduction launch failed");
  }
  return copy_reduction_scalar(state);
}

double compute_escaped_energy(core::State& state,
                              const int nr,
                              const int nz,
                              const int n_groups,
                              const double dt,
                              const int outer_r_bc,
                              const int z_bottom_bc,
                              const int z_top_bc,
                              const double T_supply_z_bottom_eV,
                              const double T_supply_z_top_eV,
                              const int state_supply_boundary_policy =
                                  kStateSupplyBoundaryPolicyLocalDCurrent,
                              const int limiter = 0,
                              const double rho_supply_z_bottom = 0.0,
                              const double rho_supply_z_top = 0.0) {
  zero_reduction_scalar(state);
  const core::State::LaunchWindow cw = state.owned_cell_window(nr * nz);
  const int total = cw.count() * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    escaped_energy_2d_kernel<<<grid, kBlock>>>(state.x_r.data(),
                                               state.x_z.data(),
                                               state.vol.data(),
                                               state.rad_E.data(),
                                               state.fld_D_cell.data(),
                                               state.fld_reduction_work.data(),
                                               cw.begin,
                                               cw.end,
                                               nr,
                                               nz,
                                               n_groups,
                                               dt,
                                               outer_r_bc,
                                               z_bottom_bc,
                                               z_top_bc,
                                               T_supply_z_bottom_eV,
                                               T_supply_z_top_eV,
                                               state.rho.data(),
                                               state.fld_sigma_R.data(),
                                               state_supply_boundary_policy,
                                               limiter,
                                               rho_supply_z_bottom,
                                               rho_supply_z_top);
    cuda_check(cudaGetLastError(), "FLD2D escaped energy launch failed");
  }
  return copy_reduction_scalar(state);
}

void compute_state_supply_flux_energy(core::State& state,
                                      const int nr,
                                      const int nz,
                                      const int n_groups,
                                      const double dt,
                                      const int z_bottom_bc,
                                      const int z_top_bc,
                                      const double T_supply_z_bottom_eV,
                                      const double T_supply_z_top_eV,
                                      const int state_supply_boundary_policy =
                                          kStateSupplyBoundaryPolicyLocalDCurrent,
                                      const int limiter = 0,
                                      const double rho_supply_z_bottom = 0.0,
                                      const double rho_supply_z_top = 0.0) {
  state.fld_state_supply_in_step = 0.0;
  state.fld_state_supply_out_step = 0.0;
  state.fld_state_supply_net_step = 0.0;
  if (z_bottom_bc != kFldBcStateSupply && z_top_bc != kFldBcStateSupply) {
    return;
  }
  auto& ws = fld_2d_workspace();
  constexpr std::size_t kNumScalars = 3U;
  ws.d_diag_scalars.resize(kNumScalars * sizeof(double));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, kNumScalars * sizeof(double)),
             "FLD2D zero state_supply flux tally failed");
  const core::State::LaunchWindow cw = state.owned_cell_window(nr * nz);
  const int total = cw.count() * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    state_supply_flux_2d_kernel<<<grid, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.rad_E.data(),
        state.fld_D_cell.data(),
        ws.d_diag_scalars.as<double>(),
        cw.begin,
        cw.end,
        nr,
        nz,
        n_groups,
        dt,
        z_bottom_bc,
        z_top_bc,
        T_supply_z_bottom_eV,
        T_supply_z_top_eV,
        state.rho.data(),
        state.fld_sigma_R.data(),
        state_supply_boundary_policy,
        limiter,
        rho_supply_z_bottom,
        rho_supply_z_top);
    cuda_check(cudaGetLastError(), "FLD2D state_supply flux tally launch failed");
  }
  double values[kNumScalars] = {};
  cuda_check(cudaMemcpy(values,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy state_supply flux tally failed");
  state.fld_state_supply_in_step = finite_or_zero(values[0]);
  state.fld_state_supply_out_step = finite_or_zero(values[1]);
  state.fld_state_supply_net_step = finite_or_zero(values[2]);
  state.fld_state_supply_in_cumulative += state.fld_state_supply_in_step;
  state.fld_state_supply_out_cumulative += state.fld_state_supply_out_step;
  state.fld_state_supply_net_cumulative += state.fld_state_supply_net_step;
}

EscapeBreakdownTotals compute_escape_breakdown_diagnostics(
    core::State& state,
    Fld2DWorkspace& ws,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt,
    const int outer_r_bc,
    const int z_bottom_bc,
    const int z_top_bc,
    const double rho_vac_threshold) {
  EscapeBreakdownTotals diagnostics;
  const int n_cells = nr * nz;
  if (n_cells <= 0 || n_groups <= 0) {
    return diagnostics;
  }
  constexpr std::size_t kNumTotals = 13U;
  constexpr std::size_t kNumRecords = 6U;
  constexpr std::size_t kTotalsBytes = kNumTotals * sizeof(double);
  constexpr std::size_t kRecordOffset = kTotalsBytes;
  constexpr std::size_t kComponentOffset =
      kRecordOffset + kNumRecords * sizeof(RogueRecord);
  constexpr std::size_t kBufferBytes =
      kComponentOffset + sizeof(EscapeBreakdownComponentRecord);
  ws.d_diag_scalars.resize(kBufferBytes);
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, kBufferBytes),
             "FLD2D zero escape-breakdown diagnostics failed");
  char* const buffer = static_cast<char*>(ws.d_diag_scalars.ptr);
  double* const totals = reinterpret_cast<double*>(buffer);
  auto* const records =
      reinterpret_cast<RogueRecord*>(buffer + kRecordOffset);
  auto* const component_record =
      reinterpret_cast<EscapeBreakdownComponentRecord*>(buffer +
                                                       kComponentOffset);
  const int grid = (n_cells + kBlock - 1) / kBlock;
  compute_escape_breakdown_kernel<<<grid, kBlock>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.rho.data(),
      state.rad_E.data(),
      state.fld_sigma_a.data(),
      state.fld_D_cell.data(),
      ws.d_col_indices.as<int>(),
      ws.d_values.as<double>(),
      nr,
      nz,
      n_groups,
      dt,
      outer_r_bc,
      z_bottom_bc,
      z_top_bc,
      rho_vac_threshold,
      totals,
      records,
      component_record);
  cuda_check(cudaGetLastError(),
             "FLD2D escape-breakdown diagnostic launch failed");
  double total_values[kNumTotals] = {};
  RogueRecord record_values[kNumRecords] = {};
  EscapeBreakdownComponentRecord component_value;
  cuda_check(cudaMemcpy(total_values,
                        totals,
                        sizeof(total_values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy escape-breakdown totals failed");
  cuda_check(cudaMemcpy(record_values,
                        records,
                        sizeof(record_values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy escape-breakdown records failed");
  cuda_check(cudaMemcpy(&component_value,
                        component_record,
                        sizeof(component_value),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy escape-breakdown component record failed");
  diagnostics.total_outer_r = finite_or_zero(total_values[0]);
  diagnostics.total_z_bottom = finite_or_zero(total_values[1]);
  diagnostics.total_z_top = finite_or_zero(total_values[2]);
  diagnostics.total_vacuum_outer_r = finite_or_zero(total_values[3]);
  diagnostics.total_vacuum_z_bottom = finite_or_zero(total_values[4]);
  diagnostics.total_vacuum_z_top = finite_or_zero(total_values[5]);
  diagnostics.sum_signed_delta_outer_r = finite_or_zero(total_values[6]);
  diagnostics.sum_signed_delta_z_bottom = finite_or_zero(total_values[7]);
  diagnostics.sum_signed_delta_z_top = finite_or_zero(total_values[8]);
  diagnostics.sum_abs_delta = finite_or_zero(total_values[9]);
  diagnostics.count_delta_gt_1 =
      static_cast<int>(llround(finite_or_zero(total_values[10])));
  diagnostics.count_delta_gt_10 =
      static_cast<int>(llround(finite_or_zero(total_values[11])));
  diagnostics.count_delta_gt_100 =
      static_cast<int>(llround(finite_or_zero(total_values[12])));
  diagnostics.max_per_cell_outer_r = finite_or_zero(record_values[0].value);
  diagnostics.max_per_cell_z_bottom = finite_or_zero(record_values[1].value);
  diagnostics.max_per_cell_z_top = finite_or_zero(record_values[2].value);
  diagnostics.max_boundary_diag_escape_delta =
      finite_or_zero(record_values[3].value);
  diagnostics.max_pos_delta = finite_or_zero(record_values[4].value);
  diagnostics.max_neg_delta = -finite_or_zero(record_values[5].value);
  const int outer_cell = record_values[0].cell_idx;
  diagnostics.max_cell_outer_r_i = (outer_cell >= 0) ? (outer_cell / nz) : -1;
  diagnostics.max_cell_outer_r_j =
      (outer_cell >= 0) ? (outer_cell - diagnostics.max_cell_outer_r_i * nz)
                        : -1;
  const int z_bottom_cell = record_values[1].cell_idx;
  diagnostics.max_cell_z_bottom_i =
      (z_bottom_cell >= 0) ? (z_bottom_cell / nz) : -1;
  diagnostics.max_cell_z_bottom_j =
      (z_bottom_cell >= 0)
          ? (z_bottom_cell - diagnostics.max_cell_z_bottom_i * nz)
          : -1;
  const int z_top_cell = record_values[2].cell_idx;
  diagnostics.max_cell_z_top_i = (z_top_cell >= 0) ? (z_top_cell / nz) : -1;
  diagnostics.max_cell_z_top_j =
      (z_top_cell >= 0) ? (z_top_cell - diagnostics.max_cell_z_top_i * nz)
                        : -1;
  const int delta_cell = record_values[3].cell_idx;
  diagnostics.max_boundary_diag_escape_delta_i =
      (delta_cell >= 0) ? (delta_cell / nz) : -1;
  diagnostics.max_boundary_diag_escape_delta_j =
      (delta_cell >= 0)
          ? (delta_cell - diagnostics.max_boundary_diag_escape_delta_i * nz)
          : -1;
  diagnostics.max_boundary_diag_escape_delta_g = record_values[3].group_idx;
  const int max_pos_cell = record_values[4].cell_idx;
  diagnostics.max_pos_delta_i =
      (max_pos_cell >= 0) ? (max_pos_cell / nz) : -1;
  diagnostics.max_pos_delta_j =
      (max_pos_cell >= 0)
          ? (max_pos_cell - diagnostics.max_pos_delta_i * nz)
          : -1;
  const int max_neg_cell = record_values[5].cell_idx;
  diagnostics.max_neg_delta_i =
      (max_neg_cell >= 0) ? (max_neg_cell / nz) : -1;
  diagnostics.max_neg_delta_j =
      (max_neg_cell >= 0)
          ? (max_neg_cell - diagnostics.max_neg_delta_i * nz)
          : -1;
  diagnostics.csr_diag_value = finite_or_zero(component_value.csr_diag_value);
  diagnostics.V_op = finite_or_zero(component_value.V_op);
  diagnostics.dt_c_sigma_V = finite_or_zero(component_value.dt_c_sigma_V);
  diagnostics.interior_diag_sum =
      finite_or_zero(component_value.interior_diag_sum);
  diagnostics.csr_boundary_diag =
      finite_or_zero(component_value.csr_boundary_diag);
  diagnostics.formula_boundary_coef =
      finite_or_zero(component_value.formula_boundary_coef);
  diagnostics.rad_E_at_cell = finite_or_zero(component_value.rad_E_at_cell);
  diagnostics.sigma_a_at_cell =
      finite_or_zero(component_value.sigma_a_at_cell);
  diagnostics.rho_at_cell = finite_or_zero(component_value.rho_at_cell);
  diagnostics.D_cell_at_cell = finite_or_zero(component_value.D_cell_at_cell);
  return diagnostics;
}

const char* fld_face_side_name(const int side) {
  switch (side) {
    case 0:
      return "i-1";
    case 1:
      return "i+1";
    case 2:
      return "j-1";
    case 3:
      return "j+1";
    default:
      return "unknown";
  }
}

void append_face_trace_side(std::ostringstream& oss,
                            const FldFaceTraceSideRecord& rec) {
  oss << " side=" << fld_face_side_name(rec.side)
      << " valid=" << rec.valid
      << " neighbor=(" << rec.neighbor_i << "," << rec.neighbor_j << ")"
      << " group=" << rec.group
      << " D_self=" << rec.D_self
      << " D_neighbor=" << rec.D_neighbor
      << " D_face=" << rec.D_face
      << " area_face=" << rec.area_face
      << " dist_face=" << rec.dist_face
      << " coef=" << rec.coef
      << " csr_col=" << rec.csr_col
      << " csr_value=" << rec.csr_value
      << " expected_csr_value=" << rec.expected_csr_value
      << " coef_finite=" << rec.coef_finite
      << " coef_fallback=" << rec.coef_fallback
      << " csr_mismatch=" << rec.csr_mismatch
      << " D_self_finite=" << rec.D_self_finite
      << " D_neighbor_finite=" << rec.D_neighbor_finite
      << " D_face_finite=" << rec.D_face_finite
      << " self_rho=" << rec.self_rho
      << " self_sigma=" << rec.self_sigma
      << " self_rad_E=" << rec.self_rad_E
      << " neighbor_rho=" << rec.neighbor_rho
      << " neighbor_sigma=" << rec.neighbor_sigma
      << " neighbor_rad_E=" << rec.neighbor_rad_E;
}

void log_fld_face_trace(core::State& state,
                        Fld2DWorkspace& ws,
                        const int nr,
                        const int nz,
                        const int n_groups,
                        const double dt,
                        const int target_cell) {
  FldFaceTraceSideRecord host_records[4] = {};
  ws.d_diag_scalars.resize(sizeof(host_records));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, sizeof(host_records)),
             "FLD2D zero face-trace diagnostics failed");
  compute_face_coef_trace_kernel<<<1, 1>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.rho.data(),
      state.rad_E.data(),
      state.fld_sigma_a.data(),
      state.fld_D_cell.data(),
      state.fld_cell_rc.data(),
      state.fld_cell_zc.data(),
      ws.d_col_indices.as<int>(),
      ws.d_values.as<double>(),
      nr,
      nz,
      n_groups,
      dt,
      target_cell,
      ws.d_diag_scalars.as<FldFaceTraceSideRecord>());
  cuda_check(cudaGetLastError(), "FLD2D face-trace launch failed");
  cuda_check(cudaMemcpy(host_records,
                        ws.d_diag_scalars.as<FldFaceTraceSideRecord>(),
                        sizeof(host_records),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy face-trace diagnostics failed");

  const int cell_i = (target_cell >= 0) ? target_cell / nz : -1;
  const int cell_j = (target_cell >= 0) ? target_cell - cell_i * nz : -1;
  std::ostringstream oss;
  oss << std::setprecision(17)
      << "[fld_2d_rz_face_trace] step=" << state.step
      << " cell=" << target_cell
      << " cell_ij=(" << cell_i << "," << cell_j << ")";
  for (const FldFaceTraceSideRecord& rec : host_records) {
    append_face_trace_side(oss, rec);
  }
  core::log_info(oss.str());
}

FldFaceGlobalMaxDiagnostics compute_fld_face_global_max_diagnostics(
    core::State& state,
    Fld2DWorkspace& ws,
    const int nr,
    const int nz,
    const int n_groups,
    const double dt) {
  FldFaceGlobalMaxDiagnostics diagnostics;
  const int n_cells = nr * nz;
  const int n_r_faces = (nr > 1) ? ((nr - 1) * nz) : 0;
  const int n_z_faces = (nz > 1) ? (nr * (nz - 1)) : 0;
  const int total = n_cells * n_groups + (n_r_faces + n_z_faces) * n_groups;
  if (total <= 0) {
    return diagnostics;
  }
  constexpr std::size_t kDCellOffset = 0U;
  constexpr std::size_t kDFaceOffset = kDCellOffset + sizeof(RogueRecord);
  constexpr std::size_t kCoefOffset =
      kDFaceOffset + sizeof(FldFaceGlobalMaxRecord);
  constexpr std::size_t kAreaDistOffset =
      kCoefOffset + sizeof(FldFaceGlobalMaxRecord);
  constexpr std::size_t kBufferBytes =
      kAreaDistOffset + sizeof(FldFaceGlobalMaxRecord);
  ws.d_diag_scalars.resize(kBufferBytes);
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, kBufferBytes),
             "FLD2D zero face-global-max diagnostics failed");
  char* const buffer = static_cast<char*>(ws.d_diag_scalars.ptr);
  auto* const max_D_cell = reinterpret_cast<RogueRecord*>(buffer + kDCellOffset);
  auto* const max_D_face =
      reinterpret_cast<FldFaceGlobalMaxRecord*>(buffer + kDFaceOffset);
  auto* const max_coef =
      reinterpret_cast<FldFaceGlobalMaxRecord*>(buffer + kCoefOffset);
  auto* const max_area_dist_ratio =
      reinterpret_cast<FldFaceGlobalMaxRecord*>(buffer + kAreaDistOffset);
  compute_face_global_max_kernel<<<1, 1>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.fld_D_cell.data(),
      state.fld_cell_rc.data(),
      state.fld_cell_zc.data(),
      nr,
      nz,
      n_groups,
      dt,
      max_D_cell,
      max_D_face,
      max_coef,
      max_area_dist_ratio);
  cuda_check(cudaGetLastError(), "FLD2D face-global-max launch failed");
  cuda_check(cudaMemcpy(&diagnostics.max_D_cell,
                        max_D_cell,
                        sizeof(diagnostics.max_D_cell),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy max D_cell diagnostic failed");
  cuda_check(cudaMemcpy(&diagnostics.max_D_face,
                        max_D_face,
                        sizeof(diagnostics.max_D_face),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy max D_face diagnostic failed");
  cuda_check(cudaMemcpy(&diagnostics.max_coef,
                        max_coef,
                        sizeof(diagnostics.max_coef),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy max coef diagnostic failed");
  cuda_check(cudaMemcpy(&diagnostics.max_area_dist_ratio,
                        max_area_dist_ratio,
                        sizeof(diagnostics.max_area_dist_ratio),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy max area/dist diagnostic failed");
  return diagnostics;
}

void append_global_face_record(std::ostringstream& oss,
                               const char* prefix,
                               const FldFaceGlobalMaxRecord& rec) {
  oss << " " << prefix << "=" << finite_or_zero(rec.value)
      << " " << prefix << "_cell=(" << rec.cell_i << "," << rec.cell_j << ")"
      << " " << prefix << "_neighbor=(" << rec.neighbor_i << ","
      << rec.neighbor_j << ")"
      << " " << prefix << "_side=" << fld_face_side_name(rec.side)
      << " " << prefix << "_group=" << rec.group
      << " " << prefix << "_D_self=" << finite_or_zero(rec.D_self)
      << " " << prefix << "_D_neighbor=" << finite_or_zero(rec.D_neighbor)
      << " " << prefix << "_D_face=" << finite_or_zero(rec.D_face)
      << " " << prefix << "_area_face=" << finite_or_zero(rec.area_face)
      << " " << prefix << "_dist_face=" << finite_or_zero(rec.dist_face)
      << " " << prefix << "_area_dist_ratio="
      << finite_or_zero(rec.area_dist_ratio)
      << " " << prefix << "_coef=" << finite_or_zero(rec.coef);
}

void log_fld_face_global_max(core::State& state,
                             Fld2DWorkspace& ws,
                             const int nr,
                             const int nz,
                             const int n_groups,
                             const double dt) {
  const FldFaceGlobalMaxDiagnostics diagnostics =
      compute_fld_face_global_max_diagnostics(state, ws, nr, nz, n_groups, dt);
  const int d_cell = diagnostics.max_D_cell.cell_idx;
  const int d_i = (d_cell >= 0) ? d_cell / nz : -1;
  const int d_j = (d_cell >= 0) ? d_cell - d_i * nz : -1;
  std::ostringstream oss;
  oss << std::setprecision(17)
      << "[fld_2d_rz_global_max] step=" << state.step
      << " max_D_cell=" << finite_or_zero(diagnostics.max_D_cell.value)
      << " max_D_cell_cell=" << d_cell
      << " max_D_cell_ij=(" << d_i << "," << d_j << ")"
      << " max_D_cell_group=" << diagnostics.max_D_cell.group_idx;
  append_global_face_record(oss, "max_D_face", diagnostics.max_D_face);
  append_global_face_record(oss, "max_coef", diagnostics.max_coef);
  append_global_face_record(oss,
                            "max_area_dist_ratio",
                            diagnostics.max_area_dist_ratio);
  core::log_info(oss.str());
}

// Indirect-drive Tr(t) resolution for the deterministic 2D marshak z faces
// (docs/design/2d_tr_drive_port_spec.md §2). IMC-2D precedence verbatim
// (source.cu emit_marshak): the per-face table (canonical key, then alias)
// wins over the constant marshak_Tr_eV (>0), which wins over the scalar
// marshak_Tr table. Solve-entry time; tables are frozen at init (no runtime
// Python).
double resolve_marshak_tr_face_eV(const core::Config& cfg,
                                  const core::State& state,
                                  const char* key,
                                  const char* alias) {
  double t_r = cfg.radiation.boundary.marshak_Tr_eV;
  if (!(t_r > 0.0) && state.marshak_Tr_1d.has_value()) {
    t_r = state.marshak_Tr_1d->eval(state.t);
  }
  const auto it_key = state.marshak_Tr_face_tables.find(key);
  if (it_key != state.marshak_Tr_face_tables.end()) {
    t_r = it_key->second.eval(state.t);
  } else {
    const auto it_alias = state.marshak_Tr_face_tables.find(alias);
    if (it_alias != state.marshak_Tr_face_tables.end()) {
      t_r = it_alias->second.eval(state.t);
    }
  }
  return std::max(t_r, 0.0);
}

double compute_marshak_in_energy(core::State& state,
                                 const int nr,
                                 const int nz,
                                 const double dt,
                                 const int z_bottom_bc,
                                 const int z_top_bc,
                                 const double marshak_flux_erg_per_cm2_s) {
  if (!(marshak_flux_erg_per_cm2_s > 0.0) ||
      (z_bottom_bc != kFldBcMarshak && z_top_bc != kFldBcMarshak)) {
    return 0.0;
  }
  zero_reduction_scalar(state);
  // r-slab: owned cells [c0, c1) are whole i-planes, so the owned radial
  // range is [c0/nz, c1/nz).
  const core::State::LaunchWindow cw = state.owned_cell_window(nr * nz);
  const int i_begin = (nz > 0) ? cw.begin / nz : 0;
  const int i_end = (nz > 0) ? cw.end / nz : 0;
  const int grid = (i_end - i_begin + kBlock - 1) / kBlock;
  if (grid > 0) {
    marshak_in_energy_2d_kernel<<<grid, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.fld_reduction_work.data(),
        i_begin,
        i_end,
        nr,
        nz,
        dt,
        z_bottom_bc,
        z_top_bc,
        marshak_flux_erg_per_cm2_s);
    cuda_check(cudaGetLastError(), "FLD2D Marshak source energy launch failed");
  }
  return copy_reduction_scalar(state);
}

struct FldConservationDiagnostics {
  double residual_sum = 0.0;
  double rad_delta = 0.0;
  double emit_minus_dep = 0.0;
};

struct FldPerRowDefectDiagnostics {
  double sum_defect = 0.0;
  double sum_vol_diff = 0.0;
  FldPerRowDefectRecord max_record;
  int num_vol_mismatch_cells = 0;
  int num_abs_gt_1e_3 = 0;
  int num_abs_gt_1e_2 = 0;
  int num_abs_gt_1e_1 = 0;
  int num_abs_gt_1 = 0;
  int num_abs_gt_10 = 0;
};

FldConservationDiagnostics compute_fld_conservation_diagnostics(
    core::State& state,
    Fld2DWorkspace& ws,
    const int n_cells,
    const int n_groups,
    const int n_rows) {
  FldConservationDiagnostics diagnostics;
  if (n_rows <= 0) {
    return diagnostics;
  }
  constexpr std::size_t kNumScalars = 3U;
  ws.d_diag_scalars.resize(kNumScalars * sizeof(double));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, kNumScalars * sizeof(double)),
             "FLD2D zero conservation diagnostics failed");
  const int grid = (n_rows + kBlock - 1) / kBlock;
  compute_operator_residual_sum_kernel<<<grid, kBlock>>>(
      ws.d_row_offsets.as<int>(),
      ws.d_col_indices.as<int>(),
      ws.d_values.as<double>(),
      ws.d_x.as<double>(),
      ws.d_rhs.as<double>(),
      ws.d_diag_scalars.as<double>(),
      n_rows);
  cuda_check(cudaGetLastError(),
             "FLD2D conservation residual diagnostic launch failed");
  compute_rad_delta_sum_kernel<<<grid, kBlock>>>(
      state.rad_E.data(),
      state.rad_E_old.data(),
      state.vol.data(),
      ws.d_diag_scalars.as<double>() + 1,
      n_cells,
      n_groups);
  cuda_check(cudaGetLastError(),
             "FLD2D conservation rad-delta diagnostic launch failed");
  compute_emit_minus_dep_sum_kernel<<<grid, kBlock>>>(
      state.rad_emit.data(),
      state.rad_dep.data(),
      ws.d_diag_scalars.as<double>() + 2,
      n_rows);
  cuda_check(cudaGetLastError(),
             "FLD2D conservation emit-dep diagnostic launch failed");
  double values[kNumScalars] = {0.0, 0.0, 0.0};
  cuda_check(cudaMemcpy(values,
                        ws.d_diag_scalars.as<double>(),
                        sizeof(values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy conservation diagnostics failed");
  diagnostics.residual_sum = finite_or_zero(values[0]);
  diagnostics.rad_delta = finite_or_zero(values[1]);
  diagnostics.emit_minus_dep = finite_or_zero(values[2]);
  return diagnostics;
}

FldPerRowDefectDiagnostics compute_fld_per_row_defect_diagnostics(
    core::State& state,
    Fld2DWorkspace& ws,
    const int nr,
    const int nz,
    const int n_groups,
    const int n_rows,
    const double dt,
    const int outer_r_bc,
    const int z_bottom_bc,
    const int z_top_bc,
    const double T_supply_z_bottom_eV,
    const double T_supply_z_top_eV,
    const bool use_fleck,
    const int state_supply_boundary_policy =
        kStateSupplyBoundaryPolicyLocalDCurrent,
    const int limiter = 0,
    const double rho_supply_z_bottom = 0.0,
    const double rho_supply_z_top = 0.0) {
  FldPerRowDefectDiagnostics diagnostics;
  if (n_rows <= 0) {
    return diagnostics;
  }
  constexpr std::size_t kScalarBytes = 2U * sizeof(double);
  constexpr std::size_t kRecordOffset = kScalarBytes;
  constexpr std::size_t kBufferBytes =
      kRecordOffset + sizeof(FldPerRowDefectRecord);
  constexpr std::size_t kNumCounts = 6U;
  ws.d_diag_scalars.resize(kBufferBytes);
  ws.d_diag_ints.resize(kNumCounts * sizeof(int));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, kBufferBytes),
             "FLD2D zero per-row defect diagnostics failed");
  cuda_check(cudaMemset(ws.d_diag_ints.ptr, 0, kNumCounts * sizeof(int)),
             "FLD2D zero per-row defect counts failed");
  char* const buffer = static_cast<char*>(ws.d_diag_scalars.ptr);
  double* const scalars = reinterpret_cast<double*>(buffer);
  auto* const max_record =
      reinterpret_cast<FldPerRowDefectRecord*>(buffer + kRecordOffset);
  const int grid = (n_rows + kBlock - 1) / kBlock;
  compute_per_row_defect_kernel<<<grid, kBlock>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.fld_sigma_a.data(),
      state.fld_eta.data(),
      use_fleck ? state.fld_nlte_f_work.data() : nullptr,
      state.rad_E.data(),
      state.rad_E_old.data(),
      state.fld_D_cell.data(),
      nr,
      nz,
      n_groups,
      dt,
      outer_r_bc,
      z_bottom_bc,
      z_top_bc,
      T_supply_z_bottom_eV,
      T_supply_z_top_eV,
      scalars,
      max_record,
      scalars + 1,
      ws.d_diag_ints.as<int>(),
      state.rho.data(),
      state.fld_sigma_R.data(),
      state_supply_boundary_policy,
      limiter,
      rho_supply_z_bottom,
      rho_supply_z_top);
  cuda_check(cudaGetLastError(),
             "FLD2D per-row defect diagnostic launch failed");
  double scalar_values[2] = {0.0, 0.0};
  int count_values[kNumCounts] = {0, 0, 0, 0, 0, 0};
  cuda_check(cudaMemcpy(scalar_values,
                        scalars,
                        sizeof(scalar_values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy per-row defect scalars failed");
  cuda_check(cudaMemcpy(&diagnostics.max_record,
                        max_record,
                        sizeof(FldPerRowDefectRecord),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy per-row defect max record failed");
  cuda_check(cudaMemcpy(count_values,
                        ws.d_diag_ints.as<int>(),
                        sizeof(count_values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy per-row defect counts failed");
  diagnostics.sum_defect = finite_or_zero(scalar_values[0]);
  diagnostics.sum_vol_diff = finite_or_zero(scalar_values[1]);
  diagnostics.max_record.value = finite_or_zero(diagnostics.max_record.value);
  diagnostics.max_record.row_defect =
      finite_or_zero(diagnostics.max_record.row_defect);
  diagnostics.max_record.V_op = finite_or_zero(diagnostics.max_record.V_op);
  diagnostics.max_record.V_state =
      finite_or_zero(diagnostics.max_record.V_state);
  diagnostics.num_vol_mismatch_cells = count_values[0];
  diagnostics.num_abs_gt_1e_3 = count_values[1];
  diagnostics.num_abs_gt_1e_2 = count_values[2];
  diagnostics.num_abs_gt_1e_1 = count_values[3];
  diagnostics.num_abs_gt_1 = count_values[4];
  diagnostics.num_abs_gt_10 = count_values[5];
  return diagnostics;
}

FldRowIdentityDiagnostics compute_fld_row_identity_diagnostics(
    core::State& state,
    Fld2DWorkspace& ws,
    const int nr,
    const int nz,
    const int n_groups,
    const int n_rows,
    const double dt,
    const int outer_r_bc,
    const int z_bottom_bc,
    const int z_top_bc,
    const double marshak_flux,
    const double T_supply_z_bottom_eV,
    const double T_supply_z_top_eV,
    const bool use_fleck,
    const int state_supply_boundary_policy =
        kStateSupplyBoundaryPolicyLocalDCurrent,
    const int limiter = 0,
    const double rho_supply_z_bottom = 0.0,
    const double rho_supply_z_top = 0.0) {
  FldRowIdentityDiagnostics diagnostics;
  if (n_rows <= 0) {
    return diagnostics;
  }
  constexpr std::size_t kNumSums = 9U;
  constexpr std::size_t kNumRecords = 4U;
  constexpr std::size_t kSumsBytes = kNumSums * sizeof(double);
  constexpr std::size_t kRecordOffset = kSumsBytes;
  constexpr std::size_t kBufferBytes =
      kRecordOffset + kNumRecords * sizeof(RogueRecord);
  ws.d_diag_scalars.resize(kBufferBytes);
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, kBufferBytes),
             "FLD2D zero row-identity diagnostics failed");
  char* const buffer = static_cast<char*>(ws.d_diag_scalars.ptr);
  double* const sums = reinterpret_cast<double*>(buffer);
  auto* const records =
      reinterpret_cast<RogueRecord*>(buffer + kRecordOffset);
  const int grid = (n_rows + kBlock - 1) / kBlock;
  compute_row_identity_diagnostic_kernel<<<grid, kBlock>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.fld_sigma_a.data(),
      state.fld_eta.data(),
      use_fleck ? state.fld_nlte_f_work.data() : nullptr,
      state.rad_E.data(),
      state.rad_E_old.data(),
      state.rad_emit.data(),
      state.rad_dep.data(),
      state.fld_D_cell.data(),
      state.fld_cell_rc.data(),
      state.fld_cell_zc.data(),
      ws.d_row_offsets.as<int>(),
      ws.d_col_indices.as<int>(),
      ws.d_values.as<double>(),
      ws.d_x.as<double>(),
      ws.d_rhs.as<double>(),
      nr,
      nz,
      n_groups,
      dt,
      outer_r_bc,
      z_bottom_bc,
      z_top_bc,
      marshak_flux,
      T_supply_z_bottom_eV,
      T_supply_z_top_eV,
      sums,
      records,
      state.rho.data(),
      state.fld_sigma_R.data(),
      state_supply_boundary_policy,
      limiter,
      rho_supply_z_bottom,
      rho_supply_z_top);
  cuda_check(cudaGetLastError(),
             "FLD2D row-identity diagnostic launch failed");
  double sum_values[kNumSums] = {};
  RogueRecord record_values[kNumRecords] = {};
  cuda_check(cudaMemcpy(sum_values,
                        sums,
                        sizeof(sum_values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy row-identity diagnostic sums failed");
  cuda_check(cudaMemcpy(record_values,
                        records,
                        sizeof(record_values),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy row-identity diagnostic records failed");
  diagnostics.sum_csr_residual = finite_or_zero(sum_values[0]);
  diagnostics.sum_matrix_formula_residual = finite_or_zero(sum_values[1]);
  diagnostics.sum_tally_formula_residual = finite_or_zero(sum_values[2]);
  diagnostics.sum_face_div = finite_or_zero(sum_values[3]);
  diagnostics.sum_abs_face_div = finite_or_zero(sum_values[4]);
  diagnostics.sum_emit_kernel = finite_or_zero(sum_values[5]);
  diagnostics.sum_dep_kernel = finite_or_zero(sum_values[6]);
  diagnostics.sum_emit_formula = finite_or_zero(sum_values[7]);
  diagnostics.sum_dep_formula = finite_or_zero(sum_values[8]);
  diagnostics.max_delta_csr_vs_matrix = record_values[0];
  diagnostics.max_delta_matrix_vs_tally = record_values[1];
  diagnostics.max_delta_emit_kernel_vs_formula = record_values[2];
  diagnostics.max_delta_dep_kernel_vs_formula = record_values[3];
  diagnostics.max_delta_csr_vs_matrix.value =
      finite_or_zero(diagnostics.max_delta_csr_vs_matrix.value);
  diagnostics.max_delta_matrix_vs_tally.value =
      finite_or_zero(diagnostics.max_delta_matrix_vs_tally.value);
  diagnostics.max_delta_emit_kernel_vs_formula.value =
      finite_or_zero(diagnostics.max_delta_emit_kernel_vs_formula.value);
  diagnostics.max_delta_dep_kernel_vs_formula.value =
      finite_or_zero(diagnostics.max_delta_dep_kernel_vs_formula.value);
  return diagnostics;
}

FldFaceSymmetryRecord compute_fld_face_symmetry_diagnostics(core::State& state,
                                                            Fld2DWorkspace& ws,
                                                            const int nr,
                                                            const int nz,
                                                            const int n_groups,
                                                            const double dt) {
  FldFaceSymmetryRecord record{};
  const int n_r_faces = (nr > 1) ? ((nr - 1) * nz) : 0;
  const int n_z_faces = (nz > 1) ? (nr * (nz - 1)) : 0;
  const int total = (n_r_faces + n_z_faces) * n_groups;
  if (total <= 0) {
    return record;
  }
  ws.d_diag_scalars.resize(sizeof(FldFaceSymmetryRecord));
  cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, sizeof(FldFaceSymmetryRecord)),
             "FLD2D zero face-symmetry record failed");
  const int grid = (total + kBlock - 1) / kBlock;
  fld_face_symmetry_audit_kernel<<<grid, kBlock>>>(
      state.x_r.data(),
      state.x_z.data(),
      state.vol.data(),
      state.rad_E.data(),
      state.fld_D_cell.data(),
      state.fld_cell_rc.data(),
      state.fld_cell_zc.data(),
      reinterpret_cast<FldFaceSymmetryRecord*>(ws.d_diag_scalars.ptr),
      nr,
      nz,
      n_groups,
      dt);
  cuda_check(cudaGetLastError(),
             "FLD2D face-symmetry conservation diagnostic launch failed");
  cuda_check(cudaMemcpy(&record,
                        ws.d_diag_scalars.ptr,
                        sizeof(FldFaceSymmetryRecord),
                        cudaMemcpyDeviceToHost),
             "FLD2D copy face-symmetry conservation diagnostic failed");
  record.value = finite_or_zero(record.value);
  record.c_lr = finite_or_zero(record.c_lr);
  record.c_rl = finite_or_zero(record.c_rl);
  record.energy_delta = finite_or_zero(record.energy_delta);
  return record;
}

void log_fld_trace_records(const std::vector<FldTraceRecord>& records,
                           const int step,
                           const int cell,
                           const int group) {
  int traced = 0;
  double Te_in_first = 0.0;
  double Te_in_last = 0.0;
  double E_old_first = 0.0;
  double E_old_last = 0.0;
  double rad_E_delta_first = 0.0;
  double rad_E_delta_last = 0.0;
  double rad_E_delta_max = 0.0;
  bool first = true;
  for (std::size_t k = 0; k < records.size(); ++k) {
    const FldTraceRecord& rec = records[k];
    if (rec.cell_matched == 0) {
      continue;
    }
    ++traced;
    if (first) {
      Te_in_first = rec.Te_in;
      E_old_first = rec.E_old_in;
      rad_E_delta_first = rec.rad_E_delta;
      first = false;
    }
    Te_in_last = rec.Te_in;
    E_old_last = rec.E_old_in;
    rad_E_delta_last = rec.rad_E_delta;
    rad_E_delta_max = std::max(rad_E_delta_max, finite_or_zero(rec.rad_E_delta));

    std::ostringstream oss;
    oss << std::setprecision(17)
        << "[fld_2d_rz_trace] step=" << step
        << " outer=" << k
        << " cell=" << cell
        << " group=" << group
        << " cell_matched=" << rec.cell_matched
        << " i=" << rec.i
        << " j=" << rec.j
        << " Te_in=" << rec.Te_in
        << " E_old=" << rec.E_old_in
        << " sigma_pa=" << rec.sigma_pa
        << " sigma_removal=" << rec.sigma_removal
        << " B_T=" << rec.B_T
        << " eta=" << rec.eta
        << " f=" << rec.fleck_f
        << " D=" << rec.D_cell
        << " rhs_V_E_old=" << rec.rhs_V_E_old
        << " rhs_dt_V_f_eta=" << rec.rhs_dt_V_f_eta
        << " rhs_dt_V_omf_csE=" << rec.rhs_dt_V_one_minus_f_csE
        << " rhs_boundary=" << rec.rhs_boundary
        << " rhs_total=" << rec.rhs_total
        << " diag_V=" << rec.diag_V
        << " diag_csigV=" << rec.diag_csigV
        << " diag_face_total=" << rec.diag_face_total
        << " diag_total=" << rec.diag_total
        << " offdiag_iL=" << rec.offdiag_iL
        << " offdiag_iR=" << rec.offdiag_iR
        << " offdiag_jB=" << rec.offdiag_jB
        << " offdiag_jT=" << rec.offdiag_jT
        << " x_raw=" << rec.x_raw
        << " x_pub=" << rec.x_pub
        << " clamp_delta=" << rec.clamp_delta
        << " Arow_x=" << rec.Arow_x
        << " Arow_x_minus_rhs=" << rec.Arow_x_minus_rhs
        << " xL=" << rec.xL
        << " xR=" << rec.xR
        << " xB=" << rec.xB
        << " xT=" << rec.xT
        << " rad_E_delta=" << rec.rad_E_delta;
    core::log_info(oss.str());
  }
  if (traced > 0) {
    std::ostringstream oss;
    oss << std::setprecision(17)
        << "[fld_2d_rz_trace_summary] step=" << step
        << " cell=" << cell
        << " group=" << group
        << " outers_traced=" << traced
        << " Te_in_first=" << Te_in_first
        << " Te_in_last=" << Te_in_last
        << " E_old_first=" << E_old_first
        << " E_old_last=" << E_old_last
        << " rad_E_delta_max=" << rad_E_delta_max
        << " rad_E_delta_first=" << rad_E_delta_first
        << " rad_E_delta_last=" << rad_E_delta_last;
    core::log_info(oss.str());
  }
}

}  // namespace

void advance_radiation_step_fld_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat,
    const double dt,
    const parallel::PartitionInfo& part,
    parallel::CommBuffers* bufs) {
  TENRYU_ASSERT(cfg.radiation.mode == core::RadiationMode::MultigroupDiffusion,
                "advance_radiation_step_fld_2d_rz requires multigroup_diffusion mode");
  TENRYU_ASSERT(state.mesh.dim == 2 || cfg.main.dimension == "2D_RZ",
                "advance_radiation_step_fld_2d_rz requires 2D_RZ state");
  TENRYU_ASSERT(dt > 0.0, "advance_radiation_step_fld_2d_rz requires dt > 0");
  const int nr = cfg.mesh.nr;
  const int nz = cfg.mesh.nz;
  const int n_cells = nr * nz;
  const int n_groups = std::max(cfg.radiation.groups, 1);
  const int n_mat = static_cast<int>(cfg.materials.materials.size());
  const std::size_t n_cell_mat =
      static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(n_mat);
  TENRYU_ASSERT(nr > 0 && nz > 0, "FLD2D requires positive mesh dimensions");
  TENRYU_ASSERT(static_cast<int>(state.rho.size()) == n_cells,
                "FLD2D requires state cell count to match nr*nz");
  TENRYU_ASSERT(state.x_r.size() == static_cast<std::size_t>((nr + 1) * (nz + 1)),
                "FLD2D requires 2D node_r size");
  TENRYU_ASSERT(state.x_z.size() == static_cast<std::size_t>((nr + 1) * (nz + 1)),
                "FLD2D requires 2D node_z size");
  TENRYU_ASSERT(planck.n_groups() == n_groups,
                "FLD2D requires Planck table group count to match Radiation.groups");
  const bool per_material_matter_update =
      cfg.numerics.materials.per_material_conservation_enabled;
  if (per_material_matter_update) {
    TENRYU_ASSERT(n_mat > 0,
                  "FLD2D per-material matter update requires at least one material");
    TENRYU_ASSERT(state.mass.size() == static_cast<std::size_t>(n_cells),
                  "FLD2D per-material matter update requires mass size n_cells");
    TENRYU_ASSERT(state.mass_per_material.size() == n_cell_mat,
                  "FLD2D per-material matter update requires mass_per_material size n_cells*n_materials");
    TENRYU_ASSERT(state.Ee_per_material.size() == n_cell_mat,
                  "FLD2D per-material matter update requires Ee_per_material size n_cells*n_materials");
  }

  ensure_state_buffers(state, n_cells, n_groups);
  // Fleck-blend reference field: snapshot at the START of THIS solve.
  // (Historic end-of-previous-solve snapshot assumed nothing modifies rad_E
  // between solves; coupling/remap paths that touch rad_E without setting
  // holo_ale_invalidated fed the blend a stale reference, injecting
  // ~(1-f) c sigma (E_old-E_now) per step. The unconditional start-of-solve
  // copy is byte-identical for every config whose rad_E is untouched between
  // solves — including the ALE case previously covered by
  // initialize_rad_E_old_if_needed — and correct under coupling. BUG-14b:
  // same repair as the 1D gamma_r arc, fld_1d_gpu.cu advance.)
  copy_rad_E_to_old(state, n_cells, n_groups);

  const int cell_grid = (n_cells + kBlock - 1) / kBlock;
  if (cell_grid > 0) {
    snapshot_Te_kernel<<<cell_grid, kBlock>>>(
        state.Te.data(), state.fld_Te_old.data(), n_cells);
    cuda_check(cudaGetLastError(), "FLD2D Te snapshot launch failed");
  }

  const int n_rows = n_cells * n_groups;
  const int nnz = n_rows * kCsrEntriesPerRow;
  auto& ws = fld_2d_workspace();
  // Option-C MPI context for the distributed CG + masked diagnostics
  // (design doc mpi_m18c_fld2d_cg_spec.md). Reset on EVERY call: the
  // part/bufs pointers are only valid for the duration of this advance.
  {
    const core::State::LaunchWindow ocw = state.owned_cell_window(n_cells);
    ws.mpi_part = &part;
    ws.mpi_bufs = bufs;
    ws.mpi_n_ranks = part.n_ranks;
    ws.mpi_c_begin = ocw.begin;
    ws.mpi_c_end = ocw.end;
    ws.mpi_n_cells = n_cells;
    ws.mpi_n_groups = n_groups;
  }
  ws.d_row_offsets.resize(sizeof(int) * static_cast<std::size_t>(n_rows + 1));
  ws.d_col_indices.resize(sizeof(int) * static_cast<std::size_t>(nnz));
  ws.d_values.resize(sizeof(double) * static_cast<std::size_t>(nnz));
  ws.d_rhs.resize(sizeof(double) * static_cast<std::size_t>(n_rows));
  ws.d_diag_inv.resize(sizeof(double) * static_cast<std::size_t>(n_rows));
  ws.d_x.resize(sizeof(double) * static_cast<std::size_t>(n_rows));

  state.fld_converged = false;
  state.fld_outer_residual = std::numeric_limits<double>::infinity();
  state.fld_outer_iterations = 0;
  reset_fld_step_diagnostics(state);

  const auto& fld = cfg.radiation.multigroup_diffusion;
  const int outer_r_bc = fld_boundary_code(fld.boundary.outer_r);
  const int z_bottom_bc = fld_boundary_code(fld.boundary.z_bottom);
  const int z_top_bc = fld_boundary_code(fld.boundary.z_top);
  const int fld_limiter_id = limiter_id(fld.flux_limiter);
  const int state_supply_boundary_policy =
      state_supply_boundary_policy_id(fld.state_supply_boundary_policy);
  const double T_supply_z_bottom_eV =
      (z_bottom_bc == kFldBcStateSupply)
          ? cfg.numerics.hydro.boundary_2d.z_bottom_cfg.supply_T_eV
          : 0.0;
  const double T_supply_z_top_eV =
      (z_top_bc == kFldBcStateSupply)
          ? cfg.numerics.hydro.boundary_2d.z_top_cfg.supply_T_eV
          : 0.0;
  const double rho_supply_z_bottom =
      (z_bottom_bc == kFldBcStateSupply)
          ? cfg.numerics.hydro.boundary_2d.z_bottom_cfg.supply_rho_g_per_cc
          : 0.0;
  const double rho_supply_z_top =
      (z_top_bc == kFldBcStateSupply)
          ? cfg.numerics.hydro.boundary_2d.z_top_cfg.supply_rho_g_per_cc
          : 0.0;
  double marshak_flux = std::max(fld.marshak.flux_erg_per_cm2_s, 0.0);
  if (fld.marshak.flux_pulse_duration_s >= 0.0 &&
      state.t >= fld.marshak.flux_pulse_duration_s) {
    marshak_flux = 0.0;
  }
  // Indirect-drive Tr(t) marshak z faces (spec docs/design/
  // 2d_tr_drive_port_spec.md): resolve once per radiation call (outer-
  // iteration invariant); per-group F_inc = 0.25*c*a_eV*Tr^4*b_g with the
  // 1D-marshak Planck weighting (grey bypass b=1). Builder validation
  // guarantees flux_erg_per_cm2_s == 0 on this route, so the legacy in-
  // kernel term adds exactly +0.0.
  double marshak_tr_bottom_eV = 0.0;
  double marshak_tr_top_eV = 0.0;
  if (z_bottom_bc == kFldBcMarshak) {
    marshak_tr_bottom_eV =
        resolve_marshak_tr_face_eV(cfg, state, "bottom_z", "z_bottom");
  }
  if (z_top_bc == kFldBcMarshak) {
    marshak_tr_top_eV =
        resolve_marshak_tr_face_eV(cfg, state, "top_z", "z_top");
  }
  const bool marshak_tr_drive =
      (marshak_tr_bottom_eV > 0.0) || (marshak_tr_top_eV > 0.0);
  double* d_marshak_tr_finc = nullptr;
  double marshak_tr_finc_sum_bottom = 0.0;
  double marshak_tr_finc_sum_top = 0.0;
  if (marshak_tr_drive) {
    std::vector<double> finc_host(2 * static_cast<std::size_t>(n_groups), 0.0);
    const auto fill_face = [&](const int face, const double t_r_eV,
                               double& sum_out) {
      if (!(t_r_eV > 0.0)) {
        return;
      }
      const double t4 = t_r_eV * t_r_eV * t_r_eV * t_r_eV;
      for (int g = 0; g < n_groups; ++g) {
        const double b =
            (n_groups == 1)
                ? 1.0
                : std::max(planck.interpolate_b_host(g, t_r_eV), 0.0);
        const double f_inc =
            0.25 * core::constants::c_light * core::constants::a_eV * t4 * b;
        finc_host[static_cast<std::size_t>(face) * n_groups + g] = f_inc;
        sum_out += f_inc;
      }
    };
    fill_face(0, marshak_tr_bottom_eV, marshak_tr_finc_sum_bottom);
    fill_face(1, marshak_tr_top_eV, marshak_tr_finc_sum_top);
    d_marshak_tr_finc = static_cast<double*>(core::device_scratch_acquire(
        "fld2d:marshak_tr_finc",
        2 * static_cast<std::size_t>(n_groups) * sizeof(double)));
    cuda_check(cudaMemcpy(d_marshak_tr_finc, finc_host.data(),
                          2 * static_cast<std::size_t>(n_groups) *
                              sizeof(double),
                          cudaMemcpyHostToDevice),
               "FLD2D marshak Tr finc upload failed");
  }
  const int max_iter = std::max(fld.max_outer_iterations, 1);
  const bool use_fleck = use_fld_fleck(mat);
  const bool use_nlte_sigma_pe =
      mat.opacity_model == "table_nlte" || mat.opacity_model == "tmat";
  const int total_grid = (n_rows + kBlock - 1) / kBlock;
  const bool verbose_fld_timing = cfg.main.verbosity == "verbose";
  const bool collect_fld_newton_diagnostics =
      verbose_fld_timing || cfg.numerics.diagnostics.production_audit.enabled;
  const bool fld_substage_audit_enabled =
      fld.diagnostic_radial_fourier_substage_enabled;
  fld_substage_audit_batch().clear();
  const FldTraceConfig& trace = fld_trace_config();
  const bool fld_trace_active =
      fld_trace_step_active(trace, verbose_fld_timing, state.step, n_cells, n_groups);
  const int trace_row =
      fld_trace_active ? trace.group * n_cells + trace.cell : -1;
  std::vector<FldTraceRecord> host_trace_records;
  if (fld_trace_active) {
    state.fld_trace_records.reset(
        (sizeof(FldTraceRecord) * static_cast<std::size_t>(max_iter) +
         sizeof(double) - 1U) /
        sizeof(double));
    host_trace_records.resize(static_cast<std::size_t>(max_iter));
  }
  if (verbose_fld_timing) {
    initialize_rogue_records(ws);
  }
  int total_cg_iters_this_step = 0;
  int max_cg_iters_this_step = 0;
  int n_early_return = 0;
  const auto should_log_cap_exit_warning = [](const std::uint64_t count) {
    return count <= 4U || ((count & (count - 1U)) == 0U);
  };
  const auto handle_cg_cap_exit =
      [&](const char* solver_name,
          const CgDiagnostics& diagnostics,
          const int outer_iter) {
        if (!diagnostics.cap_exit_unconverged) {
          return;
        }
        ++state.cg_cap_exit_unconverged;
        std::ostringstream oss;
        oss << std::setprecision(17)
            << "FLD2D " << solver_name
            << " cap exit unconverged: step=" << state.step
            << " outer=" << outer_iter
            << " group=" << diagnostics.cap_exit_true_resid_group
            << " true_rel=" << diagnostics.cap_exit_true_resid_rel
            << " true_l2_abs=" << diagnostics.cap_exit_true_resid_l2_abs
            << " tol=" << fld.cg_inner_tol
            << " iter_count=" << diagnostics.iters_executed
            << " policy=" << fld.cap_exit_policy
            << " cg_cap_exit_unconverged="
            << state.cg_cap_exit_unconverged;
        const std::string message = oss.str();
        if (fld.cap_exit_policy == "fail") {
          core::log_fatal(message);
          TENRYU_ASSERT(false, message);
        }
        if (should_log_cap_exit_warning(state.cg_cap_exit_unconverged)) {
          core::log_warning(message);
        }
      };
  compute_cell_centers(state, nr, nz);
  const bool aa_enabled = fld.outer_accel == "anderson" && n_cells > 0;
  const int aa_m = std::min(std::max(fld.anderson_m, 1), 4);
  const double aa_beta = fld.anderson_beta;
  const std::size_t aa_bytes = sizeof(double) * static_cast<std::size_t>(n_cells);
  int aa_count = 0;  // number of (u, f) pairs recorded so far this step
  if (aa_enabled) {
    for (int s = 0; s < aa_m + 1; ++s) {
      ws.d_aa_u[s].resize(aa_bytes);
      ws.d_aa_f[s].resize(aa_bytes);
    }
  }
  // Option-C ghost refreshes (design doc mpi_m18c_fld2d_cg_spec.md): the
  // owned-row assembly and matter coupling read Te/rad_E at ghost cells;
  // rad_E/rad_E_old are cell-major group-minor, so their strips ride the
  // scaled contiguous-strip exchange (r-slab).
  const auto exchange_fld_scalar_ghosts = [&](std::initializer_list<double*> fields) {
    if (part.n_ranks <= 1 || bufs == nullptr) {
      return;
    }
    double* ptrs[4];
    int n_ptrs = 0;
    for (double* q : fields) {
      if (q != nullptr) {
        ptrs[n_ptrs++] = q;
      }
    }
    if (n_ptrs > 0) {
      parallel::exchange_cell_fields(part, *bufs, ptrs, n_ptrs, n_cells,
                                     nullptr, 4);
    }
  };
  const auto exchange_fld_group_ghosts = [&](double* field) {
    if (part.n_ranks <= 1 || bufs == nullptr || field == nullptr) {
      return;
    }
    parallel::exchange_cell_strips_scaled(part, *bufs, field, n_groups, 5);
  };
  exchange_fld_group_ghosts(state.rad_E_old.data());
  for (int iter = 0; iter < max_iter; ++iter) {
    const int audit_outer_iter = iter + 1;
    const int aa_slot = aa_enabled ? (iter % (aa_m + 1)) : 0;
    if (aa_enabled) {
      cuda_check(cudaMemcpy(ws.d_aa_u[aa_slot].ptr,
                            state.Te.data(),
                            aa_bytes,
                            cudaMemcpyDeviceToDevice),
                 "FLD2D AA snapshot u_k failed");
    }
    if (fld_substage_audit_enabled) {
      fld_substage_audit_batch().clear();
      append_fld_substage_audit_field(
          ws,
          state.rad_E_old.data(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::EradBeforeFldEntry,
          diagnostics::FldSubstageAuditFieldId::Erad,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
    }
    FldTraceRecord* trace_record = nullptr;
    if (fld_trace_active) {
      trace_record =
          reinterpret_cast<FldTraceRecord*>(state.fld_trace_records.data()) + iter;
      cuda_check(cudaMemset(trace_record, 0, sizeof(FldTraceRecord)),
                 "FLD2D zero trace record failed");
    }
    exchange_fld_scalar_ghosts({state.Te.data()});
    exchange_fld_group_ghosts(state.rad_E.data());
    evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells, n_groups, dt);
    if (fld_substage_audit_enabled) {
      append_fld_substage_audit_field(
          ws,
          state.rho.data(),
          diagnostics::FldSubstageAuditFieldLayout::CellScalar,
          diagnostics::FldSubstageAuditSubstageId::OpacityBeforeDBuild,
          diagnostics::FldSubstageAuditFieldId::Rho,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
      append_fld_substage_audit_field(
          ws,
          state.fld_sigma_R.data(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::OpacityBeforeDBuild,
          diagnostics::FldSubstageAuditFieldId::SigmaR,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
    }
    compute_diffusion_coefficients(state,
                                   cfg,
                                   nr,
                                   nz,
                                   n_groups,
                                   state.fld_cell_rc.data(),
                                   state.fld_cell_zc.data());
    if (fld_substage_audit_enabled) {
      append_fld_substage_audit_field(
          ws,
          state.fld_D_cell.data(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::DCellAfterDBuild,
          diagnostics::FldSubstageAuditFieldId::DCell,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
      compute_fld_substage_boundary_audit_fields(ws,
                                                 state,
                                                 nr,
                                                 nz,
                                                 n_groups,
                                                 dt,
                                                 z_top_bc,
                                                 T_supply_z_top_eV,
                                                 state_supply_boundary_policy,
                                                 fld_limiter_id,
                                                 rho_supply_z_top);
      append_fld_substage_audit_field(
          ws,
          ws.d_audit_boundary_coeff.as<double>(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::StateSupplyBoundaryCoeff,
          diagnostics::FldSubstageAuditFieldId::StateSupplyBoundaryCoeff,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
      append_fld_substage_audit_field(
          ws,
          ws.d_audit_boundary_dt_coeff.as<double>(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::StateSupplyDtCoeff,
          diagnostics::FldSubstageAuditFieldId::StateSupplyDtCoeff,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
      append_fld_substage_audit_field(
          ws,
          ws.d_audit_boundary_source.as<double>(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::StateSupplyBoundarySource,
          diagnostics::FldSubstageAuditFieldId::StateSupplyBoundarySource,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
      append_fld_substage_audit_field(
          ws,
          ws.d_audit_boundary_diag.as<double>(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::BoundaryDiagContribution,
          diagnostics::FldSubstageAuditFieldId::BoundaryDiagContribution,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
    }
    if (verbose_fld_timing) {
      update_dcell_diagnostics(state, ws, n_rows);
      ws.d_diag_ints.resize(4U * sizeof(int));
      cuda_check(cudaMemset(ws.d_diag_ints.ptr, 0, 4U * sizeof(int)),
                 "FLD2D zero face diagnostic counts failed");
    }
    int* face_counts =
        verbose_fld_timing ? ws.d_diag_ints.as<int>() : nullptr;
    assemble_fld_2d_csr_kernel<<<(n_rows + 1 + kBlock - 1) / kBlock, kBlock>>>(
        state.x_r.data(),
        state.x_z.data(),
        state.vol.data(),
        state.fld_sigma_a.data(),
        use_fleck ? state.fld_sigma_a.data() : nullptr,
        use_fleck ? state.fld_nlte_f_work.data() : nullptr,
        state.fld_eta.data(),
        state.rad_E_old.data(),
        state.fld_D_cell.data(),
        state.Te.data(),
        state.fld_cell_rc.data(),
        state.fld_cell_zc.data(),
        ws.d_row_offsets.as<int>(),
        ws.d_col_indices.as<int>(),
        ws.d_values.as<double>(),
        ws.d_rhs.as<double>(),
        ws.d_diag_inv.as<double>(),
        nr,
        nz,
        n_groups,
        dt,
        outer_r_bc,
        z_bottom_bc,
        z_top_bc,
        marshak_flux,
        T_supply_z_bottom_eV,
        T_supply_z_top_eV,
        face_counts,
        face_counts != nullptr ? face_counts + 1 : nullptr,
        face_counts != nullptr ? face_counts + 2 : nullptr,
        face_counts != nullptr ? face_counts + 3 : nullptr,
        trace_row,
        trace_record,
        state.rho.data(),
        state.fld_sigma_R.data(),
        state_supply_boundary_policy,
        fld_limiter_id,
        rho_supply_z_bottom,
        rho_supply_z_top);
    cuda_check(cudaGetLastError(), "FLD2D CSR assembly launch failed");
    if (marshak_tr_drive) {
      const int tr_threads = 2 * nr * n_groups;
      const int tr_grid = (tr_threads + kBlock - 1) / kBlock;
      add_marshak_tr_rhs_2d_kernel<<<tr_grid, kBlock>>>(
          ws.d_rhs.as<double>(), d_marshak_tr_finc, state.x_r.data(),
          state.x_z.data(), state.vol.data(), nr, nz, n_groups, dt,
          (marshak_tr_bottom_eV > 0.0) ? 1 : 0,
          (marshak_tr_top_eV > 0.0) ? 1 : 0);
      cuda_check(cudaGetLastError(),
                 "FLD2D marshak Tr rhs post-pass launch failed");
    }
    if (fld_substage_audit_enabled) {
      append_fld_substage_audit_field(
          ws,
          ws.d_rhs.as<double>(),
          diagnostics::FldSubstageAuditFieldLayout::GroupMajor,
          diagnostics::FldSubstageAuditSubstageId::RhsAfterAssembly,
          diagnostics::FldSubstageAuditFieldId::Rhs,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
    }
    if (verbose_fld_timing) {
      int face_diagnostics[4] = {0, 0, 0, 0};
      cuda_check(cudaMemcpy(face_diagnostics,
                            ws.d_diag_ints.as<int>(),
                            sizeof(face_diagnostics),
                            cudaMemcpyDeviceToHost),
                 "FLD2D copy face diagnostic counts failed");
      state.fld_face_skip_D_count_step += face_diagnostics[0];
      state.fld_face_skip_nonfinite_count_step += face_diagnostics[1];
      state.fld_face_skip_dist_count_step += face_diagnostics[2];
      state.fld_diag_fallback_count_step += face_diagnostics[3];
      update_csr_matrix_diagnostics(state, ws, n_rows);
      if (iter == 0) {
        update_pair_symmetry_diagnostics(state, ws, n_rows);
      }
    }
    const double* x_init_src =
        (iter == 0) ? state.rad_E_old.data() : state.rad_E.data();
    init_solution_kernel<<<total_grid, kBlock>>>(
        x_init_src, ws.d_x.as<double>(), n_cells, n_groups);
    cuda_check(cudaGetLastError(), "FLD2D solution init launch failed");
    double audit_initial_residual_l2 = 0.0;
    if (fld_substage_audit_enabled) {
      const PostPublishDiagnostics initial_residual =
          compute_post_publish_solver_diagnostics_from_csr(
              ws.d_row_offsets.as<int>(),
              ws.d_col_indices.as<int>(),
              ws.d_values.as<double>(),
              ws.d_rhs.as<double>(),
              ws.d_x.as<double>(),
              n_rows,
              nnz);
      audit_initial_residual_l2 = initial_residual.true_residual_l2_abs;
    }

    int cg_iters_this_call = -1;
    CgDiagnostics cg_diagnostics;
    if (fld.linear_solver_2d == "amgx_cg" && amgx_solver_available()) {
      AmgxSolveRequest request{};
      request.n_rows = n_rows;
      request.nnz = nnz;
      request.row_offsets = ws.d_row_offsets.as<int>();
      request.col_indices = ws.d_col_indices.as<int>();
      request.values = ws.d_values.as<double>();
      request.rhs = ws.d_rhs.as<double>();
      request.solution = ws.d_x.as<double>();
      request.preset = fld.amgx_config.preset;
      solve_with_amgx(request);
      total_cg_iters_this_step = -1;
      max_cg_iters_this_step = -1;
    } else if (fld.linear_solver_2d == "cusparse_cg_zline") {
      cg_iters_this_call =
          solve_cusparse_cg_jacobi(ws.d_row_offsets.as<int>(),
                                   ws.d_col_indices.as<int>(),
                                   ws.d_values.as<double>(),
                                   ws.d_rhs.as<double>(),
                                   ws.d_diag_inv.as<double>(),
                                   ws.d_x.as<double>(),
                                   n_rows,
                                   nnz,
                                   fld.cg_inner_tol,
                                   fld.cg_tol_norm == "rhs",
                                   fld.cg_max_iter,
                                   true,
                                   &cg_diagnostics,
                                   FldPrecondMode::ZLine,
                                   nr,
                                   nz,
                                   n_groups);
      handle_cg_cap_exit("cuSPARSE CG/z-line",
                         cg_diagnostics,
                         audit_outer_iter);
    } else if (fld.linear_solver_2d == "cusparse_cg_rgmg") {
      cg_iters_this_call =
          solve_cusparse_cg_jacobi(ws.d_row_offsets.as<int>(),
                                   ws.d_col_indices.as<int>(),
                                   ws.d_values.as<double>(),
                                   ws.d_rhs.as<double>(),
                                   ws.d_diag_inv.as<double>(),
                                   ws.d_x.as<double>(),
                                   n_rows,
                                   nnz,
                                   fld.cg_inner_tol,
                                   fld.cg_tol_norm == "rhs",
                                   fld.cg_max_iter,
                                   true,
                                   &cg_diagnostics,
                                   FldPrecondMode::RGmg,
                                   nr,
                                   nz,
                                   n_groups,
                                   fld.rgmg_smoother_omega);
      handle_cg_cap_exit("cuSPARSE CG/RGMG",
                         cg_diagnostics,
                         audit_outer_iter);
    } else {
      if (fld.linear_solver_2d == "amgx_cg") {
        throw core::namelist::ConfigError(
            "linear_solver_2d=amgx_cg requested but AmgX is not linked in this "
            "build; choose cusparse_cg_zline / cusparse_cg_rgmg / jacobi "
            "explicitly");
      }
      if (fld.linear_solver_2d == "auto") {
        static bool auto_unresolved_warned = false;
        if (!auto_unresolved_warned) {
          core::log_warning(
              "FLD2D linear_solver_2d=\"auto\" reached the solver unresolved "
              "(config not built via the namelist validate() path); using the "
              "cuSPARSE CG/Jacobi fallback");
          auto_unresolved_warned = true;
        }
      }
      cg_iters_this_call =
          solve_cusparse_cg_jacobi(ws.d_row_offsets.as<int>(),
                                   ws.d_col_indices.as<int>(),
                                   ws.d_values.as<double>(),
                                   ws.d_rhs.as<double>(),
                                   ws.d_diag_inv.as<double>(),
                                   ws.d_x.as<double>(),
                                   n_rows,
                                   nnz,
                                   fld.cg_inner_tol,
                                   fld.cg_tol_norm == "rhs",
                                   fld.cg_max_iter,
                                   true,
                                   &cg_diagnostics,
                                   FldPrecondMode::Diagonal,
                                   nr,
                                   nz,
                                   n_groups);
      handle_cg_cap_exit("cuSPARSE CG/Jacobi",
                         cg_diagnostics,
                         audit_outer_iter);
    }
    if (cg_iters_this_call >= 0 && total_cg_iters_this_step >= 0) {
      total_cg_iters_this_step += cg_iters_this_call;
      max_cg_iters_this_step =
          std::max(max_cg_iters_this_step, cg_iters_this_call);
      if (cg_iters_this_call == 0) {
        ++n_early_return;
      }
    }
    if (verbose_fld_timing && cg_iters_this_call >= 0) {
      if (std::isfinite(cg_diagnostics.min_pAp_value)) {
        state.fld_cg_pAp_min_step =
            std::min(state.fld_cg_pAp_min_step, cg_diagnostics.min_pAp_value);
      }
      state.fld_cg_nonpos_pAp_count_step +=
          cg_diagnostics.count_nonpos_pAp;
      state.fld_cg_nonfinite_count_step +=
          cg_diagnostics.count_nonfinite_alpha_beta_rz;
      state.fld_cg_recurrent_resid_last_check_max_step =
          std::max(state.fld_cg_recurrent_resid_last_check_max_step,
                   cg_diagnostics.recurrent_resid_last_check);
    }
    if (verbose_fld_timing && total_grid > 0) {
      rogue_x_raw_kernel<<<total_grid, kBlock>>>(
          ws.d_x.as<double>(),
          ws.d_rogue_records.as<RogueRecord>(),
          n_cells,
          n_groups);
      cuda_check(cudaGetLastError(), "FLD2D rogue x_raw diagnostics launch failed");
    }
    if (verbose_fld_timing && cg_iters_this_call >= 0) {
      const PostPublishDiagnostics raw_diagnostics =
          compute_post_publish_solver_diagnostics(ws.d_rhs.as<double>(),
                                                  ws.d_x.as<double>(),
                                                  n_rows);
      state.fld_cg_true_residual_l2_rel_RAW_step =
          std::max(state.fld_cg_true_residual_l2_rel_RAW_step,
                   raw_diagnostics.true_residual_l2_rel);
      state.fld_cg_true_residual_max_RAW_step =
          std::max(state.fld_cg_true_residual_max_RAW_step,
                   raw_diagnostics.true_residual_max);
      state.fld_E_solver_RAW_step =
          std::max(state.fld_E_solver_RAW_step, raw_diagnostics.E_solver_abs);
    }
    if (fld_trace_active) {
      fld_trace_post_solve_kernel<<<1, 1>>>(
          ws.d_row_offsets.as<int>(),
          ws.d_col_indices.as<int>(),
          ws.d_values.as<double>(),
          ws.d_rhs.as<double>(),
          ws.d_x.as<double>(),
          state.vol.data(),
          n_groups,
          n_cells,
          nz,
          trace_row,
          trace.cell,
          trace.group,
          trace_record);
      cuda_check(cudaGetLastError(), "FLD2D trace post-solve launch failed");
    }
    ws.d_diag_scalars.resize(3U * sizeof(double));
    cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, 3U * sizeof(double)),
               "FLD2D zero publish diagnostics failed");
    const core::State::LaunchWindow pubw = state.owned_cell_window(n_cells);
    const int pub_total = pubw.count() * n_groups;
    const int pub_grid = (pub_total + kBlock - 1) / kBlock;
    publish_with_projection_kernel<<<pub_grid, kBlock>>>(
        ws.d_x.as<double>(),
        state.rad_E.data(),
        state.vol.data(),
        ws.d_diag_scalars.as<double>(),
        ws.d_diag_scalars.as<double>() + 1,
        ws.d_diag_scalars.as<double>() + 2,
        pubw.begin,
        pubw.end,
        n_cells,
        n_groups);
    cuda_check(cudaGetLastError(), "FLD2D publish solution launch failed");
    double publish_diagnostics[3] = {0.0, 0.0, 0.0};
    cuda_check(cudaMemcpy(publish_diagnostics,
                          ws.d_diag_scalars.as<double>(),
                          sizeof(publish_diagnostics),
                          cudaMemcpyDeviceToHost),
               "FLD2D copy publish diagnostics failed");
    state.fld_clamp_hits_step +=
        static_cast<int>(
            fmax(finite_or_zero(publish_diagnostics[0]), 0.0) + 0.5);
    state.fld_clamp_energy_delta_step += finite_or_zero(publish_diagnostics[1]);
    state.fld_min_x_raw_step =
        std::min(state.fld_min_x_raw_step, finite_or_zero(publish_diagnostics[2]));
    if (verbose_fld_timing && total_grid > 0) {
      rogue_rad_E_kernel<<<total_grid, kBlock>>>(
          state.rad_E.data(),
          ws.d_rogue_records.as<RogueRecord>(),
          n_cells,
          n_groups);
      cuda_check(cudaGetLastError(), "FLD2D rogue rad_E diagnostics launch failed");
    }
    PostPublishDiagnostics post_publish_diagnostics;
    bool have_post_publish_diagnostics = false;
    if (cg_iters_this_call >= 0) {
      post_publish_diagnostics =
          compute_post_publish_solver_diagnostics(ws.d_rhs.as<double>(),
                                                  ws.d_x.as<double>(),
                                                  n_rows);
      have_post_publish_diagnostics = true;
      state.fld_cg_true_residual_l2_rel_step =
          std::max(state.fld_cg_true_residual_l2_rel_step,
                   post_publish_diagnostics.true_residual_l2_rel);
      state.fld_cg_true_residual_max_step =
          std::max(state.fld_cg_true_residual_max_step,
                   post_publish_diagnostics.true_residual_max);
      state.fld_E_solver_step =
          std::max(state.fld_E_solver_step,
                   post_publish_diagnostics.E_solver_abs);
      if (verbose_fld_timing && total_grid > 0) {
        rogue_residual_kernel<<<total_grid, kBlock>>>(
            ws.d_Ap.as<double>(),
            ws.d_rogue_records.as<RogueRecord>(),
            n_cells,
            n_groups);
        cuda_check(cudaGetLastError(),
                   "FLD2D rogue residual diagnostics launch failed");
      }
    } else if (fld_substage_audit_enabled) {
      post_publish_diagnostics =
          compute_post_publish_solver_diagnostics_from_csr(
              ws.d_row_offsets.as<int>(),
              ws.d_col_indices.as<int>(),
              ws.d_values.as<double>(),
              ws.d_rhs.as<double>(),
              ws.d_x.as<double>(),
              n_rows,
              nnz);
      have_post_publish_diagnostics = true;
    }
    if (fld_substage_audit_enabled && have_post_publish_diagnostics) {
      const double audit_residual_l2_rel =
          (audit_initial_residual_l2 > 0.0)
              ? (post_publish_diagnostics.true_residual_l2_abs /
                 audit_initial_residual_l2)
              : 0.0;
      append_fld_substage_audit_field(
          ws,
          ws.d_Ap.as<double>(),
          diagnostics::FldSubstageAuditFieldLayout::GroupMajor,
          diagnostics::FldSubstageAuditSubstageId::CgTrueResidual,
          diagnostics::FldSubstageAuditFieldId::CgTrueResidual,
          nr,
          nz,
          n_groups,
          audit_outer_iter,
          audit_residual_l2_rel,
          post_publish_diagnostics.true_residual_max);
    }
    if (fld_substage_audit_enabled) {
      append_fld_substage_audit_field(
          ws,
          state.rad_E.data(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::EradAfterCgSolve,
          diagnostics::FldSubstageAuditFieldId::Erad,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
    }

    if (n_cells > 0) {
      const std::size_t shared_doubles = 3U * static_cast<std::size_t>(n_groups);
      const std::size_t shared_bytes = shared_doubles * sizeof(double);
      TENRYU_ASSERT(shared_bytes <= 32U * 1024U,
                    "FLD2D update_matter shared memory exceeds 32 KiB safety cap; "
                    "reduce n_groups");
      int* newton_counts = nullptr;
      double* newton_residuals = nullptr;
      ws.d_diag_ints.resize(4U * sizeof(int));
      cuda_check(cudaMemset(ws.d_diag_ints.ptr, 0, 4U * sizeof(int)),
                 "FLD2D zero Newton diagnostic counts failed");
      newton_counts = ws.d_diag_ints.as<int>();
      if (collect_fld_newton_diagnostics) {
        ws.d_diag_scalars.resize(3U * sizeof(double));
        cuda_check(cudaMemset(ws.d_diag_scalars.ptr, 0, 3U * sizeof(double)),
                   "FLD2D zero Newton diagnostic residuals failed");
        newton_residuals = ws.d_diag_scalars.as<double>();
      }
      const core::State::LaunchWindow mw = state.owned_cell_window(n_cells);
      update_matter_kernel<<<mw.count(), 32, shared_bytes>>>(
          state.rho.data(),
          state.vol.data(),
          per_material_matter_update ? state.mass.data() : nullptr,
          state.zbar.data(),
          (state.cv_e.size() == static_cast<std::size_t>(n_cells)) ? state.cv_e.data()
                                                                   : nullptr,
          state.fld_Te_old.data(),
          state.fld_sigma_a.data(),
          use_nlte_sigma_pe ? state.fld_sigma_pe.data() : state.fld_sigma_a.data(),
          state.fld_eta.data(),
          use_fleck ? state.fld_nlte_f_work.data() : nullptr,
          state.rad_E_old.data(),
          state.rad_E.data(),
          planck.device_view(),
          (mat.hydro_eos_backend != "exact_ideal_gas")
              ? fld_electron_eos_device_view(mat.eos_tables.get())
              : materials::DeviceEOSTableView{},
          state.Te.data(),
          state.ee.data(),
          state.Pe.data(),
          per_material_matter_update ? state.Ee_per_material.data() : nullptr,
          per_material_matter_update ? state.mass_per_material.data() : nullptr,
          state.rad_dep.data(),
          state.rad_emit.data(),
          state.fld_delta_T.data(),
          state.fld_fleck.data(),
          mw.begin,
          n_cells,
          n_groups,
          per_material_matter_update ? n_mat : 0,
          dt,
          std::max(mat.A, 1.0e-12),
          std::max(mat.ideal_gas_gamma, 1.0 + 1.0e-12),
          mat.cv_e_override,
          cfg.numerics.floors.Te,
          state.cv_e.size() == static_cast<std::size_t>(n_cells) ? 1 : 0,
          std::max(fld.max_outer_iterations, 1),
          fld.outer_tol,
          true,
          newton_counts,
          newton_counts != nullptr ? newton_counts + 1 : nullptr,
          newton_counts != nullptr ? newton_counts + 2 : nullptr,
          newton_counts != nullptr ? newton_counts + 3 : nullptr,
          newton_residuals,
          newton_residuals != nullptr ? newton_residuals + 1 : nullptr,
          newton_residuals != nullptr ? newton_residuals + 2 : nullptr);
      cuda_check(cudaGetLastError(), "FLD2D matter update launch failed");
      if (per_material_matter_update &&
          cfg.numerics.materials.lazy_cache_te_m_enabled) {
        state.Te_per_material_valid.assign(n_cell_mat, static_cast<std::uint8_t>(0));
      }
      int newton_count_values[4] = {0, 0, 0, 0};
      cuda_check(cudaMemcpy(newton_count_values,
                            ws.d_diag_ints.as<int>(),
                            sizeof(newton_count_values),
                            cudaMemcpyDeviceToHost),
                 "FLD2D copy Newton diagnostic counts failed");
      state.fld_newton_converged_count_step += newton_count_values[0];
      state.fld_newton_invalid_count_step += newton_count_values[1];
      state.fld_newton_cap_hit_count_step += newton_count_values[2];
      state.fld_newton_reject_count_step += newton_count_values[3];
      const int reject_count = newton_count_values[3];
      const int invalid_count = newton_count_values[1];
      const int cap_count = newton_count_values[2];
      if (reject_count != 0 || invalid_count != 0 || cap_count != 0) {
        static bool warned_newton_visibility = false;
        static int last_reject_count = -1;
        static int last_invalid_count = -1;
        static int last_cap_count = -1;
        if (!warned_newton_visibility ||
            reject_count != last_reject_count ||
            invalid_count != last_invalid_count ||
            cap_count != last_cap_count) {
          core::log_warning(
              "FLD2D matter Newton reject/invalid/cap counts: r=" +
              std::to_string(reject_count) + " i=" +
              std::to_string(invalid_count) + " c=" +
              std::to_string(cap_count) + " (BUG-14 visibility)");
          warned_newton_visibility = true;
          last_reject_count = reject_count;
          last_invalid_count = invalid_count;
          last_cap_count = cap_count;
        }
      }
      if (collect_fld_newton_diagnostics) {
        double newton_resid_values[3] = {0.0, 0.0, 0.0};
        cuda_check(cudaMemcpy(newton_resid_values,
                              ws.d_diag_scalars.as<double>(),
                              sizeof(newton_resid_values),
                              cudaMemcpyDeviceToHost),
                   "FLD2D copy Newton diagnostic residuals failed");
        state.fld_newton_resid_abs_max_step =
            std::max(state.fld_newton_resid_abs_max_step,
                     finite_or_zero(newton_resid_values[0]));
        state.fld_newton_resid_rel_max_step =
            std::max(state.fld_newton_resid_rel_max_step,
                     finite_or_zero(newton_resid_values[1]));
        state.fld_newton_reject_resid_rel_max_step =
            std::max(state.fld_newton_reject_resid_rel_max_step,
                     finite_or_zero(newton_resid_values[2]));
      }
    }
    if (fld_substage_audit_enabled) {
      append_fld_substage_audit_field(
          ws,
          state.rad_E.data(),
          diagnostics::FldSubstageAuditFieldLayout::CellMajorGroup,
          diagnostics::FldSubstageAuditSubstageId::EradAfterNewtonSource,
          diagnostics::FldSubstageAuditFieldId::Erad,
          nr,
          nz,
          n_groups,
          audit_outer_iter);
    }
    double residual = reduce_max_delta_T(state, n_cells);
    if (part.n_ranks > 1) {
      const parallel::Reduction reduction(part.n_ranks);
      residual = reduction.allreduce_max(residual);
    }
    state.fld_outer_residual = residual;
    state.fld_outer_iterations = iter + 1;
    if (residual < fld.outer_tol) {
      state.fld_converged = true;
      break;
    }
    if (aa_enabled && iter + 1 < max_iter) {
      const int grid_aa = (n_cells + kBlock - 1) / kBlock;
      aa_diff_kernel<<<grid_aa, kBlock>>>(state.Te.data(),
                                          ws.d_aa_u[aa_slot].as<double>(),
                                          ws.d_aa_f[aa_slot].as<double>(),
                                          n_cells);
      cuda_check(cudaGetLastError(), "FLD2D AA residual diff launch failed");
      ++aa_count;
      const int p = std::min(aa_count - 1, aa_m);
      bool mixed = false;
      if (p >= 1) {
        // Ring order: oldest-to-newest pair indices for the last p+1 records.
        int order[5];
        for (int j = 0; j <= p; ++j) {
          order[j] = (iter - p + j) % (aa_m + 1);
        }
        // Option C: gamma (and the mix/no-mix decision) must be identical
        // on every rank, so the Gram/rhs inner products are masked to the
        // owned cells (contiguous under the r-slab) and Allreduced.
        const core::State::LaunchWindow aa_cw =
            state.owned_cell_window(n_cells);
        const auto aa_dot = [&](const double* va, const double* vb) {
          if (part.n_ranks > 1) {
            const double local =
                device_dot(va + aa_cw.begin, vb + aa_cw.begin, aa_cw.count(),
                           ws.d_scalar);
            const parallel::Reduction reduction(part.n_ranks);
            return reduction.allreduce_sum(local);
          }
          return device_dot(va, vb, n_cells, ws.d_scalar);
        };
        double gram[4][4];
        double rhs_v[4];
        bool finite_ok = true;
        for (int a = 0; a < p; ++a) {
          const double* fa1 = ws.d_aa_f[order[a + 1]].as<double>();
          const double* fa0 = ws.d_aa_f[order[a]].as<double>();
          // dF_a . f_k  and  dF_a . dF_b via expansion of device dots:
          // dot(dF_a, x) = dot(f_{a+1}, x) - dot(f_a, x).
          rhs_v[a] =
              aa_dot(fa1, ws.d_aa_f[aa_slot].as<double>()) -
              aa_dot(fa0, ws.d_aa_f[aa_slot].as<double>());
          for (int b = 0; b <= a; ++b) {
            const double* fb1 = ws.d_aa_f[order[b + 1]].as<double>();
            const double* fb0 = ws.d_aa_f[order[b]].as<double>();
            const double dot_ab =
                aa_dot(fa1, fb1) -
                aa_dot(fa1, fb0) -
                aa_dot(fa0, fb1) +
                aa_dot(fa0, fb0);
            gram[a][b] = dot_ab;
            gram[b][a] = dot_ab;
          }
        }
        double trace = 0.0;
        for (int a = 0; a < p; ++a) {
          trace += gram[a][a];
          if (!std::isfinite(gram[a][a]) || !std::isfinite(rhs_v[a])) {
            finite_ok = false;
          }
        }
        if (finite_ok && trace > 0.0) {
          const double lambda = 1.0e-12 * trace / static_cast<double>(p);
          for (int a = 0; a < p; ++a) {
            gram[a][a] += lambda;
          }
          // Cholesky solve gram * gamma = rhs_v (p <= 4).
          double L[4][4] = {};
          bool chol_ok = true;
          for (int a = 0; a < p && chol_ok; ++a) {
            double diag = gram[a][a];
            for (int b = 0; b < a; ++b) {
              diag -= L[a][b] * L[a][b];
            }
            if (!(diag > 0.0)) {
              chol_ok = false;
              break;
            }
            L[a][a] = std::sqrt(diag);
            for (int r2 = a + 1; r2 < p; ++r2) {
              double v2 = gram[r2][a];
              for (int b = 0; b < a; ++b) {
                v2 -= L[r2][b] * L[a][b];
              }
              L[r2][a] = v2 / L[a][a];
            }
          }
          if (chol_ok) {
            double y[4] = {};
            for (int a = 0; a < p; ++a) {
              double v2 = rhs_v[a];
              for (int b = 0; b < a; ++b) {
                v2 -= L[a][b] * y[b];
              }
              y[a] = v2 / L[a][a];
            }
            double gamma_v[4] = {};
            for (int a = p - 1; a >= 0; --a) {
              double v2 = y[a];
              for (int b = a + 1; b < p; ++b) {
                v2 -= L[b][a] * gamma_v[b];
              }
              gamma_v[a] = v2 / L[a][a];
            }
            aa_mix_kernel<<<grid_aa, kBlock>>>(
                ws.d_aa_u[aa_slot].as<double>(),
                ws.d_aa_f[aa_slot].as<double>(),
                ws.d_aa_u[order[0]].as<double>(),
                ws.d_aa_u[p >= 1 ? order[1] : order[0]].as<double>(),
                ws.d_aa_u[p >= 2 ? order[2] : order[0]].as<double>(),
                ws.d_aa_u[p >= 3 ? order[3] : order[0]].as<double>(),
                ws.d_aa_u[p >= 4 ? order[4] : order[0]].as<double>(),
                ws.d_aa_f[order[0]].as<double>(),
                ws.d_aa_f[p >= 1 ? order[1] : order[0]].as<double>(),
                ws.d_aa_f[p >= 2 ? order[2] : order[0]].as<double>(),
                ws.d_aa_f[p >= 3 ? order[3] : order[0]].as<double>(),
                ws.d_aa_f[p >= 4 ? order[4] : order[0]].as<double>(),
                gamma_v[0],
                gamma_v[1],
                gamma_v[2],
                gamma_v[3],
                p,
                aa_beta,
                cfg.numerics.floors.Te,
                state.Te.data(),
                state.Te.data(),
                n_cells);
            cuda_check(cudaGetLastError(), "FLD2D AA mix launch failed");
            mixed = true;
          }
        }
      }
      if (!mixed && p >= 1) {
        // Degenerate LS — plain iteration this round (state.Te already holds
        // the raw Newton output; nothing to do).
      }
    }
  }
  if (!state.fld_converged && state.fld_outer_iterations >= max_iter &&
      !(state.fld_outer_residual <= fld.outer_tol)) {
    ++state.newton_cap_exit_unconverged;
    std::ostringstream oss;
    oss << std::setprecision(17)
        << "FLD2D Newton outer cap exit unconverged: step=" << state.step
        << " group=-1"
        << " outer_iterations=" << state.fld_outer_iterations
        << " outer_residual=" << state.fld_outer_residual
        << " outer_tol=" << fld.outer_tol
        << " policy=" << fld.cap_exit_policy
        << " newton_cap_exit_unconverged="
        << state.newton_cap_exit_unconverged;
    const std::string message = oss.str();
    if (fld.cap_exit_policy == "fail") {
      core::log_fatal(message);
      TENRYU_ASSERT(false, message);
    }
    if (should_log_cap_exit_warning(state.newton_cap_exit_unconverged)) {
      core::log_warning(message);
    }
  }
  if (fld_trace_active && !host_trace_records.empty()) {
    cuda_check(cudaMemcpy(host_trace_records.data(),
                          reinterpret_cast<FldTraceRecord*>(
                              state.fld_trace_records.data()),
                          sizeof(FldTraceRecord) * host_trace_records.size(),
                          cudaMemcpyDeviceToHost),
               "FLD2D copy trace records failed");
    log_fld_trace_records(host_trace_records, state.step, trace.cell, trace.group);
  }

  state.fld_escaped_step =
      compute_escaped_energy(
          state,
          nr,
          nz,
          n_groups,
          dt,
          outer_r_bc,
          z_bottom_bc,
          z_top_bc,
          T_supply_z_bottom_eV,
          T_supply_z_top_eV,
          state_supply_boundary_policy,
          fld_limiter_id,
          rho_supply_z_bottom,
          rho_supply_z_top);
  compute_state_supply_flux_energy(state,
                                   nr,
                                   nz,
                                   n_groups,
                                   dt,
                                   z_bottom_bc,
                                   z_top_bc,
                                   T_supply_z_bottom_eV,
                                   T_supply_z_top_eV,
                                   state_supply_boundary_policy,
                                   fld_limiter_id,
                                   rho_supply_z_bottom,
                                   rho_supply_z_top);
  state.fld_marshak_in_step = compute_marshak_in_energy(
      state, nr, nz, dt, z_bottom_bc, z_top_bc, marshak_flux);
  if (marshak_tr_drive) {
    // Tr(t) route: per-face grey-sum incident flux through the same
    // deterministic area reduction (face-selected via the bc arguments).
    double marshak_tr_in = 0.0;
    if (marshak_tr_finc_sum_bottom > 0.0) {
      marshak_tr_in += compute_marshak_in_energy(
          state, nr, nz, dt, kFldBcMarshak, kFldBcReflect,
          marshak_tr_finc_sum_bottom);
    }
    if (marshak_tr_finc_sum_top > 0.0) {
      marshak_tr_in += compute_marshak_in_energy(
          state, nr, nz, dt, kFldBcReflect, kFldBcMarshak,
          marshak_tr_finc_sum_top);
    }
    state.fld_marshak_in_step = marshak_tr_in;
  }
  const int face_trace_cell = fld_face_trace_cell();
  if (face_trace_cell >= 0 && face_trace_cell < n_cells &&
      state.step == fld_face_trace_step()) {
    log_fld_face_trace(state, ws, nr, nz, n_groups, dt, face_trace_cell);
    log_fld_face_global_max(state, ws, nr, nz, n_groups, dt);
  }
  constexpr double kEscapeBreakdownRhoVacThreshold = 0.011;
  if (state.step == fld_escape_breakdown_step()) {
    const EscapeBreakdownTotals escape_breakdown =
        compute_escape_breakdown_diagnostics(
            state,
            ws,
            nr,
            nz,
            n_groups,
            dt,
            outer_r_bc,
            z_bottom_bc,
            z_top_bc,
            kEscapeBreakdownRhoVacThreshold);
    std::ostringstream escape_breakdown_oss;
    escape_breakdown_oss
        << std::setprecision(17)
        << "[fld_2d_rz_escape_breakdown_prod] step=" << state.step
        << " total_outer_r=" << escape_breakdown.total_outer_r
        << " total_z_bottom=" << escape_breakdown.total_z_bottom
        << " total_z_top=" << escape_breakdown.total_z_top
        << " vacuum_outer_r="
        << escape_breakdown.total_vacuum_outer_r
        << " vacuum_z_bottom="
        << escape_breakdown.total_vacuum_z_bottom
        << " vacuum_z_top=" << escape_breakdown.total_vacuum_z_top
        << " sum_signed_delta_outer_r="
        << escape_breakdown.sum_signed_delta_outer_r
        << " sum_signed_delta_z_bottom="
        << escape_breakdown.sum_signed_delta_z_bottom
        << " sum_signed_delta_z_top="
        << escape_breakdown.sum_signed_delta_z_top
        << " sum_abs_delta=" << escape_breakdown.sum_abs_delta
        << " count_delta_gt_1=" << escape_breakdown.count_delta_gt_1
        << " count_delta_gt_10=" << escape_breakdown.count_delta_gt_10
        << " count_delta_gt_100=" << escape_breakdown.count_delta_gt_100
        << " max_per_cell_outer_r="
        << escape_breakdown.max_per_cell_outer_r
        << " max_per_cell_outer_r_at=("
        << escape_breakdown.max_cell_outer_r_i << ","
        << escape_breakdown.max_cell_outer_r_j << ")"
        << " max_per_cell_z_bottom="
        << escape_breakdown.max_per_cell_z_bottom
        << " max_per_cell_z_bottom_at=("
        << escape_breakdown.max_cell_z_bottom_i << ","
        << escape_breakdown.max_cell_z_bottom_j << ")"
        << " max_per_cell_z_top="
        << escape_breakdown.max_per_cell_z_top
        << " max_per_cell_z_top_at=("
        << escape_breakdown.max_cell_z_top_i << ","
        << escape_breakdown.max_cell_z_top_j << ")"
        << " max_boundary_diag_escape_delta="
        << escape_breakdown.max_boundary_diag_escape_delta
        << " max_boundary_diag_escape_delta_at=("
        << escape_breakdown.max_boundary_diag_escape_delta_i << ","
        << escape_breakdown.max_boundary_diag_escape_delta_j << ")"
        << " max_boundary_diag_escape_delta_g="
        << escape_breakdown.max_boundary_diag_escape_delta_g
        << " max_pos_delta=" << escape_breakdown.max_pos_delta
        << " max_pos_delta_at=(" << escape_breakdown.max_pos_delta_i << ","
        << escape_breakdown.max_pos_delta_j << ")"
        << " max_neg_delta=" << escape_breakdown.max_neg_delta
        << " max_neg_delta_at=(" << escape_breakdown.max_neg_delta_i << ","
        << escape_breakdown.max_neg_delta_j << ")"
        << " csr_diag_value=" << escape_breakdown.csr_diag_value
        << " V_op=" << escape_breakdown.V_op
        << " dt_c_sigma_V=" << escape_breakdown.dt_c_sigma_V
        << " interior_diag_sum=" << escape_breakdown.interior_diag_sum
        << " csr_boundary_diag=" << escape_breakdown.csr_boundary_diag
        << " formula_boundary_coef="
        << escape_breakdown.formula_boundary_coef
        << " rad_E_at_cell=" << escape_breakdown.rad_E_at_cell
        << " sigma_a_at_cell=" << escape_breakdown.sigma_a_at_cell
        << " rho_at_cell=" << escape_breakdown.rho_at_cell
        << " D_cell_at_cell=" << escape_breakdown.D_cell_at_cell
        << " rho_vac_threshold_used="
        << kEscapeBreakdownRhoVacThreshold
        << " fld_escaped_step_total=" << state.fld_escaped_step;
    core::log_info(escape_breakdown_oss.str());
  }
  if (verbose_fld_timing) {
    if (state.step == fld_row_identity_step()) {
      const EscapeBreakdownTotals escape_breakdown =
          compute_escape_breakdown_diagnostics(
              state,
              ws,
              nr,
              nz,
              n_groups,
              dt,
              outer_r_bc,
              z_bottom_bc,
              z_top_bc,
              kEscapeBreakdownRhoVacThreshold);
      std::ostringstream escape_breakdown_oss;
      escape_breakdown_oss
          << std::setprecision(17)
          << "[fld_2d_rz_escape_breakdown] step=" << state.step
          << " total_outer_r=" << escape_breakdown.total_outer_r
          << " total_z_bottom=" << escape_breakdown.total_z_bottom
          << " total_z_top=" << escape_breakdown.total_z_top
          << " vacuum_outer_r="
          << escape_breakdown.total_vacuum_outer_r
          << " vacuum_z_bottom="
          << escape_breakdown.total_vacuum_z_bottom
          << " vacuum_z_top=" << escape_breakdown.total_vacuum_z_top
          << " sum_signed_delta_outer_r="
          << escape_breakdown.sum_signed_delta_outer_r
          << " sum_signed_delta_z_bottom="
          << escape_breakdown.sum_signed_delta_z_bottom
          << " sum_signed_delta_z_top="
          << escape_breakdown.sum_signed_delta_z_top
          << " sum_abs_delta=" << escape_breakdown.sum_abs_delta
          << " count_delta_gt_1=" << escape_breakdown.count_delta_gt_1
          << " count_delta_gt_10=" << escape_breakdown.count_delta_gt_10
          << " count_delta_gt_100=" << escape_breakdown.count_delta_gt_100
          << " max_per_cell_outer_r="
          << escape_breakdown.max_per_cell_outer_r
          << " max_per_cell_outer_r_at=("
          << escape_breakdown.max_cell_outer_r_i << ","
          << escape_breakdown.max_cell_outer_r_j << ")"
          << " max_per_cell_z_bottom="
          << escape_breakdown.max_per_cell_z_bottom
          << " max_per_cell_z_bottom_at=("
          << escape_breakdown.max_cell_z_bottom_i << ","
          << escape_breakdown.max_cell_z_bottom_j << ")"
          << " max_per_cell_z_top="
          << escape_breakdown.max_per_cell_z_top
          << " max_per_cell_z_top_at=("
          << escape_breakdown.max_cell_z_top_i << ","
          << escape_breakdown.max_cell_z_top_j << ")"
          << " max_boundary_diag_escape_delta="
          << escape_breakdown.max_boundary_diag_escape_delta
          << " max_boundary_diag_escape_delta_at=("
          << escape_breakdown.max_boundary_diag_escape_delta_i << ","
          << escape_breakdown.max_boundary_diag_escape_delta_j << ")"
          << " max_boundary_diag_escape_delta_g="
          << escape_breakdown.max_boundary_diag_escape_delta_g
          << " max_pos_delta=" << escape_breakdown.max_pos_delta
          << " max_pos_delta_at=(" << escape_breakdown.max_pos_delta_i << ","
          << escape_breakdown.max_pos_delta_j << ")"
          << " max_neg_delta=" << escape_breakdown.max_neg_delta
          << " max_neg_delta_at=(" << escape_breakdown.max_neg_delta_i << ","
          << escape_breakdown.max_neg_delta_j << ")"
          << " csr_diag_value=" << escape_breakdown.csr_diag_value
          << " V_op=" << escape_breakdown.V_op
          << " dt_c_sigma_V=" << escape_breakdown.dt_c_sigma_V
          << " interior_diag_sum=" << escape_breakdown.interior_diag_sum
          << " csr_boundary_diag=" << escape_breakdown.csr_boundary_diag
          << " formula_boundary_coef="
          << escape_breakdown.formula_boundary_coef
          << " rad_E_at_cell=" << escape_breakdown.rad_E_at_cell
          << " sigma_a_at_cell=" << escape_breakdown.sigma_a_at_cell
          << " rho_at_cell=" << escape_breakdown.rho_at_cell
          << " D_cell_at_cell=" << escape_breakdown.D_cell_at_cell
          << " rho_vac_threshold_used="
          << kEscapeBreakdownRhoVacThreshold
          << " fld_escaped_step_total=" << state.fld_escaped_step;
      core::log_info(escape_breakdown_oss.str());

      const FldRowIdentityDiagnostics row_identity =
          compute_fld_row_identity_diagnostics(
              state,
              ws,
              nr,
              nz,
              n_groups,
              n_rows,
              dt,
              outer_r_bc,
              z_bottom_bc,
              z_top_bc,
              marshak_flux,
              T_supply_z_bottom_eV,
              T_supply_z_top_eV,
              use_fleck,
              state_supply_boundary_policy,
              fld_limiter_id,
              rho_supply_z_bottom,
              rho_supply_z_top);
      const int csr_cell = row_identity.max_delta_csr_vs_matrix.cell_idx;
      const int csr_i = (csr_cell >= 0) ? (csr_cell / nz) : -1;
      const int csr_j = (csr_cell >= 0) ? (csr_cell - csr_i * nz) : -1;
      const int matrix_tally_cell =
          row_identity.max_delta_matrix_vs_tally.cell_idx;
      const int matrix_tally_i =
          (matrix_tally_cell >= 0) ? (matrix_tally_cell / nz) : -1;
      const int matrix_tally_j =
          (matrix_tally_cell >= 0)
              ? (matrix_tally_cell - matrix_tally_i * nz)
              : -1;
      const int emit_cell =
          row_identity.max_delta_emit_kernel_vs_formula.cell_idx;
      const int emit_i = (emit_cell >= 0) ? (emit_cell / nz) : -1;
      const int emit_j = (emit_cell >= 0) ? (emit_cell - emit_i * nz) : -1;
      const int dep_cell =
          row_identity.max_delta_dep_kernel_vs_formula.cell_idx;
      const int dep_i = (dep_cell >= 0) ? (dep_cell / nz) : -1;
      const int dep_j = (dep_cell >= 0) ? (dep_cell - dep_i * nz) : -1;
      std::ostringstream row_identity_oss;
      row_identity_oss << std::setprecision(17)
                       << "[fld_2d_rz_row_identity] step=" << state.step
                       << " sum_csr_residual="
                       << row_identity.sum_csr_residual
                       << " sum_matrix_formula_residual="
                       << row_identity.sum_matrix_formula_residual
                       << " sum_tally_formula_residual="
                       << row_identity.sum_tally_formula_residual
                       << " sum_face_div=" << row_identity.sum_face_div
                       << " sum_abs_face_div="
                       << row_identity.sum_abs_face_div
                       << " sum_emit_kernel="
                       << row_identity.sum_emit_kernel
                       << " sum_emit_formula="
                       << row_identity.sum_emit_formula
                       << " sum_dep_kernel=" << row_identity.sum_dep_kernel
                       << " sum_dep_formula="
                       << row_identity.sum_dep_formula
                       << " max_delta_csr_vs_matrix="
                       << row_identity.max_delta_csr_vs_matrix.value
                       << " max_delta_csr_vs_matrix_cell=" << csr_cell
                       << " max_delta_csr_vs_matrix_i=" << csr_i
                       << " max_delta_csr_vs_matrix_j=" << csr_j
                       << " max_delta_csr_vs_matrix_group="
                       << row_identity.max_delta_csr_vs_matrix.group_idx
                       << " max_delta_matrix_vs_tally="
                       << row_identity.max_delta_matrix_vs_tally.value
                       << " max_delta_matrix_vs_tally_cell="
                       << matrix_tally_cell
                       << " max_delta_matrix_vs_tally_i="
                       << matrix_tally_i
                       << " max_delta_matrix_vs_tally_j="
                       << matrix_tally_j
                       << " max_delta_matrix_vs_tally_group="
                       << row_identity.max_delta_matrix_vs_tally.group_idx
                       << " max_delta_emit_kernel_vs_formula="
                       << row_identity.max_delta_emit_kernel_vs_formula.value
                       << " max_delta_emit_kernel_vs_formula_cell="
                       << emit_cell
                       << " max_delta_emit_kernel_vs_formula_i=" << emit_i
                       << " max_delta_emit_kernel_vs_formula_j=" << emit_j
                       << " max_delta_emit_kernel_vs_formula_group="
                       << row_identity.max_delta_emit_kernel_vs_formula.group_idx
                       << " max_delta_dep_kernel_vs_formula="
                       << row_identity.max_delta_dep_kernel_vs_formula.value
                       << " max_delta_dep_kernel_vs_formula_cell=" << dep_cell
                       << " max_delta_dep_kernel_vs_formula_i=" << dep_i
                       << " max_delta_dep_kernel_vs_formula_j=" << dep_j
                       << " max_delta_dep_kernel_vs_formula_group="
                       << row_identity.max_delta_dep_kernel_vs_formula.group_idx;
      core::log_info(row_identity_oss.str());
    }
    const FldConservationDiagnostics conservation =
        compute_fld_conservation_diagnostics(
            state, ws, n_cells, n_groups, n_rows);
    const double boundary_net =
        state.fld_escaped_step - state.fld_state_supply_out_step +
        state.fld_state_supply_net_step;
    const double operator_defect =
        conservation.rad_delta + boundary_net - conservation.emit_minus_dep;
    const double internal_pair_defect =
        operator_defect - conservation.residual_sum;
    const FldFaceSymmetryRecord face_symmetry =
        compute_fld_face_symmetry_diagnostics(
            state, ws, nr, nz, n_groups, dt);
    const FldPerRowDefectDiagnostics per_row_defect =
        compute_fld_per_row_defect_diagnostics(
            state,
            ws,
            nr,
            nz,
            n_groups,
            n_rows,
            dt,
            outer_r_bc,
            z_bottom_bc,
            z_top_bc,
            T_supply_z_bottom_eV,
            T_supply_z_top_eV,
            use_fleck,
            state_supply_boundary_policy,
            fld_limiter_id,
            rho_supply_z_bottom,
            rho_supply_z_top);
    std::ostringstream conservation_oss;
    conservation_oss << std::setprecision(17)
                     << "[fld_2d_rz_conservation] step=" << state.step
                     << " sum_Ax_minus_rhs=" << conservation.residual_sum
                     << " rad_delta=" << conservation.rad_delta
                     << " emit_minus_dep=" << conservation.emit_minus_dep
                     << " boundary_escape=" << state.fld_escaped_step
                     << " boundary_state_supply_net="
                     << state.fld_state_supply_net_step
                     << " operator_defect=" << operator_defect
                     << " cg_residual=" << conservation.residual_sum
                     << " internal_pair_defect=" << internal_pair_defect
                     << " marshak_in=" << state.fld_marshak_in_step;
    core::log_info(conservation_oss.str());
    std::ostringstream face_oss;
    face_oss << std::setprecision(17)
             << "[fld_2d_rz_face_symmetry] step=" << state.step
             << " max_face_pair_defect=" << face_symmetry.value
             << " max_face_pair_cell_l=" << face_symmetry.cell_l
             << " max_face_pair_cell_r=" << face_symmetry.cell_r
             << " max_face_pair_direction=" << face_symmetry.direction;
    core::log_info(face_oss.str());
    const int max_cell = per_row_defect.max_record.cell_idx;
    const int max_i = (max_cell >= 0) ? (max_cell / nz) : -1;
    const int max_j = (max_cell >= 0) ? (max_cell - max_i * nz) : -1;
    std::ostringstream row_oss;
    row_oss << std::setprecision(17)
            << "[fld_2d_rz_per_row_defect] step=" << state.step
            << " sum_defect=" << per_row_defect.sum_defect
            << " max_abs_defect=" << per_row_defect.max_record.value
            << " max_row_defect=" << per_row_defect.max_record.row_defect
            << " max_defect_cell=" << max_cell
            << " max_defect_i=" << max_i
            << " max_defect_j=" << max_j
            << " max_defect_group=" << per_row_defect.max_record.group_idx
            << " max_cell_V_op=" << per_row_defect.max_record.V_op
            << " max_cell_V_state=" << per_row_defect.max_record.V_state
            << " num_vol_mismatch=" << per_row_defect.num_vol_mismatch_cells
            << " sum_vol_diff=" << per_row_defect.sum_vol_diff
            << " hist_abs_gt_1e-3=" << per_row_defect.num_abs_gt_1e_3
            << " hist_abs_gt_1e-2=" << per_row_defect.num_abs_gt_1e_2
            << " hist_abs_gt_1e-1=" << per_row_defect.num_abs_gt_1e_1
            << " hist_abs_gt_1=" << per_row_defect.num_abs_gt_1
            << " hist_abs_gt_10=" << per_row_defect.num_abs_gt_10;
    core::log_info(row_oss.str());
  }
  state.holo_ale_invalidated = false;
  if (verbose_fld_timing) {
    copy_rogue_records_to_state(state, ws);
    core::log_info(std::string("[fld_2d_rz_timing] outer_iters=") +
                   std::to_string(state.fld_outer_iterations) +
                   " outer_residual=" +
                   std::to_string(state.fld_outer_residual) +
                   " cg_iters_total=" +
                   std::to_string(total_cg_iters_this_step) +
                   " cg_iters_max=" +
                   std::to_string(max_cg_iters_this_step) +
                   " early_return=" + std::to_string(n_early_return) +
                   " clamp_hits=" +
                   std::to_string(state.fld_clamp_hits_step) +
                   " clamp_E_delta=" +
                   std::to_string(state.fld_clamp_energy_delta_step) +
                   " min_x_raw=" + std::to_string(state.fld_min_x_raw_step) +
                   " cg_true_resid_l2=" +
                   std::to_string(state.fld_cg_true_residual_l2_rel_step) +
                   " cg_true_resid_max=" +
                   std::to_string(state.fld_cg_true_residual_max_step) +
                   " E_solver=" + std::to_string(state.fld_E_solver_step));
    core::log_info(std::string("[fld_2d_rz_diagnostics_h1] ") +
                   "cg_true_resid_l2_RAW=" +
                   std::to_string(state.fld_cg_true_residual_l2_rel_RAW_step) +
                   " cg_true_resid_max_RAW=" +
                   std::to_string(state.fld_cg_true_residual_max_RAW_step) +
                   " E_solver_RAW=" +
                   std::to_string(state.fld_E_solver_RAW_step) +
                   " diag_min_pos=" +
                   std::to_string(state.fld_csr_diag_min_pos_step) +
                   " diag_max=" +
                   std::to_string(state.fld_csr_diag_max_step) +
                   " weak_diag_dom_count=" +
                   std::to_string(state.fld_csr_weak_diag_dom_count_step) +
                   " csr_nonfinite_count=" +
                   std::to_string(state.fld_csr_nonfinite_count_step) +
                   " gershgorin_lower_min=" +
                   std::to_string(state.fld_gershgorin_lower_min_step) +
                   " gershgorin_upper_max=" +
                   std::to_string(state.fld_gershgorin_upper_max_step) +
                   " cg_pAp_min=" +
                   std::to_string(state.fld_cg_pAp_min_step) +
                   " cg_nonpos_pAp_count=" +
                   std::to_string(state.fld_cg_nonpos_pAp_count_step) +
                   " cg_nonfinite_count=" +
                   std::to_string(state.fld_cg_nonfinite_count_step) +
                   " cg_recurrent_resid_last=" +
                   std::to_string(
                       state.fld_cg_recurrent_resid_last_check_max_step));
    core::log_info(std::string("[fld_2d_rz_diagnostics_h2] ") +
                   "Dcell_min=" + std::to_string(state.fld_Dcell_min_step) +
                   " Dcell_max=" + std::to_string(state.fld_Dcell_max_step) +
                   " Dcell_zero_count=" +
                   std::to_string(state.fld_Dcell_zero_count_step) +
                   " Dcell_nonfinite_count=" +
                   std::to_string(state.fld_Dcell_nonfinite_count_step) +
                   " face_skip_D=" +
                   std::to_string(state.fld_face_skip_D_count_step) +
                   " face_skip_nf=" +
                   std::to_string(state.fld_face_skip_nonfinite_count_step) +
                   " face_skip_dist=" +
                   std::to_string(state.fld_face_skip_dist_count_step) +
                   " diag_fallback=" +
                   std::to_string(state.fld_diag_fallback_count_step) +
                   " pair_symm_max_diff=" +
                   std::to_string(state.fld_pair_symmetry_max_diff_step) +
                   " pair_symm_violation_count=" +
                   std::to_string(
                       state.fld_pair_symmetry_violation_count_step));
    core::log_info(std::string("[fld_2d_rz_diagnostics_h3] ") +
                   "newton_converged=" +
                   std::to_string(state.fld_newton_converged_count_step) +
                   " newton_cap_hit=" +
                   std::to_string(state.fld_newton_cap_hit_count_step) +
                   " newton_invalid=" +
                   std::to_string(state.fld_newton_invalid_count_step) +
                   " newton_resid_abs_max=" +
                   std::to_string(state.fld_newton_resid_abs_max_step) +
                   " newton_resid_rel_max=" +
                   std::to_string(state.fld_newton_resid_rel_max_step) +
                   " newton_reject=" +
                   std::to_string(state.fld_newton_reject_count_step) +
                   " newton_reject_resid_rel_max=" +
                   std::to_string(
                       state.fld_newton_reject_resid_rel_max_step));
    core::log_info(std::string("[fld_2d_rz_diagnostics_rogue] ") +
                   "rogue_max_abs_x_raw=" +
                   std::to_string(state.fld_rogue_max_abs_x_raw_step) +
                   " rogue_max_abs_x_raw_cell=" +
                   std::to_string(state.fld_rogue_max_abs_x_raw_cell_step) +
                   " rogue_max_abs_x_raw_group=" +
                   std::to_string(state.fld_rogue_max_abs_x_raw_group_step) +
                   " rogue_min_x_raw=" +
                   std::to_string(state.fld_rogue_min_x_raw_step) +
                   " rogue_min_x_raw_cell=" +
                   std::to_string(state.fld_rogue_min_x_raw_cell_step) +
                   " rogue_min_x_raw_group=" +
                   std::to_string(state.fld_rogue_min_x_raw_group_step) +
                   " rogue_max_rad_E=" +
                   std::to_string(state.fld_rogue_max_rad_E_step) +
                   " rogue_max_rad_E_cell=" +
                   std::to_string(state.fld_rogue_max_rad_E_cell_step) +
                   " rogue_max_rad_E_group=" +
                   std::to_string(state.fld_rogue_max_rad_E_group_step) +
                   " rogue_max_r_true=" +
                   std::to_string(state.fld_rogue_max_r_true_step) +
                   " rogue_max_r_true_cell=" +
                   std::to_string(state.fld_rogue_max_r_true_cell_step) +
                   " rogue_max_r_true_group=" +
                   std::to_string(state.fld_rogue_max_r_true_group_step) +
                   " rogue_max_E_solver_row=" +
                   std::to_string(state.fld_rogue_max_E_solver_row_step) +
                   " rogue_max_E_solver_row_cell=" +
                   std::to_string(
                       state.fld_rogue_max_E_solver_row_cell_step) +
                   " rogue_max_E_solver_row_group=" +
                   std::to_string(
                       state.fld_rogue_max_E_solver_row_group_step));
  } else if (collect_fld_newton_diagnostics) {
    core::log_info(std::string("[fld_2d_rz_diagnostics_h3] ") +
                   "newton_converged=" +
                   std::to_string(state.fld_newton_converged_count_step) +
                   " newton_cap_hit=" +
                   std::to_string(state.fld_newton_cap_hit_count_step) +
                   " newton_invalid=" +
                   std::to_string(state.fld_newton_invalid_count_step) +
                   " newton_resid_abs_max=" +
                   std::to_string(state.fld_newton_resid_abs_max_step) +
                   " newton_resid_rel_max=" +
                   std::to_string(state.fld_newton_resid_rel_max_step) +
                   " newton_reject=" +
                   std::to_string(state.fld_newton_reject_count_step) +
                   " newton_reject_resid_rel_max=" +
                   std::to_string(
                       state.fld_newton_reject_resid_rel_max_step));
  }
}

double fld_compute_max_reduced_flux_2d_rz(
    core::State& state,
    const core::Config& cfg,
    const PlanckTable& planck,
    const core::Config::MaterialsConfig::MatDef& mat) {
  const int nr = cfg.mesh.nr;
  const int nz = cfg.mesh.nz;
  const int n_cells = nr * nz;
  const int n_groups = std::max(cfg.radiation.groups, 1);
  ensure_state_buffers(state, n_cells, n_groups);
  evaluate_fld_opacity_and_emission(state, cfg, planck, mat, n_cells, n_groups, 1.0e-30);
  compute_diffusion_coefficients(state, cfg, nr, nz, n_groups);
  zero_reduction_scalar(state);
  const int n_faces = std::max(nr - 1, 0) * nz + nr * std::max(nz - 1, 0);
  const int total = n_faces * n_groups;
  const int grid = (total + kBlock - 1) / kBlock;
  if (grid > 0) {
    max_reduced_flux_2d_kernel<<<grid, kBlock>>>(state.rad_E.data(),
                                                 state.fld_D_cell.data(),
                                                 state.fld_cell_rc.data(),
                                                 state.fld_cell_zc.data(),
                                                 state.fld_reduction_work.data(),
                                                 nr,
                                                 nz,
                                                 n_groups);
    cuda_check(cudaGetLastError(), "FLD2D reduced flux launch failed");
  }
  return copy_reduction_scalar(state);
}

}  // namespace tenryu::radiation
