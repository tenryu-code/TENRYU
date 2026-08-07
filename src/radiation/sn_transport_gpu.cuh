#pragma once

#include "materials/eos_device_table.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"
#include "radiation/planck_table.cuh"

namespace tenryu::radiation {

constexpr int kSNBoundaryVacuum = 0;
constexpr int kSNBoundaryReflect = 1;
constexpr int kSNBoundaryMarshak = 2;

struct SNTransportGPUConfig {
  int n_angles = 8;
  int max_iterations = 200;
  double convergence_tol = 1.0e-6;
  double temperature_floor_eV = 1.0e-12;
  int block_threads_2d = 128;
  bool origin_parity_only = false;
  bool dsa_enabled = true;
  bool z_boundary_reflect = false;
  int z_bottom_boundary = kSNBoundaryVacuum;
  int z_top_boundary = kSNBoundaryVacuum;
  int r_outer_boundary = kSNBoundaryVacuum;
  double marshak_flux_erg_per_cm2_s = 0.0;
  bool linear_characteristic = false;
};

struct SNTransportGPUResult {
  int iterations = 0;
  int n_directions = 0;
  int lo_solver_iterations = 0;
  int lo_failures = 0;
  double convergence_error = 0.0;
  bool converged = false;
};

struct SNTransport1DGPUInputs {
  const double* sigma_a = nullptr;          // [n_cells * n_groups], [1/cm]
  const double* sigma_s = nullptr;          // [n_cells * n_groups], [1/cm]
  const double* source_emission = nullptr;  // [n_cells * n_groups], [erg/cm^3/s]
  const double* node_r = nullptr;           // [n_cells + 1], [cm]
  const double* vol = nullptr;              // [n_cells], [cm^3]
  double* E_out = nullptr;                  // [n_cells * n_groups], [erg/cm^3]
  double* P_rr_out = nullptr;               // [n_cells * n_groups], [erg/cm^3]
  double* chi_out = nullptr;                // [n_cells * n_groups]
  double* psi_bar = nullptr;                // optional [n_groups * n_angles * n_cells]
  int n_cells = 0;
  int n_groups = 0;
  double dt = 0.0;
};

struct SNTransport2DRZGPUInputs {
  const double* sigma_a = nullptr;          // [nr * nz * n_groups], [1/cm]
  const double* sigma_s = nullptr;          // [nr * nz * n_groups], [1/cm]
  const double* source_emission = nullptr;  // [nr * nz * n_groups], [erg/cm^3/s]
  const double* node_r = nullptr;           // [(nr + 1) * (nz + 1)], [cm]
  const double* node_z = nullptr;           // [(nr + 1) * (nz + 1)], [cm]
  const double* vol = nullptr;              // [nr * nz], [cm^3]
  double* E_out = nullptr;                  // [nr * nz * n_groups], [erg/cm^3]
  double* P_rr_out = nullptr;               // [nr * nz * n_groups], [erg/cm^3]
  double* chi_out = nullptr;                // [nr * nz * n_groups]
  double* F_z_out = nullptr;                // optional [nr * nz * n_groups], [erg/cm^2/s]
  double* P_zz_out = nullptr;               // optional [nr * nz * n_groups], [erg/cm^3]
  double* chi_z_out = nullptr;              // optional [nr * nz * n_groups]
  double* phi_sweep_out = nullptr;          // optional [nr * nz * n_groups]
  double* face_flux_raw = nullptr;          // optional [((nr+1)*nz + nr*(nz+1)) * n_groups]
  unsigned long long* radial_fixup_count = nullptr;    // optional [nr * nz * n_groups]
  double* radial_fixup_artificial_abs = nullptr;        // optional [nr * nz * n_groups]
  unsigned long long* angular_fixup_count = nullptr;   // optional [nr * nz * n_groups]
  double* angular_fixup_artificial_abs = nullptr;      // optional [nr * nz * n_groups]
  int nr = 0;
  int nz = 0;
  int n_groups = 0;
  // MPI (Option C r-slab, M18c slice-3): owned cell window + comm
  // context. Serial defaults keep the P=1 byte path (mpi_c_end == 0
  // means full range).
  const parallel::PartitionInfo* mpi_part = nullptr;
  parallel::CommBuffers* mpi_bufs = nullptr;
  int mpi_c_begin = 0;
  int mpi_c_end = 0;
};

struct SNMaterialCouplingGPUInputs {
  const double* sigma_a = nullptr;  // [n_cells * n_groups], Fleck effective absorption
  const double* sigma_s = nullptr;  // [n_cells * n_groups], Fleck effective scattering
  const double* source_emission = nullptr; // optional [n_cells * n_groups]
  double* Te = nullptr;             // [n_cells], [eV], emission/Newton temperature
  double* ee = nullptr;             // [n_cells], [erg/g]
  const double* node_r = nullptr;   // 1D: [n_cells + 1], 2D: [(nr + 1) * (nz + 1)]
  const double* node_z = nullptr;   // 2D only: [(nr + 1) * (nz + 1)]
  const double* vol = nullptr;      // [n_cells], [cm^3]
  const double* rho = nullptr;      // [n_cells], [g/cm^3]
  const double* cv_e = nullptr;     // optional [n_cells], [erg/(g*eV)]
  const double* sigma_R = nullptr;  // legacy material-coupling input, ignored by IMEX S_N
  const double* Te_old = nullptr;   // [n_cells], [eV], pre-Newton temperature
  PlanckTableDeviceView planck{};
  materials::DeviceEOSTableView electron_eos{}; // optional TMAT electron EOS
  const PlanckTable* planck_table_cpu = nullptr;
  double* E_out = nullptr;          // [n_cells * n_groups], [erg/cm^3]
  double* P_rr_out = nullptr;       // [n_cells * n_groups], [erg/cm^3]
  double* chi_out = nullptr;        // [n_cells * n_groups]
  double* F_z_out = nullptr;        // optional 2D [n_cells * n_groups], [erg/cm^2/s]
  double* P_zz_out = nullptr;       // optional 2D [n_cells * n_groups], [erg/cm^3]
  double* chi_z_out = nullptr;      // optional 2D [n_cells * n_groups]
  double* rad_dep = nullptr;        // [n_cells * n_groups], [erg]
  double* rad_emit = nullptr;       // [n_cells * n_groups], [erg]
  double* phi_sweep_out = nullptr;  // optional [n_cells * n_groups]
  double* face_flux_raw = nullptr;  // optional 2D unique-face raw flux [n_faces * n_groups]
  unsigned long long* radial_fixup_count = nullptr;    // optional 2D [n_cells * n_groups]
  double* radial_fixup_artificial_abs = nullptr;        // optional 2D [n_cells * n_groups]
  unsigned long long* angular_fixup_count = nullptr;   // optional 2D [n_cells * n_groups]
  double* angular_fixup_artificial_abs = nullptr;      // optional 2D [n_cells * n_groups]
  double* coverage = nullptr;       // optional [n_cells * n_groups]
  int dim = 1;
  int nr = 0;
  int nz = 1;
  int n_groups = 0;
  double dt = 0.0;
  bool update_material = true;
  double cv_e_const = 0.0;          // fallback mass heat capacity [erg/(g*eV)]
  double Cv_e_const = 0.0;          // optional volume heat capacity [erg/(cm^3*eV)]
  // MPI (Option C r-slab, M18c slice-3): owned cell window + comm
  // context. Serial defaults keep the P=1 byte path (mpi_c_end == 0
  // means full range).
  const parallel::PartitionInfo* mpi_part = nullptr;
  parallel::CommBuffers* mpi_bufs = nullptr;
  int mpi_c_begin = 0;
  int mpi_c_end = 0;
};

SNTransportGPUResult solve_sn_transport_1d_gpu(
    const SNTransport1DGPUInputs& in,
    const SNTransportGPUConfig& config);

SNTransportGPUResult solve_sn_transport_2d_rz_gpu(
    const SNTransport2DRZGPUInputs& in,
    const SNTransportGPUConfig& config);

SNTransportGPUResult solve_sn_material_coupling_gpu(
    const SNMaterialCouplingGPUInputs& in,
    const SNTransportGPUConfig& config);

void sn_reduce_cell_outputs_2d_for_test(
    const double* D_avg,
    const double* mu_r,
    const double* mu_z,
    const double* weights,
    double* scalar_flux,
    double* E_out,
    double* P_rr_out,
    double* F_z_out,
    double* P_zz_out,
    double* chi_out,
    double* chi_z_out,
    int n_cells,
    int n_groups,
    int n_dirs);

}  // namespace tenryu::radiation
