#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::radiation {

/// Host-side RadLite mesh data (1D only).
struct RadLiteMesh1D {
  int n_hydro = 0;    ///< Number of hydro cells
  int n_rad = 0;      ///< Number of radiation cells
  int n_groups = 0;   ///< Number of energy groups

  // --- Topology ---
  std::vector<int32_t> hydro_to_rad;   ///< [n_hydro] hydro cell -> rad cell index
  std::vector<int32_t> rad_h_begin;    ///< [n_rad]   inclusive start hydro index
  std::vector<int32_t> rad_h_end;      ///< [n_rad]   exclusive end hydro index
  std::vector<double> rad_node_r;      ///< [n_rad+1] radiation node positions [cm]

  // --- Merged coefficients ---
  std::vector<double> sigma_a_eff_rad;  ///< [n_rad * n_groups] volume-weighted absorption
  std::vector<double> sigma_s_eff_rad;  ///< [n_rad * n_groups] volume-weighted scattering
  std::vector<double> Te_rad;           ///< [n_rad] volume-weighted electron temperature

  // --- Tally redistribution weights ---
  /// w_dep[i*G+g] = V_i * sigma_a_eff_hydro[i*G+g] / sum_{j in rad} V_j *
  /// sigma_a_eff_hydro[j*G+g]
  std::vector<double> w_dep;  ///< [n_hydro * n_groups] deposition weights
  /// w_tl[i] = V_i / sum_{j in rad} V_j
  std::vector<double> w_tl;   ///< [n_hydro] energy tally weights (volume-based)

  bool enabled = false;  ///< Whether RadLite mesh is actually built (false = passthrough)
};

/// Device-side view passed to CUDA kernels.
struct RadLiteMeshDeviceView {
  const int32_t* hydro_to_rad = nullptr;   ///< [n_hydro]
  const int32_t* rad_h_begin = nullptr;    ///< [n_rad]
  const int32_t* rad_h_end = nullptr;      ///< [n_rad]
  const double* rad_node_r = nullptr;      ///< [n_rad+1]
  const double* hydro_node_r = nullptr;    ///< [n_hydro+1] (original hydro nodes, for census remap)
  const double* sigma_a_eff_rad = nullptr; ///< [n_rad * G]
  const double* sigma_s_eff_rad = nullptr; ///< [n_rad * G]
  const double* Te_rad = nullptr;          ///< [n_rad]
  int n_hydro = 0;
  int n_rad = 0;
  int n_groups = 0;
};

/// Build RadLite mesh from hydro mesh data (host-side).
/// @param n_cells  Number of hydro cells
/// @param n_groups Number of energy groups
/// @param node_r   Hydro node positions [n_cells+1]
/// @param vol      Hydro cell volumes [n_cells]
/// @param sigma_a_eff Effective absorption [n_cells * n_groups]
/// @param sigma_s_eff Effective scattering [n_cells * n_groups]
/// @param Te       Electron temperature [n_cells]
/// @param ddmc_mode DDMC mode flags [n_cells * n_groups] (nullptr if no DDMC)
/// @param material_id Material ID per cell [n_cells] (cells with different materials cannot merge)
/// @param sigma_ratio_max Maximum opacity ratio for merging (default 2.0)
/// @return Built RadLiteMesh1D
RadLiteMesh1D build_rad_lite_mesh(int n_cells,
                                  int n_groups,
                                  const double* node_r,
                                  const double* vol,
                                  const double* sigma_a_eff,
                                  const double* sigma_s_eff,
                                  const double* Te,
                                  const int8_t* ddmc_mode,  // TransportMode encoded as int8_t
                                  const int* material_id,
                                  double sigma_ratio_max = 2.0);

/// Redistribute rad-indexed tallies to hydro-indexed arrays.
/// hydro_dep[i*G+g] = rad_dep[r*G+g] * w_dep[i*G+g]
/// hydro_tl[i*G+g]  = rad_tl[r*G+g]  * w_tl[i]
void redistribute_tallies_cuda(double* hydro_dep,             ///< [n_hydro * G] output
                               double* hydro_tl,              ///< [n_hydro * G] output
                               const double* rad_dep,         ///< [n_rad * G] input from kernel
                               const double* rad_tl,          ///< [n_rad * G] input from kernel
                               const int32_t* hydro_to_rad,   ///< [n_hydro] mapping
                               const double* w_dep,           ///< [n_hydro * G] weights
                               const double* w_tl,            ///< [n_hydro] weights
                               int n_hydro,
                               int n_groups);

/// Device-side data for RadLite mesh (owns CUDA memory).
/// Call free_rad_device_data() to release.
struct RadDeviceData {
  double* sigma_a_eff = nullptr;    // [n_rad * G]
  double* sigma_s_eff = nullptr;    // [n_rad * G]
  double* Te = nullptr;             // [n_rad]
  double* vol = nullptr;            // [n_rad]
  double* node_r = nullptr;         // [n_rad+1]
  double* sigma_R = nullptr;        // [n_rad * G]
  double* sigma_a = nullptr;        // [n_rad * G]
  double* cell_dx = nullptr;        // [n_rad]
  double* fleck_f = nullptr;        // [n_rad]
  int8_t* ddmc_mode = nullptr;      // [n_rad * G]
  double* eta_cdf = nullptr;        // [n_rad * G] or nullptr
  double* rad_dep = nullptr;        // [n_rad * G]
  double* rad_E_tally = nullptr;    // [n_rad * G]
  int32_t* hydro_to_rad = nullptr;  // [n_hydro]
  double* w_dep = nullptr;          // [n_hydro * G]
  double* w_tl = nullptr;           // [n_hydro]
  double* hydro_node_r = nullptr;   // [n_hydro+1] for census remap
  int32_t* rad_h_begin = nullptr;   // [n_rad]
  int32_t* rad_h_end = nullptr;     // [n_rad]
  int n_rad = 0;
  int n_hydro = 0;
  int n_groups = 0;
  bool active = false;
};

/// Allocate and upload RadLite device arrays from host RadLiteMesh1D.
/// Also uploads auxiliary arrays (sigma_R_rad, cell_dx_rad, fleck_f_rad, ddmc_mode_rad,
/// sigma_a_rad, eta_cdf_rad) which are computed from the provided host-side hydro arrays.
/// @param mesh         The built RadLiteMesh1D (from build_rad_lite_mesh)
/// @param hydro_node_r Hydro node positions [n_hydro+1]
/// @param hydro_vol    Hydro cell volumes [n_hydro]
/// @param sigma_R_hydro Rosseland opacity [n_hydro * G] (nullptr if no DDMC)
/// @param sigma_a_hydro Raw absorption [n_hydro * G]
/// @param fleck_f_hydro Fleck factor [n_hydro]
/// @param ddmc_mode_hydro DDMC mode [n_hydro * G] as int8_t (nullptr if no DDMC)
/// @param eta_cdf_hydro NLTE eta_cdf [n_hydro * G] (nullptr if LTE)
RadDeviceData prepare_rad_device_data(const RadLiteMesh1D& mesh,
                                      const double* hydro_node_r,
                                      const double* hydro_vol,
                                      const double* sigma_R_hydro,
                                      const double* sigma_a_hydro,
                                      const double* fleck_f_hydro,
                                      const int8_t* ddmc_mode_hydro,
                                      const double* eta_cdf_hydro);

/// Free all CUDA memory owned by RadDeviceData.
void free_rad_device_data(RadDeviceData& data);

/// Remap particle cell_ids from hydro to rad indices.
/// For n particles starting at pool.cell_id[0..n-1]:
///   cell_id[i] = hydro_to_rad[cell_id[i]]  (only if cell_id[i] >= 0)
void remap_cell_ids_to_rad_cuda(int32_t* cell_id,            ///< [n] particle cell IDs (in-place)
                                const int32_t* hydro_to_rad, ///< [n_hydro] mapping
                                int n);

/// Remap particle cell_ids from rad back to hydro indices using binary search.
/// For each particle, find the hydro cell containing pos_r[i] within the rad cell's hydro
/// interval.
void remap_cell_ids_to_hydro_cuda(int32_t* cell_id,           ///< [n] particle cell IDs (in-place)
                                  const double* pos_r,        ///< [n] particle radial positions
                                  const double* hydro_node_r, ///< [n_hydro+1] hydro node positions
                                  const int32_t* rad_h_begin, ///< [n_rad] inclusive start
                                  const int32_t* rad_h_end,   ///< [n_rad] exclusive end
                                  int n,
                                  int n_rad);

}  // namespace tenryu::radiation
