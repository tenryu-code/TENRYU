#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/config.hpp"
#include "core/device_error_flags.cuh"
#include "core/state.hpp"
#include "materials/ionmix_reader.hpp"
#include "materials/ionmix_reader.cuh"
#include "parallel/comm_buffers.hpp"
#include "parallel/partition.hpp"
#include "radiation/cell_radiation_coeffs.hpp"
#include "radiation/holo_lo_solver.hpp"
#include "radiation/holo_selector.hpp"
#include "radiation/mode_selector.hpp"
#include "radiation/particle_pool.cuh"
#include "radiation/planck_table.cuh"

namespace tenryu::parallel {
struct CommBuffers;
}

namespace tenryu::radiation {

struct DtRadDiagnostics;

struct ReferenceFieldDiagnostics {
  bool valid = false;
  std::int64_t eligible_cells = 0;
  std::int64_t active_cells = 0;
  std::int64_t strong_cells = 0;
  std::int64_t hybrid_suppressed_cells = 0;
  double W_min = 0.0;
  double W_mean = 0.0;
  double W_max = 0.0;
  double tau_min = 0.0;
  double tau_mean = 0.0;
  double tau_max = 0.0;
  double chi_mean = 0.0;
  double chi_max = 0.0;
  double reduced_flux_max = 0.0;
  double knudsen_max = 0.0;
  double front_grad_Te_max = 0.0;
  double front_grad_rho_max = 0.0;
  double E_ref_total = 0.0;
};

[[nodiscard]] double difference_reference_weight(double W_max,
                                                 double tau,
                                                 double tau0,
                                                 double chi,
                                                 double chi0);

[[nodiscard]] double difference_reference_cell_sigma(double weight_sum,
                                                     double inverse_sigma_weight_sum,
                                                     double sigma_max);

class IMC {
 public:
  IMC() = default;
  ~IMC();

  IMC(const IMC&) = delete;
  IMC& operator=(const IMC&) = delete;
  IMC(IMC&&) noexcept = default;
  IMC& operator=(IMC&&) noexcept = default;

  void transport_step(core::State& state,
                      const core::Config& cfg,
                      double dt,
                      const parallel::PartitionInfo& part = parallel::PartitionInfo{},
                      parallel::CommBuffers* bufs = nullptr);
  void invalidate_difference_reference();

  [[nodiscard]] static double compute_dt_rad(const core::State& state,
                                             const core::Config& cfg);
  [[nodiscard]] static double compute_dt_rad(const core::State& state,
                                             const core::Config& cfg,
                                             DtRadDiagnostics* diag);

  [[nodiscard]] int n_alive() const { return pool_.n_alive; }
  [[nodiscard]] double census_energy() const;
  [[nodiscard]] double escaped_energy_total() const {
    return escaped_energy_total_;
  }
  [[nodiscard]] double last_numerical_loss_step() const {
    return last_numerical_loss_step_;
  }
  [[nodiscard]] double last_thermal_lost_step() const {
    return last_thermal_lost_step_;
  }
  [[nodiscard]] double last_marshak_in_step() const {
    return last_marshak_in_step_;
  }
  [[nodiscard]] double last_volume_source_step() const {
    return last_volume_source_step_;
  }
  [[nodiscard]] std::uint64_t last_interface_transitions() const {
    return last_interface_transitions_;
  }
  [[nodiscard]] std::uint64_t last_interface_reflections() const {
    return last_interface_reflections_;
  }
  [[nodiscard]] std::uint64_t last_conversion_prob_violations() const {
    return last_conversion_prob_violations_;
  }
  [[nodiscard]] std::uint64_t last_cnt_boundary() const {
    return last_cnt_boundary_;
  }
  [[nodiscard]] std::uint64_t last_cnt_scatter() const {
    return last_cnt_scatter_;
  }
  [[nodiscard]] std::uint64_t last_cnt_census() const {
    return last_cnt_census_;
  }
  [[nodiscard]] std::uint64_t last_cnt_absorb_kill() const {
    return last_cnt_absorb_kill_;
  }
  [[nodiscard]] std::uint64_t last_cnt_absorb_survive() const {
    return last_cnt_absorb_survive_;
  }
  [[nodiscard]] std::uint64_t last_cnt_roulette_kill() const {
    return last_cnt_roulette_kill_;
  }
  [[nodiscard]] double last_f_min() const {
    return last_f_min_;
  }
  [[nodiscard]] double last_f_mean() const {
    return last_f_mean_;
  }
  [[nodiscard]] double last_f_p95() const {
    return last_f_p95_;
  }
  [[nodiscard]] int last_tau_gt1() const {
    return last_tau_gt1_;
  }
  [[nodiscard]] int last_tau_gt2() const {
    return last_tau_gt2_;
  }
  [[nodiscard]] int last_tau_gt3() const {
    return last_tau_gt3_;
  }
  [[nodiscard]] int last_tau_gt4() const {
    return last_tau_gt4_;
  }
  [[nodiscard]] std::int64_t last_ddmc_mode_count() const {
    return last_ddmc_mode_count_;
  }
  [[nodiscard]] std::int64_t last_imc_mode_count() const {
    return last_imc_mode_count_;
  }
  [[nodiscard]] std::int64_t last_n_total() const {
    return last_n_total_;
  }
  [[nodiscard]] std::int64_t last_n_imc_particles() const {
    return last_n_imc_particles_;
  }
  [[nodiscard]] std::int64_t last_n_ddmc_particles() const {
    return last_n_ddmc_particles_;
  }
  [[nodiscard]] std::int64_t last_n_census() const {
    return last_n_census_;
  }
  [[nodiscard]] std::int64_t last_n_absorbed() const {
    return last_n_absorbed_;
  }
  [[nodiscard]] std::int64_t last_n_escaped() const {
    return last_n_escaped_;
  }
  [[nodiscard]] double last_ddmc_fraction() const {
    return last_ddmc_fraction_;
  }
  [[nodiscard]] double last_weight_min() const {
    return last_weight_min_;
  }
  [[nodiscard]] double last_weight_mean() const {
    return last_weight_mean_;
  }
  [[nodiscard]] double last_weight_max() const {
    return last_weight_max_;
  }
  [[nodiscard]] std::int64_t last_overshoot_count() const {
    return last_overshoot_count_;
  }
  [[nodiscard]] double last_overshoot_max() const {
    return last_overshoot_max_;
  }
  [[nodiscard]] std::int64_t last_mmatrix_violations() const {
    return last_mmatrix_violations_;
  }
  [[nodiscard]] std::int64_t last_mmatrix_fallback_count() const {
    return last_mmatrix_fallback_count_;
  }
  [[nodiscard]] std::int64_t last_omega_below_threshold() const {
    return last_omega_below_threshold_;
  }
  [[nodiscard]] std::int64_t last_ddmc_to_imc_conversions() const {
    return last_ddmc_to_imc_conversions_;
  }
  [[nodiscard]] std::int64_t last_switches_imc_to_ddmc() const {
    return last_switches_imc_to_ddmc_;
  }
  [[nodiscard]] std::int64_t last_switches_ddmc_to_imc() const {
    return last_switches_ddmc_to_imc_;
  }
  [[nodiscard]] double last_rad_momentum_deposition() const {
    return last_rad_momentum_deposition_;
  }
  [[nodiscard]] const core::DeviceErrorFlags& last_device_error_flags() const {
    return last_device_error_flags_;
  }
  [[nodiscard]] const ReferenceFieldDiagnostics& last_reference_field_diagnostics() const {
    return last_reference_field_diagnostics_;
  }
  [[nodiscard]] const HoloSelectorDiagnostics& last_holo_selector_diagnostics() const {
    return last_holo_selector_diagnostics_;
  }
  [[nodiscard]] const HoloLOResult& last_holo_lo_result() const {
    return last_holo_lo_result_;
  }
  [[nodiscard]] const CellRadiationCoeffs* last_cell_radiation_coeffs() const {
    return last_cell_radiation_coeffs_valid_ ? &last_cell_radiation_coeffs_ : nullptr;
  }
  [[nodiscard]] const std::vector<double>& last_sigma_R_max() const {
    return last_sigma_R_max_;
  }
  void set_last_overshoot_metrics(std::int64_t count, double max_ratio);
  [[nodiscard]] const PhotonPool& photon_pool() const { return pool_; }
  [[nodiscard]] PhotonPool& photon_pool() { return pool_; }
  void restore_photon_pool(PhotonPool&& pool);

  // Backward-compatible no-op overload used by older call sites.
  void transport_step(double dt);

 private:
  void ensure_diffusion_face_current_buffers(int n_cells, int n_groups);
  void ensure_diffusion_energy_buffer(int n_cells, int n_groups);
  void classify_diffusion_cells(core::State& state,
                                const core::Config& cfg,
                                const PlanckTable& planck,
                                const std::vector<double>& sigma_R,
                                const std::vector<double>& node_r,
                                int n_cells,
                                int n_groups,
                                double dt);

  struct DeviceOpacityTable {
    double* d_log_temps = nullptr;
    double* d_log_numdens = nullptr;
    double* d_kappa_PA = nullptr;
    double* d_kappa_PE = nullptr;
    double* d_kappa_R = nullptr;
    int ntemp = 0, ndens = 0, ngroups = 0;
    double log_T_min = 0.0, log_T_max = 0.0;
    double log_ni_min = 0.0, log_ni_max = 0.0;
    double T_min = 0.0, T_max = 0.0;
    std::size_t host_generation = 0;
    DeviceOpacityTable() = default;
    DeviceOpacityTable(const DeviceOpacityTable&) = delete;
    DeviceOpacityTable& operator=(const DeviceOpacityTable&) = delete;
    DeviceOpacityTable(DeviceOpacityTable&& other) noexcept;
    DeviceOpacityTable& operator=(DeviceOpacityTable&& other) noexcept;

    bool needs_upload(std::size_t gen) const {
      return host_generation != gen || d_kappa_PA == nullptr;
    }
    void upload(const materials::IonmixOpacityData& host);
    void release();
    [[nodiscard]] materials::IonmixOpacityDeviceView view() const;
  };

  struct PGRWTables {
    double* d_leak_inv_cdf = nullptr;  // [1024] theta values
    double* d_leak_cdf_xi = nullptr;   // [1024] xi values
    double* d_pos_cdf = nullptr;       // [64 * 128] 2D position CDF
    int leak_table_size = 1024;
    int pos_theta_bins = 64;
    int pos_rho_bins = 128;
    double theta_max = 3.0;
    bool initialized = false;

    PGRWTables() = default;
    PGRWTables(const PGRWTables&) = delete;
    PGRWTables& operator=(const PGRWTables&) = delete;
    PGRWTables(PGRWTables&& other) noexcept;
    PGRWTables& operator=(PGRWTables&& other) noexcept;

    void initialize();
    void release();
  };

  struct PersistentCoeffBuffers {
    double* d_sigma_a = nullptr;
    double* d_sigma_R = nullptr;
    double* d_f = nullptr;
    double* d_sigma_a_eff = nullptr;
    double* d_sigma_s_eff = nullptr;
    double* d_eta_cdf = nullptr;
    double* d_emission_bias_cdf = nullptr;
    double* d_lambda_raw = nullptr;  // [n_cells], NLTE diagnostic scalar (sigma_p_em)
    double* d_eta = nullptr;         // [n_cells * n_groups]
    int* d_g_diff_end = nullptr;     // [n_cells], exclusive PGRW diffusive group cutoff
    double* d_sigma_a_bar = nullptr; // [n_cells], PGRW collapsed absorption
    double* d_sigma_t_bar = nullptr; // [n_cells], PGRW collapsed total opacity
    double* d_D_pgrw = nullptr;      // [n_cells], PGRW diffusion coefficient
    double* d_gamma_pgrw = nullptr;  // [n_cells], PGRW diffusive emission fraction
    double* d_rad_E_tally = nullptr;
    double* d_E_escape = nullptr;
    double* d_E_numerical_loss = nullptr;
    double* d_group_bounds = nullptr;
    double* d_cell_dx = nullptr;
    void* d_error_flags = nullptr;
    unsigned long long* d_imc_absorbed = nullptr;
    unsigned long long* d_imc_escaped = nullptr;
    unsigned long long* d_cnt_boundary = nullptr;
    unsigned long long* d_cnt_scatter = nullptr;
    unsigned long long* d_cnt_census = nullptr;
    unsigned long long* d_cnt_absorb_kill = nullptr;
    unsigned long long* d_cnt_absorb_survive = nullptr;
    unsigned long long* d_cnt_roulette_kill = nullptr;
    unsigned long long* d_diffusion_interface_kills = nullptr;
    unsigned long long* d_interface_transitions = nullptr;
    unsigned long long* d_interface_reflections = nullptr;
    unsigned long long* d_conversion_prob_violations = nullptr;

    int n_cells = 0;
    int n_groups = 0;

    PersistentCoeffBuffers() = default;
    PersistentCoeffBuffers(const PersistentCoeffBuffers&) = delete;
    PersistentCoeffBuffers& operator=(const PersistentCoeffBuffers&) = delete;
    PersistentCoeffBuffers(PersistentCoeffBuffers&& other) noexcept;
    PersistentCoeffBuffers& operator=(PersistentCoeffBuffers&& other) noexcept;

    void ensure(int nc, int ng);
    void release();
  };

  PhotonPool pool_;
  PhotonPool difference_residual_pool_;
  bool pool_initialized_ = false;
  PersistentCoeffBuffers coeff_buf_;
  PlanckTable planck_cache_;
  std::vector<double> planck_cache_bounds_eV_;
  int planck_cache_n_groups_ = 0;
  int planck_cache_n_T_ = 0;
  double planck_cache_T_min_eV_ = 0.0;
  double planck_cache_T_max_eV_ = 0.0;
  bool planck_cache_valid_ = false;
  PGRWTables pgrw_tables_;
  core::CellField1D sloc_abs_wr_;
  core::CellField1D sloc_abs_wr2_;
  core::CellField1D sloc_abs_E_;
  core::CellField1D sloc_mean_r_;
  core::CellField1D sloc_sigma_;
  core::CellField1D sloc_alpha_;
  core::CellField1D sloc_prev_E_;
  double escaped_energy_total_ = 0.0;
  double last_numerical_loss_step_ = 0.0;
  double last_thermal_lost_step_ = 0.0;
  double last_marshak_in_step_ = 0.0;
  double last_volume_source_step_ = 0.0;
  std::uint64_t last_interface_transitions_ = 0;
  std::uint64_t last_interface_reflections_ = 0;
  std::uint64_t last_conversion_prob_violations_ = 0;
  std::uint64_t last_cnt_boundary_ = 0;
  std::uint64_t last_cnt_scatter_ = 0;
  std::uint64_t last_cnt_census_ = 0;
  std::uint64_t last_cnt_absorb_kill_ = 0;
  std::uint64_t last_cnt_absorb_survive_ = 0;
  std::uint64_t last_cnt_roulette_kill_ = 0;
  double last_f_min_ = 1.0;
  double last_f_mean_ = 1.0;
  double last_f_p95_ = 1.0;
  int last_tau_gt1_ = 0;
  int last_tau_gt2_ = 0;
  int last_tau_gt3_ = 0;
  int last_tau_gt4_ = 0;
  std::int64_t last_ddmc_mode_count_ = 0;
  std::int64_t last_imc_mode_count_ = 0;
  std::int64_t last_n_total_ = 0;
  std::int64_t last_n_imc_particles_ = 0;
  std::int64_t last_n_ddmc_particles_ = 0;
  std::int64_t last_n_census_ = 0;
  std::int64_t last_n_absorbed_ = 0;
  std::int64_t last_n_escaped_ = 0;
  double last_ddmc_fraction_ = 0.0;
  double last_weight_min_ = 0.0;
  double last_weight_mean_ = 0.0;
  double last_weight_max_ = 0.0;
  std::int64_t last_overshoot_count_ = 0;
  double last_overshoot_max_ = 0.0;
  std::int64_t last_mmatrix_violations_ = 0;
  std::int64_t last_mmatrix_fallback_count_ = 0;
  std::int64_t last_omega_below_threshold_ = 0;
  std::int64_t last_ddmc_to_imc_conversions_ = 0;
  std::vector<TransportMode> prev_ddmc_mode_;
  std::vector<double> prev_ddmc_tau_;
  std::vector<std::uint8_t> hysteresis_hold_count_;
  std::int64_t last_switches_imc_to_ddmc_ = 0;
  std::int64_t last_switches_ddmc_to_imc_ = 0;
  ReferenceFieldDiagnostics last_reference_field_diagnostics_{};
  HoloSelectorDiagnostics last_holo_selector_diagnostics_{};
  HoloLOResult last_holo_lo_result_{};
  std::vector<std::uint8_t> diff_cell_;
  std::vector<std::uint8_t> diff_cell_prev_;
  std::vector<std::uint8_t> diff_hold_;
  std::vector<std::uint8_t> diff_guard_cell_;
  std::vector<double> diff_tau_R_;
  std::vector<double> diff_reduced_flux_;
  std::vector<double> diff_vol_;
  std::vector<double> previous_reference_U_;
  parallel::DeviceArray previous_reference_U_device_;
  bool previous_reference_U_device_valid_ = false;
  parallel::DeviceArray difference_residual_workspace_;
  parallel::DeviceArray difference_residual_scan_workspace_;
  parallel::DeviceArray diff_ref_face_delta_U_;
  parallel::DeviceArray diff_ref_face_current_;
  parallel::DeviceArray diff_U_ref_end_;
  parallel::DeviceArray diff_E_ref_avg_;
  parallel::DeviceArray diff_ref_boundary_workspace_;
  parallel::DeviceArray diff_E_;  // [n_cells * G] radiation energy density [erg/cm^3]
  parallel::DeviceArray face_current_prev_;
  parallel::DeviceArray face_current_step_;
  parallel::DeviceArray face_current_in_;
  parallel::DeviceArray face_current_out_;
  double face_current_prev_dt_ = 0.0;
  // Predictive population controller state (affine emission model)
  double pop_ctrl_rem_ema_ = 0.006;  // removal fraction EMA: (N_start + E_t - S_t) / (N_start + E_t)
  double pop_ctrl_b_scaled_ema_ = 0.0; // ppcg-proportional emission rate EMA: E_ctrl / ppcg_t
  int pop_ctrl_ppcg_applied_ = -1;   // last applied ppcg override (-1 = not yet initialized)
  double pop_ctrl_ppcg_accum_ = 0.0; // deterministic dithering accumulator for fractional ppcg
  double last_rad_momentum_deposition_ = 0.0;
  core::DeviceErrorFlags last_device_error_flags_{};
  CellRadiationCoeffs last_cell_radiation_coeffs_;
  bool last_cell_radiation_coeffs_valid_ = false;
  std::vector<double> last_sigma_R_max_;
  // NOTE (C30): nlte_table_ cache is not thread-safe. Current usage is
  // single-threaded (one IMC instance per MPI rank), but MPI+threads or
  // async transport would require mutex protection.
  std::unique_ptr<materials::IonmixOpacityData> nlte_table_;
  bool nlte_table_loaded_ = false;
  std::string nlte_table_file_;
  DeviceOpacityTable device_opacity_table_;
  std::size_t nlte_table_generation_ = 0;
};

}  // namespace tenryu::radiation
