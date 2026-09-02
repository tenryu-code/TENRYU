#include "hydro/stripe_gate.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>

#include "core/config.hpp"
#include "core/error.hpp"
#include "core/field.hpp"
#include "core/state.hpp"

namespace tenryu::hydro {
namespace {

constexpr int kColumnStride = 192;
constexpr int kShellCellCount = 288 * kColumnStride;
constexpr int kBandCount = 5;
constexpr int kBlockSize = 256;
constexpr double kPi =
    3.1415926535897932384626433832795028841971693993751;
constexpr double kRDiskCm = 1.196e-3;
constexpr double kExclusionRadiusCm = 3.0 * kRDiskCm;
// Just above the disk feather outer edge ~1.6*R_DISK = 1.9e-3,
// measured contamination boundary.
constexpr double kDetectorExclusionRadiusCm = 2.0e-3;
// Placeholder until the deterministic zero-signal floor is calibrated.
constexpr double kEpsilonS = 1.0e-300;

enum Band : int {
  N_C0 = 0,
  N_C1 = 1,
  S_C0 = 2,
  S_C1 = 3,
  EQ_REF = 4,
};

constexpr std::array<const char*, kBandCount> kBandNames = {
    "N_C0", "N_C1", "S_C0", "S_C1", "EQ_REF"};

constexpr unsigned int band_bit(const int band) {
  return 1U << static_cast<unsigned int>(band);
}

enum class StripeGateMode {
  Off,
  Scan,
  Gate,
  Invalid,
};

struct StripeGateConfig {
  StripeGateMode mode = StripeGateMode::Off;
  double rf_a = 0.0;
  double rf_b = 0.0;
};

struct StripeGateState {
  core::DeviceArray<unsigned int> band_mask;
  core::DeviceArray<unsigned int> detector_mask;
  core::DeviceArray<double> before_k;
  core::DeviceArray<double> before_vol;
  core::DeviceArray<double> contributions;
  core::DeviceArray<int> skipped;
  std::vector<unsigned int> host_band_mask;
  std::vector<unsigned int> host_detector_mask;
  std::vector<double> row_initial_radius;
  std::array<std::size_t, kBandCount> band_sizes{};
  std::array<double, kBandCount> omega{};
  std::array<double, kBandCount> dose_plus{};
  std::array<double, kBandCount> dose_minus{};
  std::size_t n_cells = 0;
  int n_rows = 0;
  unsigned long long skipped_count = 0;
  int steps_in_window = 0;
  bool masks_built = false;
  bool capture_open = false;
  bool finalized = false;
  bool front_ever_defined = false;
  bool window_entered = false;
  bool window_missed = false;
  bool nonmonotone_flag = false;
  bool have_previous_front = false;
  double previous_front = 0.0;
};

struct StripeGateReport {
  std::array<double, kBandCount> normalized_plus{};
  std::array<double, kBandCount> normalized_minus{};
  std::array<double, 4> ratios{};
  double stripe_score = 0.0;
};

const StripeGateConfig& stripe_gate_config() {
  static const StripeGateConfig config = []() {
    StripeGateConfig parsed;
    const char* raw = std::getenv("TENRYU_I1B_SGATE");
    if (raw == nullptr || raw[0] == '\0') {
      return parsed;
    }
    if (std::strcmp(raw, "scan") == 0) {
      parsed.mode = StripeGateMode::Scan;
      return parsed;
    }

    char* end_a = nullptr;
    const double rf_a = std::strtod(raw, &end_a);
    if (end_a == raw || end_a == nullptr || *end_a != ':') {
      parsed.mode = StripeGateMode::Invalid;
      return parsed;
    }
    const char* raw_b = end_a + 1;
    char* end_b = nullptr;
    const double rf_b = std::strtod(raw_b, &end_b);
    if (end_b == raw_b || end_b == nullptr || *end_b != '\0' ||
        !std::isfinite(rf_a) || !std::isfinite(rf_b) || !(rf_a > rf_b)) {
      parsed.mode = StripeGateMode::Invalid;
      return parsed;
    }
    parsed.mode = StripeGateMode::Gate;
    parsed.rf_a = rf_a;
    parsed.rf_b = rf_b;
    return parsed;
  }();
  return config;
}

bool stripe_gate_enabled() {
  return stripe_gate_config().mode != StripeGateMode::Off;
}

void validate_activation() {
  TENRYU_ASSERT(
      stripe_gate_config().mode != StripeGateMode::Invalid,
      "TENRYU_I1B_SGATE must be 'scan' or '<rfA>:<rfB>' with rfA > rfB");
}

StripeGateState& stripe_gate_state() {
  static StripeGateState state;
  return state;
}

double stripe_gate_gamma(const core::Config& cfg) {
  TENRYU_ASSERT(!cfg.materials.materials.empty(),
                "stripe gate requires at least one material");
  return cfg.materials.materials.front().ideal_gas_gamma;
}

void validate_fields(const core::State& state, const std::size_t n_cells) {
  TENRYU_ASSERT(state.rho.size() >= n_cells && state.ee.size() >= n_cells &&
                    state.ei.size() >= n_cells && state.mass.size() >= n_cells &&
                    state.vol.size() >= n_cells,
                "stripe gate requires rho/mass/ee/ei/vol field size agreement");
}

__device__ double entropy_proxy(const double rho,
                                const double ee,
                                const double ei,
                                const double gamma) {
  return (ee + ei) * pow(rho, 1.0 - gamma);
}

__global__ void capture_before_kernel(double* before_k,
                                      double* before_vol,
                                      const unsigned int* band_mask,
                                      const unsigned int* detector_mask,
                                      const double* rho,
                                      const double* ee,
                                      const double* ei,
                                      const double* vol,
                                      const double gamma,
                                      const bool capture_k,
                                      const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  before_vol[c] = 0.0;
  if (capture_k) {
    before_k[c] = 0.0;
  }
  if (band_mask[c] == 0U && detector_mask[c] == 0U) {
    return;
  }
  before_vol[c] = vol[c];
  if (capture_k && band_mask[c] != 0U) {
    before_k[c] = entropy_proxy(rho[c], ee[c], ei[c], gamma);
  }
}

__global__ void stripe_contribution_kernel(
    double* contributions,
    int* skipped,
    const unsigned int* band_mask,
    const double* before_k,
    const double* rho,
    const double* mass,
    const double* ee,
    const double* ei,
    const double gamma,
    const int n_cells) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= n_cells) {
    return;
  }
  contributions[c] = 0.0;
  skipped[c] = 0;
  if (band_mask[c] == 0U) {
    return;
  }
  if (!(rho[c] > 1.0)) {
    skipped[c] = 1;
    return;
  }
  const double after_k = entropy_proxy(rho[c], ee[c], ei[c], gamma);
  if (!(before_k[c] > 0.0) || !(after_k > 0.0) ||
      !isfinite(before_k[c]) || !isfinite(after_k)) {
    skipped[c] = 1;
    return;
  }
  contributions[c] = mass[c] * log(after_k / before_k[c]);
}

double column_solid_angle(const int column) {
  const double theta_lo = static_cast<double>(column) * kPi /
                          static_cast<double>(kColumnStride);
  const double theta_hi = static_cast<double>(column + 1) * kPi /
                          static_cast<double>(kColumnStride);
  return 2.0 * kPi * std::abs(std::cos(theta_lo) - std::cos(theta_hi));
}

void build_masks(StripeGateState& gate, const core::State& state) {
  const auto& topo = state.mesh.topo;
  gate.n_cells =
      std::min(state.rho.size(), static_cast<std::size_t>(kShellCellCount));
  TENRYU_ASSERT(state.mesh.dim == 2,
                "stripe gate requires a 2D shell mesh");
  TENRYU_ASSERT(topo.nz == kColumnStride,
                "stripe gate requires NTHETA=192");
  TENRYU_ASSERT(gate.n_cells % kColumnStride == 0,
                "stripe gate requires complete shell radial rows");
  TENRYU_ASSERT(gate.n_cells / kColumnStride >= 3,
                "stripe gate requires at least three complete shell radial rows");
  TENRYU_ASSERT(state.rho.size() >= gate.n_cells,
                "stripe gate requires state cells to match shell topology");
  TENRYU_ASSERT(state.mesh.cell_centroid_r.size() == state.rho.size() &&
                    state.mesh.cell_centroid_z.size() == state.rho.size(),
                "stripe gate requires current cell-center coordinates");
  validate_fields(state, gate.n_cells);

  if (state.step > 0) {
    core::log_warning(
        "[s-gate] first activation occurred after step 0; masks use the "
        "coordinates present at activation");
  }

  gate.n_rows = static_cast<int>(gate.n_cells / kColumnStride);
  gate.host_band_mask.assign(gate.n_cells, 0U);
  gate.host_detector_mask.assign(gate.n_cells, 0U);
  gate.row_initial_radius.assign(
      static_cast<std::size_t>(gate.n_rows),
      std::numeric_limits<double>::quiet_NaN());
  std::vector<double> row_radius_sum(static_cast<std::size_t>(gate.n_rows),
                                     0.0);
  std::vector<int> row_radius_count(static_cast<std::size_t>(gate.n_rows), 0);
  std::array<std::array<bool, kColumnStride>, kBandCount> band_columns{};

  for (std::size_t c = 0; c < gate.n_cells; ++c) {
    const int i = static_cast<int>(c / kColumnStride);
    const int j = static_cast<int>(c % kColumnStride);
    const double r_cyl = state.mesh.cell_centroid_r[c];
    const double z = state.mesh.cell_centroid_z[c];
    TENRYU_ASSERT(std::isfinite(r_cyl) && std::isfinite(z),
                  "stripe gate initial cell center must be finite");
    const double theta0 = std::atan2(r_cyl, z);
    const double s0 = std::hypot(r_cyl, z);
    const bool is_eq = std::abs(std::cos(theta0)) <= 0.25;
    if (is_eq && s0 > kDetectorExclusionRadiusCm) {
      gate.host_detector_mask[c] = 1U;
      row_radius_sum[static_cast<std::size_t>(i)] += s0;
      ++row_radius_count[static_cast<std::size_t>(i)];
    }
    if (s0 <= kExclusionRadiusCm) {
      continue;
    }

    unsigned int mask = 0U;
    if (j == 0) {
      mask |= band_bit(N_C0);
    }
    if (j == 1) {
      mask |= band_bit(N_C1);
    }
    if (j == kColumnStride - 1) {
      mask |= band_bit(S_C0);
    }
    if (j == kColumnStride - 2) {
      mask |= band_bit(S_C1);
    }
    if (is_eq) {
      mask |= band_bit(EQ_REF);
    }
    gate.host_band_mask[c] = mask;
    for (int band = 0; band < kBandCount; ++band) {
      if ((mask & band_bit(band)) != 0U) {
        ++gate.band_sizes[band];
        band_columns[band][j] = true;
      }
    }
  }

  for (int i = 0; i < gate.n_rows; ++i) {
    const int count = row_radius_count[static_cast<std::size_t>(i)];
    if (count > 0) {
      gate.row_initial_radius[static_cast<std::size_t>(i)] =
          row_radius_sum[static_cast<std::size_t>(i)] /
          static_cast<double>(count);
    }
  }
  for (int band = 0; band < kBandCount; ++band) {
    for (int j = 0; j < kColumnStride; ++j) {
      if (band_columns[band][j]) {
        gate.omega[band] += column_solid_angle(j);
      }
    }
    TENRYU_ASSERT(gate.band_sizes[band] > 0 && gate.omega[band] > 0.0,
                  "stripe gate requires every named band to be nonempty");
  }

  gate.band_mask.reset(gate.n_cells);
  gate.band_mask.copy_from_host(gate.host_band_mask);
  gate.detector_mask.reset(gate.n_cells);
  gate.detector_mask.copy_from_host(gate.host_detector_mask);
  gate.before_vol.reset(gate.n_cells);
  if (stripe_gate_config().mode == StripeGateMode::Gate) {
    gate.before_k.reset(gate.n_cells);
    gate.contributions.reset(gate.n_cells);
    gate.skipped.reset(gate.n_cells);
  }
  gate.masks_built = true;

  std::ostringstream stream;
  stream << std::scientific << std::setprecision(6) << "[s-gate] manifest";
  for (int band = 0; band < kBandCount; ++band) {
    stream << " " << kBandNames[band] << "_size=" << gate.band_sizes[band]
           << " " << kBandNames[band] << "_Omega=" << gate.omega[band];
  }
  stream << " exclusion_radius_cm=" << kExclusionRadiusCm
         << " detector_exclusion_radius_cm="
         << kDetectorExclusionRadiusCm
         << " NTHETA=" << kColumnStride;
  core::log_info(stream.str());
}

struct FrontMeasurement {
  bool defined = false;
  double radius = std::numeric_limits<double>::quiet_NaN();
  double peak = 0.0;
};

FrontMeasurement measure_front(const StripeGateState& gate,
                               const std::vector<double>& before_vol,
                               const std::vector<double>& after_vol,
                               const double dt) {
  std::vector<double> compression(static_cast<std::size_t>(gate.n_rows), 0.0);
  std::vector<double> scratch;
  scratch.reserve(kColumnStride);
  for (int i = 0; i < gate.n_rows; ++i) {
    scratch.clear();
    for (int j = 0; j < kColumnStride; ++j) {
      const std::size_t c =
          static_cast<std::size_t>(i) * kColumnStride + j;
      if (gate.host_detector_mask[c] == 0U) {
        continue;
      }
      double comp = 0.0;
      const double v_before = before_vol[c];
      const double v_after = after_vol[c];
      if (v_before > 0.0 && std::isfinite(v_before) &&
          std::isfinite(v_after)) {
        const double rate = -(v_after - v_before) / (v_before * dt);
        if (rate > 0.0 && std::isfinite(rate)) {
          comp = rate;
        }
      }
      scratch.push_back(comp);
    }
    if (!scratch.empty()) {
      const std::size_t middle = (scratch.size() - 1U) / 2U;
      std::nth_element(scratch.begin(), scratch.begin() + middle,
                       scratch.end());
      compression[static_cast<std::size_t>(i)] = scratch[middle];
    }
  }

  int peak_begin = 0;
  double peak_sum = 0.0;
  for (int i = 0; i <= gate.n_rows - 3; ++i) {
    const double sum = compression[static_cast<std::size_t>(i)] +
                       compression[static_cast<std::size_t>(i + 1)] +
                       compression[static_cast<std::size_t>(i + 2)];
    if (sum > peak_sum) {
      peak_sum = sum;
      peak_begin = i;
    }
  }

  FrontMeasurement measurement;
  measurement.peak = peak_sum;
  if (!(peak_sum > 0.0) || !std::isfinite(peak_sum)) {
    return measurement;
  }
  double weighted_radius = 0.0;
  for (int offset = 0; offset < 3; ++offset) {
    const int i = peak_begin + offset;
    const double weight = compression[static_cast<std::size_t>(i)];
    if (weight > 0.0) {
      const double radius = gate.row_initial_radius[static_cast<std::size_t>(i)];
      if (!std::isfinite(radius)) {
        return measurement;
      }
      weighted_radius += weight * radius;
    }
  }
  measurement.radius = weighted_radius / peak_sum;
  measurement.defined = std::isfinite(measurement.radius);
  return measurement;
}

void log_front(const core::State& state,
               const double t_after,
               const FrontMeasurement& front) {
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(17)
         << "[s-gate] step=" << state.step << " t=" << t_after << " r_f=";
  if (front.defined) {
    stream << front.radius;
  } else {
    stream << "nan";
  }
  stream << " C_peak=" << front.peak;
  core::log_info(stream.str());
}

double gate_weight(StripeGateState& gate, const double front_radius) {
  const StripeGateConfig& config = stripe_gate_config();
  const bool in_window =
      front_radius >= config.rf_b && front_radius <= config.rf_a;
  double weight = 0.0;

  if (!gate.window_entered && front_radius < config.rf_b) {
    gate.window_missed = true;
  }

  if (in_window) {
    weight = 1.0;
    if (!gate.window_entered && gate.have_previous_front &&
        gate.previous_front > config.rf_a) {
      // Weight the part after the linearly interpolated rfA entry crossing.
      weight = (config.rf_a - front_radius) /
               (gate.previous_front - front_radius);
      weight = std::clamp(weight, 0.0, 1.0);
    }
    gate.window_entered = true;
  } else if (gate.window_entered && front_radius < config.rf_b &&
             gate.have_previous_front && gate.previous_front >= config.rf_b &&
             gate.previous_front > front_radius) {
    // Analogously, retain the part before the interpolated rfB exit crossing.
    weight = (gate.previous_front - config.rf_b) /
             (gate.previous_front - front_radius);
    weight = std::clamp(weight, 0.0, 1.0);
  }

  if (gate.window_entered && gate.have_previous_front &&
      front_radius - gate.previous_front >
          0.05 * (config.rf_a - config.rf_b)) {
    gate.nonmonotone_flag = true;
  }
  gate.previous_front = front_radius;
  gate.have_previous_front = true;
  return weight;
}

void accumulate_dose(StripeGateState& gate,
                     const std::vector<double>& contributions,
                     const double weight) {
  std::array<double, kBandCount> step_plus{};
  std::array<double, kBandCount> step_minus{};
  for (std::size_t c = 0; c < gate.n_cells; ++c) {
    const double ds1 = contributions[c];
    const double plus = std::max(ds1, 0.0);
    const double minus = std::max(-ds1, 0.0);
    const unsigned int mask = gate.host_band_mask[c];
    for (int band = 0; band < kBandCount; ++band) {
      if ((mask & band_bit(band)) != 0U) {
        step_plus[band] += plus;
        step_minus[band] += minus;
      }
    }
  }
  for (int band = 0; band < kBandCount; ++band) {
    // ds1 and D retain the entropy ledger's mass*ln units.
    gate.dose_plus[band] += weight * step_plus[band];
    gate.dose_minus[band] += weight * step_minus[band];
  }
  ++gate.steps_in_window;
}

StripeGateReport compute_report(const StripeGateState& gate) {
  StripeGateReport report;
  for (int band = 0; band < kBandCount; ++band) {
    if (gate.omega[band] > 0.0) {
      // Dhat retains mass*ln per steradian in the frozen cgs/eV unit system.
      report.normalized_plus[band] =
          gate.dose_plus[band] / gate.omega[band];
      report.normalized_minus[band] =
          gate.dose_minus[band] / gate.omega[band];
    }
  }

  // Solid-angle normalization cancels in each dimensionless pole/EQ ratio.
  for (int band = 0; band < 4; ++band) {
    report.ratios[band] =
        (report.normalized_plus[band] + kEpsilonS) /
        (report.normalized_plus[EQ_REF] + kEpsilonS);
  }
  for (const double ratio : report.ratios) {
    report.stripe_score =
        std::max(report.stripe_score, std::log(std::max(1.0, ratio)));
  }
  return report;
}

}  // namespace

void stripe_gate_begin(core::State& state, const core::Config& cfg) {
  if (!stripe_gate_enabled()) {
    return;
  }
  validate_activation();
  StripeGateState& gate = stripe_gate_state();
  if (gate.finalized) {
    return;
  }
  if (!gate.masks_built) {
    build_masks(gate, state);
  }
  TENRYU_ASSERT(state.rho.size() >= gate.n_cells,
                "stripe gate cell count changed after mask construction");
  validate_fields(state, gate.n_cells);
  TENRYU_ASSERT(!gate.capture_open,
                "stripe gate begin called while capture is already open");

  const bool capture_k =
      stripe_gate_config().mode == StripeGateMode::Gate;
  const int n = static_cast<int>(gate.n_cells);
  const int blocks = (n + kBlockSize - 1) / kBlockSize;
  capture_before_kernel<<<blocks, kBlockSize>>>(
      capture_k ? gate.before_k.data() : nullptr, gate.before_vol.data(),
      gate.band_mask.data(), gate.detector_mask.data(), state.rho.data(),
      state.ee.data(), state.ei.data(), state.vol.data(),
      capture_k ? stripe_gate_gamma(cfg) : 0.0, capture_k, n);
  CUDA_CHECK(cudaGetLastError());
  gate.capture_open = true;
}

void stripe_gate_end(core::State& state,
                     const core::Config& cfg,
                     const double dt,
                     const double t_after) {
  if (!stripe_gate_enabled()) {
    return;
  }
  validate_activation();
  StripeGateState& gate = stripe_gate_state();
  if (gate.finalized) {
    return;
  }
  TENRYU_ASSERT(gate.capture_open,
                "stripe gate end called without begin");
  TENRYU_ASSERT(dt > 0.0 && std::isfinite(dt),
                "stripe gate requires a positive finite hydro dt");
  validate_fields(state, gate.n_cells);

  std::vector<double> host_contributions;
  if (stripe_gate_config().mode == StripeGateMode::Gate) {
    const int n = static_cast<int>(gate.n_cells);
    const int blocks = (n + kBlockSize - 1) / kBlockSize;
    stripe_contribution_kernel<<<blocks, kBlockSize>>>(
        gate.contributions.data(), gate.skipped.data(), gate.band_mask.data(),
        gate.before_k.data(), state.rho.data(), state.mass.data(),
        state.ee.data(), state.ei.data(), stripe_gate_gamma(cfg), n);
    CUDA_CHECK(cudaGetLastError());
    gate.contributions.copy_to_host(host_contributions);
    std::vector<int> host_skipped;
    gate.skipped.copy_to_host(host_skipped);
    for (std::size_t c = 0; c < gate.n_cells; ++c) {
      gate.skipped_count +=
          static_cast<unsigned long long>(host_skipped[c] != 0);
    }
  }

  std::vector<double> host_before_vol;
  std::vector<double> host_after_vol(gate.n_cells);
  gate.before_vol.copy_to_host(host_before_vol);
  CUDA_CHECK(cudaMemcpy(host_after_vol.data(), state.vol.data(),
                        gate.n_cells * sizeof(double), cudaMemcpyDeviceToHost));
  const FrontMeasurement front =
      measure_front(gate, host_before_vol, host_after_vol, dt);
  log_front(state, t_after, front);
  gate.capture_open = false;

  if (stripe_gate_config().mode == StripeGateMode::Scan || !front.defined) {
    return;
  }
  gate.front_ever_defined = true;
  const double weight = gate_weight(gate, front.radius);
  if (weight > 0.0) {
    accumulate_dose(gate, host_contributions, weight);
    if (!gate.finalized && gate.steps_in_window > 0 &&
        gate.steps_in_window % 200 == 0) {
      const StripeGateReport report = compute_report(gate);
      std::ostringstream stream;
      stream << std::scientific << std::setprecision(17)
             << "[s-gate] interim step=" << state.step << " t=" << t_after
             << " r_f=" << front.radius
             << " steps_in_window=" << gate.steps_in_window
             << " R_N_C0=" << report.ratios[N_C0]
             << " R_N_C1=" << report.ratios[N_C1]
             << " R_S_C0=" << report.ratios[S_C0]
             << " R_S_C1=" << report.ratios[S_C1]
             << " S_stripe=" << report.stripe_score;
      core::log_info(stream.str());
    }
  }
  if (gate.window_entered &&
      front.radius < stripe_gate_config().rf_b &&
      gate.steps_in_window > 0) {
    stripe_gate_finalize();
  }
}

void stripe_gate_finalize() {
  if (!stripe_gate_enabled()) {
    return;
  }
  validate_activation();
  if (stripe_gate_config().mode == StripeGateMode::Scan) {
    return;
  }
  StripeGateState& gate = stripe_gate_state();
  TENRYU_ASSERT(!gate.capture_open,
                "stripe gate finalized with an open capture");
  if (gate.finalized) {
    return;
  }
  gate.finalized = true;

  const StripeGateReport report = compute_report(gate);

  const bool invalid = gate.nonmonotone_flag || gate.window_missed ||
                       gate.steps_in_window == 0 ||
                       !gate.front_ever_defined;
  const char* verdict =
      invalid ? "S_INVALID"
              : (report.stripe_score <= std::log(2.0) ? "S_PASS" : "S_FAIL");
  const char* qualification =
      (!invalid && report.stripe_score <= std::log(1.25)) ? "QUAL_MET"
                                                           : "QUAL_NOT_MET";

  std::ostringstream stream;
  stream << std::scientific << std::setprecision(17)
         << "[s-gate] FINAL window_cm=[" << stripe_gate_config().rf_b << ","
         << stripe_gate_config().rf_a << "] steps_in_window="
         << gate.steps_in_window << " skipped=" << gate.skipped_count << '\n';
  for (int band = 0; band < kBandCount; ++band) {
    stream << "[s-gate] FINAL " << kBandNames[band]
           << " Dhat_plus=" << report.normalized_plus[band]
           << " Dhat_minus=" << report.normalized_minus[band] << '\n';
  }
  stream << "[s-gate] FINAL R_N_C0=" << report.ratios[N_C0]
         << " R_N_C1=" << report.ratios[N_C1]
         << " R_S_C0=" << report.ratios[S_C0]
         << " R_S_C1=" << report.ratios[S_C1] << '\n'
         << "[s-gate] FINAL S_stripe=" << report.stripe_score
         << " hard_limit_ln2=" << std::log(2.0)
         << " qualification_limit_ln1p25=" << std::log(1.25) << '\n'
         << "[s-gate] FINAL VERDICT=" << verdict
         << " QUALIFICATION=" << qualification
         << " nonmonotone_flag=" << (gate.nonmonotone_flag ? 1 : 0)
         << " window_missed=" << (gate.window_missed ? 1 : 0)
         << " front_ever_defined=" << (gate.front_ever_defined ? 1 : 0);
  core::log_info(stream.str());
}

}  // namespace tenryu::hydro
