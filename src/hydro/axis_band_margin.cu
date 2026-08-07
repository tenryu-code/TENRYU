#include "hydro/axis_band_margin.cuh"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "core/error.hpp"
#include "core/state.hpp"
#include "hydro/rz_quad_volume.cuh"
#include "parallel/reduction.hpp"

namespace tenryu::hydro::ale {
namespace {

constexpr double kPi =
    3.141592653589793238462643383279502884197169399375105820974;

template <typename Field>
std::vector<double> copy_host_field(const Field& field, const char* name) {
  std::vector<double> out;
  field.copy_to_host(out);
  TENRYU_ASSERT(out.size() == field.size(),
                (std::string("axis band margin copy failed: ") + name).c_str());
  return out;
}

std::size_t node_idx(const int i, const int j, const int nz) {
  return static_cast<std::size_t>(i * (nz + 1) + j);
}

double cell_mid_dz(const std::vector<double>& x_z,
                   const int i,
                   const int j,
                   const int nz) {
  const double z_bot =
      0.5 * (x_z[node_idx(i, j, nz)] + x_z[node_idx(i + 1, j, nz)]);
  const double z_top =
      0.5 * (x_z[node_idx(i, j + 1, nz)] + x_z[node_idx(i + 1, j + 1, nz)]);
  return z_top - z_bot;
}

}  // namespace

AxisBandMarginResult compute_row_K_margin(
    const core::State& state, const int K,
    const parallel::Reduction* reduction) {
  AxisBandMarginResult result;
  result.K = K;

  if (state.mesh.dim != 2) {
    return result;
  }

  const int nr = state.mesh.topo.nr;
  const int nz = state.mesh.topo.nz;
  TENRYU_ASSERT(K >= 1, "axis band margin requires K >= 1");
  TENRYU_ASSERT(K < nr, "axis band margin requires K < mesh nr");
  TENRYU_ASSERT(nz > 0, "axis band margin requires mesh nz > 0");

  const std::size_t node_count =
      static_cast<std::size_t>(nr + 1) * static_cast<std::size_t>(nz + 1);
  TENRYU_ASSERT(state.x_r.size() >= node_count,
                "axis band margin x_r field too small");
  TENRYU_ASSERT(state.x_z.size() >= node_count,
                "axis band margin x_z field too small");

  const std::vector<double> x_r = copy_host_field(state.x_r, "x_r");
  const std::vector<double> x_z = copy_host_field(state.x_z, "x_z");

  double min_margin = std::numeric_limits<double>::infinity();
  int argmin_j = -1;
  double min_v_actual = 0.0;
  double min_v_target = 0.0;
  double min_R_K = 0.0;
  bool invalid = false;

  const int i = K - 1;
  for (int j = 0; j < nz; ++j) {
    const std::size_t n00 = node_idx(i, j, nz);
    const std::size_t n10 = node_idx(i + 1, j, nz);
    const std::size_t n11 = node_idx(i + 1, j + 1, nz);
    const std::size_t n01 = node_idx(i, j + 1, nz);

    const double v_actual = detail::rz_signed_quad_volume(
        x_r[n00], x_z[n00], x_r[n10], x_z[n10], x_r[n11], x_z[n11],
        x_r[n01], x_z[n01]);
    const double R_K = 0.5 * (x_r[n10] + x_r[n11]);
    const double dz = cell_mid_dz(x_z, i, j, nz);
    const double v_target = (kPi * R_K * R_K / static_cast<double>(K)) * dz;
    const double margin = v_actual / v_target;

    if (!std::isfinite(v_actual) || !std::isfinite(v_target) ||
        !std::isfinite(R_K) || !std::isfinite(margin) ||
        !(v_target > 0.0)) {
      invalid = true;
      continue;
    }

    if (margin < min_margin) {
      min_margin = margin;
      argmin_j = j;
      min_v_actual = v_actual;
      min_v_target = v_target;
      min_R_K = R_K;
    }
  }

  double invalid_global = invalid ? 1.0 : 0.0;
  if (reduction != nullptr) {
    invalid_global = reduction->allreduce_max(invalid_global);
    min_margin = reduction->allreduce_min(min_margin);
  }

  if (invalid_global > 0.5 || !std::isfinite(min_margin) || argmin_j < 0) {
    result.valid = false;
    return result;
  }

  result.valid = true;
  result.row_K_margin = min_margin;
  result.argmin_j = argmin_j;
  result.v_actual_min = min_v_actual;
  result.v_target_min = min_v_target;
  result.R_K_min = min_R_K;
  return result;
}

AxisBandKSelection select_axis_band_K(
    const core::State& state, const int K_initial, const int K_max,
    const double margin_threshold,
    const parallel::Reduction* reduction) {
  TENRYU_ASSERT(K_initial >= 1,
                "axis band K selection requires K_initial >= 1");
  TENRYU_ASSERT(K_max >= K_initial,
                "axis band K selection requires K_max >= K_initial");

  AxisBandKSelection selection;
  selection.K_chosen = K_max + 1;
  selection.per_K_margins.reserve(
      static_cast<std::size_t>(K_max - K_initial + 1));

  for (int K = K_initial; K <= K_max; ++K) {
    const AxisBandMarginResult result =
        compute_row_K_margin(state, K, reduction);
    const double margin = result.valid ? result.row_K_margin : 0.0;
    selection.per_K_margins.push_back(margin);
    if (!selection.any_K_healthy && result.valid &&
        margin >= margin_threshold) {
      selection.K_chosen = K;
      selection.any_K_healthy = true;
    }
  }

  return selection;
}

}  // namespace tenryu::hydro::ale
