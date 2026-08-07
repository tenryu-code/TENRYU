#pragma once

#include <array>
#include <string>
#include <utility>
#include <vector>

namespace tenryu::laser::port_geom {

struct PortFrame {
  // Rows are unit vectors in the lab frame: e1, e2, b (beam axis).
  // R maps beam-frame components to lab:
  // v_lab = e1 * v1 + e2 * v2 + b * v3.
  std::array<double, 3> e1;
  std::array<double, 3> e2;
  std::array<double, 3> b;
};

// Given the unit beam axis b, choose a = z_hat when |b_z| < 0.9 and
// a = x_hat otherwise, then form
//   e1 = (a - (a . b) b) / |a - (a . b) b|,  e2 = b x e1.
// A roll rho = roll_deg * pi / 180 about b produces
//   e1' = cos(rho) e1 + sin(rho) e2,
//   e2' = -sin(rho) e1 + cos(rho) e2.
// Precondition: ||axis_unit| - 1| <= 1e-12.
PortFrame build_port_frame(const std::array<double, 3>& axis_unit,
                           double roll_deg);

struct Port {
  int port_id;
  std::array<double, 3> axis;
  double roll_deg;
  double power_weight;
  double delta_lambda_nm;
  std::string beam_class;
};

struct PortTable {
  std::vector<Port> ports;                  // sorted ascending by port_id
  std::vector<PortFrame> frames;            // parallel to ports
  std::vector<std::pair<int, int>> pairs;   // canonical (i,j), i < j by index
  std::vector<double> delta_omega;          // signed per pair [rad/s]
  std::vector<double> axis_dot;             // b_i . b_j per pair
};

// Sort ports by port_id, assert that ids are unique, and enumerate pairs in
// lexicographic index order. For lambda_i = (lambda0_nm + delta_lambda_i) *
// 1e-7 cm, each signed detuning is
//   delta_omega_ij = 2 pi c_light (1 / lambda_j - 1 / lambda_i).
// The parallel axis entry is axis_dot_ij = b_i . b_j.
PortTable build_port_table(std::vector<Port> ports, double lambda0_nm);

struct AngularGrid {
  // Product quadrature on the unit sphere. mu and wmu are n_mu
  // Gauss-Legendre nodes and weights on [-1,1]: P_n(mu_i) = 0 and
  // wmu_i = 2 / ((1-mu_i^2) (P_n'(mu_i))^2). The azimuth nodes are
  // phi_j = 2 pi (j + 1/2) / n_phi with uniform weight 2 pi / n_phi.
  std::vector<double> mu;
  std::vector<double> wmu;
  std::vector<double> phi;
};

AngularGrid build_angular_grid(int n_mu, int n_phi);

struct BeamAngularProfile {
  // Axisymmetric profile samples I[k] = I1(mu[k]), with strictly ascending
  // mu and size at least two. Between samples,
  // I1(x) = I[k] + (I[k+1] - I[k]) (x-mu[k])/(mu[k+1]-mu[k]);
  // values outside the table are clamped to the nearest endpoint.
  std::vector<double> mu;
  std::vector<double> I;
};

struct IlluminationResult {
  double f_illum2;
  double f_union;
  double I_mean;
  double I_max;
};

// At Omega = (sqrt(1-mu^2) cos(phi), sqrt(1-mu^2) sin(phi), mu),
//   I_tot(Omega) = sum_i power_weight_i I1(b_i . Omega).
// With W = wmu * (2 pi / n_phi), the returned metrics are
//   I_mean  = sum(W I_tot) / (4 pi),
//   I_max   = max I_tot,
//   f_illum2 = (sum(W I_tot))^2 / (4 pi sum(W I_tot^2)),
//   f_union = sum(W for I_tot > I_cut) / (4 pi).
// Accumulation order is mu outer, phi inner, ports innermost.
IlluminationResult illumination_metrics(const PortTable& table,
                                        const BeamAngularProfile& profile,
                                        const AngularGrid& grid,
                                        double I_cut);

struct CommonWaveInput {
  // Local refracted unit propagation directions and intensities per port.
  // The parallel arrays are aligned with PortTable::ports.
  std::vector<std::array<double, 3>> k_hat;
  std::vector<double> I;
};

struct CommonWaveResult {
  double I_cw;       // max cluster member sum
  double I_lower;    // max_i I_i
  double I_upper;    // sum_i I_i
  int n_sigma;       // member count of the winning cluster
  int axis_index;    // winning candidate-axis index
};

// For each supplied unit candidate axis q_l, compute
//   Theta_i = acos(clamp(k_hat_i . q_l, -1, 1)).
// The candidate cone angles are the ascending unique Theta_i values, with
// values within (delta_theta_deg*pi/180)*1e-3 deduplicated. A beam is a
// member of candidate cone m when
//   |Theta_i - Theta_m| <= delta_theta_deg*pi/180,
// and I_cluster(l,m) = sum(member i) I_i. I_cw is the maximum cluster sum;
// exact ties keep the earliest (l,m) in the fixed iteration order.
CommonWaveResult common_wave_cluster(
    const CommonWaveInput& input,
    const std::vector<std::array<double, 3>>& candidate_axes,
    double delta_theta_deg);

// Return r_hat first, followed in canonical (i,j) order by
//   q_ij = (k_hat_i + k_hat_j) / |k_hat_i + k_hat_j|.
// Pairs with |k_hat_i + k_hat_j| < 1e-6 are skipped. A candidate q is
// discarded when an earlier q_old satisfies
//   |q . q_old| > 1 - 1e-10.
std::vector<std::array<double, 3>> build_candidate_axes(
    const CommonWaveInput& input, const std::array<double, 3>& r_hat);

}  // namespace tenryu::laser::port_geom
