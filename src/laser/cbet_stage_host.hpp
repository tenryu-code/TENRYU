#pragma once

// Host references of the 1D CBET stages that run on the device (cbet_stage_gpu.cuh): the loops the
// laser step ran on the host before (laser.cu fill_cbet_viz_fields, fill_port_section_outgoing_power
// and the hot-electron capture loop), on host copies of the workspace arrays. Kept for the tests
// that compare the device stages with them; the production step does not call them.

#include <cstdint>
#include <vector>

namespace tenryu::laser {

struct CbetVizHostReference {
  std::vector<double> gross;        // [n_cells] 0.5 sum_g |dQ| / V
  std::vector<double> net_inbound;  // [n_cells] sum over inbound groups of dQ / V
  double dq_abs_max = 0.0;
  double dq_abs_sum = 0.0;          // sum over (cell, group) in that order
  bool closure_ok = true;
  bool volume_ok = true;
};

// dQ [n_cells * G] (group fastest), iaw_cell [n_cells] (empty outside port_section), cell_vol.
CbetVizHostReference cbet_viz_fields_host_reference(const std::vector<double>& dQ,
                                                    const std::vector<double>& iaw_cell,
                                                    const std::vector<double>& cell_vol, int G,
                                                    int n_branches, int n_bins);

// rec_w_ps [n_ports * n_rays * cap_per_ray]; ray_rec_offset [n_rays + 1]. Returns the outgoing power
// per port; offsets_ok reports whether every ray's offsets agree with its count.
std::vector<double> ps_outgoing_power_host_reference(const std::vector<std::int32_t>& rec_count,
                                                     const std::vector<std::int64_t>& ray_rec_offset,
                                                     const std::vector<double>& rec_w_ps,
                                                     int n_ports, int n_rays, int cap_per_ray,
                                                     bool* offsets_ok);

struct CaptureHostReference {
  std::vector<double> pcross;   // [n_ports * n_config]
  std::vector<double> sum_P;    // [n_config]
  std::vector<double> sum_Pr;
  std::vector<double> sum_Pmu;
  double banked = 0.0;
  bool indices_ok = true;
};

// stage [n_ports * n_rays * n_channels * 4] rows {hit, P_cross, mu, cell}; capture_order and
// one_minus_eta [n_channels]; cell_r [n_cells] the cell centres.
CaptureHostReference ps_capture_host_reference(const std::vector<double>& stage,
                                               const std::vector<std::int32_t>& capture_order,
                                               const std::vector<double>& one_minus_eta,
                                               const std::vector<double>& cell_r, int n_ports,
                                               int n_rays, int n_channels, int n_config);

}  // namespace tenryu::laser
