#include "laser/cbet_stage_host.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace tenryu::laser {

CbetVizHostReference cbet_viz_fields_host_reference(const std::vector<double>& dQ,
                                                    const std::vector<double>& iaw_cell,
                                                    const std::vector<double>& cell_vol, const int G,
                                                    const int n_branches, const int n_bins) {
  // Former laser.cu fill_cbet_viz_fields.
  CbetVizHostReference out;
  const std::size_t n_cells = cell_vol.size();
  out.gross.assign(n_cells, 0.0);
  out.net_inbound.assign(n_cells, 0.0);
  constexpr double kClosureTiny = 1.0e-300;
  for (std::size_t cell = 0; cell < n_cells; ++cell) {
    double sum = 0.0;
    double sum_abs = 0.0;
    double sum_inbound = 0.0;
    const std::size_t row = cell * static_cast<std::size_t>(G);
    for (int g = 0; g < G; ++g) {
      const double dq = dQ[row + static_cast<std::size_t>(g)];
      const double dq_abs = std::abs(dq);
      out.dq_abs_max = std::max(out.dq_abs_max, dq_abs);
      out.dq_abs_sum += dq_abs;
      sum += dq;
      sum_abs += dq_abs;
      const int branch = (g % (n_branches * n_bins)) / n_bins;
      if (branch == 0) {
        sum_inbound += dq;
      }
    }
    const double iaw = iaw_cell.empty() ? 0.0 : iaw_cell[cell];
    if (!(std::abs(sum + iaw) <= 1.0e-9 * std::max(sum_abs + std::abs(iaw), kClosureTiny))) {
      out.closure_ok = false;
    }
    if (!(std::isfinite(cell_vol[cell]) && cell_vol[cell] > 0.0)) {
      out.volume_ok = false;
    }
    out.gross[cell] = 0.5 * sum_abs / cell_vol[cell];
    out.net_inbound[cell] = sum_inbound / cell_vol[cell];
  }
  return out;
}

std::vector<double> ps_outgoing_power_host_reference(const std::vector<std::int32_t>& rec_count,
                                                     const std::vector<std::int64_t>& ray_rec_offset,
                                                     const std::vector<double>& rec_w_ps,
                                                     const int n_ports, const int n_rays,
                                                     const int cap_per_ray, bool* offsets_ok) {
  // Former laser.cu fill_port_section_outgoing_power.
  const std::size_t records_per_port =
      static_cast<std::size_t>(n_rays) * static_cast<std::size_t>(cap_per_ray);
  std::vector<double> outgoing(static_cast<std::size_t>(n_ports), 0.0);
  bool ok = true;
  for (int port = 0; port < n_ports; ++port) {
    const double* rec_w_port = rec_w_ps.data() + static_cast<std::size_t>(port) * records_per_port;
    double sum = 0.0;
    for (int ray = 0; ray < n_rays; ++ray) {
      const std::int64_t offset_count = ray_rec_offset[static_cast<std::size_t>(ray) + 1U] -
                                        ray_rec_offset[static_cast<std::size_t>(ray)];
      if (offset_count != static_cast<std::int64_t>(rec_count[static_cast<std::size_t>(ray)])) {
        ok = false;
      }
      const int count = rec_count[static_cast<std::size_t>(ray)];
      if (count <= 0) {
        continue;
      }
      const std::size_t final_record = static_cast<std::size_t>(ray) * static_cast<std::size_t>(cap_per_ray) +
                                       static_cast<std::size_t>(count - 1);
      sum += rec_w_port[final_record];
    }
    outgoing[static_cast<std::size_t>(port)] = sum;
  }
  if (offsets_ok != nullptr) {
    *offsets_ok = ok;
  }
  return outgoing;
}

CaptureHostReference ps_capture_host_reference(const std::vector<double>& stage,
                                               const std::vector<std::int32_t>& capture_order,
                                               const std::vector<double>& one_minus_eta,
                                               const std::vector<double>& cell_r, const int n_ports,
                                               const int n_rays, const int n_channels,
                                               const int n_config) {
  // Former capture loop of laser.cu laser_step (port_section with hot-electron capture).
  CaptureHostReference out;
  out.pcross.assign(static_cast<std::size_t>(n_ports) * static_cast<std::size_t>(n_config), 0.0);
  out.sum_P.assign(static_cast<std::size_t>(n_config), 0.0);
  out.sum_Pr.assign(static_cast<std::size_t>(n_config), 0.0);
  out.sum_Pmu.assign(static_cast<std::size_t>(n_config), 0.0);
  for (int port = 0; port < n_ports; ++port) {
    for (int ray = 0; ray < n_rays; ++ray) {
      for (int k = 0; k < n_channels; ++k) {
        const std::size_t row = ((static_cast<std::size_t>(port) * static_cast<std::size_t>(n_rays) +
                                  static_cast<std::size_t>(ray)) *
                                     static_cast<std::size_t>(n_channels) +
                                 static_cast<std::size_t>(k)) *
                                4U;
        const double* const capture = stage.data() + row;
        if (!(capture[0] > 0.5) || !(capture[1] > 0.0)) {
          continue;
        }
        const int config_index = capture_order[static_cast<std::size_t>(k)];
        const int cell = static_cast<int>(capture[3]);
        if (config_index < 0 || config_index >= n_config || cell < 0 ||
            static_cast<std::size_t>(cell) >= cell_r.size()) {
          out.indices_ok = false;
          continue;
        }
        const std::size_t channel = static_cast<std::size_t>(config_index);
        out.pcross[static_cast<std::size_t>(port) * static_cast<std::size_t>(n_config) + channel] +=
            capture[1];
        out.sum_P[channel] += capture[1];
        out.sum_Pr[channel] += cell_r[static_cast<std::size_t>(cell)] * capture[1];
        out.sum_Pmu[channel] += capture[2] * capture[1];
        out.banked += capture[1] * (1.0 - one_minus_eta[static_cast<std::size_t>(k)]);
      }
    }
  }
  return out;
}

}  // namespace tenryu::laser
