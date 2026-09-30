// Retired with the Monte Carlo radiation on 2026-09-29: the test case of tests/radiation/test_sn_transport_gpu.cu (as
// it was at 5bc8f6ce3) that compared the 1D GPU sweep solve_sn_transport_1d_gpu with the CPU sweep of the HOLO S_N
// closure (radiation/sn_transport_1d.hpp); both left the build. The file's 2D test case stayed. Not built.

namespace {

std::vector<double> spherical_nodes(const int n_cells,
                                    const double r_min,
                                    const double dr) {
  std::vector<double> node(static_cast<std::size_t>(n_cells + 1), 0.0);
  for (int i = 0; i <= n_cells; ++i) {
    node[static_cast<std::size_t>(i)] = r_min + dr * static_cast<double>(i);
  }
  return node;
}

std::vector<double> spherical_volumes(const std::vector<double>& node) {
  constexpr double four_pi_over_three = 4.18879020478639098462;
  std::vector<double> vol(node.size() - 1U, 0.0);
  for (std::size_t i = 0; i < vol.size(); ++i) {
    vol[i] = four_pi_over_three *
             (node[i + 1U] * node[i + 1U] * node[i + 1U] -
              node[i] * node[i] * node[i]);
  }
  return vol;
}

}  // namespace

TEST_CASE("GPU 1D SN transport matches CPU spherical sweep",
          "[radiation][sn][gpu]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n_cells = 5;
  constexpr int n_groups = 2;
  const std::vector<double> node = spherical_nodes(n_cells, 0.0, 0.2);
  const std::vector<double> vol = spherical_volumes(node);
  std::vector<double> sigma_a(static_cast<std::size_t>(n_cells * n_groups), 0.0);
  std::vector<double> sigma_s(static_cast<std::size_t>(n_cells * n_groups), 0.0);
  std::vector<double> source(static_cast<std::size_t>(n_cells * n_groups), 0.0);
  for (int c = 0; c < n_cells; ++c) {
    for (int g = 0; g < n_groups; ++g) {
      const std::size_t idx = static_cast<std::size_t>(c * n_groups + g);
      sigma_a[idx] = 0.3 + 0.04 * static_cast<double>(c) + 0.02 * static_cast<double>(g);
      sigma_s[idx] = 0.05 + 0.01 * static_cast<double>(g);
      source[idx] = 1.0e13 * (1.0 + 0.1 * static_cast<double>(c) +
                              0.2 * static_cast<double>(g));
    }
  }

  tenryu::radiation::SNTransport1DConfig cpu_cfg{};
  cpu_cfg.n_angles = 8;
  cpu_cfg.max_iterations = 40;
  cpu_cfg.convergence_tol = 1.0e-12;
  const tenryu::radiation::SNTransport1DResult cpu =
      tenryu::radiation::solve_sn_transport_1d(sigma_a.data(),
                                               sigma_s.data(),
                                               source.data(),
                                               node.data(),
                                               vol.data(),
                                               n_cells,
                                               n_groups,
                                               cpu_cfg);
  REQUIRE(cpu.converged);

  tenryu::core::GroupField1D d_sigma_a(sigma_a.size());
  tenryu::core::GroupField1D d_sigma_s(sigma_s.size());
  tenryu::core::GroupField1D d_source(source.size());
  tenryu::core::NodeField1D d_node(node.size());
  tenryu::core::CellField1D d_vol(vol.size());
  tenryu::core::GroupField1D d_E(sigma_a.size());
  tenryu::core::GroupField1D d_P(sigma_a.size());
  tenryu::core::GroupField1D d_chi(sigma_a.size());
  d_sigma_a.copy_from_host(sigma_a);
  d_sigma_s.copy_from_host(sigma_s);
  d_source.copy_from_host(source);
  d_node.copy_from_host(node);
  d_vol.copy_from_host(vol);

  tenryu::radiation::SNTransportGPUConfig gpu_cfg{};
  gpu_cfg.n_angles = cpu_cfg.n_angles;
  gpu_cfg.max_iterations = cpu_cfg.max_iterations;
  gpu_cfg.convergence_tol = cpu_cfg.convergence_tol;

  tenryu::radiation::SNTransport1DGPUInputs in{};
  in.sigma_a = d_sigma_a.data();
  in.sigma_s = d_sigma_s.data();
  in.source_emission = d_source.data();
  in.node_r = d_node.data();
  in.vol = d_vol.data();
  in.E_out = d_E.data();
  in.P_rr_out = d_P.data();
  in.chi_out = d_chi.data();
  in.n_cells = n_cells;
  in.n_groups = n_groups;
  const tenryu::radiation::SNTransportGPUResult gpu =
      tenryu::radiation::solve_sn_transport_1d_gpu(in, gpu_cfg);
  cuda_check(cudaDeviceSynchronize());
  REQUIRE(gpu.converged);

  std::vector<double> E_gpu;
  std::vector<double> chi_gpu;
  d_E.copy_to_host(E_gpu);
  d_chi.copy_to_host(chi_gpu);
  for (std::size_t i = 0; i < E_gpu.size(); ++i) {
    REQUIRE(E_gpu[i] == Catch::Approx(cpu.E_sn[i]).epsilon(5.0e-13).margin(1.0e-20));
    REQUIRE(chi_gpu[i] == Catch::Approx(cpu.chi[i]).epsilon(5.0e-13).margin(1.0e-14));
  }
}
