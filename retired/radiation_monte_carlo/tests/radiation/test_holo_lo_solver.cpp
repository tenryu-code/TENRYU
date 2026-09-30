#include <cstdint>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/constants.hpp"
#include "materials/eos_table.hpp"
#include "radiation/holo_lo_solver.hpp"

namespace {

tenryu::radiation::HoloLOInputs base_inputs(std::vector<double>& E_lo,
                                            std::vector<double>& ee,
                                            std::vector<double>& Te,
                                            std::vector<double>& Pe,
                                            std::vector<double>& sigma_P,
                                            std::vector<double>& sigma_R,
                                            std::vector<double>& rho,
                                            std::vector<double>& mass,
                                            std::vector<double>& vol,
                                            std::vector<double>& node_r,
                                            std::vector<std::uint8_t>& coupled) {
  tenryu::radiation::HoloLOInputs in{};
  in.E_lo = E_lo.data();
  in.ee = ee.data();
  in.Te = Te.data();
  in.Pe = Pe.data();
  in.sigma_P = sigma_P.data();
  in.sigma_R = sigma_R.data();
  in.rho = rho.data();
  in.mass = mass.data();
  in.vol = vol.data();
  in.node_r = node_r.data();
  in.lo_coupled = coupled.data();
  in.n_cells = static_cast<int>(rho.size());
  in.n_groups = static_cast<int>(E_lo.size() / rho.size());
  in.dt = 1.0e-10;
  in.cv_e_const = 1.0e4;
  in.temperature_floor_eV = 1.0e-3;
  return in;
}

}  // namespace

TEST_CASE("HOLO LO source solve is finite-rate and conservative",
          "[radiation][holo]") {
  std::vector<double> E_lo = {100.0};
  std::vector<double> ee = {2.0e4};
  std::vector<double> Te = {2.0};
  std::vector<double> Pe = {0.0};
  std::vector<double> sigma_P = {
      0.25 / (tenryu::core::constants::c_light * 1.0e-10)};
  std::vector<double> sigma_R = {1.0e20};
  std::vector<double> rho = {1.0};
  std::vector<double> mass = {1.0};
  std::vector<double> vol = {1.0};
  std::vector<double> node_r = {0.0, 0.0};
  std::vector<std::uint8_t> core = {1U};
  std::vector<double> rad_dep = {0.0};
  std::vector<double> rad_emit = {0.0};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.rad_dep_lo = rad_dep.data();
  in.rad_emit_lo = rad_emit.data();

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  const double T2 = 2.0 * 2.0;
  const double B = tenryu::core::constants::a_eV * T2 * T2;
  const double alpha = tenryu::core::constants::c_light * sigma_P[0] * in.dt;
  const double expected_E = (100.0 + alpha * B) / (1.0 + alpha);
  REQUIRE(E_lo[0] == Catch::Approx(expected_E).epsilon(1.0e-10));
  REQUIRE(E_lo[0] < B);
  REQUIRE(result.matter_delta + result.rad_delta ==
          Catch::Approx(0.0).margin(1.0e-8));
  REQUIRE(rad_dep[0] - rad_emit[0] ==
          Catch::Approx(result.matter_delta).epsilon(1.0e-10));
}

TEST_CASE("HOLO LO source solve shifts in consistency-source direction",
          "[radiation][holo]") {
  std::vector<double> E_lo = {10.0};
  std::vector<double> ee = {2.0e4};
  std::vector<double> Te = {2.0};
  std::vector<double> Pe = {0.0};
  std::vector<double> sigma_P = {0.0};
  std::vector<double> sigma_R = {1.0e30};
  std::vector<double> rho = {1.0};
  std::vector<double> mass = {1.0};
  std::vector<double> vol = {1.0};
  std::vector<double> node_r = {0.0, 0.0};
  std::vector<std::uint8_t> core = {1U};
  std::vector<double> consistency_source = {4.0e10};
  std::vector<double> rad_dep = {0.0};
  std::vector<double> rad_emit = {0.0};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.consistency_source = consistency_source.data();
  in.consistency_alpha = 0.25;
  in.rad_dep_lo = rad_dep.data();
  in.rad_emit_lo = rad_emit.data();

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(E_lo[0] == Catch::Approx(11.0).epsilon(1.0e-12));
  REQUIRE(result.rad_delta == Catch::Approx(1.0).epsilon(1.0e-12));
  REQUIRE(result.matter_delta == Catch::Approx(-1.0).epsilon(1.0e-12));
  REQUIRE(rad_dep[0] - rad_emit[0] ==
          Catch::Approx(result.matter_delta).epsilon(1.0e-12));
  REQUIRE(result.conservation_error == Catch::Approx(0.0).margin(1.0e-12));
}

TEST_CASE("HOLO LO source updates radiation globally but material only in coupled cells",
          "[radiation][holo]") {
  std::vector<double> E_lo = {100.0, 100.0};
  std::vector<double> ee = {2.0e4, 2.0e4};
  std::vector<double> Te = {2.0, 2.0};
  std::vector<double> Pe = {0.0, 0.0};
  std::vector<double> sigma_P = {
      0.25 / (tenryu::core::constants::c_light * 1.0e-10),
      0.25 / (tenryu::core::constants::c_light * 1.0e-10)};
  std::vector<double> sigma_R = {1.0e30, 1.0e30};
  std::vector<double> rho = {1.0, 1.0};
  std::vector<double> mass = {1.0, 1.0};
  std::vector<double> vol = {1.0, 1.0};
  std::vector<double> node_r = {0.0, 0.0, 0.0};
  std::vector<std::uint8_t> coupled = {0U, 1U};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, coupled);
  const double ee0_before = ee[0];
  const double ee1_before = ee[1];

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(E_lo[0] > 100.0);
  REQUIRE(E_lo[1] > 100.0);
  REQUIRE(ee[0] == Catch::Approx(ee0_before));
  REQUIRE(ee[1] != Catch::Approx(ee1_before));
  REQUIRE(result.matter_delta == Catch::Approx(ee[1] - ee1_before));
}

TEST_CASE("HOLO LO source records blend deltas without material commit",
          "[radiation][holo]") {
  std::vector<double> E_lo = {100.0, 100.0};
  std::vector<double> ee = {2.0e4, 2.0e4};
  std::vector<double> Te = {2.0, 2.0};
  std::vector<double> Pe = {0.0, 0.0};
  std::vector<double> sigma_P = {
      0.25 / (tenryu::core::constants::c_light * 1.0e-10),
      0.25 / (tenryu::core::constants::c_light * 1.0e-10)};
  std::vector<double> sigma_R = {1.0e30, 1.0e30};
  std::vector<double> rho = {1.0, 1.0};
  std::vector<double> mass = {1.0, 1.0};
  std::vector<double> vol = {1.0, 1.0};
  std::vector<double> node_r = {0.0, 0.0, 0.0};
  std::vector<std::uint8_t> core = {0U, 1U};
  std::vector<std::uint8_t> patch = {1U, 1U};
  std::vector<double> lo_weight = {0.5, 1.0};
  std::vector<double> rad_dep = {0.0, 0.0};
  std::vector<double> rad_emit = {0.0, 0.0};
  std::vector<double> matter_delta_cell = {0.0, 0.0};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.cell_active = patch.data();
  in.lo_weight = lo_weight.data();
  in.rad_dep_lo = rad_dep.data();
  in.rad_emit_lo = rad_emit.data();
  in.matter_delta_lo_cell = matter_delta_cell.data();
  const double ee0_before = ee[0];
  const double ee1_before = ee[1];

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(ee[0] == Catch::Approx(ee0_before));
  REQUIRE(ee[1] != Catch::Approx(ee1_before));
  REQUIRE(matter_delta_cell[0] != Catch::Approx(0.0));
  REQUIRE(matter_delta_cell[1] != Catch::Approx(0.0));
  REQUIRE(rad_dep[0] - rad_emit[0] ==
          Catch::Approx(matter_delta_cell[0]).epsilon(1.0e-10));
  REQUIRE(rad_dep[1] - rad_emit[1] ==
          Catch::Approx(matter_delta_cell[1]).epsilon(1.0e-10));
  REQUIRE(result.matter_delta ==
          Catch::Approx(matter_delta_cell[1]).epsilon(1.0e-10));
}

TEST_CASE("HOLO LO source solve approaches equilibrium in large alpha limit",
          "[radiation][holo]") {
  std::vector<double> E_lo = {100.0};
  std::vector<double> ee = {2.0e4};
  std::vector<double> Te = {2.0};
  std::vector<double> Pe = {0.0};
  std::vector<double> sigma_P = {
      1.0e6 / (tenryu::core::constants::c_light * 1.0e-10)};
  std::vector<double> sigma_R = {1.0e20};
  std::vector<double> rho = {1.0};
  std::vector<double> mass = {1.0};
  std::vector<double> vol = {1.0};
  std::vector<double> node_r = {0.0, 0.0};
  std::vector<std::uint8_t> core = {1U};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  const double T2 = 2.0 * 2.0;
  const double B = tenryu::core::constants::a_eV * T2 * T2;
  REQUIRE(E_lo[0] == Catch::Approx(B).epsilon(1.0e-5));
  REQUIRE(result.matter_delta + result.rad_delta ==
          Catch::Approx(0.0).margin(1.0e-5));
}

TEST_CASE("HOLO LO source net remains conservative at alpha roundoff limit",
          "[radiation][holo]") {
  std::vector<double> E_lo = {0.0};
  std::vector<double> ee = {2.0e4};
  std::vector<double> Te = {2.0};
  std::vector<double> Pe = {0.0};
  std::vector<double> sigma_P = {
      1.0e16 / (tenryu::core::constants::c_light * 1.0e-10)};
  std::vector<double> sigma_R = {1.0e20};
  std::vector<double> rho = {1.0};
  std::vector<double> mass = {1.0};
  std::vector<double> vol = {1.0};
  std::vector<double> node_r = {0.0, 0.0};
  std::vector<std::uint8_t> core = {1U};
  std::vector<double> rad_dep = {0.0};
  std::vector<double> rad_emit = {0.0};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.rad_dep_lo = rad_dep.data();
  in.rad_emit_lo = rad_emit.data();

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  const double T2 = 2.0 * 2.0;
  const double B = tenryu::core::constants::a_eV * T2 * T2;
  REQUIRE(result.failures == 0);
  REQUIRE(E_lo[0] == Catch::Approx(B).epsilon(1.0e-12));
  REQUIRE(result.rad_delta == Catch::Approx(B).epsilon(1.0e-12));
  REQUIRE(result.matter_delta == Catch::Approx(-B).epsilon(1.0e-12));
  REQUIRE(rad_dep[0] - rad_emit[0] ==
          Catch::Approx(result.matter_delta).epsilon(1.0e-12));
  REQUIRE(result.matter_delta + result.rad_delta ==
          Catch::Approx(0.0).margin(1.0e-8));
}

TEST_CASE("HOLO LO source solve closes with table EOS",
          "[radiation][holo]") {
  tenryu::materials::EOSTable eos;
  eos.rho_grid = {1.0, 2.0};
  eos.T_grid_eV = {1.0, 2.0, 4.0};
  for (const double T : eos.T_grid_eV) {
    for (const double rho : eos.rho_grid) {
      eos.P_table.push_back(30.0 * rho * T);
      eos.e_table.push_back(100.0 * T);
    }
  }
  eos.finalize();

  std::vector<double> E_lo = {100.0};
  std::vector<double> ee = {200.0};
  std::vector<double> Te = {2.0};
  std::vector<double> Pe = {0.0};
  std::vector<double> sigma_P = {
      0.1 / (tenryu::core::constants::c_light * 1.0e-10)};
  std::vector<double> sigma_R = {1.0e20};
  std::vector<double> rho = {1.5};
  std::vector<double> mass = {100.0};
  std::vector<double> vol = {1.0};
  std::vector<double> node_r = {0.0, 0.0};
  std::vector<std::uint8_t> core = {1U};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.material_model = tenryu::radiation::HoloLOMaterialModel::TableEOS;
  in.electron_eos = &eos;

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(ee[0] == Catch::Approx(eos.energy(rho[0], Te[0])).epsilon(1.0e-12));
  REQUIRE(Pe[0] == Catch::Approx(eos.pressure(rho[0], Te[0])).epsilon(1.0e-12));
  REQUIRE(result.matter_delta + result.rad_delta ==
          Catch::Approx(0.0).margin(1.0e-6));
}

TEST_CASE("HOLO LO global solve leaks only through physical outer vacuum boundary",
          "[radiation][holo]") {
  std::vector<double> E_lo = {10.0};
  std::vector<double> ee = {2.0e4};
  std::vector<double> Te = {2.0};
  std::vector<double> Pe = {0.0};
  std::vector<double> sigma_P = {0.0};
  std::vector<double> sigma_R = {1.0e30};
  std::vector<double> rho = {1.0};
  std::vector<double> mass = {1.0};
  std::vector<double> vol = {1.0};
  std::vector<double> node_r = {0.0, 1.0};
  std::vector<std::uint8_t> coupled = {0U};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, coupled);
  in.dt = 1.0e-12;

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(result.boundary_E_in == Catch::Approx(0.0));
  REQUIRE(result.boundary_E_out > 0.0);
  REQUIRE(E_lo[0] < 10.0);
  REQUIRE(result.rad_delta + result.boundary_E_out ==
          Catch::Approx(0.0).margin(1.0e-10));
}

TEST_CASE("HOLO LO patch solve uses Dirichlet coupling at internal patch boundary",
          "[radiation][holo]") {
  std::vector<double> E_lo = {10.0, 20.0};
  std::vector<double> ee = {2.0e4, 2.0e4};
  std::vector<double> Te = {2.0, 2.0};
  std::vector<double> Pe = {0.0, 0.0};
  std::vector<double> sigma_P = {0.0, 0.0};
  std::vector<double> sigma_R = {16.0, 16.0};
  std::vector<double> rho = {1.0, 1.0};
  std::vector<double> mass = {1.0, 1.0};
  std::vector<double> vol = {1.0, 1.0};
  std::vector<double> node_r = {0.0, 1.0, 2.0};
  std::vector<std::uint8_t> core = {0U, 0U};
  std::vector<std::uint8_t> patch = {1U, 0U};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.cell_active = patch.data();
  in.has_physical_outer_vacuum = true;

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  const double coupling = 4.0 * 3.141592653589793238462643383279502884 *
                          tenryu::core::constants::c_light / (3.0 * sigma_R[0]);
  const double k = coupling * in.dt;
  const double expected_E0 = (10.0 + k * 20.0) / (1.0 + k);

  REQUIRE(result.failures == 0);
  REQUIRE(result.boundary_E_in == Catch::Approx(k * 20.0).epsilon(1.0e-12));
  REQUIRE(result.boundary_E_out == Catch::Approx(k * expected_E0).epsilon(1.0e-12));
  REQUIRE(E_lo[0] == Catch::Approx(expected_E0).epsilon(1.0e-12));
  REQUIRE(E_lo[1] == Catch::Approx(20.0));
  REQUIRE(result.rad_delta == Catch::Approx(expected_E0 - 10.0).epsilon(1.0e-12));
  REQUIRE(result.conservation_error == Catch::Approx(0.0).margin(1.0e-10));
}

TEST_CASE("HOLO LO patch solve uses both neighboring cells for Dirichlet coupling",
          "[radiation][holo]") {
  std::vector<double> E_lo = {30.0, 10.0, 50.0};
  std::vector<double> ee = {2.0e4, 2.0e4, 2.0e4};
  std::vector<double> Te = {2.0, 2.0, 2.0};
  std::vector<double> Pe = {0.0, 0.0, 0.0};
  std::vector<double> sigma_P = {0.0, 0.0, 0.0};
  std::vector<double> sigma_R = {16.0, 16.0, 16.0};
  std::vector<double> rho = {1.0, 1.0, 1.0};
  std::vector<double> mass = {1.0, 1.0, 1.0};
  std::vector<double> vol = {1.0, 1.0, 1.0};
  std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0};
  std::vector<std::uint8_t> core = {0U, 0U, 0U};
  std::vector<std::uint8_t> patch = {0U, 1U, 0U};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.cell_active = patch.data();
  in.has_physical_outer_vacuum = false;

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  const double pi = 3.141592653589793238462643383279502884;
  const double left_coupling =
      4.0 * pi * tenryu::core::constants::c_light / (3.0 * sigma_R[0]);
  const double right_coupling =
      16.0 * pi * tenryu::core::constants::c_light / (3.0 * sigma_R[0]);
  const double left_k = left_coupling * in.dt;
  const double right_k = right_coupling * in.dt;
  const double expected_E1 =
      (10.0 + left_k * 30.0 + right_k * 50.0) / (1.0 + left_k + right_k);

  REQUIRE(result.failures == 0);
  REQUIRE(E_lo[0] == Catch::Approx(30.0));
  REQUIRE(E_lo[1] == Catch::Approx(expected_E1).epsilon(1.0e-12));
  REQUIRE(E_lo[2] == Catch::Approx(50.0));
  REQUIRE(result.boundary_E_in ==
          Catch::Approx(left_k * 30.0 + right_k * 50.0).epsilon(1.0e-12));
  REQUIRE(result.boundary_E_out ==
          Catch::Approx((left_k + right_k) * expected_E1).epsilon(1.0e-12));
  REQUIRE(result.conservation_error == Catch::Approx(0.0).margin(1.0e-10));
}

TEST_CASE("HOLO LO patch solve accepts signed HO face-current inflow",
          "[radiation][holo]") {
  std::vector<double> E_lo = {5.0, 10.0, 5.0};
  std::vector<double> ee = {2.0e4, 2.0e4, 2.0e4};
  std::vector<double> Te = {2.0, 2.0, 2.0};
  std::vector<double> Pe = {0.0, 0.0, 0.0};
  std::vector<double> sigma_P = {0.0, 0.0, 0.0};
  std::vector<double> sigma_R = {1.0e30, 1.0e30, 1.0e30};
  std::vector<double> rho = {1.0, 1.0, 1.0};
  std::vector<double> mass = {1.0, 1.0, 1.0};
  std::vector<double> vol = {1.0, 1.0, 1.0};
  std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0};
  std::vector<std::uint8_t> core = {0U, 0U, 0U};
  std::vector<std::uint8_t> patch = {0U, 1U, 0U};
  std::vector<double> face_current = {0.0, 2.0, -3.0, 0.0};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.cell_active = patch.data();
  in.face_current = face_current.data();
  in.face_current_dt = in.dt;
  in.has_physical_outer_vacuum = false;

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(E_lo[0] == Catch::Approx(5.0));
  REQUIRE(E_lo[1] == Catch::Approx(15.0));
  REQUIRE(E_lo[2] == Catch::Approx(5.0));
  REQUIRE(result.boundary_E_in == Catch::Approx(5.0));
  REQUIRE(result.boundary_E_out == Catch::Approx(0.0));
  REQUIRE(result.boundary_limited_E == Catch::Approx(0.0));
  REQUIRE(result.rad_delta == Catch::Approx(5.0));
  REQUIRE(result.conservation_error == Catch::Approx(0.0).margin(1.0e-10));
}

TEST_CASE("HOLO LO patch solve caps signed HO face-current outflow",
          "[radiation][holo]") {
  std::vector<double> E_lo = {5.0, 1.0, 5.0};
  std::vector<double> ee = {2.0e4, 2.0e4, 2.0e4};
  std::vector<double> Te = {2.0, 2.0, 2.0};
  std::vector<double> Pe = {0.0, 0.0, 0.0};
  std::vector<double> sigma_P = {0.0, 0.0, 0.0};
  std::vector<double> sigma_R = {1.0e30, 1.0e30, 1.0e30};
  std::vector<double> rho = {1.0, 1.0, 1.0};
  std::vector<double> mass = {1.0, 1.0, 1.0};
  std::vector<double> vol = {1.0, 1.0, 1.0};
  std::vector<double> node_r = {0.0, 1.0, 2.0, 3.0};
  std::vector<std::uint8_t> core = {0U, 0U, 0U};
  std::vector<std::uint8_t> patch = {0U, 1U, 0U};
  std::vector<double> face_current = {0.0, -5.0, 0.0, 0.0};

  auto in = base_inputs(E_lo, ee, Te, Pe, sigma_P, sigma_R, rho, mass, vol, node_r, core);
  in.cell_active = patch.data();
  in.face_current = face_current.data();
  in.face_current_dt = in.dt;
  in.has_physical_outer_vacuum = false;

  const auto result = tenryu::radiation::solve_holo_lo_1d_cpu(in);

  REQUIRE(result.failures == 0);
  REQUIRE(E_lo[0] == Catch::Approx(5.0));
  REQUIRE(E_lo[1] == Catch::Approx(0.0).margin(1.0e-12));
  REQUIRE(E_lo[2] == Catch::Approx(5.0));
  REQUIRE(result.boundary_E_in == Catch::Approx(0.0));
  REQUIRE(result.boundary_E_out == Catch::Approx(1.0));
  REQUIRE(result.boundary_limited_E == Catch::Approx(4.0));
  REQUIRE(result.rad_delta == Catch::Approx(-1.0));
  REQUIRE(result.conservation_error == Catch::Approx(0.0).margin(1.0e-10));
}
