#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/config.hpp"
#include "core/state.hpp"
#include "hydro/ale_1d_remap.cuh"
#include "hydro/ale_1d_velocity_project.cuh"
#include "mesh/mesh.hpp"

namespace {

bool has_cuda_device() {
  int device_count = 0;
  const cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
}

tenryu::core::Config make_cfg(const int n) {
  tenryu::core::Config cfg;
  cfg.main.dim = 1;
  cfg.main.dimension = "1D_SPH";
  cfg.mesh.nr = n;
  cfg.mesh.nz = 1;
  cfg.mesh.r_min = 0.0;
  cfg.mesh.r_max = 1.0;
  cfg.mesh.grid_type_r = "uniform";
  cfg.radiation.groups = 1;
  cfg.numerics.ale1d.enabled = true;
  cfg.materials.materials.clear();
  tenryu::core::Config::MaterialsConfig::MatDef mat;
  mat.name = "mat0";
  mat.A = 1.0;
  mat.Z = 1.0;
  cfg.materials.materials.push_back(mat);
  return cfg;
}

template <typename Tag>
std::vector<double> to_host(const tenryu::core::Field1D<Tag>& field) {
  std::vector<double> host(field.size(), 0.0);
  field.copy_to_host(host.data());
  return host;
}

template <typename T>
std::vector<T> to_host(const tenryu::hydro::ale1d::DeviceArray<T>& field) {
  std::vector<T> host;
  field.copy_to_host(host);
  return host;
}

tenryu::hydro::ale1d::NodeConstraintMask default_mask(const int n) {
  tenryu::hydro::ale1d::NodeConstraintMask mask;
  mask.pinned.assign(static_cast<std::size_t>(n + 1), false);
  mask.pinned.front() = true;
  mask.pinned.back() = true;
  mask.n_protected_nodes = 2;
  return mask;
}

tenryu::core::State make_state(const tenryu::core::Config& cfg,
                               const std::vector<double>& rho,
                               const std::vector<double>& v) {
  tenryu::core::State state = tenryu::core::State::allocate(cfg);
  state.mesh = tenryu::mesh::create_mesh(cfg, state);
  state.vol = state.mesh.cell_vol;

  const int n = cfg.mesh.nr;
  const std::vector<double> vol = to_host(state.vol);
  std::vector<double> mass(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    mass[static_cast<std::size_t>(i)] =
        rho[static_cast<std::size_t>(i)] * vol[static_cast<std::size_t>(i)];
  }
  std::vector<double> ee(static_cast<std::size_t>(n), 1.0);
  std::vector<double> ei(static_cast<std::size_t>(n), 1.0);
  std::vector<double> rad(static_cast<std::size_t>(n), 1.0);
  std::vector<double> vf(static_cast<std::size_t>(n), 1.0);

  state.rho.copy_from_host(rho);
  state.mass.copy_from_host(mass);
  state.ee.copy_from_host(ee);
  state.ei.copy_from_host(ei);
  state.rad_E.copy_from_host(rad);
  state.volFrac.copy_from_host(vf);
  state.v_r.copy_from_host(v);
  return state;
}

std::vector<double> shifted_nodes(const std::vector<double>& old_r,
                                  const double fraction) {
  const int n = static_cast<int>(old_r.size()) - 1;
  std::vector<double> candidate = old_r;
  const double dr = old_r[1] - old_r[0];
  for (int j = 1; j < n; ++j) {
    candidate[static_cast<std::size_t>(j)] += fraction * dr;
  }
  return candidate;
}

// Projection inputs on the device: new cell masses, face mass fluxes and the
// face taper phi, and (with the kinetic energy closure) the remapped cell
// kinetic energies and candidate specific energies.
struct RemapInputs {
  tenryu::hydro::ale1d::DeviceArray<double> mass_new;
  tenryu::hydro::ale1d::DeviceArray<double> mass_flux;
  tenryu::hydro::ale1d::DeviceArray<double> phi;
  tenryu::hydro::ale1d::DeviceArray<double> ke_remap;
  tenryu::hydro::ale1d::DeviceArray<double> ee_new;
  tenryu::hydro::ale1d::DeviceArray<double> ei_new;
};

RemapInputs explicit_inputs(const std::vector<double>& mass_new,
                            const std::vector<double>& mass_flux,
                            const std::vector<double>& phi) {
  RemapInputs inputs;
  inputs.mass_new.resize(mass_new.size());
  inputs.mass_new.copy_from_host(mass_new);
  inputs.mass_flux.resize(mass_flux.size());
  inputs.mass_flux.copy_from_host(mass_flux);
  inputs.phi.resize(phi.size());
  inputs.phi.copy_from_host(phi);
  return inputs;
}

// No transport: the new masses are the given ones and every face flux is 0.
RemapInputs no_transport_inputs(const std::vector<double>& mass_new) {
  const std::size_t faces = mass_new.size() + 1U;
  return explicit_inputs(mass_new, std::vector<double>(faces, 0.0),
                         std::vector<double>(faces, 0.0));
}

RemapInputs build_remap_inputs(const tenryu::core::State& state,
                               const tenryu::core::Config& cfg,
                               const std::vector<double>& candidate) {
  const int n = cfg.mesh.nr;
  const bool closure = cfg.numerics.ale1d.ke_conservation_closure;
  tenryu::hydro::ale1d::Ale1dRemapScratch remap;
  remap.resize(n, 1, 1, closure);
  const auto result = tenryu::hydro::ale1d::remap_v3(
      state, cfg, candidate, default_mask(n), {}, remap);
  REQUIRE(result.success);
  RemapInputs inputs = explicit_inputs(to_host(remap.mass_new), to_host(remap.mass_flux),
                                       to_host(remap.phi_face));
  if (closure) {
    inputs.ke_remap.resize(static_cast<std::size_t>(n));
    inputs.ke_remap.copy_from_host(to_host(remap.ke_remap));
    inputs.ee_new.resize(static_cast<std::size_t>(n));
    inputs.ee_new.copy_from_host(to_host(remap.ee_new));
    inputs.ei_new.resize(static_cast<std::size_t>(n));
    inputs.ei_new.copy_from_host(to_host(remap.ei_new));
  }
  return inputs;
}

tenryu::hydro::ale1d::Ale1dVelocityProjectResult project(
    const tenryu::core::State& state,
    RemapInputs& inputs,
    const bool ke_conservation_closure,
    const bool two_temperature,
    const double* ke_remap,
    double* ee_new,
    double* ei_new,
    tenryu::hydro::ale1d::Ale1dVelocityProjectScratch& scratch) {
  constexpr double kLimiterTheta = 1.5;  // Numerics.ale1d.remap.limiter_theta default
  return tenryu::hydro::ale1d::project_velocity(
      state, inputs.mass_new.data(), inputs.mass_flux.data(), inputs.phi.data(),
      kLimiterTheta, ke_conservation_closure, two_temperature, ke_remap, ee_new, ei_new,
      scratch);
}

long double sum_vector(const std::vector<double>& values) {
  long double sum = 0.0L;
  for (const double value : values) {
    sum += static_cast<long double>(value);
  }
  return sum;
}

long double total_internal_energy(const std::vector<double>& mass,
                                  const std::vector<double>& ee,
                                  const std::vector<double>& ei) {
  long double total = 0.0L;
  for (std::size_t i = 0; i < mass.size(); ++i) {
    total += static_cast<long double>(mass[i]) *
             static_cast<long double>(ee[i] + ei[i]);
  }
  return total;
}

std::vector<double> cell_kinetic_energy(const std::vector<double>& mass,
                                        const std::vector<double>& velocity) {
  std::vector<double> kinetic(mass.size(), 0.0);
  for (std::size_t i = 0; i < mass.size(); ++i) {
    kinetic[i] = 0.25 * mass[i] *
                 (velocity[i] * velocity[i] +
                  velocity[i + 1] * velocity[i + 1]);
  }
  return kinetic;
}

template <typename T>
bool bitwise_equal(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.empty() ||
          std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(T)) == 0);
}

}  // namespace

TEST_CASE("ALE1D velocity projection preserves constant velocity away from center",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 128;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> v(static_cast<std::size_t>(n + 1), 100.0);
  auto state = make_state(cfg, rho, v);
  const std::vector<double> candidate = shifted_nodes(to_host(state.x_r), 0.15);
  RemapInputs inputs = build_remap_inputs(state, cfg, candidate);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  const auto v_new = to_host(scratch.v_new_node);
  CHECK(v_new.front() == Catch::Approx(0.0));
  for (int j = 1; j <= n; ++j) {
    CHECK(v_new[static_cast<std::size_t>(j)] ==
          Catch::Approx(100.0).epsilon(1.0e-12));
  }
}

// Without transport the projection returns the old nodal velocities: the
// cell-momentum projection it replaced averaged node -> cell -> node and
// smoothed any non-uniform velocity even when no face moved (Benson 1992
// §3.5.3, the inversion error).
TEST_CASE("ALE1D velocity projection is the identity when no face moves",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 96;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  std::vector<double> v(static_cast<std::size_t>(n + 1), 0.0);
  std::mt19937 rng(20260923);
  std::uniform_real_distribution<double> v_dist(-1.0e5, 1.0e5);
  std::uniform_real_distribution<double> rho_dist(0.5, 5.0);
  for (int i = 0; i < n; ++i) {
    rho[static_cast<std::size_t>(i)] = rho_dist(rng);
  }
  for (int j = 1; j <= n; ++j) {
    v[static_cast<std::size_t>(j)] = v_dist(rng);
  }
  auto state = make_state(cfg, rho, v);
  RemapInputs inputs = build_remap_inputs(state, cfg, to_host(state.x_r));
  for (const double flux : to_host(inputs.mass_flux)) {
    REQUIRE(flux == 0.0);
  }

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  CHECK(bitwise_equal(to_host(scratch.v_new_node), v));
  // The remap rebuilds each mass as density times volume (last-bit changes).
  CHECK(result.kinetic_energy_new ==
        Catch::Approx(result.kinetic_energy_old).epsilon(1.0e-14));
}

// DeBar's consistency condition: a uniform velocity over a non-uniform
// density stays uniform. The node-velocity pair is advected as a specific
// quantity with the accepted mass fluxes, so it holds to roundoff; the
// cell-momentum projection it replaced reconstructed the momentum density
// with its own limiter and did not.
TEST_CASE("ALE1D velocity projection keeps a uniform velocity over a varying density",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 160;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  for (int i = 0; i < n; ++i) {
    rho[static_cast<std::size_t>(i)] =
        1.0 + 0.8 * std::sin(0.23 * static_cast<double>(i)) +
        ((i % 7 == 0) ? 2.0 : 0.0);
  }
  std::vector<double> v(static_cast<std::size_t>(n + 1), 3.0e6);
  auto state = make_state(cfg, rho, v);
  RemapInputs inputs =
      build_remap_inputs(state, cfg, shifted_nodes(to_host(state.x_r), 0.3));

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  const auto v_new = to_host(scratch.v_new_node);
  double worst = 0.0;
  for (int j = 1; j <= n; ++j) {
    worst = std::max(worst, std::abs(v_new[static_cast<std::size_t>(j)] - 3.0e6) / 3.0e6);
  }
  INFO("largest relative departure from the uniform velocity " << worst);
  CHECK(worst <= 1.0e-13);
}

TEST_CASE("ALE1D velocity projection keeps center velocity zero",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 96;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> v(static_cast<std::size_t>(n + 1), 0.0);
  std::mt19937 rng(12345);
  std::uniform_real_distribution<double> dist(-1.0e5, 1.0e5);
  for (int j = 1; j <= n; ++j) {
    v[static_cast<std::size_t>(j)] = dist(rng);
  }
  auto state = make_state(cfg, rho, v);
  const std::vector<double> candidate = shifted_nodes(to_host(state.x_r), 0.1);
  RemapInputs inputs = build_remap_inputs(state, cfg, candidate);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  const auto v_new = to_host(scratch.v_new_node);
  CHECK(v_new.front() == Catch::Approx(0.0));
}

// The cell momenta m_i (v_i + v_{i+1}) / 2 sum to the nodal momentum
// sum_j M_j v_j; the remapped pair keeps that sum (face fluxes telescope).
TEST_CASE("ALE1D velocity projection conserves cell momentum",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 128;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 0.0);
  std::vector<double> v(static_cast<std::size_t>(n + 1), 0.0);
  for (int i = 0; i < n; ++i) {
    rho[static_cast<std::size_t>(i)] = 1.0 + 0.1 * std::sin(0.17 * i);
  }
  for (int j = 0; j <= n; ++j) {
    v[static_cast<std::size_t>(j)] = 1.0e5 * std::sin(0.11 * j);
  }
  v.front() = 0.0;
  auto state = make_state(cfg, rho, v);
  const std::vector<double> candidate = shifted_nodes(to_host(state.x_r), 0.2);
  RemapInputs inputs = build_remap_inputs(state, cfg, candidate);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  const long double p_old = sum_vector(to_host(scratch.p_old_cell));
  const long double p_new = sum_vector(to_host(scratch.p_new_cell));
  CHECK(static_cast<double>(std::abs(p_new - p_old) /
                            std::max(1.0L, std::abs(p_old))) <= 1.0e-12);
  // The new nodal velocities carry that momentum: sum_j M_j v_j with
  // M_j = (m_{j-1} + m_j) / 2, except the centre node, which is set to rest
  // (its remapped value m_0 psi^L_0 / 2 is dropped; mass fraction ~5e-7).
  const auto mass_new = to_host(inputs.mass_new);
  const auto v_new = to_host(scratch.v_new_node);
  const auto psi_left = to_host(scratch.psi_left);
  long double nodal = 0.0L;
  for (int j = 0; j <= n; ++j) {
    const double left = (j > 0) ? mass_new[static_cast<std::size_t>(j - 1)] : 0.0;
    const double right = (j < n) ? mass_new[static_cast<std::size_t>(j)] : 0.0;
    nodal += 0.5L * (left + right) * v_new[static_cast<std::size_t>(j)];
  }
  const long double centre_dropped = 0.5L * mass_new.front() * psi_left.front();
  CHECK(v_new.front() == 0.0);
  CHECK(static_cast<double>(std::abs(nodal + centre_dropped - p_old) /
                            std::max(1.0L, std::abs(p_old))) <= 1.0e-12);
}

TEST_CASE("ALE1D velocity projection KE drift is small for small displacement",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 4096;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> v(static_cast<std::size_t>(n + 1), 1.0e6);
  auto state = make_state(cfg, rho, v);
  const std::vector<double> candidate = shifted_nodes(to_host(state.x_r), 0.05);
  RemapInputs inputs = build_remap_inputs(state, cfg, candidate);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  CHECK(result.kinetic_energy_drift_rel < 1.0e-10);
}

TEST_CASE("ALE1D velocity projection reports KE drift for large displacement",
          "[hydro][ale1d][velocity]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 160;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> v(static_cast<std::size_t>(n + 1), 0.0);
  for (int j = 0; j <= n; ++j) {
    v[static_cast<std::size_t>(j)] = 2.0e5 * std::sin(2.0 * 3.141592653589793 *
                                                     j / static_cast<double>(n));
  }
  v.front() = 0.0;
  auto state = make_state(cfg, rho, v);
  const std::vector<double> candidate = shifted_nodes(to_host(state.x_r), 0.3);
  RemapInputs inputs = build_remap_inputs(state, cfg, candidate);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  CHECK(std::isfinite(result.kinetic_energy_drift_rel));
  CHECK(result.kinetic_energy_old > 0.0);
  CHECK(result.kinetic_energy_new >= 0.0);
  // The remap of a non-uniform velocity loses kinetic energy (it does not
  // create it).
  CHECK(result.kinetic_energy_new <= result.kinetic_energy_old);
}

// With transport the projected nodal kinetic energy falls short of the
// remapped kinetic energy; the closure puts the difference into the
// internal energy, so K_new + deposited = K_old.
TEST_CASE("ALE1D KE closure conserves projected kinetic plus deposited energy",
          "[hydro][ale1d][velocity][ke-closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 64;
  auto cfg = make_cfg(n);
  cfg.main.two_temperature = true;
  cfg.numerics.ale1d.ke_conservation_closure = true;
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> velocity(static_cast<std::size_t>(n + 1), 0.0);
  for (int j = 1; j <= n; ++j) {
    // Kinetic energy ~0.5 per unit mass against an internal energy of 2
    // (ee = ei = 1): the local closure deposits stay far from the floors.
    velocity[static_cast<std::size_t>(j)] =
        1.0 * std::sin(0.3 * static_cast<double>(j));
  }
  auto state = make_state(cfg, rho, velocity);
  RemapInputs inputs =
      build_remap_inputs(state, cfg, shifted_nodes(to_host(state.x_r), 0.25));
  const std::vector<double> ee_before = to_host(inputs.ee_new);
  const std::vector<double> ei_before = to_host(inputs.ei_new);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, true, cfg.main.two_temperature, inputs.ke_remap.data(),
              inputs.ee_new.data(), inputs.ei_new.data(), scratch);
  REQUIRE(result.success);
  INFO("K_old " << result.kinetic_energy_old << " K_new " << result.kinetic_energy_new
                << " deposited " << result.ke_closure_deposited);
  CHECK(result.ke_closure_deposited > 0.0);
  const double closure_residual =
      std::abs(result.kinetic_energy_new + result.ke_closure_deposited -
               result.kinetic_energy_old) /
      std::max(result.kinetic_energy_old, 1.0e-300);
  CHECK(closure_residual <= 1.0e-12);
  CHECK(result.kinetic_energy_drift_rel <= 1.0e-12);
  const std::vector<double> mass_new = to_host(inputs.mass_new);
  const long double deposited_bookkeeping =
      total_internal_energy(mass_new, to_host(inputs.ee_new), to_host(inputs.ei_new)) -
      total_internal_energy(mass_new, ee_before, ei_before);
  CHECK(result.ke_closure_deposited ==
        Catch::Approx(static_cast<double>(deposited_bookkeeping)).epsilon(1.0e-10));
  for (const double value : to_host(inputs.ee_new)) {
    CHECK(value >= 0.0);
  }
  for (const double value : to_host(inputs.ei_new)) {
    CHECK(value >= 0.0);
  }
  for (const double value : to_host(state.ee)) {
    CHECK(value == 1.0);
  }
  for (const double value : to_host(state.ei)) {
    CHECK(value == 1.0);
  }
}

TEST_CASE("ALE1D KE closure redistributes negative-clamp deficit",
          "[hydro][ale1d][velocity][ke-closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 2;
  auto cfg = make_cfg(n);
  cfg.main.two_temperature = true;
  const std::vector<double> velocity{0.0, 1.0, 0.0};
  auto state = make_state(cfg, {1.0, 1.0}, velocity);
  state.mass.copy_from_host(std::vector<double>{1.0, 1.0});
  const std::vector<double> mass_new = to_host(state.mass);
  RemapInputs inputs = no_transport_inputs(mass_new);
  std::vector<double> ke_initial =
      cell_kinetic_energy(to_host(state.mass), velocity);
  ke_initial[1] += ke_initial[0];
  ke_initial[0] = 0.0;
  const std::vector<double> ee_initial{0.005, 1.0};
  const std::vector<double> ei_initial{0.005, 1.0};
  const long double internal_old =
      total_internal_energy(mass_new, ee_initial, ei_initial);

  tenryu::hydro::ale1d::DeviceArray<double> ke_remap(n);
  tenryu::hydro::ale1d::DeviceArray<double> ee_new(n);
  tenryu::hydro::ale1d::DeviceArray<double> ei_new(n);
  ke_remap.copy_from_host(ke_initial);
  ee_new.copy_from_host(ee_initial);
  ei_new.copy_from_host(ei_initial);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result = project(state, inputs, true, cfg.main.two_temperature,
                              ke_remap.data(), ee_new.data(), ei_new.data(), scratch);
  REQUIRE(result.success);
  const std::vector<double> ee_after = to_host(ee_new);
  const std::vector<double> ei_after = to_host(ei_new);
  const long double internal_new =
      total_internal_energy(mass_new, ee_after, ei_after);
  const long double deposited_bookkeeping = internal_new - internal_old;
  CHECK(result.ke_closure_deposited ==
        Catch::Approx(static_cast<double>(deposited_bookkeeping))
            .margin(1.0e-14));
  const long double total_old =
      static_cast<long double>(result.kinetic_energy_old) + internal_old;
  const long double total_new =
      static_cast<long double>(result.kinetic_energy_new) + internal_new;
  CHECK(static_cast<double>(std::abs(total_new - total_old) /
                            std::max(std::abs(total_old), 1.0e-300L)) <=
        1.0e-12);
  CHECK(result.kinetic_energy_drift_rel <= 1.0e-12);
  CHECK(ee_after[0] == Catch::Approx(0.0).margin(1.0e-15));
  CHECK(ei_after[0] == Catch::Approx(0.0).margin(1.0e-15));
  for (const double value : ee_after) {
    CHECK(value >= 0.0);
  }
  for (const double value : ei_after) {
    CHECK(value >= 0.0);
  }
}

// The remapped kinetic energy is below the projected nodal kinetic energy by
// more than all the internal energy: the closure drains the internal energy
// to zero and the remainder stays unresolved (reported as the energy error
// against sum ke_remap + I_old, the energy the closure conserves).
TEST_CASE("ALE1D KE closure reports capacity-starved remainder",
          "[hydro][ale1d][velocity][ke-closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 2;
  auto cfg = make_cfg(n);
  cfg.main.two_temperature = true;
  const std::vector<double> velocity{0.0, 1.0, 0.0};
  auto state = make_state(cfg, {1.0, 1.0}, velocity);
  state.mass.copy_from_host(std::vector<double>{1.0, 1.0});
  const std::vector<double> mass_new{1.0, 1.0};
  RemapInputs inputs = no_transport_inputs(mass_new);
  const std::vector<double> ke_initial(n, 0.0);
  const std::vector<double> ee_initial(n, 0.005);
  const std::vector<double> ei_initial(n, 0.005);
  const long double internal_old =
      total_internal_energy(mass_new, ee_initial, ei_initial);

  tenryu::hydro::ale1d::DeviceArray<double> ke_remap(n);
  tenryu::hydro::ale1d::DeviceArray<double> ee_new(n);
  tenryu::hydro::ale1d::DeviceArray<double> ei_new(n);
  ke_remap.copy_from_host(ke_initial);
  ee_new.copy_from_host(ee_initial);
  ei_new.copy_from_host(ei_initial);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result = project(state, inputs, true, cfg.main.two_temperature,
                              ke_remap.data(), ee_new.data(), ei_new.data(), scratch);
  REQUIRE(result.success);
  const std::vector<double> ee_after = to_host(ee_new);
  const std::vector<double> ei_after = to_host(ei_new);
  const long double internal_new =
      total_internal_energy(mass_new, ee_after, ei_after);
  CHECK(result.ke_closure_deposited ==
        Catch::Approx(-static_cast<double>(internal_old)).margin(1.0e-15));

  const long double ke_remap_total = sum_vector(ke_initial);
  const long double unabsorbable_remainder =
      static_cast<long double>(result.kinetic_energy_new) - ke_remap_total - internal_old;
  const long double residual_energy_error =
      (static_cast<long double>(result.kinetic_energy_new) + internal_new) -
      (ke_remap_total + internal_old);
  CHECK(static_cast<double>(residual_energy_error) ==
        Catch::Approx(static_cast<double>(unabsorbable_remainder))
            .epsilon(1.0e-14));
  CHECK(unabsorbable_remainder > 0.0L);
  for (const double value : ee_after) {
    CHECK(value >= 0.0);
  }
  for (const double value : ei_after) {
    CHECK(value >= 0.0);
  }
}

TEST_CASE("ALE1D KE closure disabled preserves legacy drift formula",
          "[hydro][ale1d][velocity][ke-closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 32;
  auto cfg = make_cfg(n);
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> velocity(static_cast<std::size_t>(n + 1), 0.0);
  for (int j = 1; j <= n; ++j) {
    velocity[static_cast<std::size_t>(j)] =
        0.2 * std::sin(0.4 * static_cast<double>(j));
  }
  auto state = make_state(cfg, rho, velocity);
  RemapInputs inputs = build_remap_inputs(
      state, cfg, shifted_nodes(to_host(state.x_r), 0.25));

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result =
      project(state, inputs, false, false, nullptr, nullptr, nullptr, scratch);
  REQUIRE(result.success);
  const double diff =
      std::abs(result.kinetic_energy_new - result.kinetic_energy_old);
  const double expected = result.kinetic_energy_old > 1.0e-300
                              ? diff / result.kinetic_energy_old
                              : diff;
  CHECK(result.ke_closure_deposited == 0.0);
  CHECK(result.kinetic_energy_drift_rel == expected);
}

TEST_CASE("ALE1D KE closure is bitwise deterministic",
          "[hydro][ale1d][velocity][ke-closure]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 64;
  auto cfg = make_cfg(n);
  cfg.main.two_temperature = true;
  cfg.numerics.ale1d.ke_conservation_closure = true;
  std::vector<double> rho(static_cast<std::size_t>(n), 1.0);
  std::vector<double> velocity(static_cast<std::size_t>(n + 1), 0.0);
  for (int j = 1; j <= n; ++j) {
    velocity[static_cast<std::size_t>(j)] =
        0.3 * std::sin(0.17 * static_cast<double>(j));
  }
  auto state = make_state(cfg, rho, velocity);
  const std::vector<double> candidate = shifted_nodes(to_host(state.x_r), 0.2);
  RemapInputs first_inputs = build_remap_inputs(state, cfg, candidate);
  RemapInputs second_inputs = build_remap_inputs(state, cfg, candidate);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch_first;
  scratch_first.resize(n);
  const auto first = project(state, first_inputs, true, cfg.main.two_temperature,
                             first_inputs.ke_remap.data(), first_inputs.ee_new.data(),
                             first_inputs.ei_new.data(), scratch_first);
  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch_second;
  scratch_second.resize(n);
  const auto second = project(state, second_inputs, true, cfg.main.two_temperature,
                              second_inputs.ke_remap.data(), second_inputs.ee_new.data(),
                              second_inputs.ei_new.data(), scratch_second);

  REQUIRE(first.success);
  REQUIRE(second.success);
  CHECK(std::memcmp(&first.kinetic_energy_old,
                    &second.kinetic_energy_old,
                    sizeof(double)) == 0);
  CHECK(std::memcmp(&first.kinetic_energy_new,
                    &second.kinetic_energy_new,
                    sizeof(double)) == 0);
  CHECK(std::memcmp(&first.ke_closure_deposited,
                    &second.ke_closure_deposited,
                    sizeof(double)) == 0);
  CHECK(std::memcmp(&first.kinetic_energy_drift_rel,
                    &second.kinetic_energy_drift_rel,
                    sizeof(double)) == 0);
  CHECK(bitwise_equal(to_host(scratch_first.v_new_node),
                      to_host(scratch_second.v_new_node)));
  CHECK(bitwise_equal(to_host(first_inputs.ee_new), to_host(second_inputs.ee_new)));
  CHECK(bitwise_equal(to_host(first_inputs.ei_new), to_host(second_inputs.ei_new)));
}

TEST_CASE("ale1d ke closure preserves negative TMAT baselines",
          "[ale1d]") {
  if (!has_cuda_device()) {
    SKIP("CUDA device not available");
  }

  constexpr int n = 4;
  constexpr double dE = 1.0e-6;
  auto cfg = make_cfg(n);
  cfg.main.two_temperature = true;
  const std::vector<double> velocity(n + 1, 0.0);
  auto state = make_state(cfg, std::vector<double>(n, 1.0), velocity);
  const std::vector<double> mass(n, 1.0e-8);
  state.mass.copy_from_host(mass);
  RemapInputs inputs = no_transport_inputs(mass);
  const std::vector<double> ee_initial{
      -1.0e10, -2.0e10, 2.0e10, 3.0e10};
  const std::vector<double> ei_initial(n, 5.0e10);
  const std::vector<double> ke_initial(n, dE);

  tenryu::hydro::ale1d::DeviceArray<double> ke_remap(n);
  tenryu::hydro::ale1d::DeviceArray<double> ee_new(n);
  tenryu::hydro::ale1d::DeviceArray<double> ei_new(n);
  ke_remap.copy_from_host(ke_initial);
  ee_new.copy_from_host(ee_initial);
  ei_new.copy_from_host(ei_initial);

  tenryu::hydro::ale1d::Ale1dVelocityProjectScratch scratch;
  scratch.resize(n);
  const auto result = project(state, inputs, true, cfg.main.two_temperature,
                              ke_remap.data(), ee_new.data(), ei_new.data(), scratch);
  REQUIRE(result.success);

  const std::vector<double> ee_after = to_host(ee_new);
  const std::vector<double> ei_after = to_host(ei_new);
  for (int i = 0; i < 2; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    CHECK(ee_after[idx] == ee_initial[idx]);
    CHECK(ei_after[idx] - ei_initial[idx] ==
          Catch::Approx(dE / mass[idx]).epsilon(1.0e-6));
  }
  for (int i = 2; i < n; ++i) {
    const auto idx = static_cast<std::size_t>(i);
    const double energy_gain =
        mass[idx] * ((ee_after[idx] - ee_initial[idx]) +
                     (ei_after[idx] - ei_initial[idx]));
    CHECK(energy_gain == Catch::Approx(dE).epsilon(1.0e-6));
  }
  CHECK(result.ke_closure_deposited ==
        Catch::Approx(static_cast<double>(n) * dE).epsilon(1.0e-6));
  CHECK(ee_after[0] < -0.9e10);
}
