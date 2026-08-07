#pragma once

#include <cstdint>
#include <vector>

#include "core/field.hpp"

namespace tenryu::core {
struct Config;
struct State;
}  // namespace tenryu::core

namespace tenryu::hydro {

struct HydroEOSContext;

enum class ArtificialViscosityType {
  Vnr,
  Riemann,
  Csw,
  RiemannCompatible,
};

enum class CswLimiter {
  VanLeer,
  BarthJespersen,
};

class ArtificialViscosity {
 public:
  ArtificialViscosity(double c1 = 0.1,
                      double c2 = 1.5,
                      double J = 1.0,
                      double heat_C = 1.0,
                      bool eos_aware = false,
                      double eos_gamma1_ref = 5.0 / 3.0,
                      double eos_boost_max = 3.0,
                      ArtificialViscosityType type = ArtificialViscosityType::Vnr,
                      double bulk_viscosity_C = 0.0,
                      CswLimiter csw_limiter = CswLimiter::VanLeer,
                      double csw_shock_limiter_floor = 0.65,
                      bool csw_zero_uniform_compression = true,
                      bool csw_diagnostics = false);

  [[nodiscard]] double C1() const noexcept {
    return c1_;
  }

  [[nodiscard]] double C2() const noexcept {
    return c2_;
  }

  [[nodiscard]] double J() const noexcept {
    return j_;
  }

  [[nodiscard]] ArtificialViscosityType type() const noexcept {
    return type_;
  }

  [[nodiscard]] double compute_scalar(double rho,
                                      double dl,
                                      double div_u,
                                      double cs) const;

  void compute_Q_1d(core::CellField1DView Q,
                    core::ConstCellField1DView rho,
                    core::ConstCellField1DView vol,
                    core::ConstNodeField1DView node_r,
                    core::ConstNodeField1DView node_u,
                    core::ConstCellField1DView Pe,
                    core::ConstCellField1DView Pi,
                    core::ConstCellField1DView cs,
                    const std::int8_t* d_hydro_active,
                    const int geom_code,
                    core::CellField1DView chi_out = {},
                    core::CellField1DView q2_out = {},
                    core::CellField1DView div_u_out = {},
                    core::ConstCellField1DView c1_eff = {},
                    core::ConstCellField1DView c2_eff = {}) const;

  void compute_H_1d(core::CellField1DView heat_rate,
                    core::ConstCellField1DView rho,
                    core::ConstCellField1DView e,
                    core::ConstCellField1DView chi,
                    core::ConstNodeField1DView node_r,
                    const std::int8_t* d_hydro_active,
                    const int geom_code,
                    core::ConstCellField1DView heat_C_eff = {}) const;

  // Bulk viscosity is a compression-only scalar pressure: in expansion a
  // positive Q would do negative viscous work (-Q dV/dt < 0), pumping
  // internal energy back into kinetic energy — anti-dissipative and
  // entropy-violating (Caramana–Shashkov–Whalen 1998 requirement).
  void add_bulk_viscosity_1d(core::CellField1DView Q,
                             core::ConstCellField1DView rho,
                             core::ConstCellField1DView vol,
                             core::ConstNodeField1DView node_r,
                             core::ConstNodeField1DView node_u,
                             core::ConstCellField1DView cs,
                             const std::int8_t* d_hydro_active,
                             core::ConstCellField1DView cbulk_eff = {}) const;

  void compute_Q_2d(core::CellField1D& Q,
                    const core::CellField1D& rho,
                    const core::CellField1D& dl,
                    const core::CellField1D& div_u,
                    const core::CellField1D& Pe,
                    const core::CellField1D& Pi,
                    const core::CellField1D& cs,
                    const std::vector<std::int8_t>& hydro_active) const;

  void compute_q_per_material_2d(core::CellField1D& Q,
                                 core::CellField1D& Q_per_material,
                                 core::State& state,
                                 const core::CellField1D& dl,
                                 const core::CellField1D& div_u,
                                 const core::Config& cfg,
                                 const HydroEOSContext* eos_ctx,
                                 const std::vector<std::int8_t>& hydro_active) const;

 private:
  double c1_ = 0.1;
  double c2_ = 1.5;
  double j_ = 1.0;
  double heat_c_ = 1.0;
  bool eos_aware_ = false;
  double eos_gamma1_ref_ = 5.0 / 3.0;
  double eos_boost_max_ = 3.0;
  ArtificialViscosityType type_ = ArtificialViscosityType::Vnr;
  double bulk_viscosity_c_ = 0.0;
  CswLimiter csw_limiter_ = CswLimiter::VanLeer;
  double csw_shock_limiter_floor_ = 0.65;
  bool csw_zero_uniform_compression_ = true;
  bool csw_diagnostics_ = false;
};

}  // namespace tenryu::hydro
