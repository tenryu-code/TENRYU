#include "hydro/carrier_constraint.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "core/error.hpp"

namespace tenryu::hydro {
namespace {

void check_carrier_and_slaves(const mesh::BoundaryCarrier& carrier,
                              const CarrierSlaveView& slaves,
                              const std::size_t node_count) {
  TENRYU_ASSERT(
      carrier.valid,
      "carrier constraint requires a valid single closed loop (single boundary component)");
  TENRYU_ASSERT(carrier.masters.size() >= 3,
                "carrier constraint requires at least three masters");
  TENRYU_ASSERT(slaves.count >= 0, "carrier slave count must be nonnegative");
  if (slaves.count > 0) {
    TENRYU_ASSERT(slaves.node != nullptr && slaves.edge != nullptr &&
                      slaves.lambda != nullptr,
                  "carrier slave arrays must be non-null");
  }

  for (const mesh::CarrierVertex& master : carrier.masters) {
    TENRYU_ASSERT(master.mesh_node >= 0 &&
                      static_cast<std::size_t>(master.mesh_node) < node_count,
                  "carrier master node out of range");
  }

  const int n = static_cast<int>(carrier.masters.size());
  for (int i = 0; i < slaves.count; ++i) {
    TENRYU_ASSERT(slaves.edge[i] >= 0 && slaves.edge[i] < n,
                  "carrier slave edge out of range");
    TENRYU_ASSERT(slaves.lambda[i] > 0.0 && slaves.lambda[i] < 1.0,
                  "carrier slave lambda must be strictly inside its edge");
    TENRYU_ASSERT(slaves.node[i] >= 0 &&
                      static_cast<std::size_t>(slaves.node[i]) < node_count,
                  "carrier slave node out of range");
  }
}

void solve_cyclic_ldlt(const std::span<const double> d,
                       const std::span<const double> e,
                       const std::span<const double> b,
                       const std::span<double> x) {
  const int n = static_cast<int>(d.size());
  const int m = n - 1;
  std::vector<double> l(static_cast<std::size_t>(m - 1), 0.0);
  std::vector<double> p(static_cast<std::size_t>(m), 0.0);
  std::vector<double> s(static_cast<std::size_t>(m), 0.0);
  std::vector<double> u(static_cast<std::size_t>(m), 0.0);
  std::vector<double> y(static_cast<std::size_t>(m), 0.0);

  u[0] = e[n - 1];
  u[m - 1] = e[m - 1];

  p[0] = d[0];
  TENRYU_ASSERT(p[0] > 0.0, "carrier LDLT pivot");
  s[0] = u[0];
  for (int i = 1; i < m; ++i) {
    l[i - 1] = e[i - 1] / p[i - 1];
    p[i] = d[i] - l[i - 1] * e[i - 1];
    TENRYU_ASSERT(p[i] > 0.0, "carrier LDLT pivot");
    s[i] = u[i] - l[i - 1] * s[i - 1];
  }

  double border_sum = 0.0;
  for (int i = 0; i < m; ++i) {
    border_sum += s[i] * s[i] / p[i];
  }
  const double pm = d[m] - border_sum;
  TENRYU_ASSERT(pm > 0.0, "carrier LDLT pivot");

  y[0] = b[0];
  for (int i = 1; i < m; ++i) {
    y[i] = b[i] - l[i - 1] * y[i - 1];
  }
  double rhs_sum = 0.0;
  for (int i = 0; i < m; ++i) {
    rhs_sum += s[i] * y[i] / p[i];
  }
  const double ym = b[m] - rhs_sum;

  x[m] = ym / pm;
  x[m - 1] = y[m - 1] / p[m - 1] -
             (s[m - 1] / p[m - 1]) * x[m];
  for (int i = m - 2; i >= 0; --i) {
    x[i] = y[i] / p[i] - l[i] * x[i + 1] - (s[i] / p[i]) * x[m];
  }
}

}  // namespace

void validate_carrier_slave_view(const mesh::BoundaryCarrier& carrier,
                                 const CarrierSlaveView& slaves,
                                 const std::size_t node_count) {
  check_carrier_and_slaves(carrier, slaves, node_count);

  std::vector<std::uint8_t> mark(node_count, 0);
  for (const mesh::CarrierVertex& master : carrier.masters) {
    TENRYU_ASSERT(mark[master.mesh_node] == 0,
                  "duplicate carrier master node");
    mark[master.mesh_node] = 1;
  }
  for (int i = 0; i < slaves.count; ++i) {
    TENRYU_ASSERT(mark[slaves.node[i]] == 0,
                  "carrier slave node aliases a master or another slave");
    mark[slaves.node[i]] = 2;
  }
}

CondensedBoundarySystem condense_boundary_forces_and_masses(
    const mesh::BoundaryCarrier& carrier, const CarrierSlaveView& slaves,
    const std::span<const double> node_mass,
    const std::span<const double> force_r,
    const std::span<const double> force_z) {
  check_carrier_and_slaves(carrier, slaves, node_mass.size());
  TENRYU_ASSERT(force_r.size() == node_mass.size(),
                "carrier radial force size mismatch");
  TENRYU_ASSERT(force_z.size() == node_mass.size(),
                "carrier axial force size mismatch");

  const int n = static_cast<int>(carrier.masters.size());
  CondensedBoundarySystem system;
  system.n = n;
  system.diag.resize(static_cast<std::size_t>(n));
  system.off.assign(static_cast<std::size_t>(n), 0.0);
  system.f_r.resize(static_cast<std::size_t>(n));
  system.f_z.resize(static_cast<std::size_t>(n));
  system.axis.resize(static_cast<std::size_t>(n));

  for (int k = 0; k < n; ++k) {
    const int node = carrier.masters[k].mesh_node;
    TENRYU_ASSERT(node_mass[node] > 0.0,
                  "carrier master mass must be positive");
    system.diag[k] = node_mass[node];
    system.f_r[k] = force_r[node];
    system.f_z[k] = force_z[node];
    system.axis[k] =
        carrier.masters[k].bc_class == mesh::CarrierBcClass::kAxis ? 1 : 0;
  }

  for (int i = 0; i < slaves.count; ++i) {
    const int e = slaves.edge[i];
    const int A = e;
    const int B = (e + 1) % n;
    const double wA = 1.0 - slaves.lambda[i];
    const double wB = slaves.lambda[i];
    const int node = slaves.node[i];
    TENRYU_ASSERT(node_mass[node] > 0.0,
                  "carrier slave mass must be positive");
    const double ms = node_mass[node];
    const double fr = force_r[node];
    const double fz = force_z[node];
    system.diag[A] += wA * wA * ms;
    system.diag[B] += wB * wB * ms;
    system.off[e] += wA * wB * ms;
    system.f_r[A] += wA * fr;
    system.f_r[B] += wB * fr;
    system.f_z[A] += wA * fz;
    system.f_z[B] += wB * fz;
  }

  return system;
}

void solve_condensed_masters(const CondensedBoundarySystem& system,
                             const std::span<double> accel_r,
                             const std::span<double> accel_z) {
  TENRYU_ASSERT(system.n >= 3,
                "carrier condensed system requires at least three masters");
  const std::size_t n = static_cast<std::size_t>(system.n);
  TENRYU_ASSERT(system.diag.size() == n,
                "carrier condensed diagonal size mismatch");
  TENRYU_ASSERT(system.off.size() == n,
                "carrier condensed off-diagonal size mismatch");
  TENRYU_ASSERT(system.f_r.size() == n,
                "carrier condensed radial force size mismatch");
  TENRYU_ASSERT(system.f_z.size() == n,
                "carrier condensed axial force size mismatch");
  TENRYU_ASSERT(system.axis.size() == n,
                "carrier condensed axis mask size mismatch");
  TENRYU_ASSERT(accel_r.size() == n,
                "carrier radial acceleration size mismatch");
  TENRYU_ASSERT(accel_z.size() == n,
                "carrier axial acceleration size mismatch");
  for (int k = 0; k < system.n; ++k) {
    TENRYU_ASSERT(system.diag[k] > 0.0,
                  "carrier condensed diagonal must be positive");
  }
  for (int k = 0; k < system.n; ++k) {
    TENRYU_ASSERT(system.off[k] >= 0.0,
                  "carrier condensed coupling must be nonnegative");
  }

  solve_cyclic_ldlt(system.diag, system.off, system.f_z, accel_z);

  std::vector<double> d = system.diag;
  std::vector<double> e = system.off;
  std::vector<double> b = system.f_r;
  for (int k = 0; k < system.n; ++k) {
    if (system.axis[k] == 1) {
      d[k] = 1.0;
      b[k] = 0.0;
      e[k] = 0.0;
      e[(k + system.n - 1) % system.n] = 0.0;
    }
  }
  solve_cyclic_ldlt(d, e, b, accel_r);
}

void reconstruct_slaves(const mesh::BoundaryCarrier& carrier,
                        const CarrierSlaveView& slaves,
                        const std::span<double> pos_r,
                        const std::span<double> pos_z,
                        const std::span<double> vel_r,
                        const std::span<double> vel_z) {
  check_carrier_and_slaves(carrier, slaves, pos_r.size());
  TENRYU_ASSERT(pos_z.size() == pos_r.size(),
                "carrier axial position size mismatch");
  TENRYU_ASSERT(vel_r.size() == pos_r.size(),
                "carrier radial velocity size mismatch");
  TENRYU_ASSERT(vel_z.size() == pos_r.size(),
                "carrier axial velocity size mismatch");

  const int n = static_cast<int>(carrier.masters.size());
  for (int i = 0; i < slaves.count; ++i) {
    const int e = slaves.edge[i];
    const mesh::CarrierVertex& A = carrier.masters[e];
    const mesh::CarrierVertex& B = carrier.masters[(e + 1) % n];
    const int a = A.mesh_node;
    const int bnode = B.mesh_node;
    const int sn = slaves.node[i];
    const double w = slaves.lambda[i];
    const double omw = 1.0 - w;
    pos_r[sn] = omw * pos_r[a] + w * pos_r[bnode];
    pos_z[sn] = omw * pos_z[a] + w * pos_z[bnode];
    vel_r[sn] = omw * vel_r[a] + w * vel_r[bnode];
    vel_z[sn] = omw * vel_z[a] + w * vel_z[bnode];
    if (A.bc_class == mesh::CarrierBcClass::kAxis &&
        B.bc_class == mesh::CarrierBcClass::kAxis) {
      TENRYU_ASSERT(pos_r[a] == 0.0 && pos_r[bnode] == 0.0,
                    "axis-edge master off axis");
      pos_r[sn] = +0.0;
    }
  }
}

}  // namespace tenryu::hydro
