#include "core/namelist/geometry_eval_volume_cut.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

#include "core/namelist/geometry_eval.hpp"
#include "hydro/ale_remap.cuh"

namespace tenryu::core::namelist {
namespace {

constexpr double kPi = tenryu::hydro::ale::detail::kPi;

struct Quad {
  double r[4];
  double z[4];
};

struct LeafEstimate {
  double volume = 0.0;
  double rho_int = 0.0;
  double Te_int = 0.0;
  double Ti_int = 0.0;
  std::vector<double> mat_volume;
};

double quad_volume(const Quad& q) {
  return std::abs(tenryu::hydro::ale::detail::rz_signed_quad_volume(
      q.r[0], q.z[0], q.r[1], q.z[1], q.r[2], q.z[2], q.r[3], q.z[3]));
}

double clamp01(const double x) {
  return std::clamp(x, 0.0, 1.0);
}

void bilinear_point(const Quad& q,
                    const double xi,
                    const double eta,
                    double& r,
                    double& z,
                    double& jac) {
  const double omx = 1.0 - xi;
  const double ome = 1.0 - eta;
  r = omx * ome * q.r[0] + xi * ome * q.r[1] + xi * eta * q.r[2] +
      omx * eta * q.r[3];
  z = omx * ome * q.z[0] + xi * ome * q.z[1] + xi * eta * q.z[2] +
      omx * eta * q.z[3];

  const double dr_dxi =
      -ome * q.r[0] + ome * q.r[1] + eta * q.r[2] - eta * q.r[3];
  const double dz_dxi =
      -ome * q.z[0] + ome * q.z[1] + eta * q.z[2] - eta * q.z[3];
  const double dr_deta =
      -omx * q.r[0] - xi * q.r[1] + xi * q.r[2] + omx * q.r[3];
  const double dz_deta =
      -omx * q.z[0] - xi * q.z[1] + xi * q.z[2] + omx * q.z[3];
  jac = std::abs(dr_dxi * dz_deta - dr_deta * dz_dxi);
}

LeafEstimate estimate_leaf(const Quad& q,
                           const std::size_t n_mat,
                           const GeometryCallables& callables,
                           const bool use_3x3_quadrature) {
  LeafEstimate e;
  e.mat_volume.assign(n_mat, 0.0);
  e.volume = quad_volume(q);

  const std::array<double, 2> x2 = {
      0.5 - 0.5 / std::sqrt(3.0),
      0.5 + 0.5 / std::sqrt(3.0),
  };
  const std::array<double, 2> w2 = {0.5, 0.5};
  const std::array<double, 3> x3 = {
      0.5 - 0.5 * std::sqrt(3.0 / 5.0),
      0.5,
      0.5 + 0.5 * std::sqrt(3.0 / 5.0),
  };
  const std::array<double, 3> w3 = {5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0};

  const int nq = use_3x3_quadrature ? 3 : 2;
  for (int a = 0; a < nq; ++a) {
    const double xi = use_3x3_quadrature ? x3[a] : x2[a];
    const double wx = use_3x3_quadrature ? w3[a] : w2[a];
    for (int b = 0; b < nq; ++b) {
      const double eta = use_3x3_quadrature ? x3[b] : x2[b];
      const double wy = use_3x3_quadrature ? w3[b] : w2[b];
      double r = 0.0;
      double z = 0.0;
      double jac = 0.0;
      bilinear_point(q, xi, eta, r, z, jac);
      const double dV = 2.0 * kPi * r * jac * wx * wy;

      e.rho_int += callables.rho(r, z) * dV;
      e.Te_int += callables.Te(r, z) * dV;
      e.Ti_int += callables.Ti(r, z) * dV;
      for (std::size_t m = 0; m < n_mat; ++m) {
        e.mat_volume[m] += clamp01(callables.volfrac[m](r, z)) * dV;
      }
    }
  }
  return e;
}

std::array<Quad, 4> subdivide(const Quad& q) {
  Quad out0;
  Quad out1;
  Quad out2;
  Quad out3;

  const double r01 = 0.5 * (q.r[0] + q.r[1]);
  const double z01 = 0.5 * (q.z[0] + q.z[1]);
  const double r12 = 0.5 * (q.r[1] + q.r[2]);
  const double z12 = 0.5 * (q.z[1] + q.z[2]);
  const double r23 = 0.5 * (q.r[2] + q.r[3]);
  const double z23 = 0.5 * (q.z[2] + q.z[3]);
  const double r30 = 0.5 * (q.r[3] + q.r[0]);
  const double z30 = 0.5 * (q.z[3] + q.z[0]);
  const double rc = 0.25 * (q.r[0] + q.r[1] + q.r[2] + q.r[3]);
  const double zc = 0.25 * (q.z[0] + q.z[1] + q.z[2] + q.z[3]);

  out0.r[0] = q.r[0];
  out0.z[0] = q.z[0];
  out0.r[1] = r01;
  out0.z[1] = z01;
  out0.r[2] = rc;
  out0.z[2] = zc;
  out0.r[3] = r30;
  out0.z[3] = z30;

  out1.r[0] = r01;
  out1.z[0] = z01;
  out1.r[1] = q.r[1];
  out1.z[1] = q.z[1];
  out1.r[2] = r12;
  out1.z[2] = z12;
  out1.r[3] = rc;
  out1.z[3] = zc;

  out2.r[0] = rc;
  out2.z[0] = zc;
  out2.r[1] = r12;
  out2.z[1] = z12;
  out2.r[2] = q.r[2];
  out2.z[2] = q.z[2];
  out2.r[3] = r23;
  out2.z[3] = z23;

  out3.r[0] = r30;
  out3.z[0] = z30;
  out3.r[1] = rc;
  out3.z[1] = zc;
  out3.r[2] = r23;
  out3.z[2] = z23;
  out3.r[3] = q.r[3];
  out3.z[3] = q.z[3];

  return {out0, out1, out2, out3};
}

void add_estimate(VolumeCutResult& out, const LeafEstimate& leaf) {
  out.rho_volume_avg += leaf.rho_int;
  out.Te_volume_avg += leaf.Te_int;
  out.Ti_volume_avg += leaf.Ti_int;
  for (std::size_t m = 0; m < out.volfrac.size(); ++m) {
    out.volfrac[m] += leaf.mat_volume[m];
  }
}

double material_error(const LeafEstimate& parent,
                      const std::array<LeafEstimate, 4>& children) {
  double err = 0.0;
  for (std::size_t m = 0; m < parent.mat_volume.size(); ++m) {
    double sum = 0.0;
    for (const LeafEstimate& child : children) {
      sum += child.mat_volume[m];
    }
    err = std::max(err, std::abs(parent.mat_volume[m] - sum));
  }
  return err;
}

void recurse_leaf(const Quad& q,
                  const std::size_t n_mat,
                  const GeometryCallables& callables,
                  const int max_depth,
                  const double volfrac_tol,
                  const bool use_3x3_quadrature,
                  const int depth,
                  VolumeCutResult& out) {
  out.max_depth_reached = std::max(out.max_depth_reached, depth);
  const LeafEstimate parent =
      estimate_leaf(q, n_mat, callables, use_3x3_quadrature);
  const auto quads = subdivide(q);
  std::array<LeafEstimate, 4> child_estimates = {
      estimate_leaf(quads[0], n_mat, callables, use_3x3_quadrature),
      estimate_leaf(quads[1], n_mat, callables, use_3x3_quadrature),
      estimate_leaf(quads[2], n_mat, callables, use_3x3_quadrature),
      estimate_leaf(quads[3], n_mat, callables, use_3x3_quadrature)};

  const double err = material_error(parent, child_estimates);
  const double tol = volfrac_tol * std::max(parent.volume, 1.0e-300);
  if (err <= tol || depth >= max_depth) {
    if (depth >= max_depth && err > tol) {
      out.converged = false;
    }
    add_estimate(out, parent);
    ++out.leaf_count;
    return;
  }

  for (const Quad& child : quads) {
    recurse_leaf(child, n_mat, callables, max_depth, volfrac_tol,
                 use_3x3_quadrature, depth + 1, out);
  }
}

}  // namespace

VolumeCutResult adaptive_volume_cut_sample_cell(const double r0,
                                                const double z0,
                                                const double r1,
                                                const double z1,
                                                const double r2,
                                                const double z2,
                                                const double r3,
                                                const double z3,
                                                const std::size_t n_mat,
                                                const NamelistCallables& callables,
                                                const int max_depth,
                                                const double volfrac_tol,
                                                const bool use_3x3_quadrature) {
  VolumeCutResult out;
  out.volfrac.assign(n_mat, 0.0);
  out.converged = true;

  Quad q{{r0, r1, r2, r3}, {z0, z1, z2, z3}};
  out.total_volume = quad_volume(q);
  if (!(out.total_volume > 0.0) || n_mat == 0) {
    out.converged = false;
    return out;
  }

  const int clamped_depth = std::clamp(max_depth, 0, 16);
  recurse_leaf(q, n_mat, callables, clamped_depth, std::max(volfrac_tol, 0.0),
               use_3x3_quadrature, 0, out);

  const double inv_v = 1.0 / out.total_volume;
  out.rho_volume_avg *= inv_v;
  out.Te_volume_avg *= inv_v;
  out.Ti_volume_avg *= inv_v;
  for (double& f : out.volfrac) {
    f = clamp01(f * inv_v);
  }
  return out;
}

}  // namespace tenryu::core::namelist
