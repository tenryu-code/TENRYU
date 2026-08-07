#pragma once

#include <algorithm>
#include <cmath>

#ifndef TENRYU_HOST_DEVICE
#define TENRYU_HOST_DEVICE __host__ __device__
#endif

namespace tenryu::laser {

struct BilinearCell {
  int i = 0;
  int j = 0;
  double xi = 0.0;
  double eta = 0.0;
};

struct BilinearWeights {
  double w00 = 1.0;
  double w10 = 0.0;
  double w01 = 0.0;
  double w11 = 0.0;
};

struct BilinearInterp {
  TENRYU_HOST_DEVICE static inline int locate_interval(const double* nodes,
                                                       const int n_nodes,
                                                       const double x) {
    if (n_nodes <= 1) {
      return 0;
    }
    if (x <= nodes[0]) {
      return 0;
    }
    if (x >= nodes[n_nodes - 1]) {
      return n_nodes - 2;
    }

    int lo = 0;
    int hi = n_nodes - 1;
    while (hi - lo > 1) {
      const int mid = lo + (hi - lo) / 2;
      if (nodes[mid] <= x) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    return lo;
  }

  TENRYU_HOST_DEVICE static inline BilinearCell locate_cell(const double* node_R,
                                                            const double* node_Z,
                                                            const int n_nodes_r,
                                                            const int n_nodes_z,
                                                            const double R,
                                                            const double Z) {
    BilinearCell c;
    c.i = locate_interval(node_R, n_nodes_r, R);
    c.j = locate_interval(node_Z, n_nodes_z, Z);

    const double dR = node_R[c.i + 1] - node_R[c.i];
    const double dZ = node_Z[c.j + 1] - node_Z[c.j];

    c.xi = (dR > 0.0) ? (R - node_R[c.i]) / dR : 0.0;
    c.eta = (dZ > 0.0) ? (Z - node_Z[c.j]) / dZ : 0.0;
    c.xi = ::fmin(1.0, ::fmax(0.0, c.xi));
    c.eta = ::fmin(1.0, ::fmax(0.0, c.eta));
    return c;
  }

  TENRYU_HOST_DEVICE static inline BilinearCell locate_cell_local(const double* node_R,
                                                                  const double* node_Z,
                                                                  const int n_nodes_r,
                                                                  const int n_nodes_z,
                                                                  const double R,
                                                                  const double Z,
                                                                  const int i_hint,
                                                                  const int j_hint) {
    BilinearCell c;
    c.i = i_hint < 0 ? 0 : (i_hint > n_nodes_r - 2 ? n_nodes_r - 2 : i_hint);
    c.j = j_hint < 0 ? 0 : (j_hint > n_nodes_z - 2 ? n_nodes_z - 2 : j_hint);

    while (c.i > 0 && R < node_R[c.i]) {
      --c.i;
    }
    while (c.i < n_nodes_r - 2 && R >= node_R[c.i + 1]) {
      ++c.i;
    }

    while (c.j > 0 && Z < node_Z[c.j]) {
      --c.j;
    }
    while (c.j < n_nodes_z - 2 && Z >= node_Z[c.j + 1]) {
      ++c.j;
    }

    const double dR = node_R[c.i + 1] - node_R[c.i];
    const double dZ = node_Z[c.j + 1] - node_Z[c.j];

    c.xi = (dR > 0.0) ? (R - node_R[c.i]) / dR : 0.0;
    c.eta = (dZ > 0.0) ? (Z - node_Z[c.j]) / dZ : 0.0;
    c.xi = ::fmin(1.0, ::fmax(0.0, c.xi));
    c.eta = ::fmin(1.0, ::fmax(0.0, c.eta));
    return c;
  }

  TENRYU_HOST_DEVICE static inline BilinearWeights compute_weights(
      const double xi,
      const double eta) {
    BilinearWeights w;
    const double one_minus_xi = 1.0 - xi;
    const double one_minus_eta = 1.0 - eta;

    w.w00 = one_minus_xi * one_minus_eta;
    w.w10 = xi * one_minus_eta;
    w.w01 = one_minus_xi * eta;
    w.w11 = xi * eta;
    return w;
  }

  TENRYU_HOST_DEVICE static inline double interpolate(const double* field,
                                                      const int stride,
                                                      const BilinearCell& c,
                                                      const BilinearWeights& w) {
    const int n00 = c.i * stride + c.j;
    const int n10 = (c.i + 1) * stride + c.j;
    const int n01 = c.i * stride + (c.j + 1);
    const int n11 = (c.i + 1) * stride + (c.j + 1);

    return w.w00 * field[n00] + w.w10 * field[n10] + w.w01 * field[n01] +
           w.w11 * field[n11];
  }

  TENRYU_HOST_DEVICE static inline void interpolate_gradient(
      const double* grad_field_R,
      const double* grad_field_Z,
      const int stride,
      const BilinearCell& c,
      const BilinearWeights& w,
      double& out_dndR,
      double& out_dndZ) {
    out_dndR = interpolate(grad_field_R, stride, c, w);
    out_dndZ = interpolate(grad_field_Z, stride, c, w);
  }
};

}  // namespace tenryu::laser
