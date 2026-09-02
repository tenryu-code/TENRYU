#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace tenryu::hydro::ixremap {

struct Polygon {
  std::vector<double> r;
  std::vector<double> z;
};

double polygon_rz_volume(const Polygon& p);

Polygon clip_convex(const Polygon& subject, const Polygon& clip);

struct QuadSplit {
  Polygon tri[2];
  int n = 0;
};

QuadSplit split_quad_deterministic(const double r[4], const double z[4]);

double quad_pair_overlap_rz_volume(const double ra[4],
                                   const double za[4],
                                   const double rc[4],
                                   const double zc[4]);

struct PatchIntersection {
  int n_src = 0;
  int n_dst = 0;
  std::vector<double> overlap;
  std::vector<double> src_vol;
  std::vector<double> dst_vol;
  double max_row_defect = 0.0;
  double max_col_defect = 0.0;
  int split_failures = 0;
  int worst_row_index = -1;
  int worst_col_index = -1;
};

PatchIntersection build_patch_intersection(
    const std::vector<std::array<double, 8>>& src_quads,
    const std::vector<std::array<double, 8>>& dst_quads);

void transfer_piecewise_constant(
    const PatchIntersection& ix,
    const std::vector<double>& src_extensive,
    std::vector<double>& dst_extensive);

void transfer_renormalized(
    const PatchIntersection& ix,
    const std::vector<double>& src_extensive,
    std::vector<double>& dst_extensive,
    std::vector<std::uint8_t>* orphaned /* per-source flag out, nullable */);

void zonal_momentum_and_ke(const double corner_masses[4],
                           const double vr[4],
                           const double vz[4],
                           double& Pr,
                           double& Pz,
                           double& K);

void distribute_corner_masses_and_momenta(
    double m_new,
    double Pr_new,
    double Pz_new,
    const double corner_weights[4],
    double corner_masses_out[4],
    double corner_pr_out[4],
    double corner_pz_out[4]);

}  // namespace tenryu::hydro::ixremap
