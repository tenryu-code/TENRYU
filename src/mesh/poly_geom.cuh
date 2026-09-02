#pragma once

#include <cstdint>

#include "mesh/reale_contracts.hpp"

namespace tenryu::mesh::reale {

RZMoments polygon_rz_moments(const double* r, const double* z, int n);

int clip_polygon_convex(const double* subj_r,
                        const double* subj_z,
                        int subj_n,
                        const double* clip_r,
                        const double* clip_z,
                        int clip_n,
                        double* out_r,
                        double* out_z,
                        int out_cap);

RZMoments intersect_general(const double* a_r,
                            const double* a_z,
                            int a_n,
                            const double* b_r,
                            const double* b_z,
                            int b_n);

int orient2d_sign(double ax,
                  double ay,
                  double bx,
                  double by,
                  double cx,
                  double cy);

int incircle_sign(double ax,
                  double ay,
                  double bx,
                  double by,
                  double cx,
                  double cy,
                  double dx,
                  double dy);

std::uint64_t polygon_topo_hash(const EntityId* node_ids, int n);

}  // namespace tenryu::mesh::reale
