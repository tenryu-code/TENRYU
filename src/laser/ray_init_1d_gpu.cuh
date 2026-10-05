#pragma once

// The ring weights and rays of a 1D beam on the device (ray_init.cu initialize_rays_1d): the beam's
// intensity profile at the rings' reference radii times the annulus areas, their sum, the ring
// powers and, on a sphere, the rays (one per ring in the plane through the beam axis). These were
// computed on the host every step and copied up. The arithmetic repeats the host's operation by
// operation without floating-point contraction (the file is compiled with -fmad=false), the sum of
// the weights in the rings' order in one thread; the profile's exp and pow are the device's, which
// can differ from the host's in the last place.

#include <cuda_runtime.h>

#include "laser/beams.cuh"

namespace tenryu::laser::ray_init_1d {

// The ring layout of ring_layout_1d: ring k at R_k = (k + 1/2) dR of the launch plane Z_init, the
// profile evaluated at R_k reference_scale.
struct RingLayout {
  double Z_init = 0.0;
  double z_focus = 0.0;
  double dR = 0.0;
  double reference_scale = 1.0;
};

// The layout for a launch plane Z_init (LaserMesh::Z_max), as ring_layout_1d computes it.
RingLayout ring_layout(const Beam& beam, double Z_init, int rays_per_beam);

// The ring powers beam_power * w_k / sum_w (device [rays_per_beam]): w_k = profile(R_k
// reference_scale) * max(annulus area, 0), or 1 for every ring when the weights sum to no more
// than zero.
void ring_powers(const Beam& beam, const RingLayout& layout, int rays_per_beam, double beam_power,
                 double* ring_power, cudaStream_t stream);

// The sphere's rays of the rings (device arrays [rays_per_beam]): launch point (R_k, Z_init),
// direction along the line through the focus toward -Z, power and initial power the ring power,
// out-of-plane velocity 0.
void sphere_rays(const RingLayout& layout, int rays_per_beam, const double* ring_power, double* R0,
                 double* Z0, double* vR0, double* vZ0, double* vA0, double* power, double* power0,
                 cudaStream_t stream);

}  // namespace tenryu::laser::ray_init_1d
