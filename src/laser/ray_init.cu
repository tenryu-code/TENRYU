#include "laser/ray_init.cuh"
#include "core/nvtx_range.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

#include "core/device_scratch.hpp"
#include "laser/coordinate_transform.cuh"

namespace tenryu::laser {
namespace {

constexpr double kPi = 3.14159265358979323846;

inline void cuda_check(const cudaError_t err, const char* message) {
  TENRYU_ASSERT(err == cudaSuccess, message);
}

bool intersect_axis_with_lasermesh(const Vec3& source,
                                   const Vec3& dir,
                                   const LaserMesh& lmesh,
                                   Vec3& entry) {
  constexpr double kEps = 1.0e-30;
  double t_min = 1.0e300;
  bool found = false;

  auto try_t = [&](const double t) {
    if (!(t > 0.0) || !std::isfinite(t)) {
      return;
    }
    const double x = source.x + t * dir.x;
    const double y = source.y + t * dir.y;
    const double z = source.z + t * dir.z;
    const double R = std::sqrt(x * x + y * y);
    if (R <= lmesh.R_max + 1.0e-12 && z >= lmesh.Z_min - 1.0e-12 && z <= lmesh.Z_max + 1.0e-12) {
      if (t < t_min) {
        t_min = t;
        found = true;
      }
    }
  };

  if (std::abs(dir.z) > kEps) {
    try_t((lmesh.Z_min - source.z) / dir.z);
    try_t((lmesh.Z_max - source.z) / dir.z);
  }

  const double a = dir.x * dir.x + dir.y * dir.y;
  if (a > kEps) {
    const double b = 2.0 * (source.x * dir.x + source.y * dir.y);
    const double c = source.x * source.x + source.y * source.y - lmesh.R_max * lmesh.R_max;
    const double disc = b * b - 4.0 * a * c;
    if (disc >= 0.0) {
      const double sqrt_disc = std::sqrt(disc);
      const double inv2a = 0.5 / a;
      try_t((-b - sqrt_disc) * inv2a);
      try_t((-b + sqrt_disc) * inv2a);
    }
  }

  if (!found) {
    return false;
  }
  entry = Vec3{source.x + t_min * dir.x, source.y + t_min * dir.y, source.z + t_min * dir.z};
  return true;
}

bool point_inside_lasermesh(const Vec3& p, const LaserMesh& lmesh) {
  const double R = std::sqrt(p.x * p.x + p.y * p.y);
  return (R <= lmesh.R_max + 1.0e-12) && (p.z >= lmesh.Z_min - 1.0e-12) &&
         (p.z <= lmesh.Z_max + 1.0e-12);
}

bool point_on_lasermesh_boundary(const Vec3& p, const LaserMesh& lmesh) {
  const double R = std::sqrt(p.x * p.x + p.y * p.y);
  const double scale = std::max({1.0, lmesh.R_max, std::abs(lmesh.Z_min), std::abs(lmesh.Z_max)});
  const double tol = 1.0e-9 * scale;
  const bool inside = (R <= lmesh.R_max + tol) && (p.z >= lmesh.Z_min - tol) &&
                      (p.z <= lmesh.Z_max + tol);
  if (!inside) {
    return false;
  }
  const bool on_r = std::abs(R - lmesh.R_max) <= tol;
  const bool on_zmin = std::abs(p.z - lmesh.Z_min) <= tol;
  const bool on_zmax = std::abs(p.z - lmesh.Z_max) <= tol;
  return on_r || on_zmin || on_zmax;
}

bool clip_ray_start_if_outside(Vec3& start, const Vec3& dir, const LaserMesh& lmesh) {
  if (point_inside_lasermesh(start, lmesh)) {
    return true;
  }
  Vec3 clipped{};
  if (!intersect_axis_with_lasermesh(start, dir, lmesh, clipped)) {
    return false;
  }
  start = clipped;
  return point_inside_lasermesh(start, lmesh);
}

double beam_aperture_radius_cm(const Beam& beam, const LaserMesh& lmesh) {
  const double f = std::max(beam.f_number, 1.0e-12);
  const Vec3 d_hat = normalize(Vec3{beam.dir_x, beam.dir_y, beam.dir_z});
  const Vec3 focus{beam.focus_x, beam.focus_y, beam.focus_lab_z};
  const double extent = std::max({1.0, lmesh.R_max, std::abs(lmesh.Z_min), std::abs(lmesh.Z_max)});
  double L_source = 2.0 * extent;
  Vec3 source = sub(focus, mul(d_hat, L_source));
  for (int iter = 0; iter < 16 && point_inside_lasermesh(source, lmesh); ++iter) {
    L_source *= 2.0;
    source = sub(focus, mul(d_hat, L_source));
  }

  Vec3 entry{};
  if (intersect_axis_with_lasermesh(source, d_hat, lmesh, entry)) {
    const double L_entry = norm(sub(entry, focus));
    if (L_entry > 0.0 && std::isfinite(L_entry)) {
      return L_entry / (2.0 * f);
    }
  }

  const double dz = std::abs(beam.dir_z);
  if (dz > 1.0e-12) {
    const double z_entry = (beam.dir_z < 0.0) ? lmesh.Z_max : lmesh.Z_min;
    return std::abs(z_entry - beam.focus_lab_z) / (2.0 * f * dz);
  }

  return std::max(lmesh.R_max, 1.0e-12) / (2.0 * f);
}

}  // namespace

RayArray1D::~RayArray1D() {
  release();
}

RayArray1D::RayArray1D(RayArray1D&& other) noexcept {
  *this = std::move(other);
}

RayArray1D& RayArray1D::operator=(RayArray1D&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();

  R0 = other.R0;
  Z0 = other.Z0;
  vR0 = other.vR0;
  vZ0 = other.vZ0;
  vA0 = other.vA0;
  power = other.power;
  power0 = other.power0;
  n_rays = other.n_rays;
  n_rays_capacity = other.n_rays_capacity;
  pooled = other.pooled;

  other.R0 = nullptr;
  other.Z0 = nullptr;
  other.vR0 = nullptr;
  other.vZ0 = nullptr;
  other.vA0 = nullptr;
  other.power = nullptr;
  other.power0 = nullptr;
  other.n_rays = 0;
  other.n_rays_capacity = 0;
  other.pooled = false;
  return *this;
}

void RayArray1D::release() {
  if (pooled) {
    R0 = Z0 = vR0 = vZ0 = vA0 = power = power0 = nullptr;
    n_rays = 0;
    n_rays_capacity = 0;
    pooled = false;
    return;
  }
  if (R0 != nullptr) {
    cuda_check(cudaFree(R0), "RayArray1D::release cudaFree R0 failed");
    R0 = nullptr;
  }
  if (Z0 != nullptr) {
    cuda_check(cudaFree(Z0), "RayArray1D::release cudaFree Z0 failed");
    Z0 = nullptr;
  }
  if (vR0 != nullptr) {
    cuda_check(cudaFree(vR0), "RayArray1D::release cudaFree vR0 failed");
    vR0 = nullptr;
  }
  if (vZ0 != nullptr) {
    cuda_check(cudaFree(vZ0), "RayArray1D::release cudaFree vZ0 failed");
    vZ0 = nullptr;
  }
  if (vA0 != nullptr) {
    cuda_check(cudaFree(vA0), "RayArray1D::release cudaFree vA0 failed");
    vA0 = nullptr;
  }
  if (power != nullptr) {
    cuda_check(cudaFree(power), "RayArray1D::release cudaFree power failed");
    power = nullptr;
  }
  if (power0 != nullptr) {
    cuda_check(cudaFree(power0), "RayArray1D::release cudaFree power0 failed");
    power0 = nullptr;
  }
  n_rays = 0;
  n_rays_capacity = 0;
}

bool RayArray1D::empty() const {
  return n_rays == 0;
}

void RayArray1D::allocate(const int n) {
  if (!pooled && n <= n_rays_capacity) {
    n_rays = std::max(n, 0);
    return;
  }
  release();
  if (n <= 0) {
    return;
  }
  n_rays = n;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
  auto alloc_or_cleanup = [&](double*& ptr, const char* message) {
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err == cudaSuccess) {
      return;
    }
    release();
    TENRYU_ASSERT(false, message);
  };

  alloc_or_cleanup(R0, "RayArray1D::allocate cudaMalloc R0 failed");
  alloc_or_cleanup(Z0, "RayArray1D::allocate cudaMalloc Z0 failed");
  alloc_or_cleanup(vR0, "RayArray1D::allocate cudaMalloc vR0 failed");
  alloc_or_cleanup(vZ0, "RayArray1D::allocate cudaMalloc vZ0 failed");
  alloc_or_cleanup(vA0, "RayArray1D::allocate cudaMalloc vA0 failed");
  alloc_or_cleanup(power, "RayArray1D::allocate cudaMalloc power failed");
  alloc_or_cleanup(power0, "RayArray1D::allocate cudaMalloc power0 failed");
  n_rays_capacity = n;
}

void RayArray1D::allocate_pooled(const int n) {
  release();
  if (n <= 0) {
    return;
  }
  n_rays = n;
  pooled = true;
  const std::size_t n_sz = static_cast<std::size_t>(n);
  double* slab = static_cast<double*>(core::device_scratch_acquire(
      "ray_init:ray_array_1d_slab", 7ULL * n_sz * sizeof(double)));
  R0 = slab + 0 * n_sz;
  Z0 = slab + 1 * n_sz;
  vR0 = slab + 2 * n_sz;
  vZ0 = slab + 3 * n_sz;
  power = slab + 4 * n_sz;
  power0 = slab + 5 * n_sz;
  vA0 = slab + 6 * n_sz;
}

void RayArray1D::copy_from_host(const std::vector<Ray2D>& rays, cudaStream_t stream) {
  allocate_pooled(static_cast<int>(rays.size()));
  if (rays.empty()) {
    return;
  }

  const std::size_t n = rays.size();
  std::vector<double> h_slab(7 * n, 0.0);  // vA0 (slot 6) is 0: rays in the trace's plane

  for (std::size_t i = 0; i < n; ++i) {
    h_slab[0 * n + i] = rays[i].R;
    h_slab[1 * n + i] = rays[i].Z;
    h_slab[2 * n + i] = rays[i].vR;
    h_slab[3 * n + i] = rays[i].vZ;
    h_slab[4 * n + i] = rays[i].I;
    h_slab[5 * n + i] = rays[i].I0;
  }

  // Pageable-source async copy is synchronous with respect to h_slab lifetime per CUDA.
  cuda_check(cudaMemcpyAsync(R0, h_slab.data(), 7 * n * sizeof(double),
                             cudaMemcpyHostToDevice, stream),
             "RayArray1D::copy_from_host memcpyAsync slab failed");
}

namespace {

// Ring layout and weights of a 1D beam (face-based annular quadrature):
// ring k at R_k = (k + 1/2) dR of the launch plane Z_init = lmesh.Z_max, dR =
// R_beam / rays_per_beam, R_beam = |Z_init - z_focus| / (2 f_number); weight
// profile(R_k R_ref / R_beam-scaled) times the annulus area.
struct RingLayout1D {
  double Z_init = 0.0;
  double z_focus = 0.0;
  double dR = 0.0;
  std::vector<double> weights;
  double sum_w = 0.0;
};

RingLayout1D ring_layout_1d(const Beam& beam, const LaserMesh& lmesh, const int rays_per_beam) {
  RingLayout1D out;
  out.Z_init = lmesh.Z_max;
  // The beam propagates toward -Z from the launch plane Z_init, whatever its
  // lab direction (spherical symmetry); z_focus is the focus on that axis.
  out.z_focus = beam.axial_focus_1d();
  const double R_beam =
      std::abs(out.Z_init - out.z_focus) / (2.0 * std::max(beam.f_number, 1.0e-12));
  out.dR = (rays_per_beam > 0) ? (R_beam / static_cast<double>(rays_per_beam)) : 0.0;

  // Face-based annular quadrature (2026-07-26 review): ring faces at k*dR,
  // representative radii at the ring centers (k+1/2)*dR, exact annulus areas
  // pi*((k+1)^2 - k^2)*dR^2. The previous node-based layout (rays at k*dR
  // with a half-cell disk at k=0) left the outermost half ring
  // [R_beam - dR/2, R_beam] uncovered, biasing the sampled profile moments
  // inward by O(1/N).
  // The intensity profile is defined on the plane through the target centre
  // (lab z = 0) — the plane whose spot the Langdon vacuum map also describes —
  // and carried to the launch plane along the straight rays through the focus:
  // R_ref = R_k |z_f| / |Z_init - z_f|. R_k / R_beam is fixed per ring, so the
  // weights no longer depend on where the laser mesh currently ends (the former
  // evaluation at the moving launch plane with a fixed w0 narrowed the beam as
  // the corona expanded; 2026-09-23). A focus at the target centre maps every
  // ring to R_ref = 0: uniform intensity over the f-cone.
  const double launch_to_focus = std::abs(out.Z_init - out.z_focus);
  const double reference_scale =
      (launch_to_focus > 0.0) ? std::abs(out.z_focus) / launch_to_focus : 1.0;
  out.weights.assign(static_cast<std::size_t>(rays_per_beam), 0.0);
  for (int k = 0; k < rays_per_beam; ++k) {
    const double Rk = out.dR * (static_cast<double>(k) + 0.5);
    const double area = kPi * (2.0 * static_cast<double>(k) + 1.0) * out.dR * out.dR;
    const double wk = beam.profile(Rk * reference_scale) * std::max(area, 0.0);
    out.weights[static_cast<std::size_t>(k)] = wk;
    out.sum_w += wk;
  }
  if (!(out.sum_w > 0.0)) {
    std::fill(out.weights.begin(), out.weights.end(), 1.0);
    out.sum_w = static_cast<double>(rays_per_beam);
  }
  return out;
}

struct Vec3d {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

__host__ __device__ inline Vec3d v3(const double x, const double y, const double z) {
  Vec3d v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}
__host__ __device__ inline Vec3d v3_add(const Vec3d a, const Vec3d b) {
  return v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__host__ __device__ inline Vec3d v3_scale(const Vec3d a, const double s) {
  return v3(a.x * s, a.y * s, a.z * s);
}
__host__ __device__ inline double v3_dot(const Vec3d a, const Vec3d b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline Vec3d v3_cross(const Vec3d a, const Vec3d b) {
  return v3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
__host__ __device__ inline Vec3d v3_unit(const Vec3d a) {
  const double n = sqrt(v3_dot(a, a));
  return (n > 0.0) ? v3_scale(a, 1.0 / n) : a;
}

// Beam frame of a non-spherical 1D geometry: propagation p (unit), ring
// frame e1, e2 (unit, perpendicular to p), the trace's plane axes (unit
// r_axis, z_axis; a_axis out of the plane), the focus point F = -z_focus p
// (z_focus positive toward the source) and the launch-plane distance.
struct BeamFrame1D {
  Vec3d p;
  Vec3d e1;
  Vec3d e2;
  Vec3d r_axis;
  Vec3d z_axis;
  Vec3d a_axis;
  Vec3d focus;
  double Z_init = 0.0;
  double dz = 0.0;  // Z_init - z_focus
  double profile_end = 0.0;
  int planar = 0;
};

// Ray (ring k, azimuth m) of a non-spherical 1D geometry: the 3D ray through
// the focus from the ring point of the launch plane, moved along its line to
// the outside of the profile (the cylinder r = profile_end, the slab height
// profile_end), and expressed in the trace's plane.
__global__ void initialize_rays_1d_geometry_kernel(const BeamFrame1D frame,
                                                   const double* __restrict__ ring_power,
                                                   const int n_rings,
                                                   const int n_azimuth,
                                                   const double dR,
                                                   double* __restrict__ R0,
                                                   double* __restrict__ Z0,
                                                   double* __restrict__ vR0,
                                                   double* __restrict__ vZ0,
                                                   double* __restrict__ vA0,
                                                   double* __restrict__ power,
                                                   double* __restrict__ power0) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_rings * n_azimuth) {
    return;
  }
  const int k = idx / n_azimuth;
  const int m = idx - k * n_azimuth;
  const double Rk = dR * (static_cast<double>(k) + 0.5);
  const double psi = 2.0 * kPi * (static_cast<double>(m) + 0.5) / static_cast<double>(n_azimuth);
  const Vec3d offset =
      v3_add(v3_scale(frame.e1, Rk * cos(psi)), v3_scale(frame.e2, Rk * sin(psi)));
  const Vec3d P = v3_add(v3_scale(frame.p, -frame.Z_init), offset);
  // Converging on a focus beyond the launch plane, diverging from one the beam
  // has already passed (the sphere's rule).
  const Vec3d to_focus = v3_add(frame.focus, v3_scale(P, -1.0));
  Vec3d d = v3_unit((frame.dz >= 0.0) ? to_focus : v3_scale(to_focus, -1.0));
  if (!(v3_dot(d, d) > 0.0)) {
    d = frame.p;
  }
  // Start on the ray's line outside the profile (vacuum lies between).
  double t = 0.0;
  if (frame.planar != 0) {
    const double h = v3_dot(P, frame.z_axis);
    const double dh = v3_dot(d, frame.z_axis);
    if (h < frame.profile_end && dh < 0.0) {
      t = (frame.profile_end - h) / dh;  // < 0: back along the line
    }
  } else {
    const double px = v3_dot(P, frame.r_axis);
    const double pz = v3_dot(P, frame.z_axis);
    const double dx = v3_dot(d, frame.r_axis);
    const double dzt = v3_dot(d, frame.z_axis);
    const double a = dx * dx + dzt * dzt;
    const double c = px * px + pz * pz - frame.profile_end * frame.profile_end;
    if (c < 0.0 && a > 0.0) {
      // |P_perp + t d_perp| = profile_end, the root with t < 0.
      const double b = px * dx + pz * dzt;
      const double disc = b * b - a * c;
      t = (-b - sqrt(fmax(disc, 0.0))) / a;
    }
  }
  const Vec3d S = v3_add(P, v3_scale(d, t));
  R0[idx] = v3_dot(S, frame.r_axis);
  Z0[idx] = v3_dot(S, frame.z_axis);
  vR0[idx] = v3_dot(d, frame.r_axis);
  vZ0[idx] = v3_dot(d, frame.z_axis);
  vA0[idx] = v3_dot(d, frame.a_axis);
  const double w = ring_power[k] / static_cast<double>(n_azimuth);
  power[idx] = w;
  power0[idx] = w;
}

}  // namespace

int max_rays_1d_per_beam(const LaserMesh& lmesh, const int rays_per_beam,
                         const int azimuthal_rays) {
  return (lmesh.geometry_code == 0) ? rays_per_beam
                                    : rays_per_beam * std::max(azimuthal_rays, 1);
}

RayArray1D initialize_rays_1d(const Beam& beam,
                              const LaserMesh& lmesh,
                              const int rays_per_beam,
                              const double beam_power,
                              cudaStream_t stream,
                              const int azimuthal_rays) {
  const core::NvtxRange nvtx_range("laser.ray_initialization");
  RayArray1D out;
  if (rays_per_beam <= 0 || !(beam_power > 0.0)) {
    return out;
  }
  const RingLayout1D layout = ring_layout_1d(beam, lmesh, rays_per_beam);
  const double Z_init = layout.Z_init;
  const double z_focus = layout.z_focus;
  const double dR = layout.dR;
  const std::vector<double>& weights = layout.weights;
  const double sum_w = layout.sum_w;

  if (lmesh.geometry_code != 0) {
    // Cylinder (axis g = lab z) or slab (normal g = lab z, the vacuum side
    // at +g): the beam's lab direction sets the incidence.
    const Vec3d g = v3(0.0, 0.0, 1.0);
    Vec3d p = v3_unit(v3(beam.dir_x, beam.dir_y, beam.dir_z));
    BeamFrame1D frame;
    frame.planar = (lmesh.geometry_code == 2) ? 1 : 0;
    if (frame.planar != 0 && p.z > 0.0) {
      p.z = -p.z;  // the beam comes from the vacuum side
    }
    frame.p = p;
    const Vec3d gxp = v3_cross(g, p);
    const bool along_g = v3_dot(gxp, gxp) < 1.0e-24;
    frame.e1 = along_g ? v3(1.0, 0.0, 0.0) : v3_unit(gxp);
    frame.e2 = v3_cross(p, frame.e1);
    frame.Z_init = Z_init;
    frame.dz = Z_init - z_focus;
    frame.focus = v3_scale(p, -z_focus);
    frame.profile_end = lmesh.R_max;  // the profile ends at the laser mesh radius
    if (frame.planar != 0) {
      // Plane (lateral, height): lateral along the beam's tilt.
      frame.z_axis = g;
      const Vec3d tilt = v3_add(p, v3_scale(g, -v3_dot(p, g)));
      frame.r_axis = (v3_dot(tilt, tilt) > 1.0e-24) ? v3_unit(tilt) : v3(1.0, 0.0, 0.0);
      frame.a_axis = v3_cross(g, frame.r_axis);
    } else {
      // Cross-section plane: Z back toward the source (the rays travel toward
      // -Z, as on a sphere), A along the cylinder axis.
      const Vec3d p_perp = v3(p.x, p.y, 0.0);
      frame.z_axis = v3_unit(v3_scale(p_perp, -1.0));
      frame.r_axis = v3_cross(g, frame.z_axis);
      frame.a_axis = g;
    }
    // A slab at normal incidence: every azimuth gives the same ray.
    const int n_azimuth = (frame.planar != 0 && along_g) ? 1 : std::max(azimuthal_rays, 1);
    std::vector<double> ring_power(static_cast<std::size_t>(rays_per_beam), 0.0);
    for (int k = 0; k < rays_per_beam; ++k) {
      ring_power[static_cast<std::size_t>(k)] =
          beam_power * (weights[static_cast<std::size_t>(k)] / sum_w);
    }
    double* d_ring_power = static_cast<double*>(core::device_scratch_acquire(
        "ray_init:ring_power", static_cast<std::size_t>(rays_per_beam) * sizeof(double)));
    cuda_check(cudaMemcpyAsync(d_ring_power, ring_power.data(),
                               static_cast<std::size_t>(rays_per_beam) * sizeof(double),
                               cudaMemcpyHostToDevice, stream),
               "initialize_rays_1d ring power H2D failed");
    const int n_rays = rays_per_beam * n_azimuth;
    out.allocate_pooled(n_rays);
    const int block = 128;
    initialize_rays_1d_geometry_kernel<<<(n_rays + block - 1) / block, block, 0, stream>>>(
        frame, d_ring_power, rays_per_beam, n_azimuth, dR, out.R0, out.Z0, out.vR0, out.vZ0,
        out.vA0, out.power, out.power0);
    cuda_check(cudaGetLastError(), "initialize_rays_1d geometry kernel launch failed");
    return out;
  }

  std::vector<Ray2D> rays(static_cast<std::size_t>(rays_per_beam));
  for (int k = 0; k < rays_per_beam; ++k) {
    const double Rk = dR * (static_cast<double>(k) + 0.5);
    const double dz = Z_init - z_focus;
    const double L = std::sqrt(Rk * Rk + dz * dz);

    Ray2D ray;
    ray.R = Rk;
    ray.Z = Z_init;
    // Along the line through the focus, toward -Z: converging on a focus
    // below the launch plane, diverging from one the beam has already passed
    // above it. (The rays used to be launched toward +Z, away from the target,
    // in the second case.)
    ray.vR = (L > 0.0) ? ((dz >= 0.0) ? (-Rk / L) : (Rk / L)) : 0.0;
    const double vR2 = ray.vR * ray.vR;
    ray.vZ = -std::sqrt(std::max(0.0, 1.0 - vR2));

    const double w_norm = weights[static_cast<std::size_t>(k)] / sum_w;
    ray.I = beam_power * w_norm;
    ray.I0 = ray.I;
    ray.beam_id = beam.wave_id;
    ray.alive = (ray.I > 0.0) ? 1 : 0;
    rays[static_cast<std::size_t>(k)] = ray;
  }

  out.copy_from_host(rays, stream);
  return out;
}

RayArray2D::~RayArray2D() {
  release();
}

RayArray2D::RayArray2D(RayArray2D&& other) noexcept {
  *this = std::move(other);
}

RayArray2D& RayArray2D::operator=(RayArray2D&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  x0 = other.x0;
  y0 = other.y0;
  z0 = other.z0;
  vx0 = other.vx0;
  vy0 = other.vy0;
  vz0 = other.vz0;
  power = other.power;
  power0 = other.power0;
  n_rays = other.n_rays;
  n_rays_capacity = other.n_rays_capacity;

  other.x0 = nullptr;
  other.y0 = nullptr;
  other.z0 = nullptr;
  other.vx0 = nullptr;
  other.vy0 = nullptr;
  other.vz0 = nullptr;
  other.power = nullptr;
  other.power0 = nullptr;
  other.n_rays = 0;
  other.n_rays_capacity = 0;
  return *this;
}

void RayArray2D::release() {
  if (x0 != nullptr) {
    cuda_check(cudaFree(x0), "RayArray2D::release cudaFree x0 failed");
    x0 = nullptr;
  }
  if (y0 != nullptr) {
    cuda_check(cudaFree(y0), "RayArray2D::release cudaFree y0 failed");
    y0 = nullptr;
  }
  if (z0 != nullptr) {
    cuda_check(cudaFree(z0), "RayArray2D::release cudaFree z0 failed");
    z0 = nullptr;
  }
  if (vx0 != nullptr) {
    cuda_check(cudaFree(vx0), "RayArray2D::release cudaFree vx0 failed");
    vx0 = nullptr;
  }
  if (vy0 != nullptr) {
    cuda_check(cudaFree(vy0), "RayArray2D::release cudaFree vy0 failed");
    vy0 = nullptr;
  }
  if (vz0 != nullptr) {
    cuda_check(cudaFree(vz0), "RayArray2D::release cudaFree vz0 failed");
    vz0 = nullptr;
  }
  if (power != nullptr) {
    cuda_check(cudaFree(power), "RayArray2D::release cudaFree power failed");
    power = nullptr;
  }
  if (power0 != nullptr) {
    cuda_check(cudaFree(power0), "RayArray2D::release cudaFree power0 failed");
    power0 = nullptr;
  }
  n_rays = 0;
  n_rays_capacity = 0;
}

bool RayArray2D::empty() const {
  return n_rays == 0;
}

void RayArray2D::allocate(const int n) {
  if (n <= n_rays_capacity) {
    n_rays = std::max(n, 0);
    return;
  }
  release();
  if (n <= 0) {
    return;
  }
  n_rays = n;
  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(double);
  auto alloc_or_cleanup = [&](double*& ptr, const char* message) {
    const cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr), bytes);
    if (err == cudaSuccess) {
      return;
    }
    release();
    TENRYU_ASSERT(false, message);
  };
  alloc_or_cleanup(x0, "RayArray2D::allocate cudaMalloc x0 failed");
  alloc_or_cleanup(y0, "RayArray2D::allocate cudaMalloc y0 failed");
  alloc_or_cleanup(z0, "RayArray2D::allocate cudaMalloc z0 failed");
  alloc_or_cleanup(vx0, "RayArray2D::allocate cudaMalloc vx0 failed");
  alloc_or_cleanup(vy0, "RayArray2D::allocate cudaMalloc vy0 failed");
  alloc_or_cleanup(vz0, "RayArray2D::allocate cudaMalloc vz0 failed");
  alloc_or_cleanup(power, "RayArray2D::allocate cudaMalloc power failed");
  alloc_or_cleanup(power0, "RayArray2D::allocate cudaMalloc power0 failed");
  n_rays_capacity = n;
}

void RayArray2D::copy_from_host(const std::vector<Ray3D>& rays, cudaStream_t stream) {
  allocate(static_cast<int>(rays.size()));
  if (rays.empty()) {
    return;
  }
  std::vector<double> h_x0(rays.size(), 0.0);
  std::vector<double> h_y0(rays.size(), 0.0);
  std::vector<double> h_z0(rays.size(), 0.0);
  std::vector<double> h_vx0(rays.size(), 0.0);
  std::vector<double> h_vy0(rays.size(), 0.0);
  std::vector<double> h_vz0(rays.size(), 0.0);
  std::vector<double> h_P(rays.size(), 0.0);
  std::vector<double> h_P0(rays.size(), 0.0);
  for (std::size_t i = 0; i < rays.size(); ++i) {
    h_x0[i] = rays[i].x;
    h_y0[i] = rays[i].y;
    h_z0[i] = rays[i].z;
    h_vx0[i] = rays[i].vx;
    h_vy0[i] = rays[i].vy;
    h_vz0[i] = rays[i].vz;
    h_P[i] = rays[i].I;
    h_P0[i] = rays[i].I0;
  }

  const std::size_t bytes = rays.size() * sizeof(double);
  cuda_check(cudaMemcpyAsync(x0, h_x0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync x0 failed");
  cuda_check(cudaMemcpyAsync(y0, h_y0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync y0 failed");
  cuda_check(cudaMemcpyAsync(z0, h_z0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync z0 failed");
  cuda_check(cudaMemcpyAsync(vx0, h_vx0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync vx0 failed");
  cuda_check(cudaMemcpyAsync(vy0, h_vy0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync vy0 failed");
  cuda_check(cudaMemcpyAsync(vz0, h_vz0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync vz0 failed");
  cuda_check(cudaMemcpyAsync(power, h_P.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync power failed");
  cuda_check(cudaMemcpyAsync(power0, h_P0.data(), bytes, cudaMemcpyHostToDevice, stream),
             "RayArray2D::copy_from_host memcpyAsync power0 failed");
}

RayArray2D initialize_rays_2d(const Beam& beam,
                              const LaserMesh& lmesh,
                              const int rays_per_beam,
                              const double beam_power,
                              cudaStream_t stream) {
  RayArray2D out;
  int rays_per_axis = rays_per_beam;
  if (rays_per_axis < 2) {
    static bool warned_rays_per_beam_2d_min = false;
    if (!warned_rays_per_beam_2d_min) {
      core::log_warning("Laser 2D ray initialization requires rays_per_beam >= 2; clamping to 2");
      warned_rays_per_beam_2d_min = true;
    }
    rays_per_axis = 2;
  }
  if (!(beam_power > 0.0)) {
    return out;
  }

  // Preserve direction-focus azimuthal phase by applying one common rotation to both.
  const Vec3 d_hat_raw = normalize(Vec3{beam.dir_x, beam.dir_y, beam.dir_z});
  const double d_perp = std::hypot(d_hat_raw.x, d_hat_raw.y);
  double phase_ref = 0.0;
  if (d_perp > 1.0e-14) {
    phase_ref = std::atan2(d_hat_raw.y, d_hat_raw.x);
  } else {
    const double focus_perp = std::hypot(beam.focus_x, beam.focus_y);
    if (focus_perp > 1.0e-14) {
      phase_ref = std::atan2(beam.focus_y, beam.focus_x);
    }
  }
  const double cphi = std::cos(phase_ref);
  const double sphi = std::sin(phase_ref);
  const Vec3 d_hat = normalize(
      Vec3{cphi * d_hat_raw.x + sphi * d_hat_raw.y, -sphi * d_hat_raw.x + cphi * d_hat_raw.y,
           d_hat_raw.z});
  const BeamBasis basis = build_beam_basis(d_hat);
  const Vec3 focus{
      cphi * beam.focus_x + sphi * beam.focus_y,
      -sphi * beam.focus_x + cphi * beam.focus_y,
      beam.focus_lab_z,
  };
  const double R_beam = beam_aperture_radius_cm(beam, lmesh);
  if (!(R_beam > 0.0)) {
    return out;
  }
  double L_source = 2.0 * std::max(beam.f_number, 1.0e-12) * R_beam;
  const double mesh_extent = std::max(lmesh.R_max, std::abs(lmesh.Z_max - lmesh.Z_min));
  L_source = std::max(L_source, 2.0 * mesh_extent);
  Vec3 source = sub(focus, mul(d_hat, L_source));
  for (int iter = 0; iter < 16 && point_inside_lasermesh(source, lmesh); ++iter) {
    L_source *= 2.0;
    source = sub(focus, mul(d_hat, L_source));
  }
  TENRYU_ASSERT(!point_inside_lasermesh(source, lmesh),
                "initialize_rays_2d virtual source must be outside LaserMesh");
  Vec3 entry{};
  const bool ok = intersect_axis_with_lasermesh(source, d_hat, lmesh, entry);
  TENRYU_ASSERT(ok, "initialize_rays_2d failed to find beam entry on LaserMesh boundary");
  TENRYU_ASSERT(dot(d_hat, sub(entry, source)) > 0.0,
                "initialize_rays_2d entry must be forward from virtual source");
  TENRYU_ASSERT(point_on_lasermesh_boundary(entry, lmesh),
                "initialize_rays_2d entry must lie on LaserMesh boundary");

  const double du = 2.0 * R_beam / static_cast<double>(rays_per_axis - 1);
  std::vector<Ray3D> rays;
  std::vector<double> weights;
  rays.reserve(static_cast<std::size_t>(rays_per_axis) * static_cast<std::size_t>(rays_per_axis));
  weights.reserve(rays.capacity());

  double sum_w = 0.0;
  for (int p = 0; p < rays_per_axis; ++p) {
    const double u = -R_beam + static_cast<double>(p) * du;
    for (int q = 0; q < rays_per_axis; ++q) {
      const double w = -R_beam + static_cast<double>(q) * du;
      const double rr = std::sqrt(u * u + w * w);
      if (rr > R_beam) {
        continue;
      }
      Vec3 r0 = add(entry, add(mul(basis.u_hat, u), mul(basis.w_hat, w)));
      const Vec3 v0 = normalize(sub(focus, r0));
      // In oblique 2D_RZ entry, transverse offsets can place rays outside the cylindrical LM domain.
      // Clip such starts to the first boundary hit along the same propagation direction.
      (void)clip_ray_start_if_outside(r0, v0, lmesh);
      Ray3D ray{};
      ray.x = r0.x;
      ray.y = r0.y;
      ray.z = r0.z;
      ray.vx = v0.x;
      ray.vy = v0.y;
      ray.vz = v0.z;
      ray.beam_id = beam.wave_id;
      ray.alive = 1;
      rays.push_back(ray);

      const double wgt = beam.profile(rr) * du * du;
      weights.push_back(wgt);
      sum_w += wgt;
    }
  }

  if (rays.empty()) {
    return out;
  }
  if (!(sum_w > 0.0)) {
    std::fill(weights.begin(), weights.end(), 1.0);
    sum_w = static_cast<double>(weights.size());
  }
  for (std::size_t i = 0; i < rays.size(); ++i) {
    const double I = beam_power * (weights[i] / sum_w);
    rays[i].I = I;
    rays[i].I0 = I;
    rays[i].alive = (I > 0.0) ? 1 : 0;
  }

  std::vector<std::size_t> order(rays.size(), 0);
  std::iota(order.begin(), order.end(), static_cast<std::size_t>(0));
  std::sort(order.begin(), order.end(), [&](const std::size_t a, const std::size_t b) {
    const double Ra = std::sqrt(rays[a].x * rays[a].x + rays[a].y * rays[a].y);
    const double Rb = std::sqrt(rays[b].x * rays[b].x + rays[b].y * rays[b].y);
    return Ra < Rb;
  });
  std::vector<Ray3D> sorted(rays.size());
  for (std::size_t i = 0; i < order.size(); ++i) {
    sorted[i] = rays[order[i]];
  }

  out.copy_from_host(sorted, stream);
  return out;
}

std::vector<BeamGroup> group_beams_by_theta(const std::vector<Beam>& beams,
                                            bool split_delta_lambda) {
  constexpr double kThetaTol = 1.0e-6;
  constexpr double kEqTol = 1.0e-12;
  std::vector<BeamGroup> groups;
  groups.reserve(beams.size());

  auto close = [](const double a, const double b, const double tol) {
    return std::abs(a - b) <= tol;
  };

  for (int b = 0; b < static_cast<int>(beams.size()); ++b) {
    const Beam& beam = beams[static_cast<std::size_t>(b)];
    const double dnorm = std::sqrt(beam.dir_x * beam.dir_x + beam.dir_y * beam.dir_y +
                                   beam.dir_z * beam.dir_z);
    const double cos_theta = (dnorm > 0.0) ? std::clamp(beam.dir_z / dnorm, -1.0, 1.0) : 1.0;
    const double theta = std::acos(cos_theta);

    bool matched = false;
    for (BeamGroup& g : groups) {
      const Beam& rep = beams[static_cast<std::size_t>(g.beam_indices.front())];
      if (std::abs(theta - g.theta) >= kThetaTol) {
        continue;
      }
      const bool same_shape =
          close(beam.f_number, rep.f_number, kEqTol) &&
          close(beam.profile_w0_cm, rep.profile_w0_cm, kEqTol) &&
          (beam.profile_m == rep.profile_m) &&
          (beam.profile_model == rep.profile_model);
      const double beam_focus_R = std::hypot(beam.focus_x, beam.focus_y);
      const double rep_focus_R = std::hypot(rep.focus_x, rep.focus_y);
      // 2D_RZ grouping assumes axisymmetry: beams that differ only by azimuth
      // are treated as equivalent and represented by one traced member.
      const bool same_focus = close(beam_focus_R, rep_focus_R, kEqTol) &&
                              close(beam.focus_lab_z, rep.focus_lab_z, kEqTol);
      if (split_delta_lambda &&
          !close(beam.delta_lambda_nm, rep.delta_lambda_nm, kEqTol)) {
        continue;
      }
      if (!(same_shape && same_focus)) {
        continue;
      }
      g.beam_indices.push_back(b);
      matched = true;
      break;
    }

    if (!matched) {
      BeamGroup g;
      g.theta = theta;
      g.beam_indices.push_back(b);
      g.total_power = 0.0;
      groups.push_back(std::move(g));
    }
  }

  std::sort(groups.begin(), groups.end(), [](const BeamGroup& a, const BeamGroup& b) {
    return a.theta < b.theta;
  });
  return groups;
}

}  // namespace tenryu::laser
