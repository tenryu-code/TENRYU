// This is the ONLY donor-decision point for swept-volume remap (2026-07-26
// review); new consumers must take OrientedSweptVolume or
// SweptVolumeConvention, never a raw bool with a default.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace tenryu::hydro {

enum class SweptVolumeConvention : std::uint8_t {
  LegacyRawV0 = 0,          // dV_a > 0 selects cell_a as donor (historic)
  OrientedLowToHighV1 = 1,  // dV_a > 0 selects cell_b as donor (corrected)
};

struct OrientedSweptVolume {
  double signed_volume_cm3;  // the raw dV_a as passed by the caller
  int cell_a;
  int cell_b;
  int donor;
  int receiver;
};

__host__ __device__ inline OrientedSweptVolume make_oriented_swept_volume(
    const int cell_a,
    const int cell_b,
    const double dV_a,
    const SweptVolumeConvention convention) {
  OrientedSweptVolume out;
  out.signed_volume_cm3 = dV_a;
  out.cell_a = cell_a;
  out.cell_b = cell_b;
  const bool corrected = (convention == SweptVolumeConvention::OrientedLowToHighV1);
  if (corrected) {
    out.donor = (dV_a > 0.0) ? cell_b : cell_a;
  } else {
    out.donor = (dV_a > 0.0) ? cell_a : cell_b;
  }
  out.receiver = (out.donor == cell_a) ? cell_b : cell_a;
  return out;
}

__host__ __device__ inline SweptVolumeConvention swept_volume_convention_from_flag(
    const bool swept_volume_sign_fixed) {
  return swept_volume_sign_fixed ? SweptVolumeConvention::OrientedLowToHighV1
                                 : SweptVolumeConvention::LegacyRawV0;
}

struct SweptVolumeResolvedContract {
  SweptVolumeConvention plain_csr;
  SweptVolumeConvention conservative_csr;  // forced corrected in ale_driver
  SweptVolumeConvention option_b;          // forced corrected
  SweptVolumeConvention axis_band;
  SweptVolumeConvention plic;
};

inline SweptVolumeResolvedContract resolve_swept_volume_contract(
    const bool cfg_swept_volume_sign_fixed) {
  const SweptVolumeConvention base =
      swept_volume_convention_from_flag(cfg_swept_volume_sign_fixed);
  SweptVolumeResolvedContract c;
  c.plain_csr = base;
  c.conservative_csr =
      SweptVolumeConvention::OrientedLowToHighV1;  // ale_driver.cu:7110/10959 force true
  c.option_b = SweptVolumeConvention::OrientedLowToHighV1;  // same forcing site
  c.axis_band = base;
  c.plic = base;  // single-sourced from cfg (stage 2a)
  return c;
}

}  // namespace tenryu::hydro
