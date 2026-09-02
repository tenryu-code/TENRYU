#pragma once

#include <cstdint>

#include "mesh/tessellation/expansion.hpp"

namespace tenryu::mesh::tess {

struct Site {
  double r;
  double z;
  std::uint64_t stable_id;
  std::uint32_t rank;
};

struct HomogeneousPoint {
  Expansion x;
  Expansion y;
  Expansion w;
};

// Exact rational with Expansion numerator/denominator; denominator
// sign-normalized strictly positive. Comparisons cross-multiply exactly.
struct ExactRational {
  Expansion num;
  Expansion den;
};

struct SegmentHit {
  bool exists;
  HomogeneousPoint point;
};

ExactRational make_exact_rational(Expansion num, Expansion den);

int exact_rational_compare(const ExactRational& a,
                           const ExactRational& b);

int exact_rational_compare_unit(const ExactRational& a, int bound01);

double exact_rational_round(const ExactRational& a);

HomogeneousPoint circumcenter_exact(const Site& a,
                                    const Site& b,
                                    const Site& c);

SegmentHit bisector_segment_intersection_exact(
    const Site& i,
    const Site& j,
    const Site& segment_a,
    const Site& segment_b);

bool on_segment_exact(const Site& a, const Site& b, const Site& p);

int projection_order(const Site& a,
                     const Site& b,
                     const Site& p,
                     const Site& q);

int compare_event_parameters_exact(const Expansion& numerator_a,
                                   const Expansion& denominator_a,
                                   const Expansion& numerator_b,
                                   const Expansion& denominator_b);

bool event_parameter_less_exact(const Expansion& numerator_a,
                                const Expansion& denominator_a,
                                const Expansion& numerator_b,
                                const Expansion& denominator_b);

bool event_parameters_equal_exact(const Expansion& numerator_a,
                                  const Expansion& denominator_a,
                                  const Expansion& numerator_b,
                                  const Expansion& denominator_b);

}  // namespace tenryu::mesh::tess
