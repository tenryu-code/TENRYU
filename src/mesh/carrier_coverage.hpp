#pragma once

// Exact boundary-coverage gate per consult-29 response §8 and the C1/C2
// contract. ExactRational is a C2-local type — C1-w2 introduces the canonical
// exact lambda in the tessellation core and the parent wave unifies the two.

#include <vector>

#include "mesh/tessellation/expansion.hpp"

namespace tenryu::mesh {

// Exact rational lambda on a carrier edge: num/den as Shewchuk expansions with the
// denominator sign-normalized positive at construction.
struct ExactRational {
  tess::Expansion num;
  tess::Expansion den;
};

// Normalizes sign(den) > 0 (negating both parts if needed). Throws
// std::invalid_argument on a zero denominator or empty expansions.
ExactRational make_exact_rational(tess::Expansion num, tess::Expansion den);

// Convenience: exact value num/den of two finite doubles (den != 0). Throws
// std::invalid_argument on non-finite input or zero den.
ExactRational exact_rational_ratio(double num, double den);

// Three-way exact comparison (delegates to tess::compare_event_parameters_exact).
int compare_exact_rational(const ExactRational& a, const ExactRational& b);

struct CarrierEdgeInterval {
  int edge = -1;        // carrier edge id in [0, n_edges)
  ExactRational lo;     // lambda^-
  ExactRational hi;     // lambda^+
};

enum class CoverageViolation {
  kNone = 0,
  kInvalidInput,        // n_edges <= 0
  kInvalidRational,     // a lo/hi rational violates the normalized-denominator contract
  kEdgeOutOfRange,      // an interval names an edge outside [0, n_edges)
  kEmptyEdge,           // an edge has no intervals
  kDegenerate,          // !(lo < hi)
  kFirstNotZero,        // sorted first interval lo != 0
  kGap,                 // previous hi < next lo
  kOverlap,             // previous hi > next lo
  kLastNotOne,          // sorted last interval hi != 1
};

struct CoverageResult {
  bool ok = false;
  CoverageViolation violation = CoverageViolation::kNone;
  int edge = -1;            // first offending edge (kInvalidInput: -1;
                            //  kInvalidRational: the interval's edge value,
                            //  unvalidated, as given;
                            //  kEdgeOutOfRange: the offending edge VALUE)
  int interval_index = -1;  // kInvalidRational and kEdgeOutOfRange: index in the
                            //  INPUT vector;
                            //  kEmptyEdge: -1; otherwise index in the edge's SORTED list
  double lambda_approx = 0.0;  // diagnostic double approximation of the offending
                               // endpoint (0.0 where not applicable)
};

// Exact subdivision check (consult-29 §8): for every carrier edge e in [0, n_edges),
// the intervals on e, sorted by exact (lo, hi) lexicographic comparison with
// std::stable_sort, must satisfy lo_0 == 0, hi_k == lo_{k+1} for every adjacent
// pair, hi_last == 1, and lo_k < hi_k for every k. Deterministic first-violation
// report. A validation pre-pass over the input in input order (lo checked before hi
// per interval) rejects any rational with an empty num/den expansion or non-positive
// denominator sign before any exact comparison runs; this pre-pass precedes the
// edge-range pass. Scan order and per-position check order are part of the contract:
// edges ascending; within an edge, positions k ascending; at each k the degeneracy
// check precedes the k == 0 first-endpoint check, which precedes the adjacency check
// against k-1; the last-endpoint check runs after the walk.
CoverageResult check_carrier_coverage(
    int n_edges, const std::vector<CarrierEdgeInterval>& intervals);

}  // namespace tenryu::mesh
