#include "mesh/generator_targets.cuh"
#include "mesh/tessellation/voronoi_dual.hpp"
#include "mesh/tessellation/lloyd_skip.hpp"
#include "hydro/rz_corner_mass.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace tenryu::mesh::voronoi {
namespace {

constexpr int kLloydMax = 4;
int g_reale_lloyd_max = kLloydMax;
constexpr double kSweepBlend[] = {1.0, 0.5, 0.25, 0.125, 0.0};

struct OrderedInput {
  Generator generator;
  GeneratorFlowSample flow;
  bool axis = false;
};

struct TargetTimingAccumulators {
  double tessellation_ms = 0.0;
  double legacy_assembly_ms = 0.0;
};

// Representation-spacing scale of a site's coordinates: the same
// delta family the weld criterion uses (ulp spacing at the stored
// magnitudes). A proposal that moves every site by less than its own
// delta is polishing below the coordinate noise floor.
double site_representation_delta(const Generator& site) {
  const auto ulp_gap = [](const double value) {
    const double magnitude = std::fabs(value);
    return std::nextafter(magnitude,
                          std::numeric_limits<double>::infinity()) -
           magnitude;
  };
  return ulp_gap(site.r) + ulp_gap(site.z);
}

std::vector<OrderedInput> order_inputs(
    const std::vector<Generator>& generators,
    const std::vector<GeneratorFlowSample>& flow) {
  std::vector<std::size_t> order(generators.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [&generators](const std::size_t lhs,
                                 const std::size_t rhs) {
                     return generators[lhs].id < generators[rhs].id;
                   });

  std::vector<OrderedInput> ordered;
  ordered.reserve(generators.size());
  for (const std::size_t index : order) {
    ordered.push_back(
        {generators[index], flow[index], generators[index].r == 0.0});
  }
  return ordered;
}

std::vector<Generator> generators_from(
    const std::vector<OrderedInput>& ordered) {
  std::vector<Generator> generators;
  generators.reserve(ordered.size());
  for (const OrderedInput& input : ordered) {
    generators.push_back(input.generator);
  }
  return generators;
}

Tessellation build_rezone_tessellation(
    const std::vector<Generator>& generators,
    const DomainBoundary& domain,
    std::optional<mesh::tess::DelaunayTriangulation>& previous_dt,
    const bool exact_core_enabled,
    TargetTimingAccumulators& timings) {
  if (!exact_core_enabled) {
    const auto tessellation_start = std::chrono::steady_clock::now();
    Tessellation tessellation = build_constrained_voronoi(generators, domain);
    timings.tessellation_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tessellation_start)
            .count();
    return tessellation;
  }
  std::vector<mesh::tess::Site> sites;
  sites.reserve(generators.size());
  for (const Generator& generator : generators) {
    sites.push_back(mesh::tess::Site{generator.r, generator.z, generator.id, 0});
  }
  std::vector<mesh::tess::DomainPoint> domain_points;
  domain_points.reserve(domain.r.size());
  for (std::size_t vertex = 0; vertex < domain.r.size(); ++vertex) {
    domain_points.push_back(
        mesh::tess::DomainPoint{domain.r[vertex], domain.z[vertex]});
  }
  mesh::tess::RestrictedVoronoiResult result;
  if (previous_dt.has_value() &&
      previous_dt->sites.size() == generators.size()) {
    if (const char* const dump_dir = std::getenv("TENRYU_WARM_DUMP_DIR")) {
      static int warm_dump_index = 0;
      ++warm_dump_index;
      char prev_path[512];
      char moved_path[512];
      std::snprintf(prev_path, sizeof(prev_path), "%s/warm_%04d_prev.txt",
                    dump_dir, warm_dump_index);
      std::snprintf(moved_path, sizeof(moved_path), "%s/warm_%04d_moved.txt",
                    dump_dir, warm_dump_index);
      std::FILE* const prev_file = std::fopen(prev_path, "w");
      TENRYU_ASSERT(prev_file != nullptr, "open warm dump prev file");
      for (const mesh::tess::Site& site : previous_dt->sites) {
        std::fprintf(prev_file, "%llu %.17g %.17g\n",
                     static_cast<unsigned long long>(site.stable_id),
                     site.r, site.z);
      }
      TENRYU_ASSERT(std::fclose(prev_file) == 0, "close warm dump prev file");
      std::FILE* const moved_file = std::fopen(moved_path, "w");
      TENRYU_ASSERT(moved_file != nullptr, "open warm dump moved file");
      for (const mesh::tess::Site& site : sites) {
        std::fprintf(moved_file, "%llu %.17g %.17g\n",
                     static_cast<unsigned long long>(site.stable_id),
                     site.r, site.z);
      }
      TENRYU_ASSERT(std::fclose(moved_file) == 0, "close warm dump moved file");
      std::fprintf(stderr, "[warm_dump] attempt=%d prev=%s moved=%s\n",
                   warm_dump_index, prev_path, moved_path);
    }
    const auto tessellation_start = std::chrono::steady_clock::now();
    result = mesh::tess::build_restricted_voronoi_warm(
        *previous_dt, std::move(sites), std::move(domain_points));
    timings.tessellation_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tessellation_start)
            .count();
  } else {
    const auto tessellation_start = std::chrono::steady_clock::now();
    result = mesh::tess::build_restricted_voronoi(
        std::move(sites), std::move(domain_points));
    timings.tessellation_ms +=
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tessellation_start)
            .count();
  }
  if (result.used_static_fallback) {
    std::fprintf(stderr, "[warm_targets] fallback code=%s\n",
                 result.failure_code.c_str());
  }
  if (!result.ok) {
    Tessellation failed;
    failed.valid = false;
    failed.reject_reason = result.failure_code;
    return failed;
  }
  const auto legacy_assembly_start = std::chrono::steady_clock::now();
  Tessellation tessellation =
      mesh::tess::to_legacy_tessellation(result.dt, result.dcel);
  tessellation.weld_contracted = result.weld.contracted_edges;
  tessellation.weld_clusters = result.weld.clusters;
  tessellation.weld_max_displacement = result.weld.max_displacement;
  tessellation.carrier_representability_ok =
      result.carrier_representability_ok;
  tessellation.carrier_coverage_ok = result.carrier_coverage_ok;
  previous_dt = std::move(result.dt);
  timings.legacy_assembly_ms +=
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - legacy_assembly_start)
          .count();
  return tessellation;
}

Generator planar_centroid_target(const Generator& generator,
                                 const VoronoiCell& cell,
                                 const bool axis) {
  double twice_area = 0.0;
  double r_numerator = 0.0;
  double z_numerator = 0.0;
  for (std::size_t vertex = 0; vertex < cell.r.size(); ++vertex) {
    const std::size_t next = (vertex + 1) % cell.r.size();
    const double cross = cell.r[vertex] * cell.z[next] -
                         cell.r[next] * cell.z[vertex];
    twice_area += cross;
    r_numerator += (cell.r[vertex] + cell.r[next]) * cross;
    z_numerator += (cell.z[vertex] + cell.z[next]) * cross;
  }

  Generator target = generator;
  if (twice_area == 0.0) {
    return target;
  }
  target.r = axis ? 0.0 : r_numerator / (3.0 * twice_area);
  target.z = z_numerator / (3.0 * twice_area);
  return target;
}

bool accept_sweep_blend(std::vector<Generator>& proposed,
                        const std::vector<Generator>& current,
                        const std::vector<OrderedInput>& ordered,
                        const DomainBoundary& domain,
                        std::optional<mesh::tess::DelaunayTriangulation>&
                            previous_dt,
                        const bool exact_core_enabled,
                        Tessellation& accepted_tessellation,
                        double& accepted_blend,
                        TargetTimingAccumulators& timings) {
  const std::vector<Generator> full_proposal = proposed;
  for (const double blend : kSweepBlend) {
    if (blend == 0.0) {
      proposed = current;
    } else {
      for (std::size_t index = 0; index < proposed.size(); ++index) {
        proposed[index] = current[index];
        proposed[index].r = ordered[index].axis
            ? 0.0
            : current[index].r +
                  blend * (full_proposal[index].r - current[index].r);
        proposed[index].z = current[index].z +
                            blend * (full_proposal[index].z -
                                     current[index].z);
      }
    }

    Tessellation trial =
        build_rezone_tessellation(
            proposed, domain, previous_dt, exact_core_enabled, timings);
    bool certified = trial.valid;
    if (certified) {
      for (const VoronoiCell& cell : trial.cells) {
        const int nverts = static_cast<int>(cell.r.size());
        if (!hydro::rz::rz_polygon_volume_certified_positive(
                cell.r.data(), cell.z.data(), nverts) ||
            !hydro::rz::rz_polygon_area2_certified_positive(
                cell.r.data(), cell.z.data(), nverts)) {
          certified = false;
          break;
        }
      }
    }
    if (certified) {
      accepted_tessellation = std::move(trial);
      accepted_blend = blend;
      return true;
    }
  }
  return false;
}

void apply_stage2_priority_three_proxy(std::vector<Generator>& targets) {
  // Stage 2 has no priority-3 proxy yet. This explicit no-op is its hook.
  (void)targets;
}

TargetResult reject(const std::string& reason) {
  TargetResult result;
  result.reject_reason = reason;
  return result;
}

}  // namespace

void set_reale_lloyd_max(const int lloyd_max) {
  TENRYU_ASSERT(lloyd_max >= 0, "reale_lloyd_max must be non-negative");
  g_reale_lloyd_max = lloyd_max;
}

TargetResult build_generator_targets(
    const std::vector<Generator>& generators,
    const std::vector<GeneratorFlowSample>& flow,
    const DomainBoundary& domain,
    const double dt,
    const bool exact_core_enabled,
    std::optional<mesh::tess::DelaunayTriangulation>* const warm_dt) {
  const auto target_generation_start = std::chrono::steady_clock::now();
  TargetTimingAccumulators timings;
  std::optional<mesh::tess::DelaunayTriangulation> local_previous_dt;
  std::optional<mesh::tess::DelaunayTriangulation>& previous_dt =
      warm_dt != nullptr ? *warm_dt : local_previous_dt;
  if (flow.size() != generators.size()) {
    return reject("flow_size_mismatch");
  }

  const std::vector<OrderedInput> ordered = order_inputs(generators, flow);
  const std::vector<Generator> initial = generators_from(ordered);
  Tessellation tessellation =
      build_rezone_tessellation(
          initial, domain, previous_dt, exact_core_enabled, timings);
  if (!tessellation.valid) {
    return reject(tessellation.reject_reason);
  }

  TargetResult result;
  std::vector<Generator> targets = initial;
  for (std::size_t index = 0; index < targets.size(); ++index) {
    if (ordered[index].axis) {
      targets[index].r = 0.0;
    } else {
      targets[index].r += dt * ordered[index].flow.v_r;
    }
    targets[index].z += dt * ordered[index].flow.v_z;
  }
  double accepted_blend = 0.0;
  if (!accept_sweep_blend(targets, initial, ordered, domain, previous_dt,
                          exact_core_enabled, tessellation, accepted_blend,
                          timings)) {
    // Full rollback is the valid initial set, so this branch is unreachable
    // unless the admissibility transaction itself violates its invariant.
    return reject("advection_invalid");
  }
  result.sweep_blend.push_back(accepted_blend);

  apply_stage2_priority_three_proxy(targets);

  const bool lloyd_d1_disabled =
      std::getenv("TENRYU_LLOYD_D1_DISABLE") != nullptr;
  for (int iteration = 0; iteration < g_reale_lloyd_max; ++iteration) {
    const std::vector<Generator> before_sweep = targets;
    std::vector<Generator> proposed = before_sweep;
    for (std::size_t index = 0; index < proposed.size(); ++index) {
      proposed[index] = planar_centroid_target(
          before_sweep[index], tessellation.cells[index],
          ordered[index].axis);
    }
    // d1 certified skip (SS8-T2d-d1, docs/design/
    // t2_d1_certified_lloyd_skip_20260815.md): when no predicate of the
    // warm DT can flip under the proposed displacements, the proposal's
    // Delaunay topology provably equals the current one — the iteration
    // is structurally a no-op. Break WITHOUT adopting, exactly like the
    // d2 exit below; d2 stays as the floor when d1 refuses (degenerate
    // slack, mismatched warm base, disabled switch).
    if (exact_core_enabled && !lloyd_d1_disabled && previous_dt.has_value()) {
      const mesh::tess::DelaunayTriangulation& warm = *previous_dt;
      bool base_matches = warm.sites.size() == proposed.size();
      if (base_matches) {
        for (std::size_t index = 0; index < proposed.size(); ++index) {
          if (warm.sites[index].stable_id != ordered[index].generator.id ||
              warm.sites[index].r != before_sweep[index].r ||
              warm.sites[index].z != before_sweep[index].z) {
            base_matches = false;
            break;
          }
        }
      }
      if (base_matches) {
        std::vector<double> delta_r(proposed.size());
        std::vector<double> delta_z(proposed.size());
        for (std::size_t index = 0; index < proposed.size(); ++index) {
          delta_r[index] =
              std::fabs(proposed[index].r - warm.sites[index].r);
          delta_z[index] =
              std::fabs(proposed[index].z - warm.sites[index].z);
        }
        const mesh::tess::LloydSkipReport report =
            mesh::tess::certify_lloyd_noop(warm, delta_r, delta_z);
        if (report.certified) {
          result.lloyd_d1_fired = true;
          result.lloyd_d1_iteration = iteration;
          std::fprintf(stderr, "[lloyd_d1] iter=%d certs=%zu skipped=1\n",
                       iteration, report.certificates_checked);
          break;
        }
      }
    }
    // d2 convergence exit (user-approved 2026-08-15): if every proposed
    // move is below the site's own representation-spacing scale, further
    // Lloyd iterations polish below the coordinate noise floor — stop
    // WITHOUT paying the proposal's tessellation. Derived floor, no
    // tunables; kLloydMax stays the iteration ceiling.
    bool converged = true;
    for (std::size_t index = 0; index < proposed.size(); ++index) {
      const double dr = proposed[index].r - before_sweep[index].r;
      const double dz = proposed[index].z - before_sweep[index].z;
      if (std::fabs(dr) + std::fabs(dz) >
          site_representation_delta(before_sweep[index])) {
        converged = false;
        break;
      }
    }
    if (converged) {
      break;
    }
    if (!accept_sweep_blend(proposed, before_sweep, ordered, domain,
                            previous_dt, exact_core_enabled, tessellation,
                            accepted_blend, timings)) {
      return reject("advection_invalid");
    }
    targets = std::move(proposed);
    result.sweep_blend.push_back(accepted_blend);
    ++result.cvt_iterations_used;
  }

  result.targets = std::move(targets);
  result.accepted_tessellation = std::move(tessellation);
  result.tessellation_ms = timings.tessellation_ms;
  result.legacy_assembly_ms = timings.legacy_assembly_ms;
  result.target_generation_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - target_generation_start)
          .count() -
      result.tessellation_ms - result.legacy_assembly_ms;
  result.valid = true;
  return result;
}

}  // namespace tenryu::mesh::voronoi
