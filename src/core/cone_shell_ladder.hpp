#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace tenryu::core {

struct ConeShellAlongWallSpec {
  double wall_length = 0.0;
  double wall_thickness = 0.0;
  int normal_cells = 0;
  double normal_growth = 0.0;
  double tip_size_factor = 0.0;
  double base_size_factor = 0.0;
  double tip_hold = 0.0;
  double grading_length = 0.0;
  double adjacent_ratio_max = 0.0;
  double tip_rotation_length = 0.0;
  double base_rotation_length = 0.0;
  bool planar_base = false;
  std::vector<double> pinned_landmarks;
};

struct ConeShellPoint {
  double r = 0.0;
  double z = 0.0;
};

enum class ConeShellBoxFace : int {
  RMax = 0,
  ZMin = 1,
  ZMax = 2,
};

struct ConeShellBoxHit {
  double distance = std::numeric_limits<double>::infinity();
  ConeShellBoxFace face = ConeShellBoxFace::RMax;
  bool valid = false;
};

struct ConeShellExteriorRaySpec {
  double wall_length = 0.0;
  double wall_thickness = 0.0;
  double alpha = 0.0;
  double mid_tip_radius = 0.0;
  double tip_z = 0.0;
  int axis_sign = 1;
  double tip_rotation_length = 0.0;
  double base_rotation_length = 0.0;
  bool planar_base = false;
  double outer_strip_depth = 0.0;
  double box_r_max = 0.0;
  double box_z_min = 0.0;
  double box_z_max = 0.0;
};

struct ConeShellExteriorSwitches {
  std::vector<double> q;
  bool valid = false;
};

struct ConeShellAlongWallLadder {
  std::vector<double> q;
  std::vector<int> segment_cells;
  double normal_face_width = 0.0;
  double max_adjacent_ratio = std::numeric_limits<double>::infinity();
  bool counts_valid = false;
  bool adjacent_ratio_valid = false;
};

inline double cone_shell_q5(const double u) {
  const double u2 = u * u;
  const double u3 = u2 * u;
  return 10.0 * u3 - 15.0 * u3 * u + 6.0 * u3 * u2;
}

inline double cone_shell_tip_rotation(const double q,
                                      const double rotation_length) {
  if (q >= rotation_length) {
    return 0.0;
  }
  return 1.0 - cone_shell_q5(q / rotation_length);
}

inline double cone_shell_base_rotation(const double q,
                                       const double wall_length,
                                       const double rotation_length) {
  if (q <= wall_length - rotation_length) {
    return 0.0;
  }
  return cone_shell_q5(
      (q - (wall_length - rotation_length)) / rotation_length);
}

inline double cone_shell_rotation(const double q,
                                  const ConeShellExteriorRaySpec& spec) {
  const double tip =
      cone_shell_tip_rotation(q, spec.tip_rotation_length);
  if (!spec.planar_base) {
    return tip;
  }
  return tip + cone_shell_base_rotation(
                   q, spec.wall_length, spec.base_rotation_length);
}

inline ConeShellPoint cone_shell_through_direction(
    const double q, const ConeShellExteriorRaySpec& spec) {
  const double sin_alpha = std::sin(spec.alpha);
  const double cos_alpha = std::cos(spec.alpha);
  const double sigma = static_cast<double>(spec.axis_sign);
  const ConeShellPoint tau{sin_alpha, sigma * cos_alpha};
  const ConeShellPoint nu{cos_alpha, -sigma * sin_alpha};
  const double rotation = cone_shell_rotation(q, spec);
  if (rotation == 0.0) {
    return nu;
  }
  const double tan_alpha = std::tan(spec.alpha);
  const ConeShellPoint raw{
      nu.r + tan_alpha * rotation * tau.r,
      nu.z + tan_alpha * rotation * tau.z,
  };
  const double norm = std::hypot(raw.r, raw.z);
  return {raw.r / norm, raw.z / norm};
}

inline ConeShellPoint cone_shell_wall_point(
    const double q, const double n, const ConeShellExteriorRaySpec& spec) {
  const double sin_alpha = std::sin(spec.alpha);
  const double cos_alpha = std::cos(spec.alpha);
  const double sigma = static_cast<double>(spec.axis_sign);
  const ConeShellPoint tau{sin_alpha, sigma * cos_alpha};
  const ConeShellPoint nu{cos_alpha, -sigma * sin_alpha};
  const double s = q + n * std::tan(spec.alpha) *
                           cone_shell_rotation(q, spec);
  return {
      spec.mid_tip_radius + s * tau.r + n * nu.r,
      spec.tip_z + s * tau.z + n * nu.z,
  };
}

inline ConeShellPoint cone_shell_outer_far_point(
    const double q, const ConeShellExteriorRaySpec& spec) {
  const ConeShellPoint wall =
      cone_shell_wall_point(q, 0.5 * spec.wall_thickness, spec);
  const ConeShellPoint direction = cone_shell_through_direction(q, spec);
  return {
      wall.r + spec.outer_strip_depth * direction.r,
      wall.z + spec.outer_strip_depth * direction.z,
  };
}

inline ConeShellBoxHit cone_shell_box_hit(
    const double q, const ConeShellExteriorRaySpec& spec) {
  const ConeShellPoint point = cone_shell_outer_far_point(q, spec);
  const ConeShellPoint direction = cone_shell_through_direction(q, spec);
  ConeShellBoxHit hit;
  const double radial_distance =
      (spec.box_r_max - point.r) / direction.r;
  if (std::isfinite(radial_distance) && radial_distance > 0.0) {
    hit.distance = radial_distance;
    hit.face = ConeShellBoxFace::RMax;
    hit.valid = true;
  }
  if (direction.z != 0.0) {
    const double z_min_distance =
        (spec.box_z_min - point.z) / direction.z;
    if (std::isfinite(z_min_distance) && z_min_distance > 0.0 &&
        (!hit.valid || z_min_distance < hit.distance)) {
      hit.distance = z_min_distance;
      hit.face = ConeShellBoxFace::ZMin;
      hit.valid = true;
    }
    const double z_max_distance =
        (spec.box_z_max - point.z) / direction.z;
    if (std::isfinite(z_max_distance) && z_max_distance > 0.0 &&
        (!hit.valid || z_max_distance < hit.distance)) {
      hit.distance = z_max_distance;
      hit.face = ConeShellBoxFace::ZMax;
      hit.valid = true;
    }
  }
  return hit;
}

inline ConeShellExteriorSwitches find_cone_shell_exterior_switches(
    const ConeShellExteriorRaySpec& spec) {
  ConeShellExteriorSwitches result;
  constexpr int kScanIntervals = 64;
  std::array<ConeShellBoxHit, kScanIntervals + 1> hit{};
  for (int scan = 0; scan <= kScanIntervals; ++scan) {
    const double q = spec.wall_length * static_cast<double>(scan) /
                     static_cast<double>(kScanIntervals);
    hit[static_cast<std::size_t>(scan)] = cone_shell_box_hit(q, spec);
    if (!hit[static_cast<std::size_t>(scan)].valid) {
      return result;
    }
  }

  const auto crossing_value = [&spec](const double q,
                                      const double corner_z) {
    const ConeShellPoint point = cone_shell_outer_far_point(q, spec);
    const ConeShellPoint direction = cone_shell_through_direction(q, spec);
    const double corner_r_delta = spec.box_r_max - point.r;
    const double corner_z_delta = corner_z - point.z;
    return corner_r_delta * direction.z -
           corner_z_delta * direction.r;
  };

  for (int scan = 0; scan < kScanIntervals; ++scan) {
    const ConeShellBoxFace left_face =
        hit[static_cast<std::size_t>(scan)].face;
    const ConeShellBoxFace right_face =
        hit[static_cast<std::size_t>(scan + 1)].face;
    if (left_face == right_face) {
      continue;
    }
    const bool left_radial = left_face == ConeShellBoxFace::RMax;
    const bool right_radial = right_face == ConeShellBoxFace::RMax;
    if (left_radial == right_radial) {
      return result;
    }
    const ConeShellBoxFace z_face = left_radial ? right_face : left_face;
    const double corner_z = z_face == ConeShellBoxFace::ZMin
                                ? spec.box_z_min
                                : spec.box_z_max;
    double lo = spec.wall_length * static_cast<double>(scan) /
                static_cast<double>(kScanIntervals);
    double hi = spec.wall_length * static_cast<double>(scan + 1) /
                static_cast<double>(kScanIntervals);
    double f_lo = crossing_value(lo, corner_z);
    double f_hi = crossing_value(hi, corner_z);
    if (!(std::isfinite(f_lo) && std::isfinite(f_hi)) ||
        (f_lo != 0.0 && f_hi != 0.0 &&
         std::signbit(f_lo) == std::signbit(f_hi))) {
      return result;
    }
    if (f_lo == 0.0) {
      hi = lo;
    } else if (f_hi != 0.0) {
      for (int iteration = 0; iteration < 80; ++iteration) {
        const double mid = 0.5 * (lo + hi);
        const double f_mid = crossing_value(mid, corner_z);
        if (!std::isfinite(f_mid)) {
          return result;
        }
        if (f_mid == 0.0) {
          lo = mid;
          hi = mid;
          break;
        }
        if (std::signbit(f_mid) == std::signbit(f_lo)) {
          lo = mid;
          f_lo = f_mid;
        } else {
          hi = mid;
          f_hi = f_mid;
        }
      }
    }
    result.q.push_back(0.5 * (lo + hi));
  }
  std::sort(result.q.begin(), result.q.end());
  result.q.erase(std::unique(result.q.begin(), result.q.end()),
                 result.q.end());
  result.valid = true;
  return result;
}

inline double cone_shell_normal_face_width(const double wall_thickness,
                                           const int normal_cells,
                                           const double normal_growth) {
  const int half_cells = normal_cells / 2;
  return 0.5 * wall_thickness * (normal_growth - 1.0) /
         (std::pow(normal_growth, static_cast<double>(half_cells)) - 1.0);
}

inline std::vector<double> build_cone_shell_normal_ladder(
    const double wall_thickness,
    const int normal_cells,
    const double normal_growth) {
  const int half_cells = normal_cells / 2;
  const double half_thickness = 0.5 * wall_thickness;
  const double h0 = cone_shell_normal_face_width(
      wall_thickness, normal_cells, normal_growth);
  std::vector<double> nodes(static_cast<std::size_t>(normal_cells + 1), 0.0);
  nodes[0] = -half_thickness;
  double width = h0;
  for (int j = 0; j + 1 < half_cells; ++j) {
    nodes[static_cast<std::size_t>(j + 1)] =
        nodes[static_cast<std::size_t>(j)] + width;
    width *= normal_growth;
  }
  nodes[static_cast<std::size_t>(half_cells)] = 0.0;
  for (int j = 0; j < half_cells; ++j) {
    nodes[static_cast<std::size_t>(normal_cells - j)] =
        -nodes[static_cast<std::size_t>(j)];
  }
  return nodes;
}

namespace cone_shell_detail {

inline double inverse_along_wall_size(const double q,
                                      const double tip_hold,
                                      const double grading_length,
                                      const double h_tip,
                                      const double h_base) {
  if (q <= tip_hold) {
    return 1.0 / h_tip;
  }
  if (q < tip_hold + grading_length) {
    const double u = (q - tip_hold) / grading_length;
    const double log_ratio = std::log(h_base / h_tip);
    return std::exp(-log_ratio * cone_shell_q5(u)) / h_tip;
  }
  return 1.0 / h_base;
}

inline double integrate_inverse_along_wall_size(
    const double begin,
    const double end,
    const double tip_hold,
    const double grading_length,
    const double h_tip,
    const double h_base) {
  if (!(end > begin)) {
    return 0.0;
  }

  double total = 0.0;
  const double hold_end = std::min(end, tip_hold);
  if (hold_end > begin) {
    total += (hold_end - begin) / h_tip;
  }

  const double ramp_begin = std::max(begin, tip_hold);
  const double ramp_end = std::min(end, tip_hold + grading_length);
  if (ramp_end > ramp_begin) {
    constexpr int intervals = 64;
    const double step = (ramp_end - ramp_begin) /
                        static_cast<double>(intervals);
    double sum = inverse_along_wall_size(
                     ramp_begin, tip_hold, grading_length, h_tip, h_base) +
                 inverse_along_wall_size(
                     ramp_end, tip_hold, grading_length, h_tip, h_base);
    for (int k = 1; k < intervals; ++k) {
      const double q = ramp_begin + static_cast<double>(k) * step;
      const double weight = (k % 2 == 0) ? 2.0 : 4.0;
      sum += weight * inverse_along_wall_size(
                          q, tip_hold, grading_length, h_tip, h_base);
    }
    total += step * sum / 3.0;
  }

  const double base_begin = std::max(begin, tip_hold + grading_length);
  if (end > base_begin) {
    total += (end - base_begin) / h_base;
  }
  return total;
}

inline bool build_segment_nodes(const std::vector<double>& landmarks,
                                const std::vector<double>& segment_measure,
                                const std::vector<int>& segment_cells,
                                const double tip_hold,
                                const double grading_length,
                                const double h_tip,
                                const double h_base,
                                std::vector<double>* q,
                                std::vector<int>* cell_segment) {
  long long total_cells = 0;
  for (const int count : segment_cells) {
    if (count < 2) {
      return false;
    }
    total_cells += static_cast<long long>(count);
  }
  if (total_cells <= 0 ||
      total_cells > static_cast<long long>(std::numeric_limits<int>::max())) {
    return false;
  }

  q->clear();
  cell_segment->clear();
  q->reserve(static_cast<std::size_t>(total_cells + 1));
  cell_segment->reserve(static_cast<std::size_t>(total_cells));
  q->push_back(landmarks.front());
  for (std::size_t segment = 0; segment < segment_cells.size(); ++segment) {
    const double q_begin = landmarks[segment];
    const double q_end = landmarks[segment + 1U];
    const int count = segment_cells[segment];
    const double measure = segment_measure[segment];
    for (int local = 1; local < count; ++local) {
      const double target =
          (static_cast<double>(local) / static_cast<double>(count)) * measure;
      double lo = q_begin;
      double hi = q_end;
      for (int iteration = 0; iteration < 80; ++iteration) {
        const double mid = 0.5 * (lo + hi);
        const double value = integrate_inverse_along_wall_size(
            q_begin, mid, tip_hold, grading_length, h_tip, h_base);
        if (value < target) {
          lo = mid;
        } else {
          hi = mid;
        }
      }
      q->push_back(0.5 * (lo + hi));
    }
    q->push_back(q_end);
    for (int local = 0; local < count; ++local) {
      cell_segment->push_back(static_cast<int>(segment));
    }
  }
  return true;
}

inline double adjacent_ratio(const std::vector<double>& q,
                             std::size_t* first_violation,
                             const double ratio_limit) {
  double max_ratio = 1.0;
  *first_violation = q.size();
  for (std::size_t i = 0; i + 2U < q.size(); ++i) {
    const double left = q[i + 1U] - q[i];
    const double right = q[i + 2U] - q[i + 1U];
    if (!(left > 0.0 && right > 0.0)) {
      return std::numeric_limits<double>::infinity();
    }
    const double ratio = std::max(left, right) / std::min(left, right);
    max_ratio = std::max(max_ratio, ratio);
    if (*first_violation == q.size() && ratio > ratio_limit) {
      *first_violation = i;
    }
  }
  return max_ratio;
}

}  // namespace cone_shell_detail

// Normalized along-wall MONITOR measure u_monitor(q): the per-segment
// equidistributed station-count fraction (NOT the geometric fraction q/L_w,
// and NOT the station index fraction j/N_q — those coincide only segment-wise
// up to the l_ratio_max repair band). See docs/design/cone_assembly_smooth_joins_20260720.md Stage 0.
inline double cone_shell_along_wall_monitor_measure(
    const double q, const ConeShellAlongWallSpec& spec) {
  if (q <= 0.0) {
    return 0.0;
  }
  if (q >= spec.wall_length) {
    return 1.0;
  }
  const double normal_face_width = cone_shell_normal_face_width(
      spec.wall_thickness, spec.normal_cells, spec.normal_growth);
  const double h_tip = spec.tip_size_factor * normal_face_width;
  const double h_base = spec.base_size_factor * normal_face_width;
  const double total = cone_shell_detail::integrate_inverse_along_wall_size(
      0.0, spec.wall_length, spec.tip_hold, spec.grading_length, h_tip,
      h_base);
  const double partial = cone_shell_detail::integrate_inverse_along_wall_size(
      0.0, q, spec.tip_hold, spec.grading_length, h_tip, h_base);
  return partial / total;
}

inline ConeShellAlongWallLadder build_cone_shell_along_wall_ladder(
    const ConeShellAlongWallSpec& spec) {
  ConeShellAlongWallLadder result;
  result.normal_face_width = cone_shell_normal_face_width(
      spec.wall_thickness, spec.normal_cells, spec.normal_growth);
  const double h_tip = spec.tip_size_factor * result.normal_face_width;
  const double h_base = spec.base_size_factor * result.normal_face_width;

  std::vector<double> landmarks{
      0.0,
      std::clamp(spec.tip_rotation_length, 0.0, spec.wall_length),
      std::clamp(spec.tip_hold, 0.0, spec.wall_length),
      std::clamp(spec.tip_hold + spec.grading_length, 0.0,
                 spec.wall_length),
      spec.wall_length,
  };
  if (spec.planar_base) {
    landmarks.push_back(std::clamp(
        spec.wall_length - spec.base_rotation_length, 0.0,
        spec.wall_length));
  }
  landmarks.insert(landmarks.end(), spec.pinned_landmarks.begin(),
                   spec.pinned_landmarks.end());
  std::sort(landmarks.begin(), landmarks.end());
  landmarks.erase(std::unique(landmarks.begin(), landmarks.end()),
                  landmarks.end());
  if (landmarks.size() < 2U) {
    return result;
  }

  std::vector<double> segment_measure(landmarks.size() - 1U, 0.0);
  result.segment_cells.resize(landmarks.size() - 1U, 0);
  for (std::size_t segment = 0; segment + 1U < landmarks.size(); ++segment) {
    const double measure =
        cone_shell_detail::integrate_inverse_along_wall_size(
            landmarks[segment], landmarks[segment + 1U], spec.tip_hold,
            spec.grading_length, h_tip, h_base);
    segment_measure[segment] = measure;
    if (!(std::isfinite(measure) && measure >= 0.0 &&
          measure < static_cast<double>(
                        std::numeric_limits<long long>::max()))) {
      return result;
    }
    const long long rounded = std::llround(measure);
    const long long count = std::max(2LL, rounded);
    if (count > static_cast<long long>(std::numeric_limits<int>::max())) {
      return result;
    }
    result.segment_cells[segment] = static_cast<int>(count);
  }

  std::vector<int> cell_segment;
  for (int increment = 0; increment <= 8; ++increment) {
    if (!cone_shell_detail::build_segment_nodes(
            landmarks, segment_measure, result.segment_cells, spec.tip_hold,
            spec.grading_length, h_tip, h_base, &result.q, &cell_segment)) {
      return result;
    }
    result.counts_valid = true;
    std::size_t first_violation = result.q.size();
    result.max_adjacent_ratio = cone_shell_detail::adjacent_ratio(
        result.q, &first_violation, spec.adjacent_ratio_max);
    if (first_violation == result.q.size()) {
      result.adjacent_ratio_valid = true;
      break;
    }
    if (increment == 8) {
      break;
    }
    const double left = result.q[first_violation + 1U] -
                        result.q[first_violation];
    const double right = result.q[first_violation + 2U] -
                         result.q[first_violation + 1U];
    const std::size_t larger_cell =
        (right > left) ? first_violation + 1U : first_violation;
    const int segment = cell_segment[larger_cell];
    ++result.segment_cells[static_cast<std::size_t>(segment)];
  }
  return result;
}

}  // namespace tenryu::core
