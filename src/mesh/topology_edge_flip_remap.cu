#include "mesh/topology_edge_flip_remap.hpp"

#include <array>
#include <cmath>
#include <cstddef>

namespace tenryu::mesh {
namespace {

struct Point {
  double r;
  double z;
};

using Triangle = std::array<int, 3>;

struct Polygon {
  std::array<Point, 8> points{};
  int size = 0;
};

struct FieldSnapshot {
  double cell_a;
  double cell_b;
};

bool snapshot_triangle(const MultiBlockTopology& topology,
                       const std::vector<std::uint8_t>& cell_nverts,
                       const int cell,
                       Triangle& triangle) {
  if (cell < 0 ||
      static_cast<std::size_t>(cell) >= cell_nverts.size() ||
      static_cast<std::size_t>(cell + 1) >=
          topology.cell_node_csr_offsets.size() ||
      cell_nverts[static_cast<std::size_t>(cell)] != 3U) {
    return false;
  }
  const int begin =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell)];
  const int end =
      topology.cell_node_csr_offsets[static_cast<std::size_t>(cell + 1)];
  if (begin < 0 || end - begin < 3 ||
      static_cast<std::size_t>(end) >
          topology.cell_node_csr_indices.size()) {
    return false;
  }
  for (int local = 0; local < 3; ++local) {
    triangle[static_cast<std::size_t>(local)] =
        topology.cell_node_csr_indices[
            static_cast<std::size_t>(begin + local)];
  }
  return true;
}

Polygon make_polygon(const Triangle& triangle,
                     const double* node_r,
                     const double* node_z) {
  Polygon polygon;
  polygon.size = 3;
  for (int local = 0; local < 3; ++local) {
    const int node = triangle[static_cast<std::size_t>(local)];
    polygon.points[static_cast<std::size_t>(local)] =
        {node_r[node], node_z[node]};
  }
  return polygon;
}

double cross(const Point& a, const Point& b, const Point& p) {
  return (b.r - a.r) * (p.z - a.z) -
         (b.z - a.z) * (p.r - a.r);
}

double twice_signed_area(const Polygon& polygon) {
  double area2 = 0.0;
  for (int i = 0; i < polygon.size; ++i) {
    const Point& a = polygon.points[static_cast<std::size_t>(i)];
    const Point& b = polygon.points[
        static_cast<std::size_t>((i + 1) % polygon.size)];
    area2 += a.r * b.z - b.r * a.z;
  }
  return area2;
}

double polygon_area(const Polygon& polygon) {
  return 0.5 * std::abs(twice_signed_area(polygon));
}

Polygon convex_clip(Polygon subject, const Polygon& clip) {
  const double orientation =
      twice_signed_area(clip) >= 0.0 ? 1.0 : -1.0;
  for (int edge = 0; edge < clip.size && subject.size > 0; ++edge) {
    const Point& a = clip.points[static_cast<std::size_t>(edge)];
    const Point& b = clip.points[
        static_cast<std::size_t>((edge + 1) % clip.size)];
    Polygon output;
    Point start =
        subject.points[static_cast<std::size_t>(subject.size - 1)];
    double start_side = orientation * cross(a, b, start);
    for (int i = 0; i < subject.size; ++i) {
      const Point end = subject.points[static_cast<std::size_t>(i)];
      const double end_side = orientation * cross(a, b, end);
      const bool start_inside = start_side >= 0.0;
      const bool end_inside = end_side >= 0.0;
      if (start_inside != end_inside) {
        const double t = start_side / (start_side - end_side);
        output.points[static_cast<std::size_t>(output.size++)] = {
            start.r + t * (end.r - start.r),
            start.z + t * (end.z - start.z)};
      }
      if (end_inside) {
        output.points[static_cast<std::size_t>(output.size++)] = end;
      }
      start = end;
      start_side = end_side;
    }
    subject = output;
  }
  return subject;
}

double overlap_area(const Polygon& source, const Polygon& target) {
  return polygon_area(convex_clip(source, target));
}

bool area_partition_is_valid(const double first,
                             const double second,
                             const double area) {
  return std::isfinite(first) && std::isfinite(second) &&
         std::isfinite(area) && area > 0.0 &&
         std::abs((first + second) - area) <= 1.0e-12 * area;
}

void remap_field(double* field,
                 const int cell_a,
                 const int cell_b,
                 const FieldSnapshot& before,
                 const double overlap_a_a2,
                 const double overlap_a_b2,
                 const double overlap_b_a2,
                 const double overlap_b_b2,
                 const double area_a,
                 const double area_b) {
  field[cell_a] =
      before.cell_a * overlap_a_a2 / area_a +
      before.cell_b * overlap_b_a2 / area_b;
  field[cell_b] =
      before.cell_a * overlap_a_b2 / area_a +
      before.cell_b * overlap_b_b2 / area_b;
}

}  // namespace

EdgeFlipRemapResult topology_edge_flip_with_remap(
    MultiBlockTopology& topology,
    std::vector<std::uint8_t>& cell_nverts,
    const double* node_r,
    const double* node_z,
    const int n_nodes,
    EdgeFlipEvent& event,
    double* mass,
    double* mom_r,
    double* mom_z,
    double* e_int) {
  const EdgeFlipRemapResult rejected{false, 0.0, 0.0};
  if (mass == nullptr || mom_r == nullptr || mom_z == nullptr ||
      e_int == nullptr || event.cell_a == event.cell_b) {
    return rejected;
  }

  Triangle pre_a{};
  Triangle pre_b{};
  if (!snapshot_triangle(topology, cell_nverts, event.cell_a, pre_a) ||
      !snapshot_triangle(topology, cell_nverts, event.cell_b, pre_b)) {
    return rejected;
  }
  const FieldSnapshot mass_before{
      mass[event.cell_a], mass[event.cell_b]};
  const FieldSnapshot mom_r_before{
      mom_r[event.cell_a], mom_r[event.cell_b]};
  const FieldSnapshot mom_z_before{
      mom_z[event.cell_a], mom_z[event.cell_b]};
  const FieldSnapshot e_int_before{
      e_int[event.cell_a], e_int[event.cell_b]};
  const EdgeFlipEvent original_event = event;

  if (!topology_edge_flip_apply(topology, cell_nverts, node_r, node_z,
                                n_nodes, event)) {
    return rejected;
  }

  Triangle post_a{};
  Triangle post_b{};
  const bool post_valid =
      snapshot_triangle(topology, cell_nverts, event.cell_a, post_a) &&
      snapshot_triangle(topology, cell_nverts, event.cell_b, post_b);

  double area_a = 0.0;
  double area_b = 0.0;
  double overlap_a_a2 = 0.0;
  double overlap_a_b2 = 0.0;
  double overlap_b_a2 = 0.0;
  double overlap_b_b2 = 0.0;
  if (post_valid) {
    const Polygon polygon_a = make_polygon(pre_a, node_r, node_z);
    const Polygon polygon_b = make_polygon(pre_b, node_r, node_z);
    const Polygon polygon_a2 = make_polygon(post_a, node_r, node_z);
    const Polygon polygon_b2 = make_polygon(post_b, node_r, node_z);
    area_a = polygon_area(polygon_a);
    area_b = polygon_area(polygon_b);
    overlap_a_a2 = overlap_area(polygon_a, polygon_a2);
    overlap_a_b2 = overlap_area(polygon_a, polygon_b2);
    overlap_b_a2 = overlap_area(polygon_b, polygon_a2);
    overlap_b_b2 = overlap_area(polygon_b, polygon_b2);
  }

  if (!post_valid ||
      !area_partition_is_valid(overlap_a_a2, overlap_a_b2, area_a) ||
      !area_partition_is_valid(overlap_b_a2, overlap_b_b2, area_b)) {
    EdgeFlipEvent reverse = topology_edge_flip_reverse(event);
    topology_edge_flip_apply(topology, cell_nverts, node_r, node_z,
                             n_nodes, reverse);
    event = original_event;
    return rejected;
  }

  const double moved_mass =
      mass_before.cell_a * overlap_a_b2 / area_a +
      mass_before.cell_b * overlap_b_a2 / area_b;
  remap_field(mass, event.cell_a, event.cell_b, mass_before,
              overlap_a_a2, overlap_a_b2, overlap_b_a2, overlap_b_b2,
              area_a, area_b);
  remap_field(mom_r, event.cell_a, event.cell_b, mom_r_before,
              overlap_a_a2, overlap_a_b2, overlap_b_a2, overlap_b_b2,
              area_a, area_b);
  remap_field(mom_z, event.cell_a, event.cell_b, mom_z_before,
              overlap_a_a2, overlap_a_b2, overlap_b_a2, overlap_b_b2,
              area_a, area_b);
  remap_field(e_int, event.cell_a, event.cell_b, e_int_before,
              overlap_a_a2, overlap_a_b2, overlap_b_a2, overlap_b_b2,
              area_a, area_b);

  return {true, overlap_a_b2 + overlap_b_a2, moved_mass};
}

}  // namespace tenryu::mesh
