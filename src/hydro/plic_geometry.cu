#include "hydro/plic_geometry.cuh"

namespace tenryu::hydro::plic {
namespace {

static_assert(kPlicMaxPolygonVertices >= 8,
              "PLIC clipping requires an eight-vertex working buffer");

}  // namespace
}  // namespace tenryu::hydro::plic
