#pragma once

#include <cstdint>
#include <vector>

namespace tenryu::mesh {

struct CellRing {
  std::uint32_t begin = 0;
  std::uint32_t count = 0;
};

struct CellRingPage {
  std::int32_t owner = -1;
  std::uint8_t owned_count = 0;  // <= 14
  std::int32_t slots[16];  // [0]=pred halo, [1..owned]=owned, [owned+1]=succ halo
  std::uint32_t page_ordinal = 0;
};

struct RingPageSet {
  std::vector<CellRing> rings;  // per cell
  std::vector<std::int32_t> ring_vertices;  // flat, rings index into it
  std::vector<CellRingPage> pages;  // grouped by owner ascending, ordinal ascending
  bool valid = false;
};

RingPageSet build_ring_pages(int n_cells,
                             const int* csr_offsets,  // uniform stride
                             const int* csr_indices,
                             const std::uint8_t* nverts,
                             int corner_stride);

// Validates page grouping and ordinals, owned counts, exact cyclic halo ids,
// and reconstruction of every authoritative ring from its owned slots.
bool validate_ring_pages(const RingPageSet& set, int n_cells);

}  // namespace tenryu::mesh
