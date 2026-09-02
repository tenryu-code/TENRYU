#include "mesh/ring_pages.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace tenryu::mesh {
namespace {

constexpr std::size_t kOwnedSlotsPerPage = 14;

}  // namespace

RingPageSet build_ring_pages(const int n_cells,
                             const int* const csr_offsets,
                             const int* const csr_indices,
                             const std::uint8_t* const nverts,
                             const int corner_stride) {
  RingPageSet candidate;
  if (n_cells < 0 || corner_stride <= 0) {
    return {};
  }
  if (n_cells > 0 &&
      (csr_offsets == nullptr || csr_indices == nullptr || nverts == nullptr)) {
    return {};
  }
  if (n_cells > 0 && csr_offsets[0] != 0) {
    return {};
  }

  candidate.rings.reserve(static_cast<std::size_t>(n_cells));
  for (int cell = 0; cell < n_cells; ++cell) {
    const int begin = csr_offsets[cell];
    const int end = csr_offsets[cell + 1];
    if (begin < 0 || end < begin || end - begin != corner_stride) {
      return {};
    }
    const std::size_t count = nverts[cell];
    if (count == 0 || count > static_cast<std::size_t>(corner_stride) ||
        candidate.ring_vertices.size() >
            std::numeric_limits<std::uint32_t>::max() - count) {
      return {};
    }

    const std::uint32_t ring_begin =
        static_cast<std::uint32_t>(candidate.ring_vertices.size());
    candidate.rings.push_back(
        {ring_begin, static_cast<std::uint32_t>(count)});
    for (std::size_t corner = 0; corner < count; ++corner) {
      candidate.ring_vertices.push_back(
          csr_indices[begin + static_cast<int>(corner)]);
    }

    for (std::size_t owned_begin = 0, ordinal = 0;
         owned_begin < count;
         owned_begin += kOwnedSlotsPerPage, ++ordinal) {
      const std::size_t owned_count =
          std::min(kOwnedSlotsPerPage, count - owned_begin);
      CellRingPage page{};
      page.owner = cell;
      page.owned_count = static_cast<std::uint8_t>(owned_count);
      page.page_ordinal = static_cast<std::uint32_t>(ordinal);
      page.slots[0] = candidate.ring_vertices[
          static_cast<std::size_t>(ring_begin) +
          (owned_begin + count - 1) % count];
      for (std::size_t owned = 0; owned < owned_count; ++owned) {
        page.slots[owned + 1] = candidate.ring_vertices[
            static_cast<std::size_t>(ring_begin) + owned_begin + owned];
      }
      page.slots[owned_count + 1] = candidate.ring_vertices[
          static_cast<std::size_t>(ring_begin) +
          (owned_begin + owned_count) % count];
      candidate.pages.push_back(page);
    }
  }

  candidate.valid = true;
  if (!validate_ring_pages(candidate, n_cells)) {
    return {};
  }
  return candidate;
}

bool validate_ring_pages(const RingPageSet& set, const int n_cells) {
  if (!set.valid || n_cells < 0 ||
      set.rings.size() != static_cast<std::size_t>(n_cells)) {
    return false;
  }

  std::size_t ring_vertex_cursor = 0;
  std::size_t page_cursor = 0;
  for (int cell = 0; cell < n_cells; ++cell) {
    const CellRing& ring = set.rings[static_cast<std::size_t>(cell)];
    const std::size_t ring_begin = ring.begin;
    const std::size_t ring_count = ring.count;
    if (ring_count == 0 || ring_begin != ring_vertex_cursor ||
        ring_begin > set.ring_vertices.size() ||
        ring_count > set.ring_vertices.size() - ring_begin) {
      return false;
    }

    const std::size_t first_page = page_cursor;
    std::size_t owned_begin = 0;
    std::uint32_t ordinal = 0;
    while (owned_begin < ring_count) {
      if (page_cursor >= set.pages.size()) {
        return false;
      }
      const CellRingPage& page = set.pages[page_cursor];
      const std::size_t expected_owned =
          std::min(kOwnedSlotsPerPage, ring_count - owned_begin);
      if (page.owner != cell || page.page_ordinal != ordinal ||
          page.owned_count != expected_owned) {
        return false;
      }

      const std::int32_t expected_predecessor =
          set.ring_vertices[ring_begin +
                            (owned_begin + ring_count - 1) % ring_count];
      if (page.slots[0] != expected_predecessor) {
        return false;
      }
      for (std::size_t owned = 0; owned < expected_owned; ++owned) {
        if (page.slots[owned + 1] !=
            set.ring_vertices[ring_begin + owned_begin + owned]) {
          return false;
        }
      }
      const std::int32_t expected_successor =
          set.ring_vertices[ring_begin +
                            (owned_begin + expected_owned) % ring_count];
      if (page.slots[expected_owned + 1] != expected_successor) {
        return false;
      }

      owned_begin += expected_owned;
      ++page_cursor;
      ++ordinal;
    }

    const std::size_t page_count = page_cursor - first_page;
    for (std::size_t page_offset = 0; page_offset < page_count;
         ++page_offset) {
      const CellRingPage& page = set.pages[first_page + page_offset];
      const CellRingPage& previous =
          set.pages[first_page + (page_offset + page_count - 1) % page_count];
      const CellRingPage& next =
          set.pages[first_page + (page_offset + 1) % page_count];
      if (page.slots[0] != previous.slots[previous.owned_count] ||
          page.slots[static_cast<std::size_t>(page.owned_count) + 1] !=
              next.slots[1]) {
        return false;
      }
    }

    ring_vertex_cursor += ring_count;
  }

  return ring_vertex_cursor == set.ring_vertices.size() &&
         page_cursor == set.pages.size();
}

}  // namespace tenryu::mesh
