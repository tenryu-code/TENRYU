#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include "core/rng/philox_cpu.hpp"

namespace {

using tenryu::core::rng::PhiloxCpu;

TEST_CASE("PhiloxCpu uniform consumes one 32-bit word per call", "[core][rng]") {
  constexpr std::uint64_t kGlobalId = 0x0123456789ABCDEFULL;
  constexpr std::uint64_t kUserSeed = 0x0FEDCBA987654321ULL;
  constexpr std::uint64_t kStep = 37ULL;

  PhiloxCpu rng0(kGlobalId, kUserSeed, kStep, 0U);
  PhiloxCpu rng1(kGlobalId, kUserSeed, kStep, 1U);
  PhiloxCpu rng2(kGlobalId, kUserSeed, kStep, 2U);

  const double u0 = rng0.uniform();
  const double u1 = rng0.uniform();
  const double u2 = rng0.uniform();

  REQUIRE(u0 > 0.0);
  REQUIRE(u0 <= 1.0);
  REQUIRE(u1 > 0.0);
  REQUIRE(u1 <= 1.0);
  REQUIRE(u2 > 0.0);
  REQUIRE(u2 <= 1.0);

  REQUIRE(rng0.counter() == 3U);

  REQUIRE(rng1.uniform() == u1);
  REQUIRE(rng1.counter() == 2U);

  REQUIRE(rng2.uniform() == u2);
  REQUIRE(rng2.counter() == 3U);
}

}  // namespace
