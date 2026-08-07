#pragma once

#include <cstdint>

namespace tenryu::core::rng {

// CPU-side Philox4x32-10 stream compatible with curand_init(seed, subsequence, offset).
class PhiloxCpu {
 public:
  PhiloxCpu(const std::uint64_t global_id,
            const std::uint64_t user_seed,
            const std::uint64_t step_number,
            const std::uint32_t offset)
      : state_index_(0U),
        consumed_outputs_(static_cast<std::uint64_t>(offset)) {
    const std::uint64_t seed = global_id ^ user_seed;
    key_[0] = static_cast<std::uint32_t>(seed);
    key_[1] = static_cast<std::uint32_t>(seed >> 32U);

    // Match curand_init(...):
    // subsequence -> high 64 bits of the 128-bit Philox counter.
    counter_[0] = 0U;
    counter_[1] = 0U;
    counter_[2] = static_cast<std::uint32_t>(step_number);
    counter_[3] = static_cast<std::uint32_t>(step_number >> 32U);

    // Match skipahead(offset): offset is counted in 32-bit outputs.
    state_index_ = static_cast<std::uint32_t>(offset & 3U);
    add_to_counter(static_cast<std::uint64_t>(offset >> 2U));
    generate();
  }

  [[nodiscard]] double uniform() {
    // Match curand_uniform_double(curandStatePhilox4_32_10_t*):
    // _curand_uniform_double(curand(state)) where curand(state) returns one uint32.
    constexpr double kTwoPow32Inv = 0x1.0p-32;
    return static_cast<double>(next_u32()) * kTwoPow32Inv + kTwoPow32Inv;
  }

  [[nodiscard]] std::uint32_t counter() const {
    return static_cast<std::uint32_t>(consumed_outputs_);
  }

 private:
  static constexpr std::uint32_t kMul0 = 0xD2511F53u;
  static constexpr std::uint32_t kMul1 = 0xCD9E8D57u;
  static constexpr std::uint32_t kBump0 = 0x9E3779B9u;
  static constexpr std::uint32_t kBump1 = 0xBB67AE85u;

  static void mulhilo32(const std::uint32_t a,
                        const std::uint32_t b,
                        std::uint32_t& hi,
                        std::uint32_t& lo) {
    const std::uint64_t product =
        static_cast<std::uint64_t>(a) * static_cast<std::uint64_t>(b);
    hi = static_cast<std::uint32_t>(product >> 32U);
    lo = static_cast<std::uint32_t>(product);
  }

  void add_to_counter(const std::uint64_t n) {
    std::uint32_t n_hi = static_cast<std::uint32_t>(n >> 32U);
    const std::uint32_t n_lo = static_cast<std::uint32_t>(n);

    const std::uint32_t ctr_x_old = counter_[0];
    counter_[0] += n_lo;
    if (counter_[0] < ctr_x_old) {
      ++n_hi;
    }

    counter_[1] += n_hi;
    if (n_hi <= counter_[1]) {
      return;
    }
    if (++counter_[2] != 0U) {
      return;
    }
    ++counter_[3];
  }

  void increment_counter() {
    if (++counter_[0] != 0U) {
      return;
    }
    if (++counter_[1] != 0U) {
      return;
    }
    if (++counter_[2] != 0U) {
      return;
    }
    ++counter_[3];
  }

  void generate() {
    std::uint32_t c0 = counter_[0];
    std::uint32_t c1 = counter_[1];
    std::uint32_t c2 = counter_[2];
    std::uint32_t c3 = counter_[3];
    std::uint32_t k0 = key_[0];
    std::uint32_t k1 = key_[1];

    for (int round = 0; round < 10; ++round) {
      std::uint32_t hi0 = 0U;
      std::uint32_t lo0 = 0U;
      std::uint32_t hi1 = 0U;
      std::uint32_t lo1 = 0U;
      mulhilo32(kMul0, c0, hi0, lo0);
      mulhilo32(kMul1, c2, hi1, lo1);

      const std::uint32_t n0 = hi1 ^ c1 ^ k0;
      const std::uint32_t n1 = lo1;
      const std::uint32_t n2 = hi0 ^ c3 ^ k1;
      const std::uint32_t n3 = lo0;

      c0 = n0;
      c1 = n1;
      c2 = n2;
      c3 = n3;

      if (round != 9) {
        k0 += kBump0;
        k1 += kBump1;
      }
    }

    output_[0] = c0;
    output_[1] = c1;
    output_[2] = c2;
    output_[3] = c3;
  }

  [[nodiscard]] std::uint32_t next_u32() {
    const std::uint32_t value = output_[state_index_];
    ++state_index_;
    ++consumed_outputs_;

    if (state_index_ == 4U) {
      state_index_ = 0U;
      increment_counter();
      generate();
    }
    return value;
  }

  std::uint32_t counter_[4] = {0U, 0U, 0U, 0U};
  std::uint32_t key_[2] = {0U, 0U};
  std::uint32_t output_[4] = {0U, 0U, 0U, 0U};
  std::uint32_t state_index_;
  std::uint64_t consumed_outputs_;
};

}  // namespace tenryu::core::rng
