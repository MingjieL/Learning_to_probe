#pragma once

#if defined(__x86_64__) && defined(__GNUC__)
#include <immintrin.h>
#define FT_POSITIVE_AVX2 1
#endif

namespace ft_anns {

// Live IDs occupy [0, used). Construct storage only when capacity grows, so a
// SIMD mask can append several IDs with one bounds check and no value clearing.
template <typename id_t>
struct PositiveIdArena {
  std::vector<id_t> storage;
  size_t used = 0;
  void reserve(size_t capacity) { storage.resize(capacity); }
  void clear() { used = 0; }
  size_t size() const { return used; }
  size_t capacity() const { return storage.capacity(); }
  const id_t* data() const { return storage.data(); }
  id_t* append_space(size_t count) {
    const size_t required = used + count;
    if (required > storage.size()) {
      storage.resize(std::max(required, std::max<size_t>(16, storage.size() * 2)));
    }
    auto* destination = storage.data() + used;
    used = required;
    return destination;
  }
  void push_back(id_t id) { *append_space(1) = id; }
};

inline float positive_max_scalar(const float* row, int64_t buckets) {
  float maximum = row[0];
  for (int64_t bucket = 1; bucket < buckets; ++bucket) {
    if (row[bucket] > maximum) maximum = row[bucket];
  }
  return maximum;
}

template <typename id_t>
void positive_collect_scalar(const float* row, const float* thresholds, int64_t buckets,
                             float logz, float power, PositiveIdArena<id_t>& ids) {
  for (int64_t bucket = 0; bucket < buckets; ++bucket) {
    if (power * (row[bucket] - logz) > thresholds[bucket]) ids.push_back(static_cast<id_t>(bucket));
  }
}

#ifdef FT_POSITIVE_AVX2
// Separate subtraction/multiplication and ordered comparisons retain float32
// threshold decisions. The exp reduction remains in the original baseline TU.
__attribute__((target("avx2,no-fma"), optimize("no-fast-math", "fp-contract=off")))
inline float positive_max_avx2(const float* row, int64_t buckets) {
  __m256 maximum = _mm256_set1_ps(-std::numeric_limits<float>::infinity());
  int64_t bucket = 0;
  for (; bucket + 8 <= buckets; bucket += 8) {
    maximum = _mm256_max_ps(maximum, _mm256_loadu_ps(row + bucket));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, maximum);
  float result = lanes[0];
  for (int lane = 1; lane < 8; ++lane) if (lanes[lane] > result) result = lanes[lane];
  for (; bucket < buckets; ++bucket) if (row[bucket] > result) result = row[bucket];
  return result;
}

template <typename id_t>
__attribute__((target("avx2,no-fma"), optimize("no-fast-math", "fp-contract=off")))
void positive_collect_avx2(const float* row, const float* thresholds, int64_t buckets,
                           float logz, float power, PositiveIdArena<id_t>& ids) {
  const __m256 normalizer = _mm256_set1_ps(logz), exponent = _mm256_set1_ps(power);
  int64_t bucket = 0;
  for (; bucket + 8 <= buckets; bucket += 8) {
    const __m256 score = _mm256_mul_ps(_mm256_sub_ps(_mm256_loadu_ps(row + bucket), normalizer), exponent);
    unsigned mask = _mm256_movemask_ps(_mm256_cmp_ps(score, _mm256_loadu_ps(thresholds + bucket), _CMP_GT_OQ));
    if (mask) {
      auto* output = ids.append_space(__builtin_popcount(mask));
      do {
        *output++ = static_cast<id_t>(bucket + __builtin_ctz(mask));
        mask &= mask - 1;
      } while (mask);
    }
  }
  for (; bucket < buckets; ++bucket) {
    if (power * (row[bucket] - logz) > thresholds[bucket]) ids.push_back(static_cast<id_t>(bucket));
  }
}
#endif

}  // namespace ft_anns
