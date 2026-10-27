#pragma once

#include <cstdint>

namespace ft_anns {
// Store all shifted exponents in one reusable row; return the float32 inverse sum.
using PrefixExpRow = float (*)(const float*, float*, int64_t);
float prefix_exp_row_avx2(const float* logits, float* exps, int64_t buckets);
float prefix_exp_row_avx512(const float* logits, float* exps, int64_t buckets);
}  // namespace ft_anns
