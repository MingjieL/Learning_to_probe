#include "_prefix_softmax.h"

#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__)
#define FT_PREFIX_EXP_ROW prefix_exp_row_avx2
#include "_prefix_softmax_vector.h"
#else
#include <stdexcept>
namespace ft_anns {
float prefix_exp_row_avx2(const float*, float*, int64_t) {
  throw std::runtime_error("AVX2 prefix softmax is unavailable in this build");
}
}
#endif
