// Included only from the instruction-specific translation units below.
// Use the installed Torch vector operations and reduction tree. No static data.
#include <ATen/cpu/vec/functional_base.h>

namespace ft_anns {
float FT_PREFIX_EXP_ROW(const float* logits, float* exps, int64_t buckets) {
  using V = at::vec::Vectorized<float>;
  const float maximum = at::vec::reduce_all<float>(
      [](V a, V b) { return at::vec::maximum(a, b); }, logits, buckets);
  at::vec::map([maximum](V values) { return (values - V(maximum)).exp(); },
               exps, logits, buckets);
  const float denominator = at::vec::reduce_all<float>(
      [](V a, V b) { return a + b; }, exps, buckets);
  return 1.0F / denominator;
}
}  // namespace ft_anns
