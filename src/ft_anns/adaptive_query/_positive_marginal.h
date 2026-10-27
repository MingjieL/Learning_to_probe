#pragma once
#include "_positive_scan.h"

// Included after the Torch/OpenMP/STL headers by _gain_cost_native.cpp.
namespace ft_anns {

// Integer classification remains valid when the surrounding TU uses fast-math.
inline int64_t positive_stable_top_one(const float* row, int64_t buckets) {
  int64_t best = 0;
  for (int64_t bucket = 0; bucket < buckets; ++bucket) {
    uint32_t bits;
    std::memcpy(&bits, row + bucket, sizeof(bits));
    if ((bits & 0x7fffffffU) > 0x7f800000U) return bucket;
    if (row[bucket] > row[best]) best = bucket;
  }
  return best;
}

class PositiveMarginalWorkspace {
 public:
  struct Thread {
    PositiveIdArena<uint32_t> ids32;
    PositiveIdArena<int64_t> ids64;
    template <typename id_t> PositiveIdArena<id_t>& ids() {
      if constexpr (sizeof(id_t) == sizeof(uint32_t)) return ids32;
      else return ids64;
    }
  };
  int64_t buckets, batch_capacity;
  int thread_count, scratch_bits;
  using MaxRow = float (*)(const float*, int64_t);
  template <typename id_t> using CollectRow = void (*)(const float*, const float*, int64_t,
                                                       float, float, PositiveIdArena<id_t>&);
  MaxRow maximum = positive_max_scalar;
  CollectRow<uint32_t> collect32 = positive_collect_scalar<uint32_t>;
  CollectRow<int64_t> collect64 = positive_collect_scalar<int64_t>;
  std::string scan_backend = "scalar";
  std::vector<Thread> threads;
  std::vector<int64_t> offsets;
  std::vector<int> owners;
  std::vector<int64_t> collected_counts;
  int64_t expected_rows = 0, collected_rows = 0;
  bool collecting = false, all_buckets = false;
  double collected_alpha = 0.0, collected_beta = 0.0, collected_power = 1.0;
  const float* collected_thresholds = nullptr;

  PositiveMarginalWorkspace(int64_t buckets_, int64_t batch_, int scratch_bits_, bool vectorized = true)
      : buckets(buckets_), batch_capacity(batch_), thread_count(at::get_num_threads()),
        scratch_bits(scratch_bits_) {
    TORCH_CHECK(buckets > 0 && batch_capacity > 0, "invalid positive workspace dimensions");
    TORCH_CHECK(scratch_bits == 32 || scratch_bits == 64, "scratch must be uint32 or int64");
    TORCH_CHECK(scratch_bits != 32 || buckets - 1 <= std::numeric_limits<uint32_t>::max(),
                "bucket IDs exceed uint32 scratch capacity");
#ifdef FT_POSITIVE_AVX2
    if (vectorized && __builtin_cpu_supports("avx2")) {
      maximum = positive_max_avx2;
      collect32 = positive_collect_avx2<uint32_t>;
      collect64 = positive_collect_avx2<int64_t>;
      scan_backend = "avx2";
    }
#endif
    const auto workers = std::min<int64_t>(batch_capacity, thread_count);
    threads.resize(workers);
    offsets.resize(batch_capacity);
    owners.resize(batch_capacity);
    const auto capacity = ((batch_capacity + workers - 1) / workers)
                          * std::min<int64_t>(256, buckets);
    for (auto& thread : threads) {
      if (scratch_bits == 32) thread.ids32.reserve(capacity);
      else thread.ids64.reserve(capacity);
    }
  }

  int64_t allocated_bytes() const {
    int64_t bytes = (offsets.capacity() + collected_counts.capacity()) * sizeof(int64_t)
                    + owners.capacity() * sizeof(int);
    for (const auto& thread : threads) {
      bytes += thread.ids32.capacity() * sizeof(uint32_t)
               + thread.ids64.capacity() * sizeof(int64_t);
    }
    return bytes;
  }
};

template <typename id_t, bool CollectOnly = false>
std::vector<torch::Tensor> positive_marginal_plan_impl(
    const torch::Tensor& logits, const torch::Tensor& thresholds,
    double gain_power, double alpha, double beta, PositiveMarginalWorkspace& workspace) {
  TORCH_CHECK(logits.device().is_cpu() && thresholds.device().is_cpu()
                  && logits.scalar_type() == torch::kFloat32
                  && thresholds.scalar_type() == torch::kFloat32
                  && logits.is_contiguous() && thresholds.is_contiguous()
                  && logits.dim() == 2 && thresholds.dim() == 1
                  && logits.size(0) > 0 && logits.size(1) == thresholds.size(0),
              "positive planner requires matching contiguous CPU float32 inputs");
  TORCH_CHECK(std::isfinite(gain_power) && gain_power > 0.0
                  && std::isfinite(alpha) && alpha >= 0.0
                  && std::isfinite(beta) && beta >= 0.0,
              "invalid positive-marginal parameters");
  const int64_t query_count = logits.size(0), bucket_count = logits.size(1);
  TORCH_CHECK(workspace.buckets == bucket_count && query_count <= workspace.batch_capacity
                  && workspace.thread_count == at::get_num_threads(),
              "positive workspace capacity or thread count changed; prepare it again");
  const int workers = std::min<int64_t>(query_count, workspace.thread_count);
  const auto* data = logits.data_ptr<float>();
  const auto* thresholds_data = thresholds.data_ptr<float>();
  const float power = static_cast<float>(gain_power);
  const int64_t base = CollectOnly ? workspace.collected_rows : 0;
  auto counts = CollectOnly ? torch::Tensor{} : torch::empty({query_count}, logits.options().dtype(torch::kInt64));
  auto* count_data = CollectOnly ? workspace.collected_counts.data() + base : counts.data_ptr<int64_t>();
  torch::Tensor routes;
  int64_t width = 1;
  int64_t* output = nullptr;
#pragma omp parallel num_threads(workers)
  {
    const int owner = omp_get_thread_num();
    auto& ids = workspace.threads[owner].template ids<id_t>();
    if constexpr (!CollectOnly) ids.clear();
#pragma omp for schedule(static) nowait
    for (int64_t query = 0; query < query_count; ++query) {
      workspace.owners[base + query] = owner;
      workspace.offsets[base + query] = ids.size();
      if (alpha > 0.0 && beta == 0.0) {
        count_data[query] = bucket_count;
        continue;
      }
      const float* row = data + query * bucket_count;
      // Keep the existing filter's float32 reduction and comparison expressions.
      if (alpha == 0.0) {
        ids.push_back(static_cast<id_t>(positive_stable_top_one(row, bucket_count)));
      } else {
        const float maximum = workspace.maximum(row, bucket_count);
        float shifted_sum = 0.0F;
#pragma omp simd reduction(+ : shifted_sum)
        for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
          shifted_sum += std::exp(row[bucket] - maximum);
        }
        const float log_normalizer = maximum + std::log(shifted_sum);
        if constexpr (sizeof(id_t) == sizeof(uint32_t)) {
          workspace.collect32(row, thresholds_data, bucket_count, log_normalizer, power, ids);
        } else {
          workspace.collect64(row, thresholds_data, bucket_count, log_normalizer, power, ids);
        }
        if (ids.size() == static_cast<size_t>(workspace.offsets[base + query])) {
          ids.push_back(static_cast<id_t>(positive_stable_top_one(row, bucket_count)));
        }
      }
      count_data[query] = ids.size() - workspace.offsets[base + query];
    }
    if constexpr (!CollectOnly) {
#pragma omp barrier
#pragma omp single
      {
        for (int64_t query = 0; query < query_count; ++query) width = std::max(width, count_data[query]);
        routes = torch::empty({query_count, width}, logits.options().dtype(torch::kInt64));
        output = routes.data_ptr<int64_t>();
      }
#pragma omp for schedule(static) nowait
      for (int64_t query = 0; query < query_count; ++query) {
        auto* destination = output + query * width;
        const int64_t count = count_data[query];
        if (alpha > 0.0 && beta == 0.0) {
          std::iota(destination, destination + count, int64_t{0});
        } else {
          const auto& source = workspace.threads[workspace.owners[query]].template ids<id_t>();
          std::copy_n(source.data() + workspace.offsets[query], count, destination);
        }
        std::fill(destination + count, destination + width, int64_t{-1});
      }
    }
  }
  if constexpr (CollectOnly) return {};
  return {routes, counts};
}

inline std::vector<torch::Tensor> positive_marginal_plan_cpu(
    const torch::Tensor& logits, const torch::Tensor& thresholds,
    double gain_power, double alpha, double beta, PositiveMarginalWorkspace& workspace) {
  TORCH_CHECK(!workspace.collecting, "finish or restart the active positive collector before standalone planning");
  if (workspace.scratch_bits == 32) {
    return positive_marginal_plan_impl<uint32_t>(logits, thresholds, gain_power, alpha, beta, workspace);
  }
  return positive_marginal_plan_impl<int64_t>(logits, thresholds, gain_power, alpha, beta, workspace);
}

inline void positive_marginal_begin(PositiveMarginalWorkspace& workspace, int64_t rows) {
  TORCH_CHECK(rows > 0 && workspace.thread_count == at::get_num_threads(), "invalid collector size or thread count");
  workspace.expected_rows = rows;
  workspace.collected_rows = 0;
  workspace.collecting = true;
  workspace.collected_counts.resize(rows);
  workspace.offsets.resize(std::max(rows, workspace.batch_capacity));
  workspace.owners.resize(std::max(rows, workspace.batch_capacity));
  for (auto& thread : workspace.threads) {
    thread.ids32.clear();
    thread.ids64.clear();
  }
}

inline void positive_marginal_append(
    const torch::Tensor& logits, const torch::Tensor& thresholds,
    double power, double alpha, double beta, PositiveMarginalWorkspace& workspace) {
  TORCH_CHECK(workspace.collecting && logits.dim() == 2
                  && workspace.collected_rows + logits.size(0) <= workspace.expected_rows,
              "positive collector is inactive or row capacity exceeded");
  if (workspace.collected_rows == 0) {
    workspace.all_buckets = alpha > 0.0 && beta == 0.0;
    workspace.collected_alpha = alpha;
    workspace.collected_beta = beta;
    workspace.collected_power = power;
    workspace.collected_thresholds = thresholds.data_ptr<float>();
  } else {
    TORCH_CHECK(workspace.collected_alpha == alpha && workspace.collected_beta == beta
                    && workspace.collected_power == power
                    && workspace.collected_thresholds == thresholds.data_ptr<float>(),
                "positive collector policy changed between batches");
  }
  if (workspace.scratch_bits == 32) {
    positive_marginal_plan_impl<uint32_t, true>(logits, thresholds, power, alpha, beta, workspace);
  } else {
    positive_marginal_plan_impl<int64_t, true>(logits, thresholds, power, alpha, beta, workspace);
  }
  workspace.collected_rows += logits.size(0);
}

template <typename id_t>
std::vector<torch::Tensor> positive_marginal_finish_impl(PositiveMarginalWorkspace& workspace) {
  TORCH_CHECK(workspace.collecting && workspace.collected_rows == workspace.expected_rows
                  && workspace.thread_count == at::get_num_threads(),
              "positive collector is incomplete or thread count changed");
  const int64_t rows = workspace.expected_rows;
  const auto width = *std::max_element(workspace.collected_counts.begin(), workspace.collected_counts.end());
  auto options = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt64);
  auto routes = torch::empty({rows, width}, options);
  auto counts = torch::empty({rows}, options);
  auto* output = routes.data_ptr<int64_t>();
  auto* count_data = counts.data_ptr<int64_t>();
#pragma omp parallel for schedule(static) num_threads(workspace.thread_count)
  for (int64_t query = 0; query < rows; ++query) {
    const auto count = workspace.collected_counts[query];
    auto* destination = output + query * width;
    count_data[query] = count;
    if (workspace.all_buckets) {
      std::iota(destination, destination + count, int64_t{0});
    } else {
      const auto& ids = workspace.threads[workspace.owners[query]].template ids<id_t>();
      std::copy_n(ids.data() + workspace.offsets[query], count, destination);
    }
    std::fill(destination + count, destination + width, int64_t{-1});
  }
  workspace.collecting = false;
  return {routes, counts};
}

inline std::vector<torch::Tensor> positive_marginal_finish(PositiveMarginalWorkspace& workspace) {
  if (workspace.scratch_bits == 32) return positive_marginal_finish_impl<uint32_t>(workspace);
  return positive_marginal_finish_impl<int64_t>(workspace);
}

}  // namespace ft_anns
