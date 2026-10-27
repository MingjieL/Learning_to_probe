#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/Version.h>
#include <torch/version.h>
#include "_prefix_softmax.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <omp.h>
#include <vector>
#include "_positive_marginal.h"


namespace {

using ProfileClock = std::chrono::steady_clock;

template <bool Profile>
ProfileClock::time_point profile_tick() {
  if constexpr (Profile) return ProfileClock::now();
  return {};
}

template <bool Profile>
void profile_record(double* output, int64_t offset, ProfileClock::time_point start) {
  if constexpr (Profile) {
    output[offset] = std::chrono::duration<double>(ProfileClock::now() - start).count();
  }
}

// The extension otherwise uses fast-math for the threshold planner. Prefix
// scores must retain Torch's separate float32 rounding and earliest argmax.
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
struct PrefixUtility {
  float alpha, beta, best = 0.0F;
  double gain = 0.0, cost = 0.0;
  int64_t width = 1;

  void add(float probability, float bucket_cost, int64_t position) {
    gain += probability;
    cost += bucket_cost;
    const float gain_score = static_cast<float>(gain) * alpha;
    const float cost_score = static_cast<float>(cost) * beta;
    const float score = gain_score - cost_score;
    if (position == 0 || (!std::isnan(best) && (std::isnan(score) || score > best))) {
      best = score;
      width = position + 1;
    }
  }
};

torch::Tensor probability_prefix_counts_float32(
    const torch::Tensor& mass,
    const torch::Tensor& normalized_sizes,
    const torch::Tensor& selected,
    double alpha,
    double beta) {
  TORCH_CHECK(mass.device().is_cpu() && normalized_sizes.device().is_cpu()
                  && selected.device().is_cpu(), "prefix inputs must be on CPU");
  TORCH_CHECK(mass.scalar_type() == torch::kFloat32
                  && normalized_sizes.scalar_type() == torch::kFloat32
                  && selected.scalar_type() == torch::kInt64,
              "prefix inputs must be float32 mass/cost and int64 routes");
  TORCH_CHECK(mass.dim() == 2 && normalized_sizes.dim() == 1
                  && selected.dim() == 2, "invalid prefix input dimensions");
  TORCH_CHECK(mass.is_contiguous() && normalized_sizes.is_contiguous(),
              "mass and costs must be contiguous");
  TORCH_CHECK(selected.stride(1) == 1, "routes must have contiguous rows");
  TORCH_CHECK(mass.size(1) == normalized_sizes.size(0)
                  && mass.size(0) == selected.size(0)
                  && selected.size(1) > 0 && selected.size(1) <= mass.size(1),
              "prefix input shapes disagree");
  TORCH_CHECK(std::isfinite(alpha) && std::isfinite(beta)
                  && alpha >= 0 && beta >= 0, "invalid prefix coefficients");
  auto counts = torch::empty({mass.size(0)}, selected.options());
  const auto* probabilities = mass.data_ptr<float>();
  const auto* costs = normalized_sizes.data_ptr<float>();
  const auto* routes = selected.data_ptr<int64_t>();
  auto* output = counts.data_ptr<int64_t>();
  const float a = static_cast<float>(alpha);
  const float b = static_cast<float>(beta);
  std::atomic<bool> invalid{false};
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t row = 0; row < mass.size(0); ++row) {
    // CPU Torch cumsum accumulates float32 inputs in double, then rounds each
    // prefix to float32 before the separate multiply and subtract operations.
    PrefixUtility utility{a, b};
    for (int64_t position = 0; position < selected.size(1); ++position) {
      const int64_t bucket = routes[row * selected.stride(0) + position];
      if (bucket < 0 || bucket >= mass.size(1)) {
        invalid.store(true, std::memory_order_relaxed);
        break;
      }
      utility.add(probabilities[row * mass.size(1) + bucket], costs[bucket], position);
    }
    output[row] = utility.width;
  }
  TORCH_CHECK(!invalid.load(), "prefix bucket ID is out of range");
  return counts;
}
#pragma GCC pop_options

// Ordered L2PR candidates always use partition selection, including small M.
// Disable fast-math here so NaNs and signed-zero ties have a total ordering.
#pragma GCC push_options
#pragma GCC optimize ("no-fast-math", "fp-contract=off")
template <typename scalar_t>
bool sorted_top_m_is_nan(scalar_t value) {
  // Other kernels use fast-math. Bit classification also remains correct in
  // OpenMP/compiler-generated helpers that do not inherit the local options.
  if constexpr (sizeof(scalar_t) == sizeof(uint32_t)) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7fffffff)) > UINT32_C(0x7f800000);
  } else {
    uint64_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT64_C(0x7fffffffffffffff)) > UINT64_C(0x7ff0000000000000);
  }
}

template <typename scalar_t, typename index_t = int64_t>
struct StableLogitOrder {
  const scalar_t* row;
  bool operator()(index_t left, index_t right) const {
    const scalar_t a = row[left], b = row[right];
    const bool a_nan = sorted_top_m_is_nan(a), b_nan = sorted_top_m_is_nan(b);
    if (a_nan || b_nan) return a_nan && (!b_nan || left < right);
    return a > b || (a == b && left < right);
  }
};

// Canonical unsigned float keys preserve the comparator above, including all
// NaNs first, equal signed zeros, and the bucket-ID tie break. Build once per
// row rather than reclassifying floats in every selection/sort comparison.
void make_float_order_keys(const float* row, uint32_t* keys, int64_t buckets) {
  // Honor the worker's current denormal mode, just as float comparisons do.
  // A volatile load prevents constant folding when denormals-as-zero is set.
  const volatile float denormal = std::numeric_limits<float>::denorm_min();
  const uint32_t zero_limit = denormal == 0.0F ? UINT32_C(0x00800000) : UINT32_C(1);
  for (int64_t bucket = 0; bucket < buckets; ++bucket) {
    uint32_t bits;
    std::memcpy(&bits, row + bucket, sizeof(bits));
    const uint32_t magnitude = bits & UINT32_C(0x7fffffff);
    keys[bucket] = magnitude > UINT32_C(0x7f800000) ? UINT32_MAX
        : magnitude < zero_limit ? UINT32_C(0x80000000)
        : bits & UINT32_C(0x80000000) ? ~bits : bits ^ UINT32_C(0x80000000);
  }
}

struct CachedFloatOrder {
  const uint32_t* keys;
  bool operator()(int64_t left, int64_t right) const {
    return keys[left] > keys[right] || (keys[left] == keys[right] && left < right);
  }
};

template <typename Order>
void select_and_sort_prefix(std::vector<int64_t>& ids, int64_t bound, const Order& order) {
  std::iota(ids.begin(), ids.end(), int64_t{0});
  if (bound < static_cast<int64_t>(ids.size())) {
    std::nth_element(ids.begin(), ids.begin() + bound, ids.end(), order);
  }
  std::sort(ids.begin(), ids.begin() + bound, order);
}

template <typename scalar_t, bool Profile = false, typename index_t = int64_t>
void sorted_top_m_loop(const scalar_t* data, int64_t* output,
                       int64_t queries, int64_t buckets, int64_t bound,
                       double* query_times = nullptr, double* thread_times = nullptr) {
#pragma omp parallel num_threads(at::get_num_threads())
  {
    const int thread = Profile ? omp_get_thread_num() : 0;
    auto scratch_started = profile_tick<Profile>();
    std::vector<index_t> ids;
    if (bound != 1) ids.resize(buckets);
    profile_record<Profile>(thread_times, thread * 2, scratch_started);
#pragma omp for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
      const auto* row = data + query * buckets;
      auto step_started = profile_tick<Profile>();
      if (bound == 1) {
        profile_record<Profile>(query_times, query * 4, step_started);
        step_started = profile_tick<Profile>();
        index_t best = 0;
        const StableLogitOrder<scalar_t, index_t> order{row};
        for (int64_t bucket = 1; bucket < buckets; ++bucket) {
          const auto candidate = static_cast<index_t>(bucket);
          if (order(candidate, best)) best = candidate;
        }
        profile_record<Profile>(query_times, query * 4 + 1, step_started);
        step_started = profile_tick<Profile>();
        profile_record<Profile>(query_times, query * 4 + 2, step_started);
        step_started = profile_tick<Profile>();
        output[query * bound] = static_cast<int64_t>(best);
        profile_record<Profile>(query_times, query * 4 + 3, step_started);
        continue;
      }
      std::iota(ids.begin(), ids.end(), index_t{0});
      profile_record<Profile>(query_times, query * 4, step_started);
      const StableLogitOrder<scalar_t, index_t> order{row};
      step_started = profile_tick<Profile>();
      if (bound < buckets) {
        std::nth_element(ids.begin(), ids.begin() + bound, ids.end(), order);
      }
      profile_record<Profile>(query_times, query * 4 + 1, step_started);
      step_started = profile_tick<Profile>();
      std::sort(ids.begin(), ids.begin() + bound, order);
      profile_record<Profile>(query_times, query * 4 + 2, step_started);
      step_started = profile_tick<Profile>();
      std::copy_n(ids.data(), bound, output + query * bound);
      profile_record<Profile>(query_times, query * 4 + 3, step_started);
    }
    if constexpr (Profile) {
      scratch_started = profile_tick<Profile>();
      std::vector<index_t>().swap(ids);
      profile_record<Profile>(thread_times, thread * 2 + 1, scratch_started);
    }
  }
}

void validate_sorted_top_m(const torch::Tensor& logits, int64_t bound);

ft_anns::PrefixExpRow prefix_exp_row_function() {
#if defined(__x86_64__) && defined(__GNUC__) && !defined(__clang__) \
    && TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR == 5 && TORCH_VERSION_PATCH == 1
  const auto capability = at::get_cpu_capability();
  if (capability == "AVX512") return ft_anns::prefix_exp_row_avx512;
  if (capability == "AVX2") return ft_anns::prefix_exp_row_avx2;
#endif
  return nullptr;
}

std::string prefix_probability_backend() {
  return prefix_exp_row_function() ? at::get_cpu_capability() : "unavailable";
}

class PrefixPlannerWorkspace {
 public:
  struct Thread {
    std::vector<int64_t> ids, active;
    std::vector<float> exps;
    std::vector<uint32_t> order_keys;
  };

  int64_t buckets, batch_capacity, thread_capacity;
  std::vector<Thread> threads;
  std::vector<int64_t> offsets;
  std::vector<int> owners;
  int64_t initial_bytes;
  ft_anns::PrefixExpRow exp_row;
  bool cached_order_keys;

  PrefixPlannerWorkspace(int64_t buckets_, int64_t batch_, bool partial_softmax,
                         bool cache_order_keys)
      : buckets(buckets_), batch_capacity(batch_),
        thread_capacity(std::min<int64_t>(batch_, at::get_num_threads())),
        exp_row(partial_softmax ? prefix_exp_row_function() : nullptr),
        cached_order_keys(cache_order_keys) {
    TORCH_CHECK(buckets > 0 && batch_capacity > 0, "invalid prefix workspace dimensions");
    TORCH_CHECK(!partial_softmax || exp_row, "partial softmax is unavailable for this Torch/CPU build");
    threads.resize(thread_capacity);
    offsets.resize(batch_capacity);
    owners.resize(batch_capacity);
    for (auto& thread : threads) {
      thread.ids.resize(buckets);
      if (partial_softmax) thread.exps.resize(buckets);
      if (cached_order_keys) thread.order_keys.resize(buckets);
      thread.active.reserve(((batch_capacity + thread_capacity - 1) / thread_capacity)
                            * std::min<int64_t>(512, buckets));
    }
    initial_bytes = allocated_bytes();
  }

  int64_t allocated_bytes() const {
    int64_t bytes = offsets.capacity() * sizeof(int64_t) + owners.capacity() * sizeof(int);
    for (const auto& thread : threads) {
      bytes += (thread.ids.capacity() + thread.active.capacity()) * sizeof(int64_t)
               + thread.exps.capacity() * sizeof(float)
               + thread.order_keys.capacity() * sizeof(uint32_t);
    }
    return bytes;
  }
};

template <bool ComputeProbabilities>
std::vector<torch::Tensor> probability_prefix_plan_impl(
    const torch::Tensor& logits, const torch::Tensor& mass,
    const torch::Tensor& costs, int64_t bound, double alpha, double beta,
    PrefixPlannerWorkspace& workspace) {
  validate_sorted_top_m(logits, bound);
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && costs.device().is_cpu()
                  && costs.scalar_type() == torch::kFloat32 && costs.is_contiguous()
                  && costs.dim() == 1 && costs.size(0) == logits.size(1),
              "compact prefix requires matching CPU float32 logits/costs");
  const float* probabilities = nullptr;
  if constexpr (ComputeProbabilities) {
    TORCH_CHECK(workspace.exp_row, "prefix workspace must enable partial softmax");
  } else {
    TORCH_CHECK(mass.device().is_cpu() && mass.scalar_type() == torch::kFloat32
                    && mass.is_contiguous() && mass.sizes() == logits.sizes(),
                "compact prefix requires matching CPU float32 probabilities");
    probabilities = mass.data_ptr<float>();
  }
  TORCH_CHECK(std::isfinite(alpha) && alpha >= 0 && std::isfinite(beta) && beta >= 0,
              "invalid prefix coefficients");
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  const int64_t workers = std::min<int64_t>(queries, at::get_num_threads());
  TORCH_CHECK(buckets == workspace.buckets && queries <= workspace.batch_capacity
                  && workers <= workspace.thread_capacity,
              "prefix workspace capacity changed; prepare it again");
  auto counts = torch::empty({queries}, logits.options().dtype(torch::kInt64));
  auto* count_data = counts.data_ptr<int64_t>();
  const auto* data = logits.data_ptr<float>();
  const auto* bucket_costs = costs.data_ptr<float>();
  const float a = static_cast<float>(alpha), b = static_cast<float>(beta);
  torch::Tensor routes;
  int64_t width = 0;
  int64_t* output = nullptr;
#pragma omp parallel num_threads(workers)
  {
    const int owner = omp_get_thread_num();
    auto& thread = workspace.threads[owner];
    thread.active.clear();
#pragma omp for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
      const float* row = data + query * buckets;
      auto& ids = thread.ids;
      const StableLogitOrder<float> order{row};
      if (bound == 1) {
        int64_t best = 0;
        for (int64_t bucket = 1; bucket < buckets; ++bucket) {
          if (order(bucket, best)) best = bucket;
        }
        ids[0] = best;
      } else if (workspace.cached_order_keys) {
        make_float_order_keys(row, thread.order_keys.data(), buckets);
        select_and_sort_prefix(ids, bound, CachedFloatOrder{thread.order_keys.data()});
      } else {
        select_and_sort_prefix(ids, bound, order);
      }
      float inverse_z = 0.0F;
      if constexpr (ComputeProbabilities) {
        inverse_z = workspace.exp_row(row, thread.exps.data(), buckets);
      }
      PrefixUtility utility{a, b};
      for (int64_t position = 0; position < bound; ++position) {
        const int64_t bucket = ids[position];
        const float probability = ComputeProbabilities
            ? thread.exps[bucket] * inverse_z : probabilities[query * buckets + bucket];
        utility.add(probability, bucket_costs[bucket], position);
      }
      count_data[query] = utility.width;
      workspace.owners[query] = owner;
      workspace.offsets[query] = thread.active.size();
      thread.active.insert(thread.active.end(), ids.begin(), ids.begin() + utility.width);
    }
#pragma omp single
    {
      width = *std::max_element(count_data, count_data + queries);
      routes = torch::empty({queries, width}, counts.options());
      output = routes.data_ptr<int64_t>();
    }
#pragma omp for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
      const auto& arena = workspace.threads[workspace.owners[query]].active;
      std::copy_n(arena.data() + workspace.offsets[query], count_data[query], output + query * width);
      std::fill(output + query * width + count_data[query], output + (query + 1) * width, -1);
    }
  }
  return {routes, counts};
}

std::vector<torch::Tensor> probability_prefix_plan_from_mass_cpu(
    const torch::Tensor& logits, const torch::Tensor& mass,
    const torch::Tensor& costs, int64_t bound, double alpha, double beta,
    PrefixPlannerWorkspace& workspace) {
  return probability_prefix_plan_impl<false>(logits, mass, costs, bound, alpha, beta, workspace);
}

std::vector<torch::Tensor> probability_prefix_plan_cpu(
    const torch::Tensor& logits, const torch::Tensor& costs,
    int64_t bound, double alpha, double beta, PrefixPlannerWorkspace& workspace) {
  return probability_prefix_plan_impl<true>(logits, torch::Tensor(), costs, bound, alpha, beta, workspace);
}

torch::Tensor prefix_candidate_probabilities_cpu(
    const torch::Tensor& logits, const torch::Tensor& selected, PrefixPlannerWorkspace& workspace) {
  validate_sorted_top_m(logits, selected.size(1));
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 && selected.device().is_cpu()
                  && selected.dim() == 2 && selected.scalar_type() == torch::kInt64
                  && selected.size(0) == logits.size(0) && selected.stride(1) == 1
                  && workspace.exp_row && logits.size(1) == workspace.buckets
                  && logits.size(0) <= workspace.batch_capacity,
              "invalid partial probability diagnostic inputs");
  const int64_t queries = logits.size(0), buckets = logits.size(1), bound = selected.size(1);
  const int64_t workers = std::min<int64_t>(queries, at::get_num_threads());
  TORCH_CHECK(workers <= workspace.thread_capacity, "prefix workspace thread capacity changed");
  auto result = torch::empty({queries, bound}, logits.options());
  const auto* data = logits.data_ptr<float>();
  const auto* ids = selected.data_ptr<int64_t>();
  auto* output = result.data_ptr<float>();
  std::atomic<bool> invalid{false};
#pragma omp parallel for schedule(static) num_threads(workers)
  for (int64_t query = 0; query < queries; ++query) {
    auto& exps = workspace.threads[omp_get_thread_num()].exps;
    const float inverse_z = workspace.exp_row(data + query * buckets, exps.data(), buckets);
    for (int64_t position = 0; position < bound; ++position) {
      const int64_t bucket = ids[query * selected.stride(0) + position];
      if (bucket < 0 || bucket >= buckets) { invalid.store(true); break; }
      output[query * bound + position] = exps[bucket] * inverse_z;
    }
  }
  TORCH_CHECK(!invalid.load(), "partial probability bucket ID is out of range");
  return result;
}

void validate_sorted_top_m(const torch::Tensor& logits, int64_t bound) {
  TORCH_CHECK(logits.device().is_cpu() && logits.dim() == 2 && logits.is_contiguous()
                  && (logits.scalar_type() == torch::kFloat32
                      || logits.scalar_type() == torch::kFloat64),
              "sorted Top-M requires contiguous CPU float32/float64 logits");
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  TORCH_CHECK(queries > 0 && bound > 0 && bound <= buckets, "invalid sorted Top-M bound");
}

template <typename index_t>
void validate_sorted_top_m_scratch(const torch::Tensor& logits, int64_t bound) {
  validate_sorted_top_m(logits, bound);
  TORCH_CHECK(logits.size(1) <= std::numeric_limits<index_t>::max(),
              "bucket count exceeds selector scratch integer range");
}

template <typename index_t = int64_t>
torch::Tensor sorted_top_m_cpu_impl(const torch::Tensor& logits, int64_t bound) {
  validate_sorted_top_m_scratch<index_t>(logits, bound);
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  auto selected = torch::empty({queries, bound}, logits.options().dtype(torch::kInt64));
  auto* output = selected.data_ptr<int64_t>();
  if (logits.scalar_type() == torch::kFloat32) {
    sorted_top_m_loop<float, false, index_t>(logits.data_ptr<float>(), output, queries, buckets, bound);
  } else {
    sorted_top_m_loop<double, false, index_t>(logits.data_ptr<double>(), output, queries, buckets, bound);
  }
  return selected;
}

torch::Tensor sorted_top_m_cpu(const torch::Tensor& logits, int64_t bound) {
  return sorted_top_m_cpu_impl<int64_t>(logits, bound);
}

torch::Tensor sorted_top_m_int32_scratch_cpu(const torch::Tensor& logits, int64_t bound) {
  return sorted_top_m_cpu_impl<int32_t>(logits, bound);
}

// Diagnostic entry only: production instantiates the same loop with Profile=false.
template <typename index_t = int64_t>
std::vector<torch::Tensor> profile_sorted_top_m_cpu_impl(
    const torch::Tensor& logits, int64_t bound) {
  validate_sorted_top_m_scratch<index_t>(logits, bound);
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  auto started = ProfileClock::now();
  auto selected = torch::empty({queries, bound}, logits.options().dtype(torch::kInt64));
  auto timing_options = logits.options().dtype(torch::kFloat64);
  auto query_times = torch::empty({queries, 4}, timing_options);
  auto thread_times = torch::zeros({at::get_num_threads(), 2}, timing_options);
  auto shared_times = torch::empty({2}, timing_options);
  profile_record<true>(shared_times.data_ptr<double>(), 0, started);
  started = ProfileClock::now();
  if (logits.scalar_type() == torch::kFloat32) {
    sorted_top_m_loop<float, true, index_t>(logits.data_ptr<float>(), selected.data_ptr<int64_t>(),
        queries, buckets, bound, query_times.data_ptr<double>(), thread_times.data_ptr<double>());
  } else {
    sorted_top_m_loop<double, true, index_t>(logits.data_ptr<double>(), selected.data_ptr<int64_t>(),
        queries, buckets, bound, query_times.data_ptr<double>(), thread_times.data_ptr<double>());
  }
  profile_record<true>(shared_times.data_ptr<double>(), 1, started);
  return {selected, query_times, thread_times, shared_times};
}

std::vector<torch::Tensor> profile_sorted_top_m_cpu(const torch::Tensor& logits, int64_t bound) {
  return profile_sorted_top_m_cpu_impl<int64_t>(logits, bound);
}

std::vector<torch::Tensor> profile_sorted_top_m_int32_scratch_cpu(
    const torch::Tensor& logits, int64_t bound) {
  return profile_sorted_top_m_cpu_impl<int32_t>(logits, bound);
}
#pragma GCC pop_options

template <bool Profile>
std::vector<torch::Tensor> positive_marginal_filter_float32_impl(
    const torch::Tensor& logits,
    const torch::Tensor& log_thresholds,
    double gain_power,
    double alpha,
    double beta, double* query_times = nullptr, double* shared_times = nullptr) {
  TORCH_CHECK(logits.device().is_cpu() && log_thresholds.device().is_cpu(),
              "positive-marginal inputs must be on CPU");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 &&
                  log_thresholds.scalar_type() == torch::kFloat32,
              "positive-marginal inputs must be float32");
  TORCH_CHECK(logits.dim() == 2 && log_thresholds.dim() == 1,
              "invalid positive-marginal input dimensions");
  TORCH_CHECK(logits.is_contiguous() && log_thresholds.is_contiguous(),
              "positive-marginal inputs must be contiguous");
  TORCH_CHECK(logits.size(0) > 0 && logits.size(1) == log_thresholds.size(0),
              "positive-marginal input shapes disagree");
  TORCH_CHECK(std::isfinite(gain_power) && gain_power > 0.0 &&
                  std::isfinite(alpha) && alpha >= 0.0 &&
                  std::isfinite(beta) && beta >= 0.0,
              "invalid positive-marginal parameters");

  const int64_t query_count = logits.size(0);
  const int64_t bucket_count = logits.size(1);
  const float* logits_data = logits.data_ptr<float>();
  const float* thresholds_data = log_thresholds.data_ptr<float>();
  const float power = static_cast<float>(gain_power);
  auto phase_started = profile_tick<Profile>();
  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto counts = torch::empty({query_count}, options);
  auto top_ones = torch::empty({query_count}, options);
  auto log_normalizers = torch::empty(
      {query_count}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  int64_t* counts_data = counts.data_ptr<int64_t>();
  int64_t* top_ones_data = top_ones.data_ptr<int64_t>();
  float* log_normalizers_data = log_normalizers.data_ptr<float>();
  profile_record<Profile>(shared_times, 0, phase_started);
  phase_started = profile_tick<Profile>();

#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < query_count; ++query) {
    const float* row = logits_data + query * bucket_count;
    auto step_started = profile_tick<Profile>();
    float maximum = row[0];
    int64_t top_one = 0;
    for (int64_t bucket = 1; bucket < bucket_count; ++bucket) {
      if (row[bucket] > maximum) {
        maximum = row[bucket];
        top_one = bucket;
      }
    }
    top_ones_data[query] = top_one;
    profile_record<Profile>(query_times, query * 4, step_started);
    if (alpha == 0.0) {
      counts_data[query] = 1;
      log_normalizers_data[query] = 0.0F;
      continue;
    }
    if (beta == 0.0) {
      counts_data[query] = bucket_count;
      log_normalizers_data[query] = 0.0F;
      continue;
    }
    float shifted_sum = 0.0F;
    step_started = profile_tick<Profile>();
#pragma omp simd reduction(+ : shifted_sum)
    for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
      shifted_sum += std::exp(row[bucket] - maximum);
    }
    const float log_normalizer = maximum + std::log(shifted_sum);
    log_normalizers_data[query] = log_normalizer;
    profile_record<Profile>(query_times, query * 4 + 1, step_started);
    step_started = profile_tick<Profile>();
    int64_t qualifying_count = 0;
#pragma omp simd reduction(+ : qualifying_count)
    for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
      qualifying_count +=
          power * (row[bucket] - log_normalizer) > thresholds_data[bucket];
    }
    counts_data[query] = std::max<int64_t>(1, qualifying_count);
    profile_record<Profile>(query_times, query * 4 + 2, step_started);
  }
  profile_record<Profile>(shared_times, 1, phase_started);
  phase_started = profile_tick<Profile>();

  int64_t route_width = 1;
  for (int64_t query = 0; query < query_count; ++query) {
    route_width = std::max(route_width, counts_data[query]);
  }
  auto selected = torch::full({query_count, route_width}, -1, options);
  int64_t* selected_data = selected.data_ptr<int64_t>();
  profile_record<Profile>(shared_times, 2, phase_started);
  phase_started = profile_tick<Profile>();

#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < query_count; ++query) {
    int64_t* output = selected_data + query * route_width;
    const auto step_started = profile_tick<Profile>();
    if (alpha == 0.0) {
      output[0] = ft_anns::positive_stable_top_one(logits_data + query * bucket_count, bucket_count);
      profile_record<Profile>(query_times, query * 4 + 3, step_started);
      continue;
    }
    if (beta == 0.0) {
      for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
        output[bucket] = bucket;
      }
      profile_record<Profile>(query_times, query * 4 + 3, step_started);
      continue;
    }
    const float* row = logits_data + query * bucket_count;
    const float log_normalizer = log_normalizers_data[query];
    int64_t position = 0;
    for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
      if (power * (row[bucket] - log_normalizer) > thresholds_data[bucket]) {
        output[position++] = bucket;
      }
    }
    if (position == 0) {
      output[0] = ft_anns::positive_stable_top_one(row, bucket_count);
    }
    profile_record<Profile>(query_times, query * 4 + 3, step_started);
  }
  profile_record<Profile>(shared_times, 3, phase_started);
  return {selected, counts};
}

template <bool Profile>
torch::Tensor unsorted_top_m_float32_impl(const torch::Tensor& logits, int64_t bound,
                                        double* query_times = nullptr,
                                        double* shared_times = nullptr) {
  TORCH_CHECK(logits.device().is_cpu() && logits.scalar_type() == torch::kFloat32 &&
                  logits.dim() == 2 && logits.is_contiguous(),
              "Top-M requires contiguous CPU float32 logits");
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  TORCH_CHECK(queries > 0 && bound > 0 && bound <= buckets, "invalid Top-M bound");
  auto phase_started = profile_tick<Profile>();
  auto selected = torch::empty({queries, bound}, logits.options().dtype(torch::kInt64));
  const float* data = logits.data_ptr<float>();
  int64_t* output = selected.data_ptr<int64_t>();
  profile_record<Profile>(shared_times, 0, phase_started);
  phase_started = profile_tick<Profile>();
#pragma omp parallel num_threads(at::get_num_threads())
  {
    std::vector<int64_t> ids(buckets);
#pragma omp for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
      const float* row = data + query * buckets;
      auto step_started = profile_tick<Profile>();
      std::iota(ids.begin(), ids.end(), 0);
      profile_record<Profile>(query_times, query * 3, step_started);
      step_started = profile_tick<Profile>();
      if (bound < buckets) {
        std::nth_element(ids.begin(), ids.begin() + bound, ids.end(),
                         [row](int64_t left, int64_t right) {
                           return row[left] > row[right] ||
                                  (row[left] == row[right] && left < right);
                         });
      }
      profile_record<Profile>(query_times, query * 3 + 1, step_started);
      step_started = profile_tick<Profile>();
      std::copy_n(ids.data(), bound, output + query * bound);
      profile_record<Profile>(query_times, query * 3 + 2, step_started);
    }
  }
  profile_record<Profile>(shared_times, 1, phase_started);
  return selected;
}

template <bool Profile>
torch::Tensor log_normalizers_float32_impl(const torch::Tensor& logits,
                                         double* query_times = nullptr,
                                         double* shared_times = nullptr) {
  TORCH_CHECK(logits.device().is_cpu() && logits.scalar_type() == torch::kFloat32 &&
                  logits.dim() == 2 && logits.is_contiguous() &&
                  logits.size(0) > 0 && logits.size(1) > 0,
              "normalization requires non-empty contiguous CPU float32 logits");
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  auto phase_started = profile_tick<Profile>();
  auto normalizers = torch::empty({queries}, logits.options());
  const float* data = logits.data_ptr<float>();
  float* output = normalizers.data_ptr<float>();
  profile_record<Profile>(shared_times, 0, phase_started);
  phase_started = profile_tick<Profile>();
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < queries; ++query) {
    const float* row = data + query * buckets;
    auto step_started = profile_tick<Profile>();
    float maximum = row[0];
    for (int64_t bucket = 1; bucket < buckets; ++bucket) {
      maximum = std::max(maximum, row[bucket]);
    }
    profile_record<Profile>(query_times, query * 2, step_started);
    step_started = profile_tick<Profile>();
    float shifted_sum = 0.0F;
#pragma omp simd reduction(+ : shifted_sum)
    for (int64_t bucket = 0; bucket < buckets; ++bucket) {
      shifted_sum += std::exp(row[bucket] - maximum);
    }
    output[query] = maximum + std::log(shifted_sum);
    profile_record<Profile>(query_times, query * 2 + 1, step_started);
  }
  profile_record<Profile>(shared_times, 1, phase_started);
  return normalizers;
}

template <bool Profile>
std::vector<torch::Tensor> top_m_positive_marginal_float32_impl(
    const torch::Tensor& logits, const torch::Tensor& candidates,
    const torch::Tensor& normalizers, const torch::Tensor& thresholds,
    double gain_power, double alpha, double beta,
    double* query_times = nullptr, double* shared_times = nullptr) {
  TORCH_CHECK(logits.device().is_cpu() && candidates.device().is_cpu() &&
                  normalizers.device().is_cpu() && thresholds.device().is_cpu(),
              "Top-M filter requires CPU tensors");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 &&
                  candidates.scalar_type() == torch::kInt64 &&
                  normalizers.scalar_type() == torch::kFloat32 &&
                  thresholds.scalar_type() == torch::kFloat32,
              "invalid Top-M filter dtypes");
  TORCH_CHECK(logits.dim() == 2 && candidates.dim() == 2 &&
                  normalizers.dim() == 1 && thresholds.dim() == 1 &&
                  logits.is_contiguous() && candidates.is_contiguous() &&
                  normalizers.is_contiguous() && thresholds.is_contiguous(),
              "invalid Top-M filter shapes/layouts");
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  const int64_t bound = candidates.size(1);
  TORCH_CHECK(queries > 0 && bound > 0 && bound <= buckets &&
                  candidates.size(0) == queries && normalizers.size(0) == queries &&
                  thresholds.size(0) == buckets, "incompatible Top-M filter tensors");
  TORCH_CHECK(std::isfinite(gain_power) && gain_power > 0 &&
                  std::isfinite(alpha) && alpha >= 0 &&
                  std::isfinite(beta) && beta >= 0, "invalid filter coefficients");
  const float* data = logits.data_ptr<float>();
  const int64_t* routes = candidates.data_ptr<int64_t>();
  const float* log_z = normalizers.data_ptr<float>();
  const float* cutoffs = thresholds.data_ptr<float>();
  const float power = static_cast<float>(gain_power);
  auto phase_started = profile_tick<Profile>();
  auto raw_counts = torch::empty({queries}, candidates.options());
  auto counts = torch::empty({queries}, candidates.options());
  auto top_ones = torch::empty({queries}, candidates.options());
  auto flags = torch::empty({queries, bound}, candidates.options().dtype(torch::kUInt8));
  int64_t* raw = raw_counts.data_ptr<int64_t>();
  int64_t* count = counts.data_ptr<int64_t>();
  int64_t* top = top_ones.data_ptr<int64_t>();
  uint8_t* active = flags.data_ptr<uint8_t>();
  std::atomic<bool> invalid{false};
  profile_record<Profile>(shared_times, 0, phase_started);
  phase_started = profile_tick<Profile>();
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < queries; ++query) {
    const float* row = data + query * buckets;
    const auto step_started = profile_tick<Profile>();
    int64_t best = -1, qualifying = 0;
    for (int64_t position = 0; position < bound; ++position) {
      const int64_t bucket = routes[query * bound + position];
      if (bucket < 0 || bucket >= buckets) {
        invalid.store(true);
        continue;
      }
      if (best < 0 || row[bucket] > row[best] ||
          (row[bucket] == row[best] && bucket < best)) {
        best = bucket;
      }
      const bool keep = alpha > 0 &&
          (beta == 0 || power * (row[bucket] - log_z[query]) > cutoffs[bucket]);
      active[query * bound + position] = keep;
      qualifying += keep;
    }
    top[query] = best;
    raw[query] = qualifying;
    count[query] = std::max<int64_t>(1, qualifying);
    profile_record<Profile>(query_times, query * 2, step_started);
  }
  profile_record<Profile>(shared_times, 1, phase_started);
  phase_started = profile_tick<Profile>();
  TORCH_CHECK(!invalid.load(), "Top-M candidate ID out of range");
  int64_t width = 1;
  for (int64_t query = 0; query < queries; ++query) width = std::max(width, count[query]);
  auto selected = torch::full({queries, width}, -1, candidates.options());
  int64_t* output = selected.data_ptr<int64_t>();
  profile_record<Profile>(shared_times, 2, phase_started);
  phase_started = profile_tick<Profile>();
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < queries; ++query) {
    const auto step_started = profile_tick<Profile>();
    int64_t position = 0;
    for (int64_t candidate = 0; candidate < bound; ++candidate) {
      if (active[query * bound + candidate]) {
        output[query * width + position++] = routes[query * bound + candidate];
      }
    }
    if (position == 0) output[query * width] = top[query];
    profile_record<Profile>(query_times, query * 2 + 1, step_started);
  }
  profile_record<Profile>(shared_times, 3, phase_started);
  return {selected, counts, raw_counts};
}

std::vector<torch::Tensor> positive_marginal_filter_float32(
    const torch::Tensor& logits, const torch::Tensor& thresholds,
    double power, double alpha, double beta) {
  return positive_marginal_filter_float32_impl<false>(logits, thresholds, power, alpha, beta);
}

torch::Tensor unsorted_top_m_float32(const torch::Tensor& logits, int64_t bound) {
  return unsorted_top_m_float32_impl<false>(logits, bound);
}

torch::Tensor log_normalizers_float32(const torch::Tensor& logits) {
  return log_normalizers_float32_impl<false>(logits);
}

std::vector<torch::Tensor> top_m_positive_marginal_float32(
    const torch::Tensor& logits, const torch::Tensor& candidates,
    const torch::Tensor& normalizers, const torch::Tensor& thresholds,
    double power, double alpha, double beta) {
  return top_m_positive_marginal_float32_impl<false>(
      logits, candidates, normalizers, thresholds, power, alpha, beta);
}

// Diagnostic specializations retain the original loops and arithmetic. Query
// durations overlap across workers; only shared-region durations are wall time.
std::vector<torch::Tensor> profile_positive_marginal_filter_float32(
    const torch::Tensor& logits, const torch::Tensor& thresholds,
    double power, double alpha, double beta) {
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto queries = torch::zeros({logits.size(0), 4}, options);
  auto shared = torch::zeros({4}, options);
  auto result = positive_marginal_filter_float32_impl<true>(
      logits, thresholds, power, alpha, beta, queries.data_ptr<double>(), shared.data_ptr<double>());
  result.push_back(queries);
  result.push_back(shared);
  return result;
}

std::vector<torch::Tensor> profile_unsorted_top_m_float32(const torch::Tensor& logits, int64_t bound) {
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto queries = torch::zeros({logits.size(0), 3}, options);
  auto shared = torch::zeros({2}, options);
  auto selected = unsorted_top_m_float32_impl<true>(
      logits, bound, queries.data_ptr<double>(), shared.data_ptr<double>());
  return {selected, queries, shared};
}

std::vector<torch::Tensor> profile_log_normalizers_float32(const torch::Tensor& logits) {
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto queries = torch::zeros({logits.size(0), 2}, options);
  auto shared = torch::zeros({2}, options);
  auto normalizers = log_normalizers_float32_impl<true>(
      logits, queries.data_ptr<double>(), shared.data_ptr<double>());
  return {normalizers, queries, shared};
}

std::vector<torch::Tensor> profile_top_m_positive_marginal_float32(
    const torch::Tensor& logits, const torch::Tensor& candidates,
    const torch::Tensor& normalizers, const torch::Tensor& thresholds,
    double power, double alpha, double beta) {
  auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);
  auto queries = torch::zeros({logits.size(0), 2}, options);
  auto shared = torch::zeros({4}, options);
  auto result = top_m_positive_marginal_float32_impl<true>(
      logits, candidates, normalizers, thresholds, power, alpha, beta,
      queries.data_ptr<double>(), shared.data_ptr<double>());
  result.push_back(queries);
  result.push_back(shared);
  return result;
}

// The certified adapter reuses the original Top-M normalizer and predicate.
// Its compact positives are stored only at the actual batch positive width.
void check_positive_inputs(const torch::Tensor& logits,
                           const torch::Tensor& normalizers,
                           const torch::Tensor& thresholds,
                           double gain_power, double alpha, double beta) {
  TORCH_CHECK(logits.device().is_cpu() && normalizers.device().is_cpu() &&
                  thresholds.device().is_cpu(), "positive inputs require CPU");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32 &&
                  normalizers.scalar_type() == torch::kFloat32 &&
                  thresholds.scalar_type() == torch::kFloat32 &&
                  logits.dim() == 2 && normalizers.dim() == 1 && thresholds.dim() == 1 &&
                  logits.is_contiguous() && normalizers.is_contiguous() &&
                  thresholds.is_contiguous(), "invalid positive input layout");
  TORCH_CHECK(logits.size(0) > 0 && logits.size(1) > 0 &&
                  normalizers.size(0) == logits.size(0) &&
                  thresholds.size(0) == logits.size(1), "positive input shapes disagree");
  TORCH_CHECK(std::isfinite(gain_power) && gain_power > 0 &&
                  std::isfinite(alpha) && alpha >= 0 &&
                  std::isfinite(beta) && beta >= 0, "invalid positive coefficients");
}

std::vector<torch::Tensor> collect_positive_buckets_float32(
    const torch::Tensor& logits, const torch::Tensor& normalizers,
    const torch::Tensor& thresholds, double gain_power, double alpha, double beta) {
  check_positive_inputs(logits, normalizers, thresholds, gain_power, alpha, beta);
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  const float* data = logits.data_ptr<float>();
  const float* log_z = normalizers.data_ptr<float>();
  const float* cutoffs = thresholds.data_ptr<float>();
  const float power = static_cast<float>(gain_power);
  auto options = logits.options().dtype(torch::kInt64);
  auto raw_counts = torch::empty({queries}, options);
  auto worst = torch::empty({queries}, options);
  auto top = torch::empty({queries}, options);
  auto* raw = raw_counts.data_ptr<int64_t>();
  auto* last = worst.data_ptr<int64_t>();
  auto* best = top.data_ptr<int64_t>();
  std::vector<std::vector<int64_t>> positives(queries);
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < queries; ++query) {
    const float* row = data + query * buckets;
    auto& chosen = positives[query];
    int64_t top_one = 0, worst_one = -1;
    for (int64_t bucket = 0; bucket < buckets; ++bucket) {
      if (row[bucket] > row[top_one]) top_one = bucket;
      const bool keep = alpha > 0 &&
          (beta == 0 || power * (row[bucket] - log_z[query]) > cutoffs[bucket]);
      if (keep) {
        chosen.push_back(bucket);
        // Ascending bucket iteration makes the larger ID the worst equal score.
        if (worst_one < 0 || row[bucket] <= row[worst_one]) worst_one = bucket;
      }
    }
    raw[query] = chosen.size();
    last[query] = worst_one;
    best[query] = top_one;
  }
  int64_t width = 1;
  for (int64_t query = 0; query < queries; ++query) width = std::max(width, raw[query]);
  auto selected = torch::full({queries, width}, -1, options);
  auto* output = selected.data_ptr<int64_t>();
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < queries; ++query) {
    std::copy(positives[query].begin(), positives[query].end(), output + query * width);
    if (raw[query] == 0) output[query * width] = best[query];
  }
  return {selected, raw_counts, worst, top};
}

torch::Tensor positive_worst_global_ranks_float32(
    const torch::Tensor& logits, const torch::Tensor& worst) {
  TORCH_CHECK(logits.device().is_cpu() && logits.scalar_type() == torch::kFloat32 &&
                  logits.dim() == 2 && logits.is_contiguous() &&
                  logits.size(0) > 0 && logits.size(1) > 0 &&
                  worst.device().is_cpu() && worst.scalar_type() == torch::kInt64 &&
                  worst.dim() == 1 && worst.is_contiguous() &&
                  worst.size(0) == logits.size(0), "invalid global rank inputs");
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  const float* data = logits.data_ptr<float>();
  const auto* last = worst.data_ptr<int64_t>();
  auto ranks = torch::zeros_like(worst);
  auto* output = ranks.data_ptr<int64_t>();
  std::atomic<bool> invalid{false};
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t query = 0; query < queries; ++query) {
    if (last[query] == -1) continue;  // Empty P skips the certificate entirely.
    if (last[query] < 0 || last[query] >= buckets) {
      invalid.store(true);
      continue;
    }
    const float* row = data + query * buckets;
    const float value = row[last[query]];
    int64_t before = 0;
#pragma omp simd reduction(+ : before)
    for (int64_t bucket = 0; bucket < buckets; ++bucket) {
      before += row[bucket] > value || (row[bucket] == value && bucket < last[query]);
    }
    output[query] = before + 1;
  }
  TORCH_CHECK(!invalid.load(), "worst positive bucket is out of range");
  return ranks;
}

std::vector<torch::Tensor> restrict_positive_top_m_float32(
    const torch::Tensor& logits, const torch::Tensor& positives,
    const torch::Tensor& raw_counts, const torch::Tensor& ranks,
    const torch::Tensor& top, int64_t bound,
    const torch::Tensor& normalizers, const torch::Tensor& thresholds,
    double gain_power, double alpha, double beta) {
  check_positive_inputs(logits, normalizers, thresholds, gain_power, alpha, beta);
  const int64_t queries = logits.size(0), buckets = logits.size(1);
  TORCH_CHECK(bound > 0 && bound <= buckets, "invalid certified Top-M bound");
  for (const auto& tensor : {positives, raw_counts, ranks, top}) {
    TORCH_CHECK(tensor.device().is_cpu() && tensor.scalar_type() == torch::kInt64 &&
                    tensor.is_contiguous(), "invalid positive route layout");
  }
  TORCH_CHECK(positives.dim() == 2 && positives.size(0) == queries &&
                  positives.size(1) > 0 && positives.size(1) <= buckets,
              "invalid positive route shape");
  for (const auto& tensor : {raw_counts, ranks, top}) {
    TORCH_CHECK(tensor.dim() == 1 && tensor.size(0) == queries,
                "invalid positive metadata shape");
  }
  const int64_t positive_width = positives.size(1);
  const int64_t width = std::min(bound, positive_width);
  auto selected = torch::full({queries, width}, -1, positives.options());
  auto counts = torch::empty_like(raw_counts);
  auto final_raw_counts = torch::empty_like(raw_counts);
  const float* data = logits.data_ptr<float>();
  const float* log_z = normalizers.data_ptr<float>();
  const float* cutoffs = thresholds.data_ptr<float>();
  const auto* positive_ids = positives.data_ptr<int64_t>();
  const auto* raw = raw_counts.data_ptr<int64_t>();
  const auto* rank = ranks.data_ptr<int64_t>();
  const auto* best = top.data_ptr<int64_t>();
  auto* output = selected.data_ptr<int64_t>();
  auto* count = counts.data_ptr<int64_t>();
  auto* final_raw = final_raw_counts.data_ptr<int64_t>();
  const float power = static_cast<float>(gain_power);
  std::atomic<bool> invalid{false};
#pragma omp parallel num_threads(at::get_num_threads())
  {
    std::vector<int64_t> ids;  // Allocate B IDs only for a failing query.
#pragma omp for schedule(static)
    for (int64_t query = 0; query < queries; ++query) {
      if (raw[query] < 0 || raw[query] > positive_width || rank[query] < 0 ||
          rank[query] > buckets || best[query] < 0 || best[query] >= buckets ||
          (rank[query] <= bound && raw[query] > width)) {
        invalid.store(true);
        continue;
      }
      int64_t qualifying = 0;
      int64_t* destination = output + query * width;
      if (rank[query] <= bound) {
        qualifying = raw[query];
        std::copy_n(positive_ids + query * positive_width, qualifying, destination);
      } else {
        ids.resize(buckets);
        std::iota(ids.begin(), ids.end(), 0);
        const float* row = data + query * buckets;
        std::nth_element(ids.begin(), ids.begin() + bound, ids.end(),
                         [row](int64_t left, int64_t right) {
                           return row[left] > row[right] ||
                                  (row[left] == row[right] && left < right);
                         });
        for (int64_t position = 0; position < bound; ++position) {
          const int64_t bucket = ids[position];
          if (alpha > 0 && (beta == 0 ||
              power * (row[bucket] - log_z[query]) > cutoffs[bucket])) {
            destination[qualifying++] = bucket;
          }
        }
        std::sort(destination, destination + qualifying);  // Sort only survivors.
      }
      final_raw[query] = qualifying;
      count[query] = std::max<int64_t>(1, qualifying);
      if (qualifying == 0) destination[0] = best[query];
    }
  }
  TORCH_CHECK(!invalid.load(), "invalid positive count/rank/top bucket");
  return {selected, counts, final_raw_counts};
}

std::vector<torch::Tensor> marginal_threshold_float32(
    const torch::Tensor& logits,
    const torch::Tensor& normalized_sizes,
    int64_t candidate_bound,
    double alpha,
    double beta,
    bool bucket_id_order) {
  TORCH_CHECK(logits.device().is_cpu(), "logits must be on CPU");
  TORCH_CHECK(normalized_sizes.device().is_cpu(), "normalized_sizes must be on CPU");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32, "logits must be float32");
  TORCH_CHECK(
      normalized_sizes.scalar_type() == torch::kFloat32,
      "normalized_sizes must be float32");
  TORCH_CHECK(logits.is_contiguous(), "logits must be contiguous");
  TORCH_CHECK(normalized_sizes.is_contiguous(), "normalized_sizes must be contiguous");
  TORCH_CHECK(logits.dim() == 2, "logits must be a matrix");
  TORCH_CHECK(normalized_sizes.dim() == 1, "normalized_sizes must be a vector");
  TORCH_CHECK(
      logits.size(1) == normalized_sizes.size(0),
      "logits and normalized_sizes bucket counts differ");
  TORCH_CHECK(candidate_bound > 0 && candidate_bound <= logits.size(1),
              "candidate_bound is outside the bucket range");
  TORCH_CHECK(alpha > 0.0 && beta > 0.0, "alpha and beta must be positive");

  const int64_t query_count = logits.size(0);
  const int64_t bucket_count = logits.size(1);
  const float* logits_data = logits.data_ptr<float>();
  const float* sizes_data = normalized_sizes.data_ptr<float>();
  const float log_ratio = static_cast<float>(std::log(beta) - std::log(alpha));

  auto counts = torch::empty(
      {query_count}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  auto cutoffs = torch::empty(
      {query_count}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
  auto top_ones = torch::empty_like(counts);
  int64_t* counts_data = counts.data_ptr<int64_t>();
  float* cutoffs_data = cutoffs.data_ptr<float>();
  int64_t* top_ones_data = top_ones.data_ptr<int64_t>();

#pragma omp parallel for schedule(static)
  for (int64_t query = 0; query < query_count; ++query) {
    const float* row = logits_data + query * bucket_count;
    float maximum = row[0];
    int64_t top_one = 0;
    for (int64_t bucket = 1; bucket < bucket_count; ++bucket) {
      if (row[bucket] > maximum) {
        maximum = row[bucket];
        top_one = bucket;
      }
    }

    float weighted_sum = 0.0F;
#pragma omp simd reduction(+ : weighted_sum)
    for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
      weighted_sum += std::exp(row[bucket] - maximum) * sizes_data[bucket];
    }
    const float cutoff = maximum + std::log(weighted_sum) + log_ratio;
    int64_t qualifying_count = 0;
#pragma omp simd reduction(+ : qualifying_count)
    for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
      qualifying_count += row[bucket] > cutoff;
    }
    counts_data[query] = std::max<int64_t>(
        1, std::min<int64_t>(candidate_bound, qualifying_count));
    cutoffs_data[query] = cutoff;
    top_ones_data[query] = top_one;
  }

  int64_t route_width = 1;
  for (int64_t query = 0; query < query_count; ++query) {
    route_width = std::max(route_width, counts_data[query]);
  }
  auto selected = torch::full(
      {query_count, route_width},
      -1,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  int64_t* selected_data = selected.data_ptr<int64_t>();

#pragma omp parallel for schedule(static)
  for (int64_t query = 0; query < query_count; ++query) {
    const float* row = logits_data + query * bucket_count;
    int64_t* output = selected_data + query * route_width;
    const float cutoff = cutoffs_data[query];
    std::vector<int64_t> qualifying;
    qualifying.reserve(static_cast<size_t>(counts_data[query]));
    for (int64_t bucket = 0; bucket < bucket_count; ++bucket) {
      if (row[bucket] > cutoff) {
        qualifying.push_back(bucket);
      }
    }
    if (qualifying.empty()) {
      output[0] = top_ones_data[query];
      continue;
    }

    const auto logit_order = [row](int64_t left, int64_t right) {
      if (row[left] != row[right]) {
        return row[left] > row[right];
      }
      return left < right;
    };
    const bool truncated =
        static_cast<int64_t>(qualifying.size()) > candidate_bound;
    if (truncated) {
      std::partial_sort(
          qualifying.begin(),
          qualifying.begin() + candidate_bound,
          qualifying.end(),
          logit_order);
      qualifying.resize(static_cast<size_t>(candidate_bound));
    } else if (!bucket_id_order) {
      std::sort(qualifying.begin(), qualifying.end(), logit_order);
    }
    if (bucket_id_order && truncated) {
      std::sort(qualifying.begin(), qualifying.end());
    }
    std::copy(qualifying.begin(), qualifying.end(), output);
  }
  return {selected, counts};
}

template <bool ComputeNdis>
std::vector<torch::Tensor> merge_padded_plan_batches(
    const std::vector<torch::Tensor>& route_batches,
    const std::vector<torch::Tensor>& count_batches,
    const torch::Tensor& list_sizes) {
  TORCH_CHECK(!route_batches.empty(), "route batches must not be empty");
  TORCH_CHECK(
      route_batches.size() == count_batches.size(),
      "route and count batch counts differ");
  TORCH_CHECK(
      list_sizes.device().is_cpu() && list_sizes.scalar_type() == torch::kInt64 &&
          list_sizes.dim() == 1 && list_sizes.is_contiguous(),
      "list_sizes must be a contiguous CPU int64 vector");

  int64_t total_queries = 0;
  int64_t route_width = 1;
  for (size_t batch = 0; batch < route_batches.size(); ++batch) {
    const auto& routes = route_batches[batch];
    const auto& counts = count_batches[batch];
    TORCH_CHECK(
        routes.device().is_cpu() && routes.scalar_type() == torch::kInt64 &&
            routes.dim() == 2 && routes.is_contiguous(),
        "route batches must be contiguous CPU int64 matrices");
    TORCH_CHECK(
        counts.device().is_cpu() && counts.scalar_type() == torch::kInt64 &&
            counts.dim() == 1 && counts.is_contiguous(),
        "count batches must be contiguous CPU int64 vectors");
    TORCH_CHECK(routes.size(0) == counts.size(0), "route/count rows differ");
    total_queries += counts.size(0);
    route_width = std::max(route_width, routes.size(1));
  }

  auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
  auto merged_routes = torch::empty({total_queries, route_width}, options);
  auto merged_counts = torch::empty({total_queries}, options);
  auto ndis = ComputeNdis ? torch::empty({total_queries}, options) : torch::Tensor{};
  int64_t* merged_routes_data = merged_routes.data_ptr<int64_t>();
  int64_t* merged_counts_data = merged_counts.data_ptr<int64_t>();
  int64_t* ndis_data = ComputeNdis ? ndis.data_ptr<int64_t>() : nullptr;
  const int64_t* sizes_data = list_sizes.data_ptr<int64_t>();
  const int64_t bucket_count = list_sizes.size(0);
  std::atomic<int> invalid{0};

  int64_t output_offset = 0;
  for (size_t batch = 0; batch < route_batches.size(); ++batch) {
    const auto& routes = route_batches[batch];
    const auto& counts = count_batches[batch];
    const int64_t* routes_data = routes.data_ptr<int64_t>();
    const int64_t* counts_data = counts.data_ptr<int64_t>();
    const int64_t batch_queries = counts.size(0);
    const int64_t batch_width = routes.size(1);
#pragma omp parallel for schedule(static)
    for (int64_t query = 0; query < batch_queries; ++query) {
      const int64_t count = counts_data[query];
      if (count <= 0 || count > batch_width) {
        invalid.store(1, std::memory_order_relaxed);
        continue;
      }
      const int64_t* source = routes_data + query * batch_width;
      int64_t* destination =
          merged_routes_data + (output_offset + query) * route_width;
      int64_t query_ndis = 0;
      for (int64_t position = 0; position < count; ++position) {
        const int64_t bucket = source[position];
        if (bucket < 0 || bucket >= bucket_count) {
          invalid.store(2, std::memory_order_relaxed);
          break;
        }
        destination[position] = bucket;
        if constexpr (ComputeNdis) query_ndis += sizes_data[bucket];
      }
      std::fill(destination + count, destination + route_width, int64_t{-1});
      merged_counts_data[output_offset + query] = count;
      if constexpr (ComputeNdis) ndis_data[output_offset + query] = query_ndis;
    }
    output_offset += batch_queries;
  }
  TORCH_CHECK(invalid.load() == 0, "Gain-Cost plan batch is invalid");
  if constexpr (ComputeNdis) return {merged_routes, merged_counts, ndis};
  return {merged_routes, merged_counts};
}

torch::Tensor padded_plan_ndis(
    const torch::Tensor& routes, const torch::Tensor& counts,
    const torch::Tensor& list_sizes) {
  for (const auto& input : {routes, counts, list_sizes}) {
    TORCH_CHECK(input.device().is_cpu() && input.scalar_type() == torch::kInt64
                    && input.is_contiguous(),
                "NDis inputs must be contiguous CPU int64 tensors");
  }
  TORCH_CHECK(routes.dim() == 2 && counts.dim() == 1 && list_sizes.dim() == 1
                  && routes.size(0) == counts.size(0),
              "NDis input shapes disagree");
  auto ndis = torch::empty_like(counts);
  const auto* route_data = routes.data_ptr<int64_t>();
  const auto* count_data = counts.data_ptr<int64_t>();
  const auto* sizes_data = list_sizes.data_ptr<int64_t>();
  auto* output = ndis.data_ptr<int64_t>();
  std::atomic<bool> invalid{false};
#pragma omp parallel for schedule(static) num_threads(at::get_num_threads())
  for (int64_t row = 0; row < routes.size(0); ++row) {
    const auto count = count_data[row];
    if (count < 1 || count > routes.size(1)) {
      invalid.store(true, std::memory_order_relaxed);
      continue;
    }
    int64_t total = 0;
    for (int64_t position = 0; position < count; ++position) {
      const auto bucket = route_data[row * routes.size(1) + position];
      if (bucket < 0 || bucket >= list_sizes.size(0)) {
        invalid.store(true, std::memory_order_relaxed);
        break;
      }
      total += sizes_data[bucket];
    }
    output[row] = total;
  }
  TORCH_CHECK(!invalid.load(), "NDis plan is invalid");
  return ndis;
}

}  // namespace


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<ft_anns::PositiveMarginalWorkspace>(module, "PositiveMarginalWorkspace")
      .def(pybind11::init<int64_t, int64_t, int, bool>(), pybind11::arg("buckets"),
           pybind11::arg("batch_capacity"), pybind11::arg("scratch_bits"), pybind11::arg("vectorized") = true)
      .def_property_readonly("allocated_bytes", &ft_anns::PositiveMarginalWorkspace::allocated_bytes)
      .def("cancel_collection", [](ft_anns::PositiveMarginalWorkspace& workspace) {
        workspace.collecting = false;
        workspace.collected_rows = 0;
      })
      .def_readonly("scan_backend", &ft_anns::PositiveMarginalWorkspace::scan_backend)
      .def_readonly("scratch_bits", &ft_anns::PositiveMarginalWorkspace::scratch_bits);
  module.def("positive_marginal_plan_cpu", &ft_anns::positive_marginal_plan_cpu,
             "Single predicate pass with reusable uint32/int64 IDs and owned padded int64 output",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("positive_marginal_begin", &ft_anns::positive_marginal_begin,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("positive_marginal_append", &ft_anns::positive_marginal_append,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("positive_marginal_finish", &ft_anns::positive_marginal_finish,
             pybind11::call_guard<pybind11::gil_scoped_release>());
  pybind11::class_<PrefixPlannerWorkspace>(module, "PrefixPlannerWorkspace")
      .def(pybind11::init<int64_t, int64_t, bool, bool>(),
           pybind11::arg("buckets"), pybind11::arg("batch_capacity"),
           pybind11::arg("partial_softmax"), pybind11::arg("cache_order_keys") = true)
      .def_property_readonly("allocated_bytes", &PrefixPlannerWorkspace::allocated_bytes)
      .def_readonly("thread_capacity", &PrefixPlannerWorkspace::thread_capacity)
      .def_readonly("cached_order_keys", &PrefixPlannerWorkspace::cached_order_keys)
      .def_property_readonly("partial_softmax", [](const PrefixPlannerWorkspace& workspace) {
        return workspace.exp_row != nullptr;
      })
      .def_readonly("initial_bytes", &PrefixPlannerWorkspace::initial_bytes);
  module.def("probability_prefix_plan_from_mass_cpu", &probability_prefix_plan_from_mass_cpu,
             "Stable selection and exact prefix scan with only compact owned outputs",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("prefix_probability_backend", &prefix_probability_backend);
  module.def("probability_prefix_plan_cpu", &probability_prefix_plan_cpu,
             "Global normalization and stable prefix plan using reusable row scratch",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("prefix_candidate_probabilities_cpu", &prefix_candidate_probabilities_cpu,
             "Diagnostic candidate probabilities from the same row kernel as the fused planner",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("sorted_top_m_cpu", &sorted_top_m_cpu,
             "Select with std::nth_element, then sort only the stable Top-M candidates",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("profile_sorted_top_m_cpu", &profile_sorted_top_m_cpu,
             "Diagnostic stable selector: per-row init/nth/sort/copy worker seconds",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("sorted_top_m_int32_scratch_cpu", &sorted_top_m_int32_scratch_cpu,
             "Experimental int32 selector scratch with unchanged int64 output",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("profile_sorted_top_m_int32_scratch_cpu", &profile_sorted_top_m_int32_scratch_cpu,
             "Diagnostic int32 selector scratch with per-row phase times",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("profile_positive_marginal_filter_float32", &profile_positive_marginal_filter_float32,
             "Original full filter with per-query and parallel-region timing",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("profile_unsorted_top_m_float32", &profile_unsorted_top_m_float32,
             "Original Top-M selector with initialization/selection/copy timing",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("profile_log_normalizers_float32", &profile_log_normalizers_float32,
             "Original log normalization with maximum/exponential timing",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("profile_top_m_positive_marginal_float32", &profile_top_m_positive_marginal_float32,
             "Original candidate filter with count/compaction timing",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("collect_positive_buckets_float32", &collect_positive_buckets_float32,
             "Compact full-partition positives and record worst stable score",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("positive_worst_global_ranks_float32", &positive_worst_global_ranks_float32,
             "Count global ranks without selecting or sorting Top-M",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("restrict_positive_top_m_float32", &restrict_positive_top_m_float32,
             "Select global Top-M only for failed positive-set certificates",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("unsorted_top_m_float32", &unsorted_top_m_float32,
             "Stable Top-M set using unordered partition selection",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("log_normalizers_float32", &log_normalizers_float32,
             "All-bucket log-sum-exp without a full probability matrix",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("top_m_positive_marginal_float32", &top_m_positive_marginal_float32,
             "Filter and compact only Top-M candidates using cached thresholds",
             pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "probability_prefix_counts_float32",
      &probability_prefix_counts_float32,
      "Fused CPU probability prefix scan using cached bucket costs",
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "positive_marginal_filter_float32",
      &positive_marginal_filter_float32,
      "Fused CPU probability positive-marginal filter",
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "marginal_threshold_float32",
      &marginal_threshold_float32,
      "Fused CPU float32 Gain-Cost marginal threshold planner");
  module.def(
      "merge_padded_plan_batches",
      &merge_padded_plan_batches<true>,
      "Merge Gain-Cost plan batches and compute exact NDis");
  module.def(
      "pack_padded_plan_batches", &merge_padded_plan_batches<false>,
      "Pack padded routes without NDis statistics",
      pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def(
      "padded_plan_ndis", &padded_plan_ndis,
      "Compute exact per-query NDis after retrieval",
      pybind11::call_guard<pybind11::gil_scoped_release>());
}
