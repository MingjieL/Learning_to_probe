"""Build Gain-Cost Probe Plans behind one retrieval-facing interface."""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Literal

import numpy as np
import torch

from ft_anns.partition import ProbePlan
from ft_anns.partition.artifacts import PreparedPaddedProbePlan

try:
    _gain_cost_native: Any = importlib.import_module(
        "ft_anns.adaptive_query._gain_cost_native"
    )
except (ImportError, OSError):
    _gain_cost_native = None


LEGACY_GAIN_COST_PARAMETERS_FORMAT = "global-gain-cost-prefix-parameters-prototype-v4"
GAIN_COST_PARAMETERS_FORMAT = "global-gain-cost-prefix-parameters-prototype-v5"

PROBABILITY_GAIN_COST_PARAMETERS_FORMAT = (
    "global-gain-cost-prefix-parameters-prototype-v6"
)
GAIN_MODES = ("size_weighted", "probability")
PROBABILITY_PREFIX_CANDIDATE_MINIMUM = 256

ProbePlanMethod = Literal[
    "certified_short_prefix",
    "marginal_threshold",
    "full_prefix",
    "positive_marginal_filter",
    "top_m_positive_marginal_filter",
    "certified_top_m_positive_marginal_filter",
]
RouteOrder = Literal["stable_logit", "bucket_id", "unsorted_top_m"]

PROBE_PLAN_METHODS: tuple[ProbePlanMethod, ...] = (
    "certified_short_prefix",
    "marginal_threshold",
    "full_prefix",
    "positive_marginal_filter",
    "top_m_positive_marginal_filter",
    "certified_top_m_positive_marginal_filter",
)
ROUTE_ORDERS: tuple[RouteOrder, ...] = ("stable_logit", "bucket_id", "unsorted_top_m")

# Conservative legacy fast-path defaults. They are implementation details, not
# learned policy semantics.
_GAIN_COST_FAST_MIN_WIDTH = 32
_GAIN_COST_FAST_TRAINING_MULTIPLIER = 2
_GAIN_COST_CERTIFICATION_EPS_MULTIPLIER = 32


@dataclass(frozen=True)
class GainCostRetrievalSettings:
    """Frozen execution semantics carried by a Gain-Cost parameter artifact."""

    probe_plan_method: ProbePlanMethod
    route_order: RouteOrder
    gain_mode: str = "size_weighted"

    def __post_init__(self) -> None:
        if self.gain_mode not in GAIN_MODES:
            raise ValueError("unsupported Gain-Cost gain mode")
        if self.probe_plan_method in (
            "full_prefix", "positive_marginal_filter", "top_m_positive_marginal_filter",
            "certified_top_m_positive_marginal_filter",
        ):
            if self.gain_mode != "probability":
                raise ValueError(f"{self.probe_plan_method} requires probability gain")
        if self.gain_mode == "probability":
            valid_probability_pair = (
                self.probe_plan_method == "full_prefix"
                and self.route_order == "stable_logit"
            ) or (
                self.probe_plan_method == "positive_marginal_filter"
                and self.route_order == "bucket_id"
            ) or (
                self.probe_plan_method == "top_m_positive_marginal_filter"
                and self.route_order == "unsorted_top_m"
            ) or (
                self.probe_plan_method == "certified_top_m_positive_marginal_filter"
                and self.route_order == "bucket_id"
            )
            if not valid_probability_pair:
                raise ValueError(
                    "probability gain requires full_prefix/stable_logit or "
                    "positive_marginal_filter/bucket_id or "
                    "top_m_positive_marginal_filter/unsorted_top_m or "
                    "certified_top_m_positive_marginal_filter/bucket_id"
                )
        if self.route_order == "unsorted_top_m" and (
            self.probe_plan_method != "top_m_positive_marginal_filter"
        ):
            raise ValueError("unsorted_top_m requires top_m_positive_marginal_filter")
        if self.probe_plan_method not in PROBE_PLAN_METHODS:
            raise ValueError(
                f"unsupported Gain-Cost probe plan method: {self.probe_plan_method!r}"
            )
        if self.route_order not in ROUTE_ORDERS:
            raise ValueError(f"unsupported Gain-Cost route order: {self.route_order!r}")

    def as_dict(self) -> dict[str, str]:
        result = {
            "probe_plan_method": self.probe_plan_method,
            "route_order": self.route_order,
        }
        if self.gain_mode != "size_weighted":
            result["gain_mode"] = self.gain_mode
        return result


LEGACY_RETRIEVAL_SETTINGS = GainCostRetrievalSettings(
    probe_plan_method="certified_short_prefix",
    route_order="stable_logit",
)
DEFAULT_RETRIEVAL_SETTINGS = GainCostRetrievalSettings(
    probe_plan_method="marginal_threshold",
    route_order="stable_logit",
)


PROBABILITY_RETRIEVAL_SETTINGS = GainCostRetrievalSettings(
    probe_plan_method="full_prefix", route_order="stable_logit", gain_mode="probability"
)

POSITIVE_MARGINAL_RETRIEVAL_SETTINGS = GainCostRetrievalSettings(
    probe_plan_method="positive_marginal_filter",
    route_order="bucket_id",
    gain_mode="probability",
)

TOP_M_POSITIVE_MARGINAL_RETRIEVAL_SETTINGS = GainCostRetrievalSettings(
    probe_plan_method="top_m_positive_marginal_filter",
    route_order="unsorted_top_m",
    gain_mode="probability",
)

CERTIFIED_TOP_M_POSITIVE_MARGINAL_RETRIEVAL_SETTINGS = GainCostRetrievalSettings(
    probe_plan_method="certified_top_m_positive_marginal_filter",
    route_order="bucket_id",
    gain_mode="probability",
)


def validate_gain_mode(gain_mode: str, gain_power: float, cost_power: float) -> None:
    if gain_mode not in GAIN_MODES:
        raise ValueError("unsupported Gain-Cost gain mode")
    if any(not math.isfinite(power) or power <= 0 for power in (gain_power, cost_power)):
        raise ValueError("gain/cost powers must be finite and positive")


def parse_gain_cost_retrieval_settings(
    payload: Mapping[str, Any],
) -> GainCostRetrievalSettings:
    """Resolve legacy mass artifacts or validate explicit probability semantics."""
    format_name = payload.get("format")
    if format_name == LEGACY_GAIN_COST_PARAMETERS_FORMAT:
        if (
            payload.get("cache_metadata", {}).get("gain_mode", "size_weighted")
            != "size_weighted"
            or payload.get("retrieval", {}).get("gain_mode", "size_weighted")
            != "size_weighted"
        ):
            raise ValueError("legacy parameters require size_weighted gain")
        return LEGACY_RETRIEVAL_SETTINGS
    if format_name not in (
        GAIN_COST_PARAMETERS_FORMAT,
        PROBABILITY_GAIN_COST_PARAMETERS_FORMAT,
    ):
        raise ValueError("unsupported Gain-Cost parameter format")

    retrieval = payload.get("retrieval")
    if not isinstance(retrieval, Mapping):
        raise TypeError("Gain-Cost parameters are missing retrieval settings")
    method = retrieval.get("probe_plan_method")
    route_order = retrieval.get("route_order")
    if method not in PROBE_PLAN_METHODS:
        raise ValueError(f"unsupported Gain-Cost probe plan method: {method!r}")
    if route_order not in ROUTE_ORDERS:
        raise ValueError(f"unsupported Gain-Cost route order: {route_order!r}")
    gain_mode = retrieval.get("gain_mode", "size_weighted")
    expected_mode = (
        "probability"
        if format_name == PROBABILITY_GAIN_COST_PARAMETERS_FORMAT
        else "size_weighted"
    )
    metadata = payload.get("cache_metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Gain-Cost parameters contain invalid cache metadata")
    if (
        gain_mode != expected_mode
        or metadata.get("gain_mode", "size_weighted") != expected_mode
    ):
        raise ValueError("Gain-Cost format, retrieval and cache gain modes disagree")
    validate_gain_mode(
        gain_mode,
        float(metadata.get("gain_power_before_prefix_sum", 1.0)),
        float(metadata.get("cost_power_before_prefix_sum", 1.0)),
    )
    return GainCostRetrievalSettings(
        probe_plan_method=method,
        route_order=route_order,
        gain_mode=gain_mode,
    )


@dataclass(frozen=True)
class PrefixCandidateLimit:
    """Hard retrieval-only limit on a frozen Probability policy's candidates.

    Ratios accept exact fractions such as ``2/3``, round up and have a minimum
    of 256 candidates. Both modes are capped by the trained bound. Neither
    mode certifies the tail or falls back to the trained bound.
    """

    ratio: str | None = None
    count: int | None = None

    def __post_init__(self) -> None:
        if self.ratio is not None and self.count is not None:
            raise ValueError("prefix candidate ratio and count are mutually exclusive")
        if self.ratio is not None:
            try:
                ratio = Fraction(str(self.ratio))
            except (ValueError, ZeroDivisionError) as error:
                raise ValueError("prefix candidate ratio must satisfy 0 < ratio <= 1") from error
            if not 0 < ratio <= 1:
                raise ValueError("prefix candidate ratio must satisfy 0 < ratio <= 1")
            object.__setattr__(self, "ratio", str(ratio))
        if self.count is not None and (
            isinstance(self.count, bool)
            or not isinstance(self.count, int)
            or self.count < 1
        ):
            raise ValueError("prefix candidate count must be a positive integer")

    def resolve(self, trained_bound: int) -> int:
        if trained_bound < 1:
            raise ValueError("trained candidate bound must be positive")
        if self.ratio is not None:
            ratio = Fraction(self.ratio)
            scaled = (trained_bound * ratio.numerator + ratio.denominator - 1) // ratio.denominator
            return min(trained_bound, max(PROBABILITY_PREFIX_CANDIDATE_MINIMUM, scaled))
        if self.count is not None:
            return min(trained_bound, self.count)
        return trained_bound

    def coordinates(self, trained_bound: int) -> dict[str, Any]:
        effective = self.resolve(trained_bound)
        return {
            "trained_candidate_bound": trained_bound,
            "candidate_bound": effective,
            "prefix_candidate_ratio": self.ratio,
            "prefix_candidate_minimum": (
                PROBABILITY_PREFIX_CANDIDATE_MINIMUM if self.ratio is not None else None
            ),
            "prefix_candidate_limit": self.count,
            "candidate_scope": (
                "truncated_candidate_bound" if effective < trained_bound
                else "trained_candidate_bound"
            ),
        }


class GainCostProbePlanner:
    """Compile one frozen policy, then build Probe Plans for logits batches.

    ``marginal_threshold`` implements the gain_power=cost_power=1 stopping rule
    with a max-shifted weighted exponential reduction. It requires non-empty
    partition lists because removing zero-cost buckets would otherwise change
    raw-prefix semantics.

    ``positive_marginal_filter`` is a full-partition, non-prefix Probability
    adapter. It applies an independent strict utility test to every bucket and
    emits the selected set in bucket-ID order without applying candidate_bound.

    ``top_m_positive_marginal_filter`` applies that strict test only within the
    stable Top-M set. It uses global Probability normalization, retains the
    selector's unsorted order, and falls back to the stable highest-logit bucket.

    ``certified_top_m_positive_marginal_filter`` first filters the complete
    partition and certifies the global rank of its worst positive bucket. Only
    failing queries need Top-M selection. It retains the same bounded bucket
    set, emitting routes in bucket-ID order on both certificate paths.
    """

    def __init__(
        self,
        *,
        normalized_sizes: torch.Tensor,
        policy: Mapping[str, Any],
        gain_power: float,
        cost_power: float,
        settings: GainCostRetrievalSettings,
        prefix_candidate_limit: PrefixCandidateLimit | None = None,
    ) -> None:
        validate_gain_mode(settings.gain_mode, gain_power, cost_power)
        if normalized_sizes.ndim != 1 or not normalized_sizes.is_floating_point():
            raise ValueError("normalized_sizes must be a floating-point vector")
        if not bool(torch.isfinite(normalized_sizes).all()):
            raise ValueError("normalized_sizes must be finite")
        if not bool(normalized_sizes.ge(0).all()):
            raise ValueError("normalized_sizes must be non-negative")

        candidate_bound = int(policy["candidate_bound"])
        alpha = float(policy["alpha"])
        beta = float(policy["beta"])
        if (
            candidate_bound < 1
            or candidate_bound > len(normalized_sizes)
            or not math.isfinite(alpha)
            or not math.isfinite(beta)
            or alpha < 0
            or beta < 0
        ):
            raise ValueError("Gain-Cost policy has invalid coefficients")
        if (
            prefix_candidate_limit is not None
            and settings != PROBABILITY_RETRIEVAL_SETTINGS
        ):
            raise ValueError("prefix candidate limits require Probability full_prefix retrieval")
        if gain_power <= 0 or cost_power <= 0:
            raise ValueError("Gain-Cost powers must be positive")
        if settings.probe_plan_method == "marginal_threshold":
            if gain_power != 1.0 or cost_power != 1.0:
                raise ValueError(
                    "marginal_threshold requires gain_power=1 and cost_power=1"
                )
            if not bool(normalized_sizes.gt(0).all()):
                raise ValueError(
                    "marginal_threshold requires every partition list to be non-empty"
                )

        # Partition costs are query-independent: keep one owned table for the
        # planner lifetime. Only bucket-ID lookups belong in the query path.
        self._normalized_sizes = normalized_sizes.detach().clone().contiguous()
        self._policy = dict(policy)
        self._trained_candidate_bound = candidate_bound
        self._prefix_candidate_limit = prefix_candidate_limit or PrefixCandidateLimit()
        self._candidate_bound = self._prefix_candidate_limit.resolve(candidate_bound)
        self._alpha = alpha
        self._beta = beta
        self._gain_power = float(gain_power)
        self._cost_power = float(cost_power)
        self.settings = settings
        self._positive_log_thresholds = torch.zeros_like(self._normalized_sizes)
        if (
            settings.probe_plan_method in (
                "positive_marginal_filter", "top_m_positive_marginal_filter",
                "certified_top_m_positive_marginal_filter",
            )
            and alpha > 0
            and beta > 0
        ):
            positive_costs = self._normalized_sizes.gt(0)
            self._positive_log_thresholds.fill_(-torch.inf)
            self._positive_log_thresholds[positive_costs] = (
                self._normalized_sizes[positive_costs].log_().mul_(self._cost_power)
                + math.log(beta)
                - math.log(alpha)
            )
        self._log_beta_over_alpha = (
            math.log(beta) - math.log(alpha)
            if settings.probe_plan_method == "marginal_threshold"
            and alpha > 0
            and beta > 0
            else None
        )

    @property
    def candidate_bound(self) -> int:
        """The effective candidate bound used before Top-K selection."""
        return self._candidate_bound

    @property
    def prefix_candidate_coordinates(self) -> dict[str, Any]:
        return {
            **self._prefix_candidate_limit.coordinates(self._trained_candidate_bound),
            "candidate_selection_implementation": self.candidate_selection_implementation,
        }

    @property
    def candidate_selection_implementation(self) -> str:
        """Ordered candidate selector for the planner's configured CPU dtype."""
        if (
            self._normalized_sizes.device.type == "cpu"
            and self._normalized_sizes.dtype in (torch.float32, torch.float64)
            and hasattr(_gain_cost_native, "sorted_top_m_cpu")
        ):
            return "std_nth_element_then_sort"
        return "torch_topk"

    def build(self, logits: torch.Tensor) -> ProbePlan:
        """Return compact active routes for one logits batch."""
        if logits.ndim != 2 or logits.shape[1] != len(self._normalized_sizes):
            raise ValueError(
                "logits must be a matrix with one column per partition bucket"
            )
        if not logits.is_floating_point():
            raise ValueError("logits must use a floating-point dtype")
        if logits.device != self._normalized_sizes.device:
            raise ValueError("logits and normalized_sizes must use the same device")
        if len(logits) == 0:
            raise ValueError("logits must contain at least one query")

        if self.settings.probe_plan_method == "full_prefix":
            selected, probe_counts = self._build_full_prefix_tensors(logits)
            plan = _probe_plan(selected, probe_counts)
        elif self.settings.probe_plan_method == "positive_marginal_filter":
            selected, probe_counts = self._build_positive_marginal_tensors(logits)
            plan = _probe_plan(selected, probe_counts)
        elif self.settings.probe_plan_method == "top_m_positive_marginal_filter":
            selected, probe_counts = self._build_top_m_positive_marginal_tensors(logits)
            plan = _probe_plan(selected, probe_counts)
        elif self.settings.probe_plan_method == "certified_top_m_positive_marginal_filter":
            selected, probe_counts = self._build_certified_top_m_positive_marginal_tensors(logits)
            plan = _probe_plan(selected, probe_counts)
        elif self.settings.probe_plan_method == "certified_short_prefix":
            plan = self._build_certified_short_prefix(logits)
        else:
            selected, probe_counts = self._build_marginal_threshold_tensors(logits)
            plan = _probe_plan(selected, probe_counts)
        if self.settings.route_order == "bucket_id" and (
            self.settings.probe_plan_method == "certified_short_prefix"
        ):
            return _sort_active_routes_by_bucket_id(plan)
        return plan

    def build_tensors(
        self, logits: torch.Tensor, *, stage_seconds: dict[str, float] | None = None,
        probability_buffer: torch.Tensor | None = None,
        native_workspace: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build one plan batch without crossing the NumPy seam."""
        if self.settings.probe_plan_method not in (
            "marginal_threshold", "full_prefix", "positive_marginal_filter",
            "top_m_positive_marginal_filter",
            "certified_top_m_positive_marginal_filter",
        ):
            raise ValueError(
                "tensor plan batches require a threshold or full-prefix adapter"
            )
        if logits.ndim != 2 or logits.shape[1] != len(self._normalized_sizes):
            raise ValueError(
                "logits must be a matrix with one column per partition bucket"
            )
        if not logits.is_floating_point():
            raise ValueError("logits must use a floating-point dtype")
        if logits.device != self._normalized_sizes.device:
            raise ValueError("logits and normalized_sizes must use the same device")
        if len(logits) == 0:
            raise ValueError("logits must contain at least one query")
        if probability_buffer is not None and (
            self.settings != PROBABILITY_RETRIEVAL_SETTINGS
            or probability_buffer.shape != logits.shape
            or probability_buffer.dtype != logits.dtype
            or probability_buffer.device != logits.device
            or not probability_buffer.is_contiguous()
        ):
            raise ValueError("probability buffer must match Probability full-prefix logits")
        if self.settings.probe_plan_method == "full_prefix":
            return self._build_full_prefix_tensors(logits, probability_buffer=probability_buffer,
                                                   native_workspace=native_workspace)
        if self.settings.probe_plan_method == "positive_marginal_filter":
            return self._build_positive_marginal_tensors(logits, native_workspace=native_workspace)
        if self.settings.probe_plan_method == "top_m_positive_marginal_filter":
            return self._build_top_m_positive_marginal_tensors(
                logits, stage_seconds=stage_seconds
            )
        if self.settings.probe_plan_method == "certified_top_m_positive_marginal_filter":
            return self._build_certified_top_m_positive_marginal_tensors(
                logits, stage_seconds=stage_seconds
            )
        return self._build_marginal_threshold_tensors(logits)

    def profile_tensors(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Opt-in native diagnostics without changing the original loop layout.

        Worker query durations overlap and must not be summed as wall time.
        Normal build_tensors calls never read clocks or allocate timing arrays.
        """
        symbols = (
            "profile_unsorted_top_m_float32", "profile_log_normalizers_float32",
            "profile_top_m_positive_marginal_float32",
            "profile_positive_marginal_filter_float32",
        )
        if (
            self.settings.probe_plan_method not in (
                "positive_marginal_filter", "top_m_positive_marginal_filter"
            )
            or logits.device.type != "cpu" or logits.dtype != torch.float32
            or not logits.is_contiguous() or logits.ndim != 2 or not len(logits)
            or logits.shape[1] != len(self._normalized_sizes)
            or self._normalized_sizes.device.type != "cpu"
            or self._normalized_sizes.dtype != torch.float32
            or not all(hasattr(_gain_cost_native, name) for name in symbols)
        ):
            raise ValueError("profiling requires rebuilt native CPU float32 filter inputs")
        worker: dict[str, torch.Tensor] = {}
        shared: dict[str, float] = {}
        wall: dict[str, float] = {}

        def record(query_times, names, shared_times, shared_names):
            worker.update({name: query_times[:, i] for i, name in enumerate(names)})
            shared.update({name: float(shared_times[i]) for i, name in enumerate(shared_names)})

        if self.settings.probe_plan_method == "positive_marginal_filter":
            started = time.perf_counter()
            selected, counts, query_times, shared_times = (
                _gain_cost_native.profile_positive_marginal_filter_float32(
                    logits, self._positive_log_thresholds,
                    self._gain_power, self._alpha, self._beta,
                )
            )
            wall["full_filter_seconds"] = time.perf_counter() - started
            record(query_times,
                   ("global_max_top1", "global_exp_sum", "positive_count", "positive_rescan_write"),
                   shared_times,
                   ("filter_metadata_allocation", "max_normalize_count_parallel_region",
                    "route_width_and_allocation", "compaction_parallel_region"))
        else:
            started = time.perf_counter()
            candidates, query_times, shared_times = (
                _gain_cost_native.profile_unsorted_top_m_float32(logits, self._candidate_bound)
            )
            wall["candidate_selection_seconds"] = time.perf_counter() - started
            record(query_times, ("candidate_id_initialization", "candidate_nth_element", "candidate_copy"),
                   shared_times, ("candidate_output_allocation", "candidate_parallel_region"))
            started = time.perf_counter()
            if self._alpha == 0 or self._beta == 0:
                normalizers = torch.zeros(len(logits), dtype=logits.dtype)
                query_times = torch.zeros((len(logits), 2), dtype=torch.float64)
                shared_times = torch.zeros(2, dtype=torch.float64)
            else:
                normalizers, query_times, shared_times = (
                    _gain_cost_native.profile_log_normalizers_float32(logits)
                )
            wall["normalization_seconds"] = time.perf_counter() - started
            record(query_times, ("global_max", "global_exp_sum"), shared_times,
                   ("normalizer_output_allocation", "normalizer_parallel_region"))
            started = time.perf_counter()
            selected, counts, _, query_times, shared_times = (
                _gain_cost_native.profile_top_m_positive_marginal_float32(
                    logits, candidates, normalizers, self._positive_log_thresholds,
                    self._gain_power, self._alpha, self._beta,
                )
            )
            wall["filter_compaction_seconds"] = time.perf_counter() - started
            record(query_times, ("candidate_filter_count_top1", "candidate_compaction"),
                   shared_times, ("filter_metadata_allocation", "filter_parallel_region",
                                  "route_width_and_allocation", "compaction_parallel_region"))
        return selected, counts, {
            "worker_query_seconds": worker,
            "native_shared_wall_seconds": shared,
            "call_wall_seconds": wall,
        }

    def _build_certified_top_m_positive_marginal_tensors(
        self, logits: torch.Tensor, *, stage_seconds: dict[str, float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Certify P is inside the stable global Top-M; restrict only failures.

        Native normalization and utility arithmetic match the original Top-M
        filter. No probability or Q-by-M candidate matrix is materialized on
        the certified path. Empty full-partition P skips ranking and selection.
        """
        if stage_seconds is not None and logits.device.type != "cpu":
            raise ValueError("stage diagnostics require CPU logits")
        native = (
            self.implementation == "native_certified_top_m_positive_marginal_cpu_float32"
            and logits.dtype == torch.float32 and logits.is_contiguous()
        )
        started = time.perf_counter() if stage_seconds is not None else 0.0
        if self._alpha == 0 or self._beta == 0:
            normalizers = torch.zeros(len(logits), dtype=logits.dtype, device=logits.device)
        elif native:
            normalizers = _gain_cost_native.log_normalizers_float32(logits)
        else:
            normalizers = torch.logsumexp(logits, dim=1)
        if stage_seconds is not None:
            stage_seconds["normalization_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
        if native:
            positives, raw_counts, worst, top = _gain_cost_native.collect_positive_buckets_float32(
                logits, normalizers, self._positive_log_thresholds,
                self._gain_power, self._alpha, self._beta,
            )
        else:
            if self._alpha == 0:
                qualifies = torch.zeros_like(logits, dtype=torch.bool)
            elif self._beta == 0:
                qualifies = torch.ones_like(logits, dtype=torch.bool)
            else:
                qualifies = ((logits - normalizers[:, None]) * self._gain_power).gt(
                    self._positive_log_thresholds
                )
            raw_counts = qualifies.sum(1, dtype=torch.int64)
            top = logits.argmax(1)
            positives = torch.full(
                (len(logits), max(1, int(raw_counts.max().item()))), -1,
                dtype=torch.int64, device=logits.device,
            )
            worst = torch.full_like(raw_counts, -1)
            for query in range(len(logits)):
                ids = torch.nonzero(qualifies[query], as_tuple=True)[0]
                if len(ids):
                    positives[query, :len(ids)] = ids
                    values = logits[query, ids]
                    worst[query] = ids[values.eq(values.min())][-1]
                else:
                    positives[query, 0] = top[query]
        if stage_seconds is not None:
            stage_seconds["full_filter_compaction_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
        if native:
            ranks = _gain_cost_native.positive_worst_global_ranks_float32(logits, worst)
        else:
            ranks = torch.zeros_like(raw_counts)
            ids = torch.arange(logits.shape[1], device=logits.device)
            for query in range(len(logits)):
                bucket = int(worst[query].item())
                if bucket >= 0:
                    value = logits[query, bucket]
                    ranks[query] = 1 + (logits[query].gt(value) | (
                        logits[query].eq(value) & ids.lt(bucket)
                    )).sum()
        if stage_seconds is not None:
            stage_seconds["rank_certification_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
        if native:
            selected, counts, final_raw_counts = _gain_cost_native.restrict_positive_top_m_float32(
                logits, positives, raw_counts, ranks, top, self._candidate_bound,
                normalizers, self._positive_log_thresholds,
                self._gain_power, self._alpha, self._beta,
            )
        else:
            selected = torch.full(
                (len(logits), min(self._candidate_bound, positives.shape[1])), -1,
                dtype=torch.int64, device=logits.device,
            )
            final_raw_counts = raw_counts.clone()
            failed = torch.nonzero(ranks.gt(self._candidate_bound), as_tuple=True)[0]
            candidates = _unsorted_top_m_indices(logits[failed], self._candidate_bound) if len(failed) else None
            failed_position = 0
            for query in range(len(logits)):
                if int(ranks[query].item()) > self._candidate_bound:
                    candidate_ids = candidates[failed_position]
                    failed_position += 1
                    chosen = candidate_ids[qualifies[query, candidate_ids]].sort().values
                    final_raw_counts[query] = len(chosen)
                else:
                    chosen = positives[query, :int(raw_counts[query].item())]
                if len(chosen):
                    selected[query, :len(chosen)] = chosen
                else:
                    selected[query, 0] = top[query]
            counts = final_raw_counts.clamp_min(1)
        if stage_seconds is not None:
            stage_seconds["top_m_fallback_compaction_seconds"] = time.perf_counter() - started
            stage_seconds["empty_selection_count"] = int(final_raw_counts.eq(0).sum().item())
            stage_seconds["empty_positive_query_count"] = int(raw_counts.eq(0).sum().item())
            stage_seconds["certified_query_count"] = int((ranks.gt(0) & ranks.le(self._candidate_bound)).sum().item())
            stage_seconds["top_m_fallback_query_count"] = int(ranks.gt(self._candidate_bound).sum().item())
        return selected, counts

    def _build_top_m_positive_marginal_tensors(
        self, logits: torch.Tensor, *, stage_seconds: dict[str, float] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Filter the stable Top-M set, without ranking it or normalizing within M.

        Optional stage timing is for CPU diagnostic runs, separate from normal
        retrieval timing. Route order is the selector's unsorted candidate order.
        """
        if stage_seconds is not None and logits.device.type != "cpu":
            raise ValueError("stage diagnostics require CPU logits")
        native = (
            self.implementation == "native_top_m_positive_marginal_cpu_float32"
            and logits.dtype == torch.float32
            and logits.is_contiguous()
        )
        started = time.perf_counter() if stage_seconds is not None else 0.0
        if native:
            candidates = _gain_cost_native.unsorted_top_m_float32(
                logits, self._candidate_bound
            )
        else:
            candidates = _unsorted_top_m_indices(logits, self._candidate_bound)
        if stage_seconds is not None:
            stage_seconds["candidate_selection_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
        if self._alpha == 0 or self._beta == 0:
            normalizers = torch.zeros(
                len(logits), dtype=logits.dtype, device=logits.device
            )
        elif native:
            normalizers = _gain_cost_native.log_normalizers_float32(logits)
        else:
            normalizers = torch.logsumexp(logits, dim=1)
        if stage_seconds is not None:
            stage_seconds["normalization_seconds"] = time.perf_counter() - started
            started = time.perf_counter()
        if native:
            selected, counts, raw_counts = (
                _gain_cost_native.top_m_positive_marginal_float32(
                    logits, candidates, normalizers, self._positive_log_thresholds,
                    self._gain_power, self._alpha, self._beta,
                )
            )
        else:
            values = logits.gather(1, candidates)
            if self._alpha == 0:
                qualifies = torch.zeros_like(candidates, dtype=torch.bool)
            elif self._beta == 0:
                qualifies = torch.ones_like(candidates, dtype=torch.bool)
            else:
                qualifies = ((values - normalizers[:, None]) * self._gain_power).gt(
                    self._positive_log_thresholds[candidates]
                )
            raw_counts = qualifies.sum(dim=1, dtype=torch.int64)
            counts = raw_counts.clamp_min(1)
            selected = torch.full(
                (len(logits), int(counts.max().item())), -1,
                dtype=torch.int64, device=logits.device,
            )
            rows, positions = torch.nonzero(qualifies, as_tuple=True)
            compact_positions = qualifies.cumsum(dim=1)[rows, positions] - 1
            selected[rows, compact_positions] = candidates[rows, positions]
            empty = raw_counts.eq(0)
            if bool(empty.any()):
                # Torch argmax uses the earliest ID for ties; the global top
                # bucket is always inside the stable Top-M candidate set.
                selected[empty, 0] = logits[empty].argmax(dim=1)
        if stage_seconds is not None:
            stage_seconds["filter_compaction_seconds"] = time.perf_counter() - started
            stage_seconds["empty_selection_count"] = int(raw_counts.eq(0).sum().item())
        return selected, counts

    def _build_positive_marginal_tensors(
        self, logits: torch.Tensor, *, native_workspace: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select every positive-utility bucket in ascending bucket-ID order."""
        if (
            self.implementation == "fused_positive_marginal_cpu_float32"
            and logits.dtype == torch.float32
            and logits.is_contiguous()
        ):
            if native_workspace is not None:
                return _gain_cost_native.positive_marginal_plan_cpu(
                    logits, self._positive_log_thresholds, self._gain_power,
                    self._alpha, self._beta, native_workspace,
                )
            return _gain_cost_native.positive_marginal_filter_float32(
                logits,
                self._positive_log_thresholds,
                self._gain_power,
                self._alpha,
                self._beta,
            )

        if self._alpha == 0:
            counts = torch.ones(len(logits), dtype=torch.int64, device=logits.device)
            return _stable_topk_indices(logits, 1), counts
        if self._beta == 0:
            selected = torch.arange(
                logits.shape[1], dtype=torch.int64, device=logits.device
            ).expand(len(logits), -1)
            counts = torch.full(
                (len(logits),), logits.shape[1], dtype=torch.int64, device=logits.device
            )
            return selected, counts

        log_probabilities = logits - torch.logsumexp(logits, dim=1, keepdim=True)
        qualifies = log_probabilities.mul(self._gain_power).gt(
            self._positive_log_thresholds.unsqueeze(0)
        )
        qualifying_counts = qualifies.sum(dim=1, dtype=torch.int64)
        probe_counts = qualifying_counts.clamp_min(1)
        route_width = int(probe_counts.max().item())
        selected = torch.full(
            (len(logits), route_width),
            -1,
            dtype=torch.int64,
            device=logits.device,
        )
        zero_rows = torch.nonzero(qualifying_counts.eq(0), as_tuple=False).flatten()
        if zero_rows.numel():
            selected[zero_rows, 0] = _stable_topk_indices(
                logits[zero_rows], 1
            ).flatten()
        pairs = torch.nonzero(qualifies, as_tuple=False)
        if pairs.numel():
            pair_rows, pair_bucket_ids = pairs.unbind(dim=1)
            row_offsets = qualifying_counts.cumsum(dim=0) - qualifying_counts
            pair_positions = (
                torch.arange(len(pairs), dtype=torch.int64, device=logits.device)
                - row_offsets[pair_rows]
            )
            selected[pair_rows, pair_positions] = pair_bucket_ids
        return selected, probe_counts

    def _build_full_prefix_tensors(
        self, logits: torch.Tensor, *, probability_buffer: torch.Tensor | None = None,
        native_workspace: Any = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (native_workspace is not None and self.implementation == "fused_probability_cpu_float32"
                and logits.dtype == torch.float32 and logits.is_contiguous()):
            if native_workspace.partial_softmax:
                return _gain_cost_native.probability_prefix_plan_cpu(
                    logits, self._normalized_sizes, self._candidate_bound,
                    self._alpha, self._beta, native_workspace,
                )
            mass = torch.softmax(logits, dim=1, out=probability_buffer)
            return _gain_cost_native.probability_prefix_plan_from_mass_cpu(
                logits, mass, self._normalized_sizes, self._candidate_bound,
                self._alpha, self._beta, native_workspace,
            )
        selected = _stable_topk_indices(logits, self._candidate_bound)
        # Marginal signs can alternate: evaluate the entire candidate bound.
        mass = torch.softmax(logits, dim=1, out=probability_buffer)
        if (
            self.implementation == "fused_probability_cpu_float32"
            and logits.dtype == torch.float32
        ):
            counts = _gain_cost_native.probability_prefix_counts_float32(
                mass, self._normalized_sizes, selected, self._alpha, self._beta
            )
            return selected[:, : int(counts.max().item())], counts
        counts = _gain_cost_probe_counts(
            mass,
            normalized_sizes=self._normalized_sizes,
            selected=selected,
            alpha=self._alpha,
            beta=self._beta,
            gain_power=self._gain_power,
            cost_power=self._cost_power,
        )
        return selected[:, : int(counts.max().item())], counts

    @property
    def implementation(self) -> str:
        """Name the implementation selected by this installed environment."""
        if (
            self.settings.probe_plan_method == "certified_top_m_positive_marginal_filter"
            and all(hasattr(_gain_cost_native, name) for name in (
                "log_normalizers_float32", "collect_positive_buckets_float32",
                "positive_worst_global_ranks_float32", "restrict_positive_top_m_float32",
            ))
            and self._normalized_sizes.device.type == "cpu"
            and self._normalized_sizes.dtype == torch.float32
        ):
            return "native_certified_top_m_positive_marginal_cpu_float32"
        if (
            self.settings.probe_plan_method == "top_m_positive_marginal_filter"
            and all(hasattr(_gain_cost_native, name) for name in (
                "unsorted_top_m_float32", "log_normalizers_float32",
                "top_m_positive_marginal_float32",
            ))
            and self._normalized_sizes.device.type == "cpu"
            and self._normalized_sizes.dtype == torch.float32
        ):
            return "native_top_m_positive_marginal_cpu_float32"
        if (
            self.settings.probe_plan_method == "positive_marginal_filter"
            and hasattr(_gain_cost_native, "positive_marginal_filter_float32")
            and self._normalized_sizes.device.type == "cpu"
            and self._normalized_sizes.dtype == torch.float32
        ):
            return "fused_positive_marginal_cpu_float32"
        if (
            self.settings.probe_plan_method == "full_prefix"
            and self._gain_power == 1.0
            and self._cost_power == 1.0
            and hasattr(_gain_cost_native, "probability_prefix_counts_float32")
            and hasattr(_gain_cost_native, "sorted_top_m_cpu")
            and self._normalized_sizes.device.type == "cpu"
            and self._normalized_sizes.dtype == torch.float32
        ):
            return "fused_probability_cpu_float32"
        if (
            self.settings.probe_plan_method == "marginal_threshold"
            and _gain_cost_native is not None
            and self._normalized_sizes.device.type == "cpu"
            and self._normalized_sizes.dtype == torch.float32
            and self._alpha > 0
            and self._beta > 0
        ):
            return "fused_cpu_float32"
        return "torch"

    @property
    def prepared_implementation(self) -> str:
        """Name the full-batch prepared-plan implementation when available."""
        if self.implementation == "fused_cpu_float32" and hasattr(
            _gain_cost_native, "merge_padded_plan_batches"
        ):
            return "fused_cpu_float32_prepared"
        return f"{self.implementation}_prepared"

    def _build_certified_short_prefix(self, logits: torch.Tensor) -> ProbePlan:
        bound = self._candidate_bound
        fast_width = _gain_cost_fast_width(
            self._policy,
            candidate_bound=bound,
            gain_power=self._gain_power,
            cost_power=self._cost_power,
        )
        selected = _stable_topk_indices(logits, fast_width)
        mass = torch.softmax(logits, dim=1)
        last_probability = mass.gather(1, selected[:, -1:])
        mass.mul_(self._normalized_sizes.unsqueeze(0))
        mass_sum = mass.sum(dim=1, keepdim=True)
        mass.div_(mass_sum)
        probe_counts = _gain_cost_probe_counts(
            mass,
            normalized_sizes=self._normalized_sizes,
            selected=selected,
            alpha=self._alpha,
            beta=self._beta,
            gain_power=self._gain_power,
            cost_power=self._cost_power,
        )

        if fast_width < bound:
            certified = _certify_gain_cost_tail(
                last_probability,
                mass_sum=mass_sum,
                alpha=self._alpha,
                beta=self._beta,
            )
            unresolved_rows = torch.nonzero(~certified, as_tuple=False).flatten()
            if unresolved_rows.numel():
                full_selected = _stable_topk_indices(logits[unresolved_rows], bound)
                probe_counts[unresolved_rows] = _gain_cost_probe_counts(
                    mass[unresolved_rows],
                    normalized_sizes=self._normalized_sizes,
                    selected=full_selected,
                    alpha=self._alpha,
                    beta=self._beta,
                    gain_power=self._gain_power,
                    cost_power=self._cost_power,
                )

                route_width = int(probe_counts.max().item())
                merged = torch.full(
                    (len(logits), route_width),
                    -1,
                    dtype=selected.dtype,
                    device=selected.device,
                )
                short_width = min(route_width, fast_width)
                merged[:, :short_width] = selected[:, :short_width]
                merged[unresolved_rows] = full_selected[:, :route_width]
                selected = merged

        selected = selected[:, : int(probe_counts.max().item())]
        return _probe_plan(selected, probe_counts)

    def _build_marginal_threshold_tensors(
        self, logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._alpha == 0:
            probe_counts = torch.ones(
                len(logits), dtype=torch.int64, device=logits.device
            )
            return _stable_topk_indices(logits, 1), probe_counts

        if self._beta == 0:
            selected = _stable_topk_indices(logits, self._candidate_bound)
            if self.settings.route_order == "bucket_id":
                selected = torch.sort(selected, dim=1).values
            probe_counts = torch.full(
                (len(logits),),
                self._candidate_bound,
                dtype=torch.int64,
                device=logits.device,
            )
            return selected, probe_counts

        if (
            _gain_cost_native is not None
            and logits.device.type == "cpu"
            and logits.dtype == torch.float32
            and logits.is_contiguous()
            and self._normalized_sizes.dtype == torch.float32
        ):
            selected, probe_counts = _gain_cost_native.marginal_threshold_float32(
                logits,
                self._normalized_sizes,
                self._candidate_bound,
                self._alpha,
                self._beta,
                self.settings.route_order == "bucket_id",
            )
            return selected, probe_counts

        assert self._log_beta_over_alpha is not None
        row_maximum = logits.amax(dim=1, keepdim=True)
        shifted_weights = logits.sub(row_maximum)
        shifted_weights.exp_()
        shifted_weighted_sum = torch.mv(shifted_weights, self._normalized_sizes)
        cutoff = row_maximum.flatten().add(
            shifted_weighted_sum.log().add_(self._log_beta_over_alpha)
        )
        qualifies = logits.gt(cutoff.unsqueeze(1))
        qualifying_counts = qualifies.sum(dim=1, dtype=torch.int64)
        probe_counts = qualifying_counts.clamp(min=1, max=self._candidate_bound)

        if self.settings.route_order == "stable_logit":
            selected = _stable_topk_indices(logits, int(probe_counts.max().item()))
        else:
            selected = self._compact_bucket_id_routes(
                logits, qualifies, qualifying_counts, probe_counts
            )
        return selected, probe_counts

    def _compact_bucket_id_routes(
        self,
        logits: torch.Tensor,
        qualifies: torch.Tensor,
        qualifying_counts: torch.Tensor,
        probe_counts: torch.Tensor,
    ) -> torch.Tensor:
        """Compact qualifying IDs without sorting the common in-bound case."""
        route_width = int(probe_counts.max().item())
        selected = torch.full(
            (len(logits), route_width),
            -1,
            dtype=torch.int64,
            device=logits.device,
        )
        zero_rows = torch.nonzero(qualifying_counts.eq(0), as_tuple=False).flatten()
        if zero_rows.numel():
            selected[zero_rows, 0] = _stable_topk_indices(
                logits[zero_rows], 1
            ).flatten()

        pairs = torch.nonzero(qualifies, as_tuple=False)
        if pairs.numel():
            pair_rows, pair_bucket_ids = pairs.unbind(dim=1)
            row_offsets = qualifying_counts.cumsum(dim=0) - qualifying_counts
            pair_positions = (
                torch.arange(len(pairs), dtype=torch.int64, device=logits.device)
                - row_offsets[pair_rows]
            )
            in_bound = qualifying_counts[pair_rows].le(self._candidate_bound)
            selected[pair_rows[in_bound], pair_positions[in_bound]] = pair_bucket_ids[
                in_bound
            ]

        overflow_rows = torch.nonzero(
            qualifying_counts.gt(self._candidate_bound), as_tuple=False
        ).flatten()
        if overflow_rows.numel():
            bounded = _stable_topk_indices(logits[overflow_rows], self._candidate_bound)
            selected[overflow_rows, : self._candidate_bound] = torch.sort(
                bounded, dim=1
            ).values
        return selected


def _probe_plan(selected: torch.Tensor, probe_counts: torch.Tensor) -> ProbePlan:
    return ProbePlan(
        bucket_ids=selected.cpu().numpy().astype(np.int64, copy=False),
        probe_counts=probe_counts.cpu().numpy().astype(np.int64, copy=False),
    )


def merge_gain_cost_plan_batches(
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    *,
    list_sizes: torch.Tensor,
) -> PreparedPaddedProbePlan:
    """Merge compact native batches and prepare one full padded FAISS input."""
    if not batches:
        raise ValueError("Gain-Cost plan batches must not be empty")
    if (
        list_sizes.ndim != 1
        or list_sizes.dtype != torch.int64
        or list_sizes.device.type != "cpu"
    ):
        raise ValueError("list_sizes must be a CPU int64 vector")
    list_sizes = list_sizes.contiguous()
    routes = [batch[0].contiguous() for batch in batches]
    counts = [batch[1].contiguous() for batch in batches]
    if _gain_cost_native is not None and hasattr(
        _gain_cost_native, "merge_padded_plan_batches"
    ):
        merged_routes, merged_counts, ndis = (
            _gain_cost_native.merge_padded_plan_batches(routes, counts, list_sizes)
        )
    else:
        if any(
            route.ndim != 2
            or count.ndim != 1
            or len(route) != len(count)
            or route.dtype != torch.int64
            or count.dtype != torch.int64
            or route.device.type != "cpu"
            or count.device.type != "cpu"
            for route, count in zip(routes, counts, strict=True)
        ):
            raise ValueError("Gain-Cost plan batches must be aligned CPU int64 tensors")
        route_width = max(route.shape[1] for route in routes)
        total_queries = sum(len(count) for count in counts)
        merged_routes = torch.full((total_queries, route_width), -1, dtype=torch.int64)
        merged_counts = torch.empty(total_queries, dtype=torch.int64)
        offset = 0
        for route, count in zip(routes, counts, strict=True):
            stop = offset + len(count)
            merged_routes[offset:stop, : route.shape[1]].copy_(route)
            merged_counts[offset:stop].copy_(count)
            offset = stop
        active = torch.arange(route_width).unsqueeze(0) < merged_counts.unsqueeze(1)
        merged_routes.masked_fill_(~active, -1)
        active_routes = merged_routes.masked_fill(~active, 0)
        ndis = (list_sizes[active_routes] * active).sum(dim=1, dtype=torch.int64)

    return PreparedPaddedProbePlan(
        bucket_ids=merged_routes.numpy(),
        probe_counts=merged_counts.numpy(),
        ndis=ndis.numpy(),
    )


def _sort_active_routes_by_bucket_id(plan: ProbePlan) -> ProbePlan:
    bucket_ids = np.array(plan.bucket_ids, copy=True)
    for row, count in zip(bucket_ids, plan.probe_counts, strict=True):
        row[:count].sort()
    return ProbePlan(bucket_ids=bucket_ids, probe_counts=plan.probe_counts)


def _gain_cost_fast_width(
    policy: Mapping[str, Any],
    *,
    candidate_bound: int,
    gain_power: float,
    cost_power: float,
) -> int:
    if gain_power != 1.0 or cost_power != 1.0:
        return candidate_bound
    training_width = policy.get("training_width")
    if not isinstance(training_width, Mapping):
        return candidate_bound
    predicted = training_width.get("predicted")
    if not isinstance(predicted, Mapping):
        return candidate_bound
    maximum = predicted.get("max")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        return candidate_bound
    maximum = float(maximum)
    if not math.isfinite(maximum) or maximum < 1 or not maximum.is_integer():
        return candidate_bound
    training_predicted_max = int(maximum)
    if training_predicted_max >= candidate_bound:
        return candidate_bound
    target = max(
        _GAIN_COST_FAST_MIN_WIDTH,
        _GAIN_COST_FAST_TRAINING_MULTIPLIER * training_predicted_max,
    )
    if target >= candidate_bound:
        return candidate_bound
    return min(candidate_bound, 1 << (target - 1).bit_length())


def _gain_cost_probe_counts(
    mass: torch.Tensor,
    *,
    normalized_sizes: torch.Tensor,
    selected: torch.Tensor,
    alpha: float,
    beta: float,
    gain_power: float,
    cost_power: float,
) -> torch.Tensor:
    gain = mass.gather(1, selected)
    cost = normalized_sizes.unsqueeze(0).expand(mass.shape[0], -1).gather(1, selected)
    if gain_power != 1.0:
        gain.pow_(gain_power)
    if cost_power != 1.0:
        cost.pow_(cost_power)
    gain_prefix = gain.cumsum(dim=1)
    gain_prefix.mul_(alpha)
    cost_prefix = cost.cumsum(dim=1)
    cost_prefix.mul_(beta)
    gain_prefix.sub_(cost_prefix)
    return torch.argmax(gain_prefix, dim=1).to(torch.int64) + 1


def _certify_gain_cost_tail(
    last_probability: torch.Tensor,
    *,
    mass_sum: torch.Tensor,
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Certify that every bucket after the short prefix has non-positive utility."""
    relative_margin = (
        _GAIN_COST_CERTIFICATION_EPS_MULTIPLIER
        * torch.finfo(last_probability.dtype).eps
    )
    left = last_probability.flatten() * alpha
    right = mass_sum.flatten() * beta
    return (
        torch.isfinite(left)
        & torch.isfinite(right)
        & right.gt(0)
        & left.lt(right * (1.0 - relative_margin))
    )


def _unsorted_top_m_indices(logits: torch.Tensor, bound: int) -> torch.Tensor:
    """Select Top-M without sorting candidates; smaller IDs win cutoff ties."""
    if bound == logits.shape[1]:
        return torch.arange(bound, device=logits.device).expand(len(logits), -1)
    values, selected = torch.topk(logits, bound, dim=1, sorted=False)
    for row in range(len(logits)):
        cutoff = values[row].min()
        tied = torch.nonzero(logits[row].eq(cutoff), as_tuple=False).flatten()
        at_cutoff = values[row].eq(cutoff)
        needed = int(at_cutoff.sum().item())
        if len(tied) > needed:
            selected[row, at_cutoff] = tied[:needed]
    return selected


def _stable_topk_indices(logits: torch.Tensor, bound: int) -> torch.Tensor:
    """Return raw-logit top-k with stable argsort's bucket-ID tie semantics."""
    if (
        logits.device.type == "cpu"
        and logits.dtype in (torch.float32, torch.float64)
        and hasattr(_gain_cost_native, "sorted_top_m_cpu")
    ):
        return _gain_cost_native.sorted_top_m_cpu(logits.contiguous(), bound)
    # Compatibility fallback for unsupported devices/dtypes or missing builds.
    # Formal CPU Probability benchmarks require the native selector above.
    bucket_count = logits.shape[1]
    topk_width = min(bound + 1, bucket_count)
    top_values, top_indices = torch.topk(
        logits, topk_width, dim=1, largest=True, sorted=True
    )
    selected = top_indices[:, :bound]
    selected_values = top_values[:, :bound]

    if bound < bucket_count:
        boundary_tied = top_values[:, bound - 1].eq(top_values[:, bound])
        for row in torch.nonzero(boundary_tied, as_tuple=False).flatten().tolist():
            cutoff = top_values[row, bound - 1]
            higher_count = int(top_values[row, :bound].gt(cutoff).sum().item())
            tie_count = bound - higher_count
            tie_ids = torch.nonzero(logits[row].eq(cutoff), as_tuple=False).flatten()[
                :tie_count
            ]
            selected[row, higher_count:bound] = tie_ids

    if bound <= 1:
        return selected
    tie_edges = torch.nonzero(
        selected_values[:, :-1].eq(selected_values[:, 1:]), as_tuple=False
    )
    if tie_edges.numel() == 0:
        return selected

    continuation = tie_edges[1:, 0].eq(tie_edges[:-1, 0]) & tie_edges[1:, 1].eq(
        tie_edges[:-1, 1] + 1
    )
    has_previous = torch.zeros(
        len(tie_edges), dtype=torch.bool, device=tie_edges.device
    )
    has_next = torch.zeros_like(has_previous)
    has_previous[1:] = continuation
    has_next[:-1] = continuation

    pairs = tie_edges[~has_previous & ~has_next]
    if pairs.numel():
        rows, columns = pairs.unbind(dim=1)
        swap = selected[rows, columns].gt(selected[rows, columns + 1])
        rows = rows[swap]
        columns = columns[swap]
        left = selected[rows, columns].clone()
        selected[rows, columns] = selected[rows, columns + 1]
        selected[rows, columns + 1] = left

    group_starts = tie_edges[~has_previous & has_next]
    group_ends = tie_edges[has_previous & ~has_next]
    for start, end in zip(group_starts.tolist(), group_ends.tolist(), strict=True):
        row, column = start
        last = end[1] + 1
        selected[row, column : last + 1] = torch.sort(
            selected[row, column : last + 1]
        ).values
    return selected
