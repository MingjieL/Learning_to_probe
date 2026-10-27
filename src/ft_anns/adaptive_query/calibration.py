"""Offline operating-point selection and independent Recall certification."""

from __future__ import annotations

import json
import logging
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from ft_anns.adaptive_query.artifacts import CalibrationArtifact
from ft_anns.adaptive_query.calibration_execution import (
    stream_decoder_policy_batches,
)
from ft_anns.adaptive_query.evaluation import (
    compute_recall_at_k,
    count_recall_hits_at_k,
)
from ft_anns.adaptive_query.policy_library import (
    PolicyLibrary,
    PolicyLibraryCertificationReport,
    PolicyLibraryEntry,
)
from ft_anns.adaptive_query.retrieval import retrieve_probe_plan
from ft_anns.artifact_io import readonly_array, write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.compatibility import normalize_compatibility_profile
from ft_anns.config import (
    CalibrationConfig,
    CalibrationDevice,
    PolicyBudgetConfig,
    PolicyLibraryConfig,
)
from ft_anns.datasets.base_gt import FileVectorSource
from ft_anns.datasets.registry import dataset_spec
from ft_anns.datasets.workloads import WorkloadManifest, load_role_ground_truth
from ft_anns.domain import SearchExecution
from ft_anns.observability import (
    CALIBRATION_COUNTER_NAMES,
    CALIBRATION_TIMING_NAMES,
    CalibrationProgressReporter,
    SearchTimingTrace,
    calibration_runtime_observations,
    utc_now_iso,
)
from ft_anns.partition import FixedPartition, ProbePlan
from ft_anns.partition.fixed_ivf import PolicyPartition

if TYPE_CHECKING:
    from ft_anns.routing.validation import CheckpointValidationWorkload


logger = logging.getLogger(__name__)


def resolve_calibration_device(requested_device: CalibrationDevice) -> str:
    """Resolve the Router execution device for calibration."""
    cuda_available = bool(torch.cuda.is_available())
    if requested_device == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA calibration requested but PyTorch reports no CUDA device"
        )
    if requested_device == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested_device


def _validate_declared_compatibility(
    config: CalibrationConfig | PolicyLibraryConfig,
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
) -> None:
    if config.compatibility is None:
        return
    declared = config.compatibility.model_dump(mode="json", exclude_none=True)
    stored = checkpoint.metadata.get("compatibility")
    if not isinstance(stored, dict):
        raise ValueError("checkpoint Compatibility Profile is missing")
    if normalize_compatibility_profile(declared) != normalize_compatibility_profile(
        stored
    ):
        raise ValueError(
            "calibration Compatibility Profile does not match the checkpoint"
        )
    if (
        declared["dataset_name"] != partition.dataset_name
        or declared["bucket_count"] != partition.bucket_count
        or declared["kmeans_niter"] != partition.kmeans_niter
        or declared["partition_build_seed"] != partition.build_seed
    ):
        raise ValueError(
            "calibration Compatibility Profile does not match the Fixed Partition"
        )


@dataclass(frozen=True)
class CalibrationWorkload:
    queries: npt.NDArray[np.float32]
    ground_truth: npt.NDArray[np.int64]
    query_ids: npt.NDArray[np.int64]
    manifest_filename: str
    dataset_name: str
    metric: str
    source: str
    role: str = "policy_tuning"

    def __post_init__(self) -> None:
        queries = readonly_array(self.queries, np.float32)
        ground_truth = readonly_array(self.ground_truth, np.int64)
        query_ids = readonly_array(self.query_ids, np.int64)
        if (
            queries.ndim != 2
            or ground_truth.ndim != 2
            or query_ids.ndim != 1
            or not len(queries)
            or not (len(queries) == len(ground_truth) == len(query_ids))
        ):
            raise ValueError("Calibration Workload arrays are incompatible")
        if not np.isfinite(queries).all() or np.any(ground_truth < 0):
            raise ValueError("Calibration Workload contents are invalid")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "ground_truth", ground_truth)
        object.__setattr__(self, "query_ids", query_ids)


@dataclass(frozen=True)
class PolicyCostFitWorkload:
    queries: npt.NDArray[np.float32]
    query_ids: npt.NDArray[np.int64]
    manifest_filename: str
    dataset_name: str
    metric: str
    source: str
    role: str = "policy_cost_fit"

    def __post_init__(self) -> None:
        queries = readonly_array(self.queries, np.float32)
        query_ids = readonly_array(self.query_ids, np.int64)
        if (
            queries.ndim != 2
            or query_ids.ndim != 1
            or not len(queries)
            or len(queries) != len(query_ids)
            or not np.isfinite(queries).all()
        ):
            raise ValueError("Policy Cost-Fit Workload arrays are incompatible")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "query_ids", query_ids)


@dataclass(frozen=True)
class LambdaFit:
    cost_lambda: float
    target_mean_ndis: float
    actual_mean_ndis: float
    m_max: int
    iterations: int
    bracket_lower: float
    bracket_upper: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "cost_lambda": self.cost_lambda,
            "target_mean_ndis": self.target_mean_ndis,
            "actual_mean_ndis": self.actual_mean_ndis,
            "m_max": self.m_max,
            "iterations": self.iterations,
            "bracket_lower": self.bracket_lower,
            "bracket_upper": self.bracket_upper,
        }


class CumulativePolicyTable(Protocol):
    @property
    def cumulative_mass(self) -> npt.NDArray[np.float64]: ...

    @property
    def cumulative_cost(self) -> npt.NDArray[np.int64]: ...

    @property
    def candidate_bound(self) -> int: ...


@dataclass(frozen=True)
class _CostFitTable:
    cumulative_mass: npt.NDArray[np.float64]
    cumulative_cost: npt.NDArray[np.int64]

    @property
    def candidate_bound(self) -> int:
        return int(self.cumulative_mass.shape[1])


@dataclass(frozen=True)
class _CudaCostFitTable:
    cumulative_mass: torch.Tensor
    cumulative_cost: torch.Tensor

    @property
    def candidate_bound(self) -> int:
        return int(self.cumulative_mass.shape[1])


def _to_cuda_cost_fit_table(
    table: CumulativePolicyTable | _CudaCostFitTable,
) -> _CudaCostFitTable:
    if isinstance(table, _CudaCostFitTable):
        return table
    return _CudaCostFitTable(
        cumulative_mass=torch.from_numpy(
            np.ascontiguousarray(table.cumulative_mass)
        ).to("cuda"),
        cumulative_cost=torch.from_numpy(
            np.ascontiguousarray(table.cumulative_cost)
        ).to("cuda"),
    )


class _SelectedCostEvaluator:
    """Reuse selected-cost temporaries for every step of one serial fit."""

    def __init__(
        self,
        table: CumulativePolicyTable,
        *,
        m_min: int,
        m_max: int,
    ) -> None:
        first = m_min - 1
        self._mass = table.cumulative_mass[:, first:m_max]
        self._cost = table.cumulative_cost[:, first:m_max]
        self._cost_float64: npt.NDArray[np.float64] | None = None
        self._utility: npt.NDArray[np.float64] | None = None
        self._selected = np.empty(len(self._mass), dtype=np.intp)
        self._row_indices = np.arange(len(self._mass), dtype=np.intp)

    def selected_total_ndis(self, cost_lambda: float) -> int:
        if cost_lambda == 0.0:
            np.argmax(self._mass, axis=1, out=self._selected)
        else:
            if self._cost_float64 is None:
                self._cost_float64 = self._cost.astype(np.float64, copy=True)
                self._utility = np.empty(self._mass.shape, dtype=np.float64)
            assert self._utility is not None
            np.multiply(self._cost_float64, cost_lambda, out=self._utility)
            np.subtract(self._mass, self._utility, out=self._utility)
            np.argmax(self._utility, axis=1, out=self._selected)
        return int(
            self._cost[self._row_indices, self._selected].sum(dtype=np.int64)
        )


class _CudaSelectedCostEvaluator:
    """Keep exact CUDA temporaries alive for every step of one serial fit."""

    def __init__(
        self,
        table: _CudaCostFitTable,
        *,
        m_min: int,
        m_max: int,
    ) -> None:
        first = m_min - 1
        self._mass = table.cumulative_mass[:, first:m_max]
        self._cost = table.cumulative_cost[:, first:m_max]
        self._cost_float64: torch.Tensor | None = None
        self._utility: torch.Tensor | None = None
        self._selected = torch.empty(
            len(self._mass), dtype=torch.int64, device=self._mass.device
        )
        self._selected_cost = torch.empty(
            (len(self._mass), 1), dtype=torch.int64, device=self._mass.device
        )
        self._total = torch.empty(
            (), dtype=torch.int64, device=self._mass.device
        )

    def selected_total_ndis(self, cost_lambda: float) -> int:
        if cost_lambda == 0.0:
            torch.argmax(self._mass, dim=1, out=self._selected)
        else:
            if self._cost_float64 is None:
                self._cost_float64 = self._cost.to(dtype=torch.float64)
                self._utility = torch.empty_like(self._mass)
            assert self._utility is not None
            torch.mul(self._cost_float64, cost_lambda, out=self._utility)
            torch.sub(self._mass, self._utility, out=self._utility)
            torch.argmax(self._utility, dim=1, out=self._selected)
        torch.gather(
            self._cost,
            1,
            self._selected[:, None],
            out=self._selected_cost,
        )
        torch.sum(self._selected_cost, dim=(0, 1), out=self._total)
        return int(self._total.item())


_LambdaFitDecision = tuple[str, float, int, bool]


def _fit_cost_safe_lambda(
    table: CumulativePolicyTable | _CudaCostFitTable,
    *,
    device: str = "cpu",
    target_mean_ndis: float,
    m_min: int,
    m_max: int,
    initial_lambda: float,
    iterations: int,
    bracket_max: float,
    progress_callback: Callable[[], None] | None = None,
    decision_trace: list[_LambdaFitDecision] | None = None,
) -> LambdaFit:
    if (
        isinstance(target_mean_ndis, bool)
        or not isinstance(target_mean_ndis, (int, float))
        or not math.isfinite(float(target_mean_ndis))
        or target_mean_ndis <= 0
        or isinstance(m_min, bool)
        or isinstance(m_max, bool)
        or not isinstance(m_min, int)
        or not isinstance(m_max, int)
        or m_min < 1
        or m_max < m_min
        or m_max > table.candidate_bound
        or isinstance(initial_lambda, bool)
        or not math.isfinite(float(initial_lambda))
        or initial_lambda < 0
        or isinstance(iterations, bool)
        or not isinstance(iterations, int)
        or iterations <= 0
        or isinstance(bracket_max, bool)
        or not math.isfinite(float(bracket_max))
        or bracket_max <= 0
        or initial_lambda > bracket_max
    ):
        raise ValueError("cost-safe lambda fit inputs are invalid")
    query_count = len(table.cumulative_mass)
    target_total = float(target_mean_ndis) * query_count
    if device == "cuda":
        assert isinstance(table, _CudaCostFitTable)
        evaluator = _CudaSelectedCostEvaluator(table, m_min=m_min, m_max=m_max)
    else:
        assert not isinstance(table, _CudaCostFitTable)
        evaluator = _SelectedCostEvaluator(table, m_min=m_min, m_max=m_max)
    zero_cost = evaluator.selected_total_ndis(0.0)
    if decision_trace is not None:
        decision_trace.append(
            ("lambda_zero", 0.0, zero_cost, zero_cost <= target_total + 1e-9)
        )
    if zero_cost <= target_total + 1e-9:
        if decision_trace is not None:
            decision_trace.append(("safe_endpoint", 0.0, zero_cost, True))
        return LambdaFit(
            cost_lambda=0.0,
            target_mean_ndis=float(target_mean_ndis),
            actual_mean_ndis=zero_cost / query_count,
            m_max=m_max,
            iterations=0,
            bracket_lower=0.0,
            bracket_upper=0.0,
        )

    lower = 0.0
    upper = max(float(initial_lambda), float(np.finfo(np.float64).eps))
    while True:
        upper_cost = evaluator.selected_total_ndis(upper)
        if decision_trace is not None:
            decision_trace.append(
                (
                    "bracket",
                    upper,
                    upper_cost,
                    upper_cost <= target_total + 1e-9,
                )
            )
        if progress_callback is not None:
            progress_callback()
        if upper_cost <= target_total + 1e-9:
            break
        lower = upper
        upper *= 2.0
        if upper > bracket_max:
            raise ValueError("unable to bracket a cost-safe lambda")

    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_cost = evaluator.selected_total_ndis(midpoint)
        if decision_trace is not None:
            decision_trace.append(
                (
                    "binary",
                    midpoint,
                    midpoint_cost,
                    midpoint_cost <= target_total + 1e-9,
                )
            )
        if progress_callback is not None:
            progress_callback()
        if midpoint_cost > target_total + 1e-9:
            lower = midpoint
        else:
            upper = midpoint
    actual_total = evaluator.selected_total_ndis(upper)
    if decision_trace is not None:
        decision_trace.append(
            ("safe_endpoint", upper, actual_total, True)
        )
    return LambdaFit(
        cost_lambda=upper,
        target_mean_ndis=float(target_mean_ndis),
        actual_mean_ndis=actual_total / query_count,
        m_max=m_max,
        iterations=iterations,
        bracket_lower=lower,
        bracket_upper=upper,
    )


def fit_cost_safe_lambda(
    table: CumulativePolicyTable | _CudaCostFitTable,
    *,
    device: str = "cpu",
    target_mean_ndis: float,
    m_min: int,
    m_max: int,
    initial_lambda: float,
    iterations: int,
    bracket_max: float,
    progress_callback: Callable[[], None] | None = None,
) -> LambdaFit:
    """Find the safe-side lambda for one mean-NDis target and candidate cap."""
    if device not in ("cpu", "cuda"):
        raise ValueError("lambda fitting device must be cpu or cuda")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA lambda fitting requested but CUDA is unavailable")
        table = _to_cuda_cost_fit_table(table)
    return _fit_cost_safe_lambda(
        table,
        device=device,
        target_mean_ndis=target_mean_ndis,
        m_min=m_min,
        m_max=m_max,
        initial_lambda=initial_lambda,
        iterations=iterations,
        bracket_max=bracket_max,
        progress_callback=progress_callback,
    )


def _fit_cost_safe_lambda_with_trace(
    table: CumulativePolicyTable,
    *,
    target_mean_ndis: float,
    m_min: int,
    m_max: int,
    initial_lambda: float,
    iterations: int,
    bracket_max: float,
) -> tuple[LambdaFit, tuple[_LambdaFitDecision, ...]]:
    decisions: list[_LambdaFitDecision] = []
    fitted = _fit_cost_safe_lambda(
        table,
        target_mean_ndis=target_mean_ndis,
        m_min=m_min,
        m_max=m_max,
        initial_lambda=initial_lambda,
        iterations=iterations,
        bracket_max=bracket_max,
        decision_trace=decisions,
    )
    return fitted, tuple(decisions)


def generate_policy_budget_widths(
    *,
    mean_ndis_by_width: npt.ArrayLike,
    calibration_bound: int,
    minimum_fixed_bucket_count: int,
    target_policy_count: int,
) -> tuple[int, ...]:
    """Map log-spaced NDis targets to unique Decoder Fixed-Width controls."""
    costs = np.asarray(mean_ndis_by_width, dtype=np.float64)
    if (
        costs.ndim != 1
        or isinstance(calibration_bound, bool)
        or not isinstance(calibration_bound, int)
        or calibration_bound < 1
        or calibration_bound > len(costs)
        or isinstance(minimum_fixed_bucket_count, bool)
        or not isinstance(minimum_fixed_bucket_count, int)
        or minimum_fixed_bucket_count < 1
        or isinstance(target_policy_count, bool)
        or not isinstance(target_policy_count, int)
        or target_policy_count < 1
        or not np.isfinite(costs[:calibration_bound]).all()
        or np.any(costs[:calibration_bound] <= 0)
        or np.any(np.diff(costs[:calibration_bound]) <= 0)
    ):
        raise ValueError("automatic policy budget inputs are invalid")
    first_width = min(minimum_fixed_bucket_count, calibration_bound)
    eligible = costs[first_width - 1 : calibration_bound]
    requested = min(target_policy_count, len(eligible))
    if requested == 1:
        return tuple(sorted({first_width, calibration_bound}))
    target_costs = np.geomspace(eligible[0], eligible[-1], num=requested)
    log_costs = np.log(eligible)
    selected: set[int] = {first_width, calibration_bound}
    for target in target_costs[1:-1]:
        log_target = math.log(float(target))
        upper = int(np.searchsorted(log_costs, log_target, side="left"))
        if upper <= 0:
            index = 0
        elif upper >= len(log_costs):
            index = len(log_costs) - 1
        else:
            lower = upper - 1
            index = (
                lower
                if log_target - log_costs[lower]
                <= log_costs[upper] - log_target
                else upper
            )
        selected.add(first_width + index)
    return tuple(sorted(selected))


def select_validation_widths(
    *,
    evaluate_widths: Callable[
        [tuple[int, ...]], Iterable[Mapping[str, int | float]]
    ],
    lower_recall_target: float,
    upper_recall_target: float,
    maximum_fixed_bucket_count: int,
    coarse_step: int = 10,
    monotonicity_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Resolve exact lower and upper Recall widths with one shared cache."""
    if (
        not 0 < lower_recall_target <= upper_recall_target <= 1
        or isinstance(maximum_fixed_bucket_count, bool)
        or maximum_fixed_bucket_count < 1
        or isinstance(coarse_step, bool)
        or coarse_step < 1
        or not math.isfinite(monotonicity_tolerance)
        or monotonicity_tolerance < 0
    ):
        raise ValueError("validation width selection inputs are invalid")

    cache: dict[int, dict[str, int | float]] = {}

    def evaluate(requested: tuple[int, ...]) -> None:
        missing = tuple(width for width in requested if width not in cache)
        if not missing:
            return
        returned = tuple(evaluate_widths(missing))
        points_by_width: dict[int, dict[str, int | float]] = {}
        for raw in returned:
            point = dict(raw)
            width_value = point.get("fixed_bucket_count")
            recall_value = point.get("recall")
            if (
                isinstance(width_value, bool)
                or not isinstance(width_value, int)
                or width_value not in missing
                or width_value in points_by_width
                or isinstance(recall_value, bool)
                or not isinstance(recall_value, (int, float))
                or not math.isfinite(float(recall_value))
                or not 0 <= float(recall_value) <= 1
            ):
                raise ValueError("validation width evaluation is invalid")
            point["recall"] = float(recall_value)
            points_by_width[width_value] = point
        if set(points_by_width) != set(missing):
            raise ValueError("validation width evaluation is incomplete")
        cache.update(points_by_width)

    coarse_widths = tuple(
        range(coarse_step, maximum_fixed_bucket_count + 1, coarse_step)
    )
    if not coarse_widths or coarse_widths[-1] != maximum_fixed_bucket_count:
        coarse_widths = (*coarse_widths, maximum_fixed_bucket_count)
    # Large bounds make every additional exact ANN width expensive. Evaluate
    # at most 100 buckets of coarse-width span at a time and stop after the
    # upper target is crossed; small historical scans remain one shared call.
    coarse_batch_size = max(1, 100 // coarse_step)
    evaluated_coarse: list[int] = []
    for start in range(0, len(coarse_widths), coarse_batch_size):
        batch = coarse_widths[start : start + coarse_batch_size]
        evaluate(batch)
        evaluated_coarse.extend(batch)
        if any(
            float(cache[width]["recall"]) + 1e-12 >= upper_recall_target
            for width in batch
        ):
            break
    coarse_widths = tuple(evaluated_coarse)

    def first_coarse_crossing(target: float) -> int | None:
        return next(
            (
                width
                for width in coarse_widths
                if float(cache[width]["recall"]) + 1e-12 >= target
            ),
            None,
        )

    lower_crossing = first_coarse_crossing(lower_recall_target)
    upper_crossing = first_coarse_crossing(upper_recall_target)
    if upper_crossing is None:
        points = [cache[width] for width in sorted(cache)]
        raise ValueError(
            "Decoder Fixed-Width scan does not reach upper Recall target; "
            f"points={json.dumps(points, sort_keys=True)}"
        )
    assert lower_crossing is not None

    refinement: set[int] = set()
    for crossing in (lower_crossing, upper_crossing):
        interval_start = max(1, crossing - coarse_step + 1)
        refinement.update(range(interval_start, crossing))
    evaluate(tuple(sorted(refinement)))

    def exact_width(target: float, crossing: int) -> int:
        interval_start = max(1, crossing - coarse_step + 1)
        return next(
            width
            for width in range(interval_start, crossing + 1)
            if float(cache[width]["recall"]) + 1e-12 >= target
        )

    minimum_qualified_width = exact_width(lower_recall_target, lower_crossing)
    calibration_bound = exact_width(upper_recall_target, upper_crossing)
    if minimum_qualified_width > calibration_bound:
        raise ValueError(
            "non-monotonic Validation Recall produced M_min above the "
            "Calibration Bound"
        )
    points = [cache[width] for width in sorted(cache)]
    anomalies = [
        {
            "previous_width": int(previous["fixed_bucket_count"]),
            "previous_recall": float(previous["recall"]),
            "width": int(current["fixed_bucket_count"]),
            "recall": float(current["recall"]),
            "decrease": float(previous["recall"]) - float(current["recall"]),
        }
        for previous, current in zip(points, points[1:])
        if float(previous["recall"])
        > float(current["recall"]) + monotonicity_tolerance
    ]
    return {
        "minimum_qualified_width": minimum_qualified_width,
        "calibration_bound": calibration_bound,
        "lower_recall_target": lower_recall_target,
        "upper_recall_target": upper_recall_target,
        "points": points,
        "monotonicity_anomalies": anomalies,
    }


@dataclass(frozen=True)
class _CalibrationCandidate:
    cost_lambda: float
    recall: float
    mean_ndis: float
    m_max: int
    estimated_latency_cost: float | None

    @classmethod
    def from_mapping(
        cls, candidate: Mapping[str, Any], *, require_latency: bool
    ) -> _CalibrationCandidate:
        try:
            cost_lambda = float(candidate["cost_lambda"])
            recall = float(candidate["recall"])
            mean_ndis = float(candidate["mean_ndis"])
            m_max = int(candidate["m_max"])
            estimated_latency_cost = (
                float(candidate["estimated_latency_cost"])
                if require_latency
                else None
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("calibration candidate fields are invalid") from error
        if (
            isinstance(candidate.get("cost_lambda"), bool)
            or not math.isfinite(cost_lambda)
            or cost_lambda < 0
            or isinstance(candidate.get("recall"), bool)
            or not math.isfinite(recall)
            or not 0 <= recall <= 1
            or isinstance(candidate.get("mean_ndis"), bool)
            or not math.isfinite(mean_ndis)
            or mean_ndis <= 0
            or isinstance(candidate.get("m_max"), bool)
            or m_max <= 0
            or (
                estimated_latency_cost is not None
                and (
                    not math.isfinite(estimated_latency_cost)
                    or estimated_latency_cost < 0
                )
            )
        ):
            raise ValueError("calibration candidate fields are invalid")
        return cls(
            cost_lambda=cost_lambda,
            recall=recall,
            mean_ndis=mean_ndis,
            m_max=m_max,
            estimated_latency_cost=estimated_latency_cost,
        )


@dataclass(frozen=True)
class CandidateEvaluationTimingTrace(SearchTimingTrace):
    """Calibration-local extension for Decoder policy-table timing."""

    policy_table_seconds: float = 0.0
    logical_candidate_plan_evaluation_count: int = 0
    unique_active_probe_plan_count: int = 0
    reused_active_probe_plan_count: int = 0
    avoided_faiss_search_count: int = 0

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            self.policy_table_seconds < 0
            or self.logical_candidate_plan_evaluation_count < 0
            or self.unique_active_probe_plan_count < 0
            or self.reused_active_probe_plan_count < 0
            or self.avoided_faiss_search_count < 0
            or self.total_seconds + 1e-12
            < self.router_seconds
            + self.policy_table_seconds
            + self.probe_plan_seconds
            + (self.search_total_seconds or 0.0)
            + (self.metric_seconds or 0.0)
        ):
            raise ValueError("calibration timing values are inconsistent")

    @property
    def overhead_seconds(self) -> float:
        return max(
            0.0,
            self.total_seconds
            - self.router_seconds
            - self.policy_table_seconds
            - self.probe_plan_seconds
            - (self.search_total_seconds or 0.0)
            - (self.metric_seconds or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "policy_table_seconds": self.policy_table_seconds,
            "logical_candidate_plan_evaluation_count": (
                self.logical_candidate_plan_evaluation_count
            ),
            "unique_active_probe_plan_count": self.unique_active_probe_plan_count,
            "reused_active_probe_plan_count": self.reused_active_probe_plan_count,
            "avoided_faiss_search_count": self.avoided_faiss_search_count,
        }


def _merge_candidate_evaluation_timings(
    timings: Iterable[CandidateEvaluationTimingTrace],
) -> CandidateEvaluationTimingTrace:
    resolved = tuple(timings)
    if not resolved:
        raise ValueError("at least one candidate evaluation timing is required")
    executions = {timing.search_execution for timing in resolved}
    return CandidateEvaluationTimingTrace(
        router_seconds=sum(timing.router_seconds for timing in resolved),
        policy_table_seconds=sum(
            timing.policy_table_seconds for timing in resolved
        ),
        probe_plan_seconds=sum(timing.probe_plan_seconds for timing in resolved),
        search_seconds=sum(timing.search_seconds for timing in resolved),
        search_total_seconds=sum(
            float(timing.search_total_seconds or 0.0) for timing in resolved
        ),
        metric_seconds=sum(float(timing.metric_seconds or 0.0) for timing in resolved),
        total_seconds=sum(timing.total_seconds for timing in resolved),
        search_packing_seconds=sum(
            timing.search_packing_seconds for timing in resolved
        ),
        faiss_call_count=sum(timing.faiss_call_count for timing in resolved),
        search_execution=(executions.pop() if len(executions) == 1 else "mixed"),
        logical_candidate_plan_evaluation_count=sum(
            timing.logical_candidate_plan_evaluation_count for timing in resolved
        ),
        unique_active_probe_plan_count=sum(
            timing.unique_active_probe_plan_count for timing in resolved
        ),
        reused_active_probe_plan_count=sum(
            timing.reused_active_probe_plan_count for timing in resolved
        ),
        avoided_faiss_search_count=sum(
            timing.avoided_faiss_search_count for timing in resolved
        ),
    )


def _active_probe_plan_arrays(
    plan: ProbePlan,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    active = np.arange(plan.bucket_ids.shape[1])[None, :] < (
        plan.probe_counts[:, None]
    )
    return plan.probe_counts, plan.bucket_ids[active]


def _active_probe_plan_key(plan: ProbePlan) -> tuple[bytes, bytes]:
    counts, active_bucket_ids = _active_probe_plan_arrays(plan)
    return counts.tobytes(), active_bucket_ids.tobytes()


def _active_probe_plan_hash(plan: ProbePlan) -> int:
    return hash(_active_probe_plan_key(plan))


def _active_probe_plans_equal(left: ProbePlan, right: ProbePlan) -> bool:
    left_counts, left_active_bucket_ids = _active_probe_plan_arrays(left)
    right_counts, right_active_bucket_ids = _active_probe_plan_arrays(right)
    return np.array_equal(left_counts, right_counts) and np.array_equal(
        left_active_bucket_ids, right_active_bucket_ids
    )


def group_equivalent_probe_plans(
    plans: Iterable[ProbePlan],
    *,
    plan_hash: Callable[[ProbePlan], int] | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Group batch Probe Plans with identical counts and active prefixes."""
    resolved_plans = tuple(plans)
    hash_plan = _active_probe_plan_hash if plan_hash is None else plan_hash
    groups: list[list[int]] = []
    candidate_groups_by_hash: dict[int, list[int]] = {}
    for index, plan in enumerate(resolved_plans):
        plan_hash_value = hash_plan(plan)
        for group_index in candidate_groups_by_hash.get(plan_hash_value, ()):
            representative = resolved_plans[groups[group_index][0]]
            if _active_probe_plans_equal(plan, representative):
                groups[group_index].append(index)
                break
        else:
            group_index = len(groups)
            groups.append([index])
            candidate_groups_by_hash.setdefault(plan_hash_value, []).append(
                group_index
            )
    return tuple(tuple(indices) for indices in groups)


def _progress_timing_snapshot(
    *,
    router_seconds: float,
    policy_table_seconds: float,
    cost_planning_seconds: float,
    packing_seconds: float,
    faiss_seconds: float,
    metric_seconds: float,
    total_seconds: float | None = None,
) -> dict[str, float]:
    timing = {
        "router_seconds": router_seconds,
        "policy_table_seconds": policy_table_seconds,
        "cost_planning_seconds": cost_planning_seconds,
        "packing_seconds": packing_seconds,
        "faiss_seconds": faiss_seconds,
        "metric_seconds": metric_seconds,
    }
    if total_seconds is not None:
        timing["total_seconds"] = total_seconds
    return timing


def _load_policy_role_queries(
    manifest: WorkloadManifest,
    *,
    partition: FixedPartition | PolicyPartition,
    role: str,
    query_limit: int | None = None,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.int64],
    Mapping[str, Any],
    Mapping[str, Any],
]:
    if role not in manifest.metadata["roles"]:
        raise ValueError(f"Workload Manifest does not define {role}")
    metadata = manifest.metadata
    resolved = metadata["resolved_config"]
    specification = dataset_spec(metadata["dataset_name"])
    base_source = FileVectorSource(
        Path(resolved["base_path"]), specification.format, specification.dimension
    )
    source = (
        base_source
        if resolved["source"] == "base_set"
        else FileVectorSource(
            Path(resolved["query_path"]),
            resolved.get("query_format") or specification.format,
            specification.dimension,
        )
    )
    if base_source.filename != metadata["base_filename"]:
        raise ValueError("base source filename no longer matches the manifest")
    if source.filename != metadata["query_source_filename"]:
        raise ValueError("query source filename no longer matches the manifest")
    if partition.dimension != specification.dimension:
        raise ValueError("Calibration Workload dimension does not match the partition")
    if partition.metric != metadata["ground_truth"]["metric"]:
        raise ValueError("Calibration Workload metric does not match the partition")
    if len(partition.bucket_assignment) != base_source.count:
        raise ValueError("Fixed Partition base count does not match the manifest")
    query_ids = np.asarray(manifest.role_query_ids(role), dtype=np.int64)
    if query_limit is not None:
        if isinstance(query_limit, bool) or not isinstance(query_limit, int) or not 0 < query_limit <= len(query_ids):
            raise ValueError("query_limit must be positive and within the role count")
        query_ids = query_ids[:query_limit]
    return source.read_rows(query_ids), query_ids, metadata, resolved


def load_policy_cost_fit_workload(
    manifest: WorkloadManifest,
    *,
    partition: FixedPartition | PolicyPartition,
) -> PolicyCostFitWorkload:
    """Load the declared GT-free cost-fit role without opening GT rows."""
    queries, query_ids, metadata, resolved = _load_policy_role_queries(
        manifest, partition=partition, role="policy_cost_fit"
    )
    return PolicyCostFitWorkload(
        queries=queries,
        query_ids=query_ids,
        manifest_filename=manifest.filename,
        dataset_name=str(metadata["dataset_name"]),
        metric=str(metadata["ground_truth"]["metric"]),
        source=str(resolved["source"]),
    )


def load_calibration_workload(
    manifest: WorkloadManifest,
    *,
    partition: FixedPartition | PolicyPartition,
    role: str = "policy_tuning",
    query_limit: int | None = None,
    timings: dict[str, float] | None = None,
) -> CalibrationWorkload:
    """Replay one declared policy-calibration role from the frozen manifest."""
    if role not in ("policy_tuning", "policy_certification"):
        raise ValueError("unsupported calibration workload role")
    if (
        role == "policy_certification"
        and "policy_tuning" in manifest.metadata["roles"]
        and "policy_certification" in manifest.metadata["roles"]
    ):
        overlap = set(manifest.role_query_ids("policy_tuning")).intersection(
            manifest.role_query_ids("policy_certification")
        )
        if overlap:
            raise ValueError("policy_tuning and policy_certification role IDs overlap")
    started = time.perf_counter()
    queries, query_ids, metadata, resolved = _load_policy_role_queries(
        manifest, partition=partition, role=role, query_limit=query_limit
    )
    queries_finished = time.perf_counter()
    ground_truth = load_role_ground_truth(manifest, role, query_limit=query_limit)
    if timings is not None:
        timings["query_load_seconds"] = queries_finished - started
        timings["ground_truth_load_seconds"] = time.perf_counter() - queries_finished
    return CalibrationWorkload(
        queries=queries,
        ground_truth=ground_truth,
        query_ids=query_ids,
        manifest_filename=manifest.filename,
        dataset_name=str(metadata["dataset_name"]),
        metric=str(metadata["ground_truth"]["metric"]),
        source=str(resolved["source"]),
        role=role,
    )


def select_budget_ndis_candidate(
    candidates: Iterable[Mapping[str, Any]], *, budget_ndis: float
) -> dict[str, Any]:
    """Choose maximum Recall under budget with a stable aggregate tie-break."""
    if (
        isinstance(budget_ndis, bool)
        or not isinstance(budget_ndis, (int, float))
        or not math.isfinite(float(budget_ndis))
        or budget_ndis <= 0
    ):
        raise ValueError("Budget-NDis constraint must be finite and positive")
    feasible: list[tuple[_CalibrationCandidate, Mapping[str, Any]]] = []
    for candidate in candidates:
        validated = _CalibrationCandidate.from_mapping(
            candidate, require_latency=False
        )
        if validated.mean_ndis <= float(budget_ndis) + 1e-9:
            feasible.append((validated, candidate))
    if not feasible:
        raise ValueError("no calibration candidate satisfies the NDis budget")
    return dict(
        max(
            feasible,
            key=lambda item: (
                item[0].recall,
                -item[0].mean_ndis,
                -item[0].m_max,
                item[0].cost_lambda,
            ),
        )[1]
    )


def select_target_recall_candidate(
    candidates: Iterable[Mapping[str, Any]], *, target_recall: float
) -> dict[str, Any]:
    """Choose by NDis then deterministic estimated latency after Recall gating."""
    if (
        isinstance(target_recall, bool)
        or not isinstance(target_recall, (int, float))
        or not math.isfinite(float(target_recall))
        or not 0 < target_recall <= 1
    ):
        raise ValueError("Target-Recall constraint must be within (0, 1]")
    feasible: list[tuple[_CalibrationCandidate, Mapping[str, Any]]] = []
    for candidate in candidates:
        validated = _CalibrationCandidate.from_mapping(
            candidate, require_latency=True
        )
        if validated.recall + 1e-12 >= float(target_recall):
            feasible.append((validated, candidate))
    if not feasible:
        raise ValueError("no calibration candidate satisfies the Recall target")
    return dict(
        min(
            feasible,
            key=lambda item: (
                item[0].mean_ndis,
                item[0].estimated_latency_cost,
                item[0].m_max,
                -item[0].recall,
                item[0].cost_lambda,
            ),
        )[1]
    )


def recall_lower_confidence_bound(
    *, observed_recall: float, sample_count: int, confidence_level: float
) -> float:
    """Return a one-sided Hoeffding lower bound for mean query-level Recall."""
    if (
        isinstance(observed_recall, bool)
        or not isinstance(observed_recall, (int, float))
        or not math.isfinite(float(observed_recall))
        or not 0 <= observed_recall <= 1
    ):
        raise ValueError("observed Recall must be within [0, 1]")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise ValueError("certification sample count must be positive")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(float(confidence_level))
        or not 0 < confidence_level < 1
    ):
        raise ValueError("confidence level must be within (0, 1)")
    radius = math.sqrt(
        math.log(1.0 / (1.0 - float(confidence_level))) / (2.0 * sample_count)
    )
    return max(0.0, float(observed_recall) - radius)


def _evaluate_candidates(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    config: CalibrationConfig | PolicyLibraryConfig,
    policies: Iterable[Mapping[str, int | float]],
    progress: CalibrationProgressReporter | None = None,
    progress_stage: str = "candidate_evaluation",
    reuse_equivalent_probe_plans: bool | None = None,
) -> tuple[list[dict[str, int | float]], CandidateEvaluationTimingTrace]:
    started = time.perf_counter()
    router_seconds = 0.0
    policy_table_seconds = 0.0
    probe_plan_seconds = 0.0
    search_seconds = 0.0
    search_total_seconds = 0.0
    search_packing_seconds = 0.0
    faiss_call_count = 0
    logical_candidate_plan_evaluation_count = 0
    unique_active_probe_plan_count = 0
    reused_active_probe_plan_count = 0
    avoided_faiss_search_count = 0
    metric_seconds = 0.0
    aggregates = [
        {
            **dict(policy),
            "recall_hit_count": 0,
            "ndis_sum": 0,
            "probe_count_sum": 0,
        }
        for policy in policies
    ]
    if not aggregates:
        raise ValueError("calibration requires at least one policy candidate")
    candidate_bound = max(int(aggregate["m_max"]) for aggregate in aggregates)
    total_batches = math.ceil(len(workload.queries) / config.retrieval_batch_size)
    if progress is not None:
        progress.start(stage=progress_stage, total_batches=total_batches)
    for batch in stream_decoder_policy_batches(
        checkpoint=checkpoint,
        partition=partition,
        queries=workload.queries,
        candidate_bound=candidate_bound,
        batch_size=config.retrieval_batch_size,
        decoder=config.decoder,
    ):
        offset = batch.offset
        stop = batch.stop
        batch_queries = batch.queries
        batch_ground_truth = workload.ground_truth[offset:stop]
        router_seconds = batch.router_seconds
        policy_table_seconds = batch.policy_table_seconds
        batch_plans: list[ProbePlan] = []
        for aggregate in aggregates:
            probe_plan_started = time.perf_counter()
            plan = batch.policy_table.select_prefixes(
                cost_lambda=float(aggregate["cost_lambda"]),
                m_min=int(aggregate["m_min"]),
                m_max=int(aggregate["m_max"]),
            )
            batch_plans.append(plan)
            probe_plan_seconds += time.perf_counter() - probe_plan_started
        reuse_plans = (
            workload.role == config.tuning_role
            if reuse_equivalent_probe_plans is None
            else reuse_equivalent_probe_plans
        )
        plan_groups = (
            group_equivalent_probe_plans(batch_plans)
            if reuse_plans
            else tuple((index,) for index in range(len(batch_plans)))
        )
        logical_candidate_plan_evaluation_count += len(batch_plans)
        reused_active_probe_plan_count += len(batch_plans) - len(plan_groups)
        for plan_group in plan_groups:
            plan = batch_plans[plan_group[0]]
            search = retrieve_probe_plan(
                queries=batch_queries,
                partition=partition,
                probe_plan=plan,
                topk=config.recall_k,
                search_execution=config.search_execution,
            )
            search_seconds += search.timing.search_seconds
            search_total_seconds += search.timing.total_seconds
            search_packing_seconds += search.timing.packing_seconds
            faiss_call_count += search.timing.faiss_call_count
            avoided_faiss_search_count += search.timing.faiss_call_count * (
                len(plan_group) - 1
            )
            metric_started = time.perf_counter()
            recall_hit_count = count_recall_hits_at_k(
                search.neighbor_ids,
                batch_ground_truth,
                k=config.recall_k,
            )
            ndis_sum = int(search.ndis.sum(dtype=np.int64))
            probe_count_sum = int(search.probe_counts.sum(dtype=np.int64))
            for aggregate_index in plan_group:
                aggregate = aggregates[aggregate_index]
                aggregate["recall_hit_count"] += recall_hit_count
                aggregate["ndis_sum"] += ndis_sum
                aggregate["probe_count_sum"] += probe_count_sum
            metric_seconds += time.perf_counter() - metric_started
            if progress is not None:
                progress.update(
                    completed_batches=offset // config.retrieval_batch_size,
                    generated_candidate_count=len(aggregates),
                    feasible_candidate_count=len(aggregates),
                    logical_candidate_plan_evaluation_count=(
                        logical_candidate_plan_evaluation_count
                    ),
                    unique_active_probe_plan_count=(
                        unique_active_probe_plan_count
                        + len(plan_groups)
                    ),
                    reused_active_probe_plan_count=(
                        reused_active_probe_plan_count
                    ),
                    avoided_faiss_search_count=avoided_faiss_search_count,
                    faiss_call_count=faiss_call_count,
                    timing_seconds=_progress_timing_snapshot(
                        router_seconds=router_seconds,
                        policy_table_seconds=policy_table_seconds,
                        cost_planning_seconds=probe_plan_seconds,
                        packing_seconds=search_packing_seconds,
                        faiss_seconds=search_seconds,
                        metric_seconds=metric_seconds,
                    ),
                )
        unique_active_probe_plan_count += len(plan_groups)
        if progress is not None:
            progress.update(
                completed_batches=batch.completed_batch_count,
                generated_candidate_count=len(aggregates),
                feasible_candidate_count=len(aggregates),
                logical_candidate_plan_evaluation_count=(
                    logical_candidate_plan_evaluation_count
                ),
                unique_active_probe_plan_count=unique_active_probe_plan_count,
                reused_active_probe_plan_count=reused_active_probe_plan_count,
                avoided_faiss_search_count=avoided_faiss_search_count,
                faiss_call_count=faiss_call_count,
                timing_seconds=_progress_timing_snapshot(
                    router_seconds=router_seconds,
                    policy_table_seconds=policy_table_seconds,
                    cost_planning_seconds=probe_plan_seconds,
                    packing_seconds=search_packing_seconds,
                    faiss_seconds=search_seconds,
                    metric_seconds=metric_seconds,
                ),
            )
    query_count = len(workload.queries)
    candidates = [
        {
            "cost_lambda": float(aggregate["cost_lambda"]),
            "m_min": int(aggregate["m_min"]),
            "m_max": int(aggregate["m_max"]),
            "recall": int(aggregate["recall_hit_count"])
            / (query_count * config.recall_k),
            "mean_ndis": int(aggregate["ndis_sum"]) / query_count,
            "mean_probe_count": int(aggregate["probe_count_sum"]) / query_count,
            "estimated_latency_cost": int(aggregate["probe_count_sum"])
            / query_count,
        }
        for aggregate in aggregates
    ]
    total_seconds = time.perf_counter() - started
    if progress is not None:
        progress.finish(
            completed_batches=total_batches,
            generated_candidate_count=len(aggregates),
            feasible_candidate_count=len(aggregates),
            evaluated_candidate_count=len(aggregates),
            logical_candidate_plan_evaluation_count=(
                logical_candidate_plan_evaluation_count
            ),
            unique_active_probe_plan_count=unique_active_probe_plan_count,
            reused_active_probe_plan_count=reused_active_probe_plan_count,
            avoided_faiss_search_count=avoided_faiss_search_count,
            faiss_call_count=faiss_call_count,
            timing_seconds=_progress_timing_snapshot(
                router_seconds=router_seconds,
                policy_table_seconds=policy_table_seconds,
                cost_planning_seconds=probe_plan_seconds,
                packing_seconds=search_packing_seconds,
                faiss_seconds=search_seconds,
                metric_seconds=metric_seconds,
                total_seconds=total_seconds,
            ),
        )
    logger.debug(
        "candidate evaluation completed role=%s queries=%d policies=%d "
        "router_seconds=%.6f probe_plan_seconds=%.6f search_seconds=%.6f "
        "metric_seconds=%.6f total_seconds=%.6f",
        workload.role,
        len(workload.queries),
        len(aggregates),
        router_seconds,
        probe_plan_seconds,
        search_seconds,
        metric_seconds,
        total_seconds,
    )
    return candidates, CandidateEvaluationTimingTrace(
        router_seconds=router_seconds,
        policy_table_seconds=policy_table_seconds,
        probe_plan_seconds=probe_plan_seconds,
        search_seconds=search_seconds,
        search_total_seconds=search_total_seconds,
        metric_seconds=metric_seconds,
        total_seconds=total_seconds,
        search_packing_seconds=search_packing_seconds,
        faiss_call_count=faiss_call_count,
        search_execution=config.search_execution,
        logical_candidate_plan_evaluation_count=(
            logical_candidate_plan_evaluation_count
        ),
        unique_active_probe_plan_count=unique_active_probe_plan_count,
        reused_active_probe_plan_count=reused_active_probe_plan_count,
        avoided_faiss_search_count=avoided_faiss_search_count,
    )


def _fixed_m_max_policy(config: CalibrationConfig) -> dict[str, int | float]:
    if config.m_max is None:
        raise ValueError("calibration m_max must be resolved before evaluation")
    return {
        "cost_lambda": 0.0,
        "m_min": config.m_max,
        "m_max": config.m_max,
    }


@dataclass(frozen=True)
class FixedWidthCostPlanningTimingTrace:
    """Timing and plan counts for a Fixed-Width control with no search."""

    router_seconds: float
    policy_table_seconds: float
    cost_planning_seconds: float
    total_seconds: float
    unique_active_probe_plan_count: int
    faiss_call_count: int = 0

    def __post_init__(self) -> None:
        if (
            self.router_seconds < 0
            or self.policy_table_seconds < 0
            or self.cost_planning_seconds < 0
            or self.unique_active_probe_plan_count < 0
            or self.faiss_call_count != 0
            or self.total_seconds + 1e-12
            < self.router_seconds
            + self.policy_table_seconds
            + self.cost_planning_seconds
        ):
            raise ValueError("Fixed-Width cost-planning timing is inconsistent")

    @property
    def overhead_seconds(self) -> float:
        return max(
            0.0,
            self.total_seconds
            - self.router_seconds
            - self.policy_table_seconds
            - self.cost_planning_seconds,
        )

    def to_dict(self) -> dict[str, int | float]:
        return {
            "router_seconds": self.router_seconds,
            "policy_table_seconds": self.policy_table_seconds,
            "cost_planning_seconds": self.cost_planning_seconds,
            "faiss_call_count": self.faiss_call_count,
            "unique_active_probe_plan_count": self.unique_active_probe_plan_count,
            "overhead_seconds": self.overhead_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class PolicyTuningCostPlanningTimingTrace:
    """Timing and bounded retained-state evidence for tuning cost planning."""

    router_seconds: float
    policy_table_seconds: float
    cost_planning_seconds: float
    total_seconds: float
    logical_candidate_plan_evaluation_count: int
    fixed_width_plan_evaluation_count: int
    maximum_batch_query_count: int
    maximum_router_score_bytes: int
    maximum_probe_plan_bytes: int
    faiss_call_count: int = 0
    full_workload_router_scores_retained: bool = False

    def __post_init__(self) -> None:
        if (
            self.router_seconds < 0
            or self.policy_table_seconds < 0
            or self.cost_planning_seconds < 0
            or self.total_seconds + 1e-12
            < self.router_seconds
            + self.policy_table_seconds
            + self.cost_planning_seconds
            or self.logical_candidate_plan_evaluation_count < 0
            or self.fixed_width_plan_evaluation_count < 0
            or self.maximum_batch_query_count < 0
            or self.maximum_router_score_bytes < 0
            or self.maximum_probe_plan_bytes < 0
            or self.faiss_call_count != 0
            or self.full_workload_router_scores_retained
        ):
            raise ValueError("Policy Tuning cost-planning timing is inconsistent")

    @property
    def overhead_seconds(self) -> float:
        return max(
            0.0,
            self.total_seconds
            - self.router_seconds
            - self.policy_table_seconds
            - self.cost_planning_seconds,
        )

    def to_dict(self) -> dict[str, int | float | bool]:
        return {
            "router_seconds": self.router_seconds,
            "policy_table_seconds": self.policy_table_seconds,
            "cost_planning_seconds": self.cost_planning_seconds,
            "faiss_call_count": self.faiss_call_count,
            "logical_candidate_plan_evaluation_count": (
                self.logical_candidate_plan_evaluation_count
            ),
            "fixed_width_plan_evaluation_count": (
                self.fixed_width_plan_evaluation_count
            ),
            "maximum_batch_query_count": self.maximum_batch_query_count,
            "maximum_router_score_bytes": self.maximum_router_score_bytes,
            "maximum_probe_plan_bytes": self.maximum_probe_plan_bytes,
            "full_workload_router_scores_retained": (
                self.full_workload_router_scores_retained
            ),
            "overhead_seconds": self.overhead_seconds,
            "total_seconds": self.total_seconds,
        }


@dataclass(frozen=True)
class _FixedWidthBatchTiming:
    cost_planning_seconds: float = 0.0
    search_seconds: float = 0.0
    search_total_seconds: float = 0.0
    search_packing_seconds: float = 0.0
    metric_seconds: float = 0.0
    faiss_call_count: int = 0


@dataclass(frozen=True)
class _FixedWidthRunTiming:
    router_seconds: float
    policy_table_seconds: float
    cost_planning_seconds: float
    search_seconds: float
    search_total_seconds: float
    search_packing_seconds: float
    metric_seconds: float
    total_seconds: float
    faiss_call_count: int
    unique_active_probe_plan_count: int


def _run_decoder_fixed_width_controls(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    queries: npt.NDArray[np.float32],
    fixed_bucket_counts: tuple[int, ...],
    batch_size: int,
    consume_batch: Callable[
        [
            int,
            int,
            npt.NDArray[np.float32],
            tuple[tuple[int, ProbePlan], ...],
        ],
        _FixedWidthBatchTiming,
    ],
    progress: CalibrationProgressReporter | None,
    progress_stage: str,
) -> _FixedWidthRunTiming:
    if (
        not fixed_bucket_counts
        or tuple(sorted(set(fixed_bucket_counts))) != fixed_bucket_counts
        or fixed_bucket_counts[0] < 1
        or fixed_bucket_counts[-1] > partition.bucket_count
        or queries.ndim != 2
        or not len(queries)
        or queries.shape[1] != partition.dimension
        or batch_size < 1
    ):
        raise ValueError("Decoder Fixed-Width control inputs are invalid")
    started = time.perf_counter()
    router_seconds = 0.0
    policy_table_seconds = 0.0
    cost_planning_seconds = 0.0
    search_seconds = 0.0
    search_total_seconds = 0.0
    search_packing_seconds = 0.0
    metric_seconds = 0.0
    faiss_call_count = 0
    unique_active_probe_plan_count = 0
    candidate_bound = fixed_bucket_counts[-1]
    total_batches = math.ceil(len(queries) / batch_size)
    if progress is not None:
        progress.start(stage=progress_stage, total_batches=total_batches)
    for batch in stream_decoder_policy_batches(
        checkpoint=checkpoint,
        partition=partition,
        queries=queries,
        candidate_bound=candidate_bound,
        batch_size=batch_size,
    ):
        router_seconds = batch.router_seconds
        policy_table_seconds = batch.policy_table_seconds
        planning_started = time.perf_counter()
        plans = tuple(
            (width, batch.policy_table.fixed_width_plan(width))
            for width in fixed_bucket_counts
        )
        cost_planning_seconds += time.perf_counter() - planning_started
        unique_active_probe_plan_count += len(
            {_active_probe_plan_key(plan) for _, plan in plans}
        )
        batch_timing = consume_batch(
            batch.offset,
            batch.stop,
            batch.queries,
            plans,
        )
        cost_planning_seconds += batch_timing.cost_planning_seconds
        search_seconds += batch_timing.search_seconds
        search_total_seconds += batch_timing.search_total_seconds
        search_packing_seconds += batch_timing.search_packing_seconds
        metric_seconds += batch_timing.metric_seconds
        faiss_call_count += batch_timing.faiss_call_count
        if progress is not None:
            progress.update(
                completed_batches=batch.completed_batch_count,
                unique_active_probe_plan_count=unique_active_probe_plan_count,
                faiss_call_count=faiss_call_count,
                timing_seconds=_progress_timing_snapshot(
                    router_seconds=router_seconds,
                    policy_table_seconds=policy_table_seconds,
                    cost_planning_seconds=cost_planning_seconds,
                    packing_seconds=search_packing_seconds,
                    faiss_seconds=search_seconds,
                    metric_seconds=metric_seconds,
                ),
            )
    total_seconds = time.perf_counter() - started
    if progress is not None:
        progress.finish(
            completed_batches=total_batches,
            unique_active_probe_plan_count=unique_active_probe_plan_count,
            faiss_call_count=faiss_call_count,
            timing_seconds=_progress_timing_snapshot(
                router_seconds=router_seconds,
                policy_table_seconds=policy_table_seconds,
                cost_planning_seconds=cost_planning_seconds,
                packing_seconds=search_packing_seconds,
                faiss_seconds=search_seconds,
                metric_seconds=metric_seconds,
                total_seconds=total_seconds,
            ),
        )
    return _FixedWidthRunTiming(
        router_seconds=router_seconds,
        policy_table_seconds=policy_table_seconds,
        cost_planning_seconds=cost_planning_seconds,
        search_seconds=search_seconds,
        search_total_seconds=search_total_seconds,
        search_packing_seconds=search_packing_seconds,
        metric_seconds=metric_seconds,
        total_seconds=total_seconds,
        faiss_call_count=faiss_call_count,
        unique_active_probe_plan_count=unique_active_probe_plan_count,
    )


def _evaluate_decoder_fixed_widths(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    queries: npt.NDArray[np.float32],
    ground_truth: npt.NDArray[np.int64],
    fixed_bucket_counts: tuple[int, ...],
    recall_k: int,
    batch_size: int,
    search_execution: SearchExecution,
    progress: CalibrationProgressReporter | None = None,
    progress_stage: str = "decoder_fixed_width_evaluation",
) -> tuple[list[dict[str, int | float]], CandidateEvaluationTimingTrace]:
    if (
        ground_truth.ndim != 2
        or len(ground_truth) != len(queries)
        or recall_k > ground_truth.shape[1]
    ):
        raise ValueError("Decoder Fixed-Width Recall inputs are invalid")
    aggregates = {
        width: {"recall_query_sum": 0.0, "ndis_sum": 0}
        for width in fixed_bucket_counts
    }

    def evaluate_batch(
        offset: int,
        stop: int,
        batch_queries: npt.NDArray[np.float32],
        plans: tuple[tuple[int, ProbePlan], ...],
    ) -> _FixedWidthBatchTiming:
        search_seconds = 0.0
        search_total_seconds = 0.0
        search_packing_seconds = 0.0
        metric_seconds = 0.0
        faiss_call_count = 0
        for width, plan in plans:
            search = retrieve_probe_plan(
                queries=batch_queries,
                partition=partition,
                probe_plan=plan,
                topk=recall_k,
                search_execution=search_execution,
            )
            search_seconds += search.timing.search_seconds
            search_total_seconds += search.timing.total_seconds
            search_packing_seconds += search.timing.packing_seconds
            faiss_call_count += search.timing.faiss_call_count
            metric_started = time.perf_counter()
            aggregate = aggregates[width]
            aggregate["recall_query_sum"] += compute_recall_at_k(
                search.neighbor_ids,
                ground_truth[offset:stop],
                k=recall_k,
            ) * len(batch_queries)
            aggregate["ndis_sum"] += int(search.ndis.sum(dtype=np.int64))
            metric_seconds += time.perf_counter() - metric_started
        return _FixedWidthBatchTiming(
            search_seconds=search_seconds,
            search_total_seconds=search_total_seconds,
            search_packing_seconds=search_packing_seconds,
            metric_seconds=metric_seconds,
            faiss_call_count=faiss_call_count,
        )

    timing = _run_decoder_fixed_width_controls(
        checkpoint=checkpoint,
        partition=partition,
        queries=queries,
        fixed_bucket_counts=fixed_bucket_counts,
        batch_size=batch_size,
        consume_batch=evaluate_batch,
        progress=progress,
        progress_stage=progress_stage,
    )
    points = [
        {
            "fixed_bucket_count": width,
            "recall": float(aggregate["recall_query_sum"]) / len(queries),
            "mean_ndis": int(aggregate["ndis_sum"]) / len(queries),
        }
        for width, aggregate in aggregates.items()
    ]
    return points, CandidateEvaluationTimingTrace(
        router_seconds=timing.router_seconds,
        policy_table_seconds=timing.policy_table_seconds,
        probe_plan_seconds=timing.cost_planning_seconds,
        search_seconds=timing.search_seconds,
        search_total_seconds=timing.search_total_seconds,
        metric_seconds=timing.metric_seconds,
        total_seconds=timing.total_seconds,
        search_packing_seconds=timing.search_packing_seconds,
        faiss_call_count=timing.faiss_call_count,
        search_execution=search_execution,
        unique_active_probe_plan_count=timing.unique_active_probe_plan_count,
    )


def plan_decoder_fixed_width_costs(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    queries: npt.NDArray[np.float32],
    fixed_bucket_counts: tuple[int, ...],
    batch_size: int,
    progress: CalibrationProgressReporter | None = None,
    progress_stage: str = "decoder_fixed_width_cost_planning",
) -> tuple[list[dict[str, int | float]], FixedWidthCostPlanningTimingTrace]:
    """Plan exact Decoder Fixed-Width costs without evaluating Recall."""
    ndis_sums = {width: 0 for width in fixed_bucket_counts}

    def compute_batch_costs(
        _offset: int,
        _stop: int,
        _batch_queries: npt.NDArray[np.float32],
        plans: tuple[tuple[int, ProbePlan], ...],
    ) -> _FixedWidthBatchTiming:
        started = time.perf_counter()
        for width, plan in plans:
            ndis_sums[width] += int(
                partition.compute_ndis(plan).sum(dtype=np.int64)
            )
        return _FixedWidthBatchTiming(
            cost_planning_seconds=time.perf_counter() - started
        )

    timing = _run_decoder_fixed_width_controls(
        checkpoint=checkpoint,
        partition=partition,
        queries=queries,
        fixed_bucket_counts=fixed_bucket_counts,
        batch_size=batch_size,
        consume_batch=compute_batch_costs,
        progress=progress,
        progress_stage=progress_stage,
    )
    return [
        {
            "fixed_bucket_count": width,
            "mean_ndis": ndis_sums[width] / len(queries),
        }
        for width in fixed_bucket_counts
    ], FixedWidthCostPlanningTimingTrace(
        router_seconds=timing.router_seconds,
        policy_table_seconds=timing.policy_table_seconds,
        cost_planning_seconds=timing.cost_planning_seconds,
        total_seconds=timing.total_seconds,
        unique_active_probe_plan_count=timing.unique_active_probe_plan_count,
    )


def plan_policy_tuning_costs(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    queries: npt.NDArray[np.float32],
    policies: Iterable[Mapping[str, int | float]],
    fixed_bucket_counts: tuple[int, ...],
    batch_size: int,
    progress: CalibrationProgressReporter | None = None,
    progress_stage: str = "policy_tuning_cost_planning",
) -> tuple[
    list[dict[str, int | float]],
    list[dict[str, int | float]],
    PolicyTuningCostPlanningTimingTrace,
]:
    """Stream exact candidate and Fixed-Width tuning NDis without FAISS."""
    resolved_policies = tuple(dict(policy) for policy in policies)
    if (
        not resolved_policies
        or not fixed_bucket_counts
        or tuple(sorted(set(fixed_bucket_counts))) != fixed_bucket_counts
        or fixed_bucket_counts[0] < 1
        or queries.ndim != 2
        or not len(queries)
        or queries.shape[1] != partition.dimension
        or batch_size < 1
    ):
        raise ValueError("Policy Tuning cost-planning inputs are invalid")
    candidate_bound = max(
        fixed_bucket_counts[-1],
        *(int(policy["m_max"]) for policy in resolved_policies),
    )
    if candidate_bound > partition.bucket_count:
        raise ValueError("Policy Tuning cost-planning bound exceeds partition")

    started = time.perf_counter()
    router_seconds = 0.0
    policy_table_seconds = 0.0
    cost_planning_seconds = 0.0
    logical_candidate_plan_evaluation_count = 0
    fixed_width_plan_evaluation_count = 0
    maximum_batch_query_count = 0
    maximum_router_score_bytes = 0
    maximum_probe_plan_bytes = 0
    candidate_ndis_sums = [0 for _ in resolved_policies]
    fixed_width_ndis_sums = {width: 0 for width in fixed_bucket_counts}
    total_batches = math.ceil(len(queries) / batch_size)
    if progress is not None:
        progress.start(stage=progress_stage, total_batches=total_batches)

    for batch in stream_decoder_policy_batches(
        checkpoint=checkpoint,
        partition=partition,
        queries=queries,
        candidate_bound=candidate_bound,
        batch_size=batch_size,
    ):
        router_seconds = batch.router_seconds
        policy_table_seconds = batch.policy_table_seconds
        maximum_batch_query_count = max(
            maximum_batch_query_count, len(batch.queries)
        )
        maximum_router_score_bytes = max(
            maximum_router_score_bytes, batch.router_score_bytes
        )

        planning_started = time.perf_counter()
        for index, policy in enumerate(resolved_policies):
            plan = batch.policy_table.select_prefixes(
                cost_lambda=float(policy["cost_lambda"]),
                m_min=int(policy["m_min"]),
                m_max=int(policy["m_max"]),
            )
            candidate_ndis_sums[index] += int(
                partition.compute_ndis(plan).sum(dtype=np.int64)
            )
            maximum_probe_plan_bytes = max(
                maximum_probe_plan_bytes,
                plan.bucket_ids.nbytes + plan.probe_counts.nbytes,
            )
        for width in fixed_bucket_counts:
            plan = batch.policy_table.fixed_width_plan(width)
            fixed_width_ndis_sums[width] += int(
                partition.compute_ndis(plan).sum(dtype=np.int64)
            )
            maximum_probe_plan_bytes = max(
                maximum_probe_plan_bytes,
                plan.bucket_ids.nbytes + plan.probe_counts.nbytes,
            )
        cost_planning_seconds += time.perf_counter() - planning_started
        logical_candidate_plan_evaluation_count += len(resolved_policies)
        fixed_width_plan_evaluation_count += len(fixed_bucket_counts)
        if progress is not None:
            progress.update(
                completed_batches=batch.completed_batch_count,
                generated_candidate_count=len(resolved_policies),
                logical_candidate_plan_evaluation_count=(
                    logical_candidate_plan_evaluation_count
                ),
                faiss_call_count=0,
                timing_seconds=_progress_timing_snapshot(
                    router_seconds=router_seconds,
                    policy_table_seconds=policy_table_seconds,
                    cost_planning_seconds=cost_planning_seconds,
                    packing_seconds=0.0,
                    faiss_seconds=0.0,
                    metric_seconds=0.0,
                ),
            )

    total_seconds = time.perf_counter() - started
    if progress is not None:
        progress.finish(
            completed_batches=total_batches,
            generated_candidate_count=len(resolved_policies),
            logical_candidate_plan_evaluation_count=(
                logical_candidate_plan_evaluation_count
            ),
            faiss_call_count=0,
            timing_seconds=_progress_timing_snapshot(
                router_seconds=router_seconds,
                policy_table_seconds=policy_table_seconds,
                cost_planning_seconds=cost_planning_seconds,
                packing_seconds=0.0,
                faiss_seconds=0.0,
                metric_seconds=0.0,
                total_seconds=total_seconds,
            ),
        )
    query_count = len(queries)
    return (
        [
            {
                "cost_lambda": float(policy["cost_lambda"]),
                "m_min": int(policy["m_min"]),
                "m_max": int(policy["m_max"]),
                "mean_ndis": candidate_ndis_sums[index] / query_count,
            }
            for index, policy in enumerate(resolved_policies)
        ],
        [
            {
                "fixed_bucket_count": width,
                "mean_ndis": fixed_width_ndis_sums[width] / query_count,
            }
            for width in fixed_bucket_counts
        ],
        PolicyTuningCostPlanningTimingTrace(
            router_seconds=router_seconds,
            policy_table_seconds=policy_table_seconds,
            cost_planning_seconds=cost_planning_seconds,
            total_seconds=total_seconds,
            logical_candidate_plan_evaluation_count=(
                logical_candidate_plan_evaluation_count
            ),
            fixed_width_plan_evaluation_count=fixed_width_plan_evaluation_count,
            maximum_batch_query_count=maximum_batch_query_count,
            maximum_router_score_bytes=maximum_router_score_bytes,
            maximum_probe_plan_bytes=maximum_probe_plan_bytes,
        ),
    )


def _collect_cost_fit_table(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: PolicyCostFitWorkload,
    candidate_bound: int,
    batch_size: int,
    progress: CalibrationProgressReporter | None = None,
) -> tuple[_CostFitTable, dict[str, float | int]]:
    started = time.perf_counter()
    mass = np.empty((len(workload.queries), candidate_bound), dtype=np.float64)
    cost = np.empty((len(workload.queries), candidate_bound), dtype=np.int64)
    router_seconds = 0.0
    table_seconds = 0.0
    total_batches = math.ceil(len(workload.queries) / batch_size)
    if progress is not None:
        progress.start(stage="cost_fit_table", total_batches=total_batches)
    for batch in stream_decoder_policy_batches(
        checkpoint=checkpoint,
        partition=partition,
        queries=workload.queries,
        candidate_bound=candidate_bound,
        batch_size=batch_size,
    ):
        router_seconds = batch.router_seconds
        table_seconds = batch.policy_table_seconds
        mass[batch.offset : batch.stop] = batch.policy_table.cumulative_mass
        cost[batch.offset : batch.stop] = batch.policy_table.cumulative_cost
        if progress is not None:
            progress.update(
                completed_batches=batch.completed_batch_count,
                timing_seconds={
                    "router_seconds": router_seconds,
                    "policy_table_seconds": table_seconds,
                },
            )
    total_seconds = time.perf_counter() - started
    if progress is not None:
        progress.finish(
            completed_batches=total_batches,
            timing_seconds={
                "router_seconds": router_seconds,
                "policy_table_seconds": table_seconds,
                "total_seconds": total_seconds,
            },
        )
    return _CostFitTable(mass, cost), {
        "router_seconds": router_seconds,
        "policy_table_seconds": table_seconds,
        "total_seconds": total_seconds,
        "query_count": len(workload.queries),
        "candidate_bound": candidate_bound,
        "table_bytes": mass.nbytes + cost.nbytes,
    }


def _resolved_policy_caps(
    config: PolicyLibraryConfig,
    calibration_bound: int,
    default_ladder: tuple[int, ...],
) -> tuple[int, ...]:
    if config.m_max_candidates is None:
        values = set(default_ladder)
    else:
        values = set(config.m_max_candidates)
    if any(value < config.m_min or value > calibration_bound for value in values):
        raise ValueError(
            "Policy Library m_max candidates exceed the Calibration Bound"
        )
    return tuple(sorted(values))


def _resolved_policy_budgets(
    *,
    config: PolicyLibraryConfig,
    cost_table: _CostFitTable,
    calibration_bound: int,
    minimum_qualified_width: int,
    default_ladder: tuple[int, ...],
) -> tuple[PolicyBudgetConfig, ...]:
    if config.policy_budgets is not None:
        return tuple(config.policy_budgets)
    generation = config.policy_budget_generation
    assert generation is not None
    widths = default_ladder
    if generation.include_width_one_budget and minimum_qualified_width != 1:
        widths = (1, *widths)
    if len(widths) > generation.maximum_policy_count:
        raise ValueError("generated policy count exceeds maximum_policy_count")
    return tuple(
        PolicyBudgetConfig(
            policy_id=f"fixed-m{width}-budget",
            reference_fixed_bucket_count=width,
        )
        for width in widths
    )


def _policy_key(policy: Mapping[str, int | float]) -> tuple[float, int, int]:
    return (
        float(policy["cost_lambda"]),
        int(policy["m_min"]),
        int(policy["m_max"]),
    )


def classify_tuning_candidate_costs(
    *,
    policies_by_id: Mapping[str, Iterable[Mapping[str, int | float]]],
    candidate_costs: Iterable[Mapping[str, int | float]],
    target_mean_ndis_by_policy_id: Mapping[str, float],
    certification_cost_margin: float,
) -> tuple[
    tuple[dict[str, int | float], ...],
    tuple[dict[str, Any], ...],
]:
    """Keep the exact-NDis feasible union across referencing policy IDs."""
    policy_ids_by_candidate: dict[tuple[float, int, int], list[str]] = {}
    for policy_id, policies in policies_by_id.items():
        for policy in policies:
            key = _policy_key(policy)
            references = policy_ids_by_candidate.setdefault(key, [])
            if policy_id not in references:
                references.append(policy_id)

    feasible: list[dict[str, int | float]] = []
    diagnostics: list[dict[str, Any]] = []
    for candidate in candidate_costs:
        key = _policy_key(candidate)
        referencing_policy_ids = policy_ids_by_candidate[key]
        mean_ndis = float(candidate["mean_ndis"])
        feasible_policy_ids = [
            policy_id
            for policy_id in referencing_policy_ids
            if mean_ndis
            <= float(target_mean_ndis_by_policy_id[policy_id])
            * (1.0 - certification_cost_margin)
            + 1e-9
        ]
        policy = {
            "cost_lambda": key[0],
            "m_min": key[1],
            "m_max": key[2],
        }
        if feasible_policy_ids:
            feasible.append(policy)
        diagnostics.append(
            {
                **policy,
                "mean_ndis": mean_ndis,
                "status": (
                    "feasible_for_recall"
                    if feasible_policy_ids
                    else "pruned_by_ndis"
                ),
                "referencing_policy_ids": referencing_policy_ids,
                "feasible_policy_ids": feasible_policy_ids,
            }
        )
    return tuple(feasible), tuple(diagnostics)


def _write_library_failure_report(
    *,
    config: PolicyLibraryConfig,
    checkpoint: RouterCheckpoint,
    diagnostics: Mapping[str, Any],
    reason: str,
) -> Path:
    output = Path(config.output_path)
    report_path = output.with_name(f"{output.stem}.calibration-report.json")
    write_json_atomic(
        report_path,
        {
            "format": "policy-library-calibration-report-v1",
            "status": "failed",
            "reason": reason,
            "dataset_name": checkpoint.metadata.get("dataset_name"),
            "checkpoint_filename": checkpoint.filename,
            "workload_manifest_filename": checkpoint.metadata.get(
                "workload_manifest_filename"
            ),
            "resolved_config": config.model_dump(mode="json"),
            "diagnostics": dict(diagnostics),
            "failed_at": utc_now_iso(),
        },
    )
    return report_path


def calibrate_policy_library(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    validation_workload: CheckpointValidationWorkload,
    cost_fit_workload: PolicyCostFitWorkload,
    tuning_workload: CalibrationWorkload,
    config: PolicyLibraryConfig,
) -> PolicyLibrary:
    """Fit, tune, and freeze one uncertified operating point per budget."""
    calibration_started_at = utc_now_iso()
    calibration_started = time.perf_counter()
    resolved_device = resolve_calibration_device(config.device)
    checkpoint.model.to(resolved_device)
    progress = CalibrationProgressReporter(
        emit=lambda payload: logger.info(
            "calibration_progress %s", json.dumps(payload, sort_keys=True)
        ),
        requested_device=config.device,
        resolved_device=resolved_device,
        observe_runtime=lambda: calibration_runtime_observations(
            requested_device=config.device, resolved_device=resolved_device
        ),
    )
    logger.info(
        "Policy Library calibration started dataset=%s fit_queries=%d "
        "tuning_queries=%d budget_mode=%s",
        tuning_workload.dataset_name,
        len(cost_fit_workload.queries),
        len(tuning_workload.queries),
        config.budget_mode,
    )
    if checkpoint.metadata.get("loss", {}).get("name") != "ramps":
        raise ValueError(
            "adaptive Policy Library calibration requires a RAMPS checkpoint"
        )
    checkpoint.validate_for(partition)
    _validate_declared_compatibility(
        config, checkpoint=checkpoint, partition=partition
    )
    manifest = WorkloadManifest.load(config.workload_manifest_path)
    manifest.validate_checkpoint_binding(checkpoint.metadata)
    declared = (
        (cost_fit_workload, config.cost_fit_role),
        (tuning_workload, config.tuning_role),
    )
    for workload, role in declared:
        if (
            workload.manifest_filename != manifest.filename
            or workload.role != role
            or not np.array_equal(workload.query_ids, manifest.role_query_ids(role))
            or workload.dataset_name != checkpoint.metadata.get("dataset_name")
            or workload.dataset_name != partition.dataset_name
            or workload.metric != partition.metric
            or workload.queries.shape[1] != partition.dimension
        ):
            raise ValueError(
                f"Policy Library workload is not the declared {role} role"
            )
    if (
        validation_workload.manifest_filename != manifest.filename
        or not np.array_equal(
            validation_workload.query_ids,
            manifest.role_query_ids("checkpoint_validation"),
        )
    ):
        raise ValueError("Calibration Bound workload is not checkpoint_validation")
    if config.recall_k > tuning_workload.ground_truth.shape[1] or (
        config.recall_k > validation_workload.ground_truth.shape[1]
    ):
        raise ValueError("calibration Recall K exceeds the ground-truth width")
    width_selection_config = config.calibration_bound_selection
    if width_selection_config.maximum_fixed_bucket_count > partition.bucket_count:
        raise ValueError("maximum fixed bucket count exceeds the Fixed Partition")
    validation_timings: list[CandidateEvaluationTimingTrace] = []

    def evaluate_validation_widths(
        widths: tuple[int, ...],
    ) -> list[dict[str, int | float]]:
        points, timing = _evaluate_decoder_fixed_widths(
            checkpoint=checkpoint,
            partition=partition,
            queries=validation_workload.queries,
            ground_truth=validation_workload.ground_truth,
            fixed_bucket_counts=widths,
            recall_k=config.recall_k,
            batch_size=config.retrieval_batch_size,
            search_execution=config.search_execution,
            progress=progress,
            progress_stage="calibration_width_scan",
        )
        validation_timings.append(timing)
        return points

    width_selection = select_validation_widths(
        evaluate_widths=evaluate_validation_widths,
        lower_recall_target=width_selection_config.lower_recall_target,
        upper_recall_target=width_selection_config.upper_recall_target,
        maximum_fixed_bucket_count=(
            width_selection_config.maximum_fixed_bucket_count
        ),
    )
    validation_timing = _merge_candidate_evaluation_timings(validation_timings)
    validation_points = width_selection["points"]
    minimum_qualified_width = int(width_selection["minimum_qualified_width"])
    calibration_bound = int(width_selection["calibration_bound"])
    if width_selection["monotonicity_anomalies"]:
        logger.warning(
            "Validation Recall monotonicity anomalies: %s",
            json.dumps(width_selection["monotonicity_anomalies"], sort_keys=True),
        )
    manual_widths = tuple(
        entry.reference_fixed_bucket_count
        for entry in (config.policy_budgets or ())
    )
    if any(width > partition.bucket_count for width in manual_widths):
        raise ValueError("Policy Library budget width exceeds the Fixed Partition")
    cost_table, cost_fit_timing = _collect_cost_fit_table(
        checkpoint=checkpoint,
        partition=partition,
        workload=cost_fit_workload,
        candidate_bound=max((calibration_bound, *manual_widths)),
        batch_size=config.retrieval_batch_size,
        progress=progress,
    )
    generation = config.policy_budget_generation
    if config.policy_budgets is not None and config.m_max_candidates is not None:
        default_ladder: tuple[int, ...] = ()
    else:
        mean_ndis_by_width = cost_table.cumulative_cost.mean(
            axis=0, dtype=np.float64
        )
        target_policy_count = generation.target_policy_count if generation else 12
        default_ladder = generate_policy_budget_widths(
            mean_ndis_by_width=mean_ndis_by_width,
            calibration_bound=calibration_bound,
            minimum_fixed_bucket_count=minimum_qualified_width,
            target_policy_count=target_policy_count,
        )
    policy_budgets = _resolved_policy_budgets(
        config=config,
        cost_table=cost_table,
        calibration_bound=calibration_bound,
        minimum_qualified_width=minimum_qualified_width,
        default_ladder=default_ladder,
    )
    caps = _resolved_policy_caps(
        config,
        calibration_bound,
        default_ladder,
    )
    budget_widths = tuple(
        sorted(
            entry.reference_fixed_bucket_count for entry in policy_budgets
        )
    )
    fit_target_by_width = {
        width: float(
            cost_table.cumulative_cost[:, width - 1].mean(dtype=np.float64)
        )
        for width in budget_widths
    }
    policies_by_id: dict[str, list[dict[str, int | float]]] = {}
    fits_by_id: dict[str, list[dict[str, int | float]]] = {}
    lambda_fit_started = time.perf_counter()
    lambda_fit_count = (
        0
        if config.lambda_search.mode == "manual"
        else len(policy_budgets) * (1 + len(caps))
    )
    lambda_fit_table: CumulativePolicyTable | _CudaCostFitTable = cost_table
    if config.lambda_search.mode == "auto" and resolved_device == "cuda":
        lambda_fit_table = _to_cuda_cost_fit_table(cost_table)
    completed_lambda_fits = 0
    progress.start(stage="lambda_fit", total_batches=lambda_fit_count)

    def update_lambda_progress() -> None:
        progress.update(
            completed_batches=completed_lambda_fits,
            timing_seconds={
                "lambda_fit_seconds": time.perf_counter() - lambda_fit_started
            },
        )

    for budget in policy_budgets:
        if config.lambda_search.mode == "manual":
            policies_by_id[budget.policy_id] = [
                {
                    "cost_lambda": float(value),
                    "m_min": config.m_min,
                    "m_max": cap,
                }
                for value in config.lambda_search.cost_lambda_candidates
                for cap in caps
            ]
            fits_by_id[budget.policy_id] = []
            continue
        search = config.lambda_search
        target = fit_target_by_width[budget.reference_fixed_bucket_count]
        initial = fit_cost_safe_lambda(
            lambda_fit_table,
            device=resolved_device,
            target_mean_ndis=target,
            m_min=config.m_min,
            m_max=calibration_bound,
            initial_lambda=1.0,
            iterations=search.iterations,
            bracket_max=search.bracket_max,
            progress_callback=update_lambda_progress,
        )
        completed_lambda_fits += 1
        progress.update(
            completed_batches=completed_lambda_fits,
            timing_seconds={
                "lambda_fit_seconds": time.perf_counter() - lambda_fit_started
            },
        )
        fits: list[LambdaFit] = []
        policies: dict[tuple[float, int, int], dict[str, int | float]] = {}
        for cap in caps:
            fitted = fit_cost_safe_lambda(
                lambda_fit_table,
                device=resolved_device,
                target_mean_ndis=target,
                m_min=config.m_min,
                m_max=cap,
                initial_lambda=initial.cost_lambda,
                iterations=search.iterations,
                bracket_max=search.bracket_max,
                progress_callback=update_lambda_progress,
            )
            completed_lambda_fits += 1
            progress.update(
                completed_batches=completed_lambda_fits,
                timing_seconds={
                    "lambda_fit_seconds": time.perf_counter() - lambda_fit_started
                },
            )
            fits.append(fitted)
            factors = (1.0,) if fitted.cost_lambda == 0 else search.local_factors
            for factor in factors:
                policy = {
                    "cost_lambda": float(fitted.cost_lambda * factor),
                    "m_min": config.m_min,
                    "m_max": cap,
                }
                policies[_policy_key(policy)] = policy
        policies_by_id[budget.policy_id] = list(policies.values())
        fits_by_id[budget.policy_id] = [initial.to_dict()] + [
            fitted.to_dict() for fitted in fits
        ]
    lambda_fit_seconds = time.perf_counter() - lambda_fit_started
    progress.finish(
        completed_batches=lambda_fit_count,
        timing_seconds={"lambda_fit_seconds": lambda_fit_seconds},
    )
    if isinstance(lambda_fit_table, _CudaCostFitTable):
        del lambda_fit_table
    unique_policies = {
        _policy_key(policy): policy
        for policies in policies_by_id.values()
        for policy in policies
    }
    candidate_costs, tuning_controls, tuning_cost_planning_timing = (
        plan_policy_tuning_costs(
            checkpoint=checkpoint,
            partition=partition,
            queries=tuning_workload.queries,
            policies=unique_policies.values(),
            fixed_bucket_counts=budget_widths,
            batch_size=config.retrieval_batch_size,
            progress=progress,
            progress_stage="policy_tuning_control_cost_planning",
        )
    )
    tuning_control_by_width = {
        int(point["fixed_bucket_count"]): point for point in tuning_controls
    }
    tuning_target_by_policy_id = {
        budget.policy_id: float(
            tuning_control_by_width[
                budget.reference_fixed_bucket_count
            ]["mean_ndis"]
        )
        for budget in policy_budgets
    }
    feasible_policies, candidate_cost_diagnostics = (
        classify_tuning_candidate_costs(
            policies_by_id=policies_by_id,
            candidate_costs=candidate_costs,
            target_mean_ndis_by_policy_id=tuning_target_by_policy_id,
            certification_cost_margin=config.certification_cost_margin,
        )
    )
    pruned_candidate_count = sum(
        candidate["status"] == "pruned_by_ndis"
        for candidate in candidate_cost_diagnostics
    )
    progress.start(stage="policy_tuning_ndis_pruning", total_batches=1)
    progress.finish(
        completed_batches=1,
        generated_candidate_count=len(unique_policies),
        pruned_candidate_count=pruned_candidate_count,
        feasible_candidate_count=len(feasible_policies),
    )
    if not feasible_policies:
        raise ValueError("no calibration candidate satisfies the NDis budget")
    tuning_candidates, tuning_candidate_timing = _evaluate_candidates(
        checkpoint=checkpoint,
        partition=partition,
        workload=tuning_workload,
        config=config,
        policies=feasible_policies,
        progress=progress,
        progress_stage="policy_tuning_candidates",
    )
    tuning_candidate_by_key = {
        _policy_key(candidate): candidate for candidate in tuning_candidates
    }
    candidate_diagnostics_by_key = {
        _policy_key(candidate): dict(candidate)
        for candidate in candidate_cost_diagnostics
    }
    for candidate in tuning_candidates:
        diagnostic = candidate_diagnostics_by_key[_policy_key(candidate)]
        diagnostic.update(candidate)
        diagnostic["status"] = "recall_evaluated"
    selected_by_id: dict[str, dict[str, int | float]] = {}
    for budget in policy_budgets:
        candidates = [
            tuning_candidate_by_key[_policy_key(policy)]
            for policy in policies_by_id[budget.policy_id]
            if _policy_key(policy) in tuning_candidate_by_key
        ]
        target = tuning_target_by_policy_id[budget.policy_id]
        selected = select_budget_ndis_candidate(
            candidates,
            budget_ndis=target * (1.0 - config.certification_cost_margin),
        )
        selected_by_id[budget.policy_id] = selected
    common_diagnostics: dict[str, Any] = {
        "timestamps": {"started_at": calibration_started_at},
        "calibration_bound_selection": {
            "ranking": "decoder_fixed_width",
            "lower_recall_target": width_selection_config.lower_recall_target,
            "upper_recall_target": width_selection_config.upper_recall_target,
            "recall_k": config.recall_k,
            "minimum_qualified_width": minimum_qualified_width,
            "calibration_bound": calibration_bound,
            "points": validation_points,
            "monotonicity_anomalies": width_selection[
                "monotonicity_anomalies"
            ],
        },
        "reference_width_ladder": list(default_ladder),
        "m_max_candidates": list(caps),
        "resolved_policy_budgets": [
            budget.model_dump(mode="json") for budget in policy_budgets
        ],
        "policy_budget_generation": (
            config.policy_budget_generation.model_dump(mode="json")
            if config.policy_budget_generation is not None
            else None
        ),
        "lambda_search": config.lambda_search.model_dump(mode="json"),
        "budget_mode": config.budget_mode,
        "certification_cost_margin": config.certification_cost_margin,
        "certification_cost_tolerance": config.certification_cost_tolerance,
        "tuning_candidates": list(candidate_diagnostics_by_key.values()),
        "timing": {
            "calibration_width_scan": {
                **validation_timing.to_dict(),
                "control_mode": "recall_evaluated",
            },
            "cost_fit": cost_fit_timing,
            "lambda_fit_seconds": lambda_fit_seconds,
            "tuning_candidate_cost_planning": (
                tuning_cost_planning_timing.to_dict()
            ),
            "tuning_candidate_evaluation": tuning_candidate_timing.to_dict(),
            "tuning_control_cost_planning": {
                **tuning_cost_planning_timing.to_dict(),
                "control_mode": "cost_only",
                "cost_scope": "candidate_and_fixed_width",
            },
        },
    }

    entries: dict[str, PolicyLibraryEntry] = {}
    for budget in policy_budgets:
        selected = selected_by_id[budget.policy_id]
        fit_target = fit_target_by_width[budget.reference_fixed_bucket_count]
        tuning_target = float(
            tuning_control_by_width[budget.reference_fixed_bucket_count]["mean_ndis"]
        )
        artifact = CalibrationArtifact.create(
            checkpoint=checkpoint,
            partition=partition,
            cost_lambda=float(selected["cost_lambda"]),
            m_min=int(selected["m_min"]),
            m_max=int(selected["m_max"]),
            recall_k=config.recall_k,
            budget_ndis=tuning_target,
            observed_recall=float(selected["recall"]),
            mean_ndis=float(selected["mean_ndis"]),
            calibration_query_count=len(tuning_workload.queries),
            diagnostics={
                **common_diagnostics,
                "timestamps": {
                    "started_at": calibration_started_at,
                    "selected_at": utc_now_iso(),
                },
                "decoder_fixed_width_bucket_count": (
                    budget.reference_fixed_bucket_count
                ),
                "cost_fit_target_mean_ndis": fit_target,
                "tuning_target_mean_ndis": tuning_target,
                "certification_status": "not_run",
                "lambda_fits": fits_by_id[budget.policy_id],
                "candidate_count": len(policies_by_id[budget.policy_id]),
                "candidates": [
                    candidate_diagnostics_by_key[_policy_key(policy)]
                    for policy in policies_by_id[budget.policy_id]
                ],
                "tie_breaking": [
                    "higher_recall",
                    "lower_mean_ndis",
                    "lower_m_max",
                    "higher_cost_lambda",
                ],
            },
            calibration_source=tuning_workload.source,
        )
        entries[budget.policy_id] = PolicyLibraryEntry(
            reference_fixed_bucket_count=budget.reference_fixed_bucket_count,
            cost_fit_target_mean_ndis=fit_target,
            tuning_target_mean_ndis=tuning_target,
            calibration_artifact=artifact,
        )
        logger.info(
            "Policy Library operating point selected policy_id=%s decoder_width=%d "
            "m_max=%d mean_ndis=%.6f recall=%.8f",
            budget.policy_id,
            budget.reference_fixed_bucket_count,
            int(selected["m_max"]),
            float(selected["mean_ndis"]),
            float(selected["recall"]),
        )
    computation_finished_at = utc_now_iso()
    compute_seconds = time.perf_counter() - calibration_started
    recall_evaluation_timings = (
        validation_timing,
        tuning_candidate_timing,
    )
    cost_control_timings: tuple[FixedWidthCostPlanningTimingTrace, ...] = ()
    counts = {name: 0 for name in CALIBRATION_COUNTER_NAMES}
    counts.update(
        {
            "generated_candidate_count": len(unique_policies),
            "pruned_candidate_count": pruned_candidate_count,
            "feasible_candidate_count": len(feasible_policies),
            "evaluated_candidate_count": len(tuning_candidates),
            "selected_candidate_count": len(selected_by_id),
            "logical_candidate_plan_evaluation_count": sum(
                timing.logical_candidate_plan_evaluation_count
                for timing in recall_evaluation_timings
            ),
            "unique_active_probe_plan_count": sum(
                timing.unique_active_probe_plan_count
                for timing in recall_evaluation_timings
            )
            + sum(
                timing.unique_active_probe_plan_count
                for timing in cost_control_timings
            ),
            "reused_active_probe_plan_count": sum(
                timing.reused_active_probe_plan_count
                for timing in recall_evaluation_timings
            ),
            "avoided_faiss_search_count": sum(
                timing.avoided_faiss_search_count
                for timing in recall_evaluation_timings
            ),
            "faiss_call_count": sum(
                timing.faiss_call_count for timing in recall_evaluation_timings
            ),
        }
    )
    stage_timing_seconds = {name: 0.0 for name in CALIBRATION_TIMING_NAMES}
    stage_timing_seconds.update(
        {
            "router_seconds": float(cost_fit_timing["router_seconds"])
            + sum(
                timing.router_seconds for timing in recall_evaluation_timings
            )
            + sum(timing.router_seconds for timing in cost_control_timings)
            + tuning_cost_planning_timing.router_seconds,
            "policy_table_seconds": float(
                cost_fit_timing["policy_table_seconds"]
            )
            + sum(
                timing.policy_table_seconds
                for timing in recall_evaluation_timings
            )
            + sum(
                timing.policy_table_seconds for timing in cost_control_timings
            )
            + tuning_cost_planning_timing.policy_table_seconds,
            "lambda_fit_seconds": lambda_fit_seconds,
            "cost_planning_seconds": sum(
                timing.probe_plan_seconds
                for timing in recall_evaluation_timings
            )
            + sum(
                timing.cost_planning_seconds for timing in cost_control_timings
            )
            + tuning_cost_planning_timing.cost_planning_seconds,
            "packing_seconds": sum(
                timing.search_packing_seconds
                for timing in recall_evaluation_timings
            ),
            "faiss_seconds": sum(
                timing.search_seconds for timing in recall_evaluation_timings
            ),
            "metric_seconds": sum(
                timing.metric_seconds or 0.0
                for timing in recall_evaluation_timings
            ),
            "total_seconds": compute_seconds,
        }
    )
    runtime_observations = calibration_runtime_observations(
        requested_device=config.device, resolved_device=resolved_device
    )
    library = PolicyLibrary.create(
        checkpoint=checkpoint,
        partition=partition,
        entries=entries,
        calibration_contract={
            "workload_roles": {
                "validation": "checkpoint_validation",
                "cost_fit": config.cost_fit_role,
                "tuning": config.tuning_role,
                "certification": config.certification_role,
            },
            "certification_query_ids": [
                int(value)
                for value in manifest.role_query_ids(config.certification_role)
            ],
            "minimum_qualified_width": minimum_qualified_width,
            "calibration_bound": calibration_bound,
            "calibration_bound_selection": (
                config.calibration_bound_selection.model_dump(mode="json")
            ),
            "decoder_fixed_width_budget": True,
            "lambda_search": config.lambda_search.model_dump(mode="json"),
            "reference_width_ladder": list(default_ladder),
            "m_max_candidates": list(caps),
            "resolved_policy_budgets": [
                budget.model_dump(mode="json") for budget in policy_budgets
            ],
            "policy_budget_generation": (
                config.policy_budget_generation.model_dump(mode="json")
                if config.policy_budget_generation is not None
                else None
            ),
            "budget_mode": config.budget_mode,
            "tuning_headroom": config.certification_cost_margin,
            "certification_tolerance": config.certification_cost_tolerance,
            "recall_k": config.recall_k,
            "search_execution": config.search_execution,
        },
        observability={
            **runtime_observations,
            "lambda_fit_device": (
                resolved_device
                if config.lambda_search.mode == "auto"
                else None
            ),
            "timestamps": {
                "started_at": calibration_started_at,
                "computation_finished_at": computation_finished_at,
            },
            "timing": {
                "compute_seconds": compute_seconds,
                **common_diagnostics["timing"],
            },
            "counts": counts,
            "stage_timing_seconds": stage_timing_seconds,
        },
    )
    library.save(config.output_path)
    logger.info(
        "Policy Library calibration completed output=%s policies=%d "
        "compute_seconds=%.6f save_seconds=%.6f persisted_at=%s",
        config.output_path,
        len(entries),
        compute_seconds,
        float(library.persistence["save_seconds"]),
        library.persistence["persisted_at"],
    )
    return library


@dataclass(frozen=True)
class _FrozenLibraryCertificationConfig:
    recall_k: int
    retrieval_batch_size: int
    search_execution: SearchExecution
    decoder: str = "ramps_size_matched"


def certify_policy_library(
    *,
    library: PolicyLibrary,
    library_path: str | Path,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    certification_workload: CalibrationWorkload,
    output_path: str | Path | None = None,
    device: CalibrationDevice = "auto",
    retrieval_batch_size: int = 1_000,
    search_execution: SearchExecution = "grouped",
    budget_mode: str | None = None,
) -> PolicyLibraryCertificationReport:
    """Evaluate a frozen Policy Library and write independent evidence."""
    persisted_library = PolicyLibrary.load(
        library_path,
        checkpoint=checkpoint,
        partition=partition,
    )
    if persisted_library.payload != library.payload:
        raise ValueError(
            "Certification Policy Library object does not match the library file"
        )
    library = persisted_library
    contract = library.payload["calibration_contract"]
    declared_mode = str(contract["budget_mode"])
    if budget_mode is not None and budget_mode != declared_mode:
        raise ValueError("Certification budget mode does not match the Policy Library")
    role = str(contract["workload_roles"]["certification"])
    if (
        certification_workload.role != role
        or certification_workload.manifest_filename
        != library.payload["workload_manifest_filename"]
        or certification_workload.dataset_name != library.payload["dataset_name"]
        or certification_workload.dataset_name != partition.dataset_name
        or certification_workload.metric != partition.metric
        or certification_workload.queries.shape[1] != partition.dimension
        or tuple(certification_workload.query_ids.tolist())
        != tuple(contract["certification_query_ids"])
    ):
        raise ValueError(
            "Certification workload is incompatible with the Policy Library"
        )
    recall_k = int(contract["recall_k"])
    if recall_k > certification_workload.ground_truth.shape[1]:
        raise ValueError("Certification Recall K exceeds the ground-truth width")
    if retrieval_batch_size < 1:
        raise ValueError("Certification retrieval batch size must be positive")
    resolved_device = resolve_calibration_device(device)
    checkpoint.model.to(resolved_device)
    evaluation_config = _FrozenLibraryCertificationConfig(
        recall_k=recall_k,
        retrieval_batch_size=retrieval_batch_size,
        search_execution=search_execution,
    )
    entry_payloads = library.payload["entries"]
    policies = {
        policy_id: dict(payload["calibration_artifact"]["policy"])
        for policy_id, payload in entry_payloads.items()
    }
    candidates, candidate_timing = _evaluate_candidates(
        checkpoint=checkpoint,
        partition=partition,
        workload=certification_workload,
        config=evaluation_config,  # type: ignore[arg-type]
        policies=policies.values(),
        progress_stage="policy_library_certification_candidates",
        reuse_equivalent_probe_plans=True,
    )
    candidate_by_key = {_policy_key(candidate): candidate for candidate in candidates}
    widths = tuple(
        sorted(
            {
                int(payload["reference_fixed_bucket_count"])
                for payload in entry_payloads.values()
            }
        )
    )
    controls, control_timing = plan_decoder_fixed_width_costs(
        checkpoint=checkpoint,
        partition=partition,
        queries=certification_workload.queries,
        fixed_bucket_counts=widths,
        batch_size=retrieval_batch_size,
        progress_stage="policy_library_certification_controls",
    )
    control_by_width = {
        int(point["fixed_bucket_count"]): float(point["mean_ndis"])
        for point in controls
    }
    tolerance = float(contract["certification_tolerance"])
    report_entries: dict[str, dict[str, int | float | str]] = {}
    for policy_id, payload in entry_payloads.items():
        width = int(payload["reference_fixed_bucket_count"])
        candidate = candidate_by_key[_policy_key(policies[policy_id])]
        target = control_by_width[width]
        allowed = target * (1.0 + tolerance)
        observed = float(candidate["mean_ndis"])
        report_entries[policy_id] = {
            "reference_fixed_bucket_count": width,
            "target_mean_ndis": target,
            "relative_tolerance": tolerance,
            "allowed_mean_ndis": allowed,
            "observed_mean_ndis": observed,
            "observed_relative_excess": (observed - target) / target,
            "recall": float(candidate["recall"]),
            "status": "passed" if observed <= allowed + 1e-9 else "failed",
        }
    report = PolicyLibraryCertificationReport.create(
        library=library,
        library_path=library_path,
        checkpoint=checkpoint,
        partition=partition,
        certification_query_ids=[
            int(value) for value in certification_workload.query_ids.tolist()
        ],
        entries=report_entries,
        timing={
            "candidate_evaluation": candidate_timing.to_dict(),
            "fixed_width_cost_planning": control_timing.to_dict(),
            "requested_device": device,
            "resolved_device": resolved_device,
        },
    )
    destination = (
        Path(output_path)
        if output_path is not None
        else Path(library_path).with_name(
            f"{Path(library_path).stem}.certification.json"
        )
    )
    report.save(destination)
    return report


def _certify_candidate(
    *,
    candidate: Mapping[str, int | float],
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    config: CalibrationConfig,
    progress: CalibrationProgressReporter | None = None,
) -> tuple[dict[str, int | float], float, CandidateEvaluationTimingTrace]:
    certifications, timing = _evaluate_candidates(
        checkpoint=checkpoint,
        partition=partition,
        workload=workload,
        config=config,
        policies=(candidate,),
        progress=progress,
        progress_stage="policy_certification_candidates",
    )
    certification = certifications[0]
    lower_bound = recall_lower_confidence_bound(
        observed_recall=float(certification["recall"]),
        sample_count=len(workload.queries),
        confidence_level=config.certification.confidence_level,
    )
    return certification, lower_bound, timing


def calibrate_policy(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    config: CalibrationConfig,
    cost_fit_workload: PolicyCostFitWorkload | None = None,
    certification_workload: CalibrationWorkload | None = None,
    validation_workload: CheckpointValidationWorkload | None = None,
) -> CalibrationArtifact:
    """Select and persist one deployable Budget-NDis or Target-Recall policy."""
    calibration_started_at = utc_now_iso()
    calibration_started = time.perf_counter()
    resolved_device = resolve_calibration_device(config.device)
    checkpoint.model.to(resolved_device)
    progress = CalibrationProgressReporter(
        emit=lambda payload: logger.info(
            "calibration_progress %s", json.dumps(payload, sort_keys=True)
        ),
        requested_device=config.device,
        resolved_device=resolved_device,
        observe_runtime=lambda: calibration_runtime_observations(
            requested_device=config.device, resolved_device=resolved_device
        ),
    )
    logger.info(
        "calibration started dataset=%s objective=%s queries=%d lambda_mode=%s",
        workload.dataset_name,
        config.objective,
        len(workload.queries),
        config.lambda_search.mode,
    )
    if checkpoint.metadata.get("loss", {}).get("name") != (
        "ramps"
    ):
        raise ValueError("adaptive calibration requires a RAMPS checkpoint")
    checkpoint.validate_for(partition)
    _validate_declared_compatibility(
        config, checkpoint=checkpoint, partition=partition
    )
    manifest = WorkloadManifest.load(config.workload_manifest_path)
    if workload.manifest_filename != manifest.filename:
        raise ValueError("Calibration Workload Manifest filename mismatch")
    manifest.validate_checkpoint_binding(checkpoint.metadata)
    if workload.dataset_name != checkpoint.metadata.get("dataset_name"):
        raise ValueError("Calibration Workload dataset mismatch")
    if workload.dataset_name != partition.dataset_name:
        raise ValueError("Calibration Workload partition dataset mismatch")
    if workload.metric != partition.metric:
        raise ValueError("Calibration Workload metric mismatch")
    if workload.queries.shape[1] != partition.dimension:
        raise ValueError("Calibration Workload dimension mismatch")
    if workload.role != config.tuning_role or not np.array_equal(
        workload.query_ids, manifest.role_query_ids(config.tuning_role)
    ):
        raise ValueError("Calibration Workload is not the policy_tuning role")
    common_diagnostics: dict[str, object] = {}
    if config.calibration_bound_selection is not None:
        if validation_workload is None:
            raise ValueError(
                "Calibration Bound selection requires checkpoint_validation"
            )
        if (
            validation_workload.manifest_filename != manifest.filename
            or not np.array_equal(
                validation_workload.query_ids,
                manifest.role_query_ids("checkpoint_validation"),
            )
        ):
            raise ValueError(
                "Calibration Bound selection workload is not checkpoint_validation"
            )
        if (
            validation_workload.queries.shape[1] != partition.dimension
            or config.recall_k > validation_workload.ground_truth.shape[1]
        ):
            raise ValueError("Calibration Bound workload metadata is incompatible")
        bound_points, bound_timing = _evaluate_decoder_fixed_widths(
            checkpoint=checkpoint,
            partition=partition,
            fixed_bucket_counts=tuple(
                config.calibration_bound_selection.fixed_bucket_counts
            ),
            queries=validation_workload.queries,
            ground_truth=validation_workload.ground_truth,
            recall_k=config.recall_k,
            batch_size=config.retrieval_batch_size,
            search_execution=config.search_execution,
            progress=progress,
            progress_stage="calibration_bound_scan",
        )
        try:
            m_max = next(
                int(point["fixed_bucket_count"])
                for point in bound_points
                if float(point["recall"]) + 1e-12
                >= config.calibration_bound_selection.target_recall
            )
        except StopIteration as error:
            raise ValueError(
                "Decoder Fixed-Width scan does not reach target Recall"
            ) from error
        common_diagnostics["calibration_bound_selection"] = {
            "ranking": "decoder_fixed_width",
            "target_recall": config.calibration_bound_selection.target_recall,
            "recall_k": config.recall_k,
            "calibration_bound": m_max,
            "points": bound_points,
        }
        common_diagnostics["calibration_bound_timing"] = bound_timing.to_dict()
    else:
        assert config.m_max is not None
        m_max = config.m_max
    if m_max > partition.bucket_count or config.m_min > m_max:
        raise ValueError("calibration probe bounds do not fit the partition")
    if config.m_max is None:
        resolved_payload = config.model_dump(mode="json")
        resolved_payload["m_max"] = m_max
        resolved_payload["calibration_bound_selection"] = None
        config = CalibrationConfig.model_validate(resolved_payload)
    resolved_m_max = int(m_max)
    if config.recall_k > workload.ground_truth.shape[1]:
        raise ValueError("calibration Recall K exceeds the ground-truth width")
    if certification_workload is None:
        raise ValueError("calibration requires policy_certification")
    if certification_workload.manifest_filename != manifest.filename:
        raise ValueError("Certification Workload Manifest filename mismatch")
    if certification_workload.role != config.certification_role or not np.array_equal(
        certification_workload.query_ids,
        manifest.role_query_ids(config.certification_role),
    ):
        raise ValueError("Certification Workload is not the policy_certification role")
    if (
        certification_workload.dataset_name != workload.dataset_name
        or certification_workload.metric != workload.metric
        or certification_workload.queries.shape[1] != partition.dimension
    ):
        raise ValueError("Certification Workload metadata or structure mismatch")
    if config.recall_k > certification_workload.ground_truth.shape[1]:
        raise ValueError("certification Recall K exceeds the ground-truth width")
    if set(workload.query_ids).intersection(certification_workload.query_ids):
        raise ValueError("policy_tuning and policy_certification role IDs overlap")

    lambda_fits: list[dict[str, int | float]] = []
    cost_fit_timing: dict[str, float | int] | None = None
    if config.lambda_search.mode == "manual":
        policies = [
            {
                "cost_lambda": float(value),
                "m_min": config.m_min,
                "m_max": resolved_m_max,
            }
            for value in config.lambda_search.cost_lambda_candidates
        ]
    else:
        if config.objective != "budget_ndis":
            raise ValueError("automatic lambda fitting only supports budget_ndis")
        if cost_fit_workload is None:
            raise ValueError("automatic lambda fitting requires policy_cost_fit")
        if (
            cost_fit_workload.manifest_filename != manifest.filename
            or cost_fit_workload.role != config.cost_fit_role
            or not np.array_equal(
                cost_fit_workload.query_ids,
                manifest.role_query_ids(config.cost_fit_role),
            )
            or cost_fit_workload.dataset_name != workload.dataset_name
            or cost_fit_workload.metric != workload.metric
            or cost_fit_workload.queries.shape[1] != partition.dimension
        ):
            raise ValueError(
                "Policy Cost-Fit Workload is not the declared policy_cost_fit role"
            )
        cost_table, cost_fit_timing = _collect_cost_fit_table(
            checkpoint=checkpoint,
            partition=partition,
            workload=cost_fit_workload,
            candidate_bound=resolved_m_max,
            batch_size=config.retrieval_batch_size,
            progress=progress,
        )
        lambda_fit_started = time.perf_counter()
        progress.start(stage="lambda_fit", total_batches=1)

        def update_lambda_progress() -> None:
            progress.update(
                completed_batches=0,
                timing_seconds={
                    "lambda_fit_seconds": time.perf_counter()
                    - lambda_fit_started
                },
            )

        fitted = fit_cost_safe_lambda(
            cost_table,
            device=resolved_device,
            target_mean_ndis=config.constraint,
            m_min=config.m_min,
            m_max=resolved_m_max,
            initial_lambda=1.0,
            iterations=config.lambda_search.iterations,
            bracket_max=config.lambda_search.bracket_max,
            progress_callback=update_lambda_progress,
        )
        progress.finish(
            completed_batches=1,
            timing_seconds={
                "lambda_fit_seconds": time.perf_counter()
                - lambda_fit_started
            },
        )
        lambda_fits.append(fitted.to_dict())
        factors = (
            (1.0,)
            if fitted.cost_lambda == 0.0
            else tuple(config.lambda_search.local_factors)
        )
        policies = [
            {
                "cost_lambda": float(fitted.cost_lambda * factor),
                "m_min": config.m_min,
                "m_max": resolved_m_max,
            }
            for factor in factors
        ]
    common_diagnostics["lambda_search"] = {
        **config.lambda_search.model_dump(mode="json"),
        "fits": lambda_fits,
    }
    if cost_fit_timing is not None:
        common_diagnostics["cost_fit_timing"] = cost_fit_timing
    candidates, tuning_timing = _evaluate_candidates(
        checkpoint=checkpoint,
        partition=partition,
        workload=workload,
        config=config,
        policies=policies,
        progress=progress,
        progress_stage="policy_tuning_candidates",
    )
    stable_candidates = [dict(candidate) for candidate in candidates]
    if config.objective == "budget_ndis":
        selected = select_budget_ndis_candidate(
            candidates,
            budget_ndis=config.constraint
            * (1.0 - config.certification_cost_margin),
        )
        certifications, certification_timing = _evaluate_candidates(
            checkpoint=checkpoint,
            partition=partition,
            workload=certification_workload,
            config=config,
            policies=(selected,),
            progress=progress,
            progress_stage="policy_certification_candidates",
        )
        certification = certifications[0]
        if float(certification["mean_ndis"]) > config.constraint + 1e-9:
            raise ValueError("Budget-NDis cost certification failed")
        artifact = CalibrationArtifact.create(
            checkpoint=checkpoint,
            partition=partition,
            cost_lambda=float(selected["cost_lambda"]),
            m_min=int(selected["m_min"]),
            m_max=int(selected["m_max"]),
            recall_k=config.recall_k,
            budget_ndis=config.constraint,
            observed_recall=float(selected["recall"]),
            mean_ndis=float(selected["mean_ndis"]),
            calibration_query_count=len(workload.queries),
            diagnostics={
                **calibration_runtime_observations(
                    requested_device=config.device,
                    resolved_device=resolved_device,
                ),
                "timestamps": {
                    "started_at": calibration_started_at,
                    "computation_finished_at": utc_now_iso(),
                },
                "timing": {
                    "tuning_candidate_evaluation": tuning_timing.to_dict(),
                    "certification_evaluation": certification_timing.to_dict(),
                    "compute_seconds": time.perf_counter() - calibration_started,
                },
                "candidate_count": len(candidates),
                "candidates": stable_candidates,
                **common_diagnostics,
                "certification_cost_margin": config.certification_cost_margin,
                "certification": {
                    "query_count": len(certification_workload.queries),
                    "target_mean_ndis": config.constraint,
                    "observed_mean_ndis": certification["mean_ndis"],
                    "observed_recall": certification["recall"],
                    "parameter_updates": 0,
                },
                "tie_breaking": [
                    "higher_recall",
                    "lower_mean_ndis",
                    "lower_m_max",
                    "higher_cost_lambda",
                ],
            },
            calibration_source=workload.source,
        )
    else:
        fallback_used = False
        fallback_timing: CandidateEvaluationTimingTrace | None = None
        try:
            selected = select_target_recall_candidate(
                candidates, target_recall=config.constraint
            )
        except ValueError:
            if config.fallback == "fail":
                raise
            fallback_used = True
            fallback_candidates, fallback_timing = _evaluate_candidates(
                checkpoint=checkpoint,
                partition=partition,
                workload=workload,
                config=config,
                policies=(_fixed_m_max_policy(config),),
                progress=progress,
                progress_stage="target_recall_fallback_candidates",
            )
            selected = fallback_candidates[0]
        certification, lower_bound, certification_timing = _certify_candidate(
            candidate=selected,
            checkpoint=checkpoint,
            partition=partition,
            workload=certification_workload,
            config=config,
            progress=progress,
        )
        runtime_policy = selected
        if lower_bound + 1e-12 < config.constraint and not fallback_used:
            if config.fallback == "fail":
                raise ValueError("Target-Recall certification lower bound failed")
            fallback_used = True
            runtime_policy = _fixed_m_max_policy(config)
        certified = lower_bound + 1e-12 >= config.constraint
        artifact = CalibrationArtifact.create_target_recall(
            checkpoint=checkpoint,
            partition=partition,
            cost_lambda=float(runtime_policy["cost_lambda"]),
            m_min=int(runtime_policy["m_min"]),
            m_max=int(runtime_policy["m_max"]),
            recall_k=config.recall_k,
            target_recall=config.constraint,
            tuning_observed_recall=float(selected["recall"]),
            observed_recall=float(certification["recall"]),
            mean_ndis=float(selected["mean_ndis"]),
            tuning_query_count=len(workload.queries),
            certification_query_count=len(certification_workload.queries),
            recall_lower_confidence_bound=lower_bound,
            confidence_level=config.certification.confidence_level,
            certified=certified,
            fallback="fixed_m_max" if fallback_used else "none",
            diagnostics={
                **calibration_runtime_observations(
                    requested_device=config.device,
                    resolved_device=resolved_device,
                ),
                "timestamps": {
                    "started_at": calibration_started_at,
                    "computation_finished_at": utc_now_iso(),
                },
                "timing": {
                    "tuning_candidate_evaluation": tuning_timing.to_dict(),
                    "certification_evaluation": certification_timing.to_dict(),
                    "compute_seconds": time.perf_counter() - calibration_started,
                    **(
                        {"fallback_evaluation": fallback_timing.to_dict()}
                        if fallback_timing is not None
                        else {}
                    ),
                },
                "candidate_count": len(candidates),
                "candidates": stable_candidates,
                **common_diagnostics,
                "tie_breaking": [
                    "lower_mean_ndis",
                    "lower_estimated_latency_cost",
                    "lower_m_max",
                    "higher_recall",
                    "lower_cost_lambda",
                ],
                "latency_model": "mean_probe_count_v1",
                "certification_parameter_updates": 0,
                "certification_policy": {
                    name: selected[name]
                    for name in ("cost_lambda", "m_min", "m_max")
                },
                "runtime_policy": {
                    name: runtime_policy[name]
                    for name in ("cost_lambda", "m_min", "m_max")
                },
            },
            calibration_source=workload.source,
        )
    artifact.save(config.output_path)
    logger.info(
        "calibration completed output=%s objective=%s m_max=%d recall=%.8f "
        "mean_ndis=%.6f save_seconds=%.6f persisted_at=%s",
        config.output_path,
        config.objective,
        int(artifact.payload["policy"]["m_max"]),
        float(artifact.payload["selection"]["observed_recall"]),
        float(artifact.payload["selection"]["mean_ndis"]),
        float(artifact.persistence["save_seconds"]),
        artifact.persistence["persisted_at"],
    )
    return artifact
