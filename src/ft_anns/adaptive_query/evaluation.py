"""Offline metrics over cached retrieval results and ground truth."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ft_anns.adaptive_query.retrieval import RetrievalResult
from ft_anns.artifact_io import artifact_filename, write_json_atomic
from ft_anns.observability import SearchTimingTrace, utc_now_iso

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationTimingTrace:
    recall_seconds: float
    summary_seconds: float
    total_seconds: float

    def __post_init__(self) -> None:
        if (
            self.recall_seconds < 0
            or self.summary_seconds < 0
            or self.total_seconds + 1e-12
            < self.recall_seconds + self.summary_seconds
        ):
            raise ValueError("evaluation timing values are inconsistent")

    def to_dict(self) -> dict[str, float]:
        return {
            "recall_seconds": self.recall_seconds,
            "summary_seconds": self.summary_seconds,
            "total_seconds": self.total_seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "EvaluationTimingTrace":
        return cls(
            recall_seconds=float(payload["recall_seconds"]),
            summary_seconds=float(payload["summary_seconds"]),
            total_seconds=float(payload["total_seconds"]),
        )


def compute_recall_at_k(
    neighbor_ids: npt.ArrayLike,
    ground_truth: npt.ArrayLike,
    *,
    k: int,
) -> float:
    """Compute aggregate Recall@K while treating negative IDs as misses."""
    predicted = np.asarray(neighbor_ids)
    return count_recall_hits_at_k(predicted, ground_truth, k=k) / (
        len(predicted) * k
    )


def count_recall_hits_at_k(
    neighbor_ids: npt.ArrayLike,
    ground_truth: npt.ArrayLike,
    *,
    k: int,
) -> int:
    """Count Recall@K hits exactly so streaming aggregation stays integral."""
    predicted = np.asarray(neighbor_ids)
    expected = np.asarray(ground_truth)
    if (
        predicted.ndim != 2
        or expected.ndim != 2
        or len(predicted) != len(expected)
        or not len(predicted)
        or k <= 0
        or k > predicted.shape[1]
        or k > expected.shape[1]
    ):
        raise ValueError("retrieval IDs, ground truth, and Recall K are incompatible")
    predicted = predicted[:, :k]
    expected = expected[:, :k]
    matches = (predicted[:, :, None] == expected[:, None, :]) & (
        predicted[:, :, None] >= 0
    )
    return int(matches.any(axis=1).sum(dtype=np.int64))


def compute_tie_aware_recall_at_k(
    neighbor_ids: npt.ArrayLike,
    ground_truth: npt.ArrayLike,
    *,
    queries: npt.ArrayLike,
    base_vectors: npt.ArrayLike,
    metric: str,
    k: int,
) -> float:
    """Count unique retrieved IDs whose distance is within the exact K boundary."""
    predicted = np.asarray(neighbor_ids)
    expected = np.asarray(ground_truth)
    query_matrix = np.asarray(queries)
    base_matrix = np.asarray(base_vectors)
    if (
        predicted.ndim != 2
        or expected.ndim != 2
        or predicted.dtype.kind not in ("i", "u")
        or expected.dtype.kind not in ("i", "u")
        or query_matrix.ndim != 2
        or base_matrix.ndim != 2
        or not len(predicted)
        or not (len(predicted) == len(expected) == len(query_matrix))
        or query_matrix.shape[1] != base_matrix.shape[1]
        or k <= 0
        or k > predicted.shape[1]
        or k > expected.shape[1]
    ):
        raise ValueError(
            "retrieval IDs, ground truth, queries, base vectors, and Recall K "
            "are incompatible"
        )
    if metric not in ("l2", "ip"):
        raise ValueError("tie-aware Recall metric must be 'l2' or 'ip'")
    expected_prefix = expected[:, :k]
    if expected_prefix.size and (
        int(expected_prefix.min()) < 0
        or int(expected_prefix.max()) >= len(base_matrix)
    ):
        raise ValueError("ground truth contains an ID outside the base range")

    matched = 0
    batch_size = 1024
    for start in range(0, len(predicted), batch_size):
        stop = min(start + batch_size, len(predicted))
        query_batch = np.asarray(query_matrix[start:stop], dtype=np.float64)
        predicted_batch = predicted[start:stop, :k]
        valid = (predicted_batch >= 0) & (predicted_batch < len(base_matrix))
        safe_ids = np.where(valid, predicted_batch, 0)
        candidate_vectors = np.asarray(base_matrix[safe_ids], dtype=np.float64)
        boundary_vectors = np.asarray(
            base_matrix[expected[start:stop, k - 1]], dtype=np.float64
        )
        if metric == "l2":
            candidate_distances = np.sum(
                (candidate_vectors - query_batch[:, None, :]) ** 2, axis=2
            )
            boundary_distances = np.sum(
                (boundary_vectors - query_batch) ** 2, axis=1
            )
            within_boundary = candidate_distances <= boundary_distances[:, None]
            within_boundary |= np.isclose(
                candidate_distances,
                boundary_distances[:, None],
                rtol=1e-6,
                atol=1e-12,
            )
        else:
            candidate_distances = np.sum(
                candidate_vectors * query_batch[:, None, :], axis=2
            )
            boundary_distances = np.sum(boundary_vectors * query_batch, axis=1)
            within_boundary = candidate_distances >= boundary_distances[:, None]
            within_boundary |= np.isclose(
                candidate_distances,
                boundary_distances[:, None],
                rtol=1e-6,
                atol=1e-12,
            )
        unique = np.ones(predicted_batch.shape, dtype=bool)
        for column in range(1, k):
            unique[:, column] = ~np.any(
                predicted_batch[:, column, None] == predicted_batch[:, :column],
                axis=1,
            )
        matched += int(np.sum(valid & unique & within_boundary))
    return matched / (len(predicted) * k)


@dataclass(frozen=True)
class EvaluationReport:
    query_count: int
    recall_at_k: dict[int, float]
    total_probe_count: int
    mean_probe_count: float
    min_probe_count: int
    max_probe_count: int
    total_ndis: int
    mean_ndis: float
    total_latency_seconds: float
    mean_latency_seconds: float
    qps: float
    input_artifacts: dict[str, str]
    experiment_coordinates: dict[str, Any]
    evaluation_timing: EvaluationTimingTrace
    evaluation_timestamps: dict[str, str]
    retrieval_timing: dict[str, Any] = field(default_factory=dict)
    retrieval_timestamps: dict[str, str | None] = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "retrieval-evaluation-v3",
            "query_count": self.query_count,
            "recall_at_k": {str(k): value for k, value in self.recall_at_k.items()},
            "probe_counts": {
                "total": self.total_probe_count,
                "mean": self.mean_probe_count,
                "min": self.min_probe_count,
                "max": self.max_probe_count,
            },
            "total_ndis": self.total_ndis,
            "mean_ndis": self.mean_ndis,
            "total_latency_seconds": self.total_latency_seconds,
            "mean_latency_seconds": self.mean_latency_seconds,
            "qps": self.qps,
            "evaluation_timing": self.evaluation_timing.to_dict(),
            "evaluation_timestamps": self.evaluation_timestamps,
            "retrieval_timing": self.retrieval_timing,
            "retrieval_timestamps": self.retrieval_timestamps,
            "input_artifacts": self.input_artifacts,
            "experiment_coordinates": self.experiment_coordinates,
        }
        write_json_atomic(output_path, payload)

    @classmethod
    def load(cls, path: str | Path) -> "EvaluationReport":
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format") != "retrieval-evaluation-v3":
            raise ValueError("unsupported retrieval evaluation format")
        probe_counts = payload.get("probe_counts")
        if not isinstance(probe_counts, dict):
            raise ValueError("retrieval evaluation probe-count summary is invalid")
        return cls(
            query_count=int(payload["query_count"]),
            recall_at_k={
                int(k): float(value) for k, value in payload["recall_at_k"].items()
            },
            total_probe_count=int(probe_counts["total"]),
            mean_probe_count=float(probe_counts["mean"]),
            min_probe_count=int(probe_counts["min"]),
            max_probe_count=int(probe_counts["max"]),
            total_ndis=int(payload["total_ndis"]),
            mean_ndis=float(payload["mean_ndis"]),
            total_latency_seconds=float(payload["total_latency_seconds"]),
            mean_latency_seconds=float(payload["mean_latency_seconds"]),
            qps=float(payload["qps"]),
            evaluation_timing=EvaluationTimingTrace.from_dict(
                payload["evaluation_timing"]
            ),
            evaluation_timestamps={
                str(name): str(value)
                for name, value in payload["evaluation_timestamps"].items()
            },
            input_artifacts={
                str(name): str(identity)
                for name, identity in payload["input_artifacts"].items()
            },
            experiment_coordinates=dict(payload.get("experiment_coordinates", {})),
            retrieval_timing=SearchTimingTrace.from_dict(
                payload.get(
                    "retrieval_timing",
                    {
                        "router_seconds": 0.0,
                        "search_seconds": payload["total_latency_seconds"],
                        "total_seconds": payload["total_latency_seconds"],
                    },
                )
            ).to_dict(),
            retrieval_timestamps={
                str(name): (str(value) if value is not None else None)
                for name, value in payload.get("retrieval_timestamps", {}).items()
            },
        )


def evaluate_results(
    *,
    retrieval_result: RetrievalResult,
    ground_truth: npt.ArrayLike,
    recall_ks: tuple[int, ...],
    ground_truth_filename: str = "ground-truth",
) -> EvaluationReport:
    """Compute experiment-only metrics without invoking Policy or Retrieval."""
    evaluation_started_at = utc_now_iso()
    evaluation_started = time.perf_counter()
    truth = np.asarray(ground_truth)
    if truth.ndim != 2 or truth.dtype.kind not in ("i", "u"):
        raise ValueError("ground truth must be an integer matrix")
    if len(truth) != len(retrieval_result.neighbor_ids):
        raise ValueError("ground truth and retrieval result query counts differ")
    if not len(truth):
        raise ValueError("evaluation requires at least one query")
    if not recall_ks or len(set(recall_ks)) != len(recall_ks):
        raise ValueError("recall_ks must be non-empty and unique")
    maximum_k = min(truth.shape[1], retrieval_result.neighbor_ids.shape[1])
    if any(k <= 0 or k > maximum_k for k in recall_ks):
        raise ValueError("each Recall K must fit both results and ground truth")

    recall_started = time.perf_counter()
    recall_at_k: dict[int, float] = {}
    for k in recall_ks:
        recall_at_k[k] = compute_recall_at_k(
            retrieval_result.neighbor_ids, truth, k=k
        )
    recall_seconds = time.perf_counter() - recall_started

    summary_started = time.perf_counter()
    query_count = len(truth)
    total_probe_count = int(retrieval_result.probe_counts.sum(dtype=np.int64))
    total_ndis = int(retrieval_result.ndis.sum(dtype=np.int64))
    total_latency = retrieval_result.timing.total_seconds
    input_artifacts = {
        "checkpoint": retrieval_result.checkpoint_filename,
        "partition": retrieval_result.partition_filename,
        "queries": retrieval_result.query_filename,
        "retrieval_result": retrieval_result.filename,
        "ground_truth": artifact_filename(ground_truth_filename),
    }
    if retrieval_result.calibration_filename is not None:
        input_artifacts["calibration"] = retrieval_result.calibration_filename
    if retrieval_result.policy_library_filename is not None:
        input_artifacts[
            "policy_library"
        ] = retrieval_result.policy_library_filename
    summary_seconds = time.perf_counter() - summary_started
    evaluation_finished_at = utc_now_iso()
    evaluation_timing = EvaluationTimingTrace(
        recall_seconds=recall_seconds,
        summary_seconds=summary_seconds,
        total_seconds=time.perf_counter() - evaluation_started,
    )
    report = EvaluationReport(
        query_count=query_count,
        recall_at_k=recall_at_k,
        total_probe_count=total_probe_count,
        mean_probe_count=total_probe_count / query_count,
        min_probe_count=int(retrieval_result.probe_counts.min()),
        max_probe_count=int(retrieval_result.probe_counts.max()),
        total_ndis=total_ndis,
        mean_ndis=total_ndis / query_count,
        total_latency_seconds=total_latency,
        mean_latency_seconds=total_latency / query_count,
        qps=query_count / total_latency if total_latency > 0 else 0.0,
        input_artifacts=input_artifacts,
        experiment_coordinates=dict(retrieval_result.experiment_coordinates or {}),
        evaluation_timing=evaluation_timing,
        evaluation_timestamps={
            "started_at": evaluation_started_at,
            "finished_at": evaluation_finished_at,
        },
        retrieval_timing=retrieval_result.timing.to_dict(),
        retrieval_timestamps={
            "started_at": retrieval_result.started_at,
            "finished_at": retrieval_result.finished_at,
        },
    )
    logger.info(
        "evaluation completed queries=%d recall=%s mean_ndis=%.6f qps=%.6f "
        "recall_seconds=%.6f summary_seconds=%.6f total_seconds=%.6f",
        report.query_count,
        report.recall_at_k,
        report.mean_ndis,
        report.qps,
        report.evaluation_timing.recall_seconds,
        report.evaluation_timing.summary_seconds,
        report.evaluation_timing.total_seconds,
    )
    return report
