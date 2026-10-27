"""Machine-readable evidence for comparable Policy Library calibration runs."""

from __future__ import annotations

import hashlib
import os
import platform
import socket
import sys
import time
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch

from ft_anns import __version__
from ft_anns.adaptive_query.artifacts import CalibrationArtifact
from ft_anns.adaptive_query.calibration import (
    CalibrationWorkload,
    LambdaFit,
    _evaluate_candidates,
    _fit_cost_safe_lambda_with_trace,
    fit_cost_safe_lambda,
    select_budget_ndis_candidate,
)
from ft_anns.adaptive_query.evaluation import compute_recall_at_k
from ft_anns.adaptive_query.policy import DecoderPolicyTable
from ft_anns.adaptive_query.policy_library import PolicyLibrary
from ft_anns.adaptive_query.retrieval import retrieve_adaptive
from ft_anns.artifact_io import write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.config import PolicyLibraryConfig
from ft_anns.datasets import WorkloadManifest
from ft_anns.observability import calibration_runtime_observations, utc_now_iso
from ft_anns.partition import FixedPartition

_DEEP1M_EXHAUSTIVE_ROLE_COUNTS = {
    "router_train": 100_000,
    "checkpoint_validation": 10_000,
    "policy_cost_fit": 100_000,
    "policy_tuning": 100_000,
    "policy_certification": 10_000,
}
_FAISS_CHECKED_IN_DEFAULTS = {
    "search_execution": "grouped",
    "retrieval_batch_size": 1_000,
}


@dataclass(frozen=True)
class _BenchmarkCostFitTable:
    bucket_ids: np.ndarray[Any, np.dtype[np.int64]]
    cumulative_mass: np.ndarray[Any, np.dtype[np.float64]]
    cumulative_cost: np.ndarray[Any, np.dtype[np.int64]]

    @property
    def candidate_bound(self) -> int:
        return int(self.cumulative_mass.shape[1])


def _reference_selected_total_ndis(
    table: _BenchmarkCostFitTable,
    *,
    cost_lambda: float,
    m_min: int,
    m_max: int,
) -> int:
    """Frozen pre-ticket-07 selected-cost implementation."""
    first = m_min - 1
    mass = table.cumulative_mass[:, first:m_max]
    cost = table.cumulative_cost[:, first:m_max]
    selected = np.argmax(mass - cost_lambda * cost, axis=1)
    return int(cost[np.arange(len(cost)), selected].sum(dtype=np.int64))


def _reference_fit_cost_safe_lambda(
    table: _BenchmarkCostFitTable,
    *,
    target_mean_ndis: float,
    m_min: int,
    m_max: int,
    initial_lambda: float,
    iterations: int,
    bracket_max: float,
    decision_trace: list[tuple[str, float, int, bool]] | None = None,
) -> LambdaFit:
    """Frozen pre-ticket-07 safe-side search for benchmark comparison."""
    query_count = len(table.cumulative_mass)
    target_total = float(target_mean_ndis) * query_count
    zero_cost = _reference_selected_total_ndis(
        table, cost_lambda=0.0, m_min=m_min, m_max=m_max
    )
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
        upper_cost = _reference_selected_total_ndis(
            table, cost_lambda=upper, m_min=m_min, m_max=m_max
        )
        if decision_trace is not None:
            decision_trace.append(
                (
                    "bracket",
                    upper,
                    upper_cost,
                    upper_cost <= target_total + 1e-9,
                )
            )
        if upper_cost <= target_total + 1e-9:
            break
        lower = upper
        upper *= 2.0
        if upper > bracket_max:
            raise ValueError("unable to bracket a cost-safe lambda")

    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        midpoint_cost = _reference_selected_total_ndis(
            table, cost_lambda=midpoint, m_min=m_min, m_max=m_max
        )
        if decision_trace is not None:
            decision_trace.append(
                (
                    "binary",
                    midpoint,
                    midpoint_cost,
                    midpoint_cost <= target_total + 1e-9,
                )
            )
        if midpoint_cost > target_total + 1e-9:
            lower = midpoint
        else:
            upper = midpoint
    actual_total = _reference_selected_total_ndis(
        table, cost_lambda=upper, m_min=m_min, m_max=m_max
    )
    if decision_trace is not None:
        decision_trace.append(("safe_endpoint", upper, actual_total, True))
    return LambdaFit(
        cost_lambda=upper,
        target_mean_ndis=float(target_mean_ndis),
        actual_mean_ndis=actual_total / query_count,
        m_max=m_max,
        iterations=iterations,
        bracket_lower=lower,
        bracket_upper=upper,
    )


def _run_lambda_fit_schedule(
    table: _BenchmarkCostFitTable,
    *,
    budget_widths: tuple[int, ...],
    caps: tuple[int, ...],
    m_min: int,
    initial_lambda: float,
    iterations: int,
    bracket_max: float,
    reference: bool,
    collect_decisions: bool = False,
) -> tuple[tuple[LambdaFit, ...], tuple[tuple[str, float, int, bool], ...]]:
    results: list[LambdaFit] = []
    schedule_decisions: list[tuple[str, float, int, bool]] = []

    def run_fit(**fit_arguments: Any) -> LambdaFit:
        if reference:
            return _reference_fit_cost_safe_lambda(
                table,
                **fit_arguments,
                decision_trace=(schedule_decisions if collect_decisions else None),
            )
        if collect_decisions:
            fitted, decisions = _fit_cost_safe_lambda_with_trace(table, **fit_arguments)
            schedule_decisions.extend(decisions)
            return fitted
        return fit_cost_safe_lambda(table, **fit_arguments)

    for width in budget_widths:
        target = float(table.cumulative_cost[:, width - 1].mean(dtype=np.float64))
        full_bound_fit = run_fit(
            target_mean_ndis=target,
            m_min=m_min,
            m_max=table.candidate_bound,
            initial_lambda=initial_lambda,
            iterations=iterations,
            bracket_max=bracket_max,
        )
        results.append(full_bound_fit)
        for cap in caps:
            results.append(
                run_fit(
                    target_mean_ndis=target,
                    m_min=m_min,
                    m_max=cap,
                    initial_lambda=full_bound_fit.cost_lambda,
                    iterations=iterations,
                    bracket_max=bracket_max,
                )
            )
    return tuple(results), tuple(schedule_decisions)


def _reference_selected_prefixes(
    table: _BenchmarkCostFitTable, fit: LambdaFit, *, m_min: int
) -> tuple[
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    first = m_min - 1
    utility = (
        table.cumulative_mass[:, first : fit.m_max]
        - fit.cost_lambda * table.cumulative_cost[:, first : fit.m_max]
    )
    probe_counts = np.argmax(utility, axis=1).astype(np.int64) + m_min
    active = np.arange(fit.m_max)[None, :] < probe_counts[:, None]
    return probe_counts, table.bucket_ids[:, : fit.m_max][active]


def _production_selected_prefixes(
    table: DecoderPolicyTable, fit: LambdaFit, *, m_min: int
) -> tuple[
    np.ndarray[Any, np.dtype[np.int64]],
    np.ndarray[Any, np.dtype[np.int64]],
]:
    plan = table.select_prefixes(
        cost_lambda=fit.cost_lambda, m_min=m_min, m_max=fit.m_max
    )
    active = np.arange(fit.m_max)[None, :] < plan.probe_counts[:, None]
    return plan.probe_counts, plan.bucket_ids[active]


def benchmark_lambda_fit_schedule(
    *,
    cumulative_mass: np.ndarray[Any, np.dtype[np.float64]],
    cumulative_cost: np.ndarray[Any, np.dtype[np.int64]],
    bucket_ids: np.ndarray[Any, np.dtype[np.int64]] | None = None,
    budget_widths: tuple[int, ...],
    caps: tuple[int, ...],
    m_min: int,
    initial_lambda: float,
    iterations: int,
    bracket_max: float,
    repeats: int,
    benchmark_kind: str = "synthetic",
) -> dict[str, Any]:
    """Benchmark the complete serial lambda-fit schedule against its reference."""
    mass = np.asarray(cumulative_mass)
    cost = np.asarray(cumulative_cost)
    resolved_bucket_ids = (
        np.broadcast_to(np.arange(mass.shape[1], dtype=np.int64), mass.shape)
        if mass.ndim == 2 and bucket_ids is None
        else np.asarray(bucket_ids)
    )
    if (
        mass.dtype != np.float64
        or cost.dtype != np.int64
        or resolved_bucket_ids.dtype != np.int64
        or mass.ndim != 2
        or cost.shape != mass.shape
        or resolved_bucket_ids.shape != mass.shape
        or not len(mass)
        or not mass.shape[1]
        or not np.isfinite(mass).all()
        or np.any(cost <= 0)
        or not budget_widths
        or not caps
        or any(width < m_min or width > mass.shape[1] for width in budget_widths)
        or any(cap < m_min or cap > mass.shape[1] for cap in caps)
        or repeats < 1
        or benchmark_kind not in {"synthetic", "available_fixture", "real_artifact"}
    ):
        raise ValueError("lambda-fit benchmark inputs are invalid")
    table = _BenchmarkCostFitTable(resolved_bucket_ids, mass, cost)
    arguments = {
        "budget_widths": tuple(int(value) for value in budget_widths),
        "caps": tuple(int(value) for value in caps),
        "m_min": m_min,
        "initial_lambda": initial_lambda,
        "iterations": iterations,
        "bracket_max": bracket_max,
    }
    reference_results, reference_decisions = _run_lambda_fit_schedule(
        table, **arguments, reference=True, collect_decisions=True
    )
    production_results, production_decisions = _run_lambda_fit_schedule(
        table, **arguments, reference=False, collect_decisions=True
    )
    fits_exact = tuple(item.to_dict() for item in reference_results) == tuple(
        item.to_dict() for item in production_results
    )
    decision_trace_exact = reference_decisions == production_decisions
    decoder_table = DecoderPolicyTable(
        bucket_ids=table.bucket_ids,
        cumulative_mass=mass,
        cumulative_cost=cost,
    )
    prefixes_exact = all(
        all(
            np.array_equal(reference_value, production_value)
            for reference_value, production_value in zip(
                _reference_selected_prefixes(table, reference_fit, m_min=m_min),
                _production_selected_prefixes(
                    decoder_table, production_fit, m_min=m_min
                ),
                strict=True,
            )
        )
        for reference_fit, production_fit in zip(
            reference_results, production_results, strict=True
        )
    )
    if not fits_exact or not decision_trace_exact or not prefixes_exact:
        raise AssertionError("lambda-fit benchmark reference parity failed")

    timings: dict[str, list[float]] = {"reference": [], "production": []}
    for _ in range(repeats):
        for name, reference in (("reference", True), ("production", False)):
            started = time.perf_counter()
            _run_lambda_fit_schedule(table, **arguments, reference=reference)
            timings[name].append(time.perf_counter() - started)
    reference_median = float(np.median(timings["reference"]))
    production_median = float(np.median(timings["production"]))
    dominant_workspace_bytes = int(2 * mass.size * np.dtype(np.float64).itemsize)
    return {
        "format": "lambda-fit-benchmark-v1",
        "benchmark_kind": benchmark_kind,
        "inputs": {
            "query_count": len(mass),
            "candidate_bound": mass.shape[1],
            "mass_dtype": mass.dtype.name,
            "cost_dtype": cost.dtype.name,
            "bucket_ids_dtype": resolved_bucket_ids.dtype.name,
        },
        "schedule": {
            "budget_widths": list(budget_widths),
            "caps": list(caps),
            "m_min": m_min,
            "initial_lambda": initial_lambda,
            "iterations": iterations,
            "bracket_max": bracket_max,
            "fit_count": len(budget_widths) * (1 + len(caps)),
            "serial": True,
        },
        "parity": {
            "lambda_fits_exact": fits_exact,
            "decision_trace_exact": decision_trace_exact,
            "decision_count": len(production_decisions),
            "nonzero_fit_count": sum(
                fitted.iterations > 0 for fitted in production_results
            ),
            "bracket_decision_count": sum(
                phase == "bracket" for phase, *_ in production_decisions
            ),
            "binary_decision_count": sum(
                phase == "binary" for phase, *_ in production_decisions
            ),
            "safe_endpoint_count": sum(
                phase == "safe_endpoint" for phase, *_ in production_decisions
            ),
            "per_query_prefixes_exact": prefixes_exact,
            "active_bucket_prefixes_exact": prefixes_exact,
        },
        "temporary_memory_model": {
            "basis": "two full-width float64 matrix temporaries/buffers",
            "reference_dominant_workspace_bytes": dominant_workspace_bytes,
            "production_dominant_workspace_bytes": dominant_workspace_bytes,
            "dominant_workspace_increase_bytes": 0,
            "zero_safe_production_float64_workspace_bytes": 0,
        },
        "timing_seconds": {
            **timings,
            "reference_median": reference_median,
            "production_median": production_median,
            "median_speedup": reference_median / production_median,
        },
    }


def write_representative_lambda_fit_benchmark(
    *,
    report_path: str | Path,
    query_count: int = 2048,
    candidate_bound: int = 256,
    repeats: int = 5,
) -> Path:
    """Write deterministic synthetic before/after lambda-fit evidence."""
    if query_count < 1 or candidate_bound < 2 or repeats < 1:
        raise ValueError("representative lambda-fit benchmark inputs are invalid")
    seed = 20260817
    generator = np.random.default_rng(seed)
    cumulative_mass = np.cumsum(
        generator.random((query_count, candidate_bound), dtype=np.float64),
        axis=1,
        dtype=np.float64,
    )
    cumulative_mass /= cumulative_mass[:, -1, None]
    cumulative_cost = np.cumsum(
        generator.integers(
            1,
            4096,
            size=(query_count, candidate_bound),
            dtype=np.int64,
        ),
        axis=1,
        dtype=np.int64,
    )
    budget_widths = tuple(
        sorted({max(1, candidate_bound // divisor) for divisor in (32, 16, 8, 4, 2, 1)})
    )
    caps = tuple(
        sorted({max(1, candidate_bound // divisor) for divisor in (8, 4, 2, 1)})
    )
    evidence = benchmark_lambda_fit_schedule(
        cumulative_mass=cumulative_mass,
        cumulative_cost=cumulative_cost,
        budget_widths=budget_widths,
        caps=caps,
        m_min=1,
        initial_lambda=1.0,
        iterations=64,
        bracket_max=1e12,
        repeats=repeats,
    )
    evidence["generator"] = {
        "name": "numpy-pcg64-cumulative-table-v1",
        "seed": seed,
        "list_size_lower_inclusive": 1,
        "list_size_upper_exclusive": 4096,
    }
    evidence["host"] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": max(1, os.cpu_count() or 1),
    }
    evidence["software"] = {
        "ft_anns": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
    }
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, evidence)
    return output_path


def _workload_observations(manifest: WorkloadManifest) -> dict[str, Any]:
    metadata = manifest.metadata
    return {
        "manifest_path": str(manifest.path.resolve()),
        "manifest_filename": manifest.filename,
        "dataset_name": metadata.get("dataset_name"),
        "compatibility": deepcopy(metadata.get("compatibility")),
        "compatibility_slug": metadata.get("compatibility_slug"),
        "base_filename": metadata.get("base_filename"),
        "query_source_filename": metadata.get("query_source_filename"),
        "ground_truth": deepcopy(metadata.get("ground_truth")),
        "resolved_config": deepcopy(metadata.get("resolved_config")),
        "role_query_counts": {
            role: int(definition["count"])
            for role, definition in sorted(manifest.metadata["roles"].items())
        },
    }


def _benchmark_kind(workload: dict[str, Any]) -> str:
    if (
        workload["dataset_name"] == "Deep1M"
        and workload["role_query_counts"] == _DEEP1M_EXHAUSTIVE_ROLE_COUNTS
    ):
        return "deep1m_exhaustive"
    return "custom_or_synthetic"


def _policy_library_parity_identity(library: PolicyLibrary) -> dict[str, Any]:
    """Return exact, timing-free calibration results for benchmark comparison."""
    payload = library.payload
    entries = payload["entries"]
    first_entry = next(iter(entries.values()))
    diagnostics = first_entry["calibration_artifact"]["diagnostics"]
    tuning_candidates = sorted(
        deepcopy(diagnostics["tuning_candidates"]),
        key=lambda candidate: (
            float(candidate["cost_lambda"]),
            int(candidate["m_min"]),
            int(candidate["m_max"]),
        ),
    )
    candidate_by_policy = {
        (
            float(candidate["cost_lambda"]),
            int(candidate["m_min"]),
            int(candidate["m_max"]),
        ): candidate
        for candidate in tuning_candidates
        if candidate["status"] == "recall_evaluated"
    }
    selected_policies: dict[str, Any] = {}
    for policy_id in sorted(entries):
        entry = entries[policy_id]
        artifact = entry["calibration_artifact"]
        policy = artifact["policy"]
        selected_candidate = candidate_by_policy[
            (
                float(policy["cost_lambda"]),
                int(policy["m_min"]),
                int(policy["m_max"]),
            )
        ]
        selected_policies[policy_id] = {
            "reference_fixed_bucket_count": entry["reference_fixed_bucket_count"],
            "cost_fit_target_mean_ndis": entry["cost_fit_target_mean_ndis"],
            "tuning_target_mean_ndis": entry["tuning_target_mean_ndis"],
            "certification_status": entry["certification_status"],
            "policy": deepcopy(policy),
            "selection": deepcopy(artifact["selection"]),
            "mean_probe_count": selected_candidate["mean_probe_count"],
        }
    return {
        "calibration_bound": diagnostics["calibration_bound_selection"][
            "calibration_bound"
        ],
        "calibration_bound_points": deepcopy(
            diagnostics["calibration_bound_selection"]["points"]
        ),
        "tuning_candidates": tuning_candidates,
        "selected_policies": selected_policies,
    }


def _policy_library_source_identity(library: PolicyLibrary) -> dict[str, Any]:
    """Bind reusable reports to the validated source Policy Library."""
    payload = library.payload
    return {
        "library_filename": payload["library_filename"],
        "compatibility": deepcopy(payload["compatibility"]),
        "checkpoint_filename": payload["checkpoint_filename"],
        "checkpoint_compatibility": deepcopy(payload["checkpoint_compatibility"]),
        "partition_filename": payload["partition_filename"],
        "partition_compatibility": deepcopy(payload["partition_compatibility"]),
        "calibration_contract": deepcopy(payload["calibration_contract"]),
        "selected_policies": {
            policy_id: deepcopy(entry["calibration_artifact"]["policy"])
            for policy_id, entry in sorted(payload["entries"].items())
        },
    }


def summarize_faiss_execution_benchmarks(
    reports: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare common reports and identify only a strict repeated winner."""
    resolved_reports = list(reports)
    if not resolved_reports:
        raise ValueError("FAISS benchmark summary requires reports")
    reference_identity = resolved_reports[0].get("result", {}).get("parity_identity")
    if reference_identity is None:
        raise ValueError("FAISS benchmark report parity identity is missing")
    if any(
        report.get("format")
        not in {
            "calibrate-library-benchmark-v1",
            "faiss-tuning-recall-benchmark-v1",
        }
        or report.get("status") != "complete"
        for report in resolved_reports
    ):
        raise ValueError("FAISS benchmark report is incomplete or unsupported")
    if any(
        report.get("result", {}).get("parity_identity") != reference_identity
        for report in resolved_reports[1:]
    ):
        raise AssertionError("FAISS benchmark calibration result parity failed")
    reference_policy_library_identity_exact = all(
        bool(report.get("result", {}).get("reference_parity_exact", True))
        for report in resolved_reports
    )

    cells: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for report in resolved_reports:
        config = report["inputs"]["resolved_config"]
        mode = str(config["search_execution"])
        batch_size = int(config["retrieval_batch_size"])
        repeat = int(report.get("matrix_repeat", 0))
        timing = report["observability"]["stage_timing_seconds"]
        resources = report["observability"]["resources"]
        sample = {
            "repeat": repeat,
            "router_seconds": float(timing["router_seconds"]),
            "planning_seconds": float(timing["policy_table_seconds"])
            + float(timing["cost_planning_seconds"]),
            "packing_seconds": float(timing["packing_seconds"]),
            "faiss_seconds": float(timing["faiss_seconds"]),
            "metric_seconds": float(timing["metric_seconds"]),
            "total_seconds": float(timing["total_seconds"]),
            "faiss_call_count": int(
                report["observability"]["counts"]["faiss_call_count"]
            ),
            "peak_rss_bytes": int(resources["process_peak_rss_bytes"]),
            "peak_vram_bytes": resources["cuda_peak_allocated_bytes"],
            "resource_measurement_scope": report.get(
                "resource_measurement_scope",
                "matrix_process_cumulative_high_water",
            ),
        }
        cells.setdefault((mode, batch_size), []).append(sample)
    for samples in cells.values():
        samples.sort(key=lambda sample: sample["repeat"])
        if len({sample["repeat"] for sample in samples}) != len(samples):
            raise ValueError("FAISS benchmark cell repeats must be unique")

    matrix = []
    for (mode, batch_size), samples in sorted(cells.items()):
        faiss_path = [
            sample["packing_seconds"] + sample["faiss_seconds"] for sample in samples
        ]
        matrix.append(
            {
                "search_execution": mode,
                "retrieval_batch_size": batch_size,
                "samples": samples,
                "median_faiss_and_packing_seconds": float(np.median(faiss_path)),
                "median_total_seconds": float(
                    np.median([sample["total_seconds"] for sample in samples])
                ),
            }
        )

    repeat_sets = [{sample["repeat"] for sample in cell["samples"]} for cell in matrix]
    common_repeats = set.intersection(*repeat_sets)
    repeatable_winner = None
    if len(matrix) > 1 and len(common_repeats) >= 3:
        winners: list[tuple[str, int]] = []

        def strict_minimum(
            values: Mapping[tuple[str, int], float],
        ) -> tuple[str, int] | None:
            ordered = sorted(values.items(), key=lambda item: item[1])
            if len(ordered) < 2 or ordered[0][1] >= ordered[1][1]:
                return None
            return ordered[0][0]

        for repeat in sorted(common_repeats):
            by_cell = {
                (cell["search_execution"], cell["retrieval_batch_size"]): next(
                    sample for sample in cell["samples"] if sample["repeat"] == repeat
                )
                for cell in matrix
            }
            total_winner = strict_minimum(
                {key: sample["total_seconds"] for key, sample in by_cell.items()}
            )
            faiss_winner = strict_minimum(
                {
                    key: sample["packing_seconds"] + sample["faiss_seconds"]
                    for key, sample in by_cell.items()
                }
            )
            if total_winner is None or total_winner != faiss_winner:
                winners = []
                break
            winners.append(total_winner)
        if winners and len(set(winners)) == 1:
            mode, batch_size = winners[0]
            repeatable_winner = {
                "search_execution": mode,
                "retrieval_batch_size": batch_size,
            }

    resource_scopes = {
        sample["resource_measurement_scope"]
        for cell in matrix
        for sample in cell["samples"]
    }
    resource_measurement_scope = (
        next(iter(resource_scopes)) if len(resource_scopes) == 1 else "mixed"
    )
    return {
        "format": "faiss-execution-benchmark-matrix-v1",
        "parity": {
            "exact": True,
            "cross_cell_exact": True,
            "cross_cell_calibration_result_identity_exact": True,
            "reference_policy_library_identity_exact": (
                reference_policy_library_identity_exact
            ),
        },
        "resource_measurement_scope": resource_measurement_scope,
        "matrix": matrix,
        "decision": {
            "criterion": (
                "same cell strictly minimizes total and FAISS-plus-packing "
                "in every one of at least three repeats"
            ),
            "repeatable_winner": repeatable_winner,
            "checked_in_defaults": deepcopy(_FAISS_CHECKED_IN_DEFAULTS),
            "recommended_defaults": deepcopy(
                repeatable_winner or _FAISS_CHECKED_IN_DEFAULTS
            ),
            "default_change_recommended": (
                repeatable_winner is not None
                and repeatable_winner != _FAISS_CHECKED_IN_DEFAULTS
            ),
            "retain_existing_defaults": (
                repeatable_winner is None
                or repeatable_winner == _FAISS_CHECKED_IN_DEFAULTS
            ),
        },
    }


def verify_faiss_execution_parity(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    library: PolicyLibrary,
    batch_sizes: tuple[int, ...],
    query_count: int,
    topk: int,
) -> dict[str, Any]:
    """Compare grouped/padded selected-policy retrieval results exactly."""
    return _verify_faiss_execution_parity(
        checkpoint=checkpoint,
        partition=partition,
        workload=workload,
        retrieval_policies=[
            (
                policy_id,
                {"policy_library": library, "policy_id": policy_id},
            )
            for policy_id in library.policy_ids
        ],
        batch_sizes=batch_sizes,
        query_count=query_count,
        topk=topk,
    )


def verify_faiss_execution_artifact_parity(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    artifact: CalibrationArtifact,
    batch_sizes: tuple[int, ...],
    query_count: int,
    topk: int,
) -> dict[str, Any]:
    """Compare grouped/padded retrieval for one compatible sample artifact."""
    return _verify_faiss_execution_parity(
        checkpoint=checkpoint,
        partition=partition,
        workload=workload,
        retrieval_policies=[("representative-artifact", {"artifact": artifact})],
        batch_sizes=batch_sizes,
        query_count=query_count,
        topk=topk,
    )


def _verify_faiss_execution_parity(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    retrieval_policies: list[tuple[str, dict[str, Any]]],
    batch_sizes: tuple[int, ...],
    query_count: int,
    topk: int,
) -> dict[str, Any]:
    if (
        not retrieval_policies
        or not batch_sizes
        or any(batch_size < 1 for batch_size in batch_sizes)
        or query_count < 1
        or query_count > len(workload.queries)
        or topk < 1
        or topk > workload.ground_truth.shape[1]
    ):
        raise ValueError("FAISS execution parity inputs are invalid")
    queries = workload.queries[:query_count]
    ground_truth = workload.ground_truth[:query_count]
    comparisons: list[dict[str, Any]] = []
    exact_fields = {
        "neighbor_ids_exact": True,
        "recall_exact": True,
        "ndis_exact": True,
        "probe_counts_exact": True,
    }
    for policy_id, retrieval_policy in retrieval_policies:
        reference = None
        reference_recall = None
        for batch_size in batch_sizes:
            for mode in ("grouped", "padded"):
                result = retrieve_adaptive(
                    queries=queries,
                    checkpoint=checkpoint,
                    partition=partition,
                    topk=topk,
                    batch_size=batch_size,
                    query_filename=f"{workload.role}-sample",
                    search_execution=mode,
                    **retrieval_policy,
                )
                recall = compute_recall_at_k(result.neighbor_ids, ground_truth, k=topk)
                if reference is None:
                    reference = result
                    reference_recall = recall
                assert reference_recall is not None
                comparison = {
                    "policy_id": policy_id,
                    "search_execution": mode,
                    "retrieval_batch_size": batch_size,
                    "neighbor_ids_exact": bool(
                        np.array_equal(result.neighbor_ids, reference.neighbor_ids)
                    ),
                    "recall": recall,
                    "recall_exact": recall == reference_recall,
                    "ndis_exact": bool(np.array_equal(result.ndis, reference.ndis)),
                    "probe_counts_exact": bool(
                        np.array_equal(result.probe_counts, reference.probe_counts)
                    ),
                    "timing": result.timing.to_dict(),
                }
                comparisons.append(comparison)
                for name, is_exact in exact_fields.items():
                    exact_fields[name] = is_exact and comparison[name]
    return {
        "format": "faiss-execution-parity-v1",
        "dataset_name": workload.dataset_name,
        "workload_role": workload.role,
        "query_count": query_count,
        "topk": topk,
        "batch_sizes": list(batch_sizes),
        "policy_ids": [policy_id for policy_id, _ in retrieval_policies],
        "single_faiss_caller": True,
        **exact_fields,
        "exact": all(exact_fields.values()),
        "comparisons": comparisons,
    }


def benchmark_tuning_recall_execution(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    workload: CalibrationWorkload,
    config: PolicyLibraryConfig,
    library: PolicyLibrary,
    manifest: WorkloadManifest,
) -> dict[str, Any]:
    """Replay only the exact post-pruning tuning Recall stage for one cell."""
    started_at = utc_now_iso()
    started = time.perf_counter()
    reference_identity = _policy_library_parity_identity(library)
    reference_candidates = reference_identity["tuning_candidates"]
    feasible_policies = [
        {
            "cost_lambda": candidate["cost_lambda"],
            "m_min": candidate["m_min"],
            "m_max": candidate["m_max"],
        }
        for candidate in reference_candidates
        if candidate["status"] == "recall_evaluated"
    ]
    candidates, timing = _evaluate_candidates(
        checkpoint=checkpoint,
        partition=partition,
        workload=workload,
        config=config,
        policies=feasible_policies,
    )

    def policy_key(candidate: Mapping[str, Any]) -> tuple[float, int, int]:
        return (
            float(candidate["cost_lambda"]),
            int(candidate["m_min"]),
            int(candidate["m_max"]),
        )

    candidate_by_key = {policy_key(candidate): candidate for candidate in candidates}
    tuning_candidates = deepcopy(reference_candidates)
    for diagnostic in tuning_candidates:
        if diagnostic["status"] == "recall_evaluated":
            diagnostic.update(candidate_by_key[policy_key(diagnostic)])
    library_entries = library.payload["entries"]
    selected_policies: dict[str, Any] = {}
    for policy_id in sorted(library_entries):
        entry = library_entries[policy_id]
        available = [
            candidate_by_key[policy_key(candidate)]
            for candidate in tuning_candidates
            if candidate["status"] == "recall_evaluated"
            and policy_id in candidate["referencing_policy_ids"]
        ]
        selected = select_budget_ndis_candidate(
            available,
            budget_ndis=float(entry["tuning_target_mean_ndis"])
            * (1.0 - config.certification_cost_margin),
        )
        artifact = entry["calibration_artifact"]
        selection = deepcopy(artifact["selection"])
        selection["observed_recall"] = selected["recall"]
        selection["mean_ndis"] = selected["mean_ndis"]
        selection["tuning_query_count"] = len(workload.queries)
        selected_policies[policy_id] = {
            "reference_fixed_bucket_count": entry["reference_fixed_bucket_count"],
            "cost_fit_target_mean_ndis": entry["cost_fit_target_mean_ndis"],
            "tuning_target_mean_ndis": entry["tuning_target_mean_ndis"],
            "certification_status": entry["certification_status"],
            "policy": {
                name: selected[name] for name in ("cost_lambda", "m_min", "m_max")
            },
            "selection": selection,
            "mean_probe_count": selected["mean_probe_count"],
        }
    parity_identity = {
        **reference_identity,
        "tuning_candidates": tuning_candidates,
        "selected_policies": selected_policies,
    }
    runtime = calibration_runtime_observations(
        requested_device="cpu", resolved_device="cpu"
    )
    workload_observations = _workload_observations(manifest)
    benchmark_kind = _benchmark_kind(workload_observations)
    manifest_tuning_count = workload_observations["role_query_counts"].get(
        config.tuning_role
    )
    if manifest_tuning_count != len(workload.queries):
        benchmark_kind = "deep1m_real_frozen_sample"
    stage_timing = {
        "router_seconds": timing.router_seconds,
        "policy_table_seconds": timing.policy_table_seconds,
        "cost_planning_seconds": timing.probe_plan_seconds,
        "packing_seconds": timing.search_packing_seconds,
        "faiss_seconds": timing.search_seconds,
        "metric_seconds": timing.metric_seconds or 0.0,
        "total_seconds": timing.total_seconds,
    }
    return {
        "format": "faiss-tuning-recall-benchmark-v1",
        "status": "complete",
        "benchmark_kind": benchmark_kind,
        "timestamps": {
            "started_at": started_at,
            "finished_at": utc_now_iso(),
        },
        "inputs": {
            "resolved_config": config.model_dump(mode="json"),
            "checkpoint_filename": checkpoint.filename,
            "checkpoint_compatibility": deepcopy(
                checkpoint.metadata.get("compatibility")
            ),
            "partition_filename": partition.filename,
            "partition_compatibility": {
                "dataset_name": partition.dataset_name,
                "bucket_count": partition.bucket_count,
                "kmeans_niter": partition.kmeans_niter,
                "partition_build_seed": partition.build_seed,
            },
            "reference_policy_library": _policy_library_source_identity(library),
            "workload": workload_observations,
            "tuning_query_count": len(workload.queries),
            "tuning_query_ids_sha256": hashlib.sha256(
                np.ascontiguousarray(workload.query_ids).tobytes()
            ).hexdigest(),
            "feasible_candidate_count": len(feasible_policies),
        },
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_count": max(1, os.cpu_count() or 1),
        },
        "software": {
            "ft_anns": __version__,
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "faiss": getattr(faiss, "__version__", "unknown"),
        },
        "timing_seconds": {"wall": time.perf_counter() - started},
        "observability": {
            **runtime,
            "counts": {
                "faiss_call_count": timing.faiss_call_count,
                "logical_candidate_plan_evaluation_count": (
                    timing.logical_candidate_plan_evaluation_count
                ),
                "unique_active_probe_plan_count": (
                    timing.unique_active_probe_plan_count
                ),
                "reused_active_probe_plan_count": (
                    timing.reused_active_probe_plan_count
                ),
                "avoided_faiss_search_count": timing.avoided_faiss_search_count,
            },
            "stage_timing_seconds": stage_timing,
        },
        "result": {
            "parity_identity": parity_identity,
            "reference_parity_exact": parity_identity == reference_identity,
        },
    }


def write_calibrate_library_benchmark_report(
    *,
    report_path: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
    partition_path: str | Path,
    started_at: str,
    finished_at: str,
    wall_seconds: float,
    config: PolicyLibraryConfig,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    manifest: WorkloadManifest,
    library: PolicyLibrary,
) -> Path:
    """Write one self-describing calibration benchmark report atomically."""
    workload = _workload_observations(manifest)
    library_payload = library.payload
    output_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_path,
        {
            "format": "calibrate-library-benchmark-v1",
            "status": "complete",
            "benchmark_kind": _benchmark_kind(workload),
            "timestamps": {
                "started_at": started_at,
                "finished_at": finished_at,
            },
            "inputs": {
                "config_path": str(Path(config_path).resolve()),
                "resolved_config": config.model_dump(mode="json"),
                "checkpoint_path": str(Path(checkpoint_path).resolve()),
                "checkpoint_filename": checkpoint.filename,
                "checkpoint_compatibility": checkpoint.metadata.get("compatibility"),
                "partition_path": str(Path(partition_path).resolve()),
                "partition_filename": partition.filename,
                "partition_compatibility": {
                    "dataset_name": partition.dataset_name,
                    "bucket_count": partition.bucket_count,
                    "kmeans_niter": partition.kmeans_niter,
                    "partition_build_seed": partition.build_seed,
                },
                "workload": workload,
            },
            "host": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "cpu_count": max(1, os.cpu_count() or 1),
            },
            "software": {
                "ft_anns": __version__,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "torch": torch.__version__,
                "faiss": getattr(faiss, "__version__", "unknown"),
            },
            "timing_seconds": {
                "wall": wall_seconds,
                "calibration_compute": library_payload["observability"]["timing"][
                    "compute_seconds"
                ],
                "policy_library_save": library.persistence["save_seconds"],
            },
            "observability": library_payload["observability"],
            "result": {
                "policy_library_path": config.output_path,
                "policy_library_filename": library.filename,
                "policy_ids": library.policy_ids,
                "parity_identity": _policy_library_parity_identity(library),
                "persistence": library.persistence,
            },
        },
    )
    return output_path
