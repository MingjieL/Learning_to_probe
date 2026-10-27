"""True ANN checkpoint validation over a frozen Fixed Partition."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from ft_anns.adaptive_query.evaluation import compute_recall_at_k
from ft_anns.adaptive_query.policy import build_ramps_decoder_fixed_width_plan
from ft_anns.artifact_io import is_artifact_filename, readonly_array
from ft_anns.datasets.base_gt import FileVectorSource
from ft_anns.datasets.registry import dataset_spec
from ft_anns.datasets.workloads import WorkloadManifest, load_role_ground_truth
from ft_anns.domain import RoutingRanking
from ft_anns.observability import SearchTimingTrace
from ft_anns.partition import FixedPartition, ProbePlan


@dataclass(frozen=True)
class CheckpointValidationWorkload:
    queries: npt.NDArray[np.float32]
    ground_truth: npt.NDArray[np.int64]
    query_ids: npt.NDArray[np.int64]
    manifest_filename: str
    ground_truth_artifact_filename: str

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
            raise ValueError("checkpoint validation workload arrays are incompatible")
        if not np.isfinite(queries).all() or np.any(ground_truth < 0):
            raise ValueError("checkpoint validation workload contents are invalid")
        for name, value in (
            ("manifest", self.manifest_filename),
            ("ground truth", self.ground_truth_artifact_filename),
        ):
            if not is_artifact_filename(value):
                raise ValueError(
                    f"checkpoint validation {name} filename is invalid"
                )
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "ground_truth", ground_truth)
        object.__setattr__(self, "query_ids", query_ids)

    @property
    def content_kind(self) -> str:
        return self.manifest_filename


@dataclass(frozen=True)
class CheckpointValidationReport:
    query_count: int
    fixed_bucket_count: int
    ranking: RoutingRanking
    ann_recall_at_k: dict[int, float]
    total_probe_count: int
    mean_probe_count: float
    total_ndis: int
    mean_ndis: float
    timing: SearchTimingTrace

    @property
    def qps(self) -> float:
        return self.query_count / self.timing.total_seconds if self.timing.total_seconds else 0.0


@dataclass(frozen=True)
class CalibrationBoundSelectionReport:
    target_recall: float
    recall_k: int
    calibration_bound: int
    points: tuple[CheckpointValidationReport, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "target_recall": self.target_recall,
            "recall_k": self.recall_k,
            "calibration_bound": self.calibration_bound,
            "points": [
                {
                    "fixed_bucket_count": point.fixed_bucket_count,
                    "recall": point.ann_recall_at_k[self.recall_k],
                    "mean_ndis": point.mean_ndis,
                    "search_seconds": point.timing.search_seconds,
                    "total_seconds": point.timing.total_seconds,
                    "qps": point.qps,
                }
                for point in self.points
            ],
        }


def select_calibration_bound(
    *,
    model: nn.Module,
    partition: FixedPartition,
    workload: CheckpointValidationWorkload,
    fixed_bucket_counts: tuple[int, ...],
    recall_k: int,
    target_recall: float = 0.995,
) -> CalibrationBoundSelectionReport:
    """Select the first validation Fixed Bucket Count reaching target Recall."""
    if not 0 < target_recall <= 1:
        raise ValueError("target Recall must be within (0, 1]")
    if (
        not fixed_bucket_counts
        or tuple(sorted(set(fixed_bucket_counts))) != fixed_bucket_counts
    ):
        raise ValueError("Fixed Bucket Count scan points must be unique and increasing")
    reports = tuple(
        evaluate_checkpoint(
            model=model,
            partition=partition,
            workload=workload,
            fixed_bucket_count=count,
            recall_ks=(recall_k,),
        )
        for count in fixed_bucket_counts
    )
    selected = next(
        (
            report.fixed_bucket_count
            for report in reports
            if report.ann_recall_at_k[recall_k] + 1e-12 >= target_recall
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "Fixed Bucket Count scan does not reach target Recall; extend the scan"
        )
    return CalibrationBoundSelectionReport(
        target_recall=target_recall,
        recall_k=recall_k,
        calibration_bound=selected,
        points=reports,
    )


def load_checkpoint_validation_workload(
    manifest: WorkloadManifest,
    *,
    partition: FixedPartition,
) -> CheckpointValidationWorkload:
    """Replay the manifest's checkpoint-validation role with raw neighbor GT."""
    role = "checkpoint_validation"
    if role not in manifest.metadata["roles"]:
        raise ValueError("Workload Manifest does not define checkpoint_validation")
    metadata = manifest.metadata
    resolved = metadata["resolved_config"]
    specification = dataset_spec(metadata["dataset_name"])
    base_source = FileVectorSource(
        Path(resolved["base_path"]),
        specification.format,
        specification.dimension,
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
        raise ValueError("checkpoint validation dimension does not match partition")
    if partition.dataset_name != metadata["dataset_name"]:
        raise ValueError("checkpoint validation dataset does not match partition")
    if partition.metric != metadata["ground_truth"]["metric"]:
        raise ValueError("checkpoint validation metric does not match partition")
    if len(partition.bucket_assignment) != base_source.count:
        raise ValueError("Fixed Partition base count does not match the manifest")

    query_ids = np.asarray(manifest.role_query_ids(role), dtype=np.int64)
    ground_truth = load_role_ground_truth(manifest, role)
    return CheckpointValidationWorkload(
        queries=source.read_rows(query_ids),
        ground_truth=ground_truth,
        query_ids=query_ids,
        manifest_filename=manifest.filename,
        ground_truth_artifact_filename=metadata["ground_truth"][
            "artifact_filename"
        ],
    )


def evaluate_checkpoint(
    *,
    model: nn.Module,
    partition: FixedPartition,
    workload: CheckpointValidationWorkload,
    fixed_bucket_count: int,
    recall_ks: tuple[int, ...],
    ranking: RoutingRanking = "raw_logits",
    ramps_alpha: float | None = None,
) -> CheckpointValidationReport:
    """Run fixed-width validation and report true ANN Recall@K."""
    if fixed_bucket_count <= 0 or fixed_bucket_count > partition.bucket_count:
        raise ValueError("validation fixed bucket count does not fit the partition")
    if not recall_ks or len(set(recall_ks)) != len(recall_ks) or any(
        k <= 0 for k in recall_ks
    ):
        raise ValueError("validation Recall Ks must be unique positive integers")
    maximum_k = max(recall_ks)
    if maximum_k > workload.ground_truth.shape[1]:
        raise ValueError("validation Recall K exceeds the ground-truth width")
    if workload.queries.shape[1] != partition.dimension:
        raise ValueError("checkpoint validation query dimension mismatch")
    if ranking not in ("raw_logits", "ramps_size_matched"):
        raise ValueError("unsupported checkpoint validation ranking")
    if (ranking == "ramps_size_matched") != (ramps_alpha is not None):
        raise ValueError(
            "RAMPS size-matched validation requires exactly one Power Alpha"
        )

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    started = time.perf_counter()
    model.eval()
    router_started = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(np.array(workload.queries, copy=True)).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    router_seconds = time.perf_counter() - router_started
    if logits.ndim != 2 or logits.shape != (
        len(workload.queries),
        partition.bucket_count,
    ):
        raise ValueError("checkpoint validation router output shape mismatch")
    probe_plan_started = time.perf_counter()
    if ranking == "ramps_size_matched":
        assert ramps_alpha is not None
        plan = build_ramps_decoder_fixed_width_plan(
            router_scores=logits.detach().cpu().numpy(),
            list_sizes=partition.list_sizes,
            alpha=ramps_alpha,
            fixed_bucket_count=fixed_bucket_count,
            candidate_bound=fixed_bucket_count,
        )
    else:
        routes = logits.topk(
            fixed_bucket_count, dim=1, largest=True, sorted=True
        ).indices
        plan = ProbePlan(
            bucket_ids=routes.detach().cpu().numpy().astype(np.int64, copy=False),
            probe_counts=np.full(
                len(workload.queries), fixed_bucket_count, dtype=np.int64
            ),
        )
    probe_plan_seconds = time.perf_counter() - probe_plan_started
    search = partition.search_preassigned(
        workload.queries,
        plan,
        topk=maximum_k,
    )
    metric_started = time.perf_counter()
    recall = {
        k: compute_recall_at_k(search.neighbor_ids, workload.ground_truth, k=k)
        for k in recall_ks
    }
    query_count = len(workload.queries)
    total_probe_count = int(search.probe_counts.sum(dtype=np.int64))
    total_ndis = int(search.ndis.sum(dtype=np.int64))
    metric_seconds = time.perf_counter() - metric_started
    return CheckpointValidationReport(
        query_count=query_count,
        fixed_bucket_count=fixed_bucket_count,
        ranking=ranking,
        ann_recall_at_k=recall,
        total_probe_count=total_probe_count,
        mean_probe_count=total_probe_count / query_count,
        total_ndis=total_ndis,
        mean_ndis=total_ndis / query_count,
        timing=SearchTimingTrace(
            router_seconds=router_seconds,
            probe_plan_seconds=probe_plan_seconds,
            search_seconds=search.timing.search_seconds,
            search_total_seconds=search.timing.total_seconds,
            metric_seconds=metric_seconds,
            total_seconds=time.perf_counter() - started,
            search_packing_seconds=search.timing.packing_seconds,
            faiss_call_count=search.timing.faiss_call_count,
            search_execution=search.timing.execution,
            faiss_parallel_mode=search.timing.parallel_mode,
        ),
    )
