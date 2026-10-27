"""Ground-truth-free retrieval using frozen router checkpoints."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

from ft_anns.adaptive_query.artifacts import CalibrationArtifact
from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner,
    merge_gain_cost_plan_batches,
)
from ft_anns.adaptive_query.policy import (
    build_probe_plan,
    build_ramps_decoder_fixed_width_plan,
)
from ft_anns.adaptive_query.policy_library import PolicyLibrary
from ft_anns.adaptive_query.prefix_query import (
    ProbabilityPrefixQuery,
    prefix_query_ndis,
    query_probability_prefix,
)
from ft_anns.adaptive_query.positive_query import (
    L2PRPlusQuery,
    l2pr_plus_ndis,
    query_l2pr_plus,
)
from ft_anns.artifact_io import (
    artifact_filename,
    is_artifact_filename,
    readonly_array,
)
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.domain import SearchExecution, require_dataset_name
from ft_anns.observability import SearchTimingTrace, utc_now_iso
from ft_anns.partition import (
    FaissSearchTimingTrace,
    FixedPartition,
    FixedSearchResult,
    ProbePlan,
)
from ft_anns.partition.artifacts import PreparedPaddedProbePlan

logger = logging.getLogger(__name__)


RetrievalTimingTrace = SearchTimingTrace


@dataclass(frozen=True)
class _RetrievalIdentity:
    calibration_filename: str | None
    policy_library_filename: str | None
    policy_id: str | None
    experiment_coordinates: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    neighbor_ids: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float32]
    probe_counts: npt.NDArray[np.int64]
    ndis: npt.NDArray[np.int64]
    timing: RetrievalTimingTrace
    checkpoint_filename: str
    partition_filename: str
    query_filename: str
    calibration_filename: str | None = None
    policy_library_filename: str | None = None
    policy_id: str | None = None
    experiment_coordinates: dict[str, Any] | None = None
    result_filename: str = "retrieval-results.npz"
    started_at: str | None = None
    finished_at: str | None = None

    def __post_init__(self) -> None:
        neighbor_ids = readonly_array(self.neighbor_ids, np.int64)
        distances = readonly_array(self.distances, np.float32)
        probe_counts = readonly_array(self.probe_counts, np.int64)
        ndis = readonly_array(self.ndis, np.int64)
        if neighbor_ids.ndim != 2 or distances.shape != neighbor_ids.shape:
            raise ValueError("retrieval IDs and distances must be aligned matrices")
        if probe_counts.shape != (len(neighbor_ids),) or ndis.shape != (
            len(neighbor_ids),
        ):
            raise ValueError("retrieval traces require one value per query")
        for name, filename in (
            ("checkpoint", self.checkpoint_filename),
            ("partition", self.partition_filename),
            ("query", self.query_filename),
        ):
            if not is_artifact_filename(filename):
                raise ValueError(f"retrieval {name} filename is invalid")
        if self.calibration_filename is not None and not is_artifact_filename(
            self.calibration_filename
        ):
            raise ValueError("retrieval calibration filename is invalid")
        if (self.policy_library_filename is None) != (self.policy_id is None):
            raise ValueError(
                "retrieval Policy Library filename and policy ID must be paired"
            )
        if self.policy_library_filename is not None:
            if not is_artifact_filename(self.policy_library_filename):
                raise ValueError("retrieval Policy Library filename is invalid")
            if not self.policy_id:
                raise ValueError("retrieval Policy Library policy ID is invalid")
            if self.calibration_filename is None:
                raise ValueError(
                    "Policy Library retrieval requires calibration identity"
                )
        if self.experiment_coordinates is not None and not isinstance(
            self.experiment_coordinates, dict
        ):
            raise ValueError("retrieval experiment coordinates are invalid")
        if not is_artifact_filename(self.result_filename):
            raise ValueError("retrieval result filename is invalid")
        object.__setattr__(self, "neighbor_ids", neighbor_ids)
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "probe_counts", probe_counts)
        object.__setattr__(self, "ndis", ndis)
        if self.experiment_coordinates is not None:
            object.__setattr__(
                self, "experiment_coordinates", dict(self.experiment_coordinates)
            )

    @property
    def filename(self) -> str:
        return self.result_filename

    @property
    def format(self) -> str:
        return (
            "fixed-retrieval-result-v2"
            if self.calibration_filename is None
            else "adaptive-retrieval-result-v2"
        )

    def validate_for_evaluation(
        self, *, dataset_name: str, query_filename: str, top_k: int
    ) -> None:
        """Reject a cached result that does not match its evaluation inputs."""
        expected_dataset = require_dataset_name(dataset_name)
        coordinates = self.experiment_coordinates or {}
        if coordinates.get("dataset") != expected_dataset:
            raise ValueError("retrieval result dataset does not match EvaluationConfig")
        if self.query_filename != query_filename:
            raise ValueError(
                "retrieval result query filename does not match EvaluationConfig"
            )
        if self.neighbor_ids.shape[1] != top_k:
            raise ValueError(
                "retrieval result width does not match EvaluationConfig top_k"
            )

    def save(self, path: str | Path) -> None:
        """Persist a runtime-only result that can be evaluated later."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        saved_filename = artifact_filename(output_path)
        object.__setattr__(self, "result_filename", saved_filename)
        metadata = {
            "format": self.format,
            "checkpoint_filename": self.checkpoint_filename,
            "partition_filename": self.partition_filename,
            "query_filename": self.query_filename,
            "result_filename": saved_filename,
            "calibration_filename": self.calibration_filename,
            "policy_library_filename": self.policy_library_filename,
            "policy_id": self.policy_id,
            "experiment_coordinates": self.experiment_coordinates,
            "timestamps": {
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            },
            "timing": self.timing.to_dict(),
        }
        temporary_path = output_path.with_name(f".{output_path.name}.tmp")
        with temporary_path.open("wb") as stream:
            np.savez(
                stream,
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
                neighbor_ids=self.neighbor_ids,
                distances=self.distances,
                probe_counts=self.probe_counts,
                ndis=self.ndis,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(output_path)

    @classmethod
    def load(cls, path: str | Path) -> RetrievalResult:
        result_path = Path(path)
        with np.load(result_path, allow_pickle=False) as payload:
            try:
                metadata = json.loads(str(payload["metadata"].item()))
                neighbor_ids = np.array(payload["neighbor_ids"], copy=True)
                distances = np.array(payload["distances"], copy=True)
                probe_counts = np.array(payload["probe_counts"], copy=True)
                ndis = np.array(payload["ndis"], copy=True)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("retrieval result artifact is invalid") from error
        if metadata.get("format") not in (
            "fixed-retrieval-result-v2",
            "adaptive-retrieval-result-v2",
        ):
            raise ValueError("unsupported retrieval result format")
        timing = metadata.get("timing", {})
        result = cls(
            neighbor_ids=neighbor_ids,
            distances=distances,
            probe_counts=probe_counts,
            ndis=ndis,
            timing=RetrievalTimingTrace.from_dict(timing),
            checkpoint_filename=str(metadata["checkpoint_filename"]),
            partition_filename=str(metadata["partition_filename"]),
            query_filename=str(metadata["query_filename"]),
            calibration_filename=(
                str(metadata["calibration_filename"])
                if metadata.get("calibration_filename") is not None
                else None
            ),
            policy_library_filename=(
                str(metadata["policy_library_filename"])
                if metadata.get("policy_library_filename") is not None
                else None
            ),
            policy_id=(
                str(metadata["policy_id"])
                if metadata.get("policy_id") is not None
                else None
            ),
            experiment_coordinates=(
                dict(metadata["experiment_coordinates"])
                if metadata.get("experiment_coordinates") is not None
                else None
            ),
            result_filename=artifact_filename(result_path),
            started_at=metadata.get("timestamps", {}).get("started_at"),
            finished_at=metadata.get("timestamps", {}).get("finished_at"),
        )
        if metadata.get("result_filename") != artifact_filename(result_path):
            raise ValueError("retrieval result filename mismatch")
        if metadata.get("format") != result.format:
            raise ValueError("retrieval result mode and calibration identity disagree")
        return result


def _validate_queries(
    queries: npt.ArrayLike, *, partition: FixedPartition
) -> npt.NDArray[np.float32]:
    matrix = np.array(queries, dtype=np.float32, copy=True, order="C")
    if matrix.ndim != 2 or not len(matrix) or matrix.shape[1] != partition.dimension:
        raise ValueError("query matrix does not match the Fixed Partition dimension")
    return matrix


def retrieve_probability_prefix(
    query: ProbabilityPrefixQuery, *, checkpoint_filename: str,
    partition_filename: str, query_filename: str, parameters_filename: str,
    experiment_coordinates: dict[str, Any],
) -> RetrievalResult:
    """Time the compact query call; build statistics and artifacts afterwards."""
    coordinates = {**experiment_coordinates, **query.coordinates}
    started_at = utc_now_iso()
    started = time.perf_counter()
    output = query_probability_prefix(query)
    total_seconds = time.perf_counter() - started
    finished_at = utc_now_iso()
    ndis = prefix_query_ndis(query, output)
    return RetrievalResult(
        neighbor_ids=output.neighbor_ids, distances=output.distances,
        probe_counts=output.probe_counts, ndis=ndis,
        timing=RetrievalTimingTrace(
            router_seconds=output.router_seconds,
            probe_plan_seconds=output.kernel_seconds + output.assembly_seconds,
            probe_plan_kernel_seconds=output.kernel_seconds,
            probe_plan_assembly_seconds=output.assembly_seconds,
            search_seconds=output.search_seconds,
            search_total_seconds=output.packing_seconds + output.search_seconds,
            search_packing_seconds=output.packing_seconds,
            total_seconds=total_seconds, faiss_call_count=1, search_execution="padded",
            faiss_parallel_mode=int(query.index.parallel_mode),
            router_batch_size=query.router_batch_size, faiss_batch_size="full",
        ),
        checkpoint_filename=checkpoint_filename, partition_filename=partition_filename,
        query_filename=query_filename, calibration_filename=parameters_filename,
        experiment_coordinates=coordinates, started_at=started_at, finished_at=finished_at,
    )


def retrieve_l2pr_plus(
    query: L2PRPlusQuery, *, checkpoint_filename: str,
    partition_filename: str, query_filename: str, parameters_filename: str,
    experiment_coordinates: dict[str, Any],
) -> RetrievalResult:
    """Time the complete L2PR+ call; retain legacy per-query reports afterwards."""
    coordinates = {**experiment_coordinates, **query.coordinates}
    started_at = utc_now_iso()
    started = time.perf_counter()
    output = query_l2pr_plus(query)
    total_seconds = time.perf_counter() - started
    finished_at = utc_now_iso()
    if output.faiss_ndis is not None:
        coordinates["faiss_ndis_total"] = output.faiss_ndis
        coordinates["faiss_mean_ndis"] = output.faiss_ndis / len(query.queries)
    ndis = l2pr_plus_ndis(query, output)
    return RetrievalResult(
        neighbor_ids=output.neighbor_ids, distances=output.distances,
        probe_counts=output.probe_counts, ndis=ndis,
        timing=RetrievalTimingTrace(
            router_seconds=output.router_seconds,
            probe_plan_seconds=output.kernel_seconds + output.assembly_seconds,
            probe_plan_kernel_seconds=output.kernel_seconds,
            probe_plan_assembly_seconds=output.assembly_seconds,
            search_seconds=output.search_seconds,
            search_total_seconds=output.packing_seconds + output.search_seconds,
            search_packing_seconds=output.packing_seconds,
            total_seconds=total_seconds, faiss_call_count=output.faiss_call_count, search_execution="padded",
            faiss_parallel_mode=int(query.index.parallel_mode),
            router_batch_size=query.router_batch_size, faiss_batch_size=query.faiss_batch_size or "full",
        ),
        checkpoint_filename=checkpoint_filename, partition_filename=partition_filename,
        query_filename=query_filename, calibration_filename=parameters_filename,
        experiment_coordinates=coordinates, started_at=started_at, finished_at=finished_at,
    )


def _run_router(
    queries: npt.NDArray[np.float32], checkpoint: RouterCheckpoint
) -> tuple[torch.Tensor, float]:
    model = checkpoint.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        logits = model(torch.from_numpy(queries).to(device))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return logits, time.perf_counter() - started


def _retrieve_probe_plan(
    *,
    queries: npt.NDArray[np.float32],
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    probe_plan: ProbePlan | PreparedPaddedProbePlan,
    topk: int,
    router_seconds: float,
    probe_plan_seconds: float,
    started: float,
    started_at: str,
    identity: _RetrievalIdentity,
    query_filename: str,
    search_execution: SearchExecution,
    router_batch_size: int | None = None,
    faiss_batch_size: int | None = None,
    probe_plan_kernel_seconds: float | None = None,
    probe_plan_assembly_seconds: float = 0.0,
) -> RetrievalResult:
    """Execute the one low-level preassigned-search path used by both modes."""
    if faiss_batch_size is None or faiss_batch_size >= len(queries):
        search = retrieve_probe_plan(
            queries=queries,
            partition=partition,
            probe_plan=probe_plan,
            topk=topk,
            search_execution=search_execution,
        )
    else:
        search_started = time.perf_counter()
        search_batches = [
            retrieve_probe_plan(
                queries=queries[offset : offset + faiss_batch_size],
                partition=partition,
                probe_plan=(
                    PreparedPaddedProbePlan(
                        bucket_ids=probe_plan.bucket_ids[
                            offset : offset + faiss_batch_size
                        ],
                        probe_counts=probe_plan.probe_counts[
                            offset : offset + faiss_batch_size
                        ],
                        ndis=probe_plan.ndis[offset : offset + faiss_batch_size],
                    )
                    if isinstance(probe_plan, PreparedPaddedProbePlan)
                    else ProbePlan(
                        bucket_ids=probe_plan.bucket_ids[
                            offset : offset + faiss_batch_size
                        ],
                        probe_counts=probe_plan.probe_counts[
                            offset : offset + faiss_batch_size
                        ],
                    )
                ),
                topk=topk,
                search_execution=search_execution,
            )
            for offset in range(0, len(queries), faiss_batch_size)
        ]
        search = FixedSearchResult(
            neighbor_ids=np.concatenate(
                [batch.neighbor_ids for batch in search_batches]
            ),
            distances=np.concatenate([batch.distances for batch in search_batches]),
            probe_counts=np.concatenate(
                [batch.probe_counts for batch in search_batches]
            ),
            ndis=np.concatenate([batch.ndis for batch in search_batches]),
            timing=FaissSearchTimingTrace(
                search_seconds=sum(
                    batch.timing.search_seconds for batch in search_batches
                ),
                total_seconds=time.perf_counter() - search_started,
                packing_seconds=sum(
                    batch.timing.packing_seconds for batch in search_batches
                ),
                ndis_seconds=sum(batch.timing.ndis_seconds for batch in search_batches),
                faiss_call_count=sum(
                    batch.timing.faiss_call_count for batch in search_batches
                ),
                execution=search_execution,
                parallel_mode=search_batches[0].timing.parallel_mode,
            ),
        )
    result = RetrievalResult(
        neighbor_ids=search.neighbor_ids,
        distances=search.distances,
        probe_counts=search.probe_counts,
        ndis=search.ndis,
        timing=RetrievalTimingTrace(
            router_seconds=router_seconds,
            probe_plan_seconds=probe_plan_seconds,
            probe_plan_kernel_seconds=probe_plan_kernel_seconds,
            probe_plan_assembly_seconds=probe_plan_assembly_seconds,
            search_seconds=search.timing.search_seconds,
            search_total_seconds=search.timing.total_seconds,
            total_seconds=time.perf_counter() - started,
            search_packing_seconds=search.timing.packing_seconds,
            search_ndis_seconds=search.timing.ndis_seconds,
            faiss_call_count=search.timing.faiss_call_count,
            search_execution=search.timing.execution,
            faiss_parallel_mode=search.timing.parallel_mode,
            router_batch_size=router_batch_size or "full",
            faiss_batch_size=faiss_batch_size or "full",
        ),
        checkpoint_filename=checkpoint.filename,
        partition_filename=str(partition.filename),
        query_filename=query_filename,
        calibration_filename=identity.calibration_filename,
        policy_library_filename=identity.policy_library_filename,
        policy_id=identity.policy_id,
        experiment_coordinates=identity.experiment_coordinates,
        started_at=started_at,
        finished_at=utc_now_iso(),
    )
    logger.debug(
        "retrieval batch completed queries=%d router_seconds=%.6f "
        "probe_plan_seconds=%.6f search_seconds=%.6f total_seconds=%.6f",
        len(queries),
        result.timing.router_seconds,
        result.timing.probe_plan_seconds,
        result.timing.search_seconds,
        result.timing.total_seconds,
    )
    return result


def _validate_batch_size(batch_size: int | None) -> None:
    if batch_size is not None and (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("retrieval batch_size must be a positive integer")


def _validate_retrieval_batch_sizes(
    *,
    batch_size: int | None,
    router_batch_size: int | None,
    faiss_batch_size: int | None,
) -> None:
    for value in (batch_size, router_batch_size, faiss_batch_size):
        _validate_batch_size(value)
    if batch_size is not None and (
        router_batch_size is not None or faiss_batch_size is not None
    ):
        raise ValueError(
            "legacy batch_size cannot be combined with router/faiss batch sizes"
        )


def _build_plan_in_router_batches(
    *,
    queries: npt.NDArray[np.float32],
    checkpoint: RouterCheckpoint,
    router_batch_size: int | None,
    build_batch: Callable[[torch.Tensor], ProbePlan],
) -> tuple[ProbePlan, float, float]:
    router_seconds = 0.0
    probe_plan_seconds = 0.0
    plan_batches: list[ProbePlan] = []
    resolved_batch_size = router_batch_size or len(queries)
    for offset in range(0, len(queries), resolved_batch_size):
        logits, batch_router_seconds = _run_router(
            queries[offset : offset + resolved_batch_size], checkpoint
        )
        router_seconds += batch_router_seconds
        probe_plan_started = time.perf_counter()
        plan_batches.append(build_batch(logits))
        probe_plan_seconds += time.perf_counter() - probe_plan_started
    route_width = max(batch.bucket_ids.shape[1] for batch in plan_batches)
    bucket_ids = np.full((len(queries), route_width), -1, dtype=np.int64)
    probe_counts = np.empty(len(queries), dtype=np.int64)
    offset = 0
    for batch in plan_batches:
        stop = offset + len(batch.probe_counts)
        bucket_ids[offset:stop, : batch.bucket_ids.shape[1]] = batch.bucket_ids
        probe_counts[offset:stop] = batch.probe_counts
        offset = stop
    return (
        ProbePlan(
            bucket_ids=bucket_ids,
            probe_counts=probe_counts,
        ),
        router_seconds,
        probe_plan_seconds,
    )


def _build_gain_cost_plan_in_router_batches(
    *,
    queries: npt.NDArray[np.float32],
    checkpoint: RouterCheckpoint,
    router_batch_size: int | None,
    planner: GainCostProbePlanner,
    list_sizes: torch.Tensor,
) -> tuple[PreparedPaddedProbePlan, float, float, float, float]:
    """Build one partition-bound padded plan while retaining native tensors."""
    router_seconds = 0.0
    kernel_seconds = 0.0
    tensor_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    resolved_batch_size = router_batch_size or len(queries)
    for offset in range(0, len(queries), resolved_batch_size):
        logits, batch_router_seconds = _run_router(
            queries[offset : offset + resolved_batch_size], checkpoint
        )
        router_seconds += batch_router_seconds
        kernel_started = time.perf_counter()
        tensor_batches.append(planner.build_tensors(logits))
        kernel_seconds += time.perf_counter() - kernel_started

    assembly_started = time.perf_counter()
    prepared = merge_gain_cost_plan_batches(
        tensor_batches,
        list_sizes=list_sizes,
    )
    assembly_seconds = time.perf_counter() - assembly_started
    return (
        prepared,
        router_seconds,
        kernel_seconds + assembly_seconds,
        kernel_seconds,
        assembly_seconds,
    )


def _combine_batched_results(
    *,
    batches: list[RetrievalResult],
    batch_size: int,
    queries: npt.NDArray[np.float32],
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    started: float,
    started_at: str,
    identity: _RetrievalIdentity,
    query_filename: str,
) -> RetrievalResult:
    result = RetrievalResult(
        neighbor_ids=np.concatenate([batch.neighbor_ids for batch in batches]),
        distances=np.concatenate([batch.distances for batch in batches]),
        probe_counts=np.concatenate([batch.probe_counts for batch in batches]),
        ndis=np.concatenate([batch.ndis for batch in batches]),
        timing=RetrievalTimingTrace(
            router_seconds=sum(batch.timing.router_seconds for batch in batches),
            probe_plan_seconds=sum(
                batch.timing.probe_plan_seconds for batch in batches
            ),
            search_seconds=sum(batch.timing.search_seconds for batch in batches),
            search_total_seconds=sum(
                batch.timing.search_total_seconds or 0.0 for batch in batches
            ),
            total_seconds=time.perf_counter() - started,
            search_packing_seconds=sum(
                batch.timing.search_packing_seconds for batch in batches
            ),
            faiss_call_count=sum(batch.timing.faiss_call_count for batch in batches),
            search_execution=(
                batches[0].timing.search_execution
                if all(
                    batch.timing.search_execution == batches[0].timing.search_execution
                    for batch in batches
                )
                else "mixed"
            ),
            faiss_parallel_mode=(
                batches[0].timing.faiss_parallel_mode
                if all(
                    batch.timing.faiss_parallel_mode
                    == batches[0].timing.faiss_parallel_mode
                    for batch in batches
                )
                else None
            ),
            router_batch_size=batch_size,
            faiss_batch_size=batch_size,
        ),
        checkpoint_filename=checkpoint.filename,
        partition_filename=str(partition.filename),
        query_filename=query_filename,
        calibration_filename=identity.calibration_filename,
        policy_library_filename=identity.policy_library_filename,
        policy_id=identity.policy_id,
        experiment_coordinates=identity.experiment_coordinates,
        started_at=started_at,
        finished_at=utc_now_iso(),
    )
    logger.debug(
        "batched retrieval combined queries=%d batches=%d total_seconds=%.6f",
        len(queries),
        len(batches),
        result.timing.total_seconds,
    )
    return result


def _retrieve_in_batches(
    *,
    queries: npt.NDArray[np.float32],
    batch_size: int,
    retrieve_batch: Callable[[npt.NDArray[np.float32]], RetrievalResult],
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    started: float,
    started_at: str,
    identity: _RetrievalIdentity,
    query_filename: str,
) -> RetrievalResult:
    batches = [
        retrieve_batch(queries[offset : offset + batch_size])
        for offset in range(0, len(queries), batch_size)
    ]
    return _combine_batched_results(
        batches=batches,
        batch_size=batch_size,
        queries=queries,
        checkpoint=checkpoint,
        partition=partition,
        started=started,
        started_at=started_at,
        identity=identity,
        query_filename=query_filename,
    )


def _experiment_coordinates(
    checkpoint: RouterCheckpoint,
    *,
    query_mode: str,
    fixed_bucket_count: int | None,
    policy_id: str | None,
    ranking: str,
) -> dict[str, Any]:
    resolved = checkpoint.metadata.get("resolved_train_config", {})
    transform = checkpoint.metadata.get("loss", {}).get("target_size_transform", {})
    return {
        "dataset": checkpoint.metadata.get("dataset_name"),
        "loss": checkpoint.metadata.get("loss", {}).get("name"),
        "power_alpha": transform.get("alpha"),
        "seed": resolved.get("seed"),
        "query_mode": query_mode,
        "ranking": ranking,
        "fixed_bucket_count": fixed_bucket_count,
        "policy_id": policy_id,
    }


def retrieve_probe_plan(
    *,
    queries: npt.ArrayLike,
    partition: FixedPartition,
    probe_plan: ProbePlan | PreparedPaddedProbePlan,
    topk: int,
    search_execution: SearchExecution = "grouped",
) -> FixedSearchResult:
    """Execute the shared low-level preassigned search for a frozen plan."""
    return partition.search_preassigned(
        queries, probe_plan, topk=topk, execution=search_execution
    )


def retrieve_fixed(
    *,
    queries: npt.ArrayLike,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    topm: int,
    topk: int,
    batch_size: int | None = None,
    router_batch_size: int | None = None,
    faiss_batch_size: int | None = None,
    query_filename: str = "queries",
    search_execution: SearchExecution = "grouped",
) -> RetrievalResult:
    """Route raw queries to exactly topm buckets and perform IVF search."""
    started_at = utc_now_iso()
    started = time.perf_counter()
    logger.debug(
        "fixed retrieval started topm=%d topk=%d batch_size=%s",
        topm,
        topk,
        batch_size,
    )
    checkpoint.validate_for(partition)
    if topm <= 0 or topm > partition.bucket_count:
        raise ValueError("topm must be within the Fixed Partition bucket count")
    if topk <= 0:
        raise ValueError("topk must be positive")
    _validate_retrieval_batch_sizes(
        batch_size=batch_size,
        router_batch_size=router_batch_size,
        faiss_batch_size=faiss_batch_size,
    )
    query_matrix = _validate_queries(queries, partition=partition)
    coordinates = _experiment_coordinates(
        checkpoint,
        query_mode="fixed",
        fixed_bucket_count=topm,
        policy_id=None,
        ranking="raw_logits",
    )
    identity = _RetrievalIdentity(
        calibration_filename=None,
        policy_library_filename=None,
        policy_id=None,
        experiment_coordinates=coordinates,
    )
    if batch_size is not None and batch_size < len(query_matrix):
        return _retrieve_in_batches(
            queries=query_matrix,
            batch_size=batch_size,
            retrieve_batch=lambda batch: retrieve_fixed(
                queries=batch,
                checkpoint=checkpoint,
                partition=partition,
                topm=topm,
                topk=topk,
                query_filename=query_filename,
                search_execution=search_execution,
            ),
            checkpoint=checkpoint,
            partition=partition,
            started=started,
            started_at=started_at,
            identity=identity,
            query_filename=query_filename,
        )
    probe_plan, router_seconds, probe_plan_seconds = _build_plan_in_router_batches(
        queries=query_matrix,
        checkpoint=checkpoint,
        router_batch_size=router_batch_size,
        build_batch=lambda logits: ProbePlan(
            bucket_ids=logits.topk(topm, dim=1, largest=True, sorted=True)
            .indices.cpu()
            .numpy()
            .astype(np.int64, copy=False),
            probe_counts=np.full(len(logits), topm, dtype=np.int64),
        ),
    )
    return _retrieve_probe_plan(
        queries=query_matrix,
        checkpoint=checkpoint,
        partition=partition,
        probe_plan=probe_plan,
        topk=topk,
        router_seconds=router_seconds,
        probe_plan_seconds=probe_plan_seconds,
        started=started,
        started_at=started_at,
        identity=identity,
        query_filename=query_filename,
        search_execution=search_execution,
        router_batch_size=router_batch_size,
        faiss_batch_size=faiss_batch_size,
    )


def retrieve_decoder_fixed(
    *,
    queries: npt.ArrayLike,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    topm: int,
    topk: int,
    batch_size: int | None = None,
    router_batch_size: int | None = None,
    faiss_batch_size: int | None = None,
    query_filename: str = "queries",
    search_execution: SearchExecution = "grouped",
) -> RetrievalResult:
    """Route with the checkpoint's RAMPS Decoder ranking at a fixed width."""
    started_at = utc_now_iso()
    started = time.perf_counter()
    alpha = checkpoint.require_adaptive_query_support(partition)
    if topm <= 0 or topm > partition.bucket_count:
        raise ValueError("topm must be within the Fixed Partition bucket count")
    if topk <= 0:
        raise ValueError("topk must be positive")
    _validate_retrieval_batch_sizes(
        batch_size=batch_size,
        router_batch_size=router_batch_size,
        faiss_batch_size=faiss_batch_size,
    )
    query_matrix = _validate_queries(queries, partition=partition)
    identity = _RetrievalIdentity(
        calibration_filename=None,
        policy_library_filename=None,
        policy_id=None,
        experiment_coordinates=_experiment_coordinates(
            checkpoint,
            query_mode="fixed",
            fixed_bucket_count=topm,
            policy_id=None,
            ranking="ramps_size_matched",
        ),
    )
    if batch_size is not None and batch_size < len(query_matrix):
        return _retrieve_in_batches(
            queries=query_matrix,
            batch_size=batch_size,
            retrieve_batch=lambda batch: retrieve_decoder_fixed(
                queries=batch,
                checkpoint=checkpoint,
                partition=partition,
                topm=topm,
                topk=topk,
                query_filename=query_filename,
                search_execution=search_execution,
            ),
            checkpoint=checkpoint,
            partition=partition,
            started=started,
            started_at=started_at,
            identity=identity,
            query_filename=query_filename,
        )
    plan, router_seconds, probe_plan_seconds = _build_plan_in_router_batches(
        queries=query_matrix,
        checkpoint=checkpoint,
        router_batch_size=router_batch_size,
        build_batch=lambda logits: build_ramps_decoder_fixed_width_plan(
            router_scores=logits.detach().cpu().numpy(),
            list_sizes=partition.list_sizes,
            alpha=alpha,
            fixed_bucket_count=topm,
            candidate_bound=topm,
        ),
    )
    return _retrieve_probe_plan(
        queries=query_matrix,
        checkpoint=checkpoint,
        partition=partition,
        probe_plan=plan,
        topk=topk,
        router_seconds=router_seconds,
        probe_plan_seconds=probe_plan_seconds,
        started=started,
        started_at=started_at,
        identity=identity,
        query_filename=query_filename,
        search_execution=search_execution,
        router_batch_size=router_batch_size,
        faiss_batch_size=faiss_batch_size,
    )


def retrieve_adaptive(
    *,
    queries: npt.ArrayLike,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    artifact: CalibrationArtifact | None = None,
    policy_library: PolicyLibrary | None = None,
    policy_id: str | None = None,
    topk: int,
    batch_size: int | None = None,
    router_batch_size: int | None = None,
    faiss_batch_size: int | None = None,
    query_filename: str = "queries",
    search_execution: SearchExecution = "padded",
) -> RetrievalResult:
    """Execute a frozen adaptive policy without accepting ground truth."""
    started_at = utc_now_iso()
    started = time.perf_counter()
    logger.debug(
        "adaptive retrieval started topk=%d batch_size=%s policy_id=%s",
        topk,
        batch_size,
        policy_id,
    )
    if topk <= 0:
        raise ValueError("topk must be positive")
    checkpoint.validate_for(partition)
    if policy_library is not None:
        if artifact is not None:
            raise ValueError(
                "adaptive retrieval accepts either calibration or Policy Library"
            )
        if policy_id is None:
            raise ValueError("Policy Library adaptive retrieval requires a policy ID")
        selected_artifact = policy_library.resolve(
            policy_id,
            checkpoint=checkpoint,
            partition=partition,
        )
        library_filename = policy_library.filename
    else:
        if artifact is None:
            raise ValueError("adaptive retrieval requires a Calibration Artifact")
        if policy_id is not None:
            raise ValueError("policy ID requires a Policy Library")
        artifact.validate_for(checkpoint=checkpoint, partition=partition)
        selected_artifact = artifact
        library_filename = None
    _validate_retrieval_batch_sizes(
        batch_size=batch_size,
        router_batch_size=router_batch_size,
        faiss_batch_size=faiss_batch_size,
    )
    query_matrix = _validate_queries(queries, partition=partition)
    coordinates = _experiment_coordinates(
        checkpoint,
        query_mode="adaptive",
        fixed_bucket_count=None,
        policy_id=policy_id,
        ranking="ramps_size_matched",
    )
    identity = _RetrievalIdentity(
        calibration_filename=selected_artifact.filename,
        policy_library_filename=library_filename,
        policy_id=policy_id,
        experiment_coordinates=coordinates,
    )
    if batch_size is not None and batch_size < len(query_matrix):
        return _retrieve_in_batches(
            queries=query_matrix,
            batch_size=batch_size,
            retrieve_batch=lambda batch: retrieve_adaptive(
                queries=batch,
                checkpoint=checkpoint,
                partition=partition,
                artifact=artifact,
                policy_library=policy_library,
                policy_id=policy_id,
                topk=topk,
                query_filename=query_filename,
                search_execution=search_execution,
            ),
            checkpoint=checkpoint,
            partition=partition,
            started=started,
            started_at=started_at,
            identity=identity,
            query_filename=query_filename,
        )
    plan, router_seconds, probe_plan_seconds = _build_plan_in_router_batches(
        queries=query_matrix,
        checkpoint=checkpoint,
        router_batch_size=router_batch_size,
        build_batch=lambda logits: build_probe_plan(
            router_scores=logits.detach().cpu().numpy(),
            list_sizes=partition.list_sizes,
            artifact=selected_artifact,
        ),
    )
    return _retrieve_probe_plan(
        queries=query_matrix,
        checkpoint=checkpoint,
        partition=partition,
        probe_plan=plan,
        topk=topk,
        router_seconds=router_seconds,
        probe_plan_seconds=probe_plan_seconds,
        started=started,
        started_at=started_at,
        identity=identity,
        query_filename=query_filename,
        search_execution=search_execution,
        router_batch_size=router_batch_size,
        faiss_batch_size=faiss_batch_size,
    )
