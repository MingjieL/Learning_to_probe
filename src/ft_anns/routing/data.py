"""Unified query-level training samples and deterministic batching."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ft_anns.datasets.base_gt import FileVectorSource
from ft_anns.datasets.registry import dataset_spec
from ft_anns.datasets.workloads import WorkloadManifest, load_role_ground_truth
from ft_anns.partition.fixed_ivf import FixedPartition

TrainingRole = Literal["router_train", "checkpoint_validation"]


@dataclass(frozen=True)
class WorkloadProvenance:
    manifest_filename: str
    role: TrainingRole
    query_source_filename: str
    ground_truth_artifact_filename: str
    partition_filename: str
    content_kind: str


@dataclass(frozen=True)
class QueryLevelWorkload:
    queries: Tensor
    bucket_ids: Tensor
    query_ids: Tensor
    provenance: WorkloadProvenance | None = None

    def __post_init__(self) -> None:
        queries = self.queries.detach().to(dtype=torch.float32, device="cpu").clone()
        bucket_ids = self.bucket_ids.detach().to(dtype=torch.int64, device="cpu").clone()
        query_ids = self.query_ids.detach().to(dtype=torch.int64, device="cpu").clone()
        if queries.ndim != 2 or bucket_ids.ndim != 2:
            raise ValueError("queries and bucket_ids must both be matrices")
        if bucket_ids.shape[1] == 0:
            raise ValueError("each query requires one or more bucket labels")
        if query_ids.ndim != 1 or not (
            len(queries) == len(bucket_ids) == len(query_ids)
        ):
            raise ValueError("queries, bucket labels, and query IDs must align")
        if not torch.isfinite(queries).all():
            raise ValueError("queries must contain only finite values")
        object.__setattr__(self, "queries", queries)
        object.__setattr__(self, "bucket_ids", bucket_ids)
        object.__setattr__(self, "query_ids", query_ids)

    @property
    def content_kind(self) -> str:
        return "in-memory-workload"


class _QueryLevelDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, workload: QueryLevelWorkload) -> None:
        self._workload = workload

    def __len__(self) -> int:
        return len(self._workload.queries)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self._workload.queries[index], self._workload.bucket_ids[index]


@dataclass(frozen=True)
class FileBackedQueryLevelWorkload:
    """Large router workload whose vectors and GT rows stay memory-mapped."""

    query_ids: np.ndarray
    query_source: FileVectorSource
    neighbor_ids: np.ndarray
    ground_truth_k: int
    bucket_assignment: np.ndarray
    provenance: WorkloadProvenance

    def __post_init__(self) -> None:
        query_ids = np.asarray(self.query_ids, dtype=np.int64)
        if query_ids.ndim != 1 or not len(query_ids):
            raise ValueError("file-backed workload query IDs must be non-empty")
        if query_ids.size and (
            int(query_ids.min()) < 0 or int(query_ids.max()) >= self.query_source.count
        ):
            raise ValueError("file-backed workload query ID is outside the source")
        if self.neighbor_ids.ndim != 2 or self.neighbor_ids.shape[1] < self.ground_truth_k:
            raise ValueError("file-backed workload ground truth is too shallow")
        if len(self.neighbor_ids) != self.query_source.count:
            raise ValueError(
                "file-backed workload requires identity-aligned query ground truth"
            )
        object.__setattr__(self, "query_ids", query_ids)

    @property
    def content_kind(self) -> str:
        return "file-backed-workload"

    @property
    def query_count(self) -> int:
        return len(self.query_ids)

    @property
    def dimension(self) -> int:
        return self.query_source.dimension


TrainingWorkload: TypeAlias = QueryLevelWorkload | FileBackedQueryLevelWorkload


class _FileBackedBatchLoader:
    def __init__(
        self,
        workload: FileBackedQueryLevelWorkload,
        *,
        batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self._workload = workload
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._seed = seed

    def __iter__(self) -> Iterator[tuple[Tensor, Tensor]]:
        workload = self._workload
        positions = np.arange(workload.query_count, dtype=np.int64)
        if self._shuffle:
            np.random.default_rng(self._seed).shuffle(positions)
        for start in range(0, workload.query_count, self._batch_size):
            batch_ids = workload.query_ids[positions[start : start + self._batch_size]]
            queries = np.ascontiguousarray(
                workload.query_source.read_rows(batch_ids), dtype=np.float32
            )
            neighbors = np.asarray(
                workload.neighbor_ids[batch_ids, : workload.ground_truth_k]
            )
            bucket_ids = np.ascontiguousarray(
                workload.bucket_assignment[neighbors], dtype=np.int64
            )
            yield torch.from_numpy(queries), torch.from_numpy(bucket_ids)


def create_query_dataloader(
    workload: TrainingWorkload,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor]] | _FileBackedBatchLoader:
    """Batch unique queries without expanding their K bucket labels."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if isinstance(workload, FileBackedQueryLevelWorkload):
        return _FileBackedBatchLoader(
            workload,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
        )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return DataLoader(
        _QueryLevelDataset(workload),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def load_query_level_workload(
    manifest: WorkloadManifest,
    *,
    role: TrainingRole,
    partition: FixedPartition,
    file_backed: bool = False,
) -> TrainingWorkload:
    """Replay one persisted training role against a Fixed Partition."""
    if role not in ("router_train", "checkpoint_validation"):
        raise ValueError("training may only consume router_train or checkpoint_validation")
    if role not in manifest.metadata["roles"]:
        raise ValueError(f"Workload Manifest does not define role {role!r}")

    resolved = manifest.metadata["resolved_config"]
    specification = dataset_spec(manifest.metadata["dataset_name"])
    base_source = FileVectorSource(
        Path(resolved["base_path"]),
        specification.format,
        specification.dimension,
    )
    if base_source.filename != manifest.metadata["base_filename"]:
        raise ValueError("base source filename no longer matches the manifest")
    source = (
        base_source
        if resolved["source"] == "base_set"
        else FileVectorSource(
            Path(resolved["query_path"]),
            resolved.get("query_format") or specification.format,
            specification.dimension,
        )
    )
    if source.dimension != partition.dimension:
        raise ValueError("query source and Fixed Partition dimensions do not match")
    if partition.dataset_name != manifest.metadata["dataset_name"]:
        raise ValueError("query source and Fixed Partition datasets do not match")
    if len(partition.bucket_assignment) != base_source.count:
        raise ValueError("Fixed Partition base count does not match the manifest")
    if source.filename != manifest.metadata["query_source_filename"]:
        raise ValueError("query source filename no longer matches the manifest")

    query_ids = np.asarray(manifest.role_query_ids(role), dtype=np.int64)
    provenance = WorkloadProvenance(
        manifest_filename=manifest.filename,
        role=role,
        query_source_filename=manifest.metadata["query_source_filename"],
        ground_truth_artifact_filename=manifest.metadata["ground_truth"][
            "artifact_filename"
        ],
        partition_filename=str(partition.filename),
        content_kind=("file-backed-workload" if file_backed else "in-memory-workload"),
    )
    if file_backed:
        if role != "router_train":
            raise ValueError("file-backed mode is only supported for router_train")
        artifact_path = Path(manifest.metadata["ground_truth"]["artifact_path"])
        artifact_query_ids = np.load(
            artifact_path / "query_ids.npy", mmap_mode="r", allow_pickle=False
        )
        for start in range(0, len(artifact_query_ids), 1_000_000):
            stop = min(start + 1_000_000, len(artifact_query_ids))
            if not np.array_equal(
                artifact_query_ids[start:stop], np.arange(start, stop)
            ):
                raise ValueError(
                    "file-backed training requires identity-aligned ground truth"
                )
        neighbor_ids = np.load(
            artifact_path / "neighbor_ids.npy", mmap_mode="r", allow_pickle=False
        )
        return FileBackedQueryLevelWorkload(
            query_ids=query_ids,
            query_source=source,
            neighbor_ids=neighbor_ids,
            ground_truth_k=int(manifest.metadata["ground_truth"]["k"]),
            bucket_assignment=partition.bucket_assignment,
            provenance=provenance,
        )
    ground_truth = load_role_ground_truth(manifest, role)
    bucket_ids = partition.bucket_ids_for(ground_truth)
    queries = np.asarray(source.read_rows(query_ids), dtype=np.float32)
    return QueryLevelWorkload(
        queries=torch.from_numpy(np.array(queries, copy=True)),
        bucket_ids=torch.from_numpy(np.array(bucket_ids, copy=True)),
        query_ids=torch.from_numpy(np.array(query_ids, copy=True)),
        provenance=provenance,
    )
