"""Build and query immutable IVF partitions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NewType

import faiss  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from ft_anns.artifact_io import (
    artifact_filename,
    is_artifact_filename,
    write_json_atomic,
)
from ft_anns.compatibility import (
    compatibility_filename,
    compatibility_slug,
    require_compatibility_metadata,
)
from ft_anns.config import PartitionConfig
from ft_anns.domain import Metric, SearchExecution, require_dataset_name
from ft_anns.partition.artifacts import (
    FaissSearchTimingTrace,
    FixedSearchResult,
    PreparedPaddedProbePlan,
    ProbePlan,
)

_MANIFEST_FILE = "artifact.json"
_INDEX_FILE = "index.faiss"
_ARTIFACT_FORMAT = "fixed-partition-v4"
# Mode 3 keeps deterministic query-level result ordering while using finer
# query parallelism. On 100M preassigned search it matches mode 2 throughput;
# modes 1 and 2 can reorder equal-distance neighbors at the Top-K boundary.
CPU_PREASSIGNED_PARALLEL_MODE = 3
_ARRAY_FILES = {
    "bucket_assignment": "bucket_assignment.npy",
    "centroids": "centroids.npy",
    "list_sizes": "list_sizes.npy",
    "posting_offsets": "posting_offsets.npy",
    "posting_ids": "posting_ids.npy",
}

PartitionFilename = NewType("PartitionFilename", str)


def _readonly(array: npt.NDArray[Any]) -> npt.NDArray[Any]:
    if array.flags.writeable:
        readonly_buffer = memoryview(np.ascontiguousarray(array)).toreadonly()
        array = np.frombuffer(readonly_buffer, dtype=array.dtype).reshape(array.shape)
    array.flags.writeable = False
    return array


def _partition_filename(config: PartitionConfig) -> PartitionFilename:
    return PartitionFilename(
        artifact_filename(config.artifact_path)
        if config.artifact_path is not None
        else "fixed-partition"
    )


def _partition_profile(
    *, dataset_name: str, bucket_count: int, kmeans_niter: int, build_seed: int
) -> dict[str, object]:
    return {
        "dataset_name": dataset_name,
        "bucket_count": bucket_count,
        "kmeans_niter": kmeans_niter,
        "partition_build_seed": build_seed,
    }


def _posting_metadata(
    assignment: npt.NDArray[np.int64], bucket_count: int
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    list_sizes = np.bincount(assignment, minlength=bucket_count).astype(
        np.int64, copy=False
    )
    posting_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(list_sizes, dtype=np.int64))
    )
    posting_ids = np.argsort(assignment, kind="stable").astype(np.int64, copy=False)
    return list_sizes, posting_offsets, posting_ids


def _posting_sizes_and_offsets(
    assignment: npt.NDArray[np.int64], bucket_count: int
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    list_sizes = np.bincount(assignment, minlength=bucket_count).astype(
        np.int64, copy=False
    )
    posting_offsets = np.concatenate(
        (np.asarray([0], dtype=np.int64), np.cumsum(list_sizes, dtype=np.int64))
    )
    return list_sizes, posting_offsets


def _validate_posting_arrays(
    *,
    assignment: npt.NDArray[np.int64],
    list_sizes: npt.NDArray[np.int64],
    posting_offsets: npt.NDArray[np.int64],
    posting_ids: npt.NDArray[np.int64],
) -> None:
    """Validate postings in linear time without sorting the full base set."""
    bucket_count = len(list_sizes)
    expected_sizes, expected_offsets = _posting_sizes_and_offsets(
        assignment, bucket_count
    )
    if not np.array_equal(list_sizes, expected_sizes) or not np.array_equal(
        posting_offsets, expected_offsets
    ):
        raise ValueError("Fixed Partition artifact posting lists are inconsistent")
    seen = np.zeros(len(assignment), dtype=np.bool_)
    for bucket in range(bucket_count):
        ids = np.asarray(
            posting_ids[posting_offsets[bucket] : posting_offsets[bucket + 1]]
        )
        if ids.size and (
            int(ids.min()) < 0
            or int(ids.max()) >= len(assignment)
            or np.any(seen[ids])
            or len(np.unique(ids)) != len(ids)
            or np.any(assignment[ids] != bucket)
        ):
            raise ValueError("Fixed Partition artifact posting lists are inconsistent")
        seen[ids] = True
    if not np.all(seen):
        raise ValueError("Fixed Partition artifact posting lists are inconsistent")


def _positive_metadata_int(metadata: dict[str, Any], name: str) -> int:
    value = metadata.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Fixed Partition artifact has invalid {name}")
    return value


@dataclass(frozen=True)
class PolicyPartition:
    """Validated routing inputs only; no search index or posting lists."""

    bucket_assignment: npt.NDArray[np.int64]
    list_sizes: npt.NDArray[np.int64]
    metric: Metric
    build_seed: int
    kmeans_niter: int
    filename: PartitionFilename
    dataset_name: str
    base_filename: str | None
    dimension: int

    @property
    def bucket_count(self) -> int:
        return len(self.list_sizes)

    def bucket_ids_for(self, base_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        provided = np.asarray(base_ids)
        if provided.dtype.kind not in ("i", "u"):
            raise ValueError("base IDs must use an integer dtype")
        if provided.size and (
            int(provided.min()) < 0
            or int(provided.max()) >= len(self.bucket_assignment)
        ):
            raise ValueError("base ID is outside the partition assignment range")
        return self.bucket_assignment[provided].copy()


@dataclass(frozen=True, init=False)
class FixedPartition:
    """Read-only assignments and router-directed IVF; v6 has no centroids."""

    bucket_assignment: npt.NDArray[np.int64]
    centroids: npt.NDArray[np.float32] | None
    list_sizes: npt.NDArray[np.int64]
    posting_offsets: npt.NDArray[np.int64]
    posting_ids: npt.NDArray[np.int64]
    metric: Metric
    build_seed: int
    kmeans_niter: int | None
    filename: PartitionFilename
    dataset_name: str
    base_filename: str | None
    _dimension: int
    _index: faiss.IndexIVFFlat

    def __init__(self) -> None:
        raise TypeError("FixedPartition must be created with build() or load()")

    @classmethod
    def _from_validated_components(
        cls,
        *,
        bucket_assignment: npt.NDArray[np.int64],
        centroids: npt.NDArray[np.float32] | None,
        list_sizes: npt.NDArray[np.int64],
        posting_offsets: npt.NDArray[np.int64],
        posting_ids: npt.NDArray[np.int64],
        metric: Metric,
        build_seed: int,
        kmeans_niter: int | None,
        filename: PartitionFilename,
        dataset_name: str,
        base_filename: str | None,
        index: faiss.IndexIVFFlat,
    ) -> "FixedPartition":
        index.parallel_mode = CPU_PREASSIGNED_PARALLEL_MODE
        partition = object.__new__(cls)
        object.__setattr__(partition, "bucket_assignment", bucket_assignment)
        object.__setattr__(partition, "centroids", centroids)
        object.__setattr__(partition, "list_sizes", list_sizes)
        object.__setattr__(partition, "posting_offsets", posting_offsets)
        object.__setattr__(partition, "posting_ids", posting_ids)
        object.__setattr__(partition, "metric", metric)
        object.__setattr__(partition, "build_seed", build_seed)
        object.__setattr__(partition, "kmeans_niter", kmeans_niter)
        object.__setattr__(partition, "filename", filename)
        object.__setattr__(partition, "dataset_name", dataset_name)
        object.__setattr__(partition, "base_filename", base_filename)
        object.__setattr__(partition, "_dimension", int(index.d))
        object.__setattr__(partition, "_index", index)
        return partition

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def bucket_count(self) -> int:
        return int(len(self.list_sizes))

    @property
    def posting_lists(self) -> tuple[npt.NDArray[np.int64], ...]:
        return tuple(
            self.posting_ids[
                self.posting_offsets[bucket] : self.posting_offsets[bucket + 1]
            ]
            for bucket in range(self.bucket_count)
        )

    def bucket_ids_for(self, base_ids: npt.ArrayLike) -> npt.NDArray[np.int64]:
        provided = np.asarray(base_ids)
        if provided.dtype.kind not in ("i", "u"):
            raise ValueError("base IDs must use an integer dtype")
        if provided.size and (
            int(provided.min()) < 0
            or int(provided.max()) >= len(self.bucket_assignment)
        ):
            raise ValueError("base ID is outside the partition assignment range")
        return self.bucket_assignment[provided].copy()

    def compute_ndis(self, probe_plan: ProbePlan) -> npt.NDArray[np.int64]:
        """Return exact posting-list work, not a FAISS runtime counter."""
        active = (
            np.arange(probe_plan.bucket_ids.shape[1])[None, :]
            < (probe_plan.probe_counts[:, None])
        )
        active_buckets = probe_plan.bucket_ids[active]
        if active_buckets.size and (
            int(active_buckets.min()) < 0
            or int(active_buckets.max()) >= self.bucket_count
        ):
            raise ValueError(
                "Probe Plan contains a bucket ID outside the partition range"
            )
        for row, count in zip(
            probe_plan.bucket_ids, probe_plan.probe_counts, strict=True
        ):
            if len(np.unique(row[:count])) != count:
                raise ValueError("Probe Plan contains a duplicate active bucket ID")
        return _readonly(
            np.asarray(
                [
                    self.list_sizes[row[:count]].sum(dtype=np.int64)
                    for row, count in zip(
                        probe_plan.bucket_ids,
                        probe_plan.probe_counts,
                        strict=True,
                    )
                ],
                dtype=np.int64,
            )
        )

    def search_preassigned(
        self,
        queries: npt.ArrayLike,
        probe_plan: ProbePlan | PreparedPaddedProbePlan,
        *,
        topk: int,
        execution: SearchExecution = "grouped",
    ) -> FixedSearchResult:
        total_started = time.perf_counter()
        query_matrix = np.asarray(queries, dtype=np.float32, order="C")
        if query_matrix.ndim != 2 or query_matrix.shape[1] != self.dimension:
            raise ValueError(
                "query dimension mismatch: "
                f"expected (*, {self.dimension}), found {query_matrix.shape}"
            )
        if len(query_matrix) != len(probe_plan.probe_counts):
            raise ValueError("Probe Plan query count does not match queries")
        if topk <= 0:
            raise ValueError("topk must be positive")
        if execution not in ("grouped", "padded"):
            raise ValueError("search execution must be 'grouped' or 'padded'")
        prepared_padded = isinstance(probe_plan, PreparedPaddedProbePlan)
        if prepared_padded and execution != "padded":
            raise ValueError("a prepared padded Probe Plan requires padded execution")

        ndis_started = time.perf_counter()
        ndis = probe_plan.ndis if prepared_padded else self.compute_ndis(probe_plan)
        ndis_seconds = 0.0 if prepared_padded else time.perf_counter() - ndis_started

        distance_fill = np.inf if self.metric == "l2" else -np.inf
        neighbor_ids: npt.NDArray[np.int64] | None = None
        distances: npt.NDArray[np.float32] | None = None
        if execution == "grouped":
            neighbor_ids = np.full((len(query_matrix), topk), -1, dtype=np.int64)
            distances = np.full(
                (len(query_matrix), topk), distance_fill, dtype=np.float32
            )
        search_seconds = 0.0
        packing_seconds = 0.0
        faiss_call_count = 0
        if execution == "grouped":
            groups = (
                (np.flatnonzero(probe_plan.probe_counts == count), int(count))
                for count in np.unique(probe_plan.probe_counts)
            )
        else:
            groups = ((None, probe_plan.bucket_ids.shape[1]),)
        for row_ids, width in groups:
            packing_started = time.perf_counter()
            if row_ids is None:
                routes = probe_plan.bucket_ids[:, :width]
                packed_queries = query_matrix
            else:
                routes = np.ascontiguousarray(
                    probe_plan.bucket_ids[row_ids, :width], dtype=np.int64
                )
                packed_queries = np.ascontiguousarray(query_matrix[row_ids])
            if execution == "padded" and not prepared_padded:
                routes = routes.copy()
                active_widths = probe_plan.probe_counts
                routes[np.arange(width)[None, :] >= active_widths[:, None]] = -1
            coarse = np.zeros(routes.shape, dtype=np.float32)
            packing_seconds += time.perf_counter() - packing_started
            self._index.nprobe = width
            search_started = time.perf_counter()
            group_distances, group_ids = self._index.search_preassigned(
                packed_queries,
                topk,
                routes,
                coarse,
            )
            search_seconds += time.perf_counter() - search_started
            faiss_call_count += 1
            if row_ids is None:
                neighbor_ids = group_ids
                distances = group_distances
            else:
                assert neighbor_ids is not None and distances is not None
                neighbor_ids[row_ids] = group_ids
                distances[row_ids] = group_distances

        assert neighbor_ids is not None and distances is not None
        return FixedSearchResult(
            neighbor_ids=neighbor_ids,
            distances=distances,
            probe_counts=probe_plan.probe_counts,
            ndis=ndis,
            timing=FaissSearchTimingTrace(
                search_seconds=search_seconds,
                total_seconds=time.perf_counter() - total_started,
                packing_seconds=packing_seconds,
                faiss_call_count=faiss_call_count,
                execution=execution,
                parallel_mode=int(self._index.parallel_mode),
                ndis_seconds=ndis_seconds,
            ),
        )

    def save(
        self,
        path: str | Path,
        *,
        base_filename: str | None = None,
        base_path: str | Path | None = None,
    ) -> None:
        """Save compact v6 when an external base is supplied, otherwise full v4."""
        from ft_anns.partition import compact

        artifact_path = Path(path)
        bound_base = (
            base_filename
            or self.base_filename
            or (Path(base_path).name if base_path is not None else None)
        )
        if bound_base is not None and not is_artifact_filename(bound_base):
            raise ValueError("Fixed Partition base filename is invalid")
        if artifact_path.exists():
            raise FileExistsError(
                f"refusing to overwrite Fixed Partition artifact: {artifact_path}"
            )
        if self.kmeans_niter is None:
            raise ValueError("Fixed Partition K-Means niter is missing")
        if base_path is not None and Path(base_path).name != bound_base:
            raise ValueError("compact partition base filename mismatch")
        if base_path is None and self.centroids is None:
            raise ValueError(
                "centroid-free partition requires base_path for compact export; full v4 needs centroids"
            )
        base_reference = (
            compact.prepare_base_reference(self, artifact_path, Path(base_path))
            if base_path is not None
            else None
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            artifact_path.mkdir()
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite Fixed Partition artifact: {artifact_path}"
            ) from error

        arrays = {
            "bucket_assignment": self.bucket_assignment,
            "centroids": self.centroids,
            "list_sizes": self.list_sizes,
            "posting_offsets": self.posting_offsets,
            "posting_ids": self.posting_ids,
        }
        if base_reference is not None:
            arrays = {name: arrays[name] for name in compact.ARRAY_FILES}
        for name, value in arrays.items():
            np.save(artifact_path / _ARRAY_FILES[name], value, allow_pickle=False)
        if base_reference is None:
            faiss.write_index(self._index, str(artifact_path / _INDEX_FILE))

        artifact_files = (
            list(compact.ARRAY_FILES.values())
            if base_reference is not None
            else [*_ARRAY_FILES.values(), _INDEX_FILE]
        )
        saved_filename = PartitionFilename(artifact_filename(artifact_path))
        profile = _partition_profile(
            dataset_name=self.dataset_name,
            bucket_count=self.bucket_count,
            kmeans_niter=self.kmeans_niter,
            build_seed=self.build_seed,
        )
        metadata = {
            "format": compact.FORMAT
            if base_reference is not None
            else _ARTIFACT_FORMAT,
            "filename": saved_filename,
            "compatibility": profile,
            "compatibility_slug": compatibility_slug(profile),
            "dataset_name": self.dataset_name,
            "base_filename": bound_base,
            "metric": self.metric,
            "build_seed": self.build_seed,
            "dimension": self.dimension,
            "base_count": len(self.bucket_assignment),
            "bucket_count": self.bucket_count,
            "files": artifact_files,
        }
        metadata["kmeans_niter"] = self.kmeans_niter
        if base_reference is not None:
            metadata["base_source"] = base_reference
        write_json_atomic(artifact_path / _MANIFEST_FILE, metadata)
        object.__setattr__(self, "filename", saved_filename)
        object.__setattr__(self, "base_filename", bound_base)

    @classmethod
    def load_policy(cls, path: str | Path) -> PolicyPartition:
        """Validate metadata, assignments and sizes without touching search data."""
        result = cls._load(path, policy_only=True)
        assert isinstance(result, PolicyPartition)
        return result

    @classmethod
    def load(cls, path: str | Path, *, base_path: str | Path | None = None) -> "FixedPartition":
        result = cls._load(path, base_path=base_path)
        assert isinstance(result, FixedPartition)
        return result

    @classmethod
    def _load(
        cls, path: str | Path, *, base_path: str | Path | None = None,
        policy_only: bool = False,
    ) -> "FixedPartition | PolicyPartition":
        from ft_anns.partition import compact

        artifact_path = Path(path)
        manifest_path = artifact_path / _MANIFEST_FILE
        if not manifest_path.is_file():
            raise ValueError(
                f"Fixed Partition artifact is not finalized: {artifact_path}"
            )
        try:
            with manifest_path.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                "Fixed Partition artifact manifest is corrupted"
            ) from error
        if not isinstance(metadata, dict):
            raise ValueError("Fixed Partition artifact manifest must be a JSON object")
        if metadata.get("format") not in (
            _ARTIFACT_FORMAT,
            compact.LEGACY_FORMAT,
            compact.FORMAT,
        ):
            raise ValueError("unsupported Fixed Partition artifact format")

        is_compact = metadata["format"] in (compact.LEGACY_FORMAT, compact.FORMAT)
        has_centroids = metadata["format"] != compact.FORMAT
        expected_files = (
            compact.array_files(metadata["format"])
            if is_compact
            else {**_ARRAY_FILES, "index": _INDEX_FILE}
        )
        filenames = metadata.get("files")
        if not isinstance(filenames, list) or set(filenames) != set(
            expected_files.values()
        ):
            raise ValueError("Fixed Partition artifact has an invalid file list")
        checked_files = (["bucket_assignment.npy"] if policy_only else expected_files.values())
        if policy_only and "list_sizes" in expected_files:
            checked_files.append(expected_files["list_sizes"])
        for filename in checked_files:
            component_path = artifact_path / filename
            if not component_path.is_file():
                raise ValueError(f"Fixed Partition artifact is missing {filename}")

        metric = metadata.get("metric")
        if metric not in ("l2", "ip"):
            raise ValueError("Fixed Partition artifact has an invalid metric")
        build_seed = metadata.get("build_seed")
        if not isinstance(build_seed, int) or isinstance(build_seed, bool):
            raise ValueError("Fixed Partition artifact has an invalid build seed")
        kmeans_niter = metadata.get("kmeans_niter")
        if (
            not isinstance(kmeans_niter, int)
            or isinstance(kmeans_niter, bool)
            or kmeans_niter <= 0
        ):
            raise ValueError("Fixed Partition artifact has an invalid K-Means niter")
        try:
            dataset_name = require_dataset_name(
                metadata.get("dataset_name"), allow_synthetic=True
            )
        except ValueError as error:
            raise ValueError(
                "Fixed Partition artifact dataset name is invalid"
            ) from error
        require_compatibility_metadata(
            metadata,
            expected=_partition_profile(
                dataset_name=dataset_name,
                bucket_count=_positive_metadata_int(metadata, "bucket_count"),
                kmeans_niter=int(kmeans_niter),
                build_seed=build_seed,
            ),
            filename=None if dataset_name == "Synthetic8" else artifact_path,
            kind="fixed-partition",
        )
        base_filename = metadata.get("base_filename")
        if base_filename is not None and (
            not isinstance(base_filename, str)
            or not is_artifact_filename(base_filename)
        ):
            raise ValueError("Fixed Partition artifact base filename is invalid")
        base_count = _positive_metadata_int(metadata, "base_count")
        bucket_count = _positive_metadata_int(metadata, "bucket_count")
        dimension = _positive_metadata_int(metadata, "dimension")
        expected_filename = PartitionFilename(artifact_filename(artifact_path))
        if metadata.get("filename") != expected_filename:
            raise ValueError("Fixed Partition artifact filename mismatch")
        if policy_only:
            assignment = np.load(artifact_path / "bucket_assignment.npy", mmap_mode="r", allow_pickle=False)
            if assignment.dtype != np.dtype(np.int64) or assignment.shape != (base_count,):
                raise ValueError("Policy partition assignment shape or dtype mismatch")
            counts = np.zeros(bucket_count, dtype=np.int64)
            for start in range(0, base_count, 1_000_000):
                chunk = assignment[start:start + 1_000_000]
                if np.any(chunk < 0) or np.any(chunk >= bucket_count):
                    raise ValueError("Policy partition assignment is outside bucket range")
                counts += np.bincount(chunk, minlength=bucket_count)
            if "list_sizes" in expected_files:
                sizes = np.load(artifact_path / expected_files["list_sizes"], mmap_mode="r", allow_pickle=False)
                if sizes.dtype != np.dtype(np.int64) or sizes.shape != (bucket_count,) or not np.array_equal(sizes, counts):
                    raise ValueError("Policy partition list sizes do not match assignments")
            else:
                sizes = counts
            if np.any(sizes <= 0):
                raise ValueError("Policy partition contains an empty bucket")
            return PolicyPartition(_readonly(assignment), _readonly(sizes), metric,
                                   build_seed, kmeans_niter, expected_filename,
                                   dataset_name, base_filename, dimension)
        if is_compact:
            arrays, index = compact.load_components(artifact_path, metadata, base_path)
        else:
            try:
                arrays = {
                    name: np.load(
                        artifact_path / filename, mmap_mode="r", allow_pickle=False
                    )
                    for name, filename in _ARRAY_FILES.items()
                }
                index = faiss.read_index(str(artifact_path / _INDEX_FILE))
            except (OSError, ValueError, RuntimeError) as error:
                raise ValueError(
                    "Fixed Partition artifact component is corrupted"
                ) from error
        assignment = arrays["bucket_assignment"]
        centroids = arrays.get("centroids")
        list_sizes = arrays["list_sizes"]
        posting_offsets = arrays["posting_offsets"]
        posting_ids = arrays["posting_ids"]
        if (
            assignment.dtype != np.dtype(np.int64)
            or assignment.shape != (base_count,)
            or (
                has_centroids
                and (
                    centroids is None
                    or centroids.dtype != np.dtype(np.float32)
                    or centroids.shape != (bucket_count, dimension)
                )
            )
            or list_sizes.dtype != np.dtype(np.int64)
            or list_sizes.shape != (bucket_count,)
            or posting_offsets.dtype != np.dtype(np.int64)
            or posting_offsets.shape != (bucket_count + 1,)
            or posting_ids.dtype != np.dtype(np.int64)
            or posting_ids.shape != (base_count,)
        ):
            raise ValueError("Fixed Partition artifact array shape or dtype mismatch")
        if np.any(list_sizes <= 0):
            raise ValueError("Fixed Partition artifact contains an empty bucket")
        if np.any(assignment < 0) or np.any(assignment >= bucket_count):
            raise ValueError("Fixed Partition artifact posting lists are inconsistent")
        _validate_posting_arrays(
            assignment=assignment,
            list_sizes=list_sizes,
            posting_offsets=posting_offsets,
            posting_ids=posting_ids,
        )

        expected_metric = (
            faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        )
        if (
            index.d != dimension
            or index.ntotal != base_count
            or index.nlist != bucket_count
            or index.metric_type != expected_metric
        ):
            raise ValueError("Fixed Partition FAISS index metadata mismatch")
        if has_centroids:
            index_centroids = np.empty((bucket_count, dimension), dtype=np.float32)
            index.quantizer.reconstruct_n(0, bucket_count, index_centroids)
            if not np.array_equal(index_centroids, centroids):
                raise ValueError("Fixed Partition FAISS centroids mismatch")
        elif index.quantizer.ntotal != 0:
            raise ValueError("centroid-free partition must have an empty quantizer")
        for bucket in range(bucket_count):
            ids = faiss.rev_swig_ptr(
                index.invlists.get_ids(bucket), index.invlists.list_size(bucket)
            )
            expected_ids = posting_ids[
                posting_offsets[bucket] : posting_offsets[bucket + 1]
            ]
            if not np.array_equal(ids, expected_ids):
                raise ValueError("Fixed Partition FAISS posting lists mismatch")

        return cls._from_validated_components(
            bucket_assignment=_readonly(assignment),
            centroids=_readonly(centroids) if centroids is not None else None,
            list_sizes=_readonly(list_sizes),
            posting_offsets=_readonly(posting_offsets),
            posting_ids=_readonly(posting_ids),
            metric=metric,
            build_seed=build_seed,
            kmeans_niter=kmeans_niter,
            filename=expected_filename,
            dataset_name=dataset_name,
            base_filename=base_filename,
            index=index,
        )

    @classmethod
    def build(
        cls,
        base_vectors: npt.ArrayLike,
        config: PartitionConfig,
        *,
        metric: Metric = "l2",
        dataset_name: str,
    ) -> "FixedPartition":
        base = np.asarray(base_vectors, dtype=np.float32, order="C")
        if base.ndim != 2 or not len(base) or not base.shape[1]:
            raise ValueError("base vectors must be a non-empty matrix")
        if not np.isfinite(base).all():
            raise ValueError("base vectors must contain only finite values")
        if metric not in ("l2", "ip"):
            raise ValueError(f"unsupported metric for Fixed Partition: {metric!r}")
        try:
            require_dataset_name(dataset_name, allow_synthetic=True)
        except ValueError as error:
            raise ValueError("Fixed Partition dataset name is invalid") from error
        if config.bucket_count is None:
            raise ValueError("bucket_count is required to build a Fixed Partition")
        if config.bucket_count > len(base):
            raise ValueError("bucket_count cannot exceed the base-vector count")
        if config.artifact_path is not None and dataset_name != "Synthetic8":
            profile = _partition_profile(
                dataset_name=dataset_name,
                bucket_count=config.bucket_count,
                kmeans_niter=config.kmeans_niter,
                build_seed=config.build_seed,
            )
            expected_filename = compatibility_filename(profile, kind="fixed-partition")
            actual_filename = artifact_filename(config.artifact_path)
            if actual_filename != expected_filename:
                raise ValueError(
                    "Fixed Partition compatibility filename mismatch: expected "
                    f"{expected_filename!r}, found {actual_filename!r}"
                )

        faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        quantizer = faiss.IndexFlat(base.shape[1], faiss_metric)
        index = faiss.IndexIVFFlat(
            quantizer, base.shape[1], config.bucket_count, faiss_metric
        )
        index.cp.seed = config.build_seed
        index.cp.niter = config.kmeans_niter
        index.cp.min_points_per_centroid = 1
        index.train(base)

        trained_centroids = np.empty(
            (config.bucket_count, base.shape[1]), dtype=np.float32
        )
        index.quantizer.reconstruct_n(0, config.bucket_count, trained_centroids)
        _, raw_assignment = index.quantizer.search(base, 1)
        raw_assignment = raw_assignment.reshape(-1)

        centroid_order = np.lexsort(trained_centroids[:, ::-1].T)
        old_to_new = np.empty(config.bucket_count, dtype=np.int64)
        old_to_new[centroid_order] = np.arange(config.bucket_count)
        assignment = old_to_new[raw_assignment].astype(np.int64, copy=False)
        centroids = trained_centroids[centroid_order]
        list_sizes, posting_offsets = _posting_sizes_and_offsets(
            assignment, config.bucket_count
        )
        if np.any(list_sizes == 0):
            raise ValueError("Fixed Partition build produced an empty bucket")

        canonical_quantizer = faiss.IndexFlat(base.shape[1], faiss_metric)
        canonical_quantizer.add(centroids)
        canonical_index = faiss.IndexIVFFlat(
            canonical_quantizer,
            base.shape[1],
            config.bucket_count,
            faiss_metric,
        )
        canonical_index.is_trained = True
        # swig_ptr does not keep a temporary NumPy array alive for add_core.
        base_ids = np.arange(len(base), dtype=np.int64)
        canonical_index.add_core(
            len(base),
            faiss.swig_ptr(base),
            faiss.swig_ptr(base_ids),
            faiss.swig_ptr(assignment),
        )
        posting_ids = np.empty(len(base), dtype=np.int64)
        for bucket in range(config.bucket_count):
            start = int(posting_offsets[bucket])
            stop = int(posting_offsets[bucket + 1])
            posting_ids[start:stop] = faiss.rev_swig_ptr(
                canonical_index.invlists.get_ids(bucket), stop - start
            )

        assignment = _readonly(assignment)
        centroids = _readonly(centroids)
        list_sizes = _readonly(list_sizes)
        posting_offsets = _readonly(posting_offsets)
        posting_ids = _readonly(posting_ids)
        return cls._from_validated_components(
            bucket_assignment=assignment,
            centroids=centroids,
            list_sizes=list_sizes,
            posting_offsets=posting_offsets,
            posting_ids=posting_ids,
            metric=metric,
            filename=_partition_filename(config),
            build_seed=config.build_seed,
            kmeans_niter=config.kmeans_niter,
            dataset_name=dataset_name,
            base_filename=None,
            index=canonical_index,
        )
