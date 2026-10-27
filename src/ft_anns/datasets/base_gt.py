"""Resumable exact base-to-base ground truth for base-derived queries."""

from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt

from ft_anns.artifact_io import artifact_filename, write_json_atomic
from ft_anns.compatibility import (
    compatibility_filename,
    compatibility_slug,
    require_compatibility_metadata,
)
from ft_anns.config import BaseGroundTruthConfig
from ft_anns.datasets.registry import dataset_spec
from ft_anns.domain import require_dataset_name


_FINALIZED_FILE = "finalized.json"
_PROGRESS_FILE = "progress.json"
_QUERY_IDS_FILE = "query_ids.npy"
_NEIGHBOR_IDS_FILE = "neighbor_ids.npy"
_DISTANCES_FILE = "distances.npy"


class _VectorSource:
    count: int
    dimension: int
    dtype: np.dtype[Any]
    filename: str
    description: dict[str, object]

    def read_slice(self, start: int, stop: int) -> npt.NDArray[Any]:
        raise NotImplementedError

    def read_rows(self, row_ids: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        raise NotImplementedError


class _ArrayVectorSource(_VectorSource):
    def __init__(self, values: npt.ArrayLike) -> None:
        array = np.asarray(values)
        if array.ndim != 2:
            raise ValueError("base vectors must be a matrix")
        self._array = array
        self.count, self.dimension = map(int, array.shape)
        self.dtype = array.dtype
        self.filename = "in_memory"
        self.description = {
            "kind": "in_memory",
            "shape": [self.count, self.dimension],
            "dtype": str(self.dtype),
        }

    def read_slice(self, start: int, stop: int) -> npt.NDArray[Any]:
        return self._array[start:stop]

    def read_rows(self, row_ids: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        return self._array[row_ids]


class FileVectorSource(_VectorSource):
    def __init__(self, path: Path, format_name: str, expected_dimension: int) -> None:
        if not path.is_file():
            raise FileNotFoundError(f"base-vector file not found: {path}")
        self._path = path
        self._format = format_name
        self._vectors: npt.NDArray[Any] | None
        self._records: npt.NDArray[np.uint8] | None
        self.filename = artifact_filename(path)

        if format_name in ("fbin", "i8bin"):
            with path.open("rb") as stream:
                header = stream.read(8)
            if len(header) != 8:
                raise ValueError(f"{path} is too short to contain a vector header")
            count, dimension = struct.unpack("<ii", header)
            dtype = np.dtype("<f4" if format_name == "fbin" else "i1")
            expected_bytes = 8 + count * dimension * dtype.itemsize
            if count <= 0 or dimension <= 0 or path.stat().st_size != expected_bytes:
                raise ValueError(f"{path} has an invalid {format_name} shape or size")
            self._vectors = np.memmap(
                path,
                mode="r",
                dtype=dtype,
                offset=8,
                shape=(count, dimension),
            )
            self._records = None
        elif format_name == "ivecs":
            file_size = path.stat().st_size
            record_items = expected_dimension + 1
            record_bytes = record_items * np.dtype("<i4").itemsize
            if file_size == 0 or file_size % record_bytes:
                raise ValueError(f"{path} has an invalid ivecs file size")
            count = file_size // record_bytes
            dimension = expected_dimension
            records = np.memmap(
                path,
                mode="r",
                dtype="<i4",
                shape=(count, record_items),
            )
            sample_count = min(count, 128)
            sample_ids = np.linspace(
                0, count - 1, num=sample_count, dtype=np.int64
            )
            if np.any(records[sample_ids, 0] != expected_dimension):
                raise ValueError(
                    f"{path} contains an inconsistent ivecs dimension"
                )
            # Keep a strided view over the memory map. FixedPartition.build
            # performs the one required int32-to-float32 conversion; avoiding
            # an eager contiguous payload copy saves about 51 GB for Sift100M.
            self._vectors = records[:, 1:]
            self._records = None
            dtype = np.dtype("<i4")
        elif format_name in ("fvecs", "bvecs"):
            file_size = path.stat().st_size
            item_size = 4 if format_name == "fvecs" else 1
            record_bytes = 4 + expected_dimension * item_size
            if file_size == 0 or file_size % record_bytes:
                raise ValueError(f"{path} has an invalid {format_name} file size")
            count = file_size // record_bytes
            dimension = expected_dimension
            records = np.memmap(
                path,
                mode="r",
                dtype=np.uint8,
                shape=(count, record_bytes),
            )
            # Validating every interleaved record header would page through the
            # entire 50+ GB SIFT100M file on each process load. File size proves
            # record alignment; sample up to 128 evenly spaced headers as a
            # scalable corruption check (small fixtures still check every row).
            sample_count = min(count, 128)
            sample_ids = np.linspace(
                0, count - 1, num=sample_count, dtype=np.int64
            )
            headers = (
                np.ascontiguousarray(records[sample_ids, :4])
                .view("<i4")
                .reshape(-1)
            )
            if np.any(headers != expected_dimension):
                raise ValueError(
                    f"{path} contains an inconsistent {format_name} dimension"
                )
            self._records = records
            self._vectors = None
            dtype = np.dtype("<f4" if format_name == "fvecs" else "u1")
        else:
            raise ValueError(f"unsupported base-vector format: {format_name}")

        if dimension != expected_dimension:
            raise ValueError(
                f"base dimension mismatch: expected {expected_dimension}, "
                f"found {dimension}"
            )
        self.count = int(count)
        self.dimension = int(dimension)
        self.dtype = dtype
        self.description = {
            "kind": "file",
            "path": str(path),
            "format": format_name,
            "shape": [self.count, self.dimension],
            "dtype": str(dtype),
        }

    def _record_rows(self, rows: npt.NDArray[np.uint8]) -> npt.NDArray[Any]:
        values = np.ascontiguousarray(rows[:, 4:])
        if self._format == "fvecs":
            return values.view("<f4").reshape(len(rows), self.dimension)
        return values.reshape(len(rows), self.dimension)

    def read_slice(self, start: int, stop: int) -> npt.NDArray[Any]:
        if self._vectors is not None:
            return self._vectors[start:stop]
        assert self._records is not None
        return self._record_rows(self._records[start:stop])

    def read_rows(self, row_ids: npt.NDArray[np.int64]) -> npt.NDArray[Any]:
        if self._vectors is not None:
            return self._vectors[row_ids]
        assert self._records is not None
        return self._record_rows(self._records[row_ids])


def read_ivecs(path: str | Path) -> npt.NDArray[np.int32]:
    """Return a validated row-aligned view of an ivecs file."""
    ivecs_path = Path(path)
    file_size = ivecs_path.stat().st_size if ivecs_path.is_file() else 0
    if file_size < 8 or file_size % 4:
        raise ValueError(f"{ivecs_path} has an invalid ivecs file size")
    flat = np.memmap(ivecs_path, mode="r", dtype="<i4")
    width = int(flat[0])
    record_width = width + 1
    if width <= 0 or len(flat) % record_width:
        raise ValueError(f"{ivecs_path} has an invalid ivecs row width")
    records = flat.reshape(-1, record_width)
    sample_count = min(len(records), 128)
    sample_ids = np.linspace(
        0, len(records) - 1, num=sample_count, dtype=np.int64
    )
    if np.any(records[sample_ids, 0] != width):
        raise ValueError(f"{ivecs_path} has inconsistent ivecs row widths")
    return records[:, 1:]


class _IvecsNeighborSource:
    def __init__(self, path: Path, expected_rows: int) -> None:
        if path.suffix != ".ivecs":
            raise ValueError(f"existing neighbor IDs must be an .ivecs file: {path}")
        values = read_ivecs(path)
        if len(values) != expected_rows:
            raise ValueError(
                "existing neighbor-ID row count does not match the base set"
            )
        self._values = values
        self.width = int(values.shape[1])
        self.filename = artifact_filename(path)
        self.description = {
            "path": str(path),
            "format": "ivecs",
            "shape": [len(values), self.width],
            "filename": self.filename,
        }

    def read_rows(
        self, row_ids: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int32]:
        return np.asarray(self._values[row_ids], dtype=np.int32)


@dataclass(frozen=True)
class BaseGroundTruthArtifact:
    """Read-only view of one finalized Base Ground-Truth Artifact."""

    path: Path
    metadata: dict[str, Any]
    query_ids: npt.NDArray[np.int64]
    neighbor_ids: npt.NDArray[np.integer[Any]]
    distances: npt.NDArray[np.float32] | None

    @property
    def completed(self) -> bool:
        return self.metadata.get("status") == "finalized"

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self.metadata["shape"]
        return int(rows), int(columns)

    @classmethod
    def load(cls, path: str | Path) -> "BaseGroundTruthArtifact":
        artifact_path = Path(path)
        metadata_path = artifact_path / _FINALIZED_FILE
        if not metadata_path.is_file():
            raise ValueError(
                f"Base Ground-Truth Artifact is not finalized: {artifact_path}"
            )
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("format") != "base-ground-truth-v3":
            raise ValueError("unsupported Base Ground-Truth Artifact format")
        if metadata.get("status") != "finalized":
            raise ValueError("Base Ground-Truth Artifact metadata is not finalized")
        if metadata.get("artifact_filename") != artifact_filename(artifact_path):
            raise ValueError("Base Ground-Truth Artifact filename mismatch")
        artifact_kind = metadata.get("artifact_kind")
        if artifact_kind not in ("base-gt", "query-gt"):
            raise ValueError("Base Ground-Truth Artifact kind is invalid")
        require_compatibility_metadata(
            metadata,
            expected={
                "dataset_name": metadata.get("dataset_name"),
                "ground_truth_k": metadata.get("k"),
            },
            filename=(
                None
                if metadata.get("dataset_name") == "Synthetic8"
                else artifact_path
            ),
            kind=artifact_kind,
        )
        if metadata.get("self_exclusion") != "none":
            raise ValueError(
                "unsupported Base Ground-Truth self-exclusion semantics; expected "
                "base-to-base neighbors with self exclusion disabled"
            )

        try:
            dataset_name = require_dataset_name(
                metadata.get("dataset_name"), allow_synthetic=True
            )
        except ValueError as error:
            raise ValueError(
                "Base Ground-Truth Artifact dataset name is invalid"
            ) from error
        semantic_config = metadata.get("semantic_config")
        resolved_config = metadata.get("resolved_config")
        if (
            not isinstance(semantic_config, dict)
            or semantic_config.get("dataset_name") != dataset_name
            or semantic_config.get("metric") != metadata.get("metric")
            or semantic_config.get("k") != metadata.get("k")
            or not isinstance(resolved_config, dict)
            or resolved_config.get("dataset") != dataset_name
        ):
            raise ValueError(
                "Base Ground-Truth Artifact compatibility metadata mismatch"
            )
        shape = metadata.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(isinstance(value, int) and value > 0 for value in shape)
        ):
            raise ValueError("Base Ground-Truth Artifact shape metadata is invalid")
        try:
            neighbor_dtype = np.dtype(metadata["neighbor_id_dtype"])
        except (KeyError, TypeError) as error:
            raise ValueError(
                "Base Ground-Truth Artifact neighbor dtype is invalid"
            ) from error
        if neighbor_dtype.kind not in ("i", "u"):
            raise ValueError("Base Ground-Truth Artifact neighbor dtype is invalid")

        query_ids = np.load(
            artifact_path / _QUERY_IDS_FILE, mmap_mode="r", allow_pickle=False
        )
        neighbor_ids = np.load(
            artifact_path / _NEIGHBOR_IDS_FILE, mmap_mode="r", allow_pickle=False
        )
        save_distances = metadata.get("save_distances")
        if not isinstance(save_distances, bool):
            raise ValueError(
                "Base Ground-Truth Artifact distance metadata is invalid"
            )
        distances_path = artifact_path / _DISTANCES_FILE
        distances = (
            np.load(distances_path, mmap_mode="r", allow_pickle=False)
            if save_distances
            else None
        )
        if (
            query_ids.ndim != 1
            or query_ids.dtype.kind not in ("i", "u")
            or query_ids.shape != (shape[0],)
            or neighbor_ids.shape != tuple(shape)
            or neighbor_ids.dtype != neighbor_dtype
        ):
            raise ValueError(
                "Base Ground-Truth Artifact array shape or dtype mismatch"
            )
        if distances is not None and (
            distances.shape != tuple(shape) or distances.dtype != np.dtype(np.float32)
        ):
            raise ValueError(
                "Base Ground-Truth Artifact distance shape or dtype mismatch"
            )
        return cls(
            path=artifact_path,
            metadata=metadata,
            query_ids=query_ids,
            neighbor_ids=neighbor_ids,
            distances=distances,
        )


def _selected_query_ids(
    config: BaseGroundTruthConfig, base_count: int
) -> npt.NDArray[np.int64]:
    if config.scope == "all":
        query_ids = np.arange(base_count, dtype=np.int64)
    else:
        assert config.selected_ids_path is not None
        selected_path = Path(config.selected_ids_path)
        if selected_path.suffix == ".npy":
            loaded_ids = np.asarray(np.load(selected_path))
            if loaded_ids.dtype.kind not in ("i", "u"):
                raise ValueError("query IDs must use an integer dtype")
            query_ids = loaded_ids.astype(np.int64, copy=False)
        else:
            query_ids = np.asarray(np.loadtxt(selected_path, dtype=np.int64))
    return _validate_query_ids(query_ids, base_count)


def _validate_query_ids(
    query_ids: npt.ArrayLike, base_count: int
) -> npt.NDArray[np.int64]:
    provided = np.asarray(query_ids)
    if provided.dtype.kind not in ("i", "u"):
        raise ValueError("query IDs must use an integer dtype")
    resolved = provided.astype(np.int64, copy=False).reshape(-1)
    if resolved.size == 0:
        raise ValueError("query IDs must not be empty")
    if int(resolved.min()) < 0 or int(resolved.max()) >= base_count:
        raise ValueError("query ID is outside the base vector range")
    if len(np.unique(resolved)) != len(resolved):
        raise ValueError("query IDs must be unique")
    return resolved


def _semantic_config(
    config: BaseGroundTruthConfig,
    existing_neighbor_ids_filename: str | None,
) -> dict[str, object]:
    semantic: dict[str, object] = {
        "dataset_name": config.dataset,
        "format": config.format,
        "dimension": config.dimension,
        "metric": config.metric,
        "k": config.k,
        "save_distances": config.save_distances,
        "neighbor_id_dtype": config.output_dtype,
        "self_exclusion": "none",
    }
    if existing_neighbor_ids_filename is not None:
        semantic["neighbor_source"] = "existing_ivecs"
        semantic["existing_neighbor_ids_filename"] = (
            existing_neighbor_ids_filename
        )
    return semantic


def _compatibility_metadata(
    base_filename: str,
    config: BaseGroundTruthConfig,
    existing_neighbor_ids_filename: str | None,
    *,
    artifact_kind: str,
) -> dict[str, object]:
    semantic_config = _semantic_config(config, existing_neighbor_ids_filename)
    query_ids_filename = (
        artifact_filename(config.selected_ids_path)
        if config.selected_ids_path is not None
        else _QUERY_IDS_FILE
    )
    compatibility: dict[str, object] = {
        "base_filename": base_filename,
        "query_ids_filename": query_ids_filename,
        "artifact_filename": artifact_filename(config.output_path),
        "semantic_config": semantic_config,
    }
    profile = {
        "dataset_name": config.dataset,
        "ground_truth_k": config.k,
    }
    compatibility["compatibility"] = profile
    compatibility["compatibility_slug"] = compatibility_slug(profile)
    compatibility["artifact_kind"] = artifact_kind
    if existing_neighbor_ids_filename is not None:
        compatibility["existing_neighbor_ids_filename"] = (
            existing_neighbor_ids_filename
        )
    return compatibility


def _open_vector_source(
    base: npt.ArrayLike | None, config: BaseGroundTruthConfig
) -> _VectorSource:
    if base is not None:
        return _ArrayVectorSource(base)
    return FileVectorSource(Path(config.base_path), config.format, config.dimension)


def _validate_dataset_metric(
    config: BaseGroundTruthConfig, source: _VectorSource
) -> None:
    if config.dataset == "Synthetic8":
        if not isinstance(source, _ArrayVectorSource):
            raise ValueError("Synthetic8 is only valid for in-memory base fixtures")
        return
    expected_metric = dataset_spec(config.dataset).metric
    if config.metric != expected_metric:
        raise ValueError(
            f"dataset metric mismatch: {config.dataset} requires "
            f"{expected_metric!r}, found {config.metric!r}"
        )


def _validate_compatibility(
    saved: dict[str, Any], current: dict[str, object], config: BaseGroundTruthConfig
) -> None:
    if saved.get("base_filename") != current["base_filename"]:
        raise ValueError("base filename mismatch; refusing to resume artifact")
    if saved.get("query_ids_filename") != current["query_ids_filename"]:
        raise ValueError("query-ID filename mismatch; refusing to resume artifact")
    if saved.get("artifact_filename") != current["artifact_filename"]:
        raise ValueError("artifact filename mismatch; refusing to resume artifact")
    if saved.get("metric") != config.metric:
        raise ValueError("metric mismatch; refusing to resume artifact")
    if saved.get("k") != config.k:
        raise ValueError("k mismatch; refusing to resume artifact")
    if saved.get("semantic_config") != current["semantic_config"]:
        raise ValueError(
            "semantic config mismatch; refusing to resume artifact"
        )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _completed_prefix(progress: dict[str, Any], row_count: int) -> int:
    ranges = progress.get("completed_ranges")
    if not isinstance(ranges, list):
        raise ValueError("partial progress has invalid completed row ranges")
    completed = 0
    for item in ranges:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or item[0] != completed
            or not isinstance(item[1], int)
            or item[1] <= completed
            or item[1] > row_count
        ):
            raise ValueError("partial progress has invalid completed row ranges")
        completed = item[1]
    return completed


def _execution_gpu_ids(config: BaseGroundTruthConfig) -> list[int]:
    if config.device == "cpu":
        if config.gpu_ids:
            raise ValueError("gpu_ids are only valid for CUDA execution adapters")
        return []
    gpu_count = faiss.get_num_gpus()
    if gpu_count <= 0:
        raise RuntimeError(
            f"{config.device} requested but no FAISS GPU is available"
        )
    if config.device == "cuda":
        if len(config.gpu_ids) > 1:
            raise ValueError("device='cuda' accepts at most one GPU ID")
        gpu_ids = config.gpu_ids or [0]
    else:
        gpu_ids = config.gpu_ids or list(range(gpu_count))
    if len(set(gpu_ids)) != len(gpu_ids) or any(
        gpu_id < 0 or gpu_id >= gpu_count for gpu_id in gpu_ids
    ):
        raise ValueError(
            f"GPU IDs must be unique values in [0, {gpu_count - 1}]"
        )
    return gpu_ids


def _exact_index(
    dimension: int,
    metric: str,
    device: str,
    gpu_ids: list[int],
) -> tuple[faiss.Index, object | None]:
    if metric == "l2":
        cpu_index = faiss.IndexFlatL2(dimension)
    elif metric == "ip":
        cpu_index = faiss.IndexFlatIP(dimension)
    else:
        raise ValueError(f"unsupported exact-search metric: {metric}")
    if device == "cpu":
        return cpu_index, None
    if device == "cuda":
        resources = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(resources, gpu_ids[0], cpu_index), resources
    return faiss.index_cpu_to_gpus_list(cpu_index, gpus=gpu_ids), None


def build_base_ground_truth(
    *,
    base: npt.ArrayLike | None,
    query_ids: npt.ArrayLike | None,
    config: BaseGroundTruthConfig,
) -> BaseGroundTruthArtifact:
    """Build full-base or selected-row exact base-to-base ground truth."""
    invocation_started = time.perf_counter()
    source = _open_vector_source(base, config)
    if source.dimension != config.dimension:
        raise ValueError(
            f"base dimension mismatch: expected {config.dimension}, "
            f"found {source.dimension}"
        )
    base_count = source.count
    if config.k > base_count:
        raise ValueError(
            f"k={config.k} requires at least {config.k} base vectors; "
            f"found {base_count}"
        )

    if query_ids is None:
        resolved_query_ids = _selected_query_ids(config, base_count)
        resolved_scope = config.scope
    else:
        resolved_query_ids = _validate_query_ids(query_ids, base_count)
        resolved_scope = "selected_ids"

    artifact_kind = (
        "query-gt" if resolved_scope == "selected_ids" else "base-gt"
    )
    output_path = Path(config.output_path)
    if config.dataset != "Synthetic8":
        expected_filename = compatibility_filename(
            {
                "dataset_name": config.dataset,
                "ground_truth_k": config.k,
            },
            kind=artifact_kind,
        )
        if output_path.name != expected_filename:
            raise ValueError(
                "Base Ground-Truth Artifact compatibility filename mismatch: "
                f"expected {expected_filename!r}, found {output_path.name!r}"
            )

    existing_neighbors = (
        _IvecsNeighborSource(
            Path(config.existing_neighbor_ids_path), expected_rows=base_count
        )
        if config.existing_neighbor_ids_path is not None
        else None
    )
    gpu_ids = [] if existing_neighbors is not None else _execution_gpu_ids(config)
    output_path.mkdir(parents=True, exist_ok=True)
    compatibility = _compatibility_metadata(
        source.filename,
        config,
        (
            existing_neighbors.filename
            if existing_neighbors is not None
            else None
        ),
        artifact_kind=artifact_kind,
    )
    finalized_path = output_path / _FINALIZED_FILE
    progress_path = output_path / _PROGRESS_FILE
    finalized: dict[str, Any] | None = None
    progress: dict[str, Any] | None = None
    if finalized_path.is_file():
        finalized = _load_json(finalized_path)
        _validate_compatibility(finalized, compatibility, config)
    elif progress_path.is_file():
        if not config.resume:
            raise ValueError("partial artifact exists but resume is disabled")
        progress = _load_json(progress_path)
        if progress.get("format") != "base-ground-truth-progress-v2":
            raise ValueError("unsupported Base Ground-Truth progress format")
        _validate_compatibility(progress, compatibility, config)

    _validate_dataset_metric(config, source)
    if finalized is not None:
        return BaseGroundTruthArtifact.load(output_path)

    output_dtype = np.dtype(config.output_dtype)
    expected_shape = (len(resolved_query_ids), config.k)
    if progress is not None:
        completed_rows = _completed_prefix(progress, len(resolved_query_ids))
        neighbor_ids = np.lib.format.open_memmap(
            output_path / _NEIGHBOR_IDS_FILE,
            mode="r+",
        )
        if neighbor_ids.shape != expected_shape or neighbor_ids.dtype != output_dtype:
            raise ValueError("partial neighbor-ID artifact shape or dtype mismatch")
        if config.save_distances:
            distances = np.lib.format.open_memmap(
                output_path / _DISTANCES_FILE,
                mode="r+",
            )
            if distances.shape != expected_shape or distances.dtype != np.float32:
                raise ValueError("partial distance artifact shape or dtype mismatch")
        else:
            distances = None
    else:
        existing = [
            name
            for name in (_QUERY_IDS_FILE, _NEIGHBOR_IDS_FILE, _DISTANCES_FILE)
            if (output_path / name).exists()
        ]
        if existing:
            raise ValueError(
                "artifact data exists without progress metadata: " + ", ".join(existing)
            )
        np.save(output_path / _QUERY_IDS_FILE, resolved_query_ids)
        neighbor_ids = np.lib.format.open_memmap(
            output_path / _NEIGHBOR_IDS_FILE,
            mode="w+",
            dtype=output_dtype,
            shape=expected_shape,
        )
        distances = (
            np.lib.format.open_memmap(
                output_path / _DISTANCES_FILE,
                mode="w+",
                dtype=np.float32,
                shape=expected_shape,
            )
            if config.save_distances
            else None
        )
        completed_rows = 0
        progress = {
            "format": "base-ground-truth-progress-v2",
            "status": "in_progress",
            "dataset_name": config.dataset,
            "metric": config.metric,
            "k": config.k,
            "scope": resolved_scope,
            "shape": list(expected_shape),
            "neighbor_id_dtype": config.output_dtype,
            "distance_dtype": "float32" if config.save_distances else None,
            "completed_ranges": [],
            "elapsed_seconds": 0.0,
            **compatibility,
        }
        write_json_atomic(progress_path, progress)

    elapsed_before_invocation = float(progress.get("elapsed_seconds", 0.0))
    if completed_rows == len(resolved_query_ids):
        progress["status"] = "complete"
    else:
        index: faiss.Index | None = None
        resources: object | None = None
        if existing_neighbors is None:
            index, resources = _exact_index(
                config.dimension, config.metric, config.device, gpu_ids
            )
            for start in range(0, base_count, config.index_add_batch_size):
                index.add(
                    np.ascontiguousarray(
                        source.read_slice(
                            start,
                            min(start + config.index_add_batch_size, base_count),
                        ),
                        dtype=np.float32,
                    )
                )

        search_k = config.k
        for start in range(
            completed_rows, len(resolved_query_ids), config.query_batch_size
        ):
            stop = min(start + config.query_batch_size, len(resolved_query_ids))
            batch_ids = resolved_query_ids[start:stop]
            if existing_neighbors is None:
                assert index is not None
                query_vectors = np.ascontiguousarray(
                    source.read_rows(batch_ids), dtype=np.float32
                )
                batch_distances, batch_neighbors = index.search(
                    query_vectors, search_k
                )
            else:
                batch_neighbors = existing_neighbors.read_rows(batch_ids)
                batch_distances = None
            for offset, _query_id in enumerate(batch_ids):
                if np.any(batch_neighbors[offset] < 0) or np.any(
                    batch_neighbors[offset] >= base_count
                ):
                    raise ValueError(
                        "existing neighbor-ID row contains an ID outside the base range"
                    )
                row_neighbors = batch_neighbors[offset][: config.k]
                if len(row_neighbors) != config.k or np.any(row_neighbors < 0):
                    raise RuntimeError(
                        f"exact search returned fewer than {config.k} neighbors"
                    )
                if len(np.unique(row_neighbors)) != config.k:
                    raise ValueError("neighbor-ID row contains duplicate IDs")
                neighbor_ids[start + offset] = row_neighbors
                if distances is not None:
                    assert batch_distances is not None
                    row_distances = batch_distances[offset][: config.k]
                    distances[start + offset] = row_distances
            neighbor_ids.flush()
            if distances is not None:
                distances.flush()
            progress["completed_ranges"].append([start, stop])
            progress["status"] = (
                "complete" if stop == len(resolved_query_ids) else "in_progress"
            )
            progress["elapsed_seconds"] = elapsed_before_invocation + (
                time.perf_counter() - invocation_started
            )
            write_json_atomic(progress_path, progress)

    progress["elapsed_seconds"] = elapsed_before_invocation + (
        time.perf_counter() - invocation_started
    )
    progress["status"] = "complete"
    write_json_atomic(progress_path, progress)
    metadata = {
        "format": "base-ground-truth-v3",
        "status": "finalized",
        "dataset_name": config.dataset,
        "metric": config.metric,
        "k": config.k,
        "scope": resolved_scope,
        "shape": [len(resolved_query_ids), config.k],
        "neighbor_id_dtype": config.output_dtype,
        "distance_dtype": "float32" if config.save_distances else None,
        "save_distances": config.save_distances,
        "self_exclusion": "none",
        "completed_ranges": progress["completed_ranges"],
        "base_source": source.description,
        "execution": {
            "backend": (
                "ivecs_reuse" if existing_neighbors is not None else "faiss_exact"
            ),
            "device": config.device,
            "gpu_ids": gpu_ids,
            "index_add_batch_size": config.index_add_batch_size,
            "query_batch_size": config.query_batch_size,
            "index_add_batches": (
                0
                if existing_neighbors is not None
                else (base_count + config.index_add_batch_size - 1)
                // config.index_add_batch_size
            ),
            "completed_query_batches": len(progress["completed_ranges"]),
            "elapsed_seconds": progress["elapsed_seconds"],
        },
        "resolved_config": config.model_dump(mode="json"),
        **compatibility,
    }
    write_json_atomic(finalized_path, metadata)
    return BaseGroundTruthArtifact.load(output_path)
