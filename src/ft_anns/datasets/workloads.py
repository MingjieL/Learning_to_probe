"""Reproducible workload-role assignment and immutable manifests."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import numpy.typing as npt

from ft_anns.artifact_io import artifact_filename, write_json_atomic
from ft_anns.compatibility import (
    compatibility_filename,
    compatibility_slug,
    require_compatibility_metadata,
)
from ft_anns.config import BaseGroundTruthConfig, WorkloadConfig
from ft_anns.datasets.base_gt import (
    BaseGroundTruthArtifact,
    build_base_ground_truth,
)
from ft_anns.datasets.registry import dataset_spec
from ft_anns.domain import require_dataset_name

_ROLE_NAMES = (
    "router_train",
    "checkpoint_validation",
    "policy_cost_fit",
    "policy_tuning",
    "policy_certification",
)
_GROUND_TRUTH_ROLE_NAMES = tuple(
    name for name in _ROLE_NAMES if name != "policy_cost_fit"
)


def _identity_query_ids(values: npt.NDArray[Any], *, chunk_size: int = 1_000_000) -> bool:
    """Check an identity row map without allocating an equally large arange."""
    if values.ndim != 1 or values.dtype.kind not in ("i", "u"):
        return False
    for start in range(0, len(values), chunk_size):
        stop = min(start + chunk_size, len(values))
        if not np.array_equal(values[start:stop], np.arange(start, stop)):
            return False
    return True


def _ground_truth_query_ids(role_ids: dict[str, list[int]]) -> list[int]:
    return [
        value
        for name in _GROUND_TRUTH_ROLE_NAMES
        for value in role_ids.get(name, ())
    ]


@dataclass(frozen=True)
class _QueryGroundTruthArtifact:
    path: Path
    metadata: dict[str, Any]
    query_ids: np.ndarray[Any, Any]
    neighbor_ids: np.ndarray[Any, Any]


def _load_query_ground_truth(path: str | Path) -> _QueryGroundTruthArtifact:
    artifact_path = Path(path)
    metadata_path = artifact_path / "finalized.json"
    if not metadata_path.is_file():
        raise ValueError(f"Query Ground-Truth Artifact is not finalized: {artifact_path}")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format") != "query-ground-truth-v2":
        raise ValueError("unsupported Query Ground-Truth Artifact format")
    if metadata.get("status") != "finalized":
        raise ValueError("Query Ground-Truth Artifact metadata is not finalized")
    if metadata.get("artifact_filename") != artifact_filename(artifact_path):
        raise ValueError("Query Ground-Truth Artifact filename mismatch")
    query_ids = np.load(
        artifact_path / "query_ids.npy", mmap_mode="r", allow_pickle=False
    )
    neighbor_ids = np.load(
        artifact_path / "neighbor_ids.npy", mmap_mode="r", allow_pickle=False
    )
    shape = metadata.get("shape")
    try:
        require_dataset_name(metadata.get("dataset_name"))
    except ValueError as error:
        raise ValueError(
            "Query Ground-Truth Artifact dataset name is invalid"
        ) from error
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or not all(isinstance(value, int) and value > 0 for value in shape)
        or query_ids.ndim != 1
        or query_ids.dtype.kind not in ("i", "u")
        or query_ids.shape != (shape[0],)
        or neighbor_ids.dtype.kind not in ("i", "u")
        or neighbor_ids.shape != tuple(shape)
    ):
        raise ValueError("Query Ground-Truth Artifact metadata or arrays mismatch")
    return _QueryGroundTruthArtifact(
        path=artifact_path,
        metadata=metadata,
        query_ids=query_ids,
        neighbor_ids=neighbor_ids,
    )


def _base_count(path: Path, format_name: str, dimension: int) -> int:
    if not path.is_file():
        raise FileNotFoundError(f"base-vector file not found: {path}")
    if format_name in ("fbin", "i8bin"):
        with path.open("rb") as stream:
            header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"{path} is too short to contain a vector header")
        count, found_dimension = struct.unpack("<ii", header)
        item_size = 4 if format_name == "fbin" else 1
        expected_size = 8 + count * found_dimension * item_size
    elif format_name in ("fvecs", "ivecs", "bvecs"):
        item_size = 1 if format_name == "bvecs" else 4
        record_size = 4 + dimension * item_size
        count = path.stat().st_size // record_size
        found_dimension = dimension
        expected_size = count * record_size
    else:
        raise ValueError(f"unsupported base-vector format: {format_name}")
    if (
        count <= 0
        or found_dimension != dimension
        or path.stat().st_size != expected_size
    ):
        raise ValueError(f"{path} has an invalid {format_name} shape or size")
    return int(count)


def _materialize_query_ground_truth_prefix(
    *,
    config: WorkloadConfig,
    source_count: int,
    query_source_filename: str,
    base_count: int,
) -> _QueryGroundTruthArtifact:
    """Import the first K columns of an existing ivecs GT without loading it all."""
    output_path = Path(config.materialized_ground_truth_output_path)
    if output_path.exists():
        return _load_query_ground_truth(output_path)
    source_path_value = config.ground_truth.source_path
    if source_path_value is None:
        raise ValueError("raw query ground truth source_path is missing")
    source_path = Path(source_path_value)
    if source_path.suffix != ".ivecs":
        raise ValueError("raw query ground truth must be an .ivecs file")
    from ft_anns.datasets.base_gt import read_ivecs

    source = read_ivecs(source_path)
    if len(source) != source_count:
        raise ValueError(
            "raw query ground-truth row count does not match the query source"
        )
    if source.shape[1] < config.ground_truth.k:
        raise ValueError("raw query ground truth has fewer than requested K columns")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f".{output_path.name}.in-progress")
    progress_path = partial_path / "progress.json"
    contract = {
        "format": "query-ground-truth-import-progress-v1",
        "dataset_name": config.dataset,
        "base_filename": config.base_filename,
        "query_source_filename": query_source_filename,
        "source_filename": artifact_filename(source_path),
        "source_size": source_path.stat().st_size,
        "source_count": source_count,
        "source_k": int(source.shape[1]),
        "retained_k": config.ground_truth.k,
    }
    if partial_path.exists():
        try:
            with progress_path.open(encoding="utf-8") as stream:
                progress = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"query ground-truth import progress is unreadable: {partial_path}"
            ) from error
        if progress.get("contract") != contract:
            raise ValueError("query ground-truth import progress is incompatible")
        completed_rows = int(progress.get("completed_rows", -1))
        query_ids = np.lib.format.open_memmap(
            partial_path / "query_ids.npy", mode="r+"
        )
        neighbors = np.lib.format.open_memmap(
            partial_path / "neighbor_ids.npy", mode="r+"
        )
        if (
            completed_rows < 0
            or completed_rows > source_count
            or query_ids.dtype != np.dtype(np.int64)
            or query_ids.shape != (source_count,)
            or neighbors.dtype != np.dtype(np.int32)
            or neighbors.shape != (source_count, config.ground_truth.k)
        ):
            raise ValueError("query ground-truth import progress arrays are invalid")
    else:
        partial_path.mkdir()
        query_ids = np.lib.format.open_memmap(
            partial_path / "query_ids.npy",
            mode="w+",
            dtype=np.int64,
            shape=(source_count,),
        )
        neighbors = np.lib.format.open_memmap(
            partial_path / "neighbor_ids.npy",
            mode="w+",
            dtype=np.int32,
            shape=(source_count, config.ground_truth.k),
        )
        completed_rows = 0
        write_json_atomic(
            progress_path,
            {"contract": contract, "completed_rows": completed_rows},
        )
    chunk_size = 100_000
    for start in range(completed_rows, source_count, chunk_size):
        stop = min(start + chunk_size, source_count)
        selected = np.asarray(source[start:stop, : config.ground_truth.k])
        if selected.size and (
            int(selected.min()) < 0 or int(selected.max()) >= base_count
        ):
            raise ValueError(
                "raw query ground truth contains a neighbor outside the base range"
            )
        query_ids[start:stop] = np.arange(start, stop, dtype=np.int64)
        neighbors[start:stop] = selected
        query_ids.flush()
        neighbors.flush()
        write_json_atomic(
            progress_path,
            {"contract": contract, "completed_rows": stop},
        )
    metadata = {
        "format": "query-ground-truth-v2",
        "status": "finalized",
        "dataset_name": config.dataset,
        "base_filename": config.base_filename,
        "query_source_filename": query_source_filename,
        "artifact_filename": artifact_filename(output_path),
        "metric": config.ground_truth.metric,
        "k": config.ground_truth.k,
        "shape": [source_count, config.ground_truth.k],
        "self_exclusion": "none",
        "source": {
            "filename": artifact_filename(source_path),
            "format": "ivecs",
            "source_k": int(source.shape[1]),
            "retained_k": config.ground_truth.k,
        },
    }
    write_json_atomic(partial_path / "finalized.json", metadata)
    partial_path.replace(output_path)
    return _load_query_ground_truth(output_path)


def _nested_role_ids(
    config: WorkloadConfig, source_count: int
) -> dict[str, npt.NDArray[np.int64]]:
    validation_count = int(config.roles.checkpoint_validation.count or 0)
    # Put fixed-size roles before the configurable tuning role. A full seeded
    # permutation makes every role a stable prefix/slice when only the tuning
    # count changes, which keeps Validation, Router Train, and Cost-Fit fixed.
    policy_roles = ("policy_cost_fit", "policy_certification", "policy_tuning")
    policy_counts = {
        name: int(getattr(config.roles, name).count)  # type: ignore[union-attr]
        for name in policy_roles
    }
    router_count = source_count - validation_count
    configured_router_count = int(config.roles.router_train.count or 0)
    if configured_router_count != router_count:
        raise ValueError(
            "router_train count must equal query-source count minus "
            "checkpoint_validation count"
        )
    sampled_count = validation_count + sum(policy_counts.values())
    if sampled_count > source_count:
        raise ValueError("nested workload roles exceed the query source")
    sampled = np.random.default_rng(config.sampling.seed).permutation(source_count)[
        :sampled_count
    ].astype(np.int64, copy=False)
    result: dict[str, npt.NDArray[np.int64]] = {
        "checkpoint_validation": sampled[:validation_count]
    }
    offset = validation_count
    for name in policy_roles:
        count = policy_counts[name]
        result[name] = sampled[offset : offset + count]
        offset += count
    return result


def _role_ids(config: WorkloadConfig, source_count: int) -> dict[str, list[int]]:
    if config.sampling.strategy == "explicit_ids":
        explicit_result: dict[str, list[int]] = {}
        for name in _ROLE_NAMES:
            role = getattr(config.roles, name)
            if role is not None:
                assert role.ids is not None
                explicit_result[name] = list(role.ids)
        return explicit_result

    counts: list[tuple[str, int]] = []
    for name in _ROLE_NAMES:
        role = getattr(config.roles, name)
        if role is not None:
            assert role.count is not None
            counts.append((name, role.count))
    total = sum(count for _, count in counts)
    if total > source_count:
        raise ValueError(
            f"workload roles request {total} IDs from a source with {source_count} rows"
        )
    if config.sampling.strategy == "random_without_replacement":
        assigned = np.random.default_rng(config.sampling.seed).choice(
            source_count, size=total, replace=False
        )
    else:
        assert config.sampling.start is not None
        stride = config.sampling.stride or 1
        assigned = config.sampling.start + np.arange(total) * stride
    result: dict[str, list[int]] = {}
    start = 0
    for name, count in counts:
        result[name] = assigned[start : start + count].astype(int).tolist()
        start += count
    return result


def _validate_role_ids(role_ids: dict[str, list[int]], source_count: int) -> None:
    owner_by_id: dict[int, str] = {}
    for role_name, ids in role_ids.items():
        for query_id in ids:
            if query_id < 0 or query_id >= source_count:
                raise ValueError(
                    f"query ID is outside the query-source range: {query_id}"
                )
            previous_owner = owner_by_id.get(query_id)
            if previous_owner is not None:
                raise ValueError(
                    f"workload role ID overlap for {query_id}: "
                    f"{previous_owner} and {role_name}"
                )
            owner_by_id[query_id] = role_name


def _validate_ground_truth(
    artifact: BaseGroundTruthArtifact,
    config: WorkloadConfig,
    role_ids: dict[str, list[int]],
    base_count: int,
) -> None:
    _validate_ground_truth_compatibility(artifact, config)
    row_by_query_id = {
        int(query_id): row for row, query_id in enumerate(artifact.query_ids)
    }
    for query_id in _ground_truth_query_ids(role_ids):
        if query_id not in row_by_query_id:
            raise ValueError(f"ground-truth artifact is missing query ID {query_id}")
        neighbors = np.asarray(
            artifact.neighbor_ids[row_by_query_id[query_id], : config.ground_truth.k]
        )
        if len(neighbors) != config.ground_truth.k or np.any(neighbors < 0):
            raise ValueError(
                f"ground-truth row {query_id} has fewer than K valid neighbors"
            )
        if np.any(neighbors >= base_count):
            raise ValueError(
                f"ground-truth row {query_id} contains an ID outside the base range"
            )
        if len(np.unique(neighbors)) != len(neighbors):
            raise ValueError(
                f"ground-truth row {query_id} contains duplicate IDs"
            )


def _validate_ground_truth_compatibility(
    artifact: BaseGroundTruthArtifact, config: WorkloadConfig
) -> None:
    metadata = artifact.metadata
    if metadata.get("dataset_name") != config.dataset:
        raise ValueError("ground-truth dataset mismatch")
    if metadata.get("base_filename") != config.base_filename:
        raise ValueError("ground-truth base filename mismatch")
    if metadata.get("metric") != config.ground_truth.metric:
        raise ValueError("ground-truth metric mismatch")
    if metadata.get("self_exclusion") != "none":
        raise ValueError(
            "ground-truth self-exclusion mismatch; expected base-to-base "
            "neighbors with self exclusion disabled"
        )
    if int(metadata.get("k", 0)) < config.ground_truth.k:
        raise ValueError("ground-truth rows contain fewer than the requested K neighbors")


def _selected_ground_truth(
    config: WorkloadConfig,
    role_ids: dict[str, list[int]],
    specification: Any,
) -> BaseGroundTruthArtifact:
    query_ids = _ground_truth_query_ids(role_ids)
    return build_base_ground_truth(
        base=None,
        query_ids=np.asarray(query_ids, dtype=np.int64),
        config=BaseGroundTruthConfig(
            dataset=config.dataset,
            base_path=config.base_path,
            format=specification.format,
            dimension=specification.dimension,
            metric=config.ground_truth.metric,
            k=config.ground_truth.k,
            device=config.ground_truth.device,
            gpu_ids=config.ground_truth.gpu_ids,
            index_add_batch_size=config.ground_truth.index_add_batch_size,
            query_batch_size=config.ground_truth.query_batch_size,
            output_path=config.materialized_ground_truth_output_path,
        ),
    )


def _validate_query_ground_truth(
    artifact: _QueryGroundTruthArtifact,
    config: WorkloadConfig,
    role_ids: dict[str, list[int]],
    query_source_filename: str,
    base_count: int,
) -> None:
    metadata = artifact.metadata
    expected = (
        ("dataset_name", config.dataset),
        ("base_filename", config.base_filename),
        ("query_source_filename", query_source_filename),
        ("metric", config.ground_truth.metric),
        ("self_exclusion", "none"),
    )
    for field, value in expected:
        if metadata.get(field) != value:
            raise ValueError(f"query ground-truth {field.replace('_', ' ')} mismatch")
    if int(metadata.get("k", 0)) < config.ground_truth.k:
        raise ValueError("query ground-truth rows contain fewer than requested K neighbors")
    if artifact.query_ids.dtype.kind not in ("i", "u"):
        raise ValueError("query ground-truth query IDs must use an integer dtype")
    if artifact.neighbor_ids.dtype.kind not in ("i", "u"):
        raise ValueError("query ground-truth neighbor IDs must use an integer dtype")
    if artifact.neighbor_ids.ndim != 2 or artifact.neighbor_ids.shape != tuple(
        metadata.get("shape", ())
    ):
        raise ValueError("query ground-truth shape mismatch")
    if len(artifact.query_ids) != len(artifact.neighbor_ids):
        raise ValueError("query ground-truth query-ID row count mismatch")
    if _identity_query_ids(artifact.query_ids):
        required = np.unique(
            np.asarray(_ground_truth_query_ids(role_ids), dtype=np.int64)
        )
        if required.size and int(required[-1]) >= len(artifact.query_ids):
            raise ValueError(
                f"query ground-truth artifact is missing query ID {int(required[-1])}"
            )
        rows = required
    else:
        row_by_query_id = {
            int(query_id): row for row, query_id in enumerate(artifact.query_ids)
        }
        if len(row_by_query_id) != len(artifact.query_ids):
            raise ValueError("query ground-truth query IDs must be unique")
        try:
            rows = np.asarray(
                [row_by_query_id[value] for value in _ground_truth_query_ids(role_ids)],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"query ground-truth artifact is missing query ID {int(error.args[0])}"
            ) from error
    for start in range(0, len(rows), 100_000):
        selected_rows = rows[start : start + 100_000]
        neighbors = np.asarray(
            artifact.neighbor_ids[selected_rows, : config.ground_truth.k]
        )
        if neighbors.shape != (len(selected_rows), config.ground_truth.k) or (
            neighbors.size
            and (int(neighbors.min()) < 0 or int(neighbors.max()) >= base_count)
        ):
            query_id = int(artifact.query_ids[int(selected_rows[0])])
            raise ValueError(
                f"query ground-truth row {query_id} has fewer than K valid neighbors"
            )


@dataclass(frozen=True)
class WorkloadManifest:
    """Read-only view of a finalized workload manifest."""

    path: Path
    metadata: dict[str, Any]

    @property
    def filename(self) -> str:
        return str(self.metadata["manifest_filename"])

    @property
    def router_training_contract(self) -> dict[str, Any] | None:
        """Return the v4 contract that binds Router Training to its holdout."""
        if self.metadata.get("format") != "workload-manifest-v4":
            return None
        contract = self.metadata.get("router_training_contract")
        if not isinstance(contract, dict):
            raise ValueError("Workload Manifest Router Training contract is missing")
        return dict(contract)

    def validate_checkpoint_binding(self, checkpoint_metadata: Mapping[str, Any]) -> None:
        """Reject a checkpoint trained against another workload contract."""
        if checkpoint_metadata.get("workload_manifest_filename") != self.filename:
            raise ValueError("checkpoint Workload Manifest filename mismatch")
        contract = self.router_training_contract
        if contract is not None and checkpoint_metadata.get(
            "router_training_workload_contract"
        ) != contract:
            raise ValueError("checkpoint Router Training workload contract mismatch")

    @property
    def role_ids(self) -> dict[str, tuple[int, ...]]:
        return {
            name: tuple(int(value) for value in self.role_query_ids(name))
            for name in self.metadata["roles"]
        }

    def role_query_ids(self, role: str) -> npt.NDArray[np.int64]:
        """Resolve one role, including compact complement selectors in v4."""
        try:
            definition = self.metadata["roles"][role]
        except KeyError as error:
            raise ValueError(f"Workload Manifest does not define {role}") from error
        if "query_ids" in definition:
            return np.asarray(definition["query_ids"], dtype=np.int64)
        if "query_ids_file" in definition:
            relative_path = Path(str(definition["query_ids_file"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"Workload role {role} ID artifact path is invalid")
            values = np.load(
                self.path.parent / relative_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            if values.dtype != np.dtype(np.int64) or values.shape != (
                int(definition["count"]),
            ):
                raise ValueError(f"Workload role {role} ID artifact is invalid")
            return values
        selection = definition.get("selection")
        if isinstance(selection, dict) and selection.get("kind") == "complement":
            source_count = int(self.metadata["query_source_count"])
            mask = np.ones(source_count, dtype=np.bool_)
            for excluded_role in selection.get("excluded_roles", []):
                mask[self.role_query_ids(str(excluded_role))] = False
            values = np.flatnonzero(mask).astype(np.int64, copy=False)
            if len(values) != int(definition["count"]):
                raise ValueError(f"Workload role {role} complement count mismatch")
            return values
        raise ValueError(f"Workload role {role} has an unsupported selector")

    def role_count(self, role: str) -> int:
        try:
            return int(self.metadata["roles"][role]["count"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Workload Manifest does not define {role}") from error

    @classmethod
    def load(cls, path: str | Path) -> "WorkloadManifest":
        manifest_path = Path(path)
        with manifest_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        if metadata.get("format") not in (
            "workload-manifest-v3",
            "workload-manifest-v4",
        ):
            raise ValueError("unsupported Workload Manifest format")
        if metadata.get("status") != "finalized":
            raise ValueError("Workload Manifest is not finalized")
        filename = metadata.get("manifest_filename")
        if filename != artifact_filename(manifest_path):
            raise ValueError("Workload Manifest filename mismatch")
        dataset_name = metadata.get("dataset_name")
        ground_truth_metadata = metadata.get("ground_truth")
        require_compatibility_metadata(
            metadata,
            expected={
                "dataset_name": dataset_name,
                "ground_truth_k": (
                    ground_truth_metadata.get("k")
                    if isinstance(ground_truth_metadata, dict)
                    else None
                ),
            },
            filename=manifest_path,
            kind="workloads",
        )
        resolved_config = metadata.get("resolved_config")
        try:
            require_dataset_name(dataset_name)
        except ValueError as error:
            raise ValueError("Workload Manifest dataset metadata is invalid") from error
        if (
            not isinstance(resolved_config, dict)
            or resolved_config.get("dataset") != dataset_name
        ):
            raise ValueError("Workload Manifest dataset metadata is invalid")
        specification = dataset_spec(dataset_name)
        ground_truth = metadata["ground_truth"]
        source = metadata.get("resolved_config", {}).get("source")
        loaded_ground_truth: BaseGroundTruthArtifact | _QueryGroundTruthArtifact
        if source == "base_set":
            if ground_truth.get("self_exclusion") != "none":
                raise ValueError(
                    "base-set Workload Manifest uses unsupported self-exclusion "
                    "semantics"
                )
            loaded_ground_truth = BaseGroundTruthArtifact.load(
                ground_truth["artifact_path"]
            )
        elif source == "query_file":
            loaded_ground_truth = _load_query_ground_truth(
                ground_truth["artifact_path"]
            )
        else:
            raise ValueError("Workload Manifest has an unsupported query source")
        if artifact_filename(ground_truth["artifact_path"]) != ground_truth[
            "artifact_filename"
        ]:
            raise ValueError("ground-truth artifact filename mismatch")
        if (
            loaded_ground_truth.metadata.get("dataset_name") != dataset_name
            or loaded_ground_truth.metadata.get("metric") != specification.metric
            or int(loaded_ground_truth.metadata.get("k", 0))
            < int(ground_truth.get("k", 0))
        ):
            raise ValueError("Workload Manifest ground-truth metadata mismatch")
        manifest = cls(path=manifest_path, metadata=metadata)
        if metadata.get("format") == "workload-manifest-v4":
            _validate_v4_manifest_roles(manifest)
            expected_contract = _router_training_contract(metadata)
            if manifest.router_training_contract != expected_contract:
                raise ValueError("Workload Manifest Router Training contract mismatch")
        return manifest


def _router_training_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    roles = metadata.get("roles")
    if not isinstance(roles, Mapping):
        raise ValueError("Workload Manifest roles metadata is invalid")
    ground_truth = metadata.get("ground_truth")
    if not isinstance(ground_truth, Mapping):
        raise ValueError("Workload Manifest ground-truth metadata is invalid")
    router_train = roles.get("router_train")
    checkpoint_validation = roles.get("checkpoint_validation")
    if not isinstance(router_train, Mapping) or not isinstance(
        checkpoint_validation, Mapping
    ):
        raise ValueError("Workload Manifest Router Training roles are invalid")
    return {
        "format": "router-training-workload-contract-v1",
        "dataset_name": metadata.get("dataset_name"),
        "base_filename": metadata.get("base_filename"),
        "source": metadata.get("source"),
        "query_source_filename": metadata.get("query_source_filename"),
        "query_source_format": metadata.get("query_source_format"),
        "query_source_count": metadata.get("query_source_count"),
        "sampling": metadata.get("sampling"),
        "ground_truth": {
            "mode": ground_truth.get("mode"),
            "source_filename": ground_truth.get("source_filename"),
            "artifact_filename": ground_truth.get("artifact_filename"),
            "k": ground_truth.get("k"),
            "metric": ground_truth.get("metric"),
            "self_exclusion": ground_truth.get("self_exclusion"),
        },
        "roles": {
            "router_train": {
                "count": router_train.get("count"),
                "selection": router_train.get("selection"),
            },
            "checkpoint_validation": {
                "count": checkpoint_validation.get("count"),
            },
        },
    }


def _validate_v4_manifest_roles(manifest: WorkloadManifest) -> None:
    source_count = manifest.metadata.get("query_source_count")
    if not isinstance(source_count, int) or isinstance(source_count, bool):
        raise ValueError("Workload Manifest query source count is invalid")
    if source_count <= 0:
        raise ValueError("Workload Manifest query source count is invalid")
    roles = manifest.metadata.get("roles")
    required_roles = {
        "router_train",
        "checkpoint_validation",
        "policy_cost_fit",
        "policy_tuning",
        "policy_certification",
    }
    if not isinstance(roles, dict) or set(roles) != required_roles:
        raise ValueError("Workload Manifest v4 roles are invalid")
    if any(not isinstance(roles[role], dict) for role in required_roles):
        raise ValueError("Workload Manifest v4 role definitions are invalid")
    router_definition = roles["router_train"]
    selection = router_definition.get("selection")
    if not isinstance(selection, dict) or selection != {
        "kind": "complement",
        "excluded_roles": ["checkpoint_validation"],
    }:
        raise ValueError("Router Training selector is invalid")

    explicit_roles = (
        "checkpoint_validation",
        "policy_cost_fit",
        "policy_tuning",
        "policy_certification",
    )
    ids_by_role: dict[str, npt.NDArray[np.int64]] = {}
    for role in explicit_roles:
        values = manifest.role_query_ids(role)
        if values.size and (int(values.min()) < 0 or int(values.max()) >= source_count):
            raise ValueError(f"Workload role {role} query IDs are out of range")
        if len(np.unique(values)) != len(values):
            raise ValueError(f"Workload role {role} query IDs are not unique")
        ids_by_role[role] = values

    validation_ids = ids_by_role["checkpoint_validation"]
    if int(router_definition.get("count", -1)) != source_count - len(validation_ids):
        raise ValueError("Router Training complement count mismatch")
    policy_roles = (
        "policy_cost_fit",
        "policy_tuning",
        "policy_certification",
    )
    for index, role in enumerate(policy_roles):
        values = ids_by_role[role]
        if np.intersect1d(values, validation_ids, assume_unique=True).size:
            raise ValueError(f"Workload role {role} is not a Router Training subset")
        for other_role in policy_roles[index + 1 :]:
            if np.intersect1d(
                values,
                ids_by_role[other_role],
                assume_unique=True,
            ).size:
                raise ValueError("Policy workload roles are not mutually disjoint")


def load_role_ground_truth(
    manifest: WorkloadManifest, role: str, *, query_limit: int | None = None
) -> npt.NDArray[np.int64]:
    """Load exactly the manifest-declared Ground-Truth K for one query role."""
    if role not in _GROUND_TRUTH_ROLE_NAMES:
        raise ValueError(f"workload role does not carry ground truth: {role}")
    if role not in manifest.metadata["roles"]:
        raise ValueError(f"Workload Manifest does not define {role}")

    ground_truth_metadata = manifest.metadata["ground_truth"]
    ground_truth_k = int(ground_truth_metadata["k"])
    artifact_path = Path(ground_truth_metadata["artifact_path"])
    artifact_query_ids = np.load(
        artifact_path / "query_ids.npy", mmap_mode="r", allow_pickle=False
    )
    artifact_neighbors = np.load(
        artifact_path / "neighbor_ids.npy", mmap_mode="r", allow_pickle=False
    )
    query_ids = np.asarray(manifest.role_query_ids(role), dtype=np.int64)
    if query_limit is not None:
        if isinstance(query_limit, bool) or not isinstance(query_limit, int) or not 0 < query_limit <= len(query_ids):
            raise ValueError("query_limit must be positive and within the role count")
        query_ids = query_ids[:query_limit]
    if (
        len(artifact_query_ids) == len(artifact_neighbors)
        and _identity_query_ids(artifact_query_ids)
    ):
        rows = query_ids
    else:
        row_by_query_id = {
            int(query_id): row for row, query_id in enumerate(artifact_query_ids)
        }
        try:
            rows = np.asarray(
                [row_by_query_id[int(query_id)] for query_id in query_ids],
                dtype=np.int64,
            )
        except KeyError as error:
            raise ValueError(
                f"ground-truth artifact is missing query ID {int(error.args[0])}"
            ) from error
    ground_truth = np.asarray(
        artifact_neighbors[rows, :ground_truth_k], dtype=np.int64
    )
    if ground_truth.shape != (len(query_ids), ground_truth_k):
        raise ValueError("ground-truth artifact has fewer than the manifest K")
    return ground_truth


def prepare_workloads(config: WorkloadConfig) -> WorkloadManifest:
    """Assign role IDs once and persist their finalized manifest."""
    profile = {
        "dataset_name": config.dataset,
        "ground_truth_k": config.ground_truth.k,
    }
    expected_manifest_filename = compatibility_filename(
        profile, kind="workloads", extension=".json"
    )
    actual_manifest_filename = artifact_filename(config.manifest_output_path)
    if actual_manifest_filename != expected_manifest_filename:
        raise ValueError(
            "Workload Manifest compatibility filename mismatch: expected "
            f"{expected_manifest_filename!r}, found {actual_manifest_filename!r}"
        )
    manifest_path = Path(config.manifest_output_path)
    expected_ground_truth_filename = compatibility_filename(
        profile, kind="query-gt"
    )
    actual_ground_truth_filename = artifact_filename(
        config.materialized_ground_truth_output_path
    )
    if actual_ground_truth_filename != expected_ground_truth_filename:
        raise ValueError(
            "materialized ground-truth compatibility filename mismatch: expected "
            f"{expected_ground_truth_filename!r}, found "
            f"{actual_ground_truth_filename!r}"
        )
    base_path = Path(config.base_path)
    specification = dataset_spec(config.dataset)
    if config.ground_truth.metric != specification.metric:
        raise ValueError(
            f"dataset metric mismatch: {config.dataset} requires "
            f"{specification.metric!r}, found {config.ground_truth.metric!r}"
        )
    base_count = _base_count(
        base_path, specification.format, specification.dimension
    )
    base_filename = artifact_filename(base_path)
    if base_filename != config.base_filename:
        raise ValueError("base filename mismatch")
    if config.source == "base_set":
        source_count = base_count
        query_source_filename = base_filename
        query_source_format = specification.format
    else:
        assert config.query_path is not None
        query_path = Path(config.query_path)
        query_format = config.query_format or specification.format
        source_count = _base_count(
            query_path, query_format, specification.dimension
        )
        query_source_filename = artifact_filename(query_path)
        query_source_format = query_format
    nested_sampling = (
        config.sampling.strategy == "router_complement_with_policy_subsets"
    )
    if nested_sampling:
        nested_role_ids = _nested_role_ids(config, source_count)
        role_ids = {
            name: values.astype(int).tolist()
            for name, values in nested_role_ids.items()
        }
        _validate_role_ids(role_ids, source_count)
    else:
        role_ids = _role_ids(config, source_count)
        _validate_role_ids(role_ids, source_count)
    if config.source == "query_file":
        if config.ground_truth.mode == "import_ivecs_prefix":
            query_artifact = _materialize_query_ground_truth_prefix(
                config=config,
                source_count=source_count,
                query_source_filename=query_source_filename,
                base_count=base_count,
            )
        else:
            if config.ground_truth.artifact_path is None:
                raise ValueError(
                    "query_file workloads require a ground-truth artifact_path"
                )
            query_artifact = _load_query_ground_truth(
                config.ground_truth.artifact_path
            )
        _validate_query_ground_truth(
            query_artifact,
            config,
            role_ids,
            query_source_filename,
            base_count,
        )
        artifact_path = query_artifact.path
        self_exclusion = "none"
    else:
        artifact: BaseGroundTruthArtifact
        if config.ground_truth.artifact_path is None:
            artifact = _selected_ground_truth(config, role_ids, specification)
        else:
            artifact = BaseGroundTruthArtifact.load(config.ground_truth.artifact_path)
            _validate_ground_truth_compatibility(artifact, config)
            required_ids = set(_ground_truth_query_ids(role_ids))
            available_ids = {int(value) for value in artifact.query_ids}
            if not required_ids.issubset(available_ids):
                artifact = _selected_ground_truth(config, role_ids, specification)
        _validate_ground_truth(artifact, config, role_ids, base_count)
        artifact_path = artifact.path
        self_exclusion = "none"

    profile = {**profile, "dataset_name": specification.name}
    if manifest_path.exists():
        existing_manifest = WorkloadManifest.load(manifest_path)
        if (
            existing_manifest.metadata.get("format") == "workload-manifest-v4"
            and existing_manifest.metadata.get("resolved_config")
            != config.model_dump(mode="json")
        ):
            raise ValueError(
                "existing workload-manifest-v4 was prepared from a different "
                "configuration; use a separate manifest output directory"
            )
        return existing_manifest
    if nested_sampling:
        role_directory = manifest_path.with_suffix(".roles")
        role_directory.mkdir(parents=True, exist_ok=True)
        role_metadata: dict[str, Any] = {
            "router_train": {
                "count": source_count - len(role_ids["checkpoint_validation"]),
                "selection": {
                    "kind": "complement",
                    "excluded_roles": ["checkpoint_validation"],
                },
            }
        }
        for name, values in nested_role_ids.items():
            role_path = role_directory / f"{name}.npy"
            np.save(role_path, values, allow_pickle=False)
            role_metadata[name] = {
                "count": len(values),
                "query_ids_file": str(role_path.relative_to(manifest_path.parent)),
            }
    else:
        role_metadata = {
            name: {"count": len(ids), "query_ids": ids}
            for name, ids in role_ids.items()
        }

    metadata: dict[str, Any] = {
        "format": (
            "workload-manifest-v4" if nested_sampling else "workload-manifest-v3"
        ),
        "status": "finalized",
        "compatibility": profile,
        "compatibility_slug": compatibility_slug(profile),
        "resolved_config": config.model_dump(mode="json"),
        "dataset_name": specification.name,
        "base_filename": base_filename,
        "source": config.source,
        "query_source_filename": query_source_filename,
        "query_source_format": query_source_format,
        "query_source_count": source_count,
        "sampling": config.sampling.model_dump(mode="json"),
        "roles": role_metadata,
        "ground_truth": {
            "artifact_path": str(artifact_path),
            "artifact_filename": artifact_filename(artifact_path),
            "mode": config.ground_truth.mode,
            "source_filename": (
                artifact_filename(config.ground_truth.source_path)
                if config.ground_truth.source_path is not None
                else None
            ),
            "k": config.ground_truth.k,
            "metric": config.ground_truth.metric,
            "self_exclusion": self_exclusion,
        },
    }
    metadata["manifest_filename"] = artifact_filename(manifest_path)
    if nested_sampling:
        metadata["router_training_contract"] = _router_training_contract(metadata)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, metadata)
    return WorkloadManifest.load(manifest_path)
