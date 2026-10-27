"""Metadata registry for the supported 1M- and 100M-vector datasets."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from ft_anns.domain import DatasetDType, DatasetFormat, DatasetName, Metric


@dataclass(frozen=True)
class DatasetSpec:
    name: DatasetName
    base_filename: str
    format: DatasetFormat
    dtype: DatasetDType
    dimension: int
    metric: Metric
    item_size: int


@dataclass(frozen=True)
class DatasetPreflight:
    dataset: DatasetName
    path: Path
    file_name: str
    shape: tuple[int, int]
    dtype: DatasetDType
    dimension: int
    metric: Metric
    format: DatasetFormat


_DATASETS = {
    "Deep1M": DatasetSpec(
        "Deep1M", "base.1M.fbin", "fbin", "float32", 96, "l2", 4
    ),
    "Sift1M": DatasetSpec(
        "Sift1M", "sift_base.fvecs", "fvecs", "float32", 128, "l2", 4
    ),
    "SpaceV1M": DatasetSpec(
        "SpaceV1M", "base.1M.i8bin", "i8bin", "int8", 100, "l2", 1
    ),
    "Deep100M": DatasetSpec(
        "Deep100M", "base.100M.fbin", "fbin", "float32", 96, "l2", 4
    ),
    # Sift100M stores int32 coordinates in ivecs records; model consumers cast
    # the coordinates to float32 after decoding the integer payload.
    "Sift100M": DatasetSpec(
        "Sift100M", "sift_base.100M.ivecs", "ivecs", "int32", 128, "l2", 4
    ),
    "SpaceV100M": DatasetSpec(
        "SpaceV100M", "base.100M.i8bin", "i8bin", "int8", 100, "l2", 1
    ),
}


def dataset_spec(name: DatasetName) -> DatasetSpec:
    """Return the canonical format, dimension, and metric for a dataset."""
    try:
        return _DATASETS[name]
    except KeyError as error:
        allowed = ", ".join(sorted(_DATASETS))
        raise ValueError(f"Unknown dataset {name!r}; expected one of: {allowed}") from error


def _read_exact(path: Path, count: int) -> bytes:
    with path.open("rb") as stream:
        value = stream.read(count)
    if len(value) != count:
        raise ValueError(f"{path} is too short to contain a {count}-byte header")
    return value


def _preflight_bin(path: Path, spec: DatasetSpec) -> tuple[int, int]:
    rows, dimension = struct.unpack("<ii", _read_exact(path, 8))
    if rows <= 0 or dimension <= 0:
        raise ValueError(f"{path} has invalid shape header ({rows}, {dimension})")
    expected_bytes = 8 + rows * dimension * spec.item_size
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"{path} size mismatch: header expects {expected_bytes} bytes, "
            f"found {actual_bytes}"
        )
    return rows, dimension


def _preflight_fvecs(path: Path, spec: DatasetSpec) -> tuple[int, int]:
    (dimension,) = struct.unpack("<i", _read_exact(path, 4))
    if dimension <= 0:
        raise ValueError(f"{path} has invalid vector dimension {dimension}")
    record_bytes = 4 + dimension * spec.item_size
    actual_bytes = path.stat().st_size
    if actual_bytes % record_bytes:
        raise ValueError(
            f"{path} size {actual_bytes} is not divisible by record size {record_bytes}"
        )
    rows = actual_bytes // record_bytes
    if rows <= 0:
        raise ValueError(f"{path} contains no vectors")
    return rows, dimension


def preflight_dataset(name: DatasetName, root: str | Path) -> DatasetPreflight:
    """Inspect base-vector metadata without materializing the vector matrix."""
    spec = dataset_spec(name)

    path = Path(root) / spec.base_filename
    if not path.is_file():
        raise FileNotFoundError(f"{name} base vectors not found: {path}")
    if spec.format in ("fvecs", "ivecs"):
        rows, dimension = _preflight_fvecs(path, spec)
    else:
        rows, dimension = _preflight_bin(path, spec)
    if dimension != spec.dimension:
        raise ValueError(
            f"{name} dimension mismatch: expected {spec.dimension}, found {dimension}"
        )
    return DatasetPreflight(
        dataset=name,
        path=path,
        file_name=spec.base_filename,
        shape=(rows, dimension),
        dtype=spec.dtype,
        dimension=dimension,
        metric=spec.metric,
        format=spec.format,
    )
