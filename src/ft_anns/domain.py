"""Shared domain vocabulary used by config and dataset modules."""

from typing import Literal, TypeAlias


DatasetName: TypeAlias = Literal[
    "Deep1M",
    "Sift1M",
    "SpaceV1M",
    "Deep100M",
    "Sift100M",
    "SpaceV100M",
]
BaseGroundTruthDatasetName: TypeAlias = DatasetName | Literal["Synthetic8"]
Metric: TypeAlias = Literal["l2", "ip"]
DatasetFormat: TypeAlias = Literal["fbin", "fvecs", "ivecs", "bvecs", "i8bin"]
DatasetDType: TypeAlias = Literal["float32", "int32", "int8"]
SearchExecution: TypeAlias = Literal["grouped", "padded"]
SearchExecutionTrace: TypeAlias = SearchExecution | Literal["mixed"]
RoutingRanking: TypeAlias = Literal["raw_logits", "ramps_size_matched"]


def require_dataset_name(value: object, *, allow_synthetic: bool = False) -> str:
    """Return a supported dataset name or reject untrusted artifact metadata."""
    supported = {
        "Deep1M",
        "Sift1M",
        "SpaceV1M",
        "Deep100M",
        "Sift100M",
        "SpaceV100M",
    }
    if allow_synthetic:
        supported.add("Synthetic8")
    if not isinstance(value, str) or value not in supported:
        raise ValueError("dataset name is invalid")
    return value
