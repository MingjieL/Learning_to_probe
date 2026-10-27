"""Value objects exchanged at the Fixed Partition seam."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ft_anns.domain import SearchExecution


def _readonly_copy(value: npt.ArrayLike, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    array = np.frombuffer(memoryview(copied).toreadonly(), dtype=copied.dtype).reshape(
        copied.shape
    )
    array.flags.writeable = False
    return array


@dataclass(frozen=True)
class ProbePlan:
    """Explicit bucket routes and the active route count for each query."""

    bucket_ids: npt.NDArray[np.int64]
    probe_counts: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        bucket_ids = _readonly_copy(self.bucket_ids, np.int64)
        probe_counts = _readonly_copy(self.probe_counts, np.int64)
        if bucket_ids.ndim != 2:
            raise ValueError("Probe Plan bucket_ids must be a matrix")
        if probe_counts.ndim != 1 or len(probe_counts) != len(bucket_ids):
            raise ValueError("Probe Plan requires one probe count per query")
        if np.any(probe_counts <= 0) or np.any(probe_counts > bucket_ids.shape[1]):
            raise ValueError("Probe Plan counts must be within the route width")
        object.__setattr__(self, "bucket_ids", bucket_ids)
        object.__setattr__(self, "probe_counts", probe_counts)


@dataclass(frozen=True)
class PreparedPaddedProbePlan:
    """Partition-bound routes already padded and accompanied by exact NDis."""

    bucket_ids: npt.NDArray[np.int64]
    probe_counts: npt.NDArray[np.int64]
    ndis: npt.NDArray[np.int64]

    def __post_init__(self) -> None:
        bucket_ids = _readonly_copy(self.bucket_ids, np.int64)
        probe_counts = _readonly_copy(self.probe_counts, np.int64)
        ndis = _readonly_copy(self.ndis, np.int64)
        if bucket_ids.ndim != 2:
            raise ValueError("prepared Probe Plan bucket_ids must be a matrix")
        if probe_counts.shape != (len(bucket_ids),) or ndis.shape != (len(bucket_ids),):
            raise ValueError("prepared Probe Plan requires counts and NDis per query")
        if np.any(probe_counts <= 0) or np.any(probe_counts > bucket_ids.shape[1]):
            raise ValueError("prepared Probe Plan counts must be within route width")
        if np.any(ndis < 0):
            raise ValueError("prepared Probe Plan NDis must be non-negative")
        inactive = np.arange(bucket_ids.shape[1])[None, :] >= probe_counts[:, None]
        if np.any(bucket_ids[inactive] != -1):
            raise ValueError("prepared Probe Plan inactive routes must be -1")
        object.__setattr__(self, "bucket_ids", bucket_ids)
        object.__setattr__(self, "probe_counts", probe_counts)
        object.__setattr__(self, "ndis", ndis)


@dataclass(frozen=True)
class FaissSearchTimingTrace:
    search_seconds: float
    total_seconds: float
    packing_seconds: float = 0.0
    faiss_call_count: int = 0
    execution: SearchExecution = "grouped"
    parallel_mode: int = 0
    ndis_seconds: float = 0.0


@dataclass(frozen=True)
class FixedSearchResult:
    neighbor_ids: npt.NDArray[np.int64]
    distances: npt.NDArray[np.float32]
    probe_counts: npt.NDArray[np.int64]
    ndis: npt.NDArray[np.int64]
    timing: FaissSearchTimingTrace

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "neighbor_ids", _readonly_copy(self.neighbor_ids, np.int64)
        )
        object.__setattr__(
            self, "distances", _readonly_copy(self.distances, np.float32)
        )
        object.__setattr__(
            self, "probe_counts", _readonly_copy(self.probe_counts, np.int64)
        )
        object.__setattr__(self, "ndis", _readonly_copy(self.ndis, np.int64))
