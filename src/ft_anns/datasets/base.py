"""Small in-memory dataset adapter for CPU integration tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class SyntheticDataset:
    base_vectors: npt.NDArray[np.float32]
    query_vectors: npt.NDArray[np.float32]
    ground_truth: npt.NDArray[np.int64]
    metric: str = "l2"
    name: str = "Synthetic8"

    def __post_init__(self) -> None:
        base = np.array(self.base_vectors, dtype=np.float32, copy=True, order="C")
        queries = np.array(self.query_vectors, dtype=np.float32, copy=True, order="C")
        ground_truth = np.array(self.ground_truth, dtype=np.int64, copy=True, order="C")
        if base.ndim != 2 or queries.ndim != 2 or ground_truth.ndim != 2:
            raise ValueError("synthetic base, query, and ground-truth inputs must be matrices")
        if base.shape[1] != queries.shape[1]:
            raise ValueError("synthetic base and query dimensions must match")
        if len(queries) != len(ground_truth):
            raise ValueError("synthetic queries and ground truth must have equal row counts")
        if ground_truth.size and (
            int(ground_truth.min()) < 0 or int(ground_truth.max()) >= len(base)
        ):
            raise ValueError("synthetic ground-truth ID is outside the base vector range")
        for value in (base, queries, ground_truth):
            value.flags.writeable = False
        object.__setattr__(self, "base_vectors", base)
        object.__setattr__(self, "query_vectors", queries)
        object.__setattr__(self, "ground_truth", ground_truth)

    @property
    def dimension(self) -> int:
        return int(self.base_vectors.shape[1])

    @property
    def base_count(self) -> int:
        return int(self.base_vectors.shape[0])

    @property
    def query_count(self) -> int:
        return int(self.query_vectors.shape[0])

    @classmethod
    def deterministic(cls) -> "SyntheticDataset":
        """Return the eight-point L2 oracle fixture used by later workflows."""
        return cls(
            base_vectors=np.asarray(
                [
                    [0.0, 0.0],
                    [0.5, 0.0],
                    [10.0, 0.0],
                    [10.5, 0.0],
                    [11.0, 0.0],
                    [20.0, 0.0],
                    [30.0, 0.0],
                    [30.5, 0.0],
                ],
                dtype=np.float32,
            ),
            query_vectors=np.asarray(
                [[0.1, 0.0], [10.1, 0.0], [29.9, 0.0]], dtype=np.float32
            ),
            ground_truth=np.asarray([[0, 1], [2, 3], [6, 7]], dtype=np.int64),
        )
