"""Rebuild a compact partition from immutable assignments and an external base.

Only serialization changes: bucket IDs remain unchanged, and
vectors are inserted in original ID order to preserve tie ordering in FAISS.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import faiss  # type: ignore[import-untyped]
import numpy as np

from ft_anns.datasets.base_gt import FileVectorSource

if TYPE_CHECKING:
    from ft_anns.partition.fixed_ivf import FixedPartition

LEGACY_FORMAT = "fixed-partition-v5"
FORMAT = "fixed-partition-v6"
LEGACY_ARRAY_FILES = {
    "bucket_assignment": "bucket_assignment.npy",
    "centroids": "centroids.npy",
}
ARRAY_FILES = {"bucket_assignment": "bucket_assignment.npy"}


def array_files(format_name: str) -> dict[str, str]:
    return LEGACY_ARRAY_FILES if format_name == LEGACY_FORMAT else ARRAY_FILES


# Bound conversion and insertion buffers; the source itself stays memory mapped.
REBUILD_BATCH_SIZE = 100_000


def prepare_base_reference(
    partition: FixedPartition, artifact_path: Path, base_path: Path
) -> dict[str, Any]:
    """Verify export input against stored vectors before publishing an artifact."""
    source = FileVectorSource(
        base_path, base_path.suffix.lstrip("."), partition.dimension
    )
    if source.count != len(partition.bucket_assignment):
        raise ValueError("compact partition base count mismatch")
    invlists = partition._index.invlists
    for bucket in range(partition.bucket_count):
        count = invlists.list_size(bucket)
        ids = faiss.rev_swig_ptr(invlists.get_ids(bucket), count)
        # Assignment alone cannot encode a different order within a bucket.
        if np.any(ids[1:] <= ids[:-1]):
            raise ValueError("compact export requires ascending IDs within each bucket")
        expected_ids = partition.posting_ids[
            partition.posting_offsets[bucket] : partition.posting_offsets[bucket + 1]
        ]
        if not np.array_equal(ids, expected_ids):
            raise ValueError("compact export posting order mismatch")
        codes = (
            faiss.rev_swig_ptr(
                invlists.get_codes(bucket), count * partition.dimension * 4
            )
            .view(np.float32)
            .reshape(count, partition.dimension)
        )
        for start in range(0, count, REBUILD_BATCH_SIZE):
            stop = min(count, start + REBUILD_BATCH_SIZE)
            vectors = np.asarray(source.read_rows(ids[start:stop]), dtype=np.float32)
            if not np.isfinite(vectors).all() or not np.array_equal(
                vectors, codes[start:stop]
            ):
                raise ValueError(
                    "compact partition base vectors differ from the existing index"
                )
    return {
        "path": os.path.relpath(base_path.absolute(), artifact_path.absolute()),
        "filename": base_path.name,
        "format": base_path.suffix.lstrip("."),
        "file_bytes": base_path.stat().st_size,
    }


def load_components(
    artifact_path: Path, metadata: dict[str, Any], base_path: str | Path | None
) -> tuple[dict[str, np.ndarray], Any]:
    from ft_anns.partition.fixed_ivf import _posting_sizes_and_offsets

    count, buckets, dimension = (
        metadata[key] for key in ("base_count", "bucket_count", "dimension")
    )
    reference = metadata.get("base_source")
    if not isinstance(reference, dict):
        raise TypeError("compact partition is missing its base source")
    if (
        not isinstance(reference.get("path"), str)
        or not reference["path"]
        or reference.get("filename") != metadata.get("base_filename")
        or not isinstance(reference.get("file_bytes"), int)
        or isinstance(reference["file_bytes"], bool)
        or reference["file_bytes"] <= 0
        or reference.get("format") not in ("fbin", "i8bin", "ivecs", "fvecs", "bvecs")
    ):
        raise ValueError("compact partition has an invalid base source")
    resolved = (
        Path(base_path) if base_path is not None else artifact_path / reference["path"]
    )
    if resolved.name != reference["filename"]:
        raise ValueError("compact partition base filename mismatch")
    if not resolved.is_file():
        raise FileNotFoundError(f"compact partition base file not found: {resolved}")
    if resolved.stat().st_size != reference["file_bytes"]:
        raise ValueError("compact partition base file size mismatch")
    source = FileVectorSource(resolved, reference["format"], dimension)
    if source.count != count:
        raise ValueError("compact partition base count mismatch")
    arrays = {
        name: np.load(artifact_path / file, mmap_mode="r", allow_pickle=False)
        for name, file in array_files(metadata["format"]).items()
    }
    assignment, centroids = arrays["bucket_assignment"], arrays.get("centroids")
    if (
        assignment.dtype != np.dtype(np.int64)
        or assignment.shape != (count,)
        or (
            centroids is not None
            and (
                centroids.dtype != np.dtype(np.float32)
                or centroids.shape != (buckets, dimension)
            )
        )
    ):
        raise ValueError("Fixed Partition artifact array shape or dtype mismatch")
    if np.any(assignment < 0) or np.any(assignment >= buckets):
        raise ValueError("Fixed Partition artifact posting lists are inconsistent")
    if centroids is not None and not np.isfinite(centroids).all():
        raise ValueError("compact partition has non-finite centroids")
    sizes, offsets = _posting_sizes_and_offsets(assignment, buckets)
    if np.any(sizes <= 0):
        raise ValueError("Fixed Partition artifact contains an empty bucket")
    metric = (
        faiss.METRIC_L2 if metadata["metric"] == "l2" else faiss.METRIC_INNER_PRODUCT
    )
    # Routes come from assignment during insertion and the router during search.
    # v6 never calls quantizer.search or trains/reconstructs surrogate centroids.
    quantizer = faiss.IndexFlat(dimension, metric)
    if centroids is not None:
        quantizer.add(np.ascontiguousarray(centroids))
    index = faiss.IndexIVFFlat(quantizer, dimension, buckets, metric)
    index.is_trained = True
    for start in range(0, count, REBUILD_BATCH_SIZE):
        stop = min(count, start + REBUILD_BATCH_SIZE)
        vectors = np.ascontiguousarray(source.read_slice(start, stop), dtype=np.float32)
        if not np.isfinite(vectors).all():
            raise ValueError("compact partition base contains non-finite vectors")
        ids = np.arange(start, stop, dtype=np.int64)
        routes = np.ascontiguousarray(assignment[start:stop])
        index.add_core(
            len(ids),
            faiss.swig_ptr(vectors),
            faiss.swig_ptr(ids),
            faiss.swig_ptr(routes),
        )
    postings = np.empty(count, dtype=np.int64)
    for bucket in range(buckets):
        postings[offsets[bucket] : offsets[bucket + 1]] = faiss.rev_swig_ptr(
            index.invlists.get_ids(bucket), index.invlists.list_size(bucket)
        )
    arrays.update(list_sizes=sizes, posting_offsets=offsets, posting_ids=postings)
    return arrays, index
