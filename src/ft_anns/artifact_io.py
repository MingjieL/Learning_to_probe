"""Shared persistence and local artifact-filename reference helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt


def readonly_array(value: npt.ArrayLike, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
    """Return an owned, C-contiguous immutable array with the requested dtype."""
    copied = np.array(value, dtype=dtype, copy=True, order="C")
    result = np.frombuffer(memoryview(copied).toreadonly(), dtype=copied.dtype).reshape(
        copied.shape
    )
    result.flags.writeable = False
    return result


def write_json_atomic(path: Path, payload: object) -> None:
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary_path.replace(path)


def artifact_filename(path: str | Path) -> str:
    """Return the basename used as a local artifact reference."""
    name = Path(path).name
    if not name or name in (".", ".."):
        raise ValueError("artifact path must have a filename")
    return name


def is_artifact_filename(value: object) -> bool:
    """Whether *value* is a non-empty basename rather than a path or digest."""
    return (
        isinstance(value, str)
        and bool(value)
        and value not in (".", "..")
        and ":" not in value
        and Path(value).name == value
    )
