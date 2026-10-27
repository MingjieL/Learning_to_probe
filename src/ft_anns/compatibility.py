"""Canonical human-readable compatibility profiles and filenames."""

from __future__ import annotations

import math
from pathlib import Path
import re
from typing import Mapping

from ft_anns.domain import require_dataset_name


_PROFILE_FIELDS = (
    "dataset_name",
    "ground_truth_k",
    "bucket_count",
    "kmeans_niter",
    "partition_build_seed",
    "loss",
    "ramps_alpha",
    "top_k",
)
_LOSSES = {"ce", "bce", "ramps"}


def _positive_int(profile: Mapping[str, object], name: str) -> int | None:
    value = profile.get(name)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"Compatibility Profile {name} must be a positive integer")
    return value


def _slug_number(value: float) -> str:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            "Compatibility Profile ramps_alpha must be positive and finite"
        )
    text = format(value, ".15g")
    return text.replace("-", "m").replace(".", "p").replace("+", "")


def normalize_compatibility_profile(
    profile: Mapping[str, object],
) -> dict[str, object]:
    """Validate and return a canonical, JSON-ready Compatibility Profile."""
    unknown = set(profile) - set(_PROFILE_FIELDS)
    if unknown:
        raise ValueError(
            "Compatibility Profile has unknown field(s): " + ", ".join(sorted(unknown))
        )
    try:
        dataset_name = require_dataset_name(
            profile.get("dataset_name"), allow_synthetic=True
        )
    except ValueError as error:
        raise ValueError("Compatibility Profile dataset_name is invalid") from error
    normalized: dict[str, object] = {"dataset_name": dataset_name}
    for name in (
        "ground_truth_k",
        "bucket_count",
        "kmeans_niter",
        "top_k",
    ):
        value = _positive_int(profile, name)
        if value is not None:
            normalized[name] = value
    partition_build_seed = profile.get("partition_build_seed")
    if partition_build_seed is not None:
        if not isinstance(partition_build_seed, int) or isinstance(
            partition_build_seed, bool
        ):
            raise ValueError(
                "Compatibility Profile partition_build_seed must be an integer"
            )
        normalized["partition_build_seed"] = partition_build_seed
    loss = profile.get("loss")
    if loss is not None:
        if loss not in _LOSSES:
            raise ValueError("Compatibility Profile loss is invalid")
        normalized["loss"] = loss
    ramps_alpha = profile.get("ramps_alpha")
    if ramps_alpha is not None:
        if (
            isinstance(ramps_alpha, bool)
            or not isinstance(ramps_alpha, (int, float))
            or not math.isfinite(float(ramps_alpha))
            or float(ramps_alpha) <= 0
        ):
            raise ValueError(
                "Compatibility Profile ramps_alpha must be positive and finite"
            )
        if loss != "ramps":
            raise ValueError("Compatibility Profile ramps_alpha requires loss='ramps'")
        normalized["ramps_alpha"] = float(ramps_alpha)
    return normalized


def compatibility_slug(profile: Mapping[str, object]) -> str:
    """Render the canonical readable slug for a Compatibility Profile."""
    normalized = normalize_compatibility_profile(profile)
    dataset = str(normalized["dataset_name"]).lower()
    tokens = [dataset]
    token_fields = (
        ("ground_truth_k", "gtk"),
        ("bucket_count", "bucket"),
        ("kmeans_niter", "niter"),
        ("partition_build_seed", "pseed"),
    )
    for field, prefix in token_fields:
        if field in normalized:
            tokens.append(f"{prefix}{normalized[field]}")
    if "loss" in normalized:
        loss = str(normalized["loss"])
        if loss == "ramps" and "ramps_alpha" in normalized:
            loss = f"ramps-alpha{_slug_number(float(normalized['ramps_alpha']))}"
        tokens.append(loss)
    if "top_k" in normalized:
        tokens.append(f"topk{normalized['top_k']}")
    return "_".join(tokens)


def compatibility_filename(
    profile: Mapping[str, object],
    *,
    kind: str,
    extension: str = "",
) -> str:
    """Return a canonical basename from a profile and artifact/config kind."""
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", kind):
        raise ValueError("compatibility filename kind is invalid")
    if extension and not re.fullmatch(r"\.[a-z0-9]+", extension):
        raise ValueError("compatibility filename extension is invalid")
    return f"{compatibility_slug(profile)}_{kind}{extension}"


def require_compatibility_metadata(
    metadata: Mapping[str, object],
    *,
    expected: Mapping[str, object] | None = None,
    filename: str | Path | None = None,
    kind: str | None = None,
) -> dict[str, object]:
    """Validate a stored profile, its redundant slug, and optional basename."""
    raw_profile = metadata.get("compatibility")
    if not isinstance(raw_profile, Mapping):
        raise ValueError("artifact Compatibility Profile is missing or invalid")
    profile = normalize_compatibility_profile(raw_profile)
    if expected is not None and profile != normalize_compatibility_profile(expected):
        raise ValueError("artifact Compatibility Profile mismatch")
    slug = compatibility_slug(profile)
    if metadata.get("compatibility_slug") != slug:
        raise ValueError("artifact compatibility slug mismatch")
    if filename is not None:
        if kind is None:
            raise ValueError("artifact kind is required for filename validation")
        actual = Path(filename).name
        suffix = Path(actual).suffix
        expected_filename = compatibility_filename(profile, kind=kind, extension=suffix)
        if actual != expected_filename:
            raise ValueError(
                f"artifact compatibility filename mismatch: expected "
                f"{expected_filename!r}, found {actual!r}"
            )
    return profile
