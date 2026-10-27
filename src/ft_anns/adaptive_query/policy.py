"""Pure size-matched planning for RAMPS routing scores.

RAMPS means Rank-Aggregated Mass with Power-Scaled Size. The decoder replays
the checkpoint's power-scaled size contract before ranking probe candidates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ft_anns.partition import ProbePlan


def _payload(artifact: object) -> Mapping[str, Any]:
    if isinstance(artifact, Mapping):
        return artifact
    value = getattr(artifact, "payload", None)
    if isinstance(value, Mapping):
        return value
    raise ValueError("Calibration Artifact payload is invalid")


@dataclass(frozen=True)
class DecoderPolicyTable:
    """Alpha-specific ranked prefixes and their cumulative mass and cost."""

    bucket_ids: npt.NDArray[np.int64]
    cumulative_mass: npt.NDArray[np.float64]
    cumulative_cost: npt.NDArray[np.int64]

    @property
    def candidate_bound(self) -> int:
        return int(self.bucket_ids.shape[1])

    def fixed_width_plan(self, fixed_bucket_count: int) -> ProbePlan:
        if (
            isinstance(fixed_bucket_count, bool)
            or not isinstance(fixed_bucket_count, (int, np.integer))
            or fixed_bucket_count < 1
            or fixed_bucket_count > self.candidate_bound
        ):
            raise ValueError("Decoder Fixed-Width count exceeds the candidate bound")
        return ProbePlan(
            bucket_ids=self.bucket_ids,
            probe_counts=np.full(
                len(self.bucket_ids), int(fixed_bucket_count), dtype=np.int64
            ),
        )

    def select_prefixes(
        self, *, cost_lambda: float, m_min: int, m_max: int | None = None
    ) -> ProbePlan:
        resolved_max = self.candidate_bound if m_max is None else m_max
        if (
            isinstance(cost_lambda, bool)
            or not math.isfinite(cost_lambda)
            or cost_lambda < 0
            or isinstance(m_min, bool)
            or not isinstance(m_min, (int, np.integer))
            or m_min < 1
            or isinstance(resolved_max, bool)
            or not isinstance(resolved_max, (int, np.integer))
            or resolved_max < m_min
            or resolved_max > self.candidate_bound
        ):
            raise ValueError("Decoder prefix policy is invalid")
        first = int(m_min) - 1
        stop = int(resolved_max)
        utility = (
            self.cumulative_mass[:, first:stop]
            - float(cost_lambda) * self.cumulative_cost[:, first:stop]
        )
        probe_counts = np.argmax(utility, axis=1).astype(np.int64) + int(m_min)
        return ProbePlan(
            bucket_ids=self.bucket_ids[:, :stop], probe_counts=probe_counts
        )


def build_decoder_policy_table(
    *,
    router_scores: npt.ArrayLike,
    list_sizes: npt.ArrayLike,
    artifact: object,
    candidate_bound: int,
) -> DecoderPolicyTable:
    """Build the shared Decoder ranking and cumulative prefix quantities."""
    metadata = _payload(artifact)
    if metadata.get("format") != "adaptive-query-calibration-v3":
        raise ValueError("unsupported Calibration Artifact format")
    if metadata.get("decoder") != "ramps_size_matched":
        raise ValueError("Calibration Artifact decoder mismatch")
    transform = metadata.get("target_size_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("Calibration Artifact policy contract is invalid")
    try:
        mode = transform["mode"]
        alpha = float(transform["alpha"])
        anchor = float(transform["anchor"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration Artifact policy contract is invalid") from error
    if mode != "anchored_power":
        raise ValueError("Calibration Artifact anchored-power transform mismatch")
    try:
        return _build_ramps_decoder_policy_table(
            router_scores=router_scores,
            list_sizes=list_sizes,
            alpha=alpha,
            anchor=anchor,
            candidate_bound=candidate_bound,
        )
    except ValueError as error:
        raise ValueError(
            "Calibration Artifact anchored-power transform mismatch"
        ) from error


def _build_ramps_decoder_policy_table(
    *,
    router_scores: npt.ArrayLike,
    list_sizes: npt.ArrayLike,
    alpha: float,
    anchor: float,
    candidate_bound: int,
) -> DecoderPolicyTable:
    scores = np.asarray(router_scores, dtype=np.float64)
    sizes = np.asarray(list_sizes)
    expected_anchor = (
        float(sizes.mean(dtype=np.float64)) if sizes.ndim == 1 and len(sizes) else 0.0
    )
    if (
        scores.ndim != 2
        or not len(scores)
        or sizes.ndim != 1
        or scores.shape[1] != len(sizes)
        or sizes.dtype.kind not in ("i", "u")
        or np.any(sizes <= 0)
        or not np.isfinite(scores).all()
        or isinstance(alpha, bool)
        or not math.isfinite(alpha)
        or alpha <= 0
        or not math.isfinite(anchor)
        or anchor <= 0
        or not math.isclose(anchor, expected_anchor, rel_tol=0.0, abs_tol=1e-12)
        or isinstance(candidate_bound, bool)
        or not isinstance(candidate_bound, (int, np.integer))
        or candidate_bound < 1
        or candidate_bound > len(sizes)
    ):
        raise ValueError("RAMPS Decoder inputs are incompatible")

    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    effective_sizes = anchor * np.power(sizes.astype(np.float64) / anchor, alpha)
    decoded_mass = probabilities * effective_sizes[None, :]
    decoded_mass /= decoded_mass.sum(axis=1, keepdims=True)
    ranking_score = decoded_mass / sizes[None, :]

    bound = int(candidate_bound)
    bucket_ids = np.arange(len(sizes), dtype=np.int64)
    rankings = np.empty((len(scores), bound), dtype=np.int64)
    use_partial = bound * 2 <= len(sizes)
    for row_id, row in enumerate(ranking_score):
        if use_partial:
            selected = np.argpartition(-row, bound - 1)[:bound]
            cutoff = row[selected].min()
            higher = np.flatnonzero(row > cutoff)
            tied = np.flatnonzero(row == cutoff)
            selected = np.concatenate((higher, tied[: bound - len(higher)]))
            rankings[row_id] = selected[np.lexsort((selected, -row[selected]))]
        else:
            rankings[row_id] = np.lexsort((bucket_ids, -row))[:bound]
    row_ids = np.arange(len(rankings))[:, None]
    return DecoderPolicyTable(
        bucket_ids=rankings,
        cumulative_mass=np.cumsum(
            decoded_mass[row_ids, rankings], axis=1, dtype=np.float64
        ),
        cumulative_cost=np.cumsum(sizes[rankings], axis=1, dtype=np.int64),
    )


def build_decoder_fixed_width_plan(
    *,
    router_scores: npt.ArrayLike,
    list_sizes: npt.ArrayLike,
    artifact: object,
    fixed_bucket_count: int,
    candidate_bound: int,
) -> ProbePlan:
    """Build a fixed-width control under the Adaptive Decoder ranking."""
    table = build_decoder_policy_table(
        router_scores=router_scores,
        list_sizes=list_sizes,
        artifact=artifact,
        candidate_bound=candidate_bound,
    )
    return table.fixed_width_plan(fixed_bucket_count)


def build_ramps_decoder_fixed_width_plan(
    *,
    router_scores: npt.ArrayLike,
    list_sizes: npt.ArrayLike,
    alpha: float,
    fixed_bucket_count: int,
    candidate_bound: int | None = None,
) -> ProbePlan:
    """Build RAMPS Decoder Fixed-Width routes without a persisted policy artifact."""
    sizes = np.asarray(list_sizes)
    if sizes.ndim != 1 or not len(sizes):
        raise ValueError("RAMPS Decoder requires one non-empty list-size vector")
    bound = len(sizes) if candidate_bound is None else candidate_bound
    table = _build_ramps_decoder_policy_table(
        router_scores=router_scores,
        list_sizes=sizes,
        alpha=alpha,
        anchor=float(sizes.mean(dtype=np.float64)),
        candidate_bound=bound,
    )
    return table.fixed_width_plan(fixed_bucket_count)


def build_probe_plan(
    *,
    router_scores: npt.ArrayLike,
    list_sizes: npt.ArrayLike,
    artifact: object,
) -> ProbePlan:
    """Build bounded strict-prefix routes with the frozen RAMPS policy."""
    metadata = _payload(artifact)
    policy = metadata.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("Calibration Artifact policy contract is invalid")

    try:
        cost_lambda = float(policy["cost_lambda"])
        m_min = int(policy["m_min"])
        m_max = int(policy["m_max"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration Artifact policy contract is invalid") from error
    if (
        isinstance(policy.get("cost_lambda"), bool)
        or not math.isfinite(cost_lambda)
        or cost_lambda < 0
        or isinstance(policy.get("m_min"), bool)
        or isinstance(policy.get("m_max"), bool)
        or m_min < 1
        or m_max < m_min
    ):
        raise ValueError("Calibration Artifact probe bounds are invalid")
    table = build_decoder_policy_table(
        router_scores=router_scores,
        list_sizes=list_sizes,
        artifact=artifact,
        candidate_bound=m_max,
    )
    return table.select_prefixes(cost_lambda=cost_lambda, m_min=m_min)
