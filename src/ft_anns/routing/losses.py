"""Routing objectives over unified query-level bucket labels."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from ft_anns.config import RampsConfig


# RAMPS: Rank-Aggregated Mass Per Size.
_RAMPS_LOSS = "ramps"


def build_ramps_soft_targets(
    *,
    bucket_ids: Tensor,
    list_sizes: Tensor,
    rank_weight: str,
    dtype: torch.dtype,
) -> Tensor:
    """Build one normalized RAMPS soft target per query.

    RAMPS expands to Rank-Aggregated Mass Per Size: ground-truth
    neighbors contribute discounted mass by rank, while posting-list sizes are
    normalized by the largest list in the Fixed Partition.
    """
    if bucket_ids.ndim != 2 or bucket_ids.shape[1] == 0:
        raise ValueError("bucket_ids must have shape [batch, K] with K > 0")
    if bucket_ids.dtype != torch.int64:
        raise ValueError("bucket_ids must use torch.int64")
    if not dtype.is_floating_point:
        raise ValueError("RAMPS targets require a floating-point dtype")
    if rank_weight != "log":
        raise ValueError("Ticket 07 supports only RAMPS rank_weight='log'")
    class_count = int(list_sizes.shape[0]) if list_sizes.ndim == 1 else 0
    if bucket_ids.numel() and (
        int(bucket_ids.min()) < 0 or int(bucket_ids.max()) >= class_count
    ):
        raise ValueError("bucket_ids contains a label outside the list_sizes range")

    sizes = list_sizes.to(device=bucket_ids.device, dtype=dtype)
    if not torch.isfinite(sizes).all() or torch.any(sizes <= 0):
        raise ValueError("RAMPS list_sizes must be finite and strictly positive")
    normalized_sizes = sizes / sizes.max()
    ranks = torch.arange(
        1,
        bucket_ids.shape[1] + 1,
        dtype=dtype,
        device=bucket_ids.device,
    )
    rank_weights = 1.0 / torch.log2(ranks + 1.0)
    # Aggregate discounted ground-truth mass when several neighbors share a bucket.
    mass = torch.zeros(
        (bucket_ids.shape[0], class_count),
        dtype=dtype,
        device=bucket_ids.device,
    )
    mass.scatter_add_(
        1,
        bucket_ids,
        rank_weights.unsqueeze(0).expand(bucket_ids.shape[0], -1),
    )
    # Rank-weighted mass density under max-normalized posting-list size.
    quality = mass / normalized_sizes.unsqueeze(0)
    normalization = quality.sum(dim=1, keepdim=True)
    if not torch.isfinite(normalization).all() or torch.any(normalization <= 0):
        raise ValueError("RAMPS soft targets require positive finite mass per query")
    targets = quality / normalization
    if not torch.isfinite(targets).all():
        raise ValueError("RAMPS soft targets must be finite")
    return targets


def compute_routing_loss(
    *,
    loss_name: str,
    logits: Tensor,
    bucket_ids: Tensor,
    list_sizes: Tensor | None = None,
    ramps: RampsConfig | Mapping[str, object] | None = None,
) -> Tensor:
    """Compute one routing loss without changing the query-level sample shape."""
    if loss_name not in ("ce", "bce", _RAMPS_LOSS):
        raise ValueError("loss_name must be one of: ce, bce, ramps")
    if logits.ndim != 2 or bucket_ids.ndim != 2:
        raise ValueError("logits and bucket_ids must both be matrices")
    if logits.shape[0] != bucket_ids.shape[0] or bucket_ids.shape[1] == 0:
        raise ValueError("each query requires one or more bucket labels")
    if bucket_ids.dtype != torch.int64:
        raise ValueError("bucket_ids must use torch.int64")
    if bucket_ids.numel() and (
        int(bucket_ids.min()) < 0 or int(bucket_ids.max()) >= logits.shape[1]
    ):
        raise ValueError("bucket_ids contains a label outside the logits range")

    if loss_name == "ce":
        log_probabilities = F.log_softmax(logits, dim=-1)
        return -log_probabilities.gather(1, bucket_ids).mean()

    if loss_name == "bce":
        targets = torch.zeros_like(logits)
        targets.scatter_(1, bucket_ids, 1.0)
        return F.binary_cross_entropy_with_logits(logits, targets)

    if list_sizes is None or ramps is None:
        raise ValueError("RAMPS loss requires list_sizes and ramps settings")
    if list_sizes.ndim != 1 or list_sizes.shape[0] != logits.shape[1]:
        raise ValueError("RAMPS list_sizes must match the logits class dimension")
    settings = (
        ramps if isinstance(ramps, RampsConfig) else RampsConfig.model_validate(ramps)
    )
    targets = build_ramps_soft_targets(
        bucket_ids=bucket_ids,
        list_sizes=list_sizes,
        rank_weight=settings.rank_weight,
        dtype=logits.dtype,
    )
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()
