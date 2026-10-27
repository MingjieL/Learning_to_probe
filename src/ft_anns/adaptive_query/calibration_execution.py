"""Shared streaming Router and Decoder execution for calibration."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import time

import numpy as np
import numpy.typing as npt
import torch

from ft_anns.adaptive_query.policy import (
    DecoderPolicyTable,
    build_decoder_policy_table,
)
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.partition import FixedPartition


@dataclass(frozen=True)
class DecoderPolicyBatch:
    """One bounded batch plus cumulative Router/Decoder execution evidence."""

    offset: int
    completed_batch_count: int
    queries: npt.NDArray[np.float32]
    policy_table: DecoderPolicyTable
    router_seconds: float
    policy_table_seconds: float
    router_score_bytes: int

    @property
    def stop(self) -> int:
        return self.offset + len(self.queries)


def stream_decoder_policy_batches(
    *,
    checkpoint: RouterCheckpoint,
    partition: FixedPartition,
    queries: npt.NDArray[np.float32],
    candidate_bound: int,
    batch_size: int,
    decoder: str = "ramps_size_matched",
) -> Iterator[DecoderPolicyBatch]:
    """Run bounded Router batches and build their Decoder policy tables."""
    if (
        queries.ndim != 2
        or not len(queries)
        or queries.shape[1] != partition.dimension
        or isinstance(candidate_bound, bool)
        or not isinstance(candidate_bound, int)
        or candidate_bound < 1
        or candidate_bound > partition.bucket_count
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("calibration Decoder batch inputs are invalid")

    transform = checkpoint.metadata.get("loss", {}).get(
        "target_size_transform", {}
    )
    decoder_artifact = {
        "format": "adaptive-query-calibration-v3",
        "decoder": decoder,
        "target_size_transform": {
            name: transform.get(name) for name in ("mode", "alpha", "anchor")
        },
    }
    model = checkpoint.model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
    model.eval()
    router_seconds = 0.0
    policy_table_seconds = 0.0

    for batch_index, offset in enumerate(range(0, len(queries), batch_size)):
        stop = min(offset + batch_size, len(queries))
        batch_queries = queries[offset:stop]
        router_started = time.perf_counter()
        with torch.no_grad():
            scores = model(
                torch.from_numpy(np.array(batch_queries, copy=True)).to(device)
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        router_seconds += time.perf_counter() - router_started
        router_scores = scores.detach().cpu().numpy().astype(np.float64, copy=False)
        if router_scores.shape != (len(batch_queries), partition.bucket_count):
            raise ValueError("Router output shape is invalid")

        table_started = time.perf_counter()
        policy_table = build_decoder_policy_table(
            router_scores=router_scores,
            list_sizes=partition.list_sizes,
            artifact=decoder_artifact,
            candidate_bound=candidate_bound,
        )
        policy_table_seconds += time.perf_counter() - table_started
        yield DecoderPolicyBatch(
            offset=offset,
            completed_batch_count=batch_index + 1,
            queries=batch_queries,
            policy_table=policy_table,
            router_seconds=router_seconds,
            policy_table_seconds=policy_table_seconds,
            router_score_bytes=router_scores.nbytes,
        )
