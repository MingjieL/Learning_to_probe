"""Compact CPU L2PR+ queries, with reporting outside the query call."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import time

import faiss
import numpy as np
import torch

from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner,
    POSITIVE_MARGINAL_RETRIEVAL_SETTINGS,
    _gain_cost_native,
)
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.partition import FixedPartition


L2PR_PLUS_TIMING_SCOPE = "positive-marginal-query-v1"


class L2PRPlusWorkspace:
    """Internal buffers shared by serial calls and targets, never by outputs."""

    def __init__(self, *, model: torch.nn.Module, bucket_count: int,
                 batch_size: int, reuse_buffers: bool = True,
                 compact_filter: bool = True, scratch_bits: int = 32,
                 vectorized_filter: bool = True, collect_routes: bool = True,
                 reuse_hidden: bool = False) -> None:
        from ft_anns.routing.model import RawQueryRouter

        if batch_size < 1 or bucket_count < 1:
            raise ValueError("workspace dimensions must be positive")
        if scratch_bits not in (32, 64):
            raise ValueError("positive-marginal scratch must use 32 or 64 bits")
        self.model, self.bucket_count, self.batch_size = model, bucket_count, batch_size
        self.reuse_buffers = reuse_buffers
        self.thread_count = torch.get_num_threads()
        self.logits = (torch.empty((batch_size, bucket_count), dtype=torch.float32)
                       if reuse_buffers and type(model) is RawQueryRouter else None)
        self.hidden = (tuple(torch.empty((batch_size, layer.out_features), dtype=torch.float32)
                             for layer in model.network[:-1] if isinstance(layer, torch.nn.Linear))
                       if self.logits is not None and reuse_hidden else None)
        supported = all(hasattr(_gain_cost_native, name) for name in (
            "PositiveMarginalWorkspace", "positive_marginal_plan_cpu"))
        supports_scan = hasattr(getattr(_gain_cost_native, "PositiveMarginalWorkspace", None), "scan_backend")
        native_args = (bucket_count, batch_size, scratch_bits)
        self.native = (_gain_cost_native.PositiveMarginalWorkspace(
            *native_args, *([vectorized_filter] if supports_scan else []))
            if reuse_buffers and compact_filter and supported else None)
        self.scratch_bits = scratch_bits if self.native is not None else 0
        self.collect_routes = collect_routes and type(model) in (RawQueryRouter, torch.nn.Linear) and self.native is not None and all(
            hasattr(_gain_cost_native, name) for name in (
                "positive_marginal_begin", "positive_marginal_append", "positive_marginal_finish")) and hasattr(
                    self.native, "cancel_collection")
        self._coarse_storage: np.ndarray | None = None
        self._partition: FixedPartition | None = None
        self.list_sizes: torch.Tensor | None = None
        self.normalized_sizes: torch.Tensor | None = None
        self.initial_nbytes = self.nbytes

    def bind_partition(self, partition: FixedPartition, planner: GainCostProbePlanner) -> None:
        """Validate once during setup, then share frozen partition cost tables."""
        if self._partition is not None and self._partition is not partition:
            raise ValueError("L2PR+ workspace is already bound to another partition")
        if self._partition is None:
            sizes = np.array(partition.list_sizes, copy=True)
            normalized = torch.from_numpy(sizes.astype(np.float32) / float(sizes.max()))
            if not torch.equal(planner._normalized_sizes, normalized):
                raise ValueError("L2PR+ planner costs do not match the partition")
            self.list_sizes = torch.from_numpy(sizes)
            self.normalized_sizes = normalized
            self.initial_nbytes += sizes.nbytes + self.normalized_sizes.numel() * self.normalized_sizes.element_size()
        elif not torch.equal(planner._normalized_sizes, self.normalized_sizes):
            raise ValueError("L2PR+ planner costs do not match the partition")
        self._partition = partition
        # The planner's only writes to this table occur before binding. Thresholds
        # are already cached; subsequent plans only read the shared costs.
        planner._normalized_sizes = self.normalized_sizes

    @property
    def nbytes(self) -> int:
        return ((self.logits.numel() * self.logits.element_size() if self.logits is not None else 0)
                + sum(tensor.numel() * tensor.element_size() for tensor in self.hidden or ())
                + sum(tensor.numel() * tensor.element_size() for tensor in
                      (self.list_sizes, self.normalized_sizes) if tensor is not None)
                + (self.native.allocated_bytes if self.native is not None else 0)
                + (self._coarse_storage.nbytes if self._coarse_storage is not None else 0))

    def validate_threads(self) -> None:
        if self.native is not None and torch.get_num_threads() != self.thread_count:
            raise RuntimeError("Torch thread count changed after L2PR+ workspace preparation")

    def coarse_buffer(self, rows: int, width: int) -> np.ndarray:
        if not self.reuse_buffers:
            return np.zeros((rows, width), dtype=np.float32)
        elements = rows * width
        if self._coarse_storage is None or self._coarse_storage.size < elements:
            self._coarse_storage = np.zeros(elements, dtype=np.float32)
            self._coarse_storage.flags.writeable = False
        return self._coarse_storage[:elements].reshape(rows, width)


class L2PRPlusQuery:
    """Prepare input views, static policy and buffers before timing retrieval.

    Calls sharing this index/workspace are serial. No query-dependent model or
    filter result is prepared here. Per-query NDis is optional postprocessing.
    """

    def __init__(self, *, queries: np.ndarray, checkpoint: RouterCheckpoint,
                 partition: FixedPartition, planner: GainCostProbePlanner,
                 topk: int, router_batch_size: int,
                 workspace: L2PRPlusWorkspace | None = None,
                 profile_stages: bool = False,
                 faiss_batch_size: int | None = None) -> None:
        queries = np.asarray(queries, dtype=np.float32, order="C")
        if queries.ndim != 2 or queries.shape[1] != partition.dimension or not len(queries):
            raise ValueError("L2PR+ queries must be a nonempty matrix matching the partition")
        if topk < 1 or router_batch_size < 1:
            raise ValueError("topk and router_batch_size must be positive")
        if faiss_batch_size is not None and faiss_batch_size < 1:
            raise ValueError("faiss_batch_size must be positive or None")
        if planner.settings != POSITIVE_MARGINAL_RETRIEVAL_SETTINGS:
            raise ValueError("L2PR+ requires Probability positive_marginal_filter/bucket_id")
        if any(p.device.type != "cpu" or p.dtype != torch.float32
               for p in checkpoint.model.parameters()):
            raise ValueError("L2PR+ requires a CPU float32 Router")
        if (planner._normalized_sizes.device.type != "cpu"
                or planner._normalized_sizes.dtype != torch.float32
                or len(planner._normalized_sizes) != partition.bucket_count):
            raise ValueError("L2PR+ planner must match the CPU float32 partition")
        checkpoint.validate_for(partition)
        self.queries, self.query_tensor = queries, torch.from_numpy(queries)
        self.model, self.index, self.planner = checkpoint.model.eval(), partition._index, planner
        self.topk, self.router_batch_size = topk, router_batch_size
        self.faiss_batch_size = faiss_batch_size
        self.profile_stages = profile_stages
        batch_size = min(len(queries), router_batch_size)
        self.workspace = workspace or L2PRPlusWorkspace(
            model=self.model, bucket_count=partition.bucket_count, batch_size=batch_size)
        if (self.workspace.model is not self.model
                or self.workspace.bucket_count != partition.bucket_count
                or self.workspace.batch_size < batch_size):
            raise ValueError("L2PR+ workspace must match Router, partition and batch capacity")
        self.workspace.validate_threads()
        self.workspace.bind_partition(partition, planner)
        self.list_sizes = self.workspace.list_sizes
        self.native_packing = all(hasattr(_gain_cost_native, name) for name in (
            "pack_padded_plan_batches", "padded_plan_ndis"))
        # Some FAISS builds do not expose the explicit stats argument. Do not
        # fall back to process-global counters, which can contain other calls.
        try:
            supports_stats = "stats" in inspect.signature(self.index.search_preassigned).parameters
        except (TypeError, ValueError):
            supports_stats = False
        self.faiss_stats = faiss.IndexIVFStats() if supports_stats else None
        self.search_kwargs = {"stats": self.faiss_stats} if supports_stats else {}

    @property
    def coordinates(self) -> dict[str, str | int]:
        return {
            "query_execution_implementation": "compact-positive-marginal-v4",
            "query_timing_scope": L2PR_PLUS_TIMING_SCOPE,
            "query_filter_backend": "compact_native" if self.workspace.native is not None else "planner",
            "query_filter_scratch_bits": self.workspace.scratch_bits,
            "query_filter_scan": getattr(self.workspace.native, "scan_backend", "baseline"),
            "query_faiss_batch_size": self.faiss_batch_size or "full",
            "query_plan_packing": "native" if self.native_packing else "numpy",
            "query_plan_assembly": "compact_collector" if self.workspace.collect_routes else "single_batch_or_merge",
            "query_stage_timing": "enabled" if self.profile_stages else "disabled",
            "query_thresholds": "precomputed_log_cost",
            "query_workspace_bytes": self.workspace.initial_nbytes,
            "query_workspace_threads": self.workspace.thread_count,
            "query_logits_buffer": "reused" if self.workspace.logits is not None else "per_batch",
            "query_hidden_buffers": "reused_inplace_activation" if self.workspace.hidden is not None else "per_batch",
            "query_coarse_buffer": "readonly_zero_capacity" if self.workspace.reuse_buffers else "per_call",
            "ndis_source": "faiss_aggregate" if self.faiss_stats is not None else "outside_query_timer",
            "ndis_statistics_scope": "per_query_outside_query_timer",
        }


@dataclass(frozen=True)
class L2PRPlusOutput:
    neighbor_ids: np.ndarray
    distances: np.ndarray
    routes: np.ndarray
    probe_counts: np.ndarray
    faiss_ndis: int | None
    router_seconds: float = 0.0
    kernel_seconds: float = 0.0
    assembly_seconds: float = 0.0
    packing_seconds: float = 0.0
    search_seconds: float = 0.0
    faiss_call_count: int = 1


def query_l2pr_plus(query: L2PRPlusQuery) -> L2PRPlusOutput:
    """Router -> all-bucket positive filter -> padded FAISS, with no reporting.

    Time this whole call externally, including allocation/growth and temporary
    cleanup. Static log thresholds are reused; each query still computes logZ.
    """
    workspace = query.workspace
    workspace.validate_threads()
    profile = query.profile_stages
    route_batches, count_batches = [], []
    router_seconds = kernel_seconds = assembly_seconds = packing_seconds = search_seconds = 0.0
    collect = workspace.collect_routes and len(query.queries) > query.router_batch_size
    if collect:
        started = time.perf_counter() if profile else 0.0
        _gain_cost_native.positive_marginal_begin(workspace.native, len(query.queries))
        if profile:
            assembly_seconds += time.perf_counter() - started
    try:
        with torch.inference_mode():
            for offset in range(0, len(query.queries), query.router_batch_size):
                started = time.perf_counter() if profile else 0.0
                batch = query.query_tensor[offset:offset + query.router_batch_size]
                if workspace.hidden is not None:
                    logits = query.model(batch, out=workspace.logits[:len(batch)], hidden_out=workspace.hidden)
                elif workspace.logits is not None:
                    logits = query.model(batch, out=workspace.logits[:len(batch)])
                else:
                    logits = query.model(batch)
                if profile:
                    router_seconds += time.perf_counter() - started
                    started = time.perf_counter()
                if collect:
                    planner = query.planner
                    _gain_cost_native.positive_marginal_append(
                        logits, planner._positive_log_thresholds, planner._gain_power,
                        planner._alpha, planner._beta, workspace.native)
                else:
                    selected, counts = query.planner.build_tensors(logits, native_workspace=workspace.native)
                    route_batches.append(selected.contiguous())
                    count_batches.append(counts)
                    del selected, counts
                del logits
                if profile:
                    kernel_seconds += time.perf_counter() - started

            started = time.perf_counter() if profile else 0.0
            if collect:
                routes_tensor, counts_tensor = _gain_cost_native.positive_marginal_finish(workspace.native)
                routes, probe_counts = routes_tensor.numpy(), counts_tensor.numpy()
            elif len(route_batches) == 1:
                # Both native and Torch positive filters already pad inactive slots.
                routes, probe_counts = route_batches[0].numpy(), count_batches[0].numpy()
            elif query.native_packing:
                routes_tensor, counts_tensor = _gain_cost_native.pack_padded_plan_batches(
                    route_batches, count_batches, query.list_sizes)
                routes, probe_counts = routes_tensor.numpy(), counts_tensor.numpy()
            else:
                width = max(batch.shape[1] for batch in route_batches)
                routes = np.full((len(query.queries), width), -1, dtype=np.int64)
                probe_counts = np.concatenate([batch.numpy() for batch in count_batches])
                offset = 0
                for selected in route_batches:
                    routes[offset:offset + len(selected), :selected.shape[1]] = selected.numpy()
                    offset += len(selected)
                del selected
            route_batches.clear()
            count_batches.clear()
            if profile:
                assembly_seconds += time.perf_counter() - started
    except BaseException:
        if collect:
            workspace.native.cancel_collection()
        raise

    started = time.perf_counter() if profile else 0.0
    stats = query.faiss_stats
    if stats is not None:
        stats.reset()
    if query.faiss_batch_size is None or query.faiss_batch_size >= len(query.queries):
        coarse = workspace.coarse_buffer(len(query.queries), routes.shape[1])
        query.index.nprobe = routes.shape[1]
        if profile:
            packing_seconds = time.perf_counter() - started
            started = time.perf_counter()
        distances, ids = query.index.search_preassigned(
            query.queries, query.topk, routes, coarse, **query.search_kwargs)
        if profile:
            search_seconds = time.perf_counter() - started
        del coarse
        call_count = 1
    else:
        distances = np.empty((len(query.queries), query.topk), dtype=np.float32)
        ids = np.empty((len(query.queries), query.topk), dtype=np.int64)
        if profile:
            packing_seconds = time.perf_counter() - started
        call_count = 0
        for offset in range(0, len(query.queries), query.faiss_batch_size):
            started = time.perf_counter() if profile else 0.0
            stop = min(offset + query.faiss_batch_size, len(query.queries))
            width = int(probe_counts[offset:stop].max())
            batch_routes = np.ascontiguousarray(routes[offset:stop, :width])
            coarse = workspace.coarse_buffer(stop - offset, width)
            query.index.nprobe = width
            if profile:
                packing_seconds += time.perf_counter() - started
                started = time.perf_counter()
            query.index.search_preassigned(query.queries[offset:stop], query.topk, batch_routes, coarse,
                D=distances[offset:stop], I=ids[offset:stop], **query.search_kwargs)
            call_count += 1
            if profile:
                search_seconds += time.perf_counter() - started
            del batch_routes, coarse
    faiss_ndis = int(stats.ndis) if stats is not None else None
    return L2PRPlusOutput(ids, distances, routes, probe_counts, faiss_ndis,
                          router_seconds, kernel_seconds, assembly_seconds, packing_seconds, search_seconds, call_count)


def l2pr_plus_ndis(query: L2PRPlusQuery, output: L2PRPlusOutput) -> np.ndarray:
    """Optional exact per-query counts for legacy artifacts/checks, outside timing."""
    if query.native_packing:
        return _gain_cost_native.padded_plan_ndis(
            torch.from_numpy(output.routes), torch.from_numpy(output.probe_counts), query.list_sizes).numpy()
    sizes = query.list_sizes.numpy()
    return np.fromiter((sizes[row[:count]].sum() for row, count in
                        zip(output.routes, output.probe_counts, strict=True)),
                       dtype=np.int64, count=len(output.probe_counts))
