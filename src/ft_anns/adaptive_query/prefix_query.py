"""Compact CPU Probability prefix retrieval, separate from benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import torch

from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner,
    PROBABILITY_RETRIEVAL_SETTINGS,
    _gain_cost_native,
)
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.partition import FixedPartition


PREFIX_QUERY_TIMING_SCOPE = "probability-prefix-query-v1"


class ProbabilityPrefixWorkspace:
    """Reusable internal CPU buffers; share only across serial query calls.

    No final results alias these buffers. Disable reuse for a same-chain A/B
    baseline or when the resident buffer memory is not appropriate.
    """

    def __init__(self, *, model: torch.nn.Module, bucket_count: int,
                 batch_size: int, reuse_buffers: bool = True,
                 plan_backend: str = "auto", cache_order_keys: bool = True) -> None:
        # routing.__init__ also imports training/evaluation; defer this import
        # until the query is prepared to avoid their module initialization cycle.
        from ft_anns.routing.model import RawQueryRouter

        if batch_size < 1 or bucket_count < 1:
            raise ValueError("workspace dimensions must be positive")
        if plan_backend not in ("auto", "baseline", "d", "bd"):
            raise ValueError("unknown prefix plan backend")
        self.model = model
        self.bucket_count = bucket_count
        self.batch_size = batch_size
        self.reuse_buffers = reuse_buffers
        supported = all(hasattr(_gain_cost_native, name) for name in (
            "PrefixPlannerWorkspace", "probability_prefix_plan_from_mass_cpu")) and all(
                hasattr(getattr(_gain_cost_native, "PrefixPlannerWorkspace", None), name)
                for name in ("thread_capacity", "cached_order_keys"))
        if plan_backend in ("d", "bd") and not supported:
            raise RuntimeError("rebuild native extension for the compact prefix planner")
        probability_kernel = (_gain_cost_native.prefix_probability_backend()
            if hasattr(_gain_cost_native, "prefix_probability_backend") else "unavailable")
        partial = probability_kernel != "unavailable"
        if plan_backend == "bd" and not partial:
            raise RuntimeError("B/D requires the validated Torch 2.5.1 AVX2/AVX512 row kernel")
        self.plan_backend = ("bd" if partial and plan_backend in ("auto", "bd") else "d") \
            if reuse_buffers and supported and plan_backend != "baseline" else "baseline"
        self.probability_kernel = probability_kernel if self.plan_backend == "bd" else "torch_softmax"
        self.thread_count = torch.get_num_threads()
        self.mass = (torch.empty((batch_size, bucket_count), dtype=torch.float32)
                     if reuse_buffers and self.plan_backend != "bd" else None)
        # A subclass may override forward; retain its normal execution path.
        self.logits = (torch.empty((batch_size, bucket_count), dtype=torch.float32)
                       if reuse_buffers and type(model) is RawQueryRouter else None)
        self.native = (_gain_cost_native.PrefixPlannerWorkspace(
            bucket_count, batch_size, self.plan_backend == "bd", cache_order_keys)
            if self.plan_backend != "baseline" else None)
        if self.native is not None and self.native.thread_capacity != min(
                batch_size, self.thread_count):
            raise RuntimeError("prefix workspace thread capacity does not match Torch threads")
        self._coarse_storage: np.ndarray | None = None
        self.initial_nbytes = self.nbytes

    def ensure_mass(self) -> None:
        """Prepare the original softmax buffer for unsupported gain/cost powers."""
        if self.mass is None:
            self.mass = torch.empty((self.batch_size, self.bucket_count), dtype=torch.float32)
            self.initial_nbytes += self.mass.numel() * self.mass.element_size()

    @property
    def nbytes(self) -> int:
        return sum(buffer.numel() * buffer.element_size()
                   for buffer in (self.mass, self.logits) if buffer is not None) + (
                       self.native.allocated_bytes if self.native is not None else 0) + (
                       self._coarse_storage.nbytes if self._coarse_storage is not None else 0)

    @property
    def mode(self) -> str:
        if self.plan_backend == "bd":
            return "logits_and_prefix" if self.logits is not None else "prefix"
        return ("logits_and_softmax" if self.logits is not None else
                "softmax" if self.mass is not None else "none")

    def coarse_buffer(self, query_count: int, width: int) -> np.ndarray:
        """FAISS reads this zero buffer without modifying it; reuse its capacity."""
        if query_count < 1 or width < 1:
            raise ValueError("coarse buffer dimensions must be positive")
        elements = query_count * width
        if not self.reuse_buffers:
            return np.zeros((query_count, width), dtype=np.float32)
        if self._coarse_storage is None or self._coarse_storage.size < elements:
            self._coarse_storage = np.zeros(elements, dtype=np.float32)
            self._coarse_storage.flags.writeable = False
        return self._coarse_storage[:elements].reshape(query_count, width)

    def validate_threads(self) -> None:
        if self.native is not None and torch.get_num_threads() != self.thread_count:
            raise RuntimeError(
                "Torch thread count changed after prefix workspace preparation; "
                "prepare a new workspace"
            )


class ProbabilityPrefixQuery:
    """Prepare a frozen CPU query workload before starting the query timer.

    Query conversion, model setup, validation and static size tables live here.
    Each call returns fresh results; it never reuses a previous call's outputs.
    Like FixedPartition.search_preassigned, calls sharing an index are serial.
    """

    def __init__(
        self, *, queries: np.ndarray, checkpoint: RouterCheckpoint,
        partition: FixedPartition, planner: GainCostProbePlanner,
        topk: int, router_batch_size: int,
        workspace: ProbabilityPrefixWorkspace | None = None,
    ) -> None:
        queries = np.asarray(queries, dtype=np.float32, order="C")
        if queries.ndim != 2 or queries.shape[1] != partition.dimension or not len(queries):
            raise ValueError("prefix queries must be a nonempty matrix matching the partition")
        if topk < 1 or router_batch_size < 1:
            raise ValueError("topk and router_batch_size must be positive")
        if planner.settings != PROBABILITY_RETRIEVAL_SETTINGS:
            raise ValueError("prefix query requires Probability full_prefix/stable_logit")
        if any(p.device.type != "cpu" or p.dtype != torch.float32
               for p in checkpoint.model.parameters()):
            raise ValueError("prefix query requires a CPU float32 Router")
        checkpoint.validate_for(partition)
        self.queries = queries
        self.query_tensor = torch.from_numpy(queries)
        self.model = checkpoint.model.eval()
        self.index = partition._index
        self.planner = planner
        self.topk = topk
        self.router_batch_size = router_batch_size
        batch_size = min(len(queries), router_batch_size)
        self.workspace = workspace or ProbabilityPrefixWorkspace(
            model=self.model, bucket_count=partition.bucket_count, batch_size=batch_size,
            plan_backend="auto" if planner.implementation == "fused_probability_cpu_float32" else "baseline",
        )
        if (self.workspace.model is not self.model
                or self.workspace.bucket_count != partition.bucket_count
                or self.workspace.batch_size < batch_size):
            raise ValueError("prefix workspace must match the Router, partition and batch capacity")
        self.workspace.validate_threads()
        if (self.workspace.plan_backend == "bd"
                and planner.implementation != "fused_probability_cpu_float32"):
            self.workspace.ensure_mass()
        self.list_sizes = torch.from_numpy(np.array(partition.list_sizes, copy=True))
        self.native_packing = all(hasattr(_gain_cost_native, name) for name in (
            "pack_padded_plan_batches", "padded_plan_ndis",
        ))
        self._native_planner = (self.workspace.native is not None
            and self.planner.implementation == "fused_probability_cpu_float32")
        self._direct_native_batch = (self.native_packing and self._native_planner
            and len(self.queries) <= self.router_batch_size)

    @property
    def coordinates(self) -> dict[str, str | int]:
        return {
            "query_execution_implementation": "compact-prefix-v2",
            "query_timing_scope": PREFIX_QUERY_TIMING_SCOPE,
            "query_plan_packing": "native" if self.native_packing else "numpy",
            "query_plan_assembly": (
                "native_single_batch_if_supported" if self._direct_native_batch
                else "native_merge" if self.native_packing else "numpy"
            ),
            "ndis_statistics_scope": "outside_query_timer",
            "query_workspace": self.workspace.mode,
            "query_workspace_bytes": self.workspace.initial_nbytes,
            "query_workspace_threads": self.workspace.thread_count,
            "query_coarse_buffer": "readonly_zero_capacity" if self.workspace.reuse_buffers else "per_call",
            "query_candidate_order": (
                "cached_float32_keys_v1" if self._native_planner and self.workspace.native.cached_order_keys
                else "logit_comparison"
            ),
            "query_prefix_backend": (self.workspace.plan_backend
                if self.planner.implementation == "fused_probability_cpu_float32" else "baseline"),
            "query_probability_kernel": (self.workspace.probability_kernel
                if self.planner.implementation == "fused_probability_cpu_float32" else "torch_softmax"),
        }


@dataclass(frozen=True)
class PrefixQueryOutput:
    """Owned FAISS outputs and the plan needed for untimed NDis reporting."""

    neighbor_ids: np.ndarray
    distances: np.ndarray
    routes: np.ndarray
    probe_counts: np.ndarray
    router_seconds: float
    kernel_seconds: float
    assembly_seconds: float
    packing_seconds: float
    search_seconds: float


def query_probability_prefix(query: ProbabilityPrefixQuery) -> PrefixQueryOutput:
    """Run Router -> global softmax/prefix argmax -> padded FAISS search.

    Time the complete call, including allocations and temporary cleanup.
    No file I/O, metadata, NDis/Recall statistics, result wrappers or logging.
    The planner preserves stable Top-M order and the earliest prefix argmax.
    """
    route_batches, count_batches = [], []
    router_seconds = kernel_seconds = 0.0
    workspace = query.workspace
    workspace.validate_threads()
    direct_native_batch = query._direct_native_batch
    with torch.inference_mode():
        for offset in range(0, len(query.queries), query.router_batch_size):
            started = time.perf_counter()
            batch = query.query_tensor[offset:offset + query.router_batch_size]
            logits = (query.model(batch, out=workspace.logits[:len(batch)])
                      if workspace.logits is not None else query.model(batch))
            router_seconds += time.perf_counter() - started
            started = time.perf_counter()
            direct_native_batch = (direct_native_batch and logits.dtype == torch.float32
                and logits.is_contiguous())
            selected, counts = query.planner.build_tensors(
                logits, probability_buffer=(workspace.mass[:len(batch)]
                                            if workspace.mass is not None else None),
                native_workspace=workspace.native,
            )
            # A narrow view can retain the entire batch x candidate-bound storage,
            # even for a one-row batch where contiguous() would not copy it.
            if (not selected.is_contiguous() or selected.untyped_storage().nbytes()
                    > selected.numel() * selected.element_size()):
                selected = selected.clone(memory_format=torch.contiguous_format)
            route_batches.append(selected)
            count_batches.append(counts)
            del logits, selected, counts
            kernel_seconds += time.perf_counter() - started

        started = time.perf_counter()
        if direct_native_batch:
            # Native D/B planners already return an independently allocated,
            # contiguous padded matrix. Keep its Tensor alive through NumPy's
            # ndarray base instead of copying it a second time.
            routes, probe_counts = route_batches[0].numpy(), count_batches[0].numpy()
        elif query.native_packing:
            routes_tensor, counts_tensor = _gain_cost_native.pack_padded_plan_batches(
                route_batches, count_batches, query.list_sizes,
            )
            routes, probe_counts = routes_tensor.numpy(), counts_tensor.numpy()
        else:
            width = max(batch.shape[1] for batch in route_batches)
            routes = np.full((len(query.queries), width), -1, dtype=np.int64)
            probe_counts = np.concatenate([batch.numpy() for batch in count_batches])
            offset = 0
            for batch, counts in zip(route_batches, count_batches, strict=True):
                destination = routes[offset:offset + len(batch), :batch.shape[1]]
                np.copyto(destination, batch.numpy())
                destination[np.arange(batch.shape[1])[None, :] >= counts.numpy()[:, None]] = -1
                offset += len(batch)
            del batch, counts, destination
        route_batches.clear()
        count_batches.clear()
        assembly_seconds = time.perf_counter() - started

    started = time.perf_counter()
    coarse = query.workspace.coarse_buffer(len(query.queries), routes.shape[1])
    query.index.nprobe = routes.shape[1]
    packing_seconds = time.perf_counter() - started
    started = time.perf_counter()
    distances, neighbor_ids = query.index.search_preassigned(
        query.queries, query.topk, routes, coarse,
    )
    search_seconds = time.perf_counter() - started
    del coarse
    return PrefixQueryOutput(neighbor_ids, distances, routes, probe_counts,
                             router_seconds, kernel_seconds, assembly_seconds,
                             packing_seconds, search_seconds)


def prefix_query_ndis(query: ProbabilityPrefixQuery, output: PrefixQueryOutput) -> np.ndarray:
    """Exact per-query NDis for reports; call only after stopping the query timer."""
    if query.native_packing:
        return _gain_cost_native.padded_plan_ndis(
            torch.from_numpy(output.routes), torch.from_numpy(output.probe_counts),
            query.list_sizes,
        ).numpy()
    sizes = query.list_sizes.numpy()
    return np.fromiter((sizes[row[:count]].sum() for row, count in
                        zip(output.routes, output.probe_counts, strict=True)),
                       dtype=np.int64, count=len(output.probe_counts))
