"""Shared timestamp and timing primitives for workflow observability."""

from __future__ import annotations

import os
import resource
import sys
import threading
import time
import weakref
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ft_anns.domain import SearchExecutionTrace

CALIBRATION_COUNTER_NAMES = (
    "generated_candidate_count",
    "pruned_candidate_count",
    "feasible_candidate_count",
    "evaluated_candidate_count",
    "selected_candidate_count",
    "logical_candidate_plan_evaluation_count",
    "unique_active_probe_plan_count",
    "reused_active_probe_plan_count",
    "avoided_faiss_search_count",
    "faiss_call_count",
)

CALIBRATION_TIMING_NAMES = (
    "router_seconds",
    "policy_table_seconds",
    "lambda_fit_seconds",
    "cost_planning_seconds",
    "packing_seconds",
    "faiss_seconds",
    "metric_seconds",
    "total_seconds",
)


def _schedule_interval_heartbeat(
    callback: Callable[[], None], interval_seconds: float
) -> Callable[[], None]:
    callback_reference = weakref.WeakMethod(callback)
    stopped = threading.Event()

    def run() -> None:
        while not stopped.wait(interval_seconds):
            current_callback = callback_reference()
            if current_callback is None:
                return
            current_callback()
            del current_callback

    thread = threading.Thread(
        target=run,
        name="ft-anns-calibration-progress",
        daemon=True,
    )
    thread.start()

    def cancel() -> None:
        stopped.set()
        if thread is not threading.current_thread():
            thread.join()

    return cancel


def utc_now_iso() -> str:
    """Return the current UTC time in the artifact-wide millisecond format."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def calibration_runtime_observations(
    *, requested_device: str, resolved_device: str
) -> dict[str, Any]:
    """Snapshot calibration device, thread, and process resource observations."""
    import faiss
    import torch

    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        peak_rss *= 1024
    cuda_available = bool(torch.cuda.is_available())
    cuda_identity = None
    if cuda_available and resolved_device == "cuda":
        device_index = int(torch.cuda.current_device())
        properties = torch.cuda.get_device_properties(device_index)
        cuda_identity = {
            "pytorch_cuda_version": torch.version.cuda,
            "device_index": device_index,
            "name": str(properties.name),
            "compute_capability": [
                int(properties.major),
                int(properties.minor),
            ],
            "total_memory_bytes": int(properties.total_memory),
        }
    return {
        "requested_device": requested_device,
        "resolved_device": resolved_device,
        "cuda_identity": cuda_identity,
        "threads": {
            "cpu_count": max(1, os.cpu_count() or 1),
            "pytorch_intraop": int(torch.get_num_threads()),
            "pytorch_interop": int(torch.get_num_interop_threads()),
            "faiss_openmp": int(faiss.omp_get_max_threads()),
            "openmp_env": os.environ.get("OMP_NUM_THREADS"),
        },
        "resources": {
            "process_peak_rss_bytes": max(0, peak_rss),
            "cuda_peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if cuda_available and resolved_device == "cuda"
                else None
            ),
        },
    }


class CalibrationProgressReporter:
    """Emit bounded-cadence structured progress for one calibration stage."""

    def __init__(
        self,
        *,
        emit: Callable[[dict[str, Any]], None],
        clock: Callable[[], float] = time.monotonic,
        interval_seconds: float = 30.0,
        requested_device: str = "cpu",
        resolved_device: str = "cpu",
        observe_runtime: Callable[[], Mapping[str, Any]] | None = None,
        schedule_heartbeat: Callable[
            [Callable[[], None], float], Callable[[], None]
        ] = _schedule_interval_heartbeat,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("progress interval must be positive")
        self._emit = emit
        self._clock = clock
        self._interval_seconds = float(interval_seconds)
        self._requested_device = requested_device
        self._resolved_device = resolved_device
        self._observe_runtime = observe_runtime
        self._schedule_heartbeat = schedule_heartbeat
        self._cancel_heartbeat: Callable[[], None] | None = None
        self._lock = threading.RLock()
        self._stage = ""
        self._total_batches: int | None = None
        self._completed_batches = 0
        self._stage_started = 0.0
        self._last_emitted = 0.0
        self._counters = {name: 0 for name in CALIBRATION_COUNTER_NAMES}
        self._timing = {name: 0.0 for name in CALIBRATION_TIMING_NAMES}

    def start(self, *, stage: str, total_batches: int | None = None) -> None:
        if not stage or total_batches is not None and total_batches < 0:
            raise ValueError("progress stage inputs are invalid")
        self.close()
        with self._lock:
            now = self._clock()
            self._stage = stage
            self._total_batches = total_batches
            self._completed_batches = 0
            self._stage_started = now
            self._last_emitted = now
            self._counters = {name: 0 for name in CALIBRATION_COUNTER_NAMES}
            self._timing = {name: 0.0 for name in CALIBRATION_TIMING_NAMES}
            self._emit_payload("started", now)
            self._cancel_heartbeat = self._schedule_heartbeat(
                self._heartbeat, self._interval_seconds
            )

    def update(
        self,
        *,
        completed_batches: int | None = None,
        timing_seconds: Mapping[str, float] | None = None,
        **counters: int,
    ) -> None:
        with self._lock:
            now = self._clock()
            self._apply_update(completed_batches, timing_seconds, counters, now)
            if now - self._last_emitted + 1e-12 >= self._interval_seconds:
                self._last_emitted = now
                self._emit_payload("active", now)

    def finish(
        self,
        *,
        completed_batches: int | None = None,
        timing_seconds: Mapping[str, float] | None = None,
        **counters: int,
    ) -> None:
        self.close()
        with self._lock:
            now = self._clock()
            self._apply_update(completed_batches, timing_seconds, counters, now)
            self._emit_payload("completed", now)
            self._stage = ""

    def close(self) -> None:
        """Stop the active stage heartbeat without emitting completion."""
        with self._lock:
            cancel = self._cancel_heartbeat
            self._cancel_heartbeat = None
        if cancel is not None:
            cancel()

    def _heartbeat(self) -> None:
        with self._lock:
            if not self._stage:
                return
            now = self._clock()
            self._last_emitted = now
            self._emit_payload("active", now)

    def _apply_update(
        self,
        completed_batches: int | None,
        timing_seconds: Mapping[str, float] | None,
        counters: Mapping[str, int],
        now: float,
    ) -> None:
        if not self._stage:
            raise RuntimeError("progress stage has not started")
        if completed_batches is not None:
            if completed_batches < 0:
                raise ValueError("completed batch count must be non-negative")
            self._completed_batches = completed_batches
        for name, value in counters.items():
            if name not in self._counters or value < 0:
                raise ValueError(f"invalid calibration progress counter: {name}")
            self._counters[name] = int(value)
        for name, value in (timing_seconds or {}).items():
            if name not in self._timing or value < 0:
                raise ValueError(f"invalid calibration progress timing: {name}")
            self._timing[name] = float(value)
        self._timing["total_seconds"] = max(
            self._timing["total_seconds"], now - self._stage_started
        )

    def _emit_payload(self, status: str, now: float) -> None:
        timing = dict(self._timing)
        timing["total_seconds"] = max(
            timing["total_seconds"], now - self._stage_started
        )
        payload = {
            "event": "calibration_progress",
            "status": status,
            "stage": self._stage,
            "completed_batches": self._completed_batches,
            "total_batches": self._total_batches,
            "requested_device": self._requested_device,
            "resolved_device": self._resolved_device,
            **self._counters,
            "timing_seconds": timing,
        }
        if self._observe_runtime is not None:
            payload.update(self._observe_runtime())
        self._emit(payload)


@dataclass(frozen=True)
class SearchTimingTrace:
    """Serializable router, probe-plan, search, and optional metric timing."""

    router_seconds: float
    search_seconds: float
    total_seconds: float
    probe_plan_seconds: float = 0.0
    probe_plan_kernel_seconds: float | None = None
    probe_plan_assembly_seconds: float = 0.0
    search_total_seconds: float | None = None
    metric_seconds: float | None = None
    search_packing_seconds: float = 0.0
    search_ndis_seconds: float = 0.0
    faiss_call_count: int = 0
    search_execution: SearchExecutionTrace = "grouped"
    faiss_parallel_mode: int | None = None
    router_batch_size: int | str | None = None
    faiss_batch_size: int | str | None = None

    def __post_init__(self) -> None:
        search_total_seconds = (
            self.search_seconds
            if self.search_total_seconds is None
            else self.search_total_seconds
        )
        object.__setattr__(self, "search_total_seconds", search_total_seconds)
        metric_seconds = self.metric_seconds or 0.0
        probe_plan_kernel_seconds = (
            self.probe_plan_seconds
            if self.probe_plan_kernel_seconds is None
            else self.probe_plan_kernel_seconds
        )
        object.__setattr__(self, "probe_plan_kernel_seconds", probe_plan_kernel_seconds)
        if (
            self.router_seconds < 0
            or self.probe_plan_seconds < 0
            or self.search_seconds < 0
            or search_total_seconds < self.search_seconds
            or metric_seconds < 0
            or probe_plan_kernel_seconds < 0
            or self.probe_plan_assembly_seconds < 0
            or probe_plan_kernel_seconds + self.probe_plan_assembly_seconds
            > self.probe_plan_seconds + 1e-12
            or self.search_packing_seconds < 0
            or self.search_ndis_seconds < 0
            or self.faiss_call_count < 0
            or self.search_execution not in ("grouped", "padded", "mixed")
            or (self.faiss_parallel_mode is not None and self.faiss_parallel_mode < 0)
            or not self._valid_batch_size(self.router_batch_size)
            or not self._valid_batch_size(self.faiss_batch_size)
            or self.total_seconds + 1e-12
            < self.router_seconds
            + self.probe_plan_seconds
            + search_total_seconds
            + metric_seconds
        ):
            raise ValueError("search timing values are inconsistent")

    @staticmethod
    def _valid_batch_size(value: int | str | None) -> bool:
        return (
            value is None
            or value == "full"
            or (isinstance(value, int) and not isinstance(value, bool) and value > 0)
        )

    @property
    def overhead_seconds(self) -> float:
        assert self.search_total_seconds is not None
        return max(
            0.0,
            self.total_seconds
            - self.router_seconds
            - self.probe_plan_seconds
            - self.search_total_seconds
            - (self.metric_seconds or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "router_seconds": self.router_seconds,
            "probe_plan_seconds": self.probe_plan_seconds,
            "probe_plan_kernel_seconds": self.probe_plan_kernel_seconds or 0.0,
            "probe_plan_assembly_seconds": self.probe_plan_assembly_seconds,
            "search_seconds": self.search_seconds,
            "search_total_seconds": self.search_total_seconds or 0.0,
            "search_packing_seconds": self.search_packing_seconds,
            "search_ndis_seconds": self.search_ndis_seconds,
            "faiss_call_count": self.faiss_call_count,
            "search_execution": self.search_execution,
            "overhead_seconds": self.overhead_seconds,
            "total_seconds": self.total_seconds,
        }
        if self.metric_seconds is not None:
            payload["metric_seconds"] = self.metric_seconds
        if self.faiss_parallel_mode is not None:
            payload["faiss_parallel_mode"] = self.faiss_parallel_mode
        if self.router_batch_size is not None:
            payload["router_batch_size"] = self.router_batch_size
        if self.faiss_batch_size is not None:
            payload["faiss_batch_size"] = self.faiss_batch_size
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SearchTimingTrace:
        metric_seconds = payload.get("metric_seconds")
        return cls(
            router_seconds=float(payload["router_seconds"]),
            search_seconds=float(payload["search_seconds"]),
            total_seconds=float(payload["total_seconds"]),
            probe_plan_seconds=float(payload.get("probe_plan_seconds", 0.0)),
            probe_plan_kernel_seconds=float(
                payload.get(
                    "probe_plan_kernel_seconds",
                    payload.get("probe_plan_seconds", 0.0),
                )
            ),
            probe_plan_assembly_seconds=float(
                payload.get("probe_plan_assembly_seconds", 0.0)
            ),
            search_total_seconds=float(
                payload.get("search_total_seconds", payload["search_seconds"])
            ),
            metric_seconds=(
                float(metric_seconds) if metric_seconds is not None else None
            ),
            search_packing_seconds=float(payload.get("search_packing_seconds", 0.0)),
            search_ndis_seconds=float(payload.get("search_ndis_seconds", 0.0)),
            faiss_call_count=int(payload.get("faiss_call_count", 0)),
            search_execution=payload.get("search_execution", "grouped"),
            faiss_parallel_mode=(
                int(payload["faiss_parallel_mode"])
                if payload.get("faiss_parallel_mode") is not None
                else None
            ),
            router_batch_size=payload.get("router_batch_size"),
            faiss_batch_size=payload.get("faiss_batch_size"),
        )
