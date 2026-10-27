#!/usr/bin/env python3
"""Audit non-FAISS prefix query stages on real 100M Router checkpoints.

No large IVF index is loaded. Alternate the production planner and an exact
decomposition of its native path; validate routes/counts after every pair.
These are diagnostic stage wall times, not end-to-end retrieval latencies.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np
import torch

from compare_probability_top_m import DATASETS, ROOT
from ft_anns.adaptive_query import gain_cost as gc
from ft_anns.artifact_io import write_json_atomic
from ft_anns.config import RouterConfig
from ft_anns.datasets.base_gt import FileVectorSource
from ft_anns.routing.model import RawQueryRouter


def run_once(model, query_tensor, planner, list_sizes, batch_size, *, profiled):
    stages, route_batches, count_batches = {}, [], []
    def record(name, started):
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - started

    with torch.inference_mode():
        for offset in range(0, len(query_tensor), batch_size):
            started = time.perf_counter()
            logits = model(query_tensor[offset:offset + batch_size])
            record("router", started)
            if profiled:
                started = time.perf_counter()
                selected = gc._stable_topk_indices(logits, planner.candidate_bound)
                record("candidate_selection_sort", started)
                started = time.perf_counter()
                mass = torch.softmax(logits, dim=1)
                record("global_softmax", started)
                started = time.perf_counter()
                counts = gc._gain_cost_native.probability_prefix_counts_float32(
                    mass, planner._normalized_sizes, selected, planner._alpha, planner._beta,
                )
                record("prefix_scan", started)
                started = time.perf_counter()
                routes = selected[:, :int(counts.max().item())]
                record("width_reduce_slice", started)
                started = time.perf_counter()
                del mass, selected
                record("mass_release", started)
            else:
                started = time.perf_counter()
                routes, counts = planner.build_tensors(logits)
                record("planner_unsplit", started)
            started = time.perf_counter()
            if (not routes.is_contiguous() or routes.untyped_storage().nbytes()
                    > routes.numel() * routes.element_size()):
                routes = routes.clone(memory_format=torch.contiguous_format)
            route_batches.append(routes)
            count_batches.append(counts)
            del logits, routes, counts
            record("compact_and_release", started)
        started = time.perf_counter()
        routes, counts = gc._gain_cost_native.pack_padded_plan_batches(
            route_batches, count_batches, list_sizes,
        )
        route_batches.clear()
        count_batches.clear()
        record("assembly", started)
    started = time.perf_counter()
    coarse = np.zeros(tuple(routes.shape), dtype=np.float32)
    del coarse
    record("coarse_alloc_release_without_faiss", started)
    return routes, counts, stages


def profile_candidate_selection(logits, bound, *, repeats, warmup, scratch_bits=64):
    """Compare the untimed selector with an instrumented clone of its loop.

    Worker durations sum concurrent work: they are not stage wall time and
    must not be added to or substituted for complete query measurements.
    """
    native = gc._gain_cost_native
    suffix = "_int32_scratch_cpu" if scratch_bits == 32 else "_cpu"
    plain_selector = getattr(native, "sorted_top_m" + suffix, None)
    profiled_selector = getattr(native, "profile_sorted_top_m" + suffix, None)
    if plain_selector is None or profiled_selector is None:
        raise RuntimeError("rebuild the extension for candidate-stage diagnostics")
    samples = []
    for iteration in range(warmup + repeats):
        outputs, sample = {}, {}
        order = ("plain", "profiled") if iteration % 2 == 0 else ("profiled", "plain")
        for name in order:
            started = time.perf_counter()
            if name == "plain":
                selected = plain_selector(logits, bound)
            else:
                selected, row_times, thread_times, shared_times = profiled_selector(logits, bound)
            sample[name + "_wall_seconds"] = time.perf_counter() - started
            outputs[name] = selected
        if not torch.equal(outputs["plain"], outputs["profiled"]):
            raise ValueError("instrumented selector changed stable Top-M results")
        sample["worker_seconds"] = dict(zip(
            ("initialize_ids", "nth_element", "sort", "copy_selected"),
            row_times.sum(0).tolist(), strict=True,
        ))
        sample["scratch_worker_seconds"] = dict(zip(
            ("allocate", "release"), thread_times.sum(0).tolist(), strict=True,
        ))
        sample["allocation_wall_seconds"], sample["parallel_region_wall_seconds"] = shared_times.tolist()
        if iteration >= warmup:
            samples.append(sample)
        del outputs, selected, row_times, thread_times, shared_times
    median_work = {key: float(np.median([s["worker_seconds"][key] for s in samples]))
                   for key in samples[0]["worker_seconds"]}
    total_work = sum(median_work.values())
    plain_wall = float(np.median([s["plain_wall_seconds"] for s in samples]))
    profiled_wall = float(np.median([s["profiled_wall_seconds"] for s in samples]))
    return {
        "query_count": len(logits), "candidate_bound": bound, "all_selections_identical": True,
        "scratch_bits": scratch_bits,
        "note": "Cached real logits; no Router/softmax/prefix/FAISS. Worker shares are concurrent service time, not wall fractions. Profiled wall includes instrumentation.",
        "plain_wall_us_per_query": plain_wall * 1e6 / len(logits),
        "profiled_wall_us_per_query": profiled_wall * 1e6 / len(logits),
        "instrumentation_wall_overhead_percent": 100 * (profiled_wall / plain_wall - 1),
        "worker_time_share": {key: value / total_work if total_work else 0.0
                              for key, value in median_work.items()},
        "samples": samples,
    }


def compare_candidate_scratch(logits, bound, *, repeats, warmup):
    """Paired native selectors on identical cached logits, with int64 outputs."""
    native = gc._gain_cost_native
    selectors = {"int64": native.sorted_top_m_cpu,
                 "int32": native.sorted_top_m_int32_scratch_cpu}
    samples = []
    for iteration in range(warmup + repeats):
        order = ("int64", "int32") if iteration % 2 == 0 else ("int32", "int64")
        outputs, sample = {}, {"order": list(order)}
        for name in order:
            started = time.perf_counter()
            outputs[name] = selectors[name](logits, bound)
            sample[name] = time.perf_counter() - started
        if (outputs["int32"].dtype != torch.int64
                or not torch.equal(outputs["int64"], outputs["int32"])):
            raise ValueError("int32 scratch changed int64 candidate outputs")
        if iteration >= warmup:
            samples.append(sample)
        del outputs
    times = {name: np.array([s[name] for s in samples]) for name in selectors}
    return {
        "query_count": len(logits), "candidate_bound": bound, "all_selections_identical": True,
        "note": "Cached real logits; native selector call including output allocation, excluding output release. Both return int64; no Router/softmax/prefix/FAISS.",
        "median_us_per_query": {name: float(np.median(values) * 1e6 / len(logits))
                                for name, values in times.items()},
        "time_reduction_percent": float(100 * (1 - np.median(times["int32"]) / np.median(times["int64"]))),
        "int32_faster_pair_count": int((times["int32"] < times["int64"]).sum()),
        "scratch_bytes": {name: torch.get_num_threads() * logits.shape[1] * bits // 8
                          for name, bits in (("int64", 64), ("int32", 32))},
        "output_bytes": len(logits) * bound * 8,
        "samples": samples,
        "int32_diagnostics": profile_candidate_selection(
            logits, bound, repeats=repeats, warmup=warmup, scratch_bits=32),
    }


def run_dataset(args, dataset):
    output = args.output_root / dataset
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"use a fresh output directory: {output}")
    run = ROOT / "runs" / f"{dataset}_gtk100_bucket65536_niter10_pseed42_ramps"
    partition = ROOT / "artifacts" / dataset / f"{dataset}_bucket65536_niter10_pseed42_fixed-partition"
    metadata = json.loads((run / "best/metadata.json").read_text())["model"]
    model = RawQueryRouter(input_dimension=metadata["input_dimension"], bucket_count=metadata["bucket_count"],
        config=RouterConfig.model_validate({k: metadata[k] for k in ("hidden_dimensions", "activation")})).eval()
    model.load_state_dict(torch.load(run / "best/weights.pt", map_location="cpu", weights_only=True))
    parameters_path = run / "learned_gain_cost_prefix_prototype/recall_8_targets_policy_100k_fixed_a1_b01_probability/parameters.json"
    parameters = json.loads(parameters_path.read_text())
    if gc.parse_gain_cost_retrieval_settings(parameters) != gc.PROBABILITY_RETRIEVAL_SETTINGS:
        raise ValueError("requires Probability full-prefix parameters")
    if any(float(parameters.get("cache_metadata", {}).get(k, 1)) != 1 for k in (
        "gain_power_before_prefix_sum", "cost_power_before_prefix_sum")):
        raise ValueError("stage decomposition requires unit powers")
    file, format_name, _ = DATASETS[dataset]
    source = FileVectorSource(ROOT / "data" / dataset / file, format_name, model.input_dimension)
    row_ids = np.sort(np.random.default_rng(args.seed).choice(source.count, args.query_count, replace=False))
    queries = torch.from_numpy(np.array(source.read_rows(row_ids), dtype=np.float32, order="C", copy=True))
    raw_sizes = np.load(partition / "list_sizes.npy")
    sizes = torch.from_numpy(raw_sizes.astype(np.float32) / float(raw_sizes.max()))
    list_sizes = torch.from_numpy(raw_sizes.copy())
    output.mkdir(parents=True)
    np.save(output / "query_row_ids.npy", row_ids)
    payload = {"format": "probability-prefix-non-faiss-stages-v1", "status": "running",
        "dataset": dataset, "query_count": len(queries), "seed": args.seed,
        "bucket_count": model.bucket_count, "router_batch_size": args.router_batch_size,
        "repeats": args.repeats, "warmup_pairs": args.warmup, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "parameters_sha256": hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
        "note": "No FAISS search; times include deallocation. Stage medians need not sum to median total.",
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (
            Path(__file__), ROOT / "src/ft_anns/adaptive_query/gain_cost.py",
            ROOT / "src/ft_anns/adaptive_query/prefix_query.py",
            ROOT / "src/ft_anns/adaptive_query/_gain_cost_native.cpp")}, "points": {}}
    for variant in args.variants:
        limit = {"ratio_1_2": gc.PrefixCandidateLimit(ratio="1/2"),
                 "top_256": gc.PrefixCandidateLimit(count=256), "full_M": gc.PrefixCandidateLimit()}[variant]
        planner = gc.GainCostProbePlanner(normalized_sizes=sizes, policy=parameters["targets"][args.target],
            gain_power=1, cost_power=1, settings=gc.PROBABILITY_RETRIEVAL_SETTINGS, prefix_candidate_limit=limit)
        if planner.implementation != "fused_probability_cpu_float32":
            raise RuntimeError("requires native prefix and selection kernels")
        samples = []
        for iteration in range(args.warmup + args.repeats):
            order = (False, True) if iteration % 2 == 0 else (True, False)
            outputs, pair = {}, {}
            for profiled in order:
                started = time.perf_counter()
                routes, counts, stages = run_once(model, queries, planner, list_sizes,
                    args.router_batch_size, profiled=profiled)
                total = time.perf_counter() - started
                name = "decomposed" if profiled else "production_planner"
                outputs[name] = (routes, counts)
                pair[name] = {"stages_seconds": stages, "total_seconds": total}
            for a, b in zip(outputs["production_planner"], outputs["decomposed"], strict=True):
                if not torch.equal(a, b):
                    raise ValueError("decomposed routes/counts differ from production planner")
            del a, b, outputs, routes, counts
            if iteration >= args.warmup:
                samples.append(pair)
        point = {"candidate_bound": planner.candidate_bound, "all_plans_identical": True,
                 "samples": samples, "median_us_per_query": {}}
        for name in ("decomposed", "production_planner"):
            point["median_us_per_query"][name] = {
                key: float(np.median([p[name]["stages_seconds"][key] for p in samples]) * 1e6 / len(queries))
                for key in samples[0][name]["stages_seconds"]}
            point["median_us_per_query"][name]["total"] = float(
                np.median([p[name]["total_seconds"] for p in samples]) * 1e6 / len(queries))
        with torch.inference_mode():
            logits = model(queries[:args.router_batch_size])
            point["candidate_diagnostics"] = profile_candidate_selection(
                logits, planner.candidate_bound, repeats=args.repeats, warmup=args.warmup,
            )
            if args.compare_int32_scratch:
                point["scratch_comparison"] = compare_candidate_scratch(
                    logits, planner.candidate_bound, repeats=args.repeats, warmup=args.warmup,
                )
            del logits
        payload["points"][variant] = point
        write_json_atomic(output / "profile.json", payload)
        print(json.dumps({"dataset": dataset, "variant": variant, **point["median_us_per_query"],
                          "candidate_worker_time_share": point["candidate_diagnostics"]["worker_time_share"],
                          **({"scratch_comparison": {k: v for k, v in point["scratch_comparison"].items()
                                                     if k not in ("samples", "int32_diagnostics")}}
                             if args.compare_int32_scratch else {})}), flush=True)
    payload["status"] = "complete"
    write_json_atomic(output / "profile.json", payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--target", default="0.99")
    parser.add_argument("--variants", nargs="+", choices=("ratio_1_2", "top_256", "full_M"),
                        default=["ratio_1_2", "top_256", "full_M"])
    parser.add_argument("--query-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--compare-int32-scratch", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if min(args.query_count, args.threads, args.router_batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    with Path("/tmp/ft_anns_100m_cpu_search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for dataset in args.datasets:
            run_dataset(args, dataset)


if __name__ == "__main__":
    main()
