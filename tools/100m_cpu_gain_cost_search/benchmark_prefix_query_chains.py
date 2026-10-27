#!/usr/bin/env python3
"""Paired old/new prefix-query measurements on a real 100M IVF index.

Reports both historical/new query timer scopes and a common complete-call
scope including exact NDis and immutable result construction in both chains.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
from unittest.mock import patch

import faiss
import numpy as np
import torch

from compare_probability_top_m import DATASETS, ROOT, retrieve as retrieve_old
from search_sweep import _load_json, _validate_parameters
from ft_anns.adaptive_query.evaluation import compute_recall_at_k
from ft_anns.adaptive_query import gain_cost as gc
from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner, PrefixCandidateLimit, PROBABILITY_RETRIEVAL_SETTINGS,
)
from ft_anns.adaptive_query.prefix_query import ProbabilityPrefixQuery, ProbabilityPrefixWorkspace
from ft_anns.adaptive_query.retrieval import retrieve_probability_prefix
from ft_anns.artifact_io import write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.datasets.base_gt import FileVectorSource, read_ivecs
from ft_anns.partition import FixedPartition


DEFAULT_POINTS = ("0.90:ratio_1_2", "0.99:ratio_1_2", "0.99:top_256")


def validate_workspace(query, *, candidate_selector=None):
    """Check actual workload batches outside all query measurements."""
    workspace = query.workspace
    with torch.inference_mode():
        for offset in range(0, len(query.queries), query.router_batch_size):
            batch = query.query_tensor[offset:offset + query.router_batch_size]
            expected = query.model(batch)
            actual = (query.model(batch, out=workspace.logits[:len(batch)])
                      if workspace.logits is not None else query.model(batch))
            if not torch.equal(expected, actual):
                raise ValueError("workspace changes Router logits")
            expected_mass = torch.softmax(expected, 1)
            if workspace.plan_backend == "bd":
                ids = gc._stable_topk_indices(actual, query.planner.candidate_bound)
                actual_mass = gc._gain_cost_native.prefix_candidate_probabilities_cpu(actual, ids, workspace.native)
                reference_mass = expected_mass.gather(1, ids)
            else:
                actual_mass = torch.softmax(actual, 1, out=(workspace.mass[:len(batch)]
                                                           if workspace.mass is not None else None))
                reference_mass = expected_mass
            if not torch.equal(reference_mass, actual_mass):
                raise ValueError("workspace changes global probabilities")
            expected_plan = query.planner.build_tensors(expected)
            with (patch.object(gc._gain_cost_native, "sorted_top_m_cpu", candidate_selector)
                  if candidate_selector is not None else nullcontext()):
                actual_plan = query.planner.build_tensors(actual,
                    probability_buffer=workspace.mass[:len(batch)] if workspace.mass is not None else None,
                    native_workspace=workspace.native)
            expected_routes, expected_counts = expected_plan
            actual_routes, actual_counts = actual_plan
            valid = torch.arange(expected_routes.shape[1])[None, :] < expected_counts[:, None]
            if (not torch.equal(expected_counts, actual_counts)
                    or not torch.equal(expected_routes[valid], actual_routes[valid])):
                raise ValueError("workspace changes routes/counts")


def summarize_pairs(pairs: list[dict], query_count: int) -> dict:
    summary = {}
    for scope in ("reported_seconds", "complete_call_seconds"):
        old = np.array([pair["old"][scope] for pair in pairs])
        new = np.array([pair["new"][scope] for pair in pairs])
        summary[scope] = {
            "old_median_ms_per_query": float(np.median(old) * 1000 / query_count),
            "new_median_ms_per_query": float(np.median(new) * 1000 / query_count),
            "speedup_ratio_of_medians": float(np.median(old) / np.median(new)),
            "time_reduction_percent": float(100 * (1 - np.median(new) / np.median(old))),
            "paired_speedup_median": float(np.median(old / new)),
            "new_faster_pair_count": int((new < old).sum()),
            "old_ms_per_query_samples": (old * 1000 / query_count).tolist(),
            "new_ms_per_query_samples": (new * 1000 / query_count).tolist(),
        }
    summary["median_stages_ms_per_query"] = {
        chain: {name: float(np.median([pair[chain]["timing"][name] for pair in pairs])
                            * 1000 / query_count)
                for name in pairs[0][chain]["timing"] if name.endswith("_seconds")}
        for chain in ("old", "new")
    }
    return summary


def benchmark_point(*, query, checkpoint, partition, parameters_path, coordinates,
                    truth, repeats: int, warmup: int, baseline: str = "legacy") -> dict:
    """Run paired complete calls; compare every result outside measured regions."""
    baseline_query = None
    selectors = None
    if baseline == "compact_comparison_order" and query.workspace.native is None:
        raise ValueError("comparison-order baseline requires a native D/B workspace")
    if baseline in ("compact_unbuffered", "compact_buffered", "compact_d", "compact_comparison_order"):
        baseline_query = ProbabilityPrefixQuery(
            queries=query.queries, checkpoint=checkpoint, partition=partition,
            planner=query.planner, topk=query.topk, router_batch_size=query.router_batch_size,
            workspace=ProbabilityPrefixWorkspace(model=checkpoint.model,
                bucket_count=partition.bucket_count, batch_size=min(len(query.queries), query.router_batch_size),
                reuse_buffers=baseline != "compact_unbuffered",
                plan_backend=(query.workspace.plan_backend if baseline == "compact_comparison_order"
                              else "d" if baseline == "compact_d" else "baseline"),
                cache_order_keys=baseline != "compact_comparison_order"),
        )
    elif baseline == "scratch_int64":
        native = gc._gain_cost_native
        if not hasattr(native, "sorted_top_m_int32_scratch_cpu"):
            raise RuntimeError("rebuild extension for int32 scratch validation")
        if query.workspace.plan_backend != "baseline":
            query = ProbabilityPrefixQuery(
                queries=query.queries, checkpoint=checkpoint, partition=partition,
                planner=query.planner, topk=query.topk, router_batch_size=query.router_batch_size,
                workspace=ProbabilityPrefixWorkspace(model=checkpoint.model,
                    bucket_count=partition.bucket_count,
                    batch_size=min(len(query.queries), query.router_batch_size), plan_backend="baseline"),
            )
        baseline_query = query
        selectors = {"old": native.sorted_top_m_cpu,
                     "new": native.sorted_top_m_int32_scratch_cpu}
    elif baseline != "legacy":
        raise ValueError("unknown benchmark baseline")
    validate_workspace(query, candidate_selector=selectors["new"] if selectors else None)

    def selector_context(name):
        # This benchmark is serial. Restore the native binding after every call;
        # the production selector and FAISS int64 interface remain unchanged.
        return (patch.object(gc._gain_cost_native, "sorted_top_m_cpu", selectors[name])
                if selectors else nullcontext())

    def old():
        if baseline_query is not None:
            return retrieve_probability_prefix(
                baseline_query, checkpoint_filename=checkpoint.filename,
                partition_filename=partition.filename,
                query_filename=coordinates["query_filename"],
                parameters_filename=parameters_path.name, experiment_coordinates=coordinates,
            )
        return retrieve_old(query.queries, checkpoint, partition, query.planner,
                            parameters_path, coordinates, query.router_batch_size, query.topk)

    def new():
        return retrieve_probability_prefix(
            query, checkpoint_filename=checkpoint.filename,
            partition_filename=partition.filename,
            query_filename=coordinates["query_filename"],
            parameters_filename=parameters_path.name, experiment_coordinates=coordinates,
        )

    runners = {"old": old, "new": new}
    for iteration in range(warmup):
        for name in (("old", "new") if iteration % 2 == 0 else ("new", "old")):
            with selector_context(name):
                result = runners[name]()
                del result
    pairs = []
    reference_hash = None
    for iteration in range(repeats):
        order = ("old", "new") if iteration % 2 == 0 else ("new", "old")
        outputs, samples = {}, {"order": list(order)}
        for name in order:
            with selector_context(name):
                started = time.perf_counter()
                result = runners[name]()
                complete = time.perf_counter() - started
            outputs[name] = result
            samples[name] = {
                "reported_seconds": result.timing.total_seconds,
                "complete_call_seconds": complete,
                "timing": result.timing.to_dict(),
            }
        for field in ("neighbor_ids", "distances", "probe_counts", "ndis"):
            if not np.array_equal(getattr(outputs["old"], field), getattr(outputs["new"], field)):
                raise ValueError(f"old/new {field} differ at repeat {iteration + 1}")
        current_hash = hashlib.sha256(outputs["old"].neighbor_ids.tobytes()).hexdigest()
        if reference_hash is not None and current_hash != reference_hash:
            raise ValueError("neighbor results changed between repeats")
        reference_hash = current_hash
        if iteration == 0:
            quality = {
                "recall_at_k": compute_recall_at_k(result.neighbor_ids, truth, k=query.topk),
                "mean_ndis": float(result.ndis.mean()),
                "mean_probe_count": float(result.probe_counts.mean()),
                "max_probe_count": int(result.probe_counts.max()),
                "neighbor_ids_sha256": current_hash,
            }
        pairs.append(samples)
        print(json.dumps({"stage": "pair_complete", "point": coordinates["point"],
                          "repeat": iteration + 1, **{name: {
                              "reported_ms_per_query": samples[name]["reported_seconds"] * 1000 / len(query.queries),
                              "complete_ms_per_query": samples[name]["complete_call_seconds"] * 1000 / len(query.queries),
                          } for name in order}}), flush=True)
        del outputs, result
    return {"coordinates": {**coordinates, **query.coordinates}, **quality,
            "baseline": baseline,
            "baseline_query_coordinates": baseline_query.coordinates if baseline_query is not None else None,
            **({"candidate_scratch_bits": {"old": 64, "new": 32}} if selectors else {}),
            "workspace_logits_probabilities_and_plans_identical": True,
            "workspace_allocated_bytes_after_query": query.workspace.nbytes,
            "all_results_identical": True, "pairs": pairs,
            **summarize_pairs(pairs, len(query.queries))}


def write_summary(output: Path, payload: dict) -> None:
    write_json_atomic(output / "comparison.json", payload)
    lines = [f"# {payload['dataset']} old/new prefix query", "",
             f"Status: {payload['status']}; sampled queries: {payload['query_count']}; "
             f"K={payload['topk']}; threads={payload['threads']}; "
             f"router batch={payload['router_batch_size']}; repeats={payload['repeats']}.", "",
             {"legacy": "Reported compares the historical timer with the query-only timer.",
              "compact_unbuffered": "Reported compares the same compact query timer with buffers disabled/enabled.",
              "compact_buffered": "Both use the compact query timer: reference A1/A2 vs the selected fused planner backend.",
              "compact_d": "Both use the compact query timer: reference D-only vs the selected fused planner backend.",
              "compact_comparison_order": "Both use the same D/B query path: reference logit comparisons vs cached float-order keys. Assembly/coarse optimizations are shared.",
              "scratch_int64": "Both chains reuse A1/A2 buffers; compare int64/int32 selector scratch with int64 outputs. Selector binding is changed/restored outside both timers."}[payload.get("baseline", "legacy")],
             "Baseline: " + payload.get("baseline", "legacy") + ". "
             "Complete compares full function calls including NDis and result construction in both chains. "
             "All query/index loading, preparation, verification and saving are excluded. "
             "Orders alternate old/new and new/old; medians are in ms/query.", "",
             "| Point | M' | Recall | Mean NDis | Old reported | New query | Speedup | "
             "Old complete | New complete | Complete speedup | New wins |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, point in payload["points"].items():
        reported, complete = point["reported_seconds"], point["complete_call_seconds"]
        lines.append(f"| {name} | {point['coordinates']['candidate_bound']} | "
                     f"{point['recall_at_k']:.6f} | {point['mean_ndis']:.1f} | "
                     f"{reported['old_median_ms_per_query']:.6f} | {reported['new_median_ms_per_query']:.6f} | "
                     f"{reported['speedup_ratio_of_medians']:.3f}x | "
                     f"{complete['old_median_ms_per_query']:.6f} | {complete['new_median_ms_per_query']:.6f} | "
                     f"{complete['speedup_ratio_of_medians']:.3f}x | "
                     f"{complete['new_faster_pair_count']}/{payload['repeats']} |")
    lines += ["", "Sampled query performance is not a complete official-query run. "
              "Target labels are frozen policy targets, not guaranteed measured Recall."]
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--baseline", choices=("legacy", "compact_unbuffered", "compact_buffered", "compact_d", "compact_comparison_order", "scratch_int64"), default="legacy")
    parser.add_argument("--plan-backend", choices=("auto", "baseline", "d", "bd"), default="auto")
    parser.add_argument("--points", nargs="+", default=DEFAULT_POINTS,
                        help="TARGET:ratio_1_2, TARGET:top_256, or TARGET:full_M")
    parser.add_argument("--query-count", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if min(args.query_count, args.repeats, args.threads, args.router_batch_size, args.topk) < 1 or args.warmup < 0:
        parser.error("counts must be positive, warmup must be nonnegative")
    output = args.output_root / args.dataset
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"use a fresh output directory: {output}")
    # Same advisory lock as the repository's other 100M CPU benchmark runners.
    with Path("/tmp/ft_anns_100m_cpu_search.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, output)


def run(args, output: Path) -> None:
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    run_dir = ROOT / "runs" / f"{args.dataset}_gtk100_bucket65536_niter10_pseed42_ramps"
    partition_path = ROOT / "artifacts" / args.dataset / f"{args.dataset}_bucket65536_niter10_pseed42_fixed-partition"
    parameters_path = run_dir / "learned_gain_cost_prefix_prototype/recall_8_targets_policy_100k_fixed_a1_b01_probability/parameters.json"
    parameters_hash = hashlib.sha256(parameters_path.read_bytes()).hexdigest()
    started = time.perf_counter()
    print(json.dumps({"stage": "load_index", "dataset": args.dataset}), flush=True)
    partition = FixedPartition.load(partition_path)
    checkpoint = RouterCheckpoint.load(run_dir / "best", partition=partition)
    checkpoint.model.to("cpu").eval()
    targets, _, gp, cp, _, settings = _validate_parameters(
        _load_json(parameters_path), partition=partition, checkpoint=checkpoint,
    )
    if settings != PROBABILITY_RETRIEVAL_SETTINGS or partition._index.ntotal != 100_000_000:
        raise ValueError("requires a real 100M Probability full-prefix experiment")
    query_file, query_format, truth_file = DATASETS[args.dataset]
    source = FileVectorSource(ROOT / "data" / args.dataset / query_file, query_format, partition.dimension)
    if args.query_count > source.count:
        raise ValueError("requested sample is larger than the official query set")
    row_ids = np.sort(np.random.default_rng(args.seed).choice(source.count, args.query_count, replace=False))
    queries = np.array(source.read_rows(row_ids), dtype=np.float32, copy=True, order="C")
    truth = np.array(read_ivecs(ROOT / "data" / args.dataset / truth_file)[row_ids], copy=True)
    sizes = torch.from_numpy(partition.list_sizes.astype(np.float32) / float(partition.list_sizes.max()))
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "query_row_ids.npy", row_ids)
    source_files = ["src/ft_anns/adaptive_query/prefix_query.py", "src/ft_anns/adaptive_query/retrieval.py",
                    "src/ft_anns/adaptive_query/gain_cost.py", "src/ft_anns/adaptive_query/_gain_cost_native.cpp",
                    "src/ft_anns/routing/model.py", "setup.py",
                    "src/ft_anns/adaptive_query/_prefix_softmax.h",
                    "src/ft_anns/adaptive_query/_prefix_softmax_vector.h",
                    "src/ft_anns/adaptive_query/_prefix_softmax_avx2.cpp",
                    "src/ft_anns/adaptive_query/_prefix_softmax_avx512.cpp",
                    "tools/100m_cpu_gain_cost_search/compare_probability_top_m.py", str(Path(__file__).relative_to(ROOT))]
    payload = {
        "format": "probability-prefix-query-chain-ab-v1", "status": "running",
        "baseline": args.baseline,
        "dataset": args.dataset, "base_count": partition._index.ntotal,
        "bucket_count": partition.bucket_count, "topk": args.topk,
        "query_count": len(queries), "official_query_count": source.count,
        "sample_seed": args.seed, "sample_without_replacement": True,
        "sample_indices_file": "query_row_ids.npy", "queries_sha256": hashlib.sha256(queries.tobytes()).hexdigest(),
        "threads": args.threads, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "router_batch_size": args.router_batch_size, "faiss_parallel_mode": partition._index.parallel_mode,
        "repeats": args.repeats, "warmup_pairs": args.warmup,
        "parameters": str(parameters_path), "parameters_sha256": parameters_hash,
        "index_path": str(partition_path / "index.faiss"),
        "index_bytes": (partition_path / "index.faiss").stat().st_size,
        "setup_seconds": time.perf_counter() - started,
        "environment": {"torch": torch.__version__, "numpy": np.__version__, "faiss": faiss.__version__,
                        "platform": platform.platform(),
                        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source_files},
        "points": {},
    }
    write_summary(output, payload)
    print(json.dumps({"stage": "loaded", "dataset": args.dataset,
                      "setup_seconds": payload["setup_seconds"], "queries": len(queries)}), flush=True)
    workspace = None
    for point in args.points:
        key, variant = point.split(":")
        limit = {"ratio_1_2": PrefixCandidateLimit(ratio="1/2"),
                 "top_256": PrefixCandidateLimit(count=256),
                 "full_M": PrefixCandidateLimit()}[variant]
        policy = targets[key]
        planner = GainCostProbePlanner(normalized_sizes=sizes, policy=policy,
            gain_power=gp, cost_power=cp, settings=settings, prefix_candidate_limit=limit)
        preparation_started = time.perf_counter()
        query = ProbabilityPrefixQuery(queries=queries, checkpoint=checkpoint, partition=partition,
                                      planner=planner, topk=args.topk, router_batch_size=args.router_batch_size,
                                      workspace=(workspace or ProbabilityPrefixWorkspace(
                                          model=checkpoint.model, bucket_count=partition.bucket_count,
                                          batch_size=min(len(queries), args.router_batch_size),
                                          plan_backend="baseline" if args.baseline == "scratch_int64" else args.plan_backend)))
        preparation_seconds = time.perf_counter() - preparation_started
        workspace = query.workspace
        if not query.native_packing:
            raise RuntimeError("rebuild native extension before measuring")
        coordinates = {"dataset": partition.dataset_name, "point": point,
                       "query_filename": query_file, "target_recall": float(key),
                       "alpha": policy["alpha"], "beta": policy["beta"],
                       "probe_plan_implementation": planner.prepared_implementation,
                       **planner.prefix_candidate_coordinates}
        print(json.dumps({"stage": "point_started", "dataset": args.dataset,
                          "point": point, "candidate_bound": planner.candidate_bound}), flush=True)
        payload["points"][point] = benchmark_point(query=query, checkpoint=checkpoint, partition=partition,
            parameters_path=parameters_path, coordinates=coordinates, truth=truth,
            repeats=args.repeats, warmup=args.warmup, baseline=args.baseline)
        payload["points"][point]["query_preparation_seconds"] = preparation_seconds
        write_summary(output, payload)
    if hashlib.sha256(parameters_path.read_bytes()).hexdigest() != parameters_hash:
        raise ValueError("frozen parameters changed during measurement")
    payload["status"] = "complete"
    write_summary(output, payload)
    print(output / "summary.md", flush=True)


if __name__ == "__main__":
    main()
