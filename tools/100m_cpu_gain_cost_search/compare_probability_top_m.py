#!/usr/bin/env python3
"""Compare frozen Probability adapters on official queries, without fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import faiss
import numpy as np
import torch

from search_sweep import _load_json, _load_queries, _validate_parameters
from ft_anns.adaptive_query.evaluation import evaluate_results
from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner, PROBABILITY_RETRIEVAL_SETTINGS,
    POSITIVE_MARGINAL_RETRIEVAL_SETTINGS, TOP_M_POSITIVE_MARGINAL_RETRIEVAL_SETTINGS,
)
from ft_anns.adaptive_query.retrieval import (
    _RetrievalIdentity, _build_gain_cost_plan_in_router_batches,
    _retrieve_probe_plan, _run_router,
)
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.datasets.base_gt import read_ivecs
from ft_anns.observability import utc_now_iso
from ft_anns.partition import FixedPartition

ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "deep100m": ("query.public.10K.fbin", "fbin", "deep-100M.test_l2.ivecs"),
    "sift100m": ("bigann_query.bvecs", "bvecs", "sift-100M.test_l2.ivecs"),
    "spacev100m": ("query.30K.i8bin", "i8bin", "spacev-100M.test_l2.ivecs"),
}
METHODS = {
    "full_prefix": PROBABILITY_RETRIEVAL_SETTINGS,
    "positive_marginal_filter": POSITIVE_MARGINAL_RETRIEVAL_SETTINGS,
    "top_m_positive_marginal_filter": TOP_M_POSITIVE_MARGINAL_RETRIEVAL_SETTINGS,
}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def retrieve(queries, checkpoint, partition, planner, parameters_path, coordinates, batch_size, topk):
    started_at, started = utc_now_iso(), time.perf_counter()
    plan, router, planning, kernel, assembly = _build_gain_cost_plan_in_router_batches(
        queries=queries, checkpoint=checkpoint, router_batch_size=batch_size,
        planner=planner, list_sizes=torch.from_numpy(np.array(partition.list_sizes, copy=True)),
    )
    return _retrieve_probe_plan(
        queries=queries, checkpoint=checkpoint, partition=partition, probe_plan=plan,
        topk=topk, router_seconds=router, probe_plan_seconds=planning,
        started=started, started_at=started_at,
        identity=_RetrievalIdentity(parameters_path.name, None, None, coordinates),
        query_filename=coordinates["query_filename"], search_execution="padded",
        router_batch_size=batch_size, faiss_batch_size=None,
        probe_plan_kernel_seconds=kernel, probe_plan_assembly_seconds=assembly,
    )


def diagnose(queries, checkpoint, planner, all_bucket_planner, batch_size):
    """One extra full-workload planning pass, excluded from retrieval timings."""
    totals = {}
    widths = []
    equal_sets = 0
    removed_buckets = 0
    additional_buckets = 0
    for offset in range(0, len(queries), batch_size):
        logits, _ = _run_router(queries[offset:offset + batch_size], checkpoint)
        stages = {}
        selected, counts = planner.build_tensors(logits, stage_seconds=stages)
        for name, value in stages.items():
            totals[name] = totals.get(name, 0) + value
        widths.append(counts.numpy())
        all_selected, all_counts = all_bucket_planner.build_tensors(logits)
        for row, count, all_count in zip(
            range(len(logits)), counts.tolist(), all_counts.tolist(), strict=True
        ):
            chosen = set(selected[row, :count].tolist())
            unrestricted = set(all_selected[row, :all_count].tolist())
            equal_sets += chosen == unrestricted
            removed_buckets += len(unrestricted - chosen)
            additional_buckets += len(chosen - unrestricted)
    counts = np.concatenate(widths)
    return {
        "query_count": len(queries), "separate_diagnostic_pass": True,
        "stage_seconds": {k: v for k, v in totals.items() if k.endswith("_seconds")},
        "empty_selection_count": totals["empty_selection_count"],
        "empty_selection_fraction": totals["empty_selection_count"] / len(queries),
        "same_active_set_as_full_filter_query_count": equal_sets,
        "same_active_set_as_full_filter_fraction": equal_sets / len(queries),
        "mean_removed_buckets_vs_full_filter": removed_buckets / len(queries),
        "mean_additional_buckets_vs_full_filter": additional_buckets / len(queries),
        "probe_count_percentiles": dict(zip(
            ("p50", "p90", "p95", "p99", "max"),
            np.percentile(counts, [50, 90, 95, 99, 100]).tolist(), strict=True,
        )),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--bucket-count", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--query-limit", type=int, help="Smoke test only; not an official full-query run")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if min(args.bucket_count, args.topk, args.repeats, args.threads, args.router_batch_size) < 1:
        parser.error("counts must be positive")
    if args.query_limit is not None and args.query_limit < 1:
        parser.error("query-limit must be positive")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    ds = args.dataset
    run = ROOT / "runs" / f"{ds}_gtk100_bucket{args.bucket_count}_niter10_pseed42_ramps"
    parameters_path = run / "learned_gain_cost_prefix_prototype/recall_8_targets_policy_100k_fixed_a1_b01_probability/parameters.json"
    parameters = _load_json(parameters_path)
    setup_started = time.perf_counter()
    print(json.dumps({"dataset": ds, "stage": "load_partition"}), flush=True)
    partition = FixedPartition.load(ROOT / "artifacts" / ds /
                                    f"{ds}_bucket{args.bucket_count}_niter10_pseed42_fixed-partition")
    checkpoint = RouterCheckpoint.load(run / "best", partition=partition)
    checkpoint.validate_for(partition)
    checkpoint.model.to("cpu").eval()
    targets, keys, gp, cp, training_count, settings = _validate_parameters(
        parameters, partition=partition, checkpoint=checkpoint,
    )
    if settings != PROBABILITY_RETRIEVAL_SETTINGS or (gp, cp) != (1.0, 1.0):
        raise ValueError("comparison requires frozen Probability full-prefix gain/cost powers (1,1)")
    if args.targets is not None:
        if len(set(args.targets)) != len(args.targets) or set(args.targets) - set(keys):
            raise ValueError("targets must be unique existing policy keys")
        keys = tuple(args.targets)
    query_file, query_format, truth_file = DATASETS[ds]
    queries = _load_queries(ROOT / "data" / ds / query_file, query_format,
                            partition.dimension, args.query_limit)
    truth = read_ivecs(ROOT / "data" / ds / truth_file)[:len(queries)]
    if args.topk > min(100, truth.shape[1]):
        raise ValueError("topk must fit the official ground-truth prefix")
    sizes = torch.from_numpy(partition.list_sizes.astype(np.float32) /
                             float(partition.list_sizes.max()))
    setup_seconds = time.perf_counter() - setup_started
    print(json.dumps({"dataset": ds, "stage": "inputs_ready", "query_count": len(queries),
                      "setup_seconds": setup_seconds}), flush=True)
    output = args.output_root / f"bucket{args.bucket_count}" / f"k{args.topk}" / ds
    summary = {
        "format": "probability-top-m-comparison-v1", "status": "running",
        "target_keys": list(keys),
        "dataset": ds, "bucket_count": args.bucket_count, "query_count": len(queries),
        "full_official_queries": args.query_limit is None, "topk": args.topk,
        "repeats": args.repeats, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "router_batch_size": args.router_batch_size, "setup_seconds": setup_seconds,
        "parameters": str(parameters_path.relative_to(ROOT)),
        "parameters_sha256": hashlib.sha256(parameters_path.read_bytes()).hexdigest(),
        "gain_power": gp, "cost_power": cp, "policy_training_semantics": "full_prefix",
        "test_queries_used_for_fitting": False, "targets": {},
    }
    recall_ks = tuple(k for k in (10, 50, 100) if k <= args.topk)
    if args.topk not in recall_ks:
        recall_ks += (args.topk,)
    for key in keys:
        policy = targets[key]
        planners = {name: GainCostProbePlanner(normalized_sizes=sizes, policy=policy,
                     gain_power=gp, cost_power=cp, settings=adapter)
                    for name, adapter in METHODS.items()}
        if any(p.implementation == "torch" for p in planners.values()):
            raise RuntimeError("formal comparison requires the rebuilt native extension for all three methods")
        coordinates = {}
        for name, planner in planners.items():
            coordinates[name] = {
                "dataset": partition.dataset_name, "query_mode": "adaptive",
                "ranking": "gain_cost_prefix" if name == "full_prefix" else f"gain_cost_{name}",
                "fixed_bucket_count": None, "policy_id": f"gain-cost-target-{key}",
                "target_recall": float(key), "policy_training_query_count": training_count,
                "gain_mode": "probability", "gain_power": gp, "cost_power": cp,
                "probe_plan_method": name, "route_order": planner.settings.route_order,
                "probe_plan_implementation": planner.prepared_implementation,
                "policy_training_semantics": "full_prefix",
                "probability_retrieval_method": "strict_prefix" if name == "full_prefix" else name,
                "candidate_bound": int(policy["candidate_bound"]),
                "alpha": policy["alpha"], "beta": policy["beta"],
                "candidate_bound_applied": name != "positive_marginal_filter",
                "candidate_scope": {"full_prefix": "trained_candidate_bound",
                    "positive_marginal_filter": "all_partition_buckets",
                    "top_m_positive_marginal_filter": "trained_top_m_candidates"}[name],
                "query_filename": query_file,
            }
        # Warm all adapters and the search path outside every measured repeat.
        for name, planner in planners.items():
            retrieve(queries[:min(len(queries), args.router_batch_size)], checkpoint,
                     partition, planner, parameters_path, coordinates[name],
                     args.router_batch_size, args.topk)
        diagnostics = diagnose(queries, checkpoint, planners["top_m_positive_marginal_filter"],
                               planners["positive_marginal_filter"], args.router_batch_size)
        measured = {name: [] for name in METHODS}
        for repeat in range(args.repeats):
            # Rotate method order to reduce systematic first/last-run effects.
            order = list(METHODS)
            order = order[repeat % 3:] + order[:repeat % 3]
            for name in order:
                result = retrieve(queries, checkpoint, partition, planners[name], parameters_path,
                                  coordinates[name], args.router_batch_size, args.topk)
                base = output / f"repeat_{repeat + 1:02d}" / name
                result.save(base / "results" / f"gain_cost_target_{key.replace('.', '_')}.npz")
                report = evaluate_results(retrieval_result=result, ground_truth=truth,
                                           recall_ks=recall_ks, ground_truth_filename=truth_file)
                report_path = base / "reports" / f"gain_cost_target_{key.replace('.', '_')}.json"
                report.save(report_path)
                payload = _load_json(report_path)
                payload["probe_count_percentiles"] = dict(zip(
                    ("p50", "p90", "p95", "p99", "max"),
                    np.percentile(result.probe_counts, [50, 90, 95, 99, 100]).tolist(), strict=True,
                ))
                write_json(report_path, payload)
                measured[name].append(payload)
                print(json.dumps({"dataset": ds, "target": key, "repeat": repeat + 1,
                    "method": name, "recall": payload["recall_at_k"][str(args.topk)],
                    "ndis": payload["mean_ndis"], "seconds": payload["total_latency_seconds"]}), flush=True)
        methods = {}
        for name, reports in measured.items():
            samples = [p["mean_latency_seconds"] * 1000 for p in reports]
            methods[name] = {
                "recall_at_k": reports[0]["recall_at_k"], "mean_ndis": reports[0]["mean_ndis"],
                "probe_counts": reports[0]["probe_counts"],
                "probe_count_percentiles": reports[0]["probe_count_percentiles"],
                "latency_ms_per_query_samples": samples,
                "median_latency_ms_per_query": float(np.median(samples)),
                "min_latency_ms_per_query": min(samples), "max_latency_ms_per_query": max(samples),
                "median_timing_seconds": {field: float(np.median([p["retrieval_timing"][field] for p in reports]))
                    for field, value in reports[0]["retrieval_timing"].items() if isinstance(value, float)},
                "recall_stable_across_repeats": all(p["recall_at_k"] == reports[0]["recall_at_k"] for p in reports),
                "coordinates": coordinates[name],
            }
        new = methods["top_m_positive_marginal_filter"]
        summary["targets"][key] = {
            "alpha": policy["alpha"], "beta": policy["beta"], "M": policy["candidate_bound"],
            "diagnostics": diagnostics, "methods": methods,
            "new_vs_baselines": {name: {
                "recall_delta_percentage_points": 100 * (new["recall_at_k"][str(args.topk)] - methods[name]["recall_at_k"][str(args.topk)]),
                "ndis_change_percent": 100 * (new["mean_ndis"] / methods[name]["mean_ndis"] - 1),
                "latency_change_percent": 100 * (new["median_latency_ms_per_query"] / methods[name]["median_latency_ms_per_query"] - 1),
            } for name in ("full_prefix", "positive_marginal_filter")},
        }
        write_json(output / "comparison.json", summary)
    summary["status"] = "complete"
    summary["finished_at"] = utc_now_iso()
    write_json(output / "comparison.json", summary)
    lines = [f"# {ds}: frozen Probability Top-M comparison", "",
             f"Buckets: {args.bucket_count}; queries: {len(queries)}; K: {args.topk}; repeats: {args.repeats}.",
             "Parameters are reused unchanged; official queries are not used for fitting.", "",
             "| Target | M | Method | Recall | Mean NDis | Median ms/query |", "|---|---:|---|---:|---:|---:|"]
    for key, entry in summary["targets"].items():
        for name, point in entry["methods"].items():
            lines.append(f"| {key} | {entry['M']} | {name} | {point['recall_at_k'][str(args.topk)]:.6f} | {point['mean_ndis']:.1f} | {point['median_latency_ms_per_query']:.6f} |")
    lines += ["", "Per-stage diagnostics and repeat timing ranges are recorded in comparison.json.",
              "Diagnostics are a separate full-query pass and are excluded from measured retrieval timing."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
