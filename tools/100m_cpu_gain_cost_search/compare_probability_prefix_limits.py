#!/usr/bin/env python3
"""Compare L2PR ratio limits (minimum 256 candidates) and Top-N with full-M L2PR."""

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

from compare_probability_top_m import DATASETS, ROOT
from search_sweep import _load_json, _load_queries, _target_key, _validate_parameters
from ft_anns.adaptive_query.evaluation import evaluate_results
from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner, PrefixCandidateLimit, PROBABILITY_RETRIEVAL_SETTINGS,
    PROBABILITY_PREFIX_CANDIDATE_MINIMUM,
)
from ft_anns.adaptive_query.prefix_query import (
    PREFIX_QUERY_TIMING_SCOPE, ProbabilityPrefixQuery, query_probability_prefix,
)
from ft_anns.adaptive_query.retrieval import retrieve_probability_prefix
from ft_anns.artifact_io import write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.datasets.base_gt import read_ivecs
from ft_anns.partition import FixedPartition


def variants(ratios: list[str], counts: list[int]) -> dict[str, PrefixCandidateLimit]:
    """Always include the original policy; canonicalize exact ratio spellings."""
    result = {"full_M": PrefixCandidateLimit()}
    for value in ratios:
        limit = PrefixCandidateLimit(ratio=value)
        if limit.ratio == "1":
            continue
        name = "ratio_" + str(limit.ratio).replace("/", "_")
        if name in result:
            raise ValueError(f"duplicate candidate ratio: {value}")
        result[name] = limit
    for value in counts:
        limit = PrefixCandidateLimit(count=value)
        name = f"top_{value}"
        if name in result:
            raise ValueError(f"duplicate candidate count: {value}")
        result[name] = limit
    return result


def compare_results(reference, result, candidate_bound: int) -> dict:
    """Compare actual outputs, outside all retrieval timers."""
    over_bound = reference.probe_counts > candidate_bound
    same_counts = reference.probe_counts == result.probe_counts
    if not bool(same_counts[~over_bound].all()):
        raise ValueError("a retained full-M optimum changed under prefix truncation")
    return {
        "baseline_optimum_exceeds_limit_count": int(over_bound.sum()),
        "baseline_optimum_exceeds_limit_fraction": float(over_bound.mean()),
        "same_probe_count_fraction": float(same_counts.mean()),
        "at_candidate_limit_fraction": float((result.probe_counts == candidate_bound).mean()),
        "different_neighbor_id_query_count": int(np.any(
            reference.neighbor_ids != result.neighbor_ids, axis=1
        ).sum()),
        "distance_arrays_equal": bool(np.array_equal(reference.distances, result.distances)),
        "ndis_arrays_equal": bool(np.array_equal(reference.ndis, result.ndis)),
    }


def write_summary(output: Path, summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "comparison.json", summary)
    k = str(summary["topk"])
    lines = [
        "# L2PR hard prefix candidate limits", "",
        f"Dataset: {summary['dataset']}; K={k}; status={summary['status']}; "
        f"queries={summary['query_count']}; full official queries={summary['full_official_queries']}.",
        "Frozen Router, alpha/beta and global softmax are reused. Ratios use "
        f"min(M,max({summary['ratio_candidate_minimum']},ceil(r*M))); "
        "Top-N uses min(N,M). There is no tail certificate or fallback.",
        "CPU candidates use std::nth_element followed by std::sort on the selected candidates only.",
        f"Warmup: {summary['warmup']}; repeats: {summary['repeats']}. "
        "Times are medians of complete query calls; order rotates each repeat. "
        "Query loading/setup, NDis/Recall reporting and saving are outside the timer.",
        f"Timing scope: {summary['query_timing_scope']}.", "",
        "| Target | Variant | M | Effective M | Baseline optimum exceeds limit | "
        f"Recall@{k} | Recall delta (pp) | Mean NDis | ms/query | Speedup | Planner ms/query |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in summary["targets"].items():
        for name, point in entry["variants"].items():
            diag = point["comparison_to_full_M"]
            lines.append(
                f"| {key} | {name} | {entry['trained_candidate_bound']} | "
                f"{point['coordinates']['candidate_bound']} | "
                f"{diag['baseline_optimum_exceeds_limit_fraction']:.2%} | "
                f"{point['recall_at_k'][k]:.6f} | {point['recall_delta_percentage_points']:+.4f} | "
                f"{point['mean_ndis']:.1f} | {point['median_latency_ms_per_query']:.6f} | "
                f"{point['speedup_vs_full_M']:.3f}x | "
                f"{point['median_timing_seconds']['probe_plan_seconds'] * 1000 / summary['query_count']:.6f} |"
            )
    lines += ["", "Positive Recall deltas mean higher Recall. Speedup above 1 means faster.",
              "A baseline optimum within the new limit keeps its original selected prefix. "
              "Otherwise the best shorter prefix is recomputed; it is not necessarily min(old width, limit).",
              "Query-limited runs are smoke tests, not evidence of full-workload quality or speed."]
    (output / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--bucket-count", type=int, default=65536)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--ratios", nargs="+", default=["1", "2/3", "1/2", "1/3"])
    parser.add_argument("--candidate-limits", nargs="+", type=int, default=[])
    parser.add_argument("--targets", nargs="+", type=float)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-limit", type=int, help="Smoke test using the first N queries.")
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if min(args.bucket_count, args.topk, args.repeats, args.threads, args.router_batch_size) < 1:
        parser.error("counts must be positive")
    if args.query_limit is not None and args.query_limit < 1:
        parser.error("query-limit must be positive")
    try:
        selected_variants = variants(args.ratios, args.candidate_limits)
    except ValueError as error:
        parser.error(str(error))
    output = args.output_root / f"bucket{args.bucket_count}/k{args.topk}" / args.dataset
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"comparison output already contains results; use a fresh root: {output}")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    run = ROOT / "runs" / f"{args.dataset}_gtk100_bucket{args.bucket_count}_niter10_pseed42_ramps"
    parameters_path = run / (
        "learned_gain_cost_prefix_prototype/"
        "recall_8_targets_policy_100k_fixed_a1_b01_probability/parameters.json"
    )
    parameters_digest = hashlib.sha256(parameters_path.read_bytes()).hexdigest()
    parameters = _load_json(parameters_path)
    setup_started = time.perf_counter()
    print(json.dumps({"dataset": args.dataset, "stage": "load_partition"}), flush=True)
    partition = FixedPartition.load(ROOT / "artifacts" / args.dataset /
        f"{args.dataset}_bucket{args.bucket_count}_niter10_pseed42_fixed-partition")
    checkpoint = RouterCheckpoint.load(run / "best", partition=partition)
    checkpoint.validate_for(partition)
    checkpoint.model.to("cpu").eval()
    targets, keys, gp, cp, training_count, settings = _validate_parameters(
        parameters, partition=partition, checkpoint=checkpoint,
    )
    if settings != PROBABILITY_RETRIEVAL_SETTINGS or (gp, cp) != (1, 1):
        raise ValueError("comparison requires frozen Probability full-prefix powers (1,1)")
    if args.targets is not None:
        requested = tuple(_target_key(value) for value in args.targets)
        if len(set(requested)) != len(requested) or set(requested) - set(keys):
            raise ValueError("targets must be unique existing policy keys")
        keys = requested
    query_file, query_format, truth_file = DATASETS[args.dataset]
    queries = _load_queries(ROOT / "data" / args.dataset / query_file, query_format,
                            partition.dimension, args.query_limit)
    truth = read_ivecs(ROOT / "data" / args.dataset / truth_file)[:len(queries)]
    if args.topk > truth.shape[1]:
        raise ValueError("topk must fit the official ground-truth prefix")
    sizes = torch.from_numpy(partition.list_sizes.astype(np.float32) / float(partition.list_sizes.max()))
    summary = {
        "format": "probability-prefix-candidate-limits-v1", "status": "running",
        "dataset": args.dataset, "bucket_count": args.bucket_count, "topk": args.topk,
        "query_count": len(queries), "full_official_queries": args.query_limit is None,
        "repeats": args.repeats, "warmup": args.warmup, "threads": args.threads,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "router_batch_size": args.router_batch_size,
        "setup_seconds": time.perf_counter() - setup_started,
        "parameters": str(parameters_path), "parameters_sha256": parameters_digest,
        "gain_power": gp, "cost_power": cp, "tail_fallback": False,
        "ratio_candidate_minimum": PROBABILITY_PREFIX_CANDIDATE_MINIMUM,
        "candidate_selection_implementation": "std_nth_element_then_sort",
        "query_timing_scope": PREFIX_QUERY_TIMING_SCOPE,
        "test_queries_used_for_fitting": False, "target_keys": list(keys),
        "variant_names": list(selected_variants), "targets": {},
    }
    write_summary(output, summary)
    recall_ks = tuple(sorted({args.topk, *(k for k in (10, 50, 100) if k <= args.topk)}))
    workspace = None
    for key in keys:
        policy = targets[key]
        planners = {name: GainCostProbePlanner(
            normalized_sizes=sizes, policy=policy, gain_power=gp, cost_power=cp,
            settings=settings, prefix_candidate_limit=limit,
        ) for name, limit in selected_variants.items()}
        if any(p.prepared_implementation != "fused_probability_cpu_float32_prepared"
               for p in planners.values()):
            raise RuntimeError("comparison requires the rebuilt Probability extension with sorted_top_m_cpu")
        coordinates = {name: {
            "dataset": partition.dataset_name, "query_mode": "adaptive",
            "ranking": "gain_cost_prefix", "policy_id": f"gain-cost-target-{key}",
            "target_recall": float(key), "policy_training_query_count": training_count,
            "gain_mode": "probability", "gain_power": gp, "cost_power": cp,
            "alpha": policy["alpha"], "beta": policy["beta"],
            "probe_plan_method": "full_prefix", "route_order": "stable_logit",
            "probe_plan_implementation": planner.prepared_implementation,
            "policy_training_semantics": "full_prefix", "probability_retrieval_method": "strict_prefix",
            "candidate_bound_applied": True, "tail_fallback": False,
            "query_filename": query_file, "candidate_variant": name,
            **planner.prefix_candidate_coordinates,
        } for name, planner in planners.items()}
        prepared_queries = {}
        for name, planner in planners.items():
            query = ProbabilityPrefixQuery(
                queries=queries, checkpoint=checkpoint, partition=partition, planner=planner,
                topk=args.topk, router_batch_size=args.router_batch_size, workspace=workspace,
            )
            prepared_queries[name] = query
            workspace = query.workspace
        for name, query in prepared_queries.items():
            coordinates[name].update(query.coordinates)
        if args.warmup:
            for name, planner in planners.items():
                warmup_query = ProbabilityPrefixQuery(
                    queries=queries[:args.router_batch_size], checkpoint=checkpoint,
                    partition=partition, planner=planner, topk=args.topk,
                    router_batch_size=args.router_batch_size, workspace=workspace,
                )
                query_probability_prefix(warmup_query)
        measured = {name: [] for name in planners}
        comparisons = {}
        reference = None
        names = list(planners)
        for repeat in range(args.repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                result = retrieve_probability_prefix(
                    prepared_queries[name], checkpoint_filename=checkpoint.filename,
                    partition_filename=partition.filename, query_filename=query_file,
                    parameters_filename=parameters_path.name,
                    experiment_coordinates=coordinates[name],
                )
                stem = output / f"repeat_{repeat + 1:02d}" / name
                filename = f"gain_cost_target_{key.replace('.', '_')}"
                result.save(stem / "results" / f"{filename}.npz")
                report = evaluate_results(retrieval_result=result, ground_truth=truth,
                    recall_ks=recall_ks, ground_truth_filename=truth_file)
                path = stem / "reports" / f"{filename}.json"
                report.save(path)
                payload = _load_json(path)
                measured[name].append(payload)
                if repeat == 0:
                    if name == "full_M":
                        reference = result
                    comparisons[name] = compare_results(reference, result, planners[name].candidate_bound)
                    comparisons[name]["probe_count_percentiles"] = dict(zip(
                        ("p50", "p90", "p95", "p99", "max"),
                        np.percentile(result.probe_counts, [50, 90, 95, 99, 100]).tolist(), strict=True,
                    ))
                print(json.dumps({"dataset": args.dataset, "target": key, "repeat": repeat + 1,
                    "variant": name, "effective_M": planners[name].candidate_bound,
                    "recall": payload["recall_at_k"][str(args.topk)],
                    "mean_ndis": payload["mean_ndis"],
                    "ms_per_query": payload["mean_latency_seconds"] * 1000}), flush=True)
        points = {}
        for name, reports in measured.items():
            timing = reports[0]["retrieval_timing"]
            points[name] = {
                "coordinates": coordinates[name],
                "recall_at_k": {k: float(np.mean([p["recall_at_k"][k] for p in reports]))
                                for k in reports[0]["recall_at_k"]},
                "mean_ndis": float(np.mean([p["mean_ndis"] for p in reports])),
                "probe_counts": reports[0]["probe_counts"],
                "comparison_to_full_M": comparisons[name],
                "latency_ms_per_query_samples": [p["mean_latency_seconds"] * 1000 for p in reports],
                "median_latency_ms_per_query": float(np.median([p["mean_latency_seconds"] * 1000 for p in reports])),
                "median_timing_seconds": {field: float(np.median([p["retrieval_timing"][field] for p in reports]))
                                          for field in timing if field.endswith("_seconds")},
                "recall_stable_across_repeats": all(p["recall_at_k"] == reports[0]["recall_at_k"] for p in reports),
            }
        for point in points.values():
            point["recall_delta_percentage_points"] = 100 * (
                point["recall_at_k"][str(args.topk)] - points["full_M"]["recall_at_k"][str(args.topk)])
            point["speedup_vs_full_M"] = (
                points["full_M"]["median_latency_ms_per_query"] / point["median_latency_ms_per_query"])
        summary["targets"][key] = {"trained_candidate_bound": int(policy["candidate_bound"]), "variants": points}
        write_summary(output, summary)
    if hashlib.sha256(parameters_path.read_bytes()).hexdigest() != parameters_digest:
        raise ValueError("frozen parameters changed during comparison")
    summary["status"] = "complete"
    write_summary(output, summary)
    print(output / "summary.md", flush=True)


if __name__ == "__main__":
    main()
