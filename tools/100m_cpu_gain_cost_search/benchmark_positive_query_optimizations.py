#!/usr/bin/env python3
"""Paired stepwise L2PR+ benchmarks with exact outputs and the same query timer."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import time

import faiss
import numpy as np
import torch

from search_sweep import _load_json, _load_queries, _validate_parameters
from ft_anns.adaptive_query import gain_cost as gc
from ft_anns.adaptive_query.positive_query import (
    L2PRPlusQuery, L2PRPlusWorkspace, l2pr_plus_ndis, query_l2pr_plus,
)
from ft_anns.artifact_io import write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.partition import FixedPartition

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = {
    "original_filter": dict(compact_filter=False, collect_routes=False, reuse_hidden=False),
    "scalar": dict(vectorized_filter=False, collect_routes=False, reuse_hidden=False),
    "simd": dict(collect_routes=False, reuse_hidden=False),
    "collector": dict(reuse_hidden=False),
    "batch_512": dict(reuse_hidden=False, faiss_batch_size=512),
    "batch_1024": dict(reuse_hidden=False, faiss_batch_size=1024),
    "hidden": dict(reuse_hidden=True),
    "hidden_batch_1024": dict(reuse_hidden=True, faiss_batch_size=1024),
}


def benchmark_point(*, queries, checkpoint, partition, policy, powers, topk,
                    batch_size, repeats, warmup, variants=None):
    variants = list(VARIANTS) if variants is None else list(variants)
    sizes = torch.from_numpy(partition.list_sizes.astype(np.float32) / float(partition.list_sizes.max()))
    planner = gc.GainCostProbePlanner(normalized_sizes=sizes, policy=policy,
        gain_power=powers[0], cost_power=powers[1], settings=gc.POSITIVE_MARGINAL_RETRIEVAL_SETTINGS)
    prepared = {}
    for name in variants:
        options = dict(VARIANTS[name])
        faiss_batch = options.pop("faiss_batch_size", None)
        workspace = L2PRPlusWorkspace(model=checkpoint.model, bucket_count=partition.bucket_count,
            batch_size=min(batch_size, len(queries)), **options)
        prepared[name] = L2PRPlusQuery(queries=queries, checkpoint=checkpoint, partition=partition,
            planner=planner, topk=topk, router_batch_size=batch_size, workspace=workspace,
            faiss_batch_size=faiss_batch)
    # One identical retained logits buffer; per-variant arenas and hidden buffers
    # remain independent. No query-dependent data is cached between targets.
    shared_logits = next(iter(prepared.values())).workspace.logits
    for query in prepared.values():
        query.workspace.logits = shared_logits
    with torch.inference_mode():
        for offset in range(0, len(queries), batch_size):
            batch = torch.from_numpy(queries[offset:offset + batch_size])
            expected = checkpoint.model(batch)
            if shared_logits is not None:
                for query in prepared.values():
                    kwargs = {"hidden_out": query.workspace.hidden} if query.workspace.hidden is not None else {}
                    actual = checkpoint.model(batch, out=shared_logits[:len(batch)], **kwargs)
                    if not torch.equal(expected, actual):
                        raise ValueError("Router buffer reuse changes logits")
    reference = query_l2pr_plus(next(iter(prepared.values())))
    reference_ndis = l2pr_plus_ndis(next(iter(prepared.values())), reference)

    def check(query, output):
        for field in ("neighbor_ids", "distances", "routes", "probe_counts"):
            if not np.array_equal(getattr(output, field), getattr(reference, field)):
                raise ValueError(f"optimized {field} differs")
        if output.faiss_ndis is not None and output.faiss_ndis != int(reference_ndis.sum()):
            raise ValueError("FAISS aggregate NDis differs")
        if not np.array_equal(l2pr_plus_ndis(query, output), reference_ndis):
            raise ValueError("per-query NDis differs")

    for iteration in range(warmup):
        for name in variants[iteration % len(variants):] + variants[:iteration % len(variants)]:
            output = query_l2pr_plus(prepared[name])
            check(prepared[name], output)
            del output
    pairs = []
    for iteration in range(repeats):
        shift = iteration % len(variants)
        order = variants[shift:] + variants[:shift]
        row = {"order": order}
        for name in order:
            started = time.perf_counter()
            output = query_l2pr_plus(prepared[name])
            row[name] = time.perf_counter() - started
            check(prepared[name], output)
            del output
        pairs.append(row)
        print(json.dumps(dict(stage="optimization_pair", repeat=iteration + 1, query_count=len(queries))), flush=True)
    stages = {}
    for name, query in prepared.items():
        query.profile_stages = True
        output = query_l2pr_plus(query)
        check(query, output)
        stages[name] = {field: getattr(output, field) for field in (
            "router_seconds", "kernel_seconds", "assembly_seconds", "packing_seconds", "search_seconds", "faiss_call_count")}
        query.profile_stages = False
        del output
    medians = {name: statistics.median(row[name] for row in pairs) for name in variants}
    return dict(query_count=len(queries), all_results_identical=True, router_logits_identical=True,
        faiss_ndis_matches=True, frozen_policy=dict(policy), powers=list(powers),
        mean_probe_count=float(reference.probe_counts.mean()), max_probe_count=int(reference.probe_counts.max()),
        mean_ndis=float(reference_ndis.mean()), neighbor_ids_sha256=hashlib.sha256(reference.neighbor_ids.tobytes()).hexdigest(),
        median_query_seconds=medians, pairs=pairs, diagnostic_stages=stages,
        query_coordinates={name: query.coordinates for name, query in prepared.items()},
        resident_workspace_bytes={name: query.workspace.nbytes for name, query in prepared.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("deep100m", "sift100m", "spacev100m"), required=True)
    parser.add_argument("--query-limit", type=int, default=2048)
    parser.add_argument("--targets", nargs="+", default=["0.95", "0.99"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.query_limit, args.topk, args.threads, args.router_batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error("use a fresh output file")
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location("optimization_workflow", ROOT / "tools/100m_round_robin_retest/workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    with open("/tmp/ft_anns_100m_cpu_search.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = workflow.dataset_inputs(args.dataset)
        print(json.dumps(dict(stage="loading", dataset=args.dataset)), flush=True)
        partition = FixedPartition.load(info["partition"])
        checkpoint = RouterCheckpoint.load(info["checkpoint"], partition=partition)
        checkpoint.model.to("cpu").eval()
        targets, _, gp, cp, _, _ = _validate_parameters(_load_json(Path(info["parameters"])), partition=partition, checkpoint=checkpoint)
        queries = _load_queries(Path(info["queries"]), info["query_format"], partition.dimension, args.query_limit)
        sources = [ROOT / "src/ft_anns/adaptive_query" / name for name in (
            "positive_query.py", "_positive_marginal.h", "_positive_scan.h", "_gain_cost_native.cpp", "gain_cost.py")]
        sources += [ROOT / "src/ft_anns/routing/model.py", Path(__file__)]
        sources.append(Path(gc._gain_cost_native.__file__))
        report = dict(dataset=args.dataset, query_count=len(queries), full_official_queries=len(queries) == info["query_count"],
            topk=args.topk, threads=args.threads, router_batch_size=args.router_batch_size,
            cpu_affinity=sorted(os.sched_getaffinity(0)), repeats=args.repeats, warmup=args.warmup,
            torch_version=torch.__version__, faiss_version=faiss.__version__,
            timing_scope="positive-marginal-query-v1; per-query audits and equality checks outside timer",
            source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, points={})
        for target in args.targets:
            print(json.dumps(dict(stage="target", dataset=args.dataset, target=target)), flush=True)
            report["points"][target] = benchmark_point(queries=queries, checkpoint=checkpoint, partition=partition,
                policy=targets[target], powers=(gp, cp), topk=args.topk, batch_size=args.router_batch_size,
                repeats=args.repeats, warmup=args.warmup, variants=args.variants)
            write_json_atomic(args.output, report)
        report["status"] = "complete"
        write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
