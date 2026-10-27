#!/usr/bin/env python3
"""Paired L2PR+ query/complete-call checks on frozen real IVF indexes.

Keep query timer scope changes separate from the common complete-call time.
All comparisons include identical per-query NDis and immutable result outputs.
"""

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
from ft_anns.adaptive_query import gain_cost as gc, retrieval
from ft_anns.adaptive_query.positive_query import L2PRPlusQuery, L2PRPlusWorkspace
from ft_anns.artifact_io import write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.observability import utc_now_iso
from ft_anns.partition import FixedPartition

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ("legacy", "compact_original_filter", "compact_u64", "compact_u32")


def benchmark_point(*, queries, checkpoint, partition, policy, powers, topk,
                    batch_size, repeats, warmup):
    """Public benchmark seam also exercised with a small actual FAISS index."""
    sizes = torch.from_numpy(partition.list_sizes.astype(np.float32) / float(partition.list_sizes.max()))
    p = gc.GainCostProbePlanner(normalized_sizes=sizes, policy=policy,
        gain_power=powers[0], cost_power=powers[1], settings=gc.POSITIVE_MARGINAL_RETRIEVAL_SETTINGS)
    common = dict(queries=queries, checkpoint=checkpoint, partition=partition,
                  planner=p, topk=topk, router_batch_size=batch_size)
    prepared = {
        name: L2PRPlusQuery(**common, workspace=L2PRPlusWorkspace(
            model=checkpoint.model, bucket_count=partition.bucket_count,
            batch_size=min(len(queries), batch_size), compact_filter=name != "compact_original_filter",
            scratch_bits=64 if name == "compact_u64" else 32))
        for name in VARIANTS if name != "legacy"
    }
    # All variants retain the same logits allocation. Initialize serial scratch
    # independently, then share only the query-independent Router output buffer.
    shared_logits = prepared["compact_u32"].workspace.logits
    for query in prepared.values():
        query.workspace.logits = shared_logits
    list_sizes = torch.from_numpy(partition.list_sizes.copy())
    identity = retrieval._RetrievalIdentity("parameters.json", None, None, {"dataset": partition.dataset_name})

    def old():
        started_at, started = utc_now_iso(), time.perf_counter()
        plan, router, planning, kernel, assembly = retrieval._build_gain_cost_plan_in_router_batches(
            queries=queries, checkpoint=checkpoint, router_batch_size=batch_size,
            planner=p, list_sizes=list_sizes)
        return retrieval._retrieve_probe_plan(
            queries=queries, checkpoint=checkpoint, partition=partition, probe_plan=plan,
            topk=topk, router_seconds=router, probe_plan_seconds=planning,
            started=started, started_at=started_at, identity=identity,
            query_filename="query.fbin", search_execution="padded",
            router_batch_size=batch_size, faiss_batch_size=None,
            probe_plan_kernel_seconds=kernel, probe_plan_assembly_seconds=assembly)

    def run(name):
        if name == "legacy":
            return old()
        return retrieval.retrieve_l2pr_plus(prepared[name],
            checkpoint_filename=checkpoint.filename, partition_filename=partition.filename,
            query_filename="query.fbin", parameters_filename="parameters.json",
            experiment_coordinates={"dataset": partition.dataset_name})

    # Check actual full/tail batches, including exact Router output equality.
    with torch.inference_mode():
        for offset in range(0, len(queries), batch_size):
            batch = torch.from_numpy(queries[offset:offset + batch_size])
            expected_logits = checkpoint.model(batch)
            actual = checkpoint.model(batch, out=shared_logits[:len(batch)]) if shared_logits is not None else checkpoint.model(batch)
            if not torch.equal(expected_logits, actual):
                raise ValueError("reusing Router output changes logits")
            expected = p.build_tensors(expected_logits)
            for query in prepared.values():
                routes, counts = p.build_tensors(actual, native_workspace=query.workspace.native)
                if not torch.equal(routes, expected[0]) or not torch.equal(counts, expected[1]):
                    raise ValueError("compact filter changes routes/counts")
            del expected_logits, actual, expected, routes, counts
    for iteration in range(warmup):
        for name in VARIANTS[iteration % len(VARIANTS):] + VARIANTS[:iteration % len(VARIANTS)]:
            result = run(name)
            del result
    pairs = []
    quality = None
    for iteration in range(repeats):
        shift = iteration % len(VARIANTS)
        order = VARIANTS[shift:] + VARIANTS[:shift]
        outputs, row = {}, {"order": order}
        for name in order:
            started = time.perf_counter()
            output = run(name)
            complete_seconds = time.perf_counter() - started
            row[name] = dict(reported_seconds=output.timing.total_seconds,
                             complete_call_seconds=complete_seconds)
            outputs[name] = output
        reference = outputs["legacy"]
        for name, output in outputs.items():
            for field in ("neighbor_ids", "distances", "probe_counts", "ndis"):
                if not np.array_equal(getattr(reference, field), getattr(output, field)):
                    raise ValueError(f"{name}: {field} differs")
            recorded = output.experiment_coordinates.get("faiss_ndis_total")
            if recorded is not None and recorded != int(reference.ndis.sum()):
                raise ValueError("FAISS NDis differs from the exact posting-list audit")
        if quality is None:
            quality = dict(mean_ndis=float(reference.ndis.mean()),
                mean_probe_count=float(reference.probe_counts.mean()),
                max_probe_count=int(reference.probe_counts.max()),
                neighbor_ids_sha256=hashlib.sha256(reference.neighbor_ids.tobytes()).hexdigest())
        pairs.append(row)
        print(json.dumps({"stage": "pair_complete", "repeat": iteration + 1,
                          "query_count": len(queries)}), flush=True)
        del outputs, reference, output
    medians = {name: {
        field: statistics.median(row[name][field] for row in pairs)
        for field in ("reported_seconds", "complete_call_seconds")} for name in VARIANTS}
    for name in VARIANTS:
        medians[name]["complete_call_speedup_vs_legacy"] = (
            medians["legacy"]["complete_call_seconds"] / medians[name]["complete_call_seconds"])
    return dict(all_results_identical=True, router_logits_identical=True, faiss_ndis_matches=True,
        query_count=len(queries), query_coordinates={name: query.coordinates for name, query in prepared.items()},
        native_workspace_bytes={name: query.workspace.native.allocated_bytes
            if query.workspace.native is not None else 0 for name, query in prepared.items()},
        pairs=pairs, medians=medians, **quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("deep100m", "sift100m", "spacev100m"), required=True)
    parser.add_argument("--query-limit", type=int, default=2048)
    parser.add_argument("--targets", nargs="+", default=["0.95", "0.99"])
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--router-batch-size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.query_limit, args.topk, args.threads, args.router_batch_size, args.repeats) < 1 or args.warmup < 0:
        parser.error("counts must be positive and warmup nonnegative")
    if args.output.exists():
        parser.error("use a fresh output file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    faiss.omp_set_num_threads(args.threads)
    spec = importlib.util.spec_from_file_location(
        "positive_query_workflow", ROOT / "tools/100m_round_robin_retest/workflow.py")
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    with open("/tmp/ft_anns_100m_cpu_search.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        info = workflow.dataset_inputs(args.dataset)
        print(json.dumps({"stage": "loading", "dataset": args.dataset}), flush=True)
        partition = FixedPartition.load(info["partition"])
        checkpoint = RouterCheckpoint.load(info["checkpoint"], partition=partition)
        checkpoint.model.to("cpu").eval()
        payload = _load_json(Path(info["parameters"]))
        targets, _, gp, cp, _, _ = _validate_parameters(payload, partition=partition, checkpoint=checkpoint)
        queries = _load_queries(Path(info["queries"]), info["query_format"], partition.dimension, args.query_limit)
        report = dict(dataset=args.dataset, query_count=len(queries),
            full_official_queries=len(queries) == info["query_count"], topk=args.topk,
            threads=args.threads, cpu_affinity=sorted(os.sched_getaffinity(0)),
            router_batch_size=args.router_batch_size, repeats=args.repeats, warmup=args.warmup,
            torch_version=torch.__version__, faiss_version=faiss.__version__,
            note="Reported legacy/new scopes differ; use complete_call_seconds for common-scope speedup.",
            source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in [ROOT / "src/ft_anns/adaptive_query" / name
                             for name in ("positive_query.py", "_positive_marginal.h", "retrieval.py", "gain_cost.py")]},
            points={})
        for target in args.targets:
            print(json.dumps({"stage": "target", "target": target}), flush=True)
            report["points"][target] = benchmark_point(
                queries=queries, checkpoint=checkpoint, partition=partition, policy=targets[target],
                powers=(gp, cp), topk=args.topk, batch_size=args.router_batch_size,
                repeats=args.repeats, warmup=args.warmup)
            write_json_atomic(args.output, report)
        report["status"] = "complete"
        write_json_atomic(args.output, report)


if __name__ == "__main__":
    main()
