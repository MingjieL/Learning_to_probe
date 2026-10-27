"""Run a tiny file-backed training and retrieval workflow without downloads."""

import argparse
import json
from pathlib import Path
import tempfile

import faiss
import numpy as np
import torch

from ft_anns.adaptive_query import evaluate_results, retrieve_fixed
from ft_anns.adaptive_query.gain_cost import (
    GainCostProbePlanner, POSITIVE_MARGINAL_RETRIEVAL_SETTINGS,
    PROBABILITY_RETRIEVAL_SETTINGS, _gain_cost_native,
)
from ft_anns.adaptive_query.positive_query import L2PRPlusQuery, query_l2pr_plus
from ft_anns.adaptive_query.prefix_query import ProbabilityPrefixQuery, query_probability_prefix
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.config import BaseGroundTruthConfig, PartitionConfig, TrainConfig, WorkloadConfig
from ft_anns.datasets import build_base_ground_truth, prepare_workloads
from ft_anns.partition import FixedPartition
from ft_anns.routing import load_checkpoint_validation_workload, load_query_level_workload, train_router


def run(root: Path) -> dict:
    torch.set_num_threads(2)
    faiss.omp_set_num_threads(2)
    torch.manual_seed(42)
    base = np.random.default_rng(42).normal(size=(256, 96)).astype('<f4')
    base_path = root / 'base.fbin'
    with base_path.open('wb') as stream:
        stream.write(np.array(base.shape, dtype='<i4').tobytes())
        stream.write(base.tobytes())
    truth = build_base_ground_truth(base=None, query_ids=None, config=BaseGroundTruthConfig(
        dataset='Deep1M', base_path=str(base_path), format='fbin', dimension=96,
        metric='l2', k=4, query_batch_size=64, output_path=str(root / 'deep1m_gtk4_base-gt'),
    ))
    manifest = prepare_workloads(WorkloadConfig(
        dataset='Deep1M', base_path=str(base_path), base_filename=base_path.name,
        roles={'router_train': {'count': 48}, 'checkpoint_validation': {'count': 16},
               'policy_cost_fit': {'count': 8}, 'policy_tuning': {'count': 16},
               'policy_certification': {'count': 16}},
        sampling={'strategy': 'range', 'start': 0},
        ground_truth={'k': 4, 'metric': 'l2', 'artifact_path': str(truth.path)},
        manifest_output_path=str(root / 'deep1m_gtk4_workloads.json'),
        materialized_ground_truth_output_path=str(root / 'deep1m_gtk4_query-gt'),
    ))
    partition = FixedPartition.build(base, PartitionConfig(
        bucket_count=4, kmeans_niter=2, build_seed=42), dataset_name='Deep1M', metric='l2')
    partition_path = root / 'deep1m_bucket4_niter2_pseed42_fixed-partition'
    partition.save(partition_path)
    partition = FixedPartition.load(partition_path)
    train = load_query_level_workload(manifest, role='router_train', partition=partition)
    validation = load_checkpoint_validation_workload(manifest, partition=partition)
    checkpoint = train_router(config=TrainConfig(
        dataset='Deep1M', workload_manifest_path=str(manifest.path),
        partition={'artifact_path': str(partition_path), 'bucket_count': 4,
                   'kmeans_niter': 2, 'build_seed': 42},
        router={'hidden_dimensions': [16], 'activation': 'relu'}, loss='ramps',
        ramps={'rank_weight': 'log'}, batch_size=16, epochs=2,
        optimizer={'name': 'adam', 'learning_rate': .01},
        validation={'fixed_bucket_count': 4, 'ann_recall_ks': [1, 4],
                    'checkpoint_selection_recall_k': 4},
        seed=42, device='cpu', output_dir=str(root / 'run'),
    ), train_workload=train, validation_workload=validation, partition=partition)
    checkpoint = RouterCheckpoint.load(checkpoint.path, partition=partition)
    query_ids = np.asarray(manifest.role_ids['checkpoint_validation'], dtype=np.int64)
    queries = base[query_ids]
    expected_ids = np.asarray(truth.neighbor_ids[query_ids, :4])
    fixed = retrieve_fixed(queries=queries, checkpoint=checkpoint, partition=partition,
                           topm=4, topk=4, batch_size=8)
    report = evaluate_results(retrieval_result=fixed, ground_truth=expected_ids, recall_ks=(1, 4))
    if report.recall_at_k[4] != 1.0:
        raise AssertionError('Probing every bucket must recover exact nearest neighbors')
    normalized_sizes = torch.from_numpy(partition.list_sizes.astype(np.float32) /
                                        float(partition.list_sizes.max()))
    native_available = _gain_cost_native is not None
    if not native_available:
        raise AssertionError('Native extension was not loaded; reinstall the package')
    adaptive = {}
    # All-bucket coefficients give an independent exact-search oracle. A second
    # coefficient exercises variable-width routing without requiring a trained policy.
    for label, settings, query_type, query_call in (
        ('l2pr', PROBABILITY_RETRIEVAL_SETTINGS, ProbabilityPrefixQuery, query_probability_prefix),
        ('l2pr_plus', POSITIVE_MARGINAL_RETRIEVAL_SETTINGS, L2PRPlusQuery, query_l2pr_plus),
    ):
        widths = {}
        for beta in (0.0, 0.35):
            planner = GainCostProbePlanner(normalized_sizes=normalized_sizes,
                policy={'candidate_bound': 4, 'alpha': 1., 'beta': beta},
                gain_power=1., cost_power=1., settings=settings)
            query = query_type(queries=queries, checkpoint=checkpoint, partition=partition,
                               planner=planner, topk=4, router_batch_size=8)
            output = query_call(query)
            if output.neighbor_ids.shape != (16, 4) or not np.isfinite(output.distances).all():
                raise AssertionError(f'{label}: invalid neighbors or distances')
            if not np.all((output.probe_counts >= 1) & (output.probe_counts <= 4)):
                raise AssertionError(f'{label}: invalid probe counts')
            if beta == 0:
                np.testing.assert_array_equal(output.neighbor_ids, fixed.neighbor_ids)
                np.testing.assert_allclose(output.distances, fixed.distances, rtol=1e-5, atol=1e-5)
            widths[str(beta)] = output.probe_counts.tolist()
        adaptive[label] = widths
    return {'status': 'passed', 'base_vectors': 256, 'dimension': 96, 'buckets': 4,
            'training_queries': 48, 'validation_queries': 16, 'training_epochs': 2,
            'fixed_recall_at_4': report.recall_at_k[4], 'native_extension': native_available,
            'adaptive_probe_counts': adaptive}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='optional JSON result path')
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='learning_to_probe_smoke_') as directory:
        summary = run(Path(directory))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
