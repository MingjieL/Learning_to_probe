"""Command-line entry point for FT-ANNS workflows."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np

from ft_anns import __version__

if TYPE_CHECKING:
    from ft_anns.config import CalibrationConfig, PolicyLibraryConfig, TrainConfig
    from ft_anns.datasets import WorkloadManifest
    from ft_anns.partition import FixedPartition


logger = logging.getLogger(__name__)


def _apply_calibration_command_overrides(
    config: CalibrationConfig | PolicyLibraryConfig,
    *,
    output_path: str | None,
    device: str | None,
    retrieval_batch_size: int | None = None,
    search_execution: str | None = None,
) -> CalibrationConfig | PolicyLibraryConfig:
    if (
        output_path is None
        and device is None
        and retrieval_batch_size is None
        and search_execution is None
    ):
        return config
    payload = config.model_dump(mode="json")
    if output_path is not None:
        payload["output_path"] = output_path
    if device is not None:
        payload["device"] = device
    if retrieval_batch_size is not None:
        payload["retrieval_batch_size"] = retrieval_batch_size
    if search_execution is not None:
        payload["search_execution"] = search_execution
    return type(config).model_validate(payload)


def _add_logging_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-file",
        help="Human-readable rotating debug log; defaults beside command output.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )


def _configure_command_logging(
    arguments: argparse.Namespace,
    default_path: str | Path,
    *,
    append: bool = False,
) -> Path:
    from ft_anns.run_logging import configure_run_logging

    path = arguments.log_file or default_path
    if getattr(arguments, "_logging_preconfigured", False):
        return Path(path)
    return configure_run_logging(path, level=arguments.log_level, append=append)


def _configure_faiss_threads(thread_count: int | None) -> int | None:
    """Apply an explicit FAISS OpenMP limit for one search process."""
    if thread_count is None:
        return None
    if thread_count <= 0:
        raise ValueError("--faiss-threads must be positive")

    import faiss

    faiss.omp_set_num_threads(thread_count)
    return int(faiss.omp_get_max_threads())


def _batch_size_or_full(value: str) -> int | str:
    if value == "full":
        return value
    try:
        batch_size = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "batch size must be a positive integer or 'full'"
        ) from error
    if batch_size <= 0:
        raise argparse.ArgumentTypeError(
            "batch size must be a positive integer or 'full'"
        )
    return batch_size


def _retrieval_batch_options(arguments: argparse.Namespace) -> dict[str, int | None]:
    router_batch_size = arguments.router_batch_size
    faiss_batch_size = arguments.faiss_batch_size
    if arguments.batch_size is not None and (
        router_batch_size is not None or faiss_batch_size is not None
    ):
        raise ValueError(
            "--batch-size cannot be combined with --router-batch-size or "
            "--faiss-batch-size"
        )
    return {
        "batch_size": arguments.batch_size,
        "router_batch_size": (
            None if router_batch_size in (None, "full") else router_batch_size
        ),
        "faiss_batch_size": (
            None if faiss_batch_size in (None, "full") else faiss_batch_size
        ),
    }


def _resolve_search_execution(*, requested: str | None, adaptive: bool) -> str:
    if requested is not None:
        return requested
    return "padded" if adaptive else "grouped"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ft-anns",
        description="Reproducible fixed-partition FT-ANNS workflows.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command")
    build_base_gt = subcommands.add_parser(
        "build-base-gt",
        help="Build resumable exact base-to-base ground truth with self matches.",
    )
    build_base_gt.add_argument("--config", required=True)
    build_base_gt.add_argument("--output")
    prepare_workloads = subcommands.add_parser(
        "prepare-workloads",
        help="Prepare reproducible isolated workload roles.",
    )
    prepare_workloads.add_argument("--config", required=True)
    prepare_workloads.add_argument("--output")
    compact_partition = subcommands.add_parser(
        "compact-partition",
        help="Export a partition with an external base, preserving the source artifact.",
    )
    compact_partition.add_argument("--partition", required=True)
    compact_partition.add_argument("--base", required=True)
    compact_partition.add_argument("--output", required=True)
    train = subcommands.add_parser(
        "train",
        help="Train a CE, BCE, or RAMPS router and select its best checkpoint.",
    )
    train.add_argument("--config", required=True)
    train.add_argument("--workloads")
    train.add_argument("--resume")
    train.add_argument("--validation-fixed-buckets-num", type=int)
    train.add_argument("--validation-ann-recall-ks", type=int, nargs="+")
    train.add_argument("--checkpoint-selection-recall-k", type=int)
    train.add_argument("--early-stopping-patience-epochs", type=int)
    calibrate = subcommands.add_parser(
        "calibrate",
        help="Freeze one Budget-NDis adaptive operating point.",
    )
    calibrate.add_argument("--checkpoint", required=True)
    calibrate.add_argument("--partition", required=True)
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--output")
    calibrate.add_argument("--device", choices=("auto", "cpu", "cuda"))
    calibrate_library = subcommands.add_parser(
        "calibrate-library",
        help="Freeze named RAMPS policies for several Fixed-query NDis budgets.",
    )
    calibrate_library.add_argument("--checkpoint", required=True)
    calibrate_library.add_argument("--partition", required=True)
    calibrate_library.add_argument("--config", required=True)
    calibrate_library.add_argument("--output")
    calibrate_library.add_argument("--device", choices=("auto", "cpu", "cuda"))
    certify_library = subcommands.add_parser(
        "certify-library",
        help="Evaluate a frozen Policy Library and write an independent report.",
    )
    certify_library.add_argument("--library", required=True)
    certify_library.add_argument("--checkpoint", required=True)
    certify_library.add_argument("--partition", required=True)
    certify_library.add_argument("--workloads", required=True)
    certify_library.add_argument("--output")
    certify_library.add_argument("--mode", choices=("approximate", "strict"))
    certify_library.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    certify_library.add_argument("--retrieval-batch-size", type=int, default=1_000)
    certify_library.add_argument(
        "--search-execution", choices=("grouped", "padded"), default="grouped"
    )
    benchmark_calibrate_library = subcommands.add_parser(
        "benchmark-calibrate-library",
        help="Run calibrate-library and write comparable machine-readable evidence.",
    )
    benchmark_calibrate_library.add_argument("--checkpoint", required=True)
    benchmark_calibrate_library.add_argument("--partition", required=True)
    benchmark_calibrate_library.add_argument("--config", required=True)
    benchmark_calibrate_library.add_argument("--output")
    benchmark_calibrate_library.add_argument("--report", required=True)
    benchmark_calibrate_library.add_argument(
        "--device", choices=("auto", "cpu", "cuda")
    )
    benchmark_calibrate_library.add_argument("--retrieval-batch-size", type=int)
    benchmark_calibrate_library.add_argument(
        "--search-execution", choices=("grouped", "padded")
    )
    benchmark_lambda_fitting = subcommands.add_parser(
        "benchmark-lambda-fitting",
        help="Compare CPU lambda fitting with its frozen exact reference.",
    )
    benchmark_lambda_fitting.add_argument("--report", required=True)
    benchmark_lambda_fitting.add_argument("--query-count", type=int, default=2048)
    benchmark_lambda_fitting.add_argument("--candidate-bound", type=int, default=256)
    benchmark_lambda_fitting.add_argument("--repeats", type=int, default=5)
    search = subcommands.add_parser(
        "search",
        help="Run ground-truth-free retrieval and cache its results.",
    )
    search.add_argument("--mode", choices=("fixed", "adaptive"), default="fixed")
    search.add_argument(
        "--ranking",
        choices=("raw_logits",),
        help="Bucket ranking for fixed-width search; defaults to raw logits.",
    )
    search.add_argument("--checkpoint", required=True)
    search.add_argument("--partition", required=True)
    search.add_argument("--queries", required=True)
    search.add_argument("--calibration")
    search.add_argument("--policy-library")
    search.add_argument("--policy-id")
    search.add_argument("--certification-report")
    search.add_argument("--require-certification", action="store_true")
    search.add_argument(
        "--query-format",
        choices=("npy", "fbin", "fvecs", "ivecs", "bvecs", "i8bin"),
    )
    search.add_argument("--topm", type=int)
    search.add_argument("--topk", type=int, required=True)
    search.add_argument(
        "--batch-size",
        type=int,
        help="Legacy end-to-end query batch size; mutually exclusive with stage sizes.",
    )
    search.add_argument(
        "--router-batch-size",
        type=_batch_size_or_full,
        help="Router inference batch size: a positive integer or 'full' (default).",
    )
    search.add_argument(
        "--faiss-batch-size",
        type=_batch_size_or_full,
        help="FAISS input batch size: a positive integer or 'full' (default).",
    )
    search.add_argument(
        "--faiss-threads",
        type=int,
        help="Positive FAISS OpenMP thread count for this search process.",
    )
    search.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    search.add_argument(
        "--search-execution",
        choices=("grouped", "padded"),
        help="Defaults to grouped for Fixed Query and padded for Adaptive Query.",
    )
    search.add_argument("--output", required=True)
    search_sweep = subcommands.add_parser(
        "search-sweep",
        help="Run several Fixed widths or Policy Library entries with one load.",
    )
    search_sweep.add_argument("--checkpoint", required=True)
    search_sweep.add_argument("--partition", required=True)
    search_sweep.add_argument("--queries", required=True)
    search_sweep.add_argument(
        "--query-format",
        choices=("npy", "fbin", "fvecs", "ivecs", "bvecs", "i8bin"),
    )
    search_sweep.add_argument("--fixed-topms", type=int, nargs="+")
    search_sweep.add_argument(
        "--ranking",
        choices=("raw_logits",),
        default="raw_logits",
    )
    search_sweep.add_argument("--policy-library")
    search_sweep.add_argument("--policy-ids", nargs="+")
    search_sweep.add_argument("--topk", type=int, required=True)
    search_sweep.add_argument(
        "--batch-size",
        type=int,
        help="Legacy end-to-end query batch size; mutually exclusive with stage sizes.",
    )
    search_sweep.add_argument(
        "--router-batch-size",
        type=_batch_size_or_full,
        help="Router inference batch size: a positive integer or 'full' (default).",
    )
    search_sweep.add_argument(
        "--faiss-batch-size",
        type=_batch_size_or_full,
        help="FAISS input batch size: a positive integer or 'full' (default).",
    )
    search_sweep.add_argument("--faiss-threads", type=int)
    search_sweep.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    search_sweep.add_argument(
        "--search-execution",
        choices=("grouped", "padded"),
        help="Defaults to grouped for Fixed sweeps and padded for Adaptive sweeps.",
    )
    search_sweep.add_argument("--output-dir", required=True)
    search_sweep.add_argument(
        "--resume", action="store_true", help="Reuse compatible result files."
    )
    evaluate = subcommands.add_parser(
        "evaluate",
        help="Evaluate a cached retrieval result independently.",
    )
    evaluate.add_argument("--config", required=True)
    evaluate.add_argument("--results", required=True)
    evaluate.add_argument("--ground-truth")
    evaluate.add_argument("--output")
    report_matrix = subcommands.add_parser(
        "report-matrix",
        help="Validate and summarize a complete dataset experiment matrix.",
        description="Validate and summarize a complete dataset experiment matrix.",
    )
    report_matrix.add_argument("--config", required=True)
    report_matrix.add_argument("--output")
    for command_parser in (
        build_base_gt,
        prepare_workloads,
        compact_partition,
        train,
        calibrate,
        calibrate_library,
        certify_library,
        benchmark_calibrate_library,
        benchmark_lambda_fitting,
        search,
        search_sweep,
        evaluate,
        report_matrix,
    ):
        _add_logging_arguments(command_parser)
    return parser


def _load_vectors(
    path: str | Path, *, dimension: int, format_name: str | None = None
) -> np.ndarray:
    vector_path = Path(path)
    resolved_format = format_name or vector_path.suffix.removeprefix(".")
    if resolved_format == "npy":
        vectors = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    elif resolved_format in ("fbin", "fvecs", "ivecs", "bvecs", "i8bin"):
        from ft_anns.datasets.base_gt import FileVectorSource

        source = FileVectorSource(vector_path, resolved_format, dimension)
        vectors = source.read_slice(0, source.count)
    else:
        raise ValueError(
            "query format could not be inferred; pass --query-format explicitly"
        )
    matrix = np.asarray(vectors)
    if matrix.ndim != 2 or matrix.shape[1] != dimension:
        raise ValueError("query matrix does not match the checkpoint dimension")
    return np.asarray(matrix, dtype=np.float32)


def _load_ground_truth(path: str | Path) -> np.ndarray:
    ground_truth_path = Path(path)
    if ground_truth_path.suffix == ".npy":
        return np.asarray(np.load(ground_truth_path, mmap_mode="r", allow_pickle=False))
    if ground_truth_path.suffix == ".ivecs":
        from ft_anns.datasets.base_gt import read_ivecs

        return read_ivecs(ground_truth_path)
    raise ValueError("ground truth must be a .npy or .ivecs artifact")


def _load_or_build_partition(
    config: TrainConfig, manifest: WorkloadManifest
) -> FixedPartition:
    """Resolve the immutable partition declared by a TrainConfig."""
    from ft_anns.config import TrainConfig
    from ft_anns.datasets import WorkloadManifest
    from ft_anns.datasets.base_gt import FileVectorSource
    from ft_anns.datasets.registry import dataset_spec
    from ft_anns.partition import FixedPartition

    assert isinstance(config, TrainConfig)
    assert isinstance(manifest, WorkloadManifest)
    artifact_path = config.partition.artifact_path
    if artifact_path is None:
        raise ValueError("train requires partition.artifact_path")
    if manifest.metadata.get("dataset_name") != config.dataset:
        raise ValueError("TrainConfig dataset does not match the Workload Manifest")
    resolved_workload = manifest.metadata.get("resolved_config")
    if not isinstance(resolved_workload, dict):
        raise TypeError("Workload Manifest resolved config is invalid")
    specification = dataset_spec(config.dataset)
    source = FileVectorSource(
        Path(str(resolved_workload.get("base_path"))),
        specification.format,
        specification.dimension,
    )
    expected_base_filename = manifest.metadata.get("base_filename")
    if source.filename != expected_base_filename:
        raise ValueError("base source filename no longer matches the manifest")
    if Path(artifact_path).exists():
        partition = FixedPartition.load(artifact_path)
        if partition.dataset_name != config.dataset:
            raise ValueError(
                "loaded Fixed Partition dataset name does not match config"
            )
        if partition.base_filename != expected_base_filename:
            raise ValueError(
                "loaded Fixed Partition base filename does not match the manifest"
            )
        if (
            len(partition.bucket_assignment) != source.count
            or partition.dimension != specification.dimension
            or partition.metric != specification.metric
        ):
            raise ValueError("loaded Fixed Partition structure does not match dataset")
        if (
            config.partition.bucket_count is not None
            and partition.bucket_count != config.partition.bucket_count
        ):
            raise ValueError(
                "loaded Fixed Partition bucket count does not match config"
            )
        if partition.build_seed != config.partition.build_seed:
            raise ValueError("loaded Fixed Partition build seed does not match config")
        if partition.kmeans_niter != config.partition.kmeans_niter:
            raise ValueError(
                "loaded Fixed Partition K-Means niter does not match config"
            )
        return partition
    if config.partition.bucket_count is None:
        raise ValueError(
            "partition.bucket_count is required when building a Fixed Partition"
        )
    partition = FixedPartition.build(
        source.read_slice(0, source.count),
        config.partition,
        metric=specification.metric,
        dataset_name=config.dataset,
    )
    partition.save(
        artifact_path,
        base_filename=str(expected_base_filename),
        base_path=Path(str(resolved_workload.get("base_path"))),
    )
    # Avoid keeping the build-time IVF alive while reconstructing the saved one.
    del partition
    return FixedPartition.load(artifact_path)


def _run_command(arguments: argparse.Namespace) -> int:
    if arguments.command is not None and arguments.log_file is not None:
        from ft_anns.run_logging import configure_run_logging

        configure_run_logging(
            arguments.log_file,
            level=arguments.log_level,
            append=(
                arguments.command == "train"
                and getattr(arguments, "resume", None) is not None
            ),
        )
        arguments._logging_preconfigured = True
    if arguments.command == "compact-partition":
        from ft_anns.partition import FixedPartition

        output = Path(arguments.output)
        if output.exists():
            raise FileExistsError(
                f"refusing to overwrite Fixed Partition artifact: {output}"
            )
        if output.name != Path(arguments.partition).name:
            raise ValueError(
                "compact output must preserve the partition basename in a new parent directory"
            )
        partition = FixedPartition.load(arguments.partition, base_path=arguments.base)
        partition.save(output, base_path=arguments.base)
        print(f"Compact partition saved to {output}; source artifact retained")
    elif arguments.command == "build-base-gt":
        from ft_anns.config import BaseGroundTruthConfig, load_config
        from ft_anns.datasets.base_gt import build_base_ground_truth

        loaded = load_config("base_ground_truth", arguments.config)
        assert isinstance(loaded, BaseGroundTruthConfig)
        if arguments.output is not None:
            payload = loaded.model_dump()
            payload["output_path"] = arguments.output
            loaded = BaseGroundTruthConfig.model_validate(payload)
        log_path = _configure_command_logging(
            arguments, Path(loaded.output_path) / "run.log"
        )
        logger.info(
            "build-base-gt started dataset=%s output=%s",
            loaded.dataset,
            loaded.output_path,
        )
        base_gt_artifact = build_base_ground_truth(
            base=None,
            query_ids=None,
            config=loaded,
        )
        print(
            json.dumps(
                {
                    "artifact": str(base_gt_artifact.path),
                    "artifact_filename": base_gt_artifact.metadata["artifact_filename"],
                    "shape": list(base_gt_artifact.shape),
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "prepare-workloads":
        from ft_anns.config import WorkloadConfig, load_config
        from ft_anns.datasets.workloads import prepare_workloads

        loaded = load_config("workload", arguments.config)
        assert isinstance(loaded, WorkloadConfig)
        if arguments.output is not None:
            payload = loaded.model_dump()
            payload["manifest_output_path"] = arguments.output
            loaded = WorkloadConfig.model_validate(payload)
        log_path = _configure_command_logging(
            arguments, Path(loaded.manifest_output_path).with_suffix(".log")
        )
        logger.info(
            "prepare-workloads started dataset=%s output=%s",
            loaded.dataset,
            loaded.manifest_output_path,
        )
        manifest = prepare_workloads(loaded)
        print(
            json.dumps(
                {
                    "manifest": str(manifest.path),
                    "manifest_filename": manifest.filename,
                    "role_counts": {
                        name: int(role["count"])
                        for name, role in manifest.metadata["roles"].items()
                    },
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "train":
        from ft_anns.config import TrainConfig, load_config
        from ft_anns.datasets import WorkloadManifest
        from ft_anns.routing import (
            load_checkpoint_validation_workload,
            load_query_level_workload,
            train_router,
        )

        loaded = load_config("train", arguments.config)
        assert isinstance(loaded, TrainConfig)
        if arguments.workloads is not None:
            payload = loaded.model_dump()
            payload["workload_manifest_path"] = arguments.workloads
            loaded = TrainConfig.model_validate(payload)
        if any(
            value is not None
            for value in (
                arguments.validation_fixed_buckets_num,
                arguments.validation_ann_recall_ks,
                arguments.checkpoint_selection_recall_k,
                arguments.early_stopping_patience_epochs,
            )
        ):
            payload = loaded.model_dump(mode="json")
            validation = dict(payload["validation"])
            if arguments.validation_fixed_buckets_num is not None:
                validation["fixed_bucket_count"] = (
                    arguments.validation_fixed_buckets_num
                )
            if arguments.validation_ann_recall_ks is not None:
                validation["ann_recall_ks"] = arguments.validation_ann_recall_ks
            if arguments.checkpoint_selection_recall_k is not None:
                validation["checkpoint_selection_recall_k"] = (
                    arguments.checkpoint_selection_recall_k
                )
            payload["validation"] = validation
            if arguments.early_stopping_patience_epochs is not None:
                payload["early_stopping"] = {
                    "patience_epochs": arguments.early_stopping_patience_epochs
                }
            loaded = TrainConfig.model_validate(payload)
        log_path = _configure_command_logging(
            arguments,
            Path(loaded.output_dir) / "run.log",
            append=arguments.resume is not None,
        )
        logger.info(
            "train command started dataset=%s output=%s resume=%s",
            loaded.dataset,
            loaded.output_dir,
            arguments.resume,
        )
        manifest = WorkloadManifest.load(loaded.workload_manifest_path)
        partition = _load_or_build_partition(loaded, manifest)
        checkpoint = train_router(
            config=loaded,
            train_workload=load_query_level_workload(
                manifest,
                role="router_train",
                partition=partition,
                file_backed=(
                    loaded.training_data is not None
                    and loaded.training_data.mode == "file_backed"
                ),
            ),
            validation_workload=load_checkpoint_validation_workload(
                manifest, partition=partition
            ),
            partition=partition,
            resume_from=arguments.resume,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(checkpoint.path),
                    "checkpoint_filename": checkpoint.filename,
                    "selection": checkpoint.metadata["selection"],
                    "timestamps": checkpoint.metadata["timestamps"],
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "benchmark-lambda-fitting":
        from ft_anns.adaptive_query.benchmark import (
            write_representative_lambda_fit_benchmark,
        )

        log_path = _configure_command_logging(
            arguments, Path(arguments.report).with_suffix(".log")
        )
        logger.info(
            "lambda-fitting benchmark started queries=%s bound=%s repeats=%s",
            arguments.query_count,
            arguments.candidate_bound,
            arguments.repeats,
        )
        report_path = write_representative_lambda_fit_benchmark(
            report_path=arguments.report,
            query_count=arguments.query_count,
            candidate_bound=arguments.candidate_bound,
            repeats=arguments.repeats,
        )
        print(
            json.dumps(
                {
                    "benchmark_report": str(report_path),
                    "benchmark_kind": "synthetic",
                    "log": str(log_path),
                    "status": "complete",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "calibrate":
        from ft_anns.adaptive_query import (
            calibrate_policy,
            load_calibration_workload,
            load_policy_cost_fit_workload,
        )
        from ft_anns.checkpoint import RouterCheckpoint
        from ft_anns.config import CalibrationConfig, load_config
        from ft_anns.datasets import WorkloadManifest
        from ft_anns.partition import FixedPartition
        from ft_anns.routing import load_checkpoint_validation_workload

        loaded = load_config("calibration", arguments.config)
        assert isinstance(loaded, CalibrationConfig)
        loaded = _apply_calibration_command_overrides(
            loaded,
            output_path=arguments.output,
            device=arguments.device,
        )
        assert isinstance(loaded, CalibrationConfig)
        log_path = _configure_command_logging(
            arguments, Path(loaded.output_path).with_suffix(".log")
        )
        logger.info(
            "calibrate command started checkpoint=%s objective=%s output=%s",
            arguments.checkpoint,
            loaded.objective,
            loaded.output_path,
        )
        partition = FixedPartition.load(arguments.partition)
        checkpoint = RouterCheckpoint.load(arguments.checkpoint, partition=partition)
        checkpoint.require_adaptive_query_support(partition)
        manifest = WorkloadManifest.load(loaded.workload_manifest_path)
        tuning_workload = load_calibration_workload(
            manifest,
            partition=partition,
            role=loaded.tuning_role,
        )
        certification_workload = load_calibration_workload(
            manifest,
            partition=partition,
            role=loaded.certification_role,
        )
        cost_fit_workload = (
            load_policy_cost_fit_workload(manifest, partition=partition)
            if loaded.lambda_search.mode == "auto"
            else None
        )
        calibration_artifact = calibrate_policy(
            checkpoint=checkpoint,
            partition=partition,
            workload=tuning_workload,
            cost_fit_workload=cost_fit_workload,
            certification_workload=certification_workload,
            config=loaded,
            validation_workload=(
                load_checkpoint_validation_workload(manifest, partition=partition)
                if loaded.calibration_bound_selection is not None
                else None
            ),
        )
        print(
            json.dumps(
                {
                    "calibration": loaded.output_path,
                    "calibration_filename": calibration_artifact.filename,
                    "selection": calibration_artifact.payload["selection"],
                    "persistence": calibration_artifact.persistence,
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command in (
        "calibrate-library",
        "benchmark-calibrate-library",
    ):
        from ft_anns.adaptive_query import (
            calibrate_policy_library,
            load_calibration_workload,
            load_policy_cost_fit_workload,
        )
        from ft_anns.adaptive_query.benchmark import (
            write_calibrate_library_benchmark_report,
        )
        from ft_anns.checkpoint import RouterCheckpoint
        from ft_anns.config import PolicyLibraryConfig, load_config
        from ft_anns.datasets import WorkloadManifest
        from ft_anns.observability import utc_now_iso
        from ft_anns.partition import FixedPartition
        from ft_anns.routing import load_checkpoint_validation_workload

        benchmark = arguments.command == "benchmark-calibrate-library"
        benchmark_started_at = utc_now_iso()
        benchmark_started = time.perf_counter()

        loaded = load_config("policy_library", arguments.config)
        assert isinstance(loaded, PolicyLibraryConfig)
        device_override = getattr(arguments, "device", None)
        loaded = _apply_calibration_command_overrides(
            loaded,
            output_path=arguments.output,
            device=device_override,
            retrieval_batch_size=getattr(arguments, "retrieval_batch_size", None),
            search_execution=getattr(arguments, "search_execution", None),
        )
        assert isinstance(loaded, PolicyLibraryConfig)
        log_path = _configure_command_logging(
            arguments,
            (
                Path(arguments.report).with_suffix(".log")
                if benchmark
                else Path(loaded.output_path).with_suffix(".log")
            ),
        )
        logger.info(
            "calibrate-library command started checkpoint=%s budget_mode=%s output=%s",
            arguments.checkpoint,
            "manual" if loaded.policy_budgets is not None else "adaptive_ndis",
            loaded.output_path,
        )
        partition = FixedPartition.load(arguments.partition)
        checkpoint = RouterCheckpoint.load(arguments.checkpoint, partition=partition)
        checkpoint.require_adaptive_query_support(partition)
        manifest = WorkloadManifest.load(loaded.workload_manifest_path)
        cost_fit_workload = load_policy_cost_fit_workload(
            manifest,
            partition=partition,
        )
        tuning_workload = load_calibration_workload(
            manifest,
            partition=partition,
            role=loaded.tuning_role,
        )
        calibrated_library = calibrate_policy_library(
            checkpoint=checkpoint,
            partition=partition,
            validation_workload=load_checkpoint_validation_workload(
                manifest, partition=partition
            ),
            cost_fit_workload=cost_fit_workload,
            tuning_workload=tuning_workload,
            config=loaded,
        )
        benchmark_report_path = None
        if benchmark:
            benchmark_finished_at = utc_now_iso()
            benchmark_wall_seconds = time.perf_counter() - benchmark_started
            benchmark_report_path = write_calibrate_library_benchmark_report(
                report_path=arguments.report,
                config_path=arguments.config,
                checkpoint_path=arguments.checkpoint,
                partition_path=arguments.partition,
                started_at=benchmark_started_at,
                finished_at=benchmark_finished_at,
                wall_seconds=benchmark_wall_seconds,
                config=loaded,
                checkpoint=checkpoint,
                partition=partition,
                manifest=manifest,
                library=calibrated_library,
            )
        print(
            json.dumps(
                {
                    "policy_library": loaded.output_path,
                    "policy_library_filename": calibrated_library.filename,
                    "policy_ids": calibrated_library.policy_ids,
                    "persistence": calibrated_library.persistence,
                    **(
                        {"benchmark_report": str(benchmark_report_path)}
                        if benchmark_report_path is not None
                        else {}
                    ),
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "certify-library":
        from ft_anns.adaptive_query import (
            PolicyLibrary,
            certify_policy_library,
            load_calibration_workload,
        )
        from ft_anns.checkpoint import RouterCheckpoint
        from ft_anns.datasets import WorkloadManifest
        from ft_anns.partition import FixedPartition

        library_path = Path(arguments.library)
        report_path = (
            Path(arguments.output)
            if arguments.output is not None
            else library_path.with_name(f"{library_path.stem}.certification.json")
        )
        log_path = _configure_command_logging(
            arguments, report_path.with_suffix(".log")
        )
        partition = FixedPartition.load(arguments.partition)
        checkpoint = RouterCheckpoint.load(arguments.checkpoint, partition=partition)
        library = PolicyLibrary.load(
            library_path, checkpoint=checkpoint, partition=partition
        )
        manifest = WorkloadManifest.load(arguments.workloads)
        certification_workload = load_calibration_workload(
            manifest, partition=partition, role="policy_certification"
        )
        report = certify_policy_library(
            library=library,
            library_path=library_path,
            checkpoint=checkpoint,
            partition=partition,
            certification_workload=certification_workload,
            output_path=report_path,
            device=arguments.device,
            retrieval_batch_size=arguments.retrieval_batch_size,
            search_execution=arguments.search_execution,
            budget_mode=arguments.mode,
        )
        print(
            json.dumps(
                {
                    "policy_library": str(library_path),
                    "certification_report": str(report_path),
                    "certification_status": report.status,
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "search":
        import torch

        from ft_anns.adaptive_query import (
            CalibrationArtifact,
            PolicyLibrary,
            retrieve_adaptive,
            retrieve_fixed,
        )
        from ft_anns.checkpoint import RouterCheckpoint
        from ft_anns.partition import FixedPartition

        log_path = _configure_command_logging(
            arguments, Path(arguments.output).with_suffix(".log")
        )
        faiss_threads = _configure_faiss_threads(arguments.faiss_threads)
        retrieval_batch_options = _retrieval_batch_options(arguments)
        search_execution = _resolve_search_execution(
            requested=arguments.search_execution,
            adaptive=arguments.mode == "adaptive",
        )
        logger.info(
            "search command started mode=%s queries=%s output=%s faiss_threads=%s",
            arguments.mode,
            arguments.queries,
            arguments.output,
            faiss_threads,
        )
        partition = FixedPartition.load(arguments.partition)
        checkpoint = RouterCheckpoint.load(arguments.checkpoint, partition=partition)
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA retrieval requested but PyTorch reports no GPU")
        checkpoint.model.to(arguments.device)
        queries = _load_vectors(
            arguments.queries,
            dimension=partition.dimension,
            format_name=arguments.query_format,
        )
        if arguments.mode == "fixed":
            if arguments.topm is None:
                raise ValueError("fixed search requires --topm")
            if arguments.calibration is not None:
                raise ValueError("--calibration is only allowed for adaptive search")
            if (
                arguments.policy_library is not None
                or arguments.policy_id is not None
                or arguments.certification_report is not None
                or arguments.require_certification
            ):
                raise ValueError(
                    "Policy Library options are only allowed for adaptive search"
                )
            result = retrieve_fixed(
                queries=queries,
                checkpoint=checkpoint,
                partition=partition,
                topm=arguments.topm,
                topk=arguments.topk,
                query_filename=Path(arguments.queries).name,
                search_execution=search_execution,
                **retrieval_batch_options,
            )
        else:
            if arguments.ranking is not None:
                raise ValueError("--ranking is only allowed for fixed search")
            uses_calibration = arguments.calibration is not None
            uses_library = arguments.policy_library is not None
            if uses_calibration == uses_library:
                raise ValueError(
                    "adaptive search requires exactly one of --calibration or --policy-library"
                )
            if arguments.topm is not None:
                raise ValueError("--topm is only allowed for fixed search")
            runtime_calibration = (
                CalibrationArtifact.load(
                    arguments.calibration,
                    checkpoint=checkpoint,
                    partition=partition,
                )
                if uses_calibration
                else None
            )
            runtime_policy_library = (
                PolicyLibrary.load(
                    arguments.policy_library,
                    checkpoint=checkpoint,
                    partition=partition,
                )
                if uses_library
                else None
            )
            if uses_library and arguments.policy_id is None:
                raise ValueError("--policy-library requires --policy-id")
            if uses_calibration and arguments.policy_id is not None:
                raise ValueError("--policy-id is only allowed with --policy-library")
            if arguments.certification_report is not None and not uses_library:
                raise ValueError(
                    "--certification-report is only allowed with --policy-library"
                )
            if arguments.require_certification:
                if arguments.certification_report is None:
                    raise ValueError(
                        "--require-certification requires --certification-report"
                    )
                from ft_anns.adaptive_query import (
                    PolicyLibraryCertificationReport,
                )

                assert runtime_policy_library is not None
                certification_report = PolicyLibraryCertificationReport.load(
                    arguments.certification_report,
                    library=runtime_policy_library,
                    library_path=arguments.policy_library,
                )
                if certification_report.status != "passed":
                    raise ValueError(
                        "required Policy Library Certification did not pass"
                    )
            result = retrieve_adaptive(
                queries=queries,
                checkpoint=checkpoint,
                partition=partition,
                artifact=runtime_calibration,
                policy_library=runtime_policy_library,
                policy_id=arguments.policy_id,
                topk=arguments.topk,
                query_filename=Path(arguments.queries).name,
                search_execution=search_execution,
                **retrieval_batch_options,
            )
        result.save(arguments.output)
        print(
            json.dumps(
                {
                    "results": arguments.output,
                    "query_count": len(result.neighbor_ids),
                    "mode": arguments.mode,
                    "topm": arguments.topm,
                    "topk": arguments.topk,
                    "faiss_threads": faiss_threads,
                    "policy_id": arguments.policy_id,
                    "timestamps": {
                        "started_at": result.started_at,
                        "finished_at": result.finished_at,
                    },
                    "timing": result.timing.to_dict(),
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "search-sweep":
        import torch

        from ft_anns.adaptive_query import (
            PolicyLibrary,
            RetrievalResult,
            retrieve_adaptive,
            retrieve_fixed,
        )
        from ft_anns.checkpoint import RouterCheckpoint
        from ft_anns.partition import FixedPartition

        uses_fixed = arguments.fixed_topms is not None
        uses_library = arguments.policy_library is not None
        if uses_fixed == uses_library:
            raise ValueError(
                "search-sweep requires exactly one of --fixed-topms or --policy-library"
            )
        if arguments.policy_ids is not None and not uses_library:
            raise ValueError("--policy-ids requires --policy-library")
        output_dir = Path(arguments.output_dir)
        log_path = _configure_command_logging(arguments, output_dir / "sweep.log")
        faiss_threads = _configure_faiss_threads(arguments.faiss_threads)
        retrieval_batch_options = _retrieval_batch_options(arguments)
        search_execution = _resolve_search_execution(
            requested=arguments.search_execution,
            adaptive=not uses_fixed,
        )
        expected_router_batch_size = (
            arguments.batch_size
            if arguments.batch_size is not None
            else arguments.router_batch_size or "full"
        )
        expected_faiss_batch_size = (
            arguments.batch_size
            if arguments.batch_size is not None
            else arguments.faiss_batch_size or "full"
        )
        partition = FixedPartition.load(arguments.partition)
        checkpoint = RouterCheckpoint.load(arguments.checkpoint, partition=partition)
        if arguments.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA retrieval requested but PyTorch reports no GPU")
        checkpoint.model.to(arguments.device)
        queries = _load_vectors(
            arguments.queries,
            dimension=partition.dimension,
            format_name=arguments.query_format,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        query_filename = Path(arguments.queries).name
        completed: list[str] = []

        def save_or_reuse(
            path: Path,
            build: Callable[[], Any],
            *,
            expected_topm: int | None = None,
            expected_policy_id: str | None = None,
        ) -> None:
            if arguments.resume and path.exists():
                existing = RetrievalResult.load(path)
                existing.validate_for_evaluation(
                    dataset_name=partition.dataset_name,
                    query_filename=query_filename,
                    top_k=arguments.topk,
                )
                if (
                    existing.checkpoint_filename != checkpoint.filename
                    or existing.partition_filename != partition.filename
                    or existing.timing.router_batch_size
                    != expected_router_batch_size
                    or existing.timing.faiss_batch_size
                    != expected_faiss_batch_size
                    or existing.timing.search_execution != search_execution
                    or (
                        expected_topm is not None
                        and (
                            int(existing.probe_counts.min()) != expected_topm
                            or int(existing.probe_counts.max()) != expected_topm
                        )
                    )
                    or (
                        expected_policy_id is not None
                        and existing.experiment_coordinates.get("policy_id")
                        != expected_policy_id
                    )
                ):
                    raise ValueError(f"incompatible sweep result cannot resume: {path}")
            else:
                result = build()
                result.save(path)
            completed.append(str(path))

        if uses_fixed:
            assert arguments.fixed_topms is not None
            if arguments.fixed_topms != sorted(set(arguments.fixed_topms)):
                raise ValueError("--fixed-topms must be unique and increasing")
            for topm in arguments.fixed_topms:
                save_or_reuse(
                    output_dir / f"fixed_m{topm}.npz",
                    lambda topm=topm: retrieve_fixed(
                        queries=queries,
                        checkpoint=checkpoint,
                        partition=partition,
                        topm=topm,
                        topk=arguments.topk,
                        query_filename=query_filename,
                        search_execution=search_execution,
                        **retrieval_batch_options,
                    ),
                    expected_topm=topm,
                )
        else:
            library = PolicyLibrary.load(
                arguments.policy_library,
                checkpoint=checkpoint,
                partition=partition,
            )
            policy_ids = arguments.policy_ids or list(library.policy_ids)
            for policy_id in policy_ids:
                library.resolve(
                    policy_id, checkpoint=checkpoint, partition=partition
                )
                policy_slug = policy_id.replace("-", "_")
                save_or_reuse(
                    output_dir / f"adaptive_{policy_slug}.npz",
                    lambda policy_id=policy_id: retrieve_adaptive(
                        queries=queries,
                        checkpoint=checkpoint,
                        partition=partition,
                        policy_library=library,
                        policy_id=policy_id,
                        topk=arguments.topk,
                        query_filename=query_filename,
                        search_execution=search_execution,
                        **retrieval_batch_options,
                    ),
                    expected_policy_id=policy_id,
                )
        print(
            json.dumps(
                {
                    "results": completed,
                    "query_count": len(queries),
                    "faiss_threads": faiss_threads,
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "evaluate":
        from ft_anns.adaptive_query import (
            RetrievalResult,
            evaluate_results,
        )
        from ft_anns.artifact_io import artifact_filename
        from ft_anns.config import EvaluationConfig, load_config

        loaded = load_config("evaluation", arguments.config)
        assert isinstance(loaded, EvaluationConfig)
        ground_truth_path = arguments.ground_truth or loaded.ground_truth_path
        output_path = arguments.output or loaded.output_path
        log_path = _configure_command_logging(
            arguments, Path(output_path).with_suffix(".log")
        )
        logger.info(
            "evaluate command started results=%s ground_truth=%s output=%s",
            arguments.results,
            ground_truth_path,
            output_path,
        )
        result = RetrievalResult.load(arguments.results)
        result.validate_for_evaluation(
            dataset_name=loaded.dataset,
            query_filename=artifact_filename(loaded.query_path),
            top_k=loaded.top_k,
        )
        report = evaluate_results(
            retrieval_result=result,
            ground_truth=_load_ground_truth(ground_truth_path),
            recall_ks=tuple(loaded.recall_ks),
            ground_truth_filename=Path(ground_truth_path).name,
        )
        from ft_anns.observability import utc_now_iso

        evaluation_save_started = time.perf_counter()
        report.save(output_path)
        evaluation_save_seconds = time.perf_counter() - evaluation_save_started
        evaluation_persisted_at = utc_now_iso()
        logger.info(
            "evaluation report persisted output=%s save_seconds=%.6f persisted_at=%s",
            output_path,
            evaluation_save_seconds,
            evaluation_persisted_at,
        )
        print(
            json.dumps(
                {
                    "report": output_path,
                    "query_count": report.query_count,
                    "recall_at_k": report.recall_at_k,
                    "mean_ndis": report.mean_ndis,
                    "qps": report.qps,
                    "evaluation_timestamps": report.evaluation_timestamps,
                    "evaluation_timing": report.evaluation_timing.to_dict(),
                    "evaluation_persistence": {
                        "save_seconds": evaluation_save_seconds,
                        "persisted_at": evaluation_persisted_at,
                    },
                    "retrieval_timestamps": report.retrieval_timestamps,
                    "retrieval_timing": report.retrieval_timing,
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    elif arguments.command == "report-matrix":
        from ft_anns.experiments import (
            ExperimentMatrixManifest,
            build_experiment_matrix_report,
        )

        matrix_manifest = ExperimentMatrixManifest.load(arguments.config)
        if arguments.output is not None:
            payload = matrix_manifest.model_dump()
            payload["output_path"] = arguments.output
            matrix_manifest = ExperimentMatrixManifest.model_validate(payload)
        log_path = _configure_command_logging(
            arguments, Path(matrix_manifest.output_path).with_suffix(".log")
        )
        logger.info(
            "report-matrix command started dataset=%s output=%s",
            matrix_manifest.dataset,
            matrix_manifest.output_path,
        )
        matrix_report = build_experiment_matrix_report(matrix_manifest)
        print(
            json.dumps(
                {
                    "report": matrix_manifest.output_path,
                    "report_filename": matrix_report.filename,
                    "evaluation_count": len(matrix_report.payload["evaluations"]),
                    "reference_parity": matrix_report.payload["reference_parity"][
                        "passed"
                    ],
                    "log": str(log_path),
                    "status": "finalized",
                },
                sort_keys=True,
            )
        )
    logger.info("command completed command=%s", arguments.command)
    return 0


def main(argv: list[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        return _run_command(arguments)
    except Exception:
        logger.exception("command failed command=%s", arguments.command)
        raise
