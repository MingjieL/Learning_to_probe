"""Deterministic router training with true ANN checkpoint validation."""

from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ft_anns.artifact_io import write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint, router_checkpoint_format
from ft_anns.compatibility import compatibility_slug, normalize_compatibility_profile
from ft_anns.config import TrainConfig
from ft_anns.datasets import WorkloadManifest
from ft_anns.observability import utc_now_iso
from ft_anns.partition import FixedPartition
from ft_anns.routing.data import (
    QueryLevelWorkload,
    TrainingWorkload,
    create_query_dataloader,
)
from ft_anns.routing.losses import compute_routing_loss
from ft_anns.routing.model import RawQueryRouter
from ft_anns.routing.validation import (
    CheckpointValidationReport,
    CheckpointValidationWorkload,
    evaluate_checkpoint,
)

logger = logging.getLogger(__name__)


def _validation_loss(
    model: RawQueryRouter,
    workload: CheckpointValidationWorkload,
    *,
    partition: FixedPartition,
    device: torch.device,
    config: TrainConfig,
    list_sizes: Tensor,
) -> float:
    bucket_ids = partition.bucket_ids_for(workload.ground_truth)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.from_numpy(np.array(workload.queries, copy=True)).to(device)
        )
        loss = compute_routing_loss(
            loss_name=config.loss,
            logits=logits,
            bucket_ids=torch.from_numpy(np.array(bucket_ids, copy=True)).to(device),
            list_sizes=list_sizes,
            ramps=config.ramps,
        )
    return float(loss.cpu())


def _validation_payload(report: CheckpointValidationReport) -> dict[str, Any]:
    return {
        "query_count": report.query_count,
        "fixed_bucket_count": report.fixed_bucket_count,
        "ranking": report.ranking,
        "ann_recall_at_k": {
            str(k): value for k, value in sorted(report.ann_recall_at_k.items())
        },
        "total_probe_count": report.total_probe_count,
        "mean_probe_count": report.mean_probe_count,
        "total_ndis": report.total_ndis,
        "mean_ndis": report.mean_ndis,
        "timing": report.timing.to_dict(),
    }


def _initialize_jsonl(path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, payload: object) -> None:
    with path.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_partition_stats(
    path: Path,
    *,
    partition: FixedPartition,
    configured_artifact_path: str | None,
) -> None:
    sizes = np.asarray(partition.list_sizes, dtype=np.float64)
    mean = float(sizes.mean())
    standard_deviation = float(sizes.std())
    list_sizes_path = (
        str(Path(configured_artifact_path) / "list_sizes.npy")
        if configured_artifact_path is not None
        else None
    )
    write_json_atomic(
        path,
        {
            "format": "fixed-partition-stats-v1",
            "partition_filename": partition.filename,
            "list_sizes_path": list_sizes_path,
            "bucket_count": partition.bucket_count,
            "base_count": int(sizes.sum()),
            "list_size": {
                "minimum": int(sizes.min()),
                "mean": mean,
                "variance": float(sizes.var()),
                "standard_deviation": standard_deviation,
                "p50": float(np.percentile(sizes, 50)),
                "p95": float(np.percentile(sizes, 95)),
                "p99": float(np.percentile(sizes, 99)),
                "maximum": int(sizes.max()),
                "coefficient_of_variation": standard_deviation / mean,
            },
        },
    )


def _ensure_partition_stats(
    path: Path,
    *,
    partition: FixedPartition,
    configured_artifact_path: str | None,
) -> None:
    if path.exists():
        try:
            with path.open(encoding="utf-8") as stream:
                existing = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("existing partition_stats.json is unreadable") from error
        if (
            not isinstance(existing, dict)
            or existing.get("format") != "fixed-partition-stats-v1"
            or existing.get("partition_filename") != partition.filename
        ):
            raise ValueError("existing partition_stats.json is incompatible")
        return
    _write_partition_stats(
        path,
        partition=partition,
        configured_artifact_path=configured_artifact_path,
    )


def _resume_contract(
    *,
    config: TrainConfig,
    manifest: WorkloadManifest,
    partition: FixedPartition,
    train_workload: TrainingWorkload,
    validation_workload: CheckpointValidationWorkload,
) -> dict[str, Any]:
    return {
        "format": "router-training-resume-contract-v1",
        "dataset": config.dataset,
        "workload_manifest_filename": manifest.filename,
        "router_training_workload_contract": manifest.router_training_contract,
        "partition_filename": partition.filename,
        "train_workload_filename": train_workload.content_kind,
        "validation_workload_filename": validation_workload.content_kind,
        "router": config.router.model_dump(mode="json"),
        "loss": config.loss,
        "ramps": (
            config.ramps.model_dump(mode="json") if config.ramps is not None else None
        ),
        "batch_size": config.batch_size,
        "optimizer": config.optimizer.model_dump(mode="json"),
        "validation": config.validation.model_dump(mode="json"),
        "early_stopping": config.early_stopping.model_dump(mode="json"),
        "seed": config.seed,
        "device": config.device,
    }


def _cpu_state_dict(model: RawQueryRouter) -> dict[str, Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _save_last_training_state(
    path: Path,
    *,
    contract: dict[str, Any],
    model: RawQueryRouter,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: list[dict[str, Any]],
    best_state: dict[str, Tensor],
    best_epoch: int,
    best_recall: float,
    epochs_without_improvement: int,
    stopped_early: bool,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "router-training-state-v1",
        "contract": contract,
        "epoch": epoch,
        "model_state": _cpu_state_dict(model),
        "optimizer_state": optimizer.state_dict(),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
        "metrics": metrics,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_recall": best_recall,
        "epochs_without_improvement": epochs_without_improvement,
        "stopped_early": stopped_early,
    }
    temporary_state = path / ".training_state.pt.tmp"
    torch.save(payload, temporary_state)
    temporary_state.replace(path / "training_state.pt")
    write_json_atomic(
        path / "metadata.json",
        {
            "format": "router-training-last-v1",
            "contract": contract,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_recall": best_recall,
            "epochs_without_improvement": epochs_without_improvement,
            "stopped_early": stopped_early,
        },
    )


def _load_last_training_state(
    path: Path,
    *,
    expected_contract: dict[str, Any],
) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    state_path = path / "training_state.pt"
    try:
        with metadata_path.open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        state = torch.load(state_path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, ValueError, TypeError) as error:
        raise ValueError(f"resume checkpoint is unreadable: {path}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("format") != "router-training-last-v1"
        or not isinstance(state, dict)
        or state.get("format") != "router-training-state-v1"
        or metadata.get("contract") != expected_contract
        or state.get("contract") != expected_contract
    ):
        raise ValueError("resume checkpoint is incompatible with this training run")
    return state


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device
) -> None:
    for state in optimizer.state.values():
        for name, value in state.items():
            if isinstance(value, Tensor):
                state[name] = value.to(device)


def _validate_train_workload_provenance(
    workload: TrainingWorkload,
    *,
    manifest: WorkloadManifest,
    partition: FixedPartition,
) -> None:
    provenance = workload.provenance
    if provenance is None:
        raise ValueError("router_train workload has no manifest provenance")
    expected = {
        "manifest_filename": manifest.filename,
        "role": "router_train",
        "query_source_filename": manifest.metadata["query_source_filename"],
        "ground_truth_artifact_filename": manifest.metadata["ground_truth"][
            "artifact_filename"
        ],
        "partition_filename": str(partition.filename),
    }
    actual = {
        "manifest_filename": provenance.manifest_filename,
        "role": provenance.role,
        "query_source_filename": provenance.query_source_filename,
        "ground_truth_artifact_filename": provenance.ground_truth_artifact_filename,
        "partition_filename": provenance.partition_filename,
    }
    if actual != expected:
        raise ValueError("router_train workload provenance mismatch")


def _validate_validation_workload_provenance(
    workload: CheckpointValidationWorkload,
    *,
    manifest: WorkloadManifest,
) -> None:
    if workload.manifest_filename != manifest.filename:
        raise ValueError("checkpoint_validation workload manifest mismatch")
    if workload.ground_truth_artifact_filename != manifest.metadata[
        "ground_truth"
    ]["artifact_filename"]:
        raise ValueError("checkpoint_validation ground-truth filename mismatch")


def train_router(
    *,
    config: TrainConfig,
    train_workload: TrainingWorkload,
    validation_workload: CheckpointValidationWorkload,
    partition: FixedPartition,
    resume_from: str | Path | None = None,
) -> RouterCheckpoint:
    """Train on router_train and select strictly by validation ANN Recall@K."""
    run_started_at = utc_now_iso()
    run_started = time.perf_counter()
    logger.info(
        "training started dataset=%s epochs=%d output=%s resume=%s",
        config.dataset,
        config.epochs,
        config.output_dir,
        resume_from,
    )
    manifest = WorkloadManifest.load(config.workload_manifest_path)
    if manifest.metadata["dataset_name"] != config.dataset:
        raise ValueError("training dataset does not match the Workload Manifest")
    if partition.dataset_name != config.dataset:
        raise ValueError("training dataset does not match the Fixed Partition")
    if config.compatibility is not None:
        expected_profile = config.compatibility.model_dump(
            mode="json", exclude_none=True
        )
        actual_profile: dict[str, object] = {
            "dataset_name": manifest.metadata["dataset_name"],
            "ground_truth_k": manifest.metadata["ground_truth"]["k"],
            "bucket_count": partition.bucket_count,
            "kmeans_niter": partition.kmeans_niter,
            "partition_build_seed": partition.build_seed,
            "loss": config.loss,
        }
        if normalize_compatibility_profile(
            expected_profile
        ) != normalize_compatibility_profile(actual_profile):
            raise ValueError(
                "training Compatibility Profile does not match loaded artifacts"
            )
        if Path(config.output_dir).name != compatibility_slug(expected_profile):
            raise ValueError("training output-directory compatibility slug mismatch")
    _validate_train_workload_provenance(
        train_workload,
        manifest=manifest,
        partition=partition,
    )
    _validate_validation_workload_provenance(
        validation_workload,
        manifest=manifest,
    )
    train_query_ids = (
        train_workload.query_ids.detach().cpu().numpy()
        if isinstance(train_workload, QueryLevelWorkload)
        else train_workload.query_ids
    )
    if not np.array_equal(
        train_query_ids, manifest.role_query_ids("router_train")
    ):
        raise ValueError("train_workload is not the manifest router_train role")
    if not np.array_equal(
        validation_workload.query_ids,
        manifest.role_query_ids("checkpoint_validation"),
    ):
        raise ValueError(
            "validation_workload is not the manifest checkpoint_validation role"
        )
    train_dimension = (
        train_workload.queries.shape[1]
        if isinstance(train_workload, QueryLevelWorkload)
        else train_workload.dimension
    )
    if train_dimension != partition.dimension:
        raise ValueError("training query dimension does not match the partition")
    if validation_workload.queries.shape[1] != partition.dimension:
        raise ValueError("validation query dimension does not match the partition")
    if (
        isinstance(train_workload, QueryLevelWorkload)
        and train_workload.bucket_ids.numel()
        and int(train_workload.bucket_ids.max()) >= partition.bucket_count
    ):
        raise ValueError("training bucket label is outside the partition")
    if config.validation.fixed_bucket_count > partition.bucket_count:
        raise ValueError("validation fixed bucket count does not fit the partition")
    if max(config.validation.ann_recall_ks) > validation_workload.ground_truth.shape[1]:
        raise ValueError("validation Recall K exceeds the ground-truth width")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but PyTorch reports no GPU")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").unlink(missing_ok=True)
    write_json_atomic(
        output_dir / "resolved_config.json",
        {
            "format": "resolved-train-config-v1",
            "started_at": run_started_at,
            "config": config.model_dump(mode="json"),
        },
    )
    _ensure_partition_stats(
        output_dir / "partition_stats.json",
        partition=partition,
        configured_artifact_path=config.partition.artifact_path,
    )
    metrics_jsonl_path = output_dir / "training_metrics.jsonl"

    partition_filename = partition.filename
    list_sizes = torch.from_numpy(np.array(partition.list_sizes, copy=True)).to(device)
    model = RawQueryRouter(
        input_dimension=partition.dimension,
        bucket_count=partition.bucket_count,
        config=config.router,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    contract = _resume_contract(
        config=config,
        manifest=manifest,
        partition=partition,
        train_workload=train_workload,
        validation_workload=validation_workload,
    )
    metrics: list[dict[str, Any]] = []
    best_state: dict[str, Tensor] | None = None
    best_epoch = 0
    best_recall: float | None = None
    epochs_without_improvement = 0
    stopped_early = False
    selection_k = config.validation.checkpoint_selection_recall_k
    start_epoch = 1

    if resume_from is not None:
        resumed = _load_last_training_state(
            Path(resume_from), expected_contract=contract
        )
        completed_epoch = resumed.get("epoch")
        if (
            isinstance(completed_epoch, bool)
            or not isinstance(completed_epoch, int)
            or completed_epoch <= 0
            or completed_epoch > config.epochs
        ):
            raise ValueError("resume checkpoint epoch is incompatible with epoch cap")
        try:
            model.load_state_dict(resumed["model_state"], strict=True)
            optimizer.load_state_dict(resumed["optimizer_state"])
            _move_optimizer_state(optimizer, device)
            random.setstate(resumed["python_rng_state"])
            np.random.set_state(resumed["numpy_rng_state"])
            torch.set_rng_state(resumed["torch_cpu_rng_state"])
            cuda_rng_states = resumed["torch_cuda_rng_state_all"]
            if cuda_rng_states:
                if not torch.cuda.is_available() or len(cuda_rng_states) != torch.cuda.device_count():
                    raise ValueError("resume checkpoint CUDA RNG state is incompatible")
                torch.cuda.set_rng_state_all(cuda_rng_states)
            metrics = [dict(entry) for entry in resumed["metrics"]]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in resumed["best_state"].items()
            }
            best_epoch = int(resumed["best_epoch"])
            best_recall = float(resumed["best_recall"])
            epochs_without_improvement = int(
                resumed["epochs_without_improvement"]
            )
            stopped_early = bool(resumed["stopped_early"])
        except (KeyError, TypeError, RuntimeError, ValueError) as error:
            raise ValueError("resume checkpoint training state is incompatible") from error
        if len(metrics) != completed_epoch:
            raise ValueError("resume checkpoint metrics do not match its epoch")
        start_epoch = completed_epoch + 1

    _initialize_jsonl(metrics_jsonl_path)
    for restored_metric in metrics:
        _append_jsonl(metrics_jsonl_path, restored_metric)

    for epoch in range(start_epoch, config.epochs + 1):
        if stopped_early:
            break
        epoch_started_at = utc_now_iso()
        model.train()
        loss_total = 0.0
        query_count = 0
        optimizer_steps = 0
        train_started = time.perf_counter()
        loader = create_query_dataloader(
            train_workload,
            batch_size=config.batch_size,
            shuffle=True,
            seed=config.seed + epoch,
        )
        for queries, bucket_ids in loader:
            queries = queries.to(device)
            bucket_ids = bucket_ids.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_routing_loss(
                loss_name=config.loss,
                logits=model(queries),
                bucket_ids=bucket_ids,
                list_sizes=list_sizes,
                ramps=config.ramps,
            )
            loss.backward()
            optimizer.step()
            loss_total += float(loss.detach().cpu()) * len(queries)
            query_count += len(queries)
            optimizer_steps += 1
        train_seconds = time.perf_counter() - train_started

        validation_loss_started = time.perf_counter()
        validation_loss = _validation_loss(
            model,
            validation_workload,
            partition=partition,
            device=device,
            config=config,
            list_sizes=list_sizes,
        )
        validation_loss_seconds = time.perf_counter() - validation_loss_started
        checkpoint_validation_started = time.perf_counter()
        validation_report = evaluate_checkpoint(
            model=model,
            partition=partition,
            workload=validation_workload,
            fixed_bucket_count=config.validation.fixed_bucket_count,
            recall_ks=tuple(config.validation.ann_recall_ks),
            ranking=config.validation.ranking,
        )
        checkpoint_validation_seconds = (
            time.perf_counter() - checkpoint_validation_started
        )
        selection_value = validation_report.ann_recall_at_k[selection_k]
        improved = best_recall is None or selection_value > best_recall
        if improved:
            best_epoch = epoch
            best_recall = selection_value
            epochs_without_improvement = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        epoch_metrics: dict[str, Any] = {
            "epoch": epoch,
            "timestamps": {
                "started_at": epoch_started_at,
                "finished_at": utc_now_iso(),
            },
            "timing": {
                "train_seconds": train_seconds,
                "validation_loss_seconds": validation_loss_seconds,
                "checkpoint_validation_seconds": checkpoint_validation_seconds,
            },
            "train_loss": loss_total / query_count,
            "train_unique_queries": query_count,
            "optimizer_steps": optimizer_steps,
            "validation_loss": validation_loss,
            "validation": _validation_payload(validation_report),
            "checkpoint_selection": {
                "ranking": config.validation.ranking,
                "recall_k": selection_k,
                "value": selection_value,
                "improved": improved,
                "best_epoch": best_epoch,
                "best_value": best_recall,
            },
            "early_stopping": {
                "patience_epochs": config.early_stopping.patience_epochs,
                "epochs_without_improvement": epochs_without_improvement,
            },
        }
        metrics.append(epoch_metrics)
        _append_jsonl(metrics_jsonl_path, epoch_metrics)
        logger.info(
            "epoch completed epoch=%d train_loss=%.8f validation_loss=%.8f "
            "selection_recall=%.8f improved=%s train_seconds=%.6f "
            "validation_seconds=%.6f",
            epoch,
            epoch_metrics["train_loss"],
            validation_loss,
            selection_value,
            improved,
            train_seconds,
            checkpoint_validation_seconds,
        )

        patience = config.early_stopping.patience_epochs
        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
        assert best_state is not None and best_recall is not None
        _save_last_training_state(
            output_dir / "last",
            contract=contract,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=metrics,
            best_state=best_state,
            best_epoch=best_epoch,
            best_recall=best_recall,
            epochs_without_improvement=epochs_without_improvement,
            stopped_early=stopped_early,
        )
        if stopped_early:
            break

    if partition.filename != partition_filename:
        raise RuntimeError("Fixed Partition changed during router training")
    assert best_state is not None and best_recall is not None
    model.cpu().load_state_dict(best_state)
    model.eval()
    loss_metadata: dict[str, object] = {"name": config.loss}
    if config.ramps is not None:
        loss_metadata.update(
            {
                "parameters": {"rank_weight": config.ramps.rank_weight},
                "target_size_normalization": {
                    "mode": "max",
                    "max_size": int(partition.list_sizes.max()),
                    "partition_filename": partition_filename,
                },
            }
        )
    selection = {
        "role": "checkpoint_validation",
        "metric": "ann_recall_at_k",
        "ranking": config.validation.ranking,
        "recall_k": selection_k,
        "epoch": best_epoch,
        "value": best_recall,
    }
    early_stopping = {
        "patience_epochs": config.early_stopping.patience_epochs,
        "stopped_early": stopped_early,
        "epochs_without_improvement": epochs_without_improvement,
        "completed_epochs": len(metrics),
        "requested_epochs": config.epochs,
    }
    compatibility_profile: dict[str, object] = {
        "dataset_name": manifest.metadata["dataset_name"],
        "ground_truth_k": manifest.metadata["ground_truth"]["k"],
        "bucket_count": partition.bucket_count,
        "kmeans_niter": partition.kmeans_niter,
        "partition_build_seed": partition.build_seed,
        "loss": config.loss,
    }
    metadata = {
        "format": router_checkpoint_format(config.loss),
        "compatibility": compatibility_profile,
        "compatibility_slug": compatibility_slug(compatibility_profile),
        "timestamps": {
            "started_at": run_started_at,
            "finished_at": utc_now_iso(),
        },
        "dataset_name": manifest.metadata["dataset_name"],
        "ground_truth_k": manifest.metadata["ground_truth"]["k"],
        "base_filename": manifest.metadata["base_filename"],
        "workload_manifest_filename": manifest.filename,
        "router_training_workload_contract": manifest.router_training_contract,
        "partition_filename": partition_filename,
        "roles": {
            "train": "router_train",
            "validation": "checkpoint_validation",
        },
        "workload_filenames": {
            "train": train_workload.content_kind,
            "validation": validation_workload.content_kind,
        },
        "loss": loss_metadata,
        "model": {
            "format": "raw-query-router-v1",
            "identity": model.identity,
            "input_dimension": model.input_dimension,
            "bucket_count": model.bucket_count,
            "hidden_dimensions": model.router_config.hidden_dimensions,
            "activation": model.router_config.activation,
        },
        "resolved_train_config": config.model_dump(mode="json"),
        "selection": selection,
        "early_stopping": early_stopping,
        "metrics": metrics,
    }
    checkpoint = RouterCheckpoint.save(
        output_dir / "best",
        model=model,
        metadata=metadata,
        metrics=metrics,
    )
    run_finished_at = utc_now_iso()
    write_json_atomic(
        output_dir / "training_summary.json",
        {
            "format": "router-training-summary-v1",
            "timestamps": {
                "started_at": run_started_at,
                "finished_at": run_finished_at,
            },
            "timing": {
                "total_seconds": time.perf_counter() - run_started,
            },
            "selection": selection,
            "early_stopping": early_stopping,
            "best_checkpoint_path": str(checkpoint.path),
            "best_checkpoint_filename": checkpoint.filename,
        },
    )
    logger.info(
        "training completed best_epoch=%d best_recall=%.8f completed_epochs=%d "
        "stopped_early=%s total_seconds=%.6f",
        best_epoch,
        best_recall,
        len(metrics),
        stopped_early,
        time.perf_counter() - run_started,
    )
    return checkpoint
