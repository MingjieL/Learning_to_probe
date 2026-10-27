"""Dataset-matrix validation and report assembly seams."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Self

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ft_anns.adaptive_query import CalibrationArtifact, EvaluationReport, PolicyLibrary
from ft_anns.adaptive_query.policy import build_probe_plan
from ft_anns.adaptive_query.retrieval import retrieve_probe_plan
from ft_anns.artifact_io import artifact_filename, write_json_atomic
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.compatibility import (
    compatibility_filename,
    compatibility_slug,
    normalize_compatibility_profile,
    require_compatibility_metadata,
)
from ft_anns.config import PartitionConfig, WorkflowCompatibilityConfig
from ft_anns.datasets import SyntheticDataset, WorkloadManifest
from ft_anns.domain import DatasetName
from ft_anns.fixtures import load_reference_fixture
from ft_anns.partition import FixedPartition
from ft_anns.routing import compute_routing_loss


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MatrixEvaluation(_StrictModel):
    name: str
    mode: Literal["fixed", "adaptive"]
    checkpoint: Literal["ce", "bce", "ramps"]
    topm: Annotated[int, Field(gt=0)] | None = None
    calibration: Literal["budget_ndis", "target_recall"] | None = None
    policy_library: str | None = None
    policy_id: str | None = None
    report_path: str

    @model_validator(mode="after")
    def validate_mode_contract(self) -> Self:
        if not self.name:
            raise ValueError("matrix evaluation name must not be empty")
        if self.mode == "fixed" and any(
            value is not None
            for value in (self.calibration, self.policy_library, self.policy_id)
        ):
            raise ValueError("fixed matrix evaluation cannot name an adaptive policy")
        if self.mode == "fixed" and self.topm is None:
            raise ValueError("fixed matrix evaluation requires topm")
        if self.mode == "adaptive" and (
            (self.calibration is None) == (self.policy_library is None)
        ):
            raise ValueError(
                "adaptive matrix evaluation requires one calibration or Policy Library"
            )
        if (self.policy_library is None) != (self.policy_id is None):
            raise ValueError("matrix Policy Library and policy ID must be paired")
        if self.mode == "adaptive" and self.topm is not None:
            raise ValueError("adaptive matrix evaluation cannot name topm")
        if self.mode == "adaptive" and self.checkpoint != "ramps":
            raise ValueError(
                "adaptive matrix evaluation requires the RAMPS checkpoint"
            )
        return self


class MatrixPolicyLibraryEvaluation(_StrictModel):
    checkpoint: Literal["ramps"] = "ramps"
    policy_library: str
    report_directory: str
    name_prefix: str = "ramps_"

    @model_validator(mode="after")
    def validate_paths_and_names(self) -> Self:
        if (
            not self.policy_library
            or not self.report_directory
            or not self.name_prefix
        ):
            raise ValueError(
                "matrix Policy Library evaluation fields must not be empty"
            )
        return self


class ExperimentMatrixManifest(_StrictModel):
    format: Literal[
        "deep1m-experiment-matrix-v1",
        "sift1m-experiment-matrix-v1",
        "spacev1m-experiment-matrix-v1",
        "deep100m-experiment-matrix-v1",
        "sift100m-experiment-matrix-v1",
        "spacev100m-experiment-matrix-v1",
    ] = "deep1m-experiment-matrix-v1"
    dataset: DatasetName
    scope: Literal["comparative", "ramps_only"] = "comparative"
    compatibility: WorkflowCompatibilityConfig | None = None
    workload_manifest_path: str
    partition_path: str
    checkpoints: dict[Literal["ce", "bce", "ramps"], str]
    calibrations: dict[Literal["budget_ndis", "target_recall"], str]
    policy_libraries: dict[str, str] = Field(default_factory=dict)
    policy_library_evaluations: list[MatrixPolicyLibraryEvaluation] = Field(
        default_factory=list
    )
    fixed_bucket_counts: list[Annotated[int, Field(gt=0)]] = Field(
        default_factory=list
    )
    evaluations: list[MatrixEvaluation]
    loss_absolute_tolerance: Annotated[
        float, Field(ge=0, allow_inf_nan=False)
    ] = 1e-12
    output_path: str

    @model_validator(mode="after")
    def validate_complete_matrix(self) -> Self:
        expected_format = {
            "Deep1M": "deep1m-experiment-matrix-v1",
            "Sift1M": "sift1m-experiment-matrix-v1",
            "SpaceV1M": "spacev1m-experiment-matrix-v1",
            "Deep100M": "deep100m-experiment-matrix-v1",
            "Sift100M": "sift100m-experiment-matrix-v1",
            "SpaceV100M": "spacev100m-experiment-matrix-v1",
        }[self.dataset]
        if self.format != expected_format:
            raise ValueError("matrix manifest format does not match dataset")
        if self.compatibility is not None:
            if self.compatibility.dataset_name != self.dataset:
                raise ValueError(
                    "matrix Compatibility Profile dataset does not match config"
                )
            if self.compatibility.loss != "ramps":
                raise ValueError("matrix Compatibility Profile must bind RAMPS")
        expected_checkpoints = (
            {"ce", "bce", "ramps"}
            if self.scope == "comparative"
            else {"ramps"}
        )
        if set(self.checkpoints) != expected_checkpoints:
            raise ValueError(
                "matrix checkpoints do not match its comparative/RAMPS-only scope"
            )
        if (
            not self.fixed_bucket_counts
            and not self.calibrations
            and not self.policy_libraries
        ):
            raise ValueError(
                "matrix requires a Fixed Query sweep, calibration, or Policy Library"
            )
        if len(self.fixed_bucket_counts) != len(
            set(self.fixed_bucket_counts)
        ):
            raise ValueError("matrix fixed_bucket_counts must be unique")
        if self.fixed_bucket_counts != sorted(self.fixed_bucket_counts):
            raise ValueError("matrix fixed_bucket_counts must be sorted")
        if self.fixed_bucket_counts and any(
            entry.mode == "fixed" for entry in self.evaluations
        ):
            raise ValueError(
                "matrix cannot combine fixed_bucket_counts with explicit fixed evaluations"
            )
        resolved_evaluations = self.resolved_evaluations
        names = [entry.name for entry in resolved_evaluations]
        if len(names) != len(set(names)):
            raise ValueError("matrix evaluation names must be unique")
        fixed_losses = {
            entry.checkpoint for entry in resolved_evaluations if entry.mode == "fixed"
        }
        adaptive_selectors = {
            entry.calibration
            for entry in resolved_evaluations
            if entry.mode == "adaptive" and entry.calibration is not None
        }
        adaptive_policy_libraries = {
            entry.policy_library
            for entry in resolved_evaluations
            if entry.mode == "adaptive" and entry.policy_library is not None
        }
        dynamic_policy_libraries = {
            entry.policy_library for entry in self.policy_library_evaluations
        }
        if len(dynamic_policy_libraries) != len(self.policy_library_evaluations):
            raise ValueError(
                "matrix Policy Library evaluations must name unique libraries"
            )
        if adaptive_policy_libraries.intersection(dynamic_policy_libraries):
            raise ValueError(
                "matrix Policy Library cannot use explicit and dynamic evaluations"
            )
        if fixed_losses != expected_checkpoints:
            raise ValueError("matrix Fixed-M evaluations do not match its scope")
        if adaptive_selectors != set(self.calibrations):
            raise ValueError(
                "matrix requires one adaptive evaluation for each calibration"
            )
        if adaptive_policy_libraries.union(dynamic_policy_libraries) != set(
            self.policy_libraries
        ):
            raise ValueError(
                "matrix requires one adaptive evaluation for each Policy Library"
            )
        if not self.output_path:
            raise ValueError("matrix output_path must not be empty")
        return self

    @property
    def resolved_evaluations(self) -> tuple[MatrixEvaluation, ...]:
        """Expand the compact Fixed-M sweep into report-bound matrix cells."""
        report_directory = Path(self.output_path).parent
        fixed = tuple(
            MatrixEvaluation(
                name=f"{checkpoint}_fixed_m{topm}",
                mode="fixed",
                checkpoint=checkpoint,
                topm=topm,
                report_path=str(
                    report_directory / f"{checkpoint}_fixed_m{topm}.json"
                ),
            )
            for checkpoint in ("ce", "bce", "ramps")
            if checkpoint in self.checkpoints
            for topm in self.fixed_bucket_counts
        )
        return fixed + tuple(self.evaluations)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentMatrixManifest":
        matrix_path = Path(path)
        with matrix_path.open(encoding="utf-8") as stream:
            config = cls.model_validate(json.load(stream))
        if config.compatibility is None:
            raise ValueError("matrix config requires a Compatibility Profile")
        expected_filename = compatibility_filename(
            config.compatibility.model_dump(mode="json", exclude_none=True),
            kind="matrix",
            extension=".json",
        )
        if matrix_path.name != expected_filename:
            raise ValueError(
                "matrix config compatibility filename mismatch: expected "
                f"{expected_filename!r}, found {matrix_path.name!r}"
            )
        return config


class ExperimentMatrixReport:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)

    @property
    def filename(self) -> str:
        return str(self.payload["report_filename"])

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentMatrixReport":
        report_path = Path(path)
        with report_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if payload.get("format") != "ft-anns-experiment-matrix-v1":
            raise ValueError("unsupported experiment matrix report format")
        if payload.get("report_filename") != artifact_filename(report_path):
            raise ValueError("experiment matrix report filename mismatch")
        require_compatibility_metadata(
            payload, filename=report_path, kind="matrix"
        )
        return cls(payload)


_EXPECTED_LOSSES = {
    "ce": "ce",
    "bce": "bce",
    "ramps": "ramps",
}


def _fair_training_contract(checkpoint: RouterCheckpoint) -> dict[str, Any]:
    resolved = dict(checkpoint.metadata["resolved_train_config"])
    resolved.pop("loss", None)
    resolved.pop("ramps", None)
    resolved.pop("output_dir", None)
    resolved.pop("compatibility", None)
    return resolved


def _evaluation_payload(
    report: EvaluationReport, entry: MatrixEvaluation
) -> dict[str, Any]:
    return {
        "mode": entry.mode,
        "checkpoint": entry.checkpoint,
        "topm": entry.topm,
        "calibration": entry.calibration,
        "policy_library": entry.policy_library,
        "policy_id": entry.policy_id,
        "query_count": report.query_count,
        "recall_at_k": {str(k): value for k, value in report.recall_at_k.items()},
        "latency_seconds": {
            "total": report.total_latency_seconds,
            "mean": report.mean_latency_seconds,
        },
        "qps": report.qps,
        "evaluation_timing": report.evaluation_timing.to_dict(),
        "evaluation_timestamps": report.evaluation_timestamps,
        "retrieval_timing": report.retrieval_timing,
        "retrieval_timestamps": report.retrieval_timestamps,
        "total_ndis": report.total_ndis,
        "mean_ndis": report.mean_ndis,
        "probe_counts": {
            "total": report.total_probe_count,
            "mean": report.mean_probe_count,
            "min": report.min_probe_count,
            "max": report.max_probe_count,
        },
        "input_artifacts": report.input_artifacts,
    }


def build_experiment_matrix_report(
    config: ExperimentMatrixManifest,
) -> ExperimentMatrixReport:
    """Validate one complete dataset matrix and persist its auditable report."""
    manifest = WorkloadManifest.load(config.workload_manifest_path)
    if manifest.metadata.get("dataset_name") != config.dataset:
        raise ValueError("experiment matrix dataset does not match its workload")
    partition = FixedPartition.load(config.partition_path)
    checkpoints = {
        alias: RouterCheckpoint.load(path, partition=partition)
        for alias, path in config.checkpoints.items()
    }
    matrix_profile = (
        config.compatibility.model_dump(mode="json", exclude_none=True)
        if config.compatibility is not None
        else checkpoints["ramps"].metadata.get("compatibility")
    )
    if not isinstance(matrix_profile, dict):
        raise ValueError("matrix Compatibility Profile is missing")
    if config.compatibility is not None:
        declared = config.compatibility.model_dump(mode="json", exclude_none=True)
        stored = checkpoints["ramps"].metadata.get("compatibility")
        if not isinstance(stored, dict) or normalize_compatibility_profile(
            declared
        ) != normalize_compatibility_profile(stored):
            raise ValueError(
                "matrix Compatibility Profile does not match RAMPS checkpoint"
            )
    for alias, checkpoint in checkpoints.items():
        if checkpoint.metadata.get("loss", {}).get("name") != _EXPECTED_LOSSES[alias]:
            raise ValueError(f"matrix {alias} checkpoint has the wrong routing loss")
        manifest.validate_checkpoint_binding(checkpoint.metadata)
    fair_contracts = {
        alias: _fair_training_contract(checkpoint)
        for alias, checkpoint in checkpoints.items()
    }
    fair_identities = {
        json.dumps(value, sort_keys=True) for value in fair_contracts.values()
    }
    if len(fair_identities) != 1:
        raise ValueError("matrix checkpoints do not share model and training budget")
    budget_traces = {
        alias: tuple(
            (int(metric["train_unique_queries"]), int(metric["optimizer_steps"]))
            for metric in checkpoint.metrics
        )
        for alias, checkpoint in checkpoints.items()
    }
    if len(set(budget_traces.values())) != 1:
        raise ValueError(
            "matrix checkpoints saw different query or optimizer-step budgets"
        )

    calibrations = {
        objective: CalibrationArtifact.load(
            path, checkpoint=checkpoints["ramps"], partition=partition
        )
        for objective, path in config.calibrations.items()
    }
    for objective, artifact in calibrations.items():
        if artifact.payload["selection"]["objective"] != objective:
            raise ValueError("matrix calibration selector does not match its label")
    policy_libraries = {
        alias: PolicyLibrary.load(
            path, checkpoint=checkpoints["ramps"], partition=partition
        )
        for alias, path in config.policy_libraries.items()
    }

    dynamic_policy_evaluations = tuple(
        MatrixEvaluation(
            name=f"{sweep.name_prefix}{policy_id.replace('-', '_')}",
            mode="adaptive",
            checkpoint=sweep.checkpoint,
            policy_library=sweep.policy_library,
            policy_id=policy_id,
            report_path=str(
                Path(sweep.report_directory)
                / f"{sweep.name_prefix}{policy_id.replace('-', '_')}.json"
            ),
        )
        for sweep in config.policy_library_evaluations
        for policy_id in policy_libraries[sweep.policy_library].policy_ids
    )
    matrix_evaluations = config.resolved_evaluations + dynamic_policy_evaluations
    evaluation_names = [entry.name for entry in matrix_evaluations]
    if len(evaluation_names) != len(set(evaluation_names)):
        raise ValueError("resolved matrix evaluation names must be unique")

    evaluation_payloads: dict[str, dict[str, Any]] = {}
    final_evaluation_identities: set[tuple[str | None, str | None, int]] = set()
    for entry in matrix_evaluations:
        report = EvaluationReport.load(entry.report_path)
        expected_checkpoint = checkpoints[entry.checkpoint].filename
        if report.input_artifacts.get("checkpoint") != expected_checkpoint:
            raise ValueError("matrix evaluation checkpoint filename mismatch")
        if report.input_artifacts.get("partition") != partition.filename:
            raise ValueError("matrix evaluation partition filename mismatch")
        if entry.mode == "fixed":
            if "calibration" in report.input_artifacts:
                raise ValueError("fixed matrix evaluation has a calibration identity")
            if "policy_library" in report.input_artifacts:
                raise ValueError("fixed matrix evaluation has a Policy Library identity")
        elif entry.calibration is not None:
            if report.input_artifacts.get("calibration") != calibrations[
                entry.calibration
            ].filename:
                raise ValueError("adaptive matrix evaluation calibration mismatch")
            if "policy_library" in report.input_artifacts:
                raise ValueError("single calibration evaluation names a Policy Library")
        else:
            assert entry.policy_library is not None and entry.policy_id is not None
            try:
                library = policy_libraries[entry.policy_library]
            except KeyError as error:
                raise ValueError("matrix evaluation names an unknown Policy Library") from error
            resolved = library.resolve(
                entry.policy_id,
                checkpoint=checkpoints["ramps"],
                partition=partition,
            )
            if (
                report.input_artifacts.get("policy_library") != library.filename
                or report.input_artifacts.get("calibration") != resolved.filename
                or report.experiment_coordinates.get("policy_id") != entry.policy_id
            ):
                raise ValueError("adaptive matrix Policy Library selection mismatch")
        if entry.topm is not None and (
            report.min_probe_count != entry.topm
            or report.max_probe_count != entry.topm
        ):
            raise ValueError("Fixed-M matrix evaluation probe count mismatch")
        final_evaluation_identities.add(
            (
                report.input_artifacts.get("queries"),
                report.input_artifacts.get("ground_truth"),
                report.query_count,
            )
        )
        evaluation_payloads[entry.name] = _evaluation_payload(report, entry)
    if len(final_evaluation_identities) != 1:
        raise ValueError(
            "matrix evaluations do not share final query and ground-truth identities"
        )

    parity = validate_reference_parity(
        loss_absolute_tolerance=config.loss_absolute_tolerance
    )
    if not parity["passed"]:
        raise ValueError("source-reference parity validation failed")
    ground_truth = manifest.metadata["ground_truth"]
    input_artifacts = {
        "dataset": manifest.metadata["dataset_name"],
        "base": manifest.metadata["base_filename"],
        "query_source": manifest.metadata["query_source_filename"],
        "workload_manifest": manifest.filename,
        "workload_ground_truth": ground_truth["artifact_filename"],
        "partition": str(partition.filename),
        "checkpoints": {
            alias: checkpoint.filename for alias, checkpoint in checkpoints.items()
        },
        "calibrations": {
            objective: artifact.filename
            for objective, artifact in calibrations.items()
        },
        "policy_libraries": {
            alias: library.filename
            for alias, library in policy_libraries.items()
        },
    }
    payload: dict[str, Any] = {
        "format": "ft-anns-experiment-matrix-v1",
        "dataset": config.dataset,
        "compatibility": matrix_profile,
        "compatibility_slug": compatibility_slug(matrix_profile),
        "input_artifacts": input_artifacts,
        "training": {
            alias: {
                "loss": checkpoint.metadata["loss"],
                "selection": checkpoint.metadata["selection"],
                "checkpoint_filename": checkpoint.filename,
                "epoch_budgets": [
                    {
                        "unique_queries": unique_queries,
                        "optimizer_steps": optimizer_steps,
                    }
                    for unique_queries, optimizer_steps in budget_traces[alias]
                ],
            }
            for alias, checkpoint in checkpoints.items()
        },
        "calibration": {
            objective: {
                "artifact_filename": artifact.filename,
                "policy": artifact.payload["policy"],
                "selection": artifact.payload["selection"],
            }
            for objective, artifact in calibrations.items()
        },
        "policy_libraries": {
            alias: {
                "library_filename": library.filename,
                "policy_ids": list(library.policy_ids),
            }
            for alias, library in policy_libraries.items()
        },
        "evaluations": evaluation_payloads,
        "reference_parity": parity,
    }
    output_path = Path(config.output_path)
    expected_output_filename = compatibility_filename(
        matrix_profile, kind="matrix", extension=".json"
    )
    if output_path.name != expected_output_filename:
        raise ValueError(
            "matrix report compatibility filename mismatch: expected "
            f"{expected_output_filename!r}, found {output_path.name!r}"
        )
    payload["report_filename"] = artifact_filename(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_path, payload)
    return ExperimentMatrixReport(payload)


def validate_reference_parity(
    *, loss_absolute_tolerance: float
) -> dict[str, Any]:
    """Replay the static source-oracle fixture used by the experiment matrix."""
    if (
        isinstance(loss_absolute_tolerance, bool)
        or not np.isfinite(loss_absolute_tolerance)
        or loss_absolute_tolerance < 0
    ):
        raise ValueError("loss_absolute_tolerance must be finite and non-negative")
    fixture = load_reference_fixture()
    loss_fixture = fixture["losses"]
    logits = torch.tensor(loss_fixture["logits"], dtype=torch.float64)
    bucket_ids = torch.tensor(loss_fixture["bucket_ids"], dtype=torch.int64)
    list_sizes = torch.tensor(
        loss_fixture["ramps"]["list_sizes"], dtype=torch.float64
    )
    actual_losses = {
        "ce": float(
            compute_routing_loss(
                loss_name="ce", logits=logits, bucket_ids=bucket_ids
            )
        ),
        "bce": float(
            compute_routing_loss(
                loss_name="bce", logits=logits, bucket_ids=bucket_ids
            )
        ),
        "ramps": float(
            compute_routing_loss(
                loss_name="ramps",
                logits=logits,
                bucket_ids=bucket_ids,
                list_sizes=list_sizes,
                ramps={
                    "rank_weight": loss_fixture["ramps"]["rank_weight"],
                },
            )
        ),
    }
    expected_losses = loss_fixture["expected"]
    errors = {
        name: abs(value - float(expected_losses[name]))
        for name, value in actual_losses.items()
    }
    maximum_loss_error = max(errors.values())

    policy_fixture = fixture["ramps_policy"]
    probabilities = np.asarray(policy_fixture["probabilities"], dtype=np.float64)
    probe_plan = build_probe_plan(
        router_scores=np.log(probabilities),
        list_sizes=np.asarray(policy_fixture["list_sizes"], dtype=np.int64),
        artifact={
            "format": "adaptive-query-calibration-v3",
            "decoder": policy_fixture["decoder"],
            "policy": {
                "cost_lambda": policy_fixture["cost_lambda"],
                "m_min": policy_fixture["m_min"],
                "m_max": len(policy_fixture["list_sizes"]),
            },
            "target_size_transform": policy_fixture["size_transform"],
        },
    )
    probe_plan_matches = bool(
        np.array_equal(probe_plan.bucket_ids, policy_fixture["ranking"])
        and np.array_equal(probe_plan.probe_counts, policy_fixture["probe_counts"])
    )

    dataset = SyntheticDataset.deterministic()
    partition = FixedPartition.build(
        dataset.base_vectors,
        PartitionConfig(bucket_count=4, build_seed=6),
        metric="l2",
        dataset_name="Synthetic8",
    )
    expected_fixed = fixture["micro_partition"]["fixed_search"]
    ranking_scores = np.full(
        (len(dataset.query_vectors), partition.bucket_count),
        -10.0,
        dtype=np.float64,
    )
    for row_id, route in enumerate(expected_fixed["probe_buckets"]):
        for rank, bucket_id in enumerate(route):
            ranking_scores[row_id, bucket_id] = 10.0 - rank
    chained_plan = build_probe_plan(
        router_scores=ranking_scores,
        list_sizes=partition.list_sizes,
        artifact={
            "format": "adaptive-query-calibration-v3",
            "decoder": "ramps_size_matched",
            "policy": {"cost_lambda": 0.0, "m_min": 2, "m_max": 2},
            "target_size_transform": {
                "mode": "anchored_power",
                "alpha": 1.0,
                "anchor": float(partition.list_sizes.mean()),
            },
        },
    )
    probe_plan_matches = bool(
        probe_plan_matches
        and np.array_equal(chained_plan.bucket_ids, expected_fixed["probe_buckets"])
        and np.array_equal(chained_plan.probe_counts, [2, 2, 2])
    )
    searched = retrieve_probe_plan(
        queries=dataset.query_vectors,
        partition=partition,
        probe_plan=chained_plan,
        topk=len(expected_fixed["neighbor_ids"][0]),
    )
    neighbor_ids_match = bool(
        np.array_equal(searched.neighbor_ids, expected_fixed["neighbor_ids"])
    )
    ndis_match = bool(np.array_equal(searched.ndis, expected_fixed["ndis"]))
    checks: dict[str, Any] = {
        "routing_losses": {
            "absolute_tolerance": loss_absolute_tolerance,
            "max_absolute_error": maximum_loss_error,
            "absolute_errors": errors,
            "passed": maximum_loss_error <= loss_absolute_tolerance,
        },
        "probe_plan": {"exact_match": probe_plan_matches},
        "neighbor_ids": {"exact_match": neighbor_ids_match},
        "ndis": {"exact_match": ndis_match},
    }
    return {
        "oracle_repository": fixture["oracle"]["repository"],
        "oracle_commit": fixture["oracle"]["commit"],
        "execution": "policy_probe_plan_to_preassigned_search",
        "checks": checks,
        "passed": bool(
            checks["routing_losses"]["passed"]
            and probe_plan_matches
            and neighbor_ids_match
            and ndis_match
        ),
    }


__all__ = [
    "ExperimentMatrixManifest",
    "ExperimentMatrixReport",
    "MatrixEvaluation",
    "build_experiment_matrix_report",
    "validate_reference_parity",
]
