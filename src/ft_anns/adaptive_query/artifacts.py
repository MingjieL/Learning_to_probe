"""Versioned, runtime-complete adaptive-query calibration artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from ft_anns.artifact_io import (
    artifact_filename,
    is_artifact_filename,
    write_json_atomic,
)
from ft_anns.checkpoint import RouterCheckpoint
from ft_anns.compatibility import (
    compatibility_filename,
    compatibility_slug,
    require_compatibility_metadata,
)
from ft_anns.domain import require_dataset_name
from ft_anns.observability import utc_now_iso
from ft_anns.partition import FixedPartition


_FORMAT = "adaptive-query-calibration-v3"
_DECODER = "ramps_size_matched"
_RAMPS_LOSS = "ramps"


def _finite_number(value: object, *, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Calibration Artifact {name} is invalid")
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"Calibration Artifact {name} is invalid")
    return result


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Calibration Artifact {name} is invalid")
    return value


def _checkpoint_transform(
    checkpoint: RouterCheckpoint, partition: FixedPartition
) -> dict[str, float | str]:
    loss = checkpoint.metadata.get("loss")
    if not isinstance(loss, Mapping) or loss.get("name") != _RAMPS_LOSS:
        raise ValueError("adaptive calibration requires a RAMPS checkpoint")
    transform = loss.get("target_size_transform")
    if not isinstance(transform, Mapping):
        raise ValueError("RAMPS checkpoint size transform is invalid")
    try:
        mode = transform["mode"]
        alpha = _finite_number(transform["alpha"], name="alpha")
        anchor = _finite_number(transform["anchor"], name="anchor")
    except KeyError as error:
        raise ValueError("RAMPS checkpoint size transform is invalid") from error
    if (
        mode != "anchored_power"
        or alpha <= 0
        or anchor <= 0
        or transform.get("partition_filename") != partition.filename
        or not math.isclose(
            anchor,
            float(partition.list_sizes.mean()),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("RAMPS checkpoint size transform is incompatible")
    return {"mode": mode, "alpha": alpha, "anchor": anchor}


@dataclass(frozen=True)
class CalibrationArtifact:
    """Validated frozen policy parameters and aggregate selection evidence."""

    _metadata: dict[str, Any]
    _persistence: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def payload(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    @property
    def filename(self) -> str:
        return str(self._metadata["artifact_filename"])

    @property
    def persistence(self) -> dict[str, Any]:
        return deepcopy(self._persistence)

    @classmethod
    def create(
        cls,
        *,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
        cost_lambda: float,
        m_min: int,
        m_max: int,
        recall_k: int,
        budget_ndis: float,
        observed_recall: float,
        mean_ndis: float,
        calibration_query_count: int,
        diagnostics: Mapping[str, Any],
        calibration_source: str = "base_set",
    ) -> "CalibrationArtifact":
        return cls._create(
            checkpoint=checkpoint,
            partition=partition,
            cost_lambda=cost_lambda,
            m_min=m_min,
            m_max=m_max,
            selection={
                "objective": "budget_ndis",
                "recall_k": recall_k,
                "constraint": budget_ndis,
                "observed_recall": observed_recall,
                "mean_ndis": mean_ndis,
                "tuning_query_count": calibration_query_count,
            },
            calibration_query_count=calibration_query_count,
            diagnostics=diagnostics,
            calibration_source=calibration_source,
        )

    @classmethod
    def create_target_recall(
        cls,
        *,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
        cost_lambda: float,
        m_min: int,
        m_max: int,
        recall_k: int,
        target_recall: float,
        tuning_observed_recall: float,
        observed_recall: float,
        mean_ndis: float,
        tuning_query_count: int,
        certification_query_count: int,
        recall_lower_confidence_bound: float,
        confidence_level: float,
        certified: bool,
        fallback: str,
        diagnostics: Mapping[str, Any],
        calibration_source: str = "base_set",
    ) -> "CalibrationArtifact":
        return cls._create(
            checkpoint=checkpoint,
            partition=partition,
            cost_lambda=cost_lambda,
            m_min=m_min,
            m_max=m_max,
            selection={
                "objective": "target_recall",
                "recall_k": recall_k,
                "constraint": target_recall,
                "tuning_observed_recall": tuning_observed_recall,
                "observed_recall": observed_recall,
                "mean_ndis": mean_ndis,
                "tuning_query_count": tuning_query_count,
                "certification_query_count": certification_query_count,
                "recall_lower_confidence_bound": recall_lower_confidence_bound,
                "confidence_level": confidence_level,
                "confidence_method": "hoeffding",
                "certified": certified,
                "fallback": fallback,
                "certification_status": "certified" if certified else "failed",
            },
            calibration_query_count=tuning_query_count,
            diagnostics=diagnostics,
            calibration_source=calibration_source,
        )

    @classmethod
    def _create(
        cls,
        *,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
        cost_lambda: float,
        m_min: int,
        m_max: int,
        selection: Mapping[str, Any],
        calibration_query_count: int,
        diagnostics: Mapping[str, Any],
        calibration_source: str,
    ) -> "CalibrationArtifact":
        transform = _checkpoint_transform(checkpoint, partition)
        profile = checkpoint.metadata.get("compatibility")
        if not isinstance(profile, dict):
            raise ValueError("RAMPS checkpoint Compatibility Profile is missing")
        objective = str(selection["objective"]).replace("_", "-")
        payload: dict[str, Any] = {
            "format": _FORMAT,
            "compatibility": deepcopy(profile),
            "compatibility_slug": compatibility_slug(profile),
            "decoder": _DECODER,
            "policy": {
                "cost_lambda": cost_lambda,
                "m_min": m_min,
                "m_max": m_max,
            },
            "target_size_transform": transform,
            "selection": deepcopy(dict(selection)),
            "checkpoint_filename": checkpoint.filename,
            "partition_filename": str(partition.filename),
            "dataset_name": checkpoint.metadata.get("dataset_name"),
            "dimension": partition.dimension,
            "metric": partition.metric,
            "calibration_source": calibration_source,
            "workload_manifest_filename": checkpoint.metadata.get(
                "workload_manifest_filename"
            ),
            "calibration_query_count": calibration_query_count,
            "diagnostics": deepcopy(dict(diagnostics)),
        }
        payload["artifact_filename"] = compatibility_filename(
            profile, kind=f"calibration-{objective}", extension=".json"
        )
        artifact = cls._from_payload(payload)
        artifact.validate_for(checkpoint=checkpoint, partition=partition)
        return artifact

    @classmethod
    def _from_payload(cls, payload: object) -> "CalibrationArtifact":
        if not isinstance(payload, dict):
            raise ValueError("Calibration Artifact must be a JSON object")
        metadata = deepcopy(payload)
        if metadata.get("format") != _FORMAT:
            raise ValueError("unsupported Calibration Artifact format")
        if metadata.get("decoder") != _DECODER:
            raise ValueError("Calibration Artifact decoder mismatch")
        if not is_artifact_filename(metadata.get("artifact_filename")):
            raise ValueError("Calibration Artifact filename is invalid")
        policy = metadata.get("policy")
        transform = metadata.get("target_size_transform")
        selection = metadata.get("selection")
        if not all(isinstance(value, dict) for value in (policy, transform, selection)):
            raise ValueError("Calibration Artifact contract is invalid")
        assert isinstance(policy, dict)
        assert isinstance(transform, dict)
        assert isinstance(selection, dict)
        if set(policy) != {"cost_lambda", "m_min", "m_max"}:
            raise ValueError("Calibration Artifact policy contract is invalid")
        _finite_number(policy["cost_lambda"], name="cost_lambda")
        m_min = _positive_int(policy["m_min"], name="m_min")
        m_max = _positive_int(policy["m_max"], name="m_max")
        if m_max < m_min:
            raise ValueError("Calibration Artifact probe bounds are invalid")
        if set(transform) != {"mode", "alpha", "anchor"}:
            raise ValueError("Calibration Artifact size transform is invalid")
        if transform["mode"] != "anchored_power":
            raise ValueError("Calibration Artifact size transform mode mismatch")
        if _finite_number(transform["alpha"], name="alpha") <= 0:
            raise ValueError("Calibration Artifact alpha is invalid")
        if _finite_number(transform["anchor"], name="anchor") <= 0:
            raise ValueError("Calibration Artifact anchor is invalid")
        budget_selection = {
            "objective",
            "recall_k",
            "constraint",
            "observed_recall",
            "mean_ndis",
            "tuning_query_count",
        }
        target_selection = {
            "objective",
            "recall_k",
            "constraint",
            "tuning_observed_recall",
            "observed_recall",
            "mean_ndis",
            "tuning_query_count",
            "certification_query_count",
            "recall_lower_confidence_bound",
            "confidence_level",
            "confidence_method",
            "certified",
            "fallback",
            "certification_status",
        }
        objective = selection.get("objective")
        if (
            objective == "budget_ndis" and set(selection) != budget_selection
        ) or (
            objective == "target_recall" and set(selection) != target_selection
        ) or objective not in ("budget_ndis", "target_recall"):
            raise ValueError("Calibration Artifact selection contract is invalid")
        require_compatibility_metadata(
            metadata,
            filename=str(metadata["artifact_filename"]),
            kind=f"calibration-{str(objective).replace('_', '-')}",
        )
        _positive_int(selection["recall_k"], name="recall_k")
        _positive_int(selection["tuning_query_count"], name="tuning_query_count")
        constraint = _finite_number(selection["constraint"], name="constraint")
        if constraint <= 0 or (objective == "target_recall" and constraint > 1):
            raise ValueError("Calibration Artifact constraint is invalid")
        recall = _finite_number(selection["observed_recall"], name="observed_recall")
        if recall > 1:
            raise ValueError("Calibration Artifact observed_recall is invalid")
        if _finite_number(selection["mean_ndis"], name="mean_ndis") <= 0:
            raise ValueError("Calibration Artifact mean_ndis is invalid")
        if objective == "target_recall":
            tuning_recall = _finite_number(
                selection["tuning_observed_recall"], name="tuning_observed_recall"
            )
            lower_bound = _finite_number(
                selection["recall_lower_confidence_bound"],
                name="recall_lower_confidence_bound",
            )
            confidence = _finite_number(
                selection["confidence_level"], name="confidence_level"
            )
            certification_count = _positive_int(
                selection["certification_query_count"],
                name="certification_query_count",
            )
            if tuning_recall > 1 or lower_bound > recall or confidence <= 0 or confidence >= 1:
                raise ValueError("Calibration Artifact certification statistics are invalid")
            if selection["confidence_method"] != "hoeffding":
                raise ValueError("Calibration Artifact confidence method is invalid")
            if not isinstance(selection["certified"], bool):
                raise ValueError("Calibration Artifact certification status is invalid")
            certified = lower_bound + 1e-12 >= constraint
            if selection["certified"] != certified:
                raise ValueError("Calibration Artifact certified flag is invalid")
            expected_status = "certified" if certified else "failed"
            if selection["certification_status"] != expected_status:
                raise ValueError("Calibration Artifact certification status is invalid")
            if selection["fallback"] not in ("none", "fixed_m_max"):
                raise ValueError("Calibration Artifact fallback is invalid")
            if selection["fallback"] == "fixed_m_max" and m_min != m_max:
                raise ValueError("Calibration Artifact Fixed-M fallback is invalid")
            if not certified and selection["fallback"] == "none":
                raise ValueError("uncertified Calibration Artifact requires a fallback")
            if certification_count <= 0:
                raise ValueError("Calibration Artifact certification sample is invalid")
        count = _positive_int(
            metadata.get("calibration_query_count"),
            name="calibration_query_count",
        )
        if count != selection["tuning_query_count"]:
            raise ValueError("Calibration Artifact query count mismatch")
        if not isinstance(metadata.get("diagnostics"), dict):
            raise ValueError("Calibration Artifact diagnostics are invalid")
        for name in (
            "checkpoint_filename",
            "partition_filename",
            "workload_manifest_filename",
        ):
            value = metadata.get(name)
            if not is_artifact_filename(value):
                raise ValueError(f"Calibration Artifact {name} is invalid")
        try:
            require_dataset_name(metadata.get("dataset_name"))
        except ValueError as error:
            raise ValueError("Calibration Artifact dataset name is invalid") from error
        _positive_int(metadata.get("dimension"), name="dimension")
        if metadata.get("metric") not in ("l2", "ip"):
            raise ValueError("Calibration Artifact metric is invalid")
        try:
            json.dumps(metadata, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("Calibration Artifact is not finite JSON") from error
        return cls(metadata)

    def validate_for(
        self, *, checkpoint: RouterCheckpoint, partition: FixedPartition
    ) -> None:
        metadata = self._metadata
        if metadata["dataset_name"] != checkpoint.metadata.get(
            "dataset_name"
        ):
            raise ValueError("Calibration Artifact dataset mismatch")
        if metadata["dataset_name"] != partition.dataset_name:
            raise ValueError("Calibration Artifact partition dataset mismatch")
        if metadata["dimension"] != partition.dimension:
            raise ValueError("Calibration Artifact dataset dimension mismatch")
        if metadata["metric"] != partition.metric:
            raise ValueError("Calibration Artifact metric mismatch")
        if metadata["checkpoint_filename"] != checkpoint.filename:
            raise ValueError("Calibration Artifact checkpoint filename mismatch")
        if metadata["partition_filename"] != partition.filename:
            raise ValueError("Calibration Artifact partition filename mismatch")
        if metadata["workload_manifest_filename"] != checkpoint.metadata.get(
            "workload_manifest_filename"
        ):
            raise ValueError("Calibration Artifact Workload Manifest mismatch")
        require_compatibility_metadata(
            metadata, expected=checkpoint.metadata.get("compatibility", {})
        )
        expected_transform = _checkpoint_transform(checkpoint, partition)
        if metadata["target_size_transform"] != expected_transform:
            raise ValueError("Calibration Artifact RAMPS size transform mismatch")
        policy = metadata["policy"]
        if policy["m_max"] > partition.bucket_count:
            raise ValueError("Calibration Artifact probe bounds exceed the partition")

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata["artifact_filename"] = artifact_filename(output_path)
        objective = self._metadata["selection"]["objective"].replace("_", "-")
        require_compatibility_metadata(
            self._metadata,
            filename=output_path,
            kind=f"calibration-{objective}",
        )
        save_started = time.perf_counter()
        write_json_atomic(output_path, self._metadata)
        object.__setattr__(
            self,
            "_persistence",
            {
                "save_seconds": time.perf_counter() - save_started,
                "persisted_at": utc_now_iso(),
            },
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
    ) -> "CalibrationArtifact":
        artifact_path = Path(path)
        try:
            with artifact_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Calibration Artifact is unreadable") from error
        if not isinstance(payload, dict) or payload.get(
            "artifact_filename"
        ) != artifact_filename(artifact_path):
            raise ValueError("Calibration Artifact filename mismatch")
        artifact = cls._from_payload(payload)
        artifact.validate_for(checkpoint=checkpoint, partition=partition)
        return artifact
