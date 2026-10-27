"""Versioned multi-budget RAMPS adaptive Policy Library artifacts."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

from ft_anns.adaptive_query.artifacts import CalibrationArtifact
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


_FORMAT = "adaptive-query-policy-library-v5"
_CERTIFICATION_FORMAT = "policy-library-certification-v1"


def _checkpoint_compatibility(checkpoint: RouterCheckpoint) -> dict[str, Any]:
    return {
        name: deepcopy(checkpoint.metadata.get(name))
        for name in (
            "format",
            "dataset_name",
            "base_filename",
            "partition_filename",
            "workload_manifest_filename",
            "model",
            "loss",
            "resolved_train_config",
            "compatibility",
            "compatibility_slug",
        )
    }


def _partition_compatibility(partition: FixedPartition) -> dict[str, Any]:
    return {
        "dataset_name": partition.dataset_name,
        "dimension": partition.dimension,
        "metric": partition.metric,
        "bucket_count": partition.bucket_count,
        "list_size_anchor": float(partition.list_sizes.mean()),
    }


@dataclass(frozen=True)
class PolicyLibraryEntry:
    reference_fixed_bucket_count: int
    cost_fit_target_mean_ndis: float
    tuning_target_mean_ndis: float
    calibration_artifact: CalibrationArtifact
    certification_status: str = "not_run"

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "reference_fixed_bucket_count": self.reference_fixed_bucket_count,
            "cost_fit_target_mean_ndis": self.cost_fit_target_mean_ndis,
            "tuning_target_mean_ndis": self.tuning_target_mean_ndis,
            "calibration_artifact": self.calibration_artifact.payload,
            "certification_status": self.certification_status,
        }

    @classmethod
    def from_payload(cls, payload: object) -> "PolicyLibraryEntry":
        if not isinstance(payload, dict) or set(payload) != {
            "reference_fixed_bucket_count",
            "cost_fit_target_mean_ndis",
            "tuning_target_mean_ndis",
            "calibration_artifact",
            "certification_status",
        }:
            raise ValueError("Policy Library entry contract is invalid")
        fixed_count = payload["reference_fixed_bucket_count"]
        numeric_fields = (
            "cost_fit_target_mean_ndis",
            "tuning_target_mean_ndis",
        )
        if (
            isinstance(fixed_count, bool)
            or not isinstance(fixed_count, int)
            or fixed_count <= 0
            or any(
                isinstance(payload[name], bool)
                or not isinstance(payload[name], (int, float))
                or not math.isfinite(float(payload[name]))
                for name in numeric_fields
            )
            or any(float(payload[name]) <= 0 for name in numeric_fields)
            or payload["certification_status"] != "not_run"
        ):
            raise ValueError("Policy Library entry budget is invalid")
        artifact = CalibrationArtifact._from_payload(payload["calibration_artifact"])
        selection = artifact.payload["selection"]
        if (
            selection["objective"] != "budget_ndis"
            or float(selection["constraint"])
            != float(payload["tuning_target_mean_ndis"])
            or float(selection["mean_ndis"])
            > float(payload["tuning_target_mean_ndis"]) + 1e-9
        ):
            raise ValueError("Policy Library entry violates its NDis budget")
        return cls(
            reference_fixed_bucket_count=fixed_count,
            cost_fit_target_mean_ndis=float(payload["cost_fit_target_mean_ndis"]),
            tuning_target_mean_ndis=float(payload["tuning_target_mean_ndis"]),
            calibration_artifact=artifact,
            certification_status="not_run",
        )


@dataclass(frozen=True)
class _PolicyLibraryCore:
    """Named frozen adaptive operating points for several Fixed-query budgets."""

    _metadata: dict[str, Any]
    _persistence: dict[str, Any] = field(
        default_factory=dict, compare=False, repr=False
    )

    @property
    def payload(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    @property
    def filename(self) -> str:
        return str(self._metadata["library_filename"])

    @property
    def persistence(self) -> dict[str, Any]:
        return deepcopy(self._persistence)

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._metadata["entries"]))

    @classmethod
    def create(
        cls,
        *,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
        entries: Mapping[str, PolicyLibraryEntry],
        calibration_contract: Mapping[str, Any],
        observability: Mapping[str, Any] | None = None,
    ) -> "PolicyLibrary":
        if not entries:
            raise ValueError("Policy Library requires at least one entry")
        profile = checkpoint.metadata.get("compatibility")
        if not isinstance(profile, dict):
            raise ValueError("RAMPS checkpoint Compatibility Profile is missing")
        payload: dict[str, Any] = {
            "format": _FORMAT,
            "compatibility": deepcopy(profile),
            "compatibility_slug": compatibility_slug(profile),
            "decoder": "ramps_size_matched",
            "checkpoint_filename": checkpoint.filename,
            "partition_filename": partition.filename,
            "workload_manifest_filename": checkpoint.metadata.get(
                "workload_manifest_filename"
            ),
            "dataset_name": checkpoint.metadata.get("dataset_name"),
            "power_alpha": checkpoint.metadata.get("loss", {})
            .get("target_size_transform", {})
            .get("alpha"),
            "checkpoint_compatibility": _checkpoint_compatibility(checkpoint),
            "partition_compatibility": _partition_compatibility(partition),
            "calibration_contract": deepcopy(dict(calibration_contract)),
            "certification_status": "not_run",
            "entries": {
                policy_id: entry.payload for policy_id, entry in entries.items()
            },
            "observability": deepcopy(dict(observability or {})),
        }
        payload["library_filename"] = compatibility_filename(
            profile, kind="policy-library", extension=".json"
        )
        library = cls._from_payload(payload)
        library.validate_for(checkpoint=checkpoint, partition=partition)
        return library


@dataclass(frozen=True)
class PolicyLibraryCertificationReport:
    """Independent cost and Recall evidence for one frozen Policy Library."""

    _metadata: dict[str, Any]

    @property
    def payload(self) -> dict[str, Any]:
        return deepcopy(self._metadata)

    @property
    def status(self) -> str:
        return str(self._metadata["status"])

    @classmethod
    def create(
        cls,
        *,
        library: PolicyLibrary,
        library_path: str | Path,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
        certification_query_ids: list[int],
        entries: Mapping[str, Mapping[str, Any]],
        timing: Mapping[str, Any] | None = None,
    ) -> "PolicyLibraryCertificationReport":
        path = Path(library_path)
        library.validate_for(checkpoint=checkpoint, partition=partition)
        if artifact_filename(path) != library.filename:
            raise ValueError("Certification Policy Library filename mismatch")
        if not certification_query_ids:
            raise ValueError("Certification report requires query IDs")
        if set(entries) != set(library.policy_ids):
            raise ValueError("Certification report must cover every library policy")
        statuses = [str(entry.get("status")) for entry in entries.values()]
        if not statuses or any(
            status not in ("passed", "failed") for status in statuses
        ):
            raise ValueError("Certification report entry status is invalid")
        status = (
            "passed"
            if all(value == "passed" for value in statuses)
            else "failed"
            if all(value == "failed" for value in statuses)
            else "partial"
        )
        contract = library.payload["calibration_contract"]
        payload = {
            "format": _CERTIFICATION_FORMAT,
            "report_filename": f"{path.stem}.certification.json",
            "library_filename": library.filename,
            "library_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dataset_name": library.payload["dataset_name"],
            "checkpoint_filename": checkpoint.filename,
            "partition_filename": partition.filename,
            "checkpoint_compatibility": deepcopy(
                library.payload["checkpoint_compatibility"]
            ),
            "partition_compatibility": deepcopy(
                library.payload["partition_compatibility"]
            ),
            "workload_manifest_filename": library.payload[
                "workload_manifest_filename"
            ],
            "certification_role": contract["workload_roles"]["certification"],
            "certification_query_ids": list(certification_query_ids),
            "budget_mode": contract["budget_mode"],
            "tuning_headroom": contract["tuning_headroom"],
            "relative_tolerance": contract["certification_tolerance"],
            "recall_k": contract["recall_k"],
            "entries": {key: deepcopy(dict(value)) for key, value in entries.items()},
            "status": status,
            "timing": deepcopy(dict(timing or {})),
            "certified_at": utc_now_iso(),
        }
        return cls._from_payload(payload)

    @classmethod
    def _from_payload(
        cls, payload: object
    ) -> "PolicyLibraryCertificationReport":
        if (
            not isinstance(payload, dict)
            or payload.get("format") != _CERTIFICATION_FORMAT
        ):
            raise ValueError("unsupported Policy Library Certification report format")
        metadata = deepcopy(payload)
        if not is_artifact_filename(metadata.get("report_filename")):
            raise ValueError("Certification report filename is invalid")
        if not is_artifact_filename(metadata.get("library_filename")):
            raise ValueError("Certification report library filename is invalid")
        digest = metadata.get("library_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("Certification report library digest is invalid")
        if metadata.get("budget_mode") not in ("approximate", "strict"):
            raise ValueError("Certification report budget mode is invalid")
        if not isinstance(metadata.get("checkpoint_compatibility"), dict):
            raise ValueError("Certification report checkpoint compatibility is invalid")
        if not isinstance(metadata.get("partition_compatibility"), dict):
            raise ValueError("Certification report partition compatibility is invalid")
        tolerance = metadata.get("relative_tolerance")
        headroom = metadata.get("tuning_headroom")
        recall_k = metadata.get("recall_k")
        if (
            isinstance(tolerance, bool)
            or not isinstance(tolerance, (int, float))
            or not math.isfinite(float(tolerance))
            or not 0 <= float(tolerance) < 1
            or isinstance(headroom, bool)
            or not isinstance(headroom, (int, float))
            or not math.isfinite(float(headroom))
            or not 0 <= float(headroom) < 1
            or isinstance(recall_k, bool)
            or not isinstance(recall_k, int)
            or recall_k < 1
        ):
            raise ValueError("Certification report budget contract is invalid")
        if metadata.get("status") not in ("passed", "partial", "failed"):
            raise ValueError("Certification report status is invalid")
        entries = metadata.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Certification report entries are invalid")
        statuses: list[str] = []
        for policy_id, entry in entries.items():
            if not isinstance(policy_id, str) or not isinstance(entry, dict):
                raise ValueError("Certification report entry is invalid")
            required = {
                "reference_fixed_bucket_count",
                "target_mean_ndis",
                "relative_tolerance",
                "allowed_mean_ndis",
                "observed_mean_ndis",
                "observed_relative_excess",
                "recall",
                "status",
            }
            if set(entry) != required or entry["status"] not in ("passed", "failed"):
                raise ValueError("Certification report entry contract is invalid")
            statuses.append(str(entry["status"]))
            numeric = required - {"status", "reference_fixed_bucket_count"}
            if (
                isinstance(entry["reference_fixed_bucket_count"], bool)
                or not isinstance(entry["reference_fixed_bucket_count"], int)
                or entry["reference_fixed_bucket_count"] < 1
                or any(
                    isinstance(entry[name], bool)
                    or not isinstance(entry[name], (int, float))
                    or not math.isfinite(float(entry[name]))
                    for name in numeric
                )
                or not 0 <= float(entry["recall"]) <= 1
            ):
                raise ValueError("Certification report entry values are invalid")
            target = float(entry["target_mean_ndis"])
            allowed = float(entry["allowed_mean_ndis"])
            observed = float(entry["observed_mean_ndis"])
            if (
                target <= 0
                or observed <= 0
                or abs(float(entry["relative_tolerance"]) - float(tolerance))
                > 1e-12
                or abs(allowed - target * (1.0 + float(tolerance))) > 1e-9
                or abs(
                    float(entry["observed_relative_excess"])
                    - (observed - target) / target
                )
                > 1e-9
                or entry["status"]
                != ("passed" if observed <= allowed + 1e-9 else "failed")
            ):
                raise ValueError("Certification report entry evidence is inconsistent")
        expected_status = (
            "passed"
            if all(value == "passed" for value in statuses)
            else "failed"
            if all(value == "failed" for value in statuses)
            else "partial"
        )
        if metadata["status"] != expected_status:
            raise ValueError("Certification report aggregate status is invalid")
        query_ids = metadata.get("certification_query_ids")
        if (
            not isinstance(query_ids, list)
            or not query_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in query_ids
            )
        ):
            raise ValueError("Certification report query IDs are invalid")
        return cls(metadata)

    def validate_for_library(
        self, *, library: PolicyLibrary, library_path: str | Path
    ) -> None:
        path = Path(library_path)
        library_payload = library.payload
        contract = library_payload["calibration_contract"]
        expected_semantics = {
            "dataset_name": library_payload["dataset_name"],
            "checkpoint_filename": library_payload["checkpoint_filename"],
            "partition_filename": library_payload["partition_filename"],
            "workload_manifest_filename": library_payload[
                "workload_manifest_filename"
            ],
            "certification_role": contract["workload_roles"]["certification"],
            "certification_query_ids": contract["certification_query_ids"],
            "budget_mode": contract["budget_mode"],
            "tuning_headroom": contract["tuning_headroom"],
            "relative_tolerance": contract["certification_tolerance"],
            "recall_k": contract["recall_k"],
        }
        report_policy_ids = set(self._metadata["entries"])
        reference_widths_match = report_policy_ids == set(
            library.policy_ids
        ) and all(
            int(
                self._metadata["entries"][policy_id][
                    "reference_fixed_bucket_count"
                ]
            )
            == int(
                library_payload["entries"][policy_id][
                    "reference_fixed_bucket_count"
                ]
            )
            for policy_id in report_policy_ids
        )
        if (
            self._metadata["library_filename"] != library.filename
            or self._metadata["library_sha256"]
            != hashlib.sha256(path.read_bytes()).hexdigest()
            or report_policy_ids != set(library.policy_ids)
            or not reference_widths_match
            or self._metadata["checkpoint_compatibility"]
            != library_payload["checkpoint_compatibility"]
            or self._metadata["partition_compatibility"]
            != library_payload["partition_compatibility"]
            or any(
                self._metadata.get(name) != expected
                for name, expected in expected_semantics.items()
            )
        ):
            raise ValueError("Certification report does not bind this Policy Library")

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata["report_filename"] = artifact_filename(output_path)
        self._from_payload(self._metadata)
        write_json_atomic(output_path, self._metadata)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        library: PolicyLibrary,
        library_path: str | Path,
    ) -> "PolicyLibraryCertificationReport":
        report_path = Path(path)
        try:
            with report_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Certification report is unreadable") from error
        if not isinstance(payload, dict) or payload.get(
            "report_filename"
        ) != artifact_filename(report_path):
            raise ValueError("Certification report filename mismatch")
        report = cls._from_payload(payload)
        report.validate_for_library(library=library, library_path=library_path)
        return report


@dataclass(frozen=True)
class PolicyLibrary(_PolicyLibraryCore):
    """Complete persistence and compatibility behavior for a Policy Library."""

    @classmethod
    def _from_payload(cls, payload: object) -> "PolicyLibrary":
        if not isinstance(payload, dict):
            raise ValueError("Policy Library must be a JSON object")
        metadata = deepcopy(payload)
        if metadata.get("format") != _FORMAT:
            raise ValueError("unsupported Policy Library format")
        if metadata.get("decoder") != "ramps_size_matched":
            raise ValueError("Policy Library decoder mismatch")
        if metadata.get("certification_status") != "not_run":
            raise ValueError("Policy Library certification status is invalid")
        if not is_artifact_filename(metadata.get("library_filename")):
            raise ValueError("Policy Library filename is invalid")
        require_compatibility_metadata(
            metadata,
            filename=str(metadata["library_filename"]),
            kind="policy-library",
        )
        entries = metadata.get("entries")
        if not isinstance(entries, dict) or not entries:
            raise ValueError("Policy Library entries are invalid")
        for policy_id, entry in entries.items():
            if not isinstance(policy_id, str) or not policy_id:
                raise ValueError("Policy Library policy ID is invalid")
            PolicyLibraryEntry.from_payload(entry)
        if not isinstance(metadata.get("observability"), dict):
            raise ValueError("Policy Library observability is invalid")
        contract = metadata.get("calibration_contract")
        if not isinstance(contract, dict):
            raise ValueError("Policy Library calibration contract is invalid")
        roles = contract.get("workload_roles")
        certification_query_ids = contract.get("certification_query_ids")
        if (
            not isinstance(roles, dict)
            or roles.get("certification") != "policy_certification"
            or not isinstance(certification_query_ids, list)
            or not certification_query_ids
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in certification_query_ids
            )
            or len(set(certification_query_ids)) != len(certification_query_ids)
            or contract.get("budget_mode") not in ("approximate", "strict")
            or isinstance(contract.get("recall_k"), bool)
            or not isinstance(contract.get("recall_k"), int)
            or int(contract["recall_k"]) < 1
            or any(
                isinstance(contract.get(name), bool)
                or not isinstance(contract.get(name), (int, float))
                or not math.isfinite(float(contract[name]))
                or not 0 <= float(contract[name]) < 1
                for name in ("tuning_headroom", "certification_tolerance")
            )
        ):
            raise ValueError("Policy Library calibration contract is invalid")
        if not isinstance(metadata.get("checkpoint_compatibility"), dict):
            raise ValueError("Policy Library checkpoint compatibility is invalid")
        if not isinstance(metadata.get("partition_compatibility"), dict):
            raise ValueError("Policy Library partition compatibility is invalid")
        for name in (
            "checkpoint_filename",
            "partition_filename",
            "workload_manifest_filename",
        ):
            value = metadata.get(name)
            if not is_artifact_filename(value):
                raise ValueError(f"Policy Library {name} is invalid")
        try:
            require_dataset_name(metadata.get("dataset_name"))
        except ValueError as error:
            raise ValueError("Policy Library dataset name is invalid") from error
        alpha = metadata.get("power_alpha")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or alpha <= 0
        ):
            raise ValueError("Policy Library Power Alpha is invalid")
        return cls(metadata)

    def validate_for(
        self, *, checkpoint: RouterCheckpoint, partition: FixedPartition
    ) -> None:
        metadata = self._metadata
        expected = {
            "checkpoint_filename": checkpoint.filename,
            "partition_filename": partition.filename,
            "workload_manifest_filename": checkpoint.metadata.get(
                "workload_manifest_filename"
            ),
            "dataset_name": checkpoint.metadata.get("dataset_name"),
            "power_alpha": checkpoint.metadata.get("loss", {})
            .get("target_size_transform", {})
            .get("alpha"),
        }
        if any(metadata.get(name) != value for name, value in expected.items()):
            raise ValueError("Policy Library metadata is incompatible")
        if metadata.get("checkpoint_compatibility") != _checkpoint_compatibility(
            checkpoint
        ):
            raise ValueError("Policy Library checkpoint parameters are incompatible")
        if metadata.get("partition_compatibility") != _partition_compatibility(
            partition
        ):
            raise ValueError("Policy Library partition parameters are incompatible")
        if metadata.get("dataset_name") != partition.dataset_name:
            raise ValueError("Policy Library partition dataset is incompatible")
        require_compatibility_metadata(
            metadata, expected=checkpoint.metadata.get("compatibility", {})
        )
        for payload in metadata["entries"].values():
            entry = PolicyLibraryEntry.from_payload(payload)
            if entry.reference_fixed_bucket_count > partition.bucket_count:
                raise ValueError(
                    "Policy Library Decoder Fixed-Width budget exceeds partition"
                )
            entry.calibration_artifact.validate_for(
                checkpoint=checkpoint, partition=partition
            )

    def resolve(
        self,
        policy_id: str,
        *,
        checkpoint: RouterCheckpoint,
        partition: FixedPartition,
    ) -> CalibrationArtifact:
        self.validate_for(checkpoint=checkpoint, partition=partition)
        try:
            entry = self._metadata["entries"][policy_id]
        except KeyError as error:
            raise ValueError(f"unknown Policy Library policy ID: {policy_id}") from error
        artifact = PolicyLibraryEntry.from_payload(entry).calibration_artifact
        artifact.validate_for(checkpoint=checkpoint, partition=partition)
        return artifact

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata["library_filename"] = artifact_filename(output_path)
        require_compatibility_metadata(
            self._metadata,
            filename=output_path,
            kind="policy-library",
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
    ) -> "PolicyLibrary":
        library_path = Path(path)
        try:
            with library_path.open(encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("Policy Library is unreadable") from error
        if not isinstance(payload, dict) or payload.get(
            "library_filename"
        ) != artifact_filename(library_path):
            raise ValueError("Policy Library filename mismatch")
        library = cls._from_payload(payload)
        library.validate_for(checkpoint=checkpoint, partition=partition)
        return library
