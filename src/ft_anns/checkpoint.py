"""Versioned router checkpoint persistence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Never

import torch
from torch import nn

from ft_anns.artifact_io import (
    artifact_filename,
    is_artifact_filename,
    write_json_atomic,
)
from ft_anns.config import RouterConfig, RampsConfig
from ft_anns.compatibility import (
    normalize_compatibility_profile,
    require_compatibility_metadata,
)
from ft_anns.domain import require_dataset_name
from ft_anns.partition import FixedPartition
from ft_anns.partition.fixed_ivf import PolicyPartition

if TYPE_CHECKING:
    from ft_anns.routing.model import RawQueryRouter


_METADATA_FILE = "metadata.json"
_WEIGHTS_FILE = "weights.pt"
# RAMPS: Rank-Aggregated Mass Per Size.
_RAMPS_LOSS = "ramps"
_LEGACY_CHECKPOINT_FORMAT = "router-checkpoint-v4"
_MAX_NORMALIZED_RAMPS_CHECKPOINT_FORMAT = "router-checkpoint-v5"


def router_checkpoint_format(loss_name: str) -> str:
    """Return the checkpoint schema required by a training loss."""
    if loss_name == _RAMPS_LOSS:
        return _MAX_NORMALIZED_RAMPS_CHECKPOINT_FORMAT
    return _LEGACY_CHECKPOINT_FORMAT


def _validate_ramps_contract(
    metadata: dict[str, Any], partition: FixedPartition | PolicyPartition
) -> None:
    loss = metadata.get("loss")
    if not isinstance(loss, dict) or loss.get("name") != _RAMPS_LOSS:
        return
    try:
        resolved_config = metadata["resolved_train_config"]
        if (
            not isinstance(resolved_config, dict)
            or resolved_config.get("loss") != _RAMPS_LOSS
        ):
            raise ValueError("resolved loss mismatch")
        ramps = RampsConfig.model_validate(resolved_config["ramps"])
    except (KeyError, ValueError, TypeError) as error:
        raise ValueError(
            "router checkpoint RAMPS size normalization is invalid"
        ) from error
    normalization = loss.get("target_size_normalization")
    parameters = loss.get("parameters")
    if (
        parameters != {"rank_weight": ramps.rank_weight}
        or not isinstance(normalization, dict)
        or set(normalization) != {"mode", "max_size", "partition_filename"}
        or normalization.get("mode") != "max"
        or normalization.get("partition_filename") != partition.filename
    ):
        raise ValueError("router checkpoint RAMPS size normalization is incompatible")
    max_size = normalization.get("max_size")
    if (
        isinstance(max_size, bool)
        or not isinstance(max_size, int)
        or max_size != int(partition.list_sizes.max())
    ):
        raise ValueError("router checkpoint RAMPS size normalization is incompatible")


@dataclass(frozen=True)
class RouterCheckpoint:
    path: Path
    model: nn.Module
    metadata: dict[str, Any]
    metrics: tuple[dict[str, Any], ...]

    @property
    def filename(self) -> str:
        return str(self.metadata["checkpoint_filename"])

    def validate_for(self, partition: FixedPartition | PolicyPartition) -> None:
        """Validate filename, dataset, structure, and RAMPS metadata together."""
        if not is_artifact_filename(self.metadata.get("checkpoint_filename")):
            raise ValueError("router checkpoint filename is invalid")
        if self.metadata.get("partition_filename") != partition.filename:
            raise ValueError("router checkpoint Fixed Partition filename mismatch")
        try:
            checkpoint_dataset = require_dataset_name(
                self.metadata.get("dataset_name"), allow_synthetic=True
            )
        except ValueError as error:
            raise ValueError("router checkpoint dataset name is invalid") from error
        if checkpoint_dataset != partition.dataset_name:
            raise ValueError("router checkpoint Fixed Partition dataset mismatch")
        if self.metadata.get("format") in (
            _LEGACY_CHECKPOINT_FORMAT,
            _MAX_NORMALIZED_RAMPS_CHECKPOINT_FORMAT,
        ):
            loss = self.metadata.get("loss")
            loss_name = loss.get("name") if isinstance(loss, dict) else None
            expected_profile: dict[str, object] = {
                "dataset_name": checkpoint_dataset,
                "ground_truth_k": self.metadata.get("ground_truth_k"),
                "bucket_count": partition.bucket_count,
                "kmeans_niter": partition.kmeans_niter,
                "partition_build_seed": partition.build_seed,
                "loss": loss_name,
            }
            profile = require_compatibility_metadata(
                self.metadata, expected=expected_profile
            )
            resolved_config = self.metadata.get("resolved_train_config")
            if isinstance(resolved_config, dict) and resolved_config.get(
                "compatibility"
            ) is not None:
                raw_config_profile = resolved_config["compatibility"]
                try:
                    if not isinstance(raw_config_profile, Mapping):
                        raise ValueError("resolved profile is not an object")
                    config_profile = normalize_compatibility_profile(
                        raw_config_profile
                    )
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        "router checkpoint config Compatibility Profile mismatch"
                    ) from error
                if config_profile != profile:
                    raise ValueError(
                        "router checkpoint config Compatibility Profile mismatch"
                    )
                if self.path.parent.name != self.metadata["compatibility_slug"]:
                    raise ValueError(
                        "router checkpoint run-directory compatibility slug mismatch"
                    )
        checkpoint_base = self.metadata.get("base_filename")
        if (
            checkpoint_base is not None
            and partition.base_filename is not None
            and checkpoint_base != partition.base_filename
        ):
            raise ValueError("router checkpoint Fixed Partition base mismatch")
        if (
            getattr(self.model, "input_dimension", None) != partition.dimension
            or getattr(self.model, "bucket_count", None) != partition.bucket_count
        ):
            raise ValueError("router checkpoint model structure is incompatible")
        _validate_ramps_contract(self.metadata, partition)

    def require_adaptive_query_support(self, partition: FixedPartition | PolicyPartition) -> Never:
        """Reject Adaptive Query until it has a max-normalized score contract."""
        self.validate_for(partition)
        raise ValueError(
            "Adaptive Query is not yet supported for max-normalized RAMPS checkpoints"
        )

    @classmethod
    def save(
        cls,
        path: str | Path,
        *,
        model: RawQueryRouter,
        metadata: dict[str, Any],
        metrics: list[dict[str, Any]],
    ) -> "RouterCheckpoint":
        checkpoint_path = Path(path)
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        state_dict = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        complete_metadata = dict(metadata)
        complete_metadata["checkpoint_filename"] = artifact_filename(
            checkpoint_path
        )
        temporary_weights = checkpoint_path / f".{_WEIGHTS_FILE}.tmp"
        torch.save(state_dict, temporary_weights)
        temporary_weights.replace(checkpoint_path / _WEIGHTS_FILE)
        write_json_atomic(checkpoint_path / _METADATA_FILE, complete_metadata)
        model.load_state_dict(state_dict)
        model.cpu().eval()
        return cls(
            path=checkpoint_path,
            model=model,
            metadata=complete_metadata,
            metrics=tuple(dict(entry) for entry in metrics),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        partition: FixedPartition | PolicyPartition,
    ) -> "RouterCheckpoint":
        checkpoint_path = Path(path)
        with (checkpoint_path / _METADATA_FILE).open(encoding="utf-8") as stream:
            metadata = json.load(stream)
        checkpoint_format = metadata.get("format")
        loss = metadata.get("loss")
        loss_name = loss.get("name") if isinstance(loss, dict) else None
        expected_format = router_checkpoint_format(str(loss_name))
        if checkpoint_format != expected_format:
            raise ValueError("unsupported router checkpoint format")
        if metadata.get("partition_filename") != partition.filename:
            raise ValueError("router checkpoint Fixed Partition filename mismatch")
        if metadata.get("dataset_name") != partition.dataset_name:
            raise ValueError("router checkpoint Fixed Partition dataset mismatch")
        if metadata.get("checkpoint_filename") != artifact_filename(
            checkpoint_path
        ):
            raise ValueError("router checkpoint filename mismatch")
        try:
            state_dict = torch.load(
                checkpoint_path / _WEIGHTS_FILE,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, RuntimeError) as error:
            raise ValueError("router checkpoint weights are unreadable") from error
        if not isinstance(state_dict, dict) or not all(
            isinstance(value, torch.Tensor) for value in state_dict.values()
        ):
            raise ValueError("router checkpoint weights have an invalid payload")
        model_metadata = metadata["model"]
        from ft_anns.routing.model import RawQueryRouter

        model = RawQueryRouter(
            input_dimension=int(model_metadata["input_dimension"]),
            bucket_count=int(model_metadata["bucket_count"]),
            config=RouterConfig.model_validate(
                {
                    "hidden_dimensions": model_metadata["hidden_dimensions"],
                    "activation": model_metadata["activation"],
                }
            ),
        )
        if model.identity != model_metadata.get("identity"):
            raise ValueError("router checkpoint model identity mismatch")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError as error:
            raise ValueError("router checkpoint weights do not match its model") from error
        model.eval()
        metrics = metadata.get("metrics", [])
        if not isinstance(metrics, list):
            raise ValueError("router checkpoint metrics are invalid")
        checkpoint = cls(
            path=checkpoint_path,
            model=model,
            metadata=metadata,
            metrics=tuple(dict(entry) for entry in metrics),
        )
        checkpoint.validate_for(partition)
        return checkpoint
