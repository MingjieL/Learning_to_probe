"""Dataset adapters and read-only metadata inspection."""

from ft_anns.datasets.base import SyntheticDataset
from ft_anns.datasets.base_gt import (
    BaseGroundTruthArtifact,
    FileVectorSource,
    build_base_ground_truth,
)
from ft_anns.datasets.registry import DatasetPreflight, dataset_spec, preflight_dataset
from ft_anns.datasets.workloads import (
    WorkloadManifest,
    load_role_ground_truth,
    prepare_workloads,
)

__all__ = [
    "BaseGroundTruthArtifact",
    "DatasetPreflight",
    "FileVectorSource",
    "SyntheticDataset",
    "WorkloadManifest",
    "build_base_ground_truth",
    "dataset_spec",
    "load_role_ground_truth",
    "preflight_dataset",
    "prepare_workloads",
]
