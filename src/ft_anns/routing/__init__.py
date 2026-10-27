"""Query-level router training interfaces."""

from ft_anns.routing.data import (
    FileBackedQueryLevelWorkload,
    QueryLevelWorkload,
    WorkloadProvenance,
    create_query_dataloader,
    load_query_level_workload,
)
from ft_anns.routing.losses import build_ramps_soft_targets, compute_routing_loss
from ft_anns.routing.model import RawQueryRouter
from ft_anns.routing.trainer import train_router
from ft_anns.routing.validation import (
    CheckpointValidationReport,
    CheckpointValidationWorkload,
    CalibrationBoundSelectionReport,
    evaluate_checkpoint,
    load_checkpoint_validation_workload,
    select_calibration_bound,
)

__all__ = [
    "QueryLevelWorkload",
    "FileBackedQueryLevelWorkload",
    "RawQueryRouter",
    "WorkloadProvenance",
    "build_ramps_soft_targets",
    "compute_routing_loss",
    "create_query_dataloader",
    "load_query_level_workload",
    "train_router",
    "CheckpointValidationReport",
    "CheckpointValidationWorkload",
    "CalibrationBoundSelectionReport",
    "evaluate_checkpoint",
    "load_checkpoint_validation_workload",
    "select_calibration_bound",
]
