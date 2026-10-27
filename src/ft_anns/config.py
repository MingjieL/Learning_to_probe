"""Strict, workflow-specific configuration models."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ft_anns.compatibility import compatibility_filename, compatibility_slug
from ft_anns.domain import (
    BaseGroundTruthDatasetName,
    DatasetFormat,
    DatasetName,
    Metric,
    RoutingRanking,
    SearchExecution,
)

LossName: TypeAlias = Literal["ce", "bce", "ramps"]
CalibrationDevice: TypeAlias = Literal["auto", "cpu", "cuda"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ExactSearchExecutionConfig(StrictModel):
    device: Literal["cpu", "cuda", "cuda_all"] = "cpu"
    gpu_ids: list[Annotated[int, Field(ge=0)]] = Field(default_factory=list)
    index_add_batch_size: Annotated[int, Field(gt=0)] = 100_000
    query_batch_size: Annotated[int, Field(gt=0)] = 10_000

    @model_validator(mode="after")
    def validate_execution_adapter(self) -> Self:
        if self.device == "cpu" and self.gpu_ids:
            raise ValueError("gpu_ids are only allowed for CUDA execution adapters")
        if self.device == "cuda" and len(self.gpu_ids) > 1:
            raise ValueError("device='cuda' accepts at most one GPU ID")
        return self


class BaseGroundTruthConfig(ExactSearchExecutionConfig):
    dataset: BaseGroundTruthDatasetName
    base_path: str
    format: DatasetFormat
    dimension: Annotated[int, Field(gt=0)]
    metric: Metric
    scope: Literal["all", "selected_ids"] = "all"
    selected_ids_path: str | None = None
    existing_neighbor_ids_path: str | None = None
    k: Annotated[int, Field(gt=0)]
    save_distances: bool = False
    output_dtype: Literal["int32", "int64"] = "int64"
    output_path: str
    resume: bool = True

    @model_validator(mode="after")
    def validate_selected_id_source(self) -> Self:
        if self.scope == "selected_ids" and self.selected_ids_path is None:
            raise ValueError("selected_ids_path is required when scope='selected_ids'")
        if self.scope == "all" and self.selected_ids_path is not None:
            raise ValueError("selected_ids_path is only allowed when scope='selected_ids'")
        if self.existing_neighbor_ids_path is not None:
            if self.save_distances:
                raise ValueError(
                    "existing neighbor IDs cannot provide exact distances"
                )
            if self.device != "cpu" or self.gpu_ids:
                raise ValueError(
                    "existing neighbor-ID reuse requires device='cpu'"
                )
        if not self.output_path:
            raise ValueError("output_path must not be empty")
        return self


class RoleConfig(StrictModel):
    count: Annotated[int, Field(gt=0)] | None = None
    ids: list[Annotated[int, Field(ge=0)]] | None = None

    @model_validator(mode="after")
    def validate_single_id_source(self) -> Self:
        if (self.count is None) == (self.ids is None):
            raise ValueError("a workload role requires exactly one of count or ids")
        if self.ids is not None and not self.ids:
            raise ValueError("a workload role ids list must not be empty")
        return self


class RoleDefinitions(StrictModel):
    router_train: RoleConfig = Field(
        default_factory=lambda: RoleConfig(count=100_000)
    )
    checkpoint_validation: RoleConfig = Field(
        default_factory=lambda: RoleConfig(count=10_000)
    )
    policy_cost_fit: RoleConfig = Field(
        default_factory=lambda: RoleConfig(count=100_000)
    )
    policy_tuning: RoleConfig = Field(
        default_factory=lambda: RoleConfig(count=100_000)
    )
    policy_certification: RoleConfig | None = Field(
        default_factory=lambda: RoleConfig(count=10_000)
    )


class SamplingConfig(StrictModel):
    strategy: Literal[
        "range",
        "strided",
        "random_without_replacement",
        "explicit_ids",
        "router_complement_with_policy_subsets",
    ] = "random_without_replacement"
    seed: int = 20260809
    start: Annotated[int, Field(ge=0)] | None = None
    stride: Annotated[int, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def validate_strategy_parameters(self) -> Self:
        if self.strategy == "range":
            if self.start is None:
                raise ValueError("start is required for range sampling")
            if self.stride is not None:
                raise ValueError("stride is only allowed for strided sampling")
        elif self.strategy == "strided":
            if self.start is None or self.stride is None:
                raise ValueError("start and stride are required for strided sampling")
        elif self.start is not None or self.stride is not None:
            raise ValueError(
                "start and stride are only allowed for range or strided sampling"
            )
        return self


class WorkloadGroundTruthConfig(ExactSearchExecutionConfig):
    k: Annotated[int, Field(gt=0)] = 100
    metric: Metric
    mode: Literal[
        "reuse_or_exact_selected", "import_ivecs_prefix"
    ] = "reuse_or_exact_selected"
    exclude_self_by_id: bool = False
    artifact_path: str | None = None
    source_path: str | None = None

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.mode == "import_ivecs_prefix":
            if self.source_path is None or self.artifact_path is not None:
                raise ValueError(
                    "import_ivecs_prefix requires source_path and forbids artifact_path"
                )
        elif self.source_path is not None:
            raise ValueError(
                "source_path is only allowed when mode='import_ivecs_prefix'"
            )
        return self


class WorkloadConfig(StrictModel):
    dataset: DatasetName
    base_path: str
    base_filename: str
    query_path: str | None = None
    query_format: DatasetFormat | None = None
    source: Literal["base_set", "query_file"] = "base_set"
    roles: RoleDefinitions = Field(default_factory=RoleDefinitions)
    sampling: SamplingConfig = SamplingConfig()
    ground_truth: WorkloadGroundTruthConfig
    manifest_output_path: str
    materialized_ground_truth_output_path: str

    @model_validator(mode="after")
    def validate_source_and_sampling(self) -> Self:
        if self.source == "query_file" and self.query_path is None:
            raise ValueError("query_path is required when source='query_file'")
        if self.source == "base_set" and self.query_path is not None:
            raise ValueError("query_path is only allowed when source='query_file'")
        if self.source == "base_set" and self.query_format is not None:
            raise ValueError("query_format is only allowed when source='query_file'")
        if self.ground_truth.exclude_self_by_id:
            raise ValueError(
                "workloads require exclude_self_by_id=false; base-to-base ground "
                "truth keeps the query's own corpus ID"
            )

        roles = (
            self.roles.router_train,
            self.roles.checkpoint_validation,
            self.roles.policy_cost_fit,
            self.roles.policy_tuning,
            self.roles.policy_certification,
        )
        defined_roles = tuple(role for role in roles if role is not None)
        if self.sampling.strategy == "explicit_ids":
            if any(role.ids is None for role in defined_roles):
                raise ValueError("explicit_ids sampling requires role ids")
        elif any(role.count is None for role in defined_roles):
            raise ValueError(
                f"{self.sampling.strategy} sampling requires role count values"
            )
        if self.sampling.strategy == "router_complement_with_policy_subsets":
            if self.source != "query_file":
                raise ValueError(
                    "router-complement sampling requires source='query_file'"
                )
            if self.roles.policy_certification is None:
                raise ValueError(
                    "router-complement sampling requires policy_certification"
                )
        return self


class TrainingDataConfig(StrictModel):
    mode: Literal["in_memory", "file_backed"] = "in_memory"


class PartitionConfig(StrictModel):
    artifact_path: str | None = None
    bucket_count: Annotated[int, Field(gt=0)] | None = None
    build_seed: int = 1234
    kmeans_niter: Annotated[int, Field(gt=0)] = 10


class RouterConfig(StrictModel):
    hidden_dimensions: list[Annotated[int, Field(gt=0)]]
    activation: Literal["relu", "swish"] = "swish"


class RampsConfig(StrictModel):
    """Rank-Aggregated Mass Per Size target settings."""

    rank_weight: Literal["log"] = "log"


class WorkflowCompatibilityConfig(StrictModel):
    """Explicit downstream expectations for referenced workflow artifacts."""

    dataset_name: DatasetName
    ground_truth_k: Annotated[int, Field(gt=0)]
    bucket_count: Annotated[int, Field(gt=0)]
    kmeans_niter: Annotated[int, Field(gt=0)]
    partition_build_seed: int
    loss: LossName
    # Legacy Adaptive Query configs retain Power Alpha until that workflow is
    # redesigned for max-normalized RAMPS. New training configs must omit it.
    ramps_alpha: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def validate_legacy_adaptive_loss_parameters(self) -> Self:
        if self.loss != "ramps" and self.ramps_alpha is not None:
            raise ValueError(
                "ramps_alpha is allowed only for legacy Adaptive Query RAMPS profiles"
            )
        compatibility_slug(self.model_dump(mode="json", exclude_none=True))
        return self


class OptimizerConfig(StrictModel):
    name: Literal["adam"] = "adam"
    learning_rate: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)] = 0.0


class ValidationConfig(StrictModel):
    fixed_bucket_count: Annotated[int, Field(gt=0)] = 10
    ranking: Literal["raw_logits"] = "raw_logits"
    ann_recall_ks: list[Annotated[int, Field(gt=0)]] = Field(
        default_factory=lambda: [10]
    )
    checkpoint_selection_recall_k: Annotated[int, Field(gt=0)] = 10

    @model_validator(mode="after")
    def validate_measured_selection_metric(self) -> Self:
        if not self.ann_recall_ks:
            raise ValueError("ann_recall_ks must be non-empty")
        if len(set(self.ann_recall_ks)) != len(self.ann_recall_ks):
            raise ValueError("ann_recall_ks must be unique")
        if self.checkpoint_selection_recall_k not in self.ann_recall_ks:
            raise ValueError(
                "checkpoint_selection_recall_k must be measured in ann_recall_ks"
            )
        return self


class EarlyStoppingConfig(StrictModel):
    patience_epochs: Annotated[int, Field(gt=0)] | None = None


class TrainConfig(StrictModel):
    dataset: DatasetName
    workload_manifest_path: str
    partition: PartitionConfig
    compatibility: WorkflowCompatibilityConfig | None = None
    router: RouterConfig
    loss: LossName
    ramps: RampsConfig | None = None
    batch_size: Annotated[int, Field(gt=0)] = 512
    training_data: TrainingDataConfig | None = None
    epochs: Annotated[int, Field(gt=0)] = 10
    optimizer: OptimizerConfig
    validation: ValidationConfig = ValidationConfig()
    early_stopping: EarlyStoppingConfig = EarlyStoppingConfig()
    seed: int = 20260809
    device: Literal["cpu", "cuda"] = "cpu"
    output_dir: str

    @model_validator(mode="after")
    def validate_loss_specific_config(self) -> Self:
        is_ramps = self.loss == "ramps"
        if is_ramps and self.ramps is None:
            raise ValueError("ramps settings are required for the RAMPS loss")
        if not is_ramps and self.ramps is not None:
            raise ValueError("ramps settings are only allowed for the RAMPS loss")
        if self.compatibility is not None:
            expected = self.compatibility
            if expected.dataset_name != self.dataset:
                raise ValueError("Compatibility Profile dataset does not match config")
            if (
                self.partition.bucket_count is not None
                and expected.bucket_count != self.partition.bucket_count
            ):
                raise ValueError(
                    "Compatibility Profile bucket count does not match config"
                )
            if expected.kmeans_niter != self.partition.kmeans_niter:
                raise ValueError(
                    "Compatibility Profile K-Means niter does not match config"
                )
            if expected.partition_build_seed != self.partition.build_seed:
                raise ValueError(
                    "Compatibility Profile partition build seed does not match config"
                )
            if expected.loss != self.loss:
                raise ValueError("Compatibility Profile loss does not match config")
            if expected.ramps_alpha is not None:
                raise ValueError(
                    "new RAMPS training Compatibility Profiles must omit ramps_alpha"
                )
        return self


class CertificationConfig(StrictModel):
    confidence_level: Annotated[float, Field(gt=0, lt=1)] = 0.95


class CalibrationBoundSelectionConfig(StrictModel):
    target_recall: Annotated[float, Field(gt=0, le=1)] = 0.995
    fixed_bucket_counts: list[Annotated[int, Field(gt=0)]]

    @model_validator(mode="after")
    def validate_scan_points(self) -> Self:
        if not self.fixed_bucket_counts:
            raise ValueError("fixed_bucket_counts must be non-empty")
        if self.fixed_bucket_counts != sorted(set(self.fixed_bucket_counts)):
            raise ValueError("fixed_bucket_counts must be unique and increasing")
        return self


class PolicyLibraryWidthSelectionConfig(StrictModel):
    lower_recall_target: Annotated[float, Field(gt=0, le=1)]
    upper_recall_target: Annotated[float, Field(gt=0, le=1)]
    maximum_fixed_bucket_count: Annotated[int, Field(gt=0)]

    @model_validator(mode="after")
    def validate_targets(self) -> Self:
        if self.lower_recall_target > self.upper_recall_target:
            raise ValueError(
                "lower_recall_target must not exceed upper_recall_target"
            )
        return self


class AutoLambdaSearchConfig(StrictModel):
    mode: Literal["auto"] = "auto"
    iterations: Annotated[int, Field(gt=0)] = 64
    bracket_max: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 1e12
    local_factors: list[
        Annotated[float, Field(gt=0, allow_inf_nan=False)]
    ] = [0.9, 0.95, 1.0, 1.05, 1.1]

    @model_validator(mode="after")
    def validate_local_factors(self) -> Self:
        if not self.local_factors:
            raise ValueError("auto lambda local_factors must be non-empty")
        if len(set(self.local_factors)) != len(self.local_factors):
            raise ValueError("auto lambda local_factors must be unique")
        if 1.0 not in self.local_factors:
            raise ValueError("auto lambda local_factors must include 1.0")
        return self


class ManualLambdaSearchConfig(StrictModel):
    mode: Literal["manual"]
    cost_lambda_candidates: list[
        Annotated[float, Field(ge=0, allow_inf_nan=False)]
    ]

    @model_validator(mode="after")
    def validate_candidates(self) -> Self:
        if not self.cost_lambda_candidates:
            raise ValueError("cost_lambda_candidates must be non-empty")
        if len(set(self.cost_lambda_candidates)) != len(
            self.cost_lambda_candidates
        ):
            raise ValueError("cost_lambda_candidates must be unique")
        return self


LambdaSearchConfig: TypeAlias = Annotated[
    AutoLambdaSearchConfig | ManualLambdaSearchConfig,
    Field(discriminator="mode"),
]


class CalibrationConfig(StrictModel):
    device: CalibrationDevice = "auto"
    compatibility: WorkflowCompatibilityConfig | None = None
    workload_manifest_path: str
    cost_fit_role: Literal["policy_cost_fit"] = "policy_cost_fit"
    tuning_role: Literal["policy_tuning"] = "policy_tuning"
    certification_role: Literal["policy_certification"] = "policy_certification"
    decoder: Literal["ramps_size_matched"] = "ramps_size_matched"
    lambda_search: LambdaSearchConfig = Field(
        default_factory=AutoLambdaSearchConfig
    )
    m_min: Annotated[int, Field(gt=0)] = 1
    m_max: Annotated[int, Field(gt=0)] | None = None
    calibration_bound_selection: CalibrationBoundSelectionConfig | None = None
    objective: Literal["budget_ndis", "target_recall"]
    constraint: Annotated[float, Field(gt=0)]
    recall_k: Annotated[int, Field(gt=0)] = 100
    fallback: Literal["fail", "fixed_m_max"] = "fail"
    certification: CertificationConfig = CertificationConfig()
    certification_cost_margin: Annotated[
        float, Field(ge=0, lt=1, allow_inf_nan=False)
    ] = 0.0
    retrieval_batch_size: Annotated[int, Field(gt=0)] = 1_000
    search_execution: SearchExecution = "grouped"
    output_path: str

    @model_validator(mode="after")
    def validate_candidate_grid_and_probe_bounds(self) -> Self:
        if self.m_max is not None and self.m_min > self.m_max:
            raise ValueError("m_min must not exceed m_max")
        if (self.m_max is None) == (self.calibration_bound_selection is None):
            raise ValueError(
                "calibration requires exactly one of m_max or "
                "calibration_bound_selection"
            )
        if self.objective == "target_recall" and self.constraint > 1:
            raise ValueError("target_recall constraint must be at most 1")
        if self.objective == "target_recall" and self.lambda_search.mode != "manual":
            raise ValueError("target_recall requires manual lambda search")
        return self

    @property
    def cost_lambda_candidates(self) -> list[float]:
        if self.lambda_search.mode == "manual":
            return list(self.lambda_search.cost_lambda_candidates)
        return []


class PolicyBudgetConfig(StrictModel):
    policy_id: Annotated[
        str, Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    ]
    reference_fixed_bucket_count: Annotated[int, Field(gt=0)]


class PolicyBudgetGenerationConfig(StrictModel):
    mode: Literal["adaptive_ndis"] = "adaptive_ndis"
    target_policy_count: Annotated[int, Field(gt=0)] = 12
    maximum_policy_count: Annotated[int, Field(gt=0, le=16)] = 16
    spacing: Literal["log_ndis"] = "log_ndis"
    include_calibration_bound: Literal[True] = True
    include_width_one_budget: bool = False

    @model_validator(mode="after")
    def validate_policy_count(self) -> Self:
        if self.target_policy_count > self.maximum_policy_count:
            raise ValueError(
                "target_policy_count must not exceed maximum_policy_count"
            )
        return self


class PolicyLibraryConfig(StrictModel):
    device: CalibrationDevice = "auto"
    compatibility: WorkflowCompatibilityConfig | None = None
    workload_manifest_path: str
    cost_fit_role: Literal["policy_cost_fit"] = "policy_cost_fit"
    tuning_role: Literal["policy_tuning"] = "policy_tuning"
    certification_role: Literal["policy_certification"] = "policy_certification"
    decoder: Literal["ramps_size_matched"] = "ramps_size_matched"
    policy_budgets: list[PolicyBudgetConfig] | None = None
    policy_budget_generation: PolicyBudgetGenerationConfig | None = None
    calibration_bound_selection: PolicyLibraryWidthSelectionConfig
    lambda_search: LambdaSearchConfig = Field(
        default_factory=AutoLambdaSearchConfig
    )
    m_min: Annotated[int, Field(gt=0)] = 1
    m_max_candidates: list[Annotated[int, Field(gt=0)]] | None = None
    recall_k: Annotated[int, Field(gt=0)] = 100
    budget_mode: Literal["approximate", "strict"] = "approximate"
    certification_cost_margin: Annotated[
        float, Field(ge=0, lt=1, allow_inf_nan=False)
    ]
    certification_cost_tolerance: Annotated[
        float, Field(ge=0, lt=1, allow_inf_nan=False)
    ]
    retrieval_batch_size: Annotated[int, Field(gt=0)] = 1_000
    search_execution: SearchExecution = "grouped"
    output_path: str

    @model_validator(mode="before")
    @classmethod
    def default_budget_generation(cls, value: object) -> object:
        if isinstance(value, Mapping):
            payload = dict(value)
            if (
                payload.get("policy_budgets") is None
                and "policy_budget_generation" not in payload
            ):
                payload["policy_budget_generation"] = {}
            budget_mode = payload.get("budget_mode", "approximate")
            if "certification_cost_margin" not in payload:
                payload["certification_cost_margin"] = (
                    0.005 if budget_mode == "strict" else 0.0
                )
            if "certification_cost_tolerance" not in payload:
                payload["certification_cost_tolerance"] = (
                    0.0 if budget_mode == "strict" else 0.005
                )
            return payload
        return value

    @model_validator(mode="after")
    def validate_library_grid(self) -> Self:
        if (self.policy_budgets is None) == (
            self.policy_budget_generation is None
        ):
            raise ValueError(
                "Policy Library requires exactly one of policy_budgets or "
                "policy_budget_generation"
            )
        if self.policy_budgets is not None:
            if not self.policy_budgets:
                raise ValueError("policy_budgets must be non-empty")
            policy_ids = [entry.policy_id for entry in self.policy_budgets]
            if len(set(policy_ids)) != len(policy_ids):
                raise ValueError("Policy Library policy IDs must be unique")
            fixed_counts = [
                entry.reference_fixed_bucket_count for entry in self.policy_budgets
            ]
            if len(set(fixed_counts)) != len(fixed_counts):
                raise ValueError(
                    "Policy Library reference Fixed bucket counts must be unique"
                )
        if self.m_max_candidates is not None and (
            not self.m_max_candidates
            or len(set(self.m_max_candidates)) != len(self.m_max_candidates)
        ):
            raise ValueError("m_max_candidates must be non-empty and unique")
        if self.m_max_candidates is not None and any(
            value < self.m_min for value in self.m_max_candidates
        ):
            raise ValueError("every m_max candidate must be at least m_min")
        return self

    @property
    def cost_lambda_candidates(self) -> list[float]:
        if self.lambda_search.mode == "manual":
            return list(self.lambda_search.cost_lambda_candidates)
        return []


class EvaluationConfig(StrictModel):
    dataset: DatasetName
    query_path: str
    ground_truth_path: str
    top_k: Annotated[int, Field(gt=0)]
    recall_ks: list[Annotated[int, Field(gt=0)]]
    fixed_m_points: list[Annotated[int, Field(gt=0)]] = []
    retrieval_batch_size: Annotated[int, Field(gt=0)] = 1_000
    output_path: str


ConfigModel: TypeAlias = (
    BaseGroundTruthConfig
    | WorkloadConfig
    | TrainConfig
    | CalibrationConfig
    | PolicyLibraryConfig
    | EvaluationConfig
)

_CONFIG_TYPES: dict[str, type[ConfigModel]] = {
    "base_ground_truth": BaseGroundTruthConfig,
    "workload": WorkloadConfig,
    "train": TrainConfig,
    "calibration": CalibrationConfig,
    "policy_library": PolicyLibraryConfig,
    "evaluation": EvaluationConfig,
}

_PROHIBITED_PARTITION_FIELDS = {
    "dissolve",
    "gt_bucket_reassign",
    "merge",
    "recycled_bucket",
    "repartition",
    "sinkhorn",
    "split",
    "structural_update",
    "structural_update_enabled",
    "uot",
    "upd_interval",
    "upd_patient",
}


def _prohibited_field_paths(value: object, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if any(
                key_text == term or key_text.startswith(f"{term}_")
                for term in _PROHIBITED_PARTITION_FIELDS
            ):
                paths.append(path)
            paths.extend(_prohibited_field_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_prohibited_field_paths(child, f"{prefix}[{index}]"))
    return paths


def load_config(
    kind: str, source: Mapping[str, object] | str | Path
) -> ConfigModel:
    """Load one strict workflow config from a mapping or JSON file."""
    try:
        config_type = _CONFIG_TYPES[kind]
    except KeyError as error:
        allowed = ", ".join(sorted(_CONFIG_TYPES))
        raise ValueError(f"Unknown config kind {kind!r}; expected one of: {allowed}") from error

    source_path: Path | None = None
    if isinstance(source, (str, Path)):
        source_path = Path(source)
        with source_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
    else:
        payload = dict(source)
    prohibited = _prohibited_field_paths(payload)
    if prohibited:
        raise ValueError(
            "prohibited fixed-partition config field(s): " + ", ".join(prohibited)
        )
    config = config_type.model_validate(payload)
    if source_path is not None:
        expected_filename: str
        if isinstance(config, BaseGroundTruthConfig):
            profile = {
                "dataset_name": config.dataset,
                "ground_truth_k": config.k,
            }
            expected_filename = compatibility_filename(
                profile,
                kind="query-gt" if config.scope == "selected_ids" else "base-gt",
                extension=".json",
            )
        elif isinstance(config, WorkloadConfig):
            profile = {
                "dataset_name": config.dataset,
                "ground_truth_k": config.ground_truth.k,
            }
            expected_filename = compatibility_filename(
                profile, kind="workloads", extension=".json"
            )
        elif isinstance(config, TrainConfig):
            if config.compatibility is None:
                raise ValueError(
                    "file-backed training config requires a Compatibility Profile"
                )
            expected_filename = (
                compatibility_slug(
                    config.compatibility.model_dump(mode="json", exclude_none=True)
                )
                + ".json"
            )
        elif isinstance(config, CalibrationConfig):
            if config.compatibility is None:
                raise ValueError(
                    "file-backed calibration config requires a Compatibility Profile"
                )
            expected_filename = compatibility_filename(
                config.compatibility.model_dump(mode="json", exclude_none=True),
                kind=f"calibration-{config.objective.replace('_', '-')}",
                extension=".json",
            )
        elif isinstance(config, PolicyLibraryConfig):
            if config.compatibility is None:
                raise ValueError(
                    "file-backed Policy Library config requires a Compatibility Profile"
                )
            expected_filename = compatibility_filename(
                config.compatibility.model_dump(mode="json", exclude_none=True),
                kind="policy-library",
                extension=".json",
            )
        else:
            assert isinstance(config, EvaluationConfig)
            expected_filename = compatibility_filename(
                {"dataset_name": config.dataset, "top_k": config.top_k},
                kind="evaluation",
                extension=".json",
            )
        if source_path.name != expected_filename:
            raise ValueError(
                "config compatibility filename mismatch: expected "
                f"{expected_filename!r}, found {source_path.name!r}"
            )
        expected_outputs: tuple[tuple[str, str], ...]
        if isinstance(config, BaseGroundTruthConfig):
            expected_outputs = (
                (
                    config.output_path,
                    compatibility_filename(
                        {
                            "dataset_name": config.dataset,
                            "ground_truth_k": config.k,
                        },
                        kind=(
                            "query-gt"
                            if config.scope == "selected_ids"
                            else "base-gt"
                        ),
                    ),
                ),
            )
        elif isinstance(config, WorkloadConfig):
            profile = {
                "dataset_name": config.dataset,
                "ground_truth_k": config.ground_truth.k,
            }
            expected_outputs = (
                (
                    config.manifest_output_path,
                    compatibility_filename(
                        profile, kind="workloads", extension=".json"
                    ),
                ),
                (
                    config.materialized_ground_truth_output_path,
                    compatibility_filename(profile, kind="query-gt"),
                ),
            )
        elif isinstance(config, TrainConfig):
            assert config.compatibility is not None
            expected_outputs = (
                (
                    config.output_dir,
                    compatibility_slug(
                        config.compatibility.model_dump(
                            mode="json", exclude_none=True
                        )
                    ),
                ),
            )
        elif isinstance(config, (CalibrationConfig, PolicyLibraryConfig)):
            expected_outputs = ((config.output_path, expected_filename),)
        else:
            assert isinstance(config, EvaluationConfig)
            expected_outputs = ((config.output_path, expected_filename),)
        for output, expected_output_filename in expected_outputs:
            actual_output_filename = Path(output).name
            if actual_output_filename != expected_output_filename:
                raise ValueError(
                    "config output compatibility filename mismatch: expected "
                    f"{expected_output_filename!r}, found "
                    f"{actual_output_filename!r}"
                )
    return config
