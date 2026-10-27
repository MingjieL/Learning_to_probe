"""Calibration, policy, retrieval, and evaluation boundaries."""

from ft_anns.adaptive_query.artifacts import CalibrationArtifact
from ft_anns.adaptive_query.calibration import (
    CalibrationWorkload,
    LambdaFit,
    PolicyCostFitWorkload,
    calibrate_policy,
    calibrate_policy_library,
    certify_policy_library,
    classify_tuning_candidate_costs,
    generate_policy_budget_widths,
    group_equivalent_probe_plans,
    load_calibration_workload,
    load_policy_cost_fit_workload,
    plan_policy_tuning_costs,
    recall_lower_confidence_bound,
    select_budget_ndis_candidate,
    select_target_recall_candidate,
    select_validation_widths,
)
from ft_anns.adaptive_query.evaluation import (
    EvaluationReport,
    EvaluationTimingTrace,
    compute_tie_aware_recall_at_k,
    evaluate_results,
)
from ft_anns.adaptive_query.policy import build_probe_plan
from ft_anns.adaptive_query.policy_library import (
    PolicyLibrary,
    PolicyLibraryCertificationReport,
)
from ft_anns.adaptive_query.retrieval import (
    RetrievalResult,
    RetrievalTimingTrace,
    retrieve_adaptive,
    retrieve_decoder_fixed,
    retrieve_fixed,
)

__all__ = [
    "CalibrationArtifact",
    "CalibrationWorkload",
    "EvaluationReport",
    "EvaluationTimingTrace",
    "LambdaFit",
    "PolicyCostFitWorkload",
    "PolicyLibrary",
    "PolicyLibraryCertificationReport",
    "RetrievalResult",
    "RetrievalTimingTrace",
    "build_probe_plan",
    "calibrate_policy",
    "calibrate_policy_library",
    "certify_policy_library",
    "classify_tuning_candidate_costs",
    "compute_tie_aware_recall_at_k",
    "evaluate_results",
    "generate_policy_budget_widths",
    "group_equivalent_probe_plans",
    "load_calibration_workload",
    "load_policy_cost_fit_workload",
    "plan_policy_tuning_costs",
    "recall_lower_confidence_bound",
    "retrieve_adaptive",
    "retrieve_decoder_fixed",
    "retrieve_fixed",
    "select_budget_ndis_candidate",
    "select_target_recall_candidate",
    "select_validation_widths",
]
