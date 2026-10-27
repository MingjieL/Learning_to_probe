#!/usr/bin/env python3
"""Copy trained Gain-Cost parameters into a compatible retrieval profile."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, cast

from ft_anns.adaptive_query.gain_cost import (
    GAIN_COST_PARAMETERS_FORMAT,
    PROBABILITY_GAIN_COST_PARAMETERS_FORMAT,
    PROBE_PLAN_METHODS,
    ROUTE_ORDERS,
    GainCostRetrievalSettings,
    ProbePlanMethod,
    RouteOrder,
    parse_gain_cost_retrieval_settings,
)

def configure_parameters(
    payload: dict[str, Any],
    *,
    probe_plan_method: str,
    route_order: str,
) -> dict[str, Any]:
    """Return a profile copy without changing gain semantics or learned values."""
    if payload.get("format") not in (
        "global-gain-cost-prefix-parameters-prototype-v4",
        GAIN_COST_PARAMETERS_FORMAT,
        PROBABILITY_GAIN_COST_PARAMETERS_FORMAT,
    ):
        raise ValueError("unsupported Gain-Cost parameter format")
    if probe_plan_method not in PROBE_PLAN_METHODS:
        raise ValueError(
            f"unsupported Gain-Cost probe plan method: {probe_plan_method!r}"
        )
    if route_order not in ROUTE_ORDERS:
        raise ValueError(f"unsupported Gain-Cost route order: {route_order!r}")
    if probe_plan_method == "marginal_threshold":
        metadata = payload.get("cache_metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Gain-Cost parameters contain invalid cache metadata")
        gain_power = float(metadata.get("gain_power_before_prefix_sum", 1.0))
        cost_power = float(metadata.get("cost_power_before_prefix_sum", 1.0))
        if gain_power != 1.0 or cost_power != 1.0:
            raise ValueError(
                "marginal_threshold requires gain_power=1 and cost_power=1"
            )
    original_settings = parse_gain_cost_retrieval_settings(payload)
    settings = GainCostRetrievalSettings(
        probe_plan_method=cast(ProbePlanMethod, probe_plan_method),
        route_order=cast(RouteOrder, route_order),
        gain_mode=original_settings.gain_mode,
    )
    configured = dict(payload)
    configured["format"] = (
        PROBABILITY_GAIN_COST_PARAMETERS_FORMAT
        if settings.gain_mode == "probability"
        else GAIN_COST_PARAMETERS_FORMAT
    )
    configured["retrieval"] = settings.as_dict()
    parse_gain_cost_retrieval_settings(configured)
    return configured


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--probe-plan-method", choices=PROBE_PLAN_METHODS, required=True
    )
    parser.add_argument("--route-order", choices=ROUTE_ORDERS, required=True)
    arguments = parser.parse_args()

    input_path = Path(arguments.input).resolve()
    output_path = Path(arguments.output).resolve()
    if input_path == output_path:
        parser.error("--output must differ from --input; v4 artifacts are immutable")
    with input_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("Gain-Cost parameters must be a JSON object")
    configured = configure_parameters(
        payload,
        probe_plan_method=arguments.probe_plan_method,
        route_order=arguments.route_order,
    )
    _write_json_atomic(output_path, configured)
    print(output_path)


if __name__ == "__main__":
    main()
