"""Packaged, source-independent behavior oracle fixtures."""

from __future__ import annotations

from importlib.resources import files
import json
from typing import Any


def load_reference_fixture() -> dict[str, Any]:
    """Load the deterministic Ticket 01 behavior fixture."""
    resource = files(__package__).joinpath("reference_v1.json")
    return json.loads(resource.read_text(encoding="utf-8"))


__all__ = ["load_reference_fixture"]
