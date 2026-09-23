"""Freeze the sequential research decision and prevent downstream bypass."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERDICT = ROOT / "studies/direction3/RESEARCH_GATE_VERDICT_2026-09-23.json"


def test_communication_gate_cannot_open_when_upstream_gate_fails():
    data = json.loads(VERDICT.read_text(encoding="utf-8"))
    gates = data["gates"]
    required = gates["communication_system"]["required_gates"]
    expected = all(gates[name]["advance"] for name in required)
    assert gates["communication_system"]["advance"] is expected
    assert not expected


def test_gate_order_and_current_claim_boundaries_are_frozen():
    data = json.loads(VERDICT.read_text(encoding="utf-8"))
    assert data["order"] == [
        "tp_uic_contribution", "association_over_sinr", "communication_system"
    ]
    assert data["gates"]["tp_uic_contribution"]["status"] == "rejected_current_method"
    assert data["gates"]["association_over_sinr"]["status"] == "not_demonstrated"
