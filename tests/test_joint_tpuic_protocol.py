"""System-level semantic gates for the TP-UIC/cooperation experiment."""

from types import SimpleNamespace

from tools.run_joint_tpuic_coordination import (
    FORMAL_CAPABILITY_SOURCE,
    FORMAL_RETENTION_SOURCE,
    capability_protocol_error,
    select_receiver_control,
)

import numpy as np


def args(**overrides):
    values = {
        "capability_source": FORMAL_CAPABILITY_SOURCE,
        "capability_retention_source": FORMAL_RETENTION_SOURCE,
        "capability_reps": 1,
        "allow_oracle_planning": False,
        "heldout_capability": True,
        "risk_secondary_rounds": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_formal_defaults_are_belief_only_and_held_out():
    assert FORMAL_CAPABILITY_SOURCE == "predicted"
    assert FORMAL_RETENTION_SOURCE == "predicted_risk"
    assert capability_protocol_error(args()) is None


def test_certificate_free_protocol_is_all_or_nothing_and_uses_no_risk_certificate():
    assert capability_protocol_error(args(
        capability_source="none", capability_retention_source="none"
    )) is None
    error = capability_protocol_error(args(
        capability_source="none",
        capability_retention_source="predicted_risk",
    ))
    assert error is not None
    assert "both capability sources" in error
    error = capability_protocol_error(args(
        capability_source="none",
        capability_retention_source="none",
        risk_secondary_rounds=1,
    ))
    assert error is not None
    assert "cannot use prior-quantile" in error


def test_target_feedback_is_allowed_only_as_a_paired_aggregate_signal():
    assert capability_protocol_error(args(
        capability_source="target_feedback",
        capability_retention_source="target_feedback",
    )) is None
    error = capability_protocol_error(args(
        capability_source="target_feedback",
        capability_retention_source="predicted_risk",
    ))
    assert error is not None
    assert "target-level feedback" in error


def test_receiver_control_uses_exact_maxmin_then_service_then_smaller_mu():
    def state(pd, mu):
        return SimpleNamespace(
            belief_pd=np.asarray(pd, dtype=float),
            cfg=SimpleNamespace(
                cancellation=SimpleNamespace(soft_protection_mu=float(mu))
            ),
        )

    candidates = [
        state([0.5, 0.9], 10.0),
        state([0.6, 0.6], 1.0),
        state([0.6, 0.7], 0.1),
    ]
    assert select_receiver_control(candidates, weak_req=0.8) is candidates[2]


def test_receiver_target_control_requires_paired_sources_and_grid():
    assert capability_protocol_error(args(
        capability_source="controlled_state",
        capability_retention_source="controlled_state",
        cell_soft_mu_grid="0.1,1,10",
    )) is None
    error = capability_protocol_error(args(
        capability_source="controlled_state",
        capability_retention_source="predicted_risk",
        cell_soft_mu_grid="0.1,1,10",
    ))
    assert "receiver-target control" in error


def test_truth_conditioned_planning_requires_explicit_oracle_label():
    error = capability_protocol_error(args(capability_source="measured_model"))
    assert error is not None
    assert "truth-conditioned" in error

    error = capability_protocol_error(
        args(capability_retention_source="measured")
    )
    assert error is not None
    assert "truth-conditioned" in error


def test_one_draw_is_not_accepted_as_a_measured_quantile_certificate():
    error = capability_protocol_error(
        args(
            capability_source="measured_model",
            capability_retention_source="measured",
            allow_oracle_planning=True,
            capability_reps=1,
        )
    )
    assert error is not None
    assert "one-sample quantile" in error


def test_planning_and_evaluation_capabilities_must_be_independent():
    error = capability_protocol_error(args(heldout_capability=False))
    assert error is not None
    assert "must be independent" in error
