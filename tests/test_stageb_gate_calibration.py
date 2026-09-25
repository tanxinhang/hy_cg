from studies.direction4.scripts.calibrate_stageb_state_gate import freeze_gate


def test_freeze_gate_uses_only_deployable_diagnostics():
    records = [{
        "cross_fold_distance_m": 10.0,
        "folds": [
            {"gain_over_zero": 1.0, "audit_state_error_m": 0.0},
            {"gain_over_zero": 2.0, "audit_state_error_m": 999.0},
        ],
    }, {
        "cross_fold_distance_m": 20.0,
        "folds": [
            {"gain_over_zero": 3.0, "audit_state_error_m": 999.0},
            {"gain_over_zero": 4.0, "audit_state_error_m": 0.0},
        ],
    }]
    first = freeze_gate(records)
    for record in records:
        for fold in record["folds"]:
            fold["audit_state_error_m"] *= -12345.0
    assert freeze_gate(records) == first
    assert first["accept_rule"]["gain_over_zero_strictly_gt"] == 4.0
    assert first["accept_rule"]["cross_fold_distance_m_le"] == 20.0
    assert first["accept_rule"]["receiver_support_ge"] == 4
    assert first["accept_rule"]["boundary_hit_must_equal"] is False
