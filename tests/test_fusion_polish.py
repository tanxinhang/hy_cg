from itertools import product
from unittest.mock import patch

import numpy as np
import pytest

from isac_sim.config import Config, apply_preset
from isac_sim.fusion_polish import minimize_fixed_set_reports
from isac_sim.model import build_base_gains, compute_link_tables, generate_geometry
from isac_sim.reporting import ReportingPlan
from isac_sim.selection import DEFAULT_METHODS
from isac_sim.simulate import run_one_trial


def fixture():
    cfg = apply_preset(Config(), "target-local-v1")
    cfg.scale.M, cfg.scale.Q = 4, 2
    cfg.dd.use_otfs_bin_validity = False
    rng = np.random.default_rng(77)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    tables = compute_link_tables(cfg, base)
    base.edge_mask[:] = True
    tables.feasible_comm[:] = True
    selected = {0: [(0, 1), (2, 1), (1, 2)], 1: [(0, 3), (1, 3), (3, 2)]}
    plan = ReportingPlan("explicit", np.array([0, 0]))
    return cfg, base, tables, selected, plan


@pytest.mark.parametrize("seed", range(20))
def test_matches_exhaustive_assignments_with_nonmonotone_pd(seed):
    cfg, base, tables, selected, plan = fixture()
    scores = np.random.default_rng(seed).uniform(0, 1, (2, 4))

    def pd(cfg, tables, q, links, *, plan, base):
        return scores[q, plan.f_q[q]]

    with patch("isac_sim.fusion_polish.predicted_pd_for_links", side_effect=pd):
        result = minimize_fixed_set_reports(cfg, base, tables, selected, plan)
    feasible_costs = []
    for fs in product(range(4), repeat=2):
        if all(scores[q, fs[q]] >= scores[q, 0] for q in range(2)):
            feasible_costs.append(sum(j != fs[q] for q in selected for _, j in selected[q]))
    assert result.reports_after.sum() == min(feasible_costs)
    assert np.all(result.predicted_after >= result.predicted_before)
    np.testing.assert_array_equal(plan.f_q, [0, 0])
    assert np.all(result.reports_after >= result.report_lower_bound)


def test_rejects_cheapest_infeasible_report_destination():
    cfg, base, tables, selected, plan = fixture()
    # q=0's most frequent receiver is 1, but its remaining receiver 2
    # cannot report to 1. The transpose is deliberately left feasible.
    tables.feasible_comm[1, 2] = False
    with patch("isac_sim.fusion_polish.predicted_pd_for_links", return_value=0.9):
        result = minimize_fixed_set_reports(cfg, base, tables, selected, plan)
    assert result.plan.f_q[0] == 2
    assert result.reports_after[0] == 2


def test_honors_local_cap_and_preserves_empty_targets():
    cfg, base, tables, selected, plan = fixture()
    cfg.selector.max_local_observations_per_target = 1
    selected[1] = []
    with patch("isac_sim.fusion_polish.predicted_pd_for_links", return_value=0.9):
        result = minimize_fixed_set_reports(cfg, base, tables, selected, plan)
    assert result.plan.f_q[0] == 2
    assert result.plan.f_q[1] == 0
    assert result.reports_after[1] == 0


@pytest.mark.parametrize("kind", ["capacity", "interference", "anchor"])
def test_rejects_coupled_models(kind):
    cfg, base, tables, selected, plan = fixture()
    if kind == "capacity":
        cfg.fusion.max_targets_per_uav = 1
    elif kind == "interference":
        cfg.comm.interference_model = "active_set"
    else:
        cfg.selector.require_local_anchor = True
    with pytest.raises(ValueError):
        minimize_fixed_set_reports(cfg, base, tables, selected, plan)


def test_real_pipeline_preserves_observations_fine_work_and_pd_floor():
    name = "proposed_c2f_adaptive_pd_fusion_polish"
    assert name not in DEFAULT_METHODS
    cfg = apply_preset(Config(), "target-local-v1")
    cfg.scale.M, cfg.scale.Q = 5, 3
    for trial in range(3):
        runs = run_one_trial(cfg, trial, ["proposed_c2f_adaptive_pd", name])
        old, new = runs.values()
        cert = new.fusion_polish_certificate
        assert new.selected_links == old.selected_links
        assert new.fine_eval_c2f == old.fine_eval_c2f
        assert new.overhead_bits <= old.overhead_bits
        assert np.all(cert.predicted_after >= cert.predicted_before)
        # Changing the roster must not change the frozen V1 result.
        alone = run_one_trial(cfg, trial, ["proposed_c2f_adaptive_pd"])["proposed_c2f_adaptive_pd"]
        assert alone.selected_links == old.selected_links
        np.testing.assert_array_equal(alone.detected_per_target, old.detected_per_target)
