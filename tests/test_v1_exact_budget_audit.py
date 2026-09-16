"""Validation of the audit's objective and exhaustive comparison contract."""
import numpy as np
from tools.audit_v1_exact_budget import config, objective_grid, exact_job, interval
from isac_sim.fusion import selection_utility_from_pd
from isac_sim.fbl import blocklength_latency_s


def test_vector_objective_matches_canonical_scalar():
    cfg=config(10301,4,True)
    rng=np.random.default_rng(901)
    p0=rng.uniform(0,1,13)[:,None]
    p1=rng.uniform(0,1,9)[None,:]
    reports=rng.integers(0,5,(13,9))
    values=objective_grid(cfg,p0,p1,reports)
    expected=np.empty_like(values)
    for i in range(13):
        for j in range(9):
            expected[i,j]=selection_utility_from_pd(cfg,np.zeros(2),np.array([p0[i,0],p1[0,j]]))
            expected[i,j]-=cfg.selector.lambda_c*reports[i,j]*1e3*blocklength_latency_s(cfg)
    np.testing.assert_allclose(values,expected,atol=1e-14)


def test_exhaustive_optima_nested_budgets_and_gap_decomposition():
    rows=exact_job((17,10301,[0,1,2,4]))
    assert all(a['joint_F']<=b['joint_F']+1e-12 for a,b in zip(rows,rows[1:]))
    assert all(a['nearest_exact_F']<=b['nearest_exact_F']+1e-12 for a,b in zip(rows,rows[1:]))
    for r in rows:
        assert r['joint_feasible_pairs']>=9  # empty set under every placement pair
        assert abs(r['total_gap']-r['placement_gap']-r['selection_gap'])<1e-12
        assert r['joint_reports']<=r['budget']
        assert r['c2f_reports']<=r['budget']


def test_audit_keeps_v1_failure_and_capacity_contract():
    cfg=config(10302,2)
    assert cfg.detect.comm_error_model=='gaussian_replacement'
    assert cfg.scale.M==15 and cfg.scale.Q==10
    assert cfg.comm.mac_model=='serial' and cfg.comm.interference_model=='orthogonal'
    assert cfg.selector.max_links_per_target==6 and cfg.selector.max_total_links==60
    assert cfg.selector.max_remote_reports==2
    assert cfg.selector.lambda_c==.005
    assert not cfg.selector.require_local_anchor
    assert cfg.fusion.max_targets_per_uav<0


def test_zero_variance_interval_not_equivalence_certificate():
    assert interval([0,0,0]) is None
    assert interval([0,1,-1])[0]<0<interval([0,1,-1])[1]


def test_balanced_control_covers_targets_before_second_observation(monkeypatch):
    from types import SimpleNamespace
    from tools import audit_v1_balanced_budget as control
    from isac_sim.reporting import ReportingPlan
    cfg=config(1,0,True)
    cfg.selector.max_total_links=2
    gamma=np.ones((3,3,2)); gamma[:,:,0]=1000
    monkeypatch.setattr(control,'feasible_links_for_target',lambda *args:[(1,0),(2,0)])
    selected=control.balanced_select(cfg,None,SimpleNamespace(gamma_sense=gamma),
                                    ReportingPlan(mode='explicit',f_q=np.array([0,0])))
    assert [len(selected[q]) for q in range(2)]==[1,1]


def test_balanced_control_cannot_exceed_report_cap(monkeypatch):
    from types import SimpleNamespace
    from tools import audit_v1_balanced_budget as control
    from isac_sim.reporting import ReportingPlan
    cfg=config(1,0,True)
    monkeypatch.setattr(control,'feasible_links_for_target',lambda *args:[(0,1)])
    selected=control.balanced_select(cfg,None,SimpleNamespace(gamma_sense=np.ones((3,3,2))),
                                    ReportingPlan(mode='explicit',f_q=np.array([0,0])))
    assert sum(map(len,selected.values()))==0
