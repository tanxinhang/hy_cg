import numpy as np
import pytest
from isac_sim.config import Config,apply_preset,apply_overrides,validate_config
from isac_sim.model import generate_geometry,build_base_gains,compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive
from isac_sim.power_joint import optimize_power_joint
from isac_sim.simulate import remote_report_count


def setup():
    cfg=apply_overrides(apply_preset(Config(),'target-local-v1'),{
        'scale.M':3,'scale.Q':2,'geometry.area_xy':600.,'detect.target_rcs':.1,
        'detect.comm_error_model':'gaussian_replacement','selector.score_mode':'detector_pd',
        'selector.max_total_links':4,'selector.max_links_per_target':2,'selector.max_remote_reports':2})
    rng=np.random.default_rng(123)
    geom=generate_geometry(cfg,rng);base=build_base_gains(cfg,geom,rng)
    return cfg,geom,base


def test_uniform_vector_exactly_preserves_scalar_physics():
    cfg,_,base=setup()
    a=compute_link_tables(cfg,base)
    b=compute_link_tables(apply_overrides(cfg,{'radio.rho_by_uav':(.8,.8,.8)}),base)
    for field in vars(a):np.testing.assert_array_equal(getattr(a,field),getattr(b,field))


def test_power_split_changes_both_desired_signal_and_cross_interference():
    cfg,_,base=setup()
    base.edge_mask[:]=True
    a=compute_link_tables(cfg,base)
    varied=apply_overrides(cfg,{'radio.rho_by_uav':(.95,.8,.8)})
    b=compute_link_tables(varied,base)
    assert np.all(b.gamma_sense[0,2]>=a.gamma_sense[0,2])
    assert np.any(b.gamma_sense[1,2]<a.gamma_sense[1,2])
    assert b.gamma_comm[0,2]<a.gamma_comm[0,2]
    assert b.gamma_comm[1,2]<a.gamma_comm[1,2]
    cached=compute_link_tables(varied,base,reuse_from=a)
    for field in vars(b):np.testing.assert_array_equal(getattr(b,field),getattr(cached,field))
    rho=np.array(varied.radio.rho_by_uav)
    np.testing.assert_allclose(rho*cfg.radio.P_default+(1-rho)*cfg.radio.P_default,1.)


@pytest.mark.parametrize('rho',[(.8,),(.8,1.,.8),(.8,float('nan'),.8),(.8,-.2,.8)])
def test_invalid_node_split_rejected(rho):
    cfg,_,base=setup();cfg.radio.rho_by_uav=rho
    with pytest.raises(ValueError):validate_config(cfg)
    with pytest.raises(ValueError):compute_link_tables(cfg,base)


def test_power_search_monotone_predicted_objective_and_budget_feasibility():
    cfg,geom,base=setup()
    coarse=compute_link_tables(cfg,base)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    selected=select_c2f_adaptive(cfg,base,coarse,plan)[0]
    r=optimize_power_joint(cfg,base,selected,plan,rounds=1)
    assert np.all(np.diff(r.objective_trace)>0)
    assert r.joint.objective>=r.initial.objective
    assert r.joint.cfg.radio.P_default==cfg.radio.P_default
    assert remote_report_count(r.joint.selected,r.joint.plan)<=2
    assert sum(map(len,r.joint.selected.values()))<=4
    assert cfg.radio.rho_by_uav is None
    assert r.fixed_set_power.selected==r.uniform.selected
    np.testing.assert_array_equal(r.fixed_set_power.plan.f_q,r.uniform.plan.f_q)
    assert r.fixed_set_power.objective>=r.uniform.objective


def test_truth_detector_rebuilds_with_chosen_node_powers(monkeypatch):
    from isac_sim import simulate
    cfg,geom,base=setup()
    original=compute_link_tables(cfg,base)
    cfg=apply_overrides(cfg,{'radio.rho_by_uav':(.2,.5,.95)})
    coarse=compute_link_tables(cfg,base)
    fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    cached=select_c2f_adaptive(cfg,base,coarse,plan)
    observed=[]
    detector=simulate.evaluate_detection
    def record(c,tables,*args,**kwargs):
        observed.append(tables.gamma_sense.copy())
        return detector(c,tables,*args,**kwargs)
    monkeypatch.setattr(simulate,'evaluate_detection',record)
    simulate.run_method_on_trial(cfg,base,coarse,'proposed_c2f_adaptive_pd',0,
        c2f_tables=fine,plan=plan,eval_base=base,eval_tables=original,
        cached_adaptive_pd=cached)
    assert len(observed)==1
    np.testing.assert_array_equal(observed[0],fine.gamma_sense)
