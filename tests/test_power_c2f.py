import numpy as np
import pytest
from tests.test_power_joint import setup
from isac_sim.config import apply_overrides
from isac_sim.model import compute_link_tables
from isac_sim.power_tables import PowerTableCache
from isac_sim.power_c2f import optimize_power_c2f
from isac_sim.power_joint import optimize_power_joint
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive
from isac_sim.joint_polish import improve_joint_selection
from isac_sim.simulate import remote_report_count


@pytest.mark.parametrize('fractional',[False,True])
def test_cached_backend_matches_physics_and_does_not_mutate(fractional):
    cfg,_,base=setup();cfg.dd.enable_dd_fractional_penalty=fractional
    cache=PowerTableCache(cfg,base,max_cached=2)
    for rho in [(.2,.5,.95),(.95,.2,.8),(.8,.8,.8)]:
        c=apply_overrides(cfg,{'radio.rho_by_uav':rho})
        coarse=cache(c,base);before=coarse.gamma_sense.copy()
        for gain in [None,base.eta_fine]:
            a=cache(c,base,dd_gain=gain);b=compute_link_tables(c,base,dd_gain=gain)
            for k in vars(b):np.testing.assert_allclose(getattr(a,k),getattr(b,k),rtol=1e-12,atol=1e-12,err_msg=k)
        np.testing.assert_array_equal(coarse.gamma_sense,before)
    assert len(cache.cache)==2
    with pytest.raises(ValueError):cache(apply_overrides(cfg,{'radio.P_default':2.}),base)


def test_integrated_budget_monotonicity_and_fine_objective():
    cfg,geom,base=setup();coarse=compute_link_tables(cfg,base)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    selected=select_c2f_adaptive(cfg,base,coarse,plan)[0]
    result=optimize_power_c2f(cfg,base,selected,plan,rounds=1)
    s=result.joint
    assert np.all(np.diff(result.objective_trace)>0)
    assert remote_report_count(s.selected,s.plan)<=2
    assert sum(map(len,s.selected.values()))<=4
    fine=compute_link_tables(s.cfg,base,dd_gain=base.eta_fine)
    exact=improve_joint_selection(s.cfg,base,s.coarse,fine,s.selected,s.plan,passes=0)
    assert s.objective==pytest.approx(exact.objective_after,abs=1e-12)
    assert result.counters['fine_power_checks']<=result.counters['coarse_power_checks']
    assert cfg.radio.rho_by_uav is None


def test_cached_baseline_preserves_search_result():
    cfg,geom,base=setup();coarse=compute_link_tables(cfg,base)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    selected=select_c2f_adaptive(cfg,base,coarse,plan)[0]
    old=optimize_power_joint(cfg,base,selected,plan,rounds=1)
    new=optimize_power_joint(cfg,base,selected,plan,rounds=1,table_builder=PowerTableCache(cfg,base))
    assert old.joint.selected==new.joint.selected
    np.testing.assert_array_equal(old.joint.plan.f_q,new.joint.plan.f_q)
    assert old.joint.cfg.radio.rho_by_uav==new.joint.cfg.radio.rho_by_uav
    assert old.joint.objective==pytest.approx(new.joint.objective,abs=1e-12)


@pytest.mark.parametrize('rcs',[.05,.1,.2])
def test_conservative_refinement_preserves_bounded_search(rcs):
    from isac_sim.power_c2f_conservative import optimize_power_c2f_conservative
    from isac_sim.model import build_base_gains
    cfg,geom,_=setup();cfg.detect.target_rcs=rcs
    base=build_base_gains(cfg,geom,np.random.default_rng(456))
    coarse=compute_link_tables(cfg,base);plan=assign_fusion_nodes(cfg,base,coarse,geom)
    selected=select_c2f_adaptive(cfg,base,coarse,plan)[0]
    old=optimize_power_joint(cfg,base,selected,plan,table_builder=PowerTableCache(cfg,base))
    new=optimize_power_c2f_conservative(cfg,base,selected,plan)
    assert old.joint.selected==new.joint.selected
    np.testing.assert_array_equal(old.joint.plan.f_q,new.joint.plan.f_q)
    assert old.joint.cfg.radio.rho_by_uav==new.joint.cfg.radio.rho_by_uav
    assert old.joint.objective==pytest.approx(new.joint.objective,abs=1e-12)


def test_adaptive_shortlist_preserves_feasible_incumbent_for_refinement():
    from isac_sim.selection import feasible_links_for_target
    cfg,geom,base=setup();cfg.refine.shortlist_size=1
    coarse=compute_link_tables(cfg,base);plan=assign_fusion_nodes(cfg,base,coarse,geom)
    seed={q:feasible_links_for_target(cfg,base,coarse,q,plan)[-2:] for q in range(cfg.scale.Q)}
    observed=[]
    def builder(c,b,dd_gain=None):
        observed.append(dd_gain.copy())
        return compute_link_tables(c,b,dd_gain=dd_gain)
    select_c2f_adaptive(cfg,base,coarse,plan,refined_table_builder=builder,shortlist_seed=seed)
    assert len(observed)==1
    for q,links in seed.items():
        for i,j in links:assert observed[0][i,j,q]==base.eta_fine[i,j,q]
