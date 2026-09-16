import copy
import numpy as np
import pytest
from tools.audit_v1_exact_budget import config, exact_job
from isac_sim.belief import BeliefState
from isac_sim.model import generate_geometry, build_base_gains, compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive, feasible_links_for_target
from isac_sim.joint_polish import improve_joint_selection
from isac_sim.simulate import remote_report_count


def scenario(seed, trial, m, k):
    cfg=config(seed,k,True,m)
    rng=np.random.default_rng([seed,trial])
    truth=generate_geometry(cfg,rng)
    bt=build_base_gains(cfg,truth,rng)
    geom=BeliefState.from_truth(cfg,truth,rng).as_geometry(truth)
    base=build_base_gains(cfg,geom,rng,channel=bt,rcs_view='mean')
    coarse=compute_link_tables(cfg,base)
    fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    selected=select_c2f_adaptive(cfg,base,coarse,plan)[0]
    return cfg,base,coarse,fine,selected,plan


@pytest.mark.parametrize('seed,trial,m,k',[(10301,96,3,0),(10303,71,4,1)])
def test_improves_archived_counterexamples_without_exceeding_oracle(seed,trial,m,k):
    args=scenario(seed,trial,m,k)
    initial=copy.deepcopy(args[4]); fs=args[5].f_q.copy()
    result=improve_joint_selection(*args)
    oracle=exact_job((trial,seed,[k],m))[0]['joint_F']
    assert result.objective_after > result.objective_before+1e-6
    assert result.objective_after <= oracle+1e-10
    assert args[4]==initial
    np.testing.assert_array_equal(args[5].f_q,fs)
    assert remote_report_count(result.selected,result.plan)<=k
    assert sum(map(len,result.selected.values()))<=args[0].selector.max_total_links
    for q, links in result.selected.items():
        assert set(links)<=set(feasible_links_for_target(args[0],args[1],args[3],q,result.plan))


def test_zero_passes_preserves_selection_and_objective():
    args=scenario(91,2,3,1)
    r=improve_joint_selection(*args,passes=0)
    assert r.selected==args[4]
    assert r.objective_after==r.objective_before


def test_rejects_infeasible_start_and_coupled_capacities():
    args=scenario(91,2,3,1)
    args[0].selector.max_total_links=0
    with pytest.raises(ValueError,match='infeasible'):
        improve_joint_selection(*args)
    args[0].selector.max_observations_per_receiver=1
    with pytest.raises(ValueError,match='unsupported'):
        improve_joint_selection(*args)
