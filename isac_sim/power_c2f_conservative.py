"""On-demand C2F refinement preserving the serial bounded search actions.

All power-grid candidates remain. Only links reachable by the existing
frontier/destination rules need fine values; this does not assert globally
lossless C2F screening or global optimality.
"""
import numpy as np
from .config import apply_overrides
from .power_joint import PowerState
from .power_c2f import IntegratedResult
from .power_tables import PowerTableCache
from .joint_polish import improve_joint_selection
from .selection import select_c2f_adaptive,feasible_links_for_target
from .reporting import ReportingPlan


def refinement_pool(cfg,base,coarse,selected,plan,polish):
    pool={q:list(selected.get(q,[])) for q in range(cfg.scale.Q)}
    if not polish:return pool
    for q in pool:
        strength=np.max(coarse.gamma_sense[:,:,q],axis=0)
        fs=list(dict.fromkeys([int(plan.f_q[q])]+sorted(range(cfg.scale.M),key=lambda f:(-strength[f],f))[:3]))
        for f in fs:
            dest=ReportingPlan('explicit',plan.f_q.copy());dest.f_q[q]=f
            links=feasible_links_for_target(cfg,base,coarse,q,dest)
            rank=lambda e:(-float(coarse.gamma_sense[e[0],e[1],q]),e)
            pool[q]+=sorted(links,key=rank)[:6]
            pool[q]+=sorted([e for e in links if e[1]==f],key=rank)[:6]
        pool[q]=list(dict.fromkeys(pool[q]))
    return pool


def optimize_power_c2f_conservative(cfg,base,selected,plan,grid=(.2,.5,.8,.95),rounds=2):
    if rounds<1 or not grid or any(not np.isfinite(x) or not 0<x<1 for x in grid):
        raise ValueError('invalid power search settings')
    cache=PowerTableCache(cfg,base);trace=[];accepts=checks=0
    def state(c,chosen,dest,polish=0):
        coarse=cache(c,base)
        pool=refinement_pool(c,base,coarse,chosen,dest,polish)
        fine=cache.refine_pool(c,pool)
        r=improve_joint_selection(c,base,coarse,fine,chosen,dest,passes=polish,candidate_pool=pool)
        return PowerState(c,r.selected,r.plan,coarse,fine,r.objective_after)
    current=initial=state(cfg,selected,plan);trace.append(current.objective)
    def accept(candidate):
        nonlocal current
        if candidate.objective>current.objective+1e-12:
            current=candidate;trace.append(current.objective);return True
        return False
    for rho in sorted(set(grid)):
        if cfg.radio.rho_by_uav is None and rho==cfg.radio.rho:continue
        c=apply_overrides(cfg,{'radio.rho_by_uav':tuple([rho]*cfg.scale.M)})
        chosen=select_c2f_adaptive(c,base,cache(c,base),plan,refined_table_builder=cache)[0]
        accept(state(c,chosen,plan,polish=2))
    for epoch in range(rounds):
        for i in range(cfg.scale.M):
            fractions=np.array(current.cfg.radio.rho_by_uav if current.cfg.radio.rho_by_uav is not None else [current.cfg.radio.rho]*cfg.scale.M)
            best=current
            for rho in sorted(set(grid)):
                if rho==fractions[i]:continue
                trial=fractions.copy();trial[i]=rho
                c=apply_overrides(current.cfg,{'radio.rho_by_uav':tuple(trial.tolist())});checks+=1
                try:candidate=state(c,current.selected,current.plan)
                except ValueError as exc:
                    if str(exc)!='infeasible starting selection':raise
                    continue
                if candidate.objective>best.objective+1e-12:best=candidate
            if accept(best):accepts+=1
        accept(state(current.cfg,current.selected,current.plan,polish=2))
        chosen=select_c2f_adaptive(current.cfg,base,current.coarse,current.plan,refined_table_builder=cache)[0]
        accept(state(current.cfg,chosen,current.plan,polish=2))
    counts=dict(table_builds=cache.table_builds,cache_hits=cache.cache_hits,fine_entry_updates=cache.fine_updates,
        unique_refined_entries=int(cache.refined_seen.sum()),full_entries=int(cache.valid.sum()),
        fine_power_checks=checks,coarse_power_checks=0,power_accepts=accepts)
    return IntegratedResult(initial,current,trace,counts)
