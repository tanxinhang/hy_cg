"""Budgeted coarse-to-fine joint power, observation and fusion search.

Coarse power screening is heuristic. Every acceptance uses the fine belief
objective, with the incumbent retained. Neither true-PD nor global guarantees.
"""
from dataclasses import dataclass
import numpy as np
from .config import apply_overrides
from .power_joint import PowerState
from .power_tables import PowerTableCache
from .joint_polish import improve_joint_selection
from .selection import select_c2f_adaptive,feasible_links_for_target
from .reporting import ReportingPlan


@dataclass
class IntegratedResult:
    initial: PowerState
    joint: PowerState
    objective_trace: list
    counters: dict


def optimize_power_c2f(cfg,base,selected,plan,grid=(.2,.5,.8,.95),rounds=2,power_keep=2):
    if rounds<1 or power_keep<1 or not grid or any(not np.isfinite(x) or not 0<x<1 for x in grid):
        raise ValueError('invalid integrated search settings')
    cache=PowerTableCache(cfg,base)
    counts=dict(coarse_power_checks=0,fine_power_checks=0,power_accepts=0,topology_updates=0)
    trace=[]

    def state(c,chosen,dest,polish=0):
        coarse=cache(c,base)
        # Dynamic fusion-aware shortlist: incumbent, strongest global links,
        # and local evidence at every shortlisted fusion destination.
        pool={q:list(chosen.get(q,[])) for q in range(c.scale.Q)}
        if polish:
            for q in pool:
                strength=np.max(coarse.gamma_sense[:,:,q],axis=0)
                fs=list(dict.fromkeys([int(dest.f_q[q])]+sorted(range(c.scale.M),key=lambda f:(-strength[f],f))[:3]))
                for f in fs:
                    p=ReportingPlan('explicit',dest.f_q.copy());p.f_q[q]=f
                    links=feasible_links_for_target(c,base,coarse,q,p)
                    rank=lambda e:(-float(coarse.gamma_sense[e[0],e[1],q]),e)
                    pool[q]+=sorted(links,key=rank)[:6]
                    pool[q]+=sorted([e for e in links if e[1]==f],key=rank)[:6]
                pool[q]=list(dict.fromkeys(pool[q]))
        fine=cache.refine_pool(c,pool)
        result=improve_joint_selection(c,base,coarse,fine,chosen,dest,passes=polish,candidate_pool=pool)
        return PowerState(c,result.selected,result.plan,coarse,fine,result.objective_after)

    current=initial=state(cfg,selected,plan);trace.append(current.objective)
    def accept(candidate):
        nonlocal current
        if candidate.objective>current.objective+1e-12:
            current=candidate;trace.append(current.objective);return True
        return False

    def topology():
        counts['topology_updates']+=1
        accept(state(current.cfg,current.selected,current.plan,polish=2))
        chosen=select_c2f_adaptive(current.cfg,base,current.coarse,current.plan,
            refined_table_builder=cache,shortlist_seed=current.selected)[0]
        accept(state(current.cfg,chosen,current.plan,polish=2))

    for rho in sorted(set(grid)):
        if cfg.radio.rho_by_uav is None and rho==cfg.radio.rho:continue
        c=apply_overrides(cfg,{'radio.rho_by_uav':tuple([rho]*cfg.scale.M)})
        chosen=select_c2f_adaptive(c,base,cache(c,base),plan,refined_table_builder=cache)[0]
        accept(state(c,chosen,plan,polish=2))
    for epoch in range(rounds):
        for i in range(cfg.scale.M):
            fractions=np.array(current.cfg.radio.rho_by_uav if current.cfg.radio.rho_by_uav is not None else [current.cfg.radio.rho]*cfg.scale.M)
            proposals=[]
            for rho in sorted(set(grid)):
                if rho==fractions[i]:continue
                trial=fractions.copy();trial[i]=rho
                c=apply_overrides(current.cfg,{'radio.rho_by_uav':tuple(trial.tolist())})
                coarse=cache(c,base);counts['coarse_power_checks']+=1
                try:
                    score=improve_joint_selection(c,base,coarse,coarse,current.selected,current.plan,passes=0).objective_after
                except ValueError as exc:
                    if str(exc)!='infeasible starting selection':raise
                    continue
                proposals.append((score,rho,c))
            proposals.sort(key=lambda x:(-x[0],x[1]))
            best=current
            for _,_,c in proposals[:power_keep]:
                candidate=state(c,current.selected,current.plan);counts['fine_power_checks']+=1
                if candidate.objective>best.objective+1e-12:best=candidate
            if accept(best):counts['power_accepts']+=1
            # Rebuild observation/fusion candidates within the power sweep,
            # rather than freezing the topology until all nodes have moved.
            if (i+1)%5==0 or i==cfg.scale.M-1:topology()
    counts.update(table_builds=cache.table_builds,cache_hits=cache.cache_hits,
        fine_entry_updates=cache.fine_updates,unique_refined_entries=int(cache.refined_seen.sum()),
        full_entries=int(cache.valid.sum()))
    return IntegratedResult(initial,current,trace,counts)
