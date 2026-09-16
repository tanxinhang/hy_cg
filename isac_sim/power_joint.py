"""Belief-only joint power-split / observation / fusion optimization.

Each UAV retains its existing total power budget; rho_i redistributes that
budget between continuous sensing and orthogonal report transmission. Equal
allocated power is not a claim of equal consumed energy across schedules.
"""
from dataclasses import dataclass
import numpy as np
from .config import apply_overrides
from .model import compute_link_tables
from .selection import select_c2f_adaptive
from .joint_polish import improve_joint_selection


@dataclass
class PowerState:
    cfg: object
    selected: dict
    plan: object
    coarse: object
    fine: object
    objective: float


@dataclass
class PowerJointResult:
    initial: PowerState
    uniform: PowerState
    fixed_set_power: PowerState
    joint: PowerState
    objective_trace: list
    table_builds: int
    power_accepts: int


def optimize_power_joint(cfg, base, selected, plan, grid=(0.2,0.5,0.8,0.95), rounds=2, table_builder=None):
    """Uniform-split multistart followed by bounded coordinate alternation.

    Truth is never an input. The nominal 0.8 start is retained unless the
    canonical belief objective improves. Power steps preserve the selected
    set and reject newly infeasible reports. Subsequent topology steps may
    reselect observations and fusion destinations at the accepted power.
    """
    if cfg.radio.isac_power_model != 'sensing_only':
        raise ValueError('power joint experiment requires sensing_only model')
    if rounds < 1 or not grid or any(not np.isfinite(x) or not 0<x<1 for x in grid):
        raise ValueError('invalid power search settings')
    M=cfg.scale.M
    builds=accepts=0
    trace=[]
    builder=compute_link_tables if table_builder is None else table_builder

    def tables(c):
        nonlocal builds
        coarse=builder(c,base)
        fine=builder(c,base,dd_gain=base.eta_fine)
        builds+=2
        return coarse,fine

    def state(c,chosen,dest,coarse=None,fine=None,polish=0):
        if coarse is None:coarse,fine=tables(c)
        result=improve_joint_selection(c,base,coarse,fine,chosen,dest,passes=polish)
        return PowerState(c,result.selected,result.plan,coarse,fine,result.objective_after)

    initial=state(cfg,selected,plan)
    current=initial
    trace.append(current.objective)
    for rho in sorted(set(grid)):
        if cfg.radio.rho_by_uav is None and rho==cfg.radio.rho:continue
        candidate_cfg=apply_overrides(cfg,{'radio.rho_by_uav':tuple([rho]*M)})
        coarse,fine=tables(candidate_cfg)
        chosen=select_c2f_adaptive(candidate_cfg,base,coarse,plan,refined_table_builder=builder)[0]
        candidate=state(candidate_cfg,chosen,plan,coarse,fine,polish=2)
        if candidate.objective>current.objective+1e-12:
            current=candidate
            trace.append(current.objective)
    uniform=current
    first_power=None
    for epoch in range(rounds):
        for i in range(M):
            fractions=np.array(current.cfg.radio.rho_by_uav if current.cfg.radio.rho_by_uav is not None
                               else [current.cfg.radio.rho]*M,dtype=float)
            best=current
            for rho in sorted(set(grid)):
                if rho==fractions[i]:continue
                trial=fractions.copy();trial[i]=rho
                candidate_cfg=apply_overrides(current.cfg,{'radio.rho_by_uav':tuple(trial.tolist())})
                coarse,fine=tables(candidate_cfg)
                try:
                    candidate=state(candidate_cfg,current.selected,current.plan,coarse,fine)
                except ValueError as exc:
                    if str(exc)!='infeasible starting selection':raise
                    continue
                if candidate.objective>best.objective+1e-12:best=candidate
            if best is not current:
                current=best;accepts+=1;trace.append(current.objective)
        if epoch==0:first_power=current
        # Compare warm local refinement and a fresh C2F restart at new powers.
        warm=state(current.cfg,current.selected,current.plan,current.coarse,current.fine,polish=2)
        if warm.objective>current.objective+1e-12:
            current=warm;trace.append(current.objective)
        chosen=select_c2f_adaptive(current.cfg,base,current.coarse,current.plan,refined_table_builder=builder)[0]
        restart=state(current.cfg,chosen,current.plan,current.coarse,current.fine,polish=2)
        if restart.objective>current.objective+1e-12:
            current=restart;trace.append(current.objective)
    return PowerJointResult(initial,uniform,first_power,current,trace,builds,accepts)
