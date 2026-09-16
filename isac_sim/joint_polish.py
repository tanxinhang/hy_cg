"""Bounded belief-only fusion/observation coordinate search for V1 experiments.

Strict improvement of the supplied predicted objective, not a detection or
global-optimality guarantee. Fine-table construction must be charged by callers.
"""
from dataclasses import dataclass
import numpy as np

from .fusion import predicted_pd_for_links, selection_utility_from_pd
from .reporting import ReportingPlan
from .selection import feasible_links_for_target, link_cost_ms


@dataclass
class JointPolishResult:
    selected: dict
    plan: ReportingPlan
    objective_before: float
    objective_after: float
    evaluations: int
    accepted: int


def improve_joint_selection(cfg, base, coarse, fine, selected, plan,
                            passes=2, frontier=6, destinations=3, candidate_pool=None):
    """Try relocation, additions, deletions and within-target one-for-one swaps.

    Also try restarting each target from a singleton after relocation. Other
    targets stay fixed. Candidate destinations include the incumbent and the
    receivers with strongest coarse singleton sensing. No truth is accepted.
    """
    if not plan.is_explicit() or cfg.comm.mac_model != 'serial' or cfg.comm.interference_model != 'orthogonal':
        raise ValueError('requires explicit orthogonal serial V1 reporting')
    if (cfg.corr.enable or cfg.interference.sense_gate_by_active_tx
        or cfg.fusion.max_targets_per_uav >= 0
        or cfg.selector.max_observations_per_receiver >= 0
        or cfg.selector.max_observations_per_fusion_uav >= 0
        or cfg.fusion.cpu_rate_cycles_per_s >= 0
        or cfg.selector.require_local_anchor or not cfg.selector.use_target_priority):
        raise ValueError('unsupported coupled constraints or objective')
    if passes < 0 or frontier < 1 or destinations < 1:
        raise ValueError('invalid search limits')
    Q, M = cfg.scale.Q, cfg.scale.M
    current = {q: list(selected.get(q, [])) for q in range(Q)}
    output = ReportingPlan('explicit', np.array(plan.f_q, copy=True))
    if output.f_q.shape != (Q,) or np.any(output.f_q < 0) or np.any(output.f_q >= M):
        raise ValueError('invalid fusion assignment')
    cache = {}
    evaluations = 0

    def metrics(q, f, links):
        nonlocal evaluations
        key = (q, f, tuple(sorted(links)))
        if key not in cache:
            trial = ReportingPlan('explicit', output.f_q.copy())
            trial.f_q[q] = f
            pd = predicted_pd_for_links(cfg, fine, q, list(key[2]), plan=trial, base=base)
            cost = sum(link_cost_ms(cfg, fine, q, e, trial) for e in links)
            cache[key] = (pd, cost)
            evaluations += 1
        return cache[key]

    def objective(pd, cost):
        return selection_utility_from_pd(cfg, np.zeros(Q), pd) - (
            cfg.selector.lambda_c * sum(cost) if cfg.selector.use_delay_price else 0.)

    def valid(q, f, links, others_n, others_r, feasible):
        n = len(links)
        local = sum(j == f for _, j in links)
        s = cfg.selector
        return (len(set(links)) == n and set(links) <= feasible
                and n <= s.max_links_per_target and others_n+n <= s.max_total_links
                and (s.max_local_observations_per_target < 0 or local <= s.max_local_observations_per_target)
                and (s.max_remote_reports < 0 or others_r+n-local <= s.max_remote_reports))

    pd, costs = np.zeros(Q), np.zeros(Q)
    for q in range(Q):
        f = int(output.f_q[q])
        others_n = sum(len(v) for k,v in current.items() if k != q)
        others_r = sum(j != output.f_q[k] for k,v in current.items() if k != q for _,j in v)
        feasible = set(feasible_links_for_target(cfg, base, fine, q, output))
        if not valid(q, f, current[q], others_n, others_r, feasible):
            raise ValueError('infeasible starting selection')
        pd[q], costs[q] = metrics(q, f, current[q])
    start = value = objective(pd, costs)
    if not np.isfinite(start):
        raise ValueError('nonfinite starting objective')
    accepted = 0
    for _ in range(passes):
        changed = False
        for q in range(Q):
            f0 = int(output.f_q[q])
            strength = np.max(coarse.gamma_sense[:, :, q], axis=0)
            fs = list(dict.fromkeys([f0] + sorted(range(M), key=lambda f: (-strength[f], f))[:destinations]))
            others_n = sum(len(v) for k,v in current.items() if k != q)
            others_r = sum(j != output.f_q[k] for k,v in current.items() if k != q for _,j in v)
            best = None
            best_value = value
            for f in fs:
                candidate_plan = ReportingPlan('explicit', output.f_q.copy())
                candidate_plan.f_q[q] = f
                feasible = set(feasible_links_for_target(cfg, base, fine, q, candidate_plan))
                if candidate_pool is not None:
                    feasible &= set(candidate_pool.get(q,[])) | set(current[q])
                rank = lambda e: (-float(coarse.gamma_sense[e[0], e[1], q]), e)
                pool = sorted(feasible, key=rank)[:frontier]
                pool += sorted([e for e in feasible if e[1] == f], key=rank)[:frontier]
                pool = sorted(set(pool))
                old = current[q]
                sets = {tuple(sorted(old)), (), tuple(sorted(e for e in old if e in feasible))}
                for e in pool:
                    sets.add((e,))
                    if e not in old:
                        sets.add(tuple(sorted(old+[e])))
                        for removed in old:
                            sets.add(tuple(sorted([x for x in old if x != removed]+[e])))
                for removed in old:
                    sets.add(tuple(sorted(x for x in old if x != removed)))
                for links in sorted(sets):
                    if not valid(q, f, links, others_n, others_r, feasible):
                        continue
                    p, c = metrics(q, f, links)
                    ps, cs = pd.copy(), costs.copy()
                    ps[q], cs[q] = p, c
                    new = objective(ps, cs)
                    if np.isfinite(new) and new > best_value+1e-12:
                        best_value, best = new, (f, list(links), p, c)
            if best is not None:
                output.f_q[q], current[q], pd[q], costs[q] = best
                value = best_value
                accepted += 1
                changed = True
        if not changed:
            break
    return JointPolishResult(current, output, float(start), float(value), evaluations, accepted)
