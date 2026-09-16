"""Isolated exhaustive small-system and matched-hard-report-budget V1 audits.

No model promotion: exhaustive optimum is of the belief-side fine moment
objective, not of realized detection or a 15-UAV joint problem.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
import csv, hashlib, json, sys, time
import numpy as np
from scipy.special import logsumexp
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_sim.config import Config, apply_overrides, apply_preset, validate_config
from isac_sim.belief import BeliefState
from isac_sim.model import generate_geometry, build_base_gains, compute_link_tables
from isac_sim.reporting import ReportingPlan, assign_fusion_nodes
from isac_sim.selection import feasible_links_for_target, select_c2f_adaptive
from isac_sim.fusion import predicted_pd_for_links, selection_utility_from_pd
from isac_sim.simulate import run_one_trial, remote_report_count
from isac_sim.fbl import blocklength_latency_s

METHODS = ['proposed_c2f_adaptive_pd', 'sense_sinr_budgeted']


def config(seed, budget, small=False, small_m=3):
    overrides = {'detect.comm_error_model': 'gaussian_replacement',
                 'selector.score_mode': 'detector_pd',
                 'selector.max_remote_reports': budget,
                 'run.seed': seed, 'run.verbose': False}
    if small:
        overrides.update({'scale.M': small_m, 'scale.Q': 2,
                          'selector.max_links_per_target': 3,
                          'selector.max_total_links': 4})
    cfg = apply_overrides(apply_preset(Config(), 'target-local-v1'), overrides)
    validate_config(cfg)
    return cfg


def objective_grid(cfg, pd0, pd1, reports):
    """Broadcast canonical two-target utility; tested against scalar routine."""
    p0, p1 = np.broadcast_arrays(pd0, pd1)
    p = np.stack([p0, p1], axis=-1)
    req = cfg.detect.pd_required
    value = -2 * cfg.selector.softmin_tau * logsumexp(
        -np.minimum(p, req) / cfg.selector.softmin_tau, axis=-1)
    value -= cfg.selector.mu_deficit * np.sum(np.maximum(req-p, 0)**2, axis=-1)/(2*req)
    if cfg.selector.use_delay_price:
        value -= cfg.selector.lambda_c * reports * 1e3 * blocklength_latency_s(cfg)
    return value


def exact_job(job):
    trial, seed, budgets, *extra = job
    small_m = extra[0] if extra else 3
    cfg = config(seed, max(budgets), True, small_m)
    rng = np.random.default_rng([seed, trial])
    truth = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, truth, rng)
    geom = BeliefState.from_truth(cfg, truth, rng).as_geometry(truth)
    base = build_base_gains(cfg, geom, rng, channel=base_truth, rcs_view='mean')
    coarse = compute_link_tables(cfg, base)
    fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
    nearest = assign_fusion_nodes(cfg, base, coarse, geom)
    options = []
    for q in range(2):
        oq = []
        for f in range(small_m):
            plan = ReportingPlan(mode='explicit', f_q=nearest.f_q.copy())
            plan.f_q[q] = f
            links = feasible_links_for_target(cfg, base, fine, q, plan)
            for n in range(min(3, len(links))+1):
                for subset in combinations(links, n):
                    pd = predicted_pd_for_links(cfg, fine, q, list(subset), plan=plan, base=base)
                    oq.append({'f': f, 'links': subset, 'n': n,
                               'r': sum(j != f for i, j in subset), 'pd': pd})
        options.append(oq)
    a, b = options
    ns = np.array([o['n'] for o in a])[:, None] + np.array([o['n'] for o in b])[None, :]
    rs = np.array([o['r'] for o in a])[:, None] + np.array([o['r'] for o in b])[None, :]
    grid = objective_grid(cfg, np.array([o['pd'] for o in a])[:, None],
                          np.array([o['pd'] for o in b])[None, :], rs)
    fixed = ((np.array([o['f'] for o in a])[:, None] == nearest.f_q[0]) &
             (np.array([o['f'] for o in b])[None, :] == nearest.f_q[1]))
    rows = []
    for budget in budgets:
        cfg = config(seed, budget, True, small_m)
        valid = (ns <= cfg.selector.max_total_links) & (rs <= budget)
        joint = float(np.max(np.where(valid, grid, -np.inf)))
        fixed_value = float(np.max(np.where(valid & fixed, grid, -np.inf)))
        selected, _, stats = select_c2f_adaptive(cfg, base, coarse, nearest)
        pds = np.array([predicted_pd_for_links(cfg, fine, q, selected[q], plan=nearest, base=base)
                        for q in range(2)])
        reports = remote_report_count(selected, nearest)
        c2f = selection_utility_from_pd(cfg, np.zeros(2), pds)
        if cfg.selector.use_delay_price:
            c2f -= cfg.selector.lambda_c * reports * 1e3 * blocklength_latency_s(cfg)
        assert reports <= budget
        assert joint + 1e-10 >= fixed_value >= c2f - 1e-10
        ii, jj = np.unravel_index(np.argmax(np.where(valid, grid, -np.inf)), grid.shape)
        rows.append({'trial': trial, 'budget': budget, 'joint_F': joint,
                     'nearest_exact_F': fixed_value, 'nearest_c2f_F': c2f,
                     'placement_gap': joint-fixed_value, 'selection_gap': fixed_value-c2f,
                     'total_gap': joint-c2f, 'joint_feasible_pairs': int(valid.sum()),
                     'candidate_pairs_before_caps': len(a)*len(b),
                     'joint_mean_predicted_pd': (a[ii]['pd']+b[jj]['pd'])/2,
                     'c2f_mean_predicted_pd': float(pds.mean()),
                     'c2f_reports': reports, 'joint_reports': int(rs[ii, jj]),
                     'joint_plan': json.dumps([a[ii]['f'], b[jj]['f']]),
                     'joint_sets': json.dumps([a[ii]['links'], b[jj]['links']]),
                     'nearest_plan': json.dumps(nearest.f_q.tolist()),
                     'c2f_sets': json.dumps([selected[q] for q in range(2)])})
    return rows


def budget_job(job):
    trial, seed, budget, *extra = job
    cfg = config(seed, budget)
    if extra:
        cfg = apply_overrides(cfg, {'selector.max_total_links': extra[0]})
    start = time.perf_counter()
    results = run_one_trial(cfg, trial, METHODS)
    rows = []
    for method, r in results.items():
        reports = remote_report_count(r.selected_links, r.reporting_plan)
        assert reports <= budget
        assert sum(map(len, r.selected_links.values())) <= cfg.selector.max_total_links
        rows.append({'trial': trial, 'budget': budget, 'method': method,
                     'pd': r.detected/r.total_targets,
                     'pfa': r.false_alarm_overall/r.total_false_overall,
                     'reports': reports, 'observations': sum(map(len, r.selected_links.values())),
                     'bits': r.overhead_bits, 'delay_ms': r.overhead_delay_s*1e3,
                     'detected_per_target': json.dumps(r.detected_per_target.tolist()),
                     'capture': r.belief_capture_rate,
                     'seconds_two_methods': time.perf_counter()-start})
    return rows


def interval(x, confidence=.95):
    x=np.asarray(x, dtype=float)
    if len(x)<2 or np.std(x)==0:
        return None
    half=student_t.ppf((1+confidence)/2,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))
    return [float(x.mean()-half),float(x.mean()+half)]


def summarize(rows, kind):
    out={}
    budgets=sorted({r['budget'] for r in rows})
    for budget in budgets:
        group=[r for r in rows if r['budget']==budget]
        if kind=='exact':
            out[budget]={'trials':len(group)}
            for key in ['placement_gap','selection_gap','total_gap']:
                vals=np.array([r[key] for r in group])
                out[budget][key]={'mean':float(vals.mean()),'max':float(vals.max()),
                    'p95':float(np.quantile(vals,.95)),'zero_fraction':float(np.mean(vals<=1e-9))}
        else:
            stats={}
            for method in METHODS:
                g=sorted([r for r in group if r['method']==method],key=lambda r:r['trial'])
                stats[method]={k:float(np.mean([r[k] for r in g])) for k in
                               ['pd','pfa','reports','observations','bits','delay_ms','capture']}
            x=sorted([r for r in group if r['method']==METHODS[0]],key=lambda r:r['trial'])
            y=sorted([r for r in group if r['method']==METHODS[1]],key=lambda r:r['trial'])
            d=np.array([a['pd']-b['pd'] for a,b in zip(x,y)])
            df=np.array([a['pfa']-b['pfa'] for a,b in zip(x,y)])
            stats.update({'trials':len(x),'paired_pd_delta':float(d.mean()),
                          'paired_pd_ci95':interval(d),
                          'paired_pd_bonferroni95':interval(d,1-.05/len(budgets)),
                          'paired_pfa_delta':float(df.mean()),'paired_pfa_ci95':interval(df)})
            out[budget]=stats
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind',choices=['exact','budget'],required=True)
    p.add_argument('--mc',type=int,default=100)
    p.add_argument('--seed',type=int,default=10301)
    p.add_argument('--budgets',type=int,nargs='+',default=[0,1,2,4,8])
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--small-uavs',type=int,choices=[3,4],default=3)
    p.add_argument('--observation-cap',type=int,default=60)
    p.add_argument('--out',type=Path,default=Path('results_v1_exact_budget'))
    args=p.parse_args()
    out=args.out/args.kind; out.mkdir(parents=True,exist_ok=True)
    cfg=config(args.seed,max(args.budgets),args.kind=='exact',args.small_uavs)
    if args.kind=='budget':
        cfg=apply_overrides(cfg,{'selector.max_total_links':args.observation_cap})
    protocol={**vars(args),'out':str(args.out),'config':asdict(cfg),
              'scope':'Exploratory audit; fixed V1 physical model; no headline replacement.',
              'exact_scope':f'M={args.small_uavs},Q=2,Lmax=3,Ltot=4; complete candidate pool; exact fine belief moment objective, not true PD.',
              'budget_scope':'Common hard upper limits, not equal realized report counts; nominal V1 price retained in proposed method; independently ranked SINR baseline.',
              'inference':'Paired trial-level t intervals; Bonferroni across budget points; no equivalence claim. Zero variance returns null.',
              'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 [Path(__file__),*Path('isac_sim').glob('*.py')]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    jobs=([(i,args.seed,args.budgets,args.small_uavs) for i in range(args.mc)] if args.kind=='exact'
          else [(i,args.seed,k,args.observation_cap) for k in args.budgets for i in range(args.mc)])
    rows=[]
    with ProcessPoolExecutor(args.workers) as pool:
        for index,batch in enumerate(pool.map(exact_job if args.kind=='exact' else budget_job,jobs)):
            rows.extend(batch)
            if (index+1)%10==0 or index+1==len(jobs):
                print(f'{args.kind}: {index+1}/{len(jobs)} completed',flush=True)
    with (out/'trials.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=summarize(rows,args.kind)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    main()
