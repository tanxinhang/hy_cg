"""Frozen paired diagnostics for communication, prior robustness and joint search."""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, time
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config, interval
from tools.audit_v1_balanced_budget import balanced_select
from isac_sim.config import apply_overrides
from isac_sim.belief import BeliefState, belief_dd_std_bins, geometry_robust_base
from isac_sim.model import generate_geometry, build_base_gains, compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive
from isac_sim.joint_polish import improve_joint_selection
from isac_sim.simulate import run_method_on_trial, remote_report_count

CONDITIONS = {
    'k0': {'selector.max_remote_reports': 0},
    'k2': {'selector.max_remote_reports': 2},
    'k8': {'selector.max_remote_reports': 8},
    'prior250': {'prior.belief_sigma_pos_m': 250.},
    'prior500': {'prior.belief_sigma_pos_m': 500.},
    'cap12': {'selector.max_total_links': 12},
    'no_softmin': {'selector.use_softmin_alpha': False},
    'no_deficit': {'selector.mu_deficit': 0.},
}


def job(spec):
    seed, trial, condition = spec
    cfg = apply_overrides(config(seed, 8), CONDITIONS[condition])
    rng = np.random.default_rng([seed, trial])
    truth = generate_geometry(cfg, rng)
    bt = build_base_gains(cfg, truth, rng)
    tt = compute_link_tables(cfg, bt)
    belief = BeliefState.from_truth(cfg, truth, rng)
    geom = belief.as_geometry(truth)
    base = build_base_gains(cfg, geom, rng, channel=bt, rcs_view='mean')
    coarse = compute_link_tables(cfg, base)
    t0 = time.perf_counter()
    fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
    fine_seconds = time.perf_counter()-t0
    std = belief_dd_std_bins(cfg, geom, belief)
    plan = assign_fusion_nodes(cfg, base, coarse, geom)
    t0 = time.perf_counter()
    selected, d, stats = select_c2f_adaptive(cfg, base, coarse, plan)
    base_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    improved = improve_joint_selection(cfg, base, coarse, fine, selected, plan)
    joint_seconds = base_seconds+fine_seconds+time.perf_counter()-t0
    t0 = time.perf_counter()
    balanced = balanced_select(cfg, base, coarse, plan)
    balanced_seconds = time.perf_counter()-t0
    t0 = time.perf_counter()
    rb = geometry_robust_base(cfg, base)
    rc = compute_link_tables(cfg, rb)
    robust = select_c2f_adaptive(cfg, rb, rc, plan)[0]
    robust_seconds = time.perf_counter()-t0
    methods = [('v1', selected, plan, base_seconds),
               ('joint', improved.selected, improved.plan, joint_seconds),
               ('balanced', balanced, plan, balanced_seconds),
               ('robust', robust, plan, robust_seconds)]
    rows = []
    for name, chosen, destination, elapsed in methods:
        # Same method identifier makes the evaluator share trial/target random
        # streams; the only intervention is the supplied selection and plan.
        result = run_method_on_trial(cfg, base, coarse, 'proposed_c2f_adaptive_pd', trial,
            c2f_tables=fine, plan=destination, eval_base=bt, eval_tables=tt,
            belief_dd_std=std, cached_adaptive_pd=(chosen,d,stats))
        reports = remote_report_count(chosen,destination)
        assert reports <= cfg.selector.max_remote_reports
        assert sum(map(len,chosen.values())) <= cfg.selector.max_total_links
        rows.append(dict(condition=condition,trial=trial,method=name,
            pd=result.detected/result.total_targets,
            pfa=result.false_alarm_overall/result.total_false_overall,
            reports=reports,observations=sum(map(len,chosen.values())),
            active_targets=result.active_targets,capture=result.belief_capture_rate,
            bits=result.overhead_bits,delay_ms=result.overhead_delay_s*1000,
            selector_seconds=elapsed,
            objective_gain=improved.objective_after-improved.objective_before if name=='joint' else 0.,
            extra_pd_evaluations=improved.evaluations if name=='joint' else 0,
            accepted_moves=improved.accepted if name=='joint' else 0))
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--mc',type=int,default=20)
    p.add_argument('--seed',type=int,default=10916)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--conditions',nargs='+',choices=list(CONDITIONS),default=list(CONDITIONS))
    p.add_argument('--out',type=Path,default=Path('results_v1_joint_revision'))
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if (a.out/'protocol.json').exists():
        raise SystemExit('Use a fresh output directory to preserve the protocol.')
    protocol={'seed':a.seed,'mc':a.mc,'conditions':{c:CONDITIONS[c] for c in a.conditions},
        'base_config':asdict(config(a.seed,8)),
        'search':{'passes':2,'frontier':6,'destinations':3},
        'scope':'Independent-seed exploratory audit, not noninferiority confirmation. All conditions are paired; no new physical costs. Joint runtime includes fine-table construction; shared geometry/channel preprocessing excluded.',
        'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            list(Path('isac_sim').glob('*.py'))+[Path(__file__),Path('tools/audit_v1_exact_budget.py'),Path('tools/audit_v1_balanced_budget.py')]}}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    specs=[(a.seed,t,c) for c in a.conditions for t in range(a.mc)]
    rows=[]
    with (a.out/'trials.csv').open('w',newline='',encoding='utf8') as f:
        writer=None
        with ProcessPoolExecutor(a.workers) as pool:
            for n,batch in enumerate(pool.map(job,specs),1):
                if writer is None:
                    writer=csv.DictWriter(f,fieldnames=list(batch[0]));writer.writeheader()
                writer.writerows(batch);f.flush();rows.extend(batch)
                if n%5==0: print(f'{n}/{len(specs)}',flush=True)
    summary={}
    fields=['pd','pfa','reports','observations','active_targets','capture','bits','delay_ms','selector_seconds','objective_gain','extra_pd_evaluations','accepted_moves']
    for c in a.conditions:
        groups={m:sorted([r for r in rows if r['condition']==c and r['method']==m],key=lambda r:r['trial']) for m in ['v1','joint','balanced','robust']}
        s={m:{k:float(np.mean([r[k] for r in g])) for k in fields} for m,g in groups.items()}
        for m in ['joint','balanced','robust']:
            diff=[x['pd']-y['pd'] for x,y in zip(groups[m],groups['v1'])]
            s[m+'_minus_v1']={'mean':float(np.mean(diff)),'ci95':interval(diff),
                'family_ci95':interval(diff,1-.05/(3*len(a.conditions)))}
        summary[c]=s
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Finished: '+str(a.out),flush=True)


if __name__=='__main__': main()
