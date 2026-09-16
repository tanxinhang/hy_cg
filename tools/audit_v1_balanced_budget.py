"""Coverage-balanced SINR control, using the unchanged common V1 detector.

Checks original two method outputs against the completed observation-cap audit.
The cached-selection evaluator entry point is only an adapter: all three use
the same deflection-weighted CF detector, not different detector capabilities.
"""
from __future__ import annotations
import csv,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config,interval
from isac_sim.config import apply_overrides
from isac_sim.belief import BeliefState,belief_dd_std_bins
from isac_sim.model import generate_geometry,build_base_gains,compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import (select_c2f_adaptive,select_budget_ranked_baseline,
    feasible_links_for_target,remote_cap_allows,local_cap_allows,processing_caps_allow)
from isac_sim.simulate import run_method_on_trial,remote_report_count

NAMES=['proposed_c2f_adaptive_pd','sense_sinr_budgeted','balanced_sinr']


def balanced_select(cfg,base,tables,plan):
    """Fewest selected observations first, highest coarse SINR within that tier."""
    selected={q:[] for q in range(cfg.scale.Q)}
    pool=[(q,e) for q in selected for e in feasible_links_for_target(cfg,base,tables,q,plan)]
    while sum(map(len,selected.values()))<cfg.selector.max_total_links:
        candidates=[(q,e) for q,e in pool if len(selected[q])<cfg.selector.max_links_per_target
            and remote_cap_allows(cfg,selected,e,q,plan)
            and local_cap_allows(cfg,selected[q],e,q,plan)
            and processing_caps_allow(cfg,selected,e,q,plan)]
        if not candidates:
            break
        q,e=min(candidates,key=lambda item:(len(selected[item[0]]),
            -float(tables.gamma_sense[item[1][0],item[1][1],item[0]]),item[0],item[1]))
        selected[q].append(e); pool.remove((q,e))
    return selected


def job(spec):
    trial,k=spec
    cfg=apply_overrides(config(10304,k),{'selector.max_total_links':12})
    rng=np.random.default_rng([10304,trial])
    truth=generate_geometry(cfg,rng); bt=build_base_gains(cfg,truth,rng)
    tt=compute_link_tables(cfg,bt)
    belief=BeliefState.from_truth(cfg,truth,rng); geom=belief.as_geometry(truth)
    base=build_base_gains(cfg,geom,rng,channel=bt,rcs_view=cfg.prior.scheduler_rcs.lower())
    coarse=compute_link_tables(cfg,base)
    fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    std=belief_dd_std_bins(cfg,geom,belief)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    proposed=select_c2f_adaptive(cfg,base,coarse,plan)[0]
    ranked=select_budget_ranked_baseline(cfg,base,coarse,'sense_sinr_budgeted',plan)[0]
    balanced=balanced_select(cfg,base,coarse,plan)
    stats={key:0 for key in ['fine_eval_full','fine_eval_c2f','selector_score_evaluations',
                            'coordination_messages','bid_rounds']}
    rows=[]
    for name,selected in zip(NAMES,[proposed,ranked,balanced]):
        result=run_method_on_trial(cfg,base,coarse,'proposed_c2f_adaptive_pd',trial,
            c2f_tables=fine,plan=plan,eval_base=bt,eval_tables=tt,belief_dd_std=std,
            cached_adaptive_pd=(selected,np.zeros(cfg.scale.Q),stats))
        reports=remote_report_count(selected,plan)
        assert reports<=k and sum(map(len,selected.values()))<=12
        rows.append({'trial':trial,'budget':k,'method':name,
            'pd':result.detected/result.total_targets,
            'pfa':result.false_alarm_overall/result.total_false_overall,
            'pfa_active':result.false_alarm/max(result.total_false,1),
            'active_targets':result.active_targets,'reports':reports,
            'observations':sum(map(len,selected.values())),
            'selected':json.dumps(selected),'detected_per_target':json.dumps(result.detected_per_target.tolist())})
    return rows


def main():
    out=Path('results_v1_balanced_budget'); out.mkdir(exist_ok=True)
    protocol={'seed':10304,'mc':100,'budgets':[0,2,8],'observation_cap':12,
        'config':asdict(apply_overrides(config(10304,8),{'selector.max_total_links':12})),
        'scope':'Post-hoc diagnostic on same cap12 scenarios; no new independent confirmation.',
        'balanced_rule':'Min observation count tier, then maximum coarse SINR; common hard caps.',
        'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [Path(__file__),Path('tools/audit_v1_exact_budget.py')]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        for i,batch in enumerate(pool.map(job,[(t,k) for k in [0,2,8] for t in range(100)])):
            rows.extend(batch)
            if (i+1)%25==0: print(f'balanced: {i+1}/300',flush=True)
    old=list(csv.DictReader(Path('results_v1_exact_budget_obs12/budget/trials.csv').open()))
    old={(int(r['trial']),int(r['budget']),r['method']):r for r in old}
    for r in rows:
        key=(r['trial'],r['budget'],r['method'])
        if key in old:
            for field in ['pd','pfa','reports','observations']:
                assert abs(r[field]-float(old[key][field]))<1e-12,(key,field)
    with (out/'trials.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for k in [0,2,8]:
        g={name:sorted([r for r in rows if r['budget']==k and r['method']==name],key=lambda r:r['trial']) for name in NAMES}
        s={name:{key:float(np.mean([r[key] for r in g[name]])) for key in
                 ['pd','pfa','pfa_active','active_targets','reports','observations']} for name in NAMES}
        diff=[a['pd']-b['pd'] for a,b in zip(g[NAMES[0]],g['balanced_sinr'])]
        s['paired_v1_minus_balanced']={'mean':float(np.mean(diff)),'ci95':interval(diff),
                                      'bonferroni95':interval(diff,1-.05/3)}
        summary[k]=s
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__': main()
