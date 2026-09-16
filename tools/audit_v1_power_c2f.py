"""Paired fresh-scene comparison; separate algebraic caching from C2F search."""
import argparse,csv,json,time,sys,hashlib
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_power_joint import config,interval,apply_overrides,validate_config,generate_geometry,build_base_gains,BeliefState,belief_dd_std_bins,compute_link_tables,assign_fusion_nodes,select_c2f_adaptive,improve_joint_selection,run_method_on_trial,remote_report_count
from isac_sim.power_joint import optimize_power_joint
from isac_sim.power_c2f import optimize_power_c2f
from isac_sim.power_tables import PowerTableCache


def job(spec):
    seed,trial,rcs=spec
    cfg=apply_overrides(config(seed,8),{'geometry.area_xy':600.,'detect.target_rcs':rcs})
    rng=np.random.default_rng([seed,trial]);truth=generate_geometry(cfg,rng);bt=build_base_gains(cfg,truth,rng)
    belief=BeliefState.from_truth(cfg,truth,rng);geom=belief.as_geometry(truth)
    base=build_base_gains(cfg,geom,rng,channel=bt,rcs_view='mean')
    coarse=compute_link_tables(cfg,base);fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    plan=assign_fusion_nodes(cfg,base,coarse,geom);selected,d,stats=select_c2f_adaptive(cfg,base,coarse,plan)
    fixed=improve_joint_selection(cfg,base,coarse,fine,selected,plan)
    methods=['cached_serial','integrated_c2f']
    if trial%2:methods.reverse()
    rows=[]
    for method in methods:
        start=time.perf_counter()
        if method=='cached_serial':
            cache=PowerTableCache(cfg,base)
            result=optimize_power_joint(cfg,base,fixed.selected,fixed.plan,table_builder=cache)
            counts={'table_builds':cache.table_builds,'fine_entry_updates':cache.fine_updates,
                    'unique_refined_entries':int(cache.refined_seen.sum()),'full_entries':int(cache.valid.sum()),
                    'coarse_power_checks':0,'fine_power_checks':0}
        else:
            result=optimize_power_c2f(cfg,base,fixed.selected,fixed.plan)
            counts=result.counters
        elapsed=time.perf_counter()-start;s=result.joint
        exact=compute_link_tables(s.cfg,base,dd_gain=base.eta_fine)
        check=improve_joint_selection(s.cfg,base,s.coarse,exact,s.selected,s.plan,passes=0)
        assert abs(check.objective_after-s.objective)<1e-10
        detection=run_method_on_trial(s.cfg,base,s.coarse,'proposed_c2f_adaptive_pd',trial,
            c2f_tables=exact,plan=s.plan,eval_base=bt,eval_tables=compute_link_tables(s.cfg,bt),
            belief_dd_std=belief_dd_std_bins(cfg,geom,belief),cached_adaptive_pd=(s.selected,d,stats))
        rows.append(dict(rcs_m2=rcs,trial=trial,method=method,pd=detection.detected/detection.total_targets,
            pfa=detection.false_alarm_overall/detection.total_false_overall,
            reports=remote_report_count(s.selected,s.plan),observations=sum(map(len,s.selected.values())),
            seconds=elapsed,objective=s.objective,rho_vector=json.dumps(s.cfg.radio.rho_by_uav),
            **{k:counts[k] for k in ['table_builds','fine_entry_updates','unique_refined_entries','full_entries','coarse_power_checks','fine_power_checks']}))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--mc',type=int,default=30);p.add_argument('--workers',type=int,default=6)
    p.add_argument('--seed',type=int,default=10923);p.add_argument('--out',type=Path,default=Path('results_v1_power_c2f'))
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    if (a.out/'protocol.json').exists():raise SystemExit('Use fresh output directory')
    protocol=dict(seed=a.seed,mc=a.mc,workers=a.workers,rcs=[.05,.1,.2],grid=[.2,.5,.8,.95],rounds=2,power_keep=2,
        config=asdict(apply_overrides(config(a.seed,8),{'geometry.area_xy':600.})),
        scope='Exploratory paired fresh scenes. Same power grid, allocation and budgets. Timed search only, excluding shared initial selection and geometry/DD preprocessing. Both use identical cached physics; alternating execution order. Fine entry counts are logical updates, not waveform computations. No global optimality or lossless screening guarantee.',
        hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path('isac_sim').glob('*.py'))+[Path(__file__)]})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    rows=[]
    with (a.out/'trials.csv').open('w',newline='',encoding='utf8') as f:
        writer=None
        with ProcessPoolExecutor(a.workers) as pool:
            futures=[pool.submit(job,(a.seed,t,r)) for r in [.05,.1,.2] for t in range(a.mc)]
            for i,future in enumerate(as_completed(futures),1):
                batch=future.result();rows+=batch
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(batch[0]));writer.writeheader()
                writer.writerows(batch);f.flush()
                if i%5==0:print(f'{i}/{len(futures)}',flush=True)
    summary={}
    for r in [.05,.1,.2]:
        groups={m:sorted([x for x in rows if x['rcs_m2']==r and x['method']==m],key=lambda x:x['trial']) for m in ['cached_serial','integrated_c2f']}
        entry={m:{k:float(np.mean([x[k] for x in group])) for k in ['pd','pfa','reports','observations','seconds','objective','fine_entry_updates','unique_refined_entries','full_entries','coarse_power_checks','fine_power_checks']} for m,group in groups.items()}
        ds=[b['pd']-a['pd'] for a,b in zip(groups['cached_serial'],groups['integrated_c2f'])]
        entry['pd_difference']={'mean':float(np.mean(ds)),'ci95':interval(ds),'family_ci95':interval(ds,1-.05/3)}
        summary[str(r)]=entry
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
