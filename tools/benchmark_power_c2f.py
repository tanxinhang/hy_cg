"""Single-process three-way benchmark with exact action-result checks."""
import argparse,csv,json,time,sys,hashlib
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_power_joint import config,apply_overrides,generate_geometry,build_base_gains,BeliefState,belief_dd_std_bins,compute_link_tables,assign_fusion_nodes,select_c2f_adaptive,improve_joint_selection,run_method_on_trial,remote_report_count
from isac_sim.power_joint import optimize_power_joint
from isac_sim.power_c2f_conservative import optimize_power_c2f_conservative
from isac_sim.power_tables import PowerTableCache


def main():
    p=argparse.ArgumentParser();p.add_argument('--mc',type=int,default=4);p.add_argument('--seed',type=int,default=10929)
    p.add_argument('--out',type=Path,default=Path('results_v1_power_c2f_benchmark'));a=p.parse_args()
    a.out.mkdir(exist_ok=True,parents=True)
    if (a.out/'protocol.json').exists():raise SystemExit('Use a fresh directory')
    protocol=dict(seed=a.seed,mc=a.mc,rcs=[.05,.1,.2],workers=1,scope='Single-process rotated-order timing; excludes shared initial selection, geometry/DD preprocessing, truth evaluation. Three methods same grid and two rounds. Compare selected sets, fusion assignments, rho vectors and objective.',hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path('isac_sim').glob('*.py'))+[Path(__file__)]})
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    rows=[];index=0
    with (a.out/'trials.csv').open('w',newline='',encoding='utf8') as f:
        writer=None
        for rcs in [.05,.1,.2]:
            for trial in range(a.mc):
                cfg=apply_overrides(config(a.seed,8),{'geometry.area_xy':600.,'detect.target_rcs':rcs})
                rng=np.random.default_rng([a.seed,trial]);truth=generate_geometry(cfg,rng);bt=build_base_gains(cfg,truth,rng)
                belief=BeliefState.from_truth(cfg,truth,rng);geom=belief.as_geometry(truth)
                base=build_base_gains(cfg,geom,rng,channel=bt,rcs_view='mean')
                coarse=compute_link_tables(cfg,base);fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
                plan=assign_fusion_nodes(cfg,base,coarse,geom);selected,d,stats=select_c2f_adaptive(cfg,base,coarse,plan)
                fixed=improve_joint_selection(cfg,base,coarse,fine,selected,plan)
                methods=['original','cached_serial','conservative_c2f'];offset=index%3;methods=methods[offset:]+methods[:offset]
                states={};batch=[]
                for method in methods:
                    start=time.perf_counter()
                    if method=='original':
                        result=optimize_power_joint(cfg,base,fixed.selected,fixed.plan)
                        updates=unique=0
                    elif method=='cached_serial':
                        cache=PowerTableCache(cfg,base)
                        result=optimize_power_joint(cfg,base,fixed.selected,fixed.plan,table_builder=cache)
                        updates=cache.fine_updates;unique=int(cache.refined_seen.sum())
                    else:
                        result=optimize_power_c2f_conservative(cfg,base,fixed.selected,fixed.plan)
                        updates=result.counters['fine_entry_updates'];unique=result.counters['unique_refined_entries']
                    elapsed=time.perf_counter()-start;s=result.joint;states[method]=s
                    check=improve_joint_selection(s.cfg,base,s.coarse,compute_link_tables(s.cfg,base,dd_gain=base.eta_fine),s.selected,s.plan,passes=0)
                    assert abs(check.objective_after-s.objective)<1e-10
                    detection=run_method_on_trial(s.cfg,base,s.coarse,'proposed_c2f_adaptive_pd',trial,
                        c2f_tables=s.fine,plan=s.plan,eval_base=bt,eval_tables=compute_link_tables(s.cfg,bt),
                        belief_dd_std=belief_dd_std_bins(cfg,geom,belief),cached_adaptive_pd=(s.selected,d,stats))
                    row=dict(rcs_m2=rcs,trial=trial,method=method,seconds=elapsed,pd=detection.detected/detection.total_targets,
                        reports=remote_report_count(s.selected,s.plan),observations=sum(map(len,s.selected.values())),
                        objective=s.objective,fine_entry_updates=updates,unique_refined_entries=unique)
                    batch.append(row)
                old=states['original']
                for other in ['cached_serial','conservative_c2f']:
                    s=states[other]
                    assert old.selected==s.selected,(rcs,trial,other,'selection')
                    np.testing.assert_array_equal(old.plan.f_q,s.plan.f_q)
                    np.testing.assert_array_equal(old.cfg.radio.rho_by_uav,s.cfg.radio.rho_by_uav)
                    assert abs(old.objective-s.objective)<1e-10
                assert len({x['pd'] for x in batch})==1
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(batch[0]));writer.writeheader()
                writer.writerows(batch);f.flush();rows+=batch;index+=1;print(f'{index}/{3*a.mc}: identical decisions',flush=True)
    summary={m:{k:float(np.mean([x[k] for x in rows if x['method']==m])) for k in ['seconds','pd','reports','observations','fine_entry_updates','unique_refined_entries']} for m in ['original','cached_serial','conservative_c2f']}
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
