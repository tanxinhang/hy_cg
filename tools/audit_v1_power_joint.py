"""Fixed-per-UAV-budget power/selection/fusion comparison in the 600 m scene."""
import argparse,csv,json,hashlib,sys,time
from pathlib import Path
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config,interval
from tools.audit_v1_balanced_budget import balanced_select
from isac_sim.config import apply_overrides,validate_config
from isac_sim.belief import BeliefState,belief_dd_std_bins
from isac_sim.model import generate_geometry,build_base_gains,compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive
from isac_sim.joint_polish import improve_joint_selection
from isac_sim.power_joint import optimize_power_joint
from isac_sim.simulate import run_method_on_trial,remote_report_count

METHODS=['v1','joint_fixed_power','uniform_power_joint','node_power_fixed_set','power_joint','balanced_fixed_power']
GRID=(.2,.5,.8,.95)

def job(spec):
    seed,trial,rcs=spec
    cfg=apply_overrides(config(seed,8),{'geometry.area_xy':600.,'detect.target_rcs':rcs})
    validate_config(cfg)
    rng=np.random.default_rng([seed,trial])
    truth=generate_geometry(cfg,rng);bt=build_base_gains(cfg,truth,rng)
    belief=BeliefState.from_truth(cfg,truth,rng);geom=belief.as_geometry(truth)
    base=build_base_gains(cfg,geom,rng,channel=bt,rcs_view='mean')
    coarse=compute_link_tables(cfg,base);fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    std=belief_dd_std_bins(cfg,geom,belief);plan=assign_fusion_nodes(cfg,base,coarse,geom)
    selected,d,stats=select_c2f_adaptive(cfg,base,coarse,plan)
    fixed=improve_joint_selection(cfg,base,coarse,fine,selected,plan)
    start=time.perf_counter()
    power=optimize_power_joint(cfg,base,fixed.selected,fixed.plan,grid=GRID,rounds=2)
    elapsed=time.perf_counter()-start
    balanced=balanced_select(cfg,base,coarse,plan)
    cases=[('v1',cfg,selected,plan,coarse,fine),
           ('joint_fixed_power',cfg,fixed.selected,fixed.plan,coarse,fine)]
    for name,state in [('uniform_power_joint',power.uniform),('node_power_fixed_set',power.fixed_set_power),('power_joint',power.joint)]:
        cases.append((name,state.cfg,state.selected,state.plan,state.coarse,state.fine))
    cases.append(('balanced_fixed_power',cfg,balanced,plan,coarse,fine))
    rows=[]
    for name,c,chosen,dest,ct,ft in cases:
        # Rebuild truth communication and sensing under exactly the chosen
        # split. The common evaluator additionally rebuilds refined truth.
        tt=compute_link_tables(c,bt)
        result=run_method_on_trial(c,base,ct,'proposed_c2f_adaptive_pd',trial,
            c2f_tables=ft,plan=dest,eval_base=bt,eval_tables=tt,belief_dd_std=std,
            cached_adaptive_pd=(chosen,d,stats))
        reports=remote_report_count(chosen,dest)
        rho=np.array(c.radio.rho_by_uav if c.radio.rho_by_uav is not None else [c.radio.rho]*c.scale.M)
        assert reports<=8 and sum(map(len,chosen.values()))<=60
        assert np.all((rho>0)&(rho<1)) and c.radio.P_default==1.
        ps=rho*c.radio.P_default;pc=(1-rho)*c.radio.P_default
        np.testing.assert_allclose(ps+pc,np.ones(c.scale.M),rtol=0,atol=1e-12)
        rows.append(dict(rcs_m2=rcs,trial=trial,method=name,pd=result.detected/result.total_targets,
            pfa=result.false_alarm_overall/result.total_false_overall,reports=reports,
            observations=sum(map(len,chosen.values())),active_targets=result.active_targets,
            capture=result.belief_capture_rate,bits=result.overhead_bits,delay_ms=result.overhead_delay_s*1000,
            sensing_allocated_w=float(ps.sum()),communication_allocated_w=float(pc.sum()),
            rho_mean=float(rho.mean()),rho_std=float(rho.std()),rho_vector=json.dumps(rho.tolist()),
            power_search_seconds=elapsed if name=='power_joint' else 0.,
            table_builds=power.table_builds if name=='power_joint' else 0,
            accepted_power_moves=power.power_accepts if name=='power_joint' else 0,
            predicted_objective_gain=power.joint.objective-power.initial.objective if name=='power_joint' else 0.))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--mc',type=int,default=100)
    p.add_argument('--seed',type=int,default=10919);p.add_argument('--workers',type=int,default=12)
    p.add_argument('--rcs',type=float,nargs='+',default=[.05,.1,.2])
    p.add_argument('--out',type=Path,default=Path('results_v1_power_joint'))
    a=p.parse_args();a.out.mkdir(exist_ok=True,parents=True)
    if (a.out/'protocol.json').exists():raise SystemExit('Use a fresh output directory')
    protocol={'seed':a.seed,'mc':a.mc,'rcs':a.rcs,'report_cap':8,'area_side_m':600,
        'config':asdict(apply_overrides(config(a.seed,8),{'geometry.area_xy':600.})),
        'grid':GRID,'rounds':2,'methods':METHODS,
        'scope':'Same per-UAV 1 W sensing-plus-report allocation and 15 W fleet allocation; not equal actual consumed energy. Uniform-power control also reselects/polishes. Node fixed-set stage follows uniform multistart; final joint alternates 2 rounds. Belief-only search, common truth evaluator and paired random streams. No global optimality or realized detection guarantee.',
        'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in list(Path('isac_sim').glob('*.py'))+[Path(__file__),Path('tools/audit_v1_exact_budget.py'),Path('tools/audit_v1_balanced_budget.py')]}}
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    specs=[(a.seed,t,r) for r in a.rcs for t in range(a.mc)];rows=[]
    with (a.out/'trials.csv').open('w',newline='',encoding='utf8') as f:
        writer=None
        with ProcessPoolExecutor(a.workers) as pool:
            futures=[pool.submit(job,s) for s in specs]
            for i,future in enumerate(as_completed(futures),1):
                batch=future.result();rows.extend(batch)
                if writer is None:writer=csv.DictWriter(f,fieldnames=list(batch[0]));writer.writeheader()
                writer.writerows(batch);f.flush()
                if i%5==0:print(f'{i}/{len(specs)}',flush=True)
    metrics=['pd','pfa','reports','observations','active_targets','capture','bits','delay_ms',
        'sensing_allocated_w','communication_allocated_w','rho_mean','rho_std',
        'power_search_seconds','table_builds','accepted_power_moves','predicted_objective_gain']
    summary={}
    for r in a.rcs:
        g={m:sorted([x for x in rows if x['rcs_m2']==r and x['method']==m],key=lambda x:x['trial']) for m in METHODS}
        s={m:{k:float(np.mean([x[k] for x in group])) for k in metrics} for m,group in g.items()}
        for control in ['v1','joint_fixed_power','uniform_power_joint','balanced_fixed_power']:
            ds=[x['pd']-y['pd'] for x,y in zip(g['power_joint'],g[control])]
            s['power_joint_minus_'+control]={'mean':float(np.mean(ds)),'ci95':interval(ds),'family_ci95':interval(ds,1-.05/(4*len(a.rcs)))}
        summary[str(r)]=s
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Finished: '+str(a.out),flush=True)

if __name__=='__main__':main()
