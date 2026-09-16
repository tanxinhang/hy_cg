"""Independent small-system comparison against complete enumeration."""
import csv,json,sys,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config,exact_job
from isac_sim.belief import BeliefState
from isac_sim.model import generate_geometry,build_base_gains,compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive
from isac_sim.joint_polish import improve_joint_selection

def job(spec):
    trial,m=spec
    seed=10918
    cfg=config(seed,4,True,m)
    rng=np.random.default_rng([seed,trial])
    truth=generate_geometry(cfg,rng);bt=build_base_gains(cfg,truth,rng)
    geom=BeliefState.from_truth(cfg,truth,rng).as_geometry(truth)
    base=build_base_gains(cfg,geom,rng,channel=bt,rcs_view='mean')
    coarse=compute_link_tables(cfg,base);fine=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    plan=assign_fusion_nodes(cfg,base,coarse,geom)
    rows=exact_job((trial,seed,[0,1,2,4],m))
    for row in rows:
        cfg=config(seed,row['budget'],True,m)
        selected=select_c2f_adaptive(cfg,base,coarse,plan)[0]
        r=improve_joint_selection(cfg,base,coarse,fine,selected,plan)
        assert abs(r.objective_before-row['nearest_c2f_F'])<1e-10
        assert r.objective_after<=row['joint_F']+1e-10
        row.update(M=m,polished_F=r.objective_after,
                   polished_gap=row['joint_F']-r.objective_after,accepted=r.accepted)
    return rows

def main():
    out=Path('results_v1_joint_small');out.mkdir(exist_ok=True)
    if (out/'protocol.json').exists(): raise SystemExit('Output already exists')
    protocol={'seed':10918,'mc':100,'M':[3,4],'Q':2,'budgets':[0,1,2,4],
              'scope':'Independent small systems, same fixed search settings as nominal audit.',
              'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
                        [Path(__file__),Path('isac_sim/joint_polish.py'),Path('tools/audit_v1_exact_budget.py')]}}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf8')
    rows=[]
    with ProcessPoolExecutor(2) as pool:
        for i,batch in enumerate(pool.map(job,[(t,m) for m in [3,4] for t in range(100)]),1):
            rows.extend(batch)
            if i%20==0: print(f'{i}/200',flush=True)
    with (out/'trials.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for m in [3,4]:
        for k in [0,1,2,4]:
            g=[r for r in rows if r['M']==m and r['budget']==k]
            summary[f'M{m}_K{k}']={key:{'mean':float(np.mean([r[key] for r in g])),
                'max':float(max(r[key] for r in g)),
                'optimal_fraction':float(np.mean([r[key]<=1e-9 for r in g]))}
                for key in ['total_gap','polished_gap']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Finished',flush=True)
if __name__=='__main__':main()
