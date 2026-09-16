"""Resume the frozen RCS protocol with more workers, preserving completed rows."""
import csv,json,hashlib,sys,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_rcs_joint import CONDITIONS
from tools.audit_v1_joint_revision import job
from tools.audit_v1_exact_budget import interval

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    protocol=json.loads((a.out/'protocol.json').read_text())
    for source,digest in protocol['hashes'].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest,source
    for c,overrides in protocol['conditions'].items(): assert CONDITIONS[c]==overrides
    rows=list(csv.DictReader((a.out/'trials.csv').open(encoding='utf8')))
    fields=list(rows[0]);numeric=[k for k in fields if k not in ['condition','method']]
    for r in rows:
        for k in numeric:r[k]=float(r[k])
        r['trial']=int(r['trial'])
    groups={}
    for r in rows:groups.setdefault((r['condition'],r['trial']),[]).append(r)
    for key,g in groups.items():
        assert len(g)==4 and {r['method'] for r in g}=={'v1','joint','balanced','robust'},key
    pending=[(protocol['seed'],t,c) for c in protocol['conditions'] for t in range(protocol['mc']) if (c,t) not in groups]
    metadata={'completed_scenarios':len(groups),'pending_scenarios':len(pending),'workers':a.workers,
              'sources':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path('tools/audit_v1_rcs_joint.py')]},
              'note':'Numerical protocol unchanged; runtime spans different concurrent worker counts.'}
    (a.out/'resume.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    with (a.out/'trials.csv').open('a',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        with ProcessPoolExecutor(a.workers) as pool:
            tasks=[pool.submit(job,s) for s in pending]
            for i,task in enumerate(as_completed(tasks),1):
                batch=task.result();w.writerows(batch);f.flush();rows.extend(batch)
                if i%20==0:print(f'{len(groups)+i}/{len(groups)+len(pending)}',flush=True)
    summary={}
    metrics=[k for k in numeric if k!='trial']
    for c in protocol['conditions']:
        g={m:sorted([r for r in rows if r['condition']==c and r['method']==m],key=lambda r:r['trial']) for m in ['v1','joint','balanced','robust']}
        for batch in g.values():assert [r['trial'] for r in batch]==list(range(protocol['mc']))
        s={m:{k:float(np.mean([r[k] for r in batch])) for k in metrics} for m,batch in g.items()}
        for m in ['joint','balanced','robust']:
            diff=[x['pd']-y['pd'] for x,y in zip(g[m],g['v1'])]
            s[m+'_minus_v1']={'mean':float(np.mean(diff)),'ci95':interval(diff),
                'family_ci95':interval(diff,1-.05/(3*len(protocol['conditions'])))}
        summary[c]=s
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('Finished: '+str(a.out),flush=True)

if __name__=='__main__':main()
