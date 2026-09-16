"""Audit probability refinement without changing V1 selection or headline data."""
from pathlib import Path
import sys, csv, json, time, platform, hashlib
from dataclasses import asdict
import numpy as np
from scipy.stats import norm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from isac_sim.mixture_probability import mixture_tail, cf_detector_intervals
from isac_sim.config import Config, apply_preset, apply_overrides
from isac_sim.belief import BeliefState
from isac_sim.model import generate_geometry, build_base_gains, compute_link_tables
from isac_sim.reporting import assign_fusion_nodes
from isac_sim.selection import select_c2f_adaptive
from isac_sim.fusion import predicted_pd_for_links

def save(path, rows):
    with path.open("w", newline="", encoding="utf8") as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

out=Path("results_v1_probability_refinement"); out.mkdir(exist_ok=True)
wave=[]
with Path("results_isac_review_revision/waveform_llr.csv").open() as f:
    for row in csv.DictReader(f):
        row={k:float(v) for k,v in row.items()}
        refined=mixture_tail([row["gamma_effective"]], [row["report_success"]], [1.], row["threshold"])
        wave.append({"gamma":row["gamma_effective"],"chi":row["report_success"],
                     "old_pd":row["predicted_pd"],"refined_pd":refined.midpoint,
                     "empirical_pd":row["empirical_pd"],"original_exact":row["exact_mixture_pd"]})
save(out/"singleton.csv", wave)
multi=[]
for idx in range(12):
    rng=np.random.default_rng([10217,idx])
    n=(2,3,6)[idx%3]; g=rng.uniform(.08,.65,n); chi=rng.uniform(.3,1,n)
    factor=9. if idx<6 else 0.
    a=g/(1+g); k=16
    v0=k*a*a*(chi+(1-chi)*factor); gap=chi*k*a*g
    w=gap/v0; w=w/w.sum()
    var0=float(np.sum(w*w*v0)); mu=float(w@gap)
    v1=chi*k*a*a*(1+g)**2+(1-chi)*factor*k*a*a+chi*(1-chi)*(k*a*g)**2
    skew=float(np.sum(w**3*chi*2*k*a**3)/var0**1.5)
    z=norm.isf(.05); t=(z+skew*(z*z-1)/6)*np.sqrt(var0)
    old=float(norm.sf((t-mu)/np.sqrt(np.sum(w*w*v1))))
    start=time.perf_counter()
    bounds=mixture_tail(g,chi,w,t,failure_factor=factor)
    elapsed=time.perf_counter()-start
    samples=np.zeros(200000)
    for gi,ci,wi,ai in zip(g,chi,w,a):
        local=ai*(rng.gamma(k,1+gi,len(samples))-k)
        replacement=rng.normal(0,np.sqrt(factor*k)*ai,len(samples))
        samples += wi*np.where(rng.random(len(samples))<ci,local,replacement)
    emp=float(np.mean(samples>t)); se=np.sqrt(emp*(1-emp)/len(samples))
    multi.append({"case":idx,"n":n,"factor":factor,"old_pd":old,"refined_pd":bounds.midpoint,
                  "empirical_pd":emp,"mc_se":se,"seconds":elapsed,**asdict(bounds)})
save(out/"multilink.csv",multi)
print("Synthetic and saved waveform checks complete",flush=True)
cfg=apply_overrides(apply_preset(Config(),"target-local-v1"),
                    {"selector.score_mode":"detector_pd","detect.comm_error_model":"gaussian_replacement"})
real=[]
for trial in range(8):
    rng=np.random.default_rng([10218,trial])
    truth=generate_geometry(cfg,rng); base_truth=build_base_gains(cfg,truth,rng)
    belief=BeliefState.from_truth(cfg,truth,rng).as_geometry(truth)
    base=build_base_gains(cfg,belief,rng,channel=base_truth,rcs_view="mean")
    tables=compute_link_tables(cfg,base); plan=assign_fusion_nodes(cfg,base,tables,belief)
    selected,_,_=select_c2f_adaptive(cfg,base,tables,plan)
    fine_tables=compute_link_tables(cfg,base,dd_gain=base.eta_fine)
    for q, links in selected.items():
        old=predicted_pd_for_links(cfg,fine_tables,q,links,plan=plan,base=base)
        start=time.perf_counter()
        r=cf_detector_intervals(cfg,fine_tables,q,links,plan=plan,base=base)
        real.append({"trial":trial,"target":q,"n":len(links),"old_pd":old,
                     "refined_pd":r["pd"].midpoint,"pd_lower":r["pd"].lower,
                     "pd_upper":r["pd"].upper,"pfa_lower":r["pfa"].lower,
                     "pfa_upper":r["pfa"].upper,"resolved":r["pd"].converged and r["pfa"].converged,
                     "seconds":time.perf_counter()-start})
save(out/"selected_sets.csv",real)
summary={"singleton_max_old_vs_empirical":max(abs(r["old_pd"]-r["empirical_pd"]) for r in wave),
"singleton_max_refined_vs_empirical":max(abs(r["refined_pd"]-r["empirical_pd"]) for r in wave),
"singleton_max_vs_exact":max(abs(r["refined_pd"]-r["original_exact"]) for r in wave),
"multi_max_old_vs_empirical":max(abs(r["old_pd"]-r["empirical_pd"]) for r in multi),
"multi_max_refined_vs_empirical":max(abs(r["refined_pd"]-r["empirical_pd"]) for r in multi),
"multi_resolved":sum(r["converged"] for r in multi),"multi_cases":len(multi),
"multi_max_interval_width":max(r["upper"]-r["lower"] for r in multi),
"multi_median_seconds":float(np.median([r["seconds"] for r in multi])),
"selected_targets":len(real),"selected_resolved":sum(r["resolved"] for r in real),
"selected_median_audit_seconds":float(np.median([r["seconds"] for r in real])),
"selected_max_pd_shift":max(abs(r["old_pd"]-r["refined_pd"]) for r in real),
"scope":"fixed-statistic numerical evaluation only; no scheduler or headline rerun"}
(out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf8")
(out/"protocol.json").write_text(json.dumps({"config":asdict(cfg),"seed_multi":10217,"seed_sets":10218,
    "samples_per_multi_case":200000,"tolerance":.001,"platform":platform.platform(),
    "sources":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
               [Path(__file__),Path("isac_sim/mixture_probability.py")]}},indent=2),encoding="utf8")
print(json.dumps(summary,indent=2),flush=True)
