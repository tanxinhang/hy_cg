"""Validate and summarize 600 m deployment results and paired 4 km controls."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import config,interval
from isac_sim.config import apply_overrides
from isac_sim.model import generate_geometry

METHODS=['v1','joint','balanced','robust']
FIELDS=['pd','pfa','reports','observations','active_targets','capture','bits','delay_ms']

def main():
    out=Path('results_v1_rcs_600m')
    p=json.loads((out/'protocol.json').read_text())
    summary=json.loads((out/'summary.json').read_text())
    rows=list(csv.DictReader((out/'trials.csv').open(encoding='utf8')))
    oldroot=Path('results_v1_rcs_joint')
    oldp=json.loads((oldroot/'protocol.json').read_text())
    old=list(csv.DictReader((oldroot/'trials.csv').open(encoding='utf8')))
    assert p['seed']==oldp['seed'] and p['mc']==oldp['mc']==100
    assert p['base_config']==oldp['base_config']
    assert p['hashes']==oldp['hashes']
    assert len(rows)==3600
    for c,overrides in p['conditions'].items():
        assert overrides=={**oldp['conditions'][c],'geometry.area_xy':600.}
        for m in METHODS:
            g=[r for r in rows if r['condition']==c and r['method']==m]
            assert sorted(int(r['trial']) for r in g)==list(range(100))
            for r in g:
                assert all(np.isfinite(float(r[k])) for k in FIELDS)
                assert float(r['reports'])<=overrides['selector.max_remote_reports']
                assert float(r['observations'])<=60
                assert 0<=float(r['pd'])<=1 and 0<=float(r['pfa'])<=1
                assert abs(float(r['bits'])-640*float(r['reports']))<1e-9
                assert abs(float(r['delay_ms'])-2048/1.92e6*1000*float(r['reports']))<1e-9
    geomstats={}
    for width in [4000.,600.]:
        cfg=apply_overrides(config(p['seed'],8),{'geometry.area_xy':width})
        ds=[];comm=[]
        for t in range(100):
            geom=generate_geometry(cfg,np.random.default_rng([p['seed'],t]))
            ds.extend(np.linalg.norm(geom.p_uav[:,None,:]-geom.p_tgt[None,:,:],axis=-1).ravel())
            mat=np.linalg.norm(geom.p_uav[:,None,:]-geom.p_uav[None,:,:],axis=-1)
            comm.extend(mat[np.triu_indices(cfg.scale.M,k=1)])
        geomstats[str(int(width))]={}
        for name,values in [('uav_target_slant_m',ds),('uav_uav_slant_m',comm)]:
            geomstats[str(int(width))][name]={k:float(v) for k,v in zip(
                ['min','p05','median','mean','p95','max'],
                [np.min(values),np.quantile(values,.05),np.median(values),np.mean(values),np.quantile(values,.95),np.max(values)])}
    (out/'geometry.json').write_text(json.dumps(geomstats,indent=2),encoding='utf8')
    metadata={'user_distance_semantics':'horizontal deployment side length 600 m; altitudes unchanged',
        'additional_source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
            [Path('tools/audit_v1_rcs_600m.py'),Path(__file__)]}}
    (out/'comparison_protocol.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    def values(data,c,m):return sorted([r for r in data if r['condition']==c and r['method']==m],key=lambda r:int(r['trial']))
    def diff(a,b,n):
        assert [r['trial'] for r in a]==[r['trial'] for r in b]
        ds=[float(x['pd'])-float(y['pd']) for x,y in zip(a,b)]
        return {'mean':float(np.mean(ds)),'ci95':interval(ds),'family_ci95':interval(ds,1-.05/n)}
    contrasts={};budgets={}
    for c in p['conditions']:
        contrasts[c]={m:diff(values(rows,c,m),values(old,c,m),36) for m in METHODS}
    for rcs in [0.05,0.1,0.2]:
        budgets[str(rcs)]={m:diff(values(rows,f'rcs{rcs:g}_k8',m),values(rows,f'rcs{rcs:g}_k0',m),12) for m in METHODS}
    (out/'paired_geometry_contrasts.json').write_text(json.dumps(contrasts,indent=2),encoding='utf8')
    (out/'paired_budget_contrasts.json').write_text(json.dumps(budgets,indent=2),encoding='utf8')
    with (out/'rcs_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['area_side_m','rcs_m2','report_budget','method']+FIELDS)
        w.writeheader()
        for rcs in [0.05,0.1,0.2]:
            for k in [0,2,8]:
                for m in METHODS:
                    w.writerow({'area_side_m':600,'rcs_m2':rcs,'report_budget':k,'method':m,
                        **{key:summary[f'rcs{rcs:g}_k{k}'][m][key] for key in FIELDS}})
    def ci(x):return '零样本方差，不能据此证明等效' if x is None else f'[{x[0]:+.4f}, {x[1]:+.4f}]'
    lines=['# 600米部署区域：三档RCS性能对照','',
        '用户已确认600米指水平部署区域边长。保留原高度分布；不是所有无人机到目标均相距600米。', '',
        '结果：报告上限K=8时，原V1在三档RCS下的检测率为0.475、0.580、0.702，对应4 km时的0.093、0.136、0.202。区域缩小带来的三项提升均通过多重比较校正。联合改进的额外收益较小；均衡SINR的检测率更高，但使用了更多观测。原V1的K=8相对K=0增益在三档RCS下均未通过12项校正，不能把更近部署的整体改善等同于通信协同增益。', '',
        '## 设置与实际距离', '',
        '- RCS=0.05、0.1、0.2 m²；报告上限K=0、2、8；每条件100场景、seed=10917，与4 km结果逐场景配对。',
        '- 15架无人机、10目标；无人机高度800–1200 m，目标高度700–1500 m；位置/速度先验误差150 m / 15 m/s。',
        '- 功率1 W、感知占比0.8、Tx/Rx天线增益0 dBi、系统损耗0 dB、无额外净增益；观测上限6/目标、60/系统。',
        '- 固定平均RCS、失败高斯替代和原CF检测器保持不变。九个条件共享100个基础场景，不视为900个独立场景。', '',
        '| 区域边长 | UAV—目标斜距均值 | 中位数 | 5%—95%范围 |',
        '|---:|---:|---:|---:|']
    for width in ['4000','600']:
        g=geomstats[width]['uav_target_slant_m']
        lines.append(f"| {width} m | {g['mean']:.1f} m | {g['median']:.1f} m | {g['p05']:.1f}—{g['p95']:.1f} m |")
    lines += ['', '区域收缩同时改变感知路径、无人机间通信路径及干扰几何，结果属于完整紧凑部署对照，不能归因为只改变雷达路径损耗。', '',
        '## 600米区域检测率与报告数', '', '每格为平均检测概率 / 平均远程报告数。', '',
        '| RCS(m²) | 报告上限 | 原V1 | 联合改进 | 均衡SINR | 保守先验 |',
        '|---:|---:|---:|---:|---:|---:|']
    for rcs in [0.05,0.1,0.2]:
        for k in [0,2,8]:
            s=summary[f'rcs{rcs:g}_k{k}']
            lines.append(f'| {rcs:g} | {k} | '+' | '.join(f"{s[m]['pd']:.3f} / {s[m]['reports']:.2f}" for m in METHODS)+' |')
    lines += ['', '## 原V1：600米与4公里比较（K=8）', '',
        '| RCS(m²) | 4 km检测率 | 600 m检测率 | 配对差 | 36项比较校正区间 |',
        '|---:|---:|---:|---:|---|']
    oldsummary=json.loads((oldroot/'summary.json').read_text())
    for rcs in [0.05,0.1,0.2]:
        c=f'rcs{rcs:g}_k8';d=contrasts[c]['v1']
        lines.append(f"| {rcs:g} | {oldsummary[c]['v1']['pd']:.3f} | {summary[c]['v1']['pd']:.3f} | {d['mean']:+.3f} | {ci(d['family_ci95'])} |")
    lines += ['', '## 600米区域：报告预算K=8减K=0', '',
        '| RCS(m²) | 方法 | 检测差 | 配对95%区间 | 12项比较校正区间 |',
        '|---:|---|---:|---|---|']
    for rcs in [0.05,0.1,0.2]:
        for m in METHODS:
            d=budgets[str(rcs)][m]
            lines.append(f"| {rcs:g} | {m} | {d['mean']:+.4f} | {ci(d['ci95'])} | {ci(d['family_ci95'])} |")
    lines += ['', '## 算法比较（K=8，相对原V1）', '',
        '| RCS(m²) | 方法 | 检测差 | 配对95%区间 | 27项比较校正区间 |',
        '|---:|---|---:|---|---|']
    for rcs in [0.05,0.1,0.2]:
        for m in ['joint','balanced','robust']:
            d=summary[f'rcs{rcs:g}_k8'][m+'_minus_v1']
            lines.append(f"| {rcs:g} | {m} | {d['mean']:+.4f} | {ci(d['ci95'])} | {ci(d['family_ci95'])} |")
    lines += ['', '## 资源与虚警（K=8）', '',
        '| RCS(m²) | 方法 | 观测数 | 整体虚警率 | 目标覆盖 | DD门捕获率 |',
        '|---:|---|---:|---:|---:|---:|']
    for rcs in [0.05,0.1,0.2]:
        for m in METHODS:
            s=summary[f'rcs{rcs:g}_k8'][m]
            lines.append(f"| {rcs:g} | {m} | {s['observations']:.2f} | {s['pfa']:.4f} | {s['active_targets']:.2f} | {s['capture']:.4f} |")
    lines += ['', '所有方法使用相同检测器和场景随机流；未重新校准至相同实测虚警率。检测差异不显著不等于等效。', '',
        '## 复现与检查', '',
        '- 运行：`tools/audit_v1_rcs_600m.py --mc 100 --seed 10917 --workers 12 --out <新目录>`。',
        '- 输出目录：`results_v1_rcs_600m`；包含protocol.json、trials.csv、summary.json、rcs_summary.csv及两组配对区间。',
        '- 几何数据保存在geometry.json；入口脚本及汇总脚本哈希保存在comparison_protocol.json。',
        '- 3600条记录检查通过：场景无重复/缺失，所有报告与观测预算满足，数值有限，载荷/串行时延与报告数一致。',
        '- 4 km与600 m的基础配置和模型源码哈希一致，逐条件覆盖值只相差geometry.area_xy。', '']
    Path('V1_RCS_600M_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('Validated 3600 rows and paired geometry protocol; wrote V1_RCS_600M_REPORT.md')

if __name__=='__main__':main()
