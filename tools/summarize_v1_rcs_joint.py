"""Summarize the user-requested RCS sweep with paired report-budget contrasts."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import interval

def main():
    out=Path('results_v1_rcs_joint')
    p=json.loads((out/'protocol.json').read_text())
    s=json.loads((out/'summary.json').read_text())
    rows=list(csv.DictReader((out/'trials.csv').open(encoding='utf8')))
    assert len(rows)==9*4*p['mc']
    methods=['v1','joint','balanced','robust']
    fields=['pd','pfa','reports','observations','active_targets','capture','bits','delay_ms']
    for c,overrides in p['conditions'].items():
        for m in methods:
            g=[r for r in rows if r['condition']==c and r['method']==m]
            assert sorted(int(r['trial']) for r in g)==list(range(p['mc']))
            for r in g:
                assert all(np.isfinite(float(r[k])) for k in fields)
                assert float(r['reports'])<=overrides['selector.max_remote_reports']
                assert float(r['observations'])<=p['base_config']['selector']['max_total_links']
                assert abs(float(r['bits'])-640*float(r['reports']))<1e-9
                assert abs(float(r['delay_ms'])-2048/1.92e6*1000*float(r['reports']))<1e-9
    def ci(x):return '区间不可估（零样本方差）' if x is None else f'[{x[0]:+.4f}, {x[1]:+.4f}]'
    contrasts=[]
    for rcs in [0.05,0.1,0.2]:
        for m in methods:
            byk={k:sorted([r for r in rows if r['condition']==f'rcs{rcs:g}_k{k}' and r['method']==m],key=lambda r:int(r['trial'])) for k in [0,2,8]}
            for k in [2,8]:
                diff=[float(a['pd'])-float(b['pd']) for a,b in zip(byk[k],byk[0])]
                contrasts.append({'rcs_m2':rcs,'method':m,'budget':k,'pd_difference':float(np.mean(diff)),
                    'ci95':interval(diff),'family_ci95':interval(diff,1-.05/24)})
    (out/'budget_contrasts.json').write_text(json.dumps(contrasts,indent=2),encoding='utf8')
    with (out/'rcs_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['rcs_m2','report_budget','method']+fields)
        w.writeheader()
        for rcs in [0.05,0.1,0.2]:
            for k in [0,2,8]:
                for m in methods:
                    w.writerow({'rcs_m2':rcs,'report_budget':k,'method':m,
                                **{key:s[f'rcs{rcs:g}_k{k}'][m][key] for key in fields}})
    lines=['# RCS = 0.05 / 0.1 / 0.2 m²：联合优化补充实验','',
        '日期：2026-09-16。本轮将用户指定的三个RCS纳入原V1、联合改进、均衡SINR与保守先验对照。', '',
        '本轮结论：低RCS下报告预算明显更活跃。RCS=0.2 m²时，原V1的K=8相对K=0检测率提高0.035，24项校正区间为[0.0022, 0.0678]。原来的“通信影响不大”只适用于高RCS配置，不能直接外推到这里。当前四种方法的绝对检测率仍低，K=8下尚未分辨出联合改进相对V1的明确检测优势。', '',
        '## 实验口径', '',
        '- 三个RCS × 报告上限K=0、2、8，每组100场景，seed=10917，与上一轮共用场景编号。900次条件评估复用100个基础场景，不是900个独立场景。',
        '- 15架无人机、10个目标，4 km部署边长，观测上限6/目标、60/系统；位置/速度先验误差150 m / 15 m/s。',
        '- RCS以m²计，truth与scheduler均使用该场景的平均RCS；没有加入RCS未知、估计偏差或波动模型。',
        '- 已直接核验模型数组：真实与调度侧target_gain随RCS按1∶2∶4缩放，direct_gain逐元素不变。',
        '- 发射功率1 W、感知功率比例0.8；Tx/Rx增益均为0 dBi、系统损耗0 dB，无额外净增益。与上一轮相比仅改变RCS和声明的报告预算。',
        '- 原对照的RCS=50 m²。三个新值分别对应比原回波功率低约30、27、24 dB；不能直接沿用原高检测率结论。',
        '- 固定包与原CF检测器保持一致。新增结果是探索性敏感性分析，不是等效性、非劣性或真实硬件性能证明。', '',
        '## 检测概率与报告用量', '',
        '每格为平均检测概率 / 平均远程报告数。', '',
        '| RCS(m²) | 报告上限 | 原V1 | 联合改进 | 均衡SINR | 保守先验 |',
        '|---:|---:|---:|---:|---:|---:|']
    for rcs in [0.05,0.1,0.2]:
        for k in [0,2,8]:
            x=s[f'rcs{rcs:g}_k{k}']
            lines.append(f'| {rcs:g} | {k} | '+' | '.join(f"{x[m]['pd']:.3f} / {x[m]['reports']:.2f}" for m in methods)+' |')
    lines += ['', '## 通信预算影响：K=8减K=0', '',
        '| RCS(m²) | 方法 | 检测差 | 配对95%区间 | 24项预算比较校正区间 |',
        '|---:|---|---:|---|---|']
    for r in contrasts:
        if r['budget']==8:
            lines.append(f"| {r['rcs_m2']:g} | {r['method']} | {r['pd_difference']:+.4f} | {ci(r['ci95'])} | {ci(r['family_ci95'])} |")
    lines += ['', 'K=2减K=0的全部结果另存budget_contrasts.json。统计以场景为配对单位，区间跨零不代表等效。', '',
        '## K=8下的算法差异', '',
        '| RCS(m²) | 候选减V1 | 检测差 | 27项算法比较校正区间 |',
        '|---:|---|---:|---|']
    for rcs in [0.05,0.1,0.2]:
        for m in ['joint','balanced','robust']:
            r=s[f'rcs{rcs:g}_k8'][m+'_minus_v1']
            lines.append(f"| {rcs:g} | {m} | {r['mean']:+.4f} | {ci(r['family_ci95'])} |")
    lines += ['', '## K=8下的资源与虚警检查', '',
        '| RCS(m²) | 方法 | 观测数 | 整体虚警率 | 目标覆盖数 | 报告预算用满比例 |',
        '|---:|---|---:|---:|---:|---:|']
    for rcs in [0.05,0.1,0.2]:
        c=f'rcs{rcs:g}_k8'
        for m in methods:
            r=s[c][m]
            binding=np.mean([float(v['reports'])>=8 for v in rows if v['condition']==c and v['method']==m])
            lines.append(f"| {rcs:g} | {m} | {r['observations']:.2f} | {r['pfa']:.4f} | {r['active_targets']:.2f} | {binding:.0%} |")
    lines += ['', '各方法使用相同检测器与随机流，但未重新校准至相同实测虚警率，因此不称严格等虚警ROC比较。',
        '运行中将并发数由4增加到12，已保存resume.json与入口源码哈希；数值配置未改变，已完成行保留。选择时间受并发负载影响，本轮不据此作算法加速结论。', '',
        '## 文件与复现', '',
        '- `tools/audit_v1_rcs_joint.py --mc 100 --seed 10917 --workers 12 --out <新目录>`',
        '- `results_v1_rcs_joint/protocol.json`：基础配置与逐条件覆盖值；最终RCS以conditions为准。',
        '- `results_v1_rcs_joint/trials.csv`：3600条方法级记录。',
        '- `results_v1_rcs_joint/summary.json`：36个方法/条件组合及配对算法差异。',
        '- `results_v1_rcs_joint/rcs_summary.csv`：显式列出RCS、预算、方法及主要指标，便于后续制表。',
        '- `results_v1_rcs_joint/budget_contrasts.json`：24项配对预算差异。',
        '- 汇总检查通过：无重复或缺失场景，所有报告/观测预算满足，数值有限，载荷与串行时延和报告数一致。', '']
    Path('V1_RCS_JOINT_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('Validated 3600 rows; wrote V1_RCS_JOINT_REPORT.md')

if __name__=='__main__':main()
