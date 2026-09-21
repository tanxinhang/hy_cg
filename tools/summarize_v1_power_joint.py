"""Report paired detection, allocations and cost of joint node power control."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import interval

LABELS={'v1':'原V1','joint_fixed_power':'原联合（固定功率）',
        'uniform_power_joint':'统一比例优化+联合',
        'node_power_fixed_set':'逐节点功率（固定观测集阶段）',
        'power_joint':'功率+观测+融合联合',
        'balanced_fixed_power':'均衡SINR（固定功率）'}
FIELDS=['pd','pfa','reports','observations','active_targets','capture','bits','delay_ms',
        'sensing_allocated_w','communication_allocated_w','rho_mean','rho_std']

def main():
    out=Path('results_v1_power_joint')
    p=json.loads((out/'protocol.json').read_text())
    s=json.loads((out/'summary.json').read_text())
    rows=list(csv.DictReader((out/'trials.csv').open(encoding='utf8')))
    assert len(rows)==p['mc']*len(p['rcs'])*len(LABELS)
    for rcs in p['rcs']:
        for m in LABELS:
            group=[r for r in rows if float(r['rcs_m2'])==rcs and r['method']==m]
            assert sorted(int(r['trial']) for r in group)==list(range(p['mc']))
            for r in group:
                assert all(np.isfinite(float(r[k])) for k in FIELDS)
                rho=np.array(json.loads(r['rho_vector']))
                assert rho.shape==(15,) and np.all((rho>0)&(rho<1))
                assert set(np.round(rho,12))<=set(p['grid'])
                assert float(r['reports'])<=8 and float(r['observations'])<=60
                assert abs(float(r['sensing_allocated_w'])+float(r['communication_allocated_w'])-15)<1e-10
                assert abs(float(r['sensing_allocated_w'])-rho.sum())<1e-10
                assert abs(float(r['bits'])-640*float(r['reports']))<1e-9
                assert abs(float(r['delay_ms'])-2048/1.92e6*1000*float(r['reports']))<1e-9
                if m=='power_joint':assert float(r['predicted_objective_gain'])>=-1e-12
    with (out/'power_summary.csv').open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=['rcs_m2','method']+FIELDS);w.writeheader()
        for rcs in p['rcs']:
            for m in LABELS:w.writerow({'rcs_m2':rcs,'method':m,**{k:s[str(rcs)][m][k] for k in FIELDS}})
    def ci(x):return '零样本方差，不能由此推断等效' if x is None else f'[{x[0]:+.4f}, {x[1]:+.4f}]'
    lines=['# 600米场景：联合感知/通信功率分配','',
        '本轮新增逐无人机功率拆分，与观测集合、融合节点交替优化；保持每架无人机1 W预算，未增加天线增益。', '',
        '检测率（固定功率的原联合 → 功率联合）：'+ '；'.join(
            f"RCS={r:g} m²：{s[str(r)]['joint_fixed_power']['pd']:.3f} → {s[str(r)]['power_joint']['pd']:.3f}"
            for r in p['rcs'])+'。以下给出配对区间、统一功率比例对照及额外计算成本。', '',
        '## 变量与约束', '',
        '- 每机感知分配 P_s,i=ρ_i×1 W，报告分配 P_c,i=(1−ρ_i)×1 W。每机总分配恒为1 W，全系统恒为15 W。',
        '- ρ_i取离散网格{0.2, 0.5, 0.8, 0.95}。本轮优化每机内部的感知/通信比例；没有跨无人机转移总功率额度。',
        '- 在现有串行正交报告模型下，感知连续辐射，报告功率按报告时段使用。因此等功率分配预算不等于等实际平均功耗或等能量。',
        '- 每次功率更新都重建感知信号、跨节点泄漏干扰、通信信号、FBL可靠性及报告可行性；原固定总功率自干扰底项保留。',
        '- 几何为600 m区域，15架无人机/10目标，位置/速度先验误差150 m / 15 m/s，报告上限8、观测上限6/目标和60/系统。',
        '- RCS仍为0.05、0.1、0.2 m²，固定平均RCS；其余物理参数保持上一轮设置。', '',
        '## 算法与对照', '',
        '先以原联合结果为初值，比较统一ρ候选并重新选择/细化；再进行两轮逐节点功率搜索、观测换入换出、融合节点调整与C2F重启。只接受原预测目标严格改善且满足约束的候选。',
        '“统一比例优化+联合”包含每个统一ρ下的观测重选和融合优化；“逐节点功率（固定观测集阶段）”固定的是该统一比例阶段输出的观测集，并非最初V1集合。',
        '调度只读取belief；最终使用相同功率分配重建truth检测。预测目标单调改进不等于真实检测单调，也没有全局最优保证。',
        f"新seed={p['seed']}，每RCS {p['mc']}场景，六种方法使用配对随机流；三档RCS复用场景编号。此次烟雾检查与正式首场景重合，未据此调参；属于探索性对照。", '',
        '## 检测率', '',
        '| RCS(m²) | 原V1 | 原联合 | 统一比例+联合 | 逐节点固定集阶段 | 功率联合 | 均衡SINR |',
        '|---:|---:|---:|---:|---:|---:|---:|']
    for rcs in p['rcs']:
        lines.append(f'| {rcs:g} | '+' | '.join(f"{s[str(rcs)][m]['pd']:.3f}" for m in LABELS)+' |')
    lines += ['', '## 功率联合相对对照的配对检测差', '',
        '| RCS(m²) | 对照 | 检测差 | 95%区间 | 12项比较校正区间 |',
        '|---:|---|---:|---|---|']
    for rcs in p['rcs']:
        for m in ['v1','joint_fixed_power','uniform_power_joint','balanced_fixed_power']:
            d=s[str(rcs)]['power_joint_minus_'+m]
            lines.append(f"| {rcs:g} | {LABELS[m]} | {d['mean']:+.4f} | {ci(d['ci95'])} | {ci(d['family_ci95'])} |")
    lines += ['', '## 实际资源使用与虚警', '',
        '| RCS(m²) | 方法 | 报告数 | 观测数 | 整体虚警率 | 分配给感知的总功率(W) | 节点间ρ标准差均值 |',
        '|---:|---|---:|---:|---:|---:|---:|']
    for rcs in p['rcs']:
        for m in LABELS:
            d=s[str(rcs)][m]
            lines.append(f"| {rcs:g} | {LABELS[m]} | {d['reports']:.2f} | {d['observations']:.2f} | {d['pfa']:.4f} | {d['sensing_allocated_w']:.2f} | {d['rho_std']:.3f} |")
    lines += ['', '共同上限不意味着相同实际观测或报告用量。所有方法使用同一CF检测器，但未重新校准到相同实测虚警率，不称严格等虚警ROC比较。', '',
        '## 阶段消融（探索性机制检查）', '',
        '| RCS(m²) | 阶段差异 | 检测差 | 95%区间 | 本节6项校正区间 |',
        '|---:|---|---:|---|---|']
    stage_contrasts=[]
    for rcs in p['rcs']:
        for after,before in [('node_power_fixed_set','uniform_power_joint'),('power_joint','node_power_fixed_set')]:
            groups={m:sorted([r for r in rows if float(r['rcs_m2'])==rcs and r['method']==m],key=lambda r:int(r['trial'])) for m in [after,before]}
            if before=='uniform_power_joint':
                for a,b in zip(groups[after],groups[before]):
                    for key in ['reports','observations','bits','delay_ms']:
                        assert float(a[key])==float(b[key]),(rcs,a['trial'],key)
            ds=[float(a['pd'])-float(b['pd']) for a,b in zip(groups[after],groups[before])]
            d={'rcs_m2':rcs,'after':after,'before':before,'mean':float(np.mean(ds)),
               'ci95':interval(ds),'family_ci95':interval(ds,1-.05/6)}
            stage_contrasts.append(d)
            lines.append(f"| {rcs:g} | {LABELS[after]} − {LABELS[before]} | {d['mean']:+.4f} | {ci(d['ci95'])} | {ci(d['family_ci95'])} |")
    (out/'power_stage_contrasts.json').write_text(json.dumps(stage_contrasts,indent=2),encoding='utf8')
    lines += ['', '第一项固定观测集合和融合节点，隔离逐节点功率搜索的影响；第二项同时包含后续拓扑更新与第二轮功率优化，不能单独解释为拓扑收益。此节与前述12项主对照分开列为探索性机制检查。', '',
        '## 计算开销', '',
        '| RCS(m²) | 功率搜索耗时中位数(s) | 平均表重建次数 | 平均接受功率步数 | 预测目标平均增益 |',
        '|---:|---:|---:|---:|---:|']
    for rcs in p['rcs']:
        d=s[str(rcs)]['power_joint']
        secs=[float(r['power_search_seconds']) for r in rows if float(r['rcs_m2'])==rcs and r['method']=='power_joint']
        lines.append(f"| {rcs:g} | {np.median(secs):.2f} | {d['table_builds']:.1f} | {d['accepted_power_moves']:.1f} | {d['predicted_objective_gain']:.5f} |")
    lines += ['', '这些时间来自12进程并发实验，覆盖新增功率搜索及其重选，不包含原V1/原联合初值、共享几何和最终检测。当前实现是离线候选，不宣称满足毫秒级在线调度，也不能沿用旧C2F的92.9%计算下降结论。', '',
        '## 实现、验证与复现', '',
        '- `isac_sim/core/config.py` 新增radio.rho_by_uav；None保持旧统一ρ行为。',
        '- `isac_sim/sensing/model.py` 将逐节点分配同时用于感知和报告信号/干扰，启用功率比例向量时关闭潜在过期的感知表复用。',
        '- `isac_sim/power_joint.py` 实现有界交替搜索；默认算法列表未自动替换。',
        '- `tests/test_power_joint.py` 检查旧模型逐字段一致、感知/通信交叉干扰、非法分配、预算、预测目标单调性，以及truth检测确实使用新功率。',
        '- 全量测试172 passed、6 subtests passed；新增固定集阶段不改变观测集合/融合位置的断言也通过。',
        '- `tools/audit_v1_power_joint.py --mc 100 --workers 12 --seed 10919 --out <新目录>`。',
        '- `results_v1_power_joint` 保存协议、源码哈希、逐场景ρ向量、原始记录、汇总JSON与power_summary.csv。',
        '- 汇总检查：1800条记录无重复或缺失；所有功率/报告/观测约束成立；数值有限；载荷、时延与报告数一致。', '']
    Path('V1_POWER_JOINT_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    print('Validated power/observation/report budgets; wrote V1_POWER_JOINT_REPORT.md')

if __name__=='__main__':main()
