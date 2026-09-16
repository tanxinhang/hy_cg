"""Build a concise Chinese audit report from completed, immutable trial files."""
import csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_exact_budget import interval


def main():
    root=Path('results_v1_joint_revision_independent')
    summary=json.loads((root/'summary.json').read_text())
    small=json.loads(Path('results_v1_joint_small/summary.json').read_text())
    extra=json.loads(Path('results_v1_prior_communication/summary.json').read_text())
    rows=list(csv.DictReader((root/'trials.csv').open(encoding='utf8')))
    more=list(csv.DictReader(Path('results_v1_prior_communication/trials.csv').open(encoding='utf8')))
    allrows=rows+more
    def paired(c1,c0,method):
        a={int(r['trial']):float(r['pd']) for r in allrows if r['condition']==c1 and r['method']==method}
        b={int(r['trial']):float(r['pd']) for r in allrows if r['condition']==c0 and r['method']==method}
        assert set(a)==set(b) and len(a)==100
        d=[a[t]-b[t] for t in sorted(a)]
        return np.mean(d),interval(d),interval(d,1-.05/13)
    def fmt_ci(x):
        return '无法由零样本方差推断等效' if x is None else f'[{x[0]:+.4f}, {x[1]:+.4f}]'
    lines=['# V1 通信、先验与联合选择优化记录','',
        '日期：2026-09-16。按本轮用户反馈控制会议论文工作量；新增实验模块，默认算法、物理模型和旧主结果未替换。',
        '', '本轮判断：当前通信预算不是主要检测瓶颈；大先验误差下保守选择具有明确收益，但主要依赖增加观测。联合局部搜索缩小了小规模预测目标差距，原规模真实检测收益尚不明显。优先推进不确定性驱动的证据选择，通信部分保留为报告效率与预算约束。',
        '', '## 1. 本轮完成范围', '',
        '- 新增 `isac_sim/joint_polish.py`：在C2F结果后联合尝试融合位置、增加、删除、同目标换入换出，以及换节点后单观测重启。',
        '- 固定两轮搜索；每个目标保留原节点及粗级接收强度前三节点，候选保留全局前六与本地前六。只接受原预测目标严格增加的可行解。',
        '- 复用已有 `geometry_robust_base` 比较先验鲁棒策略，没有把保守路径增益包装为完整跟踪器。',
        '- 独立seed=10917，每条件100场景，原规模15无人机/10目标；各条件共享场景编号，不能累加为1000个独立场景。',
        '- 小规模独立seed=10918，M=3/4、Q=2，每种规模100场景，四个预算点进行完整枚举比较。',
        '- 不增加真实编译码、量化、硬件或完整多目标波形实验。',
        '', '## 2. 原规模结果', '',
        '每格为检测概率 / 平均远程报告数。除cap12外观测上限仍为60，不另收费或限制本地观测。', '',
        '| 条件 | 原V1 | 联合改进 | 均衡SINR | 保守先验 |',
        '|---|---:|---:|---:|---:|']
    for c,s in {**summary,**extra}.items():
        lines.append('| '+c+' | '+' | '.join(f"{s[m]['pd']:.3f} / {s[m]['reports']:.2f}" for m in ['v1','joint','balanced','robust'])+' |')
    binding=np.mean([float(r['reports'])>=8 for r in rows if r['condition']=='k8' and r['method']=='v1'])
    lines += ['', f"名义K=8下原V1平均使用{summary['k8']['v1']['reports']:.2f}个报告，预算用满比例{binding:.1%}。硬预算是否活跃，与通信价格是否影响选择，是两个不同问题。", '', '## 3. 通信是否真正影响检测', '',
        'K=0/2/8分别代表最多0/2/8个远程包；固定包2048 channel uses、带宽1.92 MHz，对应串行报告时长上限0/2.133/8.533 ms。上限不强制用满。',
        '以下比较同场景下K=8减K=0的实际检测率；误差场景的零报告控制是补充诊断，不是新的独立确认样本。', '',
        '| 先验条件 | 方法 | 检测差 | 配对95%区间 | 十三项比较校正区间 |',
        '|---|---|---:|---|---|']
    for c1,c0 in [('k8','k0'),('prior250','prior250_k0'),('prior500','prior500_k0')]:
        for m in ['v1','joint','robust']:
            mean,ci,family=paired(c1,c0,m)
            lines.append(f'| {c1} | {m} | {mean:+.4f} | {fmt_ci(ci)} | {fmt_ci(family)} |')
    lines += ['', '不能仅靠降低通信预算就保证通信成为性能瓶颈：本地观测仍可绕过报告链。若这些差异有限，应该收窄为报告效率结论，而不是人为添加没有工程依据的本地成本。',
        '', '## 4. 先验优化与算法真实检测收益', '',
        '| 条件 | 候选减V1 | 检测差 | 配对95%区间 | 原八条件24项比较校正区间 |',
        '|---|---|---:|---|---|']
    for c in ['k8','prior250','prior500','cap12']:
        for m in ['joint','robust']:
            s=summary[c][m+'_minus_v1']
            lines.append(f"| {c} | {m} | {s['mean']:+.4f} | {fmt_ci(s['ci95'])} | {fmt_ci(s['family_ci95'])} |")
    lines += ['', '保守先验比较改变选择，并可能增加观测或报告用量，不能解读为等实际资源增益。联合搜索只保证所给预测目标不下降，不保证真实检测率。大误差仍属于已有目标的再检测，盲初检和跟踪闭环未完成。',
        '', '## 5. 两个效用项消融', '',
        '同seed、同预算K=8，单独关闭soft-min或二次缺口项；对原V1比较，未重调其他参数。', '',
        '| 消融减原V1 | 检测差 | 配对95%区间 | 十三项比较校正区间 |',
        '|---|---:|---|---|']
    for c in ['no_softmin','no_deficit']:
        mean,ci,family=paired(c,'k8','v1')
        lines.append(f'| {c} | {mean:+.4f} | {fmt_ci(ci)} | {fmt_ci(family)} |')
    lines += ['', '上述十三项校正保守覆盖9项通信差、2项消融，以及2项先验误差效应；不把差异不显著称为等效或证明该项必要。',
        '', '## 6. 联合优化与小规模最优差距', '',
        '| 规模/预算 | 原平均目标差距 | 改进后 | 原达最优比例 | 改进后 |',
        '|---|---:|---:|---:|---:|']
    for c,s in small.items():
        a,b=s['total_gap'],s['polished_gap']
        lines.append(f"| {c} | {a['mean']:.6f} | {b['mean']:.6f} | {a['optimal_fraction']:.0%} | {b['optimal_fraction']:.0%} |")
    lines += ['', '历史两个反例回归通过。所有小规模改进值不超过枚举最优值。搜索范围有限、没有跨目标交换，不声称全局最优或近似比。',
        '', '## 7. 计算及成本口径', '',
        '| 条件 | V1选择时间中位数(s) | 联合改进(s) | 联合额外PD评估均值 |',
        '|---|---:|---:|---:|']
    for c in ['k8','prior250','prior500','cap12']:
        times={m:np.median([float(r['selector_seconds']) for r in rows if r['condition']==c and r['method']==m]) for m in ['v1','joint']}
        lines.append(f"| {c} | {times['v1']:.3f} | {times['joint']:.3f} | {summary[c]['joint']['extra_pd_evaluations']:.1f} |")
    lines += ['', '时间为本机多进程实验中的选择阶段观测，不是严格隔离的性能基准。联合改进计入额外精细表构造；共享几何、信道预处理和检测阶段不计入。该候选不能沿用原92.9%精细计算下降结论。',
        '载荷和串行报告时间继续用已有固定包模型；头部、控制、路由、重传及真实计算能耗没有可信参数，本轮不虚构数值，也不宣称完整系统成本。',
        '', '## 8. 文件与复现', '',
        '- `tools/audit_v1_joint_revision.py --mc 100 --seed 10917 --out <新目录>`',
        '- `tools/audit_v1_prior_communication.py --mc 100 --seed 10917 --workers 2 --conditions prior250_k0 prior500_k0 --out <新目录>`',
        '- `tools/audit_v1_joint_small.py`（固定输出目录，已有结果时拒绝覆盖）。',
        '- 三组结果目录保存协议、逐场景CSV、汇总JSON；原规模协议包含全部isac_sim源码哈希。',
        '- `tests/test_joint_polish.py`：历史反例、枚举上界、输入不变、预算可行性和不支持条件检查。',
        '- 本轮全量测试：164 passed，6 subtests passed。',
        '', '本轮结果是独立种子的探索性验证，未完成预注册非劣性检验。是否把候选升为主方法，应同时看实际检测、报告数量与新增计算，不能仅看优化目标值。', '']
    Path('V1_JOINT_REVISION_REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    contrasts={}
    for c in ['prior250','prior500']:
        mean,ci,family=paired(c,'k8','v1')
        contrasts[c+'_minus_nominal']={'mean':float(mean),'ci95':ci,'family_ci95':family}
    Path('results_v1_joint_revision_independent/prior_contrasts.json').write_text(json.dumps(contrasts,indent=2),encoding='utf8')

if __name__=='__main__':main()
