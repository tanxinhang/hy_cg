# V2 研究框架：面向弱目标保护的分布式 OTFS-ISAC 接收与协同观测选择

> 状态：自 2026-09-22 起，本文件是方向 3 的唯一研究主线。`README.md` 第 1--11 节仅作历史追溯；阵形、功率、动态 looks、轨迹和 rescue 结果不得混入 V2 核心贡献。

## 1. 一句话论点

在多无人机 OTFS-ISAC 中，我们研究强直达分量和强目标回波掩盖弱目标的问题，先用目标保护干扰消除（TP-UIC）从实际接收信号得到残差协方差，再以该协方差驱动信息感知的 UAV 观测关联；论文是否成立，分别由“抑制干扰而不损伤目标”和“在相同报告预算下改善最弱目标检测”两组证据决定。

这句话同时给出本文的边界：本文不联合优化轨迹、功率、looks、带宽、融合权重和包长，也不把“大而全系统”作为贡献。

## 2. 术语与符号账本

| 统一术语 | 首次定义 | 不再使用的混用 |
|---|---|---|
| TP-UIC | target-preserving unified interference cancellation；目标保护干扰消除 | generic IC、receiver certificate、soft/hard receiver 混称 |
| UAV sensing association | UAV 感知关联，即在预算下选择接收 UAV/观测 | link selection、coordination、scheduling 混称 |
| observation | 一个发射端--接收端--目标对应的可融合观测 | link、view、report 不加区分 |
| residual covariance | TP-UIC 后由实际残差估计的协方差 $\mathbf R_{\rm res}$ | 人为设置的 `kappa_dc` 或规划中性残差 |
| weak-target detection | 最低 RCS 目标的检测概率 $P_D^{\rm weak}$ | 只报平均 $P_D$ |
| worst-target detection | $P_D^{\rm worst}=\min_q P_{D,q}$ | overall/network P_D 未说明聚合规则 |
| All-UAV reference | 无报告预算限制的合作参考 | 必须被 proposed 战胜的 baseline |
| Oracle-$K$ | 相同预算下穷举得到的最优子集 | receiver oracle 或 perfect-channel 上界 |

## 3. 研究问题与准入门槛

### RQ1：TP-UIC 是否存在独立的算法空间？

比较 No-IC、LS/ZF-IC、SIC、TP-UIC 和 Oracle-IC，在统一 $P_{FA}$ 下同时报告：

$$
G_I=10\log_{10}\frac{P_I^{\rm before}}{P_I^{\rm after}},\qquad
L_T=-10\log_{10}\eta^{\rm protect},
$$

以及 $P_D^{\rm weak}$ 或门限无关 AUC。只有当 TP-UIC 相比常规 IC 显示更好的“干扰抑制--目标保留”折中，并在高冲突区间带来检测收益，才能把它列为贡献 1。

当前证据边界：现有 400 m、16-look 主工作点中 No-IC、TP-UIC Stage 1、TP-UIC full 和 perfect-channel 的 worst $P_D$ 均约为 0.738。这里 TP-UIC 不是活动瓶颈，因此该结果只能作为负结果和压力场景设计依据，不能作为创新成立的证据。

2026-09-22 receiver-only 强干扰筛查进一步给出：TP-UIC full 虽达到约 12.97 dB
中位对消深度和约 1.000 的目标存活率，但在 +30 dB 直达增益下相对 plain LS 的
18 个成对单元全部 AUC 持平；perfect-channel 的成对中位 AUC 增益则为 +0.14。
因此当前实现未通过 RQ1 准入门槛，但 receiver 仍存在可恢复空间。完整裁决见
[`TPUIC_FEASIBILITY_VERDICT.md`](TPUIC_FEASIBILITY_VERDICT.md)。

随后补齐直达 delay/Doppler 字典失配协方差后，强干扰下 TP-UIC 的中位 AUC 从
0.500 提升到 0.542，但相对 plain LS 的 18 个配对仍为 2 胜、16 平、0 负，成对
中位增益为 0。该修复改善了白化标定，没有改变 RQ1 的“不通过”裁决。

Detection-aware gate 进一步按 belief-side GLRT 信息量在三种现有消除算子之间选择。
它在最高冲突筛查中 24/24 次选择 TP-UIC full，但 +30 dB AUC 仍与 plain LS 相同。
因此 RQ1 已触发停止规则：现有 TP-UIC 降级为工程预处理，不再进入大样本正式轮。

### RQ2：信息感知关联是否存在优于 SINR Top-$K$ 的空间？

在 $M=8,K=3$ 的诊断场景中穷举 ${8\choose3}=56$ 个 UAV 子集，计算每个子集的 $P_D^{\rm weak}$、$P_D^{\rm worst}$、信息效用和平均 SINR。

准入条件不是 proposed 先赢，而是先证明：

1. Oracle-$K$ 对 SINR Top-$K$ 有可复现的非微小优势；
2. 存在 SINR 更高但检测更差的子集对；
3. 优势可由残差相关性、几何冗余或强弱目标耦合解释。

若 Oracle-$K$ 仅比 SINR Top-$K$ 高约 0.5% 或差异落入 Monte Carlo 不确定区间，则取消“关联是第二创新”的叙事。

## 4. 冻结的主场景

论文主场景固定为：

- $M=8$ 架 UAV，$Q=4$ 个目标；区域 $600\times600\,{\rm m}^2$；高度固定在 100--150 m；
- 2--3 个 active OTFS illuminators，其余 UAV 作为分布式接收机；
- 异质 RCS：$[0.5,0.2,0.1,0.05]\,{\rm m}^2$；
- 主实验固定发射功率、固定轨迹、固定 $N_{\rm looks}=16$，并报告对应驻留时间；
- 主检测口径为 $P_{FA}=10^{-3}$；若样本规模暂不支持，pilot 可用 $10^{-2}$，但必须明确标注；
- 主指标为 $P_D^{\rm weak}$ 和 $P_D^{\rm worst}$，平均 $P_D$ 仅作补充。

现有 $M=6,Q=3$、RCS 0.1、trial 32 数据保留为迁移诊断集，不再代表 V2 论文主场景，也不用于支持异质强弱目标结论。

## 5. 三个场景，而不是无边界 sweep

### Scene A：标准异质目标场景

$8$ UAV、$4$ 个异质 RCS 目标和中等直达干扰。它用于统一比较算法、呈现主表和系统级结果。

### Scene B：强干扰弱目标场景

固定几何、目标和噪声，仅扫描 direct-to-echo ratio（例如 20--50 dB）以及目标/干扰子空间重合度 $\xi$。它只回答 RQ1，不混入关联策略。

### Scene C：相关与冗余观测场景

构造若干几何方向相似的 UAV 和若干双基地角度互补的 UAV，使高 SINR 与高边际信息量发生分离。它只回答 RQ2，并承担 Oracle-$K$ 可研究性诊断。

## 6. 系统与接收模型

接收端在 OTFS 延迟--多普勒域建模为

$$
\mathbf Y_i^{DD}=\mathbf S_i^{\rm tar}+\mathbf I_i^{\rm dir}+\mathbf I_i^{\rm multi}+\mathbf N_i.
$$

目标和干扰都由 delay--Doppler--angle atom $\mathbf a(\tau,\nu,\theta)$ 构成。TP-UIC 输出

$$
\widetilde{\mathbf Y}_i=\mathbf Y_i-\widehat{\mathbf I}_i,
$$

并由实际残差估计 $\mathbf R_{{\rm res},i}$。从 V2 起，$\kappa$ 只能作为接收算法的测量结果，不得作为外部指定的系统能力参数。

TP-UIC 的核心约束写成

$$
\min_{\boldsymbol\beta}
\|\mathbf y-\mathbf A_I\boldsymbol\beta\|_2^2+
\lambda\|\mathbf P_T\mathbf A_I\boldsymbol\beta\|_2^2,
$$

或其等价约束形式。第一项拟合干扰，第二项惩罚对目标保护子空间的侵入。每个模块必须输出 $G_I$、$L_T$、残差协方差和检测统计量，保证贡献可消融。

## 7. 唯一算法链

$$
\text{OTFS observations}
\rightarrow \text{TP-UIC}
\rightarrow \mathbf R_{\rm res}
\rightarrow \text{local GLRT/LLR}
\rightarrow \text{UAV sensing association}
\rightarrow \text{correlation-aware fusion}.
$$

关联目标固定为

$$
\max_{\mathcal S}\min_q D_q(\mathcal S),\qquad
D_q(\mathcal S)=\boldsymbol\mu_{\mathcal S,q}^{H}
\mathbf R_{\mathcal S,q}^{-1}\boldsymbol\mu_{\mathcal S,q},
$$

约束只保留

$$
|\mathcal S|\le K,\qquad B(\mathcal S)\le B_{\max}.
$$

功率上限是固定物理条件，不作为本稿优化变量。若关联目标、通信成本或融合规则不能由这一条链解释，则移出主路径。

## 8. 公平基线分层

### 8.1 Receiver baselines：只验证 TP-UIC

所有臂使用相同场景、功率和观测：No-IC、LS/ZF-IC、SIC、TP-UIC、Oracle-IC。主图比较 $P_D^{\rm weak}$，机制图比较 $G_I$--$L_T$。

### 8.2 Association baselines：只验证观测选择

所有方法使用同一个已冻结的接收机：Random-$K$、Nearest-$K$、SINR Top-$K$、Geometry-$K$、忽略相关性的 Greedy Deflection、Proposed、Oracle-$K$。All-UAV 只作为无预算参考。

### 8.3 System-level summary：只保留五条

Single UAV、SINR Top-$K$、Proposed、All-UAV 和虚线 Oracle-$K$。任何主图不得再同时堆叠 receiver、association、power 和 fusion 的多层基线。

## 9. 四组正式实验

| 实验 | 自变量 | 主终点 | 证明对象 |
|---|---|---|---|
| E1 TP-UIC effectiveness | direct-to-echo ratio | $P_D^{\rm weak}$ / AUC | 接收端检测收益 |
| E2 target protection | 子空间重合度 $\xi$ | $G_I$ 与 $L_T$ | 抑制--保留机制 |
| E3 UAV association | $K=1,2,3,4$ | $P_D^{\rm worst}$ | 相关性感知选择价值 |
| E4 reporting trade-off | $B_{\rm report}$ | $P_D^{\rm worst}$ | 通信受限合作价值 |

每组实验先给协议，再给主结果，然后给对应消融与失败区间。不得用 Scene B 的接收机压力结果替 Scene C 的关联证据，也不得用 single-seed 结构上限写成 Monte Carlo 性能结论。

## 10. 暂时冻结与移出主路径

以下模块保留代码与历史结果，但在 RQ1、RQ2 通过准入门槛前不继续优化：

- trajectory / formation optimization；
- per-UAV power allocation 与 sensing--communication power split；
- dynamic looks、processing-gain rescue 和多帧 rescue；
- MARL；
- receiver-target 外层软乘子联合控制；
- 大规模 $15$ UAV、$10$ target stress test。

它们将来只能作为扩展或鲁棒性实验，不能参与 V2 核心算法的定义。

## 11. 当前证据的正确归档

| 已有结果 | V2 中的地位 | 允许的结论 |
|---|---|---|
| $M=6,Q=3$ 固定阵形与 400 m 真值辅助阵形 | 结构可行性诊断 | 几何是现有工作点的主要瓶颈 |
| TP-UIC receiver smoke / benchmark | 接口与压力轴诊断 | 机制接线正确；正式性能尚未成立 |
| 通信有限块长、Q/area scaling | E4 的前期 pilot | 可说明通信接入会限制融合，不可替代 V2 主场景复验 |
| 功率与 $\rho$ 扫描 | 冻结的扩展结果 | 功率是活动杠杆，但不是本稿核心贡献 |
| belief/certificate 多轮闭环 | 历史负结果 | 不再作为 V2 选择器主线 |

## 12. 执行顺序与停止规则

1. 实现并冻结 $M=8,Q=4$ 异质 RCS 的 Scene A/B/C 配置，先不新增优化变量。
2. 完成 E1/E2 的 receiver-only 五臂比较；若 TP-UIC 没有可复现的保护--抑制优势，则降级为接收预处理，不宣称贡献 1。
3. 对 Scene C 使用有预算的随机候选、窄束搜索和单交换做 SINR--检测相关性诊断；仅在极小规模单元测试中穷举以核验算法，不在正式实验中依赖枚举。若没有非微小改进空间，则停止开发 association 创新。
4. 只有前两道门均通过，才实现信息感知 greedy 方法并运行 E3。
5. E1--E3 成立后再接入有限块长报告模型完成 E4。
6. 最后才做 power、looks、area 和大规模压力扩展。

## 13. 论文结构

1. Introduction：从弱目标被强直达/强回波掩盖切入，明确“接收残差如何影响合作选择”的缺口。
2. System model：固定场景、OTFS receiver model、检测口径与通信预算。
3. Target-preserving interference cancellation：动机、机制、$G_I/L_T$ 证据钩子。
4. Residual-aware UAV sensing association：效用、约束、算法与复杂度。
5. Experiments：E1--E4 按证据链展开。
6. Discussion：报告无创新空间、低冲突饱和和 Oracle gap 过小等失败边界。

## 14. 假设与缺失证据

- 尚未生成 $M=8,Q=4$ 异质 RCS 主场景数据；本文档中的主场景是冻结设计，不是已有性能结论。
- 参考文献中的 2024--2026 工作需要在正式写作前逐篇核验题名、版本、实验设定和可支持的具体表述。
- $P_{FA}=10^{-3}$ 需要足够的 H0 样本或可靠的参数化标定；现有 n=16 smoke 不支持该门限下的 $P_D$ 结论。
- 当前 AUC 诊断提示直接干扰压力轴有效，但尚不能证明 TP-UIC 优于常规 IC。

## 15. 非枚举关联 pilot（2026-09-22）

固定 TP-UIC 后，在 $M=6,K=3$、强干扰条件下试验了 SINR Top-$K$、独立信息 Top-$K$、宽度 1 的相关性束搜索、单交换和少量随机候选。完整 $K$ 子集的实际访问比例为 40%--70%，未作全子集遍历。独立测试上，SINR Top-$K$ 的 mean AUC 为 0.509、worst AUC 为 0.444；相关性束搜索分别为 0.472 和 0.389，未显示优势。每个目标仅有 6 个校准和 6 个测试样本，结论仅用于筛查。详见 `NONENUM_ASSOCIATION_PILOT.md`。

## 16. 全局虚警校准候选方向（2026-09-22）

受控相关 GLRT 代理模型显示：全局经验校准在同分布时可将目标 $P_{FA}=0.05$ 控制到 0.048，但测试期 $+20\%$ 残差尺度漂移会使其升至 0.079；使用准确已知的失配上界可恢复到 0.048，同时将 $P_D$ 从 0.186 降到 0.131。2/3 bit 上报还引入明显的阈值粒度限制。这一结果支持继续验证“有限样本、失配和有限通信下的全局虚警控制”是否构成缺口，但尚未使用真实 TP-UIC 残差，不能作为系统贡献结论。详见 `GLOBAL_PFA_CALIBRATION_PILOT.md`。
