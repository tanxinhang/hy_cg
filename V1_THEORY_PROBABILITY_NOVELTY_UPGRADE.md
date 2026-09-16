# V1 理论、概率计算与创新性定位：本轮优化

日期：2026-09-16。对应用户标注“条件性证明、检测概率近似、全面新颖性查新能否优化”。

## 结论

可以优化，本轮已完成三项具体改进：

1. 将单观测 deflection 单调性加强为固定观测分布下的最优 ROC 支配结论。
2. 实现直接混合分布尾概率及多观测卷积区间，减少矩匹配的概率预测误差。
3. 核对最接近的回传受限感知、目标移交工作，重写主稿的差异定位。

主稿保持五页；物理系统模型不增加新机制。新增计算器作为最终选中集合的核验器，尚未替换 C2F 每一步的评分，原 MC=1000 主结论不重新标记为新算法结果。

## 一、证明如何加强

令固定本地观测在 H_h 下服从 P_h，无信息替代分布为 G，且 G 与假设无关。成功率 χ 时：
Q_h,χ = χ P_h + (1−χ)G。

对于 χ₂≥χ₁>0，对 χ₂ 的输出以 χ₁/χ₂ 概率保留，否则重新从 G 抽样。输出恰好服从 Q_h,χ₁，且此变换不依赖 h。

因此，用更可靠观测可以模拟任意较不可靠观测上的检验，保持其虚警率和检测率；在固定虚警上限 u 下，最优检测概率满足：

β*(u;χ₂) ≥ β*(u;χ₁)。

χ₁=0 的边界同样成立，χ₂=0 时两者相同。独立多观测按坐标重复此构造即可。

这是 Blackwell 统计试验退化思想的具体应用，非新的通用定理。出处：[Blackwell, 1953, Equivalent Comparisons of Experiments](https://doi.org/10.1214/aoms/1177729032)，Annals of Mathematical Statistics 24(2):265–272。正文引用，补充材料给出自包含构造证明。

**加强了什么**：结论从一个代理指标推进到最优检测器的完整 ROC 上包络。

**不能删的条件**：

- 必须固定同一感知观测分布与每个观测的替代分布。
- 替代机制不能依赖未知假设。
- 多观测构造采用独立报告；当前实现不支持任意相关证据。
- 移动融合位置可能让其他链路变差，不能由局部支配推出最近邻全局最优。
- 最优 ROC 不是当前加权和 + CF 阈值的 ROC。

最近距离不是固定集合最少报告的反例仍有效；不能靠删去假设把错误命题变成“无条件证明”。已有固定集报告数下界与可行目的节点搜索证书仍保留。

## 二、检测概率计算如何改善

新增 [mixture_probability.py](D:/Desktop/conference/isac_sim/mixture_probability.py)。

单报告直接计算加权移位 Gamma 与高斯替代的混合尾概率；多报告采用 CDF 差分得到每格质量，再卷积。模型仍是原 Gamma / Bernoulli / Gaussian 混合，没有增加信道、RCS 或新检测器。

设共同格距 h，保留范围内每个输入的向下取整为 L_e。对 s 个输入，在保留事件 B 上有：

L=ΣL_e ≤ T=ΣX_e < L+sh。

因此：

Pr(L>t,B) ≤ Pr(T>t) ≤ Pr(L+sh>t,B)+Pr(Bᶜ)。

实现不将截断质量重新归一化；保留被丢弃质量。使用 10^-9 尾质量预算，逐次减半格距，直到区间宽度≤10^-3，或达到 262144 格限制。预算耗尽返回 converged=False，不用窄区间伪装计算成功。真正零擦除使用严格 T>t 的判决语义。

**机器精度边界**：上述夹逼对精确 CDF/卷积成立。当前实现用 SciPy 和 FFT，记录浮点残差并留余量，但不是形式化区间算术证明。因此字段 floating_point_certified 明确为 False。它也不覆盖 belief 和传播模型误差。

### 实测核验

复现命令：python tools/audit_probability_refinement.py。

| 检查 | 原矩预测 | 新计算 | 含义 |
|---|---:|---:|---|
| 原六组单报告波形样本：与采样 P_D 的最大差距 | 0.0446414 | 0.00183652 | 同一阈值和同一批既有样本，改善预测；不是提高实际 P_D |
| 单报告新计算与解析参考 | — | 最大 4.44e-16 | 两种实现的解析一致性 |
| 新十二组多报告：与 20 万次采样的最大差距 | 0.0606240 | 0.00270998 | 含 a=9/a=0，每种各六组 |
| 十二组多报告计算区间 | — | 全部收敛，最大宽度 0.000863421 | 计算离散误差范围，非 Monte Carlo CI |
| 十二组多报告运行时间 | — | 中位数 0.01778 s/次 | 不等于完整 selector 时间 |
| 八个真实配置场景的 80 个选中目标集 | — | 80/80 的 P_D 与 P_FA 区间均收敛 | belief 侧核验，非真实检测率实验证明 |
| 上述选中集核验时间 | — | 中位数 0.0002595 s/目标 | 多数集合很小，不可推广到所有规模 |

十二组新测试的 SINR 在 [0.08,0.65]、χ 在 [0.3,1]，每组包含 2/3/6 个观测，随机种子 [10217,r]。真实配置使用种子 [10218,r]、当前 V1 高斯替代语义。详细 CSV、配置和源码哈希在 [结果目录](D:/Desktop/conference/results_v1_probability_refinement)。

**推荐使用位置**：最终集合和难区分的候选比较。若用于融合修正的接受判据，应要求新配置概率下界不低于旧配置上界；重叠则加密或返回未决。不要把两个区间中点的微小差别当成“不降”的证书。每一次 coarse greedy 都调用该计算器的时间代价尚未验证，故默认调度没有替换。

## 三、创新性查新的实际结果

### 检索范围与可追溯性

本轮检索覆盖以下关键词族：

- cooperative sensing / fusion node selection / wireless backhaul / OTFS；
- target handover / distributed ISAC / belief propagation；
- target-specific fusion / cooperative detection；
- fusion center selection / coarse-to-fine sensor selection；
- 2026 cooperative sensing node selection；
- Blackwell comparison / garbling / hypothesis-testing ROC。

学术 MCP 不可用；OpenAlex 备用脚本 HTTP 429。转用 Web 搜索发现候选并核对 arXiv 原文/摘要及原始出版标识。采用实际来源版本年份，未把搜索引擎“抓取日期”当成论文发表日期。没有 Scopus/WoS 全库覆盖，也未完成前后向全量引文追踪，因此称“针对性查新”，不称“全面查新完成”。

### 最接近工作对照

| 已核对工作 | 已有贡献 | 对本稿的约束与可保留差异 |
|---|---|---|
| [Chen et al., Joint Node Selection and Resource Allocation Optimization for Cooperative Sensing with a Shared Wireless Backhaul](https://arxiv.org/html/2405.16791v2), TSP 2025，DOI 10.1109/TSP.2024.3516709 | 回传 MAC 下定位，CRLB 约束、量化资源和 greedy 节点选择，MCSCA 收敛到松弛问题 KKT 集 | 不能把“通信受限节点选择”称新。本稿聚焦每目标融合目的节点导致的本地包消除、检测与 OTFS C2F 耦合；不宣称优于其定位优化 |
| [Li et al., Design and Optimization of Cooperative Sensing With Limited Backhaul Capacity](https://arxiv.org/abs/2404.03440), VTC2023-Fall，2024 上传，DOI 10.1109/VTC2023-Fall60731.2023.10333715 | 接收端向 FC 传局部估计和选定采样，KLT 量化和回传资源优化 | 回传与感知联合设计不是本稿首创；固定报告统计及 per-target receiver-local 分类是本稿较窄任务 |
| [Ge et al., Target Handover in Distributed Integrated Sensing and Communication](https://arxiv.org/html/2411.01871v1), 2024 预印本 | TPMBM 轨迹跟踪与 BS 间目标轨迹信息移交，处理共享什么、何时与何处共享 | 不能声称首次目标相关路由/交接。本稿为单次确认前的融合位置与局部统计报告选择，未实现其多时刻 tracker |
| [Bai, Ge & Wymeersch, Belief Propagation-based Target Handover in Distributed Integrated Sensing and Communication](https://arxiv.org/html/2506.23118v1), 2025 预印本 | belief 相关移交、因子图和消息传递，保持跟踪性能并减少 BS 数据交换 | “相近性能、更少通信”不是独占叙事。本稿需要依靠具体报告图、独立检测模型、C2F 和配对消融区分，不能只换成 UAV/OTFS 名称 |

Li 与 Chen 存在同类技术关联，但作者、题名及 DOI 不同，作为不同工作保留，不能合并成同一论文。Ge 与 Bai 的算法及年份不同，同样分别列出。

另发现 2026 年 [Distributed multisensor ISAC](https://www.nature.com/articles/s44459-026-00041-2)；原始页面访问受重定向限制，未获取全文，不用于断言其缺少本文机制。更宽泛的 fusion-center selection 检索也返回频谱感知工作，因尚未核对原始全文，不用它们填充“已完成查新”数量。

### 修改后的贡献表述

> 在先验辅助的双基地多 UAV OTFS 检测中，按目标配置融合目的节点，显式区分本地统计与远程报告，再执行 detector-aligned adaptive C2F；通过配对实验及融合规则消融验证通信效率，并给出报告退化的检测支配解释和固定集合报告数界。

不新增“全局最优 greedy”“精确感知联合优化”“首个目标级融合”或“新 Blackwell 定理”等主张。新概率计算是可信度工具，不另立第四个创新点。

## 四、论文和代码状态

- Introduction.tex 已补充 Ge/Bai 与回传选择的差异。
- ProposedMethod.tex 已加入最优 ROC 支配说明。
- References.tex 已补充三条对应参考文献。
- ReproducibilitySupplement.tex 第 5 节给出构造证明、离散夹逼、验证结果和检索边界。
- 系统物理模型及原 C2F 默认选择规则未更改，主稿仍为五页。
- 数值核验器有解析单报告、等尺度 Gamma 和、尺度变换、零统计、预算耗尽及非法输入检查。
- 全量测试与最终 PDF 视觉核验结论在本文件末尾追加。

## 五、仍然不能声称的内容

条件没有凭空消失，而是从“解释性代理指标”推进到了清楚限定的检测理论。数值概率误差已可核验，但 belief/物理抽象误差仍在。针对性查新已做并直接修改定位；全库全面查新和最接近算法在统一任务下的复现尚未完成。

## 六、最终验证

- 全量测试：154 passed，6 subtests passed。
- 主稿保持 5 页，补充材料为 8 页；两份 PDF 均连续编译两次。
- 最终 13 页均完成逐页视觉检查，无未解析引用、公式横向溢出、裁切或图文重叠；最终编译日志无 LaTeX/package warning 或 Overfull 提示。
- 原主比较未重跑，83.7% 等历史结果保持原证据口径；新增概率验证记录在 results_v1_probability_refinement。
