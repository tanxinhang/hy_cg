# TP-UIC 强干扰可行性筛查结论

日期：2026-09-22  
有效数据：`data/tpuic_receiver_feasibility_screen6/`  
状态：筛查结论，不是正式论文性能结果。

## 1. 裁决

当前 TP-UIC 实现**不能通过**“强干扰下抑制干扰、保留目标，并最终提高弱目标检测”的完整可行性门槛。

更准确地说：

- 干扰抑制可行：TP-UIC full 的中位结构对消深度约为 12.97 dB；
- 目标保留可行：中位目标存活率约为 1.000；
- 检测收益未实现：在 +30 dB 直达增益压力下，TP-UIC full 相对 plain LS 的 18 个成对单元全部 AUC 持平，平均和中位 $\Delta$AUC 均为 0；
- 问题仍有理论空间：perfect-channel 相对 plain LS 在 +30 dB 下的平均 $\Delta$AUC 为 +0.10、中位为 +0.14，并在 18 个单元中 12 胜、2 平、4 负。

因此，当前正确结论不是“TP-UIC 方向不可行”，而是：**强干扰下存在可恢复的检测信息，但现有 TP-UIC 的残差处理没有把约 13 dB 对消深度转化为检测统计量增益。**

## 2. 协议

- 固定 receiver-only 协议，不运行阵形、功率、关联和融合优化；
- $M=6,Q=3$，800 m 区域，RCS 0.05 $\mathrm{m}^2$；
- 全部 6 个接收机与 3 个目标，共 18 个成对单元；
- 直达增益档位为 0 和 +30 dB；实际 direct INR 中位数分别为 0.65 和 30.65 dB；
- 每个单元包含 6 个场景、每场景 2 个实现；合计 2,160 条记录；
- 比较 No-IC、plain LS、protected LS、TP-UIC full 和 perfect-channel；
- AUC 为主终点；由于每键标定/测试样本仍少，经验 $P_D/P_{FA}$ 只作诊断。

## 3. 核心结果

| 直达增益 | 方法 | 中位 AUC | 中位对消深度 | 中位目标存活率 |
|---:|---|---:|---:|---:|
| 0 dB | No-IC | 0.569 | 0.00 dB | 1.000 |
| 0 dB | plain LS | 0.569 | 12.97 dB | 0.999 |
| 0 dB | protected LS | 0.583 | 9.84 dB | 1.000 |
| 0 dB | TP-UIC full | 0.569 | 12.97 dB | 1.000 |
| 0 dB | perfect-channel | 0.639 | 不适用 | 1.000 |
| +30 dB | No-IC | 0.500 | 0.00 dB | 1.000 |
| +30 dB | plain LS | 0.500 | 12.97 dB | 0.999 |
| +30 dB | protected LS | 0.514 | 9.84 dB | 1.000 |
| +30 dB | TP-UIC full | 0.500 | 12.97 dB | 1.000 |
| +30 dB | perfect-channel | 0.639 | 不适用 | 1.000 |

强干扰把 No-IC / plain LS / TP-UIC full 的中位 AUC 从 0.569 压到 0.500，而 perfect-channel 保持 0.639。这证明压力轴有效，也证明性能缺口来自未被当前接收链兑现的残余干扰，而不是“此场景本来就没有信号”。

## 4. 成对差异

以相同 `(boost, receiver, target)` 下的 plain LS 为基准：

| 直达增益 | 方法 | 平均 $\Delta$AUC | 中位 $\Delta$AUC | 胜/平/负 |
|---:|---|---:|---:|---:|
| 0 dB | protected LS | -0.04 | -0.03 | 7/1/10 |
| 0 dB | TP-UIC full | 0.00 | 0.00 | 0/17/1 |
| 0 dB | perfect-channel | +0.02 | +0.03 | 9/3/6 |
| +30 dB | protected LS | -0.01 | 0.00 | 4/8/6 |
| +30 dB | TP-UIC full | 0.00 | 0.00 | 0/18/0 |
| +30 dB | perfect-channel | +0.10 | +0.14 | 12/2/4 |

最强冲突单元为 `(receiver=0,target=1)`，$\xi=0.360$。在 +30 dB 下，plain LS、protected LS 和 TP-UIC full 的 AUC 均为 0.583，而 perfect-channel 为 0.833。TP-UIC full 的目标存活率为 1.003，但检测差距仍为 0.250。这进一步说明“保护住目标能量”不是充分条件；残差协方差、残差方向或 GLRT 接口仍未匹配。

## 5. 原因判断

当前数据支持以下诊断：

1. `plain_ls == TP-UIC full == No-IC` 的检测表现不是缺少干扰压力，而是接收链对三者产生了等价或近等价的统计量；
2. $\kappa\approx13$ dB 只证明残差能量下降，不能证明与目标检测相关的残差方向被消除；
3. $\eta\approx1$ 只证明目标投影能量总体保存，不能保证目标条件 GLRT 的非中心参数提高；
4. protected LS 牺牲约 3 dB 对消深度换取更高目标存活率，但没有稳定 AUC 优势，说明当前保护目标与检测目标仍不一致；
5. perfect-channel 的明显间隔表明值得继续改 receiver model，而不是扩大 Monte Carlo 验证现有 TP-UIC。

## 6. 下一步门控实验

在扩大样本前，只推进一个机制修复：把 TP-UIC 的设计目标从“残差总能量 + 目标投影保护”改为“目标条件白化 GLRT 非中心参数或 deflection 的直接增益”。

建议按以下顺序执行：

1. 在高冲突单元固定 $\xi\ge0.3$，导出每个算法的残差在目标子空间、其正交补和 GLRT 白化空间中的能量；
2. 对比 TP-UIC 与 perfect-channel 的 $\mathbf R_{\rm res}$ 特征值和目标方向上的 $\boldsymbol\mu^H\mathbf R^{-1}\boldsymbol\mu$；
3. 设计 detection-aware TP-UIC 候选，只接受能提高 held-out deflection/AUC 的消除更新；
4. 用同一 18 单元筛查协议复验；只有 +30 dB 下成对 AUC 出现一致正增益，才运行每键 100+ 样本的正式轮。

停止规则：若 detection-aware 候选在高冲突构造场景中仍无法缩小相对 perfect-channel 的 AUC 差距，则 TP-UIC 降级为工程预处理，不再作为论文核心创新。

## 7. Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| 强干扰压力轴有效 | 实际 INR 中位数从 0.65 增至 30.65 dB；非 oracle AUC 降至约 0.5 | supported |
| 当前 TP-UIC 可抑制干扰并保留目标投影 | 中位 $\kappa=12.97$ dB，$\eta\approx1$ | supported |
| 当前 TP-UIC 可提高弱目标检测 | +30 dB 下 18/18 单元与 plain LS AUC 持平 | rejected in current implementation |
| receiver 仍有改进空间 | perfect-channel 在 +30 dB 下中位 AUC 0.639，成对中位增益 +0.14 | supported as screening evidence |
| detection-aware TP-UIC 会解决问题 | 尚未实现和验证 | needs evidence |

## 8. 继续诊断：直达 DD 失配协方差修复

进一步检查发现，旧 `C_res` 只传播了 `span(X)` 内的直达系数不确定性，没有传播
delay/Doppler 参数估计误差产生的字典外残差。在高冲突探针中：

- plain LS 的协方差迹仅为噪声迹的约 1.0005 倍；
- 其实际白化残差功率约为 452,110，而观测维数只有 16,384；
- perfect-channel 的白化残差功率约为 16,278，说明 Oracle 路径校准正常。

现已围绕接收机保存的直达路径估计，对 delay/Doppler 不确定性做一阶传播，并将
经过消除映射后的失配方向加入 `C_res`。修复后同一探针中 plain LS 的白化残差功率
降至约 23,614，协方差标定显著改善；perfect-channel 保持约 16,278，不承担不存在
的直达失配。

在完全相同的 2,160 条配对记录上重新计算 AUC 后：

| +30 dB 方法 | 修复前中位 AUC | 修复后中位 AUC | 相对 plain LS 胜/平/负 |
|---|---:|---:|---:|
| No-IC | 0.500 | 0.528 | 0/18/0 |
| plain LS | 0.500 | 0.528 | 基准 |
| protected LS | 0.514 | 0.500 | 5/4/9 |
| TP-UIC full | 0.500 | 0.542 | 2/16/0 |
| perfect-channel | 0.639 | 0.639 | 14/0/4 |

TP-UIC full 在强干扰下由“0 胜、18 平”变为“2 胜、16 平、0 负”，但成对中位
$\Delta$AUC 仍为 0，平均增益仅处于筛查分辨率量级。因此协方差漏项是一个真实
实现错误，修复了标定，却仍不足以让当前 TP-UIC 通过创新门槛。

更新后的开发判断：下一步不再调整硬保护权重；应直接比较候选消除算子的
$\boldsymbol\mu^H\mathbf R_{\rm res}^{-1}\boldsymbol\mu$，并以 held-out detection
information 作为接受条件。若候选仍与 plain LS 大面积相同，则 TP-UIC 的差异化
机制不存在，应按停止规则降级。

## 9. Detection-aware 候选试验

已实现实验臂 `detection_aware_tpuic`。它在 plain LS、protected LS 和 TP-UIC full
之间选择 belief-side `ncp_unit` 最大的算子；不读取检测统计量、H1/H0 标签或真值
目标回波，并强制配对 H1/H0 选择相同算子。

在最高冲突单元 `(receiver=0,target=1, xi=0.360)` 的 6 场景、2 实现和两个干扰
档位中：

- 24/24 次选择 TP-UIC full；
- +30 dB 下 detection-aware、TP-UIC full、plain LS 和 No-IC 的 AUC 均为 0.611；
- protected LS 为 0.500，perfect-channel 为 0.833；
- 因而 detection-aware gate 相对 plain LS 的 AUC 增益为 0。

这触发了停止规则：当前候选集合中，模型内信息量最大化只会重现 TP-UIC full，
而 TP-UIC full 没有兑现为检测优势。继续扩大样本只会更精确地确认同一个零差异，
不会产生新的算法机制。

### 最终裁决

1. 保留直达 DD 失配协方差修复，因为它纠正了接收机白化模型；
2. 将现有 TP-UIC 降级为可选工程预处理，不作为论文核心创新；
3. 不再运行 TP-UIC 大样本正式轮；
4. 论文主线应转向 Oracle-gap 已证明存在、但需要新估计方法才能利用的“残差场估计”，
   或直接转向 UAV association 可研究性诊断；不得把现有 TP-UIC 改名后继续包装。
