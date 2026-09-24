# 400 m dual-bottleneck receiver audit (2026-10-05)

## Decision

The 400 m regime has two separate bottlenecks and they must not share one
headline metric:

1. **cancellation robustness** is the held-out gap from TP-UIC to the
   perfect-channel receiver at the same detector and CPI count;
2. **detection capability** is the absolute perfect-channel AUC/PD and its
   improvement as independent looks are accumulated.

The new factorial gate fixes both axes explicitly: direct-interference boosts
of +10/+30/+50 dB, TP-UIC versus perfect-channel cancellation, and 1/2/4 CPI
noncoherent accumulation.  Every arm/count cell receives its own calibration
threshold and scenes remain the exchangeability unit.

## High-interference cancellation screen

The first candidate attacked the nonlinear source of the growing gap rather
than changing a cancellation weight.  It searched a local delay--Doppler grid
after projecting the observation and candidate direct dictionaries outside the
believed target-protection subspace, then rebuilt the normal TP-UIC dictionary.
The search uses no simulator-truth fields; a metamorphic test changes
`x_direct` and `h_true` and verifies an identical refined dictionary.

At 400 m / +50 dB, on a new seed and 24 held-out scenes:

| Receiver | Test AUC | Gap to perfect channel |
|---|---:|---:|
| TP-UIC | 0.550 | 0.064 |
| Single-look nonlinear-refined TP-UIC | 0.543 | 0.071 |
| Perfect channel | 0.615 | 0.000 |

The candidate changes the statistic materially but makes the ordering slightly
worse (AUC delta -0.007).  It is therefore **rejected as a single-look
receiver** and must not be wired into production.  The likely failure mode is
data reuse: a single noisy look selects a nuisance location and then tests the
same look, so the location search can follow noise or imperfectly protected
target structure.  Its roughly five-by-five grid cost is also unsuitable for
production.

An independent low-interference check at 400 m / +10 dB used master seed
`20261006` and another 24 held-out scenes:

| Receiver | Test AUC | Gap to perfect channel |
|---|---:|---:|
| TP-UIC | 0.630 | 0.066 |
| Single-look nonlinear-refined TP-UIC | 0.653 | 0.043 |
| Perfect channel | 0.696 | 0.000 |

The refined receiver gains +0.023 AUC at this point and closes 0.023 of the
oracle gap.  A same-scene paired bootstrap gives a 95% interval of
[-0.024, +0.085] for the AUC change, so the sign is not established.  Together
with the -0.007 point change at +50 dB, this is evidence of a regime-dependent
effect, not a promotable general robustness improvement.  The single-look
variant remains rejected; the low-INR result is retained as motivation for
cross-fitted multi-look DD refinement rather than discarded.

## Detection branch

The prior independent result remains the positive detector direction.  At
400 m / +10 dB, two independent TP-UIC looks increased test AUC from 0.674 to
0.739 and PD from 0.225 to 0.525, at approximately twice the observation and
receiver cost.  This improves evidence supply but does not by itself close a
high-INR cancellation gap.

## Next gate

The next cancellation candidate should share one DD state across multiple
looks while fitting an independent complex gain per look.  The DD state must
be estimated on training/reference looks or by cross-fitting, and evaluated on
a held-out look; this removes the single-look data-reuse failure.  Replace the
grid with coarse-to-fine or Gauss--Newton only after that statistical gate
passes.

### Initial hierarchical-MAP cross-fit screen

That next candidate was implemented as a two-fold screen.  CPI 1 estimates the
shared DD state and CPI 2 supplies a held-out detection statistic; the roles
are then reversed and the two held-out statistics are summed.  Each reference
look profiles an independent complex direct-path gain, while DD displacement
has a Gaussian prior.  The equal-cost baseline sums two ordinary TP-UIC looks.

On a new seed (`20261007`), with eight held-out scenes per interference level:

| Boost | Ordinary two-CPI TP-UIC | Cross-fitted hierarchical MAP | Perfect-channel two-CPI |
|---|---:|---:|---:|
| +10 dB | 0.984 | 1.000 | 0.953 |
| +50 dB | 0.828 | 0.969 | 0.969 |

The same-scene AUC changes are +0.016 at +10 dB (bootstrap interval
[0.000, 0.063]) and +0.141 at +50 dB ([0.000, 0.281]).  The +50 dB point closes
the observed oracle gap completely and reverses the single-look failure, so
the hierarchical/cross-fit structure passes the **directional screen**.
Eight scenes are far too few for promotion, perfect-channel need not rank first
in a finite sample, and two calibration scenes cannot support a finite 5% tail
threshold.  No PD/PFA claim is made.  The current two-way 5x5 coordinate grid
is also a correctness prototype, not a viable production solver.

Decision: expand the +30/+50 dB test with a real calibration partition; only
then replace the grid with shared-dictionary Gauss--Newton and measure runtime.

### Formal +30/+50 dB confirmation

The fixed candidate was rerun with master seed `20261008`, separately at each
interference level, using exactly 1 train, 40 calibration and 40 held-out test
scenes.  Each receiver received its own 5% split-conformal H0 threshold.

| Boost | Receiver | Threshold | AUC | PFA | PD | Oracle AUC gap | Oracle PD gap |
|---|---|---:|---:|---:|---:|---:|---:|
| +30 dB | ordinary two-CPI | 26.327 | 0.727 | 0.000 | 0.125 | 0.027 | 0.125 |
| +30 dB | cross-fitted MAP | 23.023 | 0.748 | 0.000 | 0.175 | 0.006 | 0.075 |
| +30 dB | perfect channel | 22.838 | 0.754 | 0.025 | 0.250 | 0.000 | 0.000 |
| +50 dB | ordinary two-CPI | 119.600 | 0.693 | 0.025 | 0.025 | 0.062 | 0.225 |
| +50 dB | cross-fitted MAP | 25.453 | 0.754 | 0.000 | 0.200 | 0.000 | 0.050 |
| +50 dB | perfect channel | 22.838 | 0.754 | 0.025 | 0.250 | 0.000 | 0.000 |

At +30 dB the MAP changes AUC by +0.021 (paired bootstrap 95% interval
[-0.001, +0.054]) and PD by +0.050 ([0.000, +0.125]).  This is a positive but
not decisive trend.  At +50 dB it changes AUC by +0.062 ([-0.021, +0.146]) and
PD by **+0.175** ([+0.075, +0.300]).  The high-interference PD improvement is
the first formally nonzero receiver gain in this sequence.  MAP PFA is 0/40 at
both levels (Wilson upper bound 0.088); this does not indicate inflation.

The +50 dB point estimate closes the AUC oracle gap and reduces the PD oracle
gap from 0.225 to 0.050.  Its oracle-minus-MAP intervals are [-0.050, +0.043]
for AUC and [0.000, +0.125] for PD.  Thus oracle parity in AUC is plausible but
not proven as an identity, while a small residual working-point PD gap remains.

Decision: **the hierarchical/cross-fit MAP passes the formal +50 dB PD gate and
is retained as the high-interference receiver candidate.**  The +30 dB result
does not independently pass a strict superiority gate.  Neither result makes
the 5x5 grid production-ready; the next implementation task is an equivalent
Gauss--Newton/LM solver followed by a numerical-equivalence and runtime gate.

### Bounded Gauss--Newton replacement gate

A trust-region reflective Gauss--Newton solver now profiles the same
look-specific complex gains, minimises the same target-protected MAP residual,
and enforces the same +/-2 sigma DD box.  Four function evaluations were fixed
from an independent three-scene runtime probe before formal confirmation.

The matched +50 dB microbenchmark gives a median pure-solver time of 0.246 s
for GN4 versus 0.551 s for the 5x5 grid, a **2.24x solver speedup**.  Every
probed H0/H1 GN4 objective was lower than the coarse-grid objective.  The gain
is not an end-to-end 2.24x speedup: TP-UIC, covariance construction and GLRT
remain common fixed costs.

On the same formal seed and 1/40/40 split used above, GN4 produces:

| Solver | AUC | PFA | PD | Threshold | Median cross-fit stage / scene |
|---|---:|---:|---:|---:|---:|
| 5x5 grid | 0.754 | 0.000 | 0.200 | 25.453 | not instrumented in frozen run |
| GN4 | 0.758 | 0.000 | 0.200 | 23.448 | 11.204 s |

The paired GN4-minus-grid AUC change is +0.0038 with a 95% interval of
[-0.0244, +0.0294]; the PD change is exactly 0 with interval [0, 0].  Final
statistics have Pearson correlation 0.988 and median relative difference
1.63%.  GN4 therefore passes numerical/detection non-degradation and pure MAP
runtime gates.  It replaces the grid as the experimental solver default.

The broader receiver is **not yet runtime-closed**: the formal GN4 cross-fit
stage still costs about 11.2 s per scene in this Python harness.  The next
runtime target is shared dictionary/Jacobian caching and analytic manifold
Jacobians; claiming a production-ready end-to-end acceleration now would be
incorrect.

The next detector candidate remains multi-look evidence accumulation, with
1/2/4-look latency curves and a larger calibration partition.  Promotion
requires both:

- a smaller TP-UIC-to-perfect AUC gap at +30 and +50 dB, without lower target
  survival or inflated held-out PFA;
- materially higher perfect-channel and TP-UIC PD at the calibrated operating
  point, reported together with observation/compute cost.

Machine evidence:

- `data/area400_boost50_refinement_screen_20261005/`
- `data/area400_boost10_refinement_screen_20261006/`
- `data/area400_crossfit_map_screen_20261007/`
- `data/area400_crossfit_map_boost30_formal_20261008/`
- `data/area400_crossfit_map_boost50_formal_20261008/`
- `data/area400_crossfit_map_formal_analysis_20261008/summary.json`
- `data/area400_crossfit_gn_nfev4_boost50_formal_20261008/`

### Receiver-target 泛化筛查（2026-10-10）

冻结 GN4 + TP-UIC 主体后，新增 400 m、4 receiver × 3 target ×
`{+10,+30,+50}` dB 的跨几何筛查。协议为 1 train / 2 calibration /
2 test，仅用于发现明显退化和估算成本；在 `P_FA=0.05` 下，2 个 calibration
样本的 split-conformal 阈值必然为 `inf`，因此本轮 PD/PFA 不可解释，也不得作为
正式通过证据。

宏平均 AUC 差值（GN-MAP TP-UIC 减普通两 CPI TP-UIC）为：+10 dB
`+0.0208`，+30 dB `-0.0208`，+50 dB `-0.0208`。+30/+50 dB 均有
10/12 个 receiver-target 单元 AUC 不退化，但两个负迁移单元说明单节点正式结果
尚不能外推为跨节点结论。裁决：`screen_only`，TP-UIC 主体继续冻结；正式门禁应
预注册跨 4 个 receiver 的代表/困难组合，并为每个单元使用至少 40 calibration、
40 test。扩样前先补增量落盘、断点恢复和受控并行，避免直接运行全笛卡尔 40/40。

产物：`data/tpuic_generalization_screen_20261010/summary.json`。

#### 原因审计协议修复

后续泛化实验不再把 `boost_index` 放入观测噪声或直达 DD 误差的随机种子。
同一 `(scene, receiver, target, look)` 在不同 boost 下复用同一随机 realization，
只改变直达径增益，从而把干扰强度变成严格配对的因果轴。

泛化门禁同时记录四层诊断，但不改变 GN4、保护预算、残差协方差或 GLRT 参数：

1. 几何/保护：被测目标是否受保护、保护秩、直达字典条件数；
2. GN：初末 MAP 代价、迭代数、最优性、边界命中，以及仅供离线审计的
   delay/Doppler 真值 RMSE；
3. TP-UIC：实测/预测残差功率、结构残差、目标存活率及风险下界；
4. 检测：raw/最终 GLRT、白化残差功率、`rho_weighted`、`xi_rel_q` 和阶段耗时。

GN 求解器的原 API 与返回值保持不变；新增诊断 API 只暴露接收机可见的优化器状态。
真值 RMSE 仅在 `tools/` 实验层计算，不参与任何估计、对消或检测决策。

#### 预注册机制审计（运行前冻结）

固定三个 receiver-target 单元，不根据新数据更换：改善候选 `(0,1)`、稳定候选
`(3,2)`、负迁移候选 `(1,1)`。每个单元都运行 `{+10,+30,+50}` dB，跨 boost
严格复用随机 realization；每点 1 train / 4 calibration / 8 test。分类仅来自旧的
非配对 2-test 筛查，所以新实验允许推翻分类。该实验只裁决失效层级，不以有限
calibration 的 PD/PFA 作正式结论，也不修改 GN4、保护预算、残差协方差或 GLRT。

运行结果（8 test/点）推翻旧筛查的负迁移分类：三个预注册单元在全部 boost
均不退化。宏 AUC 增益在 +10/+30/+50 dB 分别为 `+0.0313/+0.0417/+0.0833`；
旧负迁移候选 `(1,1)` 在 +50 dB 由 `0.5781` 提升到 `0.7500`。

分层诊断把主要问题定位到残差模型，而非 GN：GN 最终/初始代价比为
`0.01--0.56`，delay RMSE 降至初始的 `6%--9%`，且没有边界命中。保护也不是
当前首要瓶颈：`(3,2)` 仅 `12.5%` 场景保护被测目标、平均目标存活约 `0.76`，
但仍在三个 boost 全部提升。相反，真实结构残差/预测残差比从 +10 dB 的
`0.06--0.28`，扩大到 +30 dB 的 `5.97--27.92`，再扩大到 +50 dB 的
`597--2792`。下一步优先审计残差协方差的干扰强度缩放和 GN 后验不确定性传递，
不继续调整 GN 或保护预算。

产物：`data/tpuic_mechanism_audit_paired_20261010/`。

#### Held-out 残差证书筛查

直接把 sigma-point factor trace 写进 `i_res_pred` 的候选实现未通过功率缩放测试，
已撤回且未进入主线：估计器映射 `F(X)` 随直达字典幅度变化，不能把物理功率增长
和高 SNR 下更精确的投影混为一个 trace。

替代方案仅作为实验诊断：在 cross-fit 后的 held-out CPI 上，将 TP-UIC 残差投影到
全部信念目标子空间之外，再扣除名义噪声地板。`(receiver=1,target=1)` 的 4-test
严格配对筛查中，held-out 均值/真实结构残差均值在 +10/+30/+50 dB 分别约为
`29.7/1.05/0.82`；旧 `i_res_pred` 在 +50 dB 仅为真实残差的约 `0.0017`。
因此新量恢复了高干扰功率缩放，但低干扰下受噪声自由度和目标字典失配污染而过于
保守，暂不替换生产证书。下一步应改进实际对消算子下的噪声 trace 核算，并保持
held-out 独立性。

产物：`data/tpuic_heldout_certificate_screen_20261010/`。

进一步把噪声扣除改为实际低秩 TP-UIC 算子下的精确
`sigma^2 ||(I-P_A)(I-F)||_F^2`。同种子重复实验中，+10/+30/+50 dB 的证书/
真实残差均值比由 `29.7/1.05/0.82` 变为 `37.8/1.13/0.82`。精确算子会消除
部分噪声，因而正确的噪声 floor 更低；低干扰偏高反而扩大。这否定了“噪声自由度
扣错是主因”。此前据此把污染定位到目标失配/未建模回波属于缺少分量证据的推断，
不能作为优化依据；后续真值分解已对此作出更正。

产物：`data/tpuic_heldout_operator_noise_screen_20261010/`。

#### 固定算子残差真值分解

为避免继续猜测，在完全相同的 `(receiver=1,target=1)`、1 train / 1 calibration /
4 test 严格配对协议下，冻结每个 held-out CPI 已拟合的 TP-UIC 线性算子，并把
`(I-P_A)(I-F)y` 精确分解为直达径、目标、噪声及相干交叉项。分解只使用于
`tools/` 离线审计的真值，不进入接收机。所有 test 行的最大相对闭合误差为
`2.47e-12`，分量归因在数值上成立。

H1 test 均值显示：+10 dB 时噪声为 `6.252e-10`，占总残差 `99.93%`，直达径仅
`2.80e-14`；+30 dB 时噪声为 `6.252e-10`，占 `99.52%`，直达径为
`2.79e-12`；+50 dB 时直达径增至 `2.79e-10`（总量约 `30.9%`），噪声仍为
`6.252e-10`（约 `69.3%`）。目标项跨 boost 基本不变，约 `6.03e-13`，不足以
解释低干扰证书偏高；H0 也呈现相同的直达径与噪声尺度。

因此因果裁决是双区间的：低/中干扰下，held-out 能量减“期望噪声底”再截零会把
有限样本噪声能量波动变成高方差、正偏的结构残差证书；高干扰下才出现随直达功率
约 100 倍/20 dB 缩放的真实直达径残差。下一步不应扣除臆测的目标失配，而应分别
研究低干扰的噪声不确定性校准（分布/置信上界而非单点截零）和高干扰的直达径残差
协方差；两者必须用分区门禁验证，不能用一个标量修补同时掩盖。

产物：`data/tpuic_residual_component_audit_20261010/`。

#### 固定算子条件噪声能量分布

在不修改证书的前提下，对上述 4 个 test 场景、两个 cross-fit held-out 算子和
`{+10,+30,+50}` dB 进行条件分布审计。对每个已拟合算子精确求取
`L L^H`（`L=(I-P_A)(I-F)`）的压缩谱；复高斯噪声能量因此是单位 Gamma 项与
少量加权指数项之和。每个算子再进行 100,000 次谱域抽样，只用于验证尾部和截零
偏差，不重新拟合接收机。

三档 boost 的结果实质相同：均值约 `6.255e-10`，标准差约 `4.895e-12`，变异
系数 `0.007826`，矩匹配有效 shape 约 `16328.5`，偏度仅 `0.01565`。4 个场景、
两个 held-out 算子的 shape 只在 `16326.2--16331.6` 间变化；条件算子随机性对
该分布很弱。95% 和 99% 分位分别约为均值的 `1.0129` 和 `1.0183`，符合高自由度
Gamma/近正态尾部，而非重尾异常。

关键问题是当前变换本身：纯噪声能量高于其均值的概率约 `0.499`，所以
`max(Q-E[Q],0)` 在无结构残差时仍有约一半概率输出正值；其平均伪残差约
`1.95e-12`（均值的 `0.312%`）。这比 +10 dB 的真实直达径分量
`2.80e-14` 大约 70 倍，足以解释低干扰证书的数量级高估。该偏差不是噪声模型
重尾、boost 缩放错误或目标泄漏造成，而是“减期望值并截零”对有限自由度噪声的
必然整流偏差。

因此后续若优化，应优先比较带显式显著性水平的噪声超额量（例如减去条件分位数）
或报告结构残差的区间/上界，而不是继续修正均值。正式改动前还需用独立端到端
H0 样本校验所选分位的覆盖率；本轮仅完成分布诊断，不改变 TP-UIC 主线。

产物：`data/tpuic_noise_distribution_audit_20261011/`。

#### Certificate 职责隔离与 reference coefficient MAP 筛查

连续链路计算原先使用名为 `TrialCertificate` 的上下文广播 `(residual fraction,
retention)`。该对象实际承担 nominal receiver state，而不是风险保证。现将正式名称
改为 `TrialReceiverState` / `trial_receiver_state`；旧名称仅作为数值兼容别名保留。
风险证书没有自动注入该上下文的入口，因此后续 nominal 与 certificate 可分别演进，
默认 `production_wire=False` 和全部数值保持不变。

随后实现真正的 reference-only protected MAP：从 1/2/4 个独立 reference CPI
联合估计一组共享复直达系数，再应用到未参与拟合的 sensing CPI。首轮只测直达径
真值残差和后验收缩，不改变 detector。协议复用 `(receiver=1,target=1)` 的 4 个
test 场景及 `{+10,+30,+50}` dB 严格配对轴。

结果否决“跨 CPI 静态共享复系数”假设。+10 dB 下 1/2/4-reference 的真实直达残差
分别为 `2.17e-8/1.39e-8/1.52e-8`，而相同 DD 估计下的 sensing-CPI self-fit
结构残差约为 `5.51e-12`，reference MAP 差约 2500--3900 倍；+30/+50 dB 以相同
比例随直达功率缩放。把 reference 的观测替换为 oracle `x_direct` 后结果几乎不变，
排除了 reference 噪声和目标污染作为主因。

后验 trace 和传播后的 coefficient-error trace 确实近似按 `1/L` 收缩，但这只是
错误静态模型内部的不确定性收缩。reference 与 sensing 的字典相对变化严格为 0，
而 oracle 系数相对误差仍为 `1.16--1.41`、相干度仅 `0.30--0.43`。生成器为每个
CPI 独立抽取直达复相位，因此不能把 `n_cpi` 的 `1/L` covariance 缩放解释成可实现
的 coherent coefficient averaging。

裁决：保留 reference MAP 模块作为实验反例，不接入默认 TP-UIC 或 detector。
下一步若继续多 CPI，应先定义并验证跨 CPI 相位/信道演化模型；在当前 iid-phase
场景中，只能用 reference 学习 DD、协方差或超参数，不能直接迁移瞬时复系数。

产物：`data/reference_coefficient_map_screen_20261011/`。

#### 同 CPI、资源隔离的 pilot coefficient MAP

在不改变 iid 跨 CPI 相位模型的前提下，将同一 CPI 的观测行按固定随机置换严格
拆成 pilot 与 sensing 两部分。pilot 占比取 `12.5%/25%/50%`，只在 pilot 行上
做目标保护 MAP；得到的瞬时复系数应用到完全未参与拟合的 sensing 行。检测协方差
使用 reference-fit 语义：sensing 热噪声为 `sigma^2 I`，另加
`X_s C_h X_s^H`，不使用 residual certificate。

对消层面方案在中高干扰有效。+30 dB 时 pilot 的直达残差比例为
`9.26e-5/6.21e-5/1.82e-5`，对应 sensing self-fit 约
`1.08e-3/1.17e-3/1.56e-3`；+50 dB 结果为
`1.06e-4/1.50e-5/1.05e-5`。但 +10 dB 的 12.5%/25% pilot 受估计噪声影响，
分别为 `1.02e-2/3.58e-3`，差于 self-fit，50% pilot 才降至 `6.45e-4`。
资源代价也清晰：三个占比留下的目标能量约为 `83.0%/68.5%/44.5%`。

检测端采用与现有 GN-MAP 相同的 3x3 局部 DD 最大统计量。4-test 筛查的 nominal
AUC 在 12.5%/25%/50% pilot 下为 `0.3125/0.3125/0.25`（+10 dB）、
`0.3125/0.3125/0.25`（+30 dB）和 `0.3125/0.25/0.4375`（+50 dB），没有形成
随干扰增强的净检测收益。+50 dB nominal H0/H1 统计量同时膨胀，说明仅传播系数
后验漏掉了结构/DD 残差方向；加入 simulator-truth 直达残差 rank-one covariance
后，统计量恢复到与低干扰同一尺度，但 oracle AUC 仍仅 `0.3125--0.4375`。

因此首版 pilot MAP 的裁决是“对消通过、检测不通过”：它成功隔离了瞬时系数估计，
却未改善弱目标可分性，且以目标观测资源为代价。oracle covariance 也不能恢复 AUC，
说明继续单独优化 pilot covariance 不足以解决当前检测瓶颈。该路径保持实验状态，
不进入默认 TP-UIC；若后续重启，应先提高完整观测下的 perfect/nominal detector
能力，再比较结构化 pilot 设计，而不是继续扩大 pilot 比例。

产物：`data/pilot_coefficient_map_screen_20261012/`。

#### 完整观测 detector headroom 与两 block 累积

为判断 pilot 失败是否源于 detector 本身，在不扣除任何观测资源的条件下，对困难
单元 `(receiver=1,target=1)` 运行新的严格配对筛查。每个 block 独立执行完整的
`GN-DD -> TP-UIC -> sigma-point covariance -> 3x3 neighbourhood GLRT`；增强候选
仅把两个 block 的最终统计量非相干相加。perfect-channel 使用完全相同的观测和
detector。协议为 1 train / 2 calibration / 8 test，仅裁决 AUC 信息上限，不解释
有限 calibration 下的 PD/PFA。

结果表明无需先修 detector 才能获得高 AUC。perfect one-block 在三个 boost 上均为
`0.9375`；nominal one-block 在 +10/+30/+50 dB 分别为
`0.9063/0.9219/0.8906`，oracle gap 仅约 `0.016--0.047`。这与旧 4-test 小样本的
低 AUC 相反，说明后者不能代表该几何的稳定信息上限。

两 block 简单求和没有增强：perfect 降至 `0.9063`，nominal 为
`0.9063/0.8750/0.8594`。第二 block 单独仍有约 `0.89` perfect AUC 和
`0.84--0.88` nominal AUC，但其 H0/H1 尺度与第一 block 有有限样本差异，直接
相加会扰动排序。当前 one-block 已接近 perfect 上限，因此不再为该单元增加 CPI
或融合复杂度。若系统级仍需要多 block，应在独立 train/calibration 上冻结尺度
校准或似然融合，并使用至少 40 test；不能在这 8 个 test 上继续选融合权重。

裁决：完整观测 detector 在本次独立种子筛查中达到预期；首版 two-block sum 不
推广。下一主瓶颈回到检测稳定性/跨场景泛化和完整系统 PD，而不是该单元的一 block
AUC 或 coefficient cancellation depth。

产物：`data/full_observation_detector_screen_20261013/`。

#### 完整观测 detector 正式 +50 dB 门禁

冻结上述 one-block proposed，在 `(receiver=1,target=1)`、+50 dB 上运行
1 train / 40 calibration / 40 test 的独立正式门禁（master seed `20261014`）。
nominal GN-MAP TP-UIC 的 AUC 为 `0.7681`，split-conformal 阈值 `11.4388`，
test PFA `0.000`、PD `0.375`；perfect-channel AUC 为 `0.7431`，阈值
`10.6098`，PFA `0.050`、PD `0.450`。nominal-minus-perfect AUC 为 `+0.025`，
在 40-test 规模下不解释为超越 oracle；两者使用不同残差与协方差，有限样本排序
允许反转。

该结果推翻 8-test 下 `0.89--0.94` AUC 已经稳定的乐观判断。完整观测仍提供中等
偏强排序能力，但 nominal 的 H0 阈值更高且本批次零虚警，显示其 operating point
过于保守并损失 PD。下一 detector 优化应针对 nominal H0 尾部校准和阈值稳定性，
而不是继续增加 TP-UIC stage、pilot 比例或直接叠加 block。+10/+30 dB 尚未完成
同规模正式复核，不能由本结果外推。

产物：`data/full_observation_detector_formal_boost50_20261014/`。

#### +50 dB operating-point 原因审计

在不改变接收机、统计量或门限的前提下，复算 40 calibration / 40 test 的完整
样本级分布。5% split-conformal 门限是 calibration H0 的第 39/40 个次序统计量，
即第二大值。nominal 门限为 `11.4388`，但 test H0 最大值为 `11.1521`，只低
`0.2866`；若真实超限率恰为 5%，40 次测试观察到零超限的概率仍为
`0.95^40=0.1285`。因此 `PFA=0` 本身不是失配证据。

calibration 与 test H0 的主体分布也未显示可见漂移：nominal 的两样本 KS 统计量
为 `0.10`（诊断 p 值 `0.990`），perfect 为 `0.15`（`0.766`）。nominal 与
perfect 在 test 上的 Spearman 相关为 H0 `0.775`、H1 `0.863`，表明主要场景
难度排序相同。对 calibration 场景重采样后，nominal 门限 95% 诊断区间为
`[8.8522,12.1631]`，对应固定 test 的 PD 区间为 `[0.350,0.525]`；40 个尾部
校准样本不足以稳定地区分当前 `0.375` 与 `0.45`。

最直接的反事实是交换两个接收机独立得到的门限：nominal 使用 perfect 门限
`10.6098` 时得到 `PFA=0.05, PD=0.45`；perfect 使用 nominal 门限时得到
`PFA=0, PD=0.45`。所以本次 nominal 相对 perfect 的 `0.075` PD 差额由两个
独立的有限样本尾部门限实现解释，并非 TP-UIC 分数整体退化。AUC 不依赖门限，
nominal `0.7681` 的场景 bootstrap 95% 诊断区间为 `[0.6625,0.8625]`，与
perfect 的 `[0.6419,0.8363]` 大量重叠，也不支持二者存在稳定优劣。

裁决：将上一节“nominal operating point 过于保守”收窄为“该次 nominal 门限
实现偏高，但尚无系统性过保守证据”。当前首先要解决的是尾部样本量导致的门限
不确定性，而不是修改 TP-UIC 或 GLRT。下一步应保持算法冻结，以相同场景 ID 完成
+10/+30 dB 配对复核，并扩大 calibration H0 或重复独立 calibration split；只有
反复出现 nominal H0 尾部膨胀后，才审计 covariance/最大统计量的结构性失配。

机器可读原因审计：
`data/full_observation_detector_formal_boost50_20261014/cause_audit.json`。

#### +10/+30 dB 配对复核与独立 calibration repeat

冻结 one-block GN-MAP、TP-UIC、sigma-point covariance 和 3x3 GLRT，使用与
+50 dB 完全相同的 master seed、scene ID 和 split，补齐 +10/+30 dB 的
1 train / 40 calibration / 40 test 正式配对门禁。+10 dB nominal 为
`AUC=0.7425, PFA=0.025, PD=0.375`，+30 dB nominal 为
`AUC=0.7688, PFA=0.025, PD=0.350`；对应门限为 `11.1858/11.4774`。
perfect-channel 在三档 boost 下逐位保持同一结果：`AUC=0.7431, PFA=0.05,
PD=0.45`、门限 `10.6098`，验证了 boost 配对实现。

为判断 nominal 门限偏高是否稳定复现，另生成完全独立的 scene 81--120，只使用
H0 并让三档 boost 共用几何。原批次中 nominal 门限在三档均高于 perfect；但新增
批次发生反转：perfect 门限为 `13.4696`，nominal 的 +10/+30/+50 dB 门限分别为
`12.9983/11.8023/11.8152`。因此 nominal H0 尾部膨胀没有跨 calibration 批次
复现，也没有随干扰强度单调增大。

合并两批得到 80 个 calibration H0 后，nominal 的 +10/+30/+50 dB 门限为
`11.6978/11.4774/11.4582`，固定 test 上分别得到 PFA
`0.025/0.025/0.000`、PD `0.350/0.350/0.375`；perfect 合并门限为
`10.8540`，PFA `0.05`、PD `0.45`。这些 operating-point 差异仍存在，但独立
批次反转表明不能归因于 nominal covariance 的稳定结构性尾部失配。

裁决：当前不修改残差协方差、最大统计量或 TP-UIC。三档 AUC 没有显示随 boost
恶化，主要未决问题是 5% 极端次序统计量在 40--80 calibration 下仍不稳定。
若需要精确 PD operating point，应继续增加只含 H0 的 calibration 或报告门限
不确定性；不能把单批次门限差异包装成抗干扰算法缺陷。

产物：`data/full_observation_detector_formal_boost10_20261014/`、
`data/full_observation_detector_formal_boost30_20261014/` 和
`data/full_observation_detector_calibration_repeat_20261014/`。

#### 纯 H0 粗校准与精校准

采用两阶段但保持同分布的 H0-only 协议。粗阶段使用原 40 calibration 加第一次
独立 repeat 40，共 80 个 H0，用于确认批次方向会反转并冻结后续协议；精阶段新增
scene 121--240 的 120 个独立 H0，最终以全部 200 个样本计算 5% split-conformal
门限。没有根据粗阶段分数挑选精阶段场景，也没有读取新 H1，因此不产生 tail
selection bias 或 test 泄漏。

| boost | 80-H0 门限 | 80-H0 bootstrap 95% | 200-H0 门限 | 200-H0 bootstrap 95% | test PFA | test PD | PD 区间 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| +10 dB | 11.6978 | [9.6262, 12.9983] | 11.1858 | [10.4818, 12.1241] | 0.025 | 0.375 | [0.350, 0.400] |
| +30 dB | 11.4774 | [10.1024, 11.8023] | 11.5202 | [10.5690, 12.0185] | 0.025 | 0.350 | [0.350, 0.425] |
| +50 dB | 11.4582 | [9.8131, 12.1631] | 11.5009 | [10.9025, 12.1631] | 0.000 | 0.375 | [0.350, 0.450] |

精阶段使 +10 dB 门限区间明显收窄；+30/+50 dB 的区间仍宽但彼此高度重叠，且
200-H0 点估计仅差 `0.0193`，不支持高干扰导致 H0 尾部继续膨胀。固定 40-test
上，门限不确定性传播后的 PD 区间仍跨越 2--4 个样本台阶；因此 calibration
扩容解决了“单一第二大值定门限”的问题，却不能消除 test 仅有 40 场景带来的
`0.025` 分辨率限制。

裁决：200-H0 可作为当前 operating-point 校准版本；继续增加 calibration 的
边际价值已经低于扩大独立 test。TP-UIC、covariance 与 GLRT 继续冻结。若下一步
要裁决 `PD=0.35--0.375` 是否达标，应扩大严格隔离的 H0/H1 test，而非继续围绕
门限调参。

产物：`data/full_observation_detector_calibration_fine_20261015/`。

#### 最小分布式检测信息上限筛查

冻结 +50 dB one-block GN-MAP、TP-UIC、sigma-point covariance 与 detector，
在同一物理场景中分别生成 6 个接收 UAV 的本地分数。所有接收机共享 truth、
belief、base 和 scene ID，但保留各自接收噪声与直达径估计。筛查为 1 train /
2 calibration / 8 test；由于 calibration 不足以支持 5% 有限 conformal 门限，
本节只报告 AUC，不作 PD/PFA 结论。

| 固定接收集合 | sum AUC | max AUC |
|---|---:|---:|
| receiver 1 | 0.7813 | 0.7813 |
| UAV {0,1} | 0.8750 | 0.8125 |
| UAV {0,1,2} | 0.8906 | 0.8125 |
| All 6 | 0.8906 | 0.7969 |

同一 8-test 上遍历子集得到的诊断天花板为：最佳单 UAV `{0}` AUC `0.9063`，
最佳双 UAV `{0,3}` 为 `0.9219`，最佳三 UAV `{0,1,4}` 为 `0.9375`。这些子集
直接使用 test 选择，绝不能作为 Proposed 或可部署关联结果；它们只表明三节点
相对 test-best 单节点的剩余信息上限约为 `0.031`。四节点不再提高，五/六节点
分别降至 `0.9219/0.8906`，所以“更多 UAV 自动更好”不成立，原始 max 融合也
系统性差于 sum。

裁决：最小分布式方向通过“存在信息”筛查，但尚未通过“关联有价值”门禁。
正式下一步必须在独立 train 上冻结子集或低维融合规则，再使用充足 H0 calibration
和至少 40--100 个 test；核心基线应是最佳单 UAV 和 SINR Top-K，而不是较弱的
receiver 1。若独立 test 上相对最佳单 UAV的增益不能稳定达到预注册最小效应，
则不应把关联列为核心贡献。

产物：`data/distributed_detector_headroom_screen_20261016/`。

- `data/joint_dd_solver_benchmark_nfev4_20261009/`
- `tools/gate_area400_dual_axis.py`
