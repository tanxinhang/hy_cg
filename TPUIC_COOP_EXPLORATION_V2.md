# TP-UIC—协作融合持续探索协议 V2

日期：2026-09-19
固定边界：`n_cpi=1`；不使用 CPI 增益解释任何改进。

## 1. 目标与基本原则

探索分成三条独立轨道，最后汇合：

1. **TP-UIC 接收机轨道**：改善残余干扰的下尾，同时保持目标证据和 CFAR；
2. **协作与融合轨道**：在冻结接收机能力下改善弱目标、链路选择、融合与功率分配；
3. **联合闭环轨道**：让调度状态改变 TP-UIC 能力，TP-UIC 能力再反向改变调度，而不是串行拼接。

任何候选都使用相同的 truth/belief、几何、信道、H0/H1 随机流和资源约束做配对比较。
不允许用最终 `P_D` 单指标调参，也不允许把 pooled Wilson 区间当成逐场景保证。

## 2. 冻结评价口径

主场景暂定为 `M=6, Q=3, area=600 m, RCS=0.1, m_rx=4, max_tx=3`。
外推场景至少包含 `area=800 m` 和更低 RCS；所有候选先在主场景筛选，参数冻结后再外推。

### 2.1 接收机级指标

- 消除深度：中位数、P10、最差接收机，不只看均值；
- 目标生存：逐 `(receiver,target)` 的中位数、P10 和最小值；
- 条件检测证据：固定场景多次噪声实现的 conditional AUC / `P_D`；
- CFAR：逐场景经验 `P_FA`、场景聚类区间、经验分位数/名义阈值；
- 复杂度：矩阵维数、求解次数和运行时间。

### 2.2 系统级指标

- 实际平均 `P_D` 与场景聚类区间；
- 实际最弱目标 `P_D`，以及 truth-predicted worst-target `P_D`；
- `P_FA` 同时报 pooled Wilson 和按场景聚类区间；
- 发射节点数、选择链路数、捕获链路数、上报比特与时延；
- belief objective 与 truth objective 的偏差。

系统级 `P_D` 不再只对每个目标抽一次 H1。筛选阶段对固定的
`(scene,target,selected-set)` 做多次 H1/H0 重复，以区分检测噪声和场景差异。

## 3. 三条探索轨道

### A. TP-UIC 接收机

按以下顺序探索，每一步只改变一个机制：

1. **基准**：当前 hard `tp_uic_full`；
2. **协方差一致的条件标定**：保持消除算子不变，用接收机导出的残差模型做逐场景/分层阈值标定；
3. **正则化下尾控制**：探索 interference prior / ridge 强度，但参数必须由噪声、条件数或
   discrepancy principle 决定，不能按 truth `P_D` 选；
4. **稳健软保护**：保护权重由 belief covariance 和目标—干扰子空间夹角产生，而不是固定 `mu`；
5. **空间—DD 联合保护**：只利用单位范数阵列流形的可辨识性，不计入额外阵列收集增益；
6. **接收机能力证书**：输出每个 `(j,q)` 的 residual、retention、null-tail scale 和不确定区间。

已有负对照继续保留：固定 `mu` 的 quadratic soft protection、targeted hard protection、
一阶 nuisance projection。它们只有在新的条件标定下形成新的 Pareto 点时才重新进入候选集。

接收机晋级条件：

- 主场景与外推场景的 `P_FA` 区间均覆盖名义 0.05，且不存在系统性上偏；
- cancellation P10 严格优于 hard baseline，或在深度不退化时显著改善 retention P10；
- conditional detection evidence 不下降；
- 改进不能依赖 truth 信息。

### B. 协作选择与融合

优先探索已有正证据的方向：

1. RCS-robust bundle / distributionally robust link valuation；
2. 节点级 sensing/report power allocation；
3. 直接优化 worst-target `P_D` 或 CVaR，而不是只优化平均效用；
4. 与通信失败模型一致的 exact/mixture LLR 融合阈值；
5. 链路冗余从固定“至少两条”改为由捕获概率和相关性决定；
6. 发射节点数、残余干扰和报告预算的显式联合价格。

协作晋级条件：冻结同一个接收机能力表，配对提升 worst-target 指标；`P_FA` 不恶化；
所有硬资源约束逐 trial 通过；收益在 800 m / 低 RCS 外推中方向一致。

### C. TP-UIC—协作联合闭环

联合状态至少包含：

`illumination mask + selected links + fusion destinations + per-node power mode + receiver protection mode`。

每轮执行：

1. 调度器提出一个可行联合状态；
2. TP-UIC 在该状态下测量逐 `(j,q)` 能力证书；
3. 选择器和融合器用证书重新计算条件 `P_D`、null-tail 和资源效用；
4. 只接受在 belief 可见信息上严格改善且保持可行的状态；
5. 保存全程历史，并在 truth 上做独立审计，不用 truth 决定接受与否。

联合候选必须分别击败：冻结 40 dB 接收机、hard TP-UIC 串行、hard TP-UIC 联动，
并通过组件消融，证明收益不是由某一个未声明的自由度单独造成。

## 4. 当前证据与首轮决策

- 当前 hard TP-UIC 联动相对串行有效，但绝对 `P_D=0.7556`、最弱目标预测
  `P_D=0.3929`，不能晋级为最终方案；
- measured TP-UIC 相对固定 40 dB 的 truth worst-target `P_D` 低 0.0885，说明接收机
  下尾仍有改进空间；
- 固定 soft-protection 权重与 targeted-hard 尚未形成更优 Pareto 点，暂列负对照；
- nuisance-manifold 能校准 CFAR，但已有小样本显示 H1 证据下降，暂不直接接入系统；
- RCS-robust bundle 在 200 次实验中相对 nominal bundle 的 `P_D` 提升 0.056；
- 节点级 power-joint 在既有 RCS=0.1 实验中将 `P_D` 从约 0.589 提至 0.663；
- 因而首个联合候选为：**hard TP-UIC 能力证书 + RCS-robust weak-target objective +
  node-level power/mask alternation + 条件阈值标定**。

## 5. 逐轮推进规则

每轮固定执行：小样本筛选 → 失败归因 → 候选冻结 → 独立 holdout → 代码审计 →
更大配对 Monte Carlo。失败候选不删除，保留配置、随机种子和原始 CSV。

停止或回退条件：

- `P_FA` 校准失败；
- 目标生存或 conditional evidence 明显下降；
- 只有均值改善、最弱目标恶化；
- 改进只出现在调参场景而不出现在冻结 holdout；
- 资源约束、truth/belief 隔离或共同随机数配对被破坏。

## 6. Round 1 记录

### 6.1 TP-UIC ridge prior（未晋级）

在 4 阵元、三个配对场景上，以 `prior_variance=1` 为基准：

- `0.3` 的 cancellation P10 平均仅提高 0.005 dB，某场景最差目标存活下降约 0.043；
- `0.1` 的 cancellation P10 平均下降 0.176 dB，目标存活也下降。

结论：仅调整 interference ridge prior 没有形成消除—存活 Pareto 改善，保留为负对照。

### 6.2 原选择器上的 RCS rescaling（未晋级）

在同三个场景、三发射节点约束下：

- `factor=0.5`：TP-UIC robust joint 的 truth worst-target `P_D` 平均下降 0.0244，
  truth objective 下降 0.0832；
- `factor=0.75`：truth worst-target `P_D` 仅提高 0.0009，truth objective 提高
  0.0258，三个场景方向混合；
- 用每目标 50 次 H1 重复复核，`factor=0.75` 的 joint `P_D` 相对 nominal
  仅提高约 0.0089。

结论：把 RCS 下界直接乘进原 C2F 选择器不能复现 robust bundle 的收益；真正的
RCS-robust bundle 必须作为结构化候选进入后续联合闭环。

### 6.3 Gaussian-replacement exact-mixture threshold（正确但收益小）

新增确定性 centred-Gamma / Gaussian-replacement 混合分布分位数标定，默认关闭，
保持发布路径逐位不变。三场景、每目标 300 次 H0 中：

- TP-UIC robust joint `P_FA`：0.05074 → 0.04963；
- 八个臂的平均绝对校准误差：0.00181 → 0.00167。

结论：实现方向正确，但相对 Cornish--Fisher 的实际收益不足以补偿额外计算量，暂不
晋级默认融合器，保留为精确校准对照。

### 6.4 条件 `P_D` 重复（晋级为实验基础设施）

检测器新增默认值为 1 的 `num_h1_per_target`；大于 1 时在同一几何、目标和选择集合
上使用独立 keyed H1 实现。默认协议完全不变。

三场景、每目标 50 次 H1 下，nominal TP-UIC robust：

- serial `P_D=0.5267`；
- joint `P_D=0.6867`；
- joint minus serial `+0.1600`。

这取代九个单次目标判决给出的粗粒度印象，后续所有小样本候选筛选均使用条件
H1 重复；最终外推仍以独立场景聚类区间为准。

## 7. Round 2 记录

### 7.1 RCS-robust bundle 的辐射节点硬约束

`joint_bundle_column_generation` / `rcs_robust_bundle_column_generation` 新增
`allowed_transmitters`。候选池在进入 master 前过滤不允许的发射节点；单元测试证明
最终所有链路的发射端均属于声明集合。默认 `None` 保持原算法不变。

### 7.2 自由 mask 的 RCS-bundle 闭环（未晋级）

全开状态枚举全部三节点组合，由 RCS-robust bundle 的词典序目标选择 mask；随后把
该 mask 送入 TP-UIC 并继续严格改善。三个场景、每目标 50 次 H1：

- 固定 40 dB joint：相对 C2F，`P_D +0.0511`、truth worst-target `P_D +0.0866`；
- 实测 TP-UIC joint：相对 C2F，`P_D -0.1200`、truth worst-target `P_D -0.1696`、
  truth objective `-0.5089`；
- 同时 belief worst-target `P_D +0.0792`。

结论：bundle 在冻结接收机上有效，但自由选择的 mask 对单一 belief 过拟合；接入
TP-UIC 后形成“belief 更优、truth 更差”的反向缺口，不得晋级。

### 7.3 C2F mask + RCS-bundle 链路/融合 hybrid（未晋级）

为隔离 mask 失配，hybrid 先用 C2F 确定最多三个发射节点，再只允许 bundle 在该
mask 内重选链路和融合节点。相对完整 C2F joint：

- TP-UIC 深度不变；
- truth worst-target `P_D +0.0028`；
- truth objective `-0.0087`；
- conditional `P_D -0.0311`；
- 每场景平均多 2 条链路。

结论：固定可靠 mask 后，当前 bundle 的额外链路/融合自由度没有产生足够收益。
Round 2 关闭“继续增加 bundle 链路”的方向；下一优先级转为 TP-UIC 能力证书下尾
及其不确定性，而不是继续扩大协作集合。

## 8. Round 3 记录

### 8.1 规划能力证书与 held-out 接收机实现分离

联合实验新增两套能力：

- **planning certificate**：由一个或多个参考接收机实现聚合，只供 belief 调度使用；
- **held-out realization**：独立随机流生成，只用于 truth link table 和最终检测。

`capability_reps=1` 且未启用 held-out 时逐项复现旧结果；显式 held-out 后，点估计
规划与多次证书共享完全相同的评估实现，可以严格配对。

### 8.2 三次测量的20%保守分位证书（未晋级）

三个场景中，以单次规划证书为基准，三次参考测量取 residual 80%分位、retention
20%分位：

- joint conditional `P_D +0.0533`；
- truth worst-target `P_D -0.0435`；
- truth objective `-0.0784`；
- 规划 cancellation 中位数平均降低约 1.04 dB。

结论：保守标量分位改变了选择，但没有保护最弱目标，不能晋级。

### 8.3 三次测量的中位证书（未晋级）

同一批 held-out 实现中：

- joint conditional `P_D +0.0167`；
- truth worst-target `P_D -0.0080`；
- truth objective `-0.0185`；
- 每场景平均少 1.67 条链路。

结论：中位证书有轻微效率趋势，但不满足“最弱目标不退化”的主门槛。Round 3
关闭少量重复+标量分位聚合路线。下一候选必须保留逐 `(receiver,target)` 的结构，
用解析协方差或风险传播形成能力下界，而不是继续扫描分位点。

## 9. Round 4 记录

### 9.1 解析残差能力证书接口

`ReceiverMeasurement` 现在显式导出 `i_res_pred`、`i_in_pred`、
`predicted_fraction` 和 `predicted_kappa_db`。其中实测桥仍使用
`i_res / i_in`；解析桥使用 `E[I_res] / E[I_in]`，两者不再混写。
联合实验新增 `--capability-source measured|predicted`，truth 评估始终使用独立
held-out 接收机实现，默认仍为 `measured`。

首次实现把解析 `i_res_pred` 除以一次随机相位实现的 `i_in`。审计发现这把本应在
观测前稳定的能力证书重新注入了直达相位 Monte Carlo 抖动。修正后，`i_in_pred`
由直达字典每个物理路径的中心列能量求和，即与信号生成器中“中心系数为单位功率
随机相位、导数系数为零”的模型一致。单元测试固定了以下不变量：同一几何下改变
接收噪声和随机相位，`predicted_fraction` 不变，而实测 `i_in` 可以变化。

### 9.2 解析残差证书替换单次实测证书（未晋级）

固定三个场景、独立 held-out 接收机实现、每目标 20 次 H1 和 200 次 H0、exact
mixture threshold。相对单次 measured planning certificate，修正后的 predicted
certificate 在 TP-UIC robust joint 上：

- conditional `P_D +0.0167`；
- `P_FA +0.0017`；
- truth worst-target `P_D -0.0816`；
- truth objective `-0.2033`；
- predicted-vs-held-out cancellation 中位数的平均绝对误差约 `1.09 dB`。

逐场景 truth worst-target 差值为 `-0.2610 / +0.0436 / -0.0275`，收益不稳且
trial 0 明显失效。结论：保留“期望分子不能除以随机实现分母”的公式修正和解析字段，
但拒绝将 predicted certificate 设为默认。当前解析残差只描述接收机的平均估计误差，
尚未形成能保护 weakest-target 的风险下界；下一轮应优先构造 belief-side 的逐
`(receiver,target)` 目标存活/泄漏证书，替代当前仍由真值回放得到的 retention。

## 10. Round 5 记录

### 10.1 belief-side 目标存活证书

接收机新增两类不读取 truth echo 的逐 `(receiver,target)` 证书：

- `predicted_retention_by_target`：believed target 字典的物理中心列经过实际消除
  线性算子后的期望剩余能量；
- `risk_retention_by_target`：在 belief 均值及每链路 Doppler/delay 的 95% 边缘
  sigma 点上计算同一量，并取最小存活。

中心列与信号生成模型一致：每条物理目标路径只有中心系数携带单位功率随机相位，
导数列只描述局部流形而不被当作额外散射体。单元测试证明两类证书均不随 truth
相位/接收噪声变化，且 risk 不大于 nominal。

### 10.2 nominal belief retention（正确但对当前调度不可辨识）

三个场景中，nominal planning retention 中位数位于 `0.999966–0.999997`。相对
truth-derived measured planning retention，四个 TP-UIC 臂在每个场景的选择集合
完全相同，`P_D`、`P_FA`、truth worst-target 和 objective 差值均为零。

结论：它能证明“算子在 belief 中心上几乎不伤目标”，但没有为协作选择提供足够
动态范围，不晋级为控制证书，保留为中心点诊断。

### 10.3 轴向 95% sigma-point risk retention（未进入决策）

trial 0 全开状态：

- nominal retention 的 P10/最小值为 `0.9970/0.9459`；
- risk retention 为 `0.8704/0.8395`；
- 独立 realised retention 为 `0.9331/0.8846`。

risk 证书产生了真实动态范围，但仍有个别链路因联合 DD/方位误差未建模而偏乐观。
接入 trial 0 联合调度后，四个 TP-UIC 臂的 `selected_json`、链路数和最终指标仍与
measured 基线完全相同，因此按筛选门槛停止扩样。结论：当前轴向 sigma 点是有用的
风险诊断，但尚未成为有效控制量。下一轮需要同时解决两点：加入联合位置—速度诱导
的相关 DD/方位 sigma 点，并把 weakest-target 风险直接写入选择目标或约束；仅替换
链路 SINR 中一个接近 1 的乘法因子不足以改变现有 C2F 排序。

## 11. Round 6 记录

### 11.1 固定资源的 weakest-target 1-swap（未晋级）

新增 belief-side 固定预算 polish：冻结 illumination mask、fusion plan 和总链路数，
每轮删一条、加一条，只接受词典序改善
`(min_q predicted P_D,q, sum_q min(P_D,q, weak_pd_required))` 的交换，并重新检查
发射节点、每目标、接收机、融合节点和上报约束。默认 `0` 轮，发布路径不变。

三个场景、risk-retention 表上，TP-UIC joint-robust 的 polish-vs-no-polish：

- belief worst-target `P_D +0.00265`；
- truth worst-target `P_D +0.00180`；
- conditional `P_D -0.0333`；
- `P_FA -0.0017`；
- truth objective `+0.0024`，但场景方向混合。

trial 2 的三次 belief 单调交换导致 conditional `P_D -0.10`、truth objective
`-0.0921`。实现没有违反目标；失败来自 belief 风险证书不能可靠排序 truth 交换。
结论：固定资源 worst-target polish 保留为负对照，不晋级。

## 12. Round 7 记录

### 12.1 belief 协方差与实际误差生成模型一致化

审计发现 belief 生成器把目标高度固定为 truth、垂直速度固定为 0，却仍给 `z/v_z`
非零协方差。scheduler 和 receiver 因而为不会发生的垂直误差付费。现统一为水平
`(x,y,v_x,v_y)` 四维协方差；receiver DD Jacobian 与 scheduler 使用同一口径，
并新增 receive direction-cosine Jacobian。

### 12.2 相关状态 sigma-point retention（诊断，不是严格界）

同一个四维状态 sigma 点现在会在该目标的所有 bistatic paths 上一致移动 Doppler、
delay 和 bearing。trial 0 中：

- risk P10/最小 retention：`0.8702/0.6863`；
- held-out realised：`0.9331/0.8846`；
- 平均保守约 `0.0404`，但 18 个单元中仍有 5 个偏乐观，最大约 `0.0935`。

因此它只能称 sigma-point 风险排序，不能称 95% survival 下界。修正后的 trial 0
joint-robust 已是 1-swap 局部最优；serial 仅一处交换且 truth worst-target 只提高
约 `0.00265`，故停止扩样。

## 13. Round 8 记录

### 13.1 covariance-inflated TP-UIC protection 与 stage-2 旁路修复

新增默认关闭的 `cancellation.covariance_protection`：hard protection basis 加入 belief
DD/bearing 的 95% marginal sigma columns。首次 receiver screen 与 baseline 完全相同。
代码审计证明这是结构性旁路：`tp_uic_full` stage 2 从原始 `y` 重做 `[X,A]` 联合拟合，
但 `A` 仍是中心一阶字典，所以 stage-1 扩展从未进入最终算子。

修复后，covariance columns 同时进入 belief-side stage-2 target dictionary；truth echo
仍只使用物理中心/切线生成口径，并拆分 belief/truth column IDs。随后又发现 GLRT 仍按
固定“每三列一中心”解释可变宽字典；新增显式 belief/truth centre-mask provenance，
并修复 `restrict_to_target` 同步裁剪。修复前的 detector evidence 全部作废。

### 13.2 receiver-only Pareto screen（三场景）

covariance-stage2 相对 tangent baseline：

- cancellation P10 三场景全部提高，平均 `+0.0711 dB`；
- 最差 receiver cancellation 全部提高，平均 `+0.0829 dB`；
- 最差 target survival 全部提高，平均 `+0.00191`；
- median cancellation 平均 `-0.2083 dB`；
- survival P10 平均 `-0.00058`；
- median noise enhancement 平均 `+0.0816 dB`。

它形成了小幅下尾重分配候选，但不是全面 Pareto 支配。

### 13.3 条件检测证据（未晋级）

修正 centre provenance 后，同一场景/receiver 的 16 对 H1/H0：

- tangent baseline：conditional AUC `0.535`，paired probability `0.500`；
- covariance-stage2：conditional AUC `0.535`，paired probability `0.500`；
- truth-template oracle：conditional AUC `0.723`，paired probability `0.750`。

候选改变了原始统计量但没有改善 H1/H0 排序。按照“conditional detection evidence
不得下降且应支持接收机收益”的门槛，不进入 CFAR 扩样，也不接入协作闭环。其价值是
暴露并修复 stage-2 protection 旁路和可变字典 provenance；算法本身保留实验开关，
默认关闭。

## 14. Round 9 记录

### 14.1 covariance composite-subspace GLRT（未晋级）

把目标的 tangent/sigma columns 同时当作自由信号子空间，并按实际秩调整 GLRT
自由度。单场景 16 对 H1/H0 中：

- centre baseline conditional AUC `0.535`；
- tangent composite `0.430`；
- covariance composite `0.438`，paired probability `0.375`。

自由信号维数增加带来的噪声代价大于失配捕获收益，停止该路线。

### 14.2 四维相关 ±1σ max-template search（未晋级）

每个候选模板由同一水平位置/速度 sigma 点一致移动目标的所有 bistatic DD 和 bearing，
H1/H0 均在中心加八个 sigma templates 上取最大统计量。trial 0 的 conditional AUC
`0.535→0.641`，但三个场景逐项变化为 `+0.105/+0.051/-0.102`，总体仅
`0.487→0.505`。max 操作放大 H0 极值，方向不稳定，未进入 CFAR。

### 14.3 sigma-template mean statistic（holdout 未确认）

为避免 look-elsewhere 极值，固定等权平均中心与八个 sigma-template 统计量。前三场景：

- conditional AUC `0.487→0.510`；
- paired probability `0.458→0.500`；
- 场景间 AUC 标准差 `0.080→0.031`。

冻结后在 trial 3--5 holdout：conditional AUC `0.576→0.609`，但 paired probability
`0.583→0.542`。六场景合并，conditional AUC 平均差 `+0.0286`、paired 差为 `0`，
仅 3/6 场景 AUC 为正。结论：平均方向略好但不稳定，不晋级、不做专用 H0 阈值标定。

本轮同时为 detector-evidence 工具加入固定几何 residual-model 缓存和 `trial-start`，
使独立 holdout 不必重复昂贵的 affine/covariance 构造；缓存只复用几何固定模型，
每个 H1/H0 观测及统计量仍独立计算。

## 15. Round 10 记录

### 15.1 mask-role 功率与 TP-UIC 能力联动

审计现有 `power_joint.py` 后确认其功率表不接受 active mask，也没有 TP-UIC residual/
retention 回调，直接串接会把功率、消除能力和协作选择割裂。因而在 joint runner 中新增
默认关闭的 `mask_role` 实验策略：每个候选发射 mask 都重建对应 `rho_by_uav`，重新测量
TP-UIC 能力并重建选择表；检测端也使用同一有效配置。能力缓存键加入完整功率向量，避免
跨策略误复用。

激进候选 `active rho=0.95, inactive rho=0.20` 在 trial 0 失败：TP-UIC joint-robust
的 conditional `P_D 0.8667→0.7167`、`P_FA 0.0367→0.0450`、truth worst-target
`P_D 0.8579→0.4007`、truth objective `2.3449→0.9987`。planning/held-out cancellation
出现约 `2.52 dB` 缺口，且 mask 从 `[0,3,4]` 改为 `[0,1,4]`；新增通信余量不足以抵消
发射节点感知功率改变造成的 TP-UIC 能力不稳定，故立即淘汰。

随后测试守恒候选 `active rho=0.80, inactive rho=0.20`：发射节点保持 baseline 的
感知功率，仅把未激活节点的剩余功率释放给通信。TP-UIC joint-robust 的 mask、融合节点、
12 条选择链路、11 条可捕获链路、planning/held-out cancellation 以及 `P_D/P_FA`、
truth worst-target 和 truth objective 与 uniform baseline 逐项完全一致。功率向量严格为
active `0.8`、inactive `0.2`，每节点 sensing/communication 份额仍和为 1。

结论：`0.8/0.2` 实现与资源口径正确，但当前 TP-UIC 解未利用 inactive 节点新增的通信
余量，属于策略等价而非性能改进，不扩展随机试验。两组结果共同否定“只按发射角色改
功率”这一维度；下一轮应直接优化 fusion node/链路分配，并在每个候选内部重测 TP-UIC
能力，而不是继续扫描固定 role-power 比例。

## 16. Round 11 记录

### 16.1 TP-UIC-aware 静态 fusion lookahead（未晋级）

joint runner 新增显式 `fusion_rule` 记录，并用已有 `capacitated_pd_lookahead` 在
TP-UIC residual/retention 修正后的 belief table 上选择融合节点。trial 0 中 serial
融合节点从 `[5,5,5]` 变为 `[1,1,0]`，但 TP-UIC joint 最终仍为 `[0,0,0]`，其
mask、12 条链路、11 条可捕获链路、`P_D/P_FA`、truth worst-target 和 objective
与 `max_in_rate` 基线逐项相同。说明选择前的另一种静态融合规则没有改变联合不动点。

### 16.2 固定 sensing schedule 的 fusion-destination PD polish

新增严格实验函数 `maximize_fixed_set_pd`：保持 sensing links、发射 mask 和 TP-UIC
能力证书不变，逐目标枚举可行融合节点；只允许 scheduler-side predicted `P_D` 不降，
数值并列时才减少远程报告。存在跨目标 fusion/report/CPU 容量时拒绝运行，避免错误地
声称目标可分。穷举一致性与不可达目的节点过滤测试通过。

trial 0 的 TP-UIC joint-robust 融合节点由 `[0,0,0]` 变为 `[0,2,3]`，逐目标远程
报告数由 `[2,2,5]` 降为 `[2,1,4]`；belief/truth objective 都增加 `0.01067`。
但三个目标的 predicted `P_D` 完全不变，conditional `P_D=0.8667`、`P_FA=0.0367`、
truth worst-target `P_D=0.8579` 也全部不变。实现边界审计确认 selected links、mask、
TP-UIC `kappa/eta` 和 capture count 均未变化。

结论：该 polish 是有效的等检测性能通信开销压缩，但不是弱目标或消除收益，不晋级检测
主线。结合 Round 2 的 RCS-bundle/hybrid 负结果，继续更换静态 fusion heuristic 或
增加协作链路的预期收益很低；下一优先级回到 TP-UIC 接收机下尾失败机制，再用能力证书
把确定的接收机改进反馈给调度。

## 17. Round 12 记录

### 17.1 target-conditioned residual 能力接口修复

代码审计发现 `targeted_tpuic_full` 虽按 tested target 重建保护/联合算子，但
`ReceiverMeasurement` 只导出 `(receiver,)` residual fraction；`all_targets=True` 仅重算
了 `(receiver,target)` retention。joint runner 因而用 target 0 的 residual 能力覆盖全部
目标，旧 targeted 结果并非完整的 target-conditioned 联动。

现新增 measured/predicted `(M,Q)` residual certificate，并允许 link table 接收一维或
二维 residual。一维路径保持原算法；二维路径对每个目标的 sensing SINR 使用对应分母，
pair-level `rinr/sigma0` 取目标最坏 residual 作保守诊断。单元测试证明二维每个 target
slice 与单独构建的一维 table 逐项一致，raw SINR 不变，错误 shape 被拒绝。

### 17.2 修正后的 targeted TP-UIC 配对实验（未晋级）

固定 `n_cpi=1`、三个场景、每目标 20 次 H1/200 次 H0、独立 held-out receiver。
相对 union-protection `tp_uic_full`，targeted 2D 的 joint-robust 逐场景差值为：

- conditional `P_D`：`+0.0667 / -0.0167 / 0`，均值 `+0.0167`；
- `P_FA`：`+0.00167 / 0 / -0.00167`，均值不变；
- truth worst-target `P_D`：`+0.01835 / -0.00058 / -0.00506`，均值 `+0.00424`；
- truth objective：`+0.03055 / -0.00456 / -0.02322`，均值约 `+0.00092`；
- belief worst-target 三个场景全部下降。

每场发射节点数、总链路数和 truth-capture count 均与对应 union arm 相同，但选择链路可
改变。困难场景中 planning--held-out cancellation 的逐单元平均绝对误差仍约
`2.8--5.8 dB`；target conditioning 没有稳定缩小该缺口。结论：二维 residual 是必要的
模型/接口修复并保留，但 targeted 算法只有 1/3 场景正向，不晋级默认 TP-UIC。当前主瓶颈
进一步定位为能力证书的 realization 风险，而不是保护子空间是否按目标拆分。

## 18. Round 13 记录

### 18.1 production-model-aligned measured bridge（未扩样）

生产链路表的直达干扰场是各路径功率求和，而原 measured bridge 为
`i_res(realized) / i_in(realized coherent phase)`；把该比值乘回非相干功率场会保留一项
随机相位分母。新增独立实验口径 `measured_model = i_res / i_in_pred`，旧 measured 与
expected/expected predicted 口径均保留，不改变默认行为。target-conditioned 版本同样支持
`(M,Q)`。

trial 0 中，model-aligned TP-UIC joint-robust 相对原 measured bridge：conditional
`P_D/P_FA` 完全相同，truth worst-target 仅 `+0.00072`、truth objective `+0.00126`，
链路集合只发生等价排序变化。更重要的是其逐 receiver planning--held-out depth MAE 仍为
约 `6.10 dB`，没有缩小 realization 风险。说明主要不确定性来自 residual numerator，
而不仅是 coherent input denominator，故不扩展场景。

审计同时修复实验报表：`measured_model` 的 held-out `kappa` 现在由同一 model-aligned
`eval_fraction` 计算，不能再混用 realized/realized `evaluated.kappa_db`。该修复只影响
证书标签，不影响已运行实验的选择或检测结果。

## 19. Round 14 记录

### 19.1 residual numerator 二次型矩与覆盖审计

对固定几何和 `n_cpi=1`，TP-UIC 报告的 residual power 可写为两个独立二次型：
`||B h||² + ||F n||²`。其中 direct coefficients 是独立单位随机相位，noise 是 circular
complex Gaussian。新增不形成大矩阵的解析矩：结构项方差为 residual Gram 的非对角
能量，噪声项方差为 `sigma^4 tr((F F^H)^2)`。

在 trial 0 的四个 receiver 上分别做 20/40 次固定几何随机相位与噪声回放：

- empirical/analytic mean：`0.894 / 1.064 / 1.036 / 0.865`；
- empirical/analytic variance：`0.704 / 1.293 / 1.037 / 0.625`；
- alpha=0.2 Cantelli upper coverage：`1.000 / 0.875 / 0.950 / 1.000`，均不低于 0.8。

因此矩公式和单侧覆盖方向成立；该审计只重绘 receiver realization，不改几何、belief、
channel 或 CPI。

### 19.2 Cantelli risk-moment 闭环（未晋级）

把 `mean + sqrt((1-alpha)/alpha * variance)`、`alpha=0.2` 作为 planning residual
certificate，held-out 使用 model-aligned realised residual。trial 0 的 TP-UIC
joint-robust 相对 measured union baseline：

- conditional `P_D 0.8667→0.8000`；
- `P_FA 0.0367→0.0433`；
- truth worst-target `P_D 0.8579→0.5560`；
- truth objective `2.3449→1.5429`；
- 选择/捕获链路由 `12/11` 降为 `11/10`。

六个 receiver 的 planning depth 均不比 held-out 乐观（0/6 optimistic cells），但过度
保守使调度选择显著变差。结论：解析矩保留，Cantelli 作为严格负对照保留，但不扩样、
不晋级。下一候选应直接使用二次型先验预测分布的分位数，而不是继续缩放通用不等式。

## 20. Round 15 记录

### 20.1 二次型 prior-predictive residual quantile

在 Round 14 的矩之外，receiver 现导出 structural residual Gram 与 fitted-noise 的非零
特征值。固定 8192 个先验样本直接生成
`h^H G h + sum_k lambda_k Exp(1)` 的确定性分位数：随机流固定，只依赖 belief geometry
和消除算子，不读取 held-out realization，也不增加 CPI。单元测试固定谱矩等价、分位数
确定性及 `q50 <= q80`。

trial 0 四个 receiver 共 200 次随机相位/噪声回放中，80% prior quantile 的合并覆盖率
为 `161/200=0.805`；逐 receiver 为 `0.750/0.775/0.825/0.900`。因此它是总体校准的
prior-predictive quantile，但不是逐节点严格下界；该限定写入结论，不能称 80% worst-case
保证。

### 20.2 prior-quantile 联合闭环（未晋级）

把 q80 residual fraction 用作 planning capability，held-out 使用 model-aligned realised
residual。trial 0 的 TP-UIC joint-robust 相对 measured union baseline：

- conditional `P_D 0.8667→0.7500`；
- `P_FA 0.0367→0.0417`；
- truth worst-target `P_D 0.8579→0.5899`；
- truth objective 明显下降。

该候选同样在初始 selector 中形成 `[0,1,3]` mask 并已是固定点；失败不是 strict joint
admission 接受了坏的后续状态，而是风险能力表本身改变了初始链路/mask 排序。结论：
prior quantile 基础设施与校准证据保留，作为诊断/后续约束输入；直接替代 nominal link
SINR 的路线关闭，不扩样、不晋级。

## 21. Round 16--17 记录

### 21.1 prior-risk 只作二级优化：首次结果经审计作废

为避免 Round 15 的风险表直接改写 nominal 排序，新增双表固定资源 polish：nominal
measured table 保持主合同，q80 prior-risk table 只在合同允许的调度中改善风险侧
weakest-target/capped-service。首次 trial 0 的一步交换给出 conditional
`P_D 0.8667→0.8833`、`P_FA 0.0367→0.0317`，truth weakest-target 与 truth objective
不降；但审计发现 configured belief objective `1.26487→1.25837`。原实现只约束了
nominal weakest-target 和 capped service，遗漏了实际选择器总效用，因此该正结果无效，
不得用于晋级判断。

修正后把与选择器一致的 deflection/PD utility 及 delay price 纳入逐交换硬约束，并增加
回归断言。完全相同随机流重跑后交换数为 0；`P_D=0.8667`、`P_FA=0.0367`、belief
objective `1.26487`、truth weakest-target `0.85787`、truth objective `2.34490` 均与
baseline 逐项相同。说明先前的检测增益来自违反 nominal 合同的交换，而不是有效改进。

### 21.2 target-local 2-for-2 安全邻域（未晋级）

为排除一步局部障碍，新增可选 target-local 2-for-2 精确邻域：先同时移除两条同目标
链路，再同时加入两条；只审查最终调度，并重新检查发射节点、local/remote report、
processing cap。它仅在不存在安全 1-swap 时启用，最终解仍须同时保持 nominal weakest、
capped service 和 configured objective，并严格改善 prior-risk 词典序目标。

trial 0 的 TP-UIC joint-robust 仍为 0 次交换，全部检测、truth/belief 和资源指标与
baseline 完全相同。serial 状态虽存在满足合同的一步交换，但没有传播为 joint 闭环收益。
结论：在当前可靠发射 mask `[0,3,4]` 内，风险二级目标既没有安全 1-swap，也没有同目标
2-for-2 改进；不扩样。该实现保留为默认关闭的审计工具，下一轮回到 TP-UIC 本体，探索
由 belief 可观测量决定的自适应软保护，而不是继续扩大无收益的链路交换邻域。

## 22. Round 18--29 记录

### 22.1 belief-only adaptive soft TP-UIC 与量纲容差修复

新增默认关闭的 adaptive soft arm。对固定 `mu={0,0.01,0.1,1,10,100,1000}`，仅使用
belief-side residual quadratic-form mean/q80 和 correlated-state risk retention 选择；候选
必须相对 hard TP-UIC 保持 residual mean、q80，并在声明的 retention slack 内，无法满足
则逐项精确回退 hard。最初实现错误使用固定 `1e-12` 功率容差，而典型 residual power
本身约为 `1e-13`，导致安全门槛失效。逐 `(receiver,target)` 审计发现三处被错误接受的
软单元，其 predicted/q80 residual 实际均更差。现改为相对 hard residual 的 `1e-10`
比例容差，并增加算子/GLRT 低秩复现测试。

严格 `slack=0` 时 trial 0 的 18/18 单元全部回退 hard，所有指标逐项相同，证明合同正确
但没有改进点。冻结、预先声明的 `slack=0.001` 在三个独立场景形成稳定的接收机下尾
重分配：cancellation P10 分别提高 `+0.537/+0.102/+0.048 dB`；median 分别变化
`-0.172/-0.149/0 dB`；target-survival P10/最小值变化约 `1e-5` 或更小。该候选是下尾
改善而非全面 Pareto 支配。

### 22.2 conditional evidence、空间 belief covariance 与 CFAR

hard/adaptive 在 3 场景 × 3 receiver × 16 对 H1/H0 上，conditional AUC 为
`0.477/0.476`、paired probability 均为 `0.472`；唯一出现差异的固定单元扩至 64 对后
AUC 均为 `0.490`、paired probability 均为 `0.469`，故没有检测排序收益，也未确认损失。

未建模 belief error 时，两臂的 production-belief CFAR 均为 `P_FA=0.0644`，Wilson
`[0.0502,0.0824]`，q95/threshold 约 `1.043`。开启现有 belief-error covariance 暴露
空间路径缺陷：DD mismatch factor 为 4096 维，而 `m_rx=4` 的模型为 16384 维。现将
delay/Doppler 导数按 source bearing 提升到阵列空间，并加入 bearing derivative covariance。
修复后 3×3×100 H0：hard `P_FA=0.0511`、adaptive `0.0500`，Wilson 分别为
`[0.0385,0.0675]` 与 `[0.0376,0.0662]`，均覆盖 0.05。

完整 covariance whitening 会把 hard/adaptive conditional AUC 变为 `0.448/0.449`，即
adaptive 相对 calibrated hard 不下降，但校准本身有检测代价。另实现的 threshold-only
weighted-exponential 负对照过度保守：smoke 中 threshold `42.47`、经验 q95 `9.70`，
立即停止，不用于系统结果。

### 22.3 联合闭环与 frozen-planning 归因（未晋级）

adaptive 能力直接驱动联合调度时，相对 hard TP-UIC joint-robust 的三个场景：

- conditional `P_D`：`+0.0667/-0.0167/+0.0833`；
- truth weakest-target：`+0.01848/-0.00081/-0.00506`；
- truth objective：`+0.02957/-0.00730/-0.00957`。

trial 2 多选择/捕获一条链路，故又增加 frozen-planning 消融：planning certificate、mask、
fusion 与 selected links 全部使用 hard，仅 held-out residual/retention 使用 adaptive。实现
审计确认三个场景的 hard/adaptive schedule 与资源逐项相同。此时 trial 0 仍为正向，但
trial 1/2 的 `P_D` 均 `-0.0167`，truth weakest 分别 `-0.00081/-0.00506`，truth objective
分别 `-0.00730/-0.02658`，`P_FA` 均不变。失败因此属于接收机 realization 风险，不是
协作调度重排。

结论：`slack=0` 安全但完全回退，`slack=0.001` 改善 receiver P10 却不能稳定传递到系统
weakest-target/conditional detection。adaptive soft、空间 belief covariance 和拆分式
planning/evaluation 接口保留为默认关闭的实验基础设施；该软保护候选不晋级默认联合闭环。

## 23. Round 30--37 记录

### 23.1 冗余选择的共同失捕风险

原 robust fill 先最大化节点多样性，再比较 predicted `P_D`。把次序改成 `P_D` 优先后，
trial 0 为目标 1 选择互为反向的 `[3,4]/[4,3]`；两条链路受同一目标状态误差驱动并同时
失捕，捕获链路 `11→10`、weakest truth `P_D 0.8579→0`、conditional `P_D
0.8667→0.5667`。这证明边际捕获概率或反向链路数不能表示共同状态相关性。

随后构造 belief-only 共同状态 sigma points。9 点坐标轴设计在固定两链路策略的三个场景
中与原 diversity-first 逐项相同，但动态单链路证书误把 `[3,4]` 判成 9/9 覆盖。扩展到
137 点（中心、8 个轴向点、64 对确定性反向球面方向）后，线性化版本仍误判 137/137。
审计发现目标 1 的真实标准化误差半径为 3.277，略超 95% 四维半径 3.080；更重要的是，
雅可比预测 Doppler 偏差仅 1.216 bins，而精确双基地映射偏差为 8.103 bins，门宽只有
3.874 bins。故失败来自近场角度变化下的一阶 Doppler 模型，而非调度循环。

### 23.2 精确非线性边界证书与约束/效用分离

sigma points 现通过精确双基地 delay/Doppler 映射，仅 gate width 保留 covariance
Jacobian。相同 `[3,4]` 覆盖被修正为 `83/137`。100% 动态覆盖能恢复 trial 0，但为已有
多链路的目标继续加链，资源过保守。新增 `sigma_singleton`：证书只判断单链路是否脆弱，
若不足则选择一次互补链路，已有两条或更多链路时停止；有限方向集不再被误称为连续椭球
证明。

第一次跨场景实现仍把覆盖点数作为效用最大化，trial 1 选择 137/137 但检测较弱的
`[5,2]`，而不是覆盖 132/137 的 `[0,2]`，conditional `P_D 0.333→0.133`。现改为分层
规则：覆盖只作安全约束；候选达到门槛后，恢复节点多样性和 predicted `P_D` 排序；没有
候选达标时才最大化部分覆盖。探索门槛取 95% 的确定性边界方向。必须强调：这个 95% 是
离散角向覆盖容差，不是概率覆盖保证；它是在 trial 1 的失败诊断后确定的，因此 trial
0--2 只能算开发集，不能作为该门槛的独立验证集。

最终 `sigma_singleton + nonlinear map + 95%` 在 trial 0/1/2 上与原固定两链路 robust
基线的 selected/evaluated links、捕获数、conditional `P_D/P_FA`、truth weakest 和
objective 逐项一致：trial 0 为 `12/11` 条计划/捕获链路、`P_D=0.8667`、weakest
`0.8579`；trial 1 为 `14/13`、`P_D=0.3333`、weakest `0.1027`；trial 2 为
`16/16`、`P_D=0.5000`、weakest `0.1616`。三个现有场景尚未证明省链或检测增益，故该
策略保持 opt-in，不替换默认 diversity-first；但它把固定冗余启发式改造成了可审计的
共同状态风险约束，为后续未见场景验证提供了正确接口。是否能省链或提高检测必须由未参与
门槛选择的 held-out trials 决定。

### 23.3 当前 `P_D` 与 Wilson 不理想的归因

每目标只有 20 个 H1 样本，单目标经验 `P_D` 的最小跳变量为 0.05；三个目标合并后仍只有
60 个 H1。因此小于一个或两个计数的变化不能与 receiver realization 噪声区分，Wilson
区间自然较宽。困难场景的解析 truth weakest `P_D` 仅约 0.10--0.16，说明低 `P_D` 不是
Wilson 计算错误：主因仍是弱目标链路质量和 held-out TP-UIC residual realization；CFAR
修复后的 H0 `P_FA≈0.05` 且 Wilson 覆盖目标，不能通过降低阈值来无代价补偿。后续比较
继续同时报告解析 truth `P_D`、经验 conditional `P_D`、Wilson 区间和配对随机流；只有
方向一致并跨场景复现的变化才允许晋级。

### 23.4 未见 trial 3--4 审计：候选不晋级

用未参与 0.95 方向门槛选择的 trial 3--4，同时运行固定 `min_links=2 +
diversity_first` 基线和 `min_links=1 + sigma_singleton` 候选。两臂共享场景、能力测量、
held-out TP-UIC 与检测随机流。

- trial 3：候选把 joint-robust 链路由 12 减为 11，捕获链路也由 12 减为 11；经验
  `P_D=0.9667` 不变，truth objective `2.4724→2.4777`，但 weakest truth `P_D`
  `0.95345→0.95319` 略降。这是近似安全省链，但不是严格 weakest Pareto 改进。
- trial 4：候选链路/捕获数仍为 `12/11`，却把目标 1 的互补链路 `[5,3]` 换成
  `[3,4]`。边界覆盖从基线组合的 `111/137` 提升到 `137/137`，但 truth mean `P_D`
  `0.5952→0.5755`、truth objective `0.06534→0.05354`、经验 `P_D 0.6167→0.5667`，
  `P_FA 0.0383→0.0433`。方向覆盖与真实检测质量发生冲突。

曾尝试在覆盖排序前增加 belief-side predicted `P_D` 的 0.01 绝对损失合同；相同 trial 4
仍选择 `[3,4]` 且全部 truth/检测结果不变，说明当前 predicted `P_D` 也未识别该 held-out
链路质量差异。继续调节 slack 会构成对 trial 4 的过拟合，因此该试验代码已撤回。

结论：精确非线性共同状态证书是正确的风险诊断修复，但 `sigma_singleton` 调度策略在真正
held-out 场景 1/2 退化，不替换默认固定冗余。下一轮不能再只改 coverage/PD 的排序；应把
链路质量的 posterior uncertainty 与共同 capture event 联合成一个风险调整后的检测效用，
再用未见 trial 验证。
