# 方向 3（TP-UIC × 协同闭环）——首轮可行性实验

> **V2 唯一入口：** 当前研究问题、冻结场景、算法链、基线、正式实验和停止规则统一见
> [`V2_RESEARCH_FRAMEWORK.md`](V2_RESEARCH_FRAMEWORK.md)。本文件第 1--11 节仅作历史追溯；
> 第 12 节记录迁移前最后一个有效快照，不再继续向其中叠加新模块或新叙事。

> 状态说明：第 1--11 节是研究过程记录，第 12 节是迁移前快照；二者均仅用于追溯。
> 当前研究设计统一以 V2 主文件为准。已被证明口径错误的结果已删除，不得从历史段落
> 外推性能。

> 本目录只记录接收机证书、可辨识覆盖与协同选择的端到端闭环实验。
> 首轮是小样本 pilot，不作为论文性能定论，也不修改默认发布路径。

## 1. 预注册问题与判据

链条为：belief-only 证书 → selective TP-UIC → 可辨识覆盖冗余 → 协同照射反馈
→ held-out receiver 评估 → 固定虚警率下的最差目标检测概率。

成功要求：同随机场景配对后最差目标检测概率提高，P_FA 增量不超过 0.01，并同时
报告发射节点与接收机测量时间。MC=3 只检查链条和效应方向，不作显著性结论。

## 2. 实验配置

- 主口径：600 m / RCS 0.1，`paper-canonical`。
- 小规模：M=6、Q=3、M_rx=4、trial 30–32。
- 协同约束：最多 3 个发射节点，4 轮严格单调照射反馈。
- 证书：`predicted` residual + `predicted_risk` retention；held-out evaluation。
- 可辨识近似：belief 椭球边界 sigma-point 覆盖，`sigma_singleton`，阈值 0.95。
- 后处理：2 轮 nominal worst-target swap、2 轮 prior-quantile secondary swap、
  固定 sensing set 的 fusion-PD polish。
- 对照：`targeted_tpuic_full` 与 `adaptive_soft_tpuic`，同 seed、同 trial。

执行器是 `tools/run_joint_tpuic_coordination.py`；本目录不复制第二份脚本。

## 3. 首轮结果

| 接收机 / 协同臂 | P_D | P_FA | truth worst P_D | TX |
|---|---:|---:|---:|---:|
| hard serial baseline | 0.3333 | 0.0407 | 0.1114 | 6 |
| hard joint baseline | 0.4444 | 0.0630 | 0.3095 | 3 |
| hard serial + coverage | 0.4444 | 0.0333 | 0.1271 | 6 |
| hard joint + coverage | **0.5556** | 0.0704 | **0.3871** | 3 |
| soft serial baseline | 0.3333 | 0.0370 | 0.1115 | 6 |
| soft joint baseline | 0.4444 | 0.0556 | 0.2502 | 3 |
| soft serial + coverage | 0.4444 | 0.0296 | 0.1271 | 6 |
| soft joint + coverage | **0.5556** | 0.0593 | **0.3925** | 3 |

hard 闭环中，联合反馈相对串行的 ΔP_D 为 +0.1111；加入覆盖冗余后 P_D 再从
0.4444 提到 0.5556。由于 n=3，标准误同样约 0.1111，只能判定链条可执行。

adaptive-soft 相对 hard 的四个关键臂 ΔP_D 均为 0。联合覆盖臂的 truth worst P_D
仅提高 +0.0054，而联合 baseline 下降 −0.0593；接收机测量时间从约 5–11 s 增至
36–53 s。当前 soft 规则因此不具备进入大样本验证的依据。

## 4. 闭环判决

- ✅ belief-only certificate → TP-UIC → cooperation → held-out detection 已跑通。
- ✅ 可辨识覆盖冗余和联合照射反馈会改变选择与最差目标指标。
- ✅ hard/soft 单臂生产剪枝缺口已由回归测试钉住。
- ❌ 当前 adaptive-soft 没有产生可见 P_D 增益，且计算代价过高。
- ⚠️ P_FA 在 hard joint coverage 相对 serial coverage 增加约 0.037，超过 +0.01；
  soft 对应增量约 0.030，也未过门。因此当前闭环尚未达到准入条件。

下一步应先定位 soft 候选的选中率/回退率，并把可辨识性指标用于“是否请求新观测”，
而不是直接扩大 MC。只有 pilot 中出现可分辨效应且 P_FA 校准通过，才进入 MC≥53。

## 5. 数据

- `data/closure_hard_pilot/joint_tpuic_coordination.csv`
- `data/closure_soft_pilot/joint_tpuic_coordination.csv`
- `data/closure_certificate_free_pilot/joint_tpuic_coordination.csv`

## 6. 暂时移除证书：certificate-free pilot

规划侧把 receiver residual 设为统一的零残余上界、retention 设为 1，只用 belief 几何、
波形、通信与硬资源约束排序；这些 `kappa=120 dB / eta=1` 是**规划中性值**，不是实测
能力。执行与 held-out 评价仍运行真实 targeted TP-UIC。风险分位证书交换关闭。

| 臂 | ranked P_D | free P_D | ranked worst | free worst | ranked links | free links |
|---|---:|---:|---:|---:|---:|---:|
| serial baseline | 0.3333 | 0.3333 | 0.1114 | **0.1461** | 12.67 | 11.00 |
| joint baseline | 0.4444 | **0.6667** | **0.3095** | 0.2774 | 13.67 | 11.00 |
| serial combined | **0.4444** | 0.3333 | 0.1271 | **0.1461** | 12.67 | 12.00 |
| joint combined | 0.5556 | 0.5556 | **0.3871** | 0.2774 | 13.33 | 11.67 |

certificate-free 在三个 trial 中每次都选择了与 ranked 不同的链路集合，且少用约
0.7–2.7 条链路。它在离散 P_D 上没有一致劣化，joint baseline 甚至高 +0.222；但
joint combined 的连续 truth worst P_D 下降 **−0.110**，truth objective 下降 **−0.302**。
由于 P_D 只有 1/9 的分辨率，不能用 +0.222 宣称提升。当前判决是：**证书不是系统运行
的必要条件，但完全移除后，nominal 选择器没有守住 max–min 目标。** 下一步应在不恢复
逐链路证书排序的前提下，把目标 deficit、每目标服务下界和 held-out worst-PD 接受规则
做成硬约束。

## 7. 无证书约束 max–min pilot

同一 trial 30–32 进一步加入三项机制：每目标至少 2 条观测；以
`sum min(P_D,q, P_D,req)`（等价于最小化总 deficit）作为最差目标并列时的次级量；
leximin 负责产生候选、精确 `Phi=min_q P_D,q` 负责照射反馈接受。通信价格项关闭。

| 臂 | ranked worst | free worst | free max–min worst | free max–min P_D |
|---|---:|---:|---:|---:|
| serial baseline | 0.1114 | **0.1461** | 0.1460 | 0.3333 |
| joint baseline | **0.3095** | 0.2774 | 0.2552 | 0.4444 |
| serial combined | 0.1271 | **0.1461** | 0.1460 | 0.3333 |
| joint combined | **0.3871** | 0.2774 | 0.2552 | 0.4444 |

max–min 臂平均使用 10.7–12 条链路，P_FA 为 0.022–0.033。8 轮 worst-target
swap 在 12 个 TP-UIC 状态中接受 **0 次**；硬 `max_tx_nodes=3` 投影后的照射反馈也接受
0 轮。因此这次实验不是“Phi 拒绝了坏候选”，而是 nominal candidate generator 根本
没有提出可改善 Phi 的候选。判决：**K_min/deficit/Phi 三项数学规则本身不足以替代
receiver 信息；还需要不依赖证书、但能观察 TP-UIC 执行反馈的候选生成机制。**

数据：`data/closure_certificate_free_maxmin_pilot/joint_tpuic_coordination.csv`。

## 8. 目标级最小反馈 max–min pilot

为区分“完全依赖逐链路证书”和“完全看不见接收机状态”，增加一个中间臂：TP-UIC
只向规划器报告每个目标的保守残余功率与存活率，并在进入选择器前跨接收机聚合；
接收机编号被删除，因此该信号可以驱动目标服务缺口，却不能给 `(i,j,q)` 链路排序。
真值评价仍使用独立 receiver realization。其余配置与第 7 节相同。

| 臂 | free worst | target-feedback worst | ranked worst | feedback P_D | feedback P_FA |
|---|---:|---:|---:|---:|---:|
| serial baseline | **0.1460** | 0.1333 | 0.1114 | 0.3333 | 0.0370 |
| joint baseline | 0.2552 | 0.2856 | **0.3095** | **0.5556** | 0.0259 |
| serial combined | **0.1460** | 0.1333 | 0.1271 | 0.3333 | 0.0370 |
| joint combined | 0.2552 | 0.2856 | **0.3871** | 0.4444 | 0.0333 |

目标级反馈使 joint baseline 的连续最差目标指标相对完全无反馈提高 `+0.0304`，离
逐接收机-目标 ranked 方案还差 `-0.0239`；joint combined 相对 ranked 仍差
`-0.1015`。串行状态反而下降 `-0.0127`，且平均链路数从约 11–12 增至 14–17。
8 轮 polish 在串行状态共接受 6 个 swap，但联合状态仍是照射 mask fixed point、接受
0 轮。判决：**低维目标反馈能恢复一部分闭环增益，但无法定位是哪台接收机造成损失，
也没有联合调节 TP-UIC 保护强度，因此尚不符合替代证书的性能预期。** 下一步应加入
接收机级拥塞/残余标量或直接把软保护乘子作为优化变量，同时保持 held-out 评价隔离。

数据：`data/closure_target_feedback_maxmin_pilot/joint_tpuic_coordination.csv`。

## 9. TP-UIC 控制变量联合优化 V2 pilot

进一步把 TP-UIC 从固定模块改成外层可控模块：对每个照射状态分别计算
`mu in {0.1, 1, 10}` 的 soft TP-UIC、目标级反馈、链路选择与融合，再按照
`(min P_D,q, sum min(P_D,q,P_D,req), -mu)` 选择控制状态。每个候选的最终指标仍
来自独立 receiver realization，默认发布路径不变。

| 臂 | 固定 target-feedback worst | V2 controlled-soft worst | V2 P_D | V2 P_FA |
|---|---:|---:|---:|---:|
| serial baseline | 0.1333 | 0.1333 | 0.3333 | 0.0333 |
| joint baseline | 0.2856 | **0.2861** | 0.5556 | 0.0259 |
| serial combined | 0.1333 | 0.1333 | 0.3333 | 0.0333 |
| joint combined | 0.2856 | **0.2861** | 0.4444 | 0.0333 |

所有 24 个 TP-UIC 状态都选择网格下界 `mu=0.1`，但 joint worst 只提高
`+0.0005`，离 ranked combined 的 `0.3871` 仍差约 `0.101`。三 trial 总运行时间
约 603 s，是固定 target-feedback 的约 8 倍。判决：**把一个全局软保护乘子加入
外层优化仍不够；控制维度必须至少细化为 receiver-target，并为残余干扰下降与目标
自消损伤设置显式预算。** 下一版不应盲目扩大 mu 网格，而应做块坐标更新：只对当前
最差目标及其主要接收机更新 `mu_jq`，其余单元保持硬保护，并缓存共享线性系统。

数据：`data/closure_joint_soft_control_pilot/joint_tpuic_coordination.csv`。

## 10. Receiver-target 状态与逐单元控制 V3

V3 将反馈粒度恢复到 receiver-target，但不使用真值规划：每个 `(j,q)` 只报告信念侧
预测残余与风险存活率。首先用固定 soft TP-UIC（默认 `mu=1`）跑同一组 trial 30–32；
另做 trial 30 的逐单元 `mu_jq in {0.1,1,10}` 机制筛查，按信念侧
`log(eta_jq)-log(Ires_jq/Iin_j)` 独立选择控制量，最终仍 held-out 评价。

| 方案 | joint worst | joint P_D | joint P_FA | links |
|---|---:|---:|---:|---:|
| 无反馈 max–min | 0.2552 | 0.4444 | 0.0333 | 12.0 |
| 目标级聚合反馈 | 0.2856 | 0.4444 | 0.0333 | 14.0 |
| receiver-target 状态，soft `mu=1` | **0.3750** | **0.5556** | **0.0444** | 14.0 |
| ranked hard combined | 0.3871 | 0.5556 | 0.0704 | 13.3 |

receiver-target 状态相对目标级聚合反馈提高 `+0.0894`，已达到 ranked combined 的
96.9%，同时 P_FA 低 `0.0260`。但逐单元控制筛查没有验证控制律：trial 30 的 18 个
单元全部选择 `mu=0.1`，held-out worst 为 `0.6607`；同一 trial 固定 `mu=1` 时反而为
`0.8628`。因此当前最主要的结构问题是**信念侧局部 eta/residual 比与端到端 held-out
Phi 不一致**，而不是缺少更细的控制维度。下一步应以独立历史 CPI 或置信下界校准
`mu_jq` 的动作价值，并加入 trust region；不能直接用局部代理贪心选择 mu。

数据：

- `data/closure_receiver_target_state_pilot/joint_tpuic_coordination.csv`
- `data/closure_cell_soft_control_screen/joint_tpuic_coordination.csv`

## 11. Trial 32 可行性、资源与统一 PFA 审计

不再把 3 个发射节点视为固定条件。保持 `M=6,Q=3`，对激活 TX 数 1–6 扫描，正式
评价统一使用 `P_FA*=0.05`、exact Gaussian threshold、每目标 1000 个 H0 和 200 个
H1。soft TP-UIC、单 CPI、原处理增益下的最佳点是 2 TX，truth worst 仅 `0.1483`；
4/5/6 TX 分别为 `0.0585/0.0848/0.0920`，说明简单增加照射节点会增加直达干扰，
且当前选择器在无 TX 上限时会占满全部节点并进入 fixed point。

truth-geometry oracle、2 TX、每目标最多 6 条链路时 worst 为 `0.1450`；把上限放宽到
10 条、共 30 条观测后也只有 `0.1611`。逐目标 truth P_D 约为
`[0.275,0.789,0.161]`，目标 2 是瓶颈。perfect-channel（零残余、eta=1）上界也只有
`0.2954`，因此原单 CPI 信息预算在 trial 32 上不可能达到 0.8，主因不是 belief
排序或 TP-UIC。

保持 UAV/目标数不变，把有效 sensing processing gain 提高 4 倍后，perfect-channel
worst 达 `0.9530`，证明需求物理可达；真实 soft TP-UIC 在 4 倍时为 `0.7443`，6 倍时
达到 truth worst `0.9240`、经验 P_D `0.9433`、P_FA `0.0470`。但 6 倍时 belief
worst 仍只有 `0.3026`，所以这是**实现性能达标点**，还不是 belief-robust 保证达标点。
后续主问题应把处理/积累预算作为可行性资源，先满足 0.8，再最小化所需预算；不能把
TX=3 写死，也不能用小样本 P_FA 判定准入。

数据目录：`trial32_*_audit`、`trial32_*_gain*_audit`。

### 单帧最终上界核对

为决定是否进入多帧，补做原始处理增益下的最有利单帧审计：truth geometry、
perfect channel、全部 6 个 TX、每目标/总链路上限分别放宽到 30/90。可行性过滤后
共有 54 条链路且全部被选中，逐目标 P_D 为
`[0.7866, 0.8218, 0.5025]`，连续 worst `0.5025`、经验 P_D `0.6933`、P_FA
`0.0477`。因此在当前单帧波形、功率与可行链路集合内，即使完美接收机也达不到
worst P_D=0.8；多帧/积累不再只是调参选项，而是满足 trial 32 需求所必需的资源维度。
该结论是当前模型可行集上的工程上界，不宣称是未建模物理系统的数学绝对上界。

数据：`data/trial32_single_frame_all_links_upper_bound/`。
## 12. 当前有效主线（2026-09-22）

### 12.1 固定模型与评价口径

当前只采用 `paper-canonical`、600 m、RCS 0.1、M=6、Q=3、`m_rx=4`、
Swerling-II matrix covariance LLR。单个 OTFS 帧时长为

``T_frame = N*T = 64/30 kHz = 2.133 ms``。

实验必须显式声明感知驻留时间，并由
``n_looks=floor(T_dwell/T_frame)`` 派生 looks。`n_looks=16` 仅是34.133 ms
参考点，不再作为无来源的处理增益。

网络目标为 ``I_q=sum_j I_jq`` 后最大化 ``min_q I_q``，不是要求每架 UAV
独立探测所有目标。中等视角定义为声明驻留时间下单接收端 `P_D>=0.3`，覆盖约束
按每个目标分别计算，禁止用全局平均掩盖弱目标。

### 12.2 当前可靠结果

固定阵形基线在16 looks下的网络 `P_D` 为 `[1.000, 0.375, 0.114]`，中等视角
数量为 `[6, 0, 0]`。固定总能量时，单纯增加 looks 不能解决瓶颈。

采用真值辅助目标环阵、最小距离 UAV--锚点匹配和每目标两个设计视角后：

| 单机移动预算 | 中等视角数 | 16-look worst P_D |
|---:|---:|---:|
| 100 m | `[6,1,0]` | 0.141 |
| 200 m | `[6,4,0]` | 0.193 |
| 300 m | `[6,4,0]` | 0.321 |
| 400 m | `[6,4,2]` | 0.738 |
| 无约束上限 | `[6,4,5]` | 0.979 |

400 m点首次满足最差目标33.3%中等视角覆盖，20 looks（42.67 ms）跨过0.8。
8 looks时覆盖退化为 `[6,2,1]`，网络 `P_D=[1.000,0.771,0.525]`，尚未达标。
这些结果是结构可行性上限，不是现实在线控制器结果。

### 12.3 TP-UIC的实际贡献

400 m同阵形、16 looks消融：

| 接收处理 | worst P_D | 达到0.8所需looks |
|---|---:|---:|
| no IC | 0.737859 | 20 |
| TP-UIC Stage 1 | 0.738018 | 20 |
| TP-UIC full | 0.738023 | 20 |
| perfect channel subtraction | 0.737993 | 20 |

因此当前性能提升几乎全部来自阵形；直达干扰对消不是活动瓶颈。`no_ic`仍是
协方差感知 GLRT，会白化直达先验，并把其他目标放入 nuisance covariance，故
TP-UIC是在强统计干扰处理上的小修正。Full还会使目标2从Stage 1的0.94784降至
0.93757，后续必须采用逐目标收益门控，不能默认总是启用Full。

### 12.4 保留数据与下一步

当前有效数据为：

* `matrix_stochastic_active_trial32`：固定阵形基线；
* `matrix_formation_b100/b200/b300/b400_matched_trial32`：移动预算扫描；
* `matrix_formation_b400_looks8_trial32`：低looks审计；
* `matrix_formation_b400_no_ic/tp_uic_stage1/perfect_channel_trial32`：对消消融；
* `matrix_formation_free_matched_trial32`：无约束结构上限。

下一步只推进三项：belief驱动滚动阵形、通信可靠率/容量接入融合、TP-UIC压力
场景与收益门控。旧deterministic-GLRT的0.056/2229帧及0.060/885帧结果已删除，
未做最小距离匹配的早期阵形结果也已删除，不再用于任何结论。

### 12.5 通信协同接入

矩阵主线现已支持局部充分统计量/LLR包上报。每个目标选择显式融合UAV，接收端
证据按通信包成功率做精确独立擦除混合，而非默认必达。400 m、16-look首个结果
选择UAV 3作为三个目标的融合节点；当前链路速率0.96--2.90 Mbps，旧启发式可靠
率饱和为1。融合节点仅使用本地观测时目标2/3 PD为0.365/0.466，接收其余UAV的
LLR后达到0.938/0.738，协同增益分别为0.573/0.272。下一步必须用有限块长可靠
率、容量和时延压力测试替代饱和启发式，并区分belief、LLR与压缩复回波三档载荷。
