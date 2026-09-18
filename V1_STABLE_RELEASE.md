> **归档提示（2026-09-18）**：本文引用的部分 `results_*` 产物已移入 `_archive/2026-09-18/`；正文中的路径引用已同步更新为归档位置，命令行示例里的 `--out` 目录仍写作历史原名（重跑时依旧输出到该名）。

# V1 稳定版本：理论 / 模型 / 算法三层冻结契约

> 建立时间：2026-09-16 ｜ **本次修订：2026-09-18（rev. 2）**
> 适用对象：ICC 2027 会议稿（`target-local-v1` 发布线）
> 可执行清单：`release/V1_STABLE_MANIFEST.json`（由 `tools/check_release_identity.py` 生成与校验）

**rev. 2 改了什么**（数字未动，结构补全）：

1. **算法层补上了机间协调**：`coordination.py` + `Coordination(enable, rounds)` 已接进发布入口
   （§2.3）。它是本次冻结里唯一的**结构新增**，默认关闭，默认路径仍逐位一致。
2. **`coordination.enable` / `coordination.rounds` 进冻结层**（94 → **96** 键）。
3. **补了"发布默认路径"的逐位门禁**（门禁 2b）：原先只有 legacy 口径的逐位检查，
   而它**不读 κ**——也就是说 κ、门控、协调默认值这三样全在覆盖面之外。
   现在 `target-local-v1` @ mc=12/seed=2026 有独立基线（1653 格），
   并用 κ=40→30 实测确认它是**活守卫**（959 格变化、958 格超容差）。
4. **门禁 2 被证明是"环境绑定"的**，契约补记运行时（§5.6），`parity_check` 新增
   `--allow-env-noise` 以把"解释器差异"与"模型漂移"分开。
5. **κ 的辨析与推导落库**（`KAPPA_DERIVATION.md`），§5.2 从"待澄清"改为已收口，假设账本见 §2.4。
6. **归档后失效的"归档主结果 config 交叉核对"已修复**（恢复 110 键比对）。
7. **三层对应表的行号现在有守卫**（`tools/check_contract_refs.py`，32 条引用）——
   建立时立刻抓到 **5 处已漂移的行号**，测试套件对此全绿。

---

## 0. 版本标识

| 项 | 值 |
|---|---|
| 发布线 | **V1**（rev. 2，2026-09-18） |
| 主口径预设 | `target-local-v1`（= `paper-canonical` + `fusion.rule="nearest_target"`） |
| 预设注册表 | `isac_sim/config.py:809+`，`HEADLINE_RELEASE_PRESET`（`config.py:970`） |
| 干扰/时序口径 | `comm.interference_model = "orthogonal"`（感知并发 + 报告正交串行） |
| 冻结键 | **96** 个（见 §3 分类）；可标定 **18** 个 |
| 默认方法名册 | **19** 个；实验方法 **9** 个；RNG offset 表 **28** 条 |
| 回归基线（两条） | legacy/重构口径：`.workbuddy/baseline/main/main.csv`，md5 `659a07d4ce262d311303c5eb90d8b17e`<br>**发布口径**：`.workbuddy/baseline_release/main/main.csv`，md5 `417d016b912f67558750610edfc8d42e` |
| 清单 md5 | `dac23433bd078485ab761fca8d889a77` |
| **契约运行时** | **CPython 3.11.0 / numpy 2.2.6**（逐位门禁只在此运行时下有效，见 §5.6） |
| 包版本串 | `isac_sim.__version__ = "1.6.0"` ⚠️ 见 §5.4（该串属于机制轨，不代表本发布线） |

**验收门禁（5 条，全绿才算"稳定"；全部在契约运行时下执行）**：

```bash
# 门禁 1 —— 三层结构未漂移
python tools/check_release_identity.py --check
#   → 96 frozen / 0 violated / CLEAN；归档主结果交叉核对 110 键

# 门禁 2 —— 重构口径逐位一致（钉 legacy coupling + legacy eps_mode）
python tools/parity_check.py --baseline --strict
#   → runtime py3.11.0 / numpy 2.2.6；1596 cells / 0 differing / EXIT 0

# 门禁 2b —— 【rev.2 新增】发布默认路径逐位一致（preset target-local-v1）
python tools/parity_check.py --release-baseline --strict
#   → 1653 cells / 0 differing / EXIT 0
#   这道才是"默认路径 bit-exact"的真正守卫：κ / 门控 / 协调默认值都在它的覆盖面内，
#   而门禁 2 的 legacy 口径**根本不读 κ**（§5.6）

# 门禁 3 —— 行为回归
python -m pytest tests/ -q
#   → 180 passed, 6 subtests passed

# 门禁 4 —— 【rev.2 新增】三层对应表的行号守卫
python tools/check_contract_refs.py
#   → 32 references / CLEAN

# 非契约运行时下跑逐位门禁，必须显式标明这是环境噪声而不是模型漂移：
python tools/parity_check.py --baseline --strict --allow-env-noise
#   → py3.13/numpy 2.5.2：1596 cells / 25 tiny(<1e-9) / CLEAN(env-noise) / EXIT 0
```

---

## 1. "稳定"的定义

**稳定 ≠ 数字不动。** 本发布线的"稳定"指下面四条同时成立：

1. **默认路径 bit-exact**：`target-local-v1` 的解析结果相对**发布口径基线**逐位一致
   （`parity_check --release-baseline --strict`；重构口径另由 `--baseline --strict` 守卫）。
   两条都**必须在 §5.6 记录的契约运行时下执行**——换个 numpy 就会多出 25 个 ≤1e-12 的
   单元格差异，那是解释器算术，不是模型。
2. **三层对应关系有据可查**：论文里出现的每个方程，都能指向唯一一处代码实现；代码里每个改变
   物理/判决语义的开关，都能指向论文的哪一句话（§2）。
3. **结构项与工作点项分离**：改变"系统是什么"的键被冻结；改变"系统部署在哪"的键允许移动，
   且移动必须走 §7 的登记流程（§3）。
4. **未收口项被显式列出**：没有"静默未知"（§5）。

> 之所以要第 3 条：RCS 工作点仍在标定中（`detect.target_rcs` 可能从 50 m² 改），
> 如果把它和物理结构一起冻结，门禁会天天红；如果两者都不冻，就无从判断"性能变了"是
> 换了场景还是改坏了模型。

---

## 2. 三层对应契约

### 2.1 理论层：论文方程 → 代码实现

| 论文位置 | 方程内容 | 代码实现 | 状态 |
|---|---|---|---|
| `SystemModel.tex` `eq:comm_sinr_rate` L33-42 | 上报链路 SINR 与速率 | `model.py:685-705`（`gamma_comm` / `rate`） | ✅ |
| `SystemModel.tex` `eq:fbl_reliability` L49-54 | 有限码长包成功率 $\chi_{ab}=1-\mathsf Q(\cdot)$ | `fbl.py:134` `chi_from_gamma` | ✅ |
| `SystemModel.tex` `eq:bistatic_delay_doppler` L70-77 | 双站时延/多普勒 | `model.py:350-410`（`delay_bin` / `doppler_bin`）；`dd.py` | ✅ |
| `SystemModel.tex` `eq:local_dd_energy` L82-90 | 归一化有限网格 DD 泄漏 $\eta^{\rm loc}$，$W=1$ | `dd.py`（`eta_local` / `eta_fine_array`） | ✅ |
| `SystemModel.tex` `eq:sensing_sinr` L103-109 | $\gamma^s=P^s g^s G^{\rm hw}G_{\rm p}\eta^{\rm DD}\eta^{\rm col}/(N_0+I^s)$ | `model.py:800-849`（`gamma_sense`） | ✅ |
| `SystemModel.tex` L110-111（**只有一句话，没有方程**） | $I^s$ = 自残留 + 并发直射路径 | `model.py:651,654,765`：$I^{\rm dir}_j=\Gamma_{\rm dp}\sum_i P^{\rm rad}_i g^{\rm dir}_{ij}$，$\Gamma_{\rm dp}=10^{-\kappa/10}$ | ⚠️ 论文侧缺定义与方程 ⇒ §2.4、`KAPPA_DERIVATION.md` §6.1 已备好可粘贴段落 |
| `SystemModel.tex` `eq:local_llr` L119-126 | $\ell_e=\frac{\gamma}{1+\gamma}(X-K)$，$\delta,v_0,v_1$ | `llr.py:83-104` | ✅ |
| `SystemModel.tex` `eq:received_moments` L131-139 | 混合后 $\delta^{\rm eff},\bar v_0,\bar v_1$（组间项在 **H1**） | `soft_channel.py:44-87`（`_mix` + `received_moments`） | ✅ |
| `SystemModel.tex` `eq:detector_prediction` L149-155 | $t_q=\sqrt{v_{q,0}}[z_0+\frac{\kappa_0}{6}(z_0^2-1)]$，$\widehat P_D=\mathsf Q(\cdot)$ | `fusion.py:195-249` `predicted_pd_for_links` | ✅ |
| `ProposedMethod.tex` `eq:nearest_target_fusion` L7-11 | $f_q=\arg\min_m\|\widehat{\mathbf p}_q-\mathbf p_m\|$ | `reporting.py:145-250` `assign_fusion_nodes`（`rule="nearest_target"`） | ✅ |
| `ProposedMethod.tex` `eq:feasible_candidate_set` L15-21 | $\mathcal E_q$ 可行性集合 | `selection.py:162-190` `feasible_links_for_target` | ✅ |
| `ProposedMethod.tex` 局部投递界 L23-28 | $d(\chi)=\chi^2\delta^2/[v_0(a+(1-a)\chi)]$ | **无独立实现**（由 `eq:received_moments` + `_mix` 代入即得；`selection.py` 的局部投递走 `link_cost_ms=0`） | ✅ 解析结论 |
| `ProposedMethod.tex` 报告数下界 L36-41 | $N_{{\rm rem},q}\ge|\mathcal S_q|-\max_m a_{qm}$ | **无独立实现**（结构性事实，`report_dest` 决定谁免包） | ✅ 解析结论 |
| `ProposedMethod.tex` `eq:fair_sensing_utility` L48-56 | 软最小 + 二次缺口效用 | `fusion.py:435-486` `selection_utility_from_pd` / `selection_utility` | ✅ |
| `ProposedMethod.tex` `eq:task_objective` L60-69 | $\max F=U-\lambda_c\sum c_e$，$c_e$ 以 **ms** 计 | `selection.py`（`link_cost_ms = 1e3·link_delay_s`）；贪心 commit 在 `_greedy_lagrangian`（`selection.py:329`） | ✅ |
| `ProposedMethod.tex` `eq:c2f_complexity` L94-98 | $EC_{\rm c}+rC_{\rm f}$ | `selection.py:653` `select_c2f_adaptive`，`stats.fine_eval_full/c2f` | ✅ |
| `ReproducibilitySupplement.tex` L109-124 | 全期望/全方差恒等式 | `soft_channel.py:44-51` `_mix` | ✅（`tools/check_variance_placement.py` 恒等式残差 ≤4.8e-16） |

> 行号由 `tools/check_contract_refs.py` 逐条守卫（32 条引用、必须全部解析）。
> 该工具在 2026-09-18 建立时**立刻抓到 5 处已漂移的引用**——`model.py` 一次 30 行插入
> 就足以让下面每个行号失效，而测试套件不会响。

**理论层两条已定案的边界**（不要再翻案）：

- **组间项 $\chi(1-\chi)\delta^2$ 属于 H1 方差，不属于偏转分母。** 论文
  `eq:received_moments`（H1 行）与 `ProposedMethod.tex` 的 $d(\chi)$ 分母 $=v_0(a+(1-a)\chi)$
  同口径；旧代码（`603b61d`）把组间项塞进偏转分母，**与新旧两版论文都不一致**。
  2026-09-16 判定为「代码对齐论文」，**不回退**。数值复核见
  `tools/check_variance_placement.py`。
- **`isac-consistent` 的 `active_set` 是消融，不是论文口径。** 论文口径是 `orthogonal`。

### 2.2 模型层：物理与干扰记账

| 模型要素 | 冻结值 | 代码 | 论文出处 |
|---|---|---|---|
| 频谱耦合 | `interference.coupling = "shared_spectrum"` | `model.py:560-620` | `SystemModel.tex` L28-45 |
| 残留干扰三项 | `residual_self=1e-14`、`residual_multi_uav=1e-10`，走共享干扰场；直射对消由 `kappa_dc=I_sense_field·10^{-κ/10}` 控制 | `model.py:755-790`（`residual_direct` / `residual_multi`） | L110-111 |
| 直射对消深度 | `interference.direct_cancellation_db = 40.0`（**已收口**，见 §5.2；推导见 `KAPPA_DERIVATION.md`） | `model.py:651,654,765`；`config.py:286+` | ⚠️ 论文侧无方程 ⇒ §2.4 |
| 照明机调度门控 | `sense_gate_by_active_tx = False`（默认关闭 ⇒ $I^s$ 是几何常数）；开启时 `gate_echo` 同时作用于**干扰**与**观测**两条路径 | `model.py:624-650`（`gate_echo`）、`model.py:791`（静音机 `effective_sensing_power=0`） | — |
| 机间协调（新增） | `coordination.enable = False`、`coordination.rounds = 6` | `config.py:706-745`；`coordination.py`；`simulate.py:617-735` | ⚠️ 论文侧未写 ⇒ §2.4 |
| SINR 分母护栏 | `radio.eps_mode="noise_relative"`，`eps_rel_db=-30` | `model.py:57-70` `denominator_guard` | — |
| 上报干扰记账 | `comm.interference_model="orthogonal"` | `model.py:685-705` | L33-45 |
| MAC / 可靠度 / 时延 | `mac_model="serial"`、`reliability_model="fbl"`、`latency_model="blocklength"` | `fbl.py` | L47-64 |
| 硬件增益 | `radar_*_gain = 0`（`G_hw ≡ 1`）；RCS 保持 m² 量纲 | `model.py:41-54` `radar_hardware_gain` | `eq:sensing_sinr` 的 $G^{\rm hw}$ |

### 2.3 算法层：选择与融合

| 算法部件 | 冻结设定 | 代码 |
|---|---|---|
| 融合位置 | `fusion.rule="nearest_target"`，`fusion.mode="explicit"` | `reporting.py:145` |
| 选择器打分 | `selector.score_mode="exact_utility"` | `selection.py` |
| 目标函数 | 软最小 ($\tau_\alpha=0.10$, `use_softmin_alpha=True`) + 二次缺口 ($\mu_\alpha=1.0$)，权重 $\alpha\in[0,10]$ | `fusion.py:486` `target_alpha` |
| 通信价格 | `lambda_c=0.005`，`use_delay_price=True`，$c_e$ 单位 ms | `selection.py:158-160` |
| 规模约束 | `max_links_per_target=6`、`max_total_links=60`、`candidate_topk_per_target=40` | `selection.py` |
| C2F 加速 | 每目标 shortlist $J=25$（`refine.shortlist_size`），窗口 `W=1` | `selection.py:653` `select_c2f_adaptive` |
| 停止准则 | `stop_at_D_min=False`（不因 $D_{\min}$ 提前 break） | `selection.py` |
| **照明机上限/价格**（rev.2） | `selector.max_tx_nodes=None`、`selector.tx_penalty=0.0`（**free**，不改物理） | `selection.py:329` `_greedy_lagrangian` |
| **机间协调**（rev.2） | `coordination.enable=False`（默认）；开启时走"mask → 重建表 → 重选"不动点，最多 `rounds` 轮 | `simulate.py:627` `_coordinated_c2f_selection`、`coordination.py:187` `select_with_coordination` |
| 协调的评估口径 | 最终调度**用自身照明机集合的掩码**重建评估表（含 belief 真值侧），非收敛运行也按物理真值打分 | `simulate.py:710` `_coordination_eval_tables`、`simulate.py:994-1003` |
| 最优性宣称 | **无**（不宣称全局最优、不宣称子模近似比、不宣称 C2F 无损） | `CONFERENCE_PAPER_CONVERGENCE.md` |

**协调的三种口径，代码里只认一种**（rev.2 新增，别再混用）：

| 口径 | 入口 | 是否可进主结果表 |
|---|---|---|
| `coordination.rounds` 预算耗尽 | 同上 | ✅ 但必须同时报 `coordination_converged_rate`（实测 2–4 轮、收敛率 1.00） |
| 出现重复 mask（判环） | `simulate.py:694-703` | ⚠️ 必须报为"未收敛"，不许把剩余预算说成"多跑几轮有用" |
| `tools/select_with_coordination` 直调 | `tools/*` | ❌ 与发布口径不同，**不可与 V1 数字并列引用** |

**判决阶段（检测）的实现事实**（论文未单列小节，但必须记住）：

- `simulate.py:208` 无条件调用 `calibrated_fused_threshold` 作为**所有方法共用**的判决门限。
- 在 `comm_error_model != "erasure"` 且 `corr.enable=False` 时，它**精确退化为**
  `(z_0+\frac{\kappa_0}{6}(z_0^2-1))\sqrt{v_{q,0}}`（`fusion.py:347-356`），即归档 V1 的
  Cornish–Fisher 门限。
- 因此**归档 V1 的 `gaussian_replacement` 路径上，09-15 的门限替换不改变任何数值**（见 §5.1）。

---

### 2.4 假设账本：前提 / 协议 / 贡献（rev.2 新增）

三层对应表回答了"代码在哪"，这一节回答**"换了它算不算改论文"**。判据只有一个：
它是**系统是什么**（前提/协议），还是**我们做了什么**（贡献）。

| 项 | 性质 | 冻结键 | 换了意味着什么 |
|---|---|---|---|
| 直连对消深度 $\Gamma_{\rm dp}=10^{-\kappa/10}$ | **前提**——"已存在一个深度为 κ 的对消器"，**不是**本文算法 | `interference.direct_cancellation_db = 40.0` | 要在论文里重述对消能力论证并重算全套数字。它是感知能否工作的**可行性前提**：κ=0 时实测 P_D 0.0505 vs P_FA 0.0495（＝随机），协调也救不回来（0.0530） |
| 照明机门控 | **协议假设**——非照明机在感知期间静默 | `interference.sense_gate_by_active_tx = False` | 要写进协议章节并交代信令；它是协调的**必要前提**（门控关时 `compute_link_tables` 直接忽略掩码） |
| 硬件增益 $G^{\rm hw}$ | **前提**——本发布线取 **0 dB**（不假设任何雷达链路增益） | `radio.radar_net_gain_db = None`（⇒ 1） | 提升它＝引入硬件假设。15 dB 只是 sweep 的对照臂，不在发布值里 |
| 链路选择 / C2F / 融合位置 | **本文贡献** | `selector.*` / `fusion.*` | 这正是被检验的对象 |
| 照明机上限与价格 | **调参**（free，只重排同一候选集，不改物理） | `selector.max_tx_nodes` / `selector.tx_penalty` | 不算新假设；但报协调数字时必须一并给出取值 |
| 协调不动点 | **协议 + 贡献的混合体** | `coordination.enable` / `coordination.rounds` | 机制本身是贡献，"非照明机静默"是它吃掉的协议前提 —— 两者必须分开声明，否则等于把协议当算法卖 |

**为什么这张表必须存在（实测依据）**：600 m / RCS 0.1、$G^{\rm hw}=0$、MC=200 下，
P_D 从 **0.0505**（κ=0，等于虚警率）到 **0.6830**（κ=40）这段**全部由前提决定**；
算法（协调 + cap3）在这之上再叠 **+0.1725**，即总提升里约 **79% 来自那个常数、21% 来自算法**。
所以"方法性能"若只报绝对 P_D，报的其实是 κ。可站得住的三种强度：

| 声明 | 依赖 κ | 判定 |
|---|---|---|
| 绝对 P_D（如 0.7169） | 完全依赖 | 弱 —— 正文必须同时给出 κ 和它标定的几何 |
| 固定 κ 下相对基线的优势 | 中等（幅度变、方向多数保持） | 可发，但要附 κ 敏感性 |
| "协调只在干扰受限区有效；κ 压到噪声地板后归零" | **不依赖** | 强 —— 这是机制结论，不是断言 |

把第一行从"自由常数"变成"可引用量"的推导已落库（`KAPPA_DERIVATION.md`）：

$$\Gamma_{\rm dp}=\min\bigl(10\log_{10}(N_p\gamma_p),\; \Gamma_{\rm hw}\bigr)$$

即 $\Gamma_{\rm dp}=40$ dB ⇔ $N_p\gamma_p=10^{4}$（≈100 个参考元素 @20 dB，占 4096 帧的 2.4%）；
主场景需要的 52.25 dB 由整帧参考 @16.1 dB/元素即可满足。**两条必须一起声明的建模张力**：
(a) `kappa_dc * (P_rad @ direct_gain)` 是**单标量乘聚合场**，对所有照明机等深对消、逐源不可分辨；
(b) 残余被当无结构功率，忽略确定性偏置。升级到逐源 $\Gamma[i]$ 是唯一能把"对消"从前提写成
算法贡献的做法，**本版未做**。

---

## 3. 冻结项 vs 可标定项

| 类别 | 内容 | 门禁行为 |
|---|---|---|
| **冻结**（**96** 键） | 波形网格、功率切分、噪声与护栏、残留干扰系数、上报/干扰记账、检测器全部统计量、信念/调度、选择器全部权重与约束、融合位置、**直射对消深度 κ、照明机门控、机间协调开关与轮数预算** | **不匹配 = FAIL** |
| **可标定**（**18** 键） | `scale.M/Q`、`geometry.*`、`detect.target_rcs`、`waveform_impairments.enable`、`run.num_mc/seed`、`selector.max_tx_nodes`、`selector.tx_penalty` | **不匹配 = NOTE**（且必须写进 §5.3 的登记表） |
| **算法名册** | `DEFAULT_METHODS` 顺序（19）+ `EXPERIMENTAL_METHODS` 集合（9）+ `METHOD_RNG_OFFSETS`（28 条） | **不匹配 = FAIL**（RNG offset 决定 bit-exact） |
| **契约运行时** | CPython / numpy 版本 | **不匹配 = NOTE**，但必须与 `parity_check` 的判定一起读（§5.6） |

> **rev.2 为什么把 `coordination.*` 放进冻结层**：它的 dataclass 默认值是 `False` / `6`，
> 也就是说"默认路径不动"这件事**靠默认值维持**，而默认值改起来没有任何测试会响。
> `detect.comm_error_model` 就是同一个坑的现成教训（见 §5.5 与 `rerun_target_local_v1.py`
> 入口的语义断言）。`rounds` 一起冻结，理由同 `refine.shortlist_size`：它是决定调度结果的预算。

> `run.num_mc` 属可标定项是个真实陷阱：主结果用 MC=1000，而 `Run` 的**默认是 200**。
> 也就是说"预设本身"不能复现主实验，MC 规模由运行入口提供。门禁现在会把它打成 NOTE
> 而不是静默通过。

---

## 4. 默认方法名册（19）

`proposed_c2f`、`proposed_c2f_adaptive`、`proposed_c2f_adaptive_pd`、`proposed_c2f_pd`、
`proposed_c2f_full`、`proposed_c2f_full_pd`、`all_neighbor`、`random`、`nearest`、
`shortest_bistatic`、`raw_sense_sinr`、`sense_sinr`、`sense_sinr_budgeted`、`single_best`、
`topk_deflection`、`global_topk_deflection`、`cost_aware_greedy`、`exact_marginal_greedy`、
`proposed_lagrangian`

**实验方法（9，不进默认参加对比）**：`rcs_robust_bundle_cg`、`joint_bundle_cg`、
`joint_bundle_cg_exact_llr`、`fixed_fusion_bundle`、`local_only_bundle`、
`proposed_c2f_adaptive_pd_{distributed,robust,calibrated,fusion_polish}`

**主结果引用的是 `proposed_c2f_adaptive_pd`**（不是带 `fusion_polish` 后缀的候选）。

**协调可作用的方法集（rev.2，`simulate.py:617` `COORDINATION_C2F_METHODS`，6 个）**：
`proposed_c2f_adaptive`、`proposed_c2f_adaptive_pd` 及其 4 个后缀变体
（`_distributed` / `_robust` / `_calibrated` / `_fusion_polish`）。
刻意**只包含**选择器是 `select_c2f_adaptive` 的方法——只有它们有可重跑的边际规则；
对固定排序的基线"做协调"不是同一个机制，所以不列进来。

**协调诊断列（rev.2，进 `trials.csv` 与 `main.csv`）**：
`coordination_rounds` / `coordination_converged` / `coordination_n_tx`（关闭时全为 0）。
报协调数字必须同时报是否真收敛。

---

## 5. 已知偏离（本契约不宣称已解决）

### 5.1 ✅ 主结果数字已与当前代码对齐（2026-09-16 晚重跑收口）

**根因**：检测阶段的随机数流从**共享顺序流**换成了**逐链路键控流**。

代码级证据（逐字比对 `03f9612^` 与当前 HEAD）：

| 部件 | 位置 | 两版关系 |
|---|---|---|
| `threshold_from_pfa` | `model.py:112` | **逐字相同** |
| `fused_h0_variance` | `fusion.py:252` | **逐字相同** |
| `fused_h0_skewness` | `fusion.py:281` | 仅把 `corr.enable or` 改成 `corr.enable and len(links)>1`；`corr.enable=False` 时行为相同 |
| 门限公式 | 旧 `simulate.py:158-159` vs 新 `fusion.py:350` | **同式**：$(z_0+\frac{\kappa_0}{6}(z_0^2-1))\sqrt{v_{q,0}}$ |

⇒ 在 `gaussian_replacement` 路径上（= 归档 V1 的等价路径），新门限函数**精确返回旧公式**。

**真正的变化**：

```
旧（03f9612^，simulate.py:164）      F += w * draw_h1_soft_stat(cfg, tables, link, q, rng, plan)
新（当前 HEAD，simulate.py:209）     h1_rngs = keyed_rngs(q, ordered_links, True, 0)   # 每 (trial,q,link,h,draw) 独立流
```

`git show 03f9612^:isac_sim/simulate.py | grep -c keyed_rngs` = **0**。这解释了
`BASELINE_DRIFT_ATTRIBUTION.md` 里"机制 B（63 格）触发点未隔离"——**触发点就是它**。

**实测漂移（MC=1000，已收口）**：仅 **+0.0002**（0.9762 → 0.9764），三个方法均在 ±0.0003 内；
早前 MC=100 测得的 −0.007 本身落在 MC=100 的噪声内（$P_D$ 半宽约 ±0.009），**高估了量级**。
配对差值符号与显著性方向未变（$+0.0356$ 显著、$-0.0020$ 跨零）。

**验证**：`tools/report_v1_rerun_drift.py` 逐列比对 9 个 CSV，
**选路面量（偏转、观测数、报告数、payload、时延、精细评估数、belief 捕获）全部逐位一致**，
只有判决计数变化 ⇒ 确为采样路径变更，**未动物理与优化模型**。

**已同步**：正文（§Detection、§Fusion-Placement、§Operating Boundaries）、摘要、`Conclusion.tex`、
`ReproducibilitySupplement.tex`、fig2/3/4。重跑前快照见
`archive/results_target_local_v1_pre_rngfix_2026-09-16/` 与 `archive/paper_figs_pre_rngfix_2026-09-16/`。
`ReproducibilitySupplement.tex` L126-148 的历史命名说明仍然成立，无需改写。

### 5.2 `kappa_dc = 40 dB` 的标定语与实测不符 —— **已澄清（2026-09-18）**

原判据：40 dB 对消把直射压到"回波量级"，但实测 `r = residual/n0 ≈ +9.9 dB`（400 m），
而 raw 回波比 `n0` 低约 37 dB ⇒ 看似矛盾 47 dB。

**结论：两个错误叠加，均已定位（`tools/probe_kappa_necessity.py`，生产链路表口径）。**

1. **"echo" 是处理后回波**（含 `N*L = 36.1 dB` 积累），不是 raw 回波。实测
   `echo/n0 = -2.3 dB`（600 m / RCS 0.1），而 raw 回波 ≈ −37 dB，差的正是积累增益。
2. **近远比是几何量，不随场景迁移**：legacy 论文几何（4 km / RCS 50 m²）= 41.3 dB，
   当前主场景（600 m / RCS 0.1 m²）= **50.9 dB**。

正确读数：κ=40 时 `residual/n0 = +8.7 dB` vs `echo/n0 = -2.3 dB` ⇒ 残余直连比处理后回波
高 **11.0 dB**，与 50.9 − 40 = 10.9 dB 自洽。

**影响**：40 dB 在当前主场景是**偏保守**的（残余仍高于回波 11 dB），约 51 dB 才对应
"到回波量级"，60 dB 只让残余低于回波 2.4 dB 且已触及自残留/噪声地板。因此 κ 40→60
不是"从刚够跳到激进"，而是"从不够补到刚好"。**但 κ 仍是 FROZEN 假设**：改它要重新
论证对消能力并重算全套数字，不可与发布值并列引用。无对消时（κ=0）实测
P_D 0.0505 vs P_FA 0.0495 ⇒ 感知完全失效，κ 是可行性前提而非性能旋钮。

### 5.3 工作点标定登记表（可标定项的当前取值）

| 键 | 当前值 | 备注 |
|---|---|---|
| `scale.M` / `scale.Q` | 15 / 10 | 两个工作点共用 |
| `geometry.area_xy` | **4000 m**（主口径） / 600 m（小 UAV 场景） | 600 m 见 `DEPLOYMENT_RANGE_ANALYSIS.md`、`LOW_RCS_SCENARIO_500_800.md` |
| `detect.target_rcs` | **50 m²**（主口径） / 0.1 m²（小 UAV 场景） | **仍在标定**；小目标档 0.02–0.2 m² |
| `selector.max_tx_nodes` / `tx_penalty` | `None` / `0.0`（默认） | rev.2 新增登记；协调臂用 `max_tx_nodes=3`，**报数时必须一并给出** |
| `radio.radar_net_gain_db` | `None`（⇒ 0 dB） | 冻结键（列此仅为留痕）；与 `small-uav-link-budget-bridge` 的 27 dB 桥冲突，见 `RADAR_LINK_BUDGET_CALIBRATION.md` |
| `run.num_mc` / `run.seed` | 默认 200；主结果 **1000** / 2026 | 门禁会把它打成 NOTE |

**两个工作点的发布口径读数**（均为 `proposed_c2f_adaptive_pd`，$G^{\rm hw}=0$ dB，κ=40 dB）：

| 工作点 | preset | MC | P_D | 最差目标 P_D | 开销 | 来源 |
|---|---|---|---|---|---|---|
| 主口径（论文正文） | `target-local-v1` | 1000 | 0.9764 | 0.966 | — | 归档 `_archive/2026-09-18/results_target_local_v1/main/main.csv`（§5.1 重跑后） |
| 小 UAV 场景 × 协调关闭 | `small-uav-compact-800m` | 1000 | **0.6938** | 0.665 | — | `results_release_mc1000/base_off` |
| 小 UAV 场景 × 协调 + cap3 | 同上 + `coordination.enable` | 1000 | **0.8501** | 0.838 | — | `results_release_mc1000/k40_coord_cap3` |

> 小 UAV 场景的这两行**不是主结果**，是 rev.2 为了把"纯算法能闭合到什么程度"钉死而跑的
> 发布口径对照（κ 未动、硬件未动）。协调那一行还要额外吃掉 §2.4 里那条**协议假设**。

### 5.4 包版本串与实际发布线不一致

`isac_sim.__version__ = "1.6.0"` 指的是**已被排除出本文的 V1.6 机制轨**
（见 `CONFERENCE_PAPER_CONVERGENCE.md` "Material excluded"）。
发布线是 **V1**。建议改为 `1.0.0` 或在文档中永久保留该映射，避免审稿人误解。

### 5.5 其余未收口（不影响本次冻结）

融合栈三处逻辑不自洽（互斥 / 阈值退回 CF / H0-H1 不对称）；系统是单圈闭环、缺时间划分 τ；
`distributed_bids` 从未被任何 experiment 调用；`legacy` preset 只钉 2 键、结构上无法复现历史 CSV。

### 5.6 逐位门禁是**环境绑定**的（rev.2 新增，本轮实测归因）

门禁 2 的结论只有在**契约运行时**下才成立。本轮在两条运行时上各跑一次：

| 代码版本 | 运行时 | cells | differing | `--strict` |
|---|---|---|---|---|
| HEAD `acdcacc` 干净副本（`git worktree`） | py3.13.14 / numpy **2.5.2** | 1596 | **25**（全部 rel ≤1e-12） | FAIL |
| 工作区（rev.2 全部改动） | py3.13.14 / numpy **2.5.2** | 1596 | **25**（同样 25 个） | FAIL |
| 工作区（rev.2 全部改动） | py3.11.0 / numpy **2.2.6** | 1596 | **0** | **CLEAN** |

**归因**：干净副本在 3.13 下也给出**同样 25 个**单元格 ⇒ 这 25 个是解释器/`numpy` 算术差异，
**不是** rev.2 的模型漂移。判据用的是"冻结提交 vs 工作区、同运行时对比"这条对照，
而不是"数字看起来很小"。

**因此**：`release/V1_STABLE_MANIFEST.json` 记录 `environment` 字段；`check_release_identity`
在运行时不一致时打 NOTE 并把它连同 `parity_check` 的判定一起提示；`parity_check` 增加
`--allow-env-noise`——**只在所有差异都落在 `tiny` 带（rel ≤1e-9）内时**才放行，且必须与
上表这条对照一起引用，不能单独当成"通过"。

**发布口径基线同样是环境绑定的**（`.workbuddy/baseline_release/main/main.csv`，
md5 `417d016b912f67558750610edfc8d42e`，由 py3.11/numpy 2.2.6 生成）。

**它是不是"活守卫"——实测过一次**（不靠推理）：把 κ 从 40 改成 30，其余全部不动，
与发布基线在 mc=12/seed=2026 上比对：

| 指标 | 值 |
|---|---|
| 比对单元格 | 1653 |
| 发生变化 | **959**（order 255 / large 444 / moderate 241 / small 17 / tiny 2） |
| **超出容差** | **958 ⇒ 门禁 exit 1** |

⇒ 一个 10 dB 的 κ 变化会让发布口径**58% 的单元格**发生量级级变化。
这既证明门禁 2b 有效，也是 §2.4 那张假设账本最硬的注脚：**主结果的绝对水平由 κ 决定**。

### 5.7 协调相关的未收口（rev.2 新增）

1. **论文侧尚未写协调**。`ProposedMethod.tex` 没有不动点这一节，协议章节也没写"非照明机静默"。
   写进去之前，协调数字只能作**机制验证**引用。
2. **`tools/*` 的 0.9500 与发布口径不可并列**。那是另一套入口（`select_with_coordination` 直调），
   与本版发布路径的选择器/评估链路不同口径，跨口径比绝对值这个坑已踩过两次。
3. **κ=50/60 的臂只作敏感性留痕**。`results_release_mc1000/k50_coord_cap3`（P_D 0.8848）、
   `k60_coord_cap3`（0.8901）说明 κ 从 40 提到 50 再抬 +0.035、50→60 只 +0.005；
   但 **κ 仍是 FROZEN 前提，40 dB 是发布值**，这两臂不得进主结果表。
   注意：其中的 0.8848 与记忆里"κ=60 且无协调"的历史数字**数值巧合、口径不同**，不可互相印证。
4. **论文正文仍是 4000 m / RCS 50 m²**，而支撑它的产物已归档（§0 归档提示）。
   改判到小 UAV 场景时，`SimulationResults.tex`、`SUBMISSION_TRACEABILITY.md` §3、
   `ReproducibilitySupplement.tex` 需要一并处理——**本轮按用户决定未动**。
5. **`config.py` 中 `Coordination` 的 docstring 已修正口径**：原先写 "worst-target P_D 0.85 → 0.95"
   （那是 `tools/*` 口径），现改为发布口径 MC=1000 的 `P_D 0.6938 → 0.8501`、最差 `0.665 → 0.838`。

---

## 6. 复现

```bash
# 验收门禁（必须用契约运行时 py3.11 / numpy 2.2.6，见 §5.6）
python tools/check_release_identity.py --check
python tools/parity_check.py --baseline --strict         # 重构口径 → 1596 cells / 0 differing
python tools/parity_check.py --release-baseline --strict # 发布口径 → 1653 cells / 0 differing
python -m pytest tests/ -q                               # → 180 passed, 6 subtests
python tools/check_contract_refs.py                      # 三层对应表行号 → 32 refs / CLEAN

# 非契约运行时下必须显式区分解释器噪声与模型漂移
python tools/parity_check.py --baseline --strict --allow-env-noise

# 重冻版本标识（仅在有意变更冻结项后执行）
python tools/check_release_identity.py --freeze

# 重冻发布口径基线（仅在"归因完成且确属有意"后执行，见 §7）
python tools/parity_check.py --release-baseline --write-release-baseline

# 主实验（MC=1000，约 11 min @ 8 workers；长跑建议后台）
python tools/rerun_target_local_v1.py --suite main --workers 8 --out results_target_local_v1

# 完整 V1 释放树（main + full-refinement + overview + prediction-stress，约 45 min @ 8 workers）
python tools/rerun_target_local_v1.py --suite all --workers 8 --out results_target_local_v1

# 重跑后核对：选路面必须逐位一致，只有判决计数允许变化
python tools/report_v1_rerun_drift.py --old <旧快照目录> --new results_target_local_v1

# 重出论文图件（fig2/3 主稿 + fig4 补充材料）
python tools/make_target_local_v1_paper_figs.py
```

**小 UAV 场景 / 协调 / κ 相关的复现命令**（rev.2 新增；这些**不是**主结果线，
但它们把"纯算法能闭合到什么程度"钉死，且全部在冻结值上跑，不引入硬件假设）：

```bash
PY=python   # 必须 py3.11 / numpy 2.2.6

# 基线（协调关闭）：600 m / RCS 0.1 / G_hw=0 / κ=40
$PY -m isac_sim --mode main --preset small-uav-compact-800m \
    --set geometry.area_xy=600 --set detect.target_rcs=0.1 \
    --mc 1000 --workers 8 --methods proposed_c2f_adaptive_pd \
    --out results_release_mc1000/base_off --no-plots --quiet

# 协调 + 照明机上限 3（吃掉 §2.4 的协议假设）
$PY -m isac_sim --mode main --preset small-uav-compact-800m \
    --set geometry.area_xy=600 --set detect.target_rcs=0.1 \
    --set interference.sense_gate_by_active_tx=true \
    --set coordination.enable=true --set selector.max_tx_nodes=3 \
    --mc 1000 --workers 8 --methods proposed_c2f_adaptive_pd \
    --out results_release_mc1000/k40_coord_cap3 --no-plots --quiet

# κ 的推导与生产口径核对（纯解析 + 只读链路表，不改任何配置）
$PY tools/probe_kappa_necessity.py --area 600 --rcs 0.1 --trials 6
$PY tools/derive_kappa_pilot_budget.py --preset paper-canonical --area 4000 --rcs 50 --trials 6

# 回归守卫：只开门控、不开协调，必须与基线逐位一致
$PY tools/summarise_coordwire.py      # 内含 A vs B 的 7400 字段逐位比对
```

⚠️ **查 κ 不要用 `--preset legacy`**：那里 `coupling="legacy"`，`compute_link_tables`
走的是不读 κ 的分支，0–80 dB 逐位相同（会得到"κ 完全无效"的假结论）。

`--suite` 默认 `main`，即不带参数时行为与旧版一致（只跑主比较）；
`--suite all` 才是完整释放树。`tools/rerun_target_local_v1.py` 在开跑前会断言
`target-local-v1` 解析出的 `detect.comm_error_model` 仍是 `gaussian_replacement`——
该键在预设表里没有写死、靠 dataclass 默认值继承，一旦默认值变动，整条 V1 会
**换物理**而不是"漂移"，所以这道断言必须留在入口处。

---

## 7. 变更规程

任何一次改动，按下面顺序判定，**不许跳步**：

1. **改的是冻结键吗？**
   - 是 → 先回答"论文里哪句话跟着变"。回答不了，就不许改。
   - 否（可标定键）→ 直接改，并在 §5.3 登记表留下现值。
2. **跑齐全部验收门禁（5 条）。** `check_release_identity` 若 FAIL，说明有一层在无声漂移 —— 先解释，再决定
   是改回还是**有意**重冻（`--freeze`）并在本文件 §5 记录原因。
3. **改了数值路径？** 那 `parity_check --baseline` 与 `--release-baseline` 必然红。此时**不许**
   直接改基线文件：先做归因（`BASELINE_DRIFT_ATTRIBUTION.md` 的流程），确认是"对齐论文"
   还是"回归"，再决定重冻。注意两条基线管的是**两件事**（重构口径 / 发布口径），
   可能只有一条红——**只有发布口径红**通常意味着动了 κ、门控或协调默认值，那属于 §7.1 第 7、8 条。
4. **改了主结果数字？** 按顺序：
   1. 归档旧树：`cp -r _archive/2026-09-18/results_target_local_v1 archive/results_target_local_v1_pre_<变更名>_<日期>`，
      并 `md5sum` 双向校验；
   2. 重跑 `python tools/rerun_target_local_v1.py --suite <受影响的套件> --workers 8`；
   3. **核对漂移性质**：`python tools/report_v1_rerun_drift.py --old <旧树> --new results_target_local_v1`。
      退出码非 0（选路面被改动）= 动物理/优化模型，**停下来归因**，不许直接改数字。
      渲染层套件（`main`、`overview`、`prediction-stress`）都吃判决计数，所以只要检测采样路径变了，
      就**必须一并重跑**——只重跑 `main` 会让 fig3/fig4 与正文主表数字口径不一致。
   4. 重出图件：`python tools/make_target_local_v1_paper_figs.py`
      ⚠️ **不是** `tools/make_paper_figs.py`——后者读的是另一套 `_archive/2026-09-18/results_release` 协议
      （`main/main/main.csv`，P_D 量级 0.87），与 V1 释放线无关；
   5. 同步 `SUBMISSION_TRACEABILITY.md` §3 的数字映射表、`_archive/2026-09-18/results_target_local_v1/V1_EXPERIMENT_OVERVIEW.md`，
      以及摘要 / `Conclusion.tex` / 补充材料里所有引用该数字的句子；
   6. 重编论文确认页数仍在 6 页内。

### 7.1 rev.2 补充的四条（都在本轮踩过一次）

5. **动了 `model.py` / `selection.py` / `simulate.py` 的行数？** 那 §2 的行号就都可能失效。
   跑 `python tools/check_contract_refs.py`——它是唯一会发现这件事的东西
   （本轮建立时立刻抓到 5 处漂移，测试套件全绿）。
6. **逐位门禁红了，先看运行时。** 若运行时 ≠ 契约运行时（§5.6），先跑
   `parity_check --baseline --strict --allow-env-noise`；它放行**当且仅当**所有差异都在
   `tiny` 带内。若仍有超出 band 的差异，或你想宣布"环境噪声"，
   必须给出"**冻结提交 + 同一运行时**"的对照——不能凭差异小就归因。
7. **要动 κ 或门控（`sense_gate_by_active_tx`）？** 先读 §2.4：它们是**前提**不是贡献。
   动它们要在论文里改"对消能力/协议"那句话并重算全套数字；只做敏感性分析时，
   结果放 §5.7 那种留痕位置，**不得进主结果表**。κ 从 40 往上调的论证要引用
   `KAPPA_DERIVATION.md`（近远比是几何量，换场景就不迁移：4 km/RCS50 = 48.0 dB，
   600 m/RCS0.1 = 52.25 dB）。
8. **要开 `coordination.enable`？** 它同时依赖门控（`validate_config` 会拒绝"开协调却不开门控"）。
   报数时必须连同 `coordination_converged_rate` 与 `selector.max_tx_nodes` 一起给，
   并说明这对 `tools/*` 的 0.9500 是**另一套口径**（§5.7）。
