# V1 稳定版本：理论 / 模型 / 算法三层冻结契约

> 建立时间：2026-09-16
> 适用对象：ICC 2027 会议稿（`target-local-v1` 发布线）
> 可执行清单：`release/V1_STABLE_MANIFEST.json`（由 `tools/check_release_identity.py` 生成与校验）

---

## 0. 版本标识

| 项 | 值 |
|---|---|
| 发布线 | **V1** |
| 主口径预设 | `target-local-v1`（= `paper-canonical` + `fusion.rule="nearest_target"`） |
| 预设注册表 | `isac_sim/config.py:712-882`，`HEADLINE_RELEASE_PRESET`（`config.py:870`） |
| 干扰/时序口径 | `comm.interference_model = "orthogonal"`（感知并发 + 报告正交串行） |
| 冻结键 | **94** 个（见 §3 分类） |
| 默认方法名册 | **19** 个；实验方法 **9** 个 |
| 回归基线 | `.workbuddy/baseline/main/main.csv`，md5 `659a07d4ce262d311303c5eb90d8b17e` |
| 包版本串 | `isac_sim.__version__ = "1.6.0"` ⚠️ 见 §5.4（该串属于机制轨，不代表本发布线） |

**三道验收门禁**（全绿才算"稳定"）：

```bash
python tools/check_release_identity.py --check     # 三层结构未漂移        → 94 keys / 0 violated / CLEAN
python tools/parity_check.py --baseline --strict   # 默认路径逐位一致      → 1596 cells / 0 differing / EXIT 0
python -m pytest tests/ -q                         # 行为回归              → 180 passed, 6 subtests passed
```

---

## 1. "稳定"的定义

**稳定 ≠ 数字不动。** 本发布线的"稳定"指下面四条同时成立：

1. **默认路径 bit-exact**：`Config()` 与 `target-local-v1` 的解析结果，相对冻结基线逐位一致
   （由 `parity_check --strict` 守卫）。
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
| `SystemModel.tex` `eq:comm_sinr_rate` L33-42 | 上报链路 SINR 与速率 | `model.py:635-674` | ✅ |
| `SystemModel.tex` `eq:fbl_reliability` L49-54 | 有限码长包成功率 $\chi_{ab}=1-\mathsf Q(\cdot)$ | `fbl.py:134` `chi_from_gamma` | ✅ |
| `SystemModel.tex` `eq:bistatic_delay_doppler` L70-77 | 双站时延/多普勒 | `model.py:383-392`；`dd.py` | ✅ |
| `SystemModel.tex` `eq:local_dd_energy` L82-90 | 归一化有限网格 DD 泄漏 $\eta^{\rm loc}$，$W=1$ | `dd.py`（`eta_fine` / `eta_local`） | ✅ |
| `SystemModel.tex` `eq:sensing_sinr` L103-109 | $\gamma^s=P^s g^s G^{\rm hw}G_{\rm p}\eta^{\rm DD}\eta^{\rm col}/(N_0+I^s)$ | `model.py:776-794` | ✅ |
| `SystemModel.tex` `eq:local_llr` L119-126 | $\ell_e=\frac{\gamma}{1+\gamma}(X-K)$，$\delta,v_0,v_1$ | `llr.py:83-104` | ✅ |
| `SystemModel.tex` `eq:received_moments` L131-139 | 混合后 $\delta^{\rm eff},\bar v_0,\bar v_1$（组间项在 **H1**） | `soft_channel.py:44-87`（`_mix` + `received_moments`） | ✅ |
| `SystemModel.tex` `eq:detector_prediction` L149-155 | $t_q=\sqrt{v_{q,0}}[z_0+\frac{\kappa_0}{6}(z_0^2-1)]$，$\widehat P_D=\mathsf Q(\cdot)$ | `fusion.py:195-249` `predicted_pd_for_links` | ✅ |
| `ProposedMethod.tex` `eq:nearest_target_fusion` L7-11 | $f_q=\arg\min_m\|\widehat{\mathbf p}_q-\mathbf p_m\|$ | `reporting.py:145-250` `assign_fusion_nodes`（`rule="nearest_target"`） | ✅ |
| `ProposedMethod.tex` `eq:feasible_candidate_set` L15-21 | $\mathcal E_q$ 可行性集合 | `selection.py:162-190` `feasible_links_for_target` | ✅ |
| `ProposedMethod.tex` 局部投递界 L23-28 | $d(\chi)=\chi^2\delta^2/[v_0(a+(1-a)\chi)]$ | **无独立实现**（由 `eq:received_moments` + `_mix` 代入即得；`selection.py` 的局部投递走 `link_cost_ms=0`） | ✅ 解析结论 |
| `ProposedMethod.tex` 报告数下界 L36-41 | $N_{{\rm rem},q}\ge|\mathcal S_q|-\max_m a_{qm}$ | **无独立实现**（结构性事实，`report_dest` 决定谁免包） | ✅ 解析结论 |
| `ProposedMethod.tex` `eq:fair_sensing_utility` L48-56 | 软最小 + 二次缺口效用 | `fusion.py:435-486` `selection_utility_from_pd` / `selection_utility` | ✅ |
| `ProposedMethod.tex` `eq:task_objective` L60-69 | $\max F=U-\lambda_c\sum c_e$，$c_e$ 以 **ms** 计 | `selection.py`（`link_cost_ms = 1e3·link_delay_s`）；贪心 commit 在 `_greedy_lagrangian` | ✅ |
| `ProposedMethod.tex` `eq:c2f_complexity` L94-98 | $EC_{\rm c}+rC_{\rm f}$ | `selection.py:617+` `select_c2f_adaptive`，`stats.fine_eval_full/c2f` | ✅ |
| `ReproducibilitySupplement.tex` L109-124 | 全期望/全方差恒等式 | `soft_channel.py:44-51` `_mix` | ✅（`tools/check_variance_placement.py` 恒等式残差 ≤4.8e-16） |

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
| 频谱耦合 | `interference.coupling = "shared_spectrum"` | `model.py:733-753` | `SystemModel.tex` L28-45 |
| 残留干扰三项 | `residual_self=1e-14`、`residual_multi_uav=1e-10`，走共享干扰场；直射对消由 `kappa_dc=I_sense_field·10^{-κ/10}` 控制 | `model.py:732-753` | L110-111 |
| 直射对消深度 | `interference.direct_cancellation_db = 40.0` ⚠️ 见 §5.2 | `config.py` | — |
| 照明机调度门控 | `sense_gate_by_active_tx = False`（默认关闭 ⇒ $I^s$ 是几何常数） | `model.py` | — |
| SINR 分母护栏 | `radio.eps_mode="noise_relative"`，`eps_rel_db=-30` | `model.py:57-70` `denominator_guard` | — |
| 上报干扰记账 | `comm.interference_model="orthogonal"` | `model.py:635-674` | L33-45 |
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
| C2F 加速 | 每目标 shortlist $J=25$（`refine.shortlist_size`），窗口 `W=1` | `selection.py:617+` |
| 停止准则 | `stop_at_D_min=False`（不因 $D_{\min}$ 提前 break） | `selection.py` |
| 最优性宣称 | **无**（不宣称全局最优、不宣称子模近似比、不宣称 C2F 无损） | `CONFERENCE_PAPER_CONVERGENCE.md` |

**判决阶段（检测）的实现事实**（论文未单列小节，但必须记住）：

- `simulate.py:202` 无条件调用 `calibrated_fused_threshold` 作为**所有方法共用**的判决门限。
- 在 `comm_error_model != "erasure"` 且 `corr.enable=False` 时，它**精确退化为**
  `(z_0+\frac{\kappa_0}{6}(z_0^2-1))\sqrt{v_{q,0}}`（`fusion.py:347-356`），即归档 V1 的
  Cornish–Fisher 门限。
- 因此**归档 V1 的 `gaussian_replacement` 路径上，09-15 的门限替换不改变任何数值**（见 §5.1）。

---

## 3. 冻结项 vs 可标定项

| 类别 | 内容 | 门禁行为 |
|---|---|---|
| **冻结**（94 键） | 波形网格、功率切分、噪声与护栏、残留干扰系数、上报/干扰记账、检测器全部统计量、信念/调度、选择器全部权重与约束、融合位置 | **不匹配 = FAIL** |
| **可标定**（16 键） | `scale.M/Q`、`geometry.*`、`detect.target_rcs`、`waveform_impairments.enable`、`run.num_mc/seed` | **不匹配 = NOTE**（且必须写进 §5.3 的登记表） |
| **算法名册** | `DEFAULT_METHODS` 顺序 + `EXPERIMENTAL_METHODS` 集合 + `METHOD_RNG_OFFSETS` | **不匹配 = FAIL**（RNG offset 决定 bit-exact） |

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

---

## 5. 已知偏离（本契约不宣称已解决）

### 5.1 🔴 主结果数字与当前代码不可复现 —— 根因已定位到**检测阶段 RNG 流**

**结论更正**：此前把根因归给"09-15 的判决门限替换"，**这个归因是错的**。

代码级证据（逐字比对 `03f9612^` 与当前 HEAD）：

| 部件 | 位置 | 两版关系 |
|---|---|---|
| `threshold_from_pfa` | `model.py:112` | **逐字相同** |
| `fused_h0_variance` | `fusion.py:252` | **逐字相同** |
| `fused_h0_skewness` | `fusion.py:281` | 仅把 `corr.enable or` 改成 `corr.enable and len(links)>1`；`corr.enable=False` 时行为相同 |
| 门限公式 | 旧 `simulate.py:158-159` vs 新 `fusion.py:350` | **同式**：$(z_0+\frac{\kappa_0}{6}(z_0^2-1))\sqrt{v_{q,0}}$ |

⇒ 在 `gaussian_replacement` 路径上（= 归档 V1 的等价路径），新门限函数**精确返回旧公式**。

**真正的变化**：检测阶段的随机数流从**共享顺序流**换成了**逐链路键控流**。

```
旧（03f9612^，simulate.py:164）      F += w * draw_h1_soft_stat(cfg, tables, link, q, rng, plan)
新（当前 HEAD，simulate.py:209）     h1_rngs = keyed_rngs(q, ordered_links, True, 0)   # 每 (trial,q,link,h,draw) 独立流
```

`git show 03f9612^:isac_sim/simulate.py | grep -c keyed_rngs` = **0**。这解释了
`BASELINE_DRIFT_ATTRIBUTION.md` 里"机制 B（63 格）触发点未隔离"——**触发点就是它**。

**仍成立的实测事实**：MC=100 配对，归档路径 0.982 → 当前 0.975（−0.007），
而**选择路径量（D / 观测数 / 报告数 / 时延）逐位一致**。选路面一致、只有判决计数变，
与"检测阶段换 RNG 流"完全吻合。

**待办**：主实验 MC=1000 必须重跑（P0）。重跑后正文与 `figs/fig2,3` 需同步；
`ReproducibilitySupplement.tex` L126-148 已诚实描述该段历史，**无需改写，但要把
"门限替换"这一归因从 `MANUSCRIPT_CODE_CONSISTENCY_ALERT.md` 中撤回**。

### 5.2 `kappa_dc = 40 dB` 的标定语与实测不符

`config.py` 称 40 dB 对消把直射压到"回波量级"，但实测
`r = residual/n0 ≈ +9.9 dB`（400 m），而 raw 回波比 `n0` 低约 37 dB
⇒ 残留直射实际比 raw 回波高约 **47 dB**。要么是措辞（"echo" 指未相干积累的回波），
要么 41.3 dB 近远比标定与当前几何不符。**引用 κ_dc 前必须先解决**（见
`PERFORMANCE_OPTIMIZATION_ROADMAP.md` §6.2）。

### 5.3 工作点标定登记表（可标定项的当前取值）

| 键 | 当前值 | 备注 |
|---|---|---|
| `scale.M` / `scale.Q` | 15 / 10 | 主结果工作点 |
| `geometry.area_xy` | 4000 m | 4 km 基线；600 m 变体见 `DEPLOYMENT_RANGE_ANALYSIS.md` |
| `detect.target_rcs` | 50 m² | **仍在标定**；小目标档 0.02–0.2 m² |
| `radio.radar_net_gain_db` | `None`（⇒ 0 dB） | 与 `small-uav-link-budget-bridge` 的 27 dB 桥冲突，见 `RADAR_LINK_BUDGET_CALIBRATION.md` |
| `run.num_mc` / `run.seed` | 默认 200；主结果 1000 / 2026 | — |

### 5.4 包版本串与实际发布线不一致

`isac_sim.__version__ = "1.6.0"` 指的是**已被排除出本文的 V1.6 机制轨**
（见 `CONFERENCE_PAPER_CONVERGENCE.md` "Material excluded"）。
发布线是 **V1**。建议改为 `1.0.0` 或在文档中永久保留该映射，避免审稿人误解。

### 5.5 其余未收口（不影响本次冻结）

融合栈三处逻辑不自洽（互斥 / 阈值退回 CF / H0-H1 不对称）；系统是单圈闭环、缺时间划分 τ；
`distributed_bids` 从未被任何 experiment 调用；`legacy` preset 只钉 2 键、结构上无法复现历史 CSV。

---

## 6. 复现

```bash
# 三道门禁
python tools/check_release_identity.py --check
python tools/parity_check.py --baseline --strict
python -m pytest tests/ -q

# 重冻版本标识（仅在有意变更冻结项后执行）
python tools/check_release_identity.py --freeze

# 主实验（MC=1000，数小时；必须后台）
python tools/rerun_target_local_v1.py --out results_target_local_v1
```

---

## 7. 变更规程

任何一次改动，按下面顺序判定，**不许跳步**：

1. **改的是冻结键吗？**
   - 是 → 先回答"论文里哪句话跟着变"。回答不了，就不许改。
   - 否（可标定键）→ 直接改，并在 §5.3 登记表留下现值。
2. **跑三道门禁。** `check_release_identity` 若 FAIL，说明有一层在无声漂移 —— 先解释，再决定
   是改回还是**有意**重冻（`--freeze`）并在本文件 §5 记录原因。
3. **改了数值路径？** 那 `parity_check --baseline` 必然红。此时**不许**直接改基线文件：
   先做归因（`BASELINE_DRIFT_ATTRIBUTION.md` 的流程），确认是"对齐论文"还是"回归"，
   再决定重冻。
4. **改了主结果数字？** 重跑主实验 + 重跑 `tools/make_paper_figs.py`，同步
   `SUBMISSION_TRACEABILITY.md` 的数字映射表。
