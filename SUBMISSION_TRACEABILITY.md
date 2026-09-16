# 投稿可追溯边界（Submission Traceability）

日期：2026-09-16
适用范围：target-local fusion 会议稿（released V1）
用途：回答"论文里这个数字来自哪版代码、哪个结果文件"。

---

## 1. 投稿目标（已核实，2026-09-16）

| 项 | 结论 |
|---|---|
| 目标场所 | ICC 2027（官方 `ieee-icc.org` / `icc2027.ieee-icc.org`） |
| Symposium 投稿截止 | **2026-10-02**（距核实日 16 天） |
| 录用通知 / Camera-ready | 2027-01-15 / 2027-02-19 |
| 初始稿页数上限 | **6 页**（10 pt，双栏）；**超过 6 页无审直接拒** |
| 最终稿页数 | 最多 6 页；第 7–8 页每页 US$100 |
| 评审形式 | **单盲**：提交时作者列表与标题须与 EDAS 注册页完全一致，**不需要匿名** |
| 提交系统 | EDAS |
| 当前稿件 | 主稿 5 页、补充材料 8 页 → 满足上限，尚余 **1 页** |

备选 VTC2027-Spring（官方 `vtsociety.org`）：常规投稿已于 **2026-09-01 截止**；
仅 Workshop（2026-12-15 截止）仍开放。免费 5 页，最多 7 页（US$100/页）。

> ⚠️ `icc27.org` 与官方规则冲突（声称双盲、截止 8/17、投稿类别为 Abstract/
> Extended Abstract/Short Paper/Full Paper 6–8 页）。这不是 IEEE ICC 的官方站点，
> **不要向该域名投稿或据其规则排版**。

---

## 2. 本次提交的边界

基线 `62ceed2`（feat: stabilize theorem-backed sensing release）→ 当前 `b23d3e9`。

| commit | 内容 |
|---|---|
| `a0e7100` | chore：忽略可重生成的构建产物与冒烟跑（`.codex-*-build/`、`.chart-data-*/`、`ppt/.build_*/`、`ppt/**/*.pptx`、`results_*_smoke/`） |
| `4aefac8` | feat(isac_sim)：门控式新增 fusion_polish / joint_polish / power_* / mixture_probability 模块 |
| `255faf8` | feat(paper)：按 ISAC 2026 审稿意见修订正文、补充材料、架构图与算法表 |
| `902781d` | feat(tools,tests)：V1 审计、汇总脚本与回归测试（25 文件 2158 行） |
| `e6c4ed7` | feat(results)：归档 V1 系列与机制实验结果（132 文件 35856 行） |
| `b23d3e9` | docs：V1 各轮报告、审稿记录与投稿收敛契约 |

**关键声明**：`4aefac8` 的全部改动均为门控式，默认路径未变——

- `radio.rho_by_uav` 默认 `None`，此时 `P_sense = rho*P` 与改动前逐位相同；
- 新方法 `proposed_c2f_adaptive_pd_fusion_polish` 归属 `EXPERIMENTAL_METHODS`，
  使用独立的 RNG offset 142，既有方法的 offset 未被触碰；
- `select_c2f_adaptive` 的新参数 `refined_table_builder` / `shortlist_seed` 默认 `None`；
- `dd.py` 仅修正 `eta_local` 文档字符串的错误断言，无计算变更。

因此 §3 的数字虽由 9/14 的结果文件产生、代码在 9/16 有改动，**数值路径未受影响**。
⚠️ 但这是**代码级审查结论，尚未通过实跑回归**（见 §5）。

---

## 3. 正文数字 → 结果文件

### 主比较（`SimulationResults.tex` 表 1、图 `fig2_v1_main_comparison`）

来源：`results_target_local_v1/main/main.csv`（MC=1000，seed 2026）
`method` 列取 `proposed_c2f_adaptive_pd` / `exact_marginal_greedy` / `sense_sinr`。

| 正文表述 | CSV 列 | 值 |
|---|---|---|
| $P_D$ | `P_D` | 0.9762 / 0.9411 / 0.9783 |
| worst-target | `worst_target_D_mean` | 0.967 / 0.928 / 0.970 |
| 配对差值 | `paired_reference_delta_P_D` | +0.0351 / −0.0021 |
| 95% 区间 | `paired_reference_delta_ci95_{low,high}` | [0.0310, 0.0392] / [−0.0051, 0.0009] |
| 选中观测数 | `selected_observations_mean` | 12.101（三者相同） |
| 远程报告数 | `selected_links_mean` | 0.751 / — / 4.595 |
| payload / 串行时延 | `B_mean_bits` / `T_mean_ms` | 480.64→0.481 kbit、0.801 ms / 2940.8→2.941 kbit、4.901 ms |
| 精细评估次数 | `fine_eval_full_mean` / `fine_eval_c2f_mean` | 860.244 / 61.278 |

83.7% = 1 − 0.751/4.595（报告数下降，payload 与时延同源，**不构成两个独立增益**）。

### 全精细控制（MC=200）

来源：`results_target_local_v1/full-refinement-pd/main.csv`

| 正文表述 | 列 | 值 |
|---|---|---|
| C2F vs full 精细评估 | `fine_eval_c2f_mean` / `fine_eval_full_mean` | 61.06 / 853.65（92.85%） |
| $P_D$ | `P_D` | 0.980 vs 0.973 |
| 配对差值 | `paired_reference_delta_P_D` 等 | 0.0070 [0.0029, 0.0111] |

full-refinement 是**计算控制组，不是 oracle**。

### 融合位置消融（图 `fig3_v1_fusion_ablation`，MC=100）

来源：`results_target_local_v1/overview/fusion-rule/fusion-rule.csv`（`fusion_rule` 列区分）

### 运行边界（`Operating Boundaries` 小节）

来源：`results_target_local_v1/overview/geometry/geometry.csv`、
`results_target_local_v1/prediction-stress/prediction_stress.csv`

### 其余扫描（补充材料）

`overview/{blocklength,communication,lambda,scale,packetization}/`。
均为 MC=100 诊断性扫描，**不能替代主证据**。

---

## 4. 复现命令

```bash
# 主实验 + 全精细控制 + 全部 overview 扫描
E:/anaconda/3_11_python/python.exe tools/rerun_target_local_v1.py

# 全量回归测试（当前 180 passed, 6 subtests passed）
E:/anaconda/3_11_python/python.exe -m pytest tests/ -q

# 审稿意见逐条核验
E:/anaconda/3_11_python/python.exe tools/audit_isac_review_revision.py
```

论文编译（`Conference-LaTeX-template_10-17-19/`，需连编两次）：

```bash
pdflatex "Communication-Constrained Soft-Information Fusion for Multi-UAV OTFS-ISAC Cooperative Sensing.tex"
pdflatex "Communication-Constrained Soft-Information Fusion for Multi-UAV OTFS-ISAC Cooperative Sensing.tex"
pdflatex ReproducibilitySupplement.tex && pdflatex ReproducibilitySupplement.tex
```

---

## 5. 尚未收口（本文件不宣称已解决）

> **2026-09-16 更新**：第 1 项已由实测证实为**真实漂移**（不只是"未验证"），
> 详见 `MANUSCRIPT_CODE_CONSISTENCY_ALERT.md`。第 2 项已完成消融，
> 见 `V1_UTILITY_ABLATION.md`。

1. 🔴 **检测数字与当前代码不一致（最高优先级）**：判决阈值已于 2026-09-15
   （`03f9612`）由 Cornish–Fisher 近似换成 `calibrated_fused_threshold`，
   而正文的 $P_D$ 数字来自 2026-09-14 的旧判决结果。MC=100 实测漂移
   −0.007（0.982 → 0.975）。**通信效率数字不受影响**（报告数/时延/观测数逐位一致）。
   需要重跑 MC=1000 主实验并更新正文与 fig2。⚠️ 注意正文已声明使用
   "corrected H0 threshold"，因此当前是"描述与数据不符"。
2. ✅ **效用两项必要性消融已完成**：MC=100、同场景、trial 配对。移除
   soft-min 项、二次缺口项或两者，对检测**均无可检出影响**，弱目标 $P_D$
   四组完全相同。结论：**不能宣称两项必要**，但也**不足以据此删除**公式
   （MC=100 单配置、检测接近饱和、`no_softmin` 变体语义不纯）。
3. **全面新颖性查新未做**：仅在有限文献内比较，未宣称完成查新。
4. **完整多目标波形 / 编码 / 量化 / 硬件验证未做**：已在正文与补充材料声明边界。
5. **V1.6 机制轨门禁悬置**：G1–G3 pass，G4 仅 `infrastructure_pass`，
   G5 恒为 `pending_frozen_holdout`。该轨道已排除出会议稿，不阻塞投稿。
6. 补充材料当前 8 页，投稿时需按 ICC 政策确认其是否作为附件单独提交。

### 顺带发现（非缺陷）

现有结果目录的 `config.json` 记 `detect.comm_error_model = erasure`，而当前
`erasure` 已改指"真零擦除"。这**不是**模型写错——源版本 `69f3300` 的
`soft_channel.py:142` 中 `erasure` 分支返回的正是 $\mathcal N(0,3\sqrt{v_0})$
（方差 $9v_0$，$a=9$），即当时 `erasure` ≡ 当前 `gaussian_replacement`。
论文声明与当时执行一致，`rerun_target_local_v1.py` 固定 `gaussian_replacement`
也是正确的复现入口。属历史命名债。

---

## 6. 已核实的历史记录偏差

- `ISAC_V1_OPTIMIZATION_AUDIT.md` 称"图1仍含历史 greedy score/目的节点符号"——
  **已过时**：现主稿由 `ReportingArchitecture.tex` 提供架构图，符号为 $i\to q\to j\to f_q$。
- 同一文档"62 tests"、`ISAC_REVIEW_REVISION.md`"154 passed"——
  均落后于实测 **180 passed, 6 subtests passed**。
