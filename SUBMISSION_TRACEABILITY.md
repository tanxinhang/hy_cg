> **归档提示（2026-09-18）**：本文引用的部分 `results_*` 产物已移入 `_archive/2026-09-18/`；正文中的路径引用已同步更新为归档位置，命令行示例里的 `--out` 目录仍写作历史原名（重跑时依旧输出到该名）。

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

来源：`_archive/2026-09-18/results_target_local_v1/main/main.csv`（MC=1000，seed 2026）
`method` 列取 `proposed_c2f_adaptive_pd` / `exact_marginal_greedy` / `sense_sinr`。

| 正文表述 | CSV 列 | 值 |
|---|---|---|
| $P_D$ | `P_D` | 0.9764 / 0.9408 / 0.9784 |
| worst-target | `actual_worst_target_P_D` | 0.966 / 0.931 / 0.972 |
| 配对差值 | `paired_reference_delta_P_D` | +0.0356 / −0.0020 |
| 95% 区间 | `paired_reference_delta_ci95_{low,high}` | [0.0311, 0.0401] / [−0.0050, 0.0010] |
| 选中观测数 | `selected_observations_mean` | 12.101（三者相同） |
| 远程报告数 | `selected_links_mean` | 0.751 / — / 4.595 |
| payload / 串行时延 | `B_mean_bits` / `T_mean_ms` | 480.64→0.481 kbit、0.801 ms / 2940.8→2.941 kbit、4.901 ms |
| 精细评估次数 | `fine_eval_full_mean` / `fine_eval_c2f_mean` | 860.244 / 61.278 |

83.7% = 1 − 0.751/4.595（报告数下降，payload 与时延同源，**不构成两个独立增益**）。

### 全精细控制（MC=200）

来源：`_archive/2026-09-18/results_target_local_v1/full-refinement-pd/main.csv`

| 正文表述 | 列 | 值 |
|---|---|---|
| C2F vs full 精细评估 | `fine_eval_c2f_mean` / `fine_eval_full_mean` | 61.06 / 853.65（92.85%） |
| $P_D$ | `P_D` | 0.9775 vs 0.9715 |
| 配对差值 | `paired_reference_delta_P_D` 等 | 0.0060 [0.0005, 0.0115] |

full-refinement 是**计算控制组，不是 oracle**。

### 融合位置消融（图 `fig3_v1_fusion_ablation`，MC=100）

来源：`_archive/2026-09-18/results_target_local_v1/overview/fusion-rule/fusion-rule.csv`（`fusion_rule` 列区分）

| 正文表述 | 列 | max-in-rate / max-min-rate / nearest-target |
|---|---|---|
| $P_D$ | `P_D` | 0.932 / 0.967 / 0.975 |
| worst-target | `actual_worst_target_P_D` | 0.900 / 0.940 / 0.960 |
| 观测数 | `selected_observations_mean` | 19.89 / 15.61 / 12.24 |
| 报告数 | `selected_links_mean` | 6.65 / 6.40 / 0.74 |
| 时延 | `T_mean_ms` | 7.093 / 6.827 / 0.789 |

融合 UAV 数 7.13、冲突时隙 0.70 来自 `overview/packetization/{key}.json`
（`assigned_fusion_uavs_mean` / `conflict_graph_slots_mean`），三个规则下**逐位不变**。

### 运行边界（`Operating Boundaries` 小节）

来源：`_archive/2026-09-18/results_target_local_v1/overview/geometry/geometry.csv`、
`_archive/2026-09-18/results_target_local_v1/prediction-stress/prediction_stress.csv`

| 正文表述 | 来源列 | 值 |
|---|---|---|
| 位置误差 50→500 m | `prediction_stress.csv` `P_D`（`proposed_c2f_adaptive_pd`） | 0.988 → 0.906 |
| 部署边长 2500→5500 m | `geometry.csv` `P_D` | 0.989 → 0.940 |
| 同上的报告数 | `geometry.csv` `selected_links_mean` | 0.05 → 2.67 |

### 其余扫描（补充材料）

`overview/{blocklength,communication,lambda,scale,packetization}/`。
均为 MC=100 诊断性扫描，**不能替代主证据**。

---

## 4. 复现命令

```bash
# 完整 V1 释放树：main MC=1000 + full-refinement MC=200 + 6 个 overview 扫描 + prediction-stress
E:/anaconda/3_11_python/python.exe tools/rerun_target_local_v1.py --suite all --workers 8

# 只重跑主比较（--suite 默认 main，保持既有语义）
E:/anaconda/3_11_python/python.exe tools/rerun_target_local_v1.py --suite main --workers 8

# 重跑后核对"是否只动了检测采样"：选路面必须逐位一致，退出码非 0 即报警
E:/anaconda/3_11_python/python.exe tools/report_v1_rerun_drift.py \
    --old archive/results_target_local_v1_pre_rngfix_2026-09-16 \
    --new results_target_local_v1

# 重出论文图件（fig2/3 主稿 + fig4 补充材料）
E:/anaconda/3_11_python/python.exe tools/make_target_local_v1_paper_figs.py

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

> **2026-09-16 晚更新**：第 1 项**已收口**——主实验已按当前冻结代码完整重跑，
> 正文、补充材料与 fig2/3/4 已同步（见 §7）。第 2 项已完成消融，
> 见 `V1_UTILITY_ABLATION.md`。

1. ✅ **检测数字与当前代码不一致——已解决**。检测阶段的**随机数流**由共享顺序流
   改为逐链路键控流（`simulate.py:209` `keyed_rngs`），正文旧数字来自 2026-09-14 的旧实现。
   早前 MC=100 测得漂移 −0.007，**该读数本身就在 MC=100 的噪声内（半宽约 ±0.009）**，
   高估了漂移量级。MC=1000 重跑实测仅 **+0.0002**（0.9762 → 0.9764），
   三个方法均在 ±0.0003 内，配对差值的符号与显著性**完全未变**。
   `tools/report_v1_rerun_drift.py` 确认 9 个 CSV 的**选路面量逐位一致**，
   只有判决计数变化 ⇒ 确为采样路径变更，未动物理/优化模型。
   ⚠️ 早期文档把根因写成"判决门限被替换"，**已证伪**（门限公式与三个组成函数在两版间逐字相同，
   且 `calibrated_fused_threshold` 在 `gaussian_replacement` 下精确退化为该式）。
   完整证据见 `MANUSCRIPT_CODE_CONSISTENCY_ALERT.md` 顶部更正 与 `V1_STABLE_RELEASE.md` §5.1。
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

---

## 7. 2026-09-16 重跑记录

**起因**：检测阶段的随机数流已由共享顺序流改成逐链路键控流，而正文数字来自旧实现。

**执行**：

```bash
E:/anaconda/3_11_python/python.exe tools/rerun_target_local_v1.py \
    --suite all --mc 1000 --full-refinement-mc 200 --overview-mc 100 \
    --prediction-mc 100 --workers 8 --out results_target_local_v1
```

| 套件 | 规模 | 产物 |
|---|---|---|
| main | MC=1000 × 3 方法 | `main/main.csv` |
| full-refinement | MC=200 × 2 方法 | `full-refinement-pd/main.csv` |
| overview | MC=100 × 6 扫描（23 条件） | `overview/<sweep>/<sweep>.csv` |
| prediction-stress | MC=100 × 5 误差档 × 3 方法 | `prediction-stress/prediction_stress.csv` |

**结果**：选路面量逐位不变；判决计数变动 ≤0.0003（$P_D$）。
配对差值符号与显著性方向未变（$+0.0356$ 显著、$-0.0020$ 跨零）。

**归档**：

- 重跑前结果树 → `archive/results_target_local_v1_pre_rngfix_2026-09-16/`
- 重跑前图件 → `archive/paper_figs_pre_rngfix_2026-09-16/`

**已同步文件**：`SimulationResults.tex`（§Detection、§Fusion-Placement、§Operating Boundaries）、
摘要、`Conclusion.tex`、`ReproducibilitySupplement.tex`、
`_archive/2026-09-18/results_target_local_v1/V1_EXPERIMENT_OVERVIEW.md`、`SYSTEM_PERFORMANCE_STATUS.md`、
`MANUSCRIPT_CODE_CONSISTENCY_ALERT.md`、`V1_STABLE_RELEASE.md` §5.1、
`PAPER_RELEASE.md`、`TARGET_LOCAL_V1.md`、`CONFERENCE_PAPER_CONVERGENCE.md`。

**未同步（仅标注，未改写）**：`DEPLOYMENT_RANGE_ANALYSIS.md`、`RCS_IMPACT_ANALYSIS.md`、
`V1_FUSION_THEORY_AND_ALGORITHM.md`、`ppt/*`。这些文档的**结论是定性的**（几何净增益量级、
RCS 门槛），其引用的主结果数字属重跑前口径，已在文首加注。
