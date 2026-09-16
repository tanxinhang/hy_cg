# ⚠️ 稿件数字与当前代码一致性告警

日期：2026-09-16
严重级别：**投稿前必须处理**（影响正文检测数字的有效性）
状态：已定位根因，已实测确认漂移，**尚未重跑修正**

---

## 1. 结论摘要

论文的**检测性能数字**（$P_D$、worst-target $P_D$、配对差值）来自 2026-09-14
生成的 CSV，而**判决阈值的实现已在 2026-09-15 被更换**。因此：

- 用**当前代码**跑同样配置，得到的 $P_D$ 与论文数字**不一致**；
- 论文**通信效率**相关数字（报告数、payload、串行时延、观测数）**不受影响**，
  实测与旧结果**逐位一致**；
- 论文正文已在声明新模型（"the corrected H0 threshold"），**但数字来自旧模型**——
  这是"描述与数据不符"，而非"数据错误"。

---

## 2. 时间线

| 时间 | 事件 |
|---|---|
| 2026-09-14 17:46 | `84ae01c` release target-local fusion V1 and rebuild manuscript |
| **2026-09-14 18:39** | **`results_target_local_v1/main/main.csv` 生成**（主实验 MC=1000） |
| 2026-09-14 19:19 | `overview/fusion-rule/fusion-rule.csv` 生成（MC=100） |
| 2026-09-14 19:23 | `69f3300` 提交这批结果（即文档所称"生成主结果的源版本 69f3300"） |
| **2026-09-15 01:26** | **`03f9612` 判决阈值更换为 `calibrated_fused_threshold`** |
| 2026-09-15 17:59 | `efffc26` 后续相关改动 |
| 2026-09-15 23:29 | `62ceed2`（本次审计开始前的 HEAD） |

---

## 3. 根因

判决函数 `isac_sim/simulate.py: evaluate_detection` 的阈值计算被替换。

**源版本 69f3300**（结果所用）：

```python
base_thr = threshold_from_pfa(cfg)
var0 = fused_h0_variance(cfg, tables, q, links, weights, plan=plan, base=base)
skew0 = fused_h0_skewness(cfg, tables, q, links, weights, plan=plan, base=base)
z_cf = base_thr + (skew0 / 6.0) * (base_thr * base_thr - 1.0)   # Cornish–Fisher 近似
thr = z_cf * math.sqrt(max(var0, EPS))
```

**当前 HEAD**：

```python
weight_mode = fusion_weight_mode_for_method(method)
# Threshold calibration belongs to the detector and selected set, not
# to a proposed method label.  This gives every arm the same realised
# P_FA semantics under the true-erasure mixture.
thr = calibrated_fused_threshold(
    cfg, tables, q, links, weights, plan=plan, base=base,
    statistic_mode=weight_mode,
)
```

即：判决阈值从 **Cornish–Fisher 近似的正态阈值** 换成 **有限-look LLR 混合分布的
精确分位数校准**（`fusion.py: calibrated_fused_threshold`）。后者更精确，但**结果不同**。

> 注意方向：该函数在 `fusion.py` 的**预测**分支（`predicted_pd_for_links`）只在
> `weight_mode == "exact_llr_sum"` 时调用，而 `proposed_c2f_adaptive_pd` 的
> `weight_mode` 是 `"deflection"`。但**判决**分支（`evaluate_detection`）**无条件**
> 调用它。因此只看 `fusion.py` 会误判为"不影响 V1"——必须看 `simulate.py`。

---

## 4. 实测证据（MC=100，seed 2026，preset `target-local-v1`）

| 配置 | $P_D$ | actual worst-target $P_D$ | worst-target $D$ | 观测 | 报告 | 时延 (ms) |
|---|---|---|---|---|---|---|
| **9/14 代码** + `erasure`（当时语义 = 高斯 $a{=}9$） | **0.982** | 0.95 | 2310.1863613680944 | 12.24 | 0.74 | 0.7893333333333334 |
| **当前代码** + `erasure`（真零擦除） | 0.975 | 0.96 | 2310.1863613680944 | 12.24 | 0.75 | 0.8000000000000002 |
| **当前代码** + `gaussian_replacement` | **0.975** | 0.96 | 2310.1863613680944 | 12.24 | 0.74 | 0.7893333333333334 |

**读法**：

1. 第 1 行与第 3 行的**选择路径量完全一致**（$D$=2310.1863613680944、观测 12.24、
   报告 0.74、时延 0.7893333333333334），**只有 $P_D$ 不同**（0.982 → 0.975）。
   → 差异**只**来自判决阈值，与选择算法、信道、几何无关。
2. 第 2 行（真零擦除）与第 3 行 $P_D$ 相同但报告数/时延不同，说明 `erasure` 与
   `gaussian_replacement` 在**选择**上有细微差异（预测 $P_D$ 不同 → 选择略变），
   但恰好不改变检测结果。**两者不是同一个模型**。

产物：`results_v1_failure_model_check/erasure/`、`results_v1_utility_ablation/full/`。

---

## 5. 影响评估

| 论文主张 | 是否受影响 | 说明 |
|---|---|---|
| 通信效率：83.7% payload/时延下降、0.751 vs 4.595 报告 | **不受影响** | 选择路径未变，报告数/时延逐位一致 |
| 精细评估 61.278 / 860.244（92.88%） | **不受影响** | 属选择算法的计数，与判决阈值无关 |
| 主 $P_D$ 0.9762、worst-target 0.967 | **受影响** | MC=100 已实测漂移 −0.007；MC=1000 需重跑确认 |
| 配对差值 +0.0351 / −0.0021 及其区间 | **受影响** | 由判决结果计算 |
| 融合位置消融的 $P_D$ 0.919 / 0.969 / 0.982 | **受影响** | 该 CSV 同样是 9/14 生成 |
| "两效用项不宣称必要" | 已由新消融补上 | 见 `V1_UTILITY_ABLATION.md` |

---

## 6. 建议行动（按优先级）

1. **用当前代码重跑主实验 MC=1000**（`tools/rerun_target_local_v1.py`），
   用新 $P_D$/worst-target/配对差值替换正文与 fig2 的数字。
   通信数字预期不变，可先对比确认后再只改检测部分。
2. **同步重跑 `overview/fusion-rule`（MC=100）与 `prediction-stress`、`geometry`**，
   否则 fig3 与运行边界小节的数字仍是旧判决。
3. 重跑后**重建论文图**（`tools/make_paper_figs.py`）。
4. 在 `SUBMISSION_TRACEABILITY.md` 中更新 commit 边界，把"结果版本"与"代码版本"
   重新对齐（当前二者相差 1 天）。
5. 若因时间不够而无法重跑，则**必须**在正文明确声明数字对应的判决实现版本——
   但这会削弱稿件可信度，不推荐。

> 时间压力：ICC 2027 Symposium 投稿截止 **2026-10-02**，剩余约 16 天。

---

## 7. 复现命令

```bash
# 判决漂移的最小复现（MC=100，约 2.5 分钟）
E:/anaconda/3_11_python/python.exe run_isac_sim.py --mode main \
  --preset target-local-v1 --set detect.comm_error_model=gaussian_replacement \
  --mc 100 --seed 2026 --methods proposed_c2f_adaptive_pd \
  --paired-reference proposed_c2f_adaptive_pd \
  --out results_v1_failure_model_check/gaussian --quiet --no-plots

# 与 9/14 的结果对比
#   results_target_local_v1/overview/fusion-rule/fusion-rule.csv (condition=nearest_target)
#   → P_D 0.982   而上面重跑 → P_D 0.975
```

---

## 8. 附：另一个易混点（已澄清，非缺陷）

现有结果目录的 `config.json` 里 `detect.comm_error_model` 字段值均为 **`erasure`**，
而当前 `erasure` 的语义已变为"真零擦除"。**不要据此判定论文失败模型写错**：

源版本 `69f3300` 的 `soft_channel.py:142` 中，`erasure` 分支返回的正是
$\mathcal N(0,\sigma_{\rm scale}\sqrt{v_0})$，$\sigma_{\rm scale}=3.0$ → 方差 $9v_0$
→ **$a=9$ 高斯替代**。即当时的 `erasure` ≡ 当前的 `gaussian_replacement`。

所以：论文声明"Gaussian replacement, $a=9$"**与当时执行一致**；
`tools/rerun_target_local_v1.py` 显式固定 `gaussian_replacement` 也是**正确的复现入口**。
这只是历史命名债，不是模型错误。
