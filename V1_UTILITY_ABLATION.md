# V1 效用两项必要性消融（Utility-Term Necessity Ablation）

日期：2026-09-16
目的：回应 `ISAC_V1_OPTIMIZATION_AUDIT.md` 与 `ISAC_REVIEW_REVISION.md` 中悬置的
待办——"同场景效用消融，决定是否删除二次项"。

---

## 1. 被消融的两项

正文式 \eqref{eq:fair_sensing_utility} 的效用由两项组成（实现见
`isac_sim/fusion.py: selection_utility_from_pd` / `target_alpha`）：

| 项 | 数学形式 | 实现开关 | 关闭方式 |
|---|---|---|---|
| soft-min（弱目标优先） | $-Q\tau_\alpha\log\sum_q\exp[-\bar P_{D,q}/\tau_\alpha]$ | `selector.use_softmin_alpha` | 置 `False`（退化为 $\sum_q\bar P_{D,q}$） |
| 二次缺口惩罚 | $-\frac{\mu_\alpha}{2P_D^{\rm req}}\sum_q[P_D^{\rm req}-\widehat P_{D,q}]_+^2$ | `selector.mu_deficit` | 置 `0.0` |

**两项都无需改代码**：前者是既有布尔开关，后者是既有数值系数，全部走
dotted-path override。

---

## 2. 实验设计

四个配置共享**同一条随机数流**（几何、噪声、目标状态完全相同），因此可做
trial 级配对比较：

```
python run_isac_sim.py --mode main --preset target-local-v1 \
  --set detect.comm_error_model=gaussian_replacement \
  --mc 100 --seed 2026 --methods proposed_c2f_adaptive_pd \
  --paired-reference proposed_c2f_adaptive_pd \
  --out results_v1_utility_ablation/<variant> --quiet --no-plots
```

| variant | 附加 override |
|---|---|
| `full` | 无（released 配置） |
| `no_deficit_penalty` | `--set selector.mu_deficit=0.0` |
| `no_softmin` | `--set selector.use_softmin_alpha=False` |
| `neither` | 两者同时 |

产物：`results_v1_utility_ablation/<variant>/main/{main.csv,trials.csv,config.json}`。
汇总脚本：`tools/summarize_v1_utility_ablation.py`。

---

## 3. 结果（MC=100，seed 2026）

| variant | $P_D$ | 弱目标 $P_D$ | 选中观测 | 远程报告 | payload (bit) | 串行时延 (ms) |
|---|---|---|---|---|---|---|
| `full` | 0.9750 | 0.9300 | 12.240 | 0.740 | 473.6 | 0.7893 |
| `no_deficit_penalty` | 0.9740 | 0.9300 | 12.240 | 0.720 | 460.8 | 0.7680 |
| `no_softmin` | 0.9750 | 0.9300 | 12.230 | 0.720 | 460.8 | 0.7680 |
| `neither` | 0.9750 | 0.9300 | 12.230 | 0.690 | 441.6 | 0.7360 |

Trial 配对差值（vs `full`，95% trial-cluster bootstrap）：

| variant | 端点 | 差值 | 95% 区间 | 判定 |
|---|---|---|---|---|
| `no_deficit_penalty` | $P_D$ | −0.0010 | [−0.0030, +0.0000] | 未解析 |
| | 弱目标 $P_D$ | +0.0000 | [+0.0000, +0.0000] | 未解析 |
| | 远程报告 | −0.0200 | [−0.0500, +0.0000] | 未解析 |
| `no_softmin` | $P_D$ | +0.0000 | [+0.0000, +0.0000] | 未解析 |
| | 弱目标 $P_D$ | +0.0000 | [+0.0000, +0.0000] | 未解析 |
| | 远程报告 | −0.0200 | [−0.0800, +0.0300] | 未解析 |
| `neither` | $P_D$ | +0.0000 | [+0.0000, +0.0000] | 未解析 |
| | 弱目标 $P_D$ | +0.0000 | [+0.0000, +0.0000] | 未解析 |
| | 远程报告 | −0.0500 | [−0.1300, +0.0200] | 未解析 |

---

## 4. 结论

1. **在受测配置下，移除任一项、乃至两项同时移除，对检测性能均无可检出的影响。**
   弱目标 $P_D$ 在四个配置中完全相同；$P_D$ 最大差仅 0.001。
2. 移除两项后远程报告数**下降**（0.740 → 0.690），即通信成本略降而检测不变；
   但该差异**未解析**（区间覆盖 0），不能宣称为真实收益。
3. 因此**不能宣称两项"必要"**——这与论文现有口径一致，并且现在有了受控消融支撑。
4. **同时也不能据此外推为"两项无用"或删除公式**，理由见下节边界。

### 边界（必须与结论同时出现）

- **样本量**：MC=100 单配置，区间普遍覆盖 0 属"未检出差异"，**不等于等价**。
- **配置单一**：只在 `target-local-v1`（15 UAV / 10 target / $\lambda_c=0.005$ /
  $\rho=0.8$）下测过。两项的弱目标优先与缺口惩罚应在**弱目标占优、检测未饱和**
  的场景才起作用；受测场景 $P_D\approx0.975$ 接近饱和，可能恰好落在两项不敏感的区间。
- **`no_softmin` 的语义不纯**：关闭该开关时，除第一项从 soft-min 换成 $\sum_q\bar P_{D,q}$
  外，`target_alpha` 的边际权重尺度也随之改变（启用时含因子 $Q$，关闭时不含）。
  因此该变体不是"纯净地移除一个加项"，其解释需谨慎。
- 主结果（MC=1000）与本消融（MC=100）**口径不同，不能混算**。本消融属诊断性证据。

### 对论文的建议

**不删除**两个效用项，但把措辞从"未做消融、不宣称必要"升级为**如实报告本消融**：
在受测配置下两项均未检出必要性。若要主张"可删除"，需要更大样本 + 弱目标压力场景
（低 RCS / 大部署 / 高预测误差）下的复测。

---

## 5. ⚠️ 与主结果的口径差异（重要）

本次消融显式使用 `detect.comm_error_model=gaussian_replacement`，因为
`--preset target-local-v1` 会把该字段设为 `erasure`，而 `erasure` 的**语义在
2026-09-14 → 09-16 之间被改变过**：

- 源版本 `69f3300` 的 `soft_channel.py:142`：`erasure` 分支返回
  $\mathcal N(0,\sigma_{\rm scale}\sqrt{v_0})$（**高斯替代**，$\sigma_{\rm scale}=3$
  → 方差 $9v_0$ → $a=9$）；
- 当前代码：`erasure` 返回 `0.0`（**真零擦除**），而高斯行为移到
  `gaussian_replacement` 分支。

所以**现有结果 CSV 里记录的 `erasure` 字段值，对应的行为是高斯的**。

详见 `MANUSCRIPT_CODE_CONSISTENCY_ALERT.md`（判决阈值更换导致的数字漂移）。
