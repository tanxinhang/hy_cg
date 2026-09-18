# 协调辐射：接线 + 实测（2026-09-18）

## 0. 一句话

把 `coordination` 接进 `run_method_on_trial` 之后，**不靠任何硬件增益**（G_hw = 0 dB、κ = 40 dB 全冻结），
在 600 m / RCS 0.1 m² 主场景上把 worst-target P_D 从 **0.6500 抬到 0.8050**，
同时**开销下降 13%**。纯协议 + 纯算法。

推荐工作点：**`coordination.enable=true` + `selector.max_tx_nodes=3`**。

---

## 1. 接线改了什么

`select_with_coordination` 一直是旁支：只有 `tools/*` 调它，`--mode main` 从不调，
所以 `sense_gate_by_active_tx` 在发布路径上**静默空转**（`model.py` 的
`gate_echo = sense_gate_by_active_tx and mask is not None`，发布路径不传 mask）。

改动（全部 config 门控，默认关闭）：

| 文件 | 改动 |
|---|---|
| `isac_sim/config.py` | 新增 `Coordination` 段（`enable=False`, `rounds=6`）；`Config` 加字段；`validate_config` 拒绝「开了协调但没开门控」 |
| `isac_sim/simulate.py` | 新增 `COORDINATION_C2F_METHODS`、`_coordinated_c2f_selection`、`_coordination_eval_tables`；`run_method_on_trial` 加 `geom` 参数并在链首插入协调分支 |
| `isac_sim/simulate.py` | `MethodResult` 加 `coordination_rounds / _converged / _n_tx`；per-trial 与 summary 均落盘 |
| `isac_sim/report.py` | 三个诊断列进入 `main.csv` |

不动点：`mask → 重建表 → 重选`，2–6 轮预算，重复 mask 判环并上报（不静默烧预算）。
**评估表用最终调度自身导出的掩码重建**，belief 模式的真值侧同样带掩码——
否则会出现"选择时干扰少、判决时干扰多"的不自洽。

### 回归守卫（必须每次重查）
`B_gate`（只开门控、不开协调）与 `A_off` 逐位比对：**7400 字段、0 处差异**。
即"门控单独存在仍是空操作"这一既有事实没有因为我接线而改变。
检查脚本：`tools/summarise_coordwire.py`。

---

## 2. 实测（MC=200，600 m / RCS 0.1，κ=40 dB，G_hw=0 dB，belief 模式，方法 `proposed_c2f_adaptive_pd`）

seed 2026，配对差相对 `A_off`：

| 臂 | P_D | worst P_D | P_FA | bits | 照明机数 | 收敛率 | P_D/kbit | 配对 ΔP_D |
|---|---|---|---|---|---|---|---|---|
| A_off（基线） | 0.6830 | 0.6500 | 0.0491 | 11165 | — | — | 0.0612 | — |
| B_gate（守卫） | 0.6830 | 0.6500 | 0.0491 | 11165 | — | — | 0.0612 | **0（bit-exact）** |
| C 协调 | 0.7515 | 0.7250 | 0.0476 | 11098 | 9.64 | 0.99 | 0.0677 | +0.0685 ± 0.0231 |
| H 协调+cap12 | 0.7675 | 0.7450 | 0.0474 | 10995 | 8.97 | 0.99 | 0.0698 | +0.0845 |
| D 协调+cap8 | 0.7905 | 0.7500 | 0.0499 | 10029 | 7.04 | 1.00 | 0.0788 | +0.1075 ± 0.0273 |
| G 协调+cap6 | 0.8155 | 0.7850 | 0.0494 | 9360 | 5.58 | 1.00 | 0.0871 | +0.1325 |
| I 协调+pen0.05 | 0.8305 | 0.8050 | 0.0484 | 9974 | 4.24 | 0.99 | 0.0833 | +0.1475 ± 0.0286 |
| F 协调+cap4 | 0.8335 | 0.7950 | 0.0493 | 9379 | 3.90 | 1.00 | 0.0889 | +0.1505 |
| **J 协调+cap3** | **0.8555** | **0.8050** | 0.0503 | **9674** | 2.96 | 1.00 | **0.0884** | **+0.1725 ± 0.0271** |
| K 协调+cap2 | 0.8600 | 0.8200 | 0.0502 | 10531 | 2.00 | 1.00 | 0.0817 | +0.1770 |
| M 协调+cap1 | 0.8645 | 0.8250 | 0.0512 | 13824 | 1.00 | 1.00 | 0.0625 | +0.1815 |

换种子复核（seed 7777，独立配对）：

| 臂 | P_D | worst P_D | bits | P_D/kbit | 配对 ΔP_D |
|---|---|---|---|---|---|
| 基线 | 0.6945 | 0.6600 | 10778 | 0.0644 | — |
| 协调+cap3 | 0.8520 | 0.8050 | 8934 | 0.0954 | **+0.1575 ± 0.0265** |

两个种子给出的 worst P_D 完全一致（0.8050），ΔP_D 0.1575 / 0.1725，同向同量级。

---

## 3. 机制解释（为什么单独协调不够）

- **协调单独（C）只把照明机从 ~15 压到 9.64**，能静音的没几台 ⇒ 只有 +0.0685。
  这正是"发布选择器把照明机铺满全部 UAV"的后果。
- **真正起作用的是"少照明 + 关掉其余发射机"的组合**：cap 越小，发射机越少，
  感知分母里的多机泄漏直接塌掉，P_D 单调上升。
- 收益来自**干扰分母**，不是更多观测：cap3 的观测数 22.5 < 基线 33.4，
  检测反而更好，开销还低 13%。

### 膝点为什么是 cap3
- cap2 比 cap3 高 0.0045 P_D，但开销 +9%、P_D/kbit 从 0.0884 掉到 0.0817。
- cap1 的 P_D 最高（0.8645），但单照明机⇒全部远端上报，开销 **+24% 于基线**，
  P_D/kbit 0.0625 ≈ 基线，效率收益归零。
- ⇒ **cap3 是"检测 + 开销"双目标的膝点**。

---

## 4. 代价与假设账本

| 项 | 状态 |
|---|---|
| `coordination.enable` / `coordination.rounds` | **free**（新增键，不在 94 个冻结键里） |
| `selector.max_tx_nodes` | **free** |
| `selector.tx_penalty` | **free**，但可用区间极窄：0.05 可用，**0.1 与 0.2 直接把选择压成空调度**（`selection.py:531` 的保护） |
| `interference.sense_gate_by_active_tx` | **FROZEN** ⇒ 协调要付的假设是"**非照明机在感知观测量期间不辐射**"，这是**协议假设，不是硬件假设** |

计算代价：选择器评分次数 49,665 → 101,403（cap3，约 2×），来自 2–4 轮不动点重选。
换来的是开销 −13%，所以净开销是降的。

---

## 5. 写进论文前必须做的

1. **MC 提到 1000**（现在是 200）。发布口径是 MC=1000，本文所有数字需重跑一遍再引用；
   两个种子的一致性能保证方向，但绝对数要按发布口径出。
2. **声明协议假设**：非照明机静默。要和 κ（直连消除）分开列，别混成"算法免费"。
3. **别和 0.9500 那个数并列**：`tools/run_coordination_experiment.py` 历史上报的 0.85→0.95
   是工具路径、不同口径；本文的 0.8050 是发布入口、发布方法名。跨路径不可比。
4. **基线也要跑同一口径**：论文对比表里所有基线应在「门控开 + 协调关」还是「全关」下跑，
   需明确——本文基线是「全关」（= 当前发布值）。

---

## 6. 复现命令

```bash
# 基线（发布路径）
python -m isac_sim --mode main --preset small-uav-compact-800m \
  --set geometry.area_xy=600 --set detect.target_rcs=0.1 \
  --mc 1000 --methods proposed_c2f_adaptive_pd --out results_coord_release_base

# 推荐工作点：协调 + 3 台照明机
python -m isac_sim --mode main --preset small-uav-compact-800m \
  --set geometry.area_xy=600 --set detect.target_rcs=0.1 \
  --set interference.sense_gate_by_active_tx=true \
  --set coordination.enable=true --set selector.max_tx_nodes=3 \
  --mc 1000 --methods proposed_c2f_adaptive_pd --out results_coord_release_cap3

# 汇总（含 bit-exact 回归守卫）
python tools/summarise_coordwire.py
```
