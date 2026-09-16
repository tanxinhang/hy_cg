# 系统性能现状（一页总览）

> 生成时间：2026-09-16 21:50
> 口径：`target-local-v1` / `paper-canonical`，**`comm.interference_model = orthogonal`**（论文口径，见 `results_target_local_v1/main/config.json`）
> 数据来源：`results_target_local_v1/main/main.csv`（MC=1000）、`results_v1_rcs_joint/rcs_summary.csv`、
> `results_v1_rcs_wide/summary.json`、`results_v1_lever_closure/closure.json`、
> `results_v1_lowrcs_cheap/`、`results_v1_lowrcs_shortfall/`
> 相关文档：`RCS_IMPACT_ANALYSIS.md`、`DEPLOYMENT_RANGE_ANALYSIS.md`、`PERFORMANCE_OPTIMIZATION_ROADMAP.md`、
> `LOW_RCS_EVIDENCE_RESCUE.md`、`MOBILE_GEOMETRY_FEASIBILITY.md`

---

## 0. 一句话

**论文工作点（4 km / RCS ≥ 20 m²）上系统是达标且通信极省的；但换到小目标（RCS ≤ 0.2 m²）就全面不达标，
而这不是算法问题，是双站 R⁴ 链路预算问题。** 系统当前是**干扰受限**的，所以"加功率/换低噪放"这类
最直觉的优化买不到任何东西，唯一有效旋钮是硬件净增益（~17.5–20 dB）与加深直射对消。

---

## 1. 论文工作点性能（MC=1000, seed=2026, 4 km / RCS 50 m² / 15 UAV / 10 目标 / ρ=0.8）

| 方法 | P_D | 95% CI | worst-target | P_FA | 报告数 | 时延 (ms) | 比特 |
|---|---:|---|---:|---:|---:|---:|---:|
| **proposed_c2f_adaptive_pd** | **0.9764** | [0.9732, 0.9792] | **0.966** | 0.0487 | **0.751** | **0.80** | 481 |
| exact_marginal_greedy | 0.9408 | [0.9360, 0.9453] | 0.931 | 0.0493 | 5.131 | 5.47 | 3284 |
| sense_sinr | 0.9784 | [0.9754, 0.9811] | 0.972 | 0.0494 | 4.595 | 4.90 | 2941 |

逐 trial 配对：

- vs `exact_marginal_greedy`：**Δ = +0.0356**，CI [0.0311, 0.0401] —— **不含 0，显著**
- vs `sense_sinr`：**Δ = −0.0020**，CI [−0.0050, +0.0010] —— **含 0，统计上不可分**

**读法（也是论文的唯一正确卖点）**：检测性能**与最强的感知 SINR 启发式持平**，但代价降到它的 **1/6**
（报告 0.751 vs 4.595 条、时延 0.80 vs 4.90 ms、比特 481 vs 2941）。
**卖点是通信效率，不是检测增益。** 说"提升了检测性能"会被 `sense_sinr` 那一行反驳。

其他：`all_targets_satisfied_prob = 0.726`、`worst_target_satisfied_prob = 0.955`、
belief 捕获率 0.994、`selected_observations = 12.101`、`fine_eval` 61.278 vs 860.244（共同基线）。

✅ **本表已于 2026-09-16 晚按当前冻结代码重跑（MC=1000）**。与重跑前的 09-14 旧判决结果相比，
$P_D$ 变动 ≤0.0003，选路面量（报告数/时延/观测数/精细评估数）**逐位不变**，配对差值显著性方向未变。
重跑前快照：`archive/results_target_local_v1_pre_rngfix_2026-09-16/`；
核对工具：`tools/report_v1_rerun_drift.py`。见 `SUBMISSION_TRACEABILITY.md` §7。

---

## 2. 泛化性：RCS 轴（4 km，K=0 无远程报告）

| RCS (m²) | 0.05 | 0.1 | 0.2 | 1 | 5 | 20 | 50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| P_D | 0.083 | 0.113 | 0.167 | 0.434 | 0.769 | 0.940 | 0.981 |

- 单调 S 形；**P_D ≥ 0.95 需要 RCS ∈ (20, 50] m²**。
- **协同增益是倒 U 形**：Δ(K=8 − K=0) = +0.010 / +0.023 / +0.035 / **+0.037** / +0.032 / +0.007 / −0.001，
  峰值在 1 m²。低端塌陷是证据太弱，高端塌陷是**已经饱和**——调度器主动把报告数从 7.91 节流到 0.85。
  ⇒ **写论文时不能笼统说"RCS 越高协同越有用"**。
- `rcs_model = "mean"`，**未启用 Swerling / 方位起伏** ⇒ 真实小目标的 worst-case 只会更差。

## 3. 泛化性：几何轴（4 km → 600 m）

- 净赚 **+9 ~ +13 dB**（换算法不同：按 P_D≥0.95 门槛 +8.9；按逐点等效 +13.1；机制上界 +13.6）。
  回波 +24.9 dB − 直射干扰 +11.3 dB = +13.6；差额 14 dB 被"收缩同时把干扰几何也缩近"吃掉。
- 把达标所需 RCS 从 ≈21.7 m² 降到 ≈2.81 m²；但 **RCS 0.2 m² @ 600 m 仍只有 0.700**。
- 距离敏感性**依赖 RCS**：0.2 m² 时 600→4000 m 掉 0.500；50 m² 时整个跨度只动 0.009。
- **建议保留 4 km 基线**：600 m + RCS 50 m² 会退化成纯本地感知（报告 0.00 / 观测 10.00 / 时延 0.00），
  即"协同有用"与"性能可接受"在当前模型里几乎不相交。

---

## 4. 小目标性能（零天线增益的物理诚实口径）

| 场景 | 平均 P_D | 弱目标 P_D |
|---|---:|---:|
| `small-uav-compact-800m`（0.8 km / 0.05 m²） | 0.437 | 0.433 |
| 400–600 m / RCS 0.05–0.2，软件旋钮最好一档<br>（`maxmin_looks64`） | 0.728 | **0.049** |
| 基准 `base` 同场景 | 0.522 | **0.007** |
| 加 15 dB **硬件**净增益（`radar_net_gain_db=15`） | 0.932 | 0.476 |

- **shortfall 定量**：400 m / RCS 0.05 的**最差目标**距 `weak_pd_required = 0.80` 还差
  **+7.90 / +5.26 / +4.27 dB**（looks 16 / 64 / 128 感知 SINR）。
- **`lossless_heldout_worst_pd ≡ heldout_worst_pd`（oracle headroom 恒 0）**
  ⇒ 瓶颈在**证据形成**，不在传输/融合。这一条直接否掉"靠融合补弱回波"的整条路线。
- **没有任何软件旋钮能满足 0.80 弱目标要求**；报告预算轴也用尽（K=0→10 只把平均从 0.450 拉到 0.495）。
- 净增益转换区在 **10–20 dB**，要让 weak ≥ 0.80 需 **~17.5–20 dB**；且该工作点 `P_FA = 0.081` 已不合格。

⚠️ **引用陷阱**：`results_small_uav_s2_*/main/main.csv` 里那个漂亮的 `P_D = 0.984` 是
**已被否决的 27 dB 回归桥**（`RADAR_LINK_BUDGET_CALIBRATION.md` 明确标注 superseded），
**不是小目标的物理性能**，不得引用。

---

## 5. 为什么优化不动：杠杆的饱和性

感知 SINR 的组装（`model.py:778-784`）是

```
signal   = ρ·P·target_gain·G_proc·G_hw
residual = f_self·P + κ_dc·I_direct(P) + f_multi·P·G_direct
gamma    = signal / ((n0 + residual + eps)·(1 + INR))
```

**`residual ∝ P` 而 `n0` 不变** ⇒ `gamma(mP)/gamma(P) = m(1+r)/(1+mr)`，`r ≫ 1` 时趋于 1。

实测 `r = residual/n0`：400 m **+9.9 dB**、600 m **+8.0 dB** ⇒ **干扰受限**；
残留干扰 **96–97% 来自直射对消残差**（n = 9620 条链路，用"加深 20 dB 再算一次"反推）。

**实测闭合表**（gap = 最弱目标抬到 0.80 还差多少每链路 γ dB；base = 7.90，与存档 7.898 逐位吻合 ✓）

| 杠杆 | 买到 dB (400 m / 0.05) |
|---|---:|
| `kappa_dc` 40 → 50 dB | **4.27** |
| `kappa_dc` 40 → 60 dB | **5.47**（再深到 80 只多 0.17 ⇒ 用尽） |
| `G_proc` ×4 / ×16 / ×64 | 3.82 / 6.46 / 7.33 |
| `G_hw` +10 / +15 / +20 dB | 3.23 / 4.62 / 5.85 |
| `n_looks` 16 → 64 / → 256 | 2.16 / 4.05 |
| `P_default` ×100 | **0.31** |
| NF 7 → 0 dB | **0.25** |
| `kappa60 + P×100` | 7.13（gap → 0.77） |
| `G_proc×4 + kappa60 + looks64` | 7.53（gap → **0.37**，几乎闭合） |

**杠杆分四族，这个区分比"硬件 vs 算法"更重要**：

| 族 | 成员 | 是否饱和 |
|---|---|---|
| **分子类** | RCS、几何、`G_proc = N·L`、`G_hw` | **否，精确线性 ×m** |
| **分母类** | `P_default`、`noise_figure_db` | **是**（`r≫1` 时无效） |
| **减分母类** | `kappa_dc`、照明机调度 | 减到 `r < 1` 后停 |
| **样本数类** | `detect.n_looks` | 以 **√L** 进 d′ ⇒ CPI ×4 只等效 γ +3 dB |

**"降干扰解锁功率"已实测为真**：`kappa60` 单独买 5.47；再叠 `P×100` 又多买 1.66；而在 `kappa=40`
（干扰受限）时 `P×100` 只买 0.31。⇒ **优化是有序的：先把 `r` 压到 1 以下，功率才复活。**

**算法侧已经到顶**：`maxmin` 是唯一有效旋钮（+0.206）；budget 轴用尽；`corr` 负收益；
lossless oracle headroom ≡ 0。**算法只能重分配已有预算，造不出预算。**
唯一剩余空间：让选择器把"照明机 `i` 的辐射对其它接收机的干扰"计入代价（**当前是否已含该耦合未验证**）。

**移动不是自由度**：`generate_geometry` 一次性撒点、此后位置不变（速度只进 OTFS 多普勒与 belief 预测）。
快照模型下"移动"≡"一开始就摆在那"，单独增益恒为 0，有增益的是**位置**。
读作"派 2 架双站机"时是一条真路线：最好配置（dispatch 80 m + κ_dc ≥ 80 dB）给 **+32.0 dB**，
即几何最多供给 **约 +19 ~ 20 dB**，**只能补上小目标缺口（−30 dB RCS）的三分之二**，且用的是
"单目标最佳视角"这个乐观统计量。

---

## 6. 结论与行动顺序

**判定**：论文工作点达标且通信极省（0.9764 / 0.751 报告 / 0.80 ms）；
小目标不达标是**链路预算**问题，不是算法问题。

**要做的事，按优先级**：

1. ✅ **P0 已完成（2026-09-16 晚）**：MC=1000 主实验已按当前冻结代码重跑，
   `SimulationResults.tex`、摘要、`Conclusion.tex`、补充材料与 fig2/3/4 已同步。
   实测漂移仅 +0.0002（早前 MC=100 的 −0.007 读数本身在噪声内，高估了量级），
   配对差值显著性方向未变。见 `SUBMISSION_TRACEABILITY.md` §7。
2. **优化路线（若要救小目标）**：
   - 第一优先 `kappa_dc`：40 → 60 dB，买 5.47 dB，**且是唯一能让功率杠杆复活的杠杆**；
     但需先解决口径张力（`config.py:299` 声称对消到回波量级，实测残留直射比 raw 回波高约 47 dB）。
   - 其次 `G_proc`（用 `detect.sensing_processing_gain` 覆盖，**不要改 `waveform.N/L`**，
     否则会动 belief 的 DD 分辨率与 RNG 流）。
   - `G_hw` 目前没有映射到任何真实天线/EIRP 设计，引用前必须补标定论证。
   - **不要写** `P_default` 与 `noise_figure_db` 两条（各 0.31 / 0.25 dB，会被审稿人当笑话）。
3. **诚实边界**：第 5 节的 dB 是 **LLR 矩的解析界**（对角协方差高斯近似），
   **不是 Monte-Carlo 的 P_D**，只能用于排序与定量缺口，不得当作检测性能宣称值。

**未收口项**（不影响本次性能结论，但投稿前需知道）：机制 B（63 格基线漂移）仍未解释；
融合栈三处逻辑不自洽；系统是单圈闭环、缺时间划分 τ；`distributed_bids` 从未被调用。

---

## 7. 复现

```bash
# 论文主结果（MC=1000，约数小时，务必 run_in_background）
python tools/rerun_target_local_v1.py --out results_target_local_v1

# RCS 轴（4 km，K∈{0,8}，MC=100，12 workers ≈ 7.5 分钟）
python tools/audit_v1_rcs_wide.py --out results_v1_rcs_wide

# 杠杆闭合（mc=40/8 workers ≈ 3.5 分钟）
python tools/probe_lever_closure.py --mc 40 --workers 8 \
    --areas 400 600 --rcs 0.05 0.1 0.2 --out results_v1_lever_closure
```
