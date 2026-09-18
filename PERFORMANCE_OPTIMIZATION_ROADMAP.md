# 性能优化路线：杠杆的饱和性分析

> 生成时间：2026-09-16
> 复现：`python tools/probe_lever_closure.py --mc 40 --workers 8 --areas 400 600 --rcs 0.05 0.1 0.2 --out results_v1_lever_closure`
> 数据：`results_v1_lever_closure/closure.json`、`results_v1_lever_closure/links.json`
> 相关：`LOW_RCS_EVIDENCE_RESCUE.md`（旋钮实测）、`RADAR_LINK_BUDGET_CALIBRATION.md`（工作点标定）

---

## 0. 一句话结论

**当前系统是干扰受限的（残留干扰约为接收机噪声的 8–10 dB），所以"加大发射功率"和"换更好的接收机"这两条最直觉的路几乎买不到任何东西。**
能闭合缺口的只有两类杠杆：**降低分母**（直射对消深度 `kappa_dc`）与**放大分子**（相干处理增益 `G_proc`、硬件净增益 `G_hw`、RCS、几何）。其中 `kappa_dc` 单位代价回报最高，且**降干扰之后功率杠杆才会复活**。

⚠️ 本文所有"需要多少 dB"都是 **LLR 矩的解析界**（对角协方差高斯近似），不是 Monte-Carlo 的 $P_D$。它用来给杠杆排序和给缺口定量，不能当作检测性能的宣称值。

---

## 1. 诊断：为什么"加功率"不管用

感知 SINR 在 `isac_sim/model.py:778-784` 的组装是：

```
signal   = rho * P * target_gain * G_proc * G_hw * capture
residual = f_self*P + kappa_dc * I_sense_field(P) + f_multi*P_sense*G_direct
gamma    = signal / ((n0 + residual + eps) * (1 + INR))
```

关键结构：**`residual` 与 `signal` 一样正比于发射功率 `P`，而 `n0` 不变。** 因此提高功率会同时抬升信号与干扰，只有噪声项被相对压低：

$$\frac{\gamma(mP)}{\gamma(P)} = m\cdot\frac{1+r}{1+mr},\qquad r \equiv \frac{\text{residual}}{n_0+\epsilon}$$

- $r \ll 1$（噪声受限）⇒ 比值 $\to m$，功率全额兑现；
- $r \gg 1$（干扰受限）⇒ 比值 $\to 1$，功率完全无效。

**实测 $r$（`LinkTables.rinr`，选中链路上，mc=40）：**

| 场景 | $r$ 中位数 | $r$ p90 | 判读 |
|---|---:|---:|---|
| 400 m / RCS 0.05 | +9.9 dB | +13.4 dB | interference-limited |
| 400 m / RCS 0.1 | +9.8 dB | +13.2 dB | interference-limited |
| 400 m / RCS 0.2 | +9.8 dB | +13.2 dB | interference-limited |
| 600 m / RCS 0.05 | +8.0 dB | +10.9 dB | interference-limited |
| 600 m / RCS 0.1 | +8.0 dB | +11.2 dB | interference-limited |
| 600 m / RCS 0.2 | +8.0 dB | +11.2 dB | interference-limited |

**残留干扰的构成**（把 `direct_cancellation_db` 加深 20 dB 再算一次反推得到 `dshare = 直射残差/(n0+eps)`，逐元素与 `rinr` 相除）：

| 场景 | n（链路数） | `dshare` 中位数 | `rinr` 中位数 | **直射残差占比**（逐元素中位数） |
|---|---:|---:|---:|---:|
| 400 m / 0.05 | 1744 | 9.59 | 9.85 | **97.4%** |
| 400 m / 0.1 | 1614 | 9.37 | 9.64 | **97.3%** |
| 400 m / 0.2 | 1416 | 9.36 | 9.62 | **97.3%** |
| 600 m / 0.05 | 1779 | 6.00 | 6.26 | **95.8%** |
| 600 m / 0.1 | 1643 | 5.97 | 6.24 | **95.8%** |
| 600 m / 0.2 | 1424 | 5.97 | 6.24 | **95.8%** |
| 全部 | 9620 | 7.77 | 8.03 | **96.8%** |

也就是说 **残留干扰有 96–97% 来自直射对消残差**，而它正好由 `kappa_dc` 一个参数控制（`residual_self_factor=1e-14`、`residual_multi_uav_factor=1e-10` 都比 `kappa_dc=1e-4` 小六个数量级以上，剩余那 3% 就是它们）。

**这解释了 `results_v1_lowrcs_cheap` 里那个反常现象**：`power2`（+3 dB 发射功率）与 `base` 的 P_D 几乎逐位相同（0.525 vs 0.522）——不是实现 bug，是物理必然。

---

## 2. 杠杆的数学分类

由上面的组装式，杠杆分成两个家族，**这个区别比"硬件 vs 算法"更重要**：

| 家族 | 作用的项 | 对 $\gamma$ 的效果 | 是否饱和 | 成员 |
|---|---|---|---|---|
| **分子类** | `target_gain · G_proc · G_hw` | 因子 $m$ ⇒ $\gamma\times m$，**精确线性** | 否 | RCS、几何、相干处理增益、硬件净增益 |
| **分母类** | `n0 + residual` | $\gamma \times m(1+r)/(1+mr)$ | **是**（$r\gg1$ 时无效） | 发射功率、接收机噪声系数 |
| **减分母类** | 只减 `residual_direct` | $\gamma \times \tfrac{1+r}{1+r'}$ | 部分是（减到 $r<1$ 后停止） | 直射对消深度、照明机调度 |
| **样本数类** | LLR 矩 | $d' = (\sqrt{K}A - z\sqrt{B})/\sqrt{C}$ ⇒ $\sqrt{K}$ | 是（指数减半） | `detect.n_looks`（CPI 长度） |

**第四行值得单独强调**：`n_looks` 与"给 $\gamma$ 乘因子"在数学上**不同构**。融合统计量是
$$d' = \frac{K\cdot A - z\sqrt{K\cdot B}}{\sqrt{K\cdot C}} = \frac{\sqrt{K}\,A - z\sqrt{B}}{\sqrt{C}}$$
对 $\sqrt{K}$ 线性，而 $\gamma$ 的乘性因子 $k$ 满足 $d' = (kA - z\sqrt B)/\sqrt C$。**两者形式一致，故 $\sqrt{K} \equiv k$：CPI 长度翻 4 倍（+6 dB 样本）只等效于 $\gamma$ +3 dB。**

---

## 3. 实测闭合表

"gap" = 把**最弱目标**抬到 `weak_pd_required = 0.80` 还差的每链路感知 SINR 增益（dB）。聚合口径与 `probe_lowrcs_shortfall` 一致（先按目标跨 trial 求均值，再取最坏目标），因此 **base 行可与已存档的 `results_v1_lowrcs_shortfall/shortfall_summary.json` 逐位对照**——实测 `7.90` vs 存档 `7.898`，口径自洽 ✓

| 杠杆 | 400/0.05 | 400/0.1 | 400/0.2 | 600/0.05 | 600/0.1 | 600/0.2 | 买到 dB<br>(400/0.05) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **base** | **7.90** | 5.52 | 3.53 | 7.50 | 4.87 | 3.20 | — |
| — 分母类（几乎无效）— | | | | | | | |
| P_default ×2 | 7.75 | 5.37 | 3.39 | 7.26 | 4.66 | 2.99 | 0.15 |
| P_default ×10 | 7.62 | 5.25 | 3.28 | 7.05 | 4.47 | 2.84 | 0.28 |
| P_default ×100 (+20 dB) | 7.59 | 5.22 | 3.25 | 7.00 | 4.43 | 2.80 | **0.31** |
| NF 7 → 3 dB | 7.72 | 5.34 | 3.36 | 7.21 | 4.61 | 2.95 | 0.18 |
| NF 7 → 0 dB | 7.65 | 5.28 | 3.31 | 7.10 | 4.52 | 2.88 | **0.25** |
| — 减分母类 — | | | | | | | |
| **kappa_dc 40 → 50 dB** | 3.63 | 1.69 | 0.61 | 4.04 | 1.85 | 0.78 | **4.27** |
| **kappa_dc 40 → 60 dB** | 2.43 | 0.91 | 0.36 | 3.25 | 1.18 | 0.50 | **5.47** |
| kappa_dc 40 → 80 dB | 2.26 | 0.83 | 0.33 | 3.14 | 1.12 | 0.49 | 5.64 |
| — 分子类 — | | | | | | | |
| G_proc ×2 | 5.91 | 3.67 | 1.94 | 5.59 | 3.17 | 1.68 | 1.99 |
| G_proc ×4 | 4.08 | 2.11 | 0.86 | 3.89 | 1.72 | 0.83 | 3.82 |
| G_proc ×16 | 1.44 | 0.82 | 0.27 | 1.36 | 0.60 | 0.29 | 6.46 |
| **G_proc ×64** | 0.57 | 0.34 | 0.04 | 0.54 | 0.26 | 0.04 | **7.33** |
| G_hw +10 dB | 4.67 | 2.62 | 1.07 | 4.45 | 2.19 | 1.05 | 3.23 |
| G_hw +15 dB | 3.28 | 1.57 | 0.61 | 3.16 | 1.29 | 0.62 | 4.62 |
| G_hw +20 dB | 2.05 | 1.08 | 0.37 | 2.02 | 0.80 | 0.41 | 5.85 |
| RCS ×4 (+6 dB) | 4.08 | 2.11 | 0.86 | 3.89 | 1.72 | 0.83 | 3.82 |
| RCS ×10 (+10 dB) | 2.05 | 1.08 | 0.37 | 2.02 | 0.80 | 0.41 | 5.85 |
| — 样本数类 — | | | | | | | |
| n_looks 16 → 64 | 5.74 | 3.53 | 1.83 | 5.42 | 3.04 | 1.57 | 2.16 |
| n_looks 16 → 256 | 3.85 | 1.93 | 0.78 | 3.67 | 1.59 | 0.76 | 4.05 |
| — 组合 — | | | | | | | |
| **kappa 60 + P ×100** | **0.77** | 0.44 | 0.09 | 0.95 | 0.46 | 0.20 | **7.13** |
| kappa 60 + P×100 + NF 3 | 0.76 | 0.43 | 0.09 | 0.94 | 0.46 | 0.20 | 7.14 |
| kappa 60 + NF 3 | 1.31 | 0.61 | 0.24 | 1.96 | 0.64 | 0.37 | 6.59 |
| kappa 60 + n_looks 256 | 0.59 | 0.35 | 0.05 | 0.85 | 0.41 | 0.16 | 7.31 |
| G_proc ×4 + kappa 60 | 0.66 | 0.39 | 0.07 | 0.93 | 0.45 | 0.20 | 7.24 |
| **G_proc ×4 + kappa 60 + looks64** | **0.37** | 0.21 | 0.00 | 0.57 | 0.27 | 0.06 | **7.53** |

### 三条读数

1. **`kappa_dc` 的边际回报在 40→60 dB 用尽**（买 5.47 dB），再加深 20 dB 只多买 0.17 dB。原因：`r` 从 +10 dB 降到 0.35（<1），系统已经离开干扰受限区，剩下的分母就是 `n0` 本身，减不动了。
2. **"降干扰解锁功率"为真**：`kappa 60` 单独买 5.47 dB，再叠 `P×100` 又多买 1.66 dB（3.94 → 2.23 在 mc=3 预跑中；正式表 2.43 → 0.77 = 1.66）。功率在 `kappa=40` 时只买 0.31 dB。**这就是有序优化的全部内容：先把 `r` 压到 1 以下，再谈功率。**
3. **`gamma` 的乘性增益与 gap 的下降不是 1:1**：`G_proc ×4` 给 $\gamma$ +6.02 dB，但 gap 只降 3.82 dB。原因是 `required_gain` 在新的 $\gamma$ 水平上重解，而 $d'$ 对 $\gamma$ 是凹的（$\sqrt{\cdot}$ 型，低 $d'$ 区更明显）。**引用时务必区分"$\gamma$ 买了多少 dB"和"缺口降了多少 dB"。**

---

## 4. 可执行的优化路径

按"单位代价回报"排序，且区分**能不能在论文里辩护**：

| # | 动作 | 买到 dB (400/0.05) | 代价 / 物理含义 | 可辩护性 |
|---|---|---:|---|---|
| 1 | `kappa_dc` 40 → 50 dB | 4.27 | 直射对消深度再深 10 dB | 需要论证接收机对消能力 |
| 2 | `kappa_dc` 40 → 60 dB | 5.47 | 再深 20 dB（模拟+数字两级对消） | 同上，60 dB 属文献可及但需引证 |
| 3 | `G_hw` +10 dB | 3.23 | 天线增益 / EIRP / 系统损耗 | **已是被文档标为"calibration bridge"的待标定量** |
| 4 | `G_proc` ×4 | 3.82 | 相干积累 4×（更长 CPI 或更多子载波/带宽） | 需说明时间/带宽/算力代价 |
| 5 | `n_looks` 16 → 64 | 2.16 | CPI 长度 ×4 | 物理参数，已在模型内 |
| 6 | `G_proc` ×64 | 7.33 | 64× 相干积累 | 代价高但**单杠杆可闭合** |
| ✗ | `P_default` ×100 | 0.31 | 100× 发射功率 | **不要写** |
| ✗ | NF 7 → 0 dB | 0.25 | 理想接收机 | **不要写** |

**最小闭合组合**（gap ≤ 0.5 dB）：

- `G_proc ×4 + kappa_dc 60 dB + n_looks 64` → **0.37 dB**（400/0.05）
- `G_proc ×64` 单条 → 0.57 dB
- `kappa_dc 60 dB + P ×100` → 0.77 dB

### 关于 `detect.sensing_processing_gain`

`isac_sim/model.py:724` 的 `G_proc` 可被 `detect.sensing_processing_gain` 显式覆盖，**绕过 `waveform.N × waveform.L`**。
这一点在优化上有实际意义：直接改 `waveform.N/L` 会同时改变 `belief.py:107-108,171-173` 里的 DD 分辨率（`L·Δf` / `N·T`），进而改动预测误差与几何；而用这个覆盖键只改感知 SINR 的分子，**分辨率与 RNG 流都不动**。做"处理增益敏感性"实验时应当用它，不要改 `waveform`。

---

## 5. 算法侧：已经到顶

以下结论来自已存档的实测，**本轮未重复测量**：

| 旋钮 | 实测结果 | 判读 |
|---|---|---|
| `maxmin` 分配 | 平均 P_D 0.522 → 0.728（叠加 looks64），但 weak target 仅 0.007 → 0.049 | 唯一有效的算法旋钮，但幅度远不够 |
| `selector budget`（K=0→10） | 平均 0.450 → 0.495，weak 最高 0.400 | 报告预算轴已用尽 |
| `corr.enable` | 0.504 vs base 0.522 | **负收益**，且该结论在 `orthogonal` 下测的 |
| `capacitated` 融合 | 与 `nearest_target` 无分离 | — |
| lossless-report oracle | headroom ≡ 0 | 瓶颈在**证据形成**，不在传输/融合 |
| `sense_gate_by_active_tx` | 乐观上界 +2.77 dB | 调度门控是减分母类，但幅度小 |

**推论**：算法侧的动作空间是"重新分配已有预算"，而缺口是**绝对预算缺口**。这解释了为什么所有软件旋钮都停在同一个地方。**唯一的算法侧剩余空间在"减分母"上**——即让选择器把"照明机 `i` 的辐射对其它接收机的干扰"计入代价。当前 `select_c2f_adaptive` 是否已包含这一耦合，本文**未验证**，列为待查项。

---

## 6. 诚实边界

1. **解析界，不是 MC**。`d'` 由 LLR 矩按对角协方差高斯近似算出；`corr.enable` 会引入非对角项，故本表对 `corr` **乐观**。低每链路 SINR 下高斯尾部也是近似。**这张表用于排序与定量缺口，不得当作 $P_D$ 结果引用。**
2. **$\kappa_{dc}$ 是干扰抑制，不是信号增益，但它仍然是假设**。`kappa_dc` 只乘在感知分母的直连泄漏项上（`model.py:654,765`），不碰回波项；其物理依据是协作 ISAC 已知照明机波形、可重建并相减，残余由信道估计精度决定。**已实测的量级**（`tools/probe_kappa_necessity.py`，生产链路表口径）：近远比是**几何量**——legacy 论文几何（4 km / RCS 50 m²）为 41.3 dB，当前主场景（600 m / RCS 0.1 m²）为 **50.9 dB**。因此
   * 在当前主场景，40 dB **不是**"对消到回波量级"：残余直连仍比处理后回波高约 **11 dB**；
   * ~51 dB 才对应"到回波量级"，**60 dB 只让残余比回波低 2.4 dB**（且已触自残留/噪声地板，SINR −2.4 dB）；
   * 所谓"提到 60 dB 意味着残留比回波低 18.7 dB"是拿 legacy 的 41.3 dB 算出来的，**在当前几何下不成立**。
   结论方向不变（改 κ 要付假设代价并重新论证对消能力），但 40→60 在当前场景是**从"不够"补到"刚好"**，不是从"刚好"跳到"激进"。
3. **`G_hw` 目前没有映射到任何真实天线/EIRP 设计**。`RADAR_LINK_BUDGET_CALIBRATION.md` 自己写的是 "calibration bridge, not a validated platform"。
4. **本文只算了单目标视角的 SINR 缺口换算**，最坏目标的口径已对齐系统定义（先按目标平均再取最坏），但**跨目标的耦合（报告预算共享）在解析器中未建模**。
5. **已澄清：口径张力来自两个叠加的错误，不是措辞问题**（2026-09-18 追到底）。
   * "echo" 指**已含 `N*L` 相干积累的处理后回波**，不是 raw 回波。实测 600 m / RCS 0.1：`echo/n0 = -2.3 dB`，而 raw 回波约比 `n0` 低 37 dB——差的正是 `G_proc = 36.1 dB`。
   * 近远比 41.3 dB 是在 **legacy 几何（4 km / RCS 50 m²）**标定的，它是几何/RCS 的量，**搬到 600 m / RCS 0.1 后实测为 50.9 dB**。
   两者叠加才算出那个虚假的 47 dB。正确读数：κ=40 时 `residual/n0 = +8.7 dB`，而处理后回波 `= -2.3 dB` ⇒ 残余直连比（处理后）回波高 **11.0 dB**，与"50.9 − 40 = 10.9 dB"自洽。详见 `config.py` 中 `direct_cancellation_db` 的新注释。

---

## 7. 复现

```bash
# 完整表（约 3.5 分钟，mc=40/8 workers）
python tools/probe_lever_closure.py --mc 40 --workers 8 \
    --areas 400 600 --rcs 0.05 0.1 0.2 --out results_v1_lever_closure

# 快速自洽校验：base 行应复现 shortfall 存档的 7.898 (400/0.05)
python -c "import json;d=json.load(open('results_v1_lowrcs_shortfall/shortfall_summary.json'));print(d['area400_rcs0.05_looks16']['required_gain_db_worst'])"
```

设计要点（改动时勿破坏）：
- 一次 MC 记录每条选中链路的 `gamma` / `rinr` / `dshare`，**所有杠杆在闭式后处理里应用**——因为分子类精确线性、分母类有闭式，不需要为每个杠杆重跑仿真；
- `dshare` 用"把 `direct_cancellation_db` 加深 20 dB 再算一次"反推，不假设 `self`/`multi` 项的相对大小；
- `apply_steps` 显式跟踪 `d_cur`（当前噪声底）与 `res`（残留干扰绝对值）。**不要退回单变量 `tot` 递推**：噪声系数杠杆会改变噪声底，混在一起会把组合路径算成负分母（已发生过一次，表现为 gap 打印 `999.90` = inf）。

---

## 8. 增益入口的闭合性：系统里还有没有"额外增益"？（2026-09-18 增补）

**问题**：在 500–800 m / RCS 0.05–0.2 带内，缺口大到让 `P_D^req=0.95` 只剩 `G_hw` 一条路。那么系统里**还有没有没被开采的增益**？本文的回答是：入口可穷举，未开采的只有 4 个，其中**只有 1 个是真增益**。

### 8.1 入口是可穷举的，不是"再试试别的旋钮"

§1 的组装式就是 `isac_sim/model.py:743-828` 的全部物理。任何"增益"只能从下面 11 个位置进入；其余 150+ 个配置字段**不在链路上**，怎么改都动不了 $\gamma$ 与 LLR 矩：

| 位置 | 代表键 | 家族（§2） |
|---|---|---|
| 感知功率占比 `rho` | `radio.rho` | 分子 |
| 发射功率 `P` | `radio.P_default` | 分母（饱和） |
| `target_gain` | RCS、几何、`path_loss_exp`、`shadow_std_db`、`rician_K_db` | 分子 |
| `G_proc` | `waveform.N·L` 或 `detect.sensing_processing_gain` | 分子 |
| `G_hw` | `radio.radar_net_gain_db`（或 tx/rx 增益与系统损耗） | 分子 |
| `collision_penalty · dd_loss` | `dd.dd_collision_alpha`、DD 有效性 | 分子 |
| `waveform_capture` | `waveform_impairments.*`（开起来只会 ≤1） | 分子 |
| `n0` | `noise_figure_db`、带宽 | 分母（饱和） |
| 三项残差 | `interference.direct_cancellation_db`、`residual_*_factor`、照明机调度 | 减分母 |
| `eps` | `radio.eps_mode` / `eps_rel_db` | 分母护栏 |
| 样本数 / 相关 | `detect.n_looks`、`refine.*`、`corr.*` | 样本数 |

⇒「还有额外增益吗」**等价于**「这 11 个位置里哪些还没开采」，可穷举，不必逐个试旋钮。

### 8.2 新工作点（600 m / RCS 0.1）上只有 4 个位置未开采

实测（`tools/probe_lever_closure.py`，解析界、mc=24、`--areas 600 --rcs 0.1`；base gap = **5.62 dB**）：

| 未开采入口 | gap 变为 | 买到 | 判读 |
|---|---:|---:|---|
| **垂直几何降到 h 200–500 m** | **2.79** | **2.83 dB** | ✅ 真增益，且**免费** |
| `radio.rho` 0.8 → 1.0 | 4.97 | 0.65 | ⚠️ 不免费：用 20% 通信功率换，通信代价未建模 |
| `isac_power_model=joint_waveform`（并发口径） | 5.53 | 0.09 | ✗ 分母同步 ×1.25 ⇒ 退化为饱和的功率步 |
| `interference.coupling=legacy` | **6.08** | **−0.46** | ✗ **不是增益，是删掉了干扰杠杆** |

`legacy` 值得单独说：它把直射残差整项删掉、改用解耦的手调地板（`residual_direct_factor` 等），直觉上应"白拿"。实测相反——600/0.1 上**反而更差**，且 **`kappa_dc` 在 legacy 下完全失联**（40/50/60/80 dB 给出同一个 gap `6.08`）。原因：手调地板在数值上与共享谱下的 `kappa_dc·I` 同量级，而它同时抹掉了唯一能压低它的物理旋钮。**所以 legacy 是"没有干扰杠杆的模型"，不是"更好的模型"。**

三个"看起来像开关"的项也已排除：

| 项 | 实测 | 判读 |
|---|---|---|
| `refine.apply_to_all=True` | 5.621 → 5.621（逐位不变） | 结构上不可能有用：它只影响选择器候选集**之外**的链路，而 gap 只在选中链路上算 |
| `corr.enable=True` | 5.62 → 5.83（**+0.21**） | 负收益。V1 的 `corr=off` 是**乐观**设定，打开是损失不是增益 |
| `isac_power_model=joint_waveform`（**正交口径**） | 5.621 → **5.007**（−0.61） | ⚠️ **记账白拿**，见 §8.3 |

### 8.3 ⚠️ 新查出的口径不一致：正交口径下 `joint_waveform` 白送 0.97 dB

`model.py:648-651` 在 `orthogonal` 下把感知干扰场**写死**成 `P_rad_sense = P_sense`（与 `isac_power_model` 无关）；而 `model.py:782-788` 的 `effective_sensing_power` 却随功率模型取 `P_sense` 或 `P_sense+P_comm`。于是同一个开关在两个口径下含义不同：

| 口径 | `sensing_only` → `joint_waveform` | 物理上应该 |
|---|---|---|
| `full_concurrent` / `active_set`（并发） | gap 5.62 → 5.53（**−0.09 dB**） | 信号与干扰同乘 1.25 ⇒ 基本抵消 ✓ |
| **`orthogonal`（V1 发布口径）** | gap 5.621 → **5.007（−0.61 dB）** | — |

该 −0.61 dB 与 `rho 0.8→1.0` 的 0.65 dB 几乎相同 ⇒ 模型把"通信功率并用于感知"的收益**全给了分子、没给分母**。正交口径的语义是"感知观测期内不发射载荷"，所以这个增益是**记账产物而非物理**。**不要在正交口径下用 `joint_waveform` 申报增益**；若要改功率模型，必须同时改口径并重述假设。

### 8.4 高度这一项为什么值得单独说

同一面积（600 m）、同一 RCS（0.1）、同一 seed、同一 `comm_range`（实测无影响，见 §8.5），只换垂直几何：

| 几何 | UAV / 目标高度 | 双站距离中位（500 → 800 m） | gap（600/0.1） | `rinr` 中位 |
|---|---|---:|---:|---:|
| `paper-vertical` | 800–1200 / 700–1500 m | 726 → 997 m（**2.7 dB**） | 5.62 | +8.05 dB |
| `compact-small-uav` | 200–500 / 200–500 m | 577 → 865 m（**7.1 dB**） | **2.79** | +8.83 dB |

低高度把分母**抬高**了（`rinr` +8.05 → +8.83 dB），但分子的 $d^{-4}$ 涨得更快 ⇒ 净赚 2.83 dB 缺口。**"缩场地"替代不了"降高度"。**

⚠️ 不是"改一个数字"：它把平台从"中高空 UAV"改成"低空小无人机"，需独立论证（`MOBILE_GEOMETRY_FEASIBILITY.md`）。

⚠️ **2.7 / 7.1 dB 是"链路预算的 500→800 m 位移"，不是"P_D 对距离的斜率"**。实测 P_D 斜率反而
`compact` 更小（σ=0.05：−0.049 vs `paper-vertical` 的 −0.082），因为 `compact` 把整张图抬到
接近饱和区。**低高度的作用是整体抬升（+0.077…+0.211 P_D），不是把距离轴拉直**
（`LOW_RCS_SCENARIO_500_800.md` §2/§5/§5B）。

### 8.5 附带查清：`comm_range` 在 500–800 m 带内不约束（拓扑退化为完全图）

`geometry.comm_range` 只喂 `model.py:342` 的 `edge_mask`，而 `edge_mask` **确实**用在选择（`selection.py:183`）与上报（`reporting.py:216,220,229,284,386,543`）上。实测连通边数（seed 10917，前 6 个 trial）：

| 面积 | `comm_range` = 1.25×边长 | = 2500 m | 连通边 / 210 | 最大 UAV 间距 |
|---:|---:|---:|---:|---:|
| 500 | 625 | 2500 | **210 / 210** | 568 m |
| 600 | 750 | 2500 | **210 / 210** | 666 m |
| 700 | 875 | 2500 | **210 / 210** | 767 m |
| 800 | 1000 | 2500 | **210 / 210** | 869 m |
| 4000 | — | 2500 | **138 / 210** | 4238 m |

⇒ 带内 UAV 图**恒为完全图**，`comm_range` 取 625 还是 2500 结果逐位相同（这也是 §8.4 两个"低高度"臂逐格一致的原因）；论文 4 km 工作点才是部分图（138/210）。**任何依赖"邻居稀疏 / 拓扑选择"的结论不能从 4 km 搬到这条带**；带内的 `comm_range` 取值（含既有预设 `small-uav-compact-800m` 的 1000 m）是**装饰性**的。

### 8.6 排序（新带内，600 m / RCS 0.1）

| # | 动作 | gap 5.62 → | 代价 | 是不是物理增益 |
|---|---|---:|---|---|
| 1 | `G_hw` +15 dB | 1.83 | 天线/EIRP/损耗 | ✅ 真实，但需平台设计（`RADAR_LINK_BUDGET_CALIBRATION.md` 仍标 "calibration bridge"） |
| 2 | 垂直几何降到 h 200–500 m | 2.79 | 换平台假设 | ✅ **免费** |
| 3 | `kappa_dc` 40 → 60 dB | 1.52 | 需 18.7 dB 对消能力论证 | ✅ |
| 4 | `G_proc` ×64 | 0.43 | 带宽 / 时间 | ✅ |
| 5 | `n_looks` 16 → 256 | 2.13 | CPI ×16 | ✅ |
| — | `rho`→1.0 / `joint@orthogonal` | 4.97 / 5.01 | 通信代价 / 记账 | ⚠️ 非免费 / 非物理 |
| — | `joint@concurrent` / `corr` / `refine_all` / `legacy` | 5.53 / 5.83 / 5.62 / 6.08 | — | ✗ |

**没有第 12 个入口。** 算法侧（§5）的空间是"重新分配预算"，物理侧剩下的都在上表里。

### 8.7 复现

```bash
PY=E:/anaconda/3_11_python/python.exe
# base（paper-vertical，600 m / RCS 0.1）
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --out results_v1_lever_closure_pv600
# 纯低高度臂（只改高度，comm_range 不动）
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --override geometry.h_uav_min=200 --override geometry.h_uav_max=500 \
    --override geometry.h_target_min=200 --override geometry.h_target_max=500 \
    --out results_v1_lever_closure_pv600_lowalt
# 紧凑场景臂（高度 + comm_range）
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --scenario compact-small-uav --out results_v1_lever_closure_compact
# 四个"未开采入口"臂
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --override interference.coupling=legacy --out results_v1_lever_closure_arm_legacy
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --override refine.apply_to_all=True --out results_v1_lever_closure_arm_refineall
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --override corr.enable=True --out results_v1_lever_closure_arm_corr
$PY -u tools/probe_lever_closure.py --areas 600 --rcs 0.1 --mc 24 --workers 3 \
    --override radio.isac_power_model=joint_waveform \
    --out results_v1_lever_closure_arm_jointorth
```

⚠️ `--override` 与 `--scenario` 是 2026-09-18 新增的**可选**入口，默认行为（`paper-vertical`、无 `--override`）与旧版逐位一致：`results_v1_lever_closure_pv600` 的 base 行 = **5.621**，与存档 `results_v1_lever_closure/closure.json` 的 600/0.1 列**逐位相同**。
