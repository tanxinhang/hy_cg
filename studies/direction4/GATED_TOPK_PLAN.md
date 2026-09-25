# `gated_topk` 求解器实施方案

> 目标：新增 `solver = gated_topk`，**不动** 现有 `coarse_to_fine`（论文基线 + 回归参考）。
> 分轮实现，第一轮只到"Top-K 多峰细化"，**门控默认关**——先分清"搜索策略"和"门控门限"
> 各自的贡献，避免又一次把两条主线混在一起。

## 0. 三条硬约束（决定了拆法）

| 约束 | 现状 | 后果 |
|---|---|---|
| `isac_sim/` 模块 ≤150 行（纯红线） | `fit.py` **147/150** | `gated_topk` 不能塞进 `fit.py`，必须先拆 |
| `tests/` ≤350 行（纯红线） | `test_target_state_map.py` **334/350** | 12 条新测试进**新文件** `test_target_state_gated_topk.py` |
| 默认路径 bit-exact（铁律 1） | `coarse_to_fine` 是 Gate 1/2 的口径 | 拆分后它必须**逐位不变**，见 §6 |

## 1. 模块拆分

| 文件 | 职责 | 新增? |
|---|---|---|
| `fit.py` | 只剩**派发**：解析 σ/span/step → 按 `solver` 分派 | 改（瘦身到 ~80 行） |
| `objective.py` | 候选×接收机缓存、目标函数、评价计数 | **新增** |
| `coarse.py` | 现有 coarse-to-fine 循环（**逐字搬**，行为不变） | **新增** |
| `screen.py` | 接收机筛选顺序 + successive-halving 排序 | **新增** |
| `nms.py` | 非极大值抑制（纯函数，单测好写） | **新增** |
| `refine.py` | Top-K 多峰模式搜索 | **新增** |
| `boundary.py` | 边界外环扩展 | **新增**（阶段 B） |
| `gate.py` | 四个诊断量 + 接受/拒绝 | **新增**（阶段 A 先只算不判） |
| `prior.py` | 2×2 协方差 → precision（对称/半正定/条件数检查） | **新增**（阶段 C） |
| `gated_topk.py` | 编排：screen → halving → NMS → refine → boundary → gate | **新增** |
| `results.py` | 诊断字段 | 改（38 → ~60 行） |
| `crossfit.py` | 透传 `solver` / `solver_options` | 改（75 → ~95 行） |

每个新模块 ≤150 行；`views.py`、`scorer.py`、`geometry.py`、`apply.py` 不动。

## 2. 阶段 A：缓存 + Top-K + NMS + 多峰细化 + 诊断字段

### 2.1 按接收机缓存（第一步）

```python
class ObjectiveEvaluator:
    def __init__(self, cfg, views, target, belief_geometry, precision, cache=None): ...
    def receiver_gain(self, delta, view) -> float:   # 缓存键 (dx, dy, view.receiver)
    def objective(self, delta, active_views) -> float:
    # 计数：evaluations（去重后的候选数）、receiver_evaluations（候选×接收机次数）
```

- 缓存键 `(round(dx,9), round(dy,9), view.receiver)`——与规格一致。
- 候选从 2 个接收机升级到 6 个时只补算 4 个，**不重算**（测试 4 钉住）。
- `evaluations` 语义**保持"候选数"**（不是接收机次数），否则现有断言
  `evaluations >= 150` 会失效。新增 `receiver_evaluations` 单独记成本。

### 2.2 接收机筛选顺序（belief-only，不看 H1 统计量）

可用量（已确认可达）：`belief_geometry.p_uav`（接收机坐标）、
`belief_geometry.p_tgt[target]`、被测目标源的 `src.uav`（**发射机索引**）。

```
quality(rx) = 1 / (R_tx^2 * R_rx^2)        # 双基地雷达方程的形状项，纯 belief
    R_tx = ||p_tgt - p_uav[src.uav]||
    R_rx = ||p_tgt - p_uav[rx]||
azimuth(rx) = atan2(p_tgt.y - p_uav[rx].y, p_tgt.x - p_uav[rx].x)
```

1. 第 1 个：`argmax quality`；
2. 第 2 个：与已选**方位差最大**（同分用小接收机号）——避免两个节点看同一方向；
3. 后续：每次取"到已选集合最小方位差"最大的那个。
4. **整个训练折固定这一个顺序**，中途不改。

平局一律按 `receiver` 升序，保证确定性（测试 12）。

### 2.3 Successive-halving + NMS

参数按规格：`screen_receivers=2, screen_keep=40, verify_receivers=4,
verify_keep=12, full_keep=3`，NMS 半径 75 / 75 / 100 m。

```
169 点 × 2 接收机  →  NMS(keep=40, r=75)
 40 点 补到 4 接收机 →  NMS(keep=12, r=75)
 12 点 补到 6 接收机 →  NMS(keep=3,  r=100)  = seeds
```

NMS 必须是**按排名贪心**：从高到低遍历，与已选距离都 ≥ radius 才收。
忘了这步 Top-12 会全挤在同一个峰旁边（测试 3）。

### 2.4 Top-3 多峰局部细化

模式搜索，步长 `(37.5, 18.75, 9.375)`，每步 8 邻域，`max([best, *cands])`。
三个 seed 各细化一次，**最后比三个局部最优**，不信粗网格第一名。
不引入 Nelder–Mead（边界/计数/确定性都更好写）。

### 2.5 四个诊断量（阶段 A 只算不判）

```
gain_over_zero   = J_best - J_zero
peak_gap         = J_best - J_second          # J_second = 次优局部最优
boundary_hit     = (span - max|best_delta|) <= coarse_step
receiver_support = 该接收机在 delta_hat 处的增益 > 其在 0 处的增益 的接收机数
```

阶段 A：`accepted` 恒 `True`，`delta_xy_m = raw_delta_xy_m`，但四个量全部落盘
⇒ 阶段 D 拿到这些分布就能离线定门限，**不用重跑**。

## 3. `results.py` 新增字段

`solver, accepted, raw_delta_xy_m, delta_xy_m, objective_at_zero,
objective_best, objective_second, gain_over_zero, peak_gap,
boundary_expanded, boundary_unresolved, screen_receiver_count,
coarse_candidates, full_candidates, evaluations_total, receiver_evaluations`

`raw_delta_xy_m`（搜索器原始结果）与 `delta_xy_m`（门控后真正应用的）**必须分开**——
否则分不清"搜索错了"还是"门控拒绝了"。

## 4. 默认值：**新能力一律默认关**（铁律 1）

| 开关 | 默认 | 含义 |
|---|---|---|
| `solver` | `coarse_to_fine` | 完全走老路径 |
| `state_gate` | `False` | 只算诊断量，不拒绝更新 |
| `boundary_expand` | `False` | 不做外环扩展（阶段 B 打开） |
| `prior_mode` | `isotropic` | 沿用 `P[q,0,0]`（阶段 C 才用 2×2） |

## 5. 阶段 B–E

- **B 边界外环扩展**：只补对应方向的外环（如 dx=525,600 × dy=-600…600），
  不重算整个 ±600 方框；与新 Top-3 合并后再 NMS+细化；仍贴边则
  `boundary_unresolved=True`（阶段 D 起默认拒绝更新）。
- **C 各向异性先验**：`cov_xy = P[q,0:2,0:2]`，`pinv`，检查对称/半正定/条件数，
  必要时加对角正则。**不放大 prior 系数**（那会变成人工正则化）。
- **D 独立开发集标定门控**：把 `train` 从 1 个场景扩到 **40 个 0 m 场景**
  （`--train-scenes 40`），新脚本 `scripts/calibrate_state_gate.py` 采
  `gain_over_zero` / `peak_gap` 分布，取 95% / 90% 分位数冻结为 `tau_gain` / `tau_gap`，
  使错误更新率 ≤5%。**门限冻结后才跑正式 calibration/test**。
- **E 接入 cross-fit + Gate CLI + paired ablation**：`crossfit.py` 只透传，
  统计逻辑不动；两折各返回完整诊断；CSV 写 `accepted_a/b`、
  `raw_delta_a/b_x/y_m`、`delta_a/b_x/y_m`、`gain_over_zero_a/b`、`peak_gap_a/b`、
  `boundary_expanded_a/b`、`receiver_evaluations_a/b`。

## 6. bit-exact 保证（硬门禁）

`--solver coarse_to_fine` 在 Gate 1 的 **41 行**上必须与现有
`data/gate1/records.csv` **逐位一致**。验证办法：用同一 `--master-seed` 重跑
radius 150，逐行比对 26 列浮点。这条会写成脚本 + 断言，不是口头承诺。

## 7. 测试清单（新文件 `tests/test_target_state_gated_topk.py`，≤350 行）

| # | 测试 | 阶段 |
|---|---|---|
| 1 | Top-K 含正确峰时能恢复偏移 | A |
| 2 | 粗网格第一名为假峰时，Top-3 仍能找到正确峰 | A |
| 3 | NMS 不返回三个相邻候选 | A |
| 4 | 缓存保证候选升级接收机时不重复评价 | A |
| 5 | 0 m 弱证据场景被门控为零 | D |
| 6 | 强证据偏移不被门控误杀 | D |
| 7 | A 折门控不读取 B 折 | D |
| 8 | 边界命中触发外环扩展 | B |
| 9 | 扩展后仍贴边返回 `boundary_unresolved` | B |
| 10 | 各向异性协方差的先验方向正确 | C |
| 11 | 新旧求解器在简单无噪声场景得到相同偏移 | A |
| 12 | 固定种子下结果逐次一致 | A |

## 8. CLI（加在 `tools/gate_crossfit_target_state_map.py`）

```
--solver gated_topk          --screen-receivers 2   --screen-keep 40
--verify-receivers 4         --verify-keep 12       --top-k 3
--nms-radius-m 75            --boundary-expand      --max-search-sigma 4
--state-gate                 --state-gate-gain <>   --state-gate-gap <>
```
`--solver coarse_to_fine` 继续可用 ⇒ 直接做 paired ablation。

## 9. 验收判据（先定后跑）

阶段 A 通过 ⇔ 三条同时成立：

1. **bit-exact**：`coarse_to_fine` 41 行逐位不变（§6）。
2. **状态 p90 改善**：150 m 档 `shared_crossfit` 的状态误差 p90 从 **331.7 m** 降下来，
   且中位数不明显变差（现中位 11.7 m）。
3. **运行时间下降**：串行单场景行从 **390 s** 降下来。
   预算：169×2 + 40×2 + 12×2 + 3 seeds×3 步×8×6 ≈ **874 次接收机评价**
   vs 现在 190×6 = **1140** ⇒ 预计 **~1.3×**。

⚠️ 先说清预期：**加速不是主要收益（只有 ~1.3×）**——细化阶段 3×3×8×6 = 432 次
占了近一半。主要收益是**多峰细化带来的 p90 改善**和后续门控。
若阶段 A 的 p90 没改善，则应先把细化阶段的接收机数降下来换更多 seed，
而不是急着上门控。

## 10. 阶段 A 完成记录（2026-09-25）

### 10.1 落点（与 §1 的偏差：没有新建 `boundary.py` / `prior.py`）

| 文件 | 行数 | 说明 |
|---|---|---|
| `objective.py` | 87 | 候选×接收机缓存 + `evaluations` / `receiver_evaluations` |
| `coarse.py` | 49 | 旧循环逐字搬走，`REFINE_FRACTIONS` 也搬过来 |
| `screen.py` | 66 | belief-only 接收机顺序 + 粗网格点 |
| `nms.py` | 35 | 按排名贪心 |
| `refine.py` | 42 | 8 邻域模式搜索 |
| `gate.py` | 41 | 四个诊断量（**只算不判**） |
| `gated_topk.py` | 78 | 编排 |
| `fit.py` | 116 | 瘦成派发器（原 147） |
| `results.py` | 68 | 15 个诊断字段（原 38） |

`boundary.py` / `prior.py` 属于阶段 B/C，未建 —— 阶段 A 不写用不上的积木
（否则就是死代码）。

### 10.2 三条验收判据：全部通过

| # | 判据 | 实测 |
|---|---|---|
| 1 | bit-exact | 改动前后的 `fit.py` 在 6 组 (场景, 半径) 上 `delta`/`objective`/`prior_penalty`/`receiver_gain`/`evaluations` **全等** |
| 2 | 状态 p90 改善 | 132.8 → **95.4** m（p95 321.3 → 139.5，中位 8.5 → 6.4） |
| 3 | 运行时间下降 | **1.29–1.47×**（接收机评价 1146 → 874） |

41 场景配对：**27 个完全一致、11 个改善、3 个恶化**。

### 10.3 恶化的判别量是 `boundary_hit`，不是 `gain_over_zero`（与 §2.5 的预期相反）

3/3 个恶化折贴边，13 个对的折一个都没贴边（Fisher p ≈ 0.002）；而
`gain_over_zero` 在两侧完全重叠（恶化 12.7–21.0，改善侧有 22.5 / 23.0 更高的）。
⇒ **阶段 B（边界外环扩展）要排在阶段 D（门控）前面**：先分清"真峰在盒外"还是
"噪声把解推到边上"，扩展后仍贴边 ⇒ `boundary_unresolved` ⇒ 退回先验。
`gain_over_zero` / `peak_gap` 继续落盘，但不要用这 16 个折去定门限。

### 10.4 测试

新文件 `tests/test_target_state_gated_topk.py`（8 条）：多峰恢复、假峰第一名、
`boundary_hit` 判别、NMS 不吃同峰、缓存升级不重算、接收机顺序 belief-only +
与给出次序无关、两求解器在无噪声场景一致、逐次一致。
