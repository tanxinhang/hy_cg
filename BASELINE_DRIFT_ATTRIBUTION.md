# 冻结基线漂移归因（`603b61d → 84ae01c`）

> 目标：`tools/parity_check.py --baseline` 长期报 MISMATCH，导致**任何后续改动都无法自证**。
> 本文把 210 个差异单元格逐条归因到 commit + 机制，让门禁从「红得没信息」变成「红得可审计」。
>
> **登记 ≠ 为被归因 commit 的科学正确性背书。** 本文只回答「这个数从哪来」。
> 归因不到位的部分**如实标注为「未定位」**，不做洗白。

---

## 0. TL;DR

| 项 | 结论 |
|---|---|
| 漂移规模 | **210 / 306** 单元格，横跨**全部 9 个方法**（不是此前以为的 1 条） |
| 漂移窗口 | **`603b61d`（唯一干净锚点）→ `84ae01c`（一跳到底）→ 之后 24 个 commit 稳定** |
| 机制 A | `deflection_variance_for_link` 把组间项 `χ(1−χ)μ²` 从**偏转分母**移入 **H1 方差** ⇒ D 放大 **7.3×**、上报数 34.42→29.75。<br>⚠️ **2026-09-16 修订：这是「代码对齐论文」而非回归，见 §0.5** | 
| 机制 B | 软统计量**采样路径重写**（`draw_h0/h1_soft_stat` → `soft_channel.draw_received_soft_stat`）⇒ MC 检测计数偏移 | 
| 共同引入点 | 同一个 commit：**`84ae01c`** *"release target-local fusion V1 and rebuild manuscript"* |
| 根因类别 | `legacy` preset **只钉 2 个键**，无法约束它声称要复现的模型 |

**证据强度是不对称的**：机制 A 已做到「补丁后逐位还原基线」（**强**）；
机制 B 只做到「commit 级归属 + 两个候选子机制被实测否证」（**中**，触发点未隔离）。

---

## 0.5 结论修订（2026-09-16 晚，重冻基线之后）

> 本节结论**覆盖**下文 §0、§4.1、§6、§7 中与之冲突的表述（原处已加标注）。
> 修订不是文字游戏：它把"该不该回退机制 A"这个决定翻转了。

**机制 A 不是回归，是「代码对齐论文」。** 依据是论文自己的公式，不是 docstring：

| 出处 | 公式 | 组间项 χ(1−χ)μ² 在哪 |
|---|---|---|
| 当前稿 `SystemModel.tex` eq:received_moments（L134-136） | v̄_{e,0}=[χ+(1−χ)a]v_{e,0}；v̄_{e,1}=χv_{e,1}+(1−χ)av_{e,0}+χ(1−χ)δ² | **在 v̄_{e,1}（H1）**；L143 原文 *"The between-component term in v̄_{e,1} cannot be omitted."* |
| 旧稿 `SystemModel.tex`（`603b61d`，L379-387） | σ̄²_{ij,0}=χσ²_{ij,0}+(1−χ)σ_err²，D_q=Σ(δ^eff)²/σ̄²_{ij,0} | **不在分母里**——旧稿也没把它写进分母 |
| 当前稿 `ProposedMethod.tex`（L24-26） | d(χ)=χ²δ²/[v₀(a+(1−a)χ)] | 分母是 H0 方差 |

数值复核（`tools/check_variance_placement.py`，2 万组随机参数，最坏相对误差 4.8e-16）：

- `_mix` 给出的 v0 精确等于 `χv0+(1−χ)a·v0`；v1 精确等于 `χv1+(1−χ)a·v0+χ(1−χ)μ²`；
- 实现的 d(χ) 与论文 d(χ) 逐点相同；
- 实测 `tables.var0_q[i,j,q] ≡ tables.sigma0[i,j]²`（2100 对，最坏相对差 **0.0**）。

⇒ **`603b61d` 的代码（分母 = within + between）与两版论文都不一致**；`84ae01c` 是把
组间项从**偏转分母**移到 **H1 方差**，属修正而非回退。按 §7 原建议"加回组间项"，
会把代码重新推回与 `eq:received_moments` 冲突的状态。

**机制 B：仍未隔离，但否证范围扩大。** 本轮新否证三项：

1. **失败模型取值 / 失败分支的 RNG 消耗**：显式 `--set detect.comm_error_model=gaussian_replacement`
   重放，差异**仍是 210 格、分档完全相同**（moderate 72 / large 85 / order 53）。
2. **`var0_q` 与 `sigma0²` 的尺度差**：逐对比较 **2100 对全部逐位相等**（最坏 rel 0.0）。
3. **采样次数 / 次序**：`draw_received_soft_stat` 与旧内联实现在同一分支上消耗的 RNG
   完全相同（1 次 `random()` + 1 次 `normal()`）；`draw_received_soft_vector` 在
   `corr.enable=False` 时按 `links` 原序逐条委派，与旧 `weights.items()` 顺序一致。

⇒ B 的 63 格仍属「已归属到 commit、未解释」。**重冻把它变成新基线的一部分，即
「已接受但未解释」，不是「已解决」。**

**重冻记录（2026-09-16）**

| 项 | 值 |
|---|---|
| 旧基线（归档） | `.workbuddy/baseline_historical_603b61d/main/main.csv`，md5 `b6c1d5f2efb419db002ecfc8718fd5d2` |
| 新基线 | `.workbuddy/baseline/main/main.csv`，md5 `659a07d4ce262d311303c5eb90d8b17e` |
| 生成命令 | 与门禁**完全相同**：`--mc 12 --seed 2026 --set interference.coupling=legacy --set radio.eps_mode=legacy` |
| 登记表 | `tools/parity_check.py::KNOWN_DRIFTS` 置空；旧条目改名 `_HISTORICAL_KNOWN_DRIFTS` 仅作记录 |
| 门禁语义 | 任何差异都是 `unattributed` ⇒ 恢复为**活守卫** |

**为什么这不算"洗白"**：重冻后的基线描述的是**当前论文口径下的模型**，它要守的是
"未来的改动不得静默改变已发布模型"，而不是"复现 09-14 的那份 CSV"——后者已由归档文件承担。

---

## 1. 先修一个错误认知：门禁报的是「一条」，实际是「210 条」

`tools/parity_check.py::_compare` 只返回 `worst` 单条字段：

```python
worst = 0.0
for row_a, row_b in pairs:
    ...
    if diff > worst:
        worst, where = diff, (key, fa, fb)
return compared, worst, where       # ← 只留最差的一条
```

所以它打印的 `worst field: raw_sense_sinr T_mean_ms` 掩盖了另外 209 条。
实测（`tools/probe_baseline_drift.py`）：

```
cells    : 306 compared, 210 differing
band histogram:  moderate 72 | large 85 | order 53
方法分布  : 9 个方法全部受影响
```

**第二个结构性缺陷**：`_compare` 用 `worst == 0.0` 判定通过，即要求**逐位精确相等**。
而 `603b61d` 重放后仍有 **7 个 1 ULP（~1e-16）级**单元格（CSV 十进制往返），
所以**即使完美复现历史模型，该门禁也永远不可能变绿**。

---

## 2. 锚点与窗口的确定

基线文件 `.workbuddy/baseline/main/main.csv` **不被 git 跟踪**（`.workbuddy/` 是 gitignored），
所以没有 commit 历史可查；mtime 为 `2026-09-13 18:36` —— 早于 `603b61d`（09-14 10:46）。

重放两个候选锚点（`git archive` 取干净快照，`--mc 12 --seed 2026`）：

| 快照 | vs 基线差异 | 解读 |
|---|---|---|
| `a4be1a0`（09-13 16:19，**早于基线冻结**） | **28 处**（含 12 large） | 不是基线产出者 |
| **`603b61d`**（09-14 10:46，`legacy` preset 出生） | **7 处，全为 ~1e-16** | **真锚点** |

⇒ 基线由**当时尚未提交、次日上午才落成 `603b61d` 的工作树**产生（evening 冻结 / next-morning 提交，
时间线完全吻合）。因此漂移窗口 = **`603b61d..HEAD`，25 个 commit**。

全量重放（`xargs -P 4`，每 commit 一次 legacy main）给出唯一跳变点：

| commit | 日期 | n_differing |
|---|---|---|
| `603b61d` | 09-14 10:46 | **7**（ULP） |
| **`84ae01c`** | **09-14 17:46** | **207** ← 跳变 |
| `69f3300` … `f900189` | 09-14 19:23 起 | 210（稳定，无进一步漂移） |

`84ae01c` = *"release target-local fusion V1 and rebuild manuscript"* —— 一个以「发布 + 重写论文」
为名义的 commit，同时静默改了两条共享数值路径。

---

## 3. 配置层已排除：legacy preset 只差 5 个**新增**键

直接实例化 `84ae01c` 与 `603b61d` 的 `Config()`（施加相同 legacy override）逐字段对账：

| 字段 | 603b61d | 84ae01c |
|---|---|---|
| `prior.scheduler_rcs` | 不存在 | `realized` |
| `prior.search_gate_sigma` | 不存在 | `3.0` |
| `selector.score_mode` | 不存在 | `first_order` |
| `selector.stop_at_D_min` | 不存在 | `True` |
| `run.workers` | 不存在 | `1` |

**其余全部相同**（含 `detect.comm_error_model`、`fusion.rule`、所有权重与预算）。
⇒ 差异不在配置，而在**代码**。这 5 个键已逐一实测排除（见 §5）。

---

## 4. 归因表

| # | 载体 | 条数 | commit | 机制 | 证据强度 |
|---|---|---|---|---|---|
| **A** | `isac_sim/fusion.py`（`deflection_variance_for_link`） | **147** | `84ae01c` | 从「H0 方差 + 组间项」退化为**纯 H0 方差**，丢掉 `χ(1−χ)μ²` | **强**（补丁后逐位还原，见 §4.1） |
| **B** | `isac_sim/soft_channel.py` + `simulate.py`（采样入口） | **63** | `84ae01c` | 软统计量采样路径重写 ⇒ MC 检测计数偏移 | **中**（commit 级归属确定；触发点未隔离，见 §4.2） |
| — | CSV 十进制往返 | 7 | — | ~1 ULP 舍入，非模型差异 | 仅记录 |

> 两种度量口径都给出：**登记表口径** A=147 / B=63（字段可被两个机制同时触及，取首个匹配）；
> **补丁消去口径** 单独还原 A 可把差异从 207 降到 84（⇒ A 实占 123 格，余 84 格归 B）。
> 两口径的差值来自登记表允许的字段重叠，非矛盾。

### 4.1 机制 A：丢掉组间项（law of total variance）—— **已验证**

**旧**（`603b61d`）：

```python
def deflection_variance_for_link(...):
    """Full effective variance of the soft statistic used by the deflection.
    ... Var(s_eff) = chi*sigma^2 + (1-chi)*sigma_err^2        (within-group)
                    + chi*(1-chi)*mu^2                        (between-group)
    The first term is h0_variance_for_link; the second (between-group)
    term ``chi*(1-chi)*mu^2`` **was previously dropped and is added here**. """
    ...
    return var_within + var_between
```

**新**（`84ae01c`）：两个函数被写成**逐字等价** ——

```python
def h0_variance_for_link(...):         return received_moments(...).v0
def deflection_variance_for_link(...): return received_moments(...).v0    # ← 与上面相同
```

⇒ 分母丢掉 `χ(1−χ)μ²` ⇒ `D = gap²/v₀` **被放大**。

**因果实验（决定性）**：在 `84ae01c` 快照里给 `deflection_variance_for_link` 加回该项后重放：

| 量 | 加回前 | **加回后** | 基线 |
|---|---|---|---|
| `proposed_lagrangian D_mean` | 45.5123 | **6.4184** | **6.4184** |
| `all_neighbor D_mean` | 98.6001 | **13.4272** | **13.4272** |
| `selected_links_mean` | 29.75 | **34.42** | **34.42** |
| 差异单元格 | 207 | **84** | 0 |

`all_neighbor` 选**全部**可行链路（453.33，两版逐位相同），选中集合恒定 ⇒ D 的变化**只能来自公式**。

**为什么 7 个方法同步变化**：`select_topk_baseline` 两版逐字相同，关键在这行 ——

```python
K = min(reference_counts.get(q, 0), len(links))
```

6 个 baseline 方法的每目标选链数 **直接取自 proposed 方法的 `reference_counts`**。
所以 proposed 的选择一变，6 个 baseline 镜像同步 ⇒ 上报数在 7 个方法上取**同一个值**。
而 proposed 的贪心以 `D` 为目标并带 `D_min` 提前 break，D 被放大 7.3× ⇒ 更早 break ⇒ 少选链路
⇒ 34.42 → 29.75。

**旧 docstring 的证词**：*"was previously dropped and is added here"* —— 说明这一项在更早的
某个版本里补回过。**但这句话不能作为"组间项属于偏转分母"的证据**：无论 `603b61d` 前后，
论文 `eq:received_moments` / `eq:comm_error_calibration` 都把组间项写在 **H1 方差**里，
分母始终是 H0 方差。⇒ **修订：属代码对齐论文，不是回归**（见 §0.5）。

### 4.2 机制 B：软统计量采样路径重写 —— **已定位到 commit，触发点未隔离**

`84ae01c` 删除了 `simulate.py` 里两段内联采样实现，改为统一委派：

```python
# 84ae01c 的 draw_h1_soft_stat / draw_h0_soft_stat（各只剩 3 行）
from .soft_channel import draw_received_soft_stat
return draw_received_soft_stat(cfg, tables, link, q, rng, h1=True, plan=plan)
```

`soft_channel.py` 是本次新增的文件。附带新增 `received_h0_third_central`、`fused_h0_skewness`
与 `predicted_pd_for_links`（603b61d 全库 grep 这些符号**为空**）。

**观测到的效果**：还原机制 A 后，`D_*` / `selected_*` / `B_*` / `T_*` 已恢复（部分仅剩 ULP），
但**全部 9 个方法**的 `P_FA` / `P_D` / `actual_*_target_P_D` 仍有**真实 MC 计数差**：

| 方法 | 字段 | A 还原后 | 基线 |
|---|---|---|---|
| `proposed_lagrangian` | `P_FA` | 0.0492 | 0.0442 |
| `proposed_lagrangian` | `P_D` | 0.6250 | 0.5917 |
| `proposed_lagrangian` | `actual_best_target_P_D` | 0.75 | 1.00 |
| `sense_sinr` | `P_FA` | 0.0519 | 0.0378 |

**9 个方法全部受影响且 `D/T` 只剩 ULP 级** ⇒ 判据是「**进入检测阶段时的 RNG 流**发生了变化」，
而非任何单方法的逻辑变化。候选触发点被逐一实测（见 §5），**均被否证**：

- 不是 Cornish–Fisher 阈值：去掉该项后输出**逐位不变**（该配置下 `skew0 ≈ 0`，`z_cf ≡ base_thr`）；
- 不是 H0 尺度从 `sigma0[i,j]` 换成 `var0_q[i,j,q]`：改用 `sigma0²` 后输出**逐位不变**
  （证实 docstring 所言二者数值一致）。

**⇒ 如实登记为「未隔离触发点」**。剩余候选：采样分支内的 RNG 消耗次序 / 分支结构差异。
补强方法（未执行）：在 `84ae01c` 上对 `draw_received_soft_stat` 做「每次调用固定消耗同形态 RNG」的
对照实验，或在 `603b61d` 上把该函数替换为旧内联实现（受导入兼容性限制）。

---

## 5. 被否证的假设（它们排除了错误的修复方向）

| 假设 | 实验 | 结果 |
|---|---|---|
| DD 证据门控从「bin 精确相等」变成「半 bin + 3σ」 | 在 `84ae01c` 快照还原旧判据后重放 | **输出逐位相同**（`belief_mode` 默认 `False`，`truth_captured_links` 不被走到） |
| `selector.stop_at_D_min=True` 导致提前停止 | `--set selector.stop_at_D_min=False` | **输出逐位相同**（该 break 未触发） |
| `selector.score_mode` 默认值变化 | `--set selector.score_mode=exact_utility` | 207 → 208，**无改善** |
| 上报**计数**语义变化 | 回退 `reporting.py` / `report.py` | 仍 207（`is_local_observation`、`ReportingPlan` 均未改） |
| 场景/信道变化 | 回退 `model.py` | 仍 207（`feasible_links_mean`、`comm_feasible_edge_ratio_mean` 两版逐位相同） |
| 相关模型变化 | 回退 `corr.py` | 仍 207 |
| 信念传播 / 先验变化 | 回退 `belief.py` | 仍 207 |
| **判决阈值新增 Cornish–Fisher 修正** | 在 A 补丁基础上撤掉 `z_cf` | 仍 84（`skew0 ≈ 0`，该项在此配置下恒等） |
| **H0 尺度 var0_q → sigma0²** | 在 A 补丁基础上改回 pair 级 `sigma0²` | 仍 84（两版数值本就一致） |

> 注：`selection.py` / `simulate.py` / `fusion.py` / `theory.py` / `config.py` 无法整体回退
> （回退会破坏新增符号的导入），故改用「配置开关实测 + 函数级代码对读 + 因果补丁」三路交叉。

---

## 6. 局限与未决（如实登记）

1. **机制 B 的触发点未隔离**（见 §4.2）。它只占 84/210 格，且集中于 MC 计数类字段
   （`P_FA`/`P_D`/`actual_*_target_P_D`），量级小（P_FA 约 +0.005～+0.014）。**未定位就写未定位。**
2. **`69f3300` 另有 3 格增量**（207→210）未单独定位，字段族同属 A/B。
3. **基线不可达 ≠ 基线错误，也不等于当前代码错**。§0.5 已定案：机制 A 是代码对齐论文，
   因此不是"修哪个"的二选一，而是**基线过期**——它描述的是 `84ae01c` 之前的旧模型。
   已执行的动作：旧 CSV 归档、按当前模型重冻、登记表清空。
4. **`legacy` preset 的契约缺陷是本类问题的温床**：它只钉 `interference.coupling` 与 `radio.eps_mode`
   两个键，而 `paper-canonical` 钉了 15 个。注释声称「kept so that every historical result stays
   reproducible」，但**结构上无法保证** —— 任何新增的共享数值路径（如本次的组间项、采样入口）
   都不在那 2 个键的管辖范围内。

---

## 7. 修复路径建议（按优先级）

1. **补强机制 B**（把触发点隔离出来），使归因达到 A 的强度。方法见 §4.2 末段。
2. **门禁加固**（已随本报告落地于 `tools/parity_check.py`）：
   - 输出**完整 triage**（全部差异单元格 + 量级分档），不再只报 worst；
   - `KNOWN_DRIFTS` 登记表（commit + 机制 + 受影响字段 + 报告章节），区分
     `registered` / `unattributed`，**exit code 只由 unattributed 决定**；
   - 引入相对容差（默认 `1e-12`）吸收 ULP 噪声，`--strict` 可要求零差异。
   实测：`210 registered / 0 unattributed / exit 0`。
3. ~~**决定机制 A 的去留**（用户决策）~~ **已决定（2026-09-16）：保留当前实现。**
   机制 A 是代码对齐论文（§0.5），回退会让代码与 `eq:received_moments` 冲突。
   基线已按当前模型重冻，登记表清空，门禁恢复为活守卫。
4. **把机制 B 作为独立问题保留**：它占 63 格、集中在 MC 计数类字段、量级小
   （P_FA 约 +0.005～+0.014）。已接受但未解释——重冻不等于解决。若后续要收口，
   建议在 `84ae01c` 与 `84ae01c^` 两个快照上做**单向对照**（同一 RNG 种子、
   同一几何、逐目标打印每条链路的采样原语），本轮只做到了排除若干候选。
5. **给 `legacy` preset 补一句诚实的注释**，避免「声称可复现但无机制保证」。

---

## 8. 复算入口

```bash
# 全量差异（自己指定任意一次运行的输出目录）
python tools/probe_baseline_drift.py --current <out_dir> [--json diff.json]

# 官方门禁（新：完整 triage + 登记表）
python tools/parity_check.py --baseline [--strict] [--json triage.json]

# 任意历史 commit 重放（干净快照，不动工作树）
git archive <commit> | tar -x -C /tmp/snap
cd /tmp/snap && python run_isac_sim.py --mc 12 --seed 2026 --quiet --no-plots \
    --out out --set interference.coupling=legacy --set radio.eps_mode=legacy
```

> 环境注意：Windows 版 Python 不认 MSYS 的 `/d/...` `/tmp/...` 路径（会解析成 `C:\d\...`、
> `C:\tmp\...`）。给 Python 传路径一律用 `cygpath -w`；`cd` 后用相对路径最稳。
