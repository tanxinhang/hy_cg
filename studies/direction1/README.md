# 方向 1（TP-UIC 本体）——探究产物归档

> **本目录只装方向 1**（TP-UIC 本体：对消深度 / 残余记账 / 直连参数误差）。
> 方向 2（协作融合）、方向 3（联合闭环）**另建目录**，互不干扰。
> 归档时间：2026-09-21。此后方向 1 的新结论**必须落成测试**，不再新增一次性探针。

---

## 1. 一句话总判

**TP-UIC 的对消能力已经到顶（κ_struct ≈ 65 dB > 需求 52.25 dB）；报出 κ ≈ 38 dB 是因为分母里混进了噪声增强项 `‖f(n)‖²`（占残余 99.8%），而那一项是"被减掉的噪声"，不是"没消掉的干扰"。**

⇒ 方向 1 的正确发力点是**修记账 / 建模 δ**，不是"把对消器做得更深"。深挖算法本体的收益实测 ≈ 0。

---

## 2. 目录结构

| 子目录 | 内容 |
|---|---|
| `README.md` | 本文件：汇总索引（唯一入口） |
| `scripts/` | 11 个一次性探针（可复现，不是门禁，跑它们不产生 FAIL） |
| `data/` | 12 组实测产物（`.json` / `.csv`），与脚本一一对应 |
| `docs/` | 6 份审计报告（判决链与原始证据） |

---

## 3. ★ 核心实测数字（汇总）

### 3.1 κ 的三项分解（`data/diag_kappa_decomposition/`，n=12）

| 量 | 中位 | 含义 |
|---|---:|---|
| `kappa_db` | **38.12 dB** | 生产口径（当前报出的值） |
| `kappa_struct_db` | **65.37 dB** | 只算结构残差 —— **对消能力本身** |
| `estimation / structural` | **506** | 噪声增强项是结构残差的 506 倍 ⇒ 占 `i_res` 的 **99.8%** |
| `‖f(n)‖² / ‖n‖²` | **0.0012** | 它只是被减掉的 **0.12%** 噪声 |
| 残差保留噪声比 | **0.9993** | 残差保留了 **99.93%** 的噪声 |
| `cross_coherence` | **0.356** | ⚠️ **不正交** ⇒ `result.py` docstring 声称的"两者正交、和是精确的"**不成立** |

### 3.2 P_D 天花板：κ 是唯一自变量（`data/diag_direction1_ceiling/`，n=20 配对）

| κ [dB] | 0 | 20 | **36.5**（当前） | 45 | **52.25**（需求） | 60 | 68 | perfect |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P_D | 0.000 | 0.0667 | **0.4667** | 0.7167 | **0.7500** | 0.8500 | 0.8667 | **0.8833** |
| Δ vs 当前 | — | — | — | +0.250 | **+0.283** | +0.383 | +0.400 | **+0.417** |

⚠️ **边际递减极陡**：36.5→45 dB 值 **+0.250**；45→52.25 dB 只值 **+0.033**。
需求 52.25 dB 落在**平坦段** ⇒ 拿"是否达标"判成败会系统性低估真实收益。

### 3.3 记账门控落地后的端到端（`data/direction1_gated/`，n=20 配对）

| 臂 | κ [dB] | P_D | ΔP_D |
|---|---:|---:|---:|
| `measured`（默认，逐位不变） | 37.17 | 0.5167 | — |
| `structural + δ=3e-3`（δ 敏感性） | 46.96 | **0.7167** | **+0.200 ± 0.061** |
| `structural + δ=0`（**上界，不可达**） | 67.44 | 0.8667 | +0.350 ± 0.051 |

判据：G1 ✅ / G2 ✅ / G3 P_FA 极差 **0.0033** ✅（不是门限失真）。

### 3.4 κ(δ) 解析律与同步精度（`data/diag_delta_gap/`）

```
κ(δ) = −10·log10( 2.2139·δ² + 1.803e-7 )      [δ 单位为 DD 格]，拟合误差 ≤ 0.125 dB
达标 δ ≤ 1.615e-3 格 ≈ 0.84 ns（亚纳秒）
```

| 同步精度 | δ [格] | κ [dB] | 达标? |
|---|---:|---:|---|
| 0.1 ns（实验室） | 6.4e-4 | 61.6 | PASS |
| **0.84 ns（门槛）** | 1.615e-3 | 52.25 | 边界 |
| 1 ns（IEEE1588） | 1.92e-3 | 50.4 | FAIL |
| 1.56 ns（本轮取值） | 3e-3 | 46.96 | FAIL |
| 10 ns（GPS 级） | 1.92e-2 | **31** | ❌ 收益几乎归零 |

⇒ **方向 1 的收益是"同步精度"的函数，不是一个常数**。δ 未标定前，任何"真实版"数字都可能差 20 dB。

### 3.5 ⚠️ MC 噪声尺度（报数必须带上）

同一 δ=3e-3 配置、只换 rng tag，两次 n=20 给出 **P_D 0.7167 vs 0.8000**（差 0.083），
而 κ 稳定在 46.96 / 46.97 ⇒ **κ 低噪声，P_D 的 SE ≈ 0.05**。
**MC=20 只能分辨 ≥0.10 的差异**；小于此的差别不要下结论。

---

## 4. 数据资产清单（脚本 ↔ 数据 ↔ 结论）

| 脚本（`scripts/`） | 产物（`data/`） | 产出的结论 | 已由测试接管？ |
|---|---|---|---|
| `verify_tpuic_residual_identity.py` | `verify_tpuic_residual_identity/` | 残差线性分解成立（误差 1e-29）；`‖f(n)‖²` **不在**残差里 | 属记账口径，由 `test_residual_accounting_mode` 覆盖 |
| `diag_tpuic_depth_budget.py` | `diag_tpuic_depth_budget/` | 残余里 99.95% 是 `‖f(n)‖²` | ✅ 同口径断言 |
| `probe_noise_reference.py` | `probe_noise_reference/` | 链路表 `n0` ≡ 逐 bin `sigma2`（口径一致） | 静态事实，无需回归 |
| `diag_rinr_budget.py` | `diag_rinr_budget/` | `residual_self` 是自干扰地板，占 structural 臂 rinr 的 **99.1%** | — |
| `probe_offgrid_depth.py` | `probe_offgrid_depth/` | 深度按 `−20log10(ε)` 崩塌 | ✅ `test_depth_degrades_twenty_db_per_decade` |
| `sweep_direct_estimation_delta.py` | `sweep_direct_estimation_delta/` | κ(δ) 曲线、切向阶救援窗口、`n_cpi` **未接线** | ✅ `test_structural_depth_follows_inverse_square_law` |
| `diag_kappa_decomposition.py` | `diag_kappa_decomposition/` | §3.1 三项分解（D1–D4） | ✅ `test_estimation_dominates_the_residual_budget` |
| `diag_direction1_ceiling.py` | `diag_direction1_ceiling/` | §3.2 天花板曲线 | ⚠️ 端到端，未收口 |
| `run_direction1_gated.py` | `direction1_gated/` | §3.3 门控落地复核 | ⚠️ 端到端，未收口 |
| `diag_delta_gap_attribution.py` | `diag_delta_gap/` | §3.4 差距归因（H1/H2/H3 全 PASS） | ✅ |
| `calc_delta_to_sync_ns.py` | — | δ ↔ 同步精度换算 | 纯换算 |

**已删除（数字作废，不再保留脚本）**：

| 脚本 | 作废原因 | 残留数据 |
|---|---|---|
| `run_tpuic_accounting_ab.py` | 手工 `(i_res − i_est)` 是**灾难性抵消**（estimation 占 99.8%），报出三臂 A/B 0.5167 / **0.8833** | `data/tpuic_accounting_ab/`（仅留痕） |
| `run_direction1_prescription.py` | 同上。报真实版 **0.7833 (+0.267)**；门控复核后修正为 **0.7167 (+0.200)** —— **手工口径系统性高估 0.067**，且高估深度 0.80 dB | `data/direction1_prescription/`（仅留痕） |

⛔ **纪律**：要结构残差必须读 `i_res_structural` 字段或用 `residual_accounting=structural` 门控。
**禁止手工 `(i_res − i_est)`**。

---

## 5. 与产品代码的接线（不随本目录移动）

方向 1 只有**两处**真正进了产品，其余全是探究：

| 位置 | 内容 |
|---|---|
| `isac_sim/core/config/cancellation.py` | `direct_estimation_sigma_delay_bins` / `..._doppler_bins`（δ 门控，默认 **0.0** = 完美估计，逐位不变）；`residual_accounting`（`measured` / `structural`，默认 `measured`） |
| `isac_sim/receiver/cancellation/{build_direct.py, build.py, arms_score.py}` | δ 注入（字典用带误差的 bin、真值场用真 bin）；记账口径切换（实测侧与解析侧同口径） |
| `tests/test_direct_estimation_error.py`（8） | δ 门控：关闭时返回同一对象、不消耗 rng、逐位不变 |
| `tests/test_residual_accounting_mode.py`（12） | 记账口径：只改记账不改算法 |
| `tests/test_direction1_convergence.py`（14） | ★ **本目录结论的断言化**（§3.1/§3.4 的机制全部在此） |

---

## 6. 停止条件（明确不再做）

| 项 | 判定 |
|---|---|
| 继续深挖对消算法本体（更深 LS / 收缩 / 降 `d_eff`） | ⛔ **停**。中位结构残差已在自干扰地板下 **20.8 dB**，收益 ≈0 |
| 把 `n_cpi` 当深度杠杆 | ⛔ **停**。实测 κ_total 在 1/4/16/64 下**逐位相同**（36.50）—— 未接到对消臂 |
| 调切向阶数换性能 | ⛔ **停**。它是 δ 的保险不是增益；δ=0 时开它**反亏 13.3 dB**，平台 50.3 dB 低于需求 |
| 手工 `(i_res − i_est)` | ⛔ **禁**。灾难性抵消 |
| 新增一次性探针 | ⛔ **停**。新结论必须落成测试 |

---

## 7. 遗留（若要继续，按此顺序）

1. **δ 标定**（B 类）：锚定到 `waveform_impairments.sync_*_bins` —— 目前同一个同步误差存在**两个键**（一个罚分子、一个抬分母），必须同值。
2. **多普勒维**：本轮只注入延迟维。
3. **P_D 结论提到 MC≥200**（见 §3.5）。
4. **门控默认是否打开**：仍默认 `measured`（发布数字不动）。打开 = B 类建模变更。
5. 转方向 2。

---

## 8. 审计报告索引（`docs/`）

🔴 **2026-09-21 编码事故**：下列 5 份（除 `DIRECTION1_CONVERGENCE.md`）**正文已损坏不可读**
（UTF-8 字节被当 GB18030 解码后重新存盘，无法映射的字节被替换成 `?`，**信息已丢失、
无法无损还原**；备份在 `.workbuddy/enc_backup/`）。各文件已加损坏横幅。
**数字没丢**：全部结论在本文件 §3 与 `tests/test_direction1_convergence.py`（14 条断言）。
防再犯：`tests/test_study_docs_hygiene.py`（不得新增乱码文档）。

| 文档 | 内容 | 状态 |
|---|---|---|
| `TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md` | 记账判决 + off-grid 反向检验（**虚假增益判定**的由来） | ⛔ 损坏 |
| `AUDIT_TPUIC_RESIDUAL_ACCOUNTING.md` | 对上一份的对抗式复核（抓到并修正了一处换算错误） | ⛔ 损坏 |
| `AUDIT_DIRECTION1_DELTA.md` | δ 门控实现 + δ/切向阶/步长/`n_cpi` 扫描与判决 | ⛔ 损坏 |
| `AUDIT_DIRECTION1_FAILURE_ROOT_CAUSE.md` | ★ **失败根因**：对象对、靶子错（动了占 0.2% 的项） | ⛔ 损坏 |
| `AUDIT_REAL_VS_EXPECTED_GAP.md` | "真实版与预期差距"的归因（H1/H2/H3 闭合；"真实版"标签作废） | ⛔ 损坏 |
| `DIRECTION1_CONVERGENCE.md` | 收敛清单：什么已收口、什么只是历史探针 | ✅ 完好 |
