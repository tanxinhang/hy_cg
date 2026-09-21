# 方向 1 收敛清单（2026-09-21）

用户判定"研究有点偏颇了，需要收敛"。本文件给出**边界**：
什么已经收口（变成会失败的断言），什么只是历史探针（不再维护），
什么明确不再做。

---

## 1. 收敛判据

> 一条结论算**收口**，当且仅当它满足：推翻它 → 有测试会 FAIL。
>
> 只写在审计报告里的结论**不算**收口：文档不会失败，探针不会在 CI 里跑。

方向 1 此前的问题是：13 个一次性探针、4 份审计文档，但只有 2 个测试文件
（`test_direct_estimation_error.py`、`test_residual_accounting_mode.py`），
且那两个只钉住**门控本身**（开关关着逐位不变），没钉住审计跑出来的**结论**。

---

## 2. 已收口（会失败的断言）

### 2.1 门控本身

| 文件 | 覆盖 |
|---|---|
| `tests/test_direct_estimation_error.py`（8） | δ 开关关闭时返回同一对象、不消耗 rng、逐位不变；δ>0 只改字典不改物理场；字典外能量出现；深度单调退化 |
| `tests/test_residual_accounting_mode.py`（12） | `measured`/`structural` 两口径只改记账不改算法；解析侧同口径；非法值被拒 |

### 2.2 审计结论（本轮新增，14 tests）

`tests/test_direction1_convergence.py`：

| 断言 | 钉住的结论 | 实测 |
|---|---|---|
| `test_estimation_dominates_the_residual_budget` | `estimation` 占 `i_res` 主导 ⇒ 动 structural 的旋钮在默认口径必然看不见 | 比值 >0.9（实际 0.998） |
| `test_production_kappa_is_nearly_blind_to_delta` | 默认口径看不见 δ | 0.62 dB（本 preset）/ 0.17 dB（扫描口径） |
| `test_structural_kappa_is_highly_sensitive_to_delta` | structural 口径看得见 δ | 62.78 → 52.99 dB |
| `test_structural_is_an_order_of_magnitude_more_sensitive` | 两口径敏感度差一个数量级（**比值型**，与 preset 无关） | ≈16× |
| `test_structural_depth_follows_inverse_square_law` | κ(δ) 平方律：两档标定、**预测**第三档 | 拟合 ≤0.12 dB |
| `test_depth_degrades_twenty_db_per_decade` | δ 每大十倍掉 20 dB | 15–25 dB 区间 |
| `test_delta_does_not_move_the_numerator` | δ 对分子的影响可忽略 | **4e-5**（见 §4 修正） |
| `test_delta_zero_structural_depth_is_an_upper_bound` | δ=0 是上界，不是可达值 | 单调 |
| `test_sync_error_keys_agree_by_default` | 同步误差在分子侧/分母侧必须是同一个数 | 当前都 0.0 |

---

## 3. 一次性探针（已得出结论，不再维护）

> **2026-09-21 归档**：这些脚本已迁到 `studies/direction1/scripts/`，产物迁到
> `studies/direction1/data/`，本文件迁到 `studies/direction1/docs/`。
> 汇总索引见 `studies/direction1/README.md`。

这些脚本**不是门禁**，跑它们不产生 FAIL。它们的结论已由 §2 的断言接管；
留着只是为了复现与审计留痕。不要再往这个列表加新成员。

⚠️ 下表 13 项中 **2 项已删除**（手工口径，数字作废），现存 **11 项**。

| 脚本 | 产出的结论 | 已由测试接管 |
|---|---|---|
| `verify_tpuic_residual_identity.py` | 残差线性分解成立（1e-29），`‖f(n)‖²` 不在残差里 | 结论属记账口径，由 `test_residual_accounting_mode` 覆盖 |
| `probe_noise_reference.py` | 链路表 `n0` ≡ 逐 bin `sigma2`（口径一致） | —（静态事实，无需回归） |
| `diag_rinr_budget.py` | `residual_self` 是自干扰地板，占 structural 臂 rinr 的 99.1% | — |
| `diag_kappa_decomposition.py` | κ 三项分解：`estimation/structural` ≈ 506 | ✅ `test_estimation_dominates_...` |
| `probe_offgrid_depth.py` | 深度按 `−20log10(ε)` 崩塌 | ✅ `test_depth_degrades_twenty_db_per_decade` |
| `sweep_direct_estimation_delta.py` | κ(δ) 曲线、切向列救援窗口 | ✅ `test_structural_depth_follows_...` |
| `diag_direction1_ceiling.py` | P_D(κ) 天花板；perfect 档 0.8833 | ⚠️ 端到端，**未收口**（见 §5） |
| ~~`run_tpuic_accounting_ab.py`~~ | 记账三臂 A/B（手工口径） | ⛔ **已删除**：`(i_res−i_est)` 灾难性抵消，数字作废 |
| ~~`run_direction1_prescription.py`~~ | 处方验证（手工口径） | ⛔ **已删除**：报 +0.267，门控复核修正为 +0.200（手工高估 0.067） |
| `run_direction1_gated.py` | 门控落地后的端到端复核 | ⚠️ 端到端，**未收口** |
| `diag_delta_gap_attribution.py` | 差距归因 H1/H2/H3 | ✅（H1/H2 由断言覆盖，H3 由分子断言覆盖） |
| `calc_delta_to_sync_ns.py` | δ ↔ 同步精度换算 | —（纯换算） |
| `diag_tpuic_depth_budget.py` | 残余干扰 99.95% 是 `‖f(n)‖²` | ✅ 同上 |

---

## 4. 本轮修正的一条结论

⚠️ **"δ 只动分母、不动分子"不精确，已修正为"δ 对分子的影响 ≤5e-5，可忽略"**。

臂级实测：η_survive 相对变化 `3.8e-5 / 5.0e-5 / 5.1e-5`（三个 seed），
非零 ⇒ 字典变了，保护子空间跟着变，存活率确实会动一点点。
端到端下这不足以改变判决（三档 δ 的"分子固定 vs 随臂" P_D 差全为 0）。

⇒ 之前的说法是**逐位意义**上的错误。已写进
`test_delta_does_not_move_the_numerator` 的 docstring 与断言界（1e-4）。

---

## 5. 明确**不再做**（停止条件）

| 项 | 判定 |
|---|---|
| 继续深挖对消算法本体（更深的 LS / 收缩 / 降 `d_eff`） | ⛔ **停**。中位结构残差已在自干扰地板下 20.8 dB，收益 ≈0 |
| 把 `n_cpi` 当深度杠杆 | ⛔ **停**。实测未接到对消臂（1/4/16/64 逐位相同） |
| 调切向阶数换性能 | ⛔ **停**。它是 δ 的保险不是增益；δ=0 时开它反亏 13 dB，平台 50.3 dB 低于需求 |
| 用 `(i_res − i_est)` 手工取结构残差 | ⛔ **禁**。灾难性抵消，实测高估 0.80 dB |
| 新增一次性探针 | ⛔ **停**。新结论必须落成测试，否则不写 |

---

## 6. 遗留（若要继续，按此顺序）

1. **δ 标定**（B 类）：当前 δ 是自由参数，收益是它的函数。
   应锚定到 `waveform_impairments.sync_*_bins`（同一物理量，两个键）。
   达标需 **δ ≤ 1.615e-3 格 ≈ 0.84 ns**；GPS 级（10 ns）下 κ 仅 31 dB ⇒ 收益归零。
2. **多普勒维**：本轮只注入延迟维，两维联合未扫。
3. **P_D 结论提到 MC≥200**：MC=20 的 SE≈0.05，同配置换 rng 得 0.7167/0.8000。
4. **门控默认是否打开**：仍默认 `measured`（发布数字不动）。打开 = B 类。
5. **方向 2**（协作融合 / 选错链路）：belief top-1 命中 24.2%，
   被根因审计指认为当前最大杠杆；δ 门控已就位，可随时报两版。
