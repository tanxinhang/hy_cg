# 论文数据重跑汇总（isac_sim）

> **Legacy archive — not canonical release evidence.** 本目录由提交
> `603b61d` 的混合模型生成，包含 decoupled/active-set、heuristic reliability
> 等旧口径。投稿数字只能取自 `results_release/`，其配置契约见
> `PAPER_RELEASE.md`。下表原来的 `kbit` 表头有误；数值实际单位为 bit。

> 由 `tools/rerun_paper.py` 以论文参数（M=15, Q=10, UAV 30-60 m/s, 目标 50-90 m/s, MC=1000, seed=2026）生成，`tools/summarize_results.py` 归纳。

## 1. 主对比（Fig. 2）

| Method | P_D | Bits (bit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|
| proposed_lagrangian | 0.9481 | 9535.36 | 6.94 | 0.9350 |
| topk_deflection | 0.9470 | 9535.36 | 7.42 | 0.9370 |
| sense_sinr | 0.8705 | 9535.36 | 15.43 | 0.8580 |
| single_best | 0.9385 | 6400.00 | 2.54 | 0.9310 |
| nearest | 0.3358 | 9535.36 | 2.37 | 0.3180 |
| shortest_bistatic | 0.8148 | 9535.36 | 13.42 | 0.7880 |
| random | 0.2853 | 9535.36 | 16.92 | 0.2740 |
| all_neighbor | 0.9757 | 285977.60 | 688.45 | 0.9660 |

**派生数字（摘要口径）：**

- 保留 all-neighbor 检测概率：`97.2%`（论文 97.2%）
- 降低信令开销：`96.7%`（论文 96.7%）
- 降低交换时延：`99.0%`（论文 99.0%）
- 时延绝对值：all-neighbor `688.5` ms → proposed `6.9` ms
- 比 Top-K Deflection 时延低：`6.4%`（论文 6.5%）

### 1.1 干扰模型对照（active_set vs full_concurrent）

| 干扰模型 | proposed P_D | proposed 时延 (ms) | 保留率 |
|---|---|---|---|
| active_set（新，活跃集并发） | 0.9481 | 6.94 | 97.2% |
| full_concurrent（保守全并发） | 0.9146 | 9.72 | 93.8% |

## 2. C2F DD 精化（Fig. 2 / §4 DD 消融）

| Method | P_D | Fine-grid eval | T (ms) | Bits (bit) |
|---|---|---|---|---|
| proposed_lagrangian | 0.9481 | 0.0 | 6.94 | 9535 |
| proposed_c2f | 0.9572 | 247.6 | 5.76 | 8856 |
| proposed_c2f_full | 0.9537 | 446.8 | 5.76 | 8856 |
| all_neighbor | 0.9757 | 0.0 | 688.45 | 285978 |

- **C2F 精化增益**：P_D `0.9481` → `0.9572`（+0.0091，论文 0.9481→0.9572）
- **C2F 匹配全量精化**：`0.9572` vs `0.9537`（差 0.0035），但 fine-grid 评估只用 `248` / `447`
- **fine-grid 评估减少**：`44.6%`（`248` vs `447`，论文 44.6%）

## 3. 组件消融（Table II，固定预算）

| Variant | P_D | Bits (bit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|
| full | 0.9469 | 9170.56 | 6.56 | 0.9380 |
| w/o_alpha | 0.8504 | 15880.32 | 9.40 | 0.8260 |
| w/o_delay_price | 0.9482 | 9425.28 | 7.18 | 0.9380 |
| w/o_comm_error_calib | 0.8445 | 6566.40 | 8.15 | 0.8300 |
| w/o_softmin_alpha | 0.9540 | 9998.72 | 7.18 | 0.9470 |

## 4. DD 机制消融

| Variant | Method | P_D |
|---|---|---|
| full_dd | proposed_lagrangian | 0.9481 |
| full_dd | sense_sinr | 0.8705 |
| full_dd | single_best | 0.9385 |
| full_dd | all_neighbor | 0.9757 |
| w/o_dd_validity | proposed_lagrangian | 0.9481 |
| w/o_dd_validity | sense_sinr | 0.8705 |
| w/o_dd_validity | single_best | 0.9385 |
| w/o_dd_validity | all_neighbor | 0.9756 |
| w/o_dd_fractional_loss | proposed_lagrangian | 0.9648 |
| w/o_dd_fractional_loss | sense_sinr | 0.8704 |
| w/o_dd_fractional_loss | single_best | 0.9606 |
| w/o_dd_fractional_loss | all_neighbor | 0.9894 |
| w/o_dd_collision_penalty | proposed_lagrangian | 0.9488 |
| w/o_dd_collision_penalty | sense_sinr | 0.8704 |
| w/o_dd_collision_penalty | single_best | 0.9392 |
| w/o_dd_collision_penalty | all_neighbor | 0.9760 |
| w/o_all_dd_effects | proposed_lagrangian | 0.9648 |
| w/o_all_dd_effects | sense_sinr | 0.8715 |
| w/o_all_dd_effects | single_best | 0.9612 |
| w/o_all_dd_effects | all_neighbor | 0.9896 |

## λ_c 扫描（Fig. 3）

| x | Method | P_D | T (ms) | links |
|---|---|---|---|---|
| 0.0005 | - | 0.9523 | 8.50 | 16.5 |
| 0.0010 | - | 0.9511 | 7.99 | 16.0 |
| 0.0030 | - | 0.9498 | 7.24 | 15.2 |
| 0.0050 | - | 0.9481 | 6.94 | 14.9 |
| 0.0100 | - | 0.9478 | 6.52 | 14.5 |
| 0.0200 | - | 0.9488 | 6.10 | 14.1 |
| 0.0500 | - | 0.9482 | 5.44 | 13.5 |
| 0.1000 | - | 0.9485 | 4.84 | 13.1 |
| 0.2000 | - | 0.9477 | 4.29 | 12.6 |

## R_min 扫描（Fig. 5）

| x | Method | P_D | T (ms) | links |
|---|---|---|---|---|
| 0.1000 | proposed_lagrangian | 0.9678 | 4.81 | 11.6 |
| 0.1000 | sense_sinr | 0.8927 | 15.65 | 11.6 |
| 0.1000 | single_best | 0.9644 | 2.80 | 10.0 |
| 0.1000 | all_neighbor | 0.9901 | 2103.40 | 748.5 |
| 0.2000 | proposed_lagrangian | 0.9481 | 6.94 | 14.9 |
| 0.2000 | sense_sinr | 0.8705 | 15.43 | 14.9 |
| 0.2000 | single_best | 0.9385 | 2.54 | 10.0 |
| 0.2000 | all_neighbor | 0.9757 | 688.45 | 446.8 |
| 0.5000 | proposed_lagrangian | 0.8954 | 12.18 | 29.2 |
| 0.5000 | sense_sinr | 0.8564 | 16.55 | 29.2 |
| 0.5000 | single_best | 0.8656 | 2.16 | 10.0 |
| 0.5000 | all_neighbor | 0.9084 | 114.93 | 189.2 |
| 1.0000 | proposed_lagrangian | 0.7768 | 8.07 | 35.8 |
| 1.0000 | sense_sinr | 0.7769 | 8.75 | 35.8 |
| 1.0000 | single_best | 0.7406 | 1.63 | 10.0 |
| 1.0000 | all_neighbor | 0.7824 | 21.99 | 88.0 |
| 2.0000 | proposed_lagrangian | 0.4797 | 2.40 | 22.0 |
| 2.0000 | sense_sinr | 0.4758 | 2.43 | 22.0 |
| 2.0000 | single_best | 0.4553 | 0.95 | 9.4 |
| 2.0000 | all_neighbor | 0.4805 | 3.86 | 34.3 |

## 鲁棒性（Fig. 6，axis = comm_model）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| erasure | proposed_lagrangian | 0.9481 | 6.94 |
| erasure | sense_sinr | 0.8705 | 15.43 |
| erasure | single_best | 0.9385 | 2.54 |
| erasure | all_neighbor | 0.9757 | 688.45 |
| biased | proposed_lagrangian | 0.9756 | 5.76 |
| biased | sense_sinr | 0.9728 | 12.53 |
| biased | single_best | 0.9658 | 2.72 |
| biased | all_neighbor | 0.9855 | 688.45 |
| flip | proposed_lagrangian | 0.9050 | 9.48 |
| flip | sense_sinr | 0.8018 | 24.61 |
| flip | single_best | 0.9175 | 2.22 |
| flip | all_neighbor | 0.9045 | 688.45 |

## 鲁棒性（Fig. 6，axis = error_sigma）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 1.0 | proposed_lagrangian | 0.9488 | 4.87 |
| 1.0 | sense_sinr | 0.8516 | 11.87 |
| 1.0 | single_best | 0.9383 | 2.43 |
| 1.0 | all_neighbor | 0.9907 | 688.45 |
| 2.0 | proposed_lagrangian | 0.9508 | 5.81 |
| 2.0 | sense_sinr | 0.8631 | 13.47 |
| 2.0 | single_best | 0.9410 | 2.48 |
| 2.0 | all_neighbor | 0.9839 | 688.45 |
| 3.0 | proposed_lagrangian | 0.9481 | 6.94 |
| 3.0 | sense_sinr | 0.8705 | 15.43 |
| 3.0 | single_best | 0.9385 | 2.54 |
| 3.0 | all_neighbor | 0.9757 | 688.45 |
| 4.0 | proposed_lagrangian | 0.9446 | 8.27 |
| 4.0 | sense_sinr | 0.8666 | 17.82 |
| 4.0 | single_best | 0.9332 | 2.61 |
| 4.0 | all_neighbor | 0.9665 | 688.45 |

## 鲁棒性（Fig. 6，axis = residual_direct）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 1e-05 | proposed_lagrangian | 0.6550 | 29.23 |
| 1e-05 | sense_sinr | 0.6422 | 42.46 |
| 1e-05 | single_best | 0.6425 | 4.41 |
| 1e-05 | all_neighbor | 0.6677 | 665.61 |
| 0.0001 | proposed_lagrangian | 0.6506 | 29.21 |
| 0.0001 | sense_sinr | 0.6408 | 42.39 |
| 0.0001 | single_best | 0.6385 | 4.40 |
| 0.0001 | all_neighbor | 0.6646 | 665.61 |
| 0.001 | proposed_lagrangian | 0.6272 | 29.39 |
| 0.001 | sense_sinr | 0.6125 | 42.21 |
| 0.001 | single_best | 0.6193 | 4.42 |
| 0.001 | all_neighbor | 0.6404 | 665.61 |
| 0.01 | proposed_lagrangian | 0.4970 | 29.78 |
| 0.01 | sense_sinr | 0.4849 | 41.54 |
| 0.01 | single_best | 0.4964 | 4.65 |
| 0.01 | all_neighbor | 0.5014 | 665.61 |

## 鲁棒性（Fig. 6，axis = direct_cancellation）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 10.0 | proposed_lagrangian | 0.0318 | 4.30 |
| 10.0 | sense_sinr | 0.0319 | 5.46 |
| 10.0 | single_best | 0.0486 | 6.17 |
| 10.0 | all_neighbor | 0.0559 | 688.45 |
| 20.0 | proposed_lagrangian | 0.3578 | 34.07 |
| 20.0 | sense_sinr | 0.3491 | 44.25 |
| 20.0 | single_best | 0.3515 | 5.81 |
| 20.0 | all_neighbor | 0.3545 | 688.45 |
| 30.0 | proposed_lagrangian | 0.8494 | 21.45 |
| 30.0 | sense_sinr | 0.8140 | 33.76 |
| 30.0 | single_best | 0.8267 | 3.93 |
| 30.0 | all_neighbor | 0.8659 | 688.45 |
| 40.0 | proposed_lagrangian | 0.9481 | 6.94 |
| 40.0 | sense_sinr | 0.8705 | 15.43 |
| 40.0 | single_best | 0.9385 | 2.54 |
| 40.0 | all_neighbor | 0.9757 | 688.45 |
| 50.0 | proposed_lagrangian | 0.9622 | 4.72 |
| 50.0 | sense_sinr | 0.8707 | 12.43 |
| 50.0 | single_best | 0.9568 | 2.21 |
| 50.0 | all_neighbor | 0.9873 | 688.45 |
| 60.0 | proposed_lagrangian | 0.9655 | 4.41 |
| 60.0 | sense_sinr | 0.8679 | 12.07 |
| 60.0 | single_best | 0.9580 | 2.16 |
| 60.0 | all_neighbor | 0.9886 | 688.45 |

## 5. 理论深化实验（新模型 / 保证）

### 5.1 belief 失配（truth vs belief）

| belief σ_p (m) | Method | P_D | capture |
|---|---|---|---|
| 0 | proposed_lagrangian | 0.9481 | 1.000 |
| 0 | all_neighbor | 0.9757 | 1.000 |
| 50 | proposed_lagrangian | 0.3784 | 0.372 |
| 50 | all_neighbor | 0.7964 | 0.368 |
| 150 | proposed_lagrangian | 0.0762 | 0.072 |
| 150 | all_neighbor | 0.3461 | 0.072 |
| 300 | proposed_lagrangian | 0.0249 | 0.022 |
| 300 | all_neighbor | 0.1556 | 0.023 |

### 5.2 有限块长可靠性（FBL）

| n_block | χ_mean | P_D | T (ms) |
|---|---|---|---|
| 256 | 0.4040 | 0.1400 | 0.665 |
| 512 | 0.9179 | 0.5123 | 3.909 |
| 1024 | 0.9999 | 0.8653 | 9.222 |
| 2048 | 0.9999 | 0.9598 | 13.100 |
| 4096 | 1.0000 | 0.9873 | 22.308 |

### 5.3 相关感知融合消融

| corr | P_D | D_mean | links |
|---|---|---|---|
| False | 0.9481 | 25.102 | 14.9 |
| True | 0.9448 | 24.494 | 15.7 |

### 5.4 次模性审计与 greedy 保证

| mono. viol. | submod. viol. | curvature | guarantee |
|---|---|---|---|
| 0.0000 | 0.0000 | 0.8964 | 0.6604 |

### 5.5 同目标 greedy-vs-oracle 间隙

- mean gap：`0.0198`，median `0.0022`，max `0.2507`

### 5.6 通信/感知干扰耦合与直射对消预算

**干扰记账（同口径 vs 解耦残差）**

| variant | near-far (dB) | I_comm/N0 (dB) | I_sense/N0 (dB, model) | I_comm/I_sense (dB) | γ^s (dB) | P_D | links |
|---|---|---|---|---|---|---|---|
| legacy | 41.3 | 29.4 | -16.3 | 33.3 | -22.0 | 0.6506 | 33.0 |
| legacy+guard | 41.3 | 29.4 | -2.0 | 33.3 | -10.0 | 0.9629 | 14.0 |
| coupled | 41.3 | 29.4 | -16.0 | 33.3 | -22.0 | 0.6447 | 33.1 |
| coupled+guard | 41.3 | 29.4 | -1.7 | 33.3 | -10.1 | 0.9481 | 14.9 |

**直射对消扫描（ISAC 可行性门限）**

| κ_dc (dB) | I_sense/N0 (dB) | γ^s (dB) | P_D (proposed) | P_D (all-neighbour) | T (ms) |
|---|---|---|---|---|---|
| 0 | 36.1 | -43.9 | 0.0021 | 0.0465 | 0.08 |
| 10 | 26.1 | -33.9 | 0.0318 | 0.0559 | 4.30 |
| 20 | 16.1 | -24.1 | 0.3578 | 0.3545 | 34.07 |
| 30 | 6.4 | -15.2 | 0.8494 | 0.8659 | 21.45 |
| 40 | -1.7 | -10.1 | 0.9481 | 0.9757 | 6.94 |
| 50 | -5.2 | -8.8 | 0.9622 | 0.9873 | 4.72 |
| 60 | -5.8 | -8.6 | 0.9655 | 0.9886 | 4.41 |
| 80 | -5.8 | -8.6 | 0.9658 | 0.9886 | 4.40 |

> 论文 Table「Direct-path suppression budget」的数据源。运行口径由每行的 `interference_model`/`coupling`/`eps_mode` 列显式记录——
> 不同口径的结果不可直接比较（论文用 active_set）。
> 关键结论：近远比 ρ_NF ≈ 41 dB；零对消时连 all-neighbour 也只到 P_FA 水平，
> 说明调度无法替代直射抑制。

## 6. 旧模型 vs 修正模型（同一运行口径，MC=1000）

> 由 `tools/compare_isac_models.py` 生成；所有模型都固定论文工作点
> （active_set + 论文运动学），因此差异只来自干扰记账与 SINR 保护项。

| model | method | P_D | links | T (ms) | bits | chi |
|---|---|---|---|---|---|---|
| legacy | proposed_lagrangian | 0.6506 | 33.0 | 29.21 | 21130 | 0.801 |
| legacy | all_neighbor | 0.6646 | 431.0 | 665.61 | 275827 | 0.712 |
| legacy + guard fix | proposed_lagrangian | 0.9629 | 14.0 | 5.16 | 8941 | 0.932 |
| legacy + guard fix | all_neighbor | 0.9809 | 446.8 | 686.88 | 285978 | 0.713 |
| coupled | proposed_lagrangian | 0.6447 | 33.1 | 30.22 | 21183 | 0.796 |
| coupled | all_neighbor | 0.6619 | 431.0 | 667.24 | 275827 | 0.711 |
| coupled + guard fix | proposed_lagrangian | 0.9481 | 14.9 | 6.94 | 9535 | 0.910 |
| coupled + guard fix | all_neighbor | 0.9757 | 446.8 | 688.45 | 285978 | 0.712 |
| coupled, strict | proposed_lagrangian | 0.6272 | 27.3 | 2.44 | 17448 | 1.000 |
| coupled, strict | all_neighbor | 0.6467 | 135.7 | 89.17 | 86851 | 0.826 |

- **legacy**：proposed $P_D$ = `0.6506`，时延 `29.21` ms，链路 `33.0`
- **coupled + guard fix**：proposed $P_D$ = `0.9481`，时延 `6.94` ms，链路 `14.9`

