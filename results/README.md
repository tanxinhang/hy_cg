# 论文数据重跑汇总（isac_sim）

> 由 `tools/rerun_paper.py` 以论文参数（M=15, Q=10, UAV 30-60 m/s, 目标 50-90 m/s, MC=1000, seed=2026）生成，`tools/summarize_results.py` 归纳。

## 1. 主对比（Fig. 2）

| Method | P_D | Bits (kbit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|
| proposed_lagrangian | 0.6506 | 21129.60 | 29.21 | 0.6270 |
| topk_deflection | 0.6593 | 21129.60 | 32.64 | 0.6450 |
| sense_sinr | 0.6408 | 21129.60 | 42.39 | 0.6240 |
| single_best | 0.6318 | 6400.00 | 4.74 | 0.6160 |
| nearest | 0.1842 | 21129.60 | 7.44 | 0.1650 |
| shortest_bistatic | 0.5497 | 21129.60 | 39.80 | 0.5290 |
| random | 0.1374 | 21129.60 | 45.29 | 0.1230 |
| all_neighbor | 0.6646 | 275827.20 | 665.61 | 0.6350 |

**派生数字（摘要口径）：**

- 保留 all-neighbor 检测概率：`97.9%`（论文 94.8%）
- 降低信令开销：`92.3%`（论文 92.1%）
- 降低交换时延：`95.6%`（论文 94.8%）
- 时延绝对值：all-neighbor `665.6` ms → proposed `29.2` ms
- 比 Top-K Deflection 时延低：`10.5%`（论文 29%）

### 1.1 干扰模型对照（active_set vs full_concurrent）

| 干扰模型 | proposed P_D | proposed 时延 (ms) | 保留率 |
|---|---|---|---|
| active_set（新，活跃集并发） | 0.6506 | 29.21 | 97.9% |
| full_concurrent（保守全并发） | 0.6303 | 33.84 | 95.2% |

## 2. C2F DD 精化（Fig. 2 / §4 DD 消融）

| Method | P_D | Fine-grid eval | T (ms) | Bits (kbit) |
|---|---|---|---|---|
| proposed_lagrangian | 0.6506 | 0.0 | 29.21 | 21130 |
| proposed_c2f | 0.6929 | 161.5 | 27.70 | 20523 |
| proposed_c2f_full | 0.6928 | 431.0 | 27.79 | 20580 |
| all_neighbor | 0.6646 | 0.0 | 665.61 | 275827 |

- **C2F 精化增益**：P_D `0.6506` → `0.6929`（+0.0423，论文 0.6262→0.6648）
- **C2F 匹配全量精化**：`0.6929` vs `0.6928`（差 0.0001），但 fine-grid 评估只用 `162` / `431`
- **fine-grid 评估减少**：`62.5%`（`162` vs `431`，论文 41.1%）

## 3. 组件消融（Table II，固定预算）

| Variant | P_D | Bits (kbit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|
| full | 0.6547 | 16412.16 | 21.71 | 0.6380 |
| w/o_alpha | 0.6186 | 16635.52 | 22.03 | 0.6000 |
| w/o_delay_price | 0.6538 | 16607.36 | 23.65 | 0.6430 |
| w/o_comm_error_calib | 0.6061 | 12967.68 | 20.56 | 0.5840 |
| w/o_softmin_alpha | 0.6659 | 16615.04 | 21.65 | 0.6530 |

## 4. DD 机制消融

| Variant | Method | P_D |
|---|---|---|
| full_dd | proposed_lagrangian | 0.6506 |
| full_dd | sense_sinr | 0.6408 |
| full_dd | single_best | 0.6318 |
| full_dd | all_neighbor | 0.6646 |
| w/o_dd_validity | proposed_lagrangian | 0.6506 |
| w/o_dd_validity | sense_sinr | 0.6408 |
| w/o_dd_validity | single_best | 0.6318 |
| w/o_dd_validity | all_neighbor | 0.6648 |
| w/o_dd_fractional_loss | proposed_lagrangian | 0.7502 |
| w/o_dd_fractional_loss | sense_sinr | 0.7195 |
| w/o_dd_fractional_loss | single_best | 0.7204 |
| w/o_dd_fractional_loss | all_neighbor | 0.7670 |
| w/o_dd_collision_penalty | proposed_lagrangian | 0.6499 |
| w/o_dd_collision_penalty | sense_sinr | 0.6420 |
| w/o_dd_collision_penalty | single_best | 0.6332 |
| w/o_dd_collision_penalty | all_neighbor | 0.6648 |
| w/o_all_dd_effects | proposed_lagrangian | 0.7515 |
| w/o_all_dd_effects | sense_sinr | 0.7212 |
| w/o_all_dd_effects | single_best | 0.7220 |
| w/o_all_dd_effects | all_neighbor | 0.7682 |

## λ_c 扫描（Fig. 3）

| x | Method | P_D | T (ms) | links |
|---|---|---|---|---|
| 0.0005 | - | 0.6556 | 40.35 | 40.6 |
| 0.0010 | - | 0.6560 | 37.28 | 38.5 |
| 0.0030 | - | 0.6548 | 32.10 | 34.9 |
| 0.0050 | - | 0.6506 | 29.21 | 33.0 |
| 0.0100 | - | 0.6510 | 24.83 | 29.9 |
| 0.0200 | - | 0.6539 | 20.20 | 26.4 |
| 0.0500 | - | 0.6483 | 13.87 | 21.3 |
| 0.1000 | - | 0.6547 | 9.92 | 17.7 |
| 0.2000 | - | 0.6530 | 6.94 | 14.7 |

## R_min 扫描（Fig. 5）

| x | Method | P_D | T (ms) | links |
|---|---|---|---|---|
| 0.1000 | proposed_lagrangian | 0.7261 | 33.86 | 29.2 |
| 0.1000 | sense_sinr | 0.6980 | 60.31 | 29.2 |
| 0.1000 | single_best | 0.6908 | 5.66 | 10.0 |
| 0.1000 | all_neighbor | 0.7493 | 2082.24 | 732.2 |
| 0.2000 | proposed_lagrangian | 0.6506 | 29.21 | 33.0 |
| 0.2000 | sense_sinr | 0.6408 | 42.39 | 33.0 |
| 0.2000 | single_best | 0.6318 | 4.74 | 10.0 |
| 0.2000 | all_neighbor | 0.6646 | 665.61 | 431.0 |
| 0.5000 | proposed_lagrangian | 0.5111 | 15.59 | 34.2 |
| 0.5000 | sense_sinr | 0.5052 | 18.39 | 34.2 |
| 0.5000 | single_best | 0.5184 | 2.98 | 10.0 |
| 0.5000 | all_neighbor | 0.5129 | 107.16 | 181.3 |
| 1.0000 | proposed_lagrangian | 0.3841 | 4.96 | 25.2 |
| 1.0000 | sense_sinr | 0.3832 | 5.29 | 25.2 |
| 1.0000 | single_best | 0.3828 | 1.58 | 10.0 |
| 1.0000 | all_neighbor | 0.3840 | 19.07 | 83.9 |
| 2.0000 | proposed_lagrangian | 0.2132 | 1.09 | 12.7 |
| 2.0000 | sense_sinr | 0.2154 | 1.11 | 12.7 |
| 2.0000 | single_best | 0.2199 | 0.73 | 9.3 |
| 2.0000 | all_neighbor | 0.2255 | 2.99 | 32.4 |

## 鲁棒性（Fig. 6，axis = comm_model）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| erasure | proposed_lagrangian | 0.6506 | 29.21 |
| erasure | sense_sinr | 0.6408 | 42.39 |
| erasure | single_best | 0.6318 | 4.74 |
| erasure | all_neighbor | 0.6646 | 665.61 |
| biased | proposed_lagrangian | 0.6991 | 28.84 |
| biased | sense_sinr | 0.6993 | 39.30 |
| biased | single_best | 0.6581 | 4.74 |
| biased | all_neighbor | 0.7083 | 665.61 |
| flip | proposed_lagrangian | 0.5824 | 23.41 |
| flip | sense_sinr | 0.5526 | 45.86 |
| flip | single_best | 0.6174 | 4.74 |
| flip | all_neighbor | 0.5662 | 665.61 |

## 鲁棒性（Fig. 6，axis = error_sigma）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 1.0 | proposed_lagrangian | 0.7299 | 29.57 |
| 1.0 | sense_sinr | 0.7102 | 39.82 |
| 1.0 | single_best | 0.6743 | 4.74 |
| 1.0 | all_neighbor | 0.7546 | 665.61 |
| 2.0 | proposed_lagrangian | 0.6953 | 29.07 |
| 2.0 | sense_sinr | 0.6704 | 40.95 |
| 2.0 | single_best | 0.6583 | 4.74 |
| 2.0 | all_neighbor | 0.7080 | 665.61 |
| 3.0 | proposed_lagrangian | 0.6506 | 29.21 |
| 3.0 | sense_sinr | 0.6408 | 42.39 |
| 3.0 | single_best | 0.6318 | 4.74 |
| 3.0 | all_neighbor | 0.6646 | 665.61 |
| 4.0 | proposed_lagrangian | 0.6133 | 29.53 |
| 4.0 | sense_sinr | 0.5906 | 43.71 |
| 4.0 | single_best | 0.6009 | 4.74 |
| 4.0 | all_neighbor | 0.6202 | 665.61 |

## 鲁棒性（Fig. 6，axis = residual_direct）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 1e-05 | proposed_lagrangian | 0.6550 | 29.23 |
| 1e-05 | sense_sinr | 0.6422 | 42.46 |
| 1e-05 | single_best | 0.6358 | 4.74 |
| 1e-05 | all_neighbor | 0.6677 | 665.61 |
| 0.0001 | proposed_lagrangian | 0.6506 | 29.21 |
| 0.0001 | sense_sinr | 0.6408 | 42.39 |
| 0.0001 | single_best | 0.6318 | 4.74 |
| 0.0001 | all_neighbor | 0.6646 | 665.61 |
| 0.001 | proposed_lagrangian | 0.6272 | 29.39 |
| 0.001 | sense_sinr | 0.6125 | 42.21 |
| 0.001 | single_best | 0.6128 | 4.64 |
| 0.001 | all_neighbor | 0.6404 | 665.61 |
| 0.01 | proposed_lagrangian | 0.4970 | 29.78 |
| 0.01 | sense_sinr | 0.4849 | 41.54 |
| 0.01 | single_best | 0.4920 | 4.58 |
| 0.01 | all_neighbor | 0.5014 | 665.61 |

