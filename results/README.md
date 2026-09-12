# 论文数据重跑汇总（isac_sim）

> 由 `tools/rerun_paper.py` 以论文参数（M=15, Q=10, UAV 30-60 m/s, 目标 50-90 m/s, MC=1000, seed=2026）生成，`tools/summarize_results.py` 归纳。

## 1. 主对比（Fig. 2）

| Method | P_D | Bits (kbit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|
| proposed_lagrangian | 0.6105 | 17674.88 | 28.64 | 0.5920 |
| topk_deflection | 0.6191 | 17674.88 | 31.59 | 0.6040 |
| sense_sinr | 0.5855 | 17674.88 | 40.26 | 0.5630 |
| single_best | 0.5656 | 6400.00 | 8.64 | 0.5480 |
| nearest | 0.1488 | 17674.88 | 10.56 | 0.1400 |
| shortest_bistatic | 0.4783 | 17674.88 | 38.49 | 0.4540 |
| random | 0.1109 | 17674.88 | 42.88 | 0.1020 |
| all_neighbor | 0.6631 | 275827.20 | 675.39 | 0.6530 |

**派生数字（摘要口径）：**

- 保留 all-neighbor 检测概率：`92.1%`（论文 94.8%）
- 降低信令开销：`93.6%`（论文 92.1%）
- 降低交换时延：`95.8%`（论文 94.8%）
- 时延绝对值：all-neighbor `675.4` ms → proposed `28.6` ms
- 比 Top-K Deflection 时延低：`9.4%`（论文 29%）

## 2. C2F DD 精化（Fig. 2 / §4 DD 消融）

| Method | P_D | Fine-grid eval | T (ms) |
|---|---|---|---|
| proposed_lagrangian | 0.6105 | 0.0 | 28.64 |
| proposed_c2f | 0.6083 | 173.1 | 27.07 |
| proposed_c2f_full | 0.6116 | 173.1 | 27.07 |

- fine-grid 评估减少：`59.8%`（`173` vs `431`，论文 41.1%）

## 3. 组件消融（Table II，固定预算）

| Variant | P_D | Bits (kbit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|
| full | 0.6047 | 15553.92 | 25.45 | 0.5720 |
| w/o_alpha | 0.5932 | 16634.88 | 29.86 | 0.5710 |
| w/o_delay_price | 0.6077 | 16568.32 | 29.81 | 0.5860 |
| w/o_comm_error_calib | 0.5463 | 12869.76 | 26.78 | 0.5240 |
| w/o_softmin_alpha | 0.6223 | 16405.76 | 27.42 | 0.6040 |

## 4. DD 机制消融

| Variant | Method | P_D |
|---|---|---|
| full_dd | proposed_lagrangian | 0.6105 |
| full_dd | sense_sinr | 0.5855 |
| full_dd | single_best | 0.5656 |
| full_dd | all_neighbor | 0.6631 |
| w/o_dd_validity | proposed_lagrangian | 0.6105 |
| w/o_dd_validity | sense_sinr | 0.5855 |
| w/o_dd_validity | single_best | 0.5656 |
| w/o_dd_validity | all_neighbor | 0.6636 |
| w/o_dd_fractional_loss | proposed_lagrangian | 0.6922 |
| w/o_dd_fractional_loss | sense_sinr | 0.6628 |
| w/o_dd_fractional_loss | single_best | 0.6529 |
| w/o_dd_fractional_loss | all_neighbor | 0.7625 |
| w/o_dd_collision_penalty | proposed_lagrangian | 0.6106 |
| w/o_dd_collision_penalty | sense_sinr | 0.5872 |
| w/o_dd_collision_penalty | single_best | 0.5670 |
| w/o_dd_collision_penalty | all_neighbor | 0.6644 |
| w/o_all_dd_effects | proposed_lagrangian | 0.6952 |
| w/o_all_dd_effects | sense_sinr | 0.6635 |
| w/o_all_dd_effects | single_best | 0.6539 |
| w/o_all_dd_effects | all_neighbor | 0.7635 |

## λ_c 扫描（Fig. 3）

| x | Method | P_D | T (ms) | links |
|---|---|---|---|---|
| 0.001 | - | 0.6187 | 36.90 | 32.9 |
| 0.001 | - | 0.6085 | 34.83 | 31.6 |
| 0.003 | - | 0.6090 | 30.91 | 29.1 |
| 0.005 | - | 0.6105 | 28.64 | 27.6 |
| 0.010 | - | 0.6083 | 24.98 | 25.2 |
| 0.020 | - | 0.6106 | 21.15 | 22.4 |
| 0.050 | - | 0.6010 | 15.99 | 18.3 |
| 0.100 | - | 0.5938 | 12.85 | 15.4 |
| 0.200 | - | 0.5920 | 10.56 | 13.1 |

## R_min 扫描（Fig. 5）

| x | Method | P_D | T (ms) | links |
|---|---|---|---|---|
| 0.100 | proposed_lagrangian | 0.6889 | 35.98 | 25.5 |
| 0.100 | sense_sinr | 0.6606 | 60.81 | 25.5 |
| 0.100 | single_best | 0.6363 | 10.27 | 10.0 |
| 0.100 | all_neighbor | 0.7517 | 2086.03 | 732.2 |
| 0.200 | proposed_lagrangian | 0.6105 | 28.64 | 27.6 |
| 0.200 | sense_sinr | 0.5855 | 40.26 | 27.6 |
| 0.200 | single_best | 0.5656 | 8.64 | 10.0 |
| 0.200 | all_neighbor | 0.6631 | 675.39 | 431.0 |
| 0.500 | proposed_lagrangian | 0.4526 | 16.11 | 28.2 |
| 0.500 | sense_sinr | 0.4434 | 18.71 | 28.2 |
| 0.500 | single_best | 0.4222 | 5.66 | 10.0 |
| 0.500 | all_neighbor | 0.4806 | 124.75 | 181.3 |
| 1.000 | proposed_lagrangian | 0.2923 | 7.47 | 21.9 |
| 1.000 | sense_sinr | 0.2926 | 7.95 | 21.9 |
| 1.000 | single_best | 0.2807 | 3.52 | 10.0 |
| 1.000 | all_neighbor | 0.3071 | 31.48 | 83.9 |
| 2.000 | proposed_lagrangian | 0.1392 | 2.34 | 11.8 |
| 2.000 | sense_sinr | 0.1421 | 2.38 | 11.8 |
| 2.000 | single_best | 0.1483 | 1.95 | 9.3 |
| 2.000 | all_neighbor | 0.1562 | 6.79 | 32.4 |

## 鲁棒性（Fig. 6，axis = comm_model）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| erasure | proposed_lagrangian | 0.6105 | 28.64 |
| erasure | sense_sinr | 0.5855 | 40.26 |
| erasure | single_best | 0.5656 | 8.64 |
| erasure | all_neighbor | 0.6631 | 675.39 |
| biased | proposed_lagrangian | 0.6691 | 29.29 |
| biased | sense_sinr | 0.6667 | 38.38 |
| biased | single_best | 0.6135 | 8.64 |
| biased | all_neighbor | 0.7108 | 675.39 |
| flip | proposed_lagrangian | 0.5669 | 23.13 |
| flip | sense_sinr | 0.5252 | 43.20 |
| flip | single_best | 0.5422 | 8.64 |
| flip | all_neighbor | 0.5607 | 675.39 |

## 鲁棒性（Fig. 6，axis = error_sigma）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 1.0 | proposed_lagrangian | 0.6452 | 27.15 |
| 1.0 | sense_sinr | 0.6352 | 33.71 |
| 1.0 | single_best | 0.6272 | 8.64 |
| 1.0 | all_neighbor | 0.7553 | 675.39 |
| 2.0 | proposed_lagrangian | 0.6365 | 27.87 |
| 2.0 | sense_sinr | 0.6157 | 37.39 |
| 2.0 | single_best | 0.6058 | 8.64 |
| 2.0 | all_neighbor | 0.7102 | 675.39 |
| 3.0 | proposed_lagrangian | 0.6105 | 28.64 |
| 3.0 | sense_sinr | 0.5855 | 40.26 |
| 3.0 | single_best | 0.5656 | 8.64 |
| 3.0 | all_neighbor | 0.6631 | 675.39 |
| 4.0 | proposed_lagrangian | 0.5789 | 29.45 |
| 4.0 | sense_sinr | 0.5471 | 42.63 |
| 4.0 | single_best | 0.5231 | 8.64 |
| 4.0 | all_neighbor | 0.6157 | 675.39 |

## 鲁棒性（Fig. 6，axis = residual_direct）

| condition | Method | P_D | T (ms) |
|---|---|---|---|
| 1e-05 | proposed_lagrangian | 0.6115 | 28.59 |
| 1e-05 | sense_sinr | 0.5882 | 40.19 |
| 1e-05 | single_best | 0.5676 | 8.64 |
| 1e-05 | all_neighbor | 0.6652 | 675.39 |
| 0.0001 | proposed_lagrangian | 0.6105 | 28.64 |
| 0.0001 | sense_sinr | 0.5855 | 40.26 |
| 0.0001 | single_best | 0.5656 | 8.64 |
| 0.0001 | all_neighbor | 0.6631 | 675.39 |
| 0.001 | proposed_lagrangian | 0.5978 | 29.01 |
| 0.001 | sense_sinr | 0.5750 | 40.62 |
| 0.001 | single_best | 0.5513 | 8.65 |
| 0.001 | all_neighbor | 0.6468 | 675.39 |
| 0.01 | proposed_lagrangian | 0.4963 | 31.40 |
| 0.01 | sense_sinr | 0.4819 | 43.05 |
| 0.01 | single_best | 0.4481 | 8.79 |
| 0.01 | all_neighbor | 0.5257 | 675.39 |

