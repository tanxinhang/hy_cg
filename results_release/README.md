# 论文数据重跑汇总（isac_sim）

> 由 `tools/rerun_paper.py` 以 canonical 论文参数生成，`tools/summarize_results.py` 归纳。主对比使用 MC=1000、seed=2026；辅助审计的试验数以各目录 `config.json` 为准。

## 1. 主对比（Fig. 2）

| Method | P_D | Proposed - method (95% CI) | Bits (kbit) | Delay (ms) | Worst P_D |
|---|---|---|---|---|---|
| proposed_c2f | 0.8655 | +0.0000 [+0.0000, +0.0000] | 9.84 | 16.40 | 0.8540 |
| global_topk_deflection | 0.4269 | +0.4386 [+0.4235, +0.4537] | 9.84 | 16.40 | 0.4050 |
| cost_aware_greedy | 0.8603 | +0.0052 [+0.0016, +0.0088] | 9.84 | 16.40 | 0.8490 |
| exact_marginal_greedy | 0.8603 | +0.0052 [+0.0016, +0.0088] | 9.84 | 16.40 | 0.8490 |
| sense_sinr | 0.8436 | +0.0219 [+0.0174, +0.0264] | 9.84 | 16.40 | 0.8270 |
| all_neighbor | 0.9070 | -0.0415 [-0.0469, -0.0361] | 383.50 | 639.17 | 0.8980 |

**派生数字（摘要口径）：**

- 保留 all-neighbor 检测概率：`95.4%`
- 降低信令开销：`97.4%`
- 降低交换时延：`97.4%`
- 时延绝对值：all-neighbor `639.2` ms → proposed `16.4` ms

## 5. 结构审计（有限实例，不构成保证）

### 5.4 有限实例的边际收益与曲率审计

| mono. viol. | submod. viol. | curvature | matroid reference |
|---|---|---|---|
| 0.4798 | 0.4522 | 0.9000 | 0.5263 |

### 5.5 同目标 greedy-vs-oracle 间隙

- mean gap：`0.0000`，median `0.0000`，max `0.0000`
