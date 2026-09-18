# 归档结果：2026-09-18

本目录保存从项目根目录移出的 **93 个结果产物**（543 个文件，11.8 MB）。
**未做物理删除**——全部可原样取回。

## 为什么归档

清理两类不再符合真实实验场景的产物：

| 类 | 判据 | 数量 |
|---|---|---|
| **A 硬件增益** | 雷达净增益 `G_hw ≠ 0 dB` 且 ≥ 15 dB | 6 |
| **B 越界场景** | 边长 > 800 m **或** 目标 RCS > 1 m² | 87 |

论文场景已改判到 **500–800 m × RCS 0.05–0.2 m²**，
因此 4 km × 4000 m / RCS 50 m² 那一代实验，以及所有依赖额外硬件
增益才能闭合的臂，都不再是可发布口径。

## A 类明细（硬件增益）

| 产物 | G_hw |
|---|---|
| `results_coord_600m_g15k40` | 15 dB |
| `results_low_rcs_rescue_gain15_screen` | 15 dB |
| `results_low_rcs_rescue_gain17p5_screen` | 17.5 dB |
| `results_low_rcs_rescue_gain17p5_maxmin_screen` | 17.5 dB |
| `results_small_uav_s2_candidate` | 27 dB（tx/rx 各 16 dBi，损耗 5 dB） |
| `results_small_uav_s2_nearest` | 27 dB（同上） |
| `results_radar_gain_s2_coarse.csv` | 0/10/20/30 dB，且 s2 为 2000 m |
| `results_radar_gain_s2_refined.csv` | 12.5/15/17.5 dB，且 s2 为 2000 m |

> `results_coord_600m_g5k40`（5 dB）**未归档**，保留为"15 dB 可降档"
> 的对照证据。如需一并清掉，直接移入本目录即可。

## 恢复方法

```bash
# 恢复单个
cp -r _archive/2026-09-18/<name> .

# 全部恢复
cp -r _archive/2026-09-18/* . && rm _archive/2026-09-18/MANIFEST.csv
```

`MANIFEST.csv` 记录了每一项的归档原因、文件数与字节数。

## 注意：仍有文档引用这些路径

共 **38 处**引用分布在 20 个 `.md` / `.tex` 文件中，
完整清单见 `tools/_cleanup_dead_references.csv`（含行号）。

其中 `Conference-LaTeX-template_10-17-19/ReproducibilitySupplement.tex`
引用了 `results_v1_probability_refinement`。论文主结果表
（`SimulationResults.tex:27-28/47/56`）目前仍写 4000 m / RCS 50 m²，
改判完成前请不要删除本目录。

## 复现工具

```bash
python tools/scan_cleanup_candidates.py   # 重新扫描分类
python tools/apply_cleanup.py             # dry-run，打印计划
python tools/apply_cleanup.py --apply     # 执行归档移动
```
