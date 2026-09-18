# Target-local fusion V1：修复后实验总览

> **2026-09-16 重跑**：本树全部实验已在当前冻结代码下重跑一遍（检测阶段的
> 随机数流由共享顺序流改为逐链路键控流）。**选路面量逐位不变**，仅判决计数变化。
> 主比较 $P_D$ 变动 ≤0.0003。重跑前快照见
> `archive/results_target_local_v1_pre_rngfix_2026-09-16/`，
> 核对工具 `tools/report_v1_rerun_drift.py`。

## 1. 核心结论

V1 已完成 optimization contract、Swerling 命名、感知/报告拓扑、本地证据、
`P_D` 域公平要求和 CI 门控修复。现在必须区分：

- `observation`：进入融合的感知三元组；
- `remote report`：需要通过 `j -> f_q` 发送的 640-bit 包；
- `j=f_q`：本地证据，可靠率 1，通信量与报告时延均为 0。

## 2. MC=1000 主对比

| Method | \(P_D\) | 95% CI | Worst \(P_D\) | Observations | Remote reports | kbit | ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| Adaptive detector-PD C2F | 0.9764 | [0.9732, 0.9792] | 0.966 | 12.101 | 0.751 | 0.481 | 0.801 |
| Exact-marginal greedy | 0.9408 | [0.9360, 0.9453] | 0.931 | 11.728 | 5.131 | 3.284 | 5.473 |
| Sensing-SINR | 0.9784 | [0.9754, 0.9811] | 0.972 | 12.101 | 4.595 | 2.941 | 4.901 |

V1 对 exact-marginal 的配对增益为 `+0.0356 [0.0311, 0.0401]`。V1 与
Sensing-SINR 的差为 `-0.0020 [-0.0050, 0.0010]`，检测率无显著差异；
但在相同观测数下，V1 的 payload 低 83.7%。

## 3. 融合规则消融，MC=100

| Rule | \(P_D\) | Worst \(P_D\) | Observations | Remote reports | Delay (ms) | Fusion UAVs | Conflict slots |
|---|---:|---:|---:|---:|---:|---:|---:|
| max-in-rate | 0.932 | 0.900 | 19.89 | 6.65 | 7.093 | 1.00 | 6.65 |
| max-min-rate | 0.967 | 0.940 | 15.61 | 6.40 | 6.827 | 1.00 | 6.40 |
| nearest-target | 0.975 | 0.960 | 12.24 | 0.74 | 0.789 | 7.13 | 0.70 |

主要机制不再只是“短链路更可靠”，而是 target-local 融合节点能够直接保留
其本地产生的统计量。该机制也说明当前通信阈值扫描较弱，不能把结果外推到
缺少本地证据的架构。

## 4. Full-refinement detector-PD 对照，MC=200

| Variant | \(P_D\) | Observations | Reports | Fine evaluations |
|---|---:|---:|---:|---:|
| Adaptive C2F | 0.9775 | 12.055 | 0.730 | 61.06 |
| Full refinement | 0.9715 | 12.055 | 0.730 | 853.65 |

Adaptive 减少 92.85% fine evaluations，配对检测差为
`+0.0060 [0.0005, 0.0115]`。这说明 shortlist 在 belief mismatch 下还可能
产生正则化效果；full refinement 是计算对照，不是性能 oracle。

## 5. 运行边界，MC=100/condition

| Sweep | Low/nominal/high summary |
|---|---|
| Position error | 0/50/150/300/500 m 对应 \(P_D\)=0.990/0.988/0.975/0.937/0.906 |
| \(R_{\min}\) | 0.1--1 Mbit/s 均为 \(P_D=0.975\)，2 Mbit/s 时 0.972；本地证据使全部目标仍可行 |
| Area side | 2500/4000/5500 m 对应 \(P_D\)=0.989/0.975/0.940，reports=0.05/0.74/2.67 |
| \(\lambda_c\) | 0/0.005/0.02 对应 reports=4.96/0.74/0.56，\(P_D\)=0.974/0.975/0.974 |
| Blocklength | 1024/2048/3072 uses 对应 delay=0.363/0.789/1.216 ms，\(P_D\)=0.972/0.975/0.975 |
| UAV count | 10/15/20 对应 \(P_D\)=0.951/0.975/0.984，reports=2.13/0.74/0.22 |

大预测误差和稀疏部署仍是明确边界。固定块长扫描表明当前 `lambda_c` 应解释为
每份远程报告价格，而不是链路相关时延优化。

## 6. 数据索引

- `main/main.csv`：MC=1000 主对比。
- `full-refinement-pd/main.csv`：MC=200 同目标计算对照。
- `prediction-stress/prediction_stress.csv`：预测误差扫描。
- `overview/*/*.csv`：融合、通信、面积、价格、块长和规模扫描。
- `overview/packetization/*.json`：融合集中度和冲突时隙审计。

扩展扫描均标为诊断性证据，不能替代更高 MC、多随机种子或波形级验证。
