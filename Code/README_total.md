下面给出一套**对应 V18 的程序运行指令**。建议按这个顺序跑：先 Gate-1 根基验证，再跑 V18 主实验，再做统计分析，最后做参数扫描。

以下命令默认你把文件放在当前目录：

```text
gate_otfs_collision_experiments_v18.py
analyze_tracking_stats_v18.py
```

---

# 1. Gate-1：验证 DD collision 是否导致检测退化

## 1.1 快速版：Dirichlet 评估核

这个先跑，速度快，用来看趋势。

```bash
python gate_otfs_collision_experiments_v18.py   --gate gate1   --gate1-mc 100   --inner-mc 200   --eval-kernel-mode dirichlet   --proxy-kernel-mode dirichlet   --detector-mode noncoherent_matched   --coop-phase-mode random_phase   --fresh-out-dir   --out-dir gate1_dirichlet_mc100
```

输出重点看：

```text
gate1_summary.csv
gate1_collision_vs_pd.png
gate1_pd_bar.png
```

---

## 1.2 物理验证版：actual OTFS-like 评估核（这个运行过了）

这个更慢，但最重要。用于支撑“DD 碰撞确实会导致 OTFS 链路退化”。

```bash
python gate_otfs_collision_experiments_v18.py   --gate gate1   --gate1-mc 1000   --inner-mc 200   --eval-kernel-mode actual_otfs   --proxy-kernel-mode dirichlet   --detector-mode noncoherent_matched   --coop-phase-mode random_phase   --fresh-out-dir   --out-dir gate1_actual_otfs_mc1000
```

如果时间允许，把 `--gate1-mc 50` 提到：

```bash
--gate1-mc 100
```

---

# 2. V18 主实验：DD-JPDA-shared 后端（这个之前跑过）

这是目前最核心的一组结果。

```bash
python gate_otfs_collision_experiments_v18.py   --gate tracking   --tracking-mc 100   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet   --target-pd-fusion or   --tracking-reinit-lost   --tracker-mode dd_jpda_shared   --meas-pollution biased_peak   --pollution-pmax 0.35   --assignment-hysteresis   --hys-abs-margin 0.05   --kf-innovation-gate   --kf-gate-action skip   --kf-gate-chi2 13.3   --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500   --kf-reject-pos-std-m 1200   --include-global-assignment-baselines   --save-measurement-trace   --save-diagnostics   --num-workers 10   --fresh-out-dir   --out-dir tracking_v18_ddjpda_shared_mc100_main
```

主结果文件：

```text
tracking_summary.csv
tracking_trials.csv
tracking_measurement_trace.csv
tracking_trial_level_metrics.csv
tracking_gospa_time.png
tracking_loss_rate_time.png
tracking_p05_pd_time.png
tracking_switch_rate_time.png
```

---

# 3. 统计分析：CI + Wilcoxon 配对检验

主实验跑完后执行：

```bash
python analyze_tracking_stats_v18.py   --out-dir tracking_v18_ddjpda_shared_mc100_main   --start-frame 10   --baseline global_utility   --methods nearest strongest global_utility collision_aware global_utility_hys collision_aware_hys
```

输出在：

```text
tracking_v18_ddjpda_shared_mc100_main/analysis_stats/
```

重点看：

```text
frame_ge_start_summary.csv
trial_level_frame_ge_start.csv
paired_tests_vs_baseline.csv
gate_diagnostics.csv
main_table.tex
```

---

# 4. 后端对比：DD-RWIF / DD-PDA / DD-JPDA-shared

这组用于回答：

[
\text{普通概率关联能否替代 DD collision-aware assignment？}
]

## 4.1 DD-RWIF 后端

```bash
python gate_otfs_collision_experiments_v18.py   --gate tracking   --tracking-mc 100   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet   --target-pd-fusion or   --tracking-reinit-lost   --tracker-mode dd_rwif   --meas-pollution biased_peak   --pollution-pmax 0.35   --assignment-hysteresis   --hys-abs-margin 0.05   --kf-innovation-gate   --kf-gate-action skip   --kf-gate-chi2 13.3   --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500   --kf-reject-pos-std-m 1200   --include-global-assignment-baselines   --save-diagnostics   --num-workers 10   --fresh-out-dir   --out-dir tracking_v18_ddrwif_mc100
```

## 4.2 DD-PDA-like 后端

```bash
python gate_otfs_collision_experiments_v18.py   --gate tracking   --tracking-mc 100   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet   --target-pd-fusion or   --tracking-reinit-lost   --tracker-mode dd_pda   --meas-pollution biased_peak   --pollution-pmax 0.35   --assignment-hysteresis   --hys-abs-margin 0.05   --kf-innovation-gate   --kf-gate-action skip   --kf-gate-chi2 13.3   --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500   --kf-reject-pos-std-m 1200   --include-global-assignment-baselines   --save-diagnostics   --num-workers 10   --fresh-out-dir   --out-dir tracking_v18_ddpda_mc100
```

## 4.3 DD-JPDA-shared 后端

这就是主实验命令，也可以单独保留成：

```bash
python gate_otfs_collision_experiments_v18.py   --gate tracking   --tracking-mc 100   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet   --target-pd-fusion or   --tracking-reinit-lost   --tracker-mode dd_jpda_shared   --meas-pollution biased_peak   --pollution-pmax 0.35   --assignment-hysteresis   --hys-abs-margin 0.05   --kf-innovation-gate   --kf-gate-action skip   --kf-gate-chi2 13.3   --kf-abs-pos-gate-m 500   --kf-gate-pos-std-cap-m 500   --kf-reject-pos-std-m 1200   --include-global-assignment-baselines   --save-diagnostics   --num-workers 10   --fresh-out-dir   --out-dir tracking_v18_ddjpda_mc100
```

---

# 5. pollution_pmax 扫描

这个最重要，用来证明优势不是单一工作点偶然出现。

建议扫：

```text
0.10, 0.20, 0.35, 0.50
```

## PowerShell 批量运行

```powershell
foreach ($p in 0.10,0.20,0.35,0.50) {
  python gate_otfs_collision_experiments_v18.py     --gate tracking     --tracking-mc 100     --num-frames 30     --inner-mc 50     --tracking-M 15   --tracking-Q 10     --eval-kernel-mode dirichlet     --target-pd-fusion or     --tracking-reinit-lost     --tracker-mode dd_jpda_shared     --meas-pollution biased_peak     --pollution-pmax $p     --assignment-hysteresis     --hys-abs-margin 0.05     --kf-innovation-gate     --kf-gate-action skip     --kf-gate-chi2 13.3     --kf-abs-pos-gate-m 500     --kf-gate-pos-std-cap-m 500     --kf-reject-pos-std-m 1200     --include-global-assignment-baselines     --save-diagnostics     --num-workers 10     --fresh-out-dir     --out-dir "sweep_pmax_$p"
}
```

每个目录跑完后，对每个目录做统计：

```powershell
foreach ($p in 0.10,0.20,0.35,0.50) {
  python analyze_tracking_stats_v18.py `
    --out-dir "sweep_pmax_$p" `
    --start-frame 10 `
    --baseline global_utility `
    --methods nearest strongest global_utility collision_aware global_utility_hys collision_aware_hys
}
```

---

# 6. 目标密度扫描：tracking-Q

这个用于证明“目标越密，DD collision-aware 越有价值”。

建议扫：

```text
Q = 6, 8, 10, 12
```

PowerShell：

```powershell
foreach ($q in 6,8,10,12) {
  python gate_otfs_collision_experiments_v18.py `
    --gate tracking `
    --tracking-mc 100 `
    --num-frames 30 `
    --inner-mc 50 `
    --tracking-M 15 `
    --tracking-Q $q `
    --eval-kernel-mode dirichlet `
    --target-pd-fusion or `
    --tracking-reinit-lost `
    --tracker-mode dd_jpda_shared `
    --meas-pollution biased_peak `
    --pollution-pmax 0.35 `
    --assignment-hysteresis `
    --hys-abs-margin 0.05 `
    --kf-innovation-gate `
    --kf-gate-action skip `
    --kf-gate-chi2 13.3 `
    --kf-abs-pos-gate-m 500 `
    --kf-gate-pos-std-cap-m 500 `
    --kf-reject-pos-std-m 1200 `
    --include-global-assignment-baselines `
    --save-diagnostics `
    --num-workers 10 `
    --fresh-out-dir `
    --out-dir "sweep_Q_$q"
}
```

统计：

```powershell
foreach ($q in 6,8,10,12) {
  python analyze_tracking_stats_v18.py `
    --out-dir "sweep_Q_$q" `
    --start-frame 10 `
    --baseline global_utility `
    --methods nearest strongest global_utility collision_aware global_utility_hys collision_aware_hys
}
```

---

# 7. hysteresis margin 扫描

用于证明 hysteresis 是一个可调的稳定性—性能折中。

建议扫：

```text
hys_abs_margin = 0, 0.02, 0.05, 0.10, 0.20
```

PowerShell：

```powershell
foreach ($h in 0,0.02,0.05,0.10,0.20) {
  python gate_otfs_collision_experiments_v18.py `
    --gate tracking `
    --tracking-mc 100 `
    --num-frames 30 `
    --inner-mc 50 `
    --tracking-M 15 `
    --tracking-Q 10 `
    --eval-kernel-mode dirichlet `
    --target-pd-fusion or `
    --tracking-reinit-lost `
    --tracker-mode dd_jpda_shared `
    --meas-pollution biased_peak `
    --pollution-pmax 0.35 `
    --assignment-hysteresis `
    --hys-abs-margin $h `
    --kf-innovation-gate `
    --kf-gate-action skip `
    --kf-gate-chi2 13.3 `
    --kf-abs-pos-gate-m 500 `
    --kf-gate-pos-std-cap-m 500 `
    --kf-reject-pos-std-m 1200 `
    --include-global-assignment-baselines `
    --save-diagnostics `
    --num-workers 10 `
    --fresh-out-dir `
    --out-dir "sweep_hys_$h"
}
```

重点看：

```text
switch_rate
gospa_pos_m
track_loss_rate
p05_pd_target
```

---

# 8. 最终建议运行顺序

建议你实际按这个顺序跑：

```text
1. gate1_dirichlet_mc100
2. gate1_actual_otfs_mc50
3. tracking_v18_ddjpda_shared_mc100_main
4. analyze_tracking_stats_v18.py 统计主实验
5. sweep_pmax
6. sweep_Q
7. sweep_hys
8. backend 对比：dd_rwif / dd_pda / dd_jpda_shared
```

如果算力有限，最低成本版本是：

```text
1. Gate-1 actual_otfs MC=50
2. 主实验 dd_jpda_shared MC=100
3. pollution_pmax 扫描，每点 MC=50
4. 统计分析
```

这四步就足够支撑一篇会议/硕士论文主体。
