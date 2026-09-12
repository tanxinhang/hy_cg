# Gate OTFS Collision Experiments V12

V12 keeps the V11 clean-output / CRN / multiprocessing framework and adds a filter-ablation gate for the closed-loop tracking experiment.

## Main purpose

V12 is not meant to claim that a standard Kalman filter is novel. It is designed to answer three falsification questions:

1. Does a stronger standard CV-KF reduce absolute RMSE, and does the collision-aware assignment gap remain?
2. In the unbiased Gaussian measurement model, is DD-RWIF redundant with standard KF?
3. When OTFS-DD collision creates structured measurement pollution (biased/false DD peaks), does DD-RWIF provide value beyond ordinary KF?

## New options

```bash
--tracker-mode alpha|kf|dd_rwif
--meas-pollution none|biased_peak|clutter_peak
--pollution-pmax 0.35
--pollution-c0 3.0
--pollution-pos-bias-m 80
--pollution-vel-bias-mps 8
--dd-weight-alpha 1.0
--dd-weight-beta 0.15
--dd-weight-gamma-min 0.05
--kf-process-pos-std-m 1.0
--kf-process-vel-std-mps 0.4
```

## Recommended falsification commands

### 1. Standard KF, no structured pollution

```bash
python gate_otfs_collision_experiments_v12.py   --gate tracking   --tracking-mc 20   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10  --eval-kernel-mode dirichlet  --target-pd-fusion or  --tracking-reinit-lost  --tracker-mode kf  --meas-pollution none  --num-workers 10  --fresh-out-dir --out-dir tracking_v12_kf_none
```

### 2. DD-RWIF, no structured pollution

```bash
python gate_otfs_collision_experiments_v12.py  --gate tracking  --tracking-mc 20  --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10  --eval-kernel-mode dirichlet  --target-pd-fusion or --tracking-reinit-lost  --tracker-mode dd_rwif  --meas-pollution none  --num-workers 10  --fresh-out-dir  --out-dir tracking_v12_rwif_none
```

Expected diagnostic: DD-RWIF should be close to KF when measurements are unbiased Gaussian. If it is not, check whether the DD weight is double-counting Ccross.

### 3. Standard KF, biased DD-peak pollution

```bash
python gate_otfs_collision_experiments_v12.py  --gate tracking  --tracking-mc 20   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet  --target-pd-fusion or   --tracking-reinit-lost  --tracker-mode kf   --meas-pollution biased_peak  --pollution-pmax 0.35   --num-workers 10   --fresh-out-dir  --out-dir tracking_v12_kf_polluted
```

### 4. DD-RWIF, biased DD-peak pollution

```bash
python gate_otfs_collision_experiments_v12.py   --gate tracking   --tracking-mc 20   --num-frames 30   --inner-mc 50   --tracking-M 15   --tracking-Q 10   --eval-kernel-mode dirichlet  --target-pd-fusion or  --tracking-reinit-lost   --tracker-mode dd_rwif   --meas-pollution biased_peak  --pollution-pmax 0.35   --num-workers 10  --fresh-out-dir  --out-dir tracking_v12_rwif_polluted
```

Expected diagnostic: DD-RWIF should outperform standard KF mainly when structured DD measurement pollution is enabled. That is the condition under which Ccross has non-redundant information beyond Pd and Gaussian measurement variance.

## Key output columns

The tracking summary includes the previous V11 metrics and new V12 diagnostics:

- `polluted_measurement_rate`
- `pollution_prob_mean`
- `dd_info_weight_mean`

Use these to verify whether DD-RWIF is actually reacting to collision-induced pollution.
