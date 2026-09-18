> **归档提示（2026-09-18）**：本文引用的部分 `results_*` 产物已移入 `_archive/2026-09-18/`；正文中的路径引用已同步更新为归档位置，命令行示例里的 `--out` 目录仍写作历史原名（重跑时依旧输出到该名）。

# Waveform-calibrated successor roadmap

The released `target-local-v1` contract and its evidence under
`_archive/2026-09-18/results_target_local_v1/` are frozen.  Waveform work proceeds through separate
presets, methods, output directories, and manuscript fragments so an
underperforming candidate can be discarded without changing V1.

## One-sentence argument

In target-local cooperative OTFS sensing, we test whether replacing V1's
moment-based detector threshold with a PSF-grounded finite-look mixture
calibration improves false-alarm control, while initially holding selection and
communication resources fixed and limiting the claim to the compact no-CP
waveform model.

## Terminology ledger

| Canonical term | Meaning | Do not use as a synonym |
|---|---|---|
| analytical DD proxy | V1 sinc/Dirichlet gain model | waveform-level receiver |
| waveform-equivalent SINR | PSF projection divided by matched-filter background variance | raw link-budget SINR |
| waveform-calibrated detector | detector using the finite-look post-report mixture threshold | end-to-end waveform simulator |
| waveform-level simulator | frame/channel/receiver chain producing detector inputs | current phase-1 adapter |
| `K_look` | number of incoherently integrated frames | `L`, which remains the delay-grid size |

## Version isolation

- V1 preset: `target-local-v1`.
- Phase-1 preset: `target-local-waveform-v2-phase1`.
- V1 method: `proposed_c2f_adaptive_pd`.
- Phase-1 candidate: `proposed_c2f_adaptive_pd_calibrated`.
- V1 evidence directory: `_archive/2026-09-18/results_target_local_v1/`.
- Phase-1 evidence directory: `results_waveform_v2_phase1/`.
- V1 manuscript remains unchanged; candidate theory lives in
  `Conference-LaTeX-template_10-17-19/WaveformAdaptedModel_V2.tex` and is not
  included by the master document.

## Staged gates

### Phase 1 -- detector calibration

- [x] Gate 0: preserve the V1 preset, default roster, evidence directory, and
      MC=1000 release gate.
- [x] Correct OTFS transform axes and sample interval.
- [x] Validate analytical DD capture against the compact OTFS PSF.
- [x] Add exact singleton Gamma--Gaussian mixture calibration.
- [x] Add deterministic multi-report null-quantile calibration.
- [x] Isolate the candidate from the default V1 method roster.
- [x] Gate 1: pass a 24-case waveform grid spanning three desired-SINR levels,
      report failure, unresolved-target interference, clutter, multipath, and
      synchronization mismatch.  Empirical `P_FA` and empirical-versus-exact
      `P_D` are checked with a Bonferroni simultaneous tolerance over all 48
      comparisons.
- [x] Gate 2a: run MC=100 screening at ideal, representative, and stress points.
- [x] Gate 2b: run MC=200 only for candidates that pass Gate 2a; require exact
      trial-level equality of selected sets, payload, and delay in phase 1, and
      report trial-cluster intervals for `P_D` and `P_FA`.
- [x] Freeze the ideal, representative, and stress parameters before
      confirmatory evaluation.  Gate-3 changes require a new protocol label.
- [x] Gate 3: run 1000 paired trials at the frozen ideal, representative, and
      stress points.  The confirmatory run was completed but did not satisfy
      the promotion criterion.

Promotion criterion: calibration improves false-alarm accuracy without a
material detection, payload, or latency regression.  If it fails, retain V1
and archive the candidate results as a diagnostic.

Gate 1 passed with seed 2026.  Across the `24 x 5000` random-DD grid, the
maximum absolute empirical `P_FA` error was `0.008000`, and the maximum
empirical-versus-exact calibrated `P_D` error was `0.012394`.  These results
validate statistical self-consistency within the compact waveform model; they
do not establish an end-to-end detection gain.

Gate 2a passed with 100 paired trials per scenario and seed 2026.  The fixed
diagnostic impairment tuples `(clutter INR, multipath INR, unresolved-target
INR, delay-bin error, Doppler-bin error)` were:

- ideal: `(0, 0, 0, 0, 0)`;
- representative: `(0.25, 0.10, 0.10, 0.05, -0.05)`;
- stress: `(0.50, 0.25, 0.20, 0.10, -0.10)`.

The calibrated candidate and V1 had identical selected sets, payload, delay,
and deflection summaries in every trial.  Their `(V1 P_D, calibrated P_D)`
were `(0.982, 0.982)`, `(0.969, 0.969)`, and `(0.945, 0.947)` from ideal to
stress.  Calibrated `P_FA` was `0.0506`, `0.0508`, and `0.0507`, respectively.
These are screening results, not confirmatory performance claims.

Gate 2b passed with 200 paired trials per scenario.  The calibrated candidate
again preserved every V1 selected set and resource metric.  Its `(P_D, P_FA,
mean detector time)` was `(0.980, 0.0495, 14.094 ms)` for ideal,
`(0.967, 0.0498, 15.393 ms)` for representative, and
`(0.947, 0.0496, 17.149 ms)` for stress.  The paired calibrated-minus-V1 `P_D`
cluster intervals were `[0,0]`, `[0,0]`, and `[0,0.0025]`; every calibrated
`P_FA` cluster interval contained `0.05`.  Detector-only runtime was
`2.33--2.40x` V1 but remained below the preregistered provisional absolute
screening budget of `250 ms` per trial.  The ratio is reported as a cost, not
as evidence of computational efficiency.

Gate 3 was executed with 1000 paired trials per frozen scenario and seed 2026.
The candidate passed all preregistered absolute checks: its trial-cluster
`P_FA` intervals covered `0.05`, absolute `P_FA` errors were below `0.002`,
paired `P_D` passed the `0.005` noninferiority margin, all selected sets and
resource metrics were identical to V1, and detector time remained below
`250 ms`.  The confirmatory summaries were:

| scenario | V1 `P_D` | calibrated `P_D` | V1 `P_FA` | calibrated `P_FA` | calibrated detector time |
|---|---:|---:|---:|---:|---:|
| ideal | 0.9762 | 0.9764 | 0.0497 | 0.0507 | 14.118 ms |
| representative | 0.9676 | 0.9676 | 0.0496 | 0.0506 | 15.453 ms |
| stress | 0.9502 | 0.9505 | 0.0497 | 0.0506 | 17.252 ms |

Gate 3 nevertheless failed its comparative condition: calibrated `P_FA` had
to be closer to `0.05` than V1 in at least two of three scenarios, whereas it
was closer in zero of three.  Detector time was `2.33--2.39x` V1.  This is a
negative promotion result, not evidence of waveform-calibrated superiority.
V1 therefore remains the released method, Phase 2 is not entered under this
protocol, and the phase-1 implementation and results are retained only as a
reproducible diagnostic.  Any later attempt needs a new protocol label and a
prospectively defined hypothesis; the Gate-3 criterion will not be changed
after observing these results.

### Phase 2 -- calibrated selection replay

- Replace the Gaussian `P_D` approximation inside the greedy marginal with the
  calibrated fused distribution.
- Cache distribution evaluations so the comparison includes runtime cost.
- Compare selected sets against both V1 and phase 1 under identical hard caps.

Promotion criterion: a paired detection/fairness benefit with no violation of
the reporting and runtime budgets.  This phase is not implemented and is
blocked by the failed Phase-1 promotion gate.

### Phase 3 -- physical impairment calibration

- Derive clutter, unresolved-target, multipath, and synchronization terms from
  one waveform observation model instead of unrelated scalar penalties.
- Resolve whether each multipath component is associated target energy or
  background interference.
- Introduce waveform-induced cross-report covariance and sensing/refinement
  costs.

Promotion criterion: end-to-end results remain reproducible and the main claim
survives aware/unaware-scheduler and impairment ablations.

## Reproduction

Generate and check Gate 1:

```powershell
python tools/run_waveform_v2_gate1.py
```

Recheck an existing Gate-1 grid without regenerating it:

```powershell
python tools/check_waveform_v2_gate1.py
```

Generate and check Gate 2a:

```powershell
python tools/run_waveform_v2_gate2a.py --workers 8
```

Recheck existing Gate-2a results:

```powershell
python tools/check_waveform_v2_gate2a.py
```

Generate and check Gate 2b:

```powershell
python tools/run_waveform_v2_gate2b.py --workers 8
```

Recheck existing Gate-2b results:

```powershell
python tools/check_waveform_v2_gate2b.py
```

Generate and check the frozen Gate-3 confirmation:

```powershell
python tools/run_waveform_v2_gate3.py --workers 8
```

Recheck existing Gate-3 results (a nonzero exit is the recorded failed
promotion decision):

```powershell
python tools/check_waveform_v2_gate3.py
```

Smoke comparison with zero impairment magnitudes:

```powershell
python tools/rerun_waveform_v2_phase1.py --mc 20 --no-plots
```

Example diagnostic stress point (not a nominal physical setting):

```powershell
python tools/rerun_waveform_v2_phase1.py --mc 200 --no-plots `
  --clutter-inr 0.5 --multipath-inr 0.25 --unresolved-target-inr 0.2 `
  --sync-delay-bins 0.1 --sync-doppler-bins -0.1
```

Do not copy candidate values into the V1 manuscript or regenerate V1 figures
unless every promotion gate has passed and the manuscript version is explicitly
advanced.
