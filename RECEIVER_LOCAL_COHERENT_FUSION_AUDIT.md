> **归档提示（2026-09-18）**：本文引用的部分 `results_*` 产物已移入 `_archive/2026-09-18/`；正文中的路径引用已同步更新为归档位置，命令行示例里的 `--out` 目录仍写作历史原名（重跑时依旧输出到该名）。

# Receiver-local LLR and coherent-illumination audit

> This development audit is retained for provenance.  The stable theorem,
> model, oracle, and release-gate boundary is defined in
> `V16_MECHANISM_STABLE_THEORY_MODEL_ALGORITHM.md`.

## Scope

This audit keeps the difficult 15-UAV/10-target operating point fixed at
`target_rcs = 0.05 m^2`, at most five observations per target, 144-look rescue
mode, and **0 dB explicit radar net gain**.  No 17.5 dB link-budget uplift is
used in column generation, held-out evaluation, or the conclusions below.

## Implemented fusion paths

### Receiver-local exact-LLR aggregation

For receiver UAV `j`, the local statistic is

\[
L_{j,q}=\sum_{i:(i,j)\in\mathcal B_q} \ell_{ijq}.
\]

The receiver sends one scalar packet containing `L_{j,q}` to the final fusion
UAV.  Since summation is associative, this is detector-equivalent to sending
each exact LLR separately when delivery is lossless.  Under packet erasure,
all observations in the same receiver packet share one arrival indicator.  The
Monte Carlo detector models this correlated, all-or-nothing evidence loss.

Resource accounting is also hierarchical:

- one remote report per distinct remote receiver rather than per observation;
- one final-fusion input per distinct receiver;
- `n_j - 1` scalar additions are charged back to receiver `j`, so local
  aggregation does not create fictitious compute savings.

### Phase-aware coherent-illumination oracle

The canonical Swerling-II energy model has no stable deterministic target
phase.  Operational raw-IQ coherent fusion therefore cannot be claimed from
the current model.  A separate upper-bound oracle was added instead.  It only
combines transmissions that reach the same receiver with equal look count and
refinement mode.  For branch SINRs `gamma_i`,

\[
\gamma_{\mathrm{coh}}=\sum_i\gamma_i
+e^{-\sigma_\phi^2}\left[
\left(\sum_i\sqrt{\gamma_i}\right)^2-\sum_i\gamma_i
\right].
\]

`sigma_phi = 0` is perfect waveform, clock, delay, Doppler, and phase
alignment.  Large phase uncertainty reduces the oracle to the incoherent SINR
sum.  The oracle is evaluated after selection and never changes which columns
the deployable algorithm selects.  This formula audits the already selected
per-branch power allocations.  V1.6 separately defines the fair fixed-total-
power coherent headroom through the principal eigenvalue of an explicit
phase-coherence matrix.

## Zero-gain results

Three independent geometry seeds were evaluated with 8,192 held-out samples
per target/aspect operating point.

| Method | Worst-target Pd, seed 1 | seed 2 | seed 3 | mean | Remote reports, mean |
|---|---:|---:|---:|---:|---:|
| Direct per-observation LLR | 0.9703 | 0.0873 | 0.3127 | 0.4568 | 15.33 |
| Receiver-local aggregated LLR | 0.9703 | 0.0927 | 0.3127 | 0.4586 | 12.33 |
| Perfect-phase coherent oracle on local selection | 0.9768 | 0.1428 | 0.7313 | 0.6170 | 12.33 |

Receiver-local aggregation reduces remote reports by 19.6% and changes mean
worst-target Pd by only +0.0018.  Its lossless-report headroom is only +0.00024,
confirming that reporting loss is not the main bottleneck in this operating
point.  Even the perfect-phase coherent upper bound reaches only 0.6170 on
average and remains as low as 0.1428 for the hardest seed.  Therefore coherent
processing can help selected geometries, but it cannot by itself repair the
15-UAV/10-target worst-case failure.

The dominant unresolved problem remains insufficient physical evidence in the
weak geometries: bistatic path loss, adverse aspect factors, and shared-power
interference leave some targets below the detector's useful SINR region.  The
next optimization should act on geometry/trajectory, target-adaptive dwell and
power, or admission/coverage guarantees rather than adding more fusion layers.

## Reproducible artifacts

- Direct baseline: `_archive/2026-09-18/results_receiver_fusion_zero_gain_direct/run_43717bb1b6ef`
- Receiver-local result with compute accounting and shared packet erasures:
  `_archive/2026-09-18/results_receiver_fusion_zero_gain_local_final/run_010eb0f24788`
- Detector and transport implementation: `isac_sim/active_information.py` and
  `isac_sim/active_system.py`
- Coherent upper-bound implementation: `isac_sim/coherent_oracle.py`

The intermediate unaccounted local run is diagnostic only and is not a result
of record.
