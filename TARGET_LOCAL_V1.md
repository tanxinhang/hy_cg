# Target-local fusion V1

Complete experiment overview: `results_target_local_v1/V1_EXPERIMENT_OVERVIEW.md`.

## Frozen method contract

`target-local-v1` is the reproducible candidate preset for

```text
nearest-target fusion
+ adaptive two-stage C2F
+ detector-aligned greedy replay
```

It inherits `paper-canonical` and changes only `fusion.rule` from
`max_in_rate` to `nearest_target`.  It does not replace `paper-canonical` or
the released paper results.

The fusion planner must receive predicted geometry.  Missing, incorrectly
shaped, or non-finite planning geometry is an error; the implementation must
not silently fall back to another fusion rule.  In belief mode the simulation
constructs the reporting plan from `geom_belief`, `base_belief`, and
`tables_belief` before truth-only detector evaluation.

## Reproducible commands

Fast contract tests:

```powershell
python -m unittest tests.test_target_local_v1 tests.test_canonical_consistency
```

The dedicated runner executes those tests first and then uses the frozen
preset, method roster, paired reference, and seed:

```powershell
python tools/rerun_target_local_v1.py --mc 1000 --workers 4
```

Check the completed main gate against the frozen statistical and resource
contract:

```powershell
python tools/check_target_local_v1.py
```

Smoke run:

```powershell
python run_isac_sim.py --mode main --preset target-local-v1 --mc 20 --seed 2026 `
  --methods proposed_c2f_adaptive_pd exact_marginal_greedy sense_sinr `
  --paired-reference proposed_c2f_adaptive_pd `
  --out results_target_local_v1_smoke
```

Frozen main gate:

```powershell
python run_isac_sim.py --mode main --preset target-local-v1 --mc 1000 --seed 2026 `
  --workers 4 `
  --methods proposed_c2f_adaptive_pd exact_marginal_greedy sense_sinr `
  --paired-reference proposed_c2f_adaptive_pd `
  --out results_target_local_v1
```

Every run writes the full resolved `config.json`.  Promotion decisions must be
made from a fresh gate run and must report confidence intervals, not only the
historical point estimates.

## V1 acceptance gates

1. The contract and canonical-consistency test suites pass.
2. Repeating the same seed produces identical summaries; changing
   `run.workers` does not change numerical output.
3. The resolved V1 configuration differs from `paper-canonical` only at
   `fusion.rule`.
4. The planner uses no current-CPI target position or realized target RCS.
5. The MC=1000 main gate reports detection, worst-target detection, reports,
   bits, serial delay, belief capture, confidence intervals, and runtime.
6. V1 remains a candidate until robustness gates over prediction error,
   communication conditions, and target geometry have been completed.

V2 fusion-node replay, load balancing, parallel MAC, and robust assignment are
outside this frozen V1 contract.

## Incremental hardening plan

- Phase 1 (implemented): freeze the preset and fail-fast planning contract;
  add canonical-difference and deterministic-worker regression tests.
- Phase 2: run the fresh MC=1000 main gate and archive its resolved config and
  confidence intervals under `results_target_local_v1/`.
- Phase 3: add prediction-error, communication-condition, and geometry stress
  gates without changing the frozen main configuration.
- Phase 4: promote V1 only if all gates pass; otherwise retain it as a bounded
  candidate and develop V2 separately.

The first Phase-3 screen is available as:

```powershell
python tools/audit_target_local_v1_prediction.py --mc 200 --workers 4
```

Run the complete V1 overview matrix (fusion rule, communication threshold,
deployment area, delay price, blocklength, and fleet scale):

```powershell
python tools/run_target_local_v1_overview.py --mc 100 --workers 4
```

## Verified main gate

The fresh MC=1000, seed=2026 run using the canonical `nearest_target` spelling
passed `tools/check_target_local_v1.py`:

| Metric | V1 result |
|---|---:|
| `P_D` | 0.9734 |
| `P_D` 95% CI | [0.9701, 0.9764] |
| Worst-target `P_D` | 0.9680 |
| Reports | 11.993 |
| Payload | 7.6755 kbit |
| Serial delay | 12.7925 ms |
| Belief capture | 0.9940 |
| Paired gain over exact-marginal greedy | +0.0067 [0.0041, 0.0093] |
| Paired gain over sensing-SINR | +0.0085 [0.0061, 0.0109] |

For the three methods shared with the earlier MC=1000 candidate screen, every
CSV field was identical after normalizing the legacy `nearest_centroid` alias
to `nearest_target`.

## Prediction-error boundary (screening result)

The Phase-3 MC=100 screen identified the following operating boundary.  These
numbers are a screen, not final paper evidence.

| Error level (position / velocity) | `P_D` | Worst `P_D` | Belief capture | Reports |
|---|---:|---:|---:|---:|
| 0 m / 0 m/s | 0.981 | 0.950 | 1.0000 | 12.09 |
| 50 m / 5 m/s | 0.983 | 0.970 | 1.0000 | 12.08 |
| 150 m / 15 m/s (nominal) | 0.979 | 0.960 | 0.9954 | 12.16 |
| 300 m / 25 m/s | 0.960 | 0.940 | 0.9901 | 12.72 |
| 500 m / 40 m/s | 0.918 | 0.890 | 0.9868 | 13.64 |

Performance begins to degrade materially above the nominal tracker-error
setting even though belief capture remains close to 0.99.  This indicates that
the next improvement should target uncertainty-aware fusion assignment and
belief-geometry link valuation, rather than merely widening the DD search
window.  Such a change belongs to a successor candidate and must not silently
alter the frozen V1 preset.
