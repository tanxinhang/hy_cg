# Target-local fusion V1

The current V1 contract is:

```text
prediction-only nearest-target fusion
+ adaptive two-stage C2F
+ detector-PD greedy replay with a per-remote-report price
```

`target-local-v1` inherits `paper-canonical` and changes only `fusion.rule`.
The named detector-PD method switches its own score mode at dispatch, without
reading a reference method's selected-link count. The fusion rule is a
decomposition heuristic, not a jointly optimal assignment.

## Corrected modeling contract

- Scheduling uses predicted geometry and mean RCS; truth is used only for echo
  generation, DD capture, and detector evaluation.
- Every ordered `i != j` pair may provide a bistatic sensing observation.
  Communication adjacency constrains only the remote `j -> f_q` report.
- When `j == f_q`, the statistic is local: `chi=1`, payload `=0`, and reporting
  delay `=0`.
- `selected_observations_mean` counts fused sensing observations.
  `selected_links_mean` is retained for CSV compatibility and counts remote
  reports only.
- The independent Gamma-look detector is identified as a fast-fluctuation,
  Swerling-II-like model.
- The weak-target deficit is defined at `P_D^req=0.95`; `D_min` is legacy-only.
- With fixed blocklength, `lambda_c` is a per-remote-report price rather than a
  link-dependent delay optimizer.

See `V1_AUDIT_REMEDIATION.md` for the issue-by-issue audit.

## Reproducible commands

```powershell
python -m pytest -q
python tools/rerun_target_local_v1.py --mc 1000 --workers 8
python tools/check_target_local_v1.py
python tools/run_target_local_v1_overview.py --mc 100 --workers 8
python tools/audit_target_local_v1_prediction.py --mc 100 --workers 8
python tools/make_target_local_v1_paper_figs.py
```

GitHub Actions runs the unit suite, a target-local smoke experiment, and the
checked-in MC=1000 release gate.

## Verified MC=1000 gate, seed 2026

| Metric | Adaptive detector-PD C2F | Exact-marginal | Sensing-SINR |
|---|---:|---:|---:|
| Mean `P_D` | 0.9762 | 0.9411 | 0.9783 |
| 95% CI | [0.9730, 0.9790] | [0.9363, 0.9455] | [0.9753, 0.9810] |
| Worst-target `P_D` | 0.967 | 0.928 | 0.970 |
| Selected observations | 12.101 | 11.728 | 12.101 |
| Remote reports | 0.751 | 5.131 | 4.595 |
| Payload (kbit) | 0.481 | 3.284 | 2.941 |
| Serial delay (ms) | 0.801 | 5.473 | 4.901 |

The paired gain over exact-marginal greedy is `+0.0351`, 95% CI
`[0.0310, 0.0392]`. Against Sensing-SINR it is `-0.0021`, 95% CI
`[-0.0051, 0.0009]`, so detection performance is statistically tied. At the
same observation count, V1 uses 83.7% less payload than Sensing-SINR.

## Full-refinement detector-PD control, MC=200

Adaptive and full-refinement variants each select 12.055 observations and send
0.730 reports on average. Adaptive C2F evaluates 61.06 fine DD neighborhoods,
versus 853.65 for full refinement, a 92.85% reduction. Their `P_D` values are
0.980 and 0.973; the paired adaptive-minus-full difference is `+0.0070`, 95% CI
`[0.0029, 0.0111]`. Full refinement is therefore a same-objective computational
control, not an oracle.

## Status

The corrected V1 passes its local release gate. It remains bounded by its
prediction-only fusion heuristic, analytical DD proxy, fixed blocklength,
independent reports, and MC=100 diagnostic sweeps. A successor should add
fusion capacity and sensing-acquisition cost before adding broader modules.
