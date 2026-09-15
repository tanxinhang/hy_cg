# V1.4 MC=200 Promotion Report

## Frozen protocol

- Preset: small-uav-compact-800m
- Monte Carlo trials: 200
- Workers: 4
- RCS lower factor: 0.5
- Reports: uncapped
- Targets per fusion UAV: uncapped
- Bundle safety limit: 6 observations per target
- Fusion observation-count cap: disabled
- CPU budget: $F_fT_{\rm proc}=2.0\times10^7$ cycles per UAV/CPI
- CPU cost: $10^6+2\times10^6|b|$ cycles for each non-empty bundle
- Threshold calibration: 16384 samples
- Bundle shortlist: 3 local and 3 remote candidates per target-fusion pair

No parameter was changed after the MC=10 screening.

Post-audit verification: after decoupling refined receiver-table construction
from the comparison-method roster, the frozen MC=200 run was repeated. Every
exported scalar field in the main CSV matched the pre-fix result exactly. The
promotion decision and the numerical values below are therefore unchanged.

## Results

| Method | $P_D$ | cluster 95% CI | weak $P_D$ | $P_{FA}$ | mean observations | CPU max utilization |
|---|---:|---:|---:|---:|---:|---:|
| RCS-robust joint bundle CG | 0.6845 | [0.6645, 0.7045] | 0.605 | 0.0513 | 42.91 | 0.9335 |
| Nominal joint bundle CG | 0.6285 | [0.6075, 0.6495] | 0.605 | 0.0524 | 35.72 | 0.8970 |
| Fixed fusion + joint bundle | 0.6010 | [0.5790, 0.6225] | 0.550 | 0.0519 | 26.82 | 0.9690 |
| Local-only joint bundle | 0.5765 | [0.5545, 0.5975] | 0.500 | 0.0510 | 22.03 | 0.8158 |

Paired differences, defined as robust minus baseline:

| Comparison | Difference | paired 95% CI |
|---|---:|---:|
| robust − nominal | +0.0560 | [0.0420, 0.0700] |
| robust − fixed fusion | +0.0835 | [0.0649, 0.1021] |
| robust − local only | +0.1080 | [0.0898, 0.1262] |

The robust detector has $P_{FA}=0.0513$ with trial-cluster interval [0.0495, 0.0532], which contains the design value 0.05.

## Gate decision

- Correctness gate: PASS. The candidate false-alarm interval contains 0.05; all tests pass.
- Overall efficacy gate: PASS. All three paired intervals are strictly above zero.
- CPU-model relevance: PASS. Robust CPU utilization averages 0.9335 and binds in 58.5% of trials.
- Weak-target non-degradation versus nominal: PASS. Both are 0.605.
- Weak-target improvement versus nominal: NOT ESTABLISHED.
- V1.4 candidate promotion: GO.
- Automatic replacement of the existing paper headline: deferred until the manuscript/result identity is migrated explicitly.

## Interpretation

The earlier diagnosis is confirmed: the meaningful gain comes from allocating more observations under a physical CPU budget and evaluating those bundles under low-RCS risk. Fine changes in fusion location alone had much less headroom. The robust method uses more observations and reports, so its gain must be presented together with its CPU and communication cost rather than as a free improvement.

## Artifact

The post-audit source CSV is
`results_fusion_headroom_v14_refine_fix/cpu_dynamic_main.csv`; it is
field-for-field identical to the original
`results_fusion_headroom_v14_mc200/cpu_dynamic_main.csv`.
Its effective run configuration is recorded separately in
`results_fusion_headroom_v14_refine_fix/cpu_dynamic_main.config.json`.
