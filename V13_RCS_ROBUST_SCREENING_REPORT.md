# V1.3 RCS-Robust Bundle Screening Report

## Configuration

- Nominal compact-scenario RCS: `0.05 m^2`.
- Predeclared interval lower endpoint: `0.025 m^2` (`zeta=0.5`).
- Main true-erasure screening: `MC=20`, seed `2026`.
- Cross-scenario audit: 20 shared geometries at RCS factors `0.5, 1, 2`.
- Hidden-low-RCS stress: 20 shared geometries; both schedulers saw the nominal
  `0.05 m^2` table, while the detector used a hidden `0.025 m^2` truth table.
- Resource caps and all non-RCS settings were inherited from the frozen V1.2
  screening configuration.

## Correctness evidence

1. The counterfactual table scales sensing SINR linearly with RCS but
   recomputes the nonlinear LLR moments.
2. It does not mutate nominal sensing or communication tables.
3. `zeta=1` exactly reproduces nominal bundle selection and objective values.
4. On a fully enumerated small candidate system, robust column generation
   matches the full lower-endpoint restricted master.
5. All 61 unit tests pass.

## Nominal-RCS MC=20

| Method | P_D | Weak-target P_D | Active P_FA | Robust-reference paired delta |
|---|---:|---:|---:|---:|
| RCS-robust bundle CG | 0.510 | 0.450 | 0.0548 | 0 |
| Nominal bundle CG | 0.495 | 0.500 | 0.0537 | +0.015 `[-0.0348, +0.0648]` |
| Fixed-fusion bundle | 0.435 | 0.450 | 0.0535 | +0.075 `[+0.0183, +0.1317]` |
| Local-only bundle | 0.490 | 0.450 | 0.0550 | +0.020 `[-0.0445, +0.0845]` |

The robust-reference delta is `P_D(robust) - P_D(method)`. At nominal RCS,
robust selection did not show a significant loss relative to nominal bundle
CG, but MC=20 is insufficient for a non-inferiority claim.

## Frozen-decision RCS scenario audit

The robust-minus-nominal paired differences over 20 shared geometries were:

| RCS (m²) | Difference in worst deficit ↓ | Difference in total deficit ↓ |
|---:|---:|---:|
| 0.025 | approximately 0 | -0.1403 `[-0.2205, -0.0601]` |
| 0.050 | +0.0003 `[-0.0003, +0.0009]` | +0.0458 `[-0.0028, +0.0944]` |
| 0.100 | +0.0007 `[-0.0007, +0.0021]` | +0.0657 `[+0.0089, +0.1226]` |

At the declared lower endpoint, robust pricing significantly reduced aggregate
predicted detection deficit but left the bottleneck target essentially
unchanged. The trade-off appeared at high RCS, where nominal selection had a
smaller aggregate deficit. This is consistent with a conservative scheduler;
it also shows why an upper-RCS result must not be used as the robustness claim.

## Hidden-low-RCS true-detector stress

With both schedulers restricted to the nominal RCS view and the detector using
the hidden `0.025 m^2` truth, the paired robust-minus-nominal detection
difference was

```text
+0.0250, 95% CI [-0.0149, +0.0649].
```

The direction agrees with the lower-endpoint surrogate improvement, but the CI
crosses zero. The current evidence therefore does not establish realized
low-RCS superiority.

## Decision

- Correctness gate: **PASS**.
- RCS integration gate: **PASS**; RCS now changes bundle pricing and selection,
  not merely the plotted operating point.
- Promotion gate: **NO-GO**.
- Headline method and nominal `0.05 m^2` scenario remain unchanged.

The next confirmatory step is a frozen `MC=200` hidden-low-RCS run using exactly
the present interval and algorithm. No RCS endpoint, resource cap, detector
threshold, or geometry parameter may be changed before that run. If the paired
CI still crosses zero, the correct conclusion is that lower-endpoint robustness
improves the scheduling surrogate but not realized detection under this
resource regime.
