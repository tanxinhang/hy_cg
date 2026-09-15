# 15-UAV/10-Target Worst-Detection Remediation

## Outcome

The worst-target failure can be removed in the screened three-geometry test,
but not by fusion changes alone and not uniformly under the intentionally
gain-free hardware stress profile. The supported remedy combines:

1. a pricing-derived rescue column that is guaranteed to enter the global
   candidate pool;
2. a design target independent of the operational requirement, allowing true
   max--min optimization instead of stopping at the 0.95 threshold;
3. separately generated fixed-mode and active-mode candidate pools;
4. an explicit radar net-gain axis rather than an undisclosed antenna gain;
5. optional long-CPI rescue modes when a zero-gain stress test is required.

## Diagnosed causes

The corrected zero-gain 15-UAV/10-target instance was controlled by targets 4
and 7. Their held-out detection probabilities were 0.208 and 0.296, while most
other targets exceeded 0.77. The lossless-report oracle produced zero
worst-target headroom, so report erasure and fusion placement were not the
limiting mechanisms.

Two algorithmic restrictions then prevented effective rescue. First, the
48-unit per-target energy cap prevented weak targets from borrowing unused
fleet resources. Second, the best bundle returned by information pricing was
not guaranteed to appear in the enumerated column pool when pricing used more
physical candidates than exhaustive mode enumeration. The master also stopped
improving a target after its design probability reached 0.95, leaving no
hold-out margin.

The experimental runner had a separate fairness issue: its fixed baseline was
filtered from the active pool and evaluated under the active 1.25-power
full-load envelope. It now generates the fixed pool independently and evaluates
each method under its own declared full-load envelope.

## Progressive screening results

All rows use 15 UAVs, 10 targets, mean RCS 0.05 m\(^2\), the same declared
geometry family, exact LLR detection, and true report erasures. Values below
are screening results, not publication estimates.

| Configuration | Geometry seeds | Mean worst \(P_D\) | Mean target \(P_D\) | Energy | CPU |
|---|---:|---:|---:|---:|---:|
| Zero gain, 48 target energy, current modes | 1 | 0.2112 | 0.7880 | 400 | 829 |
| Zero gain, adaptive 128 target energy, up to 5 observations | 1 | 0.4617 | 0.8727 | 704 | 1177 |
| Zero gain, 64-look rescue mode | 1 | 0.7705 | 0.9335 | 1096 | 1548 |
| Zero gain, 144-look rescue, design target 0.97 | 3 | 0.4568 | 0.8271 | 3460 | 3940 |
| 15 dB net gain, standard modes, design target 0.97 | 3 | 0.8837 | 0.9807 | 119 | 285 |
| 17.5 dB net gain, standard modes, design target 0.97 | 3 | 0.9749 | 0.9917 | 88 | 200 |
| 17.5 dB net gain, standard modes, pure max--min target 1.0 | 3 | **0.9857** | **0.9981** | 120 | 291 |

At 17.5 dB and pure max--min design, the independently generated fixed nominal
baseline obtained worst/mean \(P_D=0.9944/0.9993\), energy 186.7, and CPU 431.7.
The active method is therefore not a detection-performance winner at this easy
hardware point, but it clears the 0.95 worst-target requirement while using
35.7% less energy and 32.5% less CPU. Its remaining scientific value at this
point is resource-efficient evidence acquisition, not a higher saturated
detection probability.

The zero-gain three-seed experiment is equally important. Its per-seed worst
\(P_D\) values were 0.9703, 0.0873, and 0.3127 even with a 144-look rescue mode.
This rejects the claim that computation or fusion alone can universally rescue
0.05 m\(^2\) targets under unspecified isotropic radar hardware.

## False-alarm and promotion boundary

For the 17.5 dB max--min screen, mean scenario \(P_{FA}=0.04995\), but the
maximum observed scenario value was 0.08081. The design used only 512
calibration and 4,096 held-out samples per scenario, so it is not eligible for
promotion as a strict per-scenario \(P_{FA}\le0.05\) result. A formal run must
increase calibration/evaluation samples, use more geometry seeds, and report a
binomial upper confidence bound or simultaneous false-alarm control.

## Implemented algorithm changes

- `generate_active_columns` now injects the complete information-priced bundle
  as a rescue column even when exhaustive mode enumeration uses a narrower
  physical shortlist.
- Pricing width (`--pricing-candidate-limit`) is decoupled from enumeration
  width (`--candidate-limit`), avoiding exponential column growth.
- `solve_global_active_master` accepts `pd_design_target`; values above the
  operational 0.95 requirement preserve a validation margin, while 1.0
  performs pure worst-target maximization.
- Fixed and active baselines now receive independent candidate generation and
  method-consistent interference envelopes.
- The runner exposes `--radar-net-gain-db`, `--rescue-looks`, and
  `--rescue-power-scale` as explicit experimental axes.

## Recommended canonical next run

Use the 17.5 dB point only after mapping it to a declared antenna/EIRP/system-
loss design. Retain the standard modes, target energy 48, at most three
observations, pricing width 6, enumeration width 3, analytical fusion width 3,
and `pd_design_target=1.0`. Run at least 20 independent geometry seeds with
high-resolution calibration and evaluation. Compare against independently
generated fixed nominal, local-only, and lossless-report controls. If the
active method remains slightly below fixed while using substantially fewer
resources, frame the contribution as a detection-constrained resource saving,
not a detection gain.
