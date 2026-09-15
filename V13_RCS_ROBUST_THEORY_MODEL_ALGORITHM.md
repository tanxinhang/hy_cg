# V1.3: RCS-Robust Joint Fusion--Bundle Optimization

## One-sentence argument

In communication- and processing-constrained multi-UAV sensing, we optimize
fusion assignment and observation bundles against a predeclared target-class
RCS interval by exploiting detector monotonicity, which reduces the robust
problem exactly to lower-endpoint bundle pricing while retaining the V1.2
column-generation architecture; empirical promotion remains conditional on
paired Monte Carlo and cross-scenario evidence.

## Terminology ledger

| Canonical term | Definition | Boundary |
|---|---|---|
| nominal RCS `bar{sigma}_q` | Scheduler-side target-class mean RCS | Not current-CPI truth |
| lower RCS `under{sigma}_q` | Predeclared lower endpoint of the uncertainty interval | Not fitted after results |
| RCS-robust bundle | Bundle valued at `under{sigma}_q` | Does not control target RCS |
| scenario audit | Frozen decisions re-evaluated at multiple RCS values | Not a new selection run per scenario |

## 1. Physical and uncertainty model

For observation `(i,j,q)`,

```text
g^s_ijq(sigma_q)
  = lambda^2 sigma_q / [(4 pi)^3 d_iq^2 d_jq^2],

gamma_ijq(sigma_q)
  = C_ijq sigma_q,
```

where `C_ijq >= 0` collects fixed power, hardware gain, processing gain,
delay--Doppler capture, interference, and waveform impairment terms. The
scheduler knows only that

```text
sigma_q in U_q = [under{sigma}_q, bar{sigma}_q],
under{sigma}_q = zeta_q bar{sigma}_q,  0 < zeta_q <= 1.
```

The interval represents slow target-class or model uncertainty. It is not a
second fast-fluctuation draw inside the finite-look LLR; introducing both would
double-count RCS fluctuation in the current detector abstraction.

## 2. Detector monotonicity and robust reduction

For any fixed bundle `b`, the sensing SINRs are componentwise nondecreasing in
`sigma_q`. The local LLR mean gap and deflection are

```text
delta_ijq(gamma) = L gamma^2 / (1 + gamma),
D_ijq(gamma) = L gamma^2,
```

and are nondecreasing for `gamma >= 0`. Packet success depends on the reporting
link, not RCS. Therefore the detector-aligned predicted probability
`p_qfb(sigma_q)` is nondecreasing over the declared interval.

**Proposition 1 (endpoint equivalence).** For a fixed bundle,

```text
min_(sigma_q in U_q) p_qfb(sigma_q)
  = p_qfb(under{sigma}_q).
```

Hence interval robustness does not require scenario enumeration inside the
master or pricing oracle. Every bundle receives the robust coefficient

```text
p^R_qfb = p_qfb(under{sigma}_q).
```

This reduction depends on the declared monotone RCS-to-SINR model. It does not
cover aspect-dependent uncertainty that changes different bistatic paths in
different directions; that case requires a multi-scenario extension.

## 3. Robust lexicographic master

Let `u_qfb` select one target--fusion--observation bundle and let
`d^R_q >= 0` be its lower-endpoint detection deficit. V1.3 solves

```text
lexmin (d^R_max, sum_q d^R_q,
        sum_qfb r_qfb u_qfb, sum_qfb c_qfb u_qfb)
```

subject to the V1.2 assignment and hard resource constraints, plus

```text
d^R_q >= P_D_req - sum_fb p^R_qfb u_qfb,
d^R_max >= d^R_q.
```

The first two tiers protect reliability at the RCS lower endpoint. Remote
reports and total processing are minimized only after both robust reliability
tiers are locked. No scalar robustness penalty is introduced.

For the V1.3 candidate, the exogenous global report-count ceiling is removed
(`K_report = infinity`). Reports are still minimized in the third
lexicographic tier, while receiver, fusion-processing, target-assignment, and
total-observation capacities remain hard physical constraints. This separates
"how many reports reliability needs" from an arbitrarily frozen report budget.

## 4. RCS-robust column generation

1. Freeze `bar{sigma}_q`, `zeta_q`, geometry, hardware, and all resource caps.
2. Construct nominal coarse and refined sensing tables.
3. Multiply each target's sensing SINR by `zeta_q` and recompute the nonlinear
   LLR mean and variance; communication tables remain unchanged.
4. Rank local and remote candidates with lower-endpoint detector utility.
5. Seed the restricted master with empty, singleton-family, and greedy-prefix
   bundles.
6. Solve the worst-deficit LP, price negative-reduced-cost robust bundles, and
   close this tier.
7. Lock the worst-deficit bound, close the total-deficit LP, and solve the
   accumulated four-tier integer master.
8. Evaluate the frozen solution at lower, nominal, and upper RCS scenarios and
   with the common true-erasure Monte Carlo detector.

Because only the sensing tables and bundle utilities change, V1.3 has the same
master dimension and pricing-search order as V1.2. With `zeta_q=1`, it reduces
exactly to nominal joint bundle column generation.

## 5. Experimental design and gates

The initial interval is fixed at `0.025--0.05 m^2`, corresponding to
`zeta=0.5` for the existing compact `0.05 m^2` scenario. This choice reuses the
previously declared sensitivity grid and is not selected from V1.3 outcomes.
The V1.3 screening protocol removes the global report cap; the earlier
`K_report=2` setting is retained only as the V1.2 constrained ablation.

The required comparisons are:

1. RCS-robust bundle CG versus nominal bundle CG under identical resources.
2. Both frozen decisions evaluated at `0.025`, `0.05`, and `0.10 m^2`.
3. Nominal-RCS Monte Carlo comparison against fixed-fusion and local-only
   bundle controls.
4. Degeneracy check `zeta=1`, which must reproduce nominal selection.
5. Exact small-system audit with full candidate enumeration.

Promotion requires correctness, controlled false alarm, non-inferior nominal
performance, and a paired improvement in lower-RCS worst-target detection.
Until those gates pass, V1.3 remains a candidate and does not replace the
current headline method.

## 6. Claim--evidence map

| Claim | Evidence | Status |
|---|---|---|
| RCS enters the optimization through bundle detection coefficients | Explicit RCS-to-SINR-to-LLR chain and code consistency test | Supported |
| Interval worst case occurs at the RCS lower endpoint | Monotone physical/detector model | Supported within stated boundary |
| `zeta=1` reproduces nominal bundle CG | Deterministic regression test | Supported |
| Robust selection reduces lower-endpoint aggregate predicted deficit | MC=20 paired scenario audit: -0.1403, CI `[-0.2205,-0.0601]` | Screening support |
| Robust selection improves realized low-RCS detection | Hidden-low-RCS MC=20: +0.0250, CI `[-0.0149,+0.0649]` | Unresolved |
