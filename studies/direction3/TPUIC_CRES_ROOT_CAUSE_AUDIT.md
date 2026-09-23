# TP-UIC residual-covariance root-cause audit

## Audit conclusion

The failed covariance-loading screen was based on the wrong hypothesis. The
dominant discrepancy is not a scalar under-estimation of the existing
first-order DD covariance. The canceller and covariance model currently use
incompatible approximation orders:

1. the cancellation dictionary contains the centre and first DD derivatives;
2. the physical direct field is generated at the displaced true DD location;
3. the covariance model propagates only the first DD derivatives and then
   applies the same cancellation projection;
4. that projection removes precisely the first-order directions, while the
   physical residual is dominated by curvature terms left beyond the tangent
   plane.

Consequently, multiplying the first-order covariance by 3, 10 or 100 cannot
repair the omitted second-order residual geometry.

## Where the inconsistency enters

The physical observation is correctly generated from the true direct
dictionary (`build.py`, lines 109--110), whereas the receiver dictionary is
built at perturbed DD coordinates. With first-order interference tangents,
`dictionaries.py` places the delay and Doppler derivatives in the fitted
subspace.

For one path, the true manifold is locally

$$
x(\hat\theta-\delta)h
=x(\hat\theta)h-J\delta h
+\tfrac12 H[\delta,\delta]h+O(\|\delta\|^3).
$$

The TP-UIC fit spans the first two terms. Its structural residual therefore
starts at the Hessian term. However, `_direct_mismatch_factor` constructs only
$J\Sigma_\delta^{1/2}$ and subsequently applies $(I-F)$ (`residual_model.py`,
line 78). Since $J$ is already in the range of $F$, the model suppresses the
same directions it just created and never constructs $H[\delta,\delta]$.

## Numerical evidence

Across five deterministic representative observations, the trace of the
post-cancellation first-order mismatch block divided by the actual structural
direct-residual energy was

$$
0.553,\ 1.376,\ 0.0023,\ 0.124,\ 0.0367.
$$

Thus the error is not a stable missing scale: it varies by almost three orders
of magnitude. The fraction of the actual residual energy lying in the
first-order post-projection span also varies strongly (0.20--0.97). This
explains why a common loading cannot calibrate the statistic and why the
calibration distribution has a long tail.

The loading screen is consistent with this diagnosis: loadings 3, 10 and 100
leave median calibration H0 inflation at 6.31, 6.11 and 5.61, with AUC 0.520,
0.516 and 0.516. Increasing the wrong block changes neither geometry nor
detection ranking.

## Why the oracle result was easy to over-interpret

The oracle term uses the realized vector

$$
r_{d,\mathrm{true}}=(I-F)x_{d,\mathrm{true}}
$$

from each test observation (`residual_model.py`, line 175). It therefore knows
the realized DD errors, path phases and their coherent sum. Adding
$r_dr_d^H$ does not demonstrate that the same direction can be learned from a
small, independent calibration set with changing scenes. It demonstrates only
that a genie that is handed the test residual direction can whiten it. The
0.598 oracle AUC is useful as a diagnostic ceiling, not evidence for the
previous scalar-loading estimator.

## Secondary protocol issue

The benchmark redraws DD estimation error for every `realisation` because that
index enters the observation RNG seed (`run_tpuic_receiver_benchmark.py`,
lines 387--397). Calibration and test scenes therefore represent a marginal
random-error detector, not repeated H0 snapshots under one fixed channel
estimate. That protocol is legitimate for population CFAR, but it cannot be
used to claim that a receiver learned the current realization's residual
subspace from calibration pilots. These are different operational models and
must be separated in subsequent experiments.

## Correct remediation order

1. Freeze the operational conditioning: decide whether DD error is fixed over
   a calibration/data block or redrawn between blocks.
2. Match approximation order: when first-order tangents are cancelled, derive
   the post-cancellation mean and covariance from second-order DD Hessians (or
   use sigma points/unscented propagation through the exact nonlinear
   manifold).
3. Include the nonzero curvature bias. Treating the residual solely as
   zero-mean covariance is generally incorrect because
   $E[\delta^T H\delta]\ne0$.
4. Only after the physics model is order-consistent, estimate a small PSD
   correction from calibration H0 and validate it on held-out blocks.
5. Calibrate the maximum-GLRT threshold on the final statistic with enough H0
   samples; do not use AUC to choose covariance hyperparameters.

No further loading sweep is justified before items 1--3 are resolved.

## Step 1 result: operational conditioning

The benchmark now exposes two explicit protocols:

- `realisation`: redraw direct DD estimation error for every Monte Carlo draw
  (the historical behavior);
- `scene`: hold the estimated direct dictionary fixed within a scene while
  independently redrawing path coefficients and noise.

The implementation uses a dedicated DD-error RNG, so freezing the estimate
does not accidentally freeze the direct phases, target response or noise. A
regression test checks that repeated observations have identical $X$ but
different $h$, $y$ and noise under `scene` scope.

On the same small receiver-0/target-1/+30 dB screen, changing only the scope
changes TP-UIC neighbourhood AUC from 0.520 (`realisation`) to 0.461 (`scene`),
while the perfect-channel reference remains 0.625. Structural cancellation
depth changes from 31.92 dB to 29.20 dB. The samples are too few for a
performance claim, but the material movement confirms that the operational
conditioning is not an innocuous implementation detail.

The `scene` protocol is the appropriate next basis for a receiver that uses
pilots to characterize one channel-estimate block. No Hessian correction has
been introduced in this step.

## Step 2 result: order-consistent nonlinear propagation

A truth-free `sigma_point` covariance model now propagates a $3\times3$
Gaussian cubature rule through the exact direct-path DD manifold and the actual
TP-UIC cancellation map. For each source it factors

$$
E_\delta\!\left[(I-F)x(\hat\theta+\delta)
x(\hat\theta+\delta)^H(I-F)^H\right],
$$

rather than projecting only $J\Sigma_\delta J^H$. Random independent
unit-phase direct coefficients make the unconditional residual mean zero, so
this second moment is the appropriate population covariance for the current
simulation model. The legacy first-order model remains the default.

Under the fixed-`scene` protocol and otherwise identical observations:

| covariance model | calibration median target H0 inflation | test median target H0 inflation | test median whole-residual ratio | AUC |
|---|---:|---:|---:|---:|
| projected first order | 6.18 | 11.76 | 2.09 | 0.461 |
| exact-manifold sigma point | **1.13** | **1.48** | **1.00** | **0.590** |
| perfect channel | -- | -- | -- | 0.625 |

The cancellation output is unchanged (both models have 29.20 dB median
structural depth and 0.988 target survival); only covariance accounting and
whitening change. This is the expected signature of the audited root cause:
order-consistent propagation removes most directional H0 inflation and closes
approximately 78% of the AUC gap from 0.461 to 0.625.

This remains a small mechanism screen (8 calibration and 16 test H0 samples),
so its empirical PFA/PD must not be promoted. The next step is a larger
fixed-scene calibration study with a valid finite-sample threshold, not a new
covariance tuning sweep.

## Step 3 result: first valid split-conformal gate

The receiver/covariance/detector configuration was frozen and the calibration
count was raised to 20. An initial layout with two realizations per scene was
rejected for formal conformal claims because paired scores share geometry and
DD error. The corrected experiment uses 20 independent calibration scenes and
40 independent test scenes, one score per scene.

At target $P_{FA}=0.05$, split conformal selects calibration order statistic
$20/20$. Under exchangeability and a continuous score law, its marginal
future-scene exceedance probability is at most $1/21=0.0476$. Results are:

| arm | threshold | test PFA (95% Wilson) | test PD (95% Wilson) | AUC |
|---|---:|---:|---:|---:|
| TP-UIC + sigma point | 13.118 | 0.000 (0.000--0.088) | 0.100 (0.040--0.231) | 0.575 |
| perfect channel | 12.398 | 0.000 (0.000--0.088) | 0.150 (0.071--0.291) | 0.590 |

TP-UIC median target-direction H0 inflation is 1.26 on calibration and 1.32 on
test; its median whole-residual whitening ratio is 1.00 on both. Thus the
order-consistent covariance transfers to held-out independent scenes without
the severe directional inflation seen previously.

This establishes a finite-sample-valid **single receiver/target marginal**
gate for the frozen pilot. It does not yet establish network-global PFA across
multiple receivers, targets or association choices. Detection remains weak at
this stress point: the conformal gate retains only 10% empirical PD, although
the perfect-channel reference is also only 15%. The next step must address
multiple testing/fusion with calibration performed after the complete
association-and-fusion map.

## Step 4 result: fixed two-UAV post-fusion gate

The first network extension freezes receivers 0 and 1 and a normalized sum
before calibration. With 20 independent calibration scenes, the conformal
threshold of the final fused statistic is 4.754. On 40 independent test
scenes it produces 0 false alarms and 3 detections (AUC 0.745). Receiver 1
alone has AUC 0.789 and 4 detections at its own conformal gate. Thus the
post-fusion marginal gate is wired correctly, but this fixed fusion does not
yet improve tail detection. See `FIXED_FUSION_CONFORMAL_STEP.md` for the
protocol and the three-split requirement before fitting association/fusion.
