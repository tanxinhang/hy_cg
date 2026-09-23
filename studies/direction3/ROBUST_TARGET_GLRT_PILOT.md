# Robust target-neighbourhood GLRT pilot

## Detector

For each tested target, the detector evaluates a fixed-rank whitened GLRT on a local delay--Doppler grid and uses

$$
T_q^{\max}=\max_{(\Delta\tau,\Delta\nu)\in\mathcal U_q}T_q(\Delta\tau,\Delta\nu).
$$

The pilot uses a $3\times3$ grid with radius 0.25 bin. Every analytic per-location threshold is assigned $P_{FA}/9$, giving a Bonferroni family-wise bound without an independence assumption. For reported receiver experiments, the preferred threshold is instead calibrated from independent H0 samples of the **maximum statistic itself**.

The implementation has two invariants:

- radius zero is numerically identical to the centre-template GLRT;
- enlarging the search cannot reduce the realized statistic, while its family-wise analytic threshold must increase.

## Paired screen

The front end is the combined uncertainty-weighted Jacobian cancellation plus DD-mismatch covariance. Receiver 0, target 1, direct boost 30 dB and exactly the same physical H0/H1 observations are used for both detectors.

| Arm | Detector | AUC | empirical $P_{FA}$ | empirical $P_D$ |
|---|---|---:|---:|---:|
| TP-UIC combined | centre | 0.520 | 0.250 | 0.250 |
| TP-UIC combined | neighbourhood max | 0.520 | 0.125 | 0.125 |
| perfect-channel | centre | 0.574 | 0.500 | 0.563 |
| perfect-channel | neighbourhood max | **0.625** | 0.125 | 0.313 |

The AUC result is the interpretable part of this small screen. Target-neighbourhood search recovers information under perfect cancellation, showing that target-template mismatch is real. It does not improve the TP-UIC AUC, so the current TP-UIC residual remains the dominant detection bottleneck.

The empirical $P_{FA}/P_D$ columns use only 8 calibration and 16 test samples and must not be read as reliable probabilities. With strict split-conformal calibration and target $P_{FA}=0.05$, 8 calibration H0 samples yield

$$
\left\lceil(n+1)(1-\alpha)\right\rceil=9>n,
$$

so the only finite-sample-valid threshold is $+\infty$ and both empirical $P_{FA}$ and $P_D$ are zero. The code now exposes this behavior explicitly instead of silently reporting an unsupported tail quantile.

## Decision

The robust GLRT implementation and its threshold accounting are complete at pilot level, but detector gain is not established for TP-UIC. A larger paired study is justified only after increasing calibration H0 enough to support the requested false-alarm level. Radius and grid density must be frozen from target-state uncertainty or a training split; they must not be selected on test AUC.

## ACE-style self-normalisation

Because the combined front end has a whole-residual whitened-power ratio near one while the target statistic remains inflated, an ACE-like statistic was also tested:

$$
T_q^{\mathrm{ACE}}
=
\frac{T_q^{\max}}
{\lVert C_r^{-1/2}r\rVert_2^2}.
$$

The ratio statistic deliberately refuses the ordinary chi-square threshold; it requires an explicit held-out empirical/conformal gate. On the same paired screen:

| Arm | neighbourhood GLRT AUC | neighbourhood ACE AUC |
|---|---:|---:|
| TP-UIC combined | **0.520** | 0.512 |
| perfect-channel | 0.625 | 0.625 |

The test-set median whole-residual whitening ratio is about 1.46, whereas the raw target-direction H0 inflation is about 17.6. Hence the remaining mismatch is strongly directional, not a common residual scale error. Dividing by total whitened energy cannot repair it and slightly reduces TP-UIC AUC. ACE normalisation is therefore retained only as a negative ablation.

The next covariance correction, if attempted, must act in a low-dimensional target/direct Jacobian subspace. A scalar covariance multiplier or global self-normalisation cannot address the observed anisotropy.

## Oracle directional-covariance diagnostic

Before fitting a finite-sample hybrid covariance, an intentionally
non-deployable upper-bound diagnostic was added. For estimator arms only it
augments the physical covariance with

$$
C_{\rm oracle}=C_{\rm physics}+r_d r_d^H,\qquad
r_d=(I-F)x_{d,\rm true}.
$$

Here $x_{d,\rm true}$ is the simulator-known direct field. This is explicit
truth leakage: it is not an algorithm proposal and remains disabled by
default. It tests the narrower question of whether estimating one local
residual direction could recover detection information.

On the identical paired screen (receiver 0, target 1, +30 dB direct boost,
combined Jacobian front end and $3\times3$ neighbourhood GLRT):

| Covariance used by TP-UIC | AUC | median target-direction H0 inflation | median whole-residual whitening ratio |
|---|---:|---:|---:|
| physics model | 0.520 | 17.6 | 1.46 |
| physics + oracle direct-residual rank one | **0.598** | **1.61** | **1.00** |
| perfect-channel reference | 0.625 | 1.68 | 1.00 |

The oracle closes about 74% of the AUC gap from 0.520 to 0.625 and removes the
large directional inflation. Therefore the low-dimensional covariance route
passes its mechanism screen: the negative ACE result was not evidence that the
information had been destroyed. The next implementable step is to estimate a
small correction inside a frozen local basis such as
$\operatorname{span}[X,J_I,A_q,J_q]$, using calibration H0 only and shrinkage;
the oracle vector itself must never enter a claimed receiver result.

As above, the empirical PFA/PD values are not interpreted because this screen
has only 8 calibration and 16 test H0 samples.

## Deployable constrained-loading screen

The first truth-free approximation keeps the physical DD-mismatch factor
$B_{\rm DD}$ fixed and estimates only a nonnegative loading,

$$
C_{\rm hybrid}=C_{\rm physics}+(\gamma-1)B_{\rm DD}B_{\rm DD}^H.
$$

Candidate $\gamma$ values are frozen before test evaluation and are judged
from calibration H0 directional whitening only. This is a strongly shrunk
special case of the proposed local covariance and uses no H1 or simulator
truth. The paired pilot gives:

| DD loading $\gamma$ | calibration median H0 inflation | test AUC |
|---:|---:|---:|
| 3 | 6.31 | 0.520 |
| 10 | 6.11 | 0.516 |
| 100 | 5.61 | 0.516 |

Even a 100-fold loading does not approach unit calibration and does not improve
test AUC. Therefore scalar loading of the existing Jacobian covariance is
rejected: the missing covariance is not merely an underestimated variance in
the current physical directions. This does not contradict the oracle result;
it narrows the required estimator to a full low-dimensional coefficient
covariance (including cross-terms/rotations) or a learned residual subspace.
