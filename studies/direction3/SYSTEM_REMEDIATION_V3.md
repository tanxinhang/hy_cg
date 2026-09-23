# Detection-information chain remediation (V3)

## Scope

This revision addresses the review of commit `1f8e771` without adding another selector or tuning the 16-sample PFA experiment. It first separates two mechanisms that were previously confounded:

1. direct delay--Doppler mismatch entering the cancellation estimator;
2. the same mismatch entering the residual covariance used by the GLRT.

It also removes the pooled H0/H1 covariance from the association/fusion pilot and implements the Schur-complement conditional information gain using a shared H0 covariance.

## 1. Uncertainty-aware direct manifold

The estimator can now use

$$
\widetilde X=[X_0,\sigma_\tau J_\tau,\sigma_\nu J_\nu].
$$

The Jacobian columns are uncertainty-scaled before MAP estimation. The physical direct field and noise remain bitwise paired with the nominal arm. A separate switch controls whether DD mismatch is represented in $C_r$, enabling an identifiable $2\times2$ ablation.

### Paired screen

Configuration: receiver 0, target 1, direct boost 30 dB, DD sigma 0.1 bin, 8 calibration and 16 held-out H0/H1 samples.

| Estimator Jacobian | Mismatch in $C_r$ | structural depth (dB) | median whitened H0 power / dimension | AUC | target survival |
|---|---|---:|---:|---:|---:|
| no | no | 13.9 | 21.12 | 0.512 | 0.999 |
| no | yes | 13.9 | 1.36 | 0.523 | 0.999 |
| yes | no | 31.8 | 1.36 | 0.523 | 0.987 |
| yes | yes | **31.8** | **1.29** | 0.520 | 0.987 |

The direct-manifold diagnosis is therefore only partially confirmed. Jacobian cancellation removes substantially more real direct residual, and the combined arm gives the best whole-residual whitening. However, neither improvement closes the detection gap: AUC remains approximately 0.52 versus 0.574 for perfect-channel in this small screen. The combined arm is not promoted as a detection improvement.

The result also prevents a biased conclusion: cancellation depth and covariance calibration are now reported separately from AUC. Improving the first two is insufficient evidence for the third.

## 2. Scientific gate added

The test suite now checks the simulator rather than only covariance algebra:

- adding the weighted direct Jacobian must not change the physical observation;
- median real structural cancellation depth must improve by at least 10 dB over the nominal dictionary;
- median tested-target survival must remain above 0.8;
- the covariance-only mismatch mechanism must be independently switchable;
- the benchmark records whole-residual whitened power divided by observation dimension.

An absolute whitening gate near one is intentionally not asserted yet: the current screen still reports 1.29, so such a test would correctly fail. It remains a research gate rather than a unit-test invariant.

## 3. H0-only CIG association

Association and fusion now share the same shrunk H0 covariance. The proposed incremental score is

$$
\Delta D_{i|S,q}=
\frac{(\mu_i-r_{iS}R_S^{-1}\mu_S)^2}
{r_{ii}-r_{iS}R_S^{-1}r_{Si}}.
$$

No covariance is pooled with H1. The algorithm evaluates $M+(M-1)+\cdots+(M-K+1)$ candidates and selects by worst-target conditional gain.

On the existing six-calibration/six-test pilot, the corrected CIG method selects $\{0,1,4\}$ and obtains worst AUC 0.472 and mean AUC 0.514, while the truth-assisted common SINR baseline selects $\{0,1,5\}$ and obtains worst AUC 0.472 and mean AUC 0.556. Thus the mathematics is now aligned, but the data remain insufficient and the method is not promoted.

## 4. Work deliberately not claimed complete

- A deployable hybrid physics-plus-low-rank empirical covariance has not been fitted; the current sample count cannot support a defensible correction subspace. An oracle rank-one mechanism diagnostic now justifies pursuing it, but is not itself a receiver.
- A robust target-neighbourhood GLRT is implemented and tested, but has not replaced the centre-only production default. Its maximum-statistic threshold still requires enough independent H0 calibration samples.
- An Oracle subset gap has not been claimed from six held-out samples. Exhaustive enumeration is reserved for a larger diagnostic dataset, not the proposed algorithm.
- Global PFA shrinkage is frozen. Sixteen final H0 samples cannot distinguish useful calibration methods.

## Decision

The review exposed a real estimator defect and a biased association covariance definition; both are corrected. The resulting evidence does **not** yet revive the target-detection contribution. The next admissible experiment is a larger receiver-only paired study of nominal/covariance-only/Jacobian-only/combined arms, followed by robust target-template GLRT only if the combined arm shows stable detector-level information gain. Association and PFA work remain gated.

## 5. Target-neighbourhood GLRT pilot

A fixed-rank $3\times3$ delay--Doppler maximum GLRT has now been implemented with Bonferroni family-wise analytic calibration and independent maximum-statistic empirical/conformal calibration. On the paired combined-front-end screen, TP-UIC AUC stays at 0.520, while perfect-channel AUC increases from 0.574 to 0.625. Thus target mismatch is measurable, but current TP-UIC residuals prevent the recovery from translating to the non-oracle receiver.

The threshold interface now supports strict split conformal calibration. With only 8 calibration H0 samples and $P_{FA}=0.05$, the valid threshold is necessarily infinite; this is reported as an insufficient-sample outcome rather than replaced by an unsupported empirical quantile. See `ROBUST_TARGET_GLRT_PILOT.md`.

An ACE-style normalisation was subsequently tested to distinguish global scale error from directional covariance error. It changes TP-UIC neighbourhood AUC from 0.520 to 0.512 and leaves perfect-channel AUC at 0.625. The target-direction H0 inflation is approximately 17.6 while whole-residual whitening is approximately 1.46, so the unresolved covariance error is anisotropic. Global scale correction is rejected; any hybrid correction must be restricted to the target/direct Jacobian subspace.

## 6. Oracle low-rank mechanism screen

A diagnostic-only covariance term $r_dr_d^H$, with $r_d=(I-F)x_{d,\rm true}$,
was added behind a default-off switch. It deliberately uses simulator truth and
is prohibited from production claims. On the same paired screen it raises
TP-UIC neighbourhood AUC from 0.520 to 0.598 versus the perfect-channel 0.625,
reduces median target-direction H0 inflation from about 17.6 to 1.61, and
reduces the median whole-residual whitening ratio from about 1.46 to 1.00.

This positive oracle gap establishes that the remaining target information is
recoverable by a low-dimensional directional covariance correction. The next
research step is therefore a calibration-H0-only shrinkage estimator in a
frozen local basis; the oracle direction itself is not an admissible input.

A first deployable restricted estimator was also screened by scaling only the
existing physical DD-Jacobian covariance block. Loadings 3, 10 and 100 leave
calibration median target-direction inflation at 6.31, 6.11 and 5.61 and yield
test AUC 0.520, 0.516 and 0.516, respectively. This scalar-loading family is
rejected. A viable estimator must learn coefficient cross-covariance or rotate
within the local basis rather than only increasing the existing eigenvalues.
