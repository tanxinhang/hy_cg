# V1.4 Correlation Screening

## Setup

The RCS-robust CPU-dynamic method was evaluated on the same seeded physical trials at total correlation levels 0, 0.3 and 0.6. Total correlation was split structurally as transmitter/receiver/target/DD = 0.3/0.3/0.2/0.2. Each level used MC=20.

The selector and detector both used the same covariance model:

\[
D_q=\delta_q^\mathsf T\Sigma_q^{-1}\delta_q,\qquad
w_q\propto\Sigma_q^{-1}\delta_q.
\]

## Results

| Total correlation | $P_D$ | $P_{FA}$ | mean observations | median $D$ | paired delta vs 0 | paired 95% CI |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 0.685 | 0.0492 | 42.35 | 11.19 | 0 | [0, 0] |
| 0.3 | 0.660 | 0.0486 | 40.55 | 8.30 | -0.025 | [-0.0697, 0.0197] |
| 0.6 | 0.670 | 0.0465 | 39.70 | 7.41 | -0.015 | [-0.0583, 0.0283] |

## Interpretation

Correlation reduces the effective information substantially, as shown by the falling median deflection. However, neither paired $P_D$ interval excludes zero at MC=20. The selector also chooses fewer observations as correlation rises, which is the expected behavior of covariance-aware detector-marginal pricing.

No additional hand-tuned diversity reward is justified by this screening. The next defensible step is a larger frozen correlation experiment and waveform- or measurement-based calibration of the correlation factors.

## Numerical note

Correlation creates near-tied saturated bundle values. HiGHS could classify the third lexicographic stage as infeasible under a $10^{-5}$ lock even though the previous integer incumbent remained feasible. Correlation mode now uses a $10^{-4}$ reliability lock; independent/released paths retain $10^{-5}$. This is a numerical tolerance change only and does not relax resource constraints or reorder objectives.

Source: results_fusion_headroom_v14_corr20/correlation_sensitivity.csv.
