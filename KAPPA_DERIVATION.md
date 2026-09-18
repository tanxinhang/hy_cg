# Deriving the sensing direct-path cancellation depth from a cooperative reference budget

**Status:** derivation + numerical check + literature anchors. Code and model
untouched (the default path stays bit-exact); `interference.direct_cancellation_db`
is unchanged at 40 dB. This document exists so that the constant can be
*quoted with its provenance* instead of being asserted.

**Tool:** `tools/derive_kappa_pilot_budget.py`
**Result files:** `results_kappa_pilot_budget/`, `results_kappa_pilot_budget_papercanon/`

---

## 1. The claim

The sensing SINR in the paper is

    gamma^s_ijq = P^s_i g^s_ijq G^hw_ij G_p eta^DD eta^col / (N_0 + I^s_j)      (1)

and `I^s_j` contains the illuminators' direct paths. Because this is a
*cooperative* ISAC network -- every illuminator's waveform is known to every
receiver -- the direct path is deterministic given the channel and can be
reconstructed and subtracted. The cancellation depth is therefore not a
free parameter: it is set by how well the direct channel is estimated, and
that is set by the reference (pilot) budget.

**One-sentence version for the paper:**

> The direct-path cancellation depth is `Gamma_dp = min(10 log10(N_p gamma_p),
> Gamma_hw)`, where `N_p` is the number of cooperative reference elements at
> per-element SNR `gamma_p` and `Gamma_hw` is the analogue/RF ceiling; the
> study's 40 dB is `N_p gamma_p = 1e4` (e.g. 100 reference elements at 20 dB,
> 2.4% of the 4096-element OTFS frame), and the headline scenario's requirement
> of 52.2 dB is satisfied by a full-frame reference at 16 dB per element --
> both well below the ~60 dB hardware ceiling.

---

## 2. Derivation

### 2.1 Reference-aided estimation limit

The sensing receiver `j` observes, over the reference support,

    y_j[n] = sum_i sqrt(P_i) h_ij s_i[n] + (echo) + w_j[n]      (2)

with `s_i[n]` known. Let `N_p` reference elements carry per-element power
`P_p` against noise power `sigma^2`, i.e. per-element SNR `gamma_p = P_p/sigma^2`.
For orthogonal unit-power reference symbols the least-squares / maximum-
likelihood estimate of the direct coefficient is a sample mean, whose error
variance is

    Var( h_hat_ij - h_ij ) = sigma^2 / (N_p P_p) = 1 / (N_p gamma_p)          (3)

Reconstruction and subtraction therefore leaves a residual direct power equal
to `1/(N_p gamma_p)` of the direct power, and the cancellation depth is

    Gamma_dp^(est) = 10 log10( N_p gamma_p )                                  (4)

Equation (4) is the *estimation-limited* depth. Pilot-aided OFDM channel
estimation with the ML/LS estimator is standard and its error behaviour is
well characterised (Morelli & Mengali 2001); the cooperative ISAC case is
easier than classical passive radar because the waveform need not be extracted
from a reference channel -- in this network the payload itself is known over
the control plane, so the *whole* frame is a valid reference.

### 2.2 Hardware ceiling

Equation (4) does not grow without bound. PA non-linearity, phase noise,
I/Q imbalance, ADC dynamic range and DAC quantisation leave a floor that no
amount of reference removes, because they corrupt the transmitted waveform
itself rather than the estimate of it. Published anchors:

| System | Reported cancellation | Source |
|---|---|---|
| Full-duplex ISAC, digital-domain SIC | up to **60 dB** | Liu et al., JSAC 41(9), 2023 |
| Photonics-assisted SIC front end (measured) | **35.3 dB** @1 GHz, 32.6 dB @2 GHz | Yu et al., Opt. Express 32(23), 2024 |
| Adaptive direct-signal cancellation, real DVB-T data | **10-20 dB net** SNR improvement | Brustad, FFI-Report 2014 |
| In-band full-duplex, analogue + digital | ~60-80 dB (near-field SI) | Bharadia et al., SIGCOMM 2013; Sabharwal et al., JSAC 32(9), 2014 |

The conservative, defensible ceiling for a digital-domain canceller in this
regime is the first row: **`Gamma_hw = 60 dB`**. Combined with (4),

    Gamma_dp = min( 10 log10(N_p gamma_p) , Gamma_hw )                        (5)

### 2.3 Budget table

`Gamma_dp` in dB from (5), ceiling 60 dB:

| `N_p` \ `gamma_p` | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|
| 16   | 22.0 | 27.0 | 32.0 | 37.0 | 42.0 |
| 64   | 28.1 | 33.1 | **38.1** | 43.1 | 48.1 |
| 256  | 34.1 | 39.1 | 44.1 | 49.1 | 54.1 |
| 512  | 37.1 | 42.1 | 47.1 | 52.1 | 57.1 |
| 1024 | 40.1 | 45.1 | 50.1 | 55.1 | 60.0 |
| 4096 | 46.1 | 51.1 | 56.1 | 60.0 | 60.0 |

Two readings of the shipped value: **40 dB <=> `N_p gamma_p = 1e4`**, i.e.
100 elements at 20 dB (2.4% of a 4096-element frame), or 64 elements at 22 dB.
It is the *cheapest* row of this table that still exceeds the legacy geometry's
requirement.

---

## 3. What the concrete scenarios require

The requirement is the depth at which the residual direct path drops level with
the **processed** echo (`N*L` gain included). It is measured from the
**production** link-table builder, not from an analytic replica, using the
identity

    residual/echo = rinr / ( gamma_sense * (1 + rinr) )                       (6)

which follows directly from `gamma_sense = S/(N_0 + I + eps)` and
`rinr = I/(N_0 + eps)`; both fields are produced by `compute_link_tables`.

### 3.1 Legacy paper geometry (4000 m, RCS 50 m^2, preset `paper-canonical`)

| `kappa` (dB) | sense SINR (dB) | residual/echo (dB) |
|---|---|---|
| 0  | -39.83 | +39.82 |
| 30 | -11.33 | +10.19 |
| **40** | **-6.95** | **+2.59** |
| 45 | -6.34 | +0.53 |
| 50 | -6.12 | -0.34 |
| 60 | -6.03 | -0.77 |
| 80 | -6.02 | -0.83 |

**Requirement: `kappa* = 48.03 dB`.**

This *validates the original calibration*: at 40 dB the residual sits only
2.6 dB above the processed echo -- "roughly level", exactly as the config
comment claims. The historical archived sweep
(`archive/legacy_model_results/interference-consistency/`) independently
reports `near_far_db = 41.32 dB` on the same geometry under the `active_set`
interference model; that number and this one differ by 1.5 dB purely because
of the `active_set` vs `orthogonal` reporting-model convention, and the
uncancelled sensing INR agrees to 0.4 dB (+36.09 dB then, +35.66 dB now). The
two records are consistent.

### 3.2 Headline scenario (600 m, RCS 0.1 m^2, preset `small-uav-compact-800m`)

| `kappa` (dB) | sense SINR (dB) | residual/echo (dB) |
|---|---|---|
| 0  | -50.03 | +50.03 |
| 30 | -20.11 | +20.04 |
| **40** | **-10.74** | **+10.19** |
| 50 | -4.35 | +1.41 |
| **52.2** | -- | **0.0** |
| 60 | -2.40 | -3.52 |
| 80 | -2.08 | -4.75 |

**Requirement: `kappa* = 52.25 dB`.** At the shipped 40 dB the residual is
still 10.2 dB above the processed echo -- the scenario is interference-limited
and the default under-cancels. Note the earlier analytic probe
(`tools/probe_kappa_necessity.py`) gave 50.9 dB for the same geometry; the
1.3 dB difference is the DD collision penalty and multi-UAV leakage that the
analytic branch omits and the production table includes. **Quote the
production number.**

### 3.3 The two requirements are both reachable

| Scenario | requirement `kappa*` | cheapest reference budget meeting it |
|---|---|---|
| legacy 4000 m / RCS 50 | 48.0 dB | 512 elements @ 20.9 dB (12.5% of frame) |
| headline 600 m / RCS 0.1 | 52.2 dB | 4096 elements @ 16.1 dB (full frame), or 512 @ 25.2 dB, or 1024 @ 22.1 dB |

Neither requirement approaches the 60 dB hardware ceiling. The difference
between the two (48.0 -> 52.2 dB) is **geometric**: the near-far ratio scales
with target range and RCS via `g^s = lambda^2 sigma / ((4 pi)^3 d_iq^2 d_jq^2)`
against a fixed baseline direct path, which is why the constant cannot be
carried between scenarios.

---

## 4. Consequence for the paper

1. **Move `Gamma_dp` from "asserted constant" to "derived parameter".**
   The supplement table row `Direct-path cancellation in sensing & 40 dB` gains
   a derivation and a budget interpretation. The appendix disclaimer
   ("these coefficients ... are not calibrated measurements",
   `ReproducibilitySupplement.tex:73`) is then true but far less damaging: the
   coefficient is now *computed* from a stated budget, not asserted.

2. **Say what the number is not.** It is a modelling premise about a canceller
   that is assumed to exist, not a contribution. The contributions remain the
   link selection and the coordination mask. Conflating the two is the
   reviewer-facing risk.

3. **Report the sensitivity, not just the point.** Because the headline
   scenario sits at 40 dB while requiring 52 dB, the honest presentation is
   either (a) raise the reference budget and state 52 dB, or (b) keep 40 dB and
   state that the scenario is interference-limited by 10 dB. Either is
   defensible; silently keeping 40 dB and reporting the resulting `P_D` as if
   it were a converged operating point is not.

4. **The relative conclusion survives.** That coordination only helps in the
   interference-limited regime, and that its benefit vanishes once the
   cancellation drives the system to the noise/self-residual floor (measured
   near 60 dB), is a *mechanism* statement that does not depend on the absolute
   value of `Gamma_dp`.

---

## 5. Modelling tensions that must be stated (not hidden)

1. **One scalar on an aggregated field.**
   `model.py` computes `I_sense_field = P_rad @ direct_gain` (a sum over *all*
   illuminators) and then multiplies by a single scalar
   `10^(-kappa/10)` (`model.py:651,654,765`). This assigns the *same*
   cancellation depth to every interferer, whereas physically a node cancels
   its own waveform more deeply (perfectly known) than another node's
   (known but attenuated and separately estimated). The clean upgrade is a
   per-source depth `Gamma_dp[i]`, which is also the only version in which
   cancellation could be written as an algorithmic contribution rather than a
   premise. **Not implemented**; recorded as future work.

2. **Projection-type cancellers also remove part of the target.**
   Subspace-projection and adaptive-filter cancellers remove any target energy
   that lies in the direct-path subspace (Colone et al. 2009; the GSM study
   Demissie et al. makes the same point). Equation (4) ignores this because it
   is a *pilot/estimation* argument, not a projection argument. The effect
   makes the model optimistic by an unquantified amount, and it is not
   representable by a scalar at all -- it needs the delay-Doppler structure.

3. **Residual is treated as power-scaled, not structured.**
   The model multiplies by a power factor and treats the result as an additive
   interference floor. A real canceller leaves a residual that can contain
   deterministic structure (pilot-estimation bias, I/Q image). Treating the
   residual as unstructured is the optimistic choice.

None of these three changes the headline numbers; all three belong in a
limitations paragraph.

---

## 6. Ready-to-paste LaTeX

**Symbol conflict warning:** the paper already uses `\kappa` for the Doppler
grid index (`\kappa = \nu NT`, `SystemModel.tex:78,151`). Use a distinct symbol
for the cancellation depth -- `\Gamma_{\rm dp}` below -- or a reviewer will
read the two as the same quantity.

### 6.1 Model-section equation (to follow `eq:sensing_sinr`)

```latex
The interference term in \eqref{eq:sensing_sinr} is
\begin{equation}
 I_j^{\mathrm s}=\underbrace{10^{-\Gamma_{\mathrm{dp}}/10}
 \sum_{i\ne j}P_i^{\mathrm s}g_{ij}^{\mathrm s}}_{\text{residual direct paths}}
 +\underbrace{\epsilon_{\mathrm{self}}P_j^{\mathrm s}}_{\text{self-residual}},
 \label{eq:sensing_interference}
\end{equation}
where $g_{ij}^{\mathrm s}$ is the illuminator-to-receiver direct-path gain and
$\Gamma_{\mathrm{dp}}$ is the direct-path cancellation depth. In a cooperative
ISAC network every illuminator waveform is known at every receiver, so the
direct path is deterministic given the channel and is removed by
reference-aided estimation; for $N_{\mathrm p}$ reference elements at
per-element SNR $\gamma_{\mathrm p}$ the least-squares residual is
$1/(N_{\mathrm p}\gamma_{\mathrm p})$ of the direct power, giving
\begin{equation}
 \Gamma_{\mathrm{dp}}=\min\!\left(10\log_{10}(N_{\mathrm p}\gamma_{\mathrm p}),
 \Gamma_{\mathrm hw}\right),
 \label{eq:kappa_budget}
\end{equation}
with $\Gamma_{\mathrm hw}$ the analogue/RF ceiling left by power-amplifier
non-linearity, phase noise and converter dynamic range, which additional
reference cannot remove. We use $\Gamma_{\mathrm{dp}}=40$~dB, i.e.
$N_{\mathrm p}\gamma_{\mathrm p}=10^{4}$ (100 reference elements at
$20$~dB, $2.4\%$ of the $4096$-element frame); the depth required to bring the
residual level with the processed echo is $48.0$~dB on the legacy geometry and
$52.2$~dB on the compact scenario of Section~\ref{sec:results}, both below the
$60$~dB digital-domain ceiling reported for full-duplex ISAC \cite{liu2023full}.
Because the near-far ratio is set by the bistatic geometry, $\Gamma_{\mathrm{dp}}$
is reported together with the scenario it was evaluated on.
```

### 6.2 Supplement table row (replace the bare `40 dB`)

```latex
Direct-path cancellation $\Gamma_{\rm dp}$ & $40$~dB $=\min(10\log_{10}N_{\rm p}\gamma_{\rm p},\Gamma_{\rm hw})$, $N_{\rm p}\gamma_{\rm p}=10^{4}$\\
Hardware cancellation ceiling $\Gamma_{\rm hw}$ & $60$~dB (digital-domain, \cite{liu2023full})\\
```

### 6.3 Bibliography entries (all verified)

```bibtex
@ARTICLE{sabharwal2014full,
  author  = {Sabharwal, A. and Schniter, P. and Guo, D. and Bliss, D. W. and
             Rangarajan, S. and Wichman, R.},
  title   = {In-Band Full-Duplex Wireless: Challenges and Opportunities},
  journal = {IEEE J. Sel. Areas Commun.},
  volume  = {32}, number = {9}, pages = {1637--1652}, year = {2014},
  doi     = {10.1109/JSAC.2014.2330194}}

@INPROCEEDINGS{bharadia2013full,
  author    = {Bharadia, D. and McMilin, E. and Katti, S.},
  title     = {Full Duplex Radios},
  booktitle = {Proc. ACM SIGCOMM}, pages = {375--386}, year = {2013},
  doi       = {10.1145/2486001.2486033}}

@ARTICLE{liu2023full,
  author  = {Liu, Z. and Aditya, S. and Li, H. and Clerckx, B.},
  title   = {Joint Transmit and Receive Beamforming Design in Full-Duplex
             Integrated Sensing and Communications},
  journal = {IEEE J. Sel. Areas Commun.},
  volume  = {41}, number = {9}, pages = {2907--2919}, year = {2023},
  doi     = {10.1109/JSAC.2023.3287542}}

@ARTICLE{yu2024photonics,
  author  = {Yu, X. and Ye, J. and Yan, L. and Zhou, T. and Zhong, N. and
             Zhu, Y. and Zou, X. and Pan, W.},
  title   = {Photonics-assisted self-interference cancellation for in-band
             full-duplex integrated sensing and communication transceiver},
  journal = {Opt. Express}, volume = {32}, number = {23},
  pages   = {41708--41725}, year = {2024}, doi = {10.1364/OE.541229}}

@ARTICLE{morelli2001comparison,
  author  = {Morelli, M. and Mengali, U.},
  title   = {A comparison of pilot-aided channel estimation methods for
             OFDM systems},
  journal = {IEEE Trans. Signal Process.},
  volume  = {49}, number = {12}, pages = {3065--3073}, year = {2001},
  doi     = {10.1109/78.969514}}

@ARTICLE{griffiths2005pcl,
  author  = {Griffiths, H. D. and Baker, C. J.},
  title   = {Passive coherent location radar systems. Part 1: Performance
             prediction},
  journal = {IEE Proc. Radar Sonar Navig.},
  volume  = {152}, number = {3}, pages = {153--159}, year = {2005},
  doi     = {10.1049/ip-rsn:20045082}}

@ARTICLE{colone2009multistage,
  author  = {Colone, F. and O'Hagan, D. W. and Lombardo, P. and Baker, C. J.},
  title   = {A multistage processing algorithm for disturbance removal and
             target detection in passive bistatic radar},
  journal = {IEEE Trans. Aerosp. Electron. Syst.}, year = {2009},
  doi     = {10.1109/TAES.2009.5089551}}

@TECHREPORT{brustad2014direct,
  author      = {Brustad, H. K.},
  title       = {Direct signal cancellation in passive bistatic DVB-T based
                 radar},
  institution = {Norwegian Defence Research Establishment (FFI)},
  number      = {FFI-Report 2014}, year = {2014},
  isbn        = {9788246424170}}
```

---

## 7. Reproduce

```bash
# headline scenario (600 m / RCS 0.1): requirement 52.25 dB
python tools/derive_kappa_pilot_budget.py --area 600 --rcs 0.1 --trials 6

# legacy paper geometry (4000 m / RCS 50): requirement 48.03 dB
python tools/derive_kappa_pilot_budget.py --preset paper-canonical \
    --area 4000 --rcs 50 --trials 6 --out results_kappa_pilot_budget_papercanon

# cross-check the analytic SINR branch against the production tables
python tools/probe_kappa_necessity.py --area 600 --rcs 0.1 --trials 6
```

**Note on `--preset legacy`.** Do **not** use it to check `kappa`: the `legacy`
preset pins `interference.coupling="legacy"`, and the cancellation term is only
consulted on the `shared_spectrum` branch, so `kappa` is a silent no-op there
(the sweep returns 0-80 dB identically). Use `paper-canonical` plus `--area` /
`--rcs` overrides instead, which is what the command above does.
