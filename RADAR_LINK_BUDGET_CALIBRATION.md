# Radar link-budget calibration status

## Decision

The earlier `target_rcs=42.5--50.0` results cannot be interpreted as a
small-UAV physical RCS result. In the released simulator, RCS is measured in
square metres, but the bistatic sensing equation omitted transmit antenna gain,
receive antenna gain, and system loss. RCS therefore absorbed an unstated
hardware budget.

The repaired desired-echo model is

\[
P_{r,ijq}=P_i^s
\frac{\lambda^2\sigma_q}{(4\pi)^3d_{iq}^2d_{jq}^2}
\underbrace{10^{(G_{tx}+G_{rx}-L_{sys})/10}}_{G_{hw}}
G_p\eta_{DD}\eta_{col}.
\]

Here, \(\sigma_q\) is explicitly in \(m^2\), antenna gains are in dBi, and
system loss is in dB. All three hardware terms default to zero dB, so every
previous configuration remains numerically reproducible.

## Calibration bridge, not a validated platform

The named preset `small-uav-link-budget-bridge` uses

- mean RCS: \(0.1\,m^2=-10\) dBsm;
- transmit gain: 16 dBi;
- receive gain: 16 dBi;
- system loss: 5 dB;
- net desired-path hardware gain: 27 dB.

Thus, \(0.1\times10^{27/10}=50.12\,m^2\): it reproduces the old
`target_rcs=50` echo scale to within 0.0103 dB. This decomposition makes the
assumption visible; it does **not** prove that this antenna, loss, and RCS
combination is achievable on the intended UAV platform.

## What the range calculation now means

There is no hard-coded 5 km sensing radius. Range is produced by geometry and
the fourth-power bistatic loss. The audit tool
`tools/audit_radar_hardware_budget.py` computes two quantities:

1. the net hardware gain required at a stated symmetric bistatic range; and
2. the noise-limited range of the configured hardware budget.

These are optimistic bounds because they exclude DD mismatch, target
collision, aspect loss, and residual shared-spectrum interference. A platform
claim must therefore specify at least RCS distribution, Tx/Rx gain or antenna
aperture, system loss, power/EIRP, CPI/processing gain, residual cancellation,
and required detection probability.

## Consequence for algorithm optimization

The exploratory RCS 42.5 results remain useful only as an algorithmic
transition-point experiment. They are not evidence for useful small-UAV
detection range. MC=1000 and further scene-hardness tuning stay paused until a
physical operating profile is fixed. After that, the optimization order is:

1. freeze the hardware/RCS/range profile;
2. verify single-link and fused detector calibration;
3. locate the communication-limited regime without changing physics;
4. compare the anchored look-ahead method with Nearest + Detector under the
   same local-processing and remote-report budgets;
5. run the preregistered MC=1000 gate only if the pilot is not dominated.

## Candidate point: 4 km and 1 m^2 RCS

For a symmetric bistatic point with \(d_{iq}=d_{jq}=4\) km, RCS
\(\sigma=1\,m^2\), and the provisional 27 dB net hardware budget, the
noise-limited raw SNR is \(-6.61\) dB. The minimum net gains are 23.61 dB for
raw SNR \(-10\) dB and 33.61 dB for raw SNR 0 dB. The corresponding configured
noise-limited reaches are 4.86 km and 2.73 km, respectively.

Under the analytic 16-look independent-LLR abstraction, before DD loss,
residual interference, aspect fluctuation, and cross-observation correlation,
this point gives approximately \(P_D=0.220\) for one observation, 0.341 for
two independent observations, and 0.541 for four. These are link-level ideal
bounds rather than network Monte-Carlo results. They indicate that the point is
potentially useful for studying cooperation: it is detectable but does not
saturate with one local observation.

## Reference small-UAV scenario family

The physical-scale audit now defines three named presets instead of treating
RCS as a free difficulty knob:

| Preset | Horizontal area | UAV height | Target height | Communication range | Mean RCS |
|---|---:|---:|---:|---:|---:|
| `small-uav-compact-800m` | 0.8 x 0.8 km | 200--500 m | 200--500 m | 1.0 km | 0.05 m^2 |
| `small-uav-dense-s1` | 1 x 1 km | 200--500 m | 200--500 m | 1.2 km | 0.1 m^2 |
| `small-uav-nominal-s2` | 2 x 2 km | 300--600 m | 200--800 m | 1.8 km | 0.05 m^2 |
| `small-uav-sparse-s3` | 4 x 4 km | 500--1200 m | 500--1200 m | 2.5 km | 0.02 m^2 |

All three retain 15 sensing UAVs, 10 targets, 5.9 GHz carrier, and 1 W total
transmit power. They deliberately specify **no** antenna gain: the physical
presets are separated from the 27 dB regression bridge. S2 is
the candidate headline physical scale; S1 is a functionality check and S3 is a
failure-boundary stress case. The RCS values are scenario inputs, not parameters
to be selected after observing algorithm performance.

### Superseded S2 bridge pilot

The following MC=100 result used the now-rejected 27 dB regression bridge. It
is retained only for provenance and must not be interpreted as S2 physical
performance. The diagnostic used `C_local=1`, `K_remote=2`, receiver and
fusion-observation capacities of two, target-assignment capacity one, true
report erasure, and a local anchor. The only comparison change was fusion rule.

| Fusion rule | Mean P_D | Predesignated weak-target P_D | Mean reports | Mean observations |
|---|---:|---:|---:|---:|
| capacitated PD look-ahead | 0.984 | 0.98 | 0.10 | 10.10 |
| capacity-constrained nearest target | 0.984 | 0.96 | 0.12 | 10.12 |

Paired point differences (look-ahead minus nearest) were 0.000 in mean P_D,
+0.02 for the weak target, and -0.02 reports. This is not a useful algorithmic
separation: local evidence already solves almost every target, and remote
communication is scarcely used. S2 is physically coherent but, under the
provisional 27 dB hardware budget, is an easy operating point for the current
detector. It must not be made harder by post-hoc RCS tuning; the next declared
test must instead locate a transition through a declared net-gain sensitivity
axis, after which a real antenna platform may be mapped onto that axis.

### Net-gain sensitivity audit

The physical presets now have zero unspecified antenna gain. A separate
`radio.radar_net_gain_db` override is used only as a sensitivity coordinate;
it does not assign artificial Tx and Rx antenna gains.

The S2 coarse pilot (MC=20) gave:

| Net gain | PD-look-ahead P_D / weak | Nearest P_D / weak | Look-ahead / nearest reports |
|---:|---:|---:|---:|
| 0 dB | 0.210 / 0.00 | 0.180 / 0.10 | 2.00 / 2.00 |
| 10 dB | 0.600 / 0.30 | 0.620 / 0.35 | 2.00 / 2.00 |
| 20 dB | 0.955 / 0.95 | 0.975 / 0.95 | 0.75 / 0.85 |
| 30 dB | 0.995 / 1.00 | 0.995 / 1.00 | 0.00 / 0.00 |

This brackets the non-saturated transition between 10 and 20 dB. A refined
MC=30 pilot added a same-rule local-only control:

| Net gain | Local-only P_D / weak | PD-look-ahead K=2 | Nearest K=2 |
|---:|---:|---:|---:|
| 12.5 dB | 0.690 / 0.267 | 0.663 / 0.300 | 0.710 / 0.267 |
| 15.0 dB | 0.790 / 0.467 | 0.817 / 0.500 | 0.827 / 0.433 |
| 17.5 dB | 0.887 / 0.567 | 0.897 / 0.600 | 0.877 / 0.633 |

The 15 dB point is the clearest cooperation transition in this small pilot,
but the proposed placement is not uniformly superior to nearest fusion: at
15 dB it has lower mean P_D and higher weak-target P_D, while at 17.5 dB the
ordering reverses for the weak target. These MC=30 differences are too noisy
for promotion. The next step is a paired, independent-seed replication around
15 dB and an audit of why remote evidence can occasionally reduce realized
detection despite the local anchor.

### Compact 800 m-area pilot

The requested `small-uav-compact-800m` preset uses an 800 x 800 m horizontal
area, 200--500 m UAV/target altitudes, 1 km communication radius, mean RCS
0.05 m^2, and no unspecified radar gain. Over 100 geometry draws, individual
UAV--target slant ranges had median 426 m, 90th percentile 701 m, and maximum
1.071 km; the nearest-UAV range per target had median 140 m. Thus 800 m is the
area side, not a fixed range or a hard link cutoff.

At zero net-gain override, an MC=30 pilot gave:

| Arm | Mean P_D | Weak-target P_D | Reports |
|---|---:|---:|---:|
| local-only | 0.437 | 0.433 | 0.00 |
| PD-look-ahead, K=2 | 0.463 | 0.267 | 2.00 |
| nearest fusion, K=2 | 0.447 | 0.267 | 2.00 |

The mean point estimate shows a small communication gain (+0.026 for
PD-look-ahead versus local-only), but the weak-target ordering is unstable and
MC=30 is insufficient for a claim. The next run must preserve per-trial records
and use paired contrasts before interpreting the apparent weak-target decrease.

### Compact-scenario report-budget frontier

With the compact 800 m preset and zero net-gain override fixed, an MC=20 coarse
frontier varied only the global remote-report cap:

| K_remote | PD-look-ahead P_D / weak / reports | Nearest P_D / weak / reports |
|---:|---:|---:|
| 0 | 0.450 / 0.450 / 0.00 | 0.435 / 0.450 / 0.00 |
| 1 | 0.475 / 0.350 / 1.00 | 0.440 / 0.400 / 1.00 |
| 2 | 0.500 / 0.350 / 2.00 | 0.470 / 0.300 / 2.00 |
| 4 | 0.515 / 0.350 / 3.95 | 0.505 / 0.400 / 3.95 |
| 8 | 0.490 / 0.400 / 6.80 | 0.525 / 0.400 / 6.75 |
| 10 | 0.495 / 0.400 / 7.05 | 0.530 / 0.400 / 6.85 |

For PD-look-ahead, K=2 is the provisional low-overhead knee: it improves mean
P_D by 0.050 over its K=0 control using exactly two reports. K=4 buys only an
additional 0.015 with roughly two more reports; K=8 and K=10 are dominated in
this pilot. No tested budget approaches the declared weak-target requirement
of 0.80. Increasing the global cap alone is therefore not the solution. The
next algorithmic audit should allocate remote evidence by max-min predicted PD
before optimizing total PD, while retaining the same report caps.
