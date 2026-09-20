# Joint TP-UIC–Cooperation V1

## A stable theory, system model, and distributed algorithm

**Status:** design specification; not yet a validated performance claim
**Scope:** distributed multi-UAV OTFS-ISAC confirmation/re-detection
**Canonical version:** V1.0, 2026-09-19

### Paper-ready one-sentence argument

> In distributed multi-UAV OTFS-ISAC, we formulate receiver-side interference
> cancellation and cooperative evidence acquisition as one finite-state potential
> optimization, in which each receiver exports target-conditioned residual and
> survival certificates and the network iteratively adapts illumination, target
> protection, reporting, and fusion decisions; the current evidence supports the
> need for this coupling, while its end-to-end advantage remains to be established
> by the validation protocol specified below.

---

## 0. Terminology ledger

| Canonical term | Definition in this version | Avoid |
|---|---|---|
| TP-UIC | target-preserving uncertainty-aware interference cancellation | treating it as a fixed cancellation depth |
| receiver capability certificate | compact receiver output containing residual interference, target survival, identifiability, uncertainty, and cost | raw DD observation exchange |
| cooperative observation | target-dependent bistatic tuple `(i,j,q)` | calling it only a communication link |
| illumination state | active illuminators, sensing-power modes, waveform/beam modes, and CPI allocation | active report set |
| target-protection state | receiver-specific set and weights of target manifolds protected by TP-UIC | a globally fixed “weakest three” rule |
| reporting leg | directed soft-report link `j -> f_q` | reusing the sensing pair `i -> j` |
| fusion UAV | target-specific fusion UAV `f_q` | centralized fusion centre |
| fair sensing potential | robust soft-min target-detection utility minus resource and switching prices | sum-PD objective |
| target survival | fraction `eta_jq` of target-q evidence surviving receiver cancellation | whole-field echo survival used as a per-target quantity |
| residual interference | receiver-side direct-field residual after cancellation, excluding the ordinary thermal-noise floor | total residual energy with noise mixed in |
| identifiability | whitened escape fraction `rho_jq` of target q from the nuisance subspace | array gain |
| joint state | illumination, protection, observation selection, reporting, fusion, and discrete power/beam modes | a cancellation result frozen before scheduling |
| strict-improvement admission | accept an update only if the robust fair sensing potential rises by at least `epsilon` | unconstrained fixed-point iteration |

The existing project terminology in `advice/terminology-ledger.md` remains
authoritative for all other terms.

---

# Part I. Theory

## 1. Problem statement and boundary

The system contains `M` cooperative UAVs and `Q` targets. UAV `i` may illuminate
target `q`; UAV `j` receives the bistatic echo; and the resulting local evidence
may be transported to a target-specific fusion UAV `f_q`. The receiver must
suppress strong cooperative direct-path interference without erasing weak target
evidence. At the same time, the cooperative schedule determines which direct
paths exist, which target views are valuable, and which targets each receiver
must preserve.

Consequently, cancellation and cooperation are decision-dependent in both
directions:

1. the illumination and reporting state changes the receiver interference;
2. the receiver residual changes the value and feasibility of cooperative observations;
3. the fusion deficit changes which targets deserve receiver protection;
4. the revised protection state changes cancellation depth and target survival.

The design objective is therefore not to maximize local cancellation depth. It
is to maximize robust network-level detection utility under sensing, reporting,
processing, and switching constraints.

### Explicit boundary

V1 addresses confirmation/re-detection under a predicted belief. It does not
claim blind initial acquisition, global optimality, continuous-time control, or
hardware cancellation before ADC saturation. Those are separate extensions.

## 2. Observation model

For receiver `j` in sensing interval `t`, define

\[
\mathbf y_j^{(t)} =
\mathbf X_j(\mathbf a^{(t)},\mathbf p^{(t)},\mathbf b^{(t)})
\mathbf h_j^{(t)}
+\sum_{q=1}^{Q}\mathbf A_{jq}(\widehat{\boldsymbol\xi}_q^{(t)})
\boldsymbol\alpha_{jq}^{(t)}
+\mathbf n_j^{(t)} .
\tag{1}
\]

Here:

- `a_i` is the binary illumination state;
- `p_i` is a discrete sensing-power mode;
- `b_i` is a discrete waveform/beam mode;
- `X_j` is the active direct-interference dictionary;
- `h_j` contains the complex direct-channel coefficients;
- `A_jq` is the believed spatial-delay-Doppler target manifold;
- `xi_hat_q` is the predicted target belief;
- `n_j ~ CN(0,C_n,j)` is receiver noise.

When an array is enabled, each dictionary column is lifted as

\[
\mathbf a_{\rm DD}(\tau,\nu)\otimes
\mathbf a_{\rm arr}(u),
\tag{2}
\]

so angular separation is an identifiability dimension, not an artificial power
gain. Array collection gain and subspace escape must be reported separately.

The truth echo and the belief dictionary must use different state objects and
different gain views:

\[
(\boldsymbol\xi_q,\;G^{\rm true}_{ijq})
\quad\text{versus}\quad
(\widehat{\boldsymbol\xi}_q,\;\widehat G_{ijq}).
\tag{3}
\]

No implementation may reconstruct both from one `base` object.

## 3. Reference-assisted soft TP-UIC

### 3.1 Independent reference observation

The direct channel is estimated from `N_ref` cooperative reference CPIs, not
from the same noise realization that is being cleaned. Let

\[
\bar{\mathbf z}_j = \frac{1}{N_{\rm ref}}
\sum_{c=1}^{N_{\rm ref}}\mathbf z_{j,c}.
\tag{4}
\]

The reference observations carry the same active direct dictionary but
independent receiver noise. Any target leakage into the reference is represented
explicitly rather than silently absorbed into the channel estimate.

### 3.2 Network-weighted soft protection

For target `q`, let `pi_q >= 0` be its network deficit price and let `z_jq` be
the receiver protection decision. Define

\[
\mu_{jq}=z_{jq}\,\mu_0\,\pi_q\,w_{jq},
\tag{5}
\]

where `w_jq` represents the local observation value. Receiver `j` estimates the
direct channel by

\[
\widehat{\mathbf h}_j =
\arg\min_{\mathbf h}
\left\|\bar{\mathbf z}_j-\mathbf X_j\mathbf h\right\|_{\mathbf C_{n,j}^{-1}}^2
+\lambda_h\|\mathbf h\|_2^2
+\sum_q \mu_{jq}
\left\|\mathbf P_{jq}\mathbf X_j\mathbf h\right\|_2^2,
\tag{6}
\]

where `P_jq` projects onto the believed local manifold of target `q`. Equation
(6) is a soft constraint. The existing hard-projection TP-UIC is recovered as
`mu_jq -> infinity`, whereas `mu_jq=0` gives unprotected regularized LS.

This formulation provides a continuous and measurable trade-off between direct
interference removal and target preservation. It also permits the network to
protect a target strongly at the few receivers where that target is essential,
rather than weakly protecting it everywhere.

The cleaned current observation is

\[
\mathbf r_j^{(t)}=\mathbf y_j^{(t)}-\mathbf X_j\widehat{\mathbf h}_j.
\tag{7}
\]

The estimator covariance must follow the actual subtraction operator. In
particular, a joint estimator that subtracts `X_j h_hat` must propagate
`X_j C_h,j X_j^H`, not a projected `M_j X_j C_h,j X_j^H M_j` term.

## 4. Receiver capability certificate

For a declared receiver context and joint state, receiver `j` exports

\[
\mathcal C_j =
\left(
I_j^{\rm res},
\{\eta_{jq}\}_{q=1}^Q,
\{\rho_{jq}\}_{q=1}^Q,
\mathbf C_j^{\rm res},
\sigma_{I,j},
\{\sigma_{\eta,jq}\},
c_j^{\rm ref},c_j^{\rm comp},v_j
\right).
\tag{8}
\]

The fields have the following meanings:

- `I_res_j`: expected residual direct interference under the declared state;
- `eta_jq`: survival of target-q evidence, measured for q itself;
- `rho_jq`: whitened target escape from other targets and residual nuisance;
- `C_res_j`: low-rank-plus-diagonal residual covariance;
- uncertainty fields: calibration uncertainty used by robust scheduling;
- costs: reference CPI and receiver processing costs;
- `v_j`: context/version identifier preventing stale certificates from being reused.

The certificate is a deterministic conditional expectation or a calibrated
posterior summary. A fresh one-noise-draw Monte Carlo value must not be placed
inside the scheduling loop.

For robust scheduling, define

\[
\overline I_j^{\rm res}=\widehat I_j^{\rm res}+z_{1-\delta}\sigma_{I,j},
\qquad
\underline\eta_{jq}=\left[\widehat\eta_{jq}
-z_{1-\delta}\sigma_{\eta,jq}\right]_0^1.
\tag{9}
\]

## 5. Receiver-aware cooperative evidence

For cooperative observation `(i,j,q)`, the conservative sensing quality is

\[
\underline\gamma_{ijq}(\mathbf s)=
\frac{
\underline\eta_{jq}(\mathbf s)\,
S_{ijq}(\mathbf s)
}{
N_{0,j}+\overline I_j^{\rm res}(\mathbf s)
+I_{j}^{\rm self}(\mathbf s)+I_{ijq}^{\rm wf}(\mathbf s)
}.
\tag{10}
\]

The state `s` includes illumination, power/beam modes, protection, selected
observations, reporting, and fusion assignments. Equation (10) replaces the
fixed `kappa_dc` abstraction. The same state must be used in coarse selection,
fine refinement, communication evaluation, and truth-side detection.

The local statistic and fusion rule remain detector-consistent. Let
`D_q(s)` denote the fused target deflection or exact-LLR information and let

\[
\underline P_{D,q}(\mathbf s)=
\mathcal G_q(D_q(\mathbf s),P_{FA}^{\star})
\tag{11}
\]

be the calibrated robust detection probability. The mapping `G_q` must be
validated under the same residual covariance used by TP-UIC.

## 6. Joint optimization

Define the robust fair sensing potential

\[
\Phi(\mathbf s)=
-\tau\log\sum_{q=1}^{Q}
\exp\left(-\underline P_{D,q}(\mathbf s)/\tau\right)
-\lambda_R C_R(\mathbf s)
-\lambda_E C_E(\mathbf s)
-\lambda_C C_C(\mathbf s)
-\lambda_{\Delta}d(\mathbf s,\mathbf s^{-}),
\tag{12}
\]

where the first term is a smooth approximation to the worst-target detection
probability, and the remaining terms price reporting, energy, computation, and
state switching. The previous accepted state is `s^-`.

The feasible set enforces:

1. discrete illumination, beam, waveform, and power modes;
2. per-UAV sensing and communication power limits;
3. receiver and fusion processing capacities;
4. reporting-leg feasibility and report budgets;
5. target-specific fusion assignment;
6. maximum protected targets or protection-weight budget per receiver;
7. reference-CPI budget;
8. minimum calibrated `P_FA` compliance;
9. consistency between the active illumination mask and every selected observation.

V1 deliberately uses discrete operating modes. This makes the joint state space
finite and gives the distributed admission rule a simple stability guarantee.

## 7. Stability proposition

**Proposition 1 — finite termination.** Assume that (i) the feasible joint-state
set is finite; (ii) capability certificates are deterministic for a fixed
context and state; (iii) each accepted update is feasible and increases the
same potential `Phi` by at least `epsilon > 0`; and (iv) ties are resolved by a
fixed lexicographic rule. Then the update sequence terminates after finitely
many accepted steps at an `epsilon`-coordinate-stationary state. A previously
visited state cannot recur.

**Reason.** `Phi` increases strictly after every accepted update, whereas the
feasible state set is finite. Therefore neither a cycle nor an infinite accepted
sequence is possible. The result is a stability guarantee, not a global
optimality guarantee.

When online measurements disagree with the predicted certificate, the tentative
state is rolled back unless its measured lower-confidence potential also
improves. The system returns the best feasible accepted state, never merely the
last state visited.

---

# Part II. Distributed algorithm

## 8. Roles and messages

### Receiver UAV `j`

Receiver `j` maintains its local interference/target dictionaries, solves
soft TP-UIC, and produces capability and marginal-change certificates. It does
not transmit raw DD samples or a full dense covariance.

### Target-specific fusion UAV `f_q`

Fusion UAV `f_q` maintains the evidence deficit of target `q`, computes its
deficit price `pi_q`, and aggregates bids for cooperative observations serving
that target.

### Capacity owners

Each receiver, transmitter, reporting link, and fusion UAV owns its local hard
capacity. Existing distributed bidding/column-generation mechanisms may resolve
contention, provided every accepted bundle is evaluated against the same `Phi`.

### Compact messages

Per candidate change, transmit only:

- target and local resource identifiers;
- robust marginal `Delta Phi`;
- `Delta I_res_j`, `Delta eta_jq`, and `Delta rho_jq`;
- reference, computation, energy, and reporting increments;
- context/version identifier and uncertainty bound.

## 9. Local TP-UIC response algorithm

For a fixed joint state:

1. Build the active direct dictionary from truth-independent cooperative UAV state.
2. Build believed target manifolds from the predicted belief and belief gain view.
3. Receive target deficit prices from the fusion UAVs.
4. Select or weight protected targets under the local protection budget.
5. Estimate the direct channel from independent reference CPIs using (6).
6. Apply the estimate to the current sensing observation using (7).
7. Compute per-target `eta_jq`, `rho_jq`, and the low-rank residual covariance.
8. Calibrate uncertainty and export the certificate in (8).

For scheduling, steps 5–7 use an expected or posterior response, not one random
noise realization. For final detection, they operate on the actual observation.

## 10. Joint TP-UIC–cooperation algorithm

### Initialization

1. Start from the released feasible cooperative schedule or a declared safe baseline.
2. Replace the fixed cancellation constant by a conservative initial capability table.
3. Evaluate `Phi(s_0)` and store `s_best=s_0`.
4. Construct the interference conflict graph and a deterministic coloring.

### One outer round

For round `r=0,...,R_max-1`:

1. **Deficit pricing.** Each fusion UAV computes `pi_q` from the robust target utility.
2. **Receiver response.** Each receiver updates TP-UIC for the current joint state.
3. **Local proposals.** Nodes evaluate one-coordinate or small-bundle changes:
   - activate/deactivate an illuminator;
   - switch one discrete power/beam/waveform mode;
   - add/remove one protected target;
   - add/remove one cooperative observation;
   - change one reporting leg or fusion assignment;
   - allocate/deallocate one reference CPI.
4. **Conflict-free bidding.** Only proposals from the same conflict-graph color are
   compared in parallel. Capacity owners admit a feasible non-conflicting bundle.
5. **Strict-improvement test.** Tentatively apply the bundle, rebuild every affected
   capability certificate, and accept only if
   `Phi(s_trial) >= Phi(s_r) + epsilon`.
6. **Rollback and hysteresis.** Reject a failed bundle, restore `s_r`, and blacklist
   the same local reversal for `H` rounds unless its robust gain exceeds a larger
   release threshold.
7. **Best-state checkpoint.** Update `s_best` whenever the robust potential improves.

### Stopping conditions

Stop when any condition holds:

- no feasible proposal exceeds `epsilon`;
- the round budget is exhausted;
- capability uncertainty prevents safe admission;
- all worst-target deficits meet the declared target.

Always return `s_best`. Report termination cause, accepted/rejected updates,
message count, and potential trajectory.

### Pseudocode

```text
Input: truth-independent network state, predicted target beliefs,
       feasible baseline s0, epsilon, Rmax
Output: best feasible joint state sbest and receiver certificates

s <- s0
(C, Phi) <- EvaluateJointState(s)
sbest <- s
Phibest <- Phi

for r = 0,...,Rmax-1:
    pi <- TargetDeficitPrices(C, s)
    proposals <- parallel LocalMarginalCertificates(s, C, pi)
    bundle <- ConflictFreeCapacityAuction(proposals)

    if bundle is empty:
        break

    strial <- Apply(s, bundle)
    (Ctrial, Phitrial) <- EvaluateAffectedState(strial)

    if feasible(strial) and Phitrial >= Phi + epsilon:
        s <- strial
        C <- Ctrial
        Phi <- Phitrial
        if Phi > Phibest:
            sbest <- s
            Phibest <- Phi
    else:
        RollBack(bundle)
        ApplyHysteresis(bundle)

return sbest
```

## 11. Why this is coupled rather than serial

A serial design computes cancellation once under an all-active or preselected
state and then freezes it. V1 forbids this because each admitted scheduling
change invalidates at least one of `X_j`, the protection state, `I_res_j`,
`eta_jq`, or `rho_jq`. Every accepted cooperative change therefore triggers a
receiver update before the next network decision.

Conversely, TP-UIC does not choose protection from local echo power alone. The
fusion deficit price makes local protection depend on whether the network still
needs evidence for that target. This closes both directions of the loop.

---

# Part III. Implementation contract

## 12. Canonical data structures

### `ReceiverContextV2`

Required fields:

```text
cfg
geom_true_uav             # direct-path truth; never target truth at the scheduler
target_belief
base_true                 # truth-side echo generation/evaluation
base_belief               # receiver/scheduler dictionary and gain view
illumination_mask
sensing_power_mode
beam_waveform_mode
reference_budget
protection_weights[M,Q]
processing_gain
hardware_gain
context_id
```

### `ReceiverCapability`

Required fields:

```text
i_res_mean[M]
i_res_upper[M]
kappa_db[M]
eta_mean[M,Q]
eta_lower[M,Q]
rho[M,Q]
c_res_low_rank            # diagonal floor plus low-rank factors
reference_cost[M]
compute_cost[M]
valid[M]
context_id
```

No scalar whole-field survival may be silently broadcast as a per-target
quantity. If only a whole-field approximation exists, it must be named
`eta_field_proxy` and marked invalid for target-conditioned claims.

## 13. Required code-layer separation

| Layer | Responsibility | Must not do |
|---|---|---|
| observation | generate truth observation and reference observations | reuse belief gain as truth gain |
| receiver | estimate/cancel and export capability | select network reports |
| link model | convert capability into per-link sensing statistics | invent missing receiver fields |
| selector | optimize observations/reporting using current capabilities | reuse stale capabilities after changing illumination |
| coordination | admit state changes and enforce strict potential ascent | accept a fixed-point update without checking utility |
| detector | consume residual covariance and selected evidence | whiten with a covariance from a different cancellation operator |

## 14. Configuration contract

The current `enable`/`mode` dual switch should be replaced by one enumerated mode:

```text
cancellation.receiver_model =
    "fixed"              # frozen kappa, explicit idealized baseline
    "hard_tpuic"         # current hard-projection receiver
    "soft_tpuic"         # equation (6)
    "spatial_soft_tpuic" # equation (2) + equation (6)
```

Invalid combinations must fail during configuration validation. In particular:

- `N_ref >= 1` for executable TP-UIC;
- positive regularization and tangent steps;
- declared array geometry for spatial TP-UIC;
- truth and belief contexts both present in belief-mode experiments;
- target-conditioned retention required whenever measured residual cancellation
  is used outside an explicit denominator-only ablation.

## 15. Compatibility path for the current repository

The implementation can be staged without rewriting the whole simulator.  The
current scope explicitly fixes `n_cpi=1`; CPI scaling and multi-reference-CPI
claims are deferred and are not promotion requirements for this stage:

1. split `build_observation` into truth and belief gain inputs;
2. correct the full-arm covariance operator for subtraction by `X h_hat`;
3. replace `ReceiverMeasurement` by target-conditioned `ReceiverCapability`;
4. implement and validate DD-only soft protection at `n_cpi=1`;
5. extend `compute_link_tables` to consume one complete capability object rather
   than two optional arrays;
6. inject a capability-aware table builder into coarse selection, fine refinement,
   coordination, and truth evaluation;
7. replace the current mask fixed point by strict-improvement admission;
8. add spatial-DD soft TP-UIC after the DD-only path is validated;
9. defer independent multi-CPI reference integration until explicitly reopened.

The frozen baseline remains available as an explicit comparison arm and must not
be silently mixed with an executable receiver inside one trial.

---

# Part IV. Validation and experiments

## 16. Evidence ladder

### E0 — Receiver correctness

Before any network claim:

1. At fixed `n_cpi=1`, empirical coefficient-error covariance must match the
   covariance exported by the implemented subtraction operator.  No CPI-scaling
   claim is made in the current scope.
2. Empirical residual covariance must match the exported low-rank model.
3. H0 statistics must achieve the configured `P_FA` within confidence bounds.
4. `eta_jq=1` for no cancellation and for a direct-only oracle subtraction.
5. Target truth and belief gains must be independently perturbable.
6. The joint cancellation operator and its covariance must use the same map.

### E1 — Local receiver baselines

Compare under identical geometry, reference budget, and hardware:

1. no interference cancellation;
2. fixed 40 dB abstraction;
3. plain LS;
4. hard TP-UIC;
5. soft TP-UIC;
6. spatial soft TP-UIC with 4 and 8 receive elements;
7. perfect direct-channel oracle.

Report cancellation depth, per-target survival, identifiability, calibrated
`P_D/P_FA`, runtime, and reference cost. A cancellation-depth-only table is not
sufficient.

### E2 — Coupling baselines

Use the same physical and resource budgets:

1. fixed receiver + cooperation;
2. TP-UIC measured once, then frozen (serial);
3. cooperation fixed, receiver adapted once;
4. one-round receiver-aware cooperation;
5. full joint TP-UIC–cooperation V1;
6. centralized exhaustive optimum on small instances only;
7. upper-resource reference.

The principal comparison is V1 versus the serial TP-UIC baseline, not versus an
unrealistic no-cancellation straw man.

### E3 — Mechanism ablations

- hard versus soft protection;
- fixed weakest-target protection versus deficit-priced protection;
- 1/4/8 receive elements;
- one versus multiple reference CPIs;
- receiver certificates without/with uncertainty bounds;
- fixed-point coordination versus strict-improvement admission;
- without/with switching penalty and hysteresis;
- whole-field proxy versus genuine per-target survival;
- denominator-only versus complete receiver accounting.

### E4 — Stress and boundary tests

- belief position/velocity error;
- low RCS and target-count scaling;
- angular crowding and ULA ambiguity;
- direct-channel mismatch and oscillator error;
- packet loss and fusion-UAV reassignment;
- limited reference CPI and receiver computation;
- asynchronous or delayed capability messages;
- topology and active-illuminator changes.

## 17. Primary metrics

1. worst-target calibrated `P_D` at declared `P_FA`;
2. minimum detectable RCS under fixed resource constraints;
3. robust evidence retention;
4. per-target `eta_jq` and `rho_jq` distributions;
5. residual-interference distribution, not only its median;
6. reports, slots, energy, reference CPIs, and processing cost;
7. accepted rounds, rollback count, message count, and convergence rate;
8. gap to the small-instance optimum.

## 18. Promotion gates

The joint method may replace the fixed receiver in canonical results only when:

1. receiver covariance and `P_FA` calibration pass E0;
2. no truth information enters scheduler or receiver belief dictionaries;
3. the joint method improves the primary network metric over the serial method
   with a paired confidence interval excluding zero;
4. gains survive at least one belief-error and one low-RCS stress setting;
5. termination, rollback, and message costs are reported;
6. the same capability state is used by coarse selection, refinement, and truth evaluation.

Until these gates pass, the method is an experimental branch and the fixed
40 dB model remains an explicitly idealized reference, not a claimed receiver.

---

# Part V. Current evidence, claims, and missing inputs

## 19. Claim–evidence map

| Claim | Current evidence | Status |
|---|---|---|
| Executable TP-UIC produces substantial direct-interference suppression | existing DD-only experiments report roughly 35–37 dB median depth under their stated settings | supported for the existing simulator, not yet for V1 joint operation |
| Complete receiver accounting matters | existing closed-loop study reports a material difference between denominator-only and denominator-plus-survival accounting | supported directionally; exact P_D values require the audited pipeline fix |
| DD-only identifiability is a major bottleneck | typical-template escape is reported near `4.1e-4` for one receive element | supported in the current abstraction |
| Spatial dimension can remove much of that bottleneck | existing probe reports typical `rho` of 0.742/0.941 for 4/8 elements | supported as an identifiability probe, not yet as end-to-end P_D evidence |
| Joint TP-UIC–cooperation improves network detection | audited 30-trial paired experiment at M=6, Q=3, RCS 0.1: robust 4-element TP-UIC + 3-radiator cap improves actual P_D, truth worst-P_D and truth objective with 95% intervals excluding zero | supported for the stated scenario; external-geometry stress still required |
| Strict-improvement admission prevents scheduling cycles | Proposition 1 under the stated finite-state assumptions | theoretically supported; implementation test required |
| Soft protection is better than hard projection | uniform and target-conditioned weight sweeps show no useful Pareto improvement at fixed n_cpi=1 | rejected for the implemented quadratic penalty; retain as an ablation |

## 20. Assumptions and missing inputs

1. **Target journal and paper length:** not specified; this document uses a
   generic algorithmic-paper structure.
2. **Array geometry:** element count and two-dimensional layout remain to be fixed.
3. **Reference waveform:** its target leakage, orthogonality, and time budget
   require a concrete physical definition.
4. **Deficit-price scale:** `mu_0`, `tau`, and resource prices need calibration or
   a dual-update rule.
5. **Asynchrony:** V1 assumes versioned certificates and conflict-colored updates;
   a formal delayed-message convergence result is outside this version.
6. **Hardware dynamic range:** digital TP-UIC assumes the ADC is not saturated;
   analog/RF cancellation requirements must be specified separately.
7. **Novelty:** no literature-based novelty claim is made here. Related-work and
   citation positioning require a dedicated literature search.

## 21. Stable paper claim

Before experiments, the defensible claim is:

> We formulate target-preserving receiver cancellation and cooperative evidence
> acquisition as a single decision-dependent distributed optimization and give a
> finite-state strict-improvement algorithm that terminates without mask cycles.

After the promotion gates pass, the performance clause may be added with the
measured effect size and conditions. It must not be added in advance.

## 22. Executable V1-B pilot and negative control

The executable paired experiment is
`tools/run_joint_tpuic_coordination.py`.  It closes three previously missing
interfaces: truth and belief use separate target-gain views, TP-UIC exports
genuine receiver--target survival factors, and every accepted illumination-mask
change re-runs the receiver before rebuilding both C2F stages.

The first one-seed V1-A run is intentionally retained as a negative control.
Its belief potential increased monotonically, but the constant-receiver joint
arm reduced the truth-side worst-target predicted detection probability from
0.767 to 0 because the only observation allocated to one target fell outside
the true DD window.  Thus Proposition 1 proves finite algorithmic termination,
not physical robustness under belief error.

V1-B therefore adds two belief-only protections:

1. each DD gain is discounted by a Bonferroni lower bound on window-capture
   probability obtained from the propagated delay/Doppler covariance; and
2. each target reserves at least two observations, with the reserve link chosen
   from different transmitter/receiver nodes when feasible.

On the same failed seed, these protections removed the zero-evidence event.  A
30-seed pilot at `M=6`, `Q=3` then produced the following paired changes for
robust joint minus robust serial:

| Receiver | actual P_D (95% CI) | truth worst P_D (95% CI) | truth objective (95% CI) |
|---|---:|---:|---:|
| ideal fixed 40 dB | +0.0667 [-0.0126, 0.1459] | +0.0144 [-0.0047, 0.0335] | +0.0868 [-0.0176, 0.1911] |
| measured hard TP-UIC | +0.0222 [-0.0314, 0.0759] | +0.0237 [-0.0277, 0.0751] | +0.1512 [-0.0482, 0.3505] |

Median measured TP-UIC depth was about 37 dB, below the idealized 40 dB
reference.  Both robust arms had zero zero-evidence events in 30 trials and
their empirical false-alarm rates remained near the declared 0.05.  The robust
joint update reduced active transmitters by 0.87 (fixed receiver) and 0.50
(TP-UIC) on average, with 95% intervals excluding zero.  However, none of the
three detection improvement intervals excludes zero.

Under robust joint coordination, measured hard TP-UIC was also below the ideal
40 dB reference by 0.0891 in truth worst-target predicted P_D (95% CI
[-0.1585, -0.0197]) and by 0.3433 in truth objective (95% CI
[-0.6015, -0.0851]).  Therefore the current hard TP-UIC is **not** an adequate
drop-in replacement for the ideal receiver.  The repaired loop avoids the
discovered catastrophic failure and saves radiating nodes, but it does **not**
yet establish a statistically significant detection gain.  The hard-TP-UIC
joint arm has not passed the primary promotion gate.  Larger paired Monte
Carlo and soft/spatial TP-UIC remain required before a performance claim is
valid.  Reference-CPI scaling is deliberately outside the current scope.

## 23. Scope decision: staged progress without CPI scaling

The active development order is now:

1. **Receiver correctness:** fix the full-arm covariance mismatch and pin it
   with operator-level and H0 calibration tests.
2. **DD-only soft TP-UIC:** implement equation (6), sweep the protection weight,
   and retain only Pareto points that improve cancellation without unacceptable
   target loss.
3. **Spatial soft TP-UIC:** add 4/8-element array selectivity without crediting
   array collection gain, so identifiability and link-budget gains remain
   separable.
4. **Joint integration:** export the measured capability into robust serial and
   robust joint coordination and repeat paired Monte Carlo.

Every stage uses `n_cpi=1`.  A stage advances only if receiver covariance,
`P_FA`, target survival, and paired network metrics pass their declared gates.

## 24. Audited fixed-CPI candidate

The optimization/audit loop produced one candidate that passes the in-scenario
promotion gate:

- four receive elements with unit-norm steering vectors (no array collection
  gain is credited);
- the original hard TP-UIC receiver;
- belief-aware DD capture discount and two-link diversity protection;
- a hard cap of three simultaneous radiators;
- feasibility projection before strict-improvement updates;
- `n_cpi=1` throughout.

Two apparently positive branches were rejected during the loop.  Quadratic soft
protection gave negligible retention gain until cancellation depth collapsed,
and hard first-order nuisance projection calibrated `P_FA` but projected away
strong-target evidence.  Target-conditioned hard protection also failed to
improve the four-element receiver.  These remain negative controls, not hidden
tuning attempts.

The receiver/full-arm covariance was corrected to propagate coefficient error
through `X h_hat`, matching the actual subtraction, rather than through the
stage-1 `M X` operator.  A second audit found that redundancy completion could
violate the three-radiator cap; a third found that strict admission could retain
an infeasible all-on initial state.  Both were fixed, affected trials were
rerun, and later rows replace their earlier counterparts in the consolidated
artifact.

Across 30 paired trials (`M=6`, `Q=3`, area 600 m, RCS 0.1), robust joint minus
robust serial TP-UIC was:

| Metric | Paired mean | 95% interval |
|---|---:|---:|
| actual `P_D` | +0.1778 | [0.0660, 0.2896] |
| actual `P_FA` | +0.0078 | [-0.0023, 0.0179] |
| truth worst-target predicted `P_D` | +0.1692 | [0.1048, 0.2335] |
| truth objective | +0.7129 | [0.4705, 0.9554] |
| active radiators | -3.0 | [-3.0, -3.0] |

The joint arm delivered actual `P_D=0.7556`, `P_FA=0.0556`, and exactly three
radiators.  The aggregate `P_FA` Wilson interval `[0.0475, 0.0648]` contains the
declared 0.05.  Strict-improvement histories and the hard transmitter cap pass
machine-checked audits.

Against the ideal fixed-40-dB joint arm, measured TP-UIC has indistinguishable
actual `P_D` in this sample, but remains lower in truth worst-target predicted
`P_D` by 0.0885 and in truth objective by 0.3258, with both intervals excluding
zero.  Therefore the candidate may advance to external geometry/low-RCS stress,
but it does not yet replace the ideal receiver in canonical claims.

The frozen candidate was then tested without retuning at area 800 m.  Over 10
paired trials, robust joint minus robust serial TP-UIC gave actual `P_D +0.0667`
(95% interval `[-0.0204, 0.1538]`), truth worst-target predicted `P_D +0.0599`
(`[0.0032, 0.1166]`), and truth objective `+0.3244`
(`[0.1071, 0.5418]`).  Joint `P_FA` was 0.0544.  Thus the continuous physical
metrics survive the geometry stress, while the discrete actual-`P_D` sample is
underpowered and does not yet exclude zero.  Median TP-UIC depth fell to about
33 dB at 800 m, so the stress gain is correctly attributed to coordinated
radiator reduction rather than stronger cancellation.
