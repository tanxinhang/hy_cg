# TP-UIC V2 refactor proposal audit (2026-10-02)

## Verdict

The proposal is directionally sound as an architecture cleanup, but its main
algorithmic claim is not yet strong enough to replace the current candidate.
Stage-1 demotion, explicit receiver modes, one residual contract, frozen GLRT,
and explicit experiment profiles are appropriate.  Target-risk gating is a
credible safety mechanism, not a demonstrated core detection contribution.

## Claim-by-claim audit

| Proposal | Audit | Action |
|---|---|---|
| Demote Stage 1 to an ablation | Correct: default `protected_only` full TP-UIC does not consume the Stage-1 residual | Change narrative/profile later; no numerical change |
| Per-target TP-UIC | Mechanistically valid and already partly implemented as `targeted_tpuic_full` | Wired into residual covariance and GLRT for fair testing |
| Gate protection using predicted worst-case survival | Receiver-known and strongly predictive of truth survival | Retain as an optional safety rule; do not promote as the core method |
| One `ResidualContract` | Architecturally necessary before production promotion | Design after choosing self-fit versus reference-aided semantics |
| Separate self-fit and reference-aided modes | Scientifically correct | Reference mode requires actual independent reference observations; `n_cpi` cannot merely be reinterpreted |
| Centre sigma-point residuals | Not generally correct here: the random direct coefficient has zero unconditional mean | Do not apply this formula blindly |
| Avoid nominal/sigma-point double counting | Formally valid concern | Tested a physical replacement arm; negligible numerical impact at the current operating point |
| Adaptive Taylor order and structured priors | Plausible but premature | Blocked until fixed nonlinear covariance passes multi-point validation |
| Optimize lower-tail detection information | Better aligned than cancellation depth or survival alone | May be used on train only; held-out AUC/PD remains the promotion criterion |
| Freeze the 3x3 GLRT | Correct | Retain |
| Explicit legacy/candidate/production profiles | Correct engineering control | Recommended independent cleanup task |
| Promote `(M,Q)` evidence downstream | Semantically correct, but premature | Blocked until receiver contribution passes |
| Shrink headline arm roster | Correct for reporting, not for regression coverage | Keep internal arms and publish only the frozen comparison set |

## Test 1: target-risk gate necessary conditions

Protocol: six independent scenes, six receivers, three targets, +30 dB direct
boost, 0.1-bin DD error, and one shared joint H0/H1 observation per
scene/receiver.  This gives 108 receiver-target diagnostics but only six
independent scenes, so it is a mechanism screen rather than a PFA/PD study.

Preregistered necessary conditions used `eta_gate=0.8`:

- predicted-risk versus truth-survival rank correlation at least 0.5;
- at least 20% of receiver-target cases classified as risky;
- in the risky subset, median targeted-minus-plain detection information
  positive and its lower quartile nonnegative.

Results:

| Quantity | Result |
|---|---:|
| Risk/truth rank correlation | **0.938** |
| Risky cases | 17 / 108 = **15.7%** |
| Risky median delta detection information | **+0.00561** |
| Risky lower-quartile delta detection information | **+0.00369** |
| Risky median relative information gain | about **1.0%** |
| Risky truth survival, plain -> targeted | **0.808 -> 0.900** |
| Risky exploratory AUC, plain -> targeted | **0.6125 -> 0.6159** |
| Risky median cancellation depth, plain -> targeted | **29.35 -> 29.23 dB** |

The predictor works, but risky cases miss the preregistered prevalence gate and
the large survival improvement yields only a small detection-information/AUC
movement.  This repeats an earlier lesson: survival is a useful safety
constraint but is not the detection objective.  Decision: **do not promote the
risk gate as the core contribution**.  It remains a defensible conditional
safety mechanism if later system requirements demand it.

## Test 2: nonlinear covariance accounting semantics

The historical additive model contains both the nominal post-cancellation
direct prior block and the full sigma-point direct second moment.  A new
diagnostic arm makes the full sigma-point second moment replace the nominal
block.  Both arms use identical 8 calibration and 16 test scenes, receiver 0,
target 1, +30 dB direct boost, scene-fixed DD error and the frozen 3x3 GLRT.

| Metric | Additive | Replacement |
|---|---:|---:|
| Test AUC | 0.523 | 0.523 |
| Calibration median H0 inflation | 1.443171381 | 1.443171384 |
| Test median H0 inflation | 1.309391171 | 1.309391162 |
| Test whole-residual whitening ratio | 1.000660772 | 1.000660772 |
| Maximum paired H0 score change | \- | `6.1e-7` |

The semantic concern is real, but the duplicated nominal block is numerically
negligible because the fitted nominal/tangent space is already strongly
suppressed by the cancellation transfer.  The replacement interpretation is
cleaner, but it cannot explain or improve current detection performance.

## Recommended architecture decision

Keep the current fixed nonlinear sigma-point receiver as the research
candidate.  Apply the proposal in two separated tracks:

1. **Safe cleanup:** demote Stage 1 in the narrative, define explicit profiles,
   rename removed-noise diagnostics, and draft a single residual-contract API.
2. **Scientific validation:** validate fixed sigma-point TP-UIC across seeds,
   direct INR and DD-error axes before adding per-target gates, adaptive order,
   reference frames, association, or production `(M,Q)` interfaces.

Do not perform the proposed large directory rewrite or reference-aided mode in
the same change as residual semantics.  Those changes alter provenance and the
physical observation model respectively and require separate gates.

Machine evidence:

- `data/target_risk_gate_screen_20261002/summary.json`
- `data/target_risk_gate_screen_20261002/records.csv`
- `data/covariance_semantics_additive_20261002/`
- `data/covariance_semantics_replacement_20261002/`
