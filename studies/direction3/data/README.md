# Direction 3 data index

## Current evidence

Only these matrix-chain artifacts support the current conclusions:

- `matrix_stochastic_active_trial32`: fixed-formation baseline.
- `matrix_formation_b100_matched_trial32`: 100 m movement budget.
- `matrix_formation_b200_matched_trial32`: 200 m movement budget.
- `matrix_formation_b300_matched_trial32`: 300 m movement budget.
- `matrix_formation_b400_matched_trial32`: 400 m, 16-look full TP-UIC.
- `matrix_formation_b400_looks8_trial32`: 400 m, 8-look audit.
- `matrix_formation_b400_no_ic_trial32`: no-cancellation ablation.
- `matrix_formation_b400_tp_uic_stage1_trial32`: Stage-1 ablation.
- `matrix_formation_b400_perfect_channel_trial32`: perfect-direct-path oracle.
- `matrix_formation_free_matched_trial32`: unconstrained structural bound.
- `matrix_formation_b400_comm_llr_trial32`: communication-delivered LLR fusion.

All use the stochastic covariance-LLR chain. Formation artifacts use
minimum-distance UAV-to-anchor matching. They remain trial-32 structural
audits, not multi-trial operational guarantees.

## Historical evidence

Directories named `closure_*` and earlier `trial32_*_audit` are retained only
for research provenance. They describe earlier scalar/certificate/resource
audits and must not be mixed numerically with the current matrix-chain table.

The old deterministic-GLRT artifacts reporting 0.056/2229 frames and
0.060/885 frames, plus the unmatched early formation artifacts, were deleted
because their protocol is incompatible with the current model.
