# Workspace freeze — 2026-09-23

## Purpose

This checkpoint separates the reproducible repository state from local run
caches before the next architecture phase.  It is a provenance freeze, not a
claim that the V2 research gates have passed.

## Recovery point

- Pre-cleanup commit: `da01b1edddf8facd89cb2ef3e3ff06b22d63e3e8`.
- Safety branch: `codex/pre-workspace-freeze-20260923`.
- V1 identity remains governed by `release/V1_STABLE_MANIFEST.json`.

## Retention policy

- Source, tests, specifications, study reports, and study evidence under
  `studies/direction*/data` are versioned.
- The top-level `results/` directory is a regenerable local run cache and is
  ignored.  A result becomes evidence only after it is promoted to a study
  data directory and indexed by that study.
- Directories explicitly named `_incomplete_*` are retained only as provenance
  and must not be cited as completed evidence.
- Historical bulk outputs and obsolete root-level reports removed by this
  checkpoint remain recoverable from the safety branch and Git history.

## Scientific status at freeze

- V1 release identity: clean (95/95 frozen keys; 32/32 contract references).
- V2 receiver benefit, association benefit, and communication-constrained
  multi-target fusion remain behind the gates in
  `studies/direction3/STAGE_GATE_2026-09-23.md`.
- Pilot and diagnostic artifacts must not be promoted to paper claims merely
  because they are retained in this checkpoint.

## Verification status

Environment: CPython 3.11.0, NumPy 2.2.6.

At the start of cleanup, `python -m pytest -q` reported 803 passed, 7 xfailed,
and 8 failed.  The failures are tracked cleanup/architecture debt:

1. one new upward layer dependency (`core -> sensing`);
2. six library modules above the 150-line limit;
3. two test modules above the 350-line limit;
4. untracked source/study artifacts;
5. two documents flagged for mojibake;
6. one missing study-data reference;
7. five test modules without contract docstrings;
8. one test importing another test module.

The version-control failure is resolved by this checkpoint.  The remaining
items are not waived; they must be cleared before declaring the next release
candidate.
