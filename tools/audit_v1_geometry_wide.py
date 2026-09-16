"""Deployment-distance sweep on the frozen joint V1 protocol.

Answers one operational question.  The baseline scene spreads 15 UAVs over a
4 km square (mean UAV--target slant 2109 m, mean UAV--UAV slant 2099 m), and at
that range the P_D >= 0.95 requirement is only met for RCS above roughly
20 m^2.  How much does pulling the formation in actually buy?

Two matched families, both frozen to the joint V1 protocol
(``target-local-v1`` preset, Gaussian replacement failure model, seed 10917,
MC=100, edit-free).  Only ``geometry.area_xy``, ``detect.target_rcs`` and
``selector.max_remote_reports`` move.

* Compact-deployment RCS curve -- conditions ``rcs{r}_k{k}`` for
  r in 1/5/20/50 m^2 and k in 0/8, all with ``geometry.area_xy = 600``.
  ``results_v1_rcs_600m`` already covers 0.05/0.1/0.2 m^2 at the same area, so
  concatenating the two directories yields the compact curve that can be read
  against ``results_v1_rcs_joint`` + ``results_v1_rcs_wide`` (same curve at
  4 km).  The k=0/k=8 pair also re-tests whether the inverted-U shape of the
  remote-report gain survives when the formation is compressed.

* Distance axis at RCS = 0.2 m^2, K = 8 -- conditions ``d{d}_rcs0.2_k8`` for
  d in 800/1000/1500/2000/3000 m.  With the existing 600 m and 4 km points this
  gives a single self-consistent P_D(distance) curve.

Geometry is the *only* thing that changes with ``area_xy``; altitude ranges,
radio hardware, priors and the failure model stay put, so the result is a
complete compact-deployment contrast and must not be attributed to radar path
loss alone (the UAV--UAV interference geometry shrinks in step).

Exploratory: one geometry family, one seed, 100 shared scenarios, no
equivalence or noninferiority claim.

Import-time registration of CONDITIONS is required by the Windows
multiprocessing spawn start method.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_joint_revision import CONDITIONS, main

AREA_COMPACT = 600.0
RCS_GRID = [1.0, 5.0, 20.0, 50.0]
BUDGETS = [0, 8]
DISTANCES = [800.0, 1000.0, 1500.0, 2000.0, 3000.0]
AXIS_RCS = 0.2
AXIS_BUDGET = 8

CONDITIONS.clear()
for rcs in RCS_GRID:
    for budget in BUDGETS:
        CONDITIONS[f'rcs{rcs:g}_k{budget}'] = {
            'detect.target_rcs': rcs,
            'selector.max_remote_reports': budget,
            'geometry.area_xy': AREA_COMPACT,
        }
for distance in DISTANCES:
    CONDITIONS[f'd{distance:g}_rcs{AXIS_RCS:g}_k{AXIS_BUDGET}'] = {
        'detect.target_rcs': AXIS_RCS,
        'selector.max_remote_reports': AXIS_BUDGET,
        'geometry.area_xy': distance,
    }

if __name__ == '__main__':
    main()
