"""Upper-RCS extension of the joint V1 audit.

``results_v1_rcs_joint`` already covers RCS = 0.05/0.1/0.2 m^2 on this exact
protocol.  This run adds 1/5/20/50 m^2 so the detection curve can be read over
one continuous grid spanning -13 to +17 dBsm without stitching in the V11
"transition" runs, which use a different scene (M=15/Q=10 with one local
observation per target, 50 scenarios, seed 2026).

Everything except ``detect.target_rcs`` and ``selector.max_remote_reports`` is
frozen to the joint V1 protocol: ``target-local-v1`` preset, Gaussian
replacement failure model, ``seed=10917``, edit-free.  Concatenating the two
directories therefore yields a single self-consistent sweep.

Import-time registration is required by Windows multiprocessing spawn.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_joint_revision import CONDITIONS, main

RCS_GRID = [1.0, 5.0, 20.0, 50.0]
BUDGETS = [0, 8]

CONDITIONS.clear()
for rcs in RCS_GRID:
    for budget in BUDGETS:
        CONDITIONS[f'rcs{rcs:g}_k{budget}'] = {
            'detect.target_rcs': rcs,
            'selector.max_remote_reports': budget,
        }

if __name__ == '__main__':
    main()
