"""RCS-only extension of the joint V1 audit, with matched hard report caps.

No compensating antenna gain, power, geometry or observation-cap changes.
Import-time registration is required by Windows multiprocessing spawn.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_joint_revision import CONDITIONS, main

CONDITIONS.clear()
for rcs in [0.05, 0.1, 0.2]:
    for budget in [0, 2, 8]:
        CONDITIONS[f'rcs{rcs:g}_k{budget}'] = {
            'detect.target_rcs': rcs,
            'selector.max_remote_reports': budget,
        }

if __name__ == '__main__':
    main()
