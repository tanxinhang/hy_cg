"""Matched RCS/report-budget sweep in a 600 m square deployment.

Only horizontal deployment side length changes relative to audit_v1_rcs_joint.
Altitude ranges, radio hardware and prior errors remain unchanged.
"""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_joint_revision import CONDITIONS,main

CONDITIONS.clear()
for rcs in [0.05,0.1,0.2]:
    for budget in [0,2,8]:
        CONDITIONS[f'rcs{rcs:g}_k{budget}']={
            'detect.target_rcs':rcs,
            'selector.max_remote_reports':budget,
            'geometry.area_xy':600.,
        }

if __name__=='__main__':main()
