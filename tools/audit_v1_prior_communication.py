"""Paired zero-report controls for the prior-error conditions; same seed/protocol.

Import-time registration is intentional so spawned workers see the conditions.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_v1_joint_revision import CONDITIONS, main
CONDITIONS.update({
    'prior250_k0': {'prior.belief_sigma_pos_m':250., 'selector.max_remote_reports':0},
    'prior500_k0': {'prior.belief_sigma_pos_m':500., 'selector.max_remote_reports':0},
})
if __name__=='__main__':main()
