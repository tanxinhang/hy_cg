"""Low-complexity Jacobian/WLS-MAP baseline invariants."""

from pathlib import Path

import numpy as np

from _targetstate_common import (
    RECEIVERS, TARGET, make_cfg, make_pair, make_scorers, make_world,
)
from isac_sim.receiver.target_state.fit import fit_shared_target_offset


def test_jacobian_wls_is_finite_and_uses_constant_exact_evaluations():
    cfg = make_cfg()
    truth, belief, base, belief_geom = make_world(cfg, scene=0, radius_m=50.0)
    observations = [
        make_pair(cfg, truth, belief_geom, base, rx, scene=0)[0]
        for rx in RECEIVERS
    ]
    fit = fit_shared_target_offset(
        cfg, observations, TARGET, belief, receivers=RECEIVERS,
        belief_geometry=belief_geom, scorers=make_scorers(cfg, observations),
        solver="jacobian_wls", prior_sigma_m=150.0,
    )

    assert fit.solver == "jacobian_wls"
    assert np.all(np.isfinite(fit.delta_xy_m))
    assert np.isfinite(fit.objective)
    # Only the fitted point and the prior centre are evaluated exactly.
    assert fit.evaluations <= 2
    assert fit.receiver_evaluations <= 2 * len(RECEIVERS)


def test_confirmatory_curve_freezes_coarse_to_fine_shared_solver():
    text = Path("tools/gate_crossfit_target_state_map.py").read_text(
        encoding="utf-8")
    block = text.split('if bool(getattr(args, "baseline_curve", False)):', 2)[2]
    block = block.split("else:", 1)[0]
    assert 'shared_args.solver = "coarse_to_fine"' in block
    assert 'shared_args.solver = "gated_topk"' not in block
