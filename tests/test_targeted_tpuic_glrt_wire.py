"""Target-conditioned TP-UIC must expose one coherent covariance/GLRT arm."""

import numpy as np
from types import SimpleNamespace

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation import Observation
from isac_sim.receiver.cancellation_glrt.arm_table import arm_plans


def test_targeted_arm_plan_uses_only_tested_target_block():
    cfg = Config()
    obs = Observation(
        y=np.zeros(4, complex), X=np.eye(4, 1, dtype=complex),
        A=np.eye(4, 3, dtype=complex), x_direct=np.zeros(4, complex),
        s_target=np.zeros(4, complex), h_true=np.zeros(1, complex),
        alpha_true=np.zeros(3, complex), sigma2=1.0,
        basis_belief=np.eye(4, 2, dtype=complex), weak_index=1,
        A_target_ids=np.array([0, 1, 2]),
    )
    results = {
        "tp_uic_full": SimpleNamespace(candidates=(0,)),
        "targeted_tpuic_stage1": SimpleNamespace(candidates=()),
        "targeted_tpuic_full": SimpleNamespace(candidates=(1,)),
    }
    plans = arm_plans(cfg, obs, results)
    plan = plans["targeted_tpuic_full"]
    ids = np.asarray(obs.A_target_ids)
    assert plan.subspace.rank <= int(np.sum(ids == obs.weak_index))
    assert set(ids[list(plan.candidates)]) == {obs.weak_index}
