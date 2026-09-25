"""共享脚手架：共享目标偏移 MAP 的场景/观测构造。

镜像 ``_tpuic_common.py`` 的模式：pytest 的 prepend 导入模式会把 ``tests/``
放进 ``sys.path``。信念误差**只注入到被测目标**（其余目标信念 = 真值），这是
第一版有意的隔离，理由见 ``tools/gate_crossfit_target_state_map.py``。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.cancellation.protection import _protection_basis
from isac_sim.receiver.target_state import build_scorer
from isac_sim.scenario.belief.state import BeliefState
from isac_sim.sensing.model import (
    build_base_gains,
    generate_geometry,
    radar_hardware_gain,
)
from tools import run_tpuic_receiver_benchmark as bench

TARGET = 2
RECEIVERS = (0, 1)


def make_cfg(target_rcs: float = 0.5, prior_sigma: float = 150.0):
    p = bench.build_parser().parse_args(["--out", "results/_tgt_state"])
    p.master_seed = 20261007
    p.area_xy = 400.0
    p.direct_dd_sigma = 0.10
    p.interference_tangent_order = 1
    p.interference_uncertainty_weighted = True
    p.direct_mismatch_covariance_model = "sigma_point_replacement"
    p.target_glrt_radius_bins = 0.25
    p.target_glrt_grid_points = 3
    p.max_protected_targets = 1
    p.belief_pos_sigma = prior_sigma
    p.target_rcs = target_rcs
    return bench._make_cfg(p)


def make_world(cfg, scene: int = 0, radius_m: float = 0.0, target: int = TARGET):
    """(truth, belief, base, belief_geometry)，只有被测目标带注入误差。"""
    seed = int(cfg.run.seed)
    truth = generate_geometry(cfg, np.random.default_rng([seed, scene, 101]))
    belief = BeliefState.from_truth(
        cfg, truth, np.random.default_rng([seed, scene, 202])
    )
    base = build_base_gains(cfg, truth, np.random.default_rng([seed, scene, 303]))
    xhat = np.concatenate(
        [np.array(truth.p_tgt, dtype=float), np.array(truth.v_tgt, dtype=float)],
        axis=1,
    )
    belief = replace(belief, xhat=xhat)
    angle = float(np.random.default_rng([seed, scene, 707]).uniform(0.0, 2.0 * np.pi))
    belief.xhat[target, 0] += radius_m * np.cos(angle)
    belief.xhat[target, 1] += radius_m * np.sin(angle)
    return truth, belief, base, belief.as_geometry(truth)


def make_pair(cfg, truth, belief_geom, base, rx: int, look: int = 0,
              scene: int = 0, target: int = TARGET):
    """(H1, H0)，保护集钉在被测目标上。"""
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([int(cfg.run.seed), scene, rx, look, 601])
    drng = np.random.default_rng([int(cfg.run.seed), scene, rx, 602])
    h1, h0 = cx.build_observation_pair(
        cfg, truth, belief_geom, base, rx, rng=rng, direct_error_rng=drng,
        sense_power=power, radiated_power=power,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), exclude_target=target,
        active_mask=np.ones(m, dtype=bool), weak_index=target, share_noise=False,
    )
    pin = frozenset({int(target)})
    return tuple(
        replace(obs, protected_targets=pin,
                basis_belief=_protection_basis(
                    cfg, obs.targets_belief, obs.y.size, protected_ids=pin))
        for obs in (h1, h0)
    )


def make_noiseless_look(cfg, truth, belief_geom, base, receivers,
                        target: int = TARGET):
    """``y = x_direct + s_target``：把"求解器能不能找到峰"和"信噪比够不够"分开测。"""
    out = []
    for rx in receivers:
        h1, _ = make_pair(cfg, truth, belief_geom, base, rx, target=target)
        out.append(replace(h1, y=h1.x_direct + h1.s_target))
    return out


def make_scorers(cfg, observations, target: int = TARGET):
    return [
        build_scorer(
            cfg, obs,
            gl.residual_model(
                cfg, obs, "tp_uic_full",
                cx.cancellation_arms(cfg, obs, weak_target=target,
                                     only="tp_uic_full"),
            ),
            target,
        )
        for obs in observations
    ]
