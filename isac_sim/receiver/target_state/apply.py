"""积木：把共享偏移应用到一个观测（方案 §5 的字段白名单）。"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.containers import Observation
from isac_sim.receiver.cancellation.dictionaries import target_dictionary
from isac_sim.receiver.cancellation.protection import _protection_basis
from isac_sim.receiver.target_state.geometry import (
    shift_belief_geometry,
    shifted_target_sources,
)

__all__ = ["apply_target_offset"]


def apply_target_offset(cfg: Config, obs: Observation, target: int,
                        delta_xy_m, *, receiver: int,
                        belief_geometry) -> Observation:
    """只替换接收机可知的目标字典。

    改：``targets_belief``、``A``、``basis_belief``。
    不改：``y`` / ``x_direct`` / ``s_target`` / ``h_true`` / ``alpha_true`` /
    ``sigma2`` / ``X`` / ``direct`` / ``direct_est`` / ``targets``（真值源）/
    UAV 几何 / 其它目标的一切。
    """
    delta = np.asarray(delta_xy_m, dtype=float).reshape(2)
    geom = shift_belief_geometry(belief_geometry, int(target), delta)
    sources = shifted_target_sources(cfg, geom, int(receiver), int(target),
                                     obs.targets_belief or ())
    n_bins = int(obs.y.size)
    if cfg.cancellation.protect_targets:
        A = target_dictionary(cfg, sources)
    else:
        A = np.zeros((n_bins, 0), dtype=complex)
    basis = _protection_basis(
        cfg, sources, n_bins, protected_ids=obs.protected_targets
    )
    return replace(obs, targets_belief=sources, A=A, basis_belief=basis)
