"""Truth-only decomposition of a fixed TP-UIC residual operator."""
from __future__ import annotations

import numpy as np

from isac_sim.receiver.cancellation_glrt.arm_table import arm_plans
from isac_sim.receiver.cancellation_glrt.probe import _low_rank_form


def target_basis(obs):
    target = np.asarray(obs.A, dtype=complex)
    if not target.shape[1]:
        return np.zeros((obs.y.size, 0), dtype=complex)
    basis, singular, _ = np.linalg.svd(target, full_matrices=False)
    keep = singular > 1e-10 * max(float(singular[0]), np.finfo(float).tiny)
    return basis[:, keep]


def _operator(cfg, obs, results, arm):
    plan = arm_plans(cfg, obs, results)[arm]
    basis, small, _ = _low_rank_form(cfg, obs, plan)
    return basis, small


def operator_noise_floor(cfg, obs, results, arm):
    """Exact sigma^2 ||(I-P_A)(I-F)||_F^2 from low-rank F."""
    target = target_basis(obs)
    basis, small = _operator(cfg, obs, results, arm)
    remaining = int(obs.y.size) - int(target.shape[1])
    if not basis.shape[1]:
        return float(obs.sigma2) * remaining
    tb = basis - target @ (target.conj().T @ basis)
    core = basis.conj().T @ tb
    cross = float(np.real(np.trace(core @ small)))
    correction = float(np.linalg.norm(tb @ small, "fro") ** 2)
    return float(obs.sigma2) * max(remaining - 2.0 * cross + correction, 0.0)


def decompose_residual(cfg, obs, results, arm):
    """Apply one frozen receiver operator to oracle additive components."""
    target = target_basis(obs)
    basis, small = _operator(cfg, obs, results, arm)

    def apply(vector):
        vector = np.asarray(vector, dtype=complex)
        if basis.shape[1]:
            vector = vector - basis @ (small @ (basis.conj().T @ vector))
        if target.shape[1]:
            vector = vector - target @ (target.conj().T @ vector)
        return vector

    source = {
        "direct": np.asarray(obs.x_direct, dtype=complex),
        "target": np.asarray(obs.s_target, dtype=complex),
    }
    source["noise"] = np.asarray(obs.y, dtype=complex) - sum(source.values())
    mapped = {name: apply(value) for name, value in source.items()}
    total_vector = apply(obs.y)
    component_sum = sum(mapped.values())
    energies = {name: float(np.vdot(value, value).real)
                for name, value in mapped.items()}
    total = float(np.vdot(total_vector, total_vector).real)
    scale = max(float(np.linalg.norm(total_vector)), np.finfo(float).tiny)
    return {
        **energies,
        "cross": total - sum(energies.values()),
        "total": total,
        "closure_error": float(np.linalg.norm(total_vector - component_sum) / scale),
    }


__all__ = ["decompose_residual", "operator_noise_floor", "target_basis"]
