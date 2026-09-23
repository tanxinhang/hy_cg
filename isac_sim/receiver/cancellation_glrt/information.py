"""Matrix-valued TP-UIC interface consumed by cooperation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from isac_sim.receiver.cancellation import CancellationResult, Observation
from isac_sim.receiver.cancellation_glrt.glrt import target_conditioned_glrt
from isac_sim.receiver.cancellation_glrt.residual_form import ResidualModel


@dataclass(frozen=True)
class DetectionInformation:
    """Whitened target information after cancellation and nuisance rejection."""

    target: int
    value: float
    transfer_energy: float
    identifiable_fraction: float
    dof_real: int


@dataclass(frozen=True)
class StochasticDetectionInformation:
    """Covariance-change information for a random complex target response."""

    target: int
    eigenvalues: np.ndarray
    kld_10: float
    kld_01: float
    jeffreys: float
    n_looks: int

    def gaussian_pd(self, p_fa: float) -> float:
        """Moment-matched LLR detection probability at the requested PFA."""
        lam = np.asarray(self.eigenvalues, dtype=float)
        looks = float(self.n_looks)
        beta = lam / (1.0 + lam)
        constant = -looks * float(np.sum(np.log1p(lam)))
        mu0 = constant + looks * float(np.sum(beta))
        mu1 = constant + looks * float(np.sum(lam))
        var0 = looks * float(np.sum(beta * beta))
        var1 = looks * float(np.sum(lam * lam))
        if var0 <= 0.0 or var1 <= 0.0:
            return float(p_fa)
        threshold = mu0 + float(norm.ppf(1.0 - p_fa)) * np.sqrt(var0)
        return float(norm.sf((threshold - mu1) / np.sqrt(var1)))


def detection_information(
    cfg,
    obs: Observation,
    result: CancellationResult,
    model: ResidualModel,
    target: int,
    *,
    dictionary: str = "belief",
) -> DetectionInformation:
    """Return ``a_eff^H R_res^-1 a_eff`` after nuisance projection.

    ``TargetGLRT.ncp_unit`` is precisely the accumulated whitened energy left
    after applying ``G=I-F``, whitening by ``R_res`` and projecting out other
    target templates.  Exposing it here prevents the scheduler from collapsing
    the receiver to scalar residual power and retention certificates.
    """
    out = target_conditioned_glrt(
        cfg, obs, result, model, target=int(target), dictionary=dictionary
    )
    return DetectionInformation(
        target=int(target),
        value=max(float(out.ncp_unit), 0.0),
        transfer_energy=max(float(out.g_self.sum()), 0.0),
        identifiable_fraction=float(out.rho_weighted),
        dof_real=int(out.dof_real),
    )


def stochastic_detection_information(
    cfg,
    obs: Observation,
    model: ResidualModel,
    target: int,
    *,
    dictionary: str = "belief",
) -> StochasticDetectionInformation:
    """Matrix stochastic LLR information with other targets in ``C0``.

    The nonzero eigenvalues of ``C0^-1/2 S_q C0^-1/2`` are recovered from
    ``S_factor^H C0^-1 S_factor``.  A Woodbury update adds nuisance-target
    covariance to the TP-UIC residual covariance without forming a dense
    observation-sized matrix.
    """
    from isac_sim.receiver.cancellation_glrt.target_glrt import (
        _dictionary, _dictionary_centre_mask)

    A, ids = _dictionary(cfg, obs, dictionary)
    centres = _dictionary_centre_mask(cfg, obs, dictionary, A.shape[1])
    ids = np.asarray(ids)
    q = int(target)
    signal = model.signal_transfer(A[:, (ids == q) & centres])
    nuisance = model.signal_transfer(A[:, (ids != q) & centres])
    c_inv_signal = model.cov.inverse_matrix(signal)
    if nuisance.shape[1]:
        c_inv_nuisance = model.cov.inverse_matrix(nuisance)
        middle = np.eye(nuisance.shape[1]) + nuisance.conj().T @ c_inv_nuisance
        correction = c_inv_nuisance @ np.linalg.solve(
            middle, nuisance.conj().T @ c_inv_signal
        )
        c0_inv_signal = c_inv_signal - correction
    else:
        c0_inv_signal = c_inv_signal
    gram = signal.conj().T @ c0_inv_signal
    gram = 0.5 * (gram + gram.conj().T)
    eigenvalues = np.clip(np.linalg.eigvalsh(gram).real, 0.0, None)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    looks = int(cfg.detect.n_looks)
    kld10 = looks * float(np.sum(eigenvalues - np.log1p(eigenvalues)))
    kld01 = looks * float(np.sum(
        np.log1p(eigenvalues) - eigenvalues / (1.0 + eigenvalues)
    ))
    jeffreys = looks * float(np.sum(eigenvalues**2 / (1.0 + eigenvalues)))
    return StochasticDetectionInformation(
        target=q, eigenvalues=eigenvalues, kld_10=kld10,
        kld_01=kld01, jeffreys=jeffreys, n_looks=looks,
    )
