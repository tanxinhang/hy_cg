"""Matrix-valued TP-UIC interface consumed by cooperation."""
from __future__ import annotations

from dataclasses import dataclass

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
