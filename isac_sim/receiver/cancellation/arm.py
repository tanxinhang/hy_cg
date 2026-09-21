"""cancellation 积木：单臂中间结果容器。"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Tuple

from isac_sim.receiver.cancellation.manifold import Subspace

@dataclass
class _Arm:
    """One comparison arm, stored as its action on the three components."""

    name: str
    sub_direct: np.ndarray
    sub_target: np.ndarray
    sub_noise: np.ndarray
    h_hat: np.ndarray
    c_diag: np.ndarray
    subspace: Subspace
    candidates: Tuple[int, ...] = ()
    # What the arm removes from the *tested target's own* echo.  ``None`` when the
    # observation does not carry a truth to isolate (see ``eta_survive_q``).
    sub_target_q: np.ndarray | None = None
    eta_survive_pred_q: float = 1.0
    eta_survive_risk_q: float = 1.0
    i_res_moment_mean: float = 0.0
    i_res_moment_var: float = 0.0
    residual_direct_gram: np.ndarray | None = None
    residual_noise_eigenvalues: np.ndarray | None = None
    soft_mu: float | None = None
    # How the residual is predicted for this arm.  ``estimator`` runs eq. (3);
    # the other two arms do not estimate a channel at all, so running the
    # estimator formula on them would invent a residual for an arm that has
    # none -- exactly the 27 dB that the first run reported for ``no_ic``.
    predict: str = "estimator"
