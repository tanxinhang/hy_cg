"""fields（自 ``isac_sim/types.py`` 拆出）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Tuple


CERTIFICATE_FIELDS: Tuple[str, ...] = ("residual_power", "target_retention")


CERTIFICATE_SOURCE_FIELDS: Dict[str, str] = {
    "residual_power": "i_res",
    "target_retention": "eta_survive",
}


DIAGNOSTIC_FIELDS: Tuple[str, ...] = (
    "i_in",
    "i_res_structural",
    "i_res_estimate",
    "i_res_pred",
    "i_res_moment_mean",
    "i_res_pred_var",
    "i_in_pred",
    "i_res_retained",
    "i_res_pred_estimate",
    "eta_protect",
    "eta_survive",
    "noise_enhance_db",
    "protect_dim",
    "n_coefficients",
    "matched_stat",
    "candidates",
    "kappa",
    "rho",
    "mu",
    "rank",
    "gate_target",
    "support",
    "noise_enhance",
    "covariance_eig",
    "ncp",
)
