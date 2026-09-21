"""cancellation 积木：测量缓冲区。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.core.config import Config


@dataclass
class MeasureBuffers:
    """逐接收机（以及可选的逐接收机-目标）测量累加器。

    所有数组一次性分配好默认值：分数类的中性元是 ``1.0``（乘性分子的中性值），
    功率类的中性元是 ``0.0``，软惩罚系数用 NaN 表示"该臂走了硬回退"。
    这样"从未被观测承载的目标"不会污染中位数。
    """

    fraction: np.ndarray
    eta_field: np.ndarray
    eta_q: np.ndarray
    eta_protect: np.ndarray
    i_res: np.ndarray
    i_in: np.ndarray
    i_res_est: np.ndarray
    i_res_pred: np.ndarray
    i_res_moment_mean: np.ndarray
    i_res_pred_var: np.ndarray
    i_in_pred: np.ndarray
    noise_db: np.ndarray
    selected_soft_mu: np.ndarray
    eta_by_target: np.ndarray | None
    predicted_eta_by_target: np.ndarray | None
    risk_eta_by_target: np.ndarray | None
    fraction_by_target: np.ndarray | None
    predicted_fraction_by_target: np.ndarray | None
    model_fraction_by_target: np.ndarray | None
    i_res_by_target: np.ndarray | None
    i_res_pred_by_target: np.ndarray | None
    moment_mean_by_target: np.ndarray | None
    moment_var_by_target: np.ndarray | None
    prior_quantile_fraction: np.ndarray | None
    prior_quantile_fraction_by_target: np.ndarray | None
    selected_soft_mu_by_target: np.ndarray | None

    @classmethod
    def create(
        cls, cfg: Config, *, all_targets: bool,
        residual_quantile_probability: float | None,
    ) -> "MeasureBuffers":
        m = int(cfg.scale.M)
        q = int(cfg.scale.Q)
        shape = (m, q)
        want_q = all_targets
        want_pq = residual_quantile_probability is not None
        return cls(
            fraction=np.ones(m, dtype=float),
            eta_field=np.ones(m, dtype=float),
            eta_q=np.ones(m, dtype=float),
            eta_protect=np.zeros(m, dtype=float),
            i_res=np.zeros(m, dtype=float),
            i_in=np.zeros(m, dtype=float),
            i_res_est=np.zeros(m, dtype=float),
            i_res_pred=np.zeros(m, dtype=float),
            i_res_moment_mean=np.zeros(m, dtype=float),
            i_res_pred_var=np.zeros(m, dtype=float),
            i_in_pred=np.zeros(m, dtype=float),
            noise_db=np.zeros(m, dtype=float),
            selected_soft_mu=np.full(m, np.nan, dtype=float),
            eta_by_target=np.ones(shape, dtype=float) if want_q else None,
            predicted_eta_by_target=np.ones(shape, dtype=float) if want_q else None,
            risk_eta_by_target=np.ones(shape, dtype=float) if want_q else None,
            fraction_by_target=np.ones(shape, dtype=float) if want_q else None,
            predicted_fraction_by_target=np.ones(shape, dtype=float) if want_q else None,
            model_fraction_by_target=np.ones(shape, dtype=float) if want_q else None,
            i_res_by_target=np.zeros(shape, dtype=float) if want_q else None,
            i_res_pred_by_target=np.zeros(shape, dtype=float) if want_q else None,
            moment_mean_by_target=np.zeros(shape, dtype=float) if want_q else None,
            moment_var_by_target=np.zeros(shape, dtype=float) if want_q else None,
            prior_quantile_fraction=np.ones(m, dtype=float) if want_pq else None,
            prior_quantile_fraction_by_target=(
                np.ones(shape, dtype=float) if want_q and want_pq else None
            ),
            selected_soft_mu_by_target=(
                np.full(shape, np.nan, dtype=float) if want_q else None
            ),
        )
