"""cancellation 积木：测量视图 mixin。"""

from __future__ import annotations

import numpy as np

class _MeasurementViewsB:
    def as_risk_fraction(
        self, *, tail_probability: float = 0.20, per_target: bool = False
    ) -> np.ndarray:
        """Cantelli upper certificate for model-aligned residual power.

        With probability at least ``1-tail_probability``, residual power is no
        larger than ``mean + sqrt((1-a)/a * variance)``.  The guarantee is for
        the stated random-phase/Gaussian receiver model, not for geometry error.
        """
        alpha = float(tail_probability)
        if not np.isfinite(alpha) or not 0.0 < alpha < 1.0:
            raise ValueError("tail_probability must lie in (0, 1)")
        mean = self.moment_mean_by_target if per_target else self.i_res_moment_mean
        var = self.moment_var_by_target if per_target else self.i_res_pred_var
        if mean is None or var is None:
            raise ValueError(
                "target-conditioned residual moments were not measured; call "
                "measure_receiver_context(..., all_targets=True)"
            )
        upper = np.asarray(mean, dtype=float) + np.sqrt(
            ((1.0 - alpha) / alpha) * np.maximum(np.asarray(var, dtype=float), 0.0)
        )
        denominator = np.asarray(self.i_in_pred, dtype=float)
        if per_target:
            denominator = denominator[:, None]
        out = np.ones_like(upper, dtype=float)
        np.divide(upper, denominator, out=out, where=denominator > 0.0)
        return out

    def as_prior_quantile_fraction(self, *, per_target: bool = False) -> np.ndarray:
        """Return deterministic prior-predictive residual quantiles."""
        source = (
            self.prior_quantile_fraction_by_target
            if per_target else self.prior_quantile_fraction
        )
        if source is None:
            raise ValueError(
                "prior residual quantile was not requested during measurement"
            )
        return np.asarray(source, dtype=float)

    def as_retention(
        self,
        per_target: int | None = None,
        clamp: bool = True,
        source: str = "field",
    ) -> np.ndarray:
        """分子桥：``(M,)`` 或 ``(M, Q)`` 的回波存活因子。

        ``source`` 决定桥接哪一个存活率，而这个选择**不是装饰性的** —— 在
        600 m 场景上两者相差约 1.5 dB（整场 0.697，被测目标 0.981）：

        ``"field"``  -- ``eta_survive``，整场比值。这是标量 SINR 检测器真正能
        消费的量：它统计残差里**所有**剩下的回波能量，包括 stage 2 拟合并释放
        掉的那些目标。它是保守的：它让分子为"逐目标检测器本来也不会用的能量"
        付费。

        ``"q"``      -- ``eta_survive_q``，被测目标自己的存活率。这是唯一能回答
        "对消器是否伤害了弱目标"的量，也是唯一不会随联合支撑恰好建模了多少
        **其它**目标而漂移的量。作为逐链路因子它是乐观的：它描述的是一个目标，
        不是整个场。

        报一个桥却不说明是哪一个，与报 ``kappa`` 却不说明信念模型是同一个错误，
        所以默认是保守的那个，另一个必须显式指定。

        ``clamp`` 把因子截到 ``1``。默认开启，因为实测存活率在某些接收机上
        **确实超过 1**（600 m 场景上 1.001）：联合阶段拟合了信念目标流形，于是
        可能在某个目标的块里留下比该目标真实回波**更多**的能量 —— 多出来的
        是其它回波与噪声恰好投影到该块上的那部分。那是存活比的一个伪影，不是
        免费增益 —— 标量 SINR 不得因此记功，所以桥接出来的量被截断，未截断的
        那份仍保留以便诊断。
        """
        if source not in ("field", "q", "per_target"):
            raise ValueError(
                "source must be 'field', 'q' or 'per_target', got %r" % (source,)
            )
        if source == "per_target":
            if self.eta_survive_by_target is None:
                raise ValueError(
                    "per-target survival was not measured; call "
                    "measure_receiver_context(..., all_targets=True)"
                )
            eta = np.asarray(self.eta_survive_by_target, dtype=float)
        else:
            eta = np.asarray(
                self.eta_survive if source == "field" else self.eta_survive_q,
                dtype=float,
            )
        if clamp:
            eta = np.clip(eta, 0.0, 1.0)
        if source == "per_target":
            if per_target is not None and eta.shape[1] != int(per_target):
                raise ValueError(
                    "measured per-target survival has Q=%d, requested %d"
                    % (eta.shape[1], int(per_target))
                )
            return eta
        if per_target is None:
            return eta
        return np.repeat(eta[:, None], int(per_target), axis=1)

    def as_predicted_retention(
        self, *, clamp: bool = True, risk: bool = False
    ) -> np.ndarray:
        """仅用信念的 ``(M,Q)`` 目标存活能力证书。"""
        source = (
            self.risk_retention_by_target
            if risk else self.predicted_retention_by_target
        )
        if source is None:
            raise ValueError(
                "predicted per-target survival was not formed; call "
                "measure_receiver_context(..., all_targets=True)"
            )
        eta = np.asarray(source, dtype=float)
        return np.clip(eta, 0.0, 1.0) if clamp else eta
