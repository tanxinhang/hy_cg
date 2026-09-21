"""cancellation 积木：测量视图 mixin。"""

from __future__ import annotations

import numpy as np

class _MeasurementViewsA:
    def as_fraction(self) -> np.ndarray:
        """The denominator bridge, in the shape ``compute_link_tables`` wants."""
        return np.asarray(self.fraction, dtype=float)

    def as_target_fraction(self, *, predicted: bool = False) -> np.ndarray:
        """Return the target-conditioned ``(receiver,target)`` residual bridge."""
        source = (
            self.predicted_fraction_by_target
            if predicted else self.fraction_by_target
        )
        if source is None:
            raise ValueError(
                "target-conditioned residual was not measured; call "
                "measure_receiver_context(..., all_targets=True)"
            )
        return np.asarray(source, dtype=float)

    def as_model_fraction(self, *, per_target: bool = False) -> np.ndarray:
        """Return realised residual normalised by the model's incoherent input."""
        source = self.model_fraction_by_target if per_target else self.model_fraction
        if source is None:
            raise ValueError(
                "target-conditioned model fraction was not measured; call "
                "measure_receiver_context(..., all_targets=True)"
            )
        return np.asarray(source, dtype=float)

    def as_residual_power(self, *, per_target: bool = False) -> np.ndarray:
        """Return realised absolute residual power for the model bridge.

        Absolute power is the canonical receiver-to-model contract.  Unlike a
        residual fraction, it does not require the consumer to reproduce the
        receiver's input-power denominator.  Target-conditioned values are
        recorded directly from each target-specific cancellation operator.
        """
        if not per_target:
            return np.asarray(self.i_res, dtype=float)
        if self.i_res_by_target is None:
            raise ValueError(
                "target-conditioned residual was not measured; call "
                "measure_receiver_context(..., all_targets=True)"
            )
        return np.asarray(self.i_res_by_target, dtype=float)

    def as_predicted_residual_power(self, *, per_target: bool = False) -> np.ndarray:
        """Return belief-only expected absolute residual power."""
        if not per_target:
            return np.asarray(self.i_res_pred, dtype=float)
        if self.i_res_pred_by_target is None:
            raise ValueError(
                "target-conditioned residual prediction was not formed; call "
                "measure_receiver_context(..., all_targets=True)"
            )
        return np.asarray(self.i_res_pred_by_target, dtype=float)

    def as_risk_residual_power(
        self, *, tail_probability: float = 0.20, per_target: bool = False
    ) -> np.ndarray:
        """Return the Cantelli upper certificate in absolute-power units."""
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
        return np.asarray(mean, dtype=float) + np.sqrt(
            ((1.0 - alpha) / alpha)
            * np.maximum(np.asarray(var, dtype=float), 0.0)
        )

    def as_prior_quantile_residual_power(
        self, *, per_target: bool = False
    ) -> np.ndarray:
        """Return the prior-predictive certificate in absolute-power units."""
        fraction = self.as_prior_quantile_fraction(per_target=per_target)
        scale = np.asarray(self.i_in_pred, dtype=float)
        if per_target:
            scale = scale[:, None]
        return fraction * scale
