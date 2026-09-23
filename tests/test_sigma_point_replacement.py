"""The physical sigma-point second moment must replace, not duplicate, nominal power."""

from isac_sim.core.config import Config, apply_overrides
from isac_sim.core.config.validate import validate_config


def test_sigma_point_replacement_is_an_explicit_valid_ablation():
    cfg = apply_overrides(Config(), {
        "cancellation.direct_mismatch_covariance_model": "sigma_point_replacement",
    })
    validate_config(cfg)
    assert cfg.cancellation.direct_mismatch_covariance_model == "sigma_point_replacement"
