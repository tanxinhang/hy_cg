"""射频链路预算、标定样本数与波形损伤 INR。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import math

if TYPE_CHECKING:  # pragma: no cover - 仅供类型检查
    from isac_sim.core.config.run import Config


def check_radio(cfg: Config) -> None:
    """射频链路预算、标定样本数与波形损伤 INR。"""
    direct = (cfg.radio.P_sense_by_uav, cfg.radio.P_comm_by_uav)
    if any(value is not None for value in direct):
        if not all(value is not None for value in direct):
            raise ValueError(
                "radio.P_sense_by_uav and radio.P_comm_by_uav must be specified together"
            )
        if cfg.radio.P_by_uav is not None or cfg.radio.rho_by_uav is not None:
            raise ValueError(
                "direct sensing/communication powers cannot be mixed with legacy "
                "P_by_uav/rho_by_uav"
            )
        for name, values in zip(("P_sense_by_uav", "P_comm_by_uav"), direct):
            if (len(values) != cfg.scale.M or
                    not all(math.isfinite(x) and x >= 0.0 for x in values)):
                raise ValueError(
                    f"radio.{name} must have M finite non-negative powers"
                )
    if cfg.radio.rho_by_uav is not None:
        if (len(cfg.radio.rho_by_uav) != cfg.scale.M or
                not all(math.isfinite(x) and 0.0 < x < 1.0 for x in cfg.radio.rho_by_uav)):
            raise ValueError("radio.rho_by_uav must have M finite fractions in (0, 1)")
    radar_db_terms = (
        cfg.radio.radar_tx_gain_dbi,
        cfg.radio.radar_rx_gain_dbi,
        cfg.radio.radar_system_loss_db,
    )
    if not all(math.isfinite(value) for value in radar_db_terms):
        raise ValueError("radar antenna gains and system loss must be finite")
    if cfg.radio.radar_system_loss_db < 0.0:
        raise ValueError("radio.radar_system_loss_db must be non-negative")
    if (cfg.radio.radar_net_gain_db is not None
            and not math.isfinite(cfg.radio.radar_net_gain_db)):
        raise ValueError("radio.radar_net_gain_db must be finite when specified")
    if cfg.detect.fused_calibration_samples < 512:
        raise ValueError("detect.fused_calibration_samples must be at least 512")
    impairment_values = (
        cfg.waveform_impairments.clutter_inr,
        cfg.waveform_impairments.multipath_inr,
        cfg.waveform_impairments.unresolved_target_inr,
    )
    if any(value < 0.0 for value in impairment_values):
        raise ValueError("waveform impairment INR values must be non-negative")
