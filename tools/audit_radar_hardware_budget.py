#!/usr/bin/env python3
"""Audit the explicit bistatic radar hardware budget.

The table is noise-limited and before DD/collision losses. It is therefore a
necessary, not sufficient, hardware requirement; residual interference can
only increase the required gain.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from isac_sim.config import Config, PRESETS, apply_preset  # noqa: E402
from isac_sim.model import noise_power, radar_hardware_gain, wavelength  # noqa: E402


def required_net_gain_db(cfg: Config, rcs_m2: float, range_m: float,
                         target_raw_snr_db: float) -> float:
    """Net G_tx+G_rx-L_sys needed for a symmetric bistatic link."""
    p_sense = cfg.radio.rho * cfg.radio.P_default
    g_proc = (cfg.waveform.N * cfg.waveform.L
              if cfg.detect.sensing_processing_gain is None
              else cfg.detect.sensing_processing_gain)
    propagation = wavelength(cfg)**2 * rcs_m2 / (
        (4.0 * math.pi)**3 * range_m**4
    )
    echo_without_hardware = p_sense * propagation * g_proc
    required_linear = (
        10.0**(target_raw_snr_db / 10.0) * noise_power(cfg)
        / echo_without_hardware
    )
    return 10.0 * math.log10(required_linear)


def noise_limited_range_m(cfg: Config, rcs_m2: float,
                          target_raw_snr_db: float) -> float:
    """Symmetric bistatic range supported by the configured net gain."""
    p_sense = cfg.radio.rho * cfg.radio.P_default
    g_proc = (cfg.waveform.N * cfg.waveform.L
              if cfg.detect.sensing_processing_gain is None
              else cfg.detect.sensing_processing_gain)
    numerator = (
        p_sense * wavelength(cfg)**2 * rcs_m2
        * radar_hardware_gain(cfg) * g_proc
    )
    denominator = (
        (4.0 * math.pi)**3 * noise_power(cfg)
        * 10.0**(target_raw_snr_db / 10.0)
    )
    return (numerator / denominator)**0.25


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="small-uav-link-budget-bridge",
                        choices=sorted(PRESETS))
    parser.add_argument("--rcs", type=float, default=None,
                        help="override mean RCS in m^2")
    args = parser.parse_args()

    cfg = apply_preset(Config(), args.preset)
    rcs = cfg.detect.target_rcs if args.rcs is None else args.rcs
    if rcs <= 0:
        raise ValueError("RCS must be positive and is measured in m^2")

    r = cfg.radio
    net_db = (r.radar_net_gain_db if r.radar_net_gain_db is not None else
              r.radar_tx_gain_dbi + r.radar_rx_gain_dbi - r.radar_system_loss_db)
    effective_rcs = rcs * radar_hardware_gain(cfg)
    print("EXPLICIT BISTATIC RADAR HARDWARE-BUDGET AUDIT")
    print(f"preset={args.preset}")
    print(f"RCS={rcs:g} m^2 ({10*math.log10(rcs):+.2f} dBsm)")
    print(f"G_tx={r.radar_tx_gain_dbi:.2f} dBi, G_rx={r.radar_rx_gain_dbi:.2f} dBi, "
          f"L_sys={r.radar_system_loss_db:.2f} dB, net override={r.radar_net_gain_db}")
    print(f"net hardware gain={net_db:.2f} dB; algebraic effective RCS={effective_rcs:.3f} m^2")
    print(f"P_sense={r.rho*r.P_default:.3f} W, "
          f"B={cfg.waveform.N*cfg.waveform.delta_f/1e6:.3f} MHz, "
          f"G_proc={cfg.waveform.N*cfg.waveform.L:g}, N0={noise_power(cfg):.3e} W")
    print()
    print("Minimum net hardware gain (dB), symmetric bistatic range")
    print(f"{'range':>8} {'raw SNR=-10 dB':>18} {'raw SNR=0 dB':>16}")
    for range_m in (800.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0):
        a = required_net_gain_db(cfg, rcs, range_m, -10.0)
        b = required_net_gain_db(cfg, rcs, range_m, 0.0)
        print(f"{range_m/1000:7.1f} km {a:18.2f} {b:16.2f}")
    print()
    for snr_db in (-10.0, 0.0):
        reach = noise_limited_range_m(cfg, rcs, snr_db)
        print(f"configured-gain noise-limited symmetric reach at raw SNR={snr_db:+.0f} dB: "
              f"{reach/1000:.2f} km")
    print("CAUTION: DD loss, collision loss, aspect loss, and residual interference are excluded.")


if __name__ == "__main__":
    main()
