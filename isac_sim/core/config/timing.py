"""Physical timing rules shared by configuration validation and sensing."""
from __future__ import annotations

import math


def otfs_frame_duration_s(cfg) -> float:
    duration = float(cfg.waveform.N) * float(cfg.waveform.T)
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError("waveform.N * waveform.T must be positive and finite")
    return duration


def dwell_from_looks_s(cfg, n_looks: int) -> float:
    if int(n_looks) != n_looks or int(n_looks) < 1:
        raise ValueError("n_looks must be a positive integer")
    return int(n_looks) * otfs_frame_duration_s(cfg)


def looks_from_dwell(cfg, sensing_dwell_s: float) -> int:
    dwell = float(sensing_dwell_s)
    if not math.isfinite(dwell) or dwell <= 0.0:
        raise ValueError("sensing_dwell_s must be positive and finite")
    looks = math.floor(dwell / otfs_frame_duration_s(cfg) + 1e-9)
    if looks < 1:
        raise ValueError("sensing dwell is shorter than one OTFS frame")
    return looks


def configured_looks(cfg) -> int:
    dwell = getattr(cfg.detect, "sensing_dwell_s", None)
    return looks_from_dwell(cfg, dwell) if dwell is not None else int(cfg.detect.n_looks)
