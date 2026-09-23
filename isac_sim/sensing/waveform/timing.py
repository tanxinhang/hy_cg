"""Sensing-facing compatibility exports for the core timing rules."""
from isac_sim.core.config.timing import (
    configured_looks,
    dwell_from_looks_s,
    looks_from_dwell,
    otfs_frame_duration_s,
)

__all__ = [
    "configured_looks", "dwell_from_looks_s", "looks_from_dwell",
    "otfs_frame_duration_s",
]
