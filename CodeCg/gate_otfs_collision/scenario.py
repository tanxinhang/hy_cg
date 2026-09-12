from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import json
import shutil
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .config import *


# -----------------------------
# Geometry helpers
# -----------------------------

def wrap_angle(x: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(x) + np.pi) % (2 * np.pi) - np.pi


def unit_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    n = np.linalg.norm(d) + EPS
    return d / n


def angle_xy(src: np.ndarray, dst: np.ndarray) -> float:
    d = dst[:2] - src[:2]
    return float(np.arctan2(d[1], d[0]))


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass
class Scenario:
    uav_pos: np.ndarray  # M x 3
    uav_vel: np.ndarray  # M x 3
    uav_bore: np.ndarray  # M, radians
    tgt_pos: np.ndarray  # Q x 3
    tgt_vel: np.ndarray  # Q x 3
    tgt_rcs: np.ndarray  # Q

    @property
    def M(self) -> int:
        return self.uav_pos.shape[0]

    @property
    def Q(self) -> int:
        return self.tgt_pos.shape[0]


def generate_gate1_scenario(cfg: SimConfig, rng: np.random.Generator) -> Scenario:
    """Controlled scenario: T1/T2 DD-angle close; T3/T4 DD close but angle-separated."""
    M, Q = 6, 4
    # UAVs on two arcs facing center.
    center = np.array([500.0, 500.0, cfg.target_alt_m])
    angles = np.deg2rad(np.array([-150, -105, -60, 40, 85, 130]))
    radius = 430.0
    uav_pos = np.zeros((M, 3))
    for m, a in enumerate(angles):
        uav_pos[m] = center + np.array([radius * np.cos(a), radius * np.sin(a), cfg.uav_alt_m])
    # Boresight to center.
    uav_bore = np.array([angle_xy(uav_pos[m], center) for m in range(M)])
    # Tangential-ish velocities.
    uav_vel = np.zeros((M, 3))
    for m, a in enumerate(angles):
        tangent = np.array([-np.sin(a), np.cos(a), 0.0])
        uav_vel[m] = cfg.uav_speed_mps * tangent

    # Targets: 1/2 are close in both space and direction for many UAVs.
    # 3/4 are chosen to have similar range/Doppler but wider angular separation.
    tgt_pos = np.array([
        [510.0, 490.0, cfg.target_alt_m],
        [540.0, 515.0, cfg.target_alt_m],
        [420.0, 610.0, cfg.target_alt_m],
        [620.0, 380.0, cfg.target_alt_m],
    ], dtype=float)
    tgt_vel = np.array([
        [14.0, -5.0, 0.0],
        [11.0, -3.0, 0.0],
        [12.0, 8.0, 0.0],
        [-7.0, 12.0, 0.0],
    ], dtype=float)
    # Mild RCS variation.
    tgt_rcs = np.array([1.0, 0.85, 0.9, 0.95])
    return Scenario(uav_pos, uav_vel, uav_bore, tgt_pos, tgt_vel, tgt_rcs)


def generate_clustered_scenario(cfg: SimConfig, M: int, Q: int, rng: np.random.Generator) -> Scenario:
    """Clustered UAV/target geometry for Gate 2 regime sweep."""
    G = 3
    centers = np.array([
        [250.0, 260.0, cfg.target_alt_m],
        [740.0, 260.0, cfg.target_alt_m],
        [500.0, 760.0, cfg.target_alt_m],
    ])
    # UAVs around cluster centers, elevated.
    uav_pos = np.zeros((M, 3))
    uav_vel = np.zeros((M, 3))
    uav_bore = np.zeros(M)
    for m in range(M):
        g = m % G
        # Offset around the cluster, enough spread for angular diversity.
        rad = rng.uniform(120, 260)
        ang = rng.uniform(0, 2 * np.pi)
        xy = centers[g, :2] + rad * np.array([np.cos(ang), np.sin(ang)])
        xy = np.clip(xy, 20, cfg.area_size_m - 20)
        uav_pos[m] = [xy[0], xy[1], cfg.uav_alt_m + rng.normal(0, 8)]
        uav_bore[m] = angle_xy(uav_pos[m], centers[g])
        heading = uav_bore[m] + rng.normal(0, 0.4)
        uav_vel[m] = cfg.uav_speed_mps * np.array([np.cos(heading), np.sin(heading), 0.0])

    # Targets in local groups; each group has a few closely spaced targets.
    tgt_pos = np.zeros((Q, 3))
    tgt_vel = np.zeros((Q, 3))
    tgt_rcs = np.zeros(Q)
    for q in range(Q):
        g = q % G
        group_center = centers[g, :2] + rng.normal(0, 80, size=2)
        # Local grouping radius controls natural DD-angle collisions.
        xy = group_center + rng.normal(0, 35, size=2)
        xy = np.clip(xy, 30, cfg.area_size_m - 30)
        tgt_pos[q] = [xy[0], xy[1], cfg.target_alt_m]
        vel_ang = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(5, cfg.target_speed_mps)
        tgt_vel[q] = speed * np.array([np.cos(vel_ang), np.sin(vel_ang), 0.0])
        tgt_rcs[q] = 10 ** rng.normal(0.0, 0.15)  # log-normal, mild
    return Scenario(uav_pos, uav_vel, uav_bore, tgt_pos, tgt_vel, tgt_rcs)
