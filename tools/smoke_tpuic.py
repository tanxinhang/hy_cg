"""Smoke test for isac_sim.cancellation (TP-UIC V1)."""
import math
import sys
import time

import numpy as np

sys.path.insert(0, "D:/Desktop/conference")

from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import build_base_gains, generate_geometry, noise_power
from isac_sim import cancellation as cx

cfg = apply_overrides(
    apply_preset(Config(), "small-uav-compact-800m"),
    {
        "geometry.area_xy": 600.0,
        "detect.target_rcs": 0.1,
        "run.seed": 2026,
        "run.verbose": False,
        "cancellation.enable": True,
    },
)
print("N,L =", cfg.waveform.N, cfg.waveform.L, " M,Q =", cfg.scale.M, cfg.scale.Q)
print("n0 = %.4g" % noise_power(cfg))

rng = np.random.default_rng([cfg.run.seed, 0])
geom = generate_geometry(cfg, rng)
base = build_base_gains(cfg, geom, rng)
M = cfg.scale.M
sense = np.full(M, cfg.radio.rho * cfg.radio.P_default)
G_proc = cfg.waveform.N * cfg.waveform.L

t0 = time.time()
obs = cx.build_observation(
    cfg, geom, geom, base, 0, rng=rng, sense_power=sense, radiated_power=sense,
    processing_gain=G_proc, hw_gain=1.0,
)
print("observation build: %.2fs  K=%d  d=%d  A=%s  prot_rank=%d" % (
    time.time() - t0, obs.y.size, obs.X.shape[1], obs.A.shape, obs.basis_belief.shape[1]))
i_in = float(np.vdot(obs.x_direct, obs.x_direct).real)
s_in = float(np.vdot(obs.s_target, obs.s_target).real)
model_i = float((sense @ base.direct_gain)[0])
print("||x_direct||^2      = %.4g" % i_in)
print("I_sense_field[0]   = %.4g   (model)" % model_i)
print("ratio              = %.4f" % (i_in / model_i))
print("||s_target||^2     = %.4g" % s_in)
print("n0                 = %.4g" % obs.sigma2)

t0 = time.time()
arms = cx.cancellation_arms(cfg, obs)
print("arms: %.2fs" % (time.time() - t0))
print("%16s %9s %9s %9s %9s %7s %5s" % (
    "arm", "kappa", "kappaPred", "etaProt", "etaSurv", "noiseDB", "prot"))
for name, r in arms.items():
    print("%16s %9.2f %9.2f %9.4f %9.4f %7.1f %5d" % (
        name, r.kappa_db, r.kappa_pred_db, r.eta_protect, r.eta_survive,
        r.noise_enhance_db, r.protect_dim))

U = obs.basis_belief
if U.shape[1]:
    P = lambda v: U @ (U.conj().T @ v)
    num = np.linalg.norm(P(arms["tp_uic_stage1"].residual) - P(obs.y))
    den = max(np.linalg.norm(P(obs.y)), 1e-30)
    print("invariant ||P r1 - P y||/||P y|| = %.3e" % (num / den))
print("||MX||_F/||X||_F = %.4f" % (
    np.linalg.norm(obs.X - U @ (U.conj().T @ obs.X)) / max(np.linalg.norm(obs.X), 1e-30)))
print("predicted budget kappa = %.2f dB" % float(cx.kappa_from_budget(
    cfg, np.array([model_i]), np.array([M - 1]))[0]))
