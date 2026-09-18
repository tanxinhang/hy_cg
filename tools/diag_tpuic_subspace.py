"""Diagnose the protection subspace vs the interference subspace (TP-UIC V1)."""
import sys

import numpy as np

sys.path.insert(0, "D:/Desktop/conference")

from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import build_base_gains, generate_geometry, noise_power
from isac_sim import cancellation as cx

cfg = apply_overrides(
    apply_preset(Config(), "small-uav-compact-800m"),
    {"geometry.area_xy": 600.0, "detect.target_rcs": 0.1, "run.seed": 2026,
     "run.verbose": False, "cancellation.enable": True},
)
rng = np.random.default_rng([cfg.run.seed, 0])
geom = generate_geometry(cfg, rng)
base = build_base_gains(cfg, geom, rng)
M = cfg.scale.M
sense = np.full(M, cfg.radio.rho * cfg.radio.P_default)
G_proc = cfg.waveform.N * cfg.waveform.L

obs = cx.build_observation(cfg, geom, geom, base, 0, rng=rng, sense_power=sense,
                           radiated_power=sense, processing_gain=G_proc, hw_gain=1.0)

print("DD offsets (doppler_bin, delay_bin) of the direct links at receiver 0:")
for s in obs.direct:
    print("   uav %2d  k=%+7.3f  l=%+7.3f  P*g=%.3e" % (s.uav, s.doppler_bin, s.delay_bin, s.power_at_receiver))
print("DD offsets of the echoes at receiver 0 (first 8 of %d):" % len(obs.targets))
for s in obs.targets[:8]:
    print("   uav %2d tgt %2d  k=%+7.3f  l=%+7.3f  P=%.3e" % (s.uav, s.target, s.doppler_bin, s.delay_bin, s.power))

U = obs.basis_belief
rank = U.shape[1]
print("\nprotection rank = %d / %d  (%.2f%%)" % (rank, obs.y.size, 100.0 * rank / obs.y.size))

Px = U @ (U.conj().T @ obs.x_direct)
x = obs.x_direct
print("||P x_direct||^2 / ||x_direct||^2 = %.4f" % (np.vdot(Px, Px).real / np.vdot(x, x).real))
Ps = U @ (U.conj().T @ obs.s_target)
s = obs.s_target
print("||P s_target||^2 / ||s_target||^2 = %.4f" % (np.vdot(Ps, Ps).real / np.vdot(s, s).real))

# Per-illuminator overlap with the protection subspace.
for src in obs.direct:
    a = cx.kernel_vector(cfg, src.doppler_bin, src.delay_bin)
    Pa = U @ (U.conj().T @ a)
    print("   uav %2d direct kernel -> protected fraction %.4f" % (
        src.uav, float(np.vdot(Pa, Pa).real / np.vdot(a, a).real)))

# Conditioning of the projected dictionary.
if rank:
    MX = U.astype(complex)  # placeholder
MX = obs.X - U @ (U.conj().T @ obs.X)
print("\n||MX||_F / ||X||_F = %.4f" % (np.linalg.norm(MX) / max(np.linalg.norm(obs.X), 1e-30)))
gram = MX.conj().T @ MX
print("cond(MX^H MX) = %.3e" % np.linalg.cond(gram))
g0 = obs.X.conj().T @ obs.X
print("cond(X^H X)   = %.3e" % np.linalg.cond(g0))
print("diag(X^H X)   =", np.round(np.real(np.diag(g0)), 6))
print("diag(MX^H MX) =", np.round(np.real(np.diag(gram)), 6))
