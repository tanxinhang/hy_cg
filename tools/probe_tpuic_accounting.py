"""Per-trial probe of the TP-UIC residual accounting."""
import sys

import numpy as np

sys.path.insert(0, "D:/Desktop/conference")

from isac_sim import cancellation as cx
from isac_sim.config import Config, apply_overrides, apply_preset
from isac_sim.model import build_base_gains, generate_geometry
from isac_sim.prior import perturbed_geometry
from tools.run_tp_uic_v1 import pick_receiver  # noqa: E402

cfg = apply_overrides(
    apply_preset(Config(), "small-uav-compact-800m"),
    {"geometry.area_xy": 600.0, "detect.target_rcs": 0.1, "run.seed": 2026,
     "run.verbose": False, "cancellation.enable": True},
)
rng = np.random.default_rng([cfg.run.seed, 0])
geom = generate_geometry(cfg, rng)
base = build_base_gains(cfg, geom, rng)
belief = perturbed_geometry(cfg, geom, cfg.prior.belief_sigma_pos_m,
                            cfg.prior.belief_sigma_vel_mps, rng)
j, q = pick_receiver(cfg, base)
M = cfg.scale.M
sense = np.full(M, cfg.radio.rho * cfg.radio.P_default)
obs = cx.build_observation(cfg, geom, belief, base, j, rng=rng, sense_power=sense,
                           radiated_power=sense, processing_gain=cfg.waveform.N * cfg.waveform.L,
                           hw_gain=1.0, weak_index=q)
n0 = obs.sigma2
print("receiver=%d weak=%d  n0=%.4g  K=%d  d=%d  prot_rank=%d"
      % (j, q, n0, obs.y.size, obs.X.shape[1], obs.basis_belief.shape[1]))
g = obs.X.conj().T @ obs.X
print("diag(G)/n0 =", np.round(np.real(np.diag(g)) / n0, 1))
print("cond(G)    = %.3e" % np.linalg.cond(g))

sub = cx.orthonormalise(np.zeros((obs.y.size, 0)))
for pv in (None, 1.0):
    ret, est = cx.residual_accounting(obs.X, sub, n0, pv)
    print("empty subspace, prior=%s -> retained=%.3e (%.1f n0)  estimate=%.3e (%.1f n0)"
          % (pv, ret, ret / n0, est, est / n0))

arms = cx.cancellation_arms(cfg, obs, weak_target=q)
print("\n%16s %11s %11s %11s %11s %11s %11s" % (
    "arm", "i_in", "i_res", "struct", "est(n)", "retained", "pred_est"))
for k, r in arms.items():
    print("%16s %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e" % (
        k, r.i_in, r.i_res, r.i_res_structural, r.i_res_estimate,
        r.i_res_retained, r.i_res_pred_estimate))
print("\nn0 = %.4g W" % n0)
