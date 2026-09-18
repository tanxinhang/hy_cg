"""How concentrated is the OTFS DD kernel?  Feeds the TP-UIC protection design."""
import sys

import numpy as np

sys.path.insert(0, "D:/Desktop/conference")

from isac_sim.config import Config, apply_preset
from isac_sim import cancellation as cx

cfg = apply_preset(Config(), "small-uav-compact-800m")
K = cfg.waveform.N * cfg.waveform.L
print("grid = %d x %d = %d bins" % (cfg.waveform.N, cfg.waveform.L, K))
print("%9s %9s %10s %10s %10s %10s" % ("k_frac", "l_frac", "top1", "top9", "top25", "top81"))

for k in (0.0, 0.25, 0.5, 0.37):
    for l in (0.0, 0.5, 0.31):
        a = cx.kernel_vector(cfg, k, l)
        e = np.abs(a) ** 2
        e = np.sort(e)[::-1]
        cum = np.cumsum(e)
        row = []
        for n in (1, 9, 25, 81):
            row.append(float(cum[min(n, K) - 1]))
        print("%9.3f %9.3f %10.4f %10.4f %10.4f %10.4f" % (k, l, *row))

# Off-diagonal overlap between two nearby offsets: how fast does a kernel leave
# its neighbour's tangent plane?
base_k, base_l = 2.0, 2.0
a0 = cx.kernel_vector(cfg, base_k, base_l)
print("\nneighbour overlap |<a(k0,l0), a(k,l)>|^2 relative to the tangent plane:")
for dk in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
    a = cx.kernel_vector(cfg, base_k + dk, base_l)
    print("   dk=%5.2f  |<a0,a>|^2 = %.6f" % (dk, float(abs(np.vdot(a0, a)) ** 2)))

# Rank of a protection basis built from N target tangent sets.
print("\nrank of orth([tangent sets]) as a function of the number of targets:")
for n_tgt in (1, 2, 4, 8, 10):
    blocks = []
    rng = np.random.default_rng(0)
    for t in range(n_tgt):
        for i in range(14):
            k = 2.0 + rng.uniform(-1, 1)
            l = 2.0 + rng.uniform(-1, 1)
            blocks.append(cx.tangent_columns(cfg, k, l, 1, 0.05))
    B = np.concatenate(blocks, axis=1)
    U = cx.orthonormalise(B)
    print("   %2d targets x 14 illuminators: %4d columns -> rank %4d (%.2f%% of %d)" % (
        n_tgt, B.shape[1], U.rank, 100.0 * U.rank / K, K))
