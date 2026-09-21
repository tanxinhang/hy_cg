"""把 delta（DD 格）翻译成物理上可判定的时钟同步精度（2026-09-21）。

kappa(delta) 的解析律由 tools/diag_delta_gap_attribution.py 标定：
    kappa = -10 log10(A * delta^2 + floor)
本脚本只做单位换算与反推，不跑仿真。
"""

from __future__ import annotations

import math

# 波形（paper-canonical / 默认）
L = 64
DF = 30e3
C = 3e8

# 由归因脚本标定的系数（aperture m_rx=4, M=6, Q=3, 600 m / RCS 0.1）
A = 2.1528
FLOOR = 7.9205e-08
GAMMA_REQ_DB = 52.25


def main() -> int:
    t_bin = 1.0 / (L * DF)
    r_bin = C * t_bin
    print("delay bin = %.6e s = %.2f m" % (t_bin, r_bin))
    print("kappa(delta) = -10log10(%.4f * delta^2 + %.4e)" % (A, FLOOR))
    print()
    print("%-22s %11s %10s %8s" % ("sync error", "delta[bin]", "kappa[dB]", "meet?"))
    cases = [
        ("0.1 ns (lab grade)", 0.1e-9),
        ("0.5 ns", 0.5e-9),
        ("1 ns (IEEE 1588)", 1e-9),
        ("1.56 ns (this round)", 1.5625e-9),
        ("3 ns", 3e-9),
        ("10 ns (GPS)", 10e-9),
        ("20 ns (GPS loose)", 20e-9),
    ]
    for name, tau in cases:
        d = tau / t_bin
        k = -10.0 * math.log10(A * d * d + FLOOR)
        print("%-22s %11.4g %10.2f %8s"
              % (name, d, k, "PASS" if k >= GAMMA_REQ_DB else "FAIL"))

    frac = 10.0 ** (-GAMMA_REQ_DB / 10.0)
    d_req = math.sqrt(max(frac - FLOOR, 0.0) / A)
    print("\n达标 %.2f dB 需要: delta <= %.4g bins" % (GAMMA_REQ_DB, d_req))
    print("  = %.4e s = %.3f ns = 路径长度误差 %.3f m"
          % (d_req * t_bin, d_req * t_bin * 1e9, d_req * r_bin))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
