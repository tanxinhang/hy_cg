"""鏂瑰悜 1 澶辫触鏍瑰洜 Part A锛歬appa 鐨勪笁椤瑰垎瑙ｅ疄娴嬨€?
琚瀵硅薄
--------
``isac_sim/receiver/cancellation/result.py:79`` 鐨?docstring 澹扮О::

    I_res = ||x - f(x)||^2 + ||f(n)||^2
    涓よ€呮浜わ紙f(n) 钀藉湪 span(M X) 閲岋紝缁撴瀯鎬ф畫宸笉鍦級锛屾墍浠ヨ繖涓拰鏄簿纭殑

姝ｆ槸杩欐潯鐞嗙敱鏀拺浜?鎶?``||f(n)||^2`` 璁拌繘鍒嗘瘝"銆傛柟鍚?1 鍒ゆ柇澶辫触锛屽繀椤诲洖鍒拌繖閲岋細
**杩欎釜鐞嗙敱鏈韩瀵逛笉瀵癸紵** 涓婁竴杞彧鍒ゅ喅浜嗘亽绛夊紡锛堟畫宸噷鏄?``n - f(n)``锛夛紝
浣嗘病鏈夋妸涓夐」鐨勯噺绾т笌绗﹀彿鍚屾椂瀹炴祴鍑烘潵 鈥斺€?鑰岀鍙锋鏄湰杞瀹＄殑銆?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?------------------------------
* **D1 涓诲椤?*锛歚`estimation / structural``銆傝嫢 >> 1 鈬?娣卞害琚櫔澹伴」涓诲锛?  鏀圭粨鏋勬€ф畫宸紙delta 闂ㄦ帶锛夊湪鐢熶骇鍙ｅ緞涓婂繀鐒剁湅涓嶈銆?* **D2 绗﹀彿锛堝喅瀹氭€э級**锛歚`||f(n)||^2 / ||n||^2`` = 鍣０閲岃鎶曞奖杩?``span(X)``
  骞?*琚噺鎺?*鐨勬瘮渚嬨€傚悓鏃舵姤 ``||n - f(n)||^2 / ||n||^2``锛堟畫宸噷鍓╀笅鐨勫櫔澹帮級銆?  涓よ€呬箣鍜屽簲涓?1锛堟浜ゅ垎瑙ｏ級銆傝繖鍐冲畾浜?``||f(n)||^2`` 绌剁珶鏄?  "娌℃秷鎺夌殑骞叉壈"锛坉ocstring 鐨勭珛鍦猴級杩樻槸"琚秷鎺夌殑鍣０"銆?* **D3 姝ｄ氦鎬?*锛歚`|<x - f(x), f(n)>| / (||x-f(x)|| * ||f(n)||)``銆?  docstring 澹扮О涓ら」姝ｇ‘浜わ紝瀹炴祴瀹冩槸鍚︽垚绔嬨€?* **D4 鐩磋繛/鍣０姣?*锛歚`||x||^2 / ||n||^2``锛屾妸 D1 鐨勬瘮鍊艰惤鍒扮粷瀵归噺绾т笂銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/diag_kappa_decomposition.py --trials 2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.sensing.model import build_base_gains, generate_geometry  # noqa: E402

BASE_OVERRIDES = {
    "geometry.area_xy": 600.0,
    "detect.target_rcs": 0.1,
    "scale.M": 6,
    "scale.Q": 3,
    "run.verbose": False,
    "cancellation.enable": True,
}


def make_cfg(**overrides) -> Config:
    merged = dict(BASE_OVERRIDES)
    merged.update(overrides)
    return apply_overrides(apply_preset(Config(), "small-uav-compact-800m"), merged)


def _norm2(v) -> float:
    a = np.asarray(v, dtype=complex)
    return float(np.vdot(a, a).real)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--receivers", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="studies/direction1/data/diag_kappa_decomposition")
    args = ap.parse_args()

    cfg = make_cfg(**{"run.seed": int(args.seed)})
    os.makedirs(args.out, exist_ok=True)

    m = int(cfg.scale.M)
    receivers = args.receivers if args.receivers else list(range(m))
    sense = np.full(m, float(cfg.radio.rho) * float(cfg.radio.P_default))

    rows = []
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([int(args.seed), int(trial)])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        for rx in receivers:
            orng = np.random.default_rng([int(args.seed), int(trial), int(rx)])
            obs = cx.build_observation(
                cfg, geom, geom, base, int(rx), rng=orng,
                sense_power=sense, radiated_power=sense,
                processing_gain=cfg.waveform.N * cfg.waveform.L,
                hw_gain=1.0,
            )
            res = cx.cancellation_arms(cfg, obs, only=args.arm)[args.arm]

            i_in = float(res.i_in)
            structural = float(res.i_res_structural)
            estimation = float(res.i_res_estimate)

            # --- 绾挎€ф€э細鎶?f 鍗曠嫭浣滅敤鍦ㄥ櫔澹颁笂 ---
            n_vec = (np.asarray(obs.y, dtype=complex)
                     - np.asarray(obs.x_direct, dtype=complex)
                     - np.asarray(obs.s_target, dtype=complex))
            n2 = _norm2(n_vec)
            # 鐢ㄥ悓涓€ solver 鍙杺鍣０锛屽緱鍒?r_n = n - f(n)
            def solve(y):
                import copy
                o2 = copy.copy(obs)
                o2.y = np.asarray(y, dtype=complex)
                return cx.cancellation_arms(cfg, o2, only=args.arm)[args.arm]

            r_n = solve(n_vec).residual
            r_n2 = _norm2(r_n)
            fn2 = _norm2(n_vec - r_n)          # ||f(n)||^2锛岀敱绾挎€ф€ф帹鍑?            x = np.asarray(obs.x_direct, dtype=complex)
            x_res = x - solve(x).residual      # = f(x)
            struct_vec = x - x_res             # x - f(x)
            struct_lin = _norm2(struct_vec)

            cross = abs(float(np.vdot(struct_vec, n_vec - r_n).real))
            denom = math.sqrt(struct_lin * fn2) if struct_lin > 0 and fn2 > 0 else 0.0

            rows.append({
                "trial": int(trial),
                "receiver": int(rx),
                "i_in": i_in,
                "structural": structural,
                "estimation": estimation,
                "structural_linearity": struct_lin,
                "noise_total": n2,
                "fn2": fn2,
                "residual_noise2": r_n2,
                # --- 姣斿€?---
                "est_over_struct": (estimation / structural) if structural > 0 else float("inf"),
                "fn2_over_n2": (fn2 / n2) if n2 > 0 else float("nan"),
                "resid_noise_over_n2": (r_n2 / n2) if n2 > 0 else float("nan"),
                "orthogonality_sum": ((fn2 + r_n2) / n2) if n2 > 0 else float("nan"),
                "cross_coherence": (cross / denom) if denom > 0 else 0.0,
                "x_over_n": (i_in / n2) if n2 > 0 else float("nan"),
                "kappa_db": float(res.kappa_db),
                "kappa_struct_db": (10.0 * math.log10(i_in / structural)
                                    if structural > 0 and i_in > 0 else None),
                "kappa_est_db": (10.0 * math.log10(i_in / estimation)
                                 if estimation > 0 and i_in > 0 else None),
            })
        print("  trial %d done -> %d rows" % (trial + 1, len(rows)), flush=True)

    def med(key):
        vals = [r[key] for r in rows if r[key] is not None and np.isfinite(r[key])]
        return float(np.median(vals)) if vals else float("nan")

    m = {("D1_est_over_struct_median"): med("est_over_struct"),
         "D2_fn2_over_n2_median": med("fn2_over_n2"),
         "D2_resid_noise_over_n2_median": med("resid_noise_over_n2"),
         "D2_orthogonality_sum_median": med("orthogonality_sum"),
         "D3_cross_coherence_median": med("cross_coherence"),
         "D4_x_over_n_db_median": 10.0 * math.log10(med("x_over_n")) if np.isfinite(med("x_over_n")) else None,
         "kappa_db_median": med("kappa_db"),
         "kappa_struct_db_median": med("kappa_struct_db"),
         "kappa_est_db_median": med("kappa_est_db"),
         "n_rows": len(rows)}

    print("\n=== D1 涓诲椤?===")
    print("  estimation / structural = %.4g" % m["D1_est_over_struct_median"])
    print("  kappa_db      = %.2f dB" % m["kappa_db_median"])
    print("  kappa_struct  = %.2f dB" % m["kappa_struct_db_median"])
    print("  kappa_est     = %.2f dB" % m["kappa_est_db_median"])
    print("\n=== D2 绗﹀彿锛堝喅瀹氭€э級 ===")
    print("  ||f(n)||^2 / ||n||^2        = %.6g  <- 琚姇褰卞苟鍑忔帀鐨勫櫔澹版瘮渚?
          % m["D2_fn2_over_n2_median"])
    print("  ||n - f(n)||^2 / ||n||^2    = %.6g  <- 娈嬪樊閲屽墿涓嬬殑鍣０"
          % m["D2_resid_noise_over_n2_median"])
    print("  涓よ€呬箣鍜岋紙搴斾负 1锛?         = %.9f" % m["D2_orthogonality_sum_median"])
    print("\n=== D3 姝ｄ氦鎬э紙docstring 澹扮О锛?===")
    print("  |<x-f(x), f(n)>| / 鑼冩暟绉?  = %.3e" % m["D3_cross_coherence_median"])
    print("\n=== D4 閲忕骇 ===")
    print("  ||x||^2 / ||n||^2           = %.2f dB" % m["D4_x_over_n_db_median"])

    print("\n" + json.dumps(m, indent=2, sort_keys=True))
    path = os.path.join(args.out, "decomposition.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"summary": m, "rows": rows}, fh, indent=2, sort_keys=True)
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
