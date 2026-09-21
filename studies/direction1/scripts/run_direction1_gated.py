"""鏂瑰悜 1 澶勬柟**钀藉湴鍚?*鐨勭鍒扮澶嶆牳锛?026-09-21锛夈€?
涓?:mod:`tools.run_direction1_prescription` 鐨勫叧绯?-------------------------------------------------
涓婁竴杞偅涓剼鏈槸**鎵嬪伐**鎶?``estimation`` 浠?``i_res`` 閲屽噺鎺夋潵妯℃嫙鏂板彛寰勭殑::

    structural = max(i_res - i_est, 0)

鏈剼鏈敤**鐪熸钀藉湴鐨勯棬鎺?* ``cancellation.residual_accounting`` 璺戝悓涓€浠朵簨锛?浜庢槸澶氬嚭涓€鏉′笂涓€杞病鏈夌殑鍒ゆ嵁锛?
* **G5 绛変环鎬?* 鈥斺€?闂ㄦ帶缁欏嚭鐨?``fraction`` 蹇呴』閫愪綅锛堢浉瀵?1e-12锛夌瓑浜庝笂涓€杞?  鎵嬪伐鐨?``(i_res - i_est) / i_in``銆傝繖鏉″鏋滀笉杩囷紝璇存槑闂ㄦ帶瀹炵幇鐨勪笉鏄鏂癸紝
  涓婁竴杞偅 +0.267 / +0.367 鐨勬敹鐩婂氨涓嶈兘绠楀湪鏈淇敼澶翠笂銆?
涓夎噦锛堥厤瀵癸細鍚屽嚑浣曘€佸悓鍣０銆佸悓鍒嗗瓙锛?-----------------------------------
* ``measured``          榛樿鍙ｅ緞锛堝熀绾匡紝閫愪綅绛変簬宸插彂甯冭矾寰勶級
* ``structural_delta0`` structural 鍙ｅ緞 + delta=0 鈬?**涓婄晫**锛堝瓧鍏稿畬澶囷紝涓嶅彲杈撅級
* ``structural_delta``  structural 鍙ｅ緞 + delta=3e-3 鈬?**鐪熷疄鐗?*锛堝彲鎶ョ殑鏁帮級

鍒ゆ嵁锛堣窇鍓嶅畾姝伙級
----------------
* **G1**锛歴tructural_delta0 - measured 閰嶅 螖P_D 鈮?+0.02
* **G2**锛歴tructural_delta  - measured 閰嶅 螖P_D 鈮?+0.02锛堣矾寰?1 鍦ㄧ湡瀹炵増涓嬩粛鎴愮珛锛?* **G3**锛氫笁鑷?P_FA 鏋佸樊 鈮?0.01锛堜笉鏄棬闄愬け鐪燂級
* **G5**锛氶棬鎺?鈮?鎵嬪伐锛堢浉瀵硅宸?鈮?1e-12锛?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/run_direction1_gated.py --trials 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from tools.run_tpuic_production import _detection, _selector  # noqa: E402


def build_config(args, **extra) -> Config:
    cfg = apply_preset(Config(), args.preset)
    ov = {
        "geometry.area_xy": float(args.area),
        "detect.target_rcs": float(args.rcs),
        "scale.M": int(args.m),
        "scale.Q": int(args.q),
        "run.seed": int(args.seed),
        "run.verbose": False,
        "cancellation.enable": True,
        "aperture.enable": True,
        "aperture.m_rx": int(args.m_rx),
    }
    ov.update(extra)
    return apply_overrides(cfg, ov)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--delta", type=float, default=3e-3)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--out", default="studies/direction1/data/direction1_gated")
    args = ap.parse_args(argv)

    cfg_base = build_config(args)
    cfg_s0 = build_config(args, **{"cancellation.residual_accounting": "structural"})
    cfg_sd = build_config(args, **{
        "cancellation.residual_accounting": "structural",
        "cancellation.direct_estimation_sigma_delay_bins": float(args.delta),
    })
    os.makedirs(args.out, exist_ok=True)

    rows = []
    max_def = 0.0
    max_hand = 0.0
    hand_db: list[float] = []
    gated_db: list[float] = []
    t0 = time.time()
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([cfg_base.run.seed, int(trial)])
        geom = generate_geometry(cfg_base, rng)
        base = build_base_gains(cfg_base, geom, rng)

        def measure(cfg, tag):
            mrng = np.random.default_rng([cfg_base.run.seed, 10 ** 6 + int(trial), tag])
            return cx.measure_receiver_context(
                cx.ReceiverContext.from_trial(cfg, geom, geom, base, arm=args.arm),
                rng=mrng,
            )

        m_meas = measure(cfg_base, 0)
        m_s0 = measure(cfg_s0, 1)
        m_sd = measure(cfg_sd, 2)

        # G5b锛氶棬鎺у彛寰?vs 涓婁竴杞墜宸ュ彛寰?        #
        # 鈿狅笍 涓嶈兘鎷挎墜宸ョ粨鏋滃綋"鐪熷€?鍘绘瘮**鐩稿**璇樊锛歚`estimation`` 鍗?``i_res``
        # 鐨?99.8%锛屾墍浠?``(i_res - i_est)`` 鏄竴娆?*鐏鹃毦鎬ф姷娑?*锛屼綆浣嶅叏涓紝
        # 宸€煎彲鑳藉綊闆剁敋鑷冲彇鍒拌垗鍏ユ畫鐣欍€傛墜宸ュ彛寰勫湪娴偣涓婃湰鏉ュ氨涓嶅彲闈?鈥斺€?        # 杩欐鏄繀椤绘妸鍙ｅ緞鍋氭垚闂ㄦ帶锛堢洿鎺ョ敤 ``i_res_structural`` 瀛楁锛夌殑鐞嗙敱銆?        # G5a锛堥棬鎺?== 瀹氫箟锛夊湪鑷傜骇鐢?tests/test_residual_accounting_mode.py 閽変綇锛?        # 鑱氬悎瀵硅薄 ReceiverMeasurement 涓嶆毚闇?i_res_structural锛岃繖閲屽彧閲忓け鐪熴€?        i_res = np.asarray(m_meas.i_res, dtype=float)
        i_est = np.asarray(m_meas.i_res_estimate, dtype=float)
        i_in = np.asarray(m_meas.i_in, dtype=float)
        hand = np.where(i_in > 0.0,
                        np.maximum(i_res - i_est, 0.0) / np.maximum(i_in, 1e-300), 1.0)
        gated = np.asarray(m_s0.fraction, dtype=float)
        # 涓よ€呴兘宸叉寜 i_in 褰掍竴锛屾墍浠ョ粷瀵瑰樊灏辨槸"鍗?i_in 鐨勬瘮渚?銆?        max_hand = max(max_hand, float(np.max(np.abs(gated - hand))))
        hand_db.append(float(np.median(-10.0 * np.log10(np.maximum(hand, 1e-300)))))
        gated_db.append(float(np.median(-10.0 * np.log10(np.maximum(gated, 1e-300)))))

        # 鍒嗗瓙鍥哄畾锛氭湰瀹為獙鍙兂鍔ㄥ垎姣嶅彛寰勩€?        retention = np.clip(np.asarray(m_meas.eta_survive, dtype=float), 0.0, 1.0)

        for label, m in (("measured", m_meas),
                         ("structural_delta0", m_s0),
                         ("structural_delta", m_sd)):
            frac = np.asarray(m.fraction, dtype=float)
            tables = compute_link_tables(
                cfg_base, base,
                residual_fraction_by_receiver=frac,
                target_retention_by_receiver=retention,
            )
            selected = _selector(cfg_base, base, tables)
            det_rng = np.random.default_rng([cfg_base.run.seed, 4 * 10 ** 6 + int(trial)])
            det = _detection(cfg_base, tables, selected, det_rng, base, int(trial))
            rinr = np.asarray(tables.rinr, dtype=float)
            rinr = rinr[np.isfinite(rinr) & (rinr > 0.0)]
            rows.append({
                "trial": int(trial), "arm": label,
                "p_d": det["p_d"], "p_fa": det["p_fa"],
                "n_selected": det["n_selected"],
                "kappa_db_median": float(np.median(
                    -10.0 * np.log10(np.maximum(frac, 1e-300)))),
                "rinr_db_median": (
                    float(np.median(10.0 * np.log10(rinr))) if rinr.size else float("nan")
                ),
            })
        print("  trial %d/%d [%.0f s]" % (trial + 1, args.trials, time.time() - t0),
              flush=True)

    keys = list(rows[0])
    path = os.path.join(args.out, "gated.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    def col(label, key):
        return np.array([r[key] for r in rows if r["arm"] == label], dtype=float)

    labels = ["measured", "structural_delta0", "structural_delta"]
    print("\n=== 闂ㄦ帶钀藉湴鍚庣殑绔埌绔紙閰嶅锛屽悓鍑犱綍/鍚屽櫔澹?鍚屽垎瀛愶級 ===")
    print(f"{'arm':>20} {'kappa[dB]':>10} {'P_D':>8} {'P_FA':>8} "
          f"{'links':>7} {'rinr':>8} {'dP_D':>8}")
    summary = {"arms": {}}
    base_pd = float(np.mean(col("measured", "p_d")))
    for lb in labels:
        pd_ = float(np.mean(col(lb, "p_d")))
        summary["arms"][lb] = {
            "p_d": pd_, "p_fa": float(np.mean(col(lb, "p_fa"))),
            "n_selected": float(np.mean(col(lb, "n_selected"))),
            "kappa_db": float(np.median(col(lb, "kappa_db_median"))),
            "rinr_db": float(np.median(col(lb, "rinr_db_median"))),
            "delta_p_d": pd_ - base_pd,
        }
        s = summary["arms"][lb]
        print(f"{lb:>20} {s['kappa_db']:10.2f} {pd_:8.4f} {s['p_fa']:8.4f} "
              f"{s['n_selected']:7.2f} {s['rinr_db']:8.2f} {pd_ - base_pd:+8.4f}")

    def paired(a, b):
        d = col(a, "p_d") - col(b, "p_d")
        n = d.size
        return {"delta_p_d": float(np.mean(d)),
                "se": float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
                "n_pairs": int(n)}

    for name, a, b in (("G1_model_delta0", "structural_delta0", "measured"),
                       ("G2_real_delta", "structural_delta", "measured")):
        summary[name] = paired(a, b)
        summary[name]["pass"] = bool(summary[name]["delta_p_d"] >= 0.02)

    fas = [summary["arms"][lb]["p_fa"] for lb in labels]
    summary["G3_honesty"] = {"p_fa_spread": float(max(fas) - min(fas)),
                             "pass": bool(max(fas) - min(fas) <= 0.01)}
    summary["G5a_gate_equals_definition"] = {
        "note": "闂ㄦ帶 fraction == i_res_structural/i_in锛涚敱 "
                "tests/test_residual_accounting_mode.py 鍦ㄨ噦绾ч拤浣?
                "锛堣仛鍚堝璞?ReceiverMeasurement 涓嶆毚闇?i_res_structural锛?,
    }
    summary["G5b_hand_computation_distortion"] = {
        "max_abs_error_over_i_in": max_hand,
        "hand_kappa_db_median": float(np.median(hand_db)),
        "gated_kappa_db_median": float(np.median(gated_db)),
        "note": "(i_res - i_est) 鏄伨闅炬€ф姷娑堬紙estimation 鍗?99.8%锛夛紝"
                "鎵嬪伐鍙ｅ緞鍦ㄦ诞鐐逛笂涓嶅彲闈?鈬?涓婁竴杞殑鏁板瓧鍙兘浣滈噺绾у弬鑰?,
    }

    summary["config"] = {"preset": args.preset, "area": args.area, "rcs": args.rcs,
                         "seed": args.seed, "trials": int(args.trials),
                         "delta_bins": float(args.delta)}
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("\n" + json.dumps({k: v for k, v in summary.items()
                             if k.startswith("G")}, indent=2, sort_keys=True))
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
