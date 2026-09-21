"""瀹¤锛氱湡瀹炵増锛坉elta>0锛変笌"棰勬湡"鐨勫樊璺濆埌搴曠敱浠€涔堟瀯鎴愶紙2026-09-21锛夈€?
闂
----
闂ㄦ帶钀藉湴鍚庡疄娴嬶紙n=20 閰嶅锛夛細

    measured            kappa 37.17  P_D 0.5167
    structural_delta0   kappa 67.44  P_D 0.8667   锛堟ā鍨嬬増锛宒elta=0锛?    structural_delta    kappa 46.96  P_D 0.7167   锛堢湡瀹炵増锛宒elta=3e-3锛?
"鐪熷疄鐗?姣?妯″瀷鐗?浣?0.150锛屾瘮澶╄姳鏉?perfect 浣?0.167 鈥斺€?宸窛瀛樺湪锛屼絾
**瀹冨埌搴曟槸闅愯棌 bug锛岃繕鏄煇涓凡瀹氶噺鏈哄埗鐨勮嚜鐒剁粨鏋?*锛熸湰鑴氭湰鎶婂畠鎷嗗紑銆?
涓夋潯浜掓枼鍋囪
------------
* **H1 鍏ㄩ儴鐢?kappa 瑙ｉ噴**锛堟棤 unexplained gap锛?  楠岃瘉锛氭妸姣忔。 (kappa, P_D) 鎶曞埌鐙珛娴嬪緱鐨勫ぉ鑺辨澘鏇茬嚎 P_D(kappa) 涓婏紝
  娈嬪樊搴斿湪 MC 鍣０鍐呫€傝嫢鎴愮珛锛屽樊璺?= kappa 宸窛锛屾棤闅愯棌 bug銆?* **H2 kappa 宸窛鐢?delta 鍙栧€艰В閲?*
  楠岃瘉锛歬appa(delta) 搴旀湇浠庤В鏋愬緥 ``-10log10(A*delta^2 + floor)``銆?  鑻ユ垚绔嬶紝"鐪熷疄鐗堜綆"鍙槸鍥犱负 delta 鍙栦簡 3e-3锛屼笉鏄満鍒跺け鏁堛€?* **H3 delta 杩橀澶栦激浜嗗垎瀛?*锛坋ta_survive锛夛紝浣跨湡瀹炵増鏀剁泭鍐嶆墦鎶?  楠岃瘉锛氬悓涓€妗?delta 涓嬶紝鍒嗗瓙鍥哄畾 vs 鍒嗗瓙闅忚噦锛屾瘮杈?P_D銆?  鑻ヤ袱鑰呬笉鍚岋紝璇存槑宸叉姤鐨?+0.200 閲屾湁鍒嗗瓙鎹熷け娌＄畻杩涘幓銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/diag_delta_gap_attribution.py --trials 20
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

# 澶╄姳鏉挎洸绾?P_D(kappa)锛氱嫭绔嬪疄楠屽疄娴嬶紙tools/diag_direction1_ceiling.py锛?# n=20锛屽悓 preset/鍑犱綍/鍒嗗瓙锛夈€傛湰鑴氭湰鍙嬁瀹冨綋"kappa -> P_D"鐨勪紶閫掑嚱鏁帮紝
# 鐢ㄦ潵鍒ゆ柇宸窛鑳藉惁鍏ㄩ儴褰掔粰 kappa銆?CEILING_KAPPA = [0.0, 20.0, 36.50, 45.00, 52.25, 60.00, 68.00]
CEILING_PD = [0.0000, 0.0667, 0.4667, 0.7167, 0.7500, 0.8500, 0.8667]
# perfect 妗ｏ紙kappa -> inf锛夊崟鐙锛屼笉杩涙彃鍊笺€?CEILING_PERFECT_PD = 0.8833

# 闇€姹傛繁搴︼紙鍑犱綍閲忥細600 m / RCS 0.1锛?GAMMA_REQ_DB = 52.25


def ceiling_pd(kappa_db: float) -> float:
    """澶╄姳鏉挎洸绾跨殑鍒嗘绾挎€ф彃鍊硷紙kappa 瓒呭嚭绔偣鍒欏彇绔偣锛夈€?""
    if kappa_db <= CEILING_KAPPA[0]:
        return CEILING_PD[0]
    if kappa_db >= CEILING_KAPPA[-1]:
        return CEILING_PD[-1]
    for i in range(len(CEILING_KAPPA) - 1):
        k0, k1 = CEILING_KAPPA[i], CEILING_KAPPA[i + 1]
        if k0 <= kappa_db <= k1:
            t = (kappa_db - k0) / (k1 - k0)
            return CEILING_PD[i] + t * (CEILING_PD[i + 1] - CEILING_PD[i])
    return float("nan")


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
    ap.add_argument("--deltas", type=float, nargs="*",
                    default=[0.0, 1e-3, 1.89e-3, 3e-3])
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--out", default="studies/direction1/data/diag_delta_gap")
    args = ap.parse_args(argv)

    cfg_base = build_config(args)
    # 姣忔。 delta 涓€浠?config锛坰tructural 鍙ｅ緞锛?    cfgs = {}
    for d in args.deltas:
        cfgs[float(d)] = build_config(args, **{
            "cancellation.residual_accounting": "structural",
            "cancellation.direct_estimation_sigma_delay_bins": float(d),
        })
    os.makedirs(args.out, exist_ok=True)

    rows = []
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
        ret_fixed = np.clip(np.asarray(m_meas.eta_survive, dtype=float), 0.0, 1.0)

        # tag 浠?1 璧凤紝姣忔。 delta 鍚勫崰涓€涓簰涓嶇浉鍚岀殑 tag
        for tag_i, d in enumerate(args.deltas):
            m_sd = measure(cfgs[float(d)], 1 + tag_i)
            frac = np.asarray(m_sd.fraction, dtype=float)
            ret_own = np.clip(np.asarray(m_sd.eta_survive, dtype=float), 0.0, 1.0)
            for num_mode, retention in (("fixed", ret_fixed), ("own", ret_own)):
                tables = compute_link_tables(
                    cfg_base, base,
                    residual_fraction_by_receiver=frac,
                    target_retention_by_receiver=retention,
                )
                selected = _selector(cfg_base, base, tables)
                det_rng = np.random.default_rng(
                    [cfg_base.run.seed, 4 * 10 ** 6 + int(trial)])
                det = _detection(cfg_base, tables, selected, det_rng, base, int(trial))
                rows.append({
                    "trial": int(trial),
                    "delta": float(d),
                    "numerator": num_mode,
                    "p_d": det["p_d"],
                    "p_fa": det["p_fa"],
                    "n_selected": det["n_selected"],
                    "kappa_db": float(np.median(
                        -10.0 * np.log10(np.maximum(frac, 1e-300)))),
                    "eta_median": float(np.median(retention)),
                })
        print("  trial %d/%d [%.0f s]" % (trial + 1, args.trials, time.time() - t0),
              flush=True)

    path = os.path.join(args.out, "attribution.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def sel(d, num):
        return [r for r in rows
                if abs(r["delta"] - float(d)) < 1e-15 and r["numerator"] == num]

    def mean(d, num, key):
        return float(np.mean([r[key] for r in sel(d, num)]))

    def se_paired(d, num_a, num_b):
        a = np.array([r["p_d"] for r in sel(d, num_a)], dtype=float)
        b = np.array([r["p_d"] for r in sel(d, num_b)], dtype=float)
        diff = a - b
        return float(np.std(diff, ddof=1) / math.sqrt(diff.size)) if diff.size > 1 \
            else float("nan")

    # --- 琛?1锛欻1 / H2 ----------------------------------------------------
    print("\n=== H1锛氬樊璺濊兘鍚﹀叏閮ㄥ綊缁?kappa锛燂紙鍒嗗瓙鍥哄畾锛屼笌 gated 鍚屽彛寰勶級 ===")
    print(f"{'delta':>9} {'kappa':>7} {'P_D':>7} {'P_D(ceil)':>10} "
          f"{'resid':>7} {'SE':>6} {'eta':>6}")
    table1 = []
    for d in args.deltas:
        k = float(np.median([r["kappa_db"] for r in sel(d, "fixed")]))
        pd_ = mean(d, "fixed", "p_d")
        pred = ceiling_pd(k)
        # 閰嶅 SE锛氱敤璇ユ。涓?鑷韩鍧囧€?鐨勭鏁?        arr = np.array([r["p_d"] for r in sel(d, "fixed")], dtype=float)
        se = float(np.std(arr, ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
        eta = float(np.median([r["eta_median"] for r in sel(d, "fixed")]))
        resid = pd_ - pred
        table1.append({"delta_bins": float(d), "kappa_db": k, "p_d": pd_,
                       "p_d_ceiling": pred, "residual": resid, "se": se,
                       "eta_median": eta})
        print(f"{d:9.4g} {k:7.2f} {pd_:7.4f} {pred:10.4f} "
              f"{resid:+7.4f} {se:6.4f} {eta:6.3f}")
    max_resid = max(abs(r["residual"]) for r in table1)
    max_se = max(r["se"] for r in table1)
    h1_pass = bool(max_resid <= 2.0 * max_se)

    # --- 琛?2锛欻3 鍒嗗瓙鏄惁琚?delta 浼ゅ埌 ------------------------------------
    print("\n=== H3锛歞elta 鏄惁棰濆浼や簡鍒嗗瓙锛坋ta_survive锛夛紵 ===")
    print(f"{'delta':>9} {'P_D(fix)':>9} {'P_D(own)':>9} {'diff':>8} {'SE':>6}")
    table2 = []
    for d in args.deltas:
        a = mean(d, "fixed", "p_d")
        b = mean(d, "own", "p_d")
        se = se_paired(d, "own", "fixed")
        table2.append({"delta_bins": float(d), "p_d_fixed": a, "p_d_own": b,
                       "diff": b - a, "se": se})
        print(f"{d:9.4g} {a:9.4f} {b:9.4f} {b - a:+8.4f} {se:6.4f}")
    h3_pass = bool(all(abs(r["diff"]) <= 2.0 * max(r["se"], 1e-9) for r in table2))

    # --- 琛?3锛欻2 kappa(delta) 瑙ｆ瀽寰?-------------------------------------
    print("\n=== H2锛歬appa(delta) 鏄惁鏈嶄粠 -10log10(A*delta^2 + floor)锛?===")
    ks = [r["kappa_db"] for r in table1]
    ds = [r["delta_bins"] for r in table1]
    # 鐢?delta=0 妗ｅ畾鍦版澘锛岀敤鏈€澶?delta 妗ｅ畾绯绘暟 A
    floor = 10.0 ** (-ks[0] / 10.0) if ks[0] > 0 else 0.0
    i_max = int(np.argmax(ds))
    a_coef = ((10.0 ** (-ks[i_max] / 10.0)) - floor) / (ds[i_max] ** 2) \
        if ds[i_max] > 0 else float("nan")
    print(f"  floor = {floor:.4e}  (kappa_max = {ks[0]:.2f} dB)")
    print(f"  A = {a_coef:.4f}  (鐢?delta={ds[i_max]:g} 鏍囧畾)")
    print(f"{'delta':>9} {'kappa':>7} {'fit':>7} {'err_dB':>7}")
    fit_err = []
    for d, k in zip(ds, ks):
        fit = -10.0 * math.log10(a_coef * d * d + floor) if (a_coef * d * d + floor) > 0 \
            else float("nan")
        fit_err.append(abs(fit - k))
        print(f"{d:9.4g} {k:7.2f} {fit:7.2f} {fit - k:+7.2f}")
    h2_pass = bool(max(fit_err) <= 1.5)

    # 杈炬爣鎵€闇€鐨?delta
    req_frac = 10.0 ** (-GAMMA_REQ_DB / 10.0)
    if a_coef > 0 and req_frac > floor:
        delta_req = math.sqrt((req_frac - floor) / a_coef)
    else:
        delta_req = float("nan")

    # --- 琛?4锛氭妸"宸窛"缈昏瘧鎴?delta ---------------------------------------
    print("\n=== 宸窛褰掑洜锛氱湡瀹炵増 vs 妯″瀷鐗?===")
    pd0 = table1[0]["p_d"]
    k0 = table1[0]["kappa_db"]
    for r in table1[1:]:
        gap = pd0 - r["p_d"]
        # 鐢ㄥぉ鑺辨澘鏇茬嚎鎶?gap 鎷嗘垚 kappa 娈?        explained = ceiling_pd(k0) - ceiling_pd(r["kappa_db"])
        print(f"  delta={r['delta_bins']:.4g}: kappa {k0:.2f}->{r['kappa_db']:.2f} dB "
              f"(-{k0 - r['kappa_db']:.2f}), P_D {pd0:.4f}->{r['p_d']:.4f} "
              f"(gap {gap:+.4f}, 鐢?kappa 瑙ｉ噴 {explained:+.4f})")

    summary = {
        "H1_gap_explained_by_kappa": {
            "max_abs_residual": max_resid,
            "max_se": max_se,
            "pass": h1_pass,
            "note": "娈嬪樊 = P_D 瀹炴祴 - 澶╄姳鏉挎洸绾挎彃鍊硷紱pass 琛ㄧず鏃?unexplained gap",
        },
        "H2_kappa_of_delta_law": {
            "floor": floor,
            "A": a_coef,
            "max_fit_err_db": float(max(fit_err)),
            "pass": h2_pass,
            "delta_meeting_gamma_req_bins": delta_req,
        },
        "H3_delta_hurts_numerator": {
            "pass": h3_pass,
            "note": "pass = 鍒嗗瓙鍥哄畾涓庡垎瀛愰殢鑷傛棤鏄捐憲宸紓锛宒elta 鍙姩鍒嗘瘝",
        },
        "ceiling_curve": {
            "kappa_db": CEILING_KAPPA, "p_d": CEILING_PD,
            "perfect_p_d": CEILING_PERFECT_PD,
            "source": "tools/diag_direction1_ceiling.py (n=20)",
        },
        "table1": table1, "table2": table2,
        "config": {"preset": args.preset, "area": args.area, "rcs": args.rcs,
                   "seed": args.seed, "trials": int(args.trials),
                   "deltas": [float(d) for d in args.deltas]},
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    print("\n--- 鍒ゅ喅 ---")
    for k in ("H1_gap_explained_by_kappa", "H2_kappa_of_delta_law",
              "H3_delta_hurts_numerator"):
        print(f"  {k}: {'PASS' if summary[k]['pass'] else 'FAIL'}")
    print(f"  杈炬爣 {GAMMA_REQ_DB} dB 鎵€闇€ delta <= {delta_req:.4g} bins")
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
