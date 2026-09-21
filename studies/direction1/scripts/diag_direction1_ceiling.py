"""鏂瑰悜 1 澶辫触鏍瑰洜 Part B锛氭柟鍚?1 鐨?*澶╄姳鏉?*鏈夊楂樸€?
涓轰粈涔堝繀椤昏窇
------------
"鏂瑰悜 1 澶辫触浜?鍏跺疄鏄袱涓畬鍏ㄤ笉鍚岀殑澶辫触锛屽繀椤诲垎寮€锛屽惁鍒欎細寮€閿欒嵂鏂癸細

* **瀵硅薄閿?*锛氭繁搴︽牴鏈笉鏄摱棰?鈥斺€?鍗充究鎶?kappa 鎶埌鏃犵┓锛孭_D 涔熷嚑涔庝笉鍔ㄣ€?  鈬?璇ュ仛鐨勬槸鎹㈡柟鍚戯紙绔嬮」鏃跺氨璇ョ敤鏃㈡湁鏍瑰洜瀹¤璇佷吉锛夈€?* **鏂规硶/瀹炵幇閿?*锛氭繁搴︾‘瀹炴槸鐡堕锛屾姮 kappa 鑳芥嬁寰堝锛屼絾褰撳墠瀹炵幇鎷夸笉鍒般€?  鈬?璇ュ仛鐨勬槸淇疄鐜帮紙渚嬪 ``n_cpi`` 鏈帴绾匡級銆?
杩欎袱涓殑鑽柟鐩稿弽锛岄潬鎺ㄧ悊鍒嗕笉寮€锛屽彧鑳芥妸 P_D 褰撴垚 kappa 鐨勫嚱鏁板疄娴嬪嚭鏉ャ€?
鍋氭硶
----
鍥哄畾鍑犱綍銆佸浐瀹氳皟搴﹀櫒銆佸浐瀹氭娴嬪櫒銆?*鍥哄畾鍥炴尝瀛樻椿鐜?*锛堝垎瀛愶級锛屽彧鎶婃敞鍏ョ粰
閾捐矾琛ㄧ殑娈嬩綑姣斾緥 ``fraction = 10^(-kappa/10)`` 褰撹嚜鍙橀噺鎵€?鈬?寰楀埌鐨勬槸"绾补闈犳敼鍠勫娑堟繁搴﹁兘鎷垮埌鐨勫叏閮ㄦ敹鐩?锛屽嵆鏂瑰悜 1 鐨勫ぉ鑺辨澘銆?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?------------------------------
* **F1 杈炬爣妗?*锛歚`P_D(52.25) - P_D(36.5)``銆?2.25 dB 鏄?KAPPA_DERIVATION.md
  缁欏嚭鐨勫嚑浣曢渶姹傦紝36.5 dB 鏄綋鍓嶅疄娴嬨€傗墺 +0.02 鈬?杈炬爣鏈韩鏈夋敹鐩娿€?* **F2 缁濆澶╄姳鏉?*锛歚`P_D(perfect) - P_D(36.5)``锛坧erfect = fraction 0锛夈€?  鈮?+0.05 鈬?娣卞害鏄湡鐡堕锛堟柟娉?瀹炵幇閿欙級锛? +0.02 鈬?娣卞害涓嶆槸鐡堕锛堝璞￠敊锛夈€?* **F3 楗卞拰鐐?*锛歅_D 杈惧埌澶╄姳鏉跨殑 90% 鎵€闇€鐨?kappa銆傝嫢 鈮?52.25 鈬?闇€姹傛。浣?  瀹氬緱鍚堢悊锛涜嫢杩滈珮浜庡畠 鈬?闇€姹傛。浣嶆槸杩囧害璁捐銆?* **F4 璇氬疄鎬?*锛氬悇妗?P_FA 涓?open_loop 鐩稿樊涓嶈秴杩?0.01銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/diag_direction1_ceiling.py --trials 20
"""

from __future__ import annotations

import argparse
import csv
import json
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

# 褰撳墠瀹炴祴娣卞害锛圓UDIT_DIRECTION1_DELTA.md 搂2锛宬appa_total 涓綅锛?KAPPA_MEASURED_DB = 36.5
# 鍑犱綍闇€姹傦紙KAPPA_DERIVATION.md锛?00 m / RCS 0.1锛?KAPPA_REQUIRED_DB = 52.25

DEFAULT_LEVELS = [0.0, 20.0, 36.5, 45.0, 52.25, 60.0, 68.0, 90.0]


def build_config(args) -> Config:
    cfg = apply_preset(Config(), args.preset)
    return apply_overrides(cfg, {
        "geometry.area_xy": float(args.area),
        "detect.target_rcs": float(args.rcs),
        "scale.M": int(args.m),
        "scale.Q": int(args.q),
        "run.seed": int(args.seed),
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.n_cpi": int(args.n_cpi),
        "cancellation.max_protected_targets": int(args.max_protected_targets),
        "aperture.enable": True,
        "aperture.m_rx": int(args.m_rx),
    })


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
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--levels", type=float, nargs="*", default=DEFAULT_LEVELS)
    ap.add_argument("--out", default="studies/direction1/data/diag_direction1_ceiling")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([cfg.run.seed, int(trial)])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        mrng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(trial)])
        measured = cx.measure_receiver_context(
            cx.ReceiverContext.from_trial(cfg, geom, geom, base, arm=args.arm),
            rng=mrng,
        )
        # 鍒嗗瓙鍥哄畾鍦ㄥ疄娴嬪瓨娲荤巼涓婏細鏈疄楠屽彧鎯冲姩鍒嗘瘝锛堟繁搴︼級锛屼笉鑳借鍒嗗瓙婕傜Щ銆?        retention = np.clip(np.asarray(measured.eta_survive, dtype=float), 0.0, 1.0)
        M = int(np.asarray(measured.fraction).size)

        for lvl in args.levels:
            if float(lvl) >= 90.0:
                frac = np.zeros(M, dtype=float)      # 瀹岀編瀵规秷
                label = "perfect"
            else:
                frac = np.full(M, 10.0 ** (-float(lvl) / 10.0), dtype=float)
                label = "%.2f" % float(lvl)
            tables = compute_link_tables(
                cfg, base,
                residual_fraction_by_receiver=frac,
                target_retention_by_receiver=retention,
            )
            selected = _selector(cfg, base, tables)
            det_rng = np.random.default_rng([cfg.run.seed, 3 * 10 ** 6 + int(trial)])
            det = _detection(cfg, tables, selected, det_rng, base, int(trial))
            rinr = np.asarray(tables.rinr, dtype=float)
            rinr = rinr[np.isfinite(rinr) & (rinr > 0.0)]
            rows.append({
                "trial": int(trial),
                "kappa_db": float(lvl),
                "label": label,
                "p_d": det["p_d"],
                "p_fa": det["p_fa"],
                "n_selected": det["n_selected"],
                "rinr_db_median": float(np.median(10.0 * np.log10(rinr))) if rinr.size else float("nan"),
            })
        print("  trial %d/%d [%.0f s]" % (trial + 1, args.trials, time.time() - t0),
              flush=True)

    keys = list(rows[0])
    path = os.path.join(args.out, "ceiling.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    labels = sorted({r["label"] for r in rows},
                    key=lambda s: (float("inf") if s == "perfect" else float(s)))

    def col(label, key):
        return np.array([r[key] for r in rows if r["label"] == label], dtype=float)

    print("\n=== P_D(kappa) 鈥斺€?鏂瑰悜 1 鐨勫ぉ鑺辨澘 ===")
    print(f"{'kappa[dB]':>10} {'P_D':>8} {'P_FA':>8} {'links':>7} {'rinr[dB]':>9} {'dP_D':>8}")
    base_pd = float(np.mean(col("%.2f" % KAPPA_MEASURED_DB, "p_d")))
    summary = {"levels": {}}
    for lb in labels:
        pd_ = float(np.mean(col(lb, "p_d")))
        summary["levels"][lb] = {
            "p_d": pd_,
            "p_fa": float(np.mean(col(lb, "p_fa"))),
            "n_selected": float(np.mean(col(lb, "n_selected"))),
            "rinr_db_median": float(np.median(col(lb, "rinr_db_median"))),
            "delta_p_d_vs_measured": pd_ - base_pd,
        }
        print(f"{lb:>10} {pd_:8.4f} {summary['levels'][lb]['p_fa']:8.4f} "
              f"{summary['levels'][lb]['n_selected']:7.2f} "
              f"{summary['levels'][lb]['rinr_db_median']:9.2f} {pd_ - base_pd:+8.4f}")

    def at(lvl):
        key = "%.2f" % float(lvl)
        return summary["levels"].get(key, {}).get("p_d", float("nan"))

    pd_req = at(KAPPA_REQUIRED_DB)
    pd_perf = summary["levels"].get("perfect", {}).get("p_d", float("nan"))
    # F3 楗卞拰鐐癸細杈惧埌 (measured -> perfect) 璺ㄥ害 90% 鐨勬渶浣庢。
    span = pd_perf - base_pd
    sat = None
    for lb in labels:
        if lb == "perfect":
            continue
        if np.isfinite(span) and span > 0 and summary["levels"][lb]["p_d"] - base_pd >= 0.9 * span:
            sat = float(lb)
            break
    summary["F1_required_vs_measured"] = {
        "delta_p_d": pd_req - base_pd, "pass": bool(pd_req - base_pd >= 0.02)}
    summary["F2_perfect_vs_measured"] = {
        "delta_p_d": pd_perf - base_pd, "pass": bool(pd_perf - base_pd >= 0.05)}
    summary["F3_saturation_kappa_db"] = sat
    summary["F4_honesty_p_fa_spread"] = {
        "max": float(max(v["p_fa"] for v in summary["levels"].values())),
        "min": float(min(v["p_fa"] for v in summary["levels"].values())),
        "pass": bool(max(v["p_fa"] for v in summary["levels"].values())
                     - min(v["p_fa"] for v in summary["levels"].values()) <= 0.01)}
    summary["config"] = {
        "preset": args.preset, "area": args.area, "rcs": args.rcs,
        "seed": args.seed, "trials": int(args.trials), "arm": args.arm,
        "kappa_measured_db": KAPPA_MEASURED_DB, "kappa_required_db": KAPPA_REQUIRED_DB,
    }
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print("\n" + json.dumps({k: v for k, v in summary.items() if k.startswith("F")},
                            indent=2, sort_keys=True))
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
