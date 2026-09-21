"""方向 2 的**生产路径**定论扫描：robust 门控 vs 放宽链路上限。

与 ``diag_fusion2_belief.py``（``select_lagrangian`` + 单套粗表）不同，这里走
``run_one_trial`` —— 生产路径：belief 双世界、证书作用域、C2F 细表、融合节点
分配、弱目标定义全部在位。因此 ``baseline`` 的绝对 P_D 与发布数字同口径，
配对 ΔP_D 可直接迁移。

配对方式：公共随机数（``rng = default_rng([seed, trial])``），同一 trial 的
每个臂共享几何与 belief 实现，故 Δ 用逐 trial 差的标准误。

代价维度（判断 robust vs links12 的关键）：上报比特、链路数、上报时延。

Run::
    python studies/direction2/scripts/run_robust_verdict.py --trials 2
    python studies/direction2/scripts/run_robust_verdict.py --trials 120
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from experiments.flow.simulate import run_one_trial  # noqa: E402

METHOD = "proposed_c2f"

# 主链路默认 ``production_wire=False`` ⇒ 链路表里 ``kappa_dc = 1``（**没有直连
# 对消**）⇒ 600 m / RCS 0.1 下 D_fuse ~ 1e-6、P_D = 0。发布数字不是在这个世界
# 里出的，所以定论扫描必须显式打开证书（实测 κ），否则所有臂一起躺在地板上、
# 差值恒为 0（第一轮冒烟就踩了这个坑：baseline/robust/links12 全 0.0000）。
WIRE_OVERRIDES = {"cancellation.production_wire": True}

ARMS: dict[str, dict] = {
    "baseline": {},
    "robust": {"prior.robust_geometry_for_scheduler": True},
    "links12": {"selector.max_links_per_target": 12},
    "robust_links12": {"prior.robust_geometry_for_scheduler": True,
                       "selector.max_links_per_target": 12},
}

KEYS = [
    "p_d", "p_fa", "links", "overhead_bits", "overhead_delay_s",
    "weak_detected", "d_fuse_median", "pd_per_mbit", "kappa_median_db",
]


def build_config(args, **extra) -> Config:
    cfg = apply_preset(Config(), args.preset)
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": float(args.area),
        "detect.target_rcs": float(args.rcs),
        "run.seed": int(args.seed),
        "run.verbose": False,
    })
    if args.wire:
        cfg = apply_overrides(cfg, WIRE_OVERRIDES)
    # TP-UIC 的两个口径键是**全局**的：两版对比必须是同一个接收机，臂只能覆盖
    # selector / prior。默认值（measured / 0 / 0）与发布配置同值 ⇒ 不写也会一样，
    # 这里显式写出来是为了让 summary.json 自证跑的是哪一版。
    cfg = apply_overrides(cfg, {
        "cancellation.residual_accounting": str(args.accounting),
        "cancellation.direct_estimation_sigma_delay_bins": float(args.delta),
        "cancellation.direct_estimation_sigma_doppler_bins":
            float(args.delta_doppler),
    })
    return apply_overrides(cfg, extra) if extra else cfg


def install_certificate_cache() -> dict:
    """Share one receiver measurement per trial across all arms.

    The measurement depends only on ``(seed, trial, arm, retention-mode)`` —
    geometry, belief and the truth base are arm-independent — so caching it
    reproduces *exactly* what production does (one measurement per trial,
    shared by every method) while removing a 4x cost.  It never changes a
    number: the cached certificate is the one the first arm would have built.

    Returns the ``{(seed, trial): certificate}`` map so the caller can report
    the measured ``kappa`` (it is not on ``MethodResult``).
    """
    import experiments.flow.simulate as sim

    original = sim._measure_trial_certificate
    cache: dict = {}
    seen: dict = {}

    def cached(cfg, geom, geom_belief, base_truth, index):
        key = (
            int(cfg.run.seed), int(index),
            str(cfg.cancellation.production_wire_arm),
            str(cfg.cancellation.production_wire_retention),
            # 证书 = f(几何, belief, truth base, 接收机口径)。口径键一并入 key，
            # 免得哪天在同一个进程里切 δ 却复用了旧证书（静默错）。
            str(cfg.cancellation.residual_accounting),
            float(cfg.cancellation.direct_estimation_sigma_delay_bins),
            float(cfg.cancellation.direct_estimation_sigma_doppler_bins),
        )
        if key not in cache:
            cache[key] = original(cfg, geom, geom_belief, base_truth, index)
        seen[(int(cfg.run.seed), int(index))] = cache[key]
        return cache[key]

    sim._measure_trial_certificate = cached
    return seen


def kappa_of(cert) -> float:
    """Measured cancellation depth of a certificate, in dB (median over RX)."""
    if cert is None:
        return float("nan")
    frac = np.asarray(getattr(cert, "fraction", ()), dtype=float)
    frac = frac[np.isfinite(frac) & (frac > 0.0)]
    if frac.size == 0:
        return float("nan")
    return float(-10.0 * np.log10(float(np.median(frac))))


def keys_of(res) -> dict:
    links = int(sum(len(v) for v in res.selected_links.values()))
    bits = float(res.overhead_bits)
    return {
        "p_d": int(res.detected) / max(int(res.total_targets), 1),
        "p_fa": int(res.false_alarm) / max(int(res.total_false), 1),
        "links": links,
        "overhead_bits": bits,
        "overhead_delay_s": float(res.overhead_delay_s),
        "weak_detected": int(res.weak_target_detected),
        "d_fuse_median": float(np.median(np.asarray(res.D_fuse_per_target, dtype=float))),
        "pd_per_mbit": (int(res.detected) / max(int(res.total_targets), 1))
        / max(bits / 1.0e6, 1e-12),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=10917)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--method", default=METHOD)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--wire", type=int, default=1,
                    help="1 = open cancellation.production_wire (measured kappa)")
    ap.add_argument("--accounting", default="measured",
                    choices=("measured", "structural"),
                    help="cancellation.residual_accounting。'measured' = 发布默认，"
                         "δ 被噪声增强项淹没（99.8%）⇒ 看不见；'structural' = "
                         "只记 ||x-f(x)||^2 ⇒ δ 才进入 kappa。两版数字必须用后者。")
    ap.add_argument("--delta", type=float, default=0.0,
                    help="cancellation.direct_estimation_sigma_delay_bins（DD 格）。"
                         "0 = 完美字典 = 模型版上界（不可达）")
    ap.add_argument("--delta-doppler", type=float, default=0.0,
                    help="cancellation.direct_estimation_sigma_doppler_bins")
    ap.add_argument("--share-cert", type=int, default=1,
                    help="1 = one receiver measurement per trial, shared by all "
                         "arms (arms differ only in selector/prior keys, so the "
                         "measurement is identical). 0 = measure per arm.")
    ap.add_argument("--out", default="studies/direction2/data/robust_verdict")
    args = ap.parse_args(argv)
    certs: dict = {}
    if args.wire and args.share_cert:
        certs = install_certificate_cache()

    names = [n for n in args.arms.split(",") if n]
    unknown = [n for n in names if n not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms: {unknown}")
    cfgs = {n: build_config(args, **ARMS[n]) for n in names}
    os.makedirs(args.out, exist_ok=True)

    rows: list[dict] = []
    path = os.path.join(args.out, "verdict.csv")

    def flush() -> None:
        """Incremental dump: a 2-hour run must be readable before it ends."""
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["trial", "arm"] + KEYS)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    t0 = time.time()
    for trial in range(int(args.trials)):
        for name in names:
            res = run_one_trial(cfgs[name], trial, methods=[args.method])[args.method]
            row = {"trial": trial, "arm": name}
            row.update(keys_of(res))
            row["kappa_median_db"] = kappa_of(
                certs.get((int(args.seed), trial))
            )
            rows.append(row)
        flush()
        done = trial + 1
        if done % 5 == 0 or done == int(args.trials):
            el = time.time() - t0
            print(f"  trial {done}/{args.trials}  {el:.1f}s "
                  f"({el / done * (args.trials - done):.0f}s left)", flush=True)

    path = os.path.join(args.out, "verdict.csv")
    flush()

    def col(arm, key):
        return np.array([r[key] for r in rows if r["arm"] == arm], dtype=float)

    base = col("baseline", "p_d")
    base_fa = float(np.mean(col("baseline", "p_fa")))
    summary = {
        "config": {
            "preset": args.preset, "area": args.area, "rcs": args.rcs,
            "seed": args.seed, "trials": int(args.trials), "method": args.method,
            "path": "run_one_trial (production, belief two-world + C2F)",
            "production_wire": bool(args.wire),
            "tpuic_accounting": str(args.accounting),
            "tpuic_delta_delay_bins": float(args.delta),
            "tpuic_delta_doppler_bins": float(args.delta_doppler),
        },
        "arms": {},
    }
    print()
    print("%-16s %8s %8s %7s %10s %9s %9s %9s"
          % ("arm", "P_D", "P_FA", "links", "kbit", "kappa_dB", "dP_D", "se"))
    for name in names:
        pd_ = float(np.mean(col(name, "p_d")))
        d = col(name, "p_d") - base
        n = d.size
        se = float(np.std(d, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        summary["arms"][name] = {
            "p_d": pd_,
            "p_fa": float(np.mean(col(name, "p_fa"))),
            "delta_p_fa": float(np.mean(col(name, "p_fa"))) - base_fa,
            "links": float(np.mean(col(name, "links"))),
            "overhead_kbit": float(np.mean(col(name, "overhead_bits"))) / 1.0e3,
            "weak_detected": float(np.mean(col(name, "weak_detected"))),
            "kappa_median_db": float(np.nanmean(col(name, "kappa_median_db"))),
            "delta_p_d": pd_ - float(np.mean(base)),
            "se": se,
            "z": (pd_ - float(np.mean(base))) / se if se > 0 else float("nan"),
            "resolvable": bool(abs(pd_ - float(np.mean(base))) >= 2.0 * se),
            "pd_per_mbit": float(np.mean(col(name, "pd_per_mbit"))),
        }
        s = summary["arms"][name]
        print("%-16s %8.4f %8.4f %7.2f %10.2f %9.2f %+9.4f %9.4f"
              % (name, pd_, s["p_fa"], s["links"], s["overhead_kbit"],
                 s["kappa_median_db"], s["delta_p_d"], se))

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nwrote {path} and summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
