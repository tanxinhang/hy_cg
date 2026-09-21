"""鏂瑰悜 1 鎵弿锛氱洿杩炲弬鏁颁及璁¤宸?delta 濡備綍鍚冩帀瀵规秷娣卞害銆?
涓轰粈涔堝繀椤昏窇
------------
``build_observation`` 姝ゅ墠鐢?*鐪熷€?*鍒嗘暟 DD bin 閫犲共鎵板瓧鍏革紝浜庢槸 ``span(X)``
瀹屾暣鍖呭惈鐩磋繛鍦猴紝``structural = ||x - f(x)||^2`` 琚瀯閫犲帇鎴?~0銆傞偅涓?~67 dB
鐨?缁撴瀯娣卞害"鏄?*瀛楀吀瀹屽**鐨勪骇鐗╋紝涓嶆槸绠楁硶鍔熺哗 鈥斺€?off-grid 姝ｄ氦娉ㄥ叆鎺㈤拡
锛坄`tools/probe_offgrid_depth.py``锛夊凡缁忚瘉鏄庡畠鎸?``-20log10(eps)`` 宕╁銆?
鏈剼鏈敤**鏂伴棬鎺?* ``cancellation.direct_estimation_sigma_*_bins`` 缁?delta
瀹氫环鏍硷細瀛楀吀鐢?鎺ユ敹鏈轰互涓虹殑" bin銆佺洿杩炲満浠嶇敤鐪熷€?bin锛屼簬鏄瓧鍏稿鑳介噺鐢?delta 鍐冲畾锛屾繁搴︾敱鏁版嵁鍐冲畾銆?
鍚屾椂鎵?``interference_tangent_order``锛堝垏鍚戝垪锛夌湅鑳戒笉鑳芥妸娣卞害涔板洖鏉?鈥斺€?鍒囧悜
鍒楁湰鏉ュ氨鏄负"鍒嗘暟鍋忕Щ"璁剧殑锛岃繖鏄柟鍚?1 鍞竴鍙兘鏈夊疄鐜版敮鎾戠殑鐪熸潬鏉嗐€?鈿狅笍 鍙ｅ緞绾緥锛氬繀椤荤敤**鍒嗘暟寤惰繜璇樊**娉ㄥ叆锛堟湰鑴氭湰锛夛紝涓嶈兘鐢ㄦ浜ゆ敞鍏?鈥斺€?姝ｄ氦娉ㄥ叆浼氱郴缁熸€т綆浼板垏鍚戝垪鐨勪綔鐢紙鍘熷璁″凡鎵胯杩欎竴鐐癸級銆?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?------------------------------
* **S1 鐞嗚涓€鑷存€?*锛氬瓧鍏稿鑳介噺鍒嗘暟鏄惁 鈮?``(pi^2/3) * delta^2``锛堝皬 delta
  鏋侀檺锛夈€傛垚绔?鈬?鏂伴棬鎺у缓妯＄殑鏄?鍒嗘暟鍋忕Щ"锛屼笌鐞嗚鍚屼竴涓璞°€?* **S2 杈炬爣绮惧害**锛氱粨鏋勬€ф繁搴﹁穼鐮撮渶姹?``52.25 dB`` 鐨?delta 闂ㄦ鏄灏戞牸锛?  鎹㈢畻鎴愮背 / Hz銆?* **S3 鍒囧悜鍒楁晳鎻?*锛氬湪 delta > 0 鏃讹紝``interference_tangent_order = 1``
  鐩稿 0 鑳芥姮鍥炲灏?dB銆傛姮鍥?>= 10 dB 鈬?杩欐槸鐪熸潬鏉嗭紱< 3 dB 鈬?涔颁簡娌＄敤銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/sweep_direct_estimation_delta.py --seeds 6
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

# 600 m / RCS 0.1 涓诲伐浣滅偣涓婏紝鍑犱綍瑕佹眰鐨勭洿杩炲娑堟繁搴︼紙KAPPA_DERIVATION.md锛夈€?KAPPA_REQUIRED_DB = 52.25

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


def out_of_span_fraction(obs) -> float:
    """鐩磋繛鍦洪噷钀藉湪 ``span(X)`` 涔嬪鐨勮兘閲忔瘮渚?鈥斺€?灏辨槸"娑堜笉鎺夌殑骞叉壈"銆?""
    X = np.asarray(obs.X, dtype=complex)
    x = np.asarray(obs.x_direct, dtype=complex)
    if X.shape[1] == 0:
        return 0.0
    gram = X.conj().T @ X
    coef = np.linalg.pinv(gram, hermitian=True) @ (X.conj().T @ x)
    r = x - X @ coef
    total = float(np.vdot(x, x).real)
    if total <= 0.0:
        return 0.0
    return float(np.vdot(r, r).real) / total


def grid_units(cfg):
    """1 涓欢杩熸牸 / 澶氭櫘鍕掓牸鍒嗗埆鏄灏戠背銆佸灏?Hz銆?""
    w = cfg.waveform
    delay_bin_s = 1.0 / (float(w.L) * float(w.delta_f))
    doppler_bin_hz = 1.0 / (float(w.N) * float(w.T))
    return delay_bin_s * float(w.c), doppler_bin_hz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="*",
                    default=[0.0, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1])
    ap.add_argument("--orders", type=int, nargs="*", default=[0, 1])
    ap.add_argument("--steps", type=float, nargs="*", default=[0.05])
    ap.add_argument("--ncpis", type=int, nargs="*", default=[1])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--receivers", type=int, nargs="*", default=None)
    ap.add_argument("--out", default="studies/direction1/data/sweep_direct_estimation_delta")
    args = ap.parse_args()

    probe = make_cfg()
    m = int(probe.scale.M)
    receivers = args.receivers if args.receivers else list(range(m))
    delay_m, dop_hz = grid_units(probe)
    sense = np.full(m, probe.radio.rho * probe.radio.P_default)

    rows = []
    # geom 鍙緷璧?seed锛屾寜 seed 缂撳瓨锛沷bs 鐨?rng 鎸?(seed, rx) 鍥哄畾锛屼簬鏄笉绠?    # delta 鍙栧摢涓€妗ｏ紝鍑犱綍 / h_true / noise 閮介€愪綅鐩稿悓 鈥斺€?閰嶅骞插噣銆?    cache: dict = {}
    for seed in range(int(args.seeds)):
        rng_g = np.random.default_rng([2026, seed])
        geom = generate_geometry(probe, rng_g)
        base = build_base_gains(probe, geom, rng_g)
        cache[seed] = (geom, base)

    for n_cpi in args.ncpis:
        for order in args.orders:
            for step in args.steps:
                if int(order) == 0 and float(step) != float(args.steps[0]):
                    continue  # order=0 娌℃湁鍒囧悜鍒楋紝姝ラ暱瀵瑰畠娌℃湁鎰忎箟
                for sigma in args.sigmas:
                    cfg = make_cfg(
                        **{
                            "cancellation.direct_estimation_sigma_delay_bins": float(sigma),
                            "cancellation.interference_tangent_order": int(order),
                            "cancellation.tangent_step_bins": float(step),
                            "cancellation.n_cpi": int(n_cpi),
                        }
                    )
                    for seed in range(int(args.seeds)):
                        geom, base = cache[seed]
                        for rx in receivers:
                            rng = np.random.default_rng([2026, seed, int(rx)])
                            obs = cx.build_observation(
                                cfg, geom, geom, base, int(rx), rng=rng,
                                sense_power=sense, radiated_power=sense,
                                processing_gain=cfg.waveform.N * cfg.waveform.L,
                                hw_gain=1.0,
                            )
                            res = cx.cancellation_arms(
                                cfg, obs, only="tp_uic_full"
                            )["tp_uic_full"]
                            i_in = float(res.i_in)
                            st = float(res.i_res_structural)
                            rows.append({
                                "n_cpi": int(n_cpi),
                                "sigma_bins": float(sigma),
                                "order": int(order),
                                "step_bins": float(step),
                                "seed": int(seed),
                                "receiver": int(rx),
                                "off_span_frac": out_of_span_fraction(obs),
                                "kappa_db": (
                                    float(res.kappa_db)
                                    if math.isfinite(res.kappa_db) else None
                                ),
                                "kappa_struct_db": (
                                    10.0 * math.log10(i_in / st)
                                    if st > 0.0 and i_in > 0.0 else None
                                ),
                            })
                    print(
                        f"  n_cpi={n_cpi} order={order} step={step} "
                        f"sigma={sigma:g} -> {len(rows)} rows", flush=True
                    )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "delta_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump({"required_db": KAPPA_REQUIRED_DB,
                   "delay_bin_m": delay_m, "doppler_bin_hz": dop_hz,
                   "rows": rows}, fh, indent=2)

    def block(title: str, order: int, step: float) -> None:
        print(f"\n=== {title} ===")
        print(f"{'sigma[bin]':>11} {'delay[m]':>9} {'off_span':>10} {'theory':>10} "
              f"{'k_struct':>9} {'k_total':>8} {'verdict':>8}")
        for sigma in args.sigmas:
            sel = [r for r in rows if r["sigma_bins"] == sigma
                   and r["order"] == int(order) and r["step_bins"] == float(step)
                   and r["n_cpi"] == int(args.ncpis[0])]
            if not sel:
                continue
            ks = [r["kappa_struct_db"] for r in sel if r["kappa_struct_db"] is not None]
            kt = [r["kappa_db"] for r in sel if r["kappa_db"] is not None]
            ksm = float(np.median(ks)) if ks else float("nan")
            ktm = float(np.median(kt)) if kt else float("nan")
            off = float(np.median([r["off_span_frac"] for r in sel]))
            theory = (math.pi ** 2 / 3.0) * sigma ** 2 if sigma > 0 else 0.0
            # 杈炬爣鍒ゅ畾鐢?k_total锛氶摼璺〃閲岀缉鏀?residual_direct 鐨勬槸鐜拌璁拌处鍙ｅ緞銆?            verdict = "PASS" if ktm >= KAPPA_REQUIRED_DB else "FAIL"
            print(f"{sigma:11.4g} {sigma * delay_m:9.3f} {off:10.3e} {theory:10.3e} "
                  f"{ksm:9.2f} {ktm:8.2f} {verdict:>8}")

    block("鏃犲垏鍚戝垪锛坥rder=0锛?, 0, args.steps[0])
    for step in args.steps:
        block(f"鍒囧悜闃?1锛屾闀?{step:g} 鏍?, 1, step)

    print("\n=== 鍒囧悜鍒楁晳鎻达細k_struct 涓綅锛坉B锛?===")
    print(f"{'sigma[bin]':>11}" + "".join(f"{'step' + format(s, 'g'):>10}" for s in args.steps))
    for sigma in args.sigmas:
        n0 = int(args.ncpis[0])
        base0 = [r["kappa_struct_db"] for r in rows
                 if r["sigma_bins"] == sigma and r["order"] == 0
                 and r["n_cpi"] == n0 and r["kappa_struct_db"] is not None]
        m0 = float(np.median(base0)) if base0 else float("nan")
        cells = f"{sigma:11.4g}"
        for step in args.steps:
            s1 = [r["kappa_struct_db"] for r in rows
                  if r["sigma_bins"] == sigma and r["order"] == 1
                  and r["step_bins"] == float(step) and r["n_cpi"] == n0
                  and r["kappa_struct_db"] is not None]
            m1 = float(np.median(s1)) if s1 else float("nan")
            cells += f"{m1 - m0:>+10.2f}"
        print(cells)

    if len(args.ncpis) > 1:
        sig_ref = [0.0] + [s for s in args.sigmas if s > 0][:1]
        step_ref = float(args.steps[0])
        print("\n=== 鍙傝€冮绠?n_cpi vs 瀹炴祴娣卞害锛坘_total 涓綅锛宒B锛?===")
        head = f"{'n_cpi':>7}"
        for order in (0, 1):
            for sg in sig_ref:
                head += f"{'o' + str(order) + ',d=' + format(sg, 'g'):>14}"
        print(head)
        for n_cpi in args.ncpis:
            line = f"{int(n_cpi):7d}"
            for order in (0, 1):
                for sg in sig_ref:
                    sel = [r["kappa_db"] for r in rows
                           if r["n_cpi"] == int(n_cpi) and r["order"] == order
                           and r["sigma_bins"] == float(sg)
                           and r["step_bins"] == step_ref
                           and r["kappa_db"] is not None]
                    line += f"{float(np.median(sel)):14.2f}" if sel else f"{'--':>14}"
            print(line)

    print(f"\n闇€姹傛繁搴?= {KAPPA_REQUIRED_DB} dB锛? 寤惰繜鏍?= {delay_m:.2f} m锛?
          f"1 澶氭櫘鍕掓牸 = {dop_hz:.2f} Hz")
    print(f"鍐欏叆 {args.out}/delta_sweep.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
