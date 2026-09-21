"""鎺㈤拡锛氭畫浣欏共鎵板湪閾捐矾琛ㄥ垎姣嶉噷鍒板簳鍗犲澶с€?
鍒ゅ喅鑴氭湰 :mod:`tools.verify_tpuic_residual_identity` 宸茬粡璇佹槑 ``||f(n)||^2`` 涓嶅湪
娈嬪樊閲岋紙绾挎€у垎瑙ｈ宸?1e-29锛屾畫宸櫔澹?= 鍘熷櫔澹扮殑 99.97%锛夈€備絾瀹冨彧璇存槑**璇ヤ笉璇?*
璁拌处锛屾病鏈夎鏄?*鍊煎灏?dB**锛氶偅鍙栧喅浜庨摼璺〃鍒嗘瘝閲?``n0`` 涓庢畫浣欏共鎵扮殑鐩稿澶у皬銆?
杩欓噷鎶婁袱浠朵簨涓€娆￠噺娓呮锛?
* **R1 鍙傝€冮噺**锛歚`noise_power(cfg)``锛堥摼璺〃鐢ㄧ殑 ``n0``锛変笌瑙傛祴鏋勯€犵敤鐨?  ``_noise_power(cfg)``锛堥€?bin 鐨?``sigma2``锛夋槸鍚︽槸鍚屼竴涓暟銆傝嫢涓嶆槸锛?  涓や釜涓栫晫鐨勫櫔澹板彛寰勫氨涓嶄竴鑷达紝浠讳綍"娈嬩綑浣庝簬鍣０"鐨勫垽鏂兘涓嶈兘璺ㄥ彛寰勮銆?* **R2 鍗犳瘮**锛氭敞鍏ュ疄娴嬫畫浣欏悗锛宍`rinr = residual_total / (n0 + eps)`` 鐨勫垎浣嶆暟銆?  杩欏氨鏄?娈嬩綑骞叉壈姣斿櫔澹板湴鏉块珮澶氬皯 dB"銆?* **R3 淇骞呭害**锛氭妸 ``kappa`` 鎹㈡垚鍙惈缁撴瀯娈嬪樊鐨勭増鏈?  锛坄`i_res_structural / i_in``锛夛紝``rinr`` 闄嶅灏?dB 鈥斺€?鍗崇籂姝ｈ璐﹀€煎灏戙€?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙級
----------------
* **V1**锛氳嫢 ``R2`` 涓綅 **< -6 dB**锛堟畫浣欏共鎵颁綆浜庡櫔澹板湴鏉?6 dB 浠ヤ笂锛夆噿
  娈嬩綑骞叉壈**涓嶆槸**鐡堕锛屾柟鍚?1 鐨?缁х画娣辨寲娑堥櫎娣卞害"娌℃湁绌洪棿锛?  "15 dB 娣卞害缂哄彛"鏄吉闂锛堣姹傜殑鏄竴涓凡缁忓湪鍦版澘浠ヤ笅鐨勯噺锛夈€?* **V2**锛氳嫢 ``R2`` 涓綅 **> +3 dB** 鈬?娈嬩綑骞叉壈鏄摱棰堬紝绾犳璁拌处鑳芥崲鐪熼噾鐧介摱鐨?  SINR锛屾柟鍚?1 搴斿厛鍋氳璐︿慨姝ｃ€?* 钀藉湪涓棿 鈬?涓や欢浜嬮兘涓嶄紭鍏堬紝鍏堝仛鏂瑰悜 2/3銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/probe_noise_reference.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.receiver.cancellation.protection import _noise_power  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.sensing.model.mathkit import bandwidth, noise_power  # noqa: E402


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


def _q(v, p):
    return float(np.quantile(np.asarray(v, dtype=float), p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--out", default="studies/direction1/data/probe_noise_reference")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)

    n0 = float(noise_power(cfg))
    sigma2 = float(_noise_power(cfg))
    ref = {
        "n0_link_table": n0,
        "sigma2_per_bin": sigma2,
        "n0_over_sigma2": n0 / sigma2,
        "bandwidth_hz": float(bandwidth(cfg)),
        "waveform_N": int(cfg.waveform.N),
        "waveform_L": int(cfg.waveform.L),
        "delta_f": float(cfg.waveform.delta_f),
        "m_rx": int(cfg.aperture.m_rx),
        "g_proc": float(cfg.waveform.N * cfg.waveform.L),
    }
    print("reference:", json.dumps(ref, indent=2, sort_keys=True))

    rows = []
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([cfg.run.seed, int(trial)])
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        mrng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(trial)])
        measured = cx.measure_receiver_context(
            cx.ReceiverContext.from_trial(cfg, geom, geom, base, arm=args.arm),
            rng=mrng,
        )
        frac_old = np.asarray(measured.fraction, dtype=float)
        i_res = np.asarray(measured.i_res, dtype=float)
        i_est = np.asarray(measured.i_res_estimate, dtype=float)
        i_in = np.asarray(measured.i_in, dtype=float)
        structural = np.maximum(i_res - i_est, 0.0)
        with np.errstate(divide="ignore"):
            frac_new = np.where(i_in > 0.0, structural / np.maximum(i_in, 1e-300), 1.0)

        for label, frac in (("measured", frac_old), ("structural_only", frac_new)):
            tables = compute_link_tables(cfg, base, residual_fraction_by_receiver=frac)
            rinr = np.asarray(tables.rinr, dtype=float)
            rinr = rinr[np.isfinite(rinr) & (rinr > 0.0)]
            rows.append({
                "trial": int(trial),
                "label": label,
                "rinr_db_p10": _q(10.0 * np.log10(rinr), 0.10),
                "rinr_db_p50": _q(10.0 * np.log10(rinr), 0.50),
                "rinr_db_p90": _q(10.0 * np.log10(rinr), 0.90),
                "residual_over_n0_p50": _q(rinr, 0.50),
            })
        rows.append({
            "trial": int(trial),
            "label": "kappa_db",
            "rinr_db_p10": float(np.nan),
            "rinr_db_p50": float(np.median(measured.kappa_db)),
            "rinr_db_p90": float(np.nan),
            "residual_over_n0_p50": float(np.nan),
        })

    summary = {"reference": ref, "rows": rows}
    measured_p50 = [r for r in rows if r["label"] == "measured"]
    med_rinr_db = float(np.median([r["rinr_db_p50"] for r in measured_p50]))
    new_p50 = [r for r in rows if r["label"] == "structural_only"]
    med_new_db = float(np.median([r["rinr_db_p50"] for r in new_p50]))
    verdict = {
        "R2_rinr_db_median_measured": med_rinr_db,
        "R3_rinr_db_median_structural_only": med_new_db,
        "R3_correction_gain_db": med_rinr_db - med_new_db,
        "V1_residual_below_floor": bool(med_rinr_db < -6.0),
        "V2_residual_is_bottleneck": bool(med_rinr_db > 3.0),
    }
    summary["verdict"] = verdict
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
