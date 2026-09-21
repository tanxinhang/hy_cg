"""瀹¤鎺㈤拡锛歚`rinr`` 鍒板簳鐢卞摢涓€椤规拺璧锋潵銆?
涓轰粈涔堝繀椤昏窇
------------
瀹¤ ``TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md`` 鏃跺彂鐜颁袱涓?缁撴瀯娈嬪樊"璇绘暟宸?19.6 dB锛?
* :mod:`tools.verify_tpuic_residual_identity` 鐩存帴璇?  ``CancellationResult.i_res_structural`` 鈬?``kappa_structural`` 涓綅 **66.5 dB**銆?* :mod:`tools.probe_noise_reference` 璧?``compute_link_tables`` 鐨?``rinr``
  鈬?鍙嶆帹鍙湁 **49.1 dB**銆?
涓よ€呭樊 19.6 dB锛岃€?``rinr`` 鐨勫垎瀛愭槸涓夐」涔嬪拰::

    residual_total = residual_self + residual_direct + residual_multi
    rinr           = residual_total / n0

``residual_direct`` 鎵嶆槸琚娑堟繁搴?``kappa`` 缂╂斁鐨勯偅涓€椤癸紱``residual_self``
锛堣嚜骞叉壈娈嬩綑锛変笌 ``residual_multi``锛堝鏈烘劅鐭ユ硠婕忥級**涓庡娑堝櫒鏃犲叧**銆傝嫢瀹冧滑
鏋勬垚鍦版澘锛屽垯"鎶?``residual_direct`` 鍘嬪埌 0"涔嬪悗 ``rinr`` 鍙檷鍒板湴鏉夸负姝紝
姝ゆ椂鐢?``rinr`` 鍙嶆帹缁撴瀯娈嬪樊鏄?*閿欒鐨勬崲绠?*,閭?19.6 dB 灏辨槸杩欎箞鏉ョ殑銆?
鏈剼鏈笉鍋氭崲绠?鐩存帴鎶婁笁椤规憡寮€鎵撳嵃,璁╁湴鏉胯嚜宸辫璇濄€?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?------------------------------
* **D1 鍦版澘褰掑睘**锛歚`structural_only`` 鑷傜殑 ``rinr`` 涓綅鏄惁鐢?  ``residual_self + residual_multi`` 涓诲锛堝崰姣?> 80%锛夈€?  鎴愮珛 鈬?19.6 dB 鐨勫垎姝ф槸**鎹㈢畻閿欒**锛屼笉鏄暟鎹煕鐩撅紝鍘熷璁?搂1 鐨?  ``kappa_structural`` 璇绘暟鎴愮珛锛涗笉鎴愮珛 鈬?鍘熷璁″瓨鍦ㄧ湡瀹炵殑鏁版嵁鐭涚浘銆?* **D2 娣卞害澶╄姳鏉?*锛歚`residual_direct`` 鍦?``structural_only`` 鑷備笅鏄惁宸茬粡
  浣庝簬鍦版澘鐨?1/10銆傛垚绔?鈬?缁х画娣辨寲瀵规秷娣卞害**娌℃湁绌洪棿**锛堝凡缁忓帇鍒板湴鏉夸互涓嬶級銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/diag_rinr_budget.py
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
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.sensing.model.link_tables import (  # noqa: E402
    fields as fields_mod,
    power as power_mod,
)


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
    ap.add_argument("--out", default="studies/direction1/data/diag_rinr_budget")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)

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
        i_res = np.asarray(measured.i_res, dtype=float)
        i_est = np.asarray(measured.i_res_estimate, dtype=float)
        i_in = np.asarray(measured.i_in, dtype=float)
        structural = np.maximum(i_res - i_est, 0.0)
        frac_meas = np.where(i_in > 0.0, i_res / np.maximum(i_in, 1e-300), 1.0)
        frac_struct = np.where(i_in > 0.0, structural / np.maximum(i_in, 1e-300), 1.0)

        # ---- 鍒嗗瓙鍒嗚В锛氫笁椤瑰悇鑷殑缁濆鍔熺巼 ------------------------------
        power = power_mod.power_split(cfg, None)
        fields = fields_mod.interference_fields(cfg, base, power, None)
        n0 = float(power.n0)
        self_term = float(cfg.radio.residual_self_factor) * np.asarray(power.P)
        i_field = np.asarray(fields.I_sense_field, dtype=float)

        arms = {
            "open_loop": np.ones_like(i_field),          # kappa_dc = 1
            "measured": frac_meas,
            "structural_only": frac_struct,
        }
        for label, frac in arms.items():
            direct = frac * i_field
            rinr_parts = (self_term + direct) / n0
            tables = compute_link_tables(
                cfg, base, residual_fraction_by_receiver=frac
            )
            rinr = np.asarray(tables.rinr, dtype=float)
            rinr = rinr[np.isfinite(rinr) & (rinr > 0.0)]
            rows.append({
                "trial": int(trial),
                "arm": label,
                "n0": n0,
                "residual_self_over_n0_p50": float(np.median(self_term / n0)),
                "residual_direct_over_n0_p50": float(np.median(direct / n0)),
                "residual_total_over_n0_p50": float(np.median(rinr_parts)),
                "floor_share_p50": float(np.median(self_term / np.maximum(rinr_parts * n0, 1e-300))),
                "rinr_db_p50_from_tables": float(np.median(10.0 * np.log10(rinr))),
                "rinr_db_p50_from_parts": float(np.median(10.0 * np.log10(rinr_parts))),
            })
            print("  trial %d %-16s self=%.4g direct=%.4g rinr=%.2f dB"
                  % (trial, label, float(np.median(self_term / n0)),
                     float(np.median(direct / n0)),
                     float(np.median(10.0 * np.log10(rinr)))), flush=True)

    floor_share = float(np.median([r["floor_share_p50"] for r in rows
                                   if r["arm"] == "structural_only"]))
    direct_str = float(np.median([r["residual_direct_over_n0_p50"] for r in rows
                                  if r["arm"] == "structural_only"]))
    floor_str = float(np.median([r["residual_self_over_n0_p50"] for r in rows
                                 if r["arm"] == "structural_only"]))
    verdict = {
        "D1_floor_share_structural_only": floor_share,
        "D1_discrepancy_is_unit_error": bool(floor_share > 0.8),
        "D2_direct_over_floor_structural_only": direct_str / max(floor_str, 1e-300),
        "D2_no_headroom_left": bool(direct_str < 0.1 * floor_str),
        "floor_over_n0": floor_str,
        "direct_over_n0_structural_only": direct_str,
    }
    out = {"rows": rows, "verdict": verdict}
    print(json.dumps(out, indent=2, sort_keys=True))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
