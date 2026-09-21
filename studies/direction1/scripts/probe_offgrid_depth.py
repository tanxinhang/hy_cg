"""鎺㈤拡锛氱洿杩炲満鍋忕瀛楀吀锛坥ff-grid锛夋椂锛孴P-UIC 鐨勬繁搴﹁繕鍓╁灏戙€?
涓轰粈涔堥潪璺戜笉鍙?--------------
:mod:`tools.verify_tpuic_residual_identity` 鍒ゅ嚭缁撴瀯娈嬪樊
``||x - f(x)||^2`` 鍙湁杈撳叆鐩磋繛鍦虹殑 **~1.4e-7**锛?68 dB锛夈€備絾杩欎釜鏁颁箣鎵€浠ヨ繖涔堝皬锛?鏄洜涓鸿娴嬬殑鐩磋繛鍦哄氨鏄瓧鍏歌嚜宸辩敓鎴愮殑锛歚`x_direct = X @ h_true``锛屼簬鏄?``x`` 瀹屽叏
钀藉湪 ``span(X)`` 閲岋紝浠讳綍鍒?``span(X)`` 鐨勬姇褰遍兘浼氭妸瀹冩秷骞插噣銆傞偅鏄?*瀛楀吀瀹屽**
鐨勫亣璁撅紝涓嶆槸绠楁硶鐨勫姛缁?鈥斺€?鐪熷疄鐩磋繛鍦虹殑寤惰繜/澶氭櫘鍕掍笉浼氭伆濂借惤鍦?DD 鏍肩偣涓娿€?
鏈帰閽堟妸杩欐潯鍋囪**鏍囦环**锛氱粰鐩磋繛鍦烘敞鍏ヤ竴涓笌 ``span(X)`` **姝ｄ氦**銆佸箙搴︿负
``eps * ||x||`` 鐨勫け閰嶅垎閲忥紝鍐嶆祴缁撴瀯娈嬪樊銆傜敱浜庡け閰嶅垎閲忓湪 ``span(X)`` 涔嬪锛?鍒?``span(X)`` 鐨勬姇褰卞姩涓嶄簡瀹冿紝浜庢槸::

    kappa_structural(eps) ~ -20 log10(eps)      锛堢函鎶曞奖鎯呭舰锛?
杩欐潯鏇茬嚎鎶?娣卞害"浠?绠楁硶鏈夊寮?杩樺師鎴?瀛楀吀鏈夊鍑?锛屼篃灏辨妸鏂瑰悜 1 鐨勭洰鏍囦粠
"娣辨寲瀵规秷"鏀规垚浜?鍑忓皯瀛楀吀澶遍厤"銆?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?------------------------------
* **O1 澶嶇幇**锛歚`eps = 0`` 鏃?``kappa_structural`` 搴斿洖鍒?~68 dB
  锛堜笌鍒ゅ喅鑴氭湰涓€鑷达級锛屼笉涓€鑷磋鏄庢敞鍏ユ柟寮忕牬鍧忎簡鍒殑涓滆タ銆?* **O2 杈炬爣瑁曞害**锛氱粰鍑烘弧瓒冲嚑浣曢渶姹?``GAMMA_REQ_DB = 52.25`` 鐨勬渶澶?``eps``銆?* **O3 鑴嗗急鎬?*锛氳嫢璇?``eps < 0.01``锛堢洿杩炲満瀛楀吀澶栬兘閲忓崰姣?> 0.01% 鍗充笉杈炬爣锛?  鈬?鍒?*瀛楀吀瀹屽鏄剢寮卞亣璁?*锛屾柟鍚?1 鐨勬瑙ｆ槸**鏀?off-grid**锛堝瓧鍏稿姞瀵?/ 鍒囧悜
  闃舵暟 / 绂绘牸鎼滅储锛夛紝鑰屼笉鏄户缁繁鎸栧娑堝櫒鏈韩锛?  ``structural_only`` 閭ｄ竴璺?End-to-end 澧炵泭涔熷繀椤诲湪 off-grid 涓嬮噸娴嬫墠绠楁暟銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/probe_offgrid_depth.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    generate_geometry,
    radar_hardware_gain,
)

#: 鍑犱綍闇€姹傦紙600 m / RCS 0.1锛夛紝鏉ヨ嚜 KAPPA_DERIVATION 鐨勫彛寰勩€?GAMMA_REQ_DB = 52.25


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


def _orthogonal_direction(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """鍙栦竴涓笌 ``span(X)`` 姝ｄ氦鐨勫崟浣嶅悜閲忋€?""
    n = X.shape[0]
    w = rng.normal(size=n) + 1j * rng.normal(size=n)
    if X.shape[1]:
        coeff = np.linalg.lstsq(X, w, rcond=None)[0]
        w = w - X @ coeff
        # 浜屾姝ｄ氦鍖栵紝閬垮厤鐥呮€佸瓧鍏哥暀涓嬫畫浣欏垎閲?        coeff = np.linalg.lstsq(X, w, rcond=None)[0]
        w = w - X @ coeff
    norm = float(np.sqrt(np.vdot(w, w).real))
    return w / norm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trial", type=int, default=0)
    ap.add_argument("--receivers", type=int, nargs="*", default=None)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument(
        "--eps", type=float, nargs="*",
        default=[0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
    )
    ap.add_argument("--out", default="studies/direction1/data/probe_offgrid_depth")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng([cfg.run.seed, int(args.trial)])
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    sense = np.full(cfg.scale.M, cfg.radio.rho * cfg.radio.P_default)
    ctx = cx.ReceiverContext.from_trial(
        cfg, geom, geom, base,
        sense_power=sense, radiated_power=sense,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)), arm=args.arm,
    )
    receivers = args.receivers if args.receivers else list(range(int(cfg.scale.M)))
    mrng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(args.trial)])
    aux = np.random.default_rng([cfg.run.seed, 7_000_000 + int(args.trial)])

    prune = bool(getattr(cfg.cancellation, "measure_prune_arms", True))
    only = args.arm if prune else None

    rows = []
    for j in receivers:
        obs = cx.build_observation(
            cfg, ctx.geom_true, ctx.geom_belief, ctx.base, int(j), rng=mrng,
            sense_power=ctx.sense_power, radiated_power=ctx.radiated_power,
            processing_gain=ctx.processing_gain, hw_gain=ctx.hw_gain,
            active_mask=ctx.active_mask, base_belief=ctx.base_belief,
            include_echo=True, weak_index=0,
        )
        x = np.asarray(obs.x_direct, dtype=complex)
        s = np.asarray(obs.s_target, dtype=complex)
        n_noise = np.asarray(obs.y, dtype=complex) - x - s
        X = np.asarray(obs.X, dtype=complex)
        u = _orthogonal_direction(X, aux)
        xnorm = float(np.sqrt(np.vdot(x, x).real))

        for eps in args.eps:
            # 鈿狅笍 涓ゅ閮借鏀癸細``i_res_structural`` 鏄噦鍐呴儴鎷?``obs.x_direct``
            # 鍗曠嫭鏍哥畻鐨勶紙瀹冪煡閬撶湡鍊硷級锛屽彧鏀?``obs.y`` 涓嶄細璁╁畠鍔?鈥斺€?鎴戠涓€鐗?            # 灏卞彧鏀逛簡 y锛岀粨鏋?魏 瀵?eps 瀹屽叏涓嶅搷搴旓紝閭ｆ槸娉ㄥ叆澶辨晥涓嶆槸缁撹銆?            x_off = x + (float(eps) * xnorm) * u
            obs.x_direct = x_off
            obs.y = x_off + s + n_noise
            res = cx.cancellation_arms(cfg, obs, only=only)[args.arm]
            struct = float(res.i_res_structural)
            i_in = float(res.i_in)
            kappa_struct = (
                10.0 * np.log10(i_in / struct) if struct > 0.0 else float("inf")
            )
            rows.append({
                "receiver": int(j),
                "eps": float(eps),
                "offgrid_energy_fraction": float(eps) ** 2,
                "i_in": i_in,
                "i_res": float(res.i_res),
                "i_res_structural": struct,
                "i_res_estimate": float(res.i_res_estimate),
                "kappa_db": float(res.kappa_db),
                "kappa_structural_db": float(kappa_struct),
                "eta_survive_q": float(res.eta_survive_q),
                "meets_gamma_req": bool(kappa_struct >= GAMMA_REQ_DB),
            })

    keys = list(rows[0])
    path = os.path.join(args.out, "offgrid.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"gamma_req_db": GAMMA_REQ_DB, "rows": []}
    by_eps = {}
    for r in rows:
        by_eps.setdefault(r["eps"], []).append(r)
    for eps in sorted(by_eps):
        block = by_eps[eps]
        ks = np.array([b["kappa_structural_db"] for b in block], dtype=float)
        summary["rows"].append({
            "eps": float(eps),
            "offgrid_energy_fraction": float(eps) ** 2,
            "kappa_structural_db_p10": float(np.quantile(ks, 0.10)),
            "kappa_structural_db_p50": float(np.quantile(ks, 0.50)),
            "kappa_structural_db_p90": float(np.quantile(ks, 0.90)),
            "meets_gamma_req_p10": bool(np.quantile(ks, 0.10) >= GAMMA_REQ_DB),
        })
    k0 = [r for r in summary["rows"] if r["eps"] == 0.0]
    verdict = {
        "O1_kappa_structural_at_zero_db": (
            k0[0]["kappa_structural_db_p50"] if k0 else float("nan")
        ),
        "O1_reproduces_identity_probe": bool(
            k0 and k0[0]["kappa_structural_db_p50"] > 60.0
        ),
    }
    passing = [
        r["eps"] for r in summary["rows"]
        if r["meets_gamma_req_p10"]
    ]
    eps_max = max(passing) if passing else 0.0
    verdict["O2_eps_max_meeting_req"] = float(eps_max)
    verdict["O2_energy_fraction_max"] = float(eps_max) ** 2
    verdict["O3_assumption_is_fragile"] = bool(eps_max < 0.01)
    summary["verdict"] = verdict

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
