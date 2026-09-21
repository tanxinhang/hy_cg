"""鍒ゅ喅锛歍P-UIC 鐨?``i_res`` 閲?``||f(n)||^2`` 鍒板簳鏄笉鏄?娈嬩綑骞叉壈"銆?
涓轰粈涔堝繀椤诲厛鍒ゅ喅
----------------
``CancellationResult`` 鎶婃畫浣欏共鎵板啓鎴愪袱椤逛箣鍜?:

    i_res = i_res_structural + i_res_estimate
          = ||x - f(x)||^2  +  ||f(n)||^2

鑰?``kappa_db = 10 log10(i_in / i_res)``锛岃繖涓?``kappa`` 姝ｆ槸閾捐矾琛ㄤ箻鍦ㄧ洿杩炲満涓?鐨勯偅涓瘮渚嬶紝鐩存帴鍐冲畾 SINR 鐨勫垎姣嶃€傚疄娴嬶紙600 m / RCS 0.1锛宻eed 2026锛変腑
``||f(n)||^2`` 鍗犱簡 ``i_res`` 鐨?**99.95%** 鈥斺€?涔熷氨鏄"瀵规秷娣卞害 35 dB"杩欏彞璇?鍑犱箮瀹屽叏鐢辫繖涓€椤瑰喅瀹氥€傚畠鍒板簳璇ヤ笉璇ュ湪閭ｅ効锛屽喅瀹氫簡鏂瑰悜 1 璇ュ線鍝蛋銆?
浠庤娴嬫ā鍨嬪嚭鍙戯紝娈嬪樊鏄?:

    r = y - f(y) = [x - f(x)] + [s - f(s)] + [n - f(n)]

**涓嶆槸** ``... - f(n)``銆傝嫢 ``f`` 鏄埌 ``span(MX)`` 鐨勬浜ゆ姇褰憋紙LS 浼拌鐨勬儏褰級锛?鍒?``n - f(n) = (I-P)n`` 涓?``f(n) = Pn`` 姝ｄ氦锛屼簬鏄?:

    ||n - f(n)||^2 = ||n||^2 - ||f(n)||^2   <   ||n||^2

涔熷氨鏄 ``f`` 椤烘墜**鍑忔帀**浜嗕竴鐐瑰櫔澹帮紝娈嬪樊閲岀殑鍣０鏄?``(I-P)n`` 鑰屼笉鏄?``Pn``銆?娈嬪樊閲?*鏍规湰娌℃湁** ``||f(n)||^2`` 杩欎竴椤广€傚鏋滆繖涓帹瀵兼垚绔嬶紝閭ｄ箞瀹冭璁″叆
``i_res`` 灏辨槸鎶?琚秷鎺夌殑鍣０"褰撴垚浜?娌℃秷鎺夌殑骞叉壈"锛屽垎姣嶈鎶珮銆?
鎬庝箞鍒ゅ喅锛堢函鏁板€硷紝涓嶅惈鍋囪锛?----------------------------
``f`` 鏄嚎鎬х殑銆備簬鏄彲浠ョ敤**鍗曞垎閲忚娴?*鎶婁笁椤瑰垎鍒祴鍑烘潵锛?
* 浠?``y = x`` 鈬?``residual_x = x - f(x)``
* 浠?``y = s`` 鈬?``residual_s = s - f(s)``
* 浠?``y = n`` 鈬?``residual_n = n - f(n)``

鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?------------------------------
* **I1 绾挎€у垎瑙?*锛歚`||r - (residual_x + residual_s + residual_n)|| / ||r|| < 1e-9``銆?  鎴愮珛 鈬?涓夐」鍒嗚В鏄亽绛夊紡锛屽彲浠ュ姣忎竴椤瑰崟鐙棶璐ｏ紱涓嶆垚绔?鈬?``f`` 闈炵嚎鎬э紝
  鏈剼鏈殑缁撹涓嶉€傜敤锛屽繀椤绘崲鏂规硶銆?* **I2 鍣０鐪熺浉**锛歚`||residual_n||^2 / ||n||^2``銆傝繖鏄畫宸噷**鐪熷疄**鐨勫櫔澹板姛鐜?  鐩稿杈撳叆鍣０鐨勬瘮渚嬨€傝嫢 > 0.99 鈬?瀵规秷鍣ㄥ嚑涔庢病鍔ㄥ櫔澹帮紝娈嬪樊鍣０ 鈮?``||n||^2``锛?  鑰屽畠鍦ㄩ摼璺〃閲?*宸茬粡鏄?* ``n0`` 閭ｄ竴椤癸紝涓嶈鍐嶈繘 ``i_res``銆?* **I3 璁拌处椤?*锛歚`i_res_estimate`` 搴旂瓑浜?``||f(n)||^2 = ||n - residual_n||^2``銆?  鏍稿瀹冩槸鍚︾湡鐨勭瓑浜庤繖涓噺锛堥槻姝㈡垜璇婚敊瀛楁锛夈€?* **I4 鍒ゅ喅**锛氳嫢 I1 鎴愮珛 涓?I2 > 0.99 涓?I3 鎴愮珛 鈬?**鍒?``||f(n)||^2`` 涓嶅睘
  娈嬩綑骞叉壈**锛氬畠鏃笉鍦ㄦ畫宸噷锛堟畫宸噷鏄?``n-f(n)``锛夛紝鍏堕噺鍊间篃杩滃皬浜庡櫔澹板湴鏉?  鐨勭湡姝ｈ浇浣?``||n||^2``銆傛鏃?娣卞害缂哄彛"鏄璐﹀亣璞°€?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/verify_tpuic_residual_identity.py --receivers 0 1 2
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
from isac_sim.scenario.belief import BeliefState  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    generate_geometry,
    radar_hardware_gain,
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


def _norm2(v: np.ndarray) -> float:
    return float(np.vdot(v, v).real)


def run_one(cfg, ctx, j: int, rng, arm: str) -> dict:
    obs = cx.build_observation(
        cfg, ctx.geom_true, ctx.geom_belief, ctx.base, int(j), rng=rng,
        sense_power=ctx.sense_power, radiated_power=ctx.radiated_power,
        processing_gain=ctx.processing_gain, hw_gain=ctx.hw_gain,
        active_mask=ctx.active_mask, base_belief=ctx.base_belief,
        include_echo=True, weak_index=0,
    )
    prune = bool(getattr(cfg.cancellation, "measure_prune_arms", True))
    only = arm if prune else None

    # ``Observation`` 涓嶄繚瀛樺櫔澹板悜閲忥紝浣?y = x + s + n锛屼簬鏄?n 鍙簿纭繕鍘熴€?    noise = np.asarray(obs.y, dtype=complex) - obs.x_direct - obs.s_target

    def solve(y):
        obs.y = np.asarray(y, dtype=complex)
        return cx.cancellation_arms(cfg, obs, only=only)[arm]

    full = solve(obs.y)
    r_x = solve(obs.x_direct).residual
    r_s = solve(obs.s_target).residual
    r_n = solve(noise).residual
    r = np.asarray(full.residual, dtype=complex)
    combo = r_x + r_s + r_n

    n2 = _norm2(noise)
    fn2 = _norm2(noise - r_n)                                  # ||f(n)||^2
    rn2 = _norm2(r_n)                                          # ||n - f(n)||^2
    x2 = _norm2(obs.x_direct)
    s2 = _norm2(obs.s_target)

    mismatch = _norm2(r - combo)
    return {
        "receiver": int(j),
        "arm": arm,
        "norm_y2": _norm2(obs.y),
        "norm_x2": x2,
        "norm_s2": s2,
        "norm_n2": n2,
        "i_in": float(full.i_in),
        "i_res": float(full.i_res),
        "i_res_structural": float(full.i_res_structural),
        "i_res_estimate": float(full.i_res_estimate),
        "kappa_db": float(full.kappa_db),
        "kappa_structural_db": (
            10.0 * np.log10(x2 / full.i_res_structural)
            if full.i_res_structural > 0.0 else float("inf")
        ),
        # I1 绾挎€у垎瑙?        "norm_r2": _norm2(r),
        "linearity_rel_err": float(mismatch / max(_norm2(r), 1e-300)),
        # I2 娈嬪樊閲岀湡瀹炵殑鍣０鍔熺巼姣?        "noise_survive_ratio": float(rn2 / max(n2, 1e-300)),
        "norm_n_minus_fn2": rn2,
        # I3 璁拌处椤规牳瀵?        "norm_fn2": fn2,
        "i_res_estimate_over_norm_fn2": float(
            full.i_res_estimate / max(fn2, 1e-300)
        ),
        "noise_enhance_db": float(full.noise_enhance_db),
        # 姝ｄ氦鎬э細n-f(n) 涓?f(n) 鏄惁姝ｄ氦
        "orthogonality_rel": float(
            abs(np.vdot(r_n, noise - r_n).real)
            / max(np.sqrt(rn2 * fn2), 1e-300)
        ),
        # 璁拌处宸細鎶?||f(n)||^2 鎷挎帀鑳界渷澶氬皯 dB
        "overcount_db": float(
            10.0 * np.log10(max(full.i_res, 1e-300) / max(full.i_res_structural, 1e-300))
        ),
        # 鐩稿鍣０鍦版澘锛氭畫浣欏共鎵板崰 ||n||^2 鐨勬瘮渚嬶紙涓ょ璁拌处锛?        "residual_over_noise_old": float(full.i_res / max(n2, 1e-300)),
        "residual_over_noise_new": float(
            full.i_res_structural / max(n2, 1e-300)
        ),
        "sigma2": float(obs.sigma2),
        "n_bins": int(np.asarray(obs.y).size),
    }


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
    ap.add_argument("--view", default="belief", choices=("belief", "oracle"))
    ap.add_argument("--out", default="studies/direction1/data/verify_tpuic_residual_identity")
    args = ap.parse_args(argv)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)

    rng = np.random.default_rng([cfg.run.seed, int(args.trial)])
    geom_true = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, geom_true, rng)
    belief = BeliefState.from_truth(cfg, geom_true, rng)
    geom_belief = belief.as_geometry(geom_true)
    base_belief = build_base_gains(
        cfg, geom_belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    gb, bb = (geom_true, None) if args.view == "oracle" else (geom_belief, base_belief)
    sense = np.full(cfg.scale.M, cfg.radio.rho * cfg.radio.P_default)
    ctx = cx.ReceiverContext.from_trial(
        cfg, geom_true, gb, base_truth,
        sense_power=sense, radiated_power=sense,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)),
        base_belief=bb, arm=args.arm,
    )
    receivers = args.receivers if args.receivers else list(range(int(cfg.scale.M)))
    rows = []
    mrng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(args.trial)])
    for j in receivers:
        rows.append(run_one(cfg, ctx, int(j), mrng, args.arm))

    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path = os.path.join(args.out, "identity_%s.csv" % args.view)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    def med(key):
        v = np.array([r[key] for r in rows], dtype=float)
        return float(np.median(v))

    verdict = {
        "I1_linearity_rel_err_median": med("linearity_rel_err"),
        "I1_linear_decomposition_holds": bool(med("linearity_rel_err") < 1e-9),
        "I2_noise_survive_ratio_median": med("noise_survive_ratio"),
        "I2_residual_noise_is_full_noise": bool(med("noise_survive_ratio") > 0.99),
        "I3_i_res_estimate_over_norm_fn2_median": med("i_res_estimate_over_norm_fn2"),
        "I3_field_matches": bool(abs(med("i_res_estimate_over_norm_fn2") - 1.0) < 1e-6),
        "overcount_db_median": med("overcount_db"),
        "kappa_db_median": med("kappa_db"),
        "kappa_structural_db_median": med("kappa_structural_db"),
        "residual_over_noise_old_median": med("residual_over_noise_old"),
        "residual_over_noise_new_median": med("residual_over_noise_new"),
        "orthogonality_rel_median": med("orthogonality_rel"),
        "sigma2": med("sigma2"),
        "n_bins": int(rows[0]["n_bins"]),
    }
    verdict["I4_verdict"] = (
        "f(n)_is_not_residual_interference"
        if (
            verdict["I1_linear_decomposition_holds"]
            and verdict["I2_residual_noise_is_full_noise"]
            and verdict["I3_field_matches"]
        )
        else "inconclusive"
    )
    print(json.dumps({"verdict": verdict, "rows": rows}, indent=2, sort_keys=True))
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
