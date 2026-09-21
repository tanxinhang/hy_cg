"""璇婃柇 TP-UIC 娑堥櫎娣卞害鐨勪笅灏炬瀯鎴愶紙鏂瑰悜 1 鐨勫叆鍙ｈ瘖鏂級銆?
涓轰粈涔堝厛鍋氳繖涓€姝?----------------
``kappa_db = 10 log10(i_in / i_res)``锛岃€?``i_res`` 鏄?*涓ら」涔嬪拰**::

    i_res = i_res_structural + i_res_estimate
          = ||x - f(x)||^2      + ||f(n)||^2

绗竴椤规槸浼拌鍣?*鎷掔粷**鍑忔帀鐨勫共鎵帮紙鐢变繚鎶ゅ瓙绌洪棿鍐冲畾锛夛紝绗簩椤规槸瀹冧粠鍣０閲?鎷夎繘瑙傛祴鐨勯儴鍒嗭紙鐢辨嫙鍚堢郴鏁颁釜鏁颁笌鍣０鐢靛钩鍐冲畾锛夈€備袱鑰呯殑鏀硅繘鏂瑰悜**鐩稿弽**锛?
* 鑻?estimate 鍗犱富瀵?鈬?鎷熷悎闃舵暟杩囬珮 / 娆犳鍒?鈬?鏉犳潌鏄?*闄嶉樁銆佹敹缂┿€侀檷鍣?*锛?* 鑻?structural 鍗犱富瀵?鈬?淇濇姢瀛愮┖闂村悆鎺夊お澶?鈬?鏉犳潌鏄?*淇濇姢绛栫暐銆佸瓧鍏搞€佺淮搴?*銆?
涓嶅厛娴嬫竻鍗犳瘮灏卞姩鎵嬶紝绛変簬鍦ㄤ袱涓浉鍙嶆柟鍚戦噷璧屼竴涓€傛湰鑴氭湰鍙?*娴嬮噺**锛屼笉鏀逛换浣曚唬鐮併€?
鍒ゆ嵁锛堣窇鍓嶅畾姝伙紝璺戝畬鍙垽涓嶆敼锛?----------------------------
* **D1 鍗犳瘮**锛歚`estimate / i_res`` 鐨?*涓綅**銆?  鈮?0.5 鈬?鍒や负"浼拌鍣０涓诲"锛屾柟鍚?1 搴斿厛鏀婚檷闃?姝ｅ垯鍖栵紱
  <  0.5 鈬?鍒や负"缁撴瀯娈嬪樊涓诲"锛屾柟鍚?1 搴斿厛鏀讳繚鎶ょ瓥鐣ャ€?* **D2 涓嬪熬**锛歚`kappa`` 鐨?**P10 涓庝腑浣嶄箣宸?*锛坉B锛夈€傝繖灏辨槸"涓嬪熬"鐨勫搴︼紱
  鑻?> 6 dB锛岃鏄庨€愭帴鏀舵満娣卞害鐨勬暎甯冩湰韬氨鏄崯澶辨潵婧愶紝鍊煎緱鍋氫笅灏句笓鐢ㄥ鐞嗐€?* **D3 瀛樻椿**锛歚`eta_survive_q`` 鐨?P10 / 涓綅銆傝繖鏄垎瀛愪晶鐨勫娑堜唬浠凤紱
  鑻ヤ腑浣?< 0.9锛屽垯"姣忔秷 1 dB 骞叉壈瑕佷粯 X dB 鍥炴尝"鐨勮处蹇呴』杩涚洰鏍囧嚱鏁般€?* **D4 涓嬪熬鍙娴嬫€?*锛歚`prior_quantile_fraction`` 鎹㈢畻鐨?dB 涓庡疄娴嬩腑浣嶇殑宸€?  鑻ヨ鍒掔敤鍒嗕綅鏁版瘮瀹炴祴涓綅宸?> 6 dB锛岃鏄庤鍒掍晶宸茬粡涓轰笅灏句粯浜嗗ぇ浠烽挶锛?  鍓婁笅灏剧殑鏀剁泭瑕佹寜"鍒嗕綅鏁版敼鍠?鑰屼笉鏄?涓綅鏀瑰杽"鏉ョ畻銆?
鈿狅笍 鍙ｅ緞锛氶粯璁?``--belief`` 缂虹渷涓?**oracle**锛堢湡鍊煎綋淇″康锛夛紝閭ｆ槸**鑳藉姏涓婄晫**锛?鐢熶骇绯荤粺璺戠殑鏄壈鍔ㄤ俊蹇点€備袱鑰呴兘璺戯紝宸€煎氨鏄?淇″康璇樊鍚冩帀浜嗗灏戞繁搴?銆?鍙湁鏍?``belief`` 鐨勯偅涓€鍒楁墠鑳借繘绯荤粺鏁板瓧銆?
璺?:

    PY=E:/anaconda/3_11_python/python.exe
    $PY tools/diag_tpuic_depth_budget.py --trials 4
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


def _db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.clip(np.asarray(values, dtype=float), 1e-300, None))


def _quantiles(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "p10": float("nan"), "p50": float("nan"), "p90": float("nan")}
    return {
        "n": int(v.size),
        "p10": float(np.quantile(v, 0.10)),
        "p50": float(np.quantile(v, 0.50)),
        "p90": float(np.quantile(v, 0.90)),
    }


def measure_one(cfg, ctx, j: int, rng) -> dict:
    """鍦ㄦ帴鏀舵満 j 涓婅窇涓€娆¤娴嬭噦锛屾妸娣卞害棰勭畻鎷嗗紑銆?""
    obs = cx.build_observation(
        cfg, ctx.geom_true, ctx.geom_belief, ctx.base, int(j), rng=rng,
        sense_power=ctx.sense_power, radiated_power=ctx.radiated_power,
        processing_gain=ctx.processing_gain, hw_gain=ctx.hw_gain,
        active_mask=ctx.active_mask, base_belief=ctx.base_belief,
        include_echo=True, weak_index=0,
    )
    prune = bool(getattr(cfg.cancellation, "measure_prune_arms", True))
    parts = cx.cancellation_arms(cfg, obs, only=(ctx.arm if prune else None))
    res = parts[ctx.arm]
    i_res = float(res.i_res)
    i_est = float(res.i_res_estimate)
    i_struct = float(res.i_res_structural)
    i_in = float(res.i_in)
    row = {
        "receiver": int(j),
        "i_in": i_in,
        "i_res": i_res,
        "i_res_structural": i_struct,
        "i_res_estimate": i_est,
        "estimate_share": (i_est / i_res) if i_res > 0.0 else float("nan"),
        "kappa_db": float(res.kappa_db) if np.isfinite(res.kappa_db) else float("nan"),
        "kappa_pred_db": float(res.kappa_pred_db),
        "calibration_error_db": float(res.calibration_error_db),
        "protect_dim": int(res.protect_dim),
        "n_coefficients": int(res.n_coefficients),
        "eta_protect": float(res.eta_protect),
        "eta_survive": float(res.eta_survive),
        "eta_survive_q": float(res.eta_survive_q),
        "eta_survive_pred_q": float(res.eta_survive_pred_q),
        "noise_enhance_db": float(res.noise_enhance_db),
        "s_q_energy": float(res.s_q_energy),
    }
    # 瑙勫垝渚х敤鐨勫厛楠屽垎浣嶆暟锛堜笅灏句繚鎶わ級锛歱robability = 1 - tail
    try:
        pq = cx.residual_power_prior_quantile(res, 1.0 - args_tail)
        row["prior_quantile_fraction"] = float(pq / i_in) if i_in > 0.0 else float("nan")
    except Exception:
        row["prior_quantile_fraction"] = float("nan")
    row["arm"] = ctx.arm
    return row


# 鍏堥獙鍒嗕綅鏁扮敤鐨勫熬姒傜巼锛堟ā鍧楃骇锛屼緵涓婇潰鐨勯棴鍖呰鍙栵級
args_tail = 0.20


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=4)
    ap.add_argument("--n-cpi", type=int, default=1)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--arm", default="tp_uic_full")
    ap.add_argument("--tail", type=float, default=0.20)
    ap.add_argument("--out", default="studies/direction1/data/diag_tpuic_depth_budget")
    args = ap.parse_args(argv)

    global args_tail
    args_tail = float(args.tail)

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)

    rows = []
    t0 = time.time()
    for trial in range(int(args.trials)):
        rng = np.random.default_rng([cfg.run.seed, int(trial)])
        geom_true = generate_geometry(cfg, rng)
        base_truth = build_base_gains(cfg, geom_true, rng)
        belief = BeliefState.from_truth(cfg, geom_true, rng)
        geom_belief = belief.as_geometry(geom_true)
        base_belief = build_base_gains(
            cfg, geom_belief, rng, channel=base_truth,
            rcs_view=cfg.prior.scheduler_rcs.lower(),
        )
        sense = np.full(cfg.scale.M, cfg.radio.rho * cfg.radio.P_default)
        for view, gb, bb in (
            ("oracle", geom_true, None),
            ("belief", geom_belief, base_belief),
        ):
            ctx = cx.ReceiverContext.from_trial(
                cfg, geom_true, gb, base_truth,
                sense_power=sense, radiated_power=sense,
                processing_gain=float(cfg.waveform.N * cfg.waveform.L),
                hw_gain=float(radar_hardware_gain(cfg)),
                base_belief=bb, arm=args.arm,
            )
            mrng = np.random.default_rng([cfg.run.seed, 10 ** 6 + int(trial)])
            for j in range(int(cfg.scale.M)):
                row = measure_one(cfg, ctx, j, mrng)
                row["trial"] = int(trial)
                row["view"] = view
                rows.append(row)
        print("  trial %d/%d  [%.0f s]" % (trial + 1, args.trials, time.time() - t0),
              flush=True)

    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    path = os.path.join(args.out, "depth_budget.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"config": {
        "preset": args.preset, "area": args.area, "rcs": args.rcs,
        "M": int(cfg.scale.M), "Q": int(cfg.scale.Q),
        "seed": int(args.seed), "trials": int(args.trials),
        "arm": args.arm, "tail_probability": args.tail,
    }}
    for view in ("oracle", "belief"):
        sub = [r for r in rows if r["view"] == view]
        block = {}
        block["estimate_share"] = _quantiles([r["estimate_share"] for r in sub])
        block["kappa_db"] = _quantiles([r["kappa_db"] for r in sub])
        block["eta_survive_q"] = _quantiles([r["eta_survive_q"] for r in sub])
        block["protect_dim"] = _quantiles([r["protect_dim"] for r in sub])
        block["n_coefficients"] = _quantiles([r["n_coefficients"] for r in sub])
        block["calibration_error_db"] = _quantiles(
            [r["calibration_error_db"] for r in sub]
        )
        kq = block["kappa_db"]
        block["tail_width_db"] = float(kq["p50"] - kq["p10"]) if kq["n"] else float("nan")
        pq = np.array([r["prior_quantile_fraction"] for r in sub], dtype=float)
        pq = pq[np.isfinite(pq) & (pq > 0.0)]
        block["planning_kappa_db"] = (
            float(-np.median(_db(pq))) if pq.size else float("nan")
        )
        summary[view] = block

    # 鍒ゆ嵁锛堣窇鍓嶅畾姝伙級
    b = summary["belief"]
    share = b["estimate_share"]["p50"]
    verdict = {}
    verdict["D1_dominant_term"] = (
        "estimate" if share >= 0.5 else "structural"
    )
    verdict["D1_estimate_share_median"] = share
    verdict["D2_tail_width_db"] = b["tail_width_db"]
    verdict["D2_tail_is_loss_source"] = bool(b["tail_width_db"] > 6.0)
    verdict["D3_eta_q_median"] = b["eta_survive_q"]["p50"]
    verdict["D3_eta_q_p10"] = b["eta_survive_q"]["p10"]
    pk = b["planning_kappa_db"]
    bk = b["kappa_db"]["p50"]
    verdict["D4_planning_vs_measured_db"] = float(pk - bk)
    verdict["D4_planning_pays_for_tail"] = bool((pk - bk) > 6.0)
    summary["verdict"] = verdict

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print("wrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
