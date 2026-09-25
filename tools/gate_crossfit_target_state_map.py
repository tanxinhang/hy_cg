#!/usr/bin/env python3
"""共享目标偏移 cross-fitted MAP 的分层筛查（方案 §7--§10）。

五个对照臂，每个 (误差档 × 方法) 用自己的 H0 标定门限：

    nominal          两个 CPI 都用原始 belief，不修正
    self_fit         每个 CPI 用**自己**的数据估状态并在自己身上检测
    local_crossfit   每个接收机各估一个偏移，交叉检测
    shared_crossfit  六个接收机共同估一个共享偏移（proposed）
    oracle           用真值目标位置（上界，不可部署）

第一版只引入"共享二维水平偏移"这一个变量：不改阵形、功率、速度、TP-UIC 规则
或融合权重。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from dataclasses import replace

from isac_sim.core.config import Config
from isac_sim.detection.evaluation_split import EvaluationPartition
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.receiver.cancellation.protection import _protection_basis
from isac_sim.receiver.target_state import (
    apply_target_offset,
    build_scorer,
    crossfit_target_state_score,
    fit_shared_target_offset,
    neighbourhood_energy,
)
from isac_sim.sensing.model import radar_hardware_gain
from tools import run_tpuic_receiver_benchmark as bench

ARMS = ("nominal", "self_fit", "local_crossfit", "shared_crossfit", "oracle")
BASELINE_CURVE_ARMS = (
    "nominal", "wls_map", "local_crossfit", "shared_crossfit", "oracle"
)


def _active_arms(args):
    return BASELINE_CURVE_ARMS if bool(getattr(args, "baseline_curve", False)) else ARMS


# --------------------------------------------------------------------------
# 配置与场景
# --------------------------------------------------------------------------


def _cfg(args) -> Config:
    defaults = bench.build_parser().parse_args(["--out", str(args.out)])
    defaults.master_seed = args.master_seed
    defaults.area_xy = args.area_xy
    defaults.uavs = args.uavs
    defaults.targets_count = args.targets_count
    defaults.min_uav_separation_m = getattr(args, "min_uav_separation_m", 20.0)
    defaults.m_rx = args.m_rx
    defaults.target_rcs = args.target_rcs
    defaults.direct_dd_sigma = 0.10
    defaults.interference_tangent_order = 1
    defaults.interference_uncertainty_weighted = True
    defaults.direct_mismatch_covariance_model = "sigma_point_replacement"
    defaults.target_glrt_radius_bins = args.glrt_radius
    defaults.target_glrt_grid_points = args.glrt_points
    defaults.max_protected_targets = 1
    # 声明的先验协方差（= 搜索尺度 sigma_p），与实际注入的误差半径无关。
    defaults.belief_pos_sigma = args.prior_sigma_m
    return bench._make_cfg(defaults)


def _role(scene: int, args) -> str:
    if scene < args.train_scenes:
        return "train"
    if scene < args.train_scenes + args.calibration_scenes:
        return "calibration"
    return "test"


def _belief_for_radius(cfg, truth, belief, target: int, radius_m: float,
                       scene: int):
    """只给被测目标注入半径 ``radius_m`` 的水平误差，其它目标信念 = 真值。

    隔离是**有意**的：其余目标的信念若也带 ~150 m 误差，它们的回波会以"未被
    投影掉的干扰"留在残差里，而偏移搜索会锁到那些回波上（实测：搜索峰落在
    距真值 300 m 处，与 H0 的峰重合）。第一版要测的是"共享偏移能不能被找回"，
    所以先把这条混淆去掉。
    """
    xhat = np.concatenate(
        [np.array(truth.p_tgt, dtype=float), np.array(truth.v_tgt, dtype=float)],
        axis=1,
    )
    out = replace(belief, xhat=xhat)
    angle = float(np.random.default_rng(
        [int(cfg.run.seed), int(scene), 707]
    ).uniform(0.0, 2.0 * np.pi))
    out.xhat[int(target), 0] = float(truth.p_tgt[target, 0]) + radius_m * np.cos(angle)
    out.xhat[int(target), 1] = float(truth.p_tgt[target, 1]) + radius_m * np.sin(angle)
    return out, angle


def _pair(cfg, truth, belief_geom, base, args, scene: int, rx: int, look: int):
    m = int(cfg.scale.M)
    power = np.full(m, float(cfg.radio.P_default) * float(cfg.radio.rho))
    rng = np.random.default_rng([int(cfg.run.seed), scene, rx, look, 601])
    direct_rng = np.random.default_rng([int(cfg.run.seed), scene, rx, 602])
    h1, h0 = cx.build_observation_pair(
        cfg, truth, belief_geom, base, rx,
        rng=rng, direct_error_rng=direct_rng, sense_power=power,
        radiated_power=power,
        processing_gain=float(cfg.waveform.N * cfg.waveform.L),
        hw_gain=float(radar_hardware_gain(cfg)),
        exclude_target=args.target, active_mask=np.ones(m, dtype=bool),
        weak_index=args.target, share_noise=False,
    )
    pin = frozenset({int(args.target)})
    pinned = []
    for obs in (h1, h0):
        pinned.append(replace(
            obs,
            protected_targets=pin,
            basis_belief=_protection_basis(
                cfg, obs.targets_belief, obs.y.size, protected_ids=pin
            ),
        ))
    return pinned[0], pinned[1]


# --------------------------------------------------------------------------
# 打分：一次 arm 装配 + 局部 DD 网格上的最大值聚合
# --------------------------------------------------------------------------


def _score(cfg, obs, target: int, arm: str = "tp_uic_full") -> float:
    results = cx.cancellation_arms(cfg, obs, weak_target=target, only=arm)
    model = gl.residual_model(cfg, obs, arm, results)
    return float(gl.target_neighbourhood_glrt(
        cfg, obs, results[arm], model, target=target,
        radius_bins=float(cfg.cancellation.target_glrt_radius_bins),
        grid_points=int(cfg.cancellation.target_glrt_grid_points),
        aggregation="max",
    ).statistic)


def _setup(cfg, obs, target: int):
    """(arm 结果, 残差模型, 缓存打分器) —— 一次装配，搜索反复复用。"""
    results = cx.cancellation_arms(cfg, obs, weak_target=target, only="tp_uic_full")
    model = gl.residual_model(cfg, obs, "tp_uic_full", results)
    return results, model, build_scorer(cfg, obs, model, target)


def _fused_score(cfg, observations, target: int) -> float:
    """中央融合：六个接收机的 held-out 统计量求和。"""
    return float(np.sum([_score(cfg, obs, target) for obs in observations]))


# --------------------------------------------------------------------------
# 五个臂
# --------------------------------------------------------------------------


def _arm_nominal(cfg, looks, target: int) -> tuple[float, dict]:
    total = sum(_fused_score(cfg, look, target) for look in looks)
    return float(total), {}


def _arm_self_fit(cfg, looks, target, belief, receivers, belief_geom, args):
    """每个 CPI 用**自己**的数据估状态并在自己身上检测（预期 P_D 高、P_FA 失控）。"""
    total = 0.0
    deltas = []
    for look in looks:
        scorers = [_setup(cfg, obs, target)[2] for obs in look]
        fit = fit_shared_target_offset(
            cfg, look, target, belief, receivers=receivers,
            belief_geometry=belief_geom, scorers=scorers,
            prior_sigma_m=args.prior_sigma_m, search_sigma=args.search_sigma,
            coarse_step_m=args.coarse_step_m,
        )
        applied = [
            apply_target_offset(cfg, obs, target, fit.delta_xy_m,
                                receiver=int(rx), belief_geometry=belief_geom)
            for rx, obs in zip(receivers, look)
        ]
        total += _fused_score(cfg, applied, target)
        deltas.append(np.asarray(fit.delta_xy_m, dtype=float))
    return float(total), {"self_delta": deltas}


def _arm_local_crossfit(cfg, looks, target, belief, receivers, belief_geom, args):
    """每个接收机各估一个偏移，再交叉检测：共享状态建模是否真有独立贡献。"""
    total = 0.0
    deltas = []
    slots = list(range(len(receivers)))
    for held in range(2):
        train, held_out = looks[1 - held], looks[held]
        applied = []
        for slot in slots:
            rx = int(receivers[slot])
            scorer = _setup(cfg, train[slot], target)[2]
            fit = fit_shared_target_offset(
                cfg, [train[slot]], target, belief,
                receivers=(rx,), belief_geometry=belief_geom, scorers=[scorer],
                prior_sigma_m=args.prior_sigma_m, search_sigma=args.search_sigma,
                coarse_step_m=args.coarse_step_m,
            )
            applied.append(apply_target_offset(
                cfg, held_out[slot], target, fit.delta_xy_m, receiver=rx,
                belief_geometry=belief_geom))
            deltas.append(np.asarray(fit.delta_xy_m, dtype=float))
        total += _fused_score(cfg, applied, target)
    return float(total), {"local_delta": deltas}


def _solver_options(args):
    """只有 ``gated_topk`` 读这些旋钮；``coarse_to_fine`` 一条都不认。"""
    if str(args.solver) != "gated_topk":
        return None
    return {
        "screen_receivers": int(args.screen_receivers),
        "screen_keep": int(args.screen_keep),
        "verify_receivers": int(args.verify_receivers),
        "verify_keep": int(args.verify_keep),
        "top_k": int(args.top_k),
    }


def _arm_shared_crossfit(cfg, looks, target, belief, receivers, belief_geom, args):
    """六个接收机共同估一个共享偏移（proposed）。"""
    scorers = [[_setup(cfg, obs, target)[2] for obs in look] for look in looks]

    def score_fn(observations):
        return _fused_score(cfg, observations, target)

    result = crossfit_target_state_score(
        cfg, looks[0], looks[1], target, belief, receivers=receivers,
        belief_geometry=belief_geom, scorers_a=scorers[0], scorers_b=scorers[1],
        score_fn=score_fn, prior_sigma_m=args.prior_sigma_m,
        search_sigma=args.search_sigma, solver=args.solver,
        coarse_step_m=args.coarse_step_m,
        solver_options=_solver_options(args),
        frozen_gate=getattr(args, "state_gate", None),
    )
    return float(result.statistic), {
        "delta_a": result.delta_a_xy_m,
        "delta_b": result.delta_b_xy_m,
        "objective_a": result.objective_a,
        "objective_b": result.objective_b,
        "evaluations": result.evaluations,
        "converged": result.converged,
        "accepted_a": result.accepted_a,
        "accepted_b": result.accepted_b,
        "raw_delta_a": result.raw_delta_a_xy_m,
        "raw_delta_b": result.raw_delta_b_xy_m,
    }


def _arm_oracle(cfg, looks, target, belief_geom, truth):
    """上界：把信念位置直接换成真值位置。"""
    true_delta = np.asarray(truth.p_tgt[target, :2], dtype=float) - np.asarray(
        belief_geom.p_tgt[target, :2], dtype=float)
    total = 0.0
    for look in looks:
        applied = [
            apply_target_offset(cfg, obs, target, true_delta, receiver=int(rx),
                                belief_geometry=belief_geom)
            for rx, obs in zip(range(len(look)), look)
        ]
        total += _fused_score(cfg, applied, target)
    return float(total), {}


# --------------------------------------------------------------------------
# 汇总
# --------------------------------------------------------------------------


def _wilson(success_rate: float, n: int) -> tuple[float, float]:
    return bench._wilson(success_rate, n)


def _paired_bootstrap(rows, name: str, reference: str, tau_name: float,
                      tau_ref: float, repeats: int, rng):
    """把 (方法 − 参考) 的配对差做成 bootstrap 区间（按 scene 重抽样）。

    两个方法各用**自己**的门限，因为跨方法比较必须在各自的工作点上做 —— 用
    nominal 的门限去评价 MAP 正是方案 §10 明令禁止的。
    """
    by_scene = {}
    for row in rows:
        by_scene[int(row["scene_id"])] = row
    scenes = sorted(by_scene)
    auc_diffs, pd_diffs = [], []
    for _ in range(int(repeats)):
        pick = [int(s) for s in rng.choice(scenes, size=len(scenes), replace=True)]
        h0_r = np.asarray([float(by_scene[s][f"{reference}_h0"]) for s in pick])
        h1_r = np.asarray([float(by_scene[s][f"{reference}_h1"]) for s in pick])
        h0_m = np.asarray([float(by_scene[s][f"{name}_h0"]) for s in pick])
        h1_m = np.asarray([float(by_scene[s][f"{name}_h1"]) for s in pick])
        auc_diffs.append(bench._auc(h0_m, h1_m) - bench._auc(h0_r, h1_r))
        pd_diffs.append(
            float(np.mean(h1_m > tau_name)) - float(np.mean(h1_r > tau_ref))
        )
    return {
        "delta_auc": float(np.mean(auc_diffs)),
        "delta_auc_ci95": [float(np.percentile(auc_diffs, 2.5)),
                           float(np.percentile(auc_diffs, 97.5))],
        "delta_pd": float(np.mean(pd_diffs)),
        "delta_pd_ci95": [float(np.percentile(pd_diffs, 2.5)),
                          float(np.percentile(pd_diffs, 97.5))],
    }


def _summarise(rows, args):
    evaluation = []
    thresholds = {}
    grouped = defaultdict(list)
    for row in rows:
        grouped[float(row["error_radius_m"])].append(row)
    rng = np.random.default_rng([int(args.master_seed), 909])
    for radius in sorted(grouped):
        subset = grouped[radius]
        partition = EvaluationPartition.from_records(subset, require_train=True)
        cal = partition.calibration.rows
        test = partition.test.rows
        by_method = {}
        for name in _active_arms(args):
            cal_h0 = np.asarray([float(r[f"{name}_h0"]) for r in cal])
            threshold = float(bench._calibrated_threshold(
                cal_h0, float(args.p_fa), "split_conformal"))
            h0 = np.asarray([float(r[f"{name}_h0"]) for r in test])
            h1 = np.asarray([float(r[f"{name}_h1"]) for r in test])
            pfa = float(np.mean(h0 > threshold))
            pd = float(np.mean(h1 > threshold))
            entry = {
                "error_radius_m": float(radius),
                "method": name,
                "threshold": threshold,
                "test_auc": float(bench._auc(h0, h1)),
                "empirical_pfa": pfa,
                "pfa_ci95": list(_wilson(pfa, h0.size)),
                "test_pd": pd,
                "pd_ci95": list(_wilson(pd, h1.size)),
                "n_calibration": len(cal),
                "n_test": len(test),
            }
            by_method[name] = entry
            evaluation.append(entry)
            thresholds[name] = threshold
        oracle = by_method["oracle"]
        for name in _active_arms(args):
            by_method[name]["oracle_auc_gap"] = (
                oracle["test_auc"] - by_method[name]["test_auc"])
            by_method[name]["oracle_pd_gap"] = (
                oracle["test_pd"] - by_method[name]["test_pd"])
        for name in _active_arms(args):
            if name == "nominal":
                continue
            by_method[name].update(_paired_bootstrap(
                test, name, "nominal", thresholds[name],
                thresholds["nominal"], args.bootstrap, rng))
            by_method[name]["delta_auc_vs_nominal"] = (
                by_method[name]["test_auc"] - by_method["nominal"]["test_auc"])
            by_method[name]["delta_pd_vs_nominal"] = (
                by_method[name]["test_pd"] - by_method["nominal"]["test_pd"])
        # 状态估计误差（共享臂）
        errs = [float(r["state_error_a_m"]) for r in test] + \
               [float(r["state_error_b_m"]) for r in test]
        by_method["shared_crossfit"]["state_rmse_m"] = float(np.sqrt(np.mean(np.square(errs))))
        by_method["shared_crossfit"]["state_rmse_p95_m"] = float(np.percentile(errs, 95))
        # "先验不动"的基线：估计量恒为 0，误差就是注入的 |delta|。
        # （第一版把 |delta|^2 又平方了一次，基线显示成 22500 m —— 平方套两层。）
        by_method["nominal"]["state_rmse_m"] = float(np.sqrt(np.mean(
            [float(r["true_delta_x_m"]) ** 2 + float(r["true_delta_y_m"]) ** 2
             for r in test])))
        by_method["nominal"]["state_median_m"] = by_method["nominal"]["state_rmse_m"]
        by_method["shared_crossfit"]["state_median_m"] = float(np.median(errs))
        if bool(getattr(args, "baseline_curve", False)):
            wls_errs = [float(r["wls_state_error_a_m"]) for r in test] + \
                       [float(r["wls_state_error_b_m"]) for r in test]
            by_method["wls_map"]["state_rmse_m"] = float(
                np.sqrt(np.mean(np.square(wls_errs))))
            by_method["wls_map"]["state_rmse_p95_m"] = float(
                np.percentile(wls_errs, 95))
            by_method["wls_map"]["state_median_m"] = float(np.median(wls_errs))
        runtime = [float(r["runtime_s"]) for r in test]
        for name in _active_arms(args):
            by_method[name]["median_runtime_s"] = float(np.median(runtime))
            by_method[name]["p95_runtime_s"] = float(np.percentile(runtime, 95))
    return evaluation


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def _scene_with_retry(cfg, scene: int, path, attempts: int = 4):
    """``bench._scene`` 每个误差档都会把快照**重存一遍**（它不做缓存判断）。

    Gate 2 实测：8 路并发下 164 次重存里有 2 次撞上
    ``OSError: [Errno 22] Invalid argument``（shard 4 的 scene_00004、shard 6 的
    scene_00006，都在第 3 个档位上）—— 每个场景文件只属于一个分片，所以不是
    并发写冲突，是 Windows 上偶发的临时句柄问题。一次失败就丢掉该分片剩下
    的所有行（本轮因此丢了 20 行、多花一小时），所以这里退避重试。
    """
    last: BaseException | None = None
    for k in range(int(attempts)):
        try:
            return bench._scene(cfg, scene, path)
        except OSError as exc:  # pragma: no cover - 依赖具体机器
            last = exc
            print(f"[warn] scene {scene} 快照写入失败（第 {k + 1} 次）：{exc}",
                  flush=True)
            time.sleep(2.0 * (k + 1))
    raise last  # type: ignore[misc]


def _scene_row(cfg, args, scene: int, radius: float, target: int):
    m = int(cfg.scale.M)
    receivers = tuple(range(m))
    truth, belief0, base = _scene_with_retry(
        cfg, scene, args.out / "scenes" / f"scene_{scene:05d}.npz")
    belief, angle = _belief_for_radius(cfg, truth, belief0, target, radius, scene)
    belief_geom = belief.as_geometry(truth)
    true_delta = np.asarray(truth.p_tgt[target, :2], dtype=float) - np.asarray(
        belief_geom.p_tgt[target, :2], dtype=float)
    # 一次构造、两个假设共用：``build_observation_pair`` 的 H0 是从 H1 派生的，
    # 分两次构造会重复消耗同样的随机流（结果相同但白算一遍）。
    pairs = [
        [_pair(cfg, truth, belief_geom, base, args, scene, rx, look)
         for rx in receivers]
        for look in range(2)
    ]
    looks = [[pair[0] for pair in look] for look in pairs]
    looks_h0 = [[pair[1] for pair in look] for look in pairs]
    started = time.perf_counter()
    row = {
        "split": _role(scene, args),
        "scene_id": scene,
        "error_radius_m": float(radius),
        "error_angle_rad": float(angle),
        "true_delta_x_m": float(true_delta[0]),
        "true_delta_y_m": float(true_delta[1]),
    }
    for tag, hyp in (("h1", looks), ("h0", looks_h0)):
        nominal, _ = _arm_nominal(cfg, hyp, target)
        local, _ = _arm_local_crossfit(
            cfg, hyp, target, belief, receivers, belief_geom, args)
        if bool(getattr(args, "baseline_curve", False)):
            wls_args = argparse.Namespace(**vars(args))
            wls_args.solver = "jacobian_wls"
            wls_args.state_gate = None
            shared_args = argparse.Namespace(**vars(args))
            # The confirmatory curve freezes the original coarse-to-fine MAP.
            # Gated Top-K remains a developmental ablation because its apparent
            # mean gain is concentrated in a few scenes.
            shared_args.solver = "coarse_to_fine"
            wls, extra_wls = _arm_shared_crossfit(
                cfg, hyp, target, belief, receivers, belief_geom, wls_args)
            shared, extra = _arm_shared_crossfit(
                cfg, hyp, target, belief, receivers, belief_geom, shared_args)
        else:
            self_fit, extra_self = _arm_self_fit(
                cfg, hyp, target, belief, receivers, belief_geom, args)
            shared, extra = _arm_shared_crossfit(
                cfg, hyp, target, belief, receivers, belief_geom, args)
        oracle, _ = _arm_oracle(cfg, hyp, target, belief_geom, truth)
        row[f"nominal_{tag}"] = nominal
        if bool(getattr(args, "baseline_curve", False)):
            row[f"wls_map_{tag}"] = wls
        else:
            row[f"self_fit_{tag}"] = self_fit
        row[f"local_crossfit_{tag}"] = local
        row[f"shared_crossfit_{tag}"] = shared
        row[f"oracle_{tag}"] = oracle
        if tag == "h1":
            row["estimated_delta_a_x"] = float(extra["delta_a"][0])
            row["estimated_delta_a_y"] = float(extra["delta_a"][1])
            row["estimated_delta_b_x"] = float(extra["delta_b"][0])
            row["estimated_delta_b_y"] = float(extra["delta_b"][1])
            row["state_error_a_m"] = float(np.linalg.norm(
                np.asarray(extra["delta_a"]) - true_delta))
            row["state_error_b_m"] = float(np.linalg.norm(
                np.asarray(extra["delta_b"]) - true_delta))
            row["objective_a"] = float(extra["objective_a"])
            row["objective_b"] = float(extra["objective_b"])
            row["state_gate_accepted_a"] = bool(extra["accepted_a"])
            row["state_gate_accepted_b"] = bool(extra["accepted_b"])
            row["raw_delta_a_x"] = float(extra["raw_delta_a"][0])
            row["raw_delta_a_y"] = float(extra["raw_delta_a"][1])
            row["raw_delta_b_x"] = float(extra["raw_delta_b"][0])
            row["raw_delta_b_y"] = float(extra["raw_delta_b"][1])
            if bool(getattr(args, "baseline_curve", False)):
                row["wls_state_error_a_m"] = float(np.linalg.norm(
                    np.asarray(extra_wls["delta_a"]) - true_delta))
                row["wls_state_error_b_m"] = float(np.linalg.norm(
                    np.asarray(extra_wls["delta_b"]) - true_delta))
            else:
                row["self_fit_state_error_m"] = float(np.mean([
                    float(np.linalg.norm(d - true_delta))
                    for d in extra_self["self_delta"]
                ]))
    row["runtime_s"] = float(time.perf_counter() - started)
    return row


def _read_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _min_calibration(p_fa: float) -> int:
    """split_conformal 门限为有限值所需的最少 H0 标定场景数。

    ``bench._calibrated_threshold`` 取 ``rank = ceil((n+1)(1-p_fa))``，``rank > n``
    时直接返回 ``inf`` —— 于是 P_D 恒为 0、整张表失去意义。方案 §12 写的
    "10 条标定"在 p_fa=0.05 下会静默踩中这个坑（实测 Gate-0 smoke 五臂门限全
    为 inf），所以这里把它变成一条会说话的检查。
    """
    n = 1
    while math.ceil((n + 1) * (1.0 - float(p_fa))) > n:
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--error-m", default="0,150,400")
    parser.add_argument("--train-scenes", type=int, default=4)
    parser.add_argument("--calibration-scenes", type=int, default=20)
    parser.add_argument("--test-scenes", type=int, default=20)
    parser.add_argument("--master-seed", type=int, default=20261007)
    parser.add_argument("--area-xy", type=float, default=400.0)
    parser.add_argument("--uavs", type=int, default=6)
    parser.add_argument("--targets-count", type=int, default=3)
    parser.add_argument("--min-uav-separation-m", type=float, default=20.0,
                        help="UAV 节点间的三维硬最短距离（米）")
    parser.add_argument("--m-rx", type=int, default=4)
    parser.add_argument("--target-rcs", type=float, default=0.5)
    parser.add_argument("--prior-sigma-m", type=float, default=150.0)
    parser.add_argument("--search-sigma", type=float, default=4.0)
    parser.add_argument("--coarse-step-m", type=float, default=None,
                        help="粗网格间距（米）；默认 sigma/2，因为 G(delta) 的"
                             "相关峰宽只有 25--75 m")
    parser.add_argument("--solver", default="coarse_to_fine",
                        choices=["coarse_to_fine", "gated_topk", "jacobian_wls"])
    parser.add_argument("--screen-receivers", type=int, default=2,
                        help="gated_topk 第一档用几个接收机筛粗网格")
    parser.add_argument("--screen-keep", type=int, default=40)
    parser.add_argument("--verify-receivers", type=int, default=4,
                        help="第二档补到几个接收机复筛")
    parser.add_argument("--verify-keep", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=3,
                        help="最终保留几个峰做多峰细化")
    parser.add_argument("--glrt-radius", type=float, default=0.25)
    parser.add_argument("--glrt-points", type=int, default=3)
    parser.add_argument("--p-fa", type=float, default=0.05)
    parser.add_argument("--bootstrap", type=int, default=400)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--baseline-curve", action="store_true",
                        help="运行 nominal/WLS/local/shared-coarse/oracle 正式主线")
    parser.add_argument("--state-gate-config", type=Path, default=None,
                        help="已冻结的 shared cross-fit 状态回退门禁 JSON")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟档：1 训练 + 2 标定 + 2 测试。门限恒为 inf，"
                             "只能看字段齐全 / 无真值泄漏 / 状态恢复误差；"
                             "一个 (半径, 目标) 约 30 min，用来在投几小时之前"
                             "先确认新档位跑得通。")
    args = parser.parse_args()
    args.state_gate = None
    if args.state_gate_config is not None:
        payload = json.loads(args.state_gate_config.read_text(encoding="utf-8"))
        if payload.get("status") != "FROZEN":
            raise SystemExit("state gate config is not FROZEN")
        args.state_gate = payload
    if args.smoke:
        args.train_scenes = 1
        args.calibration_scenes = 2
        args.test_scenes = 2
    args.out.mkdir(parents=True, exist_ok=True)
    radii = tuple(float(v) for v in args.error_m.split(","))
    # 起飞前检查：真值偏移落在搜索盒外时，MAP 最多只能把误差拉到盒的角上，
    # 剩下的残差是**配置**造成的，不是机制失效。Gate 2 的 550 m 档就是这么
    # 崩的（3.7σ > span=450 m，只有对角线附近够得到）—— 这个 warning 本该在
    # 投 6.8 h 之前就把它叫出来。
    span = float(args.search_sigma) * float(args.prior_sigma_m)
    for radius in radii:
        if radius > span * np.sqrt(2.0) + 1e-9:
            print(f"WARNING: radius={radius:g} m 连搜索盒的对角 "
                  f"(span*sqrt2={span * np.sqrt(2.0):.0f} m) 都够不到 —— "
                  f"这一档上 MAP 无解，P_D 与状态误差都是配置伪影", file=sys.stderr)
        elif radius > span + 1e-9:
            print(f"WARNING: radius={radius:g} m > 搜索半宽 span={span:.0f} m —— "
                  f"真值只在靠近对角线的方向上够得到，预期残差 ~"
                  f"{radius - span * np.sqrt(2.0):.0f}--{radius:.0f} m 且方差很大。"
                  f"要么调大 --search-sigma，要么声明这一档是失配压力测试",
                  file=sys.stderr)
    need = _min_calibration(args.p_fa)
    if args.calibration_scenes < need:
        print(f"WARNING: calibration-scenes={args.calibration_scenes} < {need} "
              f"-> split_conformal 门限恒为 inf，P_D 全 0（p_fa={args.p_fa}）",
              file=sys.stderr)

    if not args.summary_only:
        cfg = _cfg(args)
        total = args.train_scenes + args.calibration_scenes + args.test_scenes
        scenes = [s for s in range(total) if s % max(int(args.shards), 1) == int(args.shard)]
        # 一个场景行 ~6--15 min，而整轮要跑一百多行。只在最后写一次 CSV 意味着
        # 崩一次全丢、且中途完全看不到进度 —— 所以每行立即追加落盘，并在启动时
        # 读回已完成的行续跑。
        partial = args.out / f"partial_shard{int(args.shard):02d}.csv"
        records = []
        done = set()
        if partial.exists():
            for row in _read_records(partial):
                records.append(row)
                done.add((float(row["error_radius_m"]), int(row["scene_id"])))
        for radius in radii:
            for scene in scenes:
                if (float(radius), int(scene)) in done:
                    continue
                started = time.perf_counter()
                row = _scene_row(cfg, args, scene, radius, args.target)
                records.append(row)
                with partial.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if partial.stat().st_size == 0:
                        writer.writeheader()
                    writer.writerow(row)
                print(f"[shard {args.shard}] radius={radius:g} scene={scene} "
                      f"{time.perf_counter() - started:.0f}s "
                      f"({len(records)}/{len(radii) * len(scenes)})", flush=True)
        shard_path = args.out / f"records_shard{int(args.shard):02d}.csv"
        with shard_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {len(records)} rows -> {shard_path}")
        if int(args.shards) == 1:
            rows = records
        else:
            return
        with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    else:
        rows = []
        for shard in range(int(args.shards)):
            named = args.out / f"records_shard{shard:02d}.csv"
            # 分片没跑完时退回它逐行追加的 partial 文件：宁可少几行，也不要整轮作废。
            rows.extend(_read_records(
                named if named.exists()
                else args.out / f"partial_shard{shard:02d}.csv"))
        if not rows:
            raise SystemExit("no shard records found")
        with (args.out / "records.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    evaluation = _summarise(rows, args)
    payload = {
        "protocol": {
            "master_seed": args.master_seed,
            "target": args.target,
            "receivers": "all",
            "error_radius_m": list(radii),
            "arms": list(_active_arms(args)),
            "baseline_curve": bool(args.baseline_curve),
            "arm": "tp_uic_full",
            "looks": 2,
            "solver": args.solver,
            "search_sigma": args.search_sigma,
            "prior_sigma_m": args.prior_sigma_m,
            "target_rcs": args.target_rcs,
            "min_uav_separation_m": args.min_uav_separation_m,
            "p_fa": args.p_fa,
            "bootstrap": args.bootstrap,
            "train_scenes": args.train_scenes,
            "calibration_scenes": args.calibration_scenes,
            "test_scenes": args.test_scenes,
            "note": "other targets' belief means are pinned to truth; only the "
                    "tested target carries the injected offset",
        },
        "evaluation": evaluation,
    }
    (args.out / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
