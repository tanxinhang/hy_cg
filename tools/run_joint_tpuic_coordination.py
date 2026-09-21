"""Semantic-reset experiment for TP-UIC / cooperation coupling.

This is the first executable slice of ``JOINT_TPUIC_COORDINATION_V1.md``.  It
tests the load-bearing coupling claim without pretending that the complete soft
receiver already exists:

Planning consumes belief-only absolute residual-power and target-retention
certificates by default; truth-conditioned measurements are gated as explicit
oracle diagnostics.  Capture discount and link redundancy are separate 2x2
factors, so their effects can be attributed instead of hidden in one "robust"
arm.  Receiver evaluation is held out by default.

The executable joint action is still only the illumination mask.  Accordingly,
the output calls this an illumination-feedback experiment, not full joint
optimization over links, fusion, power, protection and reference CPI.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from itertools import combinations
import json
import math
import os
import statistics as st
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isac_sim.receiver import cancellation as cx  # noqa: E402
from isac_sim.scenario.belief import (  # noqa: E402
    BeliefState,
    belief_capture_probability_lower_bound,
    belief_capture_sigma_points,
    belief_dd_std_bins,
    truth_captured_links,
)
from isac_sim.core.config import Config, apply_overrides, apply_preset  # noqa: E402
from experiments.coordination import illuminator_mask
from isac_sim.detection.fusion import predicted_pd_for_links  # noqa: E402
from experiments.exploratory.fusion_polish import maximize_fixed_set_pd  # noqa: E402
from experiments.exploratory.bundle_master import rcs_robust_bundle_column_generation  # noqa: E402
from isac_sim.sensing.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    radar_hardware_gain,
    rescale_sensing_tables_for_rcs,
)
from experiments.selection import feasible_links_for_target, local_cap_allows, processing_caps_allow, polish_risk_secondary_pd, polish_worst_target_pd, remote_cap_allows, select_c2f_adaptive
from experiments.flow.simulate import _build_plan, evaluate_detection, rng_for_detection  # noqa: E402
from audits.theory import task_objective  # noqa: E402


METHOD = "proposed_c2f_adaptive_pd"
FORMAL_CAPABILITY_SOURCE = "predicted"
FORMAL_RETENTION_SOURCE = "predicted_risk"


@dataclass
class StateEvaluation:
    cfg: Config
    mask: np.ndarray
    selected: Dict[int, List[tuple[int, int]]]
    selected_eval: Dict[int, List[tuple[int, int]]]
    tables_truth: object
    base_truth: object
    plan: object
    belief_objective: float
    truth_objective: float
    belief_pd: np.ndarray
    truth_pd: np.ndarray
    fraction: np.ndarray | None
    retention: np.ndarray | None
    kappa_db: np.ndarray | None
    eval_fraction: np.ndarray | None
    eval_retention: np.ndarray | None
    eval_kappa_db: np.ndarray | None
    measure_seconds: float
    capture_aware: bool
    redundancy_aware: bool
    risk_polish_history: List[dict]
    risk_secondary_history: List[dict]
    fusion_pd_polish: dict
    power_policy: str
    receiver_arm: str
    control_mu: np.ndarray | None


class _ControlledMeasurement:
    """Cell-wise view assembled from a belief-selected soft-TP-UIC bank."""

    def __init__(self, bank, choice: np.ndarray):
        self.bank = tuple(bank)
        self.choice = np.asarray(choice, dtype=int)
        self.i_in_pred = np.asarray(self.bank[0].i_in_pred, dtype=float)

    def _pick(self, values) -> np.ndarray:
        stack = np.stack([np.asarray(value, dtype=float) for value in values])
        return np.take_along_axis(stack, self.choice[None, :, :], axis=0)[0]

    def as_predicted_residual_power(self, *, per_target: bool = False):
        if not per_target:
            raise ValueError("controlled TP-UIC requires per-target residuals")
        return self._pick([
            item.as_predicted_residual_power(per_target=True)
            for item in self.bank
        ])

    def as_residual_power(self, *, per_target: bool = False):
        if not per_target:
            raise ValueError("controlled TP-UIC requires per-target residuals")
        return self._pick([
            item.as_residual_power(per_target=True) for item in self.bank
        ])

    def as_predicted_retention(self, *, clamp: bool = True, risk: bool = False):
        return self._pick([
            item.as_predicted_retention(clamp=clamp, risk=risk)
            for item in self.bank
        ])

    def as_retention(self, per_target=None, clamp=True, source="field"):
        if source != "per_target":
            raise ValueError("controlled TP-UIC requires per-target retention")
        return self._pick([
            item.as_retention(
                per_target=per_target, clamp=clamp, source="per_target"
            ) for item in self.bank
        ])


def build_config(args) -> Config:
    cfg = apply_preset(Config(), args.preset)
    overrides = {
        "scale.M": int(args.m),
        "scale.Q": int(args.q),
        "geometry.area_xy": float(args.area),
        "detect.target_rcs": float(args.rcs),
        "run.seed": int(args.seed),
        "run.verbose": False,
        "refine.enable": True,
        "selector.score_mode": "detector_pd",
        "coordination.enable": True,
        "coordination.rounds": int(args.rounds),
        "interference.sense_gate_by_active_tx": True,
        "cancellation.enable": True,
        "cancellation.n_cpi": 1,
        "cancellation.max_protected_targets": min(
            int(args.max_protected_targets), int(args.q)
        ),
        "cancellation.covariance_protection": bool(args.covariance_protection),
        "cancellation.adaptive_soft_enable": (
            str(args.tpuic_arm) == "adaptive_soft_tpuic"
            or str(args.evaluation_tpuic_arm) == "adaptive_soft_tpuic"
        ),
        "cancellation.adaptive_soft_risk_slack": float(
            args.adaptive_risk_slack
        ),
        "cancellation.belief_error_in_cres": bool(args.belief_error_in_cres),
        "aperture.enable": bool(args.m_rx > 1),
        "aperture.m_rx": int(args.m_rx),
        "fusion.rule": str(args.fusion_rule),
        "selector.max_links_per_target": int(args.max_links_per_target),
        "selector.max_total_links": int(args.max_total_links),
    }
    if float(args.processing_gain_multiplier) != 1.0:
        overrides["detect.sensing_processing_gain"] = float(
            cfg.waveform.N * cfg.waveform.L
        ) * float(args.processing_gain_multiplier)
    if getattr(args, "constraint_maxmin", False):
        # Greedy ranking needs a leximin refinement because pure min has zero
        # marginal until the weakest target changes.  Resources remain hard
        # constraints; the exact Phi=min_q P_D,q is enforced on acceptance.
        overrides.update({
            "selector.maxmin_objective": True,
            "selector.use_delay_price": False,
            "selector.lambda_c": 0.0,
        })
    if int(args.max_tx_nodes) > 0:
        overrides["selector.max_tx_nodes"] = int(args.max_tx_nodes)
    return apply_overrides(
        cfg,
        overrides,
    )


def _mask_code(mask: np.ndarray) -> int:
    code = 0
    for i, active in enumerate(np.asarray(mask, dtype=bool)):
        if active:
            code |= 1 << int(i)
    return code


def _finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("inf")


def select_receiver_control(candidates, weak_req: float):
    """Choose a TP-UIC control by exact belief-side max-min service."""
    return max(candidates, key=lambda state: (
        float(np.min(state.belief_pd)),
        float(np.sum(np.minimum(state.belief_pd, float(weak_req)))),
        -float(state.cfg.cancellation.soft_protection_mu),
    ))


def evaluate_state(
    cfg: Config,
    *,
    trial: int,
    geom_true,
    geom_belief,
    base_truth,
    base_belief,
    belief_dd_std,
    capture_probability: np.ndarray,
    capture_sigma_points: np.ndarray | None,
    mask: np.ndarray,
    receiver: str,
    capture_aware: bool = False,
    redundancy_aware: bool = False,
    min_links: int = 1,
    redundancy_policy: str = "diversity_first",
    capture_coverage_required: float = 0.95,
    selection_rcs_factor: float = 1.0,
    selector_strategy: str = "c2f",
    capability_reps: int = 1,
    capability_quantile: float = 0.20,
    capability_source: str = FORMAL_CAPABILITY_SOURCE,
    capability_retention_source: str = FORMAL_RETENTION_SOURCE,
    risk_swap_rounds: int = 0,
    risk_secondary_rounds: int = 0,
    risk_secondary_pair_exchange: bool = False,
    fusion_pd_polish: bool = False,
    power_policy: str = "uniform",
    active_rho: float = 0.95,
    inactive_rho: float = 0.20,
    heldout_capability: bool = False,
    capability_cache: dict | None = None,
    tpuic_arm: str = "tp_uic_full",
    evaluation_tpuic_arm: str | None = None,
    cell_soft_mu_grid: tuple[float, ...] = (),
) -> StateEvaluation:
    """Measure one receiver state, select on belief, and evaluate on truth."""
    mask = np.asarray(mask, dtype=bool)
    m, q = int(cfg.scale.M), int(cfg.scale.Q)
    if mask.shape != (m,):
        raise ValueError("illumination mask must have shape (%d,)" % m)
    if power_policy == "mask_role":
        rho = np.where(mask, float(active_rho), float(inactive_rho))
        cfg = apply_overrides(cfg, {
            "radio.rho_by_uav": tuple(float(v) for v in rho)
        })
    elif power_policy != "uniform":
        raise ValueError("power_policy must be 'uniform' or 'mask_role'")
    rho_vector = np.asarray(
        cfg.radio.rho_by_uav
        if cfg.radio.rho_by_uav is not None
        else [cfg.radio.rho] * m,
        dtype=float,
    )

    fraction = retention = kappa = secondary_power = None
    residual_power = None
    eval_fraction = eval_retention = eval_kappa = eval_residual_power = None
    measure_seconds = 0.0
    control_mu = None
    if receiver == "tpuic":
        cache_key = (
            receiver, str(tpuic_arm), str(evaluation_tpuic_arm), _mask_code(mask),
            float(cfg.cancellation.soft_protection_mu),
            int(capability_reps), float(capability_quantile),
            str(capability_source), str(capability_retention_source),
            bool(heldout_capability), bool(int(risk_secondary_rounds) > 0),
            tuple(float(value) for value in cell_soft_mu_grid),
            tuple(float(v) for v in rho_vector),
        )
        cached = None if capability_cache is None else capability_cache.get(cache_key)
        if cached is not None:
            (
                fraction, residual_power, retention, kappa,
                eval_fraction, eval_residual_power, eval_retention, eval_kappa,
                secondary_power, cached_control_mu,
            ) = (value.copy() for value in cached)
            control_mu = None if cached_control_mu.size == 0 else cached_control_mu
        else:
            sense = rho_vector * float(cfg.radio.P_default)
            processing_gain = (
                float(cfg.waveform.N * cfg.waveform.L)
                if cfg.detect.sensing_processing_gain is None
                else float(cfg.detect.sensing_processing_gain)
            )
            ctx = cx.ReceiverContext.from_trial(
                cfg,
                geom_true,
                geom_belief,
                base_truth,
                base_belief=base_belief,
                sense_power=sense,
                radiated_power=sense * mask,
                processing_gain=processing_gain,
                hw_gain=float(radar_hardware_gain(cfg)),
                active_mask=mask,
                arm=str(tpuic_arm),
            )
            eval_arm = str(evaluation_tpuic_arm or tpuic_arm)
            eval_ctx = replace(ctx, arm=eval_arm)
            t0 = time.perf_counter()
            measurements = []
            for rep in range(int(capability_reps)):
                seed = [cfg.run.seed, 2_000_000 + int(trial), _mask_code(mask)]
                if int(capability_reps) > 1:
                    seed.append(int(rep))
                measurements.append(cx.measure_receiver_context(
                    ctx,
                    rng=np.random.default_rng(seed),
                    all_targets=True,
                    residual_quantile_probability=(
                        1.0 - float(capability_quantile)
                        if (
                            capability_source == "prior_quantile"
                            or int(risk_secondary_rounds) > 0
                        ) else None
                    ),
                ))
            if (
                int(capability_reps) == 1
                and not heldout_capability
                and eval_arm == str(tpuic_arm)
            ):
                evaluated = measurements[0]
            else:
                evaluated = cx.measure_receiver_context(
                    eval_ctx,
                    rng=np.random.default_rng([
                        cfg.run.seed, 2_500_000 + int(trial), _mask_code(mask)
                    ]),
                    all_targets=True,
                    residual_quantile_probability=(
                        1.0 - float(capability_quantile)
                        if (
                            capability_source == "prior_quantile"
                            or int(risk_secondary_rounds) > 0
                        ) else None
                    ),
                )
            if cell_soft_mu_grid:
                plan_bank = []
                eval_bank = []
                for mu in cell_soft_mu_grid:
                    cfg_mu = apply_overrides(cfg, {
                        "cancellation.soft_protection_mu": float(mu),
                    })
                    ctx_mu = replace(ctx, cfg=cfg_mu, arm="soft_tpuic")
                    reps_mu = []
                    for rep in range(int(capability_reps)):
                        seed = [
                            cfg.run.seed, 2_000_000 + int(trial),
                            _mask_code(mask),
                        ]
                        if int(capability_reps) > 1:
                            seed.append(int(rep))
                        reps_mu.append(cx.measure_receiver_context(
                            ctx_mu,
                            rng=np.random.default_rng(seed),
                            all_targets=True,
                        ))
                    plan_bank.append(reps_mu)
                    eval_bank.append(cx.measure_receiver_context(
                        replace(ctx_mu, arm="soft_tpuic"),
                        rng=np.random.default_rng([
                            cfg.run.seed, 2_500_000 + int(trial),
                            _mask_code(mask),
                        ]),
                        all_targets=True,
                    ))
                predicted_power = np.stack([
                    np.quantile(np.stack([
                        item.as_predicted_residual_power(per_target=True)
                        for item in reps
                    ]), 1.0 - float(capability_quantile), axis=0)
                    for reps in plan_bank
                ])
                predicted_eta = np.stack([
                    np.quantile(np.stack([
                        item.as_predicted_retention(risk=True) for item in reps
                    ]), float(capability_quantile), axis=0)
                    for reps in plan_bank
                ])
                input_power = np.asarray(
                    plan_bank[0][0].i_in_pred, dtype=float
                )[None, :, None]
                residual_fraction = np.ones_like(predicted_power)
                np.divide(
                    predicted_power, input_power, out=residual_fraction,
                    where=input_power > 0.0,
                )
                score = np.log(np.clip(predicted_eta, cx.EPS, None)) - np.log(
                    np.clip(residual_fraction, cx.EPS, None)
                )
                choice = np.argmax(score, axis=0)
                mu_values = np.asarray(cell_soft_mu_grid, dtype=float)
                control_mu = mu_values[choice]
                measurements = [
                    _ControlledMeasurement(
                        [plan_bank[k][rep] for k in range(len(plan_bank))],
                        choice,
                    )
                    for rep in range(int(capability_reps))
                ]
                evaluated = _ControlledMeasurement(eval_bank, choice)
                tpuic_arm = eval_arm = "controlled_soft_tpuic"
            measure_seconds = time.perf_counter() - t0
            target_conditioned = (
                str(tpuic_arm).startswith("targeted_")
                or str(tpuic_arm) in (
                    "soft_tpuic", "adaptive_soft_tpuic", "controlled_soft_tpuic",
                )
            )
            eval_target_conditioned = (
                eval_arm.startswith("targeted_")
                or eval_arm in (
                    "soft_tpuic", "adaptive_soft_tpuic", "controlled_soft_tpuic",
                )
            )
            if capability_source == "none":
                # Certificate-free planning: the scheduler sees only geometry,
                # waveform, communication state and hard budgets.  TP-UIC still
                # runs for the independent evaluation below, but its certificate
                # must not leak into the planning tables.
                template = evaluated.as_residual_power(
                    per_target=eval_target_conditioned
                )
                residual_powers = np.stack([
                    np.zeros_like(template, dtype=float)
                ])
            elif capability_source == "target_feedback":
                # One conservative state per target, with receiver identity
                # removed before selection.  This can steer target service but
                # cannot rank individual (i,j,q) links like a certificate.
                raw = np.stack([
                    item.as_predicted_residual_power(per_target=True)
                    for item in measurements
                ])
                target_level = np.quantile(
                    raw, 1.0 - float(capability_quantile), axis=(0, 1)
                )
                residual_powers = np.stack([
                    np.broadcast_to(target_level, (m, q)).copy()
                ])
            elif capability_source == "controlled_state":
                residual_powers = np.stack([
                    item.as_predicted_residual_power(per_target=True)
                    for item in measurements
                ])
            elif capability_source in ("measured", "measured_model"):
                residual_powers = np.stack([
                    item.as_residual_power(per_target=target_conditioned)
                    for item in measurements
                ])
            elif capability_source == "risk_moment":
                residual_powers = np.stack([
                    item.as_risk_residual_power(
                        tail_probability=float(capability_quantile),
                        per_target=target_conditioned,
                    ) for item in measurements
                ])
            elif capability_source == "prior_quantile":
                residual_powers = np.stack([
                    item.as_prior_quantile_residual_power(
                        per_target=target_conditioned
                    ) for item in measurements
                ])
            elif capability_source == "predicted":
                residual_powers = np.stack([
                    item.as_predicted_residual_power(
                        per_target=target_conditioned
                    ) for item in measurements
                ])
            else:
                raise ValueError(
                    "capability_source must be 'none', 'target_feedback', "
                    "'controlled_state', "
                    "'measured', "
                    "'measured_model', 'risk_moment', 'prior_quantile' or "
                    "'predicted'"
                )
            if capability_retention_source == "none":
                retentions = np.stack([
                    np.ones_like(residual_powers[0], dtype=float)
                ])
            elif capability_retention_source == "target_feedback":
                raw = np.stack([
                    item.as_predicted_retention(risk=True)
                    for item in measurements
                ])
                target_level = np.quantile(
                    raw, float(capability_quantile), axis=(0, 1)
                )
                retentions = np.stack([
                    np.broadcast_to(target_level, (m, q)).copy()
                ])
            elif capability_retention_source == "controlled_state":
                retentions = np.stack([
                    item.as_predicted_retention(risk=True)
                    for item in measurements
                ])
            elif capability_retention_source == "measured":
                retentions = np.stack([
                    item.as_retention(source="per_target", per_target=q)
                    for item in measurements
                ])
            elif capability_retention_source in ("predicted", "predicted_risk"):
                retentions = np.stack([
                    item.as_predicted_retention(
                        risk=capability_retention_source == "predicted_risk"
                    ) for item in measurements
                ])
            else:
                raise ValueError(
                    "capability_retention_source must be 'none', "
                    "'target_feedback', 'controlled_state', 'measured', "
                    "'predicted' or 'predicted_risk'"
                )
            residual_power = np.quantile(
                residual_powers, 1.0 - float(capability_quantile), axis=0
            )
            model_input = np.asarray(measurements[0].i_in_pred, dtype=float)
            denominator = (
                model_input[:, None] if target_conditioned else model_input
            )
            fraction = np.ones_like(residual_power, dtype=float)
            np.divide(
                residual_power, denominator, out=fraction, where=denominator > 0.0
            )
            if int(risk_secondary_rounds) > 0:
                secondary_power = np.quantile(
                    np.stack([
                        item.as_prior_quantile_residual_power(
                            per_target=target_conditioned
                        ) for item in measurements
                    ]),
                    1.0 - float(capability_quantile), axis=0,
                )
            retention = np.quantile(
                retentions, float(capability_quantile), axis=0
            )
            with np.errstate(divide="ignore"):
                kappa = -10.0 * np.log10(np.clip(fraction, cx.EPS, None))
            eval_residual_power = evaluated.as_residual_power(
                per_target=eval_target_conditioned
            ).copy()
            eval_input = np.asarray(evaluated.i_in_pred, dtype=float)
            eval_denominator = (
                eval_input[:, None] if eval_target_conditioned else eval_input
            )
            eval_fraction = np.ones_like(eval_residual_power, dtype=float)
            np.divide(
                eval_residual_power,
                eval_denominator,
                out=eval_fraction,
                where=eval_denominator > 0.0,
            )
            with np.errstate(divide="ignore"):
                eval_kappa = -10.0 * np.log10(
                    np.clip(eval_fraction, cx.EPS, None)
                )
            eval_retention = evaluated.as_retention(
                source="per_target", per_target=q
            ).copy()
            if capability_cache is not None:
                capability_cache[cache_key] = (
                    fraction.copy(), residual_power.copy(), retention.copy(),
                    kappa.copy(), eval_fraction.copy(),
                    eval_residual_power.copy(), eval_retention.copy(),
                    eval_kappa.copy(),
                    secondary_power.copy() if secondary_power is not None
                    else np.asarray([], dtype=float),
                    control_mu.copy() if control_mu is not None
                    else np.asarray([], dtype=float),
                )
    elif receiver != "constant":
        raise ValueError("receiver must be 'constant' or 'tpuic'")

    def build_tables(base, *, dd_gain=None, evaluation: bool = False):
        used_power = eval_residual_power if evaluation else residual_power
        used_retention = eval_retention if evaluation else retention
        return compute_link_tables(
            cfg,
            base,
            dd_gain=dd_gain,
            active_tx_mask=mask,
            residual_power_by_receiver_target=used_power,
            target_retention_by_receiver=used_retention,
        )

    coarse_gain = base_belief.dd_frac_loss
    if capture_aware:
        coarse_gain = coarse_gain * capture_probability
    tables_coarse = build_tables(base_belief, dd_gain=coarse_gain)
    selection_coarse = (
        tables_coarse
        if float(selection_rcs_factor) == 1.0
        else rescale_sensing_tables_for_rcs(
            cfg, tables_coarse, float(selection_rcs_factor)
        )
    )
    # The custom builder closes over both the receiver capability and mask, so
    # the fine replay cannot fall back to an all-on/fixed-kappa world.
    def refined_builder(cfg_, base_, dd_gain=None, **_kwargs):
        robust_gain = dd_gain
        if capture_aware:
            robust_gain = base_.eta_fine if robust_gain is None else robust_gain
            robust_gain = robust_gain * capture_probability
        rebuilt = compute_link_tables(
            cfg_,
            base_,
            dd_gain=robust_gain,
            active_tx_mask=mask,
            residual_power_by_receiver_target=residual_power,
            target_retention_by_receiver=retention,
        )
        return (
            rebuilt
            if float(selection_rcs_factor) == 1.0
            else rescale_sensing_tables_for_rcs(
                cfg_, rebuilt, float(selection_rcs_factor)
            )
        )

    belief_fine_gain = base_belief.eta_fine
    if capture_aware:
        belief_fine_gain = belief_fine_gain * capture_probability
    tables_belief = build_tables(base_belief, dd_gain=belief_fine_gain)
    tables_risk = None
    if int(risk_secondary_rounds) > 0 and receiver == "tpuic":
        if secondary_power is None or not np.size(secondary_power):
            raise RuntimeError("risk-secondary polish requires a prior quantile table")
        tables_risk = compute_link_tables(
            cfg,
            base_belief,
            dd_gain=belief_fine_gain,
            active_tx_mask=mask,
            residual_power_by_receiver_target=secondary_power,
            target_retention_by_receiver=retention,
        )
        if float(selection_rcs_factor) != 1.0:
            tables_risk = rescale_sensing_tables_for_rcs(
                cfg, tables_risk, float(selection_rcs_factor)
            )

    def c2f_solution():
        plan = _build_plan(cfg, base_belief, selection_coarse, geom_belief)
        selected, _D, _stats = select_c2f_adaptive(
            cfg,
            base_belief,
            selection_coarse,
            plan=plan,
            distributed_bids=True,
            refined_table_builder=refined_builder,
        )
        return plan, selected

    if selector_strategy == "c2f":
        plan, selected = c2f_solution()
    elif selector_strategy in ("rcs_bundle", "hybrid_bundle"):
        active = tuple(int(node) for node in np.flatnonzero(mask))
        max_tx = cfg.selector.max_tx_nodes
        allowed_sets = [active]
        if max_tx is not None and len(active) > int(max_tx):
            if selector_strategy == "hybrid_bundle":
                _seed_plan, seed_selected = c2f_solution()
                seed_tx = tuple(sorted({
                    int(link[0])
                    for links in seed_selected.values() for link in links
                }))
                if not seed_tx:
                    seed_tx = active[:int(max_tx)]
                allowed_sets = [seed_tx]
            else:
                allowed_sets = list(combinations(active, int(max_tx)))
        candidates = [
            rcs_robust_bundle_column_generation(
                cfg,
                base_belief,
                tables_coarse,
                value_tables=tables_belief,
                allowed_transmitters=allowed,
            )
            for allowed in allowed_sets
        ]
        objective_order = (
            "worst_detection_deficit",
            "total_detection_deficit",
            "remote_reports",
            "processing_load",
            "cpu_cycles",
        )
        chosen = min(
            candidates,
            key=lambda item: tuple(item.objective[key] for key in objective_order),
        )
        plan = chosen.plan
        selected = chosen.selected
    else:
        raise ValueError("unknown selector_strategy %r" % selector_strategy)

    # The ordinary marginal-utility rule is allowed to leave a target with a
    # single brittle observation.  The robust arm explicitly reserves a small
    # target-level redundancy floor, while still honoring all configured hard
    # resource caps.  This is an experiment-layer mechanism until promotion.
    if redundancy_aware and (
        min_links > 1
        or redundancy_policy in ("sigma_required", "sigma_singleton")
    ):
        total = sum(len(links) for links in selected.values())
        active_selected_tx = {
            int(link[0]) for links in selected.values() for link in links
        }
        max_tx = cfg.selector.max_tx_nodes
        for target in range(q):
            def sigma_coverage_count(links: List[tuple[int, int]]) -> int:
                if capture_sigma_points is None:
                    return 0
                covered = np.zeros(
                    int(capture_sigma_points.shape[0]), dtype=bool
                )
                for i_sel, j_sel in links:
                    covered |= capture_sigma_points[
                        :, int(i_sel), int(j_sel), target
                    ]
                return int(np.sum(covered))

            required_coverage = int(math.ceil(
                float(capture_coverage_required)
                * (0 if capture_sigma_points is None
                   else int(capture_sigma_points.shape[0]))
            ))

            def needs_more() -> bool:
                if redundancy_policy == "sigma_required":
                    return sigma_coverage_count(selected[target]) < required_coverage
                if redundancy_policy == "sigma_singleton":
                    # The certificate answers whether one observation is
                    # brittle.  Once an independently selected complement is
                    # present, stop instead of trying to prove continuous
                    # ellipsoid coverage with a finite directional design.
                    return (
                        len(selected[target]) < 2
                        and sigma_coverage_count(selected[target])
                        < required_coverage
                    )
                return len(selected[target]) < min(
                    int(min_links), cfg.selector.max_links_per_target
                )

            while (
                needs_more()
                and len(selected[target]) < cfg.selector.max_links_per_target
                and total < cfg.selector.max_total_links
            ):
                best_link = None
                best_pd = -np.inf
                best_diversity = -1
                best_coverage = -1
                best_meets_coverage = -1
                used_nodes = {
                    int(node)
                    for chosen in selected[target]
                    for node in chosen
                }
                for link in feasible_links_for_target(
                    cfg, base_belief, tables_belief, target, plan
                ):
                    if link in selected[target] or not mask[int(link[0])]:
                        continue
                    if (
                        max_tx is not None
                        and int(link[0]) not in active_selected_tx
                        and len(active_selected_tx) >= int(max_tx)
                    ):
                        continue
                    if not local_cap_allows(cfg, selected[target], link, target, plan):
                        continue
                    if not remote_cap_allows(cfg, selected, link, target, plan):
                        continue
                    if not processing_caps_allow(cfg, selected, link, target, plan):
                        continue
                    pd = predicted_pd_for_links(
                        cfg, tables_belief, target, selected[target] + [link],
                        plan=plan, base=base_belief,
                    )
                    diversity = sum(int(node) not in used_nodes for node in link)
                    improves = (
                        diversity > best_diversity
                        or (diversity == best_diversity and pd > best_pd)
                    )
                    if redundancy_policy == "pd_first":
                        improves = (
                            pd > best_pd + 1e-15
                            or (
                                abs(pd - best_pd) <= 1e-15
                                and diversity > best_diversity
                            )
                        )
                    elif redundancy_policy in (
                        "sigma_coverage", "sigma_required", "sigma_singleton"
                    ):
                        if capture_sigma_points is None:
                            raise ValueError(
                                "sigma_coverage needs joint capture sigma points"
                            )
                        covered = np.zeros(
                            int(capture_sigma_points.shape[0]), dtype=bool
                        )
                        for i_sel, j_sel in selected[target] + [link]:
                            covered |= capture_sigma_points[
                                :, int(i_sel), int(j_sel), target
                            ]
                        coverage = int(np.sum(covered))
                        if redundancy_policy == "sigma_singleton":
                            meets_coverage = int(
                                coverage >= required_coverage
                            )
                            # Coverage is a safety constraint, not the utility
                            # objective.  Once a candidate meets it, retain the
                            # established diversity/PD ordering.  Only when no
                            # candidate meets it do we maximize partial cover.
                            score = (
                                meets_coverage,
                                0 if meets_coverage else coverage,
                                diversity,
                                float(pd),
                            )
                            best_score = (
                                best_meets_coverage,
                                0 if best_meets_coverage > 0
                                else best_coverage,
                                best_diversity,
                                best_pd,
                            )
                            improves = score > best_score
                        else:
                            meets_coverage = -1
                            improves = (
                                coverage > best_coverage
                                or (
                                    coverage == best_coverage
                                    and diversity > best_diversity
                                )
                                or (
                                    coverage == best_coverage
                                    and diversity == best_diversity
                                    and pd > best_pd
                                )
                            )
                    elif redundancy_policy != "diversity_first":
                        raise ValueError(
                            "unknown redundancy_policy"
                        )
                    if improves:
                        best_diversity = diversity
                        if redundancy_policy in (
                            "sigma_coverage", "sigma_required", "sigma_singleton"
                        ):
                            best_coverage = coverage
                            best_meets_coverage = meets_coverage
                        best_pd, best_link = float(pd), link
                if best_link is None:
                    break
                selected[target].append(best_link)
                active_selected_tx.add(int(best_link[0]))
                total += 1

    risk_polish_history: List[dict] = []
    if int(risk_swap_rounds) > 0:
        selected, risk_polish_history = polish_worst_target_pd(
            cfg,
            base_belief,
            tables_belief,
            selected,
            plan=plan,
            allowed_tx_mask=mask,
            min_links_per_target=(int(min_links) if redundancy_aware else 1),
            max_rounds=int(risk_swap_rounds),
        )

    risk_secondary_history: List[dict] = []
    if int(risk_secondary_rounds) > 0 and receiver == "tpuic":
        selected, risk_secondary_history = polish_risk_secondary_pd(
            cfg,
            base_belief,
            tables_belief,
            tables_risk,
            selected,
            plan=plan,
            allowed_tx_mask=mask,
            min_links_per_target=(int(min_links) if redundancy_aware else 1),
            max_rounds=int(risk_secondary_rounds),
            allow_target_pair_exchange=bool(risk_secondary_pair_exchange),
        )

    fusion_pd_certificate: dict = {}
    if fusion_pd_polish:
        polished = maximize_fixed_set_pd(
            cfg, base_belief, tables_belief, selected, plan
        )
        plan = polished.plan
        fusion_pd_certificate = {
            "before": polished.predicted_before.tolist(),
            "after": polished.predicted_after.tolist(),
            "reports_before": polished.reports_before.tolist(),
            "reports_after": polished.reports_after.tolist(),
            "candidate_evaluations": int(polished.candidate_evaluations),
        }

    max_tx = cfg.selector.max_tx_nodes
    selected_tx = {int(link[0]) for links in selected.values() for link in links}
    if max_tx is not None and len(selected_tx) > int(max_tx):
        raise RuntimeError(
            "selection violated selector.max_tx_nodes: %d > %d"
            % (len(selected_tx), int(max_tx))
        )

    tables_truth = build_tables(
        base_truth, dd_gain=base_truth.eta_fine, evaluation=True
    )
    selected_eval = truth_captured_links(
        cfg, base_truth, base_belief, selected, belief_dd_std
    )

    belief_pd = np.asarray(
        [
            predicted_pd_for_links(
                cfg, tables_belief, target, selected.get(target, []),
                plan=plan, base=base_belief,
            )
            for target in range(q)
        ],
        dtype=float,
    )
    truth_pd = np.asarray(
        [
            predicted_pd_for_links(
                cfg, tables_truth, target, selected_eval.get(target, []),
                plan=plan, base=base_truth,
            )
            for target in range(q)
        ],
        dtype=float,
    )
    return StateEvaluation(
        cfg=cfg,
        mask=mask.copy(),
        selected=selected,
        selected_eval=selected_eval,
        tables_truth=tables_truth,
        base_truth=base_truth,
        plan=plan,
        belief_objective=float(
            task_objective(cfg, tables_belief, selected, plan, base_belief)
        ),
        truth_objective=float(
            task_objective(cfg, tables_truth, selected_eval, plan, base_truth)
        ),
        belief_pd=belief_pd,
        truth_pd=truth_pd,
        fraction=fraction,
        retention=retention,
        kappa_db=kappa,
        eval_fraction=eval_fraction,
        eval_retention=eval_retention,
        eval_kappa_db=eval_kappa,
        measure_seconds=float(measure_seconds),
        capture_aware=bool(capture_aware),
        redundancy_aware=bool(redundancy_aware),
        risk_polish_history=risk_polish_history,
        risk_secondary_history=risk_secondary_history,
        fusion_pd_polish=fusion_pd_certificate,
        power_policy=str(power_policy),
        receiver_arm=str(tpuic_arm if receiver == "tpuic" else receiver),
        control_mu=None if control_mu is None else control_mu.copy(),
    )


def strict_joint_update(
    cfg: Config,
    initial: StateEvaluation,
    evaluate,
    *,
    rounds: int,
    epsilon: float,
) -> tuple[StateEvaluation, List[dict], str]:
    """Monotone illumination feedback; return the best accepted state.

    With ``selector.maxmin_objective`` enabled, leximin proposes candidates but
    the exact ``Phi=min_q P_D,q`` decides acceptance.  This keeps the proposal
    mechanism separate from the formal max-min control objective.
    """
    state = initial
    history: List[dict] = []
    termination = "round_budget"
    for r in range(max(int(rounds), 0)):
        candidate_mask = illuminator_mask(state.selected, int(cfg.scale.M))
        objective_before = (
            float(np.min(state.belief_pd))
            if cfg.selector.maxmin_objective
            else float(state.belief_objective)
        )
        record = {
            "round": int(r),
            "mask_before": np.flatnonzero(state.mask).tolist(),
            "mask_candidate": np.flatnonzero(candidate_mask).tolist(),
            "objective_before": objective_before,
        }
        if np.array_equal(candidate_mask, state.mask):
            record.update(accepted=False, reason="fixed_point")
            history.append(record)
            termination = "fixed_point"
            break
        candidate = evaluate(candidate_mask)
        objective_candidate = (
            float(np.min(candidate.belief_pd))
            if cfg.selector.maxmin_objective
            else float(candidate.belief_objective)
        )
        gain = float(objective_candidate - objective_before)
        record.update(
            objective_candidate=objective_candidate,
            gain=gain,
            accepted=bool(gain >= float(epsilon)),
            measure_seconds=float(candidate.measure_seconds),
        )
        if gain < float(epsilon):
            record["reason"] = "insufficient_gain"
            history.append(record)
            termination = "strict_reject"
            break
        record["reason"] = "strict_improvement"
        history.append(record)
        state = candidate
    return state, history, termination


def _detect(
    cfg: Config,
    state: StateEvaluation,
    trial: int,
    *,
    exact_gaussian_threshold: bool = False,
    false_per_target: int | None = None,
    h1_per_target: int | None = None,
) -> tuple[float, float]:
    detect_cfg = state.cfg
    detect_overrides = {}
    if exact_gaussian_threshold:
        detect_overrides["detect.exact_gaussian_replacement_threshold"] = True
    if false_per_target is not None:
        detect_overrides["detect.num_false_per_target"] = int(false_per_target)
    if h1_per_target is not None:
        detect_overrides["detect.num_h1_per_target"] = int(h1_per_target)
    if detect_overrides:
        detect_cfg = apply_overrides(state.cfg, detect_overrides)
    got = evaluate_detection(
        detect_cfg,
        state.tables_truth,
        state.selected_eval,
        rng_for_detection(detect_cfg, int(trial)),
        METHOD,
        state.plan,
        state.base_truth,
        trial_index=int(trial),
    )
    detected, total, false_active, total_false_active = got[:4]
    return (
        float(detected / max(total, 1)),
        float(false_active / max(total_false_active, 1)),
    )


def _row(
    cfg: Config,
    trial: int,
    arm: str,
    state: StateEvaluation,
    history: List[dict],
    termination: str,
    selection_rcs_factor: float,
    selector_strategy: str,
    exact_gaussian_threshold: bool,
    false_per_target: int | None,
    h1_per_target: int | None,
    capability_source: str,
    capability_retention_source: str,
) -> dict:
    p_d, p_fa = _detect(
        cfg,
        state,
        trial,
        exact_gaussian_threshold=exact_gaussian_threshold,
        false_per_target=false_per_target,
        h1_per_target=h1_per_target,
    )
    return {
        "trial": int(trial),
        "arm": arm,
        "p_d": p_d,
        "p_fa": p_fa,
        "belief_objective": float(state.belief_objective),
        "truth_objective": float(state.truth_objective),
        "belief_worst_pd": float(np.min(state.belief_pd)),
        "truth_worst_pd": float(np.min(state.truth_pd)),
        "truth_mean_pd": float(np.mean(state.truth_pd)),
        "belief_pd_json": json.dumps(
            np.asarray(state.belief_pd, dtype=float).tolist(), separators=(",", ":")
        ),
        "truth_pd_json": json.dumps(
            np.asarray(state.truth_pd, dtype=float).tolist(), separators=(",", ":")
        ),
        "n_tx": int(np.sum(state.mask)),
        "n_links": int(sum(len(v) for v in state.selected.values())),
        "n_captured": int(sum(len(v) for v in state.selected_eval.values())),
        "kappa_db_median": (
            _finite_median(state.kappa_db)
            if state.kappa_db is not None
            else 0.0  # 未接线 = 没有对消（direct_cancellation_db 已删除）
        ),
        "eta_median": (
            float(np.median(state.retention))
            if state.retention is not None else 1.0
        ),
        "eval_kappa_db_median": (
            _finite_median(state.eval_kappa_db)
            if state.eval_kappa_db is not None
            else 0.0  # 未接线 = 没有对消
        ),
        "eval_eta_median": (
            float(np.median(state.eval_retention))
            if state.eval_retention is not None else 1.0
        ),
        "fraction_json": json.dumps(
            None if state.fraction is None else np.asarray(state.fraction).tolist(),
            separators=(",", ":"),
        ),
        "retention_json": json.dumps(
            None if state.retention is None else np.asarray(state.retention).tolist(),
            separators=(",", ":"),
        ),
        "eval_fraction_json": json.dumps(
            None if state.eval_fraction is None
            else np.asarray(state.eval_fraction).tolist(),
            separators=(",", ":"),
        ),
        "eval_retention_json": json.dumps(
            None if state.eval_retention is None
            else np.asarray(state.eval_retention).tolist(),
            separators=(",", ":"),
        ),
        "joint_rounds": int(len(history)),
        "accepted_rounds": int(sum(bool(h.get("accepted")) for h in history)),
        "termination": termination,
        "measure_seconds": float(
            state.measure_seconds
            + sum(float(h.get("measure_seconds", 0.0)) for h in history)
        ),
        "capture_aware": bool(state.capture_aware),
        "redundancy_aware": bool(state.redundancy_aware),
        "selection_rcs_factor": float(selection_rcs_factor),
        "selector_strategy": str(selector_strategy),
        "fusion_rule": str(state.cfg.fusion.rule),
        "exact_gaussian_threshold": bool(exact_gaussian_threshold),
        "capability_source": str(capability_source),
        "capability_retention_source": str(capability_retention_source),
        "risk_polish_swaps": int(len(state.risk_polish_history)),
        "risk_polish_json": json.dumps(
            state.risk_polish_history, separators=(",", ":")
        ),
        "risk_secondary_swaps": int(len(state.risk_secondary_history)),
        "risk_secondary_json": json.dumps(
            state.risk_secondary_history, separators=(",", ":")
        ),
        "fusion_pd_polish_json": json.dumps(
            state.fusion_pd_polish, separators=(",", ":")
        ),
        "rho_mean": float(np.mean(
            state.cfg.radio.rho_by_uav
            if state.cfg.radio.rho_by_uav is not None
            else [state.cfg.radio.rho] * int(state.cfg.scale.M)
        )),
        "rho_vector_json": json.dumps(
            list(state.cfg.radio.rho_by_uav)
            if state.cfg.radio.rho_by_uav is not None
            else [state.cfg.radio.rho] * int(state.cfg.scale.M),
            separators=(",", ":"),
        ),
        "power_policy": str(state.power_policy),
        "soft_mu": (
            float(state.cfg.cancellation.soft_protection_mu)
            if state.receiver_arm == "soft_tpuic"
            else float("nan")
        ),
        "control_mu_json": json.dumps(
            None if state.control_mu is None
            else np.asarray(state.control_mu, dtype=float).tolist(),
            separators=(",", ":"),
        ),
        "history_json": json.dumps(history, separators=(",", ":")),
        "selected_json": json.dumps(state.selected, separators=(",", ":")),
        "selected_eval_json": json.dumps(state.selected_eval, separators=(",", ":")),
        "fusion_nodes_json": json.dumps(
            np.asarray(state.plan.f_q, dtype=int).tolist(), separators=(",", ":")
        ),
    }


def run_trial(cfg: Config, args, trial: int) -> List[dict]:
    rng = np.random.default_rng([cfg.run.seed, int(trial)])
    geom_true = generate_geometry(cfg, rng)
    base_truth = build_base_gains(cfg, geom_true, rng)
    belief = BeliefState.from_truth(cfg, geom_true, rng)
    geom_belief = belief.as_geometry(geom_true)
    base_belief = build_base_gains(
        cfg,
        geom_belief,
        rng,
        channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    belief_std = belief_dd_std_bins(cfg, geom_belief, belief)
    capture_probability = belief_capture_probability_lower_bound(cfg, belief_std)
    capture_sigma = belief_capture_sigma_points(cfg, geom_belief, belief)
    if args.oracle_geometry_planning:
        geom_belief = geom_true
        base_belief = base_truth
        belief_std = np.zeros_like(belief_std, dtype=float)
        capture_probability = np.ones_like(capture_probability, dtype=float)
    all_on = np.ones(int(cfg.scale.M), dtype=bool)
    capability_cache = {}

    soft_mu_grid = tuple(float(v) for v in args.soft_mu_grid.split(",") if v.strip())

    def evaluator(
        receiver: str, capture_aware: bool, redundancy_aware: bool
    ):
        def evaluate_one(mask, control_cfg, arm):
            state = evaluate_state(
                control_cfg,
                trial=trial,
                geom_true=geom_true,
                geom_belief=geom_belief,
                base_truth=base_truth,
                base_belief=base_belief,
                belief_dd_std=belief_std,
                capture_probability=capture_probability,
                capture_sigma_points=capture_sigma,
                mask=mask,
                receiver=receiver,
                capture_aware=capture_aware,
                redundancy_aware=redundancy_aware,
                min_links=args.min_links,
                redundancy_policy=args.redundancy_policy,
                capture_coverage_required=args.capture_coverage_required,
                selection_rcs_factor=args.selection_rcs_factor,
                selector_strategy=args.selector_strategy,
                capability_reps=args.capability_reps,
                capability_quantile=args.capability_quantile,
                capability_source=args.capability_source,
                capability_retention_source=args.capability_retention_source,
                risk_swap_rounds=args.risk_swap_rounds,
                risk_secondary_rounds=args.risk_secondary_rounds,
                risk_secondary_pair_exchange=args.risk_secondary_pair_exchange,
                fusion_pd_polish=args.fusion_pd_polish,
                power_policy=args.power_policy,
                active_rho=args.active_rho,
                inactive_rho=args.inactive_rho,
                heldout_capability=args.heldout_capability,
                capability_cache=capability_cache,
                tpuic_arm=arm,
                evaluation_tpuic_arm=(args.evaluation_tpuic_arm or arm),
                cell_soft_mu_grid=tuple(
                    float(value) for value in args.cell_soft_mu_grid.split(",")
                    if value.strip()
                ),
            )
            return state

        def evaluate(mask):
            if receiver != "tpuic" or not soft_mu_grid:
                return evaluate_one(mask, cfg, args.tpuic_arm)
            candidates = []
            for mu in soft_mu_grid:
                control_cfg = apply_overrides(cfg, {
                    "cancellation.soft_protection_mu": float(mu),
                })
                candidates.append(evaluate_one(mask, control_cfg, "soft_tpuic"))
            return select_receiver_control(
                candidates, float(cfg.detect.weak_pd_required)
            )
        return evaluate

    rows = []
    variants = (
        ("baseline", False, False),
        ("capture", True, False),
        ("redundancy", False, True),
        ("combined", True, True),
    )
    for receiver in ("constant", "tpuic"):
        for variant, capture_aware, redundancy_aware in variants:
            evaluate = evaluator(receiver, capture_aware, redundancy_aware)
            serial = evaluate(all_on)
            joint_initial = serial
            if cfg.selector.max_tx_nodes is not None:
                # ``all_on`` is an intentionally unconstrained serial baseline,
                # not a feasible initial state of the capped joint problem.
                # Project once into the hard feasible set; strict-improvement
                # admission applies only between feasible states thereafter.
                feasible_mask = illuminator_mask(serial.selected, int(cfg.scale.M))
                joint_initial = evaluate(feasible_mask)
            joint, history, term = strict_joint_update(
                cfg, joint_initial, evaluate,
                rounds=args.rounds, epsilon=args.epsilon,
            )
            rows.append(_row(
                cfg, trial, receiver + "_serial_" + variant, serial, [], "serial",
                args.selection_rcs_factor,
                args.selector_strategy,
                args.exact_gaussian_threshold,
                args.false_per_target,
                args.h1_per_target,
                args.capability_source,
                args.capability_retention_source,
            ))
            rows.append(_row(
                cfg, trial, receiver + "_joint_" + variant, joint, history, term,
                args.selection_rcs_factor,
                args.selector_strategy,
                args.exact_gaussian_threshold,
                args.false_per_target,
                args.h1_per_target,
                args.capability_source,
                args.capability_retention_source,
            ))
    return rows


def _paired(rows: List[dict], a: str, b: str, key: str) -> tuple[float, float, int]:
    av = {int(r["trial"]): float(r[key]) for r in rows if r["arm"] == a}
    bv = {int(r["trial"]): float(r[key]) for r in rows if r["arm"] == b}
    ids = sorted(set(av) & set(bv))
    if not ids:
        return float("nan"), float("nan"), 0
    delta = np.asarray([bv[i] - av[i] for i in ids], dtype=float)
    se = float(np.std(delta, ddof=1) / math.sqrt(delta.size)) if delta.size > 1 else float("nan")
    return float(np.mean(delta)), se, int(delta.size)


def capability_protocol_error(args) -> str | None:
    """Return a semantic-protocol violation, independent of argparse.

    Keeping this rule in an importable function makes truth/belief isolation a
    system invariant that tests can pin, rather than a convention in a README.
    """
    if (args.capability_source == "none") != (
        args.capability_retention_source == "none"
    ):
        return (
            "certificate-free planning requires both capability sources to "
            "be 'none'"
        )
    if (args.capability_source == "target_feedback") != (
        args.capability_retention_source == "target_feedback"
    ):
        return (
            "target-level feedback requires both capability sources to be "
            "'target_feedback'"
        )
    if (args.capability_source == "controlled_state") != (
        args.capability_retention_source == "controlled_state"
    ):
        return (
            "receiver-target control requires both capability sources to be "
            "'controlled_state'"
        )
    if args.capability_source == "controlled_state" and not args.cell_soft_mu_grid:
        return "controlled_state requires --cell-soft-mu-grid"
    if args.capability_source == "none" and args.risk_secondary_rounds:
        return (
            "certificate-free planning cannot use prior-quantile secondary "
            "swaps"
        )
    oracle_sources = (
        args.capability_source in ("measured", "measured_model")
        or args.capability_retention_source == "measured"
    )
    if oracle_sources and not args.allow_oracle_planning:
        return (
            "measured planning capability is truth-conditioned; use predicted "
            "sources for formal experiments or pass --allow-oracle-planning "
            "for a labelled oracle diagnostic"
        )
    if oracle_sources and args.capability_reps < 2:
        return (
            "stochastic measured planning certificates require at least two "
            "--capability-reps; a one-sample quantile is not a certificate"
        )
    if not args.heldout_capability:
        return (
            "planning and evaluation capability must be independent; use "
            "--no-heldout-capability only in legacy reproduction, not this "
            "formal experiment"
        )
    return None


def report(rows: List[dict], args) -> None:
    print("\n=== TP-UIC / cooperation semantic-reset pilot ===")
    print("  M=%d Q=%d trials=%d rounds=%d epsilon=%g n_cpi=1" % (
        args.m, args.q, args.trials, args.rounds, args.epsilon
    ))
    print("  %-18s %8s %9s %9s %8s %8s %8s" % (
        "arm", "kappa", "eta", "P_D", "P_FA", "TX", "worstPD"
    ))
    arms = [
        receiver + "_" + mode + "_" + variant
        for receiver in ("constant", "tpuic")
        for variant in ("baseline", "capture", "redundancy", "combined")
        for mode in ("serial", "joint")
    ]
    for arm in arms:
        sub = [r for r in rows if r["arm"] == arm]
        mean = lambda key: float(np.mean([float(r[key]) for r in sub]))
        print("  %-18s %8.2f %9.4f %9.4f %8.4f %8.2f %8.4f" % (
            arm,
            st.median(float(r["kappa_db_median"]) for r in sub),
            st.median(float(r["eta_median"]) for r in sub),
            mean("p_d"), mean("p_fa"), mean("n_tx"), mean("truth_worst_pd"),
        ))
    print("\n  paired deltas, illumination feedback - serial "
          "(mean +/- 1 s.e.):")
    for receiver in ("constant", "tpuic"):
        for variant in ("baseline", "capture", "redundancy", "combined"):
            for key in ("p_d", "truth_worst_pd", "truth_objective"):
                d, se, n = _paired(
                    rows,
                    receiver + "_serial_" + variant,
                    receiver + "_joint_" + variant,
                    key,
                )
                print("    %-8s %-6s %-18s %+.5f +/- %s (n=%d)" % (
                    receiver, variant, key, d,
                    "%.5f" % se if math.isfinite(se) else "NA", n,
                ))
    print("\n  This run is a pilot unless its paired sample size and promotion gates")
    print("  are sufficient; it does not establish full joint optimization,")
    print("  soft TP-UIC benefit, or N_ref scaling.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--preset", default="paper-canonical")
    ap.add_argument("--area", type=float, default=600.0)
    ap.add_argument("--rcs", type=float, default=0.1)
    ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--q", type=int, default=3)
    ap.add_argument("--m-rx", type=int, default=1)
    ap.add_argument("--processing-gain-multiplier", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--trial-start", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epsilon", type=float, default=1e-8)
    ap.add_argument("--max-protected-targets", type=int, default=3)
    ap.add_argument("--covariance-protection", action="store_true")
    ap.add_argument("--max-tx-nodes", type=int, default=0,
                    help="0 disables the hard coordinated-radiator cap")
    ap.add_argument(
        "--tpuic-arm", default="tp_uic_full",
        choices=(
            "tp_uic_full", "targeted_tpuic_full", "soft_tpuic",
            "adaptive_soft_tpuic", "perfect_channel",
        ),
    )
    ap.add_argument(
        "--soft-mu-grid", default="",
        help=("optional comma-separated soft-protection controls; when set, "
              "each TP-UIC state jointly selects mu and its link schedule by "
              "the belief-side max-min objective"),
    )
    ap.add_argument(
        "--cell-soft-mu-grid", default="",
        help=("optional comma-separated soft controls selected independently "
              "for every receiver-target cell; use with controlled_state"),
    )
    ap.add_argument("--adaptive-risk-slack", type=float, default=0.001)
    ap.add_argument("--belief-error-in-cres", action="store_true")
    ap.add_argument(
        "--oracle-geometry-planning", action="store_true",
        help="audit only: expose truth geometry/base gains to the planner",
    )
    ap.add_argument(
        "--evaluation-tpuic-arm",
        choices=(
            "", "tp_uic_full", "targeted_tpuic_full", "soft_tpuic",
            "adaptive_soft_tpuic", "perfect_channel",
        ),
        default="",
        help="optional held-out arm; planning remains controlled by --tpuic-arm",
    )
    ap.add_argument("--min-links", type=int, default=2)
    ap.add_argument("--max-links-per-target", type=int, default=6)
    ap.add_argument("--max-total-links", type=int, default=60)
    ap.add_argument(
        "--redundancy-policy",
        choices=(
            "diversity_first", "pd_first", "sigma_coverage", "sigma_required",
            "sigma_singleton",
        ),
        default="diversity_first",
    )
    ap.add_argument(
        "--capture-coverage-required", type=float, default=0.95,
        help=(
            "required fraction of deterministic ellipsoid-boundary directions; "
            "a directional safety tolerance, not a probability guarantee or utility"
        ),
    )
    ap.add_argument(
        "--selection-rcs-factor", type=float, default=1.0,
        help="belief-only RCS factor used by selection; truth evaluation remains nominal",
    )
    ap.add_argument(
        "--selector-strategy",
        choices=("c2f", "rcs_bundle", "hybrid_bundle"),
        default="c2f",
    )
    ap.add_argument(
        "--fusion-rule",
        choices=(
            "max_in_rate", "max_min_rate", "nearest_target",
            "capacitated_value", "capacitated_pd_lookahead",
        ),
        default="max_in_rate",
        help=("belief-side fusion destination rule; the PD lookahead consumes "
              "the same TP-UIC-adjusted table as selection"),
    )
    ap.add_argument("--capability-reps", type=int, default=1)
    ap.add_argument("--capability-quantile", type=float, default=0.20)
    ap.add_argument(
        "--capability-source",
        choices=(
            "none", "target_feedback", "controlled_state", "measured",
            "measured_model", "risk_moment", "prior_quantile", "predicted",
        ),
        default=FORMAL_CAPABILITY_SOURCE,
        help=("belief-side planning residual certificate; measured variants "
              "are oracle diagnostics and require --allow-oracle-planning"),
    )
    ap.add_argument(
        "--capability-retention-source",
        choices=(
            "none", "target_feedback", "controlled_state", "measured",
            "predicted", "predicted_risk",
        ),
        default=FORMAL_RETENTION_SOURCE,
        help=("planning target-survival certificate; predicted is formed only "
              "from the belief dictionary and cancellation operator"),
    )
    ap.add_argument(
        "--risk-swap-rounds", type=int, default=0,
        help=("fixed-link-count 1-swap rounds that lexicographically improve "
              "belief-side worst-target predicted P_D"),
    )
    ap.add_argument(
        "--constraint-maxmin", action="store_true",
        help=("use leximin proposals without scalar communication price and "
              "accept illumination updates only when exact worst-target "
              "predicted P_D improves"),
    )
    ap.add_argument(
        "--risk-secondary-rounds", type=int, default=0,
        help=("fixed-resource swaps that improve prior-quantile weakest-target "
              "PD only when nominal weakest-target, service, and configured "
              "objective do not fall"),
    )
    ap.add_argument(
        "--risk-secondary-pair-exchange", action="store_true",
        help=("when no safe 1-swap exists, exactly search target-local 2-for-2 "
              "exchanges under the same nominal safety contract"),
    )
    ap.add_argument(
        "--fusion-pd-polish", action="store_true",
        help=("with the sensing schedule fixed, enumerate feasible fusion "
              "destinations and monotonically improve belief-side target PD"),
    )
    ap.add_argument(
        "--power-policy", choices=("uniform", "mask_role"), default="uniform",
    )
    ap.add_argument("--active-rho", type=float, default=0.95)
    ap.add_argument("--inactive-rho", type=float, default=0.20)
    ap.add_argument(
        "--heldout-capability",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=("evaluate receiver capability on an independent realization; "
              "enabled by default"),
    )
    ap.add_argument(
        "--allow-oracle-planning", action="store_true",
        help=("explicitly label and allow truth-conditioned measured "
              "capabilities for post-hoc oracle diagnostics"),
    )
    ap.add_argument(
        "--exact-gaussian-threshold", action="store_true",
        help="use deterministic exact-mixture calibration only for final detection",
    )
    ap.add_argument(
        "--false-per-target", type=int, default=None,
        help="override H0 repetitions per target for threshold calibration audits",
    )
    ap.add_argument(
        "--h1-per-target", type=int, default=None,
        help="independent H1 repetitions per target for conditional-P_D audits",
    )
    ap.add_argument("--out", default="results_joint_tpuic_coordination_v1a")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if not 0.0 < args.capture_coverage_required <= 1.0:
        ap.error("--capture-coverage-required must lie in (0, 1]")
    if not math.isfinite(args.processing_gain_multiplier) or (
        args.processing_gain_multiplier <= 0.0
    ):
        ap.error("--processing-gain-multiplier must be finite and positive")

    if not math.isfinite(args.selection_rcs_factor) or not (
        0.0 < args.selection_rcs_factor <= 1.0
    ):
        ap.error("--selection-rcs-factor must lie in (0, 1]")
    if args.selector_strategy != "c2f" and args.selection_rcs_factor != 1.0:
        ap.error("bundle strategies already apply prior.rcs_lower_factor")
    if args.false_per_target is not None and args.false_per_target < 1:
        ap.error("--false-per-target must be positive")
    if args.h1_per_target is not None and args.h1_per_target < 1:
        ap.error("--h1-per-target must be positive")
    if args.capability_reps < 1:
        ap.error("--capability-reps must be positive")
    if args.max_links_per_target < args.min_links:
        ap.error("--max-links-per-target must be at least --min-links")
    if args.max_total_links < args.min_links * args.q:
        ap.error("--max-total-links is too small for the per-target minimum")
    try:
        soft_mu_grid = [
            float(value) for value in args.soft_mu_grid.split(",") if value.strip()
        ]
    except ValueError:
        ap.error("--soft-mu-grid must be a comma-separated list of numbers")
    if any(not math.isfinite(value) or value < 0.0 for value in soft_mu_grid):
        ap.error("--soft-mu-grid values must be finite and non-negative")
    try:
        cell_soft_mu_grid = [
            float(value) for value in args.cell_soft_mu_grid.split(",")
            if value.strip()
        ]
    except ValueError:
        ap.error("--cell-soft-mu-grid must be a comma-separated list of numbers")
    if any(not math.isfinite(value) or value < 0.0 for value in cell_soft_mu_grid):
        ap.error("--cell-soft-mu-grid values must be finite and non-negative")
    protocol_error = capability_protocol_error(args)
    if protocol_error is not None:
        ap.error(protocol_error)
    if args.oracle_geometry_planning and not args.allow_oracle_planning:
        ap.error("--oracle-geometry-planning requires --allow-oracle-planning")
    if args.risk_swap_rounds < 0:
        ap.error("--risk-swap-rounds must be non-negative")
    if args.risk_secondary_rounds < 0:
        ap.error("--risk-secondary-rounds must be non-negative")
    if not 0.0 < args.active_rho < 1.0:
        ap.error("--active-rho must lie in (0, 1)")
    if not 0.0 < args.inactive_rho < 1.0:
        ap.error("--inactive-rho must lie in (0, 1)")
    if not math.isfinite(args.capability_quantile) or not (
        0.0 < args.capability_quantile <= 0.5
    ):
        ap.error("--capability-quantile must lie in (0, 0.5]")

    cfg = build_config(args)
    os.makedirs(args.out, exist_ok=True)
    rows: List[dict] = []
    started = time.time()
    for offset, trial in enumerate(range(
        int(args.trial_start), int(args.trial_start) + int(args.trials)
    )):
        rows.extend(run_trial(cfg, args, trial))
        if not args.quiet:
            print(
                "trial %d/%d [%.1f s]" % (
                    offset + 1, args.trials, time.time() - started
                ),
                flush=True,
            )

    path = os.path.join(args.out, "joint_tpuic_coordination.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    saved_config = dict(vars(args))
    saved_config["effective_n_cpi"] = int(cfg.cancellation.n_cpi)
    saved_config["cpi_scaling_in_scope"] = False
    with open(os.path.join(args.out, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(saved_config, fh, indent=2, sort_keys=True)
    report(rows, args)
    print("\nwrote %s" % os.path.abspath(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
