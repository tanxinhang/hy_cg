"""End-to-end matrix-information / exact-z / fixed-window audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import product

import numpy as np
from scipy.stats import norm
from scipy.optimize import linprog

from isac_sim.cooperation.reporting import (
    ReceiverDetectionQuality,
    assign_fusion_nodes,
    select_detection_reports,
)
from isac_sim.cooperation.matrix_information_ao import (
    activation_neighborhood,
    monotone_matrix_information_ao,
    protection_neighborhood,
)
from isac_sim.cooperation.power_ao import marginal_delivered_pd_power_ao
from isac_sim.cooperation.formation import target_ring_formation
from isac_sim.cooperation.scientific_validation import (
    belief_driven_target_ring_formation,
    load_frozen_scenario,
    save_frozen_scenario,
)
from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.receiver import cancellation as cx
from isac_sim.receiver import cancellation_glrt as gl
from isac_sim.scenario.belief import BeliefState
from isac_sim.sensing.model import (
    build_base_gains,
    compute_link_tables,
    generate_geometry,
    radar_hardware_gain,
)
from isac_sim.sensing.waveform.timing import (
    dwell_from_looks_s,
    looks_from_dwell,
    otfs_frame_duration_s,
)
from isac_sim.sensing.fbl import (
    blocklength_latency_s, fbl_success_prob, packet_bits,
)


def _stochastic_pd(eigenvalues, looks: int, p_fa: float) -> np.ndarray:
    """Gaussian replacement for independent-look matrix covariance LLRs."""
    out = []
    total_looks = float(looks)
    for values in eigenvalues:
        lam = np.asarray(values, dtype=float)
        beta = lam / (1.0 + lam)
        constant = -total_looks * float(np.sum(np.log1p(lam)))
        mu0 = constant + total_looks * float(np.sum(beta))
        mu1 = constant + total_looks * float(np.sum(lam))
        var0 = total_looks * float(np.sum(beta**2))
        var1 = total_looks * float(np.sum(lam**2))
        if var0 <= 0.0 or var1 <= 0.0:
            # A real zero-power role can produce an empty/zero-information
            # receiver statistic.  It has no detection power beyond the
            # randomized false-alarm operating point; do not emit NaN.
            out.append(float(p_fa))
            continue
        threshold = mu0 + float(norm.ppf(1.0 - p_fa)) * np.sqrt(var0)
        out.append(float(norm.sf((threshold - mu1) / np.sqrt(var1))))
    return np.asarray(out)


def _packet_erasure_fused_pd(
    receiver_eigenvalues, success_probability, looks: int, p_fa: float,
    receiver_correlation: float = 0.0,
) -> np.ndarray:
    """Exact expected PD for receiver LLR packets with independent erasures."""
    success = np.asarray(success_probability, dtype=float)
    m, q = success.shape
    out = np.zeros(q, dtype=float)
    for target in range(q):
        for mask in range(1 << m):
            delivered = [j for j in range(m) if mask & (1 << j)]
            probability = 1.0
            for j in range(m):
                chi = float(np.clip(success[j, target], 0.0, 1.0))
                probability *= chi if j in delivered else 1.0 - chi
            if not delivered:
                pd = float(p_fa)
            else:
                modes = np.concatenate([
                    receiver_eigenvalues[j][target] for j in delivered
                ])
                # Sensitivity-only design effect for positively correlated UAV
                # evidence. rho=0 exactly preserves the production model.
                design_effect = 1.0 + float(receiver_correlation) * (len(delivered) - 1)
                modes = modes / design_effect
                pd = float(_stochastic_pd((modes,), looks, p_fa)[0])
            out[target] += probability * pd
    return out


def _uav_centric_greedy_fusion(
    receiver_eigenvalues,
    link_success: np.ndarray,
    looks: int,
    p_fa: float,
    max_reports_per_sender: int = 1,
    max_reports_per_fusion: int = 2,
    link_feasible: np.ndarray | None = None,
    fixed_fusion_nodes: np.ndarray | None = None,
    report_latency_s: float = 0.0,
    max_total_latency_s: float = float("inf"),
    receiver_correlation: float = 0.0,
) -> dict:
    """Let UAV reports, rather than targets, form the fusion graph.

    Each target starts at the UAV with the strongest local detector.  A report
    ``(source, target)`` is then admitted only when that source chooses to spend
    one of its report opportunities and the destination has receive capacity.
    Greedy admission maximizes the network worst-target PD first and total PD
    second.  This is an auditable scheduling baseline, not a distributed-game
    claim.
    """
    quality = ReceiverDetectionQuality(
        tuple(tuple(np.asarray(modes, dtype=float) for modes in row)
              for row in receiver_eigenvalues),
        n_looks=int(looks),
        p_fa=float(p_fa),
    )
    formal = select_detection_reports(
        quality,
        link_success,
        link_feasible=link_feasible,
        max_reports_per_sender=max_reports_per_sender,
        max_reports_per_fusion=max_reports_per_fusion,
        report_latency_s=report_latency_s,
        max_total_latency_s=max_total_latency_s,
        receiver_correlation=receiver_correlation,
        fusion_nodes=fixed_fusion_nodes,
    )
    return {
        "fusion_nodes": formal.fusion_nodes,
        "selected_success": formal.delivery_probability,
        "selected_reports": [{
            "source_uav": source,
            "target": target,
            "destination_uav": destination,
            "success_probability": float(link_success[source, destination]),
        } for source, target, destination in formal.selected_reports],
        "sender_load": formal.sender_load,
        "fusion_load": formal.fusion_load,
        "local_pd": formal.local_pd,
        "delivered_pd": formal.delivered_pd,
        "total_reporting_latency_s": formal.total_latency_s,
    }

    # Legacy implementation retained temporarily below as burn-down reference;
    # the executable path above now belongs to the reporting domain module.
    success = np.asarray(link_success, dtype=float)
    m = len(receiver_eigenvalues)
    q = len(receiver_eigenvalues[0])
    if success.shape != (m, m):
        raise ValueError("link_success must be an M-by-M source-to-destination matrix")
    feasible = success > 0.0 if link_feasible is None else np.asarray(
        link_feasible, dtype=bool
    )
    if feasible.shape != (m, m):
        raise ValueError("link_feasible must be an M-by-M matrix")
    local_pd = np.vstack([
        _stochastic_pd(tuple(receiver_eigenvalues[j]), looks, p_fa)
        for j in range(m)
    ])
    fusion_nodes = (
        np.argmax(local_pd, axis=0).astype(int)
        if fixed_fusion_nodes is None
        else np.asarray(fixed_fusion_nodes, dtype=int)
    )
    if fusion_nodes.shape != (q,) or np.any((fusion_nodes < 0) | (fusion_nodes >= m)):
        raise ValueError("fixed_fusion_nodes must contain one valid UAV per target")
    selected_success = np.zeros((m, q), dtype=float)
    for target, destination in enumerate(fusion_nodes):
        selected_success[destination, target] = 1.0

    sender_load = np.zeros(m, dtype=int)
    fusion_load = np.zeros(m, dtype=int)
    selected: list[dict] = []
    delivered = _packet_erasure_fused_pd(
        receiver_eigenvalues, selected_success, looks, p_fa, receiver_correlation
    )
    def score(pd, count):
        # Performance is primary. Report count is only a final tie-breaker;
        # otherwise useful intermediate actions can be rejected and trap AO.
        return (float(np.min(pd)), float(np.sum(pd)), -int(count))

    while True:
        incumbent_score = score(delivered, len(selected))
        best = None
        for source in range(m):
            if sender_load[source] >= max_reports_per_sender:
                continue
            for target, destination in enumerate(fusion_nodes):
                if source == destination or selected_success[source, target] > 0.0:
                    continue
                chi = float(success[source, destination])
                if (not feasible[source, destination] or chi <= 0.0 or
                        fusion_load[destination] >= max_reports_per_fusion):
                    continue
                if (len(selected) + 1) * report_latency_s > max_total_latency_s + 1e-15:
                    continue
                trial_success = selected_success.copy()
                trial_success[source, target] = chi
                trial_pd = _packet_erasure_fused_pd(
                    receiver_eigenvalues, trial_success, looks, p_fa,
                    receiver_correlation,
                )
                candidate_score = score(trial_pd, len(selected) + 1)
                if (candidate_score > incumbent_score and
                        (best is None or candidate_score > best[0])):
                    best = (candidate_score, source, target, destination, chi, trial_pd)
        if best is None:
            break
        _, source, target, destination, chi, delivered = best
        selected_success[source, target] = chi
        sender_load[source] += 1
        fusion_load[destination] += 1
        selected.append({
            "source_uav": int(source),
            "target": int(target),
            "destination_uav": int(destination),
            "success_probability": float(chi),
        })

    # Add/drop/swap refinement. This repairs early greedy choices without
    # introducing ACK/retransmission semantics.
    selected_actions = {
        (item["source_uav"], item["target"], item["destination_uav"])
        for item in selected
    }
    candidates = {
        (source, target, int(fusion_nodes[target]))
        for source in range(m) for target in range(q)
        if source != int(fusion_nodes[target])
        and feasible[source, int(fusion_nodes[target])]
        and success[source, int(fusion_nodes[target])] > 0.0
    }

    def evaluate_actions(actions):
        table = np.zeros((m, q), dtype=float)
        for target, destination in enumerate(fusion_nodes):
            table[destination, target] = 1.0
        for source, target, destination in actions:
            table[source, target] = success[source, destination]
        return _packet_erasure_fused_pd(
            receiver_eigenvalues, table, looks, p_fa, receiver_correlation
        )

    def valid_actions(actions):
        if len(actions) * report_latency_s > max_total_latency_s + 1e-15:
            return False
        sender = np.zeros(m, dtype=int)
        destination = np.zeros(m, dtype=int)
        for source, _target, dest in actions:
            sender[source] += 1
            destination[dest] += 1
        return bool(np.all(sender <= max_reports_per_sender) and
                    np.all(destination <= max_reports_per_fusion))

    while True:
        incumbent_pd = evaluate_actions(selected_actions)
        incumbent_score = score(incumbent_pd, len(selected_actions))
        best_actions, best_pd, best_score = selected_actions, incumbent_pd, incumbent_score
        proposals = []
        proposals.extend(selected_actions - {old} for old in selected_actions)
        proposals.extend(selected_actions | {new} for new in candidates - selected_actions)
        proposals.extend(
            (selected_actions - {old}) | {new}
            for old in selected_actions for new in candidates - selected_actions
        )
        for proposal in proposals:
            if not valid_actions(proposal):
                continue
            pd = evaluate_actions(proposal)
            proposal_score = score(pd, len(proposal))
            if proposal_score > best_score:
                best_actions, best_pd, best_score = set(proposal), pd, proposal_score
        if best_score <= incumbent_score:
            delivered = incumbent_pd
            break
        selected_actions, delivered = best_actions, best_pd

    selected = [{
        "source_uav": int(source),
        "target": int(target),
        "destination_uav": int(destination),
        "success_probability": float(success[source, destination]),
    } for source, target, destination in sorted(selected_actions)]
    sender_load = np.zeros(m, dtype=int)
    fusion_load = np.zeros(m, dtype=int)
    for item in selected:
        sender_load[item["source_uav"]] += 1
        fusion_load[item["destination_uav"]] += 1
    selected_success = np.zeros((m, q), dtype=float)
    for target, destination in enumerate(fusion_nodes):
        selected_success[destination, target] = 1.0
    for item in selected:
        selected_success[item["source_uav"], item["target"]] = item[
            "success_probability"
        ]

    return {
        "fusion_nodes": fusion_nodes,
        "selected_success": selected_success,
        "selected_reports": selected,
        "sender_load": sender_load,
        "fusion_load": fusion_load,
        "local_pd": local_pd[fusion_nodes, np.arange(q)],
        "delivered_pd": delivered,
        "total_reporting_latency_s": float(len(selected) * report_latency_s),
    }


def _uav_centric_fusion_search(
    receiver_eigenvalues,
    link_success: np.ndarray,
    looks: int,
    p_fa: float,
    max_reports_per_sender: int,
    max_reports_per_fusion: int,
    link_feasible: np.ndarray,
    report_latency_s: float = 0.0,
    max_total_latency_s: float = float("inf"),
    receiver_correlation: float = 0.0,
) -> dict:
    """Enumerate the small M^Q fusion assignment and schedule reports for each."""
    m = len(receiver_eigenvalues)
    q = len(receiver_eigenvalues[0])
    best = None
    for nodes in product(range(m), repeat=q):
        policy = _uav_centric_greedy_fusion(
            receiver_eigenvalues, link_success, looks, p_fa,
            max_reports_per_sender=max_reports_per_sender,
            max_reports_per_fusion=max_reports_per_fusion,
            link_feasible=link_feasible,
            fixed_fusion_nodes=np.asarray(nodes, dtype=int),
            report_latency_s=report_latency_s,
            max_total_latency_s=max_total_latency_s,
            receiver_correlation=receiver_correlation,
        )
        score = (
            float(np.min(policy["delivered_pd"])),
            float(np.sum(policy["delivered_pd"])),
            -len(policy["selected_reports"]),
        )
        if best is None or score > best[0]:
            best = (score, policy)
    return best[1]


def _geometry_maxmin_power_allocation(
    base,
    total_power_per_uav: float,
    sensing_fraction: float,
    peak_power_per_uav: float,
    minimum_comm_power_w: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """One-shot heterogeneous sensing/communication power allocation.

    A linear max-min surrogate links sensing power to summed bistatic target
    gain and communication power to each UAV's best reachable reporting gain.
    Fleet sensing/communication budgets are fixed by ``sensing_fraction`` and
    every UAV obeys a joint peak-power constraint.
    """
    gain = np.asarray(base.target_gain * base.dd_frac_loss, dtype=float)
    sensing_gain = np.sum(gain, axis=1)  # transmitter x target
    m, q = sensing_gain.shape
    sensing_gain /= np.maximum(np.max(sensing_gain, axis=0, keepdims=True), 1e-30)
    direct = np.asarray(base.direct_gain, dtype=float).copy()
    reachable = np.asarray(base.edge_mask, dtype=bool).copy()
    np.fill_diagonal(reachable, False)
    direct[~reachable] = 0.0
    comm_gain = np.max(direct, axis=1)
    comm_gain /= max(float(np.max(comm_gain)), 1e-30)
    sense_budget = m * float(total_power_per_uav) * float(sensing_fraction)
    comm_budget = m * float(total_power_per_uav) * (1.0 - float(sensing_fraction))
    comm_floor = float(minimum_comm_power_w)
    if not np.isfinite(comm_floor) or comm_floor < 0.0:
        raise ValueError("minimum communication power must be finite and non-negative")
    if m * comm_floor > comm_budget + 1e-12:
        raise ValueError("minimum communication powers exceed fleet communication budget")
    if comm_floor > float(peak_power_per_uav):
        raise ValueError("minimum communication power exceeds per-UAV peak")
    # x = [P_sense(0:M), P_comm(0:M), t_sense, t_comm]
    n = 2 * m + 2
    c = np.zeros(n)
    c[-2] = -1.0
    c[-1] = -0.25
    a_ub, b_ub = [], []
    for target in range(q):
        row = np.zeros(n)
        row[:m] = -sensing_gain[:, target]
        row[-2] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)
    for source in range(m):
        row = np.zeros(n)
        row[m + source] = -comm_gain[source]
        row[-1] = 1.0
        a_ub.append(row)
        b_ub.append(0.0)
    for j in range(m):
        row = np.zeros(n)
        row[j] = 1.0
        row[m + j] = 1.0
        a_ub.append(row)
        b_ub.append(float(peak_power_per_uav))
    a_eq = np.zeros((2, n))
    a_eq[0, :m] = 1.0
    a_eq[1, m:2 * m] = 1.0
    result = linprog(
        c, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
        A_eq=a_eq, b_eq=np.array([sense_budget, comm_budget]),
        bounds=(
            [(1e-6, None)] * m
            + [(max(comm_floor, 1e-6), None)] * m
            + [(0.0, None), (0.0, None)]
        ),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"heterogeneous power allocation failed: {result.message}")
    sense = np.asarray(result.x[:m])
    comm = np.asarray(result.x[m:2 * m])
    scale = np.minimum(1.0, float(peak_power_per_uav) /
                       np.maximum(sense + comm, 1e-30))
    return sense * scale, comm * scale


def run(
    trial: int = 32,
    p_fa: float = 0.05,
    area_xy: float = 600.0,
    sensing_dwell_s: float | None = None,
    activation_search: str = "local",
    medium_view_pd: float = 0.3,
    formation_views_per_target: int = 0,
    formation_radius_m: float = 100.0,
    formation_max_movement_m: float | None = None,
    cancellation_arm: str = "tp_uic_full",
    target_count: int = 3,
    uav_count: int = 6,
    comm_n_block: int = 2048,
    max_reports_per_sender: int = 1,
    max_reports_per_fusion: int = 2,
    total_power_w: float = 1.0,
    sensing_power_fraction: float = 0.8,
    power_policy: str = "uniform",
    peak_power_w: float = 1.0,
    explicit_sense_power: tuple[float, ...] | None = None,
    explicit_comm_power: tuple[float, ...] | None = None,
    coordination_aware_ao: bool = False,
    joint_fusion_search: bool = False,
    formation_source: str = "belief",
    scenario_snapshot_in: str | None = None,
    scenario_snapshot_out: str | None = None,
    master_seed: int = 2026,
    minimum_comm_power_w: float = 0.0,
    power_ao_max_rounds: int = 4,
    minimum_report_rate_bps: float = 1.0e6,
) -> dict:
    cfg = apply_preset(Config(), "paper-canonical")
    cfg = apply_overrides(cfg, {
        "geometry.area_xy": float(area_xy),
        "detect.target_rcs": 0.1,
        "scale.M": int(uav_count),
        "scale.Q": int(target_count),
        "comm.n_block": int(comm_n_block),
        "comm.R_min": float(minimum_report_rate_bps),
        "radio.P_default": float(total_power_w),
        "radio.rho": float(sensing_power_fraction),
        "run.seed": int(master_seed),
        "run.verbose": False,
        "cancellation.enable": True,
        "cancellation.belief_error_in_cres": True,
        "aperture.enable": True,
        "aperture.m_rx": 4,
        "detect.target_response_model": "swerling2_fast",
    })
    if sensing_dwell_s is None:
        raise ValueError("declare sensing_dwell_s; n_looks is not a free experiment knob")
    if (not np.isfinite(minimum_comm_power_w) or
            minimum_comm_power_w < 0.0 or
            minimum_comm_power_w > peak_power_w):
        raise ValueError("minimum_comm_power_w must lie in [0, peak_power_w]")
    if not np.isfinite(minimum_report_rate_bps) or minimum_report_rate_bps <= 0.0:
        raise ValueError("minimum_report_rate_bps must be finite and positive")
    cfg.detect.sensing_dwell_s = float(sensing_dwell_s)
    cfg.detect.n_looks = looks_from_dwell(cfg, sensing_dwell_s)
    rng = np.random.default_rng([cfg.run.seed, int(trial)])
    if scenario_snapshot_in is not None:
        if formation_views_per_target:
            raise ValueError(
                "a frozen snapshot already defines geometry; do not apply formation"
            )
        truth, belief_state, base_truth = load_frozen_scenario(scenario_snapshot_in)
        if (truth.p_uav.shape[0] != int(cfg.scale.M) or
                truth.p_tgt.shape[0] != int(cfg.scale.Q)):
            raise ValueError("snapshot M/Q does not match requested configuration")
        initial_uav_positions = truth.p_uav.copy()
        movement_distance = np.zeros(int(cfg.scale.M), dtype=float)
        effective_movement_budget = np.zeros(int(cfg.scale.M), dtype=float)
        scenario_source = "frozen_snapshot"
    else:
        truth = generate_geometry(cfg, rng)
        initial_uav_positions = truth.p_uav.copy()
        belief_state = BeliefState.from_truth(cfg, truth, rng)
        effective_movement_budget = np.zeros(int(cfg.scale.M), dtype=float)
        if formation_views_per_target:
            if formation_source == "belief":
                if formation_max_movement_m is None:
                    raise ValueError(
                        "belief formation requires a finite max movement"
                    )
                effective_movement_budget.fill(float(formation_max_movement_m))
                truth = belief_driven_target_ring_formation(
                    cfg, truth, belief_state, formation_views_per_target,
                    horizontal_radius_m=formation_radius_m,
                    max_movement_m=effective_movement_budget,
                )
            elif formation_source == "truth_oracle":
                if formation_max_movement_m is not None:
                    effective_movement_budget.fill(float(formation_max_movement_m))
                truth = target_ring_formation(
                    cfg, truth, formation_views_per_target,
                    horizontal_radius_m=formation_radius_m,
                    max_movement_m=formation_max_movement_m,
                )
            else:
                raise ValueError("formation_source must be belief or truth_oracle")
        movement_distance = np.linalg.norm(
            truth.p_uav - initial_uav_positions, axis=1
        )
        base_truth = build_base_gains(cfg, truth, rng)
        scenario_source = "generated"
        if scenario_snapshot_out is not None:
            save_frozen_scenario(
                scenario_snapshot_out, truth, belief_state, base_truth
            )
    if power_policy == "coordination_explicit":
        sense = np.asarray(explicit_sense_power, dtype=float)
        comm = np.asarray(explicit_comm_power, dtype=float)
        expected = (int(cfg.scale.M),)
        if sense.shape != expected or comm.shape != expected:
            raise ValueError("explicit sensing/communication powers must have M entries")
        if np.any(sense < 0.0) or np.any(comm < 0.0):
            raise ValueError("explicit powers must be non-negative")
        if np.any(sense + comm > float(peak_power_w) + 1e-12):
            raise ValueError("explicit per-UAV power exceeds peak_power_w")
        if np.any(comm < float(minimum_comm_power_w) - 1e-12):
            raise ValueError("explicit communication power is below its UAV floor")
    elif power_policy == "geometry_maxmin":
        sense, comm = _geometry_maxmin_power_allocation(
            base_truth, total_power_w, sensing_power_fraction, peak_power_w,
            minimum_comm_power_w=minimum_comm_power_w,
        )
    elif power_policy in ("uniform", "marginal_pd"):
        sense = np.full(int(cfg.scale.M), cfg.radio.rho * cfg.radio.P_default)
        comm = np.full(int(cfg.scale.M), (1.0 - cfg.radio.rho) * cfg.radio.P_default)
        if np.any(comm < float(minimum_comm_power_w) - 1e-12):
            raise ValueError("uniform communication power is below its UAV floor")
    else:
        raise ValueError(
            "power_policy must be uniform, geometry_maxmin, coordination_explicit "
            "or marginal_pd"
        )
    if float(np.sum(sense + comm)) > (
            int(cfg.scale.M) * float(total_power_w) + 1e-12):
        raise ValueError("power vector exceeds the fleet power budget")
    cfg.radio.P_sense_by_uav = tuple(sense.tolist())
    cfg.radio.P_comm_by_uav = tuple(comm.tolist())
    belief = belief_state.as_geometry(truth)
    base_belief = build_base_gains(
        cfg, belief, rng, channel=base_truth,
        rcs_view=cfg.prior.scheduler_rcs.lower(),
    )
    m, q = int(cfg.scale.M), int(cfg.scale.Q)
    active = tuple(True for _ in range(m))
    gain = float(radar_hardware_gain(cfg))
    pg = float(cfg.waveform.N * cfg.waveform.L)
    targets = tuple(range(q))
    cache = {}

    def receiver_case(active_state, receiver: int, protected: frozenset[int]):
        mask = np.asarray(active_state, dtype=bool)
        key = (tuple(mask.tolist()), int(receiver), frozenset(protected))
        if key in cache:
            return cache[key]
        obs = cx.build_observation(
            cfg, truth, belief, base_truth, int(receiver),
            rng=np.random.default_rng([cfg.run.seed, int(trial), 7000 + int(receiver)]),
            sense_power=sense, radiated_power=sense * mask,
            processing_gain=pg, hw_gain=gain,
            active_mask=mask, base_belief=base_belief,
            protected_targets=frozenset(protected),
        )
        full_assembly = cancellation_arm in ("tp_uic_stage1", "perfect_channel")
        arms = cx.cancellation_arms(
            cfg, obs, only=None if full_assembly else cancellation_arm
        )
        result = arms[cancellation_arm]
        model = gl.residual_model(cfg, obs, cancellation_arm, arms)
        infos, eigenvalues = [], []
        for target in targets:
            item = gl.stochastic_detection_information(cfg, obs, model, target)
            infos.append(item.jeffreys)
            eigenvalues.append(item.eigenvalues)
        value = (np.asarray(infos), tuple(eigenvalues), int(result.protect_dim))
        cache[key] = value
        return value

    # The old rule protects all Q targets here (max_protected_targets == Q).
    baseline_z = tuple(frozenset(targets) for _ in range(m))
    rank_budget = tuple(receiver_case(active, j, baseline_z[j])[2] for j in range(m))
    candidates = []
    for j in range(m):
        def local_z(active_state, incumbent, receiver=j):
            return protection_neighborhood(
                targets, incumbent,
                lambda subset: receiver_case(
                    active_state, receiver, subset
                )[2],
                rank_budget[receiver],
            )
        candidates.append(local_z)

    def evaluate_information(active_state, protection):
        return sum((receiver_case(active_state, j, protection[j])[0]
                    for j in range(m)), np.zeros(q))

    def direct_reporting_tables(active_state):
        tables = compute_link_tables(
            cfg, base_truth, active_tx_mask=np.asarray(active_state, dtype=bool)
        )
        feasible = np.asarray(base_truth.edge_mask, dtype=bool).copy()
        feasible &= tables.rate >= float(cfg.comm.R_min)
        if cfg.comm.enforce_chi_min:
            feasible &= tables.chi_comm >= float(cfg.comm.chi_min)
        np.fill_diagonal(feasible, False)
        return tables, feasible

    report_latency_s = float(blocklength_latency_s(cfg))
    reporting_latency_budget_s = float(cfg.fusion.processing_window_s)

    def evaluate_delivered(active_state, protection):
        eigenvalues = tuple(np.concatenate([
            receiver_case(active_state, j, protection[j])[1][target]
            for j in range(m)
        ]) for target in targets)
        per_receiver = tuple(
            receiver_case(active_state, j, protection[j])[1] for j in range(m)
        )
        tables, feasible = direct_reporting_tables(active_state)
        # Keep the inner AO scalable; exact fusion-node enumeration is applied
        # once to the converged state below.
        policy = _uav_centric_greedy_fusion(
            per_receiver, tables.chi_comm, cfg.detect.n_looks, p_fa,
            max_reports_per_sender=max_reports_per_sender,
            max_reports_per_fusion=max_reports_per_fusion,
            link_feasible=feasible,
            report_latency_s=report_latency_s,
            max_total_latency_s=reporting_latency_budget_s,
        )
        # Return one value per target so the existing max-min AO can consume it.
        return np.asarray(policy["delivered_pd"], dtype=float)

    evaluate = evaluate_delivered if coordination_aware_ao else evaluate_information

    def receiver_information(active_state, protection):
        return np.vstack([
            receiver_case(active_state, j, protection[j])[0]
            for j in range(m)
        ])

    baseline_info = evaluate(active, baseline_z)
    baseline_receiver_info = receiver_information(active, baseline_z)
    def combined_eigenvalues(active_state, protection):
        return tuple(np.concatenate([
            receiver_case(active_state, j, protection[j])[1][target] for j in range(m)
        ]) for target in targets)

    baseline_eigenvalues = combined_eigenvalues(active, baseline_z)
    if activation_search == "local":
        active_candidates = activation_neighborhood
    elif activation_search == "exhaustive_oracle":
        active_candidates = tuple(
            tuple(bool(mask & (1 << i)) for i in range(m))
            for mask in range(1, 1 << m)
        )
    else:
        raise ValueError("activation_search must be local or exhaustive_oracle")
    result = monotone_matrix_information_ao(
        active, baseline_z, active_candidates, candidates, evaluate, max_rounds=20
    )
    power_ao_history = []
    power_ao_initial_delivered_pd = None
    if power_policy == "marginal_pd":
        power_ao_initial_delivered_pd = evaluate_delivered(
            result.active, result.protection
        ).tolist()

        def replay_power(candidate_sense, candidate_comm):
            # Receiver evidence depends on the entire sensing vector.  Updating
            # power therefore invalidates every TP-UIC result, not just the
            # receiver belonging to the changed UAV.
            nonlocal sense, comm
            sense = np.asarray(candidate_sense, dtype=float)
            comm = np.asarray(candidate_comm, dtype=float)
            cfg.radio.P_sense_by_uav = tuple(sense.tolist())
            cfg.radio.P_comm_by_uav = tuple(comm.tolist())
            cache.clear()
            return evaluate_delivered(result.active, result.protection)

        power_result = marginal_delivered_pd_power_ao(
            sense, comm, replay_power,
            fleet_budget_w=int(cfg.scale.M) * float(total_power_w),
            peak_power_w=float(peak_power_w),
            max_rounds=int(power_ao_max_rounds),
        )
        sense = power_result.sense.copy()
        comm = power_result.comm.copy()
        power_ao_history = list(power_result.history)
    optimized_eigenvalues = combined_eigenvalues(result.active, result.protection)
    optimized_receiver_info = receiver_information(result.active, result.protection)
    receiver_eigenvalues = tuple(
        receiver_case(result.active, j, result.protection[j])[1]
        for j in range(m)
    )
    receiver_pd = np.vstack([
        _stochastic_pd(
            tuple(receiver_case(result.active, j, result.protection[j])[1]),
            cfg.detect.n_looks,
            p_fa,
        )
        for j in range(m)
    ])
    medium_mask = receiver_pd >= float(medium_view_pd)
    medium_count = np.sum(medium_mask, axis=0)
    medium_fraction = medium_count / float(m)
    cfg.fusion.mode = "explicit"
    cfg.fusion.rule = "max_in_rate"
    comm_tables = compute_link_tables(
        cfg, base_truth, active_tx_mask=np.asarray(result.active, dtype=bool)
    )
    direct_feasible = np.asarray(base_truth.edge_mask, dtype=bool).copy()
    direct_feasible &= comm_tables.rate >= float(cfg.comm.R_min)
    if cfg.comm.enforce_chi_min:
        direct_feasible &= comm_tables.chi_comm >= float(cfg.comm.chi_min)
    np.fill_diagonal(direct_feasible, False)
    reporting_plan = assign_fusion_nodes(cfg, base_belief, comm_tables, belief)
    fusion_nodes = np.asarray(reporting_plan.f_q, dtype=int)
    report_success = np.zeros((m, q), dtype=float)
    report_rate = np.zeros((m, q), dtype=float)
    for target in targets:
        destination = int(fusion_nodes[target])
        for receiver in range(m):
            if receiver == destination:
                report_success[receiver, target] = 1.0
                report_rate[receiver, target] = np.inf
            elif destination >= 0 and base_truth.edge_mask[receiver, destination]:
                report_success[receiver, target] = float(
                    comm_tables.chi_comm[receiver, destination]
                )
                report_rate[receiver, target] = float(
                    comm_tables.rate[receiver, destination]
                )
    delivered_pd = _packet_erasure_fused_pd(
        receiver_eigenvalues, report_success, cfg.detect.n_looks, p_fa
    )
    fusion_local_pd = np.asarray([
        _stochastic_pd(
            (receiver_eigenvalues[int(fusion_nodes[target])][target],),
            cfg.detect.n_looks,
            p_fa,
        )[0]
        for target in targets
    ])
    policy_fn = _uav_centric_fusion_search if joint_fusion_search else _uav_centric_greedy_fusion
    uav_policy = policy_fn(
        receiver_eigenvalues, comm_tables.chi_comm, cfg.detect.n_looks, p_fa,
        max_reports_per_sender=max_reports_per_sender,
        max_reports_per_fusion=max_reports_per_fusion,
        link_feasible=direct_feasible,
        report_latency_s=report_latency_s,
        max_total_latency_s=reporting_latency_budget_s,
    )
    uav_first_looks_08 = None
    uav_first_policy_08 = None
    for candidate_looks in range(1, 257):
        candidate_policy = _uav_centric_greedy_fusion(
            receiver_eigenvalues, comm_tables.chi_comm, candidate_looks, p_fa,
            max_reports_per_sender=max_reports_per_sender,
            max_reports_per_fusion=max_reports_per_fusion,
            link_feasible=direct_feasible,
            report_latency_s=report_latency_s,
            max_total_latency_s=reporting_latency_budget_s,
        )
        if float(np.min(candidate_policy["delivered_pd"])) >= 0.8:
            uav_first_looks_08 = candidate_looks
            uav_first_policy_08 = candidate_policy
            break
    reliability_sweep = []
    for blocklength in (128, 256, 512, 1024, 2048):
        link_success = np.asarray(fbl_success_prob(
            comm_tables.gamma_comm, blocklength, packet_bits(cfg)
        ))
        policy = _uav_centric_greedy_fusion(
            receiver_eigenvalues, link_success, cfg.detect.n_looks, p_fa,
            max_reports_per_sender=max_reports_per_sender,
            max_reports_per_fusion=max_reports_per_fusion,
            link_feasible=direct_feasible,
            report_latency_s=float(blocklength / (
                cfg.waveform.N * cfg.waveform.delta_f
            )),
            max_total_latency_s=reporting_latency_budget_s,
        )
        remote_mask = np.asarray(base_truth.edge_mask, dtype=bool) & ~np.eye(m, dtype=bool)
        remote = link_success[remote_mask]
        reliability_sweep.append({
            "blocklength": int(blocklength),
            "block_latency_ms": float(1e3 * blocklength /
                                      (cfg.waveform.N * cfg.waveform.delta_f)),
            "minimum_link_success": float(np.min(remote)),
            "median_link_success": float(np.median(remote)),
            "maximum_link_success": float(np.max(remote)),
            "selected_report_count": int(len(policy["selected_reports"])),
            "delivered_pd": policy["delivered_pd"].tolist(),
            "worst_delivered_pd": float(np.min(policy["delivered_pd"])),
        })
    look_counts = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    optimized_pd = {
        str(k): _stochastic_pd(optimized_eigenvalues, k, p_fa).tolist()
        for k in look_counts
    }
    first_looks_08 = None
    for k in range(1, 100001):
        if float(np.min(_stochastic_pd(optimized_eigenvalues, k, p_fa))) >= 0.8:
            first_looks_08 = k
            break
    frame_duration = otfs_frame_duration_s(cfg)
    claim_limitations = [
        "analytic_moment_matched_pd_not_empirical_roc",
        "receiver_evidence_independence_in_analytic_fusion",
    ]
    if scenario_snapshot_in is None:
        claim_limitations.append("scenario_not_replayed_from_frozen_snapshot")
    if formation_source == "truth_oracle" and formation_views_per_target:
        claim_limitations.append("truth_assisted_oracle_formation")
    output = {
        "trial": int(trial),
        "master_seed": int(master_seed),
        "scenario_source": scenario_source,
        "scenario_snapshot_in": scenario_snapshot_in,
        "scenario_snapshot_out": scenario_snapshot_out,
        "performance_claim_status": "screening_only",
        "performance_claim_limitations": claim_limitations,
        "area_xy_m": float(area_xy),
        "target_count": int(q),
        "uav_count": int(m),
        "total_power_w_per_uav": float(cfg.radio.P_default),
        "sensing_power_fraction": float(cfg.radio.rho),
        "sensing_power_w_per_uav": float(cfg.radio.rho * cfg.radio.P_default),
        "communication_power_w_per_uav": float(
            (1.0 - cfg.radio.rho) * cfg.radio.P_default
        ),
        "power_policy": str(power_policy),
        "fleet_power_budget_w": int(m) * float(total_power_w),
        "peak_power_w_per_uav": float(peak_power_w),
        "minimum_communication_power_w_per_uav": float(minimum_comm_power_w),
        "communication_power_floor_satisfied": bool(np.all(
            comm >= float(minimum_comm_power_w) - 1e-12
        )),
        "sensing_power_w_by_uav": sense.tolist(),
        "communication_power_w_by_uav": comm.tolist(),
        "total_power_w_by_uav": (sense + comm).tolist(),
        "power_ao_initial_delivered_pd": power_ao_initial_delivered_pd,
        "power_ao_history": power_ao_history,
        "power_ao_max_rounds": int(power_ao_max_rounds),
        "preset": "paper-canonical",
        "aperture_enable": bool(cfg.aperture.enable),
        "m_rx": int(cfg.aperture.m_rx),
        "target_response_model": str(cfg.detect.target_response_model),
        "temporal_model": "independent_fast_scattering_looks",
        "otfs_frame_duration_ms": 1e3 * frame_duration,
        "declared_sensing_dwell_ms": 1e3 * float(sensing_dwell_s),
        "derived_n_looks": int(cfg.detect.n_looks),
        "p_fa": float(p_fa),
        "cancellation_arm": cancellation_arm,
        "formation_model": (
            "baseline" if formation_views_per_target == 0
            else ("belief_driven_bounded_target_ring"
                  if formation_source == "belief"
                  else "truth_assisted_target_ring_structural_bound")
        ),
        "formation_source": str(formation_source),
        "requested_views_per_target": int(formation_views_per_target),
        "formation_radius_m": float(formation_radius_m),
        "formation_max_movement_m": formation_max_movement_m,
        "effective_movement_budget_m_per_uav": (
            effective_movement_budget.tolist()
        ),
        "movement_distance_m_per_uav": movement_distance.tolist(),
        "movement_constraint_satisfied": bool(np.all(
            movement_distance <= effective_movement_budget + 1e-9
        )),
        "total_movement_distance_m": float(np.sum(movement_distance)),
        "rank_budget": list(rank_budget),
        "baseline_protection": [sorted(v) for v in baseline_z],
        "optimized_protection": [sorted(v) for v in result.protection],
        "baseline_active": [int(v) for v in active],
        "optimized_active": [int(v) for v in result.active],
        "n_active_tx": int(sum(result.active)),
        "activation_search": activation_search,
        "coordination_aware_ao": bool(coordination_aware_ao),
        "ao_objective_type": (
            "delivered_probability" if coordination_aware_ao
            else "jeffreys_information"
        ),
        "joint_fusion_search": bool(joint_fusion_search),
        "baseline_objective": baseline_info.tolist(),
        "optimized_objective": result.information.tolist(),
        "baseline_information": (
            baseline_info.tolist() if not coordination_aware_ao else None
        ),
        "optimized_information": (
            result.information.tolist() if not coordination_aware_ao else None
        ),
        "baseline_receiver_target_information": baseline_receiver_info.tolist(),
        "optimized_receiver_target_information": optimized_receiver_info.tolist(),
        "optimized_receiver_information_share": (
            optimized_receiver_info / np.maximum(
                np.sum(optimized_receiver_info, axis=0, keepdims=True), 1e-30
            )
        ).tolist(),
        "fusion_assumption": "perfect_central_fusion_no_reporting_loss",
        "communication_message_mode": "local_sufficient_statistic_llr_packet",
        "communication_fusion_rule": str(cfg.fusion.rule),
        "communication_fusion_node_per_target": fusion_nodes.tolist(),
        "communication_report_success_probability": report_success.tolist(),
        "communication_report_rate_bps": report_rate.tolist(),
        "communication_delivered_pd": delivered_pd.tolist(),
        "communication_delivered_worst_pd": float(np.min(delivered_pd)),
        "communication_fusion_local_only_pd": fusion_local_pd.tolist(),
        "communication_cooperation_gain_pd": (
            delivered_pd - fusion_local_pd
        ).tolist(),
        "uav_centric_policy": "best_local_then_maxmin_marginal_pd_greedy",
        "communication_reliability_model": str(cfg.comm.reliability_model),
        "minimum_report_rate_bps": float(cfg.comm.R_min),
        "communication_blocklength": int(cfg.comm.n_block),
        "communication_mac_model": str(cfg.comm.mac_model),
        "communication_ack_model": "none_one_shot_erasure",
        "communication_report_latency_ms": 1e3 * report_latency_s,
        "communication_latency_budget_ms": 1e3 * reporting_latency_budget_s,
        "communication_reliability_sweep": reliability_sweep,
        "communication_receiver_correlation_model": (
            "not_applied_in_analytic_mainline; use held-out empirical joint covariance"
        ),
        "uav_centric_max_reports_per_sender": int(max_reports_per_sender),
        "uav_centric_max_reports_per_fusion": int(max_reports_per_fusion),
        "uav_centric_fusion_node_per_target": uav_policy["fusion_nodes"].tolist(),
        "uav_centric_selected_reports": uav_policy["selected_reports"],
        "uav_centric_sender_load": uav_policy["sender_load"].tolist(),
        "uav_centric_fusion_load": uav_policy["fusion_load"].tolist(),
        "uav_centric_best_local_pd": uav_policy["local_pd"].tolist(),
        "uav_centric_delivered_pd": uav_policy["delivered_pd"].tolist(),
        "uav_centric_total_reporting_latency_ms": 1e3 * float(
            uav_policy["total_reporting_latency_s"]
        ),
        "uav_centric_cooperation_gain_pd": (
            uav_policy["delivered_pd"] - uav_policy["local_pd"]
        ).tolist(),
        "uav_centric_minimum_looks_for_worst_pd_0.8": uav_first_looks_08,
        "uav_centric_minimum_dwell_ms_for_worst_pd_0.8": (
            None if uav_first_looks_08 is None
            else 1e3 * dwell_from_looks_s(cfg, uav_first_looks_08)
        ),
        "uav_centric_pd_at_minimum_looks": (
            None if uav_first_policy_08 is None
            else uav_first_policy_08["delivered_pd"].tolist()
        ),
        "uav_centric_reports_at_minimum_looks": (
            None if uav_first_policy_08 is None
            else uav_first_policy_08["selected_reports"]
        ),
        "communication_loss_from_ideal_pd": (
            _stochastic_pd(
                optimized_eigenvalues, cfg.detect.n_looks, p_fa
            ) - delivered_pd
        ).tolist(),
        "medium_view_pd_threshold": float(medium_view_pd),
        "receiver_target_pd": receiver_pd.tolist(),
        "medium_view_mask": medium_mask.astype(int).tolist(),
        "medium_view_count_per_target": medium_count.astype(int).tolist(),
        "minimum_medium_views_per_target": int(np.min(medium_count)),
        "canonical_medium_view_gate_satisfied": bool(np.all(medium_count >= 1)),
        "medium_view_fraction_per_target": medium_fraction.tolist(),
        "worst_target_medium_view_fraction": float(np.min(medium_fraction)),
        "baseline_mode_count": [len(v) for v in baseline_eigenvalues],
        "optimized_mode_count": [len(v) for v in optimized_eigenvalues],
        "information_gain_ratio": (
            result.information / np.maximum(baseline_info, 1e-30)
        ).tolist(),
        "ao_history": list(result.history),
        "baseline_pd_by_total_looks": {
            str(k): _stochastic_pd(baseline_eigenvalues, k, p_fa).tolist()
            for k in look_counts
        },
        "optimized_pd_by_total_looks": optimized_pd,
        "look_sweep": [{
            "total_looks": k,
            "sensing_dwell_ms": 1e3 * dwell_from_looks_s(cfg, k),
            "worst_pd": float(np.min(_stochastic_pd(
                optimized_eigenvalues, k, p_fa
            ))),
            "relative_energy": float(k),
        } for k in look_counts],
        "optimized_fixed_total_energy_pd": {
            str(k): _stochastic_pd(
                tuple(v * cfg.detect.n_looks / int(k) for v in optimized_eigenvalues),
                k, p_fa,
            ).tolist()
            for k in look_counts
        },
        "minimum_total_looks_for_worst_pd_0.8_fixed_per_look_energy": first_looks_08,
        "minimum_sensing_dwell_ms_for_worst_pd_0.8": (
            None if first_looks_08 is None
            else 1e3 * dwell_from_looks_s(cfg, first_looks_08)
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=int, default=32)
    parser.add_argument("--p-fa", type=float, default=0.05)
    parser.add_argument("--area-xy", type=float, default=600.0)
    parser.add_argument(
        "--sensing-dwell-ms", type=float, required=True,
        help="physical sensing dwell; complete OTFS frames determine n_looks",
    )
    parser.add_argument(
        "--activation-search", choices=("local", "exhaustive_oracle"),
        default="local",
        help="local marginal search is the mainline; exhaustive is audit-only",
    )
    parser.add_argument("--medium-view-pd", type=float, default=0.3)
    parser.add_argument("--formation-views-per-target", type=int, default=0)
    parser.add_argument("--formation-radius-m", type=float, default=100.0)
    parser.add_argument("--formation-max-movement-m", type=float)
    parser.add_argument(
        "--formation-source", choices=("belief", "truth_oracle"),
        default="belief",
    )
    parser.add_argument("--scenario-snapshot-in")
    parser.add_argument("--scenario-snapshot-out")
    parser.add_argument("--master-seed", type=int, default=2026)
    parser.add_argument("--target-count", type=int, default=3)
    parser.add_argument("--uav-count", type=int, default=6)
    parser.add_argument("--comm-n-block", type=int, default=2048)
    parser.add_argument("--max-reports-per-sender", type=int, default=1)
    parser.add_argument("--max-reports-per-fusion", type=int, default=2)
    parser.add_argument("--total-power-w", type=float, default=1.0)
    parser.add_argument("--sensing-power-fraction", type=float, default=0.8)
    parser.add_argument(
        "--power-policy",
        choices=("uniform", "geometry_maxmin", "coordination_explicit", "marginal_pd"),
        default="uniform",
    )
    parser.add_argument("--peak-power-w", type=float, default=1.0)
    parser.add_argument("--minimum-comm-power-w", type=float, default=0.0)
    parser.add_argument(
        "--minimum-report-rate-mbps", type=float, default=1.0,
        help="minimum feasible rate for every selected reporting edge",
    )
    parser.add_argument("--power-ao-max-rounds", type=int, default=4)
    parser.add_argument("--sense-power-by-uav")
    parser.add_argument("--comm-power-by-uav")
    parser.add_argument("--coordination-aware-ao", action="store_true")
    parser.add_argument("--joint-fusion-search", action="store_true")
    parser.add_argument(
        "--cancellation-arm",
        choices=("no_ic", "plain_ls", "protected_ls", "tp_uic_stage1", "tp_uic_full", "perfect_channel"),
        default="tp_uic_full",
    )
    parser.add_argument(
        "--out",
        default="studies/direction3/data/matrix_stochastic_active_trial32/result.json",
    )
    args = parser.parse_args()
    parse_power = lambda value: (
        None if value is None else tuple(float(v) for v in value.split(","))
    )
    output = run(
        args.trial, args.p_fa, args.area_xy, args.sensing_dwell_ms / 1e3,
        args.activation_search, args.medium_view_pd,
        args.formation_views_per_target, args.formation_radius_m,
        args.formation_max_movement_m,
        args.cancellation_arm,
        args.target_count, args.uav_count, args.comm_n_block,
        args.max_reports_per_sender, args.max_reports_per_fusion,
        args.total_power_w, args.sensing_power_fraction,
        args.power_policy, args.peak_power_w,
        parse_power(args.sense_power_by_uav),
        parse_power(args.comm_power_by_uav),
        args.coordination_aware_ao,
        args.joint_fusion_search,
        args.formation_source,
        args.scenario_snapshot_in,
        args.scenario_snapshot_out,
        args.master_seed,
        args.minimum_comm_power_w,
        args.power_ao_max_rounds,
        args.minimum_report_rate_mbps * 1e6,
    )
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
