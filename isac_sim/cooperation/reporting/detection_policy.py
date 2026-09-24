"""Capacity-aware reporting policy driven by TP-UIC detector quality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from isac_sim.cooperation.reporting.detection_quality import ReceiverDetectionQuality


@dataclass(frozen=True)
class DetectionReportingPlan:
    fusion_nodes: np.ndarray
    delivery_probability: np.ndarray
    selected_reports: tuple[tuple[int, int, int], ...]
    local_pd: np.ndarray
    delivered_pd: np.ndarray
    sender_load: np.ndarray
    fusion_load: np.ndarray
    total_latency_s: float


def select_detection_reports(
    quality: ReceiverDetectionQuality,
    link_success: np.ndarray,
    *,
    link_feasible: np.ndarray | None = None,
    max_reports_per_sender: int = 1,
    max_reports_per_fusion: int = 2,
    report_latency_s: float = 0.0,
    max_total_latency_s: float = float("inf"),
    receiver_correlation: float = 0.0,
    fusion_nodes: np.ndarray | None = None,
) -> DetectionReportingPlan:
    """Select receiver LLR reports by worst-target PD, then total PD.

    C2F remains responsible for illumination/observation feasibility.  This
    policy begins only after TP-UIC has produced receiver-level information;
    it therefore never collapses a joint residual covariance into fake
    independent bistatic-link scores.
    """
    success = np.asarray(link_success, dtype=float)
    m, q = quality.shape
    if success.shape != (m, m):
        raise ValueError("link_success must be an M-by-M matrix")
    feasible = success > 0.0 if link_feasible is None else np.asarray(link_feasible, bool)
    if feasible.shape != (m, m):
        raise ValueError("link_feasible must be an M-by-M matrix")
    local = quality.local_pd()
    destinations = (
        np.argmax(local, axis=0).astype(int)
        if fusion_nodes is None else np.asarray(fusion_nodes, dtype=int)
    )
    if destinations.shape != (q,) or np.any((destinations < 0) | (destinations >= m)):
        raise ValueError("fusion_nodes must contain one valid UAV per target")

    actions: set[tuple[int, int, int]] = set()

    def delivery(items) -> np.ndarray:
        table = np.zeros((m, q), dtype=float)
        table[destinations, np.arange(q)] = 1.0
        for source, target, destination in items:
            table[source, target] = success[source, destination]
        return table

    def valid(items) -> bool:
        if len(items) * report_latency_s > max_total_latency_s + 1e-15:
            return False
        send = np.zeros(m, dtype=int)
        receive = np.zeros(m, dtype=int)
        for source, _target, destination in items:
            send[source] += 1
            receive[destination] += 1
        return bool(np.all(send <= max_reports_per_sender) and
                    np.all(receive <= max_reports_per_fusion))

    def score(items):
        pd = quality.fused_pd(delivery(items), receiver_correlation)
        return (float(np.min(pd)), float(np.sum(pd)), -len(items)), pd

    candidates = {
        (source, target, int(destinations[target]))
        for source in range(m) for target in range(q)
        if source != int(destinations[target])
        and feasible[source, int(destinations[target])]
        and success[source, int(destinations[target])] > 0.0
    }
    while True:
        incumbent_score, incumbent_pd = score(actions)
        best = (incumbent_score, actions, incumbent_pd)
        proposals = [actions | {new} for new in candidates - actions]
        proposals += [actions - {old} for old in actions]
        proposals += [
            (actions - {old}) | {new}
            for old in actions for new in candidates - actions
        ]
        for proposal in proposals:
            if not valid(proposal):
                continue
            candidate_score, candidate_pd = score(proposal)
            if candidate_score > best[0]:
                best = (candidate_score, set(proposal), candidate_pd)
        if best[0] <= incumbent_score:
            delivered_pd = incumbent_pd
            break
        actions = best[1]

    send = np.zeros(m, dtype=int)
    receive = np.zeros(m, dtype=int)
    for source, _target, destination in actions:
        send[source] += 1
        receive[destination] += 1
    return DetectionReportingPlan(
        fusion_nodes=destinations,
        delivery_probability=delivery(actions),
        selected_reports=tuple(sorted(actions)),
        local_pd=local[destinations, np.arange(q)],
        delivered_pd=delivered_pd,
        sender_load=send,
        fusion_load=receive,
        total_latency_s=float(len(actions) * report_latency_s),
    )
