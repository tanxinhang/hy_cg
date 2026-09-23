"""Reporting-payload accounting and aggregation audits.

The canonical release charges one full ``K_candidates * b_d`` packet for
every selected sensing observation ``(i, j, q)``.  This module does not change
that contract.  It quantifies two counterfactuals needed to decide whether the
contract is physically justified:

* item-only payload: one ``b_d``-bit soft statistic per observation;
* conservative packing: up to ``K_candidates`` observations produced by the
  same reporting UAV ``j`` for the same target ``q`` share one padded packet.

Packing across targets is deliberately excluded because different targets may
use different fusion nodes.  Headers and coding are also excluded, so the
counterfactual is an audit bound rather than a release-ready packet format.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

import numpy as np

from isac_sim.core.config import Config, Link
from isac_sim.sensing.fbl import blocklength_latency_s, packet_bits
from isac_sim.cooperation.reporting import ReportingPlan, is_local_observation, report_dest, slot_schedule


def packetization_audit(
    cfg: Config,
    selected: Dict[int, List[Link]],
    plan: ReportingPlan | None = None,
) -> Dict[str, float]:
    """Return current and conservatively packed reporting-load metrics."""
    sender_counts: Counter[int] = Counter()
    sender_target_counts: Counter[tuple[int, int]] = Counter()
    destination_counts: Counter[int] = Counter()
    for q, links in selected.items():
        for link in links:
            if is_local_observation(plan, link, q):
                continue
            _, j = link
            sender_counts[int(j)] += 1
            sender_target_counts[(int(j), int(q))] += 1
            destination_counts[report_dest(plan, link, q)] += 1

    reports = int(sum(sender_counts.values()))
    capacity = max(int(cfg.comm.K_candidates), 1)
    full_packet_bits = float(packet_bits(cfg))
    item_bits = float(cfg.comm.b_d)
    packed_packets = int(sum(
        math.ceil(count / capacity) for count in sender_target_counts.values()
    ))
    current_bits = reports * full_packet_bits
    packed_bits = packed_packets * full_packet_bits
    item_only_bits = reports * item_bits
    active_reporters = len(sender_counts)
    max_reports = max(sender_counts.values(), default=0)
    per_active = reports / max(active_reporters, 1)
    conflict_slots = len(slot_schedule(cfg, selected, plan))
    active_fusion_receivers = len(destination_counts)
    max_reports_per_fusion = max(destination_counts.values(), default=0)
    assigned_fusion_uavs = 0
    max_targets_per_fusion = 0
    if plan is not None and plan.is_explicit() and plan.f_q is not None:
        fusion_counts = Counter(int(v) for v in plan.f_q if int(v) >= 0)
        assigned_fusion_uavs = len(fusion_counts)
        max_targets_per_fusion = max(fusion_counts.values(), default=0)

    if cfg.comm.latency_model.lower() == "blocklength":
        block_s = blocklength_latency_s(cfg)
        current_delay_s = reports * block_s
        packed_delay_s = packed_packets * block_s
        conflict_slot_delay_s = conflict_slots * block_s
        sender_lower_bound_s = max_reports * block_s
    else:
        current_delay_s = math.nan
        packed_delay_s = math.nan
        conflict_slot_delay_s = math.nan
        sender_lower_bound_s = math.nan

    return {
        "reports": float(reports),
        "active_reporters": float(active_reporters),
        "reports_per_active_reporter": float(per_active),
        "max_reports_per_uav": float(max_reports),
        "max_bits_per_uav": float(max_reports * full_packet_bits),
        "assigned_fusion_uavs": float(assigned_fusion_uavs),
        "max_targets_per_fusion_uav": float(max_targets_per_fusion),
        "active_fusion_receivers": float(active_fusion_receivers),
        "max_reports_per_fusion_uav": float(max_reports_per_fusion),
        "current_bits": float(current_bits),
        "item_only_bits": float(item_only_bits),
        "packed_packets_same_target": float(packed_packets),
        "packed_bits_same_target": float(packed_bits),
        "packing_bit_reduction": float(
            1.0 - packed_bits / current_bits if current_bits > 0.0 else 0.0
        ),
        "current_delay_ms": float(current_delay_s * 1e3),
        "packed_delay_ms_same_target": float(packed_delay_s * 1e3),
        "conflict_graph_slots": float(conflict_slots),
        "reports_per_conflict_slot": float(reports / max(conflict_slots, 1)),
        "conflict_slot_delay_ms_no_interference": float(
            conflict_slot_delay_s * 1e3
        ),
        "sender_only_delay_lower_bound_ms": float(sender_lower_bound_s * 1e3),
    }


def summarize_packetization(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate trial-level packetization audits into mean/p90/max values."""
    if not rows:
        return {}
    out: Dict[str, float] = {}
    for key in rows[0]:
        values = np.asarray([row[key] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        out[f"{key}_mean"] = float(np.mean(finite))
        out[f"{key}_p90"] = float(np.percentile(finite, 90))
        out[f"{key}_max"] = float(np.max(finite))
    return out
