"""Observation selection solvers: detector-marginal rules and baselines.

Link-level algorithm layer, not a core primitive: it owns the search loops that
assemble observations under the link/capacity budgets.  It therefore lives outside
``isac_sim`` and builds on ``isac_sim.cooperation.primitives``.

All selectors share one signature,

    select(cfg, base, tables, ...) -> (selected, D)

where ``selected[q]`` is the ordered list of links feeding target ``q`` and ``D``
is the resulting per-target fused deflection.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np

from isac_sim.core.config import Config, Link, apply_overrides
from experiments.methods import C2F_METHODS, DEFAULT_METHODS, EXPERIMENTAL_METHODS, METHODS, METHOD_RNG_OFFSETS, c2f_method_name, fusion_weight_mode_for_method
from isac_sim.cooperation.primitives import feasible_links_for_target, link_cost_ms, link_delay_s, local_cap_allows, processing_caps_allow, remote_cap_allows, sensing_only_links_for_target, topk_links_by_marginal
from isac_sim.detection.fusion import (
    deflection_for_links,
    predicted_pd_for_links,
    selection_utility,
    selection_utility_from_pd,
    target_alpha,
)
from isac_sim.cooperation.reporting import is_local_observation, report_dest, report_rate
from isac_sim.sensing.model import BaseGains, EPS, LinkTables, compute_link_tables

def _greedy_lagrangian(
        cfg: Config,
        tables: LinkTables,
        candidates: Dict[int, List[Link]],
        plan: "object | None" = None,
        base: BaseGains | None = None,
        distributed_bids: bool = False,
        audit: Dict[str, float] | None = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Inner greedy loop shared by priced V1 and hard-budget V1.1.

    ``candidates[q]`` is the per-target candidate list.  This is the *single*
    place that implements the marginal-gain commit rule, so the proposed
    method and the C2F variant share the exact same update logic.
    """
    s = cfg.selector
    Q = cfg.scale.Q
    # --- Coordinated radiation -------------------------------------------
    # Both are off by default so the frozen release path is untouched.
    tx_penalty = float(s.tx_penalty or 0.0)
    max_tx = s.max_tx_nodes
    if max_tx is not None and int(max_tx) < 1:
        raise ValueError(f"selector.max_tx_nodes must be >= 1 or None, got {max_tx!r}")
    active_tx: set[int] = set()
    selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    selected_sets: Dict[int, set] = {q: set() for q in range(Q)}
    D_fuse = np.zeros(Q)
    pd_pred = np.zeros(Q)

    active_candidate_targets = [q for q in range(Q) if len(candidates[q]) > 0]
    if not active_candidate_targets:
        if audit is not None:
            audit.update(score_evaluations=0.0, coordination_messages=0.0, bid_rounds=0.0)
        return selected, D_fuse

    total_links = 0
    score_evaluations = 0
    bid_messages = 0
    bid_rounds = 0

    if s.require_local_anchor and s.max_local_observations_per_target != 0:
        # Seed one detector-best local observation per target. Under the V1.1
        # target-assignment cap each target has a distinct fusion UAV, so this
        # protects free local evidence before scarce receiver/fusion slots are
        # consumed by remote reports. Generic capacity checks are retained for
        # configurations without that one-to-one property.
        anchors: list[tuple[float, int, Link, float, float]] = []
        for q in active_candidate_targets:
            best: tuple[float, Link, float, float] | None = None
            for link in candidates[q]:
                if not is_local_observation(plan, link, q):
                    continue
                new_D = deflection_for_links(
                    cfg, tables, q, [link], weight_mode="deflection",
                    plan=plan, base=base,
                )
                new_pd = predicted_pd_for_links(
                    cfg, tables, q, [link], weight_mode="deflection",
                    plan=plan, base=base,
                )
                score = new_pd if s.score_mode.lower() == "detector_pd" else new_D
                if best is None or score > best[0]:
                    best = (float(score), link, float(new_D), float(new_pd))
            if best is not None:
                anchors.append((best[0], q, best[1], best[2], best[3]))

        # Hardest locally serviceable targets commit first if a generic
        # receiver/fusion configuration makes not every anchor simultaneously
        # feasible.
        anchors.sort(key=lambda item: item[0])
        for _, q, link, new_D, new_pd in anchors:
            if total_links >= s.max_total_links:
                break
            if not local_cap_allows(cfg, selected[q], link, q, plan):
                continue
            if not processing_caps_allow(cfg, selected, link, q, plan):
                continue
            if (
                    max_tx is not None
                    and int(link[0]) not in active_tx
                    and len(active_tx) >= int(max_tx)
            ):
                continue
            selected[q].append(link)
            selected_sets[q].add(link)
            active_tx.add(int(link[0]))
            D_fuse[q] = new_D
            if s.score_mode.lower() == "detector_pd":
                pd_pred[q] = new_pd
            total_links += 1

    while total_links < s.max_total_links:
        alpha = target_alpha(cfg, D_fuse)
        detector_aligned = s.score_mode.lower() == "detector_pd"
        utility_now = (
            selection_utility_from_pd(cfg, D_fuse, pd_pred)
            if detector_aligned else selection_utility(cfg, D_fuse)
        )
        best_tuple: Optional[Tuple[int, Link]] = None
        best_score = -np.inf
        best_D = 0.0
        best_pd = 0.0
        local_bids: list[tuple[float, int, Link, float, float]] = []

        for q in range(Q):
            if len(selected[q]) >= s.max_links_per_target:
                continue
            local_best: tuple[float, int, Link, float, float] | None = None
            for link in candidates[q]:
                if link in selected_sets[q]:
                    continue
                if not local_cap_allows(cfg, selected[q], link, q, plan):
                    continue
                if not remote_cap_allows(cfg, selected, link, q, plan):
                    continue
                if not processing_caps_allow(cfg, selected, link, q, plan):
                    continue
                introduces_tx = int(link[0]) not in active_tx
                if max_tx is not None and introduces_tx and len(active_tx) >= int(max_tx):
                    # Hard cap reached: only links illuminated by an already-active
                    # node may still enter. On its own this is a one-shot decision --
                    # no table rebuild, hence no fixed-point iteration and none of
                    # the non-convergence the feedback loop exhibited.
                    continue
                new_D = deflection_for_links(
                    cfg, tables, q, selected[q] + [link], weight_mode="deflection", plan=plan, base=base
                )
                marginal_D = new_D - D_fuse[q]
                if marginal_D <= s.min_marginal_D:
                    continue

                cost_ms = link_cost_ms(cfg, tables, q, link, plan)
                delay_price = s.lambda_c * cost_ms if s.use_delay_price else 0.0
                candidate_pd = 0.0
                if detector_aligned:
                    candidate_pd = predicted_pd_for_links(
                        cfg, tables, q, selected[q] + [link],
                        weight_mode="deflection", plan=plan, base=base,
                    )
                    D_trial = D_fuse.copy()
                    D_trial[q] = new_D
                    pd_trial = pd_pred.copy()
                    pd_trial[q] = candidate_pd
                    sensing_gain = selection_utility_from_pd(cfg, D_trial, pd_trial) - utility_now
                elif s.score_mode.lower() == "exact_utility":
                    D_trial = D_fuse.copy()
                    D_trial[q] = new_D
                    sensing_gain = selection_utility(cfg, D_trial) - utility_now
                else:
                    sensing_gain = alpha[q] * marginal_D
                score = sensing_gain - delay_price
                if tx_penalty and introduces_tx:
                    # Price the new radiator. Charged only when a *new* node is
                    # woken up, so it acts as a sparsity prior rather than a flat
                    # per-link tax.
                    score -= tx_penalty
                score_evaluations += 1

                if distributed_bids:
                    if local_best is None or score > local_best[0]:
                        local_best = (score, q, link, new_D, candidate_pd)
                elif score > best_score:
                    best_score = score
                    best_tuple = (q, link)
                    best_D = new_D
                    best_pd = candidate_pd

            if distributed_bids and local_best is not None:
                local_bids.append(local_best)
                bid_messages += 1

        if distributed_bids:
            bid_rounds += 1
            for score, q, link, new_D, candidate_pd in local_bids:
                if score > best_score:
                    best_score = score
                    best_tuple = (q, link)
                    best_D = new_D
                    best_pd = candidate_pd

        if best_tuple is None or best_score <= 0.0:
            break

        q_best, link_best = best_tuple
        selected[q_best].append(link_best)
        selected_sets[q_best].add(link_best)
        active_tx.add(int(link_best[0]))
        D_fuse[q_best] = best_D
        if detector_aligned:
            pd_pred[q_best] = best_pd
        total_links += 1

        if s.stop_at_D_min and np.all(D_fuse[active_candidate_targets] >= cfg.detect.D_min):
            break

    if tx_penalty > 0.0 and total_links == 0 and active_candidate_targets:
        # A finite price can exceed EVERY marginal gain, and the greedy's
        # ``score <= 0`` stopping rule then returns an empty schedule. Silently
        # selecting nothing is never a valid answer, and at these magnitudes the
        # caller has almost certainly used the wrong units (the utility scale is
        # O(0.1), so a price of O(1) cancels the whole problem). The hard cap
        # ``selector.max_tx_nodes`` has no such cliff -- prefer it.
        raise ValueError(
            "selector.tx_penalty "
            f"({tx_penalty!r}) exceeded every candidate's marginal gain, so the "
            "selection collapsed to an empty schedule. Scale the price to the "
            "utility units (order 0.1) or use selector.max_tx_nodes instead."
        )

    if audit is not None:
        audit.update(
            score_evaluations=float(score_evaluations),
            coordination_messages=float(
                bid_messages if distributed_bids else score_evaluations
            ),
            bid_rounds=float(bid_rounds if distributed_bids else 0),
        )
    return selected, D_fuse


def select_lagrangian(
        cfg: Config,
        base: BaseGains,
        tables: LinkTables,
        plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    r"""Greedy marginal-value selection.

    For each candidate link ``(i,j,q)`` the score is

        score = alpha_q * DeltaD_ijq - lambda_c * delay_ms_ijq

    and the best strictly positive score is committed at each iteration, up to
    the per-target and global link budgets.
    """
    candidates = {}
    alpha0 = target_alpha(cfg, np.zeros(cfg.scale.Q))
    for q in range(cfg.scale.Q):
        feas = feasible_links_for_target(cfg, base, tables, q, plan)
        candidates[q] = topk_links_by_marginal(cfg, tables, base, feas, q, plan, float(alpha0[q]))
    return _greedy_lagrangian(cfg, tables, candidates, plan, base)


def select_c2f(
        cfg: Config,
        base: BaseGains,
        tables_coarse: LinkTables,
        apply_to_all: bool | None = None,
        plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray, Dict[str, float]]:
    r"""Coarse-to-fine DD-aware Lagrangian link selection.

    1. **Coarse stage.**  Every feasible candidate is scored by
       ``alpha_q^(0) * DeltaD^c - lambda_c * c`` using the coarse DD gain;
       only the top :attr:`Refine.shortlist_size` advance.
    2. **Fine stage.**  The sensing-SINR table is rebuilt with the refined DD
       gain (``eta_fine``) for the shortlisted links, and the greedy loop uses
       the fine deflection.
    """
    r = cfg.refine
    Q = cfg.scale.Q

    alpha0 = target_alpha(cfg, np.zeros(Q))
    shortlist: Dict[int, List[Link]] = {}
    fine_eval_full = 0
    fine_eval_c2f = 0
    for q in range(Q):
        feas = feasible_links_for_target(cfg, base, tables_coarse, q, plan)
        fine_eval_full += len(feas)
        if not feas:
            shortlist[q] = []
            continue
        scored: List[Tuple[float, Link]] = []
        for link in feas:
            D_single = deflection_for_links(cfg, tables_coarse, q, [link], weight_mode="deflection", plan=plan,
                                            base=base)
            cost_ms = link_cost_ms(cfg, tables_coarse, q, link, plan)
            if cfg.selector.score_mode.lower() == "detector_pd":
                D_trial = np.zeros(Q)
                D_trial[q] = D_single
                pd_trial = np.zeros(Q)
                pd_trial[q] = predicted_pd_for_links(
                    cfg, tables_coarse, q, [link],
                    weight_mode="deflection", plan=plan, base=base,
                )
                gain = selection_utility_from_pd(cfg, D_trial, pd_trial) - selection_utility_from_pd(
                    cfg, np.zeros(Q), np.zeros(Q)
                )
            elif cfg.selector.score_mode.lower() == "exact_utility":
                D_trial = np.zeros(Q)
                D_trial[q] = D_single
                gain = selection_utility(cfg, D_trial) - selection_utility(cfg, np.zeros(Q))
            else:
                gain = alpha0[q] * D_single
            delay_price = cfg.selector.lambda_c * cost_ms if cfg.selector.use_delay_price else 0.0
            score = gain - delay_price
            scored.append((score, link))
        scored.sort(key=lambda x: x[0], reverse=True)
        shortlist[q] = [link for _, link in scored[: r.shortlist_size] if _ > 0.0]
        fine_eval_c2f += len(shortlist[q])

    if apply_to_all is None:
        apply_to_all = r.apply_to_all
    if apply_to_all:
        tables_fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
        fine_candidates = {
            q: feasible_links_for_target(cfg, base, tables_fine, q, plan)
            for q in range(Q)
        }
        fine_eval_c2f = fine_eval_full
    else:
        # C2F: rebuild tables with eta_fine for the shortlisted (i, j, q).
        dd_gain = base.dd_frac_loss.copy()
        for q in range(Q):
            for link in shortlist[q]:
                i, j = link
                dd_gain[i, j, q] = base.eta_fine[i, j, q]
        tables_fine = compute_link_tables(cfg, base, dd_gain=dd_gain)
        fine_candidates = shortlist

    selected, D = _greedy_lagrangian(cfg, tables_fine, fine_candidates, plan, base)
    stats = {"fine_eval_full": float(fine_eval_full), "fine_eval_c2f": float(fine_eval_c2f)}
    return selected, D, stats


def select_c2f_adaptive(
        cfg: Config,
        base: BaseGains,
        tables_coarse: LinkTables,
        plan: "object | None" = None,
        distributed_bids: bool = False,
        refined_table_builder=None,
        shortlist_seed=None,
        trajectory: "list | None" = None,
        active_tx_mask: np.ndarray | None = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray, Dict[str, float]]:
    r"""Build a greedy-consistent dynamic shortlist, then replay on fine DD.

    Static C2F freezes a single-link ranking computed at the empty set.  This
    variant instead performs a complete coarse greedy rollout.  At every
    rollout step it records the current best not-yet-shortlisted alternative
    for each eligible target, after accounting for the links already committed
    by the coarse greedy state.  The union of these evolving frontiers becomes
    the fine candidate pool, on which the canonical greedy rule is replayed.

    The two stages therefore share the same state-dependent marginal rule,
    while fine evaluation remains capped by ``shortlist_size`` per target and
    requires only one refined-table construction.

    ``active_tx_mask`` is the coordination口径 radiation mask (see
    ``coordination.py`` for its single meaning).  It is forwarded to the *fine*
    table rebuild, which otherwise silently drops the gate: the refined table is
    constructed here from ``dd_gain`` alone, so without this argument a caller
    could hand in a gated coarse table and still have the fine replay -- the
    stage that actually decides the schedule -- score an un-gated one.  Two
    guards enforce the pairing instead of trusting it: the gate must be enabled
    (otherwise the mask is decoration), and ``tables_coarse`` must already have
    been built with the same mask (checked through ``raw_gamma_sense``, which is
    zero for every muted illuminator once the echo gate is in place).
    """
    s = cfg.selector
    Q = cfg.scale.Q
    if active_tx_mask is not None:
        if not cfg.interference.sense_gate_by_active_tx:
            raise ValueError(
                "select_c2f_adaptive(active_tx_mask=...) requires "
                "interference.sense_gate_by_active_tx=True; otherwise "
                "compute_link_tables ignores the mask and the caller would "
                "believe it selected under coordination when it did not."
            )
        muted = ~np.asarray(active_tx_mask, dtype=bool)
        if muted.any() and np.any(np.asarray(tables_coarse.raw_gamma_sense)[muted] != 0.0):
            raise ValueError(
                "tables_coarse was not built with this active_tx_mask: muted "
                "illuminators still carry a non-zero raw sensing SINR, so the "
                "coarse stage would score an un-gated table. Build it with "
                "coordination.gated_tables()."
            )
    all_candidates = {
        q: feasible_links_for_target(cfg, base, tables_coarse, q, plan)
        for q in range(Q)
    }
    fine_eval_full = int(sum(len(v) for v in all_candidates.values()))
    per_target_fine_cap = {
        q: min(max(int(cfg.refine.shortlist_size), 0), len(all_candidates[q]))
        for q in range(Q)
    }

    coarse_selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    coarse_sets: Dict[int, set[Link]] = {q: set() for q in range(Q)}
    shortlist: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    shortlist_sets: Dict[int, set[Link]] = {q: set() for q in range(Q)}
    D_coarse = np.zeros(Q)
    pd_coarse = np.zeros(Q)
    total_links = 0
    detector_mode = s.score_mode.lower() == "detector_pd"
    coarse_score_evaluations = 0
    coarse_bid_messages = 0
    coarse_rounds = 0
    # With independent observations and deflection-optimal weights, fused
    # deflection is additive.  Cache each coarse singleton once so the rollout
    # updates its greedy state without repeatedly rebuilding the same fusion.
    coarse_single_D: Dict[int, Dict[Link, float]] = {q: {} for q in range(Q)}
    if not cfg.corr.enable:
        for q, links in all_candidates.items():
            coarse_single_D[q] = {
                link: deflection_for_links(
                    cfg, tables_coarse, q, [link],
                    weight_mode="deflection", plan=plan, base=base,
                )
                for link in links
            }

    def coarse_utility() -> float:
        if detector_mode:
            return selection_utility_from_pd(cfg, D_coarse, pd_coarse)
        return selection_utility(cfg, D_coarse)

    def coarse_candidate_value(q: int, link: Link) -> tuple[float, float, bool]:
        if not cfg.corr.enable:
            new_D = D_coarse[q] + coarse_single_D[q][link]
        else:
            new_D = deflection_for_links(
                cfg, tables_coarse, q, coarse_selected[q] + [link],
                weight_mode="deflection", plan=plan, base=base,
            )
        marginal_D = new_D - D_coarse[q]
        if marginal_D <= s.min_marginal_D:
            return float(new_D), 0.0, False

        new_pd = 0.0
        if detector_mode:
            new_pd = predicted_pd_for_links(
                cfg, tables_coarse, q, coarse_selected[q] + [link],
                weight_mode="deflection", plan=plan, base=base,
            )
        return float(new_D), float(new_pd), True

    def coarse_score_from_value(
            q: int,
            link: Link,
            utility_now: float,
            new_D: float,
            new_pd: float,
            valid: bool,
    ) -> float:
        if not valid:
            return -np.inf
        marginal_D = new_D - D_coarse[q]
        if detector_mode:
            D_trial = D_coarse.copy()
            D_trial[q] = new_D
            pd_trial = pd_coarse.copy()
            pd_trial[q] = new_pd
            sensing_gain = selection_utility_from_pd(cfg, D_trial, pd_trial) - utility_now
        elif s.score_mode.lower() == "exact_utility":
            D_trial = D_coarse.copy()
            D_trial[q] = new_D
            sensing_gain = selection_utility(cfg, D_trial) - utility_now
        else:
            sensing_gain = float(target_alpha(cfg, D_coarse)[q]) * marginal_D

        delay_price = (
            s.lambda_c * link_cost_ms(cfg, tables_coarse, q, link, plan)
            if s.use_delay_price else 0.0
        )
        return float(sensing_gain - delay_price)

    def coarse_score(q: int, link: Link, utility_now: float) -> tuple[float, float, float]:
        new_D, new_pd, valid = coarse_candidate_value(q, link)
        score = coarse_score_from_value(
            q, link, utility_now, new_D, new_pd, valid
        )
        return score, new_D, new_pd

    # Detector-aligned candidate moments depend only on the selected set of
    # their own target. A commit for another target changes the global fair
    # utility but not these moments, so cache the expensive per-candidate
    # predictions and recompute only their cheap global marginal scores.
    pd_value_cache: Dict[int, Dict[Link, tuple[float, float, bool]]] = {}

    while total_links < s.max_total_links:
        coarse_rounds += 1
        utility_now = coarse_utility()
        best: tuple[int, Link] | None = None
        best_score = -np.inf
        best_D = 0.0
        best_pd = 0.0
        round_bids: list[tuple[float, int, Link, float, float]] = []

        for q in range(Q):
            if len(coarse_selected[q]) >= s.max_links_per_target:
                continue
            frontier_q: tuple[float, Link] | None = None
            local_best_q: tuple[float, int, Link, float, float] | None = None
            q_has_bid = False
            if detector_mode and q not in pd_value_cache:
                pd_value_cache[q] = {
                    link: coarse_candidate_value(q, link)
                    for link in all_candidates[q]
                    if link not in coarse_sets[q]
                }
            for link in all_candidates[q]:
                if link in coarse_sets[q]:
                    continue
                if not local_cap_allows(
                        cfg, coarse_selected[q], link, q, plan
                ):
                    continue
                if not remote_cap_allows(
                        cfg, coarse_selected, link, q, plan
                ):
                    continue
                if not processing_caps_allow(
                        cfg, coarse_selected, link, q, plan
                ):
                    continue
                if detector_mode:
                    new_D, new_pd, valid = pd_value_cache[q][link]
                    score = coarse_score_from_value(
                        q, link, utility_now, new_D, new_pd, valid
                    )
                else:
                    score, new_D, new_pd = coarse_score(q, link, utility_now)
                coarse_score_evaluations += 1
                q_has_bid = True
                if link not in shortlist_sets[q] and (
                        frontier_q is None or score > frontier_q[0]
                ):
                    frontier_q = (score, link)
                if distributed_bids:
                    if local_best_q is None or score > local_best_q[0]:
                        local_best_q = (score, q, link, new_D, new_pd)
                elif score > best_score:
                    best_score = score
                    best = (q, link)
                    best_D = new_D
                    best_pd = new_pd

            if distributed_bids and q_has_bid:
                coarse_bid_messages += 1
                if local_best_q is not None:
                    round_bids.append(local_best_q)

            if (
                    frontier_q is not None
                    and frontier_q[0] > 0.0
                    and len(shortlist[q]) < per_target_fine_cap[q]
            ):
                link_front = frontier_q[1]
                shortlist[q].append(link_front)
                shortlist_sets[q].add(link_front)

        if distributed_bids:
            for score, q, link, new_D, new_pd in round_bids:
                if score > best_score:
                    best_score = score
                    best = (q, link)
                    best_D = new_D
                    best_pd = new_pd

        if best is None or best_score <= 0.0:
            break

        q_best, link_best = best
        coarse_selected[q_best].append(link_best)
        coarse_sets[q_best].add(link_best)
        D_coarse[q_best] = best_D
        if detector_mode:
            pd_coarse[q_best] = best_pd
            pd_value_cache.pop(q_best, None)
        total_links += 1

        # The committed coarse path must always be represented in the fine
        # replay, even if a target's alternative frontier reached its cap.
        if link_best not in shortlist_sets[q_best]:
            if len(shortlist[q_best]) < per_target_fine_cap[q_best]:
                shortlist[q_best].append(link_best)
                shortlist_sets[q_best].add(link_best)
            elif shortlist[q_best]:
                dropped = shortlist[q_best][-1]
                shortlist_sets[q_best].remove(dropped)
                shortlist[q_best][-1] = link_best
                shortlist_sets[q_best].add(link_best)

        if s.stop_at_D_min:
            active = [q for q in range(Q) if all_candidates[q]]
            if active and np.all(D_coarse[active] >= cfg.detect.D_min):
                break

        if trajectory is not None:
            # Diagnostic only: records the post-commit coarse utility so the
            # convergence of the greedy rollout can be measured.  Never read
            # back into any decision, so default behaviour is unchanged.
            trajectory.append({
                "stage": "coarse",
                "round": int(coarse_rounds),
                "links": int(total_links),
                "utility": float(coarse_utility()),
                "D_sum": float(np.sum(D_coarse)),
            })

    if s.require_local_anchor:
        # Guarantee that the fine replay can see a local anchor even when the
        # coarse dynamic frontier filled its target-specific shortlist with
        # stronger remote singleton candidates.
        for q in range(Q):
            local_links = [
                link for link in all_candidates[q]
                if is_local_observation(plan, link, q)
            ]
            if not local_links or any(
                    is_local_observation(plan, link, q) for link in shortlist[q]
            ):
                continue
            best_local = max(
                local_links,
                key=lambda link: predicted_pd_for_links(
                    cfg, tables_coarse, q, [link], weight_mode="deflection",
                    plan=plan, base=base,
                ),
            )
            if len(shortlist[q]) < per_target_fine_cap[q]:
                shortlist[q].append(best_local)
            elif shortlist[q]:
                shortlist[q][-1] = best_local

    if shortlist_seed is not None:
        # Preserve feasible incumbent evidence while updating the coarse
        # frontier after a power move. The seed is a candidate pool, not a
        # forced commitment; the fine greedy rule still decides what to use.
        for q in range(Q):
            keep = list(dict.fromkeys(e for e in shortlist_seed.get(q, []) if e in all_candidates[q]))
            limit = max(per_target_fine_cap[q], len(keep))
            shortlist[q] = (keep + [e for e in shortlist[q] if e not in keep])[:limit]

    dd_gain = base.dd_frac_loss.copy()
    for q, links in shortlist.items():
        for i, j in links:
            dd_gain[i, j, q] = base.eta_fine[i, j, q]
    builder = compute_link_tables if refined_table_builder is None else refined_table_builder
    if active_tx_mask is None:
        tables_fine = builder(cfg, base, dd_gain=dd_gain)
    else:
        if refined_table_builder is not None:
            # The injected builders (power_c2f / power_joint) are memoising
            # factories keyed on the config, not on a mask; silently passing the
            # mask would either be ignored or break their cache key.
            raise ValueError(
                "active_tx_mask cannot be combined with a custom "
                "refined_table_builder; build the fine table yourself or use the "
                "default builder."
            )
        tables_fine = builder(cfg, base, dd_gain=dd_gain, active_tx_mask=active_tx_mask)
    fine_audit: Dict[str, float] = {}
    selected, D_fuse = _greedy_lagrangian(
        cfg, tables_fine, shortlist, plan, base,
        distributed_bids=distributed_bids,
        audit=fine_audit,
    )

    stats = {
        "fine_eval_full": float(fine_eval_full),
        "fine_eval_c2f": float(sum(len(v) for v in shortlist.values())),
        "coarse_rounds": float(coarse_rounds),
        "fine_rounds": float(sum(len(v) for v in selected.values())),
        "selector_score_evaluations": float(
            coarse_score_evaluations + fine_audit.get("score_evaluations", 0.0)
        ),
        "coordination_messages": float(
            coarse_bid_messages + fine_audit.get("coordination_messages", 0.0)
            if distributed_bids
            else coarse_score_evaluations + fine_audit.get("score_evaluations", 0.0)
        ),
        "bid_rounds": float(
            coarse_rounds + fine_audit.get("bid_rounds", 0.0)
            if distributed_bids else 0.0
        ),
    }
    return selected, D_fuse, stats


def polish_worst_target_pd(
        cfg: Config,
        base: BaseGains,
        tables: LinkTables,
        selected: Dict[int, List[Link]],
        *,
        plan: "object | None" = None,
        allowed_tx_mask: np.ndarray | None = None,
        min_links_per_target: int = 1,
        max_rounds: int = 8,
        epsilon: float = 1e-8,
) -> tuple[Dict[int, List[Link]], list[dict]]:
    """Fixed-budget 1-swap polish for the belief-side weakest target.

    The ordinary greedy rule stops when no *additive* utility marginal remains.
    That does not imply that its fixed-size schedule is locally optimal for the
    weakest target.  This polish removes one selected observation and adds one
    feasible observation, preserving the exact total link count.  A swap is
    accepted only when it lexicographically improves

    ``(min_q P_D,q, sum_q min(P_D,q, weak_pd_required))``.

    The first component is the control objective; the capped service sum only
    breaks ties without sacrificing the weakest target.  All quantities are
    computed from the supplied belief-side tables, never from truth.
    """
    q_count = int(cfg.scale.Q)
    result = {q: list(selected.get(q, [])) for q in range(q_count)}
    total_links = sum(len(v) for v in result.values())
    floor = max(int(min_links_per_target), 0)
    tx_mask = (
        np.ones(int(cfg.scale.M), dtype=bool)
        if allowed_tx_mask is None else np.asarray(allowed_tx_mask, dtype=bool)
    )
    if tx_mask.shape != (int(cfg.scale.M),):
        raise ValueError("allowed_tx_mask has the wrong shape")

    candidates = {
        q: [
            link for link in feasible_links_for_target(cfg, base, tables, q, plan)
            if tx_mask[int(link[0])]
        ]
        for q in range(q_count)
    }

    def pd_vector(schedule: Dict[int, List[Link]]) -> np.ndarray:
        return np.asarray([
            predicted_pd_for_links(
                cfg, tables, q, schedule[q], weight_mode="deflection",
                plan=plan, base=base,
            )
            for q in range(q_count)
        ], dtype=float)

    weak_req = float(cfg.detect.weak_pd_required)

    def key(pd: np.ndarray) -> tuple[float, float]:
        return float(np.min(pd)), float(np.sum(np.minimum(pd, weak_req)))

    history: list[dict] = []
    current_pd = pd_vector(result)
    current_key = key(current_pd)
    for round_index in range(max(int(max_rounds), 0)):
        best = None
        best_pd = None
        best_key = current_key
        for q_out in range(q_count):
            if len(result[q_out]) <= floor:
                continue
            for old_link in tuple(result[q_out]):
                reduced = {q: list(result[q]) for q in range(q_count)}
                reduced[q_out].remove(old_link)
                active_tx = {
                    int(link[0]) for links in reduced.values() for link in links
                }
                for q_in in range(q_count):
                    if len(reduced[q_in]) >= int(cfg.selector.max_links_per_target):
                        continue
                    for new_link in candidates[q_in]:
                        if new_link in reduced[q_in]:
                            continue
                        max_tx = cfg.selector.max_tx_nodes
                        if (
                                max_tx is not None
                                and int(new_link[0]) not in active_tx
                                and len(active_tx) >= int(max_tx)
                        ):
                            continue
                        if not local_cap_allows(
                                cfg, reduced[q_in], new_link, q_in, plan
                        ):
                            continue
                        if not remote_cap_allows(
                                cfg, reduced, new_link, q_in, plan
                        ):
                            continue
                        if not processing_caps_allow(
                                cfg, reduced, new_link, q_in, plan
                        ):
                            continue
                        trial = {q: list(reduced[q]) for q in range(q_count)}
                        trial[q_in].append(new_link)
                        trial_pd = pd_vector(trial)
                        trial_key = key(trial_pd)
                        improves = (
                                trial_key[0] > best_key[0] + float(epsilon)
                                or (
                                        abs(trial_key[0] - best_key[0]) <= float(epsilon)
                                        and trial_key[1] > best_key[1] + float(epsilon)
                                )
                        )
                        if improves:
                            best = (q_out, old_link, q_in, new_link, trial)
                            best_pd = trial_pd
                            best_key = trial_key
        if best is None or best_pd is None:
            break
        q_out, old_link, q_in, new_link, result = best
        history.append({
            "round": int(round_index),
            "remove_target": int(q_out),
            "remove_link": tuple(int(v) for v in old_link),
            "add_target": int(q_in),
            "add_link": tuple(int(v) for v in new_link),
            "worst_pd_before": float(current_key[0]),
            "worst_pd_after": float(best_key[0]),
            "service_before": float(current_key[1]),
            "service_after": float(best_key[1]),
        })
        current_pd = best_pd
        current_key = best_key

    if sum(len(v) for v in result.values()) != total_links:
        raise RuntimeError("worst-target polish changed the total link budget")
    return result, history


def polish_risk_secondary_pd(
        cfg: Config,
        base: BaseGains,
        nominal_tables: LinkTables,
        risk_tables: LinkTables,
        selected: Dict[int, List[Link]],
        *,
        plan: "object | None" = None,
        allowed_tx_mask: np.ndarray | None = None,
        min_links_per_target: int = 1,
        max_rounds: int = 4,
        allow_target_pair_exchange: bool = False,
        epsilon: float = 1e-8,
) -> tuple[Dict[int, List[Link]], list[dict]]:
    """Improve risk-side weakest-target PD without degrading nominal service.

    This is a two-table fixed-resource 1-swap.  The nominal table remains the
    primary contract: its weakest-target PD, capped service sum, and configured
    selection objective must all be non-decreasing.  Only among those schedules
    may the risk table improve its own lexicographic weakest-target/service key.
    An optional target-local 2-for-2 exchange is evaluated only when no safe
    1-swap exists; its final schedule must satisfy the same contract, so it can
    cross a one-swap local barrier without accepting an unsafe intermediate.
    Thus a conservative residual certificate cannot replace nominal link
    valuation or select a worse nominal mask, which is the failure mode measured
    for direct risk-table substitution.
    """
    q_count = int(cfg.scale.Q)
    result = {q: list(selected.get(q, [])) for q in range(q_count)}
    total_links = sum(len(v) for v in result.values())
    floor = max(int(min_links_per_target), 0)
    tx_mask = (
        np.ones(int(cfg.scale.M), dtype=bool)
        if allowed_tx_mask is None else np.asarray(allowed_tx_mask, dtype=bool)
    )
    if tx_mask.shape != (int(cfg.scale.M),):
        raise ValueError("allowed_tx_mask has the wrong shape")
    candidates = {
        q: [
            link for link in feasible_links_for_target(
                cfg, base, nominal_tables, q, plan
            ) if tx_mask[int(link[0])]
        ] for q in range(q_count)
    }

    def pd_vector(tables: LinkTables, schedule: Dict[int, List[Link]]) -> np.ndarray:
        return np.asarray([
            predicted_pd_for_links(
                cfg, tables, q, schedule[q], weight_mode="deflection",
                plan=plan, base=base,
            ) for q in range(q_count)
        ], dtype=float)

    weak_req = float(cfg.detect.weak_pd_required)

    def key(pd: np.ndarray) -> tuple[float, float]:
        return float(np.min(pd)), float(np.sum(np.minimum(pd, weak_req)))

    def configured_objective(
            tables: LinkTables, schedule: Dict[int, List[Link]]
    ) -> float:
        deflection = np.asarray([
            deflection_for_links(
                cfg, tables, q, schedule[q], weight_mode="deflection",
                plan=plan, base=base,
            ) for q in range(q_count)
        ], dtype=float)
        pd = pd_vector(tables, schedule)
        utility = (
            selection_utility_from_pd(cfg, deflection, pd)
            if cfg.selector.score_mode.lower() == "detector_pd"
            else selection_utility(cfg, deflection)
        )
        cost = 0.0
        if cfg.selector.use_delay_price:
            cost = sum(
                float(cfg.selector.lambda_c)
                * link_cost_ms(cfg, tables, q, link, plan)
                for q in range(q_count) for link in schedule[q]
            )
        return float(utility - cost)

    nominal_pd = pd_vector(nominal_tables, result)
    risk_pd = pd_vector(risk_tables, result)
    nominal_key = key(nominal_pd)
    risk_key = key(risk_pd)
    nominal_objective = configured_objective(nominal_tables, result)
    history: list[dict] = []
    tol = float(epsilon)

    for round_index in range(max(int(max_rounds), 0)):
        best = None
        best_nominal_pd = None
        best_risk_pd = None
        best_risk_key = risk_key
        for q_out in range(q_count):
            if len(result[q_out]) <= floor:
                continue
            for old_link in tuple(result[q_out]):
                reduced = {q: list(result[q]) for q in range(q_count)}
                reduced[q_out].remove(old_link)
                active_tx = {
                    int(link[0]) for links in reduced.values() for link in links
                }
                for q_in in range(q_count):
                    if len(reduced[q_in]) >= int(cfg.selector.max_links_per_target):
                        continue
                    for new_link in candidates[q_in]:
                        if new_link in reduced[q_in]:
                            continue
                        max_tx = cfg.selector.max_tx_nodes
                        if (
                                max_tx is not None
                                and int(new_link[0]) not in active_tx
                                and len(active_tx) >= int(max_tx)
                        ):
                            continue
                        if not local_cap_allows(
                                cfg, reduced[q_in], new_link, q_in, plan
                        ):
                            continue
                        if not remote_cap_allows(
                                cfg, reduced, new_link, q_in, plan
                        ):
                            continue
                        if not processing_caps_allow(
                                cfg, reduced, new_link, q_in, plan
                        ):
                            continue
                        trial = {q: list(reduced[q]) for q in range(q_count)}
                        trial[q_in].append(new_link)
                        trial_nominal_pd = pd_vector(nominal_tables, trial)
                        trial_nominal_key = key(trial_nominal_pd)
                        if (
                                trial_nominal_key[0] < nominal_key[0] - tol
                                or trial_nominal_key[1] < nominal_key[1] - tol
                        ):
                            continue
                        trial_nominal_objective = configured_objective(
                            nominal_tables, trial
                        )
                        if trial_nominal_objective < nominal_objective - tol:
                            continue
                        trial_risk_pd = pd_vector(risk_tables, trial)
                        trial_risk_key = key(trial_risk_pd)
                        improves = (
                                trial_risk_key[0] > best_risk_key[0] + tol
                                or (
                                        abs(trial_risk_key[0] - best_risk_key[0]) <= tol
                                        and trial_risk_key[1] > best_risk_key[1] + tol
                                )
                        )
                        if improves:
                            best = (
                                "single", q_out, (old_link,), q_in, (new_link,), trial,
                                trial_nominal_key, trial_nominal_objective,
                            )
                            best_nominal_pd = trial_nominal_pd
                            best_risk_pd = trial_risk_pd
                            best_risk_key = trial_risk_key
        if best is None and allow_target_pair_exchange:
            # Search a target-local 2-for-2 neighbourhood exactly.  Both links
            # are removed before either is added, so capacity released by the
            # pair is represented correctly.  Only the final schedule is judged
            # against the nominal contract; no intermediate schedule is used.
            for q in range(q_count):
                if len(result[q]) < 2:
                    continue
                for old_pair in combinations(tuple(result[q]), 2):
                    reduced = {target: list(result[target]) for target in range(q_count)}
                    reduced[q].remove(old_pair[0])
                    reduced[q].remove(old_pair[1])
                    available = [
                        link for link in candidates[q]
                        if link not in reduced[q] and link not in old_pair
                    ]
                    for new_pair in combinations(available, 2):
                        trial = {target: list(reduced[target]) for target in range(q_count)}
                        feasible = True
                        for new_link in new_pair:
                            active_tx = {
                                int(link[0])
                                for links in trial.values() for link in links
                            }
                            max_tx = cfg.selector.max_tx_nodes
                            if (
                                    max_tx is not None
                                    and int(new_link[0]) not in active_tx
                                    and len(active_tx) >= int(max_tx)
                            ):
                                feasible = False
                                break
                            if not local_cap_allows(cfg, trial[q], new_link, q, plan):
                                feasible = False
                                break
                            if not remote_cap_allows(cfg, trial, new_link, q, plan):
                                feasible = False
                                break
                            if not processing_caps_allow(cfg, trial, new_link, q, plan):
                                feasible = False
                                break
                            trial[q].append(new_link)
                        if not feasible:
                            continue
                        trial_nominal_pd = pd_vector(nominal_tables, trial)
                        trial_nominal_key = key(trial_nominal_pd)
                        if (
                                trial_nominal_key[0] < nominal_key[0] - tol
                                or trial_nominal_key[1] < nominal_key[1] - tol
                        ):
                            continue
                        trial_nominal_objective = configured_objective(
                            nominal_tables, trial
                        )
                        if trial_nominal_objective < nominal_objective - tol:
                            continue
                        trial_risk_pd = pd_vector(risk_tables, trial)
                        trial_risk_key = key(trial_risk_pd)
                        improves = (
                                trial_risk_key[0] > best_risk_key[0] + tol
                                or (
                                        abs(trial_risk_key[0] - best_risk_key[0]) <= tol
                                        and trial_risk_key[1] > best_risk_key[1] + tol
                                )
                        )
                        if improves:
                            best = (
                                "target_pair", q, old_pair, q, new_pair, trial,
                                trial_nominal_key, trial_nominal_objective,
                            )
                            best_nominal_pd = trial_nominal_pd
                            best_risk_pd = trial_risk_pd
                            best_risk_key = trial_risk_key
        if best is None or best_nominal_pd is None or best_risk_pd is None:
            break
        (
            exchange_kind, q_out, old_links, q_in, new_links, result,
            new_nominal_key, new_nominal_objective,
        ) = best
        step = {
            "round": int(round_index),
            "exchange_kind": str(exchange_kind),
            "remove_target": int(q_out),
            "add_target": int(q_in),
            "nominal_worst_before": float(nominal_key[0]),
            "nominal_worst_after": float(new_nominal_key[0]),
            "nominal_objective_before": float(nominal_objective),
            "nominal_objective_after": float(new_nominal_objective),
            "risk_worst_before": float(risk_key[0]),
            "risk_worst_after": float(best_risk_key[0]),
        }
        if exchange_kind == "single":
            step["remove_link"] = tuple(int(v) for v in old_links[0])
            step["add_link"] = tuple(int(v) for v in new_links[0])
        else:
            step["remove_links"] = [tuple(int(v) for v in link) for link in old_links]
            step["add_links"] = [tuple(int(v) for v in link) for link in new_links]
        history.append(step)
        nominal_pd = best_nominal_pd
        risk_pd = best_risk_pd
        nominal_key = new_nominal_key
        nominal_objective = new_nominal_objective
        risk_key = best_risk_key

    if sum(len(v) for v in result.values()) != total_links:
        raise RuntimeError("risk-secondary polish changed the total link budget")
    return result, history


# ==========================================================================
# Baselines
# ==========================================================================
def select_all_neighbor(
        cfg: Config, base: BaseGains, tables: LinkTables, plan: "object | None" = None
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Upper-resource baseline that uses every feasible link."""
    selected: Dict[int, List[Link]] = {}
    D = np.zeros(cfg.scale.Q)
    for q in range(cfg.scale.Q):
        links = feasible_links_for_target(cfg, base, tables, q, plan)
        selected[q] = links
        D[q] = deflection_for_links(cfg, tables, q, links, weight_mode="deflection", plan=plan, base=base)
    return selected, D


def select_topk_baseline(
        cfg: Config,
        base: BaseGains,
        tables: LinkTables,
        reference_counts: Dict[int, int],
        method: str,
        rng: np.random.Generator,
        plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Rank links by a method-specific criterion and keep the top ``K`` per target."""
    if method in {"global_topk_deflection", "cost_aware_greedy", "exact_marginal_greedy"}:
        return select_global_budget_baseline(
            cfg, base, tables, int(sum(reference_counts.values())), method, plan
        )

    selected: Dict[int, List[Link]] = {}
    D = np.zeros(cfg.scale.Q)

    for q in range(cfg.scale.Q):
        if method == "raw_sense_sinr":
            links = sensing_only_links_for_target(cfg, base, q)
        else:
            links = feasible_links_for_target(cfg, base, tables, q, plan)

        if not links:
            selected[q] = []
            continue

        if method == "single_best":
            K = 1
        else:
            K = min(reference_counts.get(q, 0), len(links))

        if K <= 0:
            selected[q] = []
            continue

        if method == "random":
            order = rng.permutation(len(links))
        elif method == "nearest":
            distances = np.array([base.d_uu[i, j] for i, j in links])
            order = np.argsort(distances)
        elif method == "shortest_bistatic":
            scores = np.array([-base.tau[i, j, q] * cfg.waveform.c for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "raw_sense_sinr":
            scores = np.array([tables.raw_gamma_sense[i, j, q] for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "sense_sinr":
            scores = np.array([tables.gamma_sense[i, j, q] for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "single_best":
            scores = np.array(
                [deflection_for_links(cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base) for link
                 in links]
            )
            order = np.argsort(scores)[::-1]
        elif method == "topk_deflection":
            scores = np.array(
                [deflection_for_links(cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base) for link
                 in links]
            )
            order = np.argsort(scores)[::-1]
        else:
            raise ValueError(method)

        chosen = [links[int(idx)] for idx in order[:K]]
        selected[q] = chosen
        weight_mode = fusion_weight_mode_for_method(method)
        D[q] = deflection_for_links(cfg, tables, q, chosen, weight_mode=weight_mode, plan=plan, base=base)

    return selected, D


def select_budget_ranked_baseline(
        cfg: Config,
        base: BaseGains,
        tables: LinkTables,
        method: str,
        plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Rank observations without reading a proposed-method link count.

    The selector enforces the same exogenous local, receiver, fusion, report,
    per-target, and total budgets as the proposed method.  It is therefore the
    Sensing-SINR cell of the V1.1 factorial comparison, not the historical
    matched-count mechanism control.
    """
    if method != "sense_sinr_budgeted":
        raise ValueError(method)
    Q = cfg.scale.Q
    selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    ranked: List[Tuple[float, int, Link]] = []
    for q in range(Q):
        for link in feasible_links_for_target(cfg, base, tables, q, plan):
            ranked.append((float(tables.gamma_sense[link[0], link[1], q]), q, link))
    ranked.sort(key=lambda item: item[0], reverse=True)

    total = 0
    for _, q, link in ranked:
        if total >= cfg.selector.max_total_links:
            break
        if len(selected[q]) >= cfg.selector.max_links_per_target:
            continue
        if not local_cap_allows(cfg, selected[q], link, q, plan):
            continue
        if not remote_cap_allows(cfg, selected, link, q, plan):
            continue
        if not processing_caps_allow(cfg, selected, link, q, plan):
            continue
        selected[q].append(link)
        total += 1

    D = np.asarray([
        deflection_for_links(
            cfg, tables, q, selected[q], weight_mode="deflection",
            plan=plan, base=base,
        )
        for q in range(Q)
    ])
    return selected, D


def select_global_budget_baseline(
        cfg: Config,
        base: BaseGains,
        tables: LinkTables,
        total_budget: int,
        method: str,
        plan: "object | None" = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Strong baselines sharing only the proposed method's total link count."""
    Q = cfg.scale.Q
    total_budget = min(max(int(total_budget), 0), cfg.selector.max_total_links)
    candidates = {
        q: feasible_links_for_target(cfg, base, tables, q, plan) for q in range(Q)
    }

    if method == "exact_marginal_greedy":
        variant = apply_overrides(cfg, {
            "selector.score_mode": "exact_utility",
            "selector.use_delay_price": False,
            "selector.stop_at_D_min": False,
            "selector.max_total_links": total_budget,
        })
        return _greedy_lagrangian(variant, tables, candidates, plan, base)

    if method == "global_topk_deflection":
        scored: List[tuple[float, int, Link]] = []
        for q, links in candidates.items():
            for link in links:
                score = deflection_for_links(
                    cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base
                )
                scored.append((score, q, link))
        scored.sort(key=lambda x: x[0], reverse=True)
        selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
        used = 0
        for _, q, link in scored:
            if used >= total_budget:
                break
            if len(selected[q]) < cfg.selector.max_links_per_target:
                selected[q].append(link)
                used += 1
    elif method == "cost_aware_greedy":
        selected = {q: [] for q in range(Q)}
        selected_sets = {q: set() for q in range(Q)}
        D = np.zeros(Q, dtype=float)
        used = 0
        while used < total_budget:
            U0 = selection_utility(cfg, D)
            best: tuple[int, Link] | None = None
            best_ratio = -np.inf
            best_D = 0.0
            for q, links in candidates.items():
                if len(selected[q]) >= cfg.selector.max_links_per_target:
                    continue
                for link in links:
                    if link in selected_sets[q]:
                        continue
                    new_D = deflection_for_links(
                        cfg, tables, q, selected[q] + [link],
                        weight_mode="deflection", plan=plan, base=base,
                    )
                    D_trial = D.copy()
                    D_trial[q] = new_D
                    gain = selection_utility(cfg, D_trial) - U0
                    ratio = gain / max(link_cost_ms(cfg, tables, q, link, plan), EPS)
                    if ratio > best_ratio:
                        best_ratio, best, best_D = ratio, (q, link), new_D
            if best is None or best_ratio <= 0.0:
                break
            q, link = best
            selected[q].append(link)
            selected_sets[q].add(link)
            D[q] = best_D
            used += 1
    else:
        raise ValueError(method)

    D_out = np.array([
        deflection_for_links(
            cfg, tables, q, selected[q], weight_mode="deflection", plan=plan, base=base
        )
        for q in range(Q)
    ])
    return selected, D_out
