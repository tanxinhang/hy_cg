"""Observation selection: detector-marginal rules and baselines.

All selectors share one signature,

    select(cfg, base, tables, ...) -> (selected, D)

where ``selected[q]`` is the ordered list of links feeding target ``q`` and
``D`` is the resulting per-target fused deflection.

The ``plan`` argument carries the reporting architecture (which fusion UAV
``f_q`` each target reports to); ``plan=None`` keeps the legacy ``j -> i``
direction so every frozen result stays bit-exact.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config, Link, MethodName, apply_overrides
from .fusion import (
    deflection_for_links,
    fusion_weight_mode_for_method,
    predicted_pd_for_links,
    selection_utility,
    selection_utility_from_pd,
    target_alpha,
)
from .model import BaseGains, EPS, LinkTables, compute_link_tables
from .reporting import is_local_observation, report_dest, report_rate

METHODS: List[MethodName] = [
    "proposed_lagrangian",
    "proposed_c2f",
    "proposed_c2f_adaptive",
    "proposed_c2f_adaptive_pd",
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
    "proposed_c2f_pd",
    "proposed_c2f_full",
    "proposed_c2f_full_pd",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "sense_sinr_budgeted",
    "single_best",
    "topk_deflection",
    "global_topk_deflection",
    "cost_aware_greedy",
    "exact_marginal_greedy",
]

# Candidate capabilities remain CLI-selectable without silently changing the
# frozen default experiment roster or paper reproduction path.
EXPERIMENTAL_METHODS: set[str] = {
    "proposed_c2f_adaptive_pd_distributed",
    "proposed_c2f_adaptive_pd_robust",
    "proposed_c2f_adaptive_pd_calibrated",
}
DEFAULT_METHODS: List[MethodName] = [
    method for method in METHODS if method not in EXPERIMENTAL_METHODS
]

# Methods that need a second sensing-SINR table for common refined evaluation.
# Static entries map to the ``apply_to_all`` flag used by :func:`select_c2f`;
# the adaptive entry is dispatched to :func:`select_c2f_adaptive` separately.
C2F_METHODS: Dict[str, bool] = {
    "proposed_c2f": False,
    "proposed_c2f_adaptive": False,
    "proposed_c2f_adaptive_pd": False,
    "proposed_c2f_adaptive_pd_distributed": False,
    "proposed_c2f_adaptive_pd_robust": False,
    "proposed_c2f_adaptive_pd_calibrated": False,
    "proposed_c2f_pd": False,
    "proposed_c2f_full": True,
    "proposed_c2f_full_pd": True,
}

# Distinct RNG streams per method, so baselines that randomise do not share the
# proposed method's draws.
METHOD_RNG_OFFSETS: Dict[str, int] = {
    "proposed_lagrangian": 101,
    "proposed_c2f": 131,
    "proposed_c2f_adaptive": 133,
    "proposed_c2f_adaptive_pd": 135,
    "proposed_c2f_adaptive_pd_distributed": 136,
    "proposed_c2f_adaptive_pd_robust": 138,
    "proposed_c2f_adaptive_pd_calibrated": 140,
    "proposed_c2f_pd": 139,
    "proposed_c2f_full": 137,
    "proposed_c2f_full_pd": 141,
    "all_neighbor": 211,
    "random": 307,
    "nearest": 401,
    "shortest_bistatic": 457,
    "raw_sense_sinr": 461,
    "sense_sinr": 503,
    "sense_sinr_budgeted": 509,
    "single_best": 601,
    "topk_deflection": 641,
    "global_topk_deflection": 647,
    "cost_aware_greedy": 653,
    "exact_marginal_greedy": 659,
}


def c2f_method_name(apply_to_all: bool) -> str:
    """Return the canonical method name for a C2F variant."""
    return "proposed_c2f_full" if apply_to_all else "proposed_c2f"


# ==========================================================================
# Link bookkeeping
# ==========================================================================
def link_delay_s(cfg: Config, tables: LinkTables, q: int, link: Link, plan: "object | None" = None) -> float:
    """Latency of one report for target ``q`` on ``link``.

    The statistic is produced at the receiving UAV ``j`` and reported to the
    destination (``i`` in the legacy architecture, ``f_q`` in the explicit
    one); the latency honours ``cfg.comm.latency_model``.
    """
    from .fbl import report_latency_s

    if is_local_observation(plan, link, q):
        return 0.0
    return report_latency_s(cfg, report_rate(tables, plan, link, q))


def link_cost_ms(cfg: Config, tables: LinkTables, q: int, link: Link, plan: "object | None" = None) -> float:
    return 1e3 * link_delay_s(cfg, tables, q, link, plan)


def feasible_links_for_target(
    cfg: Config, base: BaseGains, tables: LinkTables, q: int, plan: "object | None" = None
) -> List[Link]:
    """Links that satisfy range, DD-validity and communication constraints.

    The communication constraint is judged on the *reporting* leg ``j -> dest``
    (``dest`` = ``i`` legacy, ``f_q`` explicit), never on the sensing pair.
    """
    links: List[Link] = []
    for i in range(cfg.scale.M):
        for j in range(cfg.scale.M):
            if i == j:
                continue
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            dest = report_dest(plan, (i, j), q)
            if dest < 0:
                continue
            if dest == j:
                links.append((i, j))
                continue
            if not base.edge_mask[j, dest]:
                continue
            if not tables.feasible_comm[dest, j]:
                continue
            links.append((i, j))
    return links


def sensing_only_links_for_target(cfg: Config, base: BaseGains, q: int) -> List[Link]:
    """Links that are usable for sensing only (communication constraint ignored)."""
    links: List[Link] = []
    for i in range(cfg.scale.M):
        for j in range(cfg.scale.M):
            if i == j:
                continue
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            links.append((i, j))
    return links


def local_cap_allows(
    cfg: Config,
    selected: List[Link],
    link: Link,
    q: int,
    plan: "object | None",
) -> bool:
    """Whether ``link`` respects the per-target local-evidence audit cap."""
    if not is_local_observation(plan, link, q):
        return True
    cap = int(cfg.selector.max_local_observations_per_target)
    if cap < 0:
        return True
    used = sum(is_local_observation(plan, chosen, q) for chosen in selected)
    return used < cap


def remote_cap_allows(
    cfg: Config,
    selected: Dict[int, List[Link]],
    link: Link,
    q: int,
    plan: "object | None",
) -> bool:
    """Whether adding ``link`` respects the global remote-report hard cap."""
    if is_local_observation(plan, link, q):
        return True
    cap = int(cfg.selector.max_remote_reports)
    if cap < 0:
        return True
    used = sum(
        not is_local_observation(plan, chosen, qq)
        for qq, links in selected.items()
        for chosen in links
    )
    return used < cap


def processing_caps_allow(
    cfg: Config,
    selected: Dict[int, List[Link]],
    link: Link,
    q: int,
    plan: "object | None",
) -> bool:
    """Whether adding an observation respects receiver and fusion capacities."""
    rx_cap = int(cfg.selector.max_observations_per_receiver)
    if rx_cap >= 0:
        receiver = int(link[1])
        rx_used = sum(
            int(chosen[1]) == receiver
            for links in selected.values()
            for chosen in links
        )
        if rx_used >= rx_cap:
            return False

    fusion_cap = int(cfg.selector.max_observations_per_fusion_uav)
    if fusion_cap >= 0:
        destination = report_dest(plan, link, q)
        fusion_used = sum(
            report_dest(plan, chosen, qq) == destination
            for qq, links in selected.items()
            for chosen in links
        )
        if fusion_used >= fusion_cap:
            return False
    return True


def topk_links_by_marginal(
    cfg: Config,
    tables: LinkTables,
    base: BaseGains,
    links: List[Link],
    q: int,
    plan: "object | None",
    alpha0_q: float,
) -> List[Link]:
    """Prune candidate links by the *first-order marginal score* of the greedy rule.

    The candidate list is ranked by the same scalar the greedy loop maximises,

        alpha_q^(0) * D_single - lambda_c * cost_ms,

    evaluated at the empty selection set.  This replaces the former hand-set
    ``beta`` ranking heuristic with the derived marginal-gain score, so the
    pruning stage and the greedy commit rule now optimise the *same* quantity
    (and, under the local-LLR model, ``D_single`` reduces to ``L*gamma^2`` with
    the fusion weight ``w propto 1 + gamma``).
    """
    topk = cfg.selector.candidate_topk_per_target
    if topk <= 0 or len(links) <= topk:
        return links
    scored: List[tuple[float, Link]] = []
    for link in links:
        D_single = deflection_for_links(cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base)
        cost_ms = link_cost_ms(cfg, tables, q, link, plan)
        if cfg.selector.score_mode.lower() == "detector_pd":
            D_trial = np.zeros(cfg.scale.Q)
            D_trial[q] = D_single
            pd_trial = np.zeros(cfg.scale.Q)
            pd_trial[q] = predicted_pd_for_links(
                cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base
            )
            gain = selection_utility_from_pd(cfg, D_trial, pd_trial) - selection_utility_from_pd(
                cfg, np.zeros(cfg.scale.Q), np.zeros(cfg.scale.Q)
            )
        elif cfg.selector.score_mode.lower() == "exact_utility":
            D_trial = np.zeros(cfg.scale.Q)
            D_trial[q] = D_single
            gain = selection_utility(cfg, D_trial) - selection_utility(
                cfg, np.zeros(cfg.scale.Q)
            )
        else:
            gain = alpha0_q * D_single
        delay_price = cfg.selector.lambda_c * cost_ms if cfg.selector.use_delay_price else 0.0
        scored.append((gain - delay_price, link))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [link for _, link in scored[:topk]]


# ==========================================================================
# Proposed selector
# ==========================================================================
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
            selected[q].append(link)
            selected_sets[q].add(link)
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
        D_fuse[q_best] = best_D
        if detector_aligned:
            pd_pred[q_best] = best_pd
        total_links += 1

        if s.stop_at_D_min and np.all(D_fuse[active_candidate_targets] >= cfg.detect.D_min):
            break

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
            D_single = deflection_for_links(cfg, tables_coarse, q, [link], weight_mode="deflection", plan=plan, base=base)
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
    """
    s = cfg.selector
    Q = cfg.scale.Q
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

    dd_gain = base.dd_frac_loss.copy()
    for q, links in shortlist.items():
        for i, j in links:
            dd_gain[i, j, q] = base.eta_fine[i, j, q]
    tables_fine = compute_link_tables(cfg, base, dd_gain=dd_gain)
    fine_audit: Dict[str, float] = {}
    selected, D_fuse = _greedy_lagrangian(
        cfg, tables_fine, shortlist, plan, base,
        distributed_bids=distributed_bids,
        audit=fine_audit,
    )

    stats = {
        "fine_eval_full": float(fine_eval_full),
        "fine_eval_c2f": float(sum(len(v) for v in shortlist.values())),
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
                [deflection_for_links(cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base) for link in links]
            )
            order = np.argsort(scores)[::-1]
        elif method == "topk_deflection":
            scores = np.array(
                [deflection_for_links(cfg, tables, q, [link], weight_mode="deflection", plan=plan, base=base) for link in links]
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
