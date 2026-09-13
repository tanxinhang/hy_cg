"""Link selection: the proposed Lagrangian rule and the baselines.

All selectors share one signature,

    select(cfg, base, tables, ...) -> (selected, D)

where ``selected[q]`` is the ordered list of links feeding target ``q`` and
``D`` is the resulting per-target fused deflection.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import Config, Link, MethodName
from .fusion import deflection_for_links, fusion_weight_mode_for_method, target_alpha
from .model import BaseGains, EPS, LinkTables, packet_bits_for_target

METHODS: List[MethodName] = [
    "proposed_lagrangian",
    "all_neighbor",
    "random",
    "nearest",
    "shortest_bistatic",
    "raw_sense_sinr",
    "sense_sinr",
    "single_best",
    "topk_deflection",
]

# Methods that need a *second* sensing-SINR table built with the refined DD
# gain.  Each entry maps the method name to the ``apply_to_all`` flag used
# inside :func:`select_c2f`.
C2F_METHODS: Dict[str, bool] = {
    "proposed_c2f": False,
    "proposed_c2f_full": True,
}

# Distinct RNG streams per method, so baselines that randomise do not share the
# proposed method's draws.
METHOD_RNG_OFFSETS: Dict[str, int] = {
    "proposed_lagrangian": 101,
    "proposed_c2f": 131,
    "proposed_c2f_full": 137,
    "all_neighbor": 211,
    "random": 307,
    "nearest": 401,
    "shortest_bistatic": 457,
    "raw_sense_sinr": 461,
    "sense_sinr": 503,
    "single_best": 601,
    "topk_deflection": 641,
}


def c2f_method_name(apply_to_all: bool) -> str:
    """Return the canonical method name for a C2F variant."""
    return "proposed_c2f_full" if apply_to_all else "proposed_c2f"


# ==========================================================================
# Link bookkeeping
# ==========================================================================
def link_delay_s(cfg: Config, tables: LinkTables, q: int, link: Link) -> float:
    i, j = link
    # Reporting direction is j -> i: the soft statistic is sent from the
    # receiving UAV j back to the transmitting UAV i.
    return packet_bits_for_target(cfg, q) / max(tables.rate[j, i], EPS)


def link_cost_ms(cfg: Config, tables: LinkTables, q: int, link: Link) -> float:
    return 1e3 * link_delay_s(cfg, tables, q, link)


def feasible_links_for_target(cfg: Config, base: BaseGains, tables: LinkTables, q: int) -> List[Link]:
    """Links that satisfy range, DD-validity and communication constraints."""
    links: List[Link] = []
    for i in range(cfg.scale.M):
        for j in range(cfg.scale.M):
            if i == j:
                continue
            if not base.edge_mask[i, j]:
                continue
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            if not tables.feasible_comm[i, j]:
                continue
            if tables.beta[i, j, q] <= 0:
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
            if not base.edge_mask[i, j]:
                continue
            if cfg.dd.use_otfs_bin_validity and not base.valid_dd[i, j, q]:
                continue
            links.append((i, j))
    return links


def topk_links_by_beta(cfg: Config, tables: LinkTables, links: List[Link], q: int) -> List[Link]:
    """Prune candidate links by the beta utility to bound selection complexity."""
    topk = cfg.selector.candidate_topk_per_target
    if topk <= 0 or len(links) <= topk:
        return links
    scores = np.array([tables.beta[i, j, q] for i, j in links], dtype=float)
    order = np.argsort(scores)[::-1][:topk]
    return [links[int(idx)] for idx in order]


# ==========================================================================
# Proposed selector
# ==========================================================================
def _greedy_lagrangian(
    cfg: Config,
    tables: LinkTables,
    candidates: Dict[int, List[Link]],
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Inner greedy loop of the proposed selector.

    ``candidates[q]`` is the per-target candidate list.  This is the *single*
    place that implements the marginal-gain commit rule, so the proposed
    method and the C2F variant share the exact same update logic.
    """
    s = cfg.selector
    Q = cfg.scale.Q
    selected: Dict[int, List[Link]] = {q: [] for q in range(Q)}
    selected_sets: Dict[int, set] = {q: set() for q in range(Q)}
    D_fuse = np.zeros(Q)

    active_candidate_targets = [q for q in range(Q) if len(candidates[q]) > 0]
    if not active_candidate_targets:
        return selected, D_fuse

    total_links = 0

    while total_links < s.max_total_links:
        alpha = target_alpha(cfg, D_fuse)
        best_tuple: Optional[Tuple[int, Link]] = None
        best_score = -np.inf
        best_D = 0.0

        for q in range(Q):
            if len(selected[q]) >= s.max_links_per_target:
                continue
            for link in candidates[q]:
                if link in selected_sets[q]:
                    continue
                new_D = deflection_for_links(cfg, tables, q, selected[q] + [link], weight_mode="deflection")
                marginal_D = new_D - D_fuse[q]
                if marginal_D <= s.min_marginal_D:
                    continue

                cost_ms = link_cost_ms(cfg, tables, q, link)
                delay_price = s.lambda_c * cost_ms if s.use_delay_price else 0.0
                score = alpha[q] * marginal_D - delay_price

                if score > best_score:
                    best_score = score
                    best_tuple = (q, link)
                    best_D = new_D

        if best_tuple is None or best_score <= 0.0:
            break

        q_best, link_best = best_tuple
        selected[q_best].append(link_best)
        selected_sets[q_best].add(link_best)
        D_fuse[q_best] = best_D
        total_links += 1

        if np.all(D_fuse[active_candidate_targets] >= cfg.detect.D_min):
            break

    return selected, D_fuse


def select_lagrangian(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    r"""Greedy marginal-value selection.

    For each candidate link ``(i,j,q)`` the score is

        score = alpha_q * DeltaD_ijq - lambda_c * delay_ms_ijq

    and the best strictly positive score is committed at each iteration, up to
    the per-target and global link budgets.
    """
    candidates = {
        q: topk_links_by_beta(cfg, tables, feasible_links_for_target(cfg, base, tables, q), q)
        for q in range(cfg.scale.Q)
    }
    return _greedy_lagrangian(cfg, tables, candidates)


def select_c2f(
    cfg: Config,
    base: BaseGains,
    tables_coarse: LinkTables,
    apply_to_all: bool | None = None,
) -> Tuple[Dict[int, List[Link]], np.ndarray, Dict[str, float]]:
    r"""Coarse-to-fine DD-aware Lagrangian link selection.

    Implements the paper's C2F strategy (Section "Proposed C2F DD-Aware
    Lagrangian Link Selection"):

    1. **Coarse stage.**  For every target ``q`` each feasible candidate is
       scored by ``alpha_q^(0) * DeltaD^c - lambda_c * c`` using the coarse
       DD gain.  ``alpha_q^(0)`` is the target-priority coefficient evaluated
       at the empty selection set.  Only the top
       :attr:`Refine.shortlist_size` candidates per target advance.

    2. **Fine stage.**  The sensing-SINR table is rebuilt with the refined
       DD gain (``eta_fine``) for the shortlisted links.  The greedy loop
       then uses the *fine* deflection and the shared commit rule
       (:func:`_greedy_lagrangian`).

    When ``cfg.refine.apply_to_all`` is set the shortlist is bypassed and
    every feasible link uses the refined gain.  This is the *full local
    refinement* comparison used to quantify the fine-grid evaluation saving
    reported in the paper.

    The returned ``stats`` dict carries the C2F cost accounting:

    * ``fine_eval_full`` -- number of ``(2W+1)^2`` window evaluations that a
      full local refinement would need (one per feasible candidate).
    * ``fine_eval_c2f`` -- number of window evaluations the C2F shortlist
      actually needs (one per shortlisted candidate).
    """
    from .model import compute_link_tables

    r = cfg.refine
    Q = cfg.scale.Q

    # 1. Coarse shortlist.
    alpha0 = target_alpha(cfg, np.zeros(Q))
    shortlist: Dict[int, List[Link]] = {}
    fine_eval_full = 0
    fine_eval_c2f = 0
    for q in range(Q):
        feas = feasible_links_for_target(cfg, base, tables_coarse, q)
        fine_eval_full += len(feas)
        if not feas:
            shortlist[q] = []
            continue
        scored: List[Tuple[float, Link]] = []
        for link in feas:
            D_single = deflection_for_links(cfg, tables_coarse, q, [link], weight_mode="deflection")
            cost_ms = link_cost_ms(cfg, tables_coarse, q, link)
            score = alpha0[q] * D_single - cfg.selector.lambda_c * cost_ms
            scored.append((score, link))
        scored.sort(key=lambda x: x[0], reverse=True)
        shortlist[q] = [link for _, link in scored[: r.shortlist_size] if _ > 0.0]
        fine_eval_c2f += len(shortlist[q])

    # 2. Fine tables and greedy.
    # ``apply_to_all`` is normally carried by the method name (see
    # :data:`C2F_METHODS`); the explicit argument lets a single run evaluate
    # both the C2F and the full-refinement variant under one configuration.
    if apply_to_all is None:
        apply_to_all = r.apply_to_all
    if apply_to_all:
        tables_fine = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
        fine_candidates = {
            q: feasible_links_for_target(cfg, base, tables_fine, q)
            for q in range(Q)
        }
        # Full local refinement evaluates every feasible candidate, not just
        # the shortlist, so its window-evaluation count equals the full count.
        fine_eval_c2f = fine_eval_full
    else:
        # C2F: rebuild tables with eta_fine for the shortlisted (i, j, q).
        # Entries outside any shortlist keep the coarse gain.
        dd_gain = base.dd_frac_loss.copy()
        for q in range(Q):
            for link in shortlist[q]:
                i, j = link
                dd_gain[i, j, q] = base.eta_fine[i, j, q]
        tables_fine = compute_link_tables(cfg, base, dd_gain=dd_gain)
        fine_candidates = shortlist

    selected, D = _greedy_lagrangian(cfg, tables_fine, fine_candidates)
    stats = {"fine_eval_full": float(fine_eval_full), "fine_eval_c2f": float(fine_eval_c2f)}
    return selected, D, stats


# ==========================================================================
# Baselines
# ==========================================================================
def select_all_neighbor(cfg: Config, base: BaseGains, tables: LinkTables) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Upper-resource baseline that uses every feasible link.

    This method intentionally ignores the per-target and global link budgets.
    It is an upper-resource reference, not a resource-fair competitor.
    """
    selected: Dict[int, List[Link]] = {}
    D = np.zeros(cfg.scale.Q)
    for q in range(cfg.scale.Q):
        links = feasible_links_for_target(cfg, base, tables, q)
        selected[q] = links
        D[q] = deflection_for_links(cfg, tables, q, links, weight_mode="deflection")
    return selected, D


def select_topk_baseline(
    cfg: Config,
    base: BaseGains,
    tables: LinkTables,
    reference_counts: Dict[int, int],
    method: str,
    rng: np.random.Generator,
) -> Tuple[Dict[int, List[Link]], np.ndarray]:
    """Rank links by a method-specific criterion and keep the top ``K`` per target.

    ``K`` is taken from the proposed method's per-target link count so that
    every baseline is compared at the same resource level.
    """
    selected: Dict[int, List[Link]] = {}
    D = np.zeros(cfg.scale.Q)

    for q in range(cfg.scale.Q):
        if method == "raw_sense_sinr":
            links = sensing_only_links_for_target(cfg, base, q)
        else:
            links = feasible_links_for_target(cfg, base, tables, q)

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
            scores = np.array([tables.beta[i, j, q] for i, j in links])
            order = np.argsort(scores)[::-1]
        elif method == "topk_deflection":
            # Rank by the marginal single-link deflection and keep the top K.
            # This is the "Top-K Deflection" baseline the paper compares
            # against in the abstract (lower delay at the same K).
            scores = np.array(
                [deflection_for_links(cfg, tables, q, [link], weight_mode="deflection") for link in links]
            )
            order = np.argsort(scores)[::-1]
        else:
            raise ValueError(method)

        chosen = [links[int(idx)] for idx in order[:K]]
        selected[q] = chosen
        weight_mode = fusion_weight_mode_for_method(method)
        D[q] = deflection_for_links(cfg, tables, q, chosen, weight_mode=weight_mode)

    return selected, D
