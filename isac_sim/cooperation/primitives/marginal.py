"""marginal（自 ``isac_sim/cooperation/primitives.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from isac_sim.core.config import Config, Link
from isac_sim.detection.fusion import (
    deflection_for_links,
    predicted_pd_for_links,
    selection_utility,
    selection_utility_from_pd,
    target_alpha,
)
from isac_sim.sensing.model import BaseGains, EPS, LinkTables

from isac_sim.cooperation.primitives.links import link_cost_ms


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
