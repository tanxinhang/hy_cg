"""Inter-UAV transmission coordination: move the TX mask *before* selection.

Motivation (measured): at 500 m / RCS 0.2 the sensing denominator is interference
dominated (``rinr = +11.5 dB``). Silencing the nodes that take part in no selected
observation raises the worst-target detection probability from 0.174 to 0.953
(``tools/probe_coordination_gain.py``). That is the single strongest lever found so
far -- stronger than 16x more looks (0.819) and far stronger than any power knob.

But the existing pipeline builds ``active_tx_mask`` *after* selection
(``simulate.py:584/873``), so the selector never sees it: audit finding F1. This
module closes the loop as a two-round fixed point:

    round 0: mask = None (everyone radiates)  -> select
    round k: mask = illuminators of round k-1 -> rebuild tables -> reselect

The mask holds the **illuminators only**. A receiver does not transmit, so adding
it to the mask changes nothing for its own target and only adds interference for
all the others -- measured as an exact no-op at the target level, which makes
illuminator-only the strictly better choice.

This is deliberately a *new* module rather than an edit to the frozen core: the
release preset keeps ``interference.sense_gate_by_active_tx = False``, so nothing
here can alter a released number.

Mask semantics (one meaning, one constructor)
---------------------------------------------
``active_tx_mask`` means exactly one thing: **the UAVs that radiate during the
sensing observation**. Under the orthogonal 口径 that is precisely the set of
*illuminators* of the selected observations, because no report payload is
radiated in that phase -- so :func:`illuminator_mask` is the only constructor,
and :func:`gated_tables` is the only place coordination tables are built.

Two silent failure modes motivated that contract, both measured before it was
enforced:

* a mask built from the *reporters* (``{j}``) mutes illuminators the schedule
  still uses. Before the echo-gate fix that was invisible (the muted node kept
  its echo, it only lost its interference), so a reporter mask scored worst
  P_D 0.4575 against a physically achievable 0.3831 -- a +0.0744 phantom.
  :func:`require_mask_covers_schedule` now turns that into an exception.
* a mask is also used as the payload-interference set under the ``active_set``
  口径, where "radiating" and "illuminating" are different sets. Combining the
  gate with that 口径 is therefore rejected in ``validate_config`` rather than
  silently reinterpreted per caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .fusion import predicted_pd_for_links
from .model import BaseGains, LinkTables, build_base_gains, compute_link_tables
from .reporting import assign_fusion_nodes
from .selection import select_c2f_adaptive, select_lagrangian

Link = Tuple[int, int]


def illuminator_mask(selected: Dict[int, List[Link]], n_uav: int | None = None) -> np.ndarray:
    """THE mask constructor: nodes that must radiate to support a schedule.

    ``n_uav`` is optional; when omitted it is inferred from the largest
    illuminator index in ``selected``. A caller-supplied ``n_uav`` smaller than
    an actual illuminator raises instead of returning a mask that silently drops
    it -- the foot-gun that produced the reporter-mask phantom.
    """
    if n_uav is None:
        n_uav = 1 + max(
            (int(i) for links in selected.values() for (i, _j) in links), default=-1
        )
    n_uav = int(n_uav)
    mask = np.zeros(n_uav, dtype=bool)
    for links in selected.values():
        for i, _j in links:
            idx = int(i)
            if not 0 <= idx < n_uav:
                raise ValueError(
                    f"illuminator index {idx} does not fit in a mask of length "
                    f"{n_uav}; pass the real scale.M"
                )
            mask[idx] = True
    return mask


def require_mask_covers_schedule(
    mask: np.ndarray, selected: Dict[int, List[Link]]
) -> None:
    """Reject a mask that would silence an illuminator the schedule still uses.

    This is the invariant that makes the coordination口径 checkable instead of
    trusted: the mask may leave nodes out, but it may not leave out a node whose
    echo the schedule counts on. Raising here is deliberate -- silently scoring
    such a schedule is how a +0.0744 phantom got measured once already.
    """
    mask = np.asarray(mask, dtype=bool)
    broken: List[tuple[int, int, int]] = []
    for q, links in selected.items():
        for (i, _j) in links:
            idx = int(i)
            if idx >= mask.size or not mask[idx]:
                broken.append((int(q), idx, int(_j)))
    if broken:
        sample = ", ".join(f"(q={q},i={i},j={j})" for q, i, j in broken[:3])
        raise ValueError(
            f"active_tx_mask silences {len(broken)} scheduled observation(s) of "
            f"the schedule it is meant to evaluate [{sample}]; a radiation mask "
            f"must contain every illuminator the schedule uses. Build it with "
            f"coordination.illuminator_mask()."
        )


def gated_tables(
    cfg,
    base: BaseGains,
    selected: Dict[int, List[Link]] | None = None,
    **table_kwargs,
) -> tuple[LinkTables, np.ndarray | None]:
    """Build the coordination口径 link table for ``selected``.

    The single place where a mask and a table meet, so no caller has to
    reproduce the convention: ``selected=None`` means "round 0", i.e. every node
    radiates and the mask is ``None``.
    """
    mask = None if selected is None else illuminator_mask(selected, cfg.scale.M)
    if mask is not None:
        require_mask_covers_schedule(mask, selected)
    return compute_link_tables(cfg, base, active_tx_mask=mask, **table_kwargs), mask


@dataclass
class RoundRecord:
    round_index: int
    n_tx: int
    n_links: int
    worst_pd: float
    mean_pd: float
    per_target_pd: List[float]
    objective: np.ndarray
    mask: np.ndarray
    mask_used: np.ndarray | None = None
    fixed_point: bool = False
    cycle_detected: bool = False
    cycle_to: int | None = None


@dataclass
class CoordinationResult:
    selected: Dict[int, List[Link]] = field(default_factory=dict)
    tables: LinkTables | None = None
    plan: object | None = None
    history: List[RoundRecord] = field(default_factory=list)

    @property
    def converged(self) -> bool:
        return bool(self.history and self.history[-1].fixed_point)

    def baseline(self) -> RoundRecord:
        """Round 0: the un-coordinated pipeline (everyone radiates)."""
        return self.history[0]

    def final(self) -> RoundRecord:
        return self.history[-1]


def lagrangian_selector(cfg, base, tables, plan, mask):
    """Cheap selector for the fixed point (the pre-C2F greedy)."""
    return select_lagrangian(cfg, base, tables, plan)


def _release_selector(cfg, base, tables, plan, mask):
    """The release method. ``mask`` reaches both of its stages.

    This is the F1 closure: ``select_c2f_adaptive`` rebuilds its fine table with
    ``dd_gain=eta_fine``, and without being handed the mask it would silently
    drop the gate there -- the coarse stage would see a gated table and the fine
    stage an ungated one, inside a single selection call.
    """
    selected, objective, _stats = select_c2f_adaptive(
        cfg, base, tables, plan=plan, active_tx_mask=mask
    )
    return selected, objective


def select_with_coordination(
    cfg,
    geom,
    *,
    rounds: int = 2,
    seed: int = 20260917,
    selector=None,
    base: BaseGains | None = None,
) -> CoordinationResult:
    """Run selection with the TX mask fed back into the link tables.

    ``rounds`` is a budget, not a promise. The measure-and-report discipline matters
    here: on 6 seeds the map settled after **2, 4, 2, 3, 2 and 3** rounds, i.e. the
    earlier claim that "2 rounds is enough" was an artefact of one seed. What moves
    is the *set* of illuminators, not their count (a seed showed 11 -> 11 -> 11 -> 11
    while the membership kept changing).

    The map ``mask -> mask'`` is deterministic but has no convergence proof, so a
    repeated mask is detected and reported as a cycle rather than silently burning
    the budget. Callers should always inspect ``history`` -- specifically
    ``converged`` / ``cycle_detected`` and the per-round ``worst_pd`` -- instead of
    assuming the loop finished for the right reason.

    ``selector`` defaults to the release method (:func:`_release_selector`) so the
    fixed point is measured on the system that is actually claimed;
    :func:`lagrangian_selector` reproduces the cheaper early measurements. It must
    accept ``(cfg, base, tables, plan, mask)`` and return ``(selected, objective)``
    -- the mask argument is what keeps the gate alive across both stages.

    ``base`` lets a paired experiment keep one deployment's channel across arms;
    when omitted it is rebuilt from ``geom`` with ``seed``.
    """
    if not cfg.interference.sense_gate_by_active_tx:
        raise ValueError(
            "coordination requires interference.sense_gate_by_active_tx=True; "
            "otherwise active_tx_mask is silently ignored by compute_link_tables "
            "under the orthogonal 口径 (model.py): the mask would be a no-op."
        )
    select_fn = _release_selector if selector is None else selector
    n_uav = cfg.scale.M
    n_tgt = cfg.scale.Q
    if base is None:
        base = build_base_gains(cfg, geom, np.random.default_rng(seed))

    result = CoordinationResult()
    prev_selected: Dict[int, List[Link]] | None = None
    seen: dict[bytes, int] = {}
    for r in range(max(int(rounds), 1)):
        # ``gated_tables`` is the only table constructor on this path, and it
        # re-asserts that the mask covers the schedule it is built from.
        tables, mask_used = gated_tables(cfg, base, prev_selected)
        plan = assign_fusion_nodes(cfg, base, tables, geom=geom)
        selected, objective = select_fn(cfg, base, tables, plan, mask_used)

        pds = [
            float(predicted_pd_for_links(cfg, tables, q, selected.get(q, []), plan=plan))
            if selected.get(q)
            else 0.0
            for q in range(n_tgt)
        ]
        new_mask = illuminator_mask(selected, n_uav)
        record = RoundRecord(
            round_index=r,
            n_tx=int(new_mask.sum()),
            n_links=int(sum(len(v) for v in selected.values())),
            worst_pd=float(min(pds)),
            mean_pd=float(sum(pds) / len(pds)),
            per_target_pd=pds,
            objective=np.asarray(objective, dtype=float).copy(),
            mask=new_mask,
            mask_used=mask_used,
        )
        if mask_used is not None and np.array_equal(new_mask, mask_used):
            record.fixed_point = True
        result.history.append(record)

        result.selected = selected
        result.tables = tables
        result.plan = plan
        if record.fixed_point:
            break

        # Cycle detection. The map mask -> mask' is deterministic, but it is NOT
        # guaranteed to reach a fixed point: measured on 6 seeds the answer settled
        # after 2-4 rounds, and the *set* (not the count) was what moved. A period-2
        # cycle would otherwise burn the whole round budget in silence and look like
        # "the budget was too small". Stop and say so instead.
        key = new_mask.tobytes()
        if key in seen:
            record.cycle_detected = True
            record.cycle_to = seen[key]
            break
        seen[key] = r
        prev_selected = selected

    return result
