"""plan（自 ``isac_sim/cooperation/reporting.py`` 拆出）。"""

from __future__ import annotations

from typing import Dict, List
from isac_sim.core.config import Config, Link

from isac_sim.cooperation.reporting.budget import is_local_observation, report_dest


def slot_schedule(
    cfg: Config,
    selected: Dict[int, List[Link]],
    plan: "ReportingPlan | None" = None,
) -> List[List[tuple]]:
    """Conflict-graph colouring of the selected reporting legs.

    Returns a list of slots; each slot is a list of ``(q, i, j, dest)`` reports
    that can be transmitted concurrently.  Two reports conflict when they share
    the same transmitter ``j`` (a UAV cannot send two reports at once) or the
    same destination ``dest`` (a receiver decodes one report at a time in a
    plain FDMA-free model).  Greedy first-fit colouring is used.

    This is the MAC model that makes the reporting interference and the latency
    describe *one* system: only reports inside the same slot co-interfere, and
    the slots run back to back.
    """
    reports: List[tuple] = [
        (q, i, j, report_dest(plan, (i, j), q))
        for q, links in selected.items() for (i, j) in links
        if not is_local_observation(plan, (i, j), q)
    ]
    slots: List[List[tuple]] = []
    for (q, i, j, dest) in reports:
        placed = False
        for slot in slots:
            if all(d != dest and j != jj for (_, _, jj, d) in slot):
                slot.append((q, i, j, dest))
                placed = True
                break
        if not placed:
            slots.append([(q, i, j, dest)])
    return slots
