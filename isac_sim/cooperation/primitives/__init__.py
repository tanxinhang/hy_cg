"""Selection primitives: reusable building blocks of observation selection."""

from isac_sim.cooperation.primitives.caps import (
    local_cap_allows,
    remote_cap_allows,
    processing_caps_allow,
)
from isac_sim.cooperation.primitives.links import (
    link_delay_s,
    link_cost_ms,
    feasible_links_for_target,
    sensing_only_links_for_target,
)
from isac_sim.cooperation.primitives.marginal import (
    topk_links_by_marginal,
)

__all__ = [
    "link_delay_s",
    "link_cost_ms",
    "feasible_links_for_target",
    "sensing_only_links_for_target",
    "local_cap_allows",
    "remote_cap_allows",
    "processing_caps_allow",
    "topk_links_by_marginal",
]
