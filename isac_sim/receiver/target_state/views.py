"""积木：把 (接收机, 观测, 打分器) 对齐成求解器要的视图。"""
from __future__ import annotations

from dataclasses import dataclass

from isac_sim.receiver.target_state.scorer import TargetStateScorer

__all__ = ["ReceiverView", "make_receiver_views"]


@dataclass(frozen=True)
class ReceiverView:
    """一个接收机在训练 CPI 上的打分器与其**名义**目标源。"""

    receiver: int
    scorer: TargetStateScorer
    sources: tuple


def make_receiver_views(receivers, observations, scorers, target) -> list:
    """三元组必须逐个对齐；源列表只取被测目标的那一份。"""
    if not (len(receivers) == len(observations) == len(scorers)):
        raise ValueError("receivers, observations and scorers must be aligned")
    return [
        ReceiverView(
            int(rx), scorer,
            tuple(s for s in (obs.targets_belief or ())
                  if int(s.target) == int(target)),
        )
        for rx, obs, scorer in zip(receivers, observations, scorers)
    ]
