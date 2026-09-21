"""cancellation 积木：字典列到目标编号 / 中心列的归属表。"""

from __future__ import annotations

import numpy as np

from isac_sim.core.config import Config
from isac_sim.receiver.cancellation.dictionaries import target_dictionary


def build_true_ids(true_ids, n_tgt_basis: int):
    """把「每个目标一个编号」铺成「每列一个编号」。

    每个源的切向块宽度相同，所以编号按块重复，中心列掩码按块取第一个。
    调用方据此可以整块选中某个目标、或整块剔除它的回波，而不必知道字典的
    内部排布。
    """
    ids_true = np.repeat(np.asarray(true_ids, dtype=int), n_tgt_basis)
    centres_true = np.zeros(ids_true.size, dtype=bool)
    centres_true[::n_tgt_basis] = True
    return ids_true, centres_true


def build_belief_ids(cfg: Config, targets_belief):
    """信念字典的归属表：逐源求块宽，再拼成整表的编号与中心掩码。"""
    id_blocks = []
    centre_blocks = []
    for src in targets_belief:
        width = target_dictionary(cfg, [src]).shape[1]
        id_blocks.append(np.full(width, int(src.target), dtype=int))
        centre_mask = np.zeros(width, dtype=bool)
        if width:
            centre_mask[0] = True
        centre_blocks.append(centre_mask)
    ids_belief = (
        np.concatenate(id_blocks) if id_blocks else np.zeros(0, dtype=int)
    )
    centres_belief = (
        np.concatenate(centre_blocks) if centre_blocks else np.zeros(0, dtype=bool)
    )
    return ids_belief, centres_belief
