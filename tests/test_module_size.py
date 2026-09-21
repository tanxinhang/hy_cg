"""模块体积门禁：积木代码一律 ≤150 行。

判据（用户 2026-09-20 定）：**一个模块是一个积木单元，最大 150 行。**
超过 150 行意味着它已经不是一个原子件，而是把若干职责揉在了一起 —— 这正是
"核心积木和具体链路分不开、调试难扩展难"的直接来源。

本门禁对 `isac_sim/`（积木库）生效。`experiments/` 是流程，不按积木口径卡，
但也冻结现状：**只许减少，不许新增超限文件**。
"""
from __future__ import annotations

import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_LINES = 150
#: 测试文件不是积木：一个文件往往要覆盖一条完整链路（建场景 -> 选择 -> 断言），
#: 拆太碎反而读不出因果。用户 2026-09-20 定的上限是 350 行。
MAX_TEST_LINES = 350
#: 2026-09-21 起：6 个历史大文件（2973 行）已全部拆到 350 行以内，
#: 「历史文件暂不设限」的豁免随之取消 —— 现在 tests/ 全目录是**纯红线**。
#: 共用脚手架以 `_` 开头，不参与收集，也不占这条预算。
SKIP_DIRS = {"__pycache__"}

#: 待拆清单（只许变短）。key = 相对路径，value = 当前行数。
#: 每拆掉一个就把这一行删掉 —— 列表清空之日，门禁就是纯红线。
#: 2026-09-20：`cancellation.py`（2760 行）与 `cancellation_glrt.py`（1433 行）
#: 已全部拆完，清单清空，此后 `isac_sim/` 是**纯红线**：任何超限模块直接失败。
BURN_DOWN: dict[str, int] = {}

#: `experiments/` 的现状冻结（流程层，不是积木；同样只许减少）
FLOW_FROZEN: dict[str, int] = {
    "experiments/flow/simulate.py": 1577,
    "experiments/flow/sweeps.py": 1361,
    "experiments/selection.py": 1346,
    "experiments/exploratory/bundle_master.py": 906,
    "experiments/app/plotting.py": 819,
    "experiments/coordination.py": 283,
    "experiments/app/report.py": 252,
    "experiments/exploratory/fusion_polish.py": 236,
    "experiments/app/cli.py": 234,
}


def _sizes(base: str) -> dict[str, int]:
    out = {}
    root = os.path.join(ROOT, base)
    if not os.path.isdir(root):
        return out
    for r, ds, fs in os.walk(root):
        ds[:] = [d for d in ds if d not in SKIP_DIRS]
        for f in sorted(fs):
            if not f.endswith(".py"):
                continue
            p = os.path.join(r, f)
            n = len(open(p, encoding="utf-8", errors="replace").read().split("\n"))
            out[os.path.relpath(p, ROOT).replace(os.sep, "/")] = n
    return out


def test_library_modules_are_at_most_150_lines():
    """积木库不得出现新的超限模块；待拆清单只许变短。"""
    sizes = _sizes("isac_sim")
    over = {p: n for p, n in sizes.items() if n > MAX_LINES}
    new = sorted(set(over) - set(BURN_DOWN))
    assert not new, (
        f"出现新的超限积木模块（>{MAX_LINES} 行）：\n  "
        + "\n  ".join(f"{p} ({over[p]} 行)" for p in new)
        + "\n积木必须拆到 150 行以内。"
    )
    # 已拆掉的不要留在清单里
    done = sorted(set(BURN_DOWN) - set(over))
    assert not done, (
        "下列文件已经不再超限，请从 BURN_DOWN 里删掉对应行：\n  " + "\n  ".join(done)
    )


def test_burn_down_list_matches_reality():
    """待拆清单里的行数必须与磁盘一致，避免清单和现实脱节。"""
    sizes = _sizes("isac_sim")
    drift = [
        f"{p}: 清单记 {n} 行，实际 {sizes.get(p, '文件不存在')}"
        for p, n in BURN_DOWN.items()
        if sizes.get(p) != n
    ]
    assert not drift, "待拆清单与实际不符：\n  " + "\n  ".join(drift)


def test_flow_modules_do_not_grow_new_oversize_files():
    """流程层不按积木口径卡，但也不许再长出新的大文件。"""
    sizes = _sizes("experiments")
    over = {p: n for p, n in sizes.items() if n > MAX_LINES}
    new = sorted(set(over) - set(FLOW_FROZEN))
    assert not new, (
        f"experiments/ 出现新的超限文件（>{MAX_LINES} 行）：\n  "
        + "\n  ".join(f"{p} ({over[p]} 行)" for p in new)
    )


def test_test_modules_are_at_most_350_lines():
    """tests/ 全目录按 350 行卡（2026-09-21 起无豁免）。

    一条链路一个文件，但不能无限膨胀。此前只对「优化模型」与「方向 1/2」两组
    生效，是因为 6 个历史大文件（最大 825 行）占全仓 48% 且可能整批退役；
    2026-09-21 它们已按主题拆完，豁免没有必要再留。
    """
    sizes = _sizes("tests")
    over = sorted(
        f"{p} ({n} 行)"
        for p, n in sizes.items()
        if n > MAX_TEST_LINES
    )
    assert not over, (
        f"测试文件超过 {MAX_TEST_LINES} 行：\n  "
        + "\n  ".join(over)
        + "\n按主题拆到 350 行以内，共用脚手架放 tests/_<topic>_common.py。"
    )
