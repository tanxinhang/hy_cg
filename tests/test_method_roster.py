"""方法名册一致性门禁。

背景（2026-09-20）
------------------
"方法名册"曾经在**三个地方**各存一份，彼此没有任何机制保证一致：

| 位置 | 形态 | 规模 |
|---|---|---|
| `isac_sim/core/config.py`（现 `core/config/aliases.py`） | `MethodName` Literal（类型别名） | 28 个名字 |
| `isac_sim/core/naming.py`（现 `experiments/app/naming.py`） | `METHOD_ORDER` + `LABELS`（展示表） | 26 + 36 键 |
| `isac_sim/detection/fusion.py`（现 `experiments/methods/policy.py`） | `fusion_weight_mode_for_method`（方法名→融合策略） | 3 个分支 |

后果不只是丑：**积木库认识了实验名册**。加一个方法要手改四处，漏一处不报错；
而且 `MethodName` 从未被运行时读取（纯注解），`primitives.py` / `selection.py`
里那两处 `import MethodName` 是**死导入** —— 说明名册是被"顺手带进来的"，
不是被需要的。

现在：名册只住在 `experiments/methods/`，积木库里**一个方法名字符串都不许有**。
本文件把这些约束钉死。
"""

from __future__ import annotations

import os
from typing import get_args

import pytest

from experiments.methods import (
    C2F_METHODS,
    DEFAULT_METHODS,
    EXPERIMENTAL_METHODS,
    METHOD_RNG_OFFSETS,
    METHODS,
    MethodName,
    fusion_weight_mode_for_method,
)
from experiments.app.naming import LABELS, METHOD_ORDER

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAC = os.path.join(ROOT, "isac_sim")
SKIP_DIRS = {"__pycache__", ".git", ".workbuddy", "venv", "results", "_archive"}


def test_literal_covers_the_roster_exactly():
    """``MethodName`` 的成员必须恰好等于 METHODS ∪ EXPERIMENTAL_METHODS。"""
    literal = set(get_args(MethodName))
    roster = set(METHODS) | set(EXPERIMENTAL_METHODS)
    assert literal == roster, (
        f"名册与 Literal 不一致：\n"
        f"  只在 Literal 里：{sorted(literal - roster)}\n"
        f"  只在名册里：{sorted(roster - literal)}"
    )


def test_roster_lists_are_self_consistent():
    """名册的派生表必须是 METHODS 的子集，且默认名册真的被包含。"""
    assert set(METHODS) <= set(get_args(MethodName))
    assert set(EXPERIMENTAL_METHODS) <= set(METHODS), "candidate 方法必须也出现在 METHODS 里"
    assert set(DEFAULT_METHODS) <= set(METHODS)
    assert set(DEFAULT_METHODS).isdisjoint(EXPERIMENTAL_METHODS), (
        "默认名册不得包含 experimental 方法 —— 否则发布口径会随候选能力漂移"
    )
    assert not (set(METHODS) - set(METHOD_RNG_OFFSETS)), (
        f"以下方法没有独立 RNG 流：{sorted(set(METHODS) - set(METHOD_RNG_OFFSETS))}"
    )
    assert set(C2F_METHODS) <= set(METHODS)


def test_display_tables_only_reference_known_methods():
    """展示表不得出现名册以外的名字（防拼写漂移）。"""
    unknown_order = sorted(set(METHOD_ORDER) - set(METHODS))
    assert not unknown_order, f"METHOD_ORDER 里有名册外的名字：{unknown_order}"
    missing_label = sorted(set(METHODS) - set(LABELS))
    assert not missing_label, f"这些方法没有展示标签：{missing_label}"


def test_fusion_weight_mode_policy_is_total():
    """方法名→融合策略表必须对名册全定义，不得静默落到默认分支。"""
    modes = {m: fusion_weight_mode_for_method(m) for m in METHODS}
    assert set(modes.values()) <= {"deflection", "equal", "exact_llr_sum"}
    # 只有这两个方法走非 deflection 分支；新增方法默认 deflection 是刻意的，
    # 但这里把它显式登记下来，避免"以为改了策略其实没改"。
    special = {m for m, mode in modes.items() if mode != "deflection"}
    assert special == {"raw_sense_sinr", "joint_bundle_cg_exact_llr"}, (
        f"非 deflection 融合策略的方法集合变了：{sorted(special)}"
    )


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(p, id=os.path.relpath(p, ROOT))
        for p in [
            os.path.join(r, f)
            for r, ds, fs in os.walk(ISAC)
            for f in fs
            if f.endswith(".py") and not any(d in SKIP_DIRS for d in r.split(os.sep))
        ]
    ],
)
def test_library_never_mentions_a_method_name(path: str):
    """积木库里不得出现任何方法名字符串 —— 名册是实验记账，不是积木。"""
    src = open(path, encoding="utf-8", errors="replace").read()
    leaked = sorted({n for n in METHODS if f'"{n}"' in src})
    assert not leaked, (
        f"{os.path.relpath(path, ROOT)} 出现了方法名 {leaked}；"
        "积木库不该认识方法名（策略/展示请放到 experiments/）"
    )


def test_library_has_no_method_name_type():
    """``MethodName`` 不得出现在积木库里（它是名册的类型，属实验层）。"""
    offenders = []
    for r, ds, fs in os.walk(ISAC):
        ds[:] = [d for d in ds if d not in SKIP_DIRS]
        for f in fs:
            if not f.endswith(".py"):
                continue
            p = os.path.join(r, f)
            for i, line in enumerate(
                open(p, encoding="utf-8", errors="replace").read().split("\n"), 1
            ):
                if "MethodName" in line:
                    offenders.append(f"{os.path.relpath(p, ROOT)}:{i}: {line.strip()[:80]}")
    assert not offenders, "积木库里仍引用 MethodName：\n  " + "\n  ".join(offenders)
