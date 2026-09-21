"""未定义名门禁（批量改 import 后的兜底）。

为什么需要它
------------
2026-09-20 的分层重构里发生过两次同一类事故：

1. 批量把多行 ``import`` 整段替换掉，``simulate.py`` 与 ``test_target_local_v1.py``
   的 solver 导入被吞掉 —— ``pytest --collect-only`` 全绿（190 tests / 0 error），
   全量跑才炸出 ``NameError``。
2. ``selection.py`` 拆分时，``primitives.py`` 只保留了块内直接出现的符号，
   ``selection_utility`` / ``selection_utility_from_pd`` / ``predicted_pd_for_links``
   被漏掉 —— 这次连**全量 pytest 都全绿**（198 passed），是端到端 CLI 冒烟才炸的。

两次都是同一个形状：**函数体里引用了某个名字，而它没有出现在任何导入/定义里**。
静态地查这个形状，比等某条测试路径跑到更可靠。

判据（刻意保守，几乎不误报）
----------------------------
把整个文件里所有**被绑定**的名字收集起来（导入、def/class、赋值、参数、
for/with/except 目标、推导式、海象、match 模式），再看每个 ``Name`` 的 Load
是否落在其中。不做作用域精算 —— 只要"整个文件从未绑定过"，就是漏了导入。
"""

from __future__ import annotations

import ast
import builtins
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGES = ("isac_sim", "experiments", "audits")
SKIP_DIRS = {"__pycache__", ".git", ".workbuddy", "venv", "results", "_archive"}

_BUILTINS = set(dir(builtins)) | {
    "__file__", "__name__", "__doc__", "__package__", "__spec__",
    "__loader__", "__builtins__", "__debug__", "__annotations__", "__all__",
}


def _bound_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names.add(n.id)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names.update(n.names)
        elif isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name):
            names.add(n.target.id)
        elif isinstance(n, ast.MatchAs) and n.name:
            names.add(n.name)
        elif isinstance(n, ast.MatchStar) and n.name:
            names.add(n.name)
    return names


def _undefined_in(path: str) -> list[str]:
    src = open(path, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    have = _bound_names(tree) | _BUILTINS
    return [
        f"L{n.lineno}: 未定义的名字 {n.id!r}"
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in have
    ]


def _iter_modules():
    for base in PACKAGES:
        bp = os.path.join(ROOT, base)
        if not os.path.isdir(bp):
            continue
        for r, ds, fs in os.walk(bp):
            ds[:] = [d for d in ds if d not in SKIP_DIRS]
            for f in sorted(fs):
                if f.endswith(".py"):
                    yield os.path.join(r, f)


@pytest.mark.parametrize(
    "path", [pytest.param(p, id=os.path.relpath(p, ROOT)) for p in _iter_modules()]
)
def test_no_undefined_names(path: str):
    problems = _undefined_in(path)
    assert not problems, (
        f"{os.path.relpath(path, ROOT)} 存在未绑定的名字（多半是漏了导入）：\n  "
        + "\n  ".join(problems)
    )
