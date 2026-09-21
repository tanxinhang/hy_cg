"""``docs/ITERATION_PLAYBOOK.md`` §5.2 的版本控制规则（V-1/V-2）做成门禁。

**为什么值得钉**：文件管理的失败模式是**沉默的** —— 东西在磁盘上、测试也绿，
但它从来没进过版本控制；等换机器 / 回滚 / 让人 clone 时才发现整个目录消失了。
本仓实测（2026-09-21）：``docs/`` 与 ``studies/`` 两个目录**整体未被 git 跟踪**，
重构删掉的旧模块（43 个）与旧测试（19 个）也未登记删除 ⇒ clone 出来是重构前的结构。

覆盖两条：
1. **V-1 必须入库**：源码 / 规范 / 测试 / 方向产物下不得有"从没 add 过"的文件。
2. **V-2 删除必须登记**：上述路径下不得有未登记的删除（重构换包后旧路径要一起 add）。

只在 git 仓库里生效；非仓库环境（如打包导出）自动跳过。
"""
from __future__ import annotations

import os
import subprocess
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: 必须受版本控制的顶层路径（``.workbuddy/`` 故意不在其中 —— 它已被 .gitignore 排除）
TRACKED_PATHS = (
    "isac_sim",
    "experiments",
    "tests",
    "docs",
    "studies",
    "tools",
    "release",
    "run_isac_sim.py",
)

#: 有意不入库的噪声（即便出现在上述路径下也不算违规）
NOISE_SUFFIX = (".pyc", ".pyo", ".log", ".bak")
NOISE_DIR = "__pycache__"


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError:
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def _is_repo() -> bool:
    return (_git("rev-parse", "--is-inside-work-tree") or "").strip() == "true"


def _status_lines(paths: tuple[str, ...]) -> list[tuple[str, str]]:
    """``[(状态, 路径)]``，只保留未跟踪(??)与删除(D)，并剔除噪声。"""
    raw = _git("status", "--porcelain", "--", *paths) or ""
    out: list[tuple[str, str]] = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        state = line[:2]
        path = line[3:].strip().replace("\\", "/")
        if NOISE_DIR in path.split("/") or path.endswith(NOISE_SUFFIX):
            continue
        if "?" in state:
            out.append(("untracked", path))
        elif "D" in state:
            out.append(("deleted", path))
    return out


class RepoTracking(unittest.TestCase):
    def test_source_and_docs_have_no_untracked_files(self) -> None:
        """V-1：新写的源码 / 规范 / 测试 / 方向产物必须已经 git add。"""
        if not _is_repo():
            self.skipTest("not a git working tree")
        untracked = [p for s, p in _status_lines(TRACKED_PATHS) if s == "untracked"]
        self.assertEqual(
            [], untracked,
            "下列文件在磁盘上但从未加入版本控制（换机/回滚会直接丢失）：\n  "
            + "\n  ".join(untracked[:40])
            + ("\n  ..." if len(untracked) > 40 else "")
            + "\n修法：git add <上面的路径>（若确属不入库的产物，请加进 .gitignore）。",
        )

    def test_deletions_are_recorded(self) -> None:
        """V-2：重构删掉的文件必须登记删除，否则 clone 出来是旧结构。"""
        if not _is_repo():
            self.skipTest("not a git working tree")
        deleted = [p for s, p in _status_lines(TRACKED_PATHS) if s == "deleted"]
        self.assertEqual(
            [], deleted,
            "下列文件已从磁盘删除但 git 仍当作存在（未登记删除）：\n  "
            + "\n  ".join(deleted[:40])
            + ("\n  ..." if len(deleted) > 40 else "")
            + "\n修法：git add -A <上面的路径>（只登记删除，不是提交）。",
        )


if __name__ == "__main__":
    unittest.main()
