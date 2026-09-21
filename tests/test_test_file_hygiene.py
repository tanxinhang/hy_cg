"""``docs/TEST_STANDARD.md`` 里可静态检查的那几条，做成会 FAIL 的断言。

规范的失败模式是"写下来但没人执行"。本文件把其中**可以用 AST 判定**的部分
（文件骨架、命名、导入方向、unittest 风格收尾）钉住；判定不了的（断言档位、
性能预算、docstring 内容质量）留在 review 环节。

覆盖：
1. **W1** 每个 ``test_*.py`` 必须有模块 docstring（写"钉什么/为什么"）。
2. **W2** 测试名与 class 名一律**英文 ASCII**。
3. **W6** 用了 ``unittest.TestCase`` 的文件必须以 ``if __name__ == "__main__"`` 结尾。
4. **F5** 禁止从 ``test_*.py`` 模块 import（共用代码只能走 ``_<topic>_common.py``）。
5. **F3** 行数不在这里卡（``test_module_size.py`` 已有 350 行红线），只顺带确认脚手架
   与测试文件的命名前缀不混用。
"""
from __future__ import annotations

import ast
import os
import re

TESTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")

#: W2：测试名必须是 ASCII 的 snake_case（允许 ``_N`` / ``_L`` 这类单字母物理量后缀）
_TEST_NAME = re.compile(r"^test[A-Za-z0-9_]*$")
#: class 名必须是 ASCII 标识符；``_`` 开头的是私有辅助类（如 ``_FakeResult``），不参与收集
_CLASS_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_MIN_DOCSTRING = 40


def _test_files() -> list[str]:
    return sorted(
        os.path.join(TESTS, f)
        for f in os.listdir(TESTS)
        if f.startswith("test_") and f.endswith(".py")
    )


def _parse(path: str):
    src = open(path, encoding="utf-8").read()
    return src, ast.parse(src)


def _names(tree: ast.Module):
    fns, classes = [], []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fns.append(node)
        elif isinstance(node, ast.ClassDef):
            classes.append(node)
    return fns, classes


def test_every_test_file_has_a_module_docstring():
    """W1：模块 docstring 是"这个文件钉什么契约"的唯一说明，不能省。"""
    bad = []
    for path in _test_files():
        src, tree = _parse(path)
        doc = ast.get_docstring(tree) or ""
        if len(doc.strip()) < _MIN_DOCSTRING:
            bad.append(f"{os.path.basename(path)}（docstring {len(doc.strip())} 字，需 ≥{_MIN_DOCSTRING}）")
    assert not bad, (
        "下列测试文件缺少模块 docstring（须写明钉什么契约 / 为什么）：\n  "
        + "\n  ".join(bad)
    )


def test_test_names_and_class_names_are_ascii():
    """W2：读名即知断言，且不得混进中文标识符（搜索与引用都会痛）。"""
    bad: list[str] = []
    for path in _test_files():
        _, tree = _parse(path)
        fns, classes = _names(tree)
        for fn in fns:
            if fn.name.startswith("test") and not _TEST_NAME.match(fn.name):
                bad.append(f"{os.path.basename(path)}::{fn.name}")
            elif not _TEST_NAME.match(fn.name):
                # 非测试函数只要求 ASCII，不要求 test 前缀
                if not fn.name.isascii():
                    bad.append(f"{os.path.basename(path)}::{fn.name}（非 ASCII）")
        for cls in classes:
            if cls.name.startswith("_"):
                continue  # 私有辅助类（如 _FakeResult），不参与收集，不卡命名风格
            if not _CLASS_NAME.match(cls.name) or not cls.name.isascii():
                bad.append(f"{os.path.basename(path)}::{cls.name}（class 名）")
    assert not bad, "测试名/class 名必须是英文 ASCII（snake_case / CapWords）：\n  " + "\n  ".join(bad)


def test_unittest_style_files_end_with_a_main_guard():
    """W6：用了 TestCase 的文件要能单文件直跑（调试一个契约时不跑全量）。"""
    bad = []
    for path in _test_files():
        src, tree = _parse(path)
        uses_unittest = any(
            isinstance(node, ast.ImportFrom) and node.module == "unittest"
            for node in ast.walk(tree)
        ) or any(
            isinstance(node, ast.ClassDef)
            and any(
                isinstance(b, ast.Name) and b.id.startswith("TestCase")
                for b in node.bases
            )
            for node in ast.walk(tree)
        )
        if uses_unittest and '__name__ == "__main__"' not in src:
            bad.append(os.path.basename(path))
    assert not bad, (
        "下列文件用了 unittest 但没有 main 守卫（无法单文件直跑）：\n  "
        + "\n  ".join(bad)
    )


def test_no_test_module_imports_another_test_module():
    """F5：共用代码只能放 ``_<topic>_common.py``。

    从 ``test_*.py`` import 会让共用代码被 pytest 二次收集，并把两个互不相关的
    契约耦在一起 —— 拆文件时这正是最容易留下的坑。
    """
    bad: list[str] = []
    for path in _test_files():
        _, tree = _parse(path)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("test"):
                    bad.append(f"{os.path.basename(path)}:{node.lineno} from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("test"):
                        bad.append(f"{os.path.basename(path)}:{node.lineno} import {alias.name}")
    assert not bad, (
        "禁止从 test_*.py 模块 import（共用代码下沉到 tests/_<topic>_common.py）：\n  "
        + "\n  ".join(bad)
    )


def test_shared_scaffolding_lives_in_underscore_modules():
    """F4：脚手架必须下划线开头（不参与收集、不占 350 行预算）。"""
    bad = []
    for f in sorted(os.listdir(TESTS)):
        if not f.endswith(".py") or f.startswith("test_"):
            continue
        if not f.startswith("_"):
            bad.append(f)
    assert not bad, (
        "tests/ 下的非测试模块必须以 _ 开头（否则会被当成脚手架却参与收集）：\n  "
        + "\n  ".join(bad)
    )
