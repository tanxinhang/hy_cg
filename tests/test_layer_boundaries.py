"""架构不变式（architecture invariants）。

这些测试不检查任何数字，只检查**结构**。它们存在的理由很具体：
2026-09-20 的分层重构暴露了"目录搬了、层边界没搬"的问题 ——
235 处引用仍走旧扁平路径，靠 ``__init__.py`` 里的 ``sys.modules`` 别名兜着，
于是写错层**不会报错**，分层无法自我维持。这里把它变成会报错的东西。

对应工具 gates：``tools/check_contract_refs.py``（文件位置契约）、
``tools/check_release_identity.py``（冻结键契约）。本文件补的是**依赖方向契约**。
"""

from __future__ import annotations

import ast
import dataclasses
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ISAC = os.path.join(ROOT, "isac_sim")

#: 包外目录：``isac_sim`` 永远不许依赖它们。
OUTSIDE = ("experiments", "audits", "tools", "tests")

#: 积木库里不许出现的 IO / 展示依赖。
IO_MODULES = ("argparse", "matplotlib", "plotly", "seaborn", "csv", "pickle")

#: 旧扁平模块名。出现即说明兼容层又在被依赖。
LEGACY_FLAT = (
    "config", "naming", "belief", "prior", "model", "dd", "waveform", "aperture",
    "soft_channel", "fbl", "cancellation", "cancellation_glrt", "fusion", "llr",
    "corr", "oracle", "reporting", "selection", "coordination", "bundle_master",
    "fusion_polish", "simulate", "experiments", "packetization", "theory",
    "cli", "report", "plotting",
)

#: 层序（由内向外）。只用来判断"向上依赖"，不是严格的 DAG 声明。
LAYER_ORDER = {
    "core": 0,
    "scenario": 1,
    "sensing": 2,
    "receiver": 3,
    "detection": 3,
    "cooperation": 4,
}

#: 已知的向上依赖（2026-09-20 实测）。**只许减少，不许新增。**
#: 这一组说明子包之间还不是 DAG —— 收敛它们是下一阶段的事，
#: 但至少不能再多长出来。
KNOWN_UPWARD_EDGES = {
    ("scenario", "sensing"),
    ("sensing", "receiver"),
    ("sensing", "detection"),
    ("sensing", "cooperation"),
    ("detection", "cooperation"),
}


def _iter_py(base: str):
    for r, ds, fs in os.walk(base):
        ds[:] = [d for d in ds if d != "__pycache__"]
        for f in fs:
            if f.endswith(".py"):
                yield os.path.join(r, f)


def _imports(path: str):
    """产出 (import 语句所在的绝对模块名, lineno, 被导入的模块名, 是否相对导入)。"""
    src = open(path, encoding="utf-8", errors="replace").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names = [a.name for a in node.names]
            yield node.module or "", node.lineno, names, bool(node.level)
        elif isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, node.lineno, [a.name], False


def _subpackage(path: str) -> str:
    rel = os.path.relpath(path, ISAC).replace(os.sep, "/")
    return rel.split("/")[0] if "/" in rel else "(root)"


def test_isac_sim_does_not_depend_on_outside_packages():
    """积木库不得依赖流程/审计/工具/测试 —— 依赖方向只能从外向内。"""
    violations = []
    for path in _iter_py(ISAC):
        for module, lineno, _names, _rel in _imports(path):
            top = module.split(".")[0]
            if top in OUTSIDE:
                violations.append(f"{os.path.relpath(path, ROOT)}:{lineno}: {module}")
    assert not violations, (
        "isac_sim 反向依赖了包外模块（层次倒置）：\n  " + "\n  ".join(violations)
    )


def test_isac_sim_has_no_io_or_presentation():
    """积木库不得做 IO 或出图；这些属于 experiments/app。"""
    violations = []
    for path in _iter_py(ISAC):
        src = open(path, encoding="utf-8", errors="replace").read()
        tree = ast.parse(src)
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.ImportFrom):
                mods = [node.module or ""]
            elif isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            for m in mods:
                if m.split(".")[0] in IO_MODULES:
                    violations.append(f"{os.path.relpath(path, ROOT)}:{node.lineno}: import {m}")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in ("open", "print"):
                    violations.append(
                        f"{os.path.relpath(path, ROOT)}:{node.lineno}: {node.func.id}()"
                    )
    assert not violations, "isac_sim 里出现了 IO/展示代码：\n  " + "\n  ".join(violations)


def test_no_legacy_flat_module_paths():
    """全仓不得再出现 ``isac_sim.<旧扁平名>`` 形式的导入。"""
    import re

    pat = re.compile(r"\bisac_sim\.([A-Za-z_]\w*)")
    violations = []
    for base in ("isac_sim", *OUTSIDE):
        bp = os.path.join(ROOT, base)
        if not os.path.isdir(bp):
            continue
        for path in _iter_py(bp):
            for i, line in enumerate(
                open(path, encoding="utf-8", errors="replace").read().split("\n"), 1
            ):
                stripped = line.strip()
                if not stripped.startswith(("from ", "import ")):
                    continue
                for m in pat.finditer(line):
                    if m.group(1) in LEGACY_FLAT:
                        violations.append(
                            f"{os.path.relpath(path, ROOT)}:{i}: {stripped[:80]}"
                        )
    assert not violations, (
        "仍有旧的扁平路径（说明兼容层又被依赖了）：\n  " + "\n  ".join(violations)
    )


def test_package_uses_absolute_imports_only():
    """isac_sim / experiments / audits 内部一律绝对导入 —— 层边界必须在每条 import 上可见。"""
    import re

    rel_pat = re.compile(r"^\s*from \.+[A-Za-z_]", re.M)
    violations = []
    for base in ("isac_sim", "experiments", "audits"):
        bp = os.path.join(ROOT, base)
        if not os.path.isdir(bp):
            continue
        for path in _iter_py(bp):
            src = open(path, encoding="utf-8", errors="replace").read()
            for m in rel_pat.finditer(src):
                lineno = src[: m.start()].count("\n") + 1
                violations.append(f"{os.path.relpath(path, ROOT)}:{lineno}")
    assert not violations, (
        "仍有相对导入（层边界不可见）：\n  " + "\n  ".join(violations)
    )


def test_no_new_upward_layer_edges():
    """子包之间的向上依赖不得新增（当前已知的一组在 KNOWN_UPWARD_EDGES 里）。"""
    edges = set()
    for path in _iter_py(ISAC):
        src_pkg = _subpackage(path)
        if src_pkg not in LAYER_ORDER:
            continue  # types.py 等包根模块只做 TYPE_CHECKING 契约，不参与排序
        for module, _lineno, _names, _rel in _imports(path):
            parts = module.split(".")
            if len(parts) < 2 or parts[0] != "isac_sim":
                continue
            dst = parts[1]
            if dst not in LAYER_ORDER or dst == src_pkg:
                continue
            if LAYER_ORDER[dst] > LAYER_ORDER[src_pkg]:
                edges.add((src_pkg, dst))
    new = edges - KNOWN_UPWARD_EDGES
    assert not new, (
        "出现了新的向上依赖（层序被破坏）：\n  "
        + "\n  ".join(f"{a} -> {b}" for a, b in sorted(new))
    )


def test_certificate_contract_is_enforced():
    """调度器接口只有两个量；投影函数必须真的拒绝残缺对象。"""
    from isac_sim import types as contract

    assert contract.CERTIFICATE_FIELDS == ("residual_power", "target_retention")
    assert set(contract.CERTIFICATE_SOURCE_FIELDS) == set(contract.CERTIFICATE_FIELDS)

    @dataclasses.dataclass
    class _FakeResult:
        i_res: float
        eta_survive: float

    view = contract.certificate_view(_FakeResult(0.25, 0.75))
    assert view.residual_power == pytest.approx(0.25)
    assert view.target_retention == pytest.approx(0.75)
    assert dataclasses.fields(contract.CertificateView).__len__() == 2

    with pytest.raises(AttributeError):
        contract.certificate_view(object())


def test_certificate_contract_matches_real_implementation():
    """契约里写的源字段必须真的存在于 CancellationResult 上。"""
    from isac_sim.receiver.cancellation import CancellationResult
    from isac_sim.types import CERTIFICATE_SOURCE_FIELDS

    field_names = {f.name for f in dataclasses.fields(CancellationResult)}
    missing = set(CERTIFICATE_SOURCE_FIELDS.values()) - field_names
    assert not missing, f"CancellationResult 缺少契约声明的字段：{sorted(missing)}"


def test_single_production_path():
    """proposed 只能有一个实现入口：experiments/flow/pipeline.py。"""
    candidates = []
    for base in ("isac_sim", "experiments", "audits", "tools"):
        bp = os.path.join(ROOT, base)
        if not os.path.isdir(bp):
            continue
        for path in _iter_py(bp):
            src = open(path, encoding="utf-8", errors="replace").read()
            if "def run_proposed_trial" in src:
                candidates.append(os.path.relpath(path, ROOT).replace(os.sep, "/"))
    assert candidates == ["experiments/flow/pipeline.py"], (
        "run_proposed_trial 必须只有一个定义处，实际：" + repr(candidates)
    )


def _package_exists(dotted: str) -> bool:
    base = os.path.join(ROOT, dotted.replace(".", os.sep))
    return os.path.exists(base + ".py") or os.path.exists(os.path.join(base, "__init__.py"))


def test_import_targets_resolve():
    """每一句 ``from a.b.c import ...`` 的目标模块必须真的存在。

    这条门禁拦的是一类**语法合法、只在运行到那一行时才炸**的事故：批量改写
    相对导入时算错层级，生成了指向不存在包的语句。本次实测在
    ``experiments/flow/sweeps.py`` 的函数体里藏了 10 处
    ``from experiments.sensing.model import ...``（应为 ``isac_sim.sensing.model``），
    逐模块 import 与全量 pytest 都没碰到 —— 只有跑到那个 sweep 模式才会炸。
    """
    prefixes = ("isac_sim", "experiments", "audits")
    missing = []
    for base in (*prefixes, "tools", "tests"):
        bp = os.path.join(ROOT, base)
        if not os.path.isdir(bp):
            continue
        for path in _iter_py(bp):
            for module, lineno, _names, is_rel in _imports(path):
                if is_rel or not module:
                    continue
                if module.split(".")[0] not in prefixes:
                    continue
                if _package_exists(module):
                    continue
                missing.append(f"{os.path.relpath(path, ROOT)}:{lineno}: {module}")
    assert not missing, (
        "导入了不存在的本仓模块（多半是相对/绝对路径算错）：\n  " + "\n  ".join(missing)
    )
