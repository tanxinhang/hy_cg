"""``studies/`` 归档的卫生门禁。

背景（2026-09-21 审计发现）：方向 1 的 5 份审计文档被**双重编码**损坏
（UTF-8 字节被当 GB18030 解码后重新存盘，且无法映射的字节被换成 `?`，信息已丢），
且有两处 README 引用了**不存在**的数据目录。两者都是"当时没人检查、后来才发现"的
腐烂 —— 本文件把这类腐烂变成会 FAIL 的断言。

覆盖三条：
1. 编码：不得出现**新的**乱码文档（已损坏的 5 份进清单，只许变短）。
2. 引用：README / 审计文档里写出的 ``data/<dir>`` 必须真实存在。
3. 归属：``scripts/`` 下的每个脚本都必须被本方向的 README 索引到。
"""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "studies")

#: GB18030 解码 UTF-8 字节时产生的特征字符（正确编码的中文文档几乎不会写出它们）
MARKERS = "\u93c4\u951b\u9225\u7468\u94a8\u8ff9"

#: 已损坏清单（只许变短）：2026-09-21 发现，无法无损还原，正文已加横幅。
KNOWN_CORRUPT = {
    "studies/direction1/docs/AUDIT_DIRECTION1_DELTA.md",
    "studies/direction1/docs/AUDIT_DIRECTION1_FAILURE_ROOT_CAUSE.md",
    "studies/direction1/docs/AUDIT_REAL_VS_EXPECTED_GAP.md",
    "studies/direction1/docs/AUDIT_TPUIC_RESIDUAL_ACCOUNTING.md",
    "studies/direction1/docs/TPUIC_RESIDUAL_ACCOUNTING_AUDIT.md",
    "studies/direction3/TPUIC_FEASIBILITY_VERDICT.md",
    "studies/direction3/V2_RESEARCH_FRAMEWORK.md",
}

_TOKEN = re.compile(r"data/([A-Za-z0-9_\-]+)")


def _rel(p: str) -> str:
    return os.path.relpath(p, ROOT).replace(os.sep, "/")


def _md_files() -> list[str]:
    out: list[str] = []
    for base in (STUDIES, os.path.join(ROOT, "docs")):
        if not os.path.isdir(base):
            continue
        for r, ds, fs in os.walk(base):
            ds[:] = [d for d in ds if d != "__pycache__"]
            out.extend(os.path.join(r, f) for f in fs if f.endswith(".md"))
    return sorted(out)


def _mojibake_score(text: str) -> int:
    return sum(text.count(c) for c in MARKERS)


def test_no_new_mojibake_documents():
    """不得再长出新的乱码文档；已损坏清单只许变短。"""
    bad = []
    for p in _md_files():
        try:
            text = open(p, encoding="utf-8").read()
        except UnicodeDecodeError:
            bad.append(f"{_rel(p)}（不是合法 UTF-8）")
            continue
        if _mojibake_score(text) and _rel(p) not in KNOWN_CORRUPT:
            bad.append(f"{_rel(p)}（双重编码特征 {_mojibake_score(text)} 处）")
    assert not bad, (
        "出现新的编码损坏文档。修复办法：若源还在，用 UTF-8 重新导出；\n"
        "  若已无法还原，加损坏横幅并把路径登记进 KNOWN_CORRUPT。\n  "
        + "\n  ".join(bad)
    )
    missing = sorted(k for k in KNOWN_CORRUPT
                     if not os.path.isfile(os.path.join(ROOT, k)))
    assert not missing, (
        "KNOWN_CORRUPT 里的文件已不存在（清单腐烂，请删掉对应行）：\n  "
        + "\n  ".join(missing)
    )


def test_readme_data_references_exist():
    """文档里引用的 ``data/<dir>`` 必须真实存在（防止改名/迁移后留下死链）。"""
    broken: list[str] = []
    for p in _md_files():
        text = open(p, encoding="utf-8").read()
        # 引用相对本方向根目录：studies/directionN/data/<token>
        parts = _rel(p).split("/")
        if len(parts) < 2 or not parts[1].startswith("direction"):
            continue
        base = os.path.join(STUDIES, parts[1])
        for m in _TOKEN.finditer(text):
            if m.end() < len(text) and text[m.end()] == "*":
                continue  # 通配引用，如 data/robust_verdict_*
            d = os.path.join(base, "data", m.group(1))
            if not os.path.isdir(d):
                broken.append(f"{_rel(p)} -> data/{m.group(1)}")
    assert not broken, "文档引用了不存在的数据目录：\n  " + "\n  ".join(sorted(set(broken)))


def test_every_study_script_is_indexed_by_its_readme():
    """``scripts/`` 下的脚本必须被本方向 README 提到 —— 不留无主的探针。"""
    orphans: list[str] = []
    for direction in sorted(os.listdir(STUDIES)):
        d = os.path.join(STUDIES, direction)
        scripts = os.path.join(d, "scripts")
        readme = os.path.join(d, "README.md")
        if not os.path.isdir(scripts) or not os.path.isfile(readme):
            continue
        text = open(readme, encoding="utf-8").read()
        for f in sorted(os.listdir(scripts)):
            if not f.endswith(".py"):
                continue
            if f not in text:
                orphans.append(f"{direction}/scripts/{f}")
    assert not orphans, (
        "下列脚本没有被本方向 README 索引（要么补进索引，要么删掉）：\n  "
        + "\n  ".join(orphans)
    )
