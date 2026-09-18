"""Rewrite references to archived result directories inside Markdown docs.

The 2026-09-18 cleanup moved a set of ``results_*`` directories into
``_archive/2026-09-18/``.  Markdown docs still pointed at the old, now
non-existent paths.  This tool rewrites those references to the archived
location and prepends a one-line provenance notice.

Command-line arguments (``--out``, ``--old``, ``--new``, ``--dir``) are left
untouched: those are reproduction commands and must keep writing to the
original output name, not into the archive.

LaTeX sources are intentionally NOT touched: the manuscript will be updated as
a whole when the main scenario is re-based.

Usage:
    python tools/apply_dead_ref_fix.py            # dry run
    python tools/apply_dead_ref_fix.py --apply    # write files
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(ROOT, "tools", "_cleanup_dead_references.csv")
ARCHIVE_REL = "_archive/2026-09-18"

NOTICE = (
    "> **归档提示（2026-09-18）**：本文引用的部分 `results_*` 产物已移入 "
    f"`{ARCHIVE_REL}/`；正文中的路径引用已同步更新为归档位置，命令行示例里的 "
    "`--out` 目录仍写作历史原名（重跑时依旧输出到该名）。"
)

# A reference directly following one of these flags is a command argument.
ARG_FLAG_RE = re.compile(r"--(out|old|new|dir)=?$")


def collect() -> dict[str, list[str]]:
    """doc -> archived directory names, longest first (so prefixes lose)."""
    by_doc: dict[str, set[str]] = defaultdict(set)
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            doc = (row.get("doc") or "").strip()
            name = (row.get("archived") or "").strip()
            if not doc or not name or doc.endswith(".tex"):
                continue
            by_doc[doc].add(name)
    return {d: sorted(names, key=len, reverse=True) for d, names in by_doc.items()}


def build_pattern(names: list[str]) -> re.Pattern:
    alts = "|".join(re.escape(n) for n in names)
    # Do not match when preceded by a path separator (keeps
    # ``archive/results_target_local_v1_pre_rngfix_...`` and the already
    # rewritten ``_archive/.../results_x`` untouched), and require a non-word
    # ASCII suffix so longer names win over their prefixes.
    return re.compile(r"(?<![/A-Za-z0-9_])(" + alts + r")(?![A-Za-z0-9_-])")


def rewrite_line(line: str, pat: re.Pattern) -> tuple[str, int]:
    out: list[str] = []
    last = 0
    hits = 0
    for m in pat.finditer(line):
        if ARG_FLAG_RE.search(line[: m.start()].rstrip()):
            continue  # command argument: keep the historical output name
        out.append(line[last : m.start()])
        out.append(f"{ARCHIVE_REL}/{m.group(1)}")
        last = m.end()
        hits += 1
    out.append(line[last:])
    return "".join(out), hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes to disk")
    args = ap.parse_args()

    by_doc = collect()
    total_files = 0
    total_hits = 0
    missing: list[str] = []
    for doc in sorted(by_doc):
        path = os.path.join(ROOT, doc)
        if not os.path.exists(path):
            missing.append(doc)
            continue
        with open(path, encoding="utf-8") as fh:
            text = fh.read()

        pat = build_pattern(by_doc[doc])
        new_lines: list[str] = []
        hits = 0
        for line in text.split("\n"):
            new_line, n = rewrite_line(line, pat)
            hits += n
            new_lines.append(new_line)
        if hits == 0:
            continue
        new_text = "\n".join(new_lines)

        if ARCHIVE_REL not in text.split("\n", 1)[0]:
            new_text = NOTICE + "\n\n" + new_text

        total_hits += hits
        total_files += 1
        print(f"{'WRITE' if args.apply else 'DRY  '} {doc}: {hits} reference(s)")
        if args.apply:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(new_text)

    print(f"\nfiles touched: {total_files}, references rewritten: {total_hits}")
    if missing:
        print("missing docs: " + ", ".join(missing))
    if not args.apply:
        print("(dry run; re-run with --apply to write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
