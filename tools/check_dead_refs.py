"""Report ``results_*`` / ``archive/*`` references in Markdown that resolve nowhere."""
from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKIP_DIRS = {"_archive", ".git", ".workbuddy", "node_modules", "__pycache__"}
PAT = re.compile(r"(?<![/\w])((?:results|archive)[A-Za-z0-9_.-]*)(?![A-Za-z0-9_-])")

missing: dict[str, set[str]] = {}
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if not fn.endswith(".md"):
            continue
        path = os.path.join(dirpath, fn)
        rel = os.path.relpath(path, ROOT)
        try:
            text = open(path, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in PAT.finditer(text):
            name = m.group(1)
            if "<" in name or "*" in name or "{" in name:
                continue
            if name.endswith("_") or name in {"results", "results_", "archived"}:
                continue  # wildcard prefixes such as ``results_v1_lever_closure_*``
            first = name.split("/")[0]
            if os.path.exists(os.path.join(ROOT, first)) or os.path.exists(
                os.path.join(ROOT, "_archive", "2026-09-18", first)
            ):
                continue
            missing.setdefault(rel, set()).add(name)

if not missing:
    print("no unresolved results_* references in Markdown")
for doc in sorted(missing):
    print(f"{doc}: " + ", ".join(sorted(missing[doc])))
