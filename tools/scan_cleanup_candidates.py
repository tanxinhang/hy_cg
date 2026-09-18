"""Classify every results_* artifact by (hardware gain, footprint, RCS).

READ-ONLY.  Emits a CSV + prints a grouped report.  Nothing is moved or
deleted here; the operator reviews the CSV and then decides.

Classification
--------------
A. hardware-gain contaminated : any artifact whose config (or arm cell)
   sets radar_net_gain_db / radar_tx_gain_dbi / radar_rx_gain_dbi /
   radar_system_loss_db to a non-default value, or whose arm/cell name
   mentions gain15 / gain17p5 / g15 / hw gain.
B. out-of-band geometry        : area_xy > 800 m OR target_rcs > 1 m^2.
C. in-band and clean           : 500-800 m x RCS 0.05-0.2 with G_hw = 0 dB.
U. unknown                     : no config/summary could be read.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

HW_KEYS = ("radar_net_gain_db", "radar_tx_gain_dbi", "radar_rx_gain_dbi",
           "radar_system_loss_db")
# default release values: net None, tx 0, rx 0, loss 0  => G_hw = 0 dB
HW_DEFAULT = {"radar_net_gain_db": None, "radar_tx_gain_dbi": 0.0,
              "radar_rx_gain_dbi": 0.0, "radar_system_loss_db": 0.0}

GAIN_NAME_RE = re.compile(
    r"gain\s*1[57](?:[._p]?5)?|gain15|gain17p5|_g(?:5|15)(?![0-9])|hwgain|"
    r"hardware[-_]?gain", re.I)
# any explicit non-zero hardware-gain number inside a directory name, e.g. g5/g15
DIR_GAIN_NUM_RE = re.compile(r"_g(\d+(?:\.\d+)?)(?![0-9])", re.I)

AREA_MAX = 800.0
RCS_MAX = 1.0


@dataclass
class Probe:
    areas: set = field(default_factory=set)
    rcss: set = field(default_factory=set)
    hw: set = field(default_factory=set)     # (key, value) that differ from default
    sources: list = field(default_factory=list)


def _coerce(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    return v


def scan_json_obj(obj: dict, probe: Probe, src: str):
    """Pull geometry / rcs / hardware fields out of an arbitrary nested dict."""
    if not isinstance(obj, dict):
        return
    for key, val in obj.items():
        lk = key.lower()
        if isinstance(val, dict):
            scan_json_obj(val, probe, src)
            continue
        if lk in ("area_xy", "area", "side_m", "footprint_m"):
            if isinstance(val, (int, float)):
                probe.areas.add(float(val))
        elif lk in ("target_rcs", "rcs", "rcs_m2", "sigma_rcs"):
            if isinstance(val, (int, float)):
                probe.rcss.add(float(val))
        elif lk in HW_KEYS:
            probe.hw.add((lk, _coerce(val)))


def scan_text(path: Path, probe: Probe):
    """Grep a flat text/CSV file for gain-ish tokens and geometry hints."""
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return
    if GAIN_NAME_RE.search(txt):
        probe.sources.append(f"file:{path.name}:gain-token")
    for m in re.finditer(r"area_xy\D{0,4}(\d+(?:\.\d+)?)", txt):
        probe.areas.add(float(m.group(1)))
    for m in re.finditer(r"target_rcs\D{0,4}(\d+(?:\.\d+)?)", txt):
        probe.rcss.add(float(m.group(1)))


def probe_dir(d: Path) -> Probe:
    p = Probe()
    if d.is_file():
        if d.suffix.lower() in (".csv", ".txt", ".log", ".md", ".tex"):
            scan_text(d, p)
        return p
    # 1) structured config / summary files
    for name in ("config.json", "summary.json", "meta.json", "protocol.json"):
        f = d / name
        if f.exists():
            try:
                scan_json_obj(json.loads(f.read_text(encoding="utf-8")), p, name)
            except Exception:
                pass
    for f in sorted(d.rglob("*.json"))[:40]:
        if f.stat().st_size > 4_000_000:
            continue
        try:
            scan_json_obj(json.loads(f.read_text(encoding="utf-8")), p, f.name)
        except Exception:
            pass
    # 2) flat files
    for f in sorted(d.rglob("*")):
        if f.is_file() and f.suffix.lower() in (".csv", ".txt", ".log"):
            if f.stat().st_size > 3_000_000:
                continue
            scan_text(f, p)
    # 3) directory name hints
    if GAIN_NAME_RE.search(d.name):
        p.sources.append("dirname:gain-token")
    return p


def dir_gain_db(name: str):
    """Explicit hardware gain encoded in a directory name, in dB.

    ``_g15``/``_g5`` -> that many dB; ``_g0`` -> 0 dB (a *clean* control arm,
    must NOT be classed as contaminated); no token -> None.
    """
    m = DIR_GAIN_NUM_RE.search(name)
    if m:
        return float(m.group(1))
    return None


def classify(d: Path, p: Probe):
    hw_off = []
    for key, val in p.hw:
        default = HW_DEFAULT[key]
        if val is None or (isinstance(val, float) and isinstance(default, float)
                           and math.isclose(val, default, abs_tol=1e-9)):
            continue
        if isinstance(val, (int, float)) and abs(float(val)) > 1e-9:
            hw_off.append((key, val))
    # a ``_g5``/``_g15`` token in the *directory name* is authoritative: it is
    # the knob the operator actually turned.  ``_g0`` means 0 dB => clean.
    dg = dir_gain_db(d.name)
    if dg is not None:
        if abs(dg) > 1e-9:
            hw_off.append(("dir_name_gain_db", dg))
        else:
            hw_off = []            # explicit 0 dB arm: never "contaminated"
            p.sources = [s for s in p.sources if "gain-token" not in s]
    # A ``gain15`` string inside a *file* is usually just a comparator column of
    # a larger sweep whose main arm is still G_hw = 0 dB.  Only the directory
    # name or an actual non-zero net-gain setting is authoritative.
    gain_token = any(s.startswith("dirname:gain-token") for s in p.sources) \
        and dg is None
    cell_only = any(s.startswith("file:") and "gain-token" in s
                    for s in p.sources) and not hw_off and not gain_token

    areas = sorted(a for a in p.areas if a > 0)
    rcss = sorted(r for r in p.rcss if r > 0)
    big_area = any(a > AREA_MAX for a in areas)
    big_rcs = any(r > RCS_MAX for r in rcss)

    reasons = []
    if hw_off:
        reasons.append("hw:" + ",".join(f"{k}={v:g}" for k, v in hw_off))
    if gain_token:
        reasons.append("name/token:gain")
    if big_area:
        reasons.append("area:" + ",".join(f"{a:g}" for a in areas if a > AREA_MAX))
    if big_rcs:
        reasons.append("rcs:" + ",".join(f"{r:g}" for r in rcss if r > RCS_MAX))

    if cell_only:
        reasons.append("note:has gain15 comparator cell (main arm G_hw=0)")

    if hw_off or gain_token:
        cat = "A-hardware-gain"
    elif big_area or big_rcs:
        cat = "B-out-of-band"
    elif areas or rcss:
        cat = "C-keep"
    else:
        cat = "U-unknown"
    return cat, "; ".join(reasons), areas, rcss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/_cleanup_scan.csv")
    args = ap.parse_args()

    entries = sorted([p for p in ROOT.glob("results*")])
    rows = []
    for d in entries:
        p = probe_dir(d)
        cat, why, areas, rcss = classify(d, p)
        tracked = False
        rows.append(dict(path=str(d.relative_to(ROOT)), category=cat,
                         reasons=why,
                         areas=",".join(f"{a:g}" for a in areas[:8]),
                         rcs=",".join(f"{r:g}" for r in rcss[:8]),
                         is_dir="yes" if d.is_dir() else "no"))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    c = Counter(r["category"] for r in rows)
    print(f"scanned {len(rows)} artifacts -> {out}")
    for k in sorted(c):
        print(f"  {k:18s} {c[k]:4d}")
    print()
    for cat in ("A-hardware-gain", "B-out-of-band"):
        print(f"===== {cat} =====")
        for r in rows:
            if r["category"] == cat:
                print(f"  {r['path']:62s} {r['reasons'][:70]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
