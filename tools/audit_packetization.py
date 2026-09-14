#!/usr/bin/env python3
"""Audit per-UAV reporting load and same-target packet aggregation potential."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isac_sim.config import Config, apply_preset
from isac_sim.packetization import packetization_audit, summarize_packetization
from isac_sim.simulate import run_one_trial


def _trial(job: tuple[Config, int, str]) -> dict[str, float]:
    cfg, trial_index, method = job
    result = run_one_trial(cfg, trial_index, methods=[method])[method]
    audit = packetization_audit(
        cfg, result.selected_links, result.reporting_plan  # type: ignore[arg-type]
    )
    audit["detected_fraction"] = result.detected / max(result.total_targets, 1)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mc", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--method", default="proposed_c2f_adaptive_pd")
    parser.add_argument(
        "--fusion-rule",
        choices=["max_in_rate", "max_min_rate", "nearest_target", "nearest_centroid"],
        default="max_in_rate",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = apply_preset(Config(), "paper-canonical")
    cfg.run.num_mc = args.mc
    cfg.fusion.rule = args.fusion_rule
    jobs = ((cfg, t, args.method) for t in range(args.mc))
    if args.workers == 1:
        rows = [_trial(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(_trial, jobs, chunksize=1))

    result = {
        "method": args.method,
        "fusion_rule": args.fusion_rule,
        "mc": args.mc,
        "packet_capacity": cfg.comm.K_candidates,
        "soft_item_bits": cfg.comm.b_d,
        "full_packet_bits": cfg.comm.K_candidates * cfg.comm.b_d,
        "summary": summarize_packetization(rows),
    }
    payload = json.dumps(result, indent=2, sort_keys=True)
    print(payload)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
