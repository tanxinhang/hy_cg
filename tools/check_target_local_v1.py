#!/usr/bin/env python3
"""Check a completed MC=1000 target-local V1 main gate."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.core.config import Config, apply_preset, iter_leaf_paths  # noqa: E402
from isac_sim.sensing.fbl import packet_bits  # noqa: E402


REQUIRED_METHODS = {
    "proposed_c2f_adaptive_pd",
    "exact_marginal_greedy",
    "sense_sinr",
}


def require(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "result_dir",
        nargs="?",
        type=Path,
        default=ROOT / "results_target_local_v1" / "main",
    )
    args = parser.parse_args()
    result_dir: Path = args.result_dir

    config_path = result_dir / "config.json"
    csv_path = result_dir / "main.csv"
    if not config_path.is_file() or not csv_path.is_file():
        parser.error(f"expected config.json and main.csv under {result_dir}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = {row["method"]: row for row in csv.DictReader(handle)}

    failures: list[str] = []
    require(config["fusion"] == {"mode": "explicit", "rule": "nearest_target"},
            "explicit nearest_target fusion is recorded", failures)
    require(int(config["run"]["num_mc"]) >= 1000,
            "at least 1000 Monte Carlo trials are recorded", failures)
    require(int(config["run"]["seed"]) == 2026,
            "frozen seed is 2026", failures)
    require(REQUIRED_METHODS.issubset(rows),
            "all frozen comparison methods are present", failures)

    # Reconstruct the resolved V1 contract and assert that it differs from the
    # paper preset only at the documented fusion rule. Runtime-only fields are
    # allowed to differ because the gate fixes MC/workers/verbosity explicitly.
    paper = dict(iter_leaf_paths(apply_preset(Config(), "paper-canonical")))
    v1 = dict(iter_leaf_paths(apply_preset(Config(), "target-local-v1")))
    changed = {key for key in paper if paper[key] != v1[key]}
    require(changed == {"fusion.rule"},
            "V1 preset differs from paper-canonical only at fusion.rule", failures)

    if REQUIRED_METHODS.issubset(rows):
        proposed = rows["proposed_c2f_adaptive_pd"]
        exact = rows["exact_marginal_greedy"]
        sinr = rows["sense_sinr"]

        def value(row: dict[str, str], field: str) -> float:
            return float(row[field])

        require(value(proposed, "P_D_ci95_low") >= 0.970,
                "proposed P_D 95% lower bound is at least 0.970", failures)
        require(value(proposed, "actual_worst_target_P_D") >= 0.960,
                "worst-target P_D is at least 0.960", failures)
        require(value(proposed, "selected_observations_mean") <= 13.0,
                "mean selected observations do not exceed 13", failures)
        require(value(proposed, "selected_links_mean") <= 1.5,
                "mean remote reports do not exceed 1.5", failures)
        require(value(proposed, "B_mean_bits") <= 1000.0,
                "mean payload does not exceed 1.0 kbit", failures)
        require(value(proposed, "T_mean_ms") <= 2.0,
                "mean serial delay does not exceed 2.0 ms", failures)
        require(value(proposed, "belief_capture_rate_mean") >= 0.990,
                "belief-guided DD capture rate is at least 0.990", failures)
        require(value(proposed, "selected_rate_satisfaction_ratio_mean") == 1.0,
                "every selected report satisfies the rate threshold", failures)
        require(value(proposed, "selected_chi_ge_min_ratio_mean") == 1.0,
                "every selected report satisfies the reliability threshold", failures)
        require(value(exact, "paired_reference_delta_ci95_low") > 0.0,
                "paired P_D gain over exact-marginal greedy has positive 95% lower bound",
                failures)
        sinr_lo = value(sinr, "paired_reference_delta_ci95_low")
        sinr_hi = value(sinr, "paired_reference_delta_ci95_high")
        require(sinr_lo <= 0.0 <= sinr_hi,
                "detector-PD and sensing-SINR P_D are statistically tied", failures)
        require(value(proposed, "B_mean_bits") < 0.25 * value(sinr, "B_mean_bits"),
                "proposed uses less than one quarter of sensing-SINR payload", failures)

        reports = value(proposed, "selected_links_mean")
        resolved_v1 = apply_preset(Config(), "target-local-v1")
        expected_bits = reports * packet_bits(resolved_v1)
        expected_delay_ms = (
            reports
            * float(config["comm"]["n_block"])
            / (float(config["waveform"]["N"]) * float(config["waveform"]["delta_f"]))
            * 1000.0
        )
        require(math.isclose(value(proposed, "B_mean_bits"), expected_bits,
                             rel_tol=0.0, abs_tol=1e-8),
                "payload equals report count times 640-bit report size", failures)
        require(math.isclose(value(proposed, "T_mean_ms"), expected_delay_ms,
                             rel_tol=0.0, abs_tol=1e-8),
                "serial delay equals report count times fixed-block duration", failures)

    if failures:
        print(f"\nTarget-local V1 gate FAILED ({len(failures)} checks).")
        return 1
    print("\nTarget-local V1 gate PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
