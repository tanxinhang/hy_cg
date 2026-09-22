"""Fit joint receiver statistics on training scenes and evaluate held-out ROC."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from isac_sim.cooperation.scientific_validation import (
    fit_joint_statistic,
    heldout_empirical_roc,
)


def _read(paths: list[str]) -> list[dict]:
    rows = []
    for source, path in enumerate(paths):
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["source"] = source
                rows.append(row)
    return rows


def evaluate(paths: list[str], train_fraction: float, split_seed: int,
             p_fa: float) -> dict:
    rows = _read(paths)
    scenes = sorted({(int(r["source"]), int(r["trial"])) for r in rows})
    if len(scenes) < 4:
        raise ValueError("at least four independent scenes are required")
    rng = np.random.default_rng(int(split_seed))
    order = rng.permutation(len(scenes))
    cut = min(max(int(round(len(scenes) * train_fraction)), 2), len(scenes) - 2)
    train_scenes = {scenes[i] for i in order[:cut]}
    test_scenes = set(scenes) - train_scenes

    groups = defaultdict(dict)
    for row in rows:
        scene = (int(row["source"]), int(row["trial"]))
        key = (
            str(row["arm"]), int(row["weak_target"]), scene,
            int(row["realisation"]), int(row["receiver"]),
        )
        groups[key] = (float(row["t_h0"]), float(row["t_h1"]))

    cases = []
    labels = sorted({(key[0], key[1]) for key in groups})
    for arm, target in labels:
        receivers = sorted({key[4] for key in groups
                            if key[0] == arm and key[1] == target})
        samples = defaultdict(dict)
        for key, values in groups.items():
            if key[0] == arm and key[1] == target:
                samples[(key[2], key[3])][key[4]] = values
        complete = [(key, value) for key, value in samples.items()
                    if all(receiver in value for receiver in receivers)]
        train = [value for (scene_real, value) in complete
                 if scene_real[0] in train_scenes]
        test = [value for (scene_real, value) in complete
                if scene_real[0] in test_scenes]
        if len(train) < 2 or not test:
            continue
        matrix = lambda values, index: np.asarray([
            [sample[receiver][index] for receiver in receivers]
            for sample in values
        ], dtype=float)
        train_h0, train_h1 = matrix(train, 0), matrix(train, 1)
        test_h0, test_h1 = matrix(test, 0), matrix(test, 1)
        weights, covariance = fit_joint_statistic(train_h0, train_h1)
        roc = heldout_empirical_roc(
            weights, train_h0, test_h0, test_h1, p_fa
        )
        std = np.sqrt(np.maximum(np.diag(covariance), 1e-30))
        correlation = covariance / np.outer(std, std)
        cases.append({
            "arm": arm,
            "target": target,
            "receivers": receivers,
            "weights": weights.tolist(),
            "training_joint_correlation": correlation.tolist(),
            "roc": roc,
        })
    return {
        "protocol": "joint_empirical_receiver_statistic_roc_v1",
        "input_files": [str(Path(path).resolve()) for path in paths],
        "split_seed": int(split_seed),
        "train_scenes": [list(scene) for scene in sorted(train_scenes)],
        "test_scenes": [list(scene) for scene in sorted(test_scenes)],
        "sets_disjoint": train_scenes.isdisjoint(test_scenes),
        "p_fa": float(p_fa),
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--train-fraction", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=20260922)
    parser.add_argument("--p-fa", type=float, default=0.05)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0.0 < args.train_fraction < 1.0:
        raise ValueError("train-fraction must lie strictly between zero and one")
    result = evaluate(args.input, args.train_fraction, args.split_seed, args.p_fa)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
