#!/usr/bin/env python3
"""Run deterministic theorem and falsification gates for V1.6."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from isac_sim.active_information import (  # noqa: E402
    configured_sensing_modes,
    price_active_information_bundle,
)
from isac_sim.config import Config, apply_overrides, apply_preset  # noqa: E402
from isac_sim.model import (  # noqa: E402
    build_base_gains,
    compute_link_tables,
    generate_geometry,
)
from isac_sim.scientific_gates import (  # noqa: E402
    PhysicalHeadroomDashboard,
    aggregate_partitioned_llrs,
    allocate_reliability_protection,
    aggregation_reliability_threshold,
    coherent_power_oracle,
    common_latent_gaussian_llrs,
    erasure_received_kl,
    evidence_delivery_variance,
    gaussian_phase_coherence,
    swerling_information,
    symmetric_complementarity_oracle,
)


def implementation_digest() -> str:
    digest = hashlib.sha256()
    paths = [
        ROOT / "isac_sim" / "scientific_gates.py",
        ROOT / "isac_sim" / "active_information.py",
        Path(__file__).resolve(),
    ]
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def run(seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)

    # G1: positive identity plus a mandatory negative shared-latent test.
    llrs = rng.normal(size=(2048, 5))
    aggregation = aggregate_partitioned_llrs(llrs, ((0, 3), (1,), (2, 4)))
    latent = common_latent_gaussian_llrs(rng.normal(size=(2048, 4)), 0.7)
    g1_pass = aggregation.max_abs_error < 1e-12 and latent.max_abs_gap > 1e-6

    # G2: exact expectation and explicit all-or-nothing variance penalty.
    local_information = np.asarray([0.2, 0.7, 1.6])
    chi_grid = np.linspace(0.0, 1.0, 11)
    erasure_error = max(
        abs(erasure_received_kl(float(np.sum(local_information)), chi)
            - chi * float(np.sum(local_information)))
        for chi in chi_grid
    )
    reliability_threshold = aggregation_reliability_threshold(
        local_information, (0.95, 0.8, 0.6)
    )
    separate_variance = evidence_delivery_variance(
        local_information, reliability_threshold, aggregated=False
    )
    aggregate_variance = evidence_delivery_variance(
        local_information, reliability_threshold, aggregated=True
    )
    protection = allocate_reliability_protection(
        local_information, (0.6, 0.6, 0.6), (1.0, 1.0, 1.0), 0.4,
        minimum_success=(0.6, 0.6, 0.7),
    )
    g2_pass = erasure_error < 1e-14 and aggregate_variance >= separate_variance

    # G3: must-fail identical views, two-sided complementarity, and coherent
    # limiting cases under one fixed total-power budget.
    alpha = 0.04
    single_information = float(swerling_information(alpha, 1.0, 16))
    identical_split_information = 2.0 * float(
        swerling_information(alpha, 0.5, 16)
    )
    complementary = symmetric_complementarity_oracle(2.0, 1.0, 1e-4, 16)
    noncomplementary = symmetric_complementarity_oracle(1.5, 1.0, 1e-4, 16)
    channel = np.asarray([1.0 + 0.0j, 0.5 + 0.0j, 0.25 + 0.0j])
    coherent_perfect = coherent_power_oracle(
        channel, 1.0, np.ones((3, 3), dtype=complex)
    )
    coherent_random = coherent_power_oracle(
        channel, 1.0, np.eye(3, dtype=complex)
    )
    coherent_partial = coherent_power_oracle(
        channel, 1.0, gaussian_phase_coherence(3, np.deg2rad(30.0))
    )
    g3_pass = all((
        identical_split_information < single_information,
        complementary.choose_cooperation,
        not noncomplementary.choose_cooperation,
        coherent_perfect.gain > coherent_partial.gain > 1.0,
        abs(coherent_random.gain - 1.0) < 1e-12,
    ))

    # G4 infrastructure: run complete-pool and restricted-pool exact pricing
    # on a tractable physical instance.  This validates the decomposition but
    # does not replace a large-system fixed-PFA detection oracle.
    cfg = apply_preset(Config(), "small-uav-compact-800m")
    cfg = apply_overrides(cfg, {
        "scale.M": 3,
        "scale.Q": 1,
        "prior.belief_mode": False,
        "dd.use_otfs_bin_validity": False,
        "detect.comm_error_model": "erasure",
        "detect.soft_stat_model": "llr",
        "active_sensing.enable": True,
        "active_sensing.energy_budget_per_target": 64.0,
        "active_sensing.max_candidates_per_pair": 2,
        "active_sensing.complete_pool_max_links": 12,
        "selector.max_links_per_target": 2,
    })
    geometry_rng = np.random.default_rng([seed, 16, 4])
    geometry = generate_geometry(cfg, geometry_rng)
    base = build_base_gains(cfg, geometry, geometry_rng, rcs_view="mean")
    coarse = compute_link_tables(cfg, base)
    refined = compute_link_tables(cfg, base, dd_gain=base.eta_fine)
    modes = configured_sensing_modes(cfg)
    full = price_active_information_bundle(
        cfg, base, coarse, refined, 0, 0, modes=modes,
        max_observations=2, candidate_strategy="full",
    )
    restricted = price_active_information_bundle(
        cfg, base, coarse, refined, 0, 0, modes=modes,
        max_observations=2, candidate_strategy="scenario_union",
    )
    single = price_active_information_bundle(
        cfg, base, coarse, refined, 0, 0, modes=modes,
        max_observations=1, candidate_strategy="full",
    )
    dashboard = PhysicalHeadroomDashboard(
        metric="kl",
        best_single=single.robust_information,
        noncoherent_full_oracle=full.robust_information,
        coherent_oracle=full.robust_information,
        restricted_oracle=restricted.robust_information,
        proposed=restricted.robust_information,
    )
    g4_pass = all((
        full.exact,
        restricted.exact,
        full.certificate_scope == "complete_pool",
        abs(dashboard.decomposition_error) < 1e-12,
        dashboard.candidate_loss >= -1e-12,
    ))

    gates = {
        "G1_sufficient_statistic": {
            "status": "pass" if g1_pass else "fail",
            "samplewise_max_abs_error": aggregation.max_abs_error,
            "common_latent_negative_test_max_abs_gap": latent.max_abs_gap,
            "claim": "exact only under an unconditional product likelihood and a disjoint exhaustive partition",
        },
        "G2_transport_information": {
            "status": "pass" if g2_pass else "fail",
            "erasure_kl_linearity_max_abs_error": erasure_error,
            "aggregate_reliability_threshold": reliability_threshold,
            "separate_delivery_variance": separate_variance,
            "aggregate_delivery_variance": aggregate_variance,
            "importance_aware_protection": asdict(protection),
            "claim": "KL expectation, not fixed-PFA ROC dominance",
        },
        "G3_physical_headroom": {
            "status": "theory_pass" if g3_pass else "fail",
            "scientific_evidence": "15-UAV/10-target full noncoherent physical oracle pending",
            "identical_split_to_single_information_ratio": (
                identical_split_information / single_information
            ),
            "strong_complementarity": asdict(complementary),
            "weak_complementarity": asdict(noncomplementary),
            "coherent_gain_perfect": coherent_perfect.gain,
            "coherent_gain_30deg": coherent_partial.gain,
            "coherent_gain_uniform_phase": coherent_random.gain,
        },
        "G4_algorithm_recovery": {
            "status": "infrastructure_pass" if g4_pass else "fail",
            "scientific_evidence": "pending large-system fixed-PFA detection oracle",
            "full_certificate_scope": full.certificate_scope,
            "full_exact": full.exact,
            "restricted_exact": restricted.exact,
            "best_single_kl": dashboard.best_single,
            "full_noncoherent_kl": dashboard.noncoherent_full_oracle,
            "restricted_kl": dashboard.restricted_oracle,
            "candidate_loss_kl": dashboard.candidate_loss,
            "algorithm_loss_kl": dashboard.algorithm_loss,
            "decomposition_error": dashboard.decomposition_error,
        },
        "G5_generalization": {
            "status": "pending_frozen_holdout",
            "development_geometry_count": 3,
            "required_protocol": "pre-specified independent geometry bank with cluster bootstrap and q05/CVaR endpoints",
            "minimum_starting_geometry_count": 30,
            "note": "sample size must ultimately be justified by cluster effect variance and the minimum relevant effect",
        },
    }
    stable = all(gates[name]["status"] == "pass" for name in (
        "G1_sufficient_statistic", "G2_transport_information",
    )) and all((
        gates["G3_physical_headroom"]["status"] == "theory_pass",
        gates["G4_algorithm_recovery"]["status"] == "infrastructure_pass",
    ))
    return {
        "artifact": "scientific_gates_v16",
        "version": "1.6.0",
        "implementation_digest": implementation_digest(),
        "seed": seed,
        "release_class": "mechanism_stable_generalization_pending" if stable else "blocked",
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument(
        "--out", type=Path, default=Path("results_scientific_gates_v16")
    )
    args = parser.parse_args()
    result = run(args.seed)
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    run_id = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    run_dir = args.out / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    output = run_dir / "scientific_gates_v16.json"
    rendered = json.dumps({**result, "run_id": run_id}, indent=2, sort_keys=True)
    if output.exists() and output.read_text(encoding="utf-8") != rendered:
        raise RuntimeError(f"refusing to overwrite mismatched artifact: {output}")
    output.write_text(rendered, encoding="utf-8")
    print(f"run directory: {run_dir}")
    print(rendered)


if __name__ == "__main__":
    main()
