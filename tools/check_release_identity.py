"""Release-identity gate for the frozen V1 theory/model/algorithm release.

The conference release claims three things at once:

* a **theory** (the equations in ``SystemModel.tex`` / ``ProposedMethod.tex``),
* a **model** (the physical channels, interference bookkeeping and detector
  statistics in ``isac_sim``), and
* an **algorithm** (the target-local placement plus the adaptive C2F selector).

Any of the three can drift silently -- a default can be edited, a method can be
renamed, an RNG stream can be reordered -- while every test still passes.  This
gate pins the *structure* of all three layers into a small manifest and fails
loudly when the working tree no longer matches it.

Deliberately separated:

* ``frozen``       -- model/detector/selection semantics.  Changing one of these
                      changes what the paper's equations describe.  A mismatch
                      is a red flag, never a routine edit.
* ``calibratable`` -- scenario inputs and the radar hardware-budget bridge
                      (deployment size, UAV/target counts, target RCS, antenna
                      gain).  These are *expected* to move while the radar link
                      budget is still being calibrated, so a mismatch is
                      reported and does not fail the gate.

Usage::

    python tools/check_release_identity.py --freeze     # record the manifest
    python tools/check_release_identity.py --check      # verify the working tree

Exit code is 0 only when every frozen item matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import isac_sim  # noqa: E402
from isac_sim.config import (  # noqa: E402
    HEADLINE_RELEASE_PRESET,
    PRESETS,
    Config,
    apply_preset,
)
from isac_sim.selection import (  # noqa: E402
    DEFAULT_METHODS,
    EXPERIMENTAL_METHODS,
    METHOD_RNG_OFFSETS,
)

MANIFEST_PATH = ROOT / "release" / "V1_STABLE_MANIFEST.json"
ARCHIVED_MAIN_CONFIG = ROOT / "results_target_local_v1" / "main" / "config.json"
BASELINE_CSV = ROOT / ".workbuddy" / "baseline" / "main" / "main.csv"

# --------------------------------------------------------------------------
# Frozen semantics: physics, interference bookkeeping, detector, selection
# --------------------------------------------------------------------------
FROZEN_PATHS: tuple[str, ...] = (
    # -- waveform / DD grid ------------------------------------------------
    "waveform.N",
    "waveform.L",
    "waveform.delta_f",
    "waveform.fc",
    "waveform.T",
    "waveform.c",
    # -- radio and sensing power split -------------------------------------
    "radio.P_default",
    "radio.rho",
    "radio.rho_by_uav",
    "radio.isac_power_model",
    "radio.noise_psd_dbm_hz",
    "radio.noise_figure_db",
    "radio.residual_self_factor",
    "radio.residual_direct_factor",
    "radio.residual_multi_uav_factor",
    "radio.rinr_sigma_factor",
    "radio.eps_mode",
    "radio.eps_rel_db",
    "radio.radar_tx_gain_dbi",
    "radio.radar_rx_gain_dbi",
    "radio.radar_system_loss_db",
    "radio.radar_net_gain_db",
    # -- MAC, reliability and the interference timing assumption -----------
    "comm.interference_model",
    "comm.mac_model",
    "comm.reliability_model",
    "comm.latency_model",
    "comm.enforce_chi_min",
    "comm.chi_min",
    "comm.R_min",
    "comm.b_d",
    "comm.n_block",
    "comm.comm_leakage_from_sensing",
    "comm.comm_direct_leakage_factor",
    "comm.K_candidates",
    "interference.coupling",
    "interference.direct_cancellation_db",
    "interference.sense_gate_by_active_tx",
    # -- detector ----------------------------------------------------------
    "detect.soft_stat_model",
    "detect.n_looks",
    "detect.Pfa_target",
    "detect.D_min",
    "detect.pd_required",
    "detect.weak_pd_required",
    "detect.rcs_model",
    "detect.rcs_aspect_enable",
    "detect.path_loss_exp",
    "detect.shadow_std_db",
    "detect.rician_K_db",
    "detect.comm_error_model",
    "detect.enable_comm_error_pollution",
    "detect.soft_error_sigma_scale",
    "detect.soft_error_flip_scale",
    "detect.soft_error_bias_scale",
    "detect.h0_error_bias_scale",
    "detect.num_false_per_target",
    "detect.fused_calibration_samples",
    "detect.soft_sigma0",
    "detect.soft_sigma_floor",
    "detect.soft_mu_scale",
    "detect.sensing_processing_gain",
    "dd.enable_dd_fractional_penalty",
    "dd.dd_collision_alpha",
    "dd.use_otfs_bin_validity",
    "refine.enable",
    "refine.mode",
    "refine.window_kernel",
    "refine.half_width",
    "refine.shortlist_size",
    "refine.kappa_dd",
    "refine.eta_min",
    "corr.enable",
    # -- scheduling / belief ----------------------------------------------
    "prior.belief_mode",
    "prior.scheduler_rcs",
    "prior.belief_sigma_pos_m",
    "prior.belief_sigma_vel_mps",
    "prior.search_gate_sigma",
    "prior.dt_s",
    # -- selection algorithm ----------------------------------------------
    "selector.score_mode",
    "selector.lambda_c",
    "selector.max_links_per_target",
    "selector.max_total_links",
    "selector.candidate_topk_per_target",
    "selector.min_marginal_D",
    "selector.stop_at_D_min",
    "selector.use_delay_price",
    "selector.use_comm_error_calibration",
    "selector.use_softmin_alpha",
    "selector.softmin_tau",
    "selector.use_target_priority",
    "selector.mu_deficit",
    "selector.alpha_floor",
    "selector.alpha_cap",
    # -- fusion placement --------------------------------------------------
    "fusion.mode",
    "fusion.rule",
)

# --------------------------------------------------------------------------
# Calibratable: scenario inputs and the provisional hardware bridge
# --------------------------------------------------------------------------
CALIBRATABLE_PATHS: tuple[str, ...] = (
    "scale.M",
    "scale.Q",
    "geometry.area_xy",
    "geometry.comm_range",
    "geometry.h_uav_min",
    "geometry.h_uav_max",
    "geometry.h_target_min",
    "geometry.h_target_max",
    "geometry.uav_speed_min",
    "geometry.uav_speed_max",
    "geometry.target_speed_min",
    "geometry.target_speed_max",
    "detect.target_rcs",
    "waveform_impairments.enable",
    "run.num_mc",
    "run.seed",
)

# Keys whose archived value is expected to differ for a *documented* reason.
# ``erasure`` at revision 69f3300 meant "Gaussian replacement with a = 9", not a
# true drop; the current code distinguishes the two.  See
# ``ReproducibilitySupplement.tex`` section "Historical error-model naming".
ARCHIVE_KNOWN_DIFFERENCES: dict[str, str] = {
    "detect.comm_error_model": (
        "historical naming debt: archived 'erasure' == Gaussian replacement a=9; "
        "current 'erasure' is a true zero drop"
    ),
}


def _resolve(cfg: object, path: str):
    node = cfg
    for part in path.split("."):
        node = getattr(node, part)
    return node


def _jsonable(value):
    if isinstance(value, tuple):
        return list(value)
    return value


def _digest(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.md5(path.read_bytes()).hexdigest()


def _resolved_config() -> Config:
    return apply_preset(Config(), HEADLINE_RELEASE_PRESET)


def _frozen_snapshot() -> dict[str, object]:
    cfg = _resolved_config()
    return {path: _jsonable(_resolve(cfg, path)) for path in FROZEN_PATHS}


def _calibratable_snapshot() -> dict[str, object]:
    cfg = _resolved_config()
    return {path: _jsonable(_resolve(cfg, path)) for path in CALIBRATABLE_PATHS}


def build_manifest() -> dict[str, object]:
    return {
        "release": "V1",
        "release_note": "frozen theory/model/algorithm structure for the ICC 2027 submission",
        "headline_preset": HEADLINE_RELEASE_PRESET,
        "package_version": isac_sim.__version__,
        "frozen": _frozen_snapshot(),
        "calibratable": _calibratable_snapshot(),
        "methods": {
            "default": list(DEFAULT_METHODS),
            "experimental": sorted(EXPERIMENTAL_METHODS),
        },
        "rng_offsets": dict(sorted(METHOD_RNG_OFFSETS.items())),
        "baseline_csv_md5": _digest(BASELINE_CSV),
        "preset_registry": sorted(PRESETS),
    }


def do_freeze() -> int:
    manifest = build_manifest()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"  headline preset : {manifest['headline_preset']}")
    print(f"  package version : {manifest['package_version']}")
    print(f"  frozen keys     : {len(manifest['frozen'])}")
    print(f"  default methods : {len(manifest['methods']['default'])}")
    print(f"  baseline md5    : {manifest['baseline_csv_md5']}")
    return 0


def do_check() -> int:
    if not MANIFEST_PATH.exists():
        print(f"FAIL: manifest missing at {MANIFEST_PATH.relative_to(ROOT)}; run --freeze first")
        return 1

    recorded = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    current = build_manifest()
    failures: list[str] = []
    notes: list[str] = []

    # 1. headline preset identity
    if current["headline_preset"] != recorded["headline_preset"]:
        failures.append(
            f"headline preset moved: {recorded['headline_preset']} -> {current['headline_preset']}"
        )

    # 2. frozen semantics, key by key, so the report names the offending key
    frozen_old = recorded["frozen"]
    frozen_new = current["frozen"]
    for key in sorted(set(frozen_old) | set(frozen_new)):
        old, new = frozen_old.get(key, "<absent>"), frozen_new.get(key, "<absent>")
        if old != new:
            failures.append(f"frozen[{key}]: {old!r} -> {new!r}")

    # 3. method roster and RNG streams
    if current["methods"]["default"] != recorded["methods"]["default"]:
        added = set(current["methods"]["default"]) - set(recorded["methods"]["default"])
        removed = set(recorded["methods"]["default"]) - set(current["methods"]["default"])
        failures.append(
            f"default method roster changed: added={sorted(added)} removed={sorted(removed)}"
        )
    if current["methods"]["experimental"] != recorded["methods"]["experimental"]:
        failures.append("experimental method set changed")
    if current["rng_offsets"] != recorded["rng_offsets"]:
        for key in sorted(set(current["rng_offsets"]) | set(recorded["rng_offsets"])):
            old = recorded["rng_offsets"].get(key, "<absent>")
            new = current["rng_offsets"].get(key, "<absent>")
            if old != new:
                failures.append(f"rng_offset[{key}]: {old} -> {new}")

    # 4. bit-exact regression baseline
    if current["baseline_csv_md5"] != recorded["baseline_csv_md5"]:
        failures.append(
            "baseline CSV changed: "
            f"{recorded['baseline_csv_md5']} -> {current['baseline_csv_md5']}"
        )

    # 5. calibratable drift is reported, never fatal
    for key in sorted(set(recorded["calibratable"]) | set(current["calibratable"])):
        old = recorded["calibratable"].get(key, "<absent>")
        new = current["calibratable"].get(key, "<absent>")
        if old != new:
            notes.append(f"calibratable[{key}]: {old!r} -> {new!r}")

    # 6. cross-check against the archived main-result configuration.  Only the
    #    frozen layer is binding here; a calibratable key legitimately differs
    #    (e.g. run.num_mc is supplied by the run entry point, not by the preset).
    archive_checked = 0
    if ARCHIVED_MAIN_CONFIG.exists():
        archived = json.loads(ARCHIVED_MAIN_CONFIG.read_text(encoding="utf-8"))
        for path, live in {**frozen_new, **current["calibratable"]}.items():
            group, _, leaf = path.partition(".")
            if group not in archived or not isinstance(archived[group], dict):
                continue
            if leaf not in archived[group]:
                continue
            old = archived[group][leaf]
            archive_checked += 1
            if old == live:
                continue
            if path in ARCHIVE_KNOWN_DIFFERENCES:
                notes.append(f"archived-main[{path}]: {ARCHIVE_KNOWN_DIFFERENCES[path]}")
            elif path in CALIBRATABLE_PATHS:
                notes.append(
                    f"archived-main[{path}]: calibratable {old!r} -> {live!r} "
                    "(supplied by the run entry point)"
                )
            else:
                failures.append(f"archived-main[{path}]: archived={old!r} current={live!r}")
    else:
        notes.append("archived main config not found; cross-check skipped")

    print(f"release identity: {recorded['release']}")
    print(f"  manifest        : {MANIFEST_PATH.relative_to(ROOT)}")
    print(f"  headline preset : {current['headline_preset']}")
    print(f"  package version : {recorded['package_version']} -> {current['package_version']}")
    print(f"  frozen keys     : {len(current['frozen'])} compared, {len(failures)} violated")
    print(f"  methods         : {len(current['methods']['default'])} default / "
          f"{len(current['methods']['experimental'])} experimental")
    print(f"  archive cross   : {archive_checked} keys compared")
    print(f"  baseline md5    : {current['baseline_csv_md5']}")

    for note in notes:
        print(f"NOTE {note}")
    for failure in failures:
        print(f"FAIL {failure}")

    print("RESULT: " + ("CLEAN" if not failures else f"{len(failures)} VIOLATION(S)"))
    return 0 if not failures else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze", action="store_true", help="record the current release identity")
    mode.add_argument("--check", action="store_true", help="verify the working tree against it")
    args = parser.parse_args(argv)
    return do_freeze() if args.freeze else do_check()


if __name__ == "__main__":
    raise SystemExit(main())
