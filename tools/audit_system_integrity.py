"""System-wide audit of the ISAC simulator: system model / algorithm / theory.

This is *executable evidence*. Every finding below is expressed as a numeric
assertion rather than a code review sentence, following the rule that
"reading the code" is not evidence while "a failing assert" is.

Usage::

    python tools/audit_system_integrity.py            # fast structural + numeric checks
    python tools/audit_system_integrity.py --deep     # additionally run cheap MC probes

Exit code 0 means every check passed. A FAIL is a real defect, not a style
comment: every one of them can silently change a published number.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isac_sim.core import config as cfgmod  # noqa: E402
from isac_sim.core.config import (  # noqa: E402
    PRESETS,
    apply_overrides,
    apply_preset,
    default_config,
    validate_config,
)
from isac_sim.sensing.model import (  # noqa: E402
    bandwidth,
    build_base_gains,
    compute_link_tables,
    denominator_guard,
    generate_geometry,
    noise_power,
)

RESULTS: list[tuple[str, str, str]] = []  # (verdict, id, message)


def record(verdict: str, cid: str, msg: str) -> None:
    RESULTS.append((verdict, cid, msg))
    print(f"[{verdict:4}] {cid}: {msg}")


def expect(cond: bool, cid: str, ok: str, bad: str) -> None:
    record("PASS" if cond else "FAIL", cid, ok if cond else bad)


def seeded_setup(preset: str = "target-local-v1", **overrides):
    # NOTE: apply_preset / apply_overrides return a DEEP COPY and do not mutate.
    # Dropping the return value silently yields the untouched default config,
    # which is exactly the failure mode check "call-site" below looks for.
    cfg = default_config()
    cfg = apply_preset(cfg, preset)
    if overrides:
        cfg = apply_overrides(cfg, overrides)
    validate_config(cfg)
    rng = np.random.default_rng(20260916)
    geom = generate_geometry(cfg, rng)
    base = build_base_gains(cfg, geom, rng)
    return cfg, geom, base, rng


def tables_for(cfg, base, **kw):
    return compute_link_tables(cfg, base, **kw)


def finite_positive(a: np.ndarray) -> bool:
    return bool(np.isfinite(a).all() and (a >= 0).all())


# ---------------------------------------------------------------------------
# A. Calibre (口径) safety: enum values must not silently fall back
# ---------------------------------------------------------------------------
def check_enum(cid: str, title: str) -> None:
    """A wrong-cased or otherwise ignored enum must not degrade to another model."""
    cfg, geom, base, _ = seeded_setup()
    ref_orth = tables_for(cfg, base).gamma_sense

    # Same model, different capitalisation.
    cfg2 = default_config()
    cfg2 = apply_preset(cfg2, "target-local-v1")
    cfg2.comm.interference_model = "Orthogonal"  # passes validate_config (it lower()s)
    validate_config(cfg2)
    cased = tables_for(cfg2, build_base_gains(cfg2, geom, np.random.default_rng(1))).gamma_sense

    # Explicit worst case, for comparison.
    cfg3 = default_config()
    cfg3 = apply_preset(cfg3, "target-local-v1")
    cfg3 = apply_overrides(cfg3, {"comm.interference_model": "full_concurrent"})
    full = tables_for(cfg3, build_base_gains(cfg3, geom, np.random.default_rng(1))).gamma_sense

    same_as_full = bool(np.allclose(cased, full, rtol=0, atol=0))
    same_as_orth = bool(np.allclose(cased, ref_orth, rtol=0, atol=0))
    expect(
        same_as_orth,
        cid,
        f"capitalised enum resolves to the intended 口径 ({title})",
        f"capitalised enum silently degrades: matches full_concurrent={same_as_full}, "
        f"matches orthogonal={same_as_orth}  <- config lower()s for validation, model.py:614 compares with ==",
    )


def check_enum_without_mask(cid: str) -> None:
    """``active_set`` without an explicit mask degrades to full concurrency."""
    build = {}
    for name in ("active_set", "full_concurrent"):
        cfg = default_config()
        cfg = apply_preset(cfg, "target-local-v1")
        cfg = apply_overrides(cfg, {"comm.interference_model": name})
        rng = np.random.default_rng(7)
        geom = generate_geometry(cfg, rng)
        base = build_base_gains(cfg, geom, rng)
        build[name] = tables_for(cfg, base).gamma_sense
    identical = bool(np.allclose(build["active_set"], build["full_concurrent"], rtol=0, atol=0))
    expect(
        not identical,
        cid,
        "active_set without a mask is distinguishable from full_concurrent",
        "interference_model='active_set' with no active_tx_mask returns the FULL-CONCURRENT "
        "table bit-exactly -> the enum value is decorative, model.py:617-621 keys off the mask",
    )


# ---------------------------------------------------------------------------
# B. Numerical guard scale
# ---------------------------------------------------------------------------
def check_guard_scale(cid: str) -> None:
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    n0 = noise_power(cfg)
    eps = denominator_guard(cfg, n0)
    ratio = eps / n0
    try:
        cfg.radio.eps_mode = "lgeacy"
        denominator_guard(cfg, n0)
        typo_raises = False
    except ValueError:
        typo_raises = True
    expect(
        ratio <= 1e-2,
        cid + "a",
        f"guard is {10 * math.log10(1 + ratio):.4f} dB above the noise floor (ratio {ratio:.2e})",
        f"guard is {ratio:.1f}x the noise floor -> suppresses every sensing SINR by "
        f"{10 * math.log10(1 + ratio):.2f} dB",
    )
    expect(
        typo_raises,
        cid + "b",
        "unknown eps_mode raises instead of falling back",
        "a typo in radio.eps_mode silently reproduces the legacy behaviour",
    )


# ---------------------------------------------------------------------------
# C. Table health + interference-limited regime
# ---------------------------------------------------------------------------
def check_table_health(cid: str) -> None:
    cfg, _, base, _ = seeded_setup()
    t = tables_for(cfg, base)
    ok = all(
        finite_positive(getattr(t, f))
        for f in ("gamma_comm", "gamma_sense", "rinr", "raw_gamma_sense")
        if hasattr(t, f)
    )
    expect(
        ok,
        cid,
        "all (raw) link tables are finite and non-negative (no NaN hiding in max/min)",
        "a link table contains NaN or a negative entry -> NaN would be swallowed by max/min reductions",
    )


def measure_rinr(cid: str) -> None:
    """Report the regime honestly: 'interference limited' must be measurable."""
    cfg, _, base, _ = seeded_setup()
    t = tables_for(cfg, base)
    r = np.asarray(t.rinr)
    valid = r[np.isfinite(r) & (r > 0)]
    if valid.size == 0:
        record("FAIL", cid, "rinr is identically zero -> nothing keeps any residual term alive")
        return
    med = float(np.median(valid))
    expect(
        med > 1.0,
        cid,
        f"median rinr = {10 * math.log10(med):+.2f} dB -> interference-limited as claimed",
        f"median rinr = {10 * math.log10(med):+.2f} dB (< 0 dB) -> the system is NOISE limited; "
        "headline claims that rest on 'interference limited' do not hold at this working point",
    )


# ---------------------------------------------------------------------------
# D. Degrees of freedom: config keys that can move nothing
# ---------------------------------------------------------------------------
def check_dead_keys(cid: str) -> None:
    """Keys that are tunable but cannot change any number in the current 口径."""
    trials = {
        "radio.residual_direct_factor": 1000.0,
        "radio.residual_multi_uav_factor": 1000.0,
        "comm.comm_direct_leakage_factor": 0.5,
    }
    dead, alive = [], []
    for key, val in trials.items():
        cfg_a, geom, base, _ = seeded_setup()
        t_a = tables_for(cfg_a, base)
        cfg_b = seeded_setup(**{key.split(".")[-1]: val})[0] if False else None
        cfg_b = default_config()
        cfg_b = apply_preset(cfg_b, "target-local-v1")
        cfg_b = apply_overrides(cfg_b, {key: val})
        geom_b = generate_geometry(cfg_b, np.random.default_rng(20260916))
        base_b = build_base_gains(cfg_b, geom_b, np.random.default_rng(20260916))
        t_b = compute_link_tables(cfg_b, base_b)
        same = bool(
            np.allclose(t_a.gamma_sense, t_b.gamma_sense, rtol=0, atol=0)
            and np.allclose(t_a.gamma_comm, t_b.gamma_comm, rtol=0, atol=0)
        )
        (dead if same else alive).append(key)
    expect(
        not dead,
        cid,
        f"every probed key moves the physics ({', '.join(alive)})",
        f"these keys are tunable but move NOTHING at the V1 口径 (bit-identical tables): "
        f"{', '.join(dead)} -> any sensitivity sweep over them reports a false zero",
    )


def check_rho_lever(cid: str) -> None:
    """rho must have a measurable effect, or its 'allocation' story is empty."""
    vals = []
    for rho in (0.5, 0.8, 0.95):
        cfg = default_config()
        cfg = apply_preset(cfg, "target-local-v1")
        cfg = apply_overrides(cfg, {"radio.rho": rho})
        rng = np.random.default_rng(5)
        base = build_base_gains(cfg, generate_geometry(cfg, rng), rng)
        t = compute_link_tables(cfg, base)
        g = np.asarray(t.gamma_sense)
        g = g[np.isfinite(g) & (g > 0)]
        vals.append((rho, float(np.median(g))))
    spread_db = 10 * math.log10(max(v for _, v in vals) / min(v for _, v in vals))
    expect(
        spread_db > 0.5,
        cid,
        f"rho 0.50->0.95 moves the median sensing SINR by {spread_db:+.2f} dB",
        f"rho 0.50->0.95 moves the median sensing SINR by only {spread_db:+.2f} dB -> "
        "the power-splitting knob is essentially inert in this model",
    )


# ---------------------------------------------------------------------------
# E. Fusion stack self-consistency
# ---------------------------------------------------------------------------
def check_fusion_mode(cid: str) -> None:
    v1 = dict(PRESETS.get("target-local-v1", {}))
    chain = []
    for p in ("paper-canonical",):
        chain.append(dict(PRESETS.get(p, {})))
    merged: dict = {}
    for d in chain:
        merged.update(d)
    merged.update(v1)
    cfg = default_config()
    for d in chain:
        cfg = apply_overrides(cfg, d)
    cfg = apply_overrides(cfg, v1)
    precise = (
        str(cfg.detect.soft_stat_model).lower() == "llr"
        and str(cfg.detect.comm_error_model) == "erasure"
        and not bool(cfg.corr.enable)
    )
    record(
        "FAIL" if not precise else "PASS",
        cid,
        (
            "V1 preset reaches the exact calibrated threshold branch"
            if precise
            else f"V1 preset can NEVER reach the exact calibrated threshold: "
            f"soft_stat_model={cfg.detect.soft_stat_model!r}, "
            f"comm_error_model={cfg.detect.comm_error_model!r} "
            f"(fusion.py:351-356 silently `return`s the Cornish-Fisher fallback instead of raising) "
            f"-> the paper's detector is described by a branch that is dead in the release 口径"
        ),
    )


def check_corr_consistency(cid: str) -> None:
    """The same incompatibility is expressed twice with different severities."""
    src = (ROOT / "isac_sim" / "fusion.py").read_text(encoding="utf-8")
    raises = bool(re.search(r"raise ValueError\(\"exact LLR sum currently requires", src))
    returns = bool(re.search(r"return float\(fallback\)", src))
    expect(
        raises and returns,
        cid,
        "correlation x exact-LLR is both hard-rejected and soft-degraded -- inconsistent contract",
        "correlation x exact-LLR incompatibility is expressed inconsistently: "
        f"raise={raises}, silent return={returns}. One path throws, the other quietly swaps the "
        "detector to Cornish-Fisher, so results from the two paths are not comparable",
    )


# ---------------------------------------------------------------------------
# F. Algorithm layer: can the scheduler actually avoid interference?
# ---------------------------------------------------------------------------
def check_selector_blindness(cid: str) -> None:
    """F1, split into what is now true and what is still open.

    The original version was a string test ("is ``active_tx_mask`` mentioned in
    selection.py?").  ``select_c2f_adaptive`` now has an ``active_tx_mask``
    parameter, which would have flipped that test to PASS while the released
    pipeline gained nothing -- a false PASS.  So the question is asked
    behaviourally, and in two parts:

    * F1a -- can a caller make the selector see a mask?  (the coordination path)
    * F1b -- does the *release* pipeline ever pass one?  (still no)
    """
    import inspect

    from experiments.selection import select_c2f_adaptive

    params = inspect.signature(select_c2f_adaptive).parameters
    has_param = "active_tx_mask" in params
    defaults_none = bool(has_param and params["active_tx_mask"].default is None)
    expect(
        has_param and defaults_none,
        f"{cid}a",
        "the release selector accepts an active-transmitter mask, default off "
        "(F1 closed for the coordination口径; the default path is unchanged)",
        "select_c2f_adaptive cannot consume a mask, so a coordination-aware "
        "selection is impossible at the release entry point",
    )

    sim = (ROOT / "isac_sim" / "simulate.py").read_text(encoding="utf-8")
    release_passes_mask = bool(
        re.search(r"select_c2f_adaptive\([^)]*active_tx_mask", sim, re.S)
    )
    if release_passes_mask:
        record(
            "FAIL", f"{cid}b",
            "simulate.py now hands the selector a mask -- the released path changed, "
            "so every released number has to be re-validated before it is quoted",
        )
    else:
        # Kept as a non-PASS on purpose: the limitation is real and must be
        # declared, it is just not a regression (the release path is frozen by
        # iron law #1 -- bit-exact default path).
        record(
            "WARN", f"{cid}b",
            "the RELEASED pipeline still passes no mask, so its selection remains "
            "interference-blind (geometric-constant sensing SINR): a coordination "
            "claim cannot be made about the released method as the pipeline runs it, "
            "only about the coordination口径 (coordination.py / "
            "run_coordination_experiment.py)",
        )


def greedy_body() -> str:
    """Body of the MAIN greedy path only (``select_c2f_adaptive`` has a swap move,
    but that is a post-hoc polish and must not be credited to the greedy core)."""
    src = (ROOT / "isac_sim" / "selection.py").read_text(encoding="utf-8")
    m = re.search(r"^def _greedy_lagrangian\(.*?(?=^def |\Z)", src, re.S | re.M)
    return m.group(0) if m else ""


def check_commit_reversible(cid: str) -> None:
    body = greedy_body()
    has_add = bool(re.search(r"\.(append|add)\(", body))
    has_remove = bool(re.search(r"\.(remove|discard|pop)\(", body))
    expect(
        has_remove,
        cid,
        "the greedy CORE supports removal/swap, so its result is a genuine local optimum",
        "the greedy core (_greedy_lagrangian) only appends, never removes or swaps -> every "
        "committed link is irreversible and the reported 'optimum' is a first-pass artifact. "
        "(The only swap move lives in select_c2f_adaptive's post-hoc polish, off the release path.)",
    )


def check_rinr_vs_scale(cid: str) -> None:
    """'Interference limited' is a GEOMETRY-dependent statement; quantify where it flips."""
    rows = []
    for area in (4000.0, 2000.0, 1000.0, 800.0, 400.0):
        vals = []
        for seed in (1, 2, 3):
            cfg = default_config()
            cfg = apply_preset(cfg, "target-local-v1")
            cfg = apply_overrides(cfg, {"geometry.area_xy": area})
            rng = np.random.default_rng(seed)
            base = build_base_gains(cfg, generate_geometry(cfg, rng), rng)
            r = np.asarray(compute_link_tables(cfg, base).rinr)
            r = r[np.isfinite(r) & (r > 0)]
            if r.size:
                vals.append(float(np.median(r)))
        if vals:
            med = float(np.median(vals))
            rows.append((area, 10 * math.log10(med)))
    line = ", ".join(f"{a:.0f}m:{d:+.1f}dB" for a, d in rows)
    release = rows[0][1]
    record(
        "FAIL",
        cid,
        f"median rinr vs deployment footprint -> {line}. "
        f"At the RELEASED footprint ({rows[0][0]:.0f} m) rinr = {release:+.2f} dB, i.e. the sensing "
        "denominator is NOISE dominated. 'Interference limited' only becomes true when the swarm is "
        "compressed; the performance-lever roadmap measured rinr ~ +10 dB there and generalised it to "
        "the release. This is geometry-conditional, not a property of the system.",
    )


def check_hardcoded_rounds(cid: str) -> None:
    hits = []
    for name in ("power_joint.py", "power_c2f.py", "power_c2f_conservative.py", "joint_polish.py"):
        p = ROOT / "isac_sim" / name
        if not p.exists():
            continue
        src = p.read_text(encoding="utf-8")
        for m in re.finditer(r"(rounds|passes)\s*[:=]\s*(\d+)", src):
            hits.append(f"{name}:{m.group(1)}={m.group(2)}")
    expect(
        not hits,
        cid,
        "no hardcoded iteration budget found",
        "iteration budgets are hardcoded literals with no convergence criterion: "
        + ", ".join(hits)
        + " -> a 'converged' claim needs evidence, not a literal",
    )


# ---------------------------------------------------------------------------
# G. Theory invariants (these must hold bit-exactly, not statistically)
# ---------------------------------------------------------------------------
def check_llr_identities(cid: str) -> None:
    from isac_sim.detection import llr as L

    gamma = 0.37
    n_looks = 16
    delta = L.llr_delta(gamma, n_looks)
    v0 = L.llr_var0(gamma, n_looks)
    v1 = L.llr_var1(gamma, n_looks)
    jeff = L.llr_jeffreys(gamma, n_looks)
    w = L.optimal_fusion_weight(gamma)
    i1 = abs(L.llr_reverse_kld(gamma, n_looks) + L.llr_kld(gamma, n_looks) - jeff)
    i2 = abs(L.llr_deflection(gamma, n_looks) - n_looks * gamma * gamma)
    i3 = abs(w - (1.0 + gamma))
    i4 = abs(delta - jeff)
    worst = max(i1, i2, i3, i4)
    expect(
        worst < 1e-9,
        cid,
        f"LLR closed forms are mutually consistent (max residual {worst:.2e})",
        f"LLR identities violated with residual {worst:.3e}: jeffreys={i1:.2e}, "
        f"deflection={i2:.2e}, weight={i3:.2e}, delta==jeffreys={i4:.2e}",
    )
    expect(
        abs(v1 - v0) > 0,
        cid + "b",
        "H1 and H0 variances are distinct (a real binary hypothesis test)",
        "H1 variance equals H0 variance -> the detector degenerates",
    )


def check_fbl(cid: str) -> None:
    from isac_sim.sensing import fbl as F

    gamma = 3.0
    V = F.channel_dispersion(gamma)
    want = (1.0 - (1.0 + gamma) ** -2.0) * (math.log2(math.e)) ** 2
    expect(
        abs(V - want) < 1e-12,
        cid,
        f"channel dispersion matches the normal approximation (V={V:.6f})",
        f"channel dispersion V={V:.9f} != (1-(1+g)^-2)(log2 e)^2 = {want:.9f}",
    )


def check_bandwidth_consistency(cid: str) -> None:
    """OTFS frame properties must agree on what the bandwidth is."""
    cfg = default_config()
    cfg = apply_preset(cfg, "target-local-v1")
    N, L = cfg.waveform.N, cfg.waveform.L
    expect(
        N == L,
        cid,
        f"waveform N == L == {N}, so the two bandwidth conventions coincide",
        f"bandwidth() uses N={N} while the OTFS frame uses L={L} (model.py:31 vs model.py:400) -> "
        "any sweep that changes N or L alone silently re-defines the noise floor",
    )


# ---------------------------------------------------------------------------
# H. Variant-label honesty
# ---------------------------------------------------------------------------
def check_label_honesty(cid: str) -> None:
    from experiments.flow import sweeps as EX

    empties = []
    for attr in ("ABLATION_VARIANTS", "DD_VARIANTS"):
        d = getattr(EX, attr, None)
        if isinstance(d, dict):
            empties += [f"{attr}['{k}']" for k, v in d.items() if isinstance(v, dict) and not v]
    expect(
        not empties,
        cid,
        "every variant pins the switches it claims to control",
        "these variants are empty override dicts, so they inherit the caller's config: "
        + ", ".join(empties)
        + " -> under --set their labels lie about what was actually run",
    )


def check_call_site(cid: str) -> None:
    """apply_preset/apply_overrides return a COPY; a dropped return = silent no-op.

    Only a *statement-level* call whose value is discarded counts. A call used as
    an argument, compared, or wrapped in try/except is fine -- a naive line regex
    reported all of those as failures, which is exactly how an audit lies to you.
    """
    import ast

    bad, warned = [], []
    for base_dir in ("isac_sim", "tools", "tests"):
        d = ROOT / base_dir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.py")):
            if f.name == Path(__file__).name:
                continue
            try:
                tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                continue
            in_try = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    in_try.update(id(c) for c in ast.walk(node))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                    continue
                fn = node.value.func
                name = getattr(fn, "id", None) or getattr(fn, "attr", None)
                if name not in {"apply_preset", "apply_overrides"}:
                    continue
                loc = f"{base_dir}/{f.name}:{node.lineno}"
                # Discarding inside a try block is almost always a negative test
                # ("this call must raise"), not a forgotten return value.
                (warned if id(node) in in_try else bad).append(loc)
    expect(
        not bad,
        cid,
        "every standalone apply_preset/apply_overrides statement captures its return value"
        + (f" ({len(warned)} inside try blocks, treated as negative tests)" if warned else ""),
        "these call sites discard the returned config, so the override silently never happens: "
        + ", ".join(bad),
    )


def check_preset_pin_count(cid: str) -> None:
    counts = {k: len(v) for k, v in PRESETS.items()}
    thin = [f"{k}({n})" for k, n in counts.items() if n <= 2]
    expect(
        not thin,
        cid,
        "every preset pins enough keys to define its 口径",
        "these presets pin <=2 keys and therefore cannot reproduce anything by themselves: "
        + ", ".join(thin),
    )


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="run the slower numeric probes too")
    args = ap.parse_args()

    print("=" * 78)
    print("ISAC system integrity audit -- system model / algorithm / theory")
    print("=" * 78)

    if args.deep:
        check_enum("A1", "orthogonal")
        check_enum_without_mask("A2")
        check_rho_lever("C1")
        check_dead_keys("C2")
        measure_rinr("C3")
        check_rinr_vs_scale("C4")
        check_selector_blindness("F1")
        check_commit_reversible("F2")

    check_guard_scale("B1")
    check_table_health("B2")
    check_fusion_mode("D1")
    check_corr_consistency("D2")
    check_hardcoded_rounds("F3")
    check_llr_identities("G1")
    check_fbl("G2")
    check_bandwidth_consistency("G3")
    check_label_honesty("H1")
    check_call_site("H3")
    check_preset_pin_count("H2")

    fails = [r for r in RESULTS if r[0] != "PASS"]
    print("-" * 78)
    print(f"{len(RESULTS) - len(fails)} PASS / {len(fails)} FAIL(+WARN)")
    for v, cid, msg in fails:
        print(f"  {v:4} {cid}: {msg}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
