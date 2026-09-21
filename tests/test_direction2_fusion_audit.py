"""方向 2（协作融合）审计结论的断言化。

这里只钉**结构性**事实 —— 跑得快、不依赖 Monte-Carlo，因此不会随 seed 漂移：

* 哪些"优化方向"其实已被 ``paper-canonical`` preset 打开（改 preset 就会 FAIL）；
* 哪些旋钮是**死参数 / 硬编码**（接上了就会 FAIL，提醒回来更新本文件）；
* ``geometry_robust_base`` 的下界性质与置信度单调性。

端到端的量级结论（加链路 +0.05、P_FA 漂移）在
``studies/direction2/data/`` 里，不做成单测：它们需要 MC=40 才跑得动。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from isac_sim.core.config import Config, apply_overrides, apply_preset
from isac_sim.scenario.belief import geometry_robust_base
from isac_sim.sensing.model import build_base_gains, generate_geometry

REPO = Path(__file__).resolve().parents[1]


def _paper_canonical(**extra) -> Config:
    cfg = apply_preset(Config(), "paper-canonical")
    ov = {"scale.M": 3, "scale.Q": 2, "run.seed": 7, "run.verbose": False}
    ov.update(extra)
    return apply_overrides(cfg, ov)


def _base(cfg: Config):
    rng = np.random.default_rng([cfg.run.seed, 0])
    geom = generate_geometry(cfg, rng)
    return build_base_gains(cfg, geom, rng)


# --------------------------------------------------------------------------
# §2 已被 preset 打开的门：再拿它们当"优化项"就是空操作
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "path,expected",
    [
        ("detect.soft_stat_model", "llr"),
        ("detect.rcs_model", "mean"),
        ("selector.score_mode", "exact_utility"),
    ],
)
def test_paper_canonical_already_uses_the_good_value(path, expected):
    """These are the *released* settings, so 'turning them on' is a no-op.

    The direction-2 scan lost four arms to exactly this; if a preset change
    reverts one of them, the corresponding "optimisation arm" silently becomes
    meaningful again and every archived delta has to be re-read.
    """
    cfg = _paper_canonical()
    obj = cfg
    for part in path.split("."):
        obj = getattr(obj, part)
    assert obj == expected


def test_paper_canonical_disables_the_dmin_early_exit():
    assert _paper_canonical().selector.stop_at_D_min is False


# --------------------------------------------------------------------------
# §6 死参数 / 硬编码：一条能力"没有旋钮"这件事本身要被钉住
# --------------------------------------------------------------------------
def test_sensing_power_scale_is_never_supplied_by_any_caller():
    """Node power is a *declared* capability with no user -- a dead parameter.

    ``sensing_power_scale_by_uav`` threads through ``compute_link_tables`` into
    ``power_split``, but no caller outside the link-table package passes it, so
    "allocate per-node sensing power" cannot be turned on by editing config.
    If someone wires it up, this test fails and the note in the direction-2
    audit must be updated.
    """
    pattern = re.compile(r"compute_link_tables\(", re.M)
    hits = 0
    for path in list((REPO / "experiments").rglob("*.py")) + \
            list((REPO / "tools").rglob("*.py")) + \
            list((REPO / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in pattern.finditer(text):
            window = text[m.end():m.end() + 400]
            if "sensing_power_scale_by_uav" in window:
                hits += 1
    for path in (REPO / "isac_sim").rglob("*.py"):
        if "link_tables" in path.parts:
            continue  # the package that declares the parameter
        text = path.read_text(encoding="utf-8", errors="ignore")
        for m in pattern.finditer(text):
            window = text[m.end():m.end() + 400]
            if "sensing_power_scale_by_uav" in window:
                hits += 1
    assert hits == 0, "a caller now supplies sensing_power_scale_by_uav"


def test_fusion_weight_mode_is_keyed_by_method_name_not_config():
    """Fusion weights are a method -> policy table, not a config gate."""
    from experiments.methods.policy import fusion_weight_mode_for_method

    assert fusion_weight_mode_for_method("proposed_c2f") == "deflection"
    assert fusion_weight_mode_for_method("raw_sense_sinr") == "equal"
    assert fusion_weight_mode_for_method("joint_bundle_cg_exact_llr") == "exact_llr_sum"
    # No config key can override it: the function takes the method name only.
    import inspect

    assert list(inspect.signature(fusion_weight_mode_for_method).parameters) == ["method"]


def test_geometry_robust_base_is_only_wired_into_the_pd_robust_method():
    """The robust-geometry view exists but the main chain does not use it."""
    text = (REPO / "experiments" / "flow" / "simulate.py").read_text(
        encoding="utf-8", errors="ignore"
    )
    lines = text.splitlines()
    idx = [
        i for i, line in enumerate(lines)
        if "geometry_robust_base" in line and not line.strip().startswith("#")
    ]
    assert idx, "the robust geometry view is no longer wired anywhere"
    # the call must sit inside the pd_robust branch (its guard is a few lines up)
    guarded = any(
        "proposed_c2f_adaptive_pd_robust" in lines[j]
        for i in idx
        for j in range(max(0, i - 5), min(len(lines), i + 3))
    )
    assert guarded, "geometry_robust_base escaped the pd_robust-only guard"


# --------------------------------------------------------------------------
# geometry_robust_base 的数学性质（方向 2 唯一有实现基础的杠杆）
# --------------------------------------------------------------------------
def test_robust_geometry_is_a_lower_bound_on_the_gain():
    cfg = _paper_canonical()
    base = _base(cfg)
    robust = geometry_robust_base(cfg, base)
    g = np.asarray(base.target_gain, dtype=float)
    gr = np.asarray(robust.target_gain, dtype=float)
    assert np.all(gr <= g + 1e-12)
    # and it bites: the confidence-0.5 ball over a 150 m sigma is not tiny
    assert float(np.min(gr / np.maximum(g, 1e-300))) < 0.9


def test_robust_geometry_is_monotone_in_the_confidence_mass():
    """More confidence mass => larger ball => smaller (more conservative) gain."""
    cfg_lo = _paper_canonical(**{"prior.robust_position_confidence": 0.30})
    cfg_hi = _paper_canonical(**{"prior.robust_position_confidence": 0.90})
    base = _base(cfg_lo)
    lo = np.asarray(geometry_robust_base(cfg_lo, base).target_gain, dtype=float)
    hi = np.asarray(geometry_robust_base(cfg_hi, base).target_gain, dtype=float)
    assert np.all(hi <= lo + 1e-12)
    assert float(np.median(hi / np.maximum(lo, 1e-300))) < 0.999


def test_robust_geometry_is_the_identity_when_the_prior_is_exact():
    """No belief error => nothing to be robust against => bit-identical."""
    cfg = _paper_canonical(**{"prior.belief_sigma_pos_m": 0.0})
    base = _base(cfg)
    robust = geometry_robust_base(cfg, base)
    assert np.array_equal(
        np.asarray(robust.target_gain), np.asarray(base.target_gain)
    )


def test_robust_geometry_requires_belief_mode():
    cfg = _paper_canonical(**{"prior.belief_mode": False})
    base = _base(cfg)
    assert geometry_robust_base(cfg, base) is base
