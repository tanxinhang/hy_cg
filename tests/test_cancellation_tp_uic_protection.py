"""TP-UIC tangent basis and stage-2 protection-support contracts.

Split out of ``test_cancellation_tp_uic.py`` (2026-09-21 audit).  Function bodies
are moved verbatim; the builders live in ``_tpuic_common.py``.  Covered by
``tests/_test_inventory.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from isac_sim.receiver import cancellation as cx
from isac_sim.sensing.model import noise_power
from _tpuic_common import make_cfg, make_observation


# --------------------------------------------------------------------------
# Module M2: the protection is a statement about a neighbourhood
# --------------------------------------------------------------------------
def test_on_grid_target_is_fully_covered_by_its_tangent_basis():
    """``a(theta)`` lies in ``span{a, da/dk, da/dl}`` by construction."""
    cfg = make_cfg()
    a = cx.kernel_vector(cfg, 2.37, 1.44)
    J = cx.tangent_columns(cfg, 2.37, 1.44, order=1, step=0.05)
    U = cx.orthonormalise(J)
    assert U.rank == 3
    residual = a - U.project(a)
    assert np.linalg.norm(residual) / np.linalg.norm(a) < 1e-10


def test_protection_covers_a_neighbouring_offset_to_first_order():
    """A target a fraction of a bin away is still mostly protected.

    The residual grows quadratically, which is the error term the method note
    quotes: ``||M a_true|| = O(||dtheta||^2)``.
    """
    cfg = make_cfg()
    J = cx.tangent_columns(cfg, 2.37, 1.44, order=1, step=0.05)
    U = cx.orthonormalise(J)
    previous, ratios = None, []
    for delta in (0.005, 0.01, 0.02, 0.04):
        a = cx.kernel_vector(cfg, 2.37 + delta, 1.44)
        leak = float(np.linalg.norm(a - U.project(a)) / np.linalg.norm(a))
        ratios.append(leak)
        if previous is not None:
            assert leak > previous, "leakage must grow with the offset"
        previous = leak
    assert ratios[-1] < 0.10, "a 0.04-bin offset should stay well inside the protection"


# --------------------------------------------------------------------------
# Numerical traps that have already bitten once
# --------------------------------------------------------------------------
def test_orthonormalise_rank_ignores_column_scaling():
    """Rank must not depend on ``1/step``: the finite-difference trap.

    The DD tangent columns are difference directions whose norms differ from the
    centre column's, and that gap grows as ``step`` shrinks.  With the default
    ``step=0.05`` it is only ~3.6x, but a rank decision made *relative to the
    largest Gram eigenvalue* lets the gap decide the answer: on one real 420-
    column block the old rule returned 219 / 186 / 148 for tolerances
    1e-12 / 1e-9 / 1e-6, while the normalise-then-absolute-threshold rule returns
    a stable 169.  A rank that moves with the tolerance cannot be an input to a
    protection-cost calculation, so scale invariance is the invariant worth
    pinning -- which is why the scales below are deliberately extreme rather
    than whatever the default ``step`` happens to produce.
    """
    cfg = make_cfg()
    block = cx.tangent_columns(cfg, 2.37, 1.44, order=1, step=0.05)
    scaled = block @ np.diag([1.0, 1000.0, 0.001])
    assert cx.orthonormalise(block).rank == cx.orthonormalise(scaled).rank


def test_positive_sigma_refuses_a_clamped_variance():
    """``EPS = 1e-12`` is larger than the noise power: it must not be a divisor."""
    assert noise_power(make_cfg()) < 1e-12, (
        "if this ever fails the EPS guard would be harmless; while it holds, "
        "clamping sigma2 to EPS silently inflates every prediction by 26x"
    )
    cx._positive_sigma(3.83e-14)
    with pytest.raises(ValueError):
        cx._positive_sigma(0.0)


# --------------------------------------------------------------------------
# Stage 2: a declared support rule, not a test whose level is an assumption
# --------------------------------------------------------------------------
def test_joint_support_is_the_protection_set_by_default():
    """The default support is declared, and it is not the gate's output.

    ``threshold = 0`` is the extreme case: every target clears it, so the gate's
    output is "all targets".  The default policy must still model only the
    protected ones -- the test would be vacuous with a positive threshold, which
    is exactly why it uses the degenerate one.
    """
    cfg = make_cfg(**{"cancellation.max_protected_targets": 1})
    obs = make_observation(cfg, belief_error=True)
    shielded = cx.protected_target_ids(cfg, obs.targets_belief or obs.targets)
    default = cx.cancellation_arms(cfg, obs, threshold=0.0)["tp_uic_full"]
    ablation = cx.cancellation_arms(
        cfg, obs, threshold=0.0, candidate_policy="statistic"
    )["tp_uic_full"]
    assert shielded, "the protection budget must select something"
    assert set(default.supported_targets) == set(shielded)
    assert set(ablation.supported_targets) == set(ablation.gate_targets)
    assert len(default.candidates) < len(ablation.candidates)
    with pytest.raises(ValueError):
        cx.cancellation_arms(cfg, obs, candidate_policy="whatever")


def test_protected_target_ids_agrees_with_the_protection_basis():
    """The extracted helper still describes the basis that is actually built."""
    cfg = make_cfg(**{"cancellation.max_protected_targets": 2})
    obs = make_observation(cfg, belief_error=True)
    sources = obs.targets_belief or obs.targets
    ids = cx.protected_target_ids(cfg, sources)
    subset = [s for s in sources if int(s.target) in ids]
    built = cx._protection_basis(cfg, subset, int(obs.y.size))
    assert np.array_equal(built, obs.basis_belief)


def test_protect_targets_false_returns_empty_basis():
    """``protect_targets=False`` must return an empty basis, not fall through.

    Regression: both early exits in ``_protection_basis`` built the zero-column
    array without returning it, so execution continued into the filtered block
    list.  The scene sizes used everywhere else never reach these branches, so
    the bug was invisible to every full-scene test.
    """
    cfg = make_cfg(**{"cancellation.protect_targets": False})
    obs = make_observation(cfg, belief_error=True)
    sources = obs.targets_belief or obs.targets
    assert sources, "the fixture must still carry sources"
    U = cx._protection_basis(cfg, sources, int(obs.y.size))
    assert U.shape == (int(obs.y.size), 0)
    assert U.dtype == np.complex128
    assert cx.protected_target_ids(cfg, sources) == frozenset()


def test_empty_protection_sources_returns_empty_basis():
    """An empty source list must return an empty basis -- never raise downstream.

    This is the shape a caller sees after an active mask filters every source
    away, or in a unit test with no targets.  Falling through here reaches
    ``np.concatenate([])``.
    """
    cfg = make_cfg()
    U = cx._protection_basis(cfg, [], 64)
    # The row count is the observation dimension ``N*L`` -- note the caller's
    # ``n_bins`` argument is not what decides it, so pin the real authority.
    assert U.shape == (int(cfg.waveform.N * cfg.waveform.L), 0)
    assert U.shape[1] == 0
    assert cx.protected_target_ids(cfg, []) == frozenset()


def test_disabled_protection_runs_the_full_arm_pipeline():
    """The empty basis must survive the whole chain, not just the constructor.

    ``protect_targets=False`` is the switch that turns TP-UIC into a plain
    canceller; with the missing ``return`` this call raised before the fix.
    """
    cfg = make_cfg(**{"cancellation.protect_targets": False})
    obs = make_observation(cfg, belief_error=True)
    assert obs.basis_belief.shape[1] == 0
    arms = cx.cancellation_arms(cfg, obs)
    full = arms["tp_uic_full"]
    assert full.i_res > 0.0
    assert 0.0 <= full.eta_survive <= 1.0
    assert np.all(np.isfinite(full.residual))
