"""Invariants of the angular-identifiability probe.

The probe answers one question -- does a spatial dimension rescue the
delay-Doppler collision -- and the whole answer rests on the probe measuring the
*same* ``rho`` as the production audit.  These tests pin that, plus the two
defects the probe actually had on its first runs (the DD-offset order and the
raw-amplitude Gram), so a future edit cannot quietly reintroduce them.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tools import probe_angular_identifiability as pb


# --------------------------------------------------------------------------
# The array model
# --------------------------------------------------------------------------
def test_beamwidth_scales_inversely_with_the_aperture():
    """``BW(M) = 0.886 / M`` in ``u``: twice the elements, half the beam."""
    assert pb.beamwidth_u(16) == pytest.approx(pb.beamwidth_u(8) / 2.0)


def test_the_array_factor_is_unit_gain_and_bounded():
    """``A(0) = 1`` and ``|A| <= 1``: unit-norm steering, no aperture gain.

    This is the confound guard.  With ``||a|| = 1`` the total collected power is
    independent of ``M``, so a rise in ``rho`` with ``M`` cannot be the echo
    simply getting stronger -- which is the only way the probe is allowed to
    report an identifiability result.
    """
    for m in (2, 4, 8, 16):
        assert pb.array_factor(m, 0.0) == pytest.approx(1.0)
        for du in (0.05, 0.2, 0.886 / m, 1.0, 3.0):
            assert abs(pb.array_factor(m, du)) <= 1.0 + 1e-12


def test_the_curve_is_a_function_of_separation_in_beamwidths_alone():
    """The core finding, pinned analytically before any simulation runs.

    ``A`` depends on ``M * delta_u`` alone *asymptotically*, so measuring the
    separation in beamwidths (``0.886 / M``) makes the curve almost ``M``-free --
    exactly free in the limit ``M * delta_u / 2 -> 0``.  That is why the report
    must use beamwidths: in degrees the same geometry reads differently for every
    aperture, and the "how many beamwidths are needed" verdict would move with the
    hardware choice it is supposed to inform.

    The deviation at small ``M`` is real and is tolerated rather than hidden, by
    asserting what is actually true -- that it *shrinks* as ``M`` grows (measured
    worst case 1.6e-2 between ``M = 4`` and ``M = 16`` at two beamwidths).
    """
    for frac in (0.25, 0.5, 1.0, 2.0, 3.0):
        close = abs(pb.analytic_rho(4, frac * pb.beamwidth_u(4))
                    - pb.analytic_rho(8, frac * pb.beamwidth_u(8)))
        closer = abs(pb.analytic_rho(8, frac * pb.beamwidth_u(8))
                     - pb.analytic_rho(16, frac * pb.beamwidth_u(16)))
        assert closer <= close + 1e-12, (frac, close, closer)
        assert closer < 1e-2, (frac, closer)


def test_the_beamwidth_convention_converges_to_the_three_db_point():
    """``rho(1 beamwidth) -> 0.5`` as ``M`` grows: that is what ``0.886 / M`` means.

    Pinning it matters because every verdict here is stated in beamwidths; if the
    constant drifted, "needs 2 beamwidths" would silently become a different angle
    and the aperture recommendation would be wrong while every number still looked
    plausible.  ``0.886 / M`` is the *asymptotic* 3 dB width -- the deviation has
    to shrink with ``M`` (measured 0.48 at M = 4, 0.495 at M = 8, 0.499 at
    M = 16), which is the assertion below rather than a fixed window that would
    pass by luck at one aperture.
    """
    dev = [abs(pb.analytic_rho(m, pb.beamwidth_u(m)) - 0.5) for m in (4, 8, 16, 32)]
    assert all(b <= a + 1e-12 for a, b in zip(dev, dev[1:])), dev
    assert dev[-1] < 0.005


def test_analytic_rho_rises_over_the_mainlobe_then_oscillates():
    """Monotone only inside the mainlobe; the sidelobes make it non-monotone.

    Written after the first version of this test failed: the Dirichlet kernel
    oscillates, so ``rho`` reaches 0.98 around two beamwidths, dips at three, and
    recovers.  Claiming monotonicity would hide that a *larger* separation is not
    always a *better* one, which matters if anyone tries to optimise separation.
    """
    m = 8
    bw = pb.beamwidth_u(m)
    mainlobe = [0.0, 0.1, 0.25, 0.5, 0.8, 1.0]
    values = [pb.analytic_rho(m, f * bw) for f in mainlobe]
    assert values[0] == pytest.approx(0.0, abs=1e-15)
    assert all(b > a for a, b in zip(values, values[1:]))
    assert pb.analytic_rho(m, 2.0 * bw) > 0.95
    assert pb.analytic_rho(m, 3.0 * bw) < pb.analytic_rho(m, 2.0 * bw)


# --------------------------------------------------------------------------
# The Gram shortcut
# --------------------------------------------------------------------------
def test_identical_templates_are_completely_masked():
    """The DD-only degeneracy: the same template cannot escape itself."""
    v = np.random.default_rng(0).normal(size=64) + 1j * np.random.default_rng(1).normal(size=64)
    gram = np.array([[np.vdot(v, v), np.vdot(v, v)],
                     [np.vdot(v, v), np.vdot(v, v)]], dtype=complex)
    assert pb.escape_from_gram(gram, 0, [1]) == pytest.approx(0.0, abs=1e-12)


def test_escape_fraction_matches_the_analytic_inner_product():
    """Two unit-norm templates: ``rho = 1 - |<a, b>|^2``."""
    rng = np.random.default_rng(7)
    a = rng.normal(size=128) + 1j * rng.normal(size=128)
    b = rng.normal(size=128) + 1j * rng.normal(size=128)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    inner = np.vdot(a, b)
    gram = np.array([[1.0, inner], [np.conj(inner), 1.0]], dtype=complex)
    assert pb.escape_from_gram(gram, 0, [1]) == pytest.approx(1.0 - abs(inner) ** 2)


def test_escape_fraction_is_invariant_to_amplitude_scaling():
    """``rho`` must not move when the templates are rescaled.

    The projection and the denominator scale together, so this is a property of
    the definition -- and it is exactly what the raw-amplitude Gram violated on
    the probe's first run, where the echo amplitudes spanned orders of magnitude,
    the Gram squared that range (vector-domain condition 5.5e6 became 3e13), and
    ``pinv`` returned ``rho = -1.4e6``.  Normalising first is not cosmetic.
    """
    rng = np.random.default_rng(11)
    base = [rng.normal(size=256) + 1j * rng.normal(size=256) for _ in range(3)]
    unit = [v / np.linalg.norm(v) for v in base]

    def rho_of(amps):
        vectors = [u * s for u, s in zip(unit, amps)]
        gram = np.zeros((3, 3), dtype=complex)
        for i in range(3):
            for j in range(3):
                gram[i, j] = np.vdot(vectors[i], vectors[j])
        return pb.escape_from_gram(gram, 0, [1, 2])

    flat = rho_of([1.0, 1.0, 1.0])
    wide = rho_of([1e-7, 3e-9, 4e-12])
    assert flat == pytest.approx(wide, rel=1e-9, abs=1e-12)
    assert 0.0 <= flat <= 1.0


def test_escape_from_gram_matches_the_vector_domain():
    """The certificate: identical arithmetic in both domains.

    Constructed so the case is **well conditioned**: the nuisance pair is
    near-orthogonal and the tested vector is deliberately spread over it, giving a
    mid-range ``rho`` -- which is also the regime the probe's verdict uses.  A
    first version used four near-collinear random vectors and disagreed by 2 %,
    which says nothing about the implementations and everything about the
    conditioning of the case; the scene-level certificate is the anchor step in
    the probe, on real production vectors (agreement there: 2.3e-15).

    Compared on the **masked** energy ``1 - rho``: near ``rho = 1`` the quantity
    ``1 - projected`` loses digits to cancellation.

    ⚠️ The ~1e-6 disagreement recorded here in an earlier version was **not** an
    unexplained numerical residue -- it was a real defect, now fixed.
    """
    rng = np.random.default_rng(23)
    n1, n2, r = (rng.normal(size=512) + 1j * rng.normal(size=512) for _ in range(3))
    v = 0.6 * n1 + 0.6 * n2 + 0.5 * r
    b = np.stack([v, n1, n2], axis=1)
    gram = b.conj().T @ b
    others = [1, 2]
    left = v - b[:, others] @ np.linalg.lstsq(b[:, others], v, rcond=None)[0]
    expected = float(np.vdot(left, left).real / np.vdot(v, v).real)
    got = pb.escape_from_gram(gram, 0, others)
    assert 0.05 < got < 0.95, "test case is degenerate: rho = %.4f" % got
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-12)


def _vector_domain_rho(v: np.ndarray, c: int) -> float:
    """Unambiguous ground truth: project in the vector domain, dense enough that
    conditioning is not the question."""
    others = [j for j in range(v.shape[1]) if j != c]
    vs = v[:, others]
    left = v[:, c] - vs @ np.linalg.lstsq(vs, v[:, c], rcond=None)[0]
    return float(np.vdot(left, left).real / np.vdot(v[:, c], v[:, c]).real)


@pytest.mark.parametrize("n_col", (2, 3, 6, 20))
def test_escape_from_gram_is_right_for_more_than_one_nuisance_column(n_col: int):
    """The defect the two-template certificate could not see.

    ``escape_from_gram`` shipped computing ``conj(u) @ G_ss^+ @ u`` where the
    projection energy is ``u @ G_ss^+ @ conj(u)``.  With **one** nuisance column
    ``G_ss^+`` is a scalar, both expressions are ``|u|^2``, and the function is
    exact -- which is exactly the case the production certificate (agreement
    2.33e-15) and the test above were taken on.  From two nuisance columns up,
    on complex data, the wrong form departs from the vector-domain projection by
    up to 0.23 and goes negative (then clipped to 0, i.e. "no escape at all").

    ``n_col = 2`` is kept in the sweep on purpose: it is the regime where the
    defect is invisible, and seeing it pass next to the others is what makes the
    point.  The columns are made correlated (a Gram that is not close to
    diagonal) because that is where the transposition bites.
    """
    rng = np.random.default_rng(1000 + n_col)
    n_dim = n_col * 8
    v = rng.normal(size=(n_dim, n_col)) + 1j * rng.normal(size=(n_dim, n_col))
    for j in range(1, n_col):  # make the nuisance set correlated
        v[:, j] = 0.7 * v[:, j - 1] + 0.3 * v[:, j]
    v = v / np.linalg.norm(v, axis=0)
    gram = v.conj().T @ v
    for c in range(min(4, n_col)):
        others = [j for j in range(n_col) if j != c]
        got = pb.escape_from_gram(gram, c, others)
        assert got == pytest.approx(_vector_domain_rho(v, c), rel=1e-8, abs=1e-10), (
            "n_col=%d c=%d: %.6f vs vector-domain %.6f" % (
                n_col, c, got, _vector_domain_rho(v, c)))


def test_the_schur_complement_agrees_and_would_have_caught_it():
    """An independent route to the same number: ``rho_c = 1 / inv(corr)[c, c]``.

    One inverse instead of one pseudo-inverse per column (that is what makes the
    per-link probe affordable), and, more importantly here, it has no freedom to
    make the transposition mistake -- so it is a second opinion that disagrees
    with the defective form by ~0.2 and with the corrected one by ~1e-12.
    """
    rng = np.random.default_rng(77)
    v = rng.normal(size=(120, 10)) + 1j * rng.normal(size=(120, 10))
    v = v / np.linalg.norm(v, axis=0)
    gram = v.conj().T @ v
    inv = np.linalg.inv(gram)
    for c in range(10):
        others = [j for j in range(10) if j != c]
        assert pb.escape_from_gram(gram, c, others) == pytest.approx(
            1.0 / float(np.real(inv[c, c])), rel=1e-8, abs=1e-10)


def test_the_gram_shortcut_keeps_the_kronecker_structure():
    """``<a_DD1 (x) a_arr1, a_DD2 (x) a_arr2> = <a_DD1, a_DD2> * A(u1, u2)``.

    The whole probe is fast because of this identity: it turns a ``K*M``-column
    projection into two small Grams.  If it ever stops holding, every reported
    number silently becomes the DD-only one again (the array factor would be
    ignored), which is the failure that looks exactly like a negative result.
    """
    rng = np.random.default_rng(31)
    m_rx = 8
    k1 = rng.normal(size=64) + 1j * rng.normal(size=64)
    k2 = rng.normal(size=64) + 1j * rng.normal(size=64)
    u1, u2 = 0.0, 0.886 / m_rx
    e1 = np.exp(1j * np.pi * np.arange(m_rx) * u1) / math.sqrt(m_rx)
    e2 = np.exp(1j * np.pi * np.arange(m_rx) * u2) / math.sqrt(m_rx)
    lhs = np.vdot(np.kron(k1, e1), np.kron(k2, e2))
    rhs = np.vdot(k1, k2) * np.vdot(e1, e2)
    assert lhs == pytest.approx(rhs, rel=1e-12)
    assert abs(np.vdot(e1, e2)) == pytest.approx(pb.array_factor(m_rx, u2 - u1))
