"""Controlled common-matrix checks for currently available linear solvers.

These tests exercise only the public direct and SVD solver interfaces.  They
provide deterministic reference matrices for later extension by the RRQR and
Tikhonov owners without importing, implementing, or assuming those solvers.
"""

from __future__ import annotations

import numpy as np

from src.solvers.direct import COND_NUMBER_FAIL_THRESHOLD, solve_direct
from src.solvers.rrqr import solve_rrqr
from src.solvers.svd_pinv import solve_svd_pinv


def controlled_spectrum_matrix() -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic matrix with four retained modes and one null mode.

    Orthogonal wrapping avoids testing only a diagonal special case while
    preserving the intended singular spectrum ``[1, 1e-2, 1e-6, 1e-12, 0]``.
    The SVD default threshold is about 1e-15 here, so its expected numerical
    rank is four and its expected truncation count is one.
    """
    rng = np.random.default_rng(2105117)
    left, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    right, _ = np.linalg.qr(rng.standard_normal((5, 5)))
    singular_values = np.array([1.0, 1e-2, 1e-6, 1e-12, 0.0])
    return left @ np.diag(singular_values) @ right.T, singular_values


def test_well_conditioned_full_rank_direct_and_svd_agree():
    """Unique well-conditioned systems must agree across the two available APIs."""
    J = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = np.array([2.0, -1.0, 3.0])

    direct_dx, direct = solve_direct(J, rhs)
    svd_dx, svd = solve_svd_pinv(J, rhs)

    assert direct["success"] is True
    assert svd["success"] is True
    assert np.isfinite(direct["runtime_sec"])
    assert np.isfinite(direct["residual_norm"])
    assert np.isfinite(svd["runtime_sec"])
    assert np.isfinite(svd["residual_norm"])
    np.testing.assert_allclose(svd_dx, direct_dx, rtol=1e-13, atol=1e-13)


def test_ill_conditioned_but_full_rank_system_remains_solvable_below_guard():
    """Conditioning near 1e10 is finite and below the direct guard, so both solve."""
    J = np.diag([1.0, 1e-5, 1e-10])
    rhs = np.array([1.0, -2e-5, 3e-10])

    direct_dx, direct = solve_direct(J, rhs)
    svd_dx, svd = solve_svd_pinv(J, rhs)

    assert direct["success"] is True
    assert np.isfinite(direct["condition_number"])
    assert direct["condition_number"] < COND_NUMBER_FAIL_THRESHOLD
    assert svd["success"] is True
    assert svd["numerical_rank"] == 3
    np.testing.assert_allclose(svd_dx, direct_dx, rtol=1e-13, atol=1e-13)


def test_direct_guard_rejects_full_rank_system_without_claiming_exact_singularity():
    """The project guard can reject a finite LU solve without a LinAlgError."""
    J = np.diag([1.0, 1e-13])
    rhs = np.array([1.0, 1e-13])

    # This independent call documents that the matrix is finite/full rank and
    # NumPy's direct routine itself can produce an update.
    assert np.all(np.isfinite(np.linalg.solve(J, rhs)))
    dx, diagnostics = solve_direct(J, rhs)

    assert dx is None
    assert diagnostics["success"] is False
    assert diagnostics["failure_mode"] == "ill_conditioned"
    assert diagnostics["condition_number"] > COND_NUMBER_FAIL_THRESHOLD
    assert np.isfinite(diagnostics["residual_norm"])


def test_exact_rank_deficiency_is_svd_success_but_direct_exact_singular_failure():
    """MP-PI intentionally handles a null mode; direct solve reports its own failure mode."""
    J = np.diag([1.0, 1e-2, 1e-6, 1e-12, 0.0])
    rhs = np.array([1.0, -1e-2, 2e-6, -3e-12, 0.5])

    direct_dx, direct = solve_direct(J, rhs)
    svd_dx, svd = solve_svd_pinv(J, rhs)

    assert direct_dx is None
    assert direct["success"] is False
    assert direct["failure_mode"] == "exact_singular"
    assert svd["success"] is True
    assert svd["numerical_rank"] == 4
    assert svd["truncated_singular_values"] == 1
    np.testing.assert_allclose(svd_dx, np.linalg.pinv(J) @ rhs, rtol=1e-13, atol=1e-13)


def test_controlled_spectrum_rank_and_diagnostics_are_finite():
    """Known singular values validate SVD rank reporting beyond a diagonal-only case."""
    J, expected_singular_values = controlled_spectrum_matrix()
    rhs = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

    dx, diagnostics = solve_svd_pinv(J, rhs)

    assert diagnostics["success"] is True
    assert diagnostics["numerical_rank"] == 4
    assert diagnostics["truncated_singular_values"] == 1
    assert np.all(np.isfinite(dx))
    assert np.isfinite(diagnostics["runtime_sec"])
    assert np.isfinite(diagnostics["residual_norm"])
    np.testing.assert_allclose(
        diagnostics["singular_values"][:4], expected_singular_values[:4], rtol=1e-12, atol=1e-15
    )


def test_well_conditioned_full_rank_direct_svd_and_rrqr_agree():
    """Unique well-conditioned systems must agree across all three available APIs."""
    J = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    rhs = np.array([2.0, -1.0, 3.0])

    direct_dx, direct = solve_direct(J, rhs)
    svd_dx, svd = solve_svd_pinv(J, rhs)
    rrqr_dx, rrqr = solve_rrqr(J, rhs)

    assert direct["success"] is True
    assert svd["success"] is True
    assert rrqr["success"] is True
    assert np.isfinite(rrqr["runtime_sec"])
    assert np.isfinite(rrqr["residual_norm"])
    np.testing.assert_allclose(rrqr_dx, direct_dx, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(rrqr_dx, svd_dx, rtol=1e-11, atol=1e-11)


def test_ill_conditioned_but_full_rank_system_rrqr_agrees_with_direct_and_svd():
    """Conditioning near 1e10 is finite and below the direct guard, so all three solve."""
    J = np.diag([1.0, 1e-5, 1e-10])
    rhs = np.array([1.0, -2e-5, 3e-10])

    direct_dx, direct = solve_direct(J, rhs)
    svd_dx, svd = solve_svd_pinv(J, rhs)
    rrqr_dx, rrqr = solve_rrqr(J, rhs)

    assert direct["success"] is True
    assert svd["success"] is True
    assert rrqr["success"] is True
    assert rrqr["numerical_rank"] == 3
    np.testing.assert_allclose(rrqr_dx, direct_dx, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rrqr_dx, svd_dx, rtol=1e-9, atol=1e-9)


def test_exact_rank_deficiency_is_rrqr_success_but_direct_exact_singular_failure():
    """RRQR intentionally handles a null mode; direct solve reports its own failure mode."""
    J = np.diag([1.0, 1e-2, 1e-6, 1e-12, 0.0])
    rhs = np.array([1.0, -1e-2, 2e-6, -3e-12, 0.5])

    direct_dx, direct = solve_direct(J, rhs)
    svd_dx, svd = solve_svd_pinv(J, rhs)
    rrqr_dx, rrqr = solve_rrqr(J, rhs)

    assert direct_dx is None
    assert direct["success"] is False
    assert direct["failure_mode"] == "exact_singular"
    assert rrqr["success"] is True
    assert rrqr["numerical_rank"] == svd["numerical_rank"] == 4
    assert rrqr["truncated_columns"] == svd["truncated_singular_values"] == 1
    np.testing.assert_allclose(rrqr_dx, svd_dx, rtol=1e-9, atol=1e-9)


def test_controlled_spectrum_rrqr_rank_matches_svd_rank():
    """RRQR and SVD must agree on numerical rank for a known non-diagonal spectrum."""
    J, expected_singular_values = controlled_spectrum_matrix()
    rhs = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

    svd_dx, svd = solve_svd_pinv(J, rhs)
    rrqr_dx, rrqr = solve_rrqr(J, rhs)

    assert rrqr["success"] is True
    assert rrqr["numerical_rank"] == svd["numerical_rank"] == 4
    assert rrqr["truncated_columns"] == svd["truncated_singular_values"] == 1
    assert np.all(np.isfinite(rrqr_dx))
    assert np.isfinite(rrqr["runtime_sec"])
    # Both are valid least-squares solutions of a rank-deficient system, so
    # they need not be numerically identical -- only equally consistent with
    # the same known spectrum. Column pivoting picks the largest-norm
    # remaining column at each step, so |R_00| is only bounded above by
    # sigma_max (interlacing), not equal to it in general.
    assert 0.0 < rrqr["r_diagonal"][0] <= expected_singular_values[0] + 1e-10
