"""Focused numerical properties for Member C's rank-revealing QR solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.solvers.rrqr import solve_rrqr


def test_full_rank_square_solution_agrees_with_direct_solve():
    """A well-conditioned square system has the same unique solution by RRQR or LU."""
    J = np.array([[4.0, 1.0], [2.0, 3.0]])
    rhs = np.array([9.0, 8.0])

    dx, diagnostics = solve_rrqr(J, rhs)

    np.testing.assert_allclose(dx, np.linalg.solve(J, rhs), rtol=1e-12, atol=1e-12)
    assert diagnostics["success"] is True
    assert diagnostics["numerical_rank"] == 2
    assert diagnostics["truncated_columns"] == 0
    assert diagnostics["lapack_rank"] == 2
    assert diagnostics["residual_norm"] < 1e-11


def test_exact_rank_deficiency_matches_pseudoinverse_least_squares_solution():
    """A dependent-column system keeps success and returns the minimum-norm LS result."""
    J = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    rhs = np.array([1.0, 2.0, 1.0])  # Intentionally inconsistent with col(J).

    dx, diagnostics = solve_rrqr(J, rhs)

    np.testing.assert_allclose(dx, np.linalg.pinv(J) @ rhs, rtol=1e-10, atol=1e-10)
    assert diagnostics["success"] is True
    assert diagnostics["numerical_rank"] == 1
    assert diagnostics["truncated_columns"] == 1
    # A nonzero least-squares residual is expected here and is not a failure.
    assert diagnostics["residual_norm"] > 0.0


def test_rtol_controls_rank_for_known_diagonal_spectrum():
    """The documented strict ``|R_ii| > tau`` rule changes rank predictably."""
    J = np.diag([1.0, 1e-6, 1e-12])
    rhs = np.ones(3)

    _, medium_cutoff = solve_rrqr(J, rhs, rtol=1e-9)
    _, coarse_cutoff = solve_rrqr(J, rhs, rtol=1e-5)

    assert medium_cutoff["rank_tolerance"] == pytest.approx(1e-9)
    assert medium_cutoff["numerical_rank"] == 2
    assert medium_cutoff["truncated_columns"] == 1
    assert coarse_cutoff["rank_tolerance"] == pytest.approx(1e-5)
    assert coarse_cutoff["numerical_rank"] == 1
    assert coarse_cutoff["truncated_columns"] == 2


def test_retained_rank_is_monotone_nonincreasing_with_rtol():
    """For a fixed spectrum, increasing the cutoff cannot restore a truncated pivot."""
    J = np.diag([1.0, 1e-5, 1e-9])
    rhs = np.ones(3)

    ranks = [
        solve_rrqr(J, rhs, rtol=rtol)[1]["numerical_rank"]
        for rtol in (1e-12, 1e-7, 1e-3)
    ]

    assert ranks == [3, 2, 1]
    assert ranks == sorted(ranks, reverse=True)


def test_underdetermined_solution_matches_pseudoinverse_solution():
    """For infinitely many exact solutions, GELSY selects a minimum-norm solution."""
    J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    rhs = np.array([1.0, -2.0])

    dx, diagnostics = solve_rrqr(J, rhs)
    reference = np.linalg.pinv(J) @ rhs

    np.testing.assert_allclose(dx, reference, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(J @ dx, rhs, rtol=1e-10, atol=1e-10)
    assert diagnostics["numerical_rank"] == 2


def test_pivot_ordering_is_a_valid_column_permutation():
    """``pivot_ordering`` must be a permutation of the matrix's column indices."""
    J = np.array([[1e-8, 3.0], [1e-9, 4.0], [1e-8, 5.0]])
    rhs = np.array([1.0, -1.0, 2.0])

    _, diagnostics = solve_rrqr(J, rhs)

    pivots = diagnostics["pivot_ordering"]
    assert sorted(pivots) == list(range(J.shape[1]))
    # Column 1 has far larger norm than column 0, so pivoting must select it first.
    assert pivots[0] == 1


def test_diagnostics_are_finite_and_internally_consistent():
    """Reported rank and conditioning quantities agree with the returned R diagonal."""
    J = np.diag([5.0, 1e-4, 1e-10])
    rhs = np.array([1.0, -2.0, 3.0])

    _, diagnostics = solve_rrqr(J, rhs, rtol=1e-6, atol=1e-8)

    r_diagonal = np.asarray(diagnostics["r_diagonal"])
    assert diagnostics["method"] == "rrqr"
    assert diagnostics["success"] is True
    assert np.isfinite(diagnostics["runtime_sec"])
    assert np.isfinite(diagnostics["residual_norm"])
    assert diagnostics["numerical_rank"] + diagnostics["truncated_columns"] == r_diagonal.size
    assert diagnostics["smallest_accepted_r_diagonal"] == pytest.approx(
        r_diagonal[diagnostics["numerical_rank"] - 1]
    )
    assert diagnostics["condition_number"] == pytest.approx(
        r_diagonal[0] / diagnostics["smallest_accepted_r_diagonal"]
    )


def test_zero_matrix_returns_zero_update_without_crashing():
    """A fully rank-zero (numerically zero) Jacobian must not attempt a real GELSY call."""
    J = np.zeros((3, 3))
    rhs = np.array([1.0, -2.0, 3.0])

    dx, diagnostics = solve_rrqr(J, rhs)

    assert diagnostics["success"] is True
    assert diagnostics["numerical_rank"] == 0
    assert diagnostics["truncated_columns"] == 3
    assert diagnostics["condition_number"] == np.inf
    np.testing.assert_array_equal(dx, np.zeros(3))
    assert diagnostics["residual_norm"] == pytest.approx(np.linalg.norm(rhs))


def test_nonfinite_input_returns_clear_solver_failure():
    """NaN data must not silently reach the factorization or produce a purported update."""
    dx, diagnostics = solve_rrqr(np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2))

    assert dx is None
    assert diagnostics["success"] is False
    assert "finite" in diagnostics["error_message"]
    assert np.isfinite(diagnostics["runtime_sec"])


def test_invalid_dimensions_raise_value_error_before_linear_algebra():
    """Shape violations are caller-contract errors, not rank-deficiency outcomes."""
    with pytest.raises(ValueError, match="rhs length"):
        solve_rrqr(np.eye(2), np.ones(3))
