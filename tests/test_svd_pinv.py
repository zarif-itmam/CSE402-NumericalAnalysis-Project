"""Focused numerical properties for Member B's explicit SVD solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.solvers.svd_pinv import solve_svd_pinv


def test_full_rank_square_solution_agrees_with_direct_solve():
    """A well-conditioned square system has the same unique solution by SVD or LU."""
    J = np.array([[4.0, 1.0], [2.0, 3.0]])
    rhs = np.array([9.0, 8.0])

    dx, diagnostics = solve_svd_pinv(J, rhs)

    np.testing.assert_allclose(dx, np.linalg.solve(J, rhs), rtol=1e-13, atol=1e-13)
    assert diagnostics["success"] is True
    assert diagnostics["numerical_rank"] == 2
    assert diagnostics["truncated_singular_values"] == 0
    assert diagnostics["residual_norm"] < 1e-12


def test_exact_rank_deficiency_matches_pseudoinverse_least_squares_solution():
    """A dependent-column system keeps success and returns NumPy's minimum-norm LS result."""
    J = np.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    rhs = np.array([1.0, 2.0, 1.0])  # Intentionally inconsistent with col(J).

    dx, diagnostics = solve_svd_pinv(J, rhs)

    np.testing.assert_allclose(dx, np.linalg.pinv(J) @ rhs, rtol=1e-13, atol=1e-13)
    assert diagnostics["success"] is True
    assert diagnostics["numerical_rank"] == 1
    assert diagnostics["truncated_singular_values"] == 1
    # A nonzero least-squares residual is expected here and is not a failure.
    assert diagnostics["residual_norm"] > 0.0


def test_rtol_controls_rank_for_known_singular_value_spectrum():
    """The documented strict ``sigma > tau`` rule changes rank predictably."""
    J = np.diag([1.0, 1e-6, 1e-12])
    rhs = np.ones(3)

    _, medium_cutoff = solve_svd_pinv(J, rhs, rtol=1e-9)
    _, coarse_cutoff = solve_svd_pinv(J, rhs, rtol=1e-5)

    assert medium_cutoff["rank_tolerance"] == pytest.approx(1e-9)
    assert medium_cutoff["numerical_rank"] == 2
    assert medium_cutoff["truncated_singular_values"] == 1
    assert coarse_cutoff["rank_tolerance"] == pytest.approx(1e-5)
    assert coarse_cutoff["numerical_rank"] == 1
    assert coarse_cutoff["truncated_singular_values"] == 2


def test_retained_rank_is_monotone_nonincreasing_with_rtol():
    """For a fixed spectrum, increasing the cutoff cannot restore a truncated mode."""
    J = np.diag([1.0, 1e-5, 1e-9])
    rhs = np.ones(3)

    ranks = [
        solve_svd_pinv(J, rhs, rtol=rtol)[1]["numerical_rank"]
        for rtol in (1e-12, 1e-7, 1e-3)
    ]

    assert ranks == [3, 2, 1]
    assert ranks == sorted(ranks, reverse=True)


def test_underdetermined_solution_is_minimum_norm_pseudoinverse_solution():
    """For infinitely many exact solutions, MP-PI selects the trusted minimum-norm one."""
    J = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
    rhs = np.array([1.0, -2.0])

    dx, diagnostics = solve_svd_pinv(J, rhs)
    reference = np.linalg.pinv(J) @ rhs

    np.testing.assert_allclose(dx, reference, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(J @ dx, rhs, rtol=1e-13, atol=1e-13)
    assert diagnostics["numerical_rank"] == 2
    assert diagnostics["step_norm"] == pytest.approx(np.linalg.norm(reference))


def test_diagnostics_are_finite_and_internally_consistent():
    """Reported rank and conditioning quantities agree with the returned spectrum."""
    J = np.diag([5.0, 1e-4, 1e-10])
    rhs = np.array([1.0, -2.0, 3.0])

    _, diagnostics = solve_svd_pinv(J, rhs, rtol=1e-6, atol=1e-8)

    singular_values = np.asarray(diagnostics["singular_values"])
    retained = singular_values > diagnostics["rank_tolerance"]
    assert diagnostics["method"] == "svd"
    assert diagnostics["success"] is True
    assert np.isfinite(diagnostics["runtime_sec"])
    assert np.isfinite(diagnostics["residual_norm"])
    assert diagnostics["numerical_rank"] == int(np.count_nonzero(retained))
    assert diagnostics["truncated_singular_values"] == singular_values.size - diagnostics["numerical_rank"]
    assert diagnostics["sigma_max"] == pytest.approx(singular_values[0])
    assert diagnostics["sigma_min"] == pytest.approx(singular_values[-1])
    assert diagnostics["effective_condition_number"] == pytest.approx(
        singular_values[0] / singular_values[retained][-1]
    )


def test_nonfinite_input_returns_clear_solver_failure():
    """NaN data must not silently reach SVD or produce a purported update."""
    dx, diagnostics = solve_svd_pinv(np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2))

    assert dx is None
    assert diagnostics["success"] is False
    assert "finite" in diagnostics["error_message"]
    assert np.isfinite(diagnostics["runtime_sec"])


def test_invalid_dimensions_raise_value_error_before_linear_algebra():
    """Shape violations are caller-contract errors, not rank-deficiency outcomes."""
    with pytest.raises(ValueError, match="rhs length"):
        solve_svd_pinv(np.eye(2), np.ones(3))
