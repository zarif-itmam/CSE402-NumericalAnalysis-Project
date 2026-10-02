"""Focused numerical properties for Member D's Tikhonov-regularized solver."""

from __future__ import annotations

import numpy as np
import pytest

from src.solvers.tikhonov import solve_tikhonov


def test_zero_lambda_recovers_ordinary_least_squares():
    """With no regularization, the augmented-system solve matches a direct solve."""
    J = np.array([[4.0, 1.0], [2.0, 3.0]])
    rhs = np.array([9.0, 8.0])

    dx, diagnostics = solve_tikhonov(J, rhs, lam=0.0)

    np.testing.assert_allclose(dx, np.linalg.solve(J, rhs), rtol=1e-10, atol=1e-10)
    assert diagnostics["success"] is True
    assert diagnostics["lambda_reg"] == 0.0
    assert diagnostics["residual_norm"] < 1e-9


def test_explicit_lambda_matches_closed_form_ridge_solution():
    """The augmented least-squares solve must equal the textbook ridge formula.

    ``(J.T @ J + lambda * I)^-1 @ J.T @ rhs`` is only used here to check the
    numerically-preferred augmented-system implementation against a known
    closed form; the solver itself never forms this normal-equations matrix.
    """
    J = np.array([[4.0, 1.0], [2.0, 3.0]])
    rhs = np.array([9.0, 8.0])
    lam = 2.0

    dx, diagnostics = solve_tikhonov(J, rhs, lam=lam)
    closed_form = np.linalg.solve(J.T @ J + lam * np.eye(2), J.T @ rhs)

    np.testing.assert_allclose(dx, closed_form, rtol=1e-10, atol=1e-10)
    assert diagnostics["lambda_reg"] == pytest.approx(lam)
    assert np.isnan(diagnostics["alpha"])
    assert np.isnan(diagnostics["sigma_max_J"])


def test_alpha_scaling_derives_lambda_from_sigma_max():
    """``alpha`` must scale to ``lambda_reg = alpha * sigma_max(J) ** 2``."""
    J = np.diag([3.0, 1.0])
    rhs = np.array([1.0, 1.0])
    alpha = 0.1

    dx, diagnostics = solve_tikhonov(J, rhs, alpha=alpha)

    expected_sigma_max = 3.0
    expected_lambda = alpha * expected_sigma_max ** 2
    assert diagnostics["sigma_max_J"] == pytest.approx(expected_sigma_max)
    assert diagnostics["lambda_reg"] == pytest.approx(expected_lambda)
    assert diagnostics["alpha"] == pytest.approx(alpha)

    closed_form = np.linalg.solve(J.T @ J + expected_lambda * np.eye(2), J.T @ rhs)
    np.testing.assert_allclose(dx, closed_form, rtol=1e-10, atol=1e-10)


def test_larger_alpha_shrinks_step_norm_monotonically():
    """Increasing regularization strength must not increase the step norm.

    This is a structural ridge-regression property (the shrinkage effect),
    not a claim about any specific accuracy or convergence outcome.
    """
    J = np.diag([1.0, 1e-3, 1e-6])
    rhs = np.array([1.0, 1.0, 1.0])

    step_norms = [
        solve_tikhonov(J, rhs, alpha=alpha)[1]["step_norm"]
        for alpha in (1e-12, 1e-8, 1e-4, 1e-2)
    ]

    assert step_norms == sorted(step_norms, reverse=True)


def test_singular_matrix_returns_finite_regularized_solution():
    """A rank-deficient/exactly-singular J must not crash: that is Tikhonov's purpose."""
    J = np.diag([1.0, 1e-2, 1e-6, 1e-12, 0.0])
    rhs = np.array([1.0, -1e-2, 2e-6, -3e-12, 0.5])

    dx, diagnostics = solve_tikhonov(J, rhs, alpha=1e-6)

    assert diagnostics["success"] is True
    assert np.all(np.isfinite(dx))
    assert np.isfinite(diagnostics["residual_norm"])
    assert np.isfinite(diagnostics["step_norm"])
    # The exactly-null column cannot be resolved even with regularization:
    # any update in that direction is penalized without reducing residual.
    assert dx[4] == pytest.approx(0.0, abs=1e-9)


def test_mutually_exclusive_alpha_and_lam_raise_value_error():
    """Supplying both regularization forms is an ambiguous caller-contract error."""
    with pytest.raises(ValueError, match="mutually exclusive"):
        solve_tikhonov(np.eye(2), np.ones(2), alpha=1e-6, lam=1e-6)


def test_negative_lambda_raises_value_error():
    """A negative ridge parameter is not a valid regularization strength."""
    with pytest.raises(ValueError, match="non-negative"):
        solve_tikhonov(np.eye(2), np.ones(2), lam=-1.0)


def test_nonfinite_input_returns_clear_solver_failure():
    """NaN data must not silently reach the augmented solve or produce a purported update."""
    dx, diagnostics = solve_tikhonov(np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2))

    assert dx is None
    assert diagnostics["success"] is False
    assert "finite" in diagnostics["error_message"]
    assert np.isfinite(diagnostics["runtime_sec"])


def test_invalid_dimensions_raise_value_error_before_linear_algebra():
    """Shape violations are caller-contract errors, not regularization outcomes."""
    with pytest.raises(ValueError, match="rhs length"):
        solve_tikhonov(np.eye(2), np.ones(3))


def test_precomputed_sigma_max_is_used_instead_of_recomputed():
    """Passing ``sigma_max`` must skip recomputation and drive lambda directly."""
    J = np.diag([3.0, 1.0])
    rhs = np.array([1.0, 1.0])

    _, diagnostics = solve_tikhonov(J, rhs, alpha=0.1, sigma_max=10.0)

    assert diagnostics["sigma_max_J"] == pytest.approx(10.0)
    assert diagnostics["lambda_reg"] == pytest.approx(0.1 * 10.0 ** 2)
