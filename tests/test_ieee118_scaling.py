"""Smoke tests for the IEEE-118 scaling benchmark (Member E, project spec section 20).

Skips gracefully if pandapower is not installed -- this is a stretch-goal
benchmark, not a hard project dependency for the core NR/solver work.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandapower")

from experiments.exp_e_ieee118_scaling import (
    block_diagonal_replicate,
    load_case118_ybus,
    median_runtime_sec,
    to_real_jacobian_form,
)
from src.solvers.direct import solve_direct


def test_load_case118_ybus_is_square_complex_and_118():
    Y = load_case118_ybus()
    assert Y.shape == (118, 118)
    assert np.iscomplexobj(Y)
    assert np.count_nonzero(Y) > 0


def test_block_diagonal_replicate_preserves_block_structure():
    Y = np.array([[1.0 + 1j, 0.5j], [0.5j, 2.0 + 1j]])
    Y2 = block_diagonal_replicate(Y, 2)

    assert Y2.shape == (4, 4)
    np.testing.assert_array_equal(Y2[:2, :2], Y)
    np.testing.assert_array_equal(Y2[2:, 2:], Y)
    np.testing.assert_array_equal(Y2[:2, 2:], np.zeros((2, 2)))
    np.testing.assert_array_equal(Y2[2:, :2], np.zeros((2, 2)))


def test_to_real_jacobian_form_matches_block_structure():
    Y = np.array([[3.0 + 4j, 1.0 - 2j], [1.0 - 2j, 5.0 + 0.5j]])
    J = to_real_jacobian_form(Y)

    assert J.shape == (4, 4)
    np.testing.assert_allclose(J[:2, :2], Y.real)
    np.testing.assert_allclose(J[:2, 2:], -Y.imag)
    np.testing.assert_allclose(J[2:, :2], Y.imag)
    np.testing.assert_allclose(J[2:, 2:], Y.real)


def test_median_runtime_sec_is_finite_and_positive_on_a_small_real_case():
    Y = load_case118_ybus()
    J = to_real_jacobian_form(Y)
    rng = np.random.default_rng(0)
    rhs = rng.standard_normal(J.shape[0])

    median_time = median_runtime_sec(solve_direct, J, rhs, repeats=2)

    assert np.isfinite(median_time)
    assert median_time > 0.0
