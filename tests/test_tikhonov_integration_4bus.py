"""Shared-Newton integration checks for Member D's Tikhonov 4-bus path."""

from __future__ import annotations

import numpy as np

from experiments.exp_d_tikhonov_4bus import run_4bus_case


def _solver_entries(result: dict) -> list[dict]:
    """Select NR history records produced by an actual linear solve."""
    return [entry for entry in result["history"] if "solver_diagnostics" in entry]


def test_healthy_case_tikhonov_dispatch_converges_and_is_close_to_direct():
    """A small regularization strength should barely perturb the healthy-case solution."""
    direct = run_4bus_case(
        label="test_healthy_direct", load_on_secondary=True, method="direct"
    )
    tikhonov = run_4bus_case(
        label="test_healthy_tikhonov",
        load_on_secondary=True,
        method="tikhonov",
        alpha=1e-10,
    )

    assert direct["converged"] is True
    assert tikhonov["converged"] is True
    # Regularization biases the solution, so this is a loose closeness check
    # (not exact agreement, unlike the SVD/RRQR unregularized comparisons).
    np.testing.assert_allclose(tikhonov["x"], direct["x"], rtol=1e-4, atol=1e-4)


def test_tikhonov_diagnostics_are_exposed_in_shared_nr_history():
    """Every shared-loop Tikhonov solve must preserve its regularization diagnostics."""
    result = run_4bus_case(
        label="test_healthy_tikhonov_diagnostics",
        load_on_secondary=True,
        method="tikhonov",
        alpha=1e-8,
    )
    solves = _solver_entries(result)

    assert result["converged"] is True
    assert solves
    for entry in solves:
        diagnostics = entry["solver_diagnostics"]
        assert entry["method"] == "tikhonov"
        assert diagnostics["method"] == "tikhonov"
        assert diagnostics["success"] is True
        assert diagnostics["alpha"] == 1e-8
        assert diagnostics["lambda_reg"] >= 0.0
        assert diagnostics["sigma_max_J"] > 0.0
        assert np.isfinite(diagnostics["residual_norm"])
        assert np.isfinite(diagnostics["runtime_sec"])


def test_floating_case_tikhonov_path_completes_without_solver_crash():
    """The floating reconstruction may converge or not, but Tikhonov diagnostics must stay valid.

    This is the case Tikhonov is actually meant for: a direct solve fails or
    ill-conditions there (see ``experiments/exp01_4bus.py`` Case B), and the
    solver must not crash even under severe ill-conditioning.
    """
    result = run_4bus_case(
        label="test_floating_tikhonov",
        load_on_secondary=False,
        method="tikhonov",
        alpha=1e-6,
    )
    solves = _solver_entries(result)

    assert solves
    assert all(entry["solver_diagnostics"]["method"] == "tikhonov" for entry in solves)
    assert all(entry["solver_diagnostics"]["success"] for entry in solves)
    assert all(np.isfinite(entry["solver_diagnostics"]["step_norm"]) for entry in solves)
    assert result["fail_reason"] is None or isinstance(result["fail_reason"], str)


def test_zero_alpha_tikhonov_matches_direct_on_healthy_case():
    """The unregularized limit (alpha=0) should reproduce the direct solver's path closely."""
    direct = run_4bus_case(
        label="test_healthy_direct", load_on_secondary=True, method="direct"
    )
    tikhonov_unregularized = run_4bus_case(
        label="test_healthy_tikhonov_zero",
        load_on_secondary=True,
        method="tikhonov",
        alpha=0.0,
    )

    assert direct["converged"] is True
    assert tikhonov_unregularized["converged"] is True
    np.testing.assert_allclose(
        tikhonov_unregularized["x"], direct["x"], rtol=1e-8, atol=1e-8
    )
