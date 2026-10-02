"""Shared-Newton integration checks for Member B's SVD 4-bus path."""

from __future__ import annotations

import numpy as np

from experiments.exp_b_svd_4bus import run_4bus_case


def _solver_entries(result: dict) -> list[dict]:
    """Select NR history records produced by an actual linear solve."""
    return [entry for entry in result["history"] if "solver_diagnostics" in entry]


def test_healthy_case_svd_dispatch_converges_and_agrees_with_direct():
    """The shared NR loop should differ only in its solver choice on the same case."""
    direct = run_4bus_case(
        label="test_healthy_direct", load_on_secondary=True, method="direct"
    )
    svd = run_4bus_case(
        label="test_healthy_svd", load_on_secondary=True, method="svd"
    )

    assert direct["converged"] is True
    assert svd["converged"] is True
    np.testing.assert_allclose(svd["x"], direct["x"], rtol=1e-9, atol=1e-9)


def test_svd_diagnostics_are_exposed_in_shared_nr_history():
    """Every shared-loop SVD solve must preserve its rank diagnostics for reporting."""
    result = run_4bus_case(
        label="test_healthy_svd_diagnostics", load_on_secondary=True, method="svd"
    )
    solves = _solver_entries(result)

    assert result["converged"] is True
    assert solves
    for entry in solves:
        diagnostics = entry["solver_diagnostics"]
        assert entry["method"] == "svd"
        assert diagnostics["method"] == "svd"
        assert diagnostics["success"] is True
        assert diagnostics["numerical_rank"] > 0
        assert diagnostics["rank_tolerance"] >= 0.0
        assert diagnostics["truncated_singular_values"] >= 0
        assert np.isfinite(diagnostics["residual_norm"])
        assert np.isfinite(diagnostics["runtime_sec"])


def test_floating_case_svd_path_completes_without_solver_crash():
    """The floating reconstruction may converge or not, but SVD diagnostics must remain valid."""
    result = run_4bus_case(
        label="test_floating_svd", load_on_secondary=False, method="svd"
    )
    solves = _solver_entries(result)

    assert solves
    assert all(entry["solver_diagnostics"]["method"] == "svd" for entry in solves)
    assert all("rank_tolerance" in entry["solver_diagnostics"] for entry in solves)
    assert all("singular_values" in entry["solver_diagnostics"] for entry in solves)
    assert result["fail_reason"] is None or isinstance(result["fail_reason"], str)
