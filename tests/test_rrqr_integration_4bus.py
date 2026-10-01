"""Shared-Newton integration checks for Member C's RRQR 4-bus path."""

from __future__ import annotations

import numpy as np

from experiments.exp_c_rrqr_4bus import run_4bus_case


def _solver_entries(result: dict) -> list[dict]:
    """Select NR history records produced by an actual linear solve."""
    return [entry for entry in result["history"] if "solver_diagnostics" in entry]


def test_healthy_case_rrqr_dispatch_converges_and_agrees_with_direct():
    """The shared NR loop should differ only in its solver choice on the same case."""
    direct = run_4bus_case(
        label="test_healthy_direct", load_on_secondary=True, method="direct"
    )
    rrqr = run_4bus_case(
        label="test_healthy_rrqr", load_on_secondary=True, method="rrqr"
    )

    assert direct["converged"] is True
    assert rrqr["converged"] is True
    np.testing.assert_allclose(rrqr["x"], direct["x"], rtol=1e-9, atol=1e-9)


def test_rrqr_diagnostics_are_exposed_in_shared_nr_history():
    """Every shared-loop RRQR solve must preserve its rank diagnostics for reporting."""
    result = run_4bus_case(
        label="test_healthy_rrqr_diagnostics", load_on_secondary=True, method="rrqr"
    )
    solves = _solver_entries(result)

    assert result["converged"] is True
    assert solves
    for entry in solves:
        diagnostics = entry["solver_diagnostics"]
        assert entry["method"] == "rrqr"
        assert diagnostics["method"] == "rrqr"
        assert diagnostics["success"] is True
        assert diagnostics["numerical_rank"] > 0
        assert diagnostics["rank_tolerance"] >= 0.0
        assert diagnostics["truncated_columns"] >= 0
        assert len(diagnostics["pivot_ordering"]) == len(diagnostics["r_diagonal"])
        assert np.isfinite(diagnostics["residual_norm"])
        assert np.isfinite(diagnostics["runtime_sec"])


def test_floating_case_rrqr_path_completes_without_solver_crash():
    """The floating reconstruction may converge or not, but RRQR diagnostics must remain valid."""
    result = run_4bus_case(
        label="test_floating_rrqr", load_on_secondary=False, method="rrqr"
    )
    solves = _solver_entries(result)

    assert solves
    assert all(entry["solver_diagnostics"]["method"] == "rrqr" for entry in solves)
    assert all("rank_tolerance" in entry["solver_diagnostics"] for entry in solves)
    assert all("r_diagonal" in entry["solver_diagnostics"] for entry in solves)
    assert result["fail_reason"] is None or isinstance(result["fail_reason"], str)


def test_svd_and_rrqr_agree_on_rank_across_the_full_4bus_history():
    """The two rank-revealing solvers should reach the same rank decision at every iteration.

    Both default thresholds use ``eps * max(J.shape)``, and Stage 04 (Member
    B) observed all-18 retained singular values through convergence on this
    reconstruction for both the healthy and floating cases, so equality is
    expected here rather than merely plausible.
    """
    from experiments.exp_b_svd_4bus import run_4bus_case as run_4bus_case_svd

    for load_on_secondary in (True, False):
        svd_result = run_4bus_case_svd(
            label="svd", load_on_secondary=load_on_secondary, method="svd"
        )
        rrqr_result = run_4bus_case(
            label="rrqr", load_on_secondary=load_on_secondary, method="rrqr"
        )

        svd_solves = _solver_entries(svd_result)
        rrqr_solves = _solver_entries(rrqr_result)
        assert len(svd_solves) == len(rrqr_solves)
        for svd_entry, rrqr_entry in zip(svd_solves, rrqr_solves):
            assert (
                svd_entry["solver_diagnostics"]["numerical_rank"]
                == rrqr_entry["solver_diagnostics"]["numerical_rank"]
            )
