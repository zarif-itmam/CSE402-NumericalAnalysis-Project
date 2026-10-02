"""Member-B SVD comparison on Member A's existing 4-bus reconstruction.

This script deliberately imports ``build_4bus_system`` rather than restating
the network.  It compares only the linear solver inside the shared
Newton--Raphson loop; state representation, mismatch, Jacobian, flat start,
stopping tolerance, iteration limit, and network data remain identical.

The transformer data cited by the imported builder are paper-specified, but
the surrounding 4-bus topology and loads are the team's reconstruction.  The
printed outcomes are therefore project results, not a claim of exact paper
reproduction.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp01_4bus import build_4bus_system
from experiments.member_b_utils import summarise_nr_history
from src.powerflow.newton import newton_raphson


TOL_F = 1e-10
MAX_ITER = 40


def run_4bus_case(*, label: str, load_on_secondary: bool, method: str) -> dict:
    """Run one existing 4-bus case through the shared NR solver interface.

    Parameters are intentionally fixed across calls so the only comparison
    variable is ``method``.  ``compute_conditioning=True`` is used here to
    expose the shared-loop raw conditioning diagnostics; it is not a runtime
    benchmark because that diagnostic itself performs an SVD each iteration.
    """
    state, ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state,
        ybus,
        x0,
        method=method,
        tol_F=TOL_F,
        max_iter=MAX_ITER,
        compute_conditioning=True,
    )
    result["state"] = state
    return result


def print_run(label: str, result: dict) -> None:
    """Print convergence, final state, and method-specific SVD diagnostics."""
    summary = summarise_nr_history(result)
    solves = summary["linear_solves"]
    print("=" * 78)
    print(label)
    print("=" * 78)
    print(f"Converged                 : {result['converged']}")
    print(f"History entries            : {summary['history_entries']}")
    print(f"Actual linear solves       : {len(solves)}")
    print(f"Final ||F||_inf            : {summary['final_F_inf']:.6e}")
    if not result["converged"]:
        print(f"Failure reason             : {result['fail_reason']}")

    if solves and solves[0]["method"] == "svd":
        print("SVD linear-solve diagnostics:")
        print(
            f"{'k':>3} {'rank':>5} {'sigma_max':>13} {'sigma_min':>13} "
            f"{'tau':>13} {'trunc':>6} {'||dx||':>13} {'||Jdx-rhs||':>13} {'t_solve(s)':>11}"
        )
        for entry in solves:
            diag = entry["solver_diagnostics"]
            print(
                f"{entry['iteration']:>3} {diag['numerical_rank']:>5} "
                f"{diag['sigma_max']:>13.6e} {diag['sigma_min']:>13.6e} "
                f"{diag['rank_tolerance']:>13.6e} "
                f"{diag['truncated_singular_values']:>6} "
                f"{diag['step_norm']:>13.6e} {diag['residual_norm']:>13.6e} "
                f"{diag['runtime_sec']:>11.6e}"
            )
    elif solves:
        last = solves[-1]["solver_diagnostics"]
        print(
            "Direct final diagnostic    : "
            f"success={last['success']}, failure_mode={last.get('failure_mode')}, "
            f"condition_number={last.get('condition_number'):.6e}"
        )

    if result["converged"]:
        voltages = result["state"].unpack(result["x"])
        print("Final voltage magnitudes (pu):")
        for key, voltage in voltages.items():
            print(f"  {key[0]}-{key[1]}: {abs(voltage):.9f}")


def run_all_cases() -> dict[str, dict]:
    """Run the four required direct/SVD and loaded/floating comparisons."""
    cases = {
        "healthy_direct": dict(load_on_secondary=True, method="direct"),
        "healthy_svd": dict(load_on_secondary=True, method="svd"),
        "floating_direct": dict(load_on_secondary=False, method="direct"),
        "floating_svd": dict(load_on_secondary=False, method="svd"),
    }
    results = {}
    for name, options in cases.items():
        result = run_4bus_case(label=name, **options)
        results[name] = result
        print_run(name, result)

    direct = results["healthy_direct"]
    svd = results["healthy_svd"]
    if direct["converged"] and svd["converged"]:
        max_state_difference = float(np.max(np.abs(direct["x"] - svd["x"])))
        print("=" * 78)
        print(f"Healthy direct-vs-SVD max |x difference|: {max_state_difference:.6e}")
    return results


if __name__ == "__main__":
    run_all_cases()
