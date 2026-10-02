"""Member-D Tikhonov comparison on Member A's existing 4-bus reconstruction.

This script deliberately imports ``build_4bus_system`` rather than restating
the network. It compares only the linear solver inside the shared
Newton--Raphson loop; state representation, mismatch, Jacobian, flat start,
stopping tolerance, iteration limit, and network data remain identical to
Member B's ``experiments/exp_b_svd_4bus.py`` and Member C's
``experiments/exp_c_rrqr_4bus.py`` so all four methods' results are directly
comparable.

The transformer data cited by the imported builder are paper-specified, but
the surrounding 4-bus topology and loads are the team's reconstruction. The
printed outcomes are therefore project results, not a claim of exact paper
reproduction.

A single fixed ``alpha`` is used here only to exercise the shared-loop
dispatch end to end; per ``the project spec`` section 5.D there is no considered
default regularization strength, so the actual sensitivity behavior is
Member D's separate ``exp_d_tikhonov_lambda_sweep_4bus.py`` script.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp01_4bus import build_4bus_system
from experiments.member_d_utils import summarise_nr_history
from src.powerflow.newton import newton_raphson


TOL_F = 1e-10
MAX_ITER = 40
DEMO_ALPHA = 1e-8


def run_4bus_case(*, label: str, load_on_secondary: bool, method: str, alpha: float | None = None) -> dict:
    """Run one existing 4-bus case through the shared NR solver interface.

    Parameters are intentionally fixed across calls so the only comparison
    variable is ``method`` (and, for Tikhonov, ``alpha``).
    ``compute_conditioning=True`` is used here to expose the shared-loop raw
    conditioning diagnostics; it is not a runtime benchmark because that
    diagnostic itself performs an SVD each iteration.
    """
    state, ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)
    solver_options = {"alpha": alpha} if method == "tikhonov" else None
    result = newton_raphson(
        state,
        ybus,
        x0,
        method=method,
        solver_options=solver_options,
        tol_F=TOL_F,
        max_iter=MAX_ITER,
        compute_conditioning=True,
    )
    result["state"] = state
    return result


def print_run(label: str, result: dict) -> None:
    """Print convergence, final state, and method-specific Tikhonov diagnostics."""
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

    if solves and solves[0]["method"] == "tikhonov":
        print("Tikhonov linear-solve diagnostics:")
        print(
            f"{'k':>3} {'alpha':>10} {'lambda':>13} {'sigma_max(J)':>13} "
            f"{'||dx||':>13} {'||Jdx-rhs||':>13} {'aug_rank':>8} {'t_solve(s)':>11}"
        )
        for entry in solves:
            diag = entry["solver_diagnostics"]
            print(
                f"{entry['iteration']:>3} {diag['alpha']:>10.3e} "
                f"{diag['lambda_reg']:>13.6e} {diag['sigma_max_J']:>13.6e} "
                f"{diag['step_norm']:>13.6e} {diag['residual_norm']:>13.6e} "
                f"{diag['augmented_rank']:>8} {diag['runtime_sec']:>11.6e}"
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
    """Run the four required direct/Tikhonov and loaded/floating comparisons."""
    cases = {
        "healthy_direct": dict(load_on_secondary=True, method="direct"),
        "healthy_tikhonov": dict(load_on_secondary=True, method="tikhonov", alpha=DEMO_ALPHA),
        "floating_direct": dict(load_on_secondary=False, method="direct"),
        "floating_tikhonov": dict(load_on_secondary=False, method="tikhonov", alpha=DEMO_ALPHA),
    }
    results = {}
    for name, options in cases.items():
        result = run_4bus_case(label=name, **options)
        results[name] = result
        print_run(name, result)

    direct = results["healthy_direct"]
    tikhonov = results["healthy_tikhonov"]
    if direct["converged"] and tikhonov["converged"]:
        max_state_difference = float(np.max(np.abs(direct["x"] - tikhonov["x"])))
        print("=" * 78)
        print(
            f"Healthy direct-vs-Tikhonov(alpha={DEMO_ALPHA:.0e}) "
            f"max |x difference|: {max_state_difference:.6e}"
        )
    return results


if __name__ == "__main__":
    run_all_cases()
