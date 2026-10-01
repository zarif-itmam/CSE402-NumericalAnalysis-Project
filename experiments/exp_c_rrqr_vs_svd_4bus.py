"""Member-C RRQR-vs-SVD rank agreement and runtime comparison on IEEE 4-bus.

Per ``the project spec`` section 5.C and ``robust_nr_powerflow_6_week_plan.md``
section 6 (Member C deliverables), this script directly compares the two
rank-revealing linear solvers -- Member B's explicit SVD Moore-Penrose
pseudo-inverse and Member C's rank-revealing QR -- on the identical 4-bus
network, initial state, tolerance, and iteration budget used throughout the
team's existing 4-bus experiments. Only the linear-solver method differs.

This does not compare against an independent reference solution; it checks
whether the two rank-revealing methods agree with each other (rank, updates,
converged state) and reports each solver's own local solve time. It is not
an end-to-end Newton-Raphson runtime benchmark: ``compute_conditioning`` is
disabled so the shared loop's own diagnostic SVD does not distort either
solver's reported time, per Member B's Stage 04 convention.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp01_4bus import build_4bus_system
from experiments.member_c_utils import summarise_nr_history
from src.powerflow.newton import newton_raphson


TOL_F = 1e-10
MAX_ITER = 40
REPEATS = 20  # Independent re-solves per case/method for a stable timing median.


def run_case(*, load_on_secondary: bool, method: str) -> dict:
    """Run one 4-bus case once through the shared NR loop (no timing repeats)."""
    state, ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state,
        ybus,
        x0,
        method=method,
        tol_F=TOL_F,
        max_iter=MAX_ITER,
        compute_conditioning=False,
    )
    result["state"] = state
    return result


def median_solver_runtime_sec(*, load_on_secondary: bool, method: str, repeats: int) -> float:
    """Median wall-clock time of the *last* linear solve, over independent re-runs.

    The last iteration is used because it is present in every converged run
    and its Jacobian conditioning is the most extreme (closest to the
    floating-secondary singularity), making it the most informative single
    point for a direct-vs-SVD-vs-RRQR comparison.
    """
    timings = []
    for _ in range(repeats):
        result = run_case(load_on_secondary=load_on_secondary, method=method)
        solves = [entry for entry in result["history"] if "solver_diagnostics" in entry]
        if not solves:
            continue
        timings.append(solves[-1]["solver_diagnostics"]["runtime_sec"])
    return float(np.median(timings)) if timings else float("nan")


def compare_case(label: str, *, load_on_secondary: bool) -> None:
    print("=" * 78)
    print(label)
    print("=" * 78)

    svd_result = run_case(load_on_secondary=load_on_secondary, method="svd")
    rrqr_result = run_case(load_on_secondary=load_on_secondary, method="rrqr")

    svd_summary = summarise_nr_history(svd_result)
    rrqr_summary = summarise_nr_history(rrqr_result)

    print(f"SVD  converged={svd_result['converged']!s:<5} "
          f"iterations={svd_summary['history_entries']} "
          f"final||F||_inf={svd_summary['final_F_inf']:.6e}")
    print(f"RRQR converged={rrqr_result['converged']!s:<5} "
          f"iterations={rrqr_summary['history_entries']} "
          f"final||F||_inf={rrqr_summary['final_F_inf']:.6e}")

    if svd_summary["linear_solves"] and rrqr_summary["linear_solves"]:
        svd_last = svd_summary["linear_solves"][-1]["solver_diagnostics"]
        rrqr_last = rrqr_summary["linear_solves"][-1]["solver_diagnostics"]
        print(
            "Final-solve rank           : "
            f"SVD numerical_rank={svd_last['numerical_rank']}, "
            f"RRQR numerical_rank={rrqr_last['numerical_rank']} "
            f"(RRQR lapack_rank={rrqr_last['lapack_rank']})"
        )

    if svd_result["converged"] and rrqr_result["converged"]:
        max_state_difference = float(np.max(np.abs(svd_result["x"] - rrqr_result["x"])))
        print(f"SVD-vs-RRQR max |x difference| at convergence: {max_state_difference:.6e}")

    svd_time = median_solver_runtime_sec(
        load_on_secondary=load_on_secondary, method="svd", repeats=REPEATS
    )
    rrqr_time = median_solver_runtime_sec(
        load_on_secondary=load_on_secondary, method="rrqr", repeats=REPEATS
    )
    direct_time = median_solver_runtime_sec(
        load_on_secondary=load_on_secondary, method="direct", repeats=REPEATS
    )
    print(
        f"Median final-iteration solver runtime over {REPEATS} runs (s): "
        f"direct={direct_time:.6e}, SVD={svd_time:.6e}, RRQR={rrqr_time:.6e}"
    )
    print(
        "Local single-machine measurement on this reconstruction's small "
        "(18x18) Jacobian -- not a claim about relative asymptotic cost, "
        "which the 6-week plan's IEEE-118 scaling task (Member E) addresses."
    )


def run_all_cases() -> None:
    compare_case("CASE A -- healthy (loaded Delta secondary)", load_on_secondary=True)
    compare_case("CASE B -- floating (no load on Delta secondary)", load_on_secondary=False)


if __name__ == "__main__":
    run_all_cases()
