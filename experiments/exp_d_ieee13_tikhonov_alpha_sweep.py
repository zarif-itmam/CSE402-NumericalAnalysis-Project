"""Tikhonov-alpha sensitivity on the nominal-load IEEE-13 reconstruction.

The final IEEE-13 result uses ``alpha=1e-8`` in the scale-aware rule

    lambda = alpha * sigma_max(J) ** 2.

This experiment makes that choice reproducible.  It holds the feeder,
initial state, Newton--Raphson tolerance, and iteration budget fixed while
changing only ``alpha``.  The logarithmic grid is the one assigned to Member
D in the project plan, with ``alpha=0`` included as the unregularized
least-squares reference.

The sweep is a sensitivity study for this reconstruction, not a universal
parameter-selection rule.  In particular, convergence need not improve
monotonically as regularization increases: too little damping leaves the
minimum-norm-step oscillation, while too much damping can stall the nonlinear
iteration.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp02_13bus import build_13bus_system
from experiments.member_d_utils import summarise_nr_history
from src.powerflow.newton import newton_raphson


TOL_F = 1e-9
MAX_ITER = 80
ALPHA_GRID: tuple[tuple[str, float], ...] = (
    ("none", 0.0),
    ("1e-12", 1e-12),
    ("1e-10", 1e-10),
    ("1e-8", 1e-8),
    ("1e-6", 1e-6),
    ("1e-4", 1e-4),
    ("1e-2", 1e-2),
)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "results" / "ieee13_tikhonov_alpha_sweep.csv"


def run_alpha(alpha: float) -> dict:
    """Run nominal-load IEEE-13 once with one Tikhonov ``alpha`` value."""
    state, ybus = build_13bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    return newton_raphson(
        state,
        ybus,
        x0,
        method="tikhonov",
        solver_options={"alpha": alpha},
        tol_F=TOL_F,
        max_iter=MAX_ITER,
        compute_conditioning=False,
    )


def summarise_run(label: str, alpha: float, result: dict) -> dict:
    """Flatten one NR run into the values needed by the report table."""
    summary = summarise_nr_history(result)
    solves = summary["linear_solves"]
    first = solves[0]["solver_diagnostics"] if solves else {}
    last = solves[-1]["solver_diagnostics"] if solves else {}
    return {
        "alpha_label": label,
        "alpha": alpha,
        "converged": result["converged"],
        "iterations": result["iterations"],
        "final_F_inf": summary["final_F_inf"],
        "fail_reason": result["fail_reason"] or "",
        "first_lambda_reg": first.get("lambda_reg", np.nan),
        "last_lambda_reg": last.get("lambda_reg", np.nan),
        "last_step_norm": last.get("step_norm", np.nan),
        "last_linear_residual_norm": last.get("residual_norm", np.nan),
        "total_tikhonov_runtime_sec": sum(
            entry["solver_diagnostics"].get("runtime_sec", 0.0) for entry in solves
        ),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_sweep(*, write_csv: bool = True, verbose: bool = True) -> list[dict]:
    """Run the fixed alpha grid and optionally write the reproducible CSV."""
    rows = []
    for label, alpha in ALPHA_GRID:
        row = summarise_run(label, alpha, run_alpha(alpha))
        rows.append(row)
        if verbose:
            print(
                f"alpha={label:6s} converged={str(row['converged']):5s} "
                f"iterations={row['iterations']:2d} "
                f"final||F||_inf={row['final_F_inf']:.6e} "
                f"last_lambda={row['last_lambda_reg']:.6e}"
            )

    if write_csv:
        _write_csv(RESULTS_PATH, rows)
        if verbose:
            print(f"Wrote {RESULTS_PATH}")
    return rows


if __name__ == "__main__":
    run_sweep()
