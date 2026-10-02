"""Reproducible Tikhonov alpha sweep on the existing 4-bus reconstruction.

The script imports Member A's 4-bus builder unchanged and changes only the
``alpha`` option passed through the existing shared Newton-Raphson Tikhonov
dispatch (``lambda_reg = alpha * sigma_max(J) ** 2``). It runs both
loaded-secondary and no-secondary-load cases over the project's suggested
sweep grid (project spec section 5.D), writes ignored local CSV/PNG
diagnostics, and prints a compact summary suitable for a stage summary.

Per project spec section 5.D, this sweep -- not a single fixed alpha -- is the
project-sanctioned way to report Tikhonov results unless a defensible fixed
regularization-strength selection procedure is developed later.

The topology and non-transformer loads are team reconstruction assumptions;
these outputs are numerical observations of that reconstruction, not an
exact reproduction of the paper's private case or a universal alpha rule.
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp01_4bus import build_4bus_system
from experiments.member_d_utils import summarise_nr_history
from src.powerflow.newton import newton_raphson


TOL_F = 1e-10
MAX_ITER = 40
# project spec section 5.D's suggested sweep grid, plus a "none" reference row
# (lambda = 0, i.e. the unregularized augmented-system limit).
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
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run_case(*, load_on_secondary: bool, alpha: float) -> dict:
    """Run the shared NR model once, varying only the Tikhonov ``alpha``.

    ``compute_conditioning`` is deliberately false: per-iteration lambda,
    sigma_max(J), and solver runtimes come from ``solve_tikhonov`` itself.
    Enabling shared conditioning would add a second SVD per iteration and
    corrupt any interpretation of end-to-end timing, matching Member B's
    Stage 04 convention for the equivalent SVD-threshold sweep.
    """
    state, ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state,
        ybus,
        x0,
        method="tikhonov",
        solver_options={"alpha": alpha},
        tol_F=TOL_F,
        max_iter=MAX_ITER,
        compute_conditioning=False,
    )
    result["state"] = state
    return result


def summarise_run(case: str, label: str, alpha: float, result: dict) -> tuple[dict, list[dict]]:
    """Flatten one NR result into alpha-level and per-solve diagnostic records."""
    nr_summary = summarise_nr_history(result)
    solves = nr_summary["linear_solves"]
    last = solves[-1]["solver_diagnostics"] if solves else {}
    voltages = result["state"].unpack(result["x"])
    voltage_magnitudes = [abs(value) for value in voltages.values()]
    summary = {
        "case": case,
        "alpha_label": label,
        "requested_alpha": alpha,
        "converged": result["converged"],
        "fail_reason": result["fail_reason"] or "",
        "history_entries": nr_summary["history_entries"],
        "successful_linear_updates": nr_summary["successful_linear_updates"],
        "final_F_inf": nr_summary["final_F_inf"],
        "last_lambda_reg": last.get("lambda_reg", np.nan),
        "last_sigma_max_J": last.get("sigma_max_J", np.nan),
        "last_step_norm": last.get("step_norm", np.nan),
        "last_linear_residual_norm": last.get("residual_norm", np.nan),
        "last_augmented_residual_norm": last.get("augmented_residual_norm", np.nan),
        "last_tikhonov_runtime_sec": last.get("runtime_sec", np.nan),
        "total_tikhonov_runtime_sec": sum(
            entry["solver_diagnostics"].get("runtime_sec", 0.0) for entry in solves
        ),
        "voltage_magnitude_min": min(voltage_magnitudes),
        "voltage_magnitude_max": max(voltage_magnitudes),
    }
    trace = []
    for entry in solves:
        diagnostics = entry["solver_diagnostics"]
        trace.append({
            "case": case,
            "alpha_label": label,
            "requested_alpha": alpha,
            "iteration": entry["iteration"],
            "F_inf": entry["F_inf_norm"],
            "lambda_reg": diagnostics["lambda_reg"],
            "sigma_max_J": diagnostics["sigma_max_J"],
            "step_norm": diagnostics["step_norm"],
            "linear_residual_norm": diagnostics["residual_norm"],
            "augmented_residual_norm": diagnostics["augmented_residual_norm"],
            "tikhonov_runtime_sec": diagnostics["runtime_sec"],
        })
    return summary, trace


def _write_csv(path: Path, rows: list[dict]) -> None:
    """Write reproducible local tabular output without adding a dependency."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_diagnostics(summary_rows: list[dict]) -> None:
    """Generate simple local step-norm/mismatch-vs-alpha figures with matplotlib."""
    floating_summary = [row for row in summary_rows if row["case"] == "floating"]
    numeric_rows = [row for row in floating_summary if row["requested_alpha"] > 0.0]
    numeric_rows.sort(key=lambda row: row["requested_alpha"])
    if not numeric_rows:
        return

    alphas = [row["requested_alpha"] for row in numeric_rows]
    step_norms = [row["last_step_norm"] for row in numeric_rows]
    mismatches = [max(row["final_F_inf"], np.finfo(float).tiny) for row in numeric_rows]
    labels = [row["alpha_label"] for row in numeric_rows]

    plt.figure(figsize=(6, 4))
    plt.loglog(alphas, step_norms, marker="o")
    for x, y, label in zip(alphas, step_norms, labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
    plt.xlabel("alpha")
    plt.ylabel("Final linear-solve ||dx||")
    plt.title("Floating 4-bus: Tikhonov shrinkage vs alpha")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tikhonov_4bus_floating_step_norm_vs_alpha.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.semilogx(alphas, mismatches, marker="o")
    plt.yscale("log")
    plt.xlabel("alpha")
    plt.ylabel("Final nonlinear ||F||_inf")
    plt.title("Floating 4-bus: final mismatch vs Tikhonov alpha")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "tikhonov_4bus_floating_mismatch_vs_alpha.png", dpi=150)
    plt.close()


def run_sweep() -> tuple[list[dict], list[dict]]:
    """Execute the fixed alpha grid for healthy and floating 4-bus cases."""
    summary_rows: list[dict] = []
    trace_rows: list[dict] = []
    for case, load_on_secondary in (("healthy", True), ("floating", False)):
        for label, alpha in ALPHA_GRID:
            result = run_case(load_on_secondary=load_on_secondary, alpha=alpha)
            summary, trace = summarise_run(case, label, alpha, result)
            summary_rows.append(summary)
            trace_rows.extend(trace)
            print(
                f"{case:8s} alpha={label:6s} converged={str(summary['converged']):5s} "
                f"updates={summary['successful_linear_updates']:2d} "
                f"F_final={summary['final_F_inf']:.3e} "
                f"lambda={summary['last_lambda_reg']:.3e} "
                f"||dx||={summary['last_step_norm']:.3e} "
                f"||Jdx-rhs||={summary['last_linear_residual_norm']:.3e}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    _write_csv(RESULTS_DIR / "tikhonov_alpha_4bus_summary.csv", summary_rows)
    _write_csv(RESULTS_DIR / "tikhonov_alpha_4bus_trace.csv", trace_rows)
    _plot_diagnostics(summary_rows)
    print(f"Wrote {RESULTS_DIR / 'tikhonov_alpha_4bus_summary.csv'}")
    print(f"Wrote {RESULTS_DIR / 'tikhonov_alpha_4bus_trace.csv'}")
    print(f"Wrote figures under {FIGURES_DIR}")
    return summary_rows, trace_rows


if __name__ == "__main__":
    run_sweep()
