"""Reproducible SVD rank-threshold sweep on the existing 4-bus reconstruction.

The script imports Member A's 4-bus builder unchanged and changes only the
``rtol`` option passed through the existing shared Newton-Raphson SVD dispatch.
It runs both loaded-secondary and no-secondary-load cases, writes ignored local
CSV/PNG diagnostics, and prints a compact summary suitable for a stage summary.

The topology and non-transformer loads are team reconstruction assumptions;
these outputs are numerical observations of that reconstruction, not an exact
reproduction of the paper's private case or a universal threshold rule.
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
from experiments.member_b_utils import summarise_nr_history
from src.powerflow.newton import newton_raphson


TOL_F = 1e-10
MAX_ITER = 40
# The first entry exercises the solver's documented default: eps * max(J.shape).
RTOL_GRID: tuple[tuple[str, float | None], ...] = (
    ("default", None),
    ("1e-16", 1e-16),
    ("1e-14", 1e-14),
    ("1e-12", 1e-12),
    ("1e-10", 1e-10),
    ("1e-8", 1e-8),
    ("1e-6", 1e-6),
)
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def run_case(*, load_on_secondary: bool, rtol: float | None) -> dict:
    """Run the shared NR model once, varying only the SVD relative cutoff.

    ``compute_conditioning`` is deliberately false: per-iteration spectra,
    ranks, and solver runtimes come from ``solve_svd_pinv`` itself.  Enabling
    shared conditioning would add a second SVD per iteration and corrupt any
    interpretation of end-to-end timing.
    """
    state, ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state,
        ybus,
        x0,
        method="svd",
        solver_options={"rtol": rtol},
        tol_F=TOL_F,
        max_iter=MAX_ITER,
        compute_conditioning=False,
    )
    result["state"] = state
    return result


def summarise_run(case: str, label: str, requested_rtol: float | None, result: dict) -> tuple[dict, list[dict]]:
    """Flatten one NR result into threshold and per-solve diagnostic records."""
    nr_summary = summarise_nr_history(result)
    solves = nr_summary["linear_solves"]
    last = solves[-1]["solver_diagnostics"] if solves else {}
    voltages = result["state"].unpack(result["x"])
    voltage_magnitudes = [abs(value) for value in voltages.values()]
    summary = {
        "case": case,
        "rtol_label": label,
        "requested_rtol": "default" if requested_rtol is None else requested_rtol,
        "actual_rtol": last.get("rtol", np.nan),
        "converged": result["converged"],
        "fail_reason": result["fail_reason"] or "",
        "history_entries": nr_summary["history_entries"],
        "successful_linear_updates": nr_summary["successful_linear_updates"],
        "final_F_inf": nr_summary["final_F_inf"],
        "last_numerical_rank": last.get("numerical_rank", np.nan),
        "last_truncated_singular_values": last.get("truncated_singular_values", np.nan),
        "last_sigma_max": last.get("sigma_max", np.nan),
        "last_sigma_min": last.get("sigma_min", np.nan),
        "last_rank_tolerance": last.get("rank_tolerance", np.nan),
        "last_step_norm": last.get("step_norm", np.nan),
        "last_linear_residual_norm": last.get("residual_norm", np.nan),
        "last_svd_runtime_sec": last.get("runtime_sec", np.nan),
        "total_svd_runtime_sec": sum(
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
            "rtol_label": label,
            "requested_rtol": "default" if requested_rtol is None else requested_rtol,
            "iteration": entry["iteration"],
            "F_inf": entry["F_inf_norm"],
            "numerical_rank": diagnostics["numerical_rank"],
            "truncated_singular_values": diagnostics["truncated_singular_values"],
            "sigma_max": diagnostics["sigma_max"],
            "sigma_min": diagnostics["sigma_min"],
            "rank_tolerance": diagnostics["rank_tolerance"],
            "step_norm": diagnostics["step_norm"],
            "linear_residual_norm": diagnostics["residual_norm"],
            "svd_runtime_sec": diagnostics["runtime_sec"],
            "singular_values": ";".join(f"{value:.17g}" for value in diagnostics["singular_values"]),
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


def _plot_diagnostics(summary_rows: list[dict], trace_rows: list[dict]) -> None:
    """Generate simple local spectral/rank/convergence figures with matplotlib."""
    floating_summary = [row for row in summary_rows if row["case"] == "floating"]
    floating_trace = [row for row in trace_rows if row["case"] == "floating"]
    default_trace = [row for row in floating_trace if row["rtol_label"] == "default"]
    if default_trace:
        last = default_trace[-1]
        spectrum = np.fromstring(last["singular_values"], sep=";")
        plt.figure(figsize=(6, 4))
        plt.semilogy(np.arange(1, spectrum.size + 1), spectrum, marker="o")
        plt.axhline(last["rank_tolerance"], color="tab:red", linestyle="--", label="tau")
        plt.xlabel("Singular-value index")
        plt.ylabel("Singular value")
        plt.title("Floating 4-bus: final default-rtol SVD spectrum")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "svd_4bus_floating_default_spectrum.png", dpi=150)
        plt.close()

    numeric_rows = [row for row in floating_summary if np.isfinite(row["actual_rtol"])]
    numeric_rows.sort(key=lambda row: row["actual_rtol"])
    if numeric_rows:
        rtols = [row["actual_rtol"] for row in numeric_rows]
        ranks = [row["last_numerical_rank"] for row in numeric_rows]
        mismatches = [max(row["final_F_inf"], np.finfo(float).tiny) for row in numeric_rows]
        labels = [row["rtol_label"] for row in numeric_rows]

        plt.figure(figsize=(6, 4))
        plt.semilogx(rtols, ranks, marker="o")
        for x, y, label in zip(rtols, ranks, labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)
        plt.xlabel("Actual relative tolerance")
        plt.ylabel("Final linear-solve numerical rank")
        plt.title("Floating 4-bus: retained rank vs SVD cutoff")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "svd_4bus_floating_rank_vs_rtol.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.semilogx(rtols, mismatches, marker="o")
        plt.yscale("log")
        plt.xlabel("Actual relative tolerance")
        plt.ylabel("Final nonlinear ||F||_inf")
        plt.title("Floating 4-bus: final mismatch vs SVD cutoff")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "svd_4bus_floating_mismatch_vs_rtol.png", dpi=150)
        plt.close()


def run_sweep() -> tuple[list[dict], list[dict]]:
    """Execute the fixed threshold grid for healthy and floating 4-bus cases."""
    summary_rows: list[dict] = []
    trace_rows: list[dict] = []
    for case, load_on_secondary in (("healthy", True), ("floating", False)):
        for label, rtol in RTOL_GRID:
            result = run_case(load_on_secondary=load_on_secondary, rtol=rtol)
            summary, trace = summarise_run(case, label, rtol, result)
            summary_rows.append(summary)
            trace_rows.extend(trace)
            print(
                f"{case:8s} rtol={label:7s} converged={str(summary['converged']):5s} "
                f"updates={summary['successful_linear_updates']:2d} "
                f"F_final={summary['final_F_inf']:.3e} "
                f"rank={summary['last_numerical_rank']:2.0f} "
                f"trunc={summary['last_truncated_singular_values']:2.0f} "
                f"sigma_min={summary['last_sigma_min']:.3e} "
                f"tau={summary['last_rank_tolerance']:.3e}"
            )

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    _write_csv(RESULTS_DIR / "svd_tolerance_4bus_summary.csv", summary_rows)
    _write_csv(RESULTS_DIR / "svd_tolerance_4bus_trace.csv", trace_rows)
    _plot_diagnostics(summary_rows, trace_rows)
    print(f"Wrote {RESULTS_DIR / 'svd_tolerance_4bus_summary.csv'}")
    print(f"Wrote {RESULTS_DIR / 'svd_tolerance_4bus_trace.csv'}")
    print(f"Wrote figures under {FIGURES_DIR}")
    return summary_rows, trace_rows


if __name__ == "__main__":
    run_sweep()
