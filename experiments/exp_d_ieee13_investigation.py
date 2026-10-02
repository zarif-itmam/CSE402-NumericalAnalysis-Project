"""Controlled IEEE-13 anomaly investigation (Member D, project spec section 25).

Runs the load-scaling, phase-imbalance, transformer-configuration, and
zero-sequence sensitivity studies the project's main novel investigation
calls for, on top of the modified IEEE-13 reconstruction in
``experiments/exp02_13bus.py``. Every run uses the shared Newton-Raphson
loop unchanged; only the network-building parameters and the linear-solver
method vary.

This script studies THIS reconstruction's own behavior -- it is not, and
does not claim to be, a reproduction of the base paper's private IEEE-13
result or its reported ~0.0208% error (project spec section 6). What it does
establish, on a feeder built directly from official topology/impedance data
with paper-specified transformer changes, is: (1) how badly the flat-start
Jacobian's conditioning responds to loading and imbalance, and (2) whether
the resulting large voltage discrepancies are concentrated in the
zero-sequence/common-mode component rather than positive-sequence or
line-to-line voltage, i.e. a first controlled test of the hypothesis in
project spec section 7.

Solver choice: SOLVE_METHOD is "tikhonov" (alpha=1e-8), not "svd"/"rrqr".
This was an empirical finding, not an arbitrary pick: at the reconstruction's
now-corrected (see git history) per-phase load scale, SVD's and RRQR's
undamped minimum-norm steps OSCILLATE rather than converge from a flat start
at nominal loading and above -- the correction they solve for at each
iteration is exact for the linearized problem but overshoots badly on this
severely ill-conditioned (2-D near-null-space) Jacobian. Tikhonov's damped
step converges cleanly up to 1.25x load; solver_robustness_study() below
reports exactly where each of the four methods stops converging, which is
itself one of this investigation's findings (see RQ1/RQ4 in
robust_nr_powerflow_6_week_plan.md section 15).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp02_13bus import build_13bus_system
from src.diagnostics.sequence_components import line_to_line_magnitudes, sequence_components_by_bus
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson


TOL_F = 1e-9
MAX_ITER = 80
SOLVE_METHOD = "tikhonov"
SOLVE_OPTIONS = {"alpha": 1e-8}  # see module docstring: SVD/RRQR oscillate at this load scale.
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def _flat_start_conditioning(state, Ybus) -> tuple[float, float, float]:
    x0 = state.init_flat_start(v_mag=1.0)
    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)
    sigma_max, sigma_min = float(sv0[0]), float(sv0[-1])
    kappa = sigma_max / sigma_min if sigma_min > 0 else float("inf")
    return sigma_max, sigma_min, kappa


def _voltage_spread_metrics(state, x) -> dict:
    V = state.unpack(x)
    phase_mags = [abs(v) for key, v in V.items() if key[0] != "SourceBus"]

    ll_mags: list[float] = []
    for bus in {key[0] for key in V if key[0] != "SourceBus"}:
        ll_mags.extend(line_to_line_magnitudes(V, bus).values())

    seq = sequence_components_by_bus(V)
    v0_mags = [abs(v0) for v0, _v1, _v2 in seq.values()]
    v1_mags = [abs(v1) for _v0, v1, _v2 in seq.values()]
    v2_mags = [abs(v2) for _v0, _v1, v2 in seq.values()]

    return {
        "phase_mag_min": min(phase_mags), "phase_mag_max": max(phase_mags),
        "phase_mag_spread": max(phase_mags) - min(phase_mags),
        "line_line_mag_min": min(ll_mags) if ll_mags else float("nan"),
        "line_line_mag_max": max(ll_mags) if ll_mags else float("nan"),
        "line_line_mag_spread": (max(ll_mags) - min(ll_mags)) if ll_mags else float("nan"),
        "v0_mean_abs": float(np.mean(v0_mags)) if v0_mags else float("nan"),
        "v0_max_abs": max(v0_mags) if v0_mags else float("nan"),
        "v1_mean_abs": float(np.mean(v1_mags)) if v1_mags else float("nan"),
        "v2_mean_abs": float(np.mean(v2_mags)) if v2_mags else float("nan"),
    }


def _run_and_summarise(label: str, state, Ybus) -> dict:
    sigma_max0, sigma_min0, kappa0 = _flat_start_conditioning(state, Ybus)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state, Ybus, x0, method=SOLVE_METHOD, solver_options=SOLVE_OPTIONS,
        tol_F=TOL_F, max_iter=MAX_ITER, compute_conditioning=True,
    )
    max_kappa = max(
        (h["kappa_2"] for h in result["history"]
         if h.get("kappa_2") is not None and np.isfinite(h["kappa_2"])),
        default=kappa0,
    )
    row = {
        "label": label,
        "converged": result["converged"],
        "iterations": result["iterations"],
        "flat_start_kappa_2": kappa0,
        "max_kappa_2": max_kappa,
        "final_F_inf": result["history"][-1]["F_inf_norm"] if result["history"] else float("nan"),
    }
    if result["converged"]:
        row.update(_voltage_spread_metrics(state, result["x"]))
    print(
        f"{label:40s} converged={str(row['converged']):5s} iters={row['iterations']:3d} "
        f"kappa0={kappa0:.3e} max_kappa={max_kappa:.3e} "
        f"F_final={row['final_F_inf']:.3e}"
    )
    if result["converged"]:
        print(
            f"{'':40s}   phase_spread={row['phase_mag_spread']:.4f}pu "
            f"line_line_spread={row['line_line_mag_spread']:.4f}pu "
            f"V0_mean={row['v0_mean_abs']:.4f}pu V1_mean={row['v1_mean_abs']:.4f}pu "
            f"V2_mean={row['v2_mean_abs']:.4f}pu"
        )
    return row


def solver_robustness_study() -> list[dict]:
    """Compare all four methods' convergence across the load-scaling grid.

    Not requested verbatim by project spec section 25, but directly answers
    RQ1/RQ4 (robustness and convergence/cost trade-offs, plan section 15)
    on this specific doubly-floating reconstruction, and explains why this
    module's other studies use Tikhonov rather than SVD/RRQR as their
    solver (see module docstring): direct fails everywhere by construction
    (the Jacobian is exactly singular even at flat start), SVD/RRQR's
    undamped minimum-norm step oscillates above a load-dependent threshold,
    and Tikhonov's damped step extends that threshold further before also
    failing at the heaviest tested load.
    """
    print("=" * 88)
    print("SOLVER ROBUSTNESS VS LOAD SCALE")
    print("=" * 88)
    rows = []
    for load_scale in (0.50, 0.75, 1.00, 1.25, 1.50):
        state, Ybus = build_13bus_system(load_scale=load_scale)
        x0 = state.init_flat_start(v_mag=1.0)
        row = {"load_scale": load_scale}
        summary = []
        for method, options in (
            ("direct", None), ("svd", None), ("rrqr", None), ("tikhonov", {"alpha": 1e-8}),
        ):
            result = newton_raphson(
                state, Ybus, x0, method=method, solver_options=options,
                tol_F=TOL_F, max_iter=MAX_ITER, compute_conditioning=False,
            )
            row[f"{method}_converged"] = result["converged"]
            row[f"{method}_iterations"] = result["iterations"]
            summary.append(f"{method}={'OK' if result['converged'] else 'FAIL'}({result['iterations']})")
        print(f"load_scale={load_scale:.2f}  " + "  ".join(summary))
        rows.append(row)
    return rows


def load_scaling_study() -> list[dict]:
    """project spec section 25.A: 0.5x-1.5x nominal loading."""
    print("=" * 88)
    print("LOAD SCALING STUDY")
    print("=" * 88)
    rows = []
    for load_scale in (0.50, 0.75, 1.00, 1.25, 1.50):
        state, Ybus = build_13bus_system(load_scale=load_scale)
        row = _run_and_summarise(f"load_scale={load_scale:.2f}", state, Ybus)
        row["load_scale"] = load_scale
        rows.append(row)
    return rows


def imbalance_study() -> list[dict]:
    """project spec section 25.B: 0/10/20/30% imbalance (this reconstruction's own rule)."""
    print("=" * 88)
    print("PHASE IMBALANCE STUDY (see IMBALANCE_PHASE_MULTIPLIER in exp02_13bus.py)")
    print("=" * 88)
    rows = []
    for imbalance in (0.00, 0.10, 0.20, 0.30):
        state, Ybus = build_13bus_system(imbalance=imbalance)
        row = _run_and_summarise(f"imbalance={imbalance:.0%}", state, Ybus)
        row["imbalance"] = imbalance
        rows.append(row)
    return rows


def transformer_configuration_study() -> list[dict]:
    """project spec section 25.C: isolate whether floating windings drive the rank deficiency."""
    print("=" * 88)
    print("TRANSFORMER CONFIGURATION STUDY")
    print("=" * 88)
    configs = [
        ("paper (T1 sec=delta, T2 pri=delta, T2 sec=delta)", dict()),
        ("T1 secondary grounded (T1 sec=yg)", dict(t1_secondary_conn="yg")),
        ("T2 fully grounded (T2 pri=yg, T2 sec=yg)", dict(t2_primary_conn="yg", t2_secondary_conn="yg")),
        ("both transformers fully grounded", dict(t1_secondary_conn="yg", t2_primary_conn="yg", t2_secondary_conn="yg")),
    ]
    rows = []
    for label, overrides in configs:
        state, Ybus = build_13bus_system(**overrides)
        row = _run_and_summarise(label, state, Ybus)
        row["config"] = label
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_all() -> dict[str, list[dict]]:
    results = {
        "solver_robustness": solver_robustness_study(),
        "load_scaling": load_scaling_study(),
        "imbalance": imbalance_study(),
        "transformer_configuration": transformer_configuration_study(),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    for name, rows in results.items():
        path = RESULTS_DIR / f"ieee13_{name}_study.csv"
        _write_csv(path, rows)
        print(f"Wrote {path}")
    return results


if __name__ == "__main__":
    run_all()
