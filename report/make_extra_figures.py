"""Extra figures for the final report (convergence traces, alpha sweeps, spectra).

Complements ``make_figures.py``. Every figure is produced by calling the real
experiment code (no transcribed numbers). Run from the repository root:

    python report/make_extra_figures.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIGDIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIGDIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
COLORS = {"direct": "#6c757d", "svd": "#e76f51", "rrqr": "#e9c46a", "tikhonov": "#2a9d8f"}
LABELS = {"direct": "Direct", "svd": "SVD", "rrqr": "RRQR", "tikhonov": r"Tikhonov ($\alpha=10^{-8}$)"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def _fhist(result):
    return [h["F_inf_norm"] for h in result["history"] if h.get("F_inf_norm") is not None]


def fig_residual_traces():
    """||F||_inf per NR iteration for each solver on IEEE-13 and IEEE-37 (nominal load)."""
    from experiments import exp02_13bus as e13
    from experiments import exp03_37bus as e37
    import io
    from contextlib import redirect_stdout

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, mod, title in ((axes[0], e13, "IEEE 13-bus"), (axes[1], e37, "IEEE 37-bus")):
        for m in ("svd", "rrqr", "tikhonov"):
            opts = {"alpha": 1e-8} if m == "tikhonov" else None
            with redirect_stdout(io.StringIO()):
                res = mod.run_case(f"{title} {m}", m, solver_options=opts)
            h = _fhist(res)
            ax.semilogy(range(len(h)), h, "-", marker={"svd": "s", "rrqr": "^", "tikhonov": "D"}[m],
                        ms=2.5, lw=1.0, color=COLORS[m], label=LABELS[m], alpha=0.9)
        ax.axhline(1e-9, color="k", ls=":", lw=0.8)
        ax.set_title(title)
        ax.set_xlabel("NR iteration")
    axes[0].set_ylabel(r"$\|F(x_k)\|_\infty$")
    axes[0].legend(fontsize=7, loc="lower left")
    save(fig, "fig7_residual_traces")


def fig_tikhonov_alpha_13bus():
    """Final residual and iteration count vs alpha on IEEE-13 (nominal load)."""
    from experiments.exp_d_ieee13_tikhonov_alpha_sweep import run_sweep

    rows = [r for r in run_sweep(write_csv=False, verbose=False) if r["alpha"] > 0]
    a = np.array([r["alpha"] for r in rows])
    f = np.array([r["final_F_inf"] for r in rows])
    ok = np.array([r["converged"] for r in rows])
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    ax.loglog(a, f, "-", color="#264653", lw=1)
    ax.scatter(a[ok], f[ok], c="#2a9d8f", s=60, marker="o", zorder=3, label="converged")
    ax.scatter(a[~ok], f[~ok], c="#e76f51", s=60, marker="x", zorder=3, label="80 iters, no convergence")
    ax.axhline(1e-9, color="k", ls=":", lw=0.8)
    ax.set_xlabel(r"Tikhonov $\alpha$")
    ax.set_ylabel(r"final $\|F\|_\infty$")
    ax.set_title("IEEE 13-bus, nominal load")
    ax.legend(fontsize=7)
    save(fig, "fig8_ieee13_alpha_sweep")


def fig_tikhonov_alpha_4bus():
    """4-bus: NR updates needed and final residual vs alpha (healthy and floating)."""
    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "results", "tikhonov_alpha_4bus_summary.csv")
    if not os.path.exists(path):
        from experiments.exp_d_tikhonov_lambda_sweep_4bus import run_sweep
        run_sweep()
    df = pd.read_csv(path)
    df = df[df["requested_alpha"] > 0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for case, mk, col in (("healthy", "o", "#264653"), ("floating", "s", "#e76f51")):
        d = df[df["case"] == case]
        ax1.semilogx(d["requested_alpha"], d["successful_linear_updates"], marker=mk, color=col, label=case, ms=4)
        ax2.loglog(d["requested_alpha"], d["final_F_inf"], marker=mk, color=col, label=case, ms=4)
    ax1.set_xlabel(r"Tikhonov $\alpha$"); ax1.set_ylabel("NR updates (cap 40)")
    ax2.set_xlabel(r"Tikhonov $\alpha$"); ax2.set_ylabel(r"final $\|F\|_\infty$")
    ax2.axhline(1e-9, color="k", ls=":", lw=0.8)
    ax1.legend(fontsize=7)
    ax1.set_title("Cost of damping"); ax2.set_title("Accuracy reached")
    save(fig, "fig9_tikhonov_alpha_4bus")


def fig_svd_threshold_4bus():
    """4-bus floating case: numerical rank retained vs SVD relative threshold."""
    import pandas as pd

    path = os.path.join(os.path.dirname(__file__), "..", "results", "svd_tolerance_4bus_summary.csv")
    if not os.path.exists(path):
        from experiments.exp_b_svd_tolerance_4bus import run_sweep
        run_sweep()
    df = pd.read_csv(path)
    df = df[df["rtol_label"] != "default"].copy()
    df["rtol"] = df["requested_rtol"].astype(float)
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    for case, mk, col in (("healthy", "o", "#264653"), ("floating", "s", "#e76f51")):
        d = df[df["case"] == case].sort_values("rtol")
        ax.semilogx(d["rtol"], d["last_numerical_rank"], marker=mk, color=col, label=case, ms=4)
    ax.set_xlabel("SVD relative threshold rtol"); ax.set_ylabel("numerical rank of $J$ (of 18)")
    ax.set_ylim(14.5, 18.5); ax.legend(fontsize=7)
    ax.set_title("SVD rank decision")
    save(fig, "fig10_svd_threshold_4bus")


def fig_singular_spectra():
    """Flat-start singular-value spectra of the reconstructed Jacobians (rank-gap picture)."""
    from experiments import exp01_4bus as e4, exp02_13bus as e13, exp03_37bus as e37, exp04_69bus as e69
    from src.powerflow.jacobian import compute_jacobian

    builders = [
        ("4-bus (Case B)", lambda: e4.build_4bus_system(load_on_secondary=False)[:2]),
        ("13-bus", lambda: e13.build_13bus_system()),
        ("37-bus", lambda: e37.build_37bus_system()),
        ("69-bus", lambda: e69.build_69bus_system()),
    ]
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    for (label, build), col in zip(builders, ("#6c757d", "#e76f51", "#e9c46a", "#2a9d8f")):
        state, ybus = build()[:2]
        x0 = state.init_flat_start(v_mag=1.0)
        sv = np.linalg.svd(compute_jacobian(x0, state, ybus), compute_uv=False)
        ax.semilogy(np.arange(1, len(sv) + 1) / len(sv), np.maximum(sv / sv[0], 1e-20), color=col, lw=1.2, label=label)
    ax.axhline(1e-6, color="k", ls=":", lw=0.8)
    ax.set_xlabel("normalised singular-value index $i/n$")
    ax.set_ylabel(r"$\sigma_i/\sigma_{\max}$")
    ax.set_ylim(1e-19, 2)
    ax.legend(fontsize=6.5, loc="lower left")
    ax.set_title("Flat-start Jacobian spectra")
    save(fig, "fig11_singular_spectra")


if __name__ == "__main__":
    fig_residual_traces()
    fig_tikhonov_alpha_13bus()
    fig_tikhonov_alpha_4bus()
    fig_svd_threshold_4bus()
    fig_singular_spectra()
    print("done")
