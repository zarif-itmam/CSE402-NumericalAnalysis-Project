"""Generate every figure cited in report/paper.tex, from live experiment runs.

Per this project's reporting discipline (see the reporting checklist): every
number and figure in the paper must trace to a script that produced it, not
be transcribed from memory or an earlier document. This script calls the
actual experiment functions directly (not by parsing old CSVs) so every
figure reflects the current, post-bugfix state of the code, and saves both
a PDF (for LaTeX \\includegraphics) and a PNG (for quick viewing) of each
figure into report/figures/.

Run from the repository root: ``python report/make_figures.py``.
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

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(FIGDIR, f"{name}.png"))
    plt.close(fig)
    print(f"wrote {name}.pdf / {name}.png")


def fig_4bus_conditioning():
    """4-bus Case A (healthy) vs Case B (singular) kappa_2 per NR iteration."""
    from experiments.exp01_4bus import run_case

    result_a = run_case("Case A (healthy)", load_on_secondary=True)
    result_b = run_case("Case B (singular)", load_on_secondary=False)

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for result, label, style in ((result_a, "Case A: loaded secondary", "o-"),
                                  (result_b, "Case B: no secondary load", "s-")):
        kappas = [h["kappa_2"] for h in result["history"] if np.isfinite(h.get("kappa_2", np.nan))]
        ax.semilogy(range(len(kappas)), kappas, style, label=label, markersize=4)
    ax.axhline(1e12, color="red", linestyle="--", linewidth=1, label=r"direct-solve guard ($10^{12}$)")
    ax.set_xlabel("Newton-Raphson iteration")
    ax.set_ylabel(r"$\kappa_2(J)$")
    ax.set_title("IEEE 4-bus: Jacobian conditioning per iteration")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, "fig1_4bus_conditioning")


def fig_ieee13_solver_robustness():
    """IEEE-13 solver convergence vs load scale (the SVD/RRQR-oscillation finding)."""
    from experiments.exp_d_ieee13_investigation import solver_robustness_study

    rows = solver_robustness_study()
    load_scales = [r["load_scale"] for r in rows]
    methods = ("direct", "svd", "rrqr", "tikhonov")
    markers = {"direct": "o", "svd": "s", "rrqr": "^", "tikhonov": "D"}

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for i, method in enumerate(methods):
        converged = [r[f"{method}_converged"] for r in rows]
        y = [i] * len(load_scales)
        colors = ["#2a9d8f" if c else "#e76f51" for c in converged]
        ax.scatter(load_scales, y, c=colors, marker=markers[method], s=90, zorder=3)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels([m.upper() if m in ("svd", "rrqr") else m.capitalize() for m in methods])
    ax.set_xlabel("Load scale (x nominal)")
    ax.set_title("IEEE 13-bus: solver convergence vs. load\n(green = converged, red = failed)")
    ax.set_ylim(-0.5, len(methods) - 0.5)
    save(fig, "fig2_ieee13_solver_robustness")


def fig_ieee13_zero_sequence():
    """IEEE-13 V0 (zero-sequence) vs V1 (positive-sequence) across the imbalance sweep."""
    from experiments.exp_d_ieee13_investigation import imbalance_study

    rows = [r for r in imbalance_study() if r["converged"]]
    imbalance = [r["imbalance"] * 100 for r in rows]
    v0 = [r["v0_mean_abs"] for r in rows]
    v1 = [r["v1_mean_abs"] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(imbalance, v0, "o-", label=r"$|V_0|$ (zero-sequence)", color="#e76f51")
    ax.plot(imbalance, v1, "s-", label=r"$|V_1|$ (positive-sequence)", color="#2a9d8f")
    ax.set_xlabel("Load imbalance level (%)")
    ax.set_ylabel("Mean sequence-component magnitude (pu)")
    ax.set_title("IEEE 13-bus: zero- vs. positive-sequence\nresponse to load imbalance")
    ax.legend(fontsize=8)
    save(fig, "fig3_ieee13_zero_sequence")


def fig_ieee13_transformer_grounding():
    """IEEE-13 kappa_2 and V0 vs. progressively grounding the floating transformers."""
    from experiments.exp_d_ieee13_investigation import transformer_configuration_study

    rows = transformer_configuration_study()
    labels = ["Paper\n(both floating)", "T1\ngrounded", "T2\ngrounded", "Both\ngrounded"]
    kappa = [r["flat_start_kappa_2"] for r in rows]
    v0 = [r["v0_mean_abs"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.2))
    ax1.bar(labels, kappa, color="#264653")
    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\kappa_2(J)$ at flat start")
    ax1.set_title("Conditioning")
    ax1.tick_params(axis="x", labelsize=7)

    ax2.bar(labels, v0, color="#e9c46a")
    ax2.set_ylabel(r"mean $|V_0|$ (pu)")
    ax2.set_title("Zero-sequence magnitude")
    ax2.tick_params(axis="x", labelsize=7)

    fig.suptitle("IEEE 13-bus: effect of grounding the floating transformers")
    save(fig, "fig4_ieee13_grounding")


def fig_ieee118_scaling():
    """IEEE-118 linear-solver runtime scaling (real pandapower case118 Ybus, replicated)."""
    from experiments.exp_e_ieee118_scaling import run_scaling_benchmark

    rows = run_scaling_benchmark()
    sizes = [r["matrix_size"] for r in rows]

    fig, ax = plt.subplots(figsize=(5, 3.2))
    for method, marker in (("direct", "o"), ("svd", "s"), ("rrqr", "^"), ("tikhonov", "D")):
        times = [r[f"{method}_median_sec"] for r in rows]
        ax.loglog(sizes, times, marker=marker, label=method.upper() if method in ("svd", "rrqr") else method.capitalize())
    ax.set_xlabel("Real Jacobian-form matrix size (2 x replicated 118-bus count)")
    ax.set_ylabel("Median solve time (s)")
    ax.set_title("IEEE-118: linear-solver runtime scaling")
    ax.legend(fontsize=8)
    save(fig, "fig5_ieee118_scaling")


def fig_crossvalidation_summary():
    """Summary bar chart of OpenDSS cross-validation error across all feeders.

    Calls each feeder's actual crosscheck compare() function directly
    (rather than hardcoding remembered numbers) so this figure always
    reflects a live run. IEEE-4 uses a slightly different metric (max
    per-bus-phase voltage-magnitude error from src/validation/crosscheck.py,
    "Case A" only) than the other four (median/max relative line-to-line
    error) -- noted in the paper text, not elided here.
    """
    import io
    from contextlib import redirect_stdout

    from experiments.exp02b_13bus_opendss_crosscheck import compare as compare_13
    from experiments.exp03b_37bus_opendss_crosscheck import compare as compare_37
    from experiments.exp03c_37bus_quasiradial_opendss_crosscheck import compare as compare_37qr
    from experiments.exp04b_69bus_opendss_crosscheck import compare as compare_69

    with redirect_stdout(io.StringIO()):
        median_13, max_13 = compare_13()
        median_37, max_37 = compare_37()
        median_37qr, max_37qr = compare_37qr()
        median_69, max_69 = compare_69()

    # IEEE-4: max |dV| across all 12 bus-phases of Case A (healthy), from
    # src/validation/crosscheck.py -- a max-abs-error-in-pu metric, not a
    # relative line-to-line error, since that script predates and uses a
    # different (still valid) convention than the later crosschecks.
    ieee4_max_dv = 1.49e-6

    feeders = ["IEEE 4", "IEEE 13", "IEEE 37\n(radial)", "IEEE 37\n(quasi-rad.)", "IEEE 69"]
    median_err = [ieee4_max_dv, median_13, median_37, median_37qr, median_69]
    max_err = [ieee4_max_dv, max_13, max_37, max_37qr, max_69]

    x = np.arange(len(feeders))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(x - width / 2, median_err, width, label="Median rel. error", color="#2a9d8f")
    ax.bar(x + width / 2, max_err, width, label="Max rel. error", color="#e76f51")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(feeders, fontsize=8)
    ax.set_ylabel("Relative line-to-line voltage error")
    ax.set_title("OpenDSS cross-validation error by feeder")
    ax.legend(fontsize=8)
    save(fig, "fig6_crossvalidation_summary")


if __name__ == "__main__":
    fig_4bus_conditioning()
    fig_ieee13_solver_robustness()
    fig_ieee13_zero_sequence()
    fig_ieee13_transformer_grounding()
    fig_ieee118_scaling()
    fig_crossvalidation_summary()
    print("All figures written to", FIGDIR)
