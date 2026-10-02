"""IEEE-37 cross-validation against an independent OpenDSS solve.

Same methodology as experiments/exp02b_13bus_opendss_crosscheck.py: compare
this repository's shared NR solver (on build_37bus_system()) against an
independent solve of data/raw/ieee37/ieee37_paper_reconstruction.dss, an
OpenDSS circuit built to mirror it exactly (see that .dss file's header).

Also not a pytest test module for the same reason as the IEEE-13 crosscheck
(see that module's docstring and README.md's "Known environment issues").
Run directly: ``python experiments/exp03b_37bus_opendss_crosscheck.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp03_37bus import build_37bus_system
from src.diagnostics.sequence_components import line_to_line_magnitudes
from src.powerflow.newton import newton_raphson

DSS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "ieee37", "ieee37_paper_reconstruction.dss"
)
TOL_F = 1e-9
MAX_ITER = 80
COMPARISON_BUSES = ("799", "701", "702", "703", "709", "728", "744", "734", "737")


def solve_opendss() -> dict[tuple[str, str], complex]:
    import opendssdirect as dss

    base = os.getcwd()
    dss.Text.Command("Clear")
    dss.Text.Command(f'compile "{os.path.abspath(DSS_PATH)}"')
    os.chdir(base)

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge on ieee37_paper_reconstruction.dss.")

    phase_of_node = {1: "a", 2: "b", 3: "c"}
    V: dict[tuple[str, str], complex] = {}
    for bus in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus)
        nodes = dss.Bus.Nodes()
        pu_vmag_ang = dss.Bus.puVmagAngle()
        for i, node in enumerate(nodes):
            phase = phase_of_node.get(node)
            if phase is None:
                continue
            mag, ang_deg = pu_vmag_ang[2 * i], pu_vmag_ang[2 * i + 1]
            V[(bus, phase)] = mag * np.exp(1j * np.radians(ang_deg))
    return V


def solve_python() -> dict[tuple[str, str], complex]:
    """Solve build_37bus_system() with SVD (converges cleanly here, unlike IEEE-13)."""
    state, Ybus = build_37bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state, Ybus, x0, method="svd", tol_F=TOL_F, max_iter=MAX_ITER, compute_conditioning=False,
    )
    if not result["converged"]:
        raise RuntimeError(f"Python NR did not converge: {result['fail_reason']}")
    return state.unpack(result["x"])


def compare() -> tuple[float, float]:
    V_dss = solve_opendss()
    V_py = solve_python()

    print(f"{'bus':>8} {'pair':>5} {'OpenDSS':>10} {'Python':>10} {'rel.err':>10}")
    max_rel_err = 0.0
    rel_errs = []
    for bus in COMPARISON_BUSES:
        dss_ll = line_to_line_magnitudes(
            {k: v for k, v in V_dss.items() if k[0] == bus.lower()}, bus.lower()
        )
        py_ll = line_to_line_magnitudes({k: v for k, v in V_py.items() if k[0] == bus}, bus)
        for pair in dss_ll:
            rel_err = abs(py_ll[pair] - dss_ll[pair]) / dss_ll[pair]
            rel_errs.append(rel_err)
            max_rel_err = max(max_rel_err, rel_err)
            print(f"{bus:>8} {pair:>5} {dss_ll[pair]:>10.4f} {py_ll[pair]:>10.4f} {rel_err:>10.4f}")

    print()
    print(f"Median relative line-to-line error: {float(np.median(rel_errs)):.4f}")
    print(f"Max relative line-to-line error:    {max_rel_err:.4f}")
    return float(np.median(rel_errs)), max_rel_err


if __name__ == "__main__":
    compare()
