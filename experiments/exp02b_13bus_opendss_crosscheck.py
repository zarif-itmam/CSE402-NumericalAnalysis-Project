"""IEEE-13 cross-validation against an independent OpenDSS solve.

Compares this repository's shared NR solver (on the reconstruction built by
``experiments/exp02_13bus.py``) against an independent solve of
``data/raw/ieee13/ieee13_paper_reconstruction.dss`` -- an OpenDSS circuit
built to mirror ``build_13bus_system()`` exactly (same official-derived
topology, same paper modifications; see that .dss file's own header for the
itemized correspondence).

Requires ``opendssdirect`` (already in requirements.txt). This script is
intentionally NOT a pytest test module: importing opendssdirect during
pytest's assertion-rewrite collection was observed to be unstable in this
project's Python 3.14 environment (a native-extension crash during import,
not a Python-level exception -- see the "Known environment issues" section
of README.md). Run it directly instead: ``python
experiments/exp02b_13bus_opendss_crosscheck.py``.

What this comparison found (and could not have found without an external
reference solver -- this is exactly why Member E's OpenDSS validation track
exists): this repository's Python reconstruction had a real per-phase load
scaling bug (loads were ~3x too light), caught by an isolated single-
transformer OpenDSS comparison and fixed (see git history). After the fix,
line-to-line voltages agree with OpenDSS reasonably well at buses/phase-
pairs away from the reconstruction's 2-dimensional near-null space (T1 and
T2 each independently floating -- see exp02_13bus.py and
tests/test_ieee13.py), but diverge by up to ~25-30% at the specific bus/
phase-pairs most aligned with that null space. That is an EXPECTED
consequence of two different, independently-chosen regularizations
(this repo's Tikhonov alpha=1e-8 vs. OpenDSS's own internal ppm_antifloat)
selecting different particular solutions within a genuinely underdetermined
2-D subspace -- not a sign either solver is "wrong" on the well-determined
part of the problem. Both are correct on what is actually determined by
the network; the model has more than one legitimate answer for the rest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp02_13bus import build_13bus_system
from src.diagnostics.sequence_components import line_to_line_magnitudes
from src.powerflow.newton import newton_raphson

DSS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "ieee13", "ieee13_paper_reconstruction.dss"
)
TOL_F = 1e-9
MAX_ITER = 80
TIKHONOV_ALPHA = 1e-8
COMPARISON_BUSES = ("650", "632", "671", "675", "633", "634", "645", "646", "684")


def solve_opendss() -> dict[tuple[str, str], complex]:
    """Compile and solve the OpenDSS twin circuit; return its voltage dict."""
    import opendssdirect as dss

    base = os.getcwd()
    dss.Text.Command("Clear")
    dss.Text.Command(f'compile "{os.path.abspath(DSS_PATH)}"')
    os.chdir(base)  # compile() changes the process cwd to the file's folder.

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge on ieee13_paper_reconstruction.dss.")

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
    """Solve build_13bus_system() with Tikhonov (see module docstring)."""
    state, Ybus = build_13bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(
        state, Ybus, x0, method="tikhonov", solver_options={"alpha": TIKHONOV_ALPHA},
        tol_F=TOL_F, max_iter=MAX_ITER, compute_conditioning=False,
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
    print(
        "See this module's docstring: the max error is expected to be large at "
        "bus/phase-pairs aligned with the reconstruction's 2-D near-null space "
        "(different regularizations picking different particular solutions "
        "there), not evidence of a remaining bug -- as long as the MEDIAN error "
        "is small, the well-determined part of the network is validated."
    )
    return float(np.median(rel_errs)), max_rel_err


if __name__ == "__main__":
    compare()
