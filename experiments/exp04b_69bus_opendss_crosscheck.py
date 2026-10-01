"""IEEE-69 cross-validation against an independent OpenDSS solve.

Same methodology as the IEEE-13/37 crosschecks: compare this repository's
shared NR solver (on build_69bus_system()) against an independent solve of
data/raw/ieee69/ieee69_paper_reconstruction.dss, an OpenDSS circuit
GENERATED PROGRAMMATICALLY (data/raw/ieee69/gen_ieee69_dss.py; see
that .dss file's own header) from the exact same
src/model/ieee69_data.py the Python side uses, to avoid hand-transcription
errors across the feeder's 65 regular branches.

Before trusting the full-system comparison, this feeder's transformer
stamping (paper R/X at 3-4, applied on this reconstruction's 12.66 kV
voltage base rather than the paper's stated 13.8 kV -- see
exp04_69bus.py's assumption B) was independently verified against an
isolated, well-conditioned single-transformer OpenDSS sub-circuit (see the
commit that added this file for the exact numbers: ~1e-8 pu agreement)
before building this full comparison, per this project's established
validate-in-isolation-first discipline.

Unlike IEEE-13/37, this feeder's direct solver converges (see
tests/test_ieee69.py) -- so "svd" is used here purely for consistency with
the other crosscheck scripts, not because direct fails.

Not a pytest test module, for the same reason as the other OpenDSS
crosschecks (see README.md's "Known environment issues"). Run directly:
``python experiments/exp04b_69bus_opendss_crosscheck.py``.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from experiments.exp04_69bus import build_69bus_system
from src.diagnostics.sequence_components import line_to_line_magnitudes
from src.powerflow.newton import newton_raphson

DSS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "raw", "ieee69", "ieee69_paper_reconstruction.dss"
)
TOL_F = 1e-9
MAX_ITER = 80
# BUS1 is slack; BUS4/28/36 are the transformer secondaries (start of each
# floating region); a spread of downstream/trunk buses for coverage.
COMPARISON_BUSES = ("BUS3", "BUS4", "BUS28", "BUS36", "BUS50", "BUS61", "BUS65", "BUS27", "BUS46")


def solve_opendss() -> dict[tuple[str, str], complex]:
    import opendssdirect as dss

    base = os.getcwd()
    dss.Text.Command("Clear")
    dss.Text.Command(f'compile "{os.path.abspath(DSS_PATH)}"')
    os.chdir(base)

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge on ieee69_paper_reconstruction.dss.")

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
    state, Ybus = build_69bus_system()
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
            print(f"{bus:>8} {pair:>5} {dss_ll[pair]:>10.4f} {py_ll[pair]:>10.4f} {rel_err:>10.2e}")

    print()
    print(f"Median relative line-to-line error: {float(np.median(rel_errs)):.3e}")
    print(f"Max relative line-to-line error:    {max_rel_err:.3e}")
    return float(np.median(rel_errs)), max_rel_err


if __name__ == "__main__":
    compare()
