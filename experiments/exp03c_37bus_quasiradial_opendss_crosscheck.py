"""IEEE-37 quasi-radial cross-validation against an independent OpenDSS solve.

Same methodology as experiments/exp03b_37bus_opendss_crosscheck.py, applied
to the quasi-radial variant (project spec section 18):
build_37bus_system(quasi_radial=True) vs. an independent solve of
data/raw/ieee37/ieee37_quasiradial_reconstruction.dss (the radial
reconstruction plus the same three added lines, generated the same way the
Python side generates them -- see that .dss file's header).

Before trusting a full-system comparison here, the quasi-radial lines'
non-paper-specified 0.15 km length AND their ohm/km (not ohm/1000ft, unlike
every other line in this feeder) unit convention were independently
verified against an isolated single-line, well-conditioned OpenDSS
sub-circuit (see the commit that added this file for the exact numbers) --
matching the project's established validate-in-isolation-first discipline,
since the full quasi-radial system is severely ill-conditioned and a wrong
per-unit conversion could otherwise hide behind that ill-conditioning the
way the IEEE-13 load-scaling bug briefly did.

Not a pytest test module, for the same reason as the other OpenDSS
crosschecks (see README.md's "Known environment issues"). Run directly:
``python experiments/exp03c_37bus_quasiradial_opendss_crosscheck.py``.
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
    os.path.dirname(__file__), "..", "data", "raw", "ieee37", "ieee37_quasiradial_reconstruction.dss"
)
TOL_F = 1e-9
MAX_ITER = 80
# Includes the six buses newly connected by the quasi-radial lines, plus
# the same trunk buses used by the radial crosscheck for continuity.
COMPARISON_BUSES = (
    "799", "701", "702", "709", "728", "744",
    "718", "733", "729", "742", "736", "741",
)


def solve_opendss() -> dict[tuple[str, str], complex]:
    import opendssdirect as dss

    base = os.getcwd()
    dss.Text.Command("Clear")
    dss.Text.Command(f'compile "{os.path.abspath(DSS_PATH)}"')
    os.chdir(base)

    if not dss.Solution.Converged():
        raise RuntimeError("OpenDSS did not converge on ieee37_quasiradial_reconstruction.dss.")

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
    """Solve build_37bus_system(quasi_radial=True). Direct-guard 1e12 still applies."""
    state, Ybus = build_37bus_system(quasi_radial=True)
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
