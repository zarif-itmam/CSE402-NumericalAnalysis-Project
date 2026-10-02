# src/validation/opendss.py
"""
OpenDSS reference-solution pipeline (Member E ownership).

Compiles a .dss feeder, solves it, and exports bus-phase voltages in
the shared reference format:
    bus, phase, V_real, V_imag, V_mag, V_angle

V_real/V_imag/V_mag are in PER-UNIT on the feeder's declared voltage
base (matching Member A's convention); V_angle is in degrees.
"""
from __future__ import annotations
from dataclasses import dataclass
import csv
import numpy as np
import opendssdirect as dss
import os


@dataclass
class OpenDSSRunResult:
    converged: bool
    iterations: int
    voltages: list[dict]   # rows in the shared reference format

# Captured at import time -- BEFORE any dss.Text.Command("compile ...") call
# has a chance to chdir the whole process out from under us. Every relative
# .dss path is resolved against this, never against os.getcwd() at call time.
_ORIGINAL_CWD = os.getcwd()

def compile_and_solve(dss_file: str, extra_commands: list[str] | None = None,
                       antifloat: str | None = None) -> OpenDSSRunResult:
    if os.path.isabs(dss_file):
        abs_path = os.path.normpath(dss_file)
    else:
        abs_path = os.path.normpath(os.path.join(_ORIGINAL_CWD, dss_file))

    if not os.path.isfile(abs_path):
        raise FileNotFoundError(f"No such .dss file: {abs_path}")

    dss.Text.Command("clear")
    dss.Text.Command(f'compile "{abs_path}"')

    for cmd in (extra_commands or []):
        dss.Text.Command(cmd)

    if antifloat == "disabled":
        dss.Text.Command("Set %PPM_Antifloat=0")

    dss.Text.Command("solve")

    converged = dss.Solution.Converged()
    iterations = dss.Solution.Iterations()
    rows = _export_bus_phase_voltages()

    # Undo OpenDSS's chdir side effect so the rest of the process
    # (and the next call to this function) isn't affected by it.
    os.chdir(_ORIGINAL_CWD)

    return OpenDSSRunResult(converged=converged, iterations=iterations, voltages=rows)

def _export_bus_phase_voltages() -> list[dict]:
    """Read per-unit bus voltages from the solved circuit, in the
    shared reference format. Phase letters use OpenDSS's 1/2/3 ->
    a/b/c mapping to match Member A's PHASES = ('a','b','c')."""
    phase_map = {1: "a", 2: "b", 3: "c"}
    rows = []

    bus_names = dss.Circuit.AllBusNames()
    for bus in bus_names:
        dss.Circuit.SetActiveBus(bus)
        v_pu = dss.Bus.puVmagAngle()   # flat list: [mag1, ang1, mag2, ang2, ...]
        node_order = dss.Bus.Nodes()   # e.g. [1, 2, 3] for a 3-phase bus

        for i, node in enumerate(node_order):
            if node not in phase_map:
                continue  # skip neutral/ground node 0
            mag = v_pu[2 * i]
            ang_deg = v_pu[2 * i + 1]
            ang_rad = np.radians(ang_deg)
            v_complex = mag * np.exp(1j * ang_rad)
            rows.append({
                "bus": bus.upper(),
                "phase": phase_map[node],
                "V_real": v_complex.real,
                "V_imag": v_complex.imag,
                "V_mag": mag,
                "V_angle": ang_deg,
            })
    return rows


def export_csv(result: OpenDSSRunResult, out_path: str) -> None:
    fieldnames = ["bus", "phase", "V_real", "V_imag", "V_mag", "V_angle"]
    # Ensure the output directory exists before writing the file
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.voltages)


if __name__ == "__main__":
    # CASE A: load on secondary (matches exp01_4bus.py load_on_secondary=True)
    res_a = compile_and_solve("data/raw/ieee4/ieee4_reconstruction.dss")
    export_csv(res_a, "results/opendss_ieee4_caseA.csv")
    print(f"CASE A converged={res_a.converged} iterations={res_a.iterations}")

    # CASE B: zero out secondary loads (matches load_on_secondary=False)
    res_b = compile_and_solve(
        "data/raw/ieee4/ieee4_reconstruction.dss",
        extra_commands=[
            "Edit Load.LOAD_BUS3 kw=0 kvar=0",
            "Edit Load.LOAD_BUS4 kw=0 kvar=0",
        ],
    )
    export_csv(res_b, "results/opendss_ieee4_caseB.csv")
    print(f"CASE B converged={res_b.converged} iterations={res_b.iterations}")