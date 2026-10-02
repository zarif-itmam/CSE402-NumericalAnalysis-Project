# src/validation/crosscheck.py
"""
Compares OpenDSS reference voltages against Member A's NR output for
the IEEE 4-bus experiment. Read-only w.r.t. Member A's code — imports
their public build/solve functions, does not modify them.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pandas as pd
from experiments.exp01_4bus import build_4bus_system
from src.powerflow.newton import newton_raphson
from src.model.state import PHASES
from src.validation.opendss import compile_and_solve, export_csv

# Tolerance is a project-defined validation criterion, not paper data.
V_MAG_TOL = 1e-3     # per-unit
V_ANGLE_TOL_DEG = 0.1



def run_nr_case(load_on_secondary: bool) -> dict:
    """Returns a dict with either converged voltages or failure diagnostics.
    Does NOT raise on non-convergence -- for CASE B that's the expected,
    reportable outcome, not an error."""
    state, Ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(state, Ybus, x0, method="direct",
                             tol_F=1e-10, max_iter=40, compute_conditioning=True)

    max_kappa = max(
        (h["kappa_2"] for h in result["history"]
         if h.get("kappa_2") is not None and np.isfinite(h["kappa_2"])),
        default=0.0,
    )

    return {
        "converged": result["converged"],
        "fail_reason": result.get("fail_reason"),
        "max_kappa": max_kappa,
        "V": state.unpack(result["x"]) if result["converged"] else None,
    }


def compare_case(label: str, load_on_secondary: bool, dss_extra_commands=None):
    nr = run_nr_case(load_on_secondary)
    dss_result = compile_and_solve(
        "data/raw/ieee4/ieee4_reconstruction.dss",
        extra_commands=dss_extra_commands,
    )

    print(f"\n=== {label} ===")

    if not nr["converged"]:
        print(f"NR solver: did NOT converge (fail_reason={nr['fail_reason']}, "
              f"max kappa_2(J)={nr['max_kappa']:.3e}) -- EXPECTED for this case, "
              f"reproducing the paper's reported singularity.")
        print(f"OpenDSS on the same topology: converged={dss_result.converged}, "
              f"iterations={dss_result.iterations}")
        if dss_result.converged:
            print("NOTE: OpenDSS converged where NR did not. This is a "
                  "qualitative finding worth writing up -- OpenDSS's default "
                  "%PPM_Antifloat admittance (see antifloat_study.py) likely "
                  "regularizes the floating-Delta singularity that the direct "
                  "NR solver correctly refuses to paper over. Not a bug in "
                  "either solver -- a difference in how each handles an "
                  "ill-conditioned/singular Jacobian.")
        return None  # no numeric pass/fail -- nothing to compare voltage-to-voltage

    # converged path: same numeric comparison as before
    nr_V = nr["V"]
    dss_rows = {(r["bus"], r["phase"]): r for r in dss_result.voltages}
    print(f"{'bus-phase':<12}{'|V| NR':>12}{'|V| DSS':>12}{'d|V|':>10}"
          f"{'ang NR':>12}{'ang DSS':>12}{'dAng':>10}{'PASS':>8}")

    all_pass = True
    for bus in ["BUS1", "BUS2", "BUS3", "BUS4"]:
        for ph in PHASES:
            v_nr = nr_V[(bus, ph)]
            key = (bus, ph)
            if key not in dss_rows:
                print(f"{bus}-{ph}: MISSING from OpenDSS export")
                all_pass = False
                continue
            v_dss = complex(dss_rows[key]["V_real"], dss_rows[key]["V_imag"])
            d_mag = abs(abs(v_nr) - abs(v_dss))
            ang_nr = np.degrees(np.angle(v_nr))
            ang_dss = np.degrees(np.angle(v_dss))
            d_ang = abs((ang_nr - ang_dss + 180) % 360 - 180)
            ok = d_mag <= V_MAG_TOL and d_ang <= V_ANGLE_TOL_DEG
            all_pass &= ok
            print(f"{bus}-{ph:<10}{abs(v_nr):>12.6f}{abs(v_dss):>12.6f}{d_mag:>10.2e}"
                  f"{ang_nr:>12.3f}{ang_dss:>12.3f}{d_ang:>10.3f}{'OK' if ok else 'FAIL':>8}")

    print(f"\n{label}: {'ALL PASS' if all_pass else 'MISMATCH DETECTED'}")
    return all_pass

if __name__ == "__main__":
    compare_case("CASE A -- healthy", load_on_secondary=True)
    compare_case(
        "CASE B -- singular (NR expected to fail/ill-condition; "
        "OpenDSS is a Gauss-Seidel/Newton-based solver with its own "
        "regularization, so exact numeric agreement is NOT expected here "
        "-- treat this as a qualitative sanity check, not a pass/fail gate)",
        load_on_secondary=False,
        dss_extra_commands=["Edit Load.LOAD_BUS3 kw=0 kvar=0", "Edit Load.LOAD_BUS4 kw=0 kvar=0"],
    )