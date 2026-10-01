# ============================================================
# PART 5 continued — experiments/exp01_4bus.py (FINAL VERSION)
# IEEE 4-bus baseline: healthy case + singular-case reproduction
# ============================================================
"""
exp01_4bus.py — IEEE 4-bus baseline experiment (Member A ownership).

Per project spec section 14 / plan.md section 4: this is the first
hard gate. Two cases are built from the SAME transformer data:

  Transformer between BUS2 and BUS3:
    Yg-Delta configuration
    6 MVA, 12.47/4.16 kV
    R = 0.01 p.u., X = 0.06 p.u.
  (all values explicitly given in project spec section 14 -- nothing
   invented for the transformer itself)

  CASE A (healthy)  : load present on the Delta secondary side
                       (BUS3, BUS4) -> should converge normally.
  CASE B (singular) : NO load anywhere on the Delta secondary side
                       -> per project spec section 3, this is the
                       configuration the paper reports as producing
                       enormous condition numbers; classical NR
                       (method="direct") should fail here.

IMPORTANT, method-agnostic note for the whole team: the topology
(BUS1-BUS2 line, BUS3-BUS4 line, exact loading values other than the
Delta-secondary on/off switch) is OUR reconstruction, not paper data
-- documented as such below. Only the transformer's own R, X, MVA,
kV, and connection type are paper-specified.
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.model.state import NetworkState, BusPhase, PHASES
from src.model.ybus import (
    stamp_line_series_admittance, stamp_transformer, build_dense_ybus
)
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson


def build_4bus_system(load_on_secondary: bool):
    """
    Build the IEEE 4-bus test system.

    Topology: BUS1(slack) --line-- BUS2 --[Yg-Delta xfmr]-- BUS3
              --line-- BUS4

    Parameters
    ----------
    load_on_secondary : bool
        True  -> CASE A: load on BUS3 and BUS4 (Delta side).
        False -> CASE B: zero load on BUS3 and BUS4 (Delta side),
                 reproducing the paper's reported failure config.
    """
    state = NetworkState()

    # BUS1: slack, balanced 1.0 pu (feeder head, HV/12.47kV side)
    V_slack = {
        "a": 1.0 + 0j,
        "b": 1.0 * np.exp(1j * -2 * np.pi / 3),
        "c": 1.0 * np.exp(1j * 2 * np.pi / 3),
    }
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name="BUS1", phase=p, active=True,
            is_slack=True, V_slack=V_slack[p]
        ))

    # BUS2: PQ, HV side, upstream of the transformer. Illustrative
    # local loading (reconstruction choice, not paper data).
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name="BUS2", phase=p, active=True,
            is_slack=False, P_spec=0.05, Q_spec=0.02
        ))

    # BUS3: PQ, LV side, Delta secondary terminal of the transformer.
    p3, q3 = (0.10, 0.04) if load_on_secondary else (0.0, 0.0)
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name="BUS3", phase=p, active=True,
            is_slack=False, P_spec=p3, Q_spec=q3
        ))

    # BUS4: PQ, downstream of BUS3, also on the Delta side.
    p4, q4 = (0.15, 0.06) if load_on_secondary else (0.0, 0.0)
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name="BUS4", phase=p, active=True,
            is_slack=False, P_spec=p4, Q_spec=q4
        ))

    state.finalize()

    Ybus_dict = {}

    # BUS1--BUS2 line (illustrative short feeder segment).
    z_line = complex(0.01, 0.03)
    Yprim_line = (1.0 / z_line) * np.eye(3)
    stamp_line_series_admittance(Ybus_dict, "BUS1", "BUS2", Yprim_line)

    # BUS2--BUS3: the transformer. THIS is paper-specified data
    # (project spec section 14): Yg-Delta, R=0.01, X=0.06 p.u.
    stamp_transformer(
        Ybus_dict,
        bus_p="BUS2", conn_p="yg",
        bus_s="BUS3", conn_s="delta",
        r_pu=0.01, x_pu=0.06,
        ground_primary_admittance=0.0,
        ground_secondary_admittance=0.0,  # Delta secondary: no extra
                                            # ground path -- the
                                            # deliberately unregularized
                                            # case under study.
    )

    # BUS3--BUS4 line (illustrative short LV feeder segment).
    z_line2 = complex(0.02, 0.02)
    Yprim_line2 = (1.0 / z_line2) * np.eye(3)
    stamp_line_series_admittance(Ybus_dict, "BUS3", "BUS4", Yprim_line2)

    Ybus = build_dense_ybus(Ybus_dict, state)
    return state, Ybus


# Reproduction criterion (documented, project-defined -- see NOTE below):
# a case counts as "reproducing the paper's reported Yg-Delta/no-secondary
# -load singularity" if EITHER (a) the direct solver formally refuses to
# converge (the ill_conditioned/exact_singular guard in solvers/direct.py
# fires), OR (b) kappa_2(J) exceeds this threshold at ANY iteration along
# the way, even if ||F||_inf happens to later cross tol_F in the well-
# conditioned complementary directions before the guard fires. (b) matters
# because in this reconstruction's 4-bus topology, zero specified load on
# the floating Delta side means almost no current excites the truly-
# singular (common-mode) direction, so ||F||_inf can slip below a modest
# tol_F while that one direction is still numerically meaningless -- kappa_2
# is the honest diagnostic here, exactly per project spec section 23's
# per-iteration logging requirement, not the pass/fail flag alone.
SINGULARITY_KAPPA_THRESHOLD = 1e10


def run_case(label: str, load_on_secondary: bool):
    print("=" * 70)
    print(label)
    print("=" * 70)

    state, Ybus = build_4bus_system(load_on_secondary=load_on_secondary)
    x0 = state.init_flat_start(v_mag=1.0)

    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)
    print(f"Flat-start Jacobian: sigma_max={sv0[0]:.6e}  "
          f"sigma_min={sv0[-1]:.6e}  "
          f"kappa_2={(sv0[0]/sv0[-1] if sv0[-1] > 0 else float('inf')):.6e}")

    result = newton_raphson(
        state, Ybus, x0, method="direct",
        tol_F=1e-10, max_iter=40, compute_conditioning=True
    )

    max_kappa = max(
        (h["kappa_2"] for h in result["history"]
         if h.get("kappa_2") is not None and np.isfinite(h["kappa_2"])),
        default=0.0,
    )
    singular_observed = (not result["converged"]) or (max_kappa > SINGULARITY_KAPPA_THRESHOLD)

    print(f"Converged                 : {result['converged']}")
    print(f"Iterations run             : {result['iterations']}")
    print(f"Max kappa_2(J) observed    : {max_kappa:.6e}")
    print(f"Severe ill-conditioning observed (kappa_2 > {SINGULARITY_KAPPA_THRESHOLD:.0e}): "
          f"{max_kappa > SINGULARITY_KAPPA_THRESHOLD}")
    if not result["converged"]:
        print(f"Fail reason   : {result['fail_reason']}")

    print("Per-iteration summary:")
    header = (f"{'k':>3} {'||F||_inf':>14} {'kappa_2(J)':>16} {'sigma_min(J)':>16} "
              f"{'||dx||':>14} {'solver_ok':>10} {'failure_mode':>18}")
    print(header)
    for h in result["history"]:
        k2 = h.get("kappa_2", float("nan"))
        smin = h.get("sigma_min", float("nan"))
        sd = h.get("solver_diagnostics", {})
        ok = sd.get("success", None)
        fm = sd.get("failure_mode", None)
        print(f"{h['iteration']:>3} {h['F_inf_norm']:>14.6e} {k2:>16.6e} {smin:>16.6e} "
              f"{h['dx_norm']:>14.6e} {str(ok):>10} {str(fm):>18}")

    if result["converged"]:
        V = state.unpack(result["x"])
        print("Converged voltages (selected):")
        for bus in ["BUS1", "BUS2", "BUS3", "BUS4"]:
            for ph in PHASES:
                v = V[(bus, ph)]
                print(f"  {bus}-{ph}: |V|={abs(v):.6f} pu, "
                      f"angle={np.degrees(np.angle(v)):7.2f} deg")

    print()
    result["max_kappa"] = max_kappa
    result["singular_observed"] = singular_observed
    return result


if __name__ == "__main__":
    result_healthy = run_case(
        "CASE A -- HEALTHY: Yg-Delta transformer, load ON Delta secondary",
        load_on_secondary=True,
    )
    result_singular = run_case(
        "CASE B -- SINGULAR: Yg-Delta transformer, NO load on Delta "
        "secondary (reproduction target per project spec section 3)",
        load_on_secondary=False,
    )

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Case A (healthy, loaded secondary)   -> converged: "
          f"{result_healthy['converged']}, max kappa_2: {result_healthy['max_kappa']:.3e}")
    print(f"Case B (singular, no secondary load) -> converged: "
          f"{result_singular['converged']}, max kappa_2: {result_singular['max_kappa']:.3e}, "
          f"severe ill-conditioning observed: {result_singular['singular_observed']} "
          f"(expected: True, reproducing paper's reported failure mode)")