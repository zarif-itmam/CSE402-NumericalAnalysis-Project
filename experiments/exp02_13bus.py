"""exp02_13bus.py -- Modified IEEE 13-node reconstruction (project spec section 15).

Builds the Jang, Kim & Kim (2023) modified IEEE-13 test case on top of the
OFFICIAL IEEE 13-node feeder data in ``src/model/ieee13_data.py`` (itself a
verbatim transcription of the public OpenDSS IEEE13Nodeckt.dss file). This
module documents every place the paper's modification is applied, and every
place this reconstruction had to make an assumption the paper does not fully
specify.

Per-unit system
----------------
System base: S_BASE_MVA = 5.0 (matches T1/"Sub" transformer's own paper-
specified rating exactly -- see project spec section 15's transformer table --
so T1's paper R/X need no rescaling).
Zone voltage bases (line-to-line, matching each zone's nominal voltage):
  115 kV zone   : SourceBus only (T1 primary). No lines live in this zone.
  4.16 kV zone  : every bus from 650 through 675/684/652/611/633 (all lines).
  0.48 kV zone  : bus 634 only (T2 secondary).
Line impedances are given in the official file as phase-domain (not
sequence-domain) ohm/mile matrices already usable directly as a line's
series impedance matrix; they are converted to per-unit by dividing by the
4.16 kV zone's base impedance Zbase = kV_LL^2 / S_base = 4.16**2/5 =
3.46112 ohm, then inverted (matrix inverse, NOT elementwise reciprocal --
required because these matrices carry real mutual coupling) to get the
primitive admittance stamp_line_series_admittance expects.

Paper modifications applied here (project spec section 15)
----------------------------------------------------------
1. Distributed load removed: the official file's own Load.670a/670b/670c
   are excluded from src/model/ieee13_data.py -- its own inline comment
   identifies them as exactly "the concentrated point load of the
   distributed load on line 632 to 671", i.e. the object the paper's
   instruction refers to. This is a direct, well-justified match, not an
   arbitrary choice.
2. Both capacitor banks (official Cap1 at 675, Cap2 at 611) are omitted --
   never transcribed into ieee13_data.py in the first place.
3. T1 ("Sub" transformer): stamped Yg (primary, SourceBus) - Delta
   (secondary, 650), R=0.08 pu, X=0.10 pu, per the paper's own table
   (project spec section 15). The official file's connection (Delta primary /
   Wye secondary) and its near-zero placeholder impedance are NOT used --
   the official file's own comment says that placeholder was chosen only so
   the *unmodified* published test case starts flat at bus 650, which is
   irrelevant once T1's connection and impedance are deliberately changed
   to reproduce the paper's floating-secondary phenomenon.
4. T2 ("XFM1" transformer): stamped Delta-Delta, R=0.02 pu, X=0.011 pu on
   ITS OWN 0.5 MVA base per the paper's table, then rescaled onto the 5 MVA
   system base used here (multiply by 5/0.5 = 10): R=0.2 pu, X=0.11 pu at
   system base. The official file's Wye-Wye connection is not used.

Reconstruction assumptions NOT explicitly specified by the paper (documented
per the project spec's instruction not to silently invent such rules)
----------------------------------------------------------------
A. "Seven loads": project spec section 15 states the paper's modified case
   has seven loads. After excluding the distributed-load stand-in (point 1
   above), the official feeder still has EIGHT distinct load locations
   (671, 634, 645, 646, 692, 675, 611, 652). The paper does not say which
   additional location would need to be dropped to reach exactly seven, and
   nothing else in the project spec singles one out. Rather than arbitrarily
   delete one more real official load to force a numeric match, this
   reconstruction keeps all eight and reports the discrepancy explicitly
   here and in data/provenance.yml, per the project spec's explicit instruction
   to document rather than invent an unspecified rule.
B. Delta-connected loads (official Load.671 [3-phase delta], Load.646 and
   Load.692 [single delta legs]) are approximated as wye-equivalent
   per-phase constant-PQ injections, because src/powerflow/mismatch.py's
   shared bus-phase model only supports a phase-to-neutral constant-PQ
   injection (see mismatch.py's docstring), not true delta-branch current
   stamping. A 3-phase delta load's total kW/kvar is split evenly across
   its three phases; a single delta LEG's kW/kvar is split evenly across
   its two named phases. This is a standard simplified-load-flow
   approximation, not an attempt to reproduce delta-load current physics --
   extending the shared model to true delta loads is future work.
C. All loads are modeled as OpenDSS "Model=1" (constant P, Q) regardless of
   the official file's per-load model number (some are Model=2 or Model=5),
   because constant-PQ is the only load model src/powerflow/mismatch.py
   implements.
D. Voltage regulators (official Reg1/Reg2/Reg3, and the near-zero-impedance
   closed switch between buses 671 and 692) are modeled as ideal (merged,
   zero-impedance) connections: bus "RG60" is folded into "650", and bus
   "692" is folded into "671" during data entry in
   src/model/ieee13_data.py's OFFICIAL_LINES/OFFICIAL_LOADS. This keeps the
   reconstruction's numerical behavior attributable to the paper's actual
   subject (transformer-connection-driven singularity), not to an
   unrelated near-singular near-zero-impedance branch or a discrete
   tap-changing control loop that is out of scope for this project's NR
   solver comparison. Not paper-specified; a tractability simplification.
E. Line shunt (charging) capacitance is not modeled (most official
   linecodes in this feeder omit a Cmatrix entirely, and the paper's own
   description does not mention line capacitance). This keeps the model
   consistent with the "no extra ground path besides what is explicitly
   modeled" philosophy used throughout the project (project spec section 11).
F. Reference angle: SourceBus is initialized at a plain balanced
   1.0∠(0,-120,120) degree reference, not the official file's Angle=30
   convention (used there only to match a specific published-angle
   reporting convention). This is an additive constant angle offset with
   no effect on relative voltages, angle differences, or any conditioning/
   singularity diagnostic used by this project.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.ieee13_data import (
    OFFICIAL_LINES,
    OFFICIAL_LOADS,
    line_impedance_ohm,
)
from src.model.state import BusPhase, NetworkState, PHASES
from src.model.ybus import build_dense_ybus, stamp_line_series_admittance, stamp_transformer
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson


S_BASE_MVA = 5.0
V_BASE_KV_115 = 115.0
V_BASE_KV_4_16 = 4.16
V_BASE_KV_0_48 = 0.48

Z_BASE_4_16 = V_BASE_KV_4_16 ** 2 / S_BASE_MVA  # ohms, 3.46112

# Paper-specified transformer data (project spec section 15's table).
T1_RATED_MVA = 5.0
T1_R_PU = 0.08
T1_X_PU = 0.10  # already on the 5 MVA system base (T1 rated == system base)

T2_RATED_MVA = 0.5
T2_R_PU_OWN_BASE = 0.02
T2_X_PU_OWN_BASE = 0.011
_T2_RESCALE = S_BASE_MVA / T2_RATED_MVA  # 10.0
T2_R_PU = T2_R_PU_OWN_BASE * _T2_RESCALE  # 0.2, on the 5 MVA system base
T2_X_PU = T2_X_PU_OWN_BASE * _T2_RESCALE  # 0.11


def _accumulate_loads() -> dict[tuple[str, str], tuple[float, float]]:
    """Split every official (non-distributed) load onto bus-phases, in pu.

    Delta-connected loads (a 3-phase object or a single delta leg) are
    approximated as an even wye-equivalent split across their named phases
    -- see module docstring, assumption B. Returns a dict accumulating P/Q
    per (bus, phase), since "692" loads land on the same merged bus as
    "671" loads (assumption D) and must be summed, not overwritten.
    """
    # Per-phase (per-bus-phase) power must be normalized against the
    # PER-PHASE base S_BASE_MVA/3, not the full 3-phase base -- confirmed
    # empirically (see git history / commit message) by cross-validating
    # against OpenDSS on an isolated T1-only circuit: using S_BASE_MVA
    # directly produced a per-phase voltage drop ~3x too small (a plain
    # kW/S_base3ph gives one-third of the correct per-unit power). This
    # matches src/powerflow/mismatch.py's actual convention, reverse-
    # engineered from Member E's already-validated ieee4_reconstruction.dss
    # (data/raw/ieee4/ieee4_reconstruction.dss): its "kW = P_spec_pu *
    # S_base" for an OpenDSS Phases=3 load object sets the object's TOTAL
    # (not per-phase) kW, which OpenDSS then splits evenly across 3 phases
    # -- i.e. physical per-phase kW = P_spec_pu * S_base / 3, the same
    # relationship applied here in reverse.
    per_phase_base_kva = S_BASE_MVA * 1000.0 / 3.0
    totals: dict[tuple[str, str], list[float]] = {}
    for load in OFFICIAL_LOADS:
        n_legs = len(load["phases"])
        p_pu_per_leg = (load["kw"] / n_legs) / per_phase_base_kva
        q_pu_per_leg = (load["kvar"] / n_legs) / per_phase_base_kva
        for phase in load["phases"]:
            key = (load["bus"], phase)
            entry = totals.setdefault(key, [0.0, 0.0])
            entry[0] += p_pu_per_leg
            entry[1] += q_pu_per_leg
    return {key: (p, q) for key, (p, q) in totals.items()}


def _active_bus_phases() -> dict[str, set[str]]:
    """Determine each bus's physically-present phases from the line list.

    Topology (which phases exist at a node), not load assignment, is what
    determines a bus's active-phase set -- e.g. bus 680 has no load but is
    still a real 3-phase node because a 3-phase line terminates there.
    """
    phases_by_bus: dict[str, set[str]] = {}
    for line in OFFICIAL_LINES:
        for bus in (line["bus1"], line["bus2"]):
            phases_by_bus.setdefault(bus, set()).update(line["phases"])
    # SourceBus and 634 only appear via the transformers, not in
    # OFFICIAL_LINES, so they are added explicitly (both are full 3-phase).
    phases_by_bus["SourceBus"] = {"a", "b", "c"}
    phases_by_bus["634"] = {"a", "b", "c"}
    return phases_by_bus


# Per-phase imbalance multipliers for the IEEE-13 imbalance sweep
# (project spec section 25.B). The paper does not fully specify a
# machine-readable phase-by-phase imbalance rule (project spec section 15),
# so this is an explicitly documented reconstruction rule, not the paper's
# own: at imbalance level L (0.0-1.0), phase a's load is scaled by
# (1+L), phase b is left at its nominal value, and phase c is scaled by
# (1-L). This keeps total three-phase load roughly constant while
# increasing phase-to-phase asymmetry monotonically with L, which is the
# property the sweep actually needs (see project spec section 25.B's goal:
# "determine whether imbalance alone explains the anomaly", not to match
# any specific real imbalance profile).
IMBALANCE_PHASE_MULTIPLIER = {"a": lambda L: 1.0 + L, "b": lambda L: 1.0, "c": lambda L: 1.0 - L}


def build_13bus_system(
    load_scale: float = 1.0,
    imbalance: float = 0.0,
    *,
    t1_secondary_conn: str = "delta",
    t2_primary_conn: str = "delta",
    t2_secondary_conn: str = "delta",
):
    """Build the modified IEEE 13-node system per this module's docstring.

    Parameters
    ----------
    load_scale : float
        Uniform multiplier on every load's P and Q, for the load-scaling
        sensitivity study (project spec section 25.A). ``1.0`` is nominal
        loading.
    imbalance : float
        Per-phase load imbalance level in ``[0.0, 1.0)`` applied on top of
        ``load_scale`` via ``IMBALANCE_PHASE_MULTIPLIER`` above, for the
        imbalance sweep (project spec section 25.B). ``0.0`` (default) is
        balanced loading, matching the paper-specified reconstruction.
    t1_secondary_conn, t2_primary_conn, t2_secondary_conn : str
        Winding connection overrides for the transformer-configuration
        sensitivity study (project spec section 25.C: "isolate whether
        floating transformer regions create the numerical rank
        deficiency"). Defaults reproduce the paper-specified Yg-Delta (T1)
        / Delta-Delta (T2) reconstruction exactly. T1's primary is always
        "yg" (not swept: the project spec's T1 spec is explicitly Yg on the
        primary; only the floating/grounded side is of interest here).
        Passing "yg" for a normally-floating winding adds a solid ground
        reference there (``ground_*_admittance`` is left at the
        ``stamp_transformer`` default of 0.0 in every case, i.e. no
        anti-float-style regularization is added even for a "yg" override
        -- grounding comes only from the winding connection itself).

    Returns
    -------
    state : NetworkState
    Ybus : numpy.ndarray, dense complex admittance matrix
    """
    if not 0.0 <= imbalance < 1.0:
        raise ValueError(f"imbalance must be in [0.0, 1.0); received {imbalance}.")

    state = NetworkState()
    phases_by_bus = _active_bus_phases()
    loads_pu = _accumulate_loads()

    # SourceBus: slack, 115 kV zone, balanced 1.0 pu (assumption F).
    v_slack = {
        "a": 1.0 + 0j,
        "b": 1.0 * np.exp(1j * -2 * np.pi / 3),
        "c": 1.0 * np.exp(1j * 2 * np.pi / 3),
    }
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name="SourceBus", phase=p, active=True,
            is_slack=True, V_slack=v_slack[p],
        ))

    # Every other bus: PQ, with P_spec/Q_spec from the accumulated,
    # load_scale- and imbalance-scaled per-phase loads (zero for a
    # bus-phase with no load).
    for bus, phases in phases_by_bus.items():
        if bus == "SourceBus":
            continue
        for p in sorted(phases, key=PHASES.index):
            p_pu, q_pu = loads_pu.get((bus, p), (0.0, 0.0))
            scale = load_scale * IMBALANCE_PHASE_MULTIPLIER[p](imbalance)
            state.add_bus_phase(BusPhase(
                bus_name=bus, phase=p, active=True, is_slack=False,
                P_spec=p_pu * scale, Q_spec=q_pu * scale,
            ))

    state.finalize()

    ybus_dict: dict = {}

    # Lines: each official linecode's ohm/mile matrix -> ohms (x length) ->
    # per-unit (/ Z_BASE_4_16) -> primitive admittance (matrix inverse, NOT
    # elementwise reciprocal, because these carry real mutual coupling).
    for line in OFFICIAL_LINES:
        z_ohm = line_impedance_ohm(line)
        z_pu = z_ohm / Z_BASE_4_16
        y_prim = np.linalg.inv(z_pu)
        stamp_line_series_admittance(
            ybus_dict, line["bus1"], line["bus2"], y_prim,
            phases_i=line["phases"], phases_j=line["phases"],
        )

    # T1 ("Sub"): Yg (SourceBus) - Delta (650) by default, paper R/X.
    # t1_secondary_conn lets the transformer-configuration study swap the
    # secondary to "yg" to add a ground reference and observe the effect
    # on conditioning.
    stamp_transformer(
        ybus_dict,
        bus_p="SourceBus", conn_p="yg",
        bus_s="650", conn_s=t1_secondary_conn,
        r_pu=T1_R_PU, x_pu=T1_X_PU,
    )

    # T2 ("XFM1"): Delta (633) - Delta (634) by default, paper R/X
    # rescaled to the system base. t2_primary_conn/t2_secondary_conn let
    # the transformer-configuration study swap either winding to "yg".
    stamp_transformer(
        ybus_dict,
        bus_p="633", conn_p=t2_primary_conn,
        bus_s="634", conn_s=t2_secondary_conn,
        r_pu=T2_R_PU, x_pu=T2_X_PU,
    )

    Ybus = build_dense_ybus(ybus_dict, state)
    return state, Ybus


def run_case(label: str, method: str, *, load_scale: float = 1.0, solver_options: dict | None = None):
    print("=" * 78)
    print(label)
    print("=" * 78)

    state, Ybus = build_13bus_system(load_scale=load_scale)
    x0 = state.init_flat_start(v_mag=1.0)

    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)
    kappa0 = sv0[0] / sv0[-1] if sv0[-1] > 0 else float("inf")
    print(f"System size (unknowns)     : {state.n_unknowns}")
    print(f"Flat-start Jacobian: sigma_max={sv0[0]:.6e}  sigma_min={sv0[-1]:.6e}  kappa_2={kappa0:.6e}")

    result = newton_raphson(
        state, Ybus, x0, method=method, solver_options=solver_options,
        tol_F=1e-9, max_iter=60, compute_conditioning=True,
    )

    max_kappa = max(
        (h["kappa_2"] for h in result["history"]
         if h.get("kappa_2") is not None and np.isfinite(h["kappa_2"])),
        default=0.0,
    )
    print(f"Converged                 : {result['converged']}")
    print(f"Iterations run             : {result['iterations']}")
    print(f"Max kappa_2(J) observed    : {max_kappa:.6e}")
    if not result["converged"]:
        print(f"Fail reason   : {result['fail_reason']}")

    if result["converged"]:
        V = state.unpack(result["x"])
        vmags = {key: abs(v) for key, v in V.items()}
        print(f"Voltage magnitude range (pu): [{min(vmags.values()):.6f}, {max(vmags.values()):.6f}]")

    result["state"] = state
    result["max_kappa"] = max_kappa
    return result


if __name__ == "__main__":
    for method in ("direct", "svd", "rrqr", "tikhonov"):
        options = {"alpha": 1e-8} if method == "tikhonov" else None
        run_case(f"IEEE 13-bus, nominal load, method={method}", method, solver_options=options)
