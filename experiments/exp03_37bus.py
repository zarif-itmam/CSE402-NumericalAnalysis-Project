"""exp03_37bus.py -- IEEE 37-node reconstruction (project spec sections 16-17).

Builds the IEEE-37 radial feeder on top of the OFFICIAL IEEE 37-node feeder
data in ``src/model/ieee37_data.py`` (itself a verbatim transcription of the
public OpenDSS ieee37.dss + IEEELineCodes.DSS files).

Unlike IEEE-13, this feeder needs almost no paper-driven modification: per
project spec section 16, the paper's description ("4.8-kV feeder, two
Delta-Delta transformers, entire feeder ungrounded... no load on the XFM
secondary") already matches the OFFICIAL topology as published -- both
SubXF and XFM1 are already Delta-Delta in the official file, and bus 775
(XFM1's secondary) already has no load anywhere in the official load list.

Per-unit system
----------------
System base: S_BASE_MVA = 2.5 (matches SubXF's own rating exactly, so its
official R/X need no rescaling).
Zone voltage bases (line-to-line, matching each zone's nominal voltage):
  230 kV zone  : SourceBus only (SubXF primary). No lines live here.
  4.8 kV zone  : every bus from 799 through the rest of the feeder trunk
                 (all lines) and XFM1's primary (709).
  0.48 kV zone : bus 775 only (XFM1 secondary, unloaded).
Line impedances are the official linecodes' ohm-per-1000ft phase-domain
matrices (already usable directly as a line's series impedance matrix),
times each line's official length (already in the matching 1000-ft unit --
see ieee37_data.py's docstring for how this was verified, not assumed),
divided by Z_BASE_4_8 = 4.8**2 / 2.5 = 9.216 ohm, then inverted (matrix
inverse, not elementwise reciprocal) for stamp_line_series_admittance.

Reconstruction decisions and assumptions (documented, not silently chosen)
----------------------------------------------------------------------------
A. Transformers: SubXF (Delta-Delta, Xhl=8%, %r=1 each winding -> total
   R=2%) and XFM1 (Delta-Delta, %r=0.045 each winding -> total R=0.09% on
   ITS OWN 500 kVA base) are stamped with the OFFICIAL published values,
   unchanged, because the paper does not give alternative R/X for this
   feeder's transformers (contrast IEEE-13's paper-specified table).
   XFM1's total R/X are rescaled onto the 2.5 MVA system base (multiply by
   2500/500 = 5): R=0.45%=0.0045 pu, X_official=1.81%*5=9.05% by default.
B. XFM1 reactance discrepancy (project spec section 17): the paper reports
   X~0.00181 pu for XFM, while the official OpenDSS model's Xhl=1.81 means
   X=0.0181 pu on XFM1's own base -- an apparent factor-of-10 difference
   that the project spec explicitly says to flag rather than silently resolve.
   build_37bus_system's ``xfm1_xhl_percent`` parameter defaults to the
   OFFICIAL sourced value (1.81); pass ``xfm1_xhl_percent=0.181`` to build
   the alternate "paper-literal" variant for comparison. Neither is
   asserted here to be the "correct" one.
C. Voltage regulators (official reg1a/reg1c, open-delta with load-drop
   compensation) and the near-zero-impedance Jumper line between 799 and
   799r are modeled as an ideal (zero-impedance) merge: bus "799r" is
   folded into "799" in src/model/ieee37_data.py's OFFICIAL_LINES. Same
   rationale as IEEE-13's regulator/switch merge (exp02_13bus.py assumption
   D): keeps the reconstruction's behavior attributable to the paper's
   actual subject (transformer-connection-driven singularity), not to a
   discrete tap-changing control loop out of scope for this project's NR
   solver comparison.
D. All loads are approximated as wye-equivalent per-phase constant-PQ
   injections (same reasoning as IEEE-13 assumption B: a single delta leg's
   kW/kvar splits evenly across its two named phases; Load.S728's total
   splits evenly across all three phases), because
   src/powerflow/mismatch.py only supports phase-to-neutral constant-PQ
   injections. Every load in this feeder is delta-connected, so this
   approximation applies universally here (there is no wye-load baseline
   to compare against, unlike IEEE-13).
E. Line shunt (charging) capacitance is not modeled, consistent with the
   project-wide policy in project spec section 11 and IEEE-13's equivalent
   assumption E.
F. Reference angle: SourceBus uses a plain balanced 1.0<(0,-120,120) degree
   reference (no effect on relative voltages or conditioning diagnostics).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.ieee37_data import OFFICIAL_LINES, OFFICIAL_LOADS, line_impedance_ohm
from src.model.state import BusPhase, NetworkState, PHASES
from src.model.ybus import build_dense_ybus, stamp_line_series_admittance, stamp_transformer
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson


S_BASE_MVA = 2.5
V_BASE_KV_4_8 = 4.8
Z_BASE_4_8 = V_BASE_KV_4_8 ** 2 / S_BASE_MVA  # ohms, 9.216

# Official transformer data (both already Delta-Delta -- see module docstring).
SUBXF_RATED_MVA = 2.5
SUBXF_R_PU = 0.02  # %r=1 on each winding -> total 2%, already on the 2.5 MVA system base.
SUBXF_X_PU = 0.08  # Xhl=8%

XFM1_RATED_MVA = 0.5
_XFM1_RESCALE = S_BASE_MVA / XFM1_RATED_MVA  # 5.0
XFM1_R_PU = (2 * 0.00045) * _XFM1_RESCALE  # %r=0.045 each winding -> 0.09% own base -> 0.0045 pu system base
XFM1_XHL_PERCENT_OFFICIAL = 1.81
XFM1_XHL_PERCENT_PAPER_LITERAL = 0.181  # project spec section 17's alternate reading; see assumption B.


def _active_bus_phases() -> dict[str, set[str]]:
    """Every bus this feeder's (all-3-phase) lines touch is a,b,c active."""
    phases_by_bus: dict[str, set[str]] = {}
    for line in OFFICIAL_LINES:
        for bus in (line["bus1"], line["bus2"]):
            phases_by_bus.setdefault(bus, set()).update(("a", "b", "c"))
    phases_by_bus["SourceBus"] = {"a", "b", "c"}
    phases_by_bus["775"] = {"a", "b", "c"}  # XFM1 secondary; unloaded, no lines.
    return phases_by_bus


def _accumulate_loads() -> dict[tuple[str, str], tuple[float, float]]:
    """Split every official load onto bus-phases, in pu (see assumption D).

    Uses the per-phase base S_BASE_MVA*1000/3, matching
    src/powerflow/mismatch.py's convention -- see exp02_13bus.py's
    _accumulate_loads for the derivation and how it was verified against
    an independent OpenDSS solve.
    """
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


# Quasi-radial variant (project spec section 18, stretch goal): the paper
# adds three lines to the radial IEEE-37 base case, connecting 718-733,
# 729-742, and 736-741, each with "both zero-sequence and positive/
# negative-sequence impedance approximately 0.01+j0.001 ohm/km". Because
# all three sequence impedances are equal, the phase-domain line is exactly
# diagonal (self impedance = Z1 = Z0, zero mutual coupling: for a
# perfectly symmetric line, phase self impedance = (Z0+2*Z1)/3 and mutual
# = (Z0-Z1)/3, both of which collapse to this simple form when Z0=Z1).
# The paper does NOT specify these three lines' lengths (project spec section
# 18 gives only the per-km impedance) -- QUASI_RADIAL_ASSUMED_LENGTH_KM is
# this reconstruction's own documented choice, not paper data: it uses
# 0.15 km (~492 ft), the approximate median length of this feeder's
# existing lines (converting OFFICIAL_LINES' 1000-ft-unit lengths to km via
# 1 length-unit = 1000 ft = 0.3048 km), as a physically reasonable
# "short cross-tie" connection given no better information exists.
QUASI_RADIAL_Z_OHM_PER_KM = complex(0.01, 0.001)
QUASI_RADIAL_ASSUMED_LENGTH_KM = 0.15
QUASI_RADIAL_EXTRA_CONNECTIONS: tuple[tuple[str, str], ...] = (
    ("718", "733"),
    ("729", "742"),
    ("736", "741"),
)


def build_37bus_system(
    load_scale: float = 1.0,
    *,
    xfm1_xhl_percent: float = XFM1_XHL_PERCENT_OFFICIAL,
    quasi_radial: bool = False,
    quasi_radial_length_km: float = QUASI_RADIAL_ASSUMED_LENGTH_KM,
):
    """Build the IEEE-37 reconstruction per this module's docstring.

    Parameters
    ----------
    load_scale : float
        Uniform multiplier on every load's P and Q. ``1.0`` is nominal.
    xfm1_xhl_percent : float
        XFM1's H-L reactance in percent (see assumption B). Defaults to the
        official sourced value (1.81); pass 0.181 for the paper-literal
        alternate reading.
    quasi_radial : bool
        If True, add the three extra lines forming the paper's quasi-radial
        variant (project spec section 18) on top of the radial base case --
        generated programmatically from this same function rather than
        maintained as a separate data module, per that section's explicit
        instruction. See QUASI_RADIAL_* module constants above for the
        (documented, non-paper-specified) length assumption.
    quasi_radial_length_km : float
        Length used for all three added lines when ``quasi_radial=True``.

    Returns
    -------
    state : NetworkState
    Ybus : numpy.ndarray, dense complex admittance matrix
    """
    state = NetworkState()
    phases_by_bus = _active_bus_phases()
    loads_pu = _accumulate_loads()

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

    for bus, phases in phases_by_bus.items():
        if bus == "SourceBus":
            continue
        for p in sorted(phases, key=PHASES.index):
            p_pu, q_pu = loads_pu.get((bus, p), (0.0, 0.0))
            state.add_bus_phase(BusPhase(
                bus_name=bus, phase=p, active=True, is_slack=False,
                P_spec=p_pu * load_scale, Q_spec=q_pu * load_scale,
            ))

    state.finalize()

    ybus_dict: dict = {}

    for line in OFFICIAL_LINES:
        z_ohm = line_impedance_ohm(line)
        z_pu = z_ohm / Z_BASE_4_8
        y_prim = np.linalg.inv(z_pu)
        stamp_line_series_admittance(ybus_dict, line["bus1"], line["bus2"], y_prim)

    if quasi_radial:
        z_ohm_extra = QUASI_RADIAL_Z_OHM_PER_KM * quasi_radial_length_km
        z_pu_extra = z_ohm_extra / Z_BASE_4_8
        y_prim_extra = (1.0 / z_pu_extra) * np.eye(3)  # diagonal: see module constants' docstring.
        for bus1, bus2 in QUASI_RADIAL_EXTRA_CONNECTIONS:
            stamp_line_series_admittance(ybus_dict, bus1, bus2, y_prim_extra)

    # SubXF: Delta(SourceBus)-Delta(799), official R/X, no extra ground path.
    stamp_transformer(
        ybus_dict,
        bus_p="SourceBus", conn_p="delta",
        bus_s="799", conn_s="delta",
        r_pu=SUBXF_R_PU, x_pu=SUBXF_X_PU,
    )

    # XFM1: Delta(709)-Delta(775), official R, X per xfm1_xhl_percent.
    xfm1_x_pu = (xfm1_xhl_percent / 100.0) * _XFM1_RESCALE
    stamp_transformer(
        ybus_dict,
        bus_p="709", conn_p="delta",
        bus_s="775", conn_s="delta",
        r_pu=XFM1_R_PU, x_pu=xfm1_x_pu,
    )

    Ybus = build_dense_ybus(ybus_dict, state)
    return state, Ybus


def run_case(label: str, method: str, *, load_scale: float = 1.0, solver_options: dict | None = None):
    print("=" * 78)
    print(label)
    print("=" * 78)

    state, Ybus = build_37bus_system(load_scale=load_scale)
    x0 = state.init_flat_start(v_mag=1.0)

    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)
    kappa0 = sv0[0] / sv0[-1] if sv0[-1] > 0 else float("inf")
    print(f"System size (unknowns)     : {state.n_unknowns}")
    print(f"Flat-start Jacobian: sigma_max={sv0[0]:.6e}  sigma_min={sv0[-1]:.6e}  kappa_2={kappa0:.6e}")

    result = newton_raphson(
        state, Ybus, x0, method=method, solver_options=solver_options,
        tol_F=1e-9, max_iter=80, compute_conditioning=True,
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
        run_case(f"IEEE 37-bus, nominal load, method={method}", method, solver_options=options)
