"""exp04_69bus.py -- Modified IEEE/PG&E 69-bus reconstruction (project spec section 19).

Builds a balanced three-phase model on top of the official MATPOWER
case69.m data in ``src/model/ieee69_data.py``, then applies Jang, Kim &
Kim (2023)'s modification: three Yg-Delta transformers at 3-4, 3-28, and
3-36 (project spec section 19's table). Per that section, this is
LOWER PRIORITY than IEEE-4/13/37 and explicitly a reconstruction, not an
exact reproduction: "the public case is balanced and the paper does not
completely document the phase-specific 30%-imbalance construction."

Unlike IEEE-13/37, case69 has no native phase-domain data at all (it is a
single-phase-equivalent / positive-sequence representation, standard for
transmission-style MATPOWER cases). This reconstruction treats it as a
BALANCED three-phase feeder: every branch is stamped as a diagonal
(zero-mutual-coupling) 3-phase line with the SAME impedance on each phase,
and every load is split identically across the three phases -- there is no
attempt here to reproduce the paper's ~30%-unbalanced variant, because (per
project spec section 19) that phase-specific construction rule is not
documented anywhere available to this project. Only the balanced base case
is built.

Reconstruction decisions (documented, not silently chosen)
------------------------------------------------------------
A. "Adding three Yg-Delta transformers AT 3-4, 3-28, 3-36": the official
   case69 topology ALREADY has plain lines at exactly these three
   locations (see BRANCHES_OHM). The paper's transformer table gives a
   same-to-same voltage rating (13.8/13.8 kV) for each, consistent with
   REPLACING each of these three specific branches with the specified
   Yg-Delta transformer at that same connection -- not adding a new,
   separate parallel branch. This reconstruction adopts "replace" as its
   reading; the paper's own wording does not unambiguously state either
   way.
B. Voltage-base discrepancy: the paper's transformer table gives 13.8 kV,
   but case69's actual bus voltage base is 12.66 kV (project spec section 19
   / src/model/ieee69_data.py's BASE_KV). This reconstruction uses 12.66 kV
   throughout (the network's actual operating voltage) and applies the
   paper's R/X per-unit values directly on the transformer's own 10 MVA
   rating (which happens to equal this reconstruction's chosen system
   base, so no MVA rescale is needed) -- the kV mismatch is flagged here
   and in data/provenance.yml as an unresolved discrepancy in the paper's
   own published data, not something this project silently corrects.
C. System base: S_BASE_MVA = 10.0, matching both case69's own baseMVA and
   the transformers' own rating exactly.
D. Balanced-only: no 30%-imbalance variant is built (see module docstring
   above) -- this is an explicitly documented gap, matching the project spec
   section 19's own statement that the construction rule is undocumented.
E. Reference angle: SourceBus (bus 1, the network's slack) uses a plain
   balanced 1.0<(0,-120,120) degree reference.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.ieee69_data import BASE_KV, BASE_MVA, BRANCHES_OHM, BUS_LOADS_MW, TRANSFORMER_LOCATIONS
from src.model.state import BusPhase, NetworkState, PHASES
from src.model.ybus import build_dense_ybus, stamp_line_series_admittance, stamp_transformer
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson


S_BASE_MVA = BASE_MVA  # 10.0, matches both case69's own base and the transformers' rating.
Z_BASE_OHM = BASE_KV ** 2 / S_BASE_MVA  # ohms, 12.66**2/10 = 16.0276

SLACK_BUS = 1
TRANSFORMER_BUS_PAIRS = set(TRANSFORMER_LOCATIONS)


def _bus_name(i: int) -> str:
    return f"BUS{i}"


def _regular_branches() -> list[tuple[int, int, float, float]]:
    """All case69 branches except the three replaced by transformers."""
    return [
        (f, t, r, x) for (f, t, r, x) in BRANCHES_OHM
        if (f, t) not in TRANSFORMER_BUS_PAIRS and (t, f) not in TRANSFORMER_BUS_PAIRS
    ]


def build_69bus_system(load_scale: float = 1.0):
    """Build the modified IEEE-69 balanced three-phase system.

    Parameters
    ----------
    load_scale : float
        Uniform multiplier on every load's P and Q. ``1.0`` is nominal.

    Returns
    -------
    state : NetworkState
    Ybus : numpy.ndarray, dense complex admittance matrix
    """
    state = NetworkState()

    v_slack = {
        "a": 1.0 + 0j,
        "b": 1.0 * np.exp(1j * -2 * np.pi / 3),
        "c": 1.0 * np.exp(1j * 2 * np.pi / 3),
    }
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name=_bus_name(SLACK_BUS), phase=p, active=True,
            is_slack=True, V_slack=v_slack[p],
        ))

    # BUG FIX (see git history): BUS_LOADS_MW's Pd/Qd are the case69 bus's
    # TOTAL three-phase load (standard MATPOWER convention), NOT a
    # per-phase value -- an earlier version of this function divided the
    # total directly by the per-phase base (S_BASE_MVA*1000/3), giving a
    # per-unit load 3x too large (identical in effect to the earlier
    # IEEE-13 per-phase-load bug, though this was a different, independent
    # mistake in a different reconstruction). The correct conversion first
    # splits the total evenly across 3 phases (matching
    # experiments/exp02_13bus.py's and exp03_37bus.py's
    # _accumulate_loads() pattern: per-phase actual kW / (S_base_kVA/3)),
    # which is algebraically just total_kW / S_base_kVA -- written with
    # the explicit /3.0 here for consistency with that established
    # convention rather than the simplified form, so the pattern is
    # recognizable across all three reconstructions. Caught via an
    # isolated single-branch/single-load OpenDSS cross-check at bus 61
    # (this feeder's single largest load, ~33% voltage error before this
    # fix, ~0.1% after) -- see
    # experiments/exp04b_69bus_opendss_crosscheck.py.
    per_phase_base_kva = S_BASE_MVA * 1000.0 / 3.0
    loads_by_bus = {i: (pd, qd) for i, pd, qd in BUS_LOADS_MW}
    for i in range(1, 70):
        if i == SLACK_BUS:
            continue
        pd_mw, qd_mw = loads_by_bus.get(i, (0.0, 0.0))
        p_pu = (pd_mw * 1000.0 / 3.0) / per_phase_base_kva * load_scale
        q_pu = (qd_mw * 1000.0 / 3.0) / per_phase_base_kva * load_scale
        for p in PHASES:
            state.add_bus_phase(BusPhase(
                bus_name=_bus_name(i), phase=p, active=True, is_slack=False,
                P_spec=p_pu, Q_spec=q_pu,
            ))

    state.finalize()

    ybus_dict: dict = {}

    for fbus, tbus, r_ohm, x_ohm in _regular_branches():
        z_pu = complex(r_ohm, x_ohm) / Z_BASE_OHM
        y_prim = (1.0 / z_pu) * np.eye(3)  # balanced: diagonal, no mutual coupling (see docstring).
        stamp_line_series_admittance(ybus_dict, _bus_name(fbus), _bus_name(tbus), y_prim)

    for (fbus, tbus), xfmr in TRANSFORMER_LOCATIONS.items():
        stamp_transformer(
            ybus_dict,
            bus_p=_bus_name(fbus), conn_p="yg",
            bus_s=_bus_name(tbus), conn_s="delta",
            r_pu=xfmr["r_pu"], x_pu=xfmr["x_pu"],
        )

    Ybus = build_dense_ybus(ybus_dict, state)
    return state, Ybus


def run_case(label: str, method: str, *, load_scale: float = 1.0, solver_options: dict | None = None):
    print("=" * 78)
    print(label)
    print("=" * 78)

    state, Ybus = build_69bus_system(load_scale=load_scale)
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
        run_case(f"IEEE 69-bus, nominal load, method={method}", method, solver_options=options)
