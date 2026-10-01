# ============================================================
# PART 2 of 5 — mismatch.py
# Power mismatch function F(x) in rectangular coordinates
# ============================================================
"""
mismatch.py — Nonlinear power-mismatch equations for the shared
three-phase Newton-Raphson engine (Member A ownership).

FORMULATION (rectangular / (e, f) coordinates)
------------------------------------------------
For bus-phase i with complex voltage V_i = e_i + j f_i, the complex
power injection implied by the network is:

    S_i = V_i * conj(I_i),   where   I_i = sum_k Ybus[i,k] * V_k

Writing Ybus[i,k] = G_ik + j B_ik and V_k = e_k + j f_k, the
current is:

    I_i = sum_k (G_ik + jB_ik)(e_k + j f_k)
        = sum_k [ (G_ik e_k - B_ik f_k) + j (G_ik f_k + B_ik e_k) ]
        = Ir_i + j Ii_i

Calculated injected power (generation-positive convention):
    P_calc_i =  e_i * Ir_i + f_i * Ii_i
    Q_calc_i =  f_i * Ir_i - e_i * Ii_i

This is the standard identity Re/Im{V * conj(I)} expanded in
rectangular form; it is exact (no linearization) and is what NR
will drive to match the specified injection.

SIGN / LOAD CONVENTION
------------------------------------------------
BusPhase.P_spec / Q_spec are stored in LOAD convention (positive =
consumption, i.e. power flowing OUT of the network INTO the load).
Specified injection (generation convention, matching P_calc/Q_calc
above) is therefore:

    P_spec_injection = -P_spec_load
    Q_spec_injection = -Q_spec_load

(no generators are modeled at PQ buses in this baseline; if a
member adds generation later, P_spec should be net injection and
this sign flip stays local to this function.)

MISMATCH
------------------------------------------------
For every ACTIVE, NON-SLACK bus-phase i:

    F_P_i(x) = P_calc_i(x) - P_spec_injection_i
    F_Q_i(x) = Q_calc_i(x) - Q_spec_injection_i

Stacked into F(x) in the SAME order as state.unknown_index, i.e.
for each non-slack active bus-phase (in the order returned by
state.non_slack_busphases()):  [.., F_P_i, F_Q_i, ..]

This ordering must match jacobian.py exactly (row i <-> (P_i,Q_i),
column j <-> (e_j,f_j)) — enforced by both modules importing the
same state.unknown_index.
"""

from __future__ import annotations
import numpy as np


def compute_mismatch(x: np.ndarray, state, Ybus: np.ndarray) -> np.ndarray:
    """
    Compute F(x) = [ΔP_1, ΔQ_1, ΔP_2, ΔQ_2, ...] for all non-slack
    active bus-phases, ordered per state.unknown_index.

    Parameters
    ----------
    x     : current unknown vector (len = state.n_unknowns)
    state : NetworkState (finalized)
    Ybus  : dense complex admittance matrix, ordered per
            state.busphase_index (see ybus.build_dense_ybus)

    Returns
    -------
    F : ndarray, shape (state.n_unknowns,)
    """
    V_dict = state.unpack(x)

    # Build a complex voltage vector ordered per busphase_index
    # (this ordering spans slack + non-slack, matching Ybus).
    n_bp = state.n_busphases
    Vvec = np.zeros(n_bp, dtype=complex)
    for key, idx in state.busphase_index.items():
        Vvec[idx] = V_dict[key]

    Ivec = Ybus @ Vvec  # I_i = sum_k Ybus[i,k] V_k, vectorized

    F = np.zeros(state.n_unknowns)
    for bp in state.non_slack_busphases():
        key = (bp.bus_name, bp.phase)
        i_bp = state.busphase_index[key]
        i_x = state.unknown_index[key]

        e_i = Vvec[i_bp].real
        f_i = Vvec[i_bp].imag
        Ir_i = Ivec[i_bp].real
        Ii_i = Ivec[i_bp].imag

        P_calc = e_i * Ir_i + f_i * Ii_i
        Q_calc = f_i * Ir_i - e_i * Ii_i

        P_spec_injection = -bp.P_spec   # load -> injection sign flip
        Q_spec_injection = -bp.Q_spec

        F[i_x] = P_calc - P_spec_injection
        F[i_x + 1] = Q_calc - Q_spec_injection

    return F