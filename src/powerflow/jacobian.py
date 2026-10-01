# ============================================================
# PART 2 continued — jacobian.py
# Analytical Jacobian J(x) = dF/d(e,f)
# ============================================================
"""
jacobian.py — Analytical Jacobian for the rectangular-coordinate
mismatch function in mismatch.py (Member A ownership).

DERIVATION
------------------------------------------------
Recall (bus-phase i, active/non-slack):

    Ir_i = sum_k (G_ik e_k - B_ik f_k)
    Ii_i = sum_k (G_ik f_k + B_ik e_k)

    P_calc_i =  e_i Ir_i + f_i Ii_i
    Q_calc_i =  f_i Ir_i - e_i Ii_i

We need dP_calc_i/de_j, dP_calc_i/df_j, dQ_calc_i/de_j, dQ_calc_i/df_j
for every (i, j) pair where i is a non-slack row and j is a
non-slack column (slack columns are dropped: slack voltage is fixed,
not an unknown, so it contributes no column to J — but it DOES
still contribute to Ir_i, Ii_i, i.e. it affects F's value, just not
its derivative).

Using product rule, and noting dIr_i/de_j = G_ij, dIr_i/df_j = -B_ij,
dIi_i/de_j = B_ij, dIi_i/df_j = G_ij (from the Ir_i, Ii_i sums above):

  CASE j != i:
    dP_calc_i/de_j =  e_i * G_ij + f_i * B_ij
    dP_calc_i/df_j = -e_i * B_ij + f_i * G_ij
    dQ_calc_i/de_j =  f_i * G_ij - e_i * B_ij
    dQ_calc_i/df_j = -f_i * B_ij - e_i * G_ij

  CASE j == i:
    (extra product-rule terms because e_i, f_i also appear in the
     P_calc_i / Q_calc_i prefactors)
    dP_calc_i/de_i = Ir_i + e_i G_ii + f_i B_ii
    dP_calc_i/df_i = Ii_i - e_i B_ii + f_i G_ii
    dQ_calc_i/de_i = -Ii_i + f_i G_ii - e_i B_ii
    dQ_calc_i/df_i =  Ir_i - f_i B_ii - e_i G_ii

These reduce to the j != i formulas PLUS (Ir_i, Ii_i, -Ii_i, Ir_i)
added to the four respective diagonal-block entries — i.e. the
diagonal block is the off-diagonal formula plus a correction term.
This is implemented below as: compute the uniform off-diagonal-style
formula for ALL (i,j) pairs, then add the diagonal correction only
on the diagonal block. This keeps the code simple and reduces the
chance of a sign-transcription bug (fewer duplicated formulas).

J is validated against centered finite differences in
tests/test_jacobian.py — treat this derivation as unverified until
that test passes with E_J below tolerance.
"""

from __future__ import annotations
import numpy as np


def compute_jacobian(x: np.ndarray, state, Ybus: np.ndarray) -> np.ndarray:
    """
    Compute the analytical Jacobian J = dF/dx.

    Returns
    -------
    J : ndarray, shape (state.n_unknowns, state.n_unknowns)
        Row block i <-> (F_P_i, F_Q_i) for non-slack bus-phase i,
        Column block j <-> (e_j, f_j) for non-slack bus-phase j,
        ordered per state.unknown_index (must match mismatch.py).
    """
    V_dict = state.unpack(x)
    n_bp = state.n_busphases
    Vvec = np.zeros(n_bp, dtype=complex)
    for key, idx in state.busphase_index.items():
        Vvec[idx] = V_dict[key]

    Ivec = Ybus @ Vvec
    G = Ybus.real
    B = Ybus.imag

    n = state.n_unknowns
    J = np.zeros((n, n))

    non_slack = state.non_slack_busphases()

    for bp_i in non_slack:
        key_i = (bp_i.bus_name, bp_i.phase)
        i_bp = state.busphase_index[key_i]
        row = state.unknown_index[key_i]  # row for F_P_i; row+1 for F_Q_i

        e_i = Vvec[i_bp].real
        f_i = Vvec[i_bp].imag
        Ir_i = Ivec[i_bp].real
        Ii_i = Ivec[i_bp].imag

        for bp_j in non_slack:
            key_j = (bp_j.bus_name, bp_j.phase)
            j_bp = state.busphase_index[key_j]
            col = state.unknown_index[key_j]  # col for e_j; col+1 for f_j

            G_ij = G[i_bp, j_bp]
            B_ij = B[i_bp, j_bp]

            # Uniform (off-diagonal-style) formula, valid everywhere:
            dP_de =  e_i * G_ij + f_i * B_ij
            dP_df = -e_i * B_ij + f_i * G_ij
            dQ_de =  f_i * G_ij - e_i * B_ij
            dQ_df = -f_i * B_ij - e_i * G_ij

            if i_bp == j_bp:
                # diagonal correction terms derived above
                dP_de += Ir_i
                dP_df += Ii_i
                dQ_de += -Ii_i
                dQ_df += Ir_i

            J[row,     col]     = dP_de
            J[row,     col + 1] = dP_df
            J[row + 1, col]     = dQ_de
            J[row + 1, col + 1] = dQ_df

    return J