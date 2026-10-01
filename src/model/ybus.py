# ============================================================
# PART 1 continued — ybus.py
# Ybus assembly: lines + Yg-Delta / Delta-Delta transformer stamping
# ============================================================
"""
ybus.py — Admittance matrix assembly for the shared three-phase model.

IMPORTANT (per project spec section 4 / 15 / 21): this module only
BUILDS Ybus as data. It never inverts it and is not itself a solver.
The Jacobian/mismatch modules consume this Ybus.

Two stamping primitives are provided:
  1. stamp_line_series_admittance  — ordinary series+shunt branch,
     phase-coupled via a 3x3 (or smaller) primitive admittance
     matrix, for a normal untransformed line.
  2. stamp_transformer              — the transformer model needed to
     reproduce the paper's singular case. A Yg-Delta transformer's
     Delta winding has NO physical neutral/ground path, i.e. the
     zero-sequence circuit is open on the Delta side. We model this
     explicitly: current can only enter/leave the Delta winding
     phase-to-phase, never phase-to-ground. This is what produces a
     near/exactly singular Jacobian when the Delta side is
     otherwise unreferenced (no other ground path on that side).
     Only "yg" (grounded wye) and "delta" windings are implemented;
     "y" (ungrounded wye) currently raises NotImplementedError -- see
     _winding_connection_matrix for why it needs more than a swapped
     connection matrix.

We use the standard two-winding transformer admittance model in
per-unit, referred to a common per-unit system, with connection
matrices C_primary, C_secondary mapping winding currents to
terminal currents:

  Yg (grounded wye) winding: C = Identity (each phase has its own
      ground-referenced terminal).
  Delta winding: C is the standard 3x3 delta connection matrix
      (each winding is connected phase-to-phase, e.g. winding 'ab'
      spans terminals a and b), so terminal currents are linear
      combinations of winding currents with NO row/column that maps
      to a ground reference. This is exactly what makes the Delta
      side's 3x3 sub-block of Ybus singular (rank 2, one dependent
      row) when it has no other ground reference of its own.
"""

from __future__ import annotations
import numpy as np
from .state import PHASES

# Delta connection matrix (standard): converts winding-current basis
# to terminal-current basis for a delta winding with winding order
# ab, bc, ca. Row i = terminal, col j = winding.
#   i_a =  i_ab - i_ca
#   i_b =  i_bc - i_ab
#   i_c =  i_ca - i_bc
_C_DELTA_RAW = np.array([
    [ 1.0,  0.0, -1.0],
    [-1.0,  1.0,  0.0],
    [ 0.0, -1.0,  1.0],
])
# Note: rows sum to zero -> this matrix is singular (rank 2). This
# is the algebraic origin of the "floating" zero-sequence reference
# on a Delta winding, carried through unchanged into Ybus.
#
# NORMALIZATION (bug fix): _C_DELTA_RAW alone is the correct exact
# relation between WINDING currents and TERMINAL currents, but
# stamp_transformer's Yw = y*I assumes every winding sees the SAME
# per-unit leakage admittance y = 1/(r_pu+j x_pu) referred to a base
# where "winding quantities" == "terminal (line) quantities" -- true
# for a wye winding (winding is phase-to-neutral, same as the line
# quantity), but NOT true for a delta winding (winding is phase-to-
# phase; relating it to line quantities on a consistent per-unit
# base introduces a 1/sqrt(3) factor). Without this, the raw
# C_DELTA@C_DELTA^T has positive/negative-sequence eigenvalues of 3
# (vs. 1 for C_WYE@C_WYE^T), i.e. the delta winding's positive-
# sequence transfer admittance came out 3x too large (impedance 3x
# too small) relative to the SAME (r_pu, x_pu) on a wye winding --
# even though transformer connection type must NOT affect positive-
# sequence impedance (a basic invariant; only zero-sequence differs
# by connection). Confirmed empirically: an identical (r=0.01,
# x=0.06 pu) transformer under identical light load converged to
# 0.993 pu secondary voltage for Yg-Yg but 0.573 pu for Yg-Delta
# before this fix -- a ~42% discrepancy with no physical basis.
# Normalizing by 1/sqrt(3) makes C_DELTA's positive/negative-sequence
# eigenvalues equal 1 (matching C_WYE) while its zero-sequence
# eigenvalue stays 0 (singularity behavior is unaffected: a scalar
# normalization cannot change matrix rank).
C_DELTA = _C_DELTA_RAW / np.sqrt(3.0)

C_WYE = np.eye(3)


def stamp_line_series_admittance(Ybus: dict, bus_i: str, bus_j: str,
                                  Yprim: np.ndarray,
                                  phases_i: tuple[str, ...] = PHASES,
                                  phases_j: tuple[str, ...] = PHASES):
    """
    Stamp a phase-coupled series branch between bus_i and bus_j into
    the sparse-dict Ybus structure.

    Ybus is a dict keyed by ((bus,phase),(bus,phase)) -> complex,
    accumulated (not overwritten) so multiple elements between the
    same terminals add correctly (standard admittance-matrix
    assembly rule).

    Yprim: primitive admittance matrix, shape (n, n) where n =
           len(phases_i) == len(phases_j) (a physical series line
           presents the SAME primitive admittance to both of its
           terminals, so phases_i and phases_j must match in count
           and Yprim must be square). Yprim is the usual line's
           self/mutual admittance block such that:
             I_i =  Yprim @ (V_i - V_j)
             I_j = -Yprim @ (V_i - V_j)
           where V_i, V_j are the per-bus PHASE VECTORS (all active
           phases at that bus), so Yprim[a,b] couples phase a to
           phase b of the SAME line, at BOTH ends, not phase a of
           bus_i to phase b of bus_j as an independent element.

    NOTE ON A PRIOR BUG (fixed): an earlier version of this function
    looped over (a_idx, b_idx) and stamped each Yprim[a,b] as if
    (bus_i, phase_a) and (bus_j, phase_b) were the two terminals of
    an isolated mutual element. That put the self-block contribution
    on the wrong diagonal entry ((bus_i,phase_a),(bus_i,phase_a))
    instead of the same-bus mutual entry ((bus_i,phase_a),(bus_i,
    phase_b)), so for any Yprim with off-diagonal (mutual) terms the
    resulting Ybus had ZERO same-bus phase-to-phase coupling and an
    inflated diagonal. Verified independently: NR on the buggy Ybus
    converged to a spurious ~0.02 p.u. state on a lightly-loaded
    2-bus test system, while an independent scipy.fsolve solve of
    the CORRECTLY block-stamped system converges to ~0.99 p.u. This
    would have silently corrupted every untransposed-line experiment
    (IEEE 13/37 both rely on mutual phase coupling).
    """
    def add(key, val):
        Ybus[key] = Ybus.get(key, 0.0) + val

    n = len(phases_i)
    if len(phases_j) != n or Yprim.shape != (n, n):
        raise ValueError(
            "stamp_line_series_admittance requires phases_i and phases_j "
            "of equal length and a square Yprim of that size: a physical "
            "series line's primitive admittance is shared by both ends."
        )

    for a_idx in range(n):
        for b_idx in range(n):
            y = Yprim[a_idx, b_idx]
            if y == 0:
                continue
            pa_i, pb_i = phases_i[a_idx], phases_i[b_idx]
            pa_j, pb_j = phases_j[a_idx], phases_j[b_idx]
            # same-bus (self) blocks: full Yprim at EACH end
            add(((bus_i, pa_i), (bus_i, pb_i)), y)
            add(((bus_j, pa_j), (bus_j, pb_j)), y)
            # cross-bus (mutual) blocks: -Yprim between the ends
            add(((bus_i, pa_i), (bus_j, pb_j)), -y)
            add(((bus_j, pa_j), (bus_i, pb_i)), -y)


def stamp_shunt_admittance(Ybus: dict, bus: str, phase: str, y_shunt: complex):
    """Stamp a phase-to-ground shunt admittance (e.g. OpenDSS
    PPM_Antifloat-equivalent regularization, or line charging)."""
    key = ((bus, phase), (bus, phase))
    Ybus[key] = Ybus.get(key, 0.0) + y_shunt


def _winding_connection_matrix(conn: str) -> np.ndarray:
    conn = conn.lower()
    if conn in ("yg", "wye_grounded", "grounded_wye"):
        return C_WYE
    if conn in ("y", "wye", "ungrounded_wye"):
        # NOT YET IMPLEMENTED -- was previously (incorrectly) aliased
        # to C_WYE on the theory that grounded vs. ungrounded wye only
        # differs by whether a ground-shunt admittance gets stamped
        # afterward. That reasoning is wrong: this whole module uses a
        # SIMPLIFIED lumped-impedance transformer model where each
        # phase's combined primary+secondary leakage is one series
        # branch stamped directly terminal-to-terminal (Y_pp=Cp Yw
        # Cp^T etc.), with no explicit internal neutral node at all --
        # valid for a GROUNDED wye (its neutral is an external 0V
        # reference, not a variable to solve for), but not for an
        # UNGROUNDED wye, whose floating neutral imposes a genuine
        # sum-of-3-currents-into-that-node = 0 constraint that must be
        # eliminated via Kron reduction before you get a terminal-only
        # admittance block. Verified empirically: with C=C_WYE=I, the
        # resulting self-admittance block is full rank 3, i.e. it
        # silently allows zero-sequence current to flow freely through
        # a winding that should have an OPEN zero-sequence circuit
        # (like a delta winding) -- physically wrong for an ungrounded
        # wye, and would misrepresent the exact floating-reference
        # phenomenon this project studies. Doing this correctly needs
        # stamp_transformer restructured to track primary- and
        # secondary-side leakage separately (so an internal floating-
        # neutral node can be introduced and Kron-reduced out) rather
        # than the current single combined-y-per-phase model. Left
        # unimplemented rather than shipping an unverified guess --
        # verify against Kersting's generalized transformer matrices
        # or OpenDSS's model before implementing. Not required by the
        # current Yg-Delta / Delta-Delta reproduction (the project spec
        # sections 14, 15, 19); only needed for the full nine-
        # combination Yg/Y/Delta condition-number study.
        raise NotImplementedError(
            "Ungrounded-wye ('y') windings are not yet implemented in "
            "stamp_transformer -- the current lumped-impedance model "
            "cannot represent a floating neutral correctly (see the "
            "comment above this line). Use 'yg' or 'delta', or "
            "implement the Kron-reduction extension first."
        )
    if conn in ("delta", "d"):
        return C_DELTA
    raise ValueError(f"Unknown winding connection '{conn}'")


def stamp_transformer(Ybus: dict,
                       bus_p: str, conn_p: str,
                       bus_s: str, conn_s: str,
                       r_pu: float, x_pu: float,
                       ground_primary_admittance: float = 0.0,
                       ground_secondary_admittance: float = 0.0):
    """
    Stamp a two-winding three-phase transformer using a per-unit
    leakage impedance (R + jX) common to all three windings and the
    winding connection matrices C_p, C_s.

    Model (short-circuit / leakage-impedance representation, the
    standard simplified per-unit two-winding transformer model used
    for power-flow studies; magnetizing branch neglected, matching
    the base paper's treatment which reports only R, X per pair —
    see project spec sections 14, 15, 19):

      Let y = 1 / (r_pu + j x_pu)   (leakage admittance, same for
      all 3 windings, referred to a common per-unit base).

      Winding-basis primitive admittance (3x3, since each of the 3
      windings sees the same series leakage y, decoupled from the
      others at the winding level):
          Yw = y * I_3

      Terminal-basis stamping via connection matrices:
          Y_pp =  C_p Yw C_p^T
          Y_ps = -C_p Yw C_s^T
          Y_sp = -C_s Yw C_p^T
          Y_ss =  C_s Yw C_s^T

      This four-block structure is stamped into Ybus exactly like a
      line, but with C_p, C_s replacing the identity used for plain
      wye-wye connections.

    ground_primary_admittance / ground_secondary_admittance:
      Optional explicit phase-to-ground shunt admittance added AFTER
      the transformer stamp, to represent a grounded neutral point
      or (for sensitivity study purposes only, per the project spec
      section 11) an anti-float-style regularizing admittance. Zero
      by default: a bare Yg-Delta transformer with an unreferenced
      Delta secondary and no other ground path in the model WILL
      produce a singular Ybus sub-block, which is the intended
      behavior for reproducing the paper's failure case.
    """
    def add(key, val):
        Ybus[key] = Ybus.get(key, 0.0) + val

    y = 1.0 / complex(r_pu, x_pu)
    Yw = y * np.eye(3)

    Cp = _winding_connection_matrix(conn_p)
    Cs = _winding_connection_matrix(conn_s)

    Y_pp = Cp @ Yw @ Cp.T
    Y_ps = -Cp @ Yw @ Cs.T
    Y_sp = -Cs @ Yw @ Cp.T
    Y_ss = Cs @ Yw @ Cs.T

    for a_idx, pa in enumerate(PHASES):
        for b_idx, pb in enumerate(PHASES):
            ip = (bus_p, pa); ip2 = (bus_p, pb)
            isec = (bus_s, pa); isec2 = (bus_s, pb)
            add((ip, ip2), Y_pp[a_idx, b_idx])
            add((ip, isec2), Y_ps[a_idx, b_idx])
            add((isec, ip2), Y_sp[a_idx, b_idx])
            add((isec, isec2), Y_ss[a_idx, b_idx])

    if ground_primary_admittance:
        for p in PHASES:
            stamp_shunt_admittance(Ybus, bus_p, p, ground_primary_admittance)
    if ground_secondary_admittance:
        for p in PHASES:
            stamp_shunt_admittance(Ybus, bus_s, p, ground_secondary_admittance)


def build_dense_ybus(Ybus_dict: dict, state) -> np.ndarray:
    """Convert the sparse-dict Ybus into a dense matrix ordered by
    state.busphase_index, restricted to ACTIVE bus-phases only."""
    n = state.n_busphases
    Y = np.zeros((n, n), dtype=complex)
    for (ii, jj), val in Ybus_dict.items():
        if ii not in state.busphase_index or jj not in state.busphase_index:
            continue  # skip entries touching inactive/unmodeled phases
        Y[state.busphase_index[ii], state.busphase_index[jj]] += val
    return Y