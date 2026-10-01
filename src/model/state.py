# ============================================================
# PART 1 of 5 — state.py
# Bus/phase indexing and state-vector representation
# ============================================================
"""
state.py — Common three-phase state representation for the shared
Newton-Raphson power flow engine (Member A ownership).

Design choices (documented per the project spec instructions — these are
OUR reconstruction/implementation choices, not paper-specified):

1. RECTANGULAR coordinates (e = Re(V), f = Im(V)) are used for the
   state vector rather than polar (|V|, angle). Rationale:
     - Mismatch equations become low-order polynomial in (e,f),
       so the analytical Jacobian has no trigonometric terms.
     - Avoids angle-wrap issues.
     - This is a standard alternative NR formulation; it does not
       change the underlying singularity phenomenon studied in the
       base paper (the Jacobian singularity is a property of the
       network's zero-sequence/floating structure, not of the
       choice of real vs polar coordinates).
2. Every bus has up to 3 phases (a, b, c). A bus may have fewer than
   3 active phases (e.g. Delta secondary laterals, single-phase
   taps). Inactive phases are masked out of the unknown vector and
   out of the mismatch vector entirely (they do not contribute rows
   or columns to J), rather than being forced to zero-inject.
3. Bus 0 (or whichever bus is flagged is_slack=True) has FIXED
   voltage on all its active phases. Slack phases are excluded from
   the unknown vector x.
4. PQ buses supply P_spec, Q_spec (positive = consumption, i.e.
   load convention) per active phase.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


PHASES = ("a", "b", "c")


@dataclass
class BusPhase:
    """One phase at one bus."""
    bus_name: str
    phase: str          # 'a', 'b', or 'c'
    active: bool = True         # False if this phase does not exist at this bus
    is_slack: bool = False      # True if voltage is fixed (not part of unknown x)
    V_slack: complex = None     # required if is_slack=True
    P_spec: float = 0.0         # per-unit, load convention (+ = consumption)
    Q_spec: float = 0.0         # per-unit, load convention (+ = consumption)


class NetworkState:
    """
    Holds bus/phase indexing and the current voltage state.

    Global index convention:
      - `busphase_index[(bus_name, phase)] -> global bus-phase id`
        for ALL active bus-phases (slack and non-slack).
      - `unknown_index[(bus_name, phase)] -> position in x`
        only for NON-slack active bus-phases. Two consecutive
        positions per bus-phase: [.., e_i, f_i, ..].
    """

    def __init__(self):
        self._busphases: list[BusPhase] = []
        self.busphase_index: dict[tuple[str, str], int] = {}
        self.unknown_index: dict[tuple[str, str], int] = {}
        self._finalized = False
        self.V: dict[tuple[str, str], complex] = {}

    # ---------------------------------------------------------
    # Construction API
    # ---------------------------------------------------------
    def add_bus_phase(self, bp: BusPhase):
        if self._finalized:
            raise RuntimeError("Cannot add bus-phases after finalize().")
        if not bp.active:
            return  # inactive phases are simply not registered
        key = (bp.bus_name, bp.phase)
        if key in self.busphase_index:
            raise ValueError(f"Duplicate bus-phase {key}")
        self.busphase_index[key] = len(self._busphases)
        self._busphases.append(bp)

    def finalize(self):
        """Call after all bus-phases are added. Builds the unknown-vector index."""
        if self._finalized:
            return
        pos = 0
        for bp in self._busphases:
            key = (bp.bus_name, bp.phase)
            if not bp.is_slack:
                self.unknown_index[key] = pos
                pos += 2  # (e, f)
        self.n_unknowns = pos
        self.n_busphases = len(self._busphases)
        self._finalized = True

    # ---------------------------------------------------------
    # Accessors
    # ---------------------------------------------------------
    def all_busphases(self) -> list[BusPhase]:
        return list(self._busphases)

    def non_slack_busphases(self) -> list[BusPhase]:
        return [bp for bp in self._busphases if not bp.is_slack]

    def get(self, bus_name: str, phase: str) -> BusPhase:
        return self._busphases[self.busphase_index[(bus_name, phase)]]

    def is_active(self, bus_name: str, phase: str) -> bool:
        return (bus_name, phase) in self.busphase_index

    # ---------------------------------------------------------
    # State vector <-> voltage dict conversions
    # ---------------------------------------------------------
    def init_flat_start(self, v_mag: float = 1.0) -> np.ndarray:
        """Flat start: slack phases at their specified value (assumed
        balanced 120-degree separated if not overridden), all PQ
        phases at v_mag∠(phase offset), matching typical NR init."""
        offset = {"a": 0.0, "b": -2 * np.pi / 3, "c": 2 * np.pi / 3}
        x = np.zeros(self.n_unknowns)
        for bp in self._busphases:
            V = bp.V_slack if bp.is_slack else v_mag * np.exp(1j * offset[bp.phase])
            self.V[(bp.bus_name, bp.phase)] = V
            if not bp.is_slack:
                i = self.unknown_index[(bp.bus_name, bp.phase)]
                x[i] = V.real
                x[i + 1] = V.imag
        return x

    def unpack(self, x: np.ndarray) -> dict[tuple[str, str], complex]:
        """Build full complex voltage dict (slack + PQ) from unknown vector x."""
        V = {}
        for bp in self._busphases:
            key = (bp.bus_name, bp.phase)
            if bp.is_slack:
                V[key] = bp.V_slack
            else:
                i = self.unknown_index[key]
                V[key] = x[i] + 1j * x[i + 1]
        return V

    def pack(self, V: dict[tuple[str, str], complex]) -> np.ndarray:
        """Inverse of unpack, restricted to non-slack unknowns."""
        x = np.zeros(self.n_unknowns)
        for key, i in self.unknown_index.items():
            x[i] = V[key].real
            x[i + 1] = V[key].imag
        return x