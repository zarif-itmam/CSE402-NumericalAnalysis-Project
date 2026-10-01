# ============================================================
# PART 4 of 5 — tests/test_jacobian.py
# Finite-difference validation of the analytical Jacobian
# ============================================================
"""
test_jacobian.py — Mandatory Jacobian validation (Member A ownership).

Per plan.md section 4 and project spec section 22, this is a HARD
GATE: no experiment result is trustworthy until this passes. It
compares the analytical Jacobian (jacobian.py) against a centered
finite-difference approximation:

    J_ij^FD ≈ [F_i(x + h e_j) - F_i(x - h e_j)] / (2h)

and reports:

    E_J = ||J_analytic - J_FD||_F / ||J_FD||_F

This file is written to run two ways:
  1. As a pytest test (`pytest tests/test_jacobian.py`) if pytest is
     available — asserts E_J below tolerance.
  2. As a plain script (`python tests/test_jacobian.py`) with no
     pytest dependency, printing a pass/fail report — useful since
     the team has "moderate Python experience" per the project spec and
     may not have pytest set up yet.

It is deliberately self-contained: it builds its own small
synthetic 3-bus test network (1 slack + 2 PQ bus-phases) rather than
depending on the IEEE 4-bus case, so this test can run and be
trusted BEFORE the 4-bus data entry (Part 5) is even done. The
IEEE 4-bus case gets its own additional Jacobian check when built.
"""

from __future__ import annotations
import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.state import NetworkState, BusPhase, PHASES
from src.model.ybus import (
    stamp_line_series_admittance, stamp_shunt_admittance, build_dense_ybus
)
from src.powerflow.mismatch import compute_mismatch
from src.powerflow.jacobian import compute_jacobian


def build_synthetic_3phase_test_system():
    """
    Small synthetic 3-phase test system for Jacobian validation:
      Bus SRC: slack, all 3 phases, balanced 1.0 pu.
      Bus L1:  PQ, all 3 phases, loaded.
    Connected by a simple phase-coupled line (self + mutual coupling,
    so off-diagonal Jacobian blocks are genuinely exercised, not
    just the diagonal).
    """
    state = NetworkState()

    V_slack = {
        "a": 1.0 + 0j,
        "b": 1.0 * np.exp(1j * -2 * np.pi / 3),
        "c": 1.0 * np.exp(1j * 2 * np.pi / 3),
    }
    for p in PHASES:
        state.add_bus_phase(BusPhase(
            bus_name="SRC", phase=p, active=True,
            is_slack=True, V_slack=V_slack[p]
        ))

    # Realistic-ish per-unit loading, deliberately unequal across
    # phases so the test also exercises imbalance, not just a
    # symmetric special case that might hide a sign bug.
    loads = {"a": (0.30, 0.10), "b": (0.25, 0.08), "c": (0.35, 0.12)}
    for p in PHASES:
        P, Q = loads[p]
        state.add_bus_phase(BusPhase(
            bus_name="L1", phase=p, active=True,
            is_slack=False, P_spec=P, Q_spec=Q
        ))

    state.finalize()

    # Phase-coupled line primitive (self admittance dominant,
    # small mutual coupling between phases -- typical untransposed
    # line structure, not a diagonal-only matrix).
    z_self = complex(0.01, 0.08)
    z_mutual = complex(0.003, 0.02)
    Zprim = np.array([
        [z_self,    z_mutual,  z_mutual],
        [z_mutual,  z_self,    z_mutual],
        [z_mutual,  z_mutual,  z_self],
    ])
    Yprim = np.linalg.inv(Zprim)

    Ybus_dict = {}
    stamp_line_series_admittance(Ybus_dict, "SRC", "L1", Yprim)

    # Small ground shunt at the PQ bus so the test system is
    # well-conditioned (this test validates DERIVATIVE correctness,
    # not singular-case behavior -- singularity reproduction is
    # Part 5's job on the real IEEE 4-bus case).
    for p in PHASES:
        stamp_shunt_admittance(Ybus_dict, "L1", p, complex(0, 1e-3))

    Ybus = build_dense_ybus(Ybus_dict, state)
    return state, Ybus


def finite_difference_jacobian(x: np.ndarray, state, Ybus: np.ndarray,
                                h: float = 1e-6) -> np.ndarray:
    """Centered finite-difference Jacobian, per plan.md section 4."""
    n = x.size
    J_fd = np.zeros((n, n))
    for j in range(n):
        x_plus = x.copy();  x_plus[j] += h
        x_minus = x.copy(); x_minus[j] -= h
        F_plus = compute_mismatch(x_plus, state, Ybus)
        F_minus = compute_mismatch(x_minus, state, Ybus)
        J_fd[:, j] = (F_plus - F_minus) / (2 * h)
    return J_fd


def compute_relative_frobenius_error(J_analytic: np.ndarray, J_fd: np.ndarray) -> float:
    """E_J = ||J_analytic - J_FD||_F / ||J_FD||_F, per plan.md section 4."""
    num = np.linalg.norm(J_analytic - J_fd, ord="fro")
    den = np.linalg.norm(J_fd, ord="fro")
    if den == 0:
        raise ValueError("J_FD has zero Frobenius norm; test system is degenerate.")
    return float(num / den)


# Tolerance rationale: centered finite differences have O(h^2)
# truncation error; with h=1e-6 in double precision this typically
# gives agreement to ~1e-6 to ~1e-8 relative error for well-scaled
# problems like this one. 1e-6 is used as a conservative pass
# threshold -- tighten if your team's h/precision setup allows it,
# but do not loosen this without understanding why the analytical
# Jacobian disagrees.
E_J_TOLERANCE = 1e-6


def run_validation(verbose: bool = True) -> float:
    state, Ybus = build_synthetic_3phase_test_system()
    x0 = state.init_flat_start(v_mag=1.0)

    # Perturb away from the flat start so the test is not
    # accidentally evaluated at a special/degenerate point (e.g.
    # x=flat-start could mask a bug that only appears once P_calc,
    # Q_calc are non-trivial).
    rng = np.random.default_rng(seed=42)
    x_test = x0 + 0.02 * rng.standard_normal(x0.size)

    J_analytic = compute_jacobian(x_test, state, Ybus)
    J_fd = finite_difference_jacobian(x_test, state, Ybus, h=1e-6)
    E_J = compute_relative_frobenius_error(J_analytic, J_fd)

    if verbose:
        print(f"State size (n_unknowns)       : {state.n_unknowns}")
        print(f"||J_analytic||_F              : {np.linalg.norm(J_analytic, 'fro'):.6e}")
        print(f"||J_FD||_F                    : {np.linalg.norm(J_fd, 'fro'):.6e}")
        print(f"||J_analytic - J_FD||_F       : {np.linalg.norm(J_analytic - J_fd, 'fro'):.6e}")
        print(f"E_J (relative Frobenius error): {E_J:.6e}")
        print(f"Tolerance                     : {E_J_TOLERANCE:.1e}")
        print(f"Result                        : {'PASS' if E_J < E_J_TOLERANCE else 'FAIL'}")

        max_abs_diff = np.max(np.abs(J_analytic - J_fd))
        i_max, j_max = np.unravel_index(np.argmax(np.abs(J_analytic - J_fd)), J_analytic.shape)
        print(f"Largest single-entry abs error: {max_abs_diff:.3e} at J[{i_max},{j_max}]"
              f"  (analytic={J_analytic[i_max,j_max]:.6e}, FD={J_fd[i_max,j_max]:.6e})")

    return E_J


# ---------------- pytest entry point ----------------
def test_jacobian_matches_finite_difference():
    E_J = run_validation(verbose=False)
    assert E_J < E_J_TOLERANCE, (
        f"Analytical Jacobian disagrees with finite-difference approximation: "
        f"E_J={E_J:.3e} exceeds tolerance {E_J_TOLERANCE:.1e}. "
        f"Do not trust downstream experiments until this passes."
    )


# ---------------- plain-script entry point ----------------
if __name__ == "__main__":
    print("=" * 60)
    print("Jacobian finite-difference validation")
    print("=" * 60)
    E_J = run_validation(verbose=True)
    print("=" * 60)
    if E_J < E_J_TOLERANCE:
        print("PASSED — analytical Jacobian is validated.")
        sys.exit(0)
    else:
        print("FAILED — do NOT trust downstream NR experiments yet.")
        sys.exit(1)