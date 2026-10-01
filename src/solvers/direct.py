# ============================================================
# PART 5 of 5 — solvers/direct.py (FINAL VERSION)
# Direct linear solver with numerical-singularity guard
# ============================================================
"""
direct.py — Classical direct linear solve for J*dx = rhs, with an
explicit numerical-singularity guard (Member A ownership).

Per project spec section 4 / plan.md section 4: MUST use
numpy.linalg.solve (LU-based). MUST NOT use numpy.linalg.inv(J) @ rhs.

WHY THE GUARD EXISTS (numerical-analysis background):
numpy.linalg.solve() only raises LinAlgError when a pivot is EXACTLY
zero in floating point -- this almost never happens for a matrix
that is near-singular but not exactly singular. Instead, floating-
point round-off gets amplified by roughly the matrix's condition
number kappa. Rule of thumb: you lose about log10(kappa) decimal
digits of accuracy out of the ~16 available in double precision.
A tiny linear-solve residual (||J@dx - rhs||) does NOT indicate an
accurate answer when kappa is huge -- residual measures how well dx
satisfies the equation, not how close dx is to the true solution,
and the two diverge arbitrarily as kappa grows. Without this guard,
Newton-Raphson can silently "converge" on a numerically corrupted
answer while reporting success. Confirmed empirically on the IEEE
4-bus singular case (experiments/exp01_4bus.py, Case B): even after
fixing two upstream Ybus stamping bugs (see model/ybus.py), kappa_2
still climbs past 1e11 during iteration while the linear-solve
residual stays tiny throughout -- residual size alone never signals
the problem; only tracking kappa_2 does. Exact numbers depend on the
current model/reconstruction and will keep shifting as the team's
feeder data matures, so don't treat any single cited value here as
fixed -- rerun experiments/exp01_4bus.py for the current number.

COND_NUMBER_FAIL_THRESHOLD = 1e12 is a project-defined engineering
threshold (not paper-specified) chosen as an early-warning cut, well
before total precision loss (~1/eps ~ 4.5e15). Document this choice
explicitly in the report; it is open to sensitivity analysis, the
same way Member B/C's rank thresholds are.
"""

from __future__ import annotations
import time
import numpy as np

COND_NUMBER_FAIL_THRESHOLD = 1e12


def solve_direct(J: np.ndarray, rhs: np.ndarray,
                  cond_fail_threshold: float = COND_NUMBER_FAIL_THRESHOLD):
    """
    Solve J @ dx = rhs directly via numpy.linalg.solve (LU-based).

    Reports success=False in TWO distinct, separately-logged cases:
      1. 'exact_singular'  : numpy.linalg.solve() itself raised
                              LinAlgError (a pivot was exactly zero).
      2. 'ill_conditioned' : solve() succeeded with no exception, but
                              condition_number(J) exceeded
                              cond_fail_threshold. dx is discarded
                              (returned as None) even though solve()
                              nominally "succeeded" -- callers can
                              never accidentally use a corrupted
                              update.

    Returns
    -------
    dx : ndarray or None
         None if either failure mode above occurred.
    diagnostics : dict with:
         'method', 'success', 'runtime_sec', 'residual_norm',
         'error_message', 'condition_number', 'failure_mode'
         (None | 'exact_singular' | 'ill_conditioned'),
         'cond_fail_threshold'.
    """
    t0 = time.perf_counter()
    diagnostics = {"method": "direct", "cond_fail_threshold": cond_fail_threshold}

    try:
        cond = float(np.linalg.cond(J))
    except np.linalg.LinAlgError:
        cond = np.inf
    diagnostics["condition_number"] = cond

    try:
        dx_raw = np.linalg.solve(J, rhs)
    except np.linalg.LinAlgError as e:
        diagnostics.update(
            success=False, failure_mode="exact_singular",
            error_message=str(e),
            runtime_sec=time.perf_counter() - t0,
            residual_norm=np.nan,
        )
        return None, diagnostics

    if not np.isfinite(cond) or cond > cond_fail_threshold:
        diagnostics.update(
            success=False, failure_mode="ill_conditioned",
            error_message=(
                f"condition number {cond:.3e} exceeds threshold "
                f"{cond_fail_threshold:.1e}; linear solve is numerically "
                f"unreliable even though numpy.linalg.solve raised no "
                f"exception (see module docstring)."
            ),
            runtime_sec=time.perf_counter() - t0,
            residual_norm=float(np.linalg.norm(J @ dx_raw - rhs)),
        )
        return None, diagnostics

    diagnostics.update(
        success=True, failure_mode=None, error_message=None,
        runtime_sec=time.perf_counter() - t0,
        residual_norm=float(np.linalg.norm(J @ dx_raw - rhs)),
    )
    return dx_raw, diagnostics