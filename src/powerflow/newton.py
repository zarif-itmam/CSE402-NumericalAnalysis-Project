# ============================================================
# PART 5 continued — powerflow/newton.py (FINAL VERSION)
# Shared Newton-Raphson loop, with the dx_norm bugfix applied
# ============================================================
"""
newton.py — Shared Newton-Raphson driver (Member A ownership).

Single nonlinear loop used by every solver method (plan.md section 3:
no member creates a separate NR loop). Members B/C/D only ever add a
branch inside solve_linear_step's dispatch.

Per-iteration diagnostics logged (plan.md section 11 / the project spec
section 23): ||F_k||_inf, ||dx_k||_2, kappa_2(J_k), sigma_min(J_k),
sigma_max(J_k), numerical rank (informational default), ||J dx+F||_2
cross-check, per-solver diagnostics dict, voltage min/max, runtime.
"""

from __future__ import annotations
import time
import numpy as np

from .mismatch import compute_mismatch
from .jacobian import compute_jacobian


def solve_linear_step(J: np.ndarray, rhs: np.ndarray, method: str = "direct",
                       options: dict | None = None):
    """Dispatch to the requested linear-solver method. Every method
    module must return the same (dx, diagnostics) contract."""
    options = options or {}

    if method == "direct":
        from ..solvers.direct import solve_direct
        return solve_direct(J, rhs, **options)

    elif method == "svd":
        from ..solvers.svd_pinv import solve_svd_pinv
        return solve_svd_pinv(J, rhs, **options)

    elif method == "rrqr":
        from ..solvers.rrqr import solve_rrqr
        return solve_rrqr(J, rhs, **options)

    elif method == "tikhonov":
        from ..solvers.tikhonov import solve_tikhonov
        return solve_tikhonov(J, rhs, **options)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Expected one of: "
            "'direct', 'svd', 'rrqr', 'tikhonov'."
        )


def _numerical_rank(J: np.ndarray, singular_values: np.ndarray) -> int:
    """Default informational rank estimate (numpy.linalg.matrix_rank
    convention). NOT any specific solver's own rank threshold --
    Members B/C log their own method-specific thresholds separately."""
    if singular_values.size == 0 or singular_values[0] == 0:
        return 0
    tol = max(J.shape) * np.finfo(float).eps * singular_values[0]
    return int(np.sum(singular_values > tol))


def newton_raphson(state, Ybus: np.ndarray, x0: np.ndarray,
                    method: str = "direct",
                    solver_options: dict | None = None,
                    tol_F: float = 1e-8,
                    max_iter: int = 30,
                    compute_conditioning: bool = True):
    """
    Run the shared Newton-Raphson iteration.

    compute_conditioning: if True, computes an SVD of J every
    iteration purely for diagnostics (kappa_2/sigma_min/sigma_max/
    rank), independent of which solve method is active. Disable for
    large-system runtime benchmarking (Member E's IEEE-118 work)
    where this extra O(n^3) SVD would distort timing comparisons.

    Returns dict with: 'converged', 'x', 'iterations', 'history',
    'fail_reason'.
    """
    solver_options = solver_options or {}
    x = x0.copy()
    history = []
    converged = False
    fail_reason = None

    for k in range(max_iter):
        t_iter0 = time.perf_counter()

        F = compute_mismatch(x, state, Ybus)
        F_inf = float(np.linalg.norm(F, ord=np.inf))

        iter_log = {"iteration": k, "F_inf_norm": F_inf, "method": method}

        if F_inf < tol_F:
            converged = True
            V_dict = state.unpack(x)
            vmags = [abs(v) for v in V_dict.values()]
            iter_log.update({
                "dx_norm": 0.0,
                "converged_this_iter": True,
                "voltage_min": min(vmags),
                "voltage_max": max(vmags),
                "total_iter_runtime_sec": time.perf_counter() - t_iter0,
            })
            history.append(iter_log)
            break

        J = compute_jacobian(x, state, Ybus)

        if compute_conditioning:
            try:
                sv = np.linalg.svd(J, compute_uv=False)
                sigma_max = float(sv[0])
                sigma_min = float(sv[-1])
                kappa2 = float(sigma_max / sigma_min) if sigma_min > 0 else np.inf
                rank_est = _numerical_rank(J, sv)
            except np.linalg.LinAlgError:
                sigma_max = sigma_min = kappa2 = np.nan
                rank_est = -1
            iter_log.update({
                "sigma_max": sigma_max, "sigma_min": sigma_min,
                "kappa_2": kappa2, "numerical_rank": rank_est,
            })

        rhs = -F
        dx, solver_diag = solve_linear_step(J, rhs, method=method, options=solver_options)
        iter_log["solver_diagnostics"] = solver_diag

        if dx is None or not solver_diag.get("success", False):
            fail_reason = solver_diag.get("error_message", "linear solve failed")
            iter_log["dx_norm"] = float("nan")
            iter_log["converged_this_iter"] = False
            iter_log["total_iter_runtime_sec"] = time.perf_counter() - t_iter0
            history.append(iter_log)
            break

        iter_log["linear_residual_norm_crosscheck"] = float(np.linalg.norm(J @ dx + F))
        dx_norm = float(np.linalg.norm(dx))
        iter_log["dx_norm"] = dx_norm

        if not np.all(np.isfinite(dx)):
            fail_reason = "non-finite update (dx contains NaN/Inf)"
            iter_log["converged_this_iter"] = False
            iter_log["total_iter_runtime_sec"] = time.perf_counter() - t_iter0
            history.append(iter_log)
            break

        x = x + dx

        V_dict = state.unpack(x)
        vmags = [abs(v) for v in V_dict.values()]
        iter_log["voltage_min"] = min(vmags)
        iter_log["voltage_max"] = max(vmags)
        iter_log["converged_this_iter"] = False
        iter_log["total_iter_runtime_sec"] = time.perf_counter() - t_iter0
        history.append(iter_log)

        if dx_norm > 1e8 or max(vmags) > 1e6:
            fail_reason = "diverged (state magnitude exploded)"
            break

    else:
        fail_reason = fail_reason or f"max_iter={max_iter} reached without convergence"

    if not converged and fail_reason is None:
        fail_reason = "did not converge (unspecified)"

    return {
        "converged": converged,
        "x": x,
        "iterations": len(history),
        "history": history,
        "fail_reason": None if converged else fail_reason,
    }