"""Explicit SVD-based Moore--Penrose pseudo-inverse linear solver.

This module is Member B's linear-solver implementation for the shared
Newton--Raphson equation ``J @ dx = rhs``.  It intentionally performs the
singular-value decomposition itself rather than delegating to
``numpy.linalg.pinv`` so that the numerical-rank decision is transparent and
reported to experiments.
"""

from __future__ import annotations

import time

import numpy as np


def _failure_diagnostics(start_time: float, message: str) -> dict:
    """Build the stable diagnostics contract used for pre-SVD failures."""
    return {
        "method": "svd",
        "success": False,
        "error_message": message,
        "runtime_sec": time.perf_counter() - start_time,
        "residual_norm": np.nan,
        "step_norm": np.nan,
        "sigma_max": np.nan,
        "sigma_min": np.nan,
        "singular_values": [],
        "rank_tolerance": np.nan,
        "rtol": np.nan,
        "atol": np.nan,
        "numerical_rank": 0,
        "truncated_singular_values": 0,
        "condition_number": np.nan,
        "effective_condition_number": np.nan,
    }


def solve_svd_pinv(
    J: np.ndarray,
    rhs: np.ndarray,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
):
    """Solve ``J @ dx = rhs`` with a thresholded Moore--Penrose inverse.

    The economy SVD is computed explicitly as ``J = U @ diag(s) @ Vh`` using
    :func:`numpy.linalg.svd` with ``full_matrices=False``.  The update is

    ``dx = V @ diag(s_plus) @ U.conj().T @ rhs``,

    where ``V = Vh.conj().T`` and ``s_plus[i]`` is ``1 / s[i]`` only when
    ``s[i] > tau``.  For real inputs the conjugate transpose is simply the
    transpose stated in the usual real-valued formula.

    Parameters
    ----------
    J : numpy.ndarray, shape (m, n)
        Finite real or complex coefficient matrix.  The Newton use case is a
        square real Jacobian, while rectangular matrices are also supported.
    rhs : numpy.ndarray, shape (m,)
        Finite real or complex right-hand-side vector.
    rtol : float or None, keyword-only
        Non-negative relative singular-value tolerance.  ``None`` selects
        ``eps * max(J.shape)``, using double-precision ``eps``.
    atol : float, keyword-only
        Non-negative absolute singular-value tolerance.  The retained rank is
        exactly ``sum(sigma_i > max(atol, rtol * sigma_max))``; equality with
        the threshold is deliberately truncated.

    Returns
    -------
    dx : numpy.ndarray or None, shape (n,)
        Minimum-norm least-squares update for successful decompositions.  It
        can have a nonzero residual for inconsistent or rank-deficient
        systems.  ``None`` denotes invalid/non-finite numerical input, SVD
        non-convergence, or a non-finite computed result.
    diagnostics : dict
        Always includes ``method`` (``"svd"``), ``success``, ``error_message``,
        ``runtime_sec``, ``residual_norm``, ``step_norm``, ``sigma_max``,
        ``sigma_min``, JSON-serializable ``singular_values``, ``rank_tolerance``,
        actual ``rtol`` and ``atol``, ``numerical_rank``,
        ``truncated_singular_values``, raw ``condition_number``, and
        thresholded ``effective_condition_number``.  The raw condition number
        uses the smallest reported singular value; the effective value uses
        the smallest *retained* singular value.  Rank deficiency is not a
        failure by itself.

    Raises
    ------
    ValueError
        If dimensions are incompatible, a tolerance is non-scalar/non-finite,
        or either tolerance is negative.  These are caller-contract errors;
        non-finite array values instead return ``(None, diagnostics)``.
    """
    t0 = time.perf_counter()

    J_array = np.asarray(J)
    rhs_array = np.asarray(rhs)
    if J_array.ndim != 2:
        raise ValueError(f"J must be two-dimensional; received shape {J_array.shape}.")
    if rhs_array.ndim != 1:
        raise ValueError(f"rhs must be one-dimensional; received shape {rhs_array.shape}.")
    if J_array.shape[0] != rhs_array.shape[0]:
        raise ValueError(
            "rhs length must equal J.shape[0]; "
            f"received {rhs_array.shape[0]} and {J_array.shape[0]}."
        )
    if min(J_array.shape) == 0:
        raise ValueError("J must have at least one row and one column.")

    try:
        rtol_used = np.finfo(float).eps * max(J_array.shape) if rtol is None else float(rtol)
        atol_used = float(atol)
    except (TypeError, ValueError) as exc:
        raise ValueError("rtol and atol must be real scalar tolerances.") from exc
    if not np.isfinite(rtol_used) or not np.isfinite(atol_used):
        raise ValueError("rtol and atol must be finite.")
    if rtol_used < 0.0 or atol_used < 0.0:
        raise ValueError("rtol and atol must be non-negative.")

    # Promote integer and single-precision inputs so the default tolerance and
    # SVD arithmetic have a consistent double-precision interpretation.
    dtype = np.result_type(J_array.dtype, rhs_array.dtype, np.float64)
    J_work = np.asarray(J_array, dtype=dtype)
    rhs_work = np.asarray(rhs_array, dtype=dtype)
    if not np.all(np.isfinite(J_work)) or not np.all(np.isfinite(rhs_work)):
        diagnostics = _failure_diagnostics(t0, "J and rhs must contain only finite values.")
        diagnostics.update(rtol=rtol_used, atol=atol_used)
        return None, diagnostics

    try:
        U, singular_values, Vh = np.linalg.svd(J_work, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        diagnostics = _failure_diagnostics(t0, f"SVD did not converge: {exc}")
        diagnostics.update(rtol=rtol_used, atol=atol_used)
        return None, diagnostics

    if not np.all(np.isfinite(singular_values)):
        diagnostics = _failure_diagnostics(t0, "SVD returned non-finite singular values.")
        diagnostics.update(rtol=rtol_used, atol=atol_used)
        return None, diagnostics

    sigma_max = float(singular_values[0])
    sigma_min = float(singular_values[-1])
    rank_tolerance = max(atol_used, rtol_used * sigma_max)
    retained = singular_values > rank_tolerance
    numerical_rank = int(np.count_nonzero(retained))
    reciprocal_singular_values = np.zeros_like(singular_values)
    reciprocal_singular_values[retained] = 1.0 / singular_values[retained]

    # Multiplication by the reciprocal spectrum makes the truncation explicit
    # and avoids concealing this project-defined rank decision in np.linalg.pinv.
    dx = Vh.conj().T @ (reciprocal_singular_values * (U.conj().T @ rhs_work))
    if not np.all(np.isfinite(dx)):
        diagnostics = _failure_diagnostics(t0, "SVD pseudo-inverse produced a non-finite update.")
        diagnostics.update(
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            singular_values=singular_values.astype(float).tolist(),
            rank_tolerance=rank_tolerance,
            rtol=rtol_used,
            atol=atol_used,
            numerical_rank=numerical_rank,
            truncated_singular_values=int(singular_values.size - numerical_rank),
        )
        return None, diagnostics

    raw_condition_number = np.inf if sigma_min == 0.0 else sigma_max / sigma_min
    effective_condition_number = (
        np.inf if numerical_rank == 0 else sigma_max / float(singular_values[retained][-1])
    )
    diagnostics = {
        "method": "svd",
        "success": True,
        "error_message": None,
        "runtime_sec": time.perf_counter() - t0,
        "residual_norm": float(np.linalg.norm(J_work @ dx - rhs_work)),
        "step_norm": float(np.linalg.norm(dx)),
        "sigma_max": sigma_max,
        "sigma_min": sigma_min,
        "singular_values": singular_values.astype(float).tolist(),
        "rank_tolerance": rank_tolerance,
        "rtol": rtol_used,
        "atol": atol_used,
        "numerical_rank": numerical_rank,
        "truncated_singular_values": int(singular_values.size - numerical_rank),
        "condition_number": float(raw_condition_number),
        "effective_condition_number": float(effective_condition_number),
    }
    return dx, diagnostics
