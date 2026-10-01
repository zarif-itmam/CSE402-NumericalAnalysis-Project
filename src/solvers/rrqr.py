"""Rank-revealing QR (RRQR / QR with column pivoting) linear solver.

This module is Member C's linear-solver implementation for the shared
Newton--Raphson equation ``J @ dx = rhs`` (see ``the project spec`` section 5.C and
``robust_nr_powerflow_6_week_plan.md`` section 6). Ordinary (unpivoted) QR
followed by naive triangular back-substitution is not sufficient for an
exactly singular or rank-deficient Jacobian, because ``R`` itself becomes
rank-deficient and back-substitution then divides by a (near-)zero pivot.
Instead this module combines two LAPACK calls driven by the same threshold:

1. an explicit economy *pivoted* QR factorization, ``J[:, piv] = Q @ R``, via
   :func:`scipy.linalg.qr` with ``pivoting=True``, used only to expose the
   numerical rank, pivot ordering, and ``|R_ii|`` diagonal as diagnostics; and
2. the actual solve via :func:`scipy.linalg.lstsq` with
   ``lapack_driver="gelsy"``, which performs a complete orthogonal
   factorization (pivoted QR plus an elimination of the trailing block) and
   returns the minimum-norm least-squares solution for a rank-deficient
   system -- not a naive triangular back-substitution.
"""

from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla


def _failure_diagnostics(start_time: float, message: str) -> dict:
    """Build the stable diagnostics contract used for pre-solve failures."""
    return {
        "method": "rrqr",
        "success": False,
        "error_message": message,
        "runtime_sec": time.perf_counter() - start_time,
        "residual_norm": np.nan,
        "step_norm": np.nan,
        "r_diagonal": [],
        "pivot_ordering": [],
        "rank_tolerance": np.nan,
        "rtol": np.nan,
        "atol": np.nan,
        "numerical_rank": 0,
        "truncated_columns": 0,
        "smallest_accepted_r_diagonal": np.nan,
        "condition_number": np.nan,
        "lapack_rank": None,
    }


def solve_rrqr(
    J: np.ndarray,
    rhs: np.ndarray,
    *,
    rtol: float | None = None,
    atol: float = 0.0,
):
    """Solve ``J @ dx = rhs`` with rank-revealing (column-pivoted) QR.

    Parameters
    ----------
    J : numpy.ndarray, shape (m, n)
        Finite real coefficient matrix. The Newton use case is a square real
        Jacobian; rectangular matrices are also supported.
    rhs : numpy.ndarray, shape (m,)
        Finite real right-hand-side vector.
    rtol : float or None, keyword-only
        Non-negative relative rank-decision tolerance, applied to the pivoted
        ``|R_ii|`` diagonal exactly as Member B's SVD solver applies it to
        the singular-value spectrum. ``None`` selects ``eps * max(J.shape)``
        using double-precision ``eps``.
    atol : float, keyword-only
        Non-negative absolute tolerance. The accepted threshold is
        ``tau = max(atol, rtol * |R_00|)``; the same ``tau`` (rescaled to a
        relative cutoff) also drives LAPACK's GELSY rank decision for the
        actual solve, so the reported diagnostic rank and the solved rank
        describe the same cutoff.

    Returns
    -------
    dx : numpy.ndarray or None, shape (n,)
        Minimum-norm least-squares update for a successful factorization. It
        can have a nonzero residual for inconsistent or rank-deficient
        systems. ``None`` denotes invalid/non-finite numerical input, a
        failed factorization, or a non-finite computed result.
    diagnostics : dict
        Always includes ``method`` (``"rrqr"``), ``success``,
        ``error_message``, ``runtime_sec``, ``residual_norm``, ``step_norm``,
        the full pivoted ``r_diagonal`` (``|R_ii|``, JSON-serializable),
        ``pivot_ordering`` (the column permutation ``piv`` such that
        ``J[:, piv] = Q @ R``), ``rank_tolerance``, actual ``rtol`` and
        ``atol``, ``numerical_rank``, ``truncated_columns``, the
        ``smallest_accepted_r_diagonal`` value, an ``|R_00| / |R_rank-1|``
        ``condition_number`` estimate, and ``lapack_rank`` (GELSY's own
        internally estimated rank, recorded for cross-checking against
        ``numerical_rank``). Rank deficiency is not a failure by itself.

    Raises
    ------
    ValueError
        If dimensions are incompatible, a tolerance is non-scalar/non-finite,
        or either tolerance is negative. These are caller-contract errors;
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

    # Promote integer/single-precision inputs, matching the SVD solver's
    # convention so the default tolerance and factorization arithmetic use a
    # consistent double-precision interpretation.
    dtype = np.result_type(J_array.dtype, rhs_array.dtype, np.float64)
    J_work = np.asarray(J_array, dtype=dtype)
    rhs_work = np.asarray(rhs_array, dtype=dtype)
    if not np.all(np.isfinite(J_work)) or not np.all(np.isfinite(rhs_work)):
        diagnostics = _failure_diagnostics(t0, "J and rhs must contain only finite values.")
        diagnostics.update(rtol=rtol_used, atol=atol_used)
        return None, diagnostics

    try:
        _, R, piv = sla.qr(J_work, mode="economic", pivoting=True)
    except sla.LinAlgError as exc:
        diagnostics = _failure_diagnostics(t0, f"Pivoted QR did not converge: {exc}")
        diagnostics.update(rtol=rtol_used, atol=atol_used)
        return None, diagnostics

    r_diagonal = np.abs(np.diag(R))
    if not np.all(np.isfinite(r_diagonal)):
        diagnostics = _failure_diagnostics(t0, "Pivoted QR returned a non-finite R diagonal.")
        diagnostics.update(rtol=rtol_used, atol=atol_used)
        return None, diagnostics

    r_max = float(r_diagonal[0]) if r_diagonal.size else 0.0
    rank_tolerance = max(atol_used, rtol_used * r_max)

    # Column pivoting selects the largest-norm remaining column at every
    # step, so |R_ii| is non-increasing in practice (not guaranteed by
    # proof, unlike SVD's singular values). Rank is therefore the number of
    # *leading* entries strictly above tau -- the standard rank-revealing
    # convention -- rather than a global count, which could otherwise
    # "readmit" a later entry that exceeds tau after an earlier drop.
    numerical_rank = 0
    for value in r_diagonal:
        if value > rank_tolerance:
            numerical_rank += 1
        else:
            break

    if r_max == 0.0:
        # J is (numerically) the zero matrix: every column is already
        # eliminated, so the minimum-norm least-squares update is exactly
        # zero regardless of rhs.
        dx = np.zeros(J_work.shape[1], dtype=dtype)
        diagnostics = {
            "method": "rrqr",
            "success": True,
            "error_message": None,
            "runtime_sec": time.perf_counter() - t0,
            "residual_norm": float(np.linalg.norm(rhs_work)),
            "step_norm": 0.0,
            "r_diagonal": r_diagonal.astype(float).tolist(),
            "pivot_ordering": piv.astype(int).tolist(),
            "rank_tolerance": rank_tolerance,
            "rtol": rtol_used,
            "atol": atol_used,
            "numerical_rank": 0,
            "truncated_columns": int(r_diagonal.size),
            "smallest_accepted_r_diagonal": np.nan,
            "condition_number": np.inf,
            "lapack_rank": 0,
        }
        return dx, diagnostics

    # Drive LAPACK's own GELSY rank decision with the same threshold used
    # above (rescaled to the relative cutoff GELSY expects), so the reported
    # diagnostic rank and the solved rank describe the same cutoff.
    rcond_used = rank_tolerance / r_max

    try:
        dx, _, lapack_rank, _ = sla.lstsq(
            J_work, rhs_work, cond=rcond_used, lapack_driver="gelsy"
        )
    except sla.LinAlgError as exc:
        diagnostics = _failure_diagnostics(t0, f"GELSY least-squares solve failed: {exc}")
        diagnostics.update(
            rtol=rtol_used,
            atol=atol_used,
            r_diagonal=r_diagonal.astype(float).tolist(),
            pivot_ordering=piv.astype(int).tolist(),
            rank_tolerance=rank_tolerance,
            numerical_rank=numerical_rank,
            truncated_columns=int(r_diagonal.size - numerical_rank),
        )
        return None, diagnostics

    if not np.all(np.isfinite(dx)):
        diagnostics = _failure_diagnostics(
            t0, "RRQR least-squares solve produced a non-finite update."
        )
        diagnostics.update(
            rtol=rtol_used,
            atol=atol_used,
            r_diagonal=r_diagonal.astype(float).tolist(),
            pivot_ordering=piv.astype(int).tolist(),
            rank_tolerance=rank_tolerance,
            numerical_rank=numerical_rank,
            truncated_columns=int(r_diagonal.size - numerical_rank),
            lapack_rank=int(lapack_rank),
        )
        return None, diagnostics

    smallest_accepted = (
        float(r_diagonal[numerical_rank - 1]) if numerical_rank > 0 else np.nan
    )
    condition_number = (
        np.inf if numerical_rank == 0 else float(r_max / smallest_accepted)
    )

    diagnostics = {
        "method": "rrqr",
        "success": True,
        "error_message": None,
        "runtime_sec": time.perf_counter() - t0,
        "residual_norm": float(np.linalg.norm(J_work @ dx - rhs_work)),
        "step_norm": float(np.linalg.norm(dx)),
        "r_diagonal": r_diagonal.astype(float).tolist(),
        "pivot_ordering": piv.astype(int).tolist(),
        "rank_tolerance": rank_tolerance,
        "rtol": rtol_used,
        "atol": atol_used,
        "numerical_rank": numerical_rank,
        "truncated_columns": int(r_diagonal.size - numerical_rank),
        "smallest_accepted_r_diagonal": smallest_accepted,
        "condition_number": condition_number,
        "lapack_rank": int(lapack_rank),
    }
    return dx, diagnostics
