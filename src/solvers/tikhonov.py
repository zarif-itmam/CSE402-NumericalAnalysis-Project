"""Tikhonov-regularized linear solver for the shared Newton-Raphson step.

This module is Member D's linear-solver implementation for
``J @ dx = rhs`` (see ``the project spec`` section 5.D and
``robust_nr_powerflow_6_week_plan.md`` section 7). It solves

    min_dx ||J @ dx - rhs||_2^2 + lambda * ||dx||_2^2

via the augmented least-squares system

    [J          ]         [rhs]
    [sqrt(lambda) * I] dx ~ [ 0 ]

using :func:`numpy.linalg.lstsq` (LAPACK ``gelsd``, SVD-based). It
deliberately never forms the normal-equations matrix
``(J.T @ J + lambda * I)`` or its inverse: doing so would square the
condition number (``kappa(J.T @ J) = kappa(J)^2``), which is exactly the
failure mode this project studies, not a fix for it.

``lambda`` is parameterized as ``lambda = alpha * sigma_max(J) ** 2`` per the
the project's scale-aware convention, so that the same ``alpha`` value has a
comparable regularization strength across Jacobians of different scale (e.g.
across NR iterations, or across feeders). There is deliberately no single
"correct" default ``alpha``: per project spec section 5.D, Tikhonov results
should be reported as a sensitivity/regularization sweep unless a defensible
fixed-lambda selection procedure is developed. The default here
(``alpha=1e-12``) is the most conservative (least intrusive) end of the
project's suggested sweep grid, not a considered choice.
"""

from __future__ import annotations

import time

import numpy as np


def _failure_diagnostics(start_time: float, message: str) -> dict:
    """Build the stable diagnostics contract used for pre-solve failures."""
    return {
        "method": "tikhonov",
        "success": False,
        "error_message": message,
        "runtime_sec": time.perf_counter() - start_time,
        "residual_norm": np.nan,
        "step_norm": np.nan,
        "alpha": np.nan,
        "lambda_reg": np.nan,
        "sigma_max_J": np.nan,
        "augmented_residual_norm": np.nan,
        "augmented_rank": None,
        "condition_number": np.nan,
    }


def solve_tikhonov(
    J: np.ndarray,
    rhs: np.ndarray,
    *,
    alpha: float | None = None,
    lam: float | None = None,
    sigma_max: float | None = None,
):
    """Solve ``J @ dx = rhs`` with Tikhonov (ridge) regularization.

    Parameters
    ----------
    J : numpy.ndarray, shape (m, n)
        Finite real coefficient matrix. The Newton use case is a square real
        Jacobian; rectangular matrices are also supported.
    rhs : numpy.ndarray, shape (m,)
        Finite real right-hand-side vector.
    alpha : float or None, keyword-only
        Non-negative scale-aware regularization strength. The actual ridge
        parameter is ``lambda_reg = alpha * sigma_max(J) ** 2``. Mutually
        exclusive with ``lam``. Defaults to ``1e-12`` if neither ``alpha``
        nor ``lam`` is given (see module docstring: not a recommended
        choice, only the most conservative point of the project's sweep
        grid).
    lam : float or None, keyword-only
        Non-negative ridge parameter supplied directly, bypassing the
        ``alpha`` scaling. Mutually exclusive with ``alpha``. Useful for
        reproducing an exact lambda value independent of ``sigma_max(J)``.
    sigma_max : float or None, keyword-only
        Precomputed spectral norm (largest singular value) of ``J``, reused
        instead of recomputing it internally. Only consulted when ``alpha``
        drives the regularization (i.e. ``lam`` is not given); ignored
        otherwise. Passing this avoids a repeated O(n^3) SVD when a caller
        (e.g. a lambda sweep, or the shared Newton loop's own conditioning
        diagnostics) has already computed it for the same ``J``.

    Returns
    -------
    dx : numpy.ndarray or None, shape (n,)
        Regularized least-squares update. ``None`` denotes invalid/
        non-finite numerical input, a failed factorization, or a non-finite
        computed result.
    diagnostics : dict
        Always includes ``method`` (``"tikhonov"``), ``success``,
        ``error_message``, ``runtime_sec``, ``residual_norm`` (of the
        *original* problem, ``||J @ dx - rhs||``, not the augmented one),
        ``step_norm``, the actually-used ``alpha`` and ``lambda_reg``,
        ``sigma_max_J`` (the value used to derive ``lambda_reg`` from
        ``alpha``, ``nan`` when ``lam`` was given directly), the augmented
        system's own residual norm and LAPACK-reported rank, and an
        informational raw ``condition_number`` of ``J`` (unregularized,
        i.e. the quantity Tikhonov is meant to counteract, not the better-
        conditioned augmented system).

    Raises
    ------
    ValueError
        If dimensions are incompatible, both ``alpha`` and ``lam`` are
        given, either is negative, or a tolerance/scalar is non-finite or
        non-scalar. These are caller-contract errors; non-finite array
        values instead return ``(None, diagnostics)``.
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

    if alpha is not None and lam is not None:
        raise ValueError("alpha and lam are mutually exclusive; supply at most one.")

    try:
        alpha_used = None if alpha is None else float(alpha)
        lam_given = None if lam is None else float(lam)
        sigma_max_given = None if sigma_max is None else float(sigma_max)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha, lam, and sigma_max must be real scalars.") from exc

    for name, value in (("alpha", alpha_used), ("lam", lam_given), ("sigma_max", sigma_max_given)):
        if value is not None and not np.isfinite(value):
            raise ValueError(f"{name} must be finite.")
        if value is not None and value < 0.0:
            raise ValueError(f"{name} must be non-negative.")

    if lam_given is None and alpha_used is None:
        alpha_used = 1e-12

    # Promote integer/single-precision inputs, matching the SVD/RRQR
    # solvers' convention for consistent double-precision arithmetic.
    dtype = np.result_type(J_array.dtype, rhs_array.dtype, np.float64)
    J_work = np.asarray(J_array, dtype=dtype)
    rhs_work = np.asarray(rhs_array, dtype=dtype)
    if not np.all(np.isfinite(J_work)) or not np.all(np.isfinite(rhs_work)):
        diagnostics = _failure_diagnostics(t0, "J and rhs must contain only finite values.")
        diagnostics.update(alpha=alpha_used if alpha_used is not None else np.nan)
        return None, diagnostics

    m, n = J_work.shape

    if lam_given is not None:
        lambda_reg = lam_given
        sigma_max_J = np.nan
        alpha_reported = np.nan
    else:
        try:
            sigma_max_J = (
                sigma_max_given
                if sigma_max_given is not None
                else float(np.linalg.norm(J_work, ord=2))
            )
        except np.linalg.LinAlgError as exc:
            diagnostics = _failure_diagnostics(t0, f"Computing sigma_max(J) failed: {exc}")
            diagnostics.update(alpha=alpha_used)
            return None, diagnostics
        if not np.isfinite(sigma_max_J):
            diagnostics = _failure_diagnostics(t0, "sigma_max(J) is non-finite.")
            diagnostics.update(alpha=alpha_used)
            return None, diagnostics
        lambda_reg = alpha_used * sigma_max_J ** 2
        alpha_reported = alpha_used

    condition_number = float(np.linalg.cond(J_work)) if m == n else np.nan

    # Augmented least-squares system. This is the numerically preferred
    # formulation: it never forms J.T @ J (which would square kappa(J)) and
    # LAPACK's SVD-based gelsd driver solves it directly.
    sqrt_lambda = np.sqrt(lambda_reg)
    J_aug = np.vstack([J_work, sqrt_lambda * np.eye(n, dtype=dtype)])
    rhs_aug = np.concatenate([rhs_work, np.zeros(n, dtype=dtype)])

    try:
        dx, aug_residues, aug_rank, _ = np.linalg.lstsq(J_aug, rhs_aug, rcond=None)
    except np.linalg.LinAlgError as exc:
        diagnostics = _failure_diagnostics(t0, f"Augmented least-squares solve failed: {exc}")
        diagnostics.update(
            alpha=alpha_reported,
            lambda_reg=lambda_reg,
            sigma_max_J=sigma_max_J,
            condition_number=condition_number,
        )
        return None, diagnostics

    if not np.all(np.isfinite(dx)):
        diagnostics = _failure_diagnostics(
            t0, "Tikhonov least-squares solve produced a non-finite update."
        )
        diagnostics.update(
            alpha=alpha_reported,
            lambda_reg=lambda_reg,
            sigma_max_J=sigma_max_J,
            augmented_rank=int(aug_rank),
            condition_number=condition_number,
        )
        return None, diagnostics

    augmented_residual_norm = float(np.linalg.norm(J_aug @ dx - rhs_aug))

    diagnostics = {
        "method": "tikhonov",
        "success": True,
        "error_message": None,
        "runtime_sec": time.perf_counter() - t0,
        "residual_norm": float(np.linalg.norm(J_work @ dx - rhs_work)),
        "step_norm": float(np.linalg.norm(dx)),
        "alpha": alpha_reported,
        "lambda_reg": float(lambda_reg),
        "sigma_max_J": sigma_max_J,
        "augmented_residual_norm": augmented_residual_norm,
        "augmented_rank": int(aug_rank),
        "condition_number": condition_number,
    }
    return dx, diagnostics
