"""Small shared reporting helpers for Member-B-owned experiments only."""

from __future__ import annotations


def summarise_nr_history(result: dict) -> dict:
    """Extract common reporting metrics from a shared Newton-Raphson result.

    Parameters
    ----------
    result : dict
        Return value of ``src.powerflow.newton.newton_raphson``.  The helper
        does not modify it and assumes a solver invocation is represented by a
        history entry containing ``solver_diagnostics``.

    Returns
    -------
    dict
        ``history_entries``, ``linear_solves`` (the original entry dicts),
        ``successful_linear_updates``, and terminal ``final_F_inf``.  The
        latter is ``nan`` only for an empty history.

    Notes
    -----
    This intentionally contains no solver/model logic.  It centralizes the
    repeated Member-B experiment reporting convention that a terminal
    convergence check is a history entry but not a linear solve.
    """
    history = result["history"]
    linear_solves = [entry for entry in history if "solver_diagnostics" in entry]
    return {
        "history_entries": len(history),
        "linear_solves": linear_solves,
        "successful_linear_updates": sum(
            entry["solver_diagnostics"].get("success", False)
            for entry in linear_solves
        ),
        "final_F_inf": history[-1]["F_inf_norm"] if history else float("nan"),
    }
