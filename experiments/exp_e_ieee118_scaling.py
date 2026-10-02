"""IEEE-118 computational scaling benchmark (Member E, project spec section 20).

Per project spec section 20 and the 6-week plan section 10, IEEE-118 is used
ONLY as a "balanced computational scaling benchmark" for the four linear
solvers' runtime -- explicitly NOT as evidence about the ungrounded
three-phase transformer singularity this project otherwise studies (this
is a real transmission-level, effectively balanced/single-sequence
network with no floating-delta phenomenon to reproduce).

Methodology
-----------
1. Load ``pandapower.networks.case118()``, run its own power flow, and
   extract the REAL converged 118x118 complex admittance matrix (not a
   synthetic random matrix) via pandapower's internal Ybus -- this anchors
   the benchmark in a genuine, realistically-sparse power-system topology
   rather than an arbitrary dense random one.
2. To get a scaling CURVE (a single 118-bus point cannot show a trend),
   block-diagonally replicate that same real Ybus 1, 2, 4, and 8 times.
   Each replicated system is electrically several disconnected copies of
   IEEE-118 -- this deliberately preserves the real matrix's sparsity
   pattern and conditioning character at each size rather than switching
   to an unrelated random matrix, at the cost of not being a single larger
   real network.
3. Following this project's actual state representation (rectangular
   (e, f) coordinates, src/model/state.py), each complex n x n admittance
   matrix Y = G + jB is expanded to the real 2n x 2n block form
   [[G, -B], [B, G]] before benchmarking -- matching the structure (not
   the exact nonlinear entries) of a genuine linearized NR Jacobian at
   this size, so the reported sizes are directly comparable to the
   n_unknowns figures reported elsewhere in this project (e.g. IEEE-13's
   64, IEEE-37's 222).
4. Each of the four solve_* functions is timed directly (not through the
   shared Newton-Raphson loop -- per project spec section 20, this is a
   balanced network with no singularity to reproduce, so running the full
   nonlinear solver here would not exercise anything this project is
   actually about). A random real right-hand side is used since only
   solver TIME, not a physically meaningful solution, is being measured.

This is dense-matrix benchmarking (matching this project's four dense
solver implementations, all of which use numpy.linalg/scipy.linalg dense
routines); a production power-flow tool would use sparse solvers at these
sizes, and the reported times should not be read as representative of
that.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.solvers.direct import solve_direct
from src.solvers.rrqr import solve_rrqr
from src.solvers.svd_pinv import solve_svd_pinv
from src.solvers.tikhonov import solve_tikhonov

REPLICATION_FACTORS = (1, 2, 4, 8)
REPEATS = 5
TIKHONOV_ALPHA = 1e-10
ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def load_case118_ybus() -> np.ndarray:
    """Return the real, converged 118x118 complex admittance matrix."""
    import pandapower as pp
    import pandapower.networks as pn

    net = pn.case118()
    pp.runpp(net)
    if not net["converged"]:
        raise RuntimeError("pandapower case118 power flow did not converge.")
    Ybus_sparse = net._ppc["internal"]["Ybus"]
    return np.asarray(Ybus_sparse.todense())


def block_diagonal_replicate(Y: np.ndarray, factor: int) -> np.ndarray:
    """Block-diagonally repeat Y `factor` times (see module docstring, step 2)."""
    if factor == 1:
        return Y
    return sla.block_diag(*([Y] * factor))


def to_real_jacobian_form(Y: np.ndarray) -> np.ndarray:
    """[[G,-B],[B,G]] real block form, matching src/model/state.py's (e,f) layout."""
    G, B = Y.real, Y.imag
    top = np.hstack([G, -B])
    bottom = np.hstack([B, G])
    return np.vstack([top, bottom])


def median_runtime_sec(solve_fn, J: np.ndarray, rhs: np.ndarray, repeats: int) -> float:
    times = []
    for _ in range(repeats):
        _, diagnostics = solve_fn(J, rhs)
        times.append(diagnostics["runtime_sec"])
    return float(np.median(times))


def run_scaling_benchmark() -> list[dict]:
    Y118 = load_case118_ybus()
    print(f"Loaded pandapower case118 Ybus: {Y118.shape}, nnz={np.count_nonzero(Y118)}")

    rows = []
    for factor in REPLICATION_FACTORS:
        Y = block_diagonal_replicate(Y118, factor)
        J = to_real_jacobian_form(Y)
        n = J.shape[0]
        rng = np.random.default_rng(118 * factor)
        rhs = rng.standard_normal(n)

        row = {"replication_factor": factor, "n_buses_equiv": 118 * factor, "matrix_size": n}
        row["direct_median_sec"] = median_runtime_sec(solve_direct, J, rhs, REPEATS)
        row["svd_median_sec"] = median_runtime_sec(solve_svd_pinv, J, rhs, REPEATS)
        row["rrqr_median_sec"] = median_runtime_sec(solve_rrqr, J, rhs, REPEATS)
        row["tikhonov_median_sec"] = median_runtime_sec(
            lambda J_, rhs_: solve_tikhonov(J_, rhs_, alpha=TIKHONOV_ALPHA), J, rhs, REPEATS
        )
        rows.append(row)
        print(
            f"n={n:5d} (118-bus x{factor:2d})  "
            f"direct={row['direct_median_sec']:.4e}s  "
            f"svd={row['svd_median_sec']:.4e}s  "
            f"rrqr={row['rrqr_median_sec']:.4e}s  "
            f"tikhonov={row['tikhonov_median_sec']:.4e}s"
        )

    if len(rows) >= 2:
        print()
        print("Empirical scaling exponent (log-log slope between consecutive sizes):")
        for method in ("direct", "svd", "rrqr", "tikhonov"):
            key = f"{method}_median_sec"
            slopes = []
            for a, b in zip(rows[:-1], rows[1:]):
                if a[key] > 0 and b[key] > 0:
                    slopes.append(
                        np.log(b[key] / a[key]) / np.log(b["matrix_size"] / a["matrix_size"])
                    )
            if slopes:
                print(f"  {method:10s}: mean exponent {np.mean(slopes):.2f} (O(n^3) direct-solve theory ~3)")

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "ieee118_scaling_benchmark.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")
    return rows


if __name__ == "__main__":
    run_scaling_benchmark()
