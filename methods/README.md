# methods/ — how each linear solver actually works, from scratch

Four self-contained explainers, one per solver compared in this project.
Each builds the underlying linear algebra up from first principles, then
walks through this repo's actual implementation line-by-line, then
summarizes what this project found empirically about that method's
strengths/weaknesses.

- [**DIRECT_SOLVE.md**](DIRECT_SOLVE.md) — Gaussian elimination / LU
  decomposition, and the condition-number guard that catches its silent
  failure mode. The baseline every other method is compared against.
- [**SVD_PSEUDOINVERSE.md**](SVD_PSEUDOINVERSE.md) — Singular Value
  Decomposition and the Moore-Penrose pseudo-inverse. **This is the method
  the base paper (Jang, Kim & Kim, IEEE Access 2023) proposes**; this
  project reproduces it here, then asks whether it's actually the best
  universal fix.
- [**RRQR.md**](RRQR.md) — Rank-Revealing QR (column-pivoted QR + LAPACK's
  GELSY complete orthogonal factorization). A different route to
  essentially the same kind of minimum-norm answer as SVD.
- [**TIKHONOV.md**](TIKHONOV.md) — Tikhonov (ridge) regularization. Instead
  of discarding the undetermined direction, damps the whole problem so
  nothing is ever left truly free.

All four share the same call contract (`(dx, diagnostics) = solve_xxx(J, rhs, ...)`)
and are dispatched from exactly one place in the codebase:
`solve_linear_step()` in [`src/powerflow/newton.py`](../src/powerflow/newton.py).
See [STRUCTURE.md](../STRUCTURE.md) for how that fits into the shared
Newton-Raphson loop, and [EXPERIMENTS.md](../EXPERIMENTS.md) for which
experiment scripts use which method on which feeder.

**The one-line summary of this project's headline finding**: no method
wins universally — Tikhonov is the only one that converges on IEEE-13 at
realistic load, while SVD and RRQR converge cleanly (and Tikhonov fails)
on IEEE-37. Each method-specific file above explains *why*, mechanically.
