# Tikhonov (Ridge) Regularization — from scratch

Implementation: [`src/solvers/tikhonov.py`](../src/solvers/tikhonov.py)

Unlike SVD and RRQR, which *discard* the undetermined direction(s) and
answer using only what's left, Tikhonov takes a different philosophy:
**never throw anything away — instead, gently nudge the whole problem so
that no direction is ever truly free.**

---

## 1. The idea: penalize big answers, a little, everywhere

Instead of solving `J·Δx = rhs` exactly, solve a *modified* problem:

```
minimize   ‖J·Δx − rhs‖²  +  λ·‖Δx‖²
```

Read this in two parts:
- `‖J·Δx − rhs‖²` — the ordinary "how well does this satisfy the
  equations" term (this alone is what a normal least-squares solve
  minimizes).
- `λ·‖Δx‖²` — an added **penalty for the answer being large**, with `λ`
  (lambda) controlling how strongly that penalty is enforced.

Why does this help with singularity? Picture the floating direction from
earlier discussions: moving along it changes `‖J·Δx − rhs‖²` by *exactly
zero* (that's what "undetermined direction" means). Normally that means
*anything* along that direction is equally acceptable to the first term —
including something enormous. But the second term, `λ·‖Δx‖²`, does **not**
stay flat along that direction — it grows as `Δx` grows, in *every*
direction, including the floating one. So even though the first term
doesn't care, the second term does, and it specifically prefers the
**smallest** possible `Δx` in that direction — nudging the otherwise-free
value toward zero rather than leaving it truly unconstrained. Every other
(well-determined) direction is affected too, but only slightly, since `λ`
is chosen to be small relative to how strongly those directions are
already pinned down.

## 2. Solving it — the "obvious" way, and why this project refuses to use it

Calculus (setting the gradient of the objective above to zero) gives the
**normal equations**:

```
(Jᵀ·J + λ·I) · Δx = Jᵀ·rhs
```

This looks like an ordinary linear solve on a new matrix
`Jᵀ·J + λ·I` — tempting, but **this project deliberately never does
this**, and the reason is itself a piece of numerical-analysis reasoning
worth understanding:

**Forming `Jᵀ·J` squares the condition number.** If `J`'s singular values
are `σ_1, ..., σ_n`, then `Jᵀ·J`'s eigenvalues are exactly `σ_1², ..., σ_n²`
(this follows directly from the SVD: `Jᵀ·J = V·Σᵀ·Uᵀ·U·Σ·Vᵀ = V·Σ²·Vᵀ`,
since `Uᵀ·U = I`). So:

```
κ(JᵀJ) = σ_max² / σ_min² = κ(J)²
```

If `J` already has `κ₂ = 1e6` (already uncomfortable), `JᵀJ` would have
`κ₂ = 1e12` — squaring exactly the problem this project studies, not
fixing it. Adding `λ·I` afterward helps, but you've already thrown away
precision you can't get back. This project's Tikhonov implementation is
built specifically to avoid ever forming this matrix.

## 3. The fix: solve an equivalent "augmented" system instead

There's a way to get the *exact same* `Δx` that the normal equations would
give, without ever forming `Jᵀ·J`. Stack two blocks on top of each other:

```
     ⎡    J     ⎤          ⎡ rhs ⎤
     ⎣ √λ · I   ⎦ · Δx  ≈  ⎣  0  ⎦
```

Call the stacked matrix `J_aug` and the stacked right-hand side `rhs_aug`.
Solving this as an **ordinary** (unregularized) least-squares problem —
minimize `‖J_aug·Δx − rhs_aug‖²` — gives exactly the ridge-regularized
answer, because:

```
‖J_aug·Δx − rhs_aug‖²
  = ‖J·Δx − rhs‖²  +  ‖√λ·I·Δx − 0‖²
  = ‖J·Δx − rhs‖²  +  λ·‖Δx‖²
```

— which is *precisely* the objective from §1. The algebra "unpacks" the
penalty term into extra rows of a bigger, but perfectly ordinary,
least-squares problem — solvable with a standard (SVD-based, so itself
robust to rank-deficiency) least-squares routine, `numpy.linalg.lstsq`,
with **no squaring of any condition number anywhere** in the process.

## 4. Choosing `λ` in a scale-aware way

```
λ = α · σ_max(J)²
```

`α` (a small dial, e.g. `1e-8`) is what a user actually chooses; `λ` itself
is derived by scaling `α` against the *strongest* singular value of the
current `J`. Why: `J`'s overall scale can vary a lot — between different
iterations of the same Newton-Raphson run, or between entirely different
feeders with different per-unit bases. If `λ` were a fixed raw number, the
"same" `λ` might be a huge, distorting penalty on a small-scale problem and
a negligible one on a large-scale problem. Scaling by `σ_max(J)²` keeps
`α`'s *effective strength* comparable across all of them — the docstring
calls this the "scale-aware convention."

## 5. A worked tiny example — every matrix, every multiplication

```
J = [[1, 1],
     [1, 1]],   rhs = [4, 4]
```
(`σ_max = 2`, from [SVD_PSEUDOINVERSE.md](SVD_PSEUDOINVERSE.md)'s worked
example). Pick `α = 0.01`, so `λ = α · σ_max² = 0.01 × 2² = 0.04`,
`√λ = √0.04 = 0.2`.

### Step 1 — build the augmented matrix (literally stack the blocks)

```
J_aug = [[1,    1  ],       rhs_aug = [4]
         [1,    1  ],                 [4]
         [0.2,  0  ],                 [0]
         [0,    0.2]]                 [0]
```
The top 2 rows are just `J` itself; the bottom 2 rows are `√λ · I` — a
`0.2` on the diagonal, literally appended as two extra equations, each
saying (in effect) "and also, `Δx_i` should be small."

### Step 2 — form the normal equations *of the augmented system* (this is safe — see why below)

```
J_augᵀ · J_aug =
  [[1·1+1·1+0.2·0.2+0·0,   1·1+1·1+0.2·0+0·0.2],     =  [[2.04, 2.00],
   [1·1+1·1+0.2·0+0·0.2,   1·1+1·1+0·0+0.2·0.2]]         [2.00, 2.04]]

J_augᵀ · rhs_aug =
  [1·4+1·4+0.2·0+0·0,   1·4+1·4+0·0+0.2·0]  =  [8.0, 8.0]
```

**Why this is safe here, even though §2 forbids forming `JᵀJ`:** the
danger in §2 was squaring `J`'s *own* condition number. But
`J_augᵀ·J_aug` is only a `2×2` matrix built from the *augmented* system,
which was deliberately constructed to have **no near-zero singular value
in the first place** (the `√λ·I` block guarantees every direction has at
least some weight). Forming normal equations of an already-well-posed
small system is ordinary, unremarkable linear algebra — it's only
dangerous when the matrix being squared was ill-conditioned to begin with.
(This worked example forms them explicitly only to show the arithmetic in
full; the actual code in `tikhonov.py` still avoids it, using `lstsq`
instead, precisely so this reasoning never has to be re-verified for every
different `J` it's given.)

### Step 3 — solve the resulting well-posed 2×2 system directly (ordinary elimination, §2 of DIRECT_SOLVE.md)

```
[[2.04, 2.00],     [Δx_1]   [8.0]
 [2.00, 2.04]]  ·  [Δx_2] = [8.0]
```
Eliminate: multiplier `= 2.00/2.04 = 0.9803921569`, do `row2 − multiplier·row1`:
```
row2_new = [2.00 - 0.9803921569×2.04,   2.04 - 0.9803921569×2.00]  =  [0,  0.0792156863]
rhs2_new =  8.0  - 0.9803921569×8.0                                 =  0.1568627451
```
Back-substitute:
```
Δx_2 = rhs2_new / row2_new[1] = 0.1568627451 / 0.0792156863 = 1.9801980198

row 1:  2.04·Δx_1 + 2.00·Δx_2 = 8.0
        Δx_1 = (8.0 - 2.00×1.9801980198) / 2.04 = 1.9801980198
```
`Δx_1 = Δx_2 = 1.9801980198...` — both come out identical, by the matrix's
symmetry (unsurprising: `J`'s two columns were identical to begin with).

### Step 4 — verify

```
J_aug @ [1.9801980198, 1.9801980198] =
  [1.9801980198+1.9801980198,  1.9801980198+1.9801980198,  0.2×1.9801980198,  0.2×1.9801980198]
  = [3.9603960396,  3.9603960396,  0.3960396040,  0.3960396040]
```
Compare to `rhs_aug = [4, 4, 0, 0]` — close but not exact (that's the whole
point: this is a *least-squares* solve, minimizing total squared error
across all 4 equations at once, not solving any of them exactly). On the
**original** problem specifically: `J @ [1.9802, 1.9802] = [3.9604, 3.9604]`
vs. `rhs = [4, 4]` — off by `0.0396` in each entry, giving
`residual_norm = ‖[3.9604,3.9604]-[4,4]‖ = √(0.0396²×2) ≈ 0.056`, matching
the code's own reported diagnostic below exactly.

**Verified against this repo's actual code** —
`solve_tikhonov(J, rhs, alpha=0.01)` returns:

```
dx = [1.98019802, 1.98019802]        <- slightly BELOW SVD's [2, 2], in BOTH entries, not just the "free" one
  alpha:                    0.01
  lambda_reg:               0.04                 <- = alpha * sigma_max(J)^2 = 0.01 * 2^2
  sigma_max_J:              2.0
  residual_norm:            0.056                <- ||J@dx - rhs|| on the ORIGINAL problem: no longer exactly zero!
  augmented_residual_norm:  0.563                <- what was actually minimized (original residual + penalty term)
  augmented_rank:           2                    <- J_aug is FULL rank (well-posed), unlike J itself (rank 1)
  step_norm:                2.800                <- vs SVD's step_norm of 2.828 -- Tikhonov's answer is slightly smaller
```

Notice `residual_norm` is no longer `~1e-15` the way SVD's and RRQR's were
— it's `0.056`, a real, nonzero amount. That's the price of damping: by
refusing to leave *any* direction totally unconstrained, Tikhonov also
nudges the well-determined "sum" direction very slightly away from
perfectly satisfying `4 = Δx_1 + Δx_2` (`1.980+1.980 = 3.960`, not exactly
`4`). With `α = 0.01` that price is tiny; as `α → 0` this answer converges
back to the exact `[2, 2]` SVD gave, and as `α` grows the price keeps
rising. That trade-off — some damping helps stability, too much damping
stops solving the real problem — is exactly why this project treats `α` as
something to be *swept and justified* per network (see
[experiments/exp_d_ieee13_tikhonov_alpha_sweep.py](../experiments/exp_d_ieee13_tikhonov_alpha_sweep.py)),
never picked arbitrarily.

*(Reproduce yourself: `python -c "from src.solvers.tikhonov import solve_tikhonov; import numpy as np; print(solve_tikhonov(np.array([[1.,1.],[1.,1.]]), np.array([4.,4.]), alpha=0.01))"`.)*

## 6. What `solve_tikhonov()` actually does ([tikhonov.py:56-237](../src/solvers/tikhonov.py#L56-L237))

```python
if alpha is not None and lam is not None:
    raise ValueError("alpha and lam are mutually exclusive; supply at most one.")
```
Two ways to control the penalty: the scale-aware `alpha` (the normal way),
or a raw `lam` supplied directly (bypassing the `σ_max(J)²` scaling —
useful for reproducing one exact numeric `λ` independent of whatever `J`
happens to be, e.g. across a sweep). Exactly one may be given; if neither
is, it defaults to `alpha=1e-12` — explicitly documented as "the most
conservative end of the sweep grid, not a considered recommendation."

```python
lambda_reg = alpha_used * sigma_max_J ** 2
```
The scaling from §4, computed via `np.linalg.norm(J_work, ord=2)`
(the spectral norm — exactly `σ_max`) unless a precomputed value was
passed in (`sigma_max`), avoiding a redundant `O(n³)` SVD when the caller —
e.g. `newton.py`'s own conditioning diagnostics — has already computed it
for this same `J`.

```python
sqrt_lambda = np.sqrt(lambda_reg)
J_aug = np.vstack([J_work, sqrt_lambda * np.eye(n, dtype=dtype)])
rhs_aug = np.concatenate([rhs_work, np.zeros(n, dtype=dtype)])
dx, aug_residues, aug_rank, _ = np.linalg.lstsq(J_aug, rhs_aug, rcond=None)
```
Exactly the augmented-system construction from §3, solved via `lstsq`
(LAPACK's `gelsd`, itself SVD-based internally — so this step is robust
even though `J_aug` is deliberately *not* rank-deficient by construction,
since the appended `√λ·I` block guarantees every column has some nonzero
weight).

```python
"residual_norm": float(np.linalg.norm(J_work @ dx - rhs_work)),        # ORIGINAL problem's residual
"augmented_residual_norm": augmented_residual_norm,                     # the AUGMENTED system's residual
"condition_number": condition_number,                                   # of J itself (unregularized) — informational
```
Three distinct numbers, kept separate on purpose: the *original* problem's
residual (how well `Δx` actually solves `J·Δx = rhs`, ignoring the
penalty), the *augmented* system's own residual (what was actually
minimized), and `J`'s own raw condition number (what Tikhonov is meant to
counteract, not a claim about the better-conditioned augmented system).

## 7. Strengths and weaknesses (as found by this project)

- **Strength**: the *only* one of the four methods that reliably converges
  on IEEE-13 at realistic load, precisely because it deliberately gives up
  exactness in favor of a damped, less-aggressive step — avoiding the
  oscillation that SVD and RRQR both exhibit there.
- **Weakness, discovered empirically**: the reverse holds on IEEE-37 — the
  *same* `alpha=1e-8` that works well on IEEE-13 **fails there**, while
  SVD/RRQR converge cleanly. And convergence is **not monotonic** in
  `alpha` — more damping is not simply "always safer"
  ([experiments/exp_d_ieee13_tikhonov_alpha_sweep.py](../experiments/exp_d_ieee13_tikhonov_alpha_sweep.py)
  found only specific values in its sweep grid actually converge on
  IEEE-13). So Tikhonov trades "always gives an answer, even when
  singular" (true of all three non-direct methods) for "needs its own
  strength parameter tuned per-network, with no universal default" — this
  network-dependence, on both sides, is the project's central finding
  against the base paper's claim that one fixed method (SVD) is the right
  universal remedy.

See [DIRECT_SOLVE.md](DIRECT_SOLVE.md), [SVD_PSEUDOINVERSE.md](SVD_PSEUDOINVERSE.md),
and [RRQR.md](RRQR.md).
