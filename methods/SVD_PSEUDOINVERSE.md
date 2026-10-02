# SVD & the Moore-Penrose Pseudo-Inverse — from scratch

*(This is the method the base paper — Jang, Kim & Kim, IEEE Access 2023 —
proposes as its fix. This project reproduces it here and then asks whether
it's actually the best choice.)*

Implementation: [`src/solvers/svd_pinv.py`](../src/solvers/svd_pinv.py)

---

## 1. The idea: re-express any matrix as "stretch along independent directions"

Every real matrix `J` (square, rectangular, singular, whatever) can be
written as a product of three matrices:

```
J = U · Σ · Vᵀ
```

- `V` is an orthogonal matrix — its columns `v_1, v_2, ..., v_n` are
  mutually perpendicular unit vectors ("directions" in the *input* space).
- `Σ` is diagonal — entries `σ_1 ≥ σ_2 ≥ ... ≥ σ_n ≥ 0` (the **singular
  values**, always non-negative, always sorted largest-first).
- `U` is orthogonal too — its columns `u_1, u_2, ..., u_n` are perpendicular
  unit vectors in the *output* space.

**Geometric picture**: multiplying any vector by `J` is *always* equivalent
to three steps: (1) rotate/reflect via `Vᵀ`, (2) stretch each axis
independently by `σ_i` via `Σ`, (3) rotate/reflect again via `U`. This
decomposition **always exists**, for *any* matrix — square or not,
singular or not. That's what makes SVD the right tool for exactly this
project's problem: unlike LU, it never "fails" just because the matrix is
singular; it just reports some `σ_i = 0`.

Each singular value `σ_i` tells you exactly how much the matrix "listens
to" the corresponding input direction `v_i`: if `σ_i` is large, that
direction is strongly, reliably determined by the equations. If `σ_i` is
tiny or exactly zero, that direction is the "floating" one this whole
project is about — nudging along `v_i` produces almost no change in the
output at all (recall the null-space / graph-Laplacian discussion: `C_DELTA`'s
`(1,1,1)` null vector shows up here as a `σ_i` at or near zero).

## 2. Why you can't just "invert" `Σ` naively

If `J` were invertible, `J⁻¹ = V · Σ⁻¹ · Uᵀ`, where `Σ⁻¹` just has `1/σ_i`
on the diagonal. But if any `σ_i = 0`, `1/σ_i` is undefined — this is
*literally* singularity, expressed through the SVD lens.

## 3. The fix: the Moore-Penrose pseudo-inverse

Instead of inverting every singular value, **only invert the ones that are
big enough to trust, and set the reciprocal of everything else to exactly
zero**:

```
σ_i⁺ = 1/σ_i   if σ_i > τ  (trustworthy — invert it)
σ_i⁺ = 0        if σ_i ≤ τ  (too close to zero — throw this direction away)
```

where `τ` is a chosen cutoff ("how small counts as basically zero"). Then:

```
Δx = V · Σ⁺ · Uᵀ · rhs
```

This `Δx` has two provable properties that make it the standard, principled
answer to "what do I do when the equations don't fully determine the
answer":
- Among all vectors that best satisfy `J·Δx ≈ rhs` (in the least-squares
  sense — smallest possible `‖J·Δx − rhs‖`), it picks the one with the
  **smallest possible `‖Δx‖`** ("minimum-norm"). In the discarded
  directions, it doesn't "guess" — it deliberately contributes exactly
  zero.
- If `J` is actually invertible (no small singular values at all), this
  formula reduces to the ordinary inverse — it's a strict generalization,
  not a different method that happens to agree sometimes.

## 4. A worked tiny example — every matrix, every multiplication

Take a deliberately singular 2×2 matrix (analogous in spirit to
`C_DELTA`'s rank-2-ness, just scaled down to 2×2 for hand computation):
```
J = [[1, 1],
     [1, 1]],     rhs = [4, 4]
```

### Step 1 — the actual SVD (real numbers from `np.linalg.svd(J)`, not rounded)

```
U  = [[-0.7071067812, -0.7071067812],
      [-0.7071067812,  0.7071067812]]

Σ  = diag(2.0, 3.3547044514e-17)          <- second value is NOT exactly 0 in floating point, but is to every
                                               digit that matters — this IS the floating common-mode direction

Vᵀ = [[-0.7071067812, -0.7071067812],
      [ 0.7071067812, -0.7071067812]]
```
`0.7071067812 ≈ 1/√2` — both `U` and `Vᵀ`'s rows are just `±(1,1)/√2` and
`±(1,−1)/√2`, exactly the "sum direction" / "difference direction" pair
predicted analytically. (`U = V` here, up to sign — a special feature of
this particular matrix being *symmetric*; for a general Jacobian, which
isn't symmetric, `U` and `V` are genuinely different rotations of the
input space vs. the output space.)

**Verify it's a real factorization** — multiply `U · Σ · Vᵀ` back together,
entry by entry, and it must reconstruct `J` exactly:
```
U @ diag(Σ) @ Vᵀ =
  [[(-0.7071)(2)(-0.7071) + (-0.7071)(3.35e-17)(0.7071),   (-0.7071)(2)(-0.7071) + (-0.7071)(3.35e-17)(-0.7071)],
   [(-0.7071)(2)(-0.7071) + ( 0.7071)(3.35e-17)(0.7071),   (-0.7071)(2)(-0.7071) + ( 0.7071)(3.35e-17)(-0.7071)]]
  ≈ [[1.0, 1.0],
     [1.0, 1.0]]                                    ==  J   ✓ (the tiny σ_2 term contributes ~1e-17, negligible)
```

### Step 2 — threshold the singular values, build Σ⁺

```
τ = max(atol, rtol · σ_max) = max(0, 4.44e-16 · 2.0) = 8.88e-16

σ_1 = 2.0            > τ  ->  keep it:  σ_1⁺ = 1/2.0 = 0.5
σ_2 = 3.3547e-17      ≤ τ  ->  discard:  σ_2⁺ = 0        (do NOT compute 1/3.35e-17 — that's the whole point)

Σ⁺ = diag(0.5, 0.0)
```

### Step 3 — the three-stage multiplication that produces `Δx`

This is `Δx = V · Σ⁺ · Uᵀ · rhs`, done **left to right, one matrix at a
time** — never forming a combined "pseudo-inverse matrix" first:

**3a. `Uᵀ · rhs`** (rotate the right-hand side into the U-basis):
```
Uᵀ @ [4, 4] = [(-0.7071)(4) + (-0.7071)(4),   (-0.7071)(4) + (0.7071)(4)]
            = [-5.6568542495,                  4.4408920985e-16]
```
Notice the second entry came out `≈ 0` (`4.44e-16`, pure floating-point
noise) — that's *rhs* telling you it has essentially no component at all
along the undetermined "difference" direction, which makes sense: both
entries of `rhs` are identical (`4, 4`), so `rhs` points *purely* along the
"sum" direction to begin with.

**3b. multiply elementwise by `Σ⁺`** (scale each direction by its inverse
strength — zero for the discarded one):
```
Σ⁺ · [-5.6568542495, 4.44e-16] = [0.5 × -5.6568542495,   0.0 × 4.44e-16]
                                = [-2.8284271247,         0.0]
```

**3c. `V · (that result)`** (rotate back from the singular-value basis into
the original `x`-space; recall `V = Vᵀᵀ`, i.e. columns of `V` are the *rows*
of `Vᵀ` above):
```
V @ [-2.8284271247, 0.0] = [(-0.7071)(-2.8284271247) + (0.7071)(0.0),
                            (-0.7071)(-2.8284271247) + (-0.7071)(0.0)]
                         = [2.0000000000, 2.0000000000]
```

**Final answer: `Δx = [2.0, 2.0]`** — matching this repo's actual
`solve_svd_pinv(J, rhs)` output exactly:
```
dx = [2. 2.]
  residual_norm:               1.26e-15   <- J@dx - rhs, essentially zero: the "sum" equation is fully satisfied
  condition_number:            5.96e+16   <- RAW: sigma_max / sigma_min (huge — J is nearly exactly singular)
  effective_condition_number:  1.0        <- EFFECTIVE: sigma_max / (smallest RETAINED sigma) = 2.0/2.0
```
Notice the last two lines: the **raw** condition number screams "this
matrix is a disaster" (`5.96e16`), while the **effective** one — computed
only over the direction actually used — is a perfectly healthy `1.0`. SVD
doesn't just detect the danger the way the direct solver's guard does
(§4-5 of [DIRECT_SOLVE.md](DIRECT_SOLVE.md)) — step 3b above *physically
zeroed out* the dangerous direction before it could ever reach the answer,
then reports how healthy what's left actually is.

*(Reproduce yourself: `python -c "import numpy as np; J=np.array([[1.,1.],[1.,1.]]); U,s,Vh=np.linalg.svd(J); print('U=',U); print('s=',s); print('Vh=',Vh)"` to see the raw decomposition, or `python -c "from src.solvers.svd_pinv import solve_svd_pinv; import numpy as np; print(solve_svd_pinv(np.array([[1.,1.],[1.,1.]]), np.array([4.,4.])))"` for the full solver.)*

## 5. What `solve_svd_pinv()` actually does ([svd_pinv.py:39-192](../src/solvers/svd_pinv.py#L39-L192))

```python
U, singular_values, Vh = np.linalg.svd(J_work, full_matrices=False)
```
Computes the decomposition directly — deliberately **not** calling
`numpy.linalg.pinv()`, so every decision below is visible and logged,
rather than hidden inside a library call.

```python
rank_tolerance = max(atol_used, rtol_used * sigma_max)
retained = singular_values > rank_tolerance
```
The cutoff `τ` is `max(atol, rtol · σ_max)` — a combination of an absolute
floor and a floor relative to the *strongest* direction (so the cutoff
scales sensibly regardless of the matrix's overall units/magnitude).
Default `rtol = eps · max(shape)` (machine epsilon times matrix size — the
standard textbook rule, same one used in `newton.py`'s own
`_numerical_rank()` diagnostic). Note strict `>`: a singular value sitting
*exactly* at the threshold is deliberately truncated, not kept.

```python
reciprocal_singular_values[retained] = 1.0 / singular_values[retained]
dx = Vh.conj().T @ (reciprocal_singular_values * (U.conj().T @ rhs_work))
```
This is the pseudo-inverse formula from §3, written out explicitly
(`.conj().T` handles the complex case; for the real Jacobians this project
actually uses, that's just an ordinary transpose).

```python
raw_condition_number = ... sigma_max / sigma_min                      # uses the SMALLEST reported σ
effective_condition_number = ... sigma_max / smallest RETAINED σ      # uses the smallest KEPT σ
```
Two different condition numbers are reported, and the module is careful
never to conflate them: the **raw** one reflects how singular `J` truly is;
the **effective** one reflects how well-conditioned the *retained* subspace
is (which is what actually matters for the reliability of `Δx`, since the
discarded directions were zeroed out deliberately, not because they were
merely "risky").

A rank-deficient result (`numerical_rank < n`) is **not** treated as a
failure by this function — that's expected and normal whenever the
Jacobian genuinely has a floating direction; the function's job is just to
handle it in a principled way, not to flag it as an error.

## 6. Strengths and weaknesses (as found by this project)

- **Strength**: always produces *a* well-defined answer, even for an
  exactly singular `J` — no crash, no arbitrary garbage, a specific,
  reproducible, minimum-norm choice.
- **Weakness, discovered empirically in this project**: the pseudo-inverse
  step is the *exact* solution to the *linearized* problem at each Newton
  iteration — but on IEEE-13's more severely (2-D) near-null Jacobian at
  realistic load, this exact-but-undamped step **overshoots and oscillates**
  rather than converging (see [experiments/exp_d_ieee13_investigation.py](../experiments/exp_d_ieee13_investigation.py)'s
  module docstring and `solver_robustness_study()`). On IEEE-37, by
  contrast, SVD converges cleanly — so the pseudo-inverse's reliability
  here depends on the specific *shape* of the singularity, not merely
  "singular vs. not," which is exactly this project's central finding
  against the base paper's universal recommendation.

See [DIRECT_SOLVE.md](DIRECT_SOLVE.md), [RRQR.md](RRQR.md) (a different
route to almost the same kind of answer), and [TIKHONOV.md](TIKHONOV.md)
(the damped alternative that avoids the oscillation, at the cost of
needing its own strength parameter tuned per network).
