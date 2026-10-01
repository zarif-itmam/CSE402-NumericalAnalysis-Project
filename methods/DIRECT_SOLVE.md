# Direct Solve (Gaussian Elimination / LU Decomposition) — from scratch

Implementation: [`src/solvers/direct.py`](../src/solvers/direct.py)

This is the "classic" method — the baseline every other solver in this
project is measured against, and the one that fails silently if you don't
add a safety check.

---

## 1. The problem

We want to solve a system of linear equations, written in matrix form:

```
J · Δx = rhs
```

`J` is an `n×n` matrix (the Jacobian, in this project), `rhs` is a known
vector, and `Δx` is the unknown vector we're solving for. If you've done
this by hand in school, you solved small versions of exactly this by
substitution — e.g.:

```
2x + y = 5
 x + 3y = 10
```

Direct solve is just doing that same substitution/elimination process,
generalized to work efficiently for any size `n`, done by a computer.

## 2. Gaussian elimination, from scratch

The classic algorithm:

1. Take the first equation. Use it to eliminate the first variable from
   every equation below it (subtract a multiple of row 1 from each other
   row so that column 1 becomes zero everywhere except row 1).
2. Move to the second equation. Use it to eliminate the second variable
   from every equation below *it*.
3. Repeat until you're left with an upper-triangular system — one where
   the last equation only involves the last variable, the second-to-last
   involves the last two, and so on.
4. Solve from the bottom up ("back-substitution"): the last equation
   directly gives you the last variable; plug that into the second-to-last
   equation to get the next one; and so on.

Worked example:
```
2x + y  = 5        2x + y      = 5
 x + 3y = 10   ->   (5/2)y     = 15/2      (row2 - 0.5*row1)
```
From row 2: `y = 3`. Substitute into row 1: `2x + 3 = 5 → x = 1`.

## 3. LU decomposition — the same idea, reusable

`numpy.linalg.solve(J, rhs)` doesn't repeat elimination from scratch if you
ever needed to solve with the same `J` again — it factors `J` once into two
triangular matrices:

```
J = L · U
```

- `L` (lower-triangular): records the *multipliers* used during
  elimination (i.e., "how much of row 1 did I subtract from row 3?").
- `U` (upper-triangular): the *result* of elimination — exactly the
  triangular system from step 3 above.

Once you have `L` and `U`, solving `J·Δx = rhs` becomes two *cheap*
triangular solves instead of one expensive elimination:
```
L·y  = rhs      (forward substitution, top to bottom)
U·Δx = y        (back substitution, bottom to top)
```
This is what "LU-based" means in the code's docstring, and it's exactly
what `numpy.linalg.solve` does under the hood.

## 4. Where this breaks down — the actual subject of this project

Elimination requires dividing by a "pivot" (the diagonal entry you're
eliminating around). If a pivot is *exactly* zero, you can't divide by it —
`numpy.linalg.solve` raises `LinAlgError` in that case. This project calls
that failure mode **`exact_singular`**.

But here's the trap this whole project studies: **a pivot is almost never
*exactly* zero in floating-point arithmetic**, even when the underlying
matrix is mathematically singular (recall: a floating Delta transformer
makes the true Jacobian singular). Instead, the pivot is some tiny nonzero
number like `1e-14` — and dividing by a tiny number doesn't crash, it just
**amplifies whatever rounding error already exists** in your inputs by a
huge factor.

### Why the condition number is the real signal

The condition number `κ₂(J) = σ_max/σ_min` (largest divided by smallest
"singular value" — see [SVD_PSEUDOINVERSE.md](SVD_PSEUDOINVERSE.md) for what
that means) tells you exactly how much this amplification is. Rule of
thumb: **you lose about `log10(κ)` decimal digits of accuracy** out of the
~16 available in double-precision floating point. If `κ₂ = 1e12`, you've
lost 12 of your 16 digits — the answer LU hands back still "solves" the
equation to a tiny residual (`‖J·Δx − rhs‖` looks small), but `Δx` itself
can be almost meaningless. **Residual size does not indicate correctness**
once `κ` is large — this is the single most important number in this
whole project.

## 5. Worked example — running the actual code, twice

Both examples below were run for real against this repo's own
`solve_direct()` (not hand-computed) — see the exact commands at the bottom
of this section.

### 5a. The `exact_singular` case

```python
J   = [[1, 1],
       [1, 1]]
rhs = [4, 4]
```

This is precisely the tiny 2-node "no anchor" system used throughout these
method files — think of it as a miniature stand-in for a floating Delta
transformer's 2-phase common-mode block: both rows say "the sum of the two
unknowns is what matters," and nothing distinguishes the unknowns from
each other individually.

Elimination: subtract row 1 from row 2 → `row2 = [1-1, 1-1] = [0, 0]`. The
second pivot is `0.0` **exactly** (no rounding involved — `1.0 - 1.0` is
exact in floating point) — so `numpy.linalg.solve` raises `LinAlgError`
immediately. Running `solve_direct(J, rhs)` gives:

```
dx = None
  condition_number:  5.96e+16      <- np.linalg.cond() still computes a number (via SVD), even though solve() itself failed
  success:           False
  failure_mode:       exact_singular
  error_message:      Singular matrix
  residual_norm:      nan          <- no dx was produced, so no residual to report
```

### 5b. The `ill_conditioned` case — every matrix, every multiplication

Now perturb that matrix *very slightly* so it's no longer exactly singular:

```python
J2   = [[1, 1],
        [1, 1 + 1e-13]]
rhs2 = [2, 2 + 1e-13]
```

**Step 1 — the LU factorization itself.** This is exactly the elimination
from §2, but now let's see the actual matrices `numpy` builds internally
(via `scipy.linalg.lu`, same algorithm `numpy.linalg.solve` uses). No row
swaps are needed here (the permutation `P` is just the identity):

```
L (records "how much of row 1 I subtracted from row 2") =
  [[1, 0],
   [1, 1]]                <- 1.0: "subtract 1x row 1 from row 2"

U (upper-triangular RESULT of that elimination) =
  [[1,              1            ],
   [0,   9.992007221626409e-14   ]]     <- the tiny surviving pivot (~1e-13, NOT exactly the
                                             1e-13 we typed in, because 1e-13 itself isn't
                                             exactly representable in binary floating point)
```

**Verify it's a real factorization** — multiply `L @ U` back together and
you get `J2` exactly:
```
L @ U = [[1*1+0*0,        1*1+0*9.99e-14      ]     = [[1, 1                ],
         [1*1+1*0,        1*1+1*9.99e-14      ]]       [1, 1.0000000000001]]   ==  J2  ✓
```

**Step 2 — forward substitution, solve `L·y = rhs2` (top to bottom).**
```
row 1:  1*y1 = 2                       ->  y1 = 2
row 2:  1*y1 + 1*y2 = 2 + 1e-13        ->  y2 = (2 + 1e-13) - 1*(2) = 9.992007221626409e-14
```
so `y = [2, 9.992007221626409e-14]` — notice `y2` is now a number
*comparable in size* to the tiny pivot `U[1,1]` itself. That's not a
coincidence — it's the setup for the dangerous division about to happen.

**Step 3 — back substitution, solve `U·dx = y` (bottom to top).**
```
row 2:  U[1,1] * dx2 = y2
        dx2 = y2 / U[1,1] = 9.992007221626409e-14 / 9.992007221626409e-14 = 1.0

row 1:  U[0,0]*dx1 + U[0,1]*dx2 = y1
        1*dx1 + 1*1.0 = 2   ->   dx1 = 1.0
```
Final answer: `dx = [1.0, 1.0]`.

**This is the exact instant where the danger lives.** `dx2` was computed as
`(tiny number) / (another tiny number)` — both around `1e-13`, i.e. both
already close to the size of ordinary floating-point rounding error. Two
numbers that small dividing cleanly to give exactly `1.0` here is almost
lucky bookkeeping (they came from consistent rounding of the same inputs);
change `rhs2` by even a little and this division's *numerator* shifts by a
comparable tiny amount relative to a denominator that never changes — which
is exactly why the *ratio* itself can swing wildly (shown concretely
below). Contrast this with the well-conditioned `2x+y=5` example in §2,
where every pivot stayed a comfortably-sized number (`2`, `5/2`) — nothing
there was ever close to cancelling down to noise-level.

Calling this repo's actual `solve_direct(J2, rhs2)` (which computes `κ₂`
*before* trusting any of the above):

```
dx = None
  condition_number:  4.00e+13
  success:           False
  failure_mode:       ill_conditioned
  error_message:      condition number 4.005e+13 exceeds threshold 1.0e+12; ...
  residual_norm:      0.0          <- note: the residual is ZERO, yet the guard still (correctly) refuses this answer
```

**Why the guard matters — the amplification, made concrete.** If you
bypass the guard entirely and just call raw `numpy.linalg.solve(J2, rhs2)`,
you get `dx = [1, 1]`, looking perfectly reasonable. Now nudge `rhs2`'s
second entry by a genuinely tiny amount — `+1e-12`, a *relative* change of
only `5×10⁻¹³` (smaller than almost any real measurement or rounding error
you'd ever introduce):

```
original rhs2            -> dx = [ 1.000,  1.000]
rhs2 + 1e-12 nudge        -> dx = [-9.009, 11.009]
relative change in input:  5e-13
relative change in output: ~1000%  (10.0x)
```

A change smaller than one part in a trillion in the *input* produced a
**10x-scale, sign-flipping** change in the *output*. This is exactly what
"you lose `log10(κ)` digits of precision" means in practice — with
`κ ≈ 4×10¹³`, about 13 of your ~16 available digits are gone, and the
"solution" is no more trustworthy than noise. The residual staying at
`0.0` the whole time (§5b above) is precisely the trap the module docstring
warns about: **a perfect residual tells you `dx` satisfies the equation,
not that `dx` is close to any meaningful physical answer.**

*(Reproduce this yourself: `python -c "from src.solvers.direct import solve_direct; import numpy as np; print(solve_direct(np.array([[1.,1.],[1.,1.]]), np.array([4.,4.])))"` from the repo root.)*

## 6. What `solve_direct()` actually does ([direct.py:48-113](../src/solvers/direct.py#L48-L113))

```python
COND_NUMBER_FAIL_THRESHOLD = 1e12
```
A project-chosen early-warning cutoff (not from the paper) — well before
total precision loss (`~1/eps ≈ 4.5e15`), but high enough not to reject
merely "somewhat imprecise" solves.

```python
cond = float(np.linalg.cond(J))          # computed BEFORE attempting the solve
dx_raw = np.linalg.solve(J, rhs)          # the actual LU solve
```

Two distinct failure modes are logged, never conflated:

| `failure_mode` | What happened | Meaning |
|---|---|---|
| `exact_singular` | `numpy.linalg.solve` itself raised `LinAlgError` | a pivot was *exactly* zero — rare in practice |
| `ill_conditioned` | the solve "succeeded," but `κ₂(J) > 1e12` | the answer is numerically untrustworthy even though nothing crashed |

```python
if not np.isfinite(cond) or cond > cond_fail_threshold:
    ...
    return None, diagnostics   # dx is DISCARDED, even though LU "succeeded"
```
This is the crucial line: **the guard actively throws away a numerically
"valid" answer** rather than letting a caller (Newton-Raphson) silently use
a corrupted update. Without this, `newton_raphson()` could keep iterating
on garbage and eventually report `converged=True` on a meaningless voltage
solution.

## 7. Strengths and weaknesses

- **Strength**: cheapest of the four methods (`O(n³)` once, no extra
  decomposition machinery), and correct on any well-conditioned problem —
  this is why it's the baseline every other solver is compared against.
- **Weakness**: has *no way* to produce a sensible answer once `J` is
  actually singular or severely ill-conditioned — it can only detect the
  problem and refuse (which is exactly its job in this project: on IEEE-4
  Case B and on IEEE-13/37 at realistic load, it's *expected* to fail,
  and the point is that it fails loudly via the guard instead of silently).

See [SVD_PSEUDOINVERSE.md](SVD_PSEUDOINVERSE.md), [RRQR.md](RRQR.md), and
[TIKHONOV.md](TIKHONOV.md) for the three ways this project handles the case
where direct solve has to give up.
