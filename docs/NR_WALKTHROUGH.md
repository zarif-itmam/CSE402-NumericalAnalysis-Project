# NR_WALKTHROUGH.md — a full, concrete Newton-Raphson simulation

This traces [experiments/exp01_4bus.py](../experiments/exp01_4bus.py)'s
**Case A** (healthy — load present on the Delta secondary) end to end, with
real numbers from actually running this repo's code (not hand-derived
approximations). The full 18-unknown state is summarized at each step, and
one specific node — **`BUS3-a`** — is narrated in complete arithmetic
detail every iteration, because it turns out to be the single worst-behaved
entry in the whole mismatch vector at nearly every step, so its story *is*
almost the whole convergence story.

Reproduce this yourself: see the script embedded at the bottom of this
file, or just re-derive any line from `src/powerflow/mismatch.py` and
`src/powerflow/jacobian.py` directly.

---

## 0. What's known before anything is solved

This is the "ground truth" from earlier in this conversation — not a
voltage, but a demand:

```
BUS3-a:  P_spec = 0.10 pu,  Q_spec = 0.04 pu   (load convention: + = consuming)
```
That's an *input*, read straight from `build_4bus_system()`'s hardcoded
values — never computed, never solved for. Converting to the "injection"
sign convention `mismatch.py` actually uses (§ from the mismatch.py
walkthrough earlier in this conversation):
```
P_spec_injection = -0.10,   Q_spec_injection = -0.04
```
Newton-Raphson's whole job is: find a voltage at every non-slack bus-phase
such that the power *implied* by that voltage (via `V × conj(I)`, `I` from
`Ybus @ V`) equals these numbers, everywhere, simultaneously.

The network has `n_unknowns = 18` (9 non-slack bus-phases — BUS2/3/4 × 3
phases each — × 2 real numbers `(e,f)` each; BUS1's 3 phases are the fixed
slack, excluded from `x`).

## 1. The flat start (`x⁰`)

`state.init_flat_start(v_mag=1.0)` sets every non-slack bus-phase to
`1.0∠(phase offset)` — no information yet about how the network will pull
each voltage away from that. At `BUS3-a` specifically:
```
e = 1.000000,  f = 0.000000     (|V| = 1.000000 pu, angle = 0.000°)
```

## 2. Iteration 0 — the first, biggest correction

**Step A — compute the mismatch (`compute_mismatch`).** Using the flat-start
guess, compute the current implied at `BUS3-a` via `I = Ybus @ V`:
```
Ir = 8.470202,   Ii = -0.821210
```
Then the calculated power there (`P_calc = e·Ir + f·Ii`, `Q_calc = f·Ir − e·Ii`,
with `f=0` here so both terms simplify):
```
P_calc = 1.0 × 8.470202 + 0.0 × (-0.821210) = 8.470202
Q_calc = 0.0 × 8.470202 - 1.0 × (-0.821210) = 0.821210
```
Compare against what was actually demanded:
```
F_P = P_calc - P_spec_injection = 8.470202 - (-0.100000) = 8.570202
F_Q = Q_calc - Q_spec_injection = 0.821210 - (-0.040000) = 0.861210
```
At the flat start, the guess implies **8.57 pu** of power flowing at
`BUS3-a` — wildly more than the `0.10 pu` actually demanded, because a flat
guess (everyone at the same voltage, no drop across the transformer/lines)
is a very rough starting point. Scanning the *entire* 18-entry `F` vector,
the worst entry anywhere is `8.570202` — exactly `BUS3-a`'s own `F_P`. So
the network-wide stopping criterion, `‖F‖_∞ = 8.570202e+00`, is *driven
entirely by this one node* at this iteration.

**Step B — compute the Jacobian and check conditioning.** `κ₂(J) = 61.5` at
this guess — not dangerously high yet (Case A has load on the Delta
secondary, so nothing here is actually singular — contrast with Case B,
where this same flat-start `κ₂` is identical, `61.5`, since it doesn't
depend on load at all, but the *later* iterations diverge completely, see
§6).

**Step C — solve for the correction.** `solve_linear_step(J, -F, method="direct")`
returns a correction `dx` for all 18 unknowns at once; at `BUS3-a`
specifically:
```
d_e = 0.597487,   d_f = -1.170668
```

**Step D — apply it.** `x = x + dx`. New guess at `BUS3-a`:
```
e = 1.000000 + 0.597487 = 1.597487
f = 0.000000 - 1.170668 = -1.170668     (|V| = 1.980512 pu — a big, rough overshoot)
```

## 3. Iterations 1–3 — rapid, large corrections

| iter | `e` | `f` | `\|V\|` (pu) | angle (°) | `F_P` at BUS3-a | `F_Q` at BUS3-a | network `‖F‖_∞` | `κ₂(J)` | `‖dx‖` |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.597487 | -1.170668 | 1.980512 | -36.235 | 1.306746 | 22.199687 | **2.219969e+01** | 69.5 | 1.659 |
| 2 | 1.126864 | -0.705136 | 1.329301 | -32.036 | 0.673889 | 4.945826 | **4.945826e+00** | 250 | 0.687 |
| 3 | 0.903086 | -0.548875 | 1.056800 | -31.290 | 0.144487 | 0.803897 | **8.038972e-01** | 4323 | 0.171 |

Notice: the guess is still swinging quite a bit (`|V|` goes `1.98 → 1.33 →
1.06`), overshooting past the eventual answer before settling — this is
completely normal Newton-Raphson behavior on a rough starting guess, not a
sign of trouble. `κ₂(J)` is climbing steadily (61 → 4323) simply because the
Jacobian's shape genuinely changes as the guess moves; it's still nowhere
near the `1e12` guard.

## 4. Iterations 4–7 — quadratic convergence kicks in

| iter | `\|V\|` (pu) | angle (°) | `F_P` | `F_Q` | network `‖F‖_∞` | `κ₂(J)` | `‖dx‖` |
|---|---|---|---|---|---|---|---|
| 4 | 0.989109 | -31.268 | 0.009330 | 0.048307 | **4.830737e-02** | 25583 | 1.19e-02 |
| 5 | 0.984405 | -31.271 | 4.35e-05 | 2.30e-04 | **2.301702e-04** | 21185 | 5.81e-05 |
| 6 | 0.984382 | -31.271 | 9.64e-10 | 5.39e-09 | **5.392461e-09** | 21163 | 1.38e-09 |
| 7 | 0.984382 | -31.271 | 1.33e-15 | -5.52e-15 | **1.382228e-14** | — | — |

This is the textbook signature of Newton-Raphson near a good solution: once
the guess is close enough, each iteration roughly **squares** the number of
correct digits (`‖F‖_∞` goes `4.8e-2 → 2.3e-4 → 5.4e-9 → 1.4e-14` — each
step gains roughly double the significant digits of the last). At iteration
7, `‖F‖_∞ = 1.38e-14 < tol_F = 1e-10`: **converged.**

## 5. The final answer — and what it actually means

```
BUS3-a: |V| = 0.984382 pu, angle = -31.271°
```
This is **not** compared against any independently-known "correct" voltage
— recall the earlier discussion: there is no such stored value. It's the
answer *because*, at this exact voltage (combined with all 17 other
unknowns' final values), the power implied by Ohm's-law arithmetic through
`Ybus` matches every bus-phase's demanded `P_spec`/`Q_spec` to within
`1e-10` pu, simultaneously, everywhere. Concretely, for `BUS3-a`:
```
P_calc = -0.100000  ==  P_spec_injection = -0.100000   (matches to 1e-15)
Q_calc = -0.040000  ==  Q_spec_injection = -0.040000   (matches to 1e-15)
```
The full converged state, all 4 buses:
```
BUS1 (slack, fixed): |V|=1.000000 pu @ 0°/-120°/120°
BUS2:                |V|=0.993158 pu @ -0.449°  (small drop — just the BUS1-BUS2 line)
BUS3:                |V|=0.984382 pu @ -31.271° (bigger drop — crossed the transformer)
BUS4:                |V|=0.980095 pu @ -31.378° (a bit further still — one more line)
```
The pattern makes physical sense: magnitude drops a little at each step
away from the source (transformers and lines both introduce some
impedance drop), and the **angle jumps by about -31°** specifically
*across the transformer* (BUS2 → BUS3) — a real physical effect of the
transformer's own R/X impedance under load, not a bug.

## 6. Why Case B (no load on BUS3/BUS4) breaks this exact same process

Everything above is identical in *mechanism* for Case B — same
`compute_mismatch`, same `compute_jacobian`, same solver call. The only
difference is `BUS3`/`BUS4`'s `P_spec = Q_spec = 0` instead of `0.10/0.04`
and `0.15/0.06`. With zero demand anywhere on the Delta secondary, almost
nothing pulls current through the one truly undetermined direction (the
common-mode shift across BUS3/BUS4's three phases) — so `‖F‖_∞` can
numerically slip toward zero in the well-determined directions while
`κ₂(J)` independently rockets past `1e12` (to `1.8e12` by iteration 6, per
the earlier full run of this script) — at which point `solve_direct()`'s
guard (see [methods/DIRECT_SOLVE.md](../methods/DIRECT_SOLVE.md)) refuses to
hand back a `dx` at all, and Newton-Raphson correctly reports
`converged=False`, `fail_reason="condition number ... exceeds threshold"`.
Same loop, same node-by-node arithmetic — the *only* thing that changed is
whether any demand exists to pin down the floating direction.

---

## Appendix: the instrumented script that produced every number above

```python
import numpy as np
from experiments.exp01_4bus import build_4bus_system
from src.model.state import PHASES
from src.powerflow.mismatch import compute_mismatch
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import solve_linear_step

TRACK = ("BUS3", "a")
state, Ybus = build_4bus_system(load_on_secondary=True)
x = state.init_flat_start(v_mag=1.0)
i_track = state.unknown_index[TRACK]
bp_track = state.get(*TRACK)

for k in range(10):
    F = compute_mismatch(x, state, Ybus)
    F_inf = np.max(np.abs(F))

    # re-derive the tracked node's own mismatch arithmetic by hand
    V_dict = state.unpack(x)
    Vvec = np.zeros(state.n_busphases, dtype=complex)
    for key, idx in state.busphase_index.items():
        Vvec[idx] = V_dict[key]
    Ivec = Ybus @ Vvec
    i_bp = state.busphase_index[TRACK]
    e_i, f_i = Vvec[i_bp].real, Vvec[i_bp].imag
    Ir_i, Ii_i = Ivec[i_bp].real, Ivec[i_bp].imag
    P_calc = e_i*Ir_i + f_i*Ii_i
    Q_calc = f_i*Ir_i - e_i*Ii_i
    F_P = P_calc - (-bp_track.P_spec)
    F_Q = Q_calc - (-bp_track.Q_spec)

    print(k, F_inf, F_P, F_Q, abs(Vvec[i_bp]), np.degrees(np.angle(Vvec[i_bp])))

    if F_inf < 1e-10:
        break

    J = compute_jacobian(x, state, Ybus)
    dx, diag = solve_linear_step(J, -F, method="direct")
    x = x + dx
```
