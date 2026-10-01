# COMMANDS.md — every runnable command in this repository, and what it does

All commands assume your working directory is the repo root
(`CSE402-NumericalAnalysis-Project/`). Every script imports via absolute
paths like `from src.model...`, so run everything from here.

Files under `experiments/` and `src/validation/` are the only things meant
to be run directly — everything under `src/model`, `src/powerflow`,
`src/solvers`, `src/diagnostics` is library code, imported by these
scripts, never run on its own. `experiments/__init__.py` and the three
`experiments/member_{b,c,d}_utils.py` files are helper modules, not
runnable (no `__main__` block).

---

## 0. Setup

```bash
pip install -r requirements.txt
```

## 1. Test suite

```bash
python -m pytest -q
```
~85+ tests, a few seconds. On Python 3.14/Windows, `opendssdirect`'s import
has intermittently crashed pytest's collection with a native C-stack fault
(no Python summary printed) — not every run; if it happens, re-run with:
```bash
python -m pytest -q --ignore=tests/test_opendss.py
```

---

## 2. Core feeder experiments (fast — prints a self-contained report)

```bash
python experiments/exp01_4bus.py
```
Smallest test case: one source, one line, one Yg-Delta transformer, another
line. Runs **Case A** (load on the Delta secondary → healthy) and **Case B**
(no load on the Delta secondary → the paper's exact floating/singular
failure case) through the direct solver, printing a per-iteration table
(error, condition number, step size) and final voltages.

```bash
python experiments/exp02_13bus.py
```
Builds the full paper-modified IEEE-13 reconstruction (two Delta-connected
transformers). Runs **all four solvers once** at nominal load, prints each
one's report.

```bash
python experiments/exp03_37bus.py
```
Same pattern for IEEE-37 (every load here is Delta-connected). Supports an
optional `quasi_radial=True` variant (adds 3 loop-closing lines) inside
`build_37bus_system()`.

```bash
python experiments/exp04_69bus.py
```
Same pattern for IEEE-69, with 3 of its original branches swapped for
Yg-Delta transformers per the paper.

## 3. Per-solver 4-bus comparisons (fast)

```bash
python experiments/exp_b_svd_4bus.py
python experiments/exp_c_rrqr_4bus.py
python experiments/exp_d_tikhonov_4bus.py
```
Three near-identical scripts (one per original "solver owner"), each
comparing the direct solver against **one specific alternative** (SVD /
RRQR / Tikhonov respectively) on the 4-bus network, printing that solver's
own diagnostics (rank, pivots, regularization strength).

```bash
python experiments/exp_c_rrqr_vs_svd_4bus.py
```
Puts SVD and RRQR head-to-head: do they agree on rank? How far apart are
their final answers? Which is faster?

## 4. Sensitivity sweeps (slower — write CSVs + plots to results/, figures/;
##    these two dirs are gitignored, so just regenerate locally as needed)

```bash
python experiments/exp_b_svd_tolerance_4bus.py
```
Sweeps SVD's truncation cutoff (`rtol`) over a grid of values on the 4-bus
network, writes CSVs + 3 plots (the singular-value spectrum with the cutoff
marked, how many directions get kept, how the final error changes).

```bash
python experiments/exp_d_tikhonov_lambda_sweep_4bus.py
```
Same sweep shape, but for Tikhonov's `alpha` penalty strength, on the 4-bus
network.

```bash
python experiments/exp_d_ieee13_tikhonov_alpha_sweep.py
```
The same idea but on the real IEEE-13 network: runs a grid of `alpha`
values and reports which ones actually let Newton-Raphson converge. This is
what makes `alpha=1e-8` (used everywhere else in the 13-bus results) a
checked, reproducible choice rather than an arbitrary number.

## 5. The main investigation (centerpiece — slower, several sub-studies)

```bash
python experiments/exp_d_ieee13_investigation.py
```
Runs four sub-studies on IEEE-13, each written to its own CSV:
- `solver_robustness_study` — all 4 solvers x 5 load levels (50%-150%) —
  where "SVD/RRQR stop converging above ~75% load, Tikhonov keeps working
  until 125%" was found.
- `load_scaling_study` — voltage-spread and sequence-component metrics
  across load levels.
- `imbalance_study` — 0/10/20/30% phase imbalance — the "imbalance loads
  almost entirely onto the undetermined zero-sequence component" finding.
- `transformer_configuration_study` — floating vs. grounded transformer
  combinations — shows grounding fixes the conditioning.

## 6. Runtime-scaling benchmark (NOT a singularity study — timing only)

```bash
python experiments/exp_e_ieee118_scaling.py
```
Loads pandapower's real, healthy 118-bus case, replicates it to larger
sizes, times all four solvers directly, estimates each one's empirical
runtime-vs-size exponent, writes a CSV.

---

## 7. OpenDSS cross-validation — plain scripts, NEVER run under pytest

```bash
python -m src.validation.opendss
```
Compiles/solves the 4-bus Case A/B directly through OpenDSS (an
independent, industry-standard tool) and exports its voltages to CSV.

```bash
python -m src.validation.crosscheck
```
Runs this project's own 4-bus solver *and* OpenDSS on the same cases;
prints a pass/fail voltage-comparison table for the healthy case, and a
qualitative note for the singular case (OpenDSS often converges there via
its own hidden regularization, which this project's direct solver
correctly refuses to do).

```bash
python -m src.validation.antifloat_study
```
Sweeps OpenDSS's own `PPM_Antifloat` setting (default / disabled /
increased) on the singular 4-bus case, to measure how much of OpenDSS's
"success" there is really just its own hidden anti-singularity admittance.

```bash
python experiments/exp02b_13bus_opendss_crosscheck.py
python experiments/exp03b_37bus_opendss_crosscheck.py
python experiments/exp03c_37bus_quasiradial_opendss_crosscheck.py
python experiments/exp04b_69bus_opendss_crosscheck.py
```
The same "solve with our code, solve with OpenDSS, compare line-to-line
voltages" pattern applied to each real IEEE feeder (radial + quasi-radial
37-bus variant included); each returns `(median_rel_err, max_rel_err)` so
`make_figures.py` can call them directly.

---

## 8. Producing the report

```bash
python report/make_figures.py
```
Re-runs the real experiments live (not cached numbers) to regenerate all 6
figures used in the paper: 4-bus conditioning, IEEE-13 solver robustness,
IEEE-13 zero-sequence response, IEEE-13 transformer grounding, IEEE-118
scaling, and the cross-validation summary.

```bash
cd report && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
Compiles the IEEEtran paper. Requires a LaTeX installation with IEEEtran —
not verified as installed in this environment; the `pip install` above
doesn't cover this step.

---

## Quick reference: what to run to sanity-check the whole repo

```bash
python -m pytest -q
python experiments/exp01_4bus.py
python experiments/exp02_13bus.py
python experiments/exp03_37bus.py
python experiments/exp04_69bus.py
python -m src.validation.crosscheck
```
