# EXPERIMENTS.md — every runnable file, in detail

For each runnable file: what network/data it takes as **input**, the
**process** it runs, and what **output** it produces (console report
and/or files on disk). See [COMMANDS.md](COMMANDS.md) for the bare
commands and the top-level [README](../README.md) for the
underlying architecture these all sit on top of.

Every experiment funnels through the same call shape:
`build_*_system(...) -> (NetworkState, Ybus)`, then
`newton_raphson(state, Ybus, x0, method=..., solver_options=...)`. Only the
`build_*` parameters and `method` ever change between experiments.

---

## Core feeder experiments

### `experiments/exp01_4bus.py`

- **Input**: a small, hand-built 4-bus network (`build_4bus_system`) — one
  fixed source, one ordinary line, one Yg-Delta transformer (R=0.01,
  X=0.06 pu), another line. Boolean flag `load_on_secondary` selects the
  case.
- **Process**: `run_case()` builds the network, runs the **direct** solver
  through the shared NR loop (`tol_F` default, `compute_conditioning=True`
  so every iteration's `κ₂(J)` is tracked), and flags
  `SINGULARITY_KAPPA_THRESHOLD = 1e10` as the "severe ill-conditioning"
  cutoff for reporting (independent of whether NR itself reported
  converged). The bottom `__main__` block runs both cases in sequence:
  - **Case A** (`load_on_secondary=True`): real load on the Delta secondary
    → healthy.
  - **Case B** (`load_on_secondary=False`): zero load on the Delta
    secondary → the paper's exact floating/singular reproduction target.
- **Output**: console only — per-iteration table (`‖F‖_∞`, `κ₂(J)`,
  `σ_min(J)`, `‖dx‖`, solver success/failure mode), final voltages if
  converged, and a Case A vs. Case B summary line. Nothing written to
  disk.

### `experiments/exp02_13bus.py`

- **Input**: `src/model/ieee13_data.py`'s official line/load data, with the
  paper's modifications applied in `build_13bus_system()` — T1 (Yg
  primary/Delta secondary) and T2 (Delta/Delta) transformers, distributed
  load and capacitor banks removed. Optional keyword overrides:
  `load_scale`, `imbalance`, `t1_secondary_conn`, `t2_primary_conn`,
  `t2_secondary_conn` (used by other scripts to explore variants; default
  run uses none).
- **Process**: `run_case(label, method, solver_options)` builds the network
  once and solves it with `tol_F=1e-9, max_iter=60,
  compute_conditioning=True`. The `__main__` block loops over **all four
  methods** (`direct`, `svd`, `rrqr`, `tikhonov` with `alpha=1e-8`) at
  nominal load and prints each one's report in turn.
- **Output**: console only — per-method convergence status, iteration
  count, max `κ₂(J)` observed, fail reason if any, final voltage magnitude
  range.

### `experiments/exp03_37bus.py`

- **Input**: `src/model/ieee37_data.py`'s official data (every load here is
  Delta-connected), assembled by `build_37bus_system(load_scale=1.0,
  xfm1_xhl_percent=..., quasi_radial=False)`. Setting `quasi_radial=True`
  programmatically adds 3 extra loop-closing lines (computed from the same
  source data, not a second hand-maintained network).
- **Process**: identical shape to `exp02_13bus.py` — same `run_case`
  signature, same `tol_F`/`max_iter`. `__main__` loops over all four
  methods at nominal (radial) load.
- **Output**: console only — same report format as `exp02_13bus.py`.

### `experiments/exp04_69bus.py`

- **Input**: `src/model/ieee69_data.py`, adapted from the Baran & Wu /
  MATPOWER `case69.m` single-phase-equivalent data and balanced across all
  three phases. `build_69bus_system(load_scale=1.0)` stamps every ordinary
  branch except the 3 the paper specifies get replaced by Yg-Delta
  transformers.
- **Process**: same shape as the 13-/37-bus scripts. `__main__` loops over
  all four methods at nominal load.
- **Output**: console only — same report format. (This is the one feeder
  whose flat-start Jacobian has *zero* near-zero singular values — the
  direct solver is expected to succeed here too, unlike 13/37.)

---

## Per-solver 4-bus comparisons

### `experiments/exp_b_svd_4bus.py`, `exp_c_rrqr_4bus.py`, `exp_d_tikhonov_4bus.py`

- **Input**: the same `build_4bus_system` as `exp01_4bus.py`.
- **Process**: `run_4bus_case(label, load_on_secondary, method, ...)` runs
  one case through one method. `run_all_cases()` runs **4 combinations**:
  healthy+direct, healthy+`<solver>`, floating+direct, floating+`<solver>`
  (`<solver>` = SVD / RRQR / Tikhonov respectively; the Tikhonov script
  uses a fixed demo `alpha` = `DEMO_ALPHA`). After running, it directly
  compares the healthy case's converged state vector between direct and
  the alternate solver (`max |x_direct - x_alt|`) as a sanity check that
  they agree when the problem isn't singular.
- **Output**: console only — `print_run()`'s detailed per-iteration report
  (customized per solver: rank/pivots for RRQR, retained-singular-value
  count for SVD, regularization strength for Tikhonov) plus the final
  state-difference sanity check.

### `experiments/exp_c_rrqr_vs_svd_4bus.py`

- **Input**: same 4-bus network, both cases.
- **Process**: `compare_case()` runs **SVD and RRQR** on the same case,
  compares whether they report the same numerical rank, how far apart
  their final answers land, then `median_solver_runtime_sec()` times just
  the **last iteration's** solve (closest to the actual singularity, so
  the most demanding one) over `REPEATS` repetitions for direct/SVD/RRQR
  each, and reports median runtimes. `run_all_cases()` runs this for both
  the healthy and floating cases.
- **Output**: console only — rank agreement, state-difference, and timing
  comparison. Explicitly framed as a small-matrix (18×18) local
  measurement, not an asymptotic-cost claim (that's `exp_e_ieee118_scaling.py`'s job).

---

## Sensitivity sweeps

### `experiments/exp_b_svd_tolerance_4bus.py`

- **Input**: the 4-bus network (both cases), swept over
  `RTOL_GRID = ("default"→None, 1e-16, 1e-14, 1e-12, 1e-10, 1e-8, ...)` —
  SVD's singular-value truncation cutoff.
- **Process**: `run_case(load_on_secondary, rtol)` for every grid point;
  `summarise_run()` flattens each run into one summary row plus a detailed
  per-iteration trace; `_plot_diagnostics()` draws 3 charts.
- **Output**:
  - `results/svd_tolerance_4bus_summary.csv`, `results/svd_tolerance_4bus_trace.csv`
  - `figures/svd_4bus_floating_default_spectrum.png` (singular-value spectrum with cutoff marked)
  - `figures/svd_4bus_floating_rank_vs_rtol.png` (kept-directions vs. cutoff)
  - `figures/svd_4bus_floating_mismatch_vs_rtol.png` (final error vs. cutoff)
  - console summary table.
  (`results/`, `figures/` are gitignored — regenerate locally.)

### `experiments/exp_d_tikhonov_lambda_sweep_4bus.py`

- **Input**: same 4-bus network, swept over
  `ALPHA_GRID = ("none"→0.0, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, ...)` —
  Tikhonov's penalty strength.
- **Process**: identical 5-function shape to the SVD sweep above
  (`run_case`, `summarise_run`, `_write_csv`, `_plot_diagnostics`,
  `run_sweep`).
- **Output**:
  - `results/tikhonov_alpha_4bus_summary.csv`, `results/tikhonov_alpha_4bus_trace.csv`
  - `figures/tikhonov_4bus_floating_step_norm_vs_alpha.png`
  - `figures/tikhonov_4bus_floating_mismatch_vs_alpha.png`

### `experiments/exp_d_ieee13_tikhonov_alpha_sweep.py`

- **Input**: the **real IEEE-13 network** at nominal load, swept over the
  same-shaped `ALPHA_GRID` (`0.0` through `1e-4`).
- **Process**: `run_alpha(alpha)` runs one Tikhonov solve at nominal load;
  `run_sweep()` runs the whole grid and reports which values actually let
  NR converge — this is the reproducible justification for the
  `alpha=1e-8` value used as the *default* Tikhonov setting everywhere else
  in the 13-bus results (the docstring notes convergence isn't monotonic in
  `alpha` — there's a real sweet spot, not "more damping is always safer").
- **Output**: `results/ieee13_tikhonov_alpha_sweep.csv` + console table
  (`alpha`, converged?, iterations, final `‖F‖_∞`).

---

## The main investigation

### `experiments/exp_d_ieee13_investigation.py`

- **Input**: the IEEE-13 network from `exp02_13bus.py::build_13bus_system`,
  reused with different keyword overrides per study. Fixed settings for
  every solve in this script: `TOL_F=1e-9`, `MAX_ITER=80`,
  `SOLVE_METHOD="tikhonov"`, `SOLVE_OPTIONS={"alpha": 1e-8}` — chosen
  because, empirically, SVD/RRQR's undamped minimum-norm steps *oscillate*
  rather than converge on this doubly-floating reconstruction at nominal
  load and above (documented in the module docstring, and independently
  confirmed by `solver_robustness_study` below).
- **Process** — four independent studies, each calling `build_13bus_system`
  fresh per data point:
  1. **`solver_robustness_study()`** — the one study that *does* vary
     `method`: all 4 solvers (`direct`, `svd`, `rrqr`,
     `tikhonov`+`alpha=1e-8`) × 5 load levels
     (`load_scale ∈ {0.50, 0.75, 1.00, 1.25, 1.50}`), `compute_conditioning=False`
     for speed. Records converged/iterations per (method, load).
  2. **`load_scaling_study()`** — same 5 load levels, Tikhonov only; records
     `_voltage_spread_metrics` (phase-magnitude spread, line-to-line
     spread, mean `|V0|`/`|V1|`/`|V2|`) plus flat-start/observed `κ₂`.
  3. **`imbalance_study()`** — `imbalance ∈ {0.00, 0.10, 0.20, 0.30}`
     (project-defined phase-scaling rule, not paper-specified — see
     `data/provenance.yml`), same metrics — this is the study behind the
     "imbalance loads onto V0, not V1" finding.
  4. **`transformer_configuration_study()`** — 4 fixed transformer-grounding
     configurations (paper's own both-floating baseline, T1 grounded, T2
     grounded, both grounded), same metrics — shows conditioning improves
     as grounding is added.
- **Output**: `run_all()` writes one CSV per study to
  `results/ieee13_{name}_study.csv` (`solver_robustness`,
  `load_scaling`, `imbalance`, `transformer_configuration`), plus a console
  progress line per data point as each study runs.

---

## Runtime-scaling benchmark

### `experiments/exp_e_ieee118_scaling.py`

- **Input**: **not** a project feeder — `pandapower.networks.case118()`,
  a real, healthy, balanced 118-bus transmission network. Its Ybus is
  extracted after pandapower runs its own power flow (a genuine converged
  admittance matrix, not a synthetic/random one).
- **Process**:
  1. `block_diagonal_replicate(Y, factor)` — duplicates the real 118-bus
     Ybus side-by-side as disconnected copies, at
     `REPLICATION_FACTORS = (1, 2, 4, 8)` — i.e. 118, 236, 472, 944 buses —
     to get a genuine-structure scaling curve instead of one single data
     point.
  2. `to_real_jacobian_form(Y)` — expands each complex `n×n` Ybus into the
     real `2n×2n` block form `[[G,-B],[B,G]]`, matching this project's
     actual rectangular-coordinate Jacobian shape (so reported sizes are
     directly comparable to e.g. IEEE-13's `n_unknowns=64`).
  3. `median_runtime_sec(solve_fn, J, rhs, repeats=5)` times each of the
     four `solve_*` functions **directly** (not through the NR loop — this
     network has no singularity to reproduce, so only raw solver speed is
     being measured) against a random real right-hand side, at each of the
     4 replicated sizes. Tikhonov uses a fixed `TIKHONOV_ALPHA=1e-10`.
- **Output**: `results/ieee118_scaling_benchmark.csv` + console table +
  an estimated "runtime grows like size^___" exponent per solver. Explicitly
  documented as dense-matrix benchmarking (matches this project's own
  dense solver implementations), not representative of a production sparse
  solver's scaling.

---

## OpenDSS cross-validation

### `src/validation/opendss.py` (`python -m src.validation.opendss`)

- **Input**: `data/raw/ieee4/ieee4_reconstruction.dss` — this project's own
  OpenDSS-language rewrite of the 4-bus network.
- **Process**: `compile_and_solve()` clears OpenDSS, compiles the `.dss`
  file (captures `os.getcwd()` at import time and restores it afterward,
  since OpenDSS's `compile` command silently changes the process's working
  directory), optionally edits loads via `extra_commands`, solves, and
  reads back every bus-phase voltage via `_export_bus_phase_voltages()`
  (mapping OpenDSS's node numbering 1/2/3 to this project's a/b/c). The
  `__main__` block runs **Case A** (as-is) and **Case B** (`Load.LOAD_BUS3`
  and `LOAD_BUS4` zeroed via `extra_commands`, matching `exp01_4bus.py`'s
  `load_on_secondary=False`).
- **Output**: `results/opendss_ieee4_caseA.csv`,
  `results/opendss_ieee4_caseB.csv` (bus, phase, V_real, V_imag, V_mag,
  V_angle) + console convergence/iteration count for each case.

### `src/validation/crosscheck.py` (`python -m src.validation.crosscheck`)

- **Input**: this project's own 4-bus solver (direct method) *and*
  `src.validation.opendss`, on the same two cases.
- **Process**: `run_nr_case()` runs this project's own solver (deliberately
  does **not** treat non-convergence as an error — for Case B, failing to
  converge with the direct method is the expected, correct behavior being
  studied). `compare_case()` runs both solvers, then either prints a
  per-bus-phase pass/fail table (tolerances `V_MAG_TOL=1e-3` pu,
  `V_ANGLE_TOL_DEG=0.1`) for Case A, or a qualitative note for Case B
  (OpenDSS often converges there via its own hidden regularization — not
  treated as a pass/fail comparison).
- **Output**: console only — the comparison table for Case A, and a
  qualitative summary for Case B.

### `src/validation/antifloat_study.py` (`python -m src.validation.antifloat_study`)

- **Input**: the same 4-bus Case B (`ieee4_reconstruction.dss`, secondary
  loads zeroed), OpenDSS's per-transformer `ppm_antifloat` setting on
  `Transformer.XFMR1`.
- **Process**: `run_sweep()` solves Case B three times via OpenDSS, at
  `SETTINGS_TO_TEST = [("default", None), ("disabled", ppm_antifloat=0),
  ("elevated", ppm_antifloat=100)]` — isolating how much of OpenDSS's own
  convergence on the singular case is really just this tiny hidden
  anti-float admittance quietly regularizing away the same singularity
  this project studies.
- **Output**: `results/antifloat_default_caseB.csv`,
  `results/antifloat_disabled_caseB.csv`, `results/antifloat_elevated_caseB.csv`
  + console convergence/iteration count per setting.

### `experiments/exp02b_13bus_opendss_crosscheck.py`, `exp03b_37bus_...`, `exp03c_37bus_quasiradial_...`, `exp04b_69bus_...`

- **Input**: the matching real IEEE feeder (13/37/37-quasi-radial/69),
  solved both by this project's own code and by OpenDSS via a paper-matched
  `.dss` twin under `data/raw/`.
- **Process**: `solve_python()` runs the shared NR loop with **method="svd"**
  for 37/37-quasi-radial/69 bus (`tol_F`, `max_iter`, `compute_conditioning=False`)
  — except the 13-bus version, which uses **method="tikhonov"**
  (`TIKHONOV_ALPHA=1e-8`), since that's the one that reliably converges on
  that specific network (per the investigation script's finding).
  `solve_opendss()` solves the matching `.dss` twin. `compare()` compares
  **line-to-line** voltage magnitudes (not raw phase voltages — the fairer
  metric on a network with an undetermined common-mode component) at a
  fixed list of representative buses (13-bus:
  `COMPARISON_BUSES = ("650","632","671","675","633","634","645","646","684")`),
  and **returns** `(median_rel_err, max_rel_err)` so `report/make_figures.py`
  can call it directly.
- **Output**: console table of per-bus relative errors + the summary tuple
  (used programmatically by `make_figures.py`, not written to a file by
  these scripts themselves).

---

## Report generation

### `report/make_figures.py`

- **Input**: calls the actual experiment functions above directly (never
  parses old CSVs), so every figure reflects the current, live state of the
  code:
  - `fig_4bus_conditioning()` → `exp01_4bus.run_case` (Case A & B)
  - `fig_ieee13_solver_robustness()` → `exp_d_ieee13_investigation.solver_robustness_study`
  - `fig_ieee13_zero_sequence()` → `exp_d_ieee13_investigation.imbalance_study`
  - `fig_ieee13_transformer_grounding()` → `exp_d_ieee13_investigation.transformer_configuration_study`
  - `fig_ieee118_scaling()` → `exp_e_ieee118_scaling.run_scaling_benchmark`
  - `fig_crossvalidation_summary()` → all four OpenDSS `compare()` functions
    (13/37/37-quasi-radial/69), plus a hardcoded IEEE-4 constant
    (`ieee4_max_dv = 1.49e-6`, a max-abs-voltage-error metric from
    `crosscheck.py`, deliberately noted as a different metric convention
    than the other four feeders' relative line-to-line error).
- **Process**: each `fig_*()` function runs its underlying experiment(s),
  builds one `matplotlib` figure, and calls `save(fig, name)`.
- **Output**: `report/figures/{name}.pdf` and `report/figures/{name}.png`
  for six figures: `fig1_4bus_conditioning`, `fig2_ieee13_solver_robustness`,
  `fig3_ieee13_zero_sequence`, `fig4_ieee13_grounding`,
  `fig5_ieee118_scaling`, `fig6_crossvalidation_summary` — these are
  tracked in git (unlike `results/`/`figures/` at the repo root, which are
  gitignored scratch output).

```bash
cd report && pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```
Compiles `report/paper.tex` (IEEEtran format) + `refs.bib` into
`report/paper.pdf`, embedding the figures above via `\includegraphics`.
Requires a local LaTeX distribution with the IEEEtran class — not verified
as installed in this environment.
