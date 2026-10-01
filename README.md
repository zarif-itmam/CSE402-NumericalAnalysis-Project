# Robust Newton–Raphson Power Flow under Singular Jacobians

**CSE 402 — Numerical Analysis, Simulation and Modelling · Section B · Group B_09**

This repository reproduces and extends

> J. Jang, D. Kim, I. Kim, *"Singularity Handling for Unbalanced Three-Phase Transformers in
> Newton–Raphson Power Flow Analyses Using Moore–Penrose Pseudo-Inverse,"* IEEE Access, vol. 11, 2023.
> [doi:10.1109/ACCESS.2023.3269503](https://doi.org/10.1109/ACCESS.2023.3269503)

Floating (Delta / ungrounded-wye) three-phase transformer connections make the Newton–Raphson (NR)
Jacobian singular or severely ill-conditioned. The base paper fixes the linear solve with an SVD-based
Moore–Penrose pseudo-inverse. We ask whether that is the best general remedy by running **one shared NR
engine** with **four interchangeable linear solvers** on reconstructed IEEE feeders.

| Solver | Module | Idea |
|---|---|---|
| Direct | [`src/solvers/direct.py`](src/solvers/direct.py) | LU solve with an explicit κ₂ guard (10¹²) |
| SVD / Moore–Penrose | [`src/solvers/svd_pinv.py`](src/solvers/svd_pinv.py) | explicit thresholded pseudo-inverse (base paper) |
| Rank-revealing QR | [`src/solvers/rrqr.py`](src/solvers/rrqr.py) | pivoted QR + LAPACK `GELSY` minimum-norm solve |
| Tikhonov | [`src/solvers/tikhonov.py`](src/solvers/tikhonov.py) | damped step via augmented least squares, λ = α·σ_max² |

## Headline results (nominal load, tolerance ‖F‖∞ < 1e-9)

| Feeder | Direct | SVD | RRQR | Tikhonov (α = 1e-8) |
|---|:-:|:-:|:-:|:-:|
| IEEE 4-bus, loaded secondary | ✓ (8) | ✓ | ✓ | ✓ |
| IEEE 4-bus, unloaded secondary | ✓ (8)\* | ✓ | ✓ | ✓ |
| IEEE 13-bus | ✗ guard | ✗ oscillates | ✗ oscillates | **✓ (15)** |
| IEEE 37-bus | ✗ guard | **✓ (10)** | **✓ (10)** | ✗ stalls |
| IEEE 69-bus | ✓ (7) | ✓ (7) | ✓ (7) | ✗ stalls |

\* κ₂ reaches ≈8×10¹¹ — just below the 10¹² guard on our machine; the margin is platform dependent.

**No single solver wins everywhere**: damped Tikhonov steps are the only ones that converge on the
13-bus feeder, while the undamped minimum-norm steps of SVD/RRQR are the only ones that converge on the 37-bus
feeder. A symmetrical-component study shows the 13-bus imbalance lands almost entirely on the undetermined
zero-sequence voltage. See the figures in [`report/figures/`](report/figures/).

## Quick start

```bash
git clone https://github.com/zarif-itmam/CSE402-NumericalAnalysis-Project.git
cd CSE402-NumericalAnalysis-Project
pip install -r requirements.txt
python -m pytest -q --ignore=tests/test_opendss.py     # 86 tests
python experiments/exp01_4bus.py                       # 4-bus failure reproduction
python experiments/exp02_13bus.py                      # 13-bus, four solvers
python experiments/exp03_37bus.py                      # 37-bus, four solvers
python experiments/exp04_69bus.py                      # 69-bus, four solvers
python report/make_figures.py && python report/make_extra_figures.py   # regenerate all figures
```

Every script is standalone and prints a self-contained report. All commands are listed in
[`docs/COMMANDS.md`](docs/COMMANDS.md); each experiment's inputs/outputs are described in
[`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md); per-solver theory notes are in [`methods/`](methods/);
a step-by-step NR trace is in [`docs/NR_WALKTHROUGH.md`](docs/NR_WALKTHROUGH.md).

## Repository map

```text
src/model/        bus-phase state, Ybus assembly (lines + Yg/Delta transformer stamping), feeder data
src/powerflow/    mismatch F(x), analytical Jacobian J(x), shared Newton–Raphson loop
src/solvers/      direct.py, svd_pinv.py, rrqr.py, tikhonov.py  (common (dx, diagnostics) contract)
src/diagnostics/  symmetrical-component (V0/V1/V2) analysis
src/validation/   OpenDSS reference pipeline and cross-checks
experiments/      one standalone script per feeder / study
tests/            pytest suite (finite-difference Jacobian gate, per-solver, per-feeder tests)
data/             raw feeder files + provenance.yml (every reconstruction assumption)
report/           figure-generation scripts and generated figures
methods/, docs/   method notes, command list, experiment reference, NR walkthrough
```

## Validation

* **Jacobian**: analytical vs centred finite differences, relative Frobenius error < 1e-9 on every feeder.
* **OpenDSS**: median / max relative line-to-line voltage error — 4-bus ≈1.5e-6 pu (max), 13-bus 6.6 % / 26.5 %,
  37-bus 2.3 % / 6.5 %, 37-bus quasi-radial 2.3 % / 6.1 %, 69-bus 3.3e-8 / 8.2e-6. Feeders with an exact null
  space (13, 37) legitimately disagree in the undetermined subspace; those without (4, 69) agree to near machine precision.
* Two real per-unit bugs (13-bus loads 3× too light, 69-bus loads 3× too heavy) were caught by comparing small,
  well-conditioned sub-circuits with OpenDSS first.

## Known limitations

Delta loads are approximated as wye-equivalents; voltage regulators are ideal; κ₂ of an exactly singular Jacobian is
round-off limited (≈10¹⁷–10¹⁸) and varies by platform, so only the *count* of near-zero singular values is meaningful.
Open data discrepancies are listed (never silently resolved) in [`data/provenance.yml`](data/provenance.yml).
`opendssdirect` can crash pytest collection on some Windows setups; run the `exp*_opendss_crosscheck.py` scripts directly.

## Team

| Member | Name | ID | Owns |
|---|---|---|---|
| A | Omar Abdur Razzaque | 2105094 | shared NR core (`src/model`, `src/powerflow`), direct solver, 4-bus baseline, docs |
| B | Md. Nazmul Islam Talukder | 2105117 | SVD / Moore–Penrose solver, SVD threshold study, IEEE 37-bus feeder |
| C | Md. Nazmus Sakib | 2105102 | RRQR solver, RRQR-vs-SVD study, IEEE 69-bus feeder |
| D | Zarif Itmam Hossain | 2105116 | Tikhonov solver, α sweeps, IEEE 13-bus feeder and zero-sequence investigation, report figures |
| E | Rubaiyat-E-Zaman | 2105104 | OpenDSS validation, data provenance, IEEE 118-bus scaling benchmark, CI |
