"""Validation and regression tests for the IEEE/PG&E 69-bus reconstruction.

Per project spec section 22, this is the hard gate for
experiments/exp04_69bus.py: no result from that module should be trusted
until test_jacobian_69bus_matches_finite_difference passes.
"""

from __future__ import annotations

import numpy as np
import pytest

from experiments.exp04_69bus import build_69bus_system
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson
from tests.test_jacobian import compute_relative_frobenius_error, finite_difference_jacobian

E_J_TOLERANCE = 1e-6


def test_jacobian_69bus_matches_finite_difference():
    """Mandatory FD validation before trusting any IEEE-69 NR result."""
    state, Ybus = build_69bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    rng = np.random.default_rng(69)
    x_test = x0 + 0.01 * rng.standard_normal(x0.size)

    J_analytic = compute_jacobian(x_test, state, Ybus)
    J_fd = finite_difference_jacobian(x_test, state, Ybus, h=1e-6)
    E_J = compute_relative_frobenius_error(J_analytic, J_fd)

    assert E_J < E_J_TOLERANCE, (
        f"IEEE-69 analytical Jacobian disagrees with finite differences: "
        f"E_J={E_J:.3e} exceeds tolerance {E_J_TOLERANCE:.1e}."
    )


def test_69bus_system_has_expected_size_and_slack():
    """Sanity check on the reconstruction's topology before trusting solver results."""
    state, Ybus = build_69bus_system()

    distinct_buses = {bp.bus_name for bp in state.all_busphases()}
    assert len(distinct_buses) == 69  # BUS1..BUS69, all fully 3-phase (balanced reconstruction)
    assert state.n_busphases == 3 * 69
    assert state.n_unknowns == 2 * 3 * 68  # 68 non-slack buses
    assert Ybus.shape == (3 * 69, 3 * 69)

    slack_phases = [bp for bp in state.all_busphases() if bp.is_slack]
    assert len(slack_phases) == 3
    assert all(bp.bus_name == "BUS1" for bp in slack_phases)


def test_direct_svd_rrqr_all_converge_and_agree():
    """Unlike IEEE-13/37, the direct solver itself converges here.

    Observed while building this reconstruction: the three Yg-Delta
    transformers sit at 3-4/3-28/3-36, each carrying substantial real load
    downstream (unlike a purely no-load floating case) -- this keeps
    kappa_2 severe (~2.3e5 at flat start, growing to ~5.7e10 during
    iteration) but finite, staying under the direct solver's 1e12 guard.
    This is a real structural difference from IEEE-13/37, not a
    reconstruction defect -- see
    test_flat_start_jacobian_is_ill_conditioned_but_not_exactly_singular.

    (The iteration-max kappa_2 figure above reflects a per-phase load
    normalization bug fix -- see git history around exp04_69bus.py's
    build_69bus_system(): BUS_LOADS_MW gives each bus's TOTAL three-phase
    load, and an earlier version treated it as already per-phase, making
    every load 3x too heavy. The flat-start kappa_2 is unaffected, since it
    does not depend on load magnitude at all -- see
    tests/test_ieee13_investigation.py's equivalent load-invariance
    finding for IEEE-13 -- but the iteration trajectory and the OpenDSS
    cross-validation below do depend on it.)
    """
    state, Ybus = build_69bus_system()
    x0 = state.init_flat_start(v_mag=1.0)

    direct = newton_raphson(state, Ybus, x0, method="direct", tol_F=1e-9, max_iter=80, compute_conditioning=False)
    svd = newton_raphson(state, Ybus, x0, method="svd", tol_F=1e-9, max_iter=80, compute_conditioning=False)
    rrqr = newton_raphson(state, Ybus, x0, method="rrqr", tol_F=1e-9, max_iter=80, compute_conditioning=False)

    assert direct["converged"] is True
    assert svd["converged"] is True
    assert rrqr["converged"] is True
    np.testing.assert_allclose(direct["x"], svd["x"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(svd["x"], rrqr["x"], rtol=1e-6, atol=1e-6)


def test_flat_start_jacobian_is_ill_conditioned_but_not_exactly_singular():
    """Structural check: severe but finite conditioning, unlike IEEE-13/37's exact singularity.

    The three Yg-Delta transformers each leave a large downstream region
    with no explicit ground path (e.g. the 3-4 transformer's secondary
    reaches 47 of the 69 buses), and Ybus itself has an exact null
    direction there (a uniform complex voltage shift across that whole
    region leaves every current unchanged -- verified separately while
    developing this reconstruction). But the actual NONLINEAR NR Jacobian
    at flat start is only ILL-conditioned, not exactly singular: linearizing
    the power-balance mismatch shows the "uniform shift" direction's
    Jacobian component is proportional to the ACTUAL CURRENT already
    flowing at each shifted bus-phase, which is nonzero here because this
    region carries substantial real load (e.g. bus 61's 1.244 MW) --
    unlike IEEE-13/37, whose floating regions' exact degeneracy survives
    despite their own loads (a difference in specific topology/loading
    this project's reconstructions were not able to fully resolve
    analytically; both behaviors are independently confirmed by the
    Jacobian finite-difference test, so neither is a bug).
    """
    state, Ybus = build_69bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)
    kappa0 = sv0[0] / sv0[-1]

    assert kappa0 > 1e4, "Expected severe (but not necessarily astronomical) ill-conditioning."
    assert sv0[-1] > 1e-6 * sv0[0], "Expected the flat-start Jacobian to NOT be near-machine-singular here."


def test_bus_load_is_split_evenly_across_three_phases_not_replicated_whole():
    """Regression test for the total-vs-per-phase load bug (see git history).

    BUS_LOADS_MW's Pd/Qd are the TOTAL three-phase load at a bus (standard
    MATPOWER convention); each phase must draw exactly one third of it, not
    the full amount. An earlier version of build_69bus_system() applied
    the full total to every phase (3x too heavy), caught via an isolated
    OpenDSS cross-check at bus 61. This test pins the fix down directly:
    at flat start (V=1.0 pu), each phase's specified real power injection
    should be very close to P_total_MW/3 in the corresponding per-unit
    base, without needing a full NR solve.
    """
    from src.model.ieee69_data import BASE_MVA, BUS_LOADS_MW

    state, _Ybus = build_69bus_system()
    bus61_load_total_mw = next(p for i, p, _q in BUS_LOADS_MW if i == 61)
    # (total_kW / 3 legs) / (S_base_kVA / 3) algebraically reduces to
    # total_MW / S_base_MVA -- written this way to make the "divide the
    # total by 3 before normalizing" step explicit, matching how
    # build_69bus_system() itself is written (and was previously NOT
    # written, hence the bug).
    per_phase_kw = (bus61_load_total_mw * 1000.0) / 3.0
    per_phase_base_kva = (BASE_MVA * 1000.0) / 3.0
    expected_p_pu_per_phase = per_phase_kw / per_phase_base_kva

    bp = state.get("BUS61", "a")
    assert bp.P_spec == pytest.approx(expected_p_pu_per_phase, rel=1e-9)
    assert bp.P_spec == pytest.approx(bus61_load_total_mw / BASE_MVA, rel=1e-9)
