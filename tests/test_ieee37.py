"""Validation and regression tests for the IEEE 37-node reconstruction.

Per project spec section 22, this is the hard gate for
experiments/exp03_37bus.py: no result from that module should be trusted
until test_jacobian_37bus_matches_finite_difference passes.
"""

from __future__ import annotations

import numpy as np

from experiments.exp03_37bus import build_37bus_system
from src.diagnostics.sequence_components import line_to_line_magnitudes
from src.powerflow.jacobian import compute_jacobian
from src.powerflow.newton import newton_raphson
from tests.test_jacobian import compute_relative_frobenius_error, finite_difference_jacobian

E_J_TOLERANCE = 1e-6


def test_jacobian_37bus_matches_finite_difference():
    """Mandatory FD validation before trusting any IEEE-37 NR result."""
    state, Ybus = build_37bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    rng = np.random.default_rng(37)
    x_test = x0 + 0.01 * rng.standard_normal(x0.size)

    J_analytic = compute_jacobian(x_test, state, Ybus)
    J_fd = finite_difference_jacobian(x_test, state, Ybus, h=1e-6)
    E_J = compute_relative_frobenius_error(J_analytic, J_fd)

    assert E_J < E_J_TOLERANCE, (
        f"IEEE-37 analytical Jacobian disagrees with finite differences: "
        f"E_J={E_J:.3e} exceeds tolerance {E_J_TOLERANCE:.1e}."
    )


def test_37bus_system_has_expected_size_and_slack():
    """Sanity check on the reconstruction's topology before trusting solver results."""
    state, Ybus = build_37bus_system()

    # The "37-node" feeder name refers to the 701-799 numbered bus family;
    # counting SourceBus and the 799/799r-merge separately gives 38 distinct
    # bus objects in this reconstruction, all fully 3-phase (every line in
    # this feeder is 3-phase) -- so total bus-phases = 3 * 38. 37 of those
    # are non-slack (SourceBus is the only slack bus).
    distinct_buses = {bp.bus_name for bp in state.all_busphases()}
    assert len(distinct_buses) == 38
    assert state.n_busphases == 3 * 38
    assert state.n_unknowns == 2 * 3 * 37  # 37 non-slack buses, all 3-phase
    assert Ybus.shape == (3 * 38, 3 * 38)

    slack_phases = [bp for bp in state.all_busphases() if bp.is_slack]
    assert len(slack_phases) == 3
    assert all(bp.bus_name == "SourceBus" for bp in slack_phases)


def test_direct_solver_fails_on_the_doubly_floating_reconstruction():
    """The direct solver must refuse this system: it is genuinely singular.

    Both SubXF and XFM1 are Delta-Delta (already true of the OFFICIAL
    feeder, per exp03_37bus.py's module docstring), so the flat-start
    Jacobian is singular to within machine precision -- this reproduces the
    paper's "entire feeder ungrounded" description, not a reconstruction
    defect.
    """
    state, Ybus = build_37bus_system()
    x0 = state.init_flat_start(v_mag=1.0)

    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)
    assert sv0[-1] < 1e-8 * sv0[0], "Expected the flat-start Jacobian to be (near-)singular."

    result = newton_raphson(
        state, Ybus, x0, method="direct", tol_F=1e-9, max_iter=10, compute_conditioning=False
    )
    assert result["converged"] is False


def test_flat_start_jacobian_has_multiple_near_zero_singular_values():
    """Structural check: both floating transformers leave a real rank deficiency.

    Observed while building this reconstruction: exactly 4 near-zero
    singular values (not the naive "1 per floating transformer" guess of
    2) -- bus 775 (XFM1's secondary) is a fully unloaded leaf reachable
    only through XFM1, which evidently contributes more than a single
    common-mode direction to the linearized null space at flat start. This
    pins the observed count down as a regression rather than asserting an
    unverified formula.
    """
    state, Ybus = build_37bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    J0 = compute_jacobian(x0, state, Ybus)
    sv0 = np.linalg.svd(J0, compute_uv=False)

    near_zero = np.sum(sv0 < 1e-6 * sv0[0])
    assert near_zero == 4


def test_svd_and_rrqr_converge_and_agree_at_nominal_load():
    """Unlike the IEEE-13 reconstruction, SVD/RRQR converge cleanly here.

    Both reach the paper's target singular topology, but this larger,
    more-distributed-load feeder does not exhibit the SVD/RRQR
    oscillation-at-nominal-load behavior documented for IEEE-13 (see
    tests/test_ieee13.py) -- a genuine per-topology difference, not
    something to force into matching.
    """
    state, Ybus = build_37bus_system()
    x0 = state.init_flat_start(v_mag=1.0)

    svd = newton_raphson(state, Ybus, x0, method="svd", tol_F=1e-9, max_iter=80, compute_conditioning=False)
    rrqr = newton_raphson(state, Ybus, x0, method="rrqr", tol_F=1e-9, max_iter=80, compute_conditioning=False)

    assert svd["converged"] is True
    assert rrqr["converged"] is True
    np.testing.assert_allclose(svd["x"], rrqr["x"], rtol=1e-8, atol=1e-8)


def test_line_to_line_voltage_is_better_conditioned_than_phase_voltage():
    """Structural check for the floating common-mode phenomenon this system induces."""
    state, Ybus = build_37bus_system()
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(state, Ybus, x0, method="svd", tol_F=1e-9, max_iter=80, compute_conditioning=False)
    assert result["converged"] is True

    V = state.unpack(result["x"])
    phase_mags = [abs(v) for key, v in V.items() if key[0] != "SourceBus"]
    phase_spread = max(phase_mags) - min(phase_mags)

    line_line_mags = []
    for bus in {key[0] for key in V if key[0] != "SourceBus"}:
        line_line_mags.extend(line_to_line_magnitudes(V, bus).values())
    line_line_spread = max(line_line_mags) - min(line_line_mags)

    assert line_line_spread < 0.5 * phase_spread


def test_jacobian_37bus_quasiradial_matches_finite_difference():
    """Mandatory FD validation for the quasi-radial variant (project spec section 18)."""
    state, Ybus = build_37bus_system(quasi_radial=True)
    x0 = state.init_flat_start(v_mag=1.0)
    rng = np.random.default_rng(1837)
    x_test = x0 + 0.01 * rng.standard_normal(x0.size)

    J_analytic = compute_jacobian(x_test, state, Ybus)
    J_fd = finite_difference_jacobian(x_test, state, Ybus, h=1e-6)
    E_J = compute_relative_frobenius_error(J_analytic, J_fd)

    assert E_J < E_J_TOLERANCE


def test_quasiradial_adds_no_new_buses_but_changes_conditioning():
    """The three extra lines close loops among existing buses (project spec section 18).

    Generated programmatically from the same build function (that section's
    explicit instruction), so unknown count is identical to the radial
    case; only the Jacobian's conditioning should change.
    """
    state_radial, Ybus_radial = build_37bus_system()
    state_quasi, Ybus_quasi = build_37bus_system(quasi_radial=True)

    assert state_quasi.n_unknowns == state_radial.n_unknowns

    x0 = state_radial.init_flat_start(v_mag=1.0)
    sv_radial = np.linalg.svd(compute_jacobian(x0, state_radial, Ybus_radial), compute_uv=False)
    sv_quasi = np.linalg.svd(compute_jacobian(x0, state_quasi, Ybus_quasi), compute_uv=False)

    # Both are floating the same way (same transformers, same connections),
    # so both must still be singular by construction...
    assert sv_radial[-1] < 1e-6 * sv_radial[0]
    assert sv_quasi[-1] < 1e-6 * sv_quasi[0]
    # ...but adding three loop-closing lines changes sigma_max (more paths
    # for current to flow) and therefore the raw conditioning number.
    assert sv_quasi[0] != sv_radial[0]


def test_quasiradial_svd_converges_at_nominal_load():
    """The quasi-radial variant should remain solvable by the rank-revealing solvers."""
    state, Ybus = build_37bus_system(quasi_radial=True)
    x0 = state.init_flat_start(v_mag=1.0)
    result = newton_raphson(state, Ybus, x0, method="svd", tol_F=1e-9, max_iter=80, compute_conditioning=False)
    assert result["converged"] is True
