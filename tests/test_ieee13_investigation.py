"""Regression tests for the controlled IEEE-13 investigation (Member D).

These assert the qualitative, structurally-expected relationships observed
when this experiment script was developed (see its module docstring), not
exact numeric values (which are properties of this specific reconstruction,
not universal constants). If a future change to the reconstruction breaks
one of these, that is worth investigating, not silencing.
"""

from __future__ import annotations

from experiments.exp02_13bus import build_13bus_system
from experiments.exp_d_ieee13_investigation import _run_and_summarise
from experiments.exp_d_ieee13_tikhonov_alpha_sweep import TOL_F, run_sweep


def test_grounding_both_transformers_drastically_improves_conditioning():
    """Progressively grounding floating windings should reduce kappa_2 and V0.

    This is the core structural claim of the transformer-configuration
    study: the paper (fully floating) case should be the worst-conditioned,
    and grounding both transformers should be the best-conditioned, with
    V0 (zero-sequence) magnitude following the same ordering.
    """
    paper_state, paper_ybus = build_13bus_system()
    grounded_state, grounded_ybus = build_13bus_system(
        t1_secondary_conn="yg", t2_primary_conn="yg", t2_secondary_conn="yg"
    )

    paper_row = _run_and_summarise("paper", paper_state, paper_ybus)
    grounded_row = _run_and_summarise("grounded", grounded_state, grounded_ybus)

    assert paper_row["converged"] is True
    assert grounded_row["converged"] is True
    assert grounded_row["flat_start_kappa_2"] < paper_row["flat_start_kappa_2"]
    # Grounding should bring conditioning down to a realistic feeder-scale
    # value, not just "somewhat better" -- this is meant to be a strong,
    # qualitatively obvious effect, not a marginal one.
    assert grounded_row["flat_start_kappa_2"] < 1e5
    assert grounded_row["v0_mean_abs"] < paper_row["v0_mean_abs"]


def test_imbalance_increases_v0_while_leaving_v1_almost_unchanged():
    """Load imbalance on this floating topology should load onto V0, not V1.

    Current post-cross-validation result: V0_mean grows monotonically with
    imbalance (0.337 -> 0.408 pu over 0%-30%) while V1_mean barely moves
    (0.807 -> 0.800), consistent with project spec section 7's hypothesis
    that the anomalous IEEE-13 discrepancy concentrates in the common-mode/
    zero-sequence component rather than the physically-normal positive-
    sequence component.
    """
    balanced_state, balanced_ybus = build_13bus_system(imbalance=0.0)
    imbalanced_state, imbalanced_ybus = build_13bus_system(imbalance=0.30)

    balanced_row = _run_and_summarise("balanced", balanced_state, balanced_ybus)
    imbalanced_row = _run_and_summarise("imbalanced", imbalanced_state, imbalanced_ybus)

    assert balanced_row["converged"] is True
    assert imbalanced_row["converged"] is True
    assert imbalanced_row["v0_mean_abs"] > balanced_row["v0_mean_abs"]
    # V1 should move far less in relative terms than V0 does.
    v0_relative_change = (
        (imbalanced_row["v0_mean_abs"] - balanced_row["v0_mean_abs"]) / balanced_row["v0_mean_abs"]
    )
    v1_relative_change = abs(
        (imbalanced_row["v1_mean_abs"] - balanced_row["v1_mean_abs"]) / balanced_row["v1_mean_abs"]
    )
    assert v1_relative_change < 0.05 * v0_relative_change


def test_load_scaling_does_not_change_flat_start_conditioning():
    """Flat-start kappa_2 depends on topology/transformer connections, not load.

    The flat-start state is always 1.0 pu regardless of P_spec/Q_spec, so
    scaling every load should leave the flat-start Jacobian's conditioning
    unchanged -- only the converged solution's voltages should move.
    """
    light_state, light_ybus = build_13bus_system(load_scale=0.5)
    heavy_state, heavy_ybus = build_13bus_system(load_scale=1.5)

    light_row = _run_and_summarise("light", light_state, light_ybus)
    heavy_row = _run_and_summarise("heavy", heavy_state, heavy_ybus)

    assert light_row["flat_start_kappa_2"] == heavy_row["flat_start_kappa_2"]


def test_nominal_ieee13_has_a_narrow_tikhonov_alpha_window():
    """The reported alpha is an empirical choice, not a universal default.

    On the fixed nominal-load reconstruction, the project's logarithmic
    sweep has one convergent point: 1e-8.  We pin the qualitative bracket as
    well as convergence itself so a future model change cannot silently make
    the report's parameter-sensitivity claim stale.
    """
    rows = run_sweep(write_csv=False, verbose=False)
    by_alpha = {row["alpha"]: row for row in rows}

    assert by_alpha[1e-8]["converged"] is True
    assert by_alpha[1e-8]["final_F_inf"] < TOL_F
    assert by_alpha[1e-10]["converged"] is False
    assert by_alpha[1e-6]["converged"] is False
    assert [row["alpha"] for row in rows if row["converged"]] == [1e-8]
