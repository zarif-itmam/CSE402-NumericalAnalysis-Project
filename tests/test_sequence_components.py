"""Unit tests for the symmetrical-component diagnostics module."""

from __future__ import annotations

import cmath

import numpy as np
import pytest

from src.diagnostics.sequence_components import (
    abc_to_seq,
    bus_sequence_components,
    line_to_line_magnitudes,
    seq_to_abc,
    sequence_components_by_bus,
)


def test_balanced_positive_sequence_has_only_v1():
    """A perfectly balanced abc-rotation set is pure positive sequence."""
    Va = 1.0 + 0j
    Vb = 1.0 * cmath.exp(-1j * 2 * cmath.pi / 3)
    Vc = 1.0 * cmath.exp(1j * 2 * cmath.pi / 3)

    v0, v1, v2 = abc_to_seq(Va, Vb, Vc)

    assert abs(v0) < 1e-12
    assert abs(v2) < 1e-12
    assert v1 == pytest.approx(1.0 + 0j, abs=1e-12)


def test_pure_common_mode_offset_is_entirely_zero_sequence():
    """Adding the same offset to all three phases must land entirely in V0."""
    offset = 0.3 + 0.1j
    Va = 1.0 + offset
    Vb = 1.0 * cmath.exp(-1j * 2 * cmath.pi / 3) + offset
    Vc = 1.0 * cmath.exp(1j * 2 * cmath.pi / 3) + offset

    v0, v1, v2 = abc_to_seq(Va, Vb, Vc)

    assert v0 == pytest.approx(offset, abs=1e-12)
    assert v1 == pytest.approx(1.0 + 0j, abs=1e-12)
    assert abs(v2) < 1e-12


def test_seq_to_abc_is_exact_inverse_of_abc_to_seq():
    """Round-tripping through the sequence domain must recover the original phases."""
    rng = np.random.default_rng(7)
    Va, Vb, Vc = (complex(*rng.standard_normal(2)) for _ in range(3))

    v0, v1, v2 = abc_to_seq(Va, Vb, Vc)
    va2, vb2, vc2 = seq_to_abc(v0, v1, v2)

    assert va2 == pytest.approx(Va, abs=1e-12)
    assert vb2 == pytest.approx(Vb, abs=1e-12)
    assert vc2 == pytest.approx(Vc, abs=1e-12)


def test_negative_sequence_from_reversed_rotation():
    """A reverse-rotation (b/c swapped) balanced set is pure negative sequence."""
    Va = 1.0 + 0j
    Vb = 1.0 * cmath.exp(1j * 2 * cmath.pi / 3)   # swapped vs. positive-sequence
    Vc = 1.0 * cmath.exp(-1j * 2 * cmath.pi / 3)  # swapped vs. positive-sequence

    v0, v1, v2 = abc_to_seq(Va, Vb, Vc)

    assert abs(v0) < 1e-12
    assert abs(v1) < 1e-12
    assert v2 == pytest.approx(1.0 + 0j, abs=1e-12)


def test_bus_sequence_components_requires_all_three_phases():
    """A bus missing a phase (e.g. a lateral) has no sequence decomposition."""
    V = {("BUS1", "a"): 1.0 + 0j, ("BUS1", "b"): -0.5 + 0.2j}  # phase c missing

    assert bus_sequence_components(V, "BUS1") is None


def test_sequence_components_by_bus_skips_incomplete_buses_only():
    """Only buses with all three phases active appear in the per-bus summary."""
    V = {
        ("FULL", "a"): 1.0 + 0j,
        ("FULL", "b"): 1.0 * cmath.exp(-1j * 2 * cmath.pi / 3),
        ("FULL", "c"): 1.0 * cmath.exp(1j * 2 * cmath.pi / 3),
        ("LATERAL", "a"): 1.0 + 0j,
        ("LATERAL", "c"): 0.9 + 0j,
    }

    result = sequence_components_by_bus(V)

    assert set(result) == {"FULL"}
    v0, v1, v2 = result["FULL"]
    assert abs(v0) < 1e-12
    assert abs(v2) < 1e-12


def test_line_to_line_magnitudes_matches_direct_computation():
    """Line-to-line magnitudes must equal |Vp1 - Vp2| for each present pair."""
    V = {
        ("BUS1", "a"): 1.0 + 0j,
        ("BUS1", "b"): -0.5 - 0.3j,
        ("BUS1", "c"): -0.4 + 0.35j,
    }

    magnitudes = line_to_line_magnitudes(V, "BUS1")

    assert magnitudes["ab"] == pytest.approx(abs(V[("BUS1", "a")] - V[("BUS1", "b")]))
    assert magnitudes["bc"] == pytest.approx(abs(V[("BUS1", "b")] - V[("BUS1", "c")]))
    assert magnitudes["ca"] == pytest.approx(abs(V[("BUS1", "c")] - V[("BUS1", "a")]))


def test_line_to_line_magnitudes_handles_partial_phase_bus():
    """A 2-phase lateral should report only the one pair it actually has."""
    V = {("LAT", "a"): 1.0 + 0j, ("LAT", "c"): 0.9 - 0.1j}

    magnitudes = line_to_line_magnitudes(V, "LAT")

    assert set(magnitudes) == {"ca"}
