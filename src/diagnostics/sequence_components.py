"""Symmetrical-component (zero/positive/negative sequence) diagnostics.

This is the shared analysis tool needed for the IEEE-13 anomaly
investigation's zero-sequence hypothesis (project spec section 7 / section
25.D): "the comparatively larger 13-node discrepancy may be concentrated
primarily in V0 rather than positive-sequence, negative-sequence, or
line-to-line voltage." It is deliberately independent of any specific
solver or feeder: it operates on a plain ``{(bus, phase): complex}`` voltage
dict (the return type of ``NetworkState.unpack``), so it can be applied to
the 4-bus, 13-bus, or any future feeder's result without modification.

Convention: the standard Fortescue transform, with the phase-rotation
operator a = 1*exp(j*120deg):

    [V0]       [1  1   1 ] [Va]
    [V1] = 1/3 [1  a  a^2] [Vb]
    [V2]       [1 a^2   a] [Vc]

V0 is the zero-sequence (common-mode) component: equal-magnitude,
equal-phase contribution added identically to all three phases. V1 is the
positive-sequence (normal balanced rotation) component. V2 is the
negative-sequence (reverse rotation, i.e. imbalance) component.
"""

from __future__ import annotations

import cmath

import numpy as np

_A = cmath.exp(1j * 2 * cmath.pi / 3)  # 1 / 120 degrees
_SEQUENCE_MATRIX = np.array([
    [1, 1, 1],
    [1, _A, _A ** 2],
    [1, _A ** 2, _A],
]) / 3.0
_INVERSE_SEQUENCE_MATRIX = np.array([
    [1, 1, 1],
    [1, _A ** 2, _A],
    [1, _A, _A ** 2],
])


def abc_to_seq(Va: complex, Vb: complex, Vc: complex) -> tuple[complex, complex, complex]:
    """Convert phase quantities (a, b, c) to sequence components (0, 1, 2)."""
    v0, v1, v2 = _SEQUENCE_MATRIX @ np.array([Va, Vb, Vc], dtype=complex)
    return complex(v0), complex(v1), complex(v2)


def seq_to_abc(V0: complex, V1: complex, V2: complex) -> tuple[complex, complex, complex]:
    """Convert sequence components (0, 1, 2) back to phase quantities (a, b, c).

    Exact inverse of :func:`abc_to_seq`; provided mainly so round-trip
    correctness can be tested directly rather than assumed.
    """
    va, vb, vc = _INVERSE_SEQUENCE_MATRIX @ np.array([V0, V1, V2], dtype=complex)
    return complex(va), complex(vb), complex(vc)


def bus_sequence_components(
    V: dict[tuple[str, str], complex], bus: str
) -> tuple[complex, complex, complex] | None:
    """Sequence components of one bus's three-phase voltage.

    Parameters
    ----------
    V : dict
        A ``{(bus, phase): complex}`` voltage dict, e.g. from
        ``NetworkState.unpack(x)``.
    bus : str
        Bus name to evaluate.

    Returns
    -------
    (V0, V1, V2) or None
        ``None`` if the bus does not have all three phases active in ``V``
        (the transform is only defined for a complete a/b/c triplet; a
        1- or 2-phase lateral bus has no meaningful sequence decomposition).
    """
    key_a, key_b, key_c = (bus, "a"), (bus, "b"), (bus, "c")
    if key_a not in V or key_b not in V or key_c not in V:
        return None
    return abc_to_seq(V[key_a], V[key_b], V[key_c])


def sequence_components_by_bus(
    V: dict[tuple[str, str], complex]
) -> dict[str, tuple[complex, complex, complex]]:
    """Sequence components for every bus in ``V`` that has all three phases.

    Buses with fewer than three active phases (single-phase or two-phase
    laterals) are silently omitted -- they have no sequence decomposition.
    """
    buses = sorted({bus for bus, _phase in V})
    result = {}
    for bus in buses:
        seq = bus_sequence_components(V, bus)
        if seq is not None:
            result[bus] = seq
    return result


def line_to_line_magnitudes(
    V: dict[tuple[str, str], complex], bus: str
) -> dict[str, float]:
    """``{"ab": |Vab|, "bc": |Vbc|, "ca": |Vca|}`` for the phases ``bus`` has active.

    Only pairs where both phases are active in ``V`` are included, so this
    also works for a 2-phase lateral bus (returning just one pair).
    """
    magnitudes = {}
    for p1, p2 in (("a", "b"), ("b", "c"), ("c", "a")):
        key1, key2 = (bus, p1), (bus, p2)
        if key1 in V and key2 in V:
            magnitudes[p1 + p2] = abs(V[key1] - V[key2])
    return magnitudes
