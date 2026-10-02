"""Raw official IEEE 13-node test feeder data (Member A/E-shared source data).

Every number in this module is copied verbatim from the official OpenDSS
implementation of the IEEE 13-node test feeder:

    Source: https://sourceforge.net/p/electricdss/code/HEAD/tree/trunk/
            Version8/Distrib/IEEETestCases/13Bus/IEEE13Nodeckt.dss
    Retrieved: 2026-09-10, via WebFetch (cited directly in the project spec
    section 13 as the project's canonical IEEE-13 data source).

This module intentionally contains ONLY that official data (line codes,
line list, official per-phase loads, official transformer nameplate/
connection data) with no paper-specified modification applied. Applying
Jang, Kim & Kim (2023)'s modifications (removed distributed load, removed
capacitor banks, Yg-Delta/Delta-Delta transformer connections, paper's own
R/X values) is done in ``experiments/exp02_13bus.py``, which imports this
module and is the place where every reconstruction assumption beyond the
official feeder is documented.

Everything below this docstring is a direct transcription check against the
fetched file -- do not hand-edit a number without re-checking it against the
source.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Line codes: self/mutual series impedance matrices, ohms per mile, in the
# PHASE domain (not sequence domain) -- usable directly as a line's series
# impedance matrix Z (see stamp_line_series_admittance's Yprim = inv(Z)
# convention). Matrices below are symmetric; the source file gives only the
# lower triangle, reproduced in full here.
# ---------------------------------------------------------------------------

LINECODES: dict[str, dict] = {
    "mtx601": {
        "nphases": 3,
        "r_ohm_per_mile": np.array([
            [0.3465, 0.1560, 0.1580],
            [0.1560, 0.3375, 0.1535],
            [0.1580, 0.1535, 0.3414],
        ]),
        "x_ohm_per_mile": np.array([
            [1.0179, 0.5017, 0.4236],
            [0.5017, 1.0478, 0.3849],
            [0.4236, 0.3849, 1.0348],
        ]),
    },
    "mtx602": {
        "nphases": 3,
        "r_ohm_per_mile": np.array([
            [0.7526, 0.1580, 0.1560],
            [0.1580, 0.7475, 0.1535],
            [0.1560, 0.1535, 0.7436],
        ]),
        "x_ohm_per_mile": np.array([
            [1.1814, 0.4236, 0.5017],
            [0.4236, 1.1983, 0.3849],
            [0.5017, 0.3849, 1.2112],
        ]),
    },
    "mtx603": {
        "nphases": 2,
        "r_ohm_per_mile": np.array([
            [1.3238, 0.2066],
            [0.2066, 1.3294],
        ]),
        "x_ohm_per_mile": np.array([
            [1.3569, 0.4591],
            [0.4591, 1.3471],
        ]),
    },
    "mtx604": {
        "nphases": 2,
        # Identical to mtx603 in the official file (see its inline comment).
        "r_ohm_per_mile": np.array([
            [1.3238, 0.2066],
            [0.2066, 1.3294],
        ]),
        "x_ohm_per_mile": np.array([
            [1.3569, 0.4591],
            [0.4591, 1.3471],
        ]),
    },
    "mtx605": {
        "nphases": 1,
        "r_ohm_per_mile": np.array([[1.3292]]),
        "x_ohm_per_mile": np.array([[1.3475]]),
    },
    "mtx606": {
        # Corrected mtx606 (Feb 3 2016, R. Dugan) -- the version actually
        # used by the official master file (it appears after, and
        # overrides, the older/uncorrected block kept only as a commented
        # reference in the source .dss file).
        "nphases": 3,
        "r_ohm_per_mile": np.array([
            [0.791721, 0.318476, 0.283450],
            [0.318476, 0.791721, 0.318476],
            [0.283450, 0.318476, 0.791721],
        ]),
        "x_ohm_per_mile": np.array([
            [0.438352, 0.0276838, -0.0184204],
            [0.0276838, 0.438352, 0.0276838],
            [-0.0184204, 0.0276838, 0.438352],
        ]),
    },
    "mtx607": {
        "nphases": 1,
        "r_ohm_per_mile": np.array([[1.3425]]),
        "x_ohm_per_mile": np.array([[0.5124]]),
    },
}


# ---------------------------------------------------------------------------
# Official line list: (name, bus1, bus2, phases-in-terminal-order,
# linecode, length_ft). ``phases`` gives the phase letter for each row/
# column of the referenced linecode's matrix, in the exact terminal order
# the official file specifies (e.g. Bus1=632.3.2 -> phases ("c", "b")).
#
# Buses "RG60" (regulator secondary) and "692" (far side of the closed
# 671-692 sectionalizing switch) are pre-merged into "650" and "671"
# respectively -- see the reconstruction note in exp02_13bus.py. This is a
# reconstruction simplification beyond the official file, not an error in
# transcription.
# ---------------------------------------------------------------------------

OFFICIAL_LINES: list[dict] = [
    {"name": "650632", "bus1": "650", "bus2": "632", "phases": ("a", "b", "c"),
     "linecode": "mtx601", "length_ft": 2000},
    {"name": "632670", "bus1": "632", "bus2": "670", "phases": ("a", "b", "c"),
     "linecode": "mtx601", "length_ft": 667},
    {"name": "670671", "bus1": "670", "bus2": "671", "phases": ("a", "b", "c"),
     "linecode": "mtx601", "length_ft": 1333},
    {"name": "671680", "bus1": "671", "bus2": "680", "phases": ("a", "b", "c"),
     "linecode": "mtx601", "length_ft": 1000},
    {"name": "632633", "bus1": "632", "bus2": "633", "phases": ("a", "b", "c"),
     "linecode": "mtx602", "length_ft": 500},
    {"name": "632645", "bus1": "632", "bus2": "645", "phases": ("c", "b"),
     "linecode": "mtx603", "length_ft": 500},
    {"name": "645646", "bus1": "645", "bus2": "646", "phases": ("c", "b"),
     "linecode": "mtx603", "length_ft": 300},
    {"name": "692675", "bus1": "671", "bus2": "675", "phases": ("a", "b", "c"),
     "linecode": "mtx606", "length_ft": 500},
    {"name": "671684", "bus1": "671", "bus2": "684", "phases": ("a", "c"),
     "linecode": "mtx604", "length_ft": 300},
    {"name": "684611", "bus1": "684", "bus2": "611", "phases": ("c",),
     "linecode": "mtx605", "length_ft": 300},
    {"name": "684652", "bus1": "684", "bus2": "652", "phases": ("a",),
     "linecode": "mtx607", "length_ft": 800},
]


# ---------------------------------------------------------------------------
# Official loads: (name, bus, terminal-phase-letters-in-listed-order, conn,
# kW, kvar). kW/kvar are the load object's OWN nameplate values exactly as
# given (for a 3-phase object these are the TOTAL three-phase kW/kvar, per
# OpenDSS convention; for a single-phase or single-delta-leg object they are
# that leg's own kW/kvar).
#
# Load.670a/670b/670c are the official file's own concentrated
# representation of the "distributed load on line 632 to 671" (see its
# inline comment) and are DELIBERATELY EXCLUDED here: Jang et al. explicitly
# say the modified 13-node case removes the distributed load.
# ---------------------------------------------------------------------------

OFFICIAL_LOADS: list[dict] = [
    {"name": "671", "bus": "671", "phases": ("a", "b", "c"), "conn": "delta",
     "kw": 1155.0, "kvar": 660.0},
    {"name": "634a", "bus": "634", "phases": ("a",), "conn": "wye",
     "kw": 160.0, "kvar": 110.0},
    {"name": "634b", "bus": "634", "phases": ("b",), "conn": "wye",
     "kw": 120.0, "kvar": 90.0},
    {"name": "634c", "bus": "634", "phases": ("c",), "conn": "wye",
     "kw": 120.0, "kvar": 90.0},
    {"name": "645", "bus": "645", "phases": ("b",), "conn": "wye",
     "kw": 170.0, "kvar": 125.0},
    {"name": "646", "bus": "646", "phases": ("b", "c"), "conn": "delta",
     "kw": 230.0, "kvar": 132.0},
    {"name": "692", "bus": "671", "phases": ("c", "a"), "conn": "delta",
     "kw": 170.0, "kvar": 151.0},
    {"name": "675a", "bus": "675", "phases": ("a",), "conn": "wye",
     "kw": 485.0, "kvar": 190.0},
    {"name": "675b", "bus": "675", "phases": ("b",), "conn": "wye",
     "kw": 68.0, "kvar": 60.0},
    {"name": "675c", "bus": "675", "phases": ("c",), "conn": "wye",
     "kw": 290.0, "kvar": 212.0},
    {"name": "611", "bus": "611", "phases": ("c",), "conn": "wye",
     "kw": 170.0, "kvar": 80.0},
    {"name": "652", "bus": "652", "phases": ("a",), "conn": "wye",
     "kw": 128.0, "kvar": 86.0},
]


def line_impedance_ohm(line: dict) -> np.ndarray:
    """Series impedance matrix (ohms) for one official line entry.

    ``r_ohm_per_mile``/``x_ohm_per_mile`` are per-mile; the official length
    is given in feet (5280 ft = 1 mile).
    """
    code = LINECODES[line["linecode"]]
    length_mi = line["length_ft"] / 5280.0
    return (code["r_ohm_per_mile"] + 1j * code["x_ohm_per_mile"]) * length_mi
