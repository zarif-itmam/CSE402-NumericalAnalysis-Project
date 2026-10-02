"""Raw official IEEE 37-node test feeder data (Member A/E-shared source data).

Every number in this module is copied verbatim from the official OpenDSS
implementation of the IEEE 37-node test feeder:

    Source (master circuit): https://sourceforge.net/p/electricdss/code/HEAD/
        tree/trunk/Version8/Distrib/IEEETestCases/37Bus/ieee37.dss
    Source (line codes 721-724): https://sourceforge.net/p/electricdss/code/
        HEAD/tree/trunk/Version8/Distrib/IEEETestCases/37Bus/IEEELineCodes.DSS
    Retrieved: 2026-09-10, via WebFetch (cited directly in the project spec
    section 13 as the project's canonical test-feeder data source).

Per project spec section 16, this is a naturally three-wire, fully-ungrounded
delta feeder -- unlike IEEE-13, the OFFICIAL (unmodified) topology already
matches the paper's description almost exactly:

  - Both transformers (SubXF and XFM1) are ALREADY Delta-Delta in the
    official file (not something this project needs to change, unlike
    IEEE-13's T1/T2).
  - Bus 775 (XFM1's secondary) already has NO load anywhere in the official
    file -- matching project spec section 16's "no load on the XFM secondary"
    exactly, with nothing to remove.

So, unlike ieee13_data.py, this module needs essentially no paper-driven
modification beyond what experiments/exp03_37bus.py documents (the
regulator/switch-style ideal-merge simplification, and the XFM1 reactance
discrepancy flagged in project spec section 17).

Line lengths in the official file are given WITHOUT an explicit ``units=``
keyword; the associated linecodes (721-724) are calibrated in ohms per 1000
feet (matching this same source file's comment on linecode 300, and
independently confirmed here by cross-checking an isolated single-line
OpenDSS circuit against this module's own conversion -- see git history:
the same investigation that caught the IEEE-13 per-phase load bug also
verified the IEEE-37 line-length convention BEFORE committing this file, to
avoid repeating that mistake). Do not apply a further ft/mile conversion:
``length`` as given IS already in the correct (1000 ft) unit for direct use
with these matrices.
"""

from __future__ import annotations

import numpy as np


LINECODES: dict[str, dict] = {
    "722": {
        "nphases": 3,
        "r_ohm_per_1000ft": np.array([
            [0.089981061, 0.030852273, 0.023371212],
            [0.030852273, 0.085000000, 0.030852273],
            [0.023371212, 0.030852273, 0.089981061],
        ]),
        "x_ohm_per_1000ft": np.array([
            [0.056306818, -0.006174242, -0.011496212],
            [-0.006174242, 0.050719697, -0.006174242],
            [-0.011496212, -0.006174242, 0.056306818],
        ]),
    },
    "723": {
        "nphases": 3,
        "r_ohm_per_1000ft": np.array([
            [0.245000000, 0.092253788, 0.086837121],
            [0.092253788, 0.246628788, 0.092253788],
            [0.086837121, 0.092253788, 0.245000000],
        ]),
        "x_ohm_per_1000ft": np.array([
            [0.127140152, 0.039981061, 0.028806818],
            [0.039981061, 0.119810606, 0.039981061],
            [0.028806818, 0.039981061, 0.127140152],
        ]),
    },
    "724": {
        "nphases": 3,
        "r_ohm_per_1000ft": np.array([
            [0.396818182, 0.098560606, 0.093295455],
            [0.098560606, 0.399015152, 0.098560606],
            [0.093295455, 0.098560606, 0.396818182],
        ]),
        "x_ohm_per_1000ft": np.array([
            [0.146931818, 0.051856061, 0.040208333],
            [0.051856061, 0.140113636, 0.051856061],
            [0.040208333, 0.051856061, 0.146931818],
        ]),
    },
    "721": {
        "nphases": 3,
        "r_ohm_per_1000ft": np.array([
            [0.055416667, 0.012746212, 0.006382576],
            [0.012746212, 0.050113636, 0.012746212],
            [0.006382576, 0.012746212, 0.055416667],
        ]),
        "x_ohm_per_1000ft": np.array([
            [0.037367424, -0.006969697, -0.007897727],
            [-0.006969697, 0.035984848, -0.006969697],
            [-0.007897727, -0.006969697, 0.037367424],
        ]),
    },
}


# (name, bus1, bus2, length in the linecode's native 1000-ft unit, linecode).
# All lines in this feeder are 3-phase (a, b, c); "799r" (the regulator's
# secondary bus) is pre-merged into "799" here -- see exp03_37bus.py.
OFFICIAL_LINES: list[dict] = [
    {"name": "L1", "bus1": "701", "bus2": "702", "linecode": "722", "length": 0.96},
    {"name": "L2", "bus1": "702", "bus2": "705", "linecode": "724", "length": 0.40},
    {"name": "L3", "bus1": "702", "bus2": "713", "linecode": "723", "length": 0.36},
    {"name": "L4", "bus1": "702", "bus2": "703", "linecode": "722", "length": 1.32},
    {"name": "L5", "bus1": "703", "bus2": "727", "linecode": "724", "length": 0.24},
    {"name": "L6", "bus1": "703", "bus2": "730", "linecode": "723", "length": 0.60},
    {"name": "L7", "bus1": "704", "bus2": "714", "linecode": "724", "length": 0.08},
    {"name": "L8", "bus1": "704", "bus2": "720", "linecode": "723", "length": 0.80},
    {"name": "L9", "bus1": "705", "bus2": "742", "linecode": "724", "length": 0.32},
    {"name": "L10", "bus1": "705", "bus2": "712", "linecode": "724", "length": 0.24},
    {"name": "L11", "bus1": "706", "bus2": "725", "linecode": "724", "length": 0.28},
    {"name": "L12", "bus1": "707", "bus2": "724", "linecode": "724", "length": 0.76},
    {"name": "L13", "bus1": "707", "bus2": "722", "linecode": "724", "length": 0.12},
    {"name": "L14", "bus1": "708", "bus2": "733", "linecode": "723", "length": 0.32},
    {"name": "L15", "bus1": "708", "bus2": "732", "linecode": "724", "length": 0.32},
    {"name": "L16", "bus1": "709", "bus2": "731", "linecode": "723", "length": 0.60},
    {"name": "L17", "bus1": "709", "bus2": "708", "linecode": "723", "length": 0.32},
    {"name": "L18", "bus1": "710", "bus2": "735", "linecode": "724", "length": 0.20},
    {"name": "L19", "bus1": "710", "bus2": "736", "linecode": "724", "length": 1.28},
    {"name": "L20", "bus1": "711", "bus2": "741", "linecode": "723", "length": 0.40},
    {"name": "L21", "bus1": "711", "bus2": "740", "linecode": "724", "length": 0.20},
    {"name": "L22", "bus1": "713", "bus2": "704", "linecode": "723", "length": 0.52},
    {"name": "L23", "bus1": "714", "bus2": "718", "linecode": "724", "length": 0.52},
    {"name": "L24", "bus1": "720", "bus2": "707", "linecode": "724", "length": 0.92},
    {"name": "L25", "bus1": "720", "bus2": "706", "linecode": "723", "length": 0.60},
    {"name": "L26", "bus1": "727", "bus2": "744", "linecode": "723", "length": 0.28},
    {"name": "L27", "bus1": "730", "bus2": "709", "linecode": "723", "length": 0.20},
    {"name": "L28", "bus1": "733", "bus2": "734", "linecode": "723", "length": 0.56},
    {"name": "L29", "bus1": "734", "bus2": "737", "linecode": "723", "length": 0.64},
    {"name": "L30", "bus1": "734", "bus2": "710", "linecode": "724", "length": 0.52},
    {"name": "L31", "bus1": "737", "bus2": "738", "linecode": "723", "length": 0.40},
    {"name": "L32", "bus1": "738", "bus2": "711", "linecode": "723", "length": 0.40},
    {"name": "L33", "bus1": "744", "bus2": "728", "linecode": "724", "length": 0.20},
    {"name": "L34", "bus1": "744", "bus2": "729", "linecode": "724", "length": 0.28},
    {"name": "L35", "bus1": "799", "bus2": "701", "linecode": "721", "length": 1.85},
]


# (name, bus, terminal-phase-letters-in-listed-order, kW, kvar). Every load
# in this feeder is delta-connected (either a single leg, e.g. "701.1.2" ->
# phases (a, b), or -- Load.S728 only -- the full three-phase object).
OFFICIAL_LOADS: list[dict] = [
    {"name": "S701a", "bus": "701", "phases": ("a", "b"), "kw": 140.0, "kvar": 70.0},
    {"name": "S701b", "bus": "701", "phases": ("b", "c"), "kw": 140.0, "kvar": 70.0},
    {"name": "S701c", "bus": "701", "phases": ("c", "a"), "kw": 350.0, "kvar": 175.0},
    {"name": "S712c", "bus": "712", "phases": ("c", "a"), "kw": 85.0, "kvar": 40.0},
    {"name": "S713c", "bus": "713", "phases": ("c", "a"), "kw": 85.0, "kvar": 40.0},
    {"name": "S714a", "bus": "714", "phases": ("a", "b"), "kw": 17.0, "kvar": 8.0},
    {"name": "S714b", "bus": "714", "phases": ("b", "c"), "kw": 21.0, "kvar": 10.0},
    {"name": "S718a", "bus": "718", "phases": ("a", "b"), "kw": 85.0, "kvar": 40.0},
    {"name": "S720c", "bus": "720", "phases": ("c", "a"), "kw": 85.0, "kvar": 40.0},
    {"name": "S722b", "bus": "722", "phases": ("b", "c"), "kw": 140.0, "kvar": 70.0},
    {"name": "S722c", "bus": "722", "phases": ("c", "a"), "kw": 21.0, "kvar": 10.0},
    {"name": "S724b", "bus": "724", "phases": ("b", "c"), "kw": 42.0, "kvar": 21.0},
    {"name": "S725b", "bus": "725", "phases": ("b", "c"), "kw": 42.0, "kvar": 21.0},
    {"name": "S727c", "bus": "727", "phases": ("c", "a"), "kw": 42.0, "kvar": 21.0},
    {"name": "S728", "bus": "728", "phases": ("a", "b", "c"), "kw": 126.0, "kvar": 63.0},
    {"name": "S729a", "bus": "729", "phases": ("a", "b"), "kw": 42.0, "kvar": 21.0},
    {"name": "S730c", "bus": "730", "phases": ("c", "a"), "kw": 85.0, "kvar": 40.0},
    {"name": "S731b", "bus": "731", "phases": ("b", "c"), "kw": 85.0, "kvar": 40.0},
    {"name": "S732c", "bus": "732", "phases": ("c", "a"), "kw": 42.0, "kvar": 21.0},
    {"name": "S733a", "bus": "733", "phases": ("a", "b"), "kw": 85.0, "kvar": 40.0},
    {"name": "S734c", "bus": "734", "phases": ("c", "a"), "kw": 42.0, "kvar": 21.0},
    {"name": "S735c", "bus": "735", "phases": ("c", "a"), "kw": 85.0, "kvar": 40.0},
    {"name": "S736b", "bus": "736", "phases": ("b", "c"), "kw": 42.0, "kvar": 21.0},
    {"name": "S737a", "bus": "737", "phases": ("a", "b"), "kw": 140.0, "kvar": 70.0},
    {"name": "S738a", "bus": "738", "phases": ("a", "b"), "kw": 126.0, "kvar": 62.0},
    {"name": "S740c", "bus": "740", "phases": ("c", "a"), "kw": 85.0, "kvar": 40.0},
    {"name": "S741c", "bus": "741", "phases": ("c", "a"), "kw": 42.0, "kvar": 21.0},
    {"name": "S742a", "bus": "742", "phases": ("a", "b"), "kw": 8.0, "kvar": 4.0},
    {"name": "S742b", "bus": "742", "phases": ("b", "c"), "kw": 85.0, "kvar": 40.0},
    {"name": "S744a", "bus": "744", "phases": ("a", "b"), "kw": 42.0, "kvar": 21.0},
]


def line_impedance_ohm(line: dict) -> np.ndarray:
    """Series impedance matrix (ohms) for one official line entry.

    ``length`` is already expressed in this feeder's native 1000-ft unit
    (see module docstring) -- no further conversion is applied.
    """
    code = LINECODES[line["linecode"]]
    return (code["r_ohm_per_1000ft"] + 1j * code["x_ohm_per_1000ft"]) * line["length"]
