"""Raw official MATPOWER case69 data (Member A/E-shared source data).

Every number in this module is copied verbatim from the official MATPOWER
case file:

    Source: https://raw.githubusercontent.com/MATPOWER/matpower/master/data/case69.m
    Retrieved: 2026-09-10, via WebFetch (cited directly in the project spec
    section 13/19 as this project's canonical IEEE-69 data source).
    Original data: a portion of the PG&E distribution system, per Baran &
    Wu (1989), "Network reconfiguration in distribution systems for loss
    reduction and load balancing," IEEE Trans. Power Delivery.

Per project spec section 19: this is fundamentally a BALANCED, single-phase-
equivalent representation (unlike IEEE-13/37, which are natively
phase-domain feeders). ``BASE_MVA=10``, ``BASE_KV=12.66`` (both from the
source file). Branch resistance/reactance are given directly in ohms
(physical, not per-unit) for each of the 68 radial branches; bus real/
reactive load is given directly in MW/MVAr (three-phase total, matching
the standard MATPOWER/pandapower convention). Exactly 48 of the 69 buses
have nonzero load, matching project spec section 19's "48 loads" -- an
independent confirmation this transcription is correct, not something
tuned to match.
"""

from __future__ import annotations


BASE_MVA = 10.0
BASE_KV = 12.66

# (bus_i, Pd_MW, Qd_MVAr) for all 69 buses.
BUS_LOADS_MW: list[tuple[int, float, float]] = [
    (1, 0.0, 0.0), (2, 0.0, 0.0), (3, 0.0, 0.0), (4, 0.0, 0.0), (5, 0.0, 0.0),
    (6, 0.0026, 0.0022), (7, 0.0404, 0.03), (8, 0.075, 0.054), (9, 0.03, 0.022),
    (10, 0.028, 0.019), (11, 0.145, 0.104), (12, 0.145, 0.104), (13, 0.008, 0.0055),
    (14, 0.008, 0.0055), (15, 0.0, 0.0), (16, 0.0455, 0.03), (17, 0.06, 0.035),
    (18, 0.06, 0.035), (19, 0.0, 0.0), (20, 0.001, 0.0006), (21, 0.114, 0.081),
    (22, 0.0053, 0.0035), (23, 0.0, 0.0), (24, 0.028, 0.02), (25, 0.0, 0.0),
    (26, 0.014, 0.01), (27, 0.014, 0.01), (28, 0.026, 0.0186), (29, 0.026, 0.0186),
    (30, 0.0, 0.0), (31, 0.0, 0.0), (32, 0.0, 0.0), (33, 0.014, 0.01),
    (34, 0.0195, 0.014), (35, 0.006, 0.004), (36, 0.026, 0.0186), (37, 0.026, 0.0186),
    (38, 0.0, 0.0), (39, 0.024, 0.017), (40, 0.024, 0.017), (41, 0.0012, 0.001),
    (42, 0.0, 0.0), (43, 0.006, 0.0043), (44, 0.0, 0.0), (45, 0.0392, 0.0263),
    (46, 0.0392, 0.0263), (47, 0.0, 0.0), (48, 0.079, 0.0564), (49, 0.3847, 0.2745),
    (50, 0.3847, 0.2745), (51, 0.0405, 0.0283), (52, 0.0036, 0.0027), (53, 0.0043, 0.0035),
    (54, 0.0264, 0.019), (55, 0.024, 0.0172), (56, 0.0, 0.0), (57, 0.0, 0.0),
    (58, 0.0, 0.0), (59, 0.1, 0.072), (60, 0.0, 0.0), (61, 1.244, 0.888),
    (62, 0.032, 0.023), (63, 0.0, 0.0), (64, 0.227, 0.162), (65, 0.059, 0.042),
    (66, 0.018, 0.013), (67, 0.018, 0.013), (68, 0.028, 0.02), (69, 0.028, 0.02),
]

# (fbus, tbus, r_ohm, x_ohm) for all 68 radial branches. Three of these
# (3-4, 3-28, 3-36) are REPLACED by Yg-Delta transformers in the paper's
# modification -- see experiments/exp04_69bus.py's module docstring for
# why "replace" rather than "add a parallel branch" is this
# reconstruction's documented reading of project spec section 19.
BRANCHES_OHM: list[tuple[int, int, float, float]] = [
    (1, 2, 0.0005, 0.0012), (2, 3, 0.0005, 0.0012), (3, 4, 0.0015, 0.0036),
    (4, 5, 0.0251, 0.0294), (5, 6, 0.366, 0.1864), (6, 7, 0.381, 0.1941),
    (7, 8, 0.0922, 0.047), (8, 9, 0.0493, 0.0251), (9, 10, 0.819, 0.2707),
    (10, 11, 0.1872, 0.0619), (11, 12, 0.7114, 0.2351), (12, 13, 1.03, 0.34),
    (13, 14, 1.044, 0.34), (14, 15, 1.058, 0.3496), (15, 16, 0.1966, 0.065),
    (16, 17, 0.3744, 0.1238), (17, 18, 0.0047, 0.0016), (18, 19, 0.3276, 0.1083),
    (19, 20, 0.2106, 0.069), (20, 21, 0.3416, 0.1129), (21, 22, 0.014, 0.0046),
    (22, 23, 0.1591, 0.0526), (23, 24, 0.3463, 0.1145), (24, 25, 0.7488, 0.2475),
    (25, 26, 0.3089, 0.1021), (26, 27, 0.1732, 0.0572), (3, 28, 0.0044, 0.0108),
    (28, 29, 0.064, 0.1565), (29, 30, 0.3978, 0.1315), (30, 31, 0.0702, 0.0232),
    (31, 32, 0.351, 0.116), (32, 33, 0.839, 0.2816), (33, 34, 1.708, 0.5646),
    (34, 35, 1.474, 0.4873), (3, 36, 0.0044, 0.0108), (36, 37, 0.064, 0.1565),
    (37, 38, 0.1053, 0.123), (38, 39, 0.0304, 0.0355), (39, 40, 0.0018, 0.0021),
    (40, 41, 0.7283, 0.8509), (41, 42, 0.31, 0.3623), (42, 43, 0.041, 0.0478),
    (43, 44, 0.0092, 0.0116), (44, 45, 0.1089, 0.1373), (45, 46, 0.0009, 0.0012),
    (4, 47, 0.0034, 0.0084), (47, 48, 0.0851, 0.2083), (48, 49, 0.2898, 0.7091),
    (49, 50, 0.0822, 0.2011), (8, 51, 0.0928, 0.0473), (51, 52, 0.3319, 0.114),
    (9, 53, 0.174, 0.0886), (53, 54, 0.203, 0.1034), (54, 55, 0.2842, 0.1447),
    (55, 56, 0.2813, 0.1433), (56, 57, 1.59, 0.5337), (57, 58, 0.7837, 0.263),
    (58, 59, 0.3042, 0.1006), (59, 60, 0.3861, 0.1172), (60, 61, 0.5075, 0.2585),
    (61, 62, 0.0974, 0.0496), (62, 63, 0.145, 0.0738), (63, 64, 0.7105, 0.3619),
    (64, 65, 1.041, 0.5302), (11, 66, 0.2012, 0.0611), (66, 67, 0.0047, 0.0014),
    (12, 68, 0.7394, 0.2444), (68, 69, 0.0047, 0.0016),
]

# Paper-specified transformer data (project spec section 19's table): the
# three branches replaced by Yg-Delta transformers.
TRANSFORMER_LOCATIONS: dict[tuple[int, int], dict] = {
    (3, 4): {"mva": 10.0, "kv": 13.8, "r_pu": 0.0007, "x_pu": 0.0018},
    (3, 28): {"mva": 10.0, "kv": 13.8, "r_pu": 0.0023, "x_pu": 0.0056},
    (3, 36): {"mva": 10.0, "kv": 13.8, "r_pu": 0.0023, "x_pu": 0.0056},
}
