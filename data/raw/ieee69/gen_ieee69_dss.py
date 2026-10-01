"""Generate ieee69_paper_reconstruction.dss from src/model/ieee69_data.py.

Run from the repository root: ``python data/raw/ieee69/gen_ieee69_dss.py``.
Regenerate this way whenever ieee69_data.py changes -- do not hand-edit the
generated .dss file's branch/load lists; 65 regular branches is too many to
safely transcribe or edit by hand (this script exists specifically to avoid
that class of error, which the IEEE-13 reconstruction's earlier per-unit
bug demonstrated is easy to introduce and easy to miss without an
independent cross-check).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from src.model.ieee69_data import BASE_KV, BASE_MVA, BRANCHES_OHM, BUS_LOADS_MW, TRANSFORMER_LOCATIONS

TRANSFORMER_PAIRS = set(TRANSFORMER_LOCATIONS)


def regular_branches():
    return [
        (f, t, r, x) for (f, t, r, x) in BRANCHES_OHM
        if (f, t) not in TRANSFORMER_PAIRS and (t, f) not in TRANSFORMER_PAIRS
    ]


def generate() -> str:
    lines = [
        "! ieee69_paper_reconstruction.dss",
        "!",
        "! Modified IEEE/PG&E 69-bus feeder (project spec section 19), generated",
        "! PROGRAMMATICALLY by gen_ieee69_dss.py from src/model/ieee69_data.py (the",
        "! same official-data module experiments/exp04_69bus.py's",
        "! build_69bus_system() uses), to avoid hand-transcription errors across 65",
        "! regular branches. This is the OpenDSS-side twin of build_69bus_system()",
        "! -- see that function's docstring for the full reconstruction rationale",
        "! (balanced three-phase treatment, the 3-4/3-28/3-36 branch-replacement",
        "! reading, the 13.8kV-vs-12.66kV voltage-base discrepancy). Regenerate",
        "! with gen_ieee69_dss.py if ieee69_data.py ever changes -- do not hand-edit",
        "! this file's branch/load lists.",
        "!",
        "! Regular branches use R1/X1/R0/X0 (no mutual coupling, matching the",
        "! Python reconstruction's diagonal Yprim = (1/z)*I stamp for a balanced",
        "! single-phase-equivalent feeder) with Length=1, units=none, so R1/X1 are",
        "! literal ohms, matching src/model/ieee69_data.py's BRANCHES_OHM directly",
        "! (independently verified against this exact R1/X1/R0/X0 syntax via an",
        "! isolated single-branch OpenDSS sub-circuit before trusting the full",
        "! comparison -- see the commit that added this file).",
        "",
        "Clear",
        "Set DefaultBaseFrequency=60",
        "",
        "new circuit.IEEE69_Paper_Reconstruction",
        f"~ basekv={BASE_KV} pu=1.0 phases=3 bus1=BUS1",
        "~ MVAsc3=1e9 MVAsc1=1e9",
        "",
    ]

    for (f, t), xfmr in TRANSFORMER_LOCATIONS.items():
        r_pct = xfmr["r_pu"] * 100.0
        x_pct = xfmr["x_pu"] * 100.0
        lines.append(f"New Transformer.T{f}{t} Phases=3 Windings=2 Xhl={x_pct}")
        lines.append(f"~ wdg=1 bus=BUS{f} conn=wye   kv={BASE_KV} kva={BASE_MVA * 1000:.0f} %r={r_pct / 2}")
        lines.append(f"~ wdg=2 bus=BUS{t} conn=delta kv={BASE_KV} kva={BASE_MVA * 1000:.0f} %r={r_pct / 2}")
    lines.append("")

    for f, t, r, x in regular_branches():
        lines.append(
            f"New Line.L{f}_{t} Phases=3 Bus1=BUS{f}.1.2.3 Bus2=BUS{t}.1.2.3 "
            f"R1={r} X1={x} R0={r} X0={x} C1=0 C0=0 Length=1 units=none"
        )
    lines.append("")

    for i, pd_mw, qd_mw in BUS_LOADS_MW:
        if i == 1 or (pd_mw == 0.0 and qd_mw == 0.0):
            continue
        # kW/kvar here are the TOTAL three-phase load; a Phases=3 Conn=Wye
        # OpenDSS load object's own kW/kvar are also interpreted as TOTAL
        # (confirmed empirically -- see git history), split evenly across
        # its 3 phases, so no extra /3 is needed on this side: this
        # matches build_69bus_system()'s corrected per-phase pu conversion
        # (total_MW / S_base_MVA), which was NOT how an earlier version of
        # that function computed it (see that function's own comment).
        #
        # kV is LINE-TO-LINE for a Phases=3 Conn=Wye load (not
        # line-to-neutral) -- using the LN value here silently trips
        # OpenDSS's outside-deadband constant-impedance fallback and gives
        # wrong injected power, a bug independently caught while
        # diagnosing an early IEEE-13 test circuit.
        #
        # vminpu=0 vmaxpu=100 widens that same deadband to effectively
        # disable the fallback, so this circuit's Model=1 loads stay pure
        # constant-PQ at every voltage, matching
        # src/powerflow/mismatch.py's model exactly. (Checked directly:
        # this made a negligible difference once the real bug above was
        # fixed, but is kept for a strictly apples-to-apples comparison.)
        kw = pd_mw * 1000.0
        kvar = qd_mw * 1000.0
        lines.append(
            f"New Load.LD{i} Bus1=BUS{i}.1.2.3 Phases=3 Conn=Wye Model=1 "
            f"kV={BASE_KV} kW={kw:.4f} kvar={kvar:.4f} vminpu=0 vmaxpu=100"
        )
    lines.append("")
    lines.append(f"Set Voltagebases=[{BASE_KV}]")
    lines.append("calcv")
    lines.append("set maxiterations=100")
    lines.append("Solve")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "ieee69_paper_reconstruction.dss")
    with open(out_path, "w", newline="\n") as f:
        f.write(generate())
    print(f"wrote {out_path}")
