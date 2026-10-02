# src/validation/antifloat_study.py
from src.validation.opendss import compile_and_solve, export_csv

# ppm_antifloat is a per-Transformer property, not a global Set option --
# it directly scales the tiny anti-float admittance stamped into that
# transformer's own primitive Y-matrix diagonal (EPRI docs, "Modeling the
# OY-OD Connection"). Must target the transformer by name: XFMR1, as
# defined in ieee4_reconstruction.dss.
SETTINGS_TO_TEST = [
    ("default", None),                                   # OpenDSS's built-in default (1 ppm)
    ("disabled", "Edit Transformer.XFMR1 ppm_antifloat=0"),
    ("elevated", "Edit Transformer.XFMR1 ppm_antifloat=100"),
]

CASE_B_COMMANDS = ["Edit Load.LOAD_BUS3 kw=0 kvar=0", "Edit Load.LOAD_BUS4 kw=0 kvar=0"]


def run_sweep():
    for label, cmd in SETTINGS_TO_TEST:
        extra = list(CASE_B_COMMANDS)
        if cmd:
            extra.append(cmd)
        result = compile_and_solve("data/raw/ieee4/ieee4_reconstruction.dss",
                                    extra_commands=extra)
        print(f"[{label:8s}] converged={result.converged}  "
              f"iterations={result.iterations}")
        export_csv(result, f"results/antifloat_{label}_caseB.csv")


if __name__ == "__main__":
    run_sweep()