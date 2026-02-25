#!/usr/bin/env python3
"""Generate sweep-negative YAMLs: same per-cell HPs as the v4 base but with
lambda_u sweeping the *negative* range [-1.0, -0.05] in steps of 0.05.

Reads sweeps/sweep_<ds>_<gnn>.yaml, copies into sweeps_negative/ with the
lambda_u grid replaced. Excludes questions (BCE branch bypasses Tsallis).
"""
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SWEEPS_IN = HERE / "sweeps"
SWEEPS_OUT = HERE / "sweeps_negative"
EXCLUDE_DATASETS = {"questions"}

# 20 values: -1.00, -0.95, ..., -0.05  (step 0.05)
LAMBDA_GRID = [round(-x * 0.05, 4) for x in range(20, 0, -1)]


def main():
    SWEEPS_OUT.mkdir(exist_ok=True)
    n = 0
    for src in sorted(SWEEPS_IN.glob("sweep_*.yaml")):
        m = re.match(r"sweep_(.+)_(gcn|gat|sage)\.yaml$", src.name)
        if not m:
            continue
        ds, gnn = m.group(1), m.group(2)
        if ds in EXCLUDE_DATASETS:
            continue
        cfg = yaml.safe_load(src.read_text())
        cfg["parameters"]["lambda_u"] = {"values": [float(x) for x in LAMBDA_GRID]}
        out = SWEEPS_OUT / src.name
        with out.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=200)
        n += 1
    print(f"Wrote {n} negative-lambda sweep YAMLs to {SWEEPS_OUT}", file=sys.stderr)
    print(f"lambda_u grid (20 values): {LAMBDA_GRID}", file=sys.stderr)


if __name__ == "__main__":
    main()
