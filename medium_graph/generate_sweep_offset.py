#!/usr/bin/env python3
"""Generate sweep-offset YAMLs: offset sweep at val-best lambda per cell.

Each (gnn, dataset) cell is run at its val-best lambda* with --tie_lambda_t,
sweeping --midpoint_offset across {-0.10, -0.05, +0.05, +0.10}. Result:
asymmetric (lambda_u, lambda_t) = (lambda* + delta, -lambda* + delta) pairs
that preserve the spread (2*lambda*) but shift the midpoint by delta.

Cells with val-best-lambda = 0 are skipped: spread = 0 means there's no
asymmetry to test (it would just sweep symmetric (delta, delta) pairs,
duplicating other experiments).

Reads:  sweeps/sweep_<ds>_<gnn>.yaml      (per-cell HPs)
        sweeps_retune_lambda.csv          (val-best lambda* per cell)
Writes: sweeps_offset/sweep_<ds>_<gnn>.yaml
"""
import csv
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
SWEEPS_IN = HERE / "sweeps"
SWEEPS_OUT = HERE / "sweeps_offset"
LAMBDA_SNAPSHOT = HERE / "sweeps_retune_lambda.csv"
EXCLUDE_DATASETS = {"questions"}

OFFSET_GRID = [-0.10, -0.05, 0.05, 0.10]


def best_lambda_per_cell():
    out = {}
    with open(LAMBDA_SNAPSHOT) as f:
        for r in csv.DictReader(f):
            out[(r["dataset"], r["gnn"])] = float(r["best_lambda_u"])
    return out


def main():
    SWEEPS_OUT.mkdir(exist_ok=True)
    best = best_lambda_per_cell()

    rows = []
    skipped_zero = []
    for src in sorted(SWEEPS_IN.glob("sweep_*.yaml")):
        m = re.match(r"sweep_(.+)_(gcn|gat|sage)\.yaml$", src.name)
        if not m:
            continue
        ds, gnn = m.group(1), m.group(2)
        if ds in EXCLUDE_DATASETS:
            continue
        if (ds, gnn) not in best:
            print(f"skip {src.name}: no val-best-lambda", file=sys.stderr)
            continue
        lstar = best[(ds, gnn)]
        if abs(lstar) < 1e-9:
            skipped_zero.append((ds, gnn))
            continue

        cfg = yaml.safe_load(src.read_text())
        cfg["parameters"]["lambda_u"] = {"value": float(lstar)}
        cfg["parameters"]["midpoint_offset"] = {
            "values": [float(x) for x in OFFSET_GRID]
        }
        out = SWEEPS_OUT / src.name
        with out.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=200)
        rows.append((ds, gnn, lstar))

    print(f"\nWrote {len(rows)} offset (asymmetric) sweep YAMLs to {SWEEPS_OUT}", file=sys.stderr)
    print(f"midpoint_offset grid: {OFFSET_GRID}", file=sys.stderr)
    if skipped_zero:
        print(f"\nSkipped {len(skipped_zero)} cells with val-best-lambda=0 "
              f"(no Tsallis spread to offset):", file=sys.stderr)
        for ds, gnn in skipped_zero:
            print(f"  {ds} {gnn}", file=sys.stderr)
    print(f"\nTotal sweep registrations: {len(rows)}", file=sys.stderr)
    print(f"Total runs (4 offsets per cell): {len(rows) * len(OFFSET_GRID)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
