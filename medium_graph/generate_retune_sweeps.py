#!/usr/bin/env python3
"""Generate 5x5 re-tune sweeps at v4-best-lambda for each cell.

For each (dataset, gnn), produce sweeps_retune/sweep_<ds>_<gnn>.yaml with:
  - lambda_u fixed at v4-best-lambda
  - lr grid:      orig * {0.25, 0.5, 1, 2, 4}      (5 values, multiplicative)
  - dropout grid: orig + {-0.1, -0.05, 0, 0.05, 0.1}  (5 values, additive)

Excludes questions (no v4-best-lambda; BCE branch bypasses Tsallis).
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict
import yaml

BASE = Path(__file__).resolve().parent
SWEEPS_IN = BASE / "sweeps"
SWEEPS_OUT = BASE / "sweeps_retune"
LAMBDA_SNAPSHOT = BASE / "sweeps_retune_lambda.csv"

LR_OFFSETS = [0.25, 0.5, 1.0, 2.0, 4.0]
DR_OFFSETS = [-0.1, -0.05, 0.0, 0.05, 0.1]
EXCLUDE_DATASETS = {"questions"}


def best_lambda_per_cell():
    """Return {(ds, gnn): best_lambda_u} from the snapshot CSV.

    The snapshot was produced from analysis/reports/runs_v4.csv (the
    fetch_runs.py output) at retune-design time. Committed so the
    generator is reproducible without re-querying W&B.
    """
    out = {}
    with open(LAMBDA_SNAPSHOT) as f:
        for r in csv.DictReader(f):
            out[(r["dataset"], r["gnn"])] = float(r["best_lambda_u"])
    return out


def transform(src_yaml_text, best_lambda):
    cfg = yaml.safe_load(src_yaml_text)
    params = cfg["parameters"]

    orig_lr = float(params["lr"]["value"])
    orig_dr = float(params["dropout"]["value"])

    lr_grid = sorted({round(orig_lr * m, 8) for m in LR_OFFSETS})
    dr_grid = sorted({round(orig_dr + o, 4) for o in DR_OFFSETS})

    params["lr"] = {"values": [float(x) for x in lr_grid]}
    params["dropout"] = {"values": [float(x) for x in dr_grid]}
    params["lambda_u"] = {"value": float(best_lambda)}
    return cfg, orig_lr, lr_grid, orig_dr, dr_grid


def dump_yaml(cfg, path):
    with path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None, width=200)


def main():
    SWEEPS_OUT.mkdir(exist_ok=True)
    best = best_lambda_per_cell()

    rows = []
    for src in sorted(SWEEPS_IN.glob("sweep_*.yaml")):
        stem = src.stem
        try:
            _, ds, gnn = stem.split("_", 2)
        except ValueError:
            continue
        if ds in EXCLUDE_DATASETS:
            print(f"skip {src.name}: excluded (no v4 data)", file=sys.stderr)
            continue
        if (ds, gnn) not in best:
            print(f"skip {src.name}: no v4-best-λ for ({ds}, {gnn})", file=sys.stderr)
            continue
        lu = best[(ds, gnn)]
        cfg, orig_lr, lr_grid, orig_dr, dr_grid = transform(src.read_text(), lu)
        out = SWEEPS_OUT / src.name
        dump_yaml(cfg, out)
        rows.append((ds, gnn, lu, orig_lr, lr_grid, orig_dr, dr_grid))

    print(f"\nGenerated {len(rows)} retune sweep YAMLs in {SWEEPS_OUT}")
    print(f"\n{'Cell':<25} {'best λ':>7} {'orig lr':>9} {'lr grid':>40}  {'orig dr':>7} {'dr grid':>30}")
    for ds, gnn, lu, olr, lrg, odr, drg in rows:
        lr_str = "[" + ", ".join(f"{x:g}" for x in lrg) + "]"
        dr_str = "[" + ", ".join(f"{x:g}" for x in drg) + "]"
        print(f"  {ds + ' ' + gnn:<25} {lu:7.2f} {olr:9.4f} {lr_str:>40}  {odr:7.2f} {dr_str:>30}")
    n_combos = sum(len(lrg) * len(drg) for *_, lrg, _, drg in rows)
    print(f"\nTotal HP combinations: {n_combos}")


if __name__ == "__main__":
    main()
