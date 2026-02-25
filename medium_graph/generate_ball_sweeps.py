#!/usr/bin/env python3
"""Generate lr x dropout "ball" sweeps around TunedGNN's best hyperparameters.

For each existing sweep in medium_graph/sweeps/, produce a sibling in
medium_graph/sweeps_ball/ that:
  - fixes lambda_u at the best value from the prior lambda_u sweep
    (extracted from reports/results_table.tex)
  - sweeps lr and dropout over TunedGNN's grid neighbors of the best value:
      lr grid       = [0.001, 0.005, 0.01]
      dropout grid  = [0.2, 0.3, 0.5, 0.7]
    The ball for a best value v is {v-1, v, v+1} clipped to the grid.
"""
from pathlib import Path
import re
import yaml

BASE = Path(__file__).resolve().parent
SWEEPS_IN = BASE / "sweeps"
SWEEPS_OUT = BASE / "sweeps_ball"
TABLE_TEX = BASE.parent / "reports" / "results_table.tex"

LR_GRID = [0.001, 0.005, 0.01]
DROPOUT_GRID = [0.2, 0.3, 0.5, 0.7]

LABEL_TO_KEY = {
    "Cora": "cora", "CiteSeer": "citeseer", "PubMed": "pubmed",
    "Computers": "amazon-computer", "Photo": "amazon-photo",
    "CS": "coauthor-cs", "Physics": "coauthor-physics",
    "WikiCS": "wikics", "Amazon-Rat.": "amazon-ratings",
    "Minesweeper": "minesweeper", "Roman-Emp.": "roman-empire",
    "Squirrel": "squirrel", "Chameleon": "chameleon", "Questions": "questions",
}
GNNS = ["gcn", "gat", "sage"]


def neighbor_ball(val, grid):
    # closest index on the grid (floats — compare with tolerance)
    i = min(range(len(grid)), key=lambda k: abs(grid[k] - val))
    return grid[max(0, i - 1): i + 2]


def parse_best_lambdas(tex_path):
    """Return {(dataset, gnn): best_lambda_u}."""
    text = Path(tex_path).read_text()
    best = {}
    for line in text.splitlines():
        m = re.match(r"^([^&]+?) & ", line)
        if not m:
            continue
        label = m.group(1).strip()
        if label not in LABEL_TO_KEY:
            continue
        key = LABEL_TO_KEY[label]
        lambdas = re.findall(r"lambda\$=([\d.]+)", line)
        if len(lambdas) != 3:
            raise ValueError(f"expected 3 lambdas for {label}, got {len(lambdas)}")
        for gnn, lu in zip(GNNS, lambdas):
            best[(key, gnn)] = float(lu)
    return best


def transform(src_yaml_text, best_lambda):
    cfg = yaml.safe_load(src_yaml_text)
    params = cfg["parameters"]

    old_lr = float(params["lr"]["value"])
    old_dropout = float(params["dropout"]["value"])
    lr_ball = neighbor_ball(old_lr, LR_GRID)
    dropout_ball = neighbor_ball(old_dropout, DROPOUT_GRID)

    params["lr"] = {"values": lr_ball}
    params["dropout"] = {"values": dropout_ball}
    params["lambda_u"] = {"value": best_lambda}

    return cfg, (old_lr, lr_ball), (old_dropout, dropout_ball)


def dump_yaml(cfg, path):
    # Preserve simple readable formatting
    with path.open("w") as f:
        yaml.safe_dump(
            cfg, f, sort_keys=False, default_flow_style=None, width=200
        )


def main():
    best_lambdas = parse_best_lambdas(TABLE_TEX)
    SWEEPS_OUT.mkdir(exist_ok=True)

    for src in sorted(SWEEPS_IN.glob("sweep_*.yaml")):
        stem = src.stem  # sweep_<dataset>_<gnn>
        _, dataset, gnn = stem.split("_", 2)
        key = (dataset, gnn)
        if key not in best_lambdas:
            print(f"skip {src.name}: no best lambda for {key}")
            continue
        lu = best_lambdas[key]
        cfg, lr_info, dp_info = transform(src.read_text(), lu)
        out = SWEEPS_OUT / src.name
        dump_yaml(cfg, out)
        print(
            f"{src.name:40s}  lambda_u={lu:.2f}  "
            f"lr {lr_info[0]}->{lr_info[1]}  dropout {dp_info[0]}->{dp_info[1]}"
        )


if __name__ == "__main__":
    main()
