#!/usr/bin/env python3
from pathlib import Path
import yaml

OUT = Path("medium_graph/sweep_baselines")
OUT.mkdir(exist_ok=True)

PROJECT = "transductive-sharpening"
ENTITY = ""

DATASETS = [
    "cora", "citeseer", "pubmed",
    "amazon-photo", "amazon-computer",
    "coauthor-cs", "coauthor-physics",
    "amazon-ratings", "roman-empire",
    "chameleon", "squirrel", "wikics", "minesweeper",
]

LAMBDAS = [round(i * 0.05, 2) for i in range(41)]

BASE_PARAMS = {
    "gnn": {"value": "mlp"},
    "seed": {"value": 123},
    "device": {"value": 0},
    "runs": {"value": 5},

    # Standard MLP config across datasets
    "hidden_channels": {"value": 512},
    "epochs": {"value": 1000},
    "lr": {"value": 0.001},
    "local_layers": {"value": 3},
    "weight_decay": {"value": 0.0005},
    "dropout": {"value": 0.5},

    # TS config
    "tsallis_q": {"value": 2.0},
    "warmup_frac": {"value": 0.0},
    "lambda_t": {"value": 0.0},
    "mi-max": {"value": 0.0},
    "lambda_u": {"values": LAMBDAS},
}

def command_for_dataset(ds: str):
    cmd = [
        "${env}",
        "${interpreter}",
        "${program}",
        "${args}",
        "--tie_lambda_t",
    ]

    if ds in {"cora", "citeseer", "pubmed"}:
        cmd.append("--rand_split_class")

    return cmd

def params_for_dataset(ds: str):
    params = dict(BASE_PARAMS)
    params["dataset"] = {"value": ds}

    if ds in {"cora", "citeseer", "pubmed"}:
        params["valid_num"] = {"value": 500}
        params["test_num"] = {"value": 1000}

    if ds == "minesweeper":
        params["metric"] = {"value": "rocauc"}

    return params

for ds in DATASETS:
    cfg = {
        "program": "medium_graph/main.py",
        "method": "grid",
        "project": PROJECT,
        "entity": ENTITY,
        "name": f"mlp_ts_{ds}",
        "metric": {
            "name": "best_val",
            "goal": "maximize",
        },
        "parameters": params_for_dataset(ds),
        "command": command_for_dataset(ds),
    }

    out = OUT / f"sweep_mlp_ts_{ds}.yaml"
    with out.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, width=200)

    print(f"Wrote {out}")

print(f"\nDone. Generated sweeps in {OUT}/")