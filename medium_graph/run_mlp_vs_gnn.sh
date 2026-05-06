#!/bin/bash
# MLP vs GNN comparison: auto-generated from TunedGNN CONFIGS.
# 6 conditions x 13 datasets x 10 runs.
set -e
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
RUNS=${RUNS:-10}
OUTDIR=results_mlp_vs_gnn
mkdir -p "$OUTDIR"

run_one() {
    local gnn=$1 ds=$2 tag=$3; shift 3
    local outfile="$OUTDIR/${gnn}_${ds}_${tag}.txt"
    if [ -f "$outfile" ]; then echo "SKIP $outfile"; return; fi
    echo ">>> $gnn $ds $tag"
    $PYTHON main.py --gnn "$gnn" --dataset "$ds" --runs "$RUNS" "$@" 2>&1 | tee "$outfile"
}

# === cora ===
run_one mlp cora baseline --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123
run_one mlp cora ts --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp cora ts_nolt --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --tsallis_q 2.0
run_one mlp cora mi --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn cora baseline --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 3 --weight_decay 0.0005 --dropout 0.7 --rand_split_class --seed 123
run_one gcn cora ts --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 3 --weight_decay 0.0005 --dropout 0.7 --rand_split_class --seed 123 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === citeseer ===
run_one mlp citeseer baseline --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --seed 123
run_one mlp citeseer ts --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp citeseer ts_nolt --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --tsallis_q 2.0
run_one mlp citeseer mi --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn citeseer baseline --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --seed 123
run_one gcn citeseer ts --hidden_channels 512 --epochs 500 --lr 0.001 --local_layers 2 --weight_decay 0.01 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === pubmed ===
run_one mlp pubmed baseline --hidden_channels 256 --epochs 500 --lr 0.005 --local_layers 2 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123
run_one mlp pubmed ts --hidden_channels 256 --epochs 500 --lr 0.005 --local_layers 2 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp pubmed ts_nolt --hidden_channels 256 --epochs 500 --lr 0.005 --local_layers 2 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 0.25 --tsallis_q 2.0
run_one mlp pubmed mi --hidden_channels 256 --epochs 500 --lr 0.005 --local_layers 2 --weight_decay 0.0005 --dropout 0.5 --rand_split_class --seed 123 --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn pubmed baseline --hidden_channels 256 --epochs 500 --lr 0.005 --local_layers 2 --weight_decay 0.0005 --dropout 0.7 --rand_split_class --seed 123
run_one gcn pubmed ts --hidden_channels 256 --epochs 500 --lr 0.005 --local_layers 2 --weight_decay 0.0005 --dropout 0.7 --rand_split_class --seed 123 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === amazon-photo ===
run_one mlp amazon-photo baseline --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln
run_one mlp amazon-photo ts --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp amazon-photo ts_nolt --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 0.25 --tsallis_q 2.0
run_one mlp amazon-photo mi --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn amazon-photo baseline --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 6 --weight_decay 5e-05 --dropout 0.5 --ln --res
run_one gcn amazon-photo ts --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 6 --weight_decay 5e-05 --dropout 0.5 --ln --res --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === amazon-computer ===
run_one mlp amazon-computer baseline --hidden_channels 512 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln
run_one mlp amazon-computer ts --hidden_channels 512 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp amazon-computer ts_nolt --hidden_channels 512 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 0.25 --tsallis_q 2.0
run_one mlp amazon-computer mi --hidden_channels 512 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn amazon-computer baseline --hidden_channels 512 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln
run_one gcn amazon-computer ts --hidden_channels 512 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 5e-05 --dropout 0.5 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === coauthor-cs ===
run_one mlp coauthor-cs baseline --hidden_channels 512 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln
run_one mlp coauthor-cs ts --hidden_channels 512 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp coauthor-cs ts_nolt --hidden_channels 512 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --lambda_u 0.25 --tsallis_q 2.0
run_one mlp coauthor-cs mi --hidden_channels 512 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn coauthor-cs baseline --hidden_channels 512 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --res
run_one gcn coauthor-cs ts --hidden_channels 512 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --res --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === coauthor-physics ===
run_one mlp coauthor-physics baseline --hidden_channels 64 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln
run_one mlp coauthor-physics ts --hidden_channels 64 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp coauthor-physics ts_nolt --hidden_channels 64 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --lambda_u 0.25 --tsallis_q 2.0
run_one mlp coauthor-physics mi --hidden_channels 64 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn coauthor-physics baseline --hidden_channels 64 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --res
run_one gcn coauthor-physics ts --hidden_channels 64 --epochs 1500 --lr 0.001 --local_layers 2 --weight_decay 0.0005 --dropout 0.3 --ln --res --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === amazon-ratings ===
run_one mlp amazon-ratings baseline --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn
run_one mlp amazon-ratings ts --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp amazon-ratings ts_nolt --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn --lambda_u 0.25 --tsallis_q 2.0
run_one mlp amazon-ratings mi --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn amazon-ratings baseline --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --bn --res
run_one gcn amazon-ratings ts --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 4 --weight_decay 0.0 --dropout 0.5 --bn --res --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === roman-empire ===
run_one mlp roman-empire baseline --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn
run_one mlp roman-empire ts --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp roman-empire ts_nolt --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn --lambda_u 0.25 --tsallis_q 2.0
run_one mlp roman-empire mi --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --bn --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn roman-empire baseline --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --bn --res --pre_linear
run_one gcn roman-empire ts --hidden_channels 512 --epochs 2500 --lr 0.001 --local_layers 9 --weight_decay 0.0 --dropout 0.5 --bn --res --pre_linear --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === chameleon ===
run_one mlp chameleon baseline --hidden_channels 512 --epochs 200 --lr 0.005 --local_layers 3 --weight_decay 0.001 --dropout 0.2
run_one mlp chameleon ts --hidden_channels 512 --epochs 200 --lr 0.005 --local_layers 3 --weight_decay 0.001 --dropout 0.2 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp chameleon ts_nolt --hidden_channels 512 --epochs 200 --lr 0.005 --local_layers 3 --weight_decay 0.001 --dropout 0.2 --lambda_u 0.25 --tsallis_q 2.0
run_one mlp chameleon mi --hidden_channels 512 --epochs 200 --lr 0.005 --local_layers 3 --weight_decay 0.001 --dropout 0.2 --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn chameleon baseline --hidden_channels 512 --epochs 200 --lr 0.005 --local_layers 5 --weight_decay 0.001 --dropout 0.2
run_one gcn chameleon ts --hidden_channels 512 --epochs 200 --lr 0.005 --local_layers 5 --weight_decay 0.001 --dropout 0.2 --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === squirrel ===
run_one mlp squirrel baseline --hidden_channels 256 --epochs 500 --lr 0.01 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --bn
run_one mlp squirrel ts --hidden_channels 256 --epochs 500 --lr 0.01 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --bn --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp squirrel ts_nolt --hidden_channels 256 --epochs 500 --lr 0.01 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --bn --lambda_u 0.25 --tsallis_q 2.0
run_one mlp squirrel mi --hidden_channels 256 --epochs 500 --lr 0.01 --local_layers 3 --weight_decay 0.0005 --dropout 0.5 --bn --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn squirrel baseline --hidden_channels 256 --epochs 500 --lr 0.01 --local_layers 4 --weight_decay 0.0005 --dropout 0.7 --bn --res
run_one gcn squirrel ts --hidden_channels 256 --epochs 500 --lr 0.01 --local_layers 4 --weight_decay 0.0005 --dropout 0.7 --bn --res --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === wikics ===
run_one mlp wikics baseline --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --ln
run_one mlp wikics ts --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp wikics ts_nolt --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --ln --lambda_u 0.25 --tsallis_q 2.0
run_one mlp wikics mi --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --ln --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn wikics baseline --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --ln
run_one gcn wikics ts --hidden_channels 256 --epochs 1000 --lr 0.001 --local_layers 3 --weight_decay 0.0 --dropout 0.5 --ln --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

# === minesweeper ===
run_one mlp minesweeper baseline --hidden_channels 64 --epochs 2000 --lr 0.01 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --bn --metric rocauc
run_one mlp minesweeper ts --hidden_channels 64 --epochs 2000 --lr 0.01 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --bn --metric rocauc --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0
run_one mlp minesweeper ts_nolt --hidden_channels 64 --epochs 2000 --lr 0.01 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --bn --metric rocauc --lambda_u 0.25 --tsallis_q 2.0
run_one mlp minesweeper mi --hidden_channels 64 --epochs 2000 --lr 0.01 --local_layers 3 --weight_decay 0.0 --dropout 0.2 --bn --metric rocauc --lambda_u 1.0 --tsallis_q 2.0 --mi-max 1.0
run_one gcn minesweeper baseline --hidden_channels 64 --epochs 2000 --lr 0.01 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --bn --res --metric rocauc
run_one gcn minesweeper ts --hidden_channels 64 --epochs 2000 --lr 0.01 --local_layers 12 --weight_decay 0.0 --dropout 0.2 --bn --res --metric rocauc --lambda_u 0.25 --lambda_t -0.25 --tsallis_q 2.0

echo "DONE: all experiments complete."
