#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python3}
RUNS=${RUNS:-5}
SEED=${SEED:-123}
OUTDIR=${OUTDIR:-results_baselines}

# One standard MLP configuration used across all datasets.
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-512}
EPOCHS=${EPOCHS:-1000}
LR=${LR:-0.001}
LOCAL_LAYERS=${LOCAL_LAYERS:-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0005}
DROPOUT=${DROPOUT:-0.5}
TSALLIS_Q=${TSALLIS_Q:-2.0}

DATASETS=(
  cora citeseer pubmed
  amazon-photo amazon-computer
  coauthor-cs coauthor-physics
  amazon-ratings roman-empire
  chameleon squirrel wikics minesweeper
)

mkdir -p "$OUTDIR"

dataset_flags() {
  local ds="$1"
  case "$ds" in
    cora|citeseer|pubmed)
      echo "--rand_split_class --valid_num 500 --test_num 1000"
      ;;
    minesweeper)
      echo "--metric rocauc"
      ;;
    *)
      echo ""
      ;;
  esac
}

run_one() {
  local ds="$1"; shift
  local tag="$1"; shift
  local outfile="$OUTDIR/mlp_${ds}_${tag}.txt"

  if [[ -f "$outfile" ]]; then
    echo "SKIP $outfile"
    return
  fi

  echo ">>> mlp $ds $tag"
  WANDB_TAGS="mlp-baseline" \
  "$PYTHON" medium_graph/main.py \
    --gnn mlp \
    --dataset "$ds" \
    --runs "$RUNS" \
    --seed "$SEED" \
    --hidden_channels "$HIDDEN_CHANNELS" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --local_layers "$LOCAL_LAYERS" \
    --weight_decay "$WEIGHT_DECAY" \
    --dropout "$DROPOUT" \
    $(dataset_flags "$ds") \
    "$@" 2>&1 | tee "$outfile"
}

for ds in "${DATASETS[@]}"; do
  run_one "$ds" "mlp-baseline"
done

echo "DONE: MLP baselines complete."
