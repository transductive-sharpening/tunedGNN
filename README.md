# Transductive Entropy Sharpening for GNNs

Fork of [TunedGNN](https://github.com/LUOyk1999/tunedGNN) (NeurIPS 2024) with transductive entropy sharpening added to the training loop.

## Setup

### Create and activate virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install PyTorch 2.7.0 (CPU or GPU)

Choose one of the following wheel tags:
- GPU (CUDA): ```cu128```
- CPU-only: ```cpu```

Set the tag:
```bash
export TORCH_TAG=cu128   # or: cpu
```

Install PyTorch:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/${TORCH_TAG}
```

### Install PyG compiled dependencies

The PyG wheels must match the same Torch tag:

```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.7.0+${TORCH_TAG}.html
pip install torch_geometric
```

### Install remaining dependencies

```bash
pip install -r requirements.txt
```

## New CLI arguments

| Flag                  | Default | Description                                             |
|-----------------------|---------|---------------------------------------------------------|
| `--lambda_u`          | 0.0     | Entropy weight on unlabeled nodes (positive = minimize) |
| `--lambda_t`          | 0.0     | Entropy weight on   labeled nodes (negative = maximize) |
| `--tsallis_q`         | 1.0     | Tsallis q parameter (1 = Shannon, 2 = Gini impurity)    |
| `--warmup_frac`       | 0.0     | Fraction of epochs to linearly ramp entropy reg         |
| `--mi-max`            | 0.0     | MI class-balance weight (0 = off, 1 = standard MI)      |

## Example: CiteSeer with Tsallis q=2

```bash
cd medium_graph

python main.py --gnn gcn --dataset citeseer --lr 0.001 --runs 10 \
  --local_layers 2 --hidden_channels 512 --weight_decay 0.01 --dropout 0.5 \
  --rand_split_class --valid_num 500 --test_num 1000 --seed 123 \
  --tsallis_q 2 --lambda_u 0.25 --lambda_t -0.25
```

To run the baseline, omit the `--tsallis_q`, `--lambda_u`, and `--lambda_t` flags.
