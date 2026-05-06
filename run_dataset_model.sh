#!/bin/bash
#SBATCH -J cora_sage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mail-type=NONE

source .venv/bin/activate

export ENTITY=""

srun 