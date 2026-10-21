#!/bin/bash
#SBATCH --job-name=p1_imp
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/p1/logs/imp_%x_%j.out
#SBATCH --error=$HOME/alz/p1/logs/imp_%x_%j.err
# usage: sbatch --job-name=EXP010 scripts/60_importance.sh --features ... --exp EXP010 --set mri+tau --task cn_ad --model lr
echo "Importance started: $(date) on $(hostname)"; echo "ARGS: $@"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/alz/p1/src:$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/alz/p1/src
python -c "import shap,sklearn;print('shap',shap.__version__,'sklearn',sklearn.__version__)"
python 60_importance.py "$@"
echo "Finished: $(date)"
