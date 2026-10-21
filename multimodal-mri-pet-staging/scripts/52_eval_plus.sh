#!/bin/bash
#SBATCH --job-name=p1_plus
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/multimodal-mri-pet-staging/logs/plus_%x_%j.out
#SBATCH --error=$HOME/alz/multimodal-mri-pet-staging/logs/plus_%x_%j.err
echo "Eval-plus started: $(date) on $(hostname) cpus=$SLURM_CPUS_PER_TASK"; echo "ARGS: $@"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/alz/multimodal-mri-pet-staging/src:$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/alz/multimodal-mri-pet-staging/src
python 52_eval_plus.py "$@"
echo "Finished: $(date)"
