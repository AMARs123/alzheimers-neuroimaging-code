#!/bin/bash
#SBATCH --job-name=p1_deep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/multimodal-mri-pet-staging/logs/deep_%j.out
#SBATCH --error=$HOME/alz/multimodal-mri-pet-staging/logs/deep_%j.err
echo "Deep features started: $(date) on $(hostname)"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export TMPDIR=${SLURM_TMPDIR:-/tmp}
cd ~/alz/multimodal-mri-pet-staging/src
python 46_deep_features.py
echo "Finished: $(date)"
