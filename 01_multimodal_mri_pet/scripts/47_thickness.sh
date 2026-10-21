#!/bin/bash
#SBATCH --job-name=p1_thk
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=150G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/p1/logs/thk_%x_%j.out
#SBATCH --error=$HOME/alz/p1/logs/thk_%x_%j.err
# usage: sbatch [--partition=himadri --time=06:00:00] scripts/47_thickness.sh [test N]
echo "Thickness started: $(date) on $(hostname) cpus=$SLURM_CPUS_PER_TASK"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export TMPDIR=${SLURM_TMPDIR:-/tmp}
cd ~/alz/p1/src
python 47_thickness.py "$@"
echo "Finished: $(date)"
