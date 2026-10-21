#!/bin/bash
#SBATCH --job-name=p1_pet
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/p1/logs/pet_%x_%j.out
#SBATCH --error=$HOME/alz/p1/logs/pet_%x_%j.err
# usage: sbatch --job-name=p1_TAU --export=ALL,KIND=TAU scripts/41_pet_extract.sh
echo "PET extract $KIND started: $(date) on $(hostname) cpus=$SLURM_CPUS_PER_TASK"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export PATH=$HOME/.local/bin:$PATH
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=2
export TMPDIR=${SLURM_TMPDIR:-/tmp}
cd ~/alz/p1/src
python 41_pet_extract.py $KIND
echo "Finished: $(date)"
