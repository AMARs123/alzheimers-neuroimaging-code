#!/bin/bash
#SBATCH --job-name=oasis_prep
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/p1/logs/oasis_prep_%j.out
#SBATCH --error=$HOME/alz/p1/logs/oasis_prep_%j.err
# OASIS-1: extract the processed T1 and the clinical text files, write NIfTI + oasis1_meta.csv (runs once, before 91_oasis_fs.sh).
echo "OASIS prep started: $(date) on $(hostname)"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
python $HOME/alz/p1/src/90_oasis_prep.py
echo "nii=$(ls $HOME/alz/oasis1/nii/*.nii.gz 2>/dev/null | wc -l)  Finished: $(date)"
