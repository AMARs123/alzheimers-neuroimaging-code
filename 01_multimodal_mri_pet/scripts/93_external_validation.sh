#!/bin/bash
#SBATCH --job-name=exp023
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/p1/logs/exp023_%j.out
#SBATCH --error=$HOME/alz/p1/logs/exp023_%j.err
# OASIS-1 features (92) then frozen-NACC-model external validation (93) -> experiments/EXP023 (never overwritten).
echo "EXP023 started: $(date) on $(hostname)"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
n=0; for d in $HOME/alz/oasis1/fs_output/*/; do [ -f "$d/stats/aseg+DKT.VINN.stats" ] && n=$((n+1)); done
echo "FastSurfer complete: $n / $(ls $HOME/alz/oasis1/nii/*.nii.gz | wc -l)"
cd $HOME/alz/p1/src
python 92_oasis_features.py || exit 1
python 93_external_validation.py $HOME/alz/p1/work/features_v2.csv $HOME/alz/oasis1/features_oasis1.csv $HOME/alz/p1/experiments/EXP023
echo "Finished: $(date)"
