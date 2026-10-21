#!/bin/bash
#SBATCH --job-name=p1_stats
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/multimodal-mri-pet-staging/logs/stats_%j.out
#SBATCH --error=$HOME/alz/multimodal-mri-pet-staging/logs/stats_%j.err
# Bootstrap CIs / paired comparisons (80_stats.py) and figures (81_figures.py) from stored predictions; one process per experiment.
echo "Stats started: $(date) on $(hostname)"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/alz/multimodal-mri-pet-staging/src:$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/alz/multimodal-mri-pet-staging; E=experiments; S=results/statistics; F=figures/experimental; mkdir -p $S $F
B=${1:-1000}
run() { python src/80_stats.py $E/$1 $S --boot $B --pairs "$2" > logs/stats_$1.log 2>&1 & }
run EXP007 "mri:mri+amy,amy:mri+amy"
run EXP008 "mri:mri+amy,amy:mri+amy"
run EXP019 ""
run EXP020 "mri:mri+tau,tau:mri+tau"
run EXP021 "mri:mri+thk,thk:mri+thk,mri:thk"
run EXP022 "mri+tau:mri+thk+tau,mri+amy+tau:mri+thk+amy+tau,mri+tau:mri+amy+tau"
for x in EXP005 EXP006 EXP007 EXP008 EXP020 EXP021 EXP022; do python src/81_figures.py $E/$x $F --model lr > logs/fig_$x.log 2>&1 & done
python src/81_figures.py $E/EXP006 $F --model stack > logs/fig_EXP006_stack.log 2>&1 &
wait
tail -n 2 logs/stats_EXP0*.log logs/fig_EXP0*.log
echo "Finished: $(date)"
