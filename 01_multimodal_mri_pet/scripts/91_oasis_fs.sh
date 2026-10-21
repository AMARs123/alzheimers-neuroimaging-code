#!/bin/bash
#SBATCH --job-name=oasis_fs
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/p1/logs/oasis_fs_%j.out
#SBATCH --error=$HOME/alz/p1/logs/oasis_fs_%j.err
# OASIS-1 external set: prepare NIfTI (resumable) and run FastSurfer --seg_only with the same options as the NACC cohort.
# Usage: sbatch 91_oasis_fs.sh <k> <n>   -> every n-th subject starting at k
echo "OASIS FastSurfer started: $(date) on $(hostname)"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export FASTSURFER_HOME=$HOME/alz/FastSurfer
K=${1:-0}; N=${2:-1}; O=$HOME/alz/oasis1; mkdir -p $O/fs_output
# NIfTI files are prepared beforehand by 90_oasis_prep.sh
i=0; ok=0; fail=0
for f in $(ls $O/nii/*.nii.gz | sort); do
  sid=$(basename $f .nii.gz); i=$((i+1))
  [ $(( (i-1) % N )) -ne "$K" ] && continue
  [ -f $O/fs_output/$sid/stats/aseg+DKT.VINN.stats ] && continue
  timeout 600 bash $FASTSURFER_HOME/run_fastsurfer.sh --t1 $f --sid $sid --sd $O/fs_output --seg_only --device cuda --no_cereb --threads 6 > $O/fs_output/$sid.log 2>&1
  if [ -f $O/fs_output/$sid/stats/aseg+DKT.VINN.stats ]; then ok=$((ok+1)); echo "$sid DONE"; else fail=$((fail+1)); echo "$sid FAIL"; tail -n 3 $O/fs_output/$sid.log; fi
done
echo "done=$ok fail=$fail total_stats=$(ls $O/fs_output/*/stats/aseg+DKT.VINN.stats 2>/dev/null | wc -l)"
echo "Finished: $(date)"
