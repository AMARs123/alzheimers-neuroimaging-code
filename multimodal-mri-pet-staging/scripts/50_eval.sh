#!/bin/bash
#SBATCH --job-name=p1_eval
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/multimodal-mri-pet-staging/logs/eval_%x_%j.out
#SBATCH --error=$HOME/alz/multimodal-mri-pet-staging/logs/eval_%x_%j.err
# usage: sbatch --job-name=EXP001 scripts/50_eval.sh --features ... --exp EXP001 ...   (arguments after the script name)
echo "Eval started: $(date) on $(hostname) cpus=$SLURM_CPUS_PER_TASK"
echo "ARGS: $@"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd ~/alz/multimodal-mri-pet-staging/src
mkdir -p ~/alz/multimodal-mri-pet-staging/experiments
python -c "import sys,sklearn,numpy,pandas,lightgbm;print('python',sys.version.split()[0],'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__,'lightgbm',lightgbm.__version__)"
python 50_eval_nested.py "$@"
echo "Finished: $(date)"
