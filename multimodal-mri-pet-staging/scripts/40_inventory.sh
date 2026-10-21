#!/bin/bash
#SBATCH --job-name=p1_inv
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=$HOME/alz/multimodal-mri-pet-staging/logs/inv_%j.out
#SBATCH --error=$HOME/alz/multimodal-mri-pet-staging/logs/inv_%j.err
echo "Inventory started: $(date) on $(hostname)"
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
cd ~/alz/multimodal-mri-pet-staging/src
python -c "import sys,pydicom,nibabel,ants,sklearn,numpy,pandas,scipy; print('python',sys.version.split()[0],'pydicom',pydicom.__version__,'nibabel',nibabel.__version__,'ants',ants.__version__,'sklearn',sklearn.__version__,'numpy',numpy.__version__,'pandas',pandas.__version__)"
which dcm2niix; dcm2niix -v 2>&1 | head -2
python -c "import lightgbm,xgboost,catboost; print('lgbm',lightgbm.__version__,'xgb',xgboost.__version__,'cat',catboost.__version__)"
python 40_inventory.py PET MRI
echo "Finished: $(date)"
