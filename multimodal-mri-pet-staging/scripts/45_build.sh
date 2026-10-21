#!/bin/bash
#SBATCH --job-name=p1_build
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=$HOME/alz/multimodal-mri-pet-staging/logs/build_%j.out
#SBATCH --error=$HOME/alz/multimodal-mri-pet-staging/logs/build_%j.err
# usage: sbatch --export=ALL,TAG=v1 scripts/45_build.sh
module load pytorch/2.8.0+cu126 2>/dev/null || echo "module pytorch/2.8.0+cu126 unavailable, using the env directly"
export PATH=/home/apps/miniconda/envs/pytorch-2.8/bin:$PATH
export PYTHONPATH=$HOME/.local/lib/python3.10/site-packages:$PYTHONPATH
cd ~/alz/multimodal-mri-pet-staging/src
python 45_build_features.py ${TAG:-v1}
