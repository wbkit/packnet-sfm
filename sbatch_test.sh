#!/bin/bash
#SBATCH --output=sbatch_log/%j.out
#SBATCH --gres=gpu:1
##SBATCH --mem=20G
#SBATCH --constraint='geforce_gtx_titan_x'
#source /itet-stor/wboettcher/net_scratch/conda/bin/conda shell.bash hook

source /itet-stor/wboettcher/net_scratch/conda/etc/profile.d/conda.sh
conda activate torch-prod
python3 scripts/train.py configs/overfit_kitti.yaml "$@"
