#!/bin/bash
#SBATCH --output=/scratch_net/biwidl213/wboettcher/Logs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --constraint='geforce_gtx_titan_x'
#source /itet-stor/wboettcher/net_scratch/conda/bin/conda shell.bash hook

source /itet-stor/wboettcher/net_scratch/conda/etc/profile.d/conda.sh
conda activate torch-prod2
source ./copy_script_for_sbatch.sh
python3 scripts/train.py configs/train_packnet_san_kitti_cont.yaml "$@"
#python3 scripts/train.py ../Checkpoints/SAN_Net_res/KITTI_raw-eigen_test23.ckpt
#python3 scripts/train.py ./data/resSAN2_epoch12.ckpt