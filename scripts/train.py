# Copyright 2020 Toyota Research Institute.  All rights reserved.
import sys
sys.path.append(".")
print(sys.path)

import argparse

from packnet_sfm.models.model_wrapper import ModelWrapper
from packnet_sfm.models.model_checkpoint import ModelCheckpoint
from packnet_sfm.trainers.base_trainer import BaseTrainer
from packnet_sfm.utils.config import parse_train_file
from packnet_sfm.utils.load import set_debug, filter_args_create
# from packnet_sfm.utils.horovod import hvd_init, rank
from packnet_sfm.loggers import WandbLogger
import torch
import torch.nn as nn


def parse_args():
    """Parse arguments for training script"""
    parser = argparse.ArgumentParser(description='PackNet-SfM training script')
    parser.add_argument('file', type=str, help='Input file (.ckpt or .yaml)')
    args = parser.parse_args()
    assert args.file.endswith(('.ckpt', '.yaml')), \
        'You need to provide a .ckpt of .yaml file'
    return args


def train(file):
    """
    Monocular depth estimation training script.

    Parameters
    ----------
    file : str
        Filepath, can be either a
        **.yaml** for a yacs configuration file or a
        **.ckpt** for a pre-trained checkpoint file.
    """
    # Initialize horovod
    # hvd_init()

    # Produce configuration and checkpoint from filename
    config, ckpt = parse_train_file(file)

    # NOTE Quick-fix of loss name
    config.checkpoint.monitor = 'KITTI_raw-eigen_test_files-groundtruth-rmse_pp_gt'
    # NOTE: For SAN checkpoint
    # config['config'] = 'configs/train_packnet_san_kitti.yaml'
    # config.wandb['dir'] = 'wandb'
    # config.wandb['entity'] = 'wbeth'
    # config.wandb['project'] = 'CVL_project'
    # config.checkpoint['s3_path'] =''
    # config.checkpoint['s3_url'] = ''
    # config.checkpoint['filepath'] = '/scratch_net/biwidl213/wboettcher/Checkpoints'
    # if config.model.depth_net.name == 'PackNetSlimEnc01':
    #     config.model.depth_net.name = 'PackNetSAN01'
    # config.datasets.train['path'] = ['/scratch/wboet/KITTI_raw/']
    # config.datasets.validation['path'] = ['/scratch/wboet/KITTI_raw/', '/scratch/wboet/KITTI_raw/']
    # config.datasets.validation['input_depth_type'] = ['velodyne','']
    
    #config2, ckpt2 = parse_train_file('configs/train_packnet_san_kitti.yaml')
    # config2.proj.filter.type = 'none' #'None', 'to_from_ch', 'modulo_ch'
    # config2.proj.filter.from_ch = 35
    # config2.proj.filter.to_ch = 43
    # config2.proj.filter.modulo_value = 2

    # Set debug if requested
    set_debug(config.debug)

    # Wandb Logger
    logger = None if config.wandb.dry_run \
        else filter_args_create(WandbLogger, config.wandb)

    # model checkpoint
    checkpoint = None if config.checkpoint.filepath == '' else \
        filter_args_create(ModelCheckpoint, config.checkpoint)

    # Initialize model wrapper
    model_wrapper = ModelWrapper(config, resume=ckpt, logger=logger, **config.proj)

    # Create trainer with args.arch parameters
    trainer = BaseTrainer(**config.arch, checkpoint=checkpoint, **config.proj)

    # Train model
    trainer.fit(model_wrapper)


if __name__ == '__main__':
    args = parse_args()
    train(args.file)
