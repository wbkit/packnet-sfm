"""
Trainers
========

Trainer classes providing an easy way to train and evaluate SfM models
when wrapped in a ModelWrapper.

Inspired by pytorch-lightning.

"""

# from packnet_sfm.trainers.horovod_trainer import HorovodTrainer
from packnet_sfm.trainers.base_trainer import BaseTrainer

# __all__ = ["HorovodTrainer"]
__all__ = ["BaseTrainer"]