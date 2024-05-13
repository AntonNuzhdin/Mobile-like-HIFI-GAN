from enum import Enum

from .wandb import WanDBWriter


def get_visualizer():
    return WanDBWriter()
