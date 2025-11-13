import torch.nn as nn
from models.finetune import Finetune
from utils.simple_mlp import StarsSimpleCosineIncrementalNet, StarsIncrementalNet


class StarsFinetune(Finetune):
    def __init__(self, args):
        super().__init__(args)
        self._network = StarsIncrementalNet(args, False)