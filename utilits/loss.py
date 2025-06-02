import torch
import torch.nn as nn
from pypots.nn.modules.loss import Criterion

class WeightedCrossEntropyLoss(Criterion):
    def __init__(self, weight_tensor):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    def forward(self, input, target):
        return self.loss_fn(input, target)
