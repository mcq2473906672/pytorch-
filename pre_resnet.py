import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights,resnet18, ResNet18_Weights


class PreResnet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(self.num_features, 10)

    def forward(self, x):
        out = self.model(x)
        return out


def pytorch_resnet34():
    return PreResnet34()

