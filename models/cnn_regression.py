from torch import nn
from .base_cnn import BaseCNN

class AgeRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = BaseCNN()
        self.fc_reg = nn.Linear(128, 1)
        self.type = "r"

    def forward(self, x):
        features = self.backbone(x)
        age = self.fc_reg(features)
        return age