from torch import nn
from .base_cnn import BaseCNN

class AgeMultiModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = BaseCNN()
        self.fc_reg = nn.Linear(128, 1)
        self.fc_class = nn.Linear(128, num_classes)
        self.type = "m"

    def forward(self, x):
        features = self.backbone(x)
        age = self.fc_reg(features)
        age_class = self.fc_class(features)
        return age, age_class