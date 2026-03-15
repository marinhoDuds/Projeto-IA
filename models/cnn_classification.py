from torch import nn
from .base_cnn import BaseCNN

class AgeClassificationModel(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = BaseCNN()
        self.fc_class = nn.Linear(128, num_classes)
        self.type = "c"

    def forward(self, x):
        features = self.backbone(x)
        age_class = self.fc_class(features)
        return age_class