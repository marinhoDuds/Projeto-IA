from torch import nn

class AgeRegressionModel(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc_reg = nn.Linear(128, 1)
        self.type = "r"

    def forward(self, x):
        features = self.backbone(x)
        age = self.fc_reg(features)
        return age