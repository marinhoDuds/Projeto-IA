from torch import nn

class AgeClassificationModel(nn.Module):
    def __init__(self, backbone, num_classes=6):
        super().__init__()
        self.backbone = backbone
        self.fc_class = nn.Linear(128, num_classes)
        self.type = "c"

    def forward(self, x):
        features = self.backbone(x)
        age_class = self.fc_class(features)
        return age_class