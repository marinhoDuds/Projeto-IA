import torch.nn as nn

class MultiLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs, labels):
        age_pred, age_class_pred = outputs
        age_true =  labels.float().unsqueeze(1) 
        class_true = labels
        return self.alpha*self.mse(age_pred, age_true) + (1-self.alpha)*self.ce(age_class_pred, class_true)