import unittest
import torch
import torch.nn as nn

from src.train import train_epoch, eval_epoch

class MockModel(nn.Module):
    def __init__(self, model_type="r"):
        super().__init__()
        self.layer = nn.Linear(10, 2)
        self.type = model_type

    def forward(self, x):
        return self.layer(x)

class ConstantLoss(nn.Module):
    """
    Função de perda que retorna sempre um valor constante. 
    """
    def forward(self, outputs, labels):
        return torch.tensor(5.0, requires_grad=True)

class TestEpochLossCalculations(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = MockModel(model_type="r").to(self.device)
        self.criterion = ConstantLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)

        self.mock_loader = []
        for _ in range(4):
            images = torch.randn(2, 10)
            age = torch.randn(2)
            age_class = torch.randint(0, 5, (2,))
            labels = (age, age_class)
            
            self.mock_loader.append((images, labels))

    def test_train_loss_calculation(self):
        """
        Verifica se a função de treinamento calcula corretamente
        a média da loss ao longo dos batches.
        """
        loss_calculada = train_epoch(
            self.model, 
            self.mock_loader, 
            self.criterion, 
            self.optimizer, 
            self.device
        )
        
        self.assertEqual(loss_calculada, 5.0)

    def test_val_loss_calculation(self):
        """
        Verifica se a função de validação calcula corretamente
        a média da loss ao longo dos batches.
        """
        loss_calculada = eval_epoch(
            self.model, 
            self.mock_loader, 
            self.criterion, 
            self.device
        )
        
        self.assertEqual(loss_calculada, 5.0)

if __name__ == '__main__':
    unittest.main()