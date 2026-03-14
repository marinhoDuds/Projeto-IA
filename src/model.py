import torch
import torch.nn as nn
import torch.nn.functional as F

class AgeClassCNN(nn.Module):
    """
    Representa um modelo de rede neural para classificação de imagens em 5 classes, sendo elas: Criança, Adolescente, Jovem, Adulto, Idoso.
    """
    def __init__(self, num_class=5):
        """
        Inicializa a arquitetura do modelo. 

        Detalhes da arquitetura:
        //TODO: Ao chegarmos em uma arquitetura final, devemos descreve-la aqui. (se for necessário)
        """
        super(AgeClassCNN, self).__init__()
         
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)

        self.fcl1 = nn.Linear(64 * 16 * 16, 128)
        self.fcl2 = nn.Linear(128, num_class)

    def forward(self, x):
        """
        Realiza a inferência do modelo dado uma entrada.
        
        return:
            //TODO:
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fcl1(x))
        x = self.fcl2(x)

        return x