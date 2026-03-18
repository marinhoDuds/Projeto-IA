import unittest
import torch
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from src.dataset import AgeDataset 

class TestAgeDataset(unittest.TestCase):
    def setUp(self):
        self.test_dir = "."
        self.img_name = "test_img_01.jpg"
        self.img_path = os.path.join(self.test_dir, self.img_name)
        Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255)).save(self.img_path)
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
    def tearDown(self):
        if os.path.exists(self.img_path):
            os.remove(self.img_path)

    def test_getitem_classification(self):
        """Verifica se o dataset retorna classe quando classification está ativo"""
        fake_class_func = lambda x: 3 
        
        ds = AgeDataset(self.test_dir, [self.img_name], [25], self.transform, fake_class_func)
        img, label = ds[0]
        
        self.assertIsInstance(label, tuple)
        self.assertEqual(label[1].item(), 3) 
        self.assertEqual(img.shape, (3, 128, 128))

    def test_getitem_regression(self):
        """Verifica se a idade é retornada corretamente mesmo em modo regressão"""
        dummy_func = lambda x: 0 
        
        ds = AgeDataset(self.test_dir, [self.img_name], [25], self.transform, dummy_func)
        img, label = ds[0]
        
        self.assertEqual(label[0].item(), 25)
        self.assertIsInstance(label[0], torch.Tensor)