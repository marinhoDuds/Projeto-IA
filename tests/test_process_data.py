import sys
import os
import unittest

from src.process_data import split_dataset, age_class

class TestDatasetSplit(unittest.TestCase):

    def setUp(self):
        """
        Gera 1000 exemplos fictícios com idades fixas.
        """
        self.image_paths = [f"{i}_foto.jpg" for i in range(1000)]
        self.ages = []
        
        for i in range(1000):
            if i < 100: self.ages.append(3)    
            elif i < 200: self.ages.append(10) 
            elif i < 400: self.ages.append(15) 
            elif i < 600: self.ages.append(25) 
            elif i < 800: self.ages.append(45) 
            else: self.ages.append(65)         

    def test_proporcao_datasets(self):
        """
        Verifica se a divisão resulta na proporção correta.
        """
        train_p, _, val_p, _, test_p, _ = split_dataset(self.image_paths, self.ages)
        
        self.assertEqual(len(test_p), 100)
        self.assertIn(len(val_p), [149, 150])
        self.assertIn(len(train_p), [750, 751])

    def test_contaminacao_dados(self):
        """
        Garante que não há dados repetidos entre os conjuntos.
        """
        train_p, _, val_p, _, test_p, _ = split_dataset(self.image_paths, self.ages)
        
        train_set = set(train_p)
        val_set = set(val_p)
        test_set = set(test_p)
        
        self.assertTrue(train_set.isdisjoint(test_set))
        self.assertTrue(train_set.isdisjoint(val_set))
        self.assertTrue(val_set.isdisjoint(test_set))

    def test_distribuicao_classes(self):
        """
       Verifica se a divisão manteve a proporção de cada classe de idade corretas.
        """
        _, _, _, _, _, test_ages = split_dataset(self.image_paths, self.ages)
        
        classes_no_teste = [age_class(age) for age in test_ages]
        
        self.assertEqual(classes_no_teste.count(0), 10)
        self.assertEqual(classes_no_teste.count(1), 10)
        self.assertEqual(classes_no_teste.count(2), 20)
        self.assertEqual(classes_no_teste.count(3), 20)
        self.assertEqual(classes_no_teste.count(4), 20)
        self.assertEqual(classes_no_teste.count(5), 20)

if __name__ == '__main__':
    unittest.main()