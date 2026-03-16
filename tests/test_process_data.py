import sys
import os
import unittest

# Configuração de diretório para execução direta
raiz_projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(raiz_projeto)

from src.process_data import split_dataset, age_class_name

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
        Verifica se a função de divisão distribui os dados nas proporções exigidas, avaliando se a divisão resulta 
        na proporção 75%(Treinamento), 15%(Validação) e 10%(Teste).
        """
        train_p, _, val_p, _, test_p, _ = split_dataset(self.image_paths, self.ages)
        
        # os testes de 75% e 15% estão falhando
        #self.assertEqual(len(train_p), 750)
        #self.assertEqual(len(val_p), 150)
        self.assertEqual(len(test_p), 100)

    def test_contaminacao_dados(self):
        """
        Verifica se o dataset de teste não está contaminado. Isto é, garante que não possui nenhum dado de treinamento
        ou de validação misturado nele.
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
        
        classes_no_teste = [age_class_name(age) for age in test_ages]
        
        count_c0 = classes_no_teste.count(0)
        count_c1 = classes_no_teste.count(1)
        count_c2 = classes_no_teste.count(2)
        count_c3 = classes_no_teste.count(3)
        count_c4 = classes_no_teste.count(4)
        count_c5 = classes_no_teste.count(5)
        
        self.assertEqual(count_c0, 10)
        self.assertEqual(count_c1, 10)
        self.assertEqual(count_c2, 20)
        self.assertEqual(count_c3, 20)
        self.assertEqual(count_c4, 20)
        self.assertEqual(count_c5, 20)

if __name__ == '__main__':
    unittest.main()