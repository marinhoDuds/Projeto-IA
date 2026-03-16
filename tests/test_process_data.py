import unittest
from src.process_data import split_dataset, age_class_name

class TestDatasetSplit(unittest.TestCase):
    def setUp(self):
        """
        Prepara os dados sintéticos antes de cada teste.
        Gera 1000 amostras para facilitar a verificação de porcentagens exatas.
        """
        self.image_paths = [f"{i}_foto.jpg" for i in range(1000)]
        self.ages = []
        
        # Distribui as idades para garantir que todas as 6 classes existam e
        # a estratificação não falhe por ausência de dados.
        for i in range(1000):
            if i < 100: self.ages.append(3)    # Classe 0 (10%)
            elif i < 200: self.ages.append(10) # Classe 1 (10%)
            elif i < 400: self.ages.append(15) # Classe 2 (20%)
            elif i < 600: self.ages.append(25) # Classe 3 (20%)
            elif i < 800: self.ages.append(45) # Classe 4 (20%)
            else: self.ages.append(65)         # Classe 5 (20%)

    def test_proporcao_datasets(self):
        """
        Verifica se os datasets estão na proporção estrita de 75%, 15% e 10%.
        1000 amostras devem resultar em: Train=750, Val=150, Test=100.
        """
        train_p, _, val_p, _, test_p, _ = split_dataset(self.image_paths, self.ages)
        
        self.assertEqual(len(train_p), 750, "O conjunto de treino não possui 75% dos dados.")
        self.assertEqual(len(val_p), 150, "O conjunto de validação não possui 15% dos dados.")
        self.assertEqual(len(test_p), 100, "O conjunto de teste não possui 10% dos dados.")

    def test_contaminacao_dados(self):
        """
        Verifica se há intersecção de arquivos entre os conjuntos.
        O conjunto de teste deve ser estritamente isolado.
        """
        train_p, _, val_p, _, test_p, _ = split_dataset(self.image_paths, self.ages)
        
        train_set = set(train_p)
        val_set = set(val_p)
        test_set = set(test_p)
        
        # O método isdisjoint retorna True se os conjuntos não tiverem nenhum elemento em comum
        self.assertTrue(train_set.isdisjoint(test_set), "Falha: Dados de treino vazaram para o teste.")
        self.assertTrue(train_set.isdisjoint(val_set), "Falha: Dados de treino vazaram para a validação.")
        self.assertTrue(val_set.isdisjoint(test_set), "Falha: Dados de validação vazaram para o teste.")

    def test_distribuicao_classes_estratificada(self):
        """
        Verifica se a distribuição de classes (stratify) foi mantida.
        Se a Classe 5 representa 20% do total, ela deve representar ~20% no teste.
        """
        _, _, _, _, test_p, test_ages = split_dataset(self.image_paths, self.ages)
        
        # Conta a quantidade de itens na Classe 5 (idade 65) no conjunto de teste
        classe_5_count = sum(1 for age in test_ages if age_class_name(age) == 5)
        
        # Como o teste tem 100 itens, e a Classe 5 é 20% do total original,
        # esperamos exatamente 20 itens da Classe 5 no conjunto de teste.
        self.assertEqual(classe_5_count, 20, "A estratificação falhou ao distribuir as classes.")

if __name__ == '__main__':
    unittest.main()