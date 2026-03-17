import unittest
import torch
from models.cnn_classification import AgeClassificationModel
from models.cnn_regression import AgeRegressionModel
from models.cnn_multi import AgeMultiModel

# --- TESTES PARA O MODELO DE CLASSIFICAÇÃO ---
class TestClassificationModel(unittest.TestCase):
    def setUp(self):
        self.model = AgeClassificationModel()
        self.fake_img = torch.randn(1, 1, 128, 128)

    def test_type(self):
        """Verifica se a etiqueta é 'c'"""
        self.assertEqual(self.model.type, "c")

    def test_output_shape(self):
        """Verifica se saem 6 classes da rede"""
        self.assertEqual(self.model.fc_class.out_features, 6)
        output = self.model(self.fake_img)
        self.assertEqual(output.shape, (1, 6))

# --- TESTES PARA O MODELO DE REGRESSÃO ---
class TestRegressionModel(unittest.TestCase):
    def setUp(self):
        self.model = AgeRegressionModel()
        self.fake_img = torch.randn(1, 1, 128, 128)

    def test_type(self):
        """Verifica se a etiqueta é 'r'"""
        self.assertEqual(self.model.type, "r")

    def test_output_shape(self):
        """Verifica se sai 1 único número (idade)"""
        self.assertEqual(self.model.fc_reg.out_features, 1)
        output = self.model(self.fake_img)
        self.assertEqual(output.shape, (1, 1))

# --- TESTES PARA O MODELO MULTI-TASK ---
class TestMultiModel(unittest.TestCase):
    def setUp(self):
        self.model = AgeMultiModel()
        self.fake_img = torch.randn(1, 1, 128, 128)

    def test_type(self):
        """Verifica se a etiqueta é 'm'"""
        self.assertEqual(self.model.type, "m")

    def test_output_shapes(self):
        """Verifica se as duas saídas (1 e 6) existem e estão corretas"""
        self.assertEqual(self.model.fc_reg.out_features, 1)
        self.assertEqual(self.model.fc_class.out_features, 6)
        
        age, age_class = self.model(self.fake_img)
        self.assertEqual(age.shape, (1, 1), "Erro no shape da idade (Regressão)")
        self.assertEqual(age_class.shape, (1, 6), "Erro no shape da classe (Classificação)")

if __name__ == '__main__':
    unittest.main()
'''
import unittest
import torch
from models.cnn_classification import AgeClassificationModel
from models.cnn_regression import AgeRegressionModel
from models.cnn_multi import AgeMultiModel

class TestModelLogic(unittest.TestCase):

    def setUp(self):
        """Este método roda antes de cada teste para preparar os modelos"""
        self.model_c = AgeClassificationModel()
        self.model_r = AgeRegressionModel()
        self.model_m = AgeMultiModel()

    def test_classification_type(self):
        """Verificar se o uso ou não de classificação está funcionando (campo type)"""
        self.assertEqual(self.model_c.type, "c")
        self.assertEqual(self.model_r.type, "r")
        self.assertEqual(self.model_m.type, "m")

    def test_output_layers(self):
        """Verifica se o número de neurônios na saída está correto"""
        self.assertEqual(self.model_c.fc_class.out_features, 6)
        self.assertEqual(self.model_r.fc_reg.out_features, 1)
        self.assertEqual(self.model_m.fc_class.out_features, 6)
        self.assertEqual(self.model_m.fc_reg.out_features, 1)

    def test_forward_pass_shape(self):
        """Verifica se o modelo aceita imagem 128x128 e retorna o formato certo"""
        fake_img = torch.randn(1, 1, 128, 128)
        output = self.model_c(fake_img)
        self.assertEqual(output.shape, (1, 6))
        self.assertTrue(torch.is_tensor(output))
        output = self.model_r(fake_img)
        self.assertEqual(output.shape, (1, 1))
        age, age_class = self.model_m(fake_img) 
        self.assertEqual(age.shape, (1, 1))       
        self.assertEqual(age_class.shape, (1, 6)) 

if __name__ == '__main__':
    unittest.main()
'''