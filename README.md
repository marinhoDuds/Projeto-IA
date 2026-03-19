# Projeto-IA - AgeCNN: Estimação de Idade utilizando CNNs

O projeto realiza uma análise comparativa do desempenho de duas arquiteturas de Redes Neurais Convolucionais (CNNs) aplicadas à tarefa de estimação de idade a partir de imagens. As arquiteturas foram projetadas e implementadas pelo próprio grupo, explorando diferentes configurações estruturais e estratégias de treinamento. O estudo investiga a capacidade de generalização dos modelos, avaliando métricas de desempenho e analisando como variações no design das redes influenciam a precisão das predições.

## Estrutura do Projeto
```
Projeto-IA/
├── models/
│   ├── base_cnn.py            # Backbone CNN base
│   ├── advanced_cnn.py        # Backbone CNN avançada
│   ├── cnn_classification.py  # Modelo de classificação de idade
│   ├── cnn_regression.py      # Modelo de regressão de idade
│   └── cnn_multi.py           # Modelo multi-task
├── notebooks/                 # Notebooks de exploração e análise
├── results/                   # Predições e plots gerados
├── src/
|   ├── dataset.py             # Classe pra definir os dados
│   ├── train.py               # Pipeline de treinamento
│   ├── inference.py           # Pipeline de inferência
│   ├── process_data.py        # Pré-processamento e split dos dados
│   ├── multi_loss.py          # Loss para o modelo multi
│   └── graphs_gen.py          # Funções utilizadas para gerar os gráficos
├── tests/                     # Testes unitários
├── run.py                     # Ponto de entrada principal (CLI)
├── requirements.txt           # Dependências do projeto
└── README.md
```

## Como Utilizar
Esta seção contém o passo a passo de como rodar o projeto.

### 1. Clonar o repositório

```bash
git clone https://github.com/marinhoDuds/Projeto-IA.git
cd Projeto-IA
```

### 2. Instalar as dependências
``` bash
pip install -r requirements.txt
```

### 3. Preparar os Dados
O dataset pode ser utilizado de duas formas:

1. Caso o dataset contenha apenas um conjunto de imagens, o pipeline realizará automaticamente a divisão (split) dos dados;
2. Caso o dataset esteja organizado nas subpastas [`train`, `val`, `test`], essa estrutura será utilizada diretamente.

Se você possuir apenas um conjunto de imagens, é possível gerar automaticamente a subdivisão do dataset com o seguinte comando:

```bash
python run.py data -p /caminho/para/o/dataset
```

### 4. Treinar o Modelo
Para treinar um modelo primeiro é necessário configurar os seguintes parametros: 

* **IMG_SIZE** - Define o tamanho para o qual as imagens serão redimensionadas antes de serem utilizadas no treinamento;
* **BATCH_SIZE** - Número de amostras processadas em cada iteração durante o treinamento;
* **LR (Learning Rate)** - Taxa de aprendizado do modelo;
* **EPOCHS** - Quantidade de épocas;
* **PATIENCE** - Número de épocas sem melhora na métrica de validação antes de acionar o early stopping;
* **backbone** - Modelo que será usado como backbone, estão disponíveis a [CNN Basica](models/base_cnn.py) e a [CNN Avançada](models/advanced_cnn.py)

Esse setup deve ser feito em [run.py](/run.py)

Após isso utilize o comando:

```bash
python run.py train -p /caminho/para/o/dataset -m <tipo_do_modelo>
```
Os seguintes tipos de modelo (`-m`) podem ser utilizados:
- `c` - [Modelo Classificação](models/cnn_classification.py)
- `r` - [Modelo Regressão](models/cnn_regression.py)
- `m` - [Modelo Multi-task](models/cnn_regression.py)


### 5. Executar Inferência
Para realizar a predição de idade em uma imagem utilize:

```bash
python run.py inference -p /caminho/para/imagem.jpg -m /caminho/para/modelo_treinado.pth
```


## Resultados e Datasets
Os resultados do treinamento e gráficos plotados são armazenados na pasta `results/`.
Os modelos treinados e dataset utilizados podem ser encontrados no [Drive do Projeto](https://drive.google.com/drive/folders/1nalnIahALxO_Te65fgcW3PT7Hg7TClJu?usp=sharing)
O notebook apresentado em sala de aula pode ser encontrado em [presentation.ipynb](notebooks/presentation.ipynb)