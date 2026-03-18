import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path

def save_plot(filename: str, output_dir: str = None):
    """
    Função auxiliar interna (privada) para salvar os gráficos.
    Centraliza a lógica de resolução de caminhos e salvamento do matplotlib.
    """
    if output_dir is None: 
        current_dir = Path(__file__).resolve().parent
        target_dir = current_dir.parent / 'graphs'
    else: 
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    full_path = target_dir / filename
    plt.savefig(full_path, dpi=300)
    plt.close()

def plot_confusion_matrix(y_true: list, y_pred: list, classes: list, title: str="Matriz de confusão", filename: str='confusion_matrix.png', main_color: str='Blues', output_dir: str=None):
    """
    Gera e salva um gráfico de Matriz de confusão.

    Args:
        y_true (lista/array): Valores reais da faixa etária.
        y_pred (lista/array): Valores previstos pelo modelo. 
        classes (lista): Nome das faixas etárias.
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        main_color (str): A principal cor do gráfico. (ex. 'Blues', 'Reds', 'Greens', 'Viridis', etc.)
        output_dir (str): Caminho para salvar o gráfico. (ex. 'Projeto/graphs/'). Salva por padrão no diretório /graphs do projeto.
    """

    graph_confusion_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(8, 6))

    sns.heatmap(graph_confusion_matrix, annot=True, cmap=main_color, fmt='d',
                xticklabels=classes, yticklabels=classes,
                cbar=True)

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Faixa etária real', fontsize=12)
    plt.xlabel('Faixa etária prevista pelo modelo', fontsize=12)
    plt.tight_layout()
    
    save_plot(filename, output_dir)

def plot_scatter(y_true: list, y_pred: list, title: str="Real vs Previsto", filename: str='scatter_plot.png', main_color: str='blue', output_dir: str=None):
    """
    Gera e salva um gráfico de dispersão comparando valores reais e previstos.

    Args:
        y_true (lista/array): Valores reais da idade.
        y_pred (lista/array): Valores previstos pelo modelo. 
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        main_color (str): A principal cor do gráfico. (ex. 'blue', 'red', 'green', etc.)
        output_dir (str): Caminho para salvar o gráfico. (ex. 'Projeto/graphs/'). Salva por padrão no diretório /graphs do projeto.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(9, 7))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color=main_color, edgecolor='white', s=60)

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Previsão esperada (y = x)')

    metrics_text = f"MAE: {mae:.2f} anos\nRMSE: {rmse:.2f} anos"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Idade real', fontsize=12)
    plt.ylabel('Idade prevista pelo modelo', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    save_plot(filename, output_dir)

def plot_error_histogram(y_true: list, y_pred: list, title: str="Distribuição de erros (Resíduos)", filename: str='histogram_erros.png', main_color: str='blue', output_dir: str=None):
    """
    Gera e salva um histograma mostrando a frequência dos erros do modelo.
    Erro = idade prevista - idade real.

    Args:
        y_true (lista/array): Valores reais da idade.
        y_pred (lista/array): Valores previstos pelo modelo. 
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        main_color (str): A principal cor do gráfico. (ex. 'blue', 'red', 'green', etc.)
        output_dir (str): Caminho para salvar o gráfico. (ex. 'Projeto/graphs/'). Salva por padrão no diretório /graphs do projeto.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    errors = y_pred - y_true

    plt.figure(figsize=(9, 6))

    sns.histplot(errors, bins=20, kde=True, color=main_color, edgecolor='white')

    plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Erro zero (acertou)')

    average_errors = np.mean(errors)
    standard_deviation = np.std(errors)

    metrics_text = f"Viés (Média de erro): {average_errors:.2f} anos\nDesvio padrão: {standard_deviation:.2f} anos"

    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Erro em anos (Positivo = Superestimou | Negativo = Subestimou)', fontsize=12)
    plt.ylabel('Frequência', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()

    save_plot(filename, output_dir)

def plot_error_boxplot(y_true_classification: list, y_true_regression: list, y_pred_regression: list, classes: list, title: str="Erro absoluto por faixa etária", filename: str='boxplot.png', main_color: str='blue', output_dir: str=None):
    """
    Gera e salva um boxplot mostrando a distribuição de erro absoluto para cada classe (faixa etária).

    Args:
        y_true_classification (lista/array): A categoria (faixa etária) de cada pessoa (ex: 'Adulto').
        y_true_regression (lista/array): A idade exata da real em anos.
        y_pred_regression (lista/array): A idade exata prevista pelo modelo. 
        classes (lista/array): Nome das faixas etárias.
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        main_color (str): A principal cor do gráfico. (ex. 'blue', 'red', 'green', etc.)
        output_dir (str): Caminho para salvar o gráfico. (ex. 'Projeto/graphs/'). Salva por padrão no diretório /graphs do projeto.
    """

    y_true_regression = np.array(y_true_regression)
    y_pred_regression = np.array(y_pred_regression)

    absotule_errors = np.abs(y_pred_regression - y_true_regression) 

    plt.figure(figsize=(10, 6))

    sns.boxplot(x=y_true_classification, y=absotule_errors, order=classes, color=main_color, showfliers=False)

    sns.stripplot(x=y_true_classification, y=absotule_errors, order=classes, color='black', alpha=0.1, jitter=True, size=4)

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Faixa etária real', fontsize=12)
    plt.ylabel('Erro absoluto (anos de diferença)', fontsize=12)
    plt.grid(axis='y', linestyle=':', alpha=0.7)
    plt.tight_layout()

    save_plot(filename, output_dir)

def plot_learning_curve(train_scores: list, val_scores: list = None, metric_name: str = 'Loss', title: str = 'Curva de Aprendizado', filename: str = 'learning_curve.png', train_color: str = 'blue', val_color: str = 'orange', output_dir: str = None):
    """
    Gera e salva um gráfico de linha mostrando a evolução de uma métrica (ex. loss, acurácia, validação, etc) ao longo das épocas.

    Args:
        train_scores (lista/array): Valores da métrica (ex: loss ou acurácia) nos dados de treino.
        val_scores (lista/array, opcional): Valores da métrica nos dados de validação.
        metric_name (str): Nome da métrica para os rótulos (ex: 'Loss', 'Acurácia').
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        train_color (str): Cor da linha de treino.
        val_color (str): Cor da linha de validação.
        output_dir (str): Caminho para salvar o gráfico. Salva por padrão no diretório /graphs do projeto.
    """
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(9, 6))

    sns.lineplot(x=epochs, y=train_scores, label=f'{metric_name} (Treino)', color=train_color, linewidth=2)

    if val_scores is not None:
        if len(train_scores) != len(val_scores):
            print(f"Aviso: O tamanho de train_scores ({len(train_scores)}) e val_scores ({len(val_scores)}) é diferente!")
        else:
            sns.lineplot(x=epochs, y=val_scores, label=f'{metric_name} (Validação)', color=val_color, linewidth=2, linestyle='--')

    plt.title(title, fontsize=14, pad=15)
    plt.xlabel('Épocas (Epochs)', fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    
    from matplotlib.ticker import MaxNLocator
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.legend(loc='best')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    save_plot(filename, output_dir)