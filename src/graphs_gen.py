import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path

def plot_confusion_matrix(y_true: list, y_pred: list, classes: list, title: str="Matriz de confusão", filename: str='confusion_matrix.png', main_color: str='Blues', output_dir: str=None):
    """
    Gera e salva um gráfico de Matriz de confusão.

    Args:
        y_true (lista/array): Valores reais da faixa etária.
        y_pred (lista/array): Valores previstos pelo modelo. 
        classes (lista): Nome das faixas etárias.
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        output_dir (str): Caminho para salvar o gráfico. (ex. 'Projeto/graphs/'). Salva por padrão no diretório /graphs do projeto.
        main_color (str): A principal cor do gráfico. (ex. 'Blues', 'Reds', 'Greens', 'Viridis', etc.)
    """

    graph_confusion_matrix = confusion_matrix(y_true, y_pred, labels=classes)

    plt.figure(figsize=(8, 6))

    sns.heatmap(graph_confusion_matrix, annot=True, cmap=main_color,
                xticklabels=classes, yticklabels=classes,
                cbar=False)

    plt.title(title, fontsize=14, pad=15)
    plt.ylabel('Faixa etária real', fontsize=12)
    plt.xlabel('Faixa etária prevista pelo modelo', fontsize=12)

    plt.tight_layout()

    if output_dir is None: 
        current_dir = Path(__file__).resolve().parent
        target_dir = current_dir.parent / 'graphs'
    else: 
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    full_path = target_dir / filename
    plt.savefig(full_path, dpi=300)
    plt.close()

def plot_scatter(y_true: list, y_pred: list, title: str="Real vs Previsto", filename: str='scatter_plot.png', main_color: str='blue', output_dir: str=None):
    """
    Gera e salva um gráfico de dispersão comparando valores reais e previstos.

    Args:
        y_true (lista/array): Valores reais da idade.
        y_pred (lista/array): Valores previstos pelo modelo. 
        title (str): Título do gráfico.
        filename (str): Nome do arquivo onde o gráfico será salvo.
        output_dir (str): Caminho para salvar o gráfico. (ex. 'Projeto/graphs/'). Salva por padrão no diretório /graphs do projeto.
        main_color (str): A principal cor do gráfico. (ex. 'blue', 'red', 'green', etc.)
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    plt.figure(figsize=(9, 7))

    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color=main_color, edgecolor='w', s=60)

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

    if output_dir is None: 
        current_dir = Path(__file__).resolve().parent
        target_dir = current_dir.parent / 'graphs'
    else: 
        target_dir = Path(output_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    full_path = target_dir / filename
    plt.savefig(full_path, dpi=300)
    plt.close()