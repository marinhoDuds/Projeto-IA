import torch
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from src.graphs_gen import plot_confusion_matrix, plot_scatter, plot_error_histogram, plot_error_boxplot, plot_learning_curve

def processar_resultados_csv(diretorio_raiz: str, diretorio_saida: str):
    root_path = Path(diretorio_raiz)
    output_path = Path(diretorio_saida)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    dicionario_faixas = {0: '0-5', 1: '6-12', 2: '13-17', 3: '18-29', 4: '30-59', 5: '60+'}
    faixas_etarias_texto = ['0-5', '6-12', '13-17', '18-29', '30-59', '60+']

    arquivos_encontrados = list(root_path.rglob('*.csv'))
    print(f"Encontrados {len(arquivos_encontrados)} arquivos CSV. Iniciando processamento...\n")

    for arquivo_csv in arquivos_encontrados:
        print(f"Gerando gráficos para: {arquivo_csv}")
        
        df = pd.read_csv(arquivo_csv).dropna(subset=['real', 'pred'])

        df['real'] = df['real'].astype(str).str.strip('[]').astype(float)
        df['pred'] = df['pred'].astype(str).str.strip('[]').astype(float)
        
        caminho_relativo = arquivo_csv.relative_to(root_path)
        nome_base = str(caminho_relativo.with_suffix('')).replace('/', '_').replace('\\', '_')
        
        nome_experimento = caminho_relativo.parent.parent.name
        
        if 'train0' in arquivo_csv.name:
            df['real'] = df['real'].astype(int)
            df['pred'] = df['pred'].astype(int)

            df['real'] = df['real'].map(dicionario_faixas)
            df['pred'] = df['pred'].map(dicionario_faixas)
            
            df = df.dropna(subset=['real', 'pred'])
            
            nome_arquivo = f"matriz_confusao_{nome_base}.png"
            plot_confusion_matrix(
                y_true=df['real'].tolist(), 
                y_pred=df['pred'].tolist(), 
                classes=faixas_etarias_texto, 
                title=f"Matriz confusão - Modelo de classificação ({nome_experimento})", 
                filename=nome_arquivo, 
                output_dir=str(output_path)
            )
                                  
        elif 'train1' in arquivo_csv.name:
            y_true = df['real'].tolist()
            y_pred = df['pred'].tolist()
            main_color = 'green'
            
            plot_scatter(
                y_true, y_pred, 
                title=f"Dispersão Regressão ({nome_experimento})", 
                filename=f"scatter_{nome_base}.png", 
                main_color=main_color,
                output_dir=str(output_path)
            )
                         
            plot_error_histogram(
                y_true, y_pred, 
                title=f"Histograma de erros - Modelo de regressão ({nome_experimento})", 
                filename=f"hist_{nome_base}.png", 
                main_color=main_color,
                output_dir=str(output_path)
            )

            limites_idades = [-1, 5, 12, 17, 29, 59, 150]

            df['classe_real'] = pd.cut(df['real'], bins=limites_idades, labels=faixas_etarias_texto)
            
            y_true_classes = df['classe_real'].tolist()
            
            plot_error_boxplot(
                y_true_classification=y_true_classes, 
                y_true_regression=y_true, 
                y_pred_regression=y_pred, 
                classes=faixas_etarias_texto, 
                title=f"Erros por faixa etária - Modelo de regressão ({nome_experimento})", 
                filename=f"boxplot_{nome_base}.png", 
                main_color=main_color,
                output_dir=str(output_path)
            )
                                 
        elif 'train2' in arquivo_csv.name:
            y_true = df['real'].tolist()
            y_pred = df['pred'].tolist()
            main_color = 'purple'
            
            plot_scatter(
                y_true, y_pred, 
                title=f"Dispersão Misto ({nome_experimento})", 
                filename=f"scatter_{nome_base}.png", 
                main_color=main_color,
                output_dir=str(output_path)
            )
            
            plot_error_histogram(
                y_true, y_pred, 
                title=f"Histograma de erros - Modelo Misto ({nome_experimento})", 
                filename=f"hist_{nome_base}.png", 
                main_color=main_color,
                output_dir=str(output_path)
            )

            limites_idades = [-1, 5, 12, 17, 29, 59, 150]

            df['classe_real'] = pd.cut(df['real'], bins=limites_idades, labels=faixas_etarias_texto)
            
            y_true_classes = df['classe_real'].tolist()
            
            plot_error_boxplot(
                y_true_classification=y_true_classes, 
                y_true_regression=y_true, 
                y_pred_regression=y_pred, 
                classes=faixas_etarias_texto, 
                title=f"Erros por faixa etária - Modelo Misto ({nome_experimento})", 
                filename=f"boxplot_{nome_base}.png", 
                main_color=main_color,
                output_dir=str(output_path)
            )

    print("\nProcessamento concluído com sucesso!")

def processar_modelos_pth(diretorio_raiz: str, diretorio_saida: str):
    root_path = Path(diretorio_raiz)
    output_path = Path(diretorio_saida)
    
    output_path.mkdir(parents=True, exist_ok=True)

    arquivos_pth = list(root_path.rglob('*.pth'))
    print(f"Encontrados {len(arquivos_pth)} arquivos .pth. Iniciando...\n")

    for arquivo_pth in arquivos_pth:
        print(f"Lendo: {arquivo_pth}")
        
        nome_experimento = arquivo_pth.parent.name
        
        caminho_relativo = arquivo_pth.relative_to(root_path)
        nome_base = str(caminho_relativo.with_suffix('')).replace('/', '_').replace('\\', '_')
        
        try:
            checkpoint = torch.load(arquivo_pth, map_location='cpu', weights_only=False) 
        except Exception as e:
            print(f"  -> Erro ao carregar {arquivo_pth.name}: {e}")
            continue

        if not isinstance(checkpoint, dict):
            print(f"  -> Aviso: {arquivo_pth.name} contém apenas os pesos (não é um dicionário). Pulando gráfico.")
            continue

        train_losses = checkpoint.get('train_losses', [])
        val_losses = checkpoint.get('val_losses', None)

        if len(train_losses) == 0:
            print(f"  -> Aviso: Histórico de 'train_loss' não encontrado em {arquivo_pth.name}.")
            continue

        nome_arquivo = arquivo_pth.name
        if nome_arquivo.endswith('-c.pth'):
            tipo_modelo = "Classificação"
        elif nome_arquivo.endswith('-r.pth'):
            tipo_modelo = "Regressão"
        elif nome_arquivo.endswith('-m.pth'):
            tipo_modelo = "Misto"
        else:
            tipo_modelo = "Desconhecido"

        plot_learning_curve(
            train_scores=train_losses,
            val_scores=val_losses,
            metric_name='Loss',
            title=f"Curva de Loss - {tipo_modelo} ({nome_experimento})",
            filename=f"loss_{nome_base}.png",
            train_color='red',
            val_color='darkred',
            output_dir=str(output_path)
        )

    print("\nProcessamento de métricas de treinamento concluído!")

def calcular_metricas_finais(diretorio_raiz: str):
    root_path = Path(diretorio_raiz)
    arquivos_encontrados = list(root_path.rglob('*.csv'))
    
    faixas_etarias_texto = ['0-5', '6-12', '13-17', '18-29', '30-59', '60+']
    limites_idades = [-1, 5, 12, 17, 29, 59, 150]
    
    print("="*50)
    print("RELATÓRIO DE DESEMPENHO DOS MODELOS")
    print("="*50)

    for arquivo_csv in arquivos_encontrados:
        df = pd.read_csv(arquivo_csv).dropna(subset=['real', 'pred'])
        
        df['real'] = df['real'].astype(str).str.strip('[]').astype(float)
        df['pred'] = df['pred'].astype(str).str.strip('[]').astype(float)
        
        nome_experimento = arquivo_csv.parent.parent.name
        
        if 'train0' in arquivo_csv.name:
            df['real'] = df['real'].astype(int)
            df['pred'] = df['pred'].astype(int)
            
            acuracia = accuracy_score(df['real'], df['pred'])
            
            print(f"\n{nome_experimento.upper()} | MODELO DE CLASSIFICAÇÃO")
            print(f"   -> Acurácia Global: {acuracia * 100:.2f}%")
            print(f"   -> (Modelos de classificação pura não possuem MAE/RMSE)")

        elif 'train1' in arquivo_csv.name:
            mae = mean_absolute_error(df['real'], df['pred'])
            rmse = np.sqrt(mean_squared_error(df['real'], df['pred']))
            
            df['classe_real'] = pd.cut(df['real'], bins=limites_idades, labels=faixas_etarias_texto)
            df['classe_pred'] = pd.cut(df['pred'], bins=limites_idades, labels=faixas_etarias_texto)
            
            df = df.dropna(subset=['classe_real', 'classe_pred'])
            
            acuracia = accuracy_score(df['classe_real'], df['classe_pred'])
            
            print(f"\n{nome_experimento.upper()} | MODELO DE REGRESSÃO")
            print(f"   -> Acurácia (convertido p/ faixas): {acuracia * 100:.2f}%")
            print(f"   -> MAE (Erro Médio Absoluto): {mae:.2f} anos")
            print(f"   -> RMSE: {rmse:.2f} anos")

        elif 'train2' in arquivo_csv.name:
            mae = mean_absolute_error(df['real'], df['pred'])
            rmse = np.sqrt(mean_squared_error(df['real'], df['pred']))
            
            df['classe_real'] = pd.cut(df['real'], bins=limites_idades, labels=faixas_etarias_texto)
            df['classe_pred'] = pd.cut(df['pred'], bins=limites_idades, labels=faixas_etarias_texto)
            
            df = df.dropna(subset=['classe_real', 'classe_pred'])
            acuracia = accuracy_score(df['classe_real'], df['classe_pred'])
            
            print(f"\n{nome_experimento.upper()} | MODELO MISTO")
            print(f"   -> Acurácia (convertido p/ faixas): {acuracia * 100:.2f}%")
            print(f"   -> MAE (Erro Médio Absoluto): {mae:.2f} anos")
            print(f"   -> RMSE: {rmse:.2f} anos")