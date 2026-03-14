import os
import shutil
import re
from pathlib import Path 
import kagglehub

def organizar_fgnet_em_classes(data_dir_origin, data_dir_target):
    """
    Lê o dataset do FGNET original e cria uma cópia para uma estrutura de pastas separada por classes de idade.

    Args:
        data_dir_origin: caminho para o diretório do dataset original do FGNET.
        data_dir_target: caminho para o diretório onde ficará a cópia do dataset estruturado em classes.
    """
    
    classes = ["Criança", "Adolecente", "Jovem", "Adulto", "Idoso"]

    for nome_classe in classes:
        pasta_classe = Path(data_dir_target) / nome_classe
        pasta_classe.mkdir(parents=True, exist_ok=True)

    arquivos = list(Path(data_dir_origin).glob('**/*.*'))

    imagens_processadas = 0
    imagens_ignoradas = 0

    for caminho_arquivo in arquivos:
        nome_arquivo = caminho_arquivo.name

        if not nome_arquivo.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        match = re.search(r'A(\d+)', nome_arquivo)

        if match:
            idade = int(match.group(1))
            classe_alvo = classificar_idade(idade)
            caminho_final = Path(data_dir_target) / classe_alvo / nome_arquivo
            shutil.copy2(caminho_arquivo, caminho_final)
            imagens_processadas += 1
        else:
            imagens_ignoradas += 1

    print(f"\nConcluído! {imagens_processadas} imagens organizadas com sucesso.")
    if imagens_ignoradas > 0:
        print(f"Aviso: {imagens_ignoradas} arquivos foram ignorados por não baterem com o padrão.")

def classificar_idade(idade):
    if idade <= 12: return "Criança"
    elif idade <= 17: return "Adolecente"
    elif idade <= 29: return "Jovem"
    elif idade <= 59: return "Adulto"
    else: return "Idoso"

if __name__ == "__main__":
    print("Verificando dataset do Kaggle...")
    caminho_download = kagglehub.dataset_download("aiolapo/fgnet-dataset")
    print(f"Dataset original localizado em: {caminho_download}")

    DATASET_DESTINO = "./dataset_idades"

    organizar_fgnet_em_classes(caminho_download, DATASET_DESTINO)