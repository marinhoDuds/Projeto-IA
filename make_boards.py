import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from collections import defaultdict
import math

def make_boards(input_dir: str, output_dir: str = None):
    """
    Agrupa imagens com o mesmo prefixo (tudo exceto o último token separado por '_')
    em uma única imagem com subplots.

    Args:
        input_dir: Pasta com os gráficos gerados (ex: 'Gráficos/')
        output_dir: Pasta de saída dos boards. Padrão: mesma pasta de input.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else input_path
    output_path.mkdir(parents=True, exist_ok=True)

    # Agrupa arquivos por prefixo (tudo antes do último _token)
    groups = defaultdict(list)
    for png in sorted(input_path.glob("*.png")):
        parts = png.stem.split("_")
        prefix = "_".join(parts[:-1])
        groups[prefix].append(png)

    print(f"Encontrados {len(groups)} grupos:\n")

    for prefix, files in sorted(groups.items()):
        files = sorted(files)
        n = len(files)

        # Pula grupos com apenas 1 arquivo (nada a combinar)
        if n == 1:
            print(f"  [SKIP] '{prefix}' — apenas 1 arquivo, sem necessidade de board.")
            continue

        # Define layout de colunas/linhas
        ncols = min(n, 3)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 8, nrows * 6))

        # Normaliza axes para sempre ser lista 2D
        if nrows == 1 and ncols == 1:
            axes = [[axes]]
        elif nrows == 1:
            axes = [axes]
        elif ncols == 1:
            axes = [[ax] for ax in axes]

        for idx, filepath in enumerate(files):
            row = idx // ncols
            col = idx % ncols
            ax = axes[row][col]

            img = mpimg.imread(filepath)
            ax.imshow(img)
            ax.axis("off")

            # Subtítulo = último token do nome (ex: train1, train2, model-c, ...)
            subtitle = filepath.stem.split("_")[-1]
            ax.set_title(subtitle, fontsize=13, pad=8)

        # Esconde eixos extras se n não preenche a grade
        for idx in range(n, nrows * ncols):
            row = idx // ncols
            col = idx % ncols
            axes[row][col].set_visible(False)

        # Título geral = prefixo (substitui _ por espaço para legibilidade)
        fig.suptitle(prefix.replace("_", "  |  "), fontsize=15, fontweight="bold", y=1.01)
        plt.tight_layout()

        out_filename = output_path / f"board_{prefix}.png"
        plt.savefig(out_filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  [OK] board_{prefix}.png  ({n} gráficos, {nrows}x{ncols})")

    print("\nPronto! Todos os boards foram salvos em:", output_path.resolve())


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python make_boards.py <pasta_com_graficos> [pasta_saida]")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    make_boards(input_dir, output_dir)