import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
    
def get_dataloaders(data_dir, batch_size=32, img_size=128, val_split=0.2):
    """
    Carrega as imagens dos diretórios, aplica transformações e divide em treino e validação.

    Args:
        data_dir: caminho para o diretório de dados.
        batch_size: o tamanho do lote de imagens que a rede deve treinar.
        img_size: o tamanho que a imagem deve ser redimensionada.
        val_split: a porcentagem de imagens que queremos separar para validação.

    Return:
        train_loader: o DataLoader de treinamento
        val_loader: o Dataloader de validação
        classes: retornn
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    dataset_completo = datasets.ImageFolder(root=data_dir, transform=transform)
    classes = dataset_completo.classes

    val_size = int(len(dataset_completo) * val_split)
    train_size = len(dataset_completo) - val_size

    train_dataset, val_dataset = random_split(
        dataset_completo,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(7)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, classes