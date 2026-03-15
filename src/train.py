import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from multi_loss import MultiLoss
from process_data import get_datasets

def train(model, dataset_path, device, img_size, batch_size, num_epochs, lr, patience):
    model_type = model.type
    train_dataset, val_dataset, test_dataset = get_datasets(dataset_path, img_size, model_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    if model_type == "r":
        criterion = nn.MSELoss(reduction='mean')
    elif model_type == "c":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = MultiLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        val_loss = eval_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(),"type": model.type}, "best_model.pth")
        else:
            epochs_without_improvement += 1

            print(
                f"No improvement "
                f"({epochs_without_improvement}/{patience})"
            )

            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    model.inference(test_dataset)
    model.load_state_dict(torch.load("best_model.pth"))
    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, leave=False, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

    return total_loss / len(loader)