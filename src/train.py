import os
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.multi_loss import MultiLoss
from src.process_data import get_datasets

def train(model, dataset_path, device, img_size, batch_size, num_epochs, lr, patience):
    model_type = model.type
    train_dataset, val_dataset, test_dataset = get_datasets(dataset_path, img_size, model_type)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if model_type == "r":
        criterion = nn.MSELoss(reduction='mean')
        print("Running on Regression Model!")
    elif model_type == "c":
        criterion = nn.CrossEntropyLoss()
        print("Running on Classification Model!")
    else:
        criterion = MultiLoss()
        print("Running on Multi Model!")
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )
        train_losses.append(train_loss)

        val_loss = eval_epoch(
            model,
            val_loader,
            criterion,
            device
        )
        val_losses.append(val_loss)
        
        print(
            f"\nEpoch {epoch}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({"model_state": model.state_dict(),"type": model.type, "train_losses":train_losses, "val_losses":val_losses}, "model.pth")
        else:
            epochs_without_improvement += 1

            print(
                f"No improvement :(\n"
                f"Patience count: {epochs_without_improvement}/{patience}"
            )

            if epochs_without_improvement >= patience:
                print("Early stopping triggered")
                break

    
    checkpoint = torch.load("model.pth")
    model.load_state_dict(checkpoint["model_state"])
    test_loss = eval_epoch(model, test_loader, criterion, device, save_outputs=True)
    print(f"Test Loss: {test_loss:.4f}")

    return model

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, labels in tqdm(loader, leave=False, desc="Training"):
        images = images.to(device)
        if model.type == "c":
            labels = labels[1].to(device)
        elif model.type == "r":  
            labels = labels[0].to(device).unsqueeze(1) 
        else:
            labels = (labels[0].to(device).unsqueeze(1), labels[1].to(device))

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

def eval_epoch(model, loader, criterion, device, save_outputs=False):
    model.eval()
    total_loss = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc="Validation"):
            images = images.to(device)
            if model.type == "c":
                labels = labels[1].to(device)
            elif model.type == "r":  
                labels = labels[0].to(device).unsqueeze(1) 
            else:
                labels = (labels[0].to(device).unsqueeze(1), labels[1].to(device))

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            if model.type == "c":
                preds = torch.argmax(outputs, dim=1)
            elif model.type == "r":
                preds = outputs.squeeze()
            else:
               preds = outputs[0].squeeze()
               labels = labels[0]

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())  

    if save_outputs:
        os.makedirs("predictions", exist_ok=True)

        existing_files = os.listdir("predictions")
        indices = []
        for f in existing_files:
            if f.startswith("predictions_train") and f.endswith(".csv"):
                num_part = f[len("predictions_train"):-len(".csv")]
                if num_part.isdigit():
                    indices.append(int(num_part))
        next_index = max(indices) + 1 if indices else 0

        df = pd.DataFrame({
            "real": all_labels,
            "pred": all_preds
        })
        filename = f"predictions/predictions_train{next_index}.csv"
        df.to_csv(filename, index=False)
        print(f"Test dataset preds on {filename}")

    return total_loss / len(loader)
