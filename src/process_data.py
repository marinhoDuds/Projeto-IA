import os
import torch
import shutil
from src.dataset import AgeDataset

from torchvision import transforms
from sklearn.model_selection import train_test_split

def age_class_name(age):
    if age <= 5: return 0
    elif age<=12: return 1
    elif age <= 17: return 2
    elif age <= 29: return 3
    elif age <= 59: return 4
    else: return 5
    
def get_image_age(dataset_path):
    image_paths =  os.listdir(dataset_path)
    ages = []

    for img in image_paths:
        age = int(img.split("_")[0])
        ages.append(age)

    return image_paths, ages

def split_dataset(image_paths, ages):
    age_classes = [age_class_name(age) for age in ages]
    train_paths, test_paths, train_ages, test_ages = train_test_split(
        image_paths,
        ages,
        test_size=0.1,
        stratify=age_classes,
        random_state=42
    )

    age_classes = [age_class_name(age) for age in train_ages]
    train_paths, val_paths, train_ages, val_ages = train_test_split(
        train_paths,
        train_ages,
        test_size=0.135,
        stratify=age_classes,
        random_state=42
    )

    return (train_paths, train_ages, val_paths, val_ages, test_paths, test_ages)
    
def get_datasets(dataset_path, img_size, model_type):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    if model_type == "r":
        classification = None   
    else:
        classification = age_class_name

    datasets = os.listdir(dataset_path)
    if (len(datasets)==3):
        train_paths, train_ages = get_image_age(os.path.join(dataset_path, datasets[0]))
        val_paths, val_ages = get_image_age(os.path.join(dataset_path, datasets[1]))
        test_paths, test_ages = get_image_age(os.path.join(dataset_path, datasets[2]))
    else:
        image_paths, ages = get_image_age(dataset_path)
        train_paths, train_ages, val_paths, val_ages, test_paths, test_ages = split_dataset(image_paths, ages) 

    train_dataset = AgeDataset(dataset_path, train_paths, train_ages, transform, classification)
    val_dataset = AgeDataset(dataset_path, val_paths, val_ages, transform, classification)
    test_datasets = AgeDataset(dataset_path, test_paths, test_ages, transform, classification)

    return train_dataset, val_dataset, test_datasets

def generate_datafolders(dataset_path, output_path="data_split"):
    image_paths, ages = get_image_age(dataset_path)
    train_paths, _, val_paths, _, test_paths, _ = split_dataset(image_paths, ages)
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "val")
    test_dir = os.path.join(output_path, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print("Generating train")
    for img in train_paths:
        src = os.path.join(dataset_path, img)
        dst = os.path.join(train_dir, img)
        shutil.copy2(src, dst)

    print("Generating val")
    for img in val_paths:
        src = os.path.join(dataset_path, img)
        dst = os.path.join(val_dir, img)
        shutil.copy2(src, dst)

    print("Generating test")
    for img in test_paths:
        src = os.path.join(dataset_path, img)
        dst = os.path.join(test_dir, img)
        shutil.copy2(src, dst)
    
    print("Split data done!")

