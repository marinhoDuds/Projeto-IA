import os
import torch

from PIL import Image
from torch.utils.data import Dataset

class AgeDataset(Dataset):
    def __init__(self, root_dir, image_paths, ages, transform, classification):
        self.root_dir = root_dir
        self.image_paths = image_paths
        self.ages = ages
        self.transform = transform
        self.classification=classification

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        age = self.ages[idx]

        if self.classification:
            label = self.classification(age)
        else:
            label = age

        return image, torch.tensor(label)
    