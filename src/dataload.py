import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

class NIHChestXrayDataset(Dataset):
    def __init__(self, csv_path, image_dir, file_list, class_names, transform=None):
        self.image_dir = image_dir
        self.class_names = class_names
        self.transform = transform

        # Load metadata
        df = pd.read_csv(csv_path, usecols=["Image Index", "Finding Labels"])

        # Convert relative image names to absolute paths
        df["Image Index"] = [os.path.join(image_dir, fname) for fname in df["Image Index"].values]

        # Restrict to file list 
        with open(file_list, "r") as f:
            valid_images = {line.strip() for line in f}

        df = df[df["Image Index"].apply(lambda p: os.path.basename(p) in valid_images)]

        # Remove "No Finding" entries 
        df = df[df["Finding Labels"] != "No Finding"].reset_index(drop=True)

        #  Multi-label one-hot encoding
        labels = np.zeros((len(df), len(class_names)))
        for i, label_str in tqdm(enumerate(df["Finding Labels"].values), total=len(df)):
            for l in label_str.split("|"):
                if l in class_names:
                    labels[i, class_names.index(l)] = 1

        self.df = df
        self.labels = labels

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "Image Index"]
        image = np.array(Image.open(img_path).convert("RGB"))
        label = torch.tensor(self.labels[idx]).float()

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, label


def compute_mean_std(image_dir, file_list):
    files = [line.strip() for line in open(file_list)]
    n = 0
    mean = torch.zeros(3)
    std = torch.zeros(3)

    for fname in tqdm(files):
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        img = transforms.ToTensor()(img)  # [3,H,W] in [0,1]
        n += 1
        mean += img.mean(dim=(1,2))
        std += img.std(dim=(1,2))

    mean /= n
    std /= n
    return mean.tolist(), std.tolist()