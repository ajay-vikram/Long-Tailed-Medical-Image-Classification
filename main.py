import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import random
from src.train import Trainer
from src.utils import TRAIN_FLAGS
from src.dataload import NIHChestXrayDataset, compute_mean_std
from torchvision import transforms
import cv2
from albumentations.pytorch import ToTensorV2
import albumentations as A

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class Controller:
    def __init__(self) -> None:
        self.args = TRAIN_FLAGS()

        # Set all random seeds
        seed = self.args.seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        generator = torch.Generator().manual_seed(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_device(self.args.gpu)

        # Transforms
        transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

        # Create datasets
        full_trainval_dataset = NIHChestXrayDataset(self.args.entry_file, self.args.image_dir, self.args.train_val_list, self.args.classes, transform)
        test_dataset = NIHChestXrayDataset(self.args.entry_file, self.args.image_dir, self.args.test_list, self.args.classes, transform)

        # Split train/val 
        train_size = int(self.args.split * len(full_trainval_dataset))
        val_size = len(full_trainval_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_trainval_dataset, [train_size, val_size], generator=generator)

        print(f"Train size: {len(train_dataset)}, Validation size: {len(val_dataset)}, Test size: {len(test_dataset)}")

        # DataLoaders
        self.train_dataloader = DataLoader(train_dataset,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            pin_memory=True, persistent_workers=True, shuffle=True, generator=generator)
        self.dev_dataloader = DataLoader(val_dataset,
            batch_size=self.args.batch_size//2, num_workers=self.args.workers,
            pin_memory=True, persistent_workers=True, shuffle=False, generator=generator)
        self.test_dataloader = DataLoader(test_dataset,
            batch_size=self.args.batch_size//2, num_workers=self.args.workers,
            pin_memory=True, persistent_workers=True, shuffle=False, generator=generator)

        self.args.num_classes = len(self.args.classes)

    def train_model(self) -> None:
        trainer = Trainer(self.args, len(self.train_dataloader), train_dataloader=self.train_dataloader)
        trainer.set_dataloaders(self.train_dataloader, self.dev_dataloader,
                                self.test_dataloader)
        trainer.run()

        return None
    

def main() -> None:
    controller = Controller()
    if controller.args.train:
        controller.train_model()

if __name__ == "__main__":
    main()
