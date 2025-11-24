import argparse
from src.run import Runner
from torchinfo import summary
import torch.nn as nn
from models.resnet import ResNet50, ResNet101, ResNet152
from models.densenet121 import DenseNet121

__all__ = ['Trainer']

class Trainer(Runner):
    def __init__(self, 
                 args: argparse.Namespace, 
                 train_dataloader_length: int) -> None:
        super().__init__(args)
        self.args = args
        self.meta_dir = args.train_dir
        self.epochs = args.train_epochs
        self.train_dataloader_length = train_dataloader_length
        self.model_id = self.args.model

        if self.model_id == "ResNet50":
            self.model = ResNet50(args.num_classes)
        elif self.model_id == "ResNet101":
            self.model = ResNet101(args.num_classes)
        elif self.model_id == "ResNet152":
            self.model = ResNet152(args.num_classes)
        elif self.model_id == "DenseNet121":
            self.model = DenseNet121(args.num_classes)
        else:
            print("Invalid Model Choice!")

        self.model = self.model.to("cuda")
        self.optimizer = self.get_optimizer(self.model, args.train_lr, args.train_alpha)
        self.scheduler = self.get_scheduler(self.optimizer, train_dataloader_length, args.train_epochs)

    def save_checkpoint(self, 
                        fname: str = None) -> None:
        checkpoint = {'epoch': self.cur_epoch,
                      'optimizer': self.optimizer,
                      'state_dict': self.model.state_dict(),
                      'best_loss': self.best_loss}
        super()._save_checkpoint(checkpoint, fname)
        return None

    def run(self) -> None:
        self.run_init()
        while self.cur_epoch < self.epochs:
            super().run_epoch()
            self.cur_epoch += 1
        self.save_training_loss()
        return None


