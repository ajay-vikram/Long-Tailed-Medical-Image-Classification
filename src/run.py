import torch
import torch.backends
import torch.backends.cudnn
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
from time import time
import os
import numpy as np
import argparse
from typing import Union, Any
from typing import List
from typing import Dict as dict
from typing import Tuple as tuple
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.utils import AverageMeter
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from src.loss import AsymmetricLoss
from src.augmentations.saliencymix import SaliencyMix
from src.augmentations.manifoldmixup import ManifoldMixup
from models.moex_densenet import MoExDenseNet

__all__ = ['Runner']

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.eval_steps = args.eval_steps
        self.criterion = self.get_criterion()
        self.salmix = SaliencyMix(beta=self.args.beta, prob=self.args.salmix_prob)
        self.manifoldmixup = ManifoldMixup(
            alpha=getattr(self.args, 'manifoldmixup_alpha', 2.0),
            prob=getattr(self.args, 'manifoldmixup_prob', 0.5),
            input_mixup=True
        )

    # Loss
    def get_criterion(self) -> nn.modules.loss._Loss:
        if self.args.loss == "bce":
            criterion = nn.BCEWithLogitsLoss()
        elif self.args.loss == "asl":
            criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
        else:
            raise ValueError(f"Unsupported loss function: {self.args.loss}")
        return criterion

    # Optimizer
    def get_optimizer(self, model: nn.Module, lr: float, alpha: float) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=alpha)
        return optimizer

    # Scheduler
    def get_scheduler(self,
                      optimizer: torch.optim.Optimizer,
                      steps_per_epoch: int,
                      total_epochs: int) -> torch.optim.lr_scheduler._LRScheduler:
        return ReduceLROnPlateau(optimizer, factor=0.1, mode='min', patience=1)

    # Dataloaders
    def set_train_dataloader(self, train_dataloader: DataLoader) -> None:
        self.train_dataloader = train_dataloader

    def set_dev_dataloader(self, dev_dataloader: DataLoader) -> None:
        self.dev_dataloader = dev_dataloader

    def set_test_dataloader(self, test_dataloader: DataLoader) -> None:
        self.test_dataloader = test_dataloader

    def set_dataloaders(self,
                        train_dataloader: DataLoader,
                        dev_dataloader: DataLoader,
                        test_dataloader: DataLoader) -> None:
        self.set_train_dataloader(train_dataloader)
        self.set_dev_dataloader(dev_dataloader)
        self.set_test_dataloader(test_dataloader)

    # Initialization
    def run_init(self) -> None:
        self.cur_epoch = 0
        self.prev_loss = 1e9
        self.train_losses, self.dev_losses = AverageMeter(), AverageMeter()
        self.train_loss_history, self.dev_loss_history = [], []
        self.best_f1 = 0.0
        self.best_acc = 0.0
        self.best_loss = float("inf")

    # Forward pass
    def forward(self, X: torch.Tensor, y: torch.Tensor, epoch: int = None) -> tuple[torch.Tensor, torch.Tensor, None]:
        X = X.to('cuda', non_blocking=True)
        y = y.to('cuda', non_blocking=True).float()   
        
        # Apply augmentations (only one can be active at a time)
        if getattr(self.args, "use_manifoldmixup", False) and self.model.training:
            X, labels_a, labels_b, lam = self.manifoldmixup(X, y)
            y_mix = lam * labels_a + (1 - lam) * labels_b
            logits = self.model(X)
            loss = self.criterion(logits, y_mix)
        elif getattr(self.args, "use_salmix", False) and self.model.training:
            X, labels_a, labels_b, lam = self.salmix(X, y)
            y = lam * labels_a + (1 - lam) * labels_b
        if getattr(self.args, "use_moex", False) and isinstance(self.model, MoExDenseNet) and self.model.training:
            apply = torch.rand(1, device=X.device).item() < getattr(self.args, "moex_prob", 0.5)
            swap_index = torch.randperm(X.size(0), device=X.device) if apply else None
            logits = self.model(
                X,
                swap_index=swap_index,
                moex_norm=getattr(self.args, "moex_norm_type", "bn"),
                moex_epsilon=getattr(self.args, "moex_epsilon", 1e-5),
                moex_layer=getattr(self.args, "moex_layer", "pool0"),
                moex_positive_only=getattr(self.args, "moex_positive_only", False),
            )
        else:
            logits = self.model(X)
        loss = self.criterion(logits, y)
        return logits, loss, None

    # Helper metrics
    @staticmethod
    def compute_metrics(y_true_: torch.Tensor, y_pred_: torch.Tensor) -> dict:
        """
        Compute multi-label metrics:
        Average Precision (area under PR curve),
        AUROC (area under ROC),
        macro-averaged F1 score plus per-class F1.
        """
        y_true_np = y_true_.cpu().numpy()
        y_pred_sigmoid = torch.sigmoid(y_pred_).cpu().numpy()  # convert logits → probabilities
        y_pred_bin = (y_pred_sigmoid > 0.5).astype(np.float32)

        # Average Precision (macro)
        ap = average_precision_score(y_true_np, y_pred_sigmoid, average="macro")
      
        # AUROC (macro)
        auroc = roc_auc_score(y_true_np, y_pred_sigmoid, average="macro")

        # F1 (macro)
        f1_macro = f1_score(y_true_np, y_pred_bin, average="macro", zero_division=0)
        f1_per_class = f1_score(y_true_np, y_pred_bin, average=None, zero_division=0)

        return {"AP": ap, "AUROC": auroc, "F1": f1_macro, "F1_per_class": f1_per_class}

    def format_per_class_f1(self, f1_per_class: np.ndarray) -> str:
        """
        Format per-class F1 scores for readable console output.
        """
        class_names = getattr(self.args, "classes", None)
        if class_names:
            pairs = zip(class_names, f1_per_class)
            return ", ".join([f"{cls}:{score:.2f}" for cls, score in pairs])
        return ", ".join([f"{idx}:{score:.2f}" for idx, score in enumerate(f1_per_class)])

    def _save_best_test_metrics(self, metrics: dict) -> None:
        """
        Persist best test metrics (AP, AUROC, macro F1, per-class F1) to disk.
        """
        per_class_lines = "\n".join(
            [f"{cls}: {score:.4f}" for cls, score in zip(self.args.classes, metrics["F1_per_class"])]
        )
        with open(os.path.join(self.meta_dir, "metrics.txt"), "w") as f:
            f.write(
                f"Best Test F1 (macro): {metrics['F1']:.4f}\n"
                f"Best Test AP: {metrics['AP']:.4f}\n"
                f"Best Test AUROC: {metrics['AUROC']:.4f}\n"
                "Per-class F1:\n"
                f"{per_class_lines}\n"
            )

    # Training loop
    def train_loop(self, epoch=None) -> float:
        self.model.train()
        epoch_start = time()
        losses = AverageMeter()
        y_true, y_pred = [], []
        t = tqdm(self.train_dataloader, leave=False, unit="batches")

        for X, y in t:
            self.optimizer.zero_grad()
            pred_y, loss, _ = self.forward(X, y, epoch)
            self.train_losses.update(loss.item(), X.shape[0])
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), X.shape[0])
            y_true.append(y.cpu())
            y_pred.append(pred_y.detach().cpu())

        # Metrics
        y_true_ = torch.cat(y_true)
        y_pred_ = torch.cat(y_pred)
        metrics = self.compute_metrics(y_true_, y_pred_)
        per_class_f1_str = self.format_per_class_f1(metrics["F1_per_class"])

        epoch_loss = losses.avg
        self.train_loss_history.append(epoch_loss)
        epoch_end = time()
        print(
            f"Train | Loss: {epoch_loss:.4f} | "
            f"AP: {metrics['AP']:.4f} | AUROC: {metrics['AUROC']:.4f} | "
            f"F1: {metrics['F1']:.2f} | F1/class: {per_class_f1_str} "
            f"| Time: {epoch_end - epoch_start:.2f}s"
        )
        self.train_losses.reset()
        return epoch_loss

    # Validation loop
    def dev_loop(self, fname: str = None) -> float:
        self.model.eval()
        epoch_start = time()
        losses = AverageMeter()
        y_true, y_pred = [], []
        t = tqdm(self.dev_dataloader, leave=False, unit="batches")

        for X, y in t:
            with torch.no_grad():
                pred_y, loss, _ = self.forward(X, y, epoch=None)
            losses.update(loss.item(), X.shape[0])
            y_true.append(y.cpu())
            y_pred.append(pred_y.detach().cpu())

        y_true_ = torch.cat(y_true)
        y_pred_ = torch.cat(y_pred)

        metrics = self.compute_metrics(y_true_, y_pred_)
        per_class_f1_str = self.format_per_class_f1(metrics["F1_per_class"])
        epoch_loss = losses.avg
        self.dev_loss_history.append(epoch_loss)
        epoch_end = time()

        print(
            f"Dev | Loss: {epoch_loss:.4f} | "
            f"AP: {metrics['AP']:.4f} | AUROC: {metrics['AUROC']:.4f} | "
            f"F1: {metrics['F1']:.2f} | F1/class: {per_class_f1_str} "
            f"| Time: {epoch_end - epoch_start:.2f}s"
        )

        self.dev_losses.reset()
        return epoch_loss

    # Test loop
    def evaluate(self,
                 train: bool = False,
                 dev: bool = False,
                 test: bool = True,
                 fname: str = None,
                 flag: bool = True) -> tuple[list[str], list[tuple[np.ndarray, np.ndarray]]]:

        res, out = [], []
        self.model.eval()
        for phase, dataloader in {"test": self.test_dataloader}.items():
            if not eval(phase):
                continue

            epoch_start = time()
            y_true, y_pred = [], []
            losses = AverageMeter()
            t = tqdm(dataloader, leave=False, unit="batches")

            for X, y in t:
                with torch.no_grad():
                    pred_y, loss, _ = self.forward(X, y, epoch=None)
                losses.update(loss.item(), X.shape[0])
                y_true.append(y.cpu())
                y_pred.append(pred_y.cpu())

            y_true_ = torch.cat(y_true)
            y_pred_ = torch.cat(y_pred)
            
            metrics = self.compute_metrics(y_true_, y_pred_)
            per_class_f1_str = self.format_per_class_f1(metrics["F1_per_class"])
            epoch_loss = losses.avg
            epoch_end = time()

            print(
                f"{phase} | Loss: {epoch_loss:.4f} | "
                f"AP: {metrics['AP']:.4f} | AUROC: {metrics['AUROC']:.4f} | "
                f"F1: {metrics['F1']:.2f} | F1/class: {per_class_f1_str} "
                f"| Time: {epoch_end - epoch_start:.2f}s"
            )

            res.append(
                f"{phase} | Loss: {epoch_loss:.4f} | AP: {metrics['AP']:.4f} | "
                f"AUROC: {metrics['AUROC']:.4f} | F1: {metrics['F1']:.2f}"
            )

            # Save best checkpoint by F1
            if flag and metrics['F1'] > self.best_f1:
                self._save_checkpoint(fname)
                self.best_f1 = metrics['F1']
                self._save_best_test_metrics(metrics)


        return res, None

    # Saving checkpoint
    def _save_checkpoint(self,
                         checkpoint: dict[str, Any] = None,
                         fname: str = None) -> None:
        print("\033[1;92m\nSaving checkpoint...\n\033[0m")
        fname = os.path.join(self.meta_dir, 'checkpoint.pt') if fname is None else fname
        torch.save(checkpoint if checkpoint else self.model.state_dict(), fname)
        return None

    # Bookkeeping
    def save_training_loss(self) -> None:
        np.savetxt(os.path.join(self.meta_dir, 'train_loss_history.txt'),
                   np.stack(self.train_loss_history))
        np.savetxt(os.path.join(self.meta_dir, 'dev_loss_history.txt'),
                   np.stack(self.dev_loss_history))

    # Epoch wrapper
    def run_epoch(self, fname: str = None) -> None:
        print('-' * 50)
        print(f'Epoch: {self.cur_epoch + 1} / {self.epochs}')
        _ = self.train_loop(epoch=self.cur_epoch)
        epoch_loss = self.dev_loop(fname=fname)
        self.scheduler.step(epoch_loss)  
        _ = self.evaluate(fname=fname)
        self.save_training_loss()
        print(f'\n->> lr: {self.optimizer.param_groups[0]["lr"]}\n')
        return None
