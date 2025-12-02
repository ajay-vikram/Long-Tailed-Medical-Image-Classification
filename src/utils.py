import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from typing import List as list
__all__ = ['TRAIN_FLAGS', 'AverageMeter', 'ConfusionMatrix']


def TRAIN_FLAGS() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Long-Tailed Medical Image Classification")

    # GPU Config
    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='GPU device to be used'
    )

    # Project
    parser.add_argument(
        '--seed',
        default=123,
        type=int,
        help='IMPORTANT:- Keep the seed same for all the stages'
    )
    parser.add_argument(
        '--proj',
        default='resnet50_baseline_v2',
        type=str,
        help='Project name'
    )

    # Augmentations
    parser.add_argument(
        '--use_moex',
        action='store_true',
        default=False,
        help='Whether to use MoEx augmentation'
    )
    parser.add_argument(
        '--moex_prob',
        default=0.5,
        type=float,
        help='Probability to apply MoEx inside MoExDenseNet'
    )
    parser.add_argument(
        '--moex_norm_type',
        default='bn',
        type=str,
        help='Normalization type for MoEx (e.g., bn, in, ln, gn4)'
    )
    parser.add_argument(
        '--moex_epsilon',
        default=1e-5,
        type=float,
        help='Numerical stability epsilon for MoEx'
    )
    parser.add_argument(
        '--moex_layer',
        default='pool0',
        type=str,
        help='Layer name in MoExDenseNet where MoEx is applied'
    )
    parser.add_argument(
        '--moex_positive_only',
        action='store_true',
        default=False,
        help='Use positive-only statistics for MoEx'
    )
    parser.add_argument(
        '--use_salmix',
        action='store_true',
        default=False,
        help='Whether to use SaliencyMix augmentation'
    )
    parser.add_argument(
        '--use_manifoldmixup',
        action='store_true',
        default=False,
        help='Whether to use Manifold Mixup augmentation'
    )
    parser.add_argument(
        '--manifoldmixup_alpha',
        default=2.0,
        type=float,
        help='Alpha parameter for Manifold Mixup Beta distribution'
    )
    parser.add_argument(
        '--manifoldmixup_prob',
        default=0.5,
        type=float,
        help='Probability of applying Manifold Mixup'
    )
    parser.add_argument(
        '--manifoldmixup_input',
        action='store_true',
        default=True,
        help='If True, apply mixup at input level (default). If False, requires model changes for hidden-layer mix.'
    )
    parser.add_argument(
        '--salmix_prob',
        default=0.5,
        type=float,
        help='SaliencyMix probability'
    )
    parser.add_argument(
        '--beta',
        default=1.0,
        type=float,
        help='Beta parameter for SaliencyMix augmentation'
    )

    # Focal Loss Parameters
    parser.add_argument(
        '--focal_alpha',
        default=0.25,
        type=float,
        help='Alpha parameter for Focal Loss (weighting factor for positive class)'
    )
    parser.add_argument(
        '--focal_gamma',
        default=2.0,
        type=float,
        help='Gamma parameter for Focal Loss (focusing parameter, higher = more focus on hard examples)'
    )

    # LDAM Loss Parameters
    parser.add_argument(
        '--ldam_max_m',
        default=0.5,
        type=float,
        help='Maximum margin for LDAM Loss (Label-Distribution-Aware Margin)'
    )
    parser.add_argument(
        '--ldam_s',
        default=30,
        type=float,
        help='Scaling factor for LDAM Loss logits'
    )

    # Equalization Loss Parameters
    parser.add_argument(
        '--eq_gamma',
        default=2.0,
        type=float,
        help='Gamma (focusing parameter) for Equalization Loss'
    )
    parser.add_argument(
        '--eq_lam',
        default=0.1,
        type=float,
        help='Lambda (EMA factor) for Equalization Loss'
    )

    # Model
    parser.add_argument(
        '--model',
        default='ResNet50',
        choices=['ResNet50', 'ResNet101', 'ResNet152', 'DenseNet121'],
        help='Which model to be used for training'
    )

    # Data
    parser.add_argument(
        '--image_dir',
        default="./data/images-224/images-224",
        help='The directory for the images'
    )
    parser.add_argument(
        '--entry_file',
        default="./data/Data_Entry_2017.csv",
        help='The data entry file for the dataset'
    )
    parser.add_argument(
        '--train_val_list',
        default="./data/train_val_list_NIH.txt",
        help='The file containing the train/val split'
    )
    parser.add_argument(
        '--test_list',
        default="./data/test_list_NIH.txt",
        help='The file containing the test split'
    )
    parser.add_argument(
        '--split',
        default=0.8,
        type=float,
        help='Proportion of the dataset to include in the train split'
    )
    parser.add_argument(
        '--inp_size',
        default=[3, 224, 224],
        type=list[int],
        help='Input size for the model'
    )
    parser.add_argument(
        '--classes',
        default=["Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia"
        ],
        type=list[str],
        help='List of classes for the dataset'
    )


    # Optimization
    parser.add_argument(
        '--train_lr',
        default=1e-3,
        type=float,
        help='Learning rate for the original F32 model'
    )
    parser.add_argument(
        '--train_alpha',
        default=1e-4,
        type=float,
        help='Weight decay for training the original F32 model'
    )
    parser.add_argument(
        '--max_lr',
        default=0.01,
        type=float,
        help='Maximum learning rate for OneCycleLR'
    )

    # Training
    parser.add_argument(
        '--train_epochs',
        default=50,
        type=int,
        help='#epochs for training the original F32 model'
    )
    parser.add_argument(
        '--batch_size',
        default=64,
        type=int
    )
    parser.add_argument(
        '--workers',
        default=4,
        type=int,
        help='#processes for the dataloading'
    )
    parser.add_argument(
        '--loss',
        default="bce",
        type=str,
        choices=['bce', 'asl', 'focal', 'focal_optimized', 'weighted_bce', 'ldam', 'balanced_softmax', 'equalization'],
        help='Loss function to be used for training. Options: bce, asl, focal, focal_optimized, weighted_bce, ldam, balanced_softmax, equalization'
    )
    parser.add_argument(
        '--eval_steps',
        default=10,
        type=int,
        help='#epoch steps during training for evaluating the model and determining \
        to modify the learning rate'
    )

    # Stages
    parser.add_argument(
        '--train',
        action='store_true',
        default=False,
        help='Train the original F32 model'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='Test the trained F32 model'
    )

    args = parser.parse_args()
    args = set_sub_folder(args)
    os.makedirs(args.train_dir, exist_ok=True) if args.train else None
    try:
        with open(os.path.join(args.meta_dir, 'train_args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    except FileNotFoundError:
        pass
    return args


def set_sub_folder(args: argparse.Namespace) -> argparse.Namespace:
    args.meta_dir = os.path.join('experiments', args.proj)
    args.train_dir = os.path.join(args.meta_dir, 'train')
    return args


class AverageMeter:

    def __init__(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, 
               val: float, 
               n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


class ConfusionMatrix:
    """
    A class to create confusion matrix and display it or calculate precision, recall, F1 score.
    """

    def __init__(self, 
                 true_values: np.ndarray, 
                 predicted_values: np.ndarray, 
                 classes: list = None, 
                 from_one_hot: bool = False):
        """
        Creates an object of ConfusionMatrix.

        Parameters:
        -----------
        true_values: ndarray
            The target values.
        predicted_values: ndarray
            The predicted outcomes.
        classes: list, optional, default: None
            The list of labels corresponding to the encoded values.
        from_one_hot: bool, optional, default: False
            Whether `true_values` and `predicted_values` are one-hot encoded (True) or not (False).
        """
        self.matrix = self._get_matrix(true_values, predicted_values, from_one_hot)
        self.classes = classes if classes else [str(i) for i in range(self.matrix.shape[0])]
        self.weights = np.sum(self.matrix, axis=1)

    def __str__(self) -> str:
        return f"{self.matrix}"

    def plot(self, 
             block: bool = True, 
             colorbar: bool = True,
             show_percentage: bool = False,
             title: str = None,
             save_file: list[str] = None) -> None:
        """
        Plots the confusion matrix.

        Parameters:
        -----------
        block: bool, optional, default: True
            Whether the block the execution of the following codes (True) or not (False).
        colorbar: bool, optional, default: True
            Whether to show the colorbar (True) or not (False).
        show_percentage: bool, optional, default: False
            If True, the values shown are the percentages wrt the total number of predictions in the particular row.
        title: str, optional, default: None
            Title of the plot.
        save_file: list[str], optional, default: None
            Paths to save the plot.
        """
        mat_to_show = self.matrix if not show_percentage \
            else self.matrix * 100 / self.matrix.sum(axis=1, keepdims=True)

        fig, ax = plt.subplots()
        im = ax.imshow(mat_to_show)
        ax.figure.colorbar(im, ax=ax) if colorbar else None

        ax.set(xticks=np.arange(self.matrix.shape[1]), yticks=np.arange(self.matrix.shape[0]),
               xticklabels=self.classes, yticklabels=self.classes)
        ax.set_title("Confusion Matrix" if title is None else title, weight="bold")
        ax.set_xlabel("Predicted label", weight="bold")
        ax.set_ylabel("True label", weight="bold")

        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

        for i in range(mat_to_show.shape[0]):
            for j in range(mat_to_show.shape[1]):
                if mat_to_show[i, j] == 0:
                    continue
                color = "w" if mat_to_show[i, j] < np.max(mat_to_show) / 2 else "k"
                ax.text(j, i, f"{mat_to_show[i, j]}" if not show_percentage else f"{mat_to_show[i, j]:.2f}", 
                        ha="center", va="center", color=color, fontsize=6, weight="bold")

        fig.tight_layout()
        [plt.savefig(path) for path in save_file]
        plt.show(block=block)

    def precision(self, 
                  average: Union[str, None] = "micro") -> Union[np.ndarray, float]:
        """
        Calculates precision.

        Parameters:
        -----------
        average: None or str, optional, default: 'micro'
            Type of averaging.

            None: Computes the precisions of individual classes.

            'micro': Computes the global average by counting the sums of the True Positives (TP), False Positives (FP).

            'macro': Computes the arithmatic mean of the per-class precisions treating all classes equally.

            'weighted': Computes the mean of all per-class precisions
            considering the number of actual occurrences of each class.
        
        Returns:
        --------
        precisions: ndarray or float
            The precision.
        """

        N = self.matrix.shape[0]
        precisions = np.zeros(N)
        for i in range(N):
            precisions[i] = self.matrix[i, i] / np.sum(self.matrix[:, i]) if np.sum(self.matrix[:, i]) != 0 else 0

        if not average or average is None:
            return precisions

        if average == "micro":
            return np.sum(self.matrix.diagonal()) / np.sum(self.matrix) if np.sum(self.matrix) != 0 else 0

        if average == "macro":
            return np.mean(precisions)

        if average == "weighted":
            return np.sum(self.weights * precisions) / np.sum(self.weights) if np.sum(self.weights) != 0 else 0

        raise Exception(f"Invalid type '{average}' for parameter 'average'. "
                        f"Possible types are: None, 'micro', 'macro', 'weighted'")

    def recall(self, 
               average: Union[str, None] = "micro") -> Union[np.ndarray, float]:
        """
        Calculates recall.

        Parameters:
        -----------
        average: None or str, optional, default: 'micro'
            Type of averaging.

            None: Computes the recalls of individual classes.

            'micro': Computes the global average by counting the sums of the True Positives (TP), False Negatives (FN).

            'macro': Computes the arithmatic mean of the per-class recalls treating all classes equally.

            'weighted': Computes the mean of all per-class recalls
            considering the number of actual occurrences of each class.

        Returns:
        --------
        recalls: ndarray or float
            The recall.
        """

        N = self.matrix.shape[0]
        recalls = np.zeros(N)
        for i in range(N):
            recalls[i] = self.matrix[i, i] / np.sum(self.matrix[i, :]) if np.sum(self.matrix[i, :]) != 0 else 0

        if not average:
            return recalls

        if average == "micro":
            return np.sum(self.matrix.diagonal()) / np.sum(self.matrix) if np.sum(self.matrix) != 0 else 0

        if average == "macro":
            return np.mean(recalls)

        if average == "weighted":
            return np.sum(self.weights * recalls) / np.sum(self.weights) if np.sum(self.weights) != 0 else 0

        raise Exception(f"Invalid type '{average}' for parameter 'average'. "
                        f"Possible types are: None, 'micro', 'macro', 'weighted'")

    def f1_score(self,
                 average: Union[str, None] = "micro") -> Union[np.ndarray, float]:
        """
        Calculates F1 score.

        Parameters:
        -----------
        average: None or str, optional, default: 'micro'
            Type of averaging.

            None: Computes the F1 scores of individual classes.

            'micro': Computes the global average by counting the sums of the
            True Positives (TP), False Positives (FP), False Negatives (FN).

            'macro': Computes the arithmatic mean of the per-class F1 scores treating all classes equally.

            'weighted': Computes the mean of all per-class F1 scores
            considering the number of actual occurrences of each class.

        Returns:
        --------
        precisions: ndarray or float
            The F1 score.
        """

        precisions = self.precision(average=None)
        recalls = self.recall(average=None)
        f1_scores = np.zeros_like(precisions)
        mask = (precisions != 0) * (recalls != 0)
        f1_scores[mask] = 2 * precisions[mask] * recalls[mask] / (precisions[mask] + recalls[mask])

        if not average:
            return f1_scores

        if average == "micro":
            return np.sum(self.matrix.diagonal()) / np.sum(self.matrix) if np.sum(self.matrix) != 0 else 0

        if average == "macro":
            return np.mean(f1_scores)

        if average == "weighted":
            return np.sum(self.weights * f1_scores) / np.sum(self.weights) if np.sum(self.weights) != 0 else 0

        raise Exception(f"Invalid type '{average}' for parameter 'average'. "
                        f"Possible types are: None, 'micro', 'macro', 'weighted'")

    @staticmethod
    def _get_matrix(true_values: np.ndarray, 
                    predicted_values: np.ndarray, 
                    from_one_hot: bool) -> np.ndarray:
        """
        Creates the confusion matrix.

        Parameters:
        -----------
        true_values: ndarray
            The target values.
        predicted_values: ndarray
            The predicted outcomes.
        from_one_hot: bool
            Whether true_values and predicted_values are one-hot encoded or not.

        Returns:
        --------
        matrix: ndarray
            The confusion matrix
        """

        if true_values.shape != predicted_values.shape:
            raise Exception("Different shapes of 'true_values' and 'predicted_values'")

        if from_one_hot:
            dim = true_values.shape[0]
            matrix = np.zeros((dim, dim))
            for j in range(true_values.shape[1]):
                matrix[np.argmax(true_values[:, j]), np.argmax(predicted_values[:, j])] += 1
            return matrix

        dim = int(np.max(np.concatenate((true_values, predicted_values)))) + 1
        matrix = np.zeros((dim, dim), dtype=int)
        for j in range(true_values.shape[0]):
            matrix[true_values[j], predicted_values[j]] += 1
        return matrix
    
