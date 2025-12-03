# Long-Tailed Medical Image Classification

This repository provides a PyTorch-based framework for multi-label classification of medical images, focusing on long-tailed distributions and robust handling of class imbalance. The main application is the NIH ChestX-ray14 dataset, but the code is adaptable to other medical imaging tasks.

## Features
- DenseNet121 and ResNet architectures
- Custom loss functions for imbalance: Focal, LDAM, Balanced Softmax, Equalization, Weighted BCE, ASL
- Advanced augmentations: SaliencyMix, ManifoldMixup, MoEx
- Automated experiment scripts (SLURM)
- Per-class and macro F1, AUROC, AP metrics
- Data exploration notebook

## Directory Structure
- `src/` — Core code: dataloading, training, loss functions, augmentations
- `models/` — Model definitions (DenseNet, ResNet)
- `experiments/` — Experiment outputs, metrics, logs
- `notebooks/` — Data exploration and visualization
- `logs/` — SLURM job logs

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ajay-vikram/Long-Tailed-Medical-Image-Classification.git
   cd Long-Tailed-Medical-Image-Classification
   ```
2. Create a conda environment (recommended):
   ```bash
   conda env create -p {env_path} -f venv.yml
   conda activate {env_path}
   ```

## Usage
### Training
Run training with customizable arguments:
```bash
python main.py --train --model DenseNet121 --loss focal --train_epochs 20 --train_lr 1e-4 --train_dir experiments/densenet121_focal --num_classes 14
```
**Important arguments:**
- `--train` : Enable training mode.
- `--model` : Model architecture (`DenseNet121`, `ResNet50`, etc.).
- `--loss` : Loss function (`focal`, `ldam`, `balanced_softmax`, `equalization`, `weighted_bce`, `asl`).
- `--train_epochs` : Number of training epochs.
- `--train_lr` : Learning rate.
- `--train_dir` : Output directory for experiment logs and checkpoints.
- `--num_classes` : Number of output classes (default: 14 for NIH ChestX-ray14).
- `--use_salmix` : Enable SaliencyMix augmentation.
- `--use_manifoldmixup` : Enable ManifoldMixup augmentation.
- `--use_moex` : Enable MoEx augmentation.
- `--salmix_prob`, `--manifoldmixup_prob`, `--moex_prob` : Probability of applying each augmentation.
- `--ldam_max_m`, `--ldam_s` : LDAM loss hyperparameters.
- `--eq_gamma`, `--eq_lam` : Equalization loss hyperparameters.
- `--focal_alpha`, `--focal_gamma` : Focal loss hyperparameters.

See `job*.sh` for SLURM job examples and more advanced configurations.

### Data Exploration
Use `notebooks/data_exploration.ipynb` to visualize label distribution and dataset statistics.

### Experiment Results
Metrics and logs are saved in `experiments/` and `logs/` after each run. Per-class and macro F1, AUROC, and AP scores are reported in `metrics.txt`.

## Citation
If you use this codebase, please cite the original NIH ChestX-ray14 paper and this repository.

## License
This project is licensed under the MIT License.
