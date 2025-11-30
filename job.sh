#!/bin/bash
#SBATCH --job-name=densenet121_transformations                      # Name of job
#SBATCH --partition=compsci-gpu                              # Partition name
#SBATCH --gres=gpu:a5000:1                                   # GPU request
#SBATCH --time=10:00:00                                      # Max run time (HH:MM:SS)
#SBATCH --mem=64G                                            # Memory per node
#SBATCH --cpus-per-task=4                                    # CPU cores
#SBATCH --output=/usr/xtmp/ap843/logs/%x_%j.out              # Stdout log
#SBATCH --error=/usr/xtmp/ap843/logs/%x_%j.err               # Stderr log

# Load modules (example - adjust to your cluster)
module load miniconda/23.9.0

# Activate environment
source activate ajay

# Go to project directory
cd /usr/xtmp/ap843

# Run your code
python main.py --train --model DenseNet121 --proj densenet121_transformations 