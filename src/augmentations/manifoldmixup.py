import torch
import torch.nn as nn
import numpy as np


class ManifoldMixup:
    """
    Manifold Mixup augmentation.
    
    Manifold Mixup interpolates hidden representations in the network to learn
    smoother decision boundaries. This implementation supports:
    1. Input-level mixing (compatible with existing interface)
    2. Hidden-layer mixing (requires model modification)
    
    Reference: Verma et al., "Manifold Mixup: Better Representations by Interpolating Hidden States" (ICML 2019)
    https://arxiv.org/abs/1806.05236
    """
    def __init__(self, alpha=2.0, prob=0.5, input_mixup=True):
        """
        Args:
            alpha: Beta distribution parameter for lambda sampling (default: 2.0)
            prob: Probability of applying mixup
            input_mixup: If True, mix at input level. If False, requires model modification for hidden layer mixing.
        """
        self.alpha = alpha
        self.prob = prob
        self.input_mixup = input_mixup
        self.lam = None
        self.index = None

    def __call__(self, images, labels):
        """
        Apply Manifold Mixup augmentation.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            labels: Tensor of shape (B, num_classes)
            
        Returns:
            images: Mixed images (if input_mixup=True) or original images (if input_mixup=False)
            labels_a: Original labels
            labels_b: Shuffled labels
            lam: Mixing coefficient
        """
        r = np.random.rand(1)
        if r < self.prob:
            # Sample lambda from Beta distribution
            self.lam = np.random.beta(self.alpha, self.alpha)
            # Ensure lambda is in [0, 1]
            self.lam = max(0, min(1, self.lam))
            
            # Get random permutation
            batch_size = images.size(0)
            self.index = torch.randperm(batch_size).to(images.device)
            
            labels_a = labels
            labels_b = labels[self.index]
            
            if self.input_mixup:
                # Mix at input level (similar to Input Mixup)
                images = self.lam * images + (1 - self.lam) * images[self.index]
                return images, labels_a, labels_b, self.lam
            else:
                # For hidden layer mixing, return original images
                # The actual mixing will happen in the model's forward pass
                return images, labels_a, labels_b, self.lam
        else:
            self.lam = 1.0
            self.index = None
            return images, labels, labels, 1.0


def manifold_mixup_criterion(criterion, y_a, y_b, lam):
    """
    Compute loss for Manifold Mixup.
    
    Args:
        criterion: Loss function
        y_a: Predictions for original samples
        y_b: Predictions for mixed samples
        lam: Mixing coefficient
        
    Returns:
        Mixed loss
    """
    return lam * criterion(y_a, y_a) + (1 - lam) * criterion(y_b, y_b)


class ManifoldMixupModel(nn.Module):
    """
    Wrapper for models to support Manifold Mixup at hidden layers.
    This allows mixing at intermediate layers, not just the input.
    """
    def __init__(self, model, mixup_layers=None):
        """
        Args:
            model: Base model to wrap
            mixup_layers: List of layer names or indices where to apply mixup.
                         If None, randomly selects a layer during training.
        """
        super(ManifoldMixupModel, self).__init__()
        self.model = model
        self.mixup_layers = mixup_layers
        self.lam = None
        self.index = None
        self.mixup_enabled = False
        
    def set_mixup(self, lam, index):
        """Set mixup parameters for this forward pass."""
        self.lam = lam
        self.index = index
        self.mixup_enabled = (lam is not None and index is not None and lam < 1.0)
        
    def clear_mixup(self):
        """Clear mixup parameters."""
        self.lam = None
        self.index = None
        self.mixup_enabled = False
        
    def _apply_mixup(self, x, layer_name=None):
        """Apply mixup to a hidden representation."""
        if self.mixup_enabled and self.lam is not None and self.index is not None:
            x_mixed = self.lam * x + (1 - self.lam) * x[self.index]
            return x_mixed
        return x
        
    def forward(self, x):
        """
        Forward pass with optional Manifold Mixup.
        This is a basic implementation - for specific models, you may need
        to override this to mix at specific layers.
        """
        if not self.mixup_enabled:
            return self.model(x)
            
        # For a generic implementation, we can mix at the input
        # For specific models, override this method to mix at desired layers
        if self.lam is not None and self.index is not None:
            x = self._apply_mixup(x)
        
        return self.model(x)



