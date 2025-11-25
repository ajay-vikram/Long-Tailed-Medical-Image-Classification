import torch
import torch.nn.functional as F
import numpy as np

class MoEx:
    """
    Mixup of Expert (MoEx) augmentation.
    Swaps normalization statistics between samples in a batch.
    Compatible with SaliencyMix interface for use in run.py.
    """
    def __init__(self, norm_type='bn', epsilon=1e-5, prob=0.5):
        self.norm_type = norm_type
        self.epsilon = epsilon
        self.prob = prob

    def compute_norm(self, x, norm_dims, positive_only):
        if positive_only:
            x_pos = F.relu(x)
            s1 = x_pos.sum(dim=norm_dims, keepdim=True)
            s2 = x_pos.pow(2).sum(dim=norm_dims, keepdim=True)
            count = x_pos.gt(0).sum(dim=norm_dims, keepdim=True)
            count[count == 0] = 1  # deal with 0/0
            mean = s1 / count
            var = s2 / count - mean.pow(2)
            std = var.add(self.epsilon).sqrt()
        else:
            mean = x.mean(dim=norm_dims, keepdim=True)
            std = x.var(dim=norm_dims, keepdim=True).add(self.epsilon).sqrt()
        return mean, std

    def moex_bbox(self, x, swap_index):
        B, C, H, W = x.shape
        if self.norm_type == 'bn':
            norm_dims = [0, 2, 3]
        elif self.norm_type == 'in':
            norm_dims = [2, 3]
        elif self.norm_type == 'ln':
            norm_dims = [1, 2, 3]
        elif self.norm_type == 'pono':
            norm_dims = [1]
        elif self.norm_type.startswith('gn'):
            if self.norm_type.startswith('gn-d'):
                # gn-d4 means GN where each group has 4 dims
                G_dim = int(self.norm_type[4:])
                G = C // G_dim
            else:
                # gn4 means GN with 4 groups
                G = int(self.norm_type[2:])
                G_dim = C // G
            x = x.view(B, G, G_dim, H, W)
            norm_dims = [2, 3, 4]
        elif self.norm_type.startswith('gpono'):
            if self.norm_type.startswith('gpono-d'):
                # gpono-d4 means GPONO where each group has 4 dims
                G_dim = int(self.norm_type[len('gpono-d'):])
                G = C // G_dim
            else:
                # gpono4 means GPONO with 4 groups
                G = int(self.norm_type[len('gpono'):])
                G_dim = C // G
            x = x.view(B, G, G_dim, H, W)
            norm_dims = [2]
        else:
            raise NotImplementedError(f'norm_type={self.norm_type}')

        mean, std = self.compute_norm(x, norm_dims, positive_only=False)
        swap_mean = mean[swap_index]
        swap_std = std[swap_index]
        scale = swap_std / std
        shift = swap_mean - mean * scale
        x = x * scale + shift
        
        # Reshape back if we used group normalization
        if self.norm_type.startswith('gn') or self.norm_type.startswith('gpono'):
            x = x.view(B, C, H, W)
        
        return x

    def __call__(self, images, labels):
        """
        Apply MoEx augmentation to images and labels.
        
        Args:
            images: Tensor of shape (B, C, H, W)
            labels: Tensor of shape (B, num_classes)
            
        Returns:
            images: Augmented images
            labels_a: Original labels
            labels_b: Swapped labels
            lam: Mixing coefficient (always 1.0 for MoEx)
        """
        r = np.random.rand(1)
        if r < self.prob:
            swap_index = torch.randperm(images.size(0)).to(images.device)
            images = self.moex_bbox(images, swap_index)
            labels_a, labels_b = labels, labels[swap_index]
            return images, labels_a, labels_b, 1.0
        else:
            return images, labels, labels, 1.0
