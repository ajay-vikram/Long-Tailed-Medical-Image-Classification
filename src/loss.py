import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    https://arxiv.org/abs/1708.02002
    
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    
    For multi-label classification:
    - gamma: focusing parameter (controls how much to down-weight easy examples)
    - alpha: weighting factor in [0, 1] to balance pos/neg examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
        """
        Args:
            alpha (float): Weight for positive class. Default: 0.25
            gamma (float): Focusing parameter. Default: 2.0
                - gamma=0: equivalent to BCE
                - gamma=2: recommended for imbalanced datasets
            reduction (str): 'mean' or 'sum'
            eps (float): Small value to avoid log(0)
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) unnormalized predictions
            targets: (B, C) binary targets for multi-label classification
        
        Returns:
            loss scalar
        """
        # Get probabilities
        p = torch.sigmoid(logits)
        
        # Compute focal term: (1 - p_t) ^ gamma
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy term: -log(p_t)
        ce = torch.where(
            targets == 1,
            -torch.log(p.clamp(min=self.eps)),
            -torch.log((1 - p).clamp(min=self.eps))
        )
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        # Compute focal loss
        loss = alpha_t * focal_weight * ce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class FocalLossOptimized(nn.Module):
    """
    Optimized Focal Loss implementation with reduced memory allocation.
    Uses in-place operations where possible.
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', eps=1e-8):
        super(FocalLossOptimized, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        
        # Compute focal term
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = torch.pow(1 - p_t, self.gamma)
        
        # Compute cross entropy
        ce = targets * torch.log(p.clamp(min=self.eps)) + (1 - targets) * torch.log((1 - p).clamp(min=self.eps))
        
        # Apply alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        # Compute loss
        loss = -alpha_t * focal_weight * ce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.mean()
        # return -loss.sum()


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for multi-label classification.
    
    Applies per-class weights based on class frequency to handle imbalance.
    Less frequent classes get higher weights during training.
    
    Usage:
        Compute class weights from training data:
        pos_count = targets.sum(dim=0)
        class_weights = (len(targets) - pos_count) / (pos_count + 1)
        
        Then pass to loss function.
    """
    def __init__(self, class_weights=None, reduction='mean', eps=1e-8):
        """
        Args:
            class_weights: (C,) tensor of weights for each class. 
                          If None, equal weights are used.
            reduction: 'mean' or 'sum'
            eps: small value to avoid log(0)
        """
        super(WeightedBCELoss, self).__init__()
        self.class_weights = class_weights
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) unnormalized predictions
            targets: (B, C) binary targets
        
        Returns:
            loss scalar
        """
        p = torch.sigmoid(logits)
        
        # BCE
        loss = -(targets * torch.log(p.clamp(min=self.eps)) + 
                 (1 - targets) * torch.log((1 - p).clamp(min=self.eps)))
        
        # Apply class weights if provided
        if self.class_weights is not None:
            # Expand weights to match batch
            if self.class_weights.device != loss.device:
                self.class_weights = self.class_weights.to(loss.device)
            loss = loss * self.class_weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class LDAMLoss(nn.Module):
    """
    Label-Distribution-Aware Margin Loss (LDAM).
    
    Reference: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss" (NeurIPS 2019)
    https://arxiv.org/abs/1906.07413
    
    Designed for long-tailed classification. Maintains per-class margins inversely 
    proportional to class frequency.
    
    For multi-label: applies margin adjustment to logits before computing loss.
    """
    def __init__(self, class_counts, max_m=0.5, s=30, reduction='mean', eps=1e-8):
        """
        Args:
            class_counts: (C,) tensor with number of positive samples per class
            max_m: maximum margin value (default: 0.5)
            s: scaling factor for logits (default: 30)
            reduction: 'mean' or 'sum'
            eps: small value to avoid log(0)
        """
        super(LDAMLoss, self).__init__()
        self.s = s
        self.eps = eps
        self.reduction = reduction
        
        # Compute per-class margins: rarer classes get larger margins
        num_classes = len(class_counts)
        min_count = class_counts.min()
        
        # Margin inversely proportional to class frequency
        margins = 1.0 / torch.sqrt(class_counts.float() + 1)
        margins = margins / margins.max() * max_m
        self.register_buffer('margins', margins)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) unnormalized predictions
            targets: (B, C) binary targets
        
        Returns:
            loss scalar
        """
        # Move margins to same device as logits
        margins = self.margins.to(logits.device)
        
        # Adjust logits with margins
        # For positive labels, subtract margin; for negative labels, add margin
        logits_adjusted = logits - margins * (2 * targets - 1)
        
        # Scale and apply sigmoid
        logits_scaled = self.s * logits_adjusted
        p = torch.sigmoid(logits_scaled)
        
        # BCE loss on adjusted logits
        loss = -(targets * torch.log(p.clamp(min=self.eps)) + 
                 (1 - targets) * torch.log((1 - p).clamp(min=self.eps)))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class BalancedSoftmaxLoss(nn.Module):
    """
    Balanced Softmax Loss for long-tailed classification.
    
    Reference: Ren et al., "Balanced Softmax Loss for Long-Tailed Visual Classification" (CVPR 2020)
    https://arxiv.org/abs/1905.12788
    
    Re-weights samples based on their class frequency by adjusting logits.
    For multi-label: applies per-class reweighting.
    """
    def __init__(self, class_counts, reduction='mean', eps=1e-8):
        """
        Args:
            class_counts: (C,) tensor with number of positive samples per class
            reduction: 'mean' or 'sum'
            eps: small value to avoid log(0)
        """
        super(BalancedSoftmaxLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        
        # Compute per-class weights
        n_i = class_counts.float()
        N = n_i.sum()
        
        # Log of class frequencies (for balanced adjustment)
        self.register_buffer('class_log_weights', torch.log(N / (n_i + 1)))

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) unnormalized predictions
            targets: (B, C) binary targets
        
        Returns:
            loss scalar
        """
        # Move class_log_weights to same device as logits
        class_log_weights = self.class_log_weights.to(logits.device)
        
        # Adjust logits with class frequency information
        logits_adjusted = logits + class_log_weights.unsqueeze(0)
        
        p = torch.sigmoid(logits_adjusted)
        
        # BCE loss
        loss = -(targets * torch.log(p.clamp(min=self.eps)) + 
                 (1 - targets) * torch.log((1 - p).clamp(min=self.eps)))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class EqualizationLoss(nn.Module):
    """
    Equalization Loss for multi-label learning.
    
    Reference: Tan et al., "Equalization Loss for Long-Tailed Object Detection" (CVPR 2021)
    https://arxiv.org/abs/2003.05397
    
    Uses exponential moving average to maintain class-balanced gradient contributions.
    Prevents head classes from dominating training.
    """
    def __init__(self, class_counts, gamma=2.0, lam=0.1, reduction='mean', eps=1e-8):
        """
        Args:
            class_counts: (C,) tensor with number of positive samples per class
            gamma: focusing parameter (default: 2.0)
            lam: lambda for EMA of class weights (default: 0.1)
            reduction: 'mean' or 'sum'
            eps: small value to avoid log(0)
        """
        super(EqualizationLoss, self).__init__()
        self.gamma = gamma
        self.lam = lam
        self.reduction = reduction
        self.eps = eps
        
        # Initialize class weights inversely proportional to frequency
        num_classes = len(class_counts)
        max_count = class_counts.max()
        inv_weights = max_count / (class_counts.float() + 1)
        inv_weights = inv_weights / inv_weights.sum() * num_classes
        self.register_buffer('class_weights', inv_weights)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) unnormalized predictions
            targets: (B, C) binary targets
        
        Returns:
            loss scalar
        """
        p = torch.sigmoid(logits)
        
        # Focal loss term
        p_t = torch.where(targets == 1, p, 1 - p)
        focal_weight = (1 - p_t) ** self.gamma
        
        # BCE
        ce = -(targets * torch.log(p.clamp(min=self.eps)) + 
               (1 - targets) * torch.log((1 - p).clamp(min=self.eps)))
        
        # Move class_weights to same device as logits
        class_weights = self.class_weights.to(logits.device)
        
        # Apply class weights with equalization
        loss = focal_weight * ce * class_weights.unsqueeze(0)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss