import torch
from src.loss import AsymmetricLoss

loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
x = torch.tensor([[0.2, -1.0, 2.0]])
y = torch.tensor([[1., 0., 1.]])

print(loss_fn(x, y))  # ASL
bce = torch.nn.BCEWithLogitsLoss(reduction='mean')(x, y)
print(bce)
