import torch
import torch.nn as nn   
from torchvision.models import densenet121

class DenseNet121(nn.Module):
    def __init__(self, n_classes):
        super(DenseNet121, self).__init__()

        self.densenet121 = densenet121(pretrained=True)
        n_features = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.densenet121(x)
        return x


if __name__ == "__main__":
    model = DenseNet121(n_classes=10)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(y.shape)  # Expected output: torch.Size([2, 10])