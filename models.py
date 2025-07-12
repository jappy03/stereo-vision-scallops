import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvNeXtBlock(nn.Module):
    """
    Blocco base dell'architettura ConvNeXt modificato per immagini grandi.
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.BatchNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + self.drop_path(x)
        return x

class EnhancedClassifier(nn.Module):
    """
    Classificatore ottimizzato per immagini 1600x1200.
    """
    def __init__(self, num_classes=2, input_channels=3):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            # Stem - riduce la dimensione dell'immagine
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Stage 1
            self._make_stage(64, 128, 2),

            # Stage 2
            self._make_stage(128, 256, 2),

            # Stage 3
            self._make_stage(256, 512, 2),

            # Global pooling
            nn.AdaptiveAvgPool2d(1)
        )

        # Classificatore
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        ]
        for _ in range(num_blocks):
            layers.append(ConvNeXtBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class DropPath(nn.Module):
    """
    Drop Path per regolarizzazione.
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output