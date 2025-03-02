import torch
import torch.nn as nn


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        # Initial convolutional layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Placeholder for residual block layers
        self.placeholder = nn.Identity()

        # Final classification layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),    # Global average pooling
            nn.Flatten(),               # Flatten for fully connected layer
            nn.Linear(64, 10)           # Fully connected layer (CIFAR-10 has 10 classes)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.placeholder(x)
        x = self.output_layer(x)
        return x


if __name__ == '__main__':
    model = ResNet()
    print('Model:', model)