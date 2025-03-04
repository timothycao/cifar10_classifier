import torch
import torch.nn as nn


# Residual Block for ResNet18 and ResNet34
class ResBlock(nn.Module):
    expansion = 1   # No expansion (used for deeper ResNets)

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()

        # First convolutional layer (3x3 kernel, batch norm, ReLU)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Second convolutional layer (3x3 kernel, batch norm)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Identity shortcut by default (if dimensions don't change)
        self.shortcut = nn.Sequential()

        # Feature map dimensions = (Channels x Spatial Size)
        # If dimensions change due to:
        # 1. Spatial size decreasing aka downsampling (i.e. 32x32 -> 16x16 when stride=2)
        # 2. Channels increasing to compensate for lost spatial details (i.e. 64 -> 128)
        if stride != 1 or in_channels != out_channels:
            # Use a projection shortcut (Option B from the paper)
            self.shortcut = nn.Sequential(
                # Where we apply 1x1 convolution to match new channel and spatial dimensions
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)    # Normalize after projection
            )
            # This ensure that shortcut can be added to residual block output (second conv layer pre-activation)
    
    def forward(self, x):
        identity = x                    # Save original input for shortcut
        x = self.conv1(x)               # First conv layer + batch norm + ReLU
        x = self.conv2(x)               # Second conv layer + batch norm (no activation yet)
        x += self.shortcut(identity)    # Add the shortcut (skip connection)
        x = nn.ReLU(inplace=True)(x)    # Final activation after addition
        return x


# ResNet Model (supports custom depth and initial channels)
class ResNet(nn.Module):
    def __init__(self, block, num_blocks_per_layer, in_channels=64, name='ResNet'):
        super(ResNet, self).__init__()
        self.in_channels = in_channels  # Initial number of channels
        self.name = name
        
        # Initial convolutional layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # Create residual layers dynamically
        self.residual_layers = self._make_layers(block, num_blocks_per_layer)

        # Final classification layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # Global average pooling
            nn.Flatten(),                   # Flatten for fully connected layer
            nn.Linear(self.in_channels, 10) # Fully connected layer (CIFAR-10 has 10 classes)
        )

    # Creates a single residual layer with multiple blocks
    def _make_layer(self, block, out_channels, num_blocks, stride):
        blocks = []
        for i in range(num_blocks):
            # First block may downsample, subsequent blocks have stride=1
            blocks.append(block(self.in_channels, out_channels, stride if i == 0 else 1))
            self.in_channels = out_channels # Update in_channels for next block

        return nn.Sequential(*blocks)
    
    # Creates all residual layers (4 total for standard ResNets)
    def _make_layers(self, block, num_blocks_per_layer):
        layers = []
        out_channels = self.in_channels
        for i, num_blocks in enumerate(num_blocks_per_layer):
            layers.append(self._make_layer(block, out_channels, num_blocks, 1 if i == 0 else 2))
            out_channels *= 2   # Double output channels for each new layer

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x


# Standard ResNet models
def ResNet18(): return ResNet(ResBlock, [2, 2, 2, 2], name='ResNet18')
def ResNet34(): return ResNet(ResBlock, [3, 4, 6, 3], name='ResNet34')


# Custom ResNet models (for testing)
def ResNetCustom(): return ResNet(ResBlock, [1, 1, 1, 1], in_channels=8, name='ResNetCustom')


if __name__ == '__main__':
    # model = ResNet18()
    model = ResNetCustom()
    print('Total model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))