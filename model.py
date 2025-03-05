import torch.nn as nn


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, skip_kernel_size=1, stride=1):
        super(ResBlock, self).__init__()

        # First convolutional layer (kernel, batch norm, ReLU)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Second convolutional layer (kernel, batch norm)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # Identity shortcut by default
        self.shortcut = nn.Sequential()

        # Feature map dimensions = (Channels x Spatial Size)
        # If dimensions change due to:
        # 1. Spatial size decreasing aka downsampling (i.e. 32x32 -> 16x16 when stride=2)
        # 2. Channels increasing to compensate for lost spatial details (i.e. 64 -> 128)
        if stride != 1 or in_channels != out_channels:
            # Use a projection shortcut (Option B from the paper)
            self.shortcut = nn.Sequential(
                # Where we apply convolution to match new channel and spatial dimensions
                nn.Conv2d(in_channels, out_channels, kernel_size=skip_kernel_size, stride=stride, padding=skip_kernel_size//2, bias=False),
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


# ResNet Model
class ResNet(nn.Module):
    def __init__(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size=1, name=None):
        super(ResNet, self).__init__()
        self.name = name
        self.in_channels = channels_per_layer[0]    # Initial number of channels
        
        # Ensure per-layer parameters have the same length
        assert len(blocks_per_layer) == len(channels_per_layer) == len(kernels_per_layer) == len(skip_kernels_per_layer), \
            'All per-layer parameters (blocks, channels, kernels, skip_kernels) must have the same length'

        # Initial convolutional layer
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, self.in_channels, kernel_size=kernels_per_layer[0], stride=1, padding=kernels_per_layer[0]//2, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True)
        )

        # Create residual layers dynamically
        self.residual_layers = self._make_layers(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer)

        # Final classification layer
        self.output_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(pool_size),
            nn.Flatten(),
            nn.Linear(channels_per_layer[-1], 10)   # CIFAR-10 has 10 classes
        )

    # Creates a single residual layer with multiple blocks
    def _make_layer(self, out_channels, num_blocks, kernel_size, skip_kernel_size, stride):
        blocks = []
        for i in range(num_blocks):
            # First block may downsample, subsequent blocks have stride=1
            blocks.append(ResBlock(self.in_channels, out_channels, kernel_size, skip_kernel_size, stride if i == 0 else 1))
            self.in_channels = out_channels # Update in_channels for next block

        return nn.Sequential(*blocks)
    
    # Creates all residual layers
    def _make_layers(self, blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer):
        layers = []
        for i in range(len(blocks_per_layer)):
            layers.append(self._make_layer(
                out_channels=channels_per_layer[i],
                num_blocks=blocks_per_layer[i],
                kernel_size=kernels_per_layer[i],
                skip_kernel_size=skip_kernels_per_layer[i],
                stride = 1 if i == 0 else 2
            ))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_layers(x)
        x = self.output_layer(x)
        return x


def create_model(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, name):
    return ResNet(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, name)


if __name__ == '__main__':
    try:
        model = create_model(
            blocks_per_layer=[2, 2, 2, 2],
            channels_per_layer=[64, 128, 256, 512],
            kernels_per_layer=[3, 3, 3, 3],
            skip_kernels_per_layer=[1, 1, 1, 1],
            pool_size=1,
            name='ResNet18'
        )
        print('Total model parameters:', sum(p.numel() for p in model.parameters() if p.requires_grad))
    except AssertionError as e:
        print(f'ERROR: {e}')