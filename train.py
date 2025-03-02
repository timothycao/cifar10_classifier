import torch
import torchvision
import torchvision.transforms as transforms
from model import ResNet


def train(model, trainloader, device):
    # Get one sample batch
    images, labels = next(iter(trainloader))
    images, labels = images.to(device), labels.to(device)

    # Print input shape (batch size, channels, height, width)
    print(f'Input shape: {images.shape}')   # Expected: [128, 3, 32, 32]

    # Pass images through the model
    outputs = model(images)

    # Print output shape (batch size, number of classes)
    print(f'Output shape: {outputs.shape}') # Expected: [128, 10]


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define data preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    # Normalize using CIFAR-10 mean and std
    ])

    # Load CIFAR-10 training dataset
    print('Loading data...')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    
    # Wrap dataset in DataLoader for batching
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    print(f'Loaded {len(trainset)} training images')

    # Initialize model
    print('Initializing model...')
    model = ResNet().to(device)
    print(f'Model initialized on {device}')

    # Test model with a single forward pass
    train(model, trainloader, device)