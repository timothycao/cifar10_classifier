import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ResNet


def train(model, trainloader, loss_func, optimizer, device, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(trainloader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()   # Reset gradients

        outputs = model(images)             # Forward pass
        loss = loss_func(outputs, labels)   # Compute loss

        loss.backward()     # Backward pass
        optimizer.step()    # Update weights

        train_loss += loss.item()   # Track total loss

        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    # Compute average loss and accuracy for the epoch
    train_loss /= len(trainloader)
    accuracy = 100 * correct / total
    print(f'Loss: {train_loss:.4f}  Accuracy: {accuracy:.2f}%')


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

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Train model for multiple epochs
    print('Training model...')
    epochs = 3
    for i in range(epochs):
        print(f'Epoch: {i+1}')
        train(model, trainloader, loss_func, optimizer, device, epoch=i)
    print('Training complete')