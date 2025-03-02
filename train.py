import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import ResNet


def train(model, trainloader, loss_func, optimizer, device):
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
    print(f'TRAIN:  Loss: {train_loss:.4f}  Accuracy: {accuracy:.2f}%')


def test(model, testloader, loss_func, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_func(outputs, labels)

            test_loss += loss.item()

            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    test_loss /= len(testloader)
    accuracy = 100 * correct / total
    print(f'TEST:   Loss: {test_loss:.4f}  Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define data preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    # Normalize using CIFAR-10 mean and std
    ])

    # Load CIFAR-10 training dataset
    print('Loading training data...')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
    print(f'Loaded {len(trainset)} training images')

    # Load CIFAR-10 test dataset
    print('Loading test data...')
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    print(f'Loaded {len(testset)} test images')

    # Initialize model
    print('Initializing model...')
    model = ResNet().to(device)
    print(f'Model initialized on {device}')

    # Define loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Train model for multiple epochs
    print('Training model...')
    epochs = 1
    for i in range(epochs):
        print(f'Epoch: {i+1}/{epochs}')
        train(model, trainloader, loss_func, optimizer, device)
        test(model, testloader, loss_func, device)
    print('Training complete')