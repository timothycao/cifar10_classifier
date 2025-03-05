import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from model import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(train_batch_size=128, test_batch_size=100, augment=False):
    # Define data preprocessing transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    # Normalize using CIFAR-10 mean and std
    ])
    
    # Data augmentation
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # Random horizontal flip
        transforms.RandomCrop(32, padding=4),   # Random cropping
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))    # Normalize using CIFAR-10 mean and std
    ]) if augment else transform

    print('Loading data...')

    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False, num_workers=2)

    return trainloader, testloader


def save_model(model, epoch, accuracy, save='every', every_n=1):
    # Ensure saved_models directory exists
    os.makedirs('saved_models', exist_ok=True)
    
    # Extract model name
    model_name = getattr(model, 'name')

    # Determine filename based on save type
    filename = None
    if save == 'every' and epoch % every_n == 0:
        filename = f'{model_name}_epoch{epoch}_acc{round(accuracy)}.pth'

    # Save model in saved_models directory
    if filename:
        torch.save(model, f'saved_models/{filename}')
        print(f'Model saved as {filename}')


def train(model, trainloader, loss_func, optimizer, device):
    model.train()
    train_loss, correct, total = 0, 0, 0

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

    # Return accuracy to track best model
    return accuracy


def test(model, testloader, loss_func, device):
    model.eval()
    test_loss, correct, total = 0, 0, 0

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

    return accuracy


def main(model, epochs, train_batch_size=128, test_batch_size=100, augment=False,
         optimizer=None, scheduler=None, save='best', every_n=1):
    # Count model parameters
    total_params = count_parameters(model)
    print(f'Total model parameters: {total_params}')
    if total_params > 5_000_000:
        raise ValueError('Model cannot have more than 5 million parameters')

    # Ensure save option is valid
    save_options = ['best', 'every']
    if save not in save_options:
        raise ValueError(f'Save option must be one of: {save_options}')

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    trainloader, testloader = load_data(train_batch_size, test_batch_size, augment)

    # Initialize model
    print('Initializing model...')
    model = model.to(device)

    # Define loss function
    loss_func = nn.CrossEntropyLoss()
    
    # Ensure optimizer is valid
    if optimizer and not isinstance(optimizer, optim.Optimizer):
        raise TypeError('Optimizer must be an instance of torch.optim.Optimizer')

    # Define optimizer
    optimizer = optimizer if optimizer else optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Ensure scheduler is valid
    if scheduler and not isinstance(scheduler, optim.lr_scheduler.LRScheduler):
        raise TypeError('Scheduler must be an instance of torch.optim.lr_scheduler.LRScheduler')

    # Train model for multiple epochs
    print('Training model...')
    best_accuracy, best_epoch = 0.0, 0
    for epoch in range(1, epochs + 1):
        print(f'Epoch: {epoch}/{epochs}')
        train_accuracy = train(model, trainloader, loss_func, optimizer, device)
        test_accuracy = test(model, testloader, loss_func, device)

        # Track best model for 'best' save type
        if test_accuracy > best_accuracy:
            best_accuracy, best_epoch = test_accuracy, epoch

        # Save model based on save type (not 'best')
        save_model(model, epoch, test_accuracy, save, every_n)

        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_accuracy)
            else:
                scheduler.step()
    
    if save == 'best':
        # Save best model using default save='every' and every_n=1
        save_model(model, best_epoch, best_accuracy)

    print('Training complete')


if __name__ == '__main__':
    try:
        model = create_model(
            blocks_per_layer=[1, 1, 1, 1],
            channels_per_layer=[8, 16, 32, 64],
            kernels_per_layer=[3, 3, 3, 3],
            skip_kernels_per_layer=[1, 1, 1, 1],
            pool_size=1,
            name='ResNetCustom'
        )

        epochs = 3
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

        main(model, epochs, augment=True, optimizer=optimizer, scheduler=scheduler)
        # main(model, epochs, save='best')    # Save best model only
        # main(model, epochs, save='every')   # Save every epoch
        # main(model, epochs, save='every', every_n=epochs)   # Save every n epochs
    except (AssertionError, ValueError, TypeError) as e:
        print(f'ERROR: {e}')