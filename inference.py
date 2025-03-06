import os
import glob
import pickle
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from model import *
from utils import get_paths


_, DATASET_PATH, SAVED_MODELS_PATH, SAVED_PREDICTIONS_PATH = get_paths()


def load_data():
    # Load inference test dataset
    print('Loading data...')
    with open(DATASET_PATH, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    test_images = data[b'data'] # Extract images

    # Reshape from (N, H, W, C) → (N, C, H, W) for PyTorch
    test_images = test_images.reshape(-1, 32, 32, 3).astype(np.uint8)  # Ensure correct shape and data type

    # Define transformation (must match training pipeline)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Apply transformation to each image individually
    test_images = torch.stack([transform(img) for img in test_images])

    return test_images


def load_model(filename, device):
    # Ensure saved_models directory exist
    os.makedirs(SAVED_MODELS_PATH, exist_ok=True)

    # Ensure file exists in saved_models directory
    if not os.path.exists(f'{SAVED_MODELS_PATH}/{filename}'):
        raise FileNotFoundError(f'{filename} not found in {SAVED_MODELS_PATH}')

    # Load model
    print(f'Loading model...')
    model = torch.load(f'{SAVED_MODELS_PATH}/{filename}', map_location=device, weights_only=False)
    model.to(device)

    return model


def save_predictions(predictions, filename):
    # Ensure saved_predictions directory exists
    os.makedirs(SAVED_PREDICTIONS_PATH, exist_ok=True)

    # Save predictions to CSV in saved_predictions directory
    filename = filename.replace('.pth', '.csv')
    df = pd.DataFrame({'ID': range(len(predictions)), 'Label': predictions})
    df.to_csv(f'{SAVED_PREDICTIONS_PATH}/{filename}', index=False)
    print(f'Predictions saved as {filename}')


def inference(model, test_images, device):
    model.eval()
    test_images = test_images.to(device)

    print('Running inference...')
    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = outputs.max(1)   # Get predicted class

    return predicted.cpu().numpy()  # Convert to NumPy array


def main(filename=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load data
    test_images = load_data()

    # Get model filename(s)
    filenames = []
    if filename:
        filenames.append(filename)
    else:
        # Find saved models
        model_paths = glob.glob(f'{SAVED_MODELS_PATH}/*.pth')
        if not model_paths:
            raise FileNotFoundError(f'No models found in {SAVED_MODELS_PATH}')
        
        for model_path in model_paths:
            filenames.append(os.path.basename(model_path))

    # Run inference and save predictions
    for filename in filenames:
        # Load model
        model = load_model(filename, device)

        # Run inference
        predictions = inference(model, test_images, device)

        # Save predictions
        save_predictions(predictions, filename)


if __name__ == '__main__':
    try:
        main()
    except (FileNotFoundError, NameError) as e:
        print(f'ERROR: {e}')