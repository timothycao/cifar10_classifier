import os
import glob
import torch
import torchvision.transforms as transforms
import pickle
import pandas as pd
from model import ResNet


def load_data():
    # Load inference test dataset
    print('Loading data...')
    with open('cifar_test_nolabel.pkl', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    test_images = torch.tensor(data[b'data']).float()   # Convert to float tensor
    # print(f'Loaded {len(test_images)} test images')

    # Preprocess images
    test_images = test_images.reshape(-1, 3, 32, 32) / 255.0    # Reshape to (batch size, channels, height, width) and scale to [0,1]
    test_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(test_images)
    # print('Test images preprocessed')

    return test_images


def load_model(filename, device):
    # Ensure saved_models directory exist
    os.makedirs('saved_models', exist_ok=True)

    # Load model
    print(f'Loading model...')
    model = ResNet().to(device)
    model.load_state_dict(torch.load(f'saved_models/{filename}', map_location=device))
    # print(f'Model {filename} loaded')

    return model


def save_predictions(predictions, filename):
    # Ensure saved_predictions directory exists
    os.makedirs('saved_predictions', exist_ok=True)

    # Save predictions to CSV in saved_predictions directory
    filename = filename.replace('.pth', '.csv')
    df = pd.DataFrame({'Id': range(len(predictions)), 'Label': predictions})
    df.to_csv(f'saved_predictions/{filename}', index=False)
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
        model_paths = glob.glob('saved_models/*.pth')
        if not model_paths:
            raise FileNotFoundError('No models found in saved_models/')
        
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
    main()