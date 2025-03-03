import os
import glob
import torch
import torchvision.transforms as transforms
import pickle
import pandas as pd
from model import ResNet


def inference(model, test_images, device):
    model.eval()
    test_images = test_images.to(device)

    with torch.no_grad():
        outputs = model(test_images)
        _, predicted = outputs.max(1)   # Get predicted class

    return predicted.cpu().numpy()  # Convert to NumPy array


if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load inference test dataset
    print('Loading test data...')
    with open('cifar_test_nolabel.pkl', 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    test_images = torch.tensor(data[b'data']).float()   # Convert to float tensor
    print(f'Loaded {len(test_images)} test images')

    # Preprocess images
    test_images = test_images.reshape(-1, 3, 32, 32) / 255.0    # Reshape to (batch size, channels, height, width) and scale to [0,1]
    test_images = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(test_images)
    print('Test images preprocessed')

    # Ensure required directories exist
    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('saved_predictions', exist_ok=True)

    # Find saved models
    model_paths = glob.glob('saved_models/*.pth')
    if not model_paths:
        raise FileNotFoundError('No models found in saved_models/')

    for model_path in model_paths:
        # Load model
        print(f'Loading model...')
        model = ResNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'Model {model_path} loaded')

        # Run inference
        print('Running inference...')
        predictions = inference(model, test_images, device)
        print('Inference complete')

        # Save predictions to CSV inside saved_predictions/
        filename = f"saved_predictions/{os.path.basename(model_path).replace('.pth', '.csv')}"
        df = pd.DataFrame({'Id': range(len(predictions)), 'Label': predictions})
        df.to_csv(filename, index=False)
        print(f'Predictions saved as {filename}')