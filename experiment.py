from model import create_model
from train import main as train
from inference import main as inference


# Model parameters
MODEL_NAME = 'ResNetCustom'
NUM_BLOCKS_PER_LAYER = [1, 1, 1, 1]
IN_CHANNELS = 8


# Training parameters
EPOCHS = 3
SAVE_MODE = 'best'  # Options: 'best', 'every'
SAVE_EVERY_N = 1


if __name__ == '__main__':
    model = create_model(NUM_BLOCKS_PER_LAYER, IN_CHANNELS, MODEL_NAME)
    
    # Train the model
    try:
        train(model, EPOCHS, SAVE_MODE, SAVE_EVERY_N)
    except (ValueError, NameError) as e:
        print(f'ERROR: {e}')
    
    # Run inference
    try:
        inference()
    except (FileNotFoundError, NameError) as e:
        print(f'ERROR: {e}')