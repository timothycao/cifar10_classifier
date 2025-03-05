from model import create_model
from train import main as train
from inference import main as inference


# Model parameters
MODEL_NAME = 'ResNetCustom'
NUM_BLOCKS_PER_LAYER = [1, 1, 1, 1]
NUM_CHANNELS_PER_LAYER = [8, 16, 32, 64]
KERNEL_SIZE_PER_LAYER = [3, 3, 3, 3]
SKIP_KERNEL_SIZE_PER_LAYER = [1, 1, 1, 1]
POOL_SIZE = 1


# Training parameters
EPOCHS = 3
SAVE_MODE = 'best'  # Options: 'best', 'every'
SAVE_EVERY_N = 1


if __name__ == '__main__':
    # Train the model
    try:
        model = create_model(
            blocks_per_layer=NUM_BLOCKS_PER_LAYER,
            channels_per_layer=NUM_CHANNELS_PER_LAYER,
            kernels_per_layer=KERNEL_SIZE_PER_LAYER,
            skip_kernels_per_layer=SKIP_KERNEL_SIZE_PER_LAYER,
            pool_size=POOL_SIZE,
            name=MODEL_NAME
        )
        train(model, EPOCHS, SAVE_MODE, SAVE_EVERY_N)
    except (ValueError, NameError, AssertionError) as e:
        print(f'ERROR: {e}')
    
    # Run inference
    try:
        inference()
    except (FileNotFoundError, NameError) as e:
        print(f'ERROR: {e}')