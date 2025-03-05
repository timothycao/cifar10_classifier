import torch.optim as optim
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
TRAIN_BATCH_SIZE = 128
TEST_BATCH_SIZE = 100
AUGMENT = True
SAVE_MODE = 'best'  # Options: 'best', 'every'
SAVE_EVERY_N = 1


if __name__ == '__main__':
    # Initialize model
    try:
        model = create_model(
            blocks_per_layer=NUM_BLOCKS_PER_LAYER,
            channels_per_layer=NUM_CHANNELS_PER_LAYER,
            kernels_per_layer=KERNEL_SIZE_PER_LAYER,
            skip_kernels_per_layer=SKIP_KERNEL_SIZE_PER_LAYER,
            pool_size=POOL_SIZE,
            name=MODEL_NAME
        )
    except AssertionError as e:
        print(f'Failed to create model: {e}')
        exit(1)


    # Train model
    try:
        # Define optimizer
        # OPTIMIZER = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        OPTIMIZER = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        # OPTIMIZER = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-8, weight_decay=1e-4)
        
        # Define scheduler
        # SCHEDULER = optim.lr_scheduler.StepLR(OPTIMIZER, step_size=10, gamma=0.1)
        # SCHEDULER = optim.lr_scheduler.MultiStepLR(OPTIMIZER, milestones=[30, 60, 90], gamma=0.1)
        # SCHEDULER = optim.lr_scheduler.ExponentialLR(OPTIMIZER, gamma=0.95)
        # SCHEDULER = optim.lr_scheduler.CosineAnnealingLR(OPTIMIZER, T_max=EPOCHS)
        SCHEDULER = optim.lr_scheduler.ReduceLROnPlateau(OPTIMIZER, mode='max', factor=0.1, patience=5)
        
        # If no scheduler, set to None
        SCHEDULER = SCHEDULER if 'SCHEDULER' in locals() else None

        # train(model, EPOCHS)
        train(model, EPOCHS, train_batch_size=TRAIN_BATCH_SIZE, test_batch_size=TEST_BATCH_SIZE, augment=AUGMENT,
              optimizer=OPTIMIZER, scheduler=SCHEDULER, save=SAVE_MODE, every_n=SAVE_EVERY_N)
    except (ValueError, TypeError) as e:
        print(f'Training failed: {e}')
        exit(1)


    # Run inference
    try:
        inference()
    except FileNotFoundError as e:
        print(f'Inference failed: {e}')