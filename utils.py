import os


def is_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ


def get_paths():
    if is_kaggle():
        traintest_dataset_path = '/kaggle/input/deep-learning-spring-2025-project-1/cifar-10-python'
        inference_dataset_path = '/kaggle/input/deep-learning-spring-2025-project-1/cifar_test_nolabel.pkl'
        saved_models_path = '/kaggle/working/saved_models'
        saved_predictions_path = '/kaggle/working/saved_predictions'
    else:
        traintest_dataset_path = 'data'
        inference_dataset_path = 'cifar_test_nolabel.pkl'
        saved_models_path = 'saved_models'
        saved_predictions_path = 'saved_predictions'
    
    return traintest_dataset_path, inference_dataset_path, saved_models_path, saved_predictions_path