# **CIFAR-10 Classifier**

## **Setup Instructions**

### **Running Locally**
#### **1. Clone the repository**  
```sh
git clone https://github.com/timothycao/cifar10_classifier.git
cd cifar10_classifier
```

#### **2. Create and activate a virtual environment**  
```sh
python3 -m venv .venv
source .venv/bin/activate   # Mac/Linux
```

#### **3. Install dependencies**  
```sh
pip install -r requirements.txt
```

#### **4. Download dataset manually or via Kaggle CLI**  
```sh
kaggle competitions download -c deep-learning-spring-2025-project-1
```
Place `cifar_test_nolabel.pkl` in the **project root directory**.

#

### **Running on Kaggle**
#### **1. Clone repository in the notebook**  
```sh
!git clone https://github.com/timothycao/cifar10_classifier.git
%cd cifar10_classifier
```

#### **2. Add dataset via the console**  
Go to **Add Input** → **Search** "Deep Learning Spring 2025: CIFAR 10 classification"

#

## **Training and Inference**

After setup, you can either:

### **Option 1: Use `example.py`**
Modify model and training parameters in `example.py` and run:
```sh
python3 example.py
```

### **Option 2: Use your own script/notebook**

#### **1. Import functions**  
```python
import torch.optim as optim
import torchvision.transforms as transforms
from model import create_model
from train import main as train
from inference import main as inference
```

#### **2. Use these functions to train and run inference**  
See below for function details and `example.py` for usage examples.

#

## **Function Overview**

#### `create_model(blocks_per_layer, channels_per_layer, kernels_per_layer, skip_kernels_per_layer, pool_size, name)`  
Creates a ResNet model with configurable architecture.

- `blocks_per_layer (list[int])`: Number of residual blocks per layer.
- `channels_per_layer (list[int])`: Number of channels per layer.
- `kernels_per_layer (list[int])`: Kernel size per layer.
- `skip_kernels_per_layer (list[int])`: Skip connection kernel size per layer.
- `pool_size (int)`: Average pooling kernel size.
- `name (str)`: Name of the model (used in saved model and prediction filename).

#### `train(model, epochs, train_batch_size, test_batch_size, augmentations, optimizer, scheduler, save, every_n)`  
Trains the model with tunable parameters and saves in `saved_models/`.

- `model (nn.Module)`: ResNet model to train (must be created using `create_model`).
- `epochs (int)`: Number of training epochs.
- `train_batch_size (int, optional)`: Training batch size. Default: `128`.
- `test_batch_size (int, optional)`: Testing batch size. Default: `100`.
- `augmentations (list[callable], optional)`: List of callable torchvision.transforms functions for data augmentation. Default: `None`.
- `optimizer (torch.optim.Optimizer, optional)`: Training optimizer (must be initialized using torch.optim). Default: `SGD(lr=0.01, momentum=0.9, weight_decay=5e-4)`.
- `scheduler (torch.optim.lr_scheduler.LRScheduler, optional)`: Learning rate scheduler (must be initialized using torch.optim.lr_scheduler). Default: `None`.
- `save (str, optional)`: Model saving method: `'best'` (only best model) or `'every'` (every `every_n` epochs). Default: `'best'`.
- `every_n (int, optional)`: Save frequency when `save='every'`. Default: `1`.

#### `inference(filename)`  
Runs inference on saved models and stores predictions in `saved_predictions/`.

- `filename (str, optional)`: If provided, runs inference on a specific saved model file. If `None`, runs inference on all saved models. Default: `None`.
