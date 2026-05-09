# PlantVillage Disease Classification with PyTorch

A deep learning image classification project built with **PyTorch** to detect plant diseases from the **PlantVillage** dataset. The project compares a custom convolutional neural network with a transfer-learning approach using **ResNet50**, including image preprocessing, data augmentation, model training, evaluation, and prediction utilities.

---

## Project Overview

This repository contains a complete plant disease classification workflow:

- Load image datasets using `torchvision.datasets.ImageFolder`
- Prepare train and validation/test dataloaders
- Apply image transformations and augmentation
- Train a custom convolutional neural network
- Fine-tune a pretrained ResNet50 model
- Track training and validation loss/accuracy
- Compare model performance visually
- Run final predictions on the test dataset

The main notebook, `PlantVillage.ipynb`, walks through the full experiment from data loading to model comparison.

---

## Dataset Structure

The project expects the dataset to be organised in an `ImageFolder` format:

```text
PlantVillage/
├── train/
│   ├── class_1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── class_2/
│   └── ...
│
└── val/
    ├── class_1/
    ├── class_2/
    └── ...
```

Each folder name represents one plant disease class.

---

## Repository Files

```text
.
├── PlantVillage.ipynb      # Main notebook for exploration, training, and evaluation
├── data_setup.py           # Creates train and test dataloaders
├── train.py                # Training and validation loop utilities
├── prediction.py           # Final model evaluation/prediction utility
└── README.md               # Project documentation
```

---

## Model Approaches

### 1. Custom CNN Model

A simple convolutional neural network was created using:

- Three convolutional blocks
- ReLU activation
- Max pooling
- Flatten layer
- Fully connected output layer
- Dropout for regularisation

This model was used as a baseline to understand how a custom architecture performs on the dataset.

### 2. ResNet50 Transfer Learning

A pretrained ResNet50 model was also used with transfer learning:

- Loaded pretrained ImageNet weights
- Frozen the feature extraction layers
- Replaced the final fully connected layer
- Adapted the output layer to match the number of PlantVillage classes

This approach is generally stronger because ResNet50 already learned useful visual features from a large dataset.

---

## Data Preprocessing and Augmentation

The project uses separate transforms for training and validation/testing.

### Training Transform

```python
transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

### Test/Validation Transform

```python
transforms.Compose([
    transforms.Resize(232),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
```

Augmentation helps the model generalise better by exposing it to slightly different versions of the same images.

---

## Training Pipeline

The training process is handled through reusable functions in `train.py`:

- `train_step()` handles one training epoch
- `test_step()` evaluates the model on validation/test data
- `train()` runs the full training loop for multiple epochs

The training function stores:

- Training loss
- Training accuracy
- Test/validation loss
- Test/validation accuracy

These values are returned in a dictionary and can be converted into a pandas DataFrame for comparison and plotting.

---

## Data Loader Utility

The `data_setup.py` file contains a helper function for creating dataloaders:

```python
def data_setup(train_dir, test_dir, transform, test_transform, batch_size):
    ...
```

It returns:

```python
train_dataloader, test_dataloader
```

This keeps the notebook cleaner and makes the project easier to reuse.

---

## Prediction / Evaluation Utility

The `prediction.py` file contains a helper function that evaluates a trained model on the test dataloader:

```python
def prediction(model, test_dataloader, loss_fn, accuracy, device):
    ...
```

It returns:

```python
test_acc, test_loss
```

---

## Requirements

Install the main dependencies:

```bash
pip install torch torchvision torchmetrics torchinfo tqdm matplotlib pandas pillow numpy
```

Recommended environment:

- Python 3.10+
- PyTorch
- CUDA-supported GPU if available

The project automatically uses GPU when available:

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## How to Run

1. Clone the repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Add the PlantVillage dataset using the expected folder structure:

```text
PlantVillage/train
PlantVillage/val
```

3. Install dependencies:

```bash
pip install torch torchvision torchmetrics torchinfo tqdm matplotlib pandas pillow numpy
```

4. Open the notebook:

```bash
jupyter notebook PlantVillage.ipynb
```

5. Run the notebook cells from top to bottom.

---

## Results and Comparison

The notebook compares three experiments:

1. Custom CNN model
2. Default ResNet50 transfer learning model
3. Augmented ResNet50 transfer learning model

The results are stored and visualised using accuracy and loss curves, making it easier to compare how each model performs during training and validation.

---

## Key Learnings

Through this project, I practised:

- Working with image datasets in PyTorch
- Building reusable training utilities
- Creating custom CNN architectures
- Using transfer learning with ResNet50
- Applying image augmentation
- Evaluating model performance
- Visualising loss and accuracy curves
- Structuring a deep learning project into multiple files

---

## Future Improvements

Possible improvements for this project:

- Train for more epochs
- Add a separate unseen test set
- Save and load the best model checkpoint
- Add a confusion matrix
- Add per-class precision, recall, and F1-score
- Build a simple web app for uploading leaf images
- Experiment with EfficientNet or Vision Transformers
- Improve class imbalance handling if needed

---

## Author

Created as a PyTorch computer vision project for plant disease classification using the PlantVillage dataset.
