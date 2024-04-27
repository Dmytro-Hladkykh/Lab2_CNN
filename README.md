# CIFAR-10 Image Classification with CNN

This project demonstrates how to train a Convolutional Neural Network (CNN) on the CIFAR-10 dataset for image classification.

## Overview

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The classes are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

The project includes the following components:

1. `data.py`: Module for loading and preprocessing the CIFAR-10 dataset.
2. `model.py`: Module defining the CNN architecture using PyTorch.
3. `train.py`: Module for training the CNN model.
4. `main.py`: Main script for running the training process.
5. `utils.py`": Module for logging.

This repository contains code for a basic training pipeline for a Convolutional Neural Network (CNN) model using PyTorch. The pipeline includes data preprocessing, model training, evaluation, and artifact collection.

### Requirements

- Python 3.x
- PyTorch
- torchvision
- matplotlib
- Pillow
- tqdm


## Getting Started

### Installation

1. Clone the repository:

    ```
    git clone https://github.com/Dmytro-Hladkykh/Lab1_CNN
    ```

2. Navigate to the project directory:

    ```
    cd Lab1_CNN
    ```

3. Install the required dependencies:

    ```
    pip install -r requirements.txt
    ```

### Usage

1. Run the training script:

    ```
    python main.py
    ```

    This will train the CNN model on the CIFAR-10 dataset using default hyperparameters.

2. (Optional) To customize the training process, you can modify the hyperparameters in `main.py` or pass them as command-line arguments.

3. After the model is trained you can modify `config.py` with `elf.train_new_model = False` to use trained model.

## Example

To demonstrate the trained model's performance, you can use the `main.py` script to train the model and then evaluate it on test data. Additionally, you can provide an example image to see how the model classifies it.


