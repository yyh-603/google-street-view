# ResNet

## Introduction

This part of the project is about training ResNet. The results of experiment are in the matrix part.

## Experiment

The experiment is done on the dataset of 5700 images, each class has 300 images. First, image transform by resizing and normalizing. Then, the ResNet is trained on the dataset.

Defalut hyperparameters:
- model = ResNet50-pretrained
- optimizer = Adam
- loss function = CrossEntropyLoss
- batch_size = 32
- num_epochs = 15
- learning_rate = 0.00003

## Code usage

The code is in the `resnet.py` file. For detailed information about the code, please refer to the comments in the code.

To modify the hyperparameters, please modify directly in the code. Currently, there is no any command line argument for the hyperparameters.

The code to split rearrange the dataset is in the `split_dataset.py` file. For detailed information about the code, please refer to the comments in the code.