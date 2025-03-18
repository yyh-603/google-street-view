# K-means

## Introduction

This part of the project is about clustering the images of the dataset using the K-means algorithm. The results of experiment are in the matrix part.

## Experiment

The experiment is done on the dataset of 5700 images, each class has 300 images. First, image features are extracted using PCA. Then, SVM model is applied to the features. Randomized search is used to find the best hyperparameters for the SVM model. The best hyperparameters are used to test the model on the test set.

Defalut hyperparameters:
- n_components = 2000
- C = loguniform(1, 10)
- gamma = loguniform(1e-9, 1e-8)
- cv = 5
- n_iter = 20

## Code usage

The code is in the `svm.py` file. For detailed information about the code, please refer to the comments in the code.
To modify the hyperparameters, please modify directly in the code. Currently, there is no any command line argument for the hyperparameters.