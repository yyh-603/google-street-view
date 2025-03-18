# K-means

## Introduction

This part of the project is about clustering the images of the dataset using the K-means algorithm. The results of experiment are in the matrix part.

## Experiment

The experiment is done on the dataset of 5700 images, each class has 300 images. First, image features are extracted using PCA. Then, the K-means algorithm is applied to the features. To predict and evaluate the clustering, each cluster is assigned to the class that has the most images in that cluster.

Defalut hyperparameters:
- n_clusters = 19
- n_components = 1200

## Code usage

The code is in the `kmeans.py` file. For detailed information about the code, please refer to the comments in the code.
To modify the hyperparameters, please modify directly in the code. Currently, there is no any command line argument for the hyperparameters.