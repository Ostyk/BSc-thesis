# BSc-thesis
Repository containing code from my BSc thesis (in progress)

Transfer Learning of the skin lesion images with a SVM classifier.

# Installation

- Python 3.6.4
- Please see requirement.txt for modules

# Code structure

## Jupyter notebooks
- **AlexNet version 2.0.ipynb**
  - Extracts data from AlexNet and saves them as diffrent models
- **Classifier pipeline.ipynb**
  - output of diffrent models of both balaneced and imbalanced dataset through my (dimentionality reduction, SVM) classifier pipeline

## Python files
- **alexnet.py**
  - Tensorflow implementation of AlexNet
- **final.py**
  - functions for the classifier pipeline
- **funcs.py**
  - functions for CNN codes extraction from Alexnet
- **merge_labels.py**
  - merging labels from the training and test datasets
- **plot_cm.py**
  - plots confusion matrix
- **resize_images.py**
  - reduces image size to fit AlexNet requirements (227x227x3)
