#Helper functions (e.g., for loading the data, small repetitive functions)

# Import required libraries
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf 
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load in CIFAR-100 Dataset
(train_images, train_labels), (test_images, test_labels) = datasets.cifar100.load_data()

# Normalize pixel values to be between 0 and 1
# Divides the images by 255 to normalize our pixel intensities to be between 0 and 1 -> Makes it much easier for the NN to train and perform
train_images, test_images = train_images / 255.0, test_images / 255.0

# Get image shape, needs to be 32x32x3 (32x32 and 3 color panes)
image_shape = train_images.shape[1:]
