import numpy as np
import os
import sys
import warnings
import scipy.spatial
import tqdm
import scipy

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

# Compare the predicted labels with the respective target labels. Return an array where 0 indicates the label was wrong, and 1 indicates the label was right.
def get_label_correctness(labels, targets):
    return (np.argmax(labels, axis = 1) == np.argmax(targets, axis = 1))

# Creates 2d histogram-like array to compare predicted labels with each other and with targets.
def confusion_matrix(labels1, labels2):
    assert labels1.ndim == 2, f"Confusion Matrix: labels1 must be 2D array. Received labels1 of shape {labels1.shape}."
    assert labels2.ndim == 2, f"Confusion Matrix: labels2 must be 2D array. Received labels2 of shape {labels2.shape}."
    
    matrix = np.zeros((labels1.shape[1], labels2.shape[1]))

    for class1, class2 in zip(np.argmax(labels1, axis = 1), np.argmax(labels2, axis = 1) ):
        matrix[class1, class2] += 1

    return matrix

# Jensen Shannon Distance between probability Distributions
def JSD(labels1, labels2):
    return scipy.spatial.distance.jensenshannon(labels1, labels2)

# Compare quality of predictions by creating 2x2 correctness comparison matrix.
# the returned matrix will count the following at each index:
#   [0,0] --> Number of times where both labels are wrong
#   [0,1] --> Number of times only the second label is correct
#   [0,1] --> Number of times only the first label is correct
#   [1,1] --> Number of times both labels are correct
def get_correctness_matrix(labels1, labels2, target):
    assert labels1.shape == labels2.shape and labels2.shape == target.shape, "Correctness Matrix: Incompatible input shapes." 

    correctness1 = get_label_correctness(labels1, target)
    correctness2 = get_label_correctness(labels2, target)

    return confusion_matrix(correctness1, correctness2)
