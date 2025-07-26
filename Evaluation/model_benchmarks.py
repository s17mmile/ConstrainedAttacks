import numpy as np
import os
import sys
import warnings
import tqdm

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



# Check for correct classification of example in first and second dataset. Gives 4 options:
    # - Original example incorrect, corresponding adversarial example incorrect ("Robust Negative")
    # - Original example incorrect, corresponding adversarial example correct ("Miracle", should be extremely rare)
    # - Original example correct, corresponding adversarial example incorrect ("Adversary")
    # - Original example correct, corresponding adversarial example correct ("Robust Positive")

def benchmark_adversarial_data(model, original_data, adversarial_data, target, return_labels = False):

    if (original_data.shape != adversarial_data.shape):
        warnings.warn(f"Benchmarking Adversarial Data: received different input shapes: {original_data.shape} and {adversarial_data.shape}. Reducing benchmark scope.")
        num_samples = np.min(original_data.shape[0], adversarial_data.shape[0])
    else:
        num_samples = original_data.shape[0]

    # Get labels and compare with target to find out the correctness
    original_labels = model.predict(original_data)
    adversarial_labels = model.predict(adversarial_data)

    original_correctness = np.array(np.argmax(original_labels, axis = 1) == np.argmax(target, axis = 1))
    adversarial_correctness = np.array(np.argmax(adversarial_labels, axis = 1) == np.argmax(target, axis = 1))

    # Go through and count how many of each case occur.
    # There might be some faster way to do this with a single numpy operation, but I can't see a way to nicely keep multiple counters going at once while iterating.
    result = np.array([[0,0],
                       [0,0]])

    for i in range(num_samples):
        result[original_correctness[i], adversarial_correctness[i]] += 1

    if return_labels:
        return original_labels, adversarial_labels, result
    else:
        return result
    



# "Confusion Matrix" with two models (original and retrained) and one dataset:
# Check for correct classification of example in dataset using both classifiers. Gives 4 options per example:
    # - Original classifier correct, retrained classifier correct ("Consistent Quality")
    # - Original classifier correct, retrained classifier incorrect ("Overcorrect")
    # - Original example incorrect, corresponding adversarial example correct ("Improvement")
    # - Original example incorrect, corresponding adversarial example incorrect ("Consistent Deficit")