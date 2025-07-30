import numpy as np
import os
import sys
import warnings
import scipy.spatial
import tqdm
import scipy
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import RocCurveDisplay, roc_auc_score

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

# Compare the predicted labels with the respective target labels. Return an array where 0 indicates the label was wrong, and 1 indicates the label was right.
def get_label_correctness(labels, targets):
    return np.argmax(labels, axis = 1) == np.argmax(targets, axis = 1)

# Creates 2d histogram-like array to compare predicted labels with each other and with targets.
def confusion_matrix(labels1, labels2):
    assert labels1.ndim == 2, f"Confusion Matrix: labels1 must be 2D array. Received labels1 of shape {labels1.shape}."
    assert labels2.ndim == 2, f"Confusion Matrix: labels2 must be 2D array. Received labels2 of shape {labels2.shape}."
    
    matrix = np.zeros((labels1.shape[1], labels2.shape[1]))

    for class1, class2 in zip(np.argmax(labels1, axis = 1), np.argmax(labels2, axis = 1) ):
        matrix[class1, class2] += 1

    return matrix

# Compute average Jensen Shannon Distance between prediction probability Distributions
def JSD(labels1, labels2):
    return scipy.spatial.distance.jensenshannon(labels1, labels2)



# Render ROC curve (almost the same as in Timo Saala's evluation example). We mostly need to be careful with one-got and integer labels.
def renderROCandGetAUROC(testLabels, testTarget, outputPath, attackName):
    num_samples = testLabels.shape[0]

    integerTestLabels = np.argmax(testLabels, axis = 1)
    integerTestTarget = np.argmax(testTarget, axis = 1)

    testLabelScore = np.array([testLabels[i,np.argmax(testTarget[i])] for i in ])

    auroc = roc_auc_score(testTarget.ravel(), testLabels.ravel())

    RocCurveDisplay.from_predictions(
        integerTestTarget.ravel(),
        testLabelScore.ravel(),
        name="Micro-average OvR",
        color="darkorange",
    )

    plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{attackName}: Micro-averaged One-vs-Rest\nReceiver Operating Characteristic. AUROC: {auroc}.")
    plt.legend()
    plt.savefig(outputPath)
    plt.close()

    return auroc



# Accuracy
def accuracy(labels, target):
    return float(np.count_nonzero(get_label_correctness(labels, target).astype(int)))/labels.size

# Accuracy per class
# Damn I could've probably made this easier and faster by just going line-by-line through the confusion matrix, but whatever.
def accuracy_per_class(labels, targets):
    num_classes = targets.shape[1]

    # Count the total and correctly classified examples of each class
    total = np.zeros(num_classes).astype("float")
    correct = np.zeros(num_classes).astype("float")

    # Loop over all examples and increment counters 
    for label, target in zip(labels, targets):
        target_class = np.argmax(target)
        total[target_class] += 1
        if np.argmax(label) == target_class:
            correct[target_class] += 1

    return correct/total

# Compare quality of predictions by creating 2x2 correctness comparison matrix.
# the returned matrix will count the following at each index:
#   [0,0] --> Number of times where both labels are wrong
#   [0,1] --> Number of times only the second label is correct
#   [1,0] --> Number of times only the first label is correct
#   [1,1] --> Number of times both labels are correct
def get_fooling_matrix(labels1, labels2, target):
    assert labels1.shape == labels2.shape and labels2.shape == target.shape, "Fooling/Learning Matrix: Incompatible input shapes." 

    correctness1 = get_label_correctness(labels1, target).astype(int)
    correctness2 = get_label_correctness(labels2, target).astype(int)

    return confusion_matrix(correctness1, correctness2)

# This is the exact same function with a different name. Simply done to reduce confusion in the name choice for different matrices:
# - A "fooling matrix" is meant to show the performance of a model on two datasets
# - A "learning matrix" is meant to show the performance of two models on one dataset
def get_learning_matrix(labels1, labels2, target):
    return get_fooling_matrix(labels1, labels2, target)