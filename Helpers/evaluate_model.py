import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.losses import MSE
import time
import sys
from sklearn.utils import shuffle
import math
import matplotlib.pyplot as plt
import os
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from scipy.spatial.distance import jensenshannon

def getLossAndAccuracyServer(modelDirectory, x_test, y_test):
    #print("Actually getting Loss and Accuracy")
    #print(modelDirectory)

    model = keras.models.load_model(modelDirectory)
    score = model.evaluate(x_test, y_test, verbose=1)
    #print("Test loss:", score[0])
    #print("Test accuracy:", score[1])
    return score


def getLossAndAccuracy(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=1)
    #print("Test loss:", score[0])
    #print("Test accuracy:", score[1])
    return score


def getFoolingRatio(model, x_test, y_test, x_test_adv, loss):
    """
    Computes the fooling ratio of adversarial examples.

    :param model: Keras model
    :param x_test: original test samples
    :param y_test: true labels for the test samples
    :param x_test_adv: adversarial examples generated from x_test
    :param loss: loss type ('CCE' for categorical cross-entropy or other for binary cross-entropy)
    :return: fooling ratio (percentage of correctly classified examples that are fooled by adversarial examples)
    """

    # Predict real and adversarial examples
    pred_real = model.predict(x_test)
    pred_adv = model.predict(x_test_adv)
    label = y_test

    correctly_classified_real = 0
    fooled = 0

    for pr, pa, l in zip(pred_real, pred_adv, label):
        if loss == 'CCE':
            # Multi-class classification case: check if the argmax of the predictions matches the label
            real_pred_class = np.argmax(pr)  # Class prediction for the real sample
            adv_pred_class = np.argmax(pa)  # Class prediction for the adversarial sample
            true_class = np.argmax(l) if len(l) > 1 else l  # True class label (assumes one-hot if len(l) > 1)

            # Check if the real prediction is correct
            if real_pred_class == true_class:
                correctly_classified_real += 1
                # Check if the adversarial example is classified differently
                if adv_pred_class != real_pred_class:
                    fooled += 1
        else:
            # Binary classification case: rounding the probability to get the class (0 or 1)
            real_pred_class = np.rint(pr)[0]  # Rounding to get the predicted class
            adv_pred_class = np.rint(pa)[0]  # Rounding adversarial example prediction
            true_class = l  # True class label

            # Check if the real prediction is correct
            if real_pred_class == true_class:
                correctly_classified_real += 1
                # Check if the adversarial example is classified differently
                if adv_pred_class != real_pred_class:
                    fooled += 1

    # Compute fooling ratio as the ratio of fooled examples to correctly classified examples
    fooling_ratio = fooled / correctly_classified_real if correctly_classified_real > 0 else 0

    return fooling_ratio


def getDeviation(x_test, x_test_adv):
    x_test = shuffle(x_test, random_state=42)

    dev = []
    for test, adv in zip(x_test, x_test_adv):
        dev.append(np.abs(test - adv))

    return np.mean(dev)


def getROC(y_train, y_test, y_pred, plot=True):
    label_binarizer = LabelBinarizer().fit(y_train)
    y_onehot_test = label_binarizer.transform(y_test)

    RocCurveDisplay.from_predictions(
        y_onehot_test.ravel(),
        y_pred.ravel(),
        name="micro-average OvR",
        color="darkorange",
    )

    if plot:
        plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Micro-averaged One-vs-Rest\nReceiver Operating Characteristic")
        plt.legend()
        plt.show()
        plt.close()

    return roc_auc_score(y_onehot_test.ravel(), y_pred.ravel())


def get_JSD(init_data, adv_data, nVars):
    freqs_init = []
    freqs_ca = []
    for i in range(0, nVars):
        tmp_var = init_data[:, i]
        freq_init, bins_init = np.histogram(tmp_var, bins=np.linspace(np.min(tmp_var), np.max(tmp_var), 30),
                                            density=False)

        tmp_var_advs = adv_data[:, i]
        freq_ca, bins_ca = np.histogram(tmp_var_advs, bins=np.linspace(np.min(tmp_var), np.max(tmp_var), 30),
                                        density=False)

        freqs_init.append(freq_init)
        freqs_ca.append(freq_ca)

    freqs_init = np.asarray(freqs_init).flatten()
    freqs_ca = np.asarray(freqs_ca).flatten()

    return jensenshannon(freqs_init, freqs_ca)


def compute_difference_per_event(advs_pgd, x_test):
    # Compute the difference per event
    difference_per_event_pgd = []
    for i in range(len(advs_pgd)):
        difference_per_event_pgd.append(np.mean(np.absolute(np.subtract(x_test[i], advs_pgd[i]))))

    return np.mean(difference_per_event_pgd)

