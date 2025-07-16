import numpy as np
import os
import gc
import sys
import multiprocessing
import tqdm
import random

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Hiding tensorflow performance warning for CPU-specific instruction set extensions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_RDSA as cRDSA



trainTestSplitSeed = 42

evaluateBase = False

dataset_name = "mnist_784"
dataset_path = "Datasets/MNIST/MNIST784_data.npy"
labels_path = "Datasets/MNIST/MNIST784_target.npy"
force_download = False
save_locally = True

model_path = "Models/MNIST/best_model.keras"
results_path = "Results/TestBaseModel/"

categoricalFeatureMaximum = 50
binCount = 100
perturbedFeatureCount = 200
RDSA_attempts = 100

n = 100
workercount = 8
chunksize = 16
adversaryFolder = "Datasets/MNIST/"

if __name__ == "__main__":

    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normaized and categorized,
    if os.path.isfile(dataset_path) and os.path.isfile(labels_path) and not force_download:
        print("Found local dataset and labels.")
        data= np.load(dataset_path, allow_pickle=True)
        target = np.load(labels_path, allow_pickle=True)
    else:
        print("Did not find dataset. Make sure it is downloaded and properly preprocessed.")

    # Load pre-trained Model
    model = keras.models.load_model(model_path)

    # Find indices of features to be considered continuous/categorical.
    numUniqueValues, continuous, categorical = RDSA_Help.featureContinuity(X_test, categoricalFeatureMaximum)

    # Generate probability density function for each continuous feature.
    # Non-continuous features are given an empty placeholder
    binEdges, binProbabilites  = RDSA_Help.featureDistributions(X_test, continuous, binCount)

    # Randomly choose a given number of continuous features to be perturbed for the first n examples
    perturbationIndexLists = [random.sample(continuous, perturbedFeatureCount) for i in range(n)]



    # Perform parallel RDSA (on first n testing samples)
    adversaries, newLabels, success = cRDSA.parallel_constrained_RDSA(
        model = model,
        dataset = data[:n],
        labels = target[:n],
        steps = RDSA_attempts,
        perturbationIndexLists = perturbationIndexLists,
        binEdges = binEdges,
        binProbabilites = binProbabilites,
        constrainer = None,
        workercount = workercount,
        chunksize = chunksize
    )

    print("saving")

    np.save(adversaryFolder + "MNIST784_adv_RDSA_data.npy", adversaries)
    np.save(adversaryFolder + "MNIST784_adv_RDSA_labels.npy", newLabels)
    np.save(adversaryFolder + "MNIST784_adv_RDSA_indicators.npy", success)
