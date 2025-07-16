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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

import keras

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_RDSA as cRDSA



# Input file paths
datasetPath = "Datasets/MNIST/train_data.npy"
targetPath = "Datasets/MNIST/train_target.npy"
modelPath = "Models/MNIST/base_model.keras"

# Output file paths
adversaryPath = "Datasets/MNIST/RDSA_train_data.npy"
newLabelPath = "Datasets/MNIST/RDSA_train_label.npy"
successPath = "Datasets/MNIST/RDSA_fooling_success.npy"

categoricalFeatureMaximum = 100
binCount = 100
perturbedFeatureCount = 200
RDSA_attempts = 100

n = 100
workercount = 8
chunksize = 16

if __name__ == "__main__":

    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normalized and categorized.
    if os.path.isfile(datasetPath) and os.path.isfile(targetPath):
        print("Found local dataset and labels.")
        data = np.load(datasetPath, allow_pickle=True)
        labels = np.load(targetPath, allow_pickle=True)
    else:
        print("Did not find dataset or labels. Make sure it is downloaded and properly preprocessed using the given helper script.")
        quit()

    # Load pre-trained Model
    model = keras.models.load_model(modelPath)
    model.summary()



    # RDSA Preparation --> TODO move this into RDSA attack script?
    # Find indices of features to be considered continuous/categorical.
    numUniqueValues, continuous, categorical = RDSA_Help.featureContinuity(data, categoricalFeatureMaximum)

    # Generate probability density function for each continuous feature.
    # Non-continuous features are given an empty placeholder
    binEdges, binProbabilites  = RDSA_Help.featureDistributions(data, continuous, binCount)

    # Randomly choose a given number of continuous features to be perturbed for the first n examples
    perturbationIndexLists = [random.sample(continuous, perturbedFeatureCount) for i in range(n)]

    # DEBUG
    quit()



    # Perform parallel RDSA (on first n testing samples)
    adversaries, newLabels, success = cRDSA.parallel_constrained_RDSA(
        model = model,
        dataset = data[:n],
        labels = labels[:n],
        steps = RDSA_attempts,
        perturbationIndexLists = perturbationIndexLists,
        binEdges = binEdges,
        binProbabilites = binProbabilites,
        constrainer = None,
        workercount = workercount,
        chunksize = chunksize
    )

    print("Saving adversaries...")

    np.save(adversaryPath, adversaries)
    np.save(newLabelPath, newLabels)
    np.save(successPath, success)

    print("Done.")