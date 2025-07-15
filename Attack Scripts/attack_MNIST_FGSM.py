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

import keras

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
# import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_FGSM as cFGSM

# Input file paths
datasetPath = "Datasets/MNIST/train_data.npy"
labelsPath = "Datasets/MNIST/train_target.npy"
modelPath = "Models/MNIST/base_model.keras"

# Output file path for fooling success indicators 
successPath = "Results/MNIST/FGSM_fooling_success.npy"

lossObject = keras.losses.CategoricalCrossentropy()
epsilon = 0.1

n = 100
workercount = 8
chunksize = 16

if __name__ == "__main__":

    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normalized and categorized.
    if os.path.isfile(datasetPath) and os.path.isfile(labelsPath):
        print("Found local dataset and labels.")
        data = np.load(datasetPath, allow_pickle=True)
        labels = np.load(labelsPath, allow_pickle=True)
    else:
        print("Did not find dataset or labels. Make sure it is downloaded and properly preprocessed using the given helper script.")
        quit()

    # Load pre-trained Model
    model = keras.models.load_model(modelPath)
    model.summary()

    # Perform parallel FGSM (on first n testing samples)
    adversaries, newLabels, success = cFGSM.parallel_constrained_FGSM(
        model = model,
        dataset = data[:n],
        labels = labels[:n],
        lossObject = lossObject,
        epsilon = epsilon,
        constrainer = None,
        workercount = workercount,
        chunksize = chunksize
    )

    print("saving")

    np.save(datasetPath.replace(".npy", "_adv.npy"), adversaries)
    np.save(labelsPath.replace(".npy", "_adv.npy"), newLabels)
    np.save(successPath, success)
