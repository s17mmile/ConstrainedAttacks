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



def constrainer(example):
    return np.clip(example, 0, 1)



# Input file paths
datasetPath = "Datasets/ImageNet/threshold_data.npy"
targetPath = "Datasets/ImageNet/threshold_target.npy"
modelPath = "Models/ImageNet/base_model.keras"

# Output file paths
adversaryPath = "Adversaries/ImageNet/RDSA_threshold_data.npy"
newLabelPath = "Adversaries/ImageNet/RDSA_threshold_labels.npy"

categoricalFeatureMaximum = 150
binCount = 100
perturbedFeatureCount = 1000
RDSA_attempts = 25

n = 128
workercount = 8
chunksize = 16

if __name__ == "__main__":

    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normalized and categorized.
    if os.path.isfile(datasetPath) and os.path.isfile(targetPath):
        print("Found local dataset and target.")
        data = np.load(datasetPath, allow_pickle=True)
        target = np.load(targetPath, allow_pickle=True)
        print("Data shape: ", data.shape)
        print("Target Shape: ", target.shape)
    else:
        print("Did not find dataset or target. Make sure it is downloaded and properly preprocessed using the given helper script.")
        quit()

    # Load pre-trained Model
    model = keras.models.load_model(modelPath)
    model.summary()



    # Perform parallel RDSA
    adversaries, newLabels = cRDSA.parallel_constrained_RDSA(
        model = model,
        dataset = data,
        targets = target,
        steps = RDSA_attempts,
        categoricalFeatureMaximum = categoricalFeatureMaximum,
        binCount = binCount,
        perturbedFeatureCount = perturbedFeatureCount,
        constrainer = constrainer,
        workercount = workercount,
        chunksize = chunksize,
        n = n
    )

    print("Saving adversaries...")

    np.save(adversaryPath, adversaries)
    np.save(newLabelPath, newLabels)

    print("Done.")