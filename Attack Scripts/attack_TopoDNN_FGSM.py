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

import tensorflow as tf
import keras

# Local imports
import Attacks.constrained_FGSM as cFGSM

def constrainer(example):
    return example



# Input file paths
datasetPath = "Datasets/TopoDNN/train_data.npy"
targetPath = "Datasets/TopoDNN/train_target.npy"
modelPath = "Models/TopoDNN/base_model.keras"

# Output file paths
adversaryPath = "Adversaries/TopoDNN/FGSM_train_data.npy"
newLabelPath = "Adversaries/TopoDNN/FGSM_train_labels.npy"

lossObject = keras.losses.BinaryCrossentropy()
epsilon = 0.1

n = 1024
workercount = 8
chunksize = 64

if __name__ == "__main__":
    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normalized and categorized.
    if os.path.isfile(datasetPath) and os.path.isfile(targetPath):
        print("Found local dataset and targets.")
        data = np.load(datasetPath, allow_pickle=True)
        target = np.load(targetPath, allow_pickle=True)
    else:
        print("Did not find dataset or targets. Make sure it is downloaded and properly preprocessed using the given helper script.")
        quit()

    # Load pre-trained Model
    model = keras.models.load_model(modelPath)
    model.summary()

    # Perform parallel FGSM
    adversaries, newLabels = cFGSM.parallel_constrained_FGSM(
        model = model,
        dataset = data[:n],
        targets = target[:n],
        lossObject = lossObject,
        epsilon = epsilon,
        constrainer = constrainer,
        workercount = workercount,
        chunksize = chunksize
    )

    print("Saving adversaries...")

    np.save(adversaryPath, adversaries)
    np.save(newLabelPath, newLabels)

    print("Done.")
