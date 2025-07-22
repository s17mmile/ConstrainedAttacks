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

# Local imports
# import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_FGSM as cFGSM



# Rescale an arrary linearly from its original range into a given one.
def linearRescale(array, newMin, newMax):
    minimum, maximum = np.min(array), np.max(array)
    m = (newMax - newMin) / (maximum - minimum)
    b = newMin - m * minimum
    scaledArray = m * array + b
    # Remove rounding errors by clipping. The difference is tiny.
    return np.clip(scaledArray, newMin, newMax)

# Todo write constrainer that re-scales everything linearly instead of just clipping it.
def constrainer(example):
    return linearRescale(example,0,1)



# Input file paths
datasetPath = "Datasets/ImageNet/threshold_data.npy"
targetPath = "Datasets/ImageNet/threshold_target.npy"
modelPath = "Models/ImageNet/base_model.keras"

# Output file paths
adversaryPath = "Adversaries/ImageNet/FGSM_threshold_data.npy"
newLabelPath = "Adversaries/ImageNet/FGSM_threshold_labels.npy"
successPath = "Adversaries/ImageNet/FGSM_threshold_success.npy"

lossObject = keras.losses.CategoricalCrossentropy()
epsilon = 0.1

n = 128
workercount = 8
chunksize = 16

if __name__ == "__main__":
    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normalized and categorized.
    if os.path.isfile(datasetPath) and os.path.isfile(targetPath):
        print("Found local dataset and labels.")
        data = np.load(datasetPath, allow_pickle=True)
        target = np.load(targetPath, allow_pickle=True)
    else:
        print("Did not find dataset or labels. Make sure it is downloaded and properly preprocessed using the given helper script.")
        quit()

    # Load pre-trained Model
    model = keras.models.load_model(modelPath)
    model.summary()

    quit()

    # Perform parallel FGSM (on first n testing samples)
    adversaries, newLabels, success = cFGSM.parallel_constrained_FGSM(
        model = model,
        dataset = data[:n],
        labels = target[:n],
        lossObject = lossObject,
        epsilon = epsilon,
        constrainer = constrainer,
        workercount = workercount,
        chunksize = chunksize
    )

    print("Saving adversaries...")

    np.save(adversaryPath, adversaries)
    np.save(newLabelPath, newLabels)
    np.save(successPath, success)

    print("Done.")
