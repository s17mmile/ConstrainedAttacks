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
# import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_PGD as cPGD



# Takes in and returns an example (as a 1D numpy array) and applies a constraint. For now, this constraint is fully arbitrary. 
def feasibilityProjector(example):
    return example

def constrainer(example):
    for i in range(28):
        example[i] = 10
        example[28*i] = 10
        example[28*i+27] = 10
        example[783-i] = 10
    return example



# Input file paths
datasetPath = "Datasets/MNIST/train_data.npy"
targetPath = "Datasets/MNIST/train_target.npy"
modelPath = "Models/MNIST/base_model.keras"

# Output file paths
adversaryPath = "Datasets/MNIST/PGD_train_data.npy"
newLabelPath = "Datasets/MNIST/PGD_train_label.npy"
successPath = "Datasets/MNIST/PGD_fooling_success.npy"

lossObject = keras.losses.CategoricalCrossentropy()
stepcount = 50
stepsize = 0.005

n = 100
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

    # Perform parallel PGD (on first n testing samples)
    adversaries, newLabels, success = cPGD.parallel_constrained_PGD(
        model = model,
        dataset = data[:n],
        labels = target[:n],
        lossObject = lossObject,
        stepcount = stepcount,
        stepsize = stepsize,
        feasibilityProjector = feasibilityProjector,
        constrainer = None,
        workercount = workercount,
        chunksize = chunksize
    )

    print("Saving adversaries...")

    np.save(adversaryPath, adversaries)
    np.save(newLabelPath, newLabels)
    np.save(successPath, success)

    print("Done.")