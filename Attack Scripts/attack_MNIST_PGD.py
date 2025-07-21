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
# import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_PGD as cPGD



# Takes in and returns an example (as a 1D numpy array) and applies a constraint. For now, this constraint is fully arbitrary. 
def feasibilityProjector(example):
    return example

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
    example = example.numpy()[0]
    linearRescale(example,0,1)
    example = tf.convert_to_tensor(np.array([example]))
    return example



# Input file paths
datasetPath = "Datasets/MNIST/train_data.npy"
targetPath = "Datasets/MNIST/train_target.npy"
modelPath = "Models/MNIST/maxpool_model.keras"

# Output file paths
adversaryPath = "Adversaries/MNIST/PGD_train_data.npy"
newLabelPath = "Adversaries/MNIST/PGD_train_labels.npy"
successPath = "Adversaries/MNIST/PGD_fooling_success.npy"

lossObject = keras.losses.CategoricalCrossentropy()
stepcount = 20
stepsize = 0.005

workercount = 8
chunksize = 128

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

    # Perform parallel PGD
    adversaries, newLabels, success = cPGD.parallel_constrained_PGD(
        model = model,
        dataset = data,
        labels = target,
        lossObject = lossObject,
        stepcount = stepcount,
        stepsize = stepsize,
        feasibilityProjector = feasibilityProjector,
        constrainer = constrainer,
        workercount = workercount,
        chunksize = chunksize
    )

    print("Saving adversaries...")

    np.save(adversaryPath, adversaries)
    np.save(newLabelPath, newLabels)
    np.save(successPath, success)

    print("Done.")