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

# Local imports
# import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_FGSM as cFGSM



dataset_path = "Datasets/ImageNet/threshold_data.npy"
target_path = "Datasets/ImageNet/threshold_target.npy"
adversaryFolder = "Datasets/ImageNet/"

# Load pre-trained Model
model = keras.applications.MobileNetV2(include_top=True, weights='imagenet')

lossObject = keras.losses.CategoricalCrossentropy()
epsilon = 0.1

n = 1000
workercount = 8
chunksize = 10



if __name__ == "__main__":

    # Load dataset
    # For the purposes of ImageNet, this assumes that the dataset has been downloaded and compiled into a single .npy file
    # This numpy array holds all images, preprocessed using the "preprocess" helper function.
    # If this has not occured, use the compileDownload() helper function to create it.
    if os.path.isfile(dataset_path) and os.path.isfile(target_path):
        print("Found local dataset and labels.")
        data = np.load(dataset_path, allow_pickle=True)
        target = np.load(target_path, allow_pickle=True)
    else:
        print("No dataset found. Ensure you have a compiled dataset (use compileDownload() helper) and the path to it is correct.")
        quit()


    # Perform parallel FGSM (on first n testing samples)
    adversaries, newLabels, success = cFGSM.parallel_constrained_FGSM(
        model = model,
        dataset = X[:10*n:10],
        labels = Y[:10*n:10],
        lossObject = lossObject,
        epsilon = epsilon,
        constrainer = None,
        workercount = workercount,
        chunksize = chunksize
    )

    print("saving")

    np.save(adversaryFolder + "ImageNetv2_adv_FGSM_data.npy", adversaries)
    np.save(adversaryFolder + "ImageNetv2_adv_FGSM_labels.npy", newLabels)
    np.save(adversaryFolder + "ImageNetv2_adv_FGSM_indicators.npy", success)
