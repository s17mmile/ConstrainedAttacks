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
# import Helpers.RDSA_Helpers as RDSA_Help
import Attacks.constrained_FGSM as cFGSM
from Helpers.ImageNet.Visualization import displayImage

trainTestSplitSeed = 42

dataset_path = "Datasets/ImageNet/ImageNetv2_data.npy"
target_path = "Datasets/ImageNet/ImageNetv2_target.npy"
adversaryFolder = "Datasets/ImageNet/"

# Load pre-trained Model
model = keras.applications.MobileNetV2(include_top=True, weights='imagenet')
# model_name = "Models/MNIST/best_model.keras"
# results_path = "Results/TestBaseModel/"

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
        X = np.load(dataset_path, allow_pickle=True)
        Y = np.load(target_path, allow_pickle=True)
    else:
        print("No dataset found. Ensure you have a compiled dataset (use compileDownload() helper) and the path to it is correct.")
        quit()

#     # Perform train-test-split.
#     # We're using a pre-trained model here, which should be trained on the same split to avoid evaluating on training examples 
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=trainTestSplitSeed)

    print("Y_Shape:", Y.shape)
    print("X_Shape:", X.shape)

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
