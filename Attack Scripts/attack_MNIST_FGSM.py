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




trainTestSplitSeed = 42

evaluateBase = False

dataset_name = "mnist_784"
dataset_path = "Datasets/MNIST/MNIST784_data.npy"
labels_path = "Datasets/MNIST/MNIST784_target.npy"
force_download = False
save_locally = True

model_path = "Models/MNIST/best_model.keras"
results_path = "Results/TestBaseModel/"

lossObject = keras.losses.CategoricalCrossentropy()
epsilon = 0.1

n = 100
workercount = 8
chunksize = 16
adversaryFolder = "Datasets/MNIST/"

if __name__ == "__main__":

    # Load dataset
    # If the dataset is saved locally, just use that instead of re-downloading. This assumes that it is already properly normaized and categorized,
    if os.path.isfile(dataset_path) and os.path.isfile(labels_path) and not force_download:
        print("Found local dataset and labels.")
        X = np.load(dataset_path, allow_pickle=True)
        Y = np.load(labels_path, allow_pickle=True)
    else:
        print("Downloading dataset.")
        baseDataset = fetch_openml(dataset_name, as_frame = False, parser="liac-arff")

        # Extract and normalize the examples
        X = StandardScaler().fit_transform(baseDataset.data.astype(np.float32))
        # Encode the labels as one-hot vectors
        Y = to_categorical(baseDataset.target.astype(int))

        if save_locally:
            print("Saving dataset.")
            np.save(dataset_path, X)
            np.save(labels_path, Y)

    # Perform train-test-split.
    # We're using a pre-trained model here, which should be trained on the same split to avoid evaluating on training examples 
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=trainTestSplitSeed)

    # Load pre-trained Model
    model = keras.models.load_model(model_path)



    # Perform parallel FGSM (on first n testing samples)
    adversaries, newLabels, success = cFGSM.parallel_constrained_FGSM(
        model = model,
        dataset = X_test[:n],
        labels = Y_test[:n],
        lossObject = lossObject,
        epsilon = epsilon,
        constrainer = None,
        workercount = workercount,
        chunksize = chunksize
    )

    print("saving")

    np.save(adversaryFolder + "MNIST784_adv_FGSM_data.npy", adversaries)
    np.save(adversaryFolder + "MNIST784_adv_FGSM_labels.npy", newLabels)
    np.save(adversaryFolder + "MNIST784_adv_FGSM_indicators.npy", success)
