# This script checks for the existence of the relevant datasets in the local folder.
# If they are not there, a download will be attempted from each source.
# Each dataset contains training and testing samples, and validation samples only where given.

import numpy as np
import os
import sys
import stat
import shutil
from urllib.request import urlretrieve
import tarfile
from sklearn.datasets import fetch_openml

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

from ImageNet.compileDownload import compileDownload
from TopoDNN.topodnnpreprocessing import topodnn_preprocess

print(os.getcwd())

# Choose which datasets to have set up here
get_mnist = True
get_cifar = True
get_imagenet = True
get_topodnn = True

# Create proper directory structure for Datasets folder
folders = ["Datasets", "Datasets/MNIST", "Datasets/ImageNet", "Datasets/CIFAR-10", "Datasets/TopoDNN"]

for name in folders:
    if not os.path.isdir(name):
        os.mkdir(name)

# Fetch and save the MNIST Dataset.
print()
if (get_mnist):
    trainDataExists = os.path.isfile("Datasets/MNIST/train_data.npy")
    trainTargetExists = os.path.isfile("Datasets/MNIST/train_target.npy")
    testDataExists = os.path.isfile("Datasets/MNIST/test_data.npy")
    testTargetExists = os.path.isfile("Datasets/MNIST/test_target.npy")

    if trainDataExists and trainTargetExists and testDataExists and testTargetExists:
        print("MNIST: Dataset present.")
    else:
        print("MNIST: Fetching Dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

        # Scale data to range [0,1] instead of [0,255]
        x_train = x_train.astype("float64")/255.0
        x_test = x_test.astype("float64")/255.0

        # Data format: add extra dimension to make each smaple (28,28,1) instead of (28,28)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # Create one-hot labels
        target_train = np.zeros((x_train.shape[0], 10)).astype(np.int64)
        target_train[np.arange(x_train.shape[0]), y_train.flatten()] = 1

        target_test = np.zeros((x_test.shape[0], 10)).astype(np.int64)
        target_test[np.arange(x_test.shape[0]), y_test.flatten()] = 1

        if not trainDataExists:     np.save("Datasets/MNIST/train_data.npy", x_train)
        if not trainTargetExists:   np.save("Datasets/MNIST/train_target.npy", target_train)
        if not testDataExists:      np.save("Datasets/MNIST/test_data.npy", x_test)
        if not testTargetExists:    np.save("Datasets/MNIST/test_target.npy", target_test)
        print("MNIST: Completed.")

# Fetch, save and extract the CIFAR-10 Dataset.
# Originally fetched the CIFAR Dataset in batches - this is now commented out as it unnecessarily complex.
# Keras handily provides the CIFAR-10 Dataset in preprocessed numpy array form. 
print()
if (get_cifar):
    trainDataExists = os.path.isfile("Datasets/CIFAR-10/train_data.npy")
    trainTargetExists = os.path.isfile("Datasets/CIFAR-10/train_target.npy")
    testDataExists = os.path.isfile("Datasets/CIFAR-10/test_data.npy")
    testTargetExists = os.path.isfile("Datasets/CIFAR-10/test_target.npy")

    if trainDataExists and trainTargetExists and testDataExists and testTargetExists:
        print("CIFAR-10: Dataset present.")
    else:
        print("CIFAR-10: Fetching Dataset...")
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Scale data to range [0,1] instead of [0,255]
        x_train = x_train.astype("float64")/255.0
        x_test = x_test.astype("float64")/255.0

        # Data format: add extra dimension to make each smaple (28,28,1) instead of (28,28)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        # Create one-hot targets
        target_train = np.zeros((x_train.shape[0], 10)).astype(np.int64)
        target_train[np.arange(x_train.shape[0]), y_train.flatten()] = 1

        target_test = np.zeros((x_test.shape[0], 10)).astype(np.int64)
        target_test[np.arange(x_test.shape[0]), y_test.flatten()] = 1

        if not trainDataExists:     np.save("Datasets/CIFAR-10/train_data.npy", x_train)
        if not trainTargetExists:   np.save("Datasets/CIFAR-10/train_target.npy", target_train)
        if not testDataExists:      np.save("Datasets/CIFAR-10/test_data.npy", x_test)
        if not testTargetExists:    np.save("Datasets/CIFAR-10/test_target.npy", target_test)
        print("CIFAR-10: Completed.")

# Fetch, save and extract all three ImageNet v2 testing Datasets.
# This fetches one of three datasets provided for ImageNet v2, depending on the exact url.
# In this case, fetching the "threshold 0.7" set.
if (get_imagenet):
    destination = "Datasets/ImageNet/"
    expected_subdir_count = 1000
    resolution = (224,224)

    urls =          [
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz?download=true",
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz?download=true",
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz?download=true"
                    ]
    
    archive_names = [
                    "imagenetv2-threshold.tar.gz",
                    "imagenetv2-matched_frequency.tar.gz",
                    "imagenetv2-top_images.tar.gz"
                    ]

    dir_names =     [
                    "imagenetv2-threshold0.7-format-val",
                    "imagenetv2-matched-frequency-format-val",
                    "imagenetv2-top-images-format-val"
                    ]
    
    
    array_names =   [
                    "threshold",
                    "matched-frequency",
                    "top-images"
                    ]

    for i in range(len(archive_names)):
        print()

        url = urls[i]
        archive_name = archive_names[i]
        dir_name = dir_names[i]
        array_name = array_names[i]

        if not os.path.isfile(destination+archive_name):
            print("ImageNet v2: Fetching archive...")
            urlretrieve(url, destination+archive_name)
            print("ImageNet v2: Complete.")
        else:
            print("ImageNet v2: Archive found locally.")
        
        if not os.path.isdir(destination+dir_name):
            print("ImageNet v2: Image directory does not exist. Creating...")
            os.mkdir(destination+dir_name)
        else:
            print("ImageNet v2: Image directory exists.")

        if (len(os.listdir(destination+dir_name)) != expected_subdir_count):
            print("ImageNet v2: Image folder has incorrect subdirectory structure. Clearing...")
            shutil.rmtree(destination+dir_name)
            print("ImageNet v2: Complete.")
        
            print("ImageNet v2: Extracting Archive...")
            archive = tarfile.open(destination+archive_name)
            archive.extractall(destination)
            archive.close()
            print("ImageNet v2: Complete.")
        else:
            print("ImageNet v2: Image directory subdirectory structure is correct.")

        if not (os.path.isfile(destination+array_name+"_data.npy") and os.path.isfile(destination+array_name+"_target.npy")):
            print("ImageNet v2: Compiling Dataset into numpy format...")
            compileDownload(destination+dir_name, resolution, destination+array_name+"_data.npy", destination+array_name+"_target.npy")
            print("ImageNet v2: Completed.")


# Fetch and save the TopoDNN dataset (training, validation and testing)
print()
if (get_topodnn):
    destination = "Datasets/TopoDNN/"
    
    # Data Fetching

    urls =      [
                "https://syncandshare.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=test.h5",
                "https://syncandshare.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=train.h5",
                "https://syncandshare.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=val.h5"
                ]

    names =     ["test", "train", "val"]

    for i in range(len(urls)):
        url = urls[i]
        name = names[i]

        # Fetching, if necessary
        if (os.path.isfile(destination + name + ".h5")):
            print("TopoDNN: \"" + name + "\" dataset found.")
        else:
            print("TopoDNN: Fetching \"" + name + "\" dataset...")
            urlretrieve(url, destination + name + ".h5")
            print("TopoDNN: Complete.")

        # Preprocessing, if necessary
        if (os.path.isfile(destination + name + "_data.npy") and os.path.isfile(destination + name + "_target.npy")):
            print("TopoDNN: Preprocessed \"" + name + "\" dataset found.")
        else:
            print("TopoDNN: Preprocessing \"" + name + "\" dataset...")
            topodnn_preprocess(destination + name + ".h5")
        
        print()