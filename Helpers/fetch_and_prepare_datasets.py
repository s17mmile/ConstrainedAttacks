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
import keras

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
        if not trainDataExists:     np.save("Datasets/MNIST/train_data.npy", x_train)
        if not trainTargetExists:   np.save("Datasets/MNIST/train_target.npy", y_train)
        if not testDataExists:      np.save("Datasets/MNIST/test_data.npy", x_test)
        if not testTargetExists:    np.save("Datasets/MNIST/test_target.npy", y_test)
        print("MNIST: Completed.")

# Fetch, save and extract the CIFAR-10 Dataset.
# Originally fetched the CIFAR Dataset in batches - this is now commented out as it unnecessarily complex.
# Keras handily provides the CIFAR-10 Dataset in preprocessed numpy array form. 
'''
if (get_cifar):
    url = "https://www.cs.toronto.edu/%7Ekriz/cifar-10-python.tar.gz"
    archive_path = "Datasets/CIFAR-10/cifar-10-python.tar.gz"

    destination = "Datasets/CIFAR-10/"
    folder = "batches"

    if not os.path.isfile(archive_path):
        print("CIFAR-10: Fetching archive.")
        urlretrieve(url, archive_path)
        print("Complete.")
    
    if (os.path.isdir(destination+folder)):
        print("CIFAR-10: Clearing image folder.")
        shutil.rmtree(destination+folder)
        print("CIFAR-10: Complete.")

    print("CIFAR-10: Extracting Archive.")
    archive = tarfile.open(archive_path)
    archive.extractall(destination)
    os.rename(destination+"cifar-10-batches-py", destination+folder)
    archive.close()
    print("CIFAR-10: Complete.")
'''
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
        if not trainDataExists:     np.save("Datasets/CIFAR-10/train_data.npy", x_train)
        if not trainTargetExists:   np.save("Datasets/CIFAR-10/train_target.npy", y_train)
        if not testDataExists:      np.save("Datasets/CIFAR-10/test_data.npy", x_test)
        if not testTargetExists:    np.save("Datasets/CIFAR-10/test_target.npy", y_test)
        print("CIFAR-10: Completed.")

# Fetch, save and extract all three ImageNet v2 testing Datasets.
# This fetches one of three datasets provided for ImageNet v2, depending on the exact url.
# In this case, fetching the "threshold 0.7" set.
if (get_imagenet):
    destination = "Datasets/ImageNet/"
    expected_subdir_count = 1000

    urls =          [
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz?download=true",
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz?download=true",
                    "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-top-images.tar.gz?download=true"
                    ]
    
    archive_paths = [
                    "Datasets/ImageNet/imagenetv2-threshold.tar.gz",
                    "Datasets/ImageNet/imagenetv2-matched_frequency.tar.gz",
                    "Datasets/ImageNet/imagenetv2-top_images.tar.gz"
                    ]

    old_names =     [
                    "imagenetv2-threshold0.7-format-val",
                    "imagenetv2-matched-frequency-format-val",
                    "imagenetv2-top-images-format-val"
                    ]
    
    
    new_names =     [
                    "imagenetv2-threshold",
                    "imagenetv2-matched_frequency",
                    "imagenetv2-top_images"
                    ]

    for i in range(len(archive_paths)):
        print()

        url = urls[i]
        archive_path = archive_paths[i]
        old_name = old_names[i]
        new_name = new_names[i]

        if not os.path.isfile(archive_path):
            print("ImageNet v2: Fetching archive.")
            urlretrieve(url, archive_path)
            print("ImageNet v2: Complete.")
        else:
            print("ImageNet v2: Archive found locally.")
        
        if not os.path.isdir(destination+new_name):
            print("ImageNet v2: Image directory does not exist. Creating.")
            os.mkdir(destination+new_name)
        else:
            print("ImageNet v2: Image directory exists.")

        if (len(os.listdir(destination+new_name)) != expected_subdir_count):
            print("ImageNet v2: Image folder has incorrect subdirectory structure. Clearing.")
            shutil.rmtree(destination+new_name)
            print("ImageNet v2: Complete.")
        
            print("ImageNet v2: Extracting Archive.")
            archive = tarfile.open(archive_path)
            archive.extractall(destination)
            archive.close()
            os.chmod(destination+old_name, stat.S_IRWXO)
            os.rename(destination+old_name, destination+new_name)
            print("ImageNet v2: Complete.")
        else:
            print("ImageNet v2: Image directory subdirectory structure is correct.")

# Fetch and save the TopoDNN dataset (training, validation and testing)
print()
if (get_topodnn):
    destination = "Datasets/TopoDNN/"
    
    url_test = "https://syncandshare.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=test.h5"
    url_train = "https://syncandshare.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=train.h5"
    url_validate = "https://syncandshare.desy.de/index.php/s/llbX3zpLhazgPJ6/download?path=%2F&files=val.h5"

    if (os.path.isfile(destination+"test.h5")):
        print("TopoDNN: Testing dataset found.")
    else:
        print("TopoDNN: Feching testing dataset.")
        urlretrieve(url_test, destination+"test.h5")
        print("TopoDNN: Complete.")

    if (os.path.isfile(destination+"train.h5")):
        print("TopoDNN: Training dataset found.")
    else:
        print("TopoDNN: Feching training dataset.")
        urlretrieve(url_train, destination+"train.h5")
        print("TopoDNN: Complete.")

    if (os.path.isfile(destination+"val.h5")):
        print("TopoDNN: Validation dataset found.")
    else:
        print("TopoDNN: Feching validation dataset.")
        urlretrieve(url_validate, destination+"val.h5")
        print("TopoDNN: Complete.")