# This script checks for the existence of the relevant datasets in the local folder.
# If they are not there, a download will be attempted from each source.
# Each dataset contains training and testing samples, and validation samples only where given.

import numpy as np
import os
import shutil
from urllib.request import urlretrieve
import tarfile
from sklearn.datasets import fetch_openml

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

# Fetch and save the MNIST784 Dataset
if (get_mnist):
    if os.path.isfile("Datasets/MNIST/MNIST784_data.npy") and os.path.isfile("Datasets/MNIST/MNIST784_target.npy"):
        print("MNIST: Dataset present.")
    else:
        print("MNIST: Fetching Dataset...")
        MNIST784 = fetch_openml("MNIST_784", as_frame = False, parser="liac-arff")
        np.save("Datasets/MNIST/MNIST784_data.npy", MNIST784.data)
        np.save("Datasets/MNIST/MNIST784_target.npy", MNIST784.target)
        print("Complete.")

# Fetch, save and extract the CIFAR-10 Dataset.
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

# Fetch, save and extract the ImageNet v2 Dataset.
# This fetches one of three datasets provided for ImageNet v2, depending on the exact url.
# In this case, fetching the "threshold 0.7" set.
if (get_imagenet):
    url = "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-threshold0.7.tar.gz?download=true"
    archive_path = "Datasets/ImageNet/imagenetv2-threshold.tar.gz"

    destination = "Datasets/ImageNet/"
    folder = "imagenetv2-threshold"

    if not os.path.isfile(archive_path):
        print("ImageNet v2: Fetching archive.")
        urlretrieve(url, archive_path)
        print("Complete.")
    
    if (os.path.isdir(destination+folder)):
        print("ImageNet v2: Clearing image folder.")
        shutil.rmtree(destination+folder)
        print("ImageNet v2: Complete.")

    print("ImageNet v2: Extracting Archive.")
    archive = tarfile.open(archive_path)
    archive.extractall(destination)
    os.rename(destination+"imagenetv2-threshold0.7-format-val", destination+folder)
    archive.close()
    print("ImageNet v2: Complete.")

# Fetch and save the TopoDNN dataset (training, validation and testing)
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