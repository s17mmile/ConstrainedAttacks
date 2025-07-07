import os
import numpy as np
import sys
import tqdm

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt

from Helpers.ImageNet.Preprocessing import preprocess

# The downloaded test set contains - in this case - 1000 sets of 10 images each, in subfolders. The below function does not rely on these numbers, though.
# We want to compile this into a usable numpy format containing each (preprocessed) image as data and the folder title as the label.
# The folder titles are just integers, thought each folder has an associated class. This will be important later.
# --> TODO find/create a list of these class index <-> class description associations?
def compileDownload(sourceFolder, resolution, resultDataPath, resultLabelPath):
    subfolders = os.listdir(sourceFolder)

    images = []
    labels = []

    print("Compilation Progress by folders:") 
    # Start a progress bar
    for subfolder in tqdm.tqdm(subfolders):
        currentFiles = os.listdir(sourceFolder + "/" + subfolder)

        for filename in currentFiles:
            imagePath = sourceFolder + "/" + subfolder + "/" + filename

            # Read in and preprocess the image
            image = tf.image.decode_image(tf.io.read_file(imagePath))
            image = preprocess(image, resolution)

            # Save the image and label
            images.append(image)
            labels.append(int(subfolder))
    
    # Encode labels as one-hot vectors
    labels = keras.utils.to_categorical(labels, len(subfolders))

    images = np.array(images)
    labels = np.array(labels)

    print(images.shape)
    print(labels.shape)

    np.save(resultDataPath, images)
    np.save(resultLabelPath, labels)





# Specifications for compiling
sourceFolder = "Datasets/ImageNET/imagenetv2-threshold"
resolution = (224,224)
resultDataPath = "Datasets/ImageNET/ImageNETv2_data.npy"
resultLabelPath = "Datasets/ImageNET/ImageNETv2_target.npy"

compileDownload(sourceFolder, resolution, resultDataPath, resultLabelPath)