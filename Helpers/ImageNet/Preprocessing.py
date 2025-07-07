import os
import numpy as np
import sys 

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt


# Helper function to preprocess the image so that it can be inputted in MobileNetV2
def preprocess(image, resolution):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, resolution)
    image = keras.applications.mobilenet_v2.preprocess_input(image)
    
    return image

