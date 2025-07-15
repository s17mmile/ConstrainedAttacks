import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Hiding tensorflow performance warning for CPU-specific instruction set extensions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras


# Load pretrained model
model = keras.applications.MobileNetV2(include_top=True, weights='imagenet')

model.save("Models/ImageNet/base_model.keras")