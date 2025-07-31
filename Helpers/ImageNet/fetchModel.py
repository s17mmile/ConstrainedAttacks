import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

# Hiding tensorflow performance warning for CPU-specific instruction set extensions
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import keras
import tensorflow as tf

# Load pretrained model
model = keras.applications.MobileNetV2(include_top=True, weights='imagenet')

# Add necessary compilation parameters. Chosen to be the same as for the other networks used.
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.CategoricalAccuracy(),
    ],
)

# Need to get Adam to set up its internal variables. So, we need to call fit once and then reset the weights.
original_weights = model.get_weights()

dummydata = np.zeros((1,224,224,3))
dummytarget = np.zeros((1,1000))
dummytarget[0,0] = 1

model.fit(dummydata, dummytarget)

model.set_weights(original_weights)



model.save("Models/ImageNet/base_model.keras")

model1 = keras.models.load_model("Models/ImageNet/base_model.keras")
