# LEGACY
# Non-functional due to tensorflow version issues, but kept for reference.



# Load the weights of a TopoDNN model (pre-converted from the "folder+.pb format" using an older keras version) into reconstructed architecture.
# This is sadly necessary as the old models - even after converting to .keras - cannot be loaded.
# The architecture of optimizers and layers has changed enough to make them incompatible, and fixing them directly would be a pain.
# Instead, I just extracted the weights.h5 files from each model to now load them here.
#   ---> Overall, the subfolders of Models/TopoDNN are not necessary for the project to function, but rather artefacts left in for completeness.

# # Check out the original models here:
# https://github.com/FAIR4HEP/xAI4toptagger/blob/main/models/TopoDNN

import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

weightsPath = "Models/TopoDNN/Weights/topodnnmodel.keras"

input_shape = (90,)

# This architecture reproduces "topodnnmodel" from the original paper.
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(units = 300, activation="relu"),
        keras.layers.Dense(units = 102, activation="relu"),
        keras.layers.Dense(units = 12, activation="relu"),
        keras.layers.Dense(units = 6, activation="relu"),
        keras.layers.Dense(units = 1, activation="sigmoid"),
    ]
)

model.summary()

# Compile TopoDNN Model
model.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.BinaryAccuracy(name="acc", dtype = np.float64),
    ]
)

# Since the model is already trained, we can skip training and load the weights directly.
model.load_weights("Models/TopoDNN/Weights/topodnnmodel.weights.h5")

model.save("Models/TopoDNN/base_model.keras")