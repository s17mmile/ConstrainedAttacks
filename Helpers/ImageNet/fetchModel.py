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

print(model.optimizer)

# Add necessary compilation parameters. Chosen to be the same as for the other networks used.
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc", dtype = np.float64),
    ]
)

print(model.optimizer)

model.save("Models/ImageNet/base_model.keras")



model1 = keras.models.load_model("Models/ImageNet/base_model.keras", compile = False)

# Add necessary compilation parameters. Chosen to be the same as for the other networks used.
model1.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc", dtype = np.float64),
    ]
)

print(model1.optimizer)