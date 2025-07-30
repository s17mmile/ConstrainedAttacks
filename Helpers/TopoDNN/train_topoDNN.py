# Train a TopoDNN model with identical architecture to the default model found in the original work.
# (https://github.com/FAIR4HEP/xAI4toptagger/blob/main/models/TopoDNN)
# I've tried whatever I reasonably can within my timeframe, but I simply cannot transfer the original TopoDNN models to keras 3.9.2.
# Not even manually setting all architecture details and just loading the weights works, and no one seems to have fixed that issue in a way that isn't just changing TF versions, which is out of the question for me now.

import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Load training dataset and targets
data = np.load("Datasets/TopoDNN/train_data.npy", allow_pickle=True)
target = np.load("Datasets/TopoDNN/train_target.npy", allow_pickle=True)

# Load testing dataset and targets for validation purposes
test_data = np.load("Datasets/TopoDNN/test_data.npy", allow_pickle=True)
test_target = np.load("Datasets/TopoDNN/test_target.npy", allow_pickle=True)

print("data shape:", data.shape)
print("target shape:", target.shape)

# Specify Model Input
input_shape = (90,)

# This architecture reproduces "topodnnmodel" from the original paper.
# The only difference is that the output layer has 2 units instead of 1 to represent the two possible classes.
# This is in closer alignment with the rest of this project, which uses one-hot encoded labels and prediction vectors with the node count being equal to the class count.
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Dense(units = 300, activation="relu"),
        keras.layers.Dense(units = 102, activation="relu"),
        keras.layers.Dense(units = 12, activation="relu"),
        keras.layers.Dense(units = 6, activation="relu"),
        keras.layers.Dense(units = 2, activation="sigmoid"),
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

# Training and Evaluation given Training and Testing Datasets. A small part of the data is set aside for validation purposes.
batch_size = 128
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="Models/TopoDNN/epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    data,
    target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.15,
    callbacks=callbacks,
)

model.evaluate(test_data, test_target, verbose=1)

model.save("Models/TopoDNN/test_model.keras")