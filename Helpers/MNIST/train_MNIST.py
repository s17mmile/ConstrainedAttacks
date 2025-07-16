import numpy as np
import os 

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Load the data and split it between train and test sets
datasetPath = "Datasets/MNIST/train_data.npy"
targetPath = "Datasets/MNIST/train_target.npy"

data = np.load(datasetPath, allow_pickle=True)
target = np.load(targetPath, allow_pickle=True)

modelPath = "Models/MNIST/"

# Info
print("data shape:", data.shape)
print("target shape:", target.shape)
print(data.shape[0], "train samples")
print(target.shape[0], "test samples")

# Model parameters
num_classes = 10
input_shape = (28, 28, 1)

# This is the Keras example architecture for MNIST. It produces strange loss gradients, which are quite often zero.
# 
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

# Compile MNIST Model
# This is some major-league weirdness. The CategoricalAccuracy metric introduces some stupid C long integer overflow error and I cannot be bothered with it. Screw the metric, I guess :)
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    # metrics=[
    #     keras.metrics.CategoricalAccuracy(name="acc", dtype = np.float64),
    # ]
)

# Training and Evaluation given Training and Testing Datasets. A small part of the data is set aside for validation purposes.
batch_size = 128
epochs = 20

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath=modelPath+"epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
]

model.fit(
    data,
    target,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=callbacks,
)

model.save(modelPath+"base_model.keras")

# score = model.evaluate(x_test, y_test, verbose=0)
