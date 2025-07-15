import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras

# Load the data and split it between train and test sets
x_train = np.load("Datasets/CIFAR-10/train_data.npy")
y_train = np.load("Datasets/CIFAR-10/train_target.npy")
x_test = np.load("Datasets/CIFAR-10/test_data.npy")
y_test = np.load("Datasets/CIFAR-10/test_target.npy")

# Model parameters
num_classes = 10
input_shape = (32, 32, 3)

# The exact architecture of this model is largely arbitrary.
# I mostly looked at the architecture of the Keras MNIST example and added extra conv2d/normalization layers as I've heard it's good practice...?
# Frankly, optimizing the structure and hyperparameters isn't really the focus of this work, so I'll run with it.
model = keras.Sequential(
    [
        keras.layers.Input(shape=input_shape),

        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding = "same"),

        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding = "same"),

        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling2D(pool_size=(2, 2), padding = "same"),

        keras.layers.Flatten(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation="softmax")
    ]
)

model.summary()

# Compile CIFAR Model using standard optimizers and metrics
model.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    metrics=[
        keras.metrics.CategoricalAccuracy(name="acc"),
    ],
)

# Training and Evaluation given Training and Testing Datasets.
# A small part of the training data is set aside for validation purposes.
# Callbacks are taken as they were when training MNIST, we will later only keep the best model as a base model. Just good to have epochs for now. 
batch_size = 64
epochs = 50

callbacks = [
    keras.callbacks.ModelCheckpoint(filepath="Models/CIFAR-10/epoch_{epoch}.keras"),
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
]

model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split = 0.15,
    callbacks=callbacks,
)

model.save("models/CIFAR-10/base_model.keras")

# Possible base model improvement by incorporating flips/shifts to augment data. ImageDataGenerator deprecated with new keras, though. Ah well.
# # Small non-adversarial augmentation step once the previous fit seems to have plateaued --> Flip/Shift training images.
# data_generator = keras.preprocessing.image.ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# train_generator = data_generator.flow(x_train, y_train, batch_size)
# steps_per_epoch = x_train.shape[0] // batch_size

# augment_callbacks = [
#     keras.callbacks.ModelCheckpoint(filepath="Models/CIFAR-10/augment_epoch_{epoch}.keras"),
#     keras.callbacks.EarlyStopping(monitor="val_loss", patience=3),
# ]

# model.fit(
#     train_generator,
#     steps_per_epoch=steps_per_epoch,
#     epochs=epochs,
#     validation_split = 0.15,
#     callbacks = augment_callbacks
# )

# score = model.evaluate(x_test, y_test, verbose=1)
