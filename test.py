import os
import numpy as np
# import keras
# import tensorflow as tf

# -------------------------------------------------------------------------------------------------
# data = np.load("Datasets/CIFAR10/test_data.npy")

# model = keras.models.load_model("Models/CIFAR10/base_model.keras")

# print(model(tf.convert_to_tensor([data[0]]), training = False).numpy()[0])


# -------------------------------------------------------------------------------------------------
def return_sth(x):
    if x:
        return 1,2,3
    else:
        return 1
    
result0 = return_sth(0)
result1 = return_sth(1)

print(result0[0])
print(result1[1])