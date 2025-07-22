import numpy as np
import os

x_old = np.load("Adversaries/CIFAR10/FGSM_train_data_old.npy", allow_pickle=True)
y_old = np.load("Adversaries/CIFAR10/FGSM_train_data_old.npy", allow_pickle=True)

x_new = np.load("Adversaries/CIFAR10/FGSM_train_data.npy", allow_pickle=True)
y_new = np.load("Adversaries/CIFAR10/FGSM_train_data.npy", allow_pickle=True)

print(np.unique(x_old == x_new, return_counts=True))
print(np.unique(y_old == y_new, return_counts=True))