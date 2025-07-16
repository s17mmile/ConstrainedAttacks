import numpy as np

datasetPath = "Datasets/MNIST/train_data.npy"

data = data = np.load(datasetPath, allow_pickle=True)

tuple = (4,7,0)

print(len(np.unique(data[:,5,3,0])))