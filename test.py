import numpy as np

t = np.load("Datasets/TopoDNN/train_target.npy")[:1024]
y = np.load("Adversaries/TopoDNN/RDSA_train_labels_10.npy")[:1024]

for a,b in zip(t,y):
    print(a,b)

print(np.unique(np.argmax(t,axis=1)==np.argmax(y, axis=1), return_counts=True))