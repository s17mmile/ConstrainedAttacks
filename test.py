import numpy as np

test = np.array([[0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,1,1,1,1],[1,0,1,1,1,0,0,1,0,1]])

data, labels = np.hsplit(test, [2])

print(data)
print(labels)

labels = labels[0]

event_count = labels.shape[0]
target = np.zeros((event_count, 2))
target[np.arange(event_count), labels] = 1

print(target)