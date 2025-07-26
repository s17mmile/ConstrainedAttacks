import os
import numpy as np
# import keras
# import tensorflow as tf



# Transform a single index (in the range [0, prod(shape)]) into the corresponding multidimensional index.
# Ok yeah this works but I just realized it's stupidly slow. I just need to generate a list of all the combinations anyway.
def transformIndex(index, shape):
    assert index >= 0
    assert index < np.prod(shape)

    transformedIndex = []

    remainder = index

    for dimension in shape:
        coordinate = remainder % dimension
        remainder = int(remainder/dimension)
        transformedIndex.append(coordinate)

    return tuple(transformedIndex)


exampleShape = (28,28,1)
featureCount = 784

allFeatureIndices = [transformIndex(i,exampleShape) for i in range(featureCount)]

print(allFeatureIndices)

numpyFeatureIndices = zip(*np.unravel_index(range(featureCount), exampleShape))

for idx in numpyFeatureIndices:
    print(idx)

print("paishdgaspiufbvsüouass")

for idx in numpyFeatureIndices:
    print(idx)