import numpy as np

def transformIndex(index, shape):
    assert index >= 0
    assert index < np.prod(shape)

    transformedIndex = []

    remainder = index

    for dimension in shape:
        coordinate = remainder % dimension
        remainder = int(remainder/dimension)
        transformedIndex.append(coordinate)

    return transformedIndex

print(transformIndex(1,(3,28,28)))