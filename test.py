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



allFeatureIndices = [transformIndex(i,exampleShape) for i in range(featureCount)]