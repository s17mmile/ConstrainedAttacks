# import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.figsize'] = (8, 8)
mpl.rcParams['axes.grid'] = False

# Rescale an arrary linearly from its original range into a given one.
def linearRescale(array, newMin, newMax):
    minimum, maximum = np.min(array), np.max(array)
    m = (newMax - newMin) / (maximum - minimum)
    b = newMin - m * minimum
    scaledArray = m * array + b
    # Remove rounding errors by clipping
    return np.clip(scaledArray, newMin, newMax)

# Helper function to extract labels from probability vector
def get_imagenet_label(probs):
    return keras.applications.imagenet_utils.decode_predictions(probs, top=1)[0][0]

def displayImage(image, description, model):
    print(image.shape)

    _, label, confidence = get_imagenet_label(model(np.array([image])))

    plt.figure()
    plt.imshow(image*0.5+0.5)
    plt.title('{} \n {} : {:.2f}% Confidence'.format(description, label, confidence*100))
    plt.show()


# Compare an ImageNet instance with a perturbed counterpart
def compare_ImageNet(originalImage, originalLabel, perturbedImage, perturbedLabel, targetLabel, index):

    # For display purposes, map values [-1,1] (as they are in the dataset) to [0,1] for each colour channel (for imshow to work with)
    originalImage = linearRescale(originalImage, 0, 1)
    perturbedImage = linearRescale(perturbedImage, 0, 1)
    
    f, ax = plt.subplots(2,2, figsize = (12,12))

    _, targetLabelName, _ = get_imagenet_label(np.array([targetLabel]))
    f.suptitle("Original vs. adversarial example with index " + str(index) + ". Target label: " + targetLabelName)

    vmin = min(np.min(originalImage), np.min(perturbedImage))
    vmax = max(np.max(originalImage), np.max(perturbedImage))

    print("Limits: " + str(vmin) + " - " + str(vmax))

    _, originalLabelName, originalConfidence = get_imagenet_label(np.array([originalLabel]))
    ax[0,0].set_title('{} : {:.2f}% Confidence'.format(originalLabelName, originalConfidence*100))
    ax[0,0].imshow(originalImage, vmin = vmin, vmax = vmax)

    _, perturbedLabelName, perturbedConfidence = get_imagenet_label(np.array([perturbedLabel]))
    ax[0,1].set_title('{} : {:.2f}% Confidence'.format(perturbedLabelName, perturbedConfidence*100))
    ax[0,1].imshow(perturbedImage, vmin = vmin, vmax = vmax)



    # At the bottom, plot the probabilities as a stair chart. Going to be very hard to see for larger category counts.
    categoryCount = originalLabel.shape[0]

    edges = np.arange(-0.5, categoryCount+0.5)

    ax[1,0].set_xlim([-0.5,categoryCount-0.5])
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_xticks(np.arange(0,categoryCount+1,categoryCount/10))
    ax[1,0].stairs(originalLabel, edges)

    ax[1,1].set_xlim([-0.5,categoryCount-0.5])
    ax[1,1].set_ylim([0,1])
    ax[1,1].set_xticks(np.arange(0,categoryCount+1,categoryCount/10))
    ax[1,1].stairs(perturbedLabel, edges)

    # plt.savefig("Results/ImageNet/ImageNet_PGD_Example.png")

    plt.show()