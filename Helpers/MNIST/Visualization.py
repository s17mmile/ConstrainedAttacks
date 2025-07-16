import matplotlib.pyplot as plt
import numpy as np

def compare_MNIST784(originalImage, originalLabel, perturbedImage, perturbedLabel, index):
    print(originalLabel)
    print(perturbedLabel)
    
    f, ax = plt.subplots(2,2, figsize = (8,8))

    f.suptitle("Original vs. adversarial example with index " + str(index))

    vmin = min(np.min(originalImage), np.min(perturbedImage))
    vmax = max(np.max(originalImage), np.max(perturbedImage))

    print("Limits: " + str(vmin) + " - " + str(vmax))

    ax[0,0].set_title("Old Label: " + str(np.argmax(originalLabel)))
    ax[0,0].imshow(np.reshape(originalImage, [28,28]), vmin = vmin, vmax = vmax)

    ax[0,1].set_title("New Label: " + str(np.argmax(perturbedLabel)))
    ax[0,1].imshow(np.reshape(perturbedImage, [28,28]), vmin = vmin, vmax = vmax)

    edges = np.arange(-0.5, 10.5)

    ax[1,0].set_xlim([-0.5,9.5])
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_xticks(np.arange(0,10))
    ax[1,0].stairs(originalLabel, edges)

    ax[1,1].set_xlim([-0.5,9.5])
    ax[1,1].set_ylim([0,1])
    ax[1,1].set_xticks(np.arange(0,10))
    ax[1,1].stairs(perturbedLabel, edges)

    plt.show()

def show_MNIST784(image):
    plt.figure()
    plt.imshow(np.reshape(image, [28,28]))
    plt.show()