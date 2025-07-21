import matplotlib.pyplot as plt
import numpy as np

def compare_CIFAR10(originalImage, originalLabel, target, perturbedImage, perturbedLabel, index):
    
    f, ax = plt.subplots(2,2, figsize = (8,8))

    f.suptitle("Original vs. adversarial example with index " + str(index) + ". Target label: " + str(np.argmax(target)) + ".")

    vmin = min(np.min(originalImage), np.min(perturbedImage))
    vmax = max(np.max(originalImage), np.max(perturbedImage))

    print("Limits: " + str(vmin) + " - " + str(vmax))

    ax[0,0].set_title("Old Label: " + str(np.argmax(originalLabel)))
    ax[0,0].imshow(originalImage, vmin = vmin, vmax = vmax)

    ax[0,1].set_title("New Label: " + str(np.argmax(perturbedLabel)))
    ax[0,1].imshow(perturbedImage, vmin = vmin, vmax = vmax)

    edges = np.arange(-0.5, 10.5)

    ax[1,0].set_xlim([-0.5,9.5])
    ax[1,0].set_ylim([0,1])
    ax[1,0].set_xticks(np.arange(0,10))
    ax[1,0].stairs(originalLabel, edges)

    ax[1,1].set_xlim([-0.5,9.5])
    ax[1,1].set_ylim([0,1])
    ax[1,1].set_xticks(np.arange(0,10))
    ax[1,1].stairs(perturbedLabel, edges)

    # plt.savefig("Results/CIFAR/CIFAR_Example.png")

    plt.show()