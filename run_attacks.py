import numpy as np
import os
import sys

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

from Attacks.attack_dispatch import AttackDispatcher
import Helpers.constrainers as constrainers




# -------------------------------------------------------------------------------------------------------------------------------------------------
# PGD stepsize specifier.

# Exponential decrease to zero in on target. Inspired by binary search.
def stepsize(step):
    return 0.05*(1/2**step)

# Exponential decrease to zero in on target. Inspired by binary search. Smaller overall because TopoDNN values are less spread out.
def stepsize_topodnn(step):
    return 0.02*(1/2**step)


# Selector Panel: Choose which attacks to perform
CIFAR_FGSM = False
CIFAR_PGD = False
CIFAR_RDSA = False

ImageNet_FGSM = False
ImageNet_PGD = False
ImageNet_RDSA = False

MNIST_FGSM = False
MNIST_PGD = False
MNIST_RDSA = False

# We add some more specifiers for TopoDNN specifically, as we want to try more different constraints.
# Note that we have two ways of performing the energy constraint.
# In total, this gives us four adversarial data variations that we want to test:
    # - clipping only
    # - clipped and removed new constituents
    # -  clipped, removed new constituents and energy-shifted all pTs the same amount
    # -  clipped, removed new constituents and energy-shifted all pTs individually

# The "Base Constraint" is just clipping the variables' values back into original ranges.
# As of writing this, we already have adversarial topoDNN data that was constrained this way (no feasibilityProjector for PGD, only clip at end).
# Thus, we do not need to re-run RDSA or FGSM at all! We can simply take the clipped data and apply the constituent conservation and both energy conservation strategies after the fact, saving loads of computation time and disk space.
TopoDNN_FGSM_clip = False
TopoDNN_PGD_clip = False
TopoDNN_RDSA_clip = False

# PGD uses the constrainers as a repeated projection function. Thus, we unfortunately need to re-attack from scratch multiple times, since we cannot just tack the constraint onto the end result.
TopoDNN_PGD_constits_clip = False
TopoDNN_PGD_constits_clip_globalEnergy = False
TopoDNN_PGD_constits_clip_particleEnergy = False




# This extra __main__ specifier is necessary to use multiprocessing without getting a recursion error.
# Unfold each code region to view the case-specific attack configuration.
if __name__ == "__main__":

    if (CIFAR_FGSM):
        try:
            print("\n\n\nCIFAR FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/CIFAR10/train_data.npy",
                targetPath="Datasets/CIFAR10/train_target.npy",
                modelPath="Models/CIFAR10/base_model.keras",
                adversaryPath="Adversaries/CIFAR10/test/FGSM_train_data.npy",
                originalLabelPath="Adversaries/CIFAR10/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/CIFAR10/test/FGSM_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                epsilon=0.05,
                constrainer=constrainers.constrainer_scale_0_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (CIFAR_PGD):
        try:
            print("\n\n\nCIFAR PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/CIFAR10/train_data.npy",
                targetPath="Datasets/CIFAR10/train_target.npy",
                modelPath="Models/CIFAR10/base_model.keras",
                adversaryPath="Adversaries/CIFAR10/test/PGD_train_data.npy",
                originalLabelPath="Adversaries/CIFAR10/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/CIFAR10/test/PGD_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                constrainer=constrainers.constrainer_scale_0_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")
    
    if (CIFAR_RDSA):
        try:   
            print("\n\n\nCIFAR RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/CIFAR10/train_data.npy",
                targetPath="Datasets/CIFAR10/train_target.npy",
                modelPath="Models/CIFAR10/base_model.keras",
                adversaryPath="Adversaries/CIFAR10/test/RDSA_train_data.npy",
                originalLabelPath="Adversaries/CIFAR10/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/CIFAR10/test/RDSA_train_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=100,
                binCount=100,
                perturbedFeatureCount=300,
                constrainer=constrainers.constrainer_scale_0_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (ImageNet_FGSM):
        try:
            print("\n\n\nImageNet FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/ImageNet/threshold_data.npy",
                targetPath="Datasets/ImageNet/threshold_target.npy",
                modelPath="Models/ImageNet/base_model.keras",
                adversaryPath="Adversaries/ImageNet/test/FGSM_threshold_data.npy",
                originalLabelPath="Adversaries/ImageNet/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/ImageNet/test/FGSM_threshold_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                epsilon=0.05,
                constrainer=constrainers.constrainer_scale_m1_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (ImageNet_PGD):
        try:
            print("\n\n\nImageNet PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/ImageNet/threshold_data.npy",
                targetPath="Datasets/ImageNet/threshold_target.npy",
                modelPath="Models/ImageNet/base_model.keras",
                adversaryPath="Adversaries/ImageNet/test/PGD_threshold_data.npy",
                originalLabelPath="Adversaries/ImageNet/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/ImageNet/test/PGD_threshold_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                constrainer=constrainers.constrainer_scale_m1_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")
        
    if (ImageNet_RDSA):
        try:
            print("\n\n\nImageNet RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/ImageNet/threshold_data.npy",
                targetPath="Datasets/ImageNet/threshold_target.npy",
                modelPath="Models/ImageNet/base_model.keras",
                adversaryPath="Adversaries/ImageNet/test/RDSA_threshold_data.npy",
                originalLabelPath="Adversaries/ImageNet/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/ImageNet/test/RDSA_threshold_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=150,
                binCount=200,
                perturbedFeatureCount=10000,
                constrainer=constrainers.constrainer_scale_m1_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (MNIST_FGSM):
        try:
            print("\n\n\nMNIST FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/MNIST/train_data.npy",
                targetPath="Datasets/MNIST/train_target.npy",
                modelPath="Models/MNIST/base_model.keras",
                adversaryPath="Adversaries/MNIST/test/FGSM_train_data.npy",
                originalLabelPath="Adversaries/MNIST/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/MNIST/test/FGSM_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                epsilon=0.05,
                constrainer=constrainers.constrainer_scale_0_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_PGD):
        try:
            print("\n\n\nMNIST PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/MNIST/train_data.npy",
                targetPath="Datasets/MNIST/train_target.npy",
                modelPath="Models/MNIST/base_model.keras",
                adversaryPath="Adversaries/MNIST/test/PGD_train_data.npy",
                originalLabelPath="Adversaries/MNIST/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/MNIST/test/PGD_train_labels.npy",
                lossObject=keras.losses.CategoricalCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                constrainer=constrainers.constrainer_scale_0_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (MNIST_RDSA):
        try:
            print("\n\n\nMNIST RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/MNIST/train_data.npy",
                targetPath="Datasets/MNIST/train_target.npy",
                modelPath="Models/MNIST/base_model.keras",
                adversaryPath="Adversaries/MNIST/test/RDSA_train_data.npy",
                originalLabelPath="Adversaries/MNIST/test/Original_train_labels.npy",
                adversarialLabelPath="Adversaries/MNIST/test/RDSA_train_labels.npy",
                attempts=25,
                categoricalFeatureMaximum=100,
                binCount=100,
                perturbedFeatureCount=200,
                constrainer=constrainers.constrainer_scale_0_1,
                return_labels=True,
                n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (TopoDNN_FGSM_clip):
        try:
            print("\n\n\nTopoDNN FGSM\n")
            AttackDispatcher(
                attack_type="FGSM",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/spreadLimit/FGSM_train_data.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                epsilon=0.02,
                constrainer=constrainers.constrainer_TopoDNN_spreadLimit,
                return_labels=False,
                n=1024,
                force_overwrite=False,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD_clip):
        try:
            print("\n\n\nTopoDNN PGD\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/spreadLimit/PGD_train_data.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                stepcount=20,
                stepsize=stepsize,
                feasibilityProjector=constrainers.constrainer_TopoDNN_spreadLimit,
                return_labels=False,
                n=1024,
                force_overwrite=False,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_RDSA_clip):
        try:
            print("\n\n\nTopoDNN RDSA\n")
            AttackDispatcher(
                attack_type="RDSA",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/spreadLimit/RDSA_train_data.npy",
                attempts=25,
                categoricalFeatureMaximum=100000,
                binCount=1000,
                perturbedFeatureCount=15,
                constrainer=constrainers.constrainer_TopoDNN_spreadLimit,
                return_labels=True,
                #n=1,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")



    if (TopoDNN_PGD_constits_clip):
        try:
            print("\n\n\nTopoDNN PGD constits clip\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/conserveConstits_spreadLimit/PGD_train_data.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                stepcount=20,
                stepsize=stepsize_topodnn,
                feasibilityProjector=constrainers.constrainer_TopoDNN_conserveConstits_spreadLimit,
                return_labels=False,
                # n=1024,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")

    if (TopoDNN_PGD_constits_clip_globalEnergy):
        try:
            print("\n\n\nTopoDNN PGD constits clip globalEnergy\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveGlobalEnergy/PGD_train_data.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                stepcount=20,
                stepsize=stepsize_topodnn,
                feasibilityProjector=constrainers.constrainer_TopoDNN_conserveConstits_spreadLimit_conserveGlobalEnergy,
                return_labels=False,
                # n=1024,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")
            
    if (TopoDNN_PGD_constits_clip_particleEnergy):
        try:
            print("\n\n\nTopoDNN PGD constits clip particleEnergy\n")
            AttackDispatcher(
                attack_type="PGD",
                datasetPath="Datasets/TopoDNN/train_data.npy",
                targetPath="Datasets/TopoDNN/train_target.npy",
                modelPath="Models/TopoDNN/base_model.keras",
                adversaryPath="Adversaries/TopoDNN/conserveConstits_spreadLimit_conserveParticleEnergy/PGD_train_data.npy",
                lossObject=keras.losses.BinaryCrossentropy(),
                stepcount=20,
                stepsize=stepsize_topodnn,
                feasibilityProjector=constrainers.constrainer_TopoDNN_conserveConstits_spreadLimit_conserveParticleEnergy,
                return_labels=False,
                # n=1024,
                force_overwrite=True,
                workercount=8,
                chunksize=512
            )
        except Exception as e:
            print(f"Failure: {e}")