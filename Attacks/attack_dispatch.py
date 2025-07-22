import numpy as np
import os
import sys
import warnings

sys.path.append(os.getcwd())

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1" 

import tensorflow as tf
import keras

import Attacks.constrained_FGSM as cFGSM
import Attacks.constrained_PGD as cPGD
import Attacks.constrained_RDSA as cRDSA




def AttackDispatcher(**kwargs):

    '''
        This calls the specified attack given parameters.
        These parameters specify:
            - The attack type
            - Attack-specific params
            - The paths to dataset/model/output files
            - A constrainer function (optional)
            - Execution parameters n, workercount and chunksize (optional)
        All other arguments are attack-specific and will be passed to the attack function.
    '''

    # Error handling. In if statement just to be able to collapse it in the IDE.
    check_arg_validity = True
    if (check_arg_validity):
        # Check if required parameters are provided and valid
        assert "attack_type" in kwargs, "Attack type must be specified."
        attack_type = kwargs["attack_type"]
        assert attack_type in ["FGSM", "PGD", "RDSA"], "Invalid attack type specified. Valuid options are: \"FGSM\", \"PGD\", \"RDSA\"."

        assert "datasetPath" in kwargs, "Dataset path must be provided."
        datasetPath = kwargs["datasetPath"]
        assert datasetPath.endswith(".npy"), "Dataset path must point to a .npy file."
        assert os.path.isfile(datasetPath), "Dataset file does not exist."
        
        assert "targetPath" in kwargs, "Target path must be provided."
        targetPath = kwargs["targetPath"]
        assert targetPath.endswith(".npy"), "Target path must point to a .npy file."
        assert os.path.isfile(targetPath), "Target file does not exist."

        assert "modelPath" in kwargs, "Model path must be provided."
        modelPath = kwargs["modelPath"]
        assert modelPath.endswith(".keras"), "Model path must point to a .keras file."
        assert os.path.isfile(modelPath), "Model file does not exist."

        assert "adversaryPath" in kwargs, "Adversary output path must be provided."
        adversaryPath = kwargs["adversaryPath"]
        assert adversaryPath.endswith(".npy"), "Adversary output path must point to a .npy file."
        assert os.path.isdir(os.path.dirname(adversaryPath)), "Adversary output directory does not exist."

        assert "newLabelPath" in kwargs, "New label output path must be provided."
        newLabelPath = kwargs["newLabelPath"]
        assert newLabelPath.endswith(".npy"), "New label output path must point to a .npy file."
        assert os.path.isdir(os.path.dirname(newLabelPath)), "New label output directory does not exist."


        # Check that the dataset can actualy be loaded and the shapes are compatible.
        # Find shape of dataset and target. We use a memory-mapped array to avoid loading the entire dataset into memory here.
        # (Later, we will load the whole thing as memory maps and multiprocessing don't always play nice together.)
        # While we could load the whole thing now and avoid the deletion (minimal overhead), I prefere to separate any validity checks from ectual loading and execution for cleanliness.
        try:
            dataset = np.load(datasetPath, allow_pickle=True, mmap_mode="r")
        except Exception as e:
            raise ValueError(f"Failed to load dataset from {datasetPath}:\n\n{e}")
        
        try:
            target = np.load(targetPath, allow_pickle=True, mmap_mode="r")
        except Exception as e:
            raise ValueError(f"Failed to load target from {targetPath}:\n\n{e}")

        assert dataset.shape[0] == target.shape[0], "Dataset and target must have the same number of samples."
        num_samples = dataset.shape[0]
        input_shape = dataset.shape[1:]

        del dataset, target



        # Check if optional parameters are provided and valid. Set defaults if not provided.
        if "n" in kwargs:
            n = kwargs["n"]
            assert isinstance(n, int) and n > 0, "n must be a positive integer."
            assert n <= num_samples, "n must not exceed the number of samples in the dataset."
        else:
            warnings.warn("n not provided, defaulting to attacking the entire dataset.")
            n = num_samples

        if "workercount" in kwargs:
            workercount = kwargs["workercount"]
            assert isinstance(workercount, int) and workercount > 0, "Workercount must be a positive integer."
        else:
            warnings.warn("Workercount not provided, defaulting to 1 worker.")
            workercount = 1

        if "chunksize" in kwargs:
            chunksize = kwargs["chunksize"]
            assert isinstance(chunksize, int) and chunksize > 0, "Chunksize must be a positive integer."
        else:
            warnings.warn("Chunksize not provided, defaulting to 1.")
            chunksize = 1

        if "constrainer" in kwargs:
            constrainer = kwargs["constrainer"]
            assert callable(constrainer), "Constrainer must be a callable function.";
        else:
            warnings.warn("Constrainer not provided, defaulting to None.")
            constrainer = None

        if "force_overwrite" in kwargs:
            force_overwrite = kwargs["force_overwrite"]
            assert isinstance(force_overwrite, bool), "Force overwrite must be a boolean value."
        else:
            warnings.warn("Force overwrite not provided, defaulting to False.")
            force_overwrite = False



        # Specific assertions based on attack type
        if attack_type == "FGSM":
            assert "epsilon" in kwargs, "Epsilon value must be provided for FGSM attack."
            epsilon = kwargs["epsilon"]
            
            assert "lossObject" in kwargs, "Loss object must be provided for FGSM attack."
            lossObject = kwargs["lossObject"]

        elif attack_type == "PGD":
            assert "stepcount" in kwargs, "Step count must be provided for PGD attack."
            stepcount = kwargs["stepcount"]
            
            assert "stepsize" in kwargs, "Step size must be provided for PGD attack."
            stepsize = kwargs["stepsize"]
            
            assert "lossObject" in kwargs, "Loss object must be provided for PGD attack."
            lossObject = kwargs["lossObject"]

            if "feasibilityProjector" in kwargs:
                feasibilityProjector = kwargs["feasibilityProjector"]
                assert callable(feasibilityProjector), "Feasibility projector must be a callable function."
            else:
                warnings.warn("Feasibility projector not provided, defaulting to None.")
                feasibilityProjector = None

        elif attack_type == "RDSA":
            assert "attempts" in kwargs, "Maximum number of attempts must be provided for RDSA attack."
            attempts = kwargs["attempts"]
            assert isinstance(attempts, int) and attempts > 0, "Maximum number of attempts must be a positive integer."

            assert "categoricalFeatureMaximum" in kwargs, "Categorical feature maximum must be provided for RDSA attack."
            categoricalFeatureMaximum = kwargs["categoricalFeatureMaximum"]
            assert isinstance(categoricalFeatureMaximum, int) and categoricalFeatureMaximum > 0, "Categorical feature maximum must be a positive integer."

            assert "binCount" in kwargs, "Bin count must be provided for RDSA attack."
            binCount = kwargs["binCount"]
            assert isinstance(binCount, int) and binCount > 0, "Bin count must be a positive integer."
            
            assert "perturbedFeatureCount" in kwargs, "Perturbed feature count must be provided for RDSA attack."
            perturbedFeatureCount = kwargs["perturbedFeatureCount"]
            assert isinstance(perturbedFeatureCount, int) and perturbedFeatureCount > 0, "Perturbed feature count must be a positive integer."
            assert perturbedFeatureCount <= np.prod(input_shape), f"The number of features to perturb ({perturbedFeatureCount}) must not exceed the number of input variables ({np.prod(input_shape)})."



    # Load dataset, without mmap_mode this time.
    dataset = np.load(datasetPath, allow_pickle=True)
    target = np.load(targetPath, allow_pickle=True)

    # Load pre-trained Model. This should be the only part after the arg validity check where an exception can occur, as the model has not been verified to be loadable yet.
    try:
        model = keras.models.load_model(modelPath)
    except Exception as e:
        raise ValueError(f"Failed to load model from {modelPath}:\n\n{e}")
    
    model.summary()



    if attack_type == "FGSM":
        adversaries, newLabels = cFGSM.parallel_constrained_FGSM(
            model=model,
            dataset=dataset[:n],
            targets=target[:n],
            lossObject=lossObject,
            epsilon=epsilon,
            constrainer=constrainer,
            workercount=workercount,
            chunksize=chunksize
        )

    elif attack_type == "PGD":
        adversaries, newLabels = cPGD.parallel_constrained_PGD(
            model=model,
            dataset=dataset[:n],
            targets=target[:n],
            lossObject=lossObject,
            stepcount=stepcount,
            stepsize=stepsize,
            feasibilityProjector=feasibilityProjector,
            constrainer=constrainer,
            workercount=workercount,
            chunksize=chunksize
        )

    elif attack_type == "RDSA":
        adversaries, newLabels = cRDSA.parallel_constrained_RDSA(
            model=model,
            dataset=dataset,
            targets=target,
            steps=attempts,
            categoricalFeatureMaximum=categoricalFeatureMaximum,
            binCount=binCount,
            perturbedFeatureCount=perturbedFeatureCount,
            constrainer=constrainer,
            workercount=workercount,
            chunksize=chunksize,
            n=n
        )



    # Flexible saving mechanism that avoids overwriting existing files unless forced explicitly allowed.
    if not os.path.isfile(adversaryPath) or force_overwrite or input("Adversary file already exists. Overwrite? (y/n): ").lower() == "y":
        print("Saving adversaries...")
        np.save(adversaryPath, adversaries)
        print("Adversaries saved.")
    elif input("Save a copy of the adversaries? (y/n): ").lower() == "y":
        print("Saving adversaries as a copy...")

        copyIndex = 0
        while os.path.isfile(adversaryPath.replace(".npy", f"_copy{copyIndex}.npy")):
            copyIndex += 1
        
        np.save(adversaryPath.replace(".npy", f"_copy{copyIndex}.npy"), adversaries)
        print("Adversaries saved as a copy.")
    else:
        print("Adversaries not saved. Discarding adversaries.")
        del adversaries

    if not os.path.isfile(newLabelPath) or force_overwrite or input("New label file already exists. Overwrite? (y/n): ").lower() == "y":
        print("Saving new labels...")
        np.save(newLabelPath, newLabels)
        print("New labels saved.")
    elif input("Save a copy of the new labels? (y/n): ").lower() == "y":
        print("Saving new labels as a copy...")

        copyIndex = 0
        while os.path.isfile(newLabelPath.replace(".npy", f"_copy{copyIndex}.npy")):
            copyIndex += 1
        
        np.save(newLabelPath.replace(".npy", f"_copy{copyIndex}.npy"), newLabels)
        print("New labels saved as a copy.")
    else:
        print("New labels not saved. Discarding new labels.")
        del newLabels



    print("Attack complete.")

    return


    