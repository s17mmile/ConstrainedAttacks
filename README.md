This directory serves as the main directory for Maximilian Miles' adversarial attack work.



The parent directory is split up as follows:

- The "Datasets" folder will contain original, unmodified datasets as well as the generated adversaries. This will be logically split by which model/use each dataset is intended for.
	- Since these datasets are large, they will not be uplaoded to the github repository.



- The "Models" folder will include the models themselves. These could be .keras or .hdf5 files.



- The "Attacks" Folder includes code to make each attack run in a general form. Ensure that syntax like "from attacks import RDSA" works as one would expect it to, importing all necessary functionality to generate the adversaries (but nothing more).
	- RDSA:
	- FGSM:
	- PGD:



- The "Training Scripts" folder includes python scripts used to initially train models using original datasets. This will be kept fully separate from the attacks for clarity and ease of use.



- The "Attack Scripts" Folder includes the python scripts to be executed when executing an adversarial attack on an already trained model. These scripts should include all case-specific information:
	- The model to be attacked (path to a .keras file, or pretrained model loaded from a link)
	- The original dataset (file path or link)
	- Attack Specifications - e.g. number of shuffled variables for RDSA
	- (If desired, specifications for re-training the model)
	- Output paths: adversaries, model diagnostics and (if applicable) retrained models will be saved in specified locations.	
		- Adversaries go into an appropriate folder within "Datasets".
		- Diagnostics will be saved in the "Results" folder.
		- Retrained models will be saved in "Models" like any other model.



- The "Helpers" Folder will include scripts used for preprocessing and model evaluation purposes. This includes reshaping, data extraction, plotting scripts, correlation analysis, etc.



- The "Results" Folder holds all model diagnostics: one subfolder for each model in "Models" to hold performance metrics in form of raw data (.NPY files) and/or relevant plots showing performance, runtime, correlations, histograms etc.



Notes:
Best practice: When performing Train-Test-Splits, use a constant random seed to avoid evaluating a model on its own training data. Here, that seed is 42.

Within Keras and TensorFlow, which provide the backend ML implementations for this project, data is regularly moved between numpy array and TensorFlow tensor formats.
This is done because some TensorFlow APIs require Tensors as input.
--> The implementation is done to prioritise ease of use: anyone using this AdversarialAttacks library can do so while exclusively manipulating numpy arrays.
--> The only time tensors are used is within the implementations of each attack method.