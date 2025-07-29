This repository contains all necessary scripts and keras models to recreate the results described in Maximilian Miles' Bachelor's Thesis.

This work deals with Adversarial Attacks on various classifiers. Each attack takes in a dataset - such as the training dataset a given classifier was trained on - and slightly modifies each entry in such a way that the model detects it as belonging to a different class than before. This results in an altered copy of the dataset which can be used to further train the previously fooled model. If all goes well, this will result in increased model performance and resistance to the previously applied adversarial strategies.

All of the attack strategies discussed have already been established in separate work. In this work, we wish to apply additional constraints to generated adversarial data to ensure its conformity to additional rules that we can set. These rules can help to increase the quality of generated data for the purpose of data augmentation and classifier re-training. Ultimately, the goal is to assess the effectiveness of these constraining procedures for different machine learning applications, with some extra focus on the task of top jet tagging in particle physics. 



This repository's purpose is twofold: 

1. It provides highly flexible and adaptable implementations of three different adversarial attack strategies, along with the option to add the aforementioned additional constraints. In that sense, I hope these implementations may be used as a small library for adversarial attacks. While the implementations are widely applicable, allow for parallel processing, and can be used without much modification (given correctly formatted input data and models, these are not so flexible), they are not fully optimized for compatibility, speed or memory efficiency. That is simply not the focus of this work and exceeds the frame set by a bachelor's thesis. 

In fact, I am almost certain there is a decent amount of computational and memory overhead within these implementations. If you wish to optimize them further or expand the existing attack strategies, go ahead. For example, staying entirely within the bounds of data types supplied by a keras backend (e.g. TensorFlow tensors) would circumvent format conversion overhead. Also, a proper multithreaded implementation that also makes use of memory mapping could save on system memory.

2. It serves as a start-to-finish introduction to adversarial attack methodology. Since all steps, including fetching datastes, performing attacks and evaluating results are supplied, it may be used as a starting point for anyone getting into the topic or hoping to improve their own classifier.



Three different attack strategies are explored and expanded upon:

1. Fast Gradient Sign Method (FGSM) - perhaps the simplest adversarial attack method. Given a model and a loss function, compute the loss gradient with respect to every single model input. To avoid particularly large or small gradients, use the gradient sign. Then, apply a constant shift of each input in the direction that locally maximizes the loss. This method, while rudimentary, is easy to implement and procides a simple proof of concept for attack modifications.

2. Projected Gradient Descent (PGD) serves as an improved, more powerful successor to FGSM. Instead of performing a single gradient calculation and step, it performs multiple such steps in succession. The step size no longer needs to be constant, but can dynamically change to allow the algorithm to zero in on an adversary. Lastly, not all possible feature combinations may be considered a "proper" model input, but we would like the calculated adversarial example to remain feasible (what exactly this means is case-specific). So, after each step, a transformation/projetcion is applied that ensures the example remains in the desired subset of feature space.

3. The [Random Distributed Shuffle Attack](https://arxiv.org/pdf/2501.05588) (RDSA) is a fundamentally different attack, as it is designed for use with particle accelerator data. Instead of applying precisely calculated perturbations to each feature of an input, only a small subset of the continuous input features is chosen to be modified. Then, instead of calculating a local loss gradient, the chosen features are randomized according to their distribution across the base dataset. This shuffling process repeats until a misclassification occurs or a set number of attempts runs out. While the attack is not actively targeting adversaries, it does retain single feature distributions due to the nature of the sampling process. As these distributions are fundamental characteristics of particle physics measurements, the generated adversarial data should better represent physical relationships, allowing for more effective classifier training.



Workflow: how do you use this repository?

1. Clone the repository into a folder of your choice.

2. Since the base datasets used for the given classifiers are quite large - on the order a Gigabyte each - they are not directly included within the repository. Git will not allow it without Git LFS or other workarounds, and it would bloat the repository size by orders of magnitude. To obtain the datasets, run the fetch_and_prepare_datasets.py script. By default, this will fetch and preprocess the following datasets (you can enable/disable them by a set of selector booleans in the script):

    - The [MNIST784 handwritten digit dataset](https://keras.io/api/datasets/mnist/), supplied by the Keras API. This includes 60000 samples for training and 10000 for testing. Each sample consists of 784 8-bit integers that form a 28x28 image of a handwritten digit. This will be preprocessed to instead contain floats in the range [0,1].

    - The [CIFAR-10 small images dataset](https://keras.io/api/datasets/cifar10/), supplied by the Keras API. This includes 50000 samples for training and 10000 for testing. Each sample is a 32x32 image with three color channels, which makes for a total of 3072 8-bit integers per sample. Each image portrays one of ten different categories of object, such as a bird, a deer, or a truck. This will be preprocessed so each color channel is represented by a float in the range [0,1].

    - Three separate [Testing Datasets for the ImageNet v2 network](https://www.tensorflow.org/datasets/catalog/imagenet_v2) (10000 samples each). This network classifies 224x224 images (three color channels, each a float within [-1,1]) into 1000 categories, such as different types of animals. These datasets are easily the largest used within this work and subsequently take more computing time than the others to attack.
    
    - The [TopoDNN Training, Testing, and Validation Datasets](https://github.com/FAIR4HEP/xAI4toptagger), discussed in more detail [here](https://scipost.org/SciPostPhys.7.1.014/pdf). These contain particle jet data from two million (combined) simulated collision events. The target label tells us whether or not each event is a "signal" jet - originating from a decaying top quark - or a "background" jet produced by gluon interactions. This is preprocessed using a modified version of the supplied topodnnpreprocessing.py script to extract the 30 jet constituents with the highest transverse momentum and align the jets along a common axis. This leaves us with 90 floating-point variables per sample, encoding the transverse momentum, pseudorapidity and azimuthal angle. Keep in mind, however, that the jet alignment in preprocessing forces three of these variables to take the same value across all samples. This leaves an effective 87 variables per sample.
    
3. At this point, the preprocessed Datasets will be present in the Datasets folder in .npy format. If you wish to perform attacks using a different dataset and model, simply supply the data to be perturbed as a .npy numpy binary file and the model as a .keras loadable classifier.

The Models folder contains several keras models, matching  

4. Perform an attack: to begin an adversarial attack, it is recommended to use the attack dispatcher provided in the "Attacks/" folder. It takes in a file path to the model to be attacked, an attack method specifier, input/output data paths and attack-specific parameters. The run_attack.py script has example configs that will perform all attack variants on all example models, so use these for reference.

Alternatively, you can directly import the constrained FGSM/PGD/RDSA implementations. This will bypass initial error handlers and type assertions, allowing you to attempt attacks using data and models in other formats. However, proper functionality is only guaranteed when inputting numpy arrays, .keras files and attack-specific floats/callables etc. as input.

5. Check out the adversarial data: the Helpers folder contains a few scripts to compare CIFAR/MNIST/ImageNet samples with their perturbed adversary counterpart. Use these to gain visual intuition about how each attack works and to verify attack integrity. These visualizations can also allow you to check for potential errors in preprocessing or attack schemes. Seriously, these came in handy for issue diagnosis several times along the way.

5. Re-train a model: The Evaluation folder contains the retrain_model.py script, which allows you to retrain an existing network, incrementally taking in parts of the generated adversary data. Since model performance improvements are all but guaranteed, it's worth it to evaluate retraining efficiency and resulting performance using different proportions of the generated data for augmentation. Who knows, maybe only using half of it will result in better performance?

6. Evaluate model performance: The Evaluation folder contains implementations of some performance evaluation metrics as well as - again - a dispatcher (examples in the evaluate_attacks.py script). These will allow you to determine key statistics about the attack's efficiency, the underlying dataset's charactristics and retrained model performance:

    - Accuracy, Recall, F1-Score: Basic Model performance evaluators.
    
    - Quantify the overall perturbation applied to the data using various distance metrics. Specifically, this conatins implementations of cosine similarity as well as the L1, L2 and L-infinity distance metrics.
    
    - Quantify the change in model predictions on the original and adversarial data: Jensen-Shannon Distance.
    
    - Single feature distributions: Generate single feature histograms to evaluate their changes.
    
    - Generate Confusion Matrices and Fooling Ratios. The confusion matrix will track classification success and failure across the original and perturbed dataset. Then, we can work out how often we were able to "fool" the network, meaning we perturbed an input such that the model makes a mistake it previously hadn't.
    
    - Corelation Matrices: Especially for the RDSA attack, we are interested in how the input features correlate after an attack is performed. One good way to visualize this is using a correlation matrix. If all goes perfectly, the correlations between different features will have been eradicated, leaving the canvas white but for the central stripe of guaranteed feature self-correlation.



Dependencies:
In creating this project, I used the following set of dependencies. All of these can be easily installed using pip. Other versions may work, but compatibility can be an issure if you're not careful.

Python:         3.11.9
Tensorflow:     2.16.1
Keras:          3.9.2
Tqdm:           4.67.1
Pytables:       3.10.2
(Numpy:         1.26.4)
(Scipy:         1.10.1)

In my experience, installing the correct python version with the given versions of keras, tensorflow, tqdm and pytables is enough to make the whole project work, as other dependencies are included with tensorflow/keras. Other important dependency versions are provided for completeness. Everything else should be given in the standard library.



















