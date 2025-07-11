# This script checks for the existence of the relevant models in the local folder.
# If they are not there, a download will be attempted from each source.
# MNIST is exempt from this, as it is trained from scratch.

import numpy as np
import os
import shutil
from urllib.request import urlretrieve
import tarfile
from sklearn.datasets import fetch_openml

print(os.getcwd())