import os
import re
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('ggplot')
np.random.seed(42)  # for reproducibility

from skimage import color, transform, restoration, io, feature

# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv2D, MaxPooling2D
# from tensorflow.keras.datasets import cifar10
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.optimizers import RMSprop
# from tensorflow.keras.preprocessing.image import ImageDataGenerator




"""
BUILDING THE IMAGE CLEANING PIPELINE
"""

img_folder = '../data/subset'

# read in an image of interest
path_string_list = os.listdir(img_folder)

# split path_string ('acura_TL_1984') on '_' (or '-')("_|-")
# path_list == list of lists (llist)
# eg. [[acura, TL, 1984],[acura, TL, 1985], [acura, GE, 1999]]
path_llist = [re.split("_", path_string) for path_string in path_string_list].sort()




