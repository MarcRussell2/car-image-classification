import os
import re
import sys
import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import main



raw_image_path = '../data/raw/binary-subset'
ip = main.ImagePipeline(raw_image_path)
ip.read()
for i in [128,64,32,16]:
    ip.resize(shape = (i,i,3))
    ip.grayscale()
    savename = 'gray_' + str(i)
    ip.save(savename)