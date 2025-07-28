#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 11:32:30 2021

@author: jgalvan
"""

import numpy as np
import pandas as pd 
import math
from datetime import datetime
from collections import defaultdict
from collections.abc import Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import random
from pathlib import Path
import os
from copy import deepcopy
from tqdm.notebook import tqdm

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
import tensorflow as tf
import tensorflow.keras as keras
num_cores = 0
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.set_soft_device_placement(True)
os.environ["OMP_NUM_THREADS"] = str(num_cores)
os.environ["KMP_BLOCKTIME"] = "30"
os.environ["KMP_SETTINGS"] = "1"
os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

# custom 
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import other_utils, file_management