import os
import glob
import random
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

import csv
import shutil


import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()


base_data_dir = config["DATA_DIR"])
