import os
import glob
import csv
import numpy as np
import SimpleITK as sitk


import json
def load_config():
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config.json'))
    with open(config_path, 'r') as f:
        return json.load(f)
config = load_config()


def calculate_birads(fgt_ratio):
    if fgt_ratio < 0.25:
        return 'A'
    elif fgt_ratio < 0.50:
        return 'B'
    elif fgt_ratio < 0.75:
        return 'C'
    else:
        return 'D'

def main():
    base_data_dir = config["DATA_DIR"]:
    main()
